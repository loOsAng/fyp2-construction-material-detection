"""Inference pipeline: image decoding, model prediction, result rendering."""

from __future__ import annotations

import time
from typing import Any, Tuple

import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an RGB numpy array."""
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    bgr = cv2.imdecode(arr, 1)
    if bgr is None:
        raise ValueError("Unable to decode the uploaded image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_clean_result(result: Any) -> np.ndarray:
    """Render masks for presentation without debug labels covering the image."""
    plotted_bgr = result.plot(
        labels=False,
        conf=False,
        boxes=False,
        masks=True,
        line_width=2,
        color_mode="class",
    )
    return cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)


def preprocess_for_shadows(
    image_bgr: np.ndarray,
    gamma: float = 0.7,
    clahe_clip: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Enhance shadow regions for YOLO inference without shifting colours.

    The pipeline is:
      1. Gamma correction lifts the overall image floor (uniform underexposure).
      2. CLAHE on the L* channel of LAB space boosts local contrast in shadows
         without affecting hue or saturation.

    OpenCV's default CLAHE clipLimit of 40.0 is far too aggressive for YOLO
    (it amplifies noise and creates hallucinated edges). clipLimit=2.0–3.0
    with an 8×8 tile grid is the recommended conservative setting per the
    OpenCV CLAHE tutorial.

    Args:
        image_bgr: BGR image as a numpy array (uint8).
        gamma: Gamma correction exponent (< 1.0 lifts shadows).
        clahe_clip: CLAHE clip limit. Keep 2.0–3.0; never use the 40.0 default.
        tile_size: CLAHE tile grid size in pixels (8 or 16).

    Returns:
        Preprocessed BGR image of the same shape and dtype.
    """
    # -- Gamma correction (lifts uniform underexposure) --
    gamma = max(gamma, 0.01)
    table = np.array(
        [(i / 255.0) ** gamma * 255 for i in np.arange(256)],
        dtype=np.uint8,
    )
    img = cv2.LUT(image_bgr, table)

    # -- CLAHE on L* channel only (preserves colour) --
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def run_inference(
    model,
    image_rgb: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 1024,
    preprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Any, float]:
    """Standard single-pass YOLO inference.

    Args:
        preprocess: When True, apply CLAHE + gamma shadow enhancement
                    before YOLO inference (see preprocess_for_shadows).
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if preprocess:
        image_bgr = preprocess_for_shadows(image_bgr)
    t0 = time.perf_counter()
    results = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - t0
    result = results[0]
    plotted_rgb = render_clean_result(result)
    return image_rgb, plotted_rgb, result, elapsed


def run_tiled_inference(
    model,
    image_rgb: np.ndarray,
    conf: float = 0.15,
    iou: float = 0.45,
    imgsz: int = 640,
    preprocess: bool = False,
    tile_size: int = 640,
    overlap: float = 0.2,
    merge_nms_iou: float = 0.3,
    tile_augmented_classes: Tuple[str, ...] | None = ("nail",),
) -> Tuple[np.ndarray, np.ndarray, Any, float]:
    """Full-frame-safe tiled inference for small / occluded objects.

    Multi-tile mode keeps full-frame detections as the baseline so large
    Brick/I-beam instances are not lost when tiles cut through them. Tile
    detections are used as a small-object augmentation, limited to
    tile_augmented_classes by default.

    Args:
        tile_size: Tile side length in pixels (default 640).
        overlap: Overlap fraction between adjacent tiles (default 0.2).
        tile_augmented_classes: Class names allowed from tiles in multi-tile
            mode. The default adds nails while preserving full-frame Brick
            and I-beam detections.
        All other args: same as run_inference.

    Returns:
        (image_rgb, plotted_rgb, merged_result, total_elapsed)
    """
    import torch
    import torch.nn.functional as F
    import torchvision.ops

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if preprocess:
        image_bgr = preprocess_for_shadows(image_bgr)

    h, w = image_rgb.shape[:2]
    stride = int(tile_size * (1.0 - overlap))

    # Generate tile positions
    y_starts = list(range(0, max(1, h - tile_size + 1), stride))
    if not y_starts or y_starts[-1] < h - tile_size:
        y_starts.append(max(0, h - tile_size))
    x_starts = list(range(0, max(1, w - tile_size + 1), stride))
    if not x_starts or x_starts[-1] < w - tile_size:
        x_starts.append(max(0, w - tile_size))

    all_box_tensors: list = []
    all_mask_tensors_hw: list = []  # full-image-sized masks per detection
    template_result = None  # first tile result with valid boxes (used as merge template)
    full_result = None

    def class_keep_indices(result, boxes, allowed_classes: Tuple[str, ...] | None):
        if allowed_classes is None:
            return torch.arange(boxes.shape[0], device=boxes.device)
        allowed = {name.casefold() for name in allowed_classes}
        keep: list[int] = []
        for idx, cls_id in enumerate(boxes[:, 5].detach().cpu().numpy()):
            cls_idx = int(cls_id)
            if isinstance(result.names, dict):
                class_name = str(result.names.get(cls_idx, cls_idx))
            else:
                class_name = str(result.names[cls_idx])
            if class_name.casefold() in allowed:
                keep.append(idx)
        return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)

    def resize_masks(mask_stack, target_h: int, target_w: int):
        if mask_stack.shape[-2:] == (target_h, target_w):
            return mask_stack
        resized = F.interpolate(
            mask_stack[:, None].float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return resized.to(device=mask_stack.device, dtype=mask_stack.dtype)

    t0 = time.perf_counter()
    num_tiles = len(y_starts) * len(x_starts)

    if num_tiles > 1:
        full_result = model.predict(
            image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False,
        )[0]
        if full_result.boxes is not None and full_result.boxes.data.shape[0] > 0:
            template_result = full_result
            all_box_tensors.append(full_result.boxes.data.clone())
            if full_result.masks is not None:
                all_mask_tensors_hw.append(resize_masks(full_result.masks.data, h, w))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile_bgr = image_bgr[y0:y1, x0:x1]

            results = model.predict(
                tile_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False,
            )
            result = results[0]

            if result.boxes is None or result.boxes.data.shape[0] == 0:
                continue

            if template_result is None:
                template_result = result

            # --- Shift box coordinates from tile space → full image ---
            keep_indices = class_keep_indices(
                result,
                result.boxes.data,
                tile_augmented_classes if num_tiles > 1 else None,
            )
            if keep_indices.numel() == 0:
                continue

            boxes = result.boxes.data[keep_indices].clone()  # [N, 6]  xyxy-conf-cls
            boxes[:, 0] += x0  # x1
            boxes[:, 1] += y0  # y1
            boxes[:, 2] += x0  # x2
            boxes[:, 3] += y0  # y2
            all_box_tensors.append(boxes)

            # --- Resize masks to full-image space ---
            if result.masks is not None:
                tile_masks = resize_masks(result.masks.data[keep_indices], y1 - y0, x1 - x0)
                n_masks, mh, mw = tile_masks.shape
                full_masks = torch.zeros((n_masks, h, w),
                                         device=tile_masks.device,
                                         dtype=tile_masks.dtype)
                full_masks[:, y0:y1, x0:x1] = tile_masks[:, : y1 - y0, : x1 - x0]
                all_mask_tensors_hw.append(full_masks)

    elapsed = time.perf_counter() - t0

    # --- Fallback: no detections ---
    # Reuse the (possibly preprocessed) image_bgr from above instead
    # of re-converting from raw image_rgb, so shadow enhancement is
    # preserved in the fallback path.
    if not all_box_tensors:
        if full_result is not None:
            plotted_rgb = render_clean_result(full_result)
            return image_rgb, plotted_rgb, full_result, elapsed
        result = model.predict(
            image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False,
        )[0]
        elapsed = time.perf_counter() - t0
        plotted_rgb = render_clean_result(result)
        return image_rgb, plotted_rgb, result, elapsed

    # --- Merge: concatenate + aggressive NMS ---
    # Use a fixed low IoU (0.3) for cross-tile dedup — detections of the
    # same object in adjacent tiles often have moderate overlap (0.3–0.5),
    # so the per-model NMS threshold is too loose for the merge pass.
    # batched_nms applies NMS per class, preventing Brick boxes from
    # suppressing overlapping I-beam or Nail detections in cluttered scenes.
    combined_boxes = torch.cat(all_box_tensors, dim=0)
    boxes_xyxy = combined_boxes[:, :4]
    scores = combined_boxes[:, 4]
    cls_ids = combined_boxes[:, 5].long()

    if len(all_box_tensors) == 1:
        nms_indices = torch.arange(combined_boxes.shape[0], device=combined_boxes.device)
    else:
        nms_indices = torchvision.ops.batched_nms(
            boxes_xyxy, scores, cls_ids, iou_threshold=merge_nms_iou,
        )

    # --- Rebuild result with full-image context ---
    # Use the first tile result that had valid detections as a template
    # (has class names, model metadata etc.). The last tile may have had
    # no detections (boxes=None), so we cannot blindly use it.
    merged = template_result
    merged.orig_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    merged.orig_shape = image_rgb.shape
    merged.boxes.data = combined_boxes[nms_indices]
    if all_mask_tensors_hw:
        combined_masks = torch.cat(all_mask_tensors_hw, dim=0)
        merged.masks.data = combined_masks[nms_indices]
    else:
        merged.masks = None

    plotted_rgb = render_clean_result(merged)
    return image_rgb, plotted_rgb, merged, elapsed
