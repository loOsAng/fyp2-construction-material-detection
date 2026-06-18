"""Mask refinement using SAM (Segment Anything Model) via Ultralytics.

YOLO provides boxes + coarse masks. SAM uses those boxes as prompts to produce
higher-quality masks with better boundary adherence. The refined masks replace
the originals in the Results object so skeleton extraction benefits automatically.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model cache (module-level singleton)
# ---------------------------------------------------------------------------
_sam_model: Optional[object] = None
_sam_model_name: Optional[str] = None


def load_sam(model_name: str = "sam2_l.pt", device: Optional[str] = None) -> object:
    """Load (or retrieve cached) SAM model via Ultralytics.

    Args:
        model_name: One of 'mobile_sam.pt', 'sam_b.pt', 'sam_l.pt',
                    'sam2_b.pt', 'sam2_s.pt', 'sam2_l.pt'.
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        Ultralytics SAM model instance.
    """
    global _sam_model, _sam_model_name
    if _sam_model is not None and _sam_model_name == model_name:
        return _sam_model

    from ultralytics import SAM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading SAM model '%s' on %s ...", model_name, device)
    _sam_model = SAM(model_name)
    if device != "cpu":
        _sam_model.to(device)
    _sam_model_name = model_name
    return _sam_model


# ---------------------------------------------------------------------------
# Mask refinement
# ---------------------------------------------------------------------------

def refine_masks(
    image_rgb: np.ndarray,
    yolo_result,
    sam_model: Optional[object] = None,
    conf_threshold: float = 0.25,
) -> Tuple[np.ndarray, object]:
    """Replace YOLO instance masks with SAM-refined masks at full image resolution.

    Uses YOLO bounding boxes as SAM box prompts + YOLO mask centroids as
    foreground point prompts. Stores SAM masks at the original image
    resolution so skeleton functions can use them directly without upscaling.

    Args:
        image_rgb:  Original RGB image (H, W, 3).
        yolo_result: Ultralytics Results object from YOLO inference.
        sam_model:   Pre-loaded SAM model (auto-loads sam2_l.pt if None).
        conf_threshold: Skip boxes with confidence below this threshold.

    Returns:
        (image_rgb, yolo_result) — yolo_result.masks.data now holds SAM masks
        at full image resolution.
    """
    if yolo_result.boxes is None:
        return image_rgb, yolo_result

    if sam_model is None:
        sam_model = load_sam("mobile_sam.pt")

    # --- Collect valid YOLO boxes -------------------------------------------------
    boxes_xyxy = yolo_result.boxes.xyxy.cpu().numpy()
    confs = yolo_result.boxes.conf.cpu().numpy()
    h, w = image_rgb.shape[:2]

    valid_boxes_xyxy: list[list[float]] = []
    valid_indices: list[int] = []
    for i, (box, conf) in enumerate(zip(boxes_xyxy, confs)):
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        valid_boxes_xyxy.append([
            max(0, float(x1)), max(0, float(y1)),
            min(float(w), float(x2)), min(float(h), float(y2)),
        ])
        valid_indices.append(i)

    if not valid_boxes_xyxy:
        return image_rgb, yolo_result

    # --- Compute foreground point prompts from YOLO mask centroids ----------------
    points: list[list[float]] = []
    labels: list[int] = []
    yolo_masks = yolo_result.masks
    for yolo_idx in valid_indices:
        if yolo_masks is not None and yolo_idx < yolo_masks.data.shape[0]:
            m = yolo_masks.data[yolo_idx].cpu().numpy().astype(np.float32)
            # Resize YOLO mask to image resolution to find accurate centroid
            m_full = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            m_binary = (m_full > 0.5).astype(np.uint8)
            moments = cv2.moments(m_binary)
            if moments["m00"] > 0:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                # Fallback: box center
                bx = valid_boxes_xyxy[len(points)]
                cx = (bx[0] + bx[2]) * 0.5
                cy = (bx[1] + bx[3]) * 0.5
        else:
            bx = valid_boxes_xyxy[len(points)]
            cx = (bx[0] + bx[2]) * 0.5
            cy = (bx[1] + bx[3]) * 0.5
        points.append([cx, cy])
        labels.append(1)  # foreground

    # --- Run SAM with box + point prompts -----------------------------------------
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    sam_results = sam_model.predict(
        source=image_bgr,
        bboxes=valid_boxes_xyxy,
        points=points,
        labels=labels,
        verbose=False,
    )
    sam_result = sam_results[0]

    if sam_result.masks is None:
        return image_rgb, yolo_result

    sam_masks_data = sam_result.masks.data  # [K, H_sam, W_sam], full-res

    # --- Build masks tensor at FULL image resolution ------------------------------
    n_total = len(boxes_xyxy)
    device = yolo_result.boxes.xyxy.device
    # Store at full image resolution so skeleton functions get maximum detail
    new_masks = torch.zeros(n_total, h, w, device=device, dtype=torch.float32)

    # First, copy YOLO masks (upsampled) for all instances as fallback
    if yolo_result.masks is not None:
        old_data = yolo_result.masks.data
        for i in range(min(n_total, old_data.shape[0])):
            m_np = old_data[i].cpu().numpy().astype(np.float32)
            m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_LINEAR)
            new_masks[i] = torch.from_numpy(m_np).to(device)

    # Then overwrite with SAM masks for the instances we processed
    for sam_idx, yolo_idx in enumerate(valid_indices):
        if sam_idx >= len(sam_masks_data):
            break
        sam_mask = sam_masks_data[sam_idx]  # [H_sam, W_sam]
        sam_np = sam_mask.cpu().numpy().astype(np.float32)
        if sam_np.shape[:2] != (h, w):
            sam_np = cv2.resize(sam_np, (w, h), interpolation=cv2.INTER_LINEAR)
        new_masks[yolo_idx] = torch.from_numpy(sam_np).to(device)

    # Replace in-place
    yolo_result.masks.data = new_masks

    return image_rgb, yolo_result


def _resize_tensor(t: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Resize a 2D tensor to target (H, W)."""
    h, w = target_shape
    np_arr = t.cpu().numpy().astype(np.float32)
    np_arr = cv2.resize(np_arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np_arr).to(device=t.device, dtype=t.dtype)
