"""Flask web application for the FYP2 VIVA construction material evidence console.

Run:
    python app.py

Open:
    http://127.0.0.1:5000
"""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(APP_ROOT / "runs" / ".ultralytics"))

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

from modules import (
    apply_mask_postprocessing,
    decode_image,
    extract_clean_geometry_overlay,
    get_confidence_stats,
    list_available_models,
    load_model,
    render_clean_result,
    run_inference,
    run_tiled_inference,
)


UPLOAD_CACHE_DIR = APP_ROOT / "runs" / "app_upload_cache"
LAST_UPLOAD_IMAGE = UPLOAD_CACHE_DIR / "last_upload.bin"
LAST_UPLOAD_LABEL = UPLOAD_CACHE_DIR / "last_upload.txt"

EVAL_METRICS_PATH = APP_ROOT / "runs" / "evaluate" / "yolov8x_retrained_1024_test" / "metrics.json"


def load_evaluation_metrics() -> Optional[Dict[str, Any]]:
    """Load pre-computed formal evaluation metrics from metrics.json."""
    if not EVAL_METRICS_PATH.exists():
        return None
    try:
        with EVAL_METRICS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


_model_cache: Dict[str, Any] = {}
_inference_cache: Dict[tuple[str, str, float, float, int, bool], tuple[np.ndarray, np.ndarray, Any, float]] = {}
_warmed_model_keys: set[str] = set()
_last_successful_primary: Optional[Dict[str, Any]] = None
_last_successful_results: Optional[List[Dict[str, Any]]] = None
_DEFAULT_MODEL_KEY = "YOLOv8x retrained"
_MAX_INFERENCE_CACHE_ITEMS = 4

# Ring buffer for latency statistics (last 20 inferences)
_LATENCY_HISTORY: List[float] = []
_MAX_LATENCY_HISTORY = 20


def effective_default_model_key(available: Dict[str, str] | None = None) -> str:
    """Prefer the selected final checkpoint, but fall back if it is absent."""
    available = available if available is not None else list_available_models()
    if _DEFAULT_MODEL_KEY in available:
        return _DEFAULT_MODEL_KEY
    return list(available.keys())[0] if available else _DEFAULT_MODEL_KEY


def _percentile(values: List[float], pct: float) -> float | None:
    """Return the pct-th percentile of a list of floats (linear interpolation)."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def get_model(model_key: str = _DEFAULT_MODEL_KEY):
    """Load and cache a YOLO model by display name. Falls back to default if key not found."""
    key = model_key or _DEFAULT_MODEL_KEY
    if key not in _model_cache:
        available = list_available_models()
        if key not in available:
            key = effective_default_model_key(available)
        path = available.get(key, "best.pt")
        _model_cache[key] = load_model(str(path))
    return _model_cache[key]


def warm_up_model(model_key: str) -> tuple[bool, str]:
    """Load a model and run a tiny dummy predict once for a smoother first demo inference."""
    key = model_key or _DEFAULT_MODEL_KEY
    if key in _warmed_model_keys:
        return True, ""

    try:
        model = get_model(key)
        predict = getattr(model, "predict", None)
        if callable(predict):
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            predict(dummy, imgsz=64, verbose=False)
        _warmed_model_keys.add(key)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def inference_cache_key(image_bytes: bytes, form_state: Dict[str, Any]) -> tuple[str, str, float, float, int, bool]:
    """Stable cache key for standard YOLO inference results."""
    digest = hashlib.sha1(image_bytes).hexdigest()
    return (
        digest,
        str(form_state.get("model", _DEFAULT_MODEL_KEY)),
        round(float(form_state["conf"]), 6),
        round(float(form_state["iou"]), 6),
        int(form_state["imgsz"]),
        bool(form_state.get("preprocess", False)),
    )


def get_or_run_inference(model, original: np.ndarray, image_bytes: bytes, form_state: Dict[str, Any]):
    """Reuse YOLO results when only the post-processing overlay changed."""
    key = inference_cache_key(image_bytes, form_state)
    cached = _inference_cache.get(key)
    if cached is not None:
        return clone_inference_value(cached)

    value = run_inference(
        model,
        original,
        form_state["conf"],
        form_state["iou"],
        form_state["imgsz"],
        preprocess=bool(form_state.get("preprocess", False)),
    )
    if len(_inference_cache) >= _MAX_INFERENCE_CACHE_ITEMS:
        _inference_cache.pop(next(iter(_inference_cache)))
    _inference_cache[key] = value
    return clone_inference_value(value)


def clone_inference_value(value):
    """Return an isolated copy of cached inference output for destructive post-processing."""
    image_rgb, seg_result, result, elapsed = value
    image_copy = image_rgb.copy() if isinstance(image_rgb, np.ndarray) else image_rgb
    seg_copy = seg_result.copy() if isinstance(seg_result, np.ndarray) else seg_result
    return image_copy, seg_copy, copy.deepcopy(result), elapsed


def pil_from_rgb(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def image_to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    pil_from_rgb(img).save(buf, format="PNG")
    return buf.getvalue()


def data_uri(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return "data:{0};base64,{1}".format(mime, encoded)


def format_points(points: Any) -> str:
    if not points:
        return ""
    return "; ".join("({0}, {1})".format(int(x), int(y)) for x, y in points)


def class_breakdown_text(per_class: Dict[str, int]) -> str:
    if not per_class:
        return "No detected classes"
    ordered = sorted(per_class.items(), key=lambda item: item[0].casefold())
    return " | ".join("{0}: {1}".format(name, count) for name, count in ordered)


def class_count(per_class: Dict[str, int], target_name: str) -> int:
    target_key = target_name.casefold()
    for name, count in per_class.items():
        if name.casefold() == target_key:
            return int(count)
    return 0


def count_instances_from_result(result) -> tuple[int, Dict[str, int]]:
    """Compute total and per-class counts directly from YOLO boxes."""
    if result.boxes is None:
        return 0, {}

    cls_ids = result.boxes.cls.cpu().numpy()
    per_class: Dict[str, int] = {}
    for cls_id in cls_ids:
        cls_idx = int(cls_id)
        if isinstance(result.names, dict):
            name = str(result.names.get(cls_idx, cls_idx))
        else:
            name = str(result.names[cls_idx])
        per_class[name] = per_class.get(name, 0) + 1
    return len(cls_ids), per_class


def count_effective_from_result(result) -> tuple[int, Dict[str, int]]:
    """Count instances whose refined mask has ≥50 non-zero pixels.

    This complements count_instances_from_result (which counts every YOLO
    detection box). A detection whose mask is entirely removed by
    post-processing (morphology + largest-component filtering) is excluded
    from the effective count so the dashboard numbers match the visible
    segmentation overlay.
    """
    if result.masks is None or result.masks.data is None:
        return 0, {}

    mask_count = len(result.masks.data)
    if mask_count == 0:
        return 0, {}

    cls_ids = (
        result.boxes.cls.cpu().numpy()
        if result.boxes is not None
        else None
    )

    import torch

    effective_per_class: Dict[str, int] = {}
    effective_total = 0
    for idx in range(mask_count):
        mask = result.masks.data[idx]
        pixel_count = int(torch.count_nonzero(mask > 0).item())
        if pixel_count < 50:
            continue
        effective_total += 1
        if cls_ids is not None and idx < len(cls_ids):
            cls_idx = int(cls_ids[idx])
            if isinstance(result.names, dict):
                name = str(result.names.get(cls_idx, cls_idx))
            else:
                name = str(result.names[cls_idx])
        else:
            # Mask without corresponding box (edge case in some
            # ultralytics versions) — count as unknown.
            name = "unknown"
        effective_per_class[name] = effective_per_class.get(name, 0) + 1

    return effective_total, effective_per_class


def analysis_rows(polyline_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in polyline_analysis:
        method = item.get("geometry_method", "clean_overlay")
        has_skeleton = "longest_path" in method or "skeleton" in method
        bp_count = len(item["branch_points"]) if has_skeleton else "N/A"
        bp_text = format_points(item["branch_points"]) if has_skeleton else "N/A"
        row = {
            "Instance": item["instance_id"],
            "Detection Index": item["detection_index"],
            "Class": item["class"],
            "Confidence": item["confidence"],
            "Length (px)": item["length_px"],
            "Angle (deg)": item["angle_deg"],
            "Method": method,
            "Endpoints": format_points(item["endpoints"]),
            "Endpoint Count": len(item["endpoints"]),
            "Branch Points": bp_text,
            "Branch Point Count": bp_count,
            "Status": item["status"],
        }
        rows.append(row)
    return rows


def rows_to_csv_data_uri(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buf.getvalue().encode("utf-8-sig")
    return data_uri(csv_bytes, "text/csv")


def get_form_state() -> Dict[str, Any]:
    try:
        imgsz = int(request.form.get("imgsz", 1024))
    except ValueError:
        imgsz = 1024
    if imgsz not in (640, 1024):
        imgsz = 1024

    try:
        conf = float(request.form.get("conf", 0.25))
    except ValueError:
        conf = 0.25
    try:
        iou = float(request.form.get("iou", 0.45))
    except ValueError:
        iou = 0.45

    conf = min(max(conf, 0.0), 1.0)
    iou = min(max(iou, 0.0), 1.0)

    available = list_available_models()
    model_key = request.form.get("model", _DEFAULT_MODEL_KEY)
    if model_key not in available:
        model_key = effective_default_model_key(available)

    preprocess = request.form.get("preprocess", "0") == "1"
    tiled = request.form.get("tiled", "0") == "1"

    return {
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "model": model_key,
        "preprocess": preprocess,
        "tiled": tiled,
    }


def cached_image_label() -> str:
    if LAST_UPLOAD_IMAGE.exists() and LAST_UPLOAD_LABEL.exists():
        return LAST_UPLOAD_LABEL.read_text(encoding="utf-8").strip()
    return ""


def load_request_images() -> List[tuple[bytes, str]]:
    """Load uploaded images (single or batch). Returns list of (bytes, label)."""
    uploaded_files = request.files.getlist("image_file")
    images: List[tuple[bytes, str]] = []

    for uploaded in uploaded_files:
        if uploaded and uploaded.filename:
            image_bytes = uploaded.read()
            if image_bytes:
                try:
                    decode_image(image_bytes)
                except ValueError:
                    continue
                image_label = Path(uploaded.filename).name
                images.append((image_bytes, image_label))

    if images:
        # Cache the first image for the "latest upload" convenience
        UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        LAST_UPLOAD_IMAGE.write_bytes(images[0][0])
        LAST_UPLOAD_LABEL.write_text(images[0][1], encoding="utf-8")
        return images

    # Fallback: use cached image
    if LAST_UPLOAD_IMAGE.exists():
        return [(LAST_UPLOAD_IMAGE.read_bytes(), cached_image_label() or "Cached upload")]

    return []


def _class_name_from_result(result, cls_id: float) -> str:
    """Resolve a YOLO class id to its display name."""
    cls_idx = int(cls_id)
    if isinstance(result.names, dict):
        return str(result.names.get(cls_idx, cls_idx))
    return str(result.names[cls_idx])


def _filter_low_conf_by_class(result, user_conf: float, ibeam_floor: float = 0.08) -> None:
    """Remove detections below per-class confidence thresholds in-place.

    I-beam detections are kept down to ibeam_floor (to catch horizontal /
    occluded beams that YOLO scores low). Other classes are filtered at
    the user's conf threshold.  This runs AFTER YOLO inference+NMS so it
    cannot recover detections already suppressed by NMS, but it prevents
    low-confidence noise from appearing in the overlay for non-I-beam classes.
    """
    import torch

    if result.boxes is None or result.boxes.data.shape[0] == 0:
        return

    boxes_data = result.boxes.data
    cls_ids = boxes_data[:, 5].cpu().numpy()
    confs = boxes_data[:, 4].cpu().numpy()

    keep_mask = torch.ones(boxes_data.shape[0], dtype=torch.bool, device=boxes_data.device)
    for idx in range(boxes_data.shape[0]):
        cls_idx = int(cls_ids[idx])
        if isinstance(result.names, dict):
            name = str(result.names.get(cls_idx, cls_idx)).casefold()
        else:
            name = str(result.names[cls_idx]).casefold()
        threshold = ibeam_floor if name == "i-beam" else user_conf
        if float(confs[idx]) < threshold:
            keep_mask[idx] = False

    if not keep_mask.all():
        result.boxes.data = boxes_data[keep_mask]
        if result.masks is not None and result.masks.data is not None:
            result.masks.data = result.masks.data[keep_mask]


def analyze_image(image_bytes: bytes, image_label: str, form_state: Dict[str, Any]) -> Dict[str, Any]:
    original = decode_image(image_bytes)

    model = get_model(form_state.get("model", _DEFAULT_MODEL_KEY))

    if form_state.get("tiled"):
        image_rgb, seg_result, result, elapsed = run_tiled_inference(
            model,
            original,
            conf=form_state["conf"],
            iou=form_state["iou"],
            imgsz=form_state["imgsz"],
            preprocess=bool(form_state.get("preprocess", False)),
        )
    else:
        image_rgb, seg_result, result, elapsed = get_or_run_inference(
            model,
            original,
            image_bytes,
            form_state,
        )

    result = apply_mask_postprocessing(image_rgb, result)

    # --- Per-class confidence boost for I-beam recall ---
    # YOLO NMS suppresses low-confidence I-beam detections (especially
    # horizontal ones). To recover them, re-run inference at a lower conf
    # when the primary run found fewer I-beams than expected, then filter
    # non-I-beam noise back to the user's threshold.
    ibeam_count_raw = 0
    if result.boxes is not None and hasattr(result.boxes, 'cls'):
        try:
            for cls_id in result.boxes.cls.cpu().numpy():
                if _class_name_from_result(result, cls_id).casefold() == "i-beam":
                    ibeam_count_raw += 1
        except Exception:
            pass
    if ibeam_count_raw == 0 and form_state["conf"] > 0.10 and not form_state.get("tiled"):
        low_conf_state = dict(form_state, conf=0.08)
        _, _, result2, _ = get_or_run_inference(model, original, image_bytes, low_conf_state)
        if (result2.boxes is not None and hasattr(result2.boxes, 'data')
                and result2.boxes.data.shape[0] > 0):
            result2 = apply_mask_postprocessing(image_rgb, result2)
            _filter_low_conf_by_class(result2, form_state["conf"], ibeam_floor=0.08)
            if result2.boxes.data.shape[0] > 0:
                result = result2
                seg_result = render_clean_result(result)

    if hasattr(result, "plot"):
        seg_result = render_clean_result(result)

    count, per_class = count_instances_from_result(result)
    effective_count, effective_per_class = count_effective_from_result(result)

    skel_vis, polyline_analysis = extract_clean_geometry_overlay(image_rgb, result)

    conf_stats = get_confidence_stats(result)

    rows = analysis_rows(polyline_analysis)
    ibeam_geometry_count = sum(
        1 for row in rows
        if str(row.get("Class", "")).casefold() == "i-beam"
    )
    latency_ok = elapsed < 1.0 and form_state["imgsz"] == 640

    # --- Update latency history ring buffer ---
    _LATENCY_HISTORY.append(elapsed)
    if len(_LATENCY_HISTORY) > _MAX_LATENCY_HISTORY:
        _LATENCY_HISTORY.pop(0)

    latency_mean = statistics.fmean(_LATENCY_HISTORY) if _LATENCY_HISTORY else elapsed
    latency_p95 = _percentile(_LATENCY_HISTORY, 95) if len(_LATENCY_HISTORY) >= 5 else None

    # Use effective (post-processed) counts as the primary metric so the
    # dashboard numbers match the visible segmentation overlay.
    # Box-based raw counts are still available for reference.
    box_count, box_per_class = count, per_class
    primary_count = effective_count
    primary_per_class = effective_per_class

    return {
        "image_label": image_label,
        "original_uri": data_uri(image_to_png_bytes(original), "image/png"),
        "seg_uri": data_uri(image_to_png_bytes(seg_result), "image/png"),
        "skeleton_uri": data_uri(image_to_png_bytes(skel_vis), "image/png"),
        "csv_uri": rows_to_csv_data_uri(rows),
        "count": primary_count,
        "brick_count": class_count(primary_per_class, "Brick"),
        "ibeam_count": class_count(primary_per_class, "I-beam"),
        "nail_count": class_count(primary_per_class, "nail"),
        "box_count": box_count,
        "box_per_class": box_per_class,
        "effective_count": effective_count,
        "effective_per_class": effective_per_class,
        "geometry_count": len(rows),
        "ibeam_geometry_count": ibeam_geometry_count,
        "per_class_text": class_breakdown_text(primary_per_class),
        "per_class": primary_per_class,
        "elapsed": elapsed,
        "latency_ok": latency_ok,
        "inference_mode": form_state.get("model", _DEFAULT_MODEL_KEY),
        "conf_stats": conf_stats,
        "analysis_rows": rows,
        "latency_mean": round(latency_mean, 3),
        "latency_p95": None if latency_p95 is None else round(latency_p95, 3),
        "latency_samples": len(_LATENCY_HISTORY),
    }


def remember_successful_results(batch_results: List[Dict[str, Any]]) -> None:
    """Store the latest successful analysis for demo-safe error recovery."""
    global _last_successful_primary, _last_successful_results
    if not batch_results:
        return
    _last_successful_results = copy.deepcopy(batch_results)
    _last_successful_primary = copy.deepcopy(batch_results[0])


def restore_last_successful_result(context: Dict[str, Any]) -> bool:
    """Attach the latest successful result to context after a failed analysis."""
    if _last_successful_primary is None:
        return False
    context["result"] = copy.deepcopy(_last_successful_primary)
    context["batch_results"] = copy.deepcopy(_last_successful_results or [_last_successful_primary])
    context["using_last_successful_result"] = True
    return True


def initial_context(error: str = "", warm_model: bool = False) -> Dict[str, Any]:
    available = list_available_models()
    default_model = effective_default_model_key(available)
    model_ready = True
    if warm_model and available:
        model_ready, warm_error = warm_up_model(default_model)
        if warm_error and not error:
            error = f"Model warm-up failed: {warm_error}"
    return {
        "form": {
            "imgsz": 1024,
            "conf": 0.25,
            "iou": 0.45,
            "model": default_model,
            "preprocess": False,
            "tiled": False,
        },
        "available_models": list(available.keys()),
        "current_model": default_model,
        "result": None,
        "batch_results": None,
        "error": error,
        "model_ready": model_ready,
        "using_last_successful_result": False,
        "cached_image_label": cached_image_label(),
        "eval_metrics": load_evaluation_metrics(),
    }



@app.route("/", methods=["GET", "POST"])
def index():
    context = initial_context(warm_model=request.method == "GET")

    if request.method == "POST":
        form_state = get_form_state()
        context["form"] = form_state
        context["current_model"] = form_state.get("model", _DEFAULT_MODEL_KEY)
        try:
            images = load_request_images()
            if not images:
                raise ValueError("Please upload an image first. After that, you can adjust the settings and analyze again without re-uploading.")

            batch_results: List[Dict[str, Any]] = []
            for image_bytes, image_label in images:
                batch_results.append(analyze_image(image_bytes, image_label, form_state))

            context["result"] = batch_results[0]  # primary result (backward compat)
            context["batch_results"] = batch_results
            context["cached_image_label"] = cached_image_label()
            remember_successful_results(batch_results)
        except MemoryError:
            context["error"] = "GPU / system memory exhausted. Try reducing resolution to 640, or restart the app."
            restore_last_successful_result(context)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "cuda" in msg or "out of memory" in msg:
                context["error"] = "CUDA out of memory. Lower the resolution to 640 or close other GPU applications."
            elif "checkpoint" in msg or "model" in msg:
                context["error"] = f"Model loading failed: {exc}. Check that the selected checkpoint exists under bestModelSelect/."
            else:
                context["error"] = f"Inference error: {exc}"
            restore_last_successful_result(context)
        except ValueError as exc:
            context["error"] = str(exc)
            restore_last_successful_result(context)
        except Exception as exc:
            context["error"] = f"Unexpected error: {exc}"
            restore_last_successful_result(context)

    return render_template("index.html", **context)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("Starting FYP2 Flask web app...")
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
