"""Model loading, caching, and discovery for YOLO segmentation models."""

import os
from pathlib import Path
from typing import Any


def _ensure_local_ultralytics_config() -> None:
    """Keep Ultralytics settings inside the project to avoid profile permission errors."""
    config_dir = Path(__file__).resolve().parents[1] / "runs" / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


_ensure_local_ultralytics_config()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DISPLAY_NAMES = {
    "yolov8x_retrained": "YOLOv8x retrained",
    "yolov8x_balanced": "YOLOv8x balanced",
}

_DISPLAY_ORDER = {
    "yolov8x_retrained": 0,
    "yolov8x_balanced": 10,
}

_VISIBLE_MODEL_DIRS = {"yolov8x_retrained", "yolov8x_balanced"}



def _load_yolo_class() -> Any:
    """Import Ultralytics lazily and patch known YOLO11 loss-class gaps."""
    try:
        import torch.nn as nn
        import ultralytics.utils.loss as loss_module
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is required to load YOLO checkpoints. "
            "Install project requirements with: pip install -r requirements.txt"
        ) from exc

    # YOLO11x checkpoints reference BCEDiceLoss and MultiChannelDiceLoss which
    # only exist in ultralytics >= 8.4.x. Register stubs so checkpoints load on
    # 8.3.226.
    for cls_name in ("BCEDiceLoss", "MultiChannelDiceLoss"):
        if not hasattr(loss_module, cls_name):
            setattr(loss_module, cls_name, type(cls_name, (nn.Module,), {}))
    return YOLO


def list_available_models() -> dict[str, str]:
    """Scan bestModelSelect/ and return {display_name: path}.

    The Flask app default is configured in app.py; this function only scans
    available checkpoints.
    """
    models_dir = _PROJECT_ROOT / "bestModelSelect"
    models: dict[str, str] = {}
    if models_dir.is_dir():
        subdirs = sorted(
            (path for path in models_dir.iterdir() if path.is_dir()),
            key=lambda path: (_DISPLAY_ORDER.get(path.name, 999), path.name.lower()),
        )
        for subdir in subdirs:
            if not subdir.is_dir():
                continue
            if _VISIBLE_MODEL_DIRS and subdir.name not in _VISIBLE_MODEL_DIRS:
                continue
            pt_files = list(subdir.glob("*.pt"))
            if not pt_files:
                continue
            display = _DISPLAY_NAMES.get(subdir.name, subdir.name)
            rel_path = str(pt_files[0].relative_to(_PROJECT_ROOT))
            models[display] = rel_path

    return models


def _resolve_path(model_path: str) -> str:
    """Resolve a model path: if absolute, use as-is; otherwise resolve against project root."""
    if os.path.isabs(model_path):
        return model_path
    resolved = _PROJECT_ROOT / model_path
    if resolved.exists():
        return str(resolved)
    return model_path


def load_model(model_path: str = "best.pt") -> Any:
    """Load a YOLO model from the given checkpoint path.

    Args:
        model_path: Path to .pt weights file, or a key from list_available_models().

    Returns:
        A YOLO model instance ready for inference.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    resolved = _resolve_path(model_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Model checkpoint not found at: {resolved}")
    YOLO = _load_yolo_class()
    return YOLO(resolved)
