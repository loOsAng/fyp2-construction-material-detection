"""Train YOLO segmentation models on the cleaned CCV2 construction dataset.

This script is intended for a rented GPU machine where the user can clone or
copy this project and start training without editing code.

Examples:
    # Default single-stage training on runs/datasets/ccv2_v5_resplit_seed42/data.yaml
    python train.py

    # Quick smoke run: one small epoch, small batch, small image size
    python train.py --quick

    # Use a YOLOv8 segmentation checkpoint
    python train.py --model yolov8x-seg.pt --batch 12

    # Use a YOLO11 segmentation checkpoint
    python train.py --model yolo11x-seg.pt --batch 8

    # Use a YOLO26 segmentation checkpoint, if supported by your Ultralytics install
    python train.py --model yolo26x-seg.pt --batch 4

    # Continue from a local segmentation checkpoint
    python train.py --checkpoint runs/train/ccv2_seg/weights/best.pt

    # Create a temporary balanced dataset under runs/datasets/balanced_*
    # The original Roboflow dataset is not modified.
    python train.py --balance --minority-classes 1 2 --minority-copies 3

    # Two-stage training is still available, but single-stage is the default.
    python train.py --mode two-stage --epochs-s1 30 --epochs-s2 120
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # Keep --help usable on machines before setup.
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent


def default_data_yaml_path() -> Path:
    """Return the first local CCV2 data.yaml found near the project."""
    candidates = (
        SCRIPT_DIR / "runs" / "datasets" / "ccv2_v5_resplit_seed42" / "data.yaml",
        SCRIPT_DIR / "dateset" / "CCV2.v5i.yolov8" / "data.yaml",
        SCRIPT_DIR.parent / "CCV2.v5i.yolov8" / "data.yaml",
        SCRIPT_DIR.parents[1] / "CCV2.v5i.yolov8" / "data.yaml",
        SCRIPT_DIR.parent / "CCV2.v2i.yolov81024" / "data.yaml",
        SCRIPT_DIR.parents[1] / "CCV2.v2i.yolov81024" / "data.yaml",
        SCRIPT_DIR / "CCV2.v2i.yolov81024" / "data.yaml",
    )
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return candidates[0].resolve()


DEFAULT_DATA_YAML = default_data_yaml_path()
DEFAULT_PROJECT = SCRIPT_DIR / "runs" / "train"
DEFAULT_DATASETS_DIR = SCRIPT_DIR / "runs" / "datasets"
DEFAULT_ULTRALYTICS_CONFIG = SCRIPT_DIR / "runs" / ".ultralytics"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("valid", "val"),
    "test": ("test",),
}


def require_yaml() -> Any:
    if yaml is None:
        raise SystemExit("PyYAML is required. Install project requirements before training.")
    return yaml


def load_yolo(checkpoint: str) -> Any:
    DEFAULT_ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_ULTRALYTICS_CONFIG))
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Ultralytics is required. Install project requirements before training.") from exc
    return YOLO(checkpoint)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return as_posix(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def read_data_yaml(path: Path) -> dict[str, Any]:
    parser = require_yaml()
    with path.open("r", encoding="utf-8") as fh:
        loaded = parser.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Invalid dataset YAML: {path}")
    return loaded


def write_data_yaml(path: Path, payload: dict[str, Any]) -> None:
    parser = require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        parser.safe_dump(jsonable(payload), fh, sort_keys=False, allow_unicode=True)


def resolve_data_yaml(data_arg: str) -> Path:
    raw = Path(data_arg).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, SCRIPT_DIR / raw])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"Dataset YAML not found. Checked: {checked}")


def infer_split_name(split_key: str, raw_path: str) -> str:
    parts = raw_path.replace("\\", "/").lower().split("/")
    for name in ("train", "valid", "val", "test"):
        if name in parts:
            return "valid" if name == "val" else name
    if split_key == "val":
        return "valid"
    return split_key


def data_root_from_yaml(data: dict[str, Any], yaml_path: Path) -> Path:
    raw_path = data.get("path")
    if raw_path:
        root = Path(str(raw_path)).expanduser()
        if not root.is_absolute():
            root = yaml_path.parent / root
        return root.resolve()
    return yaml_path.parent.resolve()


def resolve_dataset_entry(
    raw_value: str,
    split_key: str,
    yaml_path: Path,
    data_root: Path,
) -> Path:
    raw_path = Path(str(raw_value)).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((data_root / raw_path).resolve())
        candidates.append((yaml_path.parent / raw_path).resolve())

    inferred_split = infer_split_name(split_key, str(raw_value))
    candidates.append((yaml_path.parent / inferred_split / "images").resolve())

    for alias in SPLIT_ALIASES.get(split_key, (split_key,)):
        candidates.append((yaml_path.parent / alias / "images").resolve())

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    return candidates[0]


def resolve_dataset_paths(data: dict[str, Any], yaml_path: Path) -> dict[str, Any]:
    data_root = data_root_from_yaml(data, yaml_path)
    resolved: dict[str, Any] = {}

    for split_key in ("train", "val", "test"):
        if split_key not in data or data[split_key] in (None, ""):
            continue
        raw_value = data[split_key]
        if isinstance(raw_value, list):
            resolved[split_key] = [
                resolve_dataset_entry(str(item), split_key, yaml_path, data_root) for item in raw_value
            ]
        else:
            resolved[split_key] = resolve_dataset_entry(str(raw_value), split_key, yaml_path, data_root)

    return resolved


def validate_resolved_splits(resolved: dict[str, Any]) -> None:
    for split_key, path_value in resolved.items():
        paths = path_value if isinstance(path_value, list) else [path_value]
        for path in paths:
            if not Path(path).exists():
                raise SystemExit(f"Resolved {split_key} images directory does not exist: {path}")


def make_resolved_data_yaml(source_yaml: Path, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    data = read_data_yaml(source_yaml)
    resolved = resolve_dataset_paths(data, source_yaml)
    validate_resolved_splits(resolved)

    normalized = dict(data)
    normalized.pop("path", None)
    for split_key, path_value in resolved.items():
        if isinstance(path_value, list):
            normalized[split_key] = [as_posix(path) for path in path_value]
        else:
            normalized[split_key] = as_posix(path_value)

    out_yaml = out_dir / "data.yaml"
    write_data_yaml(out_yaml, normalized)
    return out_yaml, {
        "source_yaml": source_yaml,
        "resolved_yaml": out_yaml,
        "resolved_splits": resolved,
    }


def label_dir_for_images(images_dir: Path) -> Path:
    if images_dir.name.lower() == "images":
        return images_dir.parent / "labels"
    return images_dir.parent / "labels"


def read_label_classes(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    classes: list[str] = []
    with label_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                classes.append(parts[0])
    return classes


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def image_files(images_dir: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        files.extend(images_dir.glob(f"*{suffix}"))
        files.extend(images_dir.glob(f"*{suffix.upper()}"))
    return sorted(set(files))


def copy_split_to_balanced_dataset(
    src_images: Path,
    dst_split_dir: Path,
    target_classes: set[str],
    extra_copies: int,
    should_duplicate: bool,
) -> dict[str, Any]:
    src_labels = label_dir_for_images(src_images)
    dst_images = dst_split_dir / "images"
    dst_labels = dst_split_dir / "labels"
    class_counts: Counter[str] = Counter()
    copied_images = 0
    copied_labels = 0
    duplicated_images = 0
    duplicated_labels = 0
    transfer_modes: Counter[str] = Counter()

    for src_image in image_files(src_images):
        src_label = src_labels / f"{src_image.stem}.txt"
        classes = read_label_classes(src_label)
        class_counts.update(classes)

        transfer_modes[link_or_copy(src_image, dst_images / src_image.name)] += 1
        copied_images += 1
        if src_label.exists():
            transfer_modes[link_or_copy(src_label, dst_labels / src_label.name)] += 1
            copied_labels += 1

        if not should_duplicate or not classes or not (set(classes) & target_classes):
            continue

        for idx in range(extra_copies):
            suffix = f"_bal{idx + 1}"
            transfer_modes[link_or_copy(src_image, dst_images / f"{src_image.stem}{suffix}{src_image.suffix}")] += 1
            duplicated_images += 1
            if src_label.exists():
                transfer_modes[link_or_copy(src_label, dst_labels / f"{src_label.stem}{suffix}.txt")] += 1
                duplicated_labels += 1
                class_counts.update(classes)

    return {
        "source_images": src_images,
        "source_labels": src_labels,
        "output_split": dst_split_dir,
        "copied_images": copied_images,
        "copied_labels": copied_labels,
        "duplicated_images": duplicated_images,
        "duplicated_labels": duplicated_labels,
        "class_counts_after_copy": dict(sorted(class_counts.items())),
        "transfer_modes": dict(transfer_modes),
    }


def create_balanced_dataset(
    source_yaml: Path,
    datasets_dir: Path,
    target_classes: list[str],
    extra_copies: int,
    balance_splits: set[str],
) -> tuple[Path, dict[str, Any]]:
    if extra_copies < 1:
        raise SystemExit("--minority-copies must be >= 1 when --balance is used.")

    data = read_data_yaml(source_yaml)
    resolved = resolve_dataset_paths(data, source_yaml)
    validate_resolved_splits(resolved)
    out_root = datasets_dir / f"balanced_{now_tag()}"
    out_root.mkdir(parents=True, exist_ok=False)

    split_reports: dict[str, Any] = {}
    yaml_splits: dict[str, str] = {}
    for split_key in ("train", "val", "test"):
        if split_key not in resolved:
            continue
        src_value = resolved[split_key]
        if isinstance(src_value, list):
            raise SystemExit("--balance does not support dataset YAML entries that are lists.")
        dst_name = "valid" if split_key == "val" else split_key
        report = copy_split_to_balanced_dataset(
            src_images=src_value,
            dst_split_dir=out_root / dst_name,
            target_classes=set(target_classes),
            extra_copies=extra_copies,
            should_duplicate=split_key in balance_splits,
        )
        split_reports[split_key] = report
        yaml_splits[split_key] = f"{dst_name}/images"

    balanced_yaml = dict(data)
    balanced_yaml["path"] = as_posix(out_root)
    for split_key in ("train", "val", "test"):
        if split_key in yaml_splits:
            balanced_yaml[split_key] = yaml_splits[split_key]
        else:
            balanced_yaml.pop(split_key, None)

    out_yaml = out_root / "data.yaml"
    write_data_yaml(out_yaml, balanced_yaml)
    return out_yaml, {
        "source_yaml": source_yaml,
        "balanced_yaml": out_yaml,
        "balanced_root": out_root,
        "target_classes": target_classes,
        "extra_copies": extra_copies,
        "balance_splits": sorted(balance_splits),
        "splits": split_reports,
    }


def checkpoint_warning(checkpoint: str) -> None:
    ckpt = Path(checkpoint)
    name = ckpt.name.lower()
    looks_like_named_yolo = name.startswith(("yolov8", "yolo8", "yolo11", "yolo26"))
    if looks_like_named_yolo and "-seg" not in name:
        print(f"WARNING: {checkpoint} does not look like a segmentation checkpoint. Expected '*-seg.pt'.")


def train_kwargs(args: argparse.Namespace, data_yaml: Path, run_name: str, epochs: int, freeze: int, lr0: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "project": str(args.project),
        "name": run_name,
        "exist_ok": args.exist_ok,
        "optimizer": args.optimizer,
        "lr0": lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": True,
        "amp": args.amp,
        "freeze": freeze,
        "close_mosaic": args.close_mosaic,
        "val": True,
        "save": True,
        "save_period": args.save_period,
        "seed": args.seed,
        "single_cls": False,
        "cache": args.cache,
        "workers": args.workers,
        "plots": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "fliplr": 0.5,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "erasing": args.erasing,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.fraction is not None and args.fraction < 1.0:
        kwargs["fraction"] = args.fraction
    return kwargs


def save_dir_from_results(results: Any, project: Path, name: str, model: Any | None = None) -> Path:
    candidates = [results, getattr(results, "trainer", None)]
    if model is not None:
        candidates.extend([model, getattr(model, "trainer", None)])
    for candidate in candidates:
        save_dir = getattr(candidate, "save_dir", None) if candidate is not None else None
        if save_dir:
            return Path(save_dir).resolve()
    return (project / name).resolve()


def best_or_last_weight(save_dir: Path) -> Path:
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        return best
    last = save_dir / "weights" / "last.pt"
    if last.exists():
        return last
    raise SystemExit(f"No trained weights found under {save_dir / 'weights'}")


def result_metrics(results: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if isinstance(results, dict):
        metrics["results_dict"] = results
        return jsonable(metrics)
    raw = getattr(results, "results_dict", None)
    if isinstance(raw, dict):
        metrics["results_dict"] = raw
    for group_name in ("box", "seg", "mask"):
        group = getattr(results, group_name, None)
        if group is None:
            continue
        group_metrics: dict[str, Any] = {}
        for attr in ("map", "map50", "map75", "mp", "mr", "maps"):
            if hasattr(group, attr):
                group_metrics[attr] = getattr(group, attr)
        if group_metrics:
            metrics[group_name] = group_metrics
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        metrics["save_dir"] = str(save_dir)
    names = getattr(results, "names", None)
    if names is not None:
        metrics["names"] = names
    return jsonable(metrics)


def evaluate_model(
    model: Any,
    data_yaml: Path,
    args: argparse.Namespace,
    save_dir: Path,
    train_results: Any | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "train": result_metrics(train_results) if train_results is not None else {},
    }
    if args.no_eval:
        metrics["evaluation_skipped"] = True
        write_json(save_dir / "metrics.json", metrics)
        return metrics

    print(f"\nEvaluating final model on split='{args.eval_split}'...")
    eval_batch = args.batch if isinstance(args.batch, int) and args.batch > 0 else 1
    try:
        val_kwargs: dict[str, Any] = {
            "data": str(data_yaml),
            "split": args.eval_split,
            "imgsz": args.imgsz,
            "batch": eval_batch,
            "workers": args.workers,
            "augment": args.tta,
            "plots": True,
        }
        if args.device is not None:
            val_kwargs["device"] = args.device
        eval_results = model.val(**val_kwargs)
        metrics.update(
            {
                "evaluation_split": args.eval_split,
                "batch": eval_batch,
                "tta": args.tta,
                "evaluation": result_metrics(eval_results),
            }
        )
    except Exception as exc:
        metrics.update(
            {
                "evaluation_split": args.eval_split,
                "batch": eval_batch,
                "tta": args.tta,
                "evaluation_error": repr(exc),
            }
        )
        print(f"WARNING: final evaluation failed. Error recorded in metrics.json: {exc}")
    write_json(save_dir / "metrics.json", metrics)
    print(f"metrics.json saved to {save_dir / 'metrics.json'}")
    return metrics


def run_single_stage(args: argparse.Namespace, data_yaml: Path) -> tuple[Any, Any, list[dict[str, Any]]]:
    print("\n" + "=" * 72)
    print("Single-stage training")
    print("=" * 72)
    model = load_yolo(args.model)
    kwargs = train_kwargs(args, data_yaml, args.name, args.epochs, args.freeze, args.lr0)
    results = model.train(**kwargs)
    save_dir = save_dir_from_results(results, args.project, args.name, model)
    return model, results, [{"name": args.name, "mode": "single", "save_dir": save_dir, "kwargs": kwargs}]


def run_two_stage(args: argparse.Namespace, data_yaml: Path) -> tuple[Any, Any, list[dict[str, Any]]]:
    stage1_name = f"{args.name}_stage1"
    stage2_name = f"{args.name}_stage2"
    stages: list[dict[str, Any]] = []

    print("\n" + "=" * 72)
    print("Stage 1: frozen backbone")
    print("=" * 72)
    model = load_yolo(args.model)
    kwargs_s1 = train_kwargs(args, data_yaml, stage1_name, args.epochs_s1, args.freeze_s1, args.lr0_s1)
    results_s1 = model.train(**kwargs_s1)
    save_dir_s1 = save_dir_from_results(results_s1, args.project, stage1_name, model)
    stage1_weight = best_or_last_weight(save_dir_s1)
    stages.append({"name": stage1_name, "mode": "stage1", "save_dir": save_dir_s1, "weights": stage1_weight, "kwargs": kwargs_s1})

    print("\n" + "=" * 72)
    print("Stage 2: full fine-tuning")
    print("=" * 72)
    model = load_yolo(str(stage1_weight))
    kwargs_s2 = train_kwargs(args, data_yaml, stage2_name, args.epochs_s2, 0, args.lr0_s2)
    results_s2 = model.train(**kwargs_s2)
    save_dir_s2 = save_dir_from_results(results_s2, args.project, stage2_name, model)
    stages.append({"name": stage2_name, "mode": "stage2", "save_dir": save_dir_s2, "kwargs": kwargs_s2})
    return model, results_s2, stages


def parse_balance_splits(values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for value in values:
        normalized = value.lower()
        if normalized == "valid":
            normalized = "val"
        if normalized not in {"train", "val"}:
            raise SystemExit("--balance-splits only supports train and val.")
        parsed.add(normalized)
    return parsed


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.two_stage:
        args.mode = "two-stage"

    if args.imgsz is None:
        args.imgsz = 640 if args.quick else 1024
    if args.batch is None:
        args.batch = 2 if args.quick else -1
    if args.epochs is None:
        args.epochs = 1 if args.quick else 150
    if args.epochs_s1 is None:
        args.epochs_s1 = 1 if args.quick else 40
    if args.epochs_s2 is None:
        args.epochs_s2 = 1 if args.quick else 120
    if args.patience is None:
        args.patience = 1 if args.quick else 30
    if args.workers is None:
        args.workers = 0 if args.quick else 8
    if args.fraction is None:
        args.fraction = 0.05 if args.quick else 1.0
    if args.save_period is None:
        args.save_period = -1 if args.quick else 25
    if args.eval_split is None:
        args.eval_split = "val" if args.quick else "test"
    if isinstance(args.device, str) and args.device.lower() in {"", "auto", "none"}:
        args.device = None

    args.project = Path(args.project).expanduser().resolve()
    args.datasets_dir = Path(args.datasets_dir).expanduser().resolve()
    args.balance_splits = parse_balance_splits(args.balance_splits)
    return args


def build_run_config(
    args: argparse.Namespace,
    source_data_yaml: Path,
    training_data_yaml: Path,
    dataset_metadata: dict[str, Any],
    stages: list[dict[str, Any]],
    final_save_dir: Path,
    train_results: Any,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "source_data_yaml": source_data_yaml,
        "training_data_yaml": training_data_yaml,
        "model_checkpoint": args.model,
        "mode": args.mode,
        "quick": args.quick,
        "final_save_dir": final_save_dir,
        "dataset": dataset_metadata,
        "training": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "epochs": args.epochs,
            "epochs_s1": args.epochs_s1,
            "epochs_s2": args.epochs_s2,
            "patience": args.patience,
            "device": args.device,
            "workers": args.workers,
            "seed": args.seed,
            "fraction": args.fraction,
            "cache": args.cache,
            "amp": args.amp,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lr0_s1": args.lr0_s1,
            "lr0_s2": args.lr0_s2,
            "lrf": args.lrf,
            "weight_decay": args.weight_decay,
            "close_mosaic": args.close_mosaic,
        },
        "balance": {
            "enabled": args.balance,
            "minority_classes": args.minority_classes,
            "minority_copies": args.minority_copies,
            "balance_splits": sorted(args.balance_splits),
        },
        "stages": stages,
        "train_results": result_metrics(train_results),
    }


def prepare_training_dataset(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    source_data_yaml = resolve_data_yaml(args.data)
    if args.balance:
        training_data_yaml, metadata = create_balanced_dataset(
            source_yaml=source_data_yaml,
            datasets_dir=args.datasets_dir,
            target_classes=args.minority_classes,
            extra_copies=args.minority_copies,
            balance_splits=args.balance_splits,
        )
    else:
        out_dir = args.datasets_dir / f"resolved_{now_tag()}"
        training_data_yaml, metadata = make_resolved_data_yaml(source_data_yaml, out_dir)
    return source_data_yaml, training_data_yaml, metadata


def ensure_eval_split_exists(args: argparse.Namespace, training_data_yaml: Path) -> None:
    if args.no_eval:
        return
    data = read_data_yaml(training_data_yaml)
    if args.eval_split in data:
        return
    if args.eval_split == "test" and "val" in data:
        print("Requested eval split 'test' is missing; falling back to 'val'.")
        args.eval_split = "val"
        return
    raise SystemExit(f"Requested eval split '{args.eval_split}' is missing from {training_data_yaml}")


def train(args: argparse.Namespace) -> Path:
    checkpoint_warning(args.model)
    source_data_yaml, training_data_yaml, dataset_metadata = prepare_training_dataset(args)
    ensure_eval_split_exists(args, training_data_yaml)

    print(f"Source data YAML:   {source_data_yaml}")
    print(f"Training data YAML: {training_data_yaml}")
    print(f"Project:            {args.project}")
    print(f"Run name:           {args.name}")
    print(f"Mode:               {args.mode}")
    if args.quick:
        print("Quick smoke run is enabled.")
    if args.balance:
        print(f"Temporary balanced dataset: {dataset_metadata.get('balanced_root')}")

    if args.mode == "two-stage":
        model, train_results, stages = run_two_stage(args, training_data_yaml)
    else:
        model, train_results, stages = run_single_stage(args, training_data_yaml)

    final_save_dir = Path(stages[-1]["save_dir"]).resolve()
    run_config = build_run_config(
        args=args,
        source_data_yaml=source_data_yaml,
        training_data_yaml=training_data_yaml,
        dataset_metadata=dataset_metadata,
        stages=stages,
        final_save_dir=final_save_dir,
        train_results=train_results,
    )
    write_json(final_save_dir / "run_config.json", run_config)
    print(f"run_config.json saved to {final_save_dir / 'run_config.json'}")

    evaluate_model(model, training_data_yaml, args, final_save_dir, train_results)
    return final_save_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLO segmentation checkpoints on the cleaned CCV2 construction dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to dataset data.yaml.")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        default="yolov8x-seg.pt",
        help="YOLO segmentation checkpoint, e.g. yolov8x-seg.pt, yolo11x-seg.pt, yolo26x-seg.pt, or local best.pt.",
    )
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Ultralytics project directory.")
    parser.add_argument("--name", default="ccv2_seg", help="Run name.")
    parser.add_argument("--mode", choices=("single", "two-stage"), default="single", help="Training mode.")
    parser.add_argument("--two-stage", action="store_true", help="Shortcut for --mode two-stage.")
    parser.add_argument("--quick", "--smoke", action="store_true", help="Run a quick smoke training job.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size. Use -1 for Ultralytics autobatch.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs for single-stage mode.")
    parser.add_argument("--epochs-s1", type=int, default=None, help="Stage 1 epochs for two-stage mode.")
    parser.add_argument("--epochs-s2", type=int, default=None, help="Stage 2 epochs for two-stage mode.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--device", default="auto", help="Device passed to Ultralytics, e.g. auto, 0, 0,1, cpu.")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fraction", type=float, default=None, help="Dataset fraction. Quick mode defaults to 0.05.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset cache.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run name.")
    parser.add_argument("--no-eval", action="store_true", help="Skip final evaluation and still write metrics.json.")
    parser.add_argument("--eval-split", choices=("val", "test"), default=None, help="Split for final evaluation.")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation for final evaluation.")

    parser.add_argument("--balance", action="store_true", help="Create a temporary balanced dataset under runs/datasets.")
    parser.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR), help="Temporary dataset output directory.")
    parser.add_argument("--minority-classes", nargs="+", default=["1", "2"], help="Class IDs to oversample in the temp dataset.")
    parser.add_argument("--minority-copies", type=int, default=3, help="Extra copies per matching image when --balance is used.")
    parser.add_argument(
        "--balance-splits",
        nargs="+",
        default=["train"],
        help="Splits to duplicate in the temp dataset. Use train by default; valid/val is optional.",
    )

    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer.")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial LR for single-stage mode.")
    parser.add_argument("--lr0-s1", type=float, default=0.01, help="Stage 1 LR.")
    parser.add_argument("--lr0-s2", type=float, default=0.001, help="Stage 2 LR.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs.")
    parser.add_argument("--freeze", type=int, default=0, help="Layers to freeze in single-stage mode.")
    parser.add_argument("--freeze-s1", type=int, default=10, help="Layers to freeze in stage 1.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for final N epochs.")
    parser.add_argument("--save-period", type=int, default=None, help="Checkpoint save period. -1 disables periodic saves.")
    parser.set_defaults(amp=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP.")

    parser.add_argument("--degrees", type=float, default=25.0, help="Rotation augmentation.")
    parser.add_argument("--translate", type=float, default=0.2, help="Translation augmentation.")
    parser.add_argument("--scale", type=float, default=0.7, help="Scale augmentation.")
    parser.add_argument("--shear", type=float, default=8.0, help="Shear augmentation.")
    parser.add_argument("--perspective", type=float, default=0.0005, help="Perspective augmentation.")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation.")
    parser.add_argument("--mixup", type=float, default=0.15, help="MixUp augmentation.")
    parser.add_argument("--copy-paste", type=float, default=0.3, help="Copy-paste augmentation.")
    parser.add_argument("--erasing", type=float, default=0.1, help="Random erasing augmentation.")
    return parser


def main() -> None:
    parser = build_parser()
    args = finalize_args(parser.parse_args())
    final_save_dir = train(args)
    print("\nTraining finished.")
    print(f"Final run directory: {final_save_dir}")


if __name__ == "__main__":
    main()
