"""Formal evaluation script for YOLO segmentation/counting experiments.

This script intentionally keeps evaluation outputs separate from training code:
- Ultralytics YOLO.val() is used for box/mask mAP.
- Per-image predictions are used for count-error metrics and latency stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
SPLIT_DIR_CANDIDATES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}
VISUAL_LIMIT = 20
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO segmentation mAP, counts, and latency.")
    parser.add_argument("--weights", default="best.pt", help="Path to YOLO weights (.pt).")
    parser.add_argument("--data", default=str(default_data_yaml_path()), help="Path to YOLO data.yaml.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate: train, val/valid, or test.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference/validation image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for val/predict.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS/validation.")
    parser.add_argument("--output", default="runs/evaluate", help="Directory for metrics.json and count_errors.csv.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers for Ultralytics validation. Use 0 on locked-down Windows environments.",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help=f"Save up to {VISUAL_LIMIT} annotated prediction images under the output directory.",
    )
    return parser.parse_args()


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if isinstance(value, (int, float, str, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return str(value)


def normalize_split(split: str) -> str:
    key = split.strip().lower()
    if key not in SPLIT_ALIASES:
        valid = ", ".join(sorted(SPLIT_ALIASES))
        raise ValueError(f"Unsupported split '{split}'. Use one of: {valid}")
    return SPLIT_ALIASES[key]


def load_data_yaml(data_path: Path) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Data YAML must contain a mapping: {data_path}")
    return data


def names_from_data(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    nc = int(data.get("nc", 0) or 0)
    return {idx: str(idx) for idx in range(nc)}


def dataset_base(data: dict[str, Any], data_path: Path) -> Path:
    raw_path = data.get("path")
    if not raw_path:
        return data_path.parent

    root = Path(str(raw_path))
    if root.is_absolute():
        return root
    return (data_path.parent / root).resolve()


def existing_path_candidates(raw: str, base: Path, data_path: Path, split_key: str) -> Iterable[Path]:
    raw_path = Path(raw)

    if raw_path.is_absolute():
        yield raw_path
    else:
        yield (base / raw_path).resolve()
        yield (data_path.parent / raw_path).resolve()

        parts = raw_path.parts
        while parts and parts[0] in {".", ".."}:
            parts = parts[1:]
        if parts:
            yield (data_path.parent / Path(*parts)).resolve()

    for split_dir in SPLIT_DIR_CANDIDATES.get(split_key, (split_key,)):
        yield (data_path.parent / split_dir / "images").resolve()
        yield (base / split_dir / "images").resolve()


def resolve_existing_path(raw: str, base: Path, data_path: Path, split_key: str) -> Path:
    seen: set[Path] = set()
    for candidate in existing_path_candidates(raw, base, data_path, split_key):
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(p) for p in seen)
    raise FileNotFoundError(f"Could not resolve split path '{raw}'. Searched:\n  {searched}")


def resolve_split_sources(data: dict[str, Any], data_path: Path, split_key: str) -> list[Path]:
    if split_key not in data:
        raise KeyError(f"Split '{split_key}' is not defined in {data_path}")

    raw_value = data[split_key]
    base = dataset_base(data, data_path)
    raw_items = raw_value if isinstance(raw_value, list) else [raw_value]
    return [resolve_existing_path(str(item), base, data_path, split_key) for item in raw_items]


def image_paths_from_source(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p.resolve() for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)

    if source.is_file() and source.suffix.lower() == ".txt":
        images: list[Path] = []
        with source.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                path = Path(line)
                if not path.is_absolute():
                    path = (source.parent / path).resolve()
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)
        return sorted(images)

    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source.resolve()]

    raise ValueError(f"Unsupported image source: {source}")


def collect_split_images(data: dict[str, Any], data_path: Path, split_key: str) -> list[Path]:
    images: list[Path] = []
    for source in resolve_split_sources(data, data_path, split_key):
        images.extend(image_paths_from_source(source))

    unique = sorted(dict.fromkeys(images))
    if not unique:
        raise FileNotFoundError(f"No images found for split '{split_key}'")
    return unique


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_gt_counts(label_path: Path, class_count: int) -> list[int]:
    counts = [0] * class_count
    if not label_path.exists():
        return counts

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if not fields:
                continue
            try:
                class_id = int(float(fields[0]))
            except ValueError:
                continue
            if 0 <= class_id < class_count:
                counts[class_id] += 1
    return counts


def read_pred_counts(result: Any, class_count: int) -> list[int]:
    counts = [0] * class_count
    boxes = getattr(result, "boxes", None)
    classes = getattr(boxes, "cls", None) if boxes is not None else None
    if classes is None:
        return counts

    for raw_class in classes.detach().cpu().tolist():
        class_id = int(raw_class)
        if 0 <= class_id < class_count:
            counts[class_id] += 1
    return counts


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": percentile(values, 95),
    }


def count_metric(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "samples": 0,
            "gt_total": 0,
            "pred_total": 0,
            "mae": None,
            "rmse": None,
            "exact_count_accuracy": None,
        }

    abs_errors = [float(row["abs_error"]) for row in rows]
    sq_errors = [float(row["squared_error"]) for row in rows]
    exact = [1.0 if row["exact"] else 0.0 for row in rows]
    return {
        "samples": len(rows),
        "gt_total": int(sum(int(row["gt_count"]) for row in rows)),
        "pred_total": int(sum(int(row["pred_count"]) for row in rows)),
        "mae": statistics.fmean(abs_errors),
        "rmse": math.sqrt(statistics.fmean(sq_errors)),
        "exact_count_accuracy": statistics.fmean(exact),
    }


def metric_group_to_dict(group: Any) -> dict[str, Any] | None:
    if group is None:
        return None

    metric_names = ("mp", "mr", "map50", "map75", "map", "fitness")
    output: dict[str, Any] = {}
    for name in metric_names:
        if hasattr(group, name):
            attr = getattr(group, name)
            value = attr() if callable(attr) and name == "fitness" else attr
            output[name] = make_json_safe(value)
    if hasattr(group, "maps"):
        output["per_class_map50_95"] = make_json_safe(getattr(group, "maps"))
    return output


def val_metrics_to_dict(metrics: Any) -> dict[str, Any]:
    results_dict = getattr(metrics, "results_dict", None) or {}
    return {
        "box": metric_group_to_dict(getattr(metrics, "box", None)),
        "mask": metric_group_to_dict(getattr(metrics, "seg", None) or getattr(metrics, "mask", None)),
        "results_dict": make_json_safe(results_dict),
        "speed": make_json_safe(getattr(metrics, "speed", None)),
    }


def write_resolved_data_yaml(data: dict[str, Any], data_path: Path, output_dir: Path) -> Path:
    resolved = dict(data)
    for split_key in ("train", "val", "test"):
        if split_key not in data:
            continue
        try:
            sources = resolve_split_sources(data, data_path, split_key)
        except Exception:
            continue
        resolved[split_key] = [source.as_posix() for source in sources] if len(sources) > 1 else sources[0].as_posix()

    resolved_path = output_dir / "resolved_data.yaml"
    with resolved_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(make_json_safe(resolved), f, sort_keys=False, allow_unicode=True)
    return resolved_path


def evaluate_counts(
    model: Any,
    image_paths: list[Path],
    class_names: dict[int, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    class_items = sorted(class_names.items())
    class_count = max(class_names) + 1
    class_rows: list[dict[str, Any]] = []
    total_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    inference_latencies: list[float] = []
    end_to_end_latencies: list[float] = []

    visuals_dir = output_dir / "visuals"
    visuals_saved = 0
    if args.save_visuals:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        gt_counts = read_gt_counts(label_path_for_image(image_path), class_count)

        start = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            save=False,
        )
        end_to_end_ms = (time.perf_counter() - start) * 1000.0
        result = results[0]

        speed = getattr(result, "speed", None) or {}
        inference_ms = float(speed.get("inference", end_to_end_ms))
        inference_latencies.append(inference_ms)
        end_to_end_latencies.append(end_to_end_ms)

        pred_counts = read_pred_counts(result, class_count)
        image_name = image_path.name

        for class_id, class_name in class_items:
            gt_count = gt_counts[class_id]
            pred_count = pred_counts[class_id]
            error = pred_count - gt_count
            row = {
                "row_type": "class",
                "image": image_name,
                "image_path": str(image_path),
                "class_id": class_id,
                "class_name": class_name,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error * error,
                "exact": pred_count == gt_count,
                "inference_ms": inference_ms,
                "end_to_end_ms": end_to_end_ms,
            }
            class_rows.append(row)
            csv_rows.append(row)

        total_gt = sum(gt_counts)
        total_pred = sum(pred_counts)
        total_error = total_pred - total_gt
        total_row = {
            "row_type": "total",
            "image": image_name,
            "image_path": str(image_path),
            "class_id": "overall",
            "class_name": "overall",
            "gt_count": total_gt,
            "pred_count": total_pred,
            "error": total_error,
            "abs_error": abs(total_error),
            "squared_error": total_error * total_error,
            "exact": total_pred == total_gt,
            "inference_ms": inference_ms,
            "end_to_end_ms": end_to_end_ms,
        }
        total_rows.append(total_row)
        csv_rows.append(total_row)

        if args.save_visuals and visuals_saved < VISUAL_LIMIT:
            result.save(filename=str(visuals_dir / image_name))
            visuals_saved += 1

    by_class: dict[str, Any] = {}
    for class_id, class_name in class_items:
        rows = [row for row in class_rows if row["class_id"] == class_id]
        by_class[f"{class_id}:{class_name}"] = count_metric(rows)

    summary = {
        "by_class": by_class,
        "overall": count_metric(class_rows),
        "per_image_total": count_metric(total_rows),
        "latency": {
            "inference": latency_summary(inference_latencies),
            "end_to_end": latency_summary(end_to_end_latencies),
        },
        "visuals_saved": visuals_saved,
        "visuals_dir": str(visuals_dir) if args.save_visuals else None,
    }
    return summary, csv_rows


def write_count_errors_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_type",
        "image",
        "image_path",
        "class_id",
        "class_name",
        "gt_count",
        "pred_count",
        "error",
        "abs_error",
        "squared_error",
        "exact",
        "inference_ms",
        "end_to_end_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep Ultralytics settings local to the workspace/output. This avoids user
    # profile permission errors on locked-down Windows setups.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / ".ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()
    split_key = normalize_split(args.split)
    data = load_data_yaml(data_path)
    class_names = names_from_data(data)
    if not class_names:
        raise ValueError("No class names found in data.yaml")

    image_paths = collect_split_images(data, data_path, split_key)
    resolved_data_path = write_resolved_data_yaml(data, data_path, output_dir)

    model = YOLO(str(weights_path))

    val_results = model.val(
        data=str(resolved_data_path),
        split=split_key,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=str(output_dir / "ultralytics_val"),
        name=split_key,
        exist_ok=True,
        plots=False,
        verbose=False,
        workers=args.workers,
    )

    count_summary, count_rows = evaluate_counts(model, image_paths, class_names, args, output_dir)

    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "count_errors.csv"
    write_count_errors_csv(csv_path, count_rows)

    metrics = {
        "weights": str(weights_path),
        "data": str(data_path),
        "resolved_data": str(resolved_data_path),
        "split": split_key,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "num_images": len(image_paths),
        "classes": {str(k): v for k, v in class_names.items()},
        "map": val_metrics_to_dict(val_results),
        "counts": count_summary,
        "outputs": {
            "metrics_json": str(metrics_path),
            "count_errors_csv": str(csv_path),
            "ultralytics_val_dir": str(output_dir / "ultralytics_val" / split_key),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(metrics), f, indent=2, ensure_ascii=False)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote count errors: {csv_path}")


if __name__ == "__main__":
    main()
