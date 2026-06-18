"""Dataset audit utility for YOLO segmentation datasets.

This script is intentionally read-only. It summarizes image/label counts,
class imbalance, missing pairs, empty labels, and sample image dimensions.

Examples:
    python dataset_audit.py --data runs/datasets/ccv2_v5_resplit_seed42/data.yaml
    python dataset_audit.py --data dateset/CCV2.v5i.yolov8/data.yaml --json runs/dataset_audit_v5_raw.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small Roboflow data.yaml shape without requiring PyYAML."""
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = value.strip("[]").split(",")
            data[key] = [item.strip().strip("'\"") for item in items if item.strip()]
        elif value.isdigit():
            data[key] = int(value)
        else:
            data[key] = value.strip("'\"")
    return data


def _resolve_split_path(data_yaml: Path, split_value: str) -> Path:
    path = Path(split_value)
    if not path.is_absolute():
        path = (data_yaml.parent / path).resolve()
    if not path.exists() and split_value.startswith("../"):
        roboflow_path = (data_yaml.parent / split_value[3:]).resolve()
        if roboflow_path.exists():
            return roboflow_path
    return path


def _label_path_for_image(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def _image_path_for_label(label_path: Path, image_dir: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _dimension(path: Path) -> str:
    try:
        with Image.open(path) as image:
            return f"{image.width}x{image.height}"
    except Exception:
        return "unreadable"


def audit_dataset(data_yaml: Path) -> dict[str, Any]:
    config = _parse_simple_yaml(data_yaml)
    names = config.get("names", [])
    if not names:
        raise ValueError(f"No class names found in {data_yaml}")

    report: dict[str, Any] = {
        "data_yaml": str(data_yaml.resolve()),
        "names": names,
        "splits": {},
        "totals": {
            "images": 0,
            "labels": 0,
            "empty_labels": 0,
            "missing_labels": 0,
            "missing_images": 0,
            "instances": {name: 0 for name in names},
            "images_with_class": {name: 0 for name in names},
        },
    }

    total_instances = Counter()
    total_images_with_class = Counter()

    for split in ("train", "val", "valid", "test"):
        split_value = config.get(split)
        if split_value is None:
            continue

        image_dir = _resolve_split_path(data_yaml, split_value)
        label_dir = image_dir.parent / "labels"
        image_paths = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
        label_paths = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []

        instances = Counter()
        images_with_class = Counter()
        empty_labels = 0
        missing_labels = []
        missing_images = []
        dimensions = Counter()
        polygons_per_label = defaultdict(int)

        for image_path in image_paths:
            if not _label_path_for_image(image_path, label_dir).exists():
                missing_labels.append(image_path.name)
            dimensions[_dimension(image_path)] += 1

        for label_path in label_paths:
            if _image_path_for_label(label_path, image_dir) is None:
                missing_images.append(label_path.name)
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
                continue
            present = set()
            polygons_per_label[len(lines)] += 1
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                if 0 <= class_id < len(names):
                    instances[names[class_id]] += 1
                    present.add(names[class_id])
            for name in present:
                images_with_class[name] += 1

        split_report = {
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "images": len(image_paths),
            "labels": len(label_paths),
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
            "missing_images": missing_images,
            "instances": {name: instances[name] for name in names},
            "images_with_class": {name: images_with_class[name] for name in names},
            "dimensions": dict(dimensions),
            "objects_per_label_file": dict(sorted(polygons_per_label.items())),
        }
        report["splits"][split] = split_report

        report["totals"]["images"] += len(image_paths)
        report["totals"]["labels"] += len(label_paths)
        report["totals"]["empty_labels"] += empty_labels
        report["totals"]["missing_labels"] += len(missing_labels)
        report["totals"]["missing_images"] += len(missing_images)
        total_instances.update(instances)
        total_images_with_class.update(images_with_class)

    report["totals"]["instances"] = {name: total_instances[name] for name in names}
    report["totals"]["images_with_class"] = {name: total_images_with_class[name] for name in names}
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(f"Dataset: {report['data_yaml']}")
    print(f"Classes: {', '.join(report['names'])}")
    for split, item in report["splits"].items():
        print(f"\n[{split}]")
        print(f"images={item['images']} labels={item['labels']} empty_labels={item['empty_labels']}")
        print(f"missing_labels={len(item['missing_labels'])} missing_images={len(item['missing_images'])}")
        print(f"instances={item['instances']}")
        print(f"images_with_class={item['images_with_class']}")
        print(f"dimensions={item['dimensions']}")
    print("\n[totals]")
    print(json.dumps(report["totals"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a YOLO segmentation dataset.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--json", help="Optional JSON output path")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    report = audit_dataset(data_yaml)
    print_summary(report)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {out_path}")


if __name__ == "__main__":
    main()
