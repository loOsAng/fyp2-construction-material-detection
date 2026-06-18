"""Prepare a leakage-safe YOLO segmentation dataset split.

The Roboflow export can contain the same source image in both train and valid
because augmented filenames differ only after the ".rf." token. This utility
copies a dataset into a new train/valid/test layout while keeping each source
image group in exactly one split.

Example:
    python prepare_dataset_resplit.py --data dateset/CCV2.v5i.yolov8/data.yaml \
        --out runs/datasets/ccv2_v5_resplit --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_NAMES = ["Brick", "I-beam", "nail"]


@dataclass
class Sample:
    image: Path
    label: Path
    source_split: str
    group_id: str
    image_hash: str
    class_counts: Counter[int] = field(default_factory=Counter)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid data.yaml: {path}")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def resolve_split_dir(data_yaml: Path, data: dict[str, Any], split_key: str) -> Path | None:
    raw = data.get(split_key)
    if raw is None:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (data_yaml.parent / path).resolve()
    if path.exists():
        return path
    if str(raw).startswith("../"):
        roboflow_path = (data_yaml.parent / str(raw)[3:]).resolve()
        if roboflow_path.exists():
            return roboflow_path
    return path


def source_group_id(image_path: Path) -> str:
    stem = image_path.stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    return stem.lower()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def parse_label_counts(label_path: Path, nc: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not label_path.exists():
        return counts
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or len(parts[1:]) % 2 != 0:
            raise SystemExit(f"Bad YOLO segmentation polygon at {label_path}:{line_no}")
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= nc:
            raise SystemExit(f"Class id {class_id} out of range at {label_path}:{line_no}")
        counts[class_id] += 1
    return counts


def collect_samples(data_yaml: Path) -> tuple[list[str], list[Sample]]:
    data = read_yaml(data_yaml)
    names = data.get("names") or DEFAULT_NAMES
    nc = int(data.get("nc", len(names)))
    if len(names) != nc:
        raise SystemExit(f"names length ({len(names)}) does not match nc ({nc}) in {data_yaml}")

    samples: list[Sample] = []
    for split_key in ("train", "valid", "val", "test"):
        image_dir = resolve_split_dir(data_yaml, data, split_key)
        if image_dir is None or not image_dir.exists():
            continue
        canonical_split = "valid" if split_key == "val" else split_key
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = label_for_image(image_path)
            if not label_path.exists():
                raise SystemExit(f"Missing label for {image_path}")
            samples.append(
                Sample(
                    image=image_path,
                    label=label_path,
                    source_split=canonical_split,
                    group_id=source_group_id(image_path),
                    image_hash=sha1_file(image_path),
                    class_counts=parse_label_counts(label_path, nc),
                )
            )
    return list(names), samples


def dedupe_exact_images(samples: list[Sample]) -> tuple[list[Sample], list[dict[str, str]]]:
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_hash[sample.image_hash].append(sample)

    kept: list[Sample] = []
    dropped: list[dict[str, str]] = []
    for image_hash, group in sorted(by_hash.items()):
        group = sorted(group, key=lambda item: (item.source_split != "train", str(item.image)))
        kept.append(group[0])
        for sample in group[1:]:
            dropped.append(
                {
                    "hash": image_hash,
                    "kept": str(group[0].image),
                    "dropped": str(sample.image),
                }
            )
    return kept, dropped


def split_groups(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1.0")

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)

    group_items = list(groups.items())

    total_images = len(samples)
    total_counts: Counter[int] = Counter()
    total_presence: Counter[int] = Counter()
    for sample in samples:
        total_counts.update(sample.class_counts)
        for cid in sample.class_counts:
            total_presence[cid] += 1

    targets = {
        "train": {
            "images": total_images * train_ratio,
            "counts": {cid: count * train_ratio for cid, count in total_counts.items()},
            "presence": {cid: count * train_ratio for cid, count in total_presence.items()},
        },
        "valid": {
            "images": total_images * val_ratio,
            "counts": {cid: count * val_ratio for cid, count in total_counts.items()},
            "presence": {cid: count * val_ratio for cid, count in total_presence.items()},
        },
        "test": {
            "images": total_images * test_ratio,
            "counts": {cid: count * test_ratio for cid, count in total_counts.items()},
            "presence": {cid: count * test_ratio for cid, count in total_presence.items()},
        },
    }

    def group_counts(group_samples: list[Sample]) -> Counter[int]:
        counts: Counter[int] = Counter()
        for sample in group_samples:
            counts.update(sample.class_counts)
        return counts

    def group_presence(group_samples: list[Sample]) -> Counter[int]:
        presence: Counter[int] = Counter()
        for sample in group_samples:
            for cid in sample.class_counts:
                presence[cid] += 1
        return presence

    group_payload = [
        (group_id, group_samples, group_counts(group_samples), group_presence(group_samples))
        for group_id, group_samples in group_items
    ]

    def evaluate(candidate: dict[str, list[Sample]]) -> float:
        score = 0.0
        for split in ("train", "valid", "test"):
            split_counts: Counter[int] = Counter()
            split_presence: Counter[int] = Counter()
            for sample in candidate[split]:
                split_counts.update(sample.class_counts)
                for cid in sample.class_counts:
                    split_presence[cid] += 1

            image_ratio = len(candidate[split]) / max(total_images, 1)
            target_ratio = {"train": train_ratio, "valid": val_ratio, "test": test_ratio}[split]
            score += abs(image_ratio - target_ratio) * 4.0

            for cid, total_count in total_counts.items():
                if total_count <= 0:
                    continue
                class_ratio = split_counts[cid] / total_count
                score += abs(class_ratio - target_ratio)

            for cid, total_count in total_presence.items():
                if total_count <= 0:
                    continue
                presence_ratio = split_presence[cid] / total_count
                score += abs(presence_ratio - target_ratio) * 2.0

            # Missing a class in val/test makes per-class evaluation weak.
            if split in ("valid", "test"):
                for cid, total_count in total_counts.items():
                    if total_count > 0 and split_counts[cid] == 0:
                        score += 3.0
        return score

    def assign_once(trial_seed: int) -> dict[str, list[Sample]]:
        rng = random.Random(trial_seed)
        shuffled = list(group_payload)
        rng.shuffle(shuffled)

        assigned: dict[str, list[Sample]] = {"train": [], "valid": [], "test": []}
        assigned_counts: dict[str, Counter[int]] = {key: Counter() for key in assigned}
        assigned_presence: dict[str, Counter[int]] = {key: Counter() for key in assigned}

        def projected_score(
            split: str,
            group_samples: list[Sample],
            counts: Counter[int],
            presence: Counter[int],
        ) -> float:
            next_images = len(assigned[split]) + len(group_samples)
            image_score = abs(next_images - targets[split]["images"]) / max(targets[split]["images"], 1.0)
            class_score = 0.0
            for cid, target_count in targets[split]["counts"].items():
                next_count = assigned_counts[split][cid] + counts[cid]
                class_score += abs(next_count - target_count) / max(target_count, 1.0)
                if next_count > target_count * 1.25 and split in ("valid", "test"):
                    class_score += 0.8
            presence_score = 0.0
            for cid, target_count in targets[split]["presence"].items():
                next_count = assigned_presence[split][cid] + presence[cid]
                presence_score += abs(next_count - target_count) / max(target_count, 1.0)
            if next_images > targets[split]["images"] * 1.20:
                image_score += 2.0
            return image_score + class_score + presence_score * 1.5

        # Larger groups are harder to place; handle them early but keep random
        # tie-breaks so the search can find a balanced class distribution.
        shuffled.sort(key=lambda item: (-len(item[1]), rng.random()))
        for _group_id, group_samples, counts, presence in shuffled:
            split = min(
                ("train", "valid", "test"),
                key=lambda key: projected_score(key, group_samples, counts, presence),
            )
            assigned[split].extend(group_samples)
            assigned_counts[split].update(counts)
            assigned_presence[split].update(presence)
        return assigned

    def assign_random(trial_seed: int) -> dict[str, list[Sample]]:
        rng = random.Random(trial_seed)
        assigned: dict[str, list[Sample]] = {"train": [], "valid": [], "test": []}
        for _group_id, group_samples, _counts, _presence in group_payload:
            value = rng.random()
            if value < train_ratio:
                split = "train"
            elif value < train_ratio + val_ratio:
                split = "valid"
            else:
                split = "test"
            assigned[split].extend(group_samples)
        return assigned

    best = None
    best_score = float("inf")
    for offset in range(2000):
        for candidate in (
            assign_random(seed + offset),
            assign_once(seed + offset),
        ):
            score = evaluate(candidate)
            if score < best_score:
                best = candidate
                best_score = score

    if best is None:
        raise SystemExit("Failed to create a dataset split.")
    return best


def unique_output_name(sample: Sample, used_names: set[str]) -> str:
    base = sample.image.name
    if base not in used_names:
        used_names.add(base)
        return base
    suffix = sample.image_hash[:10]
    candidate = f"{sample.image.stem}_{suffix}{sample.image.suffix.lower()}"
    counter = 1
    while candidate in used_names:
        candidate = f"{sample.image.stem}_{suffix}_{counter}{sample.image.suffix.lower()}"
        counter += 1
    used_names.add(candidate)
    return candidate


def copy_dataset(out_root: Path, splits: dict[str, list[Sample]], names: list[str]) -> dict[str, Any]:
    used_names: set[str] = set()
    summary: dict[str, Any] = {"splits": {}, "names": names}

    for split, samples in splits.items():
        image_out = out_root / split / "images"
        label_out = out_root / split / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        counts: Counter[int] = Counter()
        images_with_class: Counter[int] = Counter()

        for sample in sorted(samples, key=lambda item: str(item.image)):
            out_name = unique_output_name(sample, used_names)
            out_image = image_out / out_name
            out_label = label_out / f"{Path(out_name).stem}.txt"
            shutil.copy2(sample.image, out_image)
            shutil.copy2(sample.label, out_label)
            counts.update(sample.class_counts)
            for cid in sample.class_counts:
                images_with_class[cid] += 1

        summary["splits"][split] = {
            "images": len(samples),
            "instances": {names[cid]: counts[cid] for cid in range(len(names))},
            "images_with_class": {names[cid]: images_with_class[cid] for cid in range(len(names))},
        }

    data_yaml_text = "\n".join(
        [
            f"path: {str(out_root.resolve()).replace(chr(92), '/')}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            f"nc: {len(names)}",
            f"names: {json.dumps(names, ensure_ascii=False)}",
            "",
        ]
    )
    (out_root / "data.yaml").write_text(data_yaml_text, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-split a YOLO segmentation dataset without source-image leakage.")
    parser.add_argument("--data", required=True, help="Input data.yaml")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-exact-duplicates", action="store_true", help="Do not remove exact duplicate images.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output directory.")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    out_root = Path(args.out).resolve()
    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {out_root}. Use --overwrite to replace it.")
        shutil.rmtree(out_root)

    names, samples = collect_samples(data_yaml)
    if names != DEFAULT_NAMES:
        print(f"WARNING: expected names {DEFAULT_NAMES}, got {names}")

    dropped_duplicates: list[dict[str, str]] = []
    if not args.keep_exact_duplicates:
        samples, dropped_duplicates = dedupe_exact_images(samples)

    splits = split_groups(samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    summary = copy_dataset(out_root, splits, names)
    summary["source_data_yaml"] = str(data_yaml)
    summary["output_data_yaml"] = str(out_root / "data.yaml")
    summary["dedupe_exact_images"] = not args.keep_exact_duplicates
    summary["dropped_exact_duplicates"] = dropped_duplicates
    summary["seed"] = args.seed

    summary_path = out_root / "resplit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary["splits"], indent=2, ensure_ascii=False))
    print(f"\nWrote dataset: {out_root}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
