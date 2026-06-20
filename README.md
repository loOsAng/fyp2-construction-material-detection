# Construction Material Detection & Analysis - FYP2

YOLOv8x instance segmentation with mask post-processing and I-beam main-axis geometry analysis for construction site material inspection.

## Overview

This final-year project detects, segments, and counts three construction materials: **Brick**, **I-beam**, and **nail**. It uses YOLOv8x instance segmentation for material detection and combines the predicted masks with class-aware morphology, tiled inference for small objects, and geometry extraction for elongated steel profiles.

The Flask dashboard is designed as a demonstration and inspection console. It displays original images, segmentation masks, I-beam geometry overlays, per-class counts, confidence statistics, CSV export, and formal evaluation metrics.

## Features

| Layer | Capability |
|-------|------------|
| Detection | YOLOv8x-seg instance segmentation at 1024 x 1024, 3 classes |
| Post-processing | Class-aware morphological refinement and soft neighbour isolation |
| Geometry | I-beam main-axis estimation using skeleton PCA, RANSAC, and rotated-rectangle candidates |
| Tiled inference | Sliding-window nail detection while preserving full-frame Brick and I-beam results |
| Shadow enhancement | CLAHE on the L* channel with gamma correction for darker regions |
| Dashboard | Flask web console with batch upload, metrics, CSV export, and evaluation panel |
| Training | Single-stage, two-stage, and balanced oversampling workflows for YOLOv8/YOLO11 segmentation checkpoints |
| Evaluation | Box/mask mAP, per-class count error, and latency metrics |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Main checkpoint provided through Git LFS
# bestModelSelect/yolov8x_retrained/best.pt

# Launch dashboard
python app.py
# Open http://127.0.0.1:5000

# Train
python train.py --model yolov8x-seg.pt --device 0 --imgsz 1024 --batch 12 --epochs 150

# Evaluate
python evaluate.py --weights bestModelSelect/yolov8x_retrained/best.pt   --data <dataset>/data.yaml --split test --imgsz 1024 --save-visuals
```

## Project Structure

```text
newnewnewfyp2/
|-- app.py                         Flask dashboard entry point
|-- train.py                       Training pipeline
|-- evaluate.py                    Formal evaluation script
|-- dataset_audit.py               Read-only dataset statistics
|-- prepare_dataset_resplit.py     Source-group-aware dataset splitting
|-- requirements.txt               Python dependencies
|-- modules/
|   |-- model.py                   YOLO loading and checkpoint discovery
|   |-- inference.py               Inference, tiled inference, and shadow preprocessing
|   |-- skeleton.py                Mask post-processing and I-beam geometry extraction
|   `-- refine.py                  Optional SAM mask refinement module
|-- templates/
|   `-- index.html                 Dashboard UI
|-- tests/                         Unit tests
|-- test/                          Manual test images
|-- scripts/                       Cloud training guide and notebook
|-- runs/evaluate/                 Selected formal evaluation summaries
`-- bestModelSelect/               Git LFS checkpoints and training summaries
```

## Data Flow

```text
site image -> decode_image -> YOLO inference -> mask post-processing
           -> geometry extraction -> dashboard rendering -> CSV export
```

## Model Checkpoints

The submitted checkpoints are tracked with Git LFS:

```text
bestModelSelect/yolov8x_retrained/best.pt
bestModelSelect/yolov8x_balanced/best.pt
```

After cloning, make sure Git LFS files are available:

```bash
git lfs pull
```

## Key Evaluation Results

| Metric | Value |
|--------|-------|
| Mask mAP50 | 80.2% |
| Mask mAP50-95 | 47.5% |
| Nail exact count accuracy | 84.8% |
| Median end-to-end latency | 229 ms |

## Notes

- The repository includes the submitted checkpoints through Git LFS.
- A800 comparison run summaries are retained as `results.csv` files under `bestModelSelect/`.
- Most runtime outputs under `runs/` are ignored; the selected evaluation summaries referenced in the report are retained.
- The dashboard runs locally and does not require external services.
