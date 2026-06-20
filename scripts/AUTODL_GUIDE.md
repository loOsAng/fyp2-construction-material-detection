# AutoDL Training Guide

This guide records the cloud training workflow used for the project.

## Local Preparation

Package the project code without large generated outputs:

```bash
tar --exclude=best.pt --exclude=bestModelSelect --exclude=runs -czf fyp_code.tar.gz newnewnewfyp2
```

Package the dataset separately:

```bash
tar -czf ccv2_data.tar.gz CCV2.v2i.yolov81024
```

## Cloud Instance

Recommended instance:

- GPU: A100-PCIE-40GB or equivalent
- Image: PyTorch with Python 3.10 and CUDA 12.x
- Disk: enough space for dataset, checkpoints, and training outputs

## Upload Files

Upload these files to the cloud workspace:

- `fyp_code.tar.gz`
- `ccv2_data.tar.gz`
- `train_autodl.ipynb`

Extract the archives:

```bash
tar -xzf fyp_code.tar.gz
tar -xzf ccv2_data.tar.gz
```

Check that the main files exist:

```bash
ls newnewnewfyp2/train.py
ls CCV2.v2i.yolov81024/data.yaml
```

## Training

Open `train_autodl.ipynb` in Jupyter and run the cells in order. The notebook installs dependencies, checks the dataset, and starts the selected training run.

The recommended final configuration is YOLOv8x segmentation at 1024 image size with balanced training settings.

## After Training

Download the selected checkpoint and training summary files:

```text
best.pt
results.csv
results.png
```

Place the final checkpoint under:

```text
bestModelSelect/yolov8x_retrained/best.pt
```

## Evaluation

Run formal evaluation after placing the checkpoint locally:

```bash
python evaluate.py --weights bestModelSelect/yolov8x_retrained/best.pt   --data <dataset>/data.yaml --split test --imgsz 1024 --save-visuals
```
