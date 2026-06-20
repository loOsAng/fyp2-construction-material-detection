# Construction Material Detection & Analysis — FYP2

> YOLOv8x instance segmentation + robust I-beam main-axis geometry analysis for construction site material inspection.

---

## English

### Overview

An undergraduate final-year project (FYP2) that detects, segments, and counts three construction materials — **Brick**, **I-beam**, and **nail** — from site photographs using YOLOv8x instance segmentation. The core novelty is a **robust main-axis geometry extraction algorithm** for I-beam masks: three orientation candidates (skeleton PCA, RANSAC line fitting, rotated bounding box) vote via median angle consensus to produce a clean centreline, overcoming the flange-bias problem inherent to I/H-shaped steel profiles.

A Flask web dashboard serves as a VIVA defence evidence console, rendering segmentation masks, I-beam geometry overlays, per-class counts, confidence statistics, and formal evaluation metrics in real time.

### Features

| Layer | Capability |
|-------|-----------|
| **Detection** | YOLOv8x-seg instance segmentation at 1024×1024, 3 classes |
| **Post-processing** | Class-aware morphological refinement (I-beam: 5×5 close kernel, ≥50px min area; others: 3×3, ≥20px) + soft neighbour isolation |
| **Geometry** | I-beam main-axis via 3-candidate voting (skeleton PCA + RANSAC + rotated rect) → median angle → central endpoints |
| **Tiled inference** | Sliding-window nail detection preserving full-frame Brick/I-beam results |
| **Shadow enhancement** | CLAHE on L* channel + gamma correction for dark-area small-object recall |
| **Dashboard** | Flask web console with multi-image batch upload, real-time metrics, CSV export, formal evaluation panel |
| **Training** | Single-stage / two-stage / balanced oversampling; supports YOLOv8 and YOLO11 segmentation checkpoints |
| **Evaluation** | Independent mAP (box + mask), per-class count error, latency (mean / median / P95) |

### Quick Start

```bash
# Install
pip install -r requirements.txt

# Model weights
# Place the selected checkpoint at bestModelSelect/yolov8x_retrained/best.pt
# or put a fallback checkpoint at best.pt before launching the dashboard.

# Launch dashboard
python app.py
# → http://127.0.0.1:5000

# Train
python train.py --model yolov8x-seg.pt --device 0 --imgsz 1024 --batch 12 --epochs 150

# Evaluate
python evaluate.py --weights bestModelSelect/yolov8x_retrained/best.pt \
  --data <dataset>/data.yaml --split test --imgsz 1024 --save-visuals
```

### Architecture

```
newnewnewfyp2/
├── app.py                  Flask dashboard entry
├── train.py                Training pipeline
├── evaluate.py             Formal evaluation (mAP + count + latency)
├── dataset_audit.py        Read-only dataset statistics
├── prepare_dataset_resplit.py  Source-group-aware data splitting
├── requirements.txt
│
├── modules/
│   ├── model.py            YOLO loading, discovery, YOLO11 compatibility patches
│   ├── inference.py        YOLO predict, tiled inference, shadow preprocessing
│   ├── skeleton.py         Mask post-processing, I-beam main-axis geometry (~1360 lines)
│   └── refine.py           SAM mask refinement (optional, not in default pipeline)
│
├── templates/
│   └── index.html          VIVA evidence dashboard UI
│
├── tests/                  Unit tests (model selection, brick stack performance)
├── test/                   6 manual QA images (brick/ibeam/nail ×2)
├── scripts/                Cloud training guide & notebook
│
├── best.pt                 Local default weights (not tracked)
└── bestModelSelect/        Local trained checkpoints (not tracked)
```

### Data Flow

```
site image → decode_image (RGB) → run_inference / run_tiled_inference (YOLO)
    → apply_mask_postprocessing (morphology + largest component + isolation)
    → extract_clean_geometry_overlay (I-beam 3-candidate voting → main axis)
    → dashboard render (masks / geometry overlay / counts / CSV export)
```

---

## 中文

### 概述

本科毕业设计项目（FYP2），使用 YOLOv8x 实例分割从施工现场照片中检测、分割并计数三种建筑材料——**砖（Brick）**、**工字梁（I-beam）**、**钉子（nail）**。核心创新是工字梁掩码的**鲁棒主轴线几何提取算法**：三种朝向候选方案（骨架 PCA、RANSAC 鲁棒直线拟合、旋转矩形长边）通过中位数角度投票产生一条干净的中心线，解决了 I/H 型钢掩码翼缘像素拉偏 PCA 的问题。

Flask Web 控制台作为 VIVA 答辩证据面板，实时展示分割掩码、工字梁几何覆盖图、分类计数、置信度统计和正式评估指标。

### 功能特性

| 层级 | 能力 |
|------|------|
| **检测** | YOLOv8x-seg 实例分割，1024×1024，3 类 |
| **后处理** | 按类别差异化形态学精修（I-beam: 5×5 闭运算核, ≥50px 最小面积；其他: 3×3, ≥20px）+ 软隔离去粘连 |
| **几何分析** | 工字梁主轴线：3 候选投票（骨架PCA + RANSAC + 旋转矩形）→ 中位数角度 → 中心端点 |
| **分块推理** | 滑窗增强钉子检测，同时保留全图砖/工字梁结果 |
| **阴影增强** | L* 通道 CLAHE + Gamma 校正，提升暗区小目标召回 |
| **控制台** | Flask Web 界面，支持多图批量上传、实时指标、CSV 导出、正式评估面板 |
| **训练** | 单阶段 / 两阶段 / 类别平衡过采样；兼容 YOLOv8 和 YOLO11 分割模型 |
| **评估** | 独立 mAP（box + mask）、逐类计数误差、延迟（均值/中位数/P95） |

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 模型权重
# 启动前请将最终 checkpoint 放到 bestModelSelect/yolov8x_retrained/best.pt
# 或将备用 checkpoint 放到 best.pt。

# 启动控制台
python app.py
# → 浏览器打开 http://127.0.0.1:5000

# 训练
python train.py --model yolov8x-seg.pt --device 0 --imgsz 1024 --batch 12 --epochs 150

# 评估
python evaluate.py --weights bestModelSelect/yolov8x_retrained/best.pt \
  --data <dataset>/data.yaml --split test --imgsz 1024 --save-visuals
```

### 项目结构

```
newnewnewfyp2/
├── app.py                  Flask 控制台入口
├── train.py                训练流水线
├── evaluate.py             正式评估（mAP + 计数 + 延迟）
├── dataset_audit.py        数据集只读审计
├── prepare_dataset_resplit.py  按源图像组重划分数据集
├── requirements.txt        依赖清单
│
├── modules/
│   ├── model.py            模型加载/发现/YOLO11 兼容补丁
│   ├── inference.py        YOLO 推理 + 分块推理 + 阴影预处理
│   ├── skeleton.py         掩码后处理 + 工字梁主轴线几何（~1360 行）
│   └── refine.py           SAM 掩码精修（可选，非默认流程）
│
├── templates/
│   └── index.html          VIVA 证据控制台界面
│
├── tests/                  单元测试
├── test/                   6 张手工测试图
├── scripts/                云端训练指南与 Notebook
│
├── best.pt                 本地默认权重（不纳入 Git）
└── bestModelSelect/        本地训练模型目录（不纳入 Git）
```

### 数据流

```
工地照片 → decode_image (RGB) → run_inference / run_tiled_inference (YOLO)
    → apply_mask_postprocessing (形态学 + 最大连通域 + 软隔离)
    → extract_clean_geometry_overlay (工字梁 3 候选投票 → 主轴线)
    → 控制台渲染 (分割图 / 几何覆盖图 / 计数 / CSV 导出)
```

### 关键指标（YOLOv8x retrained, 1024px, CCV2 v5 数据集）

| 指标 | 数值 |
|------|------|
| Mask mAP50 | 80.2% |
| Mask mAP50-95 | 47.5% |
| Nail 精确计数率 | 84.8% |
| 推理延迟（中位数） | 229ms |
