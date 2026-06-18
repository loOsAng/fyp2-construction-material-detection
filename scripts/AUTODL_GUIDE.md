# AutoDL 训练操作清单

## 准备工作（本地）

1. 打包项目代码：
```bash
cd D:/xuexi/2526S2/fyp2
# 只打包必要文件，排除 best.pt（太大，会训练出新的）
tar -czf fyp_code.tar.gz \
  --exclude='best.pt' \
  --exclude='runs' \
  --exclude='dataset' \
  --exclude='InnoCount*' \
  --exclude='.git' \
  --exclude='__pycache__' \
  fyp_withoutRebar/
```

2. 打包数据集：
```bash
tar -czf ccv2_data.tar.gz CCV2.v2i.yolov81024/
```

## 租机器

1. AutoDL 官网 → 算力市场
2. 筛选：GPU=A100-PCIE-40GB, 地区选你最快的
3. 镜像选：**PyTorch 2.3.0 + Python 3.10 + CUDA 12.1** (或更新版本)
4. 创建实例

## 上传文件

进入 Jupyter 后，用左侧文件浏览器：

1. 上传 `fyp_code.tar.gz` → 终端执行：
```bash
cd /root/autodl-tmp
tar -xzf fyp_code.tar.gz
# 验证
ls fyp_withoutRebar/train.py  # 应该存在
```

2. 上传 `ccv2_data.tar.gz` → 终端执行：
```bash
cd /root/autodl-tmp
tar -xzf ccv2_data.tar.gz
# 验证
ls CCV2.v2i.yolov81024/data.yaml  # 应该存在
```

3. 上传 `train_autodl.ipynb` → 在 Jupyter 里打开，按顺序执行

## 训练完

1. 执行 Notebook 最后一个 Cell（打包 results.tar.gz）
2. 下载 results.tar.gz 到本地
3. **关机！**（Jupyter 右上角菜单 → 关机）
4. 本地解压：`tar -xzf results.tar.gz`
5. 把最好的 best.pt 复制到项目根目录替换旧的

## 预计耗时

| 步骤 | 时间 |
|------|------|
| 上传文件 (~250MB) | 5-10 分钟 |
| E1: YOLOv8x balanced 200ep | ~6-7 小时 |
| E2: YOLO11x balanced 200ep | ~7-8 小时 |
| E3: YOLOv8x baseline 200ep | ~5-6 小时 |
| 评估 (3 组 × 2 分辨率) | ~10 分钟 |
| **总计** | **~20 小时** |

## 费用

A100-PCIE-40GB ≈ ¥4-5/小时 → **总费用约 ¥80-100**

如果只想跑最重要的实验，注释掉 Notebook 第 4 个 Cell 里 EXPERIMENTS 中不需要的那些。
