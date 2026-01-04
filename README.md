# YOLOv8 增量训练 (Fine-tuning) 教程

本项目记录了如何在保留 COCO 数据集原有能力的基础上，新增自定义类别（如 `wallet`）进行增量训练的步骤。

## 目录

1. [环境配置](#1-环境配置)
2. [数据集准备](#2-数据集准备)
3. [配置文件](#3-配置文件)
4. [预训练权重](#4-预训练权重)
5. [开始训练](#5-开始训练)
6. [结果评估](#6-结果评估)

---

## 1. 环境配置

建议使用 Conda 创建独立的虚拟环境，并安装 PyTorch 和 YOLOv8 官方库。

```bash
# 1. 创建虚拟环境 (Python 3.8)
conda create -n yolov8 python=3.8 -y
conda activate yolov8

# 2. 安装 PyTorch (根据你的显卡版本选择，以下为 CUDA 11.8 示例)
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. 安装 Ultralytics (YOLOv8 官方库)
pip install ultralytics
```

---

## 2. 数据集准备

这是最关键的一步。我们需要将自定义的数据集（这里以`wallet`为例）与 `COCO`数据集混合在一起。

> [!IMPORTANT]
> **注意：** 这里数据集一般放在 `datasets` 文件夹下。

请确保你的文件目录结构如下所示。

> **注意：**`wallet` 的图片和标签是直接放在 `coco` 对应的文件夹里的 。

```text
ultralytics/
    ├──datasets/
          ├──coco/
              ├──annotations/
              │   └──instances_val2017.json
              ├──images/
              │   ├──train2017/
              │   │  ├──coco_t0.jpg   # 原 COCO 图片
              │   │  ├──wallet_t0.jpg # 自定义wallet图
              │   │        .
              │   │        .
              │   │        .
              │   └──val2017/
              │      ├──coco_v0.jpg
              │      ├──wallet_v0.jpg
              │            .
              │            .
              │            .
              ├──labels/
              │   ├──train2017/
              │   │  ├──coco_t0.txt   # 原 COCO 标注
              │   │  ├──wallet_t0.txt # 自定义 wallet 标注
              │   │        .
              │   │        .
              │   │        .
              │   └──val2017/
              │       ├──coco_v0.txt
              │       ├──wallet_v0.txt
              │            .
              │            .
              │            .
```

---

## 3. 配置文件

在`datasets`文件夹下新建一个配置文件 `data.yaml`。

**注意修改点：**

- **nc**: 修改为 81（COCO 80类 + 新增 1类）[cite: 22]。
- **names**: 在列表最后添加新的类别 ID 和名称。本例中 `wallet` 的 ID 为 `80` 。

```yaml
# data.yaml
path: /path/to/your/datasets/coco # 请修改为你自己的实际数据集路径
train: images/train2017
val: images/val2017
# test: ../test/images

nc: 81 # 80 (COCO) + 1 (Wallet)

names:
  0: person
  1: bicycle
  2: car
  # ... (省略中间的 COCO 类别) ...
  79: toothbrush
  80: wallet # 新添加的类别 [cite: 21]
```

---

## 4. 预训练权重

下载官方的预训练权重文件（如 `yolov8n.pt`），并放置在 `weights/` 目录下 。

文件位置示例：

```text
ultralytics/
  ├── weights/
  │   ├── yolov8n.pt
  │   └── yolo11n.pt
  └── ...
```

---

## 5. 开始训练

新建 `train.py` 脚本并运行。针对微调任务，我们调整了部分参数（如关闭马赛克增强、使用 SGD 等）。

```python
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载预训练权重
    model = YOLO(model=r"./weights/yolov8n.pt", task="detect")

    # 开始训练
    model.train(
        data=r"./datasets/data.yaml",  # 配置文件路径
        imgsz=640,
        epochs=150,
        batch=64,
        workers=8,
        device="0",  # GPU 设备 ID
        optimizer="SGD",
        close_mosaic=3,  # 最后 3 个 epoch 关闭马赛克增强
        resume=False,  # 设置为True，继续上次训练
        mosaic=0.5,
        project="runs/train",
        name="exp_yolo_wallet",
        cache="disk",
        patience=20,
        save=True,
        val=True,
        amp=False,
    )
```

运行训练：

```bash
python train.py
```

## 6. 结果评估

运行文件 `eval_both_weight.py` 评估官网模型在 COCO 类别上的性能以及自己训练模型在COCO 类别和自定义类别上的性能。

```bash
python eval_both_weight.py
```
