# ISL Detection with YOLOv11

Real-time **Indian Sign Language (ISL)** detection using the [Ultralytics YOLO11m](https://docs.ultralytics.com/) object-detection model. The project trains a medium-sized YOLO11 network on a custom ISL dataset hosted on Roboflow and achieves **mAP50 ≈ 98.9 %** on the validation split.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Results](#results)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Indian Sign Language is the primary communication medium for the deaf and hard-of-hearing community in India. This project builds an end-to-end detection pipeline that:

1. Downloads and analyzes the ISL dataset from Roboflow.
2. Fine-tunes a pre-trained **YOLO11m** backbone on 123 ISL classes (alphabets, digits, and common words/phrases).
3. Evaluates the trained model and exports the best checkpoint for downstream use.

---

## Dataset

| Split      | Images |
|------------|-------:|
| Train      | 17,663 |
| Validation |  1,017 |
| Test       |    288 |

**Source:** [Roboflow – islfinaldataset v4](https://roboflow.com/) (`visheshs-workspace/islfinaldataset`)

### Classes (123 total)

The dataset covers:

- **Digits:** `0`–`9`
- **Alphabets:** `A`–`Z`
- **Words & Phrases:** `Bad`, `Band Aid`, `Born`, `Brother`, `Bye`, `Cough`, `Eat`, `Father`, `Food`, `Friend`, `Good`, `Hello`, `Help`, `Home`, `House`, `I-Love-You`, `Indian`, `Language`, `Like`, `Loud`, `Love`, `Mummy`, `Namaste`, `Name`, `No`, `Peace`, `Place`, `Please`, `Quiet`, `Request`, `Sleeping`, `Sorry`, `Stop`, `Strong`, `Thank-you`, `Time`, `Today`, `Water`, `What`, `When`, `Yes`, `Your`, `blue`, `college`, `drink`, `drive`, `food`, `friday`, `green`, `hii`, `home`, `how are you`, `monday`, `namaste`, `okay`, `orange`, `pink`, `purple`, `red`, `salute`, `school`, `sick`, `sign`, `sleep`, `smile`, `stand`, `stop`, `strong`, `studyy`, `sunday`, `thursday`, `tuesday`, `wednesday`, `woman`, `yellow`, `you`, and more.

---

## Model Architecture

| Property        | Value                    |
|-----------------|--------------------------|
| Base model      | `yolo11m.pt`             |
| Variant         | YOLO11m (medium)         |
| Layers          | 232                      |
| Parameters      | 20,147,841               |
| GFLOPs          | 68.7                     |
| Input size      | 640 × 640                |
| Output heads    | 3 (multi-scale detection)|

The detection head (`Detect`) is configured for **123 classes**.

---

## Training Setup

Training was performed on **Kaggle** using an NVIDIA Tesla T4 GPU (15 GB VRAM).

| Hyperparameter   | Value       |
|------------------|-------------|
| Epochs           | 50–100      |
| Batch size       | 16          |
| Image size       | 640         |
| Optimizer        | Auto (SGD)  |
| Initial LR       | 0.01        |
| Final LR factor  | 0.01        |
| Weight decay     | 0.0005      |
| Patience (ES)    | 30          |
| AMP              | Enabled     |
| Workers          | 4           |

### Data Augmentation

| Augmentation    | Value  |
|-----------------|--------|
| HSV-Hue         | 0.015  |
| HSV-Saturation  | 0.7    |
| HSV-Value       | 0.4    |
| Rotation        | ±15°   |
| Translation     | 0.1    |
| Scale           | 0.5    |
| Shear           | 2.0°   |
| Flip LR         | 0.5    |
| Mosaic          | 1.0    |
| Close-mosaic    | last 10 epochs |

---

## Results

Evaluated on the **1,017-image validation split**:

| Metric       | Value  |
|--------------|-------:|
| mAP50        | 98.91% |
| mAP50-95     | 81.84% |
| Precision    | 96.61% |
| Recall       | 97.34% |

Inference speed on Tesla T4: **~32 ms / image** (640 × 640).

---

## Requirements

```
ultralytics
roboflow
torch>=2.0
numpy
pandas
pyyaml
```

Install all dependencies with:

```bash
pip install ultralytics roboflow
```

> **Note:** CUDA-capable GPU with at least 8 GB VRAM is strongly recommended for training. CPU inference is supported but slow.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/VisheshKamble/Isldetection-yolov11.git
cd Isldetection-yolov11
```

### 2. Install dependencies

```bash
pip install ultralytics roboflow
```

### 3. Download the dataset

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("visheshs-workspace").project("islfinaldataset")
dataset = project.version(4).download("yolov8")   # YOLO11 uses the same label format as YOLOv8; fall back to "yolov8" if "yolov11" is unavailable
data_yaml_path = f"{dataset.location}/data.yaml"
```

### 4. Train

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,           # GPU index; use "cpu" for CPU training
    project="isl_training",
    name="yolov11m_isl",
)
```

### 5. Evaluate

```python
metrics = model.val()
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### 6. Run inference

```python
results = model.predict(
    source="path/to/image_or_video",
    conf=0.5,
    save=True,
)
```

### 7. Run the training notebook

Open [`isl_yolov11_training.ipynb`](isl_yolov11_training.ipynb) in Jupyter / Kaggle and execute cells sequentially.

---

## Repository Structure

```
Isldetection-yolov11/
├── isl_yolov11_training.ipynb   # Main training & evaluation notebook
├── finalmodeltrainingisl        # Kaggle notebook (JSON) used for final model training
└── README.md
```

---

## Acknowledgements

- [Ultralytics YOLO11](https://docs.ultralytics.com/) — model framework
- [Roboflow](https://roboflow.com/) — dataset hosting and management
- [Kaggle](https://www.kaggle.com/) — GPU compute environment
