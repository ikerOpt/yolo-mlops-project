# PPE Detection with YOLOv8 - MLOps Project

This project trains and evaluates a YOLOv8 object detection model to identify Personal Protective Equipment (PPE) elements in construction and industrial safety scenarios.

The model detects 14 classes, including helmets, gloves, goggles, masks, safety vests, ladders, people, and non-compliance cases such as missing PPE.

## Project Goal

The goal of this project is to build an end-to-end computer vision pipeline for PPE detection, covering:

- dataset configuration
- model training
- evaluation
- inference
- reproducible experimentation

This project is designed as a portfolio-ready AI Engineering / Computer Vision project.

## Dataset

The dataset contains labeled images for 14 classes:

- Fall-Detected
- Gloves
- Goggles
- Hardhat
- Ladder
- Mask
- NO-Gloves
- NO-Goggles
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest

### Data split

- Train images: **30,765**
- Validation images: **8,814**
- Total classes: **14**

## Model

- Model: **YOLOv8n**
- Framework: **Ultralytics YOLO**
- Training device: **NVIDIA GeForce RTX 2060 6GB**
- Image size: **640**
- Epochs: **50**
- Batch size: **8**
- AMP: **enabled**
- Workers: **0**

## Final Results

### Global metrics

| Metric | Value |
|---|---:|
| Precision | 0.691 |
| Recall | 0.791 |
| mAP@50 | 0.762 |
| mAP@50-95 | 0.488 |

### Key observations

- Strong performance on visually distinctive classes such as **Ladder**, **Goggles**, **Hardhat**, and **Person**
- Lower performance on more difficult classes such as **Mask**, **NO-Mask**, and especially **NO-Safety Vest**
- Stable training convergence with gradual performance improvements up to the final epochs

## Per-class highlights

| Class | mAP@50 |
|---|---:|
| Ladder | 0.948 |
| Goggles | 0.960 |
| Person | 0.924 |
| Hardhat | 0.895 |
| NO-Goggles | 0.933 |
| NO-Gloves | 0.882 |
| Mask | 0.493 |
| NO-Mask | 0.589 |
| NO-Safety Vest | 0.234 |

## Training Artifacts

Training plots and evaluation assets are available in:

- `assets/metrics/results.png`
- `assets/metrics/confusion_matrix.png`
- `assets/metrics/PR_curve.png`
- `assets/metrics/F1_curve.png`
- `assets/metrics/labels.jpg`
- `assets/metrics/results.csv`

## Inference Examples

Prediction examples are stored in:

- `assets/predictions/`

## Project Structure

```text
yolo-mlops-project/
├── assets/
│   ├── metrics/
│   └── predictions/
├── configs/
│   └── data.yaml
├── models/
│   └── best.pt
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md