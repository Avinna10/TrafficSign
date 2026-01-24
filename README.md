# Traffic Sign Detection with YOLOv8

A comprehensive traffic sign detection system trained on the TT100K dataset using YOLOv8. This project implements and compares baseline and attention-enhanced models for detecting 201 different traffic sign classes, achieving robust performance on Chinese road signs.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Features](#features)

## 🎯 Overview

This project implements a traffic sign detection system using YOLOv8n architecture with the following objectives:
- Detect and classify 201 different traffic sign types from the TT100K dataset
- Compare baseline YOLOv8n with attention-enhanced variants
- Achieve real-world performance on diverse road conditions
- Provide comprehensive evaluation metrics and visualizations

## 📊 Dataset

**TT100K (Tsinghua-Tencent 100K)**
- 100,000+ images with traffic sign annotations
- 201 traffic sign classes including:
  - Information signs (i1-i15)
  - Speed limits (il50-il110, pl5-pl120)
  - Prohibition signs (p1-p29)
  - Warning signs (w1-w66)
  - And many more specialized signs
- Split into train/validation/test sets
- YOLO format annotations

## 🤖 Models

### Baseline Model
- **Architecture**: YOLOv8n (Nano)
- **Configuration**: Standard YOLOv8n pretrained weights
- **Training**: 100 epochs with SGD optimizer

### Attention-Enhanced Model
- **Architecture**: YOLOv8n with attention mechanisms
- **Modifications**: Custom attention layers for improved feature extraction
- **Training**: 100 epochs with enhanced close_mosaic strategy

## 📁 Project Structure

```
TrafficSign/
├── traffic-sign.ipynb         # Main training and evaluation notebook
├── eda.ipynb                   # Exploratory data analysis
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── scripts/
│   └── utils.py                # Utility functions
```

## 🔧 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TrafficSign
```

2. Download the TT100K dataset:

Visit the official TT100K dataset page:
```
https://cg.cs.tsinghua.edu.cn/traffic-sign/
```

Download the dataset and extract it to the `tt100k_2021/` directory in the project root.

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Main dependencies:
- `ultralytics`: YOLOv8 framework
- `torch`, `torchvision`: Deep learning framework
- `opencv-python`: Image processing
- `matplotlib`: Visualization
- `pandas`: Data manipulation
- `scikit-learn`: Evaluation metrics

## 🚀 Usage

### Exploratory Data Analysis

Run the [eda.ipynb](eda.ipynb) notebook to explore the TT100K dataset:
```bash
jupyter notebook eda.ipynb
```

This notebook provides:
- Dataset statistics and distribution analysis
- Visualization of traffic sign classes
- Sample images and annotations
- Class imbalance analysis
- Data quality assessment

### Main Training and Evaluation

Run the [traffic-sign.ipynb](traffic-sign.ipynb) notebook for complete model training and evaluation:
```bash
jupyter notebook traffic-sign.ipynb
```

This notebook includes:
- Dataset preparation and YOLO format conversion
- Baseline model training and evaluation
- Attention-enhanced model training
- Performance comparison and visualizations
- Inference on test images
- Model export and deployment code

## ✨ Features

- **Dual Model Comparison**: Baseline vs. Attention-enhanced architectures
- **Comprehensive Training**: 100 epochs with systematic checkpoint saving
- **Data Augmentation**: Built-in YOLO augmentation strategies
- **Real-world Testing**: Validation on actual traffic sign images
- **Detailed Analytics**: Training curves, confusion matrices, and performance metrics
- **Reproducible**: Fixed seeds and deterministic training
- **Production Ready**: Model export and inference pipelines

## 📝 Training Configuration

### Common Parameters
- Image size: 640×640
- Batch size: 32
- Optimizer: SGD
- Device: GPU (CUDA)
- Workers: 4
- IoU threshold: 0.7
- Maximum detections: 300

### Model-Specific Settings
- **Baseline**: Patience 50, close_mosaic 20
- **Attention**: Patience 20, close_mosaic 10

## 📄 License

This project uses the TT100K dataset. Please refer to the dataset's original license terms.

## 🙏 Acknowledgments

- TT100K dataset by Tsinghua University and Tencent
- Ultralytics YOLOv8 framework
- PyTorch team
