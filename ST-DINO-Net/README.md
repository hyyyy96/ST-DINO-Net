# ST-DINO-Net: Spatiotemporal Dual-Stream Network for Ground-Based Cloud Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of the paper: **"A Spatiotemporal Method for Ground-Based Cloud Classification by Fusing Static DINO Texture and Dynamic Optical Flow"** .

## 📋 Overview

ST-DINO-Net is a dual-stream neural network for ground-based cloud image classification that integrates:
- **Static texture features** extracted by self-supervised DINO (Vision Transformer)
- **Dynamic motion features** extracted from optical flow maps using ResNet-18
- **Bidirectional Gated Fusion (BDGF)** mechanism for adaptive cross-modal fusion
- **Multi-Scale Generalized Mean (GeM) Pooling** for enhanced feature representation

## 🚀 Key Features

- Achieves **86.19% accuracy** on NCEPU-GRSCD dataset
- Outperforms ResNet-50 by **5.29%** in accuracy
- Robust performance on both RGB images and optical flow inputs
- Handles missing temporal data gracefully via adaptive gating
- Includes pre-trained models and complete training/evaluation pipeline

## 📦 Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full dependencies

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/hyyy96/ST-DINO-Net.git
cd ST-DINO-Net

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in development mode
pip install -e .
```

## 📁 Dataset Preparation

### NCEPU-GRSCD Dataset

The dataset used in this study is available at:
https://github.com/hyyy96/NCEPU-GRSCD-Dataset

Expected directory structure:

text

```
data/
├── train/
│   ├── cumulus/
│   ├── altocumulus/
│   ├── cirrus/
│   ├── clear/
│   ├── stratocumulus/
│   ├── cumulonimbus/
│   └── mixed/
├── test/
   └── [same subfolders]/

```



### Optical Flow Data

Optical flow maps can be generated using the provided script or downloaded from the repository.

## 🏃 Quick Start

### Run a quick test

bash

```
python demo/quick_test.py
```



### Evaluate a pre-trained model

bash

```
python scripts/evaluate.py \
    --weights path/to/weights.pth \
    --rgb_root data/test \
    --flow_root data/flow/test
```



### Train a new model

bash

```
python scripts/train.py \
    --train_rgb data/train \
    --train_flow data/flow/train \
    --val_rgb data/val \
    --val_flow data/flow/val \
    --epochs 100 \
    --batch_size 32
```



## 📊 Model Architecture

text

```
┌─────────────────┐    ┌─────────────────┐
│   RGB Image     │    │  Optical Flow   │
└────────┬────────┘    └────────┬────────┘
         │                      │
┌────────▼────────┐    ┌────────▼────────┐
│  DINO ViT       │    │   ResNet-18     │
│  (Frozen/FT)    │    │                 │
└────────┬────────┘    └────────┬────────┘
         │                      │
┌────────▼──────────────────────▼────────┐
│    Bidirectional Gated Fusion (BDGF)   │
│    ┌─────────────┐    ┌─────────────┐  │
│    │  S2M Attn   │    │  M2S Attn   │  │
│    └──────┬──────┘    └──────┬──────┘  │
│           └────────┬─────────┘          │
│              ┌────▼────┐                │
│              │ Gating  │                │
│              └────┬────┘                │
└───────────────────┼────────────────────┘
                    │
┌───────────────────▼────────────────────┐
│     Multi-Scale GeM Pooling             │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │GeM1 │ │GeM3 │ │GeM6 │ │Max  │       │
│  └─────┘ └─────┘ └─────┘ └──┬──┘       │
│         ┌────────────────────┘          │
│    ┌────▼────┐                          │
│    │ Concat  │                          │
│    └────┬────┘                          │
└─────────┼──────────────────────────────┘
          │
┌─────────▼─────────┐
│    Classifier     │
│   (MLP + Softmax) │
└───────────────────┘
```



## 📈 Results

| Method                 | Accuracy   | Precision  | Recall     | F1-Score   |
| :--------------------- | :--------- | :--------- | :--------- | :--------- |
| ResNet-50              | 80.90%     | 81.27%     | 80.90%     | 79.66%     |
| EfficientNet           | 77.10%     | 77.77%     | 77.10%     | 74.26%     |
| MobileNet              | 72.14%     | 79.87%     | 72.14%     | 68.69%     |
| ViT                    | 72.52%     | 76.62%     | 72.52%     | 70.40%     |
| **ST-DINO-Net (Ours)** | **86.19%** | **86.80%** | **86.19%** | **86.01%** |

## 📝 Citation

If you find this code useful for your research, please cite our paper:

bibtex

```
@article{quan2026spatiotemporal,
  title={A Spatiotemporal Method for Ground-Based Cloud Classification by Fusing Static DINO Texture and Dynamic Optical Flow},
  author={Quan, Hongyang and Zou, Lianglin and Xu, Xiaoshi and He, Jinguo and Zhang, Shuai and Song, Jifeng},
  journal={Computers \& Geosciences},
  year={2026},
  publisher={Elsevier}
}
```



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://license/) file for details.

## 📧 Contact

For questions or issues, please contact:

- Jifeng Song: songjifeng@ncepu.edu.cn