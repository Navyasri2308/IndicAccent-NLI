# IndicAccent NLI: Indian Accent Classification using HuBERT and MFCC

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**A comprehensive framework for Indian accent classification using deep learning approaches**

[Installation](#installation) • [Quick Start](#quick-start) • [Models](#models) • [Results](#results) • [Citation](#citation)

</div>

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Results](#results)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a Natural Language Inference (NLI) system for classifying Indian English accents using state-of-the-art deep learning techniques. We combine transformer-based speech representations (HuBERT) with traditional audio features (MFCC) to achieve robust accent classification across multiple Indian regions.

### Key Features
- 🎵 **Multi-modal feature extraction** (HuBERT + MFCC)
- 🏗️ **Modular architecture** for easy experimentation
- 📊 **Comprehensive evaluation** metrics and visualizations
- 🔄 **Reproducible research** with detailed documentation
- ⚡ **GPU-accelerated** training and inference

## ⚡ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Install from Source
```bash
# Clone repository
git clone https://github.com/yourusername/IndicAccent-NLI.git
cd IndicAccent-NLI

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
