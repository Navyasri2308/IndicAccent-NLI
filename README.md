# IndicAccent Classification using HuBERT and MFCC

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Indian English Accent Classification using Deep Learning**

</div>

## Table of Contents

* [Installation](#installation)
* [Quick Start](#quick-start)
* [Models](#models)
* [Dataset](#dataset)
* [Results](#results)
* [Usage](#usage)
* [Citation](#citation)
* [License](#license)

## Installation

<a name="installation"></a>

```bash
# Clone repository
git clone https://github.com/yourusername/IndicAccent-NLI.git
cd IndicAccent-NLI

# Install dependencies
pip install -r requirements.txt
```

---

# Native Language Identification (NLI) of Indian English Speakers

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![HuBERT](https://img.shields.io/badge/HuBERT-24--Layer-yellow)
![MFCC](https://img.shields.io/badge/MFCC-40coeff-red)

</div>
Using HuBERT & MFCC

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Feature Extraction](#feature-extraction)

  * [MFCC Extraction](#mfcc-extraction)
  * [HuBERT Embeddings](#hubert-embeddings)
* [Models](#models)

  * [MLP Classifier](#mlp-classifier)
  * [LSTM Model](#lstm-model)
  * [HuBERT Layer Analysis](#hubert-layer-analysis)
* [Training Pipeline](#training-pipeline)
* [Results](#results)

  * [MFCC vs HuBERT](#mfcc-vs-hubert)
  * [Age Generalization](#age-generalization)
  * [Word vs Sentence Analysis](#word-vs-sentence-analysis)
* [Visualization Outputs](#visualization-outputs)
* [Checkpoints](#checkpoints)

  * [How to Load Checkpoints](#how-to-load-checkpoints)
* [How to Run](#how-to-run)

  * [Google Colab Version](#google-colab-version)
* [Future Work](#future-work)
* [License](#license)

---

## Project Overview

This project focuses on **Native Language Identification (NLI)** of Indian English speakers.
The goal is to classify the **native language (L1)** of a speaker by analyzing their English speech accent.

We compare:

* Traditional MFCC features
* Self-supervised HuBERT speech representations
* Layer-wise analysis of HuBERT
* Word-level vs sentence-level accents
* Cross-age generalization

This project also includes:

* Model training
* Evaluation
* Visualizations
* Saved checkpoints

---

## Dataset

We use the **IndicAccentDB** dataset, which contains English speech recorded by Indian speakers from multiple linguistic backgrounds.

**Dataset features include:**

* Native language label (e.g., Telugu, Hindi, Malayalam, Tamil, Bengali, etc.)
* Speaker metadata: Age, gender, region
* Word-level recordings and sentence-level recordings
* High-quality 16kHz audio files

**Dataset loading:**

```python
from datasets import load_dataset
dataset = load_dataset("DarshanaS/IndicAccentDb")
```

## Project Structure

```
📁 NLI_Project
│── checkpoints/
│── notebooks/
│    └── IndicAccent_NLI_HuBERT_MFCC.ipynb
│── src/
│    ├── mfcc_extraction.py
│    ├── hubert_features.py
│    ├── model_mlp.py
│    ├── model_lstm.py
│    ├── train.py
│    ├── evaluate.py
│── requirements.txt
│── README.md
```

## Feature Extraction

### MFCC Extraction

MFCCs capture short-term frequency information, commonly used in speech recognition.

Settings used:

* Sampling rate: 16 kHz
* Number of coefficients: 40
* Frame length: 25 ms
* Hop length: 10 ms

Example code:

```python
import librosa

y, sr = librosa.load(audio_path, sr=16000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
```

### HuBERT Embeddings

HuBERT (facebook/hubert-large-ll60k) is a self-supervised speech model trained on 60k hours of audio.

We extract:

* All 24 hidden layers
* Mean-pooled embeddings
* Frame-level sequences for LSTM

Example code:

```python
from transformers import Wav2Vec2Processor, HubertModel

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ll60k")
model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.hidden_states
```

## Models

### MLP Classifier

Used primarily for:

* MFCC feature classification
* Mean-pooled HuBERT embeddings

Architecture:

```
Input → Dense(512) → ReLU → Dropout → Dense(NumLanguages)
```

### LSTM Model

Used for:

* Sequential MFCC frames
* Frame-level HuBERT features

Architecture:

```
LSTM(256)
Attention / final hidden state
Dense classifier
```

### HuBERT Layer Analysis

Each of the 24 HuBERT layers is evaluated individually.

Findings:

* Layer 9 → most phonetic detail
* Layer 19 → best for accent classification
* Layer 23 → too abstract, lower performance

## Training Pipeline

1. Load dataset
2. Extract MFCC/HuBERT features
3. Normalize features
4. Split train/validation/test sets
5. Train MLP/LSTM models
6. Save checkpoints
7. Evaluate performance
8. Generate visualizations

## Results

### MFCC vs HuBERT

```
Feature Type    Accuracy
MFCC            ~70%
HuBERT Mean     ~82%
HuBERT L19      ~89%
```

Conclusion: **HuBERT significantly outperforms MFCC.**

### Age Generalization

We evaluate adults (18+) → youth (10–17).

* MFCC: -20–25%
* HuBERT: -3–5%

### Word vs Sentence Analysis

Sentence-level provides best accent cues.

```
Word-level: 68%
Sentence-level: 88%
```

## Visualization Outputs

Generated:

* Confusion matrix
* HuBERT layer-wise accuracy plot
* Loss/accuracy curves
* MFCC vs HuBERT comparison
* Age generalization chart

## Checkpoints

Stored in:

```
/checkpoints/
```

Models:

* mfcc_mlp.pt
* hubert_meanpool_mlp.pt
* hubert_lstm.pt
* hubert_layer19_best.pt

### How to Load Checkpoints

```python
import torch
from model_lstm import AccentLSTM

model = AccentLSTM(...)
model.load_state_dict(torch.load("checkpoints/hubert_layer19_best.pt"))
model.eval()
```

## How to Run

Install:

```
pip install -r requirements.txt
```

Train:

```
python src/train.py
```

Evaluate:

```
python src/evaluate.py
```

### Google Colab Version

```
IndicAccent_NLI_HuBERT_MFCC.ipynb
```

Or open directly:
**Colab Link:** [https://colab.research.google.com/drive/1KrEQ4gFuzky7-14yAZAKGSBeXtcIcL9q?usp=sharing](https://colab.research.google.com/drive/1KrEQ4gFuzky7-14yAZAKGSBeXtcIcL9q?usp=sharing)

## Future Work

* Add WavLM, Whisper, Wav2Vec2 comparisons
* Speaker normalization
* Inference API
* GUI demo

## License

MIT License — free for research & academic use.
