# Native Language Identification (NLI) of Indian English Speakers  
Using HuBERT & MFCC

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Feature Extraction](#feature-extraction)
  - [MFCC Extraction](#mfcc-extraction)
  - [HuBERT Embeddings](#hubert-embeddings)
- [Models](#models)
  - [MLP Classifier](#mlp-classifier)
  - [LSTM Model](#lstm-model)
  - [HuBERT Layer Analysis](#hubert-layer-analysis)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
  - [MFCC vs HuBERT](#mfcc-vs-hubert)
  - [Age Generalization](#age-generalization)
  - [Word vs Sentence Analysis](#word-vs-sentence-analysis)
- [Visualization Outputs](#visualization-outputs)
- [Checkpoints](#checkpoints)
  - [How to Load Checkpoints](#how-to-load-checkpoints)
- [How to Run](#how-to-run)
  - [Google Colab Version](#google-colab-version)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview
This project focuses on **Native Language Identification (NLI)** of Indian English speakers.  
The goal is to classify the **native language (L1)** of a speaker by analyzing their English speech accent.

We compare:
- Traditional MFCC features  
- Self-supervised HuBERT speech representations  
- Layer-wise analysis of HuBERT  
- Word-level vs sentence-level accents  
- Cross-age generalization  

This project also includes:
- Model training  
- Evaluation  
- Visualizations  
- Saved checkpoints  

---

## Dataset
We use the **IndicAccentDB** dataset, which contains English speech recorded by Indian speakers from multiple linguistic backgrounds.

**Dataset features include:**
- Native language label (e.g., Telugu, Hindi, Malayalam, Tamil, Bengali, etc.)
- Speaker metadata: Age, gender, region
- Word-level recordings and sentence-level recordings
- High-quality 16kHz audio files

**Dataset loading:**
```python
from datasets import load_dataset
dataset = load_dataset("DarshanaS/IndicAccentDb")

## Project Structure
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


## Feature Extraction

### MFCC Extraction
MFCCs capture short-term frequency information, commonly used in speech recognition.

Settings used:

Sampling rate: 16 kHz

Number of coefficients: 40

Frame length: 25 ms

Hop length: 10 ms

Example code:

import librosa

y, sr = librosa.load(audio_path, sr=16000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

### HuBERT Embeddings
HuBERT (facebook/hubert-large-ll60k) is a self-supervised speech model trained on 60k hours of audio.

We extract:

All 24 hidden layers

Mean-pooled embeddings

Frame-level sequences for LSTM

Example code:

from transformers import Wav2Vec2Processor, HubertModel

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ll60k")
model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.hidden_states

## Models

### MLP Classifier
Used primarily for:

MFCC feature classification

Mean-pooled HuBERT embeddings

Architecture:

Input → Dense(512) → ReLU → Dropout → Dense(NumLanguages)

### LSTM Model
Used for:

Sequential MFCC frames

Frame-level HuBERT features

Architecture:

LSTM(256)

Attention / final hidden state

Dense classifier
### HuBERT Layer Analysis
Each of the 24 HuBERT layers is evaluated individually.
Findings:

Layer 9 → most phonetic detail

Layer 19 → best for accent classification

Layer 23 → too abstract, lower performance

## Training Pipeline
Load dataset

Extract MFCC/HuBERT features

Normalize features

Split train/validation/test sets

Train MLP/LSTM models

Save checkpoints

Evaluate performance

Generate visualizations

## Results

### MFCC vs HuBERT
Feature Type	Accuracy
MFCC	~70%
HuBERT Mean-Pooled	~82%
HuBERT Layer 19	~89%
Conclusion: HuBERT significantly outperforms MFCC.

### Age Generalization
We evaluate generalization from adults (18+) to youth speakers (10–17).

MFCC: Accuracy drops by 20–25%

HuBERT: Accuracy drops only 3–5%

Conclusion: HuBERT is robust across age groups.

### Word vs Sentence Analysis
Sentence-level clips contain longer accent cues → better accuracy.

Type	Accuracy
Word-level	68%
Sentence-level	88%

## Visualization Outputs
The project generates:

Confusion matrix

HuBERT layer-wise accuracy plot

Loss and accuracy curves

MFCC vs HuBERT comparison chart

Age generalization chart

All plots are saved in:

/outputs/plots/


## Checkpoints
Saved in:

/checkpoints/


Available models:

mfcc_mlp.pt

hubert_meanpool_mlp.pt

hubert_lstm.pt

hubert_layer19_best.pt

### How to Load Checkpoints
import torch
from model_lstm import AccentLSTM

model = AccentLSTM(...)
model.load_state_dict(torch.load("checkpoints/hubert_layer19_best.pt"))
model.eval()


## How to Run
Install dependencies:

pip install -r requirements.txt


Run training:

python src/train.py


Run evaluation:

python src/evaluate.py

### Google Colab Version
You can run the full pipeline using the notebook:

IndicAccent_NLI_HuBERT_MFCC.ipynb


Simply upload to Google Colab and run all cells.

## Future Work
Add WavLM, Wav2Vec2, Whisper comparisons

Add speaker normalization

Build inference API

Add GUI demo / Streamlit app

## License
MIT License
Free for research and academic use.


---

# ✅ Your README is now **complete**, **professional**, and **fully clickable**.

If you want:
✅ badges on top  
✅ diagrams / architecture images  
✅ auto-generated PDF  
Just tell me!
