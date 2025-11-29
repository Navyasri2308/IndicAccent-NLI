# Native Language Identification (NLI) of Indian English Speakers  
### Using HuBERT, MFCC, and Deep Learning

---

## 📌 Table of Contents
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

## 📌 Project Overview
Native Language Identification (NLI) aims to classify the **native language (L1)** of Indian English speakers based on accent patterns.  
This project compares **traditional MFCC acoustic features** with **self-supervised HuBERT representations**.

We evaluate:
- Accent cues captured by MFCCs  
- HuBERT layer-wise representation quality  
- Age-based generalization  
- Word-level vs sentence-level speech  

---

## 📌 Dataset
This project uses:

### **IndicAccentDB**
- Contains English audio by Indian speakers  
- Includes metadata (region, age, gender, etc.)  
- Balanced across 8+ native languages  

Loaded using HuggingFace:

```python

📌 Project Structure
📁 NLI_Project
│── dataset/
│── models/
│── checkpoints/
│── notebooks/
│── src/
│   ├── mfcc_extraction.py
│   ├── hubert_features.py
│   ├── train.py
│   ├── evaluate.py
│── README.md
│── requirements.txt

📌 Feature Extraction
### 🎧 MFCC Extraction

40-dimensional MFCCs

Frame size: 25 ms

Hop length: 10 ms

import librosa
mfcc = librosa.feature.mfcc(y, sr=16000, n_mfcc=40)

### 🤖 HuBERT Embeddings

Using:

facebook/hubert-large-ll60k


Extracting hidden states from all 24 layers.

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ll60k")
model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")

📌 Models
### 🔹 MLP Classifier

Used mainly for MFCC feature classification.

### 🔹 LSTM Model

Processes temporal sequences of MFCC/HuBERT features.

### 🔹 HuBERT Layer Analysis

We analyze which HuBERT hidden layer gives best performance.

Example:

Layer 9 → Best for phonetic info

Layer 19 → Best for accent classification

📌 Training Pipeline

Load dataset

Extract MFCC / HuBERT features

Train models

Evaluate

Generate plots

Save checkpoints

📌 Results
### ⭐ MFCC vs HuBERT
Feature	Accuracy
MFCC	~70%
HuBERT Layer 19	~89%
HuBERT Mean-pooled	~82%
### 👶 Age Generalization

Models trained on adults generalize well to 10–17 age group with HuBERT features.

### 🗣 Word vs Sentence Analysis

Sentence-level recordings give higher accuracy.

📌 Visualization Outputs

Generated automatically:

Confusion Matrix

Training Curves

Layer-wise HuBERT Accuracy Plot

Age-group Comparison

MFCC vs HuBERT Comparison

📌 Checkpoints

Saved in:

/checkpoints/


Includes:

mfcc_mlp.pt

hubert_lstm.pt

hubert_layer_19.pt

hubert_mean_pool.pt

### 🔧 How to Load Checkpoints
model.load_state_dict(torch.load("checkpoints/hubert_layer_19.pt"))
model.eval()

📌 How to Run

Install dependencies:

pip install -r requirements.txt


Run the main script:

python src/train.py

📌 Google Colab Version

➡️ Upload the notebook:
IndicAccent_NLI_HuBERT_MFCC.ipynb

Run all cells — code is fully compatible with Colab.

📌 Future Work

Add wav2vec2 and WavLM comparison

Add speaker diarization

Explore multilingual NLI

Improve dataset balancing

📌 License

MIT License
Free for research & educational use.


---

# ✅ YOUR README IS READY

If you want, I can also:

✔ convert it into a **PDF**  
✔ add **badges** (Python, HuggingFace, PyTorch, Colab)  
✔ add **images or diagrams**  
✔ add **your checkpoints section with links**

Just tell me **“generate PDF”** or **“add badges”** etc.
from datasets import load_dataset
dataset = load_dataset("DarshanaS/IndicAccentDb")
