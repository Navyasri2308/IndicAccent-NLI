# IndicAccent Classification using HuBERT and MFCC

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Indian English Accent Classification using Deep Learning**

</div>

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start) 
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Installation

<a name="installation"></a>

```bash
# Clone repository
git clone https://github.com/yourusername/IndicAccent-NLI.git
cd IndicAccent-NLI

# Install dependencies
pip install -r requirements.txt

Requirements:

Python 3.8+
PyTorch 2.0+
Transformers
Librosa
Scikit-learn

## Quick Start
<a name="quick-start"></a>

python
# Run main notebook
jupyter notebook IndicAccent_NLI_HuBERT_MFCC.ipynb

# Or use directly
from indicaccent import AccentClassifier
classifier = AccentClassifier('hubert')
prediction = classifier.predict('audio.wav')

## Models
<a name="models"></a>

1. HuBERT Classifier
Pre-trained HuBERT embeddings

Transformer-based features

High accuracy

2. MFCC Classifier
13 MFCC coefficients

Delta and delta-delta features

Fast inference

3. Ensemble
Combines HuBERT + MFCC

Best performance

## Dataset
<a name="dataset"></a>

IndicAccentDB - Indian English accents dataset

Multiple regional accents

WAV audio format

2,300+ samples

Karnataka, Kerala regions

## Results
<a name="results"></a>

Model	Accuracy	F1-Score
HuBERT	89.5%	89.3%
MFCC	81.7%	81.4%
Ensemble	91.2%	91.0%

##Usage
<a name="usage"></a>

Training
python
from indicaccent.trainer import Trainer

trainer = Trainer(model='hubert')
trainer.fit(train_data, val_data, epochs=50)

Prediction
python
classifier = AccentClassifier()
result = classifier.predict('path/to/audio.wav')
print(f"Accent: {result}")
Feature Extraction
python
from indicaccent.features import extract_features

hubert_features = extract_hubert_features(audio)
mfcc_features = extract_mfcc_features(audio)

## Citation
<a name="citation"></a>

bibtex
@misc{indicaccent2024,
  title={IndicAccent Classification using HuBERT and MFCC},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/IndicAccent-NLI}
}

##License
<a name="license"></a>

MIT License - see LICENSE file for details.
