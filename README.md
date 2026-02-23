# DACON x BDA 2nd AI Competition
## Predicting Learner Course Completion Rates

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**Final F1 Score: 0.4508**

</div>

---

## Overview

This repository contains the final submission for the DACON x BDA 2nd AI Competition. The goal is to predict whether a learner will complete an online course based on their learning behavior data and course metadata.

Our approach leverages **BERT-based text classification** combined with **ensemble learning** to achieve robust predictions through multi-model consensus voting.

---

## Pipeline Architecture

<div align="center">
<img src="assets/pipeline_architecture.png" alt="Pipeline Architecture" width="900"/>
</div>

---

## Data Analysis & Preprocessing

### Dataset Overview

| Dataset | Samples | Features | Target Variable |
|---------|---------|----------|-----------------|
| Train | 3,256 | Text + Metadata | `completed` (0/1) |
| Test | 814 | Text + Metadata | - |

### Key Data Characteristics

1. **Text Features**
   - Course descriptions and titles in Korean
   - Learner feedback and engagement notes
   - Variable text lengths requiring dynamic tokenization

2. **Class Distribution**
   - Binary classification task with moderate class imbalance
   - Positive class (completed): ~55-60%
   - Negative class (not completed): ~40-45%

3. **Preprocessing Pipeline**
   ```
   Raw Text → Cleaning → Tokenization → BERT Encoding → Model Input
   ```
   - **Text Cleaning**: Remove special characters, normalize whitespace
   - **Tokenization**: BERT WordPiece tokenization (max 512 tokens)
   - **Encoding**: [CLS] + tokens + [SEP] format for BERT input

---

## Why BERT?

<div align="center">
<img src="assets/bert_architecture.png" alt="BERT Architecture" width="800"/>
</div>

### Rationale for BERT-based Approach

| Aspect | Traditional ML | BERT-based |
|--------|---------------|------------|
| **Text Understanding** | Bag-of-words, TF-IDF | Contextual embeddings |
| **Korean Language** | Limited morphological analysis | Pre-trained on Korean corpus |
| **Semantic Capture** | Surface-level patterns | Deep semantic relationships |
| **Transfer Learning** | Not applicable | Leverages pre-trained knowledge |

### Why `klue/bert-base`?

1. **Korean Language Optimization**
   - Pre-trained on 62GB of Korean text corpus
   - Native understanding of Korean grammar and semantics
   - Handles Korean-specific tokenization (subword units)

2. **Architecture Specifications**
   - **Layers**: 12 Transformer encoder blocks
   - **Hidden Size**: 768 dimensions
   - **Attention Heads**: 12 multi-head attention
   - **Parameters**: ~110M parameters

3. **Performance Benefits**
   - Captures long-range dependencies in course descriptions
   - Understands context of learner behavior descriptions
   - Robust to vocabulary variations and typos

---

## Ensemble Strategy

<div align="center">
<img src="assets/ensemble_voting.png" alt="Ensemble Voting" width="900"/>
</div>

### 10-Model Hard Voting Ensemble

We employ a **hard voting strategy** with a **7/10 agreement threshold** for final predictions.

| # | Model File | Description | Positives |
|---|------------|-------------|-----------|
| 1 | `meta_vote_both` | Meta-learner combining 5models AND enhanced | 470 |
| 2 | `enhanced_3agree` | Enhanced BERT with 3-model agreement | 617 |
| 3 | `simcse_bert_4agree` | SimCSE BERT with 4-model agreement | 491 |
| 4 | `mega_ensemble_3agree` | Mega ensemble with 3-model agreement | 483 |
| 5 | `top3_2agree` | Top 3 models with 2-model agreement | 483 |
| 6 | `prob_avg_035` | Probability averaging (threshold 0.35) | 468 |
| 7 | `10models_7agree` | 10-model ensemble with 7-agreement | 511 |
| 8 | `5models_4agree` | 5-model ensemble with 4-agreement | 476 |
| 9 | `bert_data` | Single BERT classifier | 676 |
| 10 | `5models_3agree` | 5-model ensemble with 3-agreement | 617 |

### Voting Logic

```python
# Hard voting with 7/10 threshold
vote_sum = sum(model_predictions)  # Sum of 10 binary predictions
final_prediction = 1 if vote_sum >= 7 else 0
```

**Final Result**: 479 Positives (58.85%)

---

## Project Structure

```
final_submit/
├── main.ipynb              # Main ensemble notebook
├── generate_diagrams.py    # Architecture diagram generator
├── README.md
├── requirements.txt
├── assets/                 # Generated diagrams
│   ├── pipeline_architecture.png
│   ├── bert_architecture.png
│   └── ensemble_voting.png
├── models/                 # 10 pre-trained model predictions
│   ├── submission_meta_vote_both.csv
│   ├── submission_enhanced_3agree.csv
│   ├── submission_simcse_bert_4agree.csv
│   ├── submission_mega_ensemble_3agree.csv
│   ├── submission_top3_2agree.csv
│   ├── submission_prob_avg_035.csv
│   ├── submission_10models_7agree.csv
│   ├── submission_5models_4agree.csv
│   ├── submission_bert_data.csv
│   └── submission_5models_3agree.csv
├── outputs/                # Final submission
│   └── submission_10files_7agree.csv
└── data/                   # Dataset (not included)
    ├── train.csv
    └── test.csv
```

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Generate Final Submission

```bash
jupyter notebook main.ipynb
# or run directly:
python -c "
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path('models')
files = [
    'submission_meta_vote_both.csv',
    'submission_enhanced_3agree.csv',
    'submission_simcse_bert_4agree.csv',
    'submission_mega_ensemble_3agree.csv',
    'submission_top3_2agree.csv',
    'submission_prob_avg_035.csv',
    'submission_10models_7agree.csv',
    'submission_5models_4agree.csv',
    'submission_bert_data.csv',
    'submission_5models_3agree.csv',
]

preds = {}
for f in files:
    df = pd.read_csv(MODELS_DIR / f)
    preds[f] = df['completed'].values
    test_ids = df['ID'].values

vote_sum = sum(preds.values())
final_pred = (vote_sum >= 7).astype(int)

submission = pd.DataFrame({'ID': test_ids, 'completed': final_pred})
submission.to_csv('outputs/submission_10files_7agree.csv', index=False)
print(f'Positives: {final_pred.sum()} ({final_pred.mean()*100:.2f}%)')
"
```

---

## Results

| Metric | Value |
|--------|-------|
| **F1 Score** | **0.4508** |
| Total Samples | 814 |
| Predicted Positives | 479 (58.85%) |
| Ensemble Models | 10 |
| Voting Threshold | 7/10 |

---

## Technical Stack

- **Deep Learning**: PyTorch 2.0+
- **NLP**: Hugging Face Transformers
- **Pre-trained Model**: klue/bert-base
- **Ensemble**: Hard voting with threshold
- **Data Processing**: Pandas, NumPy

---

## License

This project is for educational and competition purposes.

---

<div align="center">

**DACON x BDA 2nd AI Competition**

Made with PyTorch and Hugging Face Transformers

</div>
