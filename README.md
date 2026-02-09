# DACON x BDA 2nd AI Competition - Predicting Learner Completion Rates

BDA 수료 예측 대회 SOTA 솔루션 (LB Score: 0.44)

## Overview

학습자의 수료 여부를 예측하는 AI 알고리즘 개발

### Key Approach: Meta Vote Both

```
meta_vote_both = 5models_4agree AND enhanced_3agree
```

- **5models_4agree**: 기존 5개 모델 중 4개 이상 동의
- **enhanced_3agree**: Enhanced 5개 모델 중 3개 이상 동의
- **meta_vote_both**: 위 두 조건 모두 만족해야 수료(1) 예측

---

## Project Structure

```
.
├── README.md
├── run_all.sh                          # Full pipeline script
├── requirements.txt
│
├── src/
│   ├── models/
│   │   ├── text/                       # Text-based models
│   │   │   ├── model1_bert_data.py     # KoELECTRA-NSMC (bert_train_data)
│   │   │   ├── model2_koelectra_detailed.py  # KoELECTRA-NSMC (train_detailed)
│   │   │   └── model3_klue_sentiment.py      # KLUE-BERT Sentiment
│   │   │
│   │   └── tabular/                    # Tabular models
│   │       ├── model5_xgboost_advanced.py
│   │       ├── model5_xgboost_enhanced.py
│   │       ├── model6_catboost_advanced.py
│   │       └── model6_catboost_enhanced.py
│   │
│   └── ensemble/                       # Ensemble scripts
│       ├── ensemble_5models.py         # 5models_4agree
│       ├── ensemble_enhanced.py        # enhanced_3agree
│       └── create_meta_vote_both.py    # Final SOTA
│
├── data/                               # Place your data here
│   ├── train.csv
│   ├── test.csv
│   ├── bert_train_data.csv
│   ├── bert_test_data.csv
│   ├── train_detailed.csv
│   └── test_detailed.csv
│
└── outputs/                            # Model predictions
    └── submission_meta_vote_both.csv   # Final submission
```

---

## Models

### Text Models (3)

| Model | Script | Pre-trained | Data |
|-------|--------|-------------|------|
| Model 1 | `model1_bert_data.py` | `koelectra-base-finetuned-nsmc` | bert_train_data.csv |
| Model 2 | `model2_koelectra_detailed.py` | `koelectra-base-finetuned-nsmc` | train_detailed.csv |
| Model 3 | `model3_klue_sentiment.py` | `klue-bert-base-sentiment` | train_detailed.csv |

### Tabular Models (4)

| Model | Script | Algorithm | Features |
|-------|--------|-----------|----------|
| Model 5 | `model5_xgboost_advanced.py` | XGBoost | Basic FE |
| Model 5 Enhanced | `model5_xgboost_enhanced.py` | XGBoost | Advanced FE |
| Model 6 | `model6_catboost_advanced.py` | CatBoost | Basic FE + Target Encoding |
| Model 6 Enhanced | `model6_catboost_enhanced.py` | CatBoost | Advanced FE + Target Encoding |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your data files in the `data/` directory.

### 3. Run Pipeline

```bash
chmod +x run_all.sh
./run_all.sh
```

Or run individual models:

```bash
# Text models
python src/models/text/model1_bert_data.py
python src/models/text/model2_koelectra_detailed.py
python src/models/text/model3_klue_sentiment.py

# Tabular models
python src/models/tabular/model5_xgboost_advanced.py
python src/models/tabular/model6_catboost_advanced.py
python src/models/tabular/model5_xgboost_enhanced.py
python src/models/tabular/model6_catboost_enhanced.py

# Ensemble
python src/ensemble/ensemble_5models.py
python src/ensemble/ensemble_enhanced.py
python src/ensemble/create_meta_vote_both.py
```

---

## Results

| Submission | Description | Positive Rate |
|------------|-------------|---------------|
| `5models_4agree` | 5 models, 4+ agree | ~58% |
| `enhanced_3agree` | Enhanced 5 models, 3+ agree | ~76% |
| **`meta_vote_both`** | **Both conditions (SOTA)** | **~58%** |

---

## Why Meta Vote Both Works

1. **Double Validation**: Both original and enhanced models must agree
2. **Minimizes False Positives**: Conservative prediction strategy
3. **Diversity**: Different feature engineering approaches

```
5models_4agree (476) AND enhanced_3agree (617) = meta_vote_both (470)
```
