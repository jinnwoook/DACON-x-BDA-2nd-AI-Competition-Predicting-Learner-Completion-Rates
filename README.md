# BDA Competition - Final Submission

## submission_10files_7agree.csv (SOTA: 0.44)

10개 앙상블 파일을 결합하여 7개 이상 동의 시 1 예측

## 폴더 구조

```
final_submit/
├── main.ipynb              # 앙상블 실행 노트북
├── README.md
├── requirements.txt
├── models/                 # 10개 모델 예측 파일
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
├── outputs/                # 최종 제출 파일
│   └── submission_10files_7agree.csv
└── data/                   # 원본 데이터
    ├── train.csv
    └── test.csv
```

## 10개 파일 구성

| # | 파일명 | 설명 | Positives |
|---|--------|------|-----------|
| 1 | meta_vote_both | 5models_4agree AND enhanced_3agree | 470 |
| 2 | enhanced_3agree | Enhanced 모델 3개 이상 동의 | 617 |
| 3 | simcse_bert_4agree | SimCSE BERT 기반 | 491 |
| 4 | mega_ensemble_3agree | 메가 앙상블 | 483 |
| 5 | top3_2agree | Top 3 모델 2개 이상 동의 | 483 |
| 6 | prob_avg_035 | 확률 평균 threshold 0.35 | 468 |
| 7 | 10models_7agree | 10개 모델 7개 이상 동의 | 511 |
| 8 | 5models_4agree | 5개 모델 4개 이상 동의 | 476 |
| 9 | bert_data | BERT 단일 모델 | 676 |
| 10 | 5models_3agree | 5개 모델 3개 이상 동의 | 617 |

## 실행 방법

```bash
# Jupyter Notebook 실행
jupyter notebook main.ipynb
```

또는 Python 스크립트로 실행:

```python
import numpy as np
import pandas as pd
from pathlib import Path

# 10개 파일 로드
MODELS_DIR = Path("models")
file_names = [
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
test_ids = None
for f in file_names:
    df = pd.read_csv(MODELS_DIR / f)
    preds[f] = df['completed'].values
    if test_ids is None:
        test_ids = df['ID'].values

# 7개 이상 동의 시 1
vote_sum = sum(preds.values())
final_pred = (vote_sum >= 7).astype(int)

# 저장
submission = pd.DataFrame({'ID': test_ids, 'completed': final_pred})
submission.to_csv("outputs/submission_10files_7agree.csv", index=False)
print(f"Positives: {final_pred.sum()}")  # 479
```

## 결과

- **Total samples**: 814
- **Positives**: 479 (58.8%)
- **Score**: 0.44 (F1-Score)
