"""
5 Models Ensemble Inference (기존 모델)
- 5개 모델의 예측을 결합하여 제출 파일 생성
- 모델: Model1, Model2, Model3, Model5, Model6
- 보팅 방식: 4개 이상 동의 시 수료(1) 예측
- Output: submission_5models_4agree.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent  # project root
OUTPUT_DIR = BASE_DIR / "outputs"

# ============================================================================
# Load Predictions
# ============================================================================
print("="*70)
print("5 Models Ensemble Inference (Model 1,2,3,5,6)")
print("="*70)

# 5개 모델 정의 (기존 모델)
models = {
    'Model1_BERT': 'submission_model1_bert.csv',
    'Model2_KoELECTRA': 'submission_model2_koelectra.csv',
    'Model3_KLUE': 'submission_model3_klue.csv',
    'Model5_XGBoost': 'submission_model5_xgboost.csv',
    'Model6_CatBoost': 'submission_model6_catboost.csv',
}

predictions = {}
for name, filename in models.items():
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        predictions[name] = df['completed'].values
        print(f"Loaded: {name} - {df['completed'].sum()} positives ({df['completed'].mean()*100:.1f}%)")
    else:
        print(f"WARNING: {filename} not found!")

# ID 가져오기
sample_df = pd.read_csv(OUTPUT_DIR / list(models.values())[0])
test_ids = sample_df['ID'].values

print(f"\nTotal test samples: {len(test_ids)}")
print(f"Models loaded: {len(predictions)}")

# ============================================================================
# Ensemble Voting
# ============================================================================
print("\n" + "="*70)
print("Ensemble Voting (5 Models: 1,2,3,5,6)")
print("="*70)

# 투표 DataFrame 생성
vote_df = pd.DataFrame({'ID': test_ids})
for name, preds in predictions.items():
    vote_df[name] = preds

# 투표 합계
model_cols = list(predictions.keys())
vote_df['vote_sum'] = vote_df[model_cols].sum(axis=1)

# 다양한 보팅 기준
print("\n보팅 결과:")
for threshold in [2, 3, 4, 5]:
    vote_df[f'vote_{threshold}'] = (vote_df['vote_sum'] >= threshold).astype(int)
    pos_count = vote_df[f'vote_{threshold}'].sum()
    pos_rate = vote_df[f'vote_{threshold}'].mean() * 100
    print(f"  {threshold}개 이상 동의: {pos_count}개 ({pos_rate:.1f}%)")

# ============================================================================
# Model Agreement Analysis
# ============================================================================
print("\n" + "="*70)
print("Model Agreement Analysis")
print("="*70)

print("\n투표 분포:")
for vote in range(6):
    count = (vote_df['vote_sum'] == vote).sum()
    pct = count / len(vote_df) * 100
    print(f"  {vote}개 모델 동의: {count}개 ({pct:.1f}%)")

# ============================================================================
# Create Final Submission (4개 이상 동의)
# ============================================================================
print("\n" + "="*70)
print("Creating Final Submissions")
print("="*70)

# 3개 이상 동의
submission_3agree = pd.DataFrame({
    'ID': test_ids,
    'completed': vote_df['vote_3'].values
})
submission_3agree.to_csv(OUTPUT_DIR / "submission_5models_3agree.csv", index=False)
print(f"\nsubmission_5models_3agree.csv saved!")
print(f"  Positive: {submission_3agree['completed'].sum()} ({submission_3agree['completed'].mean()*100:.1f}%)")

# 4개 이상 동의
submission_4agree = pd.DataFrame({
    'ID': test_ids,
    'completed': vote_df['vote_4'].values
})
submission_4agree.to_csv(OUTPUT_DIR / "submission_5models_4agree.csv", index=False)
print(f"\n* submission_5models_4agree.csv saved!")
print(f"  Positive: {submission_4agree['completed'].sum()} ({submission_4agree['completed'].mean()*100:.1f}%)")

# 5개 모두 동의
submission_5agree = pd.DataFrame({
    'ID': test_ids,
    'completed': vote_df['vote_5'].values
})
submission_5agree.to_csv(OUTPUT_DIR / "submission_5models_5agree.csv", index=False)
print(f"\nsubmission_5models_5agree.csv saved!")
print(f"  Positive: {submission_5agree['completed'].sum()} ({submission_5agree['completed'].mean()*100:.1f}%)")

print("\n" + "="*70)
print("Done!")
print("="*70)
