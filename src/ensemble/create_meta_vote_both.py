"""
Create submission_meta_vote_both.csv
====================================
핵심 로직: 5models_4agree AND enhanced_3agree

- 5models_4agree: 기존 5개 모델 중 4개 이상 동의
- enhanced_3agree: Enhanced 5개 모델 중 3개 이상 동의
- meta_vote_both: 위 두 조건 모두 만족해야 수료(1) 예측

더 보수적인 예측으로 False Positive 최소화
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
# Load Two Ensemble Predictions
# ============================================================================
print("="*70)
print("Creating submission_meta_vote_both.csv")
print("="*70)

# 1. 5models_4agree 로드
path_5models = OUTPUT_DIR / "submission_5models_4agree.csv"
if not path_5models.exists():
    print("ERROR: submission_5models_4agree.csv not found!")
    print("Please run ensemble_5models.py first.")
    exit(1)

df_5models = pd.read_csv(path_5models)
pred_5models_4agree = df_5models['completed'].values
test_ids = df_5models['ID'].values
print(f"\n[1] submission_5models_4agree.csv")
print(f"    Positive: {pred_5models_4agree.sum()} ({pred_5models_4agree.mean()*100:.1f}%)")

# 2. enhanced_3agree 로드
path_enhanced = OUTPUT_DIR / "submission_enhanced_3agree.csv"
if not path_enhanced.exists():
    print("ERROR: submission_enhanced_3agree.csv not found!")
    print("Please run ensemble_enhanced.py first.")
    exit(1)

df_enhanced = pd.read_csv(path_enhanced)
pred_enhanced_3agree = df_enhanced['completed'].values
print(f"\n[2] submission_enhanced_3agree.csv")
print(f"    Positive: {pred_enhanced_3agree.sum()} ({pred_enhanced_3agree.mean()*100:.1f}%)")

# ============================================================================
# Create meta_vote_both (AND)
# ============================================================================
print("\n" + "="*70)
print("Creating meta_vote_both = 5models_4agree AND enhanced_3agree")
print("="*70)

# AND 연산: 둘 다 1이어야 1
pred_meta_vote_both = (pred_5models_4agree & pred_enhanced_3agree).astype(int)

print(f"\n[Result] submission_meta_vote_both.csv")
print(f"    Positive: {pred_meta_vote_both.sum()} ({pred_meta_vote_both.mean()*100:.1f}%)")

# 변경 분석
diff_count = (pred_5models_4agree != pred_meta_vote_both).sum()
removed = ((pred_5models_4agree == 1) & (pred_meta_vote_both == 0)).sum()
print(f"\n    5models_4agree 대비 변경: {diff_count}개")
print(f"    1 -> 0 변경 (제거됨): {removed}개")

# ============================================================================
# Save
# ============================================================================
submission = pd.DataFrame({
    'ID': test_ids,
    'completed': pred_meta_vote_both
})
output_path = OUTPUT_DIR / "submission_meta_vote_both.csv"
submission.to_csv(output_path, index=False)

print(f"\n" + "="*70)
print(f"Saved: {output_path}")
print("="*70)

# ============================================================================
# Also create meta_vote_any (OR) for comparison
# ============================================================================
print("\n" + "-"*70)
print("Creating meta_vote_any = 5models_4agree OR enhanced_3agree")
print("-"*70)

pred_meta_vote_any = (pred_5models_4agree | pred_enhanced_3agree).astype(int)
print(f"\n[Result] submission_meta_vote_any.csv")
print(f"    Positive: {pred_meta_vote_any.sum()} ({pred_meta_vote_any.mean()*100:.1f}%)")

submission_any = pd.DataFrame({
    'ID': test_ids,
    'completed': pred_meta_vote_any
})
submission_any.to_csv(OUTPUT_DIR / "submission_meta_vote_any.csv", index=False)

print(f"\nSaved: submission_meta_vote_any.csv")
print("="*70)
