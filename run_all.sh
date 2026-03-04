#!/bin/bash
# ============================================================================
# DACON x BDA - Learner Completion Prediction Pipeline
# ============================================================================

set -e  # Exit on error

cd "$(dirname "$0")"

# Create outputs directory
mkdir -p outputs

echo "============================================================================"
echo "DACON x BDA - Learner Completion Prediction Pipeline"
echo "============================================================================"

# Step 0: Preprocessing
echo ""
echo "[0/8] Preprocessing - Generate BERT text data"
python src/preprocessing/create_bert_data.py

echo ""
echo "[0/8] Preprocessing - Generate detailed text data (for model2)"
python src/preprocessing/create_detailed_data.py

# Step 1: Text Models
echo ""
echo "[1/8] Training Model 1 - BERT (KoELECTRA-NSMC)"
python src/models/text/model1_bert_data.py

echo ""
echo "[2/8] Training Model 2 - KoELECTRA (train_detailed)"
python src/models/text/model2_koelectra_detailed.py

echo ""
echo "[3/8] Training Model 3 - KLUE Sentiment"
python src/models/text/model3_klue_sentiment.py

# Step 2: Tabular Models (Original)
echo ""
echo "[4/8] Training Model 5 - XGBoost Advanced"
python src/models/tabular/model5_xgboost_advanced.py

echo ""
echo "[5/8] Training Model 6 - CatBoost Advanced"
python src/models/tabular/model6_catboost_advanced.py

# Step 3: Tabular Models (Enhanced)
echo ""
echo "[6/8] Training Model 5 Enhanced - XGBoost"
python src/models/tabular/model5_xgboost_enhanced.py

echo ""
echo "[7/8] Training Model 6 Enhanced - CatBoost"
python src/models/tabular/model6_catboost_enhanced.py

# Step 4: Ensemble
echo ""
echo "[Ensemble 1/3] Creating 5models_4agree..."
python src/ensemble/ensemble_5models.py

echo ""
echo "[Ensemble 2/3] Creating enhanced_3agree..."
python src/ensemble/ensemble_enhanced.py

echo ""
echo "[Ensemble 3/3] Creating meta_vote_both (SOTA)..."
python src/ensemble/create_meta_vote_both.py

echo ""
echo "============================================================================"
echo "Pipeline Complete!"
echo "Final submission: outputs/submission_meta_vote_both.csv"
echo "============================================================================"
