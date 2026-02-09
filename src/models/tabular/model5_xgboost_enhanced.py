"""
Enhanced XGBoost Model with Additional Feature Engineering
- 기존 피쳐 + 새로운 피쳐 추가
- Output: submission_model5_xgboost_enhanced.csv, oof_model5_xgboost_enhanced.csv
"""

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    BASE_DIR = Path(__file__).parent.parent.parent.parent  # project root
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"

    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"

    N_SPLITS = 5
    SEED = 42

    TARGET = "completed"
    ID_COL = "ID"


cfg = Config()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(cfg.SEED)


def create_features(df):
    """향상된 피처 엔지니어링"""
    df = df.copy()

    # ==========================================
    # 기존 피쳐들
    # ==========================================

    # 1. re_registration: 재등록 여부
    df['is_re_registration'] = (df['re_registration'] == '예').astype(int)

    # 2. job: 대학생/직장인
    df['is_student'] = (df['job'] == '대학생').astype(int)
    df['is_worker'] = (df['job'] == '직장인').astype(int)

    # 3. major_type
    df['is_multiple_major'] = df['major type'].str.contains('복수', na=False).astype(int)

    # 4. class 수
    df['num_classes'] = 0
    for col in ['class1', 'class2', 'class3', 'class4']:
        if col in df.columns:
            df['num_classes'] += df[col].notna().astype(int)

    # 5. 이전 기수 참여
    prev_cols = ['previous_class_3', 'previous_class_4', 'previous_class_5',
                 'previous_class_6', 'previous_class_7', 'previous_class_8']
    df['num_prev_classes'] = 0
    for col in prev_cols:
        if col in df.columns:
            df['num_prev_classes'] += (~df[col].isin(['해당없음', '', np.nan]) & df[col].notna()).astype(int)

    # 6. 오프라인 참여 의향
    if 'hope_for_group' in df.columns:
        df['prefer_offline'] = df['hope_for_group'].str.contains('오프라인', na=False).astype(int)
        df['prefer_online'] = df['hope_for_group'].str.contains('온라인', na=False).astype(int)
        df['no_group'] = df['hope_for_group'].str.contains('아니요', na=False).astype(int)

    # 7. 자격증 보유
    if 'certificate_acquisition' in df.columns:
        df['has_certificate'] = (~df['certificate_acquisition'].isin(['없음', '', np.nan]) &
                                  df['certificate_acquisition'].notna()).astype(int)

    # 8. IT 전공
    if 'major1_1' in df.columns:
        df['is_it_major'] = df['major1_1'].str.contains('IT|컴퓨터', na=False).astype(int)

    # 9. 기존 회원
    if 'inflow_route' in df.columns:
        df['is_existing_member'] = df['inflow_route'].str.contains('기존 학회원', na=False).astype(int)

    # 10. 동기 분석
    if 'whyBDA' in df.columns:
        df['why_curriculum'] = df['whyBDA'].str.contains('커리큘럼|관리', na=False).astype(int)
        df['why_alone'] = df['whyBDA'].str.contains('혼자', na=False).astype(int)
        df['why_satisfied'] = df['whyBDA'].str.contains('만족', na=False).astype(int)
        df['why_benefit'] = df['whyBDA'].str.contains('혜택', na=False).astype(int)
        df['why_no_test'] = df['whyBDA'].str.contains('테스트|면접', na=False).astype(int)
        df['why_time'] = df['whyBDA'].str.contains('시간', na=False).astype(int)

    # 11. 목표 분석
    if 'what_to_gain' in df.columns:
        df['gain_project'] = df['what_to_gain'].str.contains('프로젝트', na=False).astype(int)
        df['gain_analysis'] = df['what_to_gain'].str.contains('분석', na=False).astype(int)
        df['gain_contest'] = df['what_to_gain'].str.contains('공모전', na=False).astype(int)

    # ==========================================
    # 새로운 피쳐들
    # ==========================================

    # 12. completed_semester - 이수 학기 수 (숫자형)
    if 'completed_semester' in df.columns:
        df['completed_semester_num'] = pd.to_numeric(df['completed_semester'], errors='coerce').fillna(0)
        # 고학년 여부 (6학기 이상)
        df['is_senior'] = (df['completed_semester_num'] >= 6).astype(int)
        # 저학년 여부 (4학기 이하)
        df['is_junior'] = (df['completed_semester_num'] <= 4).astype(int)

    # 13. time_input - 투자 예정 시간
    if 'time_input' in df.columns:
        df['time_input_num'] = pd.to_numeric(df['time_input'], errors='coerce').fillna(0)
        # 고투자 시간 (3시간 이상)
        df['high_time_commitment'] = (df['time_input_num'] >= 3).astype(int)
        # 저투자 시간 (2시간 이하)
        df['low_time_commitment'] = (df['time_input_num'] <= 2).astype(int)

    # 14. project_type - 프로젝트 타입
    if 'project_type' in df.columns:
        df['prefer_team_project'] = (df['project_type'] == '팀').astype(int)
        df['prefer_individual_project'] = (df['project_type'] == '개인').astype(int)

    # 15. desired_career_path - 희망 진로
    if 'desired_career_path' in df.columns:
        df['career_employment'] = df['desired_career_path'].str.contains('취업', na=False).astype(int)
        df['career_grad_school'] = df['desired_career_path'].str.contains('대학원', na=False).astype(int)
        df['career_startup'] = df['desired_career_path'].str.contains('창업', na=False).astype(int)
        df['career_job_change'] = df['desired_career_path'].str.contains('이직', na=False).astype(int)

    # 16. major_data - 데이터 전공 여부
    if 'major_data' in df.columns:
        df['is_data_major'] = (df['major_data'] == True) | (df['major_data'] == 'True')
        df['is_data_major'] = df['is_data_major'].astype(int)

    # 17. desired_job 분석 - 희망 직무
    if 'desired_job' in df.columns:
        # 직무 개수 (콤마로 구분)
        df['num_desired_jobs'] = df['desired_job'].fillna('').apply(
            lambda x: len([j for j in str(x).split(',') if j.strip()])
        )
        # 특정 직무 포함 여부
        df['want_data_analyst'] = df['desired_job'].str.contains('분석가', na=False).astype(int)
        df['want_data_scientist'] = df['desired_job'].str.contains('사이언티스트', na=False).astype(int)
        df['want_data_engineer'] = df['desired_job'].str.contains('엔지니어', na=False).astype(int)
        df['want_ai_expert'] = df['desired_job'].str.contains('인공지능', na=False).astype(int)
        df['want_developer'] = df['desired_job'].str.contains('개발자', na=False).astype(int)

    # 18. onedayclass_topic - 원데이클래스 희망 주제
    if 'onedayclass_topic' in df.columns:
        # 주제 개수 (콤마로 구분)
        df['num_onedayclass_topics'] = df['onedayclass_topic'].fillna('').apply(
            lambda x: len([t for t in str(x).split(',') if t.strip()])
        )
        # 특정 주제 포함 여부
        df['topic_python'] = df['onedayclass_topic'].str.contains('Python', na=False).astype(int)
        df['topic_ml'] = df['onedayclass_topic'].str.contains('머신러닝|딥러닝', na=False).astype(int)
        df['topic_sql'] = df['onedayclass_topic'].str.contains('SQL', na=False).astype(int)
        df['topic_viz'] = df['onedayclass_topic'].str.contains('시각화', na=False).astype(int)
        df['topic_crawling'] = df['onedayclass_topic'].str.contains('크롤링', na=False).astype(int)

    # 19. incumbents_level - 현직자 레벨 선호
    if 'incumbents_level' in df.columns:
        df['prefer_junior_mentor'] = df['incumbents_level'].str.contains('주니어', na=False).astype(int)
        df['prefer_senior_mentor'] = df['incumbents_level'].str.contains('시니어', na=False).astype(int)

    # 20. incumbents_company_level - 선호 회사 레벨
    if 'incumbents_company_level' in df.columns:
        df['prefer_bigtech'] = df['incumbents_company_level'].str.contains('빅테크|네카라쿠배', na=False).astype(int)
        df['prefer_conglomerate'] = df['incumbents_company_level'].str.contains('대기업', na=False).astype(int)
        df['prefer_startup'] = df['incumbents_company_level'].str.contains('스타트업', na=False).astype(int)
        df['prefer_overseas'] = df['incumbents_company_level'].str.contains('해외', na=False).astype(int)

    # 21. major_field 분석
    if 'major_field' in df.columns:
        df['field_it'] = df['major_field'].str.contains('IT|컴퓨터', na=False).astype(int)
        df['field_business'] = df['major_field'].str.contains('경영', na=False).astype(int)
        df['field_economics'] = df['major_field'].str.contains('경제', na=False).astype(int)
        df['field_social'] = df['major_field'].str.contains('사회', na=False).astype(int)
        df['field_natural'] = df['major_field'].str.contains('자연', na=False).astype(int)

    # 22. inflow_route 상세 분석
    if 'inflow_route' in df.columns:
        df['inflow_everytime'] = df['inflow_route'].str.contains('에브리타임', na=False).astype(int)
        df['inflow_instagram'] = df['inflow_route'].str.contains('인스타그램', na=False).astype(int)
        df['inflow_friend'] = df['inflow_route'].str.contains('지인', na=False).astype(int)
        df['inflow_external'] = df['inflow_route'].str.contains('대외활동', na=False).astype(int)

    # 23. incumbents_lecture 분석
    if 'incumbents_lecture' in df.columns:
        df['lecture_career'] = df['incumbents_lecture'].str.contains('커리어', na=False).astype(int)
        df['lecture_job'] = df['incumbents_lecture'].str.contains('직무', na=False).astype(int)
        df['lecture_trend'] = df['incumbents_lecture'].str.contains('트렌드', na=False).astype(int)

    # 24. 복합 피쳐
    # 의지 점수: 재등록 + 기존회원 + 만족 + 높은 시간 투자
    df['commitment_score'] = (
        df.get('is_re_registration', 0) +
        df.get('is_existing_member', 0) +
        df.get('why_satisfied', 0) +
        df.get('high_time_commitment', 0)
    )

    # 경험 점수: 이전 기수 참여 + 고학년 + 자격증
    df['experience_score'] = (
        df.get('num_prev_classes', 0) +
        df.get('is_senior', 0) +
        df.get('has_certificate', 0)
    )

    # IT 관련도: IT 전공 + 데이터 전공 + IT 분야
    df['it_relevance'] = (
        df.get('is_it_major', 0) +
        df.get('is_data_major', 0) +
        df.get('field_it', 0)
    )

    # 25. 수업 난이도 (class1 값 기준)
    if 'class1' in df.columns:
        df['class1_num'] = pd.to_numeric(df['class1'], errors='coerce').fillna(0)
        df['is_basic_class'] = (df['class1_num'] <= 3).astype(int)
        df['is_advanced_class'] = (df['class1_num'] >= 7).astype(int)

    return df


def preprocess_data(train_df, test_df):
    """데이터 전처리"""
    train_ids = train_df[cfg.ID_COL].values
    test_ids = test_df[cfg.ID_COL].values
    y = train_df[cfg.TARGET].values

    # 피처 엔지니어링
    train_df = create_features(train_df)
    test_df = create_features(test_df)

    feature_cols = [c for c in train_df.columns if c not in [cfg.ID_COL, cfg.TARGET]]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    cat_cols = []
    for col in feature_cols:
        if X_train[col].dtype == 'object':
            cat_cols.append(col)

    # Label Encoding for XGBoost
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = X_train[col].fillna('missing').astype(str)
        X_test[col] = X_test[col].fillna('missing').astype(str)
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

    # Fill numeric NaN
    num_cols = [c for c in feature_cols if c not in cat_cols]
    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    return X_train, X_test, y, train_ids, test_ids


def search_best_threshold(y_true, y_prob, pos_cap=0.70, step=0.005):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_thr, best_f1, best_pos = 0.5, -1.0, None

    for thr in np.arange(0.0, 1.0 + 1e-12, step):
        y_pred = (y_prob >= thr).astype(int)
        pos_rate = float(y_pred.mean())
        if pos_rate > pos_cap:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr, best_pos = f1, float(thr), pos_rate

    if best_f1 < 0:
        best_thr = 0.5

    return best_thr, best_f1, best_pos


def main():
    print("="*60)
    print("Enhanced XGBoost Model with Additional Features")
    print("="*60)

    train_df = pd.read_csv(cfg.DATA_DIR / cfg.TRAIN_FILE, encoding='utf-8-sig')
    test_df = pd.read_csv(cfg.DATA_DIR / cfg.TEST_FILE, encoding='utf-8-sig')

    print(f"\nTrain: {len(train_df)} rows, Test: {len(test_df)} rows")

    X_train, X_test, y, train_ids, test_ids = preprocess_data(train_df, test_df)

    print(f"Features: {len(X_train.columns)}")
    print(f"New features added!")

    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos = neg_count / pos_count

    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)

    oof_prob = np.zeros(len(X_train))
    test_probs = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        print(f"\n--- Fold {fold} ---")

        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_test)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.03,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos,
            'seed': cfg.SEED + fold,
            'tree_method': 'hist'
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=100
        )

        va_prob = model.predict(dval)
        oof_prob[va_idx] = va_prob

        thr, f1, _ = search_best_threshold(y_va, va_prob)
        fold_scores.append(f1)
        print(f"Fold {fold} - F1: {f1:.4f}, Threshold: {thr:.3f}")

        test_prob = model.predict(dtest)
        test_probs.append(test_prob)

    test_prob_mean = np.mean(test_probs, axis=0)

    print(f"\nXGBoost Enhanced Mean CV F1: {np.mean(fold_scores):.4f}")

    global_thr, global_f1, _ = search_best_threshold(y, oof_prob, pos_cap=0.70)
    test_pred = (test_prob_mean >= global_thr).astype(int)

    # Save OOF
    oof_df = pd.DataFrame({"ID": train_ids, "prob": oof_prob, "label": y})
    oof_df.to_csv(cfg.OUTPUT_DIR / "oof_model5_xgboost_enhanced.csv", index=False)

    # Save submission
    submission = pd.DataFrame({cfg.ID_COL: test_ids, cfg.TARGET: test_pred})
    submission.to_csv(cfg.OUTPUT_DIR / "submission_model5_xgboost_enhanced.csv", index=False)

    # Save test probabilities
    prob_df = pd.DataFrame({"ID": test_ids, "prob": test_prob_mean})
    prob_df.to_csv(cfg.OUTPUT_DIR / "prob_model5_xgboost_enhanced.csv", index=False)

    print(f"\nSaved: submission_model5_xgboost_enhanced.csv")
    print(f"Global threshold: {global_thr:.4f}, F1: {global_f1:.4f}")


if __name__ == "__main__":
    main()
