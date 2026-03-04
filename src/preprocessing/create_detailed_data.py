"""
정형 데이터 → 상세 텍스트 변환 스크립트
- model2_koelectra_detailed.py 에서 사용하는 train_detailed.csv / test_detailed.csv 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# ============================================================================
# Helper Functions
# ============================================================================
def safe_str(val):
    if pd.isna(val):
        return ""
    return str(val).strip()

def simplify_job(val):
    if pd.isna(val):
        return ""
    job = str(val).strip()
    if '대학생' in job:
        return "대학생"
    elif '직장인' in job:
        return "직장인"
    elif '취준' in job:
        return "취준생"
    return job

def simplify_motivation(val):
    if pd.isna(val):
        return ""
    motive = str(val).strip()
    if '혼자' in motive or '어려워' in motive:
        return "혼자 공부하기 어려워서"
    elif '커리큘럼' in motive or '관리' in motive or '큰 규모' in motive:
        return "체계적인 커리큘럼 때문에"
    elif '혜택' in motive or '현직자' in motive or '공모전' in motive:
        return "현직자 강연, 공모전 등 혜택 때문에"
    elif '이전 기수' in motive or '만족' in motive:
        return "이전 기수 만족해서 재등록"
    elif '시간' in motive and '부담' in motive:
        return "시간적으로 부담 없어서"
    elif '코딩 테스트' in motive or '면접' in motive:
        return "가입 절차가 간단해서"
    return motive[:30] if len(motive) > 30 else motive

def simplify_goal(val):
    if pd.isna(val):
        return ""
    goal = str(val).strip()
    if '프로젝트' in goal:
        return "프로젝트 경험"
    elif '데이터 분석' in goal:
        return "데이터 분석 역량"
    elif '공모전' in goal:
        return "공모전 경험"
    elif '네트워크' in goal or '인적' in goal:
        return "네트워킹"
    return goal


def convert_to_detailed_style(row):
    """상세 버전 - model2_koelectra_detailed.py 에서 사용"""
    parts = []

    # 기본 정보
    job = simplify_job(row.get('job', ''))
    major_field = safe_str(row.get('major_field', ''))

    intro = f"{job}"
    if major_field and major_field != "정보없음":
        intro += f", {major_field.split(',')[0].strip()} 전공"
    parts.append(intro)

    # 학습 시간
    if pd.notna(row.get('time_input')):
        parts.append(f"하루 {row['time_input']}시간 학습")

    # 학기
    if pd.notna(row.get('completed_semester')):
        parts.append(f"{int(row['completed_semester'])}학기")

    # 재등록
    if pd.notna(row.get('re_registration')):
        re_reg = str(row['re_registration']).lower()
        if re_reg in ['예', 'yes', 'true', '1']:
            parts.append("재등록")

    # 수강반
    classes = []
    for col in ['class1', 'class2', 'class3', 'class4']:
        if col in row.index and pd.notna(row[col]):
            classes.append(str(int(row[col])))
    if classes:
        parts.append(f"수강반 {','.join(classes[:2])}")

    # 지원 동기
    motivation = simplify_motivation(row.get('whyBDA'))
    if motivation:
        parts.append(motivation)

    # 목표
    goal = simplify_goal(row.get('what_to_gain'))
    if goal:
        parts.append(goal)

    # 자격증
    cert = safe_str(row.get('certificate_acquisition', ''))
    if cert and cert.lower() not in ['없음', 'nan', '']:
        parts.append(f"자격증: {cert}")
    else:
        parts.append("자격증 없음")

    # 희망 직무
    desired_job = safe_str(row.get('desired_job', ''))
    if desired_job:
        parts.append(f"희망: {desired_job.split(',')[0].strip()}")

    # 프로젝트 유형
    project = safe_str(row.get('project_type', ''))
    if project:
        parts.append(f"{project} 프로젝트")

    # 강의 규모 이유
    reason = safe_str(row.get('incumbents_lecture_scale_reason', ''))
    if reason and len(reason) > 5:
        parts.append(reason[:50])

    # 관심 기업
    company = safe_str(row.get('interested_company', ''))
    if company and company.lower() not in ['없음', 'nan', '', '딱히 없음', '아직 없음', '.']:
        parts.append(f"관심: {company.split(',')[0].strip()[:20]}")

    return ". ".join(parts) + "."


def main():
    print("=" * 60)
    print("train_detailed.csv / test_detailed.csv 생성")
    print("=" * 60)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)} rows")
    print(f"Test:  {len(test_df)} rows")

    train_texts = train_df.apply(convert_to_detailed_style, axis=1)
    test_texts = test_df.apply(convert_to_detailed_style, axis=1)

    train_out = pd.DataFrame({
        "ID": train_df["ID"],
        "text": train_texts,
        "label": train_df["completed"]
    })
    test_out = pd.DataFrame({
        "ID": test_df["ID"],
        "text": test_texts
    })

    train_out.to_csv(DATA_DIR / "train_detailed.csv", index=False)
    test_out.to_csv(DATA_DIR / "test_detailed.csv", index=False)

    print(f"\nSaved: {DATA_DIR / 'train_detailed.csv'}")
    print(f"Saved: {DATA_DIR / 'test_detailed.csv'}")
    print(f"Sample: {train_texts.iloc[0]}")
    print("=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
