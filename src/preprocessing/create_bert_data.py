#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.csv → bert_train_data.csv 변환 스크립트
정형 데이터를 BERT 모델 학습용 텍스트 데이터로 변환

변환 패턴:
- 지원 동기: {whyBDA}
- 얻고자 하는 점: {what_to_gain}
- 강의 규모 선택 이유: {incumbents_lecture_scale_reason}
- 관심 기업: {interested_company}
- 직업은 {job}입니다.
- 전공 유형은 {major type}이며, 전공 분야는 {major_field}입니다.
- 일일 평균 학습 시간은 {time_input}시간입니다.
- 자격증 보유 현황은 {certificate_acquisition}입니다.
- 희망 직무는 {desired_job}입니다.

Usage:
    python src/preprocessing/create_bert_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR


# ============================================================================
# Helper Functions
# ============================================================================
def safe_str(val):
    """NaN을 빈 문자열로 처리"""
    if pd.isna(val):
        return ""
    return str(val).strip()


def clean_text(text):
    """텍스트 정제 - 쉼표, 괄호, 슬래시 제거"""
    if not text:
        return ""
    # 괄호를 공백으로 대체: "(현직자" → " 현직자"
    text = re.sub(r'[()]', ' ', text)
    # 슬래시 → 공백: "현대카드/29CM" → "현대카드 29CM"
    text = re.sub(r'/', ' ', text)
    # 쉼표 → 공백: "큰 규모인 만큼, 커리큘럼" → "큰 규모인 만큼 커리큘럼"
    text = re.sub(r',\s*', ' ', text)
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def convert_row_to_text(row):
    """
    정형 데이터 행을 BERT 학습용 텍스트로 변환

    원본 bert_train_data.csv와 동일한 형식으로 변환:
    "지원 동기: ... 얻고자 하는 점: ... 강의 규모 선택 이유: ... 관심 기업: ...
     직업은 ...입니다. 전공 유형은 ...이며, 전공 분야는 ...입니다.
     일일 평균 학습 시간은 ...시간입니다. 자격증 보유 현황은 ...입니다.
     희망 직무는 ...입니다."
    """
    parts = []

    # 1. 지원 동기 (whyBDA) - 쉼표 제거
    why_bda = safe_str(row.get('whyBDA', ''))
    if why_bda:
        why_bda_clean = clean_text(why_bda)
        parts.append(f"지원 동기: {why_bda_clean}")

    # 2. 얻고자 하는 점 (what_to_gain)
    what_to_gain = safe_str(row.get('what_to_gain', ''))
    if what_to_gain:
        parts.append(f"얻고자 하는 점: {what_to_gain}")

    # 3. 강의 규모 선택 이유 (incumbents_lecture_scale_reason) - 빈 값도 유지
    scale_reason = safe_str(row.get('incumbents_lecture_scale_reason', ''))
    scale_reason_clean = clean_text(scale_reason) if scale_reason else ""
    parts.append(f"강의 규모 선택 이유: {scale_reason_clean}")

    # 4. 관심 기업 (interested_company) - 쉼표, 슬래시 제거, 빈 값도 유지
    company = safe_str(row.get('interested_company', ''))
    company_clean = clean_text(company) if company else ""
    parts.append(f"관심 기업: {company_clean}")

    # 5. 직업 (job)
    job = safe_str(row.get('job', ''))
    if job:
        parts.append(f"직업은 {job}입니다")

    # 6. 전공 정보 (major type + major_field)
    major_type = safe_str(row.get('major type', ''))
    major_field = safe_str(row.get('major_field', ''))

    if major_type and major_field:
        parts.append(f"전공 유형은 {major_type}이며, 전공 분야는 {major_field}입니다")
    elif major_type:
        parts.append(f"전공 유형은 {major_type}입니다")
    elif major_field:
        parts.append(f"전공 분야는 {major_field}입니다")

    # 7. 일일 평균 학습 시간 (time_input)
    time_input = row.get('time_input')
    if pd.notna(time_input):
        parts.append(f"일일 평균 학습 시간은 {time_input}시간입니다")

    # 8. 자격증 보유 현황 (certificate_acquisition)
    cert = safe_str(row.get('certificate_acquisition', ''))
    if cert:
        parts.append(f"자격증 보유 현황은 {cert}입니다")
    else:
        parts.append("자격증 보유 현황은 없음입니다")

    # 9. 희망 직무 (desired_job)
    desired_job = safe_str(row.get('desired_job', ''))
    if desired_job:
        parts.append(f"희망 직무는 {desired_job}입니다")

    # 각 부분 끝의 마침표 제거 (나중에 일괄 추가)
    cleaned_parts = []
    for part in parts:
        part = part.rstrip('.')
        cleaned_parts.append(part)

    # 문장 조합 (". "로 연결)
    text = ". ".join(cleaned_parts)
    if text and not text.endswith('.'):
        text += "."

    return text


def verify_conversion(original_df, converted_texts, original_bert_df):
    """변환 결과 검증"""
    print("\n" + "=" * 60)
    print("변환 결과 검증")
    print("=" * 60)

    # 첫 3개 샘플 비교
    for i in range(min(3, len(converted_texts))):
        print(f"\n[샘플 {i}]")
        print(f"변환 결과: {converted_texts.iloc[i][:100]}...")
        if original_bert_df is not None and i < len(original_bert_df):
            print(f"원본 BERT: {original_bert_df['text'].iloc[i][:100]}...")
            # 유사도 체크
            if converted_texts.iloc[i] == original_bert_df['text'].iloc[i]:
                print("✓ 완전 일치!")
            else:
                print("△ 차이 있음")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("train.csv → bert_train_data.csv 변환")
    print("=" * 60)

    # 데이터 로드
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists():
        print(f"[ERROR] {train_path} not found!")
        print("data/ 폴더에 train.csv 파일을 넣어주세요.")
        return

    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    print(f"Train 로드: {len(train_df)} rows")
    print(f"컬럼: {list(train_df.columns[:10])}...")

    # 기존 bert_train_data.csv 로드 (검증용)
    original_bert_path = DATA_DIR / "bert_train_data.csv"
    original_bert_df = None
    if original_bert_path.exists():
        original_bert_df = pd.read_csv(original_bert_path, encoding='utf-8-sig')
        print(f"기존 bert_train_data.csv 로드: {len(original_bert_df)} rows")

    # 텍스트 변환
    print("\n텍스트 변환 중...")
    train_texts = train_df.apply(convert_row_to_text, axis=1)

    # 검증
    if original_bert_df is not None:
        verify_conversion(train_df, train_texts, original_bert_df)

    # 출력 DataFrame 생성
    bert_train = pd.DataFrame({
        "ID": train_df["ID"],
        "text": train_texts,
        "label": train_df["completed"]
    })

    # 저장
    output_path = OUTPUT_DIR / "bert_train_data_generated.csv"
    bert_train.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n저장 완료: {output_path}")
    print(f"  - Total: {len(bert_train)} rows")
    print(f"  - Text length: mean={train_texts.str.len().mean():.0f}, max={train_texts.str.len().max()}")

    # Test 데이터 변환
    if test_path.exists():
        print("\n" + "=" * 60)
        print("test.csv → bert_test_data.csv 변환")
        print("=" * 60)

        test_df = pd.read_csv(test_path, encoding='utf-8-sig')
        print(f"Test 로드: {len(test_df)} rows")

        test_texts = test_df.apply(convert_row_to_text, axis=1)

        bert_test = pd.DataFrame({
            "ID": test_df["ID"],
            "text": test_texts
        })

        test_output = OUTPUT_DIR / "bert_test_data_generated.csv"
        bert_test.to_csv(test_output, index=False, encoding='utf-8-sig')
        print(f"저장 완료: {test_output}")

        # 검증
        original_test_path = DATA_DIR / "bert_test_data.csv"
        if original_test_path.exists():
            original_test_df = pd.read_csv(original_test_path, encoding='utf-8-sig')
            verify_conversion(test_df, test_texts, original_test_df)

    print("\n" + "=" * 60)
    print("변환 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
