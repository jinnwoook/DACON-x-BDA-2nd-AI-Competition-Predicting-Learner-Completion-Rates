#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.csv → bert_train_data.csv 변환 스크립트
정형 데이터를 BERT 모델 학습용 텍스트 데이터로 변환

Usage:
    python src/preprocessing/create_bert_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
    """NaN 처리"""
    if pd.isna(val):
        return "정보없음"
    return str(val).strip()


def convert_row_to_text(row):
    """
    정형 데이터 행을 BERT 학습용 텍스트로 변환

    변환 형식:
    - 지원 동기, 목표, 강의 규모 선택 이유 등 주관식 응답 포함
    - 직업, 전공, 학습 시간, 자격증, 희망 직무 등 정형 정보 포함
    """
    parts = []

    # 1. 지원 동기
    why_bda = safe_str(row.get('whyBDA', ''))
    if why_bda and why_bda != "정보없음":
        parts.append(f"지원 동기: {why_bda}")

    # 2. 얻고자 하는 점
    what_to_gain = safe_str(row.get('what_to_gain', ''))
    if what_to_gain and what_to_gain != "정보없음":
        parts.append(f"얻고자 하는 점: {what_to_gain}")

    # 3. 강의 규모 선택 이유
    scale_reason = safe_str(row.get('incumbents_lecture_scale_reason', ''))
    if scale_reason and scale_reason != "정보없음":
        parts.append(f"강의 규모 선택 이유: {scale_reason}")

    # 4. 관심 기업
    company = safe_str(row.get('interested_company', ''))
    if company and company != "정보없음":
        parts.append(f"관심 기업: {company}")

    # 5. 직업
    job = safe_str(row.get('job', ''))
    if job and job != "정보없음":
        parts.append(f"직업은 {job}입니다")

    # 6. 전공 정보
    major_type = safe_str(row.get('major type', ''))
    major_field = safe_str(row.get('major_field', ''))

    major_parts = []
    if major_type and major_type != "정보없음":
        major_parts.append(f"전공 유형은 {major_type}이며")
    if major_field and major_field != "정보없음":
        major_parts.append(f"전공 분야는 {major_field}입니다")

    if major_parts:
        parts.append(", ".join(major_parts))

    # 7. 학습 시간
    time_input = row.get('time_input')
    if pd.notna(time_input):
        parts.append(f"일일 평균 학습 시간은 {time_input}시간입니다")

    # 8. 자격증
    cert = safe_str(row.get('certificate_acquisition', ''))
    if cert:
        parts.append(f"자격증 보유 현황은 {cert}입니다")

    # 9. 희망 직무
    desired_job = safe_str(row.get('desired_job', ''))
    if desired_job and desired_job != "정보없음":
        parts.append(f"희망 직무는 {desired_job}입니다")

    # 문장 조합
    text = ". ".join(parts)
    if text and not text.endswith('.'):
        text += "."

    return text


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

    train_df = pd.read_csv(train_path)
    print(f"Train 로드: {len(train_df)} rows")

    # 텍스트 변환
    print("\n텍스트 변환 중...")
    train_texts = train_df.apply(convert_row_to_text, axis=1)

    # 출력 DataFrame 생성
    bert_train = pd.DataFrame({
        "ID": train_df["ID"],
        "text": train_texts,
        "label": train_df["completed"]
    })

    # 저장
    output_path = OUTPUT_DIR / "bert_train_data.csv"
    bert_train.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n저장 완료: {output_path}")
    print(f"  - Total: {len(bert_train)} rows")
    print(f"  - Text length: mean={train_texts.str.len().mean():.0f}, max={train_texts.str.len().max()}")

    # 샘플 출력
    print("\n[샘플 텍스트]")
    print(train_texts.iloc[0][:300] + "...")

    # Test 데이터도 변환 (있는 경우)
    if test_path.exists():
        print("\n" + "=" * 60)
        print("Test 데이터 변환")
        print("=" * 60)

        test_df = pd.read_csv(test_path)
        test_texts = test_df.apply(convert_row_to_text, axis=1)

        bert_test = pd.DataFrame({
            "ID": test_df["ID"],
            "text": test_texts
        })

        test_output = OUTPUT_DIR / "bert_test_data.csv"
        bert_test.to_csv(test_output, index=False, encoding='utf-8-sig')
        print(f"저장 완료: {test_output}")

    print("\n" + "=" * 60)
    print("변환 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
