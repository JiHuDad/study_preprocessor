#!/usr/bin/env python3
"""MS-CRED 데이터 동일성 검증 스크립트

학습 시와 추론 시 사용한 window_counts.parquet가 실제로 동일한지 확인합니다.
통계가 비슷해도 실제 데이터 내용이 다를 수 있습니다.
"""

import pandas as pd
import numpy as np
import sys

if len(sys.argv) != 3:
    print("사용법: python diagnose_mscred_data_identity.py <학습_window_counts.parquet> <추론_window_counts.parquet>")
    print("\n예시:")
    print("  python scripts/diagnose_mscred_data_identity.py \\")
    print("    models_20251106/training_workspace/window_counts.parquet \\")
    print("    inference_20251106/window_counts.parquet")
    sys.exit(1)

train_path = sys.argv[1]
infer_path = sys.argv[2]

print("=" * 70)
print("🔍 MS-CRED 데이터 동일성 검증")
print("=" * 70)
print()

try:
    train_df = pd.read_parquet(train_path)
    infer_df = pd.read_parquet(infer_path)

    # 템플릿 컬럼만 추출
    train_cols = sorted([c for c in train_df.columns if c.startswith('t') and c[1:].isdigit()])
    infer_cols = sorted([c for c in infer_df.columns if c.startswith('t') and c[1:].isdigit()])

    print(f"📁 학습 데이터: {train_path}")
    print(f"   - 윈도우 개수: {len(train_df)}")
    print(f"   - 템플릿 채널 수: {len(train_cols)}")
    print()

    print(f"📁 추론 데이터: {infer_path}")
    print(f"   - 윈도우 개수: {len(infer_df)}")
    print(f"   - 템플릿 채널 수: {len(infer_cols)}")
    print()

    print("-" * 70)
    print("🔍 기본 검사")
    print("-" * 70)

    # 1. 윈도우 개수 비교
    if len(train_df) != len(infer_df):
        print(f"❌ 윈도우 개수가 다릅니다! (학습: {len(train_df)}, 추론: {len(infer_df)})")
        print("   → 학습과 추론에 사용한 데이터가 다릅니다!")
    else:
        print(f"✅ 윈도우 개수 동일: {len(train_df)}개")

    print()

    # 2. 채널 이름 비교
    if set(train_cols) != set(infer_cols):
        print("❌ 템플릿 채널이 다릅니다!")
    else:
        print(f"✅ 템플릿 채널 동일: {len(train_cols)}개")

    print()
    print("-" * 70)
    print("🔬 데이터 내용 비교 (첫 20개 윈도우)")
    print("-" * 70)

    # 3. 실제 데이터 값 비교
    compare_rows = min(20, len(train_df), len(infer_df))
    train_data = train_df[train_cols].fillna(0).values[:compare_rows]
    infer_data = infer_df[infer_cols].fillna(0).values[:compare_rows]

    # 완전 동일성 체크
    if np.array_equal(train_data, infer_data):
        print("✅ 첫 20개 윈도우가 완전히 동일합니다!")
        print("   → 학습과 추론에 동일한 데이터를 사용했습니다.")
        identical = True
    else:
        print("❌ 첫 20개 윈도우가 다릅니다!")
        print("   → 학습과 추론에 다른 데이터를 사용했습니다!")
        identical = False

        # 차이 분석
        diff = np.abs(train_data - infer_data)
        diff_ratio = np.sum(diff > 0) / diff.size * 100

        print(f"\n   차이 통계:")
        print(f"   - 다른 값의 비율: {diff_ratio:.1f}%")
        print(f"   - 평균 절대 차이: {np.mean(diff):.4f}")
        print(f"   - 최대 차이: {np.max(diff):.4f}")

    print()
    print("-" * 70)
    print("📊 샘플 데이터 비교 (첫 3개 윈도우, 첫 5개 채널)")
    print("-" * 70)

    for row_idx in range(min(3, compare_rows)):
        print(f"\n윈도우 #{row_idx + 1}:")
        print(f"{'채널':<8} {'학습':>10} {'추론':>10} {'차이':>10} {'일치':>6}")
        print("-" * 50)

        for col_idx in range(min(5, len(train_cols))):
            col = train_cols[col_idx]
            train_val = train_data[row_idx, col_idx]
            infer_val = infer_data[row_idx, col_idx]
            diff_val = abs(train_val - infer_val)
            match = "✓" if diff_val < 0.001 else "✗"

            print(f"{col:<8} {train_val:>10.2f} {infer_val:>10.2f} {diff_val:>10.2f} {match:>6}")

    # 전체 데이터 비교 (전체 윈도우)
    if len(train_df) == len(infer_df) and set(train_cols) == set(infer_cols):
        print()
        print("-" * 70)
        print("🔍 전체 데이터 동일성 검사")
        print("-" * 70)

        train_all = train_df[train_cols].fillna(0).values
        infer_all = infer_df[infer_cols].fillna(0).values

        if np.array_equal(train_all, infer_all):
            print("✅ 전체 데이터가 완전히 동일합니다!")
            all_identical = True
        else:
            print("❌ 전체 데이터가 다릅니다!")

            diff_all = np.abs(train_all - infer_all)
            diff_ratio_all = np.sum(diff_all > 0) / diff_all.size * 100

            print(f"\n전체 차이 통계:")
            print(f"- 다른 값의 비율: {diff_ratio_all:.1f}%")
            print(f"- 평균 절대 차이: {np.mean(diff_all):.4f}")
            print(f"- 최대 차이: {np.max(diff_all):.4f}")

            all_identical = False

    print()
    print("=" * 70)
    print("💡 진단 결과")
    print("=" * 70)

    if len(train_df) != len(infer_df):
        print("❌ 학습과 추론에 사용한 데이터가 다릅니다!")
        print()
        print("원인: 윈도우 개수가 다릅니다.")
        print()
        print("확인 사항:")
        print("1. 학습과 추론에 같은 로그 파일을 사용했나요?")
        print("2. window_size와 stride 설정이 같나요?")
        print("   - train.sh: window_size=?, stride=?")
        print("   - infer.sh: window_size=50, stride=25")

    elif not identical or (len(train_df) == len(infer_df) and not all_identical):
        print("❌ 학습과 추론에 사용한 데이터가 다릅니다!")
        print()
        print("원인: 윈도우 개수는 같지만 실제 데이터 값이 다릅니다.")
        print()
        print("이것이 높은 재구성 오차의 원인입니다!")
        print("(학습 손실 0.06 vs 추론 오차 0.27)")
        print()
        print("가능한 원인:")
        print("1. 학습과 추론에 다른 로그 파일 사용")
        print("   → 학습: logs/train.log")
        print("   → 추론: logs/test.log (다른 파일!)")
        print()
        print("2. 로그 파일은 같지만 파싱 결과가 다름")
        print("   → Drain3 파서가 비결정적으로 동작")
        print("   → template_id 매핑이 달라짐")
        print()
        print("3. 전처리 파라미터가 다름")
        print("   → window_size, stride, template_col 등")
        print()
        print("해결 방법:")
        print("- 학습 시 생성된 window_counts.parquet를 직접 사용하여 추론")
        print("- 또는 학습과 동일한 로그 파일 + 파라미터로 재생성")

    else:
        print("✅ 학습과 추론에 완전히 동일한 데이터를 사용했습니다!")
        print()
        print("그럼에도 재구성 오차가 높다면 (학습 0.06 vs 추론 0.27):")
        print()
        print("가능한 원인:")
        print("1. 모델 로딩 문제")
        print("   → 체크포인트가 제대로 로드되지 않음")
        print()
        print("2. prepare_data() 동작 차이")
        print("   → 학습과 추론 시 정규화나 시퀀스 생성이 다름")
        print()
        print("3. 모델 평가 모드 문제")
        print("   → model.eval() 호출 확인")
        print()
        print("디버깅 권장:")
        print("- 학습 데이터로 추론 시 손실 계산")
        print("- prepare_data() 출력 비교")

    print("=" * 70)

except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
