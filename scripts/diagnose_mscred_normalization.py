#!/usr/bin/env python3
"""MS-CRED 정규화 통계 진단 스크립트

학습 시와 추론 시 사용된 정규화 통계값(평균, 표준편차)을 비교합니다.
정규화 통계가 다르면 동일한 데이터라도 모델이 다르게 인식합니다.
"""

import pandas as pd
import numpy as np
import sys

if len(sys.argv) != 3:
    print("사용법: python diagnose_mscred_normalization.py <학습_window_counts.parquet> <추론_window_counts.parquet>")
    print("\n예시:")
    print("  python scripts/diagnose_mscred_normalization.py \\")
    print("    models_20251106/window_counts.parquet \\")
    print("    inference_20251106/window_counts.parquet")
    sys.exit(1)

train_path = sys.argv[1]
infer_path = sys.argv[2]

print("=" * 70)
print("📊 MS-CRED 정규화 통계 진단")
print("=" * 70)
print()

try:
    train_df = pd.read_parquet(train_path)
    infer_df = pd.read_parquet(infer_path)

    # 템플릿 컬럼만 추출
    train_cols = sorted([c for c in train_df.columns if c.startswith('t') and c[1:].isdigit()])
    infer_cols = sorted([c for c in infer_df.columns if c.startswith('t') and c[1:].isdigit()])

    if set(train_cols) != set(infer_cols):
        print("❌ 템플릿 채널이 다릅니다. 먼저 diagnose_mscred.py를 실행하세요.")
        sys.exit(1)

    # 데이터 추출
    train_data = train_df[train_cols].fillna(0).values
    infer_data = infer_df[infer_cols].fillna(0).values

    # 정규화 통계 계산 (prepare_data와 동일한 방식)
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)

    infer_mean = infer_data.mean(axis=0)
    infer_std = infer_data.std(axis=0)

    print(f"📁 학습 데이터: {train_path}")
    print(f"   - 윈도우 개수: {len(train_data)}")
    print(f"   - 템플릿 개수: {len(train_cols)}")
    print()

    print(f"📁 추론 데이터: {infer_path}")
    print(f"   - 윈도우 개수: {len(infer_data)}")
    print(f"   - 템플릿 개수: {len(infer_cols)}")
    print()

    print("-" * 70)
    print("🔍 정규화 통계 비교 (첫 10개 채널)")
    print("-" * 70)
    print()

    # 상세 비교
    print(f"{'채널':<8} {'학습 평균':>12} {'추론 평균':>12} {'평균 차이':>12} {'차이율':>10}")
    print("-" * 70)

    mean_diffs = []
    for i in range(min(10, len(train_cols))):
        col = train_cols[i]
        t_mean = train_mean[i]
        i_mean = infer_mean[i]
        diff = abs(t_mean - i_mean)
        diff_pct = (diff / (abs(t_mean) + 1e-8)) * 100
        mean_diffs.append(diff_pct)
        print(f"{col:<8} {t_mean:>12.4f} {i_mean:>12.4f} {diff:>12.4f} {diff_pct:>9.2f}%")

    print()
    print(f"{'채널':<8} {'학습 표준편차':>15} {'추론 표준편차':>15} {'표준편차 차이':>15} {'차이율':>10}")
    print("-" * 70)

    std_diffs = []
    for i in range(min(10, len(train_cols))):
        col = train_cols[i]
        t_std = train_std[i]
        i_std = infer_std[i]
        diff = abs(t_std - i_std)
        diff_pct = (diff / (abs(t_std) + 1e-8)) * 100
        std_diffs.append(diff_pct)
        print(f"{col:<8} {t_std:>15.4f} {i_std:>15.4f} {diff:>15.4f} {diff_pct:>9.2f}%")

    print()
    print("-" * 70)
    print("📈 전체 통계 요약")
    print("-" * 70)

    # 전체 채널에 대한 통계
    all_mean_diffs = []
    all_std_diffs = []

    for i in range(len(train_cols)):
        mean_diff_pct = (abs(train_mean[i] - infer_mean[i]) / (abs(train_mean[i]) + 1e-8)) * 100
        std_diff_pct = (abs(train_std[i] - infer_std[i]) / (abs(train_std[i]) + 1e-8)) * 100
        all_mean_diffs.append(mean_diff_pct)
        all_std_diffs.append(std_diff_pct)

    print(f"평균 차이율:")
    print(f"  - 최소: {np.min(all_mean_diffs):.2f}%")
    print(f"  - 평균: {np.mean(all_mean_diffs):.2f}%")
    print(f"  - 최대: {np.max(all_mean_diffs):.2f}%")
    print()

    print(f"표준편차 차이율:")
    print(f"  - 최소: {np.min(all_std_diffs):.2f}%")
    print(f"  - 평균: {np.mean(all_std_diffs):.2f}%")
    print(f"  - 최대: {np.max(all_std_diffs):.2f}%")
    print()

    # 정규화 적용 예시
    print("-" * 70)
    print("🔬 정규화 적용 예시 (첫 번째 채널, 첫 윈도우)")
    print("-" * 70)

    if len(train_data) > 0 and len(infer_data) > 0:
        original_train = train_data[0, 0]
        original_infer = infer_data[0, 0]

        normalized_train = (original_train - train_mean[0]) / (train_std[0] + 1e-8)
        normalized_infer = (original_infer - infer_mean[0]) / (infer_std[0] + 1e-8)

        print(f"학습 시:")
        print(f"  원본값: {original_train:.2f}")
        print(f"  정규화 통계: mean={train_mean[0]:.2f}, std={train_std[0]:.2f}")
        print(f"  정규화 결과: {normalized_train:.4f}")
        print()

        print(f"추론 시:")
        print(f"  원본값: {original_infer:.2f}")
        print(f"  정규화 통계: mean={infer_mean[0]:.2f}, std={infer_std[0]:.2f}")
        print(f"  정규화 결과: {normalized_infer:.4f}")
        print()

        if abs(normalized_train - normalized_infer) > 0.1:
            print(f"❌ 정규화 후 값 차이: {abs(normalized_train - normalized_infer):.4f}")
            print("   → 동일한 원본값이라도 다른 정규화 통계로 인해 다른 값이 됩니다!")

    print()
    print("=" * 70)
    print("💡 진단 결과")
    print("=" * 70)

    avg_mean_diff = np.mean(all_mean_diffs)
    avg_std_diff = np.mean(all_std_diffs)

    if avg_mean_diff > 5.0 or avg_std_diff > 5.0:
        print("❌ 정규화 통계 불일치가 발견되었습니다!")
        print()
        print(f"평균 차이율: {avg_mean_diff:.2f}% (>5% 위험)")
        print(f"표준편차 차이율: {avg_std_diff:.2f}% (>5% 위험)")
        print()
        print("이것이 높은 재구성 오차의 원인입니다!")
        print()
        print("해결 방법:")
        print("1. 학습 시 정규화 통계(mean, std)를 JSON으로 저장")
        print("2. 추론 시 저장된 통계를 재사용하여 정규화")
        print("3. MSCREDTrainer.prepare_data()와 MSCREDInference.detect_anomalies() 수정 필요")
        print()
        print("참고: 이는 DeepLog vocab 문제, build_mscred_window_counts factorize 문제와")
        print("      유사한 '학습/추론 불일치' 패턴입니다.")
    else:
        print("✅ 정규화 통계가 거의 일치합니다!")
        print()
        print(f"평균 차이율: {avg_mean_diff:.2f}% (양호)")
        print(f"표준편차 차이율: {avg_std_diff:.2f}% (양호)")
        print()
        print("재구성 오차가 높다면 다른 원인을 확인하세요:")
        print("- 모델 학습 에폭이 충분한지 (최소 30+ 에폭 권장)")
        print("- 학습 손실이 충분히 감소했는지")
        print("- 데이터 양이 충분한지 (최소 100+ 윈도우 권장)")

    print("=" * 70)

except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
