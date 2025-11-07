#!/usr/bin/env python3
"""MS-CRED 재구성 오차 진단 스크립트

추론 시 재구성 오차가 높은 원인을 진단합니다.
학습 손실과 추론 오차를 비교하여 문제 여부를 판단합니다.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_reconstruction_errors(infer_result_path: str) -> None:
    """추론 결과의 재구성 오차 분석"""

    print("=" * 70)
    print("📊 MS-CRED 재구성 오차 진단")
    print("=" * 70)
    print()

    try:
        df = pd.read_parquet(infer_result_path)

        if 'reconstruction_error' not in df.columns:
            print("❌ reconstruction_error 컬럼이 없습니다.")
            print(f"   사용 가능한 컬럼: {df.columns.tolist()}")
            return

        errors = df['reconstruction_error'].values
        threshold = df['threshold'].iloc[0] if 'threshold' in df.columns else None

        print(f"📁 추론 결과 파일: {infer_result_path}")
        print(f"   윈도우 개수: {len(df)}")
        print()

        print("-" * 70)
        print("📈 재구성 오차 통계")
        print("-" * 70)
        print(f"최소값:    {np.min(errors):.6f}")
        print(f"25% 백분위: {np.percentile(errors, 25):.6f}")
        print(f"중앙값:    {np.median(errors):.6f}")
        print(f"평균값:    {np.mean(errors):.6f}")
        print(f"75% 백분위: {np.percentile(errors, 75):.6f}")
        print(f"95% 백분위: {np.percentile(errors, 95):.6f}")
        print(f"최대값:    {np.max(errors):.6f}")
        print(f"표준편차:  {np.std(errors):.6f}")
        print()

        if threshold:
            print(f"🎯 임계값 (95% 백분위): {threshold:.6f}")
            anomaly_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            anomaly_rate = anomaly_count / len(df) * 100
            print(f"🚨 이상 탐지율: {anomaly_rate:.1f}% ({anomaly_count}/{len(df)})")
            print()

        # 오차 분포
        print("-" * 70)
        print("📊 오차 분포 (히스토그램)")
        print("-" * 70)

        bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
        labels = ["0~0.05", "0.05~0.1", "0.1~0.2", "0.2~0.3", "0.3~0.5", "0.5~1.0", "1.0+"]

        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            count = np.sum((errors >= low) & (errors < high))
            pct = count / len(errors) * 100
            bar = "█" * int(pct / 2)
            print(f"{labels[i]:>10}: {bar:40} {pct:5.1f}% ({count})")

        print()
        print("-" * 70)
        print("💡 진단 결과")
        print("-" * 70)

        avg_error = np.mean(errors)

        # 평균 오차 기준 평가
        if avg_error < 0.05:
            print("✅ 재구성 오차가 매우 낮습니다! (< 0.05)")
            print("   모델이 데이터를 거의 완벽하게 재구성하고 있습니다.")
        elif avg_error < 0.1:
            print("✅ 재구성 오차가 낮습니다! (< 0.1)")
            print("   모델이 데이터를 잘 재구성하고 있습니다.")
        elif avg_error < 0.3:
            print("⚠️  재구성 오차가 보통 수준입니다 (0.1~0.3)")
            print("   이는 정규화된 데이터 기준으로 허용 가능한 범위입니다.")
            print()
            print("   확인 사항:")
            print("   1. 학습 시 마지막 에폭의 검증 손실이 얼마였나요?")
            print("      → 학습 손실이 0.2 정도였다면 추론 오차 0.2~0.3은 정상입니다")
            print()
            print("   2. 추론 데이터가 학습 데이터와 얼마나 유사한가요?")
            print("      → 완전히 동일한 데이터: 학습 손실 수준이어야 함")
            print("      → 유사한 데이터: 약간 높을 수 있음")
            print("      → 다른 분포의 데이터: 상당히 높을 수 있음")
        else:
            print("❌ 재구성 오차가 높습니다! (> 0.3)")
            print("   이는 다음 중 하나를 의미할 수 있습니다:")
            print()
            print("   1. 모델 학습이 충분하지 않음")
            print("      → 학습 에폭 증가 권장 (최소 50+ 에폭)")
            print()
            print("   2. 추론 데이터가 학습 데이터와 매우 다름")
            print("      → 학습/추론 데이터 분포 확인 필요")
            print()
            print("   3. 모델 용량이 데이터에 비해 부족")
            print("      → base_channels 증가 시도 (기본값 32)")

        print()

        # 학습/검증 분할 이슈 체크
        if len(errors) > 20:
            # 앞 80%와 뒤 20% 오차 비교 (학습은 앞 80%만 사용)
            n_train = int(len(errors) * 0.8)
            train_region_errors = errors[:n_train]
            val_region_errors = errors[n_train:]

            train_avg = np.mean(train_region_errors)
            val_avg = np.mean(val_region_errors)

            print("-" * 70)
            print("🔍 학습/검증 영역 분석")
            print("-" * 70)
            print(f"앞 80% 영역 평균 오차: {train_avg:.6f}")
            print(f"뒤 20% 영역 평균 오차: {val_avg:.6f}")
            print()

            if val_avg > train_avg * 1.5:
                print("⚠️  뒤 20% 영역의 오차가 훨씬 높습니다!")
                print("   이는 학습 시 validation_split=0.2로 인해")
                print("   뒤 20% 데이터는 학습에 사용되지 않았기 때문일 수 있습니다.")
                print()
                print("   해결 방법:")
                print("   - 추론 데이터가 학습 데이터와 동일하다면:")
                print("     앞 80% 영역의 오차를 기준으로 판단하세요")
            else:
                print("✅ 전체 영역에서 오차가 고르게 분포합니다")

        print("=" * 70)

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {infer_result_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python diagnose_mscred_error.py <mscred_infer.parquet>")
        print("\n예시:")
        print("  python scripts/diagnose_mscred_error.py inference_20251106/mscred_infer.parquet")
        sys.exit(1)

    infer_result_path = sys.argv[1]
    analyze_reconstruction_errors(infer_result_path)
