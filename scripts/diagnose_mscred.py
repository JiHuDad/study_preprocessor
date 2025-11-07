#!/usr/bin/env python3
"""MS-CRED 템플릿 매핑 진단 스크립트

학습 시와 추론 시 생성된 window_counts.parquet를 비교하여
템플릿 채널 매핑이 동일한지 확인합니다.

템플릿 매핑이 다르면 MS-CRED가 높은 재구성 오차를 보일 수 있습니다.
"""

import pandas as pd
import sys

if len(sys.argv) != 3:
    print("사용법: python diagnose_mscred.py <학습_window_counts.parquet> <추론_window_counts.parquet>")
    print("\n예시:")
    print("  python scripts/diagnose_mscred.py \\")
    print("    models_20251106/window_counts.parquet \\")
    print("    inference_20251106/window_counts.parquet")
    sys.exit(1)

train_path = sys.argv[1]
infer_path = sys.argv[2]

print("=" * 70)
print("📊 MS-CRED 템플릿 매핑 진단")
print("=" * 70)
print()

try:
    train_df = pd.read_parquet(train_path)
    infer_df = pd.read_parquet(infer_path)

    # 템플릿 채널 컬럼만 추출 (t0, t1, t2, ...)
    train_cols = sorted([c for c in train_df.columns if c.startswith('t') and c[1:].isdigit()])
    infer_cols = sorted([c for c in infer_df.columns if c.startswith('t') and c[1:].isdigit()])

    print(f"📁 학습 시 파일: {train_path}")
    print(f"   - 템플릿 채널 수: {len(train_cols)}")
    print(f"   - 윈도우 개수: {len(train_df)}")
    print()

    print(f"📁 추론 시 파일: {infer_path}")
    print(f"   - 템플릿 채널 수: {len(infer_cols)}")
    print(f"   - 윈도우 개수: {len(infer_df)}")
    print()

    print("-" * 70)
    print("🔍 진단 결과")
    print("-" * 70)

    # 1. 채널 수 비교
    if len(train_cols) != len(infer_cols):
        print(f"❌ 채널 수가 다릅니다! (학습: {len(train_cols)}, 추론: {len(infer_cols)})")
        print("   → MS-CRED 모델이 학습한 채널 수와 추론 입력의 채널 수가 다릅니다.")
        print("   → 이는 높은 재구성 오차의 원인입니다!")
    else:
        print(f"✅ 채널 수는 동일합니다 ({len(train_cols)}개)")
    print()

    # 2. 채널 이름 비교
    if set(train_cols) != set(infer_cols):
        print("❌ 채널 이름이 다릅니다!")
        missing_in_infer = sorted(list(set(train_cols) - set(infer_cols)))
        new_in_infer = sorted(list(set(infer_cols) - set(train_cols)))

        if missing_in_infer:
            print(f"   학습엔 있었지만 추론엔 없는 채널: {missing_in_infer[:10]}")
            if len(missing_in_infer) > 10:
                print(f"   ... 외 {len(missing_in_infer) - 10}개 더")

        if new_in_infer:
            print(f"   추론에 새로 생긴 채널: {new_in_infer[:10]}")
            if len(new_in_infer) > 10:
                print(f"   ... 외 {len(new_in_infer) - 10}개 더")

        print("\n   → 템플릿 매핑이 달라졌습니다!")
        print("   → MS-CRED가 학습한 패턴과 다른 데이터를 받고 있습니다.")
    else:
        print("✅ 채널 이름은 동일합니다")
    print()

    # 3. 채널 매핑 상세 비교
    print("-" * 70)
    print("📋 채널 매핑 상세 정보")
    print("-" * 70)
    print(f"학습 시 첫 10개 채널: {train_cols[:10]}")
    print(f"추론 시 첫 10개 채널: {infer_cols[:10]}")
    print()

    # 4. 첫 윈도우 데이터 샘플 비교
    print("-" * 70)
    print("📊 첫 윈도우 데이터 샘플 (처음 5개 채널)")
    print("-" * 70)

    if len(train_df) > 0 and len(train_cols) > 0:
        print("학습 시:")
        sample_cols = train_cols[:min(5, len(train_cols))]
        for col in sample_cols:
            val = train_df.iloc[0].get(col, 0)
            print(f"  {col}: {val}")

    print()

    if len(infer_df) > 0 and len(infer_cols) > 0:
        print("추론 시:")
        sample_cols = infer_cols[:min(5, len(infer_cols))]
        for col in sample_cols:
            val = infer_df.iloc[0].get(col, 0)
            print(f"  {col}: {val}")

    print()
    print("=" * 70)
    print("💡 권장 사항")
    print("=" * 70)

    if set(train_cols) != set(infer_cols) or len(train_cols) != len(infer_cols):
        print("❌ 템플릿 매핑 불일치가 발견되었습니다!")
        print()
        print("해결 방법:")
        print("1. build_mscred_window_counts() 함수에 template_mapping 파라미터 추가")
        print("2. 학습 시 factorize 결과를 JSON으로 저장")
        print("3. 추론 시 저장된 매핑을 재사용")
        print()
        print("이는 DeepLog vocab.json 문제와 동일한 원인입니다.")
    else:
        print("✅ 템플릿 매핑이 일치합니다!")
        print()
        print("재구성 오차가 높다면 다른 원인을 확인하세요:")
        print("- 모델 학습이 충분히 되었는지")
        print("- 임계값 설정이 적절한지")
        print("- 데이터 전처리 과정에서 문제가 없는지")

    print("=" * 70)

except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
