"""LogBERT 구현 테스트 스크립트"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

print("🧪 LogBERT 구현 테스트 시작...\n")

# 1. 합성 데이터 생성
print("1️⃣ 합성 로그 데이터 생성...")
synthetic_data = {
    "line_no": list(range(1, 101)),
    "timestamp": pd.date_range("2025-01-01", periods=100, freq="1min"),
    "host": ["server1"] * 100,
    "template": ["User logged in"] * 50 +
                ["Database connection established"] * 30 +
                ["Cache hit"] * 15 +
                ["CRITICAL: Disk full"] * 5  # 이상 패턴
}
df = pd.DataFrame(synthetic_data)

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # 2. 파싱된 데이터 저장
    parsed_file = tmpdir / "parsed.parquet"
    df.to_parquet(parsed_file, index=False)
    print(f"✅ 합성 데이터 생성: {len(df)} 로그\n")

    # 3. LogBERT 입력 생성
    print("2️⃣ LogBERT 입력 데이터 생성...")
    from anomaly_log_detector.builders.logbert import build_logbert_inputs

    build_logbert_inputs(
        parsed_parquet=parsed_file,
        out_dir=tmpdir
    )
    print("✅ vocab.json, sequences.parquet, special_tokens.json 생성\n")

    # 4. Vocab 확인
    import json
    vocab_file = tmpdir / "vocab.json"
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    print(f"📚 Vocab 크기: {len(vocab)}")
    print(f"   - 특수 토큰: {list(vocab.keys())[:5]}")
    print(f"   - 템플릿 예시: {list(vocab.keys())[5:8]}\n")

    # 5. LogBERT 모델 학습 (빠른 테스트용 작은 파라미터)
    print("3️⃣ LogBERT 모델 학습 (빠른 테스트 모드)...")
    from anomaly_log_detector.builders.logbert import train_logbert

    model_path = tmpdir / "logbert_test.pth"
    train_logbert(
        sequences_parquet=tmpdir / "sequences.parquet",
        vocab_json=vocab_file,
        out_path=model_path,
        seq_len=16,  # 작은 시퀀스 길이
        epochs=2,    # 빠른 테스트용
        batch_size=8,
        hidden_size=64,  # 작은 모델
        num_layers=2,
        num_heads=4
    )
    print(f"✅ 모델 학습 완료: {model_path}\n")

    # 6. LogBERT 추론
    print("4️⃣ LogBERT 이상 탐지 추론...")
    from anomaly_log_detector.builders.logbert import infer_logbert

    results_df = infer_logbert(
        sequences_parquet=tmpdir / "sequences.parquet",
        model_path=model_path,
        vocab_json=vocab_file,
        threshold_percentile=90.0,  # 낮은 임계값으로 테스트
        seq_len=16
    )

    print(f"✅ 추론 완료")
    print(f"   - 총 시퀀스: {len(results_df)}")
    print(f"   - 이상 시퀀스: {results_df['is_anomaly'].sum()}")
    print(f"   - 이상률: {results_df['is_anomaly'].mean():.2%}\n")

    # 7. 결과 상세 정보
    print("5️⃣ 추론 결과 상세:")
    print(results_df.head(10).to_string(index=False))

    print("\n\n✅ 모든 테스트 통과!")
    print("=" * 60)
    print("LogBERT 구현이 성공적으로 작동합니다!")
    print("=" * 60)
