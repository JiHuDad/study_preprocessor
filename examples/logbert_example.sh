#!/bin/bash
# LogBERT 사용 예제 스크립트

set -e  # 오류 발생 시 중단

echo "🤖 LogBERT 로그 이상탐지 예제"
echo "================================"
echo ""

# 환경 변수 설정
WORK_DIR=${1:-"data/logbert_example"}
LOG_FILE=${2:-"data/raw/synth_long.log"}

echo "📁 작업 디렉토리: $WORK_DIR"
echo "📄 로그 파일: $LOG_FILE"
echo ""

# 작업 디렉토리 생성
mkdir -p "$WORK_DIR"
mkdir -p .cache

# 1. 로그 파일이 없으면 합성 로그 생성
if [ ! -f "$LOG_FILE" ]; then
    echo "📝 1. 합성 로그 생성..."
    mkdir -p "$(dirname "$LOG_FILE")"
    alog-detect gen-synth \
        --out "$LOG_FILE" \
        --lines 2000 \
        --anomaly-rate 0.05
    echo "✅ 합성 로그 생성 완료: $LOG_FILE"
    echo ""
fi

# 2. 로그 파싱
echo "🔧 2. 로그 파싱 및 전처리..."
alog-detect parse \
    --input "$LOG_FILE" \
    --out-dir "$WORK_DIR" \
    --drain-state .cache/drain3_logbert.json
echo "✅ 파싱 완료: $WORK_DIR/parsed.parquet"
echo ""

# 3. LogBERT 입력 생성
echo "📦 3. LogBERT 입력 데이터 생성..."
alog-detect build-logbert \
    --parsed "$WORK_DIR/parsed.parquet" \
    --out-dir "$WORK_DIR" \
    --max-seq-len 512
echo "✅ LogBERT 입력 생성 완료:"
echo "   - $WORK_DIR/vocab.json"
echo "   - $WORK_DIR/sequences.parquet"
echo "   - $WORK_DIR/special_tokens.json"
echo ""

# 4. LogBERT 모델 학습
echo "🎓 4. LogBERT 모델 학습..."
echo "   (작은 모델로 빠른 학습 - 실제 사용 시 파라미터 조정 필요)"
alog-detect logbert-train \
    --seq "$WORK_DIR/sequences.parquet" \
    --vocab "$WORK_DIR/vocab.json" \
    --out .cache/logbert_example.pth \
    --seq-len 64 \
    --epochs 5 \
    --batch-size 16 \
    --hidden-size 128 \
    --num-layers 3 \
    --num-heads 4 \
    --lr 0.0001
echo "✅ 모델 학습 완료: .cache/logbert_example.pth"
echo ""

# 5. LogBERT 이상탐지 추론
echo "🔍 5. LogBERT 이상탐지 추론..."
alog-detect logbert-infer \
    --seq "$WORK_DIR/sequences.parquet" \
    --model .cache/logbert_example.pth \
    --vocab "$WORK_DIR/vocab.json" \
    --threshold-percentile 90.0 \
    --seq-len 64
echo "✅ 추론 완료: $WORK_DIR/logbert_infer.parquet"
echo ""

# 6. 결과 요약
echo "📊 6. 결과 요약"
echo "================================"
python3 - <<EOF
import pandas as pd

# 추론 결과 로드
results = pd.read_parquet('$WORK_DIR/logbert_infer.parquet')

print(f"총 시퀀스 수: {len(results)}")
print(f"이상 시퀀스 수: {results['is_anomaly'].sum()}")
print(f"이상률: {results['is_anomaly'].mean():.2%}")
print(f"\nLoss 통계:")
print(f"  - 최소: {results['avg_loss'].min():.4f}")
print(f"  - 최대: {results['avg_loss'].max():.4f}")
print(f"  - 평균: {results['avg_loss'].mean():.4f}")
print(f"  - 중앙값: {results['avg_loss'].median():.4f}")
print(f"  - 임계값: {results['threshold'].iloc[0]:.4f}")

# 상위 이상 시퀀스 표시
if results['is_anomaly'].sum() > 0:
    print(f"\n🚨 상위 이상 시퀀스 (Top 5):")
    top_anomalies = results[results['is_anomaly']].nlargest(5, 'avg_loss')
    for idx, row in top_anomalies.iterrows():
        print(f"  시퀀스 #{row['seq_idx']}: Loss={row['avg_loss']:.4f}")
EOF

echo ""
echo "✅ LogBERT 예제 완료!"
echo ""
echo "📁 생성된 파일:"
echo "   - $WORK_DIR/parsed.parquet"
echo "   - $WORK_DIR/vocab.json"
echo "   - $WORK_DIR/sequences.parquet"
echo "   - $WORK_DIR/special_tokens.json"
echo "   - $WORK_DIR/logbert_infer.parquet"
echo "   - .cache/logbert_example.pth"
echo ""
echo "💡 Tip: 실제 사용 시에는 다음 파라미터를 조정하세요:"
echo "   - --seq-len: 시퀀스 길이 (128~512)"
echo "   - --epochs: 에폭 수 (10~30)"
echo "   - --hidden-size: 은닉층 크기 (256~768)"
echo "   - --num-layers: 레이어 수 (4~12)"
echo "   - --num-heads: Attention head 수 (8~16)"
