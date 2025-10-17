#!/bin/bash

# Full Log Anomaly Detection Pipeline
# 사용법: ./run_full_pipeline.sh <로그파일경로> [출력디렉토리]

set -e  # 에러 발생시 즉시 중단

# 기본값 설정
LOG_FILE="$1"
OUTPUT_DIR="${2:-data/processed/auto_$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR=".cache"

# 인수 확인
if [ -z "$LOG_FILE" ]; then
    echo "❌ 사용법: $0 <로그파일경로> [출력디렉토리]"
    echo "예시: $0 /var/log/dmesg.log data/processed/my_analysis"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 로그 파일을 찾을 수 없습니다: $LOG_FILE"
    exit 1
fi

# 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "🚀 로그 이상탐지 파이프라인 시작"
echo "📂 입력 파일: $LOG_FILE"
echo "📂 출력 디렉토리: $OUTPUT_DIR"
echo ""

# 1단계: 로그 전처리 및 파싱
echo "1️⃣  로그 전처리 중..."
uv run alog-detect parse \
  --input "$LOG_FILE" \
  --out-dir "$OUTPUT_DIR" \
  --drain-state "$CACHE_DIR/drain3.json"

if [ ! -f "$OUTPUT_DIR/parsed.parquet" ]; then
    echo "❌ 전처리 실패: parsed.parquet 파일이 생성되지 않았습니다."
    exit 1
fi

echo "✅ 전처리 완료: $OUTPUT_DIR/parsed.parquet"
echo ""

# 2단계: DeepLog 입력 데이터 생성
echo "2️⃣  DeepLog 입력 데이터 생성 중..."
uv run alog-detect build-deeplog \
  --parsed "$OUTPUT_DIR/parsed.parquet" \
  --out-dir "$OUTPUT_DIR"

echo "✅ DeepLog 입력 완료: vocab.json, sequences.parquet"
echo ""

# 3단계: MS-CRED 입력 데이터 생성
echo "3️⃣  MS-CRED 입력 데이터 생성 중..."
uv run alog-detect build-mscred \
  --parsed "$OUTPUT_DIR/parsed.parquet" \
  --out-dir "$OUTPUT_DIR" \
  --window-size 50 --stride 25

echo "✅ MS-CRED 입력 완료: window_counts.parquet"
echo ""

# 4단계: 베이스라인 이상탐지
echo "4️⃣  베이스라인 이상탐지 실행 중..."
uv run alog-detect detect \
  --parsed "$OUTPUT_DIR/parsed.parquet" \
  --out-dir "$OUTPUT_DIR" \
  --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.95

echo "✅ 베이스라인 탐지 완료: baseline_scores.parquet"
echo ""

# 5단계: DeepLog 학습
echo "5️⃣  DeepLog 모델 학습 중..."
MODEL_PATH="$CACHE_DIR/deeplog_$(basename "$LOG_FILE" .log).pth"
uv run alog-detect deeplog-train \
  --seq "$OUTPUT_DIR/sequences.parquet" \
  --vocab "$OUTPUT_DIR/vocab.json" \
  --out "$MODEL_PATH" \
  --seq-len 50 --epochs 3

echo "✅ DeepLog 학습 완료: $MODEL_PATH"
echo ""

# 6단계: DeepLog 추론
echo "6️⃣  DeepLog 이상탐지 추론 중..."
uv run alog-detect deeplog-infer \
  --seq "$OUTPUT_DIR/sequences.parquet" \
  --model "$MODEL_PATH" \
  --k 3

echo "✅ DeepLog 추론 완료: deeplog_infer.parquet"
echo ""

# 7단계: 리포트 생성
echo "7️⃣  최종 리포트 생성 중..."
uv run alog-detect report --processed-dir "$OUTPUT_DIR"

echo "✅ 리포트 완료: $OUTPUT_DIR/report.md"
echo ""

# 결과 요약 출력
echo "🎉 모든 단계 완료!"
echo ""
echo "📊 결과 파일들:"
echo "  📝 전처리 결과: $OUTPUT_DIR/parsed.parquet"
echo "  📝 미리보기: $OUTPUT_DIR/preview.json"
echo "  📝 베이스라인 점수: $OUTPUT_DIR/baseline_scores.parquet"
echo "  📝 DeepLog 추론: $OUTPUT_DIR/deeplog_infer.parquet"
echo "  📝 최종 리포트: $OUTPUT_DIR/report.md"
echo ""

# 리포트 내용 미리보기
if [ -f "$OUTPUT_DIR/report.md" ]; then
    echo "📋 리포트 요약:"
    cat "$OUTPUT_DIR/report.md"
    echo ""
fi

echo "🔍 자세한 분석을 위해 다음 파일들을 확인하세요:"
echo "  - 베이스라인 이상 윈도우: $OUTPUT_DIR/baseline_preview.json"
echo "  - 로그 템플릿별 분석은 $OUTPUT_DIR/parsed.parquet 파일에서 확인 가능"
echo ""
echo "🚀 다른 로그 파일로 다시 실행하려면:"
echo "  $0 <새로운_로그파일> [새로운_출력디렉토리]"
