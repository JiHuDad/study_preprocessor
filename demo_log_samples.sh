#!/bin/bash

# 로그 샘플 분석 데모 스크립트
# 합성 데이터로 이상 로그 샘플 추출 기능을 시연

set -e

echo "🎬 로그 샘플 분석 데모 시작"
echo "======================================"

# 가상환경 활성화
if [ -f ".venv/bin/activate" ]; then
    echo "🔵 가상환경 활성화 중..."
    source .venv/bin/activate
fi

# 작업 디렉토리 설정
DEMO_DIR="demo_log_samples"
PROCESSED_DIR="$DEMO_DIR/processed"

echo "📁 데모 디렉토리 생성: $DEMO_DIR"
mkdir -p "$DEMO_DIR"

echo ""
echo "1️⃣  합성 로그 생성 (이상 포함)"
echo "──────────────────────────────────────"

# 합성 로그 생성 (3% 이상률)
study-preprocess gen-synth \
    --out "$DEMO_DIR/demo.log" \
    --lines 1000 \
    --anomaly-rate 0.05

echo "✅ 합성 로그 생성 완료: $DEMO_DIR/demo.log"
echo "   - 총 1000개 라인"
echo "   - 5% 이상률 (약 50개 이상 로그)"

echo ""
echo "2️⃣  로그 전처리"
echo "──────────────────────────────────────"

# 전처리 실행
study-preprocess parse \
    --input "$DEMO_DIR/demo.log" \
    --out-dir "$PROCESSED_DIR" \
    --drain-state "$DEMO_DIR/drain.json"

echo "✅ 전처리 완료: $PROCESSED_DIR/parsed.parquet"

echo ""
echo "3️⃣  이상탐지 실행"
echo "──────────────────────────────────────"

# 베이스라인 이상탐지
study-preprocess detect \
    --parsed "$PROCESSED_DIR/parsed.parquet" \
    --out-dir "$PROCESSED_DIR" \
    --window-size 50 \
    --stride 25

echo "✅ 베이스라인 이상탐지 완료"

# DeepLog 입력 생성
study-preprocess build-deeplog \
    --parsed "$PROCESSED_DIR/parsed.parquet" \
    --out-dir "$PROCESSED_DIR"

echo "✅ DeepLog 입력 생성 완료"

# DeepLog 학습 (빠른 데모용)
study-preprocess deeplog-train \
    --seq "$PROCESSED_DIR/sequences.parquet" \
    --vocab "$PROCESSED_DIR/vocab.json" \
    --out "$DEMO_DIR/deeplog_demo.pth" \
    --seq-len 20 \
    --epochs 1

echo "✅ DeepLog 학습 완료 (간단 버전)"

# DeepLog 추론
study-preprocess deeplog-infer \
    --seq "$PROCESSED_DIR/sequences.parquet" \
    --model "$DEMO_DIR/deeplog_demo.pth" \
    --k 3

echo "✅ DeepLog 추론 완료"

echo ""
echo "4️⃣  🌟 이상 로그 샘플 분석 🌟"
echo "──────────────────────────────────────"

# 로그 샘플 분석 실행
study-preprocess analyze-samples \
    --processed-dir "$PROCESSED_DIR" \
    --max-samples 3 \
    --context-lines 2

echo ""
echo "5️⃣  📄 결과 확인"
echo "──────────────────────────────────────"

# 결과 파일들 확인
SAMPLE_REPORT="$PROCESSED_DIR/log_samples_analysis/anomaly_analysis_report.md"
SAMPLE_DATA="$PROCESSED_DIR/log_samples_analysis/anomaly_samples.json"

if [ -f "$SAMPLE_REPORT" ]; then
    echo "✅ 사람이 읽기 쉬운 분석 리포트:"
    echo "   📄 $SAMPLE_REPORT"
    echo ""
    echo "📋 리포트 미리보기 (처음 30줄):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    head -30 "$SAMPLE_REPORT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
else
    echo "❌ 분석 리포트를 찾을 수 없습니다"
fi

if [ -f "$SAMPLE_DATA" ]; then
    echo "✅ 상세 분석 데이터:"
    echo "   📊 $SAMPLE_DATA"
    
    # JSON 데이터에서 요약 정보 추출 (jq가 있다면)
    if command -v jq >/dev/null 2>&1; then
        echo ""
        echo "📊 이상탐지 결과 요약:"
        for method in baseline deeplog comparative; do
            count=$(jq -r ".$method.anomaly_count // 0" "$SAMPLE_DATA" 2>/dev/null || echo "0")
            if [ "$count" != "0" ]; then
                analyzed=$(jq -r ".$method.analyzed_count // 0" "$SAMPLE_DATA" 2>/dev/null || echo "0")
                echo "  - $method: $count개 이상 발견, $analyzed개 샘플 분석됨"
            fi
        done
    fi
else
    echo "❌ 분석 데이터를 찾을 수 없습니다"
fi

echo ""
echo "6️⃣  💡 추가 분석 명령어"
echo "──────────────────────────────────────"

echo "전체 리포트 (샘플 포함):"
echo "  study-preprocess report --processed-dir $PROCESSED_DIR --with-samples"
echo ""
echo "평가 (라벨 있음):"
echo "  study-preprocess eval --processed-dir $PROCESSED_DIR --labels $DEMO_DIR/demo.log.labels.parquet"
echo ""
echo "상세 분석 리포트 보기:"
echo "  cat $SAMPLE_REPORT"
echo ""
echo "원본 로그와 이상 라벨 확인:"
echo "  head $DEMO_DIR/demo.log"
echo "  python -c \"import pandas as pd; print(pd.read_parquet('$DEMO_DIR/demo.log.labels.parquet').head())\""

echo ""
echo "🎉 로그 샘플 분석 데모 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 이제 실제 로그 파일에서 어떤 부분이 문제인지 쉽게 확인할 수 있습니다!"
echo "📄 $SAMPLE_REPORT 파일을 열어보세요."
echo ""
echo "🧹 데모 정리 (선택사항):"
echo "  rm -rf $DEMO_DIR"
