#!/bin/bash

# MS-CRED 기능 데모 스크립트
# MS-CRED 이상탐지 기능을 빠르게 체험할 수 있는 데모입니다.

set -e

echo "🚀 MS-CRED 기능 데모 시작"
echo "======================================"
echo ""

# 가상환경 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  가상환경이 활성화되지 않았습니다."
    if [ -d ".venv" ]; then
        echo "🔧 .venv 가상환경을 활성화합니다..."
        source .venv/bin/activate
        echo "✅ 가상환경 활성화 완료"
    else
        echo "❌ 가상환경을 찾을 수 없습니다. 다음 명령어로 생성하세요:"
        echo "  python3 -m venv .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install -e ."
        exit 1
    fi
fi

# 의존성 확인
echo "🔍 의존성 확인 중..."
if ! python -c "import torch, matplotlib, seaborn" 2>/dev/null; then
    echo "📦 필요한 의존성 설치 중..."
    pip install -r requirements.txt
fi

# 작업 디렉토리 설정
DEMO_DIR="demo_mscred_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEMO_DIR"

echo ""
echo "📁 데모 디렉토리: $DEMO_DIR"
echo ""

# 1. 합성 데이터 생성
echo "1️⃣  합성 로그 데이터 생성..."
alog-detect gen-synth \
  --out "$DEMO_DIR/demo.log" \
  --lines 1000 \
  --anomaly-rate 0.05

echo "✅ 합성 로그 생성 완료: 1000줄 (이상률 5%)"
echo ""

# 2. 전처리
echo "2️⃣  로그 전처리..."
alog-detect parse \
  --input "$DEMO_DIR/demo.log" \
  --out-dir "$DEMO_DIR/processed" \
  --drain-state "$DEMO_DIR/drain3.json"

echo "✅ 전처리 완료"
echo ""

# 3. MS-CRED 입력 생성
echo "3️⃣  MS-CRED 입력 데이터 생성..."
alog-detect build-mscred \
  --parsed "$DEMO_DIR/processed/parsed.parquet" \
  --out-dir "$DEMO_DIR/processed" \
  --window-size 50 \
  --stride 25

echo "✅ MS-CRED 입력 완료"
echo ""

# 4. MS-CRED 학습
echo "4️⃣  MS-CRED 모델 학습 (20 에포크)..."
alog-detect mscred-train \
  --window-counts "$DEMO_DIR/processed/window_counts.parquet" \
  --out "$DEMO_DIR/mscred_demo.pth" \
  --epochs 20

echo "✅ MS-CRED 학습 완료"
echo ""

# 5. MS-CRED 추론
echo "5️⃣  MS-CRED 이상탐지 추론..."
alog-detect mscred-infer \
  --window-counts "$DEMO_DIR/processed/window_counts.parquet" \
  --model "$DEMO_DIR/mscred_demo.pth" \
  --threshold 95.0

echo "✅ MS-CRED 추론 완료"
echo ""

# 6. 결과 분석 및 시각화
echo "6️⃣  MS-CRED 결과 분석 및 시각화..."
alog-detect analyze-mscred \
  --data-dir "$DEMO_DIR/processed" \
  --output-dir "$DEMO_DIR/analysis"

echo "✅ 분석 및 시각화 완료"
echo ""

# 7. 로그 샘플 분석 (MS-CRED 포함)
echo "7️⃣  실제 로그 샘플 분석..."
alog-detect analyze-samples \
  --processed-dir "$DEMO_DIR/processed" \
  --output-dir "$DEMO_DIR/log_samples" \
  --max-samples 3

echo "✅ 로그 샘플 분석 완료"
echo ""

# 8. 리포트 생성
echo "8️⃣  최종 리포트 생성..."
alog-detect report --processed-dir "$DEMO_DIR/processed"

echo "✅ 리포트 생성 완료"
echo ""

# 결과 요약
echo "🎉 MS-CRED 데모 완료!"
echo "======================================"
echo ""
echo "📊 생성된 파일들:"
echo "  📝 원본 로그: $DEMO_DIR/demo.log"
echo "  📝 라벨 파일: $DEMO_DIR/demo.log.labels.parquet"
echo "  📝 전처리 결과: $DEMO_DIR/processed/parsed.parquet"
echo "  📝 윈도우 카운트: $DEMO_DIR/processed/window_counts.parquet"
echo "  📝 MS-CRED 모델: $DEMO_DIR/mscred_demo.pth"
echo "  📝 MS-CRED 결과: $DEMO_DIR/processed/mscred_infer.parquet"
echo "  📝 분석 리포트: $DEMO_DIR/analysis/mscred_analysis_report.md"
echo "  📝 시각화: $DEMO_DIR/analysis/mscred_analysis.png"
echo "  📝 로그 샘플 분석: $DEMO_DIR/log_samples/anomaly_analysis_report.md"
echo "  📝 CLI 리포트: $DEMO_DIR/processed/report.md"
echo ""

# 결과 미리보기
if [ -f "$DEMO_DIR/processed/mscred_infer.parquet" ]; then
    echo "📋 MS-CRED 결과 요약:"
    python -c "
import pandas as pd
df = pd.read_parquet('$DEMO_DIR/processed/mscred_infer.parquet')
anomalies = df[df['is_anomaly'] == True]
print(f'전체 윈도우: {len(df):,}개')
print(f'이상 윈도우: {len(anomalies):,}개')
print(f'이상탐지율: {len(anomalies)/len(df):.1%}')
print(f'평균 재구성 오차: {df[\"reconstruction_error\"].mean():.4f}')
print(f'최대 재구성 오차: {df[\"reconstruction_error\"].max():.4f}')
"
    echo ""
fi

echo "🔍 자세한 분석 결과:"
echo "  - MS-CRED 분석 리포트: cat $DEMO_DIR/analysis/mscred_analysis_report.md"
echo "  - 로그 샘플 분석: cat $DEMO_DIR/log_samples/anomaly_analysis_report.md"
echo "  - CLI 리포트: cat $DEMO_DIR/processed/report.md"
echo ""

echo "🎯 MS-CRED 특징:"
echo "  ✨ 멀티스케일 컨볼루션으로 다양한 패턴 탐지"
echo "  ✨ 어텐션 메커니즘으로 중요한 특성 강조"
echo "  ✨ 재구성 오차 기반 이상탐지"
echo "  ✨ 윈도우 단위 시계열 분석"
echo ""

echo "💡 다음 단계:"
echo "  - 실제 로그로 테스트: ./run_full_pipeline_pip.sh /path/to/your.log"
echo "  - 배치 분석: ./run_enhanced_batch_analysis.sh /var/log/"
echo "  - 개별 MS-CRED 분석: alog-detect analyze-mscred --data-dir /path/to/data"
echo ""
echo "🎉 MS-CRED 데모를 완료했습니다!"
