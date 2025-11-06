#!/bin/bash
#
# 학습/추론 워크플로우 데모 스크립트
#
# 이 스크립트는 다음을 시연합니다:
# 1. 학습용 정상 로그 데이터 생성
# 2. 모델 학습 (DeepLog)
# 3. 추론용 정상/비정상 데이터 생성
# 4. 모델 추론 및 평가

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 헬퍼 함수
print_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📍 STEP $1: $2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 작업 디렉토리 설정
DEMO_DIR="demo_workflow_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEMO_DIR"
cd "$DEMO_DIR"

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║  🚀 학습/추론 워크플로우 데모                            ║"
echo "║                                                           ║"
echo "║  이 데모는 전체 이상탐지 파이프라인을 시연합니다         ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

echo "📂 작업 디렉토리: $PWD"
echo "⏱️  예상 소요 시간: 약 5-10분"
echo ""

# ============================================================================
# STEP 1: 학습용 데이터 생성
# ============================================================================
print_step 1 "학습용 정상 로그 데이터 생성"

alog-detect gen-training-data \
    --out data/raw/training.log \
    --lines 10000

print_success "학습용 데이터 생성 완료"
echo "   파일: data/raw/training.log"
echo "   라인: 10,000 (모두 정상)"

# ============================================================================
# STEP 2: 학습 데이터 전처리
# ============================================================================
print_step 2 "학습 데이터 전처리 (Drain3 템플릿 추출)"

alog-detect parse \
    --input data/raw/training.log \
    --out-dir data/processed/train \
    --drain-state .cache/drain3_train.json

print_success "전처리 완료"
echo "   출력: data/processed/train/parsed.parquet"
echo "   Drain3 상태: .cache/drain3_train.json"

# ============================================================================
# STEP 3: DeepLog 입력 생성
# ============================================================================
print_step 3 "DeepLog 입력 데이터 생성"

alog-detect build-deeplog \
    --parsed data/processed/train/parsed.parquet \
    --out-dir data/processed/train

print_success "DeepLog 입력 생성 완료"
echo "   Vocabulary: data/processed/train/vocab.json"
echo "   Sequences: data/processed/train/sequences.parquet"

# ============================================================================
# STEP 4: DeepLog 모델 학습
# ============================================================================
print_step 4 "DeepLog 모델 학습"

alog-detect deeplog-train \
    --seq data/processed/train/sequences.parquet \
    --vocab data/processed/train/vocab.json \
    --out models/deeplog.pth \
    --seq-len 50 \
    --epochs 5

print_success "모델 학습 완료"
echo "   모델 파일: models/deeplog.pth"

# ============================================================================
# STEP 5: 추론용 정상 데이터 생성 및 테스트
# ============================================================================
print_step 5 "추론용 정상 데이터 생성 (False Positive 테스트)"

alog-detect gen-inference-normal \
    --out data/raw/test_normal.log \
    --lines 1000

print_success "추론용 정상 데이터 생성 완료"

# 전처리
alog-detect parse \
    --input data/raw/test_normal.log \
    --out-dir data/processed/test_normal \
    --drain-state .cache/drain3_train.json

# DeepLog 입력 생성
alog-detect build-deeplog \
    --parsed data/processed/test_normal/parsed.parquet \
    --out-dir data/processed/test_normal

# DeepLog 추론
alog-detect deeplog-infer \
    --seq data/processed/test_normal/sequences.parquet \
    --model models/deeplog.pth \
    --vocab data/processed/test_normal/vocab.json \
    --k 3

# Baseline 탐지
alog-detect detect \
    --parsed data/processed/test_normal/parsed.parquet \
    --out-dir data/processed/test_normal \
    --window-size 50

print_success "추론용 정상 데이터 테스트 완료"
echo "   결과: data/processed/test_normal/"

# 평가
echo ""
echo "📊 정상 데이터 평가 결과:"
alog-detect eval \
    --processed-dir data/processed/test_normal \
    --labels data/raw/test_normal.log.labels.parquet \
    --window-size 50 \
    --seq-len 50

# ============================================================================
# STEP 6: 추론용 비정상 데이터 생성 및 테스트
# ============================================================================
print_step 6 "추론용 비정상 데이터 생성 (True Positive 테스트)"

alog-detect gen-inference-anomaly \
    --out data/raw/test_anomaly.log \
    --lines 1000 \
    --anomaly-rate 0.15 \
    --anomaly-types unseen \
    --anomaly-types error \
    --anomaly-types burst

print_success "추론용 비정상 데이터 생성 완료"

# 전처리
alog-detect parse \
    --input data/raw/test_anomaly.log \
    --out-dir data/processed/test_anomaly \
    --drain-state .cache/drain3_train.json

# DeepLog 입력 생성
alog-detect build-deeplog \
    --parsed data/processed/test_anomaly/parsed.parquet \
    --out-dir data/processed/test_anomaly

# DeepLog 추론
alog-detect deeplog-infer \
    --seq data/processed/test_anomaly/sequences.parquet \
    --model models/deeplog.pth \
    --vocab data/processed/test_anomaly/vocab.json \
    --k 3

# Baseline 탐지
alog-detect detect \
    --parsed data/processed/test_anomaly/parsed.parquet \
    --out-dir data/processed/test_anomaly \
    --window-size 50

print_success "추론용 비정상 데이터 테스트 완료"
echo "   결과: data/processed/test_anomaly/"

# 평가
echo ""
echo "📊 비정상 데이터 평가 결과:"
alog-detect eval \
    --processed-dir data/processed/test_anomaly \
    --labels data/raw/test_anomaly.log.labels.parquet \
    --window-size 50 \
    --seq-len 50

# ============================================================================
# STEP 7: 리포트 생성
# ============================================================================
print_step 7 "분석 리포트 생성"

# 정상 데이터 리포트
alog-detect report \
    --processed-dir data/processed/test_normal

# 비정상 데이터 리포트
alog-detect report \
    --processed-dir data/processed/test_anomaly

print_success "리포트 생성 완료"
echo "   정상 데이터: data/processed/test_normal/report.md"
echo "   비정상 데이터: data/processed/test_anomaly/report.md"

# ============================================================================
# 완료 요약
# ============================================================================
echo -e "\n${GREEN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║  ✅ 데모 완료!                                           ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

echo "📁 생성된 파일들:"
echo ""
echo "🎓 학습 데이터:"
echo "   data/raw/training.log                    - 학습용 정상 로그 (10,000줄)"
echo "   data/processed/train/                    - 전처리 결과"
echo "   models/deeplog.pth                       - 학습된 DeepLog 모델"
echo ""
echo "✅ 추론 정상 데이터 (False Positive 테스트):"
echo "   data/raw/test_normal.log                 - 추론용 정상 로그 (1,000줄)"
echo "   data/processed/test_normal/              - 분석 결과"
echo "   data/processed/test_normal/report.md     - 분석 리포트"
echo ""
echo "🚨 추론 비정상 데이터 (True Positive 테스트):"
echo "   data/raw/test_anomaly.log                - 추론용 비정상 로그 (1,000줄)"
echo "   data/raw/test_anomaly.log.meta.json      - 이상 타입 통계"
echo "   data/processed/test_anomaly/             - 분석 결과"
echo "   data/processed/test_anomaly/report.md    - 분석 리포트"
echo ""

print_success "리포트를 확인하세요:"
echo "   cat data/processed/test_normal/report.md"
echo "   cat data/processed/test_anomaly/report.md"
echo ""
echo "   cat data/processed/test_anomaly/log_samples_analysis/anomaly_analysis_report.md"
echo ""

print_warning "이 데모는 예시용입니다. 실제 로그 데이터로 테스트하세요!"

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
