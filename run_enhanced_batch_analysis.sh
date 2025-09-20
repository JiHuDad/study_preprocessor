#!/bin/bash

# 향상된 배치 로그 분석 실행 스크립트
# 하위 디렉토리 재귀 스캔 지원
# 사용법: ./run_enhanced_batch_analysis.sh <로그디렉토리> [target파일] [최대깊이] [최대파일수] [작업디렉토리]

set -e  # 에러 발생시 즉시 중단

# 기본값 설정
LOG_DIR="$1"
TARGET_FILE="$2"
MAX_DEPTH="${3:-3}"
MAX_FILES="${4:-20}"
WORK_DIR="${5:-enhanced_analysis_$(date +%Y%m%d_%H%M%S)}"

# 인수 확인
if [ -z "$LOG_DIR" ]; then
    echo "❌ 사용법: $0 <로그디렉토리> [target파일] [최대깊이] [최대파일수] [작업디렉토리]"
    echo ""
    echo "예시:"
    echo "  $0 /var/log/"
    echo "  $0 /var/log/ server1.log"  
    echo "  $0 /var/log/ server1.log 5"
    echo "  $0 /var/log/ server1.log 5 30 my_analysis"
    echo ""
    echo "📋 설명:"
    echo "  - 로그디렉토리: 분석할 로그 파일들이 있는 루트 폴더 (하위 폴더 포함)"
    echo "  - target파일: 집중 분석할 파일 (생략시 가장 큰 파일)"
    echo "  - 최대깊이: 하위 디렉토리 스캔 깊이 (기본: 3)"
    echo "  - 최대파일수: 처리할 최대 파일 수 (기본: 20)"
    echo "  - 작업디렉토리: 결과를 저장할 폴더 (생략시 자동 생성)"
    echo ""
    echo "💡 특징:"
    echo "  - 📁 하위 디렉토리 자동 재귀 스캔"
    echo "  - 📅 날짜별/카테고리별 폴더 구조 지원"
    echo "  - 🔍 로그 형식 자동 감지 및 검증"
    echo "  - 🛠️ 전처리 오류 상세 디버깅"
    exit 1
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 로그 디렉토리를 찾을 수 없습니다: $LOG_DIR"
    exit 1
fi

# 가상환경 자동 감지 및 활성화
VENV_ACTIVATED=false

# 1. 이미 활성화된 가상환경 확인
if [ -n "$VIRTUAL_ENV" ]; then
    echo "🔵 기존 가상환경 감지됨: $VIRTUAL_ENV"
    VENV_ACTIVATED=true
elif [ -f ".venv/bin/activate" ]; then
    echo "🔵 .venv 가상환경 발견, 활성화 중..."
    source .venv/bin/activate
    VENV_ACTIVATED=true
    echo "✅ 가상환경 활성화됨: $VIRTUAL_ENV"
elif [ -f "venv/bin/activate" ]; then
    echo "🔵 venv 가상환경 발견, 활성화 중..."
    source venv/bin/activate
    VENV_ACTIVATED=true
    echo "✅ 가상환경 활성화됨: $VIRTUAL_ENV"
fi

# Python 명령어 설정
PYTHON_CMD="python"
if [ "$VENV_ACTIVATED" = false ]; then
    PYTHON_CMD="python3"
fi

# 필수 파일 존재 확인
required_files=(
    "enhanced_batch_analyzer.py"
    "temporal_anomaly_detector.py" 
    "comparative_anomaly_detector.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 필수 파일이 없습니다: $file"
        echo "현재 디렉토리가 study_preprocessor 프로젝트 루트인지 확인하세요."
        exit 1
    fi
done

# 프로젝트 설치 확인
if ! $PYTHON_CMD -c "import study_preprocessor" 2>/dev/null; then
    echo "🔧 study_preprocessor 패키지 설치 중..."
    .venv/bin/pip install -e . || {
        echo "❌ 패키지 설치 실패"
        exit 1
    }
    echo "✅ 패키지 설치 완료"
fi

echo "🚀 향상된 배치 로그 분석 시작"
echo "📂 로그 디렉토리: $LOG_DIR"
if [ -n "$TARGET_FILE" ]; then
    echo "🎯 Target 파일: $TARGET_FILE"
else
    echo "🎯 Target 파일: 자동 선택 (가장 큰 파일)"
fi
echo "📊 스캔 깊이: $MAX_DEPTH, 최대 파일: $MAX_FILES개"
echo "📁 작업 디렉토리: $WORK_DIR"
echo "🐍 Python 실행: $PYTHON_CMD"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# 향상된 배치 분석 실행
if [ -n "$TARGET_FILE" ]; then
    $PYTHON_CMD enhanced_batch_analyzer.py "$LOG_DIR" \
        --target "$TARGET_FILE" \
        --max-depth "$MAX_DEPTH" \
        --max-files "$MAX_FILES" \
        --work-dir "$WORK_DIR"
else
    $PYTHON_CMD enhanced_batch_analyzer.py "$LOG_DIR" \
        --max-depth "$MAX_DEPTH" \
        --max-files "$MAX_FILES" \
        --work-dir "$WORK_DIR"
fi

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 향상된 배치 분석 완료!"
echo "⏱️  소요 시간: ${MINUTES}분 ${SECONDS}초"
echo ""

# 결과 요약 출력
if [ -f "$WORK_DIR/ENHANCED_ANALYSIS_SUMMARY.md" ]; then
    echo "📋 분석 결과 요약:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 요약 파일에서 핵심 정보만 추출하여 출력
    grep -E "^\*\*|^- ✅|^- ❌|^🚨|^✅|^⚠️|^#### 📁" "$WORK_DIR/ENHANCED_ANALYSIS_SUMMARY.md" | head -30
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

echo "📂 상세 결과 확인:"
echo "  📄 종합 리포트: $WORK_DIR/COMPREHENSIVE_ANALYSIS_REPORT.md"
echo "  📁 작업 폴더: $WORK_DIR/"
echo ""

# 결과 파일들 나열
echo "📊 생성된 결과 파일들:"
find "$WORK_DIR" -name "*.md" -o -name "*.json" | sort | while read file; do
    # macOS 호환: 현재 디렉토리 기준 상대 경로 계산
    rel_path=$(echo "$file" | sed "s|^$(pwd)/||")
    echo "  📝 $rel_path"
done

echo ""
echo "🔧 추가 분석 명령어:"
if [ -d "$WORK_DIR" ]; then
    # Target 디렉토리 찾기 (가장 큰 디렉토리 또는 target 키워드 포함)
    target_processed_dir=""
    
    # 1. 요약 리포트에서 Target 파일명 추출 시도
    if [ -f "$WORK_DIR/ENHANCED_ANALYSIS_SUMMARY.md" ]; then
        target_name=$(grep "^### Target 파일:" "$WORK_DIR/ENHANCED_ANALYSIS_SUMMARY.md" | sed 's/### Target 파일: //' | tr -d ' ')
        if [ -n "$target_name" ]; then
            # Target 파일명에서 확장자 제거하고 processed_ 디렉토리 찾기
            target_base=$(echo "$target_name" | sed 's/\.[^.]*$//')
            target_processed_dir=$(find "$WORK_DIR" -name "processed_${target_base}" -type d | head -1)
        fi
    fi
    
    # 2. Target 디렉토리를 찾지 못했으면 가장 큰 processed 디렉토리 사용
    if [ -z "$target_processed_dir" ]; then
        target_processed_dir=$(find "$WORK_DIR" -name "processed_*" -type d | while read dir; do
            size=$(du -s "$dir" 2>/dev/null | cut -f1)
            echo "$size $dir"
        done | sort -nr | head -1 | cut -d' ' -f2)
    fi
    
    if [ -n "$target_processed_dir" ] && [ -d "$target_processed_dir" ]; then
        echo "  $PYTHON_CMD analyze_results.py --data-dir $target_processed_dir"
        echo "  $PYTHON_CMD visualize_results.py --data-dir $target_processed_dir"
    else
        echo "  ⚠️  Target 처리 디렉토리를 찾을 수 없습니다"
        echo "  📁 사용 가능한 디렉토리들:"
        find "$WORK_DIR" -name "processed_*" -type d | sed 's/^/    /'
    fi
fi

echo ""
echo "💡 Tip: 종합 리포트를 보려면 다음 명령어를 사용하세요:"
echo "  cat $WORK_DIR/COMPREHENSIVE_ANALYSIS_REPORT.md"
echo ""
echo "📁 스캔된 디렉토리 구조:"
echo "  find $LOG_DIR -name '*.log' -o -name '*.txt' | head -10"
