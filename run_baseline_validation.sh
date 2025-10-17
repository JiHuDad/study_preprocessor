#!/bin/bash

# Baseline 파일 품질 검증 스크립트
# 사용법: ./run_baseline_validation.sh <baseline1.parquet> <baseline2.parquet> ... [출력디렉토리]

set -e

if [ $# -lt 1 ]; then
    echo "❌ 사용법: $0 <baseline1.parquet> <baseline2.parquet> ... [출력디렉토리]"
    echo ""
    echo "예시:"
    echo "  $0 data/processed/server1/parsed.parquet data/processed/server2/parsed.parquet"
    echo "  $0 data/processed/*/parsed.parquet baseline_validation"
    echo ""
    echo "📋 설명:"
    echo "  - 여러 baseline 파일들의 품질을 검증합니다"
    echo "  - 이상한 baseline 파일들을 필터링합니다"
    echo "  - 이상탐지 정확도 향상을 위한 권장사항을 제공합니다"
    exit 1
fi

# 마지막 인수가 디렉토리인지 확인 (출력 디렉토리로 사용)
args=("$@")
last_arg="${args[-1]}"
if [ -d "$last_arg" ] || [[ "$last_arg" != *.parquet ]]; then
    output_dir="$last_arg"
    baseline_files=("${args[@]:0:$((${#args[@]}-1))}")
else
    output_dir="baseline_validation_$(date +%Y%m%d_%H%M%S)"
    baseline_files=("${args[@]}")
fi

echo "🔍 Baseline 파일 품질 검증 시작"
echo "📊 검증할 파일: ${#baseline_files[@]}개"
echo "📁 출력 디렉토리: $output_dir"
echo ""

# 가상환경 자동 감지 및 활성화
if [ -n "$VIRTUAL_ENV" ]; then
    echo "🔵 기존 가상환경 사용: $VIRTUAL_ENV"
elif [ -f ".venv/bin/activate" ]; then
    echo "🔵 .venv 가상환경 활성화 중..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "🔵 venv 가상환경 활성화 중..."
    source venv/bin/activate
fi

# Python 명령어 설정
PYTHON_CMD="python"
if [ -z "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="python3"
fi

# 필수 파일 확인
if [ ! -f "baseline_validator.py" ]; then
    echo "❌ baseline_validator.py 파일이 없습니다"
    echo "현재 디렉토리가 study_preprocessor 프로젝트 루트인지 확인하세요."
    exit 1
fi

# 시작 시간 기록
START_TIME=$(date +%s)

echo "🔍 품질 검증 실행 중..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Baseline 검증 실행
$PYTHON_CMD baseline_validator.py "${baseline_files[@]}" --output-dir "$output_dir"

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Baseline 품질 검증 완료!"
echo "⏱️  소요 시간: ${MINUTES}분 ${SECONDS}초"
echo ""

# 결과 요약 출력
if [ -f "$output_dir/validation_result.json" ]; then
    echo "📊 검증 결과 요약:"
    
    # JSON 파일에서 주요 정보 추출
    if command -v jq >/dev/null 2>&1; then
        total=$(jq -r '.total_files' "$output_dir/validation_result.json")
        valid=$(jq -r '.valid_count' "$output_dir/validation_result.json")
        invalid=$(jq -r '.invalid_count' "$output_dir/validation_result.json")
        recommended=$(jq -r '.recommended_baselines | length' "$output_dir/validation_result.json")
        
        echo "  📋 총 파일: ${total}개"
        echo "  ✅ 유효한 파일: ${valid}개"
        echo "  ❌ 무효한 파일: ${invalid}개"
        echo "  🎯 추천 파일: ${recommended}개"
        
        if [ "$recommended" -gt 0 ]; then
            echo ""
            echo "🎯 이상탐지에 사용 권장 파일들:"
            jq -r '.recommended_baselines[]' "$output_dir/validation_result.json" | while read file; do
                echo "  ✅ $file"
            done
        fi
        
        if [ "$invalid" -gt 0 ]; then
            echo ""
            echo "❌ 품질 문제가 있는 파일들:"
            jq -r '.invalid_baselines[].file' "$output_dir/validation_result.json" | while read file; do
                echo "  🚫 $file"
            done
        fi
    else
        echo "  (상세 정보는 리포트 파일을 확인하세요)"
    fi
fi

echo ""
echo "📂 상세 결과 확인:"
echo "  📄 검증 리포트: $output_dir/baseline_validation_report.md"
echo "  📊 JSON 결과: $output_dir/validation_result.json"
echo ""

echo "💡 다음 단계:"
echo "  1. 검증 리포트를 확인하여 문제가 있는 baseline 파일들을 파악하세요"
echo "  2. 품질이 좋은 baseline 파일들만 선별하여 이상탐지에 사용하세요"
echo "  3. 필요시 더 많은 정상 로그 파일을 수집하여 baseline을 보강하세요"
echo ""

echo "🔧 추천 baseline으로 이상탐지 실행:"
if [ -f "$output_dir/validation_result.json" ] && command -v jq >/dev/null 2>&1; then
    recommended_count=$(jq -r '.recommended_baselines | length' "$output_dir/validation_result.json")
    if [ "$recommended_count" -gt 0 ]; then
        echo "  alog-detect analyze-comparative \\"
        echo "    --target your_target.parquet \\"
        echo "    --baselines \\"
        jq -r '.recommended_baselines[]' "$output_dir/validation_result.json" | head -5 | while read file; do
            echo "      $file \\"
        done | sed '$ s/ \\$//'
    else
        echo "  (품질 기준을 만족하는 baseline이 없습니다)"
    fi
fi

echo ""
echo "📋 리포트 보기:"
echo "  cat $output_dir/baseline_validation_report.md"
