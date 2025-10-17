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
echo "  - 🧠 DeepLog LSTM 이상탐지 자동 수행"
echo "  - 🔬 MS-CRED 멀티스케일 이상탐지 자동 수행"
echo "  - 📊 실제 로그 샘플 자동 추출 및 분석"
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
        echo "현재 디렉토리가 anomaly_log_detector 프로젝트 루트인지 확인하세요."
        exit 1
    fi
done

# 프로젝트 설치 확인
if ! $PYTHON_CMD -c "import anomaly_log_detector" 2>/dev/null; then
    echo "🔧 anomaly_log_detector 패키지 설치 중..."
    .venv/bin/pip install -e . || {
        echo "❌ 패키지 설치 실패"
        exit 1
    }
    echo "✅ 패키지 설치 완료"
fi

echo "🚀 향상된 배치 로그 분석 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
echo "🔄 수행할 분석 단계:"
echo "  1️⃣  로그 파일 스캔 및 Target/Baseline 선택"
echo "  2️⃣  로그 전처리 및 템플릿 추출 (Drain3)"
echo "  3️⃣  베이스라인 이상탐지 (윈도우 기반)"
echo "  4️⃣  DeepLog 학습 및 추론 (LSTM 시퀀스 예측)"
echo "  5️⃣  MS-CRED 학습 및 추론 (멀티스케일 컨볼루션)"
echo "  6️⃣  시간 기반 이상탐지 (시간대별 패턴 비교)"
echo "  7️⃣  비교 분석 (파일 간 패턴 차이)"
echo "  8️⃣  로그 샘플 추출 및 분석 (실제 이상 로그)"
echo "  9️⃣  종합 리포트 생성"
echo ""
echo "⏱️  예상 소요 시간: 5-15분 (파일 크기에 따라)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  총 소요 시간: ${MINUTES}분 ${SECONDS}초"
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
echo "  📄 요약 리포트: $WORK_DIR/ENHANCED_ANALYSIS_SUMMARY.md"
echo "  📁 작업 폴더: $WORK_DIR/"
echo ""

# 주요 결과 파일들 확인 및 요약
echo "📊 분석 결과 요약:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Target 처리 결과 확인
target_processed_count=0
baseline_processed_count=0

for dir in "$WORK_DIR"/processed_*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        
        # 분석 결과 파일들 확인
        has_parsed=$([ -f "$dir/parsed.parquet" ] && echo "✅" || echo "❌")
        has_baseline=$([ -f "$dir/baseline_scores.parquet" ] && echo "✅" || echo "❌") 
        has_deeplog=$([ -f "$dir/deeplog_infer.parquet" ] && echo "✅" || echo "❌")
        has_mscred=$([ -f "$dir/mscred_infer.parquet" ] && echo "✅" || echo "❌")
        has_temporal=$([ -d "$dir/temporal_analysis" ] && echo "✅" || echo "❌")
        has_comparative=$([ -d "$dir/comparative_analysis" ] && echo "✅" || echo "❌")
        has_samples=$([ -d "$dir/log_samples_analysis" ] && echo "✅" || echo "❌")
        
        # Target vs Baseline 구분 (target_info.json 기반)
        is_target=false
        if [ -f "$WORK_DIR/target_info.json" ]; then
            target_dir_name=$(python3 -c "import json; data=json.load(open('$WORK_DIR/target_info.json')); print(data['target_processed_dir'])" 2>/dev/null)
            if [ "$dir_name" = "$target_dir_name" ]; then
                is_target=true
            fi
        fi
        
        if [ "$is_target" = true ]; then
            echo "🎯 Target: $dir_name"
            target_processed_count=$((target_processed_count + 1))
        else
            echo "📂 Baseline: $dir_name"
            baseline_processed_count=$((baseline_processed_count + 1))
        fi
        
        echo "    전처리: $has_parsed | Baseline탐지: $has_baseline | DeepLog: $has_deeplog | MS-CRED: $has_mscred"
        echo "    시간분석: $has_temporal | 비교분석: $has_comparative | 로그샘플: $has_samples"
        echo ""
    fi
done

echo "📈 처리 결과 통계:"
echo "  🎯 Target 파일: ${target_processed_count}개"
echo "  📂 Baseline 파일: ${baseline_processed_count}개"
echo "  📁 총 처리 디렉토리: $((target_processed_count + baseline_processed_count))개"
echo ""

# 결과 파일들 나열
echo "📊 생성된 주요 파일들:"
find "$WORK_DIR" -name "*.md" | sort | while read file; do
    rel_path=$(echo "$file" | sed "s|^$(pwd)/||")
    echo "  📝 $rel_path"
done
echo ""

find "$WORK_DIR" -name "*.parquet" -o -name "*.pth" -o -name "*.json" | wc -l | while read count; do
    echo "  📊 데이터 파일: ${count}개 (parquet, pth, json)"
done

echo ""
echo "🔧 추가 분석 명령어:"
if [ -d "$WORK_DIR" ]; then
    # Target 디렉토리 찾기 (target_info.json 기반)
    target_processed_dir=""
    
    # 1. target_info.json에서 실제 Target 디렉토리 확인
    if [ -f "$WORK_DIR/target_info.json" ]; then
        target_dir_name=$(python3 -c "import json; data=json.load(open('$WORK_DIR/target_info.json')); print(data['target_processed_dir'])" 2>/dev/null)
        if [ -n "$target_dir_name" ] && [ -d "$WORK_DIR/$target_dir_name" ]; then
            target_processed_dir="$WORK_DIR/$target_dir_name"
        fi
    fi
    
    # 2. target_info.json이 없으면 기존 방식으로 fallback
    if [ -z "$target_processed_dir" ]; then
        # 가장 큰 processed 디렉토리 사용
        target_processed_dir=$(find "$WORK_DIR" -name "processed_*" -type d | while read dir; do
            size=$(du -s "$dir" 2>/dev/null | cut -f1)
            echo "$size $dir"
        done | sort -nr | head -1 | cut -d' ' -f2)
    fi
    
    if [ -n "$target_processed_dir" ] && [ -d "$target_processed_dir" ]; then
        echo "🎯 Target 분석 디렉토리: $(basename "$target_processed_dir")"
        echo ""
        
        # 사용 가능한 분석 도구들 확인 및 제안
        echo "📊 상세 분석 도구:"
        if [ -f "analyze_results.py" ]; then
            echo "  $PYTHON_CMD analyze_results.py --data-dir $target_processed_dir"
        fi
        if [ -f "visualize_results.py" ]; then
            echo "  $PYTHON_CMD visualize_results.py --data-dir $target_processed_dir"
        fi
        if [ -f "mscred_analyzer.py" ] && [ -f "$target_processed_dir/mscred_infer.parquet" ]; then
            echo "  $PYTHON_CMD mscred_analyzer.py --data-dir $target_processed_dir"
        fi
        if [ -f "log_sample_analyzer.py" ]; then
            echo "  $PYTHON_CMD log_sample_analyzer.py $target_processed_dir"
        fi
        echo ""
        
        # 각 분석 방법별 결과 확인
        echo "🔍 분석 결과 확인:"
        if [ -f "$target_processed_dir/baseline_scores.parquet" ]; then
            baseline_count=$(python3 -c "import pandas as pd; df=pd.read_parquet('$target_processed_dir/baseline_scores.parquet'); print(f'{(df[\"is_anomaly\"]==True).sum()}/{len(df)}')" 2>/dev/null || echo "N/A")
            echo "  📊 Baseline 이상: $baseline_count 윈도우"
        fi
        
        if [ -f "$target_processed_dir/deeplog_infer.parquet" ]; then
            deeplog_count=$(python3 -c "import pandas as pd; df=pd.read_parquet('$target_processed_dir/deeplog_infer.parquet'); print(f'{(df[\"in_topk\"]==False).sum()}/{len(df)}')" 2>/dev/null || echo "N/A")
            echo "  🧠 DeepLog 위반: $deeplog_count 시퀀스"
        fi
        
        if [ -f "$target_processed_dir/mscred_infer.parquet" ]; then
            mscred_count=$(python3 -c "import pandas as pd; df=pd.read_parquet('$target_processed_dir/mscred_infer.parquet'); print(f'{(df[\"is_anomaly\"]==True).sum()}/{len(df)}')" 2>/dev/null || echo "N/A")
            echo "  🔬 MS-CRED 이상: $mscred_count 윈도우"
        fi
        
        if [ -d "$target_processed_dir/temporal_analysis" ] && [ -f "$target_processed_dir/temporal_analysis/temporal_anomalies.json" ]; then
            temporal_count=$(python3 -c "import json; data=json.load(open('$target_processed_dir/temporal_analysis/temporal_anomalies.json')); print(len(data))" 2>/dev/null || echo "N/A")
            echo "  🕐 시간 기반 이상: $temporal_count 건"
        fi
        
        if [ -d "$target_processed_dir/comparative_analysis" ] && [ -f "$target_processed_dir/comparative_analysis/comparative_anomalies.json" ]; then
            comparative_count=$(python3 -c "import json; data=json.load(open('$target_processed_dir/comparative_analysis/comparative_anomalies.json')); print(len(data))" 2>/dev/null || echo "N/A")
            echo "  📊 비교 분석 이상: $comparative_count 건"
        fi
        
    else
        echo "  ⚠️  Target 처리 디렉토리를 찾을 수 없습니다"
        echo "  📁 사용 가능한 디렉토리들:"
        find "$WORK_DIR" -name "processed_*" -type d | sed 's/^/    /'
    fi
fi

echo ""
echo "💡 주요 리포트 확인 명령어:"
echo "  📄 종합 리포트: cat $WORK_DIR/COMPREHENSIVE_ANALYSIS_REPORT.md"
echo "  📄 요약 리포트: cat $WORK_DIR/ENHANCED_ANALYSIS_SUMMARY.md"

# Target 디렉토리가 있다면 로그 샘플 리포트도 추천
if [ -n "$target_processed_dir" ] && [ -d "$target_processed_dir/log_samples_analysis" ]; then
    echo "  📋 로그 샘플 분석: cat $target_processed_dir/log_samples_analysis/anomaly_analysis_report.md"
fi

echo ""
echo "🚀 빠른 실행 스크립트:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 실행 가능한 스크립트 생성
if [ -n "$target_processed_dir" ] && [ -d "$target_processed_dir" ]; then
    quick_script="$WORK_DIR/quick_analysis.sh"
    cat > "$quick_script" << EOF
#!/bin/bash
# 빠른 분석 실행 스크립트 (자동 생성됨)

echo "🔍 추가 분석 실행 중..."
cd "$(pwd)"

EOF

    if [ -f "analyze_results.py" ]; then
        echo "python analyze_results.py --data-dir $target_processed_dir" >> "$quick_script"
    fi
    
    if [ -f "visualize_results.py" ]; then
        echo "python visualize_results.py --data-dir $target_processed_dir" >> "$quick_script"
    fi
    
    if [ -f "$target_processed_dir/mscred_infer.parquet" ]; then
        echo "alog-detect analyze-mscred --data-dir $target_processed_dir --output-dir $target_processed_dir/mscred_analysis" >> "$quick_script"
    fi
    
    echo "echo '✅ 추가 분석 완료!'" >> "$quick_script"
    
    chmod +x "$quick_script"
    echo "  🎯 전체 추가 분석: ./$quick_script"
fi

echo "  📊 개별 분석:"
if [ -n "$target_processed_dir" ] && [ -f "$target_processed_dir/mscred_infer.parquet" ]; then
    echo "    alog-detect analyze-mscred --data-dir $target_processed_dir"
fi
if [ -n "$target_processed_dir" ]; then
    echo "    alog-detect analyze-samples --processed-dir $target_processed_dir"
fi

echo ""
echo "📁 스캔된 디렉토리 구조 확인:"
echo "  find $LOG_DIR -name '*.log' -o -name '*.txt' | head -10"

echo ""
echo "🎉 배치 분석이 완료되었습니다!"
echo "   - ✅ 전처리: 로그 파싱 및 템플릿 추출"
echo "   - ✅ Baseline: 윈도우 기반 이상탐지" 
echo "   - ✅ DeepLog: LSTM 시퀀스 예측 이상탐지"
echo "   - ✅ MS-CRED: 멀티스케일 컨볼루션 이상탐지"
echo "   - ✅ 시간 분석: 시간대별 패턴 비교"
echo "   - ✅ 비교 분석: 파일 간 패턴 차이"
echo "   - ✅ 로그 샘플: 실제 이상 로그 추출"
