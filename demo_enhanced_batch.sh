#!/bin/bash

# 향상된 배치 분석 데모 스크립트
# DeepLog와 MS-CRED를 포함한 전체 배치 분석 기능을 데모합니다.

set -e

echo "🚀 향상된 배치 분석 데모 시작"
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

# 데모 데이터 생성
DEMO_DIR="demo_batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEMO_DIR/logs"

echo ""
echo "📁 데모 데이터 생성: $DEMO_DIR"

# 여러 종류의 합성 로그 생성
echo "1️⃣  합성 로그 데이터 생성 중..."

# 정상 로그 (baseline용)
study-preprocess gen-synth \
  --out "$DEMO_DIR/logs/baseline1.log" \
  --lines 800 \
  --anomaly-rate 0.01

study-preprocess gen-synth \
  --out "$DEMO_DIR/logs/baseline2.log" \
  --lines 600 \
  --anomaly-rate 0.02

# Target 로그 (이상이 많은 로그)
study-preprocess gen-synth \
  --out "$DEMO_DIR/logs/target_problematic.log" \
  --lines 1000 \
  --anomaly-rate 0.08

echo "✅ 합성 로그 생성 완료"
echo "  📝 baseline1.log: 800줄 (이상률 1%)"
echo "  📝 baseline2.log: 600줄 (이상률 2%)"
echo "  📝 target_problematic.log: 1000줄 (이상률 8%)"
echo ""

# 향상된 배치 분석 실행
echo "2️⃣  향상된 배치 분석 실행..."
echo ""

./run_enhanced_batch_analysis.sh "$DEMO_DIR/logs" "target_problematic.log" 2 10 "$DEMO_DIR/analysis"

echo ""
echo "3️⃣  결과 확인 및 요약..."
echo ""

# 결과 요약 출력
if [ -d "$DEMO_DIR/analysis" ]; then
    echo "📊 생성된 분석 결과:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 주요 결과 파일들 확인
    total_files=$(find "$DEMO_DIR/analysis" -type f | wc -l)
    md_files=$(find "$DEMO_DIR/analysis" -name "*.md" | wc -l)
    parquet_files=$(find "$DEMO_DIR/analysis" -name "*.parquet" | wc -l)
    model_files=$(find "$DEMO_DIR/analysis" -name "*.pth" | wc -l)
    
    echo "  📁 총 생성 파일: ${total_files}개"
    echo "  📄 리포트 파일: ${md_files}개"
    echo "  📊 데이터 파일: ${parquet_files}개"
    echo "  🧠 모델 파일: ${model_files}개"
    echo ""
    
    # Target 분석 결과 확인
    target_dir=$(find "$DEMO_DIR/analysis" -name "processed_target_*" -type d | head -1)
    if [ -n "$target_dir" ] && [ -d "$target_dir" ]; then
        echo "🎯 Target 분석 결과:"
        
        # 각 분석 방법별 결과
        if [ -f "$target_dir/baseline_scores.parquet" ]; then
            baseline_anomalies=$(python -c "import pandas as pd; df=pd.read_parquet('$target_dir/baseline_scores.parquet'); print((df['is_anomaly']==True).sum())" 2>/dev/null || echo "N/A")
            baseline_total=$(python -c "import pandas as pd; df=pd.read_parquet('$target_dir/baseline_scores.parquet'); print(len(df))" 2>/dev/null || echo "N/A")
            echo "  📊 Baseline 이상: ${baseline_anomalies}/${baseline_total} 윈도우"
        fi
        
        if [ -f "$target_dir/deeplog_infer.parquet" ]; then
            deeplog_violations=$(python -c "import pandas as pd; df=pd.read_parquet('$target_dir/deeplog_infer.parquet'); print((df['in_topk']==False).sum())" 2>/dev/null || echo "N/A")
            deeplog_total=$(python -c "import pandas as pd; df=pd.read_parquet('$target_dir/deeplog_infer.parquet'); print(len(df))" 2>/dev/null || echo "N/A")
            echo "  🧠 DeepLog 위반: ${deeplog_violations}/${deeplog_total} 시퀀스"
        fi
        
        if [ -f "$target_dir/mscred_infer.parquet" ]; then
            mscred_anomalies=$(python -c "import pandas as pd; df=pd.read_parquet('$target_dir/mscred_infer.parquet'); print((df['is_anomaly']==True).sum())" 2>/dev/null || echo "N/A")
            mscred_total=$(python -c "import pandas as pd; df=pd.read_parquet('$target_dir/mscred_infer.parquet'); print(len(df))" 2>/dev/null || echo "N/A")
            echo "  🔬 MS-CRED 이상: ${mscred_anomalies}/${mscred_total} 윈도우"
        fi
        
        # 로그 샘플 분석 결과
        if [ -d "$target_dir/log_samples_analysis" ] && [ -f "$target_dir/log_samples_analysis/anomaly_samples.json" ]; then
            echo "  📋 로그 샘플 분석: ✅ 완료"
        fi
    fi
    
    echo ""
    echo "📄 주요 리포트 확인:"
    if [ -f "$DEMO_DIR/analysis/COMPREHENSIVE_ANALYSIS_REPORT.md" ]; then
        echo "  📄 종합 리포트: cat $DEMO_DIR/analysis/COMPREHENSIVE_ANALYSIS_REPORT.md"
    fi
    if [ -f "$DEMO_DIR/analysis/ENHANCED_ANALYSIS_SUMMARY.md" ]; then
        echo "  📄 요약 리포트: cat $DEMO_DIR/analysis/ENHANCED_ANALYSIS_SUMMARY.md"
    fi
    if [ -n "$target_dir" ] && [ -f "$target_dir/log_samples_analysis/anomaly_analysis_report.md" ]; then
        echo "  📋 로그 샘플 리포트: cat $target_dir/log_samples_analysis/anomaly_analysis_report.md"
    fi
fi

echo ""
echo "4️⃣  빠른 분석 예시..."

# 빠른 분석 실행 (있다면)
if [ -f "$DEMO_DIR/analysis/quick_analysis.sh" ]; then
    echo "🔍 추가 분석 도구 실행 중..."
    cd "$DEMO_DIR/analysis"
    ./quick_analysis.sh
    cd - > /dev/null
    echo "✅ 추가 분석 완료"
else
    echo "⚠️  추가 분석 스크립트를 찾을 수 없습니다."
fi

echo ""
echo "🎉 향상된 배치 분석 데모 완료!"
echo "======================================"
echo ""
echo "📂 데모 결과 위치: $DEMO_DIR/"
echo ""
echo "🔍 분석된 내용:"
echo "  ✅ 로그 전처리 및 템플릿 추출"
echo "  ✅ 베이스라인 이상탐지 (윈도우 기반)"
echo "  ✅ DeepLog LSTM 이상탐지"
echo "  ✅ MS-CRED 멀티스케일 이상탐지"
echo "  ✅ 시간 기반 패턴 분석"
echo "  ✅ 파일 간 비교 분석"
echo "  ✅ 실제 로그 샘플 추출"
echo "  ✅ 종합 리포트 생성"
echo ""
echo "💡 실제 로그로 테스트하려면:"
echo "  ./run_enhanced_batch_analysis.sh /var/log/"
echo "  ./run_enhanced_batch_analysis.sh /path/to/logs/ target.log"
echo ""
echo "🎯 이 데모는 DeepLog와 MS-CRED를 포함한 모든 이상탐지 방법을"
echo "   한 번에 수행하여 종합적인 분석 결과를 제공합니다!"
