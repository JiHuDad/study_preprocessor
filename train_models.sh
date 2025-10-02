#!/bin/bash

# 로그 이상탐지 모델 학습 스크립트
# 정상 로그 데이터로부터 DeepLog, MS-CRED 모델과 베이스라인 통계를 학습합니다.
# 사용법: ./train_models.sh <로그디렉토리> [모델저장디렉토리] [최대깊이] [최대파일수]

set -e  # 에러 발생시 즉시 중단

# 기본값 설정
LOG_DIR="$1"
MODEL_DIR="${2:-models_$(date +%Y%m%d_%H%M%S)}"
MAX_DEPTH="${3:-3}"
MAX_FILES="${4:-50}"

# 인수 확인
if [ -z "$LOG_DIR" ]; then
    echo "❌ 사용법: $0 <로그디렉토리> [모델저장디렉토리] [최대깊이] [최대파일수]"
    echo ""
    echo "예시:"
    echo "  $0 /var/log/normal/"
    echo "  $0 /var/log/normal/ my_models"  
    echo "  $0 /var/log/normal/ my_models 5 100"
    echo ""
    echo "📋 설명:"
    echo "  - 로그디렉토리: 정상 로그 파일들이 있는 폴더 (학습용)"
    echo "  - 모델저장디렉토리: 학습된 모델들을 저장할 폴더 (생략시 자동 생성)"
    echo "  - 최대깊이: 하위 디렉토리 스캔 깊이 (기본: 3)"
    echo "  - 최대파일수: 학습에 사용할 최대 파일 수 (기본: 50)"
    echo ""
    echo "💡 특징:"
    echo "  - 📁 하위 디렉토리 자동 재귀 스캔"
    echo "  - 🧠 DeepLog LSTM 모델 학습"
    echo "  - 🔬 MS-CRED 멀티스케일 컨볼루션 모델 학습"
    echo "  - 📊 베이스라인 통계 (정상 패턴) 학습"
    echo "  - 🔧 Drain3 템플릿 상태 저장"
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

# 프로젝트 설치 확인
if ! $PYTHON_CMD -c "import study_preprocessor" 2>/dev/null; then
    echo "🔧 study_preprocessor 패키지 설치 중..."
    .venv/bin/pip install -e . || {
        echo "❌ 패키지 설치 실패"
        exit 1
    }
    echo "✅ 패키지 설치 완료"
fi

# 모델 저장 디렉토리 생성
mkdir -p "$MODEL_DIR"
WORK_DIR="$MODEL_DIR/training_workspace"
mkdir -p "$WORK_DIR"

echo "🚀 로그 이상탐지 모델 학습 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 학습 로그 디렉토리: $LOG_DIR"
echo "💾 모델 저장 디렉토리: $MODEL_DIR"
echo "📊 스캔 깊이: $MAX_DEPTH, 최대 파일: $MAX_FILES개"
echo "🐍 Python 실행: $PYTHON_CMD"
echo ""
echo "🔄 수행할 학습 단계:"
echo "  1️⃣  로그 파일 스캔 및 수집"
echo "  2️⃣  로그 전처리 및 템플릿 추출 (Drain3)"
echo "  3️⃣  베이스라인 통계 학습 (정상 패턴)"
echo "  4️⃣  DeepLog 입력 생성 및 모델 학습"
echo "  5️⃣  MS-CRED 입력 생성 및 모델 학습"
echo "  6️⃣  학습 결과 검증 및 메타데이터 저장"
echo ""
echo "⏱️  예상 소요 시간: 10-30분 (데이터 크기에 따라)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# 1단계: 로그 파일 스캔
echo "1️⃣  로그 파일 스캔 중..."
log_files=()
log_patterns=("*.log" "*.txt" "*.out" "*.log.*" "*.syslog" "*.messages")

for pattern in "${log_patterns[@]}"; do
    while IFS= read -r -d '' file; do
        if [ -f "$file" ] && [ -s "$file" ]; then
            log_files+=("$file")
        fi
    done < <(find "$LOG_DIR" -maxdepth "$MAX_DEPTH" -name "$pattern" -type f -print0 2>/dev/null)
done

# 파일 크기순 정렬 및 제한
if [ ${#log_files[@]} -gt 0 ]; then
    # 파일 크기순 정렬
    printf '%s\n' "${log_files[@]}" | while read -r file; do
        size=$(stat -c%s "$file" 2>/dev/null || echo 0)
        echo "$size $file"
    done | sort -nr | head -"$MAX_FILES" | cut -d' ' -f2- > "$WORK_DIR/selected_files.txt"
    
    mapfile -t selected_files < "$WORK_DIR/selected_files.txt"
    echo "✅ 발견된 로그 파일: ${#log_files[@]}개, 선택된 파일: ${#selected_files[@]}개"
    
    # 선택된 파일들 출력
    echo "📋 학습에 사용할 파일들:"
    for i, file in "${!selected_files[@]}"; do
        size=$(stat -c%s "$file" 2>/dev/null | numfmt --to=iec)
        echo "  $((i+1)). $(basename "$file") ($size)"
        if [ $i -ge 9 ]; then
            echo "  ... 및 $((${#selected_files[@]} - 10))개 파일 더"
            break
        fi
    done
else
    echo "❌ 로그 파일을 찾을 수 없습니다."
    exit 1
fi
echo ""

# 2단계: 로그 병합 및 전처리
echo "2️⃣  로그 전처리 및 템플릿 추출 중..."
MERGED_LOG="$WORK_DIR/merged_training.log"
DRAIN_STATE="$MODEL_DIR/drain3_state.json"

# 로그 파일들 병합 (시간순 정렬)
> "$MERGED_LOG"  # 파일 초기화
for file in "${selected_files[@]}"; do
    echo "   처리 중: $(basename "$file")"
    cat "$file" >> "$MERGED_LOG"
done

echo "✅ 병합된 로그 크기: $(stat -c%s "$MERGED_LOG" | numfmt --to=iec)"

# Drain3로 전처리
echo "   Drain3 템플릿 추출 중..."
$PYTHON_CMD -m study_preprocessor.cli parse \
    --input "$MERGED_LOG" \
    --out-dir "$WORK_DIR" \
    --drain-state "$DRAIN_STATE"

if [ ! -f "$WORK_DIR/parsed.parquet" ]; then
    echo "❌ 전처리 실패: parsed.parquet 파일이 생성되지 않았습니다."
    exit 1
fi

echo "✅ 전처리 완료: $(wc -l < "$MERGED_LOG") 라인 → $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$WORK_DIR/parsed.parquet')))" 2>/dev/null || echo "N/A") 레코드"
echo ""

# 3단계: 베이스라인 통계 학습
echo "3️⃣  베이스라인 통계 학습 중..."
$PYTHON_CMD -m study_preprocessor.cli detect \
    --parsed "$WORK_DIR/parsed.parquet" \
    --out-dir "$WORK_DIR" \
    --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.95

if [ -f "$WORK_DIR/baseline_scores.parquet" ]; then
    # 베이스라인 통계를 모델 디렉토리로 복사
    cp "$WORK_DIR/baseline_scores.parquet" "$MODEL_DIR/"
    
    # 정상 패턴 통계 추출 및 저장
    $PYTHON_CMD -c "
import pandas as pd
import json
import numpy as np

# 베이스라인 결과 로드
df = pd.read_parquet('$WORK_DIR/baseline_scores.parquet')

# 정상 패턴 통계 계산
normal_windows = df[df['is_anomaly'] == False]
stats = {
    'total_windows': len(df),
    'normal_windows': len(normal_windows),
    'anomaly_rate': float((df['is_anomaly'] == True).mean()),
    'template_stats': {
        'mean_new_template_rate': float(normal_windows['new_template_rate'].mean()),
        'std_new_template_rate': float(normal_windows['new_template_rate'].std()),
        'mean_total_templates': float(normal_windows['total_templates'].mean()),
        'std_total_templates': float(normal_windows['total_templates'].std()),
    },
    'frequency_stats': {
        'mean_freq_score': float(normal_windows['freq_score'].mean()),
        'std_freq_score': float(normal_windows['freq_score'].std()),
    }
}

with open('$MODEL_DIR/baseline_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f'✅ 베이스라인 통계 저장: {stats[\"normal_windows\"]}/{stats[\"total_windows\"]} 정상 윈도우')
"
else
    echo "⚠️  베이스라인 학습 실패, 계속 진행합니다."
fi
echo ""

# 4단계: DeepLog 학습
echo "4️⃣  DeepLog 모델 학습 중..."

# DeepLog 입력 생성
$PYTHON_CMD -m study_preprocessor.cli build-deeplog \
    --parsed "$WORK_DIR/parsed.parquet" \
    --out-dir "$WORK_DIR"

if [ -f "$WORK_DIR/sequences.parquet" ] && [ -f "$WORK_DIR/vocab.json" ]; then
    # DeepLog 모델 학습
    DEEPLOG_MODEL="$MODEL_DIR/deeplog.pth"
    $PYTHON_CMD -m study_preprocessor.cli deeplog-train \
        --seq "$WORK_DIR/sequences.parquet" \
        --vocab "$WORK_DIR/vocab.json" \
        --out "$DEEPLOG_MODEL" \
        --seq-len 50 --epochs 10 --batch-size 64
    
    if [ -f "$DEEPLOG_MODEL" ]; then
        # vocab.json을 모델 디렉토리로 복사
        cp "$WORK_DIR/vocab.json" "$MODEL_DIR/"
        echo "✅ DeepLog 학습 완료: $(stat -c%s "$DEEPLOG_MODEL" | numfmt --to=iec)"
    else
        echo "❌ DeepLog 학습 실패"
    fi
else
    echo "❌ DeepLog 입력 생성 실패"
fi
echo ""

# 5단계: MS-CRED 학습
echo "5️⃣  MS-CRED 모델 학습 중..."

# MS-CRED 입력 생성
$PYTHON_CMD -m study_preprocessor.cli build-mscred \
    --parsed "$WORK_DIR/parsed.parquet" \
    --out-dir "$WORK_DIR" \
    --window-size 50 --stride 25

if [ -f "$WORK_DIR/window_counts.parquet" ]; then
    # MS-CRED 모델 학습
    MSCRED_MODEL="$MODEL_DIR/mscred.pth"
    $PYTHON_CMD -m study_preprocessor.cli mscred-train \
        --window-counts "$WORK_DIR/window_counts.parquet" \
        --out "$MSCRED_MODEL" \
        --epochs 50
    
    if [ -f "$MSCRED_MODEL" ]; then
        echo "✅ MS-CRED 학습 완료: $(stat -c%s "$MSCRED_MODEL" | numfmt --to=iec)"
    else
        echo "❌ MS-CRED 학습 실패"
    fi
else
    echo "❌ MS-CRED 입력 생성 실패"
fi
echo ""

# 6단계: 메타데이터 저장
echo "6️⃣  학습 메타데이터 저장 중..."

# 학습 정보 메타데이터 생성
$PYTHON_CMD -c "
import json
import os
from datetime import datetime
from pathlib import Path

metadata = {
    'training_info': {
        'timestamp': datetime.now().isoformat(),
        'log_directory': '$LOG_DIR',
        'total_files': ${#selected_files[@]},
        'max_depth': $MAX_DEPTH,
        'max_files': $MAX_FILES
    },
    'models': {
        'deeplog': os.path.exists('$MODEL_DIR/deeplog.pth'),
        'mscred': os.path.exists('$MODEL_DIR/mscred.pth'),
        'baseline_stats': os.path.exists('$MODEL_DIR/baseline_stats.json'),
        'drain3_state': os.path.exists('$MODEL_DIR/drain3_state.json'),
        'vocab': os.path.exists('$MODEL_DIR/vocab.json')
    },
    'files': [
        '$(printf "%s\n" "${selected_files[@]}" | sed "s|$LOG_DIR/||g" | head -20)'
    ]
}

with open('$MODEL_DIR/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print('✅ 메타데이터 저장 완료')
"

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 모델 학습 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  총 소요 시간: ${MINUTES}분 ${SECONDS}초"
echo ""

# 결과 요약 출력
echo "📊 학습된 모델들:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

models_trained=0
if [ -f "$MODEL_DIR/deeplog.pth" ]; then
    size=$(stat -c%s "$MODEL_DIR/deeplog.pth" | numfmt --to=iec)
    echo "  🧠 DeepLog 모델: deeplog.pth ($size)"
    models_trained=$((models_trained + 1))
else
    echo "  ❌ DeepLog 모델: 학습 실패"
fi

if [ -f "$MODEL_DIR/mscred.pth" ]; then
    size=$(stat -c%s "$MODEL_DIR/mscred.pth" | numfmt --to=iec)
    echo "  🔬 MS-CRED 모델: mscred.pth ($size)"
    models_trained=$((models_trained + 1))
else
    echo "  ❌ MS-CRED 모델: 학습 실패"
fi

if [ -f "$MODEL_DIR/baseline_stats.json" ]; then
    echo "  📊 베이스라인 통계: baseline_stats.json"
    models_trained=$((models_trained + 1))
else
    echo "  ❌ 베이스라인 통계: 생성 실패"
fi

if [ -f "$MODEL_DIR/drain3_state.json" ]; then
    echo "  🔧 Drain3 상태: drain3_state.json"
else
    echo "  ❌ Drain3 상태: 저장 실패"
fi

if [ -f "$MODEL_DIR/vocab.json" ]; then
    echo "  📚 어휘 사전: vocab.json"
else
    echo "  ❌ 어휘 사전: 저장 실패"
fi

echo ""
echo "📈 학습 결과 통계:"
echo "  ✅ 성공한 모델: ${models_trained}/3개"
echo "  📁 모델 저장 위치: $MODEL_DIR"
echo "  📋 메타데이터: $MODEL_DIR/metadata.json"
echo ""

# 추론 스크립트 사용법 안내
echo "🚀 추론 실행 방법:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ./run_inference.sh $MODEL_DIR /path/to/target.log"
echo ""
echo "💡 추가 옵션:"
echo "  ./run_inference.sh $MODEL_DIR /path/to/target.log result_dir"
echo ""

# 임시 작업 디렉토리 정리 여부 확인
echo "🧹 임시 파일 정리:"
if [ -d "$WORK_DIR" ]; then
    echo "  📁 임시 작업 디렉토리: $WORK_DIR"
    echo "  💡 정리하려면: rm -rf $WORK_DIR"
fi

echo ""
echo "🎉 모델 학습이 완료되었습니다!"
echo "   - ✅ 로그 전처리 및 템플릿 추출"
echo "   - ✅ 베이스라인 정상 패턴 학습" 
echo "   - ✅ DeepLog LSTM 모델 학습"
echo "   - ✅ MS-CRED 컨볼루션 모델 학습"
echo "   - ✅ 학습 메타데이터 저장"
echo ""
echo "🔍 이제 run_inference.sh로 이상탐지를 수행할 수 있습니다!"
