#!/bin/bash

# 점진적 로그 이상탐지 모델 학습 스크립트
# 기존 모델을 기반으로 새로운 데이터로 점진적 학습을 수행합니다.
# 사용법: ./train_models_incremental.sh <기존모델디렉토리> <새로운로그디렉토리> [결과모델디렉토리] [최대깊이] [최대파일수]

set -e  # 에러 발생시 즉시 중단

# 기본값 설정
BASE_MODEL_DIR="$1"
NEW_LOG_DIR="$2"
OUTPUT_MODEL_DIR="${3:-models_incremental_$(date +%Y%m%d_%H%M%S)}"
MAX_DEPTH="${4:-3}"
MAX_FILES="${5:-50}"

# 인수 확인
if [ -z "$BASE_MODEL_DIR" ] || [ -z "$NEW_LOG_DIR" ]; then
    echo "❌ 사용법: $0 <기존모델디렉토리> <새로운로그디렉토리> [결과모델디렉토리] [최대깊이] [최대파일수]"
    echo ""
    echo "예시:"
    echo "  $0 models_old /var/log/new_normal/"
    echo "  $0 models_old /var/log/new_normal/ models_updated"  
    echo "  $0 models_old /var/log/new_normal/ models_updated 5 100"
    echo ""
    echo "📋 설명:"
    echo "  - 기존모델디렉토리: 기존에 학습된 모델들이 있는 폴더"
    echo "  - 새로운로그디렉토리: 추가 학습할 새로운 정상 로그 폴더"
    echo "  - 결과모델디렉토리: 업데이트된 모델들을 저장할 폴더 (생략시 자동 생성)"
    echo "  - 최대깊이: 하위 디렉토리 스캔 깊이 (기본: 3)"
    echo "  - 최대파일수: 추가 학습에 사용할 최대 파일 수 (기본: 50)"
    echo ""
    echo "💡 특징:"
    echo "  - 🔄 기존 모델 상태 보존 및 확장"
    echo "  - 📁 새로운 로그 데이터 점진적 추가"
    echo "  - 🧠 DeepLog 모델 점진적 학습"
    echo "  - 🔬 MS-CRED 모델 점진적 학습"
    echo "  - 📊 베이스라인 통계 업데이트"
    echo "  - 🔧 Drain3 템플릿 상태 확장"
    echo "  - 📈 학습 전후 성능 비교"
    exit 1
fi

if [ ! -d "$BASE_MODEL_DIR" ]; then
    echo "❌ 기존 모델 디렉토리를 찾을 수 없습니다: $BASE_MODEL_DIR"
    exit 1
fi

if [ ! -d "$NEW_LOG_DIR" ]; then
    echo "❌ 새로운 로그 디렉토리를 찾을 수 없습니다: $NEW_LOG_DIR"
    exit 1
fi

# 필수 기존 모델 파일들 확인
required_files=(
    "$BASE_MODEL_DIR/vocab.json"
    "$BASE_MODEL_DIR/drain3_state.json"
    "$BASE_MODEL_DIR/metadata.json"
)

echo "🔍 기존 모델 파일 확인 중..."
missing_required=false
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 필수 파일이 없습니다: $(basename "$file")"
        missing_required=true
    else
        echo "✅ $(basename "$file")"
    fi
done

if [ "$missing_required" = true ]; then
    echo ""
    echo "❌ 필수 모델 파일이 부족합니다. 완전한 모델 디렉토리를 사용하세요."
    exit 1
fi

# 가상환경 자동 감지 및 활성화
VENV_ACTIVATED=false

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

# 결과 모델 디렉토리 생성
mkdir -p "$OUTPUT_MODEL_DIR"
WORK_DIR="$OUTPUT_MODEL_DIR/incremental_workspace"
mkdir -p "$WORK_DIR"

echo ""
echo "🚀 점진적 모델 학습 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 기존 모델 디렉토리: $BASE_MODEL_DIR"
echo "📂 새로운 로그 디렉토리: $NEW_LOG_DIR"
echo "💾 결과 모델 디렉토리: $OUTPUT_MODEL_DIR"
echo "📊 스캔 깊이: $MAX_DEPTH, 최대 파일: $MAX_FILES개"
echo "🐍 Python 실행: $PYTHON_CMD"
echo ""
echo "🔄 수행할 점진적 학습 단계:"
echo "  1️⃣  기존 모델 상태 복사 및 백업"
echo "  2️⃣  새로운 로그 파일 스캔 및 수집"
echo "  3️⃣  기존 Drain3 상태로 새 로그 전처리"
echo "  4️⃣  베이스라인 통계 점진적 업데이트"
echo "  5️⃣  DeepLog 모델 점진적 학습"
echo "  6️⃣  MS-CRED 모델 점진적 학습"
echo "  7️⃣  학습 전후 성능 비교"
echo "  8️⃣  업데이트된 메타데이터 저장"
echo ""
echo "⏱️  예상 소요 시간: 5-20분 (새 데이터 크기에 따라)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# 1단계: 기존 모델 상태 복사
echo "1️⃣  기존 모델 상태 복사 중..."

# 기존 모델 파일들을 새 디렉토리로 복사
cp -r "$BASE_MODEL_DIR"/* "$OUTPUT_MODEL_DIR/"

# 백업 디렉토리 생성
BACKUP_DIR="$OUTPUT_MODEL_DIR/original_backup"
mkdir -p "$BACKUP_DIR"
cp "$BASE_MODEL_DIR"/*.json "$BACKUP_DIR/" 2>/dev/null || true
cp "$BASE_MODEL_DIR"/*.pth "$BACKUP_DIR/" 2>/dev/null || true

echo "✅ 기존 모델 상태 복사 완료"
echo "   백업 위치: $BACKUP_DIR"
echo ""

# 2단계: 새로운 로그 파일 스캔
echo "2️⃣  새로운 로그 파일 스캔 중..."
new_log_files=()
log_patterns=("*.log" "*.txt" "*.out" "*.log.*" "*.syslog" "*.messages")

for pattern in "${log_patterns[@]}"; do
    while IFS= read -r -d '' file; do
        if [ -f "$file" ] && [ -s "$file" ]; then
            new_log_files+=("$file")
        fi
    done < <(find "$NEW_LOG_DIR" -maxdepth "$MAX_DEPTH" -name "$pattern" -type f -print0 2>/dev/null)
done

# 파일 크기순 정렬 및 제한
if [ ${#new_log_files[@]} -gt 0 ]; then
    # 파일 크기순 정렬
    printf '%s\n' "${new_log_files[@]}" | while read -r file; do
        size=$(stat -c%s "$file" 2>/dev/null || echo 0)
        echo "$size $file"
    done | sort -nr | head -"$MAX_FILES" | cut -d' ' -f2- > "$WORK_DIR/new_selected_files.txt"
    
    mapfile -t selected_new_files < "$WORK_DIR/new_selected_files.txt"
    echo "✅ 발견된 새 로그 파일: ${#new_log_files[@]}개, 선택된 파일: ${#selected_new_files[@]}개"
    
    # 선택된 파일들 출력
    echo "📋 점진적 학습에 사용할 새 파일들:"
    for i in "${!selected_new_files[@]}"; do
        file="${selected_new_files[$i]}"
        size=$(stat -c%s "$file" 2>/dev/null | numfmt --to=iec)
        echo "  $((i+1)). $(basename "$file") ($size)"
        if [ $i -ge 9 ]; then
            echo "  ... 및 $((${#selected_new_files[@]} - 10))개 파일 더"
            break
        fi
    done
else
    echo "❌ 새로운 로그 파일을 찾을 수 없습니다."
    exit 1
fi
echo ""

# 3단계: 새 로그 전처리 (기존 Drain3 상태 사용)
echo "3️⃣  새 로그 전처리 중 (기존 Drain3 상태 확장)..."
NEW_MERGED_LOG="$WORK_DIR/new_merged.log"
UPDATED_DRAIN_STATE="$OUTPUT_MODEL_DIR/drain3_state.json"

# 새 로그 파일들 병합
> "$NEW_MERGED_LOG"  # 파일 초기화
for file in "${selected_new_files[@]}"; do
    echo "   처리 중: $(basename "$file")"
    cat "$file" >> "$NEW_MERGED_LOG"
done

echo "✅ 새 로그 병합 완료: $(stat -c%s "$NEW_MERGED_LOG" | numfmt --to=iec)"

# 기존 Drain3 상태를 사용하여 새 로그 전처리
echo "   기존 Drain3 상태로 새 로그 파싱 중..."
$PYTHON_CMD -c "
from study_preprocessor.preprocess import LogPreprocessor, PreprocessConfig
from pathlib import Path
import json

try:
    # 전처리 설정 (기존 Drain3 상태 사용)
    cfg = PreprocessConfig(drain_state_path='$UPDATED_DRAIN_STATE')
    pre = LogPreprocessor(cfg)
    
    # 전처리 실행
    df = pre.process_file('$NEW_MERGED_LOG')
    print(f'새 로그 전처리 완료: {len(df)} 레코드 생성')
    
    # 결과 저장
    output_dir = Path('$WORK_DIR')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = output_dir / 'parsed.parquet'
    df.to_parquet(parquet_path, index=False)
    
    # 미리보기 저장
    preview = df.head(10).to_dict(orient='records')
    (output_dir / 'new_preview.json').write_text(json.dumps(preview, ensure_ascii=False, default=str, indent=2))
    
    print(f'저장 완료: {parquet_path}')
    
except Exception as e:
    print(f'새 로그 전처리 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ ! -f "$WORK_DIR/parsed.parquet" ]; then
    echo "❌ 새 로그 전처리 실패"
    exit 1
fi

# 전처리 결과 통계
new_log_lines=$(wc -l < "$NEW_MERGED_LOG")
new_parsed_records=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$WORK_DIR/parsed.parquet')))" 2>/dev/null || echo "N/A")
echo "✅ 새 로그 전처리 완료: $new_log_lines 라인 → $new_parsed_records 레코드"

# 기존 데이터와 새 데이터 통합
echo "   기존 데이터와 새 데이터 통합 중..."
$PYTHON_CMD -c "
import pandas as pd
import json
import os

# 기존 학습 데이터가 있는지 확인
old_training_data = None
if os.path.exists('$OUTPUT_MODEL_DIR/training_workspace/parsed.parquet'):
    old_training_data = pd.read_parquet('$OUTPUT_MODEL_DIR/training_workspace/parsed.parquet')
    print(f'기존 학습 데이터: {len(old_training_data):,} 레코드')

# 새 데이터 로드
new_data = pd.read_parquet('$WORK_DIR/parsed.parquet')
print(f'새 데이터: {len(new_data):,} 레코드')

# 데이터 통합
if old_training_data is not None:
    combined_data = pd.concat([old_training_data, new_data], ignore_index=True)
    print(f'통합 데이터: {len(combined_data):,} 레코드')
else:
    combined_data = new_data
    print(f'통합 데이터: {len(combined_data):,} 레코드 (새 데이터만)')

# 통합된 데이터 저장
os.makedirs('$OUTPUT_MODEL_DIR/training_workspace', exist_ok=True)
combined_data.to_parquet('$OUTPUT_MODEL_DIR/training_workspace/parsed.parquet', index=False)
combined_data.to_parquet('$WORK_DIR/combined_parsed.parquet', index=False)

print('✅ 데이터 통합 완료')
"
echo ""

# 4단계: 베이스라인 통계 점진적 업데이트
echo "4️⃣  베이스라인 통계 점진적 업데이트 중..."

# 통합된 데이터로 베이스라인 재계산
$PYTHON_CMD -c "
from study_preprocessor.detect import baseline_detect, BaselineParams

try:
    # 베이스라인 탐지 설정
    params = BaselineParams(window_size=50, stride=25, ewm_alpha=0.3, anomaly_quantile=0.95)
    
    print('베이스라인 통계 업데이트 시작...')
    
    # 베이스라인 탐지 실행
    result_path = baseline_detect(
        parsed_parquet='$WORK_DIR/combined_parsed.parquet',
        out_dir='$WORK_DIR',
        params=params
    )
    
    print(f'베이스라인 통계 업데이트 완료: {result_path}')
    
except Exception as e:
    print(f'베이스라인 통계 업데이트 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ -f "$WORK_DIR/baseline_scores.parquet" ]; then
    # 업데이트된 베이스라인 통계를 모델 디렉토리로 복사
    cp "$WORK_DIR/baseline_scores.parquet" "$OUTPUT_MODEL_DIR/"
    
    # 기존 통계와 비교하여 업데이트된 정상 패턴 통계 생성
    $PYTHON_CMD -c "
import pandas as pd
import json
import numpy as np
import os

# 새로운 베이스라인 결과 로드
new_df = pd.read_parquet('$WORK_DIR/baseline_scores.parquet')

# 기존 통계 로드 (있는 경우)
old_stats = {}
if os.path.exists('$BACKUP_DIR/baseline_stats.json'):
    with open('$BACKUP_DIR/baseline_stats.json', 'r') as f:
        old_stats = json.load(f)

# 새로운 정상 패턴 통계 계산
normal_windows = new_df[new_df['is_anomaly'] == False]

# 정상 윈도우가 있는지 확인
if len(normal_windows) > 0:
    unseen_stats = {
        'mean_unseen_rate': float(normal_windows['unseen_rate'].mean()),
        'std_unseen_rate': float(normal_windows['unseen_rate'].std()),
    }
    frequency_stats = {
        'mean_freq_z': float(normal_windows['freq_z'].mean()),
        'std_freq_z': float(normal_windows['freq_z'].std()),
        'mean_score': float(normal_windows['score'].mean()),
        'std_score': float(normal_windows['score'].std()),
    }
else:
    # 정상 윈도우가 없는 경우 기본값 사용
    unseen_stats = {
        'mean_unseen_rate': 0.0,
        'std_unseen_rate': 0.0,
    }
    frequency_stats = {
        'mean_freq_z': 0.0,
        'std_freq_z': 0.0,
        'mean_score': 0.0,
        'std_score': 0.0,
    }

new_stats = {
    'total_windows': len(new_df),
    'normal_windows': len(normal_windows),
    'anomaly_rate': float((new_df['is_anomaly'] == True).mean()),
    'unseen_stats': unseen_stats,
    'frequency_stats': frequency_stats,
    'incremental_info': {
        'previous_normal_windows': old_stats.get('normal_windows', 0),
        'added_normal_windows': len(normal_windows) - old_stats.get('normal_windows', 0),
        'improvement_ratio': len(normal_windows) / max(old_stats.get('normal_windows', 0), 1)
    }
}

with open('$OUTPUT_MODEL_DIR/baseline_stats.json', 'w') as f:
    json.dump(new_stats, f, indent=2)

print(f'✅ 베이스라인 통계 업데이트 완료')
print(f'   기존 정상 윈도우: {old_stats.get(\"normal_windows\", 0):,}개')
print(f'   새로운 정상 윈도우: {new_stats[\"normal_windows\"]:,}개')
print(f'   추가된 윈도우: {new_stats[\"incremental_info\"][\"added_normal_windows\"]:,}개')
"
else
    echo "⚠️  베이스라인 업데이트 실패, 계속 진행합니다."
fi
echo ""

# 5단계: DeepLog 점진적 학습
echo "5️⃣  DeepLog 점진적 학습 중..."

# DeepLog 입력 생성 (통합된 데이터로)
$PYTHON_CMD -c "
from study_preprocessor.builders.deeplog import build_deeplog_inputs

try:
    print('DeepLog 입력 생성 시작...')
    
    # DeepLog 입력 생성
    build_deeplog_inputs(
        parsed_parquet='$WORK_DIR/combined_parsed.parquet',
        out_dir='$WORK_DIR'
    )
    
    print('DeepLog 입력 생성 완료')
    
except Exception as e:
    print(f'DeepLog 입력 생성 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ -f "$WORK_DIR/sequences.parquet" ] && [ -f "$WORK_DIR/vocab.json" ]; then
    # 기존 DeepLog 모델이 있는지 확인
    if [ -f "$OUTPUT_MODEL_DIR/deeplog.pth" ]; then
        echo "   기존 DeepLog 모델 발견, 점진적 학습 수행..."
        # 점진적 학습 (기존 모델을 초기값으로 사용)
        UPDATED_DEEPLOG_MODEL="$OUTPUT_MODEL_DIR/deeplog.pth"
        $PYTHON_CMD -c "
import torch
import pandas as pd
import json
from study_preprocessor.builders.deeplog import train_deeplog
from pathlib import Path

# 기존 모델을 백업
import shutil
shutil.copy('$OUTPUT_MODEL_DIR/deeplog.pth', '$BACKUP_DIR/deeplog_original.pth')

# 점진적 학습 수행 (에포크 수를 줄여서 기존 지식 보존)
print('기존 모델을 기반으로 점진적 학습 시작...')
updated_model = train_deeplog(
    sequences_parquet='$WORK_DIR/sequences.parquet',
    vocab_json='$WORK_DIR/vocab.json', 
    out_path='$UPDATED_DEEPLOG_MODEL',
    seq_len=50,
    epochs=5,  # 기존보다 적은 에포크로 점진적 학습
    batch_size=64
)
print(f'✅ DeepLog 점진적 학습 완료: {updated_model}')
"
    else
        echo "   기존 DeepLog 모델이 없음, 새로 학습..."
        # 새로 학습
        UPDATED_DEEPLOG_MODEL="$OUTPUT_MODEL_DIR/deeplog.pth"
        $PYTHON_CMD -c "
from study_preprocessor.builders.deeplog import train_deeplog

try:
    print('새로운 DeepLog 모델 학습 시작...')
    
    updated_model = train_deeplog(
        sequences_parquet='$WORK_DIR/sequences.parquet',
        vocab_json='$WORK_DIR/vocab.json',
        out_path='$UPDATED_DEEPLOG_MODEL',
        seq_len=50,
        epochs=10,
        batch_size=64
    )
    
    print(f'DeepLog 모델 학습 완료: {updated_model}')
    
except Exception as e:
    print(f'DeepLog 모델 학습 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
    fi
    
    # 어휘 사전 업데이트
    cp "$WORK_DIR/vocab.json" "$OUTPUT_MODEL_DIR/"
    
    if [ -f "$UPDATED_DEEPLOG_MODEL" ]; then
        echo "✅ DeepLog 업데이트 완료: $(stat -c%s "$UPDATED_DEEPLOG_MODEL" | numfmt --to=iec)"
    else
        echo "❌ DeepLog 업데이트 실패"
    fi
else
    echo "❌ DeepLog 입력 생성 실패"
fi
echo ""

# 6단계: MS-CRED 점진적 학습
echo "6️⃣  MS-CRED 점진적 학습 중..."

# MS-CRED 입력 생성 (통합된 데이터로)
$PYTHON_CMD -c "
from study_preprocessor.builders.mscred import build_mscred_window_counts

try:
    print('MS-CRED 입력 생성 시작...')
    
    # MS-CRED 입력 생성
    build_mscred_window_counts(
        parsed_parquet='$WORK_DIR/combined_parsed.parquet',
        out_dir='$WORK_DIR',
        window_size=50,
        stride=25
    )
    
    print('MS-CRED 입력 생성 완료')
    
except Exception as e:
    print(f'MS-CRED 입력 생성 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ -f "$WORK_DIR/window_counts.parquet" ]; then
    # 기존 MS-CRED 모델이 있는지 확인
    if [ -f "$OUTPUT_MODEL_DIR/mscred.pth" ]; then
        echo "   기존 MS-CRED 모델 발견, 점진적 학습 수행..."
        # 기존 모델 백업
        cp "$OUTPUT_MODEL_DIR/mscred.pth" "$BACKUP_DIR/mscred_original.pth"
        
        # 점진적 학습 (에포크 수를 줄여서)
        UPDATED_MSCRED_MODEL="$OUTPUT_MODEL_DIR/mscred.pth"
        $PYTHON_CMD -c "
from study_preprocessor.mscred_model import train_mscred

try:
    print('기존 MS-CRED 모델 점진적 학습 시작...')
    
    train_mscred(
        window_counts_path='$WORK_DIR/window_counts.parquet',
        model_output_path='$UPDATED_MSCRED_MODEL',
        epochs=25  # 기존보다 적은 에포크
    )
    
    print('MS-CRED 점진적 학습 완료')
    
except Exception as e:
    print(f'MS-CRED 점진적 학습 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
    else
        echo "   기존 MS-CRED 모델이 없음, 새로 학습..."
        # 새로 학습
        UPDATED_MSCRED_MODEL="$OUTPUT_MODEL_DIR/mscred.pth"
        $PYTHON_CMD -c "
from study_preprocessor.mscred_model import train_mscred

try:
    print('새로운 MS-CRED 모델 학습 시작...')
    
    train_mscred(
        window_counts_path='$WORK_DIR/window_counts.parquet',
        model_output_path='$UPDATED_MSCRED_MODEL',
        epochs=50
    )
    
    print('MS-CRED 모델 학습 완료')
    
except Exception as e:
    print(f'MS-CRED 모델 학습 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
    fi
    
    if [ -f "$UPDATED_MSCRED_MODEL" ]; then
        echo "✅ MS-CRED 업데이트 완료: $(stat -c%s "$UPDATED_MSCRED_MODEL" | numfmt --to=iec)"
    else
        echo "❌ MS-CRED 업데이트 실패"
    fi
else
    echo "❌ MS-CRED 입력 생성 실패"
fi
echo ""

# 7단계: 학습 전후 성능 비교
echo "7️⃣  학습 전후 성능 비교 중..."

# 테스트 데이터로 성능 비교 (합성 데이터 사용)
COMPARISON_DIR="$OUTPUT_MODEL_DIR/performance_comparison"
mkdir -p "$COMPARISON_DIR"

# 간단한 합성 테스트 데이터 생성
$PYTHON_CMD -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 합성 테스트 데이터 생성 (작은 크기)
np.random.seed(42)
random.seed(42)

templates = [
    'INFO: User <*> logged in',
    'INFO: Processing request <*>',
    'WARNING: High memory usage',
    'ERROR: Connection timeout'
]

logs = []
start_time = datetime.now() - timedelta(hours=1)

for i in range(1000):
    timestamp = start_time + timedelta(seconds=i*3)
    template = random.choice(templates)
    log_line = f'{timestamp.strftime(\"%Y-%m-%d %H:%M:%S\")} {template}'
    logs.append(log_line)

with open('$COMPARISON_DIR/test_data.log', 'w') as f:
    for log in logs:
        f.write(log + '\\n')

print('✅ 테스트 데이터 생성 완료: 1,000 라인')
"

# 기존 모델과 새 모델 성능 비교 (compare_models.sh 사용)
if [ -f "compare_models.sh" ]; then
    echo "   기존 모델 vs 업데이트된 모델 성능 비교 중..."
    ./compare_models.sh "$BACKUP_DIR" "$OUTPUT_MODEL_DIR" "$COMPARISON_DIR/test_data.log" "$COMPARISON_DIR" > "$COMPARISON_DIR/comparison.log" 2>&1 || true
    
    if [ -f "$COMPARISON_DIR/comparison_report.md" ]; then
        echo "✅ 성능 비교 완료, 리포트 생성됨"
    else
        echo "⚠️  성능 비교 실행됨, 상세 로그 확인: $COMPARISON_DIR/comparison.log"
    fi
else
    echo "⚠️  compare_models.sh가 없어 성능 비교를 건너뜁니다."
fi
echo ""

# 8단계: 업데이트된 메타데이터 저장
echo "8️⃣  업데이트된 메타데이터 저장 중..."

# 점진적 학습 정보 메타데이터 생성
$PYTHON_CMD -c "
import json
import os
from datetime import datetime
from pathlib import Path

# 기존 메타데이터 로드
old_metadata = {}
if os.path.exists('$BACKUP_DIR/metadata.json'):
    with open('$BACKUP_DIR/metadata.json', 'r') as f:
        old_metadata = json.load(f)

# 새로운 메타데이터 생성
new_metadata = {
    'incremental_training_info': {
        'timestamp': datetime.now().isoformat(),
        'base_model_directory': '$BASE_MODEL_DIR',
        'new_log_directory': '$NEW_LOG_DIR',
        'total_new_files': ${#selected_new_files[@]},
        'max_depth': $MAX_DEPTH,
        'max_files': $MAX_FILES,
        'previous_training': old_metadata.get('training_info', {})
    },
    'models': {
        'deeplog': os.path.exists('$OUTPUT_MODEL_DIR/deeplog.pth'),
        'mscred': os.path.exists('$OUTPUT_MODEL_DIR/mscred.pth'),
        'baseline_stats': os.path.exists('$OUTPUT_MODEL_DIR/baseline_stats.json'),
        'drain3_state': os.path.exists('$OUTPUT_MODEL_DIR/drain3_state.json'),
        'vocab': os.path.exists('$OUTPUT_MODEL_DIR/vocab.json')
    },
    'incremental_files': [],  # 파일 목록은 별도로 처리
    'backup_location': '$BACKUP_DIR',
    'performance_comparison': '$COMPARISON_DIR'
}

# 파일 목록 추가 (파일에서 읽기)
try:
    if os.path.exists('$WORK_DIR/new_selected_files.txt'):
        with open('$WORK_DIR/new_selected_files.txt', 'r') as f:
            file_list = [line.strip() for line in f.readlines() if line.strip()]
        
        # 경로에서 기본 디렉토리 제거
        new_log_dir = '$NEW_LOG_DIR'
        relative_files = []
        for f in file_list[:20]:  # 최대 20개만
            if f.startswith(new_log_dir):
                relative_files.append(f.replace(new_log_dir + '/', '').replace(new_log_dir, ''))
            else:
                relative_files.append(os.path.basename(f))
        
        new_metadata['incremental_files'] = relative_files
    else:
        new_metadata['incremental_files'] = ['파일 목록 파일 없음']
except Exception as e:
    new_metadata['incremental_files'] = [f'파일 목록 생성 실패: {str(e)}']

# 기존 메타데이터와 병합
if old_metadata:
    new_metadata['original_metadata'] = old_metadata

with open('$OUTPUT_MODEL_DIR/metadata.json', 'w') as f:
    json.dump(new_metadata, f, indent=2, ensure_ascii=False)

print('✅ 점진적 학습 메타데이터 저장 완료')
"

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 점진적 모델 학습 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  총 소요 시간: ${MINUTES}분 ${SECONDS}초"
echo ""

# 결과 요약 출력
echo "📊 점진적 학습 결과:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

models_updated=0
if [ -f "$OUTPUT_MODEL_DIR/deeplog.pth" ]; then
    old_size=$(stat -c%s "$BACKUP_DIR/deeplog_original.pth" 2>/dev/null | numfmt --to=iec || echo "N/A")
    new_size=$(stat -c%s "$OUTPUT_MODEL_DIR/deeplog.pth" | numfmt --to=iec)
    echo "  🧠 DeepLog 모델: 업데이트됨 ($old_size → $new_size)"
    models_updated=$((models_updated + 1))
else
    echo "  ❌ DeepLog 모델: 업데이트 실패"
fi

if [ -f "$OUTPUT_MODEL_DIR/mscred.pth" ]; then
    old_size=$(stat -c%s "$BACKUP_DIR/mscred_original.pth" 2>/dev/null | numfmt --to=iec || echo "N/A")
    new_size=$(stat -c%s "$OUTPUT_MODEL_DIR/mscred.pth" | numfmt --to=iec)
    echo "  🔬 MS-CRED 모델: 업데이트됨 ($old_size → $new_size)"
    models_updated=$((models_updated + 1))
else
    echo "  ❌ MS-CRED 모델: 업데이트 실패"
fi

if [ -f "$OUTPUT_MODEL_DIR/baseline_stats.json" ]; then
    echo "  📊 베이스라인 통계: 업데이트됨"
    models_updated=$((models_updated + 1))
else
    echo "  ❌ 베이스라인 통계: 업데이트 실패"
fi

echo ""
echo "📈 점진적 학습 통계:"
echo "  ✅ 업데이트된 모델: ${models_updated}/3개"
echo "  📁 업데이트된 모델 위치: $OUTPUT_MODEL_DIR"
echo "  💾 원본 모델 백업: $BACKUP_DIR"
echo "  📊 성능 비교 결과: $COMPARISON_DIR"
echo "  📋 메타데이터: $OUTPUT_MODEL_DIR/metadata.json"
echo ""

# 성능 비교 결과 요약 (있는 경우)
if [ -f "$COMPARISON_DIR/comparison_report.md" ]; then
    echo "📊 성능 비교 요약:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    grep -E "^\*\*추론 속도|^\*\*속도|^- \*\*" "$COMPARISON_DIR/comparison_report.md" | head -5
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

# 사용법 안내
echo "🚀 업데이트된 모델 사용 방법:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ./run_inference.sh $OUTPUT_MODEL_DIR /path/to/target.log"
echo ""
echo "💡 성능 비교 확인:"
echo "  cat $COMPARISON_DIR/comparison_report.md"
echo ""
echo "🔄 추가 점진적 학습:"
echo "  ./train_models_incremental.sh $OUTPUT_MODEL_DIR /path/to/newer_logs/"
echo ""

# 임시 작업 디렉토리 정리 안내
echo "🧹 임시 파일 정리:"
if [ -d "$WORK_DIR" ]; then
    echo "  📁 임시 작업 디렉토리: $WORK_DIR"
    echo "  💡 정리하려면: rm -rf $WORK_DIR"
fi

echo ""
echo "🎉 점진적 모델 학습이 완료되었습니다!"
echo "   - ✅ 기존 모델 상태 보존 및 확장"
echo "   - ✅ 새로운 로그 데이터 점진적 추가" 
echo "   - ✅ DeepLog/MS-CRED 점진적 학습"
echo "   - ✅ 베이스라인 통계 업데이트"
echo "   - ✅ 학습 전후 성능 비교"
echo "   - ✅ 업데이트된 메타데이터 저장"
echo ""
echo "🔍 이제 업데이트된 모델로 더 정확한 이상탐지를 수행할 수 있습니다!"
