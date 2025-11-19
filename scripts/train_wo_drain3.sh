#!/bin/bash

# 로그 이상탐지 모델 학습 스크립트 (Drain3 없이 Regex-Only)
# 정상 로그 데이터로부터 DeepLog, MS-CRED 모델과 베이스라인 통계를 학습합니다.
# Python과 C 엔진이 완전히 동일한 템플릿 추출 방식을 사용합니다.
# 사용법: ./train_wo_drain3.sh <로그디렉토리> [모델저장디렉토리] [최대깊이] [최대파일수]

set -e  # 에러 발생시 즉시 중단

# 기본값 설정
LOG_DIR="$1"
MODEL_DIR="${2:-models_wo_drain3_$(date +%Y%m%d_%H%M%S)}"
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
    echo "💡 특징 (Drain3 없이):"
    echo "  - 📁 하위 디렉토리 자동 재귀 스캔"
    echo "  - 🧠 DeepLog LSTM 모델 학습"
    echo "  - 🔬 MS-CRED 멀티스케일 컨볼루션 모델 학습"
    echo "  - 📊 베이스라인 통계 (정상 패턴) 학습"
    echo "  - 🔧 Regex-only 전처리 (Python과 C 엔진 완전 동일)"
    echo "  - ⚡ Wildcard matching 불필요 (템플릿이 리터럴 값만 포함)"
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
echo "🔍 패키지 상태 확인 중..."
if ! $PYTHON_CMD -c "import anomaly_log_detector" 2>/dev/null; then
    echo "🔧 anomaly_log_detector 패키지 설치 중..."

    # 가상환경에서 pip 사용
    if [ "$VENV_ACTIVATED" = true ] && [ -n "$VIRTUAL_ENV" ]; then
        pip install -e . || {
            echo "❌ 패키지 설치 실패 (pip)"
            exit 1
        }
    elif [ -f ".venv/bin/pip" ]; then
        .venv/bin/pip install -e . || {
            echo "❌ 패키지 설치 실패 (.venv/bin/pip)"
            exit 1
        }
    else
        echo "❌ 적절한 pip을 찾을 수 없습니다."
        echo "🔍 디버깅 정보:"
        echo "   - VENV_ACTIVATED: $VENV_ACTIVATED"
        echo "   - VIRTUAL_ENV: $VIRTUAL_ENV"
        echo "   - .venv/bin/pip 존재: $([ -f ".venv/bin/pip" ] && echo "예" || echo "아니오")"
        exit 1
    fi
    echo "✅ 패키지 설치 완료"
else
    echo "✅ anomaly_log_detector 패키지 이미 설치됨"
fi

# 필수 의존성 확인 (drain3 제외)
echo "🔍 필수 의존성 확인 중..."
missing_deps=()
for dep in "pandas" "torch"; do
    if ! $PYTHON_CMD -c "import $dep" 2>/dev/null; then
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -gt 0 ]; then
    echo "❌ 누락된 의존성: ${missing_deps[*]}"
    echo "🔧 의존성 설치 중..."
    if [ "$VENV_ACTIVATED" = true ]; then
        pip install -r requirements.txt || {
            echo "❌ 의존성 설치 실패"
            exit 1
        }
    else
        .venv/bin/pip install -r requirements.txt || {
            echo "❌ 의존성 설치 실패"
            exit 1
        }
    fi
    echo "✅ 의존성 설치 완료"
else
    echo "✅ 모든 필수 의존성 확인됨"
fi

# 모델 저장 디렉토리 생성
mkdir -p "$MODEL_DIR"
WORK_DIR="$MODEL_DIR/training_workspace"
mkdir -p "$WORK_DIR"

echo "🚀 로그 이상탐지 모델 학습 시작 (Regex-Only, Drain3 없이)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 학습 로그 디렉토리: $LOG_DIR"
echo "💾 모델 저장 디렉토리: $MODEL_DIR"
echo "📊 스캔 깊이: $MAX_DEPTH, 최대 파일: $MAX_FILES개"
echo "🐍 Python 실행: $PYTHON_CMD"
echo ""
echo "🔄 수행할 학습 단계:"
echo "  1️⃣  로그 파일 스캔 및 수집"
echo "  2️⃣  로그 전처리 (Regex-only, NO Drain3)"
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
log_patterns=("*.log" "*.txt" "*.out" "*.log.*" "*.syslog" "*.messages" "*.log*")

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
    for i in "${!selected_files[@]}"; do
        file="${selected_files[$i]}"
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

# 2단계: 로그 병합 및 전처리 (Regex-only, NO Drain3)
echo "2️⃣  로그 전처리 중 (Regex-only, NO Drain3)..."
MERGED_LOG="$WORK_DIR/merged_training.log"

# 로그 파일들 병합
> "$MERGED_LOG"  # 파일 초기화
for file in "${selected_files[@]}"; do
    echo "   처리 중: $(basename "$file")"
    cat "$file" >> "$MERGED_LOG"
done

echo "✅ 병합된 로그 크기: $(stat -c%s "$MERGED_LOG" | numfmt --to=iec)"

# Regex-only 전처리 (NO Drain3)
echo "   Regex-only 전처리 중..."
echo "   📝 명령어: $PYTHON_CMD -c \"from anomaly_log_detector.preprocess import ...\""

# 전처리 실행 (Regex-only)
if ! $PYTHON_CMD -c "
from anomaly_log_detector.preprocess import parse_line, mask_message, PreprocessConfig
from pathlib import Path
import pandas as pd
import json

try:
    # Regex-only 전처리 설정 (Drain3 없음)
    cfg = PreprocessConfig(
        mask_dates=True,
        mask_paths=True,
        mask_hex=True,
        mask_ips=True,
        mask_mac=True,
        mask_uuid=True,
        mask_pid_fields=True,
        mask_device_numbers=True,
        mask_numbers=True,
    )

    rows = []

    with open('$MERGED_LOG', 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            ts, host, proc, msg = parse_line(line)

            # Apply ONLY regex masking (NO Drain3)
            template = mask_message(msg, cfg)

            rows.append({
                'line_no': idx,
                'timestamp': ts,
                'host': host,
                'process': proc,
                'raw': msg,
                'template': template,  # This is identical to C normalization
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['timestamp', 'line_no'], kind='stable', na_position='first')

    # Assign template indices (sorted order for consistency)
    unique_templates = sorted(df['template'].dropna().unique())
    template_to_idx = {tpl: idx for idx, tpl in enumerate(unique_templates)}

    df['template_index'] = df['template'].map(template_to_idx)

    print(f'전처리 완료: {len(df)} 레코드, {len(unique_templates)} 고유 템플릿 (Regex-only)')

    # 결과 저장
    output_dir = Path('$WORK_DIR')
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / 'parsed.parquet'
    df.to_parquet(parquet_path, index=False)

    # vocab.json 저장 (C format)
    vocab_c = {str(idx): tpl for tpl, idx in template_to_idx.items()}
    vocab_path = output_dir / 'vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab_c, f, indent=2, ensure_ascii=False)

    # 미리보기 저장
    preview = df.head(10).to_dict(orient='records')
    (output_dir / 'preview.json').write_text(json.dumps(preview, ensure_ascii=False, default=str, indent=2))

    print(f'저장 완료: {parquet_path}')
    print(f'어휘 저장: {vocab_path}')
    print(f'')
    print(f'=== Sample templates (first 10) ===')
    for i, tpl in enumerate(unique_templates[:10]):
        print(f'  {i}: {tpl[:80]}')

except Exception as e:
    print(f'전처리 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1; then
    echo "❌ 전처리 실행 실패"
    echo "🔍 디버깅 정보:"
    echo "   - 병합된 로그 파일: $MERGED_LOG ($([ -f "$MERGED_LOG" ] && echo "존재" || echo "없음"))"
    echo "   - 작업 디렉토리: $WORK_DIR ($([ -d "$WORK_DIR" ] && echo "존재" || echo "없음"))"
    echo "   - Python 명령어: $PYTHON_CMD"
    exit 1
fi

# 결과 파일 확인
if [ ! -f "$WORK_DIR/parsed.parquet" ]; then
    echo "❌ 전처리 실패: parsed.parquet 파일이 생성되지 않았습니다."
    echo "🔍 생성된 파일들:"
    ls -la "$WORK_DIR/" 2>/dev/null || echo "   작업 디렉토리가 비어있습니다."
    exit 1
fi

# vocab.json을 모델 디렉토리로 복사
if [ -f "$WORK_DIR/vocab.json" ]; then
    cp "$WORK_DIR/vocab.json" "$MODEL_DIR/"
    echo "✅ 어휘 사전 저장 완료"
fi

echo "✅ 전처리 완료: $(wc -l < "$MERGED_LOG") 라인 → $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$WORK_DIR/parsed.parquet')))" 2>/dev/null || echo "N/A") 레코드"
echo ""

# 3단계: 베이스라인 통계 학습
echo "3️⃣  베이스라인 통계 학습 중..."
$PYTHON_CMD -c "
from anomaly_log_detector.detect import baseline_detect, BaselineParams
from pathlib import Path

try:
    # 베이스라인 탐지 설정
    params = BaselineParams(window_size=50, stride=25, ewm_alpha=0.3, anomaly_quantile=0.95)

    print('베이스라인 탐지 시작...')

    # 베이스라인 탐지 실행
    result_path = baseline_detect(
        parsed_parquet='$WORK_DIR/parsed.parquet',
        out_dir='$WORK_DIR',
        params=params
    )

    print(f'베이스라인 탐지 완료: {result_path}')

except Exception as e:
    print(f'베이스라인 탐지 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

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
print(f'베이스라인 결과 컬럼: {list(df.columns)}')

# 정상 패턴 통계 계산
normal_windows = df[df['is_anomaly'] == False]
stats = {
    'total_windows': len(df),
    'normal_windows': len(normal_windows),
    'anomaly_rate': float((df['is_anomaly'] == True).mean()),
    'unseen_stats': {
        'mean_unseen_rate': float(normal_windows['unseen_rate'].mean()) if len(normal_windows) > 0 else 0.0,
        'std_unseen_rate': float(normal_windows['unseen_rate'].std()) if len(normal_windows) > 0 else 0.0,
    },
    'frequency_stats': {
        'mean_freq_z': float(normal_windows['freq_z'].mean()) if len(normal_windows) > 0 else 0.0,
        'std_freq_z': float(normal_windows['freq_z'].std()) if len(normal_windows) > 0 else 0.0,
        'mean_score': float(normal_windows['score'].mean()) if len(normal_windows) > 0 else 0.0,
        'std_score': float(normal_windows['score'].std()) if len(normal_windows) > 0 else 0.0,
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

# DeepLog 입력 생성 및 모델 학습
$PYTHON_CMD -c "
from anomaly_log_detector.builders.deeplog import build_deeplog_inputs, train_deeplog
import pandas as pd
from pathlib import Path
import json

try:
    print('DeepLog 입력 생성 시작...')

    # DeepLog 입력 생성
    build_deeplog_inputs(
        parsed_parquet='$WORK_DIR/parsed.parquet',
        out_dir='$WORK_DIR'
    )

    # 생성된 파일들 확인
    work_dir = Path('$WORK_DIR')
    sequences_path = work_dir / 'sequences.parquet'
    vocab_path = work_dir / 'vocab.json'

    if sequences_path.exists() and vocab_path.exists():
        # 생성된 데이터 정보 출력
        sequences_df = pd.read_parquet(sequences_path)
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        print(f'DeepLog 입력 생성 완료: {len(sequences_df)} 시퀀스, 어휘 크기: {len(vocab)}')

        # DeepLog 모델 학습
        model_path = '$MODEL_DIR/deeplog.pth'
        print('DeepLog 모델 학습 시작...')

        train_deeplog(
            sequences_parquet=str(sequences_path),
            vocab_json=str(vocab_path),
            out_path=model_path,
            seq_len=50,
            epochs=10,
            batch_size=64
        )

        print(f'DeepLog 모델 학습 완료: {model_path}')
    else:
        print('DeepLog 입력 파일 생성 실패')

except Exception as e:
    print(f'DeepLog 처리 오류: {e}')
    import traceback
    traceback.print_exc()
    # DeepLog 실패해도 계속 진행
"

DEEPLOG_MODEL="$MODEL_DIR/deeplog.pth"
if [ -f "$DEEPLOG_MODEL" ]; then
    echo "✅ DeepLog 학습 완료: $(stat -c%s "$DEEPLOG_MODEL" | numfmt --to=iec)"
else
    echo "❌ DeepLog 학습 실패"
fi
echo ""

# 5단계: MS-CRED 학습
echo "5️⃣  MS-CRED 모델 학습 중..."

# MS-CRED 입력 생성 및 모델 학습
$PYTHON_CMD -c "
from anomaly_log_detector.builders.mscred import build_mscred_window_counts
from anomaly_log_detector.mscred_model import train_mscred
from pathlib import Path

try:
    # MS-CRED 입력 생성
    print('MS-CRED 입력 생성 시작...')
    build_mscred_window_counts(
        parsed_parquet='$WORK_DIR/parsed.parquet',
        out_dir='$WORK_DIR',
        window_size=50,
        stride=25
    )

    window_counts_path = Path('$WORK_DIR') / 'window_counts.parquet'
    if window_counts_path.exists():
        print(f'MS-CRED 입력 생성 완료: {window_counts_path}')

        # MS-CRED 모델 학습
        model_path = '$MODEL_DIR/mscred.pth'
        print('MS-CRED 모델 학습 시작...')

        stats = train_mscred(
            window_counts_path=str(window_counts_path),
            model_output_path=model_path,
            epochs=50
        )

        print(f'MS-CRED 모델 학습 완료: {model_path}')
        print(f'최종 학습 손실: {stats[\"final_train_loss\"]:.4f}')
        print(f'최종 검증 손실: {stats[\"final_val_loss\"]:.4f}')
    else:
        print('MS-CRED 입력 생성 실패')

except Exception as e:
    print(f'MS-CRED 처리 오류: {e}')
    import traceback
    traceback.print_exc()
    # MS-CRED 실패해도 계속 진행
"

MSCRED_MODEL="$MODEL_DIR/mscred.pth"
if [ -f "$MSCRED_MODEL" ]; then
    echo "✅ MS-CRED 학습 완료: $(stat -c%s "$MSCRED_MODEL" | numfmt --to=iec)"
else
    echo "❌ MS-CRED 학습 실패"
fi
echo ""

# 6단계: 메타데이터 저장
echo "6️⃣  학습 메타데이터 저장 중..."

# 선택된 파일 목록을 임시 파일로 저장
printf "%s\n" "${selected_files[@]}" | sed "s|$LOG_DIR/||g" | head -20 > "$WORK_DIR/file_list.txt"

# 환경 변수로 전달
export TRAIN_WORK_DIR="$WORK_DIR"
export TRAIN_LOG_DIR="$LOG_DIR"
export TRAIN_MODEL_DIR="$MODEL_DIR"
export TRAIN_MAX_DEPTH="$MAX_DEPTH"
export TRAIN_MAX_FILES="$MAX_FILES"
export TRAIN_TOTAL_FILES="${#selected_files[@]}"

# 학습 정보 메타데이터 생성
$PYTHON_CMD <<'PYEOF'
import json
import os
from datetime import datetime
from pathlib import Path

# 환경 변수에서 읽기
work_dir = os.environ.get('TRAIN_WORK_DIR', '.')
log_dir = os.environ.get('TRAIN_LOG_DIR', '')
model_dir = os.environ.get('TRAIN_MODEL_DIR', '.')
max_depth = int(os.environ.get('TRAIN_MAX_DEPTH', '3'))
max_files = int(os.environ.get('TRAIN_MAX_FILES', '50'))
total_files = int(os.environ.get('TRAIN_TOTAL_FILES', '0'))

# 파일 목록 읽기
try:
    file_list_path = os.path.join(work_dir, 'file_list.txt')
    with open(file_list_path, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]
except:
    file_list = []

metadata = {
    'training_info': {
        'timestamp': datetime.now().isoformat(),
        'log_directory': log_dir,
        'total_files': total_files,
        'max_depth': max_depth,
        'max_files': max_files,
        'preprocessing': 'regex_only (NO Drain3)',
        'note': 'Python and C engines use identical template extraction'
    },
    'models': {
        'deeplog': os.path.exists(os.path.join(model_dir, 'deeplog.pth')),
        'mscred': os.path.exists(os.path.join(model_dir, 'mscred.pth')),
        'baseline_stats': os.path.exists(os.path.join(model_dir, 'baseline_stats.json')),
        'vocab': os.path.exists(os.path.join(model_dir, 'vocab.json'))
    },
    'files': file_list
}

metadata_path = os.path.join(model_dir, 'metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print('✅ 메타데이터 저장 완료')
PYEOF

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

if [ -f "$MODEL_DIR/vocab.json" ]; then
    echo "  📚 어휘 사전: vocab.json (Regex-only)"
else
    echo "  ❌ 어휘 사전: 저장 실패"
fi

echo ""
echo "📈 학습 결과 통계:"
echo "  ✅ 성공한 모델: ${models_trained}/3개"
echo "  📁 모델 저장 위치: $MODEL_DIR"
echo "  📋 메타데이터: $MODEL_DIR/metadata.json"
echo "  🔧 전처리 방식: Regex-only (NO Drain3)"
echo "  ⚡ Wildcard matching: 불필요 (템플릿이 리터럴 값만 포함)"
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
echo "🎉 모델 학습이 완료되었습니다! (Drain3 없이)"
echo "   - ✅ 로그 전처리 (Regex-only)"
echo "   - ✅ 베이스라인 정상 패턴 학습"
echo "   - ✅ DeepLog LSTM 모델 학습"
echo "   - ✅ MS-CRED 컨볼루션 모델 학습"
echo "   - ✅ 학습 메타데이터 저장"
echo ""
echo "💡 Python과 C 엔진이 완전히 동일한 템플릿 추출 방식을 사용합니다!"
echo "🔍 이제 run_inference.sh로 이상탐지를 수행할 수 있습니다!"
