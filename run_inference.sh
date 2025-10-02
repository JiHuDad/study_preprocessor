#!/bin/bash

# 로그 이상탐지 추론 스크립트
# 학습된 모델들을 사용하여 Target 로그 파일의 이상을 탐지합니다.
# 사용법: ./run_inference.sh <모델디렉토리> <target로그파일> [결과디렉토리]

set -e  # 에러 발생시 즉시 중단

# 기본값 설정
MODEL_DIR="$1"
TARGET_LOG="$2"
RESULT_DIR="${3:-inference_$(date +%Y%m%d_%H%M%S)}"

# 인수 확인
if [ -z "$MODEL_DIR" ] || [ -z "$TARGET_LOG" ]; then
    echo "❌ 사용법: $0 <모델디렉토리> <target로그파일> [결과디렉토리]"
    echo ""
    echo "예시:"
    echo "  $0 models_20241002_143022 /var/log/suspicious.log"
    echo "  $0 models_20241002_143022 /var/log/suspicious.log my_results"
    echo ""
    echo "📋 설명:"
    echo "  - 모델디렉토리: train_models.sh로 생성된 학습 모델 폴더"
    echo "  - target로그파일: 이상탐지를 수행할 로그 파일"
    echo "  - 결과디렉토리: 분석 결과를 저장할 폴더 (생략시 자동 생성)"
    echo ""
    echo "💡 특징:"
    echo "  - 🧠 DeepLog LSTM 이상탐지"
    echo "  - 🔬 MS-CRED 멀티스케일 이상탐지"
    echo "  - 📊 베이스라인 통계 기반 이상탐지"
    echo "  - 🕐 시간 기반 패턴 분석"
    echo "  - 📋 실제 이상 로그 샘플 추출 및 분석"
    echo "  - 📋 종합 이상탐지 리포트 생성"
    echo ""
    echo "📁 필요한 모델 파일들:"
    echo "  - deeplog.pth (DeepLog 모델)"
    echo "  - mscred.pth (MS-CRED 모델)"
    echo "  - vocab.json (어휘 사전)"
    echo "  - baseline_stats.json (베이스라인 통계)"
    echo "  - drain3_state.json (Drain3 상태)"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 모델 디렉토리를 찾을 수 없습니다: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$TARGET_LOG" ]; then
    echo "❌ Target 로그 파일을 찾을 수 없습니다: $TARGET_LOG"
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

# 필수 모델 파일들 확인
required_files=(
    "$MODEL_DIR/vocab.json"
    "$MODEL_DIR/drain3_state.json"
)

optional_files=(
    "$MODEL_DIR/deeplog.pth"
    "$MODEL_DIR/mscred.pth"
    "$MODEL_DIR/baseline_stats.json"
)

echo "🔍 모델 파일 확인 중..."
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
    echo "❌ 필수 모델 파일이 부족합니다. train_models.sh를 먼저 실행하세요."
    exit 1
fi

available_models=()
for file in "${optional_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $(basename "$file")"
        available_models+=("$(basename "$file")")
    else
        echo "⚠️  $(basename "$file") (없음)"
    fi
done

# 결과 디렉토리 생성
mkdir -p "$RESULT_DIR"

echo ""
echo "🚀 로그 이상탐지 추론 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 모델 디렉토리: $MODEL_DIR"
echo "🎯 Target 로그 파일: $TARGET_LOG"
echo "📁 결과 저장 디렉토리: $RESULT_DIR"
echo "🐍 Python 실행: $PYTHON_CMD"
echo ""
echo "🤖 사용 가능한 모델들:"
for model in "${available_models[@]}"; do
    echo "  ✅ $model"
done
echo ""
echo "🔄 수행할 추론 단계:"
echo "  1️⃣  Target 로그 전처리 (기존 Drain3 상태 사용)"
echo "  2️⃣  베이스라인 이상탐지 (학습된 통계와 비교)"
echo "  3️⃣  DeepLog 추론 (LSTM 시퀀스 예측)"
echo "  4️⃣  MS-CRED 추론 (멀티스케일 재구성 오차)"
echo "  5️⃣  시간 기반 이상탐지"
echo "  6️⃣  이상 로그 샘플 추출 및 분석"
echo "  7️⃣  종합 결과 분석 및 리포트 생성"
echo ""
echo "⏱️  예상 소요 시간: 2-10분 (파일 크기에 따라)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# 1단계: Target 로그 전처리
echo "1️⃣  Target 로그 전처리 중..."
TARGET_NAME=$(basename "$TARGET_LOG" .log)

# 기존 Drain3 상태를 사용하여 전처리
$PYTHON_CMD -c "
from study_preprocessor.preprocess import LogPreprocessor, PreprocessConfig
from pathlib import Path
import json

try:
    # 전처리 설정 (기존 Drain3 상태 사용)
    cfg = PreprocessConfig(drain_state_path='$MODEL_DIR/drain3_state.json')
    pre = LogPreprocessor(cfg)
    
    # 전처리 실행
    df = pre.process_file('$TARGET_LOG')
    print(f'Target 로그 전처리 완료: {len(df)} 레코드 생성')
    
    # 결과 저장
    output_dir = Path('$RESULT_DIR')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = output_dir / 'parsed.parquet'
    df.to_parquet(parquet_path, index=False)
    
    # 미리보기 저장
    preview = df.head(10).to_dict(orient='records')
    (output_dir / 'preview.json').write_text(json.dumps(preview, ensure_ascii=False, default=str, indent=2))
    
    print(f'저장 완료: {parquet_path}')
    
except Exception as e:
    print(f'Target 로그 전처리 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ ! -f "$RESULT_DIR/parsed.parquet" ]; then
    echo "❌ Target 로그 전처리 실패"
    exit 1
fi

# 전처리 결과 통계
log_lines=$(wc -l < "$TARGET_LOG")
parsed_records=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$RESULT_DIR/parsed.parquet')))" 2>/dev/null || echo "N/A")
echo "✅ 전처리 완료: $log_lines 라인 → $parsed_records 레코드"
echo ""

# 2단계: 베이스라인 이상탐지
echo "2️⃣  베이스라인 이상탐지 중..."
if [ -f "$MODEL_DIR/baseline_stats.json" ]; then
    # 베이스라인 탐지 실행
    $PYTHON_CMD -c "
from study_preprocessor.detect import baseline_detect, BaselineParams

try:
    # 베이스라인 탐지 설정
    params = BaselineParams(window_size=50, stride=25, ewm_alpha=0.3, anomaly_quantile=0.95)
    
    print('베이스라인 이상탐지 시작...')
    
    # 베이스라인 탐지 실행
    result_path = baseline_detect(
        parsed_parquet='$RESULT_DIR/parsed.parquet',
        out_dir='$RESULT_DIR',
        params=params
    )
    
    print(f'베이스라인 이상탐지 완료: {result_path}')
    
except Exception as e:
    print(f'베이스라인 이상탐지 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
    
    if [ -f "$RESULT_DIR/baseline_scores.parquet" ]; then
        # 학습된 통계와 비교하여 이상 정도 계산
        $PYTHON_CMD -c "
import pandas as pd
import json
import numpy as np

# 추론 결과와 학습 통계 로드
infer_df = pd.read_parquet('$RESULT_DIR/baseline_scores.parquet')
with open('$MODEL_DIR/baseline_stats.json', 'r') as f:
    train_stats = json.load(f)

# 학습된 정상 패턴과 비교
normal_unseen_rate_mean = train_stats['unseen_stats']['mean_unseen_rate']
normal_unseen_rate_std = train_stats['unseen_stats']['std_unseen_rate']
normal_freq_mean = train_stats['frequency_stats']['mean_freq_z']
normal_freq_std = train_stats['frequency_stats']['std_freq_z']

# Z-score 계산 (학습 통계 기준)
infer_df['unseen_rate_zscore'] = (infer_df['unseen_rate'] - normal_unseen_rate_mean) / (normal_unseen_rate_std + 1e-8)
infer_df['freq_zscore'] = (infer_df['freq_z'] - normal_freq_mean) / (normal_freq_std + 1e-8)

# 종합 이상 점수 계산
infer_df['anomaly_score'] = np.sqrt(infer_df['unseen_rate_zscore']**2 + infer_df['freq_zscore']**2)

# 강화된 이상 판정 (학습 통계 기준)
threshold_95 = np.percentile(infer_df['anomaly_score'], 95)
infer_df['is_strong_anomaly'] = infer_df['anomaly_score'] > threshold_95

# 결과 저장
infer_df.to_parquet('$RESULT_DIR/baseline_scores_enhanced.parquet', index=False)

# 통계 출력
total_windows = len(infer_df)
anomaly_windows = (infer_df['is_anomaly'] == True).sum()
strong_anomaly_windows = (infer_df['is_strong_anomaly'] == True).sum()

print(f'✅ 베이스라인 분석 완료:')
print(f'   📊 총 윈도우: {total_windows}개')
print(f'   🚨 기본 이상: {anomaly_windows}개 ({100*anomaly_windows/total_windows:.1f}%)')
print(f'   🔥 강한 이상: {strong_anomaly_windows}개 ({100*strong_anomaly_windows/total_windows:.1f}%)')
print(f'   📈 평균 이상 점수: {infer_df[\"anomaly_score\"].mean():.3f}')
"
    else
        echo "⚠️  베이스라인 탐지 실행 실패"
    fi
else
    echo "⚠️  베이스라인 통계 파일이 없어 건너뜁니다."
fi
echo ""

# 3단계: DeepLog 추론
echo "3️⃣  DeepLog 추론 중..."
if [ -f "$MODEL_DIR/deeplog.pth" ] && [ -f "$MODEL_DIR/vocab.json" ]; then
    # DeepLog 입력 생성
    $PYTHON_CMD -c "
from study_preprocessor.builders.deeplog import build_deeplog_inputs

try:
    print('DeepLog 입력 생성 시작...')
    
    # DeepLog 입력 생성
    build_deeplog_inputs(
        parsed_parquet='$RESULT_DIR/parsed.parquet',
        out_dir='$RESULT_DIR'
    )
    
    print('DeepLog 입력 생성 완료')
    
except Exception as e:
    print(f'DeepLog 입력 생성 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
    
    if [ -f "$RESULT_DIR/sequences.parquet" ]; then
        # DeepLog 추론 실행
        $PYTHON_CMD -c "
from study_preprocessor.builders.deeplog import infer_deeplog_topk
from pathlib import Path

try:
    print('DeepLog 추론 시작...')
    
    # DeepLog 추론 실행
    df = infer_deeplog_topk(
        sequences_parquet='$RESULT_DIR/sequences.parquet',
        model_path='$MODEL_DIR/deeplog.pth',
        k=3
    )
    
    # 결과 저장
    output_path = Path('$RESULT_DIR') / 'deeplog_infer.parquet'
    df.to_parquet(output_path, index=False)
    
    print(f'DeepLog 추론 완료: {len(df)} 시퀀스 처리, 저장됨: {output_path}')
    
except Exception as e:
    print(f'DeepLog 추론 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
        
        if [ -f "$RESULT_DIR/deeplog_infer.parquet" ]; then
            # DeepLog 결과 통계
            $PYTHON_CMD -c "
import pandas as pd
df = pd.read_parquet('$RESULT_DIR/deeplog_infer.parquet')
total_sequences = len(df)
violations = (df['in_topk'] == False).sum()
print(f'✅ DeepLog 분석 완료:')
print(f'   📊 총 시퀀스: {total_sequences}개')
if len(df) > 0:
    print(f'   🚨 Top-K 위반: {violations}개 ({100*violations/total_sequences:.1f}%)')
else:
    print('   🚨 Top-K 위반: 0개 (시퀀스 데이터 없음 - 로그가 너무 짧음)')
"
        else
            echo "⚠️  DeepLog 추론 실행 실패"
        fi
    else
        echo "⚠️  DeepLog 입력 생성 실패"
    fi
else
    echo "⚠️  DeepLog 모델이 없어 건너뜁니다."
fi
echo ""

# 4단계: MS-CRED 추론
echo "4️⃣  MS-CRED 추론 중..."
if [ -f "$MODEL_DIR/mscred.pth" ]; then
    # MS-CRED 입력 생성
    $PYTHON_CMD -c "
from study_preprocessor.builders.mscred import build_mscred_window_counts

try:
    print('MS-CRED 입력 생성 시작...')
    
    # MS-CRED 입력 생성
    build_mscred_window_counts(
        parsed_parquet='$RESULT_DIR/parsed.parquet',
        out_dir='$RESULT_DIR',
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
    
    if [ -f "$RESULT_DIR/window_counts.parquet" ]; then
        # MS-CRED 추론 실행
        $PYTHON_CMD -c "
from study_preprocessor.mscred_model import infer_mscred
from pathlib import Path

try:
    print('MS-CRED 추론 시작...')
    
    # MS-CRED 추론 실행
    output_path = Path('$RESULT_DIR') / 'mscred_infer.parquet'
    df = infer_mscred(
        window_counts_path='$RESULT_DIR/window_counts.parquet',
        model_path='$MODEL_DIR/mscred.pth',
        output_path=str(output_path),
        threshold_percentile=95.0
    )
    
    print(f'MS-CRED 추론 완료: {len(df)} 윈도우 처리, 저장됨: {output_path}')
    
except Exception as e:
    print(f'MS-CRED 추론 오류: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
        
        if [ -f "$RESULT_DIR/mscred_infer.parquet" ]; then
            # MS-CRED 결과 통계
            $PYTHON_CMD -c "
import pandas as pd
df = pd.read_parquet('$RESULT_DIR/mscred_infer.parquet')
total_windows = len(df)
anomalies = (df['is_anomaly'] == True).sum()
print(f'✅ MS-CRED 분석 완료:')
print(f'   📊 총 윈도우: {total_windows}개')
print(f'   🚨 이상 윈도우: {anomalies}개 ({100*anomalies/total_windows:.1f}%)')
print(f'   📈 평균 재구성 오차: {df[\"recon_error\"].mean():.4f}')
"
        else
            echo "⚠️  MS-CRED 추론 실행 실패"
        fi
    else
        echo "⚠️  MS-CRED 입력 생성 실패"
    fi
else
    echo "⚠️  MS-CRED 모델이 없어 건너뜁니다."
fi
echo ""

# 5단계: 시간 기반 이상탐지
echo "5️⃣  시간 기반 이상탐지 중..."
if [ -f "temporal_anomaly_detector.py" ]; then
    $PYTHON_CMD temporal_anomaly_detector.py --data-dir "$RESULT_DIR" --output-dir "$RESULT_DIR/temporal_analysis"
    
    if [ -f "$RESULT_DIR/temporal_analysis/temporal_anomalies.json" ]; then
        temporal_count=$(python3 -c "import json; data=json.load(open('$RESULT_DIR/temporal_analysis/temporal_anomalies.json')); print(len(data))" 2>/dev/null || echo "N/A")
        echo "✅ 시간 기반 분석 완료: $temporal_count 건의 이상 패턴"
    else
        echo "⚠️  시간 기반 분석 실행 실패"
    fi
else
    echo "⚠️  temporal_anomaly_detector.py가 없어 건너뜁니다."
fi
echo ""

# 6단계: 이상 로그 샘플 추출 및 분석
echo "6️⃣  이상 로그 샘플 추출 및 분석 중..."
if [ -f "log_sample_analyzer.py" ]; then
    $PYTHON_CMD log_sample_analyzer.py "$RESULT_DIR" --output-dir "$RESULT_DIR/log_samples_analysis"
    
    if [ -f "$RESULT_DIR/log_samples_analysis/anomaly_analysis_report.md" ]; then
        # 샘플 통계 추출
        sample_stats=$(python3 -c "
import json
import os
if os.path.exists('$RESULT_DIR/log_samples_analysis/anomaly_samples.json'):
    with open('$RESULT_DIR/log_samples_analysis/anomaly_samples.json', 'r') as f:
        data = json.load(f)
    
    total_samples = 0
    methods = []
    for method, results in data.items():
        analyzed_count = results.get('analyzed_count', 0)
        if analyzed_count > 0:
            total_samples += analyzed_count
            methods.append(f'{method}({analyzed_count}개)')
    
    print(f'{total_samples}개 샘플 ({', '.join(methods)})')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
        
        echo "✅ 이상 로그 샘플 분석 완료: $sample_stats"
    else
        echo "⚠️  로그 샘플 분석 실행 실패"
    fi
else
    echo "⚠️  log_sample_analyzer.py가 없어 건너뜁니다."
fi
echo ""

# 7단계: 종합 결과 분석 및 리포트 생성
echo "7️⃣  종합 결과 분석 및 리포트 생성 중..."

# 종합 리포트 생성
$PYTHON_CMD -c "
import pandas as pd
import json
import os
from datetime import datetime

# 결과 파일들 확인
result_files = {
    'baseline': '$RESULT_DIR/baseline_scores_enhanced.parquet',
    'deeplog': '$RESULT_DIR/deeplog_infer.parquet',
    'mscred': '$RESULT_DIR/mscred_infer.parquet',
    'temporal': '$RESULT_DIR/temporal_analysis/temporal_anomalies.json'
}

available_results = {k: v for k, v in result_files.items() if os.path.exists(v)}

# 리포트 생성
report_lines = []
report_lines.append('# 🔍 로그 이상탐지 추론 결과 리포트')
report_lines.append('')
report_lines.append(f'**생성 시간**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
report_lines.append(f'**Target 로그**: $TARGET_LOG')
report_lines.append(f'**모델 디렉토리**: $MODEL_DIR')
report_lines.append('')

# 기본 정보
if os.path.exists('$RESULT_DIR/parsed.parquet'):
    df = pd.read_parquet('$RESULT_DIR/parsed.parquet')
    report_lines.append('## 📊 기본 정보')
    report_lines.append('')
    report_lines.append(f'- **총 로그 레코드**: {len(df):,}개')
    report_lines.append(f'- **고유 템플릿**: {len(df[\"template_id\"].unique())}개')
    report_lines.append(f'- **시간 범위**: {df[\"timestamp\"].min()} ~ {df[\"timestamp\"].max()}')
    report_lines.append('')

# 각 모델별 결과
report_lines.append('## 🤖 모델별 이상탐지 결과')
report_lines.append('')

# 베이스라인 결과
if 'baseline' in available_results:
    df = pd.read_parquet(available_results['baseline'])
    total_windows = len(df)
    basic_anomalies = (df['is_anomaly'] == True).sum()
    strong_anomalies = (df['is_strong_anomaly'] == True).sum() if 'is_strong_anomaly' in df.columns else 0
    
    report_lines.append('### 📊 베이스라인 이상탐지')
    report_lines.append('')
    report_lines.append(f'- **총 윈도우**: {total_windows:,}개')
    report_lines.append(f'- **기본 이상**: {basic_anomalies}개 ({100*basic_anomalies/total_windows:.1f}%)')
    report_lines.append(f'- **강한 이상**: {strong_anomalies}개 ({100*strong_anomalies/total_windows:.1f}%)')
    report_lines.append(f'- **평균 이상 점수**: {df[\"anomaly_score\"].mean():.3f}' if 'anomaly_score' in df.columns else '- **평균 이상 점수**: N/A')
    report_lines.append('')

# DeepLog 결과
if 'deeplog' in available_results:
    df = pd.read_parquet(available_results['deeplog'])
    total_sequences = len(df)
    violations = (df['in_topk'] == False).sum()
    
    report_lines.append('### 🧠 DeepLog 이상탐지')
    report_lines.append('')
    report_lines.append(f'- **총 시퀀스**: {total_sequences:,}개')
    if total_sequences > 0:
        report_lines.append(f'- **Top-K 위반**: {violations}개 ({100*violations/total_sequences:.1f}%)')
    else:
        report_lines.append(f'- **Top-K 위반**: 0개 (시퀀스 없음)')
    report_lines.append('')

# MS-CRED 결과
if 'mscred' in available_results:
    df = pd.read_parquet(available_results['mscred'])
    total_windows = len(df)
    anomalies = (df['is_anomaly'] == True).sum()
    
    report_lines.append('### 🔬 MS-CRED 이상탐지')
    report_lines.append('')
    report_lines.append(f'- **총 윈도우**: {total_windows:,}개')
    report_lines.append(f'- **이상 윈도우**: {anomalies}개 ({100*anomalies/total_windows:.1f}%)')
    report_lines.append(f'- **평균 재구성 오차**: {df[\"recon_error\"].mean():.4f}')
    report_lines.append('')

# 시간 기반 결과
if 'temporal' in available_results:
    with open(available_results['temporal'], 'r') as f:
        temporal_data = json.load(f)
    
    report_lines.append('### 🕐 시간 기반 이상탐지')
    report_lines.append('')
    report_lines.append(f'- **이상 패턴**: {len(temporal_data)}건')
    report_lines.append('')

# 종합 결론
report_lines.append('## 🎯 종합 결론')
report_lines.append('')

# 이상 정도 종합 평가
anomaly_indicators = []
if 'baseline' in available_results:
    df = pd.read_parquet(available_results['baseline'])
    if 'is_strong_anomaly' in df.columns:
        strong_rate = (df['is_strong_anomaly'] == True).mean()
        if strong_rate > 0.1:
            anomaly_indicators.append(f'베이스라인 강한 이상 비율 높음 ({strong_rate:.1%})')

if 'deeplog' in available_results:
    df = pd.read_parquet(available_results['deeplog'])
    violation_rate = (df['in_topk'] == False).mean()
    if violation_rate > 0.05:
        anomaly_indicators.append(f'DeepLog 위반율 높음 ({violation_rate:.1%})')

if 'mscred' in available_results:
    df = pd.read_parquet(available_results['mscred'])
    anomaly_rate = (df['is_anomaly'] == True).mean()
    if anomaly_rate > 0.05:
        anomaly_indicators.append(f'MS-CRED 이상율 높음 ({anomaly_rate:.1%})')

if anomaly_indicators:
    report_lines.append('🚨 **주요 이상 지표**:')
    for indicator in anomaly_indicators:
        report_lines.append(f'- {indicator}')
else:
    report_lines.append('✅ **전반적으로 정상 패턴**으로 판단됩니다.')

report_lines.append('')
report_lines.append('## 📁 생성된 파일들')
report_lines.append('')
for name, path in available_results.items():
    rel_path = os.path.relpath(path, '$RESULT_DIR')
    report_lines.append(f'- **{name}**: {rel_path}')

# 로그 샘플 분석 결과 추가
if os.path.exists('$RESULT_DIR/log_samples_analysis/anomaly_analysis_report.md'):
    report_lines.append('- **log_samples**: log_samples_analysis/anomaly_analysis_report.md')

# 리포트 저장
with open('$RESULT_DIR/inference_report.md', 'w', encoding='utf-8') as f:
    f.write('\\n'.join(report_lines))

print('✅ 종합 리포트 생성 완료')
"

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 이상탐지 추론 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  총 소요 시간: ${MINUTES}분 ${SECONDS}초"
echo ""

# 결과 요약 출력
echo "📊 추론 결과 요약:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

completed_analyses=0
total_analyses=5

if [ -f "$RESULT_DIR/baseline_scores_enhanced.parquet" ]; then
    echo "  ✅ 베이스라인 이상탐지 완료"
    completed_analyses=$((completed_analyses + 1))
else
    echo "  ❌ 베이스라인 이상탐지 실패"
fi

if [ -f "$RESULT_DIR/deeplog_infer.parquet" ]; then
    echo "  ✅ DeepLog 추론 완료"
    completed_analyses=$((completed_analyses + 1))
else
    echo "  ❌ DeepLog 추론 실패"
fi

if [ -f "$RESULT_DIR/mscred_infer.parquet" ]; then
    echo "  ✅ MS-CRED 추론 완료"
    completed_analyses=$((completed_analyses + 1))
else
    echo "  ❌ MS-CRED 추론 실패"
fi

if [ -f "$RESULT_DIR/temporal_analysis/temporal_anomalies.json" ]; then
    echo "  ✅ 시간 기반 분석 완료"
    completed_analyses=$((completed_analyses + 1))
else
    echo "  ❌ 시간 기반 분석 실패"
fi

if [ -f "$RESULT_DIR/log_samples_analysis/anomaly_analysis_report.md" ]; then
    echo "  ✅ 로그 샘플 분석 완료"
    completed_analyses=$((completed_analyses + 1))
else
    echo "  ❌ 로그 샘플 분석 실패"
fi

echo ""
echo "📈 추론 결과 통계:"
echo "  ✅ 완료된 분석: ${completed_analyses}/${total_analyses}개"
echo "  📁 결과 저장 위치: $RESULT_DIR"
echo "  📋 종합 리포트: $RESULT_DIR/inference_report.md"
echo ""

# 주요 결과 파일들 나열
echo "📊 생성된 주요 파일들:"
find "$RESULT_DIR" -name "*.parquet" -o -name "*.json" -o -name "*.md" | sort | while read file; do
    rel_path=$(echo "$file" | sed "s|^$(pwd)/||")
    size=$(stat -c%s "$file" 2>/dev/null | numfmt --to=iec)
    echo "  📝 $rel_path ($size)"
done
echo ""

# 리포트 내용 미리보기
if [ -f "$RESULT_DIR/inference_report.md" ]; then
    echo "📋 추론 결과 요약:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 리포트에서 핵심 정보만 추출하여 출력
    grep -E "^\*\*|^- \*\*|^🚨|^✅" "$RESULT_DIR/inference_report.md" | head -15
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

echo "🔍 상세 분석 명령어:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📄 종합 리포트: cat $RESULT_DIR/inference_report.md"

# 로그 샘플 분석 리포트 추천
if [ -f "$RESULT_DIR/log_samples_analysis/anomaly_analysis_report.md" ]; then
    echo "  📋 이상 로그 샘플 분석: cat $RESULT_DIR/log_samples_analysis/anomaly_analysis_report.md"
    echo "  📊 상세 샘플 데이터: cat $RESULT_DIR/log_samples_analysis/anomaly_samples.json"
fi

# 추가 분석 도구들 확인 및 제안
if [ -f "analyze_results.py" ]; then
    echo "  📊 상세 분석: $PYTHON_CMD analyze_results.py --data-dir $RESULT_DIR"
fi
if [ -f "visualize_results.py" ]; then
    echo "  📈 시각화: $PYTHON_CMD visualize_results.py --data-dir $RESULT_DIR"
fi
if [ -f "mscred_analyzer.py" ] && [ -f "$RESULT_DIR/mscred_infer.parquet" ]; then
    echo "  🔬 MS-CRED 분석: $PYTHON_CMD mscred_analyzer.py --data-dir $RESULT_DIR"
fi

echo ""
echo "💡 다른 로그 파일 분석:"
echo "  ./run_inference.sh $MODEL_DIR /path/to/another.log"
echo ""
echo "🎉 이상탐지 추론이 완료되었습니다!"
echo "   - ✅ Target 로그 전처리"
echo "   - ✅ 베이스라인 통계 기반 이상탐지" 
echo "   - ✅ DeepLog LSTM 이상탐지"
echo "   - ✅ MS-CRED 컨볼루션 이상탐지"
echo "   - ✅ 시간 기반 패턴 분석"
echo "   - ✅ 이상 로그 샘플 추출 및 분석"
echo "   - ✅ 종합 결과 리포트 생성"
