#!/bin/bash

# 모델 검증 도구
# 학습된 모델들의 품질과 성능을 자동으로 검증합니다.
# 사용법: ./validate_models.sh <모델디렉토리> [검증로그파일] [결과디렉토리]

set -e  # 에러 발생시 즉시 중단

# 기본값 설정
MODEL_DIR="$1"
VALIDATION_LOG="$2"
RESULT_DIR="${3:-validation_$(date +%Y%m%d_%H%M%S)}"

# 인수 확인
if [ -z "$MODEL_DIR" ]; then
    echo "❌ 사용법: $0 <모델디렉토리> [검증로그파일] [결과디렉토리]"
    echo ""
    echo "예시:"
    echo "  $0 models_20241002_143022"
    echo "  $0 models_20241002_143022 /var/log/validation.log"
    echo "  $0 models_20241002_143022 /var/log/validation.log validation_results"
    echo ""
    echo "📋 설명:"
    echo "  - 모델디렉토리: 검증할 학습된 모델들이 있는 폴더"
    echo "  - 검증로그파일: 검증용 로그 파일 (생략시 합성 데이터 사용)"
    echo "  - 결과디렉토리: 검증 결과를 저장할 폴더 (생략시 자동 생성)"
    echo ""
    echo "💡 검증 항목:"
    echo "  - 📁 모델 파일 무결성 검사"
    echo "  - 🔧 모델 로딩 및 실행 가능성 검증"
    echo "  - 📊 추론 성능 및 속도 측정"
    echo "  - 🎯 이상탐지 정확도 평가"
    echo "  - 📈 모델 안정성 테스트"
    echo "  - 🔍 메모리 사용량 모니터링"
    echo "  - 📋 종합 품질 점수 계산"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 모델 디렉토리를 찾을 수 없습니다: $MODEL_DIR"
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

# 결과 디렉토리 생성
mkdir -p "$RESULT_DIR"

echo "🚀 모델 검증 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 모델 디렉토리: $MODEL_DIR"
if [ -n "$VALIDATION_LOG" ] && [ -f "$VALIDATION_LOG" ]; then
    echo "🎯 검증 로그: $VALIDATION_LOG"
else
    echo "🎯 검증 로그: 합성 데이터 사용"
fi
echo "📁 결과 저장 디렉토리: $RESULT_DIR"
echo "🐍 Python 실행: $PYTHON_CMD"
echo ""
echo "🔄 수행할 검증 단계:"
echo "  1️⃣  모델 파일 무결성 검사"
echo "  2️⃣  모델 로딩 및 실행 가능성 검증"
echo "  3️⃣  검증 데이터 준비"
echo "  4️⃣  추론 성능 및 속도 측정"
echo "  5️⃣  이상탐지 정확도 평가"
echo "  6️⃣  모델 안정성 테스트"
echo "  7️⃣  메모리 사용량 모니터링"
echo "  8️⃣  종합 검증 리포트 생성"
echo ""
echo "⏱️  예상 소요 시간: 5-15분 (모델 크기에 따라)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# 1단계: 모델 파일 무결성 검사
echo "1️⃣  모델 파일 무결성 검사 중..."

$PYTHON_CMD -c "
import os
import json
import hashlib
from pathlib import Path

def calculate_file_hash(filepath):
    \"\"\"파일의 SHA256 해시 계산\"\"\"
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        return f'ERROR: {str(e)}'

def check_file_integrity(model_dir):
    \"\"\"모델 파일들의 무결성 검사\"\"\"
    model_files = {
        'deeplog.pth': {'required': False, 'type': 'model'},
        'mscred.pth': {'required': False, 'type': 'model'},
        'vocab.json': {'required': True, 'type': 'config'},
        'baseline_stats.json': {'required': False, 'type': 'stats'},
        'drain3_state.json': {'required': True, 'type': 'state'},
        'metadata.json': {'required': False, 'type': 'metadata'}
    }
    
    integrity_report = {
        'model_directory': model_dir,
        'files': {},
        'summary': {
            'total_files': 0,
            'existing_files': 0,
            'required_missing': 0,
            'corrupted_files': 0,
            'total_size': 0
        }
    }
    
    for filename, info in model_files.items():
        filepath = os.path.join(model_dir, filename)
        file_info = {
            'exists': os.path.exists(filepath),
            'required': info['required'],
            'type': info['type'],
            'size': 0,
            'hash': None,
            'readable': False,
            'valid_format': False
        }
        
        if file_info['exists']:
            try:
                file_info['size'] = os.path.getsize(filepath)
                file_info['hash'] = calculate_file_hash(filepath)
                file_info['readable'] = True
                integrity_report['summary']['total_size'] += file_info['size']
                
                # 파일 형식 검증
                if filename.endswith('.json'):
                    try:
                        with open(filepath, 'r') as f:
                            json.load(f)
                        file_info['valid_format'] = True
                    except:
                        file_info['valid_format'] = False
                elif filename.endswith('.pth'):
                    # PyTorch 모델 파일 기본 검증
                    file_info['valid_format'] = file_info['size'] > 1024  # 최소 크기 체크
                else:
                    file_info['valid_format'] = True
                    
            except Exception as e:
                file_info['error'] = str(e)
                integrity_report['summary']['corrupted_files'] += 1
        else:
            if info['required']:
                integrity_report['summary']['required_missing'] += 1
        
        integrity_report['files'][filename] = file_info
        integrity_report['summary']['total_files'] += 1
        if file_info['exists']:
            integrity_report['summary']['existing_files'] += 1
    
    return integrity_report

# 무결성 검사 실행
report = check_file_integrity('$MODEL_DIR')

# 결과 저장
with open('$RESULT_DIR/integrity_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# 요약 출력
print('✅ 모델 파일 무결성 검사 완료')
print(f'   📊 총 파일: {report[\"summary\"][\"total_files\"]}개')
print(f'   ✅ 존재하는 파일: {report[\"summary\"][\"existing_files\"]}개')
print(f'   ❌ 필수 파일 누락: {report[\"summary\"][\"required_missing\"]}개')
print(f'   🚨 손상된 파일: {report[\"summary\"][\"corrupted_files\"]}개')
print(f'   💾 총 크기: {report[\"summary\"][\"total_size\"]:,} bytes')

# 개별 파일 상태
for filename, info in report['files'].items():
    status = '✅' if info['exists'] and info['readable'] and info['valid_format'] else '❌'
    required = '(필수)' if info['required'] else ''
    size = f'({info[\"size\"]:,} bytes)' if info['exists'] else ''
    print(f'   {status} {filename} {required} {size}')
"
echo ""

# 2단계: 모델 로딩 및 실행 가능성 검증
echo "2️⃣  모델 로딩 및 실행 가능성 검증 중..."

$PYTHON_CMD -c "
import os
import json
import torch
import traceback
from pathlib import Path

def test_model_loading(model_dir):
    \"\"\"모델 로딩 테스트\"\"\"
    loading_report = {
        'deeplog': {'loadable': False, 'error': None, 'model_info': {}},
        'mscred': {'loadable': False, 'error': None, 'model_info': {}},
        'vocab': {'loadable': False, 'error': None, 'vocab_size': 0},
        'baseline_stats': {'loadable': False, 'error': None, 'stats_count': 0},
        'drain3_state': {'loadable': False, 'error': None, 'template_count': 0}
    }
    
    # DeepLog 모델 로딩 테스트
    deeplog_path = os.path.join(model_dir, 'deeplog.pth')
    if os.path.exists(deeplog_path):
        try:
            model_state = torch.load(deeplog_path, map_location='cpu')
            loading_report['deeplog']['loadable'] = True
            loading_report['deeplog']['model_info'] = {
                'state_dict_keys': len(model_state.keys()) if isinstance(model_state, dict) else 'Unknown',
                'file_size': os.path.getsize(deeplog_path)
            }
        except Exception as e:
            loading_report['deeplog']['error'] = str(e)
    
    # MS-CRED 모델 로딩 테스트
    mscred_path = os.path.join(model_dir, 'mscred.pth')
    if os.path.exists(mscred_path):
        try:
            model_state = torch.load(mscred_path, map_location='cpu')
            loading_report['mscred']['loadable'] = True
            loading_report['mscred']['model_info'] = {
                'state_dict_keys': len(model_state.keys()) if isinstance(model_state, dict) else 'Unknown',
                'file_size': os.path.getsize(mscred_path)
            }
        except Exception as e:
            loading_report['mscred']['error'] = str(e)
    
    # Vocab 파일 로딩 테스트
    vocab_path = os.path.join(model_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            loading_report['vocab']['loadable'] = True
            loading_report['vocab']['vocab_size'] = len(vocab_data) if isinstance(vocab_data, dict) else 0
        except Exception as e:
            loading_report['vocab']['error'] = str(e)
    
    # 베이스라인 통계 로딩 테스트
    baseline_path = os.path.join(model_dir, 'baseline_stats.json')
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, 'r') as f:
                stats_data = json.load(f)
            loading_report['baseline_stats']['loadable'] = True
            loading_report['baseline_stats']['stats_count'] = len(stats_data) if isinstance(stats_data, dict) else 0
        except Exception as e:
            loading_report['baseline_stats']['error'] = str(e)
    
    # Drain3 상태 로딩 테스트
    drain3_path = os.path.join(model_dir, 'drain3_state.json')
    if os.path.exists(drain3_path):
        try:
            with open(drain3_path, 'r') as f:
                drain3_data = json.load(f)
            loading_report['drain3_state']['loadable'] = True
            # Drain3 상태에서 템플릿 개수 추출 시도
            template_count = 0
            if isinstance(drain3_data, dict) and 'clusters' in drain3_data:
                template_count = len(drain3_data['clusters'])
            loading_report['drain3_state']['template_count'] = template_count
        except Exception as e:
            loading_report['drain3_state']['error'] = str(e)
    
    return loading_report

# 모델 로딩 테스트 실행
loading_report = test_model_loading('$MODEL_DIR')

# 결과 저장
with open('$RESULT_DIR/loading_report.json', 'w') as f:
    json.dump(loading_report, f, indent=2, ensure_ascii=False)

# 요약 출력
print('✅ 모델 로딩 검증 완료')

for component, info in loading_report.items():
    if os.path.exists(os.path.join('$MODEL_DIR', f'{component}.pth')) or os.path.exists(os.path.join('$MODEL_DIR', f'{component}.json')):
        status = '✅' if info['loadable'] else '❌'
        error_msg = f' (오류: {info[\"error\"]})' if info['error'] else ''
        
        if component == 'deeplog' and info['loadable']:
            print(f'   {status} DeepLog 모델: 로딩 가능{error_msg}')
        elif component == 'mscred' and info['loadable']:
            print(f'   {status} MS-CRED 모델: 로딩 가능{error_msg}')
        elif component == 'vocab' and info['loadable']:
            print(f'   {status} 어휘 사전: {info[\"vocab_size\"]}개 템플릿{error_msg}')
        elif component == 'baseline_stats' and info['loadable']:
            print(f'   {status} 베이스라인 통계: 로딩 가능{error_msg}')
        elif component == 'drain3_state' and info['loadable']:
            print(f'   {status} Drain3 상태: {info[\"template_count\"]}개 클러스터{error_msg}')
        elif not info['loadable']:
            print(f'   {status} {component}: 로딩 실패{error_msg}')
"
echo ""

# 3단계: 검증 데이터 준비
echo "3️⃣  검증 데이터 준비 중..."

if [ -n "$VALIDATION_LOG" ] && [ -f "$VALIDATION_LOG" ]; then
    echo "   사용자 제공 검증 로그 사용: $VALIDATION_LOG"
    TEST_DATA="$VALIDATION_LOG"
else
    echo "   검증용 합성 데이터 생성 중..."
    # 검증용 합성 데이터 생성 (정상/이상 라벨링된 데이터)
    $PYTHON_CMD -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# 검증용 합성 로그 데이터 생성 (라벨링된 데이터)
np.random.seed(42)
random.seed(42)

# 정상 로그 템플릿들
normal_templates = [
    'INFO: User <*> logged in successfully',
    'INFO: Processing request <*> completed',
    'INFO: Database query executed in <*>ms',
    'INFO: Cache hit for key <*>',
    'INFO: Service <*> started successfully',
    'DEBUG: Memory usage: <*>%',
    'DEBUG: CPU usage: <*>%'
]

# 이상 로그 템플릿들
anomaly_templates = [
    'ERROR: Authentication failed for user <*>',
    'CRITICAL: Database connection timeout',
    'ERROR: Out of memory exception',
    'FATAL: Service <*> crashed unexpectedly',
    'ERROR: Security breach attempt detected',
    'CRITICAL: Disk space full',
    'ERROR: Network connection lost'
]

# 검증 데이터 생성
validation_logs = []
labels = []
start_time = datetime.now() - timedelta(hours=2)

# 정상 로그 (70%)
for i in range(1400):
    timestamp = start_time + timedelta(seconds=i*5)
    template = random.choice(normal_templates)
    log_line = f'{timestamp.strftime(\"%Y-%m-%d %H:%M:%S\")} {template}'
    validation_logs.append(log_line)
    labels.append(0)  # 정상 = 0

# 이상 로그 (30%)
for i in range(600):
    timestamp = start_time + timedelta(seconds=random.randint(0, 7000))
    template = random.choice(anomaly_templates)
    log_line = f'{timestamp.strftime(\"%Y-%m-%d %H:%M:%S\")} {template}'
    validation_logs.append(log_line)
    labels.append(1)  # 이상 = 1

# 시간순 정렬
combined = list(zip(validation_logs, labels))
combined.sort(key=lambda x: x[0])
validation_logs, labels = zip(*combined)

# 검증 로그 파일 저장
with open('$RESULT_DIR/validation_data.log', 'w') as f:
    for log in validation_logs:
        f.write(log + '\\n')

# 라벨 파일 저장
label_data = {
    'labels': list(labels),
    'total_logs': len(labels),
    'normal_count': labels.count(0),
    'anomaly_count': labels.count(1),
    'anomaly_rate': labels.count(1) / len(labels)
}

with open('$RESULT_DIR/validation_labels.json', 'w') as f:
    json.dump(label_data, f, indent=2)

print(f'✅ 검증 데이터 생성 완료: {len(validation_logs):,} 라인')
print(f'   - 정상 로그: {labels.count(0):,}개 ({100*labels.count(0)/len(labels):.1f}%)')
print(f'   - 이상 로그: {labels.count(1):,}개 ({100*labels.count(1)/len(labels):.1f}%)')
"
    TEST_DATA="$RESULT_DIR/validation_data.log"
fi
echo ""

# 4단계: 추론 성능 및 속도 측정
echo "4️⃣  추론 성능 및 속도 측정 중..."

# 메모리 사용량 모니터링을 위한 함수
monitor_memory() {
    local pid=$1
    local output_file=$2
    while kill -0 $pid 2>/dev/null; do
        ps -p $pid -o pid,vsz,rss,pcpu --no-headers >> "$output_file" 2>/dev/null || break
        sleep 1
    done
}

# 추론 실행 및 성능 측정
INFERENCE_RESULT="$RESULT_DIR/inference_performance"
MEMORY_LOG="$RESULT_DIR/memory_usage.log"

echo "   추론 실행 중 (메모리 모니터링 포함)..."
start_time=$(date +%s)

# 백그라운드에서 메모리 모니터링 시작
echo "PID VSZ RSS CPU" > "$MEMORY_LOG"

# 추론 실행
./run_inference.sh "$MODEL_DIR" "$TEST_DATA" "$INFERENCE_RESULT" > "$RESULT_DIR/inference.log" 2>&1 &
INFERENCE_PID=$!

# 메모리 모니터링 시작
monitor_memory $INFERENCE_PID "$MEMORY_LOG" &
MONITOR_PID=$!

# 추론 완료 대기
wait $INFERENCE_PID
INFERENCE_EXIT_CODE=$?

# 메모리 모니터링 종료
kill $MONITOR_PID 2>/dev/null || true

end_time=$(date +%s)
inference_duration=$((end_time - start_time))

echo "✅ 추론 성능 측정 완료"
echo "   ⏱️  추론 시간: ${inference_duration}초"
echo "   📊 종료 코드: $INFERENCE_EXIT_CODE"

# 메모리 사용량 분석
if [ -f "$MEMORY_LOG" ] && [ $(wc -l < "$MEMORY_LOG") -gt 1 ]; then
    $PYTHON_CMD -c "
import pandas as pd
import json

# 메모리 로그 분석
try:
    df = pd.read_csv('$MEMORY_LOG', sep='\s+', skiprows=1, names=['PID', 'VSZ', 'RSS', 'CPU'])
    if len(df) > 0:
        memory_stats = {
            'max_memory_mb': float(df['RSS'].max() / 1024),
            'avg_memory_mb': float(df['RSS'].mean() / 1024),
            'max_cpu_percent': float(df['CPU'].max()),
            'avg_cpu_percent': float(df['CPU'].mean()),
            'samples': len(df)
        }
        
        with open('$RESULT_DIR/memory_stats.json', 'w') as f:
            json.dump(memory_stats, f, indent=2)
        
        print(f'   💾 최대 메모리: {memory_stats[\"max_memory_mb\"]:.1f} MB')
        print(f'   💾 평균 메모리: {memory_stats[\"avg_memory_mb\"]:.1f} MB')
        print(f'   🖥️  최대 CPU: {memory_stats[\"max_cpu_percent\"]:.1f}%')
    else:
        print('   ⚠️  메모리 데이터 수집 실패')
except Exception as e:
    print(f'   ⚠️  메모리 분석 오류: {e}')
"
fi
echo ""

# 5단계: 이상탐지 정확도 평가
echo "5️⃣  이상탐지 정확도 평가 중..."

if [ -f "$RESULT_DIR/validation_labels.json" ]; then
    echo "   라벨링된 데이터로 정확도 계산 중..."
    
    $PYTHON_CMD -c "
import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

def evaluate_model_accuracy(inference_dir, labels_file):
    \"\"\"모델별 정확도 평가\"\"\"
    
    # 라벨 데이터 로드
    with open(labels_file, 'r') as f:
        label_data = json.load(f)
    
    true_labels = np.array(label_data['labels'])
    
    accuracy_report = {
        'total_samples': len(true_labels),
        'true_anomaly_rate': float(np.mean(true_labels)),
        'models': {}
    }
    
    # 베이스라인 모델 평가
    baseline_file = os.path.join(inference_dir, 'baseline_scores_enhanced.parquet')
    if os.path.exists(baseline_file):
        try:
            df = pd.read_parquet(baseline_file)
            # 윈도우 단위를 로그 단위로 변환 (간단한 매핑)
            window_size = 50
            predicted_labels = []
            
            for i in range(len(true_labels)):
                window_idx = i // window_size
                if window_idx < len(df):
                    # 강한 이상이 있으면 이상으로 판정
                    is_anomaly = df.iloc[window_idx].get('is_strong_anomaly', df.iloc[window_idx].get('is_anomaly', False))
                    predicted_labels.append(1 if is_anomaly else 0)
                else:
                    predicted_labels.append(0)
            
            predicted_labels = np.array(predicted_labels[:len(true_labels)])
            
            accuracy_report['models']['baseline'] = {
                'accuracy': float(accuracy_score(true_labels, predicted_labels)),
                'precision': float(precision_score(true_labels, predicted_labels, zero_division=0)),
                'recall': float(recall_score(true_labels, predicted_labels, zero_division=0)),
                'f1_score': float(f1_score(true_labels, predicted_labels, zero_division=0)),
                'predicted_anomaly_rate': float(np.mean(predicted_labels))
            }
        except Exception as e:
            accuracy_report['models']['baseline'] = {'error': str(e)}
    
    # DeepLog 모델 평가
    deeplog_file = os.path.join(inference_dir, 'deeplog_infer.parquet')
    if os.path.exists(deeplog_file):
        try:
            df = pd.read_parquet(deeplog_file)
            # 시퀀스 단위를 로그 단위로 변환
            seq_len = 50
            predicted_labels = []
            
            for i in range(len(true_labels)):
                seq_idx = max(0, i - seq_len + 1)
                if seq_idx < len(df):
                    # Top-K에 없으면 이상으로 판정
                    is_anomaly = not df.iloc[seq_idx].get('in_topk', True)
                    predicted_labels.append(1 if is_anomaly else 0)
                else:
                    predicted_labels.append(0)
            
            predicted_labels = np.array(predicted_labels[:len(true_labels)])
            
            accuracy_report['models']['deeplog'] = {
                'accuracy': float(accuracy_score(true_labels, predicted_labels)),
                'precision': float(precision_score(true_labels, predicted_labels, zero_division=0)),
                'recall': float(recall_score(true_labels, predicted_labels, zero_division=0)),
                'f1_score': float(f1_score(true_labels, predicted_labels, zero_division=0)),
                'predicted_anomaly_rate': float(np.mean(predicted_labels))
            }
        except Exception as e:
            accuracy_report['models']['deeplog'] = {'error': str(e)}
    
    # MS-CRED 모델 평가
    mscred_file = os.path.join(inference_dir, 'mscred_infer.parquet')
    if os.path.exists(mscred_file):
        try:
            df = pd.read_parquet(mscred_file)
            # 윈도우 단위를 로그 단위로 변환
            window_size = 50
            predicted_labels = []
            
            for i in range(len(true_labels)):
                window_idx = i // window_size
                if window_idx < len(df):
                    is_anomaly = df.iloc[window_idx].get('is_anomaly', False)
                    predicted_labels.append(1 if is_anomaly else 0)
                else:
                    predicted_labels.append(0)
            
            predicted_labels = np.array(predicted_labels[:len(true_labels)])
            
            accuracy_report['models']['mscred'] = {
                'accuracy': float(accuracy_score(true_labels, predicted_labels)),
                'precision': float(precision_score(true_labels, predicted_labels, zero_division=0)),
                'recall': float(recall_score(true_labels, predicted_labels, zero_division=0)),
                'f1_score': float(f1_score(true_labels, predicted_labels, zero_division=0)),
                'predicted_anomaly_rate': float(np.mean(predicted_labels))
            }
        except Exception as e:
            accuracy_report['models']['mscred'] = {'error': str(e)}
    
    return accuracy_report

# 정확도 평가 실행
if os.path.exists('$INFERENCE_RESULT'):
    accuracy_report = evaluate_model_accuracy('$INFERENCE_RESULT', '$RESULT_DIR/validation_labels.json')
    
    # 결과 저장
    with open('$RESULT_DIR/accuracy_report.json', 'w') as f:
        json.dump(accuracy_report, f, indent=2, ensure_ascii=False)
    
    print('✅ 이상탐지 정확도 평가 완료')
    print(f'   📊 총 샘플: {accuracy_report[\"total_samples\"]:,}개')
    print(f'   🎯 실제 이상율: {accuracy_report[\"true_anomaly_rate\"]:.1%}')
    
    for model_name, metrics in accuracy_report['models'].items():
        if 'error' not in metrics:
            print(f'   🤖 {model_name.upper()}:')
            print(f'      정확도: {metrics[\"accuracy\"]:.3f}')
            print(f'      정밀도: {metrics[\"precision\"]:.3f}')
            print(f'      재현율: {metrics[\"recall\"]:.3f}')
            print(f'      F1점수: {metrics[\"f1_score\"]:.3f}')
        else:
            print(f'   ❌ {model_name.upper()}: 평가 실패 ({metrics[\"error\"]})')
else:
    print('⚠️  추론 결과가 없어 정확도 평가를 건너뜁니다.')
"
else
    echo "   라벨 데이터가 없어 정확도 평가를 건너뜁니다."
fi
echo ""

# 6단계: 모델 안정성 테스트
echo "6️⃣  모델 안정성 테스트 중..."

$PYTHON_CMD -c "
import os
import json
import time
import traceback
from pathlib import Path

def stability_test(model_dir, test_runs=5):
    \"\"\"모델 안정성 테스트\"\"\"
    
    stability_report = {
        'test_runs': test_runs,
        'models': {
            'deeplog': {'success_rate': 0, 'avg_load_time': 0, 'errors': []},
            'mscred': {'success_rate': 0, 'avg_load_time': 0, 'errors': []},
            'vocab': {'success_rate': 0, 'avg_load_time': 0, 'errors': []},
            'baseline_stats': {'success_rate': 0, 'avg_load_time': 0, 'errors': []},
            'drain3_state': {'success_rate': 0, 'avg_load_time': 0, 'errors': []}
        }
    }
    
    # 각 모델에 대해 반복 로딩 테스트
    for model_name in stability_report['models'].keys():
        success_count = 0
        load_times = []
        
        for run in range(test_runs):
            try:
                start_time = time.time()
                
                if model_name == 'deeplog':
                    model_path = os.path.join(model_dir, 'deeplog.pth')
                    if os.path.exists(model_path):
                        import torch
                        torch.load(model_path, map_location='cpu')
                    else:
                        continue
                        
                elif model_name == 'mscred':
                    model_path = os.path.join(model_dir, 'mscred.pth')
                    if os.path.exists(model_path):
                        import torch
                        torch.load(model_path, map_location='cpu')
                    else:
                        continue
                        
                elif model_name in ['vocab', 'baseline_stats', 'drain3_state']:
                    file_path = os.path.join(model_dir, f'{model_name}.json')
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            json.load(f)
                    else:
                        continue
                
                load_time = time.time() - start_time
                load_times.append(load_time)
                success_count += 1
                
            except Exception as e:
                stability_report['models'][model_name]['errors'].append(f'Run {run+1}: {str(e)}')
        
        if success_count > 0:
            stability_report['models'][model_name]['success_rate'] = success_count / test_runs
            stability_report['models'][model_name]['avg_load_time'] = sum(load_times) / len(load_times)
    
    return stability_report

# 안정성 테스트 실행
stability_report = stability_test('$MODEL_DIR', test_runs=3)

# 결과 저장
with open('$RESULT_DIR/stability_report.json', 'w') as f:
    json.dump(stability_report, f, indent=2, ensure_ascii=False)

print('✅ 모델 안정성 테스트 완료')

for model_name, metrics in stability_report['models'].items():
    if metrics['success_rate'] > 0:
        print(f'   🤖 {model_name.upper()}:')
        print(f'      성공률: {metrics[\"success_rate\"]:.1%}')
        print(f'      평균 로딩시간: {metrics[\"avg_load_time\"]:.3f}초')
        if metrics['errors']:
            print(f'      오류 수: {len(metrics[\"errors\"])}개')
"
echo ""

# 7단계: 종합 검증 리포트 생성
echo "7️⃣  종합 검증 리포트 생성 중..."

$PYTHON_CMD -c "
import json
import os
from datetime import datetime

# 모든 검증 결과 로드
reports = {}

report_files = {
    'integrity': '$RESULT_DIR/integrity_report.json',
    'loading': '$RESULT_DIR/loading_report.json',
    'accuracy': '$RESULT_DIR/accuracy_report.json',
    'stability': '$RESULT_DIR/stability_report.json',
    'memory': '$RESULT_DIR/memory_stats.json'
}

for report_name, file_path in report_files.items():
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            reports[report_name] = json.load(f)

# 종합 점수 계산
def calculate_quality_score(reports):
    \"\"\"모델 품질 종합 점수 계산 (0-100점)\"\"\"
    
    score = 0
    max_score = 100
    
    # 파일 무결성 점수 (20점)
    if 'integrity' in reports:
        integrity = reports['integrity']
        file_score = 0
        if integrity['summary']['required_missing'] == 0:
            file_score += 10  # 필수 파일 모두 존재
        if integrity['summary']['corrupted_files'] == 0:
            file_score += 10  # 손상된 파일 없음
        score += file_score
    
    # 모델 로딩 점수 (20점)
    if 'loading' in reports:
        loading = reports['loading']
        loading_score = 0
        loadable_models = sum(1 for model in loading.values() if model.get('loadable', False))
        total_models = len([k for k in loading.keys() if os.path.exists(os.path.join('$MODEL_DIR', f'{k}.pth')) or os.path.exists(os.path.join('$MODEL_DIR', f'{k}.json'))])
        if total_models > 0:
            loading_score = (loadable_models / total_models) * 20
        score += loading_score
    
    # 정확도 점수 (30점)
    if 'accuracy' in reports:
        accuracy = reports['accuracy']
        accuracy_score = 0
        model_count = 0
        total_f1 = 0
        
        for model_name, metrics in accuracy.get('models', {}).items():
            if 'f1_score' in metrics:
                total_f1 += metrics['f1_score']
                model_count += 1
        
        if model_count > 0:
            avg_f1 = total_f1 / model_count
            accuracy_score = avg_f1 * 30  # F1 점수 기반
        score += accuracy_score
    
    # 안정성 점수 (20점)
    if 'stability' in reports:
        stability = reports['stability']
        stability_score = 0
        model_count = 0
        total_success_rate = 0
        
        for model_name, metrics in stability.get('models', {}).items():
            if metrics.get('success_rate', 0) > 0:
                total_success_rate += metrics['success_rate']
                model_count += 1
        
        if model_count > 0:
            avg_success_rate = total_success_rate / model_count
            stability_score = avg_success_rate * 20
        score += stability_score
    
    # 성능 점수 (10점)
    performance_score = 10  # 기본 점수 (추론이 완료되면)
    if 'memory' in reports:
        memory = reports['memory']
        # 메모리 사용량이 적절하면 추가 점수
        if memory.get('max_memory_mb', 0) < 1000:  # 1GB 미만
            performance_score = 10
        elif memory.get('max_memory_mb', 0) < 2000:  # 2GB 미만
            performance_score = 8
        else:
            performance_score = 5
    score += performance_score
    
    return min(score, max_score)

# 종합 점수 계산
quality_score = calculate_quality_score(reports)

# 리포트 생성
report_lines = []
report_lines.append('# 🔍 모델 검증 리포트')
report_lines.append('')
report_lines.append(f'**생성 시간**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
report_lines.append(f'**모델 디렉토리**: $MODEL_DIR')
report_lines.append(f'**검증 데이터**: $TEST_DATA')
report_lines.append('')

# 종합 점수
report_lines.append(f'## 🎯 종합 품질 점수: {quality_score:.1f}/100')
report_lines.append('')

if quality_score >= 90:
    report_lines.append('✅ **우수**: 모델이 매우 안정적이고 정확합니다.')
elif quality_score >= 70:
    report_lines.append('🟡 **양호**: 모델이 대체로 안정적입니다.')
elif quality_score >= 50:
    report_lines.append('🟠 **보통**: 일부 개선이 필요합니다.')
else:
    report_lines.append('🔴 **불량**: 모델에 심각한 문제가 있습니다.')

report_lines.append('')

# 파일 무결성 결과
if 'integrity' in reports:
    integrity = reports['integrity']
    report_lines.append('## 📁 파일 무결성 검사')
    report_lines.append('')
    report_lines.append(f'- **총 파일**: {integrity[\"summary\"][\"total_files\"]}개')
    report_lines.append(f'- **존재하는 파일**: {integrity[\"summary\"][\"existing_files\"]}개')
    report_lines.append(f'- **필수 파일 누락**: {integrity[\"summary\"][\"required_missing\"]}개')
    report_lines.append(f'- **손상된 파일**: {integrity[\"summary\"][\"corrupted_files\"]}개')
    report_lines.append(f'- **총 크기**: {integrity[\"summary\"][\"total_size\"]:,} bytes')
    report_lines.append('')

# 모델 로딩 결과
if 'loading' in reports:
    loading = reports['loading']
    report_lines.append('## 🔧 모델 로딩 검증')
    report_lines.append('')
    
    for model_name, info in loading.items():
        if os.path.exists(os.path.join('$MODEL_DIR', f'{model_name}.pth')) or os.path.exists(os.path.join('$MODEL_DIR', f'{model_name}.json')):
            status = '✅' if info.get('loadable', False) else '❌'
            model_display = model_name.upper().replace('_', ' ')
            
            if info.get('loadable', False):
                report_lines.append(f'- {status} **{model_display}**: 로딩 성공')
            else:
                error_msg = info.get('error', '알 수 없는 오류')
                report_lines.append(f'- {status} **{model_display}**: 로딩 실패 ({error_msg})')
    
    report_lines.append('')

# 정확도 평가 결과
if 'accuracy' in reports:
    accuracy = reports['accuracy']
    report_lines.append('## 🎯 이상탐지 정확도')
    report_lines.append('')
    report_lines.append(f'- **총 샘플**: {accuracy[\"total_samples\"]:,}개')
    report_lines.append(f'- **실제 이상율**: {accuracy[\"true_anomaly_rate\"]:.1%}')
    report_lines.append('')
    
    report_lines.append('| 모델 | 정확도 | 정밀도 | 재현율 | F1점수 |')
    report_lines.append('|------|--------|--------|--------|--------|')
    
    for model_name, metrics in accuracy.get('models', {}).items():
        if 'error' not in metrics:
            model_display = model_name.upper()
            acc = f'{metrics[\"accuracy\"]:.3f}'
            prec = f'{metrics[\"precision\"]:.3f}'
            rec = f'{metrics[\"recall\"]:.3f}'
            f1 = f'{metrics[\"f1_score\"]:.3f}'
            report_lines.append(f'| {model_display} | {acc} | {prec} | {rec} | {f1} |')
        else:
            model_display = model_name.upper()
            report_lines.append(f'| {model_display} | 오류 | 오류 | 오류 | 오류 |')
    
    report_lines.append('')

# 안정성 테스트 결과
if 'stability' in reports:
    stability = reports['stability']
    report_lines.append('## 🛡️ 모델 안정성')
    report_lines.append('')
    
    for model_name, metrics in stability.get('models', {}).items():
        if metrics.get('success_rate', 0) > 0:
            model_display = model_name.upper().replace('_', ' ')
            success_rate = f'{metrics[\"success_rate\"]:.1%}'
            load_time = f'{metrics[\"avg_load_time\"]:.3f}초'
            report_lines.append(f'- **{model_display}**: 성공률 {success_rate}, 평균 로딩시간 {load_time}')
    
    report_lines.append('')

# 메모리 사용량 결과
if 'memory' in reports:
    memory = reports['memory']
    report_lines.append('## 💾 메모리 사용량')
    report_lines.append('')
    report_lines.append(f'- **최대 메모리**: {memory[\"max_memory_mb\"]:.1f} MB')
    report_lines.append(f'- **평균 메모리**: {memory[\"avg_memory_mb\"]:.1f} MB')
    report_lines.append(f'- **최대 CPU**: {memory[\"max_cpu_percent\"]:.1f}%')
    report_lines.append('')

# 권장사항
report_lines.append('## 💡 권장사항')
report_lines.append('')

if quality_score >= 90:
    report_lines.append('- ✅ 모델이 우수한 상태입니다. 프로덕션 환경에서 사용 가능합니다.')
elif quality_score >= 70:
    report_lines.append('- 🟡 모델이 양호한 상태입니다. 일부 성능 최적화를 고려해보세요.')
else:
    report_lines.append('- 🔴 모델에 문제가 있습니다. 다음 사항을 확인하세요:')
    
    if 'integrity' in reports and reports['integrity']['summary']['required_missing'] > 0:
        report_lines.append('  - 필수 모델 파일이 누락되었습니다.')
    
    if 'loading' in reports:
        failed_models = [name for name, info in reports['loading'].items() 
                        if not info.get('loadable', False) and 
                        (os.path.exists(os.path.join('$MODEL_DIR', f'{name}.pth')) or 
                         os.path.exists(os.path.join('$MODEL_DIR', f'{name}.json')))]
        if failed_models:
            report_lines.append(f'  - 다음 모델들의 로딩에 실패했습니다: {', '.join(failed_models)}')
    
    if 'accuracy' in reports:
        low_accuracy_models = [name for name, metrics in reports['accuracy'].get('models', {}).items() 
                              if 'f1_score' in metrics and metrics['f1_score'] < 0.5]
        if low_accuracy_models:
            report_lines.append(f'  - 다음 모델들의 정확도가 낮습니다: {', '.join(low_accuracy_models)}')

report_lines.append('')

# 생성된 파일들
report_lines.append('## 📁 생성된 파일들')
report_lines.append('')
for report_name, file_path in report_files.items():
    if os.path.exists(file_path):
        rel_path = os.path.relpath(file_path, '$RESULT_DIR')
        report_lines.append(f'- **{report_name}_report**: {rel_path}')

# 리포트 저장
with open('$RESULT_DIR/validation_report.md', 'w', encoding='utf-8') as f:
    f.write('\\n'.join(report_lines))

print('✅ 종합 검증 리포트 생성 완료')
print(f'   🎯 종합 품질 점수: {quality_score:.1f}/100')
"

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 모델 검증 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  총 소요 시간: ${MINUTES}분 ${SECONDS}초"
echo ""

# 결과 요약 출력
echo "📊 검증 결과 요약:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 종합 점수 출력
if [ -f "$RESULT_DIR/validation_report.md" ]; then
    quality_score=$(grep "종합 품질 점수:" "$RESULT_DIR/validation_report.md" | grep -o '[0-9]*\.[0-9]*' | head -1)
    if [ -n "$quality_score" ]; then
        echo "  🎯 종합 품질 점수: ${quality_score}/100"
        
        # 점수에 따른 상태 표시
        if (( $(echo "$quality_score >= 90" | bc -l) )); then
            echo "  ✅ 상태: 우수 (프로덕션 사용 가능)"
        elif (( $(echo "$quality_score >= 70" | bc -l) )); then
            echo "  🟡 상태: 양호 (일부 최적화 권장)"
        elif (( $(echo "$quality_score >= 50" | bc -l) )); then
            echo "  🟠 상태: 보통 (개선 필요)"
        else
            echo "  🔴 상태: 불량 (심각한 문제)"
        fi
    fi
fi

echo "  📁 검증 결과 위치: $RESULT_DIR"
echo ""

# 주요 검증 항목별 결과
echo "📋 주요 검증 항목:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 파일 무결성
if [ -f "$RESULT_DIR/integrity_report.json" ]; then
    missing_files=$(python3 -c "import json; data=json.load(open('$RESULT_DIR/integrity_report.json')); print(data['summary']['required_missing'])" 2>/dev/null || echo "N/A")
    corrupted_files=$(python3 -c "import json; data=json.load(open('$RESULT_DIR/integrity_report.json')); print(data['summary']['corrupted_files'])" 2>/dev/null || echo "N/A")
    
    if [ "$missing_files" = "0" ] && [ "$corrupted_files" = "0" ]; then
        echo "  ✅ 파일 무결성: 모든 파일 정상"
    else
        echo "  ❌ 파일 무결성: 누락 ${missing_files}개, 손상 ${corrupted_files}개"
    fi
fi

# 모델 로딩
if [ -f "$RESULT_DIR/loading_report.json" ]; then
    loadable_count=$(python3 -c "
import json
data = json.load(open('$RESULT_DIR/loading_report.json'))
count = sum(1 for model in data.values() if model.get('loadable', False))
print(count)
" 2>/dev/null || echo "N/A")
    echo "  🔧 모델 로딩: ${loadable_count}개 모델 로딩 가능"
fi

# 추론 성능
if [ -f "$RESULT_DIR/memory_stats.json" ]; then
    max_memory=$(python3 -c "import json; data=json.load(open('$RESULT_DIR/memory_stats.json')); print(f'{data[\"max_memory_mb\"]:.1f}')" 2>/dev/null || echo "N/A")
    echo "  💾 메모리 사용: 최대 ${max_memory} MB"
fi

# 정확도
if [ -f "$RESULT_DIR/accuracy_report.json" ]; then
    avg_f1=$(python3 -c "
import json
data = json.load(open('$RESULT_DIR/accuracy_report.json'))
f1_scores = [m['f1_score'] for m in data.get('models', {}).values() if 'f1_score' in m]
if f1_scores:
    print(f'{sum(f1_scores)/len(f1_scores):.3f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
    echo "  🎯 평균 F1 점수: ${avg_f1}"
fi

echo ""
echo "📁 생성된 파일들:"
find "$RESULT_DIR" -name "*.json" -o -name "*.md" -o -name "*.log" | sort | while read file; do
    rel_path=$(echo "$file" | sed "s|^$(pwd)/||")
    size=$(stat -c%s "$file" 2>/dev/null | numfmt --to=iec)
    echo "  📝 $rel_path ($size)"
done
echo ""

# 리포트 내용 미리보기
if [ -f "$RESULT_DIR/validation_report.md" ]; then
    echo "📋 검증 결과 미리보기:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 리포트에서 핵심 정보만 추출하여 출력
    grep -E "^## |^- \*\*|^✅|^🟡|^🟠|^🔴" "$RESULT_DIR/validation_report.md" | head -15
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

echo "🔍 상세 분석:"
echo "  📄 종합 리포트: cat $RESULT_DIR/validation_report.md"
echo "  📊 JSON 데이터: ls $RESULT_DIR/*.json"
echo ""
echo "💡 모델 개선 방법:"
echo "  🔄 점진적 학습: ./train_models_incremental.sh $MODEL_DIR /path/to/new_logs/"
echo "  📊 모델 비교: ./compare_models.sh $MODEL_DIR other_model/"
echo ""
echo "🎉 모델 검증이 완료되었습니다!"
echo "   - ✅ 파일 무결성 검사"
echo "   - ✅ 모델 로딩 검증"
echo "   - ✅ 추론 성능 측정"
echo "   - ✅ 이상탐지 정확도 평가"
echo "   - ✅ 안정성 테스트"
echo "   - ✅ 메모리 사용량 모니터링"
echo "   - ✅ 종합 품질 점수 계산"
