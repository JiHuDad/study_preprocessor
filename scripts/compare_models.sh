#!/bin/bash

# 모델 비교 도구
# 여러 시점에 학습된 모델들의 성능을 비교합니다.
# 사용법: ./compare_models.sh <모델1디렉토리> <모델2디렉토리> [테스트로그파일] [결과디렉토리]

set -e  # 에러 발생시 즉시 중단

# 기본값 설정
MODEL1_DIR="$1"
MODEL2_DIR="$2"
TEST_LOG="$3"
RESULT_DIR="${4:-model_comparison_$(date +%Y%m%d_%H%M%S)}"

# 인수 확인
if [ -z "$MODEL1_DIR" ] || [ -z "$MODEL2_DIR" ]; then
    echo "❌ 사용법: $0 <모델1디렉토리> <모델2디렉토리> [테스트로그파일] [결과디렉토리]"
    echo ""
    echo "예시:"
    echo "  $0 models_old models_new"
    echo "  $0 models_old models_new /var/log/test.log"
    echo "  $0 models_old models_new /var/log/test.log comparison_results"
    echo ""
    echo "📋 설명:"
    echo "  - 모델1디렉토리: 첫 번째 비교할 모델 폴더"
    echo "  - 모델2디렉토리: 두 번째 비교할 모델 폴더"
    echo "  - 테스트로그파일: 성능 비교용 테스트 로그 (생략시 합성 데이터 사용)"
    echo "  - 결과디렉토리: 비교 결과를 저장할 폴더 (생략시 자동 생성)"
    echo ""
    echo "💡 특징:"
    echo "  - 📊 모델 메타데이터 비교"
    echo "  - 🎯 동일 테스트 데이터로 성능 비교"
    echo "  - 📈 이상탐지 정확도 및 속도 측정"
    echo "  - 📋 상세 비교 리포트 생성"
    echo "  - 🔍 모델별 강점/약점 분석"
    exit 1
fi

if [ ! -d "$MODEL1_DIR" ]; then
    echo "❌ 모델1 디렉토리를 찾을 수 없습니다: $MODEL1_DIR"
    exit 1
fi

if [ ! -d "$MODEL2_DIR" ]; then
    echo "❌ 모델2 디렉토리를 찾을 수 없습니다: $MODEL2_DIR"
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
if ! $PYTHON_CMD -c "import anomaly_log_detector" 2>/dev/null; then
    echo "🔧 anomaly_log_detector 패키지 설치 중..."
    .venv/bin/pip install -e . || {
        echo "❌ 패키지 설치 실패"
        exit 1
    }
    echo "✅ 패키지 설치 완료"
fi

# 결과 디렉토리 생성
mkdir -p "$RESULT_DIR"

echo "🚀 모델 비교 분석 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 모델1 디렉토리: $MODEL1_DIR"
echo "📂 모델2 디렉토리: $MODEL2_DIR"
if [ -n "$TEST_LOG" ] && [ -f "$TEST_LOG" ]; then
    echo "🎯 테스트 로그: $TEST_LOG"
else
    echo "🎯 테스트 로그: 합성 데이터 사용"
fi
echo "📁 결과 저장 디렉토리: $RESULT_DIR"
echo "🐍 Python 실행: $PYTHON_CMD"
echo ""
echo "🔄 수행할 비교 단계:"
echo "  1️⃣  모델 메타데이터 비교"
echo "  2️⃣  모델 파일 크기 및 구조 분석"
echo "  3️⃣  테스트 데이터 준비"
echo "  4️⃣  모델별 추론 성능 측정"
echo "  5️⃣  이상탐지 결과 비교"
echo "  6️⃣  종합 비교 리포트 생성"
echo ""
echo "⏱️  예상 소요 시간: 5-15분 (테스트 데이터 크기에 따라)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# 1단계: 모델 메타데이터 비교
echo "1️⃣  모델 메타데이터 비교 중..."

# 모델 정보 수집 및 비교
$PYTHON_CMD -c "
import json
import os
from datetime import datetime
from pathlib import Path

def get_model_info(model_dir):
    info = {
        'directory': model_dir,
        'exists': os.path.exists(model_dir),
        'files': {},
        'metadata': {},
        'sizes': {}
    }
    
    if not info['exists']:
        return info
    
    # 모델 파일들 확인
    model_files = {
        'deeplog.pth': os.path.exists(f'{model_dir}/deeplog.pth'),
        'mscred.pth': os.path.exists(f'{model_dir}/mscred.pth'),
        'vocab.json': os.path.exists(f'{model_dir}/vocab.json'),
        'baseline_stats.json': os.path.exists(f'{model_dir}/baseline_stats.json'),
        'drain3_state.json': os.path.exists(f'{model_dir}/drain3_state.json'),
        'metadata.json': os.path.exists(f'{model_dir}/metadata.json')
    }
    info['files'] = model_files
    
    # 파일 크기 정보
    for filename, exists in model_files.items():
        if exists:
            filepath = f'{model_dir}/{filename}'
            info['sizes'][filename] = os.path.getsize(filepath)
    
    # 메타데이터 로드
    if model_files['metadata.json']:
        try:
            with open(f'{model_dir}/metadata.json', 'r') as f:
                info['metadata'] = json.load(f)
        except:
            info['metadata'] = {'error': 'Failed to load metadata'}
    
    return info

# 두 모델 정보 수집
model1_info = get_model_info('$MODEL1_DIR')
model2_info = get_model_info('$MODEL2_DIR')

# 비교 결과 저장
comparison = {
    'comparison_time': datetime.now().isoformat(),
    'model1': model1_info,
    'model2': model2_info,
    'differences': {}
}

# 차이점 분석
if model1_info['exists'] and model2_info['exists']:
    # 파일 존재 여부 비교
    file_diff = {}
    for filename in model1_info['files']:
        m1_exists = model1_info['files'][filename]
        m2_exists = model2_info['files'].get(filename, False)
        if m1_exists != m2_exists:
            file_diff[filename] = {'model1': m1_exists, 'model2': m2_exists}
    comparison['differences']['files'] = file_diff
    
    # 파일 크기 비교
    size_diff = {}
    for filename in model1_info['sizes']:
        if filename in model2_info['sizes']:
            size1 = model1_info['sizes'][filename]
            size2 = model2_info['sizes'][filename]
            if abs(size1 - size2) > 1024:  # 1KB 이상 차이
                size_diff[filename] = {
                    'model1': size1,
                    'model2': size2,
                    'difference': size2 - size1,
                    'ratio': size2 / size1 if size1 > 0 else float('inf')
                }
    comparison['differences']['sizes'] = size_diff

# 결과 저장
with open('$RESULT_DIR/model_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

print('✅ 메타데이터 비교 완료')

# 요약 출력
print(f'📊 모델1 ({model1_info[\"directory\"]}):')
for filename, exists in model1_info['files'].items():
    status = '✅' if exists else '❌'
    size = f'({model1_info[\"sizes\"].get(filename, 0):,} bytes)' if exists else ''
    print(f'   {status} {filename} {size}')

print(f'📊 모델2 ({model2_info[\"directory\"]}):')
for filename, exists in model2_info['files'].items():
    status = '✅' if exists else '❌'
    size = f'({model2_info[\"sizes\"].get(filename, 0):,} bytes)' if exists else ''
    print(f'   {status} {filename} {size}')
"
echo ""

# 2단계: 테스트 데이터 준비
echo "2️⃣  테스트 데이터 준비 중..."

if [ -n "$TEST_LOG" ] && [ -f "$TEST_LOG" ]; then
    echo "   사용자 제공 테스트 로그 사용: $TEST_LOG"
    TEST_DATA="$TEST_LOG"
else
    echo "   합성 테스트 데이터 생성 중..."
    # 합성 데이터 생성
    $PYTHON_CMD -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 합성 로그 데이터 생성
np.random.seed(42)
random.seed(42)

# 기본 로그 템플릿들
templates = [
    'INFO: User <*> logged in from <*>',
    'INFO: Processing request <*>',
    'INFO: Database query executed in <*>ms',
    'WARNING: High memory usage: <*>%',
    'ERROR: Connection timeout to <*>',
    'ERROR: Failed to process <*>',
    'INFO: Service <*> started',
    'INFO: Service <*> stopped',
    'DEBUG: Cache hit for key <*>',
    'DEBUG: Cache miss for key <*>'
]

# 정상 패턴 (80%)
normal_logs = []
start_time = datetime.now() - timedelta(hours=24)

for i in range(8000):
    timestamp = start_time + timedelta(seconds=i*10)
    template = random.choice(templates[:7])  # 정상 템플릿들
    log_line = f'{timestamp.strftime(\"%Y-%m-%d %H:%M:%S\")} {template}'
    normal_logs.append(log_line)

# 이상 패턴 (20%)
anomaly_templates = [
    'CRITICAL: System overload detected',
    'ERROR: Security breach attempt from <*>',
    'FATAL: Database corruption detected',
    'ERROR: Out of memory exception',
    'CRITICAL: Service <*> crashed unexpectedly'
]

anomaly_logs = []
for i in range(2000):
    timestamp = start_time + timedelta(seconds=random.randint(0, 80000))
    template = random.choice(anomaly_templates)
    log_line = f'{timestamp.strftime(\"%Y-%m-%d %H:%M:%S\")} {template}'
    anomaly_logs.append(log_line)

# 로그 합치고 시간순 정렬
all_logs = normal_logs + anomaly_logs
all_logs.sort()

# 파일로 저장
with open('$RESULT_DIR/synthetic_test.log', 'w') as f:
    for log in all_logs:
        f.write(log + '\\n')

print(f'✅ 합성 테스트 데이터 생성: {len(all_logs):,} 라인')
print(f'   - 정상 로그: {len(normal_logs):,}개 (80%)')
print(f'   - 이상 로그: {len(anomaly_logs):,}개 (20%)')
"
    TEST_DATA="$RESULT_DIR/synthetic_test.log"
fi
echo ""

# 3단계: 모델별 추론 성능 측정
echo "3️⃣  모델별 추론 성능 측정 중..."

# 모델1으로 추론
echo "   모델1 추론 중..."
MODEL1_RESULT="$RESULT_DIR/model1_inference"
start_time=$(date +%s)
./run_inference.sh "$MODEL1_DIR" "$TEST_DATA" "$MODEL1_RESULT" > "$RESULT_DIR/model1_inference.log" 2>&1
model1_time=$(($(date +%s) - start_time))

# 모델2로 추론
echo "   모델2 추론 중..."
MODEL2_RESULT="$RESULT_DIR/model2_inference"
start_time=$(date +%s)
./run_inference.sh "$MODEL2_DIR" "$TEST_DATA" "$MODEL2_RESULT" > "$RESULT_DIR/model2_inference.log" 2>&1
model2_time=$(($(date +%s) - start_time))

echo "✅ 추론 성능 측정 완료"
echo "   모델1 소요시간: ${model1_time}초"
echo "   모델2 소요시간: ${model2_time}초"
echo ""

# 4단계: 이상탐지 결과 비교
echo "4️⃣  이상탐지 결과 비교 중..."

$PYTHON_CMD -c "
import pandas as pd
import json
import numpy as np
from pathlib import Path

def load_inference_results(result_dir):
    results = {}
    
    # 베이스라인 결과
    baseline_file = f'{result_dir}/baseline_scores_enhanced.parquet'
    if Path(baseline_file).exists():
        df = pd.read_parquet(baseline_file)
        results['baseline'] = {
            'total_windows': len(df),
            'anomaly_count': int((df['is_anomaly'] == True).sum()),
            'strong_anomaly_count': int((df['is_strong_anomaly'] == True).sum()) if 'is_strong_anomaly' in df.columns else 0,
            'anomaly_rate': float((df['is_anomaly'] == True).mean()),
            'avg_anomaly_score': float(df['anomaly_score'].mean()) if 'anomaly_score' in df.columns else 0
        }
    
    # DeepLog 결과
    deeplog_file = f'{result_dir}/deeplog_infer.parquet'
    if Path(deeplog_file).exists():
        df = pd.read_parquet(deeplog_file)
        # Enhanced 버전: prediction_ok 사용, 기존 버전: in_topk 사용
        if 'prediction_ok' in df.columns:
            violation_count = int((df['prediction_ok'] == False).sum())
            violation_rate = float((df['prediction_ok'] == False).mean()) if len(df) > 0 else 0.0
        elif 'in_topk' in df.columns:
            violation_count = int((df['in_topk'] == False).sum())
            violation_rate = float((df['in_topk'] == False).mean()) if len(df) > 0 else 0.0
        else:
            violation_count = 0
            violation_rate = 0.0

        results['deeplog'] = {
            'total_sequences': len(df),
            'violation_count': violation_count,
            'violation_rate': violation_rate
        }
    
    # MS-CRED 결과
    mscred_file = f'{result_dir}/mscred_infer.parquet'
    if Path(mscred_file).exists():
        df = pd.read_parquet(mscred_file)
        results['mscred'] = {
            'total_windows': len(df),
            'anomaly_count': int((df['is_anomaly'] == True).sum()),
            'anomaly_rate': float((df['is_anomaly'] == True).mean()),
            'avg_recon_error': float(df['reconstruction_error'].mean())
        }
    
    return results

# 두 모델의 결과 로드
model1_results = load_inference_results('$MODEL1_RESULT')
model2_results = load_inference_results('$MODEL2_RESULT')

# 성능 비교
performance_comparison = {
    'model1_time': $model1_time,
    'model2_time': $model2_time,
    'speed_improvement': ($model1_time - $model2_time) / $model1_time if $model1_time > 0 else 0,
    'model1_results': model1_results,
    'model2_results': model2_results,
    'differences': {}
}

# 결과 차이 계산
for method in ['baseline', 'deeplog', 'mscred']:
    if method in model1_results and method in model2_results:
        m1 = model1_results[method]
        m2 = model2_results[method]
        
        diff = {}
        if method == 'baseline':
            diff['anomaly_rate_diff'] = m2['anomaly_rate'] - m1['anomaly_rate']
            diff['strong_anomaly_diff'] = m2['strong_anomaly_count'] - m1['strong_anomaly_count']
        elif method == 'deeplog':
            diff['violation_rate_diff'] = m2['violation_rate'] - m1['violation_rate']
        elif method == 'mscred':
            diff['anomaly_rate_diff'] = m2['anomaly_rate'] - m1['anomaly_rate']
            diff['recon_error_diff'] = m2['avg_recon_error'] - m1['avg_recon_error']
        
        performance_comparison['differences'][method] = diff

# 결과 저장
with open('$RESULT_DIR/performance_comparison.json', 'w') as f:
    json.dump(performance_comparison, f, indent=2, ensure_ascii=False)

print('✅ 이상탐지 결과 비교 완료')

# 요약 출력
print('📊 성능 비교 요약:')
print(f'   ⏱️  추론 속도: 모델1 {$model1_time}초 vs 모델2 {$model2_time}초')

if 'baseline' in model1_results and 'baseline' in model2_results:
    m1_rate = model1_results['baseline']['anomaly_rate']
    m2_rate = model2_results['baseline']['anomaly_rate']
    print(f'   📊 베이스라인 이상율: 모델1 {m1_rate:.1%} vs 모델2 {m2_rate:.1%}')

if 'deeplog' in model1_results and 'deeplog' in model2_results:
    m1_rate = model1_results['deeplog']['violation_rate']
    m2_rate = model2_results['deeplog']['violation_rate']
    print(f'   🧠 DeepLog 위반율: 모델1 {m1_rate:.1%} vs 모델2 {m2_rate:.1%}')

if 'mscred' in model1_results and 'mscred' in model2_results:
    m1_rate = model1_results['mscred']['anomaly_rate']
    m2_rate = model2_results['mscred']['anomaly_rate']
    print(f'   🔬 MS-CRED 이상율: 모델1 {m1_rate:.1%} vs 모델2 {m2_rate:.1%}')
"
echo ""

# 5단계: 종합 비교 리포트 생성
echo "5️⃣  종합 비교 리포트 생성 중..."

$PYTHON_CMD -c "
import json
import os
from datetime import datetime

# 비교 데이터 로드
with open('$RESULT_DIR/model_comparison.json', 'r') as f:
    model_comparison = json.load(f)

with open('$RESULT_DIR/performance_comparison.json', 'r') as f:
    performance_comparison = json.load(f)

# 리포트 생성
report_lines = []
report_lines.append('# 🔍 모델 비교 분석 리포트')
report_lines.append('')
report_lines.append(f'**생성 시간**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
report_lines.append(f'**모델1**: $MODEL1_DIR')
report_lines.append(f'**모델2**: $MODEL2_DIR')
report_lines.append(f'**테스트 데이터**: $TEST_DATA')
report_lines.append('')

# 모델 기본 정보 비교
report_lines.append('## 📊 모델 기본 정보 비교')
report_lines.append('')

model1_info = model_comparison['model1']
model2_info = model_comparison['model2']

report_lines.append('| 항목 | 모델1 | 모델2 |')
report_lines.append('|------|-------|-------|')

# 파일 존재 여부
for filename in ['deeplog.pth', 'mscred.pth', 'vocab.json', 'baseline_stats.json']:
    m1_exists = '✅' if model1_info['files'].get(filename, False) else '❌'
    m2_exists = '✅' if model2_info['files'].get(filename, False) else '❌'
    report_lines.append(f'| {filename} | {m1_exists} | {m2_exists} |')

report_lines.append('')

# 파일 크기 비교
if model_comparison['differences']['sizes']:
    report_lines.append('### 📏 파일 크기 차이')
    report_lines.append('')
    report_lines.append('| 파일 | 모델1 크기 | 모델2 크기 | 차이 | 비율 |')
    report_lines.append('|------|------------|------------|------|------|')
    
    for filename, diff_info in model_comparison['differences']['sizes'].items():
        size1 = f'{diff_info[\"model1\"]:,} bytes'
        size2 = f'{diff_info[\"model2\"]:,} bytes'
        diff = f'{diff_info[\"difference\"]:+,} bytes'
        ratio = f'{diff_info[\"ratio\"]:.2f}x'
        report_lines.append(f'| {filename} | {size1} | {size2} | {diff} | {ratio} |')
    
    report_lines.append('')

# 성능 비교
report_lines.append('## ⚡ 성능 비교')
report_lines.append('')

model1_time = performance_comparison['model1_time']
model2_time = performance_comparison['model2_time']
speed_improvement = performance_comparison['speed_improvement']

report_lines.append(f'- **추론 속도**: 모델1 {model1_time}초 vs 모델2 {model2_time}초')
if speed_improvement > 0:
    report_lines.append(f'- **속도 개선**: {speed_improvement:.1%} 빨라짐')
elif speed_improvement < 0:
    report_lines.append(f'- **속도 변화**: {-speed_improvement:.1%} 느려짐')
else:
    report_lines.append(f'- **속도 변화**: 동일')
report_lines.append('')

# 이상탐지 결과 비교
report_lines.append('## 🎯 이상탐지 결과 비교')
report_lines.append('')

model1_results = performance_comparison['model1_results']
model2_results = performance_comparison['model2_results']

for method in ['baseline', 'deeplog', 'mscred']:
    if method in model1_results and method in model2_results:
        method_name = {'baseline': '베이스라인', 'deeplog': 'DeepLog', 'mscred': 'MS-CRED'}[method]
        report_lines.append(f'### {method_name}')
        report_lines.append('')
        
        m1 = model1_results[method]
        m2 = model2_results[method]
        
        if method == 'baseline':
            report_lines.append(f'- **이상 윈도우**: 모델1 {m1[\"anomaly_count\"]}개 ({m1[\"anomaly_rate\"]:.1%}) vs 모델2 {m2[\"anomaly_count\"]}개 ({m2[\"anomaly_rate\"]:.1%})')
            if 'strong_anomaly_count' in m1 and 'strong_anomaly_count' in m2:
                report_lines.append(f'- **강한 이상**: 모델1 {m1[\"strong_anomaly_count\"]}개 vs 모델2 {m2[\"strong_anomaly_count\"]}개')
        elif method == 'deeplog':
            report_lines.append(f'- **위반 시퀀스**: 모델1 {m1[\"violation_count\"]}개 ({m1[\"violation_rate\"]:.1%}) vs 모델2 {m2[\"violation_count\"]}개 ({m2[\"violation_rate\"]:.1%})')
        elif method == 'mscred':
            report_lines.append(f'- **이상 윈도우**: 모델1 {m1[\"anomaly_count\"]}개 ({m1[\"anomaly_rate\"]:.1%}) vs 모델2 {m2[\"anomaly_count\"]}개 ({m2[\"anomaly_rate\"]:.1%})')
            report_lines.append(f'- **평균 재구성 오차**: 모델1 {m1[\"avg_recon_error\"]:.4f} vs 모델2 {m2[\"avg_recon_error\"]:.4f}')
        
        report_lines.append('')

# 종합 결론
report_lines.append('## 🎯 종합 결론')
report_lines.append('')

# 성능 우위 판단
better_model = 'model2' if speed_improvement > 0 else 'model1'
report_lines.append(f'### 추론 속도')
if abs(speed_improvement) > 0.1:
    report_lines.append(f'- **{\"모델2\" if speed_improvement > 0 else \"모델1\"}**가 {abs(speed_improvement):.1%} 더 빠름')
else:
    report_lines.append('- 두 모델의 추론 속도는 비슷함')
report_lines.append('')

# 이상탐지 성능 종합
report_lines.append('### 이상탐지 성능')
differences = performance_comparison['differences']
for method, diff in differences.items():
    method_name = {'baseline': '베이스라인', 'deeplog': 'DeepLog', 'mscred': 'MS-CRED'}[method]
    
    if method == 'baseline' and 'anomaly_rate_diff' in diff:
        rate_diff = diff['anomaly_rate_diff']
        if abs(rate_diff) > 0.05:
            better = '모델2' if rate_diff > 0 else '모델1'
            report_lines.append(f'- **{method_name}**: {better}가 더 많은 이상을 탐지 (차이: {abs(rate_diff):.1%})')
    elif method == 'deeplog' and 'violation_rate_diff' in diff:
        rate_diff = diff['violation_rate_diff']
        if abs(rate_diff) > 0.02:
            better = '모델2' if rate_diff > 0 else '모델1'
            report_lines.append(f'- **{method_name}**: {better}가 더 많은 위반을 탐지 (차이: {abs(rate_diff):.1%})')
    elif method == 'mscred' and 'anomaly_rate_diff' in diff:
        rate_diff = diff['anomaly_rate_diff']
        if abs(rate_diff) > 0.05:
            better = '모델2' if rate_diff > 0 else '모델1'
            report_lines.append(f'- **{method_name}**: {better}가 더 많은 이상을 탐지 (차이: {abs(rate_diff):.1%})')

report_lines.append('')

# 권장사항
report_lines.append('### 💡 권장사항')
report_lines.append('')

if speed_improvement > 0.2:
    report_lines.append('- **모델2 사용 권장**: 속도가 크게 개선됨')
elif speed_improvement < -0.2:
    report_lines.append('- **모델1 사용 권장**: 모델2가 너무 느림')
else:
    report_lines.append('- **성능 차이 미미**: 최신 모델(모델2) 사용 권장')

report_lines.append('')

# 생성된 파일들
report_lines.append('## 📁 생성된 파일들')
report_lines.append('')
report_lines.append('- `model_comparison.json`: 모델 메타데이터 비교')
report_lines.append('- `performance_comparison.json`: 성능 비교 데이터')
report_lines.append('- `model1_inference/`: 모델1 추론 결과')
report_lines.append('- `model2_inference/`: 모델2 추론 결과')
report_lines.append('- `synthetic_test.log`: 합성 테스트 데이터 (사용된 경우)')

# 리포트 저장
with open('$RESULT_DIR/comparison_report.md', 'w', encoding='utf-8') as f:
    f.write('\\n'.join(report_lines))

print('✅ 종합 비교 리포트 생성 완료')
"

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 모델 비교 분석 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  총 소요 시간: ${MINUTES}분 ${SECONDS}초"
echo ""

# 결과 요약 출력
echo "📊 비교 결과 요약:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📂 모델1: $MODEL1_DIR"
echo "  📂 모델2: $MODEL2_DIR"
echo "  ⏱️  추론 시간: 모델1 ${model1_time}초 vs 모델2 ${model2_time}초"

if [ $model2_time -lt $model1_time ]; then
    improvement=$(( (model1_time - model2_time) * 100 / model1_time ))
    echo "  🚀 모델2가 ${improvement}% 더 빠름"
elif [ $model2_time -gt $model1_time ]; then
    degradation=$(( (model2_time - model1_time) * 100 / model1_time ))
    echo "  🐌 모델2가 ${degradation}% 더 느림"
else
    echo "  ⚖️  두 모델의 속도 동일"
fi

echo ""
echo "📁 생성된 파일들:"
echo "  📄 종합 리포트: $RESULT_DIR/comparison_report.md"
echo "  📊 비교 데이터: $RESULT_DIR/model_comparison.json"
echo "  📈 성능 데이터: $RESULT_DIR/performance_comparison.json"
echo "  📁 모델1 결과: $RESULT_DIR/model1_inference/"
echo "  📁 모델2 결과: $RESULT_DIR/model2_inference/"
echo ""

# 리포트 내용 미리보기
if [ -f "$RESULT_DIR/comparison_report.md" ]; then
    echo "📋 비교 결과 미리보기:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 리포트에서 핵심 정보만 추출하여 출력
    grep -E "^\*\*|^- \*\*|^### |^- " "$RESULT_DIR/comparison_report.md" | head -20
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

echo "🔍 상세 분석:"
echo "  📄 전체 리포트: cat $RESULT_DIR/comparison_report.md"
echo "  📊 JSON 데이터: cat $RESULT_DIR/performance_comparison.json"
echo ""
echo "🎉 모델 비교가 완료되었습니다!"
echo "   - ✅ 모델 메타데이터 비교"
echo "   - ✅ 파일 크기 및 구조 분석"
echo "   - ✅ 추론 성능 측정"
echo "   - ✅ 이상탐지 결과 비교"
echo "   - ✅ 종합 비교 리포트 생성"
