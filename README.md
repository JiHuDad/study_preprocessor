### study-preprocessor 사용 가이드

이 문서는 커널/시스템 로그(.log) 파일에 전처리와 이상탐지를 적용하는 방법을 단계별로 안내합니다. 모든 예시는 `venv + pip` 기반으로 실행합니다.

#### 1) 설치/환경
- 사전 요구: macOS/Linux, Python 3.11+
- 가상환경 생성 및 활성화:
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -U pip wheel
```
- 패키지 설치(둘 중 하나 선택)
  1) 고정 버전 설치(requirements.txt):
  ```
  pip install -r requirements.txt
  ```
  2) 개발 편의를 위한 editable 설치:
```
pip install -e .
```

#### 2) 단일 .log 파일 전처리
- 기본 실행:
```
study-preprocess parse \
  --input /path/to/your.log \
  --out-dir /path/to/outdir \
  --drain-state .cache/drain3.json
```
- 주요 산출물:
  - `/path/to/outdir/parsed.parquet`: 전처리 결과(raw/masked/template_id 등)
  - `/path/to/outdir/preview.json`: 상위 10행 미리보기(원문 vs 마스킹)

- 마스킹 옵션(개별 토글): 기본은 모두 마스킹 ON, 아래 플래그로 OFF 가능
  - `--no-mask-paths`, `--no-mask-hex`, `--no-mask-ips`, `--no-mask-mac`, `--no-mask-uuid`
  - `--no-mask-pid`, `--no-mask-device`, `--no-mask-num`
  - 예: 숫자/디바이스 접미사 마스킹을 끄고 실행
```
study-preprocess parse \
  --input /path/to/your.log \
  --out-dir /path/to/outdir \
  --no-mask-device --no-mask-num
```

- Drain3 상태 재사용
  - `--drain-state .cache/drain3.json`로 상태를 저장/누적하여 템플릿 일관성을 유지
  - 여러 파일을 순차 처리할 때 같은 `--drain-state`를 지정하면 기존 템플릿을 재사용합니다.

#### 3) 여러 .log 파일 배치 처리
- 쉘 루프 예시:
```
OUT=/path/to/processed
STATE=.cache/drain3.json
mkdir -p "$OUT"
for f in /var/log/*.log; do
  study-preprocess parse --input "$f" --out-dir "$OUT/$(basename "$f" .log)" --drain-state "$STATE"
done
```
- 결과 병합(선택):
```
python - <<'PY'
import os, pandas as pd
base = '/path/to/processed'
parts = []
for d in os.listdir(base):
    p = os.path.join(base, d, 'parsed.parquet')
    if os.path.exists(p):
        df = pd.read_parquet(p)
        df['source'] = d
        parts.append(df)
if parts:
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(base, 'merged.parquet'), index=False)
    print('Merged ->', os.path.join(base, 'merged.parquet'))
else:
    print('No parquet found')
PY
```

#### 4) DeepLog/MSCRED 입력 생성
- DeepLog 입력(사전/시퀀스):
```
study-preprocess build-deeplog \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir
```
- MS-CRED 입력(윈도우 카운트):
```
study-preprocess build-mscred \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir \
  --window-size 50 --stride 25
```

#### 5) 이상탐지 실행
- 베이스라인(새 템플릿 비율 + 빈도 급변):
```
study-preprocess detect \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir \
  --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.95
```
- DeepLog 학습/추론:
```
study-preprocess deeplog-train \
  --seq /path/to/outdir/sequences.parquet \
  --vocab /path/to/outdir/vocab.json \
  --out .cache/deeplog.pth --seq-len 50 --epochs 3

study-preprocess deeplog-infer \
  --seq /path/to/outdir/sequences.parquet \
  --model .cache/deeplog.pth --k 3
```
- 리포트/요약 생성:
```
# 기본 리포트
study-preprocess report --processed-dir /path/to/outdir

# 이상 로그 샘플 포함 리포트
study-preprocess report --processed-dir /path/to/outdir --with-samples
```
  - 포함: 베이스라인 이상 윈도우 비율, 상위 윈도우/템플릿, DeepLog 위반율
  - `--with-samples`: 실제 문제 로그 샘플과 분석 추가

- 이상 로그 샘플 분석 (단독):
```
study-preprocess analyze-samples --processed-dir /path/to/outdir
```
  - 🔍 이상탐지 결과에서 실제 문제 로그들 추출
  - 📄 사람이 읽기 쉬운 분석 리포트 생성
  - 🎯 전후 맥락과 함께 이상 패턴 설명

#### 6) 합성 데이터로 E2E 검증(옵션)
```
# 합성 로그 + 라벨 생성
study-preprocess gen-synth --out data/raw/synth_long.log --lines 1000 --anomaly-rate 0.03

# 전처리 → 빌더 → 탐지 → 학습/추론 → 리포트/평가
study-preprocess parse --input data/raw/synth_long.log --out-dir data/processed/synth --drain-state .cache/drain3.json
study-preprocess build-deeplog --parsed data/processed/synth/parsed.parquet --out-dir data/processed/synth
study-preprocess detect --parsed data/processed/synth/parsed.parquet --out-dir data/processed/synth --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.95
study-preprocess deeplog-train --seq data/processed/synth/sequences.parquet --vocab data/processed/synth/vocab.json --out .cache/deeplog_synth.pth --seq-len 20 --epochs 2
study-preprocess deeplog-infer --seq data/processed/synth/sequences.parquet --model .cache/deeplog_synth.pth --k 3
study-preprocess report --processed-dir data/processed/synth
study-preprocess eval --processed-dir data/processed/synth --labels data/raw/synth_long.log.labels.parquet --window-size 50 --seq-len 20
```

#### 7) 문제 해결 팁
- 템플릿이 과도하게 늘어나는 경우: 마스킹을 더 강하게 하거나 `--drain-state`를 유지하며 순서대로 처리
- 타임스탬프 파싱 실패: 라인 인덱스(`line_no`) 기준으로도 정렬되며, 포맷이 다른 경우 전처리 규칙 보강 필요
- 메모리: 대형 파일은 디렉터리 단위로 나눠 처리 후 병합 권장

#### 8) 자동화 스크립트 (한번에 실행)
전체 파이프라인을 한번에 실행할 수 있는 스크립트 제공:

**uv 환경용:**
```bash
./run_full_pipeline.sh /path/to/your.log [출력디렉토리]
```

**pip/venv 환경용:**
```bash
./run_full_pipeline_pip.sh /path/to/your.log [출력디렉토리]
```

자동 기능:
- 가상환경 자동 감지 및 활성화 (.venv, venv)
- 의존성 자동 설치 (필요시)
- 에러 처리 및 진행 상황 표시
- 결과 파일 자동 정리 및 요약

#### 9) 산출물 해석 요약
- `parsed.parquet`: `raw`, `masked`, `template_id`, `template`, `timestamp`, `host` 등
- `baseline_scores.parquet`: `score`, `is_anomaly`, `window_start_line`
- `deeplog_infer.parquet`: `idx`, `target`, `in_topk` (top-k 위반 여부)
- `report.md`: 상위 이상 윈도우와 기여 템플릿/요약 지표

## 🆕 새로운 분석 기능

### 🔍 이상 로그 샘플 분석 (NEW!)

**문제**: 이상탐지 결과만으로는 실제로 어떤 로그가 문제인지 알기 어려움  
**해결**: 실제 문제 로그들을 사람이 읽기 쉬운 형태로 추출하고 분석

#### 🎬 빠른 데모
```bash
# 전체 기능을 한번에 체험
./demo_log_samples.sh
```

#### 🔧 주요 기능
- **실제 로그 샘플**: 이상탐지 결과에서 문제가 되는 실제 로그들 추출
- **전후 맥락**: 이상 로그의 앞뒤 상황을 함께 표시
- **패턴 분석**: 왜 이상으로 판단되었는지 설명
- **사람 친화적**: 기술적 결과를 일반인도 이해할 수 있게 번역

#### 📄 생성되는 리포트 예시
```markdown
# 이상 로그 샘플 분석 리포트

## 🚨 이상 윈도우 #1 (라인 250~)
**기본 정보**: 이상 점수 0.95, 새 템플릿 비율 40%

**대표적인 문제 로그들**:
ERROR (에러 메시지 포함):
[2025-09-20 14:32:15] kernel: BUG: unable to handle page fault
- 템플릿: kernel BUG at <PATH>:<NUM>

**전후 맥락**:
[이전] normal CPU activity...
[이후] system attempting recovery...
```

### 📊 배치 로그 분석

#### 🔹 기본 배치 분석
단일 디렉토리의 로그 파일들을 분석:

```bash
# 폴더 내 모든 로그 파일 분석
./run_batch_analysis.sh /path/to/logs/

# 특정 파일을 Target으로 지정
./run_batch_analysis.sh /path/to/logs/ server1.log my_analysis
```

#### 🌟 향상된 배치 분석 (추천)
하위 디렉토리 재귀 스캔으로 날짜별/카테고리별 구조 지원:

```bash
# 하위 디렉토리 포함 전체 스캔
./run_enhanced_batch_analysis.sh /var/log/

# 세부 옵션 지정 (디렉토리, Target파일, 깊이, 최대파일수, 결과폴더)
./run_enhanced_batch_analysis.sh /logs/2025/09/ app.log 3 20 analysis_result

# 결과 확인
cat analysis_result/ENHANCED_ANALYSIS_SUMMARY.md
```

**💡 새로운 기능**: 이제 향상된 배치 분석에서 **이상 로그 샘플 추출**이 자동으로 포함됩니다!
- 🔍 실제 문제가 되는 로그들을 사람이 읽기 쉬운 형태로 추출
- 📄 전후 맥락과 함께 이상 패턴 설명 제공
- 🎯 이상탐지 결과를 실제 로그와 연결하여 해석 용이

**지원하는 디렉토리 구조**:
```
logs/
├── 2025-09-15/server1/application.log    # 날짜별 구조
├── 2025-09-16/server2/system.log
├── web-servers/nginx.log                 # 서비스별 구조  
└── databases/mysql.log
```

### 🕐 시간 기반 이상 탐지
시간대별/요일별 패턴 학습으로 이상 탐지:

```bash
python temporal_anomaly_detector.py --data-dir data/processed
cat data/processed/temporal_analysis/temporal_report.md
```

### 📈 파일별 비교 이상 탐지  
여러 파일 간 패턴 차이로 이상 탐지:

```bash
python comparative_anomaly_detector.py \
  --target server1/parsed.parquet \
  --baselines server2/parsed.parquet server3/parsed.parquet
```

**분석 방법 비교**:
- **기존 윈도우 방식**: 단일 파일 내 시간순 패턴 변화
- **시간 기반 탐지**: 과거 동일 시간대와 현재 비교  
- **파일별 비교**: 여러 시스템/서비스 간 상대적 차이
