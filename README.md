### Anomaly Log Detector 사용 가이드

**Anomaly Log Detector**는 커널/시스템 로그(.log) 파일에 전처리와 이상탐지를 적용하는 프레임워크입니다. DeepLog, LogBERT, MS-CRED, 그리고 통계적 베이스라인 방법을 제공합니다. 모든 예시는 `venv + pip` 기반으로 실행합니다.

## 🆕 **최신 업데이트 (2025-10-02)**

### ✨ **새로운 주요 기능들:**
- **🔄 학습/추론 분리 워크플로우**: 모델 학습과 추론을 분리하여 효율성 극대화
- **📊 모델 비교 도구**: 서로 다른 시점 모델들의 성능 객관적 비교
- **🔄 점진적 학습**: 기존 모델에 새로운 데이터를 추가하여 지속적 개선
- **🔍 자동화된 모델 검증**: 0-100점 품질 점수로 모델 상태 자동 평가
- **📋 실제 이상 로그 샘플 추출**: 이상탐지 결과에서 문제 로그들을 자동 추출 및 분석
- **🎯 외부 Target 파일 지원**: 다른 디렉토리의 파일을 Target으로 지정 가능
- **📄 종합 리포트 통합**: 모든 분석 결과를 하나의 리포트로 통합

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
alog-detect parse \
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
alog-detect parse \
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
  alog-detect parse --input "$f" --out-dir "$OUT/$(basename "$f" .log)" --drain-state "$STATE"
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

#### 4) DeepLog/LogBERT/MSCRED 입력 생성
- DeepLog 입력(사전/시퀀스):
```
alog-detect build-deeplog \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir
```
- LogBERT 입력(사전/시퀀스 + 특수 토큰):
```
alog-detect build-logbert \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir \
  --max-seq-len 512
```
- MS-CRED 입력(윈도우 카운트):
```
alog-detect build-mscred \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir \
  --window-size 50 --stride 25
```

#### 5) 이상탐지 실행
- 베이스라인(새 템플릿 비율 + 빈도 급변):
```
alog-detect detect \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir \
  --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.95
```
- DeepLog 학습/추론:
```
alog-detect deeplog-train \
  --seq /path/to/outdir/sequences.parquet \
  --vocab /path/to/outdir/vocab.json \
  --out .cache/deeplog.pth --seq-len 50 --epochs 3

alog-detect deeplog-infer \
  --seq /path/to/outdir/sequences.parquet \
  --model .cache/deeplog.pth --k 3
```
- LogBERT 학습/추론 (BERT 기반):
```
alog-detect logbert-train \
  --seq /path/to/outdir/sequences.parquet \
  --vocab /path/to/outdir/vocab.json \
  --out .cache/logbert.pth --seq-len 128 --epochs 10

alog-detect logbert-infer \
  --seq /path/to/outdir/sequences.parquet \
  --model .cache/logbert.pth \
  --vocab /path/to/outdir/vocab.json \
  --threshold-percentile 95.0
```
- MS-CRED 학습/추론:
```
alog-detect mscred-train \
  --window-counts /path/to/outdir/window_counts.parquet \
  --out .cache/mscred.pth --epochs 50

alog-detect mscred-infer \
  --window-counts /path/to/outdir/window_counts.parquet \
  --model .cache/mscred.pth --threshold 95.0
```
- 리포트/요약 생성:
```
# 기본 리포트
alog-detect report --processed-dir /path/to/outdir

# 이상 로그 샘플 포함 리포트
alog-detect report --processed-dir /path/to/outdir --with-samples
```
  - 포함: 베이스라인 이상 윈도우 비율, 상위 윈도우/템플릿, DeepLog 위반율
  - `--with-samples`: 실제 문제 로그 샘플과 분석 추가

- 이상 로그 샘플 분석 (단독):
```
alog-detect analyze-samples --processed-dir /path/to/outdir
```
  - 🔍 이상탐지 결과에서 실제 문제 로그들 추출
  - 📄 사람이 읽기 쉬운 분석 리포트 생성
  - 🎯 전후 맥락과 함께 이상 패턴 설명

#### 6) 합성 데이터로 E2E 검증(옵션)
```
# 합성 로그 + 라벨 생성
alog-detect gen-synth --out data/raw/synth_long.log --lines 1000 --anomaly-rate 0.03

# 전처리 → 빌더 → 탐지 → 학습/추론 → 리포트/평가
alog-detect parse --input data/raw/synth_long.log --out-dir data/processed/synth --drain-state .cache/drain3.json
alog-detect build-deeplog --parsed data/processed/synth/parsed.parquet --out-dir data/processed/synth
alog-detect detect --parsed data/processed/synth/parsed.parquet --out-dir data/processed/synth --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.95
alog-detect deeplog-train --seq data/processed/synth/sequences.parquet --vocab data/processed/synth/vocab.json --out .cache/deeplog_synth.pth --seq-len 20 --epochs 2
alog-detect deeplog-infer --seq data/processed/synth/sequences.parquet --model .cache/deeplog_synth.pth --k 3
alog-detect report --processed-dir data/processed/synth
alog-detect eval --processed-dir data/processed/synth --labels data/raw/synth_long.log.labels.parquet --window-size 50 --seq-len 20
```

#### 7) 문제 해결 팁
- 템플릿이 과도하게 늘어나는 경우: 마스킹을 더 강하게 하거나 `--drain-state`를 유지하며 순서대로 처리
- 타임스탬프 파싱 실패: 라인 인덱스(`line_no`) 기준으로도 정렬되며, 포맷이 다른 경우 전처리 규칙 보강 필요
- 메모리: 대형 파일은 디렉터리 단위로 나눠 처리 후 병합 권장

#### 8) 자동화 스크립트 (한번에 실행)
전체 파이프라인을 한번에 실행할 수 있는 스크립트 제공:

**uv 환경용:**
```bash
./scripts/run_full_pipeline.sh /path/to/your.log [출력디렉토리]
```

**pip/venv 환경용:**
```bash
./scripts/run_full_pipeline_pip.sh /path/to/your.log [출력디렉토리]
```

자동 기능:
- 가상환경 자동 감지 및 활성화 (.venv, venv)
- 의존성 자동 설치 (필요시)
- 에러 처리 및 진행 상황 표시
- 결과 파일 자동 정리 및 요약

#### 9) 🆕 **학습/추론 분리 워크플로우** ⭐ **추천**

효율적인 모델 재사용을 위한 새로운 워크플로우:

**1단계: 모델 학습**
```bash
# 정상 로그로 모델 학습
./scripts/train_models.sh /var/log/normal/ my_models

# 모델 품질 검증
./scripts/validate_models.sh my_models
```

**2단계: 이상탐지 추론**
```bash
# Target 로그 이상탐지 (실제 로그 샘플 포함)
./scripts/run_inference.sh my_models /var/log/suspicious.log

# 이상 로그 샘플 확인
cat inference_*/log_samples_analysis/anomaly_analysis_report.md
```

**고급 기능들:**
```bash
# 모델 성능 비교
./scripts/compare_models.sh old_models new_models

# 점진적 학습 (기존 모델 개선)
./scripts/train_models_incremental.sh old_models /var/log/new_normal/ updated_models
```

**장점:**
- 🔄 **효율성**: 한 번 학습하면 여러 Target에 재사용
- 📊 **일관성**: 동일한 기준으로 일관된 이상탐지
- 🔍 **검증**: 자동화된 모델 품질 평가 (0-100점)
- 📋 **샘플**: 실제 문제 로그들을 자동 추출 및 분석

#### 10) 산출물 해석 요약
- `parsed.parquet`: `raw`, `masked`, `template_id`, `template`, `timestamp`, `host` 등
- `baseline_scores.parquet`: `score`, `is_anomaly`, `window_start_line`
- `deeplog_infer.parquet`: `idx`, `target`, `in_topk` (top-k 위반 여부)
- 🆕 `logbert_infer.parquet`: `seq_idx`, `avg_loss`, `is_anomaly`, `threshold` (BERT 기반 이상 점수)
- `mscred_infer.parquet`: `window_idx`, `reconstruction_error`, `is_anomaly`, `threshold`
- `report.md`: 상위 이상 윈도우와 기여 템플릿/요약 지표
- 🆕 `anomaly_analysis_report.md`: 실제 이상 로그 샘플들과 상세 분석

## 🆕 새로운 이상탐지 방법

### 🤖 LogBERT - BERT 기반 로그 이상탐지 (NEW!)

**특징**: Transformer 아키텍처 기반 양방향 컨텍스트 학습
**장점**:
- 양방향 컨텍스트로 정교한 패턴 학습
- Masked Language Model(MLM) 방식으로 정상 로그 패턴 학습
- 긴 시퀀스 의존성 포착 가능
- 임베딩 공간에서 의미적 유사성 학습

#### 🚀 LogBERT 사용법
```bash
# 1. LogBERT 입력 생성 (특수 토큰 포함)
alog-detect build-logbert --parsed data/processed/parsed.parquet --out-dir data/processed

# 2. 모델 학습 (Masked Language Modeling)
alog-detect logbert-train \
  --seq data/processed/sequences.parquet \
  --vocab data/processed/vocab.json \
  --out models/logbert.pth \
  --seq-len 128 \
  --epochs 10 \
  --batch-size 32 \
  --hidden-size 256 \
  --num-layers 4 \
  --num-heads 8

# 3. 이상탐지 추론
alog-detect logbert-infer \
  --seq data/processed/sequences.parquet \
  --model models/logbert.pth \
  --vocab data/processed/vocab.json \
  --threshold-percentile 95.0

# 4. 결과 확인
cat data/processed/logbert_infer.parquet
```

**주요 파라미터**:
- `--seq-len`: 시퀀스 길이 (기본값: 128, BERT의 컨텍스트 윈도우)
- `--hidden-size`: 은닉층 크기 (기본값: 256, 작은 모델용)
- `--num-layers`: Transformer 레이어 수 (기본값: 4)
- `--num-heads`: Attention head 수 (기본값: 8)
- `--mask-ratio`: 학습 시 마스킹 비율 (기본값: 0.15, BERT 표준)

### 🔬 MS-CRED 멀티스케일 분석

**특징**: 멀티스케일 컨볼루션 오토인코더로 윈도우 단위 패턴 분석
**장점**: 다양한 스케일의 패턴을 동시에 고려하여 미세한 이상도 탐지 가능

#### 🚀 MS-CRED 사용법
```bash
# 1. MS-CRED 입력 생성
alog-detect build-mscred --parsed data/processed/parsed.parquet --out-dir data/processed

# 2. 모델 학습
alog-detect mscred-train --window-counts data/processed/window_counts.parquet --out models/mscred.pth --epochs 50

# 3. 이상탐지 추론
alog-detect mscred-infer --window-counts data/processed/window_counts.parquet --model models/mscred.pth --threshold 95.0

# 4. 결과 분석
alog-detect analyze-mscred --data-dir data/processed
```

## 🆕 새로운 분석 기능

### 🔍 이상 로그 샘플 분석 (NEW!)

**문제**: 이상탐지 결과만으로는 실제로 어떤 로그가 문제인지 알기 어려움  
**해결**: 실제 문제 로그들을 사람이 읽기 쉬운 형태로 추출하고 분석

#### 🎬 빠른 데모
```bash
# 전체 기능을 한번에 체험
./scripts/demo/demo_log_samples.sh

# MS-CRED 기능 데모
./scripts/demo/demo_mscred.sh
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
> **참고**: `run_batch_analysis.sh`는 이제 `run_enhanced_batch_analysis.sh`의 wrapper입니다.
> 직접 향상된 버전을 사용하는 것을 권장합니다.

```bash
# 폴더 내 모든 로그 파일 분석 (향상된 버전으로 자동 리디렉션)
./scripts/run_batch_analysis.sh /path/to/logs/

# 특정 파일을 Target으로 지정
./scripts/run_batch_analysis.sh /path/to/logs/ server1.log my_analysis
```

#### 🌟 향상된 배치 분석 (추천)
하위 디렉토리 재귀 스캔으로 날짜별/카테고리별 구조 지원:

```bash
# 기본 사용법: 자동 날짜/시간 폴더 생성
./scripts/run_enhanced_batch_analysis.sh /var/log/

# Target 파일 지정 (같은 디렉토리 내)
./scripts/run_enhanced_batch_analysis.sh /var/log/ system.log

# 🆕 외부 Target 파일 지원 (다른 디렉토리)
./scripts/run_enhanced_batch_analysis.sh /var/log/baseline/ /var/log/target/problem.log

# 세부 옵션 지정 (디렉토리, Target파일, 깊이, 최대파일수, 결과폴더)
./scripts/run_enhanced_batch_analysis.sh /logs/2025/09/ app.log 3 20 my_analysis

# 결과 확인 - 🆕 통합 종합 리포트
cat my_analysis/COMPREHENSIVE_ANALYSIS_REPORT.md

# 🆕 향상된 배치 분석 데모
./demo_enhanced_batch.sh
```

**🆕 최신 향상 사항**:
- 🎯 **외부 Target 파일**: 다른 디렉토리의 파일을 Target으로 지정 가능
- 📊 **20개 로그 샘플**: 타입별 최대 20개 이상 로그 샘플 자동 추출 (기존 10개 → 20개)
- 📄 **종합 리포트**: 모든 분석 결과를 `COMPREHENSIVE_ANALYSIS_REPORT.md` 하나로 통합
- 🛡️ **Baseline 품질 검증**: 문제있는 Baseline 파일 자동 필터링

## 🆕 **새로운 고급 도구들**

### 🔧 **모델 학습 도구**
```bash
# 정상 로그로 모델 학습
./scripts/train_models.sh /var/log/normal/ my_models

# 점진적 학습 (기존 모델 개선)
./scripts/train_models_incremental.sh old_models /var/log/new_normal/ updated_models
```

### 🔍 **모델 검증 및 비교**
```bash
# 모델 품질 검증 (0-100점 품질 점수)
./scripts/validate_models.sh my_models

# 두 모델 성능 비교
./scripts/compare_models.sh old_models new_models
```

### 🎯 **이상탐지 추론**
```bash
# Target 로그 이상탐지 (실제 로그 샘플 포함)
./scripts/run_inference.sh my_models /var/log/suspicious.log

# 결과 확인
cat inference_*/log_samples_analysis/anomaly_analysis_report.md
```

### 📋 **상세 가이드**
- **전체 워크플로우**: `TRAIN_INFERENCE_GUIDE.md` 참조
- **배치 분석**: `BATCH_ANALYSIS_GUIDE.md` 참조
- 🔍 **Target 검증 강화**: 잘못된 Target 지정 시 안전한 에러 처리

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
alog-detect analyze-temporal --data-dir data/processed
cat data/processed/temporal_analysis/temporal_report.md
```

### 📈 파일별 비교 이상 탐지
여러 파일 간 패턴 차이로 이상 탐지:

```bash
alog-detect analyze-comparative \
  --target server1/parsed.parquet \
  --baselines server2/parsed.parquet --baselines server3/parsed.parquet
```

**분석 방법 비교**:
- **기존 윈도우 방식**: 단일 파일 내 시간순 패턴 변화
- **시간 기반 탐지**: 과거 동일 시간대와 현재 비교  
- **파일별 비교**: 여러 시스템/서비스 간 상대적 차이

## 🔄 Hybrid System (ONNX 변환 & C 추론)

고성능 C 추론 엔진을 위한 ONNX 변환 및 자동화 도구입니다.

### 📋 주요 기능

- **자동 모델 변환**: PyTorch 모델을 ONNX로 자동 변환
- **파일 시스템 감시**: 새 모델 생성 시 자동 변환 (watch 모드)
- **배치 학습 파이프라인**: 학습부터 배포까지 전체 자동화
- **C 추론 엔진**: 고성능 실시간 이상탐지

### 🚀 빠른 시작

#### 1. 자동 모델 변환 및 배포

```bash
# 감시 모드: 새 모델 생성 시 자동 변환
python -m hybrid_system.training.auto_converter --mode watch

# 일괄 변환: 기존 모델들 변환
python -m hybrid_system.training.auto_converter --mode convert

# 전체 파이프라인: 학습 → 변환 → 배포
python -m hybrid_system.training.auto_converter \
    --mode pipeline \
    --log-file data/raw/log.log
```

#### 2. 배치 학습 파이프라인

```bash
# 전체 학습 파이프라인 자동 실행
python -m hybrid_system.training.batch_trainer \
    data/raw/log.log \
    --output-dir data/processed/batch_$(date +%Y%m%d_%H%M%S)
```

#### 3. ONNX 변환 (수동)

```bash
# DeepLog 모델 변환
python -m hybrid_system.training.model_converter \
    --deeplog-model models/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models
```

**출력 파일**:
- ✅ `deeplog.onnx` - ONNX 모델
- ✅ `deeplog_optimized.onnx` - 최적화된 ONNX 모델  
- ✅ `vocab.json` - **자동으로 C 엔진용 형식으로 변환됨!**
- ✅ `deeplog.onnx.meta.json` - 모델 메타데이터

#### 4. C 추론 엔진 사용

자세한 내용: [hybrid_system/inference/README.md](hybrid_system/inference/README.md)

```bash
# ONNX Runtime 설치
./scripts/install_onnxruntime.sh

# Inference Engine 빌드
cd hybrid_system/inference
make clean && make

# 실행
./bin/inference_engine \
    -d models/deeplog.onnx \
    -v models/vocab.json \
    -i /var/log/syslog \
    -o results.json
```

### 📚 상세 가이드

- **ONNX 변환**: [docs/guides/ONNX_CONVERSION_GUIDE.md](docs/guides/ONNX_CONVERSION_GUIDE.md)
- **C 추론 엔진**: [hybrid_system/inference/README.md](hybrid_system/inference/README.md)
- **자동 변환**: `python -m hybrid_system.training.auto_converter --help`
