# Project Renaming Guide

## 변경 사항 요약

프로젝트가 **study-preprocessor**에서 **Anomaly Log Detector**로 전면 리네이밍되었습니다.

### 주요 변경사항

| 항목 | 이전 | 이후 |
|------|------|------|
| **프로젝트명** | study-preprocessor | anomaly-log-detector |
| **패키지명** | study_preprocessor | anomaly_log_detector |
| **CLI 명령어** | `study-preprocess` | `alog-detect` |
| **디렉토리** | study_preprocessor/ | anomaly_log_detector/ |

## 사용자 액션 필요 사항

### 1. 기존 패키지 제거 및 재설치

```bash
# 가상환경 활성화
source .venv/bin/activate

# 기존 패키지 제거
pip uninstall study-preprocessor

# 새 패키지 설치
pip install -e .
```

### 2. 명령어 업데이트

모든 `study-preprocess` 명령어를 `alog-detect`로 변경하세요:

**이전:**
```bash
study-preprocess parse --input file.log --out-dir output/
study-preprocess analyze-temporal --data-dir processed/
```

**이후:**
```bash
alog-detect parse --input file.log --out-dir output/
alog-detect analyze-temporal --data-dir processed/
```

### 3. Python Import 업데이트

Python 코드에서 import 경로를 변경하세요:

**이전:**
```python
from study_preprocessor.preprocess import LogPreprocessor
from study_preprocessor.builders.deeplog import build_deeplog_inputs
```

**이후:**
```python
from anomaly_log_detector.preprocess import LogPreprocessor
from anomaly_log_detector.builders.deeplog import build_deeplog_inputs
```

### 4. Shell 스크립트

프로젝트의 모든 shell 스크립트는 자동으로 업데이트되었습니다. 하지만 **외부 스크립트**나 **자동화 도구**를 사용하는 경우 수동으로 업데이트해야 합니다.

## 하위 호환성

### Wrapper 스크립트 (임시 지원)

다음 파일들은 하위 호환성을 위해 유지되지만, deprecated 경고가 표시됩니다:

- `temporal_anomaly_detector.py`
- `comparative_anomaly_detector.py`
- `log_sample_analyzer.py`
- `mscred_analyzer.py`
- `baseline_validator.py`

이들은 새 CLI 명령어로 리디렉션됩니다. 가능한 한 빨리 새 명령어로 마이그레이션하세요.

## 변경된 파일 목록

### 핵심 파일
- ✅ `pyproject.toml` - 패키지명, CLI 엔트리포인트
- ✅ `study_preprocessor/` → `anomaly_log_detector/` - 디렉토리 전체 이동

### Python 파일 (import 경로 자동 업데이트)
- ✅ 모든 Python 파일의 import 문
- ✅ hybrid_system/ 디렉토리의 모든 파일
- ✅ Wrapper 스크립트들

### Shell 스크립트 (16개 파일)
- ✅ train_models.sh
- ✅ run_inference.sh
- ✅ compare_models.sh
- ✅ validate_models.sh
- ✅ run_enhanced_batch_analysis.sh
- ✅ demo_*.sh (모든 데모 스크립트)
- ✅ 기타 모든 .sh 파일

### 문서 파일
- ✅ README.md
- ✅ CLAUDE.md
- ✅ BATCH_ANALYSIS_GUIDE.md
- ✅ CONTEXT.md
- ✅ TRAIN_INFERENCE_GUIDE.md
- ✅ RESULTS_GUIDE.md
- ✅ .cursor/rules/*.mdc

## 검증

다음 명령어로 변경사항을 확인하세요:

```bash
# CLI 작동 확인
alog-detect --help

# 특정 명령어 테스트
alog-detect parse --help
alog-detect analyze-samples --help

# 패키지 import 확인
python -c "from anomaly_log_detector.cli import main; print('Import successful!')"
```

## 롤백 (비상시)

만약 문제가 발생하면 Git으로 롤백할 수 있습니다:

```bash
# 변경사항 확인
git status

# 특정 파일 복구
git checkout -- <파일명>

# 전체 롤백 (커밋 전)
git reset --hard HEAD
```

## 추가 정보

### 새 프로젝트 설명

**Anomaly Log Detector**는 커널/시스템 로그에 대한 종합적인 이상 탐지 프레임워크입니다:

- **DeepLog**: LSTM 기반 시퀀스 학습
- **MS-CRED**: Multi-scale convolutional autoencoder
- **Baseline**: 통계적 이상 탐지
- **Enhanced Features**: K-of-N 판정, 쿨다운, 노벨티 탐지, 세션화

### 프로젝트 목표

이번 리네이밍은 프로젝트의 실제 목적과 기능을 더 명확히 반영하기 위함입니다:

- ❌ "study-preprocessor" - 전처리만 하는 것처럼 보임
- ✅ "Anomaly Log Detector" - 실제 목적인 이상 탐지를 명확히 표현

## 문의

문제가 발생하면 GitHub Issues를 통해 보고해주세요.

---

**작성일**: 2025-10-17
**버전**: 0.1.0 → 0.1.0 (major renaming)
