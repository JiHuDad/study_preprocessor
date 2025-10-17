# 프로젝트 정리 완료 설명

Phase 2 완료 후 최상단 디렉토리 정리 작업이 완료되었습니다.

## 최상단에 남아있는 파일들 설명

### ✅ 심볼릭 링크 (Backward Compatibility)

```bash
compare_models.sh -> scripts/compare_models.sh
run_inference.sh -> scripts/run_inference.sh
train_models.sh -> scripts/train_models.sh
validate_models.sh -> scripts/validate_models.sh
```

**용도**: 하위 호환성을 위한 심볼릭 링크입니다.

**이유**:
- 기존 사용자가 `./train_models.sh` 같은 명령어를 계속 사용할 수 있도록 함
- 실제 파일은 `scripts/` 디렉토리에 있음
- 삭제하면 안 됨 - 기존 워크플로우가 깨집니다

**검증**:
```bash
$ ls -la *.sh
lrwxrwxrwx train_models.sh -> scripts/train_models.sh
```

### ✅ Requirements 파일들 (모두 필요함)

```
requirements.txt           # 핵심 실행 의존성
requirements-dev.txt       # 개발/테스트 도구
requirements_hybrid.txt    # ONNX/하이브리드 시스템
```

**requirements.txt** (핵심):
```
torch
numpy
pandas
click
drain3
```
- 기본 실행에 필요한 최소 의존성
- `pip install -e .` 시 자동 설치됨

**requirements-dev.txt** (개발):
```
pytest
pytest-cov
pytest-mock
black
flake8
mypy
isort
sphinx
```
- 테스트 실행에 필요 (`pytest`)
- 코드 품질 도구 (`black`, `flake8`)
- 문서 생성 (`sphinx`)
- 개발자만 설치: `pip install -r requirements-dev.txt`

**requirements_hybrid.txt** (고급):
```
onnx
onnxruntime
```
- ONNX 모델 변환/추론에만 필요
- 선택적 기능 (하이브리드 시스템 사용 시)
- 필요한 사용자만 설치: `pip install -r requirements_hybrid.txt`

**왜 분리했나?**:
- 일반 사용자는 ONNX 도구가 필요 없음 (용량 큼)
- 테스트 도구는 프로덕션 환경에 불필요
- 의존성 최소화로 설치 속도 향상

### ✅ 설정 파일들

```
pyproject.toml             # 패키지 설정
pytest.ini                 # pytest 설정
.gitignore                 # Git 제외 파일
```

이들은 프로젝트 루트에 있어야 하는 표준 설정 파일들입니다.

## 이동 완료된 파일들

### tools/ 디렉토리로 이동 (Deprecated Wrappers)

```
baseline_validator.py
log_sample_analyzer.py
comparative_anomaly_detector.py
temporal_anomaly_detector.py
mscred_analyzer.py
enhanced_batch_analyzer.py
visualize_results.py
```

**이유**:
- 모두 CLI로 대체됨 (`alog-detect` 명령어)
- `tools/README.md`에 마이그레이션 가이드 있음
- Deprecated 되었지만 호환성을 위해 보관

**예시**:
```bash
# 이전 방식 (tools/ - deprecated)
python tools/log_sample_analyzer.py --processed-dir processed/

# 새 방식 (CLI - 권장)
alog-detect analyze-samples --processed-dir processed/
```

### docs/development/ 디렉토리로 이동

```
PHASE2_SUMMARY.md          # Phase 2 완료 요약
CLEANUP_EXPLANATION.md     # 이 문서
```

**이유**: 개발 관련 문서들은 `docs/development/`에 위치

## 최종 프로젝트 구조

```
anomaly-log-detector/
├── anomaly_log_detector/      # 메인 Python 패키지
├── config/                    # 설정 파일 (rules.json, drain3.ini)
├── scripts/                   # 실행 스크립트
│   ├── train.sh, infer.sh    # 단순화된 이름
│   ├── train_models.sh       # 원본 이름
│   ├── demo/                 # 데모 스크립트
│   └── test/                 # 테스트 스크립트
├── tools/                     # Deprecated 래퍼 (CLI로 대체됨)
│   ├── README.md             # 마이그레이션 가이드
│   └── *.py                  # 이전 스크립트들
├── tests/                     # pytest 테스트 스위트
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                      # 문서
│   ├── guides/               # 사용자 가이드
│   └── development/          # 개발자 문서
├── examples/                  # 예시 코드/데이터
│   ├── data/
│   └── scripts/
│
├── pyproject.toml            # 패키지 설정
├── pytest.ini                # pytest 설정
├── requirements.txt          # 핵심 의존성
├── requirements-dev.txt      # 개발 의존성
├── requirements_hybrid.txt   # ONNX 의존성
│
└── (심볼릭 링크들)            # 하위 호환성
    ├── train_models.sh -> scripts/train_models.sh
    ├── run_inference.sh -> scripts/run_inference.sh
    ├── compare_models.sh -> scripts/compare_models.sh
    └── validate_models.sh -> scripts/validate_models.sh
```

## 검증 체크리스트

- ✅ 모든 Python 래퍼 파일이 tools/로 이동
- ✅ 개발 문서가 docs/development/로 이동
- ✅ 심볼릭 링크가 정상 작동 (`ls -la *.sh`)
- ✅ 3개 requirements 파일 모두 보존
- ✅ 핵심 설정 파일들 (pyproject.toml, pytest.ini) 최상단 유지

## 기존 워크플로우 호환성

모든 기존 명령어가 그대로 작동합니다:

```bash
# ✅ 여전히 작동
./train_models.sh normal.log models/
./run_inference.sh models/ suspicious.log

# ✅ 새 짧은 이름도 작동
./scripts/train.sh normal.log models/
./scripts/infer.sh models/ suspicious.log

# ✅ CLI 명령어 (권장)
alog-detect parse --input suspicious.log --out-dir processed/
alog-detect deeplog-infer-enhanced --seq processed/sequences.parquet
```

---

**정리 완료일**: 2025-10-17
**Phase**: 2 완료 + 최종 정리
