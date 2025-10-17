# Project Structure

Anomaly Log Detector 프로젝트의 디렉토리 구조입니다.

```
anomaly-log-detector/                    # 프로젝트 루트
├── README.md                             # 메인 문서
├── pyproject.toml                        # 패키지 설정
├── requirements.txt                      # 의존성
├── LICENSE                               # 라이센스
│
├── anomaly_log_detector/                 # 🐍 핵심 Python 패키지
│   ├── cli.py                           # CLI 엔트리포인트
│   ├── preprocess.py                    # 로그 전처리
│   ├── detect.py                        # 베이스라인 탐지
│   ├── eval.py                          # 평가 메트릭
│   ├── synth.py                         # 합성 로그 생성
│   ├── mscred_model.py                  # MS-CRED 모델
│   ├── builders/                        # 모델 입력 빌더
│   │   ├── deeplog.py                   # DeepLog 시퀀스 빌더
│   │   └── mscred.py                    # MS-CRED 윈도우 카운트
│   └── analyzers/                       # 분석 도구
│       ├── temporal.py                  # 시간 기반 분석
│       ├── comparative.py               # 비교 분석
│       ├── log_samples.py               # 로그 샘플 추출
│       ├── mscred_analysis.py           # MS-CRED 분석
│       └── baseline_validation.py       # 베이스라인 검증
│
├── scripts/                              # 🔧 실행 스크립트
│   ├── train_models.sh                  # 모델 학습
│   ├── run_inference.sh                 # 추론 실행
│   ├── compare_models.sh                # 모델 비교
│   ├── validate_models.sh               # 모델 검증
│   ├── train_models_incremental.sh      # 점진적 학습
│   ├── run_enhanced_batch_analysis.sh   # 배치 분석
│   ├── run_batch_analysis.sh            # 기본 배치 분석
│   ├── run_baseline_validation.sh       # 베이스라인 검증
│   ├── run_full_pipeline.sh             # 전체 파이프라인
│   ├── run_full_pipeline_pip.sh         # pip 버전 파이프라인
│   ├── demo/                            # 데모 스크립트
│   │   ├── demo_enhanced_batch.sh
│   │   ├── demo_enhanced_deeplog.sh
│   │   ├── demo_log_samples.sh
│   │   └── demo_mscred.sh
│   └── test/                            # 테스트 스크립트
│       ├── test_preprocessing.sh
│       ├── test_onnx_conversion.sh
│       └── test_hybrid_training.py
│
├── config/                               # ⚙️ 설정 파일
│   ├── drain3.ini                       # Drain3 설정
│   └── rules.json                       # 마스킹 규칙
│
├── docs/                                 # 📚 문서
│   ├── README.md                        # 문서 인덱스
│   ├── guides/                          # 사용자 가이드
│   │   ├── BATCH_ANALYSIS_GUIDE.md
│   │   ├── TRAIN_INFERENCE_GUIDE.md
│   │   ├── RESULTS_GUIDE.md
│   │   └── ANOMALY_DETECTION_METHODS.md
│   ├── api/                             # API 문서 (예정)
│   │   ├── cli-reference.md
│   │   └── python-api.md
│   └── development/                     # 개발 문서
│       ├── CONTEXT.md
│       ├── CHANGELOG_REFACTORING.md
│       ├── RENAMING_GUIDE.md
│       ├── CLAUDE.md
│       └── prd.md
│
├── examples/                             # 📖 예제
│   ├── README.md                        # 예제 가이드
│   ├── data/                            # 샘플 로그
│   │   ├── test_sample.log
│   │   └── direct_test.log
│   └── scripts/                         # 예제 스크립트
│       └── generate_test_logs.py
│
├── hybrid_system/                        # 🔄 ONNX 변환 & C 추론
│   ├── README.md
│   ├── training/                        # ONNX 변환
│   │   ├── model_converter.py
│   │   ├── batch_trainer.py
│   │   └── auto_converter.py
│   └── inference/                       # C 추론 엔진
│       └── README.md
│
├── tools/                                # 🛠️ 유틸리티 (deprecated)
│   ├── baseline_validator.py            # → alog-detect validate-baseline
│   ├── enhanced_batch_analyzer.py
│   ├── comparative_anomaly_detector.py  # → alog-detect analyze-comparative
│   ├── temporal_anomaly_detector.py     # → alog-detect analyze-temporal
│   ├── log_sample_analyzer.py           # → alog-detect analyze-samples
│   ├── mscred_analyzer.py               # → alog-detect analyze-mscred
│   └── visualize_results.py
│
├── data/                                 # 📊 사용자 데이터 (.gitignore)
│   ├── raw/
│   └── processed/
│
├── .cache/                               # 💾 캐시 (.gitignore)
│   └── drain3.json
│
└── .venv/                                # 🐍 가상환경 (.gitignore)
```

## 주요 디렉토리 설명

### anomaly_log_detector/
핵심 Python 패키지. 모든 탐지 알고리즘과 전처리 로직이 포함되어 있습니다.

### scripts/
사용자가 직접 실행하는 Shell 스크립트들. 일반적인 작업 흐름을 자동화합니다.

### config/
프로젝트 전체 설정 파일. Drain3 설정과 마스킹 규칙이 포함됩니다.

### docs/
프로젝트 문서. 사용자 가이드, API 레퍼런스, 개발 문서로 구분됩니다.

### examples/
새로운 사용자를 위한 예제 코드와 샘플 데이터.

### hybrid_system/
ONNX 변환 및 C 기반 고성능 추론 엔진.

### tools/
CLI로 통합된 유틸리티 스크립트들 (하위 호환성 유지).

## 빠른 시작

1. **설치**: [README.md](README.md)의 설치 가이드 참조
2. **첫 실행**: [examples/README.md](examples/README.md)의 예제 실행
3. **학습/추론**: [docs/guides/TRAIN_INFERENCE_GUIDE.md](docs/guides/TRAIN_INFERENCE_GUIDE.md) 참조

## 변경 이력

- **2025-10-17**: Phase 1 재구조화 완료
  - config/ 디렉토리 생성
  - scripts/ 구조화 (demo/, test/)
  - docs/ 재구조화 (guides/, api/, development/)
  - examples/ 정리

- **2025-10-16**: 프로젝트 리네이밍
  - study-preprocessor → anomaly-log-detector
  - study_preprocessor → anomaly_log_detector
  - study-preprocess → alog-detect
