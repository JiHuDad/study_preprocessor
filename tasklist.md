### Task List

#### ✅ 완료된 작업들

##### 프로젝트 기반 구축
- [x] Write PRD for log preprocessing and anomaly detection scope
- [x] Initialize Python project with uv and pyproject.toml
- [x] Add dependencies (drain3, pandas, pyarrow, click, tqdm, orjson)
- [x] Implement preprocessing module with Drain3 and masking
- [x] Create CLI entrypoint `study-preprocess`
- [x] Add sample kernel log under `data/raw` and generate processed outputs

##### 핵심 기능 구현
- [x] Document example I/O in README section of PRD
- [x] Prepare for DeepLog/MS-CRED downstream pipeline (sequence export)
- [x] Add CLI subcommands: `build-deeplog`, `build-mscred`
- [x] Add masking customization options and config examples
- [x] Implement baseline anomaly detection (window-based frequency analysis)
- [x] Implement DeepLog LSTM model training and inference
- [x] Add synthetic log generation for testing
- [x] Add evaluation metrics (Precision, Recall, F1)
- [x] Create reporting functionality for analysis results

##### 배치 분석 시스템
- [x] Implement basic batch log analyzer
- [x] Create enhanced batch analyzer with recursive directory scanning
- [x] Add support for hierarchical log structures (date/category folders)
- [x] Implement log file validation and error handling
- [x] Create automation scripts for full pipeline execution

##### 고급 이상탐지 방법
- [x] Implement temporal anomaly detection (time-based pattern comparison)
- [x] Implement comparative anomaly detection (cross-file pattern analysis)
- [x] Add multiple similarity metrics (cosine, KL divergence, jaccard)
- [x] Create time-based profiling and seasonal pattern detection

##### 문서화 및 사용성
- [x] Create comprehensive README with usage examples
- [x] Write batch analysis guide (BATCH_ANALYSIS_GUIDE.md)
- [x] Document anomaly detection methods (ANOMALY_DETECTION_METHODS.md)
- [x] Create results interpretation guide (RESULTS_GUIDE.md)
- [x] Add automation scripts with error handling
- [x] Create masking rules configuration file (rules.json)
- [x] Write project context documentation (CONTEXT.md)

#### 🔄 진행 중인 작업들
- [ ] 없음 (주요 기능 완성)

#### 🚀 향후 계획
- [ ] Real-time streaming log analysis
- [ ] Web dashboard interface for monitoring
- [ ] Integration with cloud logging services
- [ ] Distributed processing with Spark
- [ ] Additional ML models for anomaly detection
- [ ] Custom alert and notification system
