# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Anomaly Log Detector** is a comprehensive log anomaly detection framework for kernel/system logs. It provides preprocessing pipelines and multiple detection methods including DeepLog (LSTM-based), MS-CRED (multi-scale convolutional autoencoder), and baseline statistical methods. The project supports train/inference separation for efficient model reuse.

## Key Architecture

### Core Pipeline Flow
1. **Preprocessing** (`anomaly_log_detector/preprocess.py`): Drain3 template mining + regex-based masking
2. **Model Input Builders** (`anomaly_log_detector/builders/`): Generate sequences (DeepLog) or window counts (MS-CRED)
3. **Detection** (`anomaly_log_detector/detect.py`): Baseline statistical detection + model inference
4. **Analysis**: Temporal, comparative, and log sample analysis tools

### Train/Inference Separation (Recommended Workflow)
- **Training Phase**: Use `train_models.sh` to learn from normal logs → produces reusable models
- **Inference Phase**: Use `run_inference.sh` to detect anomalies in target logs with trained models
- **Benefits**: Train once, reuse for multiple targets; consistent detection criteria; automated validation

### Module Structure
```
anomaly_log_detector/           # Core Python package
├── cli.py                    # Click-based CLI entry point
├── preprocess.py             # Log parsing with Drain3 + masking
├── detect.py                 # Baseline anomaly detection
├── synth.py                  # Synthetic log generation
├── eval.py                   # Evaluation metrics (P/R/F1)
├── mscred_model.py           # MS-CRED training/inference
├── builders/                 # Model input builders
│   ├── deeplog.py            # DeepLog LSTM sequence builder
│   └── mscred.py             # MS-CRED window counts builder
└── analyzers/                # Analysis tools (Phase 3)
    ├── temporal.py           # Time-based anomaly detection
    ├── comparative.py        # Cross-file pattern comparison
    ├── log_samples.py        # Anomaly log sample extraction
    ├── mscred_analysis.py    # MS-CRED specific analysis
    └── baseline_validation.py # Baseline quality validation

hybrid_system/                # Advanced hybrid system (Phase 1.3)
├── training/                 # ONNX conversion for deployment
│   ├── model_converter.py    # PyTorch → ONNX converter
│   ├── batch_trainer.py      # Batch training coordinator
│   └── auto_converter.py     # Automated pipeline
└── inference/                # C inference engine (high-performance)
    └── README.md             # ONNX Runtime-based C API
```

## Common Commands

### Setup and Installation
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package in editable mode (recommended for development)
pip install -e .

# Install hybrid system dependencies (optional)
pip install -r requirements_hybrid.txt
```

### Model Training (Recommended Approach)
```bash
# Train models from normal logs
./train_models.sh /path/to/normal_logs/ my_models

# Validate model quality (0-100 score)
./validate_models.sh my_models

# Incremental training (improve existing models)
./train_models_incremental.sh old_models /path/to/new_normal/ updated_models
```

### Anomaly Detection Inference
```bash
# Run inference with trained models (uses DeepLog Enhanced automatically)
./run_inference.sh my_models /path/to/suspicious.log

# Inference includes:
# - DeepLog Enhanced with K-of-N judgment, cooldown, novelty detection
# - MS-CRED multi-scale reconstruction error
# - Baseline statistical anomaly detection
# - Temporal pattern analysis
# - Anomaly log sample extraction

# View human-readable anomaly analysis
cat inference_*/inference_report.md
cat inference_*/log_samples_analysis/anomaly_analysis_report.md

# View DeepLog Enhanced alerts
cat inference_*/deeplog_alerts.parquet  # Entity-based alerts with cooldown
cat inference_*/deeplog_summary.json    # Session and alert statistics
```

### Model Comparison and Validation
```bash
# Compare two models performance
./compare_models.sh old_models new_models

# Validate model with specific test log
./validate_models.sh my_models /path/to/validation.log
```

### Direct CLI Usage (Lower-Level)
```bash
# Parse raw log file
alog-detect parse --input file.log --out-dir processed/ --drain-state .cache/drain3.json

# Build DeepLog inputs (vocab + sequences)
alog-detect build-deeplog --parsed processed/parsed.parquet --out-dir processed/

# Build MS-CRED inputs (window counts)
alog-detect build-mscred --parsed processed/parsed.parquet --out-dir processed/

# Run baseline detection
alog-detect detect --parsed processed/parsed.parquet --out-dir processed/

# Train DeepLog model
alog-detect deeplog-train --seq processed/sequences.parquet --vocab processed/vocab.json --out models/deeplog.pth --epochs 3

# Train MS-CRED model
alog-detect mscred-train --window-counts processed/window_counts.parquet --out models/mscred.pth --epochs 50

# Run DeepLog inference (기본)
alog-detect deeplog-infer --seq processed/sequences.parquet --model models/deeplog.pth --k 3

# Run Enhanced DeepLog inference (알림 폭주 방지, 엔티티별 세션화)
alog-detect deeplog-infer-enhanced \
  --seq processed/sequences.parquet \
  --parsed processed/parsed.parquet \
  --model models/deeplog.pth \
  --vocab models/vocab.json \
  --top-k 3 \
  --k-of-n-k 7 --k-of-n-n 10 \
  --cooldown-seq 60 --cooldown-novelty 60 \
  --entity-column host

# Top-P 사용 예시 (top-k 대신)
alog-detect deeplog-infer-enhanced \
  --seq processed/sequences.parquet \
  --parsed processed/parsed.parquet \
  --model models/deeplog.pth \
  --vocab models/vocab.json \
  --top-p 0.9 \
  --entity-column host

# Run MS-CRED inference
alog-detect mscred-infer --window-counts processed/window_counts.parquet --model models/mscred.pth --threshold 95.0

# Generate comprehensive report
alog-detect report --processed-dir processed/ --with-samples

# Extract and analyze anomaly log samples (standalone)
alog-detect analyze-samples --processed-dir processed/
```

### Batch Analysis
```bash
# Enhanced batch analysis with recursive scanning
./run_enhanced_batch_analysis.sh /var/log/ target.log

# Basic batch analysis
./run_batch_analysis.sh /var/log/ target.log my_analysis
```

### Analysis Commands (Phase 3 - New!)
```bash
# Time-based anomaly detection
alog-detect analyze-temporal --data-dir processed/ --output-dir temporal_results/

# Cross-file pattern comparison
alog-detect analyze-comparative --target target.parquet --baselines baseline1.parquet --baselines baseline2.parquet

# MS-CRED specific analysis
alog-detect analyze-mscred --data-dir processed/ --output-dir mscred_analysis/

# Baseline quality validation
alog-detect validate-baseline file1.log file2.log --output-dir validation_results/

# Anomaly log sample extraction (already existed)
alog-detect analyze-samples --processed-dir processed/ --max-samples 10 --context-lines 3
```

### Hybrid System (Advanced)
```bash
# Portable mode: 범용 최적화 (권장 - C 추론 엔진용)
alog-detect convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab models/vocab.json \
  --output-dir models/onnx \
  --portable \
  --validate

# Maximum optimization mode: 현재 하드웨어에 특화
alog-detect convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab models/vocab.json \
  --output-dir models/onnx \
  --validate

# Convert with manual feature dimension and sequence length override
alog-detect convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab models/vocab.json \
  --output-dir models/onnx \
  --seq-len 50 \
  --feature-dim 100 \
  --portable

# Run full hybrid pipeline (train → convert → deploy)
alog-detect hybrid-pipeline --log-file /path/to/train.log --models-dir models --auto-deploy
```

**Notes**:
- `--portable` (권장): 범용 최적화 모드, 모든 환경에서 사용 가능. C 추론 엔진 배포 시 권장.
- 기본 모드: 현재 하드웨어에 특화된 최대 최적화. 같은 환경에서만 사용 권장.
- `--seq-len`: DeepLog 시퀀스 길이 오버라이드 (기본값: 모델 checkpoint에 저장된 값 사용)
  - ONNX 모델은 `dynamic_axes` 설정으로 다양한 시퀀스 길이 지원
  - C 추론 시 원하는 길이로 입력 가능 (메타데이터의 seq_len은 권장값)
- `--feature-dim`: MS-CRED 피처 차원 (템플릿 개수), vocab.json에서 자동 감지 (최소 10 보장)
- **Vocab 일관성**: Python vocab은 sorted template string order로 생성되며, C 엔진용으로 자동 변환됨 (동일한 순서 유지)

### Testing and Validation
```bash
# Generate synthetic logs for testing
alog-detect gen-synth --out data/raw/test.log --lines 1000 --anomaly-rate 0.03

# Evaluate detection performance (requires labeled data)
alog-detect eval --processed-dir processed/ --labels data/raw/test.log.labels.parquet --window-size 50

# Run full E2E pipeline demo
./run_full_pipeline_pip.sh data/raw/test.log

# Test Python vs C vocab consistency (Hybrid System)
./scripts/test_python_c_vocab_consistency.sh

# Demo specific features
./demo_log_samples.sh    # Log sample analysis demo
./demo_mscred.sh         # MS-CRED feature demo
./demo_enhanced_batch.sh # Enhanced batch analysis demo
```

## Key Configuration Files

### rules.json
Defines regex-based masking rules for log preprocessing. Masks hex addresses, IPs, paths, device numbers, PIDs, UUIDs, MAC addresses, and numeric values to <HEX>, <IP>, <PATH>, <ID>, <PID>, <UUID>, <MAC>, <NUM> respectively.

### Drain3 State
- **File**: Typically `.cache/drain3.json` or in model directories
- **Purpose**: Persists learned log templates across runs for consistency
- **Usage**: Pass via `--drain-state` to reuse templates when processing multiple files

### Model Artifacts (from train_models.sh)
```
models_YYYYMMDD_HHMMSS/
├── deeplog.pth           # DeepLog LSTM weights
├── mscred.pth            # MS-CRED convolutional weights
├── vocab.json            # Template ID → index mapping
├── baseline_stats.json   # Normal pattern statistics (EWM means, quantiles)
├── drain3_state.json     # Drain3 template state
└── metadata.json         # Training metadata (timestamp, hyperparameters)
```

## Important Output Files

### Preprocessing Outputs
- `parsed.parquet`: Main table with columns: `line_no`, `timestamp`, `host`, `process`, `raw`, `masked`, `template_id`, `template`
- `preview.json`: First 10 rows for quick inspection
- `vocab.json`: Template ID to index mapping for DeepLog
- `sequences.parquet`: Sorted template index sequences for DeepLog
- `window_counts.parquet`: Window-level template counts for MS-CRED

### Detection Outputs
- `baseline_scores.parquet`: Window-level anomaly scores (`score`, `is_anomaly`, `window_start_line`)
- `deeplog_infer.parquet`: Sequence predictions (`idx`, `target`, `in_topk` boolean)
- `mscred_infer.parquet`: Window reconstruction errors (`window_idx`, `reconstruction_error`, `is_anomaly`, `threshold`)
- `report.md`: Summary report with top anomaly windows and dominant templates

### Analysis Outputs
- `log_samples_analysis/anomaly_analysis_report.md`: Human-readable anomaly report with actual log samples and context
- `log_samples_analysis/anomaly_samples.json`: Detailed structured anomaly data
- `temporal_analysis/temporal_report.md`: Time-based pattern analysis (hourly/daily profiles)
- `comparative_analysis/comparative_report.md`: Cross-file comparison results

## Development Practices

### Virtual Environment
Always use `.venv` or `venv` for dependency isolation. Scripts auto-detect and activate if available.

### Editable Installation
For development: `pip install -e .` to reflect code changes immediately without reinstall.

### Drain3 State Management
When processing multiple related files, pass the same `--drain-state` path to ensure template consistency across runs.

### Masking Toggles
All masking is ON by default. Use `--no-mask-*` flags to disable specific masking rules:
- `--no-mask-paths`, `--no-mask-hex`, `--no-mask-ips`, `--no-mask-mac`
- `--no-mask-uuid`, `--no-mask-pid`, `--no-mask-device`, `--no-mask-num`

### Model Training Best Practices
- Use clean "normal" logs for training (no known anomalies)
- Validate models after training with `validate_models.sh` (target: 85+ quality score)
- Use incremental training to improve models with new normal data
- Compare model versions before deploying to production

### Inference Best Practices
- Always use `--with-samples` flag in reports to get actionable anomaly details
- Check `anomaly_analysis_report.md` for human-readable explanations
- Review both DeepLog (sequence-based) and MS-CRED (reconstruction-based) results for comprehensive coverage

## Detection Methods

### Baseline (Statistical)
- **Method**: Sliding window (default 50 lines, stride 25) with EWM statistics
- **Metrics**: New template ratio, frequency change (KL divergence-like)
- **Threshold**: Quantile-based (default 95th percentile)
- **Use Case**: Fast, interpretable, no training required

### DeepLog (LSTM)
- **Method**: LSTM predicts next template given sequence history
- **Metric**: Top-K violation rate (true template not in predicted top-K)
- **Hyperparameters**: `--seq-len` (default 50), `--k` (default 3), `--epochs` (default 3)
- **Use Case**: Sequence pattern anomalies, requires training

#### Enhanced DeepLog (Production-Ready)
- **추가 기능**:
  - **Top-K/Top-P**: Top-K 또는 Top-P (nucleus sampling) 선택 가능
  - **K-of-N 판정**: 엔티티별 슬라이딩 윈도우에서 N개 중 K개 이상 실패 시 알림
  - **Cooldown**: 동일 패턴 재알림 방지 (기본 60초)
  - **노벨티 탐지**: 학습 시 보지 못한 새 템플릿 발견 및 집계
  - **세션화**: 엔티티(host/process)별 세션 관리, 타임아웃 지원
  - **중복 억제**: 알림 시그니처 기반 중복 억제
- **출력**:
  - `deeplog_enhanced_detailed.parquet`: 모든 시퀀스 판정 결과
  - `deeplog_enhanced_alerts.parquet`: 실제 발생한 알림 목록
  - `deeplog_enhanced_summary.json`: 요약 정보
- **Use Case**: 프로덕션 환경에서 알림 폭주 방지, 실용적인 이상 탐지

### MS-CRED (Convolutional Autoencoder)
- **Method**: Multi-scale convolutional encoder-decoder with reconstruction error
- **Metric**: Reconstruction error exceeding threshold (default 95th percentile)
- **Hyperparameters**: `--window-size` (default 50), `--epochs` (default 50)
- **Use Case**: Multi-scale pattern anomalies, captures subtle changes

### Temporal Analysis
- **Method**: Time-of-day and day-of-week profiling
- **Use Case**: Detecting anomalies based on temporal context (e.g., unusual activity at night)
- **Tool**: `temporal_anomaly_detector.py`

### Comparative Analysis
- **Method**: Cross-file pattern comparison using cosine similarity and KL divergence
- **Use Case**: Detecting deviations when comparing target vs. baseline files
- **Tool**: `comparative_anomaly_detector.py`

## Troubleshooting

### Template Explosion
If Drain3 generates too many templates:
- Increase masking (reduce `--no-mask-*` flags)
- Adjust Drain3 `similarity_threshold` in code (default 0.4, increase to merge more)
- Use same `--drain-state` across runs

### Memory Issues
For large log files:
- Process in smaller batches or directories
- Reduce window size: `--window-size 25`
- Use streaming/chunked processing (implemented in preprocess.py)

### Model Training Failures
- Check sufficient normal log data (recommend 1000+ lines minimum)
- Verify vocab size is reasonable (10-500 templates typical)
- Reduce epochs or sequence length for faster iteration
- Check GPU availability for faster training: `torch.cuda.is_available()`

### Inference Issues
- Ensure model files exist: `vocab.json`, `drain3_state.json`, `deeplog.pth`/`mscred.pth`
- Verify target log format matches training data format
- Check that Drain3 state is compatible (same config)

### Hybrid System (Python vs C ONNX Inference)
**Problem**: High false positive rate in C inference (e.g., 50%+ detection with k=38)

**Root Cause**: Vocab index mismatch between Python training and C inference
- Python training creates vocab with: `{t: i for i, t in enumerate(sorted(unique_templates))}`
- Indices are in **sorted template string order** (NOT Drain3 template_id order)
- C engine must use the **same sorted order** for correct inference

**Solution** (Fixed in model_converter.py):
- `_convert_vocab_for_c_engine()` now preserves Python's sorted template string order
- Converts `{template: index}` → `{str(index): template}` maintaining exact order
- C engine loads vocab and sorts by JSON key (index) to match Python order

**Verification**:
```bash
./scripts/test_python_c_vocab_consistency.sh
```

**Important**: Always regenerate ONNX models after fixing vocab conversion:
```bash
alog-detect convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab models/vocab.json \
  --output-dir models/onnx \
  --portable --validate
```

### Vocab Format Error (template_id instead of template string)
**Problem**: ONNX 변환 후 vocab.json이 다음과 같은 잘못된 형식:
```json
{
  "1": 0,
  "2": 1,
  "3": 2
}
```
C 추론 엔진이 실제 템플릿 문자열이 필요한데 template_id만 있음.

**Root Cause**: `build_deeplog_inputs()`가 `template_col="template_id"` 사용
- `template_id`: Drain3가 부여한 ID (예: "1", "2", "3")
- `template`: 실제 템플릿 문자열 (예: "User <ID> logged in")

**Solution** (이미 적용됨):
- [deeplog.py:15](anomaly_log_detector/builders/deeplog.py#L15)에서 기본값을 `template_col="template"`로 변경
- [model_converter.py:72-80](hybrid_system/training/model_converter.py#L72-L80)에서 검증 로직 추가하여 에러 조기 발견

**올바른 vocab 형식**:
```json
{
  "User <ID> logged in": 0,
  "System started successfully": 1,
  "Error: <PATH> not found": 2
}
```

**재생성 방법**:

방법 1 - 전체 재생성 (권장):
```bash
alog-detect build-deeplog --parsed data/parsed.parquet --out-dir training_workspace/
alog-detect deeplog-train --seq training_workspace/sequences.parquet --vocab training_workspace/vocab.json --out training_workspace/deeplog.pth
alog-detect convert-onnx --deeplog-model training_workspace/deeplog.pth --vocab training_workspace/vocab.json --output-dir models/onnx
```

방법 2 - vocab만 변환 (빠름):
```bash
python scripts/fix_vocab_format.py --parsed data/parsed.parquet --old-vocab training_workspace/vocab.json --output training_workspace/vocab_fixed.json
# 백업 후 교체하여 ONNX 변환 (모델 재학습 권장)
```

### ONNX Export Errors (dynamo / mark_dynamic / Dim.DYNAMIC)
**Problem**: PyTorch 2.1+ 환경에서 ONNX 변환 시 다음과 같은 에러 발생:
- `torch.export.export` 호출 관련 에러
- `mark_dynamic` 사용 관련 경고 (maybe_mark_dynamic을 사용하라는 메시지)
- `Dim.DYNAMIC` 사용 관련 경고 (Dim.STATIC 또는 Dim.AUTO로 교체하라는 메시지)

**Root Cause**: PyTorch 2.1+에서 `torch.onnx.export`가 기본적으로 새로운 dynamo 기반 방식을 시도
- 새로운 방식은 일부 모델 구조(LSTM, Embedding 등)와 호환성 문제
- `dynamic_axes` 설정이 dynamo의 `Dim.DYNAMIC`으로 해석되어 충돌

**Solution** (이미 적용됨):
- [model_converter.py](hybrid_system/training/model_converter.py)에서 `dynamo=False` 명시적 지정
- 레거시 TorchScript 방식 강제 사용
- PyTorch 2.0 이하 호환성 유지 (TypeError 예외 처리)

**코드 예시**:
```python
torch.onnx.export(
    model, dummy_input, onnx_path,
    dynamo=False,  # 명시적으로 레거시 방식 사용
    dynamic_axes={...},
    ...
)
```

**다른 시스템에서 에러 발생 시**: 최신 코드를 pull하여 `dynamo=False` 수정사항 반영

### ONNX Sequence Length Constraints Error
**Problem**: `Constraints violated (L['x'].size()[1])` error during C ONNX inference

**Root Cause**: ONNX model input sequence length mismatch
- ONNX 모델 변환 시 특정 `seq_len`으로 더미 입력 생성
- 메타데이터에 `seq_len` 저장 (예: 3, 50 등)
- C 추론 시 다른 길이 입력하면 제약 위반 오류 발생

**Solution**:
1. **ONNX 모델은 이미 `dynamic_axes` 설정으로 다양한 길이 지원**
   - `input_sequence`: `{0: 'batch_size', 1: 'sequence_length'}`로 설정됨
   - 메타데이터의 `seq_len`은 **권장값**이지 강제 제약이 아님

2. **C 추론 시 메타데이터의 `seq_len` 사용 권장**:
   ```c
   // Read metadata
   seq_len = meta["seq_len"];  // 학습 시 사용한 권장 시퀀스 길이

   // Use same length for inference
   int sequence[seq_len] = {...};
   onnx_deeplog_infer(model, sequence, seq_len, vocab_size, predictions);
   ```

3. **다른 시퀀스 길이로 ONNX 변환이 필요한 경우**:
   ```bash
   alog-detect convert-onnx \
     --deeplog-model models/deeplog.pth \
     --vocab models/vocab.json \
     --output-dir models/onnx \
     --seq-len 50  # 원하는 시퀀스 길이 지정
   ```

**Note**:
- `--seq-len`을 지정하지 않으면 모델 checkpoint에 저장된 값 사용
- ONNX 모델 자체는 dynamic_axes로 유연하게 다양한 길이 지원
- 하지만 학습 시 사용한 길이와 동일한 길이 사용 권장 (성능 및 정확도)

### Script Execution
All `.sh` scripts require executable permissions. If needed: `chmod +x *.sh`

## Notes on Korean Documentation

Many documentation files and output reports are in Korean. Key terms:
- 이상탐지 = anomaly detection
- 정상 로그 = normal logs
- 학습 = training
- 추론 = inference
- 템플릿 = template
- 윈도우 = window
