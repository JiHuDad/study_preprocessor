# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**study-preprocessor** is a log anomaly detection framework for kernel/system logs. It provides preprocessing pipelines and multiple detection methods including DeepLog (LSTM-based), MS-CRED (multi-scale convolutional autoencoder), and baseline statistical methods. The project supports train/inference separation for efficient model reuse.

## Key Architecture

### Core Pipeline Flow
1. **Preprocessing** (`study_preprocessor/preprocess.py`): Drain3 template mining + regex-based masking
2. **Model Input Builders** (`study_preprocessor/builders/`): Generate sequences (DeepLog) or window counts (MS-CRED)
3. **Detection** (`study_preprocessor/detect.py`): Baseline statistical detection + model inference
4. **Analysis**: Temporal, comparative, and log sample analysis tools

### Train/Inference Separation (Recommended Workflow)
- **Training Phase**: Use `train_models.sh` to learn from normal logs → produces reusable models
- **Inference Phase**: Use `run_inference.sh` to detect anomalies in target logs with trained models
- **Benefits**: Train once, reuse for multiple targets; consistent detection criteria; automated validation

### Module Structure
```
study_preprocessor/           # Core Python package
├── cli.py                    # Click-based CLI entry point
├── preprocess.py             # Log parsing with Drain3 + masking
├── detect.py                 # Baseline anomaly detection
├── synth.py                  # Synthetic log generation
├── eval.py                   # Evaluation metrics (P/R/F1)
├── mscred_model.py           # MS-CRED training/inference
└── builders/
    ├── deeplog.py            # DeepLog LSTM sequence builder
    └── mscred.py             # MS-CRED window counts builder

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
# Run inference with trained models (includes log sample extraction)
./run_inference.sh my_models /path/to/suspicious.log

# View human-readable anomaly analysis
cat inference_*/log_samples_analysis/anomaly_analysis_report.md
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
study-preprocess parse --input file.log --out-dir processed/ --drain-state .cache/drain3.json

# Build DeepLog inputs (vocab + sequences)
study-preprocess build-deeplog --parsed processed/parsed.parquet --out-dir processed/

# Build MS-CRED inputs (window counts)
study-preprocess build-mscred --parsed processed/parsed.parquet --out-dir processed/

# Run baseline detection
study-preprocess detect --parsed processed/parsed.parquet --out-dir processed/

# Train DeepLog model
study-preprocess deeplog-train --seq processed/sequences.parquet --vocab processed/vocab.json --out models/deeplog.pth --epochs 3

# Train MS-CRED model
study-preprocess mscred-train --window-counts processed/window_counts.parquet --out models/mscred.pth --epochs 50

# Run DeepLog inference
study-preprocess deeplog-infer --seq processed/sequences.parquet --model models/deeplog.pth --k 3

# Run MS-CRED inference
study-preprocess mscred-infer --window-counts processed/window_counts.parquet --model models/mscred.pth --threshold 95.0

# Generate comprehensive report
study-preprocess report --processed-dir processed/ --with-samples

# Extract and analyze anomaly log samples (standalone)
study-preprocess analyze-samples --processed-dir processed/
```

### Batch Analysis
```bash
# Enhanced batch analysis with recursive scanning
./run_enhanced_batch_analysis.sh /var/log/ target.log

# Basic batch analysis
./run_batch_analysis.sh /var/log/ target.log my_analysis
```

### Hybrid System (Advanced)
```bash
# Convert PyTorch models to ONNX
study-preprocess convert-onnx --deeplog-model models/deeplog.pth --mscred-model models/mscred.pth --vocab models/vocab.json --output-dir models/onnx

# Run full hybrid pipeline (train → convert → deploy)
study-preprocess hybrid-pipeline --log-file /path/to/train.log --models-dir models --auto-deploy
```

### Testing and Validation
```bash
# Generate synthetic logs for testing
study-preprocess gen-synth --out data/raw/test.log --lines 1000 --anomaly-rate 0.03

# Evaluate detection performance (requires labeled data)
study-preprocess eval --processed-dir processed/ --labels data/raw/test.log.labels.parquet --window-size 50

# Run full E2E pipeline demo
./run_full_pipeline_pip.sh data/raw/test.log

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
