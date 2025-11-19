# CLAUDE.md - AI Assistant Guide

**Last Updated**: 2025-11-19
**Repository**: Anomaly Log Detector
**Version**: 0.1.0
**Python**: 3.11+

---

## 📋 Table of Contents

1. [Repository Overview](#repository-overview)
2. [Architecture & Structure](#architecture--structure)
3. [Key Components](#key-components)
4. [Development Workflows](#development-workflows)
5. [Code Conventions](#code-conventions)
6. [Testing Guidelines](#testing-guidelines)
7. [Common Tasks](#common-tasks)
8. [Important Gotchas](#important-gotchas)
9. [Quick Reference](#quick-reference)

---

## 🎯 Repository Overview

### Purpose
**Anomaly Log Detector** is a production-ready log anomaly detection framework for kernel/system logs. It provides three detection methods:

- **DeepLog**: LSTM-based sequence anomaly detection
- **MS-CRED**: Multi-scale convolutional auto-encoder for pattern analysis
- **Baseline**: Statistical detection using template novelty and frequency spikes

### Primary Language & Context
- **Documentation**: Korean (README, guides)
- **Code Comments**: Korean inline, English for public APIs
- **User Base**: Korean-speaking DevOps/Security engineers
- **Use Case**: Train on normal logs, detect anomalies in suspicious logs

### Entry Points
```bash
# CLI (25+ commands)
alog-detect --help

# Shell scripts (automated workflows)
./scripts/train_models.sh
./scripts/run_inference.sh
./scripts/run_enhanced_batch_analysis.sh

# Python API
from anomaly_log_detector import LogPreprocessor, build_deeplog_inputs
```

### Key Features
- ✅ Separated training/inference workflows for efficiency
- ✅ Model comparison and validation (0-100 quality scores)
- ✅ Incremental learning capabilities
- ✅ Real anomaly log sample extraction with context
- ✅ ONNX conversion for high-performance C inference
- ✅ Comprehensive integrated reports with actionable insights

---

## 🏗️ Architecture & Structure

### Directory Layout
```
/home/user/study_preprocessor/
├── anomaly_log_detector/       # Core Python package
│   ├── cli.py                  # Main CLI (1,081 lines, 25+ commands)
│   ├── preprocess.py           # Log parsing with Drain3
│   ├── detect.py               # Baseline detection
│   ├── eval.py                 # Evaluation metrics
│   ├── synth.py                # Synthetic log generation
│   ├── mscred_model.py         # MS-CRED implementation
│   ├── builders/               # Model input builders
│   │   ├── deeplog.py          # DeepLog sequences (41K lines)
│   │   └── mscred.py           # MS-CRED window counts (6K lines)
│   └── analyzers/              # Analysis tools
│       ├── log_samples.py      # Log sample extraction (60K lines)
│       ├── temporal.py         # Time-based analysis
│       ├── comparative.py      # Multi-file comparison
│       ├── mscred_analysis.py  # MS-CRED analysis
│       └── baseline_validation.py
├── scripts/                    # Shell automation (15+ scripts)
│   ├── train_models.sh         # Training workflow
│   ├── run_inference.sh        # Inference workflow
│   ├── compare_models.sh       # Model comparison
│   ├── validate_models.sh      # Model validation
│   └── demo/                   # Demo scripts
├── hybrid_system/              # ONNX + C inference
│   ├── training/               # PyTorch → ONNX conversion
│   │   ├── model_converter.py
│   │   ├── batch_trainer.py
│   │   └── auto_converter.py
│   └── inference/              # C inference engine
│       ├── src/                # C source
│       ├── include/            # Headers
│       └── models/             # ONNX models
├── tools/                      # Standalone utilities (deprecated)
├── docs/                       # Documentation
│   ├── guides/                 # 10+ user guides
│   └── development/            # Development docs
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── examples/                   # Example scripts/data
├── config/                     # Configuration files
│   ├── drain3.ini              # Drain3 config
│   └── rules.json              # Masking rules
└── data/                       # User data (gitignored)
```

### Data Flow Pipeline
```
Raw Log (.log)
  ↓ [alog-detect parse]
parsed.parquet (raw, masked, template_id, template, timestamp, host)
  ↓ [alog-detect build-deeplog]
sequences.parquet + vocab.json
  ↓ [alog-detect deeplog-train]
deeplog.pth
  ↓ [alog-detect deeplog-infer-enhanced]
deeplog_infer.parquet
  ↓ [alog-detect analyze-samples]
anomaly_analysis_report.md
  ↓ [alog-detect report]
comprehensive_report.md
```

### Architecture Patterns

**1. Builder Pattern**
- `builders/deeplog.py` and `builders/mscred.py` separate data preparation from model logic
- Enables reusable input generation for training/inference

**2. Analyzer Pattern**
- Pluggable analysis modules in `analyzers/`
- Each has standalone `main()` function + CLI integration
- Example: `tools/log_sample_analyzer.py` → `alog-detect analyze-samples`

**3. Hybrid System Pattern**
- **Training**: Python + PyTorch (flexibility)
- **Conversion**: Automated PyTorch → ONNX
- **Inference**: High-performance C + ONNX Runtime
- Auto-converter watches for new models

**4. Configuration Pattern**
- Type-safe dataclasses for all configurations
- Optional parameters with sensible defaults
- Full CLI flag coverage

**5. Pipeline Modularity**
- Each stage produces intermediate artifacts (Parquet/JSON)
- Stages can be run independently or chained
- Artifacts enable debugging and incremental workflows

---

## 🔑 Key Components

### CLI Module (`cli.py` - 1,081 lines)

**Entry Point**: `alog-detect` command (registered in `pyproject.toml`)

**Command Categories**:
```python
# Preprocessing
alog-detect parse                    # Drain3 log parsing

# Baseline Detection
alog-detect detect                   # Statistical detection
alog-detect validate-baseline        # Baseline quality validation

# DeepLog
alog-detect build-deeplog            # Create vocab.json + sequences.parquet
alog-detect deeplog-train            # Train LSTM model
alog-detect deeplog-infer            # Basic top-K inference
alog-detect deeplog-infer-enhanced   # Advanced inference (K-of-N, cooldown, sessions)

# MS-CRED
alog-detect build-mscred             # Create window counts
alog-detect mscred-train             # Train auto-encoder
alog-detect mscred-infer             # Inference with threshold methods

# Analysis
alog-detect analyze-samples          # Extract actual anomalous logs
alog-detect analyze-temporal         # Hourly/daily pattern analysis
alog-detect analyze-comparative      # Multi-file comparison
alog-detect analyze-mscred           # MS-CRED specific analysis

# Reporting
alog-detect report                   # Comprehensive integrated report

# Evaluation
alog-detect eval                     # Precision/Recall/F1 metrics

# Synthetic Data
alog-detect gen-synth                # General synthetic logs
alog-detect gen-training-data        # 100% normal for training
alog-detect gen-inference-normal     # Test false positives
alog-detect gen-inference-anomaly    # Test true positives

# Hybrid System
alog-detect convert-onnx             # PyTorch → ONNX conversion
alog-detect hybrid-pipeline          # Full hybrid workflow
```

### Log Preprocessor (`preprocess.py`)

**Class**: `LogPreprocessor` with `PreprocessConfig` dataclass

**Key Features**:
- **Drain3 Integration**: Template mining for log clustering
- **Comprehensive Masking**: Paths, hex, IPs (v4/v6), MAC, UUID, PID/TID, devices, numbers
- **Toggle Control**: Each mask type can be disabled via `--no-mask-*` flags
- **Format Support**: Syslog and dmesg
- **State Management**: Reusable Drain3 state for consistent templates

**Output**: `parsed.parquet` with columns:
- `raw`: Original log line
- `masked`: Masked version (for ML)
- `template_id`: Drain3 cluster ID
- `template`: Log template (constant parts)
- `timestamp`: Parsed timestamp
- `host`: Hostname (if present)

**Regex Patterns**: Pre-compiled for performance (see code for full list)

### DeepLog Builder (`builders/deeplog.py` - 41,171 lines)

**Core Functions**:
```python
build_deeplog_inputs(parsed_df, out_dir, window_size=10, stride=1)
# → Creates vocab.json (template_id → index) and sequences.parquet

train_deeplog(sequences_path, vocab_path, model_path, epochs=50, lr=0.001, ...)
# → Trains DeepLogLSTM model, saves to .pth file

infer_deeplog_enhanced(model_path, vocab_path, sequences_path, out_path, config)
# → Advanced inference with EnhancedInferenceConfig
```

**Model**: `DeepLogLSTM` (embedding → LSTM → FC → softmax)

**Enhanced Inference Features**:
- **Top-K vs Top-P**: Flexible prediction thresholds
- **K-of-N Judgment**: E.g., mark anomaly if 7 out of 10 predictions fail
- **Cooldown Mechanisms**:
  - Sequence cooldown (suppress repeated anomalies)
  - Novelty cooldown (for never-seen templates)
- **Session Tracking**: Entity-based session management
- **Novelty Detection**: Separate handling for unseen templates

**Config Example**:
```python
from anomaly_log_detector.builders.deeplog import EnhancedInferenceConfig

config = EnhancedInferenceConfig(
    top_k=10,                      # Top-K prediction
    k_of_n_k=7,                    # 7 failures...
    k_of_n_n=10,                   # ...out of 10 sequences
    sequence_cooldown=100,         # 100-sequence cooldown
    novelty_cooldown=50,           # 50-sequence for novelties
    enable_sessions=True,          # Track sessions by entity
    use_top_p=False                # Use top-K (not top-P)
)
```

### MS-CRED Model (`mscred_model.py`)

**Architecture**:
- **MultiScaleConvBlock**: Parallel convolutions (kernel sizes: 3, 5, 7)
- **AttentionModule**: Channel-wise attention
- **MSCREDEncoder/Decoder**: U-Net style auto-encoder
- **Loss**: MSE reconstruction loss

**Training**: `train_mscred(train_data, model_path, epochs=50, lr=0.001, ...)`

**Inference**: `infer_mscred(model, test_data, threshold_method='percentile_99')`

**Threshold Methods**:
- `percentile_99`: 99th percentile of reconstruction errors
- `percentile_95`: 95th percentile
- `three_sigma`: Mean + 3×σ
- `mad`: Median Absolute Deviation

### Baseline Detection (`detect.py`)

**Class**: `BaselineDetector` with `BaselineParams`

**Anomaly Scores**:
1. **Unseen Template Ratio**: `(new_templates / total_templates)` in sliding window
2. **Frequency Z-Score**: Exponential weighted moving average (EWMA) for spike detection

**Output**: `baseline_scores.parquet` with per-window scores

**Parameters**:
```python
@dataclass
class BaselineParams:
    window_size: int = 100          # Sliding window size
    stride: int = 50                # Window stride
    ewm_span: int = 20              # EWMA span for frequency
    unseen_threshold: float = 0.1   # Threshold for unseen ratio
    freq_threshold: float = 3.0     # Z-score threshold
```

### Analyzers (`analyzers/`)

**Log Sample Analyzer** (`log_samples.py` - 59,966 lines):
- Extracts actual problematic log lines from raw logs
- Provides before/after context (default: 3 lines each)
- Categorizes by anomaly type (baseline, deeplog, mscred, temporal)
- Up to 20 samples per type
- Human-readable explanations

**Temporal Analyzer** (`temporal.py`):
- Hourly/daily pattern analysis
- Detects time-based anomalies
- Visualization support

**Comparative Analyzer** (`comparative.py`):
- Multi-file comparison
- Identifies anomalies unique to target file

**MS-CRED Analyzer** (`mscred_analysis.py`):
- MS-CRED specific insights
- Reconstruction error patterns

**Baseline Validator** (`baseline_validation.py`):
- Quality scoring (0-100)
- Detects problematic baseline files
- Automatic filtering in batch mode

### Synthetic Data Generator (`synth.py`)

**Modes**:
```python
# 100% normal logs for training
alog-detect gen-training-data --num-lines 10000 --out-dir train/

# Normal logs for false positive testing
alog-detect gen-inference-normal --num-lines 1000 --out-dir test/

# Anomalous logs for true positive testing
alog-detect gen-inference-anomaly --num-lines 1000 --anomaly-rate 0.1 --out-dir test/
```

**Anomaly Types**:
- Unseen templates (never in training)
- Error messages (ERROR, FATAL, panic)
- Attack patterns (SQL injection, XSS attempts)
- System crashes (kernel panic, segfault)
- Traffic bursts (sudden frequency spikes)

**Output**: `.log` file + metadata JSON + labels CSV

---

## 🔄 Development Workflows

### Training Workflow

**Script**: `./scripts/train_models.sh`

**Steps**:
1. **Scan**: Recursively find `.log` files (configurable depth/file limits)
2. **Preprocess**: Parse with Drain3, generate `parsed.parquet`
3. **Baseline**: Build baseline statistics
4. **DeepLog**: Generate inputs → train LSTM model
5. **MS-CRED**: Generate windows → train auto-encoder
6. **Validate**: Check model quality, save metadata

**Usage**:
```bash
./scripts/train_models.sh /path/to/normal/logs /path/to/output
```

**Environment**:
- Auto-detects virtual environments (`.venv`, `venv`, `VIRTUAL_ENV`)
- Auto-installs dependencies if needed
- Progress indicators with emojis

### Inference Workflow

**Script**: `./scripts/run_inference.sh`

**Steps**:
1. **Preprocess**: Parse target log (reuse Drain3 state from training)
2. **Baseline Detection**: Statistical anomaly scores
3. **DeepLog Inference**: Enhanced inference with K-of-N judgment
4. **MS-CRED Inference**: Reconstruction-based detection
5. **Temporal Analysis**: Time-based patterns
6. **Sample Extraction**: Extract actual anomalous log lines
7. **Report Generation**: Comprehensive markdown report

**Usage**:
```bash
./scripts/run_inference.sh \
  /path/to/trained/models \
  /path/to/target.log \
  /path/to/output
```

**Output Structure**:
```
output/
├── parsed.parquet
├── baseline_scores.parquet
├── deeplog_infer.parquet
├── mscred_infer.parquet
├── temporal_analysis/
├── anomaly_analysis_report.md      # Detailed samples
└── comprehensive_report.md         # Integrated report
```

### Batch Analysis Workflow

**Script**: `./scripts/run_enhanced_batch_analysis.sh`

**Features**:
- Recursive directory scanning
- Baseline quality validation (auto-skip low-quality files)
- External target file support
- Up to 20 log samples per anomaly type
- Metadata tracking (`.meta.json`)

**Usage**:
```bash
./scripts/run_enhanced_batch_analysis.sh \
  /path/to/baselines \
  /path/to/output \
  --target /path/to/external/suspicious.log
```

### Model Comparison Workflow

**Script**: `./scripts/compare_models.sh`

**Purpose**: Compare two models trained at different times

**Usage**:
```bash
./scripts/compare_models.sh \
  /path/to/model1 \
  /path/to/model2 \
  /path/to/test/data \
  /path/to/output
```

**Output**: Side-by-side performance metrics, statistical analysis

### Incremental Learning Workflow

**Script**: `./scripts/train_models_incremental.sh`

**Purpose**: Update existing models with new data

**Usage**:
```bash
./scripts/train_models_incremental.sh \
  /path/to/existing/models \
  /path/to/new/logs \
  /path/to/output
```

**Behavior**: Loads existing model weights, continues training

---

## 📝 Code Conventions

### Naming Conventions

```python
# Package/Module
anomaly_log_detector                    # snake_case

# CLI Command
alog-detect                             # kebab-case

# Files
preprocess.py, mscred_model.py          # snake_case

# Classes
LogPreprocessor, DeepLogLSTM            # PascalCase
PreprocessConfig, EnhancedInferenceConfig

# Functions
build_deeplog_inputs()                  # snake_case
train_deeplog(), infer_mscred()

# Constants (regex patterns)
PATH_PATTERN, HEX_PATTERN               # UPPER_SNAKE_CASE

# Dataframe columns
'template_id', 'is_anomaly'             # snake_case (single quotes)
```

### Type Hints

**Always use type hints** - This is a strict convention:

```python
from __future__ import annotations  # For forward references

def build_deeplog_inputs(
    parsed_df: pd.DataFrame,
    out_dir: str | Path,
    window_size: int = 10,
    stride: int = 1
) -> tuple[Path, Path]:
    """
    Build DeepLog inputs from parsed logs.

    Args:
        parsed_df: Parsed log DataFrame
        out_dir: Output directory
        window_size: Sequence window size
        stride: Window stride

    Returns:
        Tuple of (vocab_path, sequences_path)
    """
    ...
```

**Modern syntax**:
- ✅ `str | Path` (preferred)
- ❌ `Union[str, Path]` (avoid)

### Dataclasses for Configuration

**Always use `@dataclass` for config objects**:

```python
from dataclasses import dataclass, field

@dataclass
class PreprocessConfig:
    """Configuration for log preprocessing."""

    mask_paths: bool = True
    mask_hex: bool = True
    mask_ips: bool = True
    # ... other fields

    # Complex defaults use field()
    custom_patterns: list[str] = field(default_factory=list)
```

### Error Handling

**Be informative and user-friendly**:

```python
try:
    df = pd.read_parquet(parsed_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Parsed file not found: {parsed_path}\n"
        f"Did you run 'alog-detect parse' first?"
    )
except Exception as e:
    raise ValueError(
        f"Failed to read parsed file: {e}\n"
        f"Ensure the file is a valid Parquet file."
    )
```

### Progress Indicators

**Always use `tqdm` for long operations**:

```python
from tqdm import tqdm

for epoch in tqdm(range(epochs), desc="Training DeepLog"):
    # Training logic
    ...

# With dataframe iteration
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing logs"):
    ...
```

### Documentation

**Public APIs**: English docstrings with Google style

```python
def train_deeplog(
    sequences_path: str | Path,
    vocab_path: str | Path,
    model_path: str | Path,
    epochs: int = 50,
    lr: float = 0.001
) -> None:
    """
    Train DeepLog LSTM model.

    Args:
        sequences_path: Path to sequences.parquet
        vocab_path: Path to vocab.json
        model_path: Output path for trained model
        epochs: Number of training epochs
        lr: Learning rate

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If vocab is empty or invalid
    """
```

**Internal code**: Korean comments are acceptable

```python
# 템플릿 ID를 인덱스로 변환
template_idx = vocab[template_id]

# 윈도우 슬라이딩
for i in range(0, len(sequences) - window_size + 1, stride):
    window = sequences[i:i + window_size]
```

### File Naming Conventions

**Output artifacts**:
- Parquet files: `{stage}.parquet` (e.g., `parsed.parquet`, `sequences.parquet`)
- Models: `{model_name}.pth` (e.g., `deeplog.pth`, `mscred.pth`)
- Config/Metadata: JSON format (e.g., `vocab.json`, `.meta.json`)
- State files: `{tool}_state.json` (e.g., `drain3_state.json`)
- Reports: Markdown (e.g., `report.md`, `anomaly_analysis_report.md`)

**Timestamped outputs**:
```
inference_20241002_143022/
batch_analysis_20241002_150330/
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── conftest.py              # Pytest configuration & fixtures
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_preprocess.py
│   ├── test_detect.py
│   ├── test_deeplog.py
│   └── test_mscred.py
└── integration/             # Integration tests (slower, end-to-end)
    ├── test_pipeline.py
    └── test_workflows.py
```

### Key Fixtures (from `conftest.py`)

```python
@pytest.fixture
def tmp_log_file(tmp_path) -> Path:
    """Temporary log file for testing."""

@pytest.fixture
def sample_parsed_data() -> pd.DataFrame:
    """Sample parsed DataFrame."""

@pytest.fixture
def sample_vocab() -> dict[int, int]:
    """Sample vocabulary mapping."""

@pytest.fixture
def config_dir() -> Path:
    """Config directory path."""

@pytest.fixture(autouse=True)
def clean_cache():
    """Auto-clean cache after each test."""
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_preprocess.py

# Specific test function
pytest tests/unit/test_preprocess.py::test_parse_syslog

# With coverage
pytest --cov=anomaly_log_detector --cov-report=html

# Skip slow tests
pytest -m "not slow"

# GPU tests (skip if no GPU)
pytest -m gpu  # Auto-skipped if GPU unavailable
```

### Writing Tests

**Unit test example**:
```python
def test_parse_syslog(tmp_log_file, tmp_path):
    """Test Syslog format parsing."""
    # Arrange
    log_content = "Jan 15 10:30:00 host kernel: [12345.678] message"
    tmp_log_file.write_text(log_content)

    config = PreprocessConfig(mask_paths=True)
    preprocessor = LogPreprocessor(config)

    # Act
    result = preprocessor.parse(tmp_log_file, tmp_path)

    # Assert
    assert result.exists()
    df = pd.read_parquet(result)
    assert len(df) == 1
    assert 'template_id' in df.columns
    assert df['template'].iloc[0] == 'kernel: [<NUM>] message'
```

**Integration test example**:
```python
def test_end_to_end_pipeline(tmp_path, sample_log_dir):
    """Test full training → inference pipeline."""
    # Arrange
    train_dir = tmp_path / "train"
    infer_dir = tmp_path / "infer"

    # Act - Training
    subprocess.run([
        "./scripts/train_models.sh",
        str(sample_log_dir),
        str(train_dir)
    ], check=True)

    # Act - Inference
    subprocess.run([
        "./scripts/run_inference.sh",
        str(train_dir),
        str(sample_log_dir / "suspicious.log"),
        str(infer_dir)
    ], check=True)

    # Assert
    assert (infer_dir / "comprehensive_report.md").exists()
    assert (infer_dir / "deeplog_infer.parquet").exists()
```

### Test Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: Slow running tests
    gpu: Tests requiring GPU
    integration: Integration tests
```

---

## 🛠️ Common Tasks

### Task 1: Add a New CLI Command

**Steps**:
1. Define command function in `cli.py`
2. Use `@cli.command()` decorator
3. Add comprehensive help text
4. Use dataclasses for complex options

**Example**:
```python
@cli.command(name="my-new-command")
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(), required=True)
@click.option("--threshold", type=float, default=0.5)
def my_new_command(input: str, output: str, threshold: float):
    """
    Short description of the command.

    Detailed explanation goes here.
    """
    try:
        # Implementation
        click.echo(f"Processing {input}...")
        # ...
        click.secho("✅ Success!", fg="green")
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
        raise click.Abort()
```

### Task 2: Add a New Masking Pattern

**Location**: `anomaly_log_detector/preprocess.py`

**Steps**:
1. Define regex pattern constant
2. Add to `PreprocessConfig` dataclass
3. Add CLI toggle flag
4. Implement masking in `_mask_line()` method

**Example**:
```python
# Step 1: Define pattern
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Step 2: Add config field
@dataclass
class PreprocessConfig:
    # ... existing fields
    mask_emails: bool = True

# Step 3: Add CLI flag (in cli.py)
@click.option("--no-mask-emails", is_flag=True, help="Don't mask email addresses")

# Step 4: Implement masking
def _mask_line(self, line: str) -> str:
    if self.config.mask_emails:
        line = EMAIL_PATTERN.sub('<EMAIL>', line)
    # ... other masking
    return line
```

### Task 3: Add a New Analyzer

**Steps**:
1. Create new file in `analyzers/` (e.g., `my_analyzer.py`)
2. Implement analysis logic with type hints
3. Add CLI command in `cli.py`
4. Update documentation

**Template**:
```python
# analyzers/my_analyzer.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

@dataclass
class MyAnalyzerConfig:
    """Configuration for my analyzer."""
    param1: int = 100
    param2: float = 0.5

def analyze_my_thing(
    input_path: str | Path,
    output_path: str | Path,
    config: MyAnalyzerConfig | None = None
) -> None:
    """
    Perform my custom analysis.

    Args:
        input_path: Input parquet file
        output_path: Output directory
        config: Analysis configuration
    """
    config = config or MyAnalyzerConfig()

    # Load data
    df = pd.read_parquet(input_path)

    # Analysis logic
    # ...

    # Save results
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # ...

# CLI integration (in cli.py)
@cli.command(name="analyze-my-thing")
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(), required=True)
@click.option("--param1", type=int, default=100)
@click.option("--param2", type=float, default=0.5)
def analyze_my_thing_cmd(input: str, output: str, param1: int, param2: float):
    """Run my custom analysis."""
    from anomaly_log_detector.analyzers.my_analyzer import analyze_my_thing, MyAnalyzerConfig

    config = MyAnalyzerConfig(param1=param1, param2=param2)
    analyze_my_thing(input, output, config)
    click.secho("✅ Analysis complete!", fg="green")
```

### Task 4: Modify Model Architecture

**DeepLog** (`builders/deeplog.py`):
```python
class DeepLogLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,      # Modify this
        hidden_dim: int = 256,     # Or this
        num_layers: int = 2,       # Or this
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
```

**MS-CRED** (`mscred_model.py`):
- Modify `MultiScaleConvBlock` kernel sizes
- Adjust encoder/decoder depths
- Change attention mechanism

**Important**: After modifying architecture:
1. Update model version in metadata
2. Retrain all models
3. Update documentation
4. Add migration guide if needed

### Task 5: Add New Threshold Method (MS-CRED)

**Location**: `anomaly_log_detector/mscred_model.py`

**Example**:
```python
def infer_mscred(
    model: MSCRED,
    test_loader: DataLoader,
    threshold_method: str = 'percentile_99',  # Add new method here
    device: str = 'cpu'
) -> pd.DataFrame:
    # ... existing code ...

    # Compute threshold
    if threshold_method == 'percentile_99':
        threshold = np.percentile(errors, 99)
    elif threshold_method == 'percentile_95':
        threshold = np.percentile(errors, 95)
    elif threshold_method == 'three_sigma':
        threshold = errors.mean() + 3 * errors.std()
    elif threshold_method == 'mad':
        median = np.median(errors)
        mad = np.median(np.abs(errors - median))
        threshold = median + 3 * mad
    elif threshold_method == 'my_new_method':
        # Custom threshold calculation
        threshold = my_custom_threshold_calc(errors)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
```

### Task 6: Debug Drain3 Template Issues

**Common issues**:
1. Too many templates (over-splitting)
2. Too few templates (under-splitting)
3. Inconsistent templates across runs

**Solutions**:

**Issue 1: Too many templates**
```ini
# config/drain3.ini
[SNAPSHOT]
snapshot_interval_minutes = 1

[MASKING]
masking = [
  # Add more aggressive masking
  {"regex_pattern": "\\d+", "mask_with": "<NUM>"},
  {"regex_pattern": "0x[0-9a-fA-F]+", "mask_with": "<HEX>"}
]

[DRAIN]
sim_th = 0.3          # Decrease similarity threshold (more merging)
depth = 3             # Decrease tree depth
max_children = 150    # Increase max children
```

**Issue 2: Too few templates**
```ini
[DRAIN]
sim_th = 0.5          # Increase similarity threshold (less merging)
depth = 5             # Increase tree depth
max_children = 50     # Decrease max children
```

**Issue 3: Inconsistent templates**
- Always reuse `--drain-state` file
- Process logs in same order
- Use consistent masking config

### Task 7: Handle Large Log Files

**Memory-efficient processing**:
```python
# Use chunked reading
chunk_size = 100_000
for chunk in pd.read_parquet(large_file, chunksize=chunk_size):
    # Process chunk
    process_chunk(chunk)

# Or use Dask for truly massive files
import dask.dataframe as dd
ddf = dd.read_parquet(large_file)
result = ddf.map_partitions(process_chunk).compute()
```

**Disk-based processing**:
```python
# Write intermediate results to disk
for i, chunk in enumerate(chunks):
    result = process_chunk(chunk)
    result.to_parquet(f"intermediate_{i}.parquet")

# Merge at end
parts = [pd.read_parquet(f"intermediate_{i}.parquet") for i in range(n_chunks)]
final = pd.concat(parts, ignore_index=True)
```

---

## ⚠️ Important Gotchas

### 1. Drain3 State Consistency

**Problem**: Different Drain3 states produce different template IDs

**Solution**: Always reuse the same `--drain-state` file for training and inference

```bash
# Training
alog-detect parse --input train/*.log --drain-state .cache/drain3.json

# Inference (MUST use same state)
alog-detect parse --input test.log --drain-state .cache/drain3.json
```

**Why**: Template IDs must match between training and inference. If you use a new Drain3 state, the model will see unknown template IDs.

### 2. Vocab Index Consistency

**Problem**: `vocab.json` maps template_id → model_index. If you rebuild vocab, indices change.

**Solution**: Never rebuild vocab during inference. Always use the vocab from training.

```bash
# ❌ WRONG: This rebuilds vocab
alog-detect build-deeplog --parsed new_parsed.parquet --out-dir models/

# ✅ CORRECT: Reuse existing vocab
alog-detect deeplog-infer --vocab models/vocab.json --sequences new_sequences.parquet
```

### 3. Parquet Schema Changes

**Problem**: Adding/removing columns breaks compatibility

**Solution**: Version your schemas and handle migrations

```python
# Check schema version
df = pd.read_parquet(path)
if 'schema_version' not in df.columns:
    # Migrate old schema
    df = migrate_schema_v1_to_v2(df)

# Save with version
df['schema_version'] = '2.0'
df.to_parquet(path)
```

### 4. GPU Memory Issues

**Problem**: Large models or batches cause OOM

**Solution**: Reduce batch size or use gradient accumulation

```python
# Option 1: Smaller batches
train_loader = DataLoader(dataset, batch_size=16)  # Was 64

# Option 2: Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Path Handling (Windows vs Linux)

**Problem**: Hardcoded `/` separators break on Windows

**Solution**: Always use `pathlib.Path`

```python
# ❌ BAD
output_path = base_dir + "/results/model.pth"

# ✅ GOOD
from pathlib import Path
output_path = Path(base_dir) / "results" / "model.pth"
```

### 6. Unicode in Logs

**Problem**: Some logs contain invalid UTF-8

**Solution**: Handle encoding errors gracefully

```python
# When reading files
with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Or use latin-1 for binary-safe reading
with open(log_file, 'r', encoding='latin-1') as f:
    content = f.read()
```

### 7. Timestamp Parsing Failures

**Problem**: Drain3 can't parse all timestamp formats

**Solution**: Add custom timestamp extraction before Drain3

```python
# In preprocess.py
def extract_timestamp(line: str) -> str | None:
    """Extract timestamp from log line."""
    # Try common formats
    patterns = [
        r'^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',  # Syslog
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO8601
        r'^\[\s*\d+\.\d+\]',                      # dmesg
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(0)
    return None
```

### 8. Model Serialization (PyTorch Versions)

**Problem**: Models saved in PyTorch 2.0 may not load in PyTorch 1.x

**Solution**: Use `weights_only=True` when loading for security

```python
# Secure loading
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)

# Or for full model
model = torch.load(model_path, map_location='cpu', weights_only=False)
```

### 9. ONNX Conversion Issues

**Problem**: Not all PyTorch ops are supported in ONNX

**Solution**: Test conversion early, use opset version 17+

```python
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=17,        # Use recent opset
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={           # Support variable batch sizes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### 10. Shell Script Portability

**Problem**: Scripts use bash-specific features

**Solution**: Always use `#!/usr/bin/env bash` shebang

```bash
#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Use [[ ]] instead of [ ]
if [[ -f "$file" ]]; then
    echo "File exists"
fi

# Use $() instead of backticks
current_date=$(date +%Y%m%d)
```

---

## 📚 Quick Reference

### File Locations

```
CLI Entry Point:        anomaly_log_detector/cli.py
Preprocessing:          anomaly_log_detector/preprocess.py
Baseline Detection:     anomaly_log_detector/detect.py
DeepLog:                anomaly_log_detector/builders/deeplog.py
MS-CRED:                anomaly_log_detector/mscred_model.py
Log Sample Analyzer:    anomaly_log_detector/analyzers/log_samples.py
Main README:            README.md
Project Structure:      docs/PROJECT_STRUCTURE.md
User Guides:            docs/guides/
Test Config:            pytest.ini
Package Config:         pyproject.toml
Drain3 Config:          config/drain3.ini
```

### Key Shell Scripts

```bash
# Training
./scripts/train_models.sh <log_dir> <output_dir>

# Inference
./scripts/run_inference.sh <model_dir> <target_log> <output_dir>

# Batch Analysis
./scripts/run_enhanced_batch_analysis.sh <baseline_dir> <output_dir> [--target <file>]

# Model Comparison
./scripts/compare_models.sh <model1> <model2> <test_data> <output>

# Incremental Training
./scripts/train_models_incremental.sh <existing_models> <new_logs> <output>

# Full Pipeline
./scripts/run_full_pipeline.sh <log_dir> <output_dir>
```

### CLI Quick Reference

```bash
# Parsing
alog-detect parse --input <file> --out-dir <dir> --drain-state <state.json>

# Baseline
alog-detect detect --parsed <parsed.parquet> --out-dir <dir>

# DeepLog
alog-detect build-deeplog --parsed <parsed.parquet> --out-dir <dir>
alog-detect deeplog-train --sequences <seq.parquet> --vocab <vocab.json> --model <out.pth>
alog-detect deeplog-infer-enhanced --model <model.pth> --vocab <vocab.json> --sequences <seq.parquet>

# MS-CRED
alog-detect build-mscred --parsed <parsed.parquet> --out-dir <dir>
alog-detect mscred-train --windows <windows.parquet> --model <out.pth>
alog-detect mscred-infer --model <model.pth> --windows <windows.parquet>

# Analysis
alog-detect analyze-samples --input <dir> --output <dir>
alog-detect report --input <dir> --output <report.md>
```

### Important Environment Variables

```bash
# Virtual environment auto-detection (in order of precedence)
VIRTUAL_ENV                 # Activated venv path
.venv/                      # Local .venv directory
venv/                       # Local venv directory

# Python path (for development)
PYTHONPATH=/path/to/study_preprocessor
```

### Debugging Tips

**Enable verbose logging**:
```bash
# Python logging
export PYTHONUNBUFFERED=1
python -u script.py

# Add to code
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Profile slow code**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... code to profile
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)  # Top 20 slowest functions
```

**Check GPU usage**:
```bash
# Install gpustat
pip install gpustat

# Monitor
watch -n 1 gpustat
```

**Memory profiling**:
```python
from memory_profiler import profile

@profile
def my_function():
    # ... code
    pass
```

### Common Error Messages

**Error**: `ValueError: Template ID not in vocab`
- **Cause**: Using different Drain3 state for training vs inference
- **Fix**: Reuse same `--drain-state` file

**Error**: `RuntimeError: CUDA out of memory`
- **Cause**: Model/batch too large for GPU
- **Fix**: Reduce batch size or use CPU with `--device cpu`

**Error**: `FileNotFoundError: parsed.parquet`
- **Cause**: Skipped preprocessing step
- **Fix**: Run `alog-detect parse` first

**Error**: `KeyError: 'template_id'`
- **Cause**: Parquet schema mismatch
- **Fix**: Re-run preprocessing or check Parquet file structure

**Error**: `ModuleNotFoundError: No module named 'anomaly_log_detector'`
- **Cause**: Package not installed
- **Fix**: `pip install -e .` in repo root

---

## 🤝 Contributing Guidelines

### Before Making Changes

1. **Read existing code** - Understand patterns and conventions
2. **Check documentation** - Look in `docs/guides/` for context
3. **Run tests** - Ensure current tests pass: `pytest`
4. **Create feature branch** - Use descriptive names: `feature/add-email-masking`

### Making Changes

1. **Follow conventions** - See "Code Conventions" section
2. **Add type hints** - Mandatory for all new code
3. **Write tests** - Unit tests for logic, integration for workflows
4. **Update documentation** - README, guides, and this CLAUDE.md

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes, commit frequently
git add .
git commit -m "feat: Add email masking support"

# Push to remote
git push origin feature/my-feature

# Create pull request on GitHub
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Example**:
```
feat: Add email masking to log preprocessor

- Add EMAIL_PATTERN regex
- Add mask_emails config field
- Add --no-mask-emails CLI flag
- Update tests for email masking

Closes #123
```

---

## 📞 Getting Help

**Documentation**:
- Main README: `README.md` (Korean)
- Guides: `docs/guides/`
- Project Structure: `docs/PROJECT_STRUCTURE.md`
- This file: `CLAUDE.md`

**Code Examples**:
- Example scripts: `examples/`
- Test cases: `tests/`
- Shell scripts: `scripts/`

**Issues**:
- Check existing issues on GitHub
- Search documentation for keywords
- Look at test cases for usage examples

---

## 🔄 Version History

**0.1.0** (2025-10-02):
- Initial release
- DeepLog, MS-CRED, Baseline methods
- Training/inference separation
- Model comparison and validation
- Log sample extraction
- Comprehensive reporting

---

## 📄 License

MIT License - See LICENSE file for details

---

**Note**: This document is maintained for AI assistants (like Claude) to understand the codebase. Keep it updated when making significant architectural changes.

Last updated: 2025-11-19
