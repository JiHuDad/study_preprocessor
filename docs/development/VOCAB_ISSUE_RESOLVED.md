# Vocabulary 형식 문제 해결

## 🐛 문제 상황

C inference engine을 실행할 때 vocabulary가 0 templates로 로드되어 모든 로그가 이상으로 탐지되는 문제가 발생했습니다.

```
Loaded vocabulary: 0 templates
Vocabulary loaded: 0 templates
```

## 🔍 원인 분석

Python에서 생성하는 `vocab.json`과 C inference engine이 기대하는 형식이 달랐습니다.

### 이전 형식 (잘못됨)

Python의 `build_deeplog_inputs()` 함수가 생성하는 형식:

```json
{
  "1": 0,   // template_id_string → template_index
  "2": 1,
  "3": 2,
  ...
}
```

이 형식은:
- 템플릿 ID를 인덱스로 매핑
- **실제 템플릿 문자열이 없음** ❌
- Python 모델 학습/추론에는 충분 (인덱스만 필요)
- C 엔진은 실제 문자열 필요 (로그 매칭용)

### 올바른 형식 (C 엔진 요구사항)

C inference engine의 `vocab_dict_load_from_json()` 함수가 기대하는 형식:

```json
{
  "1": "[<NUM>] usb 1-1: new high-speed USB device number <NUM> using ehci-pci",
  "2": "[<NUM>] CPU<ID>: Core temperature above threshold, cpu clock throttled",
  "3": "[<NUM>] CPU<ID>: Core temperature<PATH> normal",
  ...
}
```

이 형식은:
- 템플릿 ID를 **실제 템플릿 문자열**로 매핑 ✅
- C 엔진이 로그 라인과 템플릿을 매칭할 수 있음
- 유사도 계산으로 가장 근접한 템플릿 찾기 가능

## 🛠️ 해결 방법

### 1. 템플릿 문자열 추출 스크립트 생성

[hybrid_system/training/export_vocab_with_templates.py](../../hybrid_system/training/export_vocab_with_templates.py) 스크립트를 작성하여 `parsed.parquet` 또는 `preview.json`에서 실제 템플릿 문자열을 추출합니다.

```python
# parsed.parquet에서 template_id와 template 컬럼 읽기
df = pd.read_parquet(parsed_parquet)
template_map = {}
for _, row in df[['template_id', 'template']].drop_duplicates('template_id').iterrows():
    template_map[str(row['template_id'])] = str(row['template'])

# JSON으로 저장
with open(output_json, 'w') as f:
    json.dump(template_map, f, indent=2)
```

### 2. 올바른 vocab.json 생성

**Option A: parsed.parquet가 있는 경우**

```bash
python hybrid_system/training/export_vocab_with_templates.py \
    data/processed/parsed.parquet \
    hybrid_system/inference/models/vocab.json
```

**Option B: preview.json만 있는 경우**

```bash
python3 -c "
import json

# Read preview.json
with open('data/processed/preview.json', 'r') as f:
    preview = json.load(f)

# Extract template_id -> template mapping
template_map = {}
for item in preview:
    tid = str(item.get('template_id', ''))
    template = item.get('template', '')
    if tid and template:
        template_map[tid] = template

# Save to file
with open('hybrid_system/inference/models/vocab.json', 'w') as f:
    json.dump(template_map, f, indent=2)

print(f'Created vocab with {len(template_map)} templates')
"
```

### 3. ONNX 모델 변환 시 vocab.json 포함

모델 변환 시 올바른 형식의 vocab.json이 자동으로 복사되도록 수정:

```bash
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \  # 이전 형식
    --output-dir hybrid_system/inference/models

# 변환 후 올바른 vocab.json으로 교체
python hybrid_system/training/export_vocab_with_templates.py \
    data/processed/parsed.parquet \
    hybrid_system/inference/models/vocab.json
```

## ✅ 검증 방법

### 1. vocab.json 형식 확인

```bash
cat hybrid_system/inference/models/vocab.json
```

출력 예시:
```json
{
  "1": "[<NUM>] usb 1-1: new high-speed USB device number <NUM> using ehci-pci",
  "2": "[<NUM>] CPU<ID>: Core temperature above threshold, cpu clock throttled",
  ...
}
```

값이 숫자가 아닌 **문자열**이어야 합니다.

### 2. C inference engine 실행

```bash
cd hybrid_system/inference
make clean && make

./bin/inference_engine \
    -d models/deeplog.onnx \
    -v models/vocab.json \
    -t
```

정상 출력:
```
Loaded vocabulary: 7 templates
Vocabulary loaded: 7 templates
```

## 📝 향후 개선 사항

### 1. Python 빌드 함수 수정

`anomaly_log_detector/builders/deeplog.py`의 `build_deeplog_inputs()` 함수를 수정하여 두 가지 형식의 vocab 파일을 생성:

```python
def build_deeplog_inputs(parsed_parquet: str | Path, out_dir: str | Path, template_col: str = "template_id") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parsed_parquet)

    # Build vocab mapping (index용)
    unique_templates = [t for t in df[template_col].dropna().astype(str).unique()]
    vocab: Dict[str, int] = {t: i for i, t in enumerate(sorted(unique_templates))}
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2))

    # Build template string mapping (C 엔진용)
    template_strings = {}
    for _, row in df[[template_col, 'template']].drop_duplicates(template_col).iterrows():
        tid = str(row[template_col])
        template_str = str(row['template'])
        if not pd.isna(tid) and not pd.isna(template_str):
            template_strings[tid] = template_str

    (out / "vocab_templates.json").write_text(json.dumps(template_strings, indent=2))
```

### 2. 자동 변환 스크립트 통합

`hybrid_system/training/auto_converter.py`에 vocab.json 변환 로직 추가:

```python
def convert_vocab_for_c_engine(parsed_parquet: Path, output_vocab: Path):
    """Convert vocab.json to C engine format with template strings."""
    # ... 구현
```

### 3. README 업데이트

[hybrid_system/inference/README.md](../../hybrid_system/inference/README.md)에 vocab.json 형식 요구사항 명시

## 🔗 관련 파일

- **스크립트**: [hybrid_system/training/export_vocab_with_templates.py](../../hybrid_system/training/export_vocab_with_templates.py)
- **C 파서**: [hybrid_system/inference/src/log_parser.c](../../hybrid_system/inference/src/log_parser.c) (lines 203-310)
- **Python 빌더**: [anomaly_log_detector/builders/deeplog.py](../../anomaly_log_detector/builders/deeplog.py) (lines 15-28)
- **모델 변환기**: [hybrid_system/training/model_converter.py](../../hybrid_system/training/model_converter.py)

## 📅 해결 일자

2025-10-17

## 🏷️ 태그

`vocabulary` `onnx` `c-engine` `inference` `template-matching` `bug-fix`
