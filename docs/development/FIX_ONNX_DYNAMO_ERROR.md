# ONNX 변환 관련 에러 수정

이 문서는 ONNX 변환 시 발생하는 두 가지 주요 문제와 해결책을 다룹니다:
1. **Vocab 형식 에러**: template_id 대신 template string 사용
2. **Dynamo Export 에러**: PyTorch 2.1+ 호환성 문제

---

## 문제 1: Vocab 형식 에러 (template_id vs template string)

### 증상

ONNX 변환 후 생성된 vocab.json이 다음과 같은 형식:

```json
{
  "1": 0,
  "2": 1,
  "3": 2
}
```

### 원인

`build_deeplog_inputs()`가 **`template_col="template_id"`**를 기본값으로 사용:
- `template_id`: Drain3가 부여한 ID (문자열 "1", "2", "3" 등)
- `template`: 실제 템플릿 문자열 (예: "User <ID> logged in")

C 추론 엔진은 **실제 템플릿 문자열**이 필요한데, template_id만 저장되어 있어 추론 불가.

### 해결

#### 1. [deeplog.py:15](../../anomaly_log_detector/builders/deeplog.py#L15) 수정

기본값을 `template_col="template"`로 변경:

```python
def build_deeplog_inputs(
    parsed_parquet: str | Path,
    out_dir: str | Path,
    template_col: str = "template"  # 변경: "template_id" → "template"
) -> None:
    # Build vocab mapping using actual template strings (NOT template_id)
    # CRITICAL: Use "template" column (actual template string) for C engine compatibility
    unique_templates = [t for t in df[template_col].dropna().astype(str).unique()]
    vocab: Dict[str, int] = {t: i for i, t in enumerate(sorted(unique_templates))}
```

#### 2. [model_converter.py:72-80](../../hybrid_system/training/model_converter.py#L72-L80) 검증 로직 추가

잘못된 vocab 형식을 조기에 발견:

```python
if isinstance(first_value, int):
    # template_id 사용 여부 확인
    if first_key.isdigit() and len(first_key) <= 5:
        logger.error("❌ vocab이 template_id를 key로 사용하고 있습니다!")
        raise ValueError(
            "vocab.json이 template_id를 사용합니다. "
            "build_deeplog_inputs(template_col='template')로 재생성하세요."
        )
```

### 올바른 vocab 형식

**Python 학습용** (입력):
```json
{
  "User <ID> logged in": 0,
  "System started successfully": 1,
  "Error: <PATH> not found": 2
}
```

**C 엔진용** (ONNX 변환 후):
```json
{
  "0": "User <ID> logged in",
  "1": "System started successfully",
  "2": "Error: <PATH> not found"
}
```

### 재생성 방법

#### 방법 1: 전체 재생성 (권장)

```bash
# 1. DeepLog 입력 재생성 (template 컬럼 사용)
alog-detect build-deeplog --parsed data/parsed.parquet --out-dir training_workspace/

# 2. vocab.json 확인
head training_workspace/vocab.json
# 올바른 형식: {"actual template string": 0, ...}

# 3. 모델 재학습 (vocab이 바뀌었으므로 필수!)
alog-detect deeplog-train \
  --seq training_workspace/sequences.parquet \
  --vocab training_workspace/vocab.json \
  --out training_workspace/deeplog.pth

# 4. ONNX 변환
alog-detect convert-onnx \
  --deeplog-model training_workspace/deeplog.pth \
  --vocab training_workspace/vocab.json \
  --output-dir models/onnx
```

#### 방법 2: 기존 vocab 변환 (빠른 방법)

기존 학습된 모델을 유지하고 vocab.json만 변환:

```bash
# 1. vocab.json 변환 (template_id -> template string)
python scripts/fix_vocab_format.py \
  --parsed data/parsed.parquet \
  --old-vocab training_workspace/vocab.json \
  --output training_workspace/vocab_fixed.json

# 2. 백업 및 교체
mv training_workspace/vocab.json training_workspace/vocab_old.json
mv training_workspace/vocab_fixed.json training_workspace/vocab.json

# 3. vocab.json 확인
head training_workspace/vocab.json

# 4. ONNX 변환
alog-detect convert-onnx \
  --deeplog-model training_workspace/deeplog.pth \
  --vocab training_workspace/vocab.json \
  --output-dir models/onnx
```

**⚠️ 주의**: 방법 2는 vocab 순서가 바뀔 수 있으므로 **모델 재학습 권장** (방법 1)

---

## 문제 2: ONNX Export Dynamo Error

### 증상

PyTorch 2.1+ 환경에서 ONNX 변환 시 다음과 같은 에러 발생:

```
torch.export.export 호출 관련 에러
mark_dynamic 사용 관련 경고: maybe_mark_dynamic을 사용하라는 메시지
Dim.DYNAMIC 사용 관련 경고: Dim.STATIC 또는 Dim.AUTO로 교체하라는 메시지
Constraints violated (L['x'].size()[1])
```

## 원인 분석

### PyTorch 2.1+ ONNX Export 변경사항

PyTorch 2.1부터 `torch.onnx.export()`가 기본적으로 새로운 **dynamo 기반 방식**을 시도합니다:

1. **기존 방식 (TorchScript)**:
   - `torch.jit.trace()` 기반
   - 안정적이지만 일부 동적 연산 제한
   - `dynamic_axes` 파라미터로 동적 차원 지정

2. **새로운 방식 (Dynamo)**:
   - `torch.export.export()` 기반
   - `torch.export.Dim.DYNAMIC` 등의 새로운 API
   - 일부 모델 구조(LSTM, Embedding)와 호환성 문제

### 문제 발생 시나리오

```python
# 코드에서는 레거시 방식을 의도했지만...
torch.onnx.export(
    model, dummy_input, onnx_path,
    dynamic_axes={
        'input_sequence': {0: 'batch_size', 1: 'sequence_length'}
    }
)

# PyTorch 2.1+에서는 자동으로 dynamo 방식을 시도 → 에러 발생
```

## 해결 방법

### 수정된 코드

[hybrid_system/training/model_converter.py](../../hybrid_system/training/model_converter.py)에서 **명시적으로 레거시 방식 지정**:

```python
# DeepLog 모델 변환 (Line 147~185)
logger.info("🔄 ONNX export 시작 (TorchScript 방식)...")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # PyTorch 2.0+ 호환성: dynamo 방식 명시적 비활성화
    export_options = {
        'export_params': True,
        'opset_version': 11,
        'do_constant_folding': True,
        'input_names': ['input_sequence'],
        'output_names': ['predictions'],
        'dynamic_axes': {
            'input_sequence': {0: 'batch_size', 1: 'sequence_length'},
            'predictions': {0: 'batch_size'}
        },
        'verbose': False
    }

    # PyTorch 2.1+에서 dynamo 방식 강제 비활성화
    try:
        # dynamo=False를 시도 (PyTorch 2.1+)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            dynamo=False,  # 🔑 핵심: 명시적으로 레거시 TorchScript 방식 사용
            **export_options
        )
    except TypeError:
        # PyTorch 2.0 이하는 dynamo 파라미터 없음
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            **export_options
        )

logger.info("✅ ONNX export 성공")
```

### 핵심 변경사항

1. **`dynamo=False` 명시적 지정**: PyTorch 2.1+에서 레거시 TorchScript 방식 강제
2. **Try-Except 호환성 처리**: PyTorch 2.0 이하에서는 `dynamo` 파라미터가 없으므로 TypeError 예외 처리
3. **FutureWarning 필터링**: dynamo 관련 경고 메시지 억제

### 동일한 수정 적용 위치

- **DeepLog 모델**: [model_converter.py:147-185](../../hybrid_system/training/model_converter.py#L147-L185)
- **MSCRED 모델**: [model_converter.py:355-392](../../hybrid_system/training/model_converter.py#L355-L392)

## 사용 방법

### 다른 시스템에서 에러 발생 시

1. **최신 코드 pull**:
   ```bash
   git pull origin main
   ```

2. **ONNX 모델 재변환**:
   ```bash
   alog-detect convert-onnx \
     --deeplog-model models/deeplog.pth \
     --vocab models/vocab.json \
     --output-dir models/onnx \
     --portable
   ```

3. **변환 성공 확인**:
   ```
   🔄 ONNX export 시작 (TorchScript 방식)...
   ✅ ONNX export 성공
   🎉 ONNX 변환 완료!
   ```

## 추가 수정사항

### 1. `--seq-len` 옵션 추가

ONNX 변환 시 시퀀스 길이를 명시적으로 지정할 수 있는 옵션 추가:

```bash
alog-detect convert-onnx \
  --deeplog-model models/deeplog.pth \
  --vocab models/vocab.json \
  --seq-len 50 \  # 새로운 옵션
  --output-dir models/onnx
```

- **기본 동작**: `--seq-len`을 지정하지 않으면 모델 checkpoint에 저장된 값 사용
- **ONNX 유연성**: `dynamic_axes` 설정으로 다양한 시퀀스 길이 지원
- **권장사항**: 학습 시 사용한 길이와 동일한 길이 사용 권장 (성능 및 정확도)

### 2. 메타데이터 개선

ONNX 메타데이터에 `dynamic_axes` 정보 추가:

```json
{
  "model_type": "deeplog",
  "vocab_size": 7,
  "seq_len": 50,
  "dynamic_axes": {
    "input_sequence": {
      "0": "batch_size",
      "1": "sequence_length"
    },
    "predictions": {
      "0": "batch_size"
    }
  },
  "notes": "ONNX model supports dynamic sequence lengths via dynamic_axes. seq_len is recommended value from training."
}
```

## 참고 문서

- **PyTorch ONNX Export 가이드**: https://pytorch.org/docs/stable/onnx.html
- **PyTorch Dynamo ONNX Export**: https://pytorch.org/docs/stable/onnx_dynamo.html
- **프로젝트 Troubleshooting**: [CLAUDE.md](CLAUDE.md#onnx-export-errors-dynamo--mark_dynamic--dimdynamic)

## 테스트 확인

수정 후 다음 테스트를 수행하여 정상 동작 확인:

```bash
# 1. ONNX 변환 테스트
alog-detect convert-onnx \
  --deeplog-model models/deeplog.pth \
  --vocab models/vocab.json \
  --output-dir /tmp/test_onnx \
  --validate

# 2. 변환된 모델 검증
ls -lh /tmp/test_onnx/
# 출력 예시:
# deeplog.onnx
# deeplog.onnx.meta.json
# vocab.json

# 3. C 추론 엔진 테스트
cd hybrid_system/inference
make clean && make
./bin/inference_engine /tmp/test_onnx
```

## 버전 호환성

| PyTorch 버전 | `dynamo` 파라미터 | 동작 방식 |
|-------------|------------------|----------|
| < 2.0       | 미지원 (TypeError) | 레거시 TorchScript만 사용 |
| 2.0         | 미지원 (TypeError) | 레거시 TorchScript만 사용 |
| 2.1+        | 지원 (`dynamo=False`) | `dynamo=False`로 레거시 강제 |
| 2.4+        | 지원 (`dynamo=True` 기본값) | `dynamo=False`로 레거시 강제 |

## 결론

✅ **해결 완료**:
- PyTorch 2.1+ 환경에서 `dynamo=False` 명시로 ONNX export 안정화
- PyTorch 2.0 이하 호환성 유지
- DeepLog 및 MSCRED 모델 모두 적용

✅ **추가 개선**:
- `--seq-len` 옵션으로 시퀀스 길이 제어 가능
- 메타데이터에 동적 축 정보 명시
- 문서화 완료 ([CLAUDE.md](CLAUDE.md))

🚀 **다음 단계**:
- 다른 시스템에서 최신 코드 pull
- ONNX 모델 재변환
- C 추론 엔진 테스트
