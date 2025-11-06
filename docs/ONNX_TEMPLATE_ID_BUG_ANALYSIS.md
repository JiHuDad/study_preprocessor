# ONNX Template ID 버그 분석 및 현재 상태

## 요약

ONNX 변환 시 template ID가 잘못 사용되는 버그에 대한 종합 분석입니다.

**현재 상태**: ✅ **버그는 이미 수정되었으며, 보호 로직이 추가되어 있습니다**

---

## 1. 버그의 본질

### 문제 상황
ONNX 변환 후 생성된 vocab.json이 다음과 같은 잘못된 형식으로 생성되는 문제:

```json
{
  "1": 0,
  "2": 1,
  "3": 2
}
```

### 원인
`build_deeplog_inputs()` 함수가 **`template_col="template_id"`**로 호출되면:
- Drain3가 부여한 ID (문자열 "1", "2", "3")가 vocab의 키로 사용됨
- 실제 템플릿 문자열 (예: "User <ID> logged in")은 vocab에 포함되지 않음
- C 추론 엔진은 실제 템플릿 문자열이 필요하므로 추론 불가

### 올바른 형식
Python vocab (학습용):
```json
{
  "User <ID> logged in": 0,
  "System started successfully": 1,
  "Error: <PATH> not found": 2
}
```

C 엔진용 vocab (ONNX 변환 후):
```json
{
  "0": "User <ID> logged in",
  "1": "System started successfully",
  "2": "Error: <PATH> not found"
}
```

---

## 2. 적용된 수정사항

### 2.1 DeepLog 빌더 수정 (✅ 완료)

**파일**: `anomaly_log_detector/builders/deeplog.py:15`

```python
def build_deeplog_inputs(
    parsed_parquet: str | Path,
    out_dir: str | Path,
    template_col: str = "template"  # ✅ 기본값이 "template"로 변경됨
) -> None:
    # Build vocab mapping using actual template strings (NOT template_id)
    # CRITICAL: Use "template" column (actual template string) for C engine compatibility
    # If template_col is "template_id", this will create {"1": 0, "2": 1} which is wrong!
    unique_templates = [t for t in df[template_col].dropna().astype(str).unique()]
    vocab: Dict[str, int] = {t: i for i, t in enumerate(sorted(unique_templates))}
```

**변경 내용**:
- 기본값: `template_col="template_id"` → `template_col="template"`
- 명확한 주석 추가로 향후 오용 방지

### 2.2 Model Converter 보호 로직 추가 (✅ 완료)

**파일**: `hybrid_system/training/model_converter.py:77-85`

```python
# Case 2: Python 학습용 형식인지 확인
# 올바른 형식: {"template_string": 0, ...}
# 잘못된 형식: {"1": 0, "2": 1, ...} (template_id를 key로 사용)
if isinstance(first_value, int):
    # template_id 오용 감지: 짧은 숫자 문자열은 template_id로 간주
    if first_key.isdigit() and len(first_key) <= 5:
        logger.error("❌ vocab이 template_id를 key로 사용하고 있습니다!")
        logger.error(f"   현재 형식: {{\"{first_key}\": {first_value}, ...}}")
        logger.error("   올바른 형식: {{\"actual template string\": 0, ...}}")
        logger.error("   해결: build_deeplog_inputs()에서 template_col='template' 사용")
        raise ValueError(
            "vocab.json이 template_id를 사용합니다. "
            "build_deeplog_inputs(template_col='template')로 재생성하세요."
        )
```

**보호 기능**:
1. vocab의 첫 번째 키가 숫자 문자열이고 길이가 5 이하인 경우 감지
2. 명확한 에러 메시지와 해결 방법 제시
3. ONNX 변환 중단으로 잘못된 모델 생성 방지

---

## 3. 현재 코드베이스 상태

### ✅ 올바르게 사용되는 곳

1. **CLI 명령어** (`anomaly_log_detector/cli.py:102`):
   ```python
   build_deeplog_inputs(str(parsed_parquet), str(out_dir))  # template_col 파라미터 없음 → 기본값 "template" 사용
   ```

2. **학습 스크립트** (`scripts/train.sh:354-357`):
   ```python
   build_deeplog_inputs(
       parsed_parquet='$WORK_DIR/parsed.parquet',
       out_dir='$WORK_DIR'
       # template_col 파라미터 없음 → 기본값 "template" 사용
   )
   ```

3. **기타 스크립트**: `train_models.sh`, `train_incremental.sh` 등 모두 기본값 사용

### ⚠️ 주의: MSCRED는 template_id 사용 (정상)

**파일**: `anomaly_log_detector/builders/mscred.py:10`

```python
def build_mscred_window_counts(
    parsed_parquet: str | Path,
    out_dir: str | Path,
    template_col: str = "template_id",  # ⚠️ template_id 사용 (의도적)
    window_size: int = 50,
    stride: int = 25,
) -> None:
```

**이것은 문제가 아닙니다**:
- MSCRED는 vocab.json을 생성하지 않음
- `pd.factorize()`로 단순히 정수 코드로 변환만 함
- C 엔진용 변환이 필요 없음

---

## 4. 버그가 발생했을 수 있는 경우

만약 다음 상황이라면 버그를 경험했을 가능성이 있습니다:

### 4.1 이전 버전 코드 사용
- 기본값이 `template_col="template_id"`였던 이전 버전 사용
- 해결: 최신 코드로 업데이트

### 4.2 이전에 생성된 vocab.json 사용
- 버그 수정 전에 생성된 vocab.json을 계속 사용 중
- 해결: vocab.json을 재생성 (방법은 아래 참조)

### 4.3 잘못된 파라미터로 직접 호출
```python
# ❌ 잘못된 사용 (명시적으로 template_id 전달)
from anomaly_log_detector.builders.deeplog import build_deeplog_inputs
build_deeplog_inputs(
    "data/parsed.parquet",
    "output/",
    template_col="template_id"  # ❌ 이렇게 호출하면 안 됨!
)
```

---

## 5. 문제 진단 방법

### 5.1 vocab.json 확인

```bash
# vocab.json 파일 내용 확인
head -10 training_workspace/vocab.json
```

**올바른 형식 (✅)**:
```json
{
  "Error: <PATH> not found": 0,
  "System started successfully": 1,
  "User <ID> logged in": 2
}
```

**잘못된 형식 (❌)**:
```json
{
  "1": 0,
  "2": 1,
  "3": 2
}
```

### 5.2 ONNX 변환 테스트

```bash
# ONNX 변환 시도 - 보호 로직이 작동하는지 확인
alog-detect convert-onnx \
  --deeplog-model models/deeplog.pth \
  --vocab models/vocab.json \
  --output-dir test_output
```

**잘못된 vocab이면 다음과 같은 에러가 발생**:
```
❌ vocab이 template_id를 key로 사용하고 있습니다!
   현재 형식: {"1": 0, ...}
   올바른 형식: {"actual template string": 0, ...}
   해결: build_deeplog_inputs()에서 template_col='template' 사용
ValueError: vocab.json이 template_id를 사용합니다. build_deeplog_inputs(template_col='template')로 재생성하세요.
```

---

## 6. 수정 방법

### 방법 1: 전체 재생성 (권장)

```bash
# 1. DeepLog 입력 재생성 (올바른 기본값 사용)
alog-detect build-deeplog \
  --parsed data/parsed.parquet \
  --out-dir training_workspace/

# 2. vocab.json 확인
head training_workspace/vocab.json
# ✅ 올바른 형식: {"actual template string": 0, ...}

# 3. 모델 재학습 (vocab이 바뀌었으므로 필수!)
alog-detect deeplog-train \
  --seq training_workspace/sequences.parquet \
  --vocab training_workspace/vocab.json \
  --out training_workspace/deeplog.pth

# 4. ONNX 변환
alog-detect convert-onnx \
  --deeplog-model training_workspace/deeplog.pth \
  --vocab training_workspace/vocab.json \
  --output-dir models/onnx \
  --validate
```

### 방법 2: 기존 모델 보존 (비권장)

이미 학습된 모델을 유지하고 싶은 경우:

```bash
# 경고: vocab 순서가 바뀔 수 있어 모델 정확도가 떨어질 수 있음!
# 가능하면 방법 1 사용 권장

# 1. parsed.parquet에서 template_id와 template 매핑 추출
python3 << 'EOF'
import pandas as pd
import json

df = pd.read_parquet('data/parsed.parquet')
old_vocab = json.load(open('training_workspace/vocab.json'))

# template_id -> template 매핑 생성
id_to_template = df[['template_id', 'template']].drop_duplicates().set_index('template_id')['template'].to_dict()

# 새 vocab 생성: {"1": 0} -> {"actual template": 0}
new_vocab = {id_to_template.get(tid, tid): idx for tid, idx in old_vocab.items()}

# 저장
with open('training_workspace/vocab_fixed.json', 'w') as f:
    json.dump(new_vocab, f, indent=2)

print(f"✅ 수정된 vocab 저장: {len(new_vocab)} templates")
EOF

# 2. 백업 및 교체
mv training_workspace/vocab.json training_workspace/vocab_old.json
mv training_workspace/vocab_fixed.json training_workspace/vocab.json

# 3. ONNX 변환
alog-detect convert-onnx \
  --deeplog-model training_workspace/deeplog.pth \
  --vocab training_workspace/vocab.json \
  --output-dir models/onnx
```

---

## 7. 예방 조치

### 7.1 코드 사용 시 주의사항

✅ **올바른 사용** (권장):
```python
from anomaly_log_detector.builders.deeplog import build_deeplog_inputs

# 기본값 사용 (template_col="template")
build_deeplog_inputs("data/parsed.parquet", "output/")
```

❌ **잘못된 사용** (하지 말 것):
```python
# template_id를 명시적으로 전달하지 말 것!
build_deeplog_inputs(
    "data/parsed.parquet",
    "output/",
    template_col="template_id"  # ❌ 절대 금지!
)
```

### 7.2 CI/CD 검증 추가 (선택사항)

```bash
# 학습 파이프라인에 vocab 검증 추가
#!/bin/bash
set -e

# vocab 생성
alog-detect build-deeplog --parsed data/parsed.parquet --out-dir workspace/

# vocab 형식 검증
python3 << 'EOF'
import json
import sys

vocab = json.load(open('workspace/vocab.json'))
first_key = next(iter(vocab.keys()))
first_value = next(iter(vocab.values()))

# template_id 오용 검사
if isinstance(first_value, int) and first_key.isdigit() and len(first_key) <= 5:
    print(f"❌ ERROR: vocab uses template_id as keys!")
    print(f"   First entry: {{\"{first_key}\": {first_value}}}")
    sys.exit(1)

print(f"✅ vocab format is correct")
print(f"   First entry: {{\"{first_key}\": {first_value}}}")
EOF

# 학습 계속...
```

---

## 8. 관련 문서

- **상세 수정 내역**: [docs/development/FIX_ONNX_DYNAMO_ERROR.md](development/FIX_ONNX_DYNAMO_ERROR.md)
- **Vocab 이슈 해결**: [docs/development/VOCAB_ISSUE_RESOLVED.md](development/VOCAB_ISSUE_RESOLVED.md)
- **세션 요약**: [docs/development/SESSION_SUMMARY.md](development/SESSION_SUMMARY.md)

---

## 9. FAQ

### Q1: 현재 코드에서도 버그가 발생하나요?
**A**: ❌ 아니요. 버그는 이미 수정되었고 보호 로직도 추가되었습니다.

### Q2: 이전에 생성한 vocab.json은 어떻게 하나요?
**A**: vocab.json을 재생성하거나, 위의 "방법 2"로 수정할 수 있습니다. 단, **모델 재학습 권장**.

### Q3: ONNX 변환 시 에러가 발생합니다
**A**: vocab.json 형식을 확인하세요. 잘못된 형식이면 자동으로 감지하고 에러 메시지를 출력합니다.

### Q4: MSCRED도 같은 문제가 있나요?
**A**: ❌ 아니요. MSCRED는 vocab.json을 사용하지 않으므로 문제없습니다.

### Q5: 버그가 언제 수정되었나요?
**A**: 커밋 이력을 확인하려면:
```bash
git log --all --grep="template_col" --oneline
git log --all --grep="vocab" -- anomaly_log_detector/builders/deeplog.py
```

---

## 10. 결론

✅ **버그 상태**: 완전히 수정됨
✅ **보호 장치**: 추가됨 (잘못된 vocab 자동 감지)
✅ **향후 방지**: 기본값 변경 + 명확한 주석

**만약 현재 ONNX 변환 시 template ID 관련 문제가 발생한다면**:
1. vocab.json을 확인하여 잘못된 형식인지 체크
2. 최신 코드로 업데이트
3. vocab.json과 모델을 재생성
4. ONNX 변환 재시도

**추가 도움이 필요하면**:
- 에러 메시지 전체를 공유해 주세요
- vocab.json의 처음 5줄을 공유해 주세요
- 사용한 명령어를 공유해 주세요
