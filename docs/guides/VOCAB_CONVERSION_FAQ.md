# Vocabulary 변환 FAQ

## ❓ 자주 묻는 질문

### Q1: "템플릿 문자열을 추출할 수 없습니다" 경고가 나왔어요. export_vocab_with_templates.py를 따로 수행해야 하나요?

**답변**: **대부분의 경우 필요 없습니다!** 이 메시지는 자동 변환이 실패했을 때만 나타나는 안내입니다.

#### ✅ 자동 변환이 성공하는 경우 (99%)

```bash
$ python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models

INFO:__main__:🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...
INFO:__main__:📂 parsed.parquet에서 템플릿 추출: data/processed/parsed.parquet
INFO:__main__:✅ 7개 템플릿 추출 완료
INFO:__main__:✅ C 엔진용 vocab 형식 (template strings): 7 templates  ← 성공!
```

**조건**: `data/processed/` 디렉토리에 다음 중 하나라도 있으면 자동 변환 성공
- ✅ `parsed.parquet` (우선순위 1)
- ✅ `preview.json` (우선순위 2)

#### ⚠️  수동 변환이 필요한 경우 (1%)

```bash
WARNING:__main__:⚠️  템플릿 문자열을 추출할 수 없습니다.
WARNING:__main__:⚠️  /some/other/path에 parsed.parquet 또는 preview.json이 필요합니다.
WARNING:__main__:⚠️  C 엔진 사용을 위해 다음 스크립트를 실행하세요:
    python hybrid_system/training/export_vocab_with_templates.py \
        /some/other/path/parsed.parquet \
        hybrid_system/inference/models/vocab.json
```

**상황**: 다음과 같은 특수한 경우에만 발생
- ❌ vocab.json이 parsed.parquet와 **다른 디렉토리**에 있음
- ❌ vocab.json 디렉토리에 `parsed.parquet`도 `preview.json`도 없음

### Q2: 자동 변환이 성공했는지 어떻게 확인하나요?

#### 방법 1: 로그 메시지 확인

성공 시:
```
✅ 7개 템플릿 추출 완료
✅ C 엔진용 vocab 형식 (template strings): 7 templates
```

실패 시:
```
⚠️  템플릿 문자열을 추출할 수 없습니다.
⚠️  vocab이 인덱스 형식입니다.
```

#### 방법 2: vocab.json 파일 확인

```bash
cat hybrid_system/inference/models/vocab.json | head -5
```

**올바른 형식** (템플릿 문자열):
```json
{
  "1": "[<NUM>] usb 1-1: new high-speed USB device number <NUM> using ehci-pci",  ✅
  "2": "[<NUM>] CPU<ID>: Core temperature above threshold, cpu clock throttled",  ✅
  ...
}
```

**잘못된 형식** (인덱스):
```json
{
  "1": 0,  ❌
  "2": 1,  ❌
  ...
}
```

#### 방법 3: C inference engine으로 확인

```bash
cd hybrid_system/inference
./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -t
```

**성공**:
```
Loaded vocabulary: 7 templates  ✅
```

**실패**:
```
Loaded vocabulary: 0 templates  ❌
```

### Q3: 언제 수동으로 export_vocab_with_templates.py를 실행하나요?

#### 케이스 1: vocab.json과 parsed.parquet가 다른 위치

```bash
# 상황: vocab.json이 /tmp/vocab.json이고 parsed.parquet가 data/processed/에 있음
python hybrid_system/training/model_converter.py \
    --vocab /tmp/vocab.json \  # parsed.parquet가 같은 디렉토리에 없음!
    ...

# 해결: 수동 변환
python hybrid_system/training/export_vocab_with_templates.py \
    data/processed/parsed.parquet \
    hybrid_system/inference/models/vocab.json
```

#### 케이스 2: 특정 parsed.parquet를 사용하고 싶을 때

```bash
# 다른 데이터셋의 템플릿을 사용하고 싶을 때
python hybrid_system/training/export_vocab_with_templates.py \
    /path/to/other/dataset/parsed.parquet \
    hybrid_system/inference/models/vocab.json
```

#### 케이스 3: vocab.json만 별도로 업데이트

```bash
# ONNX 모델은 그대로 두고 vocab만 업데이트
python hybrid_system/training/export_vocab_with_templates.py \
    data/processed/parsed.parquet \
    hybrid_system/inference/models/vocab.json
```

### Q4: 표준 워크플로우는 무엇인가요?

#### 추천 워크플로우 (자동 변환)

```bash
# 1. 로그 파싱 (parsed.parquet + preview.json 생성)
alog-detect parse --input data/raw/system.log --out-dir data/processed

# 2. 모델 학습
alog-detect train-deeplog \
    --parsed data/processed/parsed.parquet \
    --out-dir data/processed

# 3. ONNX 변환 (자동으로 올바른 vocab.json 생성!)
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models

# 4. C 엔진 실행
cd hybrid_system/inference
make && ./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -t
```

**핵심**: `--vocab data/processed/vocab.json`를 지정하면, 같은 디렉토리의 `parsed.parquet`에서 자동으로 템플릿 추출!

### Q5: 디렉토리 구조가 중요한가요?

#### ✅ 권장 구조 (자동 변환 성공)

```
data/processed/
├── parsed.parquet       ← vocab.json과 같은 디렉토리!
├── preview.json         ← 또는 이것이라도 있으면 OK
├── vocab.json           ← 여기를 --vocab으로 지정
└── sequences.parquet
```

```bash
# 이렇게 실행하면 자동 변환 성공!
python model_converter.py --vocab data/processed/vocab.json ...
```

#### ❌ 문제가 되는 구조

```
data/processed/
├── parsed.parquet
└── vocab.json

/tmp/
└── vocab.json           ← vocab만 별도 위치
```

```bash
# 이렇게 실행하면 자동 변환 실패 (parsed.parquet를 못 찾음)
python model_converter.py --vocab /tmp/vocab.json ...

# 해결: 수동 변환
python export_vocab_with_templates.py \
    data/processed/parsed.parquet \
    hybrid_system/inference/models/vocab.json
```

### Q6: 자동 변환 로직은 어떻게 작동하나요?

```python
# model_converter.py의 _convert_vocab_for_c_engine()

# 1. vocab.json의 형식 확인
if 값이_이미_템플릿_문자열:
    return vocab  # 변환 불필요

# 2. vocab.json과 같은 디렉토리에서 파일 찾기
vocab_dir = Path(vocab_path).parent  # 예: data/processed/

# 3. parsed.parquet에서 추출 시도
if (vocab_dir / "parsed.parquet").exists():
    template_map = parquet에서_추출()
    return template_map  ✅

# 4. preview.json에서 추출 시도
if (vocab_dir / "preview.json").exists():
    template_map = json에서_추출()
    return template_map  ✅

# 5. 둘 다 없으면 경고
logger.warning("⚠️  템플릿 문자열을 추출할 수 없습니다.")
logger.warning("⚠️  {vocab_dir}에 parsed.parquet 또는 preview.json이 필요합니다.")
return vocab  # 원본 vocab 반환 (인덱스 형식)
```

## 📊 요약

| 상황 | 자동 변환 | 수동 변환 필요 |
|------|----------|-------------|
| vocab.json과 parsed.parquet가 같은 디렉토리 | ✅ 성공 | ❌ 불필요 |
| vocab.json과 preview.json이 같은 디렉토리 | ✅ 성공 | ❌ 불필요 |
| vocab.json만 다른 위치에 복사 | ❌ 실패 | ✅ 필요 |
| 다른 데이터셋의 템플릿 사용 | ❌ (의도적) | ✅ 필요 |

## ✅ 결론

### 대부분의 경우 (표준 워크플로우)

```bash
# 이것만 실행하면 끝!
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models

# export_vocab_with_templates.py는 실행 안 해도 됨! ✅
```

### 특수한 경우에만

```bash
# vocab.json이 parsed.parquet와 다른 위치에 있거나
# 다른 데이터셋의 템플릿을 사용하고 싶을 때만
python hybrid_system/training/export_vocab_with_templates.py \
    <parsed.parquet> \
    <output_vocab.json>
```

## 🔗 관련 문서

- [ONNX Conversion Guide](./ONNX_CONVERSION_GUIDE.md) - 전체 변환 가이드
- [Vocab Issue Resolved](../development/VOCAB_ISSUE_RESOLVED.md) - 문제 분석
- [Auto Vocab Conversion](../development/AUTO_VOCAB_CONVERSION.md) - 구현 상세
