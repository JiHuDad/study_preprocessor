# 자동 Vocab 변환 기능 구현

## 📋 요약

ONNX 모델 변환 시 vocab.json을 C inference engine용 형식으로 **자동 변환**하는 기능을 추가했습니다.

**이전**: ONNX 변환 후 수동으로 vocab.json 변환 필요
**이후**: ONNX 변환 시 자동으로 올바른 형식의 vocab.json 생성

## 🎯 문제 정의

### 배경

Python 학습 환경과 C inference engine이 서로 다른 vocab.json 형식을 사용:

| 환경 | 형식 | 예시 |
|------|------|------|
| Python 학습 | `{"template_id": index}` | `{"1": 0, "2": 1}` |
| C inference engine | `{"template_id": "template_string"}` | `{"1": "User logged in", "2": "Connection failed"}` |

### 기존 워크플로우 (불편함)

```bash
# 1. ONNX 변환
python model_converter.py --deeplog-model model.pth --vocab vocab.json --output-dir models/
# → models/vocab.json 생성 (인덱스 형식)

# 2. 수동 vocab 변환 필요!
python export_vocab_with_templates.py parsed.parquet models/vocab.json
# → models/vocab.json을 템플릿 문자열 형식으로 교체

# 3. C 엔진 실행
./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json
```

**문제점**:
- 2단계를 잊어버리기 쉬움
- 에러 메시지: "Loaded vocabulary: 0 templates"
- 모든 로그가 이상으로 탐지됨

## ✅ 해결 방법

### 구현 내용

`hybrid_system/training/model_converter.py`에 `_convert_vocab_for_c_engine()` 메서드 추가:

```python
def _convert_vocab_for_c_engine(self, vocab: Dict, vocab_path: str) -> Dict[str, str]:
    """
    vocab.json을 C 엔진용 형식으로 자동 변환

    1. 이미 템플릿 문자열 형식인지 확인
    2. 인덱스 형식이면 parsed.parquet에서 템플릿 추출
    3. parsed.parquet이 없으면 preview.json에서 추출
    4. 변환 불가능하면 경고 메시지 출력
    """
```

### 변환 로직

```
1. vocab 형식 확인
   ├─ 값이 문자열(길이 > 10)? → 이미 올바른 형식
   └─ 값이 숫자? → 변환 필요

2. parsed.parquet에서 추출 시도
   ├─ vocab_dir/parsed.parquet 존재?
   ├─ template_id, template 컬럼 존재?
   └─ 매핑 추출 성공 → 반환

3. preview.json에서 추출 시도
   ├─ vocab_dir/preview.json 존재?
   ├─ template_id, template 필드 존재?
   └─ 매핑 추출 성공 → 반환

4. 변환 실패
   └─ 경고 메시지 + 수동 변환 명령어 안내
```

### 새로운 워크플로우 (간편함)

```bash
# 1. ONNX 변환 (자동 vocab 변환!)
python model_converter.py --deeplog-model model.pth --vocab vocab.json --output-dir models/
# → models/vocab.json 생성 (자동으로 템플릿 문자열 형식!)

# 2. C 엔진 실행
./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json
# → "Loaded vocabulary: 7 templates" ✅
```

## 📊 실행 예시

### 성공 케이스

```bash
$ python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models

INFO:__main__:🔄 DeepLog 모델 변환 시작: .cache/deeplog.pth
INFO:__main__:🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...
INFO:__main__:📂 parsed.parquet에서 템플릿 추출: data/processed/parsed.parquet
INFO:__main__:✅ 7개 템플릿 추출 완료
INFO:__main__:✅ DeepLog 변환 완료: hybrid_system/inference/models/deeplog.onnx
INFO:__main__:📚 어휘 사전: hybrid_system/inference/models/vocab.json
INFO:__main__:✅ C 엔진용 vocab 형식 (template strings): 7 templates

🎉 모델 변환 완료!
✅ deeplog: hybrid_system/inference/models/deeplog.onnx
```

### 생성된 vocab.json

```json
{
  "1": "[<NUM>] usb 1-1: new high-speed USB device number <NUM> using ehci-pci",
  "2": "[<NUM>] CPU<ID>: Core temperature above threshold, cpu clock throttled",
  "3": "[<NUM>] CPU<ID>: Core temperature<PATH> normal",
  "4": "[<NUM>] eth<ID>: Link is Up - 1000Mbps<PATH> - flow control rx<PATH>",
  "5": "[<NUM>] EXT<ID>-fs (sda<ID>): mounted filesystem with ordered data mode. Opts: (null)",
  "6": "[<NUM>] audit: type=<NUM> audit(<NUM>:<NUM>): apparmor=\"DENIED\" ...",
  "7": "[<NUM>] usb 1-1: USB disconnect, device number <NUM>"
}
```

### 경고 케이스 (parsed.parquet 없음)

```bash
$ python hybrid_system/training/model_converter.py \
    --deeplog-model model.pth \
    --vocab /tmp/vocab.json \
    --output-dir models/

INFO:__main__:🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...
WARNING:__main__:⚠️  템플릿 문자열을 추출할 수 없습니다.
WARNING:__main__:⚠️  /tmp에 parsed.parquet 또는 preview.json이 필요합니다.
WARNING:__main__:⚠️  C 엔진 사용을 위해 다음 스크립트를 실행하세요:
    python hybrid_system/training/export_vocab_with_templates.py \
        /tmp/parsed.parquet \
        models/vocab.json
```

## 🔧 코드 변경 사항

### 1. ModelConverter 클래스에 메서드 추가

**파일**: `hybrid_system/training/model_converter.py`

```python
class ModelConverter:
    def _convert_vocab_for_c_engine(self, vocab: Dict, vocab_path: str) -> Dict[str, str]:
        """vocab.json을 C 엔진용 형식으로 자동 변환"""

        # 1. 형식 확인
        sample_value = next(iter(vocab.values())) if vocab else None
        if sample_value and isinstance(sample_value, str) and len(sample_value) > 10:
            logger.info("📋 vocab이 이미 템플릿 문자열 형식입니다")
            return vocab

        # 2. parsed.parquet에서 추출
        vocab_dir = Path(vocab_path).parent
        parsed_path = vocab_dir / "parsed.parquet"
        if parsed_path.exists():
            # ... pandas로 template_id → template 매핑 추출

        # 3. preview.json에서 추출
        preview_path = vocab_dir / "preview.json"
        if preview_path.exists():
            # ... JSON에서 template_id → template 매핑 추출

        # 4. 변환 실패 시 경고
        logger.warning("⚠️  템플릿 문자열을 추출할 수 없습니다.")
        return vocab
```

### 2. convert_deeplog_to_onnx 메서드 수정

**변경 전**:
```python
# 어휘 사전도 함께 복사
vocab_output = self.output_dir / "vocab.json"
with open(vocab_output, 'w') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
```

**변경 후**:
```python
# 어휘 사전 처리 (자동 변환!)
vocab_output = self.output_dir / "vocab.json"
vocab_for_c_engine = self._convert_vocab_for_c_engine(vocab, vocab_path)

with open(vocab_output, 'w') as f:
    json.dump(vocab_for_c_engine, f, ensure_ascii=False, indent=2)

# 형식 확인 메시지
sample_value = next(iter(vocab_for_c_engine.values())) if vocab_for_c_engine else None
if sample_value and isinstance(sample_value, str) and len(sample_value) > 10:
    logger.info(f"✅ C 엔진용 vocab 형식 (template strings): {len(vocab_for_c_engine)} templates")
else:
    logger.warning(f"⚠️  vocab이 인덱스 형식입니다. C 엔진 사용 시 템플릿 문자열이 필요합니다.")
```

## 📝 문서 업데이트

### 1. ONNX Conversion Guide (신규)

**파일**: `docs/guides/ONNX_CONVERSION_GUIDE.md`

- 자동 vocab 변환 기능 설명
- 빠른 시작 가이드
- 트러블슈팅 섹션
- 베스트 프랙티스

### 2. Hybrid Inference README 업데이트

**파일**: `hybrid_system/inference/README.md`

- 사용법 섹션에 ONNX 변환 명령어 추가
- 자동 vocab 변환 안내 추가

### 3. Vocab Issue Resolved (이전 문서)

**파일**: `docs/development/VOCAB_ISSUE_RESOLVED.md`

- 문제의 원인과 해결 과정 문서화
- 수동 변환 방법도 함께 보관 (하위 호환성)

## ✅ 이점

### 사용자 경험 개선

1. **단순화된 워크플로우**
   - 2단계 → 1단계로 감소
   - 수동 개입 불필요

2. **에러 방지**
   - vocab 형식 불일치 자동 해결
   - "0 templates" 에러 사전 차단

3. **명확한 피드백**
   - 변환 성공 시: "✅ C 엔진용 vocab 형식"
   - 변환 실패 시: 상세한 해결 방법 안내

### 개발자 경험 개선

1. **자동화**
   - parsed.parquet 감지
   - preview.json fallback
   - 명확한 에러 메시지

2. **유연성**
   - 이미 올바른 형식이면 건너뛰기
   - 여러 소스에서 템플릿 추출 시도
   - 실패 시에도 명확한 가이드

3. **하위 호환성**
   - 기존 수동 변환 스크립트 유지
   - 특수한 경우에 대비

## 🔄 관련 변경

1. **model_converter.py**: 자동 vocab 변환 로직 추가
2. **export_vocab_with_templates.py**: 수동 변환 스크립트 유지
3. **ONNX_CONVERSION_GUIDE.md**: 신규 가이드 문서
4. **AUTO_VOCAB_CONVERSION.md**: 이 문서 (구현 상세)
5. **hybrid_system/inference/README.md**: 사용법 업데이트

## 📊 테스트 결과

### Test 1: 자동 변환 성공

```bash
$ rm -rf hybrid_system/inference/models/*
$ python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models

✅ 결과: vocab.json에 템플릿 문자열 7개 포함
✅ C 엔진 실행: "Loaded vocabulary: 7 templates"
```

### Test 2: 이미 올바른 형식

```bash
$ python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab hybrid_system/inference/models/vocab.json \
    --output-dir hybrid_system/inference/models

✅ 결과: "📋 vocab이 이미 템플릿 문자열 형식입니다" 메시지
✅ vocab.json 유지됨
```

### Test 3: 변환 실패 (parsed.parquet 없음)

```bash
$ python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab /tmp/vocab.json \
    --output-dir /tmp/models

⚠️  결과: 경고 메시지 + 수동 변환 명령어 안내
✅ 명확한 해결 방법 제시
```

## 🎯 향후 개선 사항

### 1. CLI 통합

현재:
```bash
python hybrid_system/training/model_converter.py --deeplog-model model.pth ...
```

개선안:
```bash
alog-detect convert-onnx --deeplog-model model.pth ...
```

### 2. 배치 변환 지원

여러 모델을 한 번에 변환:
```bash
alog-detect convert-onnx-batch \
    --models-dir .cache \
    --output-dir hybrid_system/inference/models
```

### 3. 검증 강화

vocab.json 형식을 더 엄격하게 검증:
- 템플릿 문자열 최소 길이 확인
- 마스킹 토큰 존재 여부 확인
- C 엔진과의 호환성 테스트

## 📅 변경 이력

- **2025-10-17**: 자동 vocab 변환 기능 구현 및 문서화
- **관련 이슈**: Vocabulary 0 templates 문제
- **PR**: (해당 없음 - 직접 작업)

## 🔗 관련 문서

- [ONNX Conversion Guide](../guides/ONNX_CONVERSION_GUIDE.md) - 사용자 가이드
- [Vocab Issue Resolved](./VOCAB_ISSUE_RESOLVED.md) - 문제 분석 및 해결
- [Model Converter Source](../../hybrid_system/training/model_converter.py) - 소스 코드
- [Export Vocab Script](../../hybrid_system/training/export_vocab_with_templates.py) - 수동 변환 스크립트
