# 세션 요약 - ONNX 변환 문제 해결

**날짜**: 2025년 (세션 종료)
**주제**: ONNX 변환 시 발생하는 여러 문제 해결

---

## 🎯 해결한 문제들

### 1. ✅ Constraints violated (L['x'].size()[1]) 에러
**원인**: ONNX 모델 변환 시 시퀀스 길이 제약
**해결**:
- `--seq-len` 옵션 추가 (CLI 및 model_converter.py)
- 메타데이터에 dynamic_axes 정보 명시
- C 추론 시 메타데이터의 seq_len 사용 권장

### 2. ✅ PyTorch 2.1+ Dynamo Export 에러
**원인**: PyTorch 2.1+에서 torch.onnx.export가 dynamo 방식을 기본으로 시도
**해결**:
- `dynamo=False` 명시적 지정으로 레거시 TorchScript 방식 강제
- PyTorch 2.0 이하 호환성 유지 (TypeError 예외 처리)

### 3. ✅ Vocab.json이 template_id 형식 문제
**원인**: `build_deeplog_inputs()`가 `template_col="template_id"` 사용
**해결**:
- 기본값을 `template_col="template"`로 변경
- model_converter.py에 검증 로직 추가
- fix_vocab_format.py 유틸리티 생성

---

## 📝 수정된 파일들

```
modified:   anomaly_log_detector/cli.py
modified:   anomaly_log_detector/builders/deeplog.py
modified:   hybrid_system/training/model_converter.py
modified:   docs/development/CLAUDE.md
modified:   docs/development/FIX_ONNX_DYNAMO_ERROR.md
new file:   scripts/fix_vocab_format.py
```

### 주요 변경 사항

#### 1. `anomaly_log_detector/cli.py`
- `convert-onnx` 명령어에 `--seq-len` 옵션 추가

#### 2. `anomaly_log_detector/builders/deeplog.py`
```python
# 변경 전
def build_deeplog_inputs(..., template_col: str = "template_id"):

# 변경 후
def build_deeplog_inputs(..., template_col: str = "template"):
```

#### 3. `hybrid_system/training/model_converter.py`
- `dynamo=False` 명시로 레거시 TorchScript 강제 사용
- vocab 형식 검증 로직 추가 (template_id 사용 시 에러)
- `--seq-len` 파라미터 지원

#### 4. `scripts/fix_vocab_format.py` (신규)
- 기존 vocab.json을 template_id → template string으로 변환하는 유틸리티

---

## 🔄 다음 세션에서 할 일

### 즉시 필요한 작업

#### 1. 기존 vocab.json 수정
기존 학습된 모델이 있다면:

**방법 A - 전체 재생성 (권장)**:
```bash
# 1. vocab.json 재생성
alog-detect build-deeplog \
  --parsed data/parsed.parquet \
  --out-dir training_workspace/

# 2. vocab 확인
head training_workspace/vocab.json
# 형식: {"User <ID> logged in": 0, ...}

# 3. 모델 재학습
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

**방법 B - vocab만 변환 (빠름, 모델 재학습 권장)**:
```bash
# 1. vocab 변환
python scripts/fix_vocab_format.py \
  --parsed data/parsed.parquet \
  --old-vocab training_workspace/vocab.json \
  --output training_workspace/vocab_fixed.json

# 2. 백업 및 교체
mv training_workspace/vocab.json training_workspace/vocab.json.bak
mv training_workspace/vocab_fixed.json training_workspace/vocab.json

# 3. ONNX 변환
alog-detect convert-onnx \
  --deeplog-model training_workspace/deeplog.pth \
  --vocab training_workspace/vocab.json \
  --output-dir models/onnx
```

#### 2. ONNX 변환 테스트

**DeepLog 변환**:
```bash
alog-detect convert-onnx \
  --deeplog-model training_workspace/deeplog.pth \
  --vocab training_workspace/vocab.json \
  --seq-len 50 \
  --output-dir models/onnx \
  --portable \
  --validate
```

**MSCRED 변환** (shape: 94, 1, 20, 47):
```bash
alog-detect convert-onnx \
  --mscred-model training_workspace/mscred.pth \
  --vocab training_workspace/vocab.json \
  --feature-dim 47 \
  --output-dir models/onnx \
  --portable \
  --validate
```

**참고**: window_size=20이 필요한 경우 Python 코드 직접 수정:
```python
from hybrid_system.training.model_converter import ModelConverter

converter = ModelConverter(output_dir="models/onnx")
result = converter.convert_mscred_to_onnx(
    model_path="training_workspace/mscred.pth",
    window_size=20,
    feature_dim=47
)
```

#### 3. C 추론 엔진 테스트

```bash
cd hybrid_system/inference
make clean && make
./bin/inference_engine ../../models/onnx
```

---

## 📚 vocab.json 형식 정리

### Python 학습/추론용 (training_workspace/vocab.json)
```json
{
  "User <ID> logged in": 0,
  "System started successfully": 1,
  "Error: <PATH> not found": 2
}
```
- **Key**: 실제 템플릿 문자열
- **Value**: index

### C 엔진용 (models/onnx/vocab.json)
```json
{
  "0": "User <ID> logged in",
  "1": "System started successfully",
  "2": "Error: <PATH> not found"
}
```
- **Key**: index (문자열)
- **Value**: 템플릿 문자열
- **생성**: model_converter.py가 자동 변환

---

## 🛠️ 추가된 CLI 옵션

### convert-onnx 명령어

```bash
alog-detect convert-onnx --help

# 새로운 옵션들:
--seq-len INT          # DeepLog 시퀀스 길이 (기본: 모델 저장값)
--feature-dim INT      # MSCRED 피처 차원 (템플릿 개수, 기본: vocab 크기)
--portable             # 범용 최적화 (모든 환경 호환)
--validate             # 변환 후 검증
```

---

## 📖 참고 문서

- **[CLAUDE.md](CLAUDE.md)**: 프로젝트 전체 가이드 및 Troubleshooting
- **[FIX_ONNX_DYNAMO_ERROR.md](FIX_ONNX_DYNAMO_ERROR.md)**: ONNX 변환 문제 상세 가이드

### Troubleshooting 섹션 추가됨
1. **Vocab Format Error** (template_id vs template string)
2. **ONNX Export Errors** (dynamo / mark_dynamic / Dim.DYNAMIC)
3. **ONNX Sequence Length Constraints Error**

---

## ⚠️ 중요 체크리스트

다음 세션 시작 전 확인:

- [ ] 최신 코드가 다른 시스템에 pull 되었는가?
- [ ] 기존 vocab.json이 template_id 형식인가? (head vocab.json 확인)
- [ ] 새로 학습할 것인가, 기존 vocab만 변환할 것인가?
- [ ] MSCRED 변환 시 feature_dim 값을 알고 있는가?
- [ ] window_size가 기본값(50)이 아닌 경우 Python 코드 수정 필요

---

## 🎉 성과

1. ✅ PyTorch 2.1+ 호환성 확보 (dynamo=False)
2. ✅ vocab.json 형식 문제 해결 (template_id → template)
3. ✅ 시퀀스 길이 유연성 확보 (--seq-len 옵션)
4. ✅ 검증 로직 추가 (잘못된 vocab 조기 발견)
5. ✅ 문서화 완료 (CLAUDE.md, FIX_ONNX_DYNAMO_ERROR.md)
6. ✅ 변환 유틸리티 추가 (fix_vocab_format.py)

---

## 💡 팁

### vocab.json 빠른 확인
```bash
# 올바른 형식 (template string → index)
head training_workspace/vocab.json
# {
#   "User <ID> logged in": 0,
#   ...
# }

# 잘못된 형식 (template_id → index)
# {
#   "1": 0,
#   "2": 1,
#   ...
# }
```

### 변환 성공 확인
```bash
# ONNX 파일 생성 확인
ls -lh models/onnx/
# deeplog.onnx
# deeplog.onnx.meta.json
# mscred.onnx
# mscred.onnx.meta.json
# vocab.json

# vocab.json이 C 엔진용으로 변환되었는지 확인
head models/onnx/vocab.json
# {
#   "0": "User <ID> logged in",
#   ...
# }
```

---

**다음 세션 시작 시**: 이 문서를 먼저 읽고, 체크리스트를 확인한 후 작업 시작! 🚀
