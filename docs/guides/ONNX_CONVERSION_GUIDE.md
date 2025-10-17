# ONNX 변환 가이드

## 🎯 개요

PyTorch로 학습한 DeepLog/MS-CRED 모델을 ONNX 포맷으로 변환하여 C inference engine에서 사용할 수 있습니다.

## 🚀 빠른 시작

### 1단계: 모델 학습

```bash
# 로그 파싱
alog-detect parse --input data/raw/system.log --out-dir data/processed

# DeepLog 시퀀스 생성 및 학습
alog-detect train-deeplog \
    --parsed data/processed/parsed.parquet \
    --out-dir data/processed \
    --epochs 50

# 모델 저장 위치: .cache/deeplog.pth
```

### 2단계: ONNX 변환 (자동 vocab 변환 포함!)

```bash
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models
```

**출력 파일**:
- ✅ `deeplog.onnx` - ONNX 모델
- ✅ `deeplog_optimized.onnx` - 최적화된 ONNX 모델
- ✅ `vocab.json` - **자동으로 C 엔진용 형식으로 변환됨!**
- ✅ `deeplog.onnx.meta.json` - 모델 메타데이터
- ✅ `conversion_summary.json` - 변환 요약

### 3단계: C Inference Engine 실행

```bash
# ONNX Runtime 설치 (처음 한 번만)
./scripts/install_onnxruntime.sh

# Inference Engine 빌드
cd hybrid_system/inference
make clean && make

# 테스트 모드로 실행
./bin/inference_engine \
    -d models/deeplog.onnx \
    -v models/vocab.json \
    -t

# 실제 로그 파일 처리
./bin/inference_engine \
    -d models/deeplog.onnx \
    -v models/vocab.json \
    -i /var/log/syslog \
    -o results.json
```

## 📋 자동 Vocab 변환 동작 방식

### 기존 문제

Python 학습 시 생성되는 `vocab.json`:
```json
{
  "1": 0,   // template_id → index
  "2": 1,
  "3": 2
}
```

C 엔진이 필요한 `vocab.json`:
```json
{
  "1": "[<NUM>] usb device connected",  // template_id → template_string
  "2": "[<NUM>] CPU temperature high",
  "3": "[<NUM>] Network link up"
}
```

### ✅ 자동 변환 로직

`model_converter.py`는 이제 **자동으로** 올바른 형식의 vocab.json을 생성합니다:

1. **parsed.parquet에서 추출** (우선순위 1)
   - `data/processed/parsed.parquet` 파일 확인
   - `template_id`와 `template` 컬럼에서 매핑 추출

2. **preview.json에서 추출** (우선순위 2)
   - `data/processed/preview.json` 파일 확인
   - 각 항목의 `template_id`와 `template` 필드 추출

3. **경고 표시** (변환 불가능한 경우)
   - 필요한 파일이 없으면 경고 메시지 출력
   - 수동 변환 명령어 안내

## 🔧 고급 사용법

### Option 1: 검증 포함 변환

```bash
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models \
    --validate  # ONNX 모델 검증 실행
```

### Option 2: MS-CRED도 함께 변환

```bash
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --mscred-model .cache/mscred.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models
```

### Option 3: 수동 Vocab 변환 (필요 시)

parsed.parquet가 vocab.json과 다른 위치에 있는 경우:

```bash
python hybrid_system/training/export_vocab_with_templates.py \
    /path/to/parsed.parquet \
    hybrid_system/inference/models/vocab.json
```

## 🐛 트러블슈팅

### 문제 1: "Loaded vocabulary: 0 templates"

**원인**: vocab.json이 인덱스 형식입니다.

**해결**:
```bash
# 자동 변환 (권장)
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models

# 또는 수동 변환
python hybrid_system/training/export_vocab_with_templates.py \
    data/processed/parsed.parquet \
    hybrid_system/inference/models/vocab.json
```

**검증**:
```bash
# vocab.json 내용 확인 - 값이 문자열이어야 함
cat hybrid_system/inference/models/vocab.json

# 올바른 예시:
# {
#   "1": "[<NUM>] usb device connected",  ✅
#   "2": "[<NUM>] CPU temperature high"   ✅
# }

# 잘못된 예시:
# {
#   "1": 0,  ❌
#   "2": 1   ❌
# }
```

### 문제 2: "parsed.parquet 또는 preview.json이 필요합니다"

**원인**: vocab 자동 변환에 필요한 파일이 없습니다.

**해결**:

Option A: parsed.parquet 생성
```bash
alog-detect parse \
    --input data/raw/system.log \
    --out-dir data/processed
```

Option B: preview.json이 있는 경로로 vocab_path 지정
```bash
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \  # preview.json이 같은 디렉토리에 있어야 함
    --output-dir hybrid_system/inference/models
```

### 문제 3: ONNX Runtime 빌드 에러

**증상**:
```
fatal error: onnxruntime_c_api.h: 그런 파일이나 디렉터리가 없습니다
```

**해결**:
```bash
# ONNX Runtime C API 설치
./scripts/install_onnxruntime.sh

# 또는 수동 설치
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/
sudo ldconfig

# 검증
ls -l /usr/local/lib/libonnxruntime*
ls -l /usr/local/include/onnxruntime_c_api.h
```

## 📊 변환 결과 확인

### ONNX 모델 정보

```bash
# 메타데이터 확인
cat hybrid_system/inference/models/deeplog.onnx.meta.json
```

출력 예시:
```json
{
  "model_type": "deeplog",
  "vocab_size": 7,
  "seq_len": 3,
  "input_shape": [1, 3],
  "output_shape": [1, 7],
  "input_names": ["input_sequence"],
  "output_names": ["predictions"],
  "opset_version": 11
}
```

### Vocab 형식 확인

```bash
# vocab.json 확인
cat hybrid_system/inference/models/vocab.json | head -5
```

**올바른 출력**:
```json
{
  "1": "[<NUM>] usb 1-1: new high-speed USB device number <NUM> using ehci-pci",
  "2": "[<NUM>] CPU<ID>: Core temperature above threshold, cpu clock throttled",
  ...
}
```

## 🎯 베스트 프랙티스

### 1. 표준 워크플로우

```bash
# 1. 파싱 (parsed.parquet + preview.json + vocab.json 생성)
alog-detect parse --input data/raw/system.log --out-dir data/processed

# 2. 학습 (deeplog.pth 생성)
alog-detect train-deeplog --parsed data/processed/parsed.parquet --out-dir data/processed

# 3. ONNX 변환 (자동으로 올바른 vocab.json 생성!)
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models \
    --validate

# 4. C 엔진 실행
cd hybrid_system/inference
make && ./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -t
```

### 2. 디렉토리 구조

```
project/
├── data/
│   ├── raw/
│   │   └── system.log              # 원본 로그
│   └── processed/
│       ├── parsed.parquet          # 파싱 결과 (템플릿 포함!)
│       ├── preview.json            # 미리보기 (템플릿 포함!)
│       ├── vocab.json              # Python용 (인덱스 형식)
│       └── sequences.parquet       # 시퀀스 데이터
├── .cache/
│   └── deeplog.pth                 # 학습된 모델
└── hybrid_system/
    ├── training/
    │   ├── model_converter.py      # ONNX 변환 (자동 vocab 변환!)
    │   └── export_vocab_with_templates.py  # 수동 vocab 변환
    └── inference/
        ├── models/
        │   ├── deeplog.onnx        # ONNX 모델
        │   ├── vocab.json          # C 엔진용 (템플릿 문자열!)
        │   └── *.meta.json         # 메타데이터
        └── bin/
            └── inference_engine    # C 추론 엔진
```

### 3. 체크리스트

ONNX 변환 전:
- [ ] `data/processed/parsed.parquet` 존재 확인
- [ ] `data/processed/vocab.json` 존재 확인
- [ ] `.cache/deeplog.pth` 존재 확인

ONNX 변환 후:
- [ ] `hybrid_system/inference/models/deeplog.onnx` 생성 확인
- [ ] `hybrid_system/inference/models/vocab.json` 형식 확인 (템플릿 문자열!)
- [ ] Vocab 로그에 "✅ C 엔진용 vocab 형식" 메시지 확인

C 엔진 실행 전:
- [ ] ONNX Runtime C API 설치 완료
- [ ] `make` 빌드 성공
- [ ] Test 모드에서 "Loaded vocabulary: N templates" (N > 0) 확인

## 🔗 관련 문서

- [C Inference Engine README](../../hybrid_system/inference/README.md)
- [Vocab Issue Resolved](../development/VOCAB_ISSUE_RESOLVED.md)
- [Train & Inference Guide](./TRAIN_INFERENCE_GUIDE.md)
- [Model Converter Source](../../hybrid_system/training/model_converter.py)

## 📅 업데이트

- **2025-10-17**: 자동 vocab 변환 기능 추가 - 더 이상 수동 변환 불필요!
