# train_models.sh와 ONNX 변환 문제 해결

## 🐛 문제

`train_models.sh`가 `training_workspace/` 하위에 파일을 생성하여 ONNX 변환 시 자동 vocab 변환이 실패합니다.

### 현재 구조 (문제)

```
basemodel/
├── deeplog.pth                    # 모델
├── drain3_state.json
└── training_workspace/            # 작업 디렉토리
    ├── parsed.parquet             # 여기에 있음
    ├── vocab.json                 # 여기에 있음
    └── sequences.parquet
```

```bash
# ONNX 변환 시도
python model_converter.py \
    --vocab basemodel/vocab.json \           # 없음! ❌
    --output-dir basemodel/onnx

# 결과: 자동 변환 실패
WARNING:__main__:⚠️  템플릿 문자열을 추출할 수 없습니다.
WARNING:__main__:⚠️  basemodel에 parsed.parquet 또는 preview.json이 필요합니다.
```

## ✅ 해결 방법

### Option 1: vocab 경로 수정 (즉시 사용 가능!)

```bash
# train_models.sh 실행
./scripts/train_models.sh /var/log/normal/ basemodel

# ONNX 변환 (training_workspace 경로 지정!)
python hybrid_system/training/model_converter.py \
    --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/training_workspace/vocab.json \  # ← 수정!
    --output-dir basemodel/onnx
```

**자동 변환 성공**:
```
INFO:__main__:🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...
INFO:__main__:📂 parsed.parquet에서 템플릿 추출: basemodel/training_workspace/parsed.parquet
INFO:__main__:✅ 7개 템플릿 추출 완료
INFO:__main__:✅ C 엔진용 vocab 형식 (template strings): 7 templates
```

### Option 2: 파일 복사 후 변환

```bash
# train_models.sh 실행
./scripts/train_models.sh /var/log/normal/ basemodel

# 필요한 파일을 최상위로 복사
cp basemodel/training_workspace/vocab.json basemodel/
cp basemodel/training_workspace/parsed.parquet basemodel/
cp basemodel/training_workspace/sequences.parquet basemodel/
cp basemodel/training_workspace/preview.json basemodel/ 2>/dev/null || true

# ONNX 변환
python hybrid_system/training/model_converter.py \
    --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/vocab.json \
    --output-dir basemodel/onnx
```

**결과 구조**:
```
basemodel/
├── deeplog.pth
├── vocab.json                 # 복사됨
├── parsed.parquet             # 복사됨
├── sequences.parquet          # 복사됨
├── preview.json               # 복사됨
├── training_workspace/        # 원본 유지
└── onnx/
    ├── deeplog.onnx
    └── vocab.json             # 템플릿 문자열 형식
```

### Option 3: train_models.sh 수정 (권장!)

`scripts/train_models.sh`의 127-128번 라인을 수정:

**변경 전**:
```bash
# 모델 저장 디렉토리 생성
mkdir -p "$MODEL_DIR"
WORK_DIR="$MODEL_DIR/training_workspace"
mkdir -p "$WORK_DIR"
```

**변경 후**:
```bash
# 모델 저장 디렉토리 생성
mkdir -p "$MODEL_DIR"
WORK_DIR="$MODEL_DIR"  # training_workspace 제거!
```

그리고 397-398번 라인 수정:

**변경 전**:
```bash
if [ -f "$WORK_DIR/sequences.parquet" ] && [ -f "$WORK_DIR/vocab.json" ]; then
    # vocab.json을 모델 디렉토리로 복사 (입력 생성 성공시 항상)
    cp "$WORK_DIR/vocab.json" "$MODEL_DIR/"
```

**변경 후**:
```bash
if [ -f "$WORK_DIR/sequences.parquet" ] && [ -f "$WORK_DIR/vocab.json" ]; then
    # vocab.json이 이미 MODEL_DIR에 있으므로 복사 불필요
    # cp "$WORK_DIR/vocab.json" "$MODEL_DIR/"  # 주석 처리
```

**장점**: 한 번만 수정하면 이후 계속 올바른 구조 생성

### Option 4: 편의 스크립트 생성

```bash
#!/bin/bash
# scripts/train_and_convert.sh

set -e

BASEMODEL_DIR="${1:-basemodel}"
LOG_DIR="${2:-/var/log/normal}"

echo "🚀 통합 학습 및 ONNX 변환 스크립트"
echo "   - Basemodel 디렉토리: $BASEMODEL_DIR"
echo "   - 로그 디렉토리: $LOG_DIR"
echo ""

# 1. 모델 학습
echo "1️⃣  모델 학습 중..."
./scripts/train_models.sh "$LOG_DIR" "$BASEMODEL_DIR"

# 2. 파일 복사 (training_workspace → basemodel)
echo ""
echo "2️⃣  파일 정리 중..."
if [ -d "$BASEMODEL_DIR/training_workspace" ]; then
    cp "$BASEMODEL_DIR/training_workspace/vocab.json" "$BASEMODEL_DIR/"
    cp "$BASEMODEL_DIR/training_workspace/parsed.parquet" "$BASEMODEL_DIR/"
    cp "$BASEMODEL_DIR/training_workspace/sequences.parquet" "$BASEMODEL_DIR/"
    cp "$BASEMODEL_DIR/training_workspace/preview.json" "$BASEMODEL_DIR/" 2>/dev/null || true
    echo "✅ 필요한 파일을 basemodel/ 최상위로 복사 완료"
fi

# 3. ONNX 변환
echo ""
echo "3️⃣  ONNX 변환 중..."
mkdir -p "$BASEMODEL_DIR/onnx"
python hybrid_system/training/model_converter.py \
    --deeplog-model "$BASEMODEL_DIR/deeplog.pth" \
    --vocab "$BASEMODEL_DIR/vocab.json" \
    --output-dir "$BASEMODEL_DIR/onnx"

echo ""
echo "✅ 완료!"
echo "   - PyTorch 모델: $BASEMODEL_DIR/deeplog.pth"
echo "   - ONNX 모델: $BASEMODEL_DIR/onnx/deeplog.onnx"
echo "   - Python용 vocab: $BASEMODEL_DIR/vocab.json (인덱스)"
echo "   - ONNX용 vocab: $BASEMODEL_DIR/onnx/vocab.json (템플릿 문자열)"
```

사용법:
```bash
chmod +x scripts/train_and_convert.sh
./scripts/train_and_convert.sh basemodel /var/log/normal
```

## 📊 비교표

| 방법 | 장점 | 단점 | 권장도 |
|------|------|------|--------|
| Option 1: vocab 경로 수정 | 즉시 사용 가능 | 경로가 김 | ⭐⭐⭐⭐ |
| Option 2: 파일 복사 | 구조 깔끔 | 수동 작업 필요 | ⭐⭐⭐ |
| Option 3: train_models.sh 수정 | 가장 깔끔함 | 스크립트 수정 필요 | ⭐⭐⭐⭐⭐ |
| Option 4: 편의 스크립트 | 자동화됨 | 새 스크립트 필요 | ⭐⭐⭐⭐ |

## 🚀 권장 워크플로우

### 즉시 사용 (Option 1)

```bash
# 1. 학습
./scripts/train_models.sh /var/log/normal/ basemodel

# 2. ONNX 변환 (training_workspace 경로!)
python hybrid_system/training/model_converter.py \
    --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/training_workspace/vocab.json \
    --output-dir basemodel/onnx
```

### 장기 사용 (Option 3 권장)

1. `scripts/train_models.sh` 수정 (127-128번 라인)
2. 이후 정상적으로 사용:

```bash
# 1. 학습
./scripts/train_models.sh /var/log/normal/ basemodel

# 2. ONNX 변환 (깔끔한 경로!)
python hybrid_system/training/model_converter.py \
    --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/vocab.json \
    --output-dir basemodel/onnx
```

## 🔍 검증 방법

### 자동 변환 성공 확인

```bash
# ONNX 변환 실행
python model_converter.py ... 2>&1 | grep -E "(템플릿|vocab)"
```

**성공**:
```
INFO:__main__:🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...
INFO:__main__:📂 parsed.parquet에서 템플릿 추출: basemodel/.../parsed.parquet
INFO:__main__:✅ 7개 템플릿 추출 완료
INFO:__main__:✅ C 엔진용 vocab 형식 (template strings): 7 templates
```

**실패**:
```
WARNING:__main__:⚠️  템플릿 문자열을 추출할 수 없습니다.
```

### vocab.json 형식 확인

```bash
# ONNX용 vocab 확인
cat basemodel/onnx/vocab.json | head -3
```

**올바른 출력** (템플릿 문자열):
```json
{
  "1": "[<NUM>] usb 1-1: new high-speed USB device...",  ✅
  "2": "[<NUM>] CPU<ID>: Core temperature...",
```

**잘못된 출력** (인덱스):
```json
{
  "1": 0,  ❌
  "2": 1,
```

## ✅ 최종 권장사항

**Option 3 (train_models.sh 수정)을 권장합니다!**

이유:
- ✅ 한 번만 수정하면 영구적으로 해결
- ✅ 디렉토리 구조가 깔끔해짐
- ✅ 자동 변환이 항상 성공
- ✅ 문서와 일치하는 구조

수정 후 구조:
```
basemodel/
├── deeplog.pth
├── vocab.json              # Python용 (인덱스)
├── parsed.parquet          # 템플릿 문자열 포함
├── sequences.parquet
├── preview.json
│
└── onnx/
    ├── deeplog.onnx
    └── vocab.json          # ONNX/C용 (템플릿 문자열)
```

**완벽한 자동화!** 🎉
