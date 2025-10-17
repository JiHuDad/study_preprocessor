# Basemodel 디렉토리 기준 워크플로우

## 🎯 디렉토리 구조

### 목표 구조

```
basemodel/                          # 모델 통합 디렉토리
├── parsed.parquet                  # 파싱된 로그 데이터
├── preview.json                    # 파싱 미리보기
├── vocab.json                      # Python 학습용 (인덱스 형식)
├── sequences.parquet               # 시퀀스 데이터
├── deeplog.pth                     # 학습된 PyTorch 모델
│
└── onnx/                           # ONNX 변환 결과
    ├── deeplog.onnx                # ONNX 모델
    ├── deeplog_optimized.onnx      # 최적화된 ONNX
    ├── vocab.json                  # ONNX/C 엔진용 (템플릿 문자열)
    ├── deeplog.onnx.meta.json      # 메타데이터
    └── conversion_summary.json     # 변환 요약
```

## 📋 전체 워크플로우

### Step 1: 로그 파싱

```bash
# basemodel 디렉토리 생성
mkdir -p basemodel

# 로그 파싱
alog-detect parse \
    --input data/raw/system.log \
    --out-dir basemodel
```

**생성되는 파일**:
```
basemodel/
├── parsed.parquet       # ✅ 템플릿 문자열 포함
├── preview.json         # ✅ 템플릿 문자열 포함
├── vocab.json           # ⚠️  인덱스 형식 (Python 학습용)
└── sequences.parquet
```

**vocab.json 내용** (Python 학습용):
```json
{
  "1": 0,
  "2": 1,
  "3": 2
}
```

### Step 2: 모델 학습

```bash
# DeepLog 학습
alog-detect train-deeplog \
    --parsed basemodel/parsed.parquet \
    --out-dir basemodel \
    --epochs 50
```

**생성되는 파일**:
```
basemodel/
├── parsed.parquet
├── preview.json
├── vocab.json           # ⚠️  인덱스 형식 유지 (수정 안 됨)
├── sequences.parquet
└── deeplog.pth          # ✅ 새로 생성됨
```

**중요**: `vocab.json`은 **수정되지 않습니다!** 인덱스 형식 그대로 유지됩니다.

### Step 3: ONNX 변환 (자동 vocab 변환!)

```bash
# onnx 디렉토리 생성
mkdir -p basemodel/onnx

# ONNX 변환
python hybrid_system/training/model_converter.py \
    --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/vocab.json \
    --output-dir basemodel/onnx
```

**자동 처리 과정**:
1. ✅ `basemodel/vocab.json` 읽기 (인덱스 형식)
2. ✅ `basemodel/parsed.parquet` 자동 감지 (같은 디렉토리!)
3. ✅ `parsed.parquet`에서 템플릿 문자열 추출
4. ✅ `basemodel/onnx/vocab.json` 생성 (템플릿 문자열 형식)

**생성되는 파일**:
```
basemodel/
├── parsed.parquet
├── preview.json
├── vocab.json           # ⚠️  인덱스 형식 (그대로 유지!)
├── sequences.parquet
├── deeplog.pth
│
└── onnx/                # ✅ 새로 생성됨
    ├── deeplog.onnx
    ├── deeplog_optimized.onnx
    ├── vocab.json       # ✅ 템플릿 문자열 형식!
    ├── deeplog.onnx.meta.json
    └── conversion_summary.json
```

**로그 출력**:
```
INFO:__main__:🔄 DeepLog 모델 변환 시작: basemodel/deeplog.pth
INFO:__main__:🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...
INFO:__main__:📂 parsed.parquet에서 템플릿 추출: basemodel/parsed.parquet
INFO:__main__:✅ 7개 템플릿 추출 완료
INFO:__main__:✅ DeepLog 변환 완료: basemodel/onnx/deeplog.onnx
INFO:__main__:📚 어휘 사전: basemodel/onnx/vocab.json
INFO:__main__:✅ C 엔진용 vocab 형식 (template strings): 7 templates
```

### Step 4: C 엔진 실행

```bash
# ONNX Runtime 설치 (처음 한 번만)
./scripts/install_onnxruntime.sh

# C 엔진 빌드
cd hybrid_system/inference
make clean && make

# 테스트 실행
./bin/inference_engine \
    -d ../../basemodel/onnx/deeplog.onnx \
    -v ../../basemodel/onnx/vocab.json \
    -t
```

**출력**:
```
Loaded vocabulary: 7 templates  ✅
Vocabulary loaded: 7 templates  ✅
```

## 📊 두 vocab.json의 차이

### basemodel/vocab.json (Python 학습용)

```json
{
  "1": 0,
  "2": 1,
  "3": 2,
  "4": 3,
  "5": 4,
  "6": 5,
  "7": 6
}
```

**용도**: PyTorch 모델 학습
**형식**: template_id → index 매핑
**수정**: ❌ 절대 수정하지 않음

### basemodel/onnx/vocab.json (ONNX/C 엔진용)

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

**용도**: ONNX 모델 추론, C inference engine
**형식**: template_id → template_string 매핑
**생성**: ✅ `model_converter.py`가 자동 생성

## 🔍 자동 변환 작동 원리

### 핵심 로직

```python
# model_converter.py의 _convert_vocab_for_c_engine()

# 1. --vocab 옵션으로 지정된 경로의 디렉토리 추출
vocab_path = "basemodel/vocab.json"
vocab_dir = Path(vocab_path).parent  # → "basemodel/"

# 2. 같은 디렉토리에서 parsed.parquet 찾기
parsed_path = vocab_dir / "parsed.parquet"  # → "basemodel/parsed.parquet"

# 3. parsed.parquet에서 템플릿 추출
if parsed_path.exists():
    df = pd.read_parquet(parsed_path)
    template_map = {}
    for _, row in df[['template_id', 'template']].iterrows():
        template_map[row['template_id']] = row['template']

    # 4. 출력 디렉토리에 저장
    # → basemodel/onnx/vocab.json (템플릿 문자열 형식)
    return template_map
```

**핵심**: `--vocab basemodel/vocab.json`를 지정하면, 같은 디렉토리(`basemodel/`)의 `parsed.parquet`에서 자동으로 템플릿 추출!

## ✅ 당신이 해야 할 것

### 전체 명령어 (순서대로)

```bash
# 1. basemodel 디렉토리 생성
mkdir -p basemodel

# 2. 로그 파싱
alog-detect parse --input data/raw/system.log --out-dir basemodel

# 3. 모델 학습
alog-detect train-deeplog \
    --parsed basemodel/parsed.parquet \
    --out-dir basemodel \
    --epochs 50

# 4. onnx 디렉토리 생성
mkdir -p basemodel/onnx

# 5. ONNX 변환 (자동 vocab 변환!)
python hybrid_system/training/model_converter.py \
    --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/vocab.json \
    --output-dir basemodel/onnx

# 끝! 모든 것이 자동으로 처리됨 ✅
```

### 간편 스크립트

```bash
#!/bin/bash
# scripts/train_and_convert.sh

set -e

BASEMODEL_DIR="basemodel"
INPUT_LOG="data/raw/system.log"

echo "📂 Creating basemodel directory..."
mkdir -p ${BASEMODEL_DIR}/onnx

echo "1️⃣  Parsing logs..."
alog-detect parse --input ${INPUT_LOG} --out-dir ${BASEMODEL_DIR}

echo "2️⃣  Training DeepLog model..."
alog-detect train-deeplog \
    --parsed ${BASEMODEL_DIR}/parsed.parquet \
    --out-dir ${BASEMODEL_DIR} \
    --epochs 50

echo "3️⃣  Converting to ONNX (with auto vocab conversion)..."
python hybrid_system/training/model_converter.py \
    --deeplog-model ${BASEMODEL_DIR}/deeplog.pth \
    --vocab ${BASEMODEL_DIR}/vocab.json \
    --output-dir ${BASEMODEL_DIR}/onnx

echo "✅ Done! Check ${BASEMODEL_DIR}/onnx/"
echo "   - deeplog.onnx"
echo "   - vocab.json (template strings)"
```

## ❌ 하지 말아야 할 것

### 실수 1: basemodel/vocab.json 수정

```bash
# ❌ 절대 안 됨!
python hybrid_system/training/export_vocab_with_templates.py \
    basemodel/parsed.parquet \
    basemodel/vocab.json  # Python 학습용을 수정하면 안 됨!
```

**이유**: `basemodel/vocab.json`은 Python 학습용이므로 인덱스 형식으로 유지되어야 합니다.

### 실수 2: vocab.json을 다른 위치로 복사

```bash
# ❌ 자동 변환 실패!
cp basemodel/vocab.json /tmp/vocab.json
python model_converter.py --vocab /tmp/vocab.json ...
# → parsed.parquet를 찾을 수 없어서 자동 변환 실패
```

**이유**: vocab.json과 parsed.parquet가 같은 디렉토리에 있어야 자동 변환 성공!

### 실수 3: onnx/vocab.json을 학습에 사용

```bash
# ❌ 에러 발생!
cp basemodel/onnx/vocab.json basemodel/vocab.json
alog-detect train-deeplog --parsed basemodel/parsed.parquet ...
# → 템플릿 문자열을 인덱스로 변환할 수 없음!
```

## 🔍 검증 방법

### 파싱 후 확인

```bash
# 파일 존재 확인
ls -lh basemodel/
# → parsed.parquet, preview.json, vocab.json, sequences.parquet

# vocab.json 형식 확인 (인덱스여야 함)
cat basemodel/vocab.json
# → {"1": 0, "2": 1, ...}  ✅
```

### 학습 후 확인

```bash
# 모델 파일 확인
ls -lh basemodel/deeplog.pth
# → deeplog.pth 존재

# vocab.json이 그대로인지 확인
cat basemodel/vocab.json
# → {"1": 0, "2": 1, ...}  ✅ (변경 안 됨!)
```

### ONNX 변환 후 확인

```bash
# onnx 디렉토리 확인
ls -lh basemodel/onnx/
# → deeplog.onnx, vocab.json, *.meta.json

# onnx/vocab.json 형식 확인 (템플릿 문자열이어야 함)
cat basemodel/onnx/vocab.json | head -3
# → {"1": "[<NUM>] usb 1-1: new high-speed...", ...}  ✅

# basemodel/vocab.json은 그대로인지 확인
cat basemodel/vocab.json
# → {"1": 0, "2": 1, ...}  ✅ (여전히 인덱스 형식!)
```

### 로그 메시지 확인

```bash
python model_converter.py ... 2>&1 | grep -E "(템플릿|vocab)"
```

**성공**:
```
INFO:__main__:🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...
INFO:__main__:📂 parsed.parquet에서 템플릿 추출: basemodel/parsed.parquet
INFO:__main__:✅ 7개 템플릿 추출 완료
INFO:__main__:✅ C 엔진용 vocab 형식 (template strings): 7 templates
```

**실패**:
```
WARNING:__main__:⚠️  템플릿 문자열을 추출할 수 없습니다.
WARNING:__main__:⚠️  basemodel에 parsed.parquet 또는 preview.json이 필요합니다.
```

## 🌍 다른 시스템으로 이동

### 시나리오: 학습은 서버 A, 추론은 서버 B

#### 서버 A (학습 + ONNX 변환)

```bash
# 전체 프로세스 실행
mkdir -p basemodel/onnx
alog-detect parse --input data/raw/system.log --out-dir basemodel
alog-detect train-deeplog --parsed basemodel/parsed.parquet --out-dir basemodel
python model_converter.py --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/vocab.json --output-dir basemodel/onnx

# onnx 디렉토리만 압축
tar -czf basemodel_onnx.tar.gz basemodel/onnx/
```

#### 서버 B (추론)

```bash
# 파일 전송
scp serverA:basemodel_onnx.tar.gz .
tar -xzf basemodel_onnx.tar.gz

# C 엔진 실행
cd hybrid_system/inference
./bin/inference_engine \
    -d ../../basemodel/onnx/deeplog.onnx \
    -v ../../basemodel/onnx/vocab.json \
    -i /var/log/syslog
```

**중요**:
- ✅ `basemodel/onnx/` 디렉토리만 전송하면 됨
- ❌ `basemodel/vocab.json` (Python용)은 전송 불필요
- ❌ `basemodel/deeplog.pth` (PyTorch)는 전송 불필요

## 📊 요약 표

| 파일 | 위치 | 형식 | 용도 | 수정 여부 |
|------|------|------|------|----------|
| `parsed.parquet` | `basemodel/` | Parquet | 파싱 데이터 (템플릿 포함) | Parse 시 생성 |
| `vocab.json` | `basemodel/` | 인덱스 | Python 학습 | Parse 시 생성, **수정 안 됨** |
| `deeplog.pth` | `basemodel/` | PyTorch | 학습된 모델 | Train 시 생성 |
| `deeplog.onnx` | `basemodel/onnx/` | ONNX | ONNX 모델 | Convert 시 생성 |
| `vocab.json` | `basemodel/onnx/` | 템플릿 문자열 | ONNX/C 추론 | Convert 시 **자동 생성** |

## ✅ 핵심 정리

### 질문: "train 시에는 basemodel 디렉토리를 생성해서, 그 안에 파일들이 생성되고.. 이 학습된 모델을 가지고 onnx로 변환할 건데.. 그것도 basemodel 디렉토리 안에서 onnx 폴더를 생성해서 그 안에 onnx를 넣을거야."

### 답변: ✅ 완벽합니다! 그리고 이미 자동화되어 있습니다!

```bash
# 1. Parse → basemodel/ 생성
alog-detect parse --out-dir basemodel

# 2. Train → basemodel/deeplog.pth 생성
alog-detect train-deeplog --parsed basemodel/parsed.parquet --out-dir basemodel

# 3. Convert → basemodel/onnx/ 생성 (자동 vocab 변환!)
mkdir -p basemodel/onnx
python model_converter.py \
    --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/vocab.json \          # ← 인덱스 형식 (그대로 유지)
    --output-dir basemodel/onnx              # ← 템플릿 문자열 (자동 생성!)

# basemodel/vocab.json은 절대 수정하지 않음 ✅
# basemodel/onnx/vocab.json이 자동으로 올바른 형식으로 생성됨 ✅
```

**수동 작업 불필요! 모든 것이 자동으로 처리됩니다!** 🎉
