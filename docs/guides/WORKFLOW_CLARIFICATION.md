# 학습 → ONNX 변환 워크플로우 명확화

## 🎯 핵심 개념

**두 개의 vocab.json이 존재하며, 각각 다른 용도로 사용됩니다!**

| 파일 위치 | 용도 | 형식 | 생성 시점 |
|----------|------|------|----------|
| `data/processed/vocab.json` | Python 학습 | 인덱스 | `train.sh` 실행 시 |
| `hybrid_system/inference/models/vocab.json` | ONNX/C 엔진 | 템플릿 문자열 | `model_converter.py` 실행 시 |

## 📋 전체 워크플로우

### Step 1: 로그 파싱 (Parse)

```bash
alog-detect parse --input data/raw/system.log --out-dir data/processed
```

**생성되는 파일**:
```
data/processed/
├── parsed.parquet       # ✅ 템플릿 문자열 포함!
├── preview.json         # ✅ 템플릿 문자열 포함!
├── vocab.json           # ⚠️  인덱스 형식 (Python용)
└── sequences.parquet
```

**vocab.json 내용** (Python 학습용):
```json
{
  "1": 0,   // template_id → index
  "2": 1,
  "3": 2
}
```

### Step 2: 모델 학습 (Train)

```bash
# scripts/train.sh 또는
alog-detect train-deeplog \
    --parsed data/processed/parsed.parquet \
    --out-dir data/processed
```

**사용하는 파일**:
- ✅ `data/processed/vocab.json` (인덱스 형식) ← Python 학습에 적합
- ✅ `data/processed/sequences.parquet`

**생성되는 파일**:
```
.cache/
└── deeplog.pth          # PyTorch 모델
```

**중요**: `data/processed/vocab.json`은 **수정하지 않습니다!** 인덱스 형식 그대로 유지됩니다.

### Step 3: ONNX 변환 (Convert)

```bash
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \              # ← Python용 vocab 입력
    --output-dir hybrid_system/inference/models
```

**입력 파일**:
- ✅ `.cache/deeplog.pth` (PyTorch 모델)
- ✅ `data/processed/vocab.json` (인덱스 형식)
- ✅ `data/processed/parsed.parquet` (자동 감지!)

**생성되는 파일**:
```
hybrid_system/inference/models/
├── deeplog.onnx                 # ONNX 모델
├── deeplog_optimized.onnx       # 최적화된 ONNX
├── deeplog.onnx.meta.json       # 메타데이터
├── vocab.json                   # ✅ 자동으로 템플릿 문자열 형식으로 생성!
└── conversion_summary.json
```

**출력 vocab.json 내용** (ONNX/C 엔진용):
```json
{
  "1": "[<NUM>] usb 1-1: new high-speed USB device number <NUM> using ehci-pci",
  "2": "[<NUM>] CPU<ID>: Core temperature above threshold, cpu clock throttled",
  "3": "[<NUM>] CPU<ID>: Core temperature<PATH> normal"
}
```

**자동 변환 로직**:
1. `--vocab data/processed/vocab.json` 지정
2. 같은 디렉토리의 `parsed.parquet` 자동 감지
3. `parsed.parquet`에서 템플릿 문자열 추출
4. `hybrid_system/inference/models/vocab.json`을 템플릿 문자열 형식으로 생성

### Step 4: C 엔진 실행 (Inference)

```bash
cd hybrid_system/inference
make && ./bin/inference_engine \
    -d models/deeplog.onnx \
    -v models/vocab.json \        # ← 템플릿 문자열 형식
    -t
```

**사용하는 파일**:
- ✅ `models/deeplog.onnx`
- ✅ `models/vocab.json` (템플릿 문자열 형식)

## 🔄 왜 두 개의 vocab.json이 필요한가?

### Python 학습 환경

```python
# PyTorch 모델은 인덱스만 필요
vocab = {"1": 0, "2": 1, "3": 2}

# 시퀀스 데이터
sequence = [0, 1, 2, 0, 1]  # 인덱스로만 구성

# 모델 입력
input_tensor = torch.tensor([0, 1, 2])  # 인덱스
```

**이유**: PyTorch는 템플릿 ID를 숫자 인덱스로 변환하여 학습합니다. 템플릿 문자열은 필요 없습니다.

### C 추론 엔진

```c
// C 엔진은 로그 라인을 템플릿과 매칭해야 함
char* log_line = "[12345.678901] usb 1-1: new high-speed USB device...";

// 템플릿과 유사도 비교
for (i = 0; i < vocab_size; i++) {
    similarity = compare(log_line, vocab->templates[i]);
    // "usb 1-1: new high-speed" vs "[<NUM>] usb 1-1: new high-speed..."
}
```

**이유**: C 엔진은 실제 로그 라인과 템플릿을 비교하여 가장 유사한 템플릿을 찾아야 합니다. 문자열이 필수입니다.

## ❓ 자주 하는 실수

### ❌ 실수 1: Python용 vocab을 ONNX 변환 후에 수정

```bash
# 잘못된 방법
alog-detect train-deeplog ...
python model_converter.py --vocab data/processed/vocab.json ...

# vocab.json을 수동으로 수정 ❌
python export_vocab_with_templates.py \
    data/processed/parsed.parquet \
    data/processed/vocab.json  # ← Python 학습 디렉토리를 수정하면 안 됨!
```

**문제**: `data/processed/vocab.json`은 Python 학습용이므로 인덱스 형식으로 유지되어야 합니다!

**올바른 방법**:
```bash
# model_converter.py가 자동으로 올바른 위치에 생성
# data/processed/vocab.json은 그대로 둠 ✅
```

### ❌ 실수 2: ONNX vocab을 Python 학습에 사용

```bash
# 잘못된 방법
cp hybrid_system/inference/models/vocab.json data/processed/vocab.json  # ❌

alog-detect train-deeplog --parsed data/processed/parsed.parquet ...
# → 에러 발생! 템플릿 문자열을 인덱스로 변환할 수 없음
```

**올바른 방법**: 각 vocab.json을 자기 위치에 유지 ✅

### ❌ 실수 3: vocab.json을 다른 위치로 이동

```bash
# 잘못된 방법
mv data/processed/vocab.json /tmp/vocab.json

python model_converter.py --vocab /tmp/vocab.json ...
# → 자동 변환 실패! parsed.parquet를 찾을 수 없음
```

**올바른 방법**: vocab.json을 원래 위치에 유지 ✅

## ✅ 올바른 디렉토리 구조

```
project/
├── data/
│   ├── raw/
│   │   └── system.log
│   └── processed/              # Python 학습 영역
│       ├── parsed.parquet      # ✅ 템플릿 문자열 보유 (자동 변환에 사용)
│       ├── preview.json        # ✅ 템플릿 문자열 보유 (fallback)
│       ├── vocab.json          # ⚠️  인덱스 형식 (Python 학습용)
│       └── sequences.parquet
│
├── .cache/
│   └── deeplog.pth             # PyTorch 모델
│
└── hybrid_system/
    ├── training/
    │   ├── model_converter.py  # ONNX 변환 + 자동 vocab 변환
    │   └── export_vocab_with_templates.py  # 수동 변환 (거의 사용 안 함)
    └── inference/
        └── models/             # C 엔진 영역
            ├── deeplog.onnx
            └── vocab.json      # ✅ 템플릿 문자열 형식 (C 엔진용)
```

## 📝 체크리스트

### 학습 완료 후 확인

```bash
# 1. Python용 vocab 확인 (인덱스 형식이어야 함)
cat data/processed/vocab.json
# 출력: {"1": 0, "2": 1, ...}  ✅

# 2. parsed.parquet 존재 확인 (자동 변환에 필요)
ls -lh data/processed/parsed.parquet  ✅
```

### ONNX 변환 후 확인

```bash
# 1. ONNX vocab 확인 (템플릿 문자열이어야 함)
cat hybrid_system/inference/models/vocab.json
# 출력: {"1": "[<NUM>] usb ...", ...}  ✅

# 2. Python vocab은 그대로 유지 확인
cat data/processed/vocab.json
# 출력: {"1": 0, "2": 1, ...}  ✅ (변경되지 않음!)
```

### C 엔진 실행 전 확인

```bash
cd hybrid_system/inference

# 1. Vocab 로드 확인
./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -t 2>&1 | grep "Loaded vocabulary"
# 출력: Loaded vocabulary: 7 templates  ✅ (0이 아니어야 함!)
```

## 🎯 다른 시스템에서 작업할 때

### 시나리오: 학습은 서버 A, 추론은 서버 B

#### 서버 A (학습)

```bash
# 1. 파싱
alog-detect parse --input data/raw/system.log --out-dir data/processed

# 2. 학습
alog-detect train-deeplog --parsed data/processed/parsed.parquet --out-dir data/processed

# 3. ONNX 변환
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \
    --output-dir hybrid_system/inference/models

# 4. 필요한 파일만 복사 준비
tar -czf onnx_bundle.tar.gz \
    hybrid_system/inference/models/deeplog.onnx \
    hybrid_system/inference/models/vocab.json
```

#### 서버 B (추론)

```bash
# 1. 파일 전송
scp serverA:onnx_bundle.tar.gz .
tar -xzf onnx_bundle.tar.gz

# 2. C 엔진 빌드
cd hybrid_system/inference
make

# 3. 실행
./bin/inference_engine \
    -d models/deeplog.onnx \
    -v models/vocab.json \
    -i /var/log/syslog
```

**중요**: `data/processed/vocab.json`은 전송할 필요 없습니다! C 엔진은 `hybrid_system/inference/models/vocab.json`만 사용합니다.

## 📊 요약 표

| 단계 | 명령어 | vocab.json 위치 | vocab.json 형식 | 수정 여부 |
|------|--------|---------------|--------------|----------|
| Parse | `alog-detect parse` | `data/processed/` | 인덱스 생성 | 생성됨 |
| Train | `alog-detect train-deeplog` | `data/processed/` | 인덱스 (사용) | 유지 |
| Convert | `model_converter.py` | `data/processed/` (입력)<br>`hybrid_system/inference/models/` (출력) | 인덱스 (입력)<br>템플릿 문자열 (출력) | `data/processed/` 유지<br>`models/` 생성 |
| Inference | `inference_engine` | `hybrid_system/inference/models/` | 템플릿 문자열 (사용) | 유지 |

## ✅ 최종 답변

### 질문: "train.sh 수행 시에 basemodel 디렉토리의 vocab.json을 변경하는게 맞아?"

**답변**: ❌ **아니요! 변경하지 않습니다!**

- `data/processed/vocab.json`은 인덱스 형식으로 유지
- `model_converter.py`가 자동으로 `hybrid_system/inference/models/vocab.json`을 템플릿 문자열 형식으로 생성

### 질문: "basemodel 하위에 onnx를 생성하고 그 안에 vocab.json을 만들고.. convert 시에 해당 파일 위치를 넣어줘야 하는거야?"

**답변**: ❌ **아니요! 그 반대입니다!**

**올바른 순서**:
1. `--vocab data/processed/vocab.json` (인덱스 형식) 입력
2. `model_converter.py`가 자동으로 `parsed.parquet`에서 템플릿 추출
3. `--output-dir hybrid_system/inference/models/vocab.json` (템플릿 문자열) 자동 생성

## 🎉 핵심 정리

```bash
# 한 줄 요약: 이것만 하면 모든 것이 자동으로 처리됩니다!

# 1. 파싱 + 학습
alog-detect parse --input data/raw/system.log --out-dir data/processed
alog-detect train-deeplog --parsed data/processed/parsed.parquet --out-dir data/processed

# 2. ONNX 변환 (자동 vocab 변환!)
python hybrid_system/training/model_converter.py \
    --deeplog-model .cache/deeplog.pth \
    --vocab data/processed/vocab.json \          # ← 인덱스 형식 입력
    --output-dir hybrid_system/inference/models   # ← 템플릿 문자열 자동 생성!

# data/processed/vocab.json은 그대로 유지됨 ✅
# hybrid_system/inference/models/vocab.json이 자동 생성됨 ✅
```

**수동 작업 불필요!** 모든 것이 자동으로 처리됩니다! 🎉
