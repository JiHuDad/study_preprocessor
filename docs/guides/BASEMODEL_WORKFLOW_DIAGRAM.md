# Basemodel 워크플로우 다이어그램

## 🎯 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                         1. Parse 단계                            │
└─────────────────────────────────────────────────────────────────┘

Input: data/raw/system.log
   │
   │ alog-detect parse --out-dir basemodel
   ▼
basemodel/
├── parsed.parquet          ✅ 템플릿 문자열 포함
├── preview.json            ✅ 템플릿 문자열 포함
├── vocab.json              ⚠️  인덱스: {"1": 0, "2": 1, ...}
└── sequences.parquet


┌─────────────────────────────────────────────────────────────────┐
│                         2. Train 단계                            │
└─────────────────────────────────────────────────────────────────┘

basemodel/
├── parsed.parquet
├── vocab.json              ⚠️  인덱스 (사용)
└── sequences.parquet
   │
   │ alog-detect train-deeplog
   │   --parsed basemodel/parsed.parquet
   │   --out-dir basemodel
   ▼
basemodel/
├── parsed.parquet
├── preview.json
├── vocab.json              ⚠️  인덱스 (그대로 유지!)
├── sequences.parquet
└── deeplog.pth             ✅ 새로 생성!


┌─────────────────────────────────────────────────────────────────┐
│                    3. ONNX Convert 단계                          │
└─────────────────────────────────────────────────────────────────┘

basemodel/
├── deeplog.pth             (입력)
├── vocab.json              (입력: 인덱스)
└── parsed.parquet          (자동 감지!)
   │
   │ python model_converter.py
   │   --deeplog-model basemodel/deeplog.pth
   │   --vocab basemodel/vocab.json          ← 인덱스 형식
   │   --output-dir basemodel/onnx
   │
   │ [자동 처리]
   │ 1. basemodel/vocab.json 읽기 (인덱스)
   │ 2. basemodel/parsed.parquet 자동 감지
   │ 3. parsed.parquet에서 템플릿 추출
   │ 4. onnx/vocab.json 생성 (템플릿 문자열)
   ▼
basemodel/
├── parsed.parquet
├── preview.json
├── vocab.json              ⚠️  인덱스 (그대로 유지!)
├── sequences.parquet
├── deeplog.pth
│
└── onnx/                   ✅ 새로 생성!
    ├── deeplog.onnx
    ├── deeplog_optimized.onnx
    ├── vocab.json          ✅ 템플릿 문자열: {"1": "[<NUM>] usb...", ...}
    ├── deeplog.onnx.meta.json
    └── conversion_summary.json


┌─────────────────────────────────────────────────────────────────┐
│                      4. C Engine 실행                            │
└─────────────────────────────────────────────────────────────────┘

basemodel/onnx/
├── deeplog.onnx            (사용)
└── vocab.json              (사용: 템플릿 문자열)
   │
   │ ./bin/inference_engine
   │   -d basemodel/onnx/deeplog.onnx
   │   -v basemodel/onnx/vocab.json
   ▼
Output: Anomaly detection results
```

## 📊 vocab.json 변환 과정

```
┌─────────────────────────────────────────────────────────────────┐
│                  Vocab 자동 변환 상세 흐름                        │
└─────────────────────────────────────────────────────────────────┘

[입력]
--vocab basemodel/vocab.json
        │
        │ Path.parent 추출
        ▼
vocab_dir = "basemodel/"
        │
        │ 같은 디렉토리에서 파일 찾기
        ▼
┌───────────────────────────────────────┐
│  basemodel/parsed.parquet 존재?       │
└───────────────────────────────────────┘
        │
        │ Yes ✅
        ▼
┌───────────────────────────────────────┐
│  parsed.parquet 읽기                  │
│  - template_id 컬럼                   │
│  - template 컬럼                      │
└───────────────────────────────────────┘
        │
        │ 매핑 생성
        ▼
template_map = {
    "1": "[<NUM>] usb 1-1: new high-speed USB device...",
    "2": "[<NUM>] CPU<ID>: Core temperature above...",
    ...
}
        │
        │ 출력 디렉토리에 저장
        ▼
[출력]
basemodel/onnx/vocab.json (템플릿 문자열 형식)

┌───────────────────────────────────────┐
│  ✅ 자동 변환 성공!                    │
│  "✅ 7개 템플릿 추출 완료"             │
└───────────────────────────────────────┘
```

## 🔄 두 vocab.json 비교

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python 학습 영역                             │
└─────────────────────────────────────────────────────────────────┘

basemodel/vocab.json                    (인덱스 형식)
┌─────────────────────────────────────────────────────────────────┐
│ {                                                                │
│   "1": 0,         ← template_id → index                         │
│   "2": 1,                                                        │
│   "3": 2,                                                        │
│   "4": 3,                                                        │
│   "5": 4,                                                        │
│   "6": 5,                                                        │
│   "7": 6                                                         │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
        │
        │ 사용: PyTorch 학습
        │ 수정: ❌ 절대 수정 안 됨
        ▼
    [DeepLog LSTM]
    input: [0, 1, 2, ...]  (인덱스)

                    ⚠️  서로 다른 용도!

┌─────────────────────────────────────────────────────────────────┐
│                    ONNX/C 엔진 영역                              │
└─────────────────────────────────────────────────────────────────┘

basemodel/onnx/vocab.json              (템플릿 문자열 형식)
┌─────────────────────────────────────────────────────────────────┐
│ {                                                                │
│   "1": "[<NUM>] usb 1-1: new high-speed USB device number...",  │
│   "2": "[<NUM>] CPU<ID>: Core temperature above threshold...",  │
│   "3": "[<NUM>] CPU<ID>: Core temperature<PATH> normal",        │
│   "4": "[<NUM>] eth<ID>: Link is Up - 1000Mbps...",            │
│   "5": "[<NUM>] EXT<ID>-fs (sda<ID>): mounted filesystem...",  │
│   "6": "[<NUM>] audit: type=<NUM> audit(<NUM>:<NUM>): ...",    │
│   "7": "[<NUM>] usb 1-1: USB disconnect, device number <NUM>"  │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
        │
        │ 사용: C inference engine
        │ 생성: ✅ model_converter.py가 자동 생성
        ▼
    [C Engine]
    로그 라인 ←→ 템플릿 문자열 유사도 비교
```

## 🎯 핵심 포인트

```
┌─────────────────────────────────────────────────────────────────┐
│                      ⭐ 핵심 원칙                                │
└─────────────────────────────────────────────────────────────────┘

1. basemodel/vocab.json (인덱스)
   ├─ 생성: Parse 단계
   ├─ 사용: Train 단계
   ├─ 입력: Convert 단계
   └─ 수정: ❌ 절대 안 됨! 그대로 유지!

2. basemodel/onnx/vocab.json (템플릿 문자열)
   ├─ 생성: Convert 단계 (자동!)
   ├─ 사용: C Engine
   └─ 원본: basemodel/parsed.parquet에서 자동 추출

3. basemodel/parsed.parquet
   ├─ 생성: Parse 단계
   └─ 역할: 자동 변환의 핵심! 템플릿 문자열 소스

┌─────────────────────────────────────────────────────────────────┐
│                  ⚠️  하지 말아야 할 것                           │
└─────────────────────────────────────────────────────────────────┘

❌ basemodel/vocab.json 수정
❌ basemodel/vocab.json 삭제
❌ basemodel/vocab.json을 다른 위치로 이동
❌ basemodel/onnx/vocab.json을 basemodel/에 복사

┌─────────────────────────────────────────────────────────────────┐
│                    ✅ 해야 할 것                                 │
└─────────────────────────────────────────────────────────────────┘

✅ Parse → basemodel/
✅ Train → basemodel/
✅ mkdir basemodel/onnx
✅ Convert → basemodel/onnx/
✅ 그대로 두기! (자동 처리됨)
```

## 🚀 원라이너 명령어

```bash
# 전체 프로세스를 한 번에!
mkdir -p basemodel/onnx && \
alog-detect parse --input data/raw/system.log --out-dir basemodel && \
alog-detect train-deeplog --parsed basemodel/parsed.parquet --out-dir basemodel && \
python hybrid_system/training/model_converter.py \
    --deeplog-model basemodel/deeplog.pth \
    --vocab basemodel/vocab.json \
    --output-dir basemodel/onnx

# 결과 확인
echo "=== Python용 vocab (인덱스) ===" && cat basemodel/vocab.json
echo "=== ONNX용 vocab (템플릿 문자열) ===" && cat basemodel/onnx/vocab.json | head -3
```

## 📋 체크리스트

```
Parse 완료 후:
  ✅ basemodel/parsed.parquet 생성됨
  ✅ basemodel/vocab.json 생성됨 (인덱스 형식)
  ✅ basemodel/preview.json 생성됨

Train 완료 후:
  ✅ basemodel/deeplog.pth 생성됨
  ✅ basemodel/vocab.json 그대로 유지됨 (인덱스 형식)

Convert 완료 후:
  ✅ basemodel/onnx/deeplog.onnx 생성됨
  ✅ basemodel/onnx/vocab.json 생성됨 (템플릿 문자열 형식)
  ✅ basemodel/vocab.json 그대로 유지됨 (인덱스 형식)
  ✅ 로그에 "✅ 7개 템플릿 추출 완료" 표시됨

C Engine 실행:
  ✅ "Loaded vocabulary: N templates" (N > 0)
  ✅ 정상적으로 추론 실행
```
