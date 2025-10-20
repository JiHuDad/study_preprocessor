# C Inference Engine 템플릿 매칭 문제 해결

## 🐛 문제

ONNX inference engine에서 77개 템플릿이 정상적으로 로드되었지만, 모든 로그가 "Unknown template"로 처리됩니다.

### 증상

```bash
$ ./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -t

Loaded vocabulary: 77 templates  ✅
Processing test logs:
Log 1: Unknown template  ❌
Log 2: Unknown template  ❌
Log 3: Unknown template  ❌
...
```

## 🔍 원인 분석

### 1. 템플릿 형식

**vocab.json의 템플릿**:
```
[<NUM>] usb 1-1: new high-speed USB device number <NUM> using ehci-pci
```

**실제 로그**:
```
[12345.678901] usb 1-1: new high-speed USB device number 2 using ehci-pci
```

### 2. 현재 유사도 계산 로직 (문제)

```c
// log_parser.c의 string_similarity()
static int string_similarity(const char* s1, const char* s2) {
    // 앞부분부터 같은 문자 개수 세기
    for (int i = 0; i < min_len; i++) {
        if (s1[i] == s2[i]) {
            common++;
        } else {
            break;  // 다른 문자가 나오면 중단!
        }
    }
    return common;
}
```

**문제**:
```
로그:      [12345.678901] usb...
템플릿:    [<NUM>] usb...
           ↑
           '[' 다음 문자가 다름 → 유사도 = 1 (매우 낮음!)
```

### 3. 마스킹 로직 (불완전)

현재 `normalize_log_line()`은 정규표현식으로 마스킹하지만:
- 정규식이 복잡하고 느림
- Python의 마스킹과 다르게 동작
- 템플릿과 정확히 매칭되지 않음

## ✅ 해결 방법

### Option 1: Python 마스킹 사용 (권장!)

**아이디어**: C 엔진에서 복잡한 마스킹을 하지 말고, 이미 마스킹된 로그를 입력받기

#### 워크플로우 변경

**현재 (문제)**:
```
Raw 로그 → C 엔진 (마스킹 + 템플릿 매칭) → 추론
```

**개선안**:
```
Raw 로그 → Python 파서 (마스킹) → 마스킹된 로그 → C 엔진 (템플릿 매칭) → 추론
```

#### 구현

**1. Python 전처리 스크립트**:

```python
# scripts/preprocess_for_c_engine.py
import sys
from anomaly_log_detector.preprocess import mask_message

def preprocess_log_for_c_engine(input_file, output_file):
    """로그를 C 엔진용으로 전처리 (마스킹만)"""
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # Python의 mask_message 사용
            masked = mask_message(line.strip())
            f_out.write(masked + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_for_c_engine.py <input> <output>")
        sys.exit(1)

    preprocess_log_for_c_engine(sys.argv[1], sys.argv[2])
```

**2. C 엔진 사용**:

```bash
# 1. Python으로 마스킹
python scripts/preprocess_for_c_engine.py \
    /var/log/syslog \
    /tmp/masked.log

# 2. C 엔진으로 추론
./bin/inference_engine \
    -d models/deeplog.onnx \
    -v models/vocab.json \
    -i /tmp/masked.log
```

### Option 2: C 엔진의 유사도 계산 개선

더 나은 유사도 알고리즘 구현:

```c
// 개선된 string_similarity()
static int string_similarity(const char* s1, const char* s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);

    if (len1 == 0) return 0;
    if (len2 == 0) return 0;

    // Levenshtein distance 또는 토큰 기반 유사도
    // 예: 공백으로 분리된 토큰 비교

    int matches = 0;
    int total = 0;

    // 간단한 토큰 기반 비교
    char* s1_copy = strdup(s1);
    char* s2_copy = strdup(s2);

    char* token1 = strtok(s1_copy, " ");
    while (token1) {
        total++;
        char* token2 = strtok(s2_copy, " ");
        while (token2) {
            if (strcmp(token1, token2) == 0) {
                matches++;
                break;
            }
            token2 = strtok(NULL, " ");
        }
        token1 = strtok(NULL, " ");
    }

    free(s1_copy);
    free(s2_copy);

    return matches;
}
```

### Option 3: Template ID 직접 전달 (가장 빠름!)

**아이디어**: Python에서 이미 template_id를 계산했다면, 그것을 C 엔진에 전달

#### 워크플로우

```
Raw 로그 → Python 파서 (파싱 + template_id 계산) → JSON → C 엔진 (template_id 사용) → 추론
```

**장점**:
- C 엔진에서 템플릿 매칭 불필요
- 가장 빠르고 정확
- Python과 동일한 결과 보장

## 🎯 권장 해결책: Hybrid 접근

### Step 1: Python으로 전처리 (정확성)

```python
# scripts/prepare_for_onnx_inference.py
import pandas as pd
import json
from anomaly_log_detector.preprocess import parse_and_mask_logs

def prepare_logs_for_onnx(input_log, output_json, vocab_json):
    """로그를 ONNX 추론용으로 준비"""

    # 1. 로그 파싱 (Python의 drain3 사용)
    df = parse_and_mask_logs(input_log)

    # 2. vocab 로드
    with open(vocab_json) as f:
        vocab = json.load(f)

    # 3. template_id를 인덱스로 변환
    reverse_vocab = {str(v): int(k) for k, v in vocab.items()}
    df['template_index'] = df['template_id'].astype(str).map(reverse_vocab)

    # 4. JSON으로 저장
    result = {
        'logs': df[['line_no', 'timestamp', 'template_index', 'masked']].to_dict('records')
    }

    with open(output_json, 'w') as f:
        json.dump(result, f)

    print(f"✅ Prepared {len(df)} logs for ONNX inference")
    return df

if __name__ == "__main__":
    prepare_logs_for_onnx(
        'data/raw/system.log',
        'data/prepared.json',
        'basemodel/training_workspace/vocab.json'
    )
```

### Step 2: C 엔진 수정 (JSON 입력 지원)

```c
// C 엔진이 JSON을 읽고 template_index를 직접 사용
int process_prepared_json(InferenceEngine* engine, const char* json_path) {
    // JSON 파싱
    // logs 배열에서 template_index 추출
    // ONNX 모델에 template_index 시퀀스 입력
}
```

### Step 3: 통합 사용

```bash
# 1. Python으로 전처리 (정확한 template_id 계산)
python scripts/prepare_for_onnx_inference.py \
    --input data/raw/system.log \
    --vocab basemodel/training_workspace/vocab.json \
    --output data/prepared.json

# 2. C 엔진으로 고속 추론
./bin/inference_engine \
    -d models/deeplog.onnx \
    --prepared data/prepared.json \
    -o results.json
```

## 📊 비교표

| 방법 | 정확도 | 속도 | 구현 복잡도 | 권장도 |
|------|--------|------|------------|--------|
| Option 1: Python 마스킹 | 높음 | 중간 | 낮음 | ⭐⭐⭐⭐ |
| Option 2: C 유사도 개선 | 중간 | 빠름 | 높음 | ⭐⭐⭐ |
| Option 3: Template ID 전달 | 최고 | 매우 빠름 | 중간 | ⭐⭐⭐⭐⭐ |

## 🚀 임시 해결책 (지금 당장!)

C 엔진의 임계값을 낮추기:

```c
// log_parser.c의 108-115번 라인 수정

// 변경 전
if (best_similarity < template_len * 0.5) {  // 50% 임계값
    return -1;
}

// 변경 후
if (best_similarity < 5) {  // 최소 5글자만 일치하면 OK
    return -1;
}

// 또는 임계값 완전 제거 (테스트용)
// return best_template_id;  // 가장 유사한 것 무조건 반환
```

**재컴파일**:
```bash
cd hybrid_system/inference
make clean && make
./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -t
```

## ✅ 최종 권장사항

**Option 3 (Template ID 전달) + Option 1 (Python 마스킹) 조합**:

1. **단기**: 임계값 낮추기 (임시 해결)
2. **중기**: Python 마스킹 스크립트 추가
3. **장기**: Template ID를 포함한 JSON 입력 지원

### 구현 우선순위

1. ✅ **즉시**: 임계값 조정 (5분)
2. 📋 **1주일 내**: Python 전처리 스크립트 추가
3. 📋 **2주일 내**: C 엔진 JSON 입력 지원

## 🔗 관련 파일

- `hybrid_system/inference/src/log_parser.c` (108-115번 라인)
- `hybrid_system/inference/src/anomaly_detector.c`
- `anomaly_log_detector/preprocess.py` (mask_message 함수)

---

**업데이트**: 2025-10-17
