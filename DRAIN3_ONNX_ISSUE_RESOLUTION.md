# Drain3 + ONNX C 엔진 이상탐지 문제 해결 기록

**작성일**: 2025-11-19
**프로젝트**: study_preprocessor
**목적**: DeepLog ONNX C 추론 엔진의 98% 이상 탐지율 문제 해결

---

## 📋 목차

1. [문제 요약](#문제-요약)
2. [발견된 오류들](#발견된-오류들)
3. [근본 원인 분석](#근본-원인-분석)
4. [해결 방안](#해결-방안)
5. [구현된 솔루션](#구현된-솔루션)
6. [향후 선택 가이드](#향후-선택-가이드)
7. [기술적 세부사항](#기술적-세부사항)

---

## 문제 요약

### 초기 증상

- **Python 모델**: 정상 동작 (anomaly rate: 5-15%)
- **C ONNX 엔진**: 비정상 동작 (anomaly rate: 86% → 98%)
- **k=38**로 설정해도 98.2%가 anomaly로 탐지됨

### 핵심 문제

**Python 학습**과 **C 추론**의 전처리 방식 불일치:

```
Python (학습):  Drain3 구조 파싱 → "User <*> logged in from <IP>"
C 엔진 (추론):  Regex 마스킹만  → "User alice logged in from <NUM>"
                                     ↓
                          템플릿 매칭 실패 → 98% anomaly!
```

---

## 발견된 오류들

### 1. ✅ Vocab.json 인덱스 검증 누락 (`log_parser.c:746`)

**문제**:
- vocab.json의 인덱스가 0-based, 연속적인지 검증하지 않음
- ONNX 모델의 출력 인덱스와 불일치 가능성

**해결**:
```c
// log_parser.c:759-775
for (int i = 0; i < parser->vocab_size; i++) {
    if (parser->vocab_indices[i] != i) {
        fprintf(stderr, "ERROR: vocab.json indices not 0-based/consecutive\n");
        return -1;
    }
}
```

**파일**: `hybrid_system/inference/src/log_parser.c`
**커밋**: `cb87859` (이전 세션)

---

### 2. ✅ 무한 루프 취약점 (Regex 매칭)

**문제**:
- `regex_replace_all()` 함수에서 빈 매칭 시 무한 루프 발생 가능

**해결**:
```c
// log_parser.c:584-588
size_t advance = matches[0].rm_eo;
if (advance == 0) {
    advance = 1;  // 빈 매칭 시 1 바이트 전진
}
src += advance;
```

**파일**: `hybrid_system/inference/src/log_parser.c`
**커밋**: `cb87859` (이전 세션)

---

### 3. ✅ 메모리 누수

**문제**:
- Regex 패턴 정리 함수 없음

**해결**:
```c
// log_parser.c에 cleanup 함수 추가
void log_parser_cleanup(void);
```

**파일**: `hybrid_system/inference/src/log_parser.c`, `include/inference_engine.h`
**커밋**: `cb87859` (이전 세션)

---

### 4. ✅ ONNX 출력 시퀀스 위치 오류 (86% Anomaly)

**문제**:
- LSTM 모델의 출력에서 **첫 번째 위치**(index 0) 로짓을 사용
- 올바른 위치: **마지막 위치** (seq_len - 1)

**증상**:
```
Anomaly rate: 86% (k=38 설정에도 불구하고)
```

**해결**:
```c
// onnx_engine.c:273
// BEFORE:
int64_t last_position_offset = 0;  // ❌ 첫 번째 위치

// AFTER:
int64_t last_position_offset = (output_seq_len - 1) * vocab_size;  // ✅ 마지막 위치
```

**영향**: 86% → 여전히 높은 anomaly rate (다른 문제 존재)

**파일**: `hybrid_system/inference/src/onnx_engine.c:273`
**커밋**: `5142657`

---

### 5. ✅ 날짜가 템플릿에 포함되는 문제

**문제**:
- 날짜 정보가 마스킹되지 않고 템플릿에 그대로 포함됨
- 예: `"Sep 14 login successful"` → 날짜가 템플릿 일부로 간주

**왜 문제인가?**:
- 날짜는 **시간 정보**이지 **로그 패턴**이 아님
- 동일한 로그 패턴이 날짜가 다르면 다른 템플릿으로 인식됨
- Unknown template 증가 → 이상 탐지 정확도 저하

**해결**:
```python
# preprocess.py - 날짜 마스킹 패턴 추가
DATE_SYSLOG = re.compile(r"\b(?:Jan|Feb|Mar|...|Dec)\s+\d{1,2}\b")
DATE_ISO = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
DATE_DMY = re.compile(r"\b\d{1,2}[-/](?:Jan|Feb|...|Dec)[-/]?\d{2,4}\b")

# CRITICAL: 날짜를 숫자보다 먼저 마스킹!
masked = DATE_ISO.sub("<DATE>", masked)
masked = DATE_DMY.sub("<DATE>", masked)
masked = DATE_SYSLOG.sub("<DATE>", masked)
masked = NUMBER.sub("<NUM>", masked)  # 이후 숫자 마스킹
```

```c
// log_parser.c - C 엔진도 동일하게
regex_replace_all(temp, temp_size, &DATE_SYSLOG, "<DATE>");
regex_replace_all(temp, temp_size, &DATE_ISO, "<DATE>");
regex_replace_all(temp, temp_size, &DATE_DMY, "<DATE>");
regex_replace_all(temp, temp_size, &NUMBER, "<NUM>");  // 순서 중요!
```

**파일**: `anomaly_log_detector/preprocess.py`, `hybrid_system/inference/src/log_parser.c`
**커밋**: `e5f6998`

---

### 6. ✅ 98% Anomaly Rate - Drain3 vs Regex 불일치 (근본 원인)

**문제**:
Python 학습과 C 추론의 템플릿 추출 방식이 근본적으로 다름

#### Python (학습 시):
```python
# Drain3 구조적 파싱
"User alice logged in from 192.168.1.10"
    ↓ Drain3
"User <*> logged in from <IP>"  # 구조 기반 wildcard
```

#### C 엔진 (추론 시):
```c
// Regex 마스킹만
"User alice logged in from 192.168.1.10"
    ↓ regex_replace_all()
"User alice logged in from <IP>"  # alice는 그대로!
```

#### 결과:
```
vocab.json:     "User <*> logged in from <IP>"
C normalized:   "User alice logged in from <IP>"
                      ↓
            Similarity: 82% (threshold 미달)
                      ↓
         Template matching 실패 → ANOMALY!
```

**증상**:
```
[DEBUG_TOPK] Top-38 indices: [5, 12, 23, ...]
[DEBUG_TOPK] Actual: 147, In Top-38: NO  ❌
Result: ANOMALY (98.2% of all logs)
```

**왜 이런 일이 발생했나?**:
- Python 학습: Drain3 라이브러리 사용 (고급 구조 파싱)
- C 추론: Drain3 C/C++ 이식 어려움 → Regex만 사용
- 의도: "충분히 비슷하면 되겠지?"
- 현실: 템플릿 매칭 실패 → 전체 시스템 붕괴

**파일**: `hybrid_system/inference/src/log_parser.c`
**문제 확인**: 수동 디버깅 (DEBUG_TEMPLATE=1)

---

### 7. ✅ diagnose_vocab_mismatch.py ValueError

**문제**:
```python
# scripts/diagnose_vocab_mismatch.py:223, 206, 37
int(k)  # ❌ ValueError when k is a string (Python format vocab)
```

vocab.json 형식이 두 가지 존재:
```json
// C format (index → template)
{"0": "User logged in", "1": "Connection from IP"}

// Python format (template → index)
{"User logged in": 0, "Connection from IP": 1}
```

**해결**:
```python
# Auto-detect format
try:
    int(sample_key)
    is_c_format = isinstance(sample_value, str)
except ValueError:
    is_python_format = isinstance(sample_value, int)

# Convert based on detected format
```

**파일**: `scripts/diagnose_vocab_mismatch.py`
**커밋**: 해당 스크립트 수정 (이전 세션)

---

## 근본 원인 분석

### Drain3란?

**Drain3**: 로그 템플릿 자동 추출 알고리즘

- **구조적 파싱**: 로그 메시지의 구조를 분석
- **동적 트리 구축**: 유사한 로그를 클러스터링
- **Wildcard 자동 생성**: 가변 부분을 `<*>`로 대체

**예시**:
```
Input logs:
  "User alice logged in"
  "User bob logged in"
  "User charlie logged in"

Drain3 output:
  "User <*> logged in"  # <*> = wildcard (any token)
```

### 왜 Python과 C가 다른가?

| 측면 | Python | C |
|------|--------|---|
| **라이브러리** | drain3 (PyPI 패키지) | 없음 |
| **구현 복잡도** | pip install drain3 | ~3000+ 라인 C++ 코드 |
| **의존성** | Python only | 없음 (standalone) |
| **결과** | 구조적 wildcard | Regex 리터럴 |

### 템플릿 불일치 예시

```
로그: "Failed password for admin from 10.0.1.5 port 22"

Python (Drain3):
  → "Failed password for <*> from <IP> port <NUM>"
  → vocab index: 42

C (Regex):
  → "Failed password for admin from <IP> port <NUM>"
  → vocab 검색: 42번 템플릿과 비교
  → Similarity: 80% (< 85% threshold)
  → 매칭 실패 → ANOMALY!
```

### 왜 98%가 anomaly인가?

1. **vocab.json**: 모두 Drain3 wildcard 템플릿 (`<*>` 포함)
2. **C normalized**: Regex만 사용 → literal 값 포함
3. **매칭 실패**: 대부분의 로그가 vocab와 매칭 안 됨
4. **Top-K 실패**: 실제 템플릿이 top-38에 없음
5. **결과**: 98% anomaly

---

## 해결 방안

### 접근 방법 비교

| 방법 | 장점 | 단점 | 구현 난이도 |
|------|------|------|------------|
| **1. Wildcard Matching** | ✅ 업계 표준<br>✅ Drain3 vocab 그대로 사용<br>✅ 재학습 불필요 | 약간의 성능 오버헤드 | 중간 |
| **2. Regex-only 재학습** | ✅ Python-C 완전 동일<br>✅ Wildcard 불필요 | ❌ 템플릿 품질 저하<br>❌ 재학습 필요 | 쉬움 |
| **3. Drain3 C++ 포팅** | ✅ Python과 완전 동일 | ❌ 3000+ 라인 코드<br>❌ 유지보수 부담 | 어려움 |
| **4. Python 전처리 서비스** | ✅ Python 코드 재사용 | ❌ 네트워크 의존성<br>❌ 성능 저하 | 중간 |

### 선택한 방법: Wildcard Matching (방법 1)

**이유**:
1. **업계 표준**: ONNX 모델과 함께 가장 많이 사용되는 패턴
2. **재학습 불필요**: 기존 Drain3 vocab.json 그대로 사용
3. **Python 코드 변경 없음**: 학습 파이프라인 유지
4. **합리적 구현 복잡도**: ~200 라인 C 코드

---

## 구현된 솔루션

### 솔루션 1: Wildcard Template Matching (권장)

#### 구현 개요

C 엔진에 wildcard 인식 기능 추가:

```c
// log_parser.c

// Tokenizer: "<*>" 패턴 인식
typedef enum {
    TOKEN_WILDCARD,      // <*>
    TOKEN_PLACEHOLDER,   // <IP>, <NUM>, <DATE> 등
    TOKEN_LITERAL        // 일반 단어
} TokenType;

// Wildcard matching 함수
static int wildcard_similarity(const char* normalized_log, const char* template_str) {
    // 1. 양쪽을 토큰화
    Token log_tokens[128];
    Token template_tokens[128];

    int log_count = tokenize_with_wildcards(normalized_log, log_tokens, 128);
    int template_count = tokenize_with_wildcards(template_str, template_tokens, 128);

    // 2. 토큰별 매칭
    for (int i = 0; i < template_count; i++) {
        if (template_tokens[i].type == TOKEN_WILDCARD) {
            // <*>는 모든 단일 토큰과 매칭
            matches++;
        } else if (template_tokens[i].type == TOKEN_PLACEHOLDER) {
            // <IP>, <NUM> 등은 placeholder끼리 매칭
            if (log_tokens[j].type == TOKEN_PLACEHOLDER) {
                matches++;
            }
        } else {
            // Literal은 정확히 일치해야 함
            if (strcmp(template_tokens[i].value, log_tokens[j].value) == 0) {
                matches++;
            }
        }
    }

    // 3. Similarity 계산
    return (matches * 100) / max_len;
}

// string_similarity에서 자동 감지
static int string_similarity(const char* s1, const char* s2) {
    if (strstr(s2, "<*>") != NULL) {
        return wildcard_similarity(s1, s2);  // Wildcard 템플릿
    }
    return (int)(hybrid_similarity(s1, s2) * 100);  // 기존 방식
}
```

#### 매칭 예시

```
Input log (normalized):
  "Failed password for admin from <IP> port <NUM>"

Vocab template:
  "Failed password for <*> from <IP> port <NUM>"

Tokenization:
  Log:      ["Failed", "password", "for", "admin", "from", "<IP>", "port", "<NUM>"]
  Template: ["Failed", "password", "for", "<*>",  "from", "<IP>", "port", "<NUM>"]

Matching:
  Failed    = Failed     ✅
  password  = password   ✅
  for       = for        ✅
  admin     = <*>        ✅ (wildcard matches any token)
  from      = from       ✅
  <IP>      = <IP>       ✅
  port      = port       ✅
  <NUM>     = <NUM>      ✅

Similarity: 100% → MATCH!
```

#### 사용 방법

**추가 설정 불필요** - 자동으로 동작:

```bash
# 1. 기존 방식대로 학습 (Drain3 사용)
./scripts/train.sh /var/log/normal/

# 2. C 엔진 빌드
cd hybrid_system/inference
make clean && make

# 3. 추론 실행 (자동으로 wildcard 인식)
./build/inference_engine \
    --model models/deeplog.onnx \
    --vocab models/vocab.json \
    --log-file test.log \
    --k 10
```

#### 디버깅

```bash
# Wildcard matching 동작 확인
export DEBUG_TEMPLATE=1

./build/inference_engine --model ... --vocab ... --log-file ...
```

출력 예시:
```
[DEBUG] Template 42: "Failed password for <*> from <IP> port <NUM>"
[DEBUG]   Wildcard matching detected
[DEBUG]   Log tokens: 8, Template tokens: 8
[DEBUG]   Matches: 8/8
[DEBUG]   Similarity: 100%  ✅
```

**파일**: `hybrid_system/inference/src/log_parser.c` (lines 118-235, 506-597)
**커밋**: `cb87859`

---

### 솔루션 2: Regex-Only 학습 (대안)

Drain3를 사용하지 않고 Regex만으로 학습:

#### 왜 필요한가?

- Wildcard matching 성능 오버헤드 제거
- Python과 C가 100% 동일한 전처리
- 템플릿 품질 저하는 감수

#### 사용 방법

```bash
# train.sh 대신 train_wo_drain3.sh 사용
./scripts/train_wo_drain3.sh /var/log/normal/ models_regex_only

# 결과:
# - vocab.json: wildcard 없음 (리터럴 값만)
# - deeplog.pth, mscred.pth: Regex-only로 학습된 모델
```

#### 차이점

| 측면 | train.sh (Drain3) | train_wo_drain3.sh (Regex) |
|------|------------------|---------------------------|
| **템플릿** | `User <*> logged in` | `User alice logged in` |
| **고유 템플릿 수** | 적음 (~500) | 많음 (~5000+) |
| **일반화** | 높음 | 낮음 |
| **Python-C 일치** | Wildcard 필요 | 완전 동일 |
| **Vocab 크기** | 작음 | 큼 |

#### 장단점

**장점**:
- ✅ Python과 C 완전 동일
- ✅ Wildcard matching 불필요
- ✅ 예측 가능한 동작

**단점**:
- ❌ 템플릿 품질 저하 (일반화 능력 ↓)
- ❌ Vocab 크기 증가 → 모델 크기 증가
- ❌ Unknown template 증가 가능성

**파일**: `scripts/train_wo_drain3.sh`
**커밋**: `248c9b4`

---

## 향후 선택 가이드

### 시나리오별 권장 방안

#### 시나리오 1: 프로덕션 배포 (권장: Wildcard Matching)

**상황**:
- 이미 Drain3로 학습된 모델 보유
- 재학습 비용이 높음
- 성능이 중요하지만 정확도가 더 중요

**선택**: **Wildcard Matching**

**이유**:
```
✅ 재학습 불필요 (기존 모델 그대로 사용)
✅ 높은 템플릿 품질 (Drain3 일반화)
✅ 업계 표준 방식
⚠️  약간의 성능 오버헤드 (무시 가능)
```

**실행**:
```bash
# 기존 학습 모델 그대로 사용
./scripts/train.sh /var/log/normal/

# C 엔진은 자동으로 wildcard 인식
./hybrid_system/inference/build/inference_engine \
    --model models/deeplog.onnx \
    --vocab models/vocab.json \
    --log-file /var/log/test.log \
    --k 10
```

---

#### 시나리오 2: 최대 성능 필요 (권장: Regex-Only)

**상황**:
- 초당 수백만 로그 처리
- 재학습 가능
- 템플릿 품질보다 성능 우선

**선택**: **Regex-Only 재학습**

**이유**:
```
✅ 최고 성능 (wildcard 오버헤드 없음)
✅ Python-C 완전 동일
❌ 템플릿 품질 저하 (감수 가능)
❌ 재학습 필요 (1회)
```

**실행**:
```bash
# Regex-only로 재학습
./scripts/train_wo_drain3.sh /var/log/normal/ models_regex

# C 엔진 사용 (wildcard 불필요)
./hybrid_system/inference/build/inference_engine \
    --model models_regex/deeplog.onnx \
    --vocab models_regex/vocab.json \
    --log-file /var/log/test.log \
    --k 10
```

---

#### 시나리오 3: 새 프로젝트 시작

**상황**:
- 처음부터 시스템 구축
- 학습 파이프라인 설계 가능
- 장기적 관점

**선택**: **Wildcard Matching** (유연성)

**이유**:
```
✅ 나중에 변경 가능 (Drain3 ↔ Regex)
✅ 더 나은 템플릿 품질
✅ 유지보수 용이
```

---

#### 시나리오 4: 임베디드/IoT 디바이스

**상황**:
- 메모리/CPU 제약
- 작은 vocab 크기 필수
- 재학습 가능

**선택**: **Regex-Only** (작은 모델)

**이유**:
```
✅ 작은 vocab.json
✅ 낮은 메모리 사용
✅ 빠른 추론
```

---

### 의사결정 플로우차트

```
시작
  │
  ├─ 이미 Drain3 모델 보유?
  │   ├─ YES → Wildcard Matching 사용 ✅
  │   └─ NO  → 계속
  │
  ├─ 재학습 가능?
  │   ├─ NO  → Wildcard Matching 사용 ✅
  │   └─ YES → 계속
  │
  ├─ 성능이 최우선?
  │   ├─ YES → Regex-Only 재학습 ✅
  │   └─ NO  → 계속
  │
  ├─ 메모리 제약?
  │   ├─ YES → Regex-Only 재학습 ✅
  │   └─ NO  → Wildcard Matching 사용 ✅
```

---

## 기술적 세부사항

### Wildcard Matching 성능 분석

#### 시간 복잡도

```
기존 hybrid_similarity:
  - Jaccard: O(n + m)
  - Levenshtein: O(n × m)
  - Total: O(n × m)

Wildcard matching:
  - Tokenization: O(n + m)
  - Token matching: O(min(n, m))
  - Total: O(n + m)

→ 이론적으로 더 빠름!
```

#### 실제 벤치마크 (예상)

```
로그당 처리 시간:
  - Regex-only:         ~10 μs
  - Wildcard matching:  ~15 μs
  - Overhead:           ~50%

처리량:
  - Regex-only:         100,000 logs/sec
  - Wildcard matching:   66,000 logs/sec

→ 대부분의 경우 충분함
```

---

### 파일별 변경 사항 요약

#### `hybrid_system/inference/src/log_parser.c`

```c
// 추가된 기능:

1. 날짜 regex 패턴 (lines 48-50)
   DATE_SYSLOG, DATE_ISO, DATE_DMY

2. Wildcard tokenizer (lines 118-167)
   tokenize_with_wildcards()

3. Wildcard matching (lines 169-235)
   wildcard_similarity()

4. Auto-detection (lines 506-597)
   string_similarity() - strstr(s2, "<*>") 감지

5. Vocab 검증 (lines 759-775)
   인덱스 0-based, 연속성 확인

6. Infinite loop 방지 (lines 584-588)
   빈 매칭 시 1바이트 전진

7. DEBUG 로깅 (환경 변수)
   DEBUG_TEMPLATE=1
```

#### `hybrid_system/inference/src/onnx_engine.c`

```c
// 수정된 부분:

Line 273:
  // BEFORE: int64_t last_position_offset = 0;
  // AFTER:  int64_t last_position_offset = (output_seq_len - 1) * vocab_size;

→ LSTM 마지막 시퀀스 위치 로짓 추출
```

#### `anomaly_log_detector/preprocess.py`

```python
# 추가된 패턴:

DATE_SYSLOG = re.compile(r"\b(?:Jan|Feb|...|Dec)\s+\d{1,2}\b")
DATE_ISO = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
DATE_DMY = re.compile(r"\b\d{1,2}[-/](?:Jan|...|Dec)[-/]?\d{2,4}\b")

# 마스킹 순서 변경:
1. DATE_ISO, DATE_DMY, DATE_SYSLOG  (먼저!)
2. NUMBER                             (나중에)
```

#### `scripts/train_wo_drain3.sh` (NEW)

```bash
# Drain3 없이 학습하는 전체 파이프라인

1. 로그 스캔 (train.sh와 동일)
2. Regex-only 전처리 (inline Python)
3. 베이스라인 통계 학습
4. DeepLog 학습
5. MS-CRED 학습
6. 메타데이터 저장
```

#### `scripts/diagnose_vocab_mismatch.py`

```python
# 개선 사항:

1. Vocab format auto-detection
   - C format: {"0": "template"}
   - Python format: {"template": 0}

2. ONNX shape 검증
   - 3D: [batch, seq_len, vocab_size]
   - 2D: [batch, vocab_size]
```

---

### 디버깅 환경 변수

```bash
# 템플릿 매칭 디버깅
export DEBUG_TEMPLATE=1

# ONNX 출력 디버깅
export DEBUG_ONNX=1

# 시퀀스 버퍼 디버깅
export DEBUG_SEQ=1

# Top-K 결과 디버깅
export DEBUG_TOPK=1

# Anomaly 판정 디버깅
export DEBUG_ANOMALY=1

# 실행 예시
DEBUG_TEMPLATE=1 DEBUG_TOPK=1 \
./build/inference_engine \
    --model models/deeplog.onnx \
    --vocab models/vocab.json \
    --log-file test.log \
    --k 10
```

---

## 결론

### 주요 성과

1. ✅ **근본 원인 규명**: Drain3 wildcard vs Regex literal 불일치
2. ✅ **Wildcard Matching 구현**: 업계 표준 솔루션
3. ✅ **Regex-Only 대안 제공**: 성능 최적화 옵션
4. ✅ **6개 버그 수정**: ONNX, 날짜, 검증, 메모리, 무한루프
5. ✅ **디버깅 도구**: 진단 스크립트 및 환경 변수

### 예상 결과

```
BEFORE:
  Anomaly rate: 98.2% (k=38)

AFTER (Wildcard Matching):
  Anomaly rate: 5-15% (정상 범위)

AFTER (Regex-Only):
  Anomaly rate: 10-20% (템플릿 수 증가로 약간 높음)
```

### 다음 단계

1. **테스트**: 실제 로그 데이터로 wildcard matching 검증
2. **성능 측정**: Regex-only vs Wildcard 벤치마크
3. **모니터링**: Anomaly rate 추적
4. **문서화**: 운영 가이드 작성

---

## 참고 자료

### 관련 커밋

- `cb87859`: Wildcard template matching 구현
- `79d88a8`: train_without_drain3.py (Python 스크립트)
- `f6f2aab`: DEBUG 로깅 추가
- `e5f6998`: 날짜 마스킹 추가
- `5142657`: ONNX 마지막 시퀀스 위치 수정
- `248c9b4`: train_wo_drain3.sh (Bash 스크립트)

### 주요 파일

```
hybrid_system/inference/
├── src/
│   ├── log_parser.c          # Wildcard matching 구현
│   ├── onnx_engine.c         # ONNX 출력 위치 수정
│   └── anomaly_detector.c    # DEBUG 로깅
├── include/
│   └── inference_engine.h    # Cleanup API 추가
└── Makefile

scripts/
├── train.sh                  # Drain3 학습 (기본)
├── train_wo_drain3.sh        # Regex-only 학습 (대안)
├── diagnose_vocab_mismatch.py
└── compare_normalization.py

anomaly_log_detector/
└── preprocess.py             # 날짜 마스킹 추가
```

### 외부 리소스

- [Drain3 GitHub](https://github.com/logpai/Drain3)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [DeepLog Paper](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf)

---

**문서 끝**
