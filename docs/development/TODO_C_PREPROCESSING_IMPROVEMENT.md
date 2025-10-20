# TODO: C 전처리 알고리즘 개선

## 🎯 목표

실시간 로그 이상탐지를 위해 C inference engine의 로그 전처리 및 템플릿 매칭 알고리즘을 개선합니다.

## 📋 현재 상태

### 문제점

1. **유사도 계산이 너무 단순함**
   - 현재: 앞부분부터 같은 문자만 세는 prefix matching
   - 문제: 첫 글자만 달라도 유사도가 0에 가까움
   - 위치: `hybrid_system/inference/src/log_parser.c:26-46`

2. **마스킹 로직이 불완전함**
   - 현재: 정규표현식 기반 마스킹
   - 문제: Python의 `mask_message()`와 다르게 동작
   - 위치: `hybrid_system/inference/src/log_parser.c:49-83`

3. **임시 해결책 적용됨**
   - 임계값을 5글자로 낮춤 (108-119번 라인)
   - 부정확할 수 있음

### 현재 동작

```c
// log_parser.c
static int string_similarity(const char* s1, const char* s2) {
    // 앞부분부터 같은 문자 개수 세기
    for (int i = 0; i < min_len; i++) {
        if (s1[i] == s2[i]) {
            common++;
        } else {
            break;  // ❌ 첫 다른 문자에서 중단
        }
    }
    return common;
}
```

**결과**:
```
로그:      [12345.678901] usb...
템플릿:    [<NUM>] usb...
           ↑ similarity = 1 (매우 낮음!)
```

## ✅ 개선 계획

### Phase 1: 더 나은 유사도 알고리즘 구현

#### Option A: Levenshtein Distance (편집 거리)

```c
// 문자열 간 최소 편집 횟수 계산
static int levenshtein_distance(const char* s1, const char* s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);

    // 동적 계획법 테이블
    int** dp = allocate_2d_array(len1 + 1, len2 + 1);

    for (int i = 0; i <= len1; i++) {
        for (int j = 0; j <= len2; j++) {
            if (i == 0) dp[i][j] = j;
            else if (j == 0) dp[i][j] = i;
            else if (s1[i-1] == s2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + min3(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
        }
    }

    int distance = dp[len1][len2];
    free_2d_array(dp, len1 + 1);

    return distance;
}

// 유사도: 0.0 (완전 다름) ~ 1.0 (완전 같음)
static double similarity_score(const char* s1, const char* s2) {
    int distance = levenshtein_distance(s1, s2);
    int max_len = strlen(s1) > strlen(s2) ? strlen(s1) : strlen(s2);
    return 1.0 - (double)distance / max_len;
}
```

**장점**:
- ✅ 정확한 유사도 계산
- ✅ 문자 순서 고려

**단점**:
- ⚠️  O(n*m) 시간 복잡도 (느릴 수 있음)

#### Option B: 토큰 기반 Jaccard Similarity (권장!)

```c
// 공백으로 분리된 토큰 비교
typedef struct {
    char** tokens;
    int count;
} TokenSet;

static TokenSet* tokenize(const char* str) {
    TokenSet* set = malloc(sizeof(TokenSet));
    set->count = 0;
    set->tokens = malloc(256 * sizeof(char*));

    char* copy = strdup(str);
    char* token = strtok(copy, " \t");

    while (token && set->count < 256) {
        set->tokens[set->count++] = strdup(token);
        token = strtok(NULL, " \t");
    }

    free(copy);
    return set;
}

static double jaccard_similarity(const char* s1, const char* s2) {
    TokenSet* set1 = tokenize(s1);
    TokenSet* set2 = tokenize(s2);

    int intersection = 0;
    int union_size = set1->count + set2->count;

    // 교집합 계산
    for (int i = 0; i < set1->count; i++) {
        for (int j = 0; j < set2->count; j++) {
            if (strcmp(set1->tokens[i], set2->tokens[j]) == 0) {
                intersection++;
                union_size--;  // 중복 제거
                break;
            }
        }
    }

    double similarity = (double)intersection / union_size;

    free_tokenset(set1);
    free_tokenset(set2);

    return similarity;
}
```

**장점**:
- ✅ 빠름 (O(n+m))
- ✅ 순서 무관 (토큰만 일치하면 됨)
- ✅ 로그 매칭에 적합

**단점**:
- ⚠️  완전히 다른 순서는 구분 못함

#### Option C: Hybrid 접근 (최종 권장!)

```c
static double hybrid_similarity(const char* log, const char* template) {
    // 1. 토큰 기반 유사도 (빠른 필터링)
    double token_sim = jaccard_similarity(log, template);

    if (token_sim < 0.3) {
        return 0.0;  // 너무 다름, 빠른 종료
    }

    // 2. Levenshtein 기반 정밀 유사도
    double edit_sim = levenshtein_similarity(log, template);

    // 3. 가중 평균
    return 0.7 * token_sim + 0.3 * edit_sim;
}
```

### Phase 2: Python 마스킹 로직을 C로 이식

Python의 `mask_message()` 함수를 C로 재구현:

```c
// anomaly_log_detector/preprocess.py의 mask_message() 참조

static void mask_log_line(const char* input, char* output, size_t output_size) {
    // 1. Hex 주소 마스킹: 0x[0-9a-fA-F]+ → <HEX>
    // 2. IP 주소 마스킹: \d+\.\d+\.\d+\.\d+ → <IP>
    // 3. 숫자 마스킹: \d+ → <NUM>
    // 4. 경로 마스킹: /[\w/.-]+ → <PATH>
    // 5. ID 마스킹: 특정 패턴 → <ID>

    // PCRE2 라이브러리 사용 권장
    // 또는 간단한 상태 기계로 구현
}
```

#### 필요한 정규표현식 패턴 (Python → C)

| Python 패턴 | C 구현 방법 | 우선순위 |
|------------|-----------|---------|
| `r'0x[0-9a-fA-F]+'` | PCRE2 또는 수동 파싱 | 높음 |
| `r'\d+\.\d+\.\d+\.\d+'` | 수동 파싱 (간단) | 높음 |
| `r'\d+'` | 수동 파싱 (매우 간단) | 높음 |
| `r'/[\w/.-]+'` | PCRE2 또는 상태 기계 | 중간 |
| `r'\b[A-Z][A-Z0-9]+\b'` | PCRE2 | 낮음 |

### Phase 3: 성능 최적화

1. **Template 캐싱**
   ```c
   // 자주 매칭되는 템플릿을 캐시
   typedef struct {
       int template_id;
       int match_count;
   } TemplateCache;

   static TemplateCache* recent_templates[10];  // LRU 캐시
   ```

2. **Early termination**
   ```c
   // 유사도가 충분히 높으면 즉시 반환
   if (similarity > 0.95) {
       return template_id;
   }
   ```

3. **병렬 처리** (선택적)
   ```c
   // OpenMP 사용하여 템플릿 비교 병렬화
   #pragma omp parallel for
   for (int i = 0; i < vocab_size; i++) {
       // 유사도 계산
   }
   ```

## 📋 구현 체크리스트

### 필수 (Phase 1)

- [ ] Jaccard similarity 함수 구현
  - [ ] `tokenize()` 함수
  - [ ] `jaccard_similarity()` 함수
  - [ ] 메모리 관리 (malloc/free)

- [ ] Levenshtein distance 함수 구현
  - [ ] 동적 계획법 테이블
  - [ ] 최적화 (공간 복잡도 O(n) 버전)

- [ ] Hybrid similarity 함수 구현
  - [ ] 토큰 기반 필터링
  - [ ] 정밀 유사도 계산
  - [ ] 가중 평균

- [ ] 기존 `string_similarity()` 함수 교체
  - [ ] log_parser.c 수정
  - [ ] 컴파일 및 테스트

### 선택적 (Phase 2)

- [ ] Python 마스킹 로직 이식
  - [ ] Hex 주소 마스킹
  - [ ] IP 주소 마스킹
  - [ ] 숫자 마스킹
  - [ ] 경로 마스킹

- [ ] 정규표현식 라이브러리 통합
  - [ ] PCRE2 설치 및 링크
  - [ ] 패턴 컴파일 및 캐싱

### 고급 (Phase 3)

- [ ] 성능 최적화
  - [ ] Template 캐싱 구현
  - [ ] Early termination
  - [ ] 프로파일링 및 벤치마크

- [ ] 테스트 스위트 작성
  - [ ] 유사도 테스트 케이스
  - [ ] 마스킹 테스트 케이스
  - [ ] 성능 테스트

## 🔧 필요한 도구 및 라이브러리

### 기본

```bash
# 컴파일러
gcc >= 4.9

# 디버깅
valgrind  # 메모리 누수 검사
gdb       # 디버거
```

### 선택적

```bash
# 정규표현식 (Phase 2)
sudo apt-get install libpcre2-dev

# 성능 분석 (Phase 3)
sudo apt-get install valgrind
sudo apt-get install linux-tools-generic  # perf
```

## 📊 예상 성능 향상

| 항목 | 현재 | Phase 1 | Phase 2 | Phase 3 |
|------|------|---------|---------|---------|
| 매칭 정확도 | 10% | 70% | 95% | 95% |
| 처리 속도 | 빠름 | 중간 | 중간 | 빠름 |
| Unknown rate | 90% | 30% | 5% | 5% |

## 📝 테스트 계획

### 1. 유사도 함수 단위 테스트

```c
// test_similarity.c
void test_jaccard_similarity() {
    // 완전 일치
    assert(jaccard_similarity("hello world", "hello world") == 1.0);

    // 부분 일치
    assert(jaccard_similarity("hello world", "world hello") > 0.9);

    // 완전 다름
    assert(jaccard_similarity("hello", "goodbye") < 0.3);
}
```

### 2. 실제 로그 테스트

```bash
# 테스트 데이터
echo "[12345.678901] usb 1-1: new high-speed USB device" > test.log

# C 엔진 실행
./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -i test.log

# 예상 결과: Template ID 찾기 성공 (Unknown 아님)
```

### 3. 성능 벤치마크

```bash
# 1000개 로그 처리 시간 측정
time ./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -i large.log

# 목표: < 1초
```

## 🔗 참고 자료

### 알고리즘

- [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)
- [String Similarity Metrics](https://github.com/luozhouyang/python-string-similarity)

### C 구현 예제

- [String Similarity in C](https://github.com/wooorm/levenshtein.c)
- [PCRE2 Tutorial](https://www.pcre.org/current/doc/html/pcre2api.html)

### 현재 코드

- `hybrid_system/inference/src/log_parser.c` (26-46, 49-83, 108-119번 라인)
- `anomaly_log_detector/preprocess.py` (`mask_message` 함수)

## ✅ 다음 세션 시작 가이드

### 빠른 시작

```bash
# 1. 이 문서 읽기
cat docs/development/TODO_C_PREPROCESSING_IMPROVEMENT.md

# 2. 현재 코드 확인
vim hybrid_system/inference/src/log_parser.c

# 3. 테스트 환경 준비
cd hybrid_system/inference
make clean

# 4. Phase 1 시작: Jaccard similarity 구현
# (새로운 함수 추가)
```

### 권장 순서

1. **Day 1**: Jaccard similarity 구현 및 테스트
2. **Day 2**: Hybrid similarity 적용 및 검증
3. **Day 3**: (선택) Python 마스킹 이식
4. **Day 4**: (선택) 성능 최적화

---

**작성일**: 2025-10-17
**우선순위**: 높음 ⭐⭐⭐⭐⭐
**예상 작업 시간**: 1-3일
**난이도**: 중간

**목표**: 실시간 로그 이상탐지를 위한 고성능 C 전처리 엔진 완성! 🚀
