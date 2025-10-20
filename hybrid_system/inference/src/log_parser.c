#define _POSIX_C_SOURCE 200809L
#include "../include/inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <regex.h>

// ============================================================================
// MASKING PATTERNS - Ported from Python preprocess.py
// Order matters: apply in this sequence to avoid over-masking
// ============================================================================

// Masking configuration (matches Python PreprocessConfig defaults)
typedef struct {
    int mask_hex;            // 0x[0-9a-fA-F]+
    int mask_ipv4;           // IPv4 addresses
    int mask_ipv6;           // IPv6 addresses
    int mask_mac;            // MAC addresses
    int mask_uuid;           // UUIDs
    int mask_pid_fields;     // pid=, tid=, uid=, gid=
    int mask_device_numbers; // eth0 -> eth<ID>, sda1 -> sda<ID>
    int mask_numbers;        // Generic decimal numbers
    int mask_paths;          // File paths /foo/bar
} MaskingConfig;

// Default config: all masking enabled (matches Python defaults)
static const MaskingConfig DEFAULT_MASKING_CONFIG = {
    .mask_hex = 1,
    .mask_ipv4 = 1,
    .mask_ipv6 = 1,
    .mask_mac = 1,
    .mask_uuid = 1,
    .mask_pid_fields = 1,
    .mask_device_numbers = 1,
    .mask_numbers = 1,
    .mask_paths = 1
};

// Regex patterns (compiled once, reused)
static regex_t regex_hex;
static regex_t regex_ipv4;
static regex_t regex_ipv6;
static regex_t regex_mac;
static regex_t regex_uuid;
static regex_t regex_pid;
static regex_t regex_device;
static regex_t regex_decimal;
static regex_t regex_path;
static int regex_compiled = 0;

// Compile all regex patterns once at startup
static void compile_masking_patterns(void) {
    if (regex_compiled) return;

    // HEX_ADDR: 0x[0-9a-fA-F]+
    regcomp(&regex_hex, "0x[0-9a-fA-F]+", REG_EXTENDED);

    // IPV4: \b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b
    // Simplified for C: [0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}
    regcomp(&regex_ipv4, "([0-9]{1,3}\\.){3}[0-9]{1,3}", REG_EXTENDED);

    // IPV6: simplified pattern
    regcomp(&regex_ipv6, "([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}", REG_EXTENDED);

    // MAC: ([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}
    regcomp(&regex_mac, "([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", REG_EXTENDED);

    // UUID: [0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}
    regcomp(&regex_uuid, "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}", REG_EXTENDED);

    // PID: (pid|tid|uid|gid)=[0-9]+
    regcomp(&regex_pid, "(pid|tid|uid|gid)=[0-9]+", REG_EXTENDED);

    // DEVICE_NUM: ([a-zA-Z]+)([0-9]+) - captures device name and number separately
    regcomp(&regex_device, "([a-zA-Z]+)([0-9]+)", REG_EXTENDED);

    // DECIMAL: -?[0-9]+(\.[0-9]+)?
    regcomp(&regex_decimal, "-?[0-9]+(\\.[0-9]+)?", REG_EXTENDED);

    // PATH: (/|~)[a-zA-Z0-9._/-]+
    regcomp(&regex_path, "(/|~)[a-zA-Z0-9._/-]+", REG_EXTENDED);

    regex_compiled = 1;
}

// Free regex patterns
static void free_masking_patterns(void) {
    if (!regex_compiled) return;

    regfree(&regex_hex);
    regfree(&regex_ipv4);
    regfree(&regex_ipv6);
    regfree(&regex_mac);
    regfree(&regex_uuid);
    regfree(&regex_pid);
    regfree(&regex_device);
    regfree(&regex_decimal);
    regfree(&regex_path);

    regex_compiled = 0;
}

// ============================================================================
// 개선된 유사도 알고리즘
// ============================================================================

// 토큰 집합 구조체
typedef struct {
    char** tokens;
    int count;
    int capacity;
} TokenSet;

// 토큰화 함수 (공백 기준)
static TokenSet* tokenize(const char* str) {
    if (!str) return NULL;

    TokenSet* set = (TokenSet*)malloc(sizeof(TokenSet));
    if (!set) return NULL;

    set->capacity = 256;
    set->count = 0;
    set->tokens = (char**)calloc(set->capacity, sizeof(char*));
    if (!set->tokens) {
        free(set);
        return NULL;
    }

    char* copy = strdup(str);
    if (!copy) {
        free(set->tokens);
        free(set);
        return NULL;
    }

    char* token = strtok(copy, " \t\n\r");
    while (token && set->count < set->capacity) {
        set->tokens[set->count] = strdup(token);
        if (set->tokens[set->count]) {
            set->count++;
        }
        token = strtok(NULL, " \t\n\r");
    }

    free(copy);
    return set;
}

// TokenSet 메모리 해제
static void free_tokenset(TokenSet* set) {
    if (!set) return;

    if (set->tokens) {
        for (int i = 0; i < set->count; i++) {
            free(set->tokens[i]);
        }
        free(set->tokens);
    }
    free(set);
}

// Jaccard Similarity 계산 (토큰 기반)
static double jaccard_similarity(const char* s1, const char* s2) {
    TokenSet* set1 = tokenize(s1);
    TokenSet* set2 = tokenize(s2);

    if (!set1 || !set2) {
        free_tokenset(set1);
        free_tokenset(set2);
        return 0.0;
    }

    if (set1->count == 0 && set2->count == 0) {
        free_tokenset(set1);
        free_tokenset(set2);
        return 1.0;
    }

    if (set1->count == 0 || set2->count == 0) {
        free_tokenset(set1);
        free_tokenset(set2);
        return 0.0;
    }

    // 교집합 계산
    int intersection = 0;
    for (int i = 0; i < set1->count; i++) {
        for (int j = 0; j < set2->count; j++) {
            if (strcmp(set1->tokens[i], set2->tokens[j]) == 0) {
                intersection++;
                break;
            }
        }
    }

    // 합집합 크기 = set1 + set2 - 교집합
    int union_size = set1->count + set2->count - intersection;

    double similarity = (union_size > 0) ? (double)intersection / union_size : 0.0;

    free_tokenset(set1);
    free_tokenset(set2);

    return similarity;
}

// Levenshtein Distance 계산 (공간 최적화 버전)
static int levenshtein_distance(const char* s1, const char* s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);

    if (len1 == 0) return len2;
    if (len2 == 0) return len1;

    // 공간 최적화: O(min(len1, len2))만 사용
    if (len1 > len2) {
        const char* tmp = s1;
        s1 = s2;
        s2 = tmp;
        int tmp_len = len1;
        len1 = len2;
        len2 = tmp_len;
    }

    // 1차원 배열 2개만 사용 (이전 행, 현재 행)
    int* prev_row = (int*)malloc((len1 + 1) * sizeof(int));
    int* curr_row = (int*)malloc((len1 + 1) * sizeof(int));

    if (!prev_row || !curr_row) {
        free(prev_row);
        free(curr_row);
        return -1;
    }

    // 초기화
    for (int i = 0; i <= len1; i++) {
        prev_row[i] = i;
    }

    // 동적 계획법
    for (int j = 1; j <= len2; j++) {
        curr_row[0] = j;

        for (int i = 1; i <= len1; i++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;

            int deletion = prev_row[i] + 1;
            int insertion = curr_row[i-1] + 1;
            int substitution = prev_row[i-1] + cost;

            curr_row[i] = deletion < insertion ? deletion : insertion;
            curr_row[i] = curr_row[i] < substitution ? curr_row[i] : substitution;
        }

        // 행 교체
        int* tmp = prev_row;
        prev_row = curr_row;
        curr_row = tmp;
    }

    int distance = prev_row[len1];

    free(prev_row);
    free(curr_row);

    return distance;
}

// Levenshtein 기반 유사도 (0.0 ~ 1.0)
static double levenshtein_similarity(const char* s1, const char* s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    int max_len = len1 > len2 ? len1 : len2;

    if (max_len == 0) return 1.0;

    int distance = levenshtein_distance(s1, s2);
    if (distance < 0) return 0.0;  // 에러 발생

    return 1.0 - (double)distance / max_len;
}

// Hybrid Similarity (Jaccard + Levenshtein)
// 반환값: 유사도 점수 (0.0 ~ 1.0)
static double hybrid_similarity(const char* s1, const char* s2) {
    // 1단계: Jaccard similarity로 빠른 필터링
    double jaccard_sim = jaccard_similarity(s1, s2);

    // Jaccard 유사도가 너무 낮으면 즉시 반환 (성능 최적화)
    if (jaccard_sim < 0.2) {
        return jaccard_sim;  // 빠른 종료
    }

    // 2단계: Jaccard가 어느 정도 높으면 Levenshtein으로 정밀 계산
    double levenshtein_sim = levenshtein_similarity(s1, s2);

    // 3단계: 가중 평균 (Jaccard 70%, Levenshtein 30%)
    // Jaccard: 토큰 순서 무관, 빠름
    // Levenshtein: 정확하지만 느림
    return 0.7 * jaccard_sim + 0.3 * levenshtein_sim;
}

// 레거시 인터페이스 유지 (int 반환)
// 새로운 hybrid_similarity를 사용하되, int로 변환
static int string_similarity(const char* s1, const char* s2) {
    double similarity = hybrid_similarity(s1, s2);

    // 0.0 ~ 1.0을 0 ~ 100으로 변환
    return (int)(similarity * 100);
}

// Helper function to replace regex matches with mask token
static void regex_replace_all(char* str, size_t str_size, regex_t* regex, const char* mask) {
    regmatch_t matches[3]; // Support up to 2 capture groups
    char buffer[MAX_LOG_LINE_LENGTH];
    char* src = str;
    char* dst = buffer;
    size_t remaining = sizeof(buffer) - 1;

    while (regexec(regex, src, 3, matches, 0) == 0 && remaining > 0) {
        // Copy text before match
        size_t prefix_len = matches[0].rm_so;
        if (prefix_len > remaining) prefix_len = remaining;
        memcpy(dst, src, prefix_len);
        dst += prefix_len;
        remaining -= prefix_len;

        // Copy mask token
        size_t mask_len = strlen(mask);
        if (mask_len > remaining) mask_len = remaining;
        memcpy(dst, mask, mask_len);
        dst += mask_len;
        remaining -= mask_len;

        // Move source pointer past match
        src += matches[0].rm_eo;
    }

    // Copy remaining text
    size_t rest_len = strlen(src);
    if (rest_len > remaining) rest_len = remaining;
    memcpy(dst, src, rest_len);
    dst[rest_len] = '\0';

    // Copy back to original string
    strncpy(str, buffer, str_size - 1);
    str[str_size - 1] = '\0';
}

// Special handler for device numbers: eth0 -> eth<ID>
static void mask_device_numbers(char* str, size_t str_size) {
    regmatch_t matches[3];
    char buffer[MAX_LOG_LINE_LENGTH];
    char* src = str;
    char* dst = buffer;
    size_t remaining = sizeof(buffer) - 1;

    while (regexec(&regex_device, src, 3, matches, 0) == 0 && remaining > 0) {
        // Copy text before match
        size_t prefix_len = matches[0].rm_so;
        if (prefix_len > remaining) prefix_len = remaining;
        memcpy(dst, src, prefix_len);
        dst += prefix_len;
        remaining -= prefix_len;

        // Copy device name (group 1)
        size_t name_len = matches[1].rm_eo - matches[1].rm_so;
        if (name_len > remaining) name_len = remaining;
        memcpy(dst, src + matches[1].rm_so, name_len);
        dst += name_len;
        remaining -= name_len;

        // Add <ID> instead of number
        const char* mask = "<ID>";
        size_t mask_len = strlen(mask);
        if (mask_len > remaining) mask_len = remaining;
        memcpy(dst, mask, mask_len);
        dst += mask_len;
        remaining -= mask_len;

        // Move source pointer past match
        src += matches[0].rm_eo;
    }

    // Copy remaining text
    size_t rest_len = strlen(src);
    if (rest_len > remaining) rest_len = remaining;
    memcpy(dst, src, rest_len);
    dst[rest_len] = '\0';

    // Copy back to original string
    strncpy(str, buffer, str_size - 1);
    str[str_size - 1] = '\0';
}

// Special handler for PID fields: pid=1234 -> pid=<ID>
static void mask_pid_fields(char* str, size_t str_size) {
    regmatch_t matches[2];
    char buffer[MAX_LOG_LINE_LENGTH];
    char* src = str;
    char* dst = buffer;
    size_t remaining = sizeof(buffer) - 1;

    while (regexec(&regex_pid, src, 2, matches, 0) == 0 && remaining > 0) {
        // Copy text before match
        size_t prefix_len = matches[0].rm_so;
        if (prefix_len > remaining) prefix_len = remaining;
        memcpy(dst, src, prefix_len);
        dst += prefix_len;
        remaining -= prefix_len;

        // Find '=' position in the match
        char* eq_pos = strchr(src + matches[0].rm_so, '=');
        if (eq_pos) {
            size_t prefix_to_eq = eq_pos - (src + matches[0].rm_so) + 1; // Include '='
            if (prefix_to_eq > remaining) prefix_to_eq = remaining;
            memcpy(dst, src + matches[0].rm_so, prefix_to_eq);
            dst += prefix_to_eq;
            remaining -= prefix_to_eq;

            // Add <ID>
            const char* mask = "<ID>";
            size_t mask_len = strlen(mask);
            if (mask_len > remaining) mask_len = remaining;
            memcpy(dst, mask, mask_len);
            dst += mask_len;
            remaining -= mask_len;
        }

        // Move source pointer past match
        src += matches[0].rm_eo;
    }

    // Copy remaining text
    size_t rest_len = strlen(src);
    if (rest_len > remaining) rest_len = remaining;
    memcpy(dst, src, rest_len);
    dst[rest_len] = '\0';

    // Copy back to original string
    strncpy(str, buffer, str_size - 1);
    str[str_size - 1] = '\0';
}

// Main masking function - ported from Python mask_message()
// Order matters: match Python's order to avoid over-masking
static void normalize_log_line(const char* input, char* output, size_t output_size) {
    if (!input || !output || output_size == 0) {
        return;
    }

    // Ensure regex patterns are compiled
    compile_masking_patterns();

    // Start with input copy
    strncpy(output, input, output_size - 1);
    output[output_size - 1] = '\0';

    const MaskingConfig* cfg = &DEFAULT_MASKING_CONFIG;

    // Apply masks in same order as Python (paths first!)
    if (cfg->mask_paths) {
        regex_replace_all(output, output_size, &regex_path, "<PATH>");
    }

    if (cfg->mask_hex) {
        regex_replace_all(output, output_size, &regex_hex, "<HEX>");
    }

    if (cfg->mask_ipv4) {
        regex_replace_all(output, output_size, &regex_ipv4, "<IP>");
    }

    if (cfg->mask_ipv6) {
        regex_replace_all(output, output_size, &regex_ipv6, "<IP6>");
    }

    if (cfg->mask_mac) {
        regex_replace_all(output, output_size, &regex_mac, "<MAC>");
    }

    if (cfg->mask_uuid) {
        regex_replace_all(output, output_size, &regex_uuid, "<UUID>");
    }

    if (cfg->mask_pid_fields) {
        mask_pid_fields(output, output_size);
    }

    if (cfg->mask_device_numbers) {
        mask_device_numbers(output, output_size);
    }

    // Numbers last to avoid over-masking
    if (cfg->mask_numbers) {
        regex_replace_all(output, output_size, &regex_decimal, "<NUM>");
    }
}

int log_line_to_template_id(const VocabDict* vocab, const char* log_line) {
    if (!vocab || !log_line || vocab->vocab_size == 0) {
        return -1;
    }
    
    // 로그 라인 정규화
    char normalized[MAX_LOG_LINE_LENGTH];
    normalize_log_line(log_line, normalized, sizeof(normalized));
    
    // 가장 유사한 템플릿 찾기
    int best_index = -1;  // 배열 인덱스 (0-based)
    int best_similarity = -1;

    for (size_t i = 0; i < vocab->vocab_size; i++) {
        if (vocab->templates[i]) {
            int similarity = string_similarity(normalized, vocab->templates[i]);
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_index = (int)i;  // 배열 인덱스 저장
            }
        }
    }

    // 최소 유사도 임계값
    // Hybrid similarity returns 0-100
    // 50% = good match with some variations
    // This is much better than the old prefix matching which required exact character-by-character match
    if (best_index >= 0) {
        if (best_similarity < 50) {
            // 유사도가 너무 낮음 (50% 미만)
            return -1;
        }
    }

    // IMPORTANT: Return 0-based index (not template_id from JSON)
    // ONNX model expects indices 0, 1, 2, ... vocab_size-1
    return best_index;
}

// 어휘 사전 생성
VocabDict* vocab_dict_create(size_t capacity) {
    VocabDict* vocab = (VocabDict*)calloc(1, sizeof(VocabDict));
    if (!vocab) {
        return NULL;
    }
    
    vocab->templates = (char**)calloc(capacity, sizeof(char*));
    vocab->template_ids = (int*)calloc(capacity, sizeof(int));
    
    if (!vocab->templates || !vocab->template_ids) {
        vocab_dict_destroy(vocab);
        return NULL;
    }
    
    vocab->capacity = capacity;
    vocab->vocab_size = 0;
    
    return vocab;
}

void vocab_dict_destroy(VocabDict* vocab) {
    if (!vocab) {
        return;
    }
    
    if (vocab->templates) {
        for (size_t i = 0; i < vocab->vocab_size; i++) {
            if (vocab->templates[i]) {
                free(vocab->templates[i]);
            }
        }
        free(vocab->templates);
    }
    
    if (vocab->template_ids) {
        free(vocab->template_ids);
    }
    
    free(vocab);
}

// JSON 파싱을 위한 간단한 함수들
static char* find_json_string_value(const char* json, const char* key) {
    char search_pattern[256];
    snprintf(search_pattern, sizeof(search_pattern), "\"%s\":", key);
    
    char* pos = strstr(json, search_pattern);
    if (!pos) {
        return NULL;
    }
    
    // 값 부분 찾기
    pos += strlen(search_pattern);
    while (*pos && isspace(*pos)) pos++;
    
    if (*pos != '"') {
        return NULL;
    }
    
    pos++; // 시작 따옴표 건너뛰기
    char* start = pos;
    
    // 끝 따옴표 찾기
    while (*pos && *pos != '"') {
        if (*pos == '\\') pos++; // 이스케이프 문자 건너뛰기
        pos++;
    }
    
    if (*pos != '"') {
        return NULL;
    }
    
    size_t len = pos - start;
    char* result = (char*)malloc(len + 1);
    if (result) {
        strncpy(result, start, len);
        result[len] = '\0';
    }
    
    return result;
}

VocabDict* vocab_dict_load_from_json(const char* vocab_path) {
    if (!vocab_path) {
        return NULL;
    }
    
    FILE* file = fopen(vocab_path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open vocab file: %s\n", vocab_path);
        return NULL;
    }
    
    // 파일 크기 확인
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size <= 0 || file_size > 10 * 1024 * 1024) { // 10MB 제한
        fprintf(stderr, "Invalid vocab file size: %ld\n", file_size);
        fclose(file);
        return NULL;
    }
    
    // 파일 내용 읽기
    char* json_content = (char*)malloc(file_size + 1);
    if (!json_content) {
        fclose(file);
        return NULL;
    }
    
    size_t read_size = fread(json_content, 1, file_size, file);
    json_content[read_size] = '\0';
    fclose(file);
    
    // 어휘 사전 생성
    VocabDict* vocab = vocab_dict_create(MAX_VOCAB_SIZE);
    if (!vocab) {
        free(json_content);
        return NULL;
    }
    
    // JSON 파싱 (간단한 구현)
    // 형식: {"template_id": "template_string", ...}
    char* pos = json_content;
    int template_id;
    char* template_str;
    
    while ((pos = strchr(pos, '"')) != NULL) {
        pos++; // 따옴표 건너뛰기
        
        // 키(template_id) 읽기
        char* key_start = pos;
        while (*pos && *pos != '"') pos++;
        if (!*pos) break;
        
        size_t key_len = pos - key_start;
        char key[32];
        if (key_len < sizeof(key)) {
            strncpy(key, key_start, key_len);
            key[key_len] = '\0';
            template_id = atoi(key);
        } else {
            continue;
        }
        
        pos++; // 닫는 따옴표 건너뛰기
        
        // ':' 찾기
        while (*pos && *pos != ':') pos++;
        if (!*pos) break;
        pos++;
        
        // 값 찾기
        while (*pos && isspace(*pos)) pos++;
        if (*pos != '"') continue;
        pos++;
        
        char* value_start = pos;
        while (*pos && *pos != '"') {
            if (*pos == '\\') pos++; // 이스케이프 처리
            pos++;
        }
        if (!*pos) break;
        
        size_t value_len = pos - value_start;
        template_str = (char*)malloc(value_len + 1);
        if (template_str) {
            strncpy(template_str, value_start, value_len);
            template_str[value_len] = '\0';
            
            // 어휘 사전에 추가
            if (vocab->vocab_size < vocab->capacity) {
                vocab->templates[vocab->vocab_size] = template_str;
                vocab->template_ids[vocab->vocab_size] = template_id;
                vocab->vocab_size++;
            } else {
                free(template_str);
                break;
            }
        }
        
        pos++;
    }
    
    free(json_content);
    
    printf("Loaded vocabulary: %zu templates\n", vocab->vocab_size);
    return vocab;
}
