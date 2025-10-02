#include "../include/inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <regex.h>

// 간단한 로그 파싱을 위한 정규표현식 패턴들
static const char* LOG_PATTERNS[] = {
    // 일반적인 로그 패턴들
    "[0-9]{4}-[0-9]{2}-[0-9]{2}",  // 날짜
    "[0-9]{2}:[0-9]{2}:[0-9]{2}",  // 시간
    "[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+",  // IP 주소
    "[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}",  // UUID
    "/[a-zA-Z0-9/_.-]+",  // 파일 경로
    "[0-9]+",  // 숫자
    NULL
};

// 마스킹 토큰들
static const char* MASK_TOKENS[] = {
    "<*>", "<*>", "<*>", "<*>", "<*>", "<*>"
};

// 문자열 유사도 계산 (간단한 편집 거리)
static int string_similarity(const char* s1, const char* s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    
    if (len1 == 0) return len2;
    if (len2 == 0) return len1;
    
    // 간단한 유사도: 공통 부분 문자열 길이
    int common = 0;
    int min_len = len1 < len2 ? len1 : len2;
    
    for (int i = 0; i < min_len; i++) {
        if (s1[i] == s2[i]) {
            common++;
        } else {
            break;
        }
    }
    
    return common;
}

// 로그 라인 정규화 (간단한 마스킹)
static void normalize_log_line(const char* input, char* output, size_t output_size) {
    if (!input || !output || output_size == 0) {
        return;
    }
    
    strncpy(output, input, output_size - 1);
    output[output_size - 1] = '\0';
    
    // 간단한 패턴 기반 마스킹
    regex_t regex;
    regmatch_t matches[1];
    
    for (int i = 0; LOG_PATTERNS[i] != NULL; i++) {
        if (regcomp(&regex, LOG_PATTERNS[i], REG_EXTENDED) == 0) {
            char* pos = output;
            while (regexec(&regex, pos, 1, matches, 0) == 0) {
                // 매치된 부분을 마스킹 토큰으로 교체
                int match_len = matches[0].rm_eo - matches[0].rm_so;
                int mask_len = strlen(MASK_TOKENS[i]);
                
                // 교체 가능한지 확인
                if (pos + matches[0].rm_so + mask_len < output + output_size) {
                    memmove(pos + matches[0].rm_so + mask_len,
                           pos + matches[0].rm_eo,
                           strlen(pos + matches[0].rm_eo) + 1);
                    memcpy(pos + matches[0].rm_so, MASK_TOKENS[i], mask_len);
                    pos += matches[0].rm_so + mask_len;
                } else {
                    break;
                }
            }
            regfree(&regex);
        }
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
    int best_template_id = -1;
    int best_similarity = -1;
    
    for (size_t i = 0; i < vocab->vocab_size; i++) {
        if (vocab->templates[i]) {
            int similarity = string_similarity(normalized, vocab->templates[i]);
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_template_id = vocab->template_ids[i];
            }
        }
    }
    
    // 최소 유사도 임계값 (템플릿 길이의 50% 이상)
    if (best_template_id >= 0 && vocab->templates[0]) {
        int template_len = strlen(vocab->templates[0]);
        if (best_similarity < template_len * 0.5) {
            // 새로운 템플릿으로 간주
            return -1;
        }
    }
    
    return best_template_id;
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
