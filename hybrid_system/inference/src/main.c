#include "../include/inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <ctype.h>

// 전역 변수
static InferenceEngine* g_engine = NULL;
static volatile bool g_running = true;

// 시그널 핸들러
void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down...\n", sig);
    g_running = false;
}

// meta.json에서 seq_len 읽기
int read_seq_len_from_meta(const char* model_path) {
    if (!model_path) return -1;

    // model_path에서 .onnx를 .onnx.meta.json으로 변경
    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s.meta.json", model_path);

    FILE* f = fopen(meta_path, "r");
    if (!f) {
        return -1;  // meta.json이 없으면 실패
    }

    // 파일 읽기
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* content = (char*)malloc(fsize + 1);
    if (!content) {
        fclose(f);
        return -1;
    }

    fread(content, 1, fsize, f);
    fclose(f);
    content[fsize] = '\0';

    // 간단한 JSON 파싱: "seq_len": 숫자 찾기
    int seq_len = -1;
    char* seq_len_str = strstr(content, "\"seq_len\"");
    if (seq_len_str) {
        // "seq_len": 50 형태 찾기
        char* colon = strchr(seq_len_str, ':');
        if (colon) {
            // 숫자 부분 찾기
            while (*colon && !isdigit(*colon) && *colon != '-') {
                colon++;
            }
            if (*colon) {
                seq_len = atoi(colon);
            }
        }
    }

    free(content);
    return seq_len;
}

// 도움말 출력
void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -d, --deeplog PATH     DeepLog ONNX model path\n");
    printf("  -m, --mscred PATH      MS-CRED ONNX model path (optional)\n");
    printf("  -v, --vocab PATH       Vocabulary JSON file path\n");
    printf("  -s, --seq-len N        Sequence length (default: auto-detect from meta.json, fallback: 50)\n");
    printf("  -k, --top-k N          Top-K value (default: 3)\n");
    printf("  -i, --input PATH       Input log file (default: stdin)\n");
    printf("  -o, --output PATH      Output results file (default: stdout)\n");
    printf("  -t, --test             Run test mode with sample data\n");
    printf("  -h, --help             Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -d deeplog.onnx -v vocab.json -i /var/log/syslog\n", program_name);
    printf("  %s -d deeplog.onnx -m mscred.onnx -v vocab.json -t\n", program_name);
    printf("  tail -f /var/log/syslog | %s -d deeplog.onnx -v vocab.json\n", program_name);
}

// 테스트 모드 실행
int run_test_mode(InferenceEngine* engine) {
    printf("Running test mode...\n");
    
    // 테스트 로그 라인들
    const char* test_logs[] = {
        "2024-01-01 10:00:01 INFO User alice logged in successfully",
        "2024-01-01 10:00:02 INFO Processing request 12345 completed",
        "2024-01-01 10:00:03 ERROR Authentication failed for user bob",
        "2024-01-01 10:00:04 INFO Database query executed in 150ms",
        "2024-01-01 10:00:05 CRITICAL System overload detected - CPU 95%",
        "2024-01-01 10:00:06 INFO User alice logged out",
        "2024-01-01 10:00:07 FATAL Unexpected system crash - core dumped",
        "2024-01-01 10:00:08 INFO Service restarted successfully",
        NULL
    };
    
    printf("\nProcessing test logs:\n");
    printf("============================================================\n");
    
    for (int i = 0; test_logs[i] != NULL; i++) {
        AnomalyResult result;
        InferenceResult status = inference_engine_process_log(
            engine, test_logs[i], &result
        );
        
        if (status == IE_SUCCESS) {
            printf("Log %d: %s\n", i + 1, 
                   result.is_anomaly ? "ANOMALY" : "NORMAL");
            printf("  Confidence: %.3f\n", result.confidence);
            printf("  Reason: %s\n", result.reason);
            printf("  Input: %.80s%s\n", test_logs[i], 
                   strlen(test_logs[i]) > 80 ? "..." : "");
        } else {
            printf("Log %d: ERROR - %s\n", i + 1, 
                   inference_result_to_string(status));
        }
        printf("\n");
    }
    
    // 통계 출력
    inference_engine_print_stats(engine);
    
    return 0;
}

// 파일에서 로그 처리
int process_log_file(InferenceEngine* engine, const char* input_path, const char* output_path) {
    FILE* input_file = stdin;
    FILE* output_file = stdout;
    
    // 입력 파일 열기
    if (input_path && strcmp(input_path, "-") != 0) {
        input_file = fopen(input_path, "r");
        if (!input_file) {
            fprintf(stderr, "Failed to open input file: %s\n", input_path);
            return 1;
        }
    }
    
    // 출력 파일 열기
    if (output_path && strcmp(output_path, "-") != 0) {
        output_file = fopen(output_path, "w");
        if (!output_file) {
            fprintf(stderr, "Failed to open output file: %s\n", output_path);
            if (input_file != stdin) fclose(input_file);
            return 1;
        }
    }
    
    printf("Processing logs from %s...\n", 
           input_path ? input_path : "stdin");
    printf("Press Ctrl+C to stop.\n\n");
    
    // JSON 헤더 출력
    if (output_file != stdout) {
        fprintf(output_file, "[\n");
    }
    
    char line[MAX_LOG_LINE_LENGTH];
    int line_count = 0;
    int anomaly_count = 0;
    bool first_output = true;
    
    while (g_running && fgets(line, sizeof(line), input_file)) {
        // 개행 문자 제거
        line[strcspn(line, "\n")] = '\0';
        
        if (strlen(line) == 0) {
            continue;
        }
        
        AnomalyResult result;
        InferenceResult status = inference_engine_process_log(engine, line, &result);
        
        line_count++;
        
        if (status == IE_SUCCESS) {
            if (result.is_anomaly) {
                anomaly_count++;
            }
            
            // 결과 출력
            if (output_file == stdout) {
                // 콘솔 출력 (간단한 형태)
                printf("[%s] %s (%.3f) - %.80s%s\n",
                       result.is_anomaly ? "ANOMALY" : "NORMAL ",
                       result.reason,
                       result.confidence,
                       line,
                       strlen(line) > 80 ? "..." : "");
            } else {
                // JSON 파일 출력
                if (!first_output) {
                    fprintf(output_file, ",\n");
                }
                
                fprintf(output_file, "  {\n");
                fprintf(output_file, "    \"line_number\": %d,\n", line_count);
                fprintf(output_file, "    \"timestamp\": %ld,\n", time(NULL));
                fprintf(output_file, "    \"is_anomaly\": %s,\n", 
                        result.is_anomaly ? "true" : "false");
                fprintf(output_file, "    \"confidence\": %.6f,\n", result.confidence);
                fprintf(output_file, "    \"score\": %.6f,\n", result.score);
                fprintf(output_file, "    \"reason\": \"%s\",\n", result.reason);
                fprintf(output_file, "    \"predicted_template\": %d,\n", 
                        result.predicted_template);
                fprintf(output_file, "    \"log_line\": \"%s\"\n", line);
                fprintf(output_file, "  }");
                
                first_output = false;
            }
            
            // 주기적 통계 출력 (1000라인마다)
            if (line_count % 1000 == 0) {
                printf("Processed %d lines, %d anomalies (%.2f%%)\n",
                       line_count, anomaly_count,
                       line_count > 0 ? (double)anomaly_count / line_count * 100.0 : 0.0);
            }
        } else {
            fprintf(stderr, "Error processing line %d: %s\n", 
                    line_count, inference_result_to_string(status));
        }
    }
    
    // JSON 푸터 출력
    if (output_file != stdout) {
        fprintf(output_file, "\n]\n");
    }
    
    // 최종 통계
    printf("\nFinal statistics:\n");
    printf("Total lines processed: %d\n", line_count);
    printf("Anomalies detected: %d (%.2f%%)\n", 
           anomaly_count, 
           line_count > 0 ? (double)anomaly_count / line_count * 100.0 : 0.0);
    
    // 파일 닫기
    if (input_file != stdin) {
        fclose(input_file);
    }
    if (output_file != stdout) {
        fclose(output_file);
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    // 기본 설정
    const char* deeplog_path = NULL;
    const char* mscred_path = NULL;
    const char* vocab_path = NULL;
    const char* input_path = NULL;
    const char* output_path = NULL;
    int seq_len = -1;  // -1 means auto-detect
    int top_k = 3;
    bool test_mode = false;
    
    // 명령행 인수 파싱
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--deeplog") == 0) {
            if (++i < argc) deeplog_path = argv[i];
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mscred") == 0) {
            if (++i < argc) mscred_path = argv[i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--vocab") == 0) {
            if (++i < argc) vocab_path = argv[i];
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seq-len") == 0) {
            if (++i < argc) seq_len = atoi(argv[i]);
        } else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--top-k") == 0) {
            if (++i < argc) top_k = atoi(argv[i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (++i < argc) input_path = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i < argc) output_path = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
            test_mode = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // 필수 인수 확인
    if (!deeplog_path || !vocab_path) {
        fprintf(stderr, "Error: DeepLog model and vocabulary are required\n");
        print_usage(argv[0]);
        return 1;
    }

    // seq_len 자동 감지 (사용자가 지정하지 않았으면)
    if (seq_len == -1) {
        seq_len = read_seq_len_from_meta(deeplog_path);
        if (seq_len > 0) {
            printf("Auto-detected seq_len=%d from %s.meta.json\n", seq_len, deeplog_path);
        } else {
            seq_len = 50;  // 기본값으로 폴백
            fprintf(stderr, "Warning: Could not read seq_len from meta.json, using default: %d\n", seq_len);
        }
    }

    // 시그널 핸들러 설정
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("C Inference Engine Starting...\n");
    printf("DeepLog model: %s\n", deeplog_path);
    if (mscred_path) {
        printf("MS-CRED model: %s\n", mscred_path);
    }
    printf("Vocabulary: %s\n", vocab_path);
    printf("Sequence length: %d\n", seq_len);
    printf("Top-K: %d\n", top_k);
    printf("\n");

    // 추론 엔진 생성
    g_engine = inference_engine_create(seq_len, top_k);
    if (!g_engine) {
        fprintf(stderr, "Failed to create inference engine\n");
        return 1;
    }
    
    // 모델 로드
    InferenceResult result = inference_engine_load_models(
        g_engine, deeplog_path, mscred_path
    );
    if (result != IE_SUCCESS) {
        fprintf(stderr, "Failed to load models: %s\n", 
                inference_result_to_string(result));
        inference_engine_destroy(g_engine);
        return 1;
    }
    
    // 어휘 사전 로드
    result = inference_engine_load_vocab(g_engine, vocab_path);
    if (result != IE_SUCCESS) {
        fprintf(stderr, "Failed to load vocabulary: %s\n", 
                inference_result_to_string(result));
        inference_engine_destroy(g_engine);
        return 1;
    }
    
    printf("Inference engine initialized successfully!\n\n");
    
    // 상태 확인
    if (!inference_engine_is_healthy(g_engine)) {
        fprintf(stderr, "Inference engine is not healthy\n");
        inference_engine_destroy(g_engine);
        return 1;
    }
    
    int exit_code = 0;
    
    // 실행 모드에 따라 처리
    if (test_mode) {
        exit_code = run_test_mode(g_engine);
    } else {
        exit_code = process_log_file(g_engine, input_path, output_path);
    }
    
    // 정리
    printf("\nShutting down inference engine...\n");
    inference_engine_destroy(g_engine);
    g_engine = NULL;
    
    // ONNX Runtime 정리
    extern void onnx_engine_cleanup();
    onnx_engine_cleanup();
    
    printf("Inference engine stopped.\n");
    return exit_code;
}
