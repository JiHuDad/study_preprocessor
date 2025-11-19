#include "../include/inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 시퀀스 버퍼 생성
SequenceBuffer* sequence_buffer_create(size_t capacity) {
    SequenceBuffer* buffer = (SequenceBuffer*)calloc(1, sizeof(SequenceBuffer));
    if (!buffer) {
        return NULL;
    }
    
    buffer->template_ids = (int*)calloc(capacity, sizeof(int));
    if (!buffer->template_ids) {
        free(buffer);
        return NULL;
    }
    
    buffer->capacity = capacity;
    buffer->length = 0;
    buffer->head = 0;
    
    return buffer;
}

void sequence_buffer_destroy(SequenceBuffer* buffer) {
    if (!buffer) {
        return;
    }
    
    if (buffer->template_ids) {
        free(buffer->template_ids);
    }
    
    free(buffer);
}

InferenceResult sequence_buffer_add(SequenceBuffer* buffer, int template_id) {
    if (!buffer) {
        return IE_ERROR_NULL_POINTER;
    }
    
    // 순환 버퍼에 추가
    buffer->template_ids[buffer->head] = template_id;
    buffer->head = (buffer->head + 1) % buffer->capacity;
    
    if (buffer->length < buffer->capacity) {
        buffer->length++;
    }
    
    return IE_SUCCESS;
}

size_t sequence_buffer_get_sequence(
    const SequenceBuffer* buffer,
    int* sequence,
    size_t seq_len
) {
    if (!buffer || !sequence) {
        return 0;
    }
    
    size_t available = buffer->length < seq_len ? buffer->length : seq_len;
    if (available == 0) {
        return 0;
    }
    
    // 순환 버퍼에서 최신 시퀀스 추출
    size_t start_idx;
    if (buffer->length < buffer->capacity) {
        // 버퍼가 아직 가득 차지 않음
        start_idx = 0;
    } else {
        // 버퍼가 가득 참, head부터 시작
        start_idx = buffer->head;
    }
    
    for (size_t i = 0; i < available; i++) {
        size_t idx = (start_idx + buffer->length - available + i) % buffer->capacity;
        sequence[i] = buffer->template_ids[idx];
    }
    
    return available;
}

// Top-K 예측에서 이상 여부 판단
bool is_anomaly_topk(
    const float* predictions,
    size_t vocab_size,
    int actual_template,
    int k
) {
    if (!predictions || k <= 0 || actual_template < 0) {
        return true; // 안전한 기본값
    }
    
    // Debug mode
    static int debug_topk = -1;
    if (debug_topk == -1) {
        debug_topk = getenv("DEBUG_TOPK") ? 1 : 0;
    }

    // Top-K 인덱스 찾기 (간단한 선택 정렬)
    int* top_indices = (int*)malloc(k * sizeof(int));
    float* top_values = (float*)malloc(k * sizeof(float));

    if (!top_indices || !top_values) {
        free(top_indices);
        free(top_values);
        return true;
    }

    // 초기화
    for (int i = 0; i < k; i++) {
        top_indices[i] = -1;
        top_values[i] = -INFINITY;
    }

    // Top-K 찾기
    for (size_t i = 0; i < vocab_size; i++) {
        float value = predictions[i];

        // 현재 Top-K에 삽입할 위치 찾기
        for (int j = 0; j < k; j++) {
            if (value > top_values[j]) {
                // 뒤로 밀기
                for (int l = k - 1; l > j; l--) {
                    top_values[l] = top_values[l - 1];
                    top_indices[l] = top_indices[l - 1];
                }

                top_values[j] = value;
                top_indices[j] = (int)i;
                break;
            }
        }
    }

    if (debug_topk) {
        fprintf(stderr, "[DEBUG_TOPK] Top-%d indices: [", k);
        for (int i = 0; i < k && i < 20; i++) {  // Show max 20
            if (i > 0) fprintf(stderr, ", ");
            fprintf(stderr, "%d", top_indices[i]);
        }
        if (k > 20) fprintf(stderr, ", ...");
        fprintf(stderr, "]\n");
        fprintf(stderr, "[DEBUG_TOPK] Actual: %d, In Top-%d: ", actual_template, k);
    }

    // actual_template이 Top-K에 있는지 확인
    bool is_in_topk = false;
    for (int i = 0; i < k; i++) {
        if (top_indices[i] == actual_template) {
            is_in_topk = true;
            if (debug_topk) {
                fprintf(stderr, "YES (rank %d)\n", i+1);
            }
            break;
        }
    }

    if (!is_in_topk && debug_topk) {
        fprintf(stderr, "NO\n");
    }

    free(top_indices);
    free(top_values);

    return !is_in_topk; // Top-K에 없으면 이상
}

// 재구성 오차 계산
float calculate_reconstruction_error(
    const float* original,
    const float* reconstructed,
    size_t size
) {
    if (!original || !reconstructed || size == 0) {
        return 0.0f;
    }
    
    float sum_squared_error = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float diff = original[i] - reconstructed[i];
        sum_squared_error += diff * diff;
    }
    
    return sum_squared_error / size; // 평균 제곱 오차
}

// 추론 엔진 생성
InferenceEngine* inference_engine_create(int seq_len, int top_k) {
    if (seq_len <= 0 || top_k <= 0) {
        return NULL;
    }
    
    InferenceEngine* engine = (InferenceEngine*)calloc(1, sizeof(InferenceEngine));
    if (!engine) {
        return NULL;
    }
    
    // 시퀀스 버퍼 생성
    engine->sequence_buffer = sequence_buffer_create(seq_len * 2); // 여유분 확보
    if (!engine->sequence_buffer) {
        free(engine);
        return NULL;
    }
    
    // 설정 초기화
    engine->seq_len = seq_len;
    engine->top_k = top_k;
    engine->anomaly_threshold = 0.5f; // 기본 임계값
    
    // 통계 초기화
    engine->processed_count = 0;
    engine->anomaly_count = 0;
    
    engine->initialized = false;
    
    return engine;
}

void inference_engine_destroy(InferenceEngine* engine) {
    if (!engine) {
        return;
    }
    
    if (engine->deeplog_model) {
        onnx_model_destroy(engine->deeplog_model);
    }
    
    if (engine->mscred_model) {
        onnx_model_destroy(engine->mscred_model);
    }
    
    if (engine->vocab) {
        vocab_dict_destroy(engine->vocab);
    }
    
    if (engine->sequence_buffer) {
        sequence_buffer_destroy(engine->sequence_buffer);
    }
    
    free(engine);
}

InferenceResult inference_engine_load_models(
    InferenceEngine* engine,
    const char* deeplog_path,
    const char* mscred_path
) {
    if (!engine) {
        return IE_ERROR_NULL_POINTER;
    }
    
    // DeepLog 모델 로드
    if (deeplog_path) {
        engine->deeplog_model = onnx_model_load(deeplog_path);
        if (!engine->deeplog_model) {
            fprintf(stderr, "Failed to load DeepLog model: %s\n", deeplog_path);
            return IE_ERROR_ONNX_LOAD;
        }
        printf("DeepLog model loaded successfully\n");
    }
    
    // MS-CRED 모델 로드 (선택적)
    if (mscred_path) {
        engine->mscred_model = onnx_model_load(mscred_path);
        if (!engine->mscred_model) {
            fprintf(stderr, "Failed to load MS-CRED model: %s\n", mscred_path);
            // MS-CRED는 선택적이므로 계속 진행
        } else {
            printf("MS-CRED model loaded successfully\n");
        }
    }
    
    return IE_SUCCESS;
}

InferenceResult inference_engine_load_vocab(
    InferenceEngine* engine,
    const char* vocab_path
) {
    if (!engine || !vocab_path) {
        return IE_ERROR_NULL_POINTER;
    }
    
    engine->vocab = vocab_dict_load_from_json(vocab_path);
    if (!engine->vocab) {
        fprintf(stderr, "Failed to load vocabulary: %s\n", vocab_path);
        return IE_ERROR_FILE_NOT_FOUND;
    }
    
    printf("Vocabulary loaded: %zu templates\n", engine->vocab->vocab_size);
    engine->initialized = true;
    
    return IE_SUCCESS;
}

InferenceResult inference_engine_process_log(
    InferenceEngine* engine,
    const char* log_line,
    AnomalyResult* result
) {
    if (!engine || !log_line || !result || !engine->initialized) {
        return IE_ERROR_NULL_POINTER;
    }
    
    // 결과 초기화
    memset(result, 0, sizeof(AnomalyResult));
    result->is_anomaly = false;
    result->confidence = 0.0f;
    result->score = 0.0f;
    result->predicted_template = -1;
    strcpy(result->reason, "Normal");
    
    // 1. 로그 라인을 템플릿 ID로 변환
    int template_id = log_line_to_template_id(engine->vocab, log_line);
    if (template_id < 0) {
        // 새로운 템플릿 (이상 가능성)
        result->is_anomaly = true;
        result->confidence = 0.8f;
        result->score = 0.8f;
        strcpy(result->reason, "Unknown template");
        
        engine->processed_count++;
        engine->anomaly_count++;
        return IE_SUCCESS;
    }
    
    // 2. 시퀀스 버퍼에 추가
    InferenceResult seq_result = sequence_buffer_add(engine->sequence_buffer, template_id);
    if (seq_result != IE_SUCCESS) {
        return seq_result;
    }
    
    // 3. 충분한 시퀀스가 쌓였는지 확인
    if (engine->sequence_buffer->length < (size_t)engine->seq_len) {
        // 아직 충분하지 않음, 정상으로 간주
        engine->processed_count++;
        return IE_SUCCESS;
    }
    
    // 4. DeepLog 추론 실행
    if (engine->deeplog_model) {
        int* sequence = (int*)malloc(engine->seq_len * sizeof(int));
        float* predictions = (float*)malloc(engine->vocab->vocab_size * sizeof(float));
        
        if (!sequence || !predictions) {
            free(sequence);
            free(predictions);
            return IE_ERROR_MEMORY_ALLOC;
        }
        
        // 현재 시퀀스 가져오기
        size_t seq_length = sequence_buffer_get_sequence(
            engine->sequence_buffer, sequence, engine->seq_len
        );
        
        if (seq_length == (size_t)engine->seq_len) {
            // Debug: Print input sequence
            static int debug_seq = -1;
            if (debug_seq == -1) {
                debug_seq = getenv("DEBUG_SEQ") ? 1 : 0;
            }

            if (debug_seq) {
                fprintf(stderr, "[DEBUG_SEQ] Input sequence (len=%d): [", engine->seq_len);
                for (int i = 0; i < engine->seq_len; i++) {
                    if (i > 0) fprintf(stderr, ", ");
                    fprintf(stderr, "%d", sequence[i]);
                }
                fprintf(stderr, "]\n");
            }

            // DeepLog 추론 실행
            InferenceResult infer_result = onnx_deeplog_infer(
                engine->deeplog_model,
                sequence,
                engine->seq_len,
                engine->vocab->vocab_size,
                predictions
            );

            if (infer_result == IE_SUCCESS) {
                // Debug mode check
                static int debug_anomaly = -1;
                if (debug_anomaly == -1) {
                    debug_anomaly = getenv("DEBUG_ANOMALY") ? 1 : 0;
                }

                // Top-K 검사
                bool is_anomaly = is_anomaly_topk(
                    predictions,
                    engine->vocab->vocab_size,
                    template_id,
                    engine->top_k
                );

                if (debug_anomaly) {
                    fprintf(stderr, "[DEBUG_ANOMALY] Actual template: %d, Top-K=%d\n",
                            template_id, engine->top_k);
                    fprintf(stderr, "[DEBUG_ANOMALY] Result: %s\n",
                            is_anomaly ? "ANOMALY ❌" : "NORMAL ✅");
                }

                result->is_anomaly = is_anomaly;
                result->predicted_template = template_id;

                if (is_anomaly) {
                    result->confidence = 0.9f;
                    result->score = 0.9f;
                    strcpy(result->reason, "DeepLog anomaly");
                    engine->anomaly_count++;
                } else {
                    result->confidence = 0.1f;
                    result->score = 0.1f;
                    strcpy(result->reason, "DeepLog normal");
                }
            } else {
                // 추론 실패, 보수적으로 이상으로 간주
                result->is_anomaly = true;
                result->confidence = 0.5f;
                result->score = 0.5f;
                strcpy(result->reason, "DeepLog inference failed");
                engine->anomaly_count++;
            }
        }
        
        free(sequence);
        free(predictions);
    }
    
    // TODO: MS-CRED 추론 추가 (필요시)
    
    engine->processed_count++;
    return IE_SUCCESS;
}

// 통계 출력
void inference_engine_print_stats(const InferenceEngine* engine) {
    if (!engine) {
        return;
    }
    
    printf("=== Inference Engine Statistics ===\n");
    printf("Processed logs: %lu\n", engine->processed_count);
    printf("Detected anomalies: %lu\n", engine->anomaly_count);
    
    if (engine->processed_count > 0) {
        double anomaly_rate = (double)engine->anomaly_count / engine->processed_count * 100.0;
        printf("Anomaly rate: %.2f%%\n", anomaly_rate);
    }
    
    printf("Sequence length: %d\n", engine->seq_len);
    printf("Top-K: %d\n", engine->top_k);
    printf("Initialized: %s\n", engine->initialized ? "Yes" : "No");
    
    if (engine->vocab) {
        printf("Vocabulary size: %zu\n", engine->vocab->vocab_size);
    }
    
    printf("Models loaded:\n");
    printf("  DeepLog: %s\n", engine->deeplog_model ? "Yes" : "No");
    printf("  MS-CRED: %s\n", engine->mscred_model ? "Yes" : "No");
}

// 상태 확인
bool inference_engine_is_healthy(const InferenceEngine* engine) {
    if (!engine) {
        return false;
    }
    
    return engine->initialized && 
           engine->vocab != NULL && 
           engine->sequence_buffer != NULL &&
           (engine->deeplog_model != NULL || engine->mscred_model != NULL);
}

// 에러 코드를 문자열로 변환
const char* inference_result_to_string(InferenceResult result) {
    switch (result) {
        case IE_SUCCESS: return "Success";
        case IE_ERROR_NULL_POINTER: return "Null pointer error";
        case IE_ERROR_INVALID_PARAM: return "Invalid parameter";
        case IE_ERROR_MEMORY_ALLOC: return "Memory allocation error";
        case IE_ERROR_FILE_NOT_FOUND: return "File not found";
        case IE_ERROR_ONNX_LOAD: return "ONNX model load error";
        case IE_ERROR_ONNX_RUN: return "ONNX inference error";
        case IE_ERROR_BUFFER_FULL: return "Buffer full";
        case IE_ERROR_PARSE_FAILED: return "Parse failed";
        default: return "Unknown error";
    }
}
