#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 최대 상수 정의
#define MAX_LOG_LINE_LENGTH 4096
#define MAX_TEMPLATE_COUNT 10000
#define MAX_SEQUENCE_LENGTH 100
#define MAX_VOCAB_SIZE 50000
#define DEFAULT_BUFFER_SIZE 1000

// 에러 코드
typedef enum {
    IE_SUCCESS = 0,
    IE_ERROR_NULL_POINTER = -1,
    IE_ERROR_INVALID_PARAM = -2,
    IE_ERROR_MEMORY_ALLOC = -3,
    IE_ERROR_FILE_NOT_FOUND = -4,
    IE_ERROR_ONNX_LOAD = -5,
    IE_ERROR_ONNX_RUN = -6,
    IE_ERROR_BUFFER_FULL = -7,
    IE_ERROR_PARSE_FAILED = -8
} InferenceResult;

// 로그 엔트리 구조체
typedef struct {
    char content[MAX_LOG_LINE_LENGTH];
    uint64_t timestamp;
    int template_id;
    float confidence;
} LogEntry;

// 시퀀스 버퍼 구조체
typedef struct {
    int* template_ids;
    size_t length;
    size_t capacity;
    size_t head;  // 순환 버퍼용
} SequenceBuffer;

// 이상탐지 결과 구조체
typedef struct {
    bool is_anomaly;
    float confidence;
    float score;
    int predicted_template;
    char reason[256];
} AnomalyResult;

// 어휘 사전 구조체
typedef struct {
    char** templates;
    int* template_ids;
    size_t vocab_size;
    size_t capacity;
} VocabDict;

// ONNX 모델 구조체 (불투명 포인터)
typedef struct ONNXModel ONNXModel;

// 추론 엔진 구조체
typedef struct {
    ONNXModel* deeplog_model;
    ONNXModel* mscred_model;
    VocabDict* vocab;
    SequenceBuffer* sequence_buffer;
    
    // 설정
    int seq_len;
    int top_k;
    float anomaly_threshold;
    
    // 통계
    uint64_t processed_count;
    uint64_t anomaly_count;
    
    // 상태
    bool initialized;
} InferenceEngine;

// === 초기화 및 정리 함수 ===

/**
 * 추론 엔진 생성
 * @param seq_len 시퀀스 길이
 * @param top_k Top-K 값
 * @return 추론 엔진 포인터 (실패시 NULL)
 */
InferenceEngine* inference_engine_create(int seq_len, int top_k);

/**
 * 추론 엔진 해제
 * @param engine 추론 엔진 포인터
 */
void inference_engine_destroy(InferenceEngine* engine);

/**
 * ONNX 모델 로드
 * @param engine 추론 엔진
 * @param deeplog_path DeepLog ONNX 모델 경로
 * @param mscred_path MS-CRED ONNX 모델 경로 (NULL 가능)
 * @return 결과 코드
 */
InferenceResult inference_engine_load_models(
    InferenceEngine* engine,
    const char* deeplog_path,
    const char* mscred_path
);

/**
 * 어휘 사전 로드
 * @param engine 추론 엔진
 * @param vocab_path vocab.json 파일 경로
 * @return 결과 코드
 */
InferenceResult inference_engine_load_vocab(
    InferenceEngine* engine,
    const char* vocab_path
);

// === 로그 처리 함수 ===

/**
 * 로그 라인 처리 (메인 함수)
 * @param engine 추론 엔진
 * @param log_line 로그 라인
 * @param result 이상탐지 결과 (출력)
 * @return 결과 코드
 */
InferenceResult inference_engine_process_log(
    InferenceEngine* engine,
    const char* log_line,
    AnomalyResult* result
);

/**
 * 로그 라인을 템플릿 ID로 변환
 * @param vocab 어휘 사전
 * @param log_line 로그 라인
 * @return 템플릿 ID (-1이면 실패)
 */
int log_line_to_template_id(const VocabDict* vocab, const char* log_line);

/**
 * 시퀀스 버퍼에 템플릿 ID 추가
 * @param buffer 시퀀스 버퍼
 * @param template_id 템플릿 ID
 * @return 결과 코드
 */
InferenceResult sequence_buffer_add(SequenceBuffer* buffer, int template_id);

/**
 * 시퀀스 버퍼에서 현재 시퀀스 가져오기
 * @param buffer 시퀀스 버퍼
 * @param sequence 출력 시퀀스 배열
 * @param seq_len 시퀀스 길이
 * @return 실제 복사된 길이
 */
size_t sequence_buffer_get_sequence(
    const SequenceBuffer* buffer,
    int* sequence,
    size_t seq_len
);

// === ONNX 모델 함수 ===

/**
 * ONNX 모델 로드
 * @param model_path ONNX 모델 파일 경로
 * @return ONNX 모델 포인터 (실패시 NULL)
 */
ONNXModel* onnx_model_load(const char* model_path);

/**
 * ONNX 모델 해제
 * @param model ONNX 모델 포인터
 */
void onnx_model_destroy(ONNXModel* model);

/**
 * DeepLog 추론 실행
 * @param model ONNX 모델
 * @param sequence 입력 시퀀스
 * @param seq_len 시퀀스 길이
 * @param vocab_size 어휘 크기
 * @param predictions 출력 예측 배열
 * @return 결과 코드
 */
InferenceResult onnx_deeplog_infer(
    ONNXModel* model,
    const int* sequence,
    size_t seq_len,
    size_t vocab_size,
    float* predictions
);

/**
 * MS-CRED 추론 실행
 * @param model ONNX 모델
 * @param features 입력 피처
 * @param window_size 윈도우 크기
 * @param feature_dim 피처 차원
 * @param reconstruction 출력 재구성 데이터
 * @return 결과 코드
 */
InferenceResult onnx_mscred_infer(
    ONNXModel* model,
    const float* features,
    size_t window_size,
    size_t feature_dim,
    float* reconstruction
);

// === 유틸리티 함수 ===

/**
 * 어휘 사전 생성
 * @param capacity 초기 용량
 * @return 어휘 사전 포인터 (실패시 NULL)
 */
VocabDict* vocab_dict_create(size_t capacity);

/**
 * 어휘 사전 해제
 * @param vocab 어휘 사전 포인터
 */
void vocab_dict_destroy(VocabDict* vocab);

/**
 * JSON 파일에서 어휘 사전 로드
 * @param vocab_path JSON 파일 경로
 * @return 어휘 사전 포인터 (실패시 NULL)
 */
VocabDict* vocab_dict_load_from_json(const char* vocab_path);

/**
 * 시퀀스 버퍼 생성
 * @param capacity 버퍼 용량
 * @return 시퀀스 버퍼 포인터 (실패시 NULL)
 */
SequenceBuffer* sequence_buffer_create(size_t capacity);

/**
 * 시퀀스 버퍼 해제
 * @param buffer 시퀀스 버퍼 포인터
 */
void sequence_buffer_destroy(SequenceBuffer* buffer);

/**
 * Top-K 예측에서 이상 여부 판단
 * @param predictions 예측 배열
 * @param vocab_size 어휘 크기
 * @param actual_template 실제 템플릿 ID
 * @param k Top-K 값
 * @return true면 이상 (Top-K에 없음)
 */
bool is_anomaly_topk(
    const float* predictions,
    size_t vocab_size,
    int actual_template,
    int k
);

/**
 * 재구성 오차 계산
 * @param original 원본 데이터
 * @param reconstructed 재구성 데이터
 * @param size 데이터 크기
 * @return 평균 제곱 오차
 */
float calculate_reconstruction_error(
    const float* original,
    const float* reconstructed,
    size_t size
);

/**
 * 에러 코드를 문자열로 변환
 * @param result 에러 코드
 * @return 에러 메시지 문자열
 */
const char* inference_result_to_string(InferenceResult result);

// === 통계 및 모니터링 함수 ===

/**
 * 추론 엔진 통계 출력
 * @param engine 추론 엔진
 */
void inference_engine_print_stats(const InferenceEngine* engine);

/**
 * 추론 엔진 상태 확인
 * @param engine 추론 엔진
 * @return true면 정상 상태
 */
bool inference_engine_is_healthy(const InferenceEngine* engine);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_ENGINE_H
