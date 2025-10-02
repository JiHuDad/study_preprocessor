#include "../include/inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>

// ONNX 모델 구조체 정의
struct ONNXModel {
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    const OrtApi* ort_api;
    
    // 입출력 정보
    char** input_names;
    char** output_names;
    size_t input_count;
    size_t output_count;
    
    // 모델 메타데이터
    char model_path[512];
    bool is_loaded;
};

// 전역 ONNX Runtime 환경
static OrtEnv* g_ort_env = NULL;
static OrtSessionOptions* g_session_options = NULL;
static bool g_onnx_initialized = false;

// ONNX Runtime 초기화
static InferenceResult init_onnx_runtime() {
    if (g_onnx_initialized) {
        return IE_SUCCESS;
    }
    
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_api) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return IE_ERROR_ONNX_LOAD;
    }
    
    // 환경 생성
    OrtStatus* status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine", &g_ort_env);
    if (status) {
        fprintf(stderr, "Failed to create ONNX Runtime environment\n");
        ort_api->ReleaseStatus(status);
        return IE_ERROR_ONNX_LOAD;
    }
    
    // 세션 옵션 생성
    status = ort_api->CreateSessionOptions(&g_session_options);
    if (status) {
        fprintf(stderr, "Failed to create session options\n");
        ort_api->ReleaseStatus(status);
        return IE_ERROR_ONNX_LOAD;
    }
    
    // CPU 프로바이더 설정 (기본)
    status = ort_api->SetIntraOpNumThreads(g_session_options, 1);
    if (status) {
        ort_api->ReleaseStatus(status);
    }
    
    g_onnx_initialized = true;
    return IE_SUCCESS;
}

// ONNX Runtime 정리
static void cleanup_onnx_runtime() {
    if (!g_onnx_initialized) {
        return;
    }
    
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    if (g_session_options) {
        ort_api->ReleaseSessionOptions(g_session_options);
        g_session_options = NULL;
    }
    
    if (g_ort_env) {
        ort_api->ReleaseEnv(g_ort_env);
        g_ort_env = NULL;
    }
    
    g_onnx_initialized = false;
}

ONNXModel* onnx_model_load(const char* model_path) {
    if (!model_path) {
        return NULL;
    }
    
    // ONNX Runtime 초기화
    if (init_onnx_runtime() != IE_SUCCESS) {
        return NULL;
    }
    
    ONNXModel* model = (ONNXModel*)calloc(1, sizeof(ONNXModel));
    if (!model) {
        return NULL;
    }
    
    model->ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    strncpy(model->model_path, model_path, sizeof(model->model_path) - 1);
    
    // 메모리 정보 생성
    OrtStatus* status = model->ort_api->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &model->memory_info
    );
    if (status) {
        fprintf(stderr, "Failed to create memory info: %s\n", model_path);
        model->ort_api->ReleaseStatus(status);
        free(model);
        return NULL;
    }
    
    // 세션 생성
    status = model->ort_api->CreateSession(
        g_ort_env, model_path, g_session_options, &model->session
    );
    if (status) {
        fprintf(stderr, "Failed to create session: %s\n", model_path);
        model->ort_api->ReleaseStatus(status);
        model->ort_api->ReleaseMemoryInfo(model->memory_info);
        free(model);
        return NULL;
    }
    
    // 입출력 정보 가져오기
    size_t input_count, output_count;
    status = model->ort_api->SessionGetInputCount(model->session, &input_count);
    if (status) {
        model->ort_api->ReleaseStatus(status);
        onnx_model_destroy(model);
        return NULL;
    }
    
    status = model->ort_api->SessionGetOutputCount(model->session, &output_count);
    if (status) {
        model->ort_api->ReleaseStatus(status);
        onnx_model_destroy(model);
        return NULL;
    }
    
    model->input_count = input_count;
    model->output_count = output_count;
    model->is_loaded = true;
    
    printf("ONNX model loaded: %s (inputs: %zu, outputs: %zu)\n", 
           model_path, input_count, output_count);
    
    return model;
}

void onnx_model_destroy(ONNXModel* model) {
    if (!model) {
        return;
    }
    
    if (model->session) {
        model->ort_api->ReleaseSession(model->session);
    }
    
    if (model->memory_info) {
        model->ort_api->ReleaseMemoryInfo(model->memory_info);
    }
    
    // 입출력 이름 해제
    if (model->input_names) {
        for (size_t i = 0; i < model->input_count; i++) {
            if (model->input_names[i]) {
                free(model->input_names[i]);
            }
        }
        free(model->input_names);
    }
    
    if (model->output_names) {
        for (size_t i = 0; i < model->output_count; i++) {
            if (model->output_names[i]) {
                free(model->output_names[i]);
            }
        }
        free(model->output_names);
    }
    
    free(model);
}

InferenceResult onnx_deeplog_infer(
    ONNXModel* model,
    const int* sequence,
    size_t seq_len,
    size_t vocab_size,
    float* predictions
) {
    if (!model || !sequence || !predictions || !model->is_loaded) {
        return IE_ERROR_NULL_POINTER;
    }
    
    // 입력 텐서 생성 (int64_t로 변환)
    int64_t* input_data = (int64_t*)malloc(seq_len * sizeof(int64_t));
    if (!input_data) {
        return IE_ERROR_MEMORY_ALLOC;
    }
    
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = (int64_t)sequence[i];
    }
    
    // 입력 텐서 차원
    int64_t input_shape[] = {1, (int64_t)seq_len};
    size_t input_shape_len = 2;
    
    // 입력 텐서 생성
    OrtValue* input_tensor = NULL;
    OrtStatus* status = model->ort_api->CreateTensorWithDataAsOrtValue(
        model->memory_info,
        input_data,
        seq_len * sizeof(int64_t),
        input_shape,
        input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &input_tensor
    );
    
    if (status) {
        fprintf(stderr, "Failed to create input tensor\n");
        model->ort_api->ReleaseStatus(status);
        free(input_data);
        return IE_ERROR_ONNX_RUN;
    }
    
    // 출력 텐서 준비
    OrtValue* output_tensor = NULL;
    
    // 입출력 이름 (간단화)
    const char* input_names[] = {"input_sequence"};
    const char* output_names[] = {"predictions"};
    
    // 추론 실행
    status = model->ort_api->Run(
        model->session,
        NULL,  // run options
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,  // input count
        output_names,
        1,  // output count
        &output_tensor
    );
    
    if (status) {
        fprintf(stderr, "Failed to run inference\n");
        model->ort_api->ReleaseStatus(status);
        model->ort_api->ReleaseValue(input_tensor);
        free(input_data);
        return IE_ERROR_ONNX_RUN;
    }
    
    // 출력 데이터 추출
    float* output_data = NULL;
    status = model->ort_api->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status) {
        fprintf(stderr, "Failed to get output data\n");
        model->ort_api->ReleaseStatus(status);
        model->ort_api->ReleaseValue(input_tensor);
        model->ort_api->ReleaseValue(output_tensor);
        free(input_data);
        return IE_ERROR_ONNX_RUN;
    }
    
    // 결과 복사
    memcpy(predictions, output_data, vocab_size * sizeof(float));
    
    // 정리
    model->ort_api->ReleaseValue(input_tensor);
    model->ort_api->ReleaseValue(output_tensor);
    free(input_data);
    
    return IE_SUCCESS;
}

InferenceResult onnx_mscred_infer(
    ONNXModel* model,
    const float* features,
    size_t window_size,
    size_t feature_dim,
    float* reconstruction
) {
    if (!model || !features || !reconstruction || !model->is_loaded) {
        return IE_ERROR_NULL_POINTER;
    }
    
    // 입력 텐서 차원
    int64_t input_shape[] = {1, (int64_t)window_size, (int64_t)feature_dim};
    size_t input_shape_len = 3;
    size_t input_size = window_size * feature_dim * sizeof(float);
    
    // 입력 텐서 생성
    OrtValue* input_tensor = NULL;
    OrtStatus* status = model->ort_api->CreateTensorWithDataAsOrtValue(
        model->memory_info,
        (void*)features,
        input_size,
        input_shape,
        input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    if (status) {
        fprintf(stderr, "Failed to create input tensor for MS-CRED\n");
        model->ort_api->ReleaseStatus(status);
        return IE_ERROR_ONNX_RUN;
    }
    
    // 출력 텐서 준비
    OrtValue* output_tensor = NULL;
    
    // 입출력 이름
    const char* input_names[] = {"input_features"};
    const char* output_names[] = {"reconstructed"};
    
    // 추론 실행
    status = model->ort_api->Run(
        model->session,
        NULL,
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        output_names,
        1,
        &output_tensor
    );
    
    if (status) {
        fprintf(stderr, "Failed to run MS-CRED inference\n");
        model->ort_api->ReleaseStatus(status);
        model->ort_api->ReleaseValue(input_tensor);
        return IE_ERROR_ONNX_RUN;
    }
    
    // 출력 데이터 추출
    float* output_data = NULL;
    status = model->ort_api->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status) {
        fprintf(stderr, "Failed to get MS-CRED output data\n");
        model->ort_api->ReleaseStatus(status);
        model->ort_api->ReleaseValue(input_tensor);
        model->ort_api->ReleaseValue(output_tensor);
        return IE_ERROR_ONNX_RUN;
    }
    
    // 결과 복사
    memcpy(reconstruction, output_data, input_size);
    
    // 정리
    model->ort_api->ReleaseValue(input_tensor);
    model->ort_api->ReleaseValue(output_tensor);
    
    return IE_SUCCESS;
}

// 라이브러리 정리 함수 (프로그램 종료 시 호출)
void onnx_engine_cleanup() {
    cleanup_onnx_runtime();
}
