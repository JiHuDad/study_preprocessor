# C 추론 엔진

하이브리드 로그 이상탐지 시스템의 고성능 C 추론 엔진입니다.

## 🎯 특징

- **고성능**: C로 구현된 최적화된 추론 엔진
- **ONNX 지원**: PyTorch에서 변환된 ONNX 모델 실행
- **실시간 처리**: 스트림 기반 로그 처리
- **메모리 효율**: 최소한의 메모리 사용량
- **크로스 플랫폼**: Linux/macOS/Windows 지원

## 📦 의존성

### 필수 의존성
- **GCC 4.9+** 또는 **Clang 3.9+**
- **ONNX Runtime C API 1.16.0+**
- **pthread** (POSIX 스레드)
- **libm** (수학 라이브러리)

### 선택적 의존성
- **valgrind** (메모리 검사용)
- **cppcheck** (정적 분석용)
- **clang-format** (코드 포맷팅용)

## 🛠️ 빌드

### 1. ONNX Runtime 설치

```bash
# ONNX Runtime 다운로드
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz

# 시스템에 설치
sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/
sudo ldconfig
```

### 2. 의존성 확인

```bash
make check-deps
```

### 3. 빌드

```bash
# 기본 빌드
make

# 디버그 빌드
make debug

# 라이브러리 빌드
make static shared

# 모든 타겟 빌드
make all static shared
```

## 🚀 사용법

### 기본 사용법

```bash
# 기본 실행
./bin/inference_engine -d deeplog.onnx -v vocab.json -i /var/log/syslog

# 테스트 모드
./bin/inference_engine -d deeplog.onnx -v vocab.json -t

# 실시간 스트림 처리
tail -f /var/log/syslog | ./bin/inference_engine -d deeplog.onnx -v vocab.json
```

### 명령행 옵션

```
Usage: inference_engine [OPTIONS]

Options:
  -d, --deeplog PATH     DeepLog ONNX model path
  -m, --mscred PATH      MS-CRED ONNX model path (optional)
  -v, --vocab PATH       Vocabulary JSON file path
  -s, --seq-len N        Sequence length (default: 50)
  -k, --top-k N          Top-K value (default: 3)
  -i, --input PATH       Input log file (default: stdin)
  -o, --output PATH      Output results file (default: stdout)
  -t, --test             Run test mode with sample data
  -h, --help             Show help message
```

### 예시

```bash
# 파일에서 로그 처리
./bin/inference_engine \
  -d models/deeplog.onnx \
  -v models/vocab.json \
  -i /var/log/application.log \
  -o results.json

# MS-CRED 모델 포함
./bin/inference_engine \
  -d models/deeplog.onnx \
  -m models/mscred.onnx \
  -v models/vocab.json \
  -s 100 -k 5

# 실시간 모니터링
tail -f /var/log/syslog | \
./bin/inference_engine \
  -d models/deeplog.onnx \
  -v models/vocab.json
```

## 📊 출력 형식

### 콘솔 출력
```
[ANOMALY] DeepLog anomaly (0.900) - 2024-01-01 10:00:03 ERROR Authentication failed
[NORMAL ] DeepLog normal (0.100) - 2024-01-01 10:00:04 INFO Database query completed
```

### JSON 출력 (-o 옵션 사용시)
```json
[
  {
    "line_number": 1,
    "timestamp": 1704067203,
    "is_anomaly": true,
    "confidence": 0.900000,
    "score": 0.900000,
    "reason": "DeepLog anomaly",
    "predicted_template": 15,
    "log_line": "2024-01-01 10:00:03 ERROR Authentication failed for user bob"
  }
]
```

## 🧪 테스트

### 기본 테스트
```bash
make test
```

### 성능 벤치마크
```bash
make benchmark
```

### 메모리 검사
```bash
make memcheck
```

### 정적 분석
```bash
make analyze
```

## 📚 API 사용법

### C 라이브러리로 사용

```c
#include "inference_engine.h"

int main() {
    // 추론 엔진 생성
    InferenceEngine* engine = inference_engine_create(50, 3);
    
    // 모델 로드
    inference_engine_load_models(engine, "deeplog.onnx", NULL);
    inference_engine_load_vocab(engine, "vocab.json");
    
    // 로그 처리
    AnomalyResult result;
    InferenceResult status = inference_engine_process_log(
        engine, 
        "2024-01-01 10:00:01 INFO User logged in", 
        &result
    );
    
    if (status == IE_SUCCESS) {
        printf("Anomaly: %s (%.3f)\n", 
               result.is_anomaly ? "YES" : "NO", 
               result.confidence);
    }
    
    // 정리
    inference_engine_destroy(engine);
    return 0;
}
```

### 컴파일
```bash
gcc -o my_app my_app.c -linference_engine -lonnxruntime -lm
```

## 🔧 설정

### 환경 변수
- `OMP_NUM_THREADS`: OpenMP 스레드 수 (기본: 1)
- `ORT_LOGGING_LEVEL`: ONNX Runtime 로그 레벨 (0-4)

### 성능 튜닝
- **시퀀스 길이**: 더 긴 시퀀스는 정확도 향상, 메모리 사용량 증가
- **Top-K 값**: 더 큰 K는 민감도 감소, 정확도 향상
- **배치 크기**: 더 큰 배치는 처리량 향상, 지연시간 증가

## 📈 성능 특성

### 벤치마크 결과 (예상)
- **처리량**: 10,000-50,000 logs/sec
- **지연시간**: < 10ms per log
- **메모리**: 50-100MB
- **CPU**: 단일 코어 20-30%

### 확장성
- **수직 확장**: 멀티코어 CPU 활용
- **수평 확장**: 여러 인스턴스 병렬 실행
- **스트림 분할**: 로그 소스별 분산 처리

## 🐛 문제 해결

### 일반적인 문제

**1. ONNX Runtime 로드 실패**
```bash
# 라이브러리 경로 확인
ldd bin/inference_engine
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**2. 모델 파일 오류**
```bash
# ONNX 모델 검증
python -c "import onnx; onnx.checker.check_model('model.onnx')"
```

**3. 메모리 부족**
```bash
# 시퀀스 길이 줄이기
./bin/inference_engine -s 25 -d model.onnx -v vocab.json
```

### 디버그 모드
```bash
# 디버그 빌드
make debug

# GDB로 디버깅
gdb ./bin/inference_engine
(gdb) run -d model.onnx -v vocab.json -t
```

## 🤝 기여

1. 코드 포맷팅: `make format`
2. 정적 분석: `make analyze`
3. 테스트 실행: `make test memcheck`
4. 문서 업데이트

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🔗 관련 링크

- [ONNX Runtime](https://onnxruntime.ai/)
- [하이브리드 시스템 문서](../README.md)
- [Python 학습 컴포넌트](../training/README.md)
