# Examples

이 디렉토리는 Anomaly Log Detector 사용 예제와 테스트 데이터를 포함합니다.

## 디렉토리 구조

### 📂 data/
샘플 로그 파일들:
- `test_sample.log` - 기본 테스트용 샘플 로그
- `direct_test.log` - 직접 테스트용 로그

### 📜 scripts/
예제 스크립트들:
- `generate_test_logs.py` - 합성 테스트 로그 생성 스크립트

## 사용법

### 합성 로그 생성

```bash
python examples/scripts/generate_test_logs.py
```

### 샘플 로그로 테스트

```bash
# 기본 파싱 테스트
alog-detect parse --input examples/data/test_sample.log --out-dir output/

# 전체 파이프라인 테스트
./scripts/train_models.sh examples/data/test_sample.log models_test
./scripts/run_inference.sh models_test examples/data/test_sample.log
```

## 자신의 로그로 테스트하기

1. 로그 파일을 `examples/data/`에 복사
2. 위의 명령어에서 파일명만 변경하여 실행

## 추가 리소스

- [메인 문서](../README.md)
- [학습/추론 가이드](../docs/guides/TRAIN_INFERENCE_GUIDE.md)
