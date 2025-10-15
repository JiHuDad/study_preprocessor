# Study Preprocessor - 프로젝트 컨텍스트

## 📋 프로젝트 개요

**study-preprocessor**는 커널/시스템 로그의 전처리와 이상탐지를 위한 통합 파이프라인입니다. LSTM 기반 DeepLog와 시계열 이미지화 기반 MS-CRED 모델에 입력 가능한 형식으로 데이터를 변환하고, 다양한 이상탐지 방법을 제공합니다.

## 🎯 핵심 기능

### 1. 로그 전처리
- **템플릿 마이닝**: Drain3 알고리즘을 사용한 자동 로그 템플릿 추출
- **마스킹**: 정규식 기반 변수 값 마스킹 (IP, 경로, 숫자 등)
- **포맷 지원**: syslog, dmesg, 커널 로그 등 다양한 포맷

### 2. 이상탐지 방법
- **베이스라인 탐지**: 윈도우 기반 템플릿 빈도 변화 감지
- **DeepLog**: LSTM 기반 시퀀스 예측 모델
- **MS-CRED**: 멀티스케일 컨볼루션 재구성 오차 기반 탐지
- **시간 기반 탐지**: 과거 동일 시간대 패턴과 비교
- **비교 탐지**: 여러 파일 간 패턴 차이 분석

### 3. 🆕 학습/추론 분리 워크플로우
- **모델 학습**: 정상 로그로 DeepLog, MS-CRED, 베이스라인 통계 학습
- **이상탐지 추론**: 학습된 모델로 Target 로그 분석
- **모델 비교**: 서로 다른 시점 모델들의 성능 비교
- **점진적 학습**: 기존 모델에 새로운 데이터 추가 학습
- **모델 검증**: 자동화된 품질 평가 (0-100점)

### 4. 배치 분석
- **단일 파일**: 개별 로그 파일 전처리
- **디렉토리 배치**: 폴더 내 모든 로그 파일 일괄 처리
- **재귀 스캔**: 하위 디렉토리 포함 전체 스캔
- **날짜별 구조**: 날짜/카테고리별 폴더 구조 지원

## 🗂️ 프로젝트 구조

```
study_preprocessor/
├── study_preprocessor/          # 핵심 모듈
│   ├── cli.py                  # CLI 엔트리포인트
│   ├── preprocess.py           # 전처리 (마스킹 + Drain3)
│   ├── detect.py               # 베이스라인 이상탐지
│   ├── synth.py                # 합성 데이터 생성
│   ├── eval.py                 # 평가 메트릭
│   └── builders/               # 모델별 입력 빌더
│       ├── deeplog.py          # DeepLog용 시퀀스 생성
│       └── mscred.py           # MS-CRED용 윈도우 카운트
├── data/                       # 데이터 디렉토리
│   ├── raw/                    # 원시 로그 파일
│   └── processed/              # 전처리 결과
├── enhanced_batch_analyzer.py   # 향상된 배치 분석기
├── temporal_anomaly_detector.py # 시간 기반 이상탐지
├── comparative_anomaly_detector.py # 비교 기반 이상탐지
├── log_sample_analyzer.py      # 이상 로그 샘플 추출 및 분석
├── rules.json                  # 마스킹 규칙 설정
├── train_models.sh             # 🆕 모델 학습 스크립트
├── run_inference.sh            # 🆕 이상탐지 추론 스크립트
├── compare_models.sh           # 🆕 모델 비교 도구
├── train_models_incremental.sh # 🆕 점진적 학습 도구
├── validate_models.sh          # 🆕 모델 검증 도구
└── run_*.sh                    # 기타 실행 스크립트들
```

## 🔧 주요 구성 요소

### 마스킹 규칙 (rules.json)
```json
{
  "hex_addresses": "0x[0-9a-fA-F]+ → <HEX>",
  "ipv4_addresses": "192.168.1.1 → <IP>",
  "file_paths": "/usr/bin/python → <PATH>",
  "device_numbers": "eth0 → eth<ID>",
  "decimal_numbers": "123 → <NUM>"
}
```

### CLI 명령어
```bash
# 기본 전처리
study-preprocess parse --input file.log --out-dir processed/

# DeepLog 입력 생성
study-preprocess build-deeplog --parsed parsed.parquet --out-dir processed/

# 이상탐지 실행
study-preprocess detect --parsed parsed.parquet --out-dir processed/

# 배치 분석
./run_enhanced_batch_analysis.sh /var/log/
```

## 📊 산출물 형식

### 전처리 결과
- **parsed.parquet**: 주요 전처리 테이블
  - `line_no`: 라인 번호
  - `timestamp`: 파싱된 타임스탬프
  - `host`: 호스트명 (syslog)
  - `process`: 프로세스명
  - `raw`: 원본 메시지
  - `masked`: 마스킹된 메시지
  - `template_id`: Drain3 템플릿 ID
  - `template`: 추출된 템플릿

### DeepLog 입력
- **vocab.json**: 템플릿 ID → 인덱스 매핑
- **sequences.parquet**: 정렬된 템플릿 인덱스 시퀀스

### 이상탐지 결과
- **baseline_scores.parquet**: 윈도우별 이상 점수
- **deeplog_infer.parquet**: DeepLog 예측 결과
- **report.md**: 상위 이상 윈도우 요약

## 🚀 워크플로우

### 1. 단일 파일 분석
```bash
# 1. 전처리
study-preprocess parse --input /path/to/app.log --out-dir data/processed

# 2. 모델 입력 생성
study-preprocess build-deeplog --parsed data/processed/parsed.parquet --out-dir data/processed

# 3. 이상탐지
study-preprocess detect --parsed data/processed/parsed.parquet --out-dir data/processed

# 4. 리포트 생성
study-preprocess report --processed-dir data/processed
```

### 2. 🆕 학습/추론 분리 워크플로우 (추천)
```bash
# 1단계: 정상 로그로 모델 학습
./train_models.sh /var/log/normal/ my_models

# 2단계: 모델 품질 검증
./validate_models.sh my_models

# 3단계: Target 로그 이상탐지 (실제 로그 샘플 포함)
./run_inference.sh my_models /var/log/suspicious.log

# 결과 확인
cat inference_*/log_samples_analysis/anomaly_analysis_report.md
```

### 3. 배치 분석
```bash
# 향상된 배치 분석
./run_enhanced_batch_analysis.sh /var/log/ app.log 3 20 batch_result

# 결과 확인
cat batch_result/ENHANCED_ANALYSIS_SUMMARY.md
```

### 4. 시간 기반 분석
```bash
# 시간대별 패턴 학습 및 이상탐지
study-preprocess analyze-temporal --data-dir data/processed
cat data/processed/temporal_analysis/temporal_report.md
```

## 🔍 이상탐지 방법론

### 베이스라인 방법
- **윈도우 기반**: 50개 라인을 하나의 윈도우로 처리
- **빈도 분석**: 템플릿 출현 빈도의 급격한 변화 감지
- **미지 템플릿**: 새로운 템플릿의 비율 모니터링
- **EWM 통계**: 지수 가중 이동 평균으로 정상 패턴 학습

### DeepLog 방법
- **LSTM 모델**: 시퀀스 패턴 학습
- **Top-K 예측**: 다음 템플릿 상위 K개 예측
- **위반 탐지**: 예측 범위를 벗어나는 패턴 감지

### 시간 기반 방법
- **시간대별 프로파일**: 시간/요일별 정상 패턴 학습
- **계절성 고려**: 주기적 패턴 반영
- **이상 임계값**: 과거 동일 시간대 대비 편차 계산

### 비교 기반 방법
- **파일 간 비교**: 여러 시스템/서비스 로그 비교
- **유사도 측정**: 코사인 유사도, KL divergence 등
- **상대적 이상**: 다른 파일 대비 특이한 패턴 감지

## ⚙️ 설정 파라미터

### Drain3 설정
- **depth**: 4 (파싱 트리 깊이)
- **similarity_threshold**: 0.4 (템플릿 유사도 임계값)
- **max_clusters**: 1000 (최대 템플릿 수)

### 이상탐지 설정
- **window_size**: 50 (윈도우 크기)
- **stride**: 25 (슬라이딩 스트라이드)
- **ewm_alpha**: 0.3 (지수 가중 평균 계수)
- **anomaly_quantile**: 0.95 (이상 임계값 분위수)

### 배치 분석 설정
- **max_depth**: 3 (하위 디렉토리 스캔 깊이)
- **max_files**: 20 (최대 처리 파일 수)
- **log_patterns**: *.log, *.txt, *.syslog 등

## 🔬 고급 기능

### 🆕 모델 관리 도구

#### 모델 비교
```bash
# 두 모델의 성능 비교
./compare_models.sh old_models new_models /var/log/test.log
```

#### 점진적 학습
```bash
# 기존 모델에 새로운 데이터 추가 학습
./train_models_incremental.sh base_models /var/log/new_normal/ updated_models
```

#### 모델 검증
```bash
# 모델 품질 자동 평가 (0-100점)
./validate_models.sh my_models
```

### 이상 로그 샘플 분석
```bash
# 이상탐지 결과에서 실제 문제 로그들 추출
study-preprocess analyze-samples --processed-dir data/processed --output-dir log_samples
```

### 합성 데이터 생성
```bash
# 테스트용 합성 로그 생성
study-preprocess gen-synth --out data/raw/synthetic.log --lines 10000 --anomaly-rate 0.03
```

### 평가 메트릭
```bash
# Precision, Recall, F1 계산
study-preprocess eval --processed-dir data/processed --labels data/raw/synthetic.log.labels.parquet
```

### 시각화
```bash
# 결과 시각화 (추가 스크립트)
python visualize_results.py --data-dir data/processed
```

## 📈 성능 지표

### 처리 성능
- **목표**: 1GB 로그 < 2분 (M1/M2 기준)
- **메모리**: 청크 단위 스트리밍 처리
- **저장**: Parquet 형식으로 효율적 압축

### 탐지 성능
- **베이스라인**: F1 Score 0.7-0.8 (합성 데이터)
- **DeepLog**: Violation Rate 0.05-0.15 (정상 로그)
- **실시간**: 윈도우 단위 즉시 탐지 가능

## 🛠️ 개발 환경

### 필수 요구사항
- Python 3.11+
- pip 또는 uv 패키지 매니저

### 주요 의존성
- **drain3**: 템플릿 마이닝
- **pandas**: 데이터 처리
- **pyarrow**: Parquet I/O
- **torch**: DeepLog 모델
- **click**: CLI 인터페이스

### 설치 방법
```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -e .
```

## 🔄 확장 계획

### 단기 계획
- [ ] 실시간 스트리밍 처리
- [ ] 웹 대시보드 인터페이스
- [ ] 추가 로그 포맷 지원

### 장기 계획
- [ ] 분산 처리 (Spark 연동)
- [ ] 기계학습 모델 추가
- [ ] 클라우드 연동

## 📚 참고 문서

- [PRD.md](./prd.md): 제품 요구사항 명세서
- [README.md](./README.md): 사용자 가이드
- [BATCH_ANALYSIS_GUIDE.md](./BATCH_ANALYSIS_GUIDE.md): 배치 분석 가이드
- [ANOMALY_DETECTION_METHODS.md](./ANOMALY_DETECTION_METHODS.md): 이상탐지 방법론
- [RESULTS_GUIDE.md](./RESULTS_GUIDE.md): 결과 해석 가이드
