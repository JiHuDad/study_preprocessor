# 🚀 배치 로그 분석 가이드

## 📋 개요

여러 로그 파일을 한 번에 분석하고 비교할 수 있는 배치 분석 시스템입니다.

## 🆕 **최신 업데이트 (2025-10-02)**

### ✨ **새로운 주요 기능들:**
- **🔄 학습/추론 분리 워크플로우**: 효율적인 모델 재사용으로 시간 절약
- **📊 모델 비교 도구**: 서로 다른 시점 모델들의 성능 객관적 비교
- **🔄 점진적 학습**: 기존 모델에 새로운 데이터를 추가하여 지속적 개선
- **🔍 자동화된 모델 검증**: 0-100점 품질 점수로 모델 상태 자동 평가
- **📋 실제 이상 로그 샘플 추출**: 이상탐지 결과에서 문제 로그들을 자동 추출 및 분석
- **🎯 외부 Target 파일 지원**: 다른 디렉토리의 파일을 Target으로 지정 가능
- **📄 종합 리포트 통합**: 모든 분석 결과를 `COMPREHENSIVE_ANALYSIS_REPORT.md`로 통합

## 🆚 세 가지 분석 방법

### 1. **기본 배치 분석기** (`batch_log_analyzer.py`)
단일 디렉토리의 로그 파일들을 분석합니다.

```bash
# 기본 사용법
./run_batch_analysis.sh /path/to/logs/

# Target 파일 지정
./run_batch_analysis.sh /path/to/logs/ server1.log my_analysis
```

### 2. **향상된 배치 분석기** (`enhanced_batch_analyzer.py`)
하위 디렉토리를 재귀적으로 스캔하여 날짜별/카테고리별 구조를 지원합니다.

### 3. **🆕 학습/추론 분리 워크플로우** ⭐ **추천**
모델 학습과 추론을 분리하여 효율성을 극대화하는 새로운 방법입니다.

```bash
# 1단계: 정상 로그로 모델 학습
./train_models.sh /var/log/normal/ my_models

# 2단계: Target 로그 이상탐지 (실제 로그 샘플 포함)
./run_inference.sh my_models /var/log/suspicious.log

# 고급: 모델 성능 비교
./compare_models.sh old_models new_models

# 고급: 점진적 학습 (기존 모델 개선)
./train_models_incremental.sh old_models /var/log/new_normal/ updated_models

# 고급: 모델 품질 검증 (0-100점)
./validate_models.sh my_models
```

**장점:**
- 🔄 **효율성**: 한 번 학습하면 여러 Target에 재사용
- 📊 **일관성**: 동일한 기준으로 일관된 이상탐지
- 🔍 **검증**: 자동화된 모델 품질 평가
- 📋 **샘플**: 실제 문제 로그들을 자동 추출 및 분석

---

## 🎯 향상된 배치 분석 상세 사용법

```bash
# 기본 사용법: 자동 날짜/시간 폴더 생성
./run_enhanced_batch_analysis.sh /path/to/logs/

# Target 파일 지정 (같은 디렉토리 내)
./run_enhanced_batch_analysis.sh /path/to/logs/ target.log

# 🆕 외부 Target 파일 지원 (다른 디렉토리)
./run_enhanced_batch_analysis.sh /path/to/baseline/ /path/to/target/problem.log

# 세부 옵션 지정 (디렉토리, Target파일, 깊이, 최대파일수, 결과폴더)
./run_enhanced_batch_analysis.sh /path/to/logs/ target.log 3 20 my_analysis

# 결과 확인 - 🆕 통합 종합 리포트
cat my_analysis/COMPREHENSIVE_ANALYSIS_REPORT.md
```

## 🎯 향상된 배치 분석 상세 사용법

### 명령어 구문
```bash
./run_enhanced_batch_analysis.sh <로그디렉토리> [target파일] [최대깊이] [최대파일수] [작업디렉토리]
```

### 매개변수 설명
- **로그디렉토리**: 스캔할 루트 디렉토리 (하위 폴더 포함)
- **target파일**: 집중 분석할 특정 파일명 (생략시 가장 큰 파일)
- **최대깊이**: 하위 디렉토리 스캔 깊이 (기본: 3)
- **최대파일수**: 처리할 최대 파일 수 (기본: 20)
- **작업디렉토리**: 결과 저장 폴더 (생략시 자동 생성)

### 사용 예시

#### 1. 기본 스캔
```bash
# 기본 설정으로 모든 하위 디렉토리 스캔
./run_enhanced_batch_analysis.sh /var/log/
```

#### 2. 날짜별 로그 구조
```bash
# 날짜별 폴더가 있는 경우
./run_enhanced_batch_analysis.sh /logs/2025/09/ server.log 2 10
```

#### 3. 서비스별 로그 구조  
```bash
# 서비스별/호스트별 폴더 구조
./run_enhanced_batch_analysis.sh /logs/services/ application.log 4 30 analysis_result
```

## 📁 지원하는 디렉토리 구조

### 날짜별 구조
```
logs/
├── 2025-09-15/
│   ├── server1/
│   │   ├── application.log
│   │   └── system.log
│   └── server2/
│       └── service.log
├── 2025-09-16/
│   └── server1/
│       └── daily.log
└── 2025-09-17/
    └── application.log
```

### 서비스별 구조
```
logs/
├── web-servers/
│   ├── nginx.log
│   └── apache.log
├── databases/
│   ├── mysql.log
│   └── postgres.log
└── applications/
    ├── app1.log
    └── app2.log
```

### 복합 구조
```
logs/
├── production/
│   ├── 2025-09-15/
│   │   └── app.log
│   └── 2025-09-16/
│       └── app.log
└── staging/
    └── 2025-09-17/
        └── test.log
```

## 🔍 분석 기능

### 1. **로그 파일 자동 발견**
- 하위 디렉토리 재귀 스캔
- 다양한 로그 파일 확장자 지원 (`.log`, `.txt`, `.out`, `.syslog` 등)
- 파일 크기별 우선순위 정렬

### 2. **로그 형식 자동 감지**
- Syslog 형식 (`Sep 14 05:04:41 host1 kernel:`)
- ISO 타임스탬프 (`2025-09-17 10:15:32`)
- JSON 형식
- Apache Combined 로그
- 일반 텍스트

### 3. **카테고리별 분류**
- 디렉토리 구조를 기반으로 자동 카테고리 분류
- 카테고리별 결과 요약 및 비교

### 4. **다중 이상 탐지**
- **시간 기반 탐지**: 시간대별 패턴 비교
- **파일별 비교 탐지**: 시스템 간 차이점 분석
- **윈도우 기반 탐지**: 기존 패턴과의 차이점

### 5. **베이스라인 품질 검증** 🆕
- **자동 품질 평가**: 에러율, 템플릿 다양성, 로그 수량 검증
- **품질 필터링**: 문제 있는 baseline 파일 자동 제외
- **품질 기준**: 에러율 2% 이하, 경고율 5% 이하, 최소 템플릿 10개

### 6. **이상 로그 샘플 분석** 🆕
- **실제 로그 추출**: 이상탐지 결과에서 문제 로그 샘플 추출
- **맥락 정보**: 전후 3줄 컨텍스트와 함께 표시
- **사람 친화적**: 기술적 결과를 이해하기 쉬운 설명으로 번역

## 📊 결과 파일 구조

```
작업디렉토리/
├── COMPREHENSIVE_ANALYSIS_REPORT.md 🆕   # 📄 통합 종합 리포트 (모든 결과 + 로그 샘플)
├── ENHANCED_ANALYSIS_SUMMARY.md          # 📄 호환성 요약 리포트
├── processed_category1_file1/             # 📁 Target 파일 분석 결과 (완전 분석)
│   ├── parsed.parquet
│   ├── baseline_scores.parquet 🆕         # Baseline 이상 탐지 결과
│   ├── baseline_preview.json
│   ├── deeplog_infer.parquet
│   ├── deeplog.pth                        # DeepLog 모델 파일
│   ├── sequences.parquet
│   ├── vocab.json
│   ├── window_counts.parquet 🆕           # MS-CRED 입력 데이터
│   ├── temporal_analysis/
│   │   ├── temporal_report.md
│   │   ├── temporal_anomalies.json
│   │   └── temporal_profiles.json
│   ├── comparative_analysis/
│   │   ├── comparative_report.md
│   │   ├── comparative_anomalies.json
│   │   └── file_profiles.json
│   ├── log_samples_analysis/ 🆕          # 📁 20개 이상 로그 샘플 분석
│   │   ├── anomaly_analysis_report.md    # 사람이 읽기 쉬운 리포트
│   │   └── anomaly_samples.json          # 상세 샘플 데이터
│   └── report.md                          # CLI 생성 리포트
└── processed_category2_file2/             # 📁 Baseline 파일 결과 (전처리만)
    ├── parsed.parquet
    └── preview.json
```

## 🚨 이상 탐지 결과 해석

### 심각도 수준
- **🔴 심각 (High)**: 즉시 조치 필요
- **🟡 주의 (Medium)**: 추가 모니터링 권장
- **✅ 정상**: 이상 없음

### 탐지 유형
1. **temporal_volume_anomaly**: 시간대별 로그 볼륨 이상
2. **temporal_new_templates**: 새로운 로그 패턴 발견
3. **metric_anomaly_***: 통계적 지표 이상
4. **template_distribution_anomaly**: 로그 패턴 분포 이상

## 💡 사용 팁

### 1. **성능 최적화**
```bash
# 대용량 로그의 경우 파일 수 제한
./run_enhanced_batch_analysis.sh /huge/logs/ target.log 2 10
```

### 2. **특정 기간 분석**
```bash
# 특정 날짜 범위만 분석
./run_enhanced_batch_analysis.sh /logs/2025-09-15/ app.log 1 5
```

### 3. **카테고리별 분석**
```bash
# 특정 서비스만 분석
./run_enhanced_batch_analysis.sh /logs/web-servers/ nginx.log 2 15
```

### 4. **문제 해결**
- 전처리 실패시 로그 형식과 권한 확인
- 메모리 부족시 `--max-files` 수치 감소
- 권한 오류시 디렉토리 접근 권한 확인

## 🔧 추가 분석 도구

분석 완료 후 상세 분석을 위한 도구들:

```bash
# 상세 분석
python analyze_results.py --data-dir 작업디렉토리/processed_*

# 시각화
python visualize_results.py --data-dir 작업디렉토리/processed_*

# 개별 시간 분석
alog-detect analyze-temporal --data-dir 작업디렉토리/processed_*

# 개별 비교 분석
alog-detect analyze-comparative --target file1 --baselines file2 --baselines file3
```

## 📚 관련 문서

- [`RESULTS_GUIDE.md`](RESULTS_GUIDE.md): 결과 해석 가이드
- [`ANOMALY_DETECTION_METHODS.md`](ANOMALY_DETECTION_METHODS.md): 탐지 방법론
- [`README.md`](README.md): 전체 프로젝트 가이드

---

💡 **권장사항**: 처음 사용하시는 경우 작은 디렉토리로 테스트해보시고, 점진적으로 범위를 확장해보세요!
