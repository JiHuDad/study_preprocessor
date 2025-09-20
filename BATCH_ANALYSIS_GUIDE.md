# 🚀 배치 로그 분석 가이드

## 📋 개요

여러 로그 파일을 한 번에 분석하고 비교할 수 있는 배치 분석 시스템입니다.

## 🆚 두 가지 배치 분석 도구

### 1. **기본 배치 분석기** (`batch_log_analyzer.py`)
단일 디렉토리의 로그 파일들을 분석합니다.

```bash
# 기본 사용법
./run_batch_analysis.sh /path/to/logs/

# Target 파일 지정
./run_batch_analysis.sh /path/to/logs/ server1.log my_analysis
```

### 2. **향상된 배치 분석기** (`enhanced_batch_analyzer.py`) ⭐ **추천**
하위 디렉토리를 재귀적으로 스캔하여 날짜별/카테고리별 구조를 지원합니다.

```bash
# 향상된 분석 (하위 디렉토리 포함)
./run_enhanced_batch_analysis.sh /path/to/logs/

# 세부 옵션 지정
./run_enhanced_batch_analysis.sh /path/to/logs/ target.log 3 20 result_dir
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
├── ENHANCED_ANALYSIS_SUMMARY.md          # 📄 종합 요약 리포트
├── processed_category1_file1/             # 📁 파일별 분석 결과
│   ├── parsed.parquet
│   ├── baseline_scores.parquet
│   ├── deeplog_infer.parquet
│   ├── temporal_analysis/
│   │   ├── temporal_report.md
│   │   └── temporal_anomalies.json
│   ├── comparative_analysis/
│   │   ├── comparative_report.md
│   │   └── comparative_anomalies.json
│   └── log_samples_analysis/ 🆕          # 📁 이상 로그 샘플 분석
│       ├── anomaly_analysis_report.md    # 사람이 읽기 쉬운 리포트
│       └── anomaly_samples.json          # 상세 샘플 데이터
└── processed_category2_file2/             # 📁 다른 파일 분석 결과
    └── ...
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
python temporal_anomaly_detector.py --data-dir 작업디렉토리/processed_*

# 개별 비교 분석
python comparative_anomaly_detector.py --target file1 --baselines file2 file3
```

## 📚 관련 문서

- [`RESULTS_GUIDE.md`](RESULTS_GUIDE.md): 결과 해석 가이드
- [`ANOMALY_DETECTION_METHODS.md`](ANOMALY_DETECTION_METHODS.md): 탐지 방법론
- [`README.md`](README.md): 전체 프로젝트 가이드

---

💡 **권장사항**: 처음 사용하시는 경우 작은 디렉토리로 테스트해보시고, 점진적으로 범위를 확장해보세요!
