# 로그 이상 탐지 방법론 가이드

## 🎯 현재 구현된 방법

### 1. **시간 윈도우 기반 이상 탐지** (현재 방법)
**원리**: 연속된 로그 라인들을 고정 크기 윈도우로 나누어 분석
- **윈도우 크기**: 50개 로그 라인 (기본값)
- **슬라이딩 스트라이드**: 25개 라인씩 이동
- **분석 대상**: 각 윈도우 내의 템플릿 패턴 변화

**장점**:
- ✅ 실시간 탐지 가능
- ✅ 순서에 민감한 이상 패턴 감지
- ✅ 메모리 효율적

**단점**:
- ❌ 긴 주기의 패턴 놓칠 수 있음
- ❌ 윈도우 경계에서 이상이 분할될 수 있음

### 2. **템플릿 빈도 기반 탐지** (현재 방법)
**원리**: 각 로그 템플릿의 출현 빈도가 급격히 변화하는 구간 감지
- **EWM (Exponentially Weighted Moving)** 통계 사용
- **Z-score** 기반 이상치 탐지
- **미지 템플릿율** 계산

```python
# 현재 구현
freq_z = (counts - ewm_mean) / ewm_std  # 빈도 Z-score
unseen_rate = new_templates / total_templates  # 미지 템플릿 비율
score = 0.6 * freq_norm + 0.4 * unseen_norm  # 종합 점수
```

## 🆕 새로운 탐지 방법들

### 3. **날짜별 프로파일 기반 탐지**

**원리**: 과거 동일 시간대/요일의 패턴과 비교하여 이상 감지

**구현 아이디어**:
```python
# 시간별 정상 프로파일 학습
hourly_profiles = {
    "hour_0": {"template_dist": {...}, "volume": 1000},
    "hour_1": {"template_dist": {...}, "volume": 800},
    # ...
}

# 현재 시간대와 비교
current_hour = datetime.now().hour
expected_profile = hourly_profiles[f"hour_{current_hour}"]
deviation = compare_distributions(current_logs, expected_profile)
```

**장점**:
- ✅ 시간/요일별 자연스러운 패턴 변화 고려
- ✅ 계절성/주기성 반영 가능
- ✅ 정확한 baseline 제공

### 4. **파일별 비교 탐지**

**원리**: 여러 로그 파일 간의 패턴 차이를 분석하여 이상 감지

**구현 시나리오**:
```python
# 서버별/서비스별 로그 파일 비교
baseline_files = ["server1.log", "server2.log", "server3.log"]
target_file = "server4.log"  # 분석할 파일

# 각 파일의 템플릿 분포 추출
baseline_patterns = extract_template_distributions(baseline_files)
target_patterns = extract_template_distributions([target_file])

# 차이점 분석
anomalies = find_distribution_anomalies(target_patterns, baseline_patterns)
```

**장점**:
- ✅ 같은 환경의 다른 시스템과 비교
- ✅ 시스템별 특이사항 발견
- ✅ 일괄 분석으로 빠른 판단

### 5. **시계열 분해 기반 탐지**

**원리**: 로그 볼륨/패턴을 트렌드, 계절성, 잔차로 분해하여 분석

```python
# STL Decomposition 활용
from statsmodels.tsa.seasonal import STL

# 시간별 로그 볼륨 시계열
hourly_volumes = df.groupby(df['timestamp'].dt.hour).size()

# 분해: 트렌드 + 계절성 + 잔차
decomposition = STL(hourly_volumes).fit()
residuals = decomposition.resid

# 잔차에서 이상치 탐지
anomalies = detect_outliers_in_residuals(residuals)
```

### 6. **클러스터링 기반 탐지**

**원리**: 로그 윈도우들을 클러스터링하여 outlier 윈도우 탐지

```python
# 각 윈도우를 특성 벡터로 변환
window_features = [
    template_distribution,  # 템플릿 분포
    timestamp_features,     # 시간 특성
    volume_metrics,         # 볼륨 지표
    sequence_patterns       # 순서 패턴
]

# 클러스터링
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=0.3).fit(window_features)

# 노이즈(-1) 클러스터가 이상
anomaly_windows = clusters.labels_ == -1
```

## ✅ 현재 구현 상태

### Phase 1: 시간 기반 확장 (구현 완료 ✅)
1. **✅ 시간대별 프로파일 탐지** - `temporal_anomaly_detector.py`
2. **✅ 요일/월별 패턴 학습** - 시간 기반 분석 포함
3. **✅ Rolling window vs Fixed window 비교** - 다양한 윈도우 크기 지원

### Phase 2: 다중 파일 분석 (구현 완료 ✅)
1. **✅ 파일 간 패턴 비교** - `comparative_anomaly_detector.py`
2. **✅ 서버 그룹별 baseline 구축** - `enhanced_batch_analyzer.py`
3. **✅ Cross-validation 방식 이상 탐지** - 다중 baseline 비교

### Phase 3: 고급 분석 (구현 완료 ✅)
1. **✅ Baseline 이상 탐지** - Window 기반 템플릿 빈도 분석
2. **✅ 딥러닝 기반 sequence anomaly** - DeepLog LSTM 모델
3. **✅ MS-CRED 입력 생성** - Multi-Scale 윈도우 카운트 벡터

### Phase 4: 사용성 향상 (구현 완료 ✅)
1. **✅ 외부 Target 파일 지원** - 다른 디렉토리 파일을 Target으로 지정
2. **✅ Baseline 품질 검증** - 자동 품질 평가 및 필터링
3. **✅ 로그 샘플 분석** - 실제 이상 로그 20개 샘플 추출 및 설명
4. **✅ 종합 리포트 통합** - 모든 결과를 하나의 리포트로 통합
5. **✅ Target 검증 강화** - 잘못된 Target 지정 시 안전한 에러 처리

## 🆕 최신 추가 기능 (2025-09-20)

### 7. **베이스라인 품질 검증**
**원리**: 베이스라인 로그 자체의 품질을 평가하여 문제 있는 파일 필터링

**검증 기준**:
- **에러율**: 에러/경고 키워드가 포함된 로그 비율
- **템플릿 다양성**: 발견된 고유 템플릿 개수
- **로그 수량**: 분석에 충분한 로그 라인 수

**구현**:
```python
# enhanced_batch_analyzer.py 내 _validate_baseline_quality 메서드
if error_rate > 0.2:  # 20% 이상 에러 로그
    quality_issues.append(f"높은 에러율: {error_rate:.2%}")
if unique_templates < 10:  # 템플릿 부족
    quality_issues.append(f"템플릿 부족: {unique_templates}개")
```

### 8. **로그 샘플 분석**
**원리**: 이상 탐지 결과에서 실제 문제 로그를 추출하여 사람이 이해할 수 있는 형태로 제공

**특징**:
- **타입별 20개 샘플**: Baseline, DeepLog, 시간 기반, 비교 분석별로 최대 20개씩
- **맥락 정보**: 전후 3줄 컨텍스트 제공
- **자동 설명**: 각 이상의 원인과 심각도 자동 분석

**구현**:
```python
# log_sample_analyzer.py
self.max_samples_per_type = 20  # 타입별 최대 샘플 수
self.context_lines = 3  # 전후 맥락 라인 수
```

### 9. **외부 Target 파일 지원**
**원리**: 베이스라인 디렉토리와 다른 위치의 파일을 Target으로 지정 가능

**사용법**:
```bash
# 다른 디렉토리의 파일을 Target으로 지정
./run_enhanced_batch_analysis.sh /var/log/baseline/ /var/log/target/problem.log
```

**구현**:
```python
# enhanced_batch_analyzer.py 내 select_target_and_baselines 메서드
target_path = Path(target_file)
if target_path.exists() and target_path.is_file():
    # 외부 파일을 Target으로 사용
    target = (target_path, target_path.parent.name)
    baselines = log_files  # 모든 발견된 파일을 Baseline으로
```

**구현**: `baseline_validator.py`
```python
# 품질 평가 지표
quality_metrics = {
    'error_rate': error_logs / total_logs,
    'warning_rate': warning_logs / total_logs,
    'template_diversity': unique_templates,
    'log_volume': total_logs,
    'rare_template_ratio': rare_templates / unique_templates
}

# 품질 임계값
QUALITY_THRESHOLDS = {
    'max_error_rate': 0.02,      # 2% 이하
    'max_warning_rate': 0.05,    # 5% 이하
    'min_templates': 10,         # 최소 10개 템플릿
    'min_logs': 100,            # 최소 100개 로그
    'max_rare_ratio': 0.3       # 희귀 템플릿 30% 이하
}
```

**장점**:
- ✅ 이상탐지 정확도 향상
- ✅ 자동 품질 필터링
- ✅ 신뢰할 수 있는 baseline 보장

### 8. **이상 로그 샘플 분석**
**원리**: 이상탐지 결과에서 실제 문제 로그를 추출하여 사람이 이해하기 쉽게 제공

**구현**: `log_sample_analyzer.py`
```python
# 샘플 추출 및 분석
def extract_anomaly_samples(anomaly_results, parsed_logs):
    samples = []
    for anomaly in anomaly_results:
        # 이상 구간의 실제 로그 추출
        log_sample = get_log_context(parsed_logs, anomaly.line_range)
        
        # 전후 맥락 추가
        context = get_surrounding_context(parsed_logs, anomaly.line_range, 3)
        
        # 이상 원인 분석
        explanation = analyze_anomaly_pattern(log_sample, anomaly.type)
        
        samples.append({
            'sample': log_sample,
            'context': context,
            'explanation': explanation,
            'severity': classify_severity(anomaly)
        })
    
    return samples
```

**장점**:
- ✅ 실제 문제 로그 확인 가능
- ✅ 전후 맥락으로 상황 파악
- ✅ 사람 친화적인 설명 제공

## 📊 각 방법의 적용 시나리오

| 방법 | 적용 시나리오 | 장점 | 단점 |
|------|---------------|------|------|
| **현재 윈도우 방식** | 실시간 모니터링 | 빠른 탐지, 메모리 효율 | 장기 패턴 놓침 |
| **날짜별 프로파일** | 정기 점검, 스케줄 분석 | 정확한 baseline | 데이터 축적 필요 |
| **파일별 비교** | 시스템 간 비교, 배치 분석 | 상대적 이상 탐지 | 시간 정보 손실 |
| **시계열 분해** | 장기 트렌드 분석 | 계절성 고려 | 복잡성 증가 |
| **클러스터링** | 패턴 발견, 그룹화 | 새로운 패턴 발견 | 해석 어려움 |
| **베이스라인 품질검증** | 배치 분석, 품질 관리 | 정확도 향상 | 초기 설정 필요 |
| **로그 샘플 분석** | 문제 진단, 사후 분석 | 직관적 이해 | 추가 처리 시간 |

## 🎯 통합 솔루션

현재 시스템은 **모든 주요 방법들이 통합**되어 `run_enhanced_batch_analysis.sh` 하나로 실행 가능합니다:

1. **✅ 자동 베이스라인 품질 검증**
2. **✅ 시간 기반 이상탐지**  
3. **✅ 파일별 비교 이상탐지**
4. **✅ 딥러닝 기반 시퀀스 분석**
5. **✅ 이상 로그 샘플 추출 및 분석**

## 🚀 사용 방법

```bash
# 모든 기능을 한번에 실행
./run_enhanced_batch_analysis.sh /path/to/logs/

# 결과 확인 (중요도 순)
cat enhanced_analysis_*/ENHANCED_ANALYSIS_SUMMARY.md
cat enhanced_analysis_*/processed_*/log_samples_analysis/anomaly_analysis_report.md
```
