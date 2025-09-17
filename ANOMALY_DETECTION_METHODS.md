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

## 🛠️ 구현 우선순위

### Phase 1: 시간 기반 확장
1. **시간대별 프로파일 탐지**
2. **요일/월별 패턴 학습**
3. **Rolling window vs Fixed window 비교**

### Phase 2: 다중 파일 분석
1. **파일 간 패턴 비교**
2. **서버 그룹별 baseline 구축**
3. **Cross-validation 방식 이상 탐지**

### Phase 3: 고급 분석
1. **시계열 분해 탐지**
2. **딥러닝 기반 sequence anomaly**
3. **Graph-based 패턴 분석**

## 📊 각 방법의 적용 시나리오

| 방법 | 적용 시나리오 | 장점 | 단점 |
|------|---------------|------|------|
| **현재 윈도우 방식** | 실시간 모니터링 | 빠른 탐지, 메모리 효율 | 장기 패턴 놓침 |
| **날짜별 프로파일** | 정기 점검, 스케줄 분석 | 정확한 baseline | 데이터 축적 필요 |
| **파일별 비교** | 시스템 간 비교, 배치 분석 | 상대적 이상 탐지 | 시간 정보 손실 |
| **시계열 분해** | 장기 트렌드 분석 | 계절성 고려 | 복잡성 증가 |
| **클러스터링** | 패턴 발견, 그룹화 | 새로운 패턴 발견 | 해석 어려움 |

## 🚀 다음 단계

각 방법을 구현하여 현재 시스템에 추가할 수 있습니다. 어떤 방법을 우선적으로 구현해보고 싶으신가요?

1. **시간대별 프로파일 탐지** - 가장 실용적
2. **파일별 비교 탐지** - 현재 상황에 바로 적용 가능
3. **시계열 분해 탐지** - 고급 분석 기능
