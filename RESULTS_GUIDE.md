# 로그 이상 탐지 결과 해석 가이드

## 🎯 개요

이 문서는 log anomaly detection pipeline의 결과를 해석하는 방법을 설명합니다.

## 📁 결과 파일 구조

```
data/processed/  (또는 다른 데이터셋 디렉토리)
├── parsed.parquet          # 파싱된 원본 로그 데이터
├── sequences.parquet       # 템플릿 시퀀스 데이터
├── vocab.json             # 템플릿 ID → 인덱스 매핑
├── preview.json           # 로그 샘플 미리보기
├── baseline_scores.parquet # Baseline 이상 탐지 결과
├── deeplog_infer.parquet  # DeepLog 딥러닝 모델 결과
└── report.md              # 자동 생성된 요약 리포트
```

## 🔍 주요 결과 해석

### 1. Baseline Anomaly Detection

**파일**: `baseline_scores.parquet`

| 컬럼 | 설명 |
|------|------|
| `window_start_line` | 윈도우 시작 라인 번호 |
| `unseen_rate` | 윈도우 내 미지 템플릿 비율 (0.0-1.0) |
| `score` | 이상 점수 (높을수록 이상) |
| `is_anomaly` | 이상 여부 (True/False) |

**해석**:
- `unseen_rate`가 높으면 → 새로운/드문 로그 패턴 발견
- `score`가 높으면 → 이상 가능성 높음
- `is_anomaly=True` → 실제 이상으로 판정된 윈도우

### 2. DeepLog 딥러닝 모델

**파일**: `deeplog_infer.parquet`

| 컬럼 | 설명 |
|------|------|
| `idx` | 시퀀스 인덱스 |
| `target` | 예상되는 다음 템플릿 ID |
| `in_topk` | 예상이 top-k 내에 있는지 여부 |

**해석**:
- `in_topk=False` → 예측 실패 (이상 가능성)
- 위반율이 높으면 → 로그 패턴이 예측하기 어려움

## 📊 성능 지표

### 이상률 (Anomaly Rate)
- **낮음 (<5%)**: 정상적인 시스템
- **중간 (5-20%)**: 일부 비정상 패턴 존재
- **높음 (>20%)**: 시스템에 문제 가능성

### DeepLog 위반율 (Violation Rate)
- **낮음 (<20%)**: 예측 가능한 로그 패턴
- **중간 (20-50%)**: 일부 예측 어려운 패턴
- **높음 (>50%)**: 매우 복잡한/비정상적인 패턴

## 🛠️ 결과 확인 방법

### 1. 자동 분석 도구 사용
```bash
# 기본 데이터셋 분석
python analyze_results.py

# 특정 데이터셋 분석
python analyze_results.py --data-dir data/test_synthetic
```

### 2. 수동 데이터 확인
```python
import pandas as pd

# Baseline 결과 확인
baseline = pd.read_parquet('data/processed/baseline_scores.parquet')
print(baseline[baseline['is_anomaly'] == True])

# DeepLog 결과 확인
deeplog = pd.read_parquet('data/processed/deeplog_infer.parquet')
violations = deeplog[deeplog['in_topk'] == False]
print(f"위반율: {len(violations)/len(deeplog)*100:.1f}%")
```

## 🚨 이상 탐지 결과 이해하기

### Case 1: 정상 시스템
- Baseline 이상률: <5%
- DeepLog 위반율: <20%
- → 시스템이 정상적으로 작동

### Case 2: 일부 비정상
- Baseline 이상률: 5-20%
- DeepLog 위반율: 20-50%
- → 조사가 필요한 일부 패턴 존재

### Case 3: 문제 상황
- Baseline 이상률: >20%
- DeepLog 위반율: >50%
- → 시스템에 심각한 문제 가능성

## 📈 실제 사용 예시

### 예시 결과 해석:
```
🚨 Baseline Anomaly Detection 결과
- 발견된 이상 윈도우: 1개 (전체의 25.0%)
- 윈도우 시작라인 0: 점수=0.400, 미지템플릿비율=1.000

🧠 DeepLog 딥러닝 모델 결과  
- 위반율: 50.0%
```

**해석**: 
- 25%의 높은 이상률 → 시스템에 비정상적인 패턴
- 100% 미지 템플릿 → 완전히 새로운 로그 패턴 발견
- 50% DeepLog 위반율 → 예측하기 어려운 복잡한 패턴

## 💡 팁

1. **정기적 모니터링**: 위반율/이상률의 시간별 변화 추적
2. **임계값 조정**: 환경에 맞게 이상 탐지 임계값 조정
3. **패턴 분석**: 이상으로 판정된 실제 로그 내용 검토
4. **False Positive 관리**: 정상이지만 이상으로 판정된 케이스 분석

## 🔧 추가 분석

더 자세한 분석이 필요한 경우:

```bash
# CLI 도구로 상세 분석
.venv/bin/python -m study_preprocessor.cli detect --help
.venv/bin/python -m study_preprocessor.cli eval --help
```
