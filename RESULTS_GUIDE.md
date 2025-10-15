# 로그 이상 탐지 결과 해석 가이드

## 🎯 개요

이 문서는 log anomaly detection pipeline의 결과를 해석하는 방법을 설명합니다.

## 🆕 **최신 업데이트 (2025-09-20)**

### ✨ **새로운 결과 파일들:**
- **📄 `COMPREHENSIVE_ANALYSIS_REPORT.md`**: 모든 분석 결과와 로그 샘플을 통합한 종합 리포트
- **📊 `log_samples_analysis/`**: 실제 이상 로그 20개 샘플과 맥락 분석
- **🛡️ Baseline 품질 검증**: 자동 필터링된 고품질 Baseline 결과
- **🎯 외부 Target 지원**: 다른 디렉토리 파일의 분석 결과

## 📁 결과 파일 구조

### 🆕 **통합 분석 결과** (enhanced_batch_analyzer.py)
```
analysis_directory/
├── COMPREHENSIVE_ANALYSIS_REPORT.md 🆕  # 통합 종합 리포트 (모든 결과 + 로그 샘플)
├── ENHANCED_ANALYSIS_SUMMARY.md         # 호환성 요약 리포트
├── processed_target_file/               # Target 파일 완전 분석 결과
│   ├── parsed.parquet                   # 파싱된 원본 로그 데이터
│   ├── baseline_scores.parquet 🆕       # Baseline 이상 탐지 결과
│   ├── baseline_preview.json           # Baseline 분석 미리보기
│   ├── sequences.parquet               # 템플릿 시퀀스 데이터
│   ├── vocab.json                      # 템플릿 ID → 인덱스 매핑
│   ├── deeplog_infer.parquet          # DeepLog 딥러닝 모델 결과
│   ├── deeplog.pth                    # 학습된 DeepLog 모델
│   ├── window_counts.parquet 🆕       # MS-CRED 입력 데이터
│   ├── temporal_analysis/             # 시간 기반 이상 탐지
│   │   ├── temporal_report.md
│   │   ├── temporal_anomalies.json
│   │   └── temporal_profiles.json
│   ├── comparative_analysis/          # 파일 간 비교 분석
│   │   ├── comparative_report.md
│   │   ├── comparative_anomalies.json
│   │   └── file_profiles.json
│   ├── log_samples_analysis/ 🆕       # 이상 로그 샘플 분석 (20개)
│   │   ├── anomaly_analysis_report.md # 사람이 읽기 쉬운 리포트
│   │   └── anomaly_samples.json       # 상세 샘플 데이터
│   └── report.md                      # CLI 생성 리포트
└── processed_baseline_files/          # Baseline 파일들 (전처리만)
    ├── parsed.parquet
    └── preview.json
```

### **단일 파일 분석 결과**
```
data/processed/  (또는 지정된 출력 디렉토리)
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

## 🔧 추가 분석 도구

### 기본 분석 도구
```bash
# 상세 분석
python analyze_results.py --data-dir 출력디렉토리

# 시각화 및 요약
python visualize_results.py --data-dir 출력디렉토리
```

### 🕐 시간 기반 이상 탐지
과거 동일 시간대/요일의 패턴과 비교하여 이상을 탐지합니다.

```bash
# 시간 패턴 기반 분석
study-preprocess analyze-temporal --data-dir data/processed

# 결과 확인
cat data/processed/temporal_analysis/temporal_report.md
```

**분석 내용**:
- 시간대별 정상 프로파일 학습 (0-23시)
- 요일별 패턴 분석
- 볼륨 이상 감지 (예상 대비 50% 이상 차이)
- 새로운/누락된 템플릿 탐지

### 📊 파일별 비교 이상 탐지  
여러 로그 파일 간의 패턴 차이를 분석하여 이상을 탐지합니다.

```bash
# 파일 간 비교 분석
study-preprocess analyze-comparative \
  --target data/server1/parsed.parquet \
  --baselines data/server2/parsed.parquet --baselines data/server3/parsed.parquet

# 결과 확인
cat data/server1/comparative_analysis/comparative_report.md
```

**분석 내용**:
- 로그 볼륨, 템플릿 수, 에러율 비교
- 템플릿 분포 KL Divergence 계산
- 고유/누락 템플릿 분석
- Z-score 기반 이상치 탐지

### 🆕 이상 로그 샘플 분석 (NEW!)
실제 문제가 되는 로그들을 사람이 읽기 쉬운 형태로 추출하고 분석합니다.

```bash
# 로그 샘플 분석 (독립 실행)
study-preprocess analyze-samples --processed-dir data/processed

# 기존 리포트에 샘플 분석 포함
study-preprocess report --processed-dir data/processed --with-samples

# 결과 확인
cat data/processed/log_samples_analysis/anomaly_analysis_report.md
```

**분석 내용**:
- 실제 이상 로그들의 원문과 전후 맥락
- 왜 이상으로 판단되었는지 구체적 설명
- 이상 유형별 패턴 분석 및 권고사항

### 🔍 베이스라인 품질 검증 (NEW!)
베이스라인 로그의 품질을 평가하여 신뢰할 수 있는 분석을 보장합니다.

```bash
# 베이스라인 품질 검증
study-preprocess validate-baseline /path/to/baseline/logs/ --output-dir validation_result

# 자동화 스크립트
./run_baseline_validation.sh /path/to/baseline/logs/
```

**검증 지표**:
- 에러율 (2% 이하 권장)
- 경고율 (5% 이하 권장) 
- 템플릿 다양성 (최소 10개)
- 로그 볼륨 (최소 100개)
- 희귀 템플릿 비율 (30% 이하)

## 🆕 **로그 샘플 분석** (`log_samples_analysis/`)

### **📄 anomaly_analysis_report.md**
사람이 읽기 쉬운 형태의 이상 로그 샘플 분석 리포트

**구조**:
- **📊 분석 요약**: 전체 이상 개수 및 분포
- **📈 Baseline 이상 샘플**: 윈도우 기반 이상 탐지 결과 (최대 20개)
- **🧠 DeepLog 이상 샘플**: LSTM 모델 예측 실패 샘플 (최대 20개)
- **🕐 시간 기반 이상 샘플**: 시간 패턴 이상 (최대 20개)
- **📊 비교 분석 이상 샘플**: 파일 간 비교 이상 (최대 20개)

**샘플 형태**:
```markdown
**윈도우 시작라인 825** (점수: 0.638)
```
Line 825: 2025-09-20 11:10:16 hostname sshd[6623]: CRITICAL: Out of memory error...
Line 826: 2025-09-20 11:10:23 hostname systemd[1456]: CRITICAL: Out of memory error...
Line 827: 2025-09-20 11:10:30 hostname kernel[2166]: ERROR: Authentication failed...
```

**🔍 분석**: 이 윈도우는 여러 CRITICAL 레벨 오류가 연속으로 발생하여 이상으로 감지되었습니다.
```

### **📊 anomaly_samples.json**
상세한 샘플 데이터 (프로그래밍 방식 접근용)

**구조**:
```json
{
  "baseline_anomaly": {
    "anomaly_count": 3,
    "analyzed_count": 3,
    "samples": [...]
  },
  "deeplog_anomaly": {
    "anomaly_count": 111,
    "analyzed_count": 10,
    "samples": [...]
  }
}
```

### CLI 도구
```bash
# 🆕 로그 샘플 분석 (단독 실행)
study-preprocess analyze-samples <processed_dir> --max-samples 20 --context-lines 3

# 🆕 리포트 생성 (로그 샘플 포함)
study-preprocess report <processed_dir> --with-samples

# 기본 CLI 도구
.venv/bin/python -m study_preprocessor.cli detect --help
.venv/bin/python -m study_preprocessor.cli eval --help
```
