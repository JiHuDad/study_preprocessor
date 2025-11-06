# DeepLog 리포트 개선: 예측값 vs 실제값 표시

## 개요

DeepLog 이상 탐지 리포트에 **모델이 예측한 값**과 **실제 탐지된 값**을 표시하는 기능이 추가되었습니다.

이 기능을 통해 다음을 더 명확하게 이해할 수 있습니다:
- **왜 이상으로 탐지되었는지**: 모델이 예상한 패턴과 실제 발생한 패턴의 차이
- **예측 실패 원인**: 모델이 학습하지 못한 새로운 패턴 또는 드문 패턴
- **패턴 불일치 정도**: Top-K 예측에 포함되지 않은 정도

---

## 주요 변경사항

### 1. DeepLog 추론 함수 개선

**파일**: `anomaly_log_detector/builders/deeplog.py`

`infer_deeplog_topk()` 함수가 다음 정보를 추가로 반환합니다:

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| `predicted_top1` | 모델이 가장 높은 확률로 예측한 템플릿 인덱스 | `5` |
| `predicted_top2` | 두 번째로 높은 확률의 템플릿 인덱스 | `12` |
| `predicted_top3` | 세 번째로 높은 확률의 템플릿 인덱스 | `3` |
| `target_template` | 실제 발생한 템플릿 문자열 (vocab 사용 시) | `"Error: <PATH> not found"` |
| `predicted_templates` | 모델이 예측한 Top-K 템플릿 문자열들 | `"System started \| User logged in \| Connection closed"` |

### 2. 리포트 생성 함수 개선

**파일**: `anomaly_log_detector/cli.py`

리포트에 "예측 실패 상위 샘플" 섹션이 추가되어 다음 정보를 표시합니다:

#### vocab 사용 시 (권장):
```markdown
### 🔍 예측 실패 상위 샘플

#### 샘플 1
| 항목 | 내용 |
|------|------|
| **실제 발생** | `Error: Connection timeout` |
| **모델 예측 (Top-K)** | `System started | User logged in | Connection closed` |
| **분석** | 모델이 예측한 패턴과 다른 로그가 발생하여 이상으로 탐지되었습니다. |
```

#### vocab 미사용 시:
```markdown
| 샘플 | 실제 템플릿 인덱스 | 예측 Top-1 | 예측 Top-2 | 예측 Top-3 |
|------|-------------------|-----------|-----------|------------|
| #0 | 15 | 5 | 12 | 3 |
| #1 | 23 | 7 | 14 | 9 |
```

### 3. CLI 인터페이스 개선

**변경 내용**: `deeplog-infer` 명령어에 `--vocab` 옵션 추가

```bash
alog-detect deeplog-infer \
  --seq sequences.parquet \
  --model deeplog.pth \
  --vocab vocab.json \  # 📌 새로 추가된 옵션
  --k 3
```

---

## 사용 방법

### 방법 1: 전체 파이프라인 실행

전체 파이프라인 스크립트는 자동으로 vocab을 사용합니다:

```bash
# uv 환경
./scripts/run_full_pipeline.sh path/to/your_logs.log

# pip 환경
./scripts/run_full_pipeline_pip.sh path/to/your_logs.log
```

**자동으로 생성되는 리포트 위치**: `data/processed/{로그파일명}/report.md`

### 방법 2: 수동으로 DeepLog 추론 실행

#### 2.1 vocab과 함께 추론 (권장)

```bash
# 1. DeepLog 입력 생성 (vocab.json 포함)
alog-detect build-deeplog \
  --parsed data/processed/parsed.parquet \
  --out-dir data/processed

# 2. 모델 학습
alog-detect deeplog-train \
  --seq data/processed/sequences.parquet \
  --vocab data/processed/vocab.json \
  --out models/deeplog.pth

# 3. vocab과 함께 추론 실행 (예측/실제 템플릿 문자열 포함)
alog-detect deeplog-infer \
  --seq data/processed/sequences.parquet \
  --model models/deeplog.pth \
  --vocab data/processed/vocab.json \  # ✅ vocab 전달
  --k 3

# 4. 리포트 생성
alog-detect report --processed-dir data/processed
```

**결과**: 리포트에 실제 템플릿 문자열이 표시됩니다.

#### 2.2 vocab 없이 추론 (레거시)

```bash
alog-detect deeplog-infer \
  --seq data/processed/sequences.parquet \
  --model models/deeplog.pth \
  --k 3
```

**결과**: 리포트에 템플릿 인덱스만 표시됩니다.

---

## 리포트 예시

### vocab 사용 시 (권장)

```markdown
## 🧠 DeepLog 이상 탐지 (딥러닝 LSTM)

**예측 실패율**: 15.2% (전체 1000개 중 152개 실패)

**해석**: ⚠️ **주의**: 로그 패턴이 다소 복잡하거나 학습 데이터가 부족할 수 있습니다.

### 🔍 예측 실패 상위 샘플

모델이 예측하지 못한 패턴들입니다. 각 샘플은 모델의 예측값과 실제 발생한 값을 보여줍니다.

#### 샘플 1

| 항목 | 내용 |
|------|------|
| **실제 발생** | `Error: Database connection timeout after <NUM> seconds` |
| **모델 예측 (Top-K)** | `System started successfully | User <ID> authenticated | Connection established to <IP>` |
| **분석** | 모델이 예측한 패턴과 다른 로그가 발생하여 이상으로 탐지되었습니다. |

#### 샘플 2

| 항목 | 내용 |
|------|------|
| **실제 발생** | `CRITICAL: Out of memory error in process <NAME>` |
| **모델 예측 (Top-K)** | `Processing request from <IP> | User <ID> logged in | System health check passed` |
| **분석** | 모델이 예측한 패턴과 다른 로그가 발생하여 이상으로 탐지되었습니다. |

#### 샘플 3

| 항목 | 내용 |
|------|------|
| **실제 발생** | `WARNING: Disk usage above <NUM>% threshold` |
| **모델 예측 (Top-K)** | `Backup completed successfully | System started | User session created` |
| **분석** | 모델이 예측한 패턴과 다른 로그가 발생하여 이상으로 탐지되었습니다. |
```

### vocab 미사용 시

```markdown
## 🧠 DeepLog 이상 탐지 (딥러닝 LSTM)

**예측 실패율**: 15.2% (전체 1000개 중 152개 실패)

**해석**: ⚠️ **주의**: 로그 패턴이 다소 복잡하거나 학습 데이터가 부족할 수 있습니다.

### 🔍 예측 실패 상위 샘플

| 샘플 | 실제 템플릿 인덱스 | 예측 Top-1 | 예측 Top-2 | 예측 Top-3 |
|------|-------------------|-----------|-----------|------------|
| #23 | 15 | 5 | 12 | 3 |
| #47 | 23 | 7 | 14 | 9 |
| #89 | 31 | 2 | 8 | 11 |
| #112 | 8 | 17 | 22 | 6 |
| #156 | 27 | 4 | 19 | 13 |

**참고**: vocab.json을 사용하여 추론하면 실제 템플릿 문자열을 볼 수 있습니다.

```bash
alog-detect deeplog-infer --seq sequences.parquet --model model.pth --vocab vocab.json
```
```

---

## 기술적 세부사항

### vocab 구조

vocab.json은 템플릿 문자열과 인덱스의 매핑입니다:

```json
{
  "Error: Database connection timeout after <NUM> seconds": 0,
  "System started successfully": 1,
  "User <ID> authenticated": 2,
  "Connection established to <IP>": 3,
  "CRITICAL: Out of memory error in process <NAME>": 4
}
```

### 추론 결과 DataFrame 구조

vocab 사용 시 `deeplog_infer.parquet`에 저장되는 데이터:

| idx | target | in_topk | predicted_top1 | predicted_top2 | predicted_top3 | target_template | predicted_templates |
|-----|--------|---------|----------------|----------------|----------------|-----------------|---------------------|
| 0 | 1 | True | 1 | 2 | 3 | "System started..." | "System started... \| User authenticated \| Connection..." |
| 23 | 15 | False | 5 | 12 | 3 | "Error: Database..." | "System started... \| User logged in \| Connection closed" |
| 47 | 23 | False | 7 | 14 | 9 | "CRITICAL: Out..." | "Processing request... \| User logged in \| System health..." |

---

## FAQ

### Q1: vocab을 사용하지 않으면 어떻게 되나요?
**A**: 리포트에 템플릿 인덱스만 표시됩니다. 인덱스만으로는 어떤 로그 패턴인지 파악하기 어려우므로 **vocab 사용을 권장**합니다.

### Q2: 기존 추론 결과를 vocab과 함께 다시 생성할 수 있나요?
**A**: 네, 다음 명령어로 재생성할 수 있습니다:
```bash
alog-detect deeplog-infer \
  --seq data/processed/sequences.parquet \
  --model models/deeplog.pth \
  --vocab data/processed/vocab.json \
  --k 3
```

### Q3: Top-K 값을 변경할 수 있나요?
**A**: 네, `--k` 옵션으로 변경할 수 있습니다:
```bash
alog-detect deeplog-infer \
  --seq sequences.parquet \
  --model deeplog.pth \
  --vocab vocab.json \
  --k 5  # Top-5로 변경
```
Top-K 값이 커질수록:
- 더 많은 예측 후보를 고려 (더 관대한 판정)
- 예측 실패율이 낮아짐
- 하지만 미세한 이상을 놓칠 수 있음

### Q4: 예측 실패율이 높으면 어떻게 해야 하나요?
**A**: 다음 조치를 고려하세요:

1. **모델 재학습**: 더 많은 정상 로그 데이터로 재학습
   ```bash
   alog-detect deeplog-train \
     --seq sequences.parquet \
     --vocab vocab.json \
     --out models/deeplog.pth \
     --epochs 10  # 에폭 증가
   ```

2. **Top-K 값 증가**: 더 관대한 판정 기준 사용
   ```bash
   alog-detect deeplog-infer \
     --seq sequences.parquet \
     --model deeplog.pth \
     --vocab vocab.json \
     --k 5  # 3 → 5로 증가
   ```

3. **데이터 품질 확인**: 학습 데이터에 정상 패턴이 충분히 포함되어 있는지 확인

### Q5: 리포트에 표시되는 샘플 개수를 변경할 수 있나요?
**A**: 현재는 상위 5개 샘플이 표시됩니다. 소스 코드를 수정하여 변경할 수 있습니다:
```python
# anomaly_log_detector/cli.py:481
for idx, row in violations.head(5).iterrows():  # 5 → 원하는 개수로 변경
```

---

## 관련 명령어 요약

```bash
# 전체 파이프라인 (vocab 자동 사용)
./scripts/run_full_pipeline.sh your_logs.log

# DeepLog 입력 생성 (vocab 포함)
alog-detect build-deeplog --parsed parsed.parquet --out-dir output/

# DeepLog 학습
alog-detect deeplog-train \
  --seq sequences.parquet \
  --vocab vocab.json \
  --out deeplog.pth

# DeepLog 추론 (vocab 사용 - 권장)
alog-detect deeplog-infer \
  --seq sequences.parquet \
  --model deeplog.pth \
  --vocab vocab.json \
  --k 3

# 리포트 생성
alog-detect report --processed-dir data/processed

# 합성 데이터로 테스트
./scripts/demo/demo_train_inference_workflow.sh
```

---

## 다음 단계

1. **✅ 리포트 확인**: 생성된 `report.md`에서 예측 실패 샘플 확인
2. **🔍 패턴 분석**: 어떤 패턴이 자주 예측 실패하는지 분석
3. **📊 모델 개선**: 필요 시 데이터 추가 또는 파라미터 조정
4. **🚀 모니터링**: 실제 운영 환경에서 지속적으로 모니터링

---

## 참고 문서

- **전체 파이프라인**: [README.md](../README.md)
- **CLI 명령어**: `alog-detect --help`
- **DeepLog 추론 옵션**: `alog-detect deeplog-infer --help`
- **합성 데이터 생성**: [SYNTHETIC_DATA_GUIDE.md](SYNTHETIC_DATA_GUIDE.md)
- **리포트 개선 예시**: [REPORT_IMPROVEMENT_EXAMPLE.md](REPORT_IMPROVEMENT_EXAMPLE.md)
