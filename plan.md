### 계획(Plan): 로그 이상탐지 진행 전략

#### 목표
- 전처리 산출물을 바탕으로 DeepLog/MSCRED 기반 이상탐지를 단계적으로 구현/평가
- 단기: 베이스라인(룰/통계) 기반 즉시 탐지, 중기: DeepLog, 장기: MS-CRED

#### 단계별 전략
- 베이스라인(즉시 가능)
  - 새 템플릿 비율 급증 감지(윈도우별 unseen rate)
  - 템플릿 빈도 급변(EWMA z-score/Poisson)
- DeepLog
  - 입력: `sequences.parquet`, `vocab.json`
  - 학습: LSTM, seq_len≈50, top-k=3~5
  - 추론: 다음 이벤트 top-k 누락 비율을 점수화, POT/퍼센타일 임계치
- MS-CRED
  - 입력: `window_counts.parquet`(확장: 공발생/상관맵)
  - 학습: ConvLSTM AE, 재구성 오차 기반 점수 및 임계치

#### 실행 커맨드(현재)
```
# DeepLog/MSCRED 입력 생성
uv run python -c "from anomaly_log_detector.builders.deeplog import build_deeplog_inputs as b; b('data/processed/parsed.parquet','data/processed')"
uv run python -c "from anomaly_log_detector.builders.mscred import build_mscred_window_counts as b; b('data/processed/parsed.parquet','data/processed',window_size=50,stride=25)"

# 베이스라인 이상탐지
uv run alog-detect detect \
  --parsed data/processed/parsed.parquet \
  --out-dir data/processed \
  --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.8

# DeepLog 학습/추론
uv run alog-detect build-deeplog --parsed data/processed/parsed.parquet --out-dir data/processed
uv run alog-detect deeplog-train --seq data/processed/sequences.parquet --vocab data/processed/vocab.json --out .cache/deeplog.pth --seq-len 5 --epochs 2
uv run alog-detect deeplog-infer --seq data/processed/sequences.parquet --model .cache/deeplog.pth --k 3

# MS-CRED 입력 CLI
uv run alog-detect build-mscred --parsed data/processed/parsed.parquet --out-dir data/processed --window-size 50 --stride 25

# 리포트 생성
uv run alog-detect report --processed-dir data/processed
```

#### 의존성/구성
- uv, Python 3.11+, drain3/pandas/pyarrow/orjson/click/tqdm/regex, torch
- 마스킹/Drain3 파라미터는 `anomaly_log_detector/preprocess.py`에서 조정

---

### 진행 로그(Progress Log)
- [M1 완료] 전처리/마스킹/Drain3 연동, `parse` CLI 실행 및 샘플 결과 생성
- [2025-09-14] DeepLog/MSCRED 입력 생성 완료
  - 생성 파일: `data/processed/vocab.json`, `data/processed/sequences.parquet`, `data/processed/window_counts.parquet`
- [2025-09-14] 베이스라인 탐지 실행 완료
  - 생성 파일: `data/processed/baseline_scores.parquet`, `data/processed/baseline_preview.json`
  - 예시(상위 1행): `window_start_line=0, unseen_rate=1.0, freq_z=0.0, score=0.4, is_anomaly=True`
- [2025-09-14] DeepLog 학습/추론 연동 완료
  - 생성 파일: `.cache/deeplog.pth`, `data/processed/deeplog_infer.parquet`
  - 메모: 샘플 데이터셋이 작아 violation_rate가 0.0로 측정됨(정상)
- [2025-09-14] MS-CRED 입력 CLI 및 리포트 생성 완료
  - 생성 파일: `data/processed/window_counts.parquet`, `data/processed/report.md`
- [2025-09-14] 합성 로그 기반 평가 결과(PRF1)
  - 파일: `data/processed/synth/eval.txt`
  - Baseline PRF1: P=1.000 R=0.061 F1=0.114
  - DeepLog PRF1: P=0.062 R=1.000 F1=0.117
