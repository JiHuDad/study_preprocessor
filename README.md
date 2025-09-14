### study-preprocessor 사용 가이드

이 문서는 커널/시스템 로그(.log) 파일에 전처리와 이상탐지를 적용하는 방법을 단계별로 안내합니다. 모든 예시는 `uv` 기반으로 실행합니다.

#### 1) 설치/환경
- 사전 요구: macOS/Linux, Python 3.11+, Homebrew 권장
- uv 설치: `brew install uv`
- 의존성은 실행 시 자동 설치됩니다(프로젝트 루트에서 실행).

#### 2) 단일 .log 파일 전처리
- 기본 실행:
```
uv run study-preprocess parse \
  --input /path/to/your.log \
  --out-dir /path/to/outdir \
  --drain-state .cache/drain3.json
```
- 주요 산출물:
  - `/path/to/outdir/parsed.parquet`: 전처리 결과(raw/masked/template_id 등)
  - `/path/to/outdir/preview.json`: 상위 10행 미리보기(원문 vs 마스킹)

- 마스킹 옵션(개별 토글): 기본은 모두 마스킹 ON, 아래 플래그로 OFF 가능
  - `--no-mask-paths`, `--no-mask-hex`, `--no-mask-ips`, `--no-mask-mac`, `--no-mask-uuid`
  - `--no-mask-pid`, `--no-mask-device`, `--no-mask-num`
  - 예: 숫자/디바이스 접미사 마스킹을 끄고 실행
```
uv run study-preprocess parse \
  --input /path/to/your.log \
  --out-dir /path/to/outdir \
  --no-mask-device --no-mask-num
```

- Drain3 상태 재사용
  - `--drain-state .cache/drain3.json`로 상태를 저장/누적하여 템플릿 일관성을 유지
  - 여러 파일을 순차 처리할 때 같은 `--drain-state`를 지정하면 기존 템플릿을 재사용합니다.

#### 3) 여러 .log 파일 배치 처리
- 쉘 루프 예시:
```
OUT=/path/to/processed
STATE=.cache/drain3.json
mkdir -p "$OUT"
for f in /var/log/*.log; do
  uv run study-preprocess parse --input "$f" --out-dir "$OUT/$(basename "$f" .log)" --drain-state "$STATE"
done
```
- 결과 병합(선택):
```
uv run python - <<'PY'
import os, pandas as pd
base = '/path/to/processed'
parts = []
for d in os.listdir(base):
    p = os.path.join(base, d, 'parsed.parquet')
    if os.path.exists(p):
        df = pd.read_parquet(p)
        df['source'] = d
        parts.append(df)
if parts:
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(base, 'merged.parquet'), index=False)
    print('Merged ->', os.path.join(base, 'merged.parquet'))
else:
    print('No parquet found')
PY
```

#### 4) DeepLog/MSCRED 입력 생성
- DeepLog 입력(사전/시퀀스):
```
uv run study-preprocess build-deeplog \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir
```
- MS-CRED 입력(윈도우 카운트):
```
uv run study-preprocess build-mscred \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir \
  --window-size 50 --stride 25
```

#### 5) 이상탐지 실행
- 베이스라인(새 템플릿 비율 + 빈도 급변):
```
uv run study-preprocess detect \
  --parsed /path/to/outdir/parsed.parquet \
  --out-dir /path/to/outdir \
  --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.95
```
- DeepLog 학습/추론:
```
uv run study-preprocess deeplog-train \
  --seq /path/to/outdir/sequences.parquet \
  --vocab /path/to/outdir/vocab.json \
  --out .cache/deeplog.pth --seq-len 50 --epochs 3

uv run study-preprocess deeplog-infer \
  --seq /path/to/outdir/sequences.parquet \
  --model .cache/deeplog.pth --k 3
```
- 리포트/요약 생성:
```
uv run study-preprocess report --processed-dir /path/to/outdir
```
  - 포함: 베이스라인 이상 윈도우 비율, 상위 윈도우/템플릿, DeepLog 위반율

#### 6) 합성 데이터로 E2E 검증(옵션)
```
# 합성 로그 + 라벨 생성
uv run study-preprocess gen-synth --out data/raw/synth_long.log --lines 1000 --anomaly-rate 0.03

# 전처리 → 빌더 → 탐지 → 학습/추론 → 리포트/평가
uv run study-preprocess parse --input data/raw/synth_long.log --out-dir data/processed/synth --drain-state .cache/drain3.json
uv run study-preprocess build-deeplog --parsed data/processed/synth/parsed.parquet --out-dir data/processed/synth
uv run study-preprocess detect --parsed data/processed/synth/parsed.parquet --out-dir data/processed/synth --window-size 50 --stride 25 --ewm-alpha 0.3 --q 0.95
uv run study-preprocess deeplog-train --seq data/processed/synth/sequences.parquet --vocab data/processed/synth/vocab.json --out .cache/deeplog_synth.pth --seq-len 20 --epochs 2
uv run study-preprocess deeplog-infer --seq data/processed/synth/sequences.parquet --model .cache/deeplog_synth.pth --k 3
uv run study-preprocess report --processed-dir data/processed/synth
uv run study-preprocess eval --processed-dir data/processed/synth --labels data/raw/synth_long.log.labels.parquet --window-size 50 --seq-len 20
```

#### 7) 문제 해결 팁
- 템플릿이 과도하게 늘어나는 경우: 마스킹을 더 강하게 하거나 `--drain-state`를 유지하며 순서대로 처리
- 타임스탬프 파싱 실패: 라인 인덱스(`line_no`) 기준으로도 정렬되며, 포맷이 다른 경우 전처리 규칙 보강 필요
- 메모리: 대형 파일은 디렉터리 단위로 나눠 처리 후 병합 권장

#### 8) 산출물 해석 요약
- `parsed.parquet`: `raw`, `masked`, `template_id`, `template`, `timestamp`, `host` 등
- `baseline_scores.parquet`: `score`, `is_anomaly`, `window_start_line`
- `deeplog_infer.parquet`: `idx`, `target`, `in_topk` (top-k 위반 여부)
- `report.md`: 상위 이상 윈도우와 기여 템플릿/요약 지표
