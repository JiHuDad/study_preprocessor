### PRD: 커널 로그 기반 전처리 파이프라인 (DeepLog/MS-CRED 연계)

#### 목적
- 커널/시스템 로그를 안정적으로 전처리하여 이상탐지용 시퀀스/피처를 생성한다.
- LSTM 기반 DeepLog 및 시계열 이미지화 기반 MS-CRED에 입력 가능한 형식으로 결과물을 제공한다.
- 빠른 반복과 재현성을 위해 `uv` 기반 파이썬 프로젝트로 구성한다.

#### 범위 (In-Scope)
- 커널/시스템 로그(dmesg, syslog 등) 전처리
- 템플릿 마이닝(Drain3) + 규칙 기반 마스킹(정규식)
- DeepLog용 토큰 시퀀스/윈도우 시퀀스 생성
- MS-CRED용 윈도우 단위 이벤트 카운트 벡터(1차 목표) 및 공발생(co-occurrence) 매트릭스(선택적)
- CLI 제공, 배치 처리, 대용량 스트리밍 처리

#### 비범위 (Out-of-Scope)
- 모델 학습/추론(DeepLog, MS-CRED) 자체 구현은 별도 레포/노트북에서 진행
- 분산 처리(예: Spark)는 초기 버전에서 제외

#### 이해관계자 / 사용자
- 데이터/ML 엔지니어: 파이프라인 실행, 아티팩트 소비
- 연구자: 전처리 결과를 DeepLog/MS-CRED 학습 입력으로 사용

---

### 요구사항

#### 입력
- 텍스트 로그 파일: 예) `data/raw/dmesg.sample.log`
- 포맷 가정: `[timestamp] [host?] [proc?] message` 형태가 섞여 있어도 라인 기반 처리 가능해야 함
- 타임존/포맷이 불명확한 경우에도 best-effort 파싱 후 미파싱 라인은 원문 보존

#### 전처리 규칙 (마스킹 + 템플릿)
- Drain3로 로그 템플릿 마이닝 및 안정적 템플릿 ID 부여
- 규칙 기반 마스킹(정규식):
  - 숫자: 정수/실수 `-?\d+(\.\d+)?`
  - 타임스탬프: ISO/rsyslog 일반 패턴
  - 주소: `0x[0-9a-fA-F]+`
  - PID/TID: `pid=\d+`, `tid=\d+`
  - IP: IPv4/IPv6
  - MAC: `([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}`
  - UUID: RFC4122 대략 패턴
  - 파일경로: `/[\w\.\-_/]+`
  - 디바이스명 숫자 접미: `eth\d+` → `eth<ID>` 등
- 마스킹 토큰 예시: `<NUM>`, `<HEX>`, `<IP>`, `<PID>`, `<PATH>` 등
- 템플릿과 변수 분리 저장: `template`, `parameters`

#### 산출물 (아티팩트)
- 파스드 테이블(parquet):
  - 컬럼: `timestamp`(nullable), `host`(nullable), `raw`, `masked`, `template_id`, `template`
- DeepLog 입력:
  - `vocab.json`: `template_id` → 인덱스 매핑
  - `sequences.parquet`: 정렬된 `template_index` 시퀀스(옵션: 세션/호스트 단위)
  - `windows.parquet`: 슬라이딩 윈도우별 시퀀스, stride/window_size 설정
- MS-CRED 입력(1차):
  - `window_counts.parquet`: 윈도우 × 템플릿 카운트 벡터(행은 윈도우, 열은 템플릿)
  - 선택: `window_cooccurrence.parquet`: 템플릿 공발생 매트릭스(희소 표현)

#### 파이프라인 동작
- 스트리밍 파싱: 파일 라인 스트림 → 마스킹 → Drain3 업데이트/조회 → 로우 아웃풋 작성
- 메모리 효율: 청크/배치 처리, 파케이(Arrow) 저장, `orjson` 직렬화
- 성능 목표(초기): 1GB 로그 < 2분(로컬 M1/M2 기준, 최적화 전제)

#### 빠른 시작(Quickstart)
- uv 설치: `brew install uv`
- 의존성 설치 및 실행 예:
```
uv run study-preprocess parse \
  --input data/raw/dmesg.sample.log \
  --out-dir data/processed \
  --drain-state .cache/drain3.json
```
- 산출물: `data/processed/parsed.parquet`, `data/processed/preview.json`

#### CLI / API 사양
- 엔트리포인트: `study-preprocess`
- 서브커맨드:
  - `parse`: 원시 로그 → 파스드 테이블 생성(구현됨)
    - 예: `study-preprocess parse --input data/raw/dmesg.sample.log --out-dir data/processed --drain-state .cache/drain3.json`
- (임시) Python API로 빌더 사용:
```
uv run python -c "from study_preprocessor.builders.deeplog import build_deeplog_inputs as b; b('data/processed/parsed.parquet','data/processed')"
uv run python -c "from study_preprocessor.builders.mscred import build_mscred_window_counts as b; b('data/processed/parsed.parquet','data/processed',window_size=50,stride=25)"
```
- 향후 계획: `build-deeplog`, `build-mscred` 서브커맨드 추가

#### 디렉터리 구조(제안)
```
/Users/kaeulkim/dev/study_preprocessor/
  ├─ data/
  │   ├─ raw/                  # 예제 원시 로그
  │   └─ processed/            # 파스드/윈도우 산출물
  ├─ config/
  │   └─ drain3.yaml           # Drain3 파라미터(Depth, similarity 등)
  ├─ .cache/
  ├─ study_preprocessor/
  │   ├─ __init__.py
  │   ├─ cli.py                # click CLI
  │   ├─ preprocess.py         # 마스킹 + Drain3 래퍼
  │   ├─ builders/
  │   │   ├─ deeplog.py
  │   │   └─ mscred.py
  │   └─ utils/
  │       └─ io.py
  ├─ prd.md
  └─ tasklist.md
```

#### 예제 데이터 및 샘플 결과
- 예제 원시 로그: `data/raw/dmesg.sample.log`
- 실행 결과 미리보기: `data/processed/preview.json`
- Raw vs Masked 예시(일부):

| line_no | raw | masked |
|---|---|---|
| 0 | [12345.678901] usb 1-1: new high-speed USB device number 2 using ehci-pci | [<NUM>] usb 1-1: new high-speed USB device number <NUM> using ehci-pci |
| 1 | [12345.678902] CPU0: Core temperature above threshold, cpu clock throttled | [<NUM>] CPU<ID>: Core temperature above threshold, cpu clock throttled |
| 2 | [12345.678903] CPU0: Core temperature/speed normal | [<NUM>] CPU<ID>: Core temperature<PATH> normal |
| 3 | [12345.678904] eth0: Link is Up - 1000Mbps/Full - flow control rx/tx | [<NUM>] eth<ID>: Link is Up - 1000Mbps<PATH> - flow control rx<PATH> |
| 4 | [12345.678905] EXT4-fs (sda1): mounted filesystem with ordered data mode. Opts: (null) | [<NUM>] EXT<ID>-fs (sda<ID>): mounted filesystem with ordered data mode. Opts: (null) |

- 전체 컬럼/템플릿은 `data/processed/parsed.parquet`에서 확인 가능

#### 리스크/고려사항
- 로그 포맷 다양성: Drain3 파라미터 튜닝 필요
- 높은 카디널리티: 과도한 템플릿 증가 → 마스킹 규칙 강화/정규화 필요
- 타임스탬프 누락: 순서 보장 어려움 → 라인 인덱스 기반 보조 정렬

#### 마일스톤
- [x] M1: 프로젝트 스캐폴딩(uv), 파서/마스킹, `parse` CLI, 예제 실행(샘플 포함)
- [x] M2: DeepLog 빌더, 윈도우링, vocab/시퀀스 산출
- [x] M3: MS-CRED 빌더(카운트 벡터), 선택적 공발생 매트릭스
- [x] M4: 성능 튜닝 및 문서 보강
- [x] M5: 이상탐지 모듈 구현 (베이스라인, DeepLog)
- [x] M6: 배치 분석 시스템 구현
- [x] M7: 고급 이상탐지 방법 추가 (시간 기반, 비교 기반)
- [x] M8: 통합 문서화 및 사용자 가이드 완성

#### 추가 구현된 기능
- **베이스라인 이상탐지**: 윈도우 기반 템플릿 빈도 변화 감지
- **DeepLog 학습/추론**: LSTM 기반 시퀀스 예측 모델
- **합성 데이터 생성**: 테스트용 라벨된 로그 생성
- **평가 메트릭**: Precision, Recall, F1 계산
- **배치 분석**: 디렉토리 단위 일괄 처리
- **향상된 배치 분석**: 하위 디렉토리 재귀 스캔
- **시간 기반 이상탐지**: 과거 동일 시간대 패턴 비교
- **비교 기반 이상탐지**: 여러 파일 간 패턴 차이 분석
- **자동화 스크립트**: 전체 파이프라인 원클릭 실행
- **리포트 생성**: 마크다운 형식 분석 결과 요약

#### 🆕 최신 추가 기능 (2025-09-20)

##### 베이스라인 품질 검증 시스템
- **문제 정의**: 베이스라인 로그 자체에 이상이 있으면 이상탐지 정확도 저하
- **해결책**: 자동 베이스라인 품질 평가 및 필터링
- **구현**:
  - `baseline_validator.py`: 에러율, 템플릿 다양성, 로그 수량 등 품질 지표 평가
  - `enhanced_batch_analyzer.py`에 자동 베이스라인 필터링 통합
  - 품질 임계값: 에러율 2% 이하, 경고율 5% 이하, 최소 템플릿 10개, 최소 로그 100개
  - 희귀 템플릿 비율 30% 이하로 제한

##### 로그 샘플 분석 시스템
- **문제 정의**: 이상탐지 결과가 너무 기술적이어서 실제 문제 로그를 파악하기 어려움
- **해결책**: 실제 이상 로그 샘플을 사람이 읽기 쉬운 형태로 추출 및 분석
- **구현**:
  - `log_sample_analyzer.py`: 이상탐지 결과에서 실제 로그 샘플 추출
  - 전후 맥락 3줄과 함께 로그 표시
  - 이상 유형별 설명 및 패턴 분석 제공
  - 마크다운 형식 종합 리포트 생성 (`anomaly_analysis_report.md`)
  - CLI 명령어: `study-preprocess analyze-samples`
  - `study-preprocess report --with-samples` 옵션으로 통합 분석

##### CLI 확장
- **Target vs Baseline 구분**: Target은 분석할 파일, Log Directory는 baseline 학습용 파일들
- **새 명령어**:
  - `study-preprocess analyze-samples`: 독립적인 로그 샘플 분석
  - `study-preprocess report --with-samples`: 기존 리포트에 샘플 분석 포함

##### 프로젝트 관리 개선
- **`.gitignore` 강화**: 분석 결과 디렉토리, 모델 파일, Drain3 상태 파일 등 추가
- **데모 스크립트**: `demo_log_samples.sh`로 전체 기능 시연
- **문서 업데이트**: README에 새 기능 사용법 및 예시 추가

#### 수용 기준(Acceptance)
- [x] 샘플 로그로 `parse` 실행 시 Parquet 산출 및 3개+ 마스킹 타입 확인 가능
- [x] DeepLog 빌더로 시퀀스/윈도우 산출 확인
- [x] MS-CRED 빌더로 카운트 벡터 산출 확인
- [x] 베이스라인 이상탐지로 이상 윈도우 탐지 확인
- [x] DeepLog 학습 및 추론 파이프라인 동작 확인
- [x] 배치 분석으로 여러 파일 일괄 처리 확인
- [x] 전체 파이프라인 자동화 스크립트 동작 확인
- [x] 베이스라인 품질 검증으로 문제 있는 baseline 자동 필터링 확인
- [x] 로그 샘플 분석으로 실제 이상 로그 내용 및 맥락 추출 확인
- [x] 향상된 배치 분석에서 모든 새 기능들이 통합되어 동작 확인
