# 리팩토링 계획서

**작성일**: 2025-10-14
**대상 프로젝트**: alog-detector (로그 이상탐지 프레임워크)

---

## 📊 현재 상태 분석

### 프로젝트 규모
- **Python 파일**: 25개
- **Shell 스크립트**: 19개 (14개 실행 스크립트 + 5개 데모)
- **테스트 디렉토리**: 20개 이상 (약 20MB)
- **임시 출력 디렉토리**: 12개 이상 (약 7MB)

### 코드베이스 구성

#### 핵심 모듈 (`anomaly_log_detector/`)
✅ **잘 구조화됨** - 수정 불필요
- `cli.py` - CLI 진입점 (Click 기반)
- `preprocess.py` - Drain3 + 마스킹
- `detect.py` - 베이스라인 탐지
- `synth.py` - 합성 로그 생성
- `eval.py` - 평가 메트릭
- `mscred_model.py` - MS-CRED 모델
- `builders/deeplog.py` - DeepLog 입력 생성
- `builders/mscred.py` - MS-CRED 입력 생성

#### 하이브리드 시스템 (`hybrid_system/`)
✅ **새로운 기능** - Phase 1.3 완료
- `training/` - ONNX 변환 (Python)
- `inference/` - C 추론 엔진 (고성능)

#### 루트 레벨 Python 스크립트들
⚠️ **리팩토링 필요**

**1. 배치 분석기 (중복됨)**
- `batch_log_analyzer.py` (424줄) - 구형 배치 분석기
- `enhanced_batch_analyzer.py` (2,445줄) - 향상된 배치 분석기
- 상태: `enhanced_batch_analyzer.py`가 `batch_log_analyzer.py`의 기능을 완전히 포함
- **문제점**: 두 파일이 거의 동일한 기능 수행, 코드 중복

**2. 분석 도구들 (부분적으로 사용됨)**
- `temporal_anomaly_detector.py` (322줄) - ✅ 사용 중 (run_inference.sh, 배치 스크립트)
- `comparative_anomaly_detector.py` (470줄) - ✅ 사용 중 (배치 스크립트)
- `log_sample_analyzer.py` (1,429줄) - ✅ 사용 중 (CLI, 여러 스크립트)
- `mscred_analyzer.py` (523줄) - ✅ 사용 중 (run_full_pipeline, demo, 배치)
- `baseline_validator.py` (403줄) - ✅ 사용 중 (run_baseline_validation.sh)

**3. 보고서 생성 도구들 (선택적 사용)**
- `analyze_results.py` (216줄) - ❌ **사용 안 됨** (어떤 스크립트에서도 호출 없음)
- `visualize_results.py` (227줄) - ⚠️ 선택적 사용 (여러 스크립트에서 `if [ -f ]` 체크 후 사용)

**4. 테스트 파일**
- `test_hybrid_training.py` (270줄) - Phase 1.3 개발용

#### Shell 스크립트들
⚠️ **일부 정리 필요**

**핵심 워크플로우 스크립트 (✅ 유지)**
- `train_models.sh` - 모델 학습
- `run_inference.sh` - 이상탐지 추론
- `compare_models.sh` - 모델 비교
- `train_models_incremental.sh` - 점진적 학습
- `validate_models.sh` - 모델 검증
- `run_enhanced_batch_analysis.sh` - 배치 분석 (향상된 버전)

**레거시/중복 스크립트 (⚠️ 검토 필요)**
- `run_batch_analysis.sh` - 구형 배치 분석 (enhanced 버전으로 대체됨)
- `run_full_pipeline.sh` - uv 기반 (잘 사용 안 됨)
- `run_full_pipeline_pip.sh` - pip 기반
- `run_baseline_validation.sh` - 특수 목적 (유지)

**데모 스크립트 (✅ 유지)**
- `demo_enhanced_batch.sh`
- `demo_log_samples.sh`
- `demo_mscred.sh`

**테스트 스크립트**
- `test_preprocessing.sh` - 단위 테스트용

#### 테스트/임시 디렉토리들
❌ **삭제 권장** (약 27MB)

**테스트 디렉토리** (개발 중 생성된 임시 데이터):
- `test_comparison`, `test_comparison_final`, `test_comparison_quick`, `test_comparison_synthetic`
- `test_incremental`, `test_incremental_final`, `test_incremental_final2`, `test_incremental_fixed`, `test_incremental_fixed2`, `test_incremental_metadata_fix`
- `test_model_debug`, `test_model_deeplog_final`, `test_model_deeplog_fix`, `test_model_final`, `test_model_fixed`, `test_model_fixed2`, `test_model_with_mscred`
- `test_mscred_fix`, `test_mscred_final`, `test_target_fix`
- `test_logs/`

**출력 디렉토리** (실행 결과 임시 데이터):
- `inference_YYYYMMDD_HHMMSS/` (6개)
- `validation_YYYYMMDD_HHMMSS/` (6개)
- `demo_batch_20250922_015743/` (6.7MB)
- `cli_test_output/`, `direct_test_output/`, `test_preprocessing_output/`

**유지할 디렉토리**:
- `test_datasets/` - 합성 로그 생성 스크립트 (유용함)

---

## 🎯 리팩토링 목표

### 1. 코드 단순화
- 중복 코드 제거
- 사용하지 않는 파일 정리
- 명확한 책임 분리

### 2. 유지보수성 향상
- 모듈 간 결합도 감소
- 일관된 코딩 패턴
- 문서화 개선

### 3. 확장성 확보
- 새로운 탐지 방법 추가 용이
- 플러그인 구조 도입 가능성
- 테스트 용이성

### 4. 성능 최적화
- 불필요한 중복 실행 제거
- 효율적인 데이터 파이프라인

---

## 📋 리팩토링 작업 목록

### Phase 1: 정리 및 통합 (우선순위: 높음)

#### 1.1 중복 파일 제거
- [ ] **`batch_log_analyzer.py` 제거**
  - 이유: `enhanced_batch_analyzer.py`가 모든 기능 포함
  - 영향: `run_batch_analysis.sh` 수정 필요 (또는 스크립트도 제거)
  - 작업:
    1. `run_batch_analysis.sh`를 `run_enhanced_batch_analysis.sh`를 호출하도록 수정 (wrapper)
    2. 또는 `run_batch_analysis.sh` 자체 제거, README 업데이트

#### 1.2 사용하지 않는 파일 제거
- [ ] **`analyze_results.py` 제거**
  - 이유: 어떤 스크립트에서도 호출되지 않음
  - 기능: 결과 분석 및 해석 (216줄)
  - 대안: 기능이 필요하면 CLI에 통합 또는 문서에 기록 후 삭제

#### 1.3 테스트/임시 디렉토리 정리
- [ ] **개발 중 생성된 테스트 디렉토리 삭제** (~20개, 20MB)
  - `test_comparison*`, `test_incremental*`, `test_model*`, `test_mscred*`, `test_target_fix`
  - 방법: `.gitignore`에 `test_*/` 추가, 기존 디렉토리 삭제
  - 예외: `test_datasets/` 유지 (유용한 합성 로그 생성 스크립트)

- [ ] **임시 출력 디렉토리 삭제** (~12개, 7MB)
  - `inference_*/`, `validation_*/`, `demo_batch_*/`
  - 방법: `.gitignore`에 패턴 추가
  - 예외: 데모용으로 하나만 남기고 싶다면 `demo_batch_example/`로 이름 변경

#### 1.4 스크립트 통합
- [ ] **`run_batch_analysis.sh` 처리**
  - 옵션 A: `run_enhanced_batch_analysis.sh`로 리디렉션 (wrapper)
  - 옵션 B: 완전 제거, 문서 업데이트
  - 권장: 옵션 A (기존 사용자 호환성)

- [ ] **`run_full_pipeline.sh` vs `run_full_pipeline_pip.sh` 통합**
  - 현재: uv 버전과 pip 버전 분리
  - 제안: 하나의 스크립트로 통합, 자동 감지
  - 또는: pip 버전을 주 버전으로, uv 버전은 `run_full_pipeline_uv.sh`로 변경

### Phase 2: 구조 개선 (우선순위: 중간)

#### 2.1 분석 도구 모듈화
**현재 문제점**:
- `temporal_anomaly_detector.py`, `comparative_anomaly_detector.py`, `log_sample_analyzer.py`, `mscred_analyzer.py`가 루트에 분산
- 각각 독립 실행 스크립트로 존재
- CLI와 분리되어 있음

**제안 구조**:
```
anomaly_log_detector/
├── analyzers/                    # 새로운 모듈
│   ├── __init__.py
│   ├── temporal.py               # temporal_anomaly_detector 내용
│   ├── comparative.py            # comparative_anomaly_detector 내용
│   ├── log_samples.py            # log_sample_analyzer 내용
│   └── mscred_analysis.py        # mscred_analyzer 내용
└── cli.py                        # 서브명령어 추가
```

**장점**:
- 일관된 모듈 구조
- CLI로 통합 가능: `alog-detect analyze-temporal`, `alog-detect analyze-comparative`
- 테스트 및 재사용 용이
- Import 경로 명확

**마이그레이션 계획**:
1. `anomaly_log_detector/analyzers/` 디렉토리 생성
2. 각 파일을 모듈로 이동, 리팩토링
3. CLI에 새로운 서브명령어 추가
4. 루트의 원본 파일들은 CLI 호출 wrapper로 변경 (호환성)
5. 문서 업데이트 후 다음 메이저 버전에서 wrapper 제거

#### 2.2 시각화 도구 통합 또는 제거
**`visualize_results.py` 처리**:
- 현재: 선택적으로 사용 (여러 스크립트에서 `if [ -f ]` 체크)
- 문제: matplotlib/seaborn 없이 텍스트 차트만 출력 (제한적)
- 옵션:
  1. **제거**: 시각화는 사용자가 Jupyter/별도 도구 사용 권장
  2. **CLI 통합**: `alog-detect visualize` 명령어로 추가
  3. **개선**: 제대로 된 시각화 라이브러리 사용 (선택적 의존성)
- 권장: 옵션 1 (제거) 또는 옵션 2 (CLI 통합)

#### 2.3 Baseline Validator 위치 조정
**`baseline_validator.py`**:
- 현재: 루트 레벨, 특수 목적
- 용도: 베이스라인 파일들의 품질 검증
- 제안: `anomaly_log_detector/validators/` 또는 `analyzers/` 아래로 이동
- CLI 통합: `alog-detect validate-baseline`

### Phase 3: 아키텍처 개선 (우선순위: 낮음, 장기)

#### 3.1 플러그인 아키텍처 도입
**목적**: 새로운 탐지 방법 추가 용이

**제안 구조**:
```
anomaly_log_detector/
├── core/                         # 핵심 기능
│   ├── preprocess.py
│   ├── pipeline.py
│   └── base_detector.py          # 추상 베이스 클래스
├── detectors/                    # 탐지 방법들
│   ├── __init__.py
│   ├── baseline.py               # 기존 detect.py
│   ├── deeplog.py                # 기존 builders/deeplog.py 통합
│   ├── mscred.py                 # 기존 builders/mscred.py 통합
│   └── custom/                   # 사용자 정의 탐지기
└── analyzers/                    # 분석 도구들
```

**장점**:
- 일관된 인터페이스
- 새 탐지 방법 추가 시 기존 코드 수정 불필요
- 테스트 및 벤치마크 용이

#### 3.2 설정 파일 통합
**현재**:
- `rules.json` - 마스킹 규칙
- CLI 인자로 분산된 설정 (window_size, stride, ewm_alpha 등)
- Drain3 설정은 코드에 하드코딩

**제안**:
- `config.yaml` 또는 `anomaly_log_detector.toml` 생성
- 모든 설정을 한 곳에서 관리
- 환경 변수 오버라이드 지원

#### 3.3 테스트 프레임워크 구축
**현재**: 테스트 코드 부족
**제안**:
```
tests/
├── unit/
│   ├── test_preprocess.py
│   ├── test_detect.py
│   └── test_builders.py
├── integration/
│   ├── test_full_pipeline.py
│   └── test_batch_analysis.py
└── fixtures/
    └── sample_logs/
```

---

## 🚀 실행 계획

### Step 1: 즉시 실행 (1일)
1. ✅ `.gitignore` 업데이트
   ```
   # Test and temporary directories
   test_*/
   inference_*/
   validation_*/
   demo_batch_*/
   *_test_output/

   # Except useful test data
   !test_datasets/
   ```

2. 🗑️ 테스트/임시 디렉토리 삭제
   ```bash
   rm -rf test_comparison* test_incremental* test_model* test_mscred* test_target_fix
   rm -rf inference_* validation_* demo_batch_20250922_015743
   rm -rf cli_test_output direct_test_output test_preprocessing_output test_logs
   ```

3. 🗑️ 사용하지 않는 파일 제거
   ```bash
   # analyze_results.py 삭제 (사용 안 됨)
   git rm analyze_results.py
   ```

4. 📝 문서 업데이트
   - README.md: 삭제된 파일/디렉토리 언급 제거
   - CLAUDE.md: 최신 구조 반영

### Step 2: 단기 리팩토링 (2-3일)

1. **배치 분석기 통합**
   ```bash
   # batch_log_analyzer.py를 제거하고
   # run_batch_analysis.sh를 wrapper로 변경
   ```

2. **시각화 도구 처리**
   - visualize_results.py를 CLI로 통합 또는 제거 결정
   - 관련 스크립트들 수정

3. **분석 도구 정리**
   - baseline_validator.py 위치 조정 검토

### Step 3: 중기 리팩토링 (1-2주)

1. **분석 도구 모듈화**
   - `anomaly_log_detector/analyzers/` 생성
   - 파일 이동 및 리팩토링
   - CLI 통합

2. **스크립트 통합**
   - `run_full_pipeline*.sh` 통합

3. **문서 전면 업데이트**

### Step 4: 장기 개선 (필요시)

1. 플러그인 아키텍처 도입
2. 설정 파일 통합
3. 테스트 프레임워크 구축

---

## 📊 예상 효과

### 정량적 개선
- **디스크 사용량**: ~27MB 절감 (테스트/임시 디렉토리 삭제)
- **파일 수**: Python 파일 1개 이상 감소, 디렉토리 30개 이상 감소
- **라인 수**: 중복 코드 제거로 약 400-600줄 감소

### 정성적 개선
- ✅ 프로젝트 구조 명확화
- ✅ 신규 개발자 온보딩 시간 단축
- ✅ 유지보수 부담 감소
- ✅ 버그 발생 가능성 감소
- ✅ 확장성 향상

---

## ⚠️ 주의사항

### 호환성 유지
- 기존 사용자를 위해 wrapper 스크립트 제공
- 문서에 마이그레이션 가이드 포함
- 주요 변경은 메이저 버전 업에서 진행

### 백업
- 리팩토링 전 브랜치 생성: `git checkout -b refactoring-backup`
- 주요 변경마다 커밋
- 테스트 실행 후 머지

### 단계적 진행
- 한 번에 모든 것을 변경하지 않기
- 각 단계마다 테스트 및 검증
- 사용자 피드백 수렴

---

## 📝 다음 단계

리팩토링을 시작하려면:

1. **현재 작업 커밋 및 푸시**
2. **백업 브랜치 생성**: `git checkout -b pre-refactoring-backup`
3. **리팩토링 브랜치 생성**: `git checkout -b refactoring-phase1`
4. **Step 1 (즉시 실행) 진행**
5. **테스트 및 검증**
6. **PR 생성 및 리뷰**

---

**작성자 노트**: 이 계획서는 현재 코드베이스 분석을 바탕으로 작성되었습니다. 실제 리팩토링 진행 시 추가적인 이슈가 발견될 수 있으며, 그에 따라 계획을 조정할 수 있습니다.
