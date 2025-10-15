# 리팩토링 변경사항 로그

## Phase 1: 즉시 정리 (2025-10-14)

### 제거된 파일들

#### Python 파일
- ✅ `analyze_results.py` (216줄) - 사용되지 않는 결과 분석 도구

#### 삭제된 디렉토리 (~27MB)

**테스트 디렉토리** (20개):
- `test_comparison/`, `test_comparison_final/`, `test_comparison_quick/`, `test_comparison_synthetic/`
- `test_incremental/`, `test_incremental_final/`, `test_incremental_final2/`, `test_incremental_fixed/`, `test_incremental_fixed2/`, `test_incremental_metadata_fix/`
- `test_model_debug/`, `test_model_deeplog_final/`, `test_model_deeplog_fix/`, `test_model_final/`, `test_model_fixed/`, `test_model_fixed2/`, `test_model_with_mscred/`
- `test_mscred_fix/`, `test_mscred_final/`, `test_target_fix/`
- `test_logs/`

**출력 디렉토리** (13개):
- `inference_20251002_064202/`, `inference_20251002_064314/`, `inference_20251002_070613/`, `inference_20251002_070711/`, `inference_20251002_071836/`, `inference_20251002_072038/`
- `validation_20251002_065139/`, `validation_20251002_065427/`, `validation_20251002_065444/`, `validation_20251002_065519/`, `validation_20251002_070109/`, `validation_20251002_070136/`
- `demo_batch_20250922_015743/`
- `cli_test_output/`, `direct_test_output/`, `test_preprocessing_output/`
- `validation_test.log`

### 유지된 파일/디렉토리
- ✅ `test_datasets/` - 합성 로그 생성 스크립트 (유용함)
- ✅ `test_hybrid_training.py` - Phase 1.3 개발/테스트용
- ✅ `test_preprocessing.sh` - 단위 테스트 스크립트
- ✅ `test_sample.log` - 샘플 로그 파일

### 변경된 파일

#### `.gitignore`
- 추가: `!test_datasets/` - test_datasets만 예외로 추적
- 추가: `*_test_output/` - 모든 테스트 출력 디렉토리 패턴 무시

### 효과
- 📉 **디스크 공간**: ~27MB 절감
- 📉 **디렉토리 수**: 33개 감소
- 📉 **Python 파일**: 1개 감소
- ✨ **프로젝트 구조**: 더 명확하고 깔끔해짐

### 다음 단계 (Phase 2)
- [ ] `batch_log_analyzer.py` vs `enhanced_batch_analyzer.py` 통합
- [ ] `visualize_results.py` 처리 (제거 또는 CLI 통합)
- [ ] `run_batch_analysis.sh` wrapper 처리
- [ ] 문서 업데이트

### 검증 명령어
```bash
# 남은 테스트 디렉토리 확인
ls -d test_* 2>/dev/null

# 예상 결과: test_datasets, test_hybrid_training.py, test_preprocessing.sh, test_sample.log만 남음

# Git 상태 확인
git status

# 변경사항 확인
git diff .gitignore
```

### 롤백 방법 (필요시)
```bash
# analyze_results.py 복구
git restore --staged analyze_results.py
git restore analyze_results.py

# .gitignore 복구
git restore .gitignore
```

---

## Phase 2: 단기 리팩토링 (2025-10-14)

### 제거된 파일들

#### Python 파일
- ✅ `batch_log_analyzer.py` (442줄) - enhanced_batch_analyzer.py로 통합됨

### 변경된 파일

#### `run_batch_analysis.sh`
- 변경: 전체 스크립트를 간단한 wrapper로 변경 (134줄 → 16줄)
- 기능: `run_enhanced_batch_analysis.sh`로 모든 인자 전달
- 경고 메시지 추가: 향후 제거될 수 있음을 사용자에게 알림
- 호환성: 기존 사용자 영향 없음

### 유지된 파일

#### `visualize_results.py`
- 결정: **유지** (227줄)
- 이유:
  - 여러 스크립트에서 선택적으로 사용 중
  - 텍스트 기반 시각화로 유용한 기능 제공
  - CLI 통합은 Phase 3 이후로 미룸
  - 사용자가 필요시 직접 호출 가능

### 효과
- 📉 **Python 파일**: 1개 감소 (9개 → 8개)
- 📉 **라인 수**: 약 560줄 감소 (442 + 118 wrapper 감소)
- ✨ **배치 분석**: 단일 구현으로 통합, 유지보수 부담 감소
- ✅ **호환성**: 기존 사용자를 위한 wrapper 제공

### 검증 명령어
```bash
# Python 파일 수 확인
ls -1 *.py 2>/dev/null | wc -l

# run_batch_analysis.sh 테스트
./run_batch_analysis.sh --help

# Git 상태 확인
git status
```

---

## Phase 3: 분석 도구 모듈화 (2025-10-15)

### 이동된 파일들 (모듈화)

#### 분석 도구들을 `study_preprocessor/analyzers/`로 이동
- ✅ `temporal_anomaly_detector.py` → `study_preprocessor/analyzers/temporal.py` (322줄)
- ✅ `comparative_anomaly_detector.py` → `study_preprocessor/analyzers/comparative.py` (470줄)
- ✅ `log_sample_analyzer.py` → `study_preprocessor/analyzers/log_samples.py` (1,429줄)
- ✅ `mscred_analyzer.py` → `study_preprocessor/analyzers/mscred_analysis.py` (523줄)
- ✅ `baseline_validator.py` → `study_preprocessor/analyzers/baseline_validation.py` (403줄)

**총 이동된 코드**: 3,147줄

### 생성된 파일들

#### 새로운 모듈
- ✅ `study_preprocessor/analyzers/__init__.py` - 모듈 초기화

#### Wrapper 파일들 (호환성 유지)
- ✅ `temporal_anomaly_detector.py` (27줄) - deprecation wrapper
- ✅ `comparative_anomaly_detector.py` (27줄) - deprecation wrapper
- ✅ `log_sample_analyzer.py` (27줄) - deprecation wrapper
- ✅ `mscred_analyzer.py` (27줄) - deprecation wrapper
- ✅ `baseline_validator.py` (27줄) - deprecation wrapper

**기능**: 모듈로 리디렉션 + deprecation 경고 메시지

### 변경된 파일

#### `study_preprocessor/cli.py`
새로운 CLI 서브명령어 추가 (84줄 추가):
- `study-preprocess analyze-temporal` - 시간 기반 이상 탐지
- `study-preprocess analyze-comparative` - 비교 기반 이상 탐지
- `study-preprocess analyze-mscred` - MS-CRED 전용 분석
- `study-preprocess validate-baseline` - 베이스라인 품질 검증

(기존 `analyze-samples`는 이미 존재)

### 효과

#### 구조 개선
- ✨ **모듈화**: 분석 도구들이 일관된 구조로 정리됨
- ✨ **CLI 통합**: 모든 분석 도구를 CLI 명령어로 사용 가능
- ✨ **Import 경로**: 명확한 import 경로 (`study_preprocessor.analyzers.*`)

#### 호환성
- ✅ **기존 스크립트**: Wrapper로 완전한 하위 호환성 유지
- ⚠️ **Deprecation 경고**: 사용자에게 마이그레이션 권장

#### 유지보수성
- 📦 **패키지 구조**: 테스트 및 재사용 용이
- 🔍 **발견 용이성**: 모든 분석 도구가 한 곳에 모임
- 📚 **문서화**: 일관된 인터페이스

### 마이그레이션 가이드

#### 기존 방식 (여전히 작동, deprecated)
```bash
python temporal_anomaly_detector.py --data-dir data/processed
python comparative_anomaly_detector.py --target file.log --baselines b1.log b2.log
python log_sample_analyzer.py data/processed
python mscred_analyzer.py --data-dir data/processed
python baseline_validator.py file1.log file2.log
```

#### 새로운 방식 (권장)
```bash
study-preprocess analyze-temporal --data-dir data/processed
study-preprocess analyze-comparative --target file.log --baselines b1.log --baselines b2.log
study-preprocess analyze-samples --processed-dir data/processed
study-preprocess analyze-mscred --data-dir data/processed
study-preprocess validate-baseline file1.log file2.log
```

### 검증 명령어
```bash
# 모듈 구조 확인
ls -la study_preprocessor/analyzers/

# CLI 명령어 확인
study-preprocess --help

# 개별 명령어 확인
study-preprocess analyze-temporal --help
study-preprocess analyze-comparative --help
study-preprocess analyze-mscred --help
study-preprocess validate-baseline --help

# Wrapper 테스트 (deprecation 경고 출력 확인)
python temporal_anomaly_detector.py --help
```

---

## Phase 3.1: CLI 명령어 사용 일관성 개선 (2025-10-15)

### 변경 내용

#### Shell 스크립트 업데이트 (python → study-preprocess)

**run_baseline_validation.sh:**
- 변경: `python comparative_anomaly_detector.py` → `study-preprocess analyze-comparative`

**run_enhanced_batch_analysis.sh:**
- 변경: `python mscred_analyzer.py` → `study-preprocess analyze-mscred`
- 변경: `python log_sample_analyzer.py` → `study-preprocess analyze-samples`

**demo_mscred.sh:**
- 변경: `python mscred_analyzer.py` → `study-preprocess analyze-mscred`
- 변경: `python log_sample_analyzer.py` → `study-preprocess analyze-samples`

#### Python 코드 업데이트

**enhanced_batch_analyzer.py (3곳 수정):**
- 변경: `sys.executable, "log_sample_analyzer.py"` → `"study-preprocess", "analyze-samples"`
- 변경: `sys.executable, "temporal_anomaly_detector.py"` → `"study-preprocess", "analyze-temporal"`
- 변경: `sys.executable, "comparative_anomaly_detector.py"` → `"study-preprocess", "analyze-comparative"`
- 수정: `--baselines` 인자를 리스트로 확장하는 방식으로 변경 (Click 다중 옵션 지원)

**study_preprocessor/cli.py (2곳 수정):**
- 변경: subprocess로 python 스크립트 호출 → 모듈 import 및 직접 호출
- 개선: `report` 명령의 `--with-samples` 옵션에서 모듈 import 사용
- 개선: `analyze-samples` 명령에서 모듈 import 사용

#### 문서 업데이트

**README.md:**
- `python mscred_analyzer.py` → `study-preprocess analyze-mscred`
- `python temporal_anomaly_detector.py` → `study-preprocess analyze-temporal`
- `python comparative_anomaly_detector.py` → `study-preprocess analyze-comparative`

**BATCH_ANALYSIS_GUIDE.md:**
- `python temporal_anomaly_detector.py` → `study-preprocess analyze-temporal`
- `python comparative_anomaly_detector.py` → `study-preprocess analyze-comparative`

**CONTEXT.md:**
- `python temporal_anomaly_detector.py` → `study-preprocess analyze-temporal`
- `python log_sample_analyzer.py` → `study-preprocess analyze-samples`

**TRAIN_INFERENCE_GUIDE.md:**
- `python mscred_analyzer.py` → `study-preprocess analyze-mscred`

**RESULTS_GUIDE.md:**
- `python temporal_anomaly_detector.py` → `study-preprocess analyze-temporal`
- `python comparative_anomaly_detector.py` → `study-preprocess analyze-comparative`
- `python baseline_validator.py` → `study-preprocess validate-baseline`

**.cursor/rules/development-workflow.mdc:**
- 모든 개발 워크플로우 예제를 CLI 명령어로 업데이트

### 효과

#### 일관성 개선
- ✨ **명령어 통일**: 모든 스크립트와 문서가 `study-preprocess` CLI를 사용
- ✨ **사용자 경험**: 일관된 인터페이스로 학습 곡선 감소
- ✨ **유지보수성**: wrapper 대신 모듈 직접 사용으로 간접 호출 제거

#### 기능 개선
- 🔧 **모듈 통합**: cli.py에서 subprocess 대신 모듈 import로 직접 호출
- 🔧 **에러 처리**: 더 명확한 에러 메시지 및 처리
- 📚 **문서 정확성**: 사용자가 실제로 사용해야 하는 명령어로 문서화

### 영향받은 파일 (총 12개)

**코드 파일 (4개):**
- run_baseline_validation.sh
- run_enhanced_batch_analysis.sh
- demo_mscred.sh
- enhanced_batch_analyzer.py
- study_preprocessor/cli.py

**문서 파일 (6개):**
- README.md
- BATCH_ANALYSIS_GUIDE.md
- CONTEXT.md
- TRAIN_INFERENCE_GUIDE.md
- RESULTS_GUIDE.md
- .cursor/rules/development-workflow.mdc

**메타데이터 파일 (1개):**
- CHANGELOG_REFACTORING.md (이 파일)

### 검증 방법

```bash
# CLI 명령어 작동 확인
study-preprocess analyze-temporal --help
study-preprocess analyze-comparative --help
study-preprocess analyze-mscred --help
study-preprocess analyze-samples --help
study-preprocess validate-baseline --help

# 스크립트에서 python 명령어 사용 확인 (CHANGELOG 제외하고 없어야 함)
grep -r "python.*_analyzer\|python.*_detector\|python.*_validator" --include="*.sh" --include="*.py" --exclude="CHANGELOG*" .

# Wrapper는 여전히 작동 (deprecation 경고 출력)
python temporal_anomaly_detector.py --help
```

### 마이그레이션 영향

#### 기존 사용자
- ✅ **Wrapper 유지**: 기존 python 스크립트 호출은 여전히 작동 (deprecation 경고 표시)
- ✅ **자동화 스크립트**: 모든 shell 스크립트는 자동으로 새 CLI 사용
- ⚠️ **문서 참조**: 문서는 이제 권장 방식(CLI)만 표시

#### 개발자
- 🔧 **모듈 import**: cli.py에서 더 이상 subprocess 사용하지 않음
- 🔧 **타입 안정성**: 모듈 직접 호출로 타입 체크 가능
- 🔧 **디버깅**: subprocess 간접 호출 대신 직접 호출로 디버깅 용이

---

**작성자**: Claude Code
**날짜**: 2025-10-15
**Phase**: 3.1/4 완료
