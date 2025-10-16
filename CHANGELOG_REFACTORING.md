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

## Phase 4: ONNX 변환 개선 (2025-10-16)

### 수정된 버그들

#### ONNX 변환 오류 수정

**문제 1**: DeepLog ONNX 변환 시 "'dict' object has no attribute 'eval'" 오류
- **원인**: `torch.load()`로 state_dict를 로드한 후 `.eval()` 직접 호출
- **해결**: DeepLogLSTM 클래스 import 후 모델 인스턴스 생성 → state_dict 로드

**문제 2**: MS-CRED ONNX 변환 시 "cannot import name MSCRED" 오류
- **원인**: 잘못된 클래스 이름 (MSCRED vs MSCREDModel)
- **해결**: `from study_preprocessor.mscred_model import MSCREDModel`로 수정
- **추가 수정**: state_dict 키 처리 (model_state_dict vs state_dict)

**문제 3**: MS-CRED 텐서 차원 불일치 오류
- **원인**: 3D 텐서 제공, 4D 텐서 필요
- **해결**: 더미 입력을 `(1, window_size, feature_dim)` → `(1, 1, window_size, feature_dim)`로 변경

**문제 4**: LSTM 배치 크기 경고
- **원인**: batch_size != 1 및 가변 길이 시퀀스 경고
- **해결**: `warnings.catch_warnings()`로 경고 억제

**문제 5**: MS-CRED "Output size is too small" 오류 ✅ **해결됨**
- **원인**: feature_dim이 1로 잘못 감지되어 conv 출력 크기가 0이 됨
- **해결**:
  - CLI에 `--feature-dim` 옵션 추가
  - vocab.json에서 템플릿 개수 자동 감지: `len(vocab_dict)`
  - 최소값 검증: `if feature_dim < 10: feature_dim = 10`
  - `convert_all_models()` 시그니처 업데이트로 feature_dim 전달

### 변경된 파일들

#### `study_preprocessor/cli.py` (convert-onnx 명령어)
```python
@click.option("--feature-dim", type=int, default=None,
              help="MS-CRED 피처 차원 (템플릿 개수, 미지정시 vocab에서 자동 감지)")
def convert_onnx_cmd(..., feature_dim: Optional[int]):
    # vocab.json에서 템플릿 개수 자동 감지
    if mscred_model and feature_dim is None and vocab:
        try:
            with open(vocab, 'r') as f:
                vocab_dict = json.load(f)
                feature_dim = len(vocab_dict)
                click.echo(f"📊 vocab에서 템플릿 개수 감지: {feature_dim}")
        except Exception as e:
            feature_dim = 100  # 안전한 기본값
```

#### `hybrid_system/training/model_converter.py`
**convert_deeplog_to_onnx()** (Lines 28-103):
- DeepLogLSTM 클래스 import 추가
- 모델 인스턴스 생성 후 state_dict 로드
- LSTM 경고 억제

**convert_mscred_to_onnx()** (Lines 131-295):
- MSCREDModel 클래스 import (MSCRED → MSCREDModel)
- state_dict 키 처리: 'model_state_dict' 또는 'state_dict'
- feature_dim 파라미터 추가 및 자동 감지
- 최소값 검증: `max(10, feature_dim)`
- 4D 텐서 생성: `(1, 1, window_size, feature_dim)`

**convert_all_models()** (Lines 366-385):
```python
def convert_all_models(
    deeplog_model: str,
    mscred_model: str,
    vocab_path: str,
    output_dir: str = "models/onnx",
    feature_dim: Optional[int] = None  # 새로운 파라미터
) -> Dict[str, Any]:
```

### 효과

#### 기능 개선
- ✅ **자동 feature_dim 감지**: vocab.json에서 템플릿 개수 자동 추출
- ✅ **안전한 기본값**: feature_dim < 10일 때 10으로 설정 (conv 레이어 보호)
- ✅ **수동 오버라이드**: `--feature-dim` 옵션으로 명시적 지정 가능
- ✅ **경고 제거**: LSTM 배치 크기 경고 억제

#### 사용자 경험
- 🎯 **간편한 사용**: vocab만 제공하면 자동으로 올바른 차원 설정
- 🎯 **명확한 피드백**: "📊 vocab에서 템플릿 개수 감지: N" 메시지
- 🎯 **오류 방지**: 최소 차원 검증으로 conv 크기 오류 예방

### 검증 방법

```bash
# 자동 감지 테스트 (7개 템플릿 → 10으로 조정됨)
study-preprocess convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab models/vocab.json \
  --output-dir models/onnx \
  --validate

# 100개 템플릿 테스트 (100 그대로 사용)
study-preprocess convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab large_vocab.json \
  --output-dir models/onnx

# 수동 지정 테스트
study-preprocess convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab models/vocab.json \
  --feature-dim 50 \
  --output-dir models/onnx
```

### 검증 결과

**테스트 1** (7개 템플릿):
- 감지된 템플릿: 7
- 실제 사용된 feature_dim: 10 (최소값으로 조정)
- 입력 형태: `[1, 1, 50, 10]`
- 결과: ✅ 변환 성공

**테스트 2** (100개 템플릿):
- 감지된 템플릿: 100
- 실제 사용된 feature_dim: 100
- 입력 형태: `[1, 1, 50, 100]`
- 결과: ✅ 변환 성공

### 추가 개선사항

**문제 6**: MS-CRED forward() TracerWarning 경고
- **원인**: `if reconstructed.shape != input_shape:` Python boolean 비교가 ONNX 추적 중 문제 발생
- **해결**: shape 비교 제거하고 항상 `F.interpolate()` 수행 (같은 크기일 경우 no-op)

**문제 7**: ONNX Runtime 최적화 저장 실패
- **원인**: ONNX Runtime 1.9+ 이후 `session.save()` 메서드 제거됨
- **해결**: `sess_options.optimized_model_filepath` 사용하여 세션 생성 시 자동 최적화 파일 저장

### 최종 결과

#### 생성 파일들
```
output_dir/
├── deeplog.onnx                 # DeepLog 원본 ONNX 모델
├── deeplog_optimized.onnx       # 하드웨어 최적화 적용
├── deeplog.onnx.meta.json       # 메타데이터
├── mscred.onnx                  # MS-CRED 원본 ONNX 모델
├── mscred_optimized.onnx        # 하드웨어 최적화 적용
├── mscred.onnx.meta.json        # 메타데이터
├── vocab.json                   # 어휘 사전
└── conversion_summary.json      # 변환 요약
```

#### 경고 메시지
- ✅ **TracerWarning 제거**: MS-CRED forward() 수정으로 경고 사라짐
- ✅ **최적화 성공**: `*_optimized.onnx` 파일 생성 확인
- ℹ️ **GPU 경고 무시 가능**: "GPU device discovery failed" - CPU 환경에서 정상

**문제 8**: 하드웨어 특화 최적화 경고
- **경고**: "hardware specific optimizations, should only be used in the same environment"
- **원인**: `ORT_ENABLE_ALL` 최적화 레벨이 현재 하드웨어에 특화된 최적화 적용
- **해결**: `--portable` 옵션 추가
  - Portable 모드: `ORT_ENABLE_BASIC` (범용, 모든 환경에서 사용 가능)
  - 기본 모드: `ORT_ENABLE_ALL` (최대 성능, 현재 환경 특화)

### 최적화 모드 비교

| 모드 | 최적화 레벨 | 파일명 | 용도 |
|------|-------------|--------|------|
| **Portable** (권장) | ORT_ENABLE_BASIC | `*_portable.onnx` | C 추론 엔진 배포, 여러 환경 |
| **Maximum** | ORT_ENABLE_ALL | `*_optimized.onnx` | 최대 성능, 같은 환경 전용 |

### 사용 예시

```bash
# C 추론 엔진 배포용 (권장)
study-preprocess convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab models/vocab.json \
  --output-dir models/onnx \
  --portable

# 현재 환경 최대 성능
study-preprocess convert-onnx \
  --deeplog-model models/deeplog.pth \
  --mscred-model models/mscred.pth \
  --vocab models/vocab.json \
  --output-dir models/onnx
```

### 다음 단계

- [x] ONNX Runtime 설치 ✅ (v1.23.1)
- [x] TracerWarning 수정 ✅
- [x] 최적화 파일 생성 ✅
- [x] 하드웨어 특화 경고 해결 ✅ (portable 모드)
- [ ] C 추론 엔진 테스트
- [ ] 프로덕션 환경 배포 가이드 작성
- [ ] 성능 벤치마크 수행

---

**작성자**: Claude Code
**날짜**: 2025-10-16
**Phase**: 4/4 완료 (ONNX 변환 완전 최적화)
