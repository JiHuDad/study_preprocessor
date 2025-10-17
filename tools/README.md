# Tools Directory (Deprecated)

⚠️ **이 디렉토리의 스크립트들은 deprecated되었습니다.**

모든 기능이 `alog-detect` CLI로 통합되었습니다. 새로운 CLI를 사용해주세요.

## 마이그레이션 가이드

### 이전 → 현재

| 이전 스크립트 | 새 CLI 명령어 |
|--------------|--------------|
| `python temporal_anomaly_detector.py` | `alog-detect analyze-temporal` |
| `python comparative_anomaly_detector.py` | `alog-detect analyze-comparative` |
| `python log_sample_analyzer.py` | `alog-detect analyze-samples` |
| `python mscred_analyzer.py` | `alog-detect analyze-mscred` |
| `python baseline_validator.py` | `alog-detect validate-baseline` |
| `python enhanced_batch_analyzer.py` | (스크립트 사용) |
| `python visualize_results.py` | (개발 중) |

## 사용 예시

### 시간 기반 분석

**이전:**
```bash
python tools/temporal_anomaly_detector.py \
  --data-dir processed/ \
  --output-dir temporal_results/
```

**현재:**
```bash
alog-detect analyze-temporal \
  --data-dir processed/ \
  --output-dir temporal_results/
```

### 비교 분석

**이전:**
```bash
python tools/comparative_anomaly_detector.py \
  --target target.parquet \
  --baselines baseline1.parquet baseline2.parquet
```

**현재:**
```bash
alog-detect analyze-comparative \
  --target target.parquet \
  --baselines baseline1.parquet baseline2.parquet
```

### 로그 샘플 추출

**이전:**
```bash
python tools/log_sample_analyzer.py \
  --processed-dir processed/ \
  --max-samples 10
```

**현재:**
```bash
alog-detect analyze-samples \
  --processed-dir processed/ \
  --max-samples 10
```

### MS-CRED 분석

**이전:**
```bash
python tools/mscred_analyzer.py \
  --data-dir processed/ \
  --output-dir mscred_analysis/
```

**현재:**
```bash
alog-detect analyze-mscred \
  --data-dir processed/ \
  --output-dir mscred_analysis/
```

### 베이스라인 검증

**이전:**
```bash
python tools/baseline_validator.py \
  file1.log file2.log \
  --output-dir validation_results/
```

**현재:**
```bash
alog-detect validate-baseline \
  file1.log file2.log \
  --output-dir validation_results/
```

## CLI 도움말

각 명령어의 자세한 옵션을 보려면 `--help`를 사용하세요:

```bash
alog-detect --help
alog-detect analyze-temporal --help
alog-detect analyze-comparative --help
alog-detect analyze-samples --help
alog-detect analyze-mscred --help
alog-detect validate-baseline --help
```

## 이점

CLI를 사용하면:

1. ✅ **일관된 인터페이스**: 모든 기능이 하나의 명령어로
2. ✅ **더 나은 도움말**: `--help`로 모든 옵션 확인
3. ✅ **자동 완성**: Shell 자동 완성 지원 (설정 필요)
4. ✅ **버전 관리**: `alog-detect --version`으로 버전 확인
5. ✅ **에러 처리**: 더 명확한 에러 메시지

## 하위 호환성

기존 스크립트들은 여전히 작동하지만, deprecation 경고가 표시됩니다.
가능한 한 빨리 새 CLI로 마이그레이션하는 것을 권장합니다.

## 추가 정보

- [메인 문서](../README.md)
- [CLI 레퍼런스](../docs/api/cli-reference.md) (예정)
- [마이그레이션 가이드](../docs/development/RENAMING_GUIDE.md)
