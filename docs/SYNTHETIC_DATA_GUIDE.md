# 🧪 합성 로그 데이터 생성 가이드

## 📋 목차

1. [개요](#개요)
2. [데이터 타입별 설명](#데이터-타입별-설명)
3. [사용 방법](#사용-방법)
4. [이상 타입 상세](#이상-타입-상세)
5. [전체 워크플로우](#전체-워크플로우)
6. [실전 예시](#실전-예시)

---

## 개요

이 프로젝트는 **3가지 타입의 합성 로그 데이터**를 생성할 수 있습니다:

| 데이터 타입 | 용도 | 정상률 | 이상률 | 명령어 |
|-----------|------|-------|-------|--------|
| **학습용** | 모델 학습 | 100% | 0% | `gen-training-data` |
| **추론용 정상** | False Positive 테스트 | 100% | 0% | `gen-inference-normal` |
| **추론용 비정상** | True Positive 테스트 | 85% | 15% | `gen-inference-anomaly` |

### 🎯 목적

1. **학습 데이터**: 모델이 **정상 패턴**을 학습
2. **추론 정상**: 모델이 정상을 **정상으로 인식**하는지 확인 (False Positive 방지)
3. **추론 비정상**: 모델이 이상을 **이상으로 탐지**하는지 확인 (True Positive 확인)

---

## 데이터 타입별 설명

### 1️⃣ 학습용 데이터 (`gen-training-data`)

**특징:**
- ✅ 100% 정상 로그만 포함
- ✅ 7가지 정상 템플릿 랜덤 조합
- ✅ 일관된 시간 간격 (1초)
- ✅ 다양한 정상 패턴 학습

**정상 템플릿 예시:**
```
usb 1-1: new high-speed USB device number 3 using ehci-pci
CPU2: Core temperature above threshold, cpu clock throttled
CPU2: Core temperature/speed normal
eth0: Link is Up - 1000Mbps/Full - flow control rx/tx
EXT4-fs (sda1): mounted filesystem with ordered data mode. Opts: (null)
```

**사용 예:**
```bash
# 10,000줄 학습 데이터 생성
alog-detect gen-training-data \
    --out data/raw/training.log \
    --lines 10000 \
    --host train-server
```

**출력:**
- `data/raw/training.log` - 로그 파일
- `data/raw/training.log.labels.parquet` - 레이블 (모두 0)

---

### 2️⃣ 추론용 정상 데이터 (`gen-inference-normal`)

**특징:**
- ✅ 100% 정상 로그
- ✅ 학습 데이터와 **같은 템플릿**, 다른 값
- ✅ 모델이 정상으로 인식해야 함
- ✅ False Positive 비율 측정용

**목적:**
- 모델이 학습한 정상 패턴을 **제대로 인식**하는지 확인
- **False Positive** (정상을 이상으로 오탐)가 얼마나 발생하는지 측정

**사용 예:**
```bash
# 1,000줄 테스트용 정상 데이터 생성
alog-detect gen-inference-normal \
    --out data/raw/test_normal.log \
    --lines 1000 \
    --host test-server
```

**기대 결과:**
- DeepLog 위반율: **< 20%** (이상적: < 10%)
- Baseline 이상률: **< 5%**

---

### 3️⃣ 추론용 비정상 데이터 (`gen-inference-anomaly`)

**특징:**
- ⚠️ 정상 로그 + 다양한 이상 로그 혼합
- ⚠️ 기본 15% 이상률 (조절 가능)
- ⚠️ 5가지 이상 타입 선택 가능
- ⚠️ True Positive 비율 측정용

**이상 타입:**

| 타입 | 설명 | 예시 |
|------|------|------|
| `unseen` | 학습 시 보지 못한 새 템플릿 | `nvme0: I/O error on namespace 1` |
| `error` | 에러 메시지 | `ERROR: disk I/O error, dev sda, sector 12345` |
| `attack` | 보안 공격 시뮬레이션 | `Failed password for root from 192.168.1.100` |
| `crash` | 시스템 크래시 | `systemd[1]: Failed to start mysql.service` |
| `burst` | 특정 템플릿 급증 (10-30개 연속) | 같은 로그 20번 반복 |

**사용 예:**
```bash
# 모든 이상 타입 포함 (기본)
alog-detect gen-inference-anomaly \
    --out data/raw/test_anomaly.log \
    --lines 1000 \
    --anomaly-rate 0.15

# 특정 이상 타입만 선택
alog-detect gen-inference-anomaly \
    --out data/raw/test_attack.log \
    --lines 1000 \
    --anomaly-rate 0.20 \
    --anomaly-types unseen \
    --anomaly-types attack \
    --anomaly-types burst
```

**출력:**
- `data/raw/test_anomaly.log` - 로그 파일
- `data/raw/test_anomaly.log.labels.parquet` - 레이블 (이상 구간 표시, anomaly_type 포함)
- `data/raw/test_anomaly.log.meta.json` - 통계 (이상 타입별 분포)

**기대 결과:**
- DeepLog 위반율: **> 50%** (이상 탐지 성공)
- Baseline 이상률: **> 15%**

---

## 이상 타입 상세

### 🔥 `unseen` - 새로운 템플릿

**설명:** 학습 시 보지 못한 완전히 새로운 로그 패턴

**예시:**
```
nvme0: I/O error on namespace 1
nvme1: I/O error on namespace 3
kernel BUG at /usr/src/linux/mm/page_alloc.c:234
kernel BUG at /usr/src/linux/mm/page_alloc.c:567
```

**탐지 방법:**
- Baseline: 새로운 템플릿 비율(unseen_rate)로 탐지
- DeepLog: 예측 실패로 탐지

---

### ⚠️ `error` - 에러 메시지

**설명:** 시스템 에러, 경고, 치명적 오류 메시지

**예시:**
```
ERROR: disk I/O error, dev sda, sector 123456
CRITICAL: Out of memory: Kill process 1234 (systemd)
WARNING: Temperature above threshold on CPU2
FATAL: kernel panic - not syncing: VFS: Unable to mount root fs
ERROR: segmentation fault at 0x7f1234 ip 192.168.1.100 sp 0x7fff12345678 error 4
```

**탐지 방법:**
- Baseline: 에러 키워드 빈도 급증
- DeepLog: 정상 패턴과 다른 시퀀스
- 리포트: 에러 로그 자동 하이라이트

---

### 🛡️ `attack` - 보안 공격

**설명:** SSH 무차별 대입, SYN Flooding 등 보안 공격 시뮬레이션

**예시:**
```
sshd[1234]: Failed password for invalid user admin from 192.168.1.100 port 52341 ssh2
sshd[1235]: Failed password for root from 192.168.1.100 port 52342 ssh2
kernel: TCP: Possible SYN flooding on port 22. Sending cookies.
sudo: pam_unix(sudo:auth): authentication failure; uid=1000 user=root
```

**탐지 방법:**
- Baseline: 실패 로그 패턴 급증
- DeepLog: 비정상 시퀀스 패턴
- 보안 분석: 공격 시도 패턴 인식

---

### 💥 `crash` - 시스템 크래시

**설명:** 서비스 실패, 커널 패닉 등 시스템 불안정 상태

**예시:**
```
systemd[1]: Failed to start docker.service.
systemd[1]: mysql.service: Main process exited, code=killed, status=9/KILL
kernel: Oops: 0002 [#1] SMP
kernel: RIP: 0010:do_syscall_64+0x45/0x100
```

**탐지 방법:**
- Baseline: 크래시 패턴 출현
- DeepLog: 정상 시작/종료 패턴과 상이
- 운영: 서비스 재시작 필요성 판단

---

### 📈 `burst` - 템플릿 급증

**설명:** 정상 템플릿이 10-30개 연속으로 나타나는 이상 패턴

**예시:**
```
usb 1-1: new high-speed USB device number 3 using ehci-pci
usb 1-1: new high-speed USB device number 4 using ehci-pci
usb 1-1: new high-speed USB device number 5 using ehci-pci
... (20번 반복)
usb 1-1: new high-speed USB device number 23 using ehci-pci
```

**탐지 방법:**
- Baseline: **빈도 Z-score**로 탐지 (freq_z > 2.0)
- DeepLog: 반복 패턴 인식
- 실무: 하드웨어 오작동, 루프 버그 가능성

---

## 전체 워크플로우

### 📝 전체 프로세스

```
1. 학습 데이터 생성
   ↓
2. 모델 학습 (DeepLog)
   ↓
3. 추론용 정상 데이터 생성 및 테스트
   ↓ (False Positive 확인)
4. 추론용 비정상 데이터 생성 및 테스트
   ↓ (True Positive 확인)
5. 성능 평가 및 리포트 생성
```

### 🚀 한 번에 실행하기

**데모 스크립트 사용:**
```bash
# 전체 워크플로우 자동 실행
cd scripts/demo
chmod +x demo_train_inference_workflow.sh
./demo_train_inference_workflow.sh
```

**수동 실행:**
```bash
# 1. 학습 데이터 생성
alog-detect gen-training-data --out data/raw/train.log --lines 10000

# 2. 전처리 및 학습
alog-detect parse --input data/raw/train.log --out-dir data/processed/train
alog-detect build-deeplog --parsed data/processed/train/parsed.parquet --out-dir data/processed/train
alog-detect deeplog-train --seq data/processed/train/sequences.parquet --vocab data/processed/train/vocab.json --out models/deeplog.pth

# 3. 추론용 정상 데이터 테스트
alog-detect gen-inference-normal --out data/raw/test_normal.log --lines 1000
alog-detect parse --input data/raw/test_normal.log --out-dir data/processed/test_normal
alog-detect build-deeplog --parsed data/processed/test_normal/parsed.parquet --out-dir data/processed/test_normal
alog-detect deeplog-infer --seq data/processed/test_normal/sequences.parquet --model models/deeplog.pth
alog-detect detect --parsed data/processed/test_normal/parsed.parquet --out-dir data/processed/test_normal
alog-detect report --processed-dir data/processed/test_normal

# 4. 추론용 비정상 데이터 테스트
alog-detect gen-inference-anomaly --out data/raw/test_anomaly.log --lines 1000
alog-detect parse --input data/raw/test_anomaly.log --out-dir data/processed/test_anomaly
alog-detect build-deeplog --parsed data/processed/test_anomaly/parsed.parquet --out-dir data/processed/test_anomaly
alog-detect deeplog-infer --seq data/processed/test_anomaly/sequences.parquet --model models/deeplog.pth
alog-detect detect --parsed data/processed/test_anomaly/parsed.parquet --out-dir data/processed/test_anomaly
alog-detect report --processed-dir data/processed/test_anomaly

# 5. 평가
alog-detect eval --processed-dir data/processed/test_normal --labels data/raw/test_normal.log.labels.parquet
alog-detect eval --processed-dir data/processed/test_anomaly --labels data/raw/test_anomaly.log.labels.parquet
```

---

## 실전 예시

### 예시 1: 기본 워크플로우

```bash
# 학습
alog-detect gen-training-data --out train.log --lines 10000
# ... (전처리, 학습)

# 정상 테스트
alog-detect gen-inference-normal --out test_normal.log --lines 500
# ... (추론)

# 비정상 테스트
alog-detect gen-inference-anomaly --out test_anomaly.log --lines 500
# ... (추론, 평가)
```

### 예시 2: 보안 공격 시나리오만 테스트

```bash
alog-detect gen-inference-anomaly \
    --out data/raw/attack_scenario.log \
    --lines 2000 \
    --anomaly-rate 0.25 \
    --anomaly-types attack
```

### 예시 3: 다양한 이상 타입으로 스트레스 테스트

```bash
alog-detect gen-inference-anomaly \
    --out data/raw/stress_test.log \
    --lines 5000 \
    --anomaly-rate 0.30 \
    --anomaly-types unseen \
    --anomaly-types error \
    --anomaly-types attack \
    --anomaly-types crash \
    --anomaly-types burst
```

**메타데이터 확인:**
```bash
cat data/raw/stress_test.log.meta.json
```

**출력 예시:**
```json
{
  "total_lines": 5000,
  "anomaly_count": 1523,
  "anomaly_rate_actual": 0.3046,
  "anomaly_types_used": ["unseen", "error", "attack", "crash", "burst"],
  "anomaly_type_distribution": {
    "unseen": 256,
    "error": 312,
    "attack": 289,
    "crash": 178,
    "burst": 488
  }
}
```

---

## 💡 베스트 프랙티스

### ✅ DO

1. **학습 데이터는 충분히 생성** (최소 10,000줄 권장)
2. **추론 정상/비정상 모두 테스트** (균형 잡힌 평가)
3. **이상 타입별로 분리 테스트** (각 타입별 탐지율 확인)
4. **메타데이터 확인** (실제 생성된 이상 분포 파악)
5. **리포트 활용** (자동 생성된 분석 참고)

### ❌ DON'T

1. **학습 데이터에 이상 포함 금지** (100% 정상만)
2. **너무 높은 이상률 설정 피하기** (> 30%는 비현실적)
3. **단일 이상 타입만 테스트 금지** (다양한 시나리오 필요)
4. **레이블 파일 무시 금지** (평가에 필수)

---

## 📊 기대 성능 지표

### 정상 데이터 (False Positive)

| 지표 | 목표값 | 설명 |
|------|--------|------|
| DeepLog 위반율 | < 20% | 정상을 이상으로 오탐 |
| Baseline 이상률 | < 5% | 새 템플릿 비율 |

### 비정상 데이터 (True Positive)

| 지표 | 목표값 | 설명 |
|------|--------|------|
| DeepLog 위반율 | > 50% | 이상 탐지 성공 |
| Baseline 이상률 | > 15% | 이상 윈도우 탐지 |
| Precision | > 0.70 | 탐지 정확도 |
| Recall | > 0.60 | 탐지 재현율 |
| F1-Score | > 0.65 | 종합 성능 |

---

## 🔧 문제 해결

### Q: 학습 데이터가 너무 단조로워요

**A:** 다양한 정상 패턴이 필요하면 여러 파일을 생성 후 병합:
```bash
for i in {1..5}; do
    alog-detect gen-training-data --out train_$i.log --lines 2000
done
cat train_*.log > training_merged.log
```

### Q: 추론용 비정상 데이터의 이상률이 목표와 다릅니다

**A:** `--anomaly-rate`는 목표값입니다. 실제값은 `.meta.json`에서 확인:
```bash
cat test_anomaly.log.meta.json | grep anomaly_rate_actual
```

### Q: 특정 이상 타입만 집중적으로 테스트하고 싶어요

**A:** `--anomaly-types` 옵션으로 선택:
```bash
# 공격 시나리오만
alog-detect gen-inference-anomaly --out attack_only.log --anomaly-types attack

# 에러 + 크래시만
alog-detect gen-inference-anomaly --out errors.log --anomaly-types error --anomaly-types crash
```

---

## 📚 추가 자료

- **전체 워크플로우 데모**: `scripts/demo/demo_train_inference_workflow.sh`
- **README**: 전체 프로젝트 사용법
- **리포트 개선 가이드**: `docs/REPORT_IMPROVEMENT_EXAMPLE.md`

---

**작성일**: 2025-11-06
**버전**: 1.0
