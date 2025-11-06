from __future__ import annotations  # 타입 힌트에서 문자열 리터럴을 앞으로 참조하기 위한 기능 활성화

import random  # 랜덤 숫자 생성 모듈
from datetime import datetime, timedelta  # 날짜/시간 처리 모듈
from pathlib import Path  # 경로 처리를 위한 모듈
import pandas as pd  # 데이터프레임 처리를 위한 라이브러리


BASE_TEMPLATES = [  # 정상 로그 메시지 템플릿 리스트 (일반적인 시스템 로그 패턴)
    "usb 1-1: new high-speed USB device number {n} using ehci-pci",  # USB 장치 연결 메시지 템플릿
    "CPU{c}: Core temperature above threshold, cpu clock throttled",  # CPU 온도 경고 메시지 템플릿
    "CPU{c}: Core temperature/speed normal",  # CPU 온도 정상 복구 메시지 템플릿
    "eth{e}: Link is Up - 1000Mbps/Full - flow control rx/tx",  # 이더넷 링크 UP 메시지 템플릿
    "EXT4-fs (sda{p}): mounted filesystem with ordered data mode. Opts: (null)",  # 파일시스템 마운트 메시지 템플릿
    "audit: type=1400 audit({a1}:{a2}): apparmor=\"DENIED\" operation=\"open\" profile=\"snap.snap-store.ubuntu-software\" name=\"/etc/shadow\" pid={pid} comm=\"snap-store\" requested_mask=\"r\" denied_mask=\"r\" fsuid=1000 ouid=0",  # 감사(audit) 로그 거부 메시지 템플릿
    "usb 1-1: USB disconnect, device number {n}",  # USB 장치 연결 해제 메시지 템플릿
]


ANOMALY_TEMPLATES = [  # 이상 로그 메시지 템플릿 리스트 (이상 탐지 테스트용)
    # unseen style  # 보이지 않는(unseen) 패턴 스타일
    "nvme{n}: I/O error on namespace {n}",  # NVMe I/O 오류 메시지 템플릿 (이상 패턴)
    "kernel BUG at {path}:{line}",  # 커널 버그 메시지 템플릿 (이상 패턴)
    # frequency burst will be simulated by repeating existing templates  # 빈도 급증(frequency burst)은 기존 템플릿 반복으로 시뮬레이션됨
]

# 추가 이상 템플릿들 (다양한 시나리오 테스트용)
ERROR_TEMPLATES = [  # 에러 메시지 템플릿
    "ERROR: disk I/O error, dev sd{d}, sector {sec}",
    "CRITICAL: Out of memory: Kill process {pid} (systemd)",
    "WARNING: Temperature above threshold on CPU{c}",
    "FATAL: kernel panic - not syncing: VFS: Unable to mount root fs",
    "ERROR: segmentation fault at {addr} ip {ip} sp {sp} error {err}",
]

ATTACK_TEMPLATES = [  # 보안 공격 시뮬레이션 템플릿
    "sshd[{pid}]: Failed password for invalid user admin from {ip} port {port} ssh2",
    "sshd[{pid}]: Failed password for root from {ip} port {port} ssh2",
    "kernel: TCP: Possible SYN flooding on port {port}. Sending cookies.",
    "sudo: pam_unix(sudo:auth): authentication failure; logname=USER uid={uid} euid=0 tty=/dev/pts/{n} ruser=attacker rhost=  user=root",
]

CRASH_TEMPLATES = [  # 시스템 크래시 시뮬레이션 템플릿
    "systemd[1]: Failed to start {service}.service.",
    "systemd[1]: {service}.service: Main process exited, code=killed, status={sig}/KILL",
    "kernel: Oops: 0002 [#{n}] SMP",
    "kernel: RIP: 0010:{func}+{offset}/{size}",
]


def _fmt_syslog(ts: datetime, host: str, proc: str, msg: str) -> str:
    """Syslog 형식으로 로그 라인을 포맷팅하는 헬퍼 함수."""
    ts_str = ts.strftime("%b %d %H:%M:%S")  # 타임스탬프를 "월 일 시:분:초" 형식으로 변환 (예: "Jan 15 14:30:45")
    return f"{ts_str} {host} {proc}: [  {random.randint(0,99999)}.{random.randint(0,999999):06d}] {msg}"  # Syslog 형식 문자열 반환: 타임스탬프, 호스트, 프로세스, 커널 시간, 메시지


def _format_template(tpl: str) -> str:
    """템플릿의 플레이스홀더를 랜덤 값으로 채우는 헬퍼 함수.

    주의: 이 함수는 템플릿 타입(normal, anomaly, error 등)에 관계없이
    동일한 랜덤 값을 생성합니다. 템플릿 자체가 이미 타입을 구분하므로
    (BASE_TEMPLATES, ERROR_TEMPLATES, ATTACK_TEMPLATES 등)
    이 함수에서 타입별 차별화는 필요하지 않습니다.

    수정 이력:
    - 이전에 template_type 파라미터가 있었으나 함수 내에서 전혀 사용되지 않았음
    - Dead parameter로 인한 혼란 방지를 위해 파라미터 제거
    - 호출처: generate_synthetic_log(), generate_training_data(),
             generate_inference_normal(), generate_inference_anomaly()

    Args:
        tpl: 플레이스홀더를 포함한 템플릿 문자열
             예: "usb 1-1: new high-speed USB device number {n} using ehci-pci"

    Returns:
        플레이스홀더가 랜덤 값으로 치환된 문자열
    """
    values = {
        "n": random.randint(1, 9),
        "c": random.randint(0, 3),
        "e": random.randint(0, 3),
        "d": chr(ord('a') + random.randint(0, 3)),  # a-d
        "p": random.randint(1, 3),
        "a1": random.randint(10000, 99999),
        "a2": random.randint(1, 9),
        "pid": random.randint(100, 9999),
        "path": f"/usr/src/linux/mm/page_alloc.c",
        "line": random.randint(10, 999),
        "sec": random.randint(1000, 999999),
        "addr": f"0x{random.randint(0, 0xFFFFFF):06x}",
        "ip": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
        "port": random.randint(20000, 65535),
        "uid": random.randint(1000, 2000),
        "sig": random.randint(1, 15),
        "service": random.choice(["docker", "mysql", "nginx", "postgresql", "redis"]),
        "func": random.choice(["do_syscall_64", "handle_mm_fault", "page_fault"]),
        "offset": f"0x{random.randint(0, 0xFF):02x}",
        "size": f"0x{random.randint(0x100, 0xFFF):03x}",
        "err": random.randint(4, 7),
        "sp": f"0x{random.randint(0, 0xFFFFFFFFFFFF):012x}",
    }
    return tpl.format(**values)


def generate_synthetic_log(
    out_path: str | Path,  # 출력 로그 파일 경로
    num_lines: int = 5000,  # 생성할 로그 라인 수 (기본값: 5000줄)
    anomaly_rate: float = 0.02,  # 이상 로그 비율 (기본값: 2%, 즉 100줄 중 2줄이 이상)
    host: str = "host1",  # 호스트명 (기본값: "host1")
    proc: str = "kernel",  # 프로세스명 (기본값: "kernel")
    start_time: datetime | None = None,  # 시작 시간 (None이면 현재 시간 사용)
) -> Path:
    """합성 로그 파일을 생성하는 함수 (이상 탐지 알고리즘 테스트용).

    정상 로그와 이상 로그를 섞어서 생성하며, 레이블 파일도 함께 저장합니다.
    """
    out = Path(out_path)  # 경로 문자열을 Path 객체로 변환
    out.parent.mkdir(parents=True, exist_ok=True)  # 출력 디렉토리가 없으면 생성
    now = start_time or datetime.now().replace(microsecond=0)  # 시작 시간 설정 (제공되지 않으면 현재 시간, 마이크로초 제거)

    labels: list[tuple[int, int]] = []  # 레이블 리스트 초기화 (라인 번호, 이상 여부)
    with out.open("w", encoding="utf-8") as f:  # 로그 파일을 UTF-8 인코딩으로 쓰기 모드로 열기
        for i in range(num_lines):  # 지정된 라인 수만큼 반복
            ts = now + timedelta(seconds=i)  # 각 라인마다 1초씩 증가하는 타임스탬프 생성
            is_anom = random.random() < anomaly_rate  # 랜덤 값이 이상 비율보다 작으면 이상 로그로 설정
            if is_anom:  # 이상 로그인 경우
                # pick anomaly: unseen or burst of errors  # 이상 선택: 보이지 않는 패턴 또는 오류 급증
                tpl = random.choice(ANOMALY_TEMPLATES)  # 이상 템플릿 중 하나를 랜덤 선택
                msg = _format_template(tpl)
            else:  # 정상 로그인 경우
                tpl = random.choice(BASE_TEMPLATES)  # 정상 템플릿 중 하나를 랜덤 선택
                msg = _format_template(tpl)
            line = _fmt_syslog(ts, host, proc, msg)  # Syslog 형식으로 로그 라인 포맷팅
            f.write(line + "\n")  # 파일에 로그 라인 쓰기
            labels.append((i, 1 if is_anom else 0))  # 레이블 저장 (라인 번호, 이상 여부: 1=이상, 0=정상)
    # Save labels next to the log file  # 레이블을 로그 파일 옆에 저장
    lab_path = Path(str(out) + ".labels.parquet")  # 레이블 파일 경로 생성 (로그 파일명.labels.parquet)
    pd.DataFrame(labels, columns=["line_no", "is_anomaly"]).to_parquet(lab_path, index=False)  # 레이블을 Parquet 형식으로 저장
    return out  # 생성된 로그 파일 경로 반환


def generate_training_data(
    out_path: str | Path,
    num_lines: int = 10000,
    host: str = "train-host",
    proc: str = "kernel",
    start_time: datetime | None = None,
) -> Path:
    """학습용 정상 로그 데이터를 생성합니다.

    특징:
    - 100% 정상 로그만 포함
    - 다양한 정상 패턴 학습용
    - 일관된 시간 간격
    - 레이블 파일 포함 (모두 is_anomaly=0)

    Args:
        out_path: 출력 파일 경로
        num_lines: 생성할 로그 라인 수 (기본: 10000)
        host: 호스트명
        proc: 프로세스명
        start_time: 시작 시간

    Returns:
        생성된 로그 파일 경로
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    now = start_time or datetime.now().replace(microsecond=0)

    labels: list[tuple[int, int]] = []
    with out.open("w", encoding="utf-8") as f:
        for i in range(num_lines):
            ts = now + timedelta(seconds=i)
            # 정상 템플릿만 사용
            tpl = random.choice(BASE_TEMPLATES)
            msg = _format_template(tpl)
            line = _fmt_syslog(ts, host, proc, msg)
            f.write(line + "\n")
            labels.append((i, 0))  # 모두 정상(0)

    # 레이블 저장
    lab_path = Path(str(out) + ".labels.parquet")
    pd.DataFrame(labels, columns=["line_no", "is_anomaly"]).to_parquet(lab_path, index=False)
    return out


def generate_inference_normal(
    out_path: str | Path,
    num_lines: int = 1000,
    host: str = "test-host",
    proc: str = "kernel",
    start_time: datetime | None = None,
) -> Path:
    """추론용 정상 로그 데이터를 생성합니다 (False Positive 테스트용).

    특징:
    - 100% 정상 로그
    - 학습 데이터와 유사하지만 약간 다른 변형
    - 모델이 정상으로 인식해야 함
    - 레이블 파일 포함 (모두 is_anomaly=0)

    Args:
        out_path: 출력 파일 경로
        num_lines: 생성할 로그 라인 수 (기본: 1000)
        host: 호스트명
        proc: 프로세스명
        start_time: 시작 시간

    Returns:
        생성된 로그 파일 경로
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    now = start_time or datetime.now().replace(microsecond=0)

    labels: list[tuple[int, int]] = []
    with out.open("w", encoding="utf-8") as f:
        for i in range(num_lines):
            ts = now + timedelta(seconds=i)
            tpl = random.choice(BASE_TEMPLATES)
            msg = _format_template(tpl)
            line = _fmt_syslog(ts, host, proc, msg)
            f.write(line + "\n")
            labels.append((i, 0))

    lab_path = Path(str(out) + ".labels.parquet")
    pd.DataFrame(labels, columns=["line_no", "is_anomaly"]).to_parquet(lab_path, index=False)
    return out


def generate_inference_anomaly(
    out_path: str | Path,
    num_lines: int = 1000,
    anomaly_rate: float = 0.15,  # 15% 이상 비율
    anomaly_types: list[str] | None = None,
    host: str = "test-host",
    proc: str = "kernel",
    start_time: datetime | None = None,
) -> Path:
    """추론용 비정상 로그 데이터를 생성합니다 (True Positive 테스트용).

    특징:
    - 정상 로그 + 다양한 이상 시나리오
    - 모델이 이상을 탐지해야 함
    - 레이블 파일 포함 (이상 구간 표시)

    이상 타입:
    - unseen: 학습 시 보지 못한 새로운 템플릿
    - error: 에러 메시지 (ERROR, CRITICAL, FATAL)
    - attack: 보안 공격 시뮬레이션 (SSH brute force, SYN flood)
    - crash: 시스템 크래시 (서비스 실패, kernel panic)
    - burst: 특정 템플릿 급증

    Args:
        out_path: 출력 파일 경로
        num_lines: 생성할 로그 라인 수 (기본: 1000)
        anomaly_rate: 이상 로그 비율 (기본: 0.15 = 15%)
        anomaly_types: 포함할 이상 타입 리스트 (None이면 모두 포함)
        host: 호스트명
        proc: 프로세스명
        start_time: 시작 시간

    Returns:
        생성된 로그 파일 경로
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    now = start_time or datetime.now().replace(microsecond=0)

    # 기본 이상 타입 설정
    if anomaly_types is None:
        anomaly_types = ["unseen", "error", "attack", "crash", "burst"]

    # 이상 타입별 템플릿 매핑
    anomaly_template_map = {
        "unseen": ANOMALY_TEMPLATES,
        "error": ERROR_TEMPLATES,
        "attack": ATTACK_TEMPLATES,
        "crash": CRASH_TEMPLATES,
    }

    labels: list[tuple[int, int, str]] = []  # (line_no, is_anomaly, anomaly_type)
    burst_active = False
    burst_template = None
    burst_count = 0
    burst_total = 0

    with out.open("w", encoding="utf-8") as f:
        for i in range(num_lines):
            ts = now + timedelta(seconds=i)
            is_anom = False
            anom_type = "normal"

            # Burst 처리
            if burst_active and burst_count < burst_total:
                msg = _format_template(burst_template)
                is_anom = True
                anom_type = "burst"
                burst_count += 1
                if burst_count >= burst_total:
                    burst_active = False
            else:
                # 새로운 이상 발생 여부 결정
                if random.random() < anomaly_rate:
                    is_anom = True

                    # Burst 시작 가능성 (10%)
                    if "burst" in anomaly_types and random.random() < 0.1:
                        burst_active = True
                        burst_template = random.choice(BASE_TEMPLATES)  # 정상 템플릿을 급증시킴
                        burst_total = random.randint(10, 30)  # 10-30개 연속
                        burst_count = 0
                        msg = _format_template(burst_template)
                        anom_type = "burst"
                    else:
                        # 다른 이상 타입 선택
                        available_types = [t for t in anomaly_types if t != "burst"]
                        if available_types:
                            anom_type = random.choice(available_types)
                            templates = anomaly_template_map.get(anom_type, ANOMALY_TEMPLATES)
                            tpl = random.choice(templates)
                            msg = _format_template(tpl)
                        else:
                            # Fallback
                            tpl = random.choice(ANOMALY_TEMPLATES)
                            msg = _format_template(tpl)
                            anom_type = "unseen"
                else:
                    # 정상 로그
                    tpl = random.choice(BASE_TEMPLATES)
                    msg = _format_template(tpl)

            line = _fmt_syslog(ts, host, proc, msg)
            f.write(line + "\n")
            labels.append((i, 1 if is_anom else 0, anom_type))

    # 레이블 저장 (anomaly_type 포함)
    lab_path = Path(str(out) + ".labels.parquet")
    pd.DataFrame(labels, columns=["line_no", "is_anomaly", "anomaly_type"]).to_parquet(lab_path, index=False)

    # 통계 출력용 메타데이터 저장
    meta = {
        "total_lines": num_lines,
        "anomaly_count": sum(1 for _, is_anom, _ in labels if is_anom),
        "anomaly_rate_actual": sum(1 for _, is_anom, _ in labels if is_anom) / num_lines,
        "anomaly_types_used": anomaly_types,
        "anomaly_type_distribution": {}
    }

    # 이상 타입별 분포
    for anom_type in set(t for _, is_anom, t in labels if is_anom):
        count = sum(1 for _, is_anom, t in labels if is_anom and t == anom_type)
        meta["anomaly_type_distribution"][anom_type] = count

    import json
    meta_path = Path(str(out) + ".meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return out


