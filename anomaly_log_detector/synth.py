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


def _fmt_syslog(ts: datetime, host: str, proc: str, msg: str) -> str:
    """Syslog 형식으로 로그 라인을 포맷팅하는 헬퍼 함수."""
    ts_str = ts.strftime("%b %d %H:%M:%S")  # 타임스탬프를 "월 일 시:분:초" 형식으로 변환 (예: "Jan 15 14:30:45")
    return f"{ts_str} {host} {proc}: [  {random.randint(0,99999)}.{random.randint(0,999999):06d}] {msg}"  # Syslog 형식 문자열 반환: 타임스탬프, 호스트, 프로세스, 커널 시간, 메시지


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
                msg = tpl.format(  # 템플릿의 플레이스홀더를 랜덤 값으로 치환
                    n=random.randint(0, 9),  # 장치 번호나 네임스페이스 번호
                    path=f"/usr/src/linux/mm/page_alloc.c",  # 파일 경로
                    line=random.randint(10, 999),  # 라인 번호
                )
            else:  # 정상 로그인 경우
                tpl = random.choice(BASE_TEMPLATES)  # 정상 템플릿 중 하나를 랜덤 선택
                msg = tpl.format(  # 템플릿의 플레이스홀더를 랜덤 값으로 치환
                    n=random.randint(1, 9),  # 장치 번호
                    c=random.randint(0, 3),  # CPU 코어 번호
                    e=random.randint(0, 3),  # 이더넷 인터페이스 번호
                    p=random.randint(1, 3),  # 파티션 번호
                    a1=random.randint(10000, 99999),  # 감사(audit) ID 첫 번째 부분
                    a2=random.randint(1, 9),  # 감사(audit) ID 두 번째 부분
                    pid=random.randint(100, 9999),  # 프로세스 ID
                )
            line = _fmt_syslog(ts, host, proc, msg)  # Syslog 형식으로 로그 라인 포맷팅
            f.write(line + "\n")  # 파일에 로그 라인 쓰기
            labels.append((i, 1 if is_anom else 0))  # 레이블 저장 (라인 번호, 이상 여부: 1=이상, 0=정상)
    # Save labels next to the log file  # 레이블을 로그 파일 옆에 저장
    lab_path = Path(str(out) + ".labels.parquet")  # 레이블 파일 경로 생성 (로그 파일명.labels.parquet)
    pd.DataFrame(labels, columns=["line_no", "is_anomaly"]).to_parquet(lab_path, index=False)  # 레이블을 Parquet 형식으로 저장
    return out  # 생성된 로그 파일 경로 반환


