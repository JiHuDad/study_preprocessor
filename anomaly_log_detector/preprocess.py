from __future__ import annotations  # 타입 힌트에서 문자열 리터럴을 앞으로 참조하기 위한 기능 활성화

import re  # 정규표현식을 위한 모듈
import json  # JSON 데이터 처리를 위한 모듈
from dataclasses import dataclass  # 데이터 클래스를 정의하기 위한 데코레이터
from datetime import datetime  # 날짜/시간 처리를 위한 클래스
from typing import Iterable, Iterator, Optional, Tuple, Dict, Any, List  # 타입 힌트를 위한 모듈

import pandas as pd  # 데이터프레임 처리를 위한 라이브러리

try:
    from drain3 import TemplateMiner  # Drain3 로그 템플릿 마이닝 라이브러리
    from drain3.file_persistence import FilePersistence  # Drain3 상태 파일 저장을 위한 모듈
except Exception:  # pragma: no cover  # 라이브러리가 없을 경우 예외 처리
    TemplateMiner = None  # type: ignore  # 라이브러리가 없으면 None으로 설정
    FilePersistence = None  # type: ignore  # 라이브러리가 없으면 None으로 설정


# 마스킹을 위한 정규표현식 패턴들을 미리 컴파일 (성능 최적화)
HEX_ADDR = re.compile(r"0x[0-9a-fA-F]+")  # 16진수 주소 패턴 (예: 0x7f1234abcd)
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")  # IPv4 주소 패턴 (예: 192.168.1.1)
IPV6 = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")  # IPv6 주소 패턴 (예: 2001:0db8:85a3:0000:0000:8a2e:0370:7334)
MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")  # MAC 주소 패턴 (예: 00:1B:44:11:3A:B7)
UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")  # UUID 패턴 (예: 550e8400-e29b-41d4-a716-446655440000)
PID = re.compile(r"\b(?:pid|tid|uid|gid)=\d+\b")  # 프로세스/스레드 ID 필드 패턴 (예: pid=12345)
# 날짜 패턴들 (메시지 본문에 포함된 날짜 마스킹)
DATE_SYSLOG = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b")  # syslog 스타일 날짜 (예: Sep 14, Jan 1)
DATE_ISO = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")  # ISO 날짜 (예: 2024-09-14, 2024/09/14)
DATE_DMY = re.compile(r"\b\d{1,2}[-/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/]?\d{2,4}\b")  # 일-월-년 (예: 14-Sep-2024, 14/Sep/24)
DECIMAL = re.compile(r"(?<![\w./-])-?\d+(?:\.\d+)?(?![\w./-])")  # 십진수 숫자 패턴 (단어나 경로가 아닌 독립적인 숫자)
DEVICE_NUM = re.compile(r"\b([a-zA-Z]+)(\d+)\b")  # 디바이스 번호 패턴 (예: eth0, sda1)
PATH = re.compile(r"(?:(?:/|~)[\w.\-_/]+)")  # 파일 경로 패턴 (예: /usr/bin/python, ~/home/file.txt)

# Syslog 형식 라인 파싱용 정규표현식: "Sep 14 05:04:41 host kernel: [123.456] message..."
SYSLOG_RE = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"  # 타임스탬프 그룹 (월 일 시:분:초)
    r"(?P<host>[\w.\-]+)\s+"  # 호스트명 그룹
    r"(?P<proc>[\w\-/.]+)(?:\[\d+\])?:\s+"  # 프로세스명 그룹 (선택적으로 PID 포함)
    r"(?P<msg>.*)$"  # 메시지 그룹 (나머지 전체)
)

# dmesg 형식 라인 파싱용 정규표현식: "[  123.456789] message..." (호스트/프로세스 없음)
DMESG_RE = re.compile(r"^\[\s*\d+\.\d+\]\s+(?P<msg>.*)$")  # 시간 정보와 메시지만 추출


def mask_message(message: str, cfg: Optional["PreprocessConfig"] = None) -> str:
    """마스킹 규칙을 적용하여 카디널리티를 줄이는 함수.

    순서가 중요함: 구조적 토큰을 과도하게 마스킹하는 것을 피하기 위해
    경로와 ID를 일반 숫자 치환 전에 적용해야 함.
    """
    cfg = cfg or PreprocessConfig()  # 설정이 제공되지 않으면 기본 설정 사용
    masked = message  # 원본 메시지를 복사

    if cfg.mask_paths:  # 경로 마스킹이 활성화되어 있으면
        masked = PATH.sub("<PATH>", masked)  # 파일 경로를 <PATH>로 치환

    # CRITICAL: 날짜를 숫자보다 먼저 마스킹해야 날짜의 숫자 부분이 따로 마스킹되지 않음
    if cfg.mask_dates:  # 날짜 마스킹이 활성화되어 있으면
        masked = DATE_ISO.sub("<DATE>", masked)  # ISO 날짜를 <DATE>로 치환 (2024-09-14)
        masked = DATE_DMY.sub("<DATE>", masked)  # 일-월-년 형식을 <DATE>로 치환 (14-Sep-2024)
        masked = DATE_SYSLOG.sub("<DATE>", masked)  # syslog 스타일 날짜를 <DATE>로 치환 (Sep 14)

    if cfg.mask_hex:  # 16진수 마스킹이 활성화되어 있으면
        masked = HEX_ADDR.sub("<HEX>", masked)  # 16진수 주소를 <HEX>로 치환
    if cfg.mask_ips:  # IP 주소 마스킹이 활성화되어 있으면
        masked = IPV4.sub("<IP>", masked)  # IPv4 주소를 <IP>로 치환
        masked = IPV6.sub("<IP6>", masked)  # IPv6 주소를 <IP6>로 치환
    if cfg.mask_mac:  # MAC 주소 마스킹이 활성화되어 있으면
        masked = MAC.sub("<MAC>", masked)  # MAC 주소를 <MAC>로 치환
    if cfg.mask_uuid:  # UUID 마스킹이 활성화되어 있으면
        masked = UUID.sub("<UUID>", masked)  # UUID를 <UUID>로 치환
    if cfg.mask_pid_fields:  # PID 필드 마스킹이 활성화되어 있으면
        masked = PID.sub(lambda m: m.group(0).split("=")[0] + "=<ID>", masked)  # pid=12345 -> pid=<ID>로 치환
    if cfg.mask_device_numbers:  # 디바이스 번호 마스킹이 활성화되어 있으면
        masked = DEVICE_NUM.sub(lambda m: f"{m.group(1)}<ID>", masked)  # eth0 -> eth<ID>로 치환
    if cfg.mask_numbers:  # 숫자 마스킹이 활성화되어 있으면
        masked = DECIMAL.sub("<NUM>", masked)  # 모든 숫자를 <NUM>으로 치환
    return masked  # 마스킹된 메시지 반환


def parse_line(line: str) -> Tuple[Optional[datetime], Optional[str], Optional[str], str]:
    """로그 라인을 최선을 다해 파싱하여 (타임스탬프, 호스트, 프로세스, 메시지)를 반환하는 함수."""
    line = line.rstrip("\n")  # 라인 끝의 개행 문자 제거
    m = SYSLOG_RE.match(line)  # Syslog 형식에 맞는지 확인
    if m:  # Syslog 형식이 매칭되면
        ts_str = m.group("ts")  # 타임스탬프 문자열 추출
        host = m.group("host")  # 호스트명 추출
        proc = m.group("proc")  # 프로세스명 추출
        msg = m.group("msg")  # 메시지 추출
        # 연도가 없는 타임스탬프; 정렬을 위해 현재 연도를 가정
        try:
            ts = datetime.strptime(ts_str, "%b %d %H:%M:%S").replace(year=datetime.now().year)  # 타임스탬프 문자열을 datetime 객체로 변환 (연도는 현재 연도 사용)
        except Exception:  # 파싱 실패 시
            ts = None  # 타임스탬프를 None으로 설정
        return ts, host, proc, msg  # 파싱된 값들 반환

    m = DMESG_RE.match(line)  # dmesg 형식에 맞는지 확인
    if m:  # dmesg 형식이 매칭되면
        return None, None, None, m.group("msg")  # 호스트/프로세스 없이 메시지만 반환

    # 폴백: 원본 라인을 메시지로 사용
    return None, None, None, line.strip()  # 파싱할 수 없으면 원본 라인을 메시지로 반환


@dataclass  # 데이터 클래스 데코레이터 (자동으로 __init__, __repr__ 등 생성)
class PreprocessConfig:  # 전처리 설정을 저장하는 데이터 클래스
    drain_state_path: Optional[str] = None  # Drain3 상태 파일 경로 (선택적, 영속화를 위해 사용)
    mask_paths: bool = True  # 파일 경로 마스킹 여부
    mask_dates: bool = True  # 날짜 마스킹 여부 (Sep 14, 2024-09-14 등)
    mask_hex: bool = True  # 16진수 주소 마스킹 여부
    mask_ips: bool = True  # IP 주소 마스킹 여부
    mask_mac: bool = True  # MAC 주소 마스킹 여부
    mask_uuid: bool = True  # UUID 마스킹 여부
    mask_pid_fields: bool = True  # PID/TID/UID/GID 필드 마스킹 여부
    mask_device_numbers: bool = True  # 디바이스 번호 마스킹 여부
    mask_numbers: bool = True  # 일반 숫자 마스킹 여부


class LogPreprocessor:  # 로그 전처리기 클래스
    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:  # 초기화 메서드
        self.config = config or PreprocessConfig()  # 설정이 제공되지 않으면 기본 설정 사용
        self._miner = None  # Drain3 템플릿 마이너 인스턴스 (초기값 None)

        if TemplateMiner is not None:  # Drain3 라이브러리가 사용 가능하면
            persistence = None  # 영속화 핸들러 초기화
            if self.config.drain_state_path and FilePersistence is not None:  # 상태 파일 경로가 설정되어 있고 파일 영속화가 가능하면
                persistence = FilePersistence(self.config.drain_state_path)  # 파일 영속화 핸들러 생성
            # 오래된 Drain3 버전은 위치 인자로 persistence 핸들러를 기대하거나
            # 'persistence_handler' 키워드를 사용함. 호환성을 위해 위치 인자 사용.
            if persistence is not None:  # 영속화 핸들러가 있으면
                self._miner = TemplateMiner(persistence)  # 영속화와 함께 템플릿 마이너 생성
            else:  # 영속화 핸들러가 없으면
                self._miner = TemplateMiner()  # 일반 템플릿 마이너 생성

    def iter_rows(self, lines: Iterable[str]) -> Iterator[Dict[str, Any]]:  # 로그 라인들을 순회하여 딕셔너리로 변환하는 이터레이터
        for idx, line in enumerate(lines):  # 각 라인을 인덱스와 함께 순회
            ts, host, proc, msg = parse_line(line)  # 라인을 파싱하여 타임스탬프, 호스트, 프로세스, 메시지 추출
            masked = mask_message(msg, self.config)  # 메시지에 마스킹 규칙 적용

            template_id: Optional[str] = None  # 템플릿 ID 초기화
            template_str: Optional[str] = None  # 템플릿 문자열 초기화

            if self._miner is not None:  # 템플릿 마이너가 사용 가능하면
                result = self._miner.add_log_message(masked)  # 마스킹된 메시지를 마이너에 추가하고 결과 받기
                # 클러스터/템플릿 추출을 위한 견고한 로직
                cluster_id = None  # 클러스터 ID 초기화
                if isinstance(result, dict):  # 결과가 딕셔너리인 경우
                    cluster_id = result.get("cluster_id") or result.get("cluster_id_str")  # 클러스터 ID 추출 (두 가지 키 시도)
                else:  # 결과가 딕셔너리가 아닌 경우
                    cluster_id = getattr(result, "cluster_id", None)  # 객체의 속성으로 클러스터 ID 추출

                if cluster_id is not None:  # 클러스터 ID가 있으면
                    template_id = str(cluster_id)  # 클러스터 ID를 문자열로 변환
                    try:
                        cluster = self._miner.drain.id_to_cluster.get(cluster_id)  # type: ignore[attr-defined]  # 클러스터 ID로 클러스터 객체 가져오기
                        if cluster is not None:  # 클러스터 객체가 있으면
                            template_str = cluster.get_template()  # 클러스터에서 템플릿 문자열 추출
                    except Exception:  # 예외 발생 시
                        template_str = None  # 템플릿 문자열을 None으로 설정

            yield {  # 각 라인에 대한 처리 결과를 딕셔너리로 반환
                "line_no": idx,  # 라인 번호
                "timestamp": ts,  # 타임스탬프
                "host": host,  # 호스트명
                "process": proc,  # 프로세스명
                "raw": msg,  # 원본 메시지
                "masked": masked,  # 마스킹된 메시지
                "template_id": template_id,  # 템플릿 ID
                "template": template_str,  # 템플릿 문자열
            }

    def process_file(self, input_path: str) -> pd.DataFrame:  # 파일을 읽어서 데이터프레임으로 반환하는 메서드
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:  # 파일을 UTF-8 인코딩으로 열기 (오류는 무시)
            rows = list(self.iter_rows(f))  # 파일의 모든 라인을 처리하여 리스트로 변환
        df = pd.DataFrame(rows)  # 리스트를 pandas 데이터프레임으로 변환
        # 안정적인 정렬 보장: 타임스탬프 우선, 그 다음 라인 번호
        df = df.sort_values(by=["timestamp", "line_no"], kind="stable", na_position="first")  # 타임스탬프와 라인 번호로 안정 정렬 (NA값은 앞쪽에)
        return df  # 정렬된 데이터프레임 반환


