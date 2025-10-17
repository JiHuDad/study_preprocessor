from __future__ import annotations

import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Optional, Tuple, Dict, Any, List

import pandas as pd

try:
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
except Exception:  # pragma: no cover
    TemplateMiner = None  # type: ignore
    FilePersistence = None  # type: ignore


# Precompile regex patterns for masking
HEX_ADDR = re.compile(r"0x[0-9a-fA-F]+")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
IPV6 = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")
MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
PID = re.compile(r"\b(?:pid|tid|uid|gid)=\d+\b")
DECIMAL = re.compile(r"(?<![\w./-])-?\d+(?:\.\d+)?(?![\w./-])")
DEVICE_NUM = re.compile(r"\b([a-zA-Z]+)(\d+)\b")  # eth0, sda1
PATH = re.compile(r"(?:(?:/|~)[\w.\-_/]+)")

# Syslog-like line: "Sep 14 05:04:41 host kernel: [123.456] message..."
SYSLOG_RE = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>[\w.\-]+)\s+"
    r"(?P<proc>[\w\-/.]+)(?:\[\d+\])?:\s+"
    r"(?P<msg>.*)$"
)

# dmesg-like line: "[  123.456789] message..." (no host/proc)
DMESG_RE = re.compile(r"^\[\s*\d+\.\d+\]\s+(?P<msg>.*)$")


def mask_message(message: str, cfg: Optional["PreprocessConfig"] = None) -> str:
    """Apply masking rules to reduce cardinality.

    Order matters: apply path and IDs before generic numeric replaces to avoid
    over-masking structural tokens.
    """
    cfg = cfg or PreprocessConfig()
    masked = message

    if cfg.mask_paths:
        masked = PATH.sub("<PATH>", masked)
    if cfg.mask_hex:
        masked = HEX_ADDR.sub("<HEX>", masked)
    if cfg.mask_ips:
        masked = IPV4.sub("<IP>", masked)
        masked = IPV6.sub("<IP6>", masked)
    if cfg.mask_mac:
        masked = MAC.sub("<MAC>", masked)
    if cfg.mask_uuid:
        masked = UUID.sub("<UUID>", masked)
    if cfg.mask_pid_fields:
        masked = PID.sub(lambda m: m.group(0).split("=")[0] + "=<ID>", masked)
    if cfg.mask_device_numbers:
        masked = DEVICE_NUM.sub(lambda m: f"{m.group(1)}<ID>", masked)
    if cfg.mask_numbers:
        masked = DECIMAL.sub("<NUM>", masked)
    return masked


def parse_line(line: str) -> Tuple[Optional[datetime], Optional[str], Optional[str], str]:
    """Best-effort parse of a log line, returning (timestamp, host, proc, message)."""
    line = line.rstrip("\n")
    m = SYSLOG_RE.match(line)
    if m:
        ts_str = m.group("ts")
        host = m.group("host")
        proc = m.group("proc")
        msg = m.group("msg")
        # Year-less timestamp; assume current year for ordering
        try:
            ts = datetime.strptime(ts_str, "%b %d %H:%M:%S").replace(year=datetime.now().year)
        except Exception:
            ts = None
        return ts, host, proc, msg

    m = DMESG_RE.match(line)
    if m:
        return None, None, None, m.group("msg")

    # Fallback: raw line as message
    return None, None, None, line.strip()


@dataclass
class PreprocessConfig:
    drain_state_path: Optional[str] = None
    mask_paths: bool = True
    mask_hex: bool = True
    mask_ips: bool = True
    mask_mac: bool = True
    mask_uuid: bool = True
    mask_pid_fields: bool = True
    mask_device_numbers: bool = True
    mask_numbers: bool = True


class LogPreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()
        self._miner = None

        if TemplateMiner is not None:
            persistence = None
            if self.config.drain_state_path and FilePersistence is not None:
                persistence = FilePersistence(self.config.drain_state_path)
            # Older Drain3 versions expect positional persistence handler or the
            # keyword 'persistence_handler'. Use positional for compatibility.
            if persistence is not None:
                self._miner = TemplateMiner(persistence)
            else:
                self._miner = TemplateMiner()

    def iter_rows(self, lines: Iterable[str]) -> Iterator[Dict[str, Any]]:
        for idx, line in enumerate(lines):
            ts, host, proc, msg = parse_line(line)
            masked = mask_message(msg, self.config)

            template_id: Optional[str] = None
            template_str: Optional[str] = None

            if self._miner is not None:
                result = self._miner.add_log_message(masked)
                # Robust extraction of cluster/template
                cluster_id = None
                if isinstance(result, dict):
                    cluster_id = result.get("cluster_id") or result.get("cluster_id_str")
                else:
                    cluster_id = getattr(result, "cluster_id", None)

                if cluster_id is not None:
                    template_id = str(cluster_id)
                    try:
                        cluster = self._miner.drain.id_to_cluster.get(cluster_id)  # type: ignore[attr-defined]
                        if cluster is not None:
                            template_str = cluster.get_template()
                    except Exception:
                        template_str = None

            yield {
                "line_no": idx,
                "timestamp": ts,
                "host": host,
                "process": proc,
                "raw": msg,
                "masked": masked,
                "template_id": template_id,
                "template": template_str,
            }

    def process_file(self, input_path: str) -> pd.DataFrame:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            rows = list(self.iter_rows(f))
        df = pd.DataFrame(rows)
        # Ensure stable ordering: timestamp then line_no
        df = df.sort_values(by=["timestamp", "line_no"], kind="stable", na_position="first")
        return df


