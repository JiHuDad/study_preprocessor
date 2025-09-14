from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


BASE_TEMPLATES = [
    "usb 1-1: new high-speed USB device number {n} using ehci-pci",
    "CPU{c}: Core temperature above threshold, cpu clock throttled",
    "CPU{c}: Core temperature/speed normal",
    "eth{e}: Link is Up - 1000Mbps/Full - flow control rx/tx",
    "EXT4-fs (sda{p}): mounted filesystem with ordered data mode. Opts: (null)",
    "audit: type=1400 audit({a1}:{a2}): apparmor=\"DENIED\" operation=\"open\" profile=\"snap.snap-store.ubuntu-software\" name=\"/etc/shadow\" pid={pid} comm=\"snap-store\" requested_mask=\"r\" denied_mask=\"r\" fsuid=1000 ouid=0",
    "usb 1-1: USB disconnect, device number {n}",
]


ANOMALY_TEMPLATES = [
    # unseen style
    "nvme{n}: I/O error on namespace {n}",
    "kernel BUG at {path}:{line}",
    # frequency burst will be simulated by repeating existing templates
]


def _fmt_syslog(ts: datetime, host: str, proc: str, msg: str) -> str:
    ts_str = ts.strftime("%b %d %H:%M:%S")
    return f"{ts_str} {host} {proc}: [  {random.randint(0,99999)}.{random.randint(0,999999):06d}] {msg}"


def generate_synthetic_log(
    out_path: str | Path,
    num_lines: int = 5000,
    anomaly_rate: float = 0.02,
    host: str = "host1",
    proc: str = "kernel",
    start_time: datetime | None = None,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    now = start_time or datetime.now().replace(microsecond=0)

    labels: list[tuple[int, int]] = []
    with out.open("w", encoding="utf-8") as f:
        for i in range(num_lines):
            ts = now + timedelta(seconds=i)
            is_anom = random.random() < anomaly_rate
            if is_anom:
                # pick anomaly: unseen or burst of errors
                tpl = random.choice(ANOMALY_TEMPLATES)
                msg = tpl.format(
                    n=random.randint(0, 9),
                    path=f"/usr/src/linux/mm/page_alloc.c",
                    line=random.randint(10, 999),
                )
            else:
                tpl = random.choice(BASE_TEMPLATES)
                msg = tpl.format(
                    n=random.randint(1, 9),
                    c=random.randint(0, 3),
                    e=random.randint(0, 3),
                    p=random.randint(1, 3),
                    a1=random.randint(10000, 99999),
                    a2=random.randint(1, 9),
                    pid=random.randint(100, 9999),
                )
            line = _fmt_syslog(ts, host, proc, msg)
            f.write(line + "\n")
            labels.append((i, 1 if is_anom else 0))
    # Save labels next to the log file
    lab_path = Path(str(out) + ".labels.parquet")
    pd.DataFrame(labels, columns=["line_no", "is_anomaly"]).to_parquet(lab_path, index=False)
    return out


