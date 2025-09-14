from __future__ import annotations

from pathlib import Path
import json
from typing import Dict
import pandas as pd


def build_deeplog_inputs(parsed_parquet: str | Path, out_dir: str | Path, template_col: str = "template_id") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parsed_parquet)

    # Build vocab mapping
    unique_templates = [t for t in df[template_col].dropna().astype(str).unique()]
    vocab: Dict[str, int] = {t: i for i, t in enumerate(sorted(unique_templates))}
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2))

    # Map to indices and export sequences by host (if available) or global
    df = df.sort_values(["timestamp", "line_no"], kind="stable", na_position="first")
    df["template_index"] = df[template_col].map(vocab).astype("Int64")
    df[["line_no", "timestamp", "host", "template_index"]].to_parquet(out / "sequences.parquet", index=False)


