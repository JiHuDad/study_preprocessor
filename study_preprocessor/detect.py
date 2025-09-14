from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BaselineParams:
    window_size: int = 50
    stride: int = 25
    ewm_alpha: float = 0.3
    anomaly_quantile: float = 0.95


def _build_windows(df: pd.DataFrame, window_size: int, stride: int) -> List[Tuple[int, pd.DataFrame]]:
    rows: List[Tuple[int, pd.DataFrame]] = []
    for start in range(0, len(df), stride):
        end = start + window_size
        window = df.iloc[start:end]
        if window.empty:
            break
        rows.append((start, window))
    return rows


def _counts_per_window(windows: List[Tuple[int, pd.DataFrame]], code_col: str) -> Tuple[pd.DataFrame, List[int], List[Dict[int, int]]]:
    row_dicts: List[Dict[str, int]] = []
    starts: List[int] = []
    raw_counts: List[Dict[int, int]] = []

    # Collect universe of template codes
    all_codes: List[int] = sorted(set(int(x) for _, w in windows for x in w[code_col].dropna().astype(int).tolist()))
    col_names = [f"t{c}" for c in all_codes]

    for start, w in windows:
        vc = w[code_col].value_counts().astype(int).to_dict()
        raw_counts.append(vc)
        row = {f"t{int(k)}": int(v) for k, v in vc.items()}
        # Fill missing columns later by reindex
        row_dicts.append(row)
        starts.append(int(w.iloc[0]["line_no"]) if "line_no" in w.columns else int(start))

    counts_df = pd.DataFrame(row_dicts)
    counts_df = counts_df.reindex(columns=col_names, fill_value=0)
    counts_df.insert(0, "window_start_line", starts)
    return counts_df, starts, raw_counts


def baseline_detect(parsed_parquet: str | Path, out_dir: str | Path, params: BaselineParams = BaselineParams()) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parsed_parquet)
    df = df.sort_values(["timestamp", "line_no"], kind="stable", na_position="first")

    # Factorize template ids to integer codes
    codes, uniques = pd.factorize(df["template_id"].astype("string"), sort=True)
    df = df.assign(template_code=codes)

    windows = _build_windows(df, params.window_size, params.stride)
    if not windows:
        raise ValueError("No windows produced. Consider smaller window_size/stride.")

    counts_df, starts, raw_counts = _counts_per_window(windows, "template_code")
    count_cols = [c for c in counts_df.columns if c.startswith("t")]

    # EWM mean/std per template across windows
    ewm_mean = counts_df[count_cols].ewm(alpha=params.ewm_alpha, adjust=False).mean()
    ewm_var = counts_df[count_cols].ewm(alpha=params.ewm_alpha, adjust=False).var(bias=False)
    ewm_std = (ewm_var.clip(lower=1e-9)) ** 0.5
    z = (counts_df[count_cols] - ewm_mean) / ewm_std
    z = z.clip(lower=0)  # only positive spikes
    freq_z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).max(axis=1)

    # Unseen rate by window (online set of seen templates)
    seen: set[int] = set()
    unseen_rates: List[float] = []
    for vc in raw_counts:
        keys = set(int(k) for k in vc.keys())
        new = len(keys - seen)
        denom = max(1, len(keys))
        unseen_rates.append(new / denom)
        seen |= keys

    scores = pd.DataFrame({
        "window_start_line": counts_df["window_start_line"].astype(int),
        "unseen_rate": unseen_rates,
        "freq_z": freq_z.astype(float),
    })

    # Normalize and combine
    # Robust scaling: freq_z normalized by 95th percentile
    freq_cap = max(1e-9, float(np.quantile(scores["freq_z"], 0.95)))
    scores["freq_norm"] = (scores["freq_z"].clip(0, freq_cap) / freq_cap).astype(float)
    scores["unseen_norm"] = scores["unseen_rate"].clip(0, 1).astype(float)
    scores["score"] = 0.6 * scores["freq_norm"] + 0.4 * scores["unseen_norm"]

    thr = float(np.quantile(scores["score"], params.anomaly_quantile))
    scores["is_anomaly"] = (scores["score"] >= thr).astype(bool)

    out_path = out / "baseline_scores.parquet"
    scores.to_parquet(out_path, index=False)

    # Lightweight preview JSON
    preview = scores.head(20).to_dict(orient="records")
    (out / "baseline_preview.json").write_text(pd.Series(preview).to_json(orient="values"))
    return out_path


