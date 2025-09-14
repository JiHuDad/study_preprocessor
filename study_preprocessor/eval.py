from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _align_labels_to_windows(labels_df: pd.DataFrame, scores_df: pd.DataFrame, window_size: int) -> pd.Series:
    # Mark a window anomalous if any labeled anomaly falls within [start, start+window_size)
    win_labels = []
    label_idx = 0
    labels = labels_df.sort_values("line_no").reset_index(drop=True)
    anom_lines = labels[labels["is_anomaly"] == 1]["line_no"].tolist()
    for _, row in scores_df.iterrows():
        start = int(row["window_start_line"]) if "window_start_line" in row else int(row.get("start_index", 0))
        end = start + window_size
        # any anomaly line in [start, end)
        flag = any(start <= a < end for a in anom_lines)
        win_labels.append(1 if flag else 0)
    return pd.Series(win_labels, name="win_label")


def evaluate_baseline(baseline_scores: str | Path, labels_path: str | Path, window_size: int) -> Tuple[float, float, float]:
    s = pd.read_parquet(baseline_scores)
    y_true = _align_labels_to_windows(pd.read_parquet(labels_path), s, window_size)
    y_pred = (s["is_anomaly"].astype(bool)).astype(int)
    return _prf1(y_true.values, y_pred.values)


def evaluate_deeplog(infer_parquet: str | Path, labels_path: str | Path, seq_len: int) -> Tuple[float, float, float]:
    d = pd.read_parquet(infer_parquet)
    # Align by sequence index -> approximate line_no = idx + seq_len
    d = d.assign(line_no=d["idx"].astype(int) + int(seq_len))
    # Mark anomaly if in_topk == False
    d = d.assign(is_anom=(~d["in_topk"]).astype(int))

    labels = pd.read_parquet(labels_path)[["line_no", "is_anomaly"]]
    merged = pd.merge(d, labels, on="line_no", how="inner")
    y_true = merged["is_anomaly"].astype(int).values
    y_pred = merged["is_anom"].astype(int).values
    return _prf1(y_true, y_pred)


def _prf1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


