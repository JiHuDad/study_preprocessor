from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_mscred_window_counts(
    parsed_parquet: str | Path,
    out_dir: str | Path,
    template_col: str = "template_id",
    window_size: int = 50,
    stride: int = 25,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parsed_parquet)
    df = df.sort_values(["timestamp", "line_no"], kind="stable", na_position="first")

    # Factorize template ids
    codes, uniques = pd.factorize(df[template_col].astype("string"), sort=True)
    df = df.assign(template_code=codes)

    rows = []
    for start in range(0, len(df), stride):
        end = start + window_size
        window = df.iloc[start:end]
        if window.empty:
            break
        counts = window["template_code"].value_counts().to_dict()
        row = {f"t{int(k)}": int(v) for k, v in counts.items()}
        row["start_index"] = int(window.iloc[0]["line_no"]) if "line_no" in window.columns else int(start)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out / "window_counts.parquet", index=False)


