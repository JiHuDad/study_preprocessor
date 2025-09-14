import os
from pathlib import Path
import json
import click
import pandas as pd

from .preprocess import LogPreprocessor, PreprocessConfig


@click.group()
def main() -> None:
    """study-preprocess: 로그 전처리 CLI"""


@main.command()
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--drain-state", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Drain3 상태 파일 경로")
def parse(input_path: Path, out_dir: Path, drain_state: Path | None) -> None:
    """원시 로그 파일을 파싱/마스킹하고 Parquet으로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig(drain_state_path=str(drain_state) if drain_state else None)
    pre = LogPreprocessor(cfg)
    df = pre.process_file(str(input_path))
    parquet_path = out_dir / "parsed.parquet"
    df.to_parquet(parquet_path, index=False)
    # 미리보기용 일부 샘플도 JSON 저장
    preview = df.head(10).to_dict(orient="records")
    (out_dir / "preview.json").write_text(json.dumps(preview, ensure_ascii=False, default=str, indent=2))
    click.echo(f"Saved: {parquet_path}")
    click.echo(f"Preview: {out_dir / 'preview.json'}")


