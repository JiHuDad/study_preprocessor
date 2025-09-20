import os
from pathlib import Path
import json
import click
import pandas as pd

from .preprocess import LogPreprocessor, PreprocessConfig
from .detect import baseline_detect, BaselineParams
from .builders.deeplog import build_deeplog_inputs, train_deeplog, infer_deeplog_topk
from .builders.mscred import build_mscred_window_counts
from .synth import generate_synthetic_log
from .eval import evaluate_baseline, evaluate_deeplog


@click.group()
def main() -> None:
    """study-preprocess: 로그 전처리 CLI"""


@main.command()
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--drain-state", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Drain3 상태 파일 경로")
# Masking toggles
@click.option("--no-mask-paths", is_flag=True, default=False)
@click.option("--no-mask-hex", is_flag=True, default=False)
@click.option("--no-mask-ips", is_flag=True, default=False)
@click.option("--no-mask-mac", is_flag=True, default=False)
@click.option("--no-mask-uuid", is_flag=True, default=False)
@click.option("--no-mask-pid", is_flag=True, default=False)
@click.option("--no-mask-device", is_flag=True, default=False)
@click.option("--no-mask-num", is_flag=True, default=False)
def parse(input_path: Path, out_dir: Path, drain_state: Path | None,
          no_mask_paths: bool, no_mask_hex: bool, no_mask_ips: bool, no_mask_mac: bool,
          no_mask_uuid: bool, no_mask_pid: bool, no_mask_device: bool, no_mask_num: bool) -> None:
    """원시 로그 파일을 파싱/마스킹하고 Parquet으로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig(
        drain_state_path=str(drain_state) if drain_state else None,
        mask_paths=not no_mask_paths,
        mask_hex=not no_mask_hex,
        mask_ips=not no_mask_ips,
        mask_mac=not no_mask_mac,
        mask_uuid=not no_mask_uuid,
        mask_pid_fields=not no_mask_pid,
        mask_device_numbers=not no_mask_device,
        mask_numbers=not no_mask_num,
    )
    pre = LogPreprocessor(cfg)
    df = pre.process_file(str(input_path))
    parquet_path = out_dir / "parsed.parquet"
    df.to_parquet(parquet_path, index=False)
    # 미리보기용 일부 샘플도 JSON 저장
    preview = df.head(10).to_dict(orient="records")
    (out_dir / "preview.json").write_text(json.dumps(preview, ensure_ascii=False, default=str, indent=2))
    click.echo(f"Saved: {parquet_path}")
    click.echo(f"Preview: {out_dir / 'preview.json'}")


@main.command()
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--window-size", type=int, default=50)
@click.option("--stride", type=int, default=25)
@click.option("--ewm-alpha", type=float, default=0.3)
@click.option("--q", "anomaly_q", type=float, default=0.95, help="Anomaly quantile threshold")
def detect(parsed_parquet: Path, out_dir: Path, window_size: int, stride: int, ewm_alpha: float, anomaly_q: float) -> None:
    """베이스라인 이상탐지(새 템플릿 비율 + 빈도 급변) 실행."""
    params = BaselineParams(window_size=window_size, stride=stride, ewm_alpha=ewm_alpha, anomaly_quantile=anomaly_q)
    out_path = baseline_detect(str(parsed_parquet), str(out_dir), params)
    click.echo(f"Saved baseline scores: {out_path}")


@main.command("build-deeplog")
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
def build_deeplog_cmd(parsed_parquet: Path, out_dir: Path) -> None:
    """DeepLog 입력(vocab, sequences) 생성."""
    build_deeplog_inputs(str(parsed_parquet), str(out_dir))
    click.echo(f"Built DeepLog inputs under: {out_dir}")


@main.command("deeplog-train")
@click.option("--seq", "sequences_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--vocab", "vocab_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "model_out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--seq-len", type=int, default=50)
@click.option("--epochs", type=int, default=3)
def deeplog_train_cmd(sequences_parquet: Path, vocab_json: Path, model_out: Path, seq_len: int, epochs: int) -> None:
    path = train_deeplog(str(sequences_parquet), str(vocab_json), str(model_out), seq_len=seq_len, epochs=epochs)
    click.echo(f"Saved DeepLog model: {path}")


@main.command("deeplog-infer")
@click.option("--seq", "sequences_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--k", type=int, default=3)
def deeplog_infer_cmd(sequences_parquet: Path, model_path: Path, k: int) -> None:
    df = infer_deeplog_topk(str(sequences_parquet), str(model_path), k=k)
    out = Path(sequences_parquet).with_name("deeplog_infer.parquet")
    df.to_parquet(out, index=False)
    rate = 1.0 - float(df["in_topk"].mean()) if len(df) > 0 else 0.0
    click.echo(f"Saved inference: {out} (violation_rate={rate:.3f})")


@main.command("build-mscred")
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--window-size", type=int, default=50)
@click.option("--stride", type=int, default=25)
def build_mscred_cmd(parsed_parquet: Path, out_dir: Path, window_size: int, stride: int) -> None:
    """MS-CRED 입력(윈도우 카운트) 생성."""
    build_mscred_window_counts(str(parsed_parquet), str(out_dir), window_size=window_size, stride=stride)
    click.echo(f"Built MS-CRED window counts under: {out_dir}")


@main.command("report")
@click.option("--processed-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--with-samples", is_flag=True, default=False, help="이상 로그 샘플 분석 포함")
def report_cmd(processed_dir: Path, with_samples: bool) -> None:
    """산출물 요약 리포트 생성."""
    import pandas as pd
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    # Baseline
    base_path = processed_dir / "baseline_scores.parquet"
    if base_path.exists():
        s = pd.read_parquet(base_path)
        if len(s) > 0:
            rate = float((s["is_anomaly"] == True).mean())
            top = s.sort_values("score", ascending=False).head(5)
            report_lines.append(f"Baseline anomaly windows: {rate:.3f}")
            report_lines.append("Top windows (start_line, score): " + ", ".join([f"{int(r.window_start_line)}:{float(r.score):.3f}" for _, r in top.iterrows()]))
    # DeepLog
    infer_path = processed_dir / "deeplog_infer.parquet"
    if infer_path.exists():
        d = pd.read_parquet(infer_path)
        if len(d) > 0:
            viol = 1.0 - float(d["in_topk"].mean())
            report_lines.append(f"DeepLog violation rate: {viol:.3f}")
    # Top templates/messages if parsed exists and baseline flagged windows exist
    parsed = processed_dir / "parsed.parquet"
    base = processed_dir / "baseline_scores.parquet"
    if parsed.exists() and base.exists():
        import pandas as pd
        dfp = pd.read_parquet(parsed)
        s = pd.read_parquet(base)
        flagged = s[s["is_anomaly"] == True].copy()
        if len(flagged) > 0 and "template_id" in dfp.columns:
            # For each flagged window, find dominant template_id
            lines = []
            for _, row in flagged.head(5).iterrows():
                start = int(row["window_start_line"]) if "window_start_line" in row else 0
                win = dfp[(dfp["line_no"] >= start) & (dfp["line_no"] < start + 50)]
                top = (
                    win["template_id"].astype(str).value_counts().head(3).to_dict()
                )
                lines.append(f"window@{start} top_templates={top}")
            report_lines.extend(lines)
    
    # 로그 샘플 분석 추가
    if with_samples:
        click.echo("🔍 이상 로그 샘플 분석 중...")
        try:
            import sys
            import subprocess
            result = subprocess.run([
                sys.executable, "log_sample_analyzer.py",
                str(processed_dir),
                "--output-dir", str(processed_dir / "log_samples_analysis")
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                report_lines.append("Log sample analysis completed successfully")
                report_lines.append(f"Detailed analysis: {processed_dir / 'log_samples_analysis' / 'anomaly_analysis_report.md'}")
            else:
                report_lines.append(f"Log sample analysis failed: {result.stderr}")
        except Exception as e:
            report_lines.append(f"Log sample analysis error: {e}")
    
    # Save
    out_md = processed_dir / "report.md"
    if not report_lines:
        report_lines = ["No artifacts found to report."]
    out_md.write_text("\n".join(["### Detection Report"] + [f"- {line}" for line in report_lines]))
    click.echo(f"Saved report: {out_md}")
    
    if with_samples:
        sample_report = processed_dir / "log_samples_analysis" / "anomaly_analysis_report.md"
        if sample_report.exists():
            click.echo(f"📄 Human-readable log analysis: {sample_report}")
        sample_data = processed_dir / "log_samples_analysis" / "anomaly_samples.json"
        if sample_data.exists():
            click.echo(f"📊 Detailed sample data: {sample_data}")


@main.command("gen-synth")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--lines", "num_lines", type=int, default=5000)
@click.option("--anomaly-rate", type=float, default=0.02)
def gen_synth_cmd(out_path: Path, num_lines: int, anomaly_rate: float) -> None:
    """합성 장기 로그 생성."""
    p = generate_synthetic_log(str(out_path), num_lines=num_lines, anomaly_rate=anomaly_rate)
    click.echo(f"Generated synthetic log: {p}")


@main.command("eval")
@click.option("--processed-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--labels", "labels_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--window-size", type=int, default=50)
@click.option("--seq-len", type=int, default=50)
def eval_cmd(processed_dir: Path, labels_path: Path, window_size: int, seq_len: int) -> None:
    """베이스라인/DeepLog 평가(Precision/Recall/F1)."""
    out_lines = []
    base = processed_dir / "baseline_scores.parquet"
    if base.exists():
        p, r, f1 = evaluate_baseline(str(base), str(labels_path), window_size)
        out_lines.append(f"Baseline PRF1: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    dlinf = processed_dir / "deeplog_infer.parquet"
    if dlinf.exists():
        p, r, f1 = evaluate_deeplog(str(dlinf), str(labels_path), seq_len)
        out_lines.append(f"DeepLog PRF1: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    if not out_lines:
        out_lines = ["No artifacts to evaluate."]
    (processed_dir / "eval.txt").write_text("\n".join(out_lines))
    click.echo("\n".join(out_lines))


@main.command("analyze-samples")
@click.option("--processed-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--max-samples", type=int, default=5, help="타입별 최대 샘플 수")
@click.option("--context-lines", type=int, default=3, help="전후 맥락 라인 수")
def analyze_samples_cmd(processed_dir: Path, output_dir: Path, max_samples: int, context_lines: int) -> None:
    """이상 로그 샘플 추출 및 분석."""
    import sys
    import subprocess
    
    if output_dir is None:
        output_dir = processed_dir / "log_samples_analysis"
    
    click.echo("🔍 이상 로그 샘플 분석 시작...")
    
    cmd = [
        sys.executable, "log_sample_analyzer.py",
        str(processed_dir),
        "--output-dir", str(output_dir),
        "--max-samples", str(max_samples),
        "--context-lines", str(context_lines)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode == 0:
        click.echo("✅ 로그 샘플 분석 완료!")
        click.echo(f"📄 사람이 읽기 쉬운 리포트: {output_dir / 'anomaly_analysis_report.md'}")
        click.echo(f"📊 상세 분석 데이터: {output_dir / 'anomaly_samples.json'}")
    else:
        click.echo(f"❌ 분석 실패: {result.stderr}")
        return
    
    # 간단한 요약 출력
    sample_data_file = output_dir / "anomaly_samples.json"
    if sample_data_file.exists():
        import json
        try:
            with open(sample_data_file, 'r') as f:
                data = json.load(f)
            
            total_anomalies = 0
            for method, results in data.items():
                anomaly_count = results.get('anomaly_count', 0)
                total_anomalies += anomaly_count
                click.echo(f"  📊 {method}: {anomaly_count}개 이상 발견")
            
            click.echo(f"🚨 총 이상 개수: {total_anomalies}개")
        except Exception as e:
            click.echo(f"⚠️ 요약 생성 실패: {e}")


