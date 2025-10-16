import os
from pathlib import Path
from typing import Optional
import json
import click
import pandas as pd

from .preprocess import LogPreprocessor, PreprocessConfig
from .detect import baseline_detect, BaselineParams
from .builders.deeplog import (
    build_deeplog_inputs, train_deeplog, infer_deeplog_topk,
    infer_deeplog_enhanced, EnhancedInferenceConfig
)
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
    """DeepLog 추론 (기본 top-k 방식)."""
    df = infer_deeplog_topk(str(sequences_parquet), str(model_path), k=k)
    out = Path(sequences_parquet).with_name("deeplog_infer.parquet")
    df.to_parquet(out, index=False)
    rate = 1.0 - float(df["in_topk"].mean()) if len(df) > 0 else 0.0
    click.echo(f"Saved inference: {out} (violation_rate={rate:.3f})")


@main.command("deeplog-infer-enhanced")
@click.option("--seq", "sequences_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="sequences.parquet 파일 경로")
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="parsed.parquet 파일 경로")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="DeepLog 모델 경로")
@click.option("--vocab", "vocab_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="vocab.json 경로 (노벨티 탐지용)")
@click.option("--top-k", type=int, default=3, help="Top-K 값 (top-p 미설정 시 사용)")
@click.option("--top-p", type=float, default=None, help="Top-P 값 (설정 시 top-k보다 우선)")
@click.option("--k-of-n-k", type=int, default=7, help="K-of-N 판정: N개 중 K개 이상 실패 시 알림")
@click.option("--k-of-n-n", type=int, default=10, help="K-of-N 판정: 슬라이딩 윈도우 크기")
@click.option("--cooldown-seq", type=int, default=60, help="시퀀스 실패 쿨다운 (초)")
@click.option("--cooldown-novelty", type=int, default=60, help="노벨티 쿨다운 (초)")
@click.option("--session-timeout", type=int, default=300, help="세션 타임아웃 (초)")
@click.option("--entity-column", type=str, default="host", help="엔티티 컬럼명 (host, process 등)")
@click.option("--no-novelty", is_flag=True, default=False, help="노벨티 탐지 비활성화")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="출력 디렉토리 (기본: sequences.parquet과 같은 폴더)")
def deeplog_infer_enhanced_cmd(
    sequences_parquet: Path,
    parsed_parquet: Path,
    model_path: Path,
    vocab_path: Optional[Path],
    top_k: int,
    top_p: Optional[float],
    k_of_n_k: int,
    k_of_n_n: int,
    cooldown_seq: int,
    cooldown_novelty: int,
    session_timeout: int,
    entity_column: str,
    no_novelty: bool,
    out_dir: Optional[Path]
) -> None:
    """
    Enhanced DeepLog 추론: top-k/top-p, K-of-N 판정, 쿨다운, 노벨티 탐지, 세션화 지원.

    알림 폭주를 방지하고 엔티티별 세션 기반 이상 탐지를 수행합니다.
    """
    # 출력 디렉토리 설정
    if out_dir is None:
        out_dir = sequences_parquet.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 설정 생성
    config = EnhancedInferenceConfig(
        top_k=top_k,
        top_p=top_p,
        k_of_n_k=k_of_n_k,
        k_of_n_n=k_of_n_n,
        cooldown_seq_fail=cooldown_seq,
        cooldown_novelty=cooldown_novelty,
        session_timeout=session_timeout,
        entity_column=entity_column,
        novelty_enabled=not no_novelty,
        vocab_path=str(vocab_path) if vocab_path else None
    )

    click.echo("🚀 Enhanced DeepLog 추론 시작...")
    click.echo(f"  📊 Top-{'P' if top_p else 'K'}: {top_p if top_p else top_k}")
    click.echo(f"  🎯 K-of-N: {k_of_n_k}/{k_of_n_n}")
    click.echo(f"  ⏰ Cooldown: SEQ={cooldown_seq}s, NOVELTY={cooldown_novelty}s")
    click.echo(f"  🔍 노벨티 탐지: {'ON' if not no_novelty else 'OFF'}")
    click.echo(f"  👤 엔티티: {entity_column}")

    # Enhanced inference 실행
    detailed_df, alerts_df, summary = infer_deeplog_enhanced(
        str(sequences_parquet),
        str(parsed_parquet),
        str(model_path),
        config
    )

    # 결과 저장
    detailed_out = out_dir / "deeplog_enhanced_detailed.parquet"
    alerts_out = out_dir / "deeplog_enhanced_alerts.parquet"
    summary_out = out_dir / "deeplog_enhanced_summary.json"

    detailed_df.to_parquet(detailed_out, index=False)
    alerts_df.to_parquet(alerts_out, index=False)

    import json
    with open(summary_out, 'w') as f:
        # datetime을 문자열로 변환
        summary_serializable = {}
        for key, value in summary.items():
            if key == "novelty_aggregation":
                serializable_agg = {}
                for agg_key, agg_val in value.items():
                    serializable_agg[agg_key] = {
                        "count": agg_val["count"],
                        "first": agg_val["first"].isoformat() if hasattr(agg_val["first"], "isoformat") else str(agg_val["first"]),
                        "last": agg_val["last"].isoformat() if hasattr(agg_val["last"], "isoformat") else str(agg_val["last"])
                    }
                summary_serializable[key] = serializable_agg
            else:
                summary_serializable[key] = value
        json.dump(summary_serializable, f, indent=2)

    # 결과 출력
    click.echo("\n✅ Enhanced DeepLog 추론 완료!")
    click.echo(f"\n📊 요약:")
    click.echo(f"  전체 시퀀스: {summary['total_sequences']:,}개")
    click.echo(f"  실패 시퀀스: {summary['total_failures']:,}개")
    click.echo(f"  노벨티 발견: {summary['total_novels']:,}개")
    click.echo(f"  발생 알림: {summary['total_alerts']:,}개")

    if summary.get('alerts_by_type'):
        click.echo(f"\n🚨 알림 유형별:")
        for alert_type, count in summary['alerts_by_type'].items():
            click.echo(f"  - {alert_type}: {count}개")

    click.echo(f"\n📁 출력 파일:")
    click.echo(f"  상세 결과: {detailed_out}")
    click.echo(f"  알림 목록: {alerts_out}")
    click.echo(f"  요약 정보: {summary_out}")

    # 알림이 있으면 샘플 표시
    if not alerts_df.empty:
        click.echo(f"\n🔔 최근 알림 샘플 (최대 5개):")
        for _, alert in alerts_df.head(5).iterrows():
            timestamp = alert['timestamp']
            entity = alert['entity']
            alert_type = alert['alert_type']
            template_id = alert.get('template_id', 'N/A')
            click.echo(f"  [{timestamp}] {entity} - {alert_type} (template: {template_id})")


@main.command("build-mscred")
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--window-size", type=int, default=50)
@click.option("--stride", type=int, default=25)
def build_mscred_cmd(parsed_parquet: Path, out_dir: Path, window_size: int, stride: int) -> None:
    """MS-CRED 입력(윈도우 카운트) 생성."""
    build_mscred_window_counts(str(parsed_parquet), str(out_dir), window_size=window_size, stride=stride)
    click.echo(f"Built MS-CRED window counts under: {out_dir}")


@main.command("mscred-train")
@click.option("--window-counts", "window_counts_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "model_output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--epochs", type=int, default=50)
def mscred_train_cmd(window_counts_parquet: Path, model_output: Path, epochs: int) -> None:
    """MS-CRED 모델 학습."""
    from .mscred_model import train_mscred
    
    model_output.parent.mkdir(parents=True, exist_ok=True)
    stats = train_mscred(str(window_counts_parquet), str(model_output), epochs)
    
    click.echo(f"MS-CRED 학습 완료: {model_output}")
    click.echo(f"최종 학습 손실: {stats['final_train_loss']:.4f}")
    click.echo(f"최종 검증 손실: {stats['final_val_loss']:.4f}")


@main.command("mscred-infer")
@click.option("--window-counts", "window_counts_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--threshold", type=float, default=95.0, help="이상 탐지 임계값 (백분위수)")
def mscred_infer_cmd(window_counts_parquet: Path, model_path: Path, threshold: float) -> None:
    """MS-CRED 이상 탐지 추론."""
    from .mscred_model import infer_mscred
    
    out = Path(window_counts_parquet).with_name("mscred_infer.parquet")
    results_df = infer_mscred(str(window_counts_parquet), str(model_path), str(out), threshold)
    
    anomaly_rate = results_df['is_anomaly'].mean()
    click.echo(f"Saved MS-CRED inference: {out}")
    click.echo(f"Anomaly rate: {anomaly_rate:.3f} ({results_df['is_anomaly'].sum()}/{len(results_df)})")


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
    
    # MS-CRED
    mscred_path = processed_dir / "mscred_infer.parquet"
    if mscred_path.exists():
        m = pd.read_parquet(mscred_path)
        if len(m) > 0:
            anomaly_rate = float(m["is_anomaly"].mean())
            top_errors = m.nlargest(5, 'reconstruction_error')
            report_lines.append(f"MS-CRED anomaly rate: {anomaly_rate:.3f}")
            report_lines.append("Top reconstruction errors (window_idx, error): " + 
                              ", ".join([f"{int(r.window_idx)}:{float(r.reconstruction_error):.4f}" for _, r in top_errors.iterrows()]))
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
            from .analyzers.log_samples import main as log_samples_main
            import sys

            # Save current sys.argv
            old_argv = sys.argv
            sys.argv = [
                "analyze-samples",
                str(processed_dir),
                "--output-dir", str(processed_dir / "log_samples_analysis")
            ]

            try:
                log_samples_main()
                report_lines.append("Log sample analysis completed successfully")
                report_lines.append(f"Detailed analysis: {processed_dir / 'log_samples_analysis' / 'anomaly_analysis_report.md'}")
            finally:
                sys.argv = old_argv
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
    from .analyzers.log_samples import main as log_samples_main
    import sys

    if output_dir is None:
        output_dir = processed_dir / "log_samples_analysis"

    click.echo("🔍 이상 로그 샘플 분석 시작...")

    # Save current sys.argv
    old_argv = sys.argv
    sys.argv = [
        "analyze-samples",
        str(processed_dir),
        "--output-dir", str(output_dir),
        "--max-samples", str(max_samples),
        "--context-lines", str(context_lines)
    ]

    try:
        result_code = log_samples_main()
    finally:
        sys.argv = old_argv

    if result_code is None:
        result_code = 0
    result = type('obj', (object,), {'returncode': result_code, 'stdout': '', 'stderr': ''})
    
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


@main.command("convert-onnx")
@click.option("--deeplog-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="DeepLog 모델 경로")
@click.option("--mscred-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="MS-CRED 모델 경로")
@click.option("--vocab", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="어휘 사전 경로")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default="models/onnx", help="ONNX 출력 디렉토리")
@click.option("--validate", is_flag=True, default=False, help="변환 후 검증 실행")
@click.option("--feature-dim", type=int, default=None, help="MS-CRED 피처 차원 (템플릿 개수, 기본: 자동 감지)")
@click.option("--portable", is_flag=True, default=False, help="범용 최적화 모드 (모든 환경에서 사용 가능, 하드웨어 특화 최적화 제외)")
def convert_onnx_cmd(deeplog_model: Path, mscred_model: Path, vocab: Path, output_dir: Path, validate: bool, feature_dim: Optional[int], portable: bool) -> None:
    """PyTorch 모델을 ONNX 포맷으로 변환."""
    try:
        # 하이브리드 시스템 모듈 동적 임포트
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from hybrid_system.training.model_converter import convert_all_models

        if not deeplog_model and not mscred_model:
            click.echo("❌ 최소 하나의 모델 경로를 지정해야 합니다.")
            return

        click.echo("🔄 ONNX 변환 시작...")

        if portable:
            click.echo("🌍 범용 최적화 모드: 모든 환경에서 사용 가능한 모델 생성")
        else:
            click.echo("⚡ 최대 최적화 모드: 현재 하드웨어에 특화된 모델 생성")

        # vocab에서 템플릿 개수 추출 (MS-CRED용)
        if mscred_model and feature_dim is None and vocab:
            try:
                with open(vocab, 'r') as f:
                    import json
                    vocab_dict = json.load(f)
                    feature_dim = len(vocab_dict)
                    click.echo(f"📊 vocab에서 템플릿 개수 감지: {feature_dim}")
            except Exception as e:
                click.echo(f"⚠️ vocab에서 템플릿 개수 추출 실패: {e}")
                feature_dim = 100

        results = convert_all_models(
            str(deeplog_model) if deeplog_model else "",
            str(mscred_model) if mscred_model else "",
            str(vocab) if vocab else "",
            str(output_dir),
            feature_dim=feature_dim,
            portable=portable
        )
        
        click.echo("\n🎉 ONNX 변환 완료!")
        for model_name, result in results.items():
            if 'error' in result:
                click.echo(f"❌ {model_name}: {result['error']}")
            else:
                click.echo(f"✅ {model_name}: {result['onnx_path']}")
                if 'optimized_path' in result:
                    click.echo(f"⚡ 최적화됨: {result['optimized_path']}")
        
        click.echo(f"📁 변환 결과: {output_dir}")
        
    except ImportError as e:
        click.echo(f"❌ 하이브리드 시스템 모듈을 찾을 수 없습니다: {e}")
        click.echo("💡 다음 명령으로 의존성을 설치하세요:")
        click.echo("   pip install -r requirements_hybrid.txt")
    except Exception as e:
        click.echo(f"❌ ONNX 변환 실패: {e}")


@main.command("hybrid-pipeline")
@click.option("--log-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="입력 로그 파일")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), help="출력 디렉토리")
@click.option("--auto-deploy", is_flag=True, default=True, help="자동 배포 준비")
@click.option("--models-dir", type=click.Path(file_okay=False, path_type=Path), default="models", help="모델 저장 디렉토리")
def hybrid_pipeline_cmd(log_file: Path, output_dir: Path, auto_deploy: bool, models_dir: Path) -> None:
    """하이브리드 시스템 전체 파이프라인 실행 (학습 → ONNX 변환 → 배포 준비)."""
    try:
        # 하이브리드 시스템 모듈 동적 임포트
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from hybrid_system.training.auto_converter import AutoConverter
        
        click.echo("🚀 하이브리드 파이프라인 시작...")
        
        converter = AutoConverter(
            models_dir=str(models_dir),
            onnx_dir=str(models_dir / "onnx"),
            deployment_dir=str(models_dir / "deployment")
        )
        
        results = converter.run_full_pipeline(str(log_file), auto_deploy)
        
        click.echo(f"\n🎉 하이브리드 파이프라인 완료: {results['status']}")
        
        if 'training' in results['stages']:
            training = results['stages']['training']
            click.echo(f"📊 학습된 모델:")
            for model_name, model_info in training.get('models', {}).items():
                click.echo(f"  - {model_name}: {model_info['path']}")
        
        if 'conversion' in results['stages']:
            conversion = results['stages']['conversion']
            click.echo(f"🔄 변환된 모델:")
            for model_name, result in conversion.items():
                if 'error' not in result:
                    click.echo(f"  - {model_name}: {result['onnx_path']}")
        
        if 'deployment' in results['stages']:
            deployment = results['stages']['deployment']
            click.echo(f"📦 배포 준비 완료:")
            click.echo(f"  - 모델 개수: {len(deployment['models'])}개")
            click.echo(f"  - 파일 개수: {len(deployment['files'])}개")
        
        click.echo(f"📁 결과 위치: {models_dir / 'deployment'}")
        
    except ImportError as e:
        click.echo(f"❌ 하이브리드 시스템 모듈을 찾을 수 없습니다: {e}")
        click.echo("💡 다음 명령으로 의존성을 설치하세요:")
        click.echo("   pip install -r requirements_hybrid.txt")
    except Exception as e:
        click.echo(f"❌ 하이브리드 파이프라인 실패: {e}")


@main.command("analyze-temporal")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="분석할 데이터 디렉토리")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="결과 저장 디렉토리")
def analyze_temporal_cmd(data_dir: Path, output_dir: Path) -> None:
    """시간 기반 이상 탐지 분석."""
    from .analyzers.temporal import main as temporal_main
    import sys

    # 임시로 sys.argv 조작
    old_argv = sys.argv
    sys.argv = ['temporal', '--data-dir', str(data_dir)]
    if output_dir:
        sys.argv.extend(['--output-dir', str(output_dir)])

    try:
        temporal_main()
    finally:
        sys.argv = old_argv


@main.command("analyze-comparative")
@click.option("--target", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Target 파일")
@click.option("--baselines", multiple=True, required=True, help="Baseline 파일들")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="결과 저장 디렉토리")
def analyze_comparative_cmd(target: Path, baselines: tuple, output_dir: Path) -> None:
    """비교 기반 이상 탐지 분석."""
    from .analyzers.comparative import main as comparative_main
    import sys

    # 임시로 sys.argv 조작
    old_argv = sys.argv
    sys.argv = ['comparative', '--target', str(target)]
    for baseline in baselines:
        sys.argv.extend(['--baselines', baseline])
    if output_dir:
        sys.argv.extend(['--output-dir', str(output_dir)])

    try:
        comparative_main()
    finally:
        sys.argv = old_argv


@main.command("analyze-mscred")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="MS-CRED 결과 디렉토리")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="분석 결과 저장 디렉토리")
def analyze_mscred_cmd(data_dir: Path, output_dir: Path) -> None:
    """MS-CRED 전용 분석."""
    from .analyzers.mscred_analysis import main as mscred_main
    import sys

    # 임시로 sys.argv 조작
    old_argv = sys.argv
    sys.argv = ['mscred', '--data-dir', str(data_dir)]
    if output_dir:
        sys.argv.extend(['--output-dir', str(output_dir)])

    try:
        mscred_main()
    finally:
        sys.argv = old_argv


@main.command("validate-baseline")
@click.argument("baseline_files", nargs=-1, required=True)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="결과 저장 디렉토리")
def validate_baseline_cmd(baseline_files: tuple, output_dir: Path) -> None:
    """베이스라인 파일 품질 검증."""
    from .analyzers.baseline_validation import main as baseline_main
    import sys

    # 임시로 sys.argv 조작
    old_argv = sys.argv
    sys.argv = ['baseline'] + list(baseline_files)
    if output_dir:
        sys.argv.extend(['--output-dir', str(output_dir)])

    try:
        baseline_main()
    finally:
        sys.argv = old_argv


