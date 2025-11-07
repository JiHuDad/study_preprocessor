"""통합 CLI 모듈

- 목적: 파이프라인 전 과정을 CLI 명령으로 제공 (파싱 → 베이스라인/DeepLog/MS-CRED → 분석/리포트 → 평가/변환)
- 주요 명령:
  * parse: 원시 로그 파싱/마스킹 → parsed.parquet
  * detect: 베이스라인 이상탐지 → baseline_scores.parquet
  * build-deeplog / deeplog-train / deeplog-infer(-enhanced): DeepLog 입력/학습/추론
  * build-mscred / mscred-train / mscred-infer: MS-CRED 입력/학습/추론
  * report: 처리 결과 요약 리포트 생성
  * analyze-*: 다양한 분석 도구 래핑 (temporal/comparative/mscred/log samples)
  * eval: 베이스라인/DeepLog에 대한 PRF1 평가
  * convert-onnx / hybrid-pipeline: 모델 변환 및 하이브리드 파이프라인 실행
"""

import os  # 환경/경로 유틸
from pathlib import Path  # 경로 타입
from typing import Optional  # 선택적 타입
import json  # JSON 입출력
import click  # CLI 프레임워크
import pandas as pd  # 데이터 처리

from .preprocess import LogPreprocessor, PreprocessConfig  # 전처리 유틸
from .detect import baseline_detect, BaselineParams  # 베이스라인 탐지
from .builders.deeplog import (  # DeepLog 파이프라인 함수들
    build_deeplog_inputs, train_deeplog, infer_deeplog_topk,
    infer_deeplog_enhanced, EnhancedInferenceConfig
)
from .builders.mscred import build_mscred_window_counts  # MS-CRED 입력 생성
from .synth import (  # 합성 로그 생성기들
    generate_synthetic_log,
    generate_training_data,
    generate_inference_normal,
    generate_inference_anomaly,
)
from .eval import evaluate_baseline, evaluate_deeplog  # 평가 유틸


@click.group()  # 루트 커맨드 그룹
def main() -> None:  # 엔트리 포인트
    """Anomaly Log Detector: Comprehensive log anomaly detection framework with DeepLog, MS-CRED, and baseline methods"""  # CLI 설명


@main.command()  # parse 명령 등록
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # 입력 로그 파일
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)  # 출력 디렉토리
@click.option("--drain-state", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Drain3 상태 파일 경로")  # Drain3 상태 파일
# Masking toggles  # 마스킹 옵션 스위치(비활성화 토글)
@click.option("--no-mask-paths", is_flag=True, default=False)  # 경로 마스킹 해제
@click.option("--no-mask-hex", is_flag=True, default=False)  # 16진수 마스킹 해제
@click.option("--no-mask-ips", is_flag=True, default=False)  # IP 마스킹 해제
@click.option("--no-mask-mac", is_flag=True, default=False)  # MAC 마스킹 해제
@click.option("--no-mask-uuid", is_flag=True, default=False)  # UUID 마스킹 해제
@click.option("--no-mask-pid", is_flag=True, default=False)  # PID 마스킹 해제
@click.option("--no-mask-device", is_flag=True, default=False)  # 디바이스번호 마스킹 해제
@click.option("--no-mask-num", is_flag=True, default=False)  # 일반 숫자 마스킹 해제
def parse(input_path: Path, out_dir: Path, drain_state: Path | None,
          no_mask_paths: bool, no_mask_hex: bool, no_mask_ips: bool, no_mask_mac: bool,
          no_mask_uuid: bool, no_mask_pid: bool, no_mask_device: bool, no_mask_num: bool) -> None:  # 로그 파싱 엔트리
    """원시 로그 파일을 파싱/마스킹하고 Parquet으로 저장."""  # 명령 설명
    out_dir.mkdir(parents=True, exist_ok=True)  # 출력 폴더 생성
    cfg = PreprocessConfig(  # 전처리 설정 구성
        drain_state_path=str(drain_state) if drain_state else None,  # Drain 상태 경로
        mask_paths=not no_mask_paths,  # 경로 마스킹 여부
        mask_hex=not no_mask_hex,  # 16진수 마스킹 여부
        mask_ips=not no_mask_ips,  # IP 마스킹 여부
        mask_mac=not no_mask_mac,  # MAC 마스킹 여부
        mask_uuid=not no_mask_uuid,  # UUID 마스킹 여부
        mask_pid_fields=not no_mask_pid,  # PID 필드 마스킹 여부
        mask_device_numbers=not no_mask_device,  # 디바이스 번호 마스킹 여부
        mask_numbers=not no_mask_num,  # 일반 숫자 마스킹 여부
    )
    pre = LogPreprocessor(cfg)  # 전처리기 생성
    df = pre.process_file(str(input_path))  # 파일 파싱/마스킹 수행
    parquet_path = out_dir / "parsed.parquet"  # 출력 파일 경로
    df.to_parquet(parquet_path, index=False)  # Parquet 저장
    # 미리보기용 일부 샘플도 JSON 저장
    preview = df.head(10).to_dict(orient="records")  # 상위 샘플 추출
    (out_dir / "preview.json").write_text(json.dumps(preview, ensure_ascii=False, default=str, indent=2))  # JSON 저장
    click.echo(f"Saved: {parquet_path}")  # 결과 경로 출력
    click.echo(f"Preview: {out_dir / 'preview.json'}")  # 미리보기 경로 출력


@main.command()  # detect 명령 등록
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # parsed.parquet 경로
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)  # 출력 디렉토리
@click.option("--window-size", type=int, default=50)  # 윈도우 크기
@click.option("--stride", type=int, default=25)  # 스트라이드
@click.option("--ewm-alpha", type=float, default=0.3)  # EWM 알파
@click.option("--q", "anomaly_q", type=float, default=0.95, help="Anomaly quantile threshold")  # 이상 임계 백분위
def detect(parsed_parquet: Path, out_dir: Path, window_size: int, stride: int, ewm_alpha: float, anomaly_q: float) -> None:  # 베이스라인 실행
    """베이스라인 이상탐지(새 템플릿 비율 + 빈도 급변) 실행."""  # 설명
    params = BaselineParams(window_size=window_size, stride=stride, ewm_alpha=ewm_alpha, anomaly_quantile=anomaly_q)  # 파라미터 구성
    out_path = baseline_detect(str(parsed_parquet), str(out_dir), params)  # 탐지 실행
    click.echo(f"Saved baseline scores: {out_path}")  # 결과 출력


@main.command("build-deeplog")  # DeepLog 입력 생성 명령
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # parsed.parquet 경로
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)  # 출력 디렉토리
def build_deeplog_cmd(parsed_parquet: Path, out_dir: Path) -> None:  # DeepLog 입력 생성
    """DeepLog 입력(vocab, sequences) 생성."""  # 설명
    build_deeplog_inputs(str(parsed_parquet), str(out_dir))  # 생성 실행
    click.echo(f"Built DeepLog inputs under: {out_dir}")  # 완료 메시지


@main.command("deeplog-train")  # DeepLog 학습 명령
@click.option("--seq", "sequences_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # sequences.parquet
@click.option("--vocab", "vocab_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # vocab.json
@click.option("--out", "model_out", type=click.Path(dir_okay=False, path_type=Path), required=True)  # 모델 저장 경로
@click.option("--seq-len", type=int, default=50)  # 시퀀스 길이
@click.option("--epochs", type=int, default=3)  # 에폭 수
def deeplog_train_cmd(sequences_parquet: Path, vocab_json: Path, model_out: Path, seq_len: int, epochs: int) -> None:  # 학습 실행
    path = train_deeplog(str(sequences_parquet), str(vocab_json), str(model_out), seq_len=seq_len, epochs=epochs)  # 학습
    click.echo(f"Saved DeepLog model: {path}")  # 저장 경로 출력


@main.command("deeplog-infer")  # DeepLog 추론(top-k)
@click.option("--seq", "sequences_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # sequences.parquet
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # 모델 경로
@click.option("--vocab", "vocab_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="vocab.json 경로 (예측/실제 템플릿 표시용)")  # vocab 경로
@click.option("--k", type=int, default=3)  # Top-K
def deeplog_infer_cmd(sequences_parquet: Path, model_path: Path, vocab_path: Path | None, k: int) -> None:  # 추론 실행
    """DeepLog 추론 (기본 top-k 방식)."""  # 설명
    df = infer_deeplog_topk(str(sequences_parquet), str(model_path), vocab_path=str(vocab_path) if vocab_path else None, k=k)  # 추론 수행
    out = Path(sequences_parquet).with_name("deeplog_infer.parquet")  # 출력 경로
    df.to_parquet(out, index=False)  # 저장
    rate = 1.0 - float(df["in_topk"].mean()) if len(df) > 0 else 0.0  # 위반율 계산
    click.echo(f"Saved inference: {out} (violation_rate={rate:.3f})")  # 결과 출력


@main.command("deeplog-infer-enhanced")  # 향상된 DeepLog 추론
@click.option("--seq", "sequences_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="sequences.parquet 파일 경로")  # 시퀀스 파일
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="parsed.parquet 파일 경로")  # 파싱 파일
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="DeepLog 모델 경로")  # 모델 경로
@click.option("--vocab", "vocab_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="vocab.json 경로 (노벨티 탐지용)")  # vocab 경로
@click.option("--top-k", type=int, default=3, help="Top-K 값 (top-p 미설정 시 사용)")  # Top-K
@click.option("--top-p", type=float, default=None, help="Top-P 값 (설정 시 top-k보다 우선)")  # Top-P
@click.option("--k-of-n-k", type=int, default=7, help="K-of-N 판정: N개 중 K개 이상 실패 시 알림")  # K-of-N의 K
@click.option("--k-of-n-n", type=int, default=10, help="K-of-N 판정: 슬라이딩 윈도우 크기")  # K-of-N의 N
@click.option("--cooldown-seq", type=int, default=60, help="시퀀스 실패 쿨다운 (초)")  # SEQ 쿨다운
@click.option("--cooldown-novelty", type=int, default=60, help="노벨티 쿨다운 (초)")  # 노벨티 쿨다운
@click.option("--session-timeout", type=int, default=300, help="세션 타임아웃 (초)")  # 세션 타임아웃
@click.option("--entity-column", type=str, default="host", help="엔티티 컬럼명 (host, process 등)")  # 엔티티 컬럼
@click.option("--no-novelty", is_flag=True, default=False, help="노벨티 탐지 비활성화")  # 노벨티 비활성화
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="출력 디렉토리 (기본: sequences.parquet과 같은 폴더)")  # 출력 폴더
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
) -> None:  # 향상된 추론 실행
    """
    Enhanced DeepLog 추론: top-k/top-p, K-of-N 판정, 쿨다운, 노벨티 탐지, 세션화 지원.

    알림 폭주를 방지하고 엔티티별 세션 기반 이상 탐지를 수행합니다.
    """
    # 출력 디렉토리 설정  # 기본: 시퀀스 파일 폴더
    if out_dir is None:
        out_dir = sequences_parquet.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 설정 생성  # 알림/윈도우/쿨다운/노벨티 옵션 포함
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

    click.echo("🚀 Enhanced DeepLog 추론 시작...")  # 시작 로그
    click.echo(f"  📊 Top-{'P' if top_p else 'K'}: {top_p if top_p else top_k}")  # Top 선택
    click.echo(f"  🎯 K-of-N: {k_of_n_k}/{k_of_n_n}")  # K-of-N 정보
    click.echo(f"  ⏰ Cooldown: SEQ={cooldown_seq}s, NOVELTY={cooldown_novelty}s")  # 쿨다운
    click.echo(f"  🔍 노벨티 탐지: {'ON' if not no_novelty else 'OFF'}")  # 노벨티 On/Off
    click.echo(f"  👤 엔티티: {entity_column}")  # 엔티티 컬럼

    # Enhanced inference 실행  # 세부/알림/요약 반환
    detailed_df, alerts_df, summary = infer_deeplog_enhanced(
        str(sequences_parquet),
        str(parsed_parquet),
        str(model_path),
        config
    )

    # 결과 저장  # 파일 출력 경로
    detailed_out = out_dir / "deeplog_enhanced_detailed.parquet"
    alerts_out = out_dir / "deeplog_enhanced_alerts.parquet"
    summary_out = out_dir / "deeplog_enhanced_summary.json"

    detailed_df.to_parquet(detailed_out, index=False)  # 상세 저장
    alerts_df.to_parquet(alerts_out, index=False)  # 알림 저장

    import json  # 요약 JSON 저장
    with open(summary_out, 'w') as f:  # 파일 열기
        # datetime을 문자열로 변환  # 직렬화 처리
        summary_serializable = {}
        for key, value in summary.items():  # 키별 처리
            if key == "novelty_aggregation":  # 노벨티 집계 구조
                serializable_agg = {}
                for agg_key, agg_val in value.items():  # 항목별 변환
                    serializable_agg[agg_key] = {
                        "count": agg_val["count"],
                        "first": agg_val["first"].isoformat() if hasattr(agg_val["first"], "isoformat") else str(agg_val["first"]),
                        "last": agg_val["last"].isoformat() if hasattr(agg_val["last"], "isoformat") else str(agg_val["last"])
                    }
                summary_serializable[key] = serializable_agg
            else:
                summary_serializable[key] = value
        json.dump(summary_serializable, f, indent=2)  # JSON 덤프

    # 결과 출력  # 콘솔 요약
    click.echo("\n✅ Enhanced DeepLog 추론 완료!")
    click.echo(f"\n📊 요약:")
    click.echo(f"  전체 시퀀스: {summary['total_sequences']:,}개")
    click.echo(f"  실패 시퀀스: {summary['total_failures']:,}개")
    click.echo(f"  노벨티 발견: {summary['total_novels']:,}개")
    click.echo(f"  발생 알림: {summary['total_alerts']:,}개")

    if summary.get('alerts_by_type'):  # 유형별 알림 개수
        click.echo(f"\n🚨 알림 유형별:")
        for alert_type, count in summary['alerts_by_type'].items():
            click.echo(f"  - {alert_type}: {count}개")

    click.echo(f"\n📁 출력 파일:")  # 산출물 경로 안내
    click.echo(f"  상세 결과: {detailed_out}")
    click.echo(f"  알림 목록: {alerts_out}")
    click.echo(f"  요약 정보: {summary_out}")

    # 알림이 있으면 샘플 표시  # 상위 5개 출력
    if not alerts_df.empty:
        click.echo(f"\n🔔 최근 알림 샘플 (최대 5개):")
        for _, alert in alerts_df.head(5).iterrows():
            timestamp = alert['timestamp']
            entity = alert['entity']
            alert_type = alert['alert_type']
            template_id = alert.get('template_id', 'N/A')
            click.echo(f"  [{timestamp}] {entity} - {alert_type} (template: {template_id})")


@main.command("build-mscred")  # MS-CRED 입력 생성
@click.option("--parsed", "parsed_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # parsed.parquet
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)  # 출력 폴더
@click.option("--window-size", type=int, default=50)  # 윈도우 크기
@click.option("--stride", type=int, default=25)  # 스트라이드
@click.option("--template-mapping-path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="기존 template_mapping.json 경로 (추론 시 사용)")  # 템플릿 매핑 경로
def build_mscred_cmd(parsed_parquet: Path, out_dir: Path, window_size: int, stride: int, template_mapping_path: Path | None) -> None:  # 생성 실행
    """MS-CRED 입력(윈도우 카운트) 생성."""  # 설명
    build_mscred_window_counts(
        str(parsed_parquet),
        str(out_dir),
        window_size=window_size,
        stride=stride,
        template_mapping_path=str(template_mapping_path) if template_mapping_path else None
    )  # 생성
    click.echo(f"Built MS-CRED window counts under: {out_dir}")  # 완료 메시지


@main.command("mscred-train")  # MS-CRED 학습
@click.option("--window-counts", "window_counts_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # window_counts.parquet
@click.option("--out", "model_output", type=click.Path(dir_okay=False, path_type=Path), required=True)  # 모델 출력 경로
@click.option("--epochs", type=int, default=50)  # 에폭 수
def mscred_train_cmd(window_counts_parquet: Path, model_output: Path, epochs: int) -> None:  # 학습 실행
    """MS-CRED 모델 학습."""  # 설명
    from .mscred_model import train_mscred  # 지연 임포트
    
    model_output.parent.mkdir(parents=True, exist_ok=True)  # 폴더 생성
    stats = train_mscred(str(window_counts_parquet), str(model_output), epochs)  # 학습
    
    click.echo(f"MS-CRED 학습 완료: {model_output}")  # 완료
    click.echo(f"최종 학습 손실: {stats['final_train_loss']:.4f}")  # 학습 손실
    click.echo(f"최종 검증 손실: {stats['final_val_loss']:.4f}")  # 검증 손실


@main.command("mscred-infer")  # MS-CRED 추론
@click.option("--window-counts", "window_counts_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # window_counts.parquet
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # 모델 경로
@click.option("--threshold", type=float, default=None, help="수동 임계값 지정 (지정하지 않으면 학습 시 계산된 임계값 사용)")  # 수동 임계값
@click.option("--threshold-method", type=click.Choice(['99percentile', '95percentile', '99.9percentile', '3sigma', 'mad']),
              default='99percentile', help="학습 시 저장된 임계값 선택 방법 (기본값: 99percentile)")  # 임계값 선택 방법
def mscred_infer_cmd(window_counts_parquet: Path, model_path: Path, threshold: float | None, threshold_method: str) -> None:  # 추론 실행
    """MS-CRED 이상 탐지 추론."""  # 설명
    from .mscred_model import infer_mscred  # 지연 임포트

    out = Path(window_counts_parquet).with_name("mscred_infer.parquet")  # 출력 경로
    results_df = infer_mscred(str(window_counts_parquet), str(model_path), str(out), threshold, threshold_method)  # 추론

    anomaly_rate = results_df['is_anomaly'].mean()  # 이상률
    click.echo(f"Saved MS-CRED inference: {out}")  # 경로
    click.echo(f"Anomaly rate: {anomaly_rate:.3f} ({results_df['is_anomaly'].sum()}/{len(results_df)})")  # 요약


def _generate_enhanced_report(processed_dir: Path, with_samples: bool = True) -> str:
    """개선된 읽기 쉬운 리포트를 생성합니다."""
    print(f"[DEBUG] _generate_enhanced_report 시작", flush=True)
    print(f"[DEBUG]   processed_dir={processed_dir}", flush=True)
    print(f"[DEBUG]   with_samples={with_samples}", flush=True)

    import pandas as pd
    from datetime import datetime
    print(f"[DEBUG] pandas, datetime import 완료", flush=True)

    # 리포트 헤더
    report = f"""# 📊 로그 이상 탐지 분석 리포트

**생성 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**분석 대상**: `{processed_dir}`

---

"""
    print(f"[DEBUG] 리포트 헤더 생성 완료", flush=True)

    # 데이터 로드
    parsed_path = processed_dir / "parsed.parquet"
    # baseline_scores_enhanced.parquet 우선, 없으면 baseline_scores.parquet
    base_path_enhanced = processed_dir / "baseline_scores_enhanced.parquet"
    base_path = base_path_enhanced if base_path_enhanced.exists() else processed_dir / "baseline_scores.parquet"
    deeplog_path = processed_dir / "deeplog_infer.parquet"
    mscred_path = processed_dir / "mscred_infer.parquet"
    vocab_path = processed_dir / "vocab.json"

    print(f"[DEBUG] 파일 경로 설정:", flush=True)
    print(f"[DEBUG]   parsed_path={parsed_path} (exists={parsed_path.exists()})", flush=True)
    print(f"[DEBUG]   base_path={base_path} (exists={base_path.exists()})", flush=True)
    print(f"[DEBUG]   deeplog_path={deeplog_path} (exists={deeplog_path.exists()})", flush=True)
    print(f"[DEBUG]   mscred_path={mscred_path} (exists={mscred_path.exists()})", flush=True)
    print(f"[DEBUG]   vocab_path={vocab_path} (exists={vocab_path.exists()})", flush=True)

    has_data = False

    # 종합 요약 섹션
    report += "## 📋 종합 요약\n\n"
    report += "| 탐지 방법 | 이상률 | 심각도 | 상태 |\n"
    report += "|---------|--------|--------|------|\n"

    baseline_rate = None
    deeplog_viol = None
    mscred_rate = None

    # Baseline 데이터
    if base_path.exists():
        s = pd.read_parquet(base_path)
        if len(s) > 0:
            has_data = True
            baseline_rate = float((s["is_anomaly"] == True).mean())
            severity = "🟢 낮음" if baseline_rate < 0.05 else ("🟡 중간" if baseline_rate < 0.20 else "🔴 높음")
            status = "정상 범위" if baseline_rate < 0.05 else ("일부 이상 발견" if baseline_rate < 0.20 else "다수 이상 발견")
            report += f"| Baseline (통계) | {baseline_rate:.1%} | {severity} | {status} |\n"

    # DeepLog 데이터
    if deeplog_path.exists():
        d = pd.read_parquet(deeplog_path)
        if len(d) > 0:
            has_data = True
            # Enhanced 버전: prediction_ok 사용, 기존 버전: in_topk 사용
            if "prediction_ok" in d.columns:
                deeplog_viol = 1.0 - float(d["prediction_ok"].mean())
            elif "in_topk" in d.columns:
                deeplog_viol = 1.0 - float(d["in_topk"].mean())
            else:
                deeplog_viol = 0.0  # 컬럼이 없으면 기본값
            severity = "🟢 낮음" if deeplog_viol < 0.20 else ("🟡 중간" if deeplog_viol < 0.50 else "🔴 높음")
            status = "예측 가능" if deeplog_viol < 0.20 else ("패턴 복잡" if deeplog_viol < 0.50 else "예측 어려움")
            report += f"| DeepLog (딥러닝) | {deeplog_viol:.1%} | {severity} | {status} |\n"

    # MS-CRED 데이터
    if mscred_path.exists():
        m = pd.read_parquet(mscred_path)
        if len(m) > 0:
            has_data = True
            mscred_rate = float(m["is_anomaly"].mean())
            severity = "🟢 낮음" if mscred_rate < 0.05 else ("🟡 중간" if mscred_rate < 0.20 else "🔴 높음")
            status = "정상 범위" if mscred_rate < 0.05 else ("일부 이상" if mscred_rate < 0.20 else "다수 이상")
            report += f"| MS-CRED (멀티스케일) | {mscred_rate:.1%} | {severity} | {status} |\n"

    if not has_data:
        return "# 📊 로그 이상 탐지 분석 리포트\n\n**결과 없음**: 분석할 데이터가 없습니다.\n"

    # 주요 발견사항
    report += "\n### ⚠️ 주요 발견사항\n\n"
    findings = []

    if deeplog_viol and deeplog_viol > 0.50:
        findings.append(f"- **DeepLog 위반율이 {deeplog_viol:.1%}로 매우 높음** → 로그 패턴이 불규칙하거나 학습 데이터 부족")
    if baseline_rate and baseline_rate > 0.20:
        findings.append(f"- **Baseline에서 {baseline_rate:.1%}의 윈도우에서 이상 탐지** → 다수 구간에 새로운 패턴 발견")
    elif baseline_rate and baseline_rate > 0.05:
        findings.append(f"- **Baseline에서 {baseline_rate:.1%}의 윈도우에서 이상 탐지** → 일부 구간에 새로운 패턴 발견")
    if mscred_rate and mscred_rate > 0.20:
        findings.append(f"- **MS-CRED에서 {mscred_rate:.1%}의 이상 탐지** → 로그 패턴 구조가 비정상적")

    if not findings:
        findings.append("- ✅ 모든 탐지 방법에서 정상 범위 내의 결과")

    report += "\n".join(findings) + "\n\n---\n\n"

    # Baseline 상세 분석
    if base_path.exists():
        s = pd.read_parquet(base_path)
        if len(s) > 0:
            report += "## 🔍 Baseline 이상 탐지 (통계 기반)\n\n"
            anomalous = s[s["is_anomaly"] == True]
            report += f"**이상 윈도우**: 전체의 {baseline_rate:.1%} ({len(s)}개 중 {len(anomalous)}개)\n\n"

            if len(anomalous) > 0:
                report += "### 상위 이상 윈도우\n\n"
                report += "| 시작 라인 | 이상 점수 | 새 템플릿 비율 | 빈도 Z-score |\n"
                report += "|---------|---------|--------------|-------------|\n"

                top_windows = s.sort_values("score", ascending=False).head(5)
                for _, row in top_windows.iterrows():
                    start = int(row["window_start_line"])
                    score = float(row["score"])
                    unseen = float(row.get("unseen_rate", 0))
                    freq_z = float(row.get("freq_z", 0))
                    report += f"| {start} | {score:.3f} | {unseen:.1%} | {freq_z:.2f} |\n"

                # 실제 로그 샘플 추가
                if parsed_path.exists():
                    dfp = pd.read_parquet(parsed_path)

                    # vocab 로드하여 템플릿 ID -> 템플릿 텍스트 매핑
                    template_map = {}
                    if vocab_path.exists():
                        import json
                        with open(vocab_path, 'r') as f:
                            vocab = json.load(f)
                        # vocab은 template_id -> index 매핑이므로, template를 dfp에서 추출
                        for tid in dfp['template_id'].unique():
                            template_rows = dfp[dfp['template_id'] == tid]
                            if len(template_rows) > 0 and 'template' in template_rows.columns:
                                template_map[tid] = str(template_rows.iloc[0]['template'])

                    report += "\n### 📝 상위 윈도우 상세 분석\n\n"
                    for _, row in top_windows.head(3).iterrows():  # 상위 3개만
                        start = int(row["window_start_line"])
                        score = float(row["score"])
                        unseen = float(row.get("unseen_rate", 0))

                        report += f"#### 윈도우 #{start}~ (점수: {score:.3f})\n\n"

                        win_logs = dfp[(dfp["line_no"] >= start) & (dfp["line_no"] < start + 50)]
                        if len(win_logs) > 0:
                            # 주요 템플릿 분석
                            top_templates = win_logs["template_id"].value_counts().head(3)
                            report += "**주요 템플릿들**:\n"
                            for tid, count in top_templates.items():
                                template_text = template_map.get(tid, f"Template ID: {tid}")
                                report += f"- `{template_text}` - {count}회 출현\n"

                            # 실제 로그 샘플 (에러 우선)
                            error_logs = win_logs[win_logs['raw'].str.contains(
                                r'error|Error|ERROR|fail|Fail|FAIL|exception|Exception|warning|Warning|critical|Critical',
                                case=False, na=False, regex=True
                            )]

                            sample_logs = error_logs.head(3) if len(error_logs) > 0 else win_logs.head(3)

                            report += "\n**실제 로그 샘플**:\n```\n"
                            for _, log in sample_logs.iterrows():
                                timestamp = log.get('timestamp', 'N/A')
                                raw = log.get('raw', 'N/A')
                                report += f"[{timestamp}] {raw}\n"
                            report += "```\n\n"

    # DeepLog 상세 분석
    if deeplog_path.exists():
        d = pd.read_parquet(deeplog_path)
        if len(d) > 0:
            report += "---\n\n## 🧠 DeepLog 이상 탐지 (딥러닝 LSTM)\n\n"
            # Enhanced 버전: prediction_ok 사용, 기존 버전: in_topk 사용
            if "prediction_ok" in d.columns:
                violations = d[d["prediction_ok"] == False]
            elif "in_topk" in d.columns:
                violations = d[d["in_topk"] == False]
            else:
                violations = pd.DataFrame()  # 빈 DataFrame
            report += f"**예측 실패율**: {deeplog_viol:.1%} (전체 {len(d)}개 중 {len(violations)}개 실패)\n\n"

            interpretation = ""
            if deeplog_viol < 0.20:
                interpretation = "✅ **양호**: 로그 패턴이 일관되고 예측 가능합니다."
            elif deeplog_viol < 0.50:
                interpretation = "⚠️ **주의**: 로그 패턴이 다소 복잡하거나 학습 데이터가 부족할 수 있습니다."
            else:
                interpretation = "🔴 **경고**: 로그 패턴이 매우 불규칙하거나 비정상적입니다. 정상 로그로 모델 재학습을 권장합니다."

            report += f"**해석**: {interpretation}\n\n"

            # 예측 실패 샘플 표시 (예측값 vs 실제값)
            if len(violations) > 0 and with_samples:
                report += "### 🔍 예측 실패 상위 샘플\n\n"
                report += "모델이 예측하지 못한 패턴들입니다. 각 샘플은 모델의 예측값과 실제 발생한 값을 보여줍니다.\n\n"

                # vocab이 있는지 확인 (템플릿 문자열 표시용)
                has_template_info = "target_template" in d.columns and "predicted_templates" in d.columns

                if has_template_info:
                    # 템플릿 문자열 정보가 있는 경우
                    sample_count = 0
                    for idx, row in violations.head(5).iterrows():
                        sample_count += 1
                        report += f"#### 샘플 {sample_count}\n\n"
                        report += "| 항목 | 내용 |\n"
                        report += "|------|------|\n"
                        report += f"| **실제 발생** | `{row.get('target_template', 'N/A')}` |\n"
                        report += f"| **모델 예측 (Top-K)** | `{row.get('predicted_templates', 'N/A')}` |\n"
                        report += f"| **분석** | 모델이 예측한 패턴과 다른 로그가 발생하여 이상으로 탐지되었습니다. |\n\n"
                else:
                    # 인덱스 정보만 있는 경우
                    report += "| 샘플 | 실제 템플릿 인덱스 | 예측 Top-1 | 예측 Top-2 | 예측 Top-3 |\n"
                    report += "|------|-------------------|-----------|-----------|------------|\n"

                    for idx, row in violations.head(5).iterrows():
                        target = row.get('target', 'N/A')
                        pred1 = row.get('predicted_top1', '-')
                        pred2 = row.get('predicted_top2', '-')
                        pred3 = row.get('predicted_top3', '-')
                        report += f"| #{idx} | {target} | {pred1} | {pred2} | {pred3} |\n"

                    report += "\n**참고**: vocab.json을 사용하여 추론하면 실제 템플릿 문자열을 볼 수 있습니다.\n\n"
                    report += "```bash\n"
                    report += "alog-detect deeplog-infer --seq sequences.parquet --model model.pth --vocab vocab.json\n"
                    report += "```\n\n"

    # MS-CRED 상세 분석
    if mscred_path.exists():
        m = pd.read_parquet(mscred_path)
        if len(m) > 0:
            report += "---\n\n## 🔬 MS-CRED 이상 탐지 (멀티스케일 오토인코더)\n\n"
            anomalous = m[m["is_anomaly"] == True]
            report += f"**이상 윈도우**: 전체의 {mscred_rate:.1%} ({len(m)}개 중 {len(anomalous)}개)\n\n"

            if len(anomalous) > 0:
                report += "### 상위 재구성 오류\n\n"
                report += "| 윈도우 인덱스 | 재구성 오류 | 임계값 | 오류/임계값 비율 |\n"
                report += "|------------|-----------|--------|---------------|\n"

                top_errors = m.nlargest(5, 'reconstruction_error')
                for _, row in top_errors.iterrows():
                    idx = int(row["window_idx"])
                    error = float(row["reconstruction_error"])
                    threshold = float(row.get("threshold", 0))
                    ratio = error / threshold if threshold > 0 else 0
                    report += f"| {idx} | {error:.4f} | {threshold:.4f} | {ratio:.2f}x |\n"
                report += "\n"

    # 권장사항
    report += "---\n\n## 💡 권장사항\n\n"
    report += "### 🔴 즉시 조치 필요\n\n"

    recommendations = []
    if baseline_rate and baseline_rate > 0.05:
        recommendations.append("- **이상 윈도우 구간 확인**: Baseline에서 발견된 이상 구간의 로그를 상세 분석하세요")
    if deeplog_viol and deeplog_viol > 0.50:
        recommendations.append("- **DeepLog 모델 재학습**: 정상 로그 패턴으로 모델을 재학습하여 정확도를 개선하세요")
    if mscred_rate and mscred_rate > 0.20:
        recommendations.append("- **로그 패턴 구조 분석**: MS-CRED에서 발견된 구조적 이상을 분석하세요")

    if not recommendations:
        recommendations.append("- ✅ 현재 시스템 상태가 양호합니다. 정기적인 모니터링을 계속하세요")

    report += "\n".join(recommendations) + "\n\n"

    report += "### 🟡 추가 분석 권장\n\n"
    report += "- **시간대별 분석**: `alog-detect analyze-temporal --data-dir <dir>` 실행\n"
    report += "- **상세 로그 샘플 확인**: `alog-detect analyze-samples --processed-dir <dir>` 실행\n"
    report += "- **비교 분석**: 여러 시스템 간 로그 패턴 비교\n\n"

    if with_samples:
        report += "---\n\n"
        report += "## 📄 상세 분석 리포트\n\n"
        sample_report = processed_dir / "log_samples_analysis" / "anomaly_analysis_report.md"
        if sample_report.exists():
            report += f"✅ 실제 이상 로그 샘플 분석이 완료되었습니다.\n\n"
            report += f"**상세 리포트**: `{sample_report}`\n\n"
        else:
            report += "⏳ 상세 로그 샘플 분석이 진행 중입니다...\n\n"
    else:
        report += "---\n\n"
        report += "💡 **Tip**: `--with-samples` 옵션을 사용하면 실제 이상 로그 샘플과 상세 분석을 확인할 수 있습니다.\n\n"

    report += "---\n\n"
    report += f"**참고**: 이 리포트는 자동 생성되었습니다. 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    return report


@main.command("report")  # 리포트 생성
@click.option("--processed-dir", type=click.Path(file_okay=False, path_type=Path), required=True)  # 산출물 디렉토리
@click.option("--with-samples/--no-samples", default=True, help="이상 로그 샘플 분석 포함 (기본: 포함)")  # 샘플 분석 옵션
def report_cmd(processed_dir: Path, with_samples: bool) -> None:  # 리포트 실행
    """산출물 요약 리포트 생성 (개선된 읽기 쉬운 형식)."""  # 설명
    print(f"[DEBUG] report_cmd 함수 시작", flush=True)
    print(f"[DEBUG] processed_dir = {processed_dir}", flush=True)
    print(f"[DEBUG] with_samples = {with_samples}", flush=True)

    import pandas as pd  # 지역 임포트
    print(f"[DEBUG] pandas import 완료", flush=True)

    processed_dir.mkdir(parents=True, exist_ok=True)  # 폴더 생성
    print(f"[DEBUG] 디렉토리 생성 완료: {processed_dir}", flush=True)

    click.echo("📊 리포트 생성 중...")
    print(f"[DEBUG] click.echo 호출 완료", flush=True)

    # 로그 샘플 분석 먼저 실행 (with_samples=True인 경우)
    if with_samples:
        print(f"[DEBUG] with_samples=True, 로그 샘플 분석 시작", flush=True)
        click.echo("🔍 이상 로그 샘플 분석 중...")
        try:
            print(f"[DEBUG] log_samples 모듈 import 시도", flush=True)
            from .analyzers.log_samples import main as log_samples_main
            import sys
            print(f"[DEBUG] log_samples import 완료", flush=True)

            # Save current sys.argv
            old_argv = sys.argv
            sys.argv = [
                "analyze-samples",
                str(processed_dir),
                "--output-dir", str(processed_dir / "log_samples_analysis")
            ]
            print(f"[DEBUG] sys.argv 설정 완료: {sys.argv}", flush=True)

            try:  # 분석 실행 보호
                print(f"[DEBUG] log_samples_main() 호출", flush=True)
                log_samples_main()
                print(f"[DEBUG] log_samples_main() 완료", flush=True)
                click.echo("✅ 로그 샘플 분석 완료")
            finally:
                sys.argv = old_argv
                print(f"[DEBUG] sys.argv 복원 완료", flush=True)
        except Exception as e:
            print(f"[DEBUG] 로그 샘플 분석 중 예외 발생: {e}", flush=True)
            import traceback
            traceback.print_exc()
            click.echo(f"⚠️ 로그 샘플 분석 중 오류: {e}", err=True)
    else:
        print(f"[DEBUG] with_samples=False, 로그 샘플 분석 스킵", flush=True)

    # 개선된 리포트 생성
    print(f"[DEBUG] _generate_enhanced_report() 호출 시작", flush=True)
    try:
        report_content = _generate_enhanced_report(processed_dir, with_samples)
        print(f"[DEBUG] _generate_enhanced_report() 완료, 길이={len(report_content)}", flush=True)
    except Exception as e:
        print(f"[DEBUG] _generate_enhanced_report() 중 예외: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    # 리포트 저장
    print(f"[DEBUG] 리포트 파일 쓰기 시작", flush=True)
    out_md = processed_dir / "report.md"
    out_md.write_text(report_content)
    print(f"[DEBUG] 리포트 파일 쓰기 완료: {out_md}", flush=True)

    click.echo(f"\n✅ 리포트 생성 완료!")
    click.echo(f"📄 주요 리포트: {out_md}")
    print(f"[DEBUG] report_cmd 함수 종료", flush=True)

    if with_samples:  # 샘플 분석 경로 출력
        sample_report = processed_dir / "log_samples_analysis" / "anomaly_analysis_report.md"
        if sample_report.exists():
            click.echo(f"📋 상세 로그 샘플 분석: {sample_report}")
        sample_data = processed_dir / "log_samples_analysis" / "anomaly_samples.json"
        if sample_data.exists():
            click.echo(f"📊 샘플 데이터: {sample_data}")


@main.command("gen-synth")  # 합성 로그 생성
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)  # 출력 경로
@click.option("--lines", "num_lines", type=int, default=5000)  # 라인 수
@click.option("--anomaly-rate", type=float, default=0.02)  # 이상 비율
def gen_synth_cmd(out_path: Path, num_lines: int, anomaly_rate: float) -> None:  # 생성 실행
    """합성 장기 로그 생성 (정상+이상 혼합)."""  # 설명
    p = generate_synthetic_log(str(out_path), num_lines=num_lines, anomaly_rate=anomaly_rate)  # 생성 호출
    click.echo(f"✅ Generated synthetic log: {p}")  # 결과 출력
    click.echo(f"📊 Labels: {p}.labels.parquet")


@main.command("gen-training-data")  # 학습용 데이터 생성
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)  # 출력 경로
@click.option("--lines", "num_lines", type=int, default=10000, help="생성할 로그 라인 수")  # 라인 수
@click.option("--host", default="train-host", help="호스트명")  # 호스트명
def gen_training_data_cmd(out_path: Path, num_lines: int, host: str) -> None:  # 학습 데이터 생성 실행
    """학습용 정상 로그 데이터 생성 (100% 정상 로그)."""  # 설명
    click.echo("📚 학습용 정상 로그 데이터 생성 중...")
    p = generate_training_data(str(out_path), num_lines=num_lines, host=host)  # 생성 호출
    click.echo(f"✅ Generated training data: {p}")
    click.echo(f"   📊 Lines: {num_lines} (모두 정상)")
    click.echo(f"   📋 Labels: {p}.labels.parquet")
    click.echo(f"\n💡 Tip: 이 데이터로 모델을 학습하세요:")
    click.echo(f"   alog-detect parse --input {p} --out-dir data/processed/train")
    click.echo(f"   alog-detect build-deeplog --parsed data/processed/train/parsed.parquet --out-dir data/processed/train")
    click.echo(f"   alog-detect deeplog-train --seq data/processed/train/sequences.parquet --vocab data/processed/train/vocab.json --out models/deeplog.pth")


@main.command("gen-inference-normal")  # 추론용 정상 데이터 생성
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)  # 출력 경로
@click.option("--lines", "num_lines", type=int, default=1000, help="생성할 로그 라인 수")  # 라인 수
@click.option("--host", default="test-host", help="호스트명")  # 호스트명
def gen_inference_normal_cmd(out_path: Path, num_lines: int, host: str) -> None:  # 추론용 정상 데이터 생성
    """추론용 정상 로그 데이터 생성 (False Positive 테스트용, 100% 정상)."""  # 설명
    click.echo("✅ 추론용 정상 로그 데이터 생성 중...")
    p = generate_inference_normal(str(out_path), num_lines=num_lines, host=host)  # 생성 호출
    click.echo(f"✅ Generated inference normal data: {p}")
    click.echo(f"   📊 Lines: {num_lines} (모두 정상)")
    click.echo(f"   📋 Labels: {p}.labels.parquet")
    click.echo(f"\n💡 Tip: 모델이 이 데이터를 정상으로 인식해야 합니다 (False Positive 테스트):")
    click.echo(f"   alog-detect parse --input {p} --out-dir data/processed/test_normal")
    click.echo(f"   alog-detect deeplog-infer --seq data/processed/test_normal/sequences.parquet --model models/deeplog.pth --k 3")


@main.command("gen-inference-anomaly")  # 추론용 비정상 데이터 생성
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)  # 출력 경로
@click.option("--lines", "num_lines", type=int, default=1000, help="생성할 로그 라인 수")  # 라인 수
@click.option("--anomaly-rate", type=float, default=0.15, help="이상 로그 비율 (기본: 15%)")  # 이상 비율
@click.option("--anomaly-types", multiple=True, type=click.Choice(["unseen", "error", "attack", "crash", "burst"]),
              help="포함할 이상 타입 (여러 개 선택 가능, 기본: 모두)")  # 이상 타입
@click.option("--host", default="test-host", help="호스트명")  # 호스트명
def gen_inference_anomaly_cmd(out_path: Path, num_lines: int, anomaly_rate: float,
                             anomaly_types: tuple[str, ...], host: str) -> None:  # 추론용 비정상 데이터 생성
    """추론용 비정상 로그 데이터 생성 (True Positive 테스트용).

    이상 타입:
    - unseen: 학습 시 보지 못한 새로운 템플릿
    - error: 에러 메시지 (ERROR, CRITICAL, FATAL)
    - attack: 보안 공격 시뮬레이션 (SSH brute force, SYN flood)
    - crash: 시스템 크래시 (서비스 실패, kernel panic)
    - burst: 특정 템플릿 급증 (10-30개 연속)
    """  # 설명
    click.echo("🚨 추론용 비정상 로그 데이터 생성 중...")

    # anomaly_types가 비어있으면 None (모두 포함)
    types_list = list(anomaly_types) if anomaly_types else None

    p = generate_inference_anomaly(
        str(out_path),
        num_lines=num_lines,
        anomaly_rate=anomaly_rate,
        anomaly_types=types_list,
        host=host
    )  # 생성 호출

    click.echo(f"✅ Generated inference anomaly data: {p}")
    click.echo(f"   📊 Lines: {num_lines}")
    click.echo(f"   🚨 Target anomaly rate: {anomaly_rate:.1%}")
    click.echo(f"   📋 Labels: {p}.labels.parquet")
    click.echo(f"   📈 Metadata: {p}.meta.json")

    # 메타데이터 읽어서 실제 통계 표시
    import json
    meta_path = Path(str(p) + ".meta.json")
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        click.echo(f"\n📊 생성 통계:")
        click.echo(f"   실제 이상률: {meta['anomaly_rate_actual']:.1%} ({meta['anomaly_count']}/{meta['total_lines']}개)")
        if meta.get('anomaly_type_distribution'):
            click.echo(f"   이상 타입별 분포:")
            for anom_type, count in meta['anomaly_type_distribution'].items():
                click.echo(f"      - {anom_type}: {count}개")

    click.echo(f"\n💡 Tip: 모델이 이 데이터에서 이상을 탐지해야 합니다 (True Positive 테스트):")
    click.echo(f"   alog-detect parse --input {p} --out-dir data/processed/test_anomaly")
    click.echo(f"   alog-detect deeplog-infer --seq data/processed/test_anomaly/sequences.parquet --model models/deeplog.pth --k 3")
    click.echo(f"   alog-detect eval --processed-dir data/processed/test_anomaly --labels {p}.labels.parquet")


@main.command("eval")  # 평가 명령
@click.option("--processed-dir", type=click.Path(file_okay=False, path_type=Path), required=True)  # 산출물 폴더
@click.option("--labels", "labels_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)  # 라벨 경로
@click.option("--window-size", type=int, default=50)  # 윈도우 크기
@click.option("--seq-len", type=int, default=50)  # 시퀀스 길이
def eval_cmd(processed_dir: Path, labels_path: Path, window_size: int, seq_len: int) -> None:  # 평가 실행
    """베이스라인/DeepLog 평가(Precision/Recall/F1)."""  # 설명
    out_lines = []  # 출력 라인
    base = processed_dir / "baseline_scores.parquet"  # 베이스라인 경로
    if base.exists():
        p, r, f1 = evaluate_baseline(str(base), str(labels_path), window_size)  # 베이스라인 평가
        out_lines.append(f"Baseline PRF1: P={p:.3f} R={r:.3f} F1={f1:.3f}")  # 포맷팅
    dlinf = processed_dir / "deeplog_infer.parquet"  # DeepLog 경로
    if dlinf.exists():
        p, r, f1 = evaluate_deeplog(str(dlinf), str(labels_path), seq_len)  # DeepLog 평가
        out_lines.append(f"DeepLog PRF1: P={p:.3f} R={r:.3f} F1={f1:.3f}")  # 포맷팅
    if not out_lines:  # 평가 대상 없을 때
        out_lines = ["No artifacts to evaluate."]  # 메시지
    (processed_dir / "eval.txt").write_text("\n".join(out_lines))  # 파일 저장
    click.echo("\n".join(out_lines))  # 콘솔 출력


@main.command("analyze-samples")  # 이상 샘플 분석
@click.option("--processed-dir", type=click.Path(file_okay=False, path_type=Path), required=True)  # 산출물 폴더
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)  # 출력 폴더
@click.option("--max-samples", type=int, default=5, help="타입별 최대 샘플 수")  # 최대 샘플 수
@click.option("--context-lines", type=int, default=3, help="전후 맥락 라인 수")  # 문맥 라인 수
def analyze_samples_cmd(processed_dir: Path, output_dir: Path, max_samples: int, context_lines: int) -> None:  # 분석 실행
    """이상 로그 샘플 추출 및 분석."""  # 설명
    from .analyzers.log_samples import main as log_samples_main  # 분석 엔트리
    import sys  # argv 조작용
    
    if output_dir is None:
        output_dir = processed_dir / "log_samples_analysis"  # 기본 출력 폴더
    
    click.echo("🔍 이상 로그 샘플 분석 시작...")  # 시작 로그
    
    # Save current sys.argv  # 외부 메인 호환
    old_argv = sys.argv
    sys.argv = [
        "analyze-samples",
        str(processed_dir),
        "--output-dir", str(output_dir),
        "--max-samples", str(max_samples),
        "--context-lines", str(context_lines)
    ]
    
    try:
        result_code = log_samples_main()  # 실행
    finally:
        sys.argv = old_argv  # 복원
    
    if result_code is None:
        result_code = 0  # 기본 성공 코드
    result = type('obj', (object,), {'returncode': result_code, 'stdout': '', 'stderr': ''})  # 간단 래핑
    
    if result.returncode == 0:
        click.echo("✅ 로그 샘플 분석 완료!")
        click.echo(f"📄 사람이 읽기 쉬운 리포트: {output_dir / 'anomaly_analysis_report.md'}")
        click.echo(f"📊 상세 분석 데이터: {output_dir / 'anomaly_samples.json'}")
    else:
        click.echo(f"❌ 분석 실패: {result.stderr}")
        return
    
    # 간단한 요약 출력  # 방법별 이상 개수 출력
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


@main.command("convert-onnx")  # ONNX 변환
@click.option("--deeplog-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="DeepLog 모델 경로")  # DeepLog 모델
@click.option("--mscred-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="MS-CRED 모델 경로")  # MS-CRED 모델
@click.option("--vocab", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="어휘 사전 경로")  # vocab.json
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default="models/onnx", help="ONNX 출력 디렉토리")  # 출력 폴더
@click.option("--validate", is_flag=True, default=False, help="변환 후 검증 실행")  # 검증 실행
@click.option("--seq-len", type=int, default=None, help="DeepLog 시퀀스 길이 (기본: 모델에 저장된 값 사용, ONNX는 dynamic_axes로 다양한 길이 지원)")  # 시퀀스 길이
@click.option("--feature-dim", type=int, default=None, help="MS-CRED 피처 차원 (템플릿 개수, 기본: 자동 감지)")  # 피처 차원
@click.option("--portable", is_flag=True, default=False, help="범용 최적화 모드 (모든 환경에서 사용 가능, 하드웨어 특화 최적화 제외)")  # 범용 최적화
def convert_onnx_cmd(deeplog_model: Path, mscred_model: Path, vocab: Path, output_dir: Path, validate: bool, seq_len: Optional[int], feature_dim: Optional[int], portable: bool) -> None:  # 변환 실행
    """PyTorch 모델을 ONNX 포맷으로 변환."""  # 설명
    try:
        # 하이브리드 시스템 모듈 동적 임포트  # 변환 유틸 사용
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
            seq_len=seq_len,
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


@main.command("hybrid-pipeline")  # 하이브리드 파이프라인
@click.option("--log-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="입력 로그 파일")  # 입력 로그
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), help="출력 디렉토리")  # 출력 폴더
@click.option("--auto-deploy", is_flag=True, default=True, help="자동 배포 준비")  # 자동 배포
@click.option("--models-dir", type=click.Path(file_okay=False, path_type=Path), default="models", help="모델 저장 디렉토리")  # 모델 폴더
def hybrid_pipeline_cmd(log_file: Path, output_dir: Path, auto_deploy: bool, models_dir: Path) -> None:  # 파이프라인 실행
    """하이브리드 시스템 전체 파이프라인 실행 (학습 → ONNX 변환 → 배포 준비)."""  # 설명
    try:
        # 하이브리드 시스템 모듈 동적 임포트  # 파이프라인 클래스 로드
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from hybrid_system.training.auto_converter import AutoConverter
        
        click.echo("🚀 하이브리드 파이프라인 시작...")
        
        converter = AutoConverter(
            models_dir=str(models_dir),
            onnx_dir=str(models_dir / "onnx"),
            deployment_dir=str(models_dir / "deployment")
        )
        
        results = converter.run_full_pipeline(str(log_file), auto_deploy)  # 파이프라인 실행
        
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


@main.command("analyze-temporal")  # 시간 기반 분석
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="분석할 데이터 디렉토리")  # 데이터 폴더
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="결과 저장 디렉토리")  # 출력 폴더
def analyze_temporal_cmd(data_dir: Path, output_dir: Path) -> None:  # 실행
    """시간 기반 이상 탐지 분석."""  # 설명
    from .analyzers.temporal import main as temporal_main  # 모듈 메인
    import sys  # argv 조작
    
    # 임시로 sys.argv 조작  # 하위 CLI 호환
    old_argv = sys.argv
    sys.argv = ['temporal', '--data-dir', str(data_dir)]
    if output_dir:
        sys.argv.extend(['--output-dir', str(output_dir)])
    
    try:
        temporal_main()  # 실행
    finally:
        sys.argv = old_argv  # 복원


@main.command("analyze-comparative")  # 비교 분석
@click.option("--target", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Target 파일")  # 타깃 파일
@click.option("--baselines", multiple=True, required=True, help="Baseline 파일들")  # 베이스라인들
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="결과 저장 디렉토리")  # 출력 폴더
def analyze_comparative_cmd(target: Path, baselines: tuple, output_dir: Path) -> None:  # 실행
    """비교 기반 이상 탐지 분석."""  # 설명
    from .analyzers.comparative import main as comparative_main  # 메인
    import sys  # argv 조작
    
    # 임시로 sys.argv 조작  # 하위 CLI 호환
    old_argv = sys.argv
    sys.argv = ['comparative', '--target', str(target)]
    for baseline in baselines:
        sys.argv.extend(['--baselines', baseline])
    if output_dir:
        sys.argv.extend(['--output-dir', str(output_dir)])
    
    try:
        comparative_main()  # 실행
    finally:
        sys.argv = old_argv  # 복원


@main.command("analyze-mscred")  # MS-CRED 분석
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="MS-CRED 결과 디렉토리")  # 데이터 폴더
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="분석 결과 저장 디렉토리")  # 출력 폴더
def analyze_mscred_cmd(data_dir: Path, output_dir: Path) -> None:  # 실행
    """MS-CRED 전용 분석."""  # 설명
    from .analyzers.mscred_analysis import main as mscred_main  # 메인
    import sys  # argv 조작
    
    # 임시로 sys.argv 조작  # 하위 CLI 호환
    old_argv = sys.argv
    sys.argv = ['mscred', '--data-dir', str(data_dir)]
    if output_dir:
        sys.argv.extend(['--output-dir', str(output_dir)])
    
    try:
        mscred_main()  # 실행
    finally:
        sys.argv = old_argv  # 복원


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


if __name__ == "__main__":
    main()


