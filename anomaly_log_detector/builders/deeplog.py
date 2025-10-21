from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib
import pandas as pd
import torch
from torch import nn


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


class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out)
        return logits


def _make_sequences(series: pd.Series, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq = series.dropna().astype(int).tolist()
    X, Y = [], []
    for i in range(len(seq) - seq_len):
        window = seq[i : i + seq_len]
        target = seq[i + seq_len]
        X.append(window)
        Y.append(target)
    if not X:
        return torch.empty((0, seq_len), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def train_deeplog(sequences_parquet: str | Path, vocab_json: str | Path, out_path: str | Path, seq_len: int = 50, epochs: int = 3, batch_size: int = 64, lr: float = 1e-3) -> Path:
    device = torch.device("cpu")
    df = pd.read_parquet(sequences_parquet)
    with open(vocab_json, "r") as f:
        vocab: Dict[str, int] = json.load(f)
    vocab_size = (max(vocab.values()) + 1) if vocab else 1

    # 단일 시퀀스(전체) 기준 최소 구현
    X, Y = _make_sequences(df["template_index"], seq_len)
    embed_dim = 64
    hidden_dim = 128
    model = DeepLogLSTM(vocab_size=max(1, vocab_size), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    if len(X) == 0:
        torch.save({
            "vocab_size": vocab_size,
            "state_dict": model.state_dict(),
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim
        }, out_path)
        return Path(out_path)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            # 마지막 스텝의 예측만 사용
            last_logits = logits[:, -1, :]
            loss = crit(last_logits, yb)
            loss.backward()
            opt.step()

    torch.save({
        "vocab_size": vocab_size,
        "state_dict": model.state_dict(),
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim
    }, out_path)
    return Path(out_path)


@torch.no_grad()
def infer_deeplog_topk(sequences_parquet: str | Path, model_path: str | Path, k: int = 3) -> pd.DataFrame:
    """기존 단순 top-k inference (하위 호환성 유지)."""
    device = torch.device("cpu")
    state = torch.load(model_path, map_location=device)
    vocab_size = int(state.get("vocab_size", 1))
    seq_len = int(state.get("seq_len", 50))
    model = DeepLogLSTM(vocab_size=vocab_size)
    model.load_state_dict(state["state_dict"])
    model.eval()

    df = pd.read_parquet(sequences_parquet)
    X, Y = _make_sequences(df["template_index"], seq_len)
    if len(X) == 0:
        return pd.DataFrame(columns=["idx", "target", "in_topk"])
    logits = model(X)
    last_logits = logits[:, -1, :]
    topk = torch.topk(last_logits, k=k, dim=1).indices
    in_topk = (topk == Y.unsqueeze(1)).any(dim=1).cpu().numpy()
    return pd.DataFrame({"idx": range(len(in_topk)), "target": Y.cpu().numpy(), "in_topk": in_topk})


# ============================================================================
# Enhanced DeepLog Inference with alert management
# ============================================================================

@dataclass
class EnhancedInferenceConfig:
    """Enhanced DeepLog inference 설정."""
    # Top-K/Top-P 설정
    top_k: Optional[int] = 3
    top_p: Optional[float] = None  # top_p가 설정되면 top_k보다 우선

    # K-of-N 판정 (엔티티별 슬라이딩 윈도우)
    k_of_n_k: int = 7  # N개 중 K개 이상 실패 시 알림
    k_of_n_n: int = 10  # 슬라이딩 윈도우 크기

    # 쿨다운 설정 (초 단위)
    cooldown_seq_fail: int = 60  # 시퀀스 실패 쿨다운
    cooldown_novelty: int = 60  # 노벨티 쿨다운 (집계 단위)

    # 세션화 설정
    session_timeout: int = 300  # 세션 타임아웃 (초) - 5분
    entity_column: str = "host"  # 엔티티 컬럼 (host, process 등)

    # 노벨티 처리
    novelty_enabled: bool = True  # 새 템플릿 탐지 활성화
    vocab_path: Optional[str] = None  # vocab.json 경로 (노벨티 판정용)


@dataclass
class AlertRecord:
    """알림 레코드."""
    timestamp: datetime
    entity: str
    alert_type: str  # "SEQ_FAIL", "NOVELTY"
    signature: str  # 알림 시그니처 (중복 억제용)
    template_id: Optional[str] = None
    sequence_context: Optional[List[int]] = None
    details: Dict = field(default_factory=dict)


class AlertManager:
    """알림 관리 (쿨다운, 중복 억제)."""

    def __init__(self, config: EnhancedInferenceConfig):
        self.config = config
        self.last_alert_time: Dict[str, datetime] = {}  # signature -> last alert time
        self.novelty_aggregation: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)  # (entity, template) -> timestamps

    def _make_signature(self, entity: str, alert_type: str, template_id: Optional[str] = None) -> str:
        """알림 시그니처 생성."""
        if alert_type == "SEQ_FAIL":
            return f"{entity}:SEQ_FAIL"
        elif alert_type == "NOVELTY":
            return f"{entity}:NOVELTY:{template_id}"
        return f"{entity}:{alert_type}"

    def should_alert(self, entity: str, alert_type: str, current_time: datetime,
                    template_id: Optional[str] = None) -> bool:
        """쿨다운 및 중복 억제 체크."""
        signature = self._make_signature(entity, alert_type, template_id)

        # 쿨다운 체크
        if signature in self.last_alert_time:
            cooldown_seconds = (
                self.config.cooldown_seq_fail if alert_type == "SEQ_FAIL"
                else self.config.cooldown_novelty
            )
            elapsed = (current_time - self.last_alert_time[signature]).total_seconds()
            if elapsed < cooldown_seconds:
                return False

        return True

    def record_alert(self, entity: str, alert_type: str, current_time: datetime,
                    template_id: Optional[str] = None):
        """알림 기록."""
        signature = self._make_signature(entity, alert_type, template_id)
        self.last_alert_time[signature] = current_time

        # 노벨티는 추가로 집계
        if alert_type == "NOVELTY" and template_id:
            key = (entity, template_id)
            self.novelty_aggregation[key].append(current_time)
            # 1분 이전 기록은 정리
            cutoff = current_time - timedelta(seconds=self.config.cooldown_novelty)
            self.novelty_aggregation[key] = [
                t for t in self.novelty_aggregation[key] if t >= cutoff
            ]

    def get_novelty_aggregation_summary(self) -> Dict:
        """노벨티 집계 요약."""
        summary = {}
        for (entity, template_id), timestamps in self.novelty_aggregation.items():
            if timestamps:
                summary[f"{entity}:{template_id}"] = {
                    "count": len(timestamps),
                    "first": min(timestamps),
                    "last": max(timestamps)
                }
        return summary


class EntitySession:
    """엔티티별 세션 관리."""

    def __init__(self, entity: str, config: EnhancedInferenceConfig):
        self.entity = entity
        self.config = config
        self.recent_failures: deque = deque(maxlen=config.k_of_n_n)  # 최근 N개 판정 결과
        self.last_activity: Optional[datetime] = None
        self.session_id: str = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """세션 ID 생성."""
        import time
        return hashlib.md5(f"{self.entity}:{time.time()}".encode()).hexdigest()[:16]

    def update(self, is_failure: bool, current_time: datetime) -> bool:
        """
        세션 업데이트 및 K-of-N 판정.

        Returns:
            True if should alert (K개 이상 실패)
        """
        # 세션 타임아웃 체크
        if self.last_activity:
            elapsed = (current_time - self.last_activity).total_seconds()
            if elapsed > self.config.session_timeout:
                # 새 세션 시작
                self.recent_failures.clear()
                self.session_id = self._generate_session_id()

        self.last_activity = current_time
        self.recent_failures.append(is_failure)

        # K-of-N 판정
        if len(self.recent_failures) >= self.config.k_of_n_n:
            failure_count = sum(1 for f in self.recent_failures if f)
            return failure_count >= self.config.k_of_n_k

        return False


@torch.no_grad()
def infer_deeplog_enhanced(
    sequences_parquet: str | Path,
    parsed_parquet: str | Path,
    model_path: str | Path,
    config: EnhancedInferenceConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Enhanced DeepLog inference with alert management.

    Args:
        sequences_parquet: sequences.parquet 파일 경로
        parsed_parquet: parsed.parquet 파일 경로 (timestamp, entity 정보)
        model_path: 학습된 모델 경로
        config: Enhanced inference 설정

    Returns:
        (detailed_results_df, alerts_df, summary_dict)
        - detailed_results_df: 모든 시퀀스 판정 결과
        - alerts_df: 실제 발생한 알림 목록
        - summary_dict: 요약 정보
    """
    device = torch.device("cpu")
    state = torch.load(model_path, map_location=device)
    vocab_size = int(state.get("vocab_size", 1))
    seq_len = int(state.get("seq_len", 50))

    # 모델 로드
    model = DeepLogLSTM(vocab_size=vocab_size)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # 데이터 로드
    seq_df = pd.read_parquet(sequences_parquet)
    parsed_df = pd.read_parquet(parsed_parquet)

    # template_index -> template_id 매핑을 위해 parsed_df에 template_index 추가
    if "template_id" in parsed_df.columns and "template_index" not in parsed_df.columns:
        # vocab 로드하여 template_id -> template_index 매핑
        if config.vocab_path:
            try:
                with open(config.vocab_path, 'r') as f:
                    vocab = json.load(f)
                parsed_df["template_index"] = parsed_df["template_id"].map(vocab).astype("Int64")
            except Exception:
                pass

    # 노벨티 판정을 위한 vocab 로드
    known_templates = set()
    if config.novelty_enabled and config.vocab_path:
        try:
            with open(config.vocab_path, 'r') as f:
                vocab = json.load(f)
                known_templates = set(vocab.keys())
        except Exception:
            pass

    # 시퀀스 생성
    X, Y = _make_sequences(seq_df["template_index"], seq_len)
    if len(X) == 0:
        empty_detailed = pd.DataFrame(columns=[
            "idx", "timestamp", "entity", "target", "is_novel",
            "prediction_ok", "k_of_n_triggered", "session_id"
        ])
        empty_alerts = pd.DataFrame(columns=[
            "alert_id", "timestamp", "entity", "alert_type", "template_id", "details"
        ])
        return empty_detailed, empty_alerts, {}

    # 모델 추론
    logits = model(X)
    last_logits = logits[:, -1, :]  # (batch, vocab_size)
    probs = torch.softmax(last_logits, dim=1)

    # Top-K 또는 Top-P 판정
    if config.top_p is not None:
        # Top-P (nucleus sampling) 방식
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)

        # 누적 확률이 top_p 이하인 토큰들
        predictions_ok = []
        for i, target in enumerate(Y):
            # target이 top-p 누적 확률 내에 있는지 확인
            target_prob = probs[i, target].item()
            target_rank = (sorted_indices[i] == target).nonzero(as_tuple=True)[0].item()
            cumsum_at_target = cumsum_probs[i, target_rank].item()

            # target 이전 누적 확률
            cumsum_before = cumsum_probs[i, target_rank - 1].item() if target_rank > 0 else 0.0

            # target이 top-p 범위에 포함되면 OK
            ok = cumsum_before < config.top_p
            predictions_ok.append(ok)

        predictions_ok = torch.tensor(predictions_ok)
    else:
        # Top-K 방식 (기존)
        k = config.top_k or 3
        topk_indices = torch.topk(last_logits, k=min(k, vocab_size), dim=1).indices
        predictions_ok = (topk_indices == Y.unsqueeze(1)).any(dim=1)

    # 엔티티 및 타임스탬프 매핑
    # seq_df와 parsed_df를 line_no 기준으로 조인
    seq_df_sorted = seq_df.sort_values(["timestamp", "line_no"], kind="stable")

    # 시퀀스 인덱스와 parsed_df 매핑 (seq_len 이후부터 시작)
    results = []
    alerts = []

    # 엔티티별 세션 관리
    entity_sessions: Dict[str, EntitySession] = {}
    alert_manager = AlertManager(config)

    for idx in range(len(X)):
        # 현재 시퀀스의 마지막 라인 (예측 대상의 이전 라인)
        # X[idx]는 seq_len개의 template_index를 포함
        # Y[idx]는 예측 대상
        # 전체 시퀀스에서의 절대 위치
        abs_pos = idx + seq_len

        if abs_pos >= len(seq_df_sorted):
            continue

        row = seq_df_sorted.iloc[abs_pos]
        timestamp_val = row.get("timestamp", None)
        entity_val = row.get(config.entity_column, "unknown")

        # timestamp를 datetime으로 변환
        try:
            if pd.isna(timestamp_val):
                current_time = datetime.now()
            elif isinstance(timestamp_val, str):
                current_time = datetime.fromisoformat(timestamp_val.replace('Z', '+00:00'))
            elif isinstance(timestamp_val, pd.Timestamp):
                current_time = timestamp_val.to_pydatetime()
            else:
                current_time = datetime.now()
        except:
            current_time = datetime.now()

        # 노벨티 판정
        target_idx = Y[idx].item()
        target_template = None
        is_novel = False

        # template_index -> template_id 역매핑 필요
        # 여기서는 단순화: parsed_df에서 target_idx와 일치하는 template_id 찾기
        if config.novelty_enabled:
            # parsed_df에서 template_index가 target_idx인 행 찾기
            if "template_index" in parsed_df.columns:
                matching_rows = parsed_df[parsed_df["template_index"] == target_idx]
                if not matching_rows.empty and "template_id" in matching_rows.columns:
                    target_template = str(matching_rows.iloc[0]["template_id"])
                    if target_template and target_template not in known_templates:
                        is_novel = True

        # 예측 성공 여부
        pred_ok = predictions_ok[idx].item()
        is_failure = not pred_ok

        # 엔티티 세션 가져오기/생성
        if entity_val not in entity_sessions:
            entity_sessions[entity_val] = EntitySession(entity_val, config)

        session = entity_sessions[entity_val]
        k_of_n_triggered = session.update(is_failure, current_time)

        # 알림 판정
        should_alert_seq = False
        should_alert_nov = False

        if k_of_n_triggered and is_failure:
            # K-of-N 조건 만족 + 실패 -> SEQ_FAIL 알림 고려
            if alert_manager.should_alert(entity_val, "SEQ_FAIL", current_time):
                should_alert_seq = True
                alert_manager.record_alert(entity_val, "SEQ_FAIL", current_time)

                alerts.append({
                    "alert_id": len(alerts),
                    "timestamp": current_time,
                    "line_no": int(row.get("line_no", -1)),
                    "entity": entity_val,
                    "alert_type": "SEQ_FAIL",
                    "template_id": None,
                    "session_id": session.session_id,
                    "k_of_n_failures": sum(1 for f in session.recent_failures if f),
                    "details": {
                        "target_template_idx": target_idx,
                        "is_novel": is_novel,
                        "recent_failures": list(session.recent_failures)
                    }
                })

        if is_novel and target_template:
            # 노벨티 발견 -> NOVELTY 알림 고려
            if alert_manager.should_alert(entity_val, "NOVELTY", current_time, target_template):
                should_alert_nov = True
                alert_manager.record_alert(entity_val, "NOVELTY", current_time, target_template)

                alerts.append({
                    "alert_id": len(alerts),
                    "timestamp": current_time,
                    "line_no": int(row.get("line_no", -1)),
                    "entity": entity_val,
                    "alert_type": "NOVELTY",
                    "template_id": target_template,
                    "session_id": session.session_id,
                    "details": {
                        "target_template_idx": target_idx,
                        "first_occurrence": True
                    }
                })

        # 상세 결과 기록
        results.append({
            "idx": idx,
            "timestamp": current_time,
            "entity": entity_val,
            "target": target_idx,
            "is_novel": is_novel,
            "prediction_ok": pred_ok,
            "k_of_n_triggered": k_of_n_triggered,
            "session_id": session.session_id,
            "alerted_seq_fail": should_alert_seq,
            "alerted_novelty": should_alert_nov
        })

    # DataFrame 생성
    detailed_df = pd.DataFrame(results)
    alerts_df = pd.DataFrame(alerts)

    # 요약 정보
    summary = {
        "total_sequences": len(results),
        "total_failures": int((~predictions_ok).sum()),
        "total_novels": int(detailed_df["is_novel"].sum() if not detailed_df.empty else 0),
        "total_alerts": len(alerts),
        "alerts_by_type": alerts_df["alert_type"].value_counts().to_dict() if not alerts_df.empty else {},
        "total_sessions": len(entity_sessions),
        "entities_analyzed": len(entity_sessions),
        "novelty_aggregation": alert_manager.get_novelty_aggregation_summary()
    }

    return detailed_df, alerts_df, summary


