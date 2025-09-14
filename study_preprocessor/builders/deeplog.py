from __future__ import annotations

from pathlib import Path
import json
from typing import Dict
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
    model = DeepLogLSTM(vocab_size=max(1, vocab_size), embed_dim=64, hidden_dim=128).to(device)
    if len(X) == 0:
        torch.save({"vocab_size": vocab_size, "state_dict": model.state_dict(), "seq_len": seq_len}, out_path)
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

    torch.save({"vocab_size": vocab_size, "state_dict": model.state_dict(), "seq_len": seq_len}, out_path)
    return Path(out_path)


@torch.no_grad()
def infer_deeplog_topk(sequences_parquet: str | Path, model_path: str | Path, k: int = 3) -> pd.DataFrame:
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


