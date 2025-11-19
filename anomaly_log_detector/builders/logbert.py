"""LogBERT 기반 로그 이상탐지 모듈

LogBERT는 BERT(Bidirectional Encoder Representations from Transformers)를 사용하여
로그 시퀀스의 정상 패턴을 학습하고 이상을 탐지합니다.

주요 기능:
- build_logbert_inputs: LogBERT 학습용 입력 데이터 생성
- train_logbert: BERT 모델 학습 (Masked Language Modeling)
- infer_logbert: 학습된 모델로 이상 로그 탐지
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


def build_logbert_inputs(
    parsed_parquet: str | Path,
    out_dir: str | Path,
    template_col: str = "template",
    vocab_path: str | Path | None = None,
    max_seq_len: int = 512
) -> None:
    """LogBERT 학습을 위한 입력 데이터 생성 (vocab.json, sequences.parquet).

    Args:
        parsed_parquet: 파싱된 로그 데이터 Parquet 파일 경로
        out_dir: 출력 디렉토리 경로
        template_col: 템플릿 컬럼명 (기본값: "template")
        vocab_path: 기존 vocab.json 경로 (선택사항)
        max_seq_len: 최대 시퀀스 길이 (기본값: 512)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parsed_parquet)

    # Vocab 로드 또는 생성
    if vocab_path and Path(vocab_path).exists():
        with open(vocab_path, 'r') as f:
            vocab: Dict[str, int] = json.load(f)
        print(f"✅ 기존 vocab 사용: {vocab_path} (크기: {len(vocab)})")
    else:
        # BERT 특수 토큰 추가
        unique_templates = [t for t in df[template_col].dropna().astype(str).unique()]
        # 특수 토큰: [PAD], [CLS], [SEP], [MASK]
        special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        all_tokens = special_tokens + sorted(unique_templates)
        vocab = {t: i for i, t in enumerate(all_tokens)}

        # vocab.json 저장
        vocab_file = out / "vocab.json"
        vocab_file.write_text(json.dumps(vocab, indent=2))
        print(f"✅ 새로운 vocab 생성: {vocab_file} (크기: {len(vocab)})")

    # 특수 토큰 인덱스 저장
    special_indices = {
        "pad_token_id": vocab.get("[PAD]", 0),
        "cls_token_id": vocab.get("[CLS]", 1),
        "sep_token_id": vocab.get("[SEP]", 2),
        "mask_token_id": vocab.get("[MASK]", 3),
        "unk_token_id": vocab.get("[UNK]", 4)
    }
    (out / "special_tokens.json").write_text(json.dumps(special_indices, indent=2))

    # 템플릿을 인덱스로 매핑
    df = df.sort_values(["timestamp", "line_no"], kind="stable", na_position="first")
    df["template_index"] = df[template_col].map(vocab).fillna(special_indices["unk_token_id"]).astype("Int64")

    # 시퀀스로 저장
    df[["line_no", "timestamp", "host", "template_index"]].to_parquet(
        out / "sequences.parquet", index=False
    )
    print(f"✅ 시퀀스 저장: {out / 'sequences.parquet'}")


class LogBERTDataset(Dataset):
    """LogBERT용 데이터셋 클래스 (Masked Language Modeling)"""

    def __init__(
        self,
        sequences: List[int],
        seq_len: int = 128,
        mask_ratio: float = 0.15,
        mask_token_id: int = 3,
        pad_token_id: int = 0
    ):
        """
        Args:
            sequences: 템플릿 인덱스 시퀀스
            seq_len: 시퀀스 길이
            mask_ratio: 마스킹 비율 (기본값: 0.15)
            mask_token_id: [MASK] 토큰 ID
            pad_token_id: [PAD] 토큰 ID
        """
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        # 시퀀스를 고정 길이로 분할
        self.sequences = []
        for i in range(0, len(sequences), seq_len):
            seq = sequences[i:i + seq_len]
            # 패딩
            if len(seq) < seq_len:
                seq = seq + [pad_token_id] * (seq_len - len(seq))
            self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: 마스킹된 입력 시퀀스
            labels: 원본 레이블 (마스킹되지 않은 위치는 -100)
            attention_mask: 어텐션 마스크 (패딩 위치는 0)
        """
        seq = self.sequences[idx].copy()
        labels = [-100] * self.seq_len  # -100은 loss 계산에서 무시됨

        # 어텐션 마스크 생성 (패딩이 아닌 위치는 1)
        attention_mask = [1 if token != self.pad_token_id else 0 for token in seq]

        # 랜덤 마스킹
        for i in range(self.seq_len):
            if seq[i] != self.pad_token_id and np.random.random() < self.mask_ratio:
                labels[i] = seq[i]  # 원본 레이블 저장
                seq[i] = self.mask_token_id  # 마스킹

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long)
        )


class LogBERTModel(nn.Module):
    """LogBERT 모델 (간소화된 BERT 아키텍처)"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

        # MLM head
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()

        # Position IDs 생성
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Attention mask 변환 (0 -> -inf, 1 -> 0)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # 반전 (True인 곳을 마스킹)

        # Transformer encoding
        hidden_states = self.encoder(embeddings, src_key_padding_mask=attention_mask)

        # MLM prediction
        logits = self.mlm_head(hidden_states)

        return logits


def train_logbert(
    sequences_parquet: str | Path,
    vocab_json: str | Path,
    out_path: str | Path,
    seq_len: int = 128,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 5e-5,
    mask_ratio: float = 0.15,
    hidden_size: int = 256,
    num_layers: int = 4,
    num_heads: int = 8
) -> Path:
    """LogBERT 모델 학습

    Args:
        sequences_parquet: 시퀀스 데이터 Parquet 파일 경로
        vocab_json: 어휘 사전 JSON 파일 경로
        out_path: 저장할 모델 파일 경로
        seq_len: 시퀀스 길이 (기본값: 128)
        epochs: 학습 에폭 수 (기본값: 10)
        batch_size: 배치 크기 (기본값: 32)
        lr: 학습률 (기본값: 5e-5)
        mask_ratio: 마스킹 비율 (기본값: 0.15)
        hidden_size: 은닉층 크기 (기본값: 256)
        num_layers: Transformer 레이어 수 (기본값: 4)
        num_heads: Attention head 수 (기본값: 8)

    Returns:
        학습된 모델 저장 경로
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 사용 장치: {device}")

    # Vocab 로드
    with open(vocab_json, 'r') as f:
        vocab: Dict[str, int] = json.load(f)
    vocab_size = len(vocab)
    print(f"📚 Vocab 크기: {vocab_size}")

    # 특수 토큰 로드
    special_tokens_path = Path(vocab_json).parent / "special_tokens.json"
    if special_tokens_path.exists():
        with open(special_tokens_path, 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {
            "pad_token_id": 0,
            "mask_token_id": 3
        }

    # 시퀀스 로드
    df = pd.read_parquet(sequences_parquet)
    sequences = df["template_index"].dropna().astype(int).tolist()
    print(f"📊 총 로그 수: {len(sequences)}")

    # 데이터셋 생성
    dataset = LogBERTDataset(
        sequences=sequences,
        seq_len=seq_len,
        mask_ratio=mask_ratio,
        mask_token_id=special_tokens["mask_token_id"],
        pad_token_id=special_tokens["pad_token_id"]
    )
    print(f"📦 데이터셋 크기: {len(dataset)} 시퀀스")

    # 데이터 로더
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 생성
    model = LogBERTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        max_position_embeddings=seq_len
    ).to(device)

    # 옵티마이저 및 손실 함수
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # -100은 무시

    # 학습
    model.train()
    print(f"🚀 학습 시작 ({epochs} epochs)...")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for input_ids, labels, attention_mask in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Loss 계산 (마스킹된 토큰에 대해서만)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # 모델 저장
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "vocab_size": vocab_size,
        "state_dict": model.state_dict(),
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "special_tokens": special_tokens
    }, out_path)

    print(f"✅ 모델 저장: {out_path}")
    return out_path


@torch.no_grad()
def infer_logbert(
    sequences_parquet: str | Path,
    model_path: str | Path,
    vocab_json: str | Path,
    threshold_percentile: float = 95.0,
    seq_len: int = 128
) -> pd.DataFrame:
    """LogBERT 이상 탐지 추론

    Args:
        sequences_parquet: 시퀀스 데이터 Parquet 파일 경로
        model_path: 학습된 모델 파일 경로
        vocab_json: vocab.json 파일 경로
        threshold_percentile: 이상 판정 임계값 백분위수 (기본값: 95.0)
        seq_len: 시퀀스 길이

    Returns:
        추론 결과 DataFrame (seq_idx, avg_loss, is_anomaly)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    state = torch.load(model_path, map_location=device)
    vocab_size = state["vocab_size"]
    hidden_size = state.get("hidden_size", 256)
    num_layers = state.get("num_layers", 4)
    num_heads = state.get("num_heads", 8)
    special_tokens = state.get("special_tokens", {"pad_token_id": 0})

    model = LogBERTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        max_position_embeddings=seq_len
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    print(f"✅ 모델 로드: {model_path}")

    # 시퀀스 로드
    df = pd.read_parquet(sequences_parquet)
    sequences = df["template_index"].dropna().astype(int).tolist()

    # 데이터셋 생성 (추론용 - 마스킹 없음)
    pad_token_id = special_tokens["pad_token_id"]

    # 시퀀스를 고정 길이로 분할
    seq_list = []
    for i in range(0, len(sequences), seq_len):
        seq = sequences[i:i + seq_len]
        if len(seq) < seq_len:
            seq = seq + [pad_token_id] * (seq_len - len(seq))
        seq_list.append(seq)

    # 각 시퀀스의 perplexity 계산
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')

    print(f"🔍 추론 시작 ({len(seq_list)} 시퀀스)...")

    for seq in seq_list:
        input_ids = torch.tensor([seq], dtype=torch.long).to(device)
        attention_mask = torch.tensor(
            [[1 if token != pad_token_id else 0 for token in seq]],
            dtype=torch.long
        ).to(device)

        # Forward pass
        logits = model(input_ids, attention_mask)

        # Loss 계산 (각 토큰에 대해)
        labels = input_ids.clone()
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

        # 유효한 토큰(패딩 아님)에 대한 평균 loss
        valid_mask = (attention_mask.view(-1) == 1)
        if valid_mask.sum() > 0:
            avg_loss = loss[valid_mask].mean().item()
        else:
            avg_loss = 0.0

        losses.append(avg_loss)

    # 이상 탐지 (임계값 기반)
    losses_array = np.array(losses)
    threshold = np.percentile(losses_array, threshold_percentile)
    is_anomaly = losses_array > threshold

    print(f"📊 Loss 통계:")
    print(f"  - 평균: {losses_array.mean():.4f}")
    print(f"  - 중앙값: {np.median(losses_array):.4f}")
    print(f"  - 임계값 (p{threshold_percentile}): {threshold:.4f}")
    print(f"  - 이상 시퀀스 수: {is_anomaly.sum()} / {len(losses)}")

    # 결과 DataFrame 생성
    result_df = pd.DataFrame({
        "seq_idx": range(len(losses)),
        "avg_loss": losses,
        "is_anomaly": is_anomaly,
        "threshold": threshold
    })

    return result_df
