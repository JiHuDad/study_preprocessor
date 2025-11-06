from __future__ import annotations  # 타입 힌트에서 문자열 리터럴을 앞으로 참조하기 위한 기능 활성화

from pathlib import Path  # 경로 처리를 위한 모듈
import json  # JSON 데이터 처리를 위한 모듈
from typing import Dict, Optional, List, Tuple  # 타입 힌트를 위한 모듈
from dataclasses import dataclass, field  # 데이터 클래스를 정의하기 위한 데코레이터
from collections import defaultdict, deque  # 기본값이 있는 딕셔너리와 데크(양방향 큐)를 위한 모듈
from datetime import datetime, timedelta  # 날짜/시간 처리를 위한 모듈
import hashlib  # 해시 함수를 위한 모듈
import pandas as pd  # 데이터프레임 처리를 위한 라이브러리
import torch  # PyTorch 딥러닝 프레임워크
from torch import nn  # 신경망 레이어 모듈


def build_deeplog_inputs(parsed_parquet: str | Path, out_dir: str | Path, template_col: str = "template") -> None:
    """DeepLog 학습을 위한 입력 데이터 생성 함수 (vocab.json, sequences.parquet).
    
    Args:
        parsed_parquet: 파싱된 로그 데이터 Parquet 파일 경로
        out_dir: 출력 디렉토리 경로
        template_col: 템플릿 컬럼명 (기본값: "template", 실제 템플릿 문자열 사용)
    """
    out = Path(out_dir)  # 경로 문자열을 Path 객체로 변환
    out.mkdir(parents=True, exist_ok=True)  # 출력 디렉토리가 없으면 생성
    df = pd.read_parquet(parsed_parquet)  # 파싱된 로그 데이터 읽기

    # Build vocab mapping using actual template strings (NOT template_id)  # 실제 템플릿 문자열을 사용하여 어휘 매핑 생성 (template_id가 아님)
    # CRITICAL: Use "template" column (actual template string) for C engine compatibility  # 중요: C 엔진 호환성을 위해 "template" 컬럼(실제 템플릿 문자열) 사용
    # If template_col is "template_id", this will create {"1": 0, "2": 1} which is wrong!  # template_col이 "template_id"이면 {"1": 0, "2": 1}처럼 잘못된 형식이 생성됨!
    unique_templates = [t for t in df[template_col].dropna().astype(str).unique()]  # 고유한 템플릿 문자열 추출 (NaN 제외)
    vocab: Dict[str, int] = {t: i for i, t in enumerate(sorted(unique_templates))}  # 템플릿 문자열을 정렬 후 인덱스와 매핑 (템플릿 문자열 -> 인덱스)
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2))  # vocab.json 파일로 저장

    # Map to indices and export sequences by host (if available) or global  # 인덱스로 매핑하고 호스트별(가능한 경우) 또는 전역으로 시퀀스 내보내기
    df = df.sort_values(["timestamp", "line_no"], kind="stable", na_position="first")  # 타임스탬프와 라인 번호로 안정 정렬 (NA값은 앞쪽에)
    df["template_index"] = df[template_col].map(vocab).astype("Int64")  # 템플릿 문자열을 vocab을 사용하여 인덱스로 매핑
    df[["line_no", "timestamp", "host", "template_index"]].to_parquet(out / "sequences.parquet", index=False)  # sequences.parquet 파일로 저장 (라인 번호, 타임스탬프, 호스트, 템플릿 인덱스)


class DeepLogLSTM(nn.Module):  # DeepLog LSTM 신경망 모델 클래스
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):  # 초기화 메서드
        super().__init__()  # 부모 클래스(nn.Module) 초기화
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # 임베딩 레이어 (어휘 크기 -> 임베딩 차원, 패딩 인덱스는 0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)  # LSTM 레이어 (입력 차원, 은닉 차원, 레이어 수, 배치 우선 순서)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # 완전 연결(Linear) 레이어 (은닉 차원 -> 어휘 크기, 다음 토큰 예측)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 순전파 메서드
        emb = self.embed(x)  # 입력 텐서를 임베딩 벡터로 변환
        out, _ = self.lstm(emb)  # LSTM을 통과시켜 은닉 상태 출력 (히든 상태와 셀 상태는 사용하지 않음)
        logits = self.fc(out)  # 완전 연결 레이어를 통과시켜 각 위치의 어휘별 로짓(확률) 생성
        return logits  # 로짓 반환


def _make_sequences(series: pd.Series, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """시퀀스 데이터를 학습용 입력/출력 쌍으로 변환하는 헬퍼 함수.
    
    Args:
        series: 템플릿 인덱스 시퀀스
        seq_len: 시퀀스 길이
        
    Returns:
        (X, Y): 입력 시퀀스 텐서와 타깃 텐서 튜플
    """
    seq = series.dropna().astype(int).tolist()  # NaN을 제거하고 정수로 변환한 후 리스트로 변환
    X, Y = [], []  # 입력 시퀀스와 타깃 시퀀스를 저장할 리스트 초기화
    for i in range(len(seq) - seq_len):  # 슬라이딩 윈도우로 시퀀스 생성
        window = seq[i : i + seq_len]  # 입력 윈도우 (seq_len 길이)
        target = seq[i + seq_len]  # 타깃 (다음 토큰)
        X.append(window)  # 입력 시퀀스 리스트에 추가
        Y.append(target)  # 타깃 리스트에 추가
    if not X:  # 시퀀스가 비어있으면
        return torch.empty((0, seq_len), dtype=torch.long), torch.empty((0,), dtype=torch.long)  # 빈 텐서 반환
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)  # PyTorch 텐서로 변환하여 반환


def train_deeplog(sequences_parquet: str | Path, vocab_json: str | Path, out_path: str | Path, seq_len: int = 50, epochs: int = 3, batch_size: int = 64, lr: float = 1e-3) -> Path:
    """DeepLog LSTM 모델 학습 함수.
    
    Args:
        sequences_parquet: 시퀀스 데이터 Parquet 파일 경로
        vocab_json: 어휘 사전 JSON 파일 경로
        out_path: 저장할 모델 파일 경로
        seq_len: 시퀀스 길이 (기본값: 50)
        epochs: 학습 에폭 수 (기본값: 3)
        batch_size: 배치 크기 (기본값: 64)
        lr: 학습률 (기본값: 0.001)
        
    Returns:
        학습된 모델 저장 경로
    """
    device = torch.device("cpu")  # 계산 장치 설정 (CPU 사용)
    df = pd.read_parquet(sequences_parquet)  # 시퀀스 데이터 읽기
    with open(vocab_json, "r") as f:  # 어휘 사전 파일 열기
        vocab: Dict[str, int] = json.load(f)  # JSON 파일에서 어휘 사전 로드
    vocab_size = (max(vocab.values()) + 1) if vocab else 1  # 어휘 크기 계산 (최대 인덱스 + 1, 빈 경우 1)

    # 단일 시퀀스(전체) 기준 최소 구현  # 단일 시퀀스(전체) 기준 최소 구현
    X, Y = _make_sequences(df["template_index"], seq_len)  # 입력/출력 시퀀스 생성
    embed_dim = 64  # 임베딩 차원 설정
    hidden_dim = 128  # LSTM 은닉 차원 설정
    model = DeepLogLSTM(vocab_size=max(1, vocab_size), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)  # 모델 생성 및 장치로 이동
    if len(X) == 0:  # 학습 데이터가 없으면
        torch.save({  # 빈 모델 상태 저장
            "vocab_size": vocab_size,  # 어휘 크기
            "state_dict": model.state_dict(),  # 모델 상태 딕셔너리
            "seq_len": seq_len,  # 시퀀스 길이
            "embed_dim": embed_dim,  # 임베딩 차원
            "hidden_dim": hidden_dim  # 은닉 차원
        }, out_path)  # 모델 파일로 저장
        return Path(out_path)  # 저장 경로 반환

    dataset = torch.utils.data.TensorDataset(X, Y)  # 데이터셋 생성 (입력과 출력을 쌍으로 묶음)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 데이터 로더 생성 (배치 처리, 셔플 활성화)
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # Adam 옵티마이저 생성 (모델 파라미터, 학습률)
    crit = nn.CrossEntropyLoss()  # 크로스 엔트로피 손실 함수 생성

    model.train()  # 모델을 학습 모드로 설정
    for _ in range(max(1, epochs)):  # 지정된 에폭 수만큼 반복 (최소 1회)
        for xb, yb in loader:  # 각 배치마다 반복
            opt.zero_grad()  # 옵티마이저의 기울기 초기화
            logits = model(xb)  # 모델에 입력 배치를 넣어 로짓 출력
            # 마지막 스텝의 예측만 사용  # 마지막 스텝의 예측만 사용
            last_logits = logits[:, -1, :]  # 시퀀스의 마지막 위치의 로짓만 추출
            loss = crit(last_logits, yb)  # 손실 계산 (예측 vs 실제)
            loss.backward()  # 역전파로 기울기 계산
            opt.step()  # 옵티마이저로 파라미터 업데이트

    torch.save({  # 학습 완료 후 모델 저장
        "vocab_size": vocab_size,  # 어휘 크기
        "state_dict": model.state_dict(),  # 학습된 모델 상태 딕셔너리
        "seq_len": seq_len,  # 시퀀스 길이
        "embed_dim": embed_dim,  # 임베딩 차원
        "hidden_dim": hidden_dim  # 은닉 차원
    }, out_path)  # 모델 파일로 저장
    return Path(out_path)  # 저장 경로 반환


@torch.no_grad()  # 기울기 계산 비활성화 (추론 시 사용)
def infer_deeplog_topk(sequences_parquet: str | Path, model_path: str | Path, vocab_path: str | Path | None = None, k: int = 3) -> pd.DataFrame:
    """기존 단순 top-k inference (하위 호환성 유지).

    Args:
        sequences_parquet: 시퀀스 데이터 Parquet 파일 경로
        model_path: 학습된 모델 파일 경로
        vocab_path: vocab.json 파일 경로 (선택사항 - 예측/실제 템플릿 문자열 포함 시 필요)
        k: Top-K 값 (기본값: 3)

    Returns:
        추론 결과 DataFrame (idx, target, in_topk, predicted_top1~topK, target_template, predicted_templates)
    """
    device = torch.device("cpu")  # 계산 장치 설정 (CPU 사용)
    state = torch.load(model_path, map_location=device)  # 모델 상태 로드
    vocab_size = int(state.get("vocab_size", 1))  # 어휘 크기 추출
    seq_len = int(state.get("seq_len", 50))  # 시퀀스 길이 추출
    model = DeepLogLSTM(vocab_size=vocab_size)  # 모델 생성
    model.load_state_dict(state["state_dict"])  # 모델 상태 로드
    model.eval()  # 모델을 평가 모드로 설정

    # vocab 로드 (템플릿 문자열 매핑용)
    vocab_map = None  # 인덱스 -> 템플릿 문자열 매핑
    if vocab_path and Path(vocab_path).exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)  # {"template_string": index}
            # 역매핑: {index: "template_string"}
            vocab_map = {int(idx): template for template, idx in vocab.items()}

    df = pd.read_parquet(sequences_parquet)  # 시퀀스 데이터 읽기
    X, Y = _make_sequences(df["template_index"], seq_len)  # 입력/출력 시퀀스 생성
    if len(X) == 0:  # 시퀀스가 비어있으면
        base_cols = ["idx", "target", "in_topk"]
        pred_cols = [f"predicted_top{i+1}" for i in range(k)]
        if vocab_map:
            base_cols.extend(["target_template", "predicted_templates"])
        return pd.DataFrame(columns=base_cols + pred_cols)  # 빈 DataFrame 반환

    logits = model(X)  # 모델에 입력하여 로짓 출력
    last_logits = logits[:, -1, :]  # 시퀀스의 마지막 위치의 로짓만 추출
    topk = torch.topk(last_logits, k=min(k, vocab_size), dim=1).indices  # Top-K 인덱스 추출
    in_topk = (topk == Y.unsqueeze(1)).any(dim=1).cpu().numpy()  # 타깃이 Top-K에 포함되는지 확인

    # 결과 DataFrame 구성
    result_data = {
        "idx": range(len(in_topk)),
        "target": Y.cpu().numpy(),
        "in_topk": in_topk
    }

    # Top-K 예측값 인덱스 추가
    topk_np = topk.cpu().numpy()
    for i in range(k):
        if i < topk_np.shape[1]:
            result_data[f"predicted_top{i+1}"] = topk_np[:, i]
        else:
            result_data[f"predicted_top{i+1}"] = [-1] * len(in_topk)

    # vocab이 있으면 템플릿 문자열도 추가
    if vocab_map:
        target_templates = []
        predicted_templates_list = []

        for i in range(len(Y)):
            target_idx = int(Y[i].item())
            target_template = vocab_map.get(target_idx, f"<Unknown:{target_idx}>")
            target_templates.append(target_template)

            # Top-K 예측 템플릿들
            pred_templates = []
            for j in range(min(k, topk_np.shape[1])):
                pred_idx = int(topk_np[i, j])
                pred_template = vocab_map.get(pred_idx, f"<Unknown:{pred_idx}>")
                pred_templates.append(pred_template)
            predicted_templates_list.append(" | ".join(pred_templates))

        result_data["target_template"] = target_templates
        result_data["predicted_templates"] = predicted_templates_list

    return pd.DataFrame(result_data)  # 결과 DataFrame 반환


# ============================================================================  # 구분선
# Enhanced DeepLog Inference with alert management  # 향상된 DeepLog 추론 (알림 관리 포함)
# ============================================================================  # 구분선

@dataclass  # 데이터 클래스 데코레이터
class EnhancedInferenceConfig:  # 향상된 DeepLog 추론 설정 클래스
    """Enhanced DeepLog inference 설정."""  # 향상된 DeepLog 추론 설정
    # Top-K/Top-P 설정  # Top-K/Top-P 설정
    top_k: Optional[int] = 3  # Top-K 값 (기본값: 3)
    top_p: Optional[float] = None  # top_p가 설정되면 top_k보다 우선  # Top-P 값 (설정 시 top_k보다 우선, 기본값: None)

    # K-of-N 판정 (엔티티별 슬라이딩 윈도우)  # K-of-N 판정 (엔티티별 슬라이딩 윈도우)
    k_of_n_k: int = 7  # N개 중 K개 이상 실패 시 알림  # N개 중 K개 이상 실패 시 알림 (기본값: 7)
    k_of_n_n: int = 10  # 슬라이딩 윈도우 크기  # 슬라이딩 윈도우 크기 (기본값: 10)

    # 쿨다운 설정 (초 단위)  # 쿨다운 설정 (초 단위)
    cooldown_seq_fail: int = 60  # 시퀀스 실패 쿨다운  # 시퀀스 실패 쿨다운 (기본값: 60초)
    cooldown_novelty: int = 60  # 노벨티 쿨다운 (집계 단위)  # 노벨티 쿨다운 (집계 단위, 기본값: 60초)

    # 세션화 설정  # 세션화 설정
    session_timeout: int = 300  # 세션 타임아웃 (초) - 5분  # 세션 타임아웃 (초, 기본값: 300초 - 5분)
    entity_column: str = "host"  # 엔티티 컬럼 (host, process 등)  # 엔티티 컬럼 (host, process 등, 기본값: "host")

    # 노벨티 처리  # 노벨티 처리
    novelty_enabled: bool = True  # 새 템플릿 탐지 활성화  # 새 템플릿 탐지 활성화 (기본값: True)
    vocab_path: Optional[str] = None  # vocab.json 경로 (노벨티 판정용)  # vocab.json 경로 (노벨티 판정용, 기본값: None)


@dataclass  # 데이터 클래스 데코레이터
class AlertRecord:  # 알림 레코드 클래스
    """알림 레코드."""  # 알림 레코드
    timestamp: datetime  # 타임스탬프
    entity: str  # 엔티티 (호스트, 프로세스 등)
    alert_type: str  # "SEQ_FAIL", "NOVELTY"  # 알림 유형 ("SEQ_FAIL", "NOVELTY")
    signature: str  # 알림 시그니처 (중복 억제용)  # 알림 시그니처 (중복 억제용)
    template_id: Optional[str] = None  # 템플릿 ID (선택적)
    sequence_context: Optional[List[int]] = None  # 시퀀스 컨텍스트 (선택적)
    details: Dict = field(default_factory=dict)  # 상세 정보 (기본값: 빈 딕셔너리)


class AlertManager:  # 알림 관리 클래스
    """알림 관리 (쿨다운, 중복 억제)."""  # 알림 관리 (쿨다운, 중복 억제)

    def __init__(self, config: EnhancedInferenceConfig):  # 초기화 메서드
        self.config = config  # 설정 저장
        self.last_alert_time: Dict[str, datetime] = {}  # signature -> last alert time  # 시그니처별 마지막 알림 시간 딕셔너리
        self.novelty_aggregation: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)  # (entity, template) -> timestamps  # 노벨티 집계 딕셔너리 ((엔티티, 템플릿) -> 타임스탬프 리스트)

    def _make_signature(self, entity: str, alert_type: str, template_id: Optional[str] = None) -> str:  # 알림 시그니처 생성 메서드
        """알림 시그니처 생성."""  # 알림 시그니처 생성
        if alert_type == "SEQ_FAIL":  # 시퀀스 실패 알림인 경우
            return f"{entity}:SEQ_FAIL"  # "엔티티:SEQ_FAIL" 형식 반환
        elif alert_type == "NOVELTY":  # 노벨티 알림인 경우
            return f"{entity}:NOVELTY:{template_id}"  # "엔티티:NOVELTY:템플릿ID" 형식 반환
        return f"{entity}:{alert_type}"  # 기본 형식 반환

    def should_alert(self, entity: str, alert_type: str, current_time: datetime,  # 알림 발생 여부 판정 메서드
                    template_id: Optional[str] = None) -> bool:
        """쿨다운 및 중복 억제 체크."""  # 쿨다운 및 중복 억제 체크
        signature = self._make_signature(entity, alert_type, template_id)  # 알림 시그니처 생성

        # 쿨다운 체크  # 쿨다운 체크
        if signature in self.last_alert_time:  # 시그니처가 이전 알림 기록에 있으면
            cooldown_seconds = (  # 쿨다운 시간 결정
                self.config.cooldown_seq_fail if alert_type == "SEQ_FAIL"  # 시퀀스 실패 알림인 경우
                else self.config.cooldown_novelty  # 그 외는 노벨티 쿨다운
            )
            elapsed = (current_time - self.last_alert_time[signature]).total_seconds()  # 경과 시간 계산
            if elapsed < cooldown_seconds:  # 경과 시간이 쿨다운 시간보다 작으면
                return False  # 알림 발생 안 함

        return True  # 알림 발생

    def record_alert(self, entity: str, alert_type: str, current_time: datetime,  # 알림 기록 메서드
                    template_id: Optional[str] = None):
        """알림 기록."""  # 알림 기록
        signature = self._make_signature(entity, alert_type, template_id)  # 알림 시그니처 생성
        self.last_alert_time[signature] = current_time  # 마지막 알림 시간 업데이트

        # 노벨티는 추가로 집계  # 노벨티는 추가로 집계
        if alert_type == "NOVELTY" and template_id:  # 노벨티 알림이고 템플릿 ID가 있으면
            key = (entity, template_id)  # 집계 키 생성 (엔티티, 템플릿 ID)
            self.novelty_aggregation[key].append(current_time)  # 타임스탬프 추가
            # 1분 이전 기록은 정리  # 1분 이전 기록은 정리
            cutoff = current_time - timedelta(seconds=self.config.cooldown_novelty)  # 컷오프 시간 계산
            self.novelty_aggregation[key] = [  # 쿨다운 기간 이후 기록만 유지
                t for t in self.novelty_aggregation[key] if t >= cutoff  # 컷오프 시간 이후 타임스탬프만 필터링
            ]

    def get_novelty_aggregation_summary(self) -> Dict:  # 노벨티 집계 요약 메서드
        """노벨티 집계 요약."""  # 노벨티 집계 요약
        summary = {}  # 요약 딕셔너리 초기화
        for (entity, template_id), timestamps in self.novelty_aggregation.items():  # 집계 데이터 순회
            if timestamps:  # 타임스탬프가 있으면
                summary[f"{entity}:{template_id}"] = {  # 요약에 추가
                    "count": len(timestamps),  # 개수
                    "first": min(timestamps),  # 첫 발생 시간
                    "last": max(timestamps)  # 마지막 발생 시간
                }
        return summary  # 요약 반환


class EntitySession:  # 엔티티별 세션 관리 클래스
    """엔티티별 세션 관리."""  # 엔티티별 세션 관리

    def __init__(self, entity: str, config: EnhancedInferenceConfig):  # 초기화 메서드
        self.entity = entity  # 엔티티 저장
        self.config = config  # 설정 저장
        self.recent_failures: deque = deque(maxlen=config.k_of_n_n)  # 최근 N개 판정 결과  # 최근 실패 판정 결과 데크 (최대 크기: k_of_n_n)
        self.last_activity: Optional[datetime] = None  # 마지막 활동 시간 (초기값: None)
        self.session_id: str = self._generate_session_id()  # 세션 ID 생성

    def _generate_session_id(self) -> str:  # 세션 ID 생성 메서드
        """세션 ID 생성."""  # 세션 ID 생성
        import time  # 시간 모듈 임포트
        return hashlib.md5(f"{self.entity}:{time.time()}".encode()).hexdigest()[:16]  # 엔티티와 현재 시간을 MD5 해시로 변환 후 16자리 추출

    def update(self, is_failure: bool, current_time: datetime) -> bool:  # 세션 업데이트 메서드
        """
        세션 업데이트 및 K-of-N 판정.  # 세션 업데이트 및 K-of-N 판정

        Returns:  # 반환값
            True if should alert (K개 이상 실패)  # K개 이상 실패 시 True 반환
        """
        # 세션 타임아웃 체크  # 세션 타임아웃 체크
        if self.last_activity:  # 이전 활동이 있으면
            elapsed = (current_time - self.last_activity).total_seconds()  # 경과 시간 계산
            if elapsed > self.config.session_timeout:  # 타임아웃을 초과하면
                # 새 세션 시작  # 새 세션 시작
                self.recent_failures.clear()  # 실패 기록 초기화
                self.session_id = self._generate_session_id()  # 새 세션 ID 생성

        self.last_activity = current_time  # 마지막 활동 시간 업데이트
        self.recent_failures.append(is_failure)  # 실패 판정 결과 추가

        # K-of-N 판정  # K-of-N 판정
        if len(self.recent_failures) >= self.config.k_of_n_n:  # 실패 기록이 윈도우 크기 이상이면
            failure_count = sum(1 for f in self.recent_failures if f)  # 실패 개수 계산
            return failure_count >= self.config.k_of_n_k  # 실패 개수가 K 이상이면 True 반환

        return False  # 그 외는 False 반환


@torch.no_grad()  # 기울기 계산 비활성화 (추론 시 사용)
def infer_deeplog_enhanced(
    sequences_parquet: str | Path,  # 시퀀스 데이터 Parquet 파일 경로
    parsed_parquet: str | Path,  # 파싱된 로그 데이터 Parquet 파일 경로 (타임스탬프, 엔티티 정보)
    model_path: str | Path,  # 학습된 모델 파일 경로
    config: EnhancedInferenceConfig  # 향상된 추론 설정
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:  # (상세 결과 DataFrame, 알림 DataFrame, 요약 딕셔너리) 튜플 반환
    """
    Enhanced DeepLog inference with alert management.  # 향상된 DeepLog 추론 (알림 관리 포함)

    Args:  # 인자
        sequences_parquet: sequences.parquet 파일 경로  # sequences.parquet 파일 경로
        parsed_parquet: parsed.parquet 파일 경로 (timestamp, entity 정보)  # parsed.parquet 파일 경로 (타임스탬프, 엔티티 정보)
        model_path: 학습된 모델 경로  # 학습된 모델 경로
        config: Enhanced inference 설정  # 향상된 추론 설정

    Returns:  # 반환값
        (detailed_results_df, alerts_df, summary_dict)  # (상세 결과 DataFrame, 알림 DataFrame, 요약 딕셔너리)
        - detailed_results_df: 모든 시퀀스 판정 결과  # 모든 시퀀스 판정 결과
        - alerts_df: 실제 발생한 알림 목록  # 실제 발생한 알림 목록
        - summary_dict: 요약 정보  # 요약 정보
    """
    device = torch.device("cpu")  # 계산 장치 설정 (CPU 사용)
    state = torch.load(model_path, map_location=device)  # 모델 상태 로드
    vocab_size = int(state.get("vocab_size", 1))  # 어휘 크기 추출
    seq_len = int(state.get("seq_len", 50))  # 시퀀스 길이 추출

    # 모델 로드  # 모델 로드
    model = DeepLogLSTM(vocab_size=vocab_size)  # 모델 생성
    model.load_state_dict(state["state_dict"])  # 모델 상태 로드
    model.eval()  # 모델을 평가 모드로 설정

    # 데이터 로드  # 데이터 로드
    seq_df = pd.read_parquet(sequences_parquet)  # 시퀀스 데이터 읽기
    parsed_df = pd.read_parquet(parsed_parquet)  # 파싱된 로그 데이터 읽기

    # template_index -> template_id 매핑을 위해 parsed_df에 template_index 추가  # template_index -> template_id 매핑을 위해 parsed_df에 template_index 추가
    if "template_id" in parsed_df.columns and "template_index" not in parsed_df.columns:  # template_id는 있지만 template_index가 없으면
        # vocab 로드하여 template_id -> template_index 매핑  # vocab 로드하여 template_id -> template_index 매핑
        if config.vocab_path:  # vocab 경로가 설정되어 있으면
            try:  # 예외 처리
                with open(config.vocab_path, 'r') as f:  # vocab 파일 열기
                    vocab = json.load(f)  # JSON 파일에서 어휘 사전 로드
                parsed_df["template_index"] = parsed_df["template_id"].map(vocab).astype("Int64")  # template_id를 template_index로 매핑
            except Exception:  # 예외 발생 시
                pass  # 무시

    # 노벨티 판정을 위한 vocab 로드  # 노벨티 판정을 위한 vocab 로드
    known_templates = set()  # 알려진 템플릿 집합 초기화
    if config.novelty_enabled and config.vocab_path:  # 노벨티 탐지가 활성화되어 있고 vocab 경로가 있으면
        try:  # 예외 처리
            with open(config.vocab_path, 'r') as f:  # vocab 파일 열기
                vocab = json.load(f)  # JSON 파일에서 어휘 사전 로드
                known_templates = set(vocab.keys())  # vocab의 키(템플릿 문자열)를 집합으로 변환
        except Exception:  # 예외 발생 시
            pass  # 무시

    # 시퀀스 생성  # 시퀀스 생성
    X, Y = _make_sequences(seq_df["template_index"], seq_len)  # 입력/출력 시퀀스 생성
    if len(X) == 0:  # 시퀀스가 비어있으면
        empty_detailed = pd.DataFrame(columns=[  # 빈 상세 결과 DataFrame 생성
            "idx", "timestamp", "entity", "target", "is_novel",  # 컬럼: 인덱스, 타임스탬프, 엔티티, 타깃, 노벨티 여부
            "prediction_ok", "k_of_n_triggered", "session_id"  # 컬럼: 예측 성공, K-of-N 트리거, 세션 ID
        ])
        empty_alerts = pd.DataFrame(columns=[  # 빈 알림 DataFrame 생성
            "alert_id", "timestamp", "entity", "alert_type", "template_id", "details"  # 컬럼: 알림 ID, 타임스탬프, 엔티티, 알림 유형, 템플릿 ID, 상세 정보
        ])
        return empty_detailed, empty_alerts, {}  # 빈 DataFrame과 딕셔너리 반환

    # 모델 추론  # 모델 추론
    logits = model(X)  # 모델에 입력하여 로짓 출력
    last_logits = logits[:, -1, :]  # (batch, vocab_size)  # 시퀀스의 마지막 위치의 로짓만 추출
    probs = torch.softmax(last_logits, dim=1)  # 소프트맥스로 확률 계산

    # Top-K 또는 Top-P 판정  # Top-K 또는 Top-P 판정
    if config.top_p is not None:  # Top-P가 설정되어 있으면
        # Top-P (nucleus sampling) 방식  # Top-P (nucleus sampling) 방식
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)  # 확률을 내림차순으로 정렬
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)  # 누적 확률 계산

        # 누적 확률이 top_p 이하인 토큰들  # 누적 확률이 top_p 이하인 토큰들
        predictions_ok = []  # 예측 성공 여부 리스트 초기화
        for i, target in enumerate(Y):  # 각 타깃에 대해 반복
            # target이 top-p 누적 확률 내에 있는지 확인  # target이 top-p 누적 확률 내에 있는지 확인
            target_prob = probs[i, target].item()  # 타깃 확률 추출
            target_rank = (sorted_indices[i] == target).nonzero(as_tuple=True)[0].item()  # 타깃의 순위 계산
            cumsum_at_target = cumsum_probs[i, target_rank].item()  # 타깃까지의 누적 확률

            # target 이전 누적 확률  # target 이전 누적 확률
            cumsum_before = cumsum_probs[i, target_rank - 1].item() if target_rank > 0 else 0.0  # 타깃 이전까지의 누적 확률 (타깃이 첫 번째면 0.0)

            # target이 top-p 범위에 포함되면 OK  # target이 top-p 범위에 포함되면 OK
            ok = cumsum_before < config.top_p  # 타깃 이전 누적 확률이 top_p보다 작으면 OK
            predictions_ok.append(ok)  # 예측 성공 여부 추가

        predictions_ok = torch.tensor(predictions_ok)  # 텐서로 변환
    else:  # Top-P가 설정되어 있지 않으면
        # Top-K 방식 (기존)  # Top-K 방식 (기존)
        k = config.top_k or 3  # Top-K 값 설정 (없으면 3)
        topk_indices = torch.topk(last_logits, k=min(k, vocab_size), dim=1).indices  # Top-K 인덱스 추출
        predictions_ok = (topk_indices == Y.unsqueeze(1)).any(dim=1)  # 타깃이 Top-K에 포함되는지 확인

    # 엔티티 및 타임스탬프 매핑  # 엔티티 및 타임스탬프 매핑
    # seq_df와 parsed_df를 line_no 기준으로 조인  # seq_df와 parsed_df를 line_no 기준으로 조인
    seq_df_sorted = seq_df.sort_values(["timestamp", "line_no"], kind="stable")  # 타임스탬프와 라인 번호로 정렬

    # 시퀀스 인덱스와 parsed_df 매핑 (seq_len 이후부터 시작)  # 시퀀스 인덱스와 parsed_df 매핑 (seq_len 이후부터 시작)
    results = []  # 상세 결과 리스트 초기화
    alerts = []  # 알림 리스트 초기화

    # 엔티티별 세션 관리  # 엔티티별 세션 관리
    entity_sessions: Dict[str, EntitySession] = {}  # 엔티티별 세션 딕셔너리 초기화
    alert_manager = AlertManager(config)  # 알림 관리자 생성

    for idx in range(len(X)):  # 각 시퀀스에 대해 반복
        # 현재 시퀀스의 마지막 라인 (예측 대상의 이전 라인)  # 현재 시퀀스의 마지막 라인 (예측 대상의 이전 라인)
        # X[idx]는 seq_len개의 template_index를 포함  # X[idx]는 seq_len개의 template_index를 포함
        # Y[idx]는 예측 대상  # Y[idx]는 예측 대상
        # 전체 시퀀스에서의 절대 위치  # 전체 시퀀스에서의 절대 위치
        abs_pos = idx + seq_len  # 절대 위치 계산

        if abs_pos >= len(seq_df_sorted):  # 절대 위치가 시퀀스 길이를 초과하면
            continue  # 건너뛰기

        row = seq_df_sorted.iloc[abs_pos]  # 해당 위치의 행 추출
        timestamp_val = row.get("timestamp", None)  # 타임스탬프 값 추출
        entity_val = row.get(config.entity_column, "unknown")  # 엔티티 값 추출 (기본값: "unknown")

        # timestamp를 datetime으로 변환  # timestamp를 datetime으로 변환
        try:  # 예외 처리
            if pd.isna(timestamp_val):  # 타임스탬프가 NaN이면
                current_time = datetime.now()  # 현재 시간 사용
            elif isinstance(timestamp_val, str):  # 타임스탬프가 문자열이면
                current_time = datetime.fromisoformat(timestamp_val.replace('Z', '+00:00'))  # ISO 형식으로 파싱
            elif isinstance(timestamp_val, pd.Timestamp):  # 타임스탬프가 pandas Timestamp이면
                current_time = timestamp_val.to_pydatetime()  # Python datetime으로 변환
            else:  # 그 외
                current_time = datetime.now()  # 현재 시간 사용
        except:  # 예외 발생 시
            current_time = datetime.now()  # 현재 시간 사용

        # 노벨티 판정  # 노벨티 판정
        target_idx = Y[idx].item()  # 타깃 인덱스 추출
        target_template = None  # 타깃 템플릿 초기화
        is_novel = False  # 노벨티 여부 초기화

        # template_index -> template_id 역매핑 필요  # template_index -> template_id 역매핑 필요
        # 여기서는 단순화: parsed_df에서 target_idx와 일치하는 template_id 찾기  # 여기서는 단순화: parsed_df에서 target_idx와 일치하는 template_id 찾기
        if config.novelty_enabled:  # 노벨티 탐지가 활성화되어 있으면
            # parsed_df에서 template_index가 target_idx인 행 찾기  # parsed_df에서 template_index가 target_idx인 행 찾기
            if "template_index" in parsed_df.columns:  # template_index 컬럼이 있으면
                matching_rows = parsed_df[parsed_df["template_index"] == target_idx]  # 매칭되는 행 찾기
                if not matching_rows.empty and "template_id" in matching_rows.columns:  # 매칭되는 행이 있고 template_id 컬럼이 있으면
                    target_template = str(matching_rows.iloc[0]["template_id"])  # 템플릿 ID 추출
                    if target_template and target_template not in known_templates:  # 템플릿 ID가 있고 알려진 템플릿에 없으면
                        is_novel = True  # 노벨티로 설정

        # 예측 성공 여부  # 예측 성공 여부
        pred_ok = predictions_ok[idx].item()  # 예측 성공 여부 추출
        is_failure = not pred_ok  # 실패 여부 계산

        # 엔티티 세션 가져오기/생성  # 엔티티 세션 가져오기/생성
        if entity_val not in entity_sessions:  # 엔티티 세션이 없으면
            entity_sessions[entity_val] = EntitySession(entity_val, config)  # 새 세션 생성

        session = entity_sessions[entity_val]  # 엔티티 세션 가져오기
        k_of_n_triggered = session.update(is_failure, current_time)  # 세션 업데이트 및 K-of-N 판정

        # 알림 판정  # 알림 판정
        should_alert_seq = False  # 시퀀스 실패 알림 여부 초기화
        should_alert_nov = False  # 노벨티 알림 여부 초기화

        if k_of_n_triggered and is_failure:  # K-of-N 조건이 만족되고 실패인 경우
            # K-of-N 조건 만족 + 실패 -> SEQ_FAIL 알림 고려  # K-of-N 조건 만족 + 실패 -> SEQ_FAIL 알림 고려
            if alert_manager.should_alert(entity_val, "SEQ_FAIL", current_time):  # 알림 발생 가능 여부 확인
                should_alert_seq = True  # 시퀀스 실패 알림 발생
                alert_manager.record_alert(entity_val, "SEQ_FAIL", current_time)  # 알림 기록

                alerts.append({  # 알림 리스트에 추가
                    "alert_id": len(alerts),  # 알림 ID (현재 알림 개수)
                    "timestamp": current_time,  # 타임스탬프
                    "line_no": int(row.get("line_no", -1)),  # 라인 번호
                    "entity": entity_val,  # 엔티티
                    "alert_type": "SEQ_FAIL",  # 알림 유형 (시퀀스 실패)
                    "template_id": None,  # 템플릿 ID (없음)
                    "session_id": session.session_id,  # 세션 ID
                    "k_of_n_failures": sum(1 for f in session.recent_failures if f),  # K-of-N 실패 개수
                    "details": {  # 상세 정보
                        "target_template_idx": target_idx,  # 타깃 템플릿 인덱스
                        "is_novel": is_novel,  # 노벨티 여부
                        "recent_failures": list(session.recent_failures)  # 최근 실패 기록 리스트
                    }
                })

        if is_novel and target_template:  # 노벨티이고 타깃 템플릿이 있으면
            # 노벨티 발견 -> NOVELTY 알림 고려  # 노벨티 발견 -> NOVELTY 알림 고려
            if alert_manager.should_alert(entity_val, "NOVELTY", current_time, target_template):  # 알림 발생 가능 여부 확인
                should_alert_nov = True  # 노벨티 알림 발생
                alert_manager.record_alert(entity_val, "NOVELTY", current_time, target_template)  # 알림 기록

                alerts.append({  # 알림 리스트에 추가
                    "alert_id": len(alerts),  # 알림 ID (현재 알림 개수)
                    "timestamp": current_time,  # 타임스탬프
                    "line_no": int(row.get("line_no", -1)),  # 라인 번호
                    "entity": entity_val,  # 엔티티
                    "alert_type": "NOVELTY",  # 알림 유형 (노벨티)
                    "template_id": target_template,  # 템플릿 ID
                    "session_id": session.session_id,  # 세션 ID
                    "details": {  # 상세 정보
                        "target_template_idx": target_idx,  # 타깃 템플릿 인덱스
                        "first_occurrence": True  # 첫 발생 여부
                    }
                })

        # 상세 결과 기록  # 상세 결과 기록
        results.append({  # 결과 리스트에 추가
            "idx": idx,  # 인덱스
            "timestamp": current_time,  # 타임스탬프
            "entity": entity_val,  # 엔티티
            "target": target_idx,  # 타깃 인덱스
            "is_novel": is_novel,  # 노벨티 여부
            "prediction_ok": pred_ok,  # 예측 성공 여부
            "k_of_n_triggered": k_of_n_triggered,  # K-of-N 트리거 여부
            "session_id": session.session_id,  # 세션 ID
            "alerted_seq_fail": should_alert_seq,  # 시퀀스 실패 알림 여부
            "alerted_novelty": should_alert_nov  # 노벨티 알림 여부
        })

    # DataFrame 생성  # DataFrame 생성
    detailed_df = pd.DataFrame(results)  # 상세 결과 DataFrame 생성
    alerts_df = pd.DataFrame(alerts)  # 알림 DataFrame 생성

    # 요약 정보  # 요약 정보
    summary = {  # 요약 딕셔너리 생성
        "total_sequences": len(results),  # 전체 시퀀스 개수
        "total_failures": int((~predictions_ok).sum()),  # 전체 실패 개수
        "total_novels": int(detailed_df["is_novel"].sum() if not detailed_df.empty else 0),  # 전체 노벨티 개수
        "total_alerts": len(alerts),  # 전체 알림 개수
        "alerts_by_type": alerts_df["alert_type"].value_counts().to_dict() if not alerts_df.empty else {},  # 알림 유형별 개수
        "total_sessions": len(entity_sessions),  # 전체 세션 개수
        "entities_analyzed": len(entity_sessions),  # 분석된 엔티티 개수
        "novelty_aggregation": alert_manager.get_novelty_aggregation_summary()  # 노벨티 집계 요약
    }

    return detailed_df, alerts_df, summary  # 상세 결과, 알림, 요약 반환

