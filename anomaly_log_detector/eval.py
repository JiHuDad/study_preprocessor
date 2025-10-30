# 평가 유틸리티 모듈
# 목적:
# - 공통 평가 지표 계산: 베이스라인(윈도우 기반)과 DeepLog(시퀀스 기반) 결과에 대해
#   동일한 방식으로 Precision/Recall/F1을 계산하여 일관성과 재사용성을 보장
# - 라벨 정렬(정답 매핑) 유틸 제공:
#   * evaluate_baseline: 윈도우 시작(start_index/window_start_line)과 window_size로
#     라벨을 윈도우에 매핑해 정답 생성
#   * evaluate_deeplog: 시퀀스 인덱스를 라인 번호로 근사(line_no = idx + seq_len)하여
#     라벨과 조인하고 prediction_ok/in_topk 기반으로 이상 여부를 정규화
# - 중복 제거와 테스트 용이성: 평가 로직을 중앙화해 실험 비교·회귀 테스트·CI에 활용
# - 입출력 표준화: Parquet 형식 점수/라벨을 입력받아 표준 지표를 반환하는 최소 API 제공

from __future__ import annotations  # 미래 표기법: 타입 힌트 전방 참조 허용

from pathlib import Path  # 경로 타입 사용
from typing import Tuple  # 반환 타입 튜플 표기

import numpy as np  # 수치 계산용
import pandas as pd  # 데이터프레임 처리용


def _align_labels_to_windows(labels_df: pd.DataFrame, scores_df: pd.DataFrame, window_size: int) -> pd.Series:  # 라벨을 윈도우에 정렬
    # 윈도우 [start, start+window_size) 범위에 라벨 이상이 하나라도 있으면 해당 윈도우를 이상으로 표시
    win_labels = []  # 윈도우 단위 라벨 목록
    label_idx = 0  # (미사용) 라벨 인덱스 자리표시자
    labels = labels_df.sort_values("line_no").reset_index(drop=True)  # 라벨을 라인 번호로 정렬 및 재인덱싱
    anom_lines = labels[labels["is_anomaly"] == 1]["line_no"].tolist()  # 이상 라인의 라인 번호 목록
    for _, row in scores_df.iterrows():  # 각 점수(윈도우) 행 순회
        start = int(row["window_start_line"]) if "window_start_line" in row else int(row.get("start_index", 0))  # 윈도우 시작 라인 추출
        end = start + window_size  # 윈도우 끝 라인 계산 (포함 안 됨)
        # [start, end) 구간에 이상 라인이 하나라도 포함되는지 확인
        flag = any(start <= a < end for a in anom_lines)  # 이상 포함 여부
        win_labels.append(1 if flag else 0)  # 포함되면 1, 아니면 0
    return pd.Series(win_labels, name="win_label")  # 윈도우 라벨 시리즈 반환


def evaluate_baseline(baseline_scores: str | Path, labels_path: str | Path, window_size: int) -> Tuple[float, float, float]:  # 베이스라인 평가
    s = pd.read_parquet(baseline_scores)  # 베이스라인 점수 로드
    y_true = _align_labels_to_windows(pd.read_parquet(labels_path), s, window_size)  # 라벨을 윈도우에 정렬
    y_pred = (s["is_anomaly"].astype(bool)).astype(int)  # 이상 여부를 정수(0/1)로 변환
    return _prf1(y_true.values, y_pred.values)  # 정밀도/재현율/F1 계산


def evaluate_deeplog(infer_parquet: str | Path, labels_path: str | Path, seq_len: int) -> Tuple[float, float, float]:  # DeepLog 평가
    d = pd.read_parquet(infer_parquet)  # 추론 결과 로드
    # 시퀀스 인덱스로 정렬: 대략적인 라인 번호를 idx + seq_len으로 정렬
    d = d.assign(line_no=d["idx"].astype(int) + int(seq_len))  # 라인 번호 보정 컬럼 추가
    
    # 이상 플래그 생성 - 향상 버전은 prediction_ok 사용, 기존 버전은 in_topk 사용
    if 'prediction_ok' in d.columns:  # 향상 버전 결과가 있는 경우
        d = d.assign(is_anom=(~d["prediction_ok"]).astype(int))  # 성공(False) -> 이상(1)로 변환
    elif 'in_topk' in d.columns:  # 기존 버전 결과가 있는 경우
        d = d.assign(is_anom=(~d["in_topk"]).astype(int))  # Top-K 미포함(True) -> 이상(1)
    else:  # 둘 다 없는 경우
        d = d.assign(is_anom=0)  # 기본값: 이상 아님
    
    labels = pd.read_parquet(labels_path)[["line_no", "is_anomaly"]]  # 라벨 로드 (라인 번호, 이상 여부)
    merged = pd.merge(d, labels, on="line_no", how="inner")  # 라인 번호 기준으로 내부 조인
    y_true = merged["is_anomaly"].astype(int).values  # 정답 라벨 배열
    y_pred = merged["is_anom"].astype(int).values  # 예측 라벨 배열
    return _prf1(y_true, y_pred)  # PRF1 계산


def _prf1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:  # PRF1 계산 헬퍼
    tp = int(((y_true == 1) & (y_pred == 1)).sum())  # True Positive 개수
    fp = int(((y_true == 0) & (y_pred == 1)).sum())  # False Positive 개수
    fn = int(((y_true == 1) & (y_pred == 0)).sum())  # False Negative 개수
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 정밀도 계산 (분모 0 방지)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 재현율 계산 (분모 0 방지)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0  # F1 스코어
    return precision, recall, f1  # (정밀도, 재현율, F1) 반환
  # 파일 끝

