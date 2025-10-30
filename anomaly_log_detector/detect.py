"""로그 윈도우 기반 베이스라인 이상탐지 구현

- 입력: parsed.parquet (timestamp, line_no, template_id)
- 처리: 슬라이딩 윈도우로 템플릿 빈도 집계, EWM(지수이동) 통계로 스파이크 점수 + 미본 템플릿 비율 계산
- 출력: baseline_scores.parquet (윈도우별 점수/임계값 판정) 및 미리보기 JSON
"""

from __future__ import annotations  # 타입 힌트 전방 참조 허용

from dataclasses import dataclass  # 구성 파라미터 데이터클래스
from pathlib import Path  # 파일 경로 타입
from typing import Dict, List, Tuple  # 타입 힌트

import numpy as np  # 수치 계산
import pandas as pd  # 데이터프레임 처리


@dataclass  # 베이스라인 탐지 파라미터
class BaselineParams:
    window_size: int = 50  # 윈도우 크기(행)
    stride: int = 25  # 윈도우 이동 간격
    ewm_alpha: float = 0.3  # EWM 알파(최근치 가중치)
    anomaly_quantile: float = 0.95  # 이상 임계 백분위수


def _build_windows(df: pd.DataFrame, window_size: int, stride: int) -> List[Tuple[int, pd.DataFrame]]:  # 슬라이딩 윈도우 생성
    rows: List[Tuple[int, pd.DataFrame]] = []  # (시작인덱스, 서브DF) 목록
    for start in range(0, len(df), stride):  # 스트라이드 간격으로 순회
        end = start + window_size  # 종료 인덱스 계산
        window = df.iloc[start:end]  # 부분 데이터프레임 슬라이스
        if window.empty:  # 비어있으면 중단
            break
        rows.append((start, window))  # 결과에 추가
    return rows  # 윈도우 목록 반환


def _counts_per_window(windows: List[Tuple[int, pd.DataFrame]], code_col: str) -> Tuple[pd.DataFrame, List[int], List[Dict[int, int]]]:  # 윈도우별 템플릿 카운트
    row_dicts: List[Dict[str, int]] = []  # 한 윈도우의 {t코드: 카운트}
    starts: List[int] = []  # 각 윈도우 시작 라인
    raw_counts: List[Dict[int, int]] = []  # 원시 카운트(정수 코드 기준)

    # 전체 템플릿 코드 우주 수집  # 누락 컬럼을 0으로 채우기 위해 필요
    all_codes: List[int] = sorted(set(int(x) for _, w in windows for x in w[code_col].dropna().astype(int).tolist()))
    col_names = [f"t{c}" for c in all_codes]  # 컬럼명 생성 (t0, t1, ...)

    for start, w in windows:  # 각 윈도우 순회
        vc = w[code_col].value_counts().astype(int).to_dict()  # 코드별 빈도
        raw_counts.append(vc)  # 원시 카운트 저장
        row = {f"t{int(k)}": int(v) for k, v in vc.items()}  # 행 딕셔너리로 변환
        # 누락 컬럼은 이후 reindex로 채움
        row_dicts.append(row)  # 행 추가
        starts.append(int(w.iloc[0]["line_no"]) if "line_no" in w.columns else int(start))  # 시작 라인 기록

    counts_df = pd.DataFrame(row_dicts)  # 데이터프레임화
    counts_df = counts_df.reindex(columns=col_names, fill_value=0)  # 전체 우주로 재인덱싱(누락 0)
    counts_df.insert(0, "window_start_line", starts)  # 시작 라인 삽입
    return counts_df, starts, raw_counts  # (카운트DF, 시작들, 원시카운트들)


def baseline_detect(parsed_parquet: str | Path, out_dir: str | Path, params: BaselineParams = BaselineParams()) -> Path:  # 베이스라인 이상탐지 실행
    out = Path(out_dir)  # 출력 디렉토리 경로
    out.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성(존재 시 무시)

    df = pd.read_parquet(parsed_parquet)  # 파싱 로그 로드
    df = df.sort_values(["timestamp", "line_no"], kind="stable", na_position="first")  # 시간/라인 정렬

    # 템플릿 ID를 정수 코드로 팩터라이즈
    codes, uniques = pd.factorize(df["template_id"].astype("string"), sort=True)  # 코드/유니크 추출
    df = df.assign(template_code=codes)  # 코드 컬럼 추가

    windows = _build_windows(df, params.window_size, params.stride)  # 윈도우 생성
    if not windows:  # 윈도우가 없으면 에러
        raise ValueError("No windows produced. Consider smaller window_size/stride.")  # 파라미터 조정 권고

    counts_df, starts, raw_counts = _counts_per_window(windows, "template_code")  # 윈도우별 카운트 집계
    count_cols = [c for c in counts_df.columns if c.startswith("t")]  # 템플릿 컬럼 목록

    # 윈도우 축에서 템플릿별 EWM 평균/분산/표준편차와 z-스코어
    ewm_mean = counts_df[count_cols].ewm(alpha=params.ewm_alpha, adjust=False).mean()  # EWM 평균
    ewm_var = counts_df[count_cols].ewm(alpha=params.ewm_alpha, adjust=False).var(bias=False)  # EWM 분산
    ewm_std = (ewm_var.clip(lower=1e-9)) ** 0.5  # 표준편차(수치 안정성 캡)
    z = (counts_df[count_cols] - ewm_mean) / ewm_std  # z-스코어
    z = z.clip(lower=0)  # 양의 스파이크만 고려
    freq_z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).max(axis=1)  # 윈도우 내 최대 스파이크

    # 윈도우별 미본 템플릿 비율 (온라인 집합 기반)
    seen: set[int] = set()  # 지금까지 본 템플릿 코드
    unseen_rates: List[float] = []  # 미본 비율 목록
    for vc in raw_counts:  # 각 윈도우 빈도 딕셔너리
        keys = set(int(k) for k in vc.keys())  # 윈도우 내 코드들
        new = len(keys - seen)  # 새롭게 등장한 코드 수
        denom = max(1, len(keys))  # 분모(0 방지)
        unseen_rates.append(new / denom)  # 비율 계산
        seen |= keys  # 집합 업데이트

    scores = pd.DataFrame({  # 점수 프레임 구성
        "window_start_line": counts_df["window_start_line"].astype(int),  # 시작 라인
        "unseen_rate": unseen_rates,  # 미본 비율
        "freq_z": freq_z.astype(float),  # 스파이크 강도
    })

    # 정규화 및 가중 결합
    # 로버스트 스케일링: freq_z를 95% 분위수로 정규화 캡핑
    freq_cap = max(1e-9, float(np.quantile(scores["freq_z"], 0.95)))  # 캡 값 계산
    scores["freq_norm"] = (scores["freq_z"].clip(0, freq_cap) / freq_cap).astype(float)  # 정규화 스파이크
    scores["unseen_norm"] = scores["unseen_rate"].clip(0, 1).astype(float)  # 0~1 클립
    scores["score"] = 0.6 * scores["freq_norm"] + 0.4 * scores["unseen_norm"]  # 가중 합성 점수

    thr = float(np.quantile(scores["score"], params.anomaly_quantile))  # 임계값(백분위)
    scores["is_anomaly"] = (scores["score"] >= thr).astype(bool)  # 이상 플래그

    out_path = out / "baseline_scores.parquet"  # 출력 경로
    scores.to_parquet(out_path, index=False)  # Parquet 저장

    # 가벼운 미리보기 JSON 저장
    preview = scores.head(20).to_dict(orient="records")  # 상위 20개 샘플
    (out / "baseline_preview.json").write_text(pd.Series(preview).to_json(orient="values"))  # JSON 저장
    return out_path  # 결과 경로 반환
  # 파일 끝

