from __future__ import annotations  # 타입 힌트에서 문자열 리터럴을 앞으로 참조하기 위한 기능 활성화

from pathlib import Path  # 경로 처리를 위한 모듈
import pandas as pd  # 데이터프레임 처리를 위한 라이브러리


def build_mscred_window_counts(
    parsed_parquet: str | Path,  # 파싱된 로그 데이터 Parquet 파일 경로
    out_dir: str | Path,  # 출력 디렉토리 경로
    template_col: str = "template_id",  # 템플릿 컬럼명 (기본값: "template_id")
    window_size: int = 50,  # 윈도우 크기 (기본값: 50)
    stride: int = 25,  # 스트라이드 크기 (기본값: 25, 슬라이딩 윈도우 이동 간격)
) -> None:
    """MSCRED 학습을 위한 윈도우 카운트 데이터 생성 함수.
    
    슬라이딩 윈도우를 사용하여 각 윈도우 내 템플릿별 발생 빈도를 계산하고,
    window_counts.parquet 파일로 저장합니다.
    
    Args:
        parsed_parquet: 파싱된 로그 데이터 Parquet 파일 경로
        out_dir: 출력 디렉토리 경로
        template_col: 템플릿 컬럼명 (기본값: "template_id")
        window_size: 윈도우 크기 (기본값: 50)
        stride: 스트라이드 크기 (기본값: 25, 슬라이딩 윈도우 이동 간격)
    """
    out = Path(out_dir)  # 경로 문자열을 Path 객체로 변환
    out.mkdir(parents=True, exist_ok=True)  # 출력 디렉토리가 없으면 생성
    df = pd.read_parquet(parsed_parquet)  # 파싱된 로그 데이터 읽기
    df = df.sort_values(["timestamp", "line_no"], kind="stable", na_position="first")  # 타임스탬프와 라인 번호로 안정 정렬 (NA값은 앞쪽에)

    # Factorize template ids  # 템플릿 ID를 정수 코드로 변환
    codes, uniques = pd.factorize(df[template_col].astype("string"), sort=True)  # 템플릿 ID를 문자열로 변환 후 정수 인덱스로 매핑 (정렬된 순서로 인코딩)
    df = df.assign(template_code=codes)  # 데이터프레임에 템플릿 코드 컬럼 추가

    rows = []  # 결과 행들을 저장할 리스트 초기화
    for start in range(0, len(df), stride):  # 스트라이드 간격으로 시작 위치를 이동하며 반복
        end = start + window_size  # 윈도우 종료 위치 계산
        window = df.iloc[start:end]  # 현재 윈도우 데이터 추출
        if window.empty:  # 윈도우가 비어있으면
            break  # 반복 중단
        counts = window["template_code"].value_counts().to_dict()  # 윈도우 내 템플릿 코드별 발생 빈도 계산 (딕셔너리로 변환)
        row = {f"t{int(k)}": int(v) for k, v in counts.items()}  # 템플릿 코드를 키로 사용하는 딕셔너리 생성 ("t0", "t1", ... 형식)
        row["start_index"] = int(window.iloc[0]["line_no"]) if "line_no" in window.columns else int(start)  # 시작 인덱스 추가 (라인 번호가 있으면 사용, 없으면 시작 위치)
        rows.append(row)  # 결과 행 리스트에 추가

    out_df = pd.DataFrame(rows)  # 결과 리스트를 DataFrame으로 변환
    out_df.to_parquet(out / "window_counts.parquet", index=False)  # window_counts.parquet 파일로 저장


