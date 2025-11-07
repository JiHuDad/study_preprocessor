from __future__ import annotations  # 타입 힌트에서 문자열 리터럴을 앞으로 참조하기 위한 기능 활성화

from pathlib import Path  # 경로 처리를 위한 모듈
import json  # JSON 데이터 처리를 위한 모듈
import pandas as pd  # 데이터프레임 처리를 위한 라이브러리


def build_mscred_window_counts(
    parsed_parquet: str | Path,  # 파싱된 로그 데이터 Parquet 파일 경로
    out_dir: str | Path,  # 출력 디렉토리 경로
    template_col: str = "template_id",  # 템플릿 컬럼명 (기본값: "template_id")
    window_size: int = 50,  # 윈도우 크기 (기본값: 50)
    stride: int = 25,  # 스트라이드 크기 (기본값: 25, 슬라이딩 윈도우 이동 간격)
    template_mapping_path: str | Path | None = None,  # 템플릿 매핑 파일 경로 (선택사항)
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
        template_mapping_path: 기존 template_mapping.json 경로 (선택사항)
                              - 제공되면: 기존 매핑 사용 (추론 시 사용)
                              - None: 새로운 매핑 생성 (학습 시 사용)

    Important:
        추론 시에는 반드시 학습 시 생성된 template_mapping.json을 전달해야 합니다.
        그렇지 않으면 템플릿 인덱스가 달라져 재구성 오차가 완전히 잘못됩니다!
    """
    out = Path(out_dir)  # 경로 문자열을 Path 객체로 변환
    out.mkdir(parents=True, exist_ok=True)  # 출력 디렉토리가 없으면 생성
    df = pd.read_parquet(parsed_parquet)  # 파싱된 로그 데이터 읽기
    df = df.sort_values(["timestamp", "line_no"], kind="stable", na_position="first")  # 타임스탬프와 라인 번호로 안정 정렬 (NA값은 앞쪽에)

    # Template mapping 로드 또는 생성
    if template_mapping_path and Path(template_mapping_path).exists():
        # 기존 매핑 사용 (추론 시)
        with open(template_mapping_path, 'r') as f:
            template_mapping = json.load(f)  # {"template_id": code, ...}
        print(f"✅ 기존 template_mapping 사용: {template_mapping_path} (크기: {len(template_mapping)})")

        # template_id를 매핑된 코드로 변환
        df[template_col] = df[template_col].astype("string")
        codes = df[template_col].map(template_mapping).fillna(-1).astype(int)
        df = df.assign(template_code=codes)  # 데이터프레임에 템플릿 코드 컬럼 추가
    else:
        # 새로운 매핑 생성 (학습 시)
        # Factorize template ids  # 템플릿 ID를 정수 코드로 변환
        codes, uniques = pd.factorize(df[template_col].astype("string"), sort=True)  # 템플릿 ID를 문자열로 변환 후 정수 인덱스로 매핑 (정렬된 순서로 인코딩)
        df = df.assign(template_code=codes)  # 데이터프레임에 템플릿 코드 컬럼 추가

        # 매핑 저장
        template_mapping = {str(template): int(code) for code, template in enumerate(uniques)}
        mapping_path = out / "template_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(template_mapping, f, indent=2)
        print(f"✅ 새로운 template_mapping 생성: {mapping_path} (크기: {len(template_mapping)})")

    # template_mapping이 있으면 모든 템플릿 코드 준비 (추론 시 일관성 보장)
    if template_mapping_path and Path(template_mapping_path).exists():
        # 학습 시 정의된 모든 template_code
        all_template_codes = set(template_mapping.values())
    else:
        # 학습 시에는 현재 데이터의 코드만 사용
        all_template_codes = None

    rows = []  # 결과 행들을 저장할 리스트 초기화
    for start in range(0, len(df), stride):  # 스트라이드 간격으로 시작 위치를 이동하며 반복
        end = start + window_size  # 윈도우 종료 위치 계산
        window = df.iloc[start:end]  # 현재 윈도우 데이터 추출
        if window.empty:  # 윈도우가 비어있으면
            break  # 반복 중단

        # 윈도우 내 템플릿 코드별 발생 빈도 계산
        counts = window["template_code"].value_counts().to_dict()  # 윈도우 내 템플릿 코드별 발생 빈도 계산 (딕셔너리로 변환)

        # 기본 행 생성
        row = {f"t{int(k)}": int(v) for k, v in counts.items() if int(k) >= 0}  # 템플릿 코드를 키로 사용하는 딕셔너리 생성 ("t0", "t1", ... 형식), -1은 제외

        # template_mapping이 있으면 모든 채널을 0으로 초기화 (없는 것은 0)
        if all_template_codes is not None:
            for code in all_template_codes:
                col_name = f"t{int(code)}"
                if col_name not in row:
                    row[col_name] = 0  # 없는 템플릿 코드는 0으로 설정

        row["start_index"] = int(window.iloc[0]["line_no"]) if "line_no" in window.columns else int(start)  # 시작 인덱스 추가 (라인 번호가 있으면 사용, 없으면 시작 위치)
        rows.append(row)  # 결과 행 리스트에 추가

    out_df = pd.DataFrame(rows)  # 결과 리스트를 DataFrame으로 변환

    # 컬럼 정렬 (t0, t1, t2, ... 순서로)
    template_columns = sorted([c for c in out_df.columns if c.startswith('t') and c[1:].isdigit()],
                             key=lambda x: int(x[1:]))
    other_columns = [c for c in out_df.columns if not (c.startswith('t') and c[1:].isdigit())]
    out_df = out_df[template_columns + other_columns]

    # NaN을 0으로 채움 (없는 템플릿)
    out_df = out_df.fillna(0)

    out_df.to_parquet(out / "window_counts.parquet", index=False)  # window_counts.parquet 파일로 저장


