#!/usr/bin/env python3
"""
결과 분석 및 해석 도구
"""
import pandas as pd
import json
from pathlib import Path
import argparse

def analyze_results(data_dir: str = "data/processed"):
    """pipeline 실행 결과를 분석하고 해석합니다."""
    data_path = Path(data_dir)
    
    print("="*60)
    print("🔍 LOG ANOMALY DETECTION 결과 분석")
    print("="*60)
    
    # 1. 기본 정보 읽기 (안전한 처리)
    vocab = {}
    preview = {}
    
    # vocab.json 파일 확인
    vocab_file = data_path / "vocab.json"
    if vocab_file.exists():
        with open(vocab_file) as f:
            vocab = json.load(f)
    else:
        print("⚠️ vocab.json 파일이 없습니다. DeepLog 분석이 실행되지 않았을 수 있습니다.")
    
    # preview.json 파일 확인
    preview_file = data_path / "preview.json"
    if preview_file.exists():
        with open(preview_file) as f:
            preview = json.load(f)
    else:
        print("⚠️ preview.json 파일이 없습니다.")
    
    # 2. 파싱된 로그 데이터
    parsed_file = data_path / "parsed.parquet"
    if not parsed_file.exists():
        print("❌ parsed.parquet 파일이 없습니다. 전처리가 완료되지 않았습니다.")
        return
    
    parsed_df = pd.read_parquet(parsed_file)
    
    # sequences.parquet 파일 확인
    sequences_file = data_path / "sequences.parquet"
    if sequences_file.exists():
        sequences_df = pd.read_parquet(sequences_file)
    else:
        print("⚠️ sequences.parquet 파일이 없습니다. DeepLog 분석이 실행되지 않았을 수 있습니다.")
        sequences_df = None
    
    print(f"\n📊 **기본 통계**")
    print(f"- 총 로그 라인 수: {len(parsed_df)}")
    
    # 템플릿 수 계산
    if vocab:
        template_count = len(vocab)
    else:
        # vocab이 없으면 파싱된 데이터에서 직접 계산
        template_count = len(parsed_df['template_id'].unique()) if 'template_id' in parsed_df.columns else 0
    
    print(f"- 발견된 템플릿 수: {template_count}")
    
    # 안전한 timestamp 처리
    try:
        min_time = parsed_df['timestamp'].min()
        max_time = parsed_df['timestamp'].max()
        if pd.isna(min_time) or pd.isna(max_time):
            print(f"- 분석 기간: 시간 정보 없음")
        else:
            print(f"- 분석 기간: {min_time} ~ {max_time}")
    except Exception as e:
        print(f"- 분석 기간: 시간 정보 처리 오류 ({e})")
    
    # 안전한 host 처리
    try:
        hosts = parsed_df['host'].dropna().unique()
        if len(hosts) > 0:
            print(f"- 호스트: {', '.join(str(h) for h in hosts if h is not None)}")
        else:
            print(f"- 호스트: 정보 없음")
    except Exception as e:
        print(f"- 호스트: 정보 처리 오류 ({e})")
    
    # 3. 템플릿 분석
    print(f"\n🔧 **로그 템플릿 분석**")
    template_counts = parsed_df['template_id'].value_counts()
    for template_id, count in template_counts.head().items():
        template = parsed_df[parsed_df['template_id'] == template_id]['template'].iloc[0]
        print(f"- Template {template_id} (출현 {count}회): {template[:80]}...")
    
    # 4. Baseline Anomaly Detection 결과
    baseline_file = data_path / "baseline_scores.parquet"
    if baseline_file.exists():
        baseline_scores = pd.read_parquet(baseline_file)
        print(f"\n🚨 **Baseline Anomaly Detection 결과**")
        print(f"- 분석된 윈도우 수: {len(baseline_scores)}")
        anomalies = baseline_scores[baseline_scores['is_anomaly'] == True]
        print(f"- 발견된 이상 윈도우: {len(anomalies)}개 (전체의 {len(anomalies)/len(baseline_scores)*100:.1f}%)")
    else:
        print(f"\n⚠️ **Baseline Anomaly Detection 결과가 없습니다**")
        baseline_scores = None
        anomalies = None
    
    if anomalies is not None and len(anomalies) > 0:
        print("\n🔍 **이상 윈도우 상세**:")
        for _, row in anomalies.iterrows():
            start_line = int(row['window_start_line'])
            score = row['score']
            unseen_rate = row['unseen_rate']
            print(f"  - 윈도우 시작라인 {start_line}: 점수={score:.3f}, 미지템플릿비율={unseen_rate:.3f}")
            
            # 해당 윈도우의 로그 내용 표시
            window_logs = parsed_df[parsed_df['line_no'] >= start_line].head(3)  # 윈도우 크기 가정
            for _, log in window_logs.iterrows():
                print(f"    Line {log['line_no']}: {log['raw'][:60]}...")
    
    # 5. DeepLog 결과
    deeplog_file = data_path / "deeplog_infer.parquet"
    if deeplog_file.exists():
        deeplog_df = pd.read_parquet(deeplog_file)
        print(f"\n🧠 **DeepLog 딥러닝 모델 결과**")
        violations = deeplog_df[deeplog_df['in_topk'] == False]
        print(f"- 예측 실패 (위반): {len(violations)}개 / {len(deeplog_df)}개")
    else:
        print(f"\n⚠️ **DeepLog 결과가 없습니다**")
        deeplog_df = None
        violations = None
    if deeplog_df is not None and len(deeplog_df) > 0:
        print(f"- 위반율: {len(violations)/len(deeplog_df)*100:.1f}%")
    else:
        print("- 위반율: N/A (추론 결과 없음)")
    
    if violations is not None and len(violations) > 0:
        print("\n🔍 **DeepLog 위반 상세**:")
        for _, row in violations.iterrows():
            idx = int(row['idx'])
            target = int(row['target'])
            # 실제 로그 라인 찾기
            if idx < len(parsed_df):
                log_line = parsed_df.iloc[idx]
                target_template = vocab.get(str(target), f"Unknown({target})") if vocab else f"Template({target})"
                print(f"  - Line {log_line['line_no']}: 예상 템플릿 {target}번이 top-k에 없음")
                print(f"    실제 로그: {log_line['raw'][:60]}...")
    
    # 6. 종합 해석
    print(f"\n📈 **종합 해석**")
    
    # 안전한 비율 계산
    if baseline_scores is not None and anomalies is not None:
        baseline_anomaly_rate = len(anomalies) / len(baseline_scores) * 100
        if baseline_anomaly_rate > 20:
            print("⚠️  Baseline 모델에서 높은 이상률 감지 - 시스템에 비정상적인 패턴이 많음")
        elif baseline_anomaly_rate > 5:
            print("🔍 Baseline 모델에서 중간 수준의 이상 감지 - 일부 비정상 패턴 존재")
        else:
            print("✅ Baseline 모델에서 낮은 이상률 - 대부분 정상적인 로그 패턴")
    else:
        print("ℹ️  Baseline 분석 결과가 없어 이상률을 계산할 수 없습니다")
    
    if deeplog_df is not None and violations is not None and len(deeplog_df) > 0:
        deeplog_violation_rate = len(violations) / len(deeplog_df) * 100
        if deeplog_violation_rate > 50:
            print("⚠️  DeepLog 모델에서 높은 위반율 - 로그 시퀀스가 예측하기 어려운 패턴")
        elif deeplog_violation_rate > 20:
            print("🔍 DeepLog 모델에서 중간 수준의 위반율 - 일부 예측 어려운 시퀀스")
        else:
            print("✅ DeepLog 모델에서 낮은 위반율 - 대부분 예측 가능한 로그 시퀀스")
    else:
        print("ℹ️  DeepLog 분석 결과가 없어 위반율을 계산할 수 없습니다")
    
    # 7. 리포트 파일이 있다면 표시
    report_file = data_path / "report.md"
    if report_file.exists():
        print(f"\n📄 **자동 생성된 리포트**")
        with open(report_file) as f:
            print(f.read())

def main():
    parser = argparse.ArgumentParser(description="Log Anomaly Detection 결과 분석")
    parser.add_argument("--data-dir", default="data/processed", 
                       help="결과 데이터 디렉토리 (기본: data/processed)")
    
    args = parser.parse_args()
    analyze_results(args.data_dir)

if __name__ == "__main__":
    main()
