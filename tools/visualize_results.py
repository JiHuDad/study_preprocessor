#!/usr/bin/env python3
"""
결과 시각화 도구 (matplotlib/seaborn 없이)
"""
import pandas as pd
import json
from pathlib import Path
import argparse

def print_chart(title, data, max_width=50):
    """간단한 텍스트 차트를 출력합니다."""
    print(f"\n📊 {title}")
    print("="*60)
    
    if not data:
        print("데이터가 없습니다.")
        return
    
    max_val = max(data.values()) if data.values() else 0
    if max_val == 0:
        print("모든 값이 0입니다.")
        return
    
    for label, value in data.items():
        bar_length = int((value / max_val) * max_width)
        bar = "█" * bar_length
        print(f"{label:20} |{bar:<{max_width}} {value}")

def visualize_results(data_dir: str = "data/processed"):
    """결과를 시각화합니다."""
    data_path = Path(data_dir)
    
    print("="*60)
    print("📈 LOG ANOMALY DETECTION 시각화")
    print("="*60)
    
    # 기본 데이터 로드
    parsed_df = pd.read_parquet(data_path / "parsed.parquet")
    baseline_scores = pd.read_parquet(data_path / "baseline_scores.parquet")
    deeplog_df = pd.read_parquet(data_path / "deeplog_infer.parquet")
    
    with open(data_path / "vocab.json") as f:
        vocab = json.load(f)
    
    # 1. 템플릿 분포
    template_counts = parsed_df['template_id'].value_counts().head(10)
    template_data = {}
    for template_id, count in template_counts.items():
        template_data[f"T{template_id}"] = count
    
    print_chart("템플릿 출현 빈도 (Top 10)", template_data)
    
    # 2. 시간대별 로그 분포
    try:
        parsed_df['hour'] = pd.to_datetime(parsed_df['timestamp']).dt.hour
        hourly_counts = parsed_df['hour'].value_counts().sort_index()
        hourly_data = {f"{h:02d}시": count for h, count in hourly_counts.items()}
    except Exception as e:
        print(f"시간대별 분포 처리 오류: {e}")
        hourly_data = {"시간정보없음": len(parsed_df)}
    
    print_chart("시간대별 로그 분포", hourly_data)
    
    # 3. Baseline 이상 점수 분포
    score_bins = pd.cut(baseline_scores['score'], bins=5, labels=['매우낮음', '낮음', '보통', '높음', '매우높음'])
    score_data = score_bins.value_counts().to_dict()
    
    print_chart("Baseline 이상 점수 분포", score_data)
    
    # 4. 이상 윈도우 vs 정상 윈도우
    anomaly_data = {
        "정상 윈도우": len(baseline_scores[baseline_scores['is_anomaly'] == False]),
        "이상 윈도우": len(baseline_scores[baseline_scores['is_anomaly'] == True])
    }
    
    print_chart("이상 탐지 결과", anomaly_data)
    
    # 5. DeepLog 성능
    # Enhanced 버전: prediction_ok 사용, 기존 버전: in_topk 사용
    if 'prediction_ok' in deeplog_df.columns:
        success_count = len(deeplog_df[deeplog_df['prediction_ok'] == True])
        failure_count = len(deeplog_df[deeplog_df['prediction_ok'] == False])
    elif 'in_topk' in deeplog_df.columns:
        success_count = len(deeplog_df[deeplog_df['in_topk'] == True])
        failure_count = len(deeplog_df[deeplog_df['in_topk'] == False])
    else:
        success_count = 0
        failure_count = 0

    deeplog_data = {
        "예측 성공": success_count,
        "예측 실패": failure_count
    }

    print_chart("DeepLog 예측 성능", deeplog_data)

    # 6. 상세 통계
    print(f"\n📊 상세 통계")
    print("="*60)
    print(f"총 로그 라인 수: {len(parsed_df):,}")
    print(f"발견된 템플릿 수: {len(vocab)}")
    print(f"분석 윈도우 수: {len(baseline_scores)}")
    print(f"이상률: {len(baseline_scores[baseline_scores['is_anomaly']==True])/len(baseline_scores)*100:.1f}%")
    if len(deeplog_df) > 0:
        violation_rate = failure_count / len(deeplog_df) * 100
        print(f"DeepLog 위반율: {violation_rate:.1f}%")
    else:
        print("DeepLog 위반율: N/A (추론 결과 없음)")
    print(f"평균 이상 점수: {baseline_scores['score'].mean():.3f}")
    print(f"최대 이상 점수: {baseline_scores['score'].max():.3f}")
    
    # 7. 이상 윈도우 상세 분석
    anomalies = baseline_scores[baseline_scores['is_anomaly'] == True]
    if len(anomalies) > 0:
        print(f"\n🚨 이상 윈도우 상세 분석")
        print("="*60)
        
        print("이상 점수별 분포:")
        for _, row in anomalies.sort_values('score', ascending=False).head(5).iterrows():
            start_line = int(row['window_start_line'])
            score = row['score']
            unseen_rate = row['unseen_rate']
            print(f"  라인 {start_line:3d}: 점수={score:.3f}, 미지율={unseen_rate:.3f}")
    
    # 8. 템플릿별 상세 정보
    print(f"\n🔧 주요 템플릿 정보")
    print("="*60)
    
    for template_id, count in template_counts.head(5).items():
        template = parsed_df[parsed_df['template_id'] == template_id]['template'].iloc[0]
        print(f"Template {template_id} ({count}회):")
        print(f"  {template[:70]}...")
        print()

def create_summary_report(data_dir: str = "data/processed"):
    """간단한 요약 리포트를 생성합니다."""
    data_path = Path(data_dir)
    
    parsed_df = pd.read_parquet(data_path / "parsed.parquet")
    baseline_scores = pd.read_parquet(data_path / "baseline_scores.parquet")
    deeplog_df = pd.read_parquet(data_path / "deeplog_infer.parquet")
    
    anomaly_rate = len(baseline_scores[baseline_scores['is_anomaly']==True])/len(baseline_scores)*100

    # Enhanced 버전: prediction_ok 사용, 기존 버전: in_topk 사용
    if len(deeplog_df) > 0:
        if 'prediction_ok' in deeplog_df.columns:
            violation_rate = len(deeplog_df[deeplog_df['prediction_ok']==False])/len(deeplog_df)*100
        elif 'in_topk' in deeplog_df.columns:
            violation_rate = len(deeplog_df[deeplog_df['in_topk']==False])/len(deeplog_df)*100
        else:
            violation_rate = 0
    else:
        violation_rate = 0
    
    # 안전한 시간 정보 처리
    try:
        min_time = parsed_df['timestamp'].min()
        max_time = parsed_df['timestamp'].max()
        if pd.isna(min_time) or pd.isna(max_time):
            time_range = "시간 정보 없음"
        else:
            time_range = f"{min_time} ~ {max_time}"
    except Exception:
        time_range = "시간 정보 처리 오류"
    
    report = f"""
# 로그 이상 탐지 결과 요약

## 기본 정보
- 분석 데이터: {data_dir}
- 총 로그 라인: {len(parsed_df):,}개
- 분석 기간: {time_range}

## 핵심 지표
- **이상률**: {anomaly_rate:.1f}% ({len(baseline_scores[baseline_scores['is_anomaly']==True])}개 / {len(baseline_scores)}개 윈도우)
- **DeepLog 위반율**: {violation_rate:.1f}% ({int(violation_rate * len(deeplog_df) / 100) if len(deeplog_df) > 0 else 0}개 / {len(deeplog_df)}개 시퀀스)

## 해석
"""
    
    if anomaly_rate < 5:
        report += "✅ 낮은 이상률 - 시스템이 정상적으로 작동\n"
    elif anomaly_rate < 20:
        report += "🔍 중간 수준의 이상률 - 일부 비정상 패턴 존재\n"
    else:
        report += "⚠️ 높은 이상률 - 시스템 점검 필요\n"
    
    if violation_rate < 20:
        report += "✅ 낮은 위반율 - 예측 가능한 로그 패턴\n"
    elif violation_rate < 50:
        report += "🔍 중간 수준의 위반율 - 일부 복잡한 패턴\n"
    else:
        report += "⚠️ 높은 위반율 - 매우 복잡하거나 비정상적인 패턴\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Log Anomaly Detection 결과 시각화")
    parser.add_argument("--data-dir", default="data/processed", 
                       help="결과 데이터 디렉토리 (기본: data/processed)")
    parser.add_argument("--summary", action="store_true",
                       help="요약 리포트만 출력")
    
    args = parser.parse_args()
    
    if args.summary:
        print(create_summary_report(args.data_dir))
    else:
        visualize_results(args.data_dir)

if __name__ == "__main__":
    main()
