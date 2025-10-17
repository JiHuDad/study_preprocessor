#!/usr/bin/env python3
"""
시간 기반 이상 탐지 도구
- 시간대별/요일별 프로파일 학습
- 과거 동일 시간대와 현재 패턴 비교
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
from collections import defaultdict

class TemporalAnomalyDetector:
    def __init__(self):
        self.hourly_profiles = {}
        self.daily_profiles = {}
        self.template_baseline = {}
    
    def build_temporal_profiles(self, parsed_df: pd.DataFrame) -> Dict:
        """시간대별/요일별 정상 프로파일을 구축합니다."""
        
        # timestamp를 datetime으로 변환
        parsed_df['datetime'] = pd.to_datetime(parsed_df['timestamp'])
        parsed_df['hour'] = parsed_df['datetime'].dt.hour
        parsed_df['weekday'] = parsed_df['datetime'].dt.weekday  # 0=Monday
        parsed_df['date'] = parsed_df['datetime'].dt.date
        
        profiles = {
            'hourly': {},
            'daily': {},
            'baseline_period': {
                'start': parsed_df['datetime'].min(),
                'end': parsed_df['datetime'].max(),
                'total_days': (parsed_df['datetime'].max() - parsed_df['datetime'].min()).days + 1
            }
        }
        
        # 시간대별 프로파일 (0-23시)
        for hour in range(24):
            hour_data = parsed_df[parsed_df['hour'] == hour]
            if len(hour_data) > 0:
                template_dist = hour_data['template_id'].value_counts(normalize=True).to_dict()
                profiles['hourly'][hour] = {
                    'volume_mean': len(hour_data) / profiles['baseline_period']['total_days'],
                    'volume_std': len(hour_data) * 0.1,  # 임시 추정치
                    'template_distribution': template_dist,
                    'top_templates': list(hour_data['template_id'].value_counts().head(5).index),
                    'unique_templates': len(hour_data['template_id'].unique())
                }
        
        # 요일별 프로파일 (0=월요일)
        for weekday in range(7):
            day_data = parsed_df[parsed_df['weekday'] == weekday]
            if len(day_data) > 0:
                template_dist = day_data['template_id'].value_counts(normalize=True).to_dict()
                profiles['daily'][weekday] = {
                    'volume_mean': len(day_data) / (profiles['baseline_period']['total_days'] // 7 + 1),
                    'template_distribution': template_dist,
                    'top_templates': list(day_data['template_id'].value_counts().head(10).index),
                    'unique_templates': len(day_data['template_id'].unique())
                }
        
        return profiles
    
    def detect_temporal_anomalies(self, parsed_df: pd.DataFrame, profiles: Dict, 
                                time_window_hours: int = 1) -> List[Dict]:
        """시간대별 프로파일과 비교하여 이상을 탐지합니다."""
        
        parsed_df['datetime'] = pd.to_datetime(parsed_df['timestamp'])
        parsed_df['hour'] = parsed_df['datetime'].dt.hour
        parsed_df['weekday'] = parsed_df['datetime'].dt.weekday
        
        anomalies = []
        
        # 시간별로 그룹화하여 분석
        for hour in sorted(parsed_df['hour'].unique()):
            hour_data = parsed_df[parsed_df['hour'] == hour]
            
            if hour not in profiles['hourly']:
                # 프로파일이 없는 시간대는 모두 이상으로 표시
                anomalies.append({
                    'type': 'temporal_unseen_hour',
                    'hour': hour,
                    'severity': 'high',
                    'description': f'{hour}시는 과거 데이터에 없는 시간대입니다',
                    'actual_volume': len(hour_data),
                    'expected_volume': 'unknown'
                })
                continue
            
            expected_profile = profiles['hourly'][hour]
            actual_volume = len(hour_data)
            expected_volume = expected_profile['volume_mean']
            
            # 볼륨 이상 검사
            volume_deviation = abs(actual_volume - expected_volume) / max(expected_volume, 1)
            if volume_deviation > 0.5:  # 50% 이상 차이
                anomalies.append({
                    'type': 'temporal_volume_anomaly',
                    'hour': hour,
                    'severity': 'high' if volume_deviation > 1.0 else 'medium',
                    'description': f'{hour}시 로그 볼륨이 예상과 {volume_deviation:.1%} 차이',
                    'actual_volume': actual_volume,
                    'expected_volume': expected_volume,
                    'deviation': volume_deviation
                })
            
            # 템플릿 분포 이상 검사
            if len(hour_data) > 0:
                actual_templates = set(hour_data['template_id'].unique())
                expected_templates = set(expected_profile['top_templates'])
                
                # 새로운 템플릿 발견
                new_templates = actual_templates - expected_templates
                if new_templates and len(new_templates) > len(expected_templates) * 0.2:
                    anomalies.append({
                        'type': 'temporal_new_templates',
                        'hour': hour,
                        'severity': 'medium',
                        'description': f'{hour}시에 {len(new_templates)}개의 새로운 템플릿 발견',
                        'new_templates': list(new_templates)[:5],  # 최대 5개만 표시
                        'expected_templates_count': len(expected_templates),
                        'new_templates_count': len(new_templates)
                    })
                
                # 기존 주요 템플릿 누락
                missing_templates = expected_templates - actual_templates
                if missing_templates:
                    anomalies.append({
                        'type': 'temporal_missing_templates',
                        'hour': hour,
                        'severity': 'medium',
                        'description': f'{hour}시에 평소 주요 템플릿 {len(missing_templates)}개 누락',
                        'missing_templates': list(missing_templates),
                        'missing_count': len(missing_templates)
                    })
        
        return anomalies
    
    def generate_temporal_report(self, profiles: Dict, anomalies: List[Dict], 
                               output_path: str) -> str:
        """시간 기반 이상 탐지 리포트를 생성합니다."""
        
        report = f"""# 시간 기반 이상 탐지 리포트

## 📊 기준 프로파일 정보
- **분석 기간**: {profiles['baseline_period']['start']} ~ {profiles['baseline_period']['end']}
- **총 분석 일수**: {profiles['baseline_period']['total_days']}일
- **시간대별 프로파일**: {len(profiles['hourly'])}개
- **요일별 프로파일**: {len(profiles['daily'])}개

## 🚨 발견된 이상 현상

"""
        
        if not anomalies:
            report += "✅ 시간 기반 이상 현상이 발견되지 않았습니다.\n\n"
        else:
            # 심각도별 분류
            high_severity = [a for a in anomalies if a['severity'] == 'high']
            medium_severity = [a for a in anomalies if a['severity'] == 'medium']
            
            if high_severity:
                report += f"### 🔴 심각한 이상 ({len(high_severity)}개)\n\n"
                for anomaly in high_severity:
                    report += f"- **{anomaly['type']}** ({anomaly['hour']}시): {anomaly['description']}\n"
                report += "\n"
            
            if medium_severity:
                report += f"### 🟡 주의 필요 ({len(medium_severity)}개)\n\n"
                for anomaly in medium_severity:
                    report += f"- **{anomaly['type']}** ({anomaly['hour']}시): {anomaly['description']}\n"
                report += "\n"
        
        # 시간대별 요약
        report += "## 📈 시간대별 활동 패턴\n\n"
        if 'hourly' in profiles:
            for hour in sorted(profiles['hourly'].keys()):
                profile = profiles['hourly'][hour]
                report += f"- **{hour:02d}시**: 평균 {profile['volume_mean']:.1f}개 로그, 주요 템플릿 {profile['unique_templates']}개\n"
        
        # 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

def analyze_temporal_patterns(data_dir: str, output_dir: str = None):
    """시간 패턴 기반 이상 탐지를 실행합니다."""
    
    data_path = Path(data_dir)
    if output_dir is None:
        output_dir = data_path / "temporal_analysis"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🕐 시간 기반 이상 탐지 시작...")
    
    # 데이터 로드
    parsed_df = pd.read_parquet(data_path / "parsed.parquet")
    print(f"📊 로드된 로그: {len(parsed_df)}개")
    
    # 탐지기 초기화
    detector = TemporalAnomalyDetector()
    
    # 프로파일 구축
    print("📈 시간대별 프로파일 구축 중...")
    profiles = detector.build_temporal_profiles(parsed_df)
    
    # 프로파일 저장
    with open(output_path / "temporal_profiles.json", 'w') as f:
        # datetime 객체는 JSON serializable하지 않으므로 문자열로 변환
        profiles_copy = profiles.copy()
        profiles_copy['baseline_period']['start'] = str(profiles_copy['baseline_period']['start'])
        profiles_copy['baseline_period']['end'] = str(profiles_copy['baseline_period']['end'])
        json.dump(profiles_copy, f, indent=2, ensure_ascii=False)
    
    # 이상 탐지
    print("🔍 시간 기반 이상 탐지 중...")
    anomalies = detector.detect_temporal_anomalies(parsed_df, profiles)
    
    # 이상 탐지 결과 저장 (numpy types을 Python native types으로 변환)
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    anomalies_serializable = convert_numpy_types(anomalies)
    with open(output_path / "temporal_anomalies.json", 'w') as f:
        json.dump(anomalies_serializable, f, indent=2, ensure_ascii=False)
    
    # 리포트 생성
    print("📄 리포트 생성 중...")
    report = detector.generate_temporal_report(
        profiles, anomalies, 
        output_path / "temporal_report.md"
    )
    
    print(f"✅ 완료! 결과는 {output_path}에 저장되었습니다.")
    print(f"🚨 발견된 이상: {len(anomalies)}개")
    
    # 간단한 요약 출력
    if anomalies:
        high_count = len([a for a in anomalies if a['severity'] == 'high'])
        medium_count = len([a for a in anomalies if a['severity'] == 'medium'])
        print(f"   - 심각: {high_count}개, 주의: {medium_count}개")
    
    return anomalies, profiles

def main():
    parser = argparse.ArgumentParser(description="시간 패턴 기반 로그 이상 탐지")
    parser.add_argument("--data-dir", required=True, 
                       help="분석할 데이터 디렉토리 (parsed.parquet 포함)")
    parser.add_argument("--output-dir", 
                       help="결과 출력 디렉토리 (기본: data-dir/temporal_analysis)")
    
    args = parser.parse_args()
    analyze_temporal_patterns(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
