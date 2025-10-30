#!/usr/bin/env python3  # 실행 스크립트 셔뱅
"""시간 기반 이상 탐지 도구 요약

- 목적: 과거 시간대/요일 패턴을 학습하고 현재 로그와 비교하여 이상 탐지
- 방법: 시간대별 볼륨 편차, 템플릿 분포 변화(신규/누락 템플릿) 검출
- 출력: temporal_profiles.json, temporal_anomalies.json, temporal_report.md
"""  # 모듈 요약 설명
import pandas as pd  # 데이터프레임 처리
import numpy as np  # 수치 계산
import json  # JSON 입출력
from pathlib import Path  # 경로 처리
from datetime import datetime, timedelta  # 시간 처리
from typing import Dict, List, Optional  # 타입 힌트
import argparse  # CLI 인자 파서
from collections import defaultdict  # (미사용) 기본 dict

class TemporalAnomalyDetector:  # 시간 기반 이상탐지기
    def __init__(self):  # 초기화
        self.hourly_profiles = {}  # 시간대별 프로파일 저장소
        self.daily_profiles = {}  # 요일별 프로파일 저장소
        self.template_baseline = {}  # 템플릿 기준치 (예비)
    
    def build_temporal_profiles(self, parsed_df: pd.DataFrame) -> Dict:  # 시간/요일 프로파일 구축
        """시간대별/요일별 정상 프로파일을 구축합니다."""  # API 설명
        
        # timestamp를 datetime으로 변환  # 파생 컬럼 생성
        parsed_df['datetime'] = pd.to_datetime(parsed_df['timestamp'])
        parsed_df['hour'] = parsed_df['datetime'].dt.hour
        parsed_df['weekday'] = parsed_df['datetime'].dt.weekday  # 0=Monday
        parsed_df['date'] = parsed_df['datetime'].dt.date
        
        profiles = {
            'hourly': {},  # 시간대별 프로파일
            'daily': {},  # 요일별 프로파일
            'baseline_period': {
                'start': parsed_df['datetime'].min(),  # 기준 시작
                'end': parsed_df['datetime'].max(),  # 기준 종료
                'total_days': (parsed_df['datetime'].max() - parsed_df['datetime'].min()).days + 1  # 총 일수
            }
        }
        
        # 시간대별 프로파일 (0-23시)
        for hour in range(24):  # 각 시간대 순회
            hour_data = parsed_df[parsed_df['hour'] == hour]
            if len(hour_data) > 0:
                template_dist = hour_data['template_id'].value_counts(normalize=True).to_dict()  # 템플릿 분포
                profiles['hourly'][hour] = {
                    'volume_mean': len(hour_data) / profiles['baseline_period']['total_days'],  # 일 평균 볼륨
                    'volume_std': len(hour_data) * 0.1,  # 임시 표준편차 추정치
                    'template_distribution': template_dist,  # 분포
                    'top_templates': list(hour_data['template_id'].value_counts().head(5).index),  # 상위 템플릿
                    'unique_templates': len(hour_data['template_id'].unique())  # 고유 템플릿 수
                }
        
        # 요일별 프로파일 (0=월요일)
        for weekday in range(7):  # 월~일 순회
            day_data = parsed_df[parsed_df['weekday'] == weekday]
            if len(day_data) > 0:
                template_dist = day_data['template_id'].value_counts(normalize=True).to_dict()  # 템플릿 분포
                profiles['daily'][weekday] = {
                    'volume_mean': len(day_data) / (profiles['baseline_period']['total_days'] // 7 + 1),  # 요일 평균 볼륨
                    'template_distribution': template_dist,  # 분포
                    'top_templates': list(day_data['template_id'].value_counts().head(10).index),  # 상위 템플릿
                    'unique_templates': len(day_data['template_id'].unique())  # 고유 템플릿 수
                }
        
        return profiles  # 프로파일 반환
    
    def detect_temporal_anomalies(self, parsed_df: pd.DataFrame, profiles: Dict, 
                                time_window_hours: int = 1) -> List[Dict]:  # 시간대 비교 이상 탐지
        """시간대별 프로파일과 비교하여 이상을 탐지합니다."""  # API 설명
        
        parsed_df['datetime'] = pd.to_datetime(parsed_df['timestamp'])  # 파생 컬럼 재생성
        parsed_df['hour'] = parsed_df['datetime'].dt.hour
        parsed_df['weekday'] = parsed_df['datetime'].dt.weekday
        
        anomalies = []  # 이상 목록
        
        # 시간별로 그룹화하여 분석
        for hour in sorted(parsed_df['hour'].unique()):  # 관측된 시간대 순회
            hour_data = parsed_df[parsed_df['hour'] == hour]
            
            if hour not in profiles['hourly']:
                # 프로파일이 없는 시간대는 모두 이상으로 표시
                anomalies.append({
                    'type': 'temporal_unseen_hour',  # 미학습 시간대
                    'hour': hour,
                    'severity': 'high',
                    'description': f'{hour}시는 과거 데이터에 없는 시간대입니다',
                    'actual_volume': len(hour_data),
                    'expected_volume': 'unknown'
                })
                continue
            
            expected_profile = profiles['hourly'][hour]  # 기대 프로파일
            actual_volume = len(hour_data)  # 실제 볼륨
            expected_volume = expected_profile['volume_mean']  # 기대 볼륨
            
            # 볼륨 이상 검사  # 상대 편차 계산
            volume_deviation = abs(actual_volume - expected_volume) / max(expected_volume, 1)
            if volume_deviation > 0.5:  # 50% 이상 차이
                anomalies.append({
                    'type': 'temporal_volume_anomaly',  # 볼륨 이상
                    'hour': hour,
                    'severity': 'high' if volume_deviation > 1.0 else 'medium',
                    'description': f'{hour}시 로그 볼륨이 예상과 {volume_deviation:.1%} 차이',
                    'actual_volume': actual_volume,
                    'expected_volume': expected_volume,
                    'deviation': volume_deviation
                })
            
            # 템플릿 분포 이상 검사  # 신규/누락 템플릿 확인
            if len(hour_data) > 0:
                actual_templates = set(hour_data['template_id'].unique())
                expected_templates = set(expected_profile['top_templates'])
                
                # 새로운 템플릿 발견
                new_templates = actual_templates - expected_templates
                if new_templates and len(new_templates) > len(expected_templates) * 0.2:
                    anomalies.append({
                        'type': 'temporal_new_templates',  # 신규 템플릿 증가
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
                        'type': 'temporal_missing_templates',  # 주요 템플릿 누락
                        'hour': hour,
                        'severity': 'medium',
                        'description': f'{hour}시에 평소 주요 템플릿 {len(missing_templates)}개 누락',
                        'missing_templates': list(missing_templates),
                        'missing_count': len(missing_templates)
                    })
        
        return anomalies  # 이상 목록 반환
    
    def generate_temporal_report(self, profiles: Dict, anomalies: List[Dict], 
                               output_path: str) -> str:  # 리포트 생성
        """시간 기반 이상 탐지 리포트를 생성합니다."""  # API 설명
        
        report = f"""# 시간 기반 이상 탐지 리포트

## 📊 기준 프로파일 정보
- **분석 기간**: {profiles['baseline_period']['start']} ~ {profiles['baseline_period']['end']}
- **총 분석 일수**: {profiles['baseline_period']['total_days']}일
- **시간대별 프로파일**: {len(profiles['hourly'])}개
- **요일별 프로파일**: {len(profiles['daily'])}개

## 🚨 발견된 이상 현상

"""
        
        if not anomalies:  # 이상 없음
            report += "✅ 시간 기반 이상 현상이 발견되지 않았습니다.\n\n"
        else:
            # 심각도별 분류  # high/medium 그룹
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
        
        # 시간대별 요약  # 각 시간대 핵심 정보
        report += "## 📈 시간대별 활동 패턴\n\n"
        if 'hourly' in profiles:
            for hour in sorted(profiles['hourly'].keys()):
                profile = profiles['hourly'][hour]
                report += f"- **{hour:02d}시**: 평균 {profile['volume_mean']:.1f}개 로그, 주요 템플릿 {profile['unique_templates']}개\n"
        
        # 파일로 저장  # 마크다운 리포트 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report  # 리포트 문자열 반환

def analyze_temporal_patterns(data_dir: str, output_dir: str = None):  # 시간 기반 탐지 실행
    """시간 패턴 기반 이상 탐지를 실행합니다."""  # API 설명
    
    data_path = Path(data_dir)  # 데이터 폴더
    if output_dir is None:
        output_dir = data_path / "temporal_analysis"  # 기본 출력 폴더
    output_path = Path(output_dir)  # Path 화
    output_path.mkdir(parents=True, exist_ok=True)  # 폴더 생성
    
    print("🕐 시간 기반 이상 탐지 시작...")  # 시작 로그
    
    # 데이터 로드
    parsed_df = pd.read_parquet(data_path / "parsed.parquet")  # 파싱 로그 로드
    print(f"📊 로드된 로그: {len(parsed_df)}개")
    
    # 탐지기 초기화
    detector = TemporalAnomalyDetector()  # 인스턴스 생성
    
    # 프로파일 구축
    print("📈 시간대별 프로파일 구축 중...")
    profiles = detector.build_temporal_profiles(parsed_df)  # 프로파일 생성
    
    # 프로파일 저장
    with open(output_path / "temporal_profiles.json", 'w') as f:
        # datetime 객체는 JSON serializable하지 않으므로 문자열로 변환
        profiles_copy = profiles.copy()
        profiles_copy['baseline_period']['start'] = str(profiles_copy['baseline_period']['start'])
        profiles_copy['baseline_period']['end'] = str(profiles_copy['baseline_period']['end'])
        json.dump(profiles_copy, f, indent=2, ensure_ascii=False)
    
    # 이상 탐지
    print("🔍 시간 기반 이상 탐지 중...")
    anomalies = detector.detect_temporal_anomalies(parsed_df, profiles)  # 이상 탐지
    
    # 이상 탐지 결과 저장 (numpy types을 Python native types으로 변환)  # 직렬화 보정
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
    
    # 간단한 요약 출력  # 심각도 카운트
    if anomalies:
        high_count = len([a for a in anomalies if a['severity'] == 'high'])
        medium_count = len([a for a in anomalies if a['severity'] == 'medium'])
        print(f"   - 심각: {high_count}개, 주의: {medium_count}개")
    
    return anomalies, profiles  # 결과 반환

def main():  # CLI 엔트리 포인트
    parser = argparse.ArgumentParser(description="시간 패턴 기반 로그 이상 탐지")  # 인자 파서
    parser.add_argument("--data-dir", required=True, 
                       help="분석할 데이터 디렉토리 (parsed.parquet 포함)")  # 데이터 폴더
    parser.add_argument("--output-dir", 
                       help="결과 출력 디렉토리 (기본: data-dir/temporal_analysis)")  # 출력 폴더
    
    args = parser.parse_args()  # 파싱
    analyze_temporal_patterns(args.data_dir, args.output_dir)  # 실행

if __name__ == "__main__":  # 직접 실행 시
    main()  # 메인 호출
