#!/usr/bin/env python3
"""
파일별 비교 이상 탐지 도구
- 여러 로그 파일 간 패턴 비교
- Baseline 파일들과 Target 파일 비교
- 서버/서비스별 차이점 분석
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import Counter

class ComparativeAnomalyDetector:
    def __init__(self):
        self.baseline_profiles = {}
        self.comparison_threshold = 0.3  # 30% 이상 차이는 이상으로 간주
    
    def extract_file_profile(self, parsed_df: pd.DataFrame, file_name: str) -> Dict:
        """단일 파일의 프로파일을 추출합니다."""
        
        if len(parsed_df) == 0:
            return {
                'file_name': file_name,
                'total_logs': 0,
                'unique_templates': 0,
                'template_distribution': {},
                'error': 'Empty dataset'
            }
        
        # 기본 통계
        template_counts = parsed_df['template_id'].value_counts()
        template_dist = (template_counts / len(parsed_df)).to_dict()
        
        # 호스트별 분포
        host_dist = parsed_df['host'].value_counts(normalize=True).to_dict() if 'host' in parsed_df.columns else {}
        
        # 프로세스별 분포
        process_dist = parsed_df['process'].value_counts(normalize=True).to_dict() if 'process' in parsed_df.columns else {}
        
        # 시간 범위
        time_range = {
            'start': str(parsed_df['timestamp'].min()) if 'timestamp' in parsed_df.columns else None,
            'end': str(parsed_df['timestamp'].max()) if 'timestamp' in parsed_df.columns else None,
            'duration_hours': None
        }
        
        if time_range['start'] and time_range['end']:
            start_dt = pd.to_datetime(time_range['start'])
            end_dt = pd.to_datetime(time_range['end'])
            time_range['duration_hours'] = (end_dt - start_dt).total_seconds() / 3600
        
        # 에러/경고 로그 비율 (키워드 기반 추정)
        error_keywords = ['error', 'ERROR', 'fail', 'FAIL', 'exception', 'Exception', 'panic', 'fatal']
        warning_keywords = ['warn', 'WARN', 'warning', 'WARNING']
        
        error_logs = parsed_df[parsed_df['raw'].str.contains('|'.join(error_keywords), case=False, na=False)]
        warning_logs = parsed_df[parsed_df['raw'].str.contains('|'.join(warning_keywords), case=False, na=False)]
        
        profile = {
            'file_name': file_name,
            'total_logs': len(parsed_df),
            'unique_templates': len(template_counts),
            'template_distribution': template_dist,
            'top_templates': list(template_counts.head(10).index),
            'host_distribution': host_dist,
            'process_distribution': process_dist,
            'time_range': time_range,
            'error_rate': len(error_logs) / len(parsed_df),
            'warning_rate': len(warning_logs) / len(parsed_df),
            'logs_per_hour': len(parsed_df) / max(time_range['duration_hours'], 1) if time_range['duration_hours'] else None,
            'template_entropy': self._calculate_entropy(template_dist),
            'rare_templates_count': len([t for t, count in template_counts.items() if count == 1])
        }
        
        return profile
    
    def _calculate_entropy(self, distribution: Dict) -> float:
        """분포의 엔트로피를 계산합니다."""
        if not distribution:
            return 0.0
        
        probs = list(distribution.values())
        probs = [p for p in probs if p > 0]  # 0 확률 제거
        
        if not probs:
            return 0.0
        
        return -sum(p * np.log2(p) for p in probs)
    
    def compare_distributions(self, target_dist: Dict, baseline_dist: Dict) -> Dict:
        """두 분포 간의 차이를 분석합니다."""
        
        # KL Divergence 계산 (분포 간 차이 측정)
        def kl_divergence(p, q):
            # 작은 값으로 smoothing
            epsilon = 1e-10
            all_keys = set(p.keys()) | set(q.keys())
            
            p_smooth = {k: p.get(k, 0) + epsilon for k in all_keys}
            q_smooth = {k: q.get(k, 0) + epsilon for k in all_keys}
            
            # 정규화
            p_sum = sum(p_smooth.values())
            q_sum = sum(q_smooth.values())
            p_norm = {k: v/p_sum for k, v in p_smooth.items()}
            q_norm = {k: v/q_sum for k, v in q_smooth.items()}
            
            return sum(p_norm[k] * np.log(p_norm[k] / q_norm[k]) for k in all_keys)
        
        kl_div = kl_divergence(target_dist, baseline_dist)
        
        # 공통 템플릿과 유니크 템플릿 분석
        target_templates = set(target_dist.keys())
        baseline_templates = set(baseline_dist.keys())
        
        common_templates = target_templates & baseline_templates
        target_only = target_templates - baseline_templates
        baseline_only = baseline_templates - target_templates
        
        # 상위 템플릿 차이 분석
        def get_top_templates(dist, n=5):
            return sorted(dist.items(), key=lambda x: x[1], reverse=True)[:n]
        
        target_top = get_top_templates(target_dist)
        baseline_top = get_top_templates(baseline_dist)
        
        return {
            'kl_divergence': kl_div,
            'common_templates': len(common_templates),
            'target_only_templates': len(target_only),
            'baseline_only_templates': len(baseline_only),
            'jaccard_similarity': len(common_templates) / len(target_templates | baseline_templates) if target_templates | baseline_templates else 0,
            'target_top_templates': target_top,
            'baseline_top_templates': baseline_top,
            'unique_in_target': list(target_only)[:10],  # 최대 10개
            'missing_from_target': list(baseline_only)[:10]  # 최대 10개
        }
    
    def detect_comparative_anomalies(self, target_profile: Dict, 
                                   baseline_profiles: List[Dict]) -> List[Dict]:
        """Target 파일과 Baseline 파일들을 비교하여 이상을 탐지합니다."""
        
        anomalies = []
        
        # Baseline 평균 계산
        if not baseline_profiles:
            return [{
                'type': 'no_baseline',
                'severity': 'high',
                'description': 'Baseline 파일이 없어 비교할 수 없습니다'
            }]
        
        # 수치 지표들의 평균과 표준편차 계산
        metrics = ['total_logs', 'unique_templates', 'error_rate', 'warning_rate', 
                  'template_entropy', 'rare_templates_count']
        
        baseline_stats = {}
        for metric in metrics:
            values = [p.get(metric, 0) for p in baseline_profiles if p.get(metric) is not None]
            if values:
                baseline_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 각 지표별 이상 검사
        for metric, stats in baseline_stats.items():
            target_value = target_profile.get(metric, 0)
            if target_value is None:
                continue
                
            # Z-score 계산
            z_score = abs(target_value - stats['mean']) / max(stats['std'], 1e-6)
            
            if z_score > 2.0:  # 2σ 이상 차이
                deviation_pct = abs(target_value - stats['mean']) / max(stats['mean'], 1e-6)
                anomalies.append({
                    'type': f'metric_anomaly_{metric}',
                    'severity': 'high' if z_score > 3.0 else 'medium',
                    'description': f'{metric}이 baseline 대비 {deviation_pct:.1%} 차이 (Z-score: {z_score:.2f})',
                    'target_value': target_value,
                    'baseline_mean': stats['mean'],
                    'baseline_std': stats['std'],
                    'z_score': z_score
                })
        
        # 템플릿 분포 비교
        if baseline_profiles and 'template_distribution' in target_profile:
            # 모든 baseline의 템플릿 분포를 평균화
            baseline_combined = {}
            for profile in baseline_profiles:
                for template, prob in profile.get('template_distribution', {}).items():
                    baseline_combined[template] = baseline_combined.get(template, 0) + prob
            
            # 평균화
            baseline_count = len(baseline_profiles)
            baseline_avg_dist = {k: v/baseline_count for k, v in baseline_combined.items()}
            
            # 분포 비교
            comparison = self.compare_distributions(
                target_profile['template_distribution'], 
                baseline_avg_dist
            )
            
            # KL Divergence가 높으면 이상
            if comparison['kl_divergence'] > 1.0:  # 임계값
                anomalies.append({
                    'type': 'template_distribution_anomaly',
                    'severity': 'high' if comparison['kl_divergence'] > 2.0 else 'medium',
                    'description': f'템플릿 분포가 baseline과 크게 다름 (KL: {comparison["kl_divergence"]:.3f})',
                    'kl_divergence': comparison['kl_divergence'],
                    'jaccard_similarity': comparison['jaccard_similarity'],
                    'unique_templates': comparison['target_only_templates'],
                    'missing_templates': comparison['baseline_only_templates']
                })
            
            # 고유 템플릿이 많으면 이상
            if comparison['target_only_templates'] > len(baseline_avg_dist) * 0.2:
                anomalies.append({
                    'type': 'excessive_unique_templates',
                    'severity': 'medium',
                    'description': f'Target에만 있는 템플릿이 {comparison["target_only_templates"]}개로 과다',
                    'unique_templates_count': comparison['target_only_templates'],
                    'unique_templates': comparison['unique_in_target']
                })
        
        return anomalies
    
    def generate_comparative_report(self, target_profile: Dict, baseline_profiles: List[Dict], 
                                  anomalies: List[Dict], output_path: str) -> str:
        """비교 분석 리포트를 생성합니다."""
        
        report = f"""# 파일별 비교 이상 탐지 리포트

## 📊 분석 대상
- **Target 파일**: {target_profile['file_name']}
- **Baseline 파일**: {len(baseline_profiles)}개
- **Target 로그 수**: {target_profile['total_logs']:,}개
- **Target 템플릿 수**: {target_profile['unique_templates']}개

## 📈 Baseline 통계
"""
        
        if baseline_profiles:
            total_logs = [p['total_logs'] for p in baseline_profiles]
            unique_templates = [p['unique_templates'] for p in baseline_profiles]
            error_rates = [p.get('error_rate', 0) for p in baseline_profiles]
            
            report += f"""
- **평균 로그 수**: {np.mean(total_logs):,.0f}개 (±{np.std(total_logs):.0f})
- **평균 템플릿 수**: {np.mean(unique_templates):.0f}개 (±{np.std(unique_templates):.0f})
- **평균 에러율**: {np.mean(error_rates):.2%} (±{np.std(error_rates):.2%})

### Baseline 파일 목록
"""
            for i, profile in enumerate(baseline_profiles, 1):
                report += f"{i}. {profile['file_name']} - {profile['total_logs']:,}개 로그, {profile['unique_templates']}개 템플릿\n"
        
        # 이상 탐지 결과
        report += "\n## 🚨 발견된 이상 현상\n\n"
        
        if not anomalies:
            report += "✅ 비교 분석에서 이상 현상이 발견되지 않았습니다.\n\n"
        else:
            # 심각도별 분류
            high_severity = [a for a in anomalies if a['severity'] == 'high']
            medium_severity = [a for a in anomalies if a['severity'] == 'medium']
            
            if high_severity:
                report += f"### 🔴 심각한 이상 ({len(high_severity)}개)\n\n"
                for anomaly in high_severity:
                    report += f"- **{anomaly['type']}**: {anomaly['description']}\n"
                report += "\n"
            
            if medium_severity:
                report += f"### 🟡 주의 필요 ({len(medium_severity)}개)\n\n"
                for anomaly in medium_severity:
                    report += f"- **{anomaly['type']}**: {anomaly['description']}\n"
                report += "\n"
        
        # 상세 비교 표
        report += "## 📋 상세 비교표\n\n"
        report += "| 지표 | Target | Baseline 평균 | 차이 |\n"
        report += "|------|---------|---------------|------|\n"
        
        if baseline_profiles:
            metrics = [
                ('로그 수', 'total_logs'),
                ('템플릿 수', 'unique_templates'), 
                ('에러율', 'error_rate'),
                ('경고율', 'warning_rate'),
                ('엔트로피', 'template_entropy')
            ]
            
            for metric_name, metric_key in metrics:
                target_val = target_profile.get(metric_key, 0)
                baseline_vals = [p.get(metric_key, 0) for p in baseline_profiles if p.get(metric_key) is not None]
                
                if baseline_vals:
                    baseline_mean = np.mean(baseline_vals)
                    diff_pct = (target_val - baseline_mean) / max(baseline_mean, 1e-6) * 100
                    
                    if metric_key in ['error_rate', 'warning_rate']:
                        report += f"| {metric_name} | {target_val:.2%} | {baseline_mean:.2%} | {diff_pct:+.1f}% |\n"
                    elif metric_key == 'template_entropy':
                        report += f"| {metric_name} | {target_val:.2f} | {baseline_mean:.2f} | {diff_pct:+.1f}% |\n"
                    else:
                        report += f"| {metric_name} | {target_val:,} | {baseline_mean:,.0f} | {diff_pct:+.1f}% |\n"
        
        # 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

def compare_log_files(target_file: str, baseline_files: List[str], output_dir: str = None):
    """여러 로그 파일을 비교하여 이상을 탐지합니다."""
    
    target_path = Path(target_file)
    if output_dir is None:
        output_dir = target_path.parent / "comparative_analysis"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔍 파일별 비교 이상 탐지 시작...")
    
    detector = ComparativeAnomalyDetector()
    
    # Target 파일 분석
    print(f"📊 Target 파일 분석: {target_path.name}")
    target_df = pd.read_parquet(target_file)
    target_profile = detector.extract_file_profile(target_df, target_path.name)
    
    # Baseline 파일들 분석
    baseline_profiles = []
    print(f"📊 Baseline 파일 {len(baseline_files)}개 분석...")
    
    for baseline_file in baseline_files:
        baseline_path = Path(baseline_file)
        try:
            baseline_df = pd.read_parquet(baseline_file)
            profile = detector.extract_file_profile(baseline_df, baseline_path.name)
            baseline_profiles.append(profile)
            print(f"   ✅ {baseline_path.name}: {profile['total_logs']}개 로그")
        except Exception as e:
            print(f"   ❌ {baseline_path.name}: 로드 실패 ({e})")
    
    # 프로파일 저장
    # 프로파일 저장 (numpy types 변환)
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
    
    all_profiles = {
        'target': convert_numpy_types(target_profile),
        'baselines': convert_numpy_types(baseline_profiles)
    }
    with open(output_path / "file_profiles.json", 'w') as f:
        json.dump(all_profiles, f, indent=2, ensure_ascii=False)
    
    # 이상 탐지
    print("🔍 비교 분석 중...")
    anomalies = detector.detect_comparative_anomalies(target_profile, baseline_profiles)
    
    anomalies_serializable = convert_numpy_types(anomalies)
    with open(output_path / "comparative_anomalies.json", 'w') as f:
        json.dump(anomalies_serializable, f, indent=2, ensure_ascii=False)
    
    # 리포트 생성
    print("📄 리포트 생성 중...")
    report = detector.generate_comparative_report(
        target_profile, baseline_profiles, anomalies,
        output_path / "comparative_report.md"
    )
    
    print(f"✅ 완료! 결과는 {output_path}에 저장되었습니다.")
    print(f"🚨 발견된 이상: {len(anomalies)}개")
    
    # 간단한 요약 출력
    if anomalies:
        high_count = len([a for a in anomalies if a['severity'] == 'high'])
        medium_count = len([a for a in anomalies if a['severity'] == 'medium'])
        print(f"   - 심각: {high_count}개, 주의: {medium_count}개")
    
    return anomalies, target_profile, baseline_profiles

def main():
    parser = argparse.ArgumentParser(description="파일별 비교 로그 이상 탐지")
    parser.add_argument("--target", required=True, 
                       help="분석할 target 파일 (parsed.parquet)")
    parser.add_argument("--baselines", required=True, nargs='+',
                       help="비교할 baseline 파일들 (parsed.parquet)")
    parser.add_argument("--output-dir", 
                       help="결과 출력 디렉토리")
    
    args = parser.parse_args()
    compare_log_files(args.target, args.baselines, args.output_dir)

if __name__ == "__main__":
    main()
