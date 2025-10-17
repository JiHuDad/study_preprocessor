#!/usr/bin/env python3
"""
Baseline 파일 품질 검증 도구
- 이상한 baseline 파일들을 사전에 필터링
- 다양한 휴리스틱으로 baseline 품질 평가
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import json

class BaselineValidator:
    def __init__(self):
        # 정상 baseline의 기준값들 (경험적 임계값)
        self.thresholds = {
            'max_error_rate': 0.02,      # 2% 이하 에러율
            'max_warning_rate': 0.05,    # 5% 이하 경고율  
            'min_template_count': 10,    # 최소 10개 템플릿
            'max_rare_template_ratio': 0.3,  # 희귀 템플릿 30% 이하
            'min_log_count': 100,        # 최소 100개 로그
            'max_template_entropy': 10,  # 템플릿 엔트로피 상한
            'min_template_entropy': 2,   # 템플릿 엔트로피 하한
        }
    
    def validate_single_baseline(self, parsed_file: str) -> Dict:
        """단일 baseline 파일을 검증합니다."""
        try:
            df = pd.read_parquet(parsed_file)
            file_name = Path(parsed_file).name
            
            # 기본 통계 계산
            total_logs = len(df)
            if total_logs == 0:
                return {
                    'file': file_name,
                    'valid': False,
                    'score': 0.0,
                    'issues': ['Empty dataset'],
                    'stats': {}
                }
            
            # 템플릿 분석
            template_counts = df['template_id'].value_counts()
            unique_templates = len(template_counts)
            rare_templates = len([t for t, count in template_counts.items() if count == 1])
            rare_template_ratio = rare_templates / max(unique_templates, 1)
            
            # 템플릿 분포 엔트로피
            template_probs = template_counts / total_logs
            template_entropy = -sum(p * np.log2(p) for p in template_probs if p > 0)
            
            # 에러/경고 로그 분석
            error_keywords = ['error', 'ERROR', 'fail', 'FAIL', 'exception', 'Exception', 'panic', 'fatal', 'crash']
            warning_keywords = ['warn', 'WARN', 'warning', 'WARNING', 'deprecated']
            
            error_logs = df[df['raw'].str.contains('|'.join(error_keywords), case=False, na=False)]
            warning_logs = df[df['raw'].str.contains('|'.join(warning_keywords), case=False, na=False)]
            
            error_rate = len(error_logs) / total_logs
            warning_rate = len(warning_logs) / total_logs
            
            # 시간 일관성 검사
            time_consistency = True
            time_gaps = []
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp').copy()
                df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], errors='coerce')
                valid_timestamps = df_sorted.dropna(subset=['timestamp'])
                
                if len(valid_timestamps) > 1:
                    time_diffs = valid_timestamps['timestamp'].diff().dt.total_seconds()
                    # 1시간 이상 갭이 있으면 의심
                    large_gaps = time_diffs[time_diffs > 3600]
                    time_gaps = large_gaps.tolist()
                    time_consistency = len(large_gaps) < total_logs * 0.1  # 10% 이하
            
            # 로그 밀도 분석 (시간당 로그 수)
            logs_per_hour = None
            if 'timestamp' in df.columns and len(df) > 1:
                try:
                    start_time = pd.to_datetime(df['timestamp'].min())
                    end_time = pd.to_datetime(df['timestamp'].max())
                    duration_hours = (end_time - start_time).total_seconds() / 3600
                    if duration_hours > 0:
                        logs_per_hour = total_logs / duration_hours
                except:
                    pass
            
            # 통계 정리
            stats = {
                'total_logs': total_logs,
                'unique_templates': unique_templates,
                'rare_templates': rare_templates,
                'rare_template_ratio': rare_template_ratio,
                'template_entropy': template_entropy,
                'error_rate': error_rate,
                'warning_rate': warning_rate,
                'time_consistency': time_consistency,
                'large_time_gaps': len(time_gaps),
                'logs_per_hour': logs_per_hour
            }
            
            # 검증 결과 계산
            issues = []
            score = 1.0  # 완벽한 점수에서 시작하여 문제마다 감점
            
            # 각 기준별 검사
            if error_rate > self.thresholds['max_error_rate']:
                issues.append(f"높은 에러율: {error_rate:.2%} (기준: {self.thresholds['max_error_rate']:.2%})")
                score -= 0.3
            
            if warning_rate > self.thresholds['max_warning_rate']:
                issues.append(f"높은 경고율: {warning_rate:.2%} (기준: {self.thresholds['max_warning_rate']:.2%})")
                score -= 0.2
            
            if unique_templates < self.thresholds['min_template_count']:
                issues.append(f"템플릿 부족: {unique_templates}개 (기준: {self.thresholds['min_template_count']}개)")
                score -= 0.2
            
            if rare_template_ratio > self.thresholds['max_rare_template_ratio']:
                issues.append(f"희귀 템플릿 과다: {rare_template_ratio:.2%} (기준: {self.thresholds['max_rare_template_ratio']:.2%})")
                score -= 0.1
            
            if total_logs < self.thresholds['min_log_count']:
                issues.append(f"로그 수 부족: {total_logs}개 (기준: {self.thresholds['min_log_count']}개)")
                score -= 0.3
            
            if template_entropy > self.thresholds['max_template_entropy']:
                issues.append(f"템플릿 엔트로피 과다: {template_entropy:.2f} (기준: {self.thresholds['max_template_entropy']})")
                score -= 0.2
            
            if template_entropy < self.thresholds['min_template_entropy']:
                issues.append(f"템플릿 다양성 부족: {template_entropy:.2f} (기준: {self.thresholds['min_template_entropy']})")
                score -= 0.1
            
            if not time_consistency:
                issues.append("시간 일관성 문제: 큰 시간 갭이 다수 존재")
                score -= 0.1
            
            # 최종 점수 조정
            score = max(0.0, score)
            is_valid = score >= 0.7 and len(issues) <= 2  # 70% 이상 점수, 문제 2개 이하
            
            return {
                'file': file_name,
                'valid': is_valid,
                'score': score,
                'issues': issues,
                'stats': stats
            }
            
        except Exception as e:
            return {
                'file': Path(parsed_file).name,
                'valid': False,
                'score': 0.0,
                'issues': [f'처리 오류: {str(e)}'],
                'stats': {}
            }
    
    def validate_multiple_baselines(self, baseline_files: List[str]) -> Dict:
        """여러 baseline 파일들을 검증하고 필터링합니다."""
        
        print(f"🔍 {len(baseline_files)}개 baseline 파일 검증 중...")
        
        results = []
        for baseline_file in baseline_files:
            print(f"   검증: {Path(baseline_file).name}")
            result = self.validate_single_baseline(baseline_file)
            results.append(result)
        
        # 결과 분석
        valid_baselines = [r for r in results if r['valid']]
        invalid_baselines = [r for r in results if not r['valid']]
        
        # 상호 일관성 검사 (valid한 것들끼리)
        if len(valid_baselines) > 1:
            consistency_issues = self._check_mutual_consistency(valid_baselines, baseline_files)
        else:
            consistency_issues = []
        
        print(f"\n📊 검증 결과:")
        print(f"   ✅ 유효: {len(valid_baselines)}개")
        print(f"   ❌ 무효: {len(invalid_baselines)}개")
        print(f"   ⚠️  일관성 문제: {len(consistency_issues)}개")
        
        return {
            'total_files': len(baseline_files),
            'valid_count': len(valid_baselines),
            'invalid_count': len(invalid_baselines),
            'valid_baselines': valid_baselines,
            'invalid_baselines': invalid_baselines,
            'consistency_issues': consistency_issues,
            'recommended_baselines': [r['file'] for r in valid_baselines if r['score'] >= 0.8]
        }
    
    def _check_mutual_consistency(self, valid_results: List[Dict], baseline_files: List[str]) -> List[Dict]:
        """유효한 baseline들 간의 상호 일관성을 검사합니다."""
        
        consistency_issues = []
        
        # 주요 지표들의 분포 분석
        metrics = ['error_rate', 'warning_rate', 'template_entropy', 'logs_per_hour']
        
        for metric in metrics:
            values = []
            for result in valid_results:
                val = result['stats'].get(metric)
                if val is not None:
                    values.append(val)
            
            if len(values) < 2:
                continue
                
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Outlier 탐지 (2σ 밖의 값들)
            for i, (result, val) in enumerate(zip(valid_results, values)):
                if abs(val - mean_val) > 2 * std_val:
                    consistency_issues.append({
                        'file': result['file'],
                        'metric': metric,
                        'value': val,
                        'mean': mean_val,
                        'std': std_val,
                        'description': f'{metric}이 다른 baseline들과 {abs(val - mean_val)/max(mean_val, 1e-6)*100:.1f}% 차이'
                    })
        
        return consistency_issues
    
    def generate_validation_report(self, validation_result: Dict, output_file: str = None) -> str:
        """검증 결과 리포트를 생성합니다."""
        
        report = f"""# Baseline 파일 검증 리포트

## 📊 전체 요약
- **총 파일 수**: {validation_result['total_files']}개
- **유효한 파일**: {validation_result['valid_count']}개
- **무효한 파일**: {validation_result['invalid_count']}개
- **추천 파일**: {len(validation_result['recommended_baselines'])}개

## ✅ 유효한 Baseline 파일들

"""
        
        for result in validation_result['valid_baselines']:
            stats = result['stats']
            report += f"""### {result['file']}
- **품질 점수**: {result['score']:.2f}/1.0
- **로그 수**: {stats.get('total_logs', 0):,}개
- **템플릿 수**: {stats.get('unique_templates', 0)}개
- **에러율**: {stats.get('error_rate', 0):.2%}
- **경고율**: {stats.get('warning_rate', 0):.2%}
- **템플릿 엔트로피**: {stats.get('template_entropy', 0):.2f}
"""
            if result['issues']:
                report += f"- **경미한 이슈**: {', '.join(result['issues'])}\n"
            report += "\n"
        
        if validation_result['invalid_baselines']:
            report += "## ❌ 무효한 Baseline 파일들\n\n"
            for result in validation_result['invalid_baselines']:
                report += f"""### {result['file']}
- **품질 점수**: {result['score']:.2f}/1.0
- **주요 문제**: {', '.join(result['issues'])}

"""
        
        if validation_result['consistency_issues']:
            report += "## ⚠️  일관성 문제\n\n"
            for issue in validation_result['consistency_issues']:
                report += f"- **{issue['file']}**: {issue['description']}\n"
            report += "\n"
        
        report += f"""## 💡 권장사항

### 사용 권장 파일들
{chr(10).join('- ' + f for f in validation_result['recommended_baselines'])}

### 품질 개선 방법
1. **에러율이 높은 파일들**: 해당 시기의 시스템 상태 확인
2. **로그 수가 적은 파일들**: 더 긴 기간의 로그 수집
3. **템플릿 다양성 문제**: 다양한 시스템 활동이 포함된 기간 선택
4. **시간 일관성 문제**: 로그 수집 중단 구간 확인

### 이상탐지 정확도 향상을 위한 조치
- 품질 점수 0.8 이상 파일들만 baseline으로 사용
- 최소 3개 이상의 유효한 baseline 확보
- 일관성 문제가 있는 파일들은 제외 고려
"""
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 검증 리포트 저장: {output_file}")
        
        return report

def validate_baseline_files(baseline_files: List[str], output_dir: str = None):
    """Baseline 파일들을 검증하고 리포트를 생성합니다."""
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    validator = BaselineValidator()
    
    # 검증 실행
    result = validator.validate_multiple_baselines(baseline_files)
    
    # 결과 저장
    if output_dir:
        # JSON 결과 저장
        with open(output_path / "validation_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        # 리포트 생성
        report = validator.generate_validation_report(result, output_path / "baseline_validation_report.md")
        
        print(f"📂 결과 저장 위치: {output_path}")
    else:
        validator.generate_validation_report(result)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Baseline 파일 품질 검증 도구")
    parser.add_argument("baseline_files", nargs='+', help="검증할 baseline 파일들 (parsed.parquet)")
    parser.add_argument("--output-dir", help="결과 출력 디렉토리")
    parser.add_argument("--score-threshold", type=float, default=0.7, help="유효 baseline 최소 점수 (기본: 0.7)")
    
    args = parser.parse_args()
    
    # 임계값 업데이트
    validator = BaselineValidator()
    
    result = validate_baseline_files(args.baseline_files, args.output_dir)
    
    print(f"\n🎯 권장 baseline 파일들:")
    for file in result['recommended_baselines']:
        print(f"   ✅ {file}")

if __name__ == "__main__":
    main()
