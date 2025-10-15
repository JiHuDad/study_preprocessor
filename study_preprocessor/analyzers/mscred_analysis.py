#!/usr/bin/env python3
"""
MS-CRED 결과 분석 및 시각화 도구

MS-CRED 이상탐지 결과를 분석하고 시각화하여 리포트를 생성합니다.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse


class MSCREDAnalyzer:
    """MS-CRED 결과 분석기"""
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.load_results()
    
    def load_results(self):
        """MS-CRED 관련 파일들 로드"""
        
        # MS-CRED 추론 결과
        mscred_path = self.data_dir / "mscred_infer.parquet"
        if mscred_path.exists():
            self.results['mscred'] = pd.read_parquet(mscred_path)
            print(f"✅ MS-CRED 결과 로드: {len(self.results['mscred'])} 윈도우")
        
        # 윈도우 카운트 데이터
        window_counts_path = self.data_dir / "window_counts.parquet"
        if window_counts_path.exists():
            self.results['window_counts'] = pd.read_parquet(window_counts_path)
            print(f"✅ 윈도우 카운트 로드: {len(self.results['window_counts'])} 윈도우")
        
        # 원본 파싱 데이터 (맥락 분석용)
        parsed_path = self.data_dir / "parsed.parquet"
        if parsed_path.exists():
            self.results['parsed'] = pd.read_parquet(parsed_path)
            print(f"✅ 파싱 데이터 로드: {len(self.results['parsed'])} 로그")
        
        # 다른 이상탐지 결과들과 비교용
        baseline_path = self.data_dir / "baseline_scores.parquet"
        if baseline_path.exists():
            self.results['baseline'] = pd.read_parquet(baseline_path)
        
        deeplog_path = self.data_dir / "deeplog_infer.parquet"
        if deeplog_path.exists():
            self.results['deeplog'] = pd.read_parquet(deeplog_path)
    
    def analyze_reconstruction_errors(self) -> Dict:
        """재구성 오차 통계 분석"""
        if 'mscred' not in self.results:
            return {}
        
        df = self.results['mscred']
        errors = df['reconstruction_error']
        
        stats = {
            'count': len(errors),
            'mean': float(errors.mean()),
            'std': float(errors.std()),
            'min': float(errors.min()),
            'max': float(errors.max()),
            'median': float(errors.median()),
            'q75': float(errors.quantile(0.75)),
            'q95': float(errors.quantile(0.95)),
            'q99': float(errors.quantile(0.99)),
            'anomaly_count': int(df['is_anomaly'].sum()),
            'anomaly_rate': float(df['is_anomaly'].mean()),
            'threshold': float(df['threshold'].iloc[0]) if len(df) > 0 else 0.0
        }
        
        return stats
    
    def find_anomaly_patterns(self) -> List[Dict]:
        """이상 패턴 분석"""
        if 'mscred' not in self.results or 'window_counts' not in self.results:
            return []
        
        mscred_df = self.results['mscred']
        window_counts_df = self.results['window_counts']
        
        # 이상으로 판정된 윈도우들
        anomaly_windows = mscred_df[mscred_df['is_anomaly'] == True].copy()
        
        patterns = []
        for _, anomaly in anomaly_windows.head(20).iterrows():  # 상위 20개 이상 윈도우
            window_idx = int(anomaly['window_idx'])
            
            # 해당 윈도우의 템플릿 카운트 패턴
            if window_idx < len(window_counts_df):
                window_data = window_counts_df.iloc[window_idx]
                template_cols = [col for col in window_data.index if col.startswith('t')]
                template_counts = {col: int(window_data[col]) for col in template_cols if pd.notna(window_data[col]) and window_data[col] > 0}
                
                # 가장 빈번한 템플릿들
                top_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                pattern = {
                    'window_idx': window_idx,
                    'start_index': int(anomaly.get('start_index', window_idx)),
                    'reconstruction_error': float(anomaly['reconstruction_error']),
                    'top_templates': top_templates,
                    'total_logs': sum(template_counts.values()),
                    'unique_templates': len(template_counts)
                }
                patterns.append(pattern)
        
        return patterns
    
    def compare_with_other_methods(self) -> Dict:
        """다른 이상탐지 방법과 비교"""
        comparison = {
            'mscred_anomalies': set(),
            'baseline_anomalies': set(),
            'deeplog_anomalies': set(),
            'overlap_stats': {}
        }
        
        # MS-CRED 이상 윈도우들
        if 'mscred' in self.results:
            mscred_df = self.results['mscred']
            mscred_anomalies = mscred_df[mscred_df['is_anomaly'] == True]['window_idx'].tolist()
            comparison['mscred_anomalies'] = set(mscred_anomalies)
        
        # Baseline 이상 윈도우들 (window_start_line을 윈도우 인덱스로 변환)
        if 'baseline' in self.results:
            baseline_df = self.results['baseline']
            baseline_anomalies = baseline_df[baseline_df['is_anomaly'] == True]
            if 'window_start_line' in baseline_anomalies.columns:
                # start_line을 window_idx로 변환 (대략적)
                baseline_window_idx = (baseline_anomalies['window_start_line'] // 25).astype(int).tolist()
                comparison['baseline_anomalies'] = set(baseline_window_idx)
        
        # DeepLog는 시퀀스 기반이므로 직접 비교 어려움
        if 'deeplog' in self.results:
            deeplog_df = self.results['deeplog']
            violations = deeplog_df[deeplog_df['in_topk'] == False]
            comparison['deeplog_violations'] = len(violations)
        
        # 겹치는 부분 계산
        mscred_set = comparison['mscred_anomalies']
        baseline_set = comparison['baseline_anomalies']
        
        if mscred_set and baseline_set:
            overlap = mscred_set.intersection(baseline_set)
            comparison['overlap_stats'] = {
                'mscred_only': len(mscred_set - baseline_set),
                'baseline_only': len(baseline_set - mscred_set),
                'both_methods': len(overlap),
                'overlap_ratio': len(overlap) / len(mscred_set.union(baseline_set)) if mscred_set.union(baseline_set) else 0.0
            }
        
        return comparison
    
    def extract_anomaly_logs(self) -> List[Dict]:
        """이상 윈도우의 실제 로그 내용 추출"""
        if 'mscred' not in self.results or 'parsed' not in self.results:
            return []
        
        mscred_df = self.results['mscred']
        parsed_df = self.results['parsed']
        
        anomaly_windows = mscred_df[mscred_df['is_anomaly'] == True].nlargest(10, 'reconstruction_error')
        
        anomaly_logs = []
        for _, anomaly in anomaly_windows.iterrows():
            start_index = int(anomaly.get('start_index', anomaly['window_idx'] * 25))
            
            # 윈도우 범위의 로그들 추출 (50개 로그)
            window_logs = parsed_df[
                (parsed_df['line_no'] >= start_index) & 
                (parsed_df['line_no'] < start_index + 50)
            ].copy()
            
            if len(window_logs) > 0:
                # 에러 키워드 검출
                error_logs = window_logs[
                    window_logs['raw'].str.contains(
                        r'error|Error|ERROR|fail|Fail|FAIL|exception|Exception|EXCEPTION|warning|Warning|WARNING|critical|Critical|CRITICAL',
                        case=False, na=False, regex=True
                    )
                ]
                
                # 대표 로그들 선택
                sample_logs = window_logs.head(5)['raw'].tolist()
                error_samples = error_logs.head(3)['raw'].tolist() if len(error_logs) > 0 else []
                
                anomaly_info = {
                    'window_idx': int(anomaly['window_idx']),
                    'start_index': start_index,
                    'reconstruction_error': float(anomaly['reconstruction_error']),
                    'total_logs': len(window_logs),
                    'error_logs': len(error_logs),
                    'error_rate': len(error_logs) / len(window_logs) if len(window_logs) > 0 else 0.0,
                    'sample_logs': sample_logs,
                    'error_samples': error_samples,
                    'dominant_templates': window_logs['template'].value_counts().head(3).to_dict()
                }
                anomaly_logs.append(anomaly_info)
        
        return anomaly_logs
    
    def create_visualization(self, output_dir: str | Path):
        """시각화 생성"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'mscred' not in self.results:
            print("❌ MS-CRED 결과가 없어 시각화를 생성할 수 없습니다.")
            return
        
        df = self.results['mscred']
        
        # 1. 재구성 오차 분포
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df['reconstruction_error'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(df['threshold'].iloc[0], color='red', linestyle='--', label=f'Threshold: {df["threshold"].iloc[0]:.4f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('MS-CRED Reconstruction Error Distribution')
        plt.legend()
        
        # 2. 시간 순서별 재구성 오차
        plt.subplot(2, 2, 2)
        plt.plot(df['window_idx'], df['reconstruction_error'], alpha=0.6)
        anomalies = df[df['is_anomaly'] == True]
        if len(anomalies) > 0:
            plt.scatter(anomalies['window_idx'], anomalies['reconstruction_error'], 
                       color='red', s=30, alpha=0.8, label='Anomalies')
        plt.axhline(df['threshold'].iloc[0], color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Window Index')
        plt.ylabel('Reconstruction Error')
        plt.title('MS-CRED Errors Over Time')
        plt.legend()
        
        # 3. 이상률 통계
        plt.subplot(2, 2, 3)
        anomaly_rate = df['is_anomaly'].mean()
        normal_rate = 1 - anomaly_rate
        plt.pie([normal_rate, anomaly_rate], labels=['Normal', 'Anomaly'], 
                autopct='%1.1f%%', colors=['lightblue', 'red'])
        plt.title('MS-CRED Anomaly Distribution')
        
        # 4. 상위 이상 윈도우들
        plt.subplot(2, 2, 4)
        top_anomalies = df.nlargest(10, 'reconstruction_error')
        plt.barh(range(len(top_anomalies)), top_anomalies['reconstruction_error'])
        plt.yticks(range(len(top_anomalies)), [f"W{int(w)}" for w in top_anomalies['window_idx']])
        plt.xlabel('Reconstruction Error')
        plt.title('Top 10 Anomalous Windows')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mscred_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ MS-CRED 시각화 저장: {output_dir / 'mscred_analysis.png'}")
        
        # 5. 방법 간 비교 (다른 결과가 있는 경우)
        comparison = self.compare_with_other_methods()
        if comparison['overlap_stats']:
            plt.figure(figsize=(10, 6))
            
            stats = comparison['overlap_stats']
            categories = ['MS-CRED Only', 'Baseline Only', 'Both Methods']
            values = [stats['mscred_only'], stats['baseline_only'], stats['both_methods']]
            
            plt.bar(categories, values, color=['blue', 'green', 'orange'])
            plt.ylabel('Number of Anomalous Windows')
            plt.title('Anomaly Detection Method Comparison')
            
            for i, v in enumerate(values):
                plt.text(i, v + 0.1, str(v), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 방법 비교 시각화 저장: {output_dir / 'method_comparison.png'}")
    
    def generate_report(self, output_path: str | Path):
        """MS-CRED 분석 리포트 생성"""
        output_path = Path(output_path)
        
        if 'mscred' not in self.results:
            print("❌ MS-CRED 결과가 없어 리포트를 생성할 수 없습니다.")
            return
        
        # 분석 수행
        error_stats = self.analyze_reconstruction_errors()
        patterns = self.find_anomaly_patterns()
        comparison = self.compare_with_other_methods()
        anomaly_logs = self.extract_anomaly_logs()
        
        # 리포트 작성
        report_lines = [
            "# MS-CRED 이상탐지 분석 리포트",
            f"\n**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**분석 데이터**: {self.data_dir}",
            
            "\n## 📊 재구성 오차 통계",
            f"- **총 윈도우 수**: {error_stats['count']:,}",
            f"- **평균 재구성 오차**: {error_stats['mean']:.4f}",
            f"- **표준편차**: {error_stats['std']:.4f}",
            f"- **최대 오차**: {error_stats['max']:.4f}",
            f"- **95%ile**: {error_stats['q95']:.4f}",
            f"- **99%ile**: {error_stats['q99']:.4f}",
            f"- **임계값**: {error_stats['threshold']:.4f}",
            
            "\n## 🚨 이상탐지 결과",
            f"- **이상 윈도우 수**: {error_stats['anomaly_count']:,}",
            f"- **이상탐지율**: {error_stats['anomaly_rate']:.1%}",
        ]
        
        # 다른 방법과의 비교
        if comparison['overlap_stats']:
            stats = comparison['overlap_stats']
            report_lines.extend([
                "\n## 🔍 다른 이상탐지 방법과의 비교",
                f"- **MS-CRED 단독 탐지**: {stats['mscred_only']} 윈도우",
                f"- **Baseline 단독 탐지**: {stats['baseline_only']} 윈도우", 
                f"- **두 방법 모두 탐지**: {stats['both_methods']} 윈도우",
                f"- **겹침 비율**: {stats['overlap_ratio']:.1%}",
            ])
        
        # 상위 이상 패턴들
        if patterns:
            report_lines.extend([
                "\n## 🎯 주요 이상 패턴 (상위 10개)",
                ""
            ])
            
            for i, pattern in enumerate(patterns[:10], 1):
                report_lines.extend([
                    f"### {i}. 윈도우 {pattern['window_idx']} (시작: {pattern['start_index']})",
                    f"- **재구성 오차**: {pattern['reconstruction_error']:.4f}",
                    f"- **총 로그 수**: {pattern['total_logs']}",
                    f"- **고유 템플릿 수**: {pattern['unique_templates']}",
                    f"- **주요 템플릿**: {', '.join([f'{t}({c})' for t, c in pattern['top_templates'][:3]])}",
                    ""
                ])
        
        # 실제 이상 로그 샘플들
        if anomaly_logs:
            report_lines.extend([
                "\n## 📋 실제 이상 로그 샘플들",
                ""
            ])
            
            for i, log_info in enumerate(anomaly_logs[:5], 1):
                report_lines.extend([
                    f"### {i}. 윈도우 {log_info['window_idx']} (에러율: {log_info['error_rate']:.1%})",
                    f"**재구성 오차**: {log_info['reconstruction_error']:.4f}",
                    f"**로그 수**: {log_info['total_logs']} (에러: {log_info['error_logs']})",
                    ""
                ])
                
                if log_info['error_samples']:
                    report_lines.extend([
                        "**🚨 에러 로그 샘플들**:",
                        ""
                    ])
                    for error_log in log_info['error_samples']:
                        report_lines.append(f"```\n{error_log[:200]}...\n```")
                    report_lines.append("")
                
                if log_info['sample_logs']:
                    report_lines.extend([
                        "**📄 일반 로그 샘플들**:",
                        ""
                    ])
                    for sample_log in log_info['sample_logs'][:3]:
                        report_lines.append(f"```\n{sample_log[:150]}...\n```")
                    report_lines.append("")
        
        # 결론 및 권장사항
        report_lines.extend([
            "\n## 💡 결론 및 권장사항",
            "",
            f"- MS-CRED는 총 {error_stats['count']:,}개 윈도우 중 {error_stats['anomaly_count']:,}개를 이상으로 탐지했습니다.",
            f"- 이상탐지율 {error_stats['anomaly_rate']:.1%}는 "
            + ("정상 범위" if error_stats['anomaly_rate'] < 0.1 else "높은 편") + "입니다.",
        ])
        
        if comparison['overlap_stats']:
            overlap_ratio = comparison['overlap_stats']['overlap_ratio']
            if overlap_ratio > 0.5:
                report_lines.append("- 다른 이상탐지 방법과 높은 일치율을 보여 신뢰성이 높습니다.")
            else:
                report_lines.append("- 다른 방법과 다른 패턴을 탐지하므로 상호 보완적으로 활용하세요.")
        
        report_lines.extend([
            "- 높은 재구성 오차를 보이는 윈도우들을 우선적으로 검토하세요.",
            "- 에러 로그가 집중된 구간은 시스템 문제 발생 가능성이 높습니다.",
            ""
        ])
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ MS-CRED 분석 리포트 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MS-CRED 결과 분석 및 리포트 생성")
    parser.add_argument("--data-dir", type=str, required=True, 
                       help="MS-CRED 결과가 포함된 데이터 디렉토리")
    parser.add_argument("--output-dir", type=str, 
                       help="출력 디렉토리 (기본: 데이터 디렉토리)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    
    if not data_dir.exists():
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return
    
    print(f"🔍 MS-CRED 결과 분석 시작: {data_dir}")
    
    # 분석기 생성 및 실행
    analyzer = MSCREDAnalyzer(data_dir)
    
    # 시각화 생성
    analyzer.create_visualization(output_dir)
    
    # 리포트 생성
    report_path = output_dir / "mscred_analysis_report.md"
    analyzer.generate_report(report_path)
    
    print(f"✅ MS-CRED 분석 완료! 결과는 {output_dir}에서 확인하세요.")


if __name__ == "__main__":
    main()
