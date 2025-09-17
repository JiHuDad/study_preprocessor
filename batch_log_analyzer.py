#!/usr/bin/env python3
"""
배치 로그 분석기
- 폴더 내 여러 로그 파일들을 자동 전처리 및 분석
- 특정 Target 파일과 다른 파일들을 비교 분석
- 시간 기반 이상 탐지도 함께 수행
"""
import os
import sys
import subprocess
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse
import shutil
from glob import glob

class BatchLogAnalyzer:
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir or "batch_analysis")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.processed_files = {}
        self.analysis_results = {}
        
    def find_log_files(self, input_dir: str, pattern: str = "*.log") -> List[Path]:
        """입력 디렉토리에서 로그 파일들을 찾습니다."""
        input_path = Path(input_dir)
        log_files = []
        
        # 패턴에 맞는 파일들 찾기
        for pattern_variant in [pattern, "*.txt", "*.log*"]:
            files = list(input_path.glob(pattern_variant))
            log_files.extend(files)
        
        # 중복 제거 및 정렬
        log_files = sorted(list(set(log_files)))
        
        print(f"📂 발견된 로그 파일: {len(log_files)}개")
        for i, file_path in enumerate(log_files, 1):
            file_size = file_path.stat().st_size / (1024*1024)  # MB
            print(f"  {i:2d}. {file_path.name} ({file_size:.1f} MB)")
        
        return log_files
    
    def preprocess_log_file(self, log_file: Path) -> Dict:
        """단일 로그 파일을 전처리합니다."""
        file_name = log_file.stem
        output_dir = self.work_dir / f"processed_{file_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 전처리 중: {log_file.name}")
        
        try:
            # study-preprocess 명령어로 전처리 실행
            cmd = [
                sys.executable, "-m", "study_preprocessor.cli", "parse",
                "--input", str(log_file),
                "--out-dir", str(output_dir),
                "--drain-state", str(self.work_dir / f"drain_{file_name}.json")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ 전처리 실패: {log_file.name}")
                print(f"Error: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'file_path': log_file,
                    'output_dir': output_dir
                }
            
            # 결과 파일 확인
            parsed_file = output_dir / "parsed.parquet"
            if not parsed_file.exists():
                print(f"❌ 파싱 결과 없음: {log_file.name}")
                return {
                    'success': False,
                    'error': 'No parsed.parquet generated',
                    'file_path': log_file,
                    'output_dir': output_dir
                }
            
            # 기본 통계 수집
            df = pd.read_parquet(parsed_file)
            stats = {
                'total_logs': len(df),
                'unique_templates': len(df['template_id'].unique()) if 'template_id' in df.columns else 0,
                'time_range': {
                    'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                    'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else None,
                },
                'hosts': list(df['host'].unique()) if 'host' in df.columns else [],
                'processes': list(df['process'].unique()) if 'process' in df.columns else []
            }
            
            print(f"✅ 완료: {stats['total_logs']:,}개 로그, {stats['unique_templates']}개 템플릿")
            
            return {
                'success': True,
                'file_path': log_file,
                'output_dir': output_dir,
                'parsed_file': parsed_file,
                'stats': stats
            }
            
        except Exception as e:
            print(f"❌ 예외 발생: {log_file.name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': log_file,
                'output_dir': output_dir
            }
    
    def run_temporal_analysis(self, target_result: Dict) -> Dict:
        """시간 기반 이상 탐지를 실행합니다."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        print(f"\n🕐 시간 기반 이상 탐지: {target_result['file_path'].name}")
        
        try:
            # temporal_anomaly_detector 실행
            cmd = [
                sys.executable, "temporal_anomaly_detector.py",
                "--data-dir", str(target_result['output_dir']),
                "--output-dir", str(target_result['output_dir'] / "temporal_analysis")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ 시간 분석 실패")
                return {'success': False, 'error': result.stderr}
            
            # 결과 로드
            temporal_dir = target_result['output_dir'] / "temporal_analysis"
            anomalies_file = temporal_dir / "temporal_anomalies.json"
            
            temporal_result = {'success': True, 'anomalies': []}
            if anomalies_file.exists():
                with open(anomalies_file) as f:
                    temporal_result['anomalies'] = json.load(f)
            
            print(f"✅ 시간 분석 완료: {len(temporal_result['anomalies'])}개 이상 발견")
            return temporal_result
            
        except Exception as e:
            print(f"❌ 시간 분석 예외: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_comparative_analysis(self, target_result: Dict, baseline_results: List[Dict]) -> Dict:
        """파일별 비교 이상 탐지를 실행합니다."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        # 성공한 baseline 파일들만 선별
        valid_baselines = [r for r in baseline_results if r['success']]
        if not valid_baselines:
            print("⚠️ 비교할 baseline 파일이 없습니다")
            return {'success': False, 'error': 'No valid baseline files'}
        
        print(f"\n📊 파일별 비교 분석: {target_result['file_path'].name} vs {len(valid_baselines)}개 파일")
        
        try:
            # baseline 파일 경로들 수집
            baseline_paths = [str(r['parsed_file']) for r in valid_baselines]
            
            # comparative_anomaly_detector 실행
            cmd = [
                sys.executable, "comparative_anomaly_detector.py",
                "--target", str(target_result['parsed_file']),
                "--baselines"] + baseline_paths + [
                "--output-dir", str(target_result['output_dir'] / "comparative_analysis")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ 비교 분석 실패")
                return {'success': False, 'error': result.stderr}
            
            # 결과 로드
            comp_dir = target_result['output_dir'] / "comparative_analysis"
            anomalies_file = comp_dir / "comparative_anomalies.json"
            
            comp_result = {'success': True, 'anomalies': [], 'baseline_count': len(valid_baselines)}
            if anomalies_file.exists():
                with open(anomalies_file) as f:
                    comp_result['anomalies'] = json.load(f)
            
            print(f"✅ 비교 분석 완료: {len(comp_result['anomalies'])}개 이상 발견")
            return comp_result
            
        except Exception as e:
            print(f"❌ 비교 분석 예외: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_summary_report(self, target_result: Dict, baseline_results: List[Dict],
                              temporal_result: Dict, comparative_result: Dict) -> str:
        """전체 분석 결과 요약 리포트를 생성합니다."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_name = target_result['file_path'].name
        
        report = f"""# 배치 로그 분석 요약 리포트

**분석 시간**: {timestamp}  
**Target 파일**: {target_name}  
**Baseline 파일**: {len([r for r in baseline_results if r['success']])}개

## 📊 파일별 전처리 결과

### Target 파일: {target_name}
"""
        
        if target_result['success']:
            stats = target_result['stats']
            report += f"""- ✅ **성공**: {stats['total_logs']:,}개 로그, {stats['unique_templates']}개 템플릿
- **시간 범위**: {stats['time_range']['start']} ~ {stats['time_range']['end']}
- **호스트**: {len(stats['hosts'])}개 ({', '.join(stats['hosts'][:3])}{'...' if len(stats['hosts']) > 3 else ''})
- **프로세스**: {len(stats['processes'])}개
"""
        else:
            report += f"- ❌ **실패**: {target_result['error']}\n"
        
        report += "\n### Baseline 파일들\n"
        for i, result in enumerate(baseline_results, 1):
            if result['success']:
                stats = result['stats']
                report += f"{i}. ✅ **{result['file_path'].name}**: {stats['total_logs']:,}개 로그, {stats['unique_templates']}개 템플릿\n"
            else:
                report += f"{i}. ❌ **{result['file_path'].name}**: {result['error']}\n"
        
        # 시간 기반 분석 결과
        report += "\n## 🕐 시간 기반 이상 탐지 결과\n\n"
        if temporal_result['success']:
            anomalies = temporal_result['anomalies']
            if anomalies:
                high_count = len([a for a in anomalies if a.get('severity') == 'high'])
                medium_count = len([a for a in anomalies if a.get('severity') == 'medium'])
                report += f"🚨 **발견된 이상**: {len(anomalies)}개 (심각: {high_count}개, 주의: {medium_count}개)\n\n"
                
                for anomaly in anomalies[:5]:  # 최대 5개만 표시
                    report += f"- **{anomaly.get('type', 'unknown')}** ({anomaly.get('hour', '?')}시): {anomaly.get('description', 'No description')}\n"
                
                if len(anomalies) > 5:
                    report += f"- ... 및 {len(anomalies) - 5}개 추가\n"
            else:
                report += "✅ 시간 기반 이상 현상이 발견되지 않았습니다.\n"
        else:
            report += f"❌ 시간 기반 분석 실패: {temporal_result.get('error', 'Unknown error')}\n"
        
        # 파일별 비교 분석 결과
        report += "\n## 📊 파일별 비교 이상 탐지 결과\n\n"
        if comparative_result['success']:
            anomalies = comparative_result['anomalies']
            baseline_count = comparative_result['baseline_count']
            report += f"**비교 대상**: {baseline_count}개 baseline 파일\n\n"
            
            if anomalies:
                high_count = len([a for a in anomalies if a.get('severity') == 'high'])
                medium_count = len([a for a in anomalies if a.get('severity') == 'medium'])
                report += f"🚨 **발견된 이상**: {len(anomalies)}개 (심각: {high_count}개, 주의: {medium_count}개)\n\n"
                
                for anomaly in anomalies[:5]:  # 최대 5개만 표시
                    report += f"- **{anomaly.get('type', 'unknown')}**: {anomaly.get('description', 'No description')}\n"
                
                if len(anomalies) > 5:
                    report += f"- ... 및 {len(anomalies) - 5}개 추가\n"
            else:
                report += "✅ 파일별 비교에서 이상 현상이 발견되지 않았습니다.\n"
        else:
            report += f"❌ 파일별 비교 분석 실패: {comparative_result.get('error', 'Unknown error')}\n"
        
        # 권고사항
        report += "\n## 💡 권고사항\n\n"
        
        total_anomalies = 0
        if temporal_result['success']:
            total_anomalies += len(temporal_result.get('anomalies', []))
        if comparative_result['success']:
            total_anomalies += len(comparative_result.get('anomalies', []))
        
        if total_anomalies == 0:
            report += "✅ 모든 분석에서 이상 현상이 발견되지 않았습니다. 시스템이 정상적으로 작동하고 있는 것으로 보입니다.\n"
        elif total_anomalies < 5:
            report += "🔍 일부 이상 현상이 발견되었습니다. 상세 리포트를 검토하여 추가 조사가 필요한지 확인하세요.\n"
        else:
            report += "⚠️ 다수의 이상 현상이 발견되었습니다. 시스템 점검이 필요할 수 있습니다.\n"
        
        report += f"""
## 📂 상세 결과 파일
- **Target 전처리 결과**: `{target_result['output_dir']}/parsed.parquet`
- **시간 기반 분석**: `{target_result['output_dir']}/temporal_analysis/temporal_report.md`
- **파일별 비교 분석**: `{target_result['output_dir']}/comparative_analysis/comparative_report.md`

## 🔧 추가 분석 명령어
```bash
# 상세 분석
python analyze_results.py --data-dir {target_result['output_dir']}

# 시각화
python visualize_results.py --data-dir {target_result['output_dir']}
```
"""
        
        return report
    
    def analyze_directory(self, input_dir: str, target_file: str = None, 
                         file_pattern: str = "*.log") -> Dict:
        """디렉토리 내 로그 파일들을 일괄 분석합니다."""
        
        print("🚀 배치 로그 분석 시작")
        print(f"📂 입력 디렉토리: {input_dir}")
        print(f"🎯 작업 디렉토리: {self.work_dir}")
        
        # 1. 로그 파일 찾기
        log_files = self.find_log_files(input_dir, file_pattern)
        if not log_files:
            print("❌ 로그 파일을 찾을 수 없습니다")
            return {'success': False, 'error': 'No log files found'}
        
        # 2. Target 파일 결정
        if target_file:
            target_path = Path(target_file)
            if target_path not in log_files:
                # 파일명으로 매칭 시도
                target_matches = [f for f in log_files if f.name == target_path.name]
                if target_matches:
                    target_path = target_matches[0]
                else:
                    print(f"❌ Target 파일을 찾을 수 없습니다: {target_file}")
                    return {'success': False, 'error': f'Target file not found: {target_file}'}
        else:
            # 첫 번째 파일을 target으로 설정
            target_path = log_files[0]
            print(f"🎯 Target 파일 자동 선택: {target_path.name}")
        
        baseline_files = [f for f in log_files if f != target_path]
        print(f"📊 Baseline 파일: {len(baseline_files)}개")
        
        # 3. 모든 파일 전처리
        print(f"\n{'='*60}")
        print("📋 전처리 단계")
        print(f"{'='*60}")
        
        # Target 파일 전처리
        target_result = self.preprocess_log_file(target_path)
        
        # Baseline 파일들 전처리
        baseline_results = []
        for baseline_file in baseline_files:
            result = self.preprocess_log_file(baseline_file)
            baseline_results.append(result)
        
        # 4. 시간 기반 이상 탐지
        print(f"\n{'='*60}")
        print("🕐 시간 기반 이상 탐지")
        print(f"{'='*60}")
        temporal_result = self.run_temporal_analysis(target_result)
        
        # 5. 파일별 비교 이상 탐지
        print(f"\n{'='*60}")
        print("📊 파일별 비교 이상 탐지")
        print(f"{'='*60}")
        comparative_result = self.run_comparative_analysis(target_result, baseline_results)
        
        # 6. 요약 리포트 생성
        print(f"\n{'='*60}")
        print("📄 요약 리포트 생성")
        print(f"{'='*60}")
        
        summary_report = self.generate_summary_report(
            target_result, baseline_results, temporal_result, comparative_result
        )
        
        # 요약 리포트 저장
        summary_file = self.work_dir / "BATCH_ANALYSIS_SUMMARY.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"📋 요약 리포트 저장: {summary_file}")
        
        # 결과 출력
        print(f"\n{'='*60}")
        print("✅ 배치 분석 완료")
        print(f"{'='*60}")
        
        # 간단한 요약 출력
        success_count = len([r for r in baseline_results if r['success']]) + (1 if target_result['success'] else 0)
        total_count = len(baseline_results) + 1
        print(f"📊 전처리 성공: {success_count}/{total_count}개 파일")
        
        if temporal_result['success']:
            temporal_anomalies = len(temporal_result.get('anomalies', []))
            print(f"🕐 시간 기반 이상: {temporal_anomalies}개")
        
        if comparative_result['success']:
            comp_anomalies = len(comparative_result.get('anomalies', []))
            print(f"📊 비교 분석 이상: {comp_anomalies}개")
        
        print(f"📄 요약 리포트: {summary_file}")
        
        return {
            'success': True,
            'target_result': target_result,
            'baseline_results': baseline_results,
            'temporal_result': temporal_result,
            'comparative_result': comparative_result,
            'summary_file': summary_file
        }

def main():
    parser = argparse.ArgumentParser(description="배치 로그 분석기")
    parser.add_argument("input_dir", help="로그 파일들이 있는 디렉토리")
    parser.add_argument("--target", help="분석할 target 파일 (지정하지 않으면 첫 번째 파일)")
    parser.add_argument("--pattern", default="*.log", help="로그 파일 패턴 (기본: *.log)")
    parser.add_argument("--work-dir", help="작업 디렉토리 (기본: ./batch_analysis)")
    
    args = parser.parse_args()
    
    # 분석기 초기화 및 실행
    analyzer = BatchLogAnalyzer(args.work_dir)
    result = analyzer.analyze_directory(
        args.input_dir, 
        args.target, 
        args.pattern
    )
    
    if not result['success']:
        print(f"❌ 분석 실패: {result['error']}")
        sys.exit(1)
    
    print("\n🎉 모든 분석이 완료되었습니다!")

if __name__ == "__main__":
    main()
