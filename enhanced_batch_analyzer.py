#!/usr/bin/env python3
"""
향상된 배치 로그 분석기
- 하위 디렉토리 재귀적 스캔 지원
- 날짜별/카테고리별 폴더 구조 지원
- 로그 형식 자동 감지 및 검증
- 전처리 오류 디버깅 강화
"""
import os
import sys
import subprocess
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import argparse
import shutil
from glob import glob
import re

class EnhancedBatchAnalyzer:
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir or f"enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.processed_files = {}
        self.analysis_results = {}
        
        # 로그 파일 패턴들
        self.log_patterns = [
            "*.log", "*.txt", "*.out", 
            "*.log.*", "*.syslog", "*.messages",
            "*.access", "*.error", "*.debug", "*.log*"
        ]
        
    def find_log_files_recursive(self, input_dir: str, max_depth: int = 3) -> List[Tuple[Path, str]]:
        """하위 디렉토리를 포함하여 로그 파일들을 재귀적으로 찾습니다."""
        input_path = Path(input_dir)
        log_files = []
        
        print(f"📂 로그 파일 검색 중: {input_dir} (최대 깊이: {max_depth})")
        
        def scan_directory(dir_path: Path, current_depth: int = 0):
            if current_depth > max_depth:
                return
                
            try:
                # 현재 디렉토리에서 로그 파일 찾기
                for pattern in self.log_patterns:
                    for file_path in dir_path.glob(pattern):
                        if file_path.is_file() and file_path.stat().st_size > 0:
                            # 상대 경로 생성 (디렉토리 구조 정보 보존)
                            rel_path = file_path.relative_to(input_path)
                            category = str(rel_path.parent) if rel_path.parent != Path('.') else "root"
                            log_files.append((file_path, category))
                
                # 하위 디렉토리 재귀 스캔
                for item in dir_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        scan_directory(item, current_depth + 1)
                        
            except PermissionError:
                print(f"⚠️  권한 없음: {dir_path}")
            except Exception as e:
                print(f"⚠️  스캔 오류: {dir_path} - {e}")
        
        scan_directory(input_path)
        
        # 파일 크기별 정렬 (큰 파일부터)
        log_files.sort(key=lambda x: x[0].stat().st_size, reverse=True)
        
        print(f"📊 발견된 로그 파일: {len(log_files)}개")
        
        # 카테고리별 그룹화 출력
        categories = {}
        for file_path, category in log_files:
            if category not in categories:
                categories[category] = []
            categories[category].append(file_path)
        
        for category, files in categories.items():
            print(f"  📁 {category}: {len(files)}개 파일")
            for i, file_path in enumerate(files[:3], 1):  # 최대 3개만 표시
                file_size = file_path.stat().st_size / (1024*1024)  # MB
                print(f"    {i}. {file_path.name} ({file_size:.1f} MB)")
            if len(files) > 3:
                print(f"    ... 및 {len(files) - 3}개 추가")
        
        return log_files
    
    def validate_log_format(self, log_file: Path) -> Dict:
        """로그 파일의 형식을 검증하고 샘플을 분석합니다."""
        try:
            # 파일 크기 체크
            file_size = log_file.stat().st_size
            if file_size == 0:
                return {'valid': False, 'error': 'Empty file'}
            
            # 처음 10라인 읽기
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    sample_lines.append(line.strip())
            
            if not sample_lines:
                return {'valid': False, 'error': 'No readable lines'}
            
            # 로그 형식 패턴 분석
            patterns = {
                'syslog': r'^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+:',  # Sep 14 05:04:41 host1 kernel:
                'timestamp_iso': r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',      # 2025-09-17 10:15:32
                'timestamp_bracket': r'^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]', # [2025-09-17 10:15:32]
                'apache_combined': r'^\S+\s+\S+\s+\S+\s+\[',                        # Apache combined log
                'json': r'^\s*\{.*\}\s*$',                                         # JSON format
                'generic': r'.',                                                    # Fallback
            }
            
            detected_format = 'unknown'
            for format_name, pattern in patterns.items():
                if any(re.match(pattern, line) for line in sample_lines):
                    detected_format = format_name
                    break
            
            # 통계 수집
            total_lines = sum(1 for _ in open(log_file, 'r', encoding='utf-8', errors='ignore'))
            
            return {
                'valid': True,
                'format': detected_format,
                'file_size_mb': file_size / (1024*1024),
                'total_lines': total_lines,
                'sample_lines': sample_lines[:3],
                'encoding': 'utf-8'  # 기본값
            }
            
        except UnicodeDecodeError:
            # UTF-8 실패시 다른 인코딩 시도
            try:
                with open(log_file, 'r', encoding='latin-1') as f:
                    sample_lines = [f.readline().strip() for _ in range(3)]
                return {
                    'valid': True,
                    'format': 'binary_or_latin1',
                    'file_size_mb': file_size / (1024*1024),
                    'sample_lines': sample_lines,
                    'encoding': 'latin-1',
                    'warning': 'Non-UTF8 encoding detected'
                }
            except Exception:
                return {'valid': False, 'error': 'Encoding issues'}
        
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def preprocess_log_file(self, log_file: Path, category: str = "") -> Dict:
        """단일 로그 파일을 전처리합니다 (디버깅 강화)."""
        file_name = log_file.stem
        if category and category != "root":
            safe_category = re.sub(r'[^\w\-_]', '_', category)
            output_dir = self.work_dir / f"processed_{safe_category}_{file_name}"
        else:
            output_dir = self.work_dir / f"processed_{file_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 전처리 중: {log_file.name} (카테고리: {category or 'root'})")
        
        # 로그 형식 검증
        validation = self.validate_log_format(log_file)
        if not validation['valid']:
            print(f"❌ 로그 형식 오류: {validation['error']}")
            return {
                'success': False,
                'error': f"Log format validation failed: {validation['error']}",
                'file_path': log_file,
                'category': category,
                'output_dir': output_dir
            }
        
        print(f"📋 파일 정보: {validation['file_size_mb']:.1f}MB, {validation.get('total_lines', '?')}라인, 형식: {validation['format']}")
        
        try:
            # study-preprocess 바이너리로 전처리 실행
            cmd = [
                "study-preprocess", "parse",
                "--input", str(log_file),
                "--out-dir", str(output_dir),
                "--drain-state", str(self.work_dir / f"drain_{file_name}.json")
            ]
            
            print(f"🔧 실행 명령어: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ 전처리 실패 (코드: {result.returncode})")
                print(f"📄 표준 출력: {result.stdout}")
                print(f"📄 표준 에러: {result.stderr}")
                return {
                    'success': False,
                    'error': f"Process failed with code {result.returncode}: {result.stderr}",
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'file_path': log_file,
                    'category': category,
                    'output_dir': output_dir,
                    'validation': validation
                }
            
            # 결과 파일 확인
            parsed_file = output_dir / "parsed.parquet"
            if not parsed_file.exists():
                print(f"❌ 파싱 결과 없음: parsed.parquet이 생성되지 않음")
                print(f"📁 출력 디렉토리 내용:")
                try:
                    for item in output_dir.iterdir():
                        print(f"  - {item.name}")
                except:
                    print("  (디렉토리 읽기 실패)")
                
                return {
                    'success': False,
                    'error': 'No parsed.parquet generated',
                    'stdout': result.stdout,
                    'file_path': log_file,
                    'category': category,
                    'output_dir': output_dir,
                    'validation': validation
                }
            
            # 결과 통계 수집
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
                'category': category,
                'output_dir': output_dir,
                'parsed_file': parsed_file,
                'stats': stats,
                'validation': validation
            }
            
        except Exception as e:
            print(f"❌ 예외 발생: {log_file.name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': log_file,
                'category': category,
                'output_dir': output_dir,
                'validation': validation
            }
    
    def select_target_and_baselines(self, log_files: List[Tuple[Path, str]], 
                                  target_file: str = None) -> Tuple[Tuple[Path, str], List[Tuple[Path, str]]]:
        """Target 파일과 Baseline 파일들을 선택합니다."""
        
        if target_file:
            # 절대 경로 또는 상대 경로로 지정된 Target 파일 처리
            target_path = Path(target_file)
            
            # 1. 절대/상대 경로로 지정된 파일이 실제로 존재하는지 확인
            if target_path.exists() and target_path.is_file():
                print(f"🎯 외부 Target 파일 발견: {target_file}")
                # 외부 파일을 Target으로 사용, 카테고리는 부모 디렉토리명
                category = target_path.parent.name
                target = (target_path, category)
                baselines = log_files  # 모든 발견된 로그 파일들을 Baseline으로 사용
            else:
                # 2. 기존 방식: 발견된 로그 파일들 중에서 Target 찾기
                target_matches = [
                    (f, c) for f, c in log_files 
                    if f.name == target_file or str(f) == target_file
                ]
                
                if target_matches:
                    target = target_matches[0]
                    baselines = [(f, c) for f, c in log_files if (f, c) != target]
                else:
                    print(f"❌ 지정된 Target 파일을 찾을 수 없음: {target_file}")
                    print("📋 사용 가능한 파일들:")
                    for i, (f, c) in enumerate(log_files[:10], 1):
                        print(f"  {i}. {f.name} (카테고리: {c})")
                    print(f"\n💡 올바른 Target 파일명을 지정하거나, Target 파일을 생략하여 자동 선택하세요.")
                    raise ValueError(f"Target 파일을 찾을 수 없습니다: {target_file}")
        else:
            # 가장 큰 파일을 Target으로 선택
            target = log_files[0] if log_files else None
            baselines = log_files[1:] if len(log_files) > 1 else []
        
        if target:
            print(f"🎯 Target 파일: {target[0].name} (카테고리: {target[1]})")
            print(f"📊 Baseline 파일: {len(baselines)}개")
            
            # 카테고리별 Baseline 분포
            baseline_categories = {}
            for _, category in baselines:
                baseline_categories[category] = baseline_categories.get(category, 0) + 1
            
            for category, count in baseline_categories.items():
                print(f"  - {category}: {count}개")
        
        return target, baselines
    
    def _validate_baseline_quality(self, baseline_results: List[Dict]) -> List[Dict]:
        """Baseline 파일들의 품질을 검증하고 문제가 있는 것들을 필터링합니다."""
        
        quality_baselines = []
        
        for result in baseline_results:
            try:
                # 기본 통계 로드
                df = pd.read_parquet(result['parsed_file'])
                
                # 품질 검증 기준들
                total_logs = len(df)
                unique_templates = len(df['template_id'].unique()) if 'template_id' in df.columns else 0
                
                # 에러/경고 로그 분석
                error_keywords = ['error', 'ERROR', 'fail', 'FAIL', 'exception', 'Exception', 'panic', 'fatal']
                warning_keywords = ['warn', 'WARN', 'warning', 'WARNING']
                
                error_logs = df[df['raw'].str.contains('|'.join(error_keywords), case=False, na=False)]
                warning_logs = df[df['raw'].str.contains('|'.join(warning_keywords), case=False, na=False)]
                
                error_rate = len(error_logs) / max(total_logs, 1)
                warning_rate = len(warning_logs) / max(total_logs, 1)
                
                # 템플릿 분포 분석
                template_counts = df['template_id'].value_counts() if 'template_id' in df.columns else pd.Series()
                rare_templates = len([t for t, count in template_counts.items() if count == 1])
                rare_template_ratio = rare_templates / max(unique_templates, 1)
                
                # 품질 기준 체크
                quality_issues = []
                
                if error_rate > 0.02:  # 2% 이상 에러율
                    quality_issues.append(f"높은 에러율: {error_rate:.2%}")
                
                if warning_rate > 0.05:  # 5% 이상 경고율
                    quality_issues.append(f"높은 경고율: {warning_rate:.2%}")
                
                if unique_templates < 10:  # 최소 10개 템플릿
                    quality_issues.append(f"템플릿 부족: {unique_templates}개")
                
                if total_logs < 100:  # 최소 100개 로그
                    quality_issues.append(f"로그 수 부족: {total_logs}개")
                
                if rare_template_ratio > 0.3:  # 희귀 템플릿 30% 이상
                    quality_issues.append(f"희귀 템플릿 과다: {rare_template_ratio:.1%}")
                
                # 품질 기준 통과 여부
                if len(quality_issues) <= 1:  # 최대 1개 문제까지 허용
                    quality_baselines.append(result)
                    if quality_issues:
                        print(f"   ⚠️  {result['file_path'].name}: {quality_issues[0]} (경미함)")
                    else:
                        print(f"   ✅ {result['file_path'].name}: 품질 양호")
                else:
                    print(f"   ❌ {result['file_path'].name}: {', '.join(quality_issues)}")
                
            except Exception as e:
                print(f"   ❌ {result['file_path'].name}: 검증 오류 ({e})")
        
        return quality_baselines
    
    def run_log_sample_analysis(self, target_result: Dict) -> Dict:
        """Target 파일의 이상 로그 샘플을 분석합니다."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        print(f"🔍 이상 로그 샘플 분석: {target_result['file_path'].name}")
        
        try:
            # 로그 샘플 분석 실행
            cmd = [
                sys.executable, "log_sample_analyzer.py",
                str(target_result['output_dir']),
                "--output-dir", str(target_result['output_dir'] / "log_samples_analysis"),
                "--max-samples", "20",
                "--context-lines", "3"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ 로그 샘플 분석 실패: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            # 결과 로드
            sample_data_file = target_result['output_dir'] / "log_samples_analysis" / "anomaly_samples.json"
            sample_report_file = target_result['output_dir'] / "log_samples_analysis" / "anomaly_analysis_report.md"
            
            sample_result = {'success': True, 'total_anomalies': 0, 'analysis_summary': {}}
            
            if sample_data_file.exists():
                with open(sample_data_file, 'r') as f:
                    sample_data = json.load(f)
                
                # 요약 통계 계산
                for method, results in sample_data.items():
                    anomaly_count = results.get('anomaly_count', 0)
                    sample_result['total_anomalies'] += anomaly_count
                    sample_result['analysis_summary'][method] = {
                        'anomaly_count': anomaly_count,
                        'analyzed_count': results.get('analyzed_count', 0),
                        'method_description': results.get('method', 'Unknown')
                    }
                
                sample_result['data_file'] = sample_data_file
                sample_result['report_file'] = sample_report_file
            
            print(f"✅ 로그 샘플 분석 완료: 총 {sample_result['total_anomalies']}개 이상 발견")
            return sample_result
            
        except Exception as e:
            print(f"❌ 로그 샘플 분석 예외: {e}")
            return {'success': False, 'error': str(e)}

    def run_cli_report(self, target_result: Dict) -> Dict:
        """CLI 리포트 생성 실행 (로그 샘플 분석 포함)."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        print(f"📄 CLI 리포트 생성: {target_result['file_path'].name}")
        
        try:
            # CLI report 실행 (로그 샘플 분석 포함)
            print("  📊 CLI 리포트 및 로그 샘플 분석 생성 중...")
            
            cmd = [
                sys.executable, "-c", 
                f"""
import sys
sys.path.append('.')
from study_preprocessor.cli import main
import click
from pathlib import Path

# CLI report 실행
ctx = click.Context(main)
ctx.invoke(main.commands['report'], 
          processed_dir=Path('{target_result['output_dir']}'), 
          with_samples=True)  # 로그 샘플 분석을 기본으로 포함
print("CLI report generation completed")
"""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ CLI 리포트 생성 실패: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            # 결과 파일 확인
            report_file = target_result['output_dir'] / "report.md"
            
            if not report_file.exists():
                return {'success': False, 'error': 'CLI report file not generated'}
            
            print(f"  ✅ CLI 리포트 생성 완료: {report_file.name}")
            
            return {
                'success': True,
                'report_file': report_file
            }
            
        except Exception as e:
            print(f"❌ CLI 리포트 생성 예외: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_enhanced_analysis(self, input_dir: str, target_file: str = None, 
                            max_depth: int = 3, max_files: int = 20) -> Dict:
        """향상된 배치 분석을 실행합니다."""
        
        print("🚀 향상된 배치 로그 분석 시작")
        print(f"📂 입력 디렉토리: {input_dir}")
        print(f"🎯 작업 디렉토리: {self.work_dir}")
        print(f"📊 최대 깊이: {max_depth}, 최대 파일 수: {max_files}")
        
        # 1. 로그 파일 재귀 탐색
        log_files = self.find_log_files_recursive(input_dir, max_depth)
        if not log_files:
            print("❌ 로그 파일을 찾을 수 없습니다")
            return {'success': False, 'error': 'No log files found'}
        
        # 2. Target 및 Baseline 선택 (파일 수 제한 전에 수행)
        target_info, baseline_infos = self.select_target_and_baselines(log_files, target_file)
        if not target_info:
            print("❌ Target 파일이 없습니다")
            return {'success': False, 'error': 'No target file'}
        
        # 3. 파일 수 제한 (Target은 항상 포함, Baseline만 제한)
        if len(baseline_infos) > max_files - 1:  # Target 1개 제외하고 제한
            print(f"⚠️  Baseline 파일이 많아 상위 {max_files - 1}개만 분석합니다")
            baseline_infos = baseline_infos[:max_files - 1]
        
        # 4. 전처리 실행
        print(f"\n{'='*60}")
        print("📋 전처리 단계")
        print(f"{'='*60}")
        
        # Target 전처리
        target_result = self.preprocess_log_file(target_info[0], target_info[1])
        
        # Baseline 전처리
        baseline_results = []
        for baseline_file, baseline_category in baseline_infos:
            result = self.preprocess_log_file(baseline_file, baseline_category)
            baseline_results.append(result)
        
        # 5. 성공한 파일들만으로 분석 진행
        successful_baselines = [r for r in baseline_results if r['success']]
        
        print(f"\n📊 전처리 결과 요약:")
        print(f"  - Target: {'✅' if target_result['success'] else '❌'} {target_info[0].name}")
        print(f"  - Baseline 성공: {len(successful_baselines)}/{len(baseline_results)}개")
        
        # 6. Target 파일 Full Pipeline 분석 실행
        target_baseline_result = {'success': False, 'error': 'Target preprocessing failed'}
        target_deeplog_result = {'success': False, 'error': 'Target preprocessing failed'}
        target_mscred_result = {'success': False, 'error': 'Target preprocessing failed'}
        target_temporal_result = {'success': False, 'error': 'Target preprocessing failed'}
        comparative_result = {'success': False, 'error': 'Target preprocessing failed'}
        
        if target_result['success']:
            print(f"\n{'='*60}")
            print(f"🎯 Target 파일 분석: {target_result['file_path'].name}")
            print(f"{'='*60}")
            
            # Target Baseline 이상 탐지
            print("📈 Baseline 이상 탐지 중...")
            target_baseline_result = self.run_baseline_analysis(target_result)
            
            # Target DeepLog 분석 
            print("🧠 DeepLog 딥러닝 분석 중...")
            target_deeplog_result = self.run_deeplog_analysis(target_result)
            
            # Target MS-CRED 입력 생성
            print("📊 MS-CRED 입력 생성 중...")
            target_mscred_build_result = self.run_mscred_build(target_result)
            
            # Target MS-CRED 학습 및 추론
            target_mscred_result = {'success': False, 'error': 'MS-CRED build failed'}
            if target_mscred_build_result['success']:
                print("🔬 MS-CRED 학습 및 이상탐지 중...")
                target_mscred_result = self.run_mscred_analysis(target_result)
            
            # Target 시간 기반 분석
            print("🕐 시간 기반 이상 탐지 중...")
            target_temporal_result = self.run_temporal_analysis(target_result)
        
        # 7. Baseline 파일들 개별 분석 실행
        baseline_analysis_results = []
        for i, baseline_result in enumerate(successful_baselines):
            print(f"\n{'='*60}")
            print(f"📂 Baseline 파일 분석 {i+1}/{len(successful_baselines)}: {baseline_result['file_path'].name}")
            print(f"{'='*60}")
            
            baseline_analysis = {
                'file_info': baseline_result,
                'baseline_result': {'success': False},
                'deeplog_result': {'success': False},
                'mscred_result': {'success': False},
                'temporal_result': {'success': False}
            }
            
            # Baseline 이상 탐지
            print("📈 Baseline 이상 탐지 중...")
            baseline_analysis['baseline_result'] = self.run_baseline_analysis(baseline_result)
            
            # DeepLog 분석
            print("🧠 DeepLog 딥러닝 분석 중...")
            baseline_analysis['deeplog_result'] = self.run_deeplog_analysis(baseline_result)
            
            # MS-CRED 입력 생성 및 분석
            print("📊 MS-CRED 입력 생성 중...")
            mscred_build_result = self.run_mscred_build(baseline_result)
            
            if mscred_build_result['success']:
                print("🔬 MS-CRED 학습 및 이상탐지 중...")
                baseline_analysis['mscred_result'] = self.run_mscred_analysis(baseline_result)
            else:
                baseline_analysis['mscred_result'] = {'success': False, 'error': 'MS-CRED build failed'}
            
            # 시간 기반 분석
            print("🕐 시간 기반 이상 탐지 중...")
            baseline_analysis['temporal_result'] = self.run_temporal_analysis(baseline_result)
            
            baseline_analysis_results.append(baseline_analysis)
        
        # 8. 파일별 비교 분석 (모든 파일 대상)
        if successful_baselines and target_result['success']:
            print(f"\n{'='*60}")
            print("📊 파일별 비교 이상 탐지")
            print(f"{'='*60}")
            comparative_result = self.run_comparative_analysis(target_result, successful_baselines)
        
        # 9. CLI 리포트 생성 (로그 샘플 분석 포함)
        print(f"\n{'='*60}")
        print("📄 CLI 리포트 및 로그 샘플 분석")
        print(f"{'='*60}")
        
        # Target CLI 리포트
        target_cli_report_result = {'success': False, 'error': 'Target preprocessing failed'}
        if target_result['success']:
            print(f"📄 Target CLI 리포트: {target_result['file_path'].name}")
            target_cli_report_result = self.run_cli_report(target_result)
        
        # Baseline CLI 리포트들
        baseline_cli_reports = []
        for i, baseline_analysis in enumerate(baseline_analysis_results):
            baseline_file_info = baseline_analysis['file_info']
            print(f"📄 Baseline CLI 리포트 {i+1}/{len(baseline_analysis_results)}: {baseline_file_info['file_path'].name}")
            cli_report = self.run_cli_report(baseline_file_info)
            baseline_cli_reports.append(cli_report)
        
        # 10. 종합 리포트 생성
        print(f"\n{'='*60}")
        print("📄 종합 리포트 생성")
        print(f"{'='*60}")
        
        summary_report = self.generate_comprehensive_report(
            target_result, baseline_results, target_baseline_result, target_deeplog_result, target_mscred_result, 
            target_temporal_result, comparative_result, target_cli_report_result, input_dir, max_depth,
            baseline_analysis_results, baseline_cli_reports
        )
        
        comprehensive_report_file = self.work_dir / "COMPREHENSIVE_ANALYSIS_REPORT.md"
        with open(comprehensive_report_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        # 기존 요약 파일도 호환성을 위해 생성
        legacy_summary_file = self.work_dir / "ENHANCED_ANALYSIS_SUMMARY.md"
        with open(legacy_summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"📋 종합 리포트 저장: {comprehensive_report_file}")
        print(f"📋 호환성 리포트: {legacy_summary_file}")
        
        # 결과 요약 출력
        print(f"\n{'='*60}")
        print("✅ 향상된 배치 분석 완료")
        print(f"{'='*60}")
        
        success_count = len([r for r in baseline_results if r['success']]) + (1 if target_result['success'] else 0)
        total_count = len(baseline_results) + 1
        print(f"📊 전처리 성공: {success_count}/{total_count}개 파일")
        
        # Target 분석 결과 요약
        if target_baseline_result['success']:
            baseline_anomalies = target_baseline_result['anomaly_windows']
            baseline_rate = target_baseline_result['anomaly_rate']
            print(f"🎯 Target Baseline 이상: {baseline_anomalies}개 윈도우 ({baseline_rate:.1%})")
        
        if target_deeplog_result['success']:
            deeplog_violations = target_deeplog_result.get('violations', 0)
            deeplog_total = target_deeplog_result.get('total_sequences', 0)
            print(f"🎯 Target DeepLog 위반: {deeplog_violations}/{deeplog_total}개 시퀀스")
        
        if target_mscred_result['success']:
            mscred_anomalies = target_mscred_result.get('anomalies', 0)
            mscred_total = target_mscred_result.get('total_windows', 0)
            print(f"🎯 Target MS-CRED 이상: {mscred_anomalies}/{mscred_total}개 윈도우")
        
        if target_temporal_result['success']:
            temporal_anomalies = len(target_temporal_result.get('anomalies', []))
            print(f"🎯 Target 시간 기반 이상: {temporal_anomalies}개")
        
        # Baseline 분석 결과 요약
        baseline_total_anomalies = 0
        baseline_successful_analyses = 0
        for baseline_analysis in baseline_analysis_results:
            if baseline_analysis['baseline_result']['success']:
                baseline_total_anomalies += baseline_analysis['baseline_result'].get('anomaly_windows', 0)
                baseline_successful_analyses += 1
        
        if baseline_successful_analyses > 0:
            print(f"📂 Baseline 파일들 이상: 총 {baseline_total_anomalies}개 윈도우 ({baseline_successful_analyses}개 파일)")
        
        if comparative_result['success']:
            comp_anomalies = len(comparative_result.get('anomalies', []))
            print(f"📊 비교 분석 이상: {comp_anomalies}개")
        
        if target_cli_report_result['success']:
            print(f"📄 Target CLI 리포트: {target_cli_report_result['report_file']}")
        
        print(f"📄 종합 리포트: {comprehensive_report_file}")
        print(f"📊 분석된 파일: Target 1개 + Baseline {len(baseline_analysis_results)}개 = 총 {1 + len(baseline_analysis_results)}개")
        
        return {
            'success': True,
            'target_result': target_result,
            'baseline_results': baseline_results,
            'target_baseline_result': target_baseline_result,
            'target_deeplog_result': target_deeplog_result,
            'target_mscred_result': target_mscred_result,
            'target_temporal_result': target_temporal_result,
            'baseline_analysis_results': baseline_analysis_results,
            'comparative_result': comparative_result,
            'target_cli_report_result': target_cli_report_result,
            'baseline_cli_reports': baseline_cli_reports,
            'comprehensive_report_file': comprehensive_report_file,
            'summary_file': legacy_summary_file,  # 호환성
            'total_files_found': len(log_files),
            'files_processed': total_count
        }
    
    def run_baseline_analysis(self, target_result: Dict) -> Dict:
        """Baseline 이상 탐지 실행."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        print(f"📈 Baseline 이상 탐지: {target_result['file_path'].name}")
        
        try:
            parsed_file = target_result['output_dir'] / "parsed.parquet"
            
            # Baseline 이상 탐지 실행
            print("  📊 Window 기반 이상 탐지 중...")
            
            cmd = [
                sys.executable, "-c", 
                f"""
import sys
sys.path.append('.')
from study_preprocessor.detect import baseline_detect
from study_preprocessor.detect import BaselineParams

# Baseline 이상 탐지 실행
baseline_detect(
    parsed_parquet='{parsed_file}',
    out_dir='{target_result['output_dir']}',
    params=BaselineParams(
        window_size=50,
        stride=25,
        ewm_alpha=0.3,
        anomaly_quantile=0.95
    )
)
print("Baseline detection completed")
"""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ Baseline 이상 탐지 실패: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            # 결과 파일 확인
            baseline_scores_file = target_result['output_dir'] / "baseline_scores.parquet"
            baseline_preview_file = target_result['output_dir'] / "baseline_preview.json"
            
            if not baseline_scores_file.exists():
                return {'success': False, 'error': 'Baseline scores file not generated'}
            
            # 결과 요약
            import pandas as pd
            scores_df = pd.read_parquet(baseline_scores_file)
            total_windows = len(scores_df)
            anomaly_windows = len(scores_df[scores_df['is_anomaly'] == True])
            anomaly_rate = anomaly_windows / total_windows if total_windows > 0 else 0
            
            print(f"  ✅ Baseline 이상 탐지 완료: {anomaly_windows}/{total_windows} 윈도우 이상 ({anomaly_rate:.1%})")
            
            return {
                'success': True,
                'scores_file': baseline_scores_file,
                'preview_file': baseline_preview_file,
                'total_windows': total_windows,
                'anomaly_windows': anomaly_windows,
                'anomaly_rate': anomaly_rate
            }
            
        except Exception as e:
            print(f"❌ Baseline 이상 탐지 예외: {e}")
            return {'success': False, 'error': str(e)}

    def run_deeplog_analysis(self, target_result: Dict) -> Dict:
        """DeepLog 학습 및 추론 실행."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        print(f"🧠 DeepLog 분석: {target_result['file_path'].name}")
        
        try:
            parsed_file = target_result['output_dir'] / "parsed.parquet"
            
            # 1. DeepLog 입력 생성 (vocab.json, sequences.parquet)
            print("  📊 DeepLog 입력 생성 중...")
            
            try:
                from study_preprocessor.builders.deeplog import build_deeplog_inputs
                build_deeplog_inputs(str(parsed_file), str(target_result['output_dir']))
                
                # 필수 파일 확인
                vocab_file = target_result['output_dir'] / "vocab.json"
                sequences_file = target_result['output_dir'] / "sequences.parquet"
                
                if not vocab_file.exists() or not sequences_file.exists():
                    return {'success': False, 'error': 'DeepLog input files not generated'}
                
                print(f"  ✅ 입력 파일 생성: vocab.json, sequences.parquet")
                
            except Exception as e:
                print(f"❌ DeepLog 입력 생성 실패: {e}")
                return {'success': False, 'error': f'DeepLog input build failed: {e}'}
            
            # 2. DeepLog 학습
            print("  🎯 DeepLog 모델 학습 중...")
            model_path = target_result['output_dir'] / "deeplog.pth"
            
            try:
                from study_preprocessor.builders.deeplog import train_deeplog
                train_deeplog(
                    str(sequences_file),
                    str(vocab_file), 
                    str(model_path),
                    seq_len=50,
                    epochs=2  # 빠른 실행을 위해 2로 감소
                )
                
                if not model_path.exists():
                    return {'success': False, 'error': 'DeepLog model file not generated'}
                
                print(f"  ✅ 모델 학습 완료: {model_path.name}")
                
            except Exception as e:
                print(f"❌ DeepLog 학습 실패: {e}")
                return {'success': False, 'error': f'DeepLog training failed: {e}'}
            
            # 3. DeepLog 추론 (메모리 최적화를 위해 조건부 실행)
            print("  🔍 DeepLog 추론 중...")
            
            try:
                # 시퀀스 크기 확인
                import pandas as pd
                df_check = pd.read_parquet(sequences_file)
                total_sequences = len(df_check)
                
                # 메모리 절약을 위해 큰 파일은 샘플링
                if total_sequences > 50000:
                    print(f"  ⚠️ 대용량 시퀀스 ({total_sequences:,}개) - 샘플링하여 추론")
                    sample_df = df_check.sample(n=min(50000, total_sequences), random_state=42)
                    sample_file = target_result['output_dir'] / "sequences_sample.parquet"
                    sample_df.to_parquet(sample_file, index=False)
                    inference_input = str(sample_file)
                else:
                    inference_input = str(sequences_file)
                
                from study_preprocessor.builders.deeplog import infer_deeplog_topk
                infer_df = infer_deeplog_topk(inference_input, str(model_path), k=3)
                
                # 결과 저장
                infer_file = target_result['output_dir'] / "deeplog_infer.parquet"
                infer_df.to_parquet(infer_file, index=False)
                
                if not infer_file.exists():
                    return {'success': False, 'error': 'DeepLog inference file not generated'}
                
                print(f"  ✅ 추론 완료: {len(infer_df):,}개 시퀀스 분석")
                
            except Exception as e:
                print(f"❌ DeepLog 추론 실패: {e}")
                return {'success': False, 'error': f'DeepLog inference failed: {e}'}
            
            # 결과 요약
            import pandas as pd
            deeplog_df = pd.read_parquet(infer_file)
            violations = deeplog_df[deeplog_df['in_topk'] == False]
            violation_rate = len(violations) / len(deeplog_df) if len(deeplog_df) > 0 else 0
            
            print(f"✅ DeepLog 분석 완료: 위반율 {violation_rate:.1%} ({len(violations)}/{len(deeplog_df)})")
            
            return {
                'success': True,
                'model_path': model_path,
                'inference_file': infer_file,
                'total_sequences': len(deeplog_df),
                'violations': len(violations),
                'violation_rate': violation_rate
            }
            
        except Exception as e:
            print(f"❌ DeepLog 분석 예외: {e}")
            return {'success': False, 'error': str(e)}

    def run_mscred_build(self, target_result: Dict) -> Dict:
        """MS-CRED 입력 생성 실행."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        print(f"📊 MS-CRED 입력 생성: {target_result['file_path'].name}")
        
        try:
            parsed_file = target_result['output_dir'] / "parsed.parquet"
            
            # MS-CRED 입력 생성
            print("  📊 윈도우 카운트 벡터 생성 중...")
            
            cmd = [
                sys.executable, "-c", 
                f"""
import sys
sys.path.append('.')
from study_preprocessor.builders.mscred import build_mscred_window_counts

# MS-CRED 입력 생성
build_mscred_window_counts(
    parsed_parquet='{parsed_file}',
    out_dir='{target_result['output_dir']}',
    window_size=50,
    stride=25
)
print("MS-CRED input generation completed")
"""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ MS-CRED 입력 생성 실패: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            # 결과 파일 확인
            window_counts_file = target_result['output_dir'] / "window_counts.parquet"
            
            if not window_counts_file.exists():
                return {'success': False, 'error': 'MS-CRED window counts file not generated'}
            
            # 결과 요약
            import pandas as pd
            counts_df = pd.read_parquet(window_counts_file)
            num_windows = len(counts_df)
            num_templates = len(counts_df.columns) - 1 if 'window_start' in counts_df.columns else len(counts_df.columns)
            
            print(f"  ✅ MS-CRED 입력 생성 완료: {num_windows}개 윈도우 × {num_templates}개 템플릿")
            
            return {
                'success': True,
                'window_counts_file': window_counts_file,
                'num_windows': num_windows,
                'num_templates': num_templates
            }
            
        except Exception as e:
            print(f"❌ MS-CRED 입력 생성 예외: {e}")
            return {'success': False, 'error': str(e)}

    def run_mscred_analysis(self, target_result: Dict) -> Dict:
        """MS-CRED 학습 및 추론 실행."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        print(f"🧠 MS-CRED 학습/추론: {target_result['file_path'].name}")
        
        try:
            output_dir = target_result['output_dir']
            window_counts_file = output_dir / "window_counts.parquet"
            
            if not window_counts_file.exists():
                return {'success': False, 'error': 'Window counts file not found'}
            
            # MS-CRED 모델 경로
            model_file = self.work_dir / f"mscred_{target_result['file_path'].stem}.pth"
            infer_file = output_dir / "mscred_infer.parquet"
            
            # 1. MS-CRED 학습
            print("  🧠 MS-CRED 모델 학습 중...")
            
            cmd = [
                sys.executable, "-c", 
                f"""
import sys
sys.path.append('.')
from study_preprocessor.mscred_model import train_mscred

# MS-CRED 학습
try:
    stats = train_mscred(
        window_counts_path='{window_counts_file}',
        model_output_path='{model_file}',
        epochs=30  # 배치 분석용으로 적당한 에포크
    )
    print(f"학습 완료 - 최종 손실: {{stats['final_train_loss']:.4f}}")
except Exception as e:
    print(f"학습 실패: {{e}}")
    raise
"""
            ]
            
            result = subprocess.run(cmd, cwd=self.work_dir, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"❌ MS-CRED 학습 실패:")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
                return {'success': False, 'error': 'MS-CRED training failed'}
            
            if not model_file.exists():
                return {'success': False, 'error': 'MS-CRED model not created'}
            
            print(f"  ✅ 모델 학습 완료: {model_file}")
            
            # 2. MS-CRED 추론
            print("  🔍 MS-CRED 이상탐지 추론 중...")
            
            cmd = [
                sys.executable, "-c", 
                f"""
import sys
sys.path.append('.')
from study_preprocessor.mscred_model import infer_mscred

# MS-CRED 추론
try:
    results_df = infer_mscred(
        window_counts_path='{window_counts_file}',
        model_path='{model_file}',
        output_path='{infer_file}',
        threshold_percentile=95.0
    )
    print(f"추론 완료 - 이상탐지율: {{results_df['is_anomaly'].mean():.1%}}")
except Exception as e:
    print(f"추론 실패: {{e}}")
    raise
"""
            ]
            
            result = subprocess.run(cmd, cwd=self.work_dir, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"❌ MS-CRED 추론 실패:")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
                return {'success': False, 'error': 'MS-CRED inference failed'}
            
            if not infer_file.exists():
                return {'success': False, 'error': 'MS-CRED inference file not created'}
            
            print(f"  ✅ 추론 완료: {infer_file}")
            
            # 결과 요약
            import pandas as pd
            mscred_df = pd.read_parquet(infer_file)
            anomalies = mscred_df[mscred_df['is_anomaly'] == True]
            anomaly_rate = len(anomalies) / len(mscred_df) if len(mscred_df) > 0 else 0
            
            print(f"✅ MS-CRED 분석 완료: 이상탐지율 {anomaly_rate:.1%} ({len(anomalies)}/{len(mscred_df)})")
            
            return {
                'success': True,
                'model_path': model_file,
                'inference_file': infer_file,
                'total_windows': len(mscred_df),
                'anomalies': len(anomalies),
                'anomaly_rate': anomaly_rate
            }
            
        except Exception as e:
            print(f"❌ MS-CRED 분석 예외: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_temporal_analysis(self, target_result: Dict) -> Dict:
        """시간 기반 이상 탐지 실행 (기존과 동일)."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        print(f"🕐 시간 기반 이상 탐지: {target_result['file_path'].name}")
        
        try:
            cmd = [
                sys.executable, "temporal_anomaly_detector.py",
                "--data-dir", str(target_result['output_dir']),
                "--output-dir", str(target_result['output_dir'] / "temporal_analysis")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ 시간 분석 실패: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
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
        """파일별 비교 분석 실행 (baseline 품질 검증 추가)."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        valid_baselines = [r for r in baseline_results if r['success']]
        if not valid_baselines:
            print("⚠️ 비교할 baseline 파일이 없습니다")
            return {'success': False, 'error': 'No valid baseline files'}
        
        # Baseline 품질 검증 추가
        print(f"🔍 {len(valid_baselines)}개 baseline 파일 품질 검증 중...")
        validated_baselines = self._validate_baseline_quality(valid_baselines)
        
        if len(validated_baselines) < len(valid_baselines):
            filtered_count = len(valid_baselines) - len(validated_baselines)
            print(f"⚠️  품질 문제로 {filtered_count}개 baseline 파일 제외됨")
        
        if not validated_baselines:
            print("❌ 품질 기준을 만족하는 baseline 파일이 없습니다")
            return {'success': False, 'error': 'No quality baselines after validation'}
        
        print(f"📊 파일별 비교 분석: {target_result['file_path'].name} vs {len(validated_baselines)}개 검증된 파일")
        
        try:
            baseline_paths = [str(r['parsed_file']) for r in validated_baselines]
            
            cmd = [
                sys.executable, "comparative_anomaly_detector.py",
                "--target", str(target_result['parsed_file']),
                "--baselines"] + baseline_paths + [
                "--output-dir", str(target_result['output_dir'] / "comparative_analysis")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ 비교 분석 실패: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            comp_dir = target_result['output_dir'] / "comparative_analysis"
            anomalies_file = comp_dir / "comparative_anomalies.json"
            
            comp_result = {'success': True, 'anomalies': [], 'baseline_count': len(validated_baselines)}
            if anomalies_file.exists():
                with open(anomalies_file) as f:
                    comp_result['anomalies'] = json.load(f)
            
            print(f"✅ 비교 분석 완료: {len(comp_result['anomalies'])}개 이상 발견")
            return comp_result
            
        except Exception as e:
            print(f"❌ 비교 분석 예외: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_comprehensive_report(self, target_result: Dict, baseline_results: List[Dict],
                                     target_baseline_result: Dict, target_deeplog_result: Dict, target_mscred_result: Dict, 
                                     target_temporal_result: Dict, comparative_result: Dict, target_cli_report_result: Dict, 
                                     input_dir: str, max_depth: int, baseline_analysis_results: List[Dict] = None, 
                                     baseline_cli_reports: List[Dict] = None) -> str:
        """종합 통합 리포트 생성 - 모든 분석 결과를 하나의 리포트로 통합."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_name = target_result['file_path'].name
        
        report = f"""# 향상된 배치 로그 분석 리포트

**분석 시간**: {timestamp}  
**입력 디렉토리**: {input_dir}  
**스캔 깊이**: {max_depth}  
**Target 파일**: {target_name}  
**발견된 파일**: {len(baseline_results) + 1}개  

## 📂 디렉토리 구조 분석

### Target 파일: {target_name}
"""
        
        if target_result['success']:
            stats = target_result['stats']
            validation = target_result.get('validation', {})
            
            # 안전한 시간 범위 처리
            time_range = stats.get('time_range', {})
            start_time = time_range.get('start') if time_range else None
            end_time = time_range.get('end') if time_range else None
            if start_time and end_time and start_time != 'None' and end_time != 'None':
                time_range_str = f"{start_time} ~ {end_time}"
            else:
                time_range_str = "시간 정보 없음"
            
            # 안전한 호스트 정보 처리
            hosts = stats.get('hosts', [])
            if hosts and any(h for h in hosts if h is not None):
                valid_hosts = [str(h) for h in hosts if h is not None]
                hosts_str = f"{len(valid_hosts)}개 ({', '.join(valid_hosts[:3])}{'...' if len(valid_hosts) > 3 else ''})"
            else:
                hosts_str = "호스트 정보 없음"
            
            report += f"""- ✅ **성공**: {stats['total_logs']:,}개 로그, {stats['unique_templates']}개 템플릿
- **카테고리**: {target_result.get('category', 'root')}
- **파일 크기**: {validation.get('file_size_mb', 0):.1f}MB
- **로그 형식**: {validation.get('format', 'unknown')}
- **시간 범위**: {time_range_str}
- **호스트**: {hosts_str}
"""
        else:
            validation = target_result.get('validation', {})
            report += f"""- ❌ **실패**: {target_result['error']}
- **카테고리**: {target_result.get('category', 'root')}
- **파일 크기**: {validation.get('file_size_mb', 0):.1f}MB
- **로그 형식**: {validation.get('format', 'unknown')}
"""
        
        # 카테고리별 결과 정리
        categories = {}
        for result in baseline_results:
            category = result.get('category', 'root')
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        report += "\n### Baseline 파일들 (카테고리별)\n"
        for category, results in categories.items():
            report += f"\n#### 📁 {category}\n"
            for i, result in enumerate(results, 1):
                if result['success']:
                    stats = result['stats']
                    validation = result.get('validation', {})
                    report += f"{i}. ✅ **{result['file_path'].name}**: {stats['total_logs']:,}개 로그, {stats['unique_templates']}개 템플릿 ({validation.get('file_size_mb', 0):.1f}MB)\n"
                else:
                    validation = result.get('validation', {})
                    report += f"{i}. ❌ **{result['file_path'].name}**: {result['error']} ({validation.get('file_size_mb', 0):.1f}MB)\n"
        
        # Target Baseline 결과 추가
        report += "\n## 📈 Target 파일 Baseline 이상 탐지 결과\n\n"
        if target_baseline_result['success']:
            total_windows = target_baseline_result['total_windows']
            anomaly_windows = target_baseline_result['anomaly_windows']
            anomaly_rate = target_baseline_result['anomaly_rate']
            
            report += f"**분석 윈도우**: {total_windows:,}개\n"
            report += f"**이상 윈도우**: {anomaly_windows:,}개\n"
            report += f"**이상율**: {anomaly_rate:.1%}\n\n"
            
            if anomaly_rate > 0.1:  # 10% 이상
                report += "🚨 **높은 이상율**: 다수의 윈도우에서 이상 패턴이 감지되었습니다.\n"
            elif anomaly_rate > 0.05:  # 5% 이상
                report += "⚠️ **중간 이상율**: 일부 윈도우에서 이상 패턴이 감지되었습니다.\n"
            else:
                report += "✅ **낮은 이상율**: 대부분 정상적인 로그 패턴입니다.\n"
        else:
            report += f"❌ **Target Baseline 분석 실패**: {target_baseline_result['error']}\n"
        
        # Baseline 파일들 Baseline 결과 추가
        if baseline_analysis_results:
            report += "\n## 📂 Baseline 파일들 이상 탐지 결과\n\n"
            baseline_total_anomalies = 0
            baseline_total_windows = 0
            baseline_successful = 0
            
            for i, baseline_analysis in enumerate(baseline_analysis_results, 1):
                file_info = baseline_analysis['file_info']
                baseline_result = baseline_analysis['baseline_result']
                
                report += f"### {i}. {file_info['file_path'].name}\n"
                if baseline_result['success']:
                    anomaly_windows = baseline_result['anomaly_windows']
                    total_windows = baseline_result['total_windows']
                    anomaly_rate = baseline_result['anomaly_rate']
                    
                    baseline_total_anomalies += anomaly_windows
                    baseline_total_windows += total_windows
                    baseline_successful += 1
                    
                    report += f"- 이상 윈도우: {anomaly_windows:,}/{total_windows:,}개 ({anomaly_rate:.1%})\n"
                    if anomaly_rate > 0.1:
                        report += "- 🚨 높은 이상율 감지\n"
                    elif anomaly_rate > 0.05:
                        report += "- ⚠️ 중간 이상율 감지\n"
                    else:
                        report += "- ✅ 정상 수준\n"
                else:
                    report += f"- ❌ 분석 실패: {baseline_result['error']}\n"
                report += "\n"
            
            if baseline_successful > 0:
                overall_rate = baseline_total_anomalies / max(baseline_total_windows, 1)
                report += f"**전체 Baseline 파일 요약**: {baseline_total_anomalies:,}/{baseline_total_windows:,}개 이상 윈도우 ({overall_rate:.1%})\n\n"
        
        # Target DeepLog 결과 추가
        report += "\n## 🧠 Target 파일 DeepLog 딥러닝 분석 결과\n\n"
        if target_deeplog_result['success']:
            total_sequences = target_deeplog_result['total_sequences']
            violations = target_deeplog_result['violations']
            violation_rate = target_deeplog_result['violation_rate']
            
            report += f"**전체 시퀀스**: {total_sequences:,}개\n"
            report += f"**예측 실패**: {violations:,}개\n"
            report += f"**위반율**: {violation_rate:.1%}\n\n"
            
            if violation_rate > 0.5:  # 50% 이상
                report += "🚨 **높은 위반율**: 로그 패턴이 매우 예측하기 어려운 상태입니다.\n"
            elif violation_rate > 0.2:  # 20% 이상
                report += "🔍 **중간 위반율**: 일부 예측 어려운 로그 패턴이 존재합니다.\n"
            else:
                report += "✅ **낮은 위반율**: 대부분 예측 가능한 로그 패턴입니다.\n"
        else:
            report += f"❌ Target DeepLog 분석 실패: {target_deeplog_result.get('error', 'Unknown error')}\n"
        
        # Baseline 파일들 DeepLog 결과 추가
        if baseline_analysis_results:
            report += "\n## 📂 Baseline 파일들 DeepLog 분석 결과\n\n"
            deeplog_total_violations = 0
            deeplog_total_sequences = 0
            deeplog_successful = 0
            
            for i, baseline_analysis in enumerate(baseline_analysis_results, 1):
                file_info = baseline_analysis['file_info']
                deeplog_result = baseline_analysis['deeplog_result']
                
                report += f"### {i}. {file_info['file_path'].name}\n"
                if deeplog_result['success']:
                    violations = deeplog_result['violations']
                    total_sequences = deeplog_result['total_sequences']
                    violation_rate = deeplog_result['violation_rate']
                    
                    deeplog_total_violations += violations
                    deeplog_total_sequences += total_sequences
                    deeplog_successful += 1
                    
                    report += f"- 예측 실패: {violations:,}/{total_sequences:,}개 시퀀스 ({violation_rate:.1%})\n"
                    if violation_rate > 0.1:
                        report += "- 🚨 높은 위반율 감지\n"
                    elif violation_rate > 0.05:
                        report += "- ⚠️ 중간 위반율 감지\n"
                    else:
                        report += "- ✅ 정상 수준\n"
                else:
                    report += f"- ❌ 분석 실패: {deeplog_result['error']}\n"
                report += "\n"
            
            if deeplog_successful > 0:
                overall_violation_rate = deeplog_total_violations / max(deeplog_total_sequences, 1)
                report += f"**전체 Baseline 파일 요약**: {deeplog_total_violations:,}/{deeplog_total_sequences:,}개 위반 시퀀스 ({overall_violation_rate:.1%})\n\n"
        
        # Target MS-CRED 결과
        report += "\n## 🔬 Target 파일 MS-CRED 멀티스케일 분석 결과\n\n"
        if target_mscred_result['success']:
            total_windows = target_mscred_result['total_windows']
            anomalies = target_mscred_result['anomalies']
            anomaly_rate = target_mscred_result['anomaly_rate']
            
            report += f"**전체 윈도우**: {total_windows:,}개\n"
            report += f"**이상 윈도우**: {anomalies:,}개\n"
            report += f"**이상탐지율**: {anomaly_rate:.1%}\n\n"
            
            if anomaly_rate > 0.2:  # 20% 이상
                report += "🚨 **높은 이상률**: 많은 윈도우에서 비정상적인 패턴이 감지되었습니다.\n"
            elif anomaly_rate > 0.05:  # 5% 이상
                report += "🔍 **중간 이상률**: 일부 윈도우에서 주목할 만한 패턴 변화가 있습니다.\n"
            else:
                report += "✅ **낮은 이상률**: 대부분 정상적인 로그 패턴을 보입니다.\n"
        else:
            report += f"❌ Target MS-CRED 분석 실패: {target_mscred_result.get('error', 'Unknown error')}\n"
        
        # Baseline 파일들 MS-CRED 결과 추가
        if baseline_analysis_results:
            report += "\n## 📂 Baseline 파일들 MS-CRED 분석 결과\n\n"
            mscred_total_anomalies = 0
            mscred_total_windows = 0
            mscred_successful = 0
            
            for i, baseline_analysis in enumerate(baseline_analysis_results, 1):
                file_info = baseline_analysis['file_info']
                mscred_result = baseline_analysis['mscred_result']
                
                report += f"### {i}. {file_info['file_path'].name}\n"
                if mscred_result['success']:
                    anomalies = mscred_result['anomalies']
                    total_windows = mscred_result['total_windows']
                    anomaly_rate = mscred_result['anomaly_rate']
                    
                    mscred_total_anomalies += anomalies
                    mscred_total_windows += total_windows
                    mscred_successful += 1
                    
                    report += f"- 이상 윈도우: {anomalies:,}/{total_windows:,}개 ({anomaly_rate:.1%})\n"
                    if anomaly_rate > 0.2:
                        report += "- 🚨 높은 이상률 감지\n"
                    elif anomaly_rate > 0.05:
                        report += "- ⚠️ 중간 이상률 감지\n"
                    else:
                        report += "- ✅ 정상 수준\n"
                else:
                    report += f"- ❌ 분석 실패: {mscred_result['error']}\n"
                report += "\n"
            
            if mscred_successful > 0:
                overall_mscred_rate = mscred_total_anomalies / max(mscred_total_windows, 1)
                report += f"**전체 Baseline 파일 요약**: {mscred_total_anomalies:,}/{mscred_total_windows:,}개 이상 윈도우 ({overall_mscred_rate:.1%})\n\n"
        
        # Target 시간 기반 분석
        report += "\n## 🕐 Target 파일 시간 기반 이상 탐지 결과\n\n"
        if target_temporal_result['success']:
            anomalies = target_temporal_result['anomalies']
            if anomalies:
                high_count = len([a for a in anomalies if a.get('severity') == 'high'])
                medium_count = len([a for a in anomalies if a.get('severity') == 'medium'])
                report += f"🚨 **발견된 이상**: {len(anomalies)}개 (심각: {high_count}개, 주의: {medium_count}개)\n\n"
                
                for anomaly in anomalies[:5]:
                    report += f"- **{anomaly.get('type', 'unknown')}** ({anomaly.get('hour', '?')}시): {anomaly.get('description', 'No description')}\n"
                
                if len(anomalies) > 5:
                    report += f"- ... 및 {len(anomalies) - 5}개 추가\n"
            else:
                report += "✅ 시간 기반 이상 현상이 발견되지 않았습니다.\n"
        else:
            report += f"❌ Target 시간 기반 분석 실패: {target_temporal_result.get('error', 'Unknown error')}\n"
        
        # Baseline 파일들 시간 기반 분석 결과 추가
        if baseline_analysis_results:
            report += "\n## 📂 Baseline 파일들 시간 기반 분석 결과\n\n"
            temporal_total_anomalies = 0
            temporal_successful = 0
            
            for i, baseline_analysis in enumerate(baseline_analysis_results, 1):
                file_info = baseline_analysis['file_info']
                temporal_result = baseline_analysis['temporal_result']
                
                report += f"### {i}. {file_info['file_path'].name}\n"
                if temporal_result['success']:
                    anomalies = temporal_result['anomalies']
                    temporal_total_anomalies += len(anomalies)
                    temporal_successful += 1
                    
                    if anomalies:
                        high_count = len([a for a in anomalies if a.get('severity') == 'high'])
                        medium_count = len([a for a in anomalies if a.get('severity') == 'medium'])
                        report += f"- 발견된 이상: {len(anomalies)}개 (심각: {high_count}개, 주의: {medium_count}개)\n"
                        if high_count > 0:
                            report += "- 🚨 심각한 시간 패턴 이상 감지\n"
                        elif medium_count > 0:
                            report += "- ⚠️ 중간 수준 시간 패턴 이상 감지\n"
                        else:
                            report += "- ✅ 경미한 시간 패턴 변화\n"
                    else:
                        report += "- ✅ 시간 패턴 정상\n"
                else:
                    report += f"- ❌ 분석 실패: {temporal_result['error']}\n"
                report += "\n"
            
            if temporal_successful > 0:
                report += f"**전체 Baseline 파일 요약**: 총 {temporal_total_anomalies}개 시간 기반 이상 감지 ({temporal_successful}개 파일)\n\n"
        
        report += "\n## 📊 파일별 비교 이상 탐지 결과\n\n"
        if comparative_result['success']:
            anomalies = comparative_result['anomalies']
            baseline_count = comparative_result['baseline_count']
            report += f"**비교 대상**: {baseline_count}개 baseline 파일\n\n"
            
            if anomalies:
                high_count = len([a for a in anomalies if a.get('severity') == 'high'])
                medium_count = len([a for a in anomalies if a.get('severity') == 'medium'])
                report += f"🚨 **발견된 이상**: {len(anomalies)}개 (심각: {high_count}개, 주의: {medium_count}개)\n\n"
                
                for anomaly in anomalies[:5]:
                    report += f"- **{anomaly.get('type', 'unknown')}**: {anomaly.get('description', 'No description')}\n"
                
                if len(anomalies) > 5:
                    report += f"- ... 및 {len(anomalies) - 5}개 추가\n"
            else:
                report += "✅ 파일별 비교에서 이상 현상이 발견되지 않았습니다.\n"
        else:
            report += f"❌ 파일별 비교 분석 실패: {comparative_result.get('error', 'Unknown error')}\n"
        
        # CLI 리포트 생성 결과 추가 (로그 샘플 분석 포함)
        report += "\n## 📄 CLI 리포트 및 로그 샘플 분석 결과\n\n"
        if target_cli_report_result['success']:
            report_file = target_cli_report_result.get('report_file')
            if report_file:
                report += f"**CLI 리포트**: `{report_file}`\n"
                report += "→ 기본 탐지 결과 및 통계 정보를 확인할 수 있습니다.\n\n"
            
            # 로그 샘플 분석 파일 확인
            target_dir = target_result['output_dir']
            sample_analysis_dir = target_dir / "log_samples_analysis"
            if sample_analysis_dir.exists():
                sample_report = sample_analysis_dir / "anomaly_analysis_report.md"
                sample_data = sample_analysis_dir / "anomaly_samples.json"
                
                if sample_report.exists():
                    report += f"**로그 샘플 분석 리포트**: `{sample_report}`\n"
                    report += "→ 실제 문제 로그들과 전후 맥락을 확인할 수 있습니다.\n\n"
                
                if sample_data.exists():
                    report += f"**상세 샘플 데이터**: `{sample_data}`\n"
                    report += "→ 구조화된 이상 로그 샘플 데이터입니다.\n\n"
            
            report += "✅ CLI 리포트 및 로그 샘플 분석이 정상적으로 생성되었습니다.\n"
        else:
            report += f"❌ Target CLI 리포트 생성 실패: {target_cli_report_result.get('error', 'Unknown error')}\n"
        
        # Baseline CLI 리포트들 요약
        if baseline_cli_reports:
            successful_baseline_reports = [r for r in baseline_cli_reports if r['success']]
            report += f"\n📂 **Baseline CLI 리포트**: {len(successful_baseline_reports)}/{len(baseline_cli_reports)}개 성공\n"
        
        # 실제 로그 샘플 통합
        report += self._add_log_samples_to_report(target_result, target_baseline_result, target_deeplog_result, target_mscred_result, target_temporal_result, comparative_result)
        
        # 권고사항 및 상세 결과
        total_anomalies = 0
        if target_temporal_result['success']:
            total_anomalies += len(target_temporal_result.get('anomalies', []))
        if comparative_result['success']:
            total_anomalies += len(comparative_result.get('anomalies', []))
        if target_baseline_result['success']:
            total_anomalies += target_baseline_result.get('anomaly_windows', 0)
        
        report += "\n## 💡 권고사항\n\n"
        if total_anomalies == 0:
            report += "✅ 모든 분석에서 이상 현상이 발견되지 않았습니다.\n"
        elif total_anomalies < 5:
            report += "🔍 일부 이상 현상이 발견되었습니다. 상세 분석을 권장합니다.\n"
        else:
            report += "⚠️ 다수의 이상 현상이 발견되었습니다. 긴급 점검이 필요할 수 있습니다.\n"
        
        if target_result['success']:
            report += f"""
## 📂 상세 결과 파일
- **Target 분석 결과**: `{target_result['output_dir']}/`
- **시간 기반 분석**: `{target_result['output_dir']}/temporal_analysis/temporal_report.md`
- **파일별 비교 분석**: `{target_result['output_dir']}/comparative_analysis/comparative_report.md`

## 🔧 추가 분석 명령어
```bash
# 상세 로그 샘플 분석 (단독 실행)
study-preprocess analyze-samples --processed-dir {target_result['output_dir']}

# 로그 샘플 포함 리포트 생성
study-preprocess report --processed-dir {target_result['output_dir']} --with-samples

# 상세 분석
python analyze_results.py --data-dir {target_result['output_dir']}

# 시각화
python visualize_results.py --data-dir {target_result['output_dir']}
```
"""
        
        return report
    
    def _add_log_samples_to_report(self, target_result: Dict, baseline_result: Dict, deeplog_result: Dict, 
                                   mscred_result: Dict, temporal_result: Dict, comparative_result: Dict) -> str:
        """실제 로그 샘플들을 리포트에 직접 포함합니다."""
        if not target_result['success']:
            return "\n## 🔍 로그 샘플 분석\n\n❌ Target 전처리 실패로 로그 샘플을 분석할 수 없습니다.\n"
        
        report = "\n## 🔍 실제 로그 샘플 분석\n\n"
        report += "다음은 각 분석 방법으로 발견된 실제 문제 로그들의 샘플입니다.\n\n"
        
        # Baseline 이상 로그 샘플
        if baseline_result['success'] and baseline_result.get('anomaly_windows', 0) > 0:
            report += self._extract_baseline_samples(target_result)
        
        # DeepLog 이상 로그 샘플  
        if deeplog_result['success'] and deeplog_result.get('violations', 0) > 0:
            report += self._extract_deeplog_samples(target_result)
        
        # MS-CRED 이상 로그 샘플
        if mscred_result['success'] and mscred_result.get('anomalies', 0) > 0:
            report += self._extract_mscred_samples(target_result)
        
        # 시간 기반 이상 로그 샘플
        if temporal_result['success'] and len(temporal_result.get('anomalies', [])) > 0:
            report += self._extract_temporal_samples(target_result, temporal_result)
        
        # 비교 분석 이상 로그 샘플
        if comparative_result['success'] and len(comparative_result.get('anomalies', [])) > 0:
            report += self._extract_comparative_samples(target_result, comparative_result)
        
        return report
    
    def _extract_baseline_samples(self, target_result: Dict) -> str:
        """Baseline 이상 로그 샘플을 추출합니다."""
        try:
            import pandas as pd
            
            baseline_scores_file = target_result['output_dir'] / "baseline_scores.parquet"
            parsed_file = target_result['output_dir'] / "parsed.parquet"
            
            if not baseline_scores_file.exists() or not parsed_file.exists():
                return ""
            
            scores_df = pd.read_parquet(baseline_scores_file)
            parsed_df = pd.read_parquet(parsed_file)
            
            anomaly_windows = scores_df[scores_df['is_anomaly'] == True].head(3)
            
            if len(anomaly_windows) == 0:
                return ""
            
            sample_text = "### 📈 Baseline 이상 탐지 샘플\n\n"
            
            for _, window in anomaly_windows.iterrows():
                start_line = int(window['window_start_line'])
                score = window['score']
                
                # 해당 윈도우의 로그들 추출
                window_logs = parsed_df[
                    (parsed_df['line_no'] >= start_line) & 
                    (parsed_df['line_no'] < start_line + 50)
                ].head(5)
                
                sample_text += f"**윈도우 시작라인 {start_line}** (점수: {score:.3f})\n"
                sample_text += "```\n"
                for _, log in window_logs.iterrows():
                    log_text = str(log.get('raw', ''))[:100]
                    sample_text += f"Line {log['line_no']}: {log_text}...\n"
                sample_text += "```\n\n"
            
            return sample_text
            
        except Exception as e:
            return f"⚠️ Baseline 샘플 추출 실패: {e}\n\n"
    
    def _extract_deeplog_samples(self, target_result: Dict) -> str:
        """DeepLog 이상 로그 샘플을 추출합니다."""
        try:
            import pandas as pd
            
            deeplog_file = target_result['output_dir'] / "deeplog_infer.parquet"
            parsed_file = target_result['output_dir'] / "parsed.parquet"
            
            if not deeplog_file.exists() or not parsed_file.exists():
                return ""
            
            deeplog_df = pd.read_parquet(deeplog_file)
            parsed_df = pd.read_parquet(parsed_file)
            
            violations = deeplog_df[deeplog_df['in_topk'] == False].head(5)
            
            if len(violations) == 0:
                return ""
            
            sample_text = "### 🧠 DeepLog 이상 탐지 샘플\n\n"
            
            for _, violation in violations.iterrows():
                line_no = violation['line_no']
                
                # 해당 라인의 로그 추출
                log_line = parsed_df[parsed_df['line_no'] == line_no]
                
                if len(log_line) > 0:
                    log = log_line.iloc[0]
                    log_text = str(log.get('raw', ''))
                    template_id = log.get('template_id', 'Unknown')
                    
                    sample_text += f"**Line {line_no}** (Template ID: {template_id})\n"
                    sample_text += f"```\n{log_text}\n```\n"
                    sample_text += f"→ 예측 실패: 이 로그 패턴이 예상되지 않았습니다.\n\n"
            
            return sample_text
            
        except Exception as e:
            return f"⚠️ DeepLog 샘플 추출 실패: {e}\n\n"
    
    def _extract_mscred_samples(self, target_result: Dict) -> str:
        """MS-CRED 이상 로그 샘플을 추출합니다."""
        try:
            import pandas as pd
            
            mscred_file = target_result['output_dir'] / "mscred_infer.parquet"
            parsed_file = target_result['output_dir'] / "parsed.parquet"
            window_counts_file = target_result['output_dir'] / "window_counts.parquet"
            
            if not mscred_file.exists() or not parsed_file.exists():
                return ""
            
            mscred_df = pd.read_parquet(mscred_file)
            parsed_df = pd.read_parquet(parsed_file)
            
            # 상위 5개 이상 윈도우 추출 (재구성 오차가 높은 순)
            anomaly_windows = mscred_df[mscred_df['is_anomaly'] == True].nlargest(5, 'reconstruction_error')
            
            if len(anomaly_windows) == 0:
                return ""
            
            sample_text = "### 🔬 MS-CRED 이상 탐지 샘플\n\n"
            
            for i, (_, anomaly) in enumerate(anomaly_windows.iterrows(), 1):
                window_idx = int(anomaly['window_idx'])
                start_index = int(anomaly.get('start_index', window_idx * 25))  # 기본 stride=25
                reconstruction_error = float(anomaly['reconstruction_error'])
                
                # 윈도우 범위의 로그들 추출 (50개 라인 기본)
                window_logs = parsed_df[
                    (parsed_df['line_no'] >= start_index) & 
                    (parsed_df['line_no'] < start_index + 50)
                ].copy()
                
                if len(window_logs) == 0:
                    continue
                
                # 에러 로그와 일반 로그 분리
                error_logs = window_logs[
                    window_logs['raw'].str.contains(
                        r'error|Error|ERROR|fail|Fail|FAIL|exception|Exception|EXCEPTION|warning|Warning|WARNING|critical|Critical|CRITICAL',
                        case=False, na=False, regex=True
                    )
                ]
                
                # 윈도우 정보
                sample_text += f"**이상 윈도우 #{i}** (윈도우 ID: {window_idx}, 시작 라인: {start_index})\n"
                sample_text += f"- 재구성 오차: {reconstruction_error:.4f}\n"
                sample_text += f"- 총 로그 수: {len(window_logs)}\n"
                sample_text += f"- 에러 로그 수: {len(error_logs)}\n\n"
                
                # 에러 로그 샘플 (최대 3개)
                if len(error_logs) > 0:
                    sample_text += "**🚨 에러 로그 샘플:**\n"
                    for _, error_log in error_logs.head(3).iterrows():
                        line_no = error_log['line_no']
                        log_text = str(error_log.get('raw', ''))
                        template_id = error_log.get('template_id', 'Unknown')
                        
                        sample_text += f"- Line {line_no} (Template: {template_id})\n"
                        sample_text += f"```\n{log_text[:200]}{'...' if len(log_text) > 200 else ''}\n```\n"
                    sample_text += "\n"
                
                # 일반 로그 샘플 (최대 2개)
                normal_logs = window_logs[~window_logs.index.isin(error_logs.index)]
                if len(normal_logs) > 0:
                    sample_text += "**📄 윈도우 내 다른 로그들:**\n"
                    for _, normal_log in normal_logs.head(2).iterrows():
                        line_no = normal_log['line_no']
                        log_text = str(normal_log.get('raw', ''))
                        
                        sample_text += f"- Line {line_no}\n"
                        sample_text += f"```\n{log_text[:150]}{'...' if len(log_text) > 150 else ''}\n```\n"
                    sample_text += "\n"
                
                # 템플릿 분포 정보
                template_counts = window_logs['template_id'].value_counts().head(3)
                if len(template_counts) > 0:
                    sample_text += "**📊 주요 템플릿:**\n"
                    for template_id, count in template_counts.items():
                        sample_text += f"- Template {template_id}: {count}회\n"
                    sample_text += "\n"
                
                sample_text += f"→ **분석**: 이 윈도우는 정상 패턴과 달리 재구성 오차가 {reconstruction_error:.4f}로 높게 나타났습니다.\n\n"
                sample_text += "---\n\n"
            
            return sample_text
            
        except Exception as e:
            return f"⚠️ MS-CRED 샘플 추출 실패: {e}\n\n"
    
    def _extract_temporal_samples(self, target_result: Dict, temporal_result: Dict) -> str:
        """시간 기반 이상 로그 샘플을 추출합니다."""
        try:
            anomalies = temporal_result.get('anomalies', [])[:3]
            if not anomalies:
                return ""
            
            sample_text = "### 🕐 시간 기반 이상 탐지 샘플\n\n"
            
            for anomaly in anomalies:
                hour = anomaly.get('hour', 'Unknown')
                anomaly_type = anomaly.get('type', 'Unknown')
                description = anomaly.get('description', 'No description')
                severity = anomaly.get('severity', 'medium')
                
                severity_icon = {'high': '🚨', 'medium': '⚠️', 'low': '🔍'}.get(severity, '⚠️')
                
                sample_text += f"**{severity_icon} {hour}시 이상 현상**\n"
                sample_text += f"- **유형**: {anomaly_type}\n"
                sample_text += f"- **설명**: {description}\n"
                sample_text += f"- **심각도**: {severity}\n\n"
            
            return sample_text
            
        except Exception as e:
            return f"⚠️ 시간 기반 샘플 추출 실패: {e}\n\n"
    
    def _extract_comparative_samples(self, target_result: Dict, comparative_result: Dict) -> str:
        """비교 분석 이상 로그 샘플을 추출합니다."""
        try:
            anomalies = comparative_result.get('anomalies', [])[:3]
            if not anomalies:
                return ""
            
            sample_text = "### 📊 비교 분석 이상 탐지 샘플\n\n"
            
            for anomaly in anomalies:
                anomaly_type = anomaly.get('type', 'Unknown')
                description = anomaly.get('description', 'No description')
                severity = anomaly.get('severity', 'medium')
                
                severity_icon = {'high': '🚨', 'medium': '⚠️', 'low': '🔍'}.get(severity, '⚠️')
                
                sample_text += f"**{severity_icon} {anomaly_type} 이상**\n"
                sample_text += f"- **설명**: {description}\n"
                sample_text += f"- **심각도**: {severity}\n\n"
            
            return sample_text
            
        except Exception as e:
            return f"⚠️ 비교 분석 샘플 추출 실패: {e}\n\n"

def main():
    parser = argparse.ArgumentParser(description="향상된 배치 로그 분석기")
    parser.add_argument("input_dir", help="로그 파일들이 있는 루트 디렉토리")
    parser.add_argument("--target", help="분석할 target 파일 (지정하지 않으면 가장 큰 파일)")
    parser.add_argument("--max-depth", type=int, default=3, help="하위 디렉토리 스캔 최대 깊이 (기본: 3)")
    parser.add_argument("--max-files", type=int, default=20, help="최대 처리 파일 수 (기본: 20)")
    parser.add_argument("--work-dir", help="작업 디렉토리 (기본: 자동 생성)")
    
    args = parser.parse_args()
    
    # 분석기 초기화 및 실행
    analyzer = EnhancedBatchAnalyzer(args.work_dir)
    result = analyzer.run_enhanced_analysis(
        args.input_dir, 
        args.target,
        args.max_depth,
        args.max_files
    )
    
    if not result['success']:
        print(f"❌ 분석 실패: {result['error']}")
        sys.exit(1)
    
    print(f"\n🎉 향상된 배치 분석 완료!")
    print(f"📊 처리된 파일: {result['files_processed']}/{result['total_files_found']}개")

if __name__ == "__main__":
    main()
