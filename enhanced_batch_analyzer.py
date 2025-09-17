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
            "*.access", "*.error", "*.debug"
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
            # study-preprocess 명령어로 전처리 실행 (가상환경 python 사용)
            venv_python = Path(".venv/bin/python")
            if venv_python.exists():
                python_cmd = str(venv_python)
            else:
                python_cmd = sys.executable
            
            cmd = [
                python_cmd, "-m", "study_preprocessor.cli", "parse",
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
            # 지정된 Target 파일 찾기
            target_matches = [
                (f, c) for f, c in log_files 
                if f.name == target_file or str(f) == target_file
            ]
            
            if target_matches:
                target = target_matches[0]
                baselines = [(f, c) for f, c in log_files if (f, c) != target]
            else:
                print(f"⚠️  지정된 Target 파일을 찾을 수 없음: {target_file}")
                print("📋 사용 가능한 파일들:")
                for i, (f, c) in enumerate(log_files[:10], 1):
                    print(f"  {i}. {f.name} (카테고리: {c})")
                target = log_files[0] if log_files else None
                baselines = log_files[1:] if len(log_files) > 1 else []
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
        
        # 파일 수 제한
        if len(log_files) > max_files:
            print(f"⚠️  파일 수가 많아 상위 {max_files}개만 분석합니다")
            log_files = log_files[:max_files]
        
        # 2. Target 및 Baseline 선택
        target_info, baseline_infos = self.select_target_and_baselines(log_files, target_file)
        if not target_info:
            print("❌ Target 파일이 없습니다")
            return {'success': False, 'error': 'No target file'}
        
        # 3. 전처리 실행
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
        
        # 4. 성공한 파일들만으로 분석 진행
        successful_baselines = [r for r in baseline_results if r['success']]
        
        print(f"\n📊 전처리 결과 요약:")
        print(f"  - Target: {'✅' if target_result['success'] else '❌'} {target_info[0].name}")
        print(f"  - Baseline 성공: {len(successful_baselines)}/{len(baseline_results)}개")
        
        # 5. 분석 실행 (기존 코드와 동일)
        temporal_result = {'success': False, 'error': 'Target preprocessing failed'}
        comparative_result = {'success': False, 'error': 'Target preprocessing failed'}
        
        if target_result['success']:
            # 시간 기반 분석
            print(f"\n{'='*60}")
            print("🕐 시간 기반 이상 탐지")
            print(f"{'='*60}")
            temporal_result = self.run_temporal_analysis(target_result)
            
            # 파일별 비교 분석
            if successful_baselines:
                print(f"\n{'='*60}")
                print("📊 파일별 비교 이상 탐지")
                print(f"{'='*60}")
                comparative_result = self.run_comparative_analysis(target_result, successful_baselines)
        
        # 6. 요약 리포트 생성
        print(f"\n{'='*60}")
        print("📄 요약 리포트 생성")
        print(f"{'='*60}")
        
        summary_report = self.generate_enhanced_summary_report(
            target_result, baseline_results, temporal_result, comparative_result,
            input_dir, max_depth
        )
        
        summary_file = self.work_dir / "ENHANCED_ANALYSIS_SUMMARY.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"📋 요약 리포트 저장: {summary_file}")
        
        # 결과 요약 출력
        print(f"\n{'='*60}")
        print("✅ 향상된 배치 분석 완료")
        print(f"{'='*60}")
        
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
            'summary_file': summary_file,
            'total_files_found': len(log_files),
            'files_processed': total_count
        }
    
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
        """파일별 비교 분석 실행 (기존과 동일)."""
        if not target_result['success']:
            return {'success': False, 'error': 'Target preprocessing failed'}
        
        valid_baselines = [r for r in baseline_results if r['success']]
        if not valid_baselines:
            print("⚠️ 비교할 baseline 파일이 없습니다")
            return {'success': False, 'error': 'No valid baseline files'}
        
        print(f"📊 파일별 비교 분석: {target_result['file_path'].name} vs {len(valid_baselines)}개 파일")
        
        try:
            baseline_paths = [str(r['parsed_file']) for r in valid_baselines]
            
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
            
            comp_result = {'success': True, 'anomalies': [], 'baseline_count': len(valid_baselines)}
            if anomalies_file.exists():
                with open(anomalies_file) as f:
                    comp_result['anomalies'] = json.load(f)
            
            print(f"✅ 비교 분석 완료: {len(comp_result['anomalies'])}개 이상 발견")
            return comp_result
            
        except Exception as e:
            print(f"❌ 비교 분석 예외: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_enhanced_summary_report(self, target_result: Dict, baseline_results: List[Dict],
                                       temporal_result: Dict, comparative_result: Dict,
                                       input_dir: str, max_depth: int) -> str:
        """향상된 요약 리포트 생성."""
        
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
            report += f"""- ✅ **성공**: {stats['total_logs']:,}개 로그, {stats['unique_templates']}개 템플릿
- **카테고리**: {target_result.get('category', 'root')}
- **파일 크기**: {validation.get('file_size_mb', 0):.1f}MB
- **로그 형식**: {validation.get('format', 'unknown')}
- **시간 범위**: {stats['time_range']['start']} ~ {stats['time_range']['end']}
- **호스트**: {len(stats['hosts'])}개 ({', '.join(stats['hosts'][:3])}{'...' if len(stats['hosts']) > 3 else ''})
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
        
        # 나머지는 기존과 동일...
        report += "\n## 🕐 시간 기반 이상 탐지 결과\n\n"
        if temporal_result['success']:
            anomalies = temporal_result['anomalies']
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
            report += f"❌ 시간 기반 분석 실패: {temporal_result.get('error', 'Unknown error')}\n"
        
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
        
        # 권고사항 및 상세 결과
        total_anomalies = 0
        if temporal_result['success']:
            total_anomalies += len(temporal_result.get('anomalies', []))
        if comparative_result['success']:
            total_anomalies += len(comparative_result.get('anomalies', []))
        
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
# 상세 분석
python analyze_results.py --data-dir {target_result['output_dir']}

# 시각화
python visualize_results.py --data-dir {target_result['output_dir']}
```
"""
        
        return report

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
