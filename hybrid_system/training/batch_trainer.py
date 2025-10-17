#!/usr/bin/env python3
"""
기존 Python 학습 파이프라인을 래핑하고 최적화하는 모듈
하이브리드 시스템의 학습 컴포넌트
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchTrainer:
    """배치 학습 파이프라인 관리자"""
    
    def __init__(self, 
                 cache_dir: str = ".cache",
                 models_dir: str = "models",
                 use_venv: bool = True):
        self.cache_dir = Path(cache_dir)
        self.models_dir = Path(models_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.use_venv = use_venv
        
        # Python 명령어 설정
        self.python_cmd = self._setup_python_command()
    
    def _setup_python_command(self) -> str:
        """Python 명령어 설정 (가상환경 고려)"""
        if self.use_venv and os.environ.get('VIRTUAL_ENV'):
            return "python"
        elif self.use_venv and Path(".venv/bin/python").exists():
            return ".venv/bin/python"
        else:
            return "python3"
    
    def train_full_pipeline(self, 
                          log_file: str, 
                          output_dir: Optional[str] = None,
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        전체 학습 파이프라인 실행
        
        Args:
            log_file: 입력 로그 파일 경로
            output_dir: 출력 디렉토리 (None이면 자동 생성)
            config: 학습 설정
            
        Returns:
            학습 결과 정보
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"data/processed/batch_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 기본 설정
        default_config = {
            'seq_len': 50,
            'deeplog_epochs': 3,
            'mscred_epochs': 30,
            'window_size': 50,
            'stride': 25,
            'ewm_alpha': 0.3,
            'anomaly_q': 0.95
        }
        
        if config:
            default_config.update(config)
        
        logger.info(f"🚀 배치 학습 파이프라인 시작")
        logger.info(f"📂 입력 파일: {log_file}")
        logger.info(f"📂 출력 디렉토리: {output_dir}")
        logger.info(f"⚙️ 설정: {default_config}")
        
        results = {
            'log_file': log_file,
            'output_dir': output_dir,
            'config': default_config,
            'steps': {},
            'models': {},
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # 1단계: 로그 전처리
            logger.info("1️⃣ 로그 전처리 중...")
            self._run_preprocessing(log_file, output_dir, results)
            
            # 2단계: DeepLog 데이터 준비
            logger.info("2️⃣ DeepLog 데이터 준비 중...")
            self._prepare_deeplog_data(output_dir, results)
            
            # 3단계: MS-CRED 데이터 준비
            logger.info("3️⃣ MS-CRED 데이터 준비 중...")
            self._prepare_mscred_data(output_dir, default_config, results)
            
            # 4단계: 베이스라인 이상탐지
            logger.info("4️⃣ 베이스라인 이상탐지 중...")
            self._run_baseline_detection(output_dir, default_config, results)
            
            # 5단계: DeepLog 학습
            logger.info("5️⃣ DeepLog 모델 학습 중...")
            self._train_deeplog(output_dir, default_config, results)
            
            # 6단계: MS-CRED 학습
            logger.info("6️⃣ MS-CRED 모델 학습 중...")
            self._train_mscred(output_dir, default_config, results)
            
            # 7단계: 리포트 생성
            logger.info("7️⃣ 리포트 생성 중...")
            self._generate_report(output_dir, results)
            
            results['status'] = 'success'
            results['end_time'] = datetime.now().isoformat()
            
            logger.info("✅ 배치 학습 파이프라인 완료!")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            logger.error(f"❌ 배치 학습 실패: {e}")
            raise
        
        # 결과 저장
        results_path = Path(output_dir) / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return results
    
    def _run_command(self, cmd: List[str], step_name: str) -> Dict[str, Any]:
        """명령어 실행 및 결과 기록"""
        logger.info(f"실행 중: {' '.join(cmd)}")
        
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = datetime.now()
        
        step_result = {
            'command': ' '.join(cmd),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode != 0:
            logger.error(f"❌ {step_name} 실패: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
        
        logger.info(f"✅ {step_name} 완료 ({step_result['duration_seconds']:.1f}초)")
        return step_result
    
    def _run_preprocessing(self, log_file: str, output_dir: str, results: Dict[str, Any]):
        """로그 전처리 실행"""
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "parse",
            "--input", log_file,
            "--out-dir", output_dir,
            "--drain-state", str(self.cache_dir / "drain3_state.json")
        ]
        
        step_result = self._run_command(cmd, "로그 전처리")
        results['steps']['preprocessing'] = step_result
        
        # 결과 파일 확인
        parsed_file = Path(output_dir) / "parsed.parquet"
        if not parsed_file.exists():
            raise FileNotFoundError(f"전처리 결과 파일이 생성되지 않음: {parsed_file}")
        
        results['files'] = results.get('files', {})
        results['files']['parsed'] = str(parsed_file)
    
    def _prepare_deeplog_data(self, output_dir: str, results: Dict[str, Any]):
        """DeepLog 입력 데이터 준비"""
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "build-deeplog",
            "--parsed", str(Path(output_dir) / "parsed.parquet"),
            "--out-dir", output_dir
        ]
        
        step_result = self._run_command(cmd, "DeepLog 데이터 준비")
        results['steps']['deeplog_data'] = step_result
        
        # 결과 파일 확인
        vocab_file = Path(output_dir) / "vocab.json"
        sequences_file = Path(output_dir) / "sequences.parquet"
        
        if not vocab_file.exists() or not sequences_file.exists():
            raise FileNotFoundError("DeepLog 입력 파일이 생성되지 않음")
        
        results['files']['vocab'] = str(vocab_file)
        results['files']['sequences'] = str(sequences_file)
    
    def _prepare_mscred_data(self, output_dir: str, config: Dict[str, Any], results: Dict[str, Any]):
        """MS-CRED 입력 데이터 준비"""
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "build-mscred",
            "--parsed", str(Path(output_dir) / "parsed.parquet"),
            "--out-dir", output_dir,
            "--window-size", str(config['window_size']),
            "--stride", str(config['stride'])
        ]
        
        step_result = self._run_command(cmd, "MS-CRED 데이터 준비")
        results['steps']['mscred_data'] = step_result
        
        # 결과 파일 확인
        window_counts_file = Path(output_dir) / "window_counts.parquet"
        if not window_counts_file.exists():
            raise FileNotFoundError("MS-CRED 입력 파일이 생성되지 않음")
        
        results['files']['window_counts'] = str(window_counts_file)
    
    def _run_baseline_detection(self, output_dir: str, config: Dict[str, Any], results: Dict[str, Any]):
        """베이스라인 이상탐지 실행"""
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "detect",
            "--parsed", str(Path(output_dir) / "parsed.parquet"),
            "--out-dir", output_dir,
            "--window-size", str(config['window_size']),
            "--stride", str(config['stride']),
            "--ewm-alpha", str(config['ewm_alpha']),
            "--q", str(config['anomaly_q'])
        ]
        
        step_result = self._run_command(cmd, "베이스라인 이상탐지")
        results['steps']['baseline_detection'] = step_result
        
        # 결과 파일 확인
        baseline_file = Path(output_dir) / "baseline_scores.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError("베이스라인 결과 파일이 생성되지 않음")
        
        results['files']['baseline_scores'] = str(baseline_file)
    
    def _train_deeplog(self, output_dir: str, config: Dict[str, Any], results: Dict[str, Any]):
        """DeepLog 모델 학습"""
        model_name = f"deeplog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        model_path = self.models_dir / model_name
        
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "deeplog-train",
            "--seq", str(Path(output_dir) / "sequences.parquet"),
            "--vocab", str(Path(output_dir) / "vocab.json"),
            "--out", str(model_path),
            "--seq-len", str(config['seq_len']),
            "--epochs", str(config['deeplog_epochs'])
        ]
        
        step_result = self._run_command(cmd, "DeepLog 학습")
        results['steps']['deeplog_training'] = step_result
        
        # 모델 파일 확인
        if not model_path.exists():
            raise FileNotFoundError(f"DeepLog 모델이 생성되지 않음: {model_path}")
        
        results['models']['deeplog'] = {
            'path': str(model_path),
            'type': 'pytorch',
            'config': {
                'seq_len': config['seq_len'],
                'epochs': config['deeplog_epochs']
            }
        }
        
        # DeepLog 추론도 실행
        self._run_deeplog_inference(output_dir, model_path, results)
    
    def _run_deeplog_inference(self, output_dir: str, model_path: Path, results: Dict[str, Any]):
        """DeepLog 추론 실행"""
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "deeplog-infer",
            "--seq", str(Path(output_dir) / "sequences.parquet"),
            "--model", str(model_path),
            "--k", "3"
        ]
        
        step_result = self._run_command(cmd, "DeepLog 추론")
        results['steps']['deeplog_inference'] = step_result
        
        # 결과 파일 확인
        inference_file = Path(output_dir) / "deeplog_infer.parquet"
        if not inference_file.exists():
            raise FileNotFoundError("DeepLog 추론 결과가 생성되지 않음")
        
        results['files']['deeplog_inference'] = str(inference_file)
    
    def _train_mscred(self, output_dir: str, config: Dict[str, Any], results: Dict[str, Any]):
        """MS-CRED 모델 학습"""
        model_name = f"mscred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        model_path = self.models_dir / model_name
        
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "mscred-train",
            "--window-counts", str(Path(output_dir) / "window_counts.parquet"),
            "--out", str(model_path),
            "--epochs", str(config['mscred_epochs'])
        ]
        
        try:
            step_result = self._run_command(cmd, "MS-CRED 학습")
            results['steps']['mscred_training'] = step_result
            
            # 모델 파일 확인
            if model_path.exists():
                results['models']['mscred'] = {
                    'path': str(model_path),
                    'type': 'pytorch',
                    'config': {
                        'epochs': config['mscred_epochs']
                    }
                }
                
                # MS-CRED 추론도 실행
                self._run_mscred_inference(output_dir, model_path, results)
            else:
                logger.warning("MS-CRED 모델 파일이 생성되지 않음")
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"MS-CRED 학습 실패 (계속 진행): {e}")
            results['steps']['mscred_training'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def _run_mscred_inference(self, output_dir: str, model_path: Path, results: Dict[str, Any]):
        """MS-CRED 추론 실행"""
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "mscred-infer",
            "--window-counts", str(Path(output_dir) / "window_counts.parquet"),
            "--model", str(model_path),
            "--threshold", "95.0"
        ]
        
        try:
            step_result = self._run_command(cmd, "MS-CRED 추론")
            results['steps']['mscred_inference'] = step_result
            
            # 결과 파일 확인
            inference_file = Path(output_dir) / "mscred_infer.parquet"
            if inference_file.exists():
                results['files']['mscred_inference'] = str(inference_file)
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"MS-CRED 추론 실패 (계속 진행): {e}")
            results['steps']['mscred_inference'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def _generate_report(self, output_dir: str, results: Dict[str, Any]):
        """리포트 생성"""
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "report",
            "--processed-dir", output_dir
        ]
        
        step_result = self._run_command(cmd, "리포트 생성")
        results['steps']['report'] = step_result
        
        # 리포트 파일 확인
        report_file = Path(output_dir) / "report.md"
        if report_file.exists():
            results['files']['report'] = str(report_file)


def main():
    """CLI 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(description="배치 학습 파이프라인 실행")
    parser.add_argument("log_file", help="입력 로그 파일")
    parser.add_argument("--output-dir", help="출력 디렉토리")
    parser.add_argument("--config", help="설정 JSON 파일")
    parser.add_argument("--seq-len", type=int, default=50, help="시퀀스 길이")
    parser.add_argument("--deeplog-epochs", type=int, default=3, help="DeepLog 에포크")
    parser.add_argument("--mscred-epochs", type=int, default=30, help="MS-CRED 에포크")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # CLI 인수로 설정 업데이트
    config.update({
        'seq_len': args.seq_len,
        'deeplog_epochs': args.deeplog_epochs,
        'mscred_epochs': args.mscred_epochs
    })
    
    # 학습 실행
    trainer = BatchTrainer()
    results = trainer.train_full_pipeline(
        args.log_file,
        args.output_dir,
        config
    )
    
    print(f"\n🎉 학습 완료!")
    print(f"📂 결과 디렉토리: {results['output_dir']}")
    print(f"📊 학습된 모델:")
    for model_name, model_info in results.get('models', {}).items():
        print(f"  - {model_name}: {model_info['path']}")


if __name__ == "__main__":
    main()
