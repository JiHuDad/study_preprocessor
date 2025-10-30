#!/usr/bin/env python3  # 실행 스크립트 셔뱅
"""기존 Python 학습 파이프라인을 래핑하고 최적화하는 모듈 요약

- 목적: 로그 이상탐지 모델(DeepLog, MS-CRED, 베이스라인)의 전체 학습 파이프라인을 자동화
- 주요 기능:
  * 로그 전처리 (Drain3 파싱)
  * DeepLog/MS-CRED 데이터 준비 (vocab, sequences, window counts)
  * 베이스라인 이상탐지 실행
  * DeepLog LSTM 모델 학습 및 추론
  * MS-CRED 컨볼루션 모델 학습 및 추론
  * 학습 결과 리포트 생성
- 실행 순서:
  1. 로그 전처리 → 2. DeepLog 데이터 준비 → 3. MS-CRED 데이터 준비
  → 4. 베이스라인 이상탐지 → 5. DeepLog 학습 → 6. MS-CRED 학습 → 7. 리포트 생성
- 출력: 학습된 모델(.pth), 추론 결과(.parquet), 리포트(.md), 학습 결과 JSON
"""  # 모듈 요약 설명
import os  # 환경 변수/경로 유틸
import json  # JSON 입출력
import subprocess  # 외부 명령 실행
import sys  # 시스템 함수
from pathlib import Path  # 경로 처리
from typing import Dict, Any, List, Optional  # 타입 힌트
import logging  # 로깅 프레임워크
from datetime import datetime  # 날짜/시간 처리

# 로깅 설정  # INFO 레벨 기본 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # 모듈 로거 생성


class BatchTrainer:  # 배치 학습 파이프라인 관리자 클래스
    """배치 학습 파이프라인 관리자"""  # 클래스 설명
    
    def __init__(self, 
                 cache_dir: str = ".cache",  # 캐시 디렉토리
                 models_dir: str = "models",  # 모델 저장 디렉토리
                 use_venv: bool = True):  # 가상환경 사용 여부
        self.cache_dir = Path(cache_dir)  # 캐시 경로
        self.models_dir = Path(models_dir)  # 모델 경로
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # 캐시 폴더 생성
        self.models_dir.mkdir(parents=True, exist_ok=True)  # 모델 폴더 생성
        self.use_venv = use_venv  # 가상환경 사용 플래그
        
        # Python 명령어 설정  # Python 실행 경로 결정
        self.python_cmd = self._setup_python_command()  # Python 명령어 설정
    
    def _setup_python_command(self) -> str:  # Python 명령어 설정
        """Python 명령어 설정 (가상환경 고려)"""  # 메서드 설명
        if self.use_venv and os.environ.get('VIRTUAL_ENV'):  # 가상환경 활성화됨
            return "python"  # 활성화된 환경의 python 사용
        elif self.use_venv and Path(".venv/bin/python").exists():  # .venv 존재
            return ".venv/bin/python"  # 로컬 venv의 python 사용
        else:
            return "python3"  # 시스템 python3 사용
    
    def train_full_pipeline(self, 
                          log_file: str,  # 입력 로그 파일
                          output_dir: Optional[str] = None,  # 출력 디렉토리
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # 전체 학습 파이프라인 실행
        """
        전체 학습 파이프라인 실행
        
        Args:
            log_file: 입력 로그 파일 경로  # 입력 파일 경로
            output_dir: 출력 디렉토리 (None이면 자동 생성)  # 출력 폴더
            config: 학습 설정  # 설정 딕셔너리
            
        Returns:
            학습 결과 정보  # 결과 딕셔너리
        """  # API 설명
        if output_dir is None:  # 출력 디렉토리 미지정
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 타임스탬프 생성
            output_dir = f"data/processed/batch_{timestamp}"  # 자동 디렉토리명 생성
        
        output_path = Path(output_dir)  # 출력 경로
        output_path.mkdir(parents=True, exist_ok=True)  # 출력 폴더 생성
        
        # 기본 설정  # 기본 파라미터 정의
        default_config = {
            'seq_len': 50,  # 시퀀스 길이
            'deeplog_epochs': 3,  # DeepLog 에포크 수
            'mscred_epochs': 30,  # MS-CRED 에포크 수
            'window_size': 50,  # 윈도우 크기
            'stride': 25,  # 스트라이드
            'ewm_alpha': 0.3,  # EWMA 알파
            'anomaly_q': 0.95  # 이상 임계값 백분위수
        }
        
        if config:  # 사용자 설정 존재
            default_config.update(config)  # 설정 병합
        
        logger.info(f"🚀 배치 학습 파이프라인 시작")  # 시작 로그
        logger.info(f"📂 입력 파일: {log_file}")
        logger.info(f"📂 출력 디렉토리: {output_dir}")
        logger.info(f"⚙️ 설정: {default_config}")
        
        results = {
            'log_file': log_file,  # 입력 파일
            'output_dir': output_dir,  # 출력 디렉토리
            'config': default_config,  # 설정
            'steps': {},  # 단계별 결과
            'models': {},  # 모델 정보
            'start_time': datetime.now().isoformat()  # 시작 시간
        }
        
        try:
            # 1단계: 로그 전처리  # Drain3 파싱
            logger.info("1️⃣ 로그 전처리 중...")
            self._run_preprocessing(log_file, output_dir, results)  # 전처리 실행
            
            # 2단계: DeepLog 데이터 준비  # vocab, sequences 생성
            logger.info("2️⃣ DeepLog 데이터 준비 중...")
            self._prepare_deeplog_data(output_dir, results)  # DeepLog 데이터 준비
            
            # 3단계: MS-CRED 데이터 준비  # window counts 생성
            logger.info("3️⃣ MS-CRED 데이터 준비 중...")
            self._prepare_mscred_data(output_dir, default_config, results)  # MS-CRED 데이터 준비
            
            # 4단계: 베이스라인 이상탐지  # 통계 기반 이상탐지
            logger.info("4️⃣ 베이스라인 이상탐지 중...")
            self._run_baseline_detection(output_dir, default_config, results)  # 베이스라인 실행
            
            # 5단계: DeepLog 학습  # LSTM 모델 학습
            logger.info("5️⃣ DeepLog 모델 학습 중...")
            self._train_deeplog(output_dir, default_config, results)  # DeepLog 학습
            
            # 6단계: MS-CRED 학습  # 컨볼루션 모델 학습
            logger.info("6️⃣ MS-CRED 모델 학습 중...")
            self._train_mscred(output_dir, default_config, results)  # MS-CRED 학습
            
            # 7단계: 리포트 생성  # 결과 리포트
            logger.info("7️⃣ 리포트 생성 중...")
            self._generate_report(output_dir, results)  # 리포트 생성
            
            results['status'] = 'success'  # 성공 상태
            results['end_time'] = datetime.now().isoformat()  # 종료 시간
            
            logger.info("✅ 배치 학습 파이프라인 완료!")  # 완료 로그
            
        except Exception as e:
            results['status'] = 'failed'  # 실패 상태
            results['error'] = str(e)  # 에러 메시지
            results['end_time'] = datetime.now().isoformat()  # 종료 시간
            logger.error(f"❌ 배치 학습 실패: {e}")  # 에러 로그
            raise  # 예외 재발생
        
        # 결과 저장  # JSON 파일 저장
        results_path = Path(output_dir) / "training_results.json"  # 결과 파일 경로
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)  # JSON 저장
        
        return results  # 결과 반환
    
    def _run_command(self, cmd: List[str], step_name: str) -> Dict[str, Any]:  # 명령어 실행
        """명령어 실행 및 결과 기록"""  # 메서드 설명
        logger.info(f"실행 중: {' '.join(cmd)}")  # 명령 로그
        
        start_time = datetime.now()  # 시작 시간 기록
        result = subprocess.run(cmd, capture_output=True, text=True)  # 명령 실행
        end_time = datetime.now()  # 종료 시간 기록
        
        step_result = {
            'command': ' '.join(cmd),  # 실행 명령
            'start_time': start_time.isoformat(),  # 시작 시간
            'end_time': end_time.isoformat(),  # 종료 시간
            'duration_seconds': (end_time - start_time).total_seconds(),  # 실행 시간
            'return_code': result.returncode,  # 반환 코드
            'stdout': result.stdout,  # 표준 출력
            'stderr': result.stderr  # 표준 에러
        }
        
        if result.returncode != 0:  # 실행 실패
            logger.error(f"❌ {step_name} 실패: {result.stderr}")  # 에러 로그
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)  # 예외 발생
        
        logger.info(f"✅ {step_name} 완료 ({step_result['duration_seconds']:.1f}초)")  # 완료 로그
        return step_result  # 결과 반환
    
    def _run_preprocessing(self, log_file: str, output_dir: str, results: Dict[str, Any]):  # 로그 전처리 실행
        """로그 전처리 실행"""  # 메서드 설명
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "parse",  # CLI parse 명령
            "--input", log_file,  # 입력 파일
            "--out-dir", output_dir,  # 출력 디렉토리
            "--drain-state", str(self.cache_dir / "drain3_state.json")  # Drain3 상태 파일
        ]
        
        step_result = self._run_command(cmd, "로그 전처리")  # 명령 실행
        results['steps']['preprocessing'] = step_result  # 결과 저장
        
        # 결과 파일 확인  # parsed.parquet 파일 확인
        parsed_file = Path(output_dir) / "parsed.parquet"  # 파싱 결과 파일 경로
        if not parsed_file.exists():  # 파일 없음
            raise FileNotFoundError(f"전처리 결과 파일이 생성되지 않음: {parsed_file}")  # 에러 발생
        
        results['files'] = results.get('files', {})  # files 딕셔너리 초기화
        results['files']['parsed'] = str(parsed_file)  # 파싱 파일 경로 저장
    
    def _prepare_deeplog_data(self, output_dir: str, results: Dict[str, Any]):  # DeepLog 데이터 준비
        """DeepLog 입력 데이터 준비"""  # 메서드 설명
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "build-deeplog",  # CLI build-deeplog 명령
            "--parsed", str(Path(output_dir) / "parsed.parquet"),  # 파싱 결과 파일
            "--out-dir", output_dir  # 출력 디렉토리
        ]
        
        step_result = self._run_command(cmd, "DeepLog 데이터 준비")  # 명령 실행
        results['steps']['deeplog_data'] = step_result  # 결과 저장
        
        # 결과 파일 확인  # vocab, sequences 파일 확인
        vocab_file = Path(output_dir) / "vocab.json"  # 어휘 사전 파일 경로
        sequences_file = Path(output_dir) / "sequences.parquet"  # 시퀀스 파일 경로
        
        if not vocab_file.exists() or not sequences_file.exists():  # 파일 없음
            raise FileNotFoundError("DeepLog 입력 파일이 생성되지 않음")  # 에러 발생
        
        results['files']['vocab'] = str(vocab_file)  # vocab 파일 경로 저장
        results['files']['sequences'] = str(sequences_file)  # sequences 파일 경로 저장
    
    def _prepare_mscred_data(self, output_dir: str, config: Dict[str, Any], results: Dict[str, Any]):  # MS-CRED 데이터 준비
        """MS-CRED 입력 데이터 준비"""  # 메서드 설명
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "build-mscred",  # CLI build-mscred 명령
            "--parsed", str(Path(output_dir) / "parsed.parquet"),  # 파싱 결과 파일
            "--out-dir", output_dir,  # 출력 디렉토리
            "--window-size", str(config['window_size']),  # 윈도우 크기
            "--stride", str(config['stride'])  # 스트라이드
        ]
        
        step_result = self._run_command(cmd, "MS-CRED 데이터 준비")  # 명령 실행
        results['steps']['mscred_data'] = step_result  # 결과 저장
        
        # 결과 파일 확인  # window_counts 파일 확인
        window_counts_file = Path(output_dir) / "window_counts.parquet"  # 윈도우 카운트 파일 경로
        if not window_counts_file.exists():  # 파일 없음
            raise FileNotFoundError("MS-CRED 입력 파일이 생성되지 않음")  # 에러 발생
        
        results['files']['window_counts'] = str(window_counts_file)  # window_counts 파일 경로 저장
    
    def _run_baseline_detection(self, output_dir: str, config: Dict[str, Any], results: Dict[str, Any]):  # 베이스라인 이상탐지 실행
        """베이스라인 이상탐지 실행"""  # 메서드 설명
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "detect",  # CLI detect 명령
            "--parsed", str(Path(output_dir) / "parsed.parquet"),  # 파싱 결과 파일
            "--out-dir", output_dir,  # 출력 디렉토리
            "--window-size", str(config['window_size']),  # 윈도우 크기
            "--stride", str(config['stride']),  # 스트라이드
            "--ewm-alpha", str(config['ewm_alpha']),  # EWMA 알파
            "--q", str(config['anomaly_q'])  # 이상 임계값 백분위수
        ]
        
        step_result = self._run_command(cmd, "베이스라인 이상탐지")  # 명령 실행
        results['steps']['baseline_detection'] = step_result  # 결과 저장
        
        # 결과 파일 확인  # baseline_scores 파일 확인
        baseline_file = Path(output_dir) / "baseline_scores.parquet"  # 베이스라인 점수 파일 경로
        if not baseline_file.exists():  # 파일 없음
            raise FileNotFoundError("베이스라인 결과 파일이 생성되지 않음")  # 에러 발생
        
        results['files']['baseline_scores'] = str(baseline_file)  # baseline_scores 파일 경로 저장
    
    def _train_deeplog(self, output_dir: str, config: Dict[str, Any], results: Dict[str, Any]):  # DeepLog 모델 학습
        """DeepLog 모델 학습"""  # 메서드 설명
        model_name = f"deeplog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"  # 모델 파일명 생성
        model_path = self.models_dir / model_name  # 모델 경로
        
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "deeplog-train",  # CLI deeplog-train 명령
            "--seq", str(Path(output_dir) / "sequences.parquet"),  # 시퀀스 파일
            "--vocab", str(Path(output_dir) / "vocab.json"),  # 어휘 사전 파일
            "--out", str(model_path),  # 출력 모델 경로
            "--seq-len", str(config['seq_len']),  # 시퀀스 길이
            "--epochs", str(config['deeplog_epochs'])  # 에포크 수
        ]
        
        step_result = self._run_command(cmd, "DeepLog 학습")  # 명령 실행
        results['steps']['deeplog_training'] = step_result  # 결과 저장
        
        # 모델 파일 확인  # 학습된 모델 파일 확인
        if not model_path.exists():  # 파일 없음
            raise FileNotFoundError(f"DeepLog 모델이 생성되지 않음: {model_path}")  # 에러 발생
        
        results['models']['deeplog'] = {
            'path': str(model_path),  # 모델 경로
            'type': 'pytorch',  # 모델 타입
            'config': {
                'seq_len': config['seq_len'],  # 시퀀스 길이
                'epochs': config['deeplog_epochs']  # 에포크 수
            }
        }
        
        # DeepLog 추론도 실행  # 학습 후 추론 수행
        self._run_deeplog_inference(output_dir, model_path, results)  # 추론 실행
    
    def _run_deeplog_inference(self, output_dir: str, model_path: Path, results: Dict[str, Any]):  # DeepLog 추론 실행
        """DeepLog 추론 실행"""  # 메서드 설명
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "deeplog-infer",  # CLI deeplog-infer 명령
            "--seq", str(Path(output_dir) / "sequences.parquet"),  # 시퀀스 파일
            "--model", str(model_path),  # 학습된 모델 경로
            "--k", "3"  # top-k 값
        ]
        
        step_result = self._run_command(cmd, "DeepLog 추론")  # 명령 실행
        results['steps']['deeplog_inference'] = step_result  # 결과 저장
        
        # 결과 파일 확인  # 추론 결과 파일 확인
        inference_file = Path(output_dir) / "deeplog_infer.parquet"  # 추론 결과 파일 경로
        if not inference_file.exists():  # 파일 없음
            raise FileNotFoundError("DeepLog 추론 결과가 생성되지 않음")  # 에러 발생
        
        results['files']['deeplog_inference'] = str(inference_file)  # 추론 파일 경로 저장
    
    def _train_mscred(self, output_dir: str, config: Dict[str, Any], results: Dict[str, Any]):  # MS-CRED 모델 학습
        """MS-CRED 모델 학습"""  # 메서드 설명
        model_name = f"mscred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"  # 모델 파일명 생성
        model_path = self.models_dir / model_name  # 모델 경로
        
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "mscred-train",  # CLI mscred-train 명령
            "--window-counts", str(Path(output_dir) / "window_counts.parquet"),  # 윈도우 카운트 파일
            "--out", str(model_path),  # 출력 모델 경로
            "--epochs", str(config['mscred_epochs'])  # 에포크 수
        ]
        
        try:
            step_result = self._run_command(cmd, "MS-CRED 학습")  # 명령 실행
            results['steps']['mscred_training'] = step_result  # 결과 저장
            
            # 모델 파일 확인  # 학습된 모델 파일 확인
            if model_path.exists():  # 파일 존재
                results['models']['mscred'] = {
                    'path': str(model_path),  # 모델 경로
                    'type': 'pytorch',  # 모델 타입
                    'config': {
                        'epochs': config['mscred_epochs']  # 에포크 수
                    }
                }
                
                # MS-CRED 추론도 실행  # 학습 후 추론 수행
                self._run_mscred_inference(output_dir, model_path, results)  # 추론 실행
            else:  # 파일 없음
                logger.warning("MS-CRED 모델 파일이 생성되지 않음")  # 경고 로그
                
        except subprocess.CalledProcessError as e:  # 명령 실행 실패
            logger.warning(f"MS-CRED 학습 실패 (계속 진행): {e}")  # 경고 로그
            results['steps']['mscred_training'] = {
                'status': 'failed',  # 실패 상태
                'error': str(e)  # 에러 메시지
            }
    
    def _run_mscred_inference(self, output_dir: str, model_path: Path, results: Dict[str, Any]):  # MS-CRED 추론 실행
        """MS-CRED 추론 실행"""  # 메서드 설명
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "mscred-infer",  # CLI mscred-infer 명령
            "--window-counts", str(Path(output_dir) / "window_counts.parquet"),  # 윈도우 카운트 파일
            "--model", str(model_path),  # 학습된 모델 경로
            "--threshold", "95.0"  # 이상 임계값
        ]
        
        try:
            step_result = self._run_command(cmd, "MS-CRED 추론")  # 명령 실행
            results['steps']['mscred_inference'] = step_result  # 결과 저장
            
            # 결과 파일 확인  # 추론 결과 파일 확인
            inference_file = Path(output_dir) / "mscred_infer.parquet"  # 추론 결과 파일 경로
            if inference_file.exists():  # 파일 존재
                results['files']['mscred_inference'] = str(inference_file)  # 추론 파일 경로 저장
                
        except subprocess.CalledProcessError as e:  # 명령 실행 실패
            logger.warning(f"MS-CRED 추론 실패 (계속 진행): {e}")  # 경고 로그
            results['steps']['mscred_inference'] = {
                'status': 'failed',  # 실패 상태
                'error': str(e)  # 에러 메시지
            }
    
    def _generate_report(self, output_dir: str, results: Dict[str, Any]):  # 리포트 생성
        """리포트 생성"""  # 메서드 설명
        cmd = [
            self.python_cmd, "-m", "anomaly_log_detector.cli", "report",  # CLI report 명령
            "--processed-dir", output_dir  # 처리된 디렉토리
        ]
        
        step_result = self._run_command(cmd, "리포트 생성")  # 명령 실행
        results['steps']['report'] = step_result  # 결과 저장
        
        # 리포트 파일 확인  # report.md 파일 확인
        report_file = Path(output_dir) / "report.md"  # 리포트 파일 경로
        if report_file.exists():  # 파일 존재
            results['files']['report'] = str(report_file)  # 리포트 파일 경로 저장


def main():  # CLI 인터페이스 메인 함수
    """CLI 인터페이스"""  # 함수 설명
    import argparse  # 인자 파싱
    
    parser = argparse.ArgumentParser(description="배치 학습 파이프라인 실행")  # 파서 생성
    parser.add_argument("log_file", help="입력 로그 파일")  # 로그 파일 인자
    parser.add_argument("--output-dir", help="출력 디렉토리")  # 출력 디렉토리 인자
    parser.add_argument("--config", help="설정 JSON 파일")  # 설정 파일 인자
    parser.add_argument("--seq-len", type=int, default=50, help="시퀀스 길이")  # 시퀀스 길이 인자
    parser.add_argument("--deeplog-epochs", type=int, default=3, help="DeepLog 에포크")  # DeepLog 에포크 인자
    parser.add_argument("--mscred-epochs", type=int, default=30, help="MS-CRED 에포크")  # MS-CRED 에포크 인자
    
    args = parser.parse_args()  # 인자 파싱
    
    # 설정 로드  # JSON 설정 파일 로드
    config = {}  # 설정 딕셔너리 초기화
    if args.config and Path(args.config).exists():  # 설정 파일 존재
        with open(args.config, 'r') as f:
            config = json.load(f)  # JSON 로드
    
    # CLI 인수로 설정 업데이트  # 명령줄 인자로 설정 병합
    config.update({
        'seq_len': args.seq_len,  # 시퀀스 길이
        'deeplog_epochs': args.deeplog_epochs,  # DeepLog 에포크
        'mscred_epochs': args.mscred_epochs  # MS-CRED 에포크
    })
    
    # 학습 실행  # 파이프라인 실행
    trainer = BatchTrainer()  # 학습기 생성
    results = trainer.train_full_pipeline(
        args.log_file,  # 로그 파일
        args.output_dir,  # 출력 디렉토리
        config  # 설정
    )
    
    print(f"\n🎉 학습 완료!")  # 완료 메시지
    print(f"📂 결과 디렉토리: {results['output_dir']}")  # 결과 디렉토리 출력
    print(f"📊 학습된 모델:")  # 모델 목록 헤더
    for model_name, model_info in results.get('models', {}).items():  # 각 모델 정보
        print(f"  - {model_name}: {model_info['path']}")  # 모델 경로 출력


if __name__ == "__main__":  # 스크립트 직접 실행
    main()  # 메인 함수 호출
