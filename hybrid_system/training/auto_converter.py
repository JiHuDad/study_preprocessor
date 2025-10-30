#!/usr/bin/env python3  # 실행 스크립트 셔뱅
"""자동 모델 변환 및 배포 스크립트 요약

- 목적: PyTorch 모델 학습 완료 후 자동으로 ONNX 변환 및 C 엔진 배포 패키지 준비
- 주요 기능:
  * 파일 시스템 감시(watchdog)를 통한 모델 파일 자동 감지 및 변환
  * 기존 모델 일괄 변환 (DeepLog, MS-CRED)
  * ONNX 모델을 배포 디렉토리로 복사 및 배포 패키지 생성
  * 전체 파이프라인 실행 (학습 → 변환 → 배포) 자동화
- 실행 모드:
  * watch: 모델 디렉토리 감시 모드 (새 모델 생성 시 자동 변환)
  * convert: 기존 모델 일괄 변환 모드
  * pipeline: 전체 파이프라인 모드 (학습부터 배포까지)
- 출력: ONNX 모델(.onnx), 메타데이터(.meta.json), vocab.json, 배포 정보(JSON)
"""  # 모듈 요약 설명
import os  # 환경 변수/경로 유틸
import json  # JSON 입출력
import time  # 시간 처리
import shutil  # 파일 복사/이동
from pathlib import Path  # 경로 처리
from typing import Dict, Any, Optional  # 타입 힌트
import logging  # 로깅 프레임워크
from datetime import datetime  # 날짜/시간 처리
from watchdog.observers import Observer  # 파일 시스템 감시자
from watchdog.events import FileSystemEventHandler  # 파일 시스템 이벤트 핸들러

from .model_converter import ModelConverter  # 모델 변환기
from .batch_trainer import BatchTrainer  # 배치 학습기

# 로깅 설정  # INFO 레벨 기본 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # 모듈 로거 생성


class ModelWatcher(FileSystemEventHandler):  # 모델 파일 감시자 클래스
    """모델 파일 변경 감지 및 자동 변환"""  # 클래스 설명
    
    def __init__(self, converter: ModelConverter, watch_patterns: Dict[str, str]):  # 초기화
        self.converter = converter  # 모델 변환기
        self.watch_patterns = watch_patterns  # {파일패턴: 모델타입}  # 감시 패턴 딕셔너리
        self.last_processed = {}  # 중복 처리 방지  # 처리 이력 딕셔너리
    
    def on_created(self, event):  # 파일 생성 이벤트 핸들러
        """새 파일 생성 시 처리"""  # 메서드 설명
        if not event.is_directory:  # 디렉토리가 아님
            self._process_file(event.src_path)  # 파일 처리
    
    def on_modified(self, event):  # 파일 수정 이벤트 핸들러
        """파일 수정 시 처리"""  # 메서드 설명
        if not event.is_directory:  # 디렉토리가 아님
            self._process_file(event.src_path)  # 파일 처리
    
    def _process_file(self, file_path: str):  # 파일 처리
        """파일 처리 (모델 변환)"""  # 메서드 설명
        file_path = Path(file_path)  # 경로 객체로 변환
        
        # 중복 처리 방지 (1분 내 같은 파일)  # 중복 변환 방지
        now = time.time()  # 현재 시간
        if file_path.name in self.last_processed:  # 이전에 처리한 파일
            if now - self.last_processed[file_path.name] < 60:  # 1분 이내
                return  # 처리 건너뛰기
        
        self.last_processed[file_path.name] = now  # 처리 시간 기록
        
        # 패턴 매칭  # 모델 타입 감지
        for pattern, model_type in self.watch_patterns.items():  # 각 패턴 확인
            if pattern in file_path.name and file_path.suffix == '.pth':  # 모델 파일 확인
                logger.info(f"🔍 새 {model_type} 모델 감지: {file_path}")  # 감지 로그
                self._convert_model(file_path, model_type)  # 모델 변환
                break  # 처리 완료
    
    def _convert_model(self, model_path: Path, model_type: str):  # 모델 변환 실행
        """모델 변환 실행"""  # 메서드 설명
        try:
            if model_type == 'deeplog':  # DeepLog 모델
                # vocab.json 찾기  # 어휘 사전 파일 찾기
                vocab_path = self._find_vocab_file(model_path)  # vocab 파일 검색
                if vocab_path:  # vocab 파일 발견
                    result = self.converter.convert_deeplog_to_onnx(
                        str(model_path), str(vocab_path)  # DeepLog 변환 호출
                    )
                    logger.info(f"✅ DeepLog 자동 변환 완료: {result['onnx_path']}")  # 완료 로그
                else:  # vocab 파일 없음
                    logger.warning("vocab.json을 찾을 수 없어 DeepLog 변환 건너뜀")  # 경고 로그
            
            elif model_type == 'mscred':  # MS-CRED 모델
                result = self.converter.convert_mscred_to_onnx(str(model_path))  # MS-CRED 변환 호출
                logger.info(f"✅ MS-CRED 자동 변환 완료: {result['onnx_path']}")  # 완료 로그
            
        except Exception as e:
            logger.error(f"❌ {model_type} 자동 변환 실패: {e}")  # 에러 로그
    
    def _find_vocab_file(self, model_path: Path) -> Optional[Path]:  # vocab 파일 찾기
        """DeepLog용 vocab.json 파일 찾기"""  # 메서드 설명
        # 같은 디렉토리에서 찾기  # 모델과 같은 폴더 검색
        vocab_path = model_path.parent / "vocab.json"  # vocab 파일 경로
        if vocab_path.exists():  # 파일 존재
            return vocab_path  # 경로 반환
        
        # 상위 디렉토리들에서 찾기  # 상위 폴더들 검색
        for parent in model_path.parents:  # 각 상위 폴더 확인
            vocab_path = parent / "vocab.json"  # vocab 파일 경로
            if vocab_path.exists():  # 파일 존재
                return vocab_path  # 경로 반환
        
        # data/processed 디렉토리에서 찾기  # 일반적인 위치 검색
        data_dirs = ["data/processed", "../data/processed", "../../data/processed"]  # 검색 경로 목록
        for data_dir in data_dirs:  # 각 경로 확인
            vocab_path = Path(data_dir) / "vocab.json"  # vocab 파일 경로
            if vocab_path.exists():  # 파일 존재
                return vocab_path  # 경로 반환
        
        return None  # 찾지 못함


class AutoConverter:  # 자동 변환 및 배포 관리자 클래스
    """자동 변환 및 배포 관리자"""  # 클래스 설명
    
    def __init__(self, 
                 models_dir: str = "models",  # 모델 디렉토리
                 onnx_dir: str = "models/onnx",  # ONNX 출력 디렉토리
                 deployment_dir: str = "models/deployment"):  # 배포 디렉토리
        self.models_dir = Path(models_dir)  # 모델 경로
        self.onnx_dir = Path(onnx_dir)  # ONNX 경로
        self.deployment_dir = Path(deployment_dir)  # 배포 경로
        
        # 디렉토리 생성  # 필요한 폴더 생성
        self.models_dir.mkdir(parents=True, exist_ok=True)  # 모델 폴더 생성
        self.onnx_dir.mkdir(parents=True, exist_ok=True)  # ONNX 폴더 생성
        self.deployment_dir.mkdir(parents=True, exist_ok=True)  # 배포 폴더 생성
        
        self.converter = ModelConverter(str(self.onnx_dir))  # 모델 변환기 생성
        self.observer = None  # 파일 감시자 초기화
    
    def start_watching(self):  # 모델 디렉토리 감시 시작
        """모델 디렉토리 감시 시작"""  # 메서드 설명
        watch_patterns = {
            'deeplog': 'deeplog',  # DeepLog 패턴
            'mscred': 'mscred'  # MS-CRED 패턴
        }
        
        event_handler = ModelWatcher(self.converter, watch_patterns)  # 이벤트 핸들러 생성
        self.observer = Observer()  # 감시자 생성
        self.observer.schedule(event_handler, str(self.models_dir), recursive=True)  # 감시 등록
        self.observer.start()  # 감시 시작
        
        logger.info(f"🔍 모델 디렉토리 감시 시작: {self.models_dir}")  # 시작 로그
        logger.info("새로운 모델이 생성되면 자동으로 ONNX 변환을 수행합니다.")  # 안내 메시지
    
    def stop_watching(self):  # 모델 디렉토리 감시 중단
        """모델 디렉토리 감시 중단"""  # 메서드 설명
        if self.observer:  # 감시자 존재
            self.observer.stop()  # 감시 중단
            self.observer.join()  # 스레드 종료 대기
            logger.info("🛑 모델 감시 중단")  # 중단 로그
    
    def convert_existing_models(self) -> Dict[str, Any]:  # 기존 모델 일괄 변환
        """기존 모델들을 일괄 변환"""  # 메서드 설명
        logger.info("🔄 기존 모델 일괄 변환 시작")  # 시작 로그
        
        results = {}  # 결과 딕셔너리
        
        # DeepLog 모델 찾기  # DeepLog 모델 검색
        deeplog_models = list(self.models_dir.glob("*deeplog*.pth"))  # DeepLog 모델 목록
        for model_path in deeplog_models:  # 각 모델 처리
            try:
                vocab_path = self._find_vocab_for_model(model_path)  # vocab 파일 찾기
                if vocab_path:  # vocab 파일 발견
                    result = self.converter.convert_deeplog_to_onnx(
                        str(model_path), str(vocab_path)  # DeepLog 변환 호출
                    )
                    results[f"deeplog_{model_path.stem}"] = result  # 결과 저장
                    logger.info(f"✅ DeepLog 변환: {model_path.name}")  # 완료 로그
                else:  # vocab 파일 없음
                    logger.warning(f"⚠️ vocab.json 없음: {model_path.name}")  # 경고 로그
            except Exception as e:
                logger.error(f"❌ DeepLog 변환 실패 {model_path.name}: {e}")  # 에러 로그
        
        # MS-CRED 모델 찾기  # MS-CRED 모델 검색
        mscred_models = list(self.models_dir.glob("*mscred*.pth"))  # MS-CRED 모델 목록
        for model_path in mscred_models:  # 각 모델 처리
            try:
                result = self.converter.convert_mscred_to_onnx(str(model_path))  # MS-CRED 변환 호출
                results[f"mscred_{model_path.stem}"] = result  # 결과 저장
                logger.info(f"✅ MS-CRED 변환: {model_path.name}")  # 완료 로그
            except Exception as e:
                logger.error(f"❌ MS-CRED 변환 실패 {model_path.name}: {e}")  # 에러 로그
        
        logger.info(f"🎉 일괄 변환 완료: {len(results)}개 모델")  # 완료 로그
        return results  # 결과 반환
    
    def _find_vocab_for_model(self, model_path: Path) -> Optional[Path]:  # vocab 파일 찾기
        """모델에 해당하는 vocab.json 찾기"""  # 메서드 설명
        # 모델과 같은 디렉토리  # 같은 폴더 검색
        vocab_path = model_path.parent / "vocab.json"  # vocab 파일 경로
        if vocab_path.exists():  # 파일 존재
            return vocab_path  # 경로 반환
        
        # 일반적인 위치들 검색  # 일반적인 위치 검색
        search_paths = [
            "data/processed/vocab.json",  # 일반적인 경로 1
            "../data/processed/vocab.json",  # 일반적인 경로 2
            "../../data/processed/vocab.json",  # 일반적인 경로 3
            self.models_dir / "vocab.json"  # 모델 디렉토리
        ]
        
        for search_path in search_paths:  # 각 경로 확인
            vocab_path = Path(search_path)  # 경로 객체 생성
            if vocab_path.exists():  # 파일 존재
                return vocab_path  # 경로 반환
        
        return None  # 찾지 못함
    
    def prepare_deployment_package(self, model_name: Optional[str] = None) -> Dict[str, Any]:  # 배포 패키지 준비
        """C 엔진 배포용 패키지 준비"""  # 메서드 설명
        logger.info("📦 배포 패키지 준비 중...")  # 시작 로그
        
        deployment_info = {
            'timestamp': datetime.now().isoformat(),  # 타임스탬프
            'models': {},  # 모델 정보
            'files': []  # 파일 목록
        }
        
        # ONNX 모델들 복사  # ONNX 모델 복사
        onnx_files = list(self.onnx_dir.glob("*.onnx"))  # ONNX 파일 목록
        for onnx_file in onnx_files:  # 각 파일 처리
            if model_name and model_name not in onnx_file.name:  # 필터링
                continue  # 건너뛰기
            
            # 배포 디렉토리로 복사  # ONNX 파일 복사
            dest_path = self.deployment_dir / onnx_file.name  # 대상 경로
            shutil.copy2(onnx_file, dest_path)  # 파일 복사
            
            # 메타데이터도 복사  # 메타데이터 파일 복사
            meta_file = onnx_file.with_suffix('.onnx.meta.json')  # 메타데이터 파일 경로
            if meta_file.exists():  # 파일 존재
                meta_dest = self.deployment_dir / meta_file.name  # 대상 경로
                shutil.copy2(meta_file, meta_dest)  # 파일 복사
                deployment_info['files'].append(str(meta_dest))  # 파일 목록 추가
            
            deployment_info['models'][onnx_file.stem] = {
                'onnx_path': str(dest_path),  # ONNX 경로
                'size_mb': dest_path.stat().st_size / (1024 * 1024)  # 파일 크기 (MB)
            }
            deployment_info['files'].append(str(dest_path))  # 파일 목록 추가
        
        # vocab.json 복사  # 어휘 사전 파일 복사
        vocab_file = self.onnx_dir / "vocab.json"  # vocab 파일 경로
        if vocab_file.exists():  # 파일 존재
            vocab_dest = self.deployment_dir / "vocab.json"  # 대상 경로
            shutil.copy2(vocab_file, vocab_dest)  # 파일 복사
            deployment_info['files'].append(str(vocab_dest))  # 파일 목록 추가
        
        # 배포 정보 저장  # 배포 정보 JSON 저장
        info_file = self.deployment_dir / "deployment_info.json"  # 정보 파일 경로
        with open(info_file, 'w') as f:
            json.dump(deployment_info, f, indent=2, default=str)  # JSON 저장
        
        logger.info(f"✅ 배포 패키지 준비 완료: {self.deployment_dir}")  # 완료 로그
        logger.info(f"📊 포함된 모델: {list(deployment_info['models'].keys())}")  # 모델 목록 로그
        
        return deployment_info  # 배포 정보 반환
    
    def run_full_pipeline(self, log_file: str, auto_deploy: bool = True) -> Dict[str, Any]:  # 전체 파이프라인 실행
        """전체 파이프라인 실행 (학습 → 변환 → 배포)"""  # 메서드 설명
        logger.info("🚀 전체 하이브리드 파이프라인 시작")  # 시작 로그
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),  # 시작 시간
            'log_file': log_file,  # 로그 파일
            'stages': {}  # 단계별 결과
        }
        
        try:
            # 1단계: 배치 학습  # 배치 학습 실행
            logger.info("1️⃣ 배치 학습 단계")
            trainer = BatchTrainer(models_dir=str(self.models_dir))  # 학습기 생성
            training_results = trainer.train_full_pipeline(log_file)  # 학습 실행
            pipeline_results['stages']['training'] = training_results  # 결과 저장
            
            # 2단계: ONNX 변환  # ONNX 변환 실행
            logger.info("2️⃣ ONNX 변환 단계")
            conversion_results = {}  # 변환 결과 딕셔너리
            
            for model_name, model_info in training_results.get('models', {}).items():  # 각 모델 변환
                try:
                    model_path = model_info['path']  # 모델 경로
                    
                    if model_name == 'deeplog':  # DeepLog 모델
                        vocab_path = training_results['files']['vocab']  # vocab 파일 경로
                        result = self.converter.convert_deeplog_to_onnx(
                            model_path, vocab_path  # DeepLog 변환 호출
                        )
                    elif model_name == 'mscred':  # MS-CRED 모델
                        result = self.converter.convert_mscred_to_onnx(model_path)  # MS-CRED 변환 호출
                    
                    conversion_results[model_name] = result  # 결과 저장
                    logger.info(f"✅ {model_name} 변환 완료")  # 완료 로그
                    
                except Exception as e:
                    logger.error(f"❌ {model_name} 변환 실패: {e}")  # 에러 로그
                    conversion_results[model_name] = {'error': str(e)}  # 에러 저장
            
            pipeline_results['stages']['conversion'] = conversion_results  # 변환 결과 저장
            
            # 3단계: 배포 준비  # 배포 패키지 준비
            if auto_deploy and conversion_results:  # 자동 배포 활성화 및 변환 성공
                logger.info("3️⃣ 배포 준비 단계")
                deployment_info = self.prepare_deployment_package()  # 배포 패키지 준비
                pipeline_results['stages']['deployment'] = deployment_info  # 배포 정보 저장
            
            pipeline_results['status'] = 'success'  # 성공 상태
            pipeline_results['end_time'] = datetime.now().isoformat()  # 종료 시간
            
            logger.info("🎉 전체 하이브리드 파이프라인 완료!")  # 완료 로그
            
        except Exception as e:
            pipeline_results['status'] = 'failed'  # 실패 상태
            pipeline_results['error'] = str(e)  # 에러 메시지
            pipeline_results['end_time'] = datetime.now().isoformat()  # 종료 시간
            logger.error(f"❌ 파이프라인 실패: {e}")  # 에러 로그
            raise  # 예외 재발생
        
        # 결과 저장  # 파이프라인 결과 JSON 저장
        results_file = self.deployment_dir / "pipeline_results.json"  # 결과 파일 경로
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)  # JSON 저장
        
        return pipeline_results  # 파이프라인 결과 반환


def main():  # CLI 인터페이스 메인 함수
    """CLI 인터페이스"""  # 함수 설명
    import argparse  # 인자 파싱
    
    parser = argparse.ArgumentParser(description="자동 모델 변환 및 배포")  # 파서 생성
    parser.add_argument("--mode", choices=['watch', 'convert', 'pipeline'], 
                       default='convert', help="실행 모드")  # 실행 모드 인자
    parser.add_argument("--log-file", help="학습용 로그 파일 (pipeline 모드)")  # 로그 파일 인자
    parser.add_argument("--models-dir", default="models", help="모델 디렉토리")  # 모델 디렉토리 인자
    parser.add_argument("--onnx-dir", default="models/onnx", help="ONNX 출력 디렉토리")  # ONNX 디렉토리 인자
    parser.add_argument("--deployment-dir", default="models/deployment", help="배포 디렉토리")  # 배포 디렉토리 인자
    
    args = parser.parse_args()  # 인자 파싱
    
    converter = AutoConverter(args.models_dir, args.onnx_dir, args.deployment_dir)  # 변환기 생성
    
    if args.mode == 'watch':  # 감시 모드
        # 감시 모드  # 파일 시스템 감시 시작
        converter.start_watching()  # 감시 시작
        try:
            while True:  # 무한 루프
                time.sleep(1)  # 1초 대기
        except KeyboardInterrupt:  # Ctrl+C 입력
            converter.stop_watching()  # 감시 중단
            logger.info("👋 감시 모드 종료")  # 종료 로그
    
    elif args.mode == 'convert':  # 변환 모드
        # 변환 모드  # 기존 모델 변환
        results = converter.convert_existing_models()  # 모델 변환 실행
        if results:  # 변환 성공
            converter.prepare_deployment_package()  # 배포 패키지 준비
        print(f"✅ 변환 완료: {len(results)}개 모델")  # 완료 메시지
    
    elif args.mode == 'pipeline':  # 파이프라인 모드
        # 전체 파이프라인 모드  # 전체 파이프라인 실행
        if not args.log_file:  # 로그 파일 없음
            print("❌ pipeline 모드는 --log-file이 필요합니다")  # 에러 메시지
            return  # 종료
        
        results = converter.run_full_pipeline(args.log_file)  # 파이프라인 실행
        print(f"✅ 파이프라인 완료: {results['status']}")  # 완료 메시지


if __name__ == "__main__":  # 스크립트 직접 실행
    main()  # 메인 함수 호출
