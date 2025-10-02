#!/usr/bin/env python3
"""
자동 모델 변환 및 배포 스크립트
학습 완료 후 자동으로 ONNX 변환 및 C 엔진 배포 준비
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .model_converter import ModelConverter
from .batch_trainer import BatchTrainer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWatcher(FileSystemEventHandler):
    """모델 파일 변경 감지 및 자동 변환"""
    
    def __init__(self, converter: ModelConverter, watch_patterns: Dict[str, str]):
        self.converter = converter
        self.watch_patterns = watch_patterns  # {파일패턴: 모델타입}
        self.last_processed = {}  # 중복 처리 방지
    
    def on_created(self, event):
        """새 파일 생성 시 처리"""
        if not event.is_directory:
            self._process_file(event.src_path)
    
    def on_modified(self, event):
        """파일 수정 시 처리"""
        if not event.is_directory:
            self._process_file(event.src_path)
    
    def _process_file(self, file_path: str):
        """파일 처리 (모델 변환)"""
        file_path = Path(file_path)
        
        # 중복 처리 방지 (1분 내 같은 파일)
        now = time.time()
        if file_path.name in self.last_processed:
            if now - self.last_processed[file_path.name] < 60:
                return
        
        self.last_processed[file_path.name] = now
        
        # 패턴 매칭
        for pattern, model_type in self.watch_patterns.items():
            if pattern in file_path.name and file_path.suffix == '.pth':
                logger.info(f"🔍 새 {model_type} 모델 감지: {file_path}")
                self._convert_model(file_path, model_type)
                break
    
    def _convert_model(self, model_path: Path, model_type: str):
        """모델 변환 실행"""
        try:
            if model_type == 'deeplog':
                # vocab.json 찾기
                vocab_path = self._find_vocab_file(model_path)
                if vocab_path:
                    result = self.converter.convert_deeplog_to_onnx(
                        str(model_path), str(vocab_path)
                    )
                    logger.info(f"✅ DeepLog 자동 변환 완료: {result['onnx_path']}")
                else:
                    logger.warning("vocab.json을 찾을 수 없어 DeepLog 변환 건너뜀")
            
            elif model_type == 'mscred':
                result = self.converter.convert_mscred_to_onnx(str(model_path))
                logger.info(f"✅ MS-CRED 자동 변환 완료: {result['onnx_path']}")
            
        except Exception as e:
            logger.error(f"❌ {model_type} 자동 변환 실패: {e}")
    
    def _find_vocab_file(self, model_path: Path) -> Optional[Path]:
        """DeepLog용 vocab.json 파일 찾기"""
        # 같은 디렉토리에서 찾기
        vocab_path = model_path.parent / "vocab.json"
        if vocab_path.exists():
            return vocab_path
        
        # 상위 디렉토리들에서 찾기
        for parent in model_path.parents:
            vocab_path = parent / "vocab.json"
            if vocab_path.exists():
                return vocab_path
        
        # data/processed 디렉토리에서 찾기
        data_dirs = ["data/processed", "../data/processed", "../../data/processed"]
        for data_dir in data_dirs:
            vocab_path = Path(data_dir) / "vocab.json"
            if vocab_path.exists():
                return vocab_path
        
        return None


class AutoConverter:
    """자동 변환 및 배포 관리자"""
    
    def __init__(self, 
                 models_dir: str = "models",
                 onnx_dir: str = "models/onnx",
                 deployment_dir: str = "models/deployment"):
        self.models_dir = Path(models_dir)
        self.onnx_dir = Path(onnx_dir)
        self.deployment_dir = Path(deployment_dir)
        
        # 디렉토리 생성
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        self.converter = ModelConverter(str(self.onnx_dir))
        self.observer = None
    
    def start_watching(self):
        """모델 디렉토리 감시 시작"""
        watch_patterns = {
            'deeplog': 'deeplog',
            'mscred': 'mscred'
        }
        
        event_handler = ModelWatcher(self.converter, watch_patterns)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.models_dir), recursive=True)
        self.observer.start()
        
        logger.info(f"🔍 모델 디렉토리 감시 시작: {self.models_dir}")
        logger.info("새로운 모델이 생성되면 자동으로 ONNX 변환을 수행합니다.")
    
    def stop_watching(self):
        """모델 디렉토리 감시 중단"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("🛑 모델 감시 중단")
    
    def convert_existing_models(self) -> Dict[str, Any]:
        """기존 모델들을 일괄 변환"""
        logger.info("🔄 기존 모델 일괄 변환 시작")
        
        results = {}
        
        # DeepLog 모델 찾기
        deeplog_models = list(self.models_dir.glob("*deeplog*.pth"))
        for model_path in deeplog_models:
            try:
                vocab_path = self._find_vocab_for_model(model_path)
                if vocab_path:
                    result = self.converter.convert_deeplog_to_onnx(
                        str(model_path), str(vocab_path)
                    )
                    results[f"deeplog_{model_path.stem}"] = result
                    logger.info(f"✅ DeepLog 변환: {model_path.name}")
                else:
                    logger.warning(f"⚠️ vocab.json 없음: {model_path.name}")
            except Exception as e:
                logger.error(f"❌ DeepLog 변환 실패 {model_path.name}: {e}")
        
        # MS-CRED 모델 찾기
        mscred_models = list(self.models_dir.glob("*mscred*.pth"))
        for model_path in mscred_models:
            try:
                result = self.converter.convert_mscred_to_onnx(str(model_path))
                results[f"mscred_{model_path.stem}"] = result
                logger.info(f"✅ MS-CRED 변환: {model_path.name}")
            except Exception as e:
                logger.error(f"❌ MS-CRED 변환 실패 {model_path.name}: {e}")
        
        logger.info(f"🎉 일괄 변환 완료: {len(results)}개 모델")
        return results
    
    def _find_vocab_for_model(self, model_path: Path) -> Optional[Path]:
        """모델에 해당하는 vocab.json 찾기"""
        # 모델과 같은 디렉토리
        vocab_path = model_path.parent / "vocab.json"
        if vocab_path.exists():
            return vocab_path
        
        # 일반적인 위치들 검색
        search_paths = [
            "data/processed/vocab.json",
            "../data/processed/vocab.json",
            "../../data/processed/vocab.json",
            self.models_dir / "vocab.json"
        ]
        
        for search_path in search_paths:
            vocab_path = Path(search_path)
            if vocab_path.exists():
                return vocab_path
        
        return None
    
    def prepare_deployment_package(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """C 엔진 배포용 패키지 준비"""
        logger.info("📦 배포 패키지 준비 중...")
        
        deployment_info = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'files': []
        }
        
        # ONNX 모델들 복사
        onnx_files = list(self.onnx_dir.glob("*.onnx"))
        for onnx_file in onnx_files:
            if model_name and model_name not in onnx_file.name:
                continue
            
            # 배포 디렉토리로 복사
            dest_path = self.deployment_dir / onnx_file.name
            shutil.copy2(onnx_file, dest_path)
            
            # 메타데이터도 복사
            meta_file = onnx_file.with_suffix('.onnx.meta.json')
            if meta_file.exists():
                meta_dest = self.deployment_dir / meta_file.name
                shutil.copy2(meta_file, meta_dest)
                deployment_info['files'].append(str(meta_dest))
            
            deployment_info['models'][onnx_file.stem] = {
                'onnx_path': str(dest_path),
                'size_mb': dest_path.stat().st_size / (1024 * 1024)
            }
            deployment_info['files'].append(str(dest_path))
        
        # vocab.json 복사
        vocab_file = self.onnx_dir / "vocab.json"
        if vocab_file.exists():
            vocab_dest = self.deployment_dir / "vocab.json"
            shutil.copy2(vocab_file, vocab_dest)
            deployment_info['files'].append(str(vocab_dest))
        
        # 배포 정보 저장
        info_file = self.deployment_dir / "deployment_info.json"
        with open(info_file, 'w') as f:
            json.dump(deployment_info, f, indent=2, default=str)
        
        logger.info(f"✅ 배포 패키지 준비 완료: {self.deployment_dir}")
        logger.info(f"📊 포함된 모델: {list(deployment_info['models'].keys())}")
        
        return deployment_info
    
    def run_full_pipeline(self, log_file: str, auto_deploy: bool = True) -> Dict[str, Any]:
        """전체 파이프라인 실행 (학습 → 변환 → 배포)"""
        logger.info("🚀 전체 하이브리드 파이프라인 시작")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'log_file': log_file,
            'stages': {}
        }
        
        try:
            # 1단계: 배치 학습
            logger.info("1️⃣ 배치 학습 단계")
            trainer = BatchTrainer(models_dir=str(self.models_dir))
            training_results = trainer.train_full_pipeline(log_file)
            pipeline_results['stages']['training'] = training_results
            
            # 2단계: ONNX 변환
            logger.info("2️⃣ ONNX 변환 단계")
            conversion_results = {}
            
            for model_name, model_info in training_results.get('models', {}).items():
                try:
                    model_path = model_info['path']
                    
                    if model_name == 'deeplog':
                        vocab_path = training_results['files']['vocab']
                        result = self.converter.convert_deeplog_to_onnx(
                            model_path, vocab_path
                        )
                    elif model_name == 'mscred':
                        result = self.converter.convert_mscred_to_onnx(model_path)
                    
                    conversion_results[model_name] = result
                    logger.info(f"✅ {model_name} 변환 완료")
                    
                except Exception as e:
                    logger.error(f"❌ {model_name} 변환 실패: {e}")
                    conversion_results[model_name] = {'error': str(e)}
            
            pipeline_results['stages']['conversion'] = conversion_results
            
            # 3단계: 배포 준비
            if auto_deploy and conversion_results:
                logger.info("3️⃣ 배포 준비 단계")
                deployment_info = self.prepare_deployment_package()
                pipeline_results['stages']['deployment'] = deployment_info
            
            pipeline_results['status'] = 'success'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            logger.info("🎉 전체 하이브리드 파이프라인 완료!")
            
        except Exception as e:
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
            logger.error(f"❌ 파이프라인 실패: {e}")
            raise
        
        # 결과 저장
        results_file = self.deployment_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        return pipeline_results


def main():
    """CLI 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(description="자동 모델 변환 및 배포")
    parser.add_argument("--mode", choices=['watch', 'convert', 'pipeline'], 
                       default='convert', help="실행 모드")
    parser.add_argument("--log-file", help="학습용 로그 파일 (pipeline 모드)")
    parser.add_argument("--models-dir", default="models", help="모델 디렉토리")
    parser.add_argument("--onnx-dir", default="models/onnx", help="ONNX 출력 디렉토리")
    parser.add_argument("--deployment-dir", default="models/deployment", help="배포 디렉토리")
    
    args = parser.parse_args()
    
    converter = AutoConverter(args.models_dir, args.onnx_dir, args.deployment_dir)
    
    if args.mode == 'watch':
        # 감시 모드
        converter.start_watching()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            converter.stop_watching()
            logger.info("👋 감시 모드 종료")
    
    elif args.mode == 'convert':
        # 변환 모드
        results = converter.convert_existing_models()
        if results:
            converter.prepare_deployment_package()
        print(f"✅ 변환 완료: {len(results)}개 모델")
    
    elif args.mode == 'pipeline':
        # 전체 파이프라인 모드
        if not args.log_file:
            print("❌ pipeline 모드는 --log-file이 필요합니다")
            return
        
        results = converter.run_full_pipeline(args.log_file)
        print(f"✅ 파이프라인 완료: {results['status']}")


if __name__ == "__main__":
    main()
