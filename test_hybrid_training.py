#!/usr/bin/env python3
"""
하이브리드 시스템 Python 학습 컴포넌트 테스트
Phase 1.1 & 1.2 검증용 스크립트
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

def test_batch_trainer():
    """배치 학습 파이프라인 테스트"""
    print("🧪 배치 학습 파이프라인 테스트...")
    
    try:
        # 테스트용 임시 디렉토리
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 테스트 로그 파일 생성
            test_log = temp_path / "test.log"
            test_log.write_text("""
2024-01-01 10:00:01 INFO User alice logged in
2024-01-01 10:00:02 INFO Processing request 12345
2024-01-01 10:00:03 ERROR Authentication failed for user bob
2024-01-01 10:00:04 INFO Database query completed in 150ms
2024-01-01 10:00:05 CRITICAL System overload detected
2024-01-01 10:00:06 INFO User alice logged out
""".strip())
            
            # 배치 학습 실행 (간단한 설정)
            sys.path.append(str(Path(__file__).parent))
            from hybrid_system.training.batch_trainer import BatchTrainer
            
            trainer = BatchTrainer(
                cache_dir=str(temp_path / "cache"),
                models_dir=str(temp_path / "models")
            )
            
            # 빠른 테스트를 위한 설정
            config = {
                'seq_len': 5,
                'deeplog_epochs': 1,
                'mscred_epochs': 1,
                'window_size': 10,
                'stride': 5
            }
            
            results = trainer.train_full_pipeline(
                str(test_log),
                str(temp_path / "output"),
                config
            )
            
            print("✅ 배치 학습 테스트 성공")
            print(f"   - 상태: {results['status']}")
            print(f"   - 단계: {len(results['steps'])}개 완료")
            print(f"   - 모델: {len(results.get('models', {}))}개 생성")
            
            return results
            
    except Exception as e:
        print(f"❌ 배치 학습 테스트 실패: {e}")
        return None

def test_model_converter():
    """ONNX 모델 변환 테스트"""
    print("\n🧪 ONNX 모델 변환 테스트...")
    
    try:
        # 기존 모델 파일 찾기
        model_files = list(Path(".").glob("**/*.pth"))
        vocab_files = list(Path(".").glob("**/vocab.json"))
        
        if not model_files:
            print("⚠️ 테스트용 PyTorch 모델을 찾을 수 없습니다.")
            print("   먼저 기본 학습을 실행하세요: ./run_full_pipeline_pip.sh test_sample.log")
            return None
        
        if not vocab_files:
            print("⚠️ vocab.json 파일을 찾을 수 없습니다.")
            return None
        
        # 변환 테스트
        sys.path.append(str(Path(__file__).parent))
        from hybrid_system.training.model_converter import ModelConverter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            converter = ModelConverter(temp_dir)
            
            # DeepLog 모델 찾기
            deeplog_model = None
            for model_file in model_files:
                if 'deeplog' in model_file.name.lower():
                    deeplog_model = model_file
                    break
            
            if deeplog_model:
                result = converter.convert_deeplog_to_onnx(
                    str(deeplog_model),
                    str(vocab_files[0])
                )
                
                print("✅ DeepLog ONNX 변환 성공")
                print(f"   - ONNX 파일: {result['onnx_path']}")
                print(f"   - 메타데이터: {result['metadata_path']}")
                
                # 검증
                if converter.validate_onnx_model(result['onnx_path']):
                    print("✅ ONNX 모델 검증 성공")
                else:
                    print("❌ ONNX 모델 검증 실패")
                
                return result
            else:
                print("⚠️ DeepLog 모델을 찾을 수 없습니다.")
                return None
                
    except Exception as e:
        print(f"❌ ONNX 변환 테스트 실패: {e}")
        return None

def test_auto_converter():
    """자동 변환기 테스트"""
    print("\n🧪 자동 변환기 테스트...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from hybrid_system.training.auto_converter import AutoConverter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            converter = AutoConverter(
                models_dir=str(temp_path / "models"),
                onnx_dir=str(temp_path / "onnx"),
                deployment_dir=str(temp_path / "deployment")
            )
            
            # 기존 모델 변환 테스트
            results = converter.convert_existing_models()
            
            print("✅ 자동 변환기 테스트 성공")
            print(f"   - 변환된 모델: {len(results)}개")
            
            if results:
                # 배포 패키지 준비 테스트
                deployment_info = converter.prepare_deployment_package()
                print("✅ 배포 패키지 준비 성공")
                print(f"   - 배포 모델: {len(deployment_info['models'])}개")
                print(f"   - 배포 파일: {len(deployment_info['files'])}개")
            
            return results
            
    except Exception as e:
        print(f"❌ 자동 변환기 테스트 실패: {e}")
        return None

def test_cli_integration():
    """CLI 통합 테스트"""
    print("\n🧪 CLI 통합 테스트...")
    
    try:
        import subprocess
        
        # CLI 도움말 테스트
        result = subprocess.run([
            sys.executable, "-m", "study_preprocessor.cli", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ CLI 기본 동작 확인")
            
            # 새로운 명령어 확인
            if "convert-onnx" in result.stdout:
                print("✅ convert-onnx 명령어 등록됨")
            else:
                print("⚠️ convert-onnx 명령어 미등록")
            
            if "hybrid-pipeline" in result.stdout:
                print("✅ hybrid-pipeline 명령어 등록됨")
            else:
                print("⚠️ hybrid-pipeline 명령어 미등록")
            
            return True
        else:
            print(f"❌ CLI 테스트 실패: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI 통합 테스트 실패: {e}")
        return False

def check_dependencies():
    """의존성 확인"""
    print("🔍 의존성 확인...")
    
    required_packages = [
        'torch',
        'onnx',
        'onnxruntime',
        'watchdog',
        'pandas',
        'click'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (누락)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 누락된 패키지 설치:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("   또는: pip install -r requirements_hybrid.txt")
        return False
    
    return True

def main():
    """메인 테스트 실행"""
    print("🚀 하이브리드 시스템 Python 컴포넌트 테스트 시작")
    print("=" * 60)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ 의존성 문제로 테스트 중단")
        return False
    
    print("\n" + "=" * 60)
    
    # 테스트 실행
    results = {}
    
    # 1. 배치 학습 테스트
    results['batch_trainer'] = test_batch_trainer()
    
    # 2. ONNX 변환 테스트
    results['model_converter'] = test_model_converter()
    
    # 3. 자동 변환기 테스트
    results['auto_converter'] = test_auto_converter()
    
    # 4. CLI 통합 테스트
    results['cli_integration'] = test_cli_integration()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약:")
    
    success_count = 0
    total_count = len(results)
    
    for test_name, result in results.items():
        if result:
            print(f"✅ {test_name}: 성공")
            success_count += 1
        else:
            print(f"❌ {test_name}: 실패")
    
    print(f"\n🎯 성공률: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    
    if success_count == total_count:
        print("🎉 모든 테스트 통과! Phase 1.1 & 1.2 완료")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 문제를 해결하고 다시 시도하세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
