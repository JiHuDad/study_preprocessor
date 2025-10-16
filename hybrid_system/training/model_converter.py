#!/usr/bin/env python3
"""
PyTorch 모델을 ONNX 포맷으로 변환하는 도구
하이브리드 시스템의 핵심 컴포넌트: Python 학습 → C 추론 브리지
"""

import os
import json
import torch
import torch.onnx
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConverter:
    """PyTorch 모델을 ONNX로 변환하는 클래스"""
    
    def __init__(self, output_dir: str = "models/onnx"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_deeplog_to_onnx(
        self,
        model_path: str,
        vocab_path: str,
        output_name: str = "deeplog.onnx",
        seq_len: int = 50
    ) -> Dict[str, Any]:
        """
        DeepLog 모델을 ONNX로 변환

        Args:
            model_path: PyTorch 모델 파일 경로
            vocab_path: 어휘 사전 파일 경로
            output_name: 출력 ONNX 파일명
            seq_len: 시퀀스 길이

        Returns:
            변환 결과 정보
        """
        logger.info(f"🔄 DeepLog 모델 변환 시작: {model_path}")

        # 어휘 사전 로드
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)

        # PyTorch 모델 로드
        # DeepLog 모델 클래스 import
        import sys
        from pathlib import Path
        # study_preprocessor 패키지 경로 추가
        root_dir = Path(__file__).parent.parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        from study_preprocessor.builders.deeplog import DeepLogLSTM

        # state dict 로드
        state = torch.load(model_path, map_location='cpu')
        model_vocab_size = int(state.get("vocab_size", vocab_size))
        model_seq_len = int(state.get("seq_len", seq_len))

        # 모델 생성 및 가중치 로드
        model = DeepLogLSTM(vocab_size=model_vocab_size)
        model.load_state_dict(state["state_dict"])
        model.eval()

        # seq_len 업데이트 (모델에 저장된 값 사용)
        seq_len = model_seq_len
        
        # 더미 입력 생성 (배치 크기 1, 시퀀스 길이)
        dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        
        # ONNX 변환
        onnx_path = self.output_dir / output_name
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,  # 호환성을 위해 안정적인 버전 사용
            do_constant_folding=True,
            input_names=['input_sequence'],
            output_names=['predictions'],
            dynamic_axes={
                'input_sequence': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        # 메타데이터 저장
        metadata = {
            'model_type': 'deeplog',
            'vocab_size': vocab_size,
            'seq_len': seq_len,
            'input_shape': [1, seq_len],
            'output_shape': [1, vocab_size],
            'input_names': ['input_sequence'],
            'output_names': ['predictions'],
            'opset_version': 11
        }
        
        metadata_path = self.output_dir / f"{output_name}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 어휘 사전도 함께 복사
        vocab_output = self.output_dir / "vocab.json"
        with open(vocab_output, 'w') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ DeepLog 변환 완료: {onnx_path}")
        logger.info(f"📊 메타데이터: {metadata_path}")
        logger.info(f"📚 어휘 사전: {vocab_output}")
        
        return {
            'onnx_path': str(onnx_path),
            'metadata_path': str(metadata_path),
            'vocab_path': str(vocab_output),
            'metadata': metadata
        }
    
    def convert_mscred_to_onnx(
        self,
        model_path: str,
        output_name: str = "mscred.onnx",
        window_size: int = 50,
        feature_dim: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        MS-CRED 모델을 ONNX로 변환
        
        Args:
            model_path: PyTorch 모델 파일 경로
            output_name: 출력 ONNX 파일명
            window_size: 윈도우 크기
            feature_dim: 피처 차원 (자동 감지 시도)
            
        Returns:
            변환 결과 정보
        """
        logger.info(f"🔄 MS-CRED 모델 변환 시작: {model_path}")

        # MS-CRED 모델 클래스 import
        import sys
        from pathlib import Path
        # study_preprocessor 패키지 경로 추가
        root_dir = Path(__file__).parent.parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        from study_preprocessor.mscred_model import MSCREDModel

        # state dict 로드
        state = torch.load(model_path, map_location='cpu')

        # 모델 파라미터 추출
        # MS-CRED는 'model_state_dict' 키로 저장됨
        if isinstance(state, dict):
            if 'model_state_dict' in state:
                # MSCREDTrainer.save_model() 형식
                state_dict = state['model_state_dict']
                saved_feature_dim = state.get('feature_dim', feature_dim)
                saved_window_size = state.get('window_size', window_size)
            elif 'state_dict' in state:
                # 다른 저장 형식
                state_dict = state['state_dict']
                saved_feature_dim = state.get('feature_dim', feature_dim)
                saved_window_size = state.get('window_size', window_size)
            else:
                # state_dict만 있는 경우 (구버전)
                state_dict = state
                saved_feature_dim = feature_dim
                saved_window_size = window_size
        else:
            state_dict = state
            saved_feature_dim = feature_dim
            saved_window_size = window_size

        # 모델 생성
        # MSCREDModel의 파라미터: input_channels, base_channels
        # feature_dim은 ONNX export시 입력 크기 결정에만 사용
        try:
            # 기본값으로 모델 생성 (input_channels=1, base_channels=32)
            model = MSCREDModel(input_channels=1, base_channels=32)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            logger.warning(f"MSCREDModel 로딩 실패, 재시도: {e}")
            # state_dict 키를 확인하여 파라미터 추정
            try:
                # state dict에서 첫 Conv 레이어의 in_channels 추출 시도
                first_conv_key = [k for k in state_dict.keys() if 'encoder' in k and 'conv' in k and 'weight' in k][0]
                in_channels = state_dict[first_conv_key].shape[1]
                model = MSCREDModel(input_channels=in_channels, base_channels=32)
                model.load_state_dict(state_dict)
                model.eval()
            except:
                # 그래도 실패하면 state 자체가 모델일 수 있음
                if hasattr(state, 'eval'):
                    model = state
                    model.eval()
                else:
                    raise RuntimeError(f"MSCREDModel 로딩 실패: {e}")
        
        # 피처 차원 자동 감지 (모델의 첫 번째 레이어에서)
        if feature_dim is None:
            try:
                # 일반적인 MS-CRED 구조에서 첫 번째 Conv 레이어 찾기
                for module in model.modules():
                    if hasattr(module, 'in_channels'):
                        feature_dim = module.in_channels
                        break
                
                if feature_dim is None:
                    feature_dim = 100  # 기본값
                    logger.warning(f"피처 차원 자동 감지 실패, 기본값 사용: {feature_dim}")
            except Exception as e:
                feature_dim = 100
                logger.warning(f"피처 차원 감지 오류: {e}, 기본값 사용: {feature_dim}")
        
        # 더미 입력 생성 (배치 크기 1, 윈도우 크기, 피처 차원)
        dummy_input = torch.randn(1, window_size, feature_dim)
        
        # ONNX 변환
        onnx_path = self.output_dir / output_name
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=['reconstructed'],
            dynamic_axes={
                'input_features': {0: 'batch_size'},
                'reconstructed': {0: 'batch_size'}
            }
        )
        
        # 메타데이터 저장
        metadata = {
            'model_type': 'mscred',
            'window_size': window_size,
            'feature_dim': feature_dim,
            'input_shape': [1, window_size, feature_dim],
            'output_shape': [1, window_size, feature_dim],
            'input_names': ['input_features'],
            'output_names': ['reconstructed'],
            'opset_version': 11
        }
        
        metadata_path = self.output_dir / f"{output_name}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ MS-CRED 변환 완료: {onnx_path}")
        logger.info(f"📊 메타데이터: {metadata_path}")
        
        return {
            'onnx_path': str(onnx_path),
            'metadata_path': str(metadata_path),
            'metadata': metadata
        }
    
    def validate_onnx_model(self, onnx_path: str) -> bool:
        """
        ONNX 모델의 유효성 검증
        
        Args:
            onnx_path: ONNX 모델 파일 경로
            
        Returns:
            유효성 검증 결과
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # ONNX 모델 로드 및 검증
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            # ONNX Runtime으로 실행 가능성 확인
            session = ort.InferenceSession(onnx_path)
            
            logger.info(f"✅ ONNX 모델 검증 성공: {onnx_path}")
            logger.info(f"📊 입력: {[input.name for input in session.get_inputs()]}")
            logger.info(f"📊 출력: {[output.name for output in session.get_outputs()]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX 모델 검증 실패: {e}")
            return False
    
    def optimize_onnx_model(self, onnx_path: str) -> str:
        """
        ONNX 모델 최적화 (그래프 최적화, 상수 폴딩 등)
        
        Args:
            onnx_path: 원본 ONNX 모델 경로
            
        Returns:
            최적화된 모델 경로
        """
        try:
            import onnxruntime as ort
            
            # 최적화 설정
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 최적화된 모델 경로
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            
            # 세션 생성 (최적화 적용)
            session = ort.InferenceSession(onnx_path, sess_options)
            
            # 최적화된 그래프 저장 (ONNX Runtime 1.9+)
            try:
                session.save(optimized_path)
                logger.info(f"✅ ONNX 모델 최적화 완료: {optimized_path}")
                return optimized_path
            except AttributeError:
                logger.warning("ONNX Runtime 버전이 낮아 최적화 저장 불가, 원본 반환")
                return onnx_path
                
        except Exception as e:
            logger.error(f"❌ ONNX 모델 최적화 실패: {e}")
            return onnx_path


def convert_all_models(
    deeplog_model: str,
    mscred_model: str,
    vocab_path: str,
    output_dir: str = "models/onnx"
) -> Dict[str, Any]:
    """
    모든 모델을 일괄 변환
    
    Args:
        deeplog_model: DeepLog 모델 경로
        mscred_model: MS-CRED 모델 경로
        vocab_path: 어휘 사전 경로
        output_dir: 출력 디렉토리
        
    Returns:
        변환 결과 요약
    """
    converter = ModelConverter(output_dir)
    results = {}
    
    # DeepLog 변환
    if os.path.exists(deeplog_model):
        try:
            deeplog_result = converter.convert_deeplog_to_onnx(
                deeplog_model, vocab_path
            )
            
            # 검증 및 최적화
            if converter.validate_onnx_model(deeplog_result['onnx_path']):
                optimized_path = converter.optimize_onnx_model(deeplog_result['onnx_path'])
                deeplog_result['optimized_path'] = optimized_path
            
            results['deeplog'] = deeplog_result
            
        except Exception as e:
            logger.error(f"❌ DeepLog 변환 실패: {e}")
            results['deeplog'] = {'error': str(e)}
    
    # MS-CRED 변환
    if os.path.exists(mscred_model):
        try:
            mscred_result = converter.convert_mscred_to_onnx(mscred_model)
            
            # 검증 및 최적화
            if converter.validate_onnx_model(mscred_result['onnx_path']):
                optimized_path = converter.optimize_onnx_model(mscred_result['onnx_path'])
                mscred_result['optimized_path'] = optimized_path
            
            results['mscred'] = mscred_result
            
        except Exception as e:
            logger.error(f"❌ MS-CRED 변환 실패: {e}")
            results['mscred'] = {'error': str(e)}
    
    # 변환 요약 저장
    summary_path = Path(output_dir) / "conversion_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📋 변환 요약 저장: {summary_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch 모델을 ONNX로 변환")
    parser.add_argument("--deeplog-model", type=str, help="DeepLog 모델 경로")
    parser.add_argument("--mscred-model", type=str, help="MS-CRED 모델 경로")
    parser.add_argument("--vocab", type=str, help="어휘 사전 경로")
    parser.add_argument("--output-dir", type=str, default="models/onnx", help="출력 디렉토리")
    parser.add_argument("--validate", action="store_true", help="변환 후 검증 실행")
    
    args = parser.parse_args()
    
    if not args.deeplog_model and not args.mscred_model:
        print("❌ 최소 하나의 모델 경로를 지정해야 합니다.")
        parser.print_help()
        exit(1)
    
    # 변환 실행
    results = convert_all_models(
        args.deeplog_model or "",
        args.mscred_model or "",
        args.vocab or "",
        args.output_dir
    )
    
    # 결과 출력
    print("\n🎉 모델 변환 완료!")
    for model_name, result in results.items():
        if 'error' in result:
            print(f"❌ {model_name}: {result['error']}")
        else:
            print(f"✅ {model_name}: {result['onnx_path']}")
            if 'optimized_path' in result:
                print(f"⚡ 최적화됨: {result['optimized_path']}")
