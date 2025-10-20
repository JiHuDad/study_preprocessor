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

    def _convert_vocab_for_c_engine(self, vocab: Dict, vocab_path: str) -> Dict[str, str]:
        """
        vocab.json을 C 엔진용 형식으로 변환

        Python 학습용: {"template_id": index} 형식
        C 엔진용: {"template_id": "template_string"} 형식

        Args:
            vocab: 원본 vocab (인덱스 형식일 수 있음)
            vocab_path: vocab.json 파일 경로 (parsed.parquet 찾기 위해 사용)

        Returns:
            C 엔진용 vocab (템플릿 문자열 포함)
        """
        # 이미 템플릿 문자열 형식인지 확인
        sample_value = next(iter(vocab.values())) if vocab else None
        if sample_value and isinstance(sample_value, str) and len(sample_value) > 10:
            # 이미 올바른 형식
            logger.info("📋 vocab이 이미 템플릿 문자열 형식입니다")
            return vocab

        # 인덱스 형식이므로 변환 필요
        logger.info("🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...")

        # vocab.json과 같은 디렉토리에서 parsed.parquet 또는 preview.json 찾기
        vocab_dir = Path(vocab_path).parent

        # Option 1: parsed.parquet에서 추출
        parsed_path = vocab_dir / "parsed.parquet"
        if parsed_path.exists():
            try:
                import pandas as pd
                logger.info(f"📂 parsed.parquet에서 템플릿 추출: {parsed_path}")
                df = pd.read_parquet(parsed_path)

                if 'template_id' in df.columns and 'template' in df.columns:
                    # CRITICAL: Sort by template_id to match training vocab order!
                    # Training vocab is created with: {t: i for i, t in enumerate(sorted(unique_templates))}
                    df_unique = df[['template_id', 'template']].drop_duplicates('template_id').copy()
                    df_unique['template_id_int'] = df_unique['template_id'].astype(str).astype(int)
                    df_unique = df_unique.sort_values('template_id_int')

                    template_map = {}
                    for _, row in df_unique.iterrows():
                        tid = str(row['template_id'])
                        template_str = str(row['template'])
                        if not pd.isna(tid) and not pd.isna(template_str):
                            template_map[tid] = template_str

                    if template_map:
                        logger.info(f"✅ {len(template_map)}개 템플릿 추출 완료 (정렬된 순서)")
                        return template_map
            except Exception as e:
                logger.warning(f"parsed.parquet 처리 실패: {e}")

        # Option 2: preview.json에서 추출
        preview_path = vocab_dir / "preview.json"
        if preview_path.exists():
            try:
                logger.info(f"📂 preview.json에서 템플릿 추출: {preview_path}")
                with open(preview_path, 'r') as f:
                    preview = json.load(f)

                template_map = {}
                for item in preview:
                    tid = str(item.get('template_id', ''))
                    template = item.get('template', '')
                    if tid and template:
                        template_map[tid] = template

                if template_map:
                    logger.info(f"✅ {len(template_map)}개 템플릿 추출 완료")
                    return template_map
            except Exception as e:
                logger.warning(f"preview.json 처리 실패: {e}")

        # 변환 실패 - 원본 vocab 반환하고 경고
        logger.warning("⚠️  템플릿 문자열을 추출할 수 없습니다.")
        logger.warning(f"⚠️  {vocab_dir}에 parsed.parquet 또는 preview.json이 필요합니다.")
        logger.warning("⚠️  C 엔진 사용을 위해 다음 스크립트를 실행하세요:")
        logger.warning(f"    python hybrid_system/training/export_vocab_with_templates.py \\")
        logger.warning(f"        {vocab_dir}/parsed.parquet \\")
        logger.warning(f"        {self.output_dir}/vocab.json")
        return vocab
        
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
        # anomaly_log_detector 패키지 경로 추가
        root_dir = Path(__file__).parent.parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        from anomaly_log_detector.builders.deeplog import DeepLogLSTM

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

        # PyTorch 버전에 따라 적절한 ONNX export 방식 선택
        import warnings
        pytorch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])

        # PyTorch 2.9+ 또는 dynamo 지원 버전에서는 새로운 방식 사용 시도
        export_success = False

        if pytorch_version >= (2, 4):  # dynamo 기반 export는 2.4+에서 사용 가능
            try:
                logger.info("🔄 PyTorch dynamo 기반 ONNX export 시도...")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)

                    torch.onnx.export(
                        model,
                        dummy_input,
                        str(onnx_path),
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input_sequence'],
                        output_names=['predictions'],
                        dynamic_axes={
                            'input_sequence': {0: 'batch_size', 1: 'sequence_length'},
                            'predictions': {0: 'batch_size'}
                        },
                        dynamo=True,  # 새로운 export 방식
                        verbose=False
                    )
                export_success = True
                logger.info("✅ dynamo 기반 export 성공")
            except Exception as e:
                logger.warning(f"⚠️  dynamo export 실패: {e}")
                logger.info("🔄 레거시 TorchScript 방식으로 재시도...")

        # 레거시 방식 또는 dynamo 실패 시
        if not export_success:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)

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
                        'input_sequence': {0: 'batch_size', 1: 'sequence_length'},
                        'predictions': {0: 'batch_size'}
                    },
                    verbose=False
                )
            logger.info("✅ 레거시 TorchScript 방식 export 성공")
        
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
        
        # 어휘 사전 처리
        vocab_output = self.output_dir / "vocab.json"

        # vocab.json 형식 확인 및 변환
        vocab_for_c_engine = self._convert_vocab_for_c_engine(vocab, vocab_path)

        with open(vocab_output, 'w') as f:
            json.dump(vocab_for_c_engine, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ DeepLog 변환 완료: {onnx_path}")
        logger.info(f"📊 메타데이터: {metadata_path}")
        logger.info(f"📚 어휘 사전: {vocab_output}")

        # vocab 형식 확인 메시지
        sample_value = next(iter(vocab_for_c_engine.values())) if vocab_for_c_engine else None
        if sample_value and isinstance(sample_value, str) and len(sample_value) > 10:
            logger.info(f"✅ C 엔진용 vocab 형식 (template strings): {len(vocab_for_c_engine)} templates")
        else:
            logger.warning(f"⚠️  vocab이 인덱스 형식입니다. C 엔진 사용 시 템플릿 문자열이 필요합니다.")
        
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
        # anomaly_log_detector 패키지 경로 추가
        root_dir = Path(__file__).parent.parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        from anomaly_log_detector.mscred_model import MSCREDModel

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
        
        # 피처 차원 결정
        # MS-CRED의 feature_dim은 템플릿 개수 (width)를 의미
        if feature_dim is None:
            # saved state에서 확인
            if saved_feature_dim and saved_feature_dim > 1:
                feature_dim = saved_feature_dim
            else:
                # 기본값: 일반적인 로그 템플릿 개수
                # 너무 작으면 conv 계산이 실패하므로 충분히 큰 값 사용
                feature_dim = 100
                logger.warning(f"피처 차원을 명시하지 않음. 기본값 사용: {feature_dim}")
                logger.warning(f"실제 사용한 템플릿 개수와 맞지 않으면 --feature-dim 옵션으로 지정하세요.")

        # 최소 크기 검증 (conv 레이어를 통과하기 위한 최소 크기)
        # MultiScaleConvBlock의 최대 kernel_size=7, padding=3이므로
        # 최소 feature_dim >= 7 필요
        if feature_dim < 10:
            logger.warning(f"feature_dim({feature_dim})이 너무 작습니다. 최소 10으로 설정합니다.")
            feature_dim = 10

        # 더미 입력 생성
        # MSCREDModel은 4D 텐서 (batch, channels, height, width) 기대
        # - batch: 1
        # - channels: 1 (input_channels)
        # - height: window_size (시간 축)
        # - width: feature_dim (템플릿 수)
        logger.info(f"MSCRED 더미 입력 크기: (1, 1, {window_size}, {feature_dim})")
        dummy_input = torch.randn(1, 1, window_size, feature_dim)
        
        # ONNX 변환
        onnx_path = self.output_dir / output_name

        # 불필요한 경고 억제
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

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
                },
                verbose=False
            )
        
        # 메타데이터 저장
        metadata = {
            'model_type': 'mscred',
            'window_size': window_size,
            'feature_dim': feature_dim,
            'input_shape': [1, 1, window_size, feature_dim],  # (batch, channels, height, width)
            'output_shape': [1, 1, window_size, feature_dim],
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
    
    def optimize_onnx_model(self, onnx_path: str, portable: bool = False) -> str:
        """
        ONNX 모델 최적화 (그래프 최적화, 상수 폴딩 등)

        Args:
            onnx_path: 원본 ONNX 모델 경로
            portable: True이면 범용 최적화만 적용 (하드웨어 특화 최적화 제외)
                     False이면 최대 최적화 적용 (현재 하드웨어에 특화)

        Returns:
            최적화된 모델 경로
        """
        try:
            import onnxruntime as ort

            # 최적화 설정
            sess_options = ort.SessionOptions()

            if portable:
                # 범용 최적화: 모든 환경에서 사용 가능
                # ORT_ENABLE_BASIC: 기본 그래프 최적화만 적용
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                suffix = '_portable'
                logger.info("🌍 범용 최적화 모드 (모든 환경에서 사용 가능)")
            else:
                # 최대 최적화: 현재 하드웨어에 특화
                # ORT_ENABLE_ALL: 하드웨어별 최적화 포함
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                suffix = '_optimized'
                logger.info("⚡ 최대 최적화 모드 (현재 환경에 특화)")

            sess_options.optimized_model_filepath = onnx_path.replace('.onnx', f'{suffix}.onnx')

            # 세션 생성 시 자동으로 최적화된 모델 저장
            session = ort.InferenceSession(onnx_path, sess_options)

            optimized_path = sess_options.optimized_model_filepath

            # 최적화 파일이 생성되었는지 확인
            import os
            if os.path.exists(optimized_path):
                logger.info(f"✅ ONNX 모델 최적화 완료: {optimized_path}")
                return optimized_path
            else:
                # 최적화가 적용되었지만 파일로 저장되지 않음 (메모리 내 최적화)
                logger.info(f"⚡ ONNX 모델 최적화 적용됨 (메모리 내): {onnx_path}")
                return onnx_path

        except Exception as e:
            logger.error(f"❌ ONNX 모델 최적화 실패: {e}")
            return onnx_path


def convert_all_models(
    deeplog_model: str,
    mscred_model: str,
    vocab_path: str,
    output_dir: str = "models/onnx",
    feature_dim: Optional[int] = None,
    portable: bool = False
) -> Dict[str, Any]:
    """
    모든 모델을 일괄 변환

    Args:
        deeplog_model: DeepLog 모델 경로
        mscred_model: MS-CRED 모델 경로
        vocab_path: 어휘 사전 경로
        output_dir: 출력 디렉토리
        feature_dim: MS-CRED 피처 차원 (템플릿 개수, None이면 자동 감지)
        portable: True이면 범용 최적화 (모든 환경), False이면 최대 최적화 (현재 하드웨어)

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
                optimized_path = converter.optimize_onnx_model(
                    deeplog_result['onnx_path'],
                    portable=portable
                )
                deeplog_result['optimized_path'] = optimized_path

            results['deeplog'] = deeplog_result

        except Exception as e:
            logger.error(f"❌ DeepLog 변환 실패: {e}")
            results['deeplog'] = {'error': str(e)}

    # MS-CRED 변환
    if os.path.exists(mscred_model):
        try:
            mscred_result = converter.convert_mscred_to_onnx(
                mscred_model,
                feature_dim=feature_dim
            )

            # 검증 및 최적화
            if converter.validate_onnx_model(mscred_result['onnx_path']):
                optimized_path = converter.optimize_onnx_model(
                    mscred_result['onnx_path'],
                    portable=portable
                )
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
