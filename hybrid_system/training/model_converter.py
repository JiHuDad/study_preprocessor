#!/usr/bin/env python3  # 실행 스크립트 셔뱅
"""PyTorch 모델을 ONNX 포맷으로 변환하는 도구 요약

- 목적: Python 학습 모델(PyTorch)을 C 추론 엔진에서 사용할 수 있도록 ONNX로 변환
- 변환 대상: DeepLog LSTM, MS-CRED 컨볼루션 모델
- 핵심 기능:
  * DeepLog/MS-CRED 모델을 ONNX로 export (dynamic_axes 지원)
  * vocab.json을 Python 형식에서 C 엔진용 형식으로 변환
  * ONNX 모델 검증 및 최적화 (범용/하드웨어 특화 옵션)
- 출력: ONNX 모델(.onnx), 메타데이터(.meta.json), C 엔진용 vocab.json
"""  # 모듈 요약 설명
import os  # 환경 변수/경로 유틸
import json  # JSON 입출력
import torch  # PyTorch 기본 모듈
import torch.onnx  # ONNX export API
import numpy as np  # (미사용) 수치 연산
from pathlib import Path  # 경로 처리
from typing import Dict, Any, Optional, Tuple  # 타입 힌트
import logging  # 로깅 프레임워크

# 로깅 설정  # INFO 레벨 기본 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # 모듈 로거 생성


class ModelConverter:  # 모델 변환기 클래스
    """PyTorch 모델을 ONNX로 변환하는 클래스"""  # 클래스 설명

    def __init__(self, output_dir: str = "models/onnx"):  # 초기화
        self.output_dir = Path(output_dir)  # 출력 경로
        self.output_dir.mkdir(parents=True, exist_ok=True)  # 폴더 생성

    def _convert_vocab_for_c_engine(self, vocab: Dict, vocab_path: str) -> Dict[str, str]:  # C 엔진용 vocab 변환
        """
        vocab.json을 C 엔진용 형식으로 변환

        Python 학습용: {"template_string": index} 형식 (sorted template string order)
        C 엔진용: {"index": "template_string"} 형식 (same sorted order)

        CRITICAL: Python vocab is created with:
            {t: i for i, t in enumerate(sorted(unique_templates))}
        So vocab indices are in SORTED TEMPLATE STRING order, NOT template_id order!

        Args:
            vocab: 원본 vocab ({"template_string": index} 형식)  # 입력 vocab
            vocab_path: vocab.json 파일 경로 (참고용)  # 참고 경로

        Returns:
            C 엔진용 vocab ({"index": "template_string"} 형식)  # 출력 vocab
        """  # API 설명
        # vocab 형식 확인  # 빈 vocab 처리
        if not vocab:
            logger.warning("⚠️  빈 vocab")  # 경고 로그
            return vocab  # 그대로 반환

        # 첫 번째 항목으로 형식 판단  # 샘플 항목으로 형식 추정
        first_key = next(iter(vocab.keys()))  # 첫 키 추출
        first_value = next(iter(vocab.values()))  # 첫 값 추출

        # Case 1: 이미 C 엔진용 형식 {"0": "template_string", ...}  # 형식 검사
        if isinstance(first_value, str) and not first_key.isdigit():  # 값은 문자열, 키는 숫자 아님
            # 잘못된 형식 경고 (key가 숫자가 아닌 경우)
            logger.warning(f"⚠️  vocab 형식이 이상합니다: key='{first_key}', value='{first_value}'")
            logger.warning("⚠️  예상 형식: {{\"template_string\": index}} 또는 {{\"index\": \"template_string\"}}")

        if isinstance(first_value, str) and first_key.isdigit():  # C 엔진용 형식 확인
            # 이미 C 엔진용 형식 {"0": "template string"}
            logger.info("📋 vocab이 이미 C 엔진용 템플릿 문자열 형식입니다")
            return vocab  # 변환 불필요

        # Case 2: Python 학습용 형식인지 확인  # Python 형식 체크
        # 올바른 형식: {"template_string": 0, ...}
        # 잘못된 형식: {"1": 0, "2": 1, ...} (template_id를 key로 사용)
        if isinstance(first_value, int):  # 값이 정수면 Python 형식 가능
            # 실제 템플릿 문자열인지 확인  # template_id 오용 방지
            # template_id는 보통 짧은 숫자 문자열이므로 길이로 구분
            if first_key.isdigit() and len(first_key) <= 5:  # template_id 오용 감지
                logger.error("❌ vocab이 template_id를 key로 사용하고 있습니다!")
                logger.error(f"   현재 형식: {{\"{first_key}\": {first_value}, ...}}")
                logger.error("   올바른 형식: {{\"actual template string\": 0, ...}}")
                logger.error("   해결: build_deeplog_inputs()에서 template_col='template' 사용")
                raise ValueError(
                    "vocab.json이 template_id를 사용합니다. "
                    "build_deeplog_inputs(template_col='template')로 재생성하세요."
                )

        # Python vocab 형식 확인: {template_string: index}  # 변환 수행
        # 변환: {index: template_string}
        logger.info("🔄 vocab을 C 엔진용 템플릿 문자열 형식으로 변환 중...")

        # CRITICAL: Python vocab의 인덱스 순서를 그대로 유지!  # 순서 보존 중요
        # Python: {template: idx} where idx는 sorted(template) 순서
        # C: {str(idx): template} 동일한 순서
        template_map = {}  # 결과 딕셔너리
        for template_str, vocab_idx in vocab.items():  # 각 항목 변환
            template_map[str(vocab_idx)] = template_str  # 인덱스: 템플릿 문자열

        if template_map:  # 변환 성공
            logger.info(f"✅ {len(template_map)}개 템플릿 변환 완료")
            logger.info(f"📊 Python vocab 순서 유지 (sorted template string order)")

            # 검증: 인덱스가 연속적인지 확인  # 인덱스 연속성 체크
            indices = sorted([int(k) for k in template_map.keys()])
            expected_indices = list(range(len(indices)))  # 기대 인덱스
            if indices != expected_indices:  # 불연속 감지
                logger.warning(f"⚠️  vocab 인덱스가 연속적이지 않습니다!")
                logger.warning(f"   기대: {expected_indices[:5]}...")
                logger.warning(f"   실제: {indices[:5]}...")

            return template_map  # 변환 결과 반환

        # 변환 실패  # 변환 실패 처리
        logger.warning("⚠️  vocab 변환 실패")
        return vocab  # 원본 반환
        
    def convert_deeplog_to_onnx(
        self,
        model_path: str,
        vocab_path: str,
        output_name: str = "deeplog.onnx",
        seq_len: Optional[int] = None
    ) -> Dict[str, Any]:  # DeepLog를 ONNX로 변환
        """
        DeepLog 모델을 ONNX로 변환

        Args:
            model_path: PyTorch 모델 파일 경로  # 입력 모델 경로
            vocab_path: 어휘 사전 파일 경로  # vocab.json 경로
            output_name: 출력 ONNX 파일명  # 출력 파일명
            seq_len: 시퀀스 길이 (None이면 모델에 저장된 값 사용,  # 동적 길이 지원
                    ONNX는 dynamic_axes로 다양한 길이 지원)

        Returns:
            변환 결과 정보  # ONNX 경로, 메타데이터, vocab 경로
        """  # API 설명
        logger.info(f"🔄 DeepLog 모델 변환 시작: {model_path}")  # 시작 로그

        # 어휘 사전 로드  # vocab 파일 읽기
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)  # 어휘 크기

        # PyTorch 모델 로드  # DeepLog 클래스 import
        # DeepLog 모델 클래스 import
        import sys
        from pathlib import Path
        # anomaly_log_detector 패키지 경로 추가
        root_dir = Path(__file__).parent.parent.parent  # 프로젝트 루트
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))  # 경로 추가

        from anomaly_log_detector.builders.deeplog import DeepLogLSTM  # LSTM 클래스 임포트

        # state dict 로드  # 체크포인트 로드
        state = torch.load(model_path, map_location='cpu')  # CPU에서 로드
        model_vocab_size = int(state.get("vocab_size", vocab_size))  # 모델 어휘 크기
        model_seq_len = int(state.get("seq_len", 50))  # 기본값 50  # 학습 시퀀스 길이
        model_embed_dim = int(state.get("embed_dim", 64))  # 기본값 64  # 임베딩 차원
        model_hidden_dim = int(state.get("hidden_dim", 128))  # 기본값 128  # 은닉 차원

        # seq_len 결정: 파라미터로 지정되지 않으면 모델에 저장된 값 사용  # 시퀀스 길이 결정
        if seq_len is None:
            seq_len = model_seq_len  # 모델 저장값 사용

        # 모델 생성 및 가중치 로드  # 인스턴스 생성 및 가중치 복원
        logger.info(f"📊 모델 파라미터: vocab_size={model_vocab_size}, embed_dim={model_embed_dim}, hidden_dim={model_hidden_dim}, seq_len={seq_len}")
        model = DeepLogLSTM(vocab_size=model_vocab_size, embed_dim=model_embed_dim, hidden_dim=model_hidden_dim)  # 모델 생성
        model.load_state_dict(state["state_dict"])  # 가중치 로드
        model.eval()  # 평가 모드 설정
        
        # 더미 입력 생성 (배치 크기 1, 시퀀스 길이)  # ONNX export용 샘플 입력
        dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)  # 랜덤 정수 시퀀스

        # ONNX 변환  # ONNX 파일 경로 설정
        onnx_path = self.output_dir / output_name  # 출력 경로

        # ONNX export (안정적인 레거시 방식 사용)  # TorchScript 방식 사용
        # Note: dynamo 기반 export는 일부 모델에서 오류 발생하므로 레거시 방식 사용
        import warnings

        logger.info("🔄 ONNX export 시작 (TorchScript 방식)...")  # export 시작 로그
        with warnings.catch_warnings():  # 경고 억제
            warnings.filterwarnings("ignore", category=UserWarning)  # 사용자 경고 무시
            warnings.filterwarnings("ignore", category=DeprecationWarning)  # 폐지 경고 무시
            warnings.filterwarnings("ignore", category=FutureWarning)  # 미래 경고 무시

            # PyTorch 2.0+ 호환성: dynamo 방식 명시적 비활성화  # export 옵션 설정
            export_options = {
                'export_params': True,  # 파라미터 포함
                'opset_version': 11,  # 호환성을 위해 안정적인 버전 사용  # ONNX opset 버전
                'do_constant_folding': True,  # 상수 폴딩 활성화
                'input_names': ['input_sequence'],  # 입력 이름
                'output_names': ['predictions'],  # 출력 이름
                'dynamic_axes': {
                    'input_sequence': {0: 'batch_size', 1: 'sequence_length'},  # 동적 배치/시퀀스
                    'predictions': {0: 'batch_size'}  # 동적 배치
                },
                'verbose': False  # 상세 출력 비활성화
            }

            # PyTorch 2.1+에서 dynamo 방식 강제 비활성화  # 버전 호환성 처리
            try:
                # dynamo=False를 시도 (PyTorch 2.1+)
                torch.onnx.export(
                    model,  # 변환할 모델
                    dummy_input,  # 더미 입력
                    str(onnx_path),  # 출력 경로
                    dynamo=False,  # 명시적으로 레거시 TorchScript 방식 사용  # dynamo 비활성화
                    **export_options
                )
            except TypeError:  # PyTorch 2.0 이하
                # PyTorch 2.0 이하는 dynamo 파라미터 없음
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    **export_options
                )
        logger.info("✅ ONNX export 성공")  # 성공 로그
        
        # 메타데이터 저장  # 모델 정보 저장
        metadata = {
            'model_type': 'deeplog',  # 모델 타입
            'vocab_size': vocab_size,  # 어휘 크기
            'seq_len': seq_len,  # 학습 시 사용한 시퀀스 길이 (권장값)  # 권장 시퀀스 길이
            'input_shape': [1, seq_len],  # 예시 입력 형태  # 샘플 입력 형태
            'output_shape': [1, vocab_size],  # 출력 형태
            'input_names': ['input_sequence'],  # 입력 이름
            'output_names': ['predictions'],  # 출력 이름
            'opset_version': 11,  # opset 버전
            'dynamic_axes': {
                'input_sequence': {
                    '0': 'batch_size',  # 배치 동적
                    '1': 'sequence_length'  # 동적: 다양한 길이 지원  # 시퀀스 동적
                },
                'predictions': {
                    '0': 'batch_size'  # 배치 동적
                }
            },
            'notes': 'ONNX model supports dynamic sequence lengths via dynamic_axes. seq_len is recommended value from training.'  # 주석
        }
        
        metadata_path = self.output_dir / f"{output_name}.meta.json"  # 메타데이터 경로
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)  # JSON 저장
        
        # 어휘 사전 처리  # C 엔진용 vocab 변환
        vocab_output = self.output_dir / "vocab.json"  # vocab 출력 경로

        # vocab.json 형식 확인 및 변환  # 변환 실행
        vocab_for_c_engine = self._convert_vocab_for_c_engine(vocab, vocab_path)  # 변환 메서드 호출

        with open(vocab_output, 'w') as f:
            json.dump(vocab_for_c_engine, f, ensure_ascii=False, indent=2)  # C 엔진용 vocab 저장

        logger.info(f"✅ DeepLog 변환 완료: {onnx_path}")  # 완료 로그
        logger.info(f"📊 메타데이터: {metadata_path}")
        logger.info(f"📚 어휘 사전: {vocab_output}")

        # vocab 형식 확인 메시지  # 템플릿 문자열 형식 검증
        sample_value = next(iter(vocab_for_c_engine.values())) if vocab_for_c_engine else None
        if sample_value and isinstance(sample_value, str) and len(sample_value) > 10:  # 템플릿 문자열 확인
            logger.info(f"✅ C 엔진용 vocab 형식 (template strings): {len(vocab_for_c_engine)} templates")
        else:
            logger.warning(f"⚠️  vocab이 인덱스 형식입니다. C 엔진 사용 시 템플릿 문자열이 필요합니다.")
        
        return {
            'onnx_path': str(onnx_path),  # ONNX 경로
            'metadata_path': str(metadata_path),  # 메타데이터 경로
            'vocab_path': str(vocab_output),  # vocab 경로
            'metadata': metadata  # 메타데이터 딕셔너리
        }  # 결과 반환
    
    def convert_mscred_to_onnx(
        self,
        model_path: str,
        output_name: str = "mscred.onnx",
        window_size: int = 50,
        feature_dim: Optional[int] = None
    ) -> Dict[str, Any]:  # MS-CRED를 ONNX로 변환
        """
        MS-CRED 모델을 ONNX로 변환
        
        Args:
            model_path: PyTorch 모델 파일 경로  # 입력 모델 경로
            output_name: 출력 ONNX 파일명  # 출력 파일명
            window_size: 윈도우 크기  # 시간 윈도우 크기
            feature_dim: 피처 차원 (자동 감지 시도)  # 템플릿 개수
            
        Returns:
            변환 결과 정보  # ONNX 경로, 메타데이터
        """  # API 설명
        logger.info(f"🔄 MS-CRED 모델 변환 시작: {model_path}")  # 시작 로그

        # MS-CRED 모델 클래스 import  # 모듈 import
        import sys
        from pathlib import Path
        # anomaly_log_detector 패키지 경로 추가
        root_dir = Path(__file__).parent.parent.parent  # 프로젝트 루트
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))  # 경로 추가

        from anomaly_log_detector.mscred_model import MSCREDModel  # MS-CRED 클래스 임포트

        # state dict 로드  # 체크포인트 로드
        state = torch.load(model_path, map_location='cpu')  # CPU에서 로드

        # 모델 파라미터 추출  # 체크포인트에서 파라미터 추출
        # MS-CRED는 'model_state_dict' 키로 저장됨
        if isinstance(state, dict):  # 딕셔너리 형식 체크
            if 'model_state_dict' in state:  # MSCREDTrainer 형식
                # MSCREDTrainer.save_model() 형식
                state_dict = state['model_state_dict']  # 상태 딕셔너리 추출
                saved_feature_dim = state.get('feature_dim', feature_dim)  # 피처 차원
                saved_window_size = state.get('window_size', window_size)  # 윈도우 크기
            elif 'state_dict' in state:  # 다른 형식
                # 다른 저장 형식
                state_dict = state['state_dict']  # 상태 딕셔너리
                saved_feature_dim = state.get('feature_dim', feature_dim)  # 피처 차원
                saved_window_size = state.get('window_size', window_size)  # 윈도우 크기
            else:
                # state_dict만 있는 경우 (구버전)  # 구버전 처리
                state_dict = state  # 전체 state 사용
                saved_feature_dim = feature_dim  # 파라미터 사용
                saved_window_size = window_size
        else:
            state_dict = state  # state 자체가 딕셔너리
            saved_feature_dim = feature_dim
            saved_window_size = window_size

        # 모델 생성  # 인스턴스 생성
        # MSCREDModel의 파라미터: input_channels, base_channels
        # feature_dim은 ONNX export시 입력 크기 결정에만 사용
        try:
            # 기본값으로 모델 생성 (input_channels=1, base_channels=32)  # 기본 파라미터
            model = MSCREDModel(input_channels=1, base_channels=32)  # 모델 생성
            model.load_state_dict(state_dict)  # 가중치 로드
            model.eval()  # 평가 모드
        except Exception as e:
            logger.warning(f"MSCREDModel 로딩 실패, 재시도: {e}")  # 재시도 로그
            # state_dict 키를 확인하여 파라미터 추정  # 파라미터 자동 추정
            try:
                # state dict에서 첫 Conv 레이어의 in_channels 추출 시도  # 채널 수 추정
                first_conv_key = [k for k in state_dict.keys() if 'encoder' in k and 'conv' in k and 'weight' in k][0]  # 첫 Conv 레이어 찾기
                in_channels = state_dict[first_conv_key].shape[1]  # 입력 채널 수 추출
                model = MSCREDModel(input_channels=in_channels, base_channels=32)  # 추정 파라미터로 생성
                model.load_state_dict(state_dict)  # 가중치 로드
                model.eval()  # 평가 모드
            except:
                # 그래도 실패하면 state 자체가 모델일 수 있음  # 모델 객체인지 확인
                if hasattr(state, 'eval'):  # eval 메서드 존재 확인
                    model = state  # state를 모델로 사용
                    model.eval()  # 평가 모드
                else:
                    raise RuntimeError(f"MSCREDModel 로딩 실패: {e}")  # 에러 발생
        
        # 피처 차원 결정  # 템플릿 개수 결정
        # MS-CRED의 feature_dim은 템플릿 개수 (width)를 의미
        if feature_dim is None:  # 파라미터로 지정 안됨
            # saved state에서 확인  # 저장된 값 확인
            if saved_feature_dim and saved_feature_dim > 1:  # 저장된 값 유효
                feature_dim = saved_feature_dim  # 저장된 값 사용
            else:
                # 기본값: 일반적인 로그 템플릿 개수  # 기본값 사용
                # 너무 작으면 conv 계산이 실패하므로 충분히 큰 값 사용
                feature_dim = 100  # 기본 템플릿 개수
                logger.warning(f"피처 차원을 명시하지 않음. 기본값 사용: {feature_dim}")
                logger.warning(f"실제 사용한 템플릿 개수와 맞지 않으면 --feature-dim 옵션으로 지정하세요.")

        # 최소 크기 검증 (conv 레이어를 통과하기 위한 최소 크기)  # 최소 크기 체크
        # MultiScaleConvBlock의 최대 kernel_size=7, padding=3이므로
        # 최소 feature_dim >= 7 필요
        if feature_dim < 10:  # 최소 크기 미달
            logger.warning(f"feature_dim({feature_dim})이 너무 작습니다. 최소 10으로 설정합니다.")
            feature_dim = 10  # 최소값으로 설정

        # 더미 입력 생성  # ONNX export용 샘플 입력
        # MSCREDModel은 4D 텐서 (batch, channels, height, width) 기대
        # - batch: 1
        # - channels: 1 (input_channels)
        # - height: window_size (시간 축)
        # - width: feature_dim (템플릿 수)
        logger.info(f"MSCRED 더미 입력 크기: (1, 1, {window_size}, {feature_dim})")
        dummy_input = torch.randn(1, 1, window_size, feature_dim)  # 랜덤 4D 텐서
        
        # ONNX 변환  # ONNX 파일 경로
        onnx_path = self.output_dir / output_name  # 출력 경로

        # 불필요한 경고 억제  # 경고 필터링
        import warnings
        with warnings.catch_warnings():  # 경고 컨텍스트
            warnings.filterwarnings("ignore", category=UserWarning)  # 사용자 경고 무시
            warnings.filterwarnings("ignore", category=DeprecationWarning)  # 폐지 경고 무시
            warnings.filterwarnings("ignore", category=FutureWarning)  # 미래 경고 무시

            # PyTorch 2.0+ 호환성: dynamo 방식 명시적 비활성화  # export 옵션 설정
            export_options = {
                'export_params': True,  # 파라미터 포함
                'opset_version': 11,  # opset 버전
                'do_constant_folding': True,  # 상수 폴딩
                'input_names': ['input_features'],  # 입력 이름
                'output_names': ['reconstructed'],  # 출력 이름
                'dynamic_axes': {
                    'input_features': {0: 'batch_size'},  # 동적 배치
                    'reconstructed': {0: 'batch_size'}  # 동적 배치
                },
                'verbose': False  # 상세 출력 비활성화
            }

            # PyTorch 2.1+에서 dynamo 방식 강제 비활성화  # 버전 호환성 처리
            try:
                torch.onnx.export(
                    model,  # 변환할 모델
                    dummy_input,  # 더미 입력
                    str(onnx_path),  # 출력 경로
                    dynamo=False,  # 명시적으로 레거시 TorchScript 방식 사용  # dynamo 비활성화
                    **export_options
                )
            except TypeError:  # PyTorch 2.0 이하
                # PyTorch 2.0 이하는 dynamo 파라미터 없음
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    **export_options
                )
        
        # 메타데이터 저장  # 모델 정보 저장
        metadata = {
            'model_type': 'mscred',  # 모델 타입
            'window_size': window_size,  # 윈도우 크기
            'feature_dim': feature_dim,  # 피처 차원
            'input_shape': [1, 1, window_size, feature_dim],  # (batch, channels, height, width)  # 입력 형태
            'output_shape': [1, 1, window_size, feature_dim],  # 출력 형태
            'input_names': ['input_features'],  # 입력 이름
            'output_names': ['reconstructed'],  # 출력 이름
            'opset_version': 11  # opset 버전
        }
        
        metadata_path = self.output_dir / f"{output_name}.meta.json"  # 메타데이터 경로
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)  # JSON 저장
        
        logger.info(f"✅ MS-CRED 변환 완료: {onnx_path}")  # 완료 로그
        logger.info(f"📊 메타데이터: {metadata_path}")
        
        return {
            'onnx_path': str(onnx_path),  # ONNX 경로
            'metadata_path': str(metadata_path),  # 메타데이터 경로
            'metadata': metadata  # 메타데이터 딕셔너리
        }  # 결과 반환
    
    def validate_onnx_model(self, onnx_path: str) -> bool:  # ONNX 모델 검증
        """
        ONNX 모델의 유효성 검증
        
        Args:
            onnx_path: ONNX 모델 파일 경로  # 입력 경로
            
        Returns:
            유효성 검증 결과  # 검증 성공 여부
        """  # API 설명
        try:
            import onnx  # ONNX 라이브러리
            import onnxruntime as ort  # ONNX Runtime
            
            # ONNX 모델 로드 및 검증  # 모델 로드
            model = onnx.load(onnx_path)  # 모델 파일 로드
            onnx.checker.check_model(model)  # 모델 유효성 검사
            
            # ONNX Runtime으로 실행 가능성 확인  # 실행 가능성 체크
            session = ort.InferenceSession(onnx_path)  # 추론 세션 생성
            
            logger.info(f"✅ ONNX 모델 검증 성공: {onnx_path}")  # 성공 로그
            logger.info(f"📊 입력: {[input.name for input in session.get_inputs()]}")  # 입력 정보
            logger.info(f"📊 출력: {[output.name for output in session.get_outputs()]}")  # 출력 정보
            
            return True  # 검증 성공
            
        except Exception as e:
            logger.error(f"❌ ONNX 모델 검증 실패: {e}")  # 실패 로그
            return False  # 검증 실패
    
    def optimize_onnx_model(self, onnx_path: str, portable: bool = False) -> str:  # ONNX 모델 최적화
        """
        ONNX 모델 최적화 (그래프 최적화, 상수 폴딩 등)

        Args:
            onnx_path: 원본 ONNX 모델 경로  # 입력 경로
            portable: True이면 범용 최적화만 적용 (하드웨어 특화 최적화 제외)  # 범용/특화 선택
                     False이면 최대 최적화 적용 (현재 하드웨어에 특화)

        Returns:
            최적화된 모델 경로  # 최적화된 파일 경로
        """  # API 설명
        try:
            import onnxruntime as ort  # ONNX Runtime

            # 최적화 설정  # 세션 옵션 설정
            sess_options = ort.SessionOptions()  # 세션 옵션 생성

            if portable:  # 범용 모드
                # 범용 최적화: 모든 환경에서 사용 가능  # 범용 최적화 적용
                # ORT_ENABLE_BASIC: 기본 그래프 최적화만 적용
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # 기본 최적화
                suffix = '_portable'  # 파일명 접미사
                logger.info("🌍 범용 최적화 모드 (모든 환경에서 사용 가능)")
            else:  # 최대 최적화 모드
                # 최대 최적화: 현재 하드웨어에 특화  # 하드웨어 특화 최적화
                # ORT_ENABLE_ALL: 하드웨어별 최적화 포함
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 최대 최적화
                suffix = '_optimized'  # 파일명 접미사
                logger.info("⚡ 최대 최적화 모드 (현재 환경에 특화)")

            sess_options.optimized_model_filepath = onnx_path.replace('.onnx', f'{suffix}.onnx')  # 최적화 파일 경로

            # 세션 생성 시 자동으로 최적화된 모델 저장  # 최적화 실행
            session = ort.InferenceSession(onnx_path, sess_options)  # 세션 생성 및 최적화

            optimized_path = sess_options.optimized_model_filepath  # 최적화 파일 경로

            # 최적화 파일이 생성되었는지 확인  # 파일 존재 확인
            import os
            if os.path.exists(optimized_path):  # 파일 존재
                logger.info(f"✅ ONNX 모델 최적화 완료: {optimized_path}")
                return optimized_path  # 최적화 파일 경로 반환
            else:
                # 최적화가 적용되었지만 파일로 저장되지 않음 (메모리 내 최적화)  # 메모리 최적화
                logger.info(f"⚡ ONNX 모델 최적화 적용됨 (메모리 내): {onnx_path}")
                return onnx_path  # 원본 경로 반환

        except Exception as e:
            logger.error(f"❌ ONNX 모델 최적화 실패: {e}")  # 실패 로그
            return onnx_path  # 원본 경로 반환


def convert_all_models(
    deeplog_model: str,
    mscred_model: str,
    vocab_path: str,
    output_dir: str = "models/onnx",
    seq_len: Optional[int] = None,
    feature_dim: Optional[int] = None,
    portable: bool = False
) -> Dict[str, Any]:  # 모든 모델 일괄 변환
    """
    모든 모델을 일괄 변환

    Args:
        deeplog_model: DeepLog 모델 경로  # 입력 모델 경로
        mscred_model: MS-CRED 모델 경로  # 입력 모델 경로
        vocab_path: 어휘 사전 경로  # vocab.json 경로
        output_dir: 출력 디렉토리  # 출력 폴더
        seq_len: DeepLog 시퀀스 길이 (None이면 모델 저장값 사용)  # 시퀀스 길이
        feature_dim: MS-CRED 피처 차원 (템플릿 개수, None이면 자동 감지)  # 템플릿 개수
        portable: True이면 범용 최적화 (모든 환경), False이면 최대 최적화 (현재 하드웨어)  # 최적화 모드

    Returns:
        변환 결과 요약  # 결과 딕셔너리
    """  # API 설명
    converter = ModelConverter(output_dir)  # 변환기 생성
    results = {}  # 결과 딕셔너리

    # DeepLog 변환  # DeepLog 모델 처리
    if os.path.exists(deeplog_model):  # 파일 존재 확인
        try:
            deeplog_result = converter.convert_deeplog_to_onnx(
                deeplog_model, vocab_path, seq_len=seq_len  # DeepLog 변환 호출
            )

            # 검증 및 최적화  # ONNX 검증 및 최적화
            if converter.validate_onnx_model(deeplog_result['onnx_path']):  # 검증 성공
                optimized_path = converter.optimize_onnx_model(
                    deeplog_result['onnx_path'],  # 최적화 실행
                    portable=portable
                )
                deeplog_result['optimized_path'] = optimized_path  # 최적화 경로 추가

            results['deeplog'] = deeplog_result  # 결과 저장

        except Exception as e:
            logger.error(f"❌ DeepLog 변환 실패: {e}")  # 에러 로그
            results['deeplog'] = {'error': str(e)}  # 에러 저장

    # MS-CRED 변환  # MS-CRED 모델 처리
    if os.path.exists(mscred_model):  # 파일 존재 확인
        try:
            mscred_result = converter.convert_mscred_to_onnx(
                mscred_model,  # MS-CRED 변환 호출
                feature_dim=feature_dim
            )

            # 검증 및 최적화  # ONNX 검증 및 최적화
            if converter.validate_onnx_model(mscred_result['onnx_path']):  # 검증 성공
                optimized_path = converter.optimize_onnx_model(
                    mscred_result['onnx_path'],  # 최적화 실행
                    portable=portable
                )
                mscred_result['optimized_path'] = optimized_path  # 최적화 경로 추가

            results['mscred'] = mscred_result  # 결과 저장

        except Exception as e:
            logger.error(f"❌ MS-CRED 변환 실패: {e}")  # 에러 로그
            results['mscred'] = {'error': str(e)}  # 에러 저장
    
    # 변환 요약 저장  # 결과 요약 저장
    summary_path = Path(output_dir) / "conversion_summary.json"  # 요약 파일 경로
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # JSON 저장
    
    logger.info(f"📋 변환 요약 저장: {summary_path}")  # 저장 로그
    
    return results  # 결과 반환


if __name__ == "__main__":  # 스크립트 직접 실행
    import argparse  # 인자 파싱
    
    parser = argparse.ArgumentParser(description="PyTorch 모델을 ONNX로 변환")  # 파서 생성
    parser.add_argument("--deeplog-model", type=str, help="DeepLog 모델 경로")  # DeepLog 인자
    parser.add_argument("--mscred-model", type=str, help="MS-CRED 모델 경로")  # MS-CRED 인자
    parser.add_argument("--vocab", type=str, help="어휘 사전 경로")  # vocab 인자
    parser.add_argument("--output-dir", type=str, default="models/onnx", help="출력 디렉토리")  # 출력 인자
    parser.add_argument("--validate", action="store_true", help="변환 후 검증 실행")  # 검증 인자
    
    args = parser.parse_args()  # 인자 파싱
    
    if not args.deeplog_model and not args.mscred_model:  # 모델 경로 없음
        print("❌ 최소 하나의 모델 경로를 지정해야 합니다.")  # 에러 메시지
        parser.print_help()  # 도움말 출력
        exit(1)  # 종료
    
    # 변환 실행  # 변환 호출
    results = convert_all_models(
        args.deeplog_model or "",  # DeepLog 경로
        args.mscred_model or "",  # MS-CRED 경로
        args.vocab or "",  # vocab 경로
        args.output_dir  # 출력 디렉토리
    )
    
    # 결과 출력  # 결과 표시
    print("\n🎉 모델 변환 완료!")  # 완료 메시지
    for model_name, result in results.items():  # 각 모델 결과
        if 'error' in result:  # 에러 있음
            print(f"❌ {model_name}: {result['error']}")  # 에러 출력
        else:  # 성공
            print(f"✅ {model_name}: {result['onnx_path']}")  # 경로 출력
            if 'optimized_path' in result:  # 최적화됨
                print(f"⚡ 최적화됨: {result['optimized_path']}")  # 최적화 경로 출력
