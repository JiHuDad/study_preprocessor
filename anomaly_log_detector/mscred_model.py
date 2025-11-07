"""  # 모듈 전체 설명 시작
MS-CRED (Multi-Scale Convolutional Recurrent Encoder-Decoder) 모델 구현  # MS-CRED 모델 구현을 설명

MS-CRED는 시계열 이상탐지를 위한 멀티스케일 컨볼루션 오토인코더입니다.  # 모델 개요
로그 템플릿 카운트 매트릭스를 입력으로 받아 재구성 오차를 통해 이상을 탐지합니다.  # 입력과 목적
"""  # 모듈 설명 끝

from __future__ import annotations  # 순환 참조 등 타입 힌트를 문자열 없이 사용하기 위함

import torch  # PyTorch 기본 모듈
import torch.nn as nn  # 신경망 레이어 정의용
import torch.nn.functional as F  # 함수형 API (손실, 활성함수 등)
from pathlib import Path  # 경로 처리를 위한 Path 클래스
import pandas as pd  # 데이터프레임 처리용
import numpy as np  # 수치 연산 라이브러리
from typing import Tuple, List, Dict, Optional  # 타입 힌트용
import json  # JSON 입출력 (현재 사용은 없지만 유지)
from tqdm import tqdm  # 진행률 표시 바


class MultiScaleConvBlock(nn.Module):  # 멀티스케일 컨볼루션 블록 정의
    """멀티스케일 컨볼루션 블록"""  # 클래스 설명
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [3, 5, 7]):  # 생성자
        super().__init__()  # 부모 클래스 초기화
        self.kernel_sizes = kernel_sizes  # 사용할 커널 크기 리스트 저장
        
        # 각 커널의 채널 수를 균등하게 분배하되 나머지는 첫 번째 커널에 할당  # 채널 분배 로직
        channels_per_kernel = out_channels // len(kernel_sizes)  # 커널당 기본 채널 수
        remainder = out_channels % len(kernel_sizes)  # 분배 후 남는 채널 수
        
        # 각 커널 크기별 컨볼루션 레이어  # 병렬 다중 커널
        self.convs = nn.ModuleList()  # 동적 레이어 리스트
        for i, k in enumerate(kernel_sizes):  # 각 커널 크기에 대해 반복
            # 첫 번째 커널에 나머지 채널 추가  # 채널 균형 유지
            current_channels = channels_per_kernel + (remainder if i == 0 else 0)  # 현재 커널의 출력 채널 수
            self.convs.append(
                nn.Conv2d(in_channels, current_channels, kernel_size=k, padding=k//2)  # 같은 크기 유지 패딩
            )  # 컨볼루션 레이어 추가
        
        self.bn = nn.BatchNorm2d(out_channels)  # 배치 정규화로 안정화
        self.relu = nn.ReLU(inplace=True)  # ReLU 활성함수 (inplace로 메모리 절약)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 순전파 정의
        # 각 커널 크기별로 컨볼루션 적용 후 채널 차원에서 결합  # 멀티스케일 특징 결합
        conv_outputs = [conv(x) for conv in self.convs]  # 각 컨볼루션의 출력 리스트
        out = torch.cat(conv_outputs, dim=1)  # 채널 방향으로 합치기
        out = self.bn(out)  # 배치 정규화 적용
        out = self.relu(out)  # ReLU 활성화
        return out  # 출력 반환


class AttentionModule(nn.Module):  # 어텐션 모듈 정의
    """어텐션 모듈"""  # 클래스 설명
    
    def __init__(self, channels: int):  # 생성자
        super().__init__()  # 부모 초기화
        self.conv_att = nn.Conv2d(channels, 1, kernel_size=1)  # 채널 축소를 통한 주의 가중치 생성
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 순전파
        # 채널별 어텐션 가중치 계산  # 중요 영역 강조
        att_weights = torch.sigmoid(self.conv_att(x))  # 0~1 범위의 가중치
        return x * att_weights  # 입력에 가중치 적용


class MSCREDEncoder(nn.Module):  # 인코더 부분 정의
    """MS-CRED 인코더"""  # 클래스 설명
    
    def __init__(self, input_channels: int = 1, base_channels: int = 32):  # 생성자
        super().__init__()  # 부모 초기화
        
        # 멀티스케일 컨볼루션 레이어들  # 점차 채널 수 증가
        self.conv1 = MultiScaleConvBlock(input_channels, base_channels)  # 1단계 특징 추출
        self.conv2 = MultiScaleConvBlock(base_channels, base_channels * 2)  # 2단계 특징 추출
        self.conv3 = MultiScaleConvBlock(base_channels * 2, base_channels * 4)  # 3단계 특징 추출
        
        # 어텐션 모듈들  # 각 스케일 출력에 주의 메커니즘 적용
        self.att1 = AttentionModule(base_channels)  # 1단계 어텐션
        self.att2 = AttentionModule(base_channels * 2)  # 2단계 어텐션
        self.att3 = AttentionModule(base_channels * 4)  # 3단계 어텐션
        
        # 다운샘플링  # 공간 크기를 절반으로 줄임
        self.pool = nn.MaxPool2d(2, 2)  # 커널 2, 스트라이드 2
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # 순전파, 다중 스케일 특징 반환
        # x: (batch, 1, height, width) - height: time_steps, width: num_templates  # 입력 형태 설명
        
        # 첫 번째 스케일  # 초기 특징 추출과 어텐션
        x1 = self.conv1(x)  # conv1 통과
        x1_att = self.att1(x1)  # 어텐션 적용
        
        # 두 번째 스케일  # 다운샘플링 후 2단계 특징
        x2 = self.pool(x1_att)  # 맥스풀링
        x2 = self.conv2(x2)  # conv2 통과
        x2_att = self.att2(x2)  # 어텐션 적용
        
        # 세 번째 스케일  # 재다운샘플링 후 3단계 특징
        x3 = self.pool(x2_att)  # 맥스풀링
        x3 = self.conv3(x3)  # conv3 통과
        x3_att = self.att3(x3)  # 어텐션 적용
        
        return [x1_att, x2_att, x3_att]  # 각 스케일의 어텐션 결과 반환


class MSCREDDecoder(nn.Module):  # 디코더 정의
    """MS-CRED 디코더 (간소화 버전)"""  # 클래스 설명
    
    def __init__(self, base_channels: int = 32, output_channels: int = 1):  # 생성자
        super().__init__()  # 부모 초기화
        
        # 간단한 업샘플링 디코더  # ConvTranspose2d로 해상도 복원
        self.deconv3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 
                                         kernel_size=4, stride=2, padding=1)  # 3->2 단계 업샘플링
        self.deconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 
                                         kernel_size=4, stride=2, padding=1)  # 2->1 단계 업샘플링
        self.deconv1 = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)  # 최종 채널 투영
        
        self.relu = nn.ReLU(inplace=True)  # 활성함수
        
    def forward(self, encoder_outputs: List[torch.Tensor]) -> torch.Tensor:  # 순전파
        # 가장 작은 피처맵(가장 깊은 인코딩)만 사용  # 간소화된 skip 연결 없음
        _, _, x3_att = encoder_outputs  # 세 번째 스케일만 사용
        
        # 업샘플링으로 원래 크기로 복원  # 단계적 복원
        x = self.relu(self.deconv3(x3_att))  # 첫 업샘플링
        x = self.relu(self.deconv2(x))  # 두 번째 업샘플링
        x = self.deconv1(x)  # 최종 출력 채널로 투영
        
        return x  # 재구성 결과 반환


class MSCREDModel(nn.Module):  # 인코더-디코더 결합 모델
    """MS-CRED 전체 모델"""  # 클래스 설명
    
    def __init__(self, input_channels: int = 1, base_channels: int = 32):  # 생성자
        super().__init__()  # 부모 초기화
        self.encoder = MSCREDEncoder(input_channels, base_channels)  # 인코더 생성
        self.decoder = MSCREDDecoder(base_channels, input_channels)  # 디코더 생성
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 순전파 정의
        # 입력 크기 저장  # ONNX 호환 보정용
        input_shape = x.shape  # 배치, 채널, 높이, 너비

        encoded = self.encoder(x)  # 인코더 통과하여 멀티스케일 특징 추출
        reconstructed = self.decoder(encoded)  # 디코더로 재구성 수행

        # 출력 크기를 입력 크기와 정확히 맞춤  # 후처리 크기 보정
        # ONNX 추적 호환성: shape 비교 대신 항상 interpolate 수행  # 안정적 추적
        # (입력과 출력이 같으면 interpolate는 no-op)  # 동일 크기 시 영향 없음
        reconstructed = F.interpolate(
            reconstructed,  # 보정 대상 텐서
            size=(input_shape[2], input_shape[3]),  # 높이, 너비를 입력과 동일하게
            mode='bilinear',  # 양선형 보간
            align_corners=False  # 좌표 정렬 비활성화로 일반적 설정
        )  # 보간 결과

        return reconstructed  # 최종 재구성 결과 반환
    
    def compute_reconstruction_error(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:  # 오차 계산 함수
        """재구성 오차 계산"""  # 함수 설명
        return F.mse_loss(reconstructed, x, reduction='none').mean(dim=[1, 2, 3])  # 배치별 평균 MSE 반환


class MSCREDTrainer:  # 학습 관리 클래스
    """MS-CRED 모델 트레이너"""  # 클래스 설명
    
    def __init__(self, model: MSCREDModel, device: str = 'cpu'):  # 생성자
        self.model = model.to(device)  # 모델 디바이스 이동
        self.device = device  # 디바이스 저장
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam 옵티마이저
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )  # 검증 손실 감소 정체 시 학습률 감소
        
    def prepare_data(self, window_counts_df: pd.DataFrame, target_num_templates: Optional[int] = None) -> Tuple[torch.Tensor, int]:  # 데이터 준비
        """윈도우 카운트 데이터를 텐서로 변환

        Args:
            window_counts_df: 윈도우 카운트 데이터프레임  # 입력 데이터프레임
            target_num_templates: 목표 템플릿 개수 (추론 시 학습 모델과 맞추기 위해)  # 템플릿 정합 옵션

        Returns:
            (텐서 데이터, 실제 템플릿 개수)  # 텐서와 템플릿 수 반환
        """  # 설명 끝
        # start_index 컬럼 제거하고 템플릿 카운트만 추출  # 특징만 선택
        template_cols = [col for col in window_counts_df.columns if col.startswith('t')]  # 템플릿 컬럼 목록
        data = window_counts_df[template_cols].fillna(0).values  # NaN을 0으로 채우고 배열화
        actual_num_templates = data.shape[1]  # 실제 템플릿 개수

        # 목표 템플릿 개수가 지정된 경우 크기 조정  # 추론/학습 정합성 유지
        if target_num_templates is not None and actual_num_templates != target_num_templates:  # 정합 필요 여부 확인
            if actual_num_templates < target_num_templates:  # 부족하면 패딩
                # 패딩: 부족한 템플릿은 0으로 채움  # 차원 확장
                padding = np.zeros((data.shape[0], target_num_templates - actual_num_templates))  # 0 패딩 생성
                data = np.hstack([data, padding])  # 우측에 붙여 확장
                print(f"📊 템플릿 개수 패딩: {actual_num_templates} → {target_num_templates}")  # 로그 출력
            else:  # 초과하면 절단
                # Truncate: 초과 템플릿은 잘라냄  # 차원 축소
                data = data[:, :target_num_templates]  # 좌측부터 target까지 유지
                print(f"📊 템플릿 개수 축소: {actual_num_templates} → {target_num_templates}")  # 로그 출력
            actual_num_templates = target_num_templates  # 실제 개수 갱신

        # 정규화  # 컬럼별 평균0, 표준편차1로 스케일링
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)  # 수치 안정성 위해 epsilon 추가

        # 2D 매트릭스를 이미지 형태로 변환 (time_steps, num_templates)  # CNN 입력으로 변형
        # 시퀀스 길이 설정 (예: 20개 윈도우씩)  # 고정 길이 시퀀스 구성
        seq_len = 20  # 시퀀스 길이
        sequences = []  # 시퀀스 누적 리스트

        for i in range(len(data) - seq_len + 1):  # 가능한 시작 위치 순회
            seq = data[i:i+seq_len]  # (seq_len, num_templates)  # 윈도우 슬라이스
            sequences.append(seq)  # 시퀀스 추가

        if not sequences:  # 시퀀스가 하나도 없는 경우
            # 데이터가 부족한 경우 패딩  # 상하단 0 패딩으로 길이 맞춤
            padded_data = np.pad(data, ((0, seq_len - len(data)), (0, 0)), mode='constant')  # 패딩 적용
            sequences = [padded_data]  # 단일 시퀀스로 구성

        # (batch, 1, time_steps, num_templates) 형태로 변환  # 채널 차원 추가
        sequences = np.array(sequences)  # 리스트를 배열로 변환
        return torch.FloatTensor(sequences).unsqueeze(1), actual_num_templates  # 텐서와 템플릿 수 반환
    
    def train(self, window_counts_path: str | Path, epochs: int = 50,
              validation_split: float = 0.2) -> Dict:  # 모델 학습 루틴
        """모델 학습"""  # 함수 설명

        # 데이터 로드  # 입력 파일에서 파케 로드
        df = pd.read_parquet(window_counts_path)  # DataFrame 로드
        data_tensor, num_templates = self.prepare_data(df)  # 텐서 변환 및 템플릿 수 획득
        data_tensor = data_tensor.to(self.device)  # 디바이스로 이동
        
        # 학습/검증 분할  # 홀드아웃 검증
        n_train = int(len(data_tensor) * (1 - validation_split))  # 학습 데이터 크기
        train_data = data_tensor[:n_train]  # 학습 부분
        val_data = data_tensor[n_train:]  # 검증 부분
        
        print(f"📊 학습 데이터: {len(train_data)}, 검증 데이터: {len(val_data)}")  # 데이터 수 출력
        print(f"📐 데이터 형태: {data_tensor.shape}")  # 텐서 형태 출력
        
        train_losses = []  # 에폭별 학습 손실 기록
        val_losses = []  # 에폭별 검증 손실 기록

        self.model.train()  # 학습 모드 설정

        for epoch in tqdm(range(epochs), desc="MS-CRED 학습"):  # 에폭 반복
            # 학습  # 배치 반복으로 최적화
            epoch_train_loss = 0  # 에폭 학습 손실 누적
            for i in range(0, len(train_data), 32):  # 배치 크기 32  # 미니배치 학습
                batch = train_data[i:i+32]  # 배치 슬라이스

                self.optimizer.zero_grad()  # 그래디언트 초기화
                reconstructed = self.model(batch)  # 순전파
                loss = F.mse_loss(reconstructed, batch)  # 재구성 손실(MSE)
                loss.backward()  # 역전파
                self.optimizer.step()  # 파라미터 업데이트

                epoch_train_loss += loss.item()  # 손실 누적

            # 검증  # 평가 모드에서 손실 측정
            self.model.eval()  # 평가 모드 설정
            epoch_val_loss = 0  # 에폭 검증 손실 누적
            with torch.no_grad():  # 그래디언트 비활성화
                for i in range(0, len(val_data), 32):  # 배치 반복
                    batch = val_data[i:i+32]  # 배치 슬라이스
                    reconstructed = self.model(batch)  # 순전파
                    loss = F.mse_loss(reconstructed, batch)  # 손실 계산
                    epoch_val_loss += loss.item()  # 누적

            self.model.train()  # 다시 학습 모드로 전환

            avg_train_loss = epoch_train_loss / (len(train_data) // 32 + 1)  # 평균 학습 손실
            avg_val_loss = epoch_val_loss / (len(val_data) // 32 + 1) if len(val_data) > 0 else 0  # 평균 검증 손실

            train_losses.append(avg_train_loss)  # 기록 추가
            val_losses.append(avg_val_loss)  # 기록 추가

            self.scheduler.step(avg_val_loss)  # Plateau 스케줄러 갱신

            if epoch % 10 == 0:  # 10에폭마다 로그 출력
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")  # 진행 로그

        # 학습 완료 후 검증 데이터의 재구성 오차 수집 (임계값 계산용)
        print("\n📊 임계값 계산을 위한 검증 데이터 재구성 오차 수집 중...")
        self.model.eval()  # 평가 모드
        val_reconstruction_errors = []  # 검증 데이터 재구성 오차 리스트

        with torch.no_grad():  # 그래디언트 비활성화
            for i in range(0, len(val_data), 32):  # 배치 반복
                batch = val_data[i:i+32]  # 배치 슬라이스
                reconstructed = self.model(batch)  # 재구성
                errors = self.model.compute_reconstruction_error(batch, reconstructed)  # 샘플별 재구성 오차
                val_reconstruction_errors.extend(errors.cpu().numpy())  # CPU로 이동 후 추가

        # 임계값 계산 (정상 데이터 분포 기반)
        val_errors_array = np.array(val_reconstruction_errors)
        threshold_stats = {
            # 백분위수 기반 임계값
            'threshold_95': float(np.percentile(val_errors_array, 95.0)),
            'threshold_99': float(np.percentile(val_errors_array, 99.0)),
            'threshold_99_9': float(np.percentile(val_errors_array, 99.9)),
            # 통계적 방법 (3-sigma)
            'mean': float(np.mean(val_errors_array)),
            'std': float(np.std(val_errors_array)),
            'threshold_3sigma': float(np.mean(val_errors_array) + 3 * np.std(val_errors_array)),
            # 중앙값 기반 (MAD - Median Absolute Deviation)
            'median': float(np.median(val_errors_array)),
            'mad': float(np.median(np.abs(val_errors_array - np.median(val_errors_array)))),
            'threshold_mad': float(np.median(val_errors_array) + 3 * 1.4826 * np.median(np.abs(val_errors_array - np.median(val_errors_array)))),
        }

        print(f"✅ 임계값 계산 완료:")
        print(f"   - 95 백분위수: {threshold_stats['threshold_95']:.4f}")
        print(f"   - 99 백분위수: {threshold_stats['threshold_99']:.4f} (권장)")
        print(f"   - 99.9 백분위수: {threshold_stats['threshold_99_9']:.4f}")
        print(f"   - 3-sigma: {threshold_stats['threshold_3sigma']:.4f}")
        print(f"   - MAD (3*1.4826): {threshold_stats['threshold_mad']:.4f}")

        return {
            'train_losses': train_losses,  # 학습 손실 목록
            'val_losses': val_losses,  # 검증 손실 목록
            'final_train_loss': train_losses[-1],  # 최종 학습 손실
            'final_val_loss': val_losses[-1],  # 최종 검증 손실
            'num_templates': num_templates,  # 사용된 템플릿 개수
            'threshold_stats': threshold_stats,  # 임계값 통계 추가
            'val_reconstruction_errors': val_reconstruction_errors,  # 검증 오차 목록 (선택적 분석용)
        }  # 통계 반환
    
    def save_model(self, path: str | Path, num_templates: int, threshold_stats: Optional[Dict] = None):  # 체크포인트 저장
        """모델 저장 (템플릿 개수 및 임계값 메타데이터 포함)"""  # 함수 설명
        checkpoint = {
            'model_state_dict': self.model.state_dict(),  # 모델 가중치
            'optimizer_state_dict': self.optimizer.state_dict(),  # 옵티마이저 상태
            'num_templates': num_templates,  # 학습 시 템플릿 개수 저장
        }

        # 임계값 통계 추가 (있는 경우)
        if threshold_stats is not None:
            checkpoint['threshold_stats'] = threshold_stats

        torch.save(checkpoint, path)  # 파일로 저장
        print(f"✅ 모델 저장 완료: {path}")  # 저장 로그
        print(f"📊 저장된 템플릿 개수: {num_templates}")  # 메타데이터 로그

        if threshold_stats is not None:
            print(f"📊 저장된 임계값 정보:")
            print(f"   - 99 백분위수 (권장): {threshold_stats['threshold_99']:.4f}")
            print(f"   - 3-sigma: {threshold_stats['threshold_3sigma']:.4f}")
    
    def load_model(self, path: str | Path):  # 체크포인트 로드
        """모델 로드"""  # 함수 설명
        checkpoint = torch.load(path, map_location=self.device)  # 디바이스에 맞게 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])  # 가중치 복원
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 옵티마이저 상태 복원
        print(f"✅ 모델 로드 완료: {path}")  # 완료 로그


class MSCREDInference:  # 추론 전용 클래스
    """MS-CRED 추론"""  # 클래스 설명

    def __init__(self, model_path: str | Path, device: str = 'cpu'):  # 생성자
        self.device = device  # 디바이스 저장
        self.model = MSCREDModel().to(device)  # 모델 인스턴스 생성 및 이동

        # 모델 로드  # 체크포인트에서 가중치 복원
        checkpoint = torch.load(model_path, map_location=device)  # 디바이스에 맞춰 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])  # 가중치 설정
        self.model.eval()  # 평가 모드로 전환

        # 학습 시 사용된 템플릿 개수 로드  # 입력 차원 정합성 확인용
        self.num_templates = checkpoint.get('num_templates', None)  # 존재 시 저장
        if self.num_templates is not None:  # 메타데이터가 있을 경우
            print(f"📊 학습 시 템플릿 개수: {self.num_templates}")  # 정보 출력
        else:  # 없을 경우 경고
            print("⚠️  경고: 모델에 템플릿 개수 메타데이터가 없습니다. 차원 불일치 문제가 발생할 수 있습니다.")  # 경고 출력

        # 학습 시 계산된 임계값 통계 로드
        self.threshold_stats = checkpoint.get('threshold_stats', None)
        if self.threshold_stats is not None:
            print(f"📊 학습 시 계산된 임계값:")
            print(f"   - 95 백분위수: {self.threshold_stats['threshold_95']:.4f}")
            print(f"   - 99 백분위수: {self.threshold_stats['threshold_99']:.4f} (권장)")
            print(f"   - 99.9 백분위수: {self.threshold_stats['threshold_99_9']:.4f}")
            print(f"   - 3-sigma: {self.threshold_stats['threshold_3sigma']:.4f}")
        else:
            print("⚠️  경고: 모델에 임계값 정보가 없습니다. 추론 데이터 기반 임계값을 사용합니다.")

    def detect_anomalies(self, window_counts_path: str | Path,
                        threshold: Optional[float] = None,
                        threshold_method: str = '99percentile') -> pd.DataFrame:  # 이상 탐지 실행
        """이상 탐지 수행

        Args:
            window_counts_path: window_counts.parquet 파일 경로
            threshold: 수동 임계값 (지정 시 이 값 사용, None이면 자동)
            threshold_method: 자동 임계값 방법
                - '99percentile': 학습 시 99 백분위수 (권장, 기본값)
                - '95percentile': 학습 시 95 백분위수
                - '99.9percentile': 학습 시 99.9 백분위수
                - '3sigma': 학습 시 평균 + 3*표준편차
                - 'mad': 학습 시 중앙값 + 3*1.4826*MAD
                - 'inference_adaptive': 추론 데이터 기반 95 백분위수 (구 방식, 권장 안함)

        Returns:
            결과 DataFrame (window_idx, start_index, reconstruction_error, is_anomaly, threshold)
        """

        # 데이터 준비  # 학습 시 템플릿 수에 맞춰 변환
        trainer = MSCREDTrainer(self.model, self.device)  # 트레이너 생성 (데이터 준비 재사용)
        df = pd.read_parquet(window_counts_path)  # 입력 데이터 로드
        data_tensor, actual_num_templates = trainer.prepare_data(df, target_num_templates=self.num_templates)  # 정합 변환
        data_tensor = data_tensor.to(self.device)  # 디바이스 이동

        if self.num_templates is not None and actual_num_templates != self.num_templates:  # 차이 존재 시
            print(f"⚠️  템플릿 개수가 학습 시와 다릅니다. 자동으로 조정되었습니다.")  # 경고 출력

        print(f"📊 추론 데이터 형태: {data_tensor.shape}")  # 입력 텐서 형태 로그

        # 재구성 오차 계산  # 배치별로 계산
        reconstruction_errors = []  # 오차 누적 리스트

        with torch.no_grad():  # 그래디언트 비활성화 (추론)
            for i in range(0, len(data_tensor), 32):  # 배치 반복
                batch = data_tensor[i:i+32]  # 배치 슬라이스
                reconstructed = self.model(batch)  # 재구성 수행
                errors = self.model.compute_reconstruction_error(batch, reconstructed)  # 오차 계산
                reconstruction_errors.extend(errors.cpu().numpy())  # CPU로 이동 후 리스트에 추가

        errors_array = np.array(reconstruction_errors)  # 리스트를 배열로 변환

        # 임계값 결정
        if threshold is not None:
            # 사용자 지정 임계값 사용
            final_threshold = threshold
            print(f"📌 사용자 지정 임계값 사용: {final_threshold:.4f}")
        elif self.threshold_stats is not None:
            # 학습 시 계산된 임계값 사용 (권장)
            if threshold_method == '99percentile':
                final_threshold = self.threshold_stats['threshold_99']
                print(f"✅ 학습 시 99 백분위수 임계값 사용: {final_threshold:.4f} (권장)")
            elif threshold_method == '95percentile':
                final_threshold = self.threshold_stats['threshold_95']
                print(f"✅ 학습 시 95 백분위수 임계값 사용: {final_threshold:.4f}")
            elif threshold_method == '99.9percentile':
                final_threshold = self.threshold_stats['threshold_99_9']
                print(f"✅ 학습 시 99.9 백분위수 임계값 사용: {final_threshold:.4f}")
            elif threshold_method == '3sigma':
                final_threshold = self.threshold_stats['threshold_3sigma']
                print(f"✅ 학습 시 3-sigma 임계값 사용: {final_threshold:.4f}")
            elif threshold_method == 'mad':
                final_threshold = self.threshold_stats['threshold_mad']
                print(f"✅ 학습 시 MAD 임계값 사용: {final_threshold:.4f}")
            elif threshold_method == 'inference_adaptive':
                final_threshold = np.percentile(errors_array, 95.0)
                print(f"⚠️  추론 데이터 기반 95 백분위수 사용: {final_threshold:.4f} (권장하지 않음)")
            else:
                # 잘못된 방법명, 기본값 사용
                final_threshold = self.threshold_stats['threshold_99']
                print(f"⚠️  알 수 없는 방법 '{threshold_method}', 99 백분위수 사용: {final_threshold:.4f}")
        else:
            # 폴백: 추론 데이터 기반 (구 방식)
            final_threshold = np.percentile(errors_array, 95.0)
            print(f"⚠️  학습 시 임계값 정보 없음. 추론 데이터 기반 95 백분위수 사용: {final_threshold:.4f}")
            print(f"   (순환 논리 문제 가능성 있음 - 모델 재학습 권장)")

        # 추론 데이터 통계 출력 (참고용)
        print(f"📊 추론 데이터 재구성 오차 통계:")
        print(f"   - 평균: {np.mean(errors_array):.4f}")
        print(f"   - 중앙값: {np.median(errors_array):.4f}")
        print(f"   - 표준편차: {np.std(errors_array):.4f}")
        print(f"   - 최소: {np.min(errors_array):.4f}")
        print(f"   - 최대: {np.max(errors_array):.4f}")

        # 결과 데이터프레임 생성  # 각 윈도우별 결과 구성
        results = []  # 결과 딕셔너리 목록
        for i, error in enumerate(reconstruction_errors):  # 각 오차에 대해 반복
            is_anomaly = error > final_threshold  # 임계값 초과 여부

            # 원본 윈도우 정보 매핑  # 시작 인덱스 매핑
            if i < len(df):  # 데이터프레임 범위 내이면
                start_index = df.iloc[i].get('start_index', i)  # 컬럼이 있으면 사용, 없으면 i
            else:  # 범위를 넘으면
                start_index = i  # 인덱스 사용

            results.append({  # 결과 레코드 추가
                'window_idx': i,  # 윈도우 인덱스
                'start_index': start_index,  # 시작 인덱스
                'reconstruction_error': float(error),  # 재구성 오차
                'is_anomaly': bool(is_anomaly),  # 이상 여부
                'threshold': float(final_threshold)  # 사용 임계값
            })  # 레코드 완료

        results_df = pd.DataFrame(results)  # 결과를 데이터프레임으로 변환

        # 통계 출력  # 요약 정보 표시
        anomaly_rate = results_df['is_anomaly'].mean()  # 이상 비율
        print(f"📈 최종 임계값: {final_threshold:.4f}")  # 임계값 로그
        print(f"🚨 이상 탐지율: {anomaly_rate:.3f} ({results_df['is_anomaly'].sum()}/{len(results_df)})")  # 탐지율 로그

        return results_df  # 결과 반환


def train_mscred(window_counts_path: str | Path, model_output_path: str | Path,
                epochs: int = 50) -> Dict:  # 단일 호출 학습 함수
    """MS-CRED 모델 학습 함수"""  # 함수 설명
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # CUDA 가능 시 GPU 사용
    print(f"🔧 사용 디바이스: {device}")  # 디바이스 로그

    model = MSCREDModel()  # 모델 생성
    trainer = MSCREDTrainer(model, device)  # 트레이너 생성

    # 학습 실행  # 지정 에폭만큼 학습
    training_stats = trainer.train(window_counts_path, epochs)  # 학습 통계 획득

    # 모델 저장 (템플릿 개수 및 임계값 메타데이터 포함)  # 추후 추론 정합성 보장
    num_templates = training_stats['num_templates']  # 템플릿 수 추출
    threshold_stats = training_stats.get('threshold_stats')  # 임계값 통계 추출
    trainer.save_model(model_output_path, num_templates, threshold_stats)  # 체크포인트 저장

    return training_stats  # 학습 결과 반환


def infer_mscred(window_counts_path: str | Path, model_path: str | Path,
                output_path: str | Path,
                threshold: Optional[float] = None,
                threshold_method: str = '99percentile') -> pd.DataFrame:  # 단일 호출 추론 함수
    """MS-CRED 이상 탐지 추론 함수

    Args:
        window_counts_path: window_counts.parquet 파일 경로
        model_path: 학습된 모델 파일 경로
        output_path: 결과 저장 경로
        threshold: 수동 임계값 (None이면 자동)
        threshold_method: 자동 임계값 방법 (기본: '99percentile')
            - '99percentile': 학습 시 99 백분위수 (권장)
            - '95percentile': 학습 시 95 백분위수
            - '99.9percentile': 학습 시 99.9 백분위수
            - '3sigma': 학습 시 평균 + 3*표준편차
            - 'mad': 학습 시 중앙값 + 3*1.4826*MAD

    Returns:
        결과 DataFrame
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 디바이스 선택
    print(f"🔧 사용 디바이스: {device}")  # 디바이스 로그

    # 추론 실행  # 모델 로드 및 이상 탐지
    inference = MSCREDInference(model_path, device)  # 추론 객체 생성
    results_df = inference.detect_anomalies(window_counts_path, threshold, threshold_method)  # 이상 탐지 실행

    # 결과 저장  # 파케 포맷으로 출력
    results_df.to_parquet(output_path, index=False)  # 파일 저장
    print(f"✅ MS-CRED 추론 결과 저장: {output_path}")  # 완료 로그

    return results_df  # 결과 반환
