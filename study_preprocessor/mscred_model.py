"""
MS-CRED (Multi-Scale Convolutional Recurrent Encoder-Decoder) 모델 구현

MS-CRED는 시계열 이상탐지를 위한 멀티스케일 컨볼루션 오토인코더입니다.
로그 템플릿 카운트 매트릭스를 입력으로 받아 재구성 오차를 통해 이상을 탐지합니다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import json
from tqdm import tqdm


class MultiScaleConvBlock(nn.Module):
    """멀티스케일 컨볼루션 블록"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
        # 각 커널의 채널 수를 균등하게 분배하되 나머지는 첫 번째 커널에 할당
        channels_per_kernel = out_channels // len(kernel_sizes)
        remainder = out_channels % len(kernel_sizes)
        
        # 각 커널 크기별 컨볼루션 레이어
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # 첫 번째 커널에 나머지 채널 추가
            current_channels = channels_per_kernel + (remainder if i == 0 else 0)
            self.convs.append(
                nn.Conv2d(in_channels, current_channels, kernel_size=k, padding=k//2)
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 각 커널 크기별로 컨볼루션 적용 후 채널 차원에서 결합
        conv_outputs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outputs, dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class AttentionModule(nn.Module):
    """어텐션 모듈"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv_att = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 채널별 어텐션 가중치 계산
        att_weights = torch.sigmoid(self.conv_att(x))
        return x * att_weights


class MSCREDEncoder(nn.Module):
    """MS-CRED 인코더"""
    
    def __init__(self, input_channels: int = 1, base_channels: int = 32):
        super().__init__()
        
        # 멀티스케일 컨볼루션 레이어들
        self.conv1 = MultiScaleConvBlock(input_channels, base_channels)
        self.conv2 = MultiScaleConvBlock(base_channels, base_channels * 2)
        self.conv3 = MultiScaleConvBlock(base_channels * 2, base_channels * 4)
        
        # 어텐션 모듈들
        self.att1 = AttentionModule(base_channels)
        self.att2 = AttentionModule(base_channels * 2)
        self.att3 = AttentionModule(base_channels * 4)
        
        # 다운샘플링
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (batch, 1, height, width) - height: time_steps, width: num_templates
        
        # 첫 번째 스케일
        x1 = self.conv1(x)
        x1_att = self.att1(x1)
        
        # 두 번째 스케일 
        x2 = self.pool(x1_att)
        x2 = self.conv2(x2)
        x2_att = self.att2(x2)
        
        # 세 번째 스케일
        x3 = self.pool(x2_att)
        x3 = self.conv3(x3)
        x3_att = self.att3(x3)
        
        return [x1_att, x2_att, x3_att]


class MSCREDDecoder(nn.Module):
    """MS-CRED 디코더 (간소화 버전)"""
    
    def __init__(self, base_channels: int = 32, output_channels: int = 1):
        super().__init__()
        
        # 간단한 업샘플링 디코더
        self.deconv3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 
                                         kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 
                                         kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, encoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        # 가장 작은 피처맵(가장 깊은 인코딩)만 사용
        _, _, x3_att = encoder_outputs
        
        # 업샘플링으로 원래 크기로 복원
        x = self.relu(self.deconv3(x3_att))
        x = self.relu(self.deconv2(x))
        x = self.deconv1(x)
        
        return x


class MSCREDModel(nn.Module):
    """MS-CRED 전체 모델"""
    
    def __init__(self, input_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.encoder = MSCREDEncoder(input_channels, base_channels)
        self.decoder = MSCREDDecoder(base_channels, input_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 크기 저장
        input_shape = x.shape

        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        # 출력 크기를 입력 크기와 정확히 맞춤
        # ONNX 추적 호환성: shape 비교 대신 항상 interpolate 수행
        # (입력과 출력이 같으면 interpolate는 no-op)
        reconstructed = F.interpolate(
            reconstructed,
            size=(input_shape[2], input_shape[3]),
            mode='bilinear',
            align_corners=False
        )

        return reconstructed
    
    def compute_reconstruction_error(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """재구성 오차 계산"""
        return F.mse_loss(reconstructed, x, reduction='none').mean(dim=[1, 2, 3])


class MSCREDTrainer:
    """MS-CRED 모델 트레이너"""
    
    def __init__(self, model: MSCREDModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def prepare_data(self, window_counts_df: pd.DataFrame) -> torch.Tensor:
        """윈도우 카운트 데이터를 텐서로 변환"""
        # start_index 컬럼 제거하고 템플릿 카운트만 추출
        template_cols = [col for col in window_counts_df.columns if col.startswith('t')]
        data = window_counts_df[template_cols].fillna(0).values
        
        # 정규화
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        # 2D 매트릭스를 이미지 형태로 변환 (time_steps, num_templates)
        # 시퀀스 길이 설정 (예: 20개 윈도우씩)
        seq_len = 20
        sequences = []
        
        for i in range(len(data) - seq_len + 1):
            seq = data[i:i+seq_len]  # (seq_len, num_templates)
            sequences.append(seq)
        
        if not sequences:
            # 데이터가 부족한 경우 패딩
            padded_data = np.pad(data, ((0, seq_len - len(data)), (0, 0)), mode='constant')
            sequences = [padded_data]
        
        # (batch, 1, time_steps, num_templates) 형태로 변환
        sequences = np.array(sequences)
        return torch.FloatTensor(sequences).unsqueeze(1)
    
    def train(self, window_counts_path: str | Path, epochs: int = 50, 
              validation_split: float = 0.2) -> Dict:
        """모델 학습"""
        
        # 데이터 로드
        df = pd.read_parquet(window_counts_path)
        data_tensor = self.prepare_data(df).to(self.device)
        
        # 학습/검증 분할
        n_train = int(len(data_tensor) * (1 - validation_split))
        train_data = data_tensor[:n_train]
        val_data = data_tensor[n_train:]
        
        print(f"📊 학습 데이터: {len(train_data)}, 검증 데이터: {len(val_data)}")
        print(f"📐 데이터 형태: {data_tensor.shape}")
        
        train_losses = []
        val_losses = []
        
        self.model.train()
        
        for epoch in tqdm(range(epochs), desc="MS-CRED 학습"):
            # 학습
            epoch_train_loss = 0
            for i in range(0, len(train_data), 32):  # 배치 크기 32
                batch = train_data[i:i+32]
                
                self.optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = F.mse_loss(reconstructed, batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # 검증
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for i in range(0, len(val_data), 32):
                    batch = val_data[i:i+32]
                    reconstructed = self.model(batch)
                    loss = F.mse_loss(reconstructed, batch)
                    epoch_val_loss += loss.item()
            
            self.model.train()
            
            avg_train_loss = epoch_train_loss / (len(train_data) // 32 + 1)
            avg_val_loss = epoch_val_loss / (len(val_data) // 32 + 1) if len(val_data) > 0 else 0
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            self.scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
    def save_model(self, path: str | Path):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✅ 모델 저장 완료: {path}")
    
    def load_model(self, path: str | Path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 모델 로드 완료: {path}")


class MSCREDInference:
    """MS-CRED 추론"""
    
    def __init__(self, model_path: str | Path, device: str = 'cpu'):
        self.device = device
        self.model = MSCREDModel().to(device)
        
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def detect_anomalies(self, window_counts_path: str | Path, 
                        threshold_percentile: float = 95.0) -> pd.DataFrame:
        """이상 탐지 수행"""
        
        # 데이터 준비
        trainer = MSCREDTrainer(self.model, self.device)
        df = pd.read_parquet(window_counts_path)
        data_tensor = trainer.prepare_data(df).to(self.device)
        
        print(f"📊 추론 데이터 형태: {data_tensor.shape}")
        
        # 재구성 오차 계산
        reconstruction_errors = []
        
        with torch.no_grad():
            for i in range(0, len(data_tensor), 32):
                batch = data_tensor[i:i+32]
                reconstructed = self.model(batch)
                errors = self.model.compute_reconstruction_error(batch, reconstructed)
                reconstruction_errors.extend(errors.cpu().numpy())
        
        # 임계값 계산
        errors_array = np.array(reconstruction_errors)
        threshold = np.percentile(errors_array, threshold_percentile)
        
        # 결과 데이터프레임 생성
        results = []
        for i, error in enumerate(reconstruction_errors):
            is_anomaly = error > threshold
            
            # 원본 윈도우 정보 매핑
            if i < len(df):
                start_index = df.iloc[i].get('start_index', i)
            else:
                start_index = i
            
            results.append({
                'window_idx': i,
                'start_index': start_index,
                'reconstruction_error': float(error),
                'is_anomaly': bool(is_anomaly),
                'threshold': float(threshold)
            })
        
        results_df = pd.DataFrame(results)
        
        # 통계 출력
        anomaly_rate = results_df['is_anomaly'].mean()
        print(f"📈 재구성 오차 임계값: {threshold:.4f} ({threshold_percentile}%)")
        print(f"🚨 이상 탐지율: {anomaly_rate:.3f} ({results_df['is_anomaly'].sum()}/{len(results_df)})")
        
        return results_df


def train_mscred(window_counts_path: str | Path, model_output_path: str | Path, 
                epochs: int = 50) -> Dict:
    """MS-CRED 모델 학습 함수"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 사용 디바이스: {device}")
    
    model = MSCREDModel()
    trainer = MSCREDTrainer(model, device)
    
    # 학습 실행
    training_stats = trainer.train(window_counts_path, epochs)
    
    # 모델 저장
    trainer.save_model(model_output_path)
    
    return training_stats


def infer_mscred(window_counts_path: str | Path, model_path: str | Path, 
                output_path: str | Path, threshold_percentile: float = 95.0) -> pd.DataFrame:
    """MS-CRED 이상 탐지 추론 함수"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 사용 디바이스: {device}")
    
    # 추론 실행
    inference = MSCREDInference(model_path, device)
    results_df = inference.detect_anomalies(window_counts_path, threshold_percentile)
    
    # 결과 저장
    results_df.to_parquet(output_path, index=False)
    print(f"✅ MS-CRED 추론 결과 저장: {output_path}")
    
    return results_df
