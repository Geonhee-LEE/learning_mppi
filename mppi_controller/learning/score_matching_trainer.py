"""
Denoising Score Matching 학습기

Score function s_θ(x, σ, ctx) ≈ ∇_x log p_σ(x|ctx) 를 학습.
다중 σ 스케일에서 DSM Loss를 최소화.

DSM Loss:
    L(θ) = E_{σ, ε} [||s_θ(x + σε, σ, ctx) - (-ε/σ)||²]
    σ ~ {σ_1, ..., σ_L}  (기하 스케줄)
    ε ~ N(0, I)

References:
    - Song & Ermon (2019) — Generative Modeling by Estimating Gradients
    - Vincent (2011) — Connection Between Score Matching and Denoising
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List
from pathlib import Path

from mppi_controller.models.learned.score_network import ScoreNetwork


class ScoreMatchingTrainer:
    """
    Denoising Score Matching 학습기

    다중 σ 스케일에서 score function을 학습.

    Args:
        control_seq_dim: N * nu (평탄화된 제어 시퀀스 차원)
        context_dim: nx (상태 차원)
        hidden_dims: 은닉층 차원 리스트
        n_sigma_levels: DSM 노이즈 스케일 개수
        sigma_min: 최소 노이즈 스케일
        sigma_max: 최대 노이즈 스케일
        lr: 학습률
        weight_decay: L2 정규화 계수
        device: "cpu" 또는 "cuda"
    """

    def __init__(
        self,
        control_seq_dim: int,
        context_dim: int,
        hidden_dims: Optional[List[int]] = None,
        n_sigma_levels: int = 10,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.control_seq_dim = control_seq_dim
        self.context_dim = context_dim
        self.device = torch.device(device)

        # 기하 σ 스케줄: σ_min → σ_max
        self.sigma_levels = torch.tensor(
            np.geomspace(sigma_min, sigma_max, n_sigma_levels),
            dtype=torch.float32,
            device=self.device,
        )

        self.model = ScoreNetwork(
            control_seq_dim=control_seq_dim,
            context_dim=context_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> Dict:
        """
        DSM Loss로 score function 학습

        Args:
            states: (M, nx) 상태 배열
            controls: (M, N, nu) 최적 제어 시퀀스
            epochs: 에포크 수
            batch_size: 배치 크기

        Returns:
            metrics: 학습 메트릭 (losses, final_loss 등)
        """
        M = states.shape[0]

        # 제어 시퀀스 평탄화: (M, N, nu) → (M, N*nu)
        x_flat = controls.reshape(M, -1)

        # Tensor 변환
        x_t = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
        ctx_t = torch.tensor(states, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(ctx_t, x_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for ctx_batch, x_batch in loader:
                loss = self._dsm_step(ctx_batch, x_batch)
                epoch_loss += loss
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))

        return {
            "losses": losses,
            "final_loss": losses[-1] if losses else float("inf"),
            "epochs": epochs,
            "num_samples": M,
        }

    def _dsm_step(self, context: torch.Tensor, x: torch.Tensor) -> float:
        """
        단일 DSM 학습 스텝

        L(θ) = E[||s_θ(x + σε, σ, ctx) - (-ε/σ)||²]

        Args:
            context: (B, context_dim) 상태
            x: (B, control_seq_dim) 깨끗한 제어 시퀀스

        Returns:
            loss value (float)
        """
        B = x.shape[0]

        # 랜덤 σ 레벨 선택
        sigma_idx = torch.randint(0, len(self.sigma_levels), (B,), device=self.device)
        sigma = self.sigma_levels[sigma_idx]  # (B,)

        # 노이즈 생성
        eps = torch.randn_like(x)  # (B, control_seq_dim)

        # 노이즈 추가: x_noisy = x + σ·ε
        x_noisy = x + sigma[:, None] * eps

        # Score 예측
        score_pred = self.model(x_noisy, sigma, context)

        # DSM 타겟: -ε/σ
        target = -eps / sigma[:, None]

        # MSE loss
        loss = nn.functional.mse_loss(score_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def get_model(self) -> ScoreNetwork:
        """학습된 모델 반환"""
        return self.model

    def save_model(self, path: str):
        """모델 저장"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "control_seq_dim": self.control_seq_dim,
            "context_dim": self.context_dim,
            "sigma_levels": self.sigma_levels.cpu().numpy(),
        }
        torch.save(checkpoint, str(save_path))

    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(
            str(path), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
