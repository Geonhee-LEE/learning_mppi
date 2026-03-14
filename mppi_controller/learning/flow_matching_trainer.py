"""
Flow Matching 학습 파이프라인

Conditional Flow Matching (CFM) Loss로 속도장 v_θ 학습.

CFM Loss:
    x₁ = data (optimal controls)
    x₀ ~ N(0, I)
    t ~ U[0, 1]
    x_t = (1-t)*x₀ + t*x₁           (optimal transport path)
    target_v = x₁ - x₀               (constant velocity)
    loss = ||v_θ(x_t, t, state) - target_v||²

References:
    - Lipman et al. (2023) — Flow Matching
    - Kurtz & Burdick (2025) — GPC self-supervised flow
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List
from pathlib import Path

from mppi_controller.models.learned.flow_matching_model import FlowMatchingModel


class FlowMatchingTrainer:
    """
    CFM 속도장 학습기

    Args:
        control_seq_dim: N * nu (평탄화된 제어 시퀀스 차원)
        context_dim: nx (상태 차원)
        hidden_dims: 은닉층 차원 리스트
        lr: 학습률
        weight_decay: L2 정규화 계수
        device: "cpu" 또는 "cuda"
    """

    def __init__(
        self,
        control_seq_dim: int,
        context_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.control_seq_dim = control_seq_dim
        self.context_dim = context_dim
        self.device = torch.device(device)

        self.model = FlowMatchingModel(
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
        optimal_controls: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> Dict:
        """
        CFM Loss로 학습

        Args:
            states: (M, nx) 상태 배열
            optimal_controls: (M, N, nu) 최적 제어 시퀀스
            epochs: 에포크 수
            batch_size: 배치 크기

        Returns:
            metrics: 학습 메트릭 (losses, final_loss 등)
        """
        M = states.shape[0]

        # 제어 시퀀스 평탄화: (M, N, nu) → (M, N*nu)
        x1_flat = optimal_controls.reshape(M, -1)

        # 정규화 통계 저장
        self._x1_mean = x1_flat.mean(axis=0)
        self._x1_std = x1_flat.std(axis=0) + 1e-8
        self._ctx_mean = states.mean(axis=0)
        self._ctx_std = states.std(axis=0) + 1e-8

        # Tensor 변환
        x1_t = torch.tensor(x1_flat, dtype=torch.float32, device=self.device)
        ctx_t = torch.tensor(states, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(ctx_t, x1_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for ctx_batch, x1_batch in loader:
                loss = self._cfm_step(ctx_batch, x1_batch)
                epoch_loss += loss
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))

        return {
            "losses": losses,
            "final_loss": losses[-1] if losses else float("inf"),
            "epochs": epochs,
            "num_samples": M,
        }

    def _cfm_step(
        self, context: torch.Tensor, x1: torch.Tensor
    ) -> float:
        """
        단일 CFM 학습 스텝

        Args:
            context: (B, nx) 상태
            x1: (B, control_seq_dim) 최적 제어 (target)

        Returns:
            loss value (float)
        """
        B = x1.shape[0]

        # x₀ ~ N(0, I)
        x0 = torch.randn_like(x1)

        # t ~ U[0, 1]
        t = torch.rand(B, device=self.device)

        # Optimal transport interpolation: x_t = (1-t)*x₀ + t*x₁
        t_expand = t[:, None]
        x_t = (1 - t_expand) * x0 + t_expand * x1

        # Target velocity: v* = x₁ - x₀
        target_v = x1 - x0

        # Model prediction
        pred_v = self.model(x_t, t, context)

        # MSE loss
        loss = nn.functional.mse_loss(pred_v, target_v)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_model(self) -> FlowMatchingModel:
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
        }
        if hasattr(self, "_x1_mean"):
            checkpoint["x1_mean"] = self._x1_mean
            checkpoint["x1_std"] = self._x1_std
            checkpoint["ctx_mean"] = self._ctx_mean
            checkpoint["ctx_std"] = self._ctx_std
        torch.save(checkpoint, str(save_path))

    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "x1_mean" in checkpoint:
            self._x1_mean = checkpoint["x1_mean"]
            self._x1_std = checkpoint["x1_std"]
            self._ctx_mean = checkpoint["ctx_mean"]
            self._ctx_std = checkpoint["ctx_std"]
