"""
LoRA (Low-Rank Adaptation) 기반 동역학 모델

사전 학습된 MLP의 각 Linear 층에 Low-Rank Adapter를 삽입하여
소수 파라미터(~10%)만으로 빠른 온라인 적응을 수행.

Reference:
    Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"
    Pan et al. (2025) "Learning on the Fly" (UZH)

Usage:
    lora = LoRADynamics(state_dim=9, control_dim=8,
                         model_path="best_model.pth",
                         lora_rank=4, lora_alpha=1.0,
                         inner_lr=0.01, inner_steps=5)
    lora.save_meta_weights()

    # Online adaptation (same interface as MAMLDynamics)
    loss = lora.adapt(states, controls, next_states, dt)

    # MPPI rollout
    state_dot = lora.forward_dynamics(state, control)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from mppi_controller.models.learned.neural_dynamics import NeuralDynamics


class LoRALinear(nn.Module):
    """
    LoRA-augmented Linear layer.

    y = W @ x + b + (alpha / rank) * A @ B @ x

    Original weights (W, b) are frozen.
    Only A (d_out, rank) and B (rank, d_in) are trainable.
    A is initialized to zero → initial output matches original.
    """

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        d_out, d_in = original.weight.shape

        # Freeze original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        # LoRA matrices: A @ B @ x → (d_out, rank) @ (rank, d_in) @ (d_in,)
        # A initialized to zero → output starts identical to original
        self.lora_A = nn.Parameter(torch.zeros(d_out, rank))
        # B initialized with scaled normal
        self.lora_B = nn.Parameter(torch.randn(rank, d_in) / np.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        # LoRA delta: x @ B^T @ A^T * scale
        lora_delta = (x @ self.lora_B.T @ self.lora_A.T) * self.scale
        return base + lora_delta

    def reset_lora(self):
        """Reset LoRA to zero (= return to base model)."""
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B, std=1.0 / np.sqrt(self.rank))


class LoRADynamics(NeuralDynamics):
    """
    LoRA 기반 온라인 적응 동역학 모델.

    NeuralDynamics를 상속하며, 각 Linear layer를 LoRALinear로 교체.
    MAMLDynamics와 동일한 adapt() 인터페이스 제공 → ResidualDynamics + MPPI에 호환.

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        model_path: 사전 학습된 모델 경로
        device: 'cpu' or 'cuda'
        lora_rank: LoRA 행렬 rank (작을수록 파라미터 적음)
        lora_alpha: LoRA 스케일링 팩터
        inner_lr: 적응 학습률
        inner_steps: 적응 gradient step 수
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
    ):
        super().__init__(state_dim, control_dim, model_path, device)

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self._meta_lora_state = None
        self._online_optimizer = None

        # Apply LoRA to loaded model
        if self.model is not None:
            self._apply_lora()

    def _apply_lora(self):
        """Replace all nn.Linear in the model with LoRALinear."""
        self._replace_linear_with_lora(self.model)

    def _replace_linear_with_lora(self, module: nn.Module):
        """Recursively replace nn.Linear layers with LoRALinear."""
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                lora_layer = LoRALinear(
                    child, rank=self.lora_rank, alpha=self.lora_alpha
                ).to(self.device)
                setattr(module, name, lora_layer)
            elif isinstance(child, nn.Sequential):
                # Replace inside Sequential using index access to handle shared modules
                new_layers = []
                for i in range(len(child)):
                    sub_child = child[i]
                    if isinstance(sub_child, nn.Linear):
                        lora_layer = LoRALinear(
                            sub_child, rank=self.lora_rank, alpha=self.lora_alpha
                        ).to(self.device)
                        new_layers.append(lora_layer)
                    else:
                        new_layers.append(sub_child)
                setattr(module, name, nn.Sequential(*new_layers))
            else:
                self._replace_linear_with_lora(child)

    def _get_lora_params(self):
        """Get only LoRA parameters (A, B matrices)."""
        params = []
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                params.append(module.lora_A)
                params.append(module.lora_B)
        return params

    def get_trainable_params(self) -> int:
        """학습 가능 파라미터 수 (LoRA only)."""
        return sum(p.numel() for p in self._get_lora_params())

    def get_total_params(self) -> int:
        """전체 파라미터 수."""
        return sum(p.numel() for p in self.model.parameters())

    def save_meta_weights(self):
        """현재 LoRA state를 메타 파라미터로 저장."""
        lora_params = self._get_lora_params()
        self._meta_lora_state = [p.data.clone() for p in lora_params]

    def restore_meta_weights(self):
        """메타 LoRA 파라미터로 복원."""
        if self._meta_lora_state is not None:
            lora_params = self._get_lora_params()
            for p, saved in zip(lora_params, self._meta_lora_state):
                p.data.copy_(saved)

    def reset_lora(self):
        """LoRA를 0으로 리셋 (= 원본 사전학습 모델)."""
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.reset_lora()

    def adapt(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        dt: float,
        restore: bool = True,
        sample_weights: Optional[np.ndarray] = None,
        temporal_decay: Optional[float] = None,
    ) -> float:
        """
        Few-shot 적응: LoRA 파라미터만 SGD 업데이트.

        MAMLDynamics.adapt()와 동일 인터페이스.

        Args:
            states: (M, nx) 최근 상태
            controls: (M, nu) 최근 제어
            next_states: (M, nx) 다음 상태
            dt: 시간 간격
            restore: True → 메타 LoRA 복원 후 적응
            sample_weights: (M,) 샘플별 가중치
            temporal_decay: 시간 감쇠 비율

        Returns:
            float: 최종 loss 값
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if restore:
            self.restore_meta_weights()
        self.model.train()

        # Target: state_dot = (next_state - state) / dt
        targets = (next_states - states) / dt

        # Angle wrapping for theta (index 2)
        if states.shape[1] >= 3:
            theta_diff = next_states[:, 2] - states[:, 2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            targets[:, 2] = theta_diff / dt

        inputs_t = self._prepare_inputs(states, controls)
        targets_t = self._prepare_targets(targets)

        # Compute sample weights
        M = states.shape[0]
        weights = np.ones(M, dtype=np.float32)

        if temporal_decay is not None:
            decay_weights = np.array(
                [temporal_decay ** (M - 1 - i) for i in range(M)],
                dtype=np.float32,
            )
            weights *= decay_weights

        if sample_weights is not None:
            weights *= sample_weights.astype(np.float32)

        weights /= weights.sum()
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Optimizer: only LoRA params
        lora_params = self._get_lora_params()
        if restore or self._online_optimizer is None:
            self._online_optimizer = torch.optim.SGD(
                lora_params, lr=self.inner_lr
            )

        use_weighted = temporal_decay is not None or sample_weights is not None

        loss_val = 0.0
        for _ in range(self.inner_steps):
            self._online_optimizer.zero_grad()
            pred = self.model(inputs_t)
            if use_weighted:
                per_sample_loss = ((pred - targets_t) ** 2).mean(dim=1)
                loss = (per_sample_loss * weights_t).sum()
            else:
                loss = F.mse_loss(pred, targets_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            self._online_optimizer.step()
            loss_val = loss.item()

        self.model.eval()
        return loss_val

    def _prepare_inputs(self, states, controls):
        """numpy → normalized tensor [state_norm, control_norm]."""
        if self.norm_stats is not None:
            state_norm = (states - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_norm = (controls - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
        else:
            state_norm = states
            control_norm = controls

        inputs = np.concatenate([state_norm, control_norm], axis=1)
        return torch.FloatTensor(inputs).to(self.device)

    def _prepare_targets(self, state_dots):
        """numpy → normalized tensor."""
        if self.norm_stats is not None:
            targets_norm = (state_dots - self.norm_stats["state_dot_mean"]) / self.norm_stats["state_dot_std"]
        else:
            targets_norm = state_dots

        return torch.FloatTensor(targets_norm).to(self.device)

    def __repr__(self) -> str:
        if self.model is not None:
            total = self.get_total_params()
            trainable = self.get_trainable_params()
            ratio = trainable / total * 100 if total > 0 else 0
            return (
                f"LoRADynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"rank={self.lora_rank}, "
                f"trainable={trainable:,}/{total:,} ({ratio:.1f}%), "
                f"inner_lr={self.inner_lr}, "
                f"inner_steps={self.inner_steps})"
            )
        return (
            f"LoRADynamics("
            f"state_dim={self._state_dim}, "
            f"control_dim={self._control_dim}, "
            f"loaded=False)"
        )
