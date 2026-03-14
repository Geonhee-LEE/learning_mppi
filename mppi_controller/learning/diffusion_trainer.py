"""
Diffusion 모델 학습기

DDPM 학습: L = ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t, c)||²

FlowMatchingTrainer와 동일 인터페이스 (호환 가능).

수식:
    Forward process: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε,  ε ~ N(0, I)
    학습 목표: ε_θ가 ε를 예측하도록 최소화
    Loss: E_{x_0, ε, t} [||ε - ε_θ(x_t, t, c)||²]

References:
    Ho et al. (2020) — DDPM
    Song et al. (2021) — DDIM
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DiffusionTrainer:
    """
    DDPM 노이즈 예측 모델 학습기.

    FlowMatchingTrainer와 동일 인터페이스:
        trainer.add_sample(state, control_sequence)
        trainer.train(epochs=10)

    Args:
        control_seq_dim: N * nu (플래튼된 제어 차원)
        context_dim: 상태 nx
        hidden_dims: 은닉층 차원
        T: 학습 타임스텝 수 (1000)
        beta_schedule: "cosine" | "linear"
        lr: 학습률
        batch_size: 배치 크기
        use_unet: True이면 TemporalUNet1D, False이면 MLPDiffusionModel
        control_dim: TemporalUNet1D용 단일 타임스텝 제어 차원 (nu)
        seq_len: TemporalUNet1D용 호라이즌 (N)
    """

    def __init__(
        self,
        control_seq_dim: int,
        context_dim: int,
        hidden_dims: Optional[List[int]] = None,
        T: int = 1000,
        beta_schedule: str = "cosine",
        lr: float = 3e-4,
        batch_size: int = 64,
        use_unet: bool = False,
        control_dim: Optional[int] = None,
        seq_len: Optional[int] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch가 필요합니다: pip install torch")

        self.control_seq_dim = control_seq_dim
        self.context_dim = context_dim
        self.T = T
        self.batch_size = batch_size

        # 노이즈 스케줄
        self.alpha_bar = self._build_schedule(T, beta_schedule)

        # 모델 초기화
        from mppi_controller.models.learned.diffusion_model import (
            MLPDiffusionModel, TemporalUNet1D
        )
        if use_unet and control_dim is not None and seq_len is not None:
            self.model = TemporalUNet1D(
                control_dim=control_dim,
                seq_len=seq_len,
                context_dim=context_dim,
            )
        else:
            self.model = MLPDiffusionModel(
                control_seq_dim=control_seq_dim,
                context_dim=context_dim,
                hidden_dims=hidden_dims or [256, 256, 256],
            )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.1
        )

        # 데이터 버퍼
        self._control_seqs: List[np.ndarray] = []  # (N*nu,)
        self._contexts: List[np.ndarray] = []       # (nx,)

        # 학습 이력
        self.train_losses: List[float] = []
        self.is_trained: bool = False

    def _build_schedule(self, T: int, schedule: str) -> np.ndarray:
        """ᾱ_t 스케줄 계산."""
        if schedule == "cosine":
            s = 0.008
            steps = np.arange(T + 1)
            f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
            alpha_bar = f / f[0]
        else:  # linear
            betas = np.linspace(1e-4, 0.02, T)
            alphas = 1.0 - betas
            alpha_bar = np.concatenate([[1.0], np.cumprod(alphas)])

        return np.clip(alpha_bar, 1e-5, 1.0 - 1e-5).astype(np.float32)

    def add_sample(self, state: np.ndarray, control_sequence: np.ndarray) -> None:
        """
        학습 샘플 추가.

        Args:
            state: (nx,) 현재 상태 (컨텍스트)
            control_sequence: (N, nu) 최적 제어 시퀀스
        """
        self._contexts.append(state.copy())
        self._control_seqs.append(control_sequence.flatten().copy())

    def add_batch(
        self,
        states: np.ndarray,
        control_sequences: np.ndarray,
    ) -> None:
        """
        배치 학습 샘플 추가.

        Args:
            states: (B, nx)
            control_sequences: (B, N, nu) 또는 (B, N*nu)
        """
        B = states.shape[0]
        ctrl = control_sequences.reshape(B, -1)
        for i in range(B):
            self._contexts.append(states[i].copy())
            self._control_seqs.append(ctrl[i].copy())

    def train(
        self,
        epochs: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        DDPM 학습.

        Args:
            epochs: 에포크 수
            verbose: 로그 출력 여부

        Returns:
            stats: {'loss': 마지막 손실, 'epochs': 학습 에포크 수, 'n_samples': 샘플 수}
        """
        n = len(self._control_seqs)
        if n < 2:
            return {"loss": float("nan"), "epochs": 0, "n_samples": n}

        # 데이터셋 구성
        X = torch.tensor(np.array(self._control_seqs), dtype=torch.float32)  # (n, D)
        C = torch.tensor(np.array(self._contexts), dtype=torch.float32)       # (n, nx)
        dataset = TensorDataset(X, C)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        alpha_bar_t = torch.tensor(self.alpha_bar, dtype=torch.float32)

        self.model.train()
        last_loss = float("nan")

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x0_batch, ctx_batch in loader:
                B = x0_batch.shape[0]

                # 균일 타임스텝 샘플
                t_idx = torch.randint(1, self.T + 1, (B,))
                ab_t = alpha_bar_t[t_idx].view(B, 1)

                # Forward: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
                eps = torch.randn_like(x0_batch)
                x_t = torch.sqrt(ab_t) * x0_batch + torch.sqrt(1.0 - ab_t) * eps

                # 정규화된 타임스텝 [0, 1]
                t_norm = t_idx.float() / self.T

                # 노이즈 예측
                eps_pred = self.model(x_t, t_norm, ctx_batch)

                # Loss: ||ε - ε_θ||²
                loss = nn.functional.mse_loss(eps_pred, eps)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            last_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(last_loss)
            self.scheduler.step()

            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{epochs} | Diffusion Loss: {last_loss:.4f}")

        self.model.eval()
        self.is_trained = True

        return {"loss": last_loss, "epochs": epochs, "n_samples": n}

    def get_model(self):
        """학습된 모델 반환."""
        return self.model

    def save(self, path: str) -> None:
        """모델 저장."""
        import torch
        torch.save({
            "model_state": self.model.state_dict(),
            "train_losses": self.train_losses,
            "alpha_bar": self.alpha_bar,
            "control_seq_dim": self.control_seq_dim,
            "context_dim": self.context_dim,
            "is_trained": self.is_trained,
        }, path)

    def load(self, path: str) -> None:
        """모델 로드."""
        import torch
        ckpt = torch.load(path, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.train_losses = ckpt.get("train_losses", [])
        self.is_trained = ckpt.get("is_trained", True)
        self.model.eval()
