"""
Score Network — Denoising Score Matching용 score function 네트워크

s_θ(U, σ, state) ≈ ∇_U log p(U|state) 를 근사.
MPPI 가우시안 샘플에 score 방향 bias를 추가하여 저비용 영역으로 유도.

수식:
    DSM Loss: L(θ) = E[||s_θ(U + σε, σ, state) - (-ε/σ)||²]
    Score-guided: ε_guided = ε + α · σ² · s_θ(U + ε, σ, state)

References:
    - Song & Ermon (2019) — Generative Modeling by Estimating Gradients
    - Li & Chen (2025) — Score-guided MPPI
"""

import numpy as np
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SigmaEmbedding(nn.Module):
    """
    Sinusoidal σ 임베딩

    로그 스케일 σ를 고차원 주기 임베딩으로 변환.
    DiffusionModel의 시간 임베딩과 동일 패턴.

    Args:
        emb_dim: 임베딩 차원
    """

    def __init__(self, emb_dim: int = 64):
        if not HAS_TORCH:
            raise ImportError("PyTorch가 필요합니다: pip install torch")
        super().__init__()
        self.emb_dim = emb_dim
        half = emb_dim // 2
        # 주파수 사전 계산
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32) * (np.log(10000) / max(half - 1, 1))
        )
        self.register_buffer("freqs", freqs)

    def forward(self, sigma: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            sigma: (B,) 노이즈 스케일

        Returns:
            emb: (B, emb_dim)
        """
        # log(sigma)를 사용하여 스케일 독립적 임베딩
        log_sigma = torch.log(sigma.float().clamp(min=1e-8)).view(-1, 1)
        args = log_sigma * self.freqs.view(1, -1)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, emb_dim)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ScoreNetwork(nn.Module):
    """
    Score function s_θ(U, σ, state) → ∇_U log p(U|state)

    입력: [U_flat, σ_emb, state] 결합
    출력: score vector (control_seq_dim,)

    Zero-init 출력층: 학습 초기 s_θ ≈ 0 → 순수 가우시안과 동일.

    Args:
        control_seq_dim: N * nu (평탄화된 제어 시퀀스 차원)
        context_dim: nx (상태 차원)
        hidden_dims: 은닉층 차원 리스트
        sigma_emb_dim: σ 임베딩 차원
    """

    def __init__(
        self,
        control_seq_dim: int,
        context_dim: int,
        hidden_dims: Optional[List[int]] = None,
        sigma_emb_dim: int = 64,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch가 필요합니다: pip install torch")
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.control_seq_dim = control_seq_dim
        self.context_dim = context_dim
        self.sigma_emb_dim = sigma_emb_dim

        # σ 임베딩
        self.sigma_embedding = SigmaEmbedding(sigma_emb_dim)

        # 입력 차원: U_flat + σ_emb + state
        in_dim = control_seq_dim + sigma_emb_dim + context_dim

        # MLP with residual connections (2-layer blocks)
        layers = []
        prev_dim = in_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim

        self.net = nn.Sequential(*layers)

        # Zero-init 출력층 (학습 초기 score ≈ 0)
        self.output_layer = nn.Linear(prev_dim, control_seq_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        U: "torch.Tensor",
        sigma: "torch.Tensor",
        context: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Score function 계산

        Args:
            U: (B, control_seq_dim) 제어 시퀀스
            sigma: (B,) 노이즈 스케일
            context: (B, context_dim) 상태 컨텍스트

        Returns:
            score: (B, control_seq_dim) score vector
        """
        B = U.shape[0]

        # σ 임베딩
        sigma_emb = self.sigma_embedding(sigma)  # (B, sigma_emb_dim)

        # 입력 결합
        if context is not None:
            h = torch.cat([U, sigma_emb, context], dim=-1)
        else:
            dummy = torch.zeros(B, self.context_dim, device=U.device)
            h = torch.cat([U, sigma_emb, dummy], dim=-1)

        h = self.net(h)
        return self.output_layer(h)
