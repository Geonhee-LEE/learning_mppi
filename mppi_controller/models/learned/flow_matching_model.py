"""
Conditional Flow Matching 속도장 모델

CFM 기반 v_θ(x_t, t, context) MLP.
노이즈 x₀~N(0,I)를 ODE 적분하여 최적 제어 시퀀스 x₁으로 전송.

References:
    - Lipman et al. (2023) — Flow Matching for Generative Modeling
    - Kurtz & Burdick (2025) — GPC: self-supervised flow for MPPI
    - Mizuta & Leung (2025) — CFM-MPPI
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time t ∈ [0, 1]"""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or scalar time in [0, 1]
        Returns:
            emb: (B, embed_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatchingModel(nn.Module):
    """
    CFM 속도장 v_θ(x_t, t, context)

    입력: [x_t (flat control seq), time_emb(t), state (context)]
    출력: v (flat control seq) — 속도 벡터

    Args:
        control_seq_dim: N * nu (평탄화된 제어 시퀀스 차원)
        context_dim: nx (상태 차원)
        hidden_dims: 은닉층 차원 리스트
        time_embed_dim: 시간 임베딩 차원
    """

    def __init__(
        self,
        control_seq_dim: int,
        context_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.control_seq_dim = control_seq_dim
        self.context_dim = context_dim
        self.time_embed_dim = time_embed_dim

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)

        # MLP: [x_t, time_emb, context] → v
        input_dim = control_seq_dim + time_embed_dim + context_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, control_seq_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        속도장 예측

        Args:
            x_t: (B, control_seq_dim) — 현재 위치
            t: (B,) — 시간 in [0, 1]
            context: (B, context_dim) — 조건 (상태)

        Returns:
            v: (B, control_seq_dim) — 속도 벡터
        """
        t_emb = self.time_embedding(t)  # (B, time_embed_dim)
        inp = torch.cat([x_t, t_emb, context], dim=-1)
        return self.network(inp)

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        num_samples: int = 1,
        num_steps: int = 5,
        solver: str = "euler",
    ) -> torch.Tensor:
        """
        ODE 적분으로 제어 시퀀스 생성: x₀ ~ N(0,I) → x₁

        Args:
            context: (context_dim,) 또는 (B, context_dim) 상태
            num_samples: 생성할 샘플 수
            num_steps: ODE 적분 스텝 수
            solver: "euler" 또는 "midpoint"

        Returns:
            x1: (num_samples, control_seq_dim) — 생성된 제어 시퀀스
        """
        self.eval()
        device = next(self.parameters()).device

        # context 확장: (num_samples, context_dim)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        ctx = context.expand(num_samples, -1).to(device)

        # x₀ ~ N(0, I)
        x = torch.randn(num_samples, self.control_seq_dim, device=device)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((num_samples,), t_val, device=device)

            if solver == "euler":
                v = self.forward(x, t, ctx)
                x = x + v * dt
            elif solver == "midpoint":
                v1 = self.forward(x, t, ctx)
                x_mid = x + v1 * (dt / 2)
                t_mid = torch.full((num_samples,), t_val + dt / 2, device=device)
                v2 = self.forward(x_mid, t_mid, ctx)
                x = x + v2 * dt
            else:
                raise ValueError(f"Unknown solver: {solver}")

        return x
