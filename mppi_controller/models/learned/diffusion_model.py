"""
Diffusion 모델 — MPPI 궤적 샘플러용 노이즈 예측 네트워크

DDPM (Denoising Diffusion Probabilistic Model) 기반.
학습된 ε_θ(x_t, t, context) 모델로 역확산(denoising)을 수행하여
최적 제어 시퀀스 분포를 근사.

Flow-MPPI의 FlowMatchingModel과 동일 인터페이스 (결합 가능).

수식:
    Forward:  q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
    Backward: ε_θ(x_t, t, c) = predicted noise
    DDIM:     x_{t-1} = √ᾱ_{t-1} * (x_t - √(1-ᾱ_t)*ε_θ)/√ᾱ_t
                       + √(1-ᾱ_{t-1}) * ε_θ

References:
    Ho et al. (2020) — DDPM
    Song et al. (2021) — DDIM (accelerated sampling)
    Chi et al. (2023) — Diffusion Policy (manipulation)
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _sinusoidal_embedding(t: "torch.Tensor", dim: int) -> "torch.Tensor":
    """
    Sinusoidal 시간 임베딩.

    Args:
        t: (B,) 또는 (B, 1) 정규화된 타임스텝 [0, 1]
        dim: 임베딩 차원

    Returns:
        emb: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=t.device) * (np.log(10000) / (half - 1))
    )
    args = t.float().view(-1, 1) * freqs.view(1, -1)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MLPDiffusionModel(nn.Module):
    """
    MLP 기반 노이즈 예측 모델 (경량).

    모바일 로봇 제어 시퀀스의 노이즈 예측.
    FlowMatchingModel과 동일 구조이지만 속도장 대신 노이즈 예측.

    Input: [x_t (N*nu), time_emb(t), context (nx)]
    Output: ε_θ (N*nu) — 예측 노이즈

    Args:
        control_seq_dim: 제어 시퀀스 플래튼 차원 (N * nu)
        context_dim: 컨텍스트 차원 (nx — 상태)
        hidden_dims: 은닉층 차원 리스트
        time_emb_dim: 시간 임베딩 차원
        dropout: 드롭아웃 비율
    """

    def __init__(
        self,
        control_seq_dim: int,
        context_dim: int,
        hidden_dims: Optional[List[int]] = None,
        time_emb_dim: int = 64,
        dropout: float = 0.0,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch가 필요합니다: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        self.control_seq_dim = control_seq_dim
        self.context_dim = context_dim
        self.time_emb_dim = time_emb_dim

        # 시간 임베딩 → 은닉층 투영
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # 입력 차원: x_t + time_emb + context
        in_dim = control_seq_dim + time_emb_dim + context_dim

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, control_seq_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: "torch.Tensor",
        t: "torch.Tensor",
        context: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        노이즈 예측.

        Args:
            x_t: (B, control_seq_dim) 노이즈가 섞인 제어 시퀀스
            t: (B,) 정규화된 타임스텝 [0, 1]
            context: (B, context_dim) 조건 컨텍스트 (상태)

        Returns:
            eps_pred: (B, control_seq_dim) 예측 노이즈
        """
        B = x_t.shape[0]

        # 시간 임베딩
        t_emb = _sinusoidal_embedding(t, self.time_emb_dim)  # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (B, time_emb_dim)

        # 입력 결합
        if context is not None:
            h = torch.cat([x_t, t_emb, context], dim=-1)
        else:
            # 컨텍스트 없이: 0으로 패딩
            dummy = torch.zeros(B, self.context_dim, device=x_t.device)
            h = torch.cat([x_t, t_emb, dummy], dim=-1)

        return self.net(h)


class TemporalUNet1D(nn.Module):
    """
    1D Temporal U-Net 노이즈 예측 모델 (고표현력).

    Down-sampling → Bottleneck → Up-sampling 구조.
    Skip connections로 세부 정보 보존.

    제어 시퀀스를 길이 N 시계열로 처리.

    Args:
        control_dim: 단일 타임스텝 제어 차원 (nu)
        seq_len: 호라이즌 길이 (N)
        context_dim: 컨텍스트 차원 (nx)
        channels: U-Net 채널 구성 (기본: [32, 64, 128])
        time_emb_dim: 시간 임베딩 차원
    """

    def __init__(
        self,
        control_dim: int,
        seq_len: int,
        context_dim: int,
        channels: Optional[List[int]] = None,
        time_emb_dim: int = 64,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch가 필요합니다")

        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        self.control_dim = control_dim
        self.seq_len = seq_len
        self.context_dim = context_dim
        self.time_emb_dim = time_emb_dim

        # 시간 임베딩
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, channels[0]),
        )

        # 컨텍스트 투영
        self.ctx_proj = nn.Linear(context_dim, channels[0])

        # Encoder
        self.down_convs = nn.ModuleList()
        in_ch = control_dim
        for ch in channels:
            self.down_convs.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(4, ch), ch),
                nn.SiLU(),
                nn.Conv1d(ch, ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(4, ch), ch),
                nn.SiLU(),
            ))
            in_ch = ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1] * 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, channels[-1] * 2), channels[-1] * 2),
            nn.SiLU(),
            nn.Conv1d(channels[-1] * 2, channels[-1], kernel_size=3, padding=1),
            nn.GroupNorm(min(4, channels[-1]), channels[-1]),
            nn.SiLU(),
        )

        # Decoder (역순)
        self.up_convs = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            ch = channels[i]
            prev_ch = channels[i - 1] if i > 0 else control_dim
            in_dec = ch * 2  # skip connection concat
            self.up_convs.append(nn.Sequential(
                nn.Conv1d(in_dec, ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(4, ch), ch),
                nn.SiLU(),
                nn.Conv1d(ch, prev_ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(4, prev_ch), prev_ch),
                nn.SiLU(),
            ))

        # 출력
        self.out_conv = nn.Conv1d(control_dim, control_dim, kernel_size=1)

    def forward(
        self,
        x_t: "torch.Tensor",
        t: "torch.Tensor",
        context: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Args:
            x_t: (B, N*nu) 노이즈 섞인 제어 시퀀스
            t: (B,) 타임스텝
            context: (B, nx) 상태 컨텍스트

        Returns:
            eps_pred: (B, N*nu)
        """
        B = x_t.shape[0]
        N, nu = self.seq_len, self.control_dim

        # (B, N*nu) → (B, nu, N) — Conv1D 입력 형식
        h = x_t.view(B, N, nu).transpose(1, 2)  # (B, nu, N)

        # 시간 + 컨텍스트 임베딩
        t_emb = _sinusoidal_embedding(t, self.time_emb_dim)
        t_feat = self.time_mlp(t_emb)  # (B, channels[0])

        if context is not None:
            ctx_feat = self.ctx_proj(context)  # (B, channels[0])
            bias = (t_feat + ctx_feat).unsqueeze(-1)  # (B, channels[0], 1)
        else:
            bias = t_feat.unsqueeze(-1)

        # Encoder
        skips = []
        for conv in self.down_convs:
            h = conv(h)
            if h.shape[1] == bias.shape[1]:
                h = h + bias
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for up_conv, skip in zip(self.up_convs, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = up_conv(h)

        # 출력 (B, nu, N) → (B, N*nu)
        h = self.out_conv(h)
        return h.transpose(1, 2).reshape(B, N * nu)


# ─────────────────────────────────────────────────────────────────────────────
# NumPy fallback (PyTorch 없이 사용 가능한 간단한 MLP)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyLinear:
    """NumPy 기반 선형 레이어."""

    def __init__(self, in_dim: int, out_dim: int):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


class NumpyMLPDiffusion:
    """
    NumPy 기반 MLP 노이즈 예측 모델 (PyTorch 없이 사용).

    추론 전용 (학습은 PyTorch 버전 사용 후 가중치 export).

    Args:
        control_seq_dim: N * nu
        context_dim: nx
        hidden_dims: 은닉층 차원
    """

    def __init__(
        self,
        control_seq_dim: int,
        context_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.control_seq_dim = control_seq_dim
        self.context_dim = context_dim

        in_dim = control_seq_dim + 64 + context_dim  # 64: time_emb_dim
        dims = [in_dim] + hidden_dims + [control_seq_dim]

        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(NumpyLinear(dims[i], dims[i + 1]))

    def _time_embed(self, t: np.ndarray) -> np.ndarray:
        """(B,) → (B, 64)"""
        dim = 64
        half = dim // 2
        freqs = np.exp(-np.arange(half) * (np.log(10000) / (half - 1)))
        args = t[:, None] * freqs[None, :]
        return np.concatenate([np.sin(args), np.cos(args)], axis=-1)

    def predict(self, x_t: np.ndarray, t: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x_t: (B, control_seq_dim)
            t: (B,)
            context: (B, context_dim) or None

        Returns:
            eps: (B, control_seq_dim)
        """
        t_emb = self._time_embed(t)

        if context is None:
            context = np.zeros((x_t.shape[0], self.context_dim))

        h = np.concatenate([x_t, t_emb, context], axis=-1)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)  # ReLU

        return h

    def load_weights(self, weights: dict) -> None:
        """사전 학습 가중치 로드 (PyTorch export 형식)."""
        for i, layer in enumerate(self.layers):
            key_w = f"net.{i * 2}.weight"
            key_b = f"net.{i * 2}.bias"
            if key_w in weights:
                layer.W = weights[key_w].T
                layer.b = weights[key_b]
