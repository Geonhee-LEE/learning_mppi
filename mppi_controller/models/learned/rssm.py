"""
RSSM (Recurrent State Space Model)

Dreamer/TD-MPC 스타일 잠재 세계 모델.

아키텍처:
    Prior:     p(z_t | h_t)         — 과거 히스토리에서 잠재 사전 예측
    Posterior: q(z_t | h_t, o_t)   — 관측을 포함한 사후 추정
    Recurrent: h_{t+1} = f(h_t, z_t, a_t)  — GRU 기반 결정적 상태 전이

잠재 공간 상상 (Imagination):
    h_{t+1}, z_{t+1} = Prior(h_t, z_t, a_t)  — 관측 없이 N스텝 예측
    → MPPI rollout을 잠재 공간에서 수행

수식:
    ELBO = E[log p(o_t|h_t,z_t)] - KL[q(z_t|h_t,o_t) || p(z_t|h_t)]

References:
    Hafner et al. (2019) — Dream to Control (Dreamer)
    Hafner et al. (2020) — Mastering Atari with Discrete World Models (DreamerV2)
    Hansen et al. (2022) — TD-MPC
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch 모듈 (선택적 의존성)
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class _MLP(nn.Module):
        """간단한 MLP (SiLU 활성화)."""

        def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, n_layers: int = 2):
            super().__init__()
            layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            layers.append(nn.Linear(hidden, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class RSSMCore(nn.Module):
        """
        RSSM 핵심 모듈.

        결정적 경로: h_t (GRU hidden state)
        확률적 경로: z_t (가우시안 잠재 변수)

        Args:
            obs_dim: 관측 차원 (nx)
            action_dim: 행동 차원 (nu)
            deter_dim: 결정적 상태 차원 (GRU hidden)
            stoch_dim: 확률적 상태 차원 (z)
            hidden_dim: MLP 은닉 차원
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            deter_dim: int = 256,
            stoch_dim: int = 64,
            hidden_dim: int = 256,
        ):
            if not HAS_TORCH:
                raise ImportError("PyTorch 필요: pip install torch")
            super().__init__()

            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.deter_dim = deter_dim
            self.stoch_dim = stoch_dim

            # 결정적 GRU: h_{t+1} = GRU(h_t, [z_t, a_t])
            self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)

            # Prior: p(z_t | h_t) → (mean, logvar)
            self.prior_net = _MLP(deter_dim, stoch_dim * 2, hidden_dim)

            # Posterior: q(z_t | h_t, o_t) → (mean, logvar)
            self.posterior_net = _MLP(deter_dim + obs_dim, stoch_dim * 2, hidden_dim)

            # 디코더: p(o_t | h_t, z_t) → 관측 재구성
            self.decoder = _MLP(deter_dim + stoch_dim, obs_dim, hidden_dim)

            # 보상 예측기: r_t = f(h_t, z_t)
            self.reward_net = _MLP(deter_dim + stoch_dim, 1, hidden_dim // 2, n_layers=1)

        def initial_state(self, batch_size: int, device=None) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """초기 (h, z) 반환."""
            if device is None:
                device = next(self.parameters()).device
            h = torch.zeros(batch_size, self.deter_dim, device=device)
            z = torch.zeros(batch_size, self.stoch_dim, device=device)
            return h, z

        def _sample_gaussian(
            self, mean: "torch.Tensor", logvar: "torch.Tensor"
        ) -> "torch.Tensor":
            """Reparameterization trick."""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

        def prior_step(
            self,
            h: "torch.Tensor",
            z: "torch.Tensor",
            action: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Prior 전이 (관측 없이).

            h_{t+1} = GRU(h_t, [z_t, a_t])
            z_{t+1} ~ p(z | h_{t+1})

            Args:
                h: (B, deter_dim)
                z: (B, stoch_dim)
                action: (B, action_dim)

            Returns:
                h_next, z_next, prior_mean, prior_logvar
            """
            gru_input = torch.cat([z, action], dim=-1)
            h_next = self.gru(gru_input, h)

            prior_params = self.prior_net(h_next)
            prior_mean = prior_params[:, :self.stoch_dim]
            prior_logvar = prior_params[:, self.stoch_dim:].clamp(-4, 4)
            z_next = self._sample_gaussian(prior_mean, prior_logvar)

            return h_next, z_next, prior_mean, prior_logvar

        def posterior_step(
            self,
            h: "torch.Tensor",
            z: "torch.Tensor",
            action: "torch.Tensor",
            obs: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Posterior 전이 (관측 포함).

            h_{t+1} = GRU(h_t, [z_t, a_t])
            z_{t+1} ~ q(z | h_{t+1}, o_{t+1})

            Returns:
                h_next, z_next, post_mean, post_logvar, prior_mean, prior_logvar
            """
            gru_input = torch.cat([z, action], dim=-1)
            h_next = self.gru(gru_input, h)

            # Prior
            prior_params = self.prior_net(h_next)
            prior_mean = prior_params[:, :self.stoch_dim]
            prior_logvar = prior_params[:, self.stoch_dim:].clamp(-4, 4)

            # Posterior
            post_params = self.posterior_net(torch.cat([h_next, obs], dim=-1))
            post_mean = post_params[:, :self.stoch_dim]
            post_logvar = post_params[:, self.stoch_dim:].clamp(-4, 4)
            z_next = self._sample_gaussian(post_mean, post_logvar)

            return h_next, z_next, post_mean, post_logvar, prior_mean, prior_logvar

        def decode(self, h: "torch.Tensor", z: "torch.Tensor") -> "torch.Tensor":
            """잠재 상태 → 관측 재구성."""
            return self.decoder(torch.cat([h, z], dim=-1))

        def predict_reward(self, h: "torch.Tensor", z: "torch.Tensor") -> "torch.Tensor":
            """잠재 상태 → 보상 예측."""
            return self.reward_net(torch.cat([h, z], dim=-1)).squeeze(-1)

        def imagine(
            self,
            h: "torch.Tensor",
            z: "torch.Tensor",
            actions: "torch.Tensor",
        ) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
            """
            잠재 공간 상상 (Imagination) — N스텝 Prior 롤아웃.

            Args:
                h: (B, deter_dim) 초기 h
                z: (B, stoch_dim) 초기 z
                actions: (B, N, action_dim) 행동 시퀀스

            Returns:
                hs: list of (B, deter_dim), len=N+1
                zs: list of (B, stoch_dim), len=N+1
            """
            N = actions.shape[1]
            hs = [h]
            zs = [z]

            for t in range(N):
                h, z, _, _ = self.prior_step(h, z, actions[:, t, :])
                hs.append(h)
                zs.append(z)

            return hs, zs

        def kl_loss(
            self,
            post_mean: "torch.Tensor",
            post_logvar: "torch.Tensor",
            prior_mean: "torch.Tensor",
            prior_logvar: "torch.Tensor",
            free_nats: float = 1.0,
        ) -> "torch.Tensor":
            """
            KL 발산: KL[q || p]

            KL = 0.5 * (logvar_p - logvar_q + exp(logvar_q-logvar_p) +
                         (mean_q-mean_p)²/exp(logvar_p) - 1)
            """
            kl = 0.5 * (
                prior_logvar - post_logvar
                + torch.exp(post_logvar - prior_logvar)
                + (post_mean - prior_mean) ** 2 / torch.exp(prior_logvar)
                - 1
            )
            kl = kl.sum(dim=-1)  # (B,)
            return torch.clamp(kl, min=free_nats).mean()


# ─────────────────────────────────────────────────────────────────────────────
# NumPy 폴백 — PyTorch 없이 사용 가능한 선형 세계 모델
# ─────────────────────────────────────────────────────────────────────────────

class LinearWorldModel:
    """
    선형 잠재 세계 모델 (NumPy 기반, PyTorch 불필요).

    비선형 동역학을 선형 잠재 공간에서 근사:
        z_{t+1} = A @ z_t + B @ a_t + w
        o_t = C @ z_t

    EDMD / PCA 기반으로 데이터에서 학습.

    Args:
        obs_dim: 관측 차원 (nx)
        action_dim: 행동 차원 (nu)
        latent_dim: 잠재 차원
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # 행렬 초기화
        self.A = np.eye(latent_dim)
        self.B = np.zeros((latent_dim, action_dim))
        self.C = np.zeros((obs_dim, latent_dim))
        self.C[:min(obs_dim, latent_dim), :min(obs_dim, latent_dim)] = np.eye(
            min(obs_dim, latent_dim)
        )

        # PCA 인코더 (관측 → 잠재)
        self.W_enc = np.random.randn(latent_dim, obs_dim) * 0.01
        self.b_enc = np.zeros(latent_dim)

        self._is_fitted = False

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """관측 → 잠재 상태. obs: (..., nx) → (..., latent_dim)."""
        return obs @ self.W_enc.T + self.b_enc

    def decode(self, z: np.ndarray) -> np.ndarray:
        """잠재 상태 → 관측. z: (..., latent_dim) → (..., nx)."""
        return z @ self.C.T

    def predict(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """잠재 공간 1스텝 예측. z: (..., d), a: (..., nu)."""
        return z @ self.A.T + action @ self.B.T

    def imagine(
        self,
        z0: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """
        잠재 공간 N스텝 롤아웃.

        Args:
            z0: (K, latent_dim) 초기 잠재 상태
            actions: (K, N, nu)

        Returns:
            zs: (K, N+1, latent_dim)
        """
        K, N = actions.shape[:2]
        zs = np.zeros((K, N + 1, self.latent_dim))
        zs[:, 0, :] = z0

        for t in range(N):
            zs[:, t + 1, :] = (
                zs[:, t, :] @ self.A.T + actions[:, t, :] @ self.B.T
            )
        return zs

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        reg: float = 1e-4,
    ) -> float:
        """
        EDMD + PCA로 선형 세계 모델 학습.

        Args:
            observations: (T, nx) 관측 시퀀스
            actions: (T-1, nu) 행동 시퀀스
            reg: 정규화 계수

        Returns:
            rmse: 재구성 오차
        """
        T = observations.shape[0]
        if T < 3:
            raise ValueError("최소 3개 관측 필요")

        # PCA 인코더 학습 (관측 → 잠재)
        # latent_dim > obs_dim인 경우 나머지는 랜덤 투영으로 패딩
        obs_centered = observations - observations.mean(axis=0)
        cov = obs_centered.T @ obs_centered / T
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            n_pca = min(self.obs_dim, self.latent_dim)
            pca_W = eigvecs[:, idx[:n_pca]].T  # (n_pca, nx)
            if n_pca < self.latent_dim:
                # 나머지 행은 랜덤 투영으로 패딩
                extra = np.random.randn(self.latent_dim - n_pca, self.obs_dim) * 0.1
                self.W_enc = np.vstack([pca_W, extra])  # (latent_dim, nx)
            else:
                self.W_enc = pca_W  # (latent_dim, nx)
            self.b_enc = -observations.mean(axis=0) @ self.W_enc.T
        except Exception:
            self.W_enc = np.random.randn(self.latent_dim, self.obs_dim) * 0.1
            self.b_enc = np.zeros(self.latent_dim)

        # 잠재 상태 계산
        Z = self.encode(observations)  # (T, latent_dim)
        Z_curr = Z[:T - 1]             # (T-1, latent_dim)
        Z_next = Z[1:]                  # (T-1, latent_dim)
        U = actions[:T - 1]             # (T-1, nu)

        # 선형 회귀: [A | B] 학습
        Psi = np.hstack([Z_curr, U])    # (T-1, latent+nu)
        A_b = Psi.T @ Psi + reg * np.eye(Psi.shape[1])
        b_vec = Psi.T @ Z_next
        AB = np.linalg.solve(A_b, b_vec)  # (latent+nu, latent)

        self.A = AB[:self.latent_dim, :].T   # (latent, latent)
        self.B = AB[self.latent_dim:, :].T   # (latent, nu)

        # 디코더 C 학습 (잠재 → 관측)
        A3 = Z.T @ Z + reg * np.eye(self.latent_dim)
        b3 = Z.T @ observations
        self.C = np.linalg.solve(A3, b3).T  # (nx, latent)

        self._is_fitted = True

        # 오차 계산
        Z_pred = Z_curr @ self.A.T + U @ self.B.T
        obs_pred = Z_pred @ self.C.T
        rmse = float(np.sqrt(np.mean((obs_pred - observations[1:]) ** 2)))
        return rmse
