"""
ALPaCA (Adaptive Learning for Probabilistic Connectionist Architecture) 동역학 모델

메타 학습된 feature extractor (frozen MLP) + Bayesian linear regression으로
closed-form 적응 수행. SGD 없이 행렬 연산만으로 적응하므로 매우 빠르다.

Usage:
    alpaca = ALPaCADynamics(state_dim=5, control_dim=2,
                            model_path="alpaca_meta_model.pth")

    # Closed-form 적응 (gradient 없음)
    alpaca.adapt(states, controls, next_states, dt)

    # MPPI rollout에 사용
    state_dot = alpaca.forward_dynamics(state, control)

    # 예측 불확실성
    unc = alpaca.get_uncertainty(state, control)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from mppi_controller.models.base_model import RobotModel


class FeatureExtractor(nn.Module):
    """메타 학습된 feature extractor MLP (frozen after meta-training)."""

    def __init__(self, input_dim, feature_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, feature_dim))

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.network(x)


class ALPaCADynamics(RobotModel):
    """
    ALPaCA 기반 동역학 모델 — Bayesian last-layer 적응.

    구성:
    1. Feature extractor φ(x, u): MLP → d차원 특징 벡터 (메타 학습, frozen)
    2. Bayesian linear regression: y = W·φ(x,u) + ε
       - Prior: μ₀ (nx×d), Λ₀ (d×d), β (noise precision)
       - Posterior (closed-form):
         Λ_n = Λ₀ + β·Σ φᵢφᵢᵀ
         μ_n = Λ_n⁻¹·(Λ₀·μ₀ᵀ + β·Σ yᵢᵀ·φᵢ)ᵀ

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        model_path: 메타 학습된 모델 경로
        device: 'cpu' or 'cuda'
        feature_dim: 특징 벡터 차원
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
        feature_dim: int = 64,
    ):
        self._state_dim = state_dim
        self._control_dim = control_dim
        self._feature_dim = feature_dim
        self.device = torch.device(device)

        self.feature_extractor = None
        self.norm_stats = None

        # Bayesian prior
        self._mu_0 = np.zeros((state_dim, feature_dim))  # prior mean (nx × d)
        self._Lambda_0 = np.eye(feature_dim)  # prior precision (d × d)
        self._beta = 1.0  # noise precision

        # Posterior (starts at prior)
        self._mu_n = self._mu_0.copy()
        self._Lambda_n = self._Lambda_0.copy()
        self._Lambda_n_inv = np.linalg.inv(self._Lambda_n)

        if model_path is not None:
            self._load_model(model_path)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def control_dim(self) -> int:
        return self._control_dim

    @property
    def model_type(self) -> str:
        return "learned"

    def _extract_features(self, state, control):
        """
        Feature extraction: (state, control) → φ (d,) or (batch, d).

        Handles normalization and torch conversion.
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not loaded.")

        # Normalize
        if self.norm_stats is not None:
            state_norm = (state - self.norm_stats["state_mean"]) / (self.norm_stats["state_std"] + 1e-8)
            control_norm = (control - self.norm_stats["control_mean"]) / (self.norm_stats["control_std"] + 1e-8)
        else:
            state_norm = state
            control_norm = control

        is_single = state.ndim == 1
        if is_single:
            inputs = np.concatenate([state_norm, control_norm])[np.newaxis, :]
        else:
            inputs = np.concatenate([state_norm, control_norm], axis=1)

        inputs_t = torch.FloatTensor(inputs).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(inputs_t).cpu().numpy()

        if is_single:
            return features[0]  # (d,)
        return features  # (batch, d)

    def forward_dynamics(self, state, control):
        """
        Bayesian 예측: y = μ_n · φ(x, u).

        Returns:
            state_dot: (nx,) or (batch, nx)
        """
        phi = self._extract_features(state, control)  # (d,) or (batch, d)

        if state.ndim == 1:
            # (nx, d) @ (d,) → (nx,)
            pred = self._mu_n @ phi
        else:
            # (batch, d) @ (d, nx) → (batch, nx)
            pred = phi @ self._mu_n.T

        # Denormalize
        if self.norm_stats is not None:
            pred = pred * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]

        return pred

    def adapt(self, states, controls, next_states, dt, restore=True, **kwargs):
        """
        Closed-form Bayesian 적응 (SGD 없음).

        1. restore=True → prior로 리셋 후 적응 (standard)
        2. Feature 추출 → posterior 업데이트

        Args:
            states: (M, nx) 상태
            controls: (M, nu) 제어
            next_states: (M, nx) 다음 상태
            dt: 시간 간격
            restore: True → prior로 리셋 후 적응

        Returns:
            float: 최종 예측 MSE
        """
        if restore:
            self.restore_prior()

        # Target: state_dot
        targets = (next_states - states) / dt
        if states.shape[1] >= 3:
            theta_diff = next_states[:, 2] - states[:, 2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            targets[:, 2] = theta_diff / dt

        # Normalize targets
        if self.norm_stats is not None:
            targets_norm = (targets - self.norm_stats["state_dot_mean"]) / (self.norm_stats["state_dot_std"] + 1e-8)
        else:
            targets_norm = targets

        # Feature extraction
        phi = self._extract_features(states, controls)  # (M, d)
        M = phi.shape[0]

        # Bayesian update:
        # Λ_n = Λ₀ + β·Σ φᵢφᵢᵀ
        phi_outer_sum = phi.T @ phi  # (d, d)
        self._Lambda_n = self._Lambda_0 + self._beta * phi_outer_sum

        # Λ_n⁻¹
        self._Lambda_n_inv = np.linalg.inv(self._Lambda_n + 1e-6 * np.eye(self._feature_dim))

        # μ_n = Λ_n⁻¹ · (Λ₀·μ₀ᵀ + β·Σ yᵢᵀ·φᵢ)ᵀ
        # Λ₀·μ₀ᵀ: (d, d) @ (d, nx) = (d, nx)  where μ₀ᵀ is (d, nx)
        prior_term = self._Lambda_0 @ self._mu_0.T  # (d, nx)
        data_term = self._beta * (phi.T @ targets_norm)  # (d, nx)
        self._mu_n = (self._Lambda_n_inv @ (prior_term + data_term)).T  # (nx, d)

        # 예측 오차 계산
        pred = phi @ self._mu_n.T  # (M, nx)
        mse = float(np.mean((pred - targets_norm) ** 2))
        return mse

    def get_uncertainty(self, state, control):
        """
        예측 불확실성: σ² = (1/β)·φᵀ·Λ_n⁻¹·φ.

        Args:
            state: (nx,) or (batch, nx)
            control: (nu,) or (batch, nu)

        Returns:
            uncertainty: (nx,) or (batch, nx) - 각 출력 차원별 예측 분산
        """
        phi = self._extract_features(state, control)

        if state.ndim == 1:
            # Scalar variance: φᵀ·Λ⁻¹·φ
            var_scalar = (1.0 / self._beta) * (phi @ self._Lambda_n_inv @ phi)
            return np.full(self._state_dim, np.sqrt(max(var_scalar, 0.0)))
        else:
            # (batch,) variance per sample
            # (batch, d) @ (d, d) → (batch, d), then sum → (batch,)
            var_batch = (1.0 / self._beta) * np.sum((phi @ self._Lambda_n_inv) * phi, axis=1)
            return np.sqrt(np.maximum(var_batch, 0.0))[:, np.newaxis] * np.ones((1, self._state_dim))

    def restore_prior(self):
        """Prior로 리셋."""
        self._mu_n = self._mu_0.copy()
        self._Lambda_n = self._Lambda_0.copy()
        self._Lambda_n_inv = np.linalg.inv(self._Lambda_n)

    def _load_model(self, model_path: str):
        """메타 학습된 ALPaCA 모델 로드."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        config = checkpoint["config"]
        self._feature_dim = config["feature_dim"]
        hidden_dims = config.get("hidden_dims", [128, 128])

        input_dim = config["state_dim"] + config["control_dim"]
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            feature_dim=self._feature_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        self.feature_extractor.eval()

        # Bayesian prior
        self._mu_0 = checkpoint["mu_0"].copy()
        self._Lambda_0 = checkpoint["Lambda_0"].copy()
        self._beta = float(checkpoint["beta"])

        # Norm stats
        self.norm_stats = checkpoint.get("norm_stats")

        # Reset posterior to prior
        self.restore_prior()

    def load_model(self, model_path: str):
        """Public load interface."""
        self._load_model(model_path)

    def get_control_bounds(self):
        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])
        return lower, upper

    def state_to_dict(self, state):
        if self._state_dim == 5:
            return {
                "x": state[0], "y": state[1], "theta": state[2],
                "v": state[3], "omega": state[4],
            }
        return {"x": state[0], "y": state[1], "theta": state[2]}

    def normalize_state(self, state):
        result = state.copy()
        result[..., 2] = np.arctan2(np.sin(state[..., 2]), np.cos(state[..., 2]))
        return result

    def __repr__(self) -> str:
        if self.feature_extractor is not None:
            n_params = sum(p.numel() for p in self.feature_extractor.parameters())
            return (
                f"ALPaCADynamics(state_dim={self._state_dim}, "
                f"feature_dim={self._feature_dim}, "
                f"extractor_params={n_params:,}, "
                f"β={self._beta:.2f})"
            )
        return f"ALPaCADynamics(state_dim={self._state_dim}, loaded=False)"
