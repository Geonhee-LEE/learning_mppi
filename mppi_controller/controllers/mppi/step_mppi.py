"""
Step-MPPI Controller (Single-Step MPPI via Differentiable Predictive Control)
— 43번째 변형

신경망이 MPPI proposal 분포(평균 시퀀스 + 대각 공분산 스케일)를 파라미터화
하도록 학습한다. 장기 호라이즌 MPC 목적(비용 + 제약 페널티 + 최대 엔트로피
정규화)을 학습 시점에 주입하여, 런타임에는 짧은(단일) 스텝 최적화만으로 장기
계획 정보를 암묵적으로 활용 → 초저지연.

핵심 수식:
    proposal 네트워크: (Δμ_θ(φ), logσ_θ(φ)) = NN(φ),  φ = 특징(state, ref)
    런타임 평균:  μ = U_warm + blend · Δμ_θ
    샘플:        U_k = μ + diag(σ_base·exp(logσ_θ)) ε_k,  ε_k ~ N(0, I)
    자기지도 학습:  L(θ) = E[‖Δμ_θ − (U* − U_warm)‖²] − τ·Σ logσ_θ
    zero-init 출력층 → 학습 전 Δμ_θ≈0, σ≈σ_base → Vanilla로 graceful 퇴화.

Reference: arXiv:2604.01539, "Toward Single-Step MPPI via Differentiable
Predictive Control" (2026)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import StepMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch 부재 환경
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class ProposalNetwork(nn.Module):
        """
        proposal 분포 MLP.

        입력: 특징 φ (state + reference 요약)
        출력:
            mean_delta: (N·nu,) → (N, nu) warm-start 잔차
            log_std:    (nu,)   대각 공분산 스케일 로그
        출력층 zero-init → 초기 출력 0 (graceful degradation).
        """

        def __init__(
            self,
            input_dim: int,
            N: int,
            nu: int,
            hidden_dim: int = 64,
            n_layers: int = 2,
        ):
            super().__init__()
            self.N = N
            self.nu = nu
            layers: List[nn.Module] = []
            d = input_dim
            for _ in range(n_layers):
                layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
                d = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.mean_head = nn.Linear(d, N * nu)
            self.logstd_head = nn.Linear(d, nu)
            # zero-init 출력층
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
            nn.init.zeros_(self.logstd_head.weight)
            nn.init.zeros_(self.logstd_head.bias)

        def forward(self, x):
            h = self.backbone(x)
            mean_delta = self.mean_head(h).view(-1, self.N, self.nu)
            log_std = self.logstd_head(h)
            return mean_delta, log_std


class StepExperienceBuffer:
    """(특징 φ, 잔차 타깃 U*−U_warm) 링 버퍼 + 비용 기록."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.features: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.costs: List[float] = []
        self._ptr = 0

    def add(self, feature: np.ndarray, target: np.ndarray, cost: float):
        if len(self.features) < self.capacity:
            self.features.append(feature.copy())
            self.targets.append(target.copy())
            self.costs.append(float(cost))
        else:
            self.features[self._ptr] = feature.copy()
            self.targets[self._ptr] = target.copy()
            self.costs[self._ptr] = float(cost)
            self._ptr = (self._ptr + 1) % self.capacity

    def __len__(self):
        return len(self.features)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        n = len(self.features)
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        F = np.stack([self.features[i] for i in idx])
        T = np.stack([self.targets[i] for i in idx])
        return F, T


class ProposalTrainer:
    """
    proposal 네트워크 자기지도 학습기.

    손실: MSE(mean_delta, target_residual) − entropy_weight · Σ log_std
    (최대 엔트로피 정규화로 공분산 붕괴 방지)
    """

    def __init__(self, net, lr: float, entropy_weight: float, device: str = "cpu"):
        self.net = net
        self.device = device
        self.entropy_weight = entropy_weight
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self._loss_history: List[float] = []

    def train_step(self, features: np.ndarray, targets: np.ndarray) -> float:
        F = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        T = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        mean_delta, log_std = self.net(F)
        mse = torch.mean((mean_delta - T) ** 2)
        entropy = log_std.mean()
        loss = mse - self.entropy_weight * entropy
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        val = float(loss.item())
        self._loss_history.append(val)
        return val


class StepMPPIController(MPPIController):
    """
    Step-MPPI Controller — 43번째 변형

    학습된 proposal 분포로 단일(짧은) 스텝 최적화를 수행한다. 매 스텝:
      1. 특징 φ(state, ref) → 네트워크 → (Δμ, logσ)
      2. μ = U_warm + blend·Δμ,  σ_eff = σ_base·exp(logσ) (클리핑)
      3. μ 주위 K개 샘플 → rollout/cost/softmax → 가중 평균 해 U*
      4. (φ, U*−U_warm)를 버퍼에 저장, 주기적 자기지도 학습
    torch 부재 또는 use_learned_proposal=False → Vanilla MPPI로 동작.
    """

    def __init__(
        self,
        model: RobotModel,
        params: StepMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.step_params = params

        self.nx = model.state_dim
        self.nu = model.control_dim
        self.N = params.N

        base_sigma = np.asarray(params.sigma, dtype=float)
        if base_sigma.size == 1:
            base_sigma = np.full(self.nu, float(base_sigma.reshape(-1)[0]))
        self._base_sigma = base_sigma.copy()

        self._rng = np.random.default_rng(getattr(params, "seed", None))
        self._step_count = 0

        # 학습 인프라
        self._use_net = (
            _TORCH_AVAILABLE and params.use_learned_proposal
        )
        self.input_dim = 3 * self.nx  # [state, ref_first_err, ref_terminal_err]
        if self._use_net:
            device = params.device if params.device in ("cpu", "cuda") else "cpu"
            if device == "cuda" and not (
                _TORCH_AVAILABLE and torch.cuda.is_available()
            ):
                device = "cpu"
            self._device = device
            self.net = ProposalNetwork(
                self.input_dim, self.N, self.nu,
                hidden_dim=params.proposal_hidden_dim,
                n_layers=params.proposal_n_layers,
            ).to(device)
            self.trainer = ProposalTrainer(
                self.net, params.proposal_lr, params.entropy_weight, device
            )
            self.buffer = StepExperienceBuffer(params.buffer_size)
        else:
            self.net = None
            self.trainer = None
            self.buffer = None

    # ------------------------------------------------------------------ #
    def _build_features(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> np.ndarray:
        """특징 φ = [state, ref[1]-state, ref[-1]-state] (3·nx,)."""
        ref = reference_trajectory
        first = ref[1] if ref.shape[0] > 1 else ref[0]
        term = ref[-1]
        return np.concatenate([state, first - state, term - state]).astype(np.float64)

    def _net_proposal(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """네트워크 → (mean_delta (N,nu), sigma_eff (nu,))."""
        with torch.no_grad():
            F = torch.as_tensor(
                features[None], dtype=torch.float32, device=self._device
            )
            mean_delta, log_std = self.net(F)
            mean_delta = mean_delta[0].cpu().numpy()
            scale = np.exp(np.clip(log_std[0].cpu().numpy(), -1.386, 1.386))  # ~[0.25,4]
        sigma_eff = self._base_sigma * scale
        return mean_delta, sigma_eff

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        K = self.params.K
        N = self.N
        nu = self.nu
        self._step_count += 1

        U_warm = self.U.copy()
        features = self._build_features(state, reference_trajectory)

        if self._use_net:
            mean_delta, sigma_eff = self._net_proposal(features)
            mu = U_warm + self.step_params.blend_ratio * mean_delta
            if not self.step_params.learn_covariance:
                sigma_eff = self._base_sigma
        else:
            mean_delta = np.zeros((N, nu))
            mu = U_warm
            sigma_eff = self._base_sigma

        if self.u_min is not None and self.u_max is not None:
            mu = np.clip(mu, self.u_min, self.u_max)

        # 단일(짧은) 스텝 최적화 반복
        sample_trajectories = None
        weights = None
        costs = None
        for _ in range(self.step_params.lookahead_steps):
            noise = self._rng.normal(0.0, 1.0, (K, N, nu)) * sigma_eff[None, None, :]
            sampled_controls = mu[None] + noise
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)
                noise = sampled_controls - mu[None]

            sample_trajectories = self.dynamics_wrapper.rollout(
                state, sampled_controls
            )
            costs = self.cost_function.compute_cost(
                sample_trajectories, sampled_controls, reference_trajectory
            )
            weights = self._compute_weights(costs, self.params.lambda_)
            mu = mu + np.sum(weights[:, None, None] * noise, axis=0)
            if self.u_min is not None and self.u_max is not None:
                mu = np.clip(mu, self.u_min, self.u_max)

        solution_U = mu.copy()

        # 자기지도 학습 데이터 수집 (잔차 타깃)
        if self._use_net and self.step_params.online_training:
            target_residual = solution_U - U_warm
            self.buffer.add(features, target_residual, float(np.min(costs)))
            self._maybe_train()

        self.U = solution_U
        optimal_control = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        ess = self._compute_ess(weights)
        best_idx = int(np.argmin(costs))

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "step_stats": {
                "use_net": self._use_net,
                "lookahead_steps": self.step_params.lookahead_steps,
                "proposal_delta_norm": float(np.linalg.norm(mean_delta)),
                "sigma_eff": sigma_eff.copy(),
                "buffer_size": len(self.buffer) if self.buffer else 0,
                "train_count": (
                    len(self.trainer._loss_history) if self.trainer else 0
                ),
            },
        }
        self.last_info = info
        return optimal_control, info

    def _maybe_train(self):
        """주기적 자기지도 학습."""
        if len(self.buffer) < self.step_params.min_train_samples:
            return
        if self._step_count % self.step_params.train_interval != 0:
            return
        F, T = self.buffer.sample_batch(
            self.step_params.train_batch_size, self._rng
        )
        self.trainer.train_step(F, T)

    def get_step_statistics(self) -> Dict:
        return {
            "use_net": self._use_net,
            "step_count": self._step_count,
            "buffer_size": len(self.buffer) if self.buffer else 0,
            "train_count": len(self.trainer._loss_history) if self.trainer else 0,
            "last_loss": (
                self.trainer._loss_history[-1]
                if self.trainer and self.trainer._loss_history else None
            ),
        }

    def reset(self):
        super().reset()
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"StepMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"use_net={self._use_net}, "
            f"lookahead_steps={self.step_params.lookahead_steps}, "
            f"K={self.params.K})"
        )
