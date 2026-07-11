"""
PGD-MPPI Controller (Preconditioned Gradient Descent MPPI) — 40번째 변형

MPPI를 KL 정규화 변분 문제의 전처리 경사 하강(preconditioned gradient descent)
으로 재해석한다. 고정 공분산 가우시안의 경우, 표준 MPPI 업데이트는 unit-step
(α=1) 전처리 경사 스텝으로 정확히 복원된다. 이를 일반화하여
  (1) 스텝 크기 α 조절,
  (2) 제어 주기당 다중 경사 스텝,
  (3) Gibbs-tilted 분포의 공분산을 이용한 공분산 전처리 적응
을 지원한다.

핵심 수식:
    자유 에너지       F(μ) = -λ log E_{q}[exp(-S(U)/λ)]
    softmax 가중치    w_i = exp(-(S_i - min S)/λ) / Σ_j exp(...)
    전처리 경사       g̃ = Σ_i w_i ε_i
    평균 업데이트     μ ← μ + α · g̃           (α=1, 1스텝 → 표준 MPPI)
    공분산 전처리     Σ ← (1-β)Σ + β · Cov_w(U)  (tilted 경험 공분산)

기본값(step_size=1.0, n_grad_steps=1, adapt_covariance=False)에서 Vanilla
MPPI와 정확히 동일하게 동작 (graceful superset).

Reference: arXiv:2603.24489, "Model Predictive Path Integral Control as
Preconditioned Gradient Descent" (2026)
"""

import numpy as np
from typing import Dict, Tuple, Optional

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import PGDMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class PGDMPPIController(MPPIController):
    """
    PGD-MPPI Controller — 40번째 변형

    표준 MPPI를 전처리 경사 하강의 한 스텝으로 보고 이를 일반화한다.
    매 제어 주기마다 `n_grad_steps`회의 전처리 경사 업데이트를 수행하며,
    각 스텝은 현재 평균 μ 주위에서 노이즈를 (재)샘플링하고 softmax 가중
    경사 g̃ = Σ w_i ε_i 로 μ를 갱신한다. 선택적으로 Gibbs-tilted 경험
    공분산으로 샘플링 공분산을 적응시킨다(전처리 적응).

    Args:
        model: RobotModel 인스턴스
        params: PGDMPPIParams
        cost_function: CostFunction (None → 기본 복합 비용)
        noise_sampler: NoiseSampler (사용되지 않음; PGD는 적응 공분산을 위해
            자체 RNG로 샘플링. 호환성을 위해 인자는 유지)
    """

    def __init__(
        self,
        model: RobotModel,
        params: PGDMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.pgd_params = params

        nu = model.control_dim
        base_sigma = np.asarray(params.sigma, dtype=float)
        if base_sigma.ndim == 0 or base_sigma.size == 1:
            base_sigma = np.full(nu, float(base_sigma.reshape(-1)[0]))
        self._base_sigma = base_sigma.copy()          # (nu,) 기준 표준편차
        self._sigma_scale = np.ones(nu)               # (nu,) 적응 스케일

        # 적응 공분산용 독립 RNG (재현성)
        self._rng = np.random.default_rng(getattr(params, "seed", None))

        # 통계
        self._grad_norm_history = []

    # ------------------------------------------------------------------ #
    def _current_sigma(self) -> np.ndarray:
        """현재(적응된) 표준편차 (nu,)"""
        return self._base_sigma * self._sigma_scale

    def _sample_noise(self, K: int, sigma: np.ndarray) -> np.ndarray:
        """현재 sigma로 노이즈 샘플링 (K, N, nu)"""
        N = self.params.N
        nu = self._base_sigma.shape[0]
        return self._rng.normal(0.0, 1.0, (K, N, nu)) * sigma[None, None, :]

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        PGD-MPPI 제어 계산.

        n_grad_steps회 전처리 경사 업데이트를 수행한 뒤 receding horizon
        시프트하고 첫 제어를 반환한다.
        """
        K = self.params.K
        N = self.params.N
        lambda_ = self.params.lambda_
        step_size = self.pgd_params.step_size

        mu = self.U.copy()  # (N, nu) 작업 평균

        sample_trajectories = None
        weights = None
        costs = None
        last_grad_norm = 0.0

        for it in range(self.pgd_params.n_grad_steps):
            sigma = self._current_sigma()

            # 첫 스텝이거나 재샘플링 옵션이면 새 노이즈
            if it == 0 or self.pgd_params.resample_each_step:
                noise = self._sample_noise(K, sigma)  # (K, N, nu)

            sampled_controls = mu[None] + noise  # (K, N, nu)
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)
                noise = sampled_controls - mu[None]

            sample_trajectories = self.dynamics_wrapper.rollout(
                state, sampled_controls
            )
            costs = self.cost_function.compute_cost(
                sample_trajectories, sampled_controls, reference_trajectory
            )
            weights = self._compute_weights(costs, lambda_)

            # 전처리 경사 g̃ = Σ w_i ε_i
            grad = np.sum(weights[:, None, None] * noise, axis=0)  # (N, nu)

            if self.pgd_params.normalize_gradient:
                ess = self._compute_ess(weights)
                grad = grad * (K / max(ess, 1.0))

            last_grad_norm = float(np.linalg.norm(grad))
            mu = mu + step_size * grad

            if self.u_min is not None and self.u_max is not None:
                mu = np.clip(mu, self.u_min, self.u_max)

            # 공분산 전처리 적응 (Gibbs-tilted 경험 공분산)
            if self.pgd_params.adapt_covariance:
                self._adapt_covariance(sampled_controls, mu, weights)

        self._grad_norm_history.append(last_grad_norm)

        # 업데이트된 평균 저장 + receding horizon 시프트
        self.U = mu
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
            "temperature": lambda_,
            "ess": ess,
            "num_samples": K,
            "pgd_stats": {
                "n_grad_steps": self.pgd_params.n_grad_steps,
                "step_size": step_size,
                "grad_norm": last_grad_norm,
                "sigma_scale": self._sigma_scale.copy(),
            },
        }
        self.last_info = info
        return optimal_control, info

    def _adapt_covariance(
        self, sampled_controls: np.ndarray, mu: np.ndarray, weights: np.ndarray
    ):
        """
        Gibbs-tilted 경험 공분산으로 샘플링 표준편차 스케일을 적응.

        per-control-dim 가중 분산을 계산하고, 기준 분산 대비 스케일을 EMA로
        갱신한 뒤 [cov_min_scale, cov_max_scale]로 클리핑.
        """
        # 가중 분산 (per dim): E_w[(U-μ)^2], 시간축 평균
        dev = sampled_controls - mu[None]                  # (K, N, nu)
        var_w = np.sum(
            weights[:, None, None] * dev ** 2, axis=0
        )                                                  # (N, nu)
        var_w = np.mean(var_w, axis=0)                     # (nu,)
        std_w = np.sqrt(np.maximum(var_w, 1e-12))          # (nu,)

        target_scale = std_w / np.maximum(self._base_sigma, 1e-9)
        beta = self.pgd_params.cov_step_size
        self._sigma_scale = (1.0 - beta) * self._sigma_scale + beta * target_scale
        self._sigma_scale = np.clip(
            self._sigma_scale,
            self.pgd_params.cov_min_scale,
            self.pgd_params.cov_max_scale,
        )

    def get_pgd_statistics(self) -> Dict:
        """누적 PGD 통계 반환"""
        hist = self._grad_norm_history
        return {
            "n_steps": len(hist),
            "mean_grad_norm": float(np.mean(hist)) if hist else 0.0,
            "sigma_scale": self._sigma_scale.copy(),
        }

    def reset(self):
        super().reset()
        self._sigma_scale = np.ones(self._base_sigma.shape[0])
        self._grad_norm_history = []

    def __repr__(self) -> str:
        return (
            f"PGDMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"step_size={self.pgd_params.step_size}, "
            f"n_grad_steps={self.pgd_params.n_grad_steps}, "
            f"adapt_covariance={self.pgd_params.adapt_covariance}, "
            f"K={self.params.K})"
        )
