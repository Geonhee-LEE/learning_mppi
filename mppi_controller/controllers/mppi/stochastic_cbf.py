"""
Stochastic CBF 비용 함수 (Itô correction) + Risk-Aware path-integral CBF 비용

확률 미분 방정식 (SDE) 동역학:
    dx = (f(x) + g(x)·u) dt + σ(x) dw

에 대한 두 가지 안전 비용:

1. StochasticCBFCost — 확률적 zeroing-CBF (Clark 2021, cbfkit stochastic_cbfs.py):
   Itô 보정에 의해 barrier drift 조건에 0.5·Tr[σᵀ·∇²h·σ] 이 추가됨.

   생성자(generator) 조건:
       A h(x) = ∇h·(f + g·u) + 0.5·Tr[σᵀ∇²h σ] ≥ -α·h + β

   원형 barrier h = ||p - p_o||² - R² 의 경우 위치 블록 Hessian = 2I 이므로
       0.5·Tr[σᵀ(2I)σ] = Σ_i σ_pos_i²    (해석적 — autodiff 불필요)

   이산시간 제약 (결정론적 rollout 의 Δh 가 ∇h·(f+gu) 를 근사):
       Δh_t/dt + α·h_t + ito_term ≥ β

   ※ 부호 주의: 볼록한 원형 barrier 에서는 ∇²h ⪰ 0 이므로 Itô 항이
   양수 — 등방성 노이즈는 기대 제곱 거리를 '증가' 시키므로 조건을
   완화한다 (cbfkit 수학 그대로). 노이즈에 대한 보수성은 buffer β > 0,
   또는 아래 RiskAwareCBFCost (σ 가 클수록 마진 증가) 로 확보한다.

2. RiskAwareCBFCost — 경로 적분(path-integral) 위험 한계
   (Black et al., CDC 2023; cbfkit path_integral_barrier.py):

       P(min_t h(x_t) < 0) ≤ ρ   을 보장하기 위해
       h(x_t) ≥ margin(t) = sqrt(2·t)·η·erfinv(1 - 2ρ)

   여기서 η 는 제약 집합 위에서 ||∇h·σ|| 의 상한.
   근사: η 를 진짜 sup_{x∈S} 대신 현재 샘플 궤적 배치가 방문하는
   영역 위의 최대값 max_{k,t} ||∇h(x_t^k)·σ_pos|| 로 계산한다
   (배치가 방문하는 영역에 대해서는 유효한 상한; 전역 상한이
   필요하면 grad_bound 인자로 직접 지정).

Ref:
    A. Clark, "Control Barrier Functions for Stochastic Systems" (Automatica 2021)
    M. Black et al., "Safety Under Uncertainty: Tight Bounds with
    Risk-Aware Control Barrier Functions" (CDC 2023)
    cbfkit: generate_constraints/stochastic_cbfs.py, path_integral_barrier.py
"""

import numpy as np
from scipy.special import erfinv
from typing import Dict, List, Optional, Union

from mppi_controller.controllers.mppi.cost_functions import CostFunction


class StochasticCBFCost(CostFunction):
    """
    Stochastic CBF 비용 함수 (Itô correction)

    이산시간 제약 (모듈 docstring 참조):
        C_t = Δh_t/dt + α·h_t + ito - β ≥ 0
    위반 비용:
        cost = weight · Σ_t max(0, -C_t)

    σ = 0, β = 0 이면 vanilla 이산 CBF 로 축약:
        StochasticCBFCost(alpha=a, weight=w·dt)
        ≡ ControlBarrierCost(cbf_alpha=a·dt, cbf_weight=w)

    Args:
        obstacles: List of (x, y, radius) 장애물 정의
        alpha: class-K 게인 (연속시간 rate, > 0)
        beta: 안전 buffer (≥ 0). 노이즈 보수성 확보용.
        sigma_process: (nx,) 상태별 프로세스 노이즈 표준편차
            (위치 성분 [0], [1] 만 Itô 항에 사용). None 이면 0.
        weight: 위반 비용 가중치
        safety_margin: 추가 안전 마진 (m)
        dt: 타임스텝 (초)
    """

    def __init__(
        self,
        obstacles: List[tuple],
        alpha: float = 1.0,
        beta: float = 0.0,
        sigma_process: Optional[np.ndarray] = None,
        weight: float = 1000.0,
        safety_margin: float = 0.1,
        dt: float = 0.05,
    ):
        assert alpha > 0, f"alpha must be > 0, got {alpha}"
        assert beta >= 0, f"beta must be >= 0, got {beta}"
        assert dt > 0, f"dt must be > 0, got {dt}"

        self.obstacles = obstacles
        self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.safety_margin = safety_margin
        self.dt = dt

        if sigma_process is None:
            sigma_process = np.zeros(2)
        self.sigma_process = np.atleast_1d(np.asarray(sigma_process, dtype=float))

        # Itô correction: 0.5·Tr[σᵀ(∇²h)σ], ∇²h = 2I (위치 블록)
        #   = 0.5 · Σ_i 2·σ_pos_i² = Σ_i σ_pos_i²
        sigma_pos = self.sigma_process[:2]
        self.ito_correction = float(np.sum(sigma_pos**2))

    def get_ito_correction(self) -> float:
        """해석적 Itô 보정값 Σ_i σ_pos_i² 반환"""
        return self.ito_correction

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Stochastic CBF 위반 비용 계산 (fully vectorized)

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            costs: (K,) 각 샘플의 비용
        """
        K = trajectories.shape[0]
        costs = np.zeros(K)

        if not self.obstacles:
            return costs

        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin

            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            h = dx**2 + dy**2 - effective_r**2  # (K, N+1)

            # C_t = Δh/dt + α·h_t + ito - β ≥ 0
            condition = (
                (h[:, 1:] - h[:, :-1]) / self.dt
                + self.alpha * h[:, :-1]
                + self.ito_correction
                - self.beta
            )  # (K, N)

            violation = np.maximum(0.0, -condition)
            costs += self.weight * np.sum(violation, axis=1)

        return costs

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.obstacles = obstacles

    def __repr__(self) -> str:
        return (
            f"StochasticCBFCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"ito={self.ito_correction:.4f}, "
            f"weight={self.weight})"
        )


class RiskAwareCBFCost(CostFunction):
    """
    Risk-Aware path-integral CBF 비용 함수 (Black CDC 2023)

    P(min_t h(x_t) < 0) ≤ ρ 를 위해 시간 의존 마진을 요구:
        h(x_t) ≥ margin(t) = sqrt(2·t) · η · erfinv(1 - 2ρ)

    (마진은 Brownian martingale 항 ∫∇h·σ dw 의 reflection bound
     P(sup_s≤t |M_s| ≥ c) ≤ 2(1 - Φ(c/√⟨M⟩_t)), ⟨M⟩_t ≤ η²t 에서 유도)

    η 근사 (모듈 docstring 참조): 기본은 현재 배치 궤적 위의
    max ||∇h·σ_pos||. grad_bound 지정 시 η = grad_bound · ||σ_pos||₂.

    위반 비용:
        cost = weight · Σ_t max(0, margin(t) - h(x_t))

    Args:
        obstacles: List of (x, y, radius) 장애물 정의
        rho: 위험 예산 (0 < ρ < 1). ρ=0.5 → 마진 0 (erfinv(0)=0),
            ρ < 0.5 일수록 보수적.
        sigma_process: (nx,) 상태별 프로세스 노이즈 표준편차
            (위치 성분만 사용). None 이면 0.
        weight: 위반 비용 가중치
        safety_margin: 추가 안전 마진 (m)
        dt: 타임스텝 (초)
        grad_bound: ||∇h|| 의 사용자 지정 상한 (선택).
            지정 시 η = grad_bound·||σ_pos||₂ 고정.
    """

    def __init__(
        self,
        obstacles: List[tuple],
        rho: float = 0.05,
        sigma_process: Optional[np.ndarray] = None,
        weight: float = 1000.0,
        safety_margin: float = 0.1,
        dt: float = 0.05,
        grad_bound: Optional[float] = None,
    ):
        assert 0.0 < rho < 1.0, f"rho must be in (0, 1), got {rho}"
        assert dt > 0, f"dt must be > 0, got {dt}"

        self.obstacles = obstacles
        self.rho = rho
        self.weight = weight
        self.safety_margin = safety_margin
        self.dt = dt
        self.grad_bound = grad_bound

        if sigma_process is None:
            sigma_process = np.zeros(2)
        self.sigma_process = np.atleast_1d(np.asarray(sigma_process, dtype=float))
        self.sigma_pos = self.sigma_process[:2]
        if self.sigma_pos.shape[0] < 2:
            self.sigma_pos = np.pad(self.sigma_pos, (0, 2 - self.sigma_pos.shape[0]))

        # erfinv(1 - 2ρ): ρ=0.5 → 0, ρ→0 → +∞
        self.erf_factor = float(erfinv(1.0 - 2.0 * rho))

        # get_margin() 기본 η: 경계 부근 ||∇h|| ≈ 2·r_eff 사용
        if grad_bound is not None:
            self._eta_default = grad_bound * float(np.linalg.norm(self.sigma_pos))
        elif self.obstacles:
            max_r_eff = max(obs[2] + safety_margin for obs in self.obstacles)
            self._eta_default = 2.0 * max_r_eff * float(np.linalg.norm(self.sigma_pos))
        else:
            self._eta_default = 0.0

        # 마지막 compute_cost 에서 계산된 η (디버깅/플롯용)
        self._last_eta: Optional[float] = None

    def get_margin(
        self,
        t: Union[float, np.ndarray],
        eta: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """
        시간 t 에서의 위험 인지 마진 반환:
            margin(t) = sqrt(2·t) · η · erfinv(1 - 2ρ)

        Args:
            t: 시간 (초), 스칼라 또는 배열
            eta: ||∇h·σ|| 상한. None 이면 마지막 compute_cost 의 값,
                그것도 없으면 경계 기반 기본값 사용.

        Returns:
            margin: t 와 같은 형상
        """
        if eta is None:
            eta = self._last_eta if self._last_eta is not None else self._eta_default
        t_arr = np.asarray(t, dtype=float)
        margin = np.sqrt(2.0 * t_arr) * eta * self.erf_factor
        if np.isscalar(t) or t_arr.ndim == 0:
            return float(margin)
        return margin

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Risk-aware 마진 위반 비용 계산 (fully vectorized)

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            costs: (K,) 각 샘플의 비용
        """
        K, n_steps = trajectories.shape[0], trajectories.shape[1]
        costs = np.zeros(K)

        if not self.obstacles:
            return costs

        positions = trajectories[:, :, :2]  # (K, N+1, 2)
        t_grid = np.arange(n_steps) * self.dt  # (N+1,)
        sqrt_2t = np.sqrt(2.0 * t_grid)  # (N+1,)

        sigma_norm = float(np.linalg.norm(self.sigma_pos))
        max_eta = 0.0

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin

            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            h = dx**2 + dy**2 - effective_r**2  # (K, N+1)

            # η: ||∇h·σ_pos|| = sqrt((2dx·σx)² + (2dy·σy)²) 의 배치 상한
            if self.grad_bound is not None:
                eta = self.grad_bound * sigma_norm
            else:
                grad_sigma_norm = np.sqrt(
                    (2.0 * dx * self.sigma_pos[0]) ** 2
                    + (2.0 * dy * self.sigma_pos[1]) ** 2
                )  # (K, N+1)
                eta = float(np.max(grad_sigma_norm)) if grad_sigma_norm.size else 0.0
            max_eta = max(max_eta, eta)

            margin = sqrt_2t * eta * self.erf_factor  # (N+1,)

            violation = np.maximum(0.0, margin[np.newaxis, :] - h)  # (K, N+1)
            costs += self.weight * np.sum(violation, axis=1)

        self._last_eta = max_eta
        return costs

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.obstacles = obstacles

    def __repr__(self) -> str:
        return (
            f"RiskAwareCBFCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"rho={self.rho}, "
            f"erf_factor={self.erf_factor:.4f}, "
            f"weight={self.weight})"
        )
