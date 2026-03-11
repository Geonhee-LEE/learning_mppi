"""
Chance Constraint 비용 함수 (C2U-MPPI)

Unscented Transform으로 전파된 공분산을 기반으로
장애물의 유효 반경을 동적 확장하여 확률적 충돌 회피.

r_eff = r + margin_factor * κ_α * √(trace(Σ_pos))
  κ_α = Φ^{-1}(1 - α)  (정규분포 분위수)

공분산이 클수록(불확실할수록) 장애물이 커지는 효과.
"""

import numpy as np
from typing import List, Optional
from mppi_controller.controllers.mppi.cost_functions import CostFunction


def _normal_ppf(p: float) -> float:
    """정규분포 역함수 (percent point function) 근사 — scipy 의존 없음"""
    # Abramowitz & Stegun 26.2.23 근사
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p < 0.5:
        return -_normal_ppf(1 - p)
    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)


class ChanceConstraintCost(CostFunction):
    """
    확률적 기회 제약 비용 함수

    공분산 궤적과 결합하여 불확실성에 비례하는 장애물 확장.
    공분산이 없으면 고정 반경 장애물 비용으로 fallback.

    Args:
        obstacles: 원형 장애물 [(x, y, radius), ...]
        chance_alpha: 충돌 허용 확률 (0.05 → κ_α ≈ 1.645)
        weight: 비용 가중치
        margin_factor: κ_α 스케일 조정 계수
    """

    def __init__(
        self,
        obstacles: List[tuple],
        chance_alpha: float = 0.05,
        weight: float = 500.0,
        margin_factor: float = 1.0,
    ):
        self.obstacles = list(obstacles)
        self.chance_alpha = chance_alpha
        self.weight = weight
        self.margin_factor = margin_factor

        # κ_α = Φ^{-1}(1 - α)
        self.kappa_alpha = _normal_ppf(1 - chance_alpha)

        # 공분산 궤적 (C2UMPPIController가 매 스텝 설정)
        self._cov_trajectory: Optional[List[np.ndarray]] = None
        self._effective_radii: Optional[np.ndarray] = None

    def set_covariance_trajectory(self, cov_trajectory: List[np.ndarray]):
        """
        공분산 궤적 설정 (매 compute_control 호출 전에 호출됨)

        Args:
            cov_trajectory: [(nx, nx)] * (N+1) 각 타임스텝의 공분산 행렬
        """
        self._cov_trajectory = cov_trajectory
        self._precompute_effective_radii()

    def _precompute_effective_radii(self):
        """공분산에서 유효 반경 사전 계산"""
        if self._cov_trajectory is None or len(self.obstacles) == 0:
            self._effective_radii = None
            return

        N_plus_1 = len(self._cov_trajectory)
        n_obs = len(self.obstacles)
        self._effective_radii = np.zeros((N_plus_1, n_obs))

        for t in range(N_plus_1):
            cov = self._cov_trajectory[t]
            # 위치 공분산 (2x2 블록)
            sigma_pos = cov[:2, :2]
            sigma_eff = np.sqrt(np.trace(sigma_pos))

            for j, (_, _, r) in enumerate(self.obstacles):
                self._effective_radii[t, j] = (
                    r + self.margin_factor * self.kappa_alpha * sigma_eff
                )

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        기회 제약 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        K, N_plus_1, _ = trajectories.shape
        costs = np.zeros(K)

        if len(self.obstacles) == 0:
            return costs

        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        for j, (ox, oy, r) in enumerate(self.obstacles):
            # 거리: (K, N+1)
            dx = positions[:, :, 0] - ox
            dy = positions[:, :, 1] - oy
            dist = np.sqrt(dx**2 + dy**2)

            # 유효 반경: (N+1,) 또는 고정
            if self._effective_radii is not None:
                r_eff = self._effective_radii[:N_plus_1, j]  # (N+1,)
            else:
                r_eff = r  # 고정 반경 (공분산 없음)

            # 위반량: (K, N+1)
            violation = np.maximum(0.0, r_eff - dist)

            # 이차 페널티
            costs += self.weight * np.sum(violation**2, axis=1)

        return costs

    def get_effective_radii(self) -> Optional[np.ndarray]:
        """유효 반경 반환 (N+1, n_obstacles) 또는 None"""
        return self._effective_radii

    def reset_covariance(self):
        """공분산 초기화"""
        self._cov_trajectory = None
        self._effective_radii = None
