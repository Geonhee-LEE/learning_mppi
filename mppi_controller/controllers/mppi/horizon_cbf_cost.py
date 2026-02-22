"""
Horizon-Weighted CBF 비용 함수

시간 할인(γ^t) CBF 비용 — 가까운 미래의 barrier 위반에 더 높은 페널티를 부여.
safe_control 리포의 MPC-CBF horizon 아이디어를 MPPI 비용 함수로 변환.

기존 ControlBarrierCost와 동일한 CostFunction 인터페이스.
γ < 1: 가까운 미래 위반 중시 (보수적)
γ = 1: 기존 CBF와 동일
"""

import numpy as np
from typing import List, Dict
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class HorizonWeightedCBFCost(CostFunction):
    """
    Horizon-Weighted Discrete-time CBF 비용 함수

    Barrier function:
        h(x) = ||p - p_obs||² - (r + margin)²

    CBF 조건 (discrete-time):
        h(x_{t+1}) - (1 - alpha) * h(x_t) >= 0

    시간 할인 비용:
        cost = Σ_t γ^t · weight · max(0, -[h(x_{t+1}) - (1-α)·h(x_t)])

    Args:
        obstacles: List of (x, y, radius) 장애물 정의
        weight: CBF 위반 비용 가중치
        cbf_alpha: Class-K function 파라미터 (0 < alpha <= 1)
        discount_gamma: 시간 할인율 (0 < gamma <= 1)
        safety_margin: 추가 안전 마진 (m)
    """

    def __init__(
        self,
        obstacles: List[tuple],
        weight: float = 100.0,
        cbf_alpha: float = 0.3,
        discount_gamma: float = 0.9,
        safety_margin: float = 0.05,
    ):
        self.obstacles = obstacles
        self.weight = weight
        self.cbf_alpha = cbf_alpha
        self.discount_gamma = discount_gamma
        self.safety_margin = safety_margin

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        시간 할인 CBF 위반 비용 계산 (fully vectorized)

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            costs: (K,) 각 샘플의 CBF 비용
        """
        K, Np1 = trajectories.shape[0], trajectories.shape[1]
        N = Np1 - 1
        costs = np.zeros(K)

        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        # precompute discount factors γ^0, γ^1, ..., γ^{N-1}
        discounts = self.discount_gamma ** np.arange(N)  # (N,)

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin

            # 거리 제곱 (K, N+1)
            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            dist_sq = dx**2 + dy**2

            # Barrier value: h(x) = ||p - p_obs||^2 - r_eff^2
            h = dist_sq - effective_r**2  # (K, N+1)

            # Discrete CBF 조건: h(x_{t+1}) - (1-alpha)*h(x_t) >= 0
            cbf_condition = h[:, 1:] - (1.0 - self.cbf_alpha) * h[:, :-1]  # (K, N)
            violation = np.maximum(0.0, -cbf_condition)  # (K, N)

            # 시간 할인 비용: γ^t · violation
            costs += self.weight * np.sum(violation * discounts[np.newaxis, :], axis=1)

        return costs

    def get_barrier_info(self, trajectories: np.ndarray) -> Dict:
        """
        Barrier 정보 반환 (디버깅/시각화용)

        Args:
            trajectories: (K, N+1, nx) 또는 (N+1, nx) 궤적

        Returns:
            dict:
                - barrier_values: 각 장애물별 barrier 값
                - min_barrier: 전체 최소 barrier 값
                - is_safe: 모든 시간스텝에서 안전 여부
        """
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, :, :]

        positions = trajectories[:, :, :2]

        if len(self.obstacles) == 0:
            return {
                "barrier_values": np.array([]),
                "min_barrier": float('inf'),
                "is_safe": True,
            }

        all_barrier_values = []

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin
            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            dist_sq = dx**2 + dy**2
            h = dist_sq - effective_r**2
            all_barrier_values.append(h)

        # 모든 장애물에 대한 barrier 스택
        barrier_stack = np.array(all_barrier_values)  # (num_obs, K, N+1)
        min_barrier = np.min(barrier_stack)

        return {
            "barrier_values": barrier_stack,
            "min_barrier": float(min_barrier),
            "is_safe": bool(min_barrier > 0),
        }

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.obstacles = obstacles

    def __repr__(self) -> str:
        return (
            f"HorizonWeightedCBFCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"weight={self.weight}, "
            f"alpha={self.cbf_alpha}, "
            f"gamma={self.discount_gamma})"
        )
