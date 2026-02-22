"""
Hard CBF 비용 함수

이진 거부 CBF — 장애물을 관통하는 궤적을 완전히 배제.
h(x) < 0인 시간스텝이 하나라도 있으면 rejection_cost (1e6) 부과.
safe_control의 hard constraint를 sampling 기반에서 모방.
"""

import numpy as np
from typing import List, Dict
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class HardCBFCost(CostFunction):
    """
    Hard CBF 비용 함수 — 이진 거부

    궤적의 어떤 시간스텝에서든 h(x) < 0 이면 전체 궤적 비용 = rejection_cost.
    softmax 가중치에서 사실상 0으로 만듦.

    Barrier function:
        h(x) = ||p - p_obs||² - (r + margin)²

    비용:
        cost = rejection_cost   if ∃t: h(x_t) < 0
        cost = 0                otherwise

    Args:
        obstacles: List of (x, y, radius) 장애물 정의
        rejection_cost: 위반 궤적에 부과할 비용 (매우 큰 값)
        safety_margin: 추가 안전 마진 (m)
    """

    def __init__(
        self,
        obstacles: List[tuple],
        rejection_cost: float = 1e6,
        safety_margin: float = 0.05,
    ):
        self.obstacles = obstacles
        self.rejection_cost = rejection_cost
        self.safety_margin = safety_margin

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Hard CBF 비용 계산 (fully vectorized)

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            costs: (K,) 각 샘플의 비용 (0 또는 rejection_cost)
        """
        K = trajectories.shape[0]
        costs = np.zeros(K)

        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        # 모든 장애물에 대해 위반 검사 (vectorized)
        violated = np.zeros(K, dtype=bool)

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin

            # 거리 제곱 (K, N+1)
            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            dist_sq = dx**2 + dy**2

            # Barrier value: h(x) = ||p - p_obs||^2 - r_eff^2
            h = dist_sq - effective_r**2  # (K, N+1)

            # 어떤 시간스텝이든 h < 0 이면 위반
            violated |= np.any(h < 0, axis=1)  # (K,)

        costs[violated] = self.rejection_cost
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
            f"HardCBFCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"rejection_cost={self.rejection_cost})"
        )
