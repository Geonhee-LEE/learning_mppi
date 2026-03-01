"""
Collision Cone CBF (C3BF) 비용 함수

상대 속도 방향을 고려한 안전 barrier.
기존 거리 기반 CBF와 달리, 장애물에서 멀어지는 방향이면
가까이 있어도 비용을 부과하지 않음.

Ref: Thirugnanam et al. (2024)
     "Safety-Critical Control with Collision Cone Control Barrier Functions"

Barrier function:
    h = <p_rel, v_rel> + ||p_rel|| · ||v_rel|| · cos(φ_safe)
    cos(φ_safe) = sqrt(||p_rel||² - R²) / ||p_rel||

    h > 0 ⟹ 상대 속도 벡터가 충돌 콘 밖 (안전)
    h ≤ 0 ⟹ 충돌 콘 안쪽으로 접근 (위험)
"""

import numpy as np
from typing import List, Tuple, Dict
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class CollisionConeCBFCost(CostFunction):
    """
    Collision Cone CBF (C3BF) 비용 함수

    기존 거리 기반 CBF와 달리 **상대 속도 방향**을 고려.
    장애물과 가까워도 멀어지는 방향이면 비용 0.

    장애물 정보는 (x, y, radius, vx, vy) 5-tuple.
    obstacle_tracker.py가 제공하는 속도 정보를 직접 활용.

    Args:
        obstacles: [(x, y, radius, vx, vy), ...] 동적 장애물
        cbf_alpha: CBF decay rate (0 < α ≤ 1)
        cbf_weight: CBF 위반 비용 가중치
        safety_margin: 추가 안전 마진 (m)
        dt: 시간 간격 (로봇 속도 계산용)
    """

    def __init__(
        self,
        obstacles: List[Tuple[float, ...]] = None,
        cbf_alpha: float = 0.3,
        cbf_weight: float = 1000.0,
        safety_margin: float = 0.1,
        dt: float = 0.05,
    ):
        self.obstacles = obstacles or []
        self.cbf_alpha = cbf_alpha
        self.cbf_weight = cbf_weight
        self.safety_margin = safety_margin
        self.dt = dt

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        C3BF 위반 비용 계산

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적 [x, y, θ, ...]
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스

        Returns:
            costs: (K,)
        """
        K = trajectories.shape[0]
        costs = np.zeros(K)

        if not self.obstacles:
            return costs

        # 로봇 위치: (K, N+1, 2)
        pos = trajectories[:, :, :2]

        # 로봇 속도 (유한 차분): (K, N, 2)
        robot_vel = (pos[:, 1:, :] - pos[:, :-1, :]) / self.dt

        for obs in self.obstacles:
            if len(obs) >= 5:
                ox, oy, obs_r, ovx, ovy = obs[0], obs[1], obs[2], obs[3], obs[4]
            else:
                ox, oy, obs_r = obs[0], obs[1], obs[2]
                ovx, ovy = 0.0, 0.0

            effective_r = obs_r + self.safety_margin

            # 상대 위치: (K, N, 2) — 로봇에서 장애물까지가 아니라
            # 장애물에서 로봇까지 (p_robot - p_obs)
            p_rel_x = pos[:, :-1, 0] - ox  # (K, N)
            p_rel_y = pos[:, :-1, 1] - oy  # (K, N)

            # 상대 속도: v_robot - v_obs (K, N)
            v_rel_x = robot_vel[:, :, 0] - ovx
            v_rel_y = robot_vel[:, :, 1] - ovy

            # ||p_rel||
            dist = np.sqrt(p_rel_x**2 + p_rel_y**2)  # (K, N)
            dist_safe = np.maximum(dist, effective_r + 1e-6)

            # cos(φ_safe) = sqrt(||p_rel||² - R²) / ||p_rel||
            cos_phi = np.sqrt(
                np.maximum(dist_safe**2 - effective_r**2, 0.0)
            ) / dist_safe

            # ||v_rel||
            v_rel_norm = np.sqrt(v_rel_x**2 + v_rel_y**2 + 1e-10)

            # <p_rel, v_rel>
            dot_pv = p_rel_x * v_rel_x + p_rel_y * v_rel_y

            # h = <p_rel, v_rel> + ||p_rel|| · ||v_rel|| · cos(φ)
            h = dot_pv + dist_safe * v_rel_norm * cos_phi  # (K, N)

            # Collision cone 위반: h < 0이면 속도 벡터가 충돌 콘 내부
            cbf_violation = np.maximum(0.0, -h)

            costs += self.cbf_weight * np.sum(cbf_violation, axis=1)

        return costs

    def get_barrier_info(self, trajectory: np.ndarray) -> Dict:
        """
        Barrier 정보 반환 (디버깅/시각화용)

        Args:
            trajectory: (N+1, nx) 또는 (K, N+1, nx)

        Returns:
            dict: barrier_values, min_barrier, is_safe
        """
        if trajectory.ndim == 2:
            trajectory = trajectory[np.newaxis, :, :]

        if not self.obstacles:
            return {
                "barrier_values": np.array([]),
                "min_barrier": float("inf"),
                "is_safe": True,
            }

        pos = trajectory[:, :, :2]
        robot_vel = (pos[:, 1:, :] - pos[:, :-1, :]) / self.dt

        all_h = []
        for obs in self.obstacles:
            if len(obs) >= 5:
                ox, oy, obs_r, ovx, ovy = obs[0], obs[1], obs[2], obs[3], obs[4]
            else:
                ox, oy, obs_r = obs[0], obs[1], obs[2]
                ovx, ovy = 0.0, 0.0

            effective_r = obs_r + self.safety_margin

            p_rel_x = pos[:, :-1, 0] - ox
            p_rel_y = pos[:, :-1, 1] - oy
            v_rel_x = robot_vel[:, :, 0] - ovx
            v_rel_y = robot_vel[:, :, 1] - ovy

            dist = np.sqrt(p_rel_x**2 + p_rel_y**2)
            dist_safe = np.maximum(dist, effective_r + 1e-6)
            cos_phi = np.sqrt(
                np.maximum(dist_safe**2 - effective_r**2, 0.0)
            ) / dist_safe
            v_rel_norm = np.sqrt(v_rel_x**2 + v_rel_y**2 + 1e-10)
            dot_pv = p_rel_x * v_rel_x + p_rel_y * v_rel_y
            h = dot_pv + dist_safe * v_rel_norm * cos_phi

            all_h.append(h)

        barrier_stack = np.array(all_h)
        min_barrier = float(np.min(barrier_stack))

        return {
            "barrier_values": barrier_stack,
            "min_barrier": min_barrier,
            "is_safe": min_barrier > 0,
        }

    def update_obstacles(self, obstacles: List[Tuple[float, ...]]):
        """동적 장애물 업데이트 — (x, y, r, vx, vy) 형태"""
        self.obstacles = obstacles

    def __repr__(self) -> str:
        return (
            f"CollisionConeCBFCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"alpha={self.cbf_alpha}, weight={self.cbf_weight})"
        )
