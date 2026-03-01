"""
CBF Safety Filter (Approach B)

QP 기반 안전 필터로 MPPI 출력을 사후 보정.
최소한의 제어 수정으로 CBF 안전 제약 만족.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize


class CBFSafetyFilter:
    """
    QP 기반 CBF 안전 필터

    MPPI가 출력한 제어를 최소한으로 수정하여 CBF 제약을 만족시킴.

    최적화 문제:
        min  ||u - u_mppi||^2
        s.t. Lf*h + Lg*h*u + alpha*h >= 0  (각 장애물)
             u_min <= u <= u_max

    Differential drive 기구학:
        x_dot = v*cos(theta)
        y_dot = v*sin(theta)
        theta_dot = omega

    h(x) = (x - x_obs)^2 + (y - y_obs)^2 - r_eff^2

    Lie derivatives:
        Lf*h = 0  (kinematic, no drift)
        Lg*h = [2(x-xo)*cos(theta) + 2(y-yo)*sin(theta), 0]

    Args:
        obstacles: List of (x, y, radius) 장애물 정의
        cbf_alpha: Class-K function 파라미터
        safety_margin: 추가 안전 마진 (m)
    """

    def __init__(
        self,
        obstacles: List[tuple],
        cbf_alpha: float = 0.1,
        safety_margin: float = 0.1,
    ):
        self.obstacles = obstacles
        self.cbf_alpha = cbf_alpha
        self.safety_margin = safety_margin
        self.filter_stats = []

    def filter_control(
        self,
        state: np.ndarray,
        u_mppi: np.ndarray,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        안전 필터 적용

        Args:
            state: (3,) [x, y, theta] 현재 상태
            u_mppi: (2,) [v, omega] MPPI 출력 제어
            u_min: (2,) 제어 하한
            u_max: (2,) 제어 상한

        Returns:
            u_safe: (2,) 안전 보정된 제어
            info: dict - 필터 정보
        """
        x, y, theta = state[0], state[1], state[2]

        # 모든 장애물에 대해 제약 조건 계산
        constraints = []
        barrier_values = []

        for obs_x, obs_y, obs_r in self.obstacles:
            Lf_h, Lg_h, h = self._compute_lie_derivatives(
                state, obs_x, obs_y, obs_r
            )
            barrier_values.append(h)

            # CBF 제약: Lf_h + Lg_h @ u + alpha * h >= 0
            # → Lg_h @ u >= -(Lf_h + alpha * h)
            def cbf_constraint(u, Lf_h=Lf_h, Lg_h=Lg_h, h=h):
                return Lf_h + Lg_h @ u + self.cbf_alpha * h

            constraints.append({
                "type": "ineq",
                "fun": cbf_constraint,
            })

        # 이미 안전한지 확인 (빠른 경로)
        all_safe = True
        for con in constraints:
            if con["fun"](u_mppi) < 0:
                all_safe = False
                break

        if all_safe:
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "barrier_values": barrier_values,
                "min_barrier": min(barrier_values) if barrier_values else float("inf"),
            }
            self.filter_stats.append(info)
            return u_mppi.copy(), info

        # QP 풀기: min ||u - u_mppi||^2
        def objective(u):
            diff = u - u_mppi
            return 0.5 * np.dot(diff, diff)

        def objective_jac(u):
            return u - u_mppi

        # 제어 제약 (bounds)
        bounds = None
        if u_min is not None and u_max is not None:
            bounds = list(zip(u_min, u_max))

        result = minimize(
            objective,
            x0=u_mppi.copy(),
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-8},
        )

        if result.success:
            u_safe = result.x
        else:
            # 최적화 실패 시: 제약 만족 여부 확인
            u_candidate = result.x
            constraints_satisfied = all(
                c["fun"](u_candidate) >= -1e-6 for c in constraints
            )
            if constraints_satisfied:
                u_safe = u_candidate
            else:
                # 안전하지 않은 결과 → 정지 제어 (v=0, ω=0)
                u_safe = np.zeros_like(u_mppi)

        correction_norm = np.linalg.norm(u_safe - u_mppi)

        info = {
            "filtered": True,
            "correction_norm": float(correction_norm),
            "barrier_values": barrier_values,
            "min_barrier": min(barrier_values) if barrier_values else float("inf"),
            "optimization_success": result.success,
        }
        self.filter_stats.append(info)

        return u_safe, info

    def _compute_lie_derivatives(
        self,
        state: np.ndarray,
        obs_x: float,
        obs_y: float,
        obs_r: float,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Lie derivative 계산

        h(x) = (x-xo)^2 + (y-yo)^2 - r_eff^2

        dh/dx = [2(x-xo), 2(y-yo), 0]

        f(x) = [0, 0, 0] (drift, kinematic이므로 0)
        g(x) = [[cos(theta), 0],
                 [sin(theta), 0],
                 [0, 1]]

        Lf_h = dh/dx @ f(x) = 0
        Lg_h = dh/dx @ g(x) = [2(x-xo)*cos(theta) + 2(y-yo)*sin(theta), 0]

        Args:
            state: (3,) [x, y, theta]
            obs_x, obs_y, obs_r: 장애물 위치/반경

        Returns:
            Lf_h: float (= 0 for kinematic)
            Lg_h: (2,) Lie derivative w.r.t. control
            h: float, barrier value
        """
        x, y, theta = state[0], state[1], state[2]
        effective_r = obs_r + self.safety_margin

        # Barrier value
        h = (x - obs_x) ** 2 + (y - obs_y) ** 2 - effective_r ** 2

        # Lie derivatives
        Lf_h = 0.0  # kinematic, no drift

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        Lg_h = np.array([
            2.0 * (x - obs_x) * cos_theta + 2.0 * (y - obs_y) * sin_theta,
            0.0,
        ])

        return Lf_h, Lg_h, h

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.obstacles = obstacles

    def get_filter_statistics(self) -> Dict:
        """필터 통계 반환"""
        if not self.filter_stats:
            return {
                "num_filtered": 0,
                "mean_correction_norm": 0.0,
                "filter_rate": 0.0,
            }

        num_filtered = sum(1 for s in self.filter_stats if s["filtered"])
        correction_norms = [s["correction_norm"] for s in self.filter_stats]

        return {
            "num_filtered": num_filtered,
            "total_steps": len(self.filter_stats),
            "filter_rate": num_filtered / len(self.filter_stats),
            "mean_correction_norm": float(np.mean(correction_norms)),
            "max_correction_norm": float(np.max(correction_norms)),
        }

    def reset(self):
        """통계 초기화"""
        self.filter_stats = []

    def __repr__(self) -> str:
        return (
            f"CBFSafetyFilter("
            f"num_obstacles={len(self.obstacles)}, "
            f"alpha={self.cbf_alpha})"
        )
