"""
Optimal-Decay CBF Safety Filter

표준 CBF의 고정 decay rate(α)를 최적화 변수(ω)로 확장.
장애물 밀집 환경에서 표준 CBF가 infeasible일 때도 해를 보장.

최적화 문제:
    min  ||u - u_mppi||² + p_sb · (ω - 1)²
    s.t. Lf·h + Lg·h·u + α·ω·h ≥ 0  (각 장애물)
         u_min ≤ u ≤ u_max
         0 ≤ ω ≤ 1

ω = 1: 표준 CBF와 동일 (full safety)
ω < 1: CBF 제약 완화 (graceful degradation)
ω = 0: 안전 제약 해제 (최후의 수단)

p_sb (penalty weight): ω가 1에서 벗어날수록 큰 페널티
    → 가능한 한 표준 CBF를 유지, 불가능할 때만 완화

Ref: Zeng et al. (2021)
     "Safety-Critical Model Predictive Control with Discrete-Time
      Control Barrier Function"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
from mppi_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter


class OptimalDecayCBFSafetyFilter(CBFSafetyFilter):
    """
    Optimal-Decay CBF Safety Filter

    CBFSafetyFilter를 확장하여 decay rate ω를 최적화 변수에 포함.
    표준 CBF가 infeasible한 상황에서도 guaranteed feasibility 제공.

    Args:
        obstacles: [(x, y, radius), ...] 장애물 리스트
        cbf_alpha: Class-K function 파라미터 (base decay rate)
        safety_margin: 추가 안전 마진 (m)
        penalty_weight: ω 이탈 페널티 가중치 (p_sb, 기본 1e4)
        omega_min: ω 하한 (기본 0.0)
        omega_max: ω 상한 (기본 1.0)
    """

    def __init__(
        self,
        obstacles: List[tuple],
        cbf_alpha: float = 0.1,
        safety_margin: float = 0.1,
        penalty_weight: float = 1e4,
        omega_min: float = 0.0,
        omega_max: float = 1.0,
    ):
        super().__init__(obstacles, cbf_alpha, safety_margin)
        assert 0.0 <= omega_min <= omega_max <= 1.0, \
            f"omega bounds must satisfy 0 <= omega_min <= omega_max <= 1, got [{omega_min}, {omega_max}]"
        self.penalty_weight = penalty_weight
        self.omega_min = omega_min
        self.omega_max = omega_max

    def filter_control(
        self,
        state: np.ndarray,
        u_mppi: np.ndarray,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimal-Decay CBF 안전 필터 적용

        결정 변수: z = [v, omega_ctrl, ω]  (nu + 1)

        Args:
            state: (3,) [x, y, theta]
            u_mppi: (2,) [v, omega] MPPI 출력 제어
            u_min: (2,) 제어 하한
            u_max: (2,) 제어 상한

        Returns:
            u_safe: (2,) 안전 보정된 제어
            info: dict - 필터 정보 (optimal_omega 포함)
        """
        if not self.obstacles:
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "barrier_values": [],
                "min_barrier": float("inf"),
                "optimal_omega": 1.0,
            }
            self.filter_stats.append(info)
            return u_mppi.copy(), info

        # Lie derivative 사전 계산
        lie_data = []
        barrier_values = []
        for obs in self.obstacles:
            obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
            Lf_h, Lg_h, h = self._compute_lie_derivatives(
                state, obs_x, obs_y, obs_r
            )
            lie_data.append((Lf_h, Lg_h, h))
            barrier_values.append(h)

        # 표준 CBF로 먼저 검사 (ω=1)
        all_safe = True
        for Lf_h, Lg_h, h in lie_data:
            cbf_val = Lf_h + Lg_h @ u_mppi + self.cbf_alpha * h
            if cbf_val < 0:
                all_safe = False
                break

        if all_safe:
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "barrier_values": barrier_values,
                "min_barrier": min(barrier_values) if barrier_values else float("inf"),
                "optimal_omega": 1.0,
            }
            self.filter_stats.append(info)
            return u_mppi.copy(), info

        # 결정 변수: z = [u (nu), ω (1)]
        nu = len(u_mppi)
        z0 = np.concatenate([u_mppi.copy(), [1.0]])  # 초기: ω=1

        # 목적 함수: ||u - u_mppi||² + p_sb·(ω - 1)²
        def objective(z):
            u = z[:nu]
            omega_decay = z[nu]
            control_cost = 0.5 * np.dot(u - u_mppi, u - u_mppi)
            decay_penalty = 0.5 * self.penalty_weight * (omega_decay - 1.0) ** 2
            return control_cost + decay_penalty

        def objective_jac(z):
            u = z[:nu]
            omega_decay = z[nu]
            grad = np.zeros(nu + 1)
            grad[:nu] = u - u_mppi
            grad[nu] = self.penalty_weight * (omega_decay - 1.0)
            return grad

        # CBF 제약: Lf_h + Lg_h·u + α·ω·h ≥ 0 (각 장애물)
        constraints = []
        for Lf_h, Lg_h, h in lie_data:
            def make_constraint(Lf_h_=Lf_h, Lg_h_=Lg_h, h_=h):
                def con_fun(z):
                    u = z[:nu]
                    omega_decay = z[nu]
                    return Lf_h_ + Lg_h_ @ u + self.cbf_alpha * omega_decay * h_
                return con_fun

            constraints.append({
                "type": "ineq",
                "fun": make_constraint(),
            })

        # Bounds: [u_min..u_max, omega_min..omega_max]
        bounds = []
        if u_min is not None and u_max is not None:
            for i in range(nu):
                bounds.append((u_min[i], u_max[i]))
        else:
            for i in range(nu):
                bounds.append((None, None))
        bounds.append((self.omega_min, self.omega_max))

        result = minimize(
            objective,
            x0=z0,
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        u_safe = result.x[:nu]
        optimal_omega = float(result.x[nu])

        correction_norm = np.linalg.norm(u_safe - u_mppi)

        info = {
            "filtered": True,
            "correction_norm": float(correction_norm),
            "barrier_values": barrier_values,
            "min_barrier": min(barrier_values) if barrier_values else float("inf"),
            "optimization_success": result.success,
            "optimal_omega": optimal_omega,
            "decay_relaxed": optimal_omega < 0.99,
        }
        self.filter_stats.append(info)

        return u_safe, info

    def get_filter_statistics(self) -> Dict:
        """필터 통계 (ω 통계 포함)"""
        base_stats = super().get_filter_statistics()

        if not self.filter_stats:
            base_stats["mean_omega"] = 1.0
            base_stats["min_omega"] = 1.0
            base_stats["relaxation_rate"] = 0.0
            return base_stats

        omegas = [s.get("optimal_omega", 1.0) for s in self.filter_stats]
        relaxed = sum(1 for s in self.filter_stats if s.get("decay_relaxed", False))

        base_stats["mean_omega"] = float(np.mean(omegas))
        base_stats["min_omega"] = float(np.min(omegas))
        base_stats["relaxation_rate"] = relaxed / len(self.filter_stats)

        return base_stats

    def __repr__(self) -> str:
        return (
            f"OptimalDecayCBFSafetyFilter("
            f"num_obstacles={len(self.obstacles)}, "
            f"alpha={self.cbf_alpha}, "
            f"p_sb={self.penalty_weight})"
        )
