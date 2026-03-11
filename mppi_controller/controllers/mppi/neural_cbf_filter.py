"""
Neural CBF Safety Filter

학습된 Neural CBF로 QP 기반 안전 필터.
CBFSafetyFilter의 drop-in 대체로, Lie derivative를 autograd로 계산.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from scipy.optimize import minimize

from mppi_controller.learning.neural_cbf_trainer import NeuralCBFTrainer


def _default_g_matrix(state: np.ndarray) -> np.ndarray:
    """
    Differential drive 기본 g(x) 행렬

    x_dot = v*cos(θ), y_dot = v*sin(θ), θ_dot = ω
    g(x) = [[cos(θ), 0], [sin(θ), 0], [0, 1]]

    Args:
        state: (3,) [x, y, theta]

    Returns:
        g: (3, 2) control-affine 행렬
    """
    theta = state[2]
    return np.array([
        [np.cos(theta), 0.0],
        [np.sin(theta), 0.0],
        [0.0, 1.0],
    ])


class NeuralCBFSafetyFilter:
    """
    Neural CBF 기반 QP 안전 필터

    최적화:
        min  ||u - u_mppi||²
        s.t. Lf_h + Lg_h @ u + α·h(x) ≥ 0
             u_min ≤ u ≤ u_max

    Lie derivative를 autograd로 계산 → 임의 형상 장애물 대응.

    Args:
        neural_cbf_trainer: 학습된 NeuralCBFTrainer 인스턴스
        cbf_alpha: Class-K function 파라미터
        safety_margin: 추가 안전 마진
        g_matrix_fn: state → g(x) 행렬 (None이면 diff-drive 기본값)
    """

    def __init__(
        self,
        neural_cbf_trainer: NeuralCBFTrainer,
        cbf_alpha: float = 0.1,
        safety_margin: float = 0.0,
        g_matrix_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.trainer = neural_cbf_trainer
        self.cbf_alpha = cbf_alpha
        self.safety_margin = safety_margin
        self.g_matrix_fn = g_matrix_fn or _default_g_matrix
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
            state: (nx,) 현재 상태
            u_mppi: (nu,) MPPI 출력 제어
            u_min: (nu,) 제어 하한
            u_max: (nu,) 제어 상한

        Returns:
            u_safe: (nu,) 안전 보정된 제어
            info: dict - 필터 정보
        """
        Lf_h, Lg_h, h = self._compute_lie_derivatives_neural(state)

        # CBF 제약: Lf_h + Lg_h @ u + α·(h - margin) ≥ 0
        effective_h = h - self.safety_margin

        def cbf_constraint(u):
            return Lf_h + Lg_h @ u + self.cbf_alpha * effective_h

        # 이미 안전한지 확인
        if cbf_constraint(u_mppi) >= 0:
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "barrier_value": float(h),
                "min_barrier": float(h),
            }
            self.filter_stats.append(info)
            return u_mppi.copy(), info

        # QP: min ||u - u_mppi||²
        def objective(u):
            diff = u - u_mppi
            return 0.5 * np.dot(diff, diff)

        def objective_jac(u):
            return u - u_mppi

        bounds = None
        if u_min is not None and u_max is not None:
            bounds = list(zip(u_min, u_max))

        constraints = [{"type": "ineq", "fun": cbf_constraint}]

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
            u_candidate = result.x
            if cbf_constraint(u_candidate) >= -1e-6:
                u_safe = u_candidate
            else:
                u_safe = np.zeros_like(u_mppi)

        correction_norm = np.linalg.norm(u_safe - u_mppi)

        info = {
            "filtered": True,
            "correction_norm": float(correction_norm),
            "barrier_value": float(h),
            "min_barrier": float(h),
            "optimization_success": result.success,
        }
        self.filter_stats.append(info)

        return u_safe, info

    def _compute_lie_derivatives_neural(
        self, state: np.ndarray
    ) -> Tuple[float, np.ndarray, float]:
        """
        Neural CBF의 Lie derivative 계산

        Lf_h = ∂h/∂x @ f(x) = 0 (kinematic, no drift)
        Lg_h = ∂h/∂x @ g(x)

        Args:
            state: (nx,)

        Returns:
            Lf_h: float
            Lg_h: (nu,)
            h: float (barrier 값)
        """
        h = self.trainer.predict_h(state)
        grad_h = self.trainer.predict_gradient(state)  # (nx,)

        # f(x) = 0 for kinematic model
        Lf_h = 0.0

        # g(x) from provided function
        g_x = self.g_matrix_fn(state)  # (nx, nu)
        Lg_h = grad_h @ g_x  # (nu,)

        return Lf_h, Lg_h, float(h)

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
            f"NeuralCBFSafetyFilter("
            f"alpha={self.cbf_alpha}, "
            f"margin={self.safety_margin})"
        )
