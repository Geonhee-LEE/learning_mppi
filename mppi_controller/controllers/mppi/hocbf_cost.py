"""
High-Order CBF (HOCBF, exponential form) 비용 함수 + 안전 필터

상대 차수(relative degree) 2 시스템 — 제어가 가속도 레벨인
5D/6D 동역학 모델 — 에서 위치 barrier 제약을 가능하게 함.

수학 (cbfkit certificates/rectifiers.py 의 exponential cascade 포팅):
    ψ0(x) = h(x)
    ψ1(x) = ψ̇0(x) + λ1·ψ0(x)
    제약:  ψ̇1(x) + λ2·ψ1(x) ≥ 0

이산시간 궤적 (K, N+1, nx) 위에서는 유한 차분 cascade:
    ψ0_t = h(x_t)
    ψ1_t = (ψ0_{t+1} - ψ0_t)/dt + λ1·ψ0_t                  (K, N)
    C_t  = (ψ1_{t+1} - ψ1_t)/dt + λ2·ψ1_t ≥ 0              (K, N-1)

위반 비용:
    cost = weight · Σ_t max(0, -C_t)²    (penalty="squared", 기본)
    cost = weight · Σ_t max(0, -C_t)     (penalty="linear")

relative_degree=1 이면 표준 이산 CBF 로 축약:
    C_t = (ψ0_{t+1} - ψ0_t)/dt + λ1·ψ0_t ≥ 0
        = (1/dt)·[h_{t+1} - (1 - λ1·dt)·h_t]
    → penalty="linear", λ1 = α/dt, weight' = weight·dt 로 두면
      기존 ControlBarrierCost(alpha=α, weight=weight) 와 정확히 일치.

Ref:
    Xiao & Belta (2019), "Control Barrier Functions for Systems with
    High Relative Degree" (exponential CBF)
    cbfkit: src/cbfkit/certificates/rectifiers.py (rectify_relative_degree)
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.models.base_model import RobotModel


# ─────────────────────────────────────────────────────────────
# Relative degree 자동 검출 (cbfkit rectifiers 의 sampling 방식 포팅)
# ─────────────────────────────────────────────────────────────

def detect_relative_degree(
    model: RobotModel,
    h_grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_samples: int = 20,
    tol: float = 1e-8,
    rng: Optional[int] = 0,
) -> int:
    """
    Barrier 의 상대 차수를 샘플링 기반으로 검출 (1 또는 2 반환).

    cbfkit rectifiers 의 방식: 무작위 상태에서 ∂h/∂x·g(x) 의
    총 절대값(제어 권한)을 계산 — 임계값 미만이면 차수 ≥ 2.

    g(x) 는 제어-어파인 모델 가정 하에 제어에 대한 유한 차분으로 계산:
        g_j(x) = [f(x, δ·e_j) - f(x, 0)] / δ    (어파인이면 정확)

    Args:
        model: RobotModel (forward_dynamics 제공)
        h_grad_fn: x → ∂h/∂x (nx,). None 이면 원점 장애물에 대한
            위치 barrier h = ||p||² 의 gradient [2x, 2y, 0, ...] 사용.
        n_samples: 무작위 상태 샘플 수
        tol: 제어 권한 임계값
        rng: 랜덤 시드

    Returns:
        1 (∂h/∂x·g ≠ 0: 표준 CBF 가능) 또는 2 (제어 권한 없음: HOCBF 필요)
    """
    nx = model.state_dim
    nu = model.control_dim
    generator = np.random.default_rng(rng)

    if h_grad_fn is None:
        def h_grad_fn(x: np.ndarray) -> np.ndarray:
            grad = np.zeros(nx)
            grad[0] = 2.0 * x[0]
            grad[1] = 2.0 * x[1]
            return grad

    delta = 1e-3
    total_authority = 0.0

    for _ in range(n_samples):
        x = generator.normal(size=nx)
        u0 = np.zeros(nu)
        f0 = model.forward_dynamics(x, u0)
        grad = h_grad_fn(x)

        for j in range(nu):
            uj = np.zeros(nu)
            uj[j] = delta
            g_col = (model.forward_dynamics(x, uj) - f0) / delta
            total_authority += abs(float(grad @ g_col))

    if not np.isfinite(total_authority):
        raise ValueError(
            "relative degree 검출 중 NaN/Inf 발생 — "
            "dynamics 또는 barrier gradient 수치 안정성 확인 필요"
        )

    return 1 if total_authority > tol else 2


# ─────────────────────────────────────────────────────────────
# HOCBF MPPI 비용
# ─────────────────────────────────────────────────────────────

class HOCBFCost(CostFunction):
    """
    High-Order CBF (exponential form) MPPI 비용 함수

    원형 장애물 barrier:
        h(x) = ||p - p_o||² - (r + margin)²

    이산시간 cascade (모듈 docstring 참조) 위반을 페널티로 부과.

    Args:
        obstacles: List of (x, y, radius) 장애물 정의
        lambda1: 1차 exponential 게인 (> 0)
        lambda2: 2차 exponential 게인 (> 0, relative_degree=2 에서만 사용)
        weight: 위반 비용 가중치
        safety_margin: 추가 안전 마진 (m)
        dt: 유한 차분 타임스텝 (초)
        relative_degree: 1 또는 2. 1 이면 표준 이산 CBF 로 축약.
        penalty: "squared" (기본) 또는 "linear"
        use_hard_rejection: True 이면 h < 0 인 궤적에 rejection_cost 추가
            (hard_cbf_cost 방식의 이진 거부)
        rejection_cost: 하드 거부 비용
    """

    def __init__(
        self,
        obstacles: List[tuple],
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        weight: float = 1000.0,
        safety_margin: float = 0.1,
        dt: float = 0.05,
        relative_degree: int = 2,
        penalty: str = "squared",
        use_hard_rejection: bool = False,
        rejection_cost: float = 1e6,
    ):
        assert lambda1 > 0, f"lambda1 must be > 0, got {lambda1}"
        assert lambda2 > 0, f"lambda2 must be > 0, got {lambda2}"
        assert relative_degree in (1, 2), \
            f"relative_degree must be 1 or 2, got {relative_degree}"
        assert penalty in ("squared", "linear"), \
            f"penalty must be 'squared' or 'linear', got {penalty}"
        assert dt > 0, f"dt must be > 0, got {dt}"

        self.obstacles = obstacles
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.weight = weight
        self.safety_margin = safety_margin
        self.dt = dt
        self.relative_degree = relative_degree
        self.penalty = penalty
        self.use_hard_rejection = use_hard_rejection
        self.rejection_cost = rejection_cost

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        HOCBF 위반 비용 계산 (fully vectorized — K 루프 없음)

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            costs: (K,) 각 샘플의 HOCBF 비용
        """
        K = trajectories.shape[0]
        n_steps = trajectories.shape[1]
        costs = np.zeros(K)

        if not self.obstacles:
            return costs

        positions = trajectories[:, :, :2]  # (K, N+1, 2)
        violated = np.zeros(K, dtype=bool)

        # relative_degree=2 는 최소 3개 시점 필요 → 부족하면 rd=1 로 폴백
        effective_rd = self.relative_degree
        if effective_rd == 2 and n_steps < 3:
            effective_rd = 1

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin

            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            h = dx**2 + dy**2 - effective_r**2  # ψ0: (K, N+1)

            # 1차 cascade: ψ1_t = (ψ0_{t+1} - ψ0_t)/dt + λ1·ψ0_t
            psi1 = (h[:, 1:] - h[:, :-1]) / self.dt + self.lambda1 * h[:, :-1]

            if effective_rd == 1:
                constraint = psi1  # (K, N)
            else:
                # 2차 cascade: C_t = (ψ1_{t+1} - ψ1_t)/dt + λ2·ψ1_t
                constraint = (
                    (psi1[:, 1:] - psi1[:, :-1]) / self.dt
                    + self.lambda2 * psi1[:, :-1]
                )  # (K, N-1)

            violation = np.maximum(0.0, -constraint)
            if self.penalty == "squared":
                violation = violation**2

            costs += self.weight * np.sum(violation, axis=1)

            if self.use_hard_rejection:
                violated |= np.any(h < 0, axis=1)

        if self.use_hard_rejection:
            costs[violated] += self.rejection_cost

        return costs

    def get_barrier_info(self, trajectories: np.ndarray) -> Dict:
        """Barrier 정보 반환 (디버깅/시각화용)"""
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, :, :]

        positions = trajectories[:, :, :2]

        if len(self.obstacles) == 0:
            return {
                "barrier_values": np.array([]),
                "min_barrier": float("inf"),
                "is_safe": True,
            }

        all_barrier_values = []
        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin
            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            h = dx**2 + dy**2 - effective_r**2
            all_barrier_values.append(h)

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
            f"HOCBFCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"rd={self.relative_degree}, "
            f"lambda1={self.lambda1}, lambda2={self.lambda2}, "
            f"weight={self.weight})"
        )


# ─────────────────────────────────────────────────────────────
# HOCBF 해석적 안전 필터 (단일 제약, closed-form)
# ─────────────────────────────────────────────────────────────

class HOCBFFilter:
    """
    HOCBF 해석적 안전 필터 (제어-어파인 모델)

    연속시간 exponential cascade:
        ψ1(x) = ∇h(x)·f(x) + λ1·h(x)     (drift 만 — rd=2 에서 ḣ 은 u 무관)
        제약:   ∇ψ1·(f + g·u) + λ2·ψ1 ≥ 0
        → a = ∇ψ1·g(x),  b = ∇ψ1·f(x) + λ2·ψ1(x),  aᵀu + b ≥ 0

    relative_degree=1 이면 표준 CBF 필터:
        a = ∇h·g(x),  b = ∇h·f(x) + λ1·h(x)

    위반 시 closed-form 최소 노름 보정:
        u* = u_nom + max(0, -(b + aᵀu_nom)) · a / (aᵀa + eps)
    이후 모델 제어 제약으로 클리핑.

    다중 장애물: 가장 위반이 큰 제약을 n_passes 회 반복 적용.

    야코비안은 유한 차분으로 계산 (∇ψ1: 상태에 대한 central difference,
    g(x): 제어에 대한 forward difference — 어파인이면 정확).

    Args:
        model: RobotModel
        obstacles: [(x, y, radius), ...]
        lambda1, lambda2: exponential 게인 (> 0)
        safety_margin: 추가 안전 마진 (m)
        relative_degree: 1, 2 또는 None (None 이면 자동 검출)
        n_passes: 다중 장애물 반복 보정 횟수
        eps: aᵀa 정규화 상수
        fd_eps: 상태 유한 차분 스텝
    """

    def __init__(
        self,
        model: RobotModel,
        obstacles: List[tuple],
        lambda1: float = 2.0,
        lambda2: float = 2.0,
        safety_margin: float = 0.1,
        relative_degree: Optional[int] = None,
        n_passes: int = 3,
        eps: float = 1e-8,
        fd_eps: float = 1e-5,
    ):
        assert lambda1 > 0 and lambda2 > 0, "lambda gains must be > 0"

        self.model = model
        self.obstacles = obstacles
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.safety_margin = safety_margin
        self.n_passes = n_passes
        self.eps = eps
        self.fd_eps = fd_eps

        if relative_degree is None:
            relative_degree = detect_relative_degree(model)
        assert relative_degree in (1, 2)
        self.relative_degree = relative_degree

        bounds = model.get_control_bounds()
        if bounds is not None:
            self._u_min, self._u_max = bounds
        else:
            self._u_min, self._u_max = None, None

        self.filter_stats: List[Dict] = []

    # ── 내부 헬퍼 ──────────────────────────────────────────

    def _h(self, x: np.ndarray, obs: tuple) -> float:
        obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
        r_eff = obs_r + self.safety_margin
        return float((x[0] - obs_x) ** 2 + (x[1] - obs_y) ** 2 - r_eff**2)

    def _h_grad(self, x: np.ndarray, obs: tuple) -> np.ndarray:
        grad = np.zeros(self.model.state_dim)
        grad[0] = 2.0 * (x[0] - obs[0])
        grad[1] = 2.0 * (x[1] - obs[1])
        return grad

    def _f(self, x: np.ndarray) -> np.ndarray:
        """Drift f(x) = forward_dynamics(x, 0) (제어-어파인 가정)"""
        return self.model.forward_dynamics(x, np.zeros(self.model.control_dim))

    def _g(self, x: np.ndarray) -> np.ndarray:
        """g(x) 열 계산: (nx, nu). 제어 유한 차분 (어파인이면 정확)"""
        nu = self.model.control_dim
        f0 = self._f(x)
        delta = 1e-3
        g = np.zeros((self.model.state_dim, nu))
        for j in range(nu):
            uj = np.zeros(nu)
            uj[j] = delta
            g[:, j] = (self.model.forward_dynamics(x, uj) - f0) / delta
        return g

    def _psi1(self, x: np.ndarray, obs: tuple) -> float:
        """ψ1(x) = ∇h·f(x) + λ1·h(x)"""
        return float(self._h_grad(x, obs) @ self._f(x)) + self.lambda1 * self._h(x, obs)

    def _psi1_grad(self, x: np.ndarray, obs: tuple) -> np.ndarray:
        """∇ψ1 — 상태 central difference"""
        nx = self.model.state_dim
        grad = np.zeros(nx)
        for i in range(nx):
            e = np.zeros(nx)
            e[i] = self.fd_eps
            grad[i] = (
                self._psi1(x + e, obs) - self._psi1(x - e, obs)
            ) / (2.0 * self.fd_eps)
        return grad

    def _constraint_terms(
        self, x: np.ndarray, obs: tuple
    ) -> Tuple[np.ndarray, float]:
        """제약 aᵀu + b ≥ 0 의 (a, b) 반환"""
        f = self._f(x)
        g = self._g(x)

        if self.relative_degree == 1:
            grad_h = self._h_grad(x, obs)
            a = grad_h @ g
            b = float(grad_h @ f) + self.lambda1 * self._h(x, obs)
        else:
            grad_psi1 = self._psi1_grad(x, obs)
            a = grad_psi1 @ g
            b = float(grad_psi1 @ f) + self.lambda2 * self._psi1(x, obs)

        return a, b

    # ── 공개 API ──────────────────────────────────────────

    def filter_control(
        self,
        state: np.ndarray,
        u_nom: np.ndarray,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        HOCBF 안전 필터 적용

        Args:
            state: (nx,) 현재 상태
            u_nom: (nu,) 명목 제어 (예: MPPI 출력)
            u_min, u_max: 제어 제약 (None 이면 모델 제약 사용)

        Returns:
            u_safe: (nu,) 안전 보정된 제어
            info: dict (filtered, correction_norm, n_corrections, min_constraint)
        """
        lo = u_min if u_min is not None else self._u_min
        hi = u_max if u_max is not None else self._u_max

        u = np.asarray(u_nom, dtype=float).copy()

        if not self.obstacles:
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "n_corrections": 0,
                "min_constraint": float("inf"),
            }
            self.filter_stats.append(info)
            return u, info

        n_corrections = 0
        min_constraint = float("inf")

        for _ in range(self.n_passes):
            # 가장 위반이 큰 제약 탐색
            worst_val = float("inf")
            worst_a, worst_b = None, None
            for obs in self.obstacles:
                a, b = self._constraint_terms(state, obs)
                val = float(a @ u + b)
                if val < worst_val:
                    worst_val = val
                    worst_a, worst_b = a, b

            min_constraint = min(min_constraint, worst_val)
            if worst_val >= 0.0:
                break

            # closed-form 최소 노름 보정
            correction = max(0.0, -(worst_b + float(worst_a @ u)))
            u = u + correction * worst_a / (float(worst_a @ worst_a) + self.eps)

            if lo is not None and hi is not None:
                u = np.clip(u, lo, hi)

            n_corrections += 1

        correction_norm = float(np.linalg.norm(u - np.asarray(u_nom, dtype=float)))
        info = {
            "filtered": n_corrections > 0,
            "correction_norm": correction_norm,
            "n_corrections": n_corrections,
            "min_constraint": float(min_constraint),
        }
        self.filter_stats.append(info)

        return u, info

    def get_filter_statistics(self) -> Dict:
        """필터 누적 통계"""
        if not self.filter_stats:
            return {
                "total_calls": 0,
                "filter_rate": 0.0,
                "mean_correction": 0.0,
            }
        n = len(self.filter_stats)
        filtered = sum(1 for s in self.filter_stats if s["filtered"])
        corrections = [s["correction_norm"] for s in self.filter_stats]
        return {
            "total_calls": n,
            "filter_rate": filtered / n,
            "mean_correction": float(np.mean(corrections)),
        }

    def __repr__(self) -> str:
        return (
            f"HOCBFFilter("
            f"num_obstacles={len(self.obstacles)}, "
            f"rd={self.relative_degree}, "
            f"lambda1={self.lambda1}, lambda2={self.lambda2})"
        )
