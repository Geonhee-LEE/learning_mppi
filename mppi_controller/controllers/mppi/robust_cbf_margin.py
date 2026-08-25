"""
Robust CBF 비용 함수 (bounded disturbance)

유계 외란이 있는 제어-어파인 시스템:
    ẋ = f(x) + g(x)·u + M·w,   ||w||∞ ≤ w_max

에 대한 Robust CBF (Jankovic 2014 스타일, cbfkit robust_cbf_clf_qp 포팅):
CBF 조건을 최악 외란 항으로 강화(tighten):

    ḣ + α(h) - ||∇h·M||·w_max ≥ 0

이산시간 (기존 ControlBarrierCost 의 이산 CBF 조건 재사용):
    h_{t+1} - (1 - α)·h_t - dt·||∇h_t·M||·w_max ≥ 0

(외란이 한 스텝 dt 동안 h 를 최대 ||∇h·M||·w_max·dt 만큼 감소시킬 수
있으므로 그만큼 마진을 예약)

w_max = 0 이면 vanilla ControlBarrierCost 와 정확히 동일.
마진 항은 w_max 에 선형.

노름 선택 (cbfkit robustness_terms.py):
    norm="two": ||∇h·M||₂ · w_max   (||w||₂ ≤ w_max 외란)
    norm="sup": Σ_j |(∇h·M)_j| · w_max  (||w||∞ ≤ w_max 외란, 1-노름 쌍대)

Ref:
    M. Jankovic, "Robust control barrier functions for constrained
    stabilization of nonlinear systems" (Automatica 2018; 초기 버전 2014)
    cbfkit: generate_constraints/robust_cbfs.py, utils/robustness_terms.py
"""

import numpy as np
from typing import List, Optional

from mppi_controller.controllers.mppi.cost_functions import CostFunction


class RobustCBFCost(CostFunction):
    """
    Robust CBF 비용 함수 — 유계 외란에 대한 최악 경우 마진

    Barrier:
        h(x) = ||p - p_o||² - (r + margin)²

    강화된 이산 CBF 조건:
        C_t = h_{t+1} - (1 - α)·h_t - dt·||∇h_t·M||·w_max ≥ 0

    위반 비용:
        cost = weight · Σ_t max(0, -C_t)

    Args:
        obstacles: List of (x, y, radius) 장애물 정의
        w_max: 외란 노름 상한 (≥ 0). 0 이면 vanilla CBF 와 동일.
        M: 외란 입력 행렬. 기본 None → 위치 블록 항등 I₂
            (외란이 위치 상태에 직접 작용).
            (2, m) 행렬 또는 (nx, m) 행렬 허용 — 위치 barrier 는
            상위 2행(위치 행)만 h 에 영향을 주므로 그 부분만 사용.
        alpha: 이산 class-K 파라미터 (0 < α ≤ 1)
        weight: 위반 비용 가중치
        safety_margin: 추가 안전 마진 (m)
        dt: 타임스텝 (초)
        norm: "two" (2-노름 외란) 또는 "sup" (∞-노름 외란)
    """

    def __init__(
        self,
        obstacles: List[tuple],
        w_max: float = 0.0,
        M: Optional[np.ndarray] = None,
        alpha: float = 0.1,
        weight: float = 1000.0,
        safety_margin: float = 0.1,
        dt: float = 0.05,
        norm: str = "two",
    ):
        assert w_max >= 0, f"w_max must be >= 0, got {w_max}"
        assert 0 < alpha <= 1, f"alpha must be in (0, 1], got {alpha}"
        assert dt > 0, f"dt must be > 0, got {dt}"
        assert norm in ("two", "sup"), f"norm must be 'two' or 'sup', got {norm}"

        self.obstacles = obstacles
        self.w_max = w_max
        self.alpha = alpha
        self.weight = weight
        self.safety_margin = safety_margin
        self.dt = dt
        self.norm = norm

        if M is None:
            M = np.eye(2)
        M = np.atleast_2d(np.asarray(M, dtype=float))
        # 위치 barrier 는 M 의 위치 행(상위 2행)만 h 에 영향
        self.M_pos = M[:2, :]

    def _robust_term(self, grad_pos: np.ndarray) -> np.ndarray:
        """
        최악 외란 항 ||∇h·M||·w_max 계산 (배치)

        Args:
            grad_pos: (..., 2) 위치 gradient ∇h_pos = [2dx, 2dy]

        Returns:
            (...,) 최악 외란에 의한 ḣ 감소율 상한
        """
        gM = grad_pos @ self.M_pos  # (..., m)
        if self.norm == "two":
            rob = np.linalg.norm(gM, axis=-1)
        else:  # sup-norm 외란 → 1-노름 쌍대
            rob = np.sum(np.abs(gM), axis=-1)
        return rob * self.w_max

    def get_robust_margin(self, trajectories: np.ndarray) -> np.ndarray:
        """
        각 장애물/스텝별 마진 항 dt·||∇h_t·M||·w_max 반환 (테스트/플롯용)

        Args:
            trajectories: (K, N+1, nx) 궤적

        Returns:
            margins: (num_obs, K, N) 마진 항
        """
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, :, :]

        positions = trajectories[:, :-1, :2]  # (K, N, 2) — 조건은 t 스텝 기준
        margins = []
        for obs_x, obs_y, _obs_r in self.obstacles:
            grad_pos = 2.0 * (positions - np.array([obs_x, obs_y]))  # (K, N, 2)
            margins.append(self.dt * self._robust_term(grad_pos))
        return np.array(margins)  # (num_obs, K, N)

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Robust CBF 위반 비용 계산 (fully vectorized)

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

            # ∇h 위치 성분 (t = 0..N-1 스텝에서 평가)
            grad_pos = np.stack(
                [2.0 * dx[:, :-1], 2.0 * dy[:, :-1]], axis=-1
            )  # (K, N, 2)
            robust_margin = self.dt * self._robust_term(grad_pos)  # (K, N)

            # 강화된 이산 CBF 조건
            condition = (
                h[:, 1:] - (1.0 - self.alpha) * h[:, :-1] - robust_margin
            )  # (K, N)

            violation = np.maximum(0.0, -condition)
            costs += self.weight * np.sum(violation, axis=1)

        return costs

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.obstacles = obstacles

    def __repr__(self) -> str:
        return (
            f"RobustCBFCost("
            f"num_obstacles={len(self.obstacles)}, "
            f"w_max={self.w_max}, alpha={self.alpha}, "
            f"norm={self.norm}, weight={self.weight})"
        )
