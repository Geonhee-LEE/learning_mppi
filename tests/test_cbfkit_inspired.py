"""
cbfkit 영감 안전 모듈 테스트

- HOCBF (High-Order CBF, exponential form): hocbf_cost.py
- Stochastic CBF (Itô correction) + Risk-Aware path-integral CBF: stochastic_cbf.py
- Robust CBF (bounded disturbance margin): robust_cbf_margin.py
"""

import numpy as np
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost
from mppi_controller.controllers.mppi.hocbf_cost import (
    HOCBFCost,
    HOCBFFilter,
    detect_relative_degree,
)
from mppi_controller.controllers.mppi.stochastic_cbf import (
    StochasticCBFCost,
    RiskAwareCBFCost,
)
from mppi_controller.controllers.mppi.robust_cbf_margin import RobustCBFCost
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.sampling import GaussianSampler
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)


# ─────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────

def random_trajectories(K=32, N=20, nx=3, span=2.5, seed=0):
    """장애물 주변 무작위 궤적 (일부는 위반)"""
    rng = np.random.default_rng(seed)
    traj = np.zeros((K, N + 1, nx))
    traj[:, :, :2] = rng.uniform(-span, span, size=(K, N + 1, 2))
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, nx))
    return traj, ctrl, ref


def straight_through_trajectory(K=8, N=20, nx=3, y=0.0):
    """장애물(원점)을 관통하는 직선 궤적"""
    traj = np.zeros((K, N + 1, nx))
    traj[:, :, 0] = np.linspace(-2.0, 2.0, N + 1)
    traj[:, :, 1] = y
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, nx))
    return traj, ctrl, ref


def stationary_trajectory(K=4, N=20, nx=3, x=0.0, y=0.0):
    """정지 궤적"""
    traj = np.zeros((K, N + 1, nx))
    traj[:, :, 0] = x
    traj[:, :, 1] = y
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N + 1, nx))
    return traj, ctrl, ref


# ══════════════════════════════════════════════════
# 1. HOCBF — relative degree 검출
# ══════════════════════════════════════════════════

class TestDetectRelativeDegree:
    def test_kinematic_is_degree_1(self):
        """DifferentialDriveKinematic: 위치 barrier 는 상대 차수 1"""
        model = DifferentialDriveKinematic()
        assert detect_relative_degree(model) == 1

    def test_dynamic_is_degree_2(self):
        """DifferentialDriveDynamic: 제어(가속도)가 위치에 직접 작용 안 함 → 2"""
        model = DifferentialDriveDynamic()
        assert detect_relative_degree(model) == 2

    def test_custom_grad_velocity_barrier_degree_1(self):
        """동역학 모델이라도 속도에 의존하는 barrier 는 차수 1"""
        model = DifferentialDriveDynamic()

        def h_grad_fn(x):
            grad = np.zeros(model.state_dim)
            grad[3] = 2.0 * x[3]  # h = v² 형태
            return grad

        assert detect_relative_degree(model, h_grad_fn=h_grad_fn) == 1


# ══════════════════════════════════════════════════
# 2. HOCBF — 비용 함수
# ══════════════════════════════════════════════════

class TestHOCBFCost:
    def test_rd1_linear_reduces_to_standard_cbf(self):
        """rd=1 + linear penalty = 기존 ControlBarrierCost 와 정확히 동일"""
        obstacles = [(0.5, 0.3, 0.4), (-1.0, 1.0, 0.3)]
        alpha_d, w, dt, margin = 0.1, 500.0, 0.05, 0.1

        hocbf = HOCBFCost(
            obstacles,
            lambda1=alpha_d / dt,  # λ1·dt = α
            weight=w * dt,
            safety_margin=margin,
            dt=dt,
            relative_degree=1,
            penalty="linear",
        )
        vanilla = ControlBarrierCost(
            obstacles, cbf_alpha=alpha_d, cbf_weight=w, safety_margin=margin
        )

        traj, ctrl, ref = random_trajectories(seed=42)
        c_hocbf = hocbf.compute_cost(traj, ctrl, ref)
        c_vanilla = vanilla.compute_cost(traj, ctrl, ref)

        assert np.max(c_vanilla) > 0, "테스트 궤적에 위반이 있어야 함"
        np.testing.assert_allclose(c_hocbf, c_vanilla, rtol=1e-10)

    def test_no_obstacles_zero_cost(self):
        cost_fn = HOCBFCost([], relative_degree=2)
        traj, ctrl, ref = random_trajectories()
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (traj.shape[0],)
        assert np.allclose(costs, 0.0)

    def test_zero_cost_far_from_obstacles(self):
        """장애물에서 먼 정지 궤적: cascade 전 항이 양수 → 비용 0"""
        obstacles = [(0.0, 0.0, 0.3)]
        for rd in (1, 2):
            cost_fn = HOCBFCost(obstacles, relative_degree=rd)
            traj, ctrl, ref = stationary_trajectory(x=5.0, y=5.0)
            costs = cost_fn.compute_cost(traj, ctrl, ref)
            assert np.allclose(costs, 0.0), f"rd={rd} 에서 비용이 0이어야 함"

    def test_positive_cost_when_violating(self):
        """위반 궤적: 비용 > 0 (rd=1: 관통, rd=2: 내부 정지)"""
        obstacles = [(0.0, 0.0, 0.4)]

        # rd=1: 관통 궤적 → h < 0 구간에서 Δh/dt + λ1·h < 0
        cost1 = HOCBFCost(obstacles, relative_degree=1)
        traj, ctrl, ref = straight_through_trajectory()
        assert np.all(cost1.compute_cost(traj, ctrl, ref) > 0)

        # rd=2: 장애물 내부 정지 → ψ1 = λ1·h < 0, C = λ2·λ1·h < 0
        # (참고: 빠른 관통 궤적은 ψ1 상승률이 커서 cascade 를 만족할 수
        #  있음 — HOCBF 는 초기조건 ψ_i(0) ≥ 0 하에서만 h ≥ 0 을 보장.
        #  관통 자체의 이진 배제는 use_hard_rejection 으로 처리)
        cost2 = HOCBFCost(obstacles, relative_degree=2)
        traj2, ctrl2, ref2 = stationary_trajectory(x=0.0, y=0.0)
        assert np.all(cost2.compute_cost(traj2, ctrl2, ref2) > 0)

    def test_batch_shape_and_finite(self):
        """(K, N+1, nx) → (K,) 형상, 유한값"""
        obstacles = [(0.0, 0.0, 0.4), (1.0, 1.0, 0.2)]
        for nx in (3, 5):
            cost_fn = HOCBFCost(obstacles, relative_degree=2)
            traj, ctrl, ref = random_trajectories(K=64, N=15, nx=nx, seed=7)
            costs = cost_fn.compute_cost(traj, ctrl, ref)
            assert costs.shape == (64,)
            assert np.all(np.isfinite(costs))
            assert np.all(costs >= 0)

    def test_hard_rejection(self):
        """use_hard_rejection: h < 0 궤적에 rejection_cost 추가"""
        obstacles = [(0.0, 0.0, 0.4)]
        cost_fn = HOCBFCost(
            obstacles, relative_degree=2,
            use_hard_rejection=True, rejection_cost=1e6,
        )

        traj_bad, ctrl, ref = straight_through_trajectory(K=4)
        traj_safe, _, _ = stationary_trajectory(K=4, x=5.0, y=5.0)

        costs_bad = cost_fn.compute_cost(traj_bad, ctrl, ref)
        costs_safe = cost_fn.compute_cost(traj_safe, ctrl, ref)

        assert np.all(costs_bad >= 1e6)
        assert np.allclose(costs_safe, 0.0)

    def test_squared_penalty_default(self):
        """기본 penalty=squared: linear 대비 위반 크기에 초선형"""
        obstacles = [(0.0, 0.0, 0.4)]
        sq = HOCBFCost(obstacles, relative_degree=1, penalty="squared", weight=1.0)
        lin = HOCBFCost(obstacles, relative_degree=1, penalty="linear", weight=1.0)
        traj, ctrl, ref = straight_through_trajectory()
        c_sq = sq.compute_cost(traj, ctrl, ref)
        c_lin = lin.compute_cost(traj, ctrl, ref)
        assert np.all(c_sq > 0) and np.all(c_lin > 0)
        assert not np.allclose(c_sq, c_lin)


# ══════════════════════════════════════════════════
# 3. HOCBF — 안전 필터
# ══════════════════════════════════════════════════

class TestHOCBFFilter:
    def test_safe_control_unchanged(self):
        """장애물에서 먼 상태: u_nom 그대로 통과"""
        model = DifferentialDriveKinematic()
        filt = HOCBFFilter(model, obstacles=[(0.0, 0.0, 0.3)])
        assert filt.relative_degree == 1  # 자동 검출

        state = np.array([5.0, 5.0, 0.0])
        u_nom = np.array([0.5, 0.1])
        u_safe, info = filt.filter_control(state, u_nom)

        assert not info["filtered"]
        np.testing.assert_allclose(u_safe, u_nom)

    def test_corrects_unsafe_control(self):
        """동역학 모델, 장애물로 고속 접근: 제약값 개선 방향으로 보정"""
        model = DifferentialDriveDynamic()
        obstacles = [(1.2, 0.0, 0.3)]
        filt = HOCBFFilter(model, obstacles, lambda1=2.0, lambda2=2.0)
        assert filt.relative_degree == 2  # 자동 검출

        # 장애물 방향으로 v=1.5 로 이동 중, 최대 가속 명령
        state = np.array([0.0, 0.0, 0.0, 1.5, 0.0])
        u_nom = np.array([2.0, 0.0])

        a, b = filt._constraint_terms(state, obstacles[0])
        val_before = float(a @ u_nom + b)
        assert val_before < 0, "테스트 셋업: u_nom 이 제약을 위반해야 함"

        u_safe, info = filt.filter_control(state, u_nom)
        val_after = float(a @ u_safe + b)

        assert info["filtered"]
        assert info["correction_norm"] > 0
        assert val_after > val_before, "보정 후 제약값이 개선되어야 함"
        assert u_safe[0] < u_nom[0], "장애물 접근 → 감속 방향 보정"

    def test_respects_control_bounds(self):
        """보정 후 제어가 모델 제약 내에 있어야 함"""
        model = DifferentialDriveDynamic(a_max=2.0, alpha_max=2.0)
        obstacles = [(1.0, 0.0, 0.3)]
        filt = HOCBFFilter(model, obstacles)

        # 매우 가까이 + 고속 → 큰 보정 필요 → 클리핑
        state = np.array([0.55, 0.0, 0.0, 2.0, 0.0])
        u_nom = np.array([2.0, 0.0])
        u_safe, _ = filt.filter_control(state, u_nom)

        lo, hi = model.get_control_bounds()
        assert np.all(u_safe >= lo - 1e-9)
        assert np.all(u_safe <= hi + 1e-9)

    def test_multiple_obstacles_iterative(self):
        """다중 장애물: 가장 위반이 큰 제약부터 반복 보정"""
        model = DifferentialDriveDynamic()
        obstacles = [(1.2, 0.3, 0.3), (1.2, -0.3, 0.3), (5.0, 5.0, 0.3)]
        filt = HOCBFFilter(model, obstacles, n_passes=3)

        state = np.array([0.0, 0.0, 0.0, 1.5, 0.0])
        u_nom = np.array([2.0, 0.0])
        u_safe, info = filt.filter_control(state, u_nom)

        assert np.all(np.isfinite(u_safe))
        assert info["n_corrections"] >= 1
        stats = filt.get_filter_statistics()
        assert stats["total_calls"] == 1

    def test_no_obstacles_pass_through(self):
        model = DifferentialDriveKinematic()
        filt = HOCBFFilter(model, obstacles=[])
        u_nom = np.array([0.5, 0.2])
        u_safe, info = filt.filter_control(np.array([0.0, 0.0, 0.0]), u_nom)
        assert not info["filtered"]
        np.testing.assert_allclose(u_safe, u_nom)


# ══════════════════════════════════════════════════
# 4. Stochastic CBF (Itô correction)
# ══════════════════════════════════════════════════

class TestStochasticCBFCost:
    def test_ito_term_exact_analytic(self):
        """Itô 항 = Σ_i σ_pos_i² = 0.5·Tr[σᵀ(2I)σ] (해석적)"""
        sigma = np.array([0.3, 0.4, 0.2])  # 위치 [0.3, 0.4] 만 사용
        cost_fn = StochasticCBFCost([(0.0, 0.0, 0.5)], sigma_process=sigma)

        expected = 0.3**2 + 0.4**2  # = 0.25
        assert cost_fn.get_ito_correction() == pytest.approx(expected)

        # 일반 trace 공식으로 수치 검증: 0.5·Tr[σ_posᵀ·(2I)·σ_pos]
        sigma_pos = np.diag(sigma[:2])
        hess = 2.0 * np.eye(2)
        trace_val = 0.5 * np.trace(sigma_pos.T @ hess @ sigma_pos)
        assert cost_fn.get_ito_correction() == pytest.approx(trace_val)

    def test_sigma_zero_reduces_to_vanilla(self):
        """σ=0, β=0: vanilla ControlBarrierCost 와 정확히 동일 (스케일 환산)"""
        obstacles = [(0.5, 0.3, 0.4), (-1.0, 1.0, 0.3)]
        a, w, dt, margin = 2.0, 500.0, 0.05, 0.1

        sto = StochasticCBFCost(
            obstacles, alpha=a, beta=0.0, sigma_process=None,
            weight=w * dt, safety_margin=margin, dt=dt,
        )
        vanilla = ControlBarrierCost(
            obstacles, cbf_alpha=a * dt, cbf_weight=w, safety_margin=margin
        )

        traj, ctrl, ref = random_trajectories(seed=3)
        c_sto = sto.compute_cost(traj, ctrl, ref)
        c_van = vanilla.compute_cost(traj, ctrl, ref)

        assert np.max(c_van) > 0
        np.testing.assert_allclose(c_sto, c_van, rtol=1e-10)

    def test_ito_shifts_cost_by_exact_amount(self):
        """깊은 위반 궤적에서 비용 차이 = weight·N·ito (해석적 검증)"""
        obstacles = [(0.0, 0.0, 1.0)]
        N, w, alpha = 20, 100.0, 1.0
        sigma = np.array([0.3, 0.4])
        ito = 0.25

        base = StochasticCBFCost(
            obstacles, alpha=alpha, sigma_process=None,
            weight=w, safety_margin=0.0,
        )
        noisy = StochasticCBFCost(
            obstacles, alpha=alpha, sigma_process=sigma,
            weight=w, safety_margin=0.0,
        )

        # 장애물 중심 정지: h = -1, condition = α·h + ito (모든 스텝 위반)
        traj, ctrl, ref = stationary_trajectory(K=4, N=N, x=0.0, y=0.0)
        c_base = base.compute_cost(traj, ctrl, ref)
        c_noisy = noisy.compute_cost(traj, ctrl, ref)

        np.testing.assert_allclose(c_base - c_noisy, w * N * ito, rtol=1e-10)

    def test_beta_buffer_more_conservative(self):
        """buffer β 증가 → 비용 증가 (경계 부근)"""
        obstacles = [(0.0, 0.0, 0.5)]
        # 경계 살짝 바깥 정지 궤적: h 작은 양수, condition = α·h
        traj, ctrl, ref = stationary_trajectory(x=0.65, y=0.0)

        c0 = StochasticCBFCost(
            obstacles, alpha=1.0, beta=0.0, safety_margin=0.1
        ).compute_cost(traj, ctrl, ref)
        c1 = StochasticCBFCost(
            obstacles, alpha=1.0, beta=2.0, safety_margin=0.1
        ).compute_cost(traj, ctrl, ref)

        assert np.allclose(c0, 0.0)
        assert np.all(c1 > 0), "β buffer 가 경계 부근 비용을 증가시켜야 함"

    def test_higher_sigma_relaxes_convex_barrier(self):
        """볼록 원형 barrier: Itô 항이 양수 → σ 증가 시 비용 감소
        (cbfkit 수학 그대로 — 노이즈 보수성은 β 또는 RiskAwareCBFCost 로)"""
        obstacles = [(0.0, 0.0, 0.5)]
        traj, ctrl, ref = straight_through_trajectory()

        c0 = StochasticCBFCost(obstacles, alpha=1.0).compute_cost(traj, ctrl, ref)
        c1 = StochasticCBFCost(
            obstacles, alpha=1.0, sigma_process=np.array([0.5, 0.5])
        ).compute_cost(traj, ctrl, ref)

        assert np.all(c1 <= c0 + 1e-12)

    def test_batch_shape_and_no_obstacles(self):
        cost_fn = StochasticCBFCost([], sigma_process=np.array([0.1, 0.1]))
        traj, ctrl, ref = random_trajectories(K=64, N=15, nx=5)
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (64,)
        assert np.allclose(costs, 0.0)


# ══════════════════════════════════════════════════
# 5. Risk-Aware path-integral CBF
# ══════════════════════════════════════════════════

class TestRiskAwareCBFCost:
    def _make(self, rho=0.05, sigma=0.1):
        return RiskAwareCBFCost(
            [(0.0, 0.0, 0.5)],
            rho=rho,
            sigma_process=np.array([sigma, sigma]),
            safety_margin=0.1,
            dt=0.05,
        )

    def test_margin_monotone_in_t(self):
        """margin(t) 는 t 에 대해 단조 증가"""
        cost_fn = self._make(rho=0.05)
        ts = np.linspace(0.0, 3.0, 30)
        margins = cost_fn.get_margin(ts)
        assert np.all(np.diff(margins) >= 0)
        assert cost_fn.get_margin(0.0) == pytest.approx(0.0)

    def test_margin_decreasing_in_rho(self):
        """위험 예산 ρ 증가 → margin 감소"""
        t = 1.5
        margins = [self._make(rho=r).get_margin(t) for r in (0.01, 0.05, 0.2, 0.4)]
        assert all(m1 > m2 for m1, m2 in zip(margins, margins[1:]))

    def test_rho_half_zero_margin(self):
        """ρ=0.5 → erfinv(0)=0 → margin=0"""
        cost_fn = self._make(rho=0.5)
        assert cost_fn.get_margin(2.0) == pytest.approx(0.0, abs=1e-12)

    def test_margin_formula_exact(self):
        """margin(t) = sqrt(2t)·η·erfinv(1-2ρ) 공식 검증"""
        from scipy.special import erfinv
        cost_fn = self._make(rho=0.05)
        eta, t = 0.7, 2.0
        expected = np.sqrt(2.0 * t) * eta * erfinv(1.0 - 2.0 * 0.05)
        assert cost_fn.get_margin(t, eta=eta) == pytest.approx(expected)

    def test_more_conservative_than_vanilla_near_boundary(self):
        """경계를 스치는 궤적: vanilla 비용 0, risk-aware 비용 > 0"""
        obstacles = [(0.0, 0.0, 0.5)]
        # h = 0.7² - 0.6² = 0.13 > 0 (안전하지만 경계 근처)
        traj, ctrl, ref = stationary_trajectory(x=0.7, y=0.0, N=30)

        vanilla = ControlBarrierCost(
            obstacles, cbf_alpha=0.1, cbf_weight=1000.0, safety_margin=0.1
        )
        ra = RiskAwareCBFCost(
            obstacles, rho=0.05, sigma_process=np.array([0.2, 0.2]),
            safety_margin=0.1, dt=0.05,
        )

        c_van = vanilla.compute_cost(traj, ctrl, ref)
        c_ra = ra.compute_cost(traj, ctrl, ref)

        assert np.allclose(c_van, 0.0)
        assert np.all(c_ra > 0), "risk-aware 는 경계 부근에서 더 보수적이어야 함"

    def test_higher_sigma_more_conservative(self):
        """σ 증가 → margin 증가 → 비용 증가 (경계 부근)"""
        traj, ctrl, ref = stationary_trajectory(x=0.75, y=0.0, N=30)
        costs = [
            np.sum(self._make(rho=0.05, sigma=s).compute_cost(traj, ctrl, ref))
            for s in (0.0, 0.1, 0.3)
        ]
        assert costs[0] == pytest.approx(0.0)
        assert costs[1] < costs[2]

    def test_zero_cost_far_and_batch_shape(self):
        """장애물에서 충분히 먼 궤적: 비용 0, 형상 (K,)"""
        cost_fn = self._make(rho=0.1, sigma=0.05)
        traj, ctrl, ref = stationary_trajectory(K=16, x=5.0, y=5.0)
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (16,)
        assert np.allclose(costs, 0.0)


# ══════════════════════════════════════════════════
# 6. Robust CBF (bounded disturbance)
# ══════════════════════════════════════════════════

class TestRobustCBFCost:
    def test_wmax_zero_reduces_to_vanilla(self):
        """w_max=0: vanilla ControlBarrierCost 와 정확히 동일"""
        obstacles = [(0.5, 0.3, 0.4), (-1.0, 1.0, 0.3)]
        alpha, w, margin = 0.1, 500.0, 0.1

        robust = RobustCBFCost(
            obstacles, w_max=0.0, alpha=alpha, weight=w, safety_margin=margin
        )
        vanilla = ControlBarrierCost(
            obstacles, cbf_alpha=alpha, cbf_weight=w, safety_margin=margin
        )

        traj, ctrl, ref = random_trajectories(seed=11)
        c_rob = robust.compute_cost(traj, ctrl, ref)
        c_van = vanilla.compute_cost(traj, ctrl, ref)

        assert np.max(c_van) > 0
        np.testing.assert_allclose(c_rob, c_van, rtol=1e-10)

    def test_margin_linear_in_wmax(self):
        """마진 항이 w_max 에 정확히 선형"""
        obstacles = [(0.0, 0.0, 0.4)]
        traj, _, _ = random_trajectories(seed=5)

        m1 = RobustCBFCost(obstacles, w_max=1.0).get_robust_margin(traj)
        m2 = RobustCBFCost(obstacles, w_max=2.0).get_robust_margin(traj)
        m0 = RobustCBFCost(obstacles, w_max=0.0).get_robust_margin(traj)

        assert np.max(m1) > 0
        np.testing.assert_allclose(m2, 2.0 * m1, rtol=1e-12)
        np.testing.assert_allclose(m0, 0.0)

    def test_more_conservative_with_wmax(self):
        """w_max 증가 → 비용 단조 증가 (경계 부근)"""
        obstacles = [(0.0, 0.0, 0.5)]
        traj, ctrl, ref = straight_through_trajectory(y=0.65)

        costs = [
            np.sum(
                RobustCBFCost(
                    obstacles, w_max=w, alpha=0.1, safety_margin=0.1
                ).compute_cost(traj, ctrl, ref)
            )
            for w in (0.0, 0.5, 2.0)
        ]
        assert costs[0] <= costs[1] <= costs[2]
        assert costs[2] > costs[0], "w_max 증가가 비용을 증가시켜야 함"

    def test_sup_norm_at_least_two_norm(self):
        """1-노름 쌍대(sup 외란) ≥ 2-노름 마진"""
        obstacles = [(0.0, 0.0, 0.4)]
        traj, _, _ = random_trajectories(seed=9)

        m_two = RobustCBFCost(obstacles, w_max=1.0, norm="two").get_robust_margin(traj)
        m_sup = RobustCBFCost(obstacles, w_max=1.0, norm="sup").get_robust_margin(traj)
        assert np.all(m_sup >= m_two - 1e-12)

    def test_batch_shape_and_no_obstacles(self):
        cost_fn = RobustCBFCost([], w_max=1.0)
        traj, ctrl, ref = random_trajectories(K=64, N=15, nx=5)
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (64,)
        assert np.allclose(costs, 0.0)

    def test_custom_disturbance_matrix(self):
        """M 이 x 방향 외란만 허용하면 y 방향 gradient 는 마진에 미기여"""
        obstacles = [(0.0, 0.0, 0.4)]
        M = np.array([[1.0], [0.0]])  # 외란이 x 에만 작용
        cost_fn = RobustCBFCost(obstacles, w_max=1.0, M=M)

        # 궤적이 장애물 정북(y 축 위): ∇h = [0, 2dy] → ∇h·M = 0
        traj, _, _ = stationary_trajectory(x=0.0, y=1.0)
        margin = cost_fn.get_robust_margin(traj)
        np.testing.assert_allclose(margin, 0.0, atol=1e-12)


# ══════════════════════════════════════════════════
# 7. 통합 smoke 테스트 (MPPI + 각 신규 비용)
# ══════════════════════════════════════════════════

# 장애물: 원형 레퍼런스 (r=3) 경로 위, 시작점(3,0)에서 약간 앞
OBS_ANGLE = 0.15
INTEGRATION_OBSTACLE = (
    3.0 * np.cos(OBS_ANGLE),
    3.0 * np.sin(OBS_ANGLE),
    0.15,
)


def _make_safety_cost(name, model, dt):
    obstacles = [INTEGRATION_OBSTACLE]
    kwargs = dict(weight=3000.0, safety_margin=0.15, dt=dt)
    if name == "hocbf":
        rd = detect_relative_degree(model)
        return HOCBFCost(obstacles, lambda1=2.0, lambda2=2.0,
                         relative_degree=rd, **kwargs)
    if name == "stochastic":
        return StochasticCBFCost(
            obstacles, alpha=2.0, beta=0.5,
            sigma_process=np.array([0.05, 0.05]), **kwargs,
        )
    if name == "risk_aware":
        return RiskAwareCBFCost(
            obstacles, rho=0.05,
            sigma_process=np.array([0.05, 0.05]), **kwargs,
        )
    if name == "robust":
        return RobustCBFCost(obstacles, w_max=0.1, alpha=0.15, **kwargs)
    raise ValueError(name)


@pytest.mark.parametrize("model_name", ["kinematic", "dynamic"])
@pytest.mark.parametrize("cost_name", ["hocbf", "stochastic", "risk_aware", "robust"])
def test_integration_smoke(model_name, cost_name):
    """MPPI + Composite(tracking + 신규 안전 비용): 충돌 없음 + 유한 RMSE"""
    dt, N, K, num_steps = 0.05, 15, 128, 35

    if model_name == "kinematic":
        model = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0)
        nx = 3
        Q = np.array([10.0, 10.0, 1.0])
        state = np.array([3.0, 0.0, np.pi / 2])
    else:
        model = DifferentialDriveDynamic()
        nx = 5
        Q = np.array([10.0, 10.0, 1.0, 0.0, 0.0])
        state = np.array([3.0, 0.0, np.pi / 2, 0.3, 0.1])

    def traj_fn(t):
        pt3 = circle_trajectory(t, radius=3.0)  # (3,) [x, y, θ]
        if nx == 3:
            return pt3
        return np.concatenate([pt3, np.zeros(2)])  # v, ω 참조는 미사용 (Q=0)

    safety_cost = _make_safety_cost(cost_name, model, dt)
    cost = CompositeMPPICost([
        StateTrackingCost(Q),
        TerminalCost(Q),
        ControlEffortCost(np.array([0.1, 0.1])),
        safety_cost,
    ])

    params = MPPIParams(
        K=K, N=N, dt=dt, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=Q, R=np.array([0.1, 0.1]),
    )
    ctrl = MPPIController(
        model, params, cost_function=cost,
        noise_sampler=GaussianSampler(np.array([0.5, 0.5]), seed=123),
    )

    obs_x, obs_y, obs_r = INTEGRATION_OBSTACLE
    errors = []
    min_dist = float("inf")

    for step in range(num_steps):
        t = step * dt
        ref = generate_reference_trajectory(traj_fn, t, N, dt)
        control, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(control))

        state = state + model.forward_dynamics(state, control) * dt

        dist = np.hypot(state[0] - obs_x, state[1] - obs_y)
        min_dist = min(min_dist, dist)

        ref_pt = circle_trajectory(t, radius=3.0)
        errors.append(np.hypot(state[0] - ref_pt[0], state[1] - ref_pt[1]))

    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))

    assert min_dist > obs_r, (
        f"[{model_name}/{cost_name}] 충돌: min_dist={min_dist:.3f} <= r={obs_r}"
    )
    assert np.isfinite(rmse), f"[{model_name}/{cost_name}] RMSE 비유한"
    assert rmse < 2.0, f"[{model_name}/{cost_name}] RMSE={rmse:.3f} 과대"
