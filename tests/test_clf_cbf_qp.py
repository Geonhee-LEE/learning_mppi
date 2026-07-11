"""
CLF-CBF-QP 컨트롤러 테스트 (cbfkit-inspired native numpy port)

- CBFCLFQPSolver: 해석적 fast path / SLSQP 폴백 / slack / infeasible 처리
- CLFCBFQPController: 기구학(3D) / 동역학(5D) 수렴 + 충돌 회피
- CBFOnlyQPController: CBF-only 베이스라인
- 인터페이스 / Simulator 통합 / solve time
"""

import numpy as np
import pytest

from mppi_controller.controllers.mppi.clf_cbf_qp import (
    CBFCLFQPSolver,
    CBFOnlyQPController,
    CLFCBFQPController,
    CLFCBFQPParams,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.simulation import Simulator


# ============================================================================
#  헬퍼
# ============================================================================

REQUIRED_INFO_KEYS = {
    "clf_value",
    "min_barrier",
    "qp_feasible",
    "delta",
    "active_constraints",
    "solve_time",
}


def make_goal_reference(goal_xy, n=31, nx=3):
    """정지 목표 레퍼런스 (N+1, nx)"""
    ref = np.zeros((n, nx))
    ref[:, 0] = goal_xy[0]
    ref[:, 1] = goal_xy[1]
    return ref


def rollout(model, controller, x0, ref, steps, dt=0.05):
    """수동 rollout: (states, infos)"""
    x = np.asarray(x0, dtype=float).copy()
    states, infos = [x.copy()], []
    for _ in range(steps):
        u, info = controller.compute_control(x, ref)
        x = model.normalize_state(model.step(x, u, dt))
        states.append(x.copy())
        infos.append(info)
    return np.array(states), infos


def min_center_clearance(states, obstacles):
    """로봇 중심 - 장애물 표면 최소 거리"""
    best = np.inf
    for ox, oy, orad in obstacles:
        d = np.linalg.norm(states[:, :2] - np.array([ox, oy]), axis=1) - orad
        best = min(best, float(d.min()))
    return best


# ============================================================================
#  1. Solver 테스트
# ============================================================================


class TestCBFCLFQPSolver:
    def test_unconstrained_returns_unom(self):
        """제약 없음 → u_nom 그대로 반환"""
        solver = CBFCLFQPSolver()
        u_nom = np.array([0.7, -0.3])
        u, feasible, info = solver.solve(u_nom)
        assert feasible
        np.testing.assert_allclose(u, u_nom, atol=1e-10)
        assert info["delta"] == 0.0
        assert "analytic" in info["method"]

    def test_bounds_clip_only(self):
        """제약 없음 + 경계 초과 u_nom → 클리핑된 값 (feasible)"""
        solver = CBFCLFQPSolver()
        u, feasible, _ = solver.solve(
            np.array([2.0, -3.0]),
            u_min=np.array([-1.0, -1.0]),
            u_max=np.array([1.0, 1.0]),
        )
        assert feasible
        np.testing.assert_allclose(u, [1.0, -1.0], atol=1e-9)

    def test_inactive_cbf_returns_unom(self):
        """CBF가 이미 만족되면 u_nom 유지"""
        solver = CBFCLFQPSolver()
        u_nom = np.array([0.5, 0.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([-1.0])  # 0.5 >= -1.0 만족
        u, feasible, info = solver.solve(u_nom, A_cbf=A, b_cbf=b)
        assert feasible
        np.testing.assert_allclose(u, u_nom, atol=1e-10)
        assert info["max_cbf_violation"] == 0.0

    def test_single_cbf_analytic_matches_slsqp(self):
        """단일 활성 CBF: 해석적 투영 == SLSQP 해"""
        u_nom = np.array([1.0, 0.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.5])  # A@u_nom = 1.0 < 1.5 위반

        solver_fast = CBFCLFQPSolver(use_analytic=True)
        u_fast, feas_fast, info_fast = solver_fast.solve(u_nom, A_cbf=A, b_cbf=b)

        solver_slsqp = CBFCLFQPSolver(use_analytic=False)
        u_slow, feas_slow, info_slow = solver_slsqp.solve(u_nom, A_cbf=A, b_cbf=b)

        assert feas_fast and feas_slow
        assert "analytic" in info_fast["method"]
        assert "slsqp" in info_slow["method"]
        np.testing.assert_allclose(u_fast, u_slow, atol=1e-5)
        # 닫힌형 검증: u = u_nom + g(c - g·u_nom)/|g|² = [1.25, 0.25]
        np.testing.assert_allclose(u_fast, [1.25, 0.25], atol=1e-8)
        assert A @ u_fast >= b - 1e-8

    def test_single_cbf_with_weighted_P(self):
        """비단위 P 행렬에서도 해석적 == SLSQP"""
        P = np.diag([2.0, 0.5])
        u_nom = np.array([0.0, 0.0])
        A = np.array([[1.0, 2.0]])
        b = np.array([1.0])

        u_fast, f1, _ = CBFCLFQPSolver(P=P, use_analytic=True).solve(
            u_nom, A_cbf=A, b_cbf=b
        )
        u_slow, f2, _ = CBFCLFQPSolver(P=P, use_analytic=False).solve(
            u_nom, A_cbf=A, b_cbf=b
        )
        assert f1 and f2
        np.testing.assert_allclose(u_fast, u_slow, atol=1e-5)

    def test_multiple_active_cbf_slsqp(self):
        """다중 활성 CBF → SLSQP 폴백, 모든 제약 만족"""
        solver = CBFCLFQPSolver()
        u_nom = np.array([-1.0, -1.0])
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.2, 0.3])  # 둘 다 위반
        u, feasible, info = solver.solve(u_nom, A_cbf=A, b_cbf=b)
        assert feasible
        assert np.all(A @ u >= b - 1e-6)
        np.testing.assert_allclose(u, [0.2, 0.3], atol=1e-5)

    def test_infeasible_bounds_best_effort(self):
        """CBF가 경계 밖 요구 → feasible=False + 경계 내 best-effort"""
        solver = CBFCLFQPSolver()
        u_nom = np.array([0.0, 0.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([10.0])  # u[0] >= 10 vs u_max = 1
        u_min, u_max = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        u, feasible, info = solver.solve(
            u_nom, A_cbf=A, b_cbf=b, u_min=u_min, u_max=u_max
        )
        assert not feasible
        assert np.all(u >= u_min - 1e-9) and np.all(u <= u_max + 1e-9)
        assert info["max_cbf_violation"] > 0.0

    def test_clf_slack_activates_when_conflicting_with_cbf(self):
        """CLF-CBF 충돌 → 안전(CBF) 우선, δ > 0"""
        solver = CBFCLFQPSolver(lambda_clf=100.0)
        u_nom = np.array([0.0, 0.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([0.5])  # CBF: u[0] >= 0.5
        a_clf = np.array([1.0, 0.0])
        b_clf = -0.5  # CLF: u[0] <= -0.5 + δ → 충돌
        u, feasible, info = solver.solve(
            u_nom, A_cbf=A, b_cbf=b, a_clf=a_clf, b_clf=b_clf
        )
        assert feasible
        assert u[0] >= 0.5 - 1e-6  # 안전 승리
        assert info["delta"] >= 1.0 - 1e-4  # δ ≥ u[0] - b_clf = 1.0

    def test_clf_soft_tradeoff_analytic_matches_slsqp(self):
        """CBF 없음 + CLF 활성: 닫힌형 tradeoff == SLSQP"""
        u_nom = np.array([1.0, 0.5])
        a_clf = np.array([1.0, 0.0])
        b_clf = 0.0  # u[0] <= δ → u_nom 위반
        lam = 10.0
        u_fast, f1, i1 = CBFCLFQPSolver(lambda_clf=lam, use_analytic=True).solve(
            u_nom, a_clf=a_clf, b_clf=b_clf
        )
        u_slow, f2, i2 = CBFCLFQPSolver(lambda_clf=lam, use_analytic=False).solve(
            u_nom, a_clf=a_clf, b_clf=b_clf
        )
        assert f1 and f2
        np.testing.assert_allclose(u_fast, u_slow, atol=1e-4)
        np.testing.assert_allclose(i1["delta"], i2["delta"], atol=1e-4)
        # 닫힌형: s* = s/(1+λq) = 1/11
        assert i1["delta"] == pytest.approx(1.0 / 11.0, abs=1e-6)

    def test_active_constraints_reported(self):
        """활성 제약 인덱스 보고"""
        solver = CBFCLFQPSolver()
        u_nom = np.array([-1.0, 0.0])
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.5, -10.0])  # 첫 번째만 활성
        u, feasible, info = solver.solve(u_nom, A_cbf=A, b_cbf=b)
        assert feasible
        assert 0 in info["active_constraints"]
        assert 1 not in info["active_constraints"]


# ============================================================================
#  2. 기구학 컨트롤러 테스트
# ============================================================================


class TestKinematicController:
    def setup_method(self):
        self.model = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0)
        self.params = CLFCBFQPParams(dt=0.05)

    def test_interface_shapes_and_info_keys(self):
        """(2,) 제어 + 필수 info 키"""
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=[(0.0, 0.5, 0.3)])
        ref = make_goal_reference([1.0, 0.0])
        u, info = ctrl.compute_control(np.array([-1.0, 0.0, 0.0]), ref)
        assert u.shape == (2,)
        assert np.all(np.isfinite(u))
        assert REQUIRED_INFO_KEYS.issubset(info.keys())
        assert isinstance(info["qp_feasible"], bool)

    def test_converges_to_goal_clf_decreases(self):
        """정지 목표 수렴: 거리 감소 + CLF 값 감소"""
        ctrl = CLFCBFQPController(self.model, self.params)
        goal = np.array([1.5, 0.5])
        ref = make_goal_reference(goal)
        x0 = np.array([-1.5, 0.0, 0.0])
        states, infos = rollout(self.model, ctrl, x0, ref, steps=200)

        d0 = np.linalg.norm(x0[:2] - goal)
        d_final = np.linalg.norm(states[-1, :2] - goal)
        # near-identity 오프셋: 로봇 중심은 목표에서 lookahead_d 이내로 수렴
        tol = self.params.lookahead_d + 0.05
        assert d_final < tol, f"목표 미도달: 최종 거리 {d_final:.3f} (허용 {tol:.2f})"
        assert d_final < 0.1 * d0
        # CLF 값 감소 (초반 10스텝 평균 vs 마지막 10스텝 평균)
        V = [i["clf_value"] for i in infos]
        assert np.mean(V[-10:]) < 0.05 * np.mean(V[:10])

    def test_obstacle_between_start_and_goal_no_collision(self):
        """경로 중앙 장애물: 200스텝 무충돌 + 진행"""
        obstacles = [(0.0, 0.2, 0.35)]
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=obstacles)
        goal = np.array([1.5, 0.0])
        ref = make_goal_reference(goal)
        x0 = np.array([-1.5, 0.0, 0.0])
        states, infos = rollout(self.model, ctrl, x0, ref, steps=200)

        # 무충돌: 로봇 중심이 장애물 표면 밖
        clearance = min_center_clearance(states, obstacles)
        assert clearance > 0.0, f"충돌 발생: clearance={clearance:.3f}"
        # QP 배리어 (인플레이션 포함) 항상 양수
        min_h = min(i["min_barrier"] for i in infos)
        assert min_h > 0.0, f"배리어 위반: min h={min_h:.4f}"
        # 진행: 목표에 접근
        d_final = np.linalg.norm(states[-1, :2] - goal)
        assert d_final < 0.5, f"진행 실패: 최종 거리 {d_final:.3f}"

    def test_multiple_obstacles_barrier_never_violated(self):
        """다중 장애물: min_barrier > 0 유지"""
        obstacles = [(-0.5, 0.3, 0.25), (0.5, -0.3, 0.25), (0.0, 0.8, 0.3)]
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=obstacles)
        ref = make_goal_reference([1.5, 0.3])
        states, infos = rollout(
            self.model, ctrl, np.array([-1.5, 0.0, 0.0]), ref, steps=200
        )
        min_h = min(i["min_barrier"] for i in infos)
        assert min_h > 0.0
        assert min_center_clearance(states, obstacles) > 0.0

    def test_control_bounds_respected(self):
        """제어 경계 준수"""
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=[(0.0, 0.1, 0.4)])
        ref = make_goal_reference([2.0, 0.0])
        x = np.array([-2.0, 0.0, 0.0])
        for _ in range(100):
            u, _ = ctrl.compute_control(x, ref)
            assert -1.0 - 1e-9 <= u[0] <= 1.0 + 1e-9
            assert -2.0 - 1e-9 <= u[1] <= 2.0 + 1e-9
            x = self.model.normalize_state(self.model.step(x, u, 0.05))

    def test_safety_wins_delta_activates_near_obstacle(self):
        """장애물이 목표 위에 있을 때: CBF 유지 + δ 활성 이력 존재"""
        # 목표가 장애물 안전 반경 내 → CLF는 위반될 수밖에 없음
        obstacles = [(1.0, 0.0, 0.4)]
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=obstacles)
        ref = make_goal_reference([1.0, 0.0])  # 장애물 중심이 목표
        states, infos = rollout(
            self.model, ctrl, np.array([-1.0, 0.05, 0.0]), ref, steps=150
        )
        # 안전은 항상 유지
        assert min_center_clearance(states, obstacles) > 0.0
        # 접근 후 CLF slack이 활성화된 스텝 존재
        assert any(i["delta"] > 1e-6 for i in infos)


# ============================================================================
#  3. 동역학 (5D) 컨트롤러 테스트
# ============================================================================


class TestDynamicController:
    def setup_method(self):
        self.model = DifferentialDriveDynamic(
            a_max=2.0, alpha_max=2.0, v_max=1.0, omega_max=2.0
        )
        self.params = CLFCBFQPParams(dt=0.05, alpha_cbf=1.5, lambda_hocbf=2.0)

    def test_dynamic_interface(self):
        """5D 상태 → (2,) 제어 + info 키"""
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=[(0.0, 0.3, 0.3)])
        ref = make_goal_reference([1.0, 0.0], nx=5)
        u, info = ctrl.compute_control(np.array([-1.0, 0.0, 0.0, 0.0, 0.0]), ref)
        assert u.shape == (2,)
        assert np.all(np.isfinite(u))
        assert REQUIRED_INFO_KEYS.issubset(info.keys())

    def test_dynamic_no_collision(self):
        """5D 모델 장애물 회피: 무충돌"""
        obstacles = [(0.0, 0.2, 0.3)]
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=obstacles)
        ref = make_goal_reference([1.5, 0.0], nx=5)
        x0 = np.array([-1.5, 0.0, 0.0, 0.0, 0.0])
        states, infos = rollout(self.model, ctrl, x0, ref, steps=300)

        assert min_center_clearance(states, obstacles) > 0.0
        min_h = min(i["min_barrier"] for i in infos)
        assert min_h > 0.0

    def test_dynamic_makes_progress(self):
        """5D 모델: 장애물 우회 후 목표 접근"""
        obstacles = [(0.0, 0.2, 0.3)]
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=obstacles)
        goal = np.array([1.5, 0.0])
        ref = make_goal_reference(goal, nx=5)
        x0 = np.array([-1.5, 0.0, 0.0, 0.0, 0.0])
        states, _ = rollout(self.model, ctrl, x0, ref, steps=300)
        d0 = np.linalg.norm(x0[:2] - goal)
        d_final = np.linalg.norm(states[-1, :2] - goal)
        assert d_final < 0.5 * d0, f"진행 실패: {d0:.2f} → {d_final:.2f}"

    def test_dynamic_bounded_controls(self):
        """가속도 경계 준수"""
        ctrl = CLFCBFQPController(self.model, self.params, obstacles=[(0.0, 0.1, 0.3)])
        ref = make_goal_reference([1.5, 0.0], nx=5)
        x = np.array([-1.5, 0.0, 0.0, 0.5, 0.0])
        for _ in range(150):
            u, _ = ctrl.compute_control(x, ref)
            assert -2.0 - 1e-9 <= u[0] <= 2.0 + 1e-9
            assert -2.0 - 1e-9 <= u[1] <= 2.0 + 1e-9
            x = self.model.normalize_state(self.model.step(x, u, 0.05))


# ============================================================================
#  4. CBF-Only 컨트롤러 테스트
# ============================================================================


class TestCBFOnlyController:
    def test_no_clf_delta_zero(self):
        """CBF-only: δ == 0, clf_value == 0"""
        model = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0)
        ctrl = CBFOnlyQPController(model, CLFCBFQPParams(), obstacles=[(0.0, 0.3, 0.3)])
        ref = make_goal_reference([1.0, 0.0])
        u, info = ctrl.compute_control(np.array([-1.0, 0.0, 0.0]), ref)
        assert u.shape == (2,)
        assert info["delta"] == 0.0
        assert info["clf_value"] == 0.0

    def test_cbf_only_avoids_obstacle(self):
        """CBF-only도 무충돌 + 진행"""
        model = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0)
        obstacles = [(0.0, 0.2, 0.35)]
        ctrl = CBFOnlyQPController(model, CLFCBFQPParams(dt=0.05), obstacles=obstacles)
        goal = np.array([1.5, 0.0])
        ref = make_goal_reference(goal)
        states, infos = rollout(model, ctrl, np.array([-1.5, 0.0, 0.0]), ref, steps=200)
        assert min_center_clearance(states, obstacles) > 0.0
        assert min(i["min_barrier"] for i in infos) > 0.0
        d_final = np.linalg.norm(states[-1, :2] - goal)
        assert d_final < 0.5


# ============================================================================
#  5. Simulator 통합 + 성능
# ============================================================================


class TestIntegration:
    def test_simulator_50_steps(self):
        """mppi_controller.simulation.Simulator 내에서 50스텝 동작"""
        model = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0)
        ctrl = CLFCBFQPController(
            model, CLFCBFQPParams(dt=0.05), obstacles=[(0.0, 0.3, 0.3)]
        )
        sim = Simulator(model, ctrl, dt=0.05)
        sim.reset(np.array([-1.5, 0.0, 0.0]))
        ref = make_goal_reference([1.5, 0.0])
        for _ in range(50):
            step_info = sim.step(ref)
            assert np.all(np.isfinite(step_info["control"]))
        assert len(sim.history["state"]) == 50
        assert np.all(np.isfinite(np.array(sim.history["state"])))

    def test_solve_time_below_10ms(self):
        """평균 solve time ≪ 10ms (장애물 3개 시나리오)"""
        model = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0)
        obstacles = [(-0.5, 0.3, 0.25), (0.5, -0.3, 0.25), (0.0, 0.6, 0.3)]
        ctrl = CLFCBFQPController(model, CLFCBFQPParams(dt=0.05), obstacles=obstacles)
        ref = make_goal_reference([1.5, 0.2])
        x = np.array([-1.5, 0.0, 0.0])
        times = []
        for _ in range(100):
            u, info = ctrl.compute_control(x, ref)
            times.append(info["solve_time"])
            x = model.normalize_state(model.step(x, u, 0.05))
        mean_ms = 1000.0 * float(np.mean(times))
        assert mean_ms < 10.0, f"평균 solve time {mean_ms:.2f}ms >= 10ms"

    def test_reset_and_repr(self):
        """reset() 동작 + repr"""
        model = DifferentialDriveKinematic()
        ctrl = CLFCBFQPController(model, CLFCBFQPParams())
        ref = make_goal_reference([1.0, 0.0])
        ctrl.compute_control(np.array([0.0, 0.0, 0.0]), ref)
        assert ctrl.step_count == 1
        ctrl.reset()
        assert ctrl.step_count == 0
        assert "CLFCBFQPController" in repr(ctrl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
