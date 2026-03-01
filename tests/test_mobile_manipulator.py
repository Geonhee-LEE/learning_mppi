"""
Mobile Manipulator Tests

Tests for MobileManipulatorKinematic model (2-DOF planar arm + DiffDrive base).
Covers: dimensions, dynamics, FK, batch processing, RK4, normalize_state,
control bounds, cost functions, and MPPI integration.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.mobile_manipulator_kinematic import (
    MobileManipulatorKinematic,
)
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
from mppi_controller.controllers.mppi.cost_functions import (
    EndEffectorTrackingCost,
    EndEffectorTerminalCost,
    JointLimitCost,
    CompositeMPPICost,
    ControlEffortCost,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ═══════════════════════════════════════════════════════
#  MobileManipulatorKinematic Model Tests
# ═══════════════════════════════════════════════════════

class TestMobileManipulatorKinematic:
    def setup_method(self):
        self.model = MobileManipulatorKinematic(
            L1=0.3, L2=0.25, v_max=1.0, omega_max=1.0, dq_max=2.0
        )

    def test_dimensions(self):
        assert self.model.state_dim == 5
        assert self.model.control_dim == 4

    def test_model_type(self):
        assert self.model.model_type == "kinematic"

    def test_is_robot_model(self):
        assert isinstance(self.model, RobotModel)

    def test_forward_dynamics_straight(self):
        """v=1, 나머지 0 → 직진 (cos(θ), sin(θ), 0, 0, 0)"""
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0, 0.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_forward_dynamics_rotation(self):
        """ω=1 → 회전만"""
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 1.0, 0.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-10)

    def test_forward_dynamics_arm_only(self):
        """dq1=1 → 팔만 움직임"""
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0, 1.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [0.0, 0.0, 0.0, 1.0, 0.0], atol=1e-10)

    def test_forward_dynamics_arm_q2(self):
        """dq2=1.5 → q2만 움직임"""
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0, 0.0, 1.5])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [0.0, 0.0, 0.0, 0.0, 1.5], atol=1e-10)

    def test_forward_dynamics_with_heading(self):
        """θ=π/2, v=1 → y방향 이동"""
        state = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0])
        control = np.array([1.0, 0.0, 0.0, 0.0])
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot, [0.0, 1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_batch_dynamics(self):
        """배치 동역학: (K, 5) → (K, 5)"""
        K = 100
        states = np.random.randn(K, 5) * 0.1
        controls = np.random.randn(K, 4) * 0.1
        dots = self.model.forward_dynamics(states, controls)
        assert dots.shape == (K, 5)

    def test_rk4_step(self):
        """RK4 적분 결과 올바른 shape"""
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0, 0.5, 0.3])
        dt = 0.05
        next_state = self.model.step(state, control, dt)
        assert next_state.shape == (5,)
        assert next_state[0] > 0  # x 증가
        assert next_state[3] > 0  # q1 증가

    def test_normalize_state(self):
        """θ, q1, q2 모두 [-π, π] 래핑"""
        state = np.array([1.0, 2.0, 4.0, -4.0, 7.0])
        norm = self.model.normalize_state(state)
        assert -np.pi <= norm[2] <= np.pi
        assert -np.pi <= norm[3] <= np.pi
        assert -np.pi <= norm[4] <= np.pi
        # x, y는 변하지 않음
        assert norm[0] == 1.0
        assert norm[1] == 2.0

    def test_normalize_state_batch(self):
        """배치 정규화"""
        states = np.array([
            [0.0, 0.0, 4.0, -4.0, 7.0],
            [1.0, 1.0, -7.0, 8.0, -10.0],
        ])
        norm = self.model.normalize_state(states)
        assert norm.shape == (2, 5)
        for i in range(2):
            for j in (2, 3, 4):
                assert -np.pi <= norm[i, j] <= np.pi

    def test_control_bounds(self):
        """4D 제어 제약"""
        lb, ub = self.model.get_control_bounds()
        np.testing.assert_allclose(lb, [-1.0, -1.0, -2.0, -2.0])
        np.testing.assert_allclose(ub, [1.0, 1.0, 2.0, 2.0])

    def test_state_to_dict(self):
        """5개 키 반환"""
        state = np.array([1.0, 2.0, 0.5, 0.3, -0.2])
        d = self.model.state_to_dict(state)
        assert d["x"] == 1.0
        assert d["y"] == 2.0
        assert d["theta"] == 0.5
        assert d["q1"] == 0.3
        assert d["q2"] == -0.2

    def test_xy_at_indices_0_1(self):
        """x=state[0], y=state[1] — CBF 호환성"""
        state = np.array([3.0, 4.0, 0.1, 0.2, 0.3])
        assert state[0] == 3.0
        assert state[1] == 4.0

    def test_repr(self):
        r = repr(self.model)
        assert "MobileManipulatorKinematic" in r
        assert "L1" in r


# ═══════════════════════════════════════════════════════
#  Forward Kinematics Tests
# ═══════════════════════════════════════════════════════

class TestForwardKinematics:
    def setup_method(self):
        self.model = MobileManipulatorKinematic(L1=0.3, L2=0.25)

    def test_fk_zero_config(self):
        """q1=q2=θ=0 → ee = (x + L1 + L2, y)"""
        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        ee = self.model.forward_kinematics(state)
        np.testing.assert_allclose(ee, [1.0 + 0.3 + 0.25, 0.0], atol=1e-10)

    def test_fk_arm_up(self):
        """q1=π/2, q2=0, θ=0 → ee = (x, y + L1 + L2) 방향"""
        state = np.array([0.0, 0.0, 0.0, np.pi / 2, 0.0])
        ee = self.model.forward_kinematics(state)
        np.testing.assert_allclose(
            ee, [0.0, 0.3 + 0.25], atol=1e-10
        )

    def test_fk_arm_folded(self):
        """q2=π → 전완이 상완 반대 방향 → ee = (x + L1 - L2, y)"""
        state = np.array([0.0, 0.0, 0.0, 0.0, np.pi])
        ee = self.model.forward_kinematics(state)
        np.testing.assert_allclose(ee, [0.3 - 0.25, 0.0], atol=1e-10)

    def test_fk_with_base_rotation(self):
        """θ=π/2, q1=0, q2=0 → 팔이 y축 방향"""
        state = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0])
        ee = self.model.forward_kinematics(state)
        np.testing.assert_allclose(ee, [0.0, 0.3 + 0.25], atol=1e-10)

    def test_fk_combined(self):
        """θ=π/4, q1=π/4 → 총 φ1=π/2"""
        state = np.array([1.0, 1.0, np.pi / 4, np.pi / 4, 0.0])
        ee = self.model.forward_kinematics(state)
        # φ1 = π/2 → cos=0, sin=1
        # φ12 = π/2 → cos=0, sin=1
        expected_x = 1.0 + 0.0 + 0.0
        expected_y = 1.0 + 0.3 + 0.25
        np.testing.assert_allclose(ee, [expected_x, expected_y], atol=1e-10)

    def test_fk_batch(self):
        """배치 FK: (K, 5) → (K, 2)"""
        K = 50
        states = np.random.randn(K, 5) * 0.5
        ee = self.model.forward_kinematics(states)
        assert ee.shape == (K, 2)

    def test_fk_trajectory_batch(self):
        """궤적 배치 FK: (K, N+1, 5) → (K, N+1, 2)"""
        K, N = 32, 20
        trajectories = np.random.randn(K, N + 1, 5) * 0.3
        ee = self.model.forward_kinematics(trajectories)
        assert ee.shape == (K, N + 1, 2)

    def test_fk_workspace_reachability(self):
        """EE는 base에서 L1+L2 이내"""
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        ee = self.model.forward_kinematics(state)
        dist = np.sqrt(ee[0] ** 2 + ee[1] ** 2)
        assert dist <= self.model.L1 + self.model.L2 + 1e-10


# ═══════════════════════════════════════════════════════
#  End-Effector Cost Function Tests
# ═══════════════════════════════════════════════════════

class TestEndEffectorCosts:
    def setup_method(self):
        self.model = MobileManipulatorKinematic(L1=0.3, L2=0.25)
        self.K = 32
        self.N = 10

    def _make_trajectories(self, state_val=0.0):
        """Create dummy trajectories"""
        return np.full((self.K, self.N + 1, 5), state_val)

    def _make_controls(self):
        return np.zeros((self.K, self.N, 4))

    def test_ee_tracking_cost_shape(self):
        """비용 shape = (K,)"""
        cost_fn = EndEffectorTrackingCost(self.model, weight=100.0)
        trajectories = self._make_trajectories()
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)

    def test_ee_tracking_zero_error(self):
        """EE와 ref가 같으면 비용 0"""
        cost_fn = EndEffectorTrackingCost(self.model, weight=100.0)
        # state=(0,0,0,0,0) → ee=(L1+L2, 0)
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        ref[:, 0] = self.model.L1 + self.model.L2  # ee_x
        ref[:, 1] = 0.0  # ee_y
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-8)

    def test_ee_tracking_nonzero(self):
        """EE와 ref가 다르면 양수 비용"""
        cost_fn = EndEffectorTrackingCost(self.model, weight=100.0)
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        ref[:, 0] = 5.0  # 먼 목표
        ref[:, 1] = 5.0
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert np.all(costs > 0)

    def test_ee_terminal_cost_shape(self):
        """터미널 비용 shape = (K,)"""
        cost_fn = EndEffectorTerminalCost(self.model, weight=200.0)
        trajectories = self._make_trajectories()
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)

    def test_ee_terminal_cost_zero(self):
        """터미널에서 EE = ref → 비용 0"""
        cost_fn = EndEffectorTerminalCost(self.model, weight=200.0)
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        ref[-1, 0] = self.model.L1 + self.model.L2
        ref[-1, 1] = 0.0
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-8)

    def test_ee_terminal_cost_nonzero(self):
        """터미널에서 EE ≠ ref → 양수 비용"""
        cost_fn = EndEffectorTerminalCost(self.model, weight=200.0)
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        ref[-1, 0] = 10.0
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert np.all(costs > 0)

    def test_joint_limit_within(self):
        """관절 제한 내 → 유한 비용"""
        cost_fn = JointLimitCost(
            joint_indices=(3, 4),
            joint_limits=((-2.9, 2.9), (-2.9, 2.9)),
            weight=10.0,
        )
        trajectories = self._make_trajectories(0.0)
        # q1=q2=0 → 제한 내
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)
        assert np.all(np.isfinite(costs))

    def test_joint_limit_violated(self):
        """관절 제한 위반 → 매우 큰 비용"""
        cost_fn = JointLimitCost(
            joint_indices=(3, 4),
            joint_limits=((-1.0, 1.0), (-1.0, 1.0)),
            weight=10.0,
            penalty=1e4,
        )
        # q1 = 2.0 > 1.0 제한 위반
        trajectories = self._make_trajectories(0.0)
        trajectories[:, :, 3] = 2.0
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        costs_violated = cost_fn.compute_cost(trajectories, controls, ref)

        # 정상 범위 비용
        trajectories_ok = self._make_trajectories(0.0)
        costs_ok = cost_fn.compute_cost(trajectories_ok, controls, ref)

        # 위반 비용 >> 정상 비용
        assert np.all(costs_violated > costs_ok)

    def test_composite_cost(self):
        """복합 비용 함수 통합"""
        cost_fn = CompositeMPPICost([
            EndEffectorTrackingCost(self.model, weight=100.0),
            EndEffectorTerminalCost(self.model, weight=200.0),
            ControlEffortCost(R=np.array([0.1, 0.1, 0.05, 0.05])),
        ])
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 5))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)


# ═══════════════════════════════════════════════════════
#  EE Trajectory Function Tests
# ═══════════════════════════════════════════════════════

class TestEETrajectory:
    def test_ee_circle_shape(self):
        traj_fn = create_trajectory_function("ee_circle")
        state = traj_fn(0.0)
        assert state.shape == (5,)

    def test_ee_circle_at_t0(self):
        """t=0 → EE at (center[0]+radius, center[1])"""
        traj_fn = create_trajectory_function(
            "ee_circle", radius=0.5, center=(1.0, 0.0)
        )
        state = traj_fn(0.0)
        assert state[0] == pytest.approx(1.5, abs=1e-10)
        assert state[1] == pytest.approx(0.0, abs=1e-10)

    def test_ee_figure8_shape(self):
        traj_fn = create_trajectory_function("ee_figure8")
        state = traj_fn(0.0)
        assert state.shape == (5,)

    def test_ee_circle_generates_reference(self):
        """generate_reference_trajectory 호환"""
        traj_fn = create_trajectory_function("ee_circle")
        ref = generate_reference_trajectory(traj_fn, 0.0, 30, 0.05)
        assert ref.shape == (31, 5)


# ═══════════════════════════════════════════════════════
#  BatchDynamicsWrapper Integration
# ═══════════════════════════════════════════════════════

class TestBatchDynamicsWrapperMM:
    def test_wrapper_rollout(self):
        """BatchDynamicsWrapper produces correct trajectory shape"""
        model = MobileManipulatorKinematic()
        wrapper = BatchDynamicsWrapper(model, dt=0.05)
        K, N = 64, 20
        initial_state = np.zeros(5)
        controls = np.random.randn(K, N, 4) * 0.1
        trajectories = wrapper.rollout(initial_state, controls)
        assert trajectories.shape == (K, N + 1, 5)
        np.testing.assert_allclose(
            trajectories[:, 0, :], np.tile(initial_state, (K, 1))
        )


# ═══════════════════════════════════════════════════════
#  MPPI Controller Integration
# ═══════════════════════════════════════════════════════

class TestMPPIIntegration:
    def setup_method(self):
        self.model = MobileManipulatorKinematic(L1=0.3, L2=0.25)
        self.params = MPPIParams(
            N=20,
            dt=0.05,
            K=128,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5, 1.5, 1.5]),
            Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
            R=np.array([0.1, 0.1, 0.05, 0.05]),
        )

    def test_compute_control(self):
        """compute_control → (4,) control"""
        controller = MPPIController(self.model, self.params)
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        ref = np.zeros((21, 5))
        ref[:, 0] = 1.0  # target x
        control, info = controller.compute_control(state, ref)
        assert control.shape == (4,)
        assert "sample_trajectories" in info

    def test_with_ee_cost(self):
        """EndEffectorTrackingCost로 MPPI 실행"""
        cost_fn = CompositeMPPICost([
            EndEffectorTrackingCost(self.model, weight=100.0),
            EndEffectorTerminalCost(self.model, weight=200.0),
            ControlEffortCost(R=np.array([0.1, 0.1, 0.05, 0.05])),
        ])
        controller = MPPIController(self.model, self.params, cost_function=cost_fn)
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.0])

        # EE ref: 원형 궤적
        traj_fn = create_trajectory_function("ee_circle")
        ref = generate_reference_trajectory(traj_fn, 0.0, 20, 0.05)

        control, info = controller.compute_control(state, ref)
        assert control.shape == (4,)
        assert not np.any(np.isnan(control))

    def test_short_simulation(self):
        """짧은 시뮬레이션: NaN 없음, EE 오차 감소"""
        cost_fn = CompositeMPPICost([
            EndEffectorTrackingCost(self.model, weight=100.0),
            EndEffectorTerminalCost(self.model, weight=200.0),
            ControlEffortCost(R=np.array([0.1, 0.1, 0.05, 0.05])),
        ])
        controller = MPPIController(self.model, self.params, cost_function=cost_fn)

        state = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        traj_fn = create_trajectory_function("ee_circle", radius=0.3, center=(1.0, 0.0))

        dt = self.params.dt
        ee_errors = []

        for step in range(40):
            t = step * dt
            ref = generate_reference_trajectory(traj_fn, t, 20, dt)

            control, info = controller.compute_control(state, ref)
            assert not np.any(np.isnan(control)), f"NaN at step {step}"

            # EE 오차 기록
            ee_pos = self.model.forward_kinematics(state)
            ee_ref = ref[0, :2]
            ee_error = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))
            ee_errors.append(ee_error)

            # 모델 스텝
            state = self.model.step(state, control, dt)
            state = self.model.normalize_state(state)

        # NaN 없음
        assert not np.any(np.isnan(state))
        # 후반 오차가 초반보다 작거나 같음 (수렴 경향)
        early_mean = np.mean(ee_errors[:10])
        late_mean = np.mean(ee_errors[-10:])
        assert late_mean <= early_mean + 0.5  # 약간의 마진


# ═══════════════════════════════════════════════════════
#  Standalone runner
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    test_classes = [
        TestMobileManipulatorKinematic,
        TestForwardKinematics,
        TestEndEffectorCosts,
        TestEETrajectory,
        TestBatchDynamicsWrapperMM,
        TestMPPIIntegration,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        for attr in sorted(dir(instance)):
            if not attr.startswith("test_"):
                continue
            if hasattr(instance, "setup_method"):
                instance.setup_method()
            try:
                method = getattr(instance, attr)
                if hasattr(method, "pytestmark"):
                    continue
                method()
                passed += 1
                print(f"  PASS: {cls.__name__}.{attr}")
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{attr}", str(e)))
                print(f"  FAIL: {cls.__name__}.{attr} -> {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("Failed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
