"""
Mobile Manipulator 6-DOF Swerve Tests

Tests for MobileManipulator6DOFSwerveKinematic model
(UR5-style 6-DOF arm + Swerve holonomic base).
Covers: dimensions, holonomic dynamics, FK reuse, batch processing,
cost functions, and MPPI integration.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.mobile_manipulator_6dof_swerve_kinematic import (
    MobileManipulator6DOFSwerveKinematic,
)
from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
from mppi_controller.controllers.mppi.cost_functions import (
    EndEffector3DTrackingCost,
    EndEffector3DTerminalCost,
    EndEffectorPoseTrackingCost,
    EndEffectorPoseTerminalCost,
    CompositeMPPICost,
    ControlEffortCost,
    JointLimitCost,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ═══════════════════════════════════════════════════════
#  Model Basic Tests
# ═══════════════════════════════════════════════════════

class TestMobileManipulator6DOFSwerveKinematic:
    def setup_method(self):
        self.model = MobileManipulator6DOFSwerveKinematic(
            vx_max=1.0, vy_max=1.0, omega_max=1.0, dq_max=2.0
        )

    def test_dimensions(self):
        assert self.model.state_dim == 9
        assert self.model.control_dim == 9  # vx, vy, ω + 6 joints

    def test_model_type(self):
        assert self.model.model_type == "kinematic"

    def test_is_robot_model(self):
        assert isinstance(self.model, RobotModel)

    def test_inherits_6dof(self):
        """DiffDrive 6-DOF 모델을 상속"""
        assert isinstance(self.model, MobileManipulator6DOFKinematic)

    def test_forward_dynamics_forward(self):
        """vx=1, 나머지 0 → 직진"""
        state = np.zeros(9)
        control = np.zeros(9)
        control[0] = 1.0  # vx
        dot = self.model.forward_dynamics(state, control)
        expected = np.zeros(9)
        expected[0] = 1.0  # x_dot = vx·cos(0) = 1
        np.testing.assert_allclose(dot, expected, atol=1e-10)

    def test_forward_dynamics_lateral(self):
        """vy=1, θ=0 → y방향 이동 (holonomic!)"""
        state = np.zeros(9)
        control = np.zeros(9)
        control[1] = 1.0  # vy
        dot = self.model.forward_dynamics(state, control)
        # ẋ = -vy·sin(0) = 0, ẏ = vy·cos(0) = 1
        np.testing.assert_allclose(dot[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(dot[1], 1.0, atol=1e-10)

    def test_forward_dynamics_rotation(self):
        """ω=1 → 회전만"""
        state = np.zeros(9)
        control = np.zeros(9)
        control[2] = 1.0  # ω
        dot = self.model.forward_dynamics(state, control)
        expected = np.zeros(9)
        expected[2] = 1.0
        np.testing.assert_allclose(dot, expected, atol=1e-10)

    def test_forward_dynamics_arm_only(self):
        """dq1=1 → q1만 움직임"""
        state = np.zeros(9)
        control = np.zeros(9)
        control[3] = 1.0  # dq1
        dot = self.model.forward_dynamics(state, control)
        expected = np.zeros(9)
        expected[3] = 1.0
        np.testing.assert_allclose(dot, expected, atol=1e-10)

    def test_forward_dynamics_arm_q6(self):
        """dq6=1.5 → q6만 움직임"""
        state = np.zeros(9)
        control = np.zeros(9)
        control[8] = 1.5  # dq6
        dot = self.model.forward_dynamics(state, control)
        expected = np.zeros(9)
        expected[8] = 1.5
        np.testing.assert_allclose(dot, expected, atol=1e-10)

    def test_forward_dynamics_with_heading(self):
        """θ=π/2, vx=1 → y방향 이동"""
        state = np.zeros(9)
        state[2] = np.pi / 2
        control = np.zeros(9)
        control[0] = 1.0  # vx
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(dot[1], 1.0, atol=1e-10)

    def test_forward_dynamics_diagonal(self):
        """vx=1, vy=1, θ=0 → 대각 이동"""
        state = np.zeros(9)
        control = np.zeros(9)
        control[0] = 1.0  # vx
        control[1] = 1.0  # vy
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot[0], 1.0, atol=1e-10)  # vx·cos(0) - vy·sin(0) = 1
        np.testing.assert_allclose(dot[1], 1.0, atol=1e-10)  # vx·sin(0) + vy·cos(0) = 1

    def test_batch_dynamics(self):
        """배치: (K, 9) → (K, 9)"""
        K = 100
        states = np.random.randn(K, 9) * 0.1
        controls = np.random.randn(K, 9) * 0.1
        dots = self.model.forward_dynamics(states, controls)
        assert dots.shape == (K, 9)

    def test_rk4_step(self):
        """RK4 적분"""
        state = np.zeros(9)
        control = np.zeros(9)
        control[0] = 1.0  # vx
        control[1] = 0.5  # vy
        control[3] = 0.5  # dq1
        dt = 0.05
        next_state = self.model.step(state, control, dt)
        assert next_state.shape == (9,)
        assert next_state[0] > 0  # x 증가
        assert next_state[1] > 0  # y 증가 (vy > 0)
        assert next_state[3] > 0  # q1 증가

    def test_normalize_state(self):
        """θ, q1~q6 모두 [-π, π] 래핑"""
        state = np.array([1.0, 2.0, 4.0, -4.0, 7.0, -7.0, 8.0, -8.0, 10.0])
        norm = self.model.normalize_state(state)
        for idx in range(2, 9):
            assert -np.pi <= norm[idx] <= np.pi
        assert norm[0] == 1.0
        assert norm[1] == 2.0

    def test_control_bounds(self):
        """9D 제어 제약"""
        lb, ub = self.model.get_control_bounds()
        assert lb.shape == (9,)
        assert ub.shape == (9,)
        np.testing.assert_allclose(lb[:3], [-1.0, -1.0, -1.0])
        np.testing.assert_allclose(ub[:3], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(lb[3:], [-2.0] * 6)
        np.testing.assert_allclose(ub[3:], [2.0] * 6)

    def test_state_to_dict(self):
        """9개 키 + base_type 반환"""
        state = np.array([1.0, 2.0, 0.5, 0.3, -0.2, 0.1, -0.1, 0.4, -0.4])
        d = self.model.state_to_dict(state)
        assert d["x"] == 1.0
        assert d["y"] == 2.0
        assert d["theta"] == 0.5
        assert d["q1"] == 0.3
        assert d["base_type"] == "swerve"

    def test_xy_at_indices_0_1(self):
        """CBF 호환성"""
        state = np.array([3.0, 4.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert state[0] == 3.0
        assert state[1] == 4.0


# ═══════════════════════════════════════════════════════
#  FK Reuse Tests (same DH chain as DiffDrive version)
# ═══════════════════════════════════════════════════════

class TestFKReuse6DOFSwerve:
    def setup_method(self):
        self.swerve = MobileManipulator6DOFSwerveKinematic()
        self.diff = MobileManipulator6DOFKinematic()

    def test_fk_same_as_diffdrive(self):
        """동일 state → 동일 FK 결과 (DH chain 재사용 검증)"""
        state = np.array([0.5, 0.3, 0.2, 0.1, -0.1, 0.3, -0.2, 0.15, -0.1])
        ee_swerve = self.swerve.forward_kinematics(state)
        ee_diff = self.diff.forward_kinematics(state)
        np.testing.assert_allclose(ee_swerve, ee_diff, atol=1e-12)

    def test_fk_pose_same_as_diffdrive(self):
        """동일 state → 동일 pose"""
        state = np.array([0.3, 0.2, 0.4, 0.5, -0.3, 0.2, -0.1, 0.4, -0.2])
        pose_swerve = self.swerve.forward_kinematics_pose(state)
        pose_diff = self.diff.forward_kinematics_pose(state)
        np.testing.assert_allclose(pose_swerve, pose_diff, atol=1e-12)

    def test_fk_full_same_as_diffdrive(self):
        """동일 state → 동일 4x4 transform"""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.1, -0.2, -0.3])
        T_swerve = self.swerve.forward_kinematics_full(state)
        T_diff = self.diff.forward_kinematics_full(state)
        np.testing.assert_allclose(T_swerve, T_diff, atol=1e-12)

    def test_joint_positions_same(self):
        """동일 state → 동일 관절 위치"""
        state = np.array([0.2, 0.1, 0.5, -0.3, 0.4, -0.2, 0.1, -0.5, 0.3])
        pos_swerve = self.swerve.get_joint_positions(state)
        pos_diff = self.diff.get_joint_positions(state)
        np.testing.assert_allclose(pos_swerve, pos_diff, atol=1e-12)

    def test_fk_batch(self):
        """배치 FK: (K, 9) → (K, 3)"""
        K = 50
        states = np.random.randn(K, 9) * 0.3
        ee = self.swerve.forward_kinematics(states)
        assert ee.shape == (K, 3)

    def test_fk_trajectory_batch(self):
        """궤적 배치 FK: (K, N+1, 9) → (K, N+1, 3)"""
        K, N = 32, 20
        trajectories = np.random.randn(K, N + 1, 9) * 0.3
        ee = self.swerve.forward_kinematics(trajectories)
        assert ee.shape == (K, N + 1, 3)


# ═══════════════════════════════════════════════════════
#  Holonomic vs Non-holonomic Comparison
# ═══════════════════════════════════════════════════════

class TestHolonomicAdvantage:
    def setup_method(self):
        self.swerve = MobileManipulator6DOFSwerveKinematic()
        self.diff = MobileManipulator6DOFKinematic()

    def test_lateral_motion(self):
        """Swerve는 횡방향 이동 가능, DiffDrive는 불가"""
        state = np.zeros(9)
        # Swerve: vy=1 → y 증가
        swerve_ctrl = np.zeros(9)
        swerve_ctrl[1] = 1.0  # vy
        swerve_dot = self.swerve.forward_dynamics(state, swerve_ctrl)
        assert swerve_dot[1] > 0.9  # y방향 즉시 이동

        # DiffDrive: v=0, ω=0 → 횡방향 직접 이동 불가
        diff_ctrl = np.zeros(8)
        diff_dot = self.diff.forward_dynamics(state, diff_ctrl)
        np.testing.assert_allclose(diff_dot[:2], 0.0, atol=1e-10)

    def test_swerve_step_lateral(self):
        """Swerve RK4 스텝으로 횡방향 이동 확인"""
        state = np.zeros(9)
        control = np.zeros(9)
        control[1] = 1.0  # vy
        next_state = self.swerve.step(state, control, 0.1)
        assert next_state[1] > 0.05  # y 이동


# ═══════════════════════════════════════════════════════
#  3D Cost Functions with Swerve Model
# ═══════════════════════════════════════════════════════

class TestEndEffector3DCostsSwerve:
    def setup_method(self):
        self.model = MobileManipulator6DOFSwerveKinematic()
        self.K = 32
        self.N = 10

    def _make_trajectories(self, state_val=0.0):
        return np.full((self.K, self.N + 1, 9), state_val)

    def _make_controls(self):
        return np.zeros((self.K, self.N, 9))

    def test_3d_tracking_shape(self):
        cost_fn = EndEffector3DTrackingCost(self.model, weight=100.0)
        costs = cost_fn.compute_cost(
            self._make_trajectories(), self._make_controls(),
            np.zeros((self.N + 1, 9)),
        )
        assert costs.shape == (self.K,)

    def test_3d_tracking_zero(self):
        cost_fn = EndEffector3DTrackingCost(self.model, weight=100.0)
        ee_pos = self.model.forward_kinematics(np.zeros(9))
        ref = np.zeros((self.N + 1, 9))
        ref[:, :3] = ee_pos
        costs = cost_fn.compute_cost(
            self._make_trajectories(0.0), self._make_controls(), ref,
        )
        np.testing.assert_allclose(costs, 0.0, atol=1e-6)

    def test_pose_tracking_shape(self):
        cost_fn = EndEffectorPoseTrackingCost(
            self.model, pos_weight=100.0, ori_weight=10.0
        )
        costs = cost_fn.compute_cost(
            self._make_trajectories(), self._make_controls(),
            np.zeros((self.N + 1, 9)),
        )
        assert costs.shape == (self.K,)

    def test_joint_limit_6dof(self):
        cost_fn = JointLimitCost(
            joint_indices=(3, 4, 5, 6, 7, 8),
            joint_limits=((-2.9, 2.9),) * 6,
            weight=10.0,
        )
        costs = cost_fn.compute_cost(
            self._make_trajectories(0.0), self._make_controls(),
            np.zeros((self.N + 1, 9)),
        )
        assert costs.shape == (self.K,)
        assert np.all(np.isfinite(costs))


# ═══════════════════════════════════════════════════════
#  BatchDynamicsWrapper Integration
# ═══════════════════════════════════════════════════════

class TestBatchDynamicsWrapper6DOFSwerve:
    def test_wrapper_rollout(self):
        model = MobileManipulator6DOFSwerveKinematic()
        wrapper = BatchDynamicsWrapper(model, dt=0.05)
        K, N = 64, 20
        initial_state = np.zeros(9)
        controls = np.random.randn(K, N, 9) * 0.1
        trajectories = wrapper.rollout(initial_state, controls)
        assert trajectories.shape == (K, N + 1, 9)
        np.testing.assert_allclose(
            trajectories[:, 0, :], np.tile(initial_state, (K, 1))
        )


# ═══════════════════════════════════════════════════════
#  MPPI Controller Integration
# ═══════════════════════════════════════════════════════

class TestMPPIIntegration6DOFSwerve:
    def setup_method(self):
        self.model = MobileManipulator6DOFSwerveKinematic()
        self.params = MPPIParams(
            N=20,
            dt=0.05,
            K=128,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5, 0.5] + [1.5] * 6),
            Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
            R=np.array([0.1, 0.1, 0.1] + [0.05] * 6),
        )

    def test_compute_control(self):
        """compute_control → (9,) control"""
        cost_fn = CompositeMPPICost([
            EndEffector3DTrackingCost(self.model, weight=100.0),
            ControlEffortCost(R=np.array([0.1, 0.1, 0.1] + [0.05] * 6)),
        ])
        controller = MPPIController(self.model, self.params, cost_function=cost_fn)
        state = np.zeros(9)
        ref = np.zeros((21, 9))
        ref[:, 0] = 0.5
        ref[:, 2] = 0.4
        control, info = controller.compute_control(state, ref)
        assert control.shape == (9,)
        assert "sample_trajectories" in info

    def test_with_pose_cost(self):
        """EndEffectorPoseTrackingCost → NaN 없음"""
        cost_fn = CompositeMPPICost([
            EndEffectorPoseTrackingCost(self.model, pos_weight=100.0, ori_weight=10.0),
            EndEffectorPoseTerminalCost(self.model, pos_weight=200.0, ori_weight=20.0),
            ControlEffortCost(R=np.array([0.1, 0.1, 0.1] + [0.05] * 6)),
        ])
        controller = MPPIController(self.model, self.params, cost_function=cost_fn)
        state = np.zeros(9)
        traj_fn = create_trajectory_function("ee_3d_circle")
        ref = generate_reference_trajectory(traj_fn, 0.0, 20, 0.05)
        control, info = controller.compute_control(state, ref)
        assert control.shape == (9,)
        assert not np.any(np.isnan(control))

    def test_short_simulation(self):
        """짧은 시뮬레이션: NaN 없음, EE 오차 수렴"""
        cost_fn = CompositeMPPICost([
            EndEffector3DTrackingCost(self.model, weight=100.0),
            EndEffector3DTerminalCost(self.model, weight=200.0),
            ControlEffortCost(R=np.array([0.1, 0.1, 0.1] + [0.05] * 6)),
        ])
        controller = MPPIController(self.model, self.params, cost_function=cost_fn)

        state = np.zeros(9)
        traj_fn = create_trajectory_function(
            "ee_3d_circle", radius=0.3, center=(0.4, 0.0, 0.4)
        )

        dt = self.params.dt
        ee_errors = []

        for step in range(40):
            t = step * dt
            ref = generate_reference_trajectory(traj_fn, t, 20, dt)

            control, info = controller.compute_control(state, ref)
            assert not np.any(np.isnan(control)), f"NaN at step {step}"

            ee_pos = self.model.forward_kinematics(state)
            ee_ref = ref[0, :3]
            ee_error = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))
            ee_errors.append(ee_error)

            state = self.model.step(state, control, dt)
            state = self.model.normalize_state(state)

        assert not np.any(np.isnan(state))
        early_mean = np.mean(ee_errors[:10])
        late_mean = np.mean(ee_errors[-10:])
        assert late_mean <= early_mean + 0.5


# ═══════════════════════════════════════════════════════
#  Standalone runner
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    test_classes = [
        TestMobileManipulator6DOFSwerveKinematic,
        TestFKReuse6DOFSwerve,
        TestHolonomicAdvantage,
        TestEndEffector3DCostsSwerve,
        TestBatchDynamicsWrapper6DOFSwerve,
        TestMPPIIntegration6DOFSwerve,
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
                getattr(instance, attr)()
                passed += 1
                print(f"  PASS: {cls.__name__}.{attr}")
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{attr}", str(e)))
                print(f"  FAIL: {cls.__name__}.{attr} -> {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
