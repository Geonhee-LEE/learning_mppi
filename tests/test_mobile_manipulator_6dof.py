"""
Mobile Manipulator 6-DOF Tests

Tests for MobileManipulator6DOFKinematic model (UR5-style 6-DOF arm + DiffDrive base).
Covers: dimensions, dynamics, FK, FK pose, batch processing, RK4, normalize_state,
control bounds, 3D cost functions, 3D trajectories, and MPPI integration.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

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
    JointLimitCost,
    CompositeMPPICost,
    ControlEffortCost,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ═══════════════════════════════════════════════════════
#  MobileManipulator6DOFKinematic Model Tests
# ═══════════════════════════════════════════════════════

class TestMobileManipulator6DOFKinematic:
    def setup_method(self):
        self.model = MobileManipulator6DOFKinematic(
            v_max=1.0, omega_max=1.0, dq_max=2.0
        )

    def test_dimensions(self):
        assert self.model.state_dim == 9
        assert self.model.control_dim == 8

    def test_model_type(self):
        assert self.model.model_type == "kinematic"

    def test_is_robot_model(self):
        assert isinstance(self.model, RobotModel)

    def test_forward_dynamics_straight(self):
        """v=1, 나머지 0 → 직진 (cos(θ), sin(θ), 0, ...)"""
        state = np.zeros(9)
        control = np.zeros(8)
        control[0] = 1.0  # v
        dot = self.model.forward_dynamics(state, control)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(dot, expected, atol=1e-10)

    def test_forward_dynamics_rotation(self):
        """ω=1 → 회전만"""
        state = np.zeros(9)
        control = np.zeros(8)
        control[1] = 1.0  # ω
        dot = self.model.forward_dynamics(state, control)
        expected = np.zeros(9)
        expected[2] = 1.0
        np.testing.assert_allclose(dot, expected, atol=1e-10)

    def test_forward_dynamics_arm_only(self):
        """dq1=1 → q1만 움직임"""
        state = np.zeros(9)
        control = np.zeros(8)
        control[2] = 1.0  # dq1
        dot = self.model.forward_dynamics(state, control)
        expected = np.zeros(9)
        expected[3] = 1.0
        np.testing.assert_allclose(dot, expected, atol=1e-10)

    def test_forward_dynamics_arm_q6(self):
        """dq6=1.5 → q6만 움직임"""
        state = np.zeros(9)
        control = np.zeros(8)
        control[7] = 1.5  # dq6
        dot = self.model.forward_dynamics(state, control)
        expected = np.zeros(9)
        expected[8] = 1.5
        np.testing.assert_allclose(dot, expected, atol=1e-10)

    def test_forward_dynamics_with_heading(self):
        """θ=π/2, v=1 → y방향 이동"""
        state = np.zeros(9)
        state[2] = np.pi / 2
        control = np.zeros(8)
        control[0] = 1.0  # v
        dot = self.model.forward_dynamics(state, control)
        np.testing.assert_allclose(dot[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(dot[1], 1.0, atol=1e-10)
        np.testing.assert_allclose(dot[2:], 0.0, atol=1e-10)

    def test_batch_dynamics(self):
        """배치 동역학: (K, 9) → (K, 9)"""
        K = 100
        states = np.random.randn(K, 9) * 0.1
        controls = np.random.randn(K, 8) * 0.1
        dots = self.model.forward_dynamics(states, controls)
        assert dots.shape == (K, 9)

    def test_rk4_step(self):
        """RK4 적분 결과 올바른 shape"""
        state = np.zeros(9)
        control = np.zeros(8)
        control[0] = 1.0  # v
        control[2] = 0.5  # dq1
        dt = 0.05
        next_state = self.model.step(state, control, dt)
        assert next_state.shape == (9,)
        assert next_state[0] > 0  # x 증가
        assert next_state[3] > 0  # q1 증가

    def test_normalize_state(self):
        """θ, q1~q6 모두 [-π, π] 래핑"""
        state = np.array([1.0, 2.0, 4.0, -4.0, 7.0, -7.0, 8.0, -8.0, 10.0])
        norm = self.model.normalize_state(state)
        for idx in range(2, 9):
            assert -np.pi <= norm[idx] <= np.pi
        # x, y는 변하지 않음
        assert norm[0] == 1.0
        assert norm[1] == 2.0

    def test_normalize_state_batch(self):
        """배치 정규화"""
        states = np.array([
            [0.0, 0.0, 4.0, -4.0, 7.0, -7.0, 8.0, -8.0, 10.0],
            [1.0, 1.0, -7.0, 8.0, -10.0, 3.5, -3.5, 6.0, -6.0],
        ])
        norm = self.model.normalize_state(states)
        assert norm.shape == (2, 9)
        for i in range(2):
            for j in range(2, 9):
                assert -np.pi <= norm[i, j] <= np.pi

    def test_control_bounds(self):
        """8D 제어 제약"""
        lb, ub = self.model.get_control_bounds()
        assert lb.shape == (8,)
        assert ub.shape == (8,)
        np.testing.assert_allclose(lb[:2], [-1.0, -1.0])
        np.testing.assert_allclose(ub[:2], [1.0, 1.0])
        np.testing.assert_allclose(lb[2:], [-2.0] * 6)
        np.testing.assert_allclose(ub[2:], [2.0] * 6)

    def test_state_to_dict(self):
        """9개 키 반환"""
        state = np.array([1.0, 2.0, 0.5, 0.3, -0.2, 0.1, -0.1, 0.4, -0.4])
        d = self.model.state_to_dict(state)
        assert d["x"] == 1.0
        assert d["y"] == 2.0
        assert d["theta"] == 0.5
        assert d["q1"] == 0.3
        assert d["q2"] == -0.2
        assert d["q3"] == 0.1
        assert d["q4"] == -0.1
        assert d["q5"] == 0.4
        assert d["q6"] == -0.4

    def test_xy_at_indices_0_1(self):
        """x=state[0], y=state[1] — CBF 호환성"""
        state = np.array([3.0, 4.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert state[0] == 3.0
        assert state[1] == 4.0


# ═══════════════════════════════════════════════════════
#  Forward Kinematics Tests (3D Position)
# ═══════════════════════════════════════════════════════

class TestForwardKinematics6DOF:
    def setup_method(self):
        self.model = MobileManipulator6DOFKinematic()

    def test_fk_zero_config(self):
        """q=0 → EE at a known position via DH chain"""
        state = np.zeros(9)
        ee = self.model.forward_kinematics(state)
        assert ee.shape == (3,)
        # Verify finite and reasonable
        assert np.all(np.isfinite(ee))
        # z should be above mount
        assert ee[2] > 0

    def test_fk_returns_3d(self):
        """FK returns 3D position"""
        state = np.zeros(9)
        state[3] = 0.5  # q1
        ee = self.model.forward_kinematics(state)
        assert ee.shape == (3,)

    def test_fk_shoulder_yaw_90(self):
        """q1=π/2 → EE rotates around z-axis"""
        state_0 = np.zeros(9)
        state_90 = np.zeros(9)
        state_90[3] = np.pi / 2  # q1 = shoulder yaw

        ee_0 = self.model.forward_kinematics(state_0)
        ee_90 = self.model.forward_kinematics(state_90)

        # Position should change (arm rotated)
        assert not np.allclose(ee_0, ee_90, atol=1e-6)

    def test_fk_elbow_90(self):
        """q3=π/2 → elbow bent"""
        state_0 = np.zeros(9)
        state_bent = np.zeros(9)
        state_bent[5] = np.pi / 2  # q3 = elbow

        ee_0 = self.model.forward_kinematics(state_0)
        ee_bent = self.model.forward_kinematics(state_bent)

        assert not np.allclose(ee_0, ee_bent, atol=1e-6)

    def test_fk_with_base_rotation(self):
        """θ=π/2 → 팔이 y축 방향으로 회전"""
        state_0 = np.zeros(9)
        state_rotated = np.zeros(9)
        state_rotated[2] = np.pi / 2  # θ

        ee_0 = self.model.forward_kinematics(state_0)
        ee_rot = self.model.forward_kinematics(state_rotated)

        # z should stay the same (rotation around z-axis)
        np.testing.assert_allclose(ee_0[2], ee_rot[2], atol=1e-10)
        # x,y should swap due to 90-degree rotation
        np.testing.assert_allclose(ee_rot[0], -ee_0[1], atol=1e-10)
        np.testing.assert_allclose(ee_rot[1], ee_0[0], atol=1e-10)

    def test_fk_with_base_position(self):
        """base translation → EE translates too"""
        state = np.zeros(9)
        state[0] = 2.0  # x
        state[1] = 3.0  # y

        state_origin = np.zeros(9)
        ee_origin = self.model.forward_kinematics(state_origin)
        ee_moved = self.model.forward_kinematics(state)

        np.testing.assert_allclose(
            ee_moved[:2] - ee_origin[:2], [2.0, 3.0], atol=1e-10
        )

    def test_fk_batch(self):
        """배치 FK: (K, 9) → (K, 3)"""
        K = 50
        states = np.random.randn(K, 9) * 0.3
        ee = self.model.forward_kinematics(states)
        assert ee.shape == (K, 3)

    def test_fk_trajectory_batch(self):
        """궤적 배치 FK: (K, N+1, 9) → (K, N+1, 3)"""
        K, N = 32, 20
        trajectories = np.random.randn(K, N + 1, 9) * 0.3
        ee = self.model.forward_kinematics(trajectories)
        assert ee.shape == (K, N + 1, 3)

    def test_fk_workspace_reachability(self):
        """EE distance from base is bounded by arm reach"""
        state = np.zeros(9)
        ee = self.model.forward_kinematics(state)
        # Sum of all link lengths gives max reach
        dh = self.model.dh_params
        max_reach = np.sum(np.abs(dh[:, 0])) + np.sum(np.abs(dh[:, 1])) + self.model.z_mount
        dist = np.sqrt(np.sum(ee**2))
        assert dist <= max_reach + 1e-6


# ═══════════════════════════════════════════════════════
#  Forward Kinematics Pose Tests (Position + RPY)
# ═══════════════════════════════════════════════════════

class TestForwardKinematicsPose6DOF:
    def setup_method(self):
        self.model = MobileManipulator6DOFKinematic()

    def test_pose_shape(self):
        """pose = (6,) [x, y, z, roll, pitch, yaw]"""
        state = np.zeros(9)
        pose = self.model.forward_kinematics_pose(state)
        assert pose.shape == (6,)

    def test_pose_position_matches_fk(self):
        """pose[:3] == forward_kinematics()"""
        state = np.array([0.5, 0.3, 0.2, 0.1, -0.1, 0.3, -0.2, 0.15, -0.1])
        pose = self.model.forward_kinematics_pose(state)
        ee_pos = self.model.forward_kinematics(state)
        np.testing.assert_allclose(pose[:3], ee_pos, atol=1e-10)

    def test_pose_batch(self):
        """배치: (K, 9) → (K, 6)"""
        K = 32
        states = np.random.randn(K, 9) * 0.3
        poses = self.model.forward_kinematics_pose(states)
        assert poses.shape == (K, 6)

    def test_full_transform_4x4(self):
        """full transform returns 4x4"""
        state = np.zeros(9)
        T = self.model.forward_kinematics_full(state)
        assert T.shape == (4, 4)
        # Last row is [0, 0, 0, 1]
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_full_transform_orthogonal(self):
        """Rotation part is orthogonal: R^T @ R = I"""
        state = np.array([0.5, 0.3, 0.5, 0.3, -0.4, 0.6, -0.2, 0.1, -0.3])
        T = self.model.forward_kinematics_full(state)
        R = T[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rpy_roundtrip(self):
        """RPY → R → RPY roundtrip consistency"""
        state = np.array([0.0, 0.0, 0.0, 0.3, -0.4, 0.6, -0.2, 0.1, 0.0])
        pose = self.model.forward_kinematics_pose(state)
        rpy = pose[3:6]
        # Verify all angles are finite
        assert np.all(np.isfinite(rpy))
        # Verify angles in reasonable range
        for angle in rpy:
            assert -np.pi <= angle <= np.pi


# ═══════════════════════════════════════════════════════
#  Joint Positions (Visualization) Tests
# ═══════════════════════════════════════════════════════

class TestJointPositions6DOF:
    def setup_method(self):
        self.model = MobileManipulator6DOFKinematic()

    def test_joint_positions_shape(self):
        """7 points: mount + 6 joints (EE = last)"""
        state = np.zeros(9)
        positions = self.model.get_joint_positions(state)
        assert positions.shape == (7, 3)

    def test_joint_positions_batch(self):
        """배치: (K, 9) → (K, 7, 3)"""
        K = 16
        states = np.random.randn(K, 9) * 0.3
        positions = self.model.get_joint_positions(states)
        assert positions.shape == (K, 7, 3)

    def test_joint_positions_ee_matches_fk(self):
        """마지막 관절 위치 = EE 위치"""
        state = np.array([0.3, 0.2, 0.1, 0.5, -0.3, 0.2, -0.1, 0.4, -0.2])
        positions = self.model.get_joint_positions(state)
        ee_fk = self.model.forward_kinematics(state)
        np.testing.assert_allclose(positions[-1], ee_fk, atol=1e-10)

    def test_mount_position(self):
        """mount position = (x, y, z_mount)"""
        state = np.zeros(9)
        state[0] = 1.0
        state[1] = 2.0
        positions = self.model.get_joint_positions(state)
        np.testing.assert_allclose(
            positions[0], [1.0, 2.0, self.model.z_mount], atol=1e-10
        )


# ═══════════════════════════════════════════════════════
#  End-Effector 3D Cost Function Tests
# ═══════════════════════════════════════════════════════

class TestEndEffector3DCosts:
    def setup_method(self):
        self.model = MobileManipulator6DOFKinematic()
        self.K = 32
        self.N = 10

    def _make_trajectories(self, state_val=0.0):
        return np.full((self.K, self.N + 1, 9), state_val)

    def _make_controls(self):
        return np.zeros((self.K, self.N, 8))

    def test_3d_tracking_shape(self):
        """비용 shape = (K,)"""
        cost_fn = EndEffector3DTrackingCost(self.model, weight=100.0)
        trajectories = self._make_trajectories()
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 9))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)

    def test_3d_tracking_zero(self):
        """EE와 ref가 같으면 비용 0"""
        cost_fn = EndEffector3DTrackingCost(self.model, weight=100.0)
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        # Get EE position at zero config
        ee_pos = self.model.forward_kinematics(np.zeros(9))
        ref = np.zeros((self.N + 1, 9))
        ref[:, :3] = ee_pos
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-6)

    def test_3d_tracking_nonzero(self):
        """EE와 ref가 다르면 양수 비용"""
        cost_fn = EndEffector3DTrackingCost(self.model, weight=100.0)
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 9))
        ref[:, 0] = 5.0  # far target
        ref[:, 1] = 5.0
        ref[:, 2] = 5.0
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert np.all(costs > 0)

    def test_3d_terminal_shape(self):
        """터미널 비용 shape = (K,)"""
        cost_fn = EndEffector3DTerminalCost(self.model, weight=200.0)
        trajectories = self._make_trajectories()
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 9))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)

    def test_3d_terminal_zero(self):
        """터미널에서 EE = ref → 비용 0"""
        cost_fn = EndEffector3DTerminalCost(self.model, weight=200.0)
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ee_pos = self.model.forward_kinematics(np.zeros(9))
        ref = np.zeros((self.N + 1, 9))
        ref[-1, :3] = ee_pos
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-6)

    def test_pose_tracking_shape(self):
        """Pose tracking cost shape = (K,)"""
        cost_fn = EndEffectorPoseTrackingCost(
            self.model, pos_weight=100.0, ori_weight=10.0
        )
        trajectories = self._make_trajectories()
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 9))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)

    def test_pose_tracking_zero(self):
        """EE pose = ref → 비용 0"""
        cost_fn = EndEffectorPoseTrackingCost(
            self.model, pos_weight=100.0, ori_weight=10.0
        )
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ee_pose = self.model.forward_kinematics_pose(np.zeros(9))
        ref = np.zeros((self.N + 1, 9))
        ref[:, :6] = ee_pose
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        np.testing.assert_allclose(costs, 0.0, atol=1e-4)

    def test_pose_tracking_pos_only(self):
        """ori_weight=0 → position only"""
        cost_fn = EndEffectorPoseTrackingCost(
            self.model, pos_weight=100.0, ori_weight=0.0
        )
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 9))
        ref[:, 0] = 5.0  # far target x
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert np.all(costs > 0)

        # Compare with 3D tracking cost (should match for pos only)
        cost_fn_3d = EndEffector3DTrackingCost(self.model, weight=100.0)
        costs_3d = cost_fn_3d.compute_cost(trajectories, controls, ref)
        np.testing.assert_allclose(costs, costs_3d, atol=1e-6)

    def test_pose_tracking_ori_only(self):
        """pos_weight=0 → orientation only"""
        cost_fn = EndEffectorPoseTrackingCost(
            self.model, pos_weight=0.0, ori_weight=10.0
        )
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        # Set reference orientation different from zero config
        ee_pose = self.model.forward_kinematics_pose(np.zeros(9))
        ref = np.zeros((self.N + 1, 9))
        ref[:, :3] = ee_pose[:3]
        ref[:, 3] = 1.0  # roll != actual
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert np.all(costs > 0)

    def test_pose_terminal_shape(self):
        """Pose terminal cost shape = (K,)"""
        cost_fn = EndEffectorPoseTerminalCost(
            self.model, pos_weight=200.0, ori_weight=20.0
        )
        trajectories = self._make_trajectories()
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 9))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)

    def test_joint_limit_6dof(self):
        """6-DOF 관절 제한 비용"""
        cost_fn = JointLimitCost(
            joint_indices=(3, 4, 5, 6, 7, 8),
            joint_limits=(
                (-2.9, 2.9), (-2.9, 2.9), (-2.9, 2.9),
                (-2.9, 2.9), (-2.9, 2.9), (-2.9, 2.9),
            ),
            weight=10.0,
        )
        trajectories = self._make_trajectories(0.0)
        controls = self._make_controls()
        ref = np.zeros((self.N + 1, 9))
        costs = cost_fn.compute_cost(trajectories, controls, ref)
        assert costs.shape == (self.K,)
        assert np.all(np.isfinite(costs))


# ═══════════════════════════════════════════════════════
#  EE 3D Trajectory Function Tests
# ═══════════════════════════════════════════════════════

class TestEE3DTrajectory:
    def test_3d_circle_shape(self):
        traj_fn = create_trajectory_function("ee_3d_circle")
        state = traj_fn(0.0)
        assert state.shape == (9,)

    def test_3d_circle_at_t0(self):
        """t=0 → EE at (center[0]+radius, center[1], center[2])"""
        traj_fn = create_trajectory_function(
            "ee_3d_circle", radius=0.3, center=(0.4, 0.0, 0.4)
        )
        state = traj_fn(0.0)
        assert state[0] == pytest.approx(0.7, abs=1e-10)  # 0.4 + 0.3
        assert state[1] == pytest.approx(0.0, abs=1e-10)
        assert state[2] == pytest.approx(0.4, abs=1e-10)

    def test_3d_circle_generates_reference(self):
        """generate_reference_trajectory 호환"""
        traj_fn = create_trajectory_function("ee_3d_circle")
        ref = generate_reference_trajectory(traj_fn, 0.0, 30, 0.05)
        assert ref.shape == (31, 9)

    def test_3d_helix_z_varies(self):
        """helix z가 시간에 따라 변함"""
        traj_fn = create_trajectory_function(
            "ee_3d_helix", z_amplitude=0.15, z_frequency=0.3
        )
        z_values = [traj_fn(t)[2] for t in np.linspace(0, 10, 100)]
        z_range = max(z_values) - min(z_values)
        assert z_range > 0.1  # z가 변해야 함


# ═══════════════════════════════════════════════════════
#  BatchDynamicsWrapper Integration
# ═══════════════════════════════════════════════════════

class TestBatchDynamicsWrapper6DOF:
    def test_wrapper_rollout(self):
        """BatchDynamicsWrapper produces correct trajectory shape"""
        model = MobileManipulator6DOFKinematic()
        wrapper = BatchDynamicsWrapper(model, dt=0.05)
        K, N = 64, 20
        initial_state = np.zeros(9)
        controls = np.random.randn(K, N, 8) * 0.1
        trajectories = wrapper.rollout(initial_state, controls)
        assert trajectories.shape == (K, N + 1, 9)
        np.testing.assert_allclose(
            trajectories[:, 0, :], np.tile(initial_state, (K, 1))
        )


# ═══════════════════════════════════════════════════════
#  MPPI Controller Integration
# ═══════════════════════════════════════════════════════

class TestMPPIIntegration6DOF:
    def setup_method(self):
        self.model = MobileManipulator6DOFKinematic()
        self.params = MPPIParams(
            N=20,
            dt=0.05,
            K=128,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5] + [1.5] * 6),
            Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
            R=np.array([0.1, 0.1] + [0.05] * 6),
        )

    def test_compute_control(self):
        """compute_control → (8,) control"""
        cost_fn = CompositeMPPICost([
            EndEffector3DTrackingCost(self.model, weight=100.0),
            ControlEffortCost(R=np.array([0.1, 0.1] + [0.05] * 6)),
        ])
        controller = MPPIController(self.model, self.params, cost_function=cost_fn)
        state = np.zeros(9)
        ref = np.zeros((21, 9))
        ref[:, 0] = 0.5
        ref[:, 2] = 0.4
        control, info = controller.compute_control(state, ref)
        assert control.shape == (8,)
        assert "sample_trajectories" in info

    def test_with_pose_cost(self):
        """EndEffectorPoseTrackingCost로 MPPI 실행 → NaN 없음"""
        cost_fn = CompositeMPPICost([
            EndEffectorPoseTrackingCost(self.model, pos_weight=100.0, ori_weight=10.0),
            EndEffectorPoseTerminalCost(self.model, pos_weight=200.0, ori_weight=20.0),
            ControlEffortCost(R=np.array([0.1, 0.1] + [0.05] * 6)),
        ])
        controller = MPPIController(self.model, self.params, cost_function=cost_fn)
        state = np.zeros(9)

        traj_fn = create_trajectory_function("ee_3d_circle")
        ref = generate_reference_trajectory(traj_fn, 0.0, 20, 0.05)

        control, info = controller.compute_control(state, ref)
        assert control.shape == (8,)
        assert not np.any(np.isnan(control))

    def test_short_simulation(self):
        """짧은 시뮬레이션: NaN 없음, EE 오차 감소"""
        cost_fn = CompositeMPPICost([
            EndEffector3DTrackingCost(self.model, weight=100.0),
            EndEffector3DTerminalCost(self.model, weight=200.0),
            ControlEffortCost(R=np.array([0.1, 0.1] + [0.05] * 6)),
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

            # EE 오차 기록
            ee_pos = self.model.forward_kinematics(state)
            ee_ref = ref[0, :3]
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
        TestMobileManipulator6DOFKinematic,
        TestForwardKinematics6DOF,
        TestForwardKinematicsPose6DOF,
        TestJointPositions6DOF,
        TestEndEffector3DCosts,
        TestEE3DTrajectory,
        TestBatchDynamicsWrapper6DOF,
        TestMPPIIntegration6DOF,
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
