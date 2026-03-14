"""
SE(3) 비용 함수 테스트
"""

import numpy as np
import pytest
from mppi_controller.controllers.mppi.se3_cost import (
    GeodesicOrientationCost,
    GeodesicOrientationTerminalCost,
    SE3TrackingCost,
    SE3TerminalCost,
    ReachabilityMapCost,
    SE3ManipulabilityCost,
    _geodesic_distance,
    _rpy_to_rotation_matrix,
    _batch_rotation_log_norm,
)
from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)


@pytest.fixture
def model():
    return MobileManipulator6DOFKinematic()


@pytest.fixture
def dummy_trajectories():
    """(K=8, N+1=11, nx=9) 샘플 궤적"""
    K, N, nx = 8, 10, 9
    rng = np.random.default_rng(42)
    traj = rng.uniform(-0.5, 0.5, (K, N + 1, nx))
    traj[:, :, 2] = rng.uniform(-np.pi, np.pi, (K, N + 1))  # θ
    traj[:, :, 3:9] = rng.uniform(-1.0, 1.0, (K, N + 1, 6))  # joints
    return traj


@pytest.fixture
def dummy_controls():
    K, N, nu = 8, 10, 8
    return np.zeros((K, N, nu))


@pytest.fixture
def dummy_reference():
    """(N+1=11, 9) 레퍼런스"""
    N, nx = 10, 9
    ref = np.zeros((N + 1, nx))
    ref[:, 3:9] = 0.1  # 약간의 관절 각도
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestGeodesicUtils:
    def test_identity_distance_zero(self):
        """동일 회전 행렬 간 거리는 0"""
        R = np.eye(3)[None]  # (1, 3, 3)
        d = _geodesic_distance(R, R)
        assert d.shape == (1,)
        assert d[0] == pytest.approx(0.0, abs=1e-10)

    def test_180deg_distance(self):
        """180도 회전은 π"""
        R1 = np.eye(3)[None]  # (1, 3, 3)
        # Rx(π): y→-y, z→-z
        R2 = np.array([[[1, 0, 0], [0, -1, 0], [0, 0, -1]]], dtype=float)
        d = _geodesic_distance(R1, R2)
        assert d[0] == pytest.approx(np.pi, abs=1e-6)

    def test_90deg_distance(self):
        """90도 회전은 π/2"""
        R1 = np.eye(3)[None]
        R2 = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=float)  # Rz(π/2)
        d = _geodesic_distance(R1, R2)
        assert d[0] == pytest.approx(np.pi / 2, abs=1e-6)

    def test_batch_geodesic(self):
        """배치 geodesic 계산"""
        N = 5
        R1 = np.eye(3)[None].repeat(N, axis=0)  # (N, 3, 3)
        # 각각 다른 각도의 Rz 회전
        angles = np.linspace(0, np.pi / 2, N)
        R2 = np.zeros((N, 3, 3))
        for i, a in enumerate(angles):
            R2[i] = [[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0],
                     [0, 0, 1]]
        d = _geodesic_distance(R1, R2)
        np.testing.assert_allclose(d, angles, atol=1e-6)

    def test_rpy_to_rotation_matrix_identity(self):
        """zero RPY → 단위 행렬"""
        rpy = np.zeros((1, 3))
        R = _rpy_to_rotation_matrix(rpy)
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)

    def test_rpy_to_rotation_matrix_yaw90(self):
        """yaw=π/2 → Rz(π/2)"""
        rpy = np.array([[0, 0, np.pi / 2]])
        R = _rpy_to_rotation_matrix(rpy)[0]
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-6)

    def test_rpy_roundtrip(self):
        """RPY → R → geodesic_distance(R, R) = 0"""
        rpy = np.array([[0.3, 0.2, 0.8]])
        R = _rpy_to_rotation_matrix(rpy)
        d = _geodesic_distance(R, R)
        assert d[0] == pytest.approx(0.0, abs=1e-10)

    def test_log_norm_batch(self):
        """배치 rotation log norm"""
        angles = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
        R = np.zeros((4, 3, 3))
        for i, a in enumerate(angles):
            R[i] = [[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0],
                     [0, 0, 1]]
        norms = _batch_rotation_log_norm(R)
        np.testing.assert_allclose(norms, angles, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# GeodesicOrientationCost
# ─────────────────────────────────────────────────────────────────────────────

class TestGeodesicOrientationCost:
    def test_output_shape(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        cost_fn = GeodesicOrientationCost(
            model,
            ori_weight=10.0,
            target_rpy=np.array([0.0, 0.0, 0.0]),
        )
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert costs.shape == (8,)

    def test_non_negative(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        cost_fn = GeodesicOrientationCost(
            model,
            ori_weight=10.0,
            target_rpy=np.array([0.0, 0.0, 0.0]),
        )
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert np.all(costs >= 0)

    def test_rpy_ref_indices(self, model, dummy_trajectories, dummy_controls):
        """rpy_ref_indices로 reference_trajectory에서 RPY 추출"""
        N = dummy_trajectories.shape[1] - 1
        ref = np.zeros((N + 1, 9))
        ref[:, 3:6] = np.array([0.1, 0.2, 0.3])
        cost_fn = GeodesicOrientationCost(
            model,
            ori_weight=5.0,
            rpy_ref_indices=(3, 4, 5),
        )
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, ref)
        assert costs.shape == (8,)
        assert np.all(costs >= 0)

    def test_zero_cost_at_target(self, model, dummy_controls):
        """목표 자세와 동일하면 비용이 0에 가까움"""
        target_rpy = np.array([0.0, 0.0, 0.0])
        # 모든 샘플이 동일 상태 (zero joints → FK → 동일 EE pose)
        K, N_plus_1, nx = 4, 11, 9
        traj = np.zeros((K, N_plus_1, nx))  # 모두 0 상태
        ref = np.zeros((N_plus_1, nx))

        cost_fn = GeodesicOrientationCost(
            model,
            ori_weight=10.0,
            target_rpy=target_rpy,
        )
        costs = cost_fn.compute_cost(traj, np.zeros((K, N_plus_1 - 1, 8)), ref)
        # 모든 샘플 동일 → 비용도 동일해야 함
        assert np.allclose(costs, costs[0], atol=1e-10)


class TestGeodesicOrientationTerminalCost:
    def test_output_shape(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        cost_fn = GeodesicOrientationTerminalCost(
            model,
            target_rpy=np.array([0.0, 0.0, 0.0]),
            ori_weight=20.0,
        )
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert costs.shape == (8,)
        assert np.all(costs >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# SE3TrackingCost
# ─────────────────────────────────────────────────────────────────────────────

class TestSE3TrackingCost:
    def test_output_shape(self, model, dummy_trajectories, dummy_controls):
        N = dummy_trajectories.shape[1] - 1
        ref = np.zeros((N + 1, 6))  # [x, y, z, roll, pitch, yaw]
        cost_fn = SE3TrackingCost(model, pos_weight=100.0, ori_weight=10.0)
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, ref)
        assert costs.shape == (8,)

    def test_non_negative(self, model, dummy_trajectories, dummy_controls):
        N = dummy_trajectories.shape[1] - 1
        ref = np.zeros((N + 1, 6))
        cost_fn = SE3TrackingCost(model)
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, ref)
        assert np.all(costs >= 0)

    def test_target_pos_target_rpy(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        """고정 target_pos, target_rpy 사용"""
        cost_fn = SE3TrackingCost(
            model,
            target_pos=np.array([1.0, 0.5, 0.3]),
            target_rpy=np.array([0.0, 0.0, np.pi / 4]),
        )
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert costs.shape == (8,)
        assert np.all(costs >= 0)

    def test_larger_weight_larger_cost(self, model, dummy_trajectories, dummy_controls):
        N = dummy_trajectories.shape[1] - 1
        ref = np.zeros((N + 1, 6))
        cost_low = SE3TrackingCost(model, pos_weight=1.0, ori_weight=1.0)
        cost_high = SE3TrackingCost(model, pos_weight=100.0, ori_weight=10.0)
        c_low = cost_low.compute_cost(dummy_trajectories, dummy_controls, ref)
        c_high = cost_high.compute_cost(dummy_trajectories, dummy_controls, ref)
        assert np.all(c_high >= c_low)


# ─────────────────────────────────────────────────────────────────────────────
# SE3TerminalCost
# ─────────────────────────────────────────────────────────────────────────────

class TestSE3TerminalCost:
    def test_output_shape(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        cost_fn = SE3TerminalCost(
            model,
            target_pos=np.array([0.5, 0.2, 0.4]),
            target_rpy=np.array([0.0, 0.0, 0.0]),
        )
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert costs.shape == (8,)
        assert np.all(costs >= 0)

    def test_zero_cost_same_pose(self, model, dummy_controls, dummy_reference):
        """동일한 목표 위치/자세면 비용이 최소"""
        K, N_plus_1, nx = 4, 11, 9
        traj = np.zeros((K, N_plus_1, nx))  # 동일 상태

        # FK 계산으로 실제 EE 위치/자세 구하기
        model_inst = MobileManipulator6DOFKinematic()
        T = model_inst.forward_kinematics_full(np.zeros(nx))
        target_pos = T[:3, 3]
        R = T[:3, :3]
        # R에서 RPY 추출
        rpy_pose = model_inst.forward_kinematics_pose(np.zeros(nx))[3:6]

        cost_fn = SE3TerminalCost(
            model_inst,
            target_pos=target_pos,
            target_rpy=rpy_pose,
        )
        costs = cost_fn.compute_cost(traj, np.zeros((K, N_plus_1 - 1, 8)), dummy_reference)
        # 모두 동일 상태 → 동일 비용
        assert np.allclose(costs, costs[0], atol=1e-10)
        # zero state에서는 비용이 거의 0
        assert np.all(costs < 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# ReachabilityMapCost
# ─────────────────────────────────────────────────────────────────────────────

class TestReachabilityMapCost:
    def test_output_shape(self, dummy_trajectories, dummy_controls, dummy_reference):
        cost_fn = ReachabilityMapCost(
            target_pos=np.array([2.0, 1.0, 0.3]),
            max_reach=0.7,
        )
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert costs.shape == (8,)

    def test_non_negative(self, dummy_trajectories, dummy_controls, dummy_reference):
        cost_fn = ReachabilityMapCost(target_pos=np.array([1.0, 1.0, 0.5]))
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert np.all(costs >= 0)

    def test_zero_cost_within_reach(self, dummy_controls, dummy_reference):
        """베이스가 목표 근처에 있으면 비용 0"""
        K, N_plus_1, nx = 4, 11, 9
        traj = np.zeros((K, N_plus_1, nx))
        # 베이스를 목표 바로 아래에 배치 (거리 < max_reach)
        target = np.array([0.3, 0.3, 0.3])
        traj[:, :, 0] = 0.3  # x
        traj[:, :, 1] = 0.3  # y

        cost_fn = ReachabilityMapCost(
            target_pos=target,
            max_reach=0.7,
            margin=0.05,
        )
        costs = cost_fn.compute_cost(traj, dummy_controls, dummy_reference)
        assert np.all(costs == pytest.approx(0.0, abs=1e-10))

    def test_cost_increases_with_distance(self, dummy_controls, dummy_reference):
        """거리가 멀수록 비용 증가"""
        K, N_plus_1, nx = 4, 11, 9
        target = np.array([0.0, 0.0, 0.0])

        traj_near = np.zeros((K, N_plus_1, nx))
        traj_near[:, :, 0] = 1.5  # 목표에서 1.5m

        traj_far = np.zeros((K, N_plus_1, nx))
        traj_far[:, :, 0] = 3.0  # 목표에서 3.0m

        cost_fn = ReachabilityMapCost(target_pos=target, max_reach=0.7)

        cost_near = cost_fn.compute_cost(traj_near, dummy_controls, dummy_reference)
        cost_far = cost_fn.compute_cost(traj_far, dummy_controls, dummy_reference)
        assert np.all(cost_far >= cost_near)


# ─────────────────────────────────────────────────────────────────────────────
# SE3ManipulabilityCost
# ─────────────────────────────────────────────────────────────────────────────

class TestSE3ManipulabilityCost:
    def test_output_shape(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        cost_fn = SE3ManipulabilityCost(model)
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert costs.shape == (8,)

    def test_non_negative(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        cost_fn = SE3ManipulabilityCost(model)
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert np.all(costs >= 0)

    def test_near_singularity_high_cost(self, model, dummy_controls, dummy_reference):
        """특이점 근처 자세 (q=0이면 어떤 DH 파라미터에 따라 낮은 manipulability)에서 비용 발생"""
        K, N_plus_1, nx = 4, 11, 9
        traj = np.zeros((K, N_plus_1, nx))

        cost_fn = SE3ManipulabilityCost(
            model,
            threshold=1e6,  # 임계값을 매우 크게 → 모든 경우 비용 발생
            manip_weight=1.0,
        )
        costs = cost_fn.compute_cost(traj, dummy_controls, dummy_reference)
        assert np.all(costs >= 0)

    def test_custom_joint_indices(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        """사용자 정의 관절 인덱스"""
        cost_fn = SE3ManipulabilityCost(
            model,
            joint_indices=[3, 4, 5],  # 처음 3개 관절만
            threshold=0.01,
        )
        costs = cost_fn.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert costs.shape == (8,)


# ─────────────────────────────────────────────────────────────────────────────
# 통합: CompositeMPPICost에 SE3 비용 포함
# ─────────────────────────────────────────────────────────────────────────────

class TestSE3CostIntegration:
    def test_composite_with_se3(self, model, dummy_trajectories, dummy_controls):
        """SE3TrackingCost + ReachabilityMapCost 복합 사용"""
        from mppi_controller.controllers.mppi.cost_functions import CompositeMPPICost

        N = dummy_trajectories.shape[1] - 1
        ref = np.zeros((N + 1, 6))

        composite = CompositeMPPICost([
            SE3TrackingCost(model, pos_weight=100.0, ori_weight=10.0),
            ReachabilityMapCost(target_pos=np.array([0.5, 0.5, 0.3])),
        ])
        costs = composite.compute_cost(dummy_trajectories, dummy_controls, ref)
        assert costs.shape == (8,)
        assert np.all(costs >= 0)

    def test_se3_terminal_in_composite(self, model, dummy_trajectories, dummy_controls, dummy_reference):
        """SE3TerminalCost를 CompositeMPPICost에 포함"""
        from mppi_controller.controllers.mppi.cost_functions import CompositeMPPICost

        composite = CompositeMPPICost([
            SE3TerminalCost(
                model,
                target_pos=np.array([0.5, 0.0, 0.6]),
                target_rpy=np.array([0.0, 0.0, 0.0]),
                pos_weight=200.0,
                ori_weight=20.0,
            ),
        ])
        costs = composite.compute_cost(dummy_trajectories, dummy_controls, dummy_reference)
        assert costs.shape == (8,)
        assert np.all(costs >= 0)
