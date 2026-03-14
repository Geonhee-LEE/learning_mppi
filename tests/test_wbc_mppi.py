"""
WBC-MPPI (Whole-Body Control MPPI) 테스트
"""

import numpy as np
import pytest
from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.controllers.mppi.wbc_mppi import (
    WBCMPPIController,
    WBCNoiseSampler,
    DEFAULT_JOINT_LIMITS_6DOF,
)
from mppi_controller.controllers.mppi.mppi_params import WBCMPPIParams
from mppi_controller.controllers.mppi.manipulation_costs import (
    ReachabilityWorkspaceCost,
    ArmSingularityAvoidanceCost,
    GraspApproachCost,
    CollisionFreeSweepCost,
    WBCBaseNavigationCost,
    JointVelocitySmoothCost,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    return MobileManipulator6DOFKinematic()


@pytest.fixture
def params():
    return WBCMPPIParams(
        N=10,
        K=32,
        dt=0.05,
        lambda_=1.0,
        sigma=np.concatenate([np.array([0.3, 0.3]), np.ones(6) * 0.5]),
        Q=np.zeros(9),  # EE 비용으로 대체 (상태 추적 비용 비활성화)
        R=np.zeros(8),
        ee_pos_weight=100.0,
        ee_ori_weight=5.0,
        singularity_weight=5.0,
        reachability_weight=20.0,
        task_mode="ee_tracking",
    )


@pytest.fixture
def ee_target():
    return np.array([0.5, 0.3, 0.6])


@pytest.fixture
def ee_target_rpy():
    return np.array([0.0, -np.pi / 4, 0.0])


@pytest.fixture
def state():
    """[x, y, θ, q1..q6]"""
    return np.array([0.0, 0.0, 0.0, 0.1, -0.5, 0.8, 0.0, -0.3, 0.0])


@pytest.fixture
def reference():
    """(N+1, 9) 참조 궤적"""
    N = 10
    ref = np.zeros((N + 1, 9))
    ref[:, 0] = np.linspace(0, 0.5, N + 1)  # x 이동
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# WBCMPPIParams 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestWBCMPPIParams:
    def test_defaults(self):
        p = WBCMPPIParams(
            sigma=np.ones(8) * 0.3,
            Q=np.zeros(9),
            R=np.zeros(8),
        )
        assert p.ee_pos_weight == 100.0
        assert p.ee_ori_weight == 10.0
        assert p.task_mode == "ee_tracking"
        assert p.max_arm_reach > p.min_arm_reach

    def test_invalid_task_mode(self):
        with pytest.raises(AssertionError):
            WBCMPPIParams(
                sigma=np.ones(8) * 0.3,
                Q=np.zeros(9),
                R=np.zeros(8),
                task_mode="invalid",
            )

    def test_all_task_modes(self):
        for mode in ("ee_tracking", "navigation", "both"):
            p = WBCMPPIParams(
                sigma=np.ones(8) * 0.3,
                Q=np.zeros(9),
                R=np.zeros(8),
                task_mode=mode,
            )
            assert p.task_mode == mode

    def test_reach_consistency(self):
        with pytest.raises(AssertionError):
            WBCMPPIParams(
                sigma=np.ones(8) * 0.3,
                Q=np.zeros(9),
                R=np.zeros(8),
                max_arm_reach=0.3,
                min_arm_reach=0.5,  # min > max → 에러
            )


# ─────────────────────────────────────────────────────────────────────────────
# WBCNoiseSampler 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestWBCNoiseSampler:
    def test_output_shape(self):
        sampler = WBCNoiseSampler()
        U = np.zeros((10, 8))
        noise = sampler.sample(U, K=32)
        assert noise.shape == (32, 10, 8)

    def test_ee_tracking_mode_lower_base_noise(self):
        """EE 추적 모드: 베이스 노이즈가 팔 노이즈보다 낮음"""
        sampler = WBCNoiseSampler(
            sigma_base=np.array([0.3, 0.3]),
            sigma_arm=np.ones(6) * 0.5,
            task_mode="ee_tracking",
        )
        # ee_tracking 모드에서는 sigma_base가 0.5배로 감소
        assert np.all(sampler.sigma_base < sampler.sigma_arm[:2] + 0.1)

    def test_navigation_mode_lower_arm_noise(self):
        """네비게이션 모드: 팔 노이즈가 감소"""
        sampler_nav = WBCNoiseSampler(
            sigma_base=np.array([0.3, 0.3]),
            sigma_arm=np.ones(6) * 0.5,
            task_mode="navigation",
        )
        sampler_both = WBCNoiseSampler(
            sigma_base=np.array([0.3, 0.3]),
            sigma_arm=np.ones(6) * 0.5,
            task_mode="both",
        )
        assert np.all(sampler_nav.sigma_arm <= sampler_both.sigma_arm)

    def test_different_noise_per_dof(self):
        """베이스와 팔에 다른 노이즈"""
        sampler = WBCNoiseSampler(
            sigma_base=np.array([0.1, 0.1]),
            sigma_arm=np.ones(6) * 2.0,
            task_mode="both",
        )
        # 팔 sigma가 훨씬 크므로 팔 노이즈가 더 큼
        assert sampler.sigma_arm.mean() > sampler.sigma_base.mean()


# ─────────────────────────────────────────────────────────────────────────────
# WBCMPPIController 기본 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestWBCMPPIController:
    def test_instantiation(self, model, params, ee_target, ee_target_rpy):
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=ee_target,
            ee_target_rpy=ee_target_rpy,
        )
        assert ctrl is not None
        assert ctrl.ee_target_pos is not None
        assert ctrl.ee_target_rpy is not None

    def test_instantiation_no_ee_target(self, model, params):
        """EE 목표 없이도 생성 가능 (내비게이션 모드)"""
        params.task_mode = "navigation"
        ctrl = WBCMPPIController(model, params)
        assert ctrl.ee_target_pos is None

    def test_control_output_shape(self, model, params, ee_target, state, reference):
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=ee_target,
        )
        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (8,)  # [v, ω, dq1..6]

    def test_control_output_finite(self, model, params, ee_target, state, reference):
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=ee_target,
        )
        u, info = ctrl.compute_control(state, reference)
        assert np.all(np.isfinite(u))

    def test_info_dict_keys(self, model, params, ee_target, state, reference):
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=ee_target,
        )
        u, info = ctrl.compute_control(state, reference)
        assert "sample_trajectories" in info
        assert "sample_weights" in info
        assert "best_trajectory" in info
        assert "ee_position" in info
        assert "task_mode" in info

    def test_info_sample_trajectories_shape(self, model, params, ee_target, state, reference):
        params.K = 16
        params.N = 10
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=ee_target,
        )
        u, info = ctrl.compute_control(state, reference)
        K = info["sample_trajectories"].shape[0]
        assert K == 16
        assert info["sample_trajectories"].shape[2] == 9  # nx=9

    def test_info_ee_position(self, model, params, ee_target, state, reference):
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=ee_target,
        )
        u, info = ctrl.compute_control(state, reference)
        assert info["ee_position"].shape == (3,)

    def test_info_task_mode(self, model, params, ee_target, state, reference):
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=ee_target,
        )
        u, info = ctrl.compute_control(state, reference)
        assert info["task_mode"] == "ee_tracking"


# ─────────────────────────────────────────────────────────────────────────────
# set_ee_target 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestSetEETarget:
    def test_set_ee_target_updates(self, model, params, state, reference):
        ctrl = WBCMPPIController(model, params, ee_target_pos=np.array([0.5, 0.3, 0.6]))
        new_target = np.array([1.0, 0.5, 0.8])
        ctrl.set_ee_target(new_target)
        np.testing.assert_array_equal(ctrl.ee_target_pos, new_target)

    def test_set_ee_target_with_rpy(self, model, params, state, reference):
        ctrl = WBCMPPIController(model, params, ee_target_pos=np.array([0.5, 0.3, 0.6]))
        new_pos = np.array([0.8, 0.2, 0.5])
        new_rpy = np.array([0.0, -np.pi / 2, 0.0])
        ctrl.set_ee_target(new_pos, new_rpy)
        np.testing.assert_array_equal(ctrl.ee_target_pos, new_pos)
        np.testing.assert_array_equal(ctrl.ee_target_rpy, new_rpy)

    def test_set_ee_target_then_compute(self, model, params, state, reference):
        """목표 업데이트 후 compute_control 정상 동작"""
        ctrl = WBCMPPIController(model, params, ee_target_pos=np.array([0.5, 0.3, 0.6]))
        ctrl.set_ee_target(np.array([0.8, 0.2, 0.5]))
        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (8,)

    def test_set_obstacles(self, model, params, state, reference):
        ctrl = WBCMPPIController(model, params, ee_target_pos=np.array([0.5, 0.3, 0.6]))
        ctrl.set_obstacles([(2.0, 1.0, 0.3)])
        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (8,)


# ─────────────────────────────────────────────────────────────────────────────
# Manipulation Costs 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestReachabilityWorkspaceCost:
    @pytest.fixture
    def traj_data(self):
        K, N_plus_1, nx = 8, 11, 9
        traj = np.zeros((K, N_plus_1, nx))
        ctrl = np.zeros((K, N_plus_1 - 1, 8))
        ref = np.zeros((N_plus_1, 9))
        return traj, ctrl, ref

    def test_output_shape(self, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = ReachabilityWorkspaceCost(target_pos=np.array([1.0, 1.0, 0.5]))
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (8,)

    def test_non_negative(self, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = ReachabilityWorkspaceCost(target_pos=np.array([1.0, 1.0, 0.5]))
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert np.all(costs >= 0)

    def test_within_reach_zero_cost(self, traj_data):
        traj, ctrl, ref = traj_data
        # 베이스를 목표에서 [min_reach, max_reach] 범위 내에 배치
        target = np.array([0.0, 0.0, 0.5])
        # 베이스 위치: (0.4, 0) → 목표까지 수평 거리 = 0.4 (min=0.1 < 0.4 < max=0.7)
        traj[:, :, 0] = 0.4
        traj[:, :, 1] = 0.0

        cost_fn = ReachabilityWorkspaceCost(
            target_pos=target, max_reach=0.7, min_reach=0.1
        )
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert np.all(costs == pytest.approx(0.0, abs=1e-10))

    def test_far_from_target_high_cost(self, traj_data):
        traj, ctrl, ref = traj_data
        target = np.array([0.0, 0.0, 0.0])
        traj[:, :, 0] = 5.0  # 멀리 배치

        cost_fn = ReachabilityWorkspaceCost(target_pos=target, max_reach=0.7)
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert np.all(costs > 0)


class TestArmSingularityAvoidanceCost:
    @pytest.fixture
    def traj_data(self):
        K, N_plus_1, nx = 4, 11, 9
        rng = np.random.default_rng(0)
        traj = rng.uniform(-1.0, 1.0, (K, N_plus_1, nx))
        ctrl = np.zeros((K, N_plus_1 - 1, 8))
        ref = np.zeros((N_plus_1, 9))
        return traj, ctrl, ref

    def test_output_shape(self, model, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = ArmSingularityAvoidanceCost(model, threshold=0.02)
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (4,)

    def test_non_negative(self, model, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = ArmSingularityAvoidanceCost(model)
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert np.all(costs >= 0)

    def test_large_threshold_larger_cost(self, model, traj_data):
        traj, ctrl, ref = traj_data
        cost_low = ArmSingularityAvoidanceCost(model, threshold=0.0, weight=1.0)
        cost_high = ArmSingularityAvoidanceCost(model, threshold=1e6, weight=1.0)
        c_low = cost_low.compute_cost(traj, ctrl, ref)
        c_high = cost_high.compute_cost(traj, ctrl, ref)
        assert np.all(c_high >= c_low)


class TestGraspApproachCost:
    @pytest.fixture
    def traj_data(self, model):
        K, N_plus_1, nx = 4, 11, 9
        rng = np.random.default_rng(1)
        traj = rng.uniform(-0.5, 0.5, (K, N_plus_1, nx))
        ctrl = np.zeros((K, N_plus_1 - 1, 8))
        ref = np.zeros((N_plus_1, 9))
        return traj, ctrl, ref

    def test_output_shape(self, model, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = GraspApproachCost(
            model,
            target_pos=np.array([0.5, 0.3, 0.6]),
            approach_dir=np.array([0.0, 0.0, -1.0]),
        )
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (4,)

    def test_non_negative(self, model, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = GraspApproachCost(
            model,
            target_pos=np.array([0.5, 0.3, 0.6]),
        )
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert np.all(costs >= 0)

    def test_fixed_approach_dir(self, model, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = GraspApproachCost(
            model,
            target_pos=np.array([0.5, 0.3, 0.6]),
            approach_dir=np.array([0.0, 0.0, -1.0]),  # 수직 하강
            approach_weight=10.0,
        )
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (4,)
        assert np.all(np.isfinite(costs))


class TestCollisionFreeSweepCost:
    @pytest.fixture
    def traj_data(self):
        K, N_plus_1, nx = 4, 11, 9
        traj = np.zeros((K, N_plus_1, nx))
        ctrl = np.zeros((K, N_plus_1 - 1, 8))
        ref = np.zeros((N_plus_1, 9))
        return traj, ctrl, ref

    def test_output_shape(self, model, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = CollisionFreeSweepCost(
            model,
            obstacles=[(2.0, 0.0, 0.3)],
        )
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert costs.shape == (4,)

    def test_non_negative(self, model, traj_data):
        traj, ctrl, ref = traj_data
        cost_fn = CollisionFreeSweepCost(model, obstacles=[(2.0, 0.0, 0.3)])
        costs = cost_fn.compute_cost(traj, ctrl, ref)
        assert np.all(costs >= 0)

    def test_collision_increases_cost(self, model, traj_data):
        traj_safe, ctrl, ref = traj_data
        traj_safe[:, :, :2] = 5.0  # 장애물에서 멀리

        traj_collide = traj_safe.copy()
        traj_collide[:, :, :2] = 0.0  # 장애물 근처 (장애물: (2,0,0.3) → 거리>0)

        obs = [(0.0, 0.0, 0.3)]  # 원점에 장애물
        cost_fn = CollisionFreeSweepCost(model, obstacles=obs)

        c_safe = cost_fn.compute_cost(traj_safe, ctrl, ref)
        c_collide = cost_fn.compute_cost(traj_collide, ctrl, ref)

        # 장애물 근처가 더 높은 비용
        assert np.mean(c_collide) >= np.mean(c_safe)


class TestWBCBaseNavigationCost:
    def test_output_shape(self):
        K, N, nu = 8, 10, 8
        controls = np.random.randn(K, N, nu)
        traj = np.zeros((K, N + 1, 9))
        ref = np.zeros((N + 1, 9))
        cost_fn = WBCBaseNavigationCost()
        costs = cost_fn.compute_cost(traj, controls, ref)
        assert costs.shape == (K,)

    def test_zero_base_vel_zero_cost(self):
        K, N, nu = 4, 10, 8
        controls = np.zeros((K, N, nu))
        traj = np.zeros((K, N + 1, 9))
        ref = np.zeros((N + 1, 9))
        cost_fn = WBCBaseNavigationCost()
        costs = cost_fn.compute_cost(traj, controls, ref)
        assert np.all(costs == pytest.approx(0.0))

    def test_larger_vel_larger_cost(self):
        K, N, nu = 4, 10, 8
        ctrl_slow = np.zeros((K, N, nu))
        ctrl_slow[:, :, 0] = 0.1  # v=0.1
        ctrl_fast = np.zeros((K, N, nu))
        ctrl_fast[:, :, 0] = 1.0  # v=1.0
        traj = np.zeros((K, N + 1, 9))
        ref = np.zeros((N + 1, 9))
        cost_fn = WBCBaseNavigationCost()
        assert np.all(
            cost_fn.compute_cost(traj, ctrl_fast, ref)
            > cost_fn.compute_cost(traj, ctrl_slow, ref)
        )


class TestJointVelocitySmoothCost:
    def test_output_shape(self):
        K, N, nu = 8, 10, 8
        controls = np.random.randn(K, N, nu)
        traj = np.zeros((K, N + 1, 9))
        ref = np.zeros((N + 1, 9))
        cost_fn = JointVelocitySmoothCost()
        costs = cost_fn.compute_cost(traj, controls, ref)
        assert costs.shape == (K,)

    def test_constant_vel_zero_cost(self):
        """일정 속도 → 부드러움 비용 0"""
        K, N, nu = 4, 10, 8
        controls = np.ones((K, N, nu)) * 0.5  # 일정 속도
        traj = np.zeros((K, N + 1, 9))
        ref = np.zeros((N + 1, 9))
        cost_fn = JointVelocitySmoothCost()
        costs = cost_fn.compute_cost(traj, controls, ref)
        assert np.all(costs == pytest.approx(0.0, abs=1e-10))

    def test_jerky_vel_high_cost(self):
        """진동하는 속도 → 높은 부드러움 비용"""
        K, N, nu = 4, 10, 8
        controls = np.zeros((K, N, nu))
        # 짝수/홀수 스텝 교번
        controls[:, ::2, 2:8] = 1.0
        controls[:, 1::2, 2:8] = -1.0
        traj = np.zeros((K, N + 1, 9))
        ref = np.zeros((N + 1, 9))
        cost_fn = JointVelocitySmoothCost(smooth_weight=1.0)
        costs = cost_fn.compute_cost(traj, controls, ref)
        assert np.all(costs > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 통합 테스트: WBC-MPPI 전체 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

class TestWBCMPPIIntegration:
    def test_ee_tracking_mode_full_pipeline(self, model):
        """EE 추적 모드 전체 파이프라인"""
        params = WBCMPPIParams(
            N=10, K=32,
            sigma=np.concatenate([np.array([0.3, 0.3]), np.ones(6) * 0.5]),
            Q=np.zeros(9), R=np.zeros(8),
            ee_pos_weight=100.0,
            task_mode="ee_tracking",
        )
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=np.array([0.5, 0.2, 0.6]),
            ee_target_rpy=np.array([0.0, -np.pi / 3, 0.0]),
        )
        state = np.zeros(9)
        ref = np.zeros((11, 9))
        u, info = ctrl.compute_control(state, ref)

        assert u.shape == (8,)
        assert np.all(np.isfinite(u))
        assert info["sample_weights"].shape == (32,)
        assert np.isclose(info["sample_weights"].sum(), 1.0, atol=1e-6)

    def test_both_mode_pipeline(self, model):
        """both 모드: EE 추적 + 베이스 이동 동시"""
        params = WBCMPPIParams(
            N=10, K=32,
            sigma=np.concatenate([np.array([0.5, 0.5]), np.ones(6) * 0.5]),
            Q=np.array([1.0, 1.0, 0.1] + [0.0] * 6),
            R=np.zeros(8),
            ee_pos_weight=50.0,
            task_mode="both",
        )
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=np.array([1.0, 0.5, 0.7]),
        )
        state = np.array([0.5, 0.0, 0.0, 0.1, -0.3, 0.5, 0.0, -0.2, 0.0])
        ref = np.zeros((11, 9))
        ref[:, 0] = np.linspace(0.5, 1.0, 11)

        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))

    def test_with_obstacles(self, model):
        """장애물 있는 환경에서 WBC-MPPI"""
        params = WBCMPPIParams(
            N=10, K=32,
            sigma=np.concatenate([np.array([0.3, 0.3]), np.ones(6) * 0.5]),
            Q=np.zeros(9), R=np.zeros(8),
            ee_pos_weight=80.0,
        )
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=np.array([0.5, 0.2, 0.6]),
            obstacles=[(1.0, 0.5, 0.3), (2.0, -1.0, 0.2)],
        )
        state = np.zeros(9)
        ref = np.zeros((11, 9))
        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))

    def test_sequential_calls(self, model, params):
        """연속 호출: 이전 U가 warm-start로 사용됨"""
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=np.array([0.5, 0.2, 0.6]),
        )
        state = np.zeros(9)
        ref = np.zeros((11, 9))

        u1, _ = ctrl.compute_control(state, ref)
        u2, _ = ctrl.compute_control(state, ref)

        # 두 번의 계산 모두 유효
        assert np.all(np.isfinite(u1))
        assert np.all(np.isfinite(u2))

    def test_dynamic_target_update(self, model, params):
        """동적 목표 변경 후 제어"""
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=np.array([0.5, 0.2, 0.6]),
        )
        state = np.zeros(9)
        ref = np.zeros((11, 9))

        u1, _ = ctrl.compute_control(state, ref)

        # 목표 변경
        ctrl.set_ee_target(np.array([1.0, -0.5, 0.4]), np.array([0.0, 0.0, np.pi / 2]))
        u2, info = ctrl.compute_control(state, ref)

        assert np.all(np.isfinite(u2))
        np.testing.assert_array_almost_equal(ctrl.ee_target_pos, [1.0, -0.5, 0.4])
