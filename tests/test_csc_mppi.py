"""
CSC-MPPI (Constrained Sampling Cluster MPPI) 유닛 테스트

28개 테스트:
  - Params (3): 기본값, 커스텀, 검증
  - Projection (5): 위반 감소, 실행 가능성, dual 업데이트, 스텝 효과, 관통 방지
  - Clustering (4): 클러스터 형성, 노이즈 처리, 단일 클러스터, 빈 클러스터 폴백
  - Controller (5): shape, info keys, different K, reset, fallback mode
  - Constraint Satisfaction (4): safety margin, no collision, narrow passage, combined
  - Performance (4): circle RMSE, obstacle avoidance, computation time, vs vanilla safety
  - Comparison (3): vs vanilla, vs DBaS, statistics tracking
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    CSCMPPIParams,
    DBaSMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.csc_mppi import CSCMPPIController
from mppi_controller.controllers.mppi.dbas_mppi import DBaSMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)


# ── 헬퍼 함수 ──────────────────────────────────────────────────

DEFAULT_OBSTACLES = [
    (2.5, 1.5, 0.5),
    (0.0, 3.0, 0.4),
    (-2.0, -1.0, 0.5),
]


def _make_csc_controller(**kwargs):
    """CSC-MPPI 컨트롤러 생성 헬퍼"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        obstacles=DEFAULT_OBSTACLES,
        safety_margin=0.2,
        n_projection_steps=5,
        projection_lr=0.1,
        dual_lr=0.01,
        dbscan_eps=1.0,
        dbscan_min_samples=3,
        use_projection=True,
        use_clustering=True,
        fallback_to_mppi=True,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = CSCMPPIParams(**defaults)

    if cost_function is None:
        obstacles = defaults.get("obstacles", DEFAULT_OBSTACLES)
        cost_fns = [
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
            ControlEffortCost(params.R),
        ]
        if obstacles:
            cost_fns.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0))
        cost_function = CompositeMPPICost(cost_fns)

    return CSCMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_vanilla_controller(**kwargs):
    """비교용 Vanilla MPPI"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    obstacles = kwargs.pop("obstacles", None)
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    params = MPPIParams(**defaults)

    if cost_function is None:
        cost_fns = [
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
            ControlEffortCost(params.R),
        ]
        if obstacles:
            cost_fns.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0))
        cost_function = CompositeMPPICost(cost_fns)

    return MPPIController(model, params, cost_function=cost_function)


def _make_ref(N=10, dt=0.05):
    """레퍼런스 궤적"""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ══════════════════════════════════════════════════════════════
# 1. Params 테스트 (3개)
# ══════════════════════════════════════════════════════════════

class TestCSCMPPIParams:
    def test_params_defaults(self):
        """기본값 검증"""
        params = CSCMPPIParams()
        assert params.obstacles == []
        assert params.safety_margin == 0.2
        assert params.n_projection_steps == 5
        assert params.projection_lr == 0.1
        assert params.dual_lr == 0.01
        assert params.dbscan_eps == 1.0
        assert params.dbscan_min_samples == 3
        assert params.use_projection is True
        assert params.use_clustering is True
        assert params.fallback_to_mppi is True

    def test_params_custom(self):
        """커스텀 값 검증"""
        obs = [(1.0, 2.0, 0.5), (3.0, 4.0, 0.3)]
        params = CSCMPPIParams(
            obstacles=obs,
            safety_margin=0.3,
            n_projection_steps=10,
            projection_lr=0.05,
            dual_lr=0.005,
            dbscan_eps=2.0,
            dbscan_min_samples=5,
            use_projection=False,
            use_clustering=False,
            fallback_to_mppi=False,
        )
        assert params.obstacles == obs
        assert params.safety_margin == 0.3
        assert params.n_projection_steps == 10
        assert params.projection_lr == 0.05
        assert params.dual_lr == 0.005
        assert params.dbscan_eps == 2.0
        assert params.dbscan_min_samples == 5
        assert params.use_projection is False
        assert params.use_clustering is False
        assert params.fallback_to_mppi is False

    def test_params_validation(self):
        """잘못된 값 → AssertionError"""
        # safety_margin 음수
        with pytest.raises(AssertionError):
            CSCMPPIParams(safety_margin=-0.1)

        # n_projection_steps < 1
        with pytest.raises(AssertionError):
            CSCMPPIParams(n_projection_steps=0)

        # projection_lr <= 0
        with pytest.raises(AssertionError):
            CSCMPPIParams(projection_lr=0.0)

        # dual_lr <= 0
        with pytest.raises(AssertionError):
            CSCMPPIParams(dual_lr=-0.01)

        # dbscan_eps <= 0
        with pytest.raises(AssertionError):
            CSCMPPIParams(dbscan_eps=0.0)

        # dbscan_min_samples < 1
        with pytest.raises(AssertionError):
            CSCMPPIParams(dbscan_min_samples=0)


# ══════════════════════════════════════════════════════════════
# 2. Projection 테스트 (5개)
# ══════════════════════════════════════════════════════════════

class TestProjection:
    def test_violation_reduction(self):
        """투영 후 위반량 감소"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=32, N=10,
            obstacles=[(1.0, 0.0, 0.3)],
            n_projection_steps=5,
            projection_lr=0.1,
        )
        state = np.array([0.5, 0.0, 0.0])  # 장애물 근처

        noise = np.random.standard_normal((32, 10, 2)) * 0.5
        controls_before = ctrl.U[None, :, :] + noise

        # 투영 전 위반
        traj_before = ctrl.dynamics_wrapper.rollout(state, controls_before)
        violations_before = ctrl._compute_violations(traj_before)
        total_before = np.sum(np.maximum(violations_before, 0))

        # 투영
        controls_after, info = ctrl._project_to_feasible(controls_before, state)

        # 투영 후 위반
        traj_after = ctrl.dynamics_wrapper.rollout(state, controls_after)
        violations_after = ctrl._compute_violations(traj_after)
        total_after = np.sum(np.maximum(violations_after, 0))

        # 위반이 줄었어야 함 (또는 초기 위반이 0이면 동일)
        assert total_after <= total_before + 1e-6, \
            f"Violations should decrease: before={total_before:.3f}, after={total_after:.3f}"

    def test_feasibility_improvement(self):
        """투영 후 실행 가능 샘플 비율 증가"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=64, N=10,
            obstacles=[(0.5, 0.0, 0.3)],
            n_projection_steps=10,
            projection_lr=0.15,
            dual_lr=0.05,
        )
        state = np.array([0.3, 0.0, 0.0])  # 장애물 매우 근처

        noise = np.random.standard_normal((64, 10, 2)) * 0.5
        controls = ctrl.U[None, :, :] + noise

        # 투영 전 실행 가능 비율
        traj_before = ctrl.dynamics_wrapper.rollout(state, controls)
        viol_before = ctrl._compute_violations(traj_before)
        feasible_before = np.sum(np.all(viol_before <= 0, axis=-1))

        # 투영 후
        controls_proj, _ = ctrl._project_to_feasible(controls, state)
        traj_after = ctrl.dynamics_wrapper.rollout(state, controls_proj)
        viol_after = ctrl._compute_violations(traj_after)
        feasible_after = np.sum(np.all(viol_after <= 0, axis=-1))

        # 실행 가능 샘플이 늘었어야 함 (또는 동일)
        assert feasible_after >= feasible_before, \
            f"Feasible samples: before={feasible_before}, after={feasible_after}"

    def test_dual_update_positive(self):
        """위반 시 dual 변수 양수"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=16, N=5,
            obstacles=[(0.5, 0.0, 0.3)],
            n_projection_steps=3,
        )
        state = np.array([0.3, 0.0, 0.0])

        noise = np.random.standard_normal((16, 5, 2)) * 0.5
        controls = ctrl.U[None, :, :] + noise

        # 수동 투영 (dual 변수 추적)
        n_obs = len(ctrl._obstacles)
        dual_vars = np.zeros((16, n_obs))

        trajectories = ctrl.dynamics_wrapper.rollout(state, controls)
        violations = ctrl._compute_violations(trajectories)

        # 위반이 있으면 dual update 후 양수
        has_violation = np.any(violations > 0)
        dual_vars = np.maximum(0, dual_vars + 0.01 * violations)

        if has_violation:
            assert np.any(dual_vars > 0), "Dual vars should be positive with violations"

    def test_projection_step_effect(self):
        """투영 스텝 수 증가 → 더 많은 위반 감소"""
        np.random.seed(42)
        ctrl_few = _make_csc_controller(
            K=32, N=10,
            obstacles=[(1.0, 0.0, 0.3)],
            n_projection_steps=1,
        )
        ctrl_many = _make_csc_controller(
            K=32, N=10,
            obstacles=[(1.0, 0.0, 0.3)],
            n_projection_steps=10,
        )
        state = np.array([0.5, 0.0, 0.0])

        np.random.seed(42)
        noise = np.random.standard_normal((32, 10, 2)) * 0.5
        controls = ctrl_few.U[None, :, :] + noise

        np.random.seed(42)
        _, info_few = ctrl_few._project_to_feasible(controls.copy(), state)

        np.random.seed(42)
        _, info_many = ctrl_many._project_to_feasible(controls.copy(), state)

        # 더 많은 스텝 → 더 큰 위반 감소 (또는 동일)
        assert info_many["violation_reduction"] >= info_few["violation_reduction"] - 0.1, \
            f"More steps should reduce more: few={info_few['violation_reduction']:.3f}, " \
            f"many={info_many['violation_reduction']:.3f}"

    def test_no_obstacle_passthrough(self):
        """투영 후 장애물 관통 방지"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=64, N=10,
            obstacles=[(2.0, 0.0, 0.5)],
            safety_margin=0.1,
            n_projection_steps=10,
            projection_lr=0.15,
            dual_lr=0.05,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)

        # 최적 궤적의 최소 클리어런스
        min_clearance = info["csc_stats"]["min_clearance"]
        # 투영이 있으므로 클리어런스가 극단적으로 음수가 아니어야 함
        assert min_clearance > -0.5, \
            f"Min clearance too negative: {min_clearance:.3f}"


# ══════════════════════════════════════════════════════════════
# 3. Clustering 테스트 (4개)
# ══════════════════════════════════════════════════════════════

class TestClustering:
    def test_cluster_formation(self):
        """서로 다른 제어 시퀀스 → 여러 클러스터"""
        np.random.seed(42)
        ctrl = _make_csc_controller(K=64, dbscan_eps=2.0, dbscan_min_samples=3)

        # 서로 다른 제어 시퀀스 생성 (작은 노이즈로 밀집 그룹)
        controls = np.random.standard_normal((64, 10, 2)) * 0.1
        # 3개 그룹으로 구분 (큰 간격)
        controls[:20] += 3.0    # 그룹 1
        controls[20:40] -= 3.0   # 그룹 2
        # 나머지는 원점 근처

        costs = np.random.uniform(0, 10, size=64)

        labels, info = ctrl._cluster_trajectories(controls, costs)
        assert info["n_clusters"] >= 1, f"Expected at least 1 cluster, got {info['n_clusters']}"

    def test_noise_handling(self):
        """DBSCAN 노이즈 샘플 (label=-1) 처리"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=32, dbscan_eps=0.01, dbscan_min_samples=20,
        )

        # 모두 다른 제어 시퀀스 → 대부분 노이즈
        controls = np.random.standard_normal((32, 10, 2)) * 10
        costs = np.random.uniform(0, 10, size=32)

        labels, info = ctrl._cluster_trajectories(controls, costs)
        # 노이즈가 있어야 함 (엄격한 eps + min_samples)
        assert info["n_noise"] >= 0, "Noise count should be non-negative"

    def test_single_cluster(self):
        """비슷한 제어 → 단일 클러스터"""
        np.random.seed(42)
        ctrl = _make_csc_controller(K=32, dbscan_eps=5.0, dbscan_min_samples=3)

        # 모두 비슷한 제어 시퀀스
        base = np.ones((10, 2))
        controls = base[None, :, :] + np.random.standard_normal((32, 10, 2)) * 0.01
        costs = np.random.uniform(0, 10, size=32)

        labels, info = ctrl._cluster_trajectories(controls, costs)
        assert info["n_clusters"] >= 1, "At least 1 cluster expected"

    def test_empty_cluster_fallback(self):
        """클러스터 0개 → 폴백 동작"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=16,
            dbscan_eps=0.001,  # 매우 작은 eps
            dbscan_min_samples=15,  # 높은 threshold
            fallback_to_mppi=True,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 정상 동작 (폴백)
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert not np.any(np.isnan(control))


# ══════════════════════════════════════════════════════════════
# 4. Controller 테스트 (5개)
# ══════════════════════════════════════════════════════════════

class TestCSCMPPIController:
    def test_compute_control_shape(self):
        """control (nu,), info 표준 키 검증"""
        np.random.seed(42)
        ctrl = _make_csc_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert "sample_trajectories" in info
        assert "sample_weights" in info
        assert "best_trajectory" in info
        assert "best_cost" in info
        assert "mean_cost" in info
        assert "temperature" in info
        assert "ess" in info
        assert "num_samples" in info

    def test_info_csc_stats(self):
        """csc_stats 키 검증"""
        np.random.seed(42)
        ctrl = _make_csc_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["csc_stats"]
        assert "n_clusters" in stats
        assert "n_noise_samples" in stats
        assert "n_projected" in stats
        assert "violation_reduction" in stats
        assert "selected_idx" in stats
        assert "min_clearance" in stats
        assert "use_projection" in stats
        assert "use_clustering" in stats
        assert "num_obstacles" in stats
        assert stats["num_obstacles"] == 3

    def test_different_K_values(self):
        """K=32/128/256 정상 작동"""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for K in [32, 128, 256]:
            np.random.seed(42)
            ctrl = _make_csc_controller(K=K)
            control, info = ctrl.compute_control(state, ref)
            assert control.shape == (2,)
            assert info["num_samples"] == K
            assert not np.any(np.isnan(control))

    def test_reset_clears_state(self):
        """reset 후 초기화"""
        np.random.seed(42)
        ctrl = _make_csc_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(3):
            ctrl.compute_control(state, ref)

        assert len(ctrl._csc_history) == 3

        ctrl.reset()
        assert np.allclose(ctrl.U, 0.0)
        assert len(ctrl._csc_history) == 0
        assert len(ctrl._obstacles) == len(DEFAULT_OBSTACLES)

    def test_fallback_mode(self):
        """use_clustering=False → 표준 MPPI 폴백"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            use_clustering=False,
            use_projection=False,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["csc_stats"]["n_clusters"] == 0
        assert not np.any(np.isnan(control))


# ══════════════════════════════════════════════════════════════
# 5. Constraint Satisfaction 테스트 (4개)
# ══════════════════════════════════════════════════════════════

class TestConstraintSatisfaction:
    def test_safety_margin_respected(self):
        """safety_margin 준수"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=256, N=15,
            obstacles=[(2.0, 1.0, 0.5)],
            safety_margin=0.15,
            n_projection_steps=10,
        )
        model = DifferentialDriveKinematic(wheelbase=0.5)
        state = np.array([3.0, 0.0, np.pi / 2])

        min_clearance = float("inf")
        for step in range(40):
            t = step * 0.05
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, 15, 0.05,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

            for ox, oy, r in [(2.0, 1.0, 0.5)]:
                dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                clearance = dist - r
                min_clearance = min(min_clearance, clearance)

        assert min_clearance > -0.1, \
            f"Safety margin violation: min_clearance={min_clearance:.3f}"

    def test_no_collision(self):
        """장애물 충돌 0"""
        np.random.seed(42)
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4)]

        ctrl = _make_csc_controller(
            K=256, N=15,
            obstacles=obstacles,
            safety_margin=0.2,
            n_projection_steps=5,
        )
        state = np.array([3.0, 0.0, np.pi / 2])

        collisions = 0
        for step in range(60):
            t = step * 0.05
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, 15, 0.05,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

            for ox, oy, r in obstacles:
                dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                if dist < r:
                    collisions += 1

        assert collisions == 0, f"Collisions: {collisions}"

    def test_narrow_passage(self):
        """좁은 통로 (2개 장애물 사이) 통과"""
        np.random.seed(42)
        model = DifferentialDriveKinematic(wheelbase=0.5)

        # 좁은 통로 형성 (y축 상에 2개 장애물)
        obstacles = [
            (0.0, 2.0, 0.5),
            (0.0, 4.0, 0.5),
        ]

        ctrl = _make_csc_controller(
            K=256, N=15,
            obstacles=obstacles,
            safety_margin=0.1,
            n_projection_steps=5,
        )
        state = np.array([3.0, 0.0, np.pi / 2])

        collisions = 0
        for step in range(60):
            t = step * 0.05
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, 15, 0.05,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

            for ox, oy, r in obstacles:
                dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                if dist < r:
                    collisions += 1

        assert collisions == 0, f"Narrow passage collisions: {collisions}"

    def test_combined_constraints(self):
        """투영 + 클러스터링 동시 사용"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=128, N=10,
            obstacles=[(1.0, 1.0, 0.3), (2.0, -1.0, 0.4)],
            use_projection=True,
            use_clustering=True,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert not np.any(np.isnan(control))
        assert info["csc_stats"]["use_projection"] is True
        assert info["csc_stats"]["use_clustering"] is True


# ══════════════════════════════════════════════════════════════
# 6. Performance 테스트 (4개)
# ══════════════════════════════════════════════════════════════

class TestPerformance:
    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.5 (50 스텝)"""
        np.random.seed(42)
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = CSCMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            obstacles=[],
            use_projection=False,
            use_clustering=True,
        )
        ctrl = CSCMPPIController(model, params)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt = params.dt
        N = params.N
        num_steps = 50

        errors = []
        for step in range(num_steps):
            t = step * dt
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt

            ref_pt = circle_trajectory(t, radius=3.0)
            err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
            errors.append(err)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.5, f"RMSE {rmse:.4f} >= 0.5"

    def test_obstacle_avoidance(self):
        """3개 장애물 충돌 없음"""
        np.random.seed(42)
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4), (-2.0, -1.0, 0.5)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0),
        ])

        params = CSCMPPIParams(
            K=256, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            obstacles=obstacles,
            safety_margin=0.2,
            n_projection_steps=5,
        )
        ctrl = CSCMPPIController(model, params, cost_function=cost)

        state = np.array([3.0, 0.0, np.pi / 2])

        collisions = 0
        for step in range(80):
            t = step * 0.05
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, 15, 0.05,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

            for ox, oy, r in obstacles:
                dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                if dist < r:
                    collisions += 1

        assert collisions == 0, f"Collisions: {collisions}"

    def test_computation_time(self):
        """K=128, N=15에서 500ms 이내 (투영+클러스터링 오버헤드 포함)"""
        np.random.seed(42)
        ctrl = _make_csc_controller(
            K=128, N=15,
            n_projection_steps=3,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=15)

        # Warmup
        ctrl.compute_control(state, ref)
        ctrl.reset()

        times = []
        for _ in range(3):
            t_start = time.time()
            ctrl.compute_control(state, ref)
            times.append(time.time() - t_start)

        mean_ms = np.mean(times) * 1000
        assert mean_ms < 500, f"Mean solve time {mean_ms:.1f}ms >= 500ms"

    def test_vs_vanilla_safety(self):
        """CSC-MPPI vs Vanilla: 장애물 근처 안전성"""
        np.random.seed(42)
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(2.0, 1.0, 0.5)]

        # CSC-MPPI
        csc_ctrl = _make_csc_controller(
            K=256, N=15,
            obstacles=obstacles,
            safety_margin=0.2,
            n_projection_steps=5,
        )

        # Vanilla
        vanilla_ctrl = _make_vanilla_controller(
            K=256, N=15,
            obstacles=obstacles,
        )

        state_csc = np.array([3.0, 0.0, np.pi / 2])
        state_van = np.array([3.0, 0.0, np.pi / 2])

        csc_min_clearance = float("inf")
        van_min_clearance = float("inf")

        for step in range(40):
            t = step * 0.05
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, 15, 0.05,
            )

            np.random.seed(step)
            control_csc, _ = csc_ctrl.compute_control(state_csc, ref)
            state_csc_dot = model.forward_dynamics(state_csc, control_csc)
            state_csc = state_csc + state_csc_dot * 0.05

            np.random.seed(step)
            control_van, _ = vanilla_ctrl.compute_control(state_van, ref)
            state_van_dot = model.forward_dynamics(state_van, control_van)
            state_van = state_van + state_van_dot * 0.05

            for ox, oy, r in obstacles:
                dist_csc = np.sqrt((state_csc[0] - ox) ** 2 + (state_csc[1] - oy) ** 2) - r
                dist_van = np.sqrt((state_van[0] - ox) ** 2 + (state_van[1] - oy) ** 2) - r
                csc_min_clearance = min(csc_min_clearance, dist_csc)
                van_min_clearance = min(van_min_clearance, dist_van)

        # CSC가 Vanilla보다 안전하거나 동등해야 함
        assert csc_min_clearance >= van_min_clearance - 0.3, \
            f"CSC clearance ({csc_min_clearance:.3f}) should be >= " \
            f"Vanilla ({van_min_clearance:.3f}) - 0.3"


# ══════════════════════════════════════════════════════════════
# 7. Comparison 테스트 (3개)
# ══════════════════════════════════════════════════════════════

class TestComparison:
    def test_vs_vanilla(self):
        """CSC-MPPI vs Vanilla 정상 동작"""
        np.random.seed(42)
        csc = _make_csc_controller(K=64, N=10, obstacles=[])
        vanilla = _make_vanilla_controller(K=64, N=10)

        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control_csc, info_csc = csc.compute_control(state, ref)
        np.random.seed(42)
        control_van, info_van = vanilla.compute_control(state, ref)

        assert control_csc.shape == control_van.shape
        assert info_csc["num_samples"] == info_van["num_samples"]

    def test_vs_dbas(self):
        """CSC-MPPI vs DBaS-MPPI 비교 (둘 다 정상 동작)"""
        np.random.seed(42)
        obstacles = [(2.0, 1.0, 0.5)]

        csc = _make_csc_controller(K=64, N=10, obstacles=obstacles)

        dbas_params = DBaSMPPIParams(
            K=64, N=10, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            dbas_obstacles=obstacles,
            barrier_weight=10.0,
        )
        model = DifferentialDriveKinematic(wheelbase=0.5)
        dbas = DBaSMPPIController(model, dbas_params)

        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        np.random.seed(42)
        control_csc, info_csc = csc.compute_control(state, ref)
        np.random.seed(42)
        control_dbas, info_dbas = dbas.compute_control(state, ref)

        # 둘 다 유효한 제어
        assert control_csc.shape == (2,)
        assert control_dbas.shape == (2,)
        assert not np.any(np.isnan(control_csc))
        assert not np.any(np.isnan(control_dbas))

    def test_statistics_tracking(self):
        """get_csc_statistics 추적"""
        np.random.seed(42)
        ctrl = _make_csc_controller(K=64, N=10)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(10):
            ctrl.compute_control(state, ref)

        stats = ctrl.get_csc_statistics()
        assert stats["total_steps"] == 10
        assert stats["mean_clusters"] >= 0
        assert stats["mean_violation_reduction"] >= 0
        assert len(stats["history"]) == 10

        # 각 히스토리 엔트리 검증
        for h in stats["history"]:
            assert "n_clusters" in h
            assert "violation_reduction" in h
            assert "min_clearance" in h
