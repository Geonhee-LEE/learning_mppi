"""
DBaS-MPPI (Discrete Barrier States MPPI) 유닛 테스트
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    DBaSMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dbas_mppi import DBaSMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


# ── 헬퍼 ─────────────────────────────────────────────────────

DEFAULT_OBSTACLES = [
    (2.5, 2.0, 0.4),
    (-1.5, 3.0, 0.5),
    (1.0, -3.0, 0.3),
]

DEFAULT_WALLS = [
    ('x', -5.0, 1),   # x >= -5
    ('x', 5.0, -1),   # x <= 5
]


def _make_dbas_controller(**kwargs):
    """헬퍼: DBaS-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        dbas_obstacles=DEFAULT_OBSTACLES,
        dbas_walls=[],
        barrier_weight=10.0,
        barrier_gamma=0.5,
        exploration_coeff=1.0,
        h_min=1e-6,
        safety_margin=0.1,
        use_adaptive_exploration=True,
    )
    defaults.update(kwargs)
    params = DBaSMPPIParams(**defaults)
    return DBaSMPPIController(model, params)


def _make_vanilla_controller(**kwargs):
    """헬퍼: Vanilla MPPI 컨트롤러 생성 (장애물 ObstacleCost 사용)"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    obstacles = kwargs.pop("obstacles", None)
    defaults.update(kwargs)
    params = MPPIParams(**defaults)

    cost_fns = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        cost_fns.append(ObstacleCost(obstacles, safety_margin=0.1, cost_weight=2000.0))
    cost = CompositeMPPICost(cost_fns)

    return MPPIController(model, params, cost_function=cost)


def _make_ref(N=10, dt=0.05):
    """헬퍼: 레퍼런스 궤적 생성"""
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


# ── Params Tests (3) ─────────────────────────────────────────


def test_params_defaults():
    """기본값 검증"""
    params = DBaSMPPIParams()
    assert params.dbas_obstacles == []
    assert params.dbas_walls == []
    assert params.barrier_weight == 10.0
    assert params.barrier_gamma == 0.5
    assert params.exploration_coeff == 1.0
    assert params.h_min == 1e-6
    assert params.safety_margin == 0.1
    assert params.use_adaptive_exploration is True


def test_params_custom():
    """커스텀 값"""
    obs = [(1.0, 2.0, 0.5)]
    walls = [('x', 3.0, 1)]
    params = DBaSMPPIParams(
        dbas_obstacles=obs,
        dbas_walls=walls,
        barrier_weight=20.0,
        barrier_gamma=0.3,
        exploration_coeff=2.0,
        h_min=1e-4,
        safety_margin=0.2,
        use_adaptive_exploration=False,
    )
    assert params.dbas_obstacles == obs
    assert params.dbas_walls == walls
    assert params.barrier_weight == 20.0
    assert params.barrier_gamma == 0.3
    assert params.exploration_coeff == 2.0
    assert params.h_min == 1e-4
    assert params.safety_margin == 0.2
    assert params.use_adaptive_exploration is False


def test_params_validation():
    """잘못된 값 -> AssertionError"""
    # gamma=0
    try:
        DBaSMPPIParams(barrier_gamma=0.0)
        assert False, "Should have raised AssertionError for gamma=0"
    except AssertionError:
        pass

    # gamma=1
    try:
        DBaSMPPIParams(barrier_gamma=1.0)
        assert False, "Should have raised AssertionError for gamma=1"
    except AssertionError:
        pass

    # h_min=0
    try:
        DBaSMPPIParams(h_min=0.0)
        assert False, "Should have raised AssertionError for h_min=0"
    except AssertionError:
        pass

    # negative barrier_weight
    try:
        DBaSMPPIParams(barrier_weight=-1.0)
        assert False, "Should have raised AssertionError for negative barrier_weight"
    except AssertionError:
        pass


# ── Barrier Function Tests (5) ──────────────────────────────


def test_constraint_values_circle():
    """원형 h: 외부 > 0, 내부 < 0"""
    ctrl = _make_dbas_controller(
        dbas_obstacles=[(0.0, 0.0, 1.0)], safety_margin=0.0
    )

    # 외부 (거리 3, 반경 1 -> h = 9 - 1 = 8)
    outside = np.array([[3.0, 0.0]])
    h_out = ctrl._compute_constraint_values(outside)
    assert h_out[0, 0] > 0, f"Outside h should be > 0, got {h_out[0, 0]}"

    # 내부 (거리 0.5, 반경 1 -> h = 0.25 - 1 = -0.75)
    inside = np.array([[0.5, 0.0]])
    h_in = ctrl._compute_constraint_values(inside)
    assert h_in[0, 0] < 0, f"Inside h should be < 0, got {h_in[0, 0]}"


def test_constraint_values_wall():
    """벽 h: 안전 방향 > 0, 위반 < 0"""
    ctrl = _make_dbas_controller(
        dbas_obstacles=[], dbas_walls=[('x', 2.0, -1)]  # x <= 2
    )

    # 안전 (x=1.0 <= 2.0 -> h = -1*(1 - 2) = 1 > 0)
    safe = np.array([[1.0, 0.0]])
    h_safe = ctrl._compute_constraint_values(safe)
    assert h_safe[0, 0] > 0, f"Safe wall h should be > 0, got {h_safe[0, 0]}"

    # 위반 (x=3.0 > 2.0 -> h = -1*(3 - 2) = -1 < 0)
    violate = np.array([[3.0, 0.0]])
    h_viol = ctrl._compute_constraint_values(violate)
    assert h_viol[0, 0] < 0, f"Violated wall h should be < 0, got {h_viol[0, 0]}"


def test_barrier_function_safe():
    """h >> 0 -> B ~ 0"""
    ctrl = _make_dbas_controller()
    h = np.array([100.0])
    B = ctrl._barrier_function(h)
    assert B[0] < 0.1, f"Barrier for safe h should be small, got {B[0]}"


def test_barrier_function_boundary():
    """h -> 0 -> B 큰 값"""
    ctrl = _make_dbas_controller(h_min=1e-6)
    h = np.array([1e-5])
    B = ctrl._barrier_function(h)
    assert B[0] > 5.0, f"Barrier near boundary should be large, got {B[0]}"


def test_barrier_function_clipping():
    """h < h_min -> -log(h_min)"""
    ctrl = _make_dbas_controller(h_min=1e-4)
    h_neg = np.array([-1.0])
    h_tiny = np.array([1e-8])

    B_neg = ctrl._barrier_function(h_neg)
    B_tiny = ctrl._barrier_function(h_tiny)
    expected = -np.log(1e-4)

    assert np.isclose(B_neg[0], expected, rtol=1e-6), \
        f"B(h<0) should be -log(h_min), got {B_neg[0]}, expected {expected}"
    assert np.isclose(B_tiny[0], expected, rtol=1e-6), \
        f"B(h<<h_min) should be -log(h_min), got {B_tiny[0]}, expected {expected}"


# ── Barrier Cost Tests (4) ──────────────────────────────────


def test_barrier_cost_shape():
    """barrier_costs shape = (K,)"""
    np.random.seed(42)
    ctrl = _make_dbas_controller(K=32, N=8)
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref(N=8)

    # Rollout 수동 생성
    noise = np.random.standard_normal((32, 8, 2)) * 0.5
    controls = ctrl.U[None, :, :] + noise
    trajectories = ctrl.dynamics_wrapper.rollout(state, controls)

    costs, states = ctrl._compute_barrier_cost(trajectories, ref)
    assert costs.shape == (32,), f"barrier costs shape {costs.shape}"
    assert states.shape[0] == 32
    assert states.shape[1] == 9  # N+1


def test_barrier_cost_safe_traj():
    """장애물 없는 곳 -> barrier cost ~ 0"""
    ctrl = _make_dbas_controller(dbas_obstacles=[], dbas_walls=[])
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    K, N = 16, 10
    controls = np.zeros((K, N, 2))
    trajectories = ctrl.dynamics_wrapper.rollout(state, controls)

    costs, _ = ctrl._compute_barrier_cost(trajectories, ref)
    assert np.allclose(costs, 0.0), \
        f"No obstacles -> barrier cost should be 0, got {costs}"


def test_barrier_cost_near_obstacle():
    """장애물 근처 -> barrier cost 큰 값"""
    np.random.seed(42)
    # 장애물이 원점에 있고 로봇이 바로 옆에서 시작
    ctrl = _make_dbas_controller(
        dbas_obstacles=[(0.0, 0.0, 0.5)],
        safety_margin=0.1,
        barrier_weight=10.0,
        K=32, N=10,
    )
    state = np.array([0.7, 0.0, 0.0])  # 장애물 바로 옆
    ref = _make_ref()

    controls = np.zeros((32, 10, 2)) + 0.01
    trajectories = ctrl.dynamics_wrapper.rollout(state, controls)

    costs_near, _ = ctrl._compute_barrier_cost(trajectories, ref)

    # 멀리서 시작
    state_far = np.array([10.0, 10.0, 0.0])
    trajectories_far = ctrl.dynamics_wrapper.rollout(state_far, controls)
    costs_far, _ = ctrl._compute_barrier_cost(trajectories_far, ref)

    assert np.mean(costs_near) > np.mean(costs_far), \
        f"Near obstacle cost ({np.mean(costs_near):.2f}) should > far ({np.mean(costs_far):.2f})"


def test_barrier_cost_weight_scale():
    """RB * 2 -> cost 비례 증가"""
    np.random.seed(42)
    ctrl1 = _make_dbas_controller(barrier_weight=10.0, K=32, N=10)
    ctrl2 = _make_dbas_controller(barrier_weight=20.0, K=32, N=10)

    state = np.array([2.0, 1.5, 0.0])
    ref = _make_ref()

    controls = np.random.standard_normal((32, 10, 2)) * 0.3
    traj = ctrl1.dynamics_wrapper.rollout(state, controls)

    costs1, _ = ctrl1._compute_barrier_cost(traj, ref)
    costs2, _ = ctrl2._compute_barrier_cost(traj, ref)

    # costs2 should be ~2x costs1 (same trajectories, different weight)
    ratio = np.mean(costs2) / (np.mean(costs1) + 1e-10)
    assert 1.5 < ratio < 2.5, \
        f"Weight doubling should ~double cost, ratio = {ratio:.2f}"


# ── Controller Basic Tests (5) ──────────────────────────────


def test_compute_control_shape():
    """control (nu,), info keys 확인"""
    np.random.seed(42)
    ctrl = _make_dbas_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"control shape: {control.shape}"
    assert isinstance(info, dict)

    required_keys = [
        "sample_trajectories", "sample_weights", "best_trajectory",
        "best_cost", "mean_cost", "temperature", "ess", "num_samples",
        "dbas_stats", "barrier_costs", "barrier_states",
    ]
    for key in required_keys:
        assert key in info, f"missing key: {key}"

    assert info["sample_trajectories"].shape == (64, 11, 3)
    assert info["sample_weights"].shape == (64,)


def test_compute_control_no_obstacle():
    """장애물 없음 -> 정상 동작"""
    np.random.seed(42)
    ctrl = _make_dbas_controller(dbas_obstacles=[], dbas_walls=[])
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(control)), "NaN in control"
    assert not np.any(np.isinf(control)), "Inf in control"
    assert info["dbas_stats"]["barrier_cost_best"] == 0.0


def test_info_dbas_stats():
    """dbas_stats 키/값 확인"""
    np.random.seed(42)
    ctrl = _make_dbas_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)
    stats = info["dbas_stats"]

    required_stats_keys = [
        "adaptive_scale", "sigma_eff", "barrier_cost_mean",
        "barrier_cost_best", "min_constraint", "num_obstacles", "num_walls",
    ]
    for key in required_stats_keys:
        assert key in stats, f"missing dbas_stats key: {key}"

    assert stats["num_obstacles"] == 3
    assert stats["num_walls"] == 0
    assert isinstance(stats["adaptive_scale"], float)
    assert isinstance(stats["sigma_eff"], list)


def test_adaptive_scale_near_obs():
    """장애물 근처 -> adaptive_scale > 0"""
    np.random.seed(42)
    ctrl = _make_dbas_controller(
        dbas_obstacles=[(1.0, 0.0, 0.3)],
        exploration_coeff=1.0,
        use_adaptive_exploration=True,
    )
    # 장애물 바로 옆에서 시작
    state = np.array([0.5, 0.0, 0.0])
    ref = _make_ref()

    ctrl.compute_control(state, ref)
    assert ctrl._adaptive_scale > 0.5, \
        f"Near obstacle, adaptive_scale should be > 0.5, got {ctrl._adaptive_scale}"


def test_adaptive_scale_free_space():
    """장애물 멀리 -> adaptive_scale ~ mu (최소값)"""
    np.random.seed(42)
    ctrl = _make_dbas_controller(
        dbas_obstacles=[(100.0, 100.0, 0.3)],  # 매우 먼 장애물
        exploration_coeff=1.0,
        use_adaptive_exploration=True,
    )
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    ctrl.compute_control(state, ref)
    # 장애물이 멀면 barrier cost ~ 0 -> Se = mu * log(e + 0) = mu * 1
    assert ctrl._adaptive_scale < 2.0, \
        f"Far from obstacle, adaptive_scale should be small, got {ctrl._adaptive_scale}"


# ── Performance Tests (4) ───────────────────────────────────


def test_circle_tracking_rmse():
    """50스텝 원형 추적, RMSE < 2.0m"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    ctrl = _make_dbas_controller(K=128, N=15)
    state = np.array([5.0, 0.0, np.pi / 2])
    errors = []

    for step in range(50):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)
        error = np.linalg.norm(state[:2] - ref[0, :2])
        errors.append(error)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    assert rmse < 2.0, f"DBaS RMSE too high: {rmse:.4f}"


def test_obstacle_avoidance():
    """장애물 회피: 충돌 0"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    obstacles = [(3.0, 0.5, 0.5), (0.0, 3.0, 0.5)]
    ctrl = _make_dbas_controller(
        dbas_obstacles=obstacles,
        barrier_weight=50.0,
        K=256, N=15,
        safety_margin=0.1,
    )
    state = np.array([5.0, 0.0, np.pi / 2])

    collisions = 0
    for step in range(60):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)

        # 충돌 체크
        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            if dist < r:
                collisions += 1

    assert collisions == 0, f"DBaS had {collisions} collisions"


def test_wall_avoidance():
    """벽 제약 회피"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    walls = [('y', 4.0, -1)]  # y <= 4
    ctrl = _make_dbas_controller(
        dbas_obstacles=[],
        dbas_walls=walls,
        barrier_weight=50.0,
        K=128, N=15,
        safety_margin=0.1,
    )
    state = np.array([5.0, 0.0, np.pi / 2])

    max_y = -float('inf')
    for step in range(50):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)
        max_y = max(max_y, state[1])

    # 벽 제약이 있으면 y가 크게 넘지 않아야 함
    assert max_y < 5.5, f"Wall avoidance: max_y = {max_y:.2f} (wall at 4.0)"


def test_narrow_corridor():
    """좁은 통로 통과: 충돌 0"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # 벽으로 형성된 좁은 통로 (y=1.5 ~ y=3.5, 폭 2.0)
    walls = [
        ('y', 1.5, 1),   # y >= 1.5
        ('y', 3.5, -1),  # y <= 3.5
    ]
    ctrl = _make_dbas_controller(
        dbas_obstacles=[],
        dbas_walls=walls,
        barrier_weight=50.0,
        K=128, N=10,
        safety_margin=0.05,
    )

    # 통로 안에서 시작
    state = np.array([0.0, 2.5, 0.0])

    # 직진 레퍼런스 (통로 안에서 x 방향 이동)
    def straight_ref(t):
        return np.array([t * 0.5, 2.5, 0.0])

    wall_violations = 0
    for step in range(40):
        t = step * 0.05
        ref = generate_reference_trajectory(straight_ref, t, 10, 0.05)
        control, _ = ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)

        if state[1] < 1.3 or state[1] > 3.7:
            wall_violations += 1

    assert wall_violations == 0, f"Corridor violations: {wall_violations}"


# ── Comparison Tests (4) ────────────────────────────────────


def test_dbas_vs_vanilla_obstacle():
    """Vanilla 충돌 가능 vs DBaS 회피 검증"""
    obstacles = [(2.0, 1.0, 0.5)]
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # DBaS
    np.random.seed(42)
    dbas = _make_dbas_controller(
        dbas_obstacles=obstacles,
        barrier_weight=50.0,
        K=256, N=15,
    )
    state_dbas = np.array([5.0, 0.0, np.pi / 2])
    dbas_min_clearance = float('inf')

    for step in range(60):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = dbas.compute_control(state_dbas, ref)
        state_dbas = model.step(state_dbas, control, 0.05)
        for ox, oy, r in obstacles:
            clearance = np.sqrt((state_dbas[0] - ox) ** 2 + (state_dbas[1] - oy) ** 2) - r
            dbas_min_clearance = min(dbas_min_clearance, clearance)

    # DBaS는 최소 클리어런스가 양수여야 함
    assert dbas_min_clearance > -0.1, \
        f"DBaS min clearance too negative: {dbas_min_clearance:.3f}"


def test_adaptive_vs_fixed_noise():
    """use_adaptive_exploration True vs False"""
    np.random.seed(42)
    ctrl_adaptive = _make_dbas_controller(
        use_adaptive_exploration=True,
        exploration_coeff=1.0,
    )
    state = np.array([2.0, 1.5, 0.0])
    ref = _make_ref()

    ctrl_adaptive.compute_control(state, ref)
    scale_adaptive = ctrl_adaptive._adaptive_scale

    np.random.seed(42)
    ctrl_fixed = _make_dbas_controller(
        use_adaptive_exploration=False,
        exploration_coeff=1.0,
    )
    ctrl_fixed.compute_control(state, ref)
    scale_fixed = ctrl_fixed._adaptive_scale

    # 적응적: scale > 0, 고정: scale = 0
    assert scale_adaptive > 0, f"Adaptive scale should be > 0, got {scale_adaptive}"
    assert scale_fixed == 0.0, f"Fixed scale should be 0, got {scale_fixed}"


def test_dynamic_obstacle_update():
    """update_obstacles 후 새 장애물 회피"""
    np.random.seed(42)
    ctrl = _make_dbas_controller(
        dbas_obstacles=[],
        barrier_weight=50.0,
        K=128, N=10,
    )
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    # 장애물 없이 한 번 실행
    _, info1 = ctrl.compute_control(state, ref)
    assert info1["dbas_stats"]["num_obstacles"] == 0

    # 장애물 추가
    ctrl.update_obstacles([(0.5, 0.5, 0.3)])
    _, info2 = ctrl.compute_control(state, ref)
    assert info2["dbas_stats"]["num_obstacles"] == 1
    assert info2["dbas_stats"]["barrier_cost_mean"] > 0


def test_different_K_values():
    """K=16/64/128 모두 정상"""
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for K in [16, 64, 128]:
        np.random.seed(42)
        ctrl = _make_dbas_controller(K=K)
        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,), f"K={K}: wrong control shape"
        assert not np.any(np.isnan(control)), f"K={K}: NaN"
        assert info["num_samples"] == K


# ── Integration Tests (3) ───────────────────────────────────


def test_reset_clears_state():
    """reset 후 adaptive_scale, history 초기화"""
    np.random.seed(42)
    ctrl = _make_dbas_controller()
    state = np.array([2.0, 1.5, 0.0])
    ref = _make_ref()

    ctrl.compute_control(state, ref)
    assert ctrl._adaptive_scale > 0 or len(ctrl._dbas_history) > 0

    ctrl.reset()
    assert ctrl._adaptive_scale == 0.0
    assert len(ctrl._dbas_history) == 0
    assert len(ctrl._obstacles) == len(DEFAULT_OBSTACLES)
    assert np.allclose(ctrl.U, 0.0)


def test_repr():
    """__repr__ 내용 검증"""
    ctrl = _make_dbas_controller(
        dbas_obstacles=[(1, 2, 0.3), (3, 4, 0.5)],
        dbas_walls=[('x', 5.0, -1)],
        barrier_weight=20.0,
        barrier_gamma=0.3,
        K=256,
    )
    repr_str = repr(ctrl)

    assert "DBaSMPPIController" in repr_str
    assert "DifferentialDriveKinematic" in repr_str
    assert "obstacles=2" in repr_str
    assert "walls=1" in repr_str
    assert "barrier_weight=20.0" in repr_str
    assert "gamma=0.3" in repr_str
    assert "K=256" in repr_str


def test_numerical_stability():
    """극단적 위치에서 NaN/Inf 없음"""
    np.random.seed(42)
    ctrl = _make_dbas_controller(
        dbas_obstacles=[(0.0, 0.0, 1.0)],
        h_min=1e-6,
        K=64, N=10,
    )

    extreme_states = [
        np.array([0.0, 0.0, 0.0]),       # 장애물 중심
        np.array([1.0, 0.0, 0.0]),        # 장애물 경계
        np.array([100.0, 100.0, 0.0]),    # 매우 먼 곳
        np.array([-0.5, 0.0, np.pi]),     # 장애물 내부
    ]
    ref = _make_ref()

    for state in extreme_states:
        control, info = ctrl.compute_control(state, ref)
        assert not np.any(np.isnan(control)), \
            f"NaN at state={state}: control={control}"
        assert not np.any(np.isinf(control)), \
            f"Inf at state={state}: control={control}"
        assert np.isfinite(info["best_cost"]), \
            f"Non-finite cost at state={state}"


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DBaS-MPPI Unit Tests")
    print("=" * 60)

    tests = [
        # Params (3)
        test_params_defaults,
        test_params_custom,
        test_params_validation,
        # Barrier Function (5)
        test_constraint_values_circle,
        test_constraint_values_wall,
        test_barrier_function_safe,
        test_barrier_function_boundary,
        test_barrier_function_clipping,
        # Barrier Cost (4)
        test_barrier_cost_shape,
        test_barrier_cost_safe_traj,
        test_barrier_cost_near_obstacle,
        test_barrier_cost_weight_scale,
        # Controller Basic (5)
        test_compute_control_shape,
        test_compute_control_no_obstacle,
        test_info_dbas_stats,
        test_adaptive_scale_near_obs,
        test_adaptive_scale_free_space,
        # Performance (4)
        test_circle_tracking_rmse,
        test_obstacle_avoidance,
        test_wall_avoidance,
        test_narrow_corridor,
        # Comparison (4)
        test_dbas_vs_vanilla_obstacle,
        test_adaptive_vs_fixed_noise,
        test_dynamic_obstacle_update,
        test_different_K_values,
        # Integration (3)
        test_reset_clears_state,
        test_repr,
        test_numerical_stability,
    ]

    try:
        for t in tests:
            t()
            print(f"  PASS: {t.__name__}")
        print(f"\n{'=' * 60}")
        print(f"  All {len(tests)} Tests Passed!")
        print(f"{'=' * 60}")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
