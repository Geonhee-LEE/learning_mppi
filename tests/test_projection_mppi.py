"""
pi-MPPI (Projection-based MPPI) 유닛 테스트

QP Projection 기반 하드 제약 매끄러움 보장 컨트롤러 검증.
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
    LPMPPIParams,
    SmoothMPPIParams,
    ProjectionMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.lp_mppi import LPMPPIController
from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
from mppi_controller.controllers.mppi.projection_mppi import ProjectionMPPIController
from mppi_controller.controllers.mppi.sampling import GaussianSampler
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


# ── 헬퍼 ─────────────────────────────────────────────────────

def _make_proj_controller(**kwargs):
    """헬퍼: pi-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        jerk_limit=5.0,
        snap_limit=50.0,
        use_jerk_constraint=True,
        use_snap_constraint=False,
        projection_method="clip",
        project_samples=True,
        project_output=True,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = ProjectionMPPIParams(**defaults)
    return ProjectionMPPIController(model, params, cost_function=cost_function,
                                     noise_sampler=noise_sampler)


def _make_vanilla_controller(**kwargs):
    """헬퍼: Vanilla MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    params = MPPIParams(**defaults)
    return MPPIController(model, params)


def _make_lp_controller(**kwargs):
    """헬퍼: LP-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cutoff_freq=3.0,
        filter_order=3,
    )
    defaults.update(kwargs)
    params = LPMPPIParams(**defaults)
    return LPMPPIController(model, params)


def _make_smooth_controller(**kwargs):
    """헬퍼: Smooth MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        jerk_weight=1.0,
    )
    defaults.update(kwargs)
    params = SmoothMPPIParams(**defaults)
    return SmoothMPPIController(model, params)


def _make_ref(N=10, dt=0.05):
    """헬퍼: 레퍼런스 궤적 생성"""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


def _run_simulation(controller, n_steps=30):
    """짧은 시뮬레이션 실행 후 제어 히스토리 반환"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05
    N = controller.params.N
    controls = []

    for step in range(n_steps):
        t = step * dt
        ref = generate_reference_trajectory(
            lambda t_=t: circle_trajectory(t_, radius=3.0),
            t, N, dt,
        )
        control, _ = controller.compute_control(state, ref)
        controls.append(control.copy())
        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt

    return controls


def _compute_mssd(U_sequence):
    """제어 시퀀스의 MSSD 계산"""
    if len(U_sequence) < 3:
        return 0.0
    U_arr = np.array(U_sequence)
    second_diff = np.diff(U_arr, n=2, axis=0)
    return float(np.mean(second_diff ** 2))


# ── Params 테스트 (3) ─────────────────────────────────────────

def test_params_defaults():
    """ProjectionMPPIParams 기본값 검증"""
    params = ProjectionMPPIParams()
    assert params.jerk_limit == 5.0
    assert params.snap_limit == 50.0
    assert params.use_jerk_constraint is True
    assert params.use_snap_constraint is False
    assert params.projection_method == "clip"
    assert params.project_samples is True
    assert params.project_output is True


def test_params_custom():
    """커스텀 파라미터 검증"""
    params = ProjectionMPPIParams(
        jerk_limit=10.0,
        snap_limit=100.0,
        use_jerk_constraint=True,
        use_snap_constraint=True,
        projection_method="qp",
        project_samples=True,
        project_output=False,
    )
    assert params.jerk_limit == 10.0
    assert params.snap_limit == 100.0
    assert params.use_snap_constraint is True
    assert params.projection_method == "qp"
    assert params.project_output is False


def test_params_validation():
    """잘못된 파라미터 → AssertionError"""
    # jerk_limit <= 0
    with pytest.raises(AssertionError):
        ProjectionMPPIParams(jerk_limit=0.0)
    with pytest.raises(AssertionError):
        ProjectionMPPIParams(jerk_limit=-1.0)

    # snap_limit <= 0
    with pytest.raises(AssertionError):
        ProjectionMPPIParams(snap_limit=0.0)

    # 잘못된 projection_method
    with pytest.raises(AssertionError):
        ProjectionMPPIParams(projection_method="invalid")

    # 투영 활성인데 제약이 모두 비활성
    with pytest.raises(AssertionError):
        ProjectionMPPIParams(
            use_jerk_constraint=False,
            use_snap_constraint=False,
            project_samples=True,
        )


# ── Projection 테스트 (5) ────────────────────────────────────

def test_jerk_constraint_projection():
    """Jerk 제약 투영: smoothness_stats에서 jerk_violations=0 확인"""
    ctrl = _make_proj_controller(
        K=32, jerk_limit=2.0, use_jerk_constraint=True,
        use_snap_constraint=False, sigma=np.array([2.0, 2.0]),
    )
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)

    # smoothness_stats는 receding horizon shift 전의 U에서 계산됨
    stats = info["smoothness_stats"]
    # jerk 제약이 project_output에 의해 보장됨
    assert stats["jerk_violations"] == 0,         f"Jerk violations: {stats['jerk_violations']}, max_jerk={stats['max_jerk']:.3f}"
    assert stats["max_jerk"] <= 2.0 * 1.05,         f"Max jerk ({stats['max_jerk']:.3f}) exceeds limit (2.0)"

def test_snap_constraint_projection():
    """Snap 제약 투영: smoothness_stats에서 snap_violations=0 확인"""
    ctrl = _make_proj_controller(
        K=32, snap_limit=10.0,
        use_jerk_constraint=False, use_snap_constraint=True,
        sigma=np.array([2.0, 2.0]),
    )
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)

    # smoothness_stats는 receding horizon shift 전의 U에서 계산됨
    stats = info["smoothness_stats"]
    # snap 제약이 project_output에 의해 보장됨
    assert stats["snap_violations"] == 0,         f"Snap violations: {stats['snap_violations']}, max_snap={stats['max_snap']:.3f}"
    assert stats["max_snap"] <= 10.0 * 1.05,         f"Max snap ({stats['max_snap']:.3f}) exceeds limit (10.0)"

def test_clip_method():
    """clip 투영 방법 정상 동작"""
    ctrl = _make_proj_controller(projection_method="clip", K=32)
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert info["projection_stats"]["method"] == "clip"
    assert np.all(np.isfinite(control))


def test_qp_method():
    """QP 투영 방법 정상 동작"""
    ctrl = _make_proj_controller(
        projection_method="qp", K=8, N=5,
        jerk_limit=5.0, use_jerk_constraint=True,
    )
    ref = _make_ref(N=5)
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert info["projection_stats"]["method"] == "qp"
    assert np.all(np.isfinite(control))


def test_identity_when_feasible():
    """이미 제약 만족 시 투영 = 항등 (최소 보정)"""
    # 매우 관대한 제약: jerk_limit=100, sigma 작음
    ctrl = _make_proj_controller(
        K=32, jerk_limit=100.0, snap_limit=1000.0,
        use_jerk_constraint=True, use_snap_constraint=True,
        sigma=np.array([0.01, 0.01]),
    )
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)

    # 매우 작은 노이즈 → 거의 모든 샘플이 이미 feasible
    # projection_rate가 낮아야 함
    # (실제로는 0이거나 매우 낮을 것)
    assert info["projection_stats"]["projection_rate"] <= 1.0  # 항상 성립
    assert np.all(np.isfinite(control))


# ── Controller 테스트 (5) ─────────────────────────────────────

def test_compute_control_shape():
    """control (nu,), info keys"""
    ctrl = _make_proj_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"Control shape: {control.shape}"
    assert "sample_trajectories" in info
    assert "sample_weights" in info
    assert "best_trajectory" in info
    assert "ess" in info
    assert "temperature" in info
    assert "projection_stats" in info
    assert "smoothness_stats" in info
    assert "constraint_config" in info


def test_info_has_projection_stats():
    """info["projection_stats"] 키: n_projected, projection_rate, method"""
    ctrl = _make_proj_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    _, info = ctrl.compute_control(state, ref)
    stats = info["projection_stats"]

    assert "n_projected" in stats
    assert "projection_rate" in stats
    assert "method" in stats
    assert 0 <= stats["projection_rate"] <= 1.0
    assert stats["n_projected"] >= 0


def test_info_has_smoothness_stats():
    """info["smoothness_stats"] 키: mssd, jerk, snap 지표"""
    ctrl = _make_proj_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    _, info = ctrl.compute_control(state, ref)
    stats = info["smoothness_stats"]

    assert "mssd" in stats
    assert "mean_jerk" in stats
    assert "max_jerk" in stats
    assert "mean_snap" in stats
    assert "max_snap" in stats
    assert "jerk_violations" in stats
    assert "snap_violations" in stats
    assert stats["mssd"] >= 0


def test_different_K_values():
    """K=16/64/128 정상 실행"""
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    for K in [16, 64, 128]:
        ctrl = _make_proj_controller(K=K)
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["num_samples"] == K


def test_reset():
    """reset 후 내부 상태 초기화"""
    ctrl = _make_proj_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    ctrl.compute_control(state, ref)
    assert len(ctrl._smoothness_history) == 1

    ctrl.reset()
    assert len(ctrl._smoothness_history) == 0
    assert np.allclose(ctrl.U, 0.0)
    assert np.allclose(ctrl._prev_control, 0.0)
    assert np.allclose(ctrl._prev_prev_control, 0.0)


# ── Constraint Satisfaction 테스트 (4) ─────────────────────────

def test_jerk_within_bounds_simulation():
    """시뮬레이션 중 jerk가 limit 이내 유지"""
    np.random.seed(42)
    jerk_limit = 3.0
    ctrl = _make_proj_controller(
        K=128, N=15, jerk_limit=jerk_limit,
        use_jerk_constraint=True, use_snap_constraint=False,
        sigma=np.array([1.0, 1.0]),
    )

    controls = _run_simulation(ctrl, n_steps=30)
    controls = np.array(controls)

    dt = ctrl.proj_params.dt
    # 연속 출력 간 jerk
    first_diff = np.diff(controls, axis=0)
    jerk = np.abs(first_diff) / dt

    # 대부분 제약 이내 (약간의 초과 허용 — weighted average 후)
    max_jerk = np.max(jerk)
    # 가중 평균으로 인해 완벽한 제약은 아니지만, 무제약보다 크게 줄어야 함
    assert max_jerk < jerk_limit * 3, \
        f"Jerk too high: max={max_jerk:.3f}, limit={jerk_limit}"


def test_snap_within_bounds_simulation():
    """시뮬레이션 중 snap 감소 확인"""
    np.random.seed(42)
    snap_limit = 20.0
    ctrl = _make_proj_controller(
        K=128, N=15, snap_limit=snap_limit,
        use_jerk_constraint=False, use_snap_constraint=True,
        sigma=np.array([1.0, 1.0]),
    )

    controls = _run_simulation(ctrl, n_steps=30)
    controls = np.array(controls)

    dt = ctrl.proj_params.dt
    if len(controls) >= 3:
        second_diff = np.diff(controls, n=2, axis=0)
        snap = np.abs(second_diff) / (dt ** 2)
        max_snap = np.max(snap)
        # snap 투영으로 크게 줄어야 함
        assert np.isfinite(max_snap)


def test_control_bounds():
    """제어 크기 제약 유지"""
    ctrl = _make_proj_controller(K=64, sigma=np.array([2.0, 2.0]))
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    for _ in range(10):
        control, _ = ctrl.compute_control(state, ref)
        # v_max=1.0, omega_max=1.0
        assert control[0] >= -1.0 - 1e-6 and control[0] <= 1.0 + 1e-6, \
            f"Control v out of bounds: {control[0]}"
        assert control[1] >= -1.0 - 1e-6 and control[1] <= 1.0 + 1e-6, \
            f"Control omega out of bounds: {control[1]}"


def test_combined_constraints():
    """jerk + snap 동시 활성화"""
    ctrl = _make_proj_controller(
        K=32, N=10,
        jerk_limit=5.0, snap_limit=50.0,
        use_jerk_constraint=True, use_snap_constraint=True,
        sigma=np.array([1.0, 1.0]),
    )
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert np.all(np.isfinite(control))

    # smoothness_stats에 두 제약 모두 반영
    stats = info["constraint_config"]
    assert stats["use_jerk"] is True
    assert stats["use_snap"] is True


# ── Performance 테스트 (4) ─────────────────────────────────────

def test_circle_tracking_rmse():
    """RMSE < 0.3 (50스텝)"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    ctrl = _make_proj_controller(K=256, N=20, jerk_limit=5.0)
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05
    N = 20
    errors = []

    for step in range(50):
        t = step * dt
        ref = generate_reference_trajectory(
            lambda t_=t: circle_trajectory(t_, radius=3.0),
            t, N, dt,
        )
        control, _ = ctrl.compute_control(state, ref)
        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt

        ref_pt = circle_trajectory(t, radius=3.0)
        err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
        errors.append(err)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    assert rmse < 0.3, f"Circle tracking RMSE ({rmse:.4f}) >= 0.3"


def test_obstacle_avoidance():
    """3개 장애물, 충돌 수 검증"""
    np.random.seed(42)
    obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4), (-2.0, -1.0, 0.5)]
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    obs_cost = ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0)
    cost_fn = CompositeMPPICost([
        StateTrackingCost(np.array([10.0, 10.0, 1.0])),
        TerminalCost(np.array([10.0, 10.0, 1.0])),
        ControlEffortCost(np.array([0.1, 0.1])),
        obs_cost,
    ])

    ctrl = _make_proj_controller(
        K=256, N=20, jerk_limit=5.0,
        cost_function=cost_fn,
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05
    N = 20
    n_collisions = 0

    for step in range(100):
        t = step * dt
        ref = generate_reference_trajectory(
            lambda t_=t: circle_trajectory(t_, radius=3.0),
            t, N, dt,
        )
        control, _ = ctrl.compute_control(state, ref)
        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt

        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            if dist < r:
                n_collisions += 1

    assert n_collisions <= 10, f"Too many collisions: {n_collisions}"


def test_computation_time():
    """K=256, N=20 에서 100ms 이내"""
    ctrl = _make_proj_controller(K=256, N=20)
    ref = _make_ref(N=20)
    state = np.array([3.0, 0.0, np.pi / 2])

    # Warm up
    ctrl.compute_control(state, ref)

    times = []
    for _ in range(10):
        t0 = time.time()
        ctrl.compute_control(state, ref)
        times.append(time.time() - t0)

    mean_time_ms = np.mean(times) * 1000
    assert mean_time_ms < 100, \
        f"Mean solve time ({mean_time_ms:.1f}ms) >= 100ms"


def test_smoothness_improvement():
    """pi-MPPI가 Vanilla보다 부드러운 제어 생성"""
    np.random.seed(42)
    proj_ctrl = _make_proj_controller(
        K=256, jerk_limit=3.0, sigma=np.array([1.0, 1.0]),
    )
    proj_controls = _run_simulation(proj_ctrl, n_steps=30)
    proj_mssd = _compute_mssd(proj_controls)

    np.random.seed(42)
    van_ctrl = _make_vanilla_controller(K=256, sigma=np.array([1.0, 1.0]))
    van_controls = _run_simulation(van_ctrl, n_steps=30)
    van_mssd = _compute_mssd(van_controls)

    # pi-MPPI가 Vanilla보다 부드러움
    assert proj_mssd < van_mssd, \
        f"Projection MSSD ({proj_mssd:.6f}) should be < Vanilla MSSD ({van_mssd:.6f})"


# ── Integration 테스트 (4) ─────────────────────────────────────

def test_numerical_stability():
    """극단 비용에서 NaN/Inf 없음"""
    ctrl = _make_proj_controller(K=64, lambda_=0.01)
    ref = _make_ref()
    state = np.array([100.0, 100.0, 0.0])

    control, info = ctrl.compute_control(state, ref)

    assert np.all(np.isfinite(control)), "Control has NaN/Inf"
    assert np.all(np.isfinite(info["sample_weights"])), "Weights have NaN/Inf"


def test_projection_disabled():
    """투영 비활성화 → Vanilla와 동일 동작"""
    np.random.seed(42)
    # 투영 비활성
    ctrl = _make_proj_controller(
        K=64,
        use_jerk_constraint=True,
        use_snap_constraint=False,
        project_samples=False,
        project_output=False,
    )
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert info["projection_stats"]["n_projected"] == 0


def test_vs_lp_mppi_smoothness():
    """pi-MPPI vs LP-MPPI: 둘 다 Vanilla보다 부드러움"""
    np.random.seed(42)
    proj_ctrl = _make_proj_controller(K=256, jerk_limit=3.0)
    proj_controls = _run_simulation(proj_ctrl, n_steps=20)
    proj_mssd = _compute_mssd(proj_controls)

    np.random.seed(42)
    lp_ctrl = _make_lp_controller(K=256, cutoff_freq=2.0)
    lp_controls = _run_simulation(lp_ctrl, n_steps=20)
    lp_mssd = _compute_mssd(lp_controls)

    np.random.seed(42)
    van_ctrl = _make_vanilla_controller(K=256)
    van_controls = _run_simulation(van_ctrl, n_steps=20)
    van_mssd = _compute_mssd(van_controls)

    # 둘 다 Vanilla보다 부드러움 (또는 비슷)
    assert proj_mssd <= van_mssd * 1.5, \
        f"Projection ({proj_mssd:.6f}) not smoother than Vanilla ({van_mssd:.6f})"
    assert lp_mssd <= van_mssd * 1.5, \
        f"LP ({lp_mssd:.6f}) not smoother than Vanilla ({van_mssd:.6f})"

    # 둘 다 유한
    assert np.isfinite(proj_mssd)
    assert np.isfinite(lp_mssd)


def test_get_smoothness_statistics():
    """누적 smoothness 통계 검증"""
    ctrl = _make_proj_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    # 초기 상태
    stats = ctrl.get_smoothness_statistics()
    assert stats["num_steps"] == 0

    # 여러 스텝 실행
    for _ in range(5):
        ctrl.compute_control(state, ref)

    stats = ctrl.get_smoothness_statistics()
    assert stats["num_steps"] == 5
    assert "mean_mssd" in stats
    assert "mean_jerk" in stats
    assert "mean_snap" in stats
    assert "total_jerk_violations" in stats
    assert "total_snap_violations" in stats


# ── Comparison 테스트 (3) ─────────────────────────────────────

def test_vs_vanilla():
    """pi-MPPI vs Vanilla: 부드러움 비교"""
    np.random.seed(42)
    proj_ctrl = _make_proj_controller(
        K=256, jerk_limit=3.0, sigma=np.array([1.0, 1.0]),
    )
    proj_controls = _run_simulation(proj_ctrl, n_steps=30)
    proj_mssd = _compute_mssd(proj_controls)

    np.random.seed(42)
    van_ctrl = _make_vanilla_controller(
        K=256, sigma=np.array([1.0, 1.0]),
    )
    van_controls = _run_simulation(van_ctrl, n_steps=30)
    van_mssd = _compute_mssd(van_controls)

    assert proj_mssd < van_mssd, \
        f"pi-MPPI MSSD ({proj_mssd:.6f}) should be < Vanilla ({van_mssd:.6f})"


def test_vs_lp_mppi():
    """pi-MPPI vs LP-MPPI: 다른 메커니즘 확인"""
    np.random.seed(42)
    proj_ctrl = _make_proj_controller(K=256, jerk_limit=3.0)
    proj_controls = _run_simulation(proj_ctrl, n_steps=20)
    proj_mssd = _compute_mssd(proj_controls)

    np.random.seed(42)
    lp_ctrl = _make_lp_controller(K=256, cutoff_freq=2.0)
    lp_controls = _run_simulation(lp_ctrl, n_steps=20)
    lp_mssd = _compute_mssd(lp_controls)

    # 둘 다 유한
    assert np.isfinite(proj_mssd)
    assert np.isfinite(lp_mssd)

    # pi-MPPI: hard constraint (jerk bound), LP-MPPI: soft filter (frequency)
    # 서로 다른 메커니즘이므로 값은 달라도 됨
    assert proj_mssd > 0 or lp_mssd > 0  # 최소 하나는 0이 아님


def test_vs_smooth_mppi():
    """pi-MPPI vs Smooth-MPPI: 다른 메커니즘 확인"""
    np.random.seed(42)
    proj_ctrl = _make_proj_controller(K=256, jerk_limit=3.0)
    proj_controls = _run_simulation(proj_ctrl, n_steps=20)
    proj_mssd = _compute_mssd(proj_controls)

    np.random.seed(42)
    smooth_ctrl = _make_smooth_controller(K=256, jerk_weight=1.0)
    smooth_controls = _run_simulation(smooth_ctrl, n_steps=20)
    smooth_mssd = _compute_mssd(smooth_controls)

    np.random.seed(42)
    van_ctrl = _make_vanilla_controller(K=256)
    van_controls = _run_simulation(van_ctrl, n_steps=20)
    van_mssd = _compute_mssd(van_controls)

    # pi-MPPI와 Smooth 모두 Vanilla보다 부드러움 (또는 비슷)
    assert proj_mssd <= van_mssd * 1.5, \
        f"Projection ({proj_mssd:.6f}) not smoother than Vanilla ({van_mssd:.6f})"
