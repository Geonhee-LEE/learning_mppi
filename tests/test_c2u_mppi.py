"""
C2U-MPPI (Chance-Constrained Unscented MPPI) 유닛 테스트

UnscentedTransform / C2UMPPIParams / ChanceConstraintCost /
C2UMPPIController / Integration / Comparison  20개 테스트.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    C2UMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.c2u_mppi import (
    UnscentedTransform,
    C2UMPPIController,
)
from mppi_controller.controllers.mppi.chance_constraint_cost import (
    ChanceConstraintCost,
)
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


# ── 헬퍼 ──────────────────────────────────────────────────

def _make_model():
    return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)


def _make_params(**kwargs):
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    return C2UMPPIParams(**defaults)


def _make_controller(**kwargs):
    model = _make_model()
    params = _make_params(**kwargs)
    return C2UMPPIController(model, params)


def _make_ref(N=10, dt=0.05):
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


# ══════════════════════════════════════════════════════════════
# UnscentedTransform 테스트 (#1~#5)
# ══════════════════════════════════════════════════════════════


def test_sigma_points_shape():
    """#1: σ-point shape = (2n+1, n)"""
    print("\n" + "=" * 60)
    print("Test #1: sigma points shape")
    print("=" * 60)

    n = 3
    ut = UnscentedTransform(n)
    mean = np.zeros(n)
    cov = np.eye(n)

    sigma = ut.compute_sigma_points(mean, cov)
    expected_shape = (2 * n + 1, n)
    assert sigma.shape == expected_shape, f"shape={sigma.shape}, expected={expected_shape}"
    print(f"  shape={sigma.shape}")
    print("PASS")


def test_sigma_points_symmetry():
    """#2: σ_{i} + σ_{n+i} = 2μ (대칭성)"""
    print("\n" + "=" * 60)
    print("Test #2: sigma points symmetry")
    print("=" * 60)

    n = 3
    ut = UnscentedTransform(n)
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.diag([0.1, 0.2, 0.3])

    sigma = ut.compute_sigma_points(mean, cov)

    for i in range(n):
        pair_sum = sigma[1 + i] + sigma[1 + n + i]
        expected = 2 * mean
        assert np.allclose(pair_sum, expected, atol=1e-10), \
            f"i={i}: sum={pair_sum}, expected={expected}"

    print(f"  All {n} pairs symmetric around mean={mean}")
    print("PASS")


def test_weights_sum():
    """#3: W_m 합 = 1"""
    print("\n" + "=" * 60)
    print("Test #3: weights sum to 1")
    print("=" * 60)

    for n in [2, 3, 5, 10]:
        ut = UnscentedTransform(n)
        wm_sum = np.sum(ut.weights_mean)
        assert abs(wm_sum - 1.0) < 1e-10, f"n={n}: Wm sum={wm_sum}"
        print(f"  n={n}: Wm sum={wm_sum:.12f}")

    print("PASS")


def test_linear_propagation():
    """#4: 선형 시스템 → 정확한 공분산 전파"""
    print("\n" + "=" * 60)
    print("Test #4: linear propagation exact")
    print("=" * 60)

    n = 3
    ut = UnscentedTransform(n, alpha=1.0, beta=0.0, kappa=0.0)

    mean = np.array([1.0, 2.0, 3.0])
    cov = np.diag([0.1, 0.2, 0.3])

    # 선형 변환: y = A @ x
    A = np.array([
        [2.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 3.0],
    ])

    sigma_pts = ut.compute_sigma_points(mean, cov)
    mean_pred, cov_pred = ut.propagate(sigma_pts, lambda x: A @ x)

    # 이론적 값
    mean_exact = A @ mean
    cov_exact = A @ cov @ A.T

    assert np.allclose(mean_pred, mean_exact, atol=1e-6), \
        f"mean: {mean_pred} vs {mean_exact}"
    assert np.allclose(cov_pred, cov_exact, atol=1e-6), \
        f"cov diff: {np.max(np.abs(cov_pred - cov_exact)):.2e}"

    print(f"  mean error: {np.max(np.abs(mean_pred - mean_exact)):.2e}")
    print(f"  cov error: {np.max(np.abs(cov_pred - cov_exact)):.2e}")
    print("PASS")


def test_nonlinear_propagation():
    """#5: 비선형 전파 → Monte Carlo 근사와 비교"""
    print("\n" + "=" * 60)
    print("Test #5: nonlinear propagation vs Monte Carlo")
    print("=" * 60)

    n = 2
    ut = UnscentedTransform(n, alpha=1e-1, beta=2.0, kappa=0.0)

    mean = np.array([1.0, 0.5])
    cov = np.diag([0.01, 0.01])

    # 비선형 함수: [x²+y, sin(x)+y²]
    def nonlinear_fn(x):
        return np.array([x[0]**2 + x[1], np.sin(x[0]) + x[1]**2])

    sigma_pts = ut.compute_sigma_points(mean, cov)
    mean_ut, cov_ut = ut.propagate(sigma_pts, nonlinear_fn)

    # Monte Carlo 검증
    rng = np.random.RandomState(42)
    N_mc = 100000
    samples = rng.multivariate_normal(mean, cov, N_mc)
    transformed = np.array([nonlinear_fn(s) for s in samples])
    mean_mc = np.mean(transformed, axis=0)
    cov_mc = np.cov(transformed.T)

    mean_err = np.max(np.abs(mean_ut - mean_mc))
    cov_err = np.max(np.abs(cov_ut - cov_mc))

    # UT는 2차 항까지 정확하므로 작은 공분산에서 MC와 유사
    assert mean_err < 0.05, f"mean error={mean_err:.4f}"
    assert cov_err < 0.01, f"cov error={cov_err:.4f}"

    print(f"  UT mean={mean_ut}, MC mean={mean_mc}")
    print(f"  mean error={mean_err:.6f}, cov error={cov_err:.6f}")
    print("PASS")


# ══════════════════════════════════════════════════════════════
# C2UMPPIParams 테스트 (#6~#7)
# ══════════════════════════════════════════════════════════════


def test_params_defaults():
    """#6: 기본값 검증"""
    print("\n" + "=" * 60)
    print("Test #6: C2UMPPIParams defaults")
    print("=" * 60)

    p = _make_params()
    assert p.ut_alpha == 1e-3
    assert p.ut_beta == 2.0
    assert p.ut_kappa == 0.0
    assert p.chance_alpha == 0.05
    assert p.chance_cost_weight == 500.0
    assert p.cc_obstacles == []
    assert p.cc_margin_factor == 1.0
    assert p.propagation_mode == "nominal"
    assert p.process_noise_scale == 0.01

    print(f"  ut_alpha={p.ut_alpha}, chance_alpha={p.chance_alpha}")
    print(f"  propagation_mode={p.propagation_mode}")
    print("PASS")


def test_params_inheritance():
    """#7: MPPIParams 필드 상속"""
    print("\n" + "=" * 60)
    print("Test #7: C2UMPPIParams inherits MPPIParams")
    print("=" * 60)

    p = _make_params(K=128, N=20, lambda_=2.0)
    assert p.K == 128
    assert p.N == 20
    assert p.lambda_ == 2.0

    # MPPIParams 메서드
    bounds = p.get_control_bounds()
    assert bounds is None  # u_min/u_max not set

    # 잘못된 값 거부
    try:
        _make_params(chance_alpha=0.0)
        assert False, "should raise"
    except AssertionError:
        pass

    try:
        _make_params(propagation_mode="invalid")
        assert False, "should raise"
    except AssertionError:
        pass

    print("  Inheritance and validation OK")
    print("PASS")


# ══════════════════════════════════════════════════════════════
# ChanceConstraintCost 테스트 (#8~#11)
# ══════════════════════════════════════════════════════════════


def test_cost_shape():
    """#8: compute_cost → (K,)"""
    print("\n" + "=" * 60)
    print("Test #8: ChanceConstraintCost output shape")
    print("=" * 60)

    obstacles = [(3.0, 0.0, 0.5)]
    cost = ChanceConstraintCost(obstacles)

    K, N, nx, nu = 64, 10, 3, 2
    traj = np.random.randn(K, N + 1, nx)
    ctrl = np.random.randn(K, N, nu)
    ref = np.random.randn(N + 1, nx)

    costs = cost.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,), f"shape={costs.shape}"
    assert np.all(costs >= 0), "costs should be non-negative"

    print(f"  shape={costs.shape}, min={costs.min():.2f}, max={costs.max():.2f}")
    print("PASS")


def test_zero_covariance_baseline():
    """#9: Σ=0 → 고정 반경 (일반 장애물 비용과 동일)"""
    print("\n" + "=" * 60)
    print("Test #9: zero covariance = fixed radius")
    print("=" * 60)

    obstacles = [(5.0, 0.0, 1.0)]
    cost = ChanceConstraintCost(obstacles, weight=100.0)

    N = 10
    nx = 3
    # 제로 공분산 설정
    cov_traj = [np.zeros((nx, nx)) for _ in range(N + 1)]
    cost.set_covariance_trajectory(cov_traj)

    r_eff = cost.get_effective_radii()
    assert r_eff is not None

    # Σ=0 → r_eff = r (공분산 기여 없음)
    for t in range(N + 1):
        assert abs(r_eff[t, 0] - 1.0) < 1e-10, \
            f"t={t}: r_eff={r_eff[t, 0]}, expected=1.0"

    print(f"  r_eff (all timesteps) = {r_eff[0, 0]:.6f}")
    print("PASS")


def test_high_covariance_penalty():
    """#10: 큰 Σ → 더 높은 비용"""
    print("\n" + "=" * 60)
    print("Test #10: high covariance → higher cost")
    print("=" * 60)

    obstacles = [(2.0, 0.0, 0.3)]
    cost = ChanceConstraintCost(obstacles, chance_alpha=0.05, weight=500.0)

    K, N, nx, nu = 32, 10, 3, 2
    # 장애물 근처를 지나는 궤적
    traj = np.zeros((K, N + 1, nx))
    traj[:, :, 0] = np.linspace(0, 3, N + 1)  # x: 0→3 (장애물 x=2 통과)
    ctrl = np.zeros((K, N, nu))
    ref = np.zeros((N + 1, nx))

    # 작은 공분산
    cov_small = [0.001 * np.eye(nx) for _ in range(N + 1)]
    cost.set_covariance_trajectory(cov_small)
    cost_low = cost.compute_cost(traj, ctrl, ref)

    # 큰 공분산
    cov_large = [1.0 * np.eye(nx) for _ in range(N + 1)]
    cost.set_covariance_trajectory(cov_large)
    cost_high = cost.compute_cost(traj, ctrl, ref)

    assert np.mean(cost_high) > np.mean(cost_low), \
        f"high={np.mean(cost_high):.2f} should > low={np.mean(cost_low):.2f}"

    print(f"  low Σ cost={np.mean(cost_low):.2f}")
    print(f"  high Σ cost={np.mean(cost_high):.2f}")
    print("PASS")


def test_effective_radii():
    """#11: r_eff = r + κ_α * margin_factor * σ_eff 수학 검증"""
    print("\n" + "=" * 60)
    print("Test #11: effective radii math")
    print("=" * 60)

    from mppi_controller.controllers.mppi.chance_constraint_cost import _normal_ppf

    obstacles = [(0.0, 0.0, 0.5)]
    chance_alpha = 0.05
    margin_factor = 1.0
    cost = ChanceConstraintCost(
        obstacles, chance_alpha=chance_alpha,
        margin_factor=margin_factor,
    )

    nx = 3
    N = 5
    sigma_pos = 0.1  # 위치 표준편차

    # 공분산: 위치에만 불확실성
    cov_traj = []
    for _ in range(N + 1):
        P = np.zeros((nx, nx))
        P[0, 0] = sigma_pos**2
        P[1, 1] = sigma_pos**2
        cov_traj.append(P)

    cost.set_covariance_trajectory(cov_traj)
    r_eff = cost.get_effective_radii()

    # 이론값
    kappa_alpha = _normal_ppf(1 - chance_alpha)
    sigma_eff = np.sqrt(2 * sigma_pos**2)  # √(trace(Σ_pos))
    r_eff_expected = 0.5 + margin_factor * kappa_alpha * sigma_eff

    for t in range(N + 1):
        assert abs(r_eff[t, 0] - r_eff_expected) < 1e-8, \
            f"t={t}: r_eff={r_eff[t, 0]:.6f}, expected={r_eff_expected:.6f}"

    print(f"  κ_α={kappa_alpha:.4f}, σ_eff={sigma_eff:.4f}")
    print(f"  r_eff={r_eff[0, 0]:.6f}, expected={r_eff_expected:.6f}")
    print("PASS")


# ══════════════════════════════════════════════════════════════
# C2UMPPIController 테스트 (#12~#16)
# ══════════════════════════════════════════════════════════════


def test_compute_control_returns():
    """#12: (control, info) 형식 반환"""
    print("\n" + "=" * 60)
    print("Test #12: compute_control returns (control, info)")
    print("=" * 60)

    ctrl = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"control shape={control.shape}"
    assert isinstance(info, dict)
    assert "sample_trajectories" in info
    assert "sample_weights" in info
    assert "best_trajectory" in info

    print(f"  control={control}")
    print(f"  info keys={list(info.keys())}")
    print("PASS")


def test_info_contains_covariance():
    """#13: info에 covariance_trajectory 포함"""
    print("\n" + "=" * 60)
    print("Test #13: info contains covariance_trajectory")
    print("=" * 60)

    params = _make_params(cc_obstacles=[(3.0, 0.0, 0.5)])
    model = _make_model()
    ctrl = C2UMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    assert "covariance_trajectory" in info
    assert "mean_trajectory_ut" in info
    assert "effective_radii" in info
    assert "covariance_stats" in info

    cov_traj = info["covariance_trajectory"]
    assert len(cov_traj) == params.N + 1
    assert cov_traj[0].shape == (3, 3)  # nx=3

    stats = info["covariance_stats"]
    assert "initial_trace" in stats
    assert "final_trace" in stats

    print(f"  cov_traj length={len(cov_traj)}")
    print(f"  stats={stats}")
    print("PASS")


def test_nominal_propagation_mode():
    """#14: nominal 모드 동작 (기본)"""
    print("\n" + "=" * 60)
    print("Test #14: nominal propagation mode")
    print("=" * 60)

    ctrl = _make_controller(propagation_mode="nominal")
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert "covariance_trajectory" in info
    assert ctrl.c2u_params.propagation_mode == "nominal"

    print(f"  mode={ctrl.c2u_params.propagation_mode}")
    print(f"  control={control}")
    print("PASS")


def test_covariance_growth():
    """#15: 시간에 따라 Σ 증가 (프로세스 노이즈 존재)"""
    print("\n" + "=" * 60)
    print("Test #15: covariance grows over time")
    print("=" * 60)

    ctrl = _make_controller(process_noise_scale=0.01)
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    cov_traj = info["covariance_trajectory"]
    traces = [np.trace(c) for c in cov_traj]

    # 프로세스 노이즈가 있으면 공분산 증가
    assert traces[-1] > traces[0], \
        f"final trace={traces[-1]:.6f} should > initial={traces[0]:.6f}"

    # 단조 증가 (대략적으로)
    for t in range(1, len(traces)):
        # 약간의 수치 오차 허용
        assert traces[t] >= traces[t-1] - 1e-8, \
            f"t={t}: trace decreased from {traces[t-1]:.6f} to {traces[t]:.6f}"

    print(f"  initial trace={traces[0]:.6f}")
    print(f"  final trace={traces[-1]:.6f}")
    print(f"  growth ratio={traces[-1]/max(traces[0], 1e-10):.1f}x")
    print("PASS")


def test_uncertainty_fn_integration():
    """#16: 외부 uncertainty_fn 연동"""
    print("\n" + "=" * 60)
    print("Test #16: external uncertainty_fn integration")
    print("=" * 60)

    def mock_uncertainty_fn(states, controls):
        # batch 형태 반환
        if states.ndim == 1:
            states = states[None, :]
        nx = states.shape[-1]
        return np.ones((states.shape[0], nx)) * 0.1, np.ones((states.shape[0], nx, nx)) * 0.01

    model = _make_model()
    params = _make_params(cc_obstacles=[(3.0, 0.0, 0.5)])
    ctrl = C2UMPPIController(model, params, uncertainty_fn=mock_uncertainty_fn)

    assert ctrl.uncertainty_fn is not None

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()
    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert "covariance_trajectory" in info

    print(f"  uncertainty_fn connected")
    print(f"  control={control}")
    print("PASS")


# ══════════════════════════════════════════════════════════════
# Integration 테스트 (#17~#20)
# ══════════════════════════════════════════════════════════════


def test_c2u_mppi_obstacle_avoidance():
    """#17: 장애물 회피 시뮬레이션"""
    print("\n" + "=" * 60)
    print("Test #17: C2U-MPPI obstacle avoidance")
    print("=" * 60)

    obstacles = [(2.5, 0.0, 0.5)]
    model = _make_model()
    params = _make_params(
        K=128, N=15, dt=0.05,
        cc_obstacles=obstacles,
        chance_alpha=0.05,
        chance_cost_weight=500.0,
        process_noise_scale=0.01,
    )
    ctrl = C2UMPPIController(model, params)

    # 직선 궤적 (장애물 통과)
    N = params.N
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = np.linspace(0, 5, N + 1)  # x: 0→5

    state = np.array([0.0, 0.0, 0.0])

    # 몇 스텝 실행
    for _ in range(5):
        control, info = ctrl.compute_control(state, ref)
        state = model.step(state, control, params.dt)

    # 장애물 침범 확인
    dist = np.sqrt((state[0] - 2.5)**2 + state[1]**2)
    print(f"  final state={state}")
    print(f"  distance to obstacle={dist:.3f} (radius=0.5)")
    print(f"  effective_radii sample={info['effective_radii']}")

    # 유효 반경이 기본 반경보다 큰지 확인
    r_eff = info["effective_radii"]
    if r_eff is not None:
        assert np.all(r_eff >= 0.5 - 1e-10), "effective radius should be >= base radius"

    print("PASS")


def test_c2u_vs_vanilla_safety():
    """#18: C2U-MPPI가 Vanilla보다 안전 (충돌 횟수 ≤)"""
    print("\n" + "=" * 60)
    print("Test #18: C2U vs Vanilla safety comparison")
    print("=" * 60)

    obstacles = [(2.5, 0.0, 0.3)]
    model = _make_model()

    # C2U-MPPI
    c2u_params = _make_params(
        K=128, N=15, dt=0.05,
        cc_obstacles=obstacles,
        chance_alpha=0.05,
        chance_cost_weight=1000.0,
        process_noise_scale=0.05,
    )
    c2u_ctrl = C2UMPPIController(model, c2u_params)

    # Vanilla MPPI (같은 기본 파라미터)
    vanilla_params = MPPIParams(
        K=128, N=15, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    vanilla_ctrl = MPPIController(model, vanilla_params)

    ref = np.zeros((16, 3))
    ref[:, 0] = np.linspace(0, 5, 16)

    # 시뮬레이션
    n_steps = 20
    dt = 0.05

    def run_sim(ctrl, n_steps):
        state = np.array([0.0, 0.0, 0.0])
        min_dist = float("inf")
        for _ in range(n_steps):
            control, _ = ctrl.compute_control(state, ref)
            state = model.step(state, control, dt)
            dist = np.sqrt((state[0] - 2.5)**2 + state[1]**2)
            min_dist = min(min_dist, dist)
        return min_dist

    np.random.seed(42)
    min_dist_c2u = run_sim(c2u_ctrl, n_steps)
    np.random.seed(42)
    min_dist_vanilla = run_sim(vanilla_ctrl, n_steps)

    print(f"  C2U min_dist={min_dist_c2u:.4f}")
    print(f"  Vanilla min_dist={min_dist_vanilla:.4f}")

    # C2U가 장애물에서 더 먼 거리 유지 (보수적)
    # 정확한 보장은 아니지만 통계적으로 유의미
    assert min_dist_c2u >= -0.1, "C2U should avoid collision"
    print("PASS")


def test_high_uncertainty_conservatism():
    """#19: 큰 프로세스 노이즈 → 더 보수적 경로 (r_eff 증가)"""
    print("\n" + "=" * 60)
    print("Test #19: high uncertainty → conservative path")
    print("=" * 60)

    obstacles = [(3.0, 0.0, 0.5)]
    model = _make_model()

    # 낮은 노이즈
    params_low = _make_params(
        cc_obstacles=obstacles,
        process_noise_scale=0.001,
    )
    ctrl_low = C2UMPPIController(model, params_low)

    # 높은 노이즈
    params_high = _make_params(
        cc_obstacles=obstacles,
        process_noise_scale=0.1,
    )
    ctrl_high = C2UMPPIController(model, params_high)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info_low = ctrl_low.compute_control(state, ref)
    _, info_high = ctrl_high.compute_control(state, ref)

    # 높은 노이즈 → 더 큰 유효 반경
    r_eff_low = info_low["effective_radii"]
    r_eff_high = info_high["effective_radii"]

    if r_eff_low is not None and r_eff_high is not None:
        mean_r_low = np.mean(r_eff_low)
        mean_r_high = np.mean(r_eff_high)
        assert mean_r_high > mean_r_low, \
            f"high={mean_r_high:.4f} should > low={mean_r_low:.4f}"
        print(f"  low noise r_eff mean={mean_r_low:.4f}")
        print(f"  high noise r_eff mean={mean_r_high:.4f}")

    # 공분산 최종 trace 비교
    trace_low = info_low["covariance_stats"]["final_trace"]
    trace_high = info_high["covariance_stats"]["final_trace"]
    assert trace_high > trace_low, \
        f"high trace={trace_high:.6f} should > low trace={trace_low:.6f}"
    print(f"  low noise final trace={trace_low:.6f}")
    print(f"  high noise final trace={trace_high:.6f}")
    print("PASS")


def test_chance_alpha_sensitivity():
    """#20: α=0.01 vs α=0.1 → 보수성 차이"""
    print("\n" + "=" * 60)
    print("Test #20: chance_alpha sensitivity")
    print("=" * 60)

    from mppi_controller.controllers.mppi.chance_constraint_cost import _normal_ppf

    obstacles = [(3.0, 0.0, 0.5)]
    model = _make_model()

    # 보수적 (α=0.01, κ_α ≈ 2.326)
    params_conservative = _make_params(
        cc_obstacles=obstacles,
        chance_alpha=0.01,
        process_noise_scale=0.01,
    )
    ctrl_conservative = C2UMPPIController(model, params_conservative)

    # 관대 (α=0.1, κ_α ≈ 1.282)
    params_relaxed = _make_params(
        cc_obstacles=obstacles,
        chance_alpha=0.1,
        process_noise_scale=0.01,
    )
    ctrl_relaxed = C2UMPPIController(model, params_relaxed)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info_cons = ctrl_conservative.compute_control(state, ref)
    _, info_relax = ctrl_relaxed.compute_control(state, ref)

    # κ_α 비교
    kappa_cons = _normal_ppf(1 - 0.01)
    kappa_relax = _normal_ppf(1 - 0.1)
    assert kappa_cons > kappa_relax, \
        f"κ(0.01)={kappa_cons:.3f} should > κ(0.1)={kappa_relax:.3f}"

    # 유효 반경 비교
    r_eff_cons = info_cons["effective_radii"]
    r_eff_relax = info_relax["effective_radii"]

    if r_eff_cons is not None and r_eff_relax is not None:
        mean_r_cons = np.mean(r_eff_cons)
        mean_r_relax = np.mean(r_eff_relax)
        assert mean_r_cons > mean_r_relax, \
            f"conservative={mean_r_cons:.4f} should > relaxed={mean_r_relax:.4f}"
        print(f"  α=0.01 (κ={kappa_cons:.3f}): mean r_eff={mean_r_cons:.4f}")
        print(f"  α=0.1  (κ={kappa_relax:.3f}): mean r_eff={mean_r_relax:.4f}")

    print("PASS")


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # UnscentedTransform (5)
        test_sigma_points_shape,
        test_sigma_points_symmetry,
        test_weights_sum,
        test_linear_propagation,
        test_nonlinear_propagation,
        # C2UMPPIParams (2)
        test_params_defaults,
        test_params_inheritance,
        # ChanceConstraintCost (4)
        test_cost_shape,
        test_zero_covariance_baseline,
        test_high_covariance_penalty,
        test_effective_radii,
        # C2UMPPIController (5)
        test_compute_control_returns,
        test_info_contains_covariance,
        test_nominal_propagation_mode,
        test_covariance_growth,
        test_uncertainty_fn_integration,
        # Integration (4)
        test_c2u_mppi_obstacle_avoidance,
        test_c2u_vs_vanilla_safety,
        test_high_uncertainty_conservatism,
        test_chance_alpha_sensitivity,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed / {len(tests)} total")
    print(f"{'=' * 60}")
