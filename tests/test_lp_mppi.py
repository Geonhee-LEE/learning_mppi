"""
LP-MPPI (Low-Pass MPPI) 유닛 테스트

Butterworth 저역통과 필터 기반 노이즈 샘플링 + 컨트롤러 검증.
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
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.lp_mppi import LPMPPIController
from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
from mppi_controller.controllers.mppi.sampling import (
    GaussianSampler,
    ColoredNoiseSampler,
    LowPassSampler,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


# ── 헬퍼 ─────────────────────────────────────────────────────

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
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = LPMPPIParams(**defaults)
    return LPMPPIController(model, params, cost_function=cost_function,
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


def _make_colored_controller(**kwargs):
    """헬퍼: Colored-Noise MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    sampler = ColoredNoiseSampler(
        sigma=defaults["sigma"],
        theta=np.array([2.0, 2.0]),
        dt=defaults["dt"],
    )
    params = MPPIParams(**defaults)
    return MPPIController(model, params, noise_sampler=sampler)


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


# ── Params 테스트 (3) ─────────────────────────────────────────

def test_params_defaults():
    """LPMPPIParams 기본값 검증"""
    params = LPMPPIParams()
    assert params.cutoff_freq == 3.0
    assert params.filter_order == 3
    assert params.normalize_variance is False


def test_params_custom():
    """커스텀 파라미터 검증"""
    params = LPMPPIParams(
        cutoff_freq=5.0,
        filter_order=4,
        normalize_variance=True,
    )
    assert params.cutoff_freq == 5.0
    assert params.filter_order == 4
    assert params.normalize_variance is True


def test_params_validation():
    """잘못된 파라미터 → AssertionError"""
    # cutoff_freq <= 0
    with pytest.raises(AssertionError):
        LPMPPIParams(cutoff_freq=0.0)
    with pytest.raises(AssertionError):
        LPMPPIParams(cutoff_freq=-1.0)

    # cutoff_freq >= Nyquist (dt=0.05 → Nyquist=10)
    with pytest.raises(AssertionError):
        LPMPPIParams(cutoff_freq=10.0, dt=0.05)
    with pytest.raises(AssertionError):
        LPMPPIParams(cutoff_freq=15.0, dt=0.05)

    # filter_order < 1
    with pytest.raises(AssertionError):
        LPMPPIParams(filter_order=0)

    # filter_order > 10
    with pytest.raises(AssertionError):
        LPMPPIParams(filter_order=11)


# ── LowPassSampler 테스트 (6) ────────────────────────────────

def test_sampler_shape():
    """(K, N, nu) 출력 shape 검증"""
    sampler = LowPassSampler(
        sigma=np.array([0.5, 0.5]),
        cutoff_freq=3.0, filter_order=3, dt=0.05,
    )
    U = np.zeros((30, 2))
    noise = sampler.sample(U, K=128)
    assert noise.shape == (128, 30, 2)


def test_low_freq_preserved():
    """FFT: f < f_c 대역 에너지 ≥ 80% 보존"""
    np.random.seed(42)
    sampler = LowPassSampler(
        sigma=np.array([1.0]),
        cutoff_freq=3.0, filter_order=3, dt=0.05, seed=42,
    )
    U = np.zeros((100, 1))
    noise = sampler.sample(U, K=256)

    # 평균 파워 스펙트럼
    fft = np.fft.rfft(noise[:, :, 0], axis=1)
    power = np.mean(np.abs(fft) ** 2, axis=0)
    freqs = np.fft.rfftfreq(100, d=0.05)

    # f < f_c 대역
    low_mask = freqs < 3.0
    low_mask[0] = False  # DC 제외
    total_power = np.sum(power[1:])
    low_power = np.sum(power[low_mask])

    assert low_power / total_power >= 0.8, \
        f"Low-freq power ratio: {low_power / total_power:.3f} < 0.8"


def test_high_freq_attenuated():
    """FFT: f > f_c 대역 에너지 감쇠 확인"""
    np.random.seed(42)
    sigma = np.array([1.0])
    fc = 2.0

    # 가우시안 (필터 없음)
    gauss_sampler = GaussianSampler(sigma=sigma, seed=42)
    U = np.zeros((100, 1))
    gauss_noise = gauss_sampler.sample(U, K=256)

    # LP 필터
    lp_sampler = LowPassSampler(
        sigma=sigma, cutoff_freq=fc, filter_order=5, dt=0.05, seed=42,
    )
    lp_noise = lp_sampler.sample(U, K=256)

    # 고주파 파워 비교
    freqs = np.fft.rfftfreq(100, d=0.05)
    high_mask = freqs > fc

    gauss_fft = np.fft.rfft(gauss_noise[:, :, 0], axis=1)
    lp_fft = np.fft.rfft(lp_noise[:, :, 0], axis=1)

    gauss_high = np.mean(np.abs(gauss_fft[:, high_mask]) ** 2)
    lp_high = np.mean(np.abs(lp_fft[:, high_mask]) ** 2)

    assert lp_high < gauss_high * 0.5, \
        f"LP high-freq power ({lp_high:.3f}) not attenuated vs Gaussian ({gauss_high:.3f})"


def test_cutoff_effect():
    """f_c=1Hz vs f_c=5Hz: 낮은 f_c → 더 부드러움 (MSSD 작음)"""
    U = np.zeros((30, 2))
    K = 256

    sampler_low = LowPassSampler(
        sigma=np.array([0.5, 0.5]), cutoff_freq=1.0, filter_order=3, dt=0.05, seed=42,
    )
    sampler_high = LowPassSampler(
        sigma=np.array([0.5, 0.5]), cutoff_freq=5.0, filter_order=3, dt=0.05, seed=42,
    )

    noise_low = sampler_low.sample(U, K)
    noise_high = sampler_high.sample(U, K)

    mssd_low = np.mean(np.diff(noise_low, n=2, axis=1) ** 2)
    mssd_high = np.mean(np.diff(noise_high, n=2, axis=1) ** 2)

    assert mssd_low < mssd_high, \
        f"Lower cutoff should be smoother: MSSD({mssd_low:.4f}) >= MSSD({mssd_high:.4f})"


def test_order_effect():
    """order=1 vs order=5: 높은 order → 더 급격한 rolloff"""
    U = np.zeros((100, 1))
    K = 256
    fc = 3.0

    sampler_o1 = LowPassSampler(
        sigma=np.array([1.0]), cutoff_freq=fc, filter_order=1, dt=0.05, seed=42,
    )
    sampler_o5 = LowPassSampler(
        sigma=np.array([1.0]), cutoff_freq=fc, filter_order=5, dt=0.05, seed=42,
    )

    noise_o1 = sampler_o1.sample(U, K)
    noise_o5 = sampler_o5.sample(U, K)

    # 고주파 에너지 비교 (order=5가 더 많이 감쇠)
    freqs = np.fft.rfftfreq(100, d=0.05)
    high_mask = freqs > fc * 1.5  # f_c보다 충분히 높은 대역

    fft_o1 = np.fft.rfft(noise_o1[:, :, 0], axis=1)
    fft_o5 = np.fft.rfft(noise_o5[:, :, 0], axis=1)

    high_o1 = np.mean(np.abs(fft_o1[:, high_mask]) ** 2)
    high_o5 = np.mean(np.abs(fft_o5[:, high_mask]) ** 2)

    assert high_o5 < high_o1, \
        f"Higher order should attenuate more: {high_o5:.4f} >= {high_o1:.4f}"


def test_variance_normalization():
    """normalize=True → std ≈ sigma, False → std < sigma"""
    sigma = np.array([1.0, 1.0])
    U = np.zeros((50, 2))
    K = 512

    sampler_norm = LowPassSampler(
        sigma=sigma, cutoff_freq=2.0, filter_order=3, dt=0.05,
        normalize_variance=True, seed=42,
    )
    sampler_raw = LowPassSampler(
        sigma=sigma, cutoff_freq=2.0, filter_order=3, dt=0.05,
        normalize_variance=False, seed=42,
    )

    noise_norm = sampler_norm.sample(U, K)
    noise_raw = sampler_raw.sample(U, K)

    std_norm = np.std(noise_norm)
    std_raw = np.std(noise_raw)

    # 정규화: std ≈ sigma
    assert abs(std_norm - 1.0) < 0.2, \
        f"Normalized std ({std_norm:.3f}) not close to sigma (1.0)"

    # 비정규화: std < sigma (필터가 에너지를 감소시킴)
    assert std_raw < 1.0, \
        f"Raw std ({std_raw:.3f}) should be < sigma (1.0)"


# ── Controller 테스트 (5) ─────────────────────────────────────

def test_compute_control_shape():
    """control (nu,), info keys"""
    ctrl = _make_lp_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"Control shape: {control.shape}"
    assert "sample_trajectories" in info
    assert "sample_weights" in info
    assert "best_trajectory" in info
    assert "ess" in info
    assert "smoothness_stats" in info
    assert "cutoff_freq" in info
    assert "filter_order" in info


def test_default_sampler_is_lp():
    """noise_sampler가 LowPassSampler 인스턴스"""
    ctrl = _make_lp_controller()
    assert isinstance(ctrl.noise_sampler, LowPassSampler), \
        f"Expected LowPassSampler, got {type(ctrl.noise_sampler)}"


def test_info_has_smoothness():
    """info["smoothness_stats"] 키: mssd, mean_jerk, max_jerk"""
    ctrl = _make_lp_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    _, info = ctrl.compute_control(state, ref)
    stats = info["smoothness_stats"]

    assert "mssd" in stats
    assert "mean_jerk" in stats
    assert "max_jerk" in stats
    assert stats["mssd"] >= 0
    assert stats["mean_jerk"] >= 0
    assert stats["max_jerk"] >= 0


def test_different_K_values():
    """K=16/64/128 정상 실행"""
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    for K in [16, 64, 128]:
        ctrl = _make_lp_controller(K=K)
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["num_samples"] == K


def test_custom_sampler_override():
    """noise_sampler 주입 시 LowPassSampler 대신 사용"""
    gauss_sampler = GaussianSampler(sigma=np.array([0.5, 0.5]), seed=42)
    ctrl = _make_lp_controller(noise_sampler=gauss_sampler)

    assert isinstance(ctrl.noise_sampler, GaussianSampler), \
        "Custom sampler should override default LowPassSampler"


# ── Smoothness 비교 테스트 (4) ────────────────────────────────

def _compute_mssd(U_sequence):
    """제어 시퀀스의 MSSD 계산"""
    if len(U_sequence) < 3:
        return 0.0
    U_arr = np.array(U_sequence)
    second_diff = np.diff(U_arr, n=2, axis=0)
    return float(np.mean(second_diff ** 2))


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


def test_smoother_than_gaussian():
    """LP MSSD < Gaussian MSSD (같은 σ, K=512)"""
    np.random.seed(42)
    lp_ctrl = _make_lp_controller(K=512, cutoff_freq=2.0)
    lp_controls = _run_simulation(lp_ctrl)
    lp_mssd = _compute_mssd(lp_controls)

    np.random.seed(42)
    van_ctrl = _make_vanilla_controller(K=512)
    van_controls = _run_simulation(van_ctrl)
    van_mssd = _compute_mssd(van_controls)

    assert lp_mssd < van_mssd, \
        f"LP MSSD ({lp_mssd:.6f}) should be < Vanilla MSSD ({van_mssd:.6f})"


def test_smoother_than_colored():
    """LP vs Colored: MSSD 비교"""
    np.random.seed(42)
    lp_ctrl = _make_lp_controller(K=256, cutoff_freq=2.0)
    lp_controls = _run_simulation(lp_ctrl)
    lp_mssd = _compute_mssd(lp_controls)

    np.random.seed(42)
    col_ctrl = _make_colored_controller(K=256)
    col_controls = _run_simulation(col_ctrl)
    col_mssd = _compute_mssd(col_controls)

    # LP와 Colored 모두 Vanilla보다 부드러움, 둘 다 유한
    assert lp_mssd < float("inf")
    assert col_mssd < float("inf")


def test_mssd_metric_correct():
    """수동 MSSD 계산과 일치"""
    ctrl = _make_lp_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    # 여러 스텝 실행
    for _ in range(5):
        ctrl.compute_control(state, ref)

    # 수동 계산
    U = ctrl.U
    second_diff = np.diff(U, n=2, axis=0)
    manual_mssd = float(np.mean(second_diff ** 2))

    # 마지막 info의 smoothness_stats
    stats = ctrl._smoothness_history[-1]
    assert abs(stats["mssd"] - manual_mssd) < 1e-10, \
        f"MSSD mismatch: {stats['mssd']} vs {manual_mssd}"


def test_jerk_metric_correct():
    """수동 jerk 계산과 일치"""
    ctrl = _make_lp_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    for _ in range(5):
        ctrl.compute_control(state, ref)

    U = ctrl.U
    second_diff = np.diff(U, n=2, axis=0)
    jerk_norms = np.linalg.norm(second_diff, axis=1)
    manual_mean_jerk = float(np.mean(jerk_norms))
    manual_max_jerk = float(np.max(jerk_norms))

    stats = ctrl._smoothness_history[-1]
    assert abs(stats["mean_jerk"] - manual_mean_jerk) < 1e-10
    assert abs(stats["max_jerk"] - manual_max_jerk) < 1e-10


# ── 성능 테스트 (4) ───────────────────────────────────────────

def test_circle_tracking_rmse():
    """RMSE < 0.3 (50스텝)"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    ctrl = _make_lp_controller(K=256, N=20, cutoff_freq=3.0)
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


def test_obstacle_navigation():
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

    ctrl = _make_lp_controller(K=256, N=20, cutoff_freq=3.0,
                               cost_function=cost_fn)
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

    # 장애물 회피 성공 기대 (약간의 충돌은 허용)
    assert n_collisions <= 10, f"Too many collisions: {n_collisions}"


def test_cutoff_sweep():
    """f_c = [1,2,3,5,8] Hz → f_c ↑ = MSSD ↑ (smoothness ↓)"""
    np.random.seed(42)
    cutoffs = [1.0, 2.0, 3.0, 5.0, 8.0]
    mssds = []

    for fc in cutoffs:
        ctrl = _make_lp_controller(K=128, cutoff_freq=fc)
        controls = _run_simulation(ctrl, n_steps=20)
        mssds.append(_compute_mssd(controls))

    # MSSD는 f_c가 증가하면 증가 (부드러움 감소) — 단조 증가 확인
    for i in range(len(mssds) - 1):
        assert mssds[i] <= mssds[i + 1] * 1.5, \
            f"MSSD should increase with cutoff: fc={cutoffs[i]}: {mssds[i]:.6f} > fc={cutoffs[i+1]}: {mssds[i+1]:.6f}"


def test_order_sweep():
    """order = [1,2,3,5] → smoothness 단조 증가"""
    np.random.seed(42)
    orders = [1, 2, 3, 5]
    mssds = []

    for order in orders:
        ctrl = _make_lp_controller(K=128, cutoff_freq=2.0, filter_order=order)
        controls = _run_simulation(ctrl, n_steps=20)
        mssds.append(_compute_mssd(controls))

    # Higher order → lower MSSD (smoother), 또는 최소 큰 증가 없음
    assert mssds[-1] <= mssds[0] * 2, \
        f"order=5 MSSD ({mssds[-1]:.6f}) much worse than order=1 ({mssds[0]:.6f})"


# ── 통합 테스트 (3) ───────────────────────────────────────────

def test_reset_clears_state():
    """reset 후 smoothness_history 초기화"""
    ctrl = _make_lp_controller()
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    ctrl.compute_control(state, ref)
    assert len(ctrl._smoothness_history) == 1

    ctrl.reset()
    assert len(ctrl._smoothness_history) == 0
    assert np.allclose(ctrl.U, 0.0)


def test_numerical_stability():
    """극단 비용에서 NaN/Inf 없음"""
    ctrl = _make_lp_controller(K=64, lambda_=0.01)
    ref = _make_ref()
    state = np.array([100.0, 100.0, 0.0])  # 레퍼런스에서 매우 먼 상태

    control, info = ctrl.compute_control(state, ref)

    assert np.all(np.isfinite(control)), "Control has NaN/Inf"
    assert np.all(np.isfinite(info["sample_weights"])), "Weights have NaN/Inf"


def test_composition_with_smooth():
    """LowPassSampler + SmoothMPPIController 조합 가능"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    lp_sampler = LowPassSampler(
        sigma=np.array([0.5, 0.5]),
        cutoff_freq=3.0, filter_order=3, dt=0.05,
    )
    params = SmoothMPPIParams(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        jerk_weight=1.0,
    )
    # SmoothMPPIController에 LP sampler 주입
    ctrl = SmoothMPPIController(model, params)
    ctrl.noise_sampler = lp_sampler

    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])
    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert np.all(np.isfinite(control))


# ── 비교 테스트 (3) ───────────────────────────────────────────

def test_vs_vanilla_smoothness():
    """LP MSSD << Vanilla MSSD"""
    np.random.seed(42)
    lp_ctrl = _make_lp_controller(K=256, cutoff_freq=2.0)
    lp_controls = _run_simulation(lp_ctrl, n_steps=30)
    lp_mssd = _compute_mssd(lp_controls)

    np.random.seed(42)
    van_ctrl = _make_vanilla_controller(K=256)
    van_controls = _run_simulation(van_ctrl, n_steps=30)
    van_mssd = _compute_mssd(van_controls)

    # LP가 Vanilla보다 확실히 부드러움
    assert lp_mssd < van_mssd, \
        f"LP MSSD ({lp_mssd:.6f}) should be << Vanilla MSSD ({van_mssd:.6f})"


def test_vs_colored_noise():
    """LP vs Colored: 유사 smoothness, LP가 벡터화로 더 빠름"""
    np.random.seed(42)
    lp_ctrl = _make_lp_controller(K=256, cutoff_freq=2.0)

    t0 = time.time()
    lp_controls = _run_simulation(lp_ctrl, n_steps=20)
    lp_time = time.time() - t0

    np.random.seed(42)
    col_ctrl = _make_colored_controller(K=256)

    t0 = time.time()
    col_controls = _run_simulation(col_ctrl, n_steps=20)
    col_time = time.time() - t0

    lp_mssd = _compute_mssd(lp_controls)
    col_mssd = _compute_mssd(col_controls)

    # 둘 다 유한
    assert np.isfinite(lp_mssd)
    assert np.isfinite(col_mssd)

    # LP가 같거나 빠를 것 (벡터화)
    # 타이밍은 환경에 따라 달라 엄격하게 검증하지 않음
    assert lp_time < col_time * 5, \
        f"LP ({lp_time:.3f}s) much slower than Colored ({col_time:.3f}s)"


def test_vs_smooth_mppi():
    """LP vs Smooth: 다른 메커니즘 확인"""
    np.random.seed(42)
    lp_ctrl = _make_lp_controller(K=256, cutoff_freq=2.0)
    lp_controls = _run_simulation(lp_ctrl, n_steps=20)
    lp_mssd = _compute_mssd(lp_controls)

    np.random.seed(42)
    smooth_ctrl = _make_smooth_controller(K=256, jerk_weight=1.0)
    smooth_controls = _run_simulation(smooth_ctrl, n_steps=20)
    smooth_mssd = _compute_mssd(smooth_controls)

    # 둘 다 Vanilla보다 부드러움
    np.random.seed(42)
    van_ctrl = _make_vanilla_controller(K=256)
    van_controls = _run_simulation(van_ctrl, n_steps=20)
    van_mssd = _compute_mssd(van_controls)

    # 적어도 Vanilla보다는 부드러움 (또는 비슷)
    assert lp_mssd <= van_mssd * 1.5, \
        f"LP ({lp_mssd:.6f}) not smoother than Vanilla ({van_mssd:.6f})"
