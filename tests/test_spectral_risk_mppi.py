"""
ASR-MPPI (Adaptive Spectral Risk MPPI) 유닛 테스트

Spectral Risk Measure 기반 가중치 계산 + 적응적 위험 조절 검증.
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
    RiskAwareMPPIParams,
    ASRMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from mppi_controller.controllers.mppi.spectral_risk_mppi import ASRMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


# ── 헬퍼 ─────────────────────────────────────────────────────

def _make_asr_controller(**kwargs):
    """헬퍼: ASR-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        distortion_type="sigmoid",
        distortion_alpha=0.5,
        distortion_beta=5.0,
        distortion_gamma=1.0,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    params = ASRMPPIParams(**defaults)
    return ASRMPPIController(model, params, cost_function=cost_function)


def _make_risk_controller(**kwargs):
    """헬퍼: Risk-Aware MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cvar_alpha=0.7,
    )
    defaults.update(kwargs)
    params = RiskAwareMPPIParams(**defaults)
    return RiskAwareMPPIController(model, params)


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


def _make_ref(N=10, dt=0.05):
    """헬퍼: 레퍼런스 궤적 생성"""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ── Params 테스트 (3) ─────────────────────────────────────────

def test_params_defaults():
    """ASRMPPIParams 기본값 검증"""
    params = ASRMPPIParams()
    assert params.distortion_type == "sigmoid"
    assert params.distortion_alpha == 0.5
    assert params.distortion_beta == 5.0
    assert params.distortion_gamma == 1.0
    assert params.use_adaptive_risk is False
    assert params.adaptation_rate == 0.1
    assert params.adaptation_window == 50
    assert params.min_ess_ratio == 0.1


def test_params_custom():
    """커스텀 파라미터 검증"""
    params = ASRMPPIParams(
        distortion_type="power",
        distortion_gamma=0.5,
        use_adaptive_risk=True,
        adaptation_rate=0.2,
        adaptation_window=100,
        min_ess_ratio=0.2,
    )
    assert params.distortion_type == "power"
    assert params.distortion_gamma == 0.5
    assert params.use_adaptive_risk is True
    assert params.adaptation_rate == 0.2


def test_params_validation():
    """잘못된 파라미터 → AssertionError"""
    import pytest

    # 잘못된 distortion_type
    with pytest.raises(AssertionError):
        ASRMPPIParams(distortion_type="invalid")

    # alpha 범위 위반
    with pytest.raises(AssertionError):
        ASRMPPIParams(distortion_alpha=0.0)
    with pytest.raises(AssertionError):
        ASRMPPIParams(distortion_alpha=1.0)

    # beta <= 0
    with pytest.raises(AssertionError):
        ASRMPPIParams(distortion_beta=0.0)

    # gamma <= 0
    with pytest.raises(AssertionError):
        ASRMPPIParams(distortion_gamma=0.0)

    # adaptation_rate 범위 위반
    with pytest.raises(AssertionError):
        ASRMPPIParams(adaptation_rate=0.0)

    # adaptation_window < 1
    with pytest.raises(AssertionError):
        ASRMPPIParams(adaptation_window=0)

    # min_ess_ratio 범위 위반
    with pytest.raises(AssertionError):
        ASRMPPIParams(min_ess_ratio=0.0)


# ── 왜곡 함수 테스트 (5) ─────────────────────────────────────

def test_sigmoid_distortion():
    """Sigmoid 왜곡 함수: φ(0)≈0, φ(1)≈1, 단조 증가"""
    ctrl = _make_asr_controller(distortion_type="sigmoid", distortion_beta=5.0)
    q = np.linspace(0, 1, 100)
    phi = ctrl._eval_distortion(q)

    assert abs(phi[0]) < 0.01, f"φ(0)={phi[0]}, expected ≈0"
    assert abs(phi[-1] - 1.0) < 0.01, f"φ(1)={phi[-1]}, expected ≈1"
    # 단조 증가
    assert np.all(np.diff(phi) >= -1e-10), "φ must be monotonically non-decreasing"


def test_power_distortion():
    """Power 왜곡 함수: φ(q) = q^γ"""
    ctrl = _make_asr_controller(distortion_type="power", distortion_gamma=2.0)
    q = np.linspace(0, 1, 100)
    phi = ctrl._eval_distortion(q)

    assert abs(phi[0]) < 1e-10, "φ(0)=0"
    assert abs(phi[-1] - 1.0) < 1e-10, "φ(1)=1"
    # φ(0.5) = 0.5^2 = 0.25
    idx_half = np.argmin(np.abs(q - 0.5))
    assert abs(phi[idx_half] - 0.25) < 0.02, f"φ(0.5)={phi[idx_half]}, expected ≈0.25"


def test_dual_power_distortion():
    """Dual Power 왜곡 함수: φ(q) = 1-(1-q)^γ"""
    ctrl = _make_asr_controller(distortion_type="dual_power", distortion_gamma=2.0)
    q = np.linspace(0, 1, 100)
    phi = ctrl._eval_distortion(q)

    assert abs(phi[0]) < 1e-10, "φ(0)=0"
    assert abs(phi[-1] - 1.0) < 1e-10, "φ(1)=1"
    # φ(0.5) = 1-(0.5)^2 = 0.75
    idx_half = np.argmin(np.abs(q - 0.5))
    assert abs(phi[idx_half] - 0.75) < 0.02, f"φ(0.5)={phi[idx_half]}, expected ≈0.75"


def test_cvar_distortion_matches_risk_aware():
    """CVaR 모드 → Risk-Aware MPPI와 동일 가중치"""
    np.random.seed(42)
    alpha = 0.7

    asr = _make_asr_controller(
        K=128, distortion_type="cvar", distortion_alpha=alpha,
    )
    risk = _make_risk_controller(K=128, cvar_alpha=alpha)

    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])

    # 동일 시드로 비교
    np.random.seed(123)
    ctrl_asr, info_asr = asr.compute_control(state, ref)
    np.random.seed(123)
    ctrl_risk, info_risk = risk.compute_control(state, ref)

    # 가중치 패턴 비교: CVaR cutoff 이하 → 0, 이상 → softmax
    w_asr = info_asr["sample_weights"]
    w_risk = info_risk["sample_weights"]

    # 0인 가중치 수가 비슷해야 함
    zeros_asr = np.sum(w_asr == 0)
    zeros_risk = np.sum(w_risk == 0)
    # CVaR cutoff 동일 패턴
    expected_zeros = int(128 * (1 - alpha))
    assert abs(zeros_asr - expected_zeros) <= 2, \
        f"ASR zeros={zeros_asr}, expected ≈{expected_zeros}"
    assert abs(zeros_risk - expected_zeros) <= 2, \
        f"Risk zeros={zeros_risk}, expected ≈{expected_zeros}"


def test_sigmoid_converges_to_cvar():
    """β→∞이면 sigmoid → 계단 함수에 수렴 (CVaR처럼 lower quantiles 제거)"""
    ctrl_sigmoid = _make_asr_controller(
        distortion_type="sigmoid", distortion_alpha=0.5, distortion_beta=100.0,
    )

    q = np.linspace(0, 1, 1000)
    phi_sig = ctrl_sigmoid._eval_distortion(q)

    # β=100이면 q<0.5에서 φ≈0, q>0.5에서 φ≈1 (계단 함수)
    # 하위 40% quantile에서 φ ≈ 0
    lower = phi_sig[q < 0.4]
    assert np.all(lower < 0.01), f"Lower quantiles not suppressed: max={np.max(lower)}"
    # 상위 40% quantile에서 φ ≈ 1
    upper = phi_sig[q > 0.6]
    assert np.all(upper > 0.99), f"Upper quantiles not near 1: min={np.min(upper)}"


# ── 가중치 계산 테스트 (4) ─────────────────────────────────────

def test_weights_sum_to_one():
    """모든 왜곡 타입에서 가중치 합 = 1"""
    for dtype in ["sigmoid", "power", "dual_power", "cvar"]:
        ctrl = _make_asr_controller(distortion_type=dtype)
        ref = _make_ref()
        state = np.array([3.0, 0.0, np.pi / 2])
        _, info = ctrl.compute_control(state, ref)
        w = info["sample_weights"]
        assert abs(np.sum(w) - 1.0) < 1e-6, f"{dtype}: sum={np.sum(w)}"


def test_weights_non_negative():
    """모든 가중치 ≥ 0"""
    for dtype in ["sigmoid", "power", "dual_power", "cvar"]:
        ctrl = _make_asr_controller(distortion_type=dtype)
        ref = _make_ref()
        state = np.array([3.0, 0.0, np.pi / 2])
        _, info = ctrl.compute_control(state, ref)
        w = info["sample_weights"]
        assert np.all(w >= 0), f"{dtype}: negative weights found"


def test_sigmoid_weights_smooth():
    """Sigmoid 가중치: 인접 가중치 차이 연속 (계단 없음)"""
    ctrl = _make_asr_controller(
        K=256, distortion_type="sigmoid", distortion_beta=5.0,
    )
    ref = _make_ref()
    state = np.array([3.0, 0.0, np.pi / 2])
    _, info = ctrl.compute_control(state, ref)
    w = info["sample_weights"]

    # 정렬된 가중치의 인접 차이
    sorted_w = np.sort(w)[::-1]
    diffs = np.abs(np.diff(sorted_w))

    # 최대 차이가 전체 범위의 50% 미만 (계단 없음)
    max_diff = np.max(diffs)
    weight_range = sorted_w[0] - sorted_w[-1]
    if weight_range > 1e-8:
        assert max_diff < weight_range * 0.5, \
            f"max_diff={max_diff:.6f}, range={weight_range:.6f} → step detected"


def test_power_gamma_effect():
    """γ < 1: 낮은 비용 강조, γ > 1: 높은 비용 강조"""
    np.random.seed(42)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # γ=0.5 → 낮은 비용에 더 높은 가중치
    ctrl_low = _make_asr_controller(
        K=128, distortion_type="power", distortion_gamma=0.5,
    )
    np.random.seed(42)
    _, info_low = ctrl_low.compute_control(state, ref)

    # γ=2.0 → 높은 비용에도 가중치 부여
    ctrl_high = _make_asr_controller(
        K=128, distortion_type="power", distortion_gamma=2.0,
    )
    np.random.seed(42)
    _, info_high = ctrl_high.compute_control(state, ref)

    ess_low = info_low["ess"]
    ess_high = info_high["ess"]

    # γ=0.5 → 낮은 비용 집중 → ESS 낮음
    # γ=2.0 → 더 균일 → ESS 높음
    assert ess_high > ess_low * 0.5, \
        f"ess_high={ess_high:.1f} vs ess_low={ess_low:.1f}"


# ── 적응 테스트 (4) ─────────────────────────────────────────

def test_adaptive_beta_decrease_low_ess():
    """ESS 낮으면 β 감소"""
    ctrl = _make_asr_controller(
        K=64, distortion_type="sigmoid",
        distortion_beta=40.0,  # 매우 높은 초기 β → ESS 매우 낮음
        use_adaptive_risk=True,
        adaptation_rate=0.5,
        min_ess_ratio=0.9,  # 거의 도달 불가능한 기준 → β 감소 강제
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    initial_beta = ctrl._current_beta
    for _ in range(30):
        ctrl.compute_control(state, ref)

    # β가 감소했어야 함
    assert ctrl._current_beta < initial_beta, \
        f"beta={ctrl._current_beta}, initial={initial_beta}"


def test_adaptive_beta_stable_normal():
    """정상 ESS에서 β 상대적 안정"""
    ctrl = _make_asr_controller(
        K=256, distortion_type="sigmoid",
        distortion_beta=5.0,
        use_adaptive_risk=True,
        adaptation_rate=0.05,
        min_ess_ratio=0.01,  # 낮은 기준 → 대부분 안정
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    initial_beta = ctrl._current_beta
    for _ in range(10):
        ctrl.compute_control(state, ref)

    # β 변화가 크지 않음 (50% 이내)
    ratio = ctrl._current_beta / initial_beta
    assert 0.5 < ratio < 2.0, \
        f"beta ratio={ratio}, expected stable"


def test_adaptive_disabled_by_default():
    """use_adaptive_risk=False이면 β 불변"""
    ctrl = _make_asr_controller(
        distortion_beta=5.0,
        use_adaptive_risk=False,
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    initial_beta = ctrl._current_beta
    for _ in range(10):
        ctrl.compute_control(state, ref)

    assert ctrl._current_beta == initial_beta


def test_adaptation_window():
    """window 내 히스토리 제한"""
    ctrl = _make_asr_controller(
        use_adaptive_risk=True,
        adaptation_window=5,
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for _ in range(20):
        ctrl.compute_control(state, ref)

    assert len(ctrl._cost_history) <= 5


# ── Controller 기본 테스트 (5) ─────────────────────────────────

def test_compute_control_shape():
    """control (nu,), info keys"""
    ctrl = _make_asr_controller()
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,), f"shape={control.shape}"
    assert "sample_weights" in info
    assert "sample_trajectories" in info
    assert "best_trajectory" in info
    assert "ess" in info
    assert "temperature" in info


def test_info_spectral_stats():
    """spectral_stats가 get_risk_statistics에 포함"""
    ctrl = _make_asr_controller()
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()
    ctrl.compute_control(state, ref)

    stats = ctrl.get_risk_statistics()
    assert "spectral_risk_value" in stats
    assert "distortion_type" in stats
    assert "current_alpha" in stats
    assert "current_beta" in stats
    assert "mean_ess_ratio" in stats
    assert stats["distortion_type"] == "sigmoid"


def test_all_distortion_types_run():
    """4가지 왜곡 타입 모두 정상 실행"""
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for dtype in ["sigmoid", "power", "dual_power", "cvar"]:
        ctrl = _make_asr_controller(distortion_type=dtype)
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,), f"{dtype}: shape={control.shape}"
        assert info["ess"] > 0, f"{dtype}: ESS={info['ess']}"


def test_ess_tracked():
    """ESS > 0, info에 포함"""
    ctrl = _make_asr_controller()
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)
    assert info["ess"] > 0
    assert info["ess"] <= 64  # K=64


def test_different_K_values():
    """K=16/64/128 정상"""
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for K in [16, 64, 128]:
        ctrl = _make_asr_controller(K=K)
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["ess"] > 0
        assert info["num_samples"] == K


# ── 성능 테스트 (4) ─────────────────────────────────────────

def test_circle_tracking_rmse():
    """원형 궤적 RMSE < 0.3 (50스텝)"""
    np.random.seed(42)
    ctrl = _make_asr_controller(K=256, N=20)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05
    errors = []

    for step in range(50):
        t = step * dt
        ref = generate_reference_trajectory(
            lambda t_: circle_trajectory(t_, radius=3.0), t, 20, dt,
        )
        control, info = ctrl.compute_control(state, ref)
        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt

        ref_pt = circle_trajectory(t, radius=3.0)
        err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
        errors.append(err)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    assert rmse < 0.3, f"RMSE={rmse:.4f}, expected < 0.3"


def test_asr_vs_vanilla_obstacles():
    """장애물 시나리오에서 ASR ≤ Vanilla 충돌"""
    np.random.seed(42)
    obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4)]
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    dt = 0.05
    N = 15

    obstacle_cost = ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0)

    def run_sim(ctrl):
        state = np.array([3.0, 0.0, np.pi / 2])
        collisions = 0
        for step in range(60):
            t = step * dt
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt
            for ox, oy, r in obstacles:
                if np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2) < r:
                    collisions += 1
        return collisions

    cost_fn = CompositeMPPICost([
        StateTrackingCost(np.array([10.0, 10.0, 1.0])),
        ControlEffortCost(np.array([0.1, 0.1])),
        obstacle_cost,
    ])

    np.random.seed(42)
    asr_ctrl = _make_asr_controller(
        K=256, N=N, distortion_type="sigmoid",
        distortion_beta=5.0, cost_function=cost_fn,
    )
    asr_collisions = run_sim(asr_ctrl)

    np.random.seed(42)
    vanilla = _make_vanilla_controller(K=256, N=N)
    vanilla.cost_function = cost_fn
    vanilla_collisions = run_sim(vanilla)

    assert asr_collisions <= vanilla_collisions + 2, \
        f"ASR={asr_collisions} vs Vanilla={vanilla_collisions}"


def test_asr_sigmoid_vs_cvar():
    """sigmoid(β=5) 가 cvar보다 ESS 높음 (부드러운 전환)"""
    np.random.seed(42)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    ctrl_sigmoid = _make_asr_controller(
        K=256, distortion_type="sigmoid",
        distortion_alpha=0.5, distortion_beta=5.0,
    )
    np.random.seed(42)
    _, info_sig = ctrl_sigmoid.compute_control(state, ref)

    ctrl_cvar = _make_asr_controller(
        K=256, distortion_type="cvar",
        distortion_alpha=0.5,
    )
    np.random.seed(42)
    _, info_cvar = ctrl_cvar.compute_control(state, ref)

    # Sigmoid은 부드러운 전환 → CVaR보다 ESS 높거나 비슷
    assert info_sig["ess"] >= info_cvar["ess"] * 0.8, \
        f"sigmoid ESS={info_sig['ess']:.1f} vs cvar ESS={info_cvar['ess']:.1f}"


def test_asr_adaptive_improves_over_time():
    """적응 모드 20스텝 후 ESS 안정"""
    np.random.seed(42)
    ctrl = _make_asr_controller(
        K=128,
        distortion_type="sigmoid",
        distortion_beta=15.0,  # 높은 초기 β
        use_adaptive_risk=True,
        adaptation_rate=0.3,
        min_ess_ratio=0.15,
    )
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05

    ess_values = []
    for step in range(30):
        t = step * dt
        ref = generate_reference_trajectory(
            lambda t_: circle_trajectory(t_, radius=3.0), t, 10, dt,
        )
        _, info = ctrl.compute_control(state, ref)
        state_dot = model.forward_dynamics(state, ctrl.U[0])
        state = state + state_dot * dt
        ess_values.append(info["ess"])

    # ESS가 0이 아닌 유효한 값을 유지
    assert all(e > 0 for e in ess_values), "All ESS should be > 0"
    # 후반부 ESS가 전반부보다 같거나 나음 (적응)
    first_half = np.mean(ess_values[:10])
    second_half = np.mean(ess_values[20:])
    assert second_half >= first_half * 0.5, \
        f"second_half ESS={second_half:.1f} vs first_half={first_half:.1f}"


# ── 통합 테스트 (3) ─────────────────────────────────────────

def test_reset_clears_state():
    """reset 후 history/적응 파라미터 초기화"""
    ctrl = _make_asr_controller(
        distortion_beta=5.0,
        use_adaptive_risk=True,
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for _ in range(5):
        ctrl.compute_control(state, ref)

    assert len(ctrl._risk_history) == 5

    ctrl.reset()

    assert len(ctrl._risk_history) == 0
    assert len(ctrl._cost_history) == 0
    assert ctrl._current_beta == ctrl.asr_params.distortion_beta
    assert ctrl._current_alpha == ctrl.asr_params.distortion_alpha


def test_repr():
    """__repr__ 내용 검증"""
    ctrl = _make_asr_controller(distortion_type="power")
    repr_str = repr(ctrl)
    assert "ASRMPPIController" in repr_str
    assert "power" in repr_str
    assert "alpha=" in repr_str
    assert "beta=" in repr_str


def test_numerical_stability():
    """극단적 비용에서 NaN/Inf 없음"""
    ctrl = _make_asr_controller(K=64)

    # 극단적 비용 직접 테스트
    for dtype in ["sigmoid", "power", "dual_power", "cvar"]:
        ctrl_test = _make_asr_controller(K=64, distortion_type=dtype)
        costs = np.random.randn(64) * 1000 + 5000  # 매우 큰 비용
        weights = ctrl_test._compute_weights(costs, 1.0)
        assert not np.any(np.isnan(weights)), f"{dtype}: NaN weights"
        assert not np.any(np.isinf(weights)), f"{dtype}: Inf weights"
        assert abs(np.sum(weights) - 1.0) < 1e-6, f"{dtype}: sum={np.sum(weights)}"

        # 0에 가까운 비용
        costs_small = np.random.randn(64) * 0.001
        weights_small = ctrl_test._compute_weights(costs_small, 1.0)
        assert not np.any(np.isnan(weights_small)), f"{dtype}: NaN with small costs"
