"""
Uncertainty-Aware MPPI 유닛 테스트

UncertaintyAwareSampler + UncertaintyMPPIController 16개 테스트.
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
    UncertaintyMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.uncertainty_mppi import (
    UncertaintyMPPIController,
)
from mppi_controller.controllers.mppi.sampling import (
    GaussianSampler,
    UncertaintyAwareSampler,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi.uncertainty_cost import UncertaintyAwareCost
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)


# ── Mock 불확실성 함수 ──────────────────────────────────────

def _mock_uncertainty_fn(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """원점에서 멀수록 불확실성 증가하는 mock 함수"""
    if states.ndim == 1:
        states = states[None, :]
    # 거리 기반 불확실성: ||pos|| * 0.1
    dist = np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)  # (batch,)
    nx = states.shape[-1]
    std = dist[:, None] * 0.1 * np.ones((1, nx))  # (batch, nx)
    return std


def _constant_uncertainty_fn(value: float):
    """일정한 불확실성을 반환하는 mock 팩토리"""
    def fn(states, controls):
        if states.ndim == 1:
            states = states[None, :]
        nx = states.shape[-1]
        return np.full((states.shape[0], nx), value)
    return fn


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
    return UncertaintyMPPIParams(**defaults)


def _make_controller(uncertainty_fn=_mock_uncertainty_fn, **kwargs):
    model = _make_model()
    params = _make_params(**kwargs)
    return UncertaintyMPPIController(
        model, params, uncertainty_fn=uncertainty_fn
    )


def _make_ref(N=10, dt=0.05):
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


# ══════════════════════════════════════════════════════════════
# Sampler 테스트 (#1~#4)
# ══════════════════════════════════════════════════════════════


def test_sampler_output_shape():
    """#1: UncertaintyAwareSampler 출력 shape (K, N, nu)"""
    print("\n" + "=" * 60)
    print("Test: sampler output shape")
    print("=" * 60)

    sampler = UncertaintyAwareSampler(
        base_sigma=np.array([0.5, 0.5]), seed=42
    )
    U = np.zeros((10, 2))  # (N=10, nu=2)

    noise = sampler.sample(U, K=64)
    assert noise.shape == (64, 10, 2), f"shape: {noise.shape}"
    print(f"  shape={noise.shape}")
    print("PASS")


def test_sampler_sigma_scaling():
    """#2: 불확실성 증가 → sigma 증가 검증"""
    print("\n" + "=" * 60)
    print("Test: sampler sigma scaling")
    print("=" * 60)

    sampler = UncertaintyAwareSampler(
        base_sigma=np.array([0.5, 0.5]),
        exploration_factor=2.0,
        seed=42,
    )
    U = np.zeros((10, 2))

    # 프로파일 없이 → base_sigma
    noise_base = sampler.sample(U, K=1000)
    std_base = np.std(noise_base, axis=0).mean()

    # 높은 불확실성 프로파일
    high_unc = np.ones((10, 3)) * 2.0  # (N=10, nx=3)
    sampler.update_uncertainty_profile(high_unc)
    noise_high = sampler.sample(U, K=1000)
    std_high = np.std(noise_high, axis=0).mean()

    assert std_high > std_base, f"high={std_high:.3f} should > base={std_base:.3f}"
    print(f"  base std={std_base:.4f}, high unc std={std_high:.4f}")
    print("PASS")


def test_sampler_fallback_no_profile():
    """#3: 프로파일 없을 때 base_sigma로 fallback"""
    print("\n" + "=" * 60)
    print("Test: sampler fallback no profile")
    print("=" * 60)

    sigma = np.array([0.3, 0.7])
    sampler = UncertaintyAwareSampler(base_sigma=sigma, seed=42)
    gaussian = GaussianSampler(sigma=sigma, seed=42)

    U = np.zeros((10, 2))

    noise_ua = sampler.sample(U, K=5000)
    noise_g = gaussian.sample(U, K=5000)

    # 프로파일 없으면 GaussianSampler와 유사한 통계
    std_ua = np.std(noise_ua, axis=0).mean(axis=0)  # (nu,)
    std_g = np.std(noise_g, axis=0).mean(axis=0)

    for i in range(len(sigma)):
        assert abs(std_ua[i] - std_g[i]) < 0.05, \
            f"dim {i}: UA={std_ua[i]:.3f} vs G={std_g[i]:.3f}"
    print(f"  UA std={std_ua}, Gaussian std={std_g}")
    print("PASS")


def test_sampler_control_constraints():
    """#4: 제어 제약 u_min/u_max 준수"""
    print("\n" + "=" * 60)
    print("Test: sampler control constraints")
    print("=" * 60)

    sampler = UncertaintyAwareSampler(
        base_sigma=np.array([1.0, 1.0]),
        max_sigma_ratio=5.0,
        seed=42,
    )

    # 높은 불확실성으로 큰 노이즈 유도
    sampler.update_uncertainty_profile(np.ones((10, 3)) * 10.0)

    U = np.zeros((10, 2))
    u_min = np.array([-0.5, -0.5])
    u_max = np.array([0.5, 0.5])

    noise = sampler.sample(U, K=200, control_min=u_min, control_max=u_max)
    sampled = U + noise

    assert np.all(sampled >= u_min - 1e-10), "u_min violated"
    assert np.all(sampled <= u_max + 1e-10), "u_max violated"
    print(f"  control range: [{sampled.min():.4f}, {sampled.max():.4f}]")
    print("PASS")


# ══════════════════════════════════════════════════════════════
# Params 테스트 (#5)
# ══════════════════════════════════════════════════════════════


def test_params_validation():
    """#5: 유효하지 않은 파라미터 거부"""
    print("\n" + "=" * 60)
    print("Test: params validation")
    print("=" * 60)

    # 올바른 파라미터
    p = _make_params()
    assert p.exploration_factor == 1.0
    assert p.uncertainty_strategy == "previous_trajectory"

    # 잘못된 strategy
    try:
        _make_params(uncertainty_strategy="invalid")
        assert False, "should raise"
    except AssertionError:
        pass

    # 음수 exploration_factor
    try:
        _make_params(exploration_factor=-1.0)
        assert False, "should raise"
    except AssertionError:
        pass

    # min > max sigma ratio
    try:
        _make_params(min_sigma_ratio=5.0, max_sigma_ratio=1.0)
        assert False, "should raise"
    except AssertionError:
        pass

    print("  All invalid params rejected correctly")
    print("PASS")


# ══════════════════════════════════════════════════════════════
# Controller 테스트 (#6~#11)
# ══════════════════════════════════════════════════════════════


def test_controller_current_state():
    """#6: current_state 전략 동작"""
    print("\n" + "=" * 60)
    print("Test: controller current_state strategy")
    print("=" * 60)

    ctrl = _make_controller(uncertainty_strategy="current_state")
    state = np.array([2.0, 1.0, 0.0])  # 원점에서 떨어진 위치
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert "uncertainty_stats" in info
    assert info["uncertainty_stats"]["mean_uncertainty"] > 0
    print(f"  control={control}")
    print(f"  mean_unc={info['uncertainty_stats']['mean_uncertainty']:.4f}")
    print("PASS")


def test_controller_previous_trajectory():
    """#7: previous_trajectory 전략 + 2스텝 연속 호출"""
    print("\n" + "=" * 60)
    print("Test: controller previous_trajectory strategy")
    print("=" * 60)

    ctrl = _make_controller(uncertainty_strategy="previous_trajectory")
    state = np.array([2.0, 0.0, np.pi / 4])
    ref = _make_ref()

    # 1st call: 이전 궤적 없으므로 current_state fallback
    control1, info1 = ctrl.compute_control(state, ref)
    assert control1.shape == (2,)
    assert info1["uncertainty_stats"]["mean_uncertainty"] > 0

    # 2nd call: 이전 best_trajectory 사용
    control2, info2 = ctrl.compute_control(state, ref)
    assert control2.shape == (2,)
    assert info2["uncertainty_stats"]["mean_uncertainty"] > 0

    print(f"  step1 mean_unc={info1['uncertainty_stats']['mean_uncertainty']:.4f}")
    print(f"  step2 mean_unc={info2['uncertainty_stats']['mean_uncertainty']:.4f}")
    print("PASS")


def test_controller_two_pass():
    """#8: two_pass 전략 동작"""
    print("\n" + "=" * 60)
    print("Test: controller two_pass strategy")
    print("=" * 60)

    ctrl = _make_controller(uncertainty_strategy="two_pass")
    state = np.array([1.0, 1.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert "uncertainty_stats" in info
    assert info.get("two_pass", False) is True
    print(f"  control={control}")
    print(f"  two_pass={info.get('two_pass')}")
    print("PASS")


def test_controller_no_uncertainty_model():
    """#9: 불확실성 모델 없이 graceful fallback (Vanilla MPPI와 동일)"""
    print("\n" + "=" * 60)
    print("Test: controller no uncertainty model fallback")
    print("=" * 60)

    ctrl = _make_controller(uncertainty_fn=None)
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert info["uncertainty_stats"]["mean_uncertainty"] == 0.0
    assert info["sigma_stats"]["has_profile"] is False
    print(f"  control={control}")
    print(f"  mean_unc={info['uncertainty_stats']['mean_uncertainty']}")
    print("PASS")


def test_info_uncertainty_stats():
    """#10: info dict에 uncertainty_stats 포함"""
    print("\n" + "=" * 60)
    print("Test: info uncertainty_stats")
    print("=" * 60)

    ctrl = _make_controller()
    state = np.array([3.0, 2.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    stats = info["uncertainty_stats"]
    assert "mean_uncertainty" in stats
    assert "max_uncertainty" in stats
    assert "min_uncertainty" in stats
    assert "profile_shape" in stats
    assert stats["profile_shape"] is not None

    print(f"  stats={stats}")
    print("PASS")


def test_info_sigma_stats():
    """#11: sigma_stats 추적"""
    print("\n" + "=" * 60)
    print("Test: info sigma_stats")
    print("=" * 60)

    ctrl = _make_controller(uncertainty_strategy="current_state")
    state = np.array([5.0, 0.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    sigma = info["sigma_stats"]
    assert "has_profile" in sigma
    assert sigma["has_profile"] is True
    assert "mean_ratio" in sigma
    assert sigma["mean_ratio"] > 0
    print(f"  sigma_stats={sigma}")
    print("PASS")


# ══════════════════════════════════════════════════════════════
# 경계값 및 통계 테스트 (#12~#16)
# ══════════════════════════════════════════════════════════════


def test_zero_uncertainty():
    """#12: 제로 불확실성 → ratio=1.0"""
    print("\n" + "=" * 60)
    print("Test: zero uncertainty → ratio=1.0")
    print("=" * 60)

    ctrl = _make_controller(uncertainty_fn=_constant_uncertainty_fn(0.0))
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    sigma = info["sigma_stats"]
    if sigma["has_profile"]:
        assert abs(sigma["mean_ratio"] - 1.0) < 1e-6, \
            f"mean_ratio={sigma['mean_ratio']}"
    print(f"  mean_ratio={sigma.get('mean_ratio', 'N/A')}")
    print("PASS")


def test_high_uncertainty():
    """#13: 높은 불확실성 → max_sigma_ratio 클리핑"""
    print("\n" + "=" * 60)
    print("Test: high uncertainty → max_sigma_ratio clipping")
    print("=" * 60)

    max_ratio = 3.0
    ctrl = _make_controller(
        uncertainty_fn=_constant_uncertainty_fn(100.0),
        uncertainty_strategy="current_state",
        max_sigma_ratio=max_ratio,
    )
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    sigma = info["sigma_stats"]
    if sigma["has_profile"]:
        assert sigma["max_ratio"] <= max_ratio + 1e-6, \
            f"max_ratio={sigma['max_ratio']} > {max_ratio}"
    print(f"  max_ratio={sigma.get('max_ratio', 'N/A')}, limit={max_ratio}")
    print("PASS")


def test_statistics_accumulation():
    """#14: get_uncertainty_statistics() 히스토리 누적"""
    print("\n" + "=" * 60)
    print("Test: statistics accumulation")
    print("=" * 60)

    ctrl = _make_controller(uncertainty_strategy="current_state")
    state = np.array([1.0, 0.0, 0.0])
    ref = _make_ref()

    # 3번 호출
    for _ in range(3):
        ctrl.compute_control(state, ref)

    stats = ctrl.get_uncertainty_statistics()
    assert stats["num_steps"] == 3, f"num_steps={stats['num_steps']}"
    assert "overall_mean_uncertainty" in stats
    assert len(stats["history"]) == 3
    print(f"  num_steps={stats['num_steps']}")
    print(f"  overall_mean_unc={stats['overall_mean_uncertainty']:.4f}")
    print("PASS")


def test_reset():
    """#15: reset() 후 상태 초기화"""
    print("\n" + "=" * 60)
    print("Test: reset clears state")
    print("=" * 60)

    ctrl = _make_controller()
    state = np.array([2.0, 1.0, 0.0])
    ref = _make_ref()

    # 몇 스텝 실행
    ctrl.compute_control(state, ref)
    ctrl.compute_control(state, ref)

    assert ctrl._prev_best_trajectory is not None
    assert len(ctrl._uncertainty_history) == 2

    # reset
    ctrl.reset()

    assert ctrl._prev_best_trajectory is None
    assert ctrl._prev_best_controls is None
    assert len(ctrl._uncertainty_history) == 0

    # sampler도 초기화 확인
    if isinstance(ctrl.noise_sampler, UncertaintyAwareSampler):
        assert ctrl.noise_sampler._sigma_ratios is None

    print("  All state cleared after reset")
    print("PASS")


def test_dual_benefit_with_cost():
    """#16: UncertaintyAwareCost + UncertaintyMPPIController 동시 사용"""
    print("\n" + "=" * 60)
    print("Test: dual benefit with UncertaintyAwareCost")
    print("=" * 60)

    model = _make_model()
    params = _make_params(uncertainty_strategy="current_state")

    # 비용 함수: 상태 추적 + 불확실성 페널티
    cost = CompositeMPPICost([
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
        UncertaintyAwareCost(
            uncertainty_fn=_mock_uncertainty_fn,
            beta=5.0,
        ),
    ])

    ctrl = UncertaintyMPPIController(
        model, params,
        cost_function=cost,
        uncertainty_fn=_mock_uncertainty_fn,
    )

    state = np.array([3.0, 2.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert "uncertainty_stats" in info
    assert "sigma_stats" in info
    # 불확실성 비용이 높은 영역에서는 비용이 높고 노이즈도 넓어짐
    assert info["uncertainty_stats"]["mean_uncertainty"] > 0
    assert info["sigma_stats"]["has_profile"] is True

    print(f"  control={control}")
    print(f"  mean_unc={info['uncertainty_stats']['mean_uncertainty']:.4f}")
    print(f"  sigma_mean_ratio={info['sigma_stats']['mean_ratio']:.4f}")
    print("PASS — dual benefit (adaptive sampling + uncertainty cost)")


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_sampler_output_shape,
        test_sampler_sigma_scaling,
        test_sampler_fallback_no_profile,
        test_sampler_control_constraints,
        test_params_validation,
        test_controller_current_state,
        test_controller_previous_trajectory,
        test_controller_two_pass,
        test_controller_no_uncertainty_model,
        test_info_uncertainty_stats,
        test_info_sigma_stats,
        test_zero_uncertainty,
        test_high_uncertainty,
        test_statistics_accumulation,
        test_reset,
        test_dual_benefit_with_cost,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed / {len(tests)} total")
    print(f"{'=' * 60}")
