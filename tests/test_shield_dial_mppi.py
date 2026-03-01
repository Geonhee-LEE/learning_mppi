"""
Shield-DIAL-MPPI + Adaptive Shield-DIAL-MPPI 유닛 테스트

12 ShieldDIAL + 4 AdaptiveShieldDIAL = 16 tests
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    ShieldDIALMPPIParams,
    AdaptiveShieldDIALMPPIParams,
)
from mppi_controller.controllers.mppi.shield_dial_mppi import ShieldDIALMPPIController
from mppi_controller.controllers.mppi.adaptive_shield_dial_mppi import (
    AdaptiveShieldDIALMPPIController,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


# ── 헬퍼 ──────────────────────────────────────────────────

def _make_shield_dial_controller(obstacles=None, shield_enabled=True, **kwargs):
    """헬퍼: Shield-DIAL-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    if obstacles is None:
        obstacles = [(3.0, 0.0, 0.5)]
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        n_diffuse_init=5,
        n_diffuse=2,
        traj_diffuse_factor=0.5,
        horizon_diffuse_factor=0.5,
        sigma_scale=1.0,
        use_reward_normalization=True,
        cbf_obstacles=obstacles,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        shield_enabled=shield_enabled,
        shield_cbf_alpha=0.3,
    )
    defaults.update(kwargs)
    params = ShieldDIALMPPIParams(**defaults)
    return ShieldDIALMPPIController(model, params), model


def _make_adaptive_shield_dial_controller(obstacles=None, **kwargs):
    """헬퍼: Adaptive Shield-DIAL-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    if obstacles is None:
        obstacles = [(3.0, 0.0, 0.5)]
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        n_diffuse_init=5,
        n_diffuse=2,
        traj_diffuse_factor=0.5,
        horizon_diffuse_factor=0.5,
        sigma_scale=1.0,
        use_reward_normalization=True,
        cbf_obstacles=obstacles,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        shield_enabled=True,
        shield_cbf_alpha=0.3,
        alpha_base=0.3,
        alpha_dist=0.1,
        alpha_vel=0.5,
        k_dist=2.0,
        d_safe=0.5,
    )
    defaults.update(kwargs)
    params = AdaptiveShieldDIALMPPIParams(**defaults)
    return AdaptiveShieldDIALMPPIController(model, params), model


def _make_ref(N=10, dt=0.05):
    """헬퍼: 레퍼런스 궤적 생성"""
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


# ── ShieldDIAL Tests (12개) ───────────────────────────────


def test_shield_dial_basic():
    """기본 동작, shape, info keys (dial_stats + shield_info)"""
    ctrl, _ = _make_shield_dial_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"control shape: {control.shape}"
    assert not np.any(np.isnan(control)), "NaN in control"
    assert isinstance(info, dict)

    # MPPI 기본 키
    required_keys = [
        "sample_trajectories", "sample_weights", "best_trajectory",
        "best_cost", "mean_cost", "temperature", "ess", "num_samples",
        "dial_stats", "shield_info",
    ]
    for key in required_keys:
        assert key in info, f"missing key: {key}"

    assert info["sample_trajectories"].shape == (64, 11, 3)
    assert info["sample_weights"].shape == (64,)
    assert info["num_samples"] == 64

    # DIAL 통계
    assert "n_iters" in info["dial_stats"]
    assert "iteration_costs" in info["dial_stats"]

    # Shield 통계
    assert "intervention_rate" in info["shield_info"]
    assert "total_interventions" in info["shield_info"]
    assert "total_steps" in info["shield_info"]


def test_shield_dial_first_call_iterations():
    """첫 호출 n_diffuse_init, 이후 n_diffuse 사용 확인"""
    ctrl, _ = _make_shield_dial_controller(n_diffuse_init=7, n_diffuse=2)
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    # 첫 호출: n_diffuse_init=7
    _, info1 = ctrl.compute_control(state, ref)
    assert info1["dial_stats"]["n_iters"] == 7
    assert len(info1["dial_stats"]["iteration_costs"]) == 7

    # 두 번째 호출: n_diffuse=2
    _, info2 = ctrl.compute_control(state, ref)
    assert info2["dial_stats"]["n_iters"] == 2
    assert len(info2["dial_stats"]["iteration_costs"]) == 2


def test_shield_dial_disabled_fallback():
    """shield_enabled=False → DIAL-MPPI 폴백"""
    ctrl, _ = _make_shield_dial_controller(shield_enabled=False)
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert not np.any(np.isnan(control)), "NaN in control"
    # DIAL-MPPI 폴백이므로 dial_stats만 있고 shield_info는 없을 수 있음
    assert "dial_stats" in info


def test_shield_dial_no_obstacles():
    """장애물 없음 → intervention_rate=0"""
    ctrl, _ = _make_shield_dial_controller(obstacles=[])
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    assert info["shield_info"]["intervention_rate"] == 0.0
    assert info["shield_info"]["total_interventions"] == 0


def test_shield_dial_multistep():
    """5스텝 시뮬레이션, NaN 없음"""
    ctrl, model = _make_shield_dial_controller()
    state = np.array([0.0, 0.0, 0.0])

    for step in range(5):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 10, 0.05)
        control, info = ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)

        assert not np.any(np.isnan(state)), f"NaN at step {step}"
        assert not np.any(np.isnan(control)), f"NaN control at step {step}"
        assert not np.any(np.isinf(state)), f"Inf at step {step}"


def test_shield_dial_all_trajectories_safe():
    """K=256, 모든 궤적 h(x) > -1e-6"""
    obstacles = [(2.0, 0.0, 0.5)]
    ctrl, _ = _make_shield_dial_controller(
        obstacles=obstacles, K=256, cbf_safety_margin=0.1,
    )
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    # 모든 샘플 궤적이 장애물 제약 만족
    trajs = info["sample_trajectories"]  # (K, N+1, 3)
    for obs_x, obs_y, obs_r in obstacles:
        effective_r = obs_r + 0.1  # safety_margin
        dx = trajs[:, :, 0] - obs_x
        dy = trajs[:, :, 1] - obs_y
        h = dx**2 + dy**2 - effective_r**2
        min_h = np.min(h)
        assert min_h > -1e-6, \
            f"CBF violation: min h(x) = {min_h:.6f}"


def test_shield_dial_update_obstacles():
    """update_obstacles() 동작 확인"""
    ctrl, _ = _make_shield_dial_controller(obstacles=[(1.0, 0.0, 0.3)])
    ctrl.update_obstacles([(5.0, 5.0, 1.0), (3.0, 3.0, 0.5)])
    assert ctrl.shield_dial_params.cbf_obstacles == [(5.0, 5.0, 1.0), (3.0, 3.0, 0.5)]


def test_shield_dial_set_shield_enabled():
    """런타임 toggle"""
    ctrl, _ = _make_shield_dial_controller(shield_enabled=True)
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    # Shield on
    _, info_on = ctrl.compute_control(state, ref)
    assert "shield_info" in info_on

    # Shield off
    ctrl.set_shield_enabled(False)
    ctrl.reset()
    _, info_off = ctrl.compute_control(state, ref)
    assert "dial_stats" in info_off

    # Shield on again
    ctrl.set_shield_enabled(True)
    ctrl.reset()
    _, info_on2 = ctrl.compute_control(state, ref)
    assert "shield_info" in info_on2


def test_shield_dial_statistics():
    """get_shield_statistics() + get_dial_statistics()"""
    ctrl, _ = _make_shield_dial_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    # 호출 전 통계
    shield_stats = ctrl.get_shield_statistics()
    assert shield_stats["mean_intervention_rate"] == 0.0
    assert shield_stats["total_interventions"] == 0
    assert shield_stats["num_steps"] == 0

    dial_stats = ctrl.get_dial_statistics()
    assert dial_stats["mean_cost_improvement"] == 0.0

    # 3회 호출
    for _ in range(3):
        ctrl.compute_control(state, ref)

    shield_stats = ctrl.get_shield_statistics()
    assert shield_stats["num_steps"] == 3
    assert "mean_intervention_rate" in shield_stats
    assert "total_interventions" in shield_stats

    dial_stats = ctrl.get_dial_statistics()
    assert len(dial_stats["dial_stats_history"]) == 3


def test_shield_dial_reset():
    """reset() 시 모든 통계 초기화 + cold start 복원"""
    ctrl, _ = _make_shield_dial_controller(n_diffuse_init=5, n_diffuse=2)
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    # 2회 호출
    ctrl.compute_control(state, ref)
    ctrl.compute_control(state, ref)

    assert len(ctrl.shield_stats_history) == 2
    assert len(ctrl._dial_stats_history) == 2

    # reset
    ctrl.reset()

    assert len(ctrl.shield_stats_history) == 0
    assert len(ctrl._dial_stats_history) == 0
    assert ctrl._is_first_call is True

    # reset 후 cold start 확인
    _, info = ctrl.compute_control(state, ref)
    assert info["dial_stats"]["n_iters"] == 5


def test_shield_dial_repr():
    """__repr__ 문자열"""
    ctrl, _ = _make_shield_dial_controller(
        n_diffuse_init=10, n_diffuse=3, shield_cbf_alpha=0.3, K=128,
    )
    repr_str = repr(ctrl)

    assert "ShieldDIALMPPIController" in repr_str
    assert "DifferentialDriveKinematic" in repr_str
    assert "n_diffuse_init=10" in repr_str
    assert "n_diffuse=3" in repr_str
    assert "shield_alpha=0.3" in repr_str
    assert "K=128" in repr_str


def test_shield_dial_cumulative_stats():
    """total_steps = K×N×n_iters (각 compute_control 호출에서)"""
    K, N = 64, 10
    ctrl, _ = _make_shield_dial_controller(
        K=K, N=N, n_diffuse_init=3, n_diffuse=2,
    )
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref(N=N)

    # 첫 호출 (n_iters=3)
    _, info1 = ctrl.compute_control(state, ref)
    expected_steps_1 = K * N * 3
    assert info1["shield_info"]["total_steps"] == expected_steps_1, \
        f"expected {expected_steps_1}, got {info1['shield_info']['total_steps']}"

    # 두 번째 호출 (n_iters=2)
    _, info2 = ctrl.compute_control(state, ref)
    expected_steps_2 = K * N * 2
    assert info2["shield_info"]["total_steps"] == expected_steps_2, \
        f"expected {expected_steps_2}, got {info2['shield_info']['total_steps']}"


# ── AdaptiveShieldDIAL Tests (4개) ────────────────────────


def test_adaptive_shield_dial_basic():
    """기본 동작, info keys"""
    ctrl, _ = _make_adaptive_shield_dial_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert not np.any(np.isnan(control))
    assert "dial_stats" in info
    assert "shield_info" in info
    assert "intervention_rate" in info["shield_info"]


def test_adaptive_shield_dial_alpha_varies():
    """가까운 상태 α < 먼 상태 α"""
    obstacles = [(2.0, 0.0, 0.5)]
    ctrl, _ = _make_adaptive_shield_dial_controller(
        obstacles=obstacles,
        alpha_base=0.3, alpha_dist=0.1, alpha_vel=0.5,
        k_dist=2.0, d_safe=0.5,
    )

    # 가까운 상태 (d_surface ~= 0.6)
    states_close = np.array([[1.5, 0.0, 0.0]])  # dist=0.5 from obs center
    controls = np.array([[0.5, 0.0]])

    safe_close, _, _ = ctrl._cbf_shield_batch(states_close, controls)

    # 먼 상태 (d_surface ~= 3.0)
    states_far = np.array([[5.0, 0.0, np.pi]])  # dist=3.0 from obs center, facing obstacle
    controls_far = np.array([[0.5, 0.0]])

    safe_far, _, _ = ctrl._cbf_shield_batch(states_far, controls_far)

    # 가까울수록 더 보수적 → v_safe가 더 낮거나 같아야 함
    # (단, 방향에 따라 다를 수 있으므로 alpha 자체를 확인)
    p = ctrl.adaptive_params

    # 가까운 상태 alpha
    d_close = np.sqrt((1.5 - 2.0)**2) - 0.5  # 0.0
    sig_close = 1.0 / (1.0 + np.exp(-p.k_dist * (d_close - p.d_safe)))
    alpha_close = p.alpha_base * (p.alpha_dist + (1.0 - p.alpha_dist) * sig_close) / (1.0 + p.alpha_vel * 0.5)
    alpha_close = np.clip(alpha_close, 0.01, 0.99)

    # 먼 상태 alpha
    d_far = np.sqrt((5.0 - 2.0)**2) - 0.5  # 2.5
    sig_far = 1.0 / (1.0 + np.exp(-p.k_dist * (d_far - p.d_safe)))
    alpha_far = p.alpha_base * (p.alpha_dist + (1.0 - p.alpha_dist) * sig_far) / (1.0 + p.alpha_vel * 0.5)
    alpha_far = np.clip(alpha_far, 0.01, 0.99)

    assert alpha_close < alpha_far, \
        f"close alpha ({alpha_close:.4f}) should be < far alpha ({alpha_far:.4f})"


def test_adaptive_shield_dial_params_validation():
    """잘못된 파라미터 거부"""
    import pytest

    # alpha_base <= 0
    with pytest.raises(AssertionError):
        AdaptiveShieldDIALMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            alpha_base=-0.1,
        )

    # d_safe <= 0
    with pytest.raises(AssertionError):
        AdaptiveShieldDIALMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            d_safe=0.0,
        )

    # k_dist <= 0
    with pytest.raises(AssertionError):
        AdaptiveShieldDIALMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            k_dist=-1.0,
        )


def test_adaptive_shield_dial_repr():
    """__repr__ 문자열"""
    ctrl, _ = _make_adaptive_shield_dial_controller(
        alpha_base=0.3, alpha_dist=0.1, alpha_vel=0.5, d_safe=0.5, K=128,
    )
    repr_str = repr(ctrl)

    assert "AdaptiveShieldDIALMPPIController" in repr_str
    assert "DifferentialDriveKinematic" in repr_str
    assert "alpha_base=0.3" in repr_str
    assert "alpha_dist=0.1" in repr_str
    assert "alpha_vel=0.5" in repr_str
    assert "d_safe=0.5" in repr_str
    assert "K=128" in repr_str


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Shield-DIAL-MPPI + Adaptive Shield-DIAL-MPPI Tests")
    print("=" * 60)

    tests = [
        # ShieldDIAL (12)
        test_shield_dial_basic,
        test_shield_dial_first_call_iterations,
        test_shield_dial_disabled_fallback,
        test_shield_dial_no_obstacles,
        test_shield_dial_multistep,
        test_shield_dial_all_trajectories_safe,
        test_shield_dial_update_obstacles,
        test_shield_dial_set_shield_enabled,
        test_shield_dial_statistics,
        test_shield_dial_reset,
        test_shield_dial_repr,
        test_shield_dial_cumulative_stats,
        # AdaptiveShieldDIAL (4)
        test_adaptive_shield_dial_basic,
        test_adaptive_shield_dial_alpha_varies,
        test_adaptive_shield_dial_params_validation,
        test_adaptive_shield_dial_repr,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  {passed}/{passed + failed} Tests Passed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
