"""
DIAL-MPPI (Diffusion Annealing MPPI) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, DIALMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


def _make_dial_controller(**kwargs):
    """헬퍼: DIAL-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
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
    )
    defaults.update(kwargs)
    params = DIALMPPIParams(**defaults)
    return DIALMPPIController(model, params)


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
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


# ── Tests ──────────────────────────────────────────────────


def test_dial_basic():
    """기본 동작, 출력 shape, info dict 키"""
    ctrl = _make_dial_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"control shape: {control.shape}"
    assert isinstance(info, dict)

    required_keys = [
        "sample_trajectories", "sample_weights", "best_trajectory",
        "best_cost", "mean_cost", "temperature", "ess", "num_samples",
        "dial_stats",
    ]
    for key in required_keys:
        assert key in info, f"missing key: {key}"

    assert info["sample_trajectories"].shape == (64, 11, 3)
    assert info["sample_weights"].shape == (64,)
    assert info["num_samples"] == 64
    assert "n_iters" in info["dial_stats"]
    assert "iteration_costs" in info["dial_stats"]


def test_first_call_init_iterations():
    """첫 호출 n_diffuse_init, 이후 n_diffuse 사용 확인"""
    ctrl = _make_dial_controller(n_diffuse_init=7, n_diffuse=2)
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # 첫 호출: n_diffuse_init=7
    _, info1 = ctrl.compute_control(state, ref)
    assert info1["dial_stats"]["n_iters"] == 7, \
        f"first call n_iters={info1['dial_stats']['n_iters']}, expected 7"
    assert len(info1["dial_stats"]["iteration_costs"]) == 7

    # 두 번째 호출: n_diffuse=2
    _, info2 = ctrl.compute_control(state, ref)
    assert info2["dial_stats"]["n_iters"] == 2, \
        f"second call n_iters={info2['dial_stats']['n_iters']}, expected 2"
    assert len(info2["dial_stats"]["iteration_costs"]) == 2


def test_reset_restores_first_call():
    """reset() 후 다시 n_diffuse_init 사용"""
    ctrl = _make_dial_controller(n_diffuse_init=5, n_diffuse=2)
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # 첫 호출
    ctrl.compute_control(state, ref)
    # 두 번째 호출 (warm start)
    _, info2 = ctrl.compute_control(state, ref)
    assert info2["dial_stats"]["n_iters"] == 2

    # reset 후 다시 cold start
    ctrl.reset()
    _, info3 = ctrl.compute_control(state, ref)
    assert info3["dial_stats"]["n_iters"] == 5, \
        f"after reset n_iters={info3['dial_stats']['n_iters']}, expected 5"

    # U가 초기화 확인
    assert ctrl._dial_stats_history is not None
    assert len(ctrl._dial_stats_history) == 1  # reset 후 1회만 호출


def test_cost_improvement():
    """반복별 비용 기록 및 어닐링 노이즈 감소 확인"""
    np.random.seed(42)
    ctrl = _make_dial_controller(
        n_diffuse_init=10, K=256,
        traj_diffuse_factor=0.5,
        sigma=np.array([1.0, 1.0]),
    )
    state = np.array([3.0, 2.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)
    costs = info["dial_stats"]["iteration_costs"]

    # 반복 횟수 올바른지 확인
    assert len(costs) == 10, f"expected 10 iteration costs, got {len(costs)}"

    # 모든 비용이 유한한 양수인지 확인
    for i, c in enumerate(costs):
        assert np.isfinite(c), f"iteration {i} cost is not finite: {c}"
        assert c >= 0, f"iteration {i} cost is negative: {c}"

    # 어닐링 효과: 마지막 반복의 best_cost가 합리적인 범위 내
    assert info["best_cost"] < 1e6, f"best_cost too high: {info['best_cost']}"

    # cost_improvement 기록 확인
    dial_stats = info["dial_stats"]
    expected_improvement = costs[0] - costs[-1]
    assert abs(dial_stats["cost_improvement"] - expected_improvement) < 1e-10


def test_horizon_noise_profile():
    """호라이즌 노이즈 프로파일 shape 및 단조 증가 확인"""
    ctrl = _make_dial_controller(N=20, horizon_diffuse_factor=0.5, sigma_scale=1.0)

    profile = ctrl._horizon_profile
    assert profile.shape == (20,), f"profile shape: {profile.shape}"

    # t=0(가까운 미래) < t=N-1(먼 미래): 단조 증가
    for t in range(len(profile) - 1):
        assert profile[t] <= profile[t + 1] + 1e-10, \
            f"not monotonically increasing at t={t}: {profile[t]} > {profile[t+1]}"

    # t=N-1에서 sigma_scale 값
    assert abs(profile[-1] - 1.0) < 1e-10, \
        f"profile[-1]={profile[-1]}, expected sigma_scale=1.0"


def test_annealing_effect():
    """공격적(0.1) vs 완만(0.9) 어닐링 비교"""
    np.random.seed(42)
    state = np.array([3.0, 2.0, 0.0])  # 궤적에서 떨어진 위치
    ref = _make_ref()

    # 공격적 어닐링
    ctrl_aggressive = _make_dial_controller(
        traj_diffuse_factor=0.1, n_diffuse_init=5, K=256,
        sigma=np.array([1.0, 1.0]),
    )
    control_agg, info_agg = ctrl_aggressive.compute_control(state, ref)

    # 완만한 어닐링
    np.random.seed(42)
    ctrl_gentle = _make_dial_controller(
        traj_diffuse_factor=0.9, n_diffuse_init=5, K=256,
        sigma=np.array([1.0, 1.0]),
    )
    control_gentle, info_gentle = ctrl_gentle.compute_control(state, ref)

    # 둘 다 유효한 제어 출력
    assert not np.any(np.isnan(control_agg)), "NaN in aggressive control"
    assert not np.any(np.isnan(control_gentle)), "NaN in gentle control"
    assert not np.any(np.isinf(control_agg)), "Inf in aggressive control"
    assert not np.any(np.isinf(control_gentle)), "Inf in gentle control"

    # 어닐링 스케줄이 다르면 반복별 비용 패턴도 달라야 함
    costs_agg = info_agg["dial_stats"]["iteration_costs"]
    costs_gentle = info_gentle["dial_stats"]["iteration_costs"]
    assert not np.allclose(costs_agg, costs_gentle, atol=1e-3), \
        "aggressive and gentle should have different cost trajectories"


def test_reward_normalization_toggle():
    """보상 정규화 on/off 모두 유효한 출력"""
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # 정규화 on
    np.random.seed(42)
    ctrl_on = _make_dial_controller(use_reward_normalization=True)
    control_on, info_on = ctrl_on.compute_control(state, ref)

    # 정규화 off
    np.random.seed(42)
    ctrl_off = _make_dial_controller(use_reward_normalization=False)
    control_off, info_off = ctrl_off.compute_control(state, ref)

    # 둘 다 유효
    assert not np.any(np.isnan(control_on)), "NaN with normalization on"
    assert not np.any(np.isnan(control_off)), "NaN with normalization off"
    assert not np.any(np.isinf(control_on)), "Inf with normalization on"
    assert not np.any(np.isinf(control_off)), "Inf with normalization off"

    # 가중치 합 = 1
    assert abs(np.sum(info_on["sample_weights"]) - 1.0) < 1e-6
    assert abs(np.sum(info_off["sample_weights"]) - 1.0) < 1e-6


def test_dial_vs_vanilla_tracking():
    """원형 궤적 추적 RMSE 비교"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # Vanilla MPPI
    vanilla_ctrl = _make_vanilla_controller(K=256, N=15)
    state = np.array([5.0, 0.0, np.pi / 2])
    vanilla_errors = []

    for step in range(50):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = vanilla_ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)
        error = np.linalg.norm(state[:2] - ref[0, :2])
        vanilla_errors.append(error)

    vanilla_rmse = np.sqrt(np.mean(np.array(vanilla_errors) ** 2))

    # DIAL-MPPI (같은 시드)
    np.random.seed(42)
    dial_ctrl = _make_dial_controller(
        K=256, N=15,
        n_diffuse_init=5, n_diffuse=3,
        traj_diffuse_factor=0.5,
    )
    state = np.array([5.0, 0.0, np.pi / 2])
    dial_errors = []

    for step in range(50):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = dial_ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)
        error = np.linalg.norm(state[:2] - ref[0, :2])
        dial_errors.append(error)

    dial_rmse = np.sqrt(np.mean(np.array(dial_errors) ** 2))

    # 둘 다 합리적인 RMSE (< 2m)
    assert vanilla_rmse < 2.0, f"vanilla RMSE too high: {vanilla_rmse:.4f}"
    assert dial_rmse < 2.0, f"DIAL RMSE too high: {dial_rmse:.4f}"


def test_statistics():
    """get_dial_statistics() 반환값 검증"""
    ctrl = _make_dial_controller(n_diffuse_init=5, n_diffuse=2)
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # 호출 전 통계
    stats_empty = ctrl.get_dial_statistics()
    assert stats_empty["mean_cost_improvement"] == 0.0
    assert stats_empty["mean_n_iters"] == 0.0
    assert stats_empty["last_iteration_costs"] == []
    assert stats_empty["dial_stats_history"] == []

    # 2회 호출 후 통계
    ctrl.compute_control(state, ref)
    ctrl.compute_control(state, ref)

    stats = ctrl.get_dial_statistics()
    assert "mean_cost_improvement" in stats
    assert "mean_n_iters" in stats
    assert "last_iteration_costs" in stats
    assert "dial_stats_history" in stats
    assert len(stats["dial_stats_history"]) == 2
    assert stats["mean_n_iters"] == (5 + 2) / 2  # init + runtime


def test_repr():
    """__repr__ 문자열 포함 내용"""
    ctrl = _make_dial_controller(
        n_diffuse_init=10, n_diffuse=3, traj_diffuse_factor=0.5, K=128,
    )
    repr_str = repr(ctrl)

    assert "DIALMPPIController" in repr_str
    assert "DifferentialDriveKinematic" in repr_str
    assert "n_diffuse_init=10" in repr_str
    assert "n_diffuse=3" in repr_str
    assert "traj_factor=0.5" in repr_str
    assert "K=128" in repr_str


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DIAL-MPPI Unit Tests")
    print("=" * 60)

    tests = [
        test_dial_basic,
        test_first_call_init_iterations,
        test_reset_restores_first_call,
        test_cost_improvement,
        test_horizon_noise_profile,
        test_annealing_effect,
        test_reward_normalization_toggle,
        test_dial_vs_vanilla_tracking,
        test_statistics,
        test_repr,
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
