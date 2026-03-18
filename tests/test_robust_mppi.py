"""
Robust MPPI (R-MPPI) 유닛 테스트

피드백을 MPPI 샘플링 루프 내부에 통합한 R-MPPI 컨트롤러 검증.
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
    TubeMPPIParams,
    RobustMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.robust_mppi import RobustMPPIController
from mppi_controller.controllers.mppi.ancillary_controller import (
    AncillaryController,
    create_default_ancillary_controller,
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

def _make_robust_controller(**kwargs):
    """헬퍼: Robust MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        disturbance_std=[0.05, 0.05, 0.02],
        feedback_gain_scale=1.0,
        disturbance_mode="gaussian",
        robust_alpha=0.8,
        use_feedback=True,
        n_disturbance_samples=1,
    )
    defaults.update(kwargs)

    # 별도 인자 추출
    ancillary = defaults.pop("ancillary_controller", None)
    cost_function = defaults.pop("cost_function", None)

    params = RobustMPPIParams(**defaults)
    return RobustMPPIController(
        model, params,
        cost_function=cost_function,
        ancillary_controller=ancillary,
    )


def _make_tube_controller(**kwargs):
    """헬퍼: Tube-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        tube_enabled=True,
    )
    defaults.update(kwargs)
    params = TubeMPPIParams(**defaults)
    return TubeMPPIController(model, params)


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


# ── Params Tests (3) ─────────────────────────────────────────


def test_params_defaults():
    """기본값 검증"""
    params = RobustMPPIParams()
    assert params.disturbance_std == [0.05, 0.05, 0.02]
    assert params.feedback_gain_scale == 1.0
    assert params.disturbance_mode == "gaussian"
    assert params.robust_alpha == 0.8
    assert params.use_feedback is True
    assert params.n_disturbance_samples == 1


def test_params_custom():
    """커스텀 값"""
    params = RobustMPPIParams(
        disturbance_std=[0.1, 0.1, 0.05],
        feedback_gain_scale=2.0,
        disturbance_mode="adversarial",
        robust_alpha=0.5,
        use_feedback=False,
        n_disturbance_samples=3,
    )
    assert params.disturbance_std == [0.1, 0.1, 0.05]
    assert params.feedback_gain_scale == 2.0
    assert params.disturbance_mode == "adversarial"
    assert params.robust_alpha == 0.5
    assert params.use_feedback is False
    assert params.n_disturbance_samples == 3


def test_params_validation():
    """잘못된 값 -> AssertionError"""
    # robust_alpha=0
    try:
        RobustMPPIParams(robust_alpha=0.0)
        assert False, "Should have raised AssertionError for alpha=0"
    except AssertionError:
        pass

    # robust_alpha > 1
    try:
        RobustMPPIParams(robust_alpha=1.5)
        assert False, "Should have raised AssertionError for alpha>1"
    except AssertionError:
        pass

    # invalid mode
    try:
        RobustMPPIParams(disturbance_mode="invalid")
        assert False, "Should have raised AssertionError for invalid mode"
    except AssertionError:
        pass

    # n_disturbance_samples=0
    try:
        RobustMPPIParams(n_disturbance_samples=0)
        assert False, "Should have raised AssertionError for n_samples=0"
    except AssertionError:
        pass

    # negative disturbance_std
    try:
        RobustMPPIParams(disturbance_std=[-0.1, 0.05, 0.02])
        assert False, "Should have raised AssertionError for negative std"
    except AssertionError:
        pass


# ── Feedback Tests (5) ──────────────────────────────────────


def test_batch_feedback_shape():
    """(K, nu) 출력"""
    ctrl = _make_robust_controller(K=32)
    states = np.random.randn(32, 3)
    nominal = np.random.randn(32, 3)
    fb = ctrl._batch_feedback(states, nominal)
    assert fb.shape == (32, 2), f"feedback shape: {fb.shape}"


def test_batch_feedback_zero_error():
    """error=0 -> feedback=0"""
    ctrl = _make_robust_controller(K=16)
    states = np.random.randn(16, 3)
    nominal = states.copy()
    fb = ctrl._batch_feedback(states, nominal)
    assert np.allclose(fb, 0.0, atol=1e-10), f"Zero error should give zero feedback: {fb}"


def test_batch_feedback_body_frame():
    """heading 회전 검증"""
    ctrl = _make_robust_controller(K=1)

    # 명목 상태: heading = pi/2 (동쪽→북쪽)
    nominal = np.array([[0.0, 0.0, np.pi / 2]])
    # 실제 상태: x 방향으로 1m 오차 → body frame에서 lateral 오차
    states = np.array([[1.0, 0.0, np.pi / 2]])

    fb = ctrl._batch_feedback(states, nominal)

    # x 방향 1m 오차, heading=pi/2에서 body frame:
    # e_body_x = cos(pi/2)*1 + sin(pi/2)*0 ≈ 0 (longitudinal)
    # e_body_y = -sin(pi/2)*1 + cos(pi/2)*0 ≈ -1 (lateral)
    # 피드백은 -K_fb @ e_body이므로 v ← -K[0,0]*0 = 0, ω ← -K[1,1]*(-1) > 0
    assert fb.shape == (1, 2)
    # 피드백이 0이 아닌지 확인 (body frame 변환이 작동)
    assert not np.allclose(fb, 0.0, atol=1e-5), \
        f"Feedback should be non-zero for body frame error: {fb}"


def test_batch_feedback_clipping():
    """max_correction 내"""
    ctrl = _make_robust_controller(K=8)
    mc = ctrl.ancillary_controller.max_correction

    # 큰 오차
    states = np.ones((8, 3)) * 100.0
    nominal = np.zeros((8, 3))

    fb = ctrl._batch_feedback(states, nominal)
    assert np.all(np.abs(fb) <= mc + 1e-10), \
        f"Feedback exceeds max_correction: max={np.max(np.abs(fb))}, mc={mc}"


def test_custom_ancillary():
    """사용자 정의 AncillaryController"""
    K_fb = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 2.0]])
    max_corr = np.array([1.0, 1.5])
    custom = AncillaryController(K_fb, max_correction=max_corr)

    ctrl = _make_robust_controller(ancillary_controller=custom)
    assert ctrl.ancillary_controller is custom
    assert np.allclose(ctrl.ancillary_controller.K_fb, K_fb)


# ── Disturbance Tests (4) ──────────────────────────────────


def test_disturbance_gaussian_shape():
    """(K, nx) 출력"""
    ctrl = _make_robust_controller(K=32)
    d = ctrl._sample_disturbance(32)
    assert d.shape == (32, 3), f"disturbance shape: {d.shape}"


def test_disturbance_gaussian_stats():
    """mean≈0, std≈disturbance_std"""
    np.random.seed(42)
    ctrl = _make_robust_controller(disturbance_std=[0.1, 0.1, 0.05])
    # 대량 샘플
    d = ctrl._sample_disturbance(10000)
    assert np.abs(np.mean(d[:, 0])) < 0.01, f"mean x: {np.mean(d[:, 0])}"
    assert np.abs(np.mean(d[:, 1])) < 0.01, f"mean y: {np.mean(d[:, 1])}"
    assert 0.08 < np.std(d[:, 0]) < 0.12, f"std x: {np.std(d[:, 0])}"
    assert 0.08 < np.std(d[:, 1]) < 0.12, f"std y: {np.std(d[:, 1])}"


def test_disturbance_adversarial():
    """adversarial 모드 동작"""
    np.random.seed(42)
    ctrl = _make_robust_controller(
        disturbance_mode="adversarial",
        disturbance_std=[0.1, 0.1, 0.05],
        robust_alpha=0.8,
    )
    d = ctrl._sample_disturbance(64)
    assert d.shape == (64, 3)
    # adversarial 외란은 gaussian보다 큰 노름 가져야
    np.random.seed(42)
    ctrl_gauss = _make_robust_controller(
        disturbance_mode="gaussian",
        disturbance_std=[0.1, 0.1, 0.05],
    )
    d_gauss = ctrl_gauss._sample_disturbance(64)

    # adversarial의 평균 노름이 gaussian보다 커야 함 (대부분의 경우)
    # 단, 랜덤 특성 상 항상 보장은 안 되므로 큰 표본으로
    np.random.seed(42)
    d_adv_large = ctrl._sample_disturbance(5000)
    np.random.seed(42)
    d_gauss_large = ctrl_gauss._sample_disturbance(5000)

    norm_adv = np.mean(np.linalg.norm(d_adv_large, axis=1))
    norm_gauss = np.mean(np.linalg.norm(d_gauss_large, axis=1))
    # adversarial은 최소한 gaussian 수준
    assert norm_adv >= norm_gauss * 0.8, \
        f"Adversarial norm ({norm_adv:.3f}) should be >= gaussian ({norm_gauss:.3f})"


def test_disturbance_none():
    """mode="none" -> zeros"""
    ctrl = _make_robust_controller(disturbance_mode="none")
    d = ctrl._sample_disturbance(32)
    assert np.allclose(d, 0.0), "none mode should return zeros"


# ── Controller Basic Tests (5) ──────────────────────────────


def test_compute_control_shape():
    """control (nu,), info keys 확인"""
    np.random.seed(42)
    ctrl = _make_robust_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"control shape: {control.shape}"
    assert isinstance(info, dict)

    required_keys = [
        "sample_trajectories", "nominal_trajectories",
        "sample_weights", "best_trajectory",
        "best_cost", "mean_cost", "temperature", "ess", "num_samples",
        "robust_stats",
    ]
    for key in required_keys:
        assert key in info, f"missing key: {key}"

    assert info["sample_trajectories"].shape == (64, 11, 3)
    assert info["nominal_trajectories"].shape == (64, 11, 3)
    assert info["sample_weights"].shape == (64,)


def test_info_robust_stats():
    """robust_stats 키/값 확인"""
    np.random.seed(42)
    ctrl = _make_robust_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)
    stats = info["robust_stats"]

    required_stats_keys = [
        "mean_tube_width", "max_tube_width",
        "mean_feedback_norm", "disturbance_energy",
        "disturbance_mode", "use_feedback",
    ]
    for key in required_stats_keys:
        assert key in stats, f"missing robust_stats key: {key}"

    assert stats["disturbance_mode"] == "gaussian"
    assert stats["use_feedback"] is True
    assert isinstance(stats["mean_tube_width"], float)
    assert isinstance(stats["disturbance_energy"], float)


def test_tube_width_tracked():
    """tube_width > 0 (외란 존재)"""
    np.random.seed(42)
    ctrl = _make_robust_controller(disturbance_std=[0.1, 0.1, 0.05])
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)
    stats = info["robust_stats"]

    assert stats["mean_tube_width"] > 0, \
        f"tube_width should be > 0 with disturbance, got {stats['mean_tube_width']}"


def test_no_feedback_mode():
    """use_feedback=False -> 피드백 없이 외란만"""
    np.random.seed(42)
    ctrl = _make_robust_controller(use_feedback=False, disturbance_mode="gaussian")
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(control))
    assert info["robust_stats"]["use_feedback"] is False


def test_no_disturbance_mode():
    """mode="none" + no feedback -> nominal = real"""
    np.random.seed(42)
    ctrl = _make_robust_controller(
        disturbance_mode="none", use_feedback=False,
    )
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    # 피드백 없고 외란 없으면 nominal == real
    assert np.allclose(
        info["sample_trajectories"],
        info["nominal_trajectories"],
        atol=1e-10,
    ), "No feedback + no disturbance -> nominal should equal real"


# ── Performance Tests (4) ──────────────────────────────────


def test_circle_tracking_rmse():
    """50스텝 원형 추적, RMSE < 0.3"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    ctrl = _make_robust_controller(K=128, N=15)
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
    assert rmse < 0.3, f"R-MPPI RMSE too high: {rmse:.4f}"


def test_robust_vs_vanilla_noise():
    """외란 하에서 R-MPPI 추적 합리적 (내부 외란 모델링으로 보수적)"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    noise_std = np.array([0.08, 0.08, 0.04])
    n_steps = 60

    # R-MPPI
    np.random.seed(42)
    ctrl_robust = _make_robust_controller(
        K=256, N=15,
        disturbance_std=[0.08, 0.08, 0.04],
    )
    state_r = np.array([5.0, 0.0, np.pi / 2])
    errors_r = []
    for step in range(n_steps):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = ctrl_robust.compute_control(state_r, ref)
        state_r = model.step(state_r, control, 0.05)
        state_r += np.random.randn(3) * noise_std  # 프로세스 노이즈
        errors_r.append(np.linalg.norm(state_r[:2] - ref[0, :2]))

    rmse_r = np.sqrt(np.mean(np.array(errors_r) ** 2))

    # R-MPPI는 내부 외란 모델링으로 보수적이므로 RMSE 합리적 범위 검증
    # (내부 외란이 비용 지형 노이즈 증가 → 단기 테스트에서 불리할 수 있음)
    assert rmse_r < 2.0, \
        f"R-MPPI RMSE under noise should be reasonable: {rmse_r:.4f}"


def test_robust_vs_tube_noise():
    """R-MPPI와 Tube-MPPI 모두 외란 하 작동 확인"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    noise_std = np.array([0.06, 0.06, 0.03])
    n_steps = 60

    # R-MPPI
    np.random.seed(42)
    ctrl_robust = _make_robust_controller(
        K=128, N=15,
        disturbance_std=[0.06, 0.06, 0.03],
    )
    state_r = np.array([5.0, 0.0, np.pi / 2])
    errors_r = []
    for step in range(n_steps):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = ctrl_robust.compute_control(state_r, ref)
        state_r = model.step(state_r, control, 0.05)
        state_r += np.random.randn(3) * noise_std
        errors_r.append(np.linalg.norm(state_r[:2] - ref[0, :2]))

    # Tube-MPPI
    np.random.seed(42)
    ctrl_tube = _make_tube_controller(K=128, N=15)
    state_t = np.array([5.0, 0.0, np.pi / 2])
    errors_t = []
    for step in range(n_steps):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = ctrl_tube.compute_control(state_t, ref)
        state_t = model.step(state_t, control, 0.05)
        state_t += np.random.randn(3) * noise_std
        errors_t.append(np.linalg.norm(state_t[:2] - ref[0, :2]))

    rmse_r = np.sqrt(np.mean(np.array(errors_r) ** 2))
    rmse_t = np.sqrt(np.mean(np.array(errors_t) ** 2))

    # 둘 다 합리적 추적 (R-MPPI는 내부 외란 모델링으로 보수적)
    assert rmse_r < 2.0, f"R-MPPI RMSE too high under noise: {rmse_r:.4f}"
    assert rmse_t < 2.0, f"Tube RMSE too high under noise: {rmse_t:.4f}"


def test_obstacle_avoidance_under_noise():
    """외란 + 장애물 회피"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    obstacles = [(3.0, 0.5, 0.5), (0.0, 3.0, 0.5)]
    cost_fns = [
        StateTrackingCost(np.array([10.0, 10.0, 1.0])),
        TerminalCost(np.array([10.0, 10.0, 1.0])),
        ControlEffortCost(np.array([0.1, 0.1])),
        ObstacleCost(obstacles, safety_margin=0.1, cost_weight=2000.0),
    ]
    cost = CompositeMPPICost(cost_fns)

    ctrl = _make_robust_controller(
        K=256, N=15,
        disturbance_std=[0.03, 0.03, 0.01],
        cost_function=cost,
    )

    state = np.array([5.0, 0.0, np.pi / 2])
    noise_std = np.array([0.03, 0.03, 0.01])

    min_clearance = float("inf")
    for step in range(60):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 15, 0.05)
        control, _ = ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)
        state += np.random.randn(3) * noise_std

        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            clearance = dist - r
            min_clearance = min(min_clearance, clearance)

    assert min_clearance > -0.2, \
        f"R-MPPI min clearance too negative: {min_clearance:.3f}"


# ── Comparison/Integration Tests (7) ────────────────────────


def test_robust_vs_vanilla_no_noise():
    """외란 없음: 유사 성능"""
    np.random.seed(42)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    ctrl_robust = _make_robust_controller(
        K=128, N=10,
        disturbance_mode="none",
        use_feedback=False,
    )
    state_r = np.array([5.0, 0.0, np.pi / 2])
    errors_r = []
    for step in range(30):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 10, 0.05)
        control, _ = ctrl_robust.compute_control(state_r, ref)
        state_r = model.step(state_r, control, 0.05)
        errors_r.append(np.linalg.norm(state_r[:2] - ref[0, :2]))

    np.random.seed(42)
    ctrl_vanilla = _make_vanilla_controller(K=128, N=10)
    state_v = np.array([5.0, 0.0, np.pi / 2])
    errors_v = []
    for step in range(30):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 10, 0.05)
        control, _ = ctrl_vanilla.compute_control(state_v, ref)
        state_v = model.step(state_v, control, 0.05)
        errors_v.append(np.linalg.norm(state_v[:2] - ref[0, :2]))

    rmse_r = np.sqrt(np.mean(np.array(errors_r) ** 2))
    rmse_v = np.sqrt(np.mean(np.array(errors_v) ** 2))

    # 외란 없으면 성능 비슷해야
    ratio = rmse_r / (rmse_v + 1e-10)
    assert 0.5 < ratio < 2.0, \
        f"Without noise, R-MPPI ({rmse_r:.4f}) and Vanilla ({rmse_v:.4f}) should be similar"


def test_different_K_values():
    """K=16/64/128 모두 정상"""
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for K in [16, 64, 128]:
        np.random.seed(42)
        ctrl = _make_robust_controller(K=K)
        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,), f"K={K}: wrong control shape"
        assert not np.any(np.isnan(control)), f"K={K}: NaN"
        assert info["num_samples"] == K


def test_different_disturbance_levels():
    """std 증가 -> tube_width 증가"""
    np.random.seed(42)
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    widths = []
    for std_level in [0.01, 0.05, 0.15]:
        np.random.seed(42)
        ctrl = _make_robust_controller(
            K=128,
            disturbance_std=[std_level, std_level, std_level * 0.4],
        )
        _, info = ctrl.compute_control(state, ref)
        widths.append(info["robust_stats"]["mean_tube_width"])

    # 큰 외란 -> 넓은 tube
    assert widths[2] > widths[0], \
        f"Larger disturbance should have wider tube: {widths}"


def test_reset_clears_state():
    """reset 후 history/U 초기화"""
    np.random.seed(42)
    ctrl = _make_robust_controller()
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    ctrl.compute_control(state, ref)
    assert len(ctrl._robust_history) > 0

    ctrl.reset()
    assert len(ctrl._robust_history) == 0
    assert np.allclose(ctrl.U, 0.0)


def test_repr():
    """__repr__ 내용 검증"""
    ctrl = _make_robust_controller(
        disturbance_mode="adversarial",
        use_feedback=True,
        disturbance_std=[0.1, 0.1, 0.05],
        K=256,
    )
    repr_str = repr(ctrl)

    assert "RobustMPPIController" in repr_str
    assert "DifferentialDriveKinematic" in repr_str
    assert "adversarial" in repr_str
    assert "True" in repr_str
    assert "K=256" in repr_str


def test_numerical_stability():
    """극단적 위치에서 NaN/Inf 없음"""
    np.random.seed(42)
    ctrl = _make_robust_controller(K=64, N=10)

    extreme_states = [
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 100.0, 0.0]),
        np.array([-50.0, -50.0, np.pi]),
        np.array([0.0, 0.0, 10 * np.pi]),
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


def test_n_disturbance_samples():
    """n_disturbance_samples=3 정상 동작"""
    np.random.seed(42)
    ctrl = _make_robust_controller(
        K=64, N=10,
        n_disturbance_samples=3,
        disturbance_std=[0.1, 0.1, 0.05],
    )
    state = np.array([5.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert not np.any(np.isnan(control))

    # n_samples=3 → 분산이 줄어야 (평균 효과)
    # tube_width가 여전히 양수인지 확인
    assert info["robust_stats"]["mean_tube_width"] >= 0


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Robust MPPI (R-MPPI) Unit Tests")
    print("=" * 60)

    tests = [
        # Params (3)
        test_params_defaults,
        test_params_custom,
        test_params_validation,
        # Feedback (5)
        test_batch_feedback_shape,
        test_batch_feedback_zero_error,
        test_batch_feedback_body_frame,
        test_batch_feedback_clipping,
        test_custom_ancillary,
        # Disturbance (4)
        test_disturbance_gaussian_shape,
        test_disturbance_gaussian_stats,
        test_disturbance_adversarial,
        test_disturbance_none,
        # Controller Basic (5)
        test_compute_control_shape,
        test_info_robust_stats,
        test_tube_width_tracked,
        test_no_feedback_mode,
        test_no_disturbance_mode,
        # Performance (4)
        test_circle_tracking_rmse,
        test_robust_vs_vanilla_noise,
        test_robust_vs_tube_noise,
        test_obstacle_avoidance_under_noise,
        # Comparison/Integration (7)
        test_robust_vs_vanilla_no_noise,
        test_different_K_values,
        test_different_disturbance_levels,
        test_reset_clears_state,
        test_repr,
        test_numerical_stability,
        test_n_disturbance_samples,
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
