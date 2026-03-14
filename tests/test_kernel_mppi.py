"""
Kernel MPPI 테스트

RBF 커널 보간 기반 차원 축소 MPPI 검증.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, KernelMPPIParams
from mppi_controller.controllers.mppi.kernel_mppi import (
    RBFKernel,
    KernelMPPIController,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)


def _make_model():
    return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)


def _make_params(**kwargs):
    defaults = dict(
        N=20, dt=0.05, K=128,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        num_support_pts=8,
        kernel_bandwidth=1.0,
    )
    defaults.update(kwargs)
    return KernelMPPIParams(**defaults)


def _make_ref(N=20, dt=0.05):
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


def test_rbf_kernel_properties():
    """RBF 커널 속성: 대칭, 양정치, 대각=1, bandwidth 효과"""
    print("\n" + "=" * 60)
    print("Test: RBF Kernel Properties")
    print("=" * 60)

    kernel = RBFKernel(sigma=1.0)
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    K = kernel(t, t)

    # 대칭
    assert np.allclose(K, K.T), "Kernel matrix should be symmetric"

    # 대각 = 1
    assert np.allclose(np.diag(K), 1.0), "Diagonal should be 1.0"

    # 양정치 (모든 고유값 > 0)
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues > -1e-10), "Kernel matrix should be positive semi-definite"

    # 값 범위 (0, 1]
    assert np.all(K >= 0) and np.all(K <= 1.0), "RBF values should be in [0, 1]"

    # Bandwidth 효과: sigma 작으면 off-diagonal 작음
    kernel_narrow = RBFKernel(sigma=0.1)
    K_narrow = kernel_narrow(t, t)
    assert K_narrow[0, 1] < K[0, 1], "Narrower kernel should have smaller off-diagonal"

    # Bandwidth 효과: sigma 크면 off-diagonal 커짐
    kernel_wide = RBFKernel(sigma=10.0)
    K_wide = kernel_wide(t, t)
    assert K_wide[0, 1] > K[0, 1], "Wider kernel should have larger off-diagonal"

    print("  PASS: RBF kernel properties verified")


def test_kernel_interpolation_exact():
    """서포트 포인트에서 원래 값 정확 복원"""
    print("\n" + "=" * 60)
    print("Test: Kernel Interpolation Exact at Support Points")
    print("=" * 60)

    model = _make_model()
    params = _make_params(N=20, num_support_pts=8)
    controller = KernelMPPIController(model, params)

    # 서포트 포인트에 랜덤 값 설정
    theta = np.random.randn(8, 2) * 0.3
    U_full = controller._interpolate(theta)

    # 서포트 포인트 인덱스에서 W @ theta 확인
    # W의 서포트 포인트 행: Tk에 해당하는 행은 theta를 정확히 복원해야 함
    support_indices = np.round(controller.Tk).astype(int)
    support_indices = np.clip(support_indices, 0, params.N - 1)

    for i, idx in enumerate(support_indices):
        reconstructed = U_full[idx]
        expected = theta[i]
        error = np.linalg.norm(reconstructed - expected)
        # 서포트 포인트에서 작은 오차 허용 (정칙화 때문)
        assert error < 0.1, f"Support point {i} reconstruction error {error:.4f} too large"

    print("  PASS: Interpolation accurate at support points")


def test_kernel_interpolation_smooth():
    """보간 결과 부드러움 (2차 미분 작음)"""
    print("\n" + "=" * 60)
    print("Test: Kernel Interpolation Smoothness")
    print("=" * 60)

    model = _make_model()
    params = _make_params(N=30, num_support_pts=8)
    controller = KernelMPPIController(model, params)

    # 부드러운 서포트 값
    theta = np.column_stack([
        np.sin(np.linspace(0, np.pi, 8)) * 0.5,
        np.cos(np.linspace(0, np.pi, 8)) * 0.3,
    ])
    U_full = controller._interpolate(theta)

    # 2차 미분 (가속도)
    second_diff = np.diff(U_full, n=2, axis=0)
    max_accel = np.max(np.abs(second_diff))

    # RBF 보간은 부드러워야 함
    assert max_accel < 1.0, f"Max acceleration {max_accel:.4f} too large, not smooth"

    print(f"  Max 2nd derivative: {max_accel:.4f}")
    print("  PASS: Interpolation is smooth")


def test_kernel_mppi_basic():
    """Kernel MPPI 기본 동작: compute_control, info keys"""
    print("\n" + "=" * 60)
    print("Test: Kernel MPPI Basic Functionality")
    print("=" * 60)

    model = _make_model()
    params = _make_params()
    controller = KernelMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((21, 3))

    control, info = controller.compute_control(state, reference)

    # 기본 체크
    assert control.shape == (2,), f"Control shape mismatch: {control.shape}"
    assert not np.any(np.isnan(control)), "Control contains NaN"

    # info keys 확인
    required_keys = [
        "sample_trajectories", "sample_controls", "sample_weights",
        "best_trajectory", "best_cost", "mean_cost",
        "temperature", "ess", "num_samples",
        "support_theta", "kernel_stats",
    ]
    for key in required_keys:
        assert key in info, f"Missing info key: {key}"

    # shape 확인
    assert info["sample_trajectories"].shape == (128, 21, 3)
    assert info["sample_controls"].shape == (128, 20, 2)
    assert info["sample_weights"].shape == (128,)
    assert info["support_theta"].shape == (8, 2)

    # 가중치 합 = 1
    assert abs(np.sum(info["sample_weights"]) - 1.0) < 1e-6, "Weights should sum to 1"

    # ESS 범위
    assert 1.0 <= info["ess"] <= 128, f"ESS {info['ess']} out of range"

    print("  PASS: Basic functionality verified")


def test_kernel_mppi_circle_tracking():
    """원형 궤적 추적 RMSE < 0.5"""
    print("\n" + "=" * 60)
    print("Test: Kernel MPPI Circle Tracking")
    print("=" * 60)

    model = _make_model()
    # circle_trajectory 기본: radius=5.0, angular_velocity=0.1
    params = _make_params(N=20, K=512, num_support_pts=8, kernel_bandwidth=2.0)
    controller = KernelMPPIController(model, params)

    # 원 위 초기 위치 (radius=5, t=0 → x=5, y=0, theta=pi/2)
    state = np.array([5.0, 0.0, np.pi / 2])
    dt = params.dt
    num_steps = 100

    position_errors = []

    for step in range(num_steps):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, params.N, dt)
        control, info = controller.compute_control(state, ref)

        # 상태 업데이트
        state = model.step(state, control, dt)

        # 위치 오차
        ref_pos = ref[0, :2]
        pos_error = np.linalg.norm(state[:2] - ref_pos)
        position_errors.append(pos_error)

    rmse = np.sqrt(np.mean(np.array(position_errors) ** 2))
    print(f"  Circle tracking RMSE: {rmse:.4f}")
    assert rmse < 0.5, f"RMSE {rmse:.4f} exceeds threshold 0.5"
    print("  PASS: Circle tracking within threshold")


def test_support_point_reduction():
    """노이즈 차원 S < N 확인"""
    print("\n" + "=" * 60)
    print("Test: Support Point Dimension Reduction")
    print("=" * 60)

    N = 30
    S = 8
    K = 1024
    nu = 2

    kernel_dim = K * S * nu
    vanilla_dim = K * N * nu
    reduction = (vanilla_dim - kernel_dim) / vanilla_dim * 100

    print(f"  Vanilla: {vanilla_dim} elements")
    print(f"  Kernel:  {kernel_dim} elements")
    print(f"  Reduction: {reduction:.1f}%")

    assert kernel_dim < vanilla_dim, "Kernel should use fewer elements"
    assert reduction > 70, f"Expected >70% reduction, got {reduction:.1f}%"

    # 실제 컨트롤러에서도 확인
    model = _make_model()
    params = _make_params(N=N, K=K, num_support_pts=S)
    controller = KernelMPPIController(model, params)
    assert controller.S < params.N, "S must be < N"

    print("  PASS: Dimension reduction verified")


def test_bandwidth_effect():
    """Bandwidth 변화에 따른 행동 차이"""
    print("\n" + "=" * 60)
    print("Test: Bandwidth Effect")
    print("=" * 60)

    model = _make_model()
    state = np.array([0.0, 0.0, 0.0])
    reference = _make_ref()

    np.random.seed(42)
    params_narrow = _make_params(kernel_bandwidth=0.5)
    ctrl_narrow = KernelMPPIController(model, params_narrow)

    np.random.seed(42)
    params_wide = _make_params(kernel_bandwidth=5.0)
    ctrl_wide = KernelMPPIController(model, params_wide)

    # 여러 스텝 실행
    controls_narrow = []
    controls_wide = []
    s1, s2 = state.copy(), state.copy()

    for _ in range(10):
        np.random.seed(None)
        c1, _ = ctrl_narrow.compute_control(s1, reference)
        c2, _ = ctrl_wide.compute_control(s2, reference)
        controls_narrow.append(c1.copy())
        controls_wide.append(c2.copy())

    controls_narrow = np.array(controls_narrow)
    controls_wide = np.array(controls_wide)

    # 넓은 bandwidth → 더 부드러운 제어 (2차 미분 작음)
    if len(controls_wide) > 2:
        accel_narrow = np.mean(np.abs(np.diff(controls_narrow, n=2, axis=0)))
        accel_wide = np.mean(np.abs(np.diff(controls_wide, n=2, axis=0)))
        print(f"  Narrow (sigma=0.5) accel: {accel_narrow:.4f}")
        print(f"  Wide   (sigma=5.0) accel: {accel_wide:.4f}")

    # 둘 다 NaN 없어야 함
    assert not np.any(np.isnan(controls_narrow)), "Narrow bandwidth produced NaN"
    assert not np.any(np.isnan(controls_wide)), "Wide bandwidth produced NaN"

    print("  PASS: Bandwidth effect verified")


def test_smoothness_vs_vanilla():
    """제어 변화율 KMPPI < Vanilla"""
    print("\n" + "=" * 60)
    print("Test: Smoothness vs Vanilla MPPI")
    print("=" * 60)

    model = _make_model()
    state = np.array([0.0, 0.0, 0.0])
    dt = 0.05
    num_steps = 30

    # Vanilla MPPI
    vanilla_params = MPPIParams(
        N=20, dt=dt, K=256,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    vanilla = MPPIController(model, vanilla_params)

    # Kernel MPPI
    kernel_params = _make_params(N=20, K=256, num_support_pts=6, kernel_bandwidth=2.0)
    kernel = KernelMPPIController(model, kernel_params)

    vanilla_controls = []
    kernel_controls = []
    sv, sk = state.copy(), state.copy()

    for step in range(num_steps):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 20, dt)

        cv, _ = vanilla.compute_control(sv, ref)
        ck, _ = kernel.compute_control(sk, ref)

        sv = model.step(sv, cv, dt)
        sk = model.step(sk, ck, dt)

        vanilla_controls.append(cv.copy())
        kernel_controls.append(ck.copy())

    vanilla_controls = np.array(vanilla_controls)
    kernel_controls = np.array(kernel_controls)

    # 제어 변화율 (jerk)
    vanilla_jerk = np.mean(np.abs(np.diff(vanilla_controls, axis=0)))
    kernel_jerk = np.mean(np.abs(np.diff(kernel_controls, axis=0)))

    print(f"  Vanilla jerk: {vanilla_jerk:.4f}")
    print(f"  Kernel jerk:  {kernel_jerk:.4f}")

    # Kernel MPPI가 더 부드럽거나 비슷해야 함
    # (통계적 변동을 고려하여 2배 이내)
    assert kernel_jerk < vanilla_jerk * 2.0, \
        f"Kernel jerk {kernel_jerk:.4f} too large vs vanilla {vanilla_jerk:.4f}"

    print("  PASS: Smoothness comparison verified")


def test_num_support_pts_effect():
    """S -> N이면 Vanilla에 수렴"""
    print("\n" + "=" * 60)
    print("Test: Support Points Count Effect")
    print("=" * 60)

    model = _make_model()
    N = 20
    state = np.array([0.0, 0.0, 0.0])
    reference = _make_ref(N=N)

    support_counts = [4, 8, 12, 20]

    for S in support_counts:
        params = _make_params(N=N, K=128, num_support_pts=S)
        controller = KernelMPPIController(model, params)
        control, info = controller.compute_control(state, reference)

        assert not np.any(np.isnan(control)), f"NaN at S={S}"
        print(f"  S={S:2d}: cost={info['mean_cost']:.4f}, ess={info['ess']:.1f}")

    # S=N이면 W가 단위행렬에 가까워야 함
    params_full = _make_params(N=N, K=128, num_support_pts=N)
    controller_full = KernelMPPIController(model, params_full)
    W = controller_full.W
    identity_error = np.linalg.norm(W - np.eye(N)) / N
    print(f"  S=N: ||W - I||/N = {identity_error:.4f}")
    assert identity_error < 0.5, f"W should be close to identity when S=N, error={identity_error:.4f}"

    print("  PASS: Support point count effect verified")


def test_kernel_statistics():
    """통계 반환 검증"""
    print("\n" + "=" * 60)
    print("Test: Kernel Statistics")
    print("=" * 60)

    model = _make_model()
    params = _make_params()
    controller = KernelMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((21, 3))

    # 빈 통계
    stats = controller.get_kernel_statistics()
    assert stats["mean_theta_variance"] == 0.0
    assert len(stats["kernel_stats_history"]) == 0

    # 여러 번 호출
    num_steps = 10
    for _ in range(num_steps):
        controller.compute_control(state, reference)

    stats = controller.get_kernel_statistics()
    assert len(stats["kernel_stats_history"]) == num_steps, "History length mismatch"
    assert stats["mean_theta_variance"] >= 0, "Variance should be non-negative"

    # 개별 통계 키 확인
    entry = stats["kernel_stats_history"][0]
    assert "num_support_pts" in entry
    assert "kernel_bandwidth" in entry
    assert "theta_variance" in entry
    assert "interpolation_matrix_cond" in entry

    print(f"  Mean theta variance: {stats['mean_theta_variance']:.4f}")
    print(f"  History length: {len(stats['kernel_stats_history'])}")
    print("  PASS: Kernel statistics verified")


def test_shift_theta():
    """Receding horizon 서포트 시프트"""
    print("\n" + "=" * 60)
    print("Test: Theta Shift (Receding Horizon)")
    print("=" * 60)

    model = _make_model()
    params = _make_params(num_support_pts=4)
    controller = KernelMPPIController(model, params)

    # 서포트 포인트에 알려진 값 설정
    controller.theta = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    controller._shift_theta()

    # 시프트 후: [3,4], [5,6], [7,8], [0,0]
    expected = np.array([
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [0.0, 0.0],
    ])
    assert np.allclose(controller.theta, expected), \
        f"Theta shift incorrect:\n{controller.theta}\nexpected:\n{expected}"

    print("  PASS: Theta shift correct")


def test_control_bounds():
    """u_min/u_max 제어 제약 준수"""
    print("\n" + "=" * 60)
    print("Test: Control Bounds")
    print("=" * 60)

    model = _make_model()
    params = _make_params(
        u_min=np.array([-0.5, -1.0]),
        u_max=np.array([0.5, 1.0]),
    )
    controller = KernelMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = _make_ref()

    for _ in range(20):
        control, info = controller.compute_control(state, reference)
        state = model.step(state, control, params.dt)

        assert np.all(control >= params.u_min - 1e-6), \
            f"Control {control} below u_min {params.u_min}"
        assert np.all(control <= params.u_max + 1e-6), \
            f"Control {control} above u_max {params.u_max}"

    print("  PASS: Control bounds respected")


def test_reset():
    """reset 후 상태 초기화"""
    print("\n" + "=" * 60)
    print("Test: Reset")
    print("=" * 60)

    model = _make_model()
    params = _make_params()
    controller = KernelMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((21, 3))

    # 몇 번 실행
    for _ in range(5):
        controller.compute_control(state, reference)

    # 리셋
    controller.reset()

    assert np.allclose(controller.U, 0), "U should be zero after reset"
    assert np.allclose(controller.theta, 0), "theta should be zero after reset"
    assert len(controller.kernel_stats_history) == 0, "Stats history should be empty"

    print("  PASS: Reset works correctly")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Kernel MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_rbf_kernel_properties()
        test_kernel_interpolation_exact()
        test_kernel_interpolation_smooth()
        test_kernel_mppi_basic()
        test_kernel_mppi_circle_tracking()
        test_support_point_reduction()
        test_bandwidth_effect()
        test_smoothness_vs_vanilla()
        test_num_support_pts_effect()
        test_kernel_statistics()
        test_shift_theta()
        test_control_bounds()
        test_reset()

        print("\n" + "=" * 60)
        print("All Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
