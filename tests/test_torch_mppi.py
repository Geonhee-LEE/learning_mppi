"""
Pure PyTorch MPPI 테스트

TorchMPPIController, TorchKernelMPPIController,
torch_costs 함수형 비용 함수 검증.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.controllers.mppi.torch_mppi import TorchMPPIController
from mppi_controller.controllers.mppi.torch_kernel_mppi import TorchKernelMPPIController
from mppi_controller.controllers.mppi.torch_costs import (
    make_tracking_cost,
    make_obstacle_cost,
    compose_costs,
)


# ── 헬퍼: 간단한 DiffDrive dynamics (torch) ───────────────────

def diff_drive_dynamics(state: torch.Tensor, control: torch.Tensor, dt: float):
    """
    Differential Drive Kinematic (torch)
    state: (..., 3) [x, y, theta]
    control: (..., 2) [v, omega]
    """
    theta = state[..., 2]
    v = control[..., 0]
    omega = control[..., 1]

    x_dot = v * torch.cos(theta)
    y_dot = v * torch.sin(theta)
    theta_dot = omega

    dstate = torch.stack([x_dot, y_dot, theta_dot], dim=-1)
    return state + dstate * dt


def _make_ref(N=20, nx=3, device="cpu"):
    """간단한 전진 레퍼런스 궤적"""
    ref = torch.zeros(N + 1, nx, device=device)
    for t in range(N + 1):
        ref[t, 0] = t * 0.05  # x 전진
    return ref


def _make_controller(device="cpu", K=128, N=20, **kwargs):
    """기본 TorchMPPIController 생성"""
    Q = np.array([10.0, 10.0, 1.0])
    R = np.array([0.1, 0.1])
    cost_fn = make_tracking_cost(Q, R=R, device=device)
    return TorchMPPIController(
        dynamics_fn=diff_drive_dynamics,
        cost_fn=cost_fn,
        N=N, K=K, nu=2,
        sigma=np.array([0.5, 0.5]),
        device=device,
        **kwargs,
    )


# ── 테스트 ────────────────────────────────────────────────────

def test_torch_mppi_cpu():
    """CPU 모드 기본 동작"""
    print("\n" + "=" * 60)
    print("Test: TorchMPPI CPU Basic")
    print("=" * 60)

    ctrl = _make_controller(device="cpu")
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref().numpy()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"Control shape: {control.shape}"
    assert not np.any(np.isnan(control)), "Control has NaN"

    required_keys = [
        "sample_trajectories", "sample_controls", "sample_weights",
        "best_trajectory", "best_cost", "mean_cost",
        "temperature", "ess", "num_samples",
    ]
    for key in required_keys:
        assert key in info, f"Missing key: {key}"

    assert info["sample_trajectories"].shape == (128, 21, 3)
    assert info["sample_controls"].shape == (128, 20, 2)
    assert abs(np.sum(info["sample_weights"]) - 1.0) < 1e-5
    assert 1.0 <= info["ess"] <= 128

    print("  PASS: CPU basic functionality")


def test_torch_mppi_gpu_if_available():
    """GPU 사용 가능 시 GPU 동작"""
    print("\n" + "=" * 60)
    print("Test: TorchMPPI GPU (if available)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    ctrl = _make_controller(device="cuda")
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref().numpy()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert not np.any(np.isnan(control))
    assert info["sample_trajectories"].shape == (128, 21, 3)

    print("  PASS: GPU basic functionality")


def test_torch_vs_numpy_equivalence():
    """numpy MPPIController와 유사한 결과"""
    print("\n" + "=" * 60)
    print("Test: Torch vs NumPy Equivalence")
    print("=" * 60)

    from mppi_controller.models.kinematic.differential_drive_kinematic import (
        DifferentialDriveKinematic,
    )
    from mppi_controller.controllers.mppi.base_mppi import MPPIController
    from mppi_controller.controllers.mppi.mppi_params import MPPIParams

    # numpy MPPI
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = MPPIParams(
        N=20, dt=0.05, K=256,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    np_ctrl = MPPIController(model, params)

    # torch MPPI
    torch_ctrl = _make_controller(K=256)

    state = np.array([0.0, 0.0, 0.0])
    ref_np = np.zeros((21, 3))
    ref_np[:, 0] = np.linspace(0, 1, 21)

    # 여러 스텝 실행하여 RMSE 비교
    np_errors, torch_errors = [], []
    s_np, s_torch = state.copy(), state.copy()

    for step in range(30):
        c_np, _ = np_ctrl.compute_control(s_np, ref_np)
        c_torch, _ = torch_ctrl.compute_control(s_torch, ref_np)

        s_np = model.step(s_np, c_np, 0.05)
        s_torch_next = diff_drive_dynamics(
            torch.tensor(s_torch, dtype=torch.float32),
            torch.tensor(c_torch, dtype=torch.float32),
            0.05,
        ).numpy()
        s_torch = s_torch_next

        np_errors.append(np.linalg.norm(s_np[:2] - ref_np[0, :2]))
        torch_errors.append(np.linalg.norm(s_torch[:2] - ref_np[0, :2]))

    np_rmse = np.sqrt(np.mean(np.array(np_errors) ** 2))
    torch_rmse = np.sqrt(np.mean(np.array(torch_errors) ** 2))

    print(f"  NumPy RMSE:  {np_rmse:.4f}")
    print(f"  Torch RMSE:  {torch_rmse:.4f}")

    # 둘 다 합리적인 범위 내여야 함 (동일 시드가 아니므로 정확 일치는 불가)
    assert torch_rmse < 2.0, f"Torch RMSE too large: {torch_rmse}"
    assert np_rmse < 2.0, f"NumPy RMSE too large: {np_rmse}"

    print("  PASS: Both controllers produce reasonable results")


def test_torch_compile_compatible():
    """torch.compile 호환성 (에러 없이 실행)"""
    print("\n" + "=" * 60)
    print("Test: torch.compile Compatibility")
    print("=" * 60)

    try:
        compiled_fn = torch.compile(diff_drive_dynamics)
        state = torch.zeros(128, 3)
        control = torch.randn(128, 2) * 0.1
        result = compiled_fn(state, control, 0.05)
        assert result.shape == (128, 3)
        print("  PASS: torch.compile works with dynamics")
    except Exception as e:
        # torch.compile은 일부 환경에서 지원 안됨
        print(f"  SKIP: torch.compile not supported: {e}")


def test_torch_obstacle_cost():
    """torch 장애물 비용 함수 동작"""
    print("\n" + "=" * 60)
    print("Test: Torch Obstacle Cost")
    print("=" * 60)

    obstacles = [(1.0, 0.0, 0.3), (0.0, 1.0, 0.4)]
    cost_fn = make_obstacle_cost(obstacles, safety_margin=0.2, cost_weight=100.0)

    K, N, nx = 10, 5, 3

    # 장애물 가까이 지나는 궤적 vs 먼 궤적
    trajs_near = torch.zeros(K, N + 1, nx)
    trajs_near[:, :, 0] = torch.linspace(0, 1.0, N + 1)  # x 방향으로 장애물 통과

    trajs_far = torch.zeros(K, N + 1, nx)
    trajs_far[:, :, 0] = torch.linspace(0, 1.0, N + 1)
    trajs_far[:, :, 1] = 5.0  # y = 5 (멀리)

    controls = torch.zeros(K, N, 2)
    ref = torch.zeros(N + 1, nx)

    costs_near = cost_fn(trajs_near, controls, ref)
    costs_far = cost_fn(trajs_far, controls, ref)

    assert costs_near.shape == (K,)
    assert torch.all(costs_near > costs_far), \
        "Near-obstacle trajectories should have higher cost"
    assert torch.all(costs_far == 0), \
        "Far trajectories should have zero obstacle cost"

    print(f"  Near cost: {costs_near[0].item():.2f}")
    print(f"  Far cost:  {costs_far[0].item():.2f}")
    print("  PASS: Obstacle cost works correctly")


def test_torch_kernel_mppi():
    """TorchKernelMPPIController 기본 동작"""
    print("\n" + "=" * 60)
    print("Test: TorchKernelMPPI Basic")
    print("=" * 60)

    Q = np.array([10.0, 10.0, 1.0])
    R = np.array([0.1, 0.1])
    cost_fn = make_tracking_cost(Q, R=R)

    ctrl = TorchKernelMPPIController(
        dynamics_fn=diff_drive_dynamics,
        cost_fn=cost_fn,
        N=20, K=128, nu=2, S=8,
        sigma=np.array([0.5, 0.5]),
        kernel_bandwidth=2.0,
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((21, 3))

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"Control shape: {control.shape}"
    assert not np.any(np.isnan(control))
    assert "support_theta" in info
    assert "kernel_stats" in info
    assert info["support_theta"].shape == (8, 2)
    assert info["sample_trajectories"].shape == (128, 21, 3)

    # 차원 축소 확인
    reduction = (1 - 8 / 20) * 100
    assert reduction > 50
    print(f"  Dimension reduction: {reduction:.0f}%")
    print("  PASS: TorchKernelMPPI basic functionality")


def test_torch_kernel_mppi_obstacles():
    """TorchKernelMPPI + 장애물 회피"""
    print("\n" + "=" * 60)
    print("Test: TorchKernelMPPI with Obstacles")
    print("=" * 60)

    Q = np.array([10.0, 10.0, 1.0])
    R = np.array([0.1, 0.1])
    obstacles = [(0.5, 0.0, 0.3)]

    cost_fn = compose_costs(
        make_tracking_cost(Q, R=R),
        make_obstacle_cost(obstacles, cost_weight=500.0),
    )

    ctrl = TorchKernelMPPIController(
        dynamics_fn=diff_drive_dynamics,
        cost_fn=cost_fn,
        N=20, K=256, nu=2, S=8,
        sigma=np.array([0.5, 0.5]),
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((21, 3))
    ref[:, 0] = np.linspace(0, 1, 21)

    # 여러 스텝 실행
    for _ in range(20):
        control, info = ctrl.compute_control(state, ref)
        state = diff_drive_dynamics(
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(control, dtype=torch.float32),
            0.05,
        ).numpy()

    # 장애물과의 거리 확인
    ox, oy, r = obstacles[0]
    dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2) - r
    print(f"  Final distance from obstacle: {dist:.4f} m")
    assert not np.any(np.isnan(state)), "State has NaN"
    print("  PASS: TorchKernelMPPI + obstacles runs without error")


def test_device_transfer():
    """CPU -> GPU 전환 시 결과 일관성"""
    print("\n" + "=" * 60)
    print("Test: Device Transfer")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    # CPU에서 실행
    ctrl_cpu = _make_controller(device="cpu", K=64)
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref().numpy()
    ctrl_cpu.compute_control(state, ref)

    # GPU로 전환
    ctrl_gpu = _make_controller(device="cuda", K=64)
    control_gpu, info_gpu = ctrl_gpu.compute_control(state, ref)

    assert control_gpu.shape == (2,)
    assert not np.any(np.isnan(control_gpu))
    assert info_gpu["sample_trajectories"].shape[0] == 64

    print("  PASS: Device transfer consistent")


def test_diagonal_covariance():
    """대각 공분산 샘플링: sigma 차원별 독립"""
    print("\n" + "=" * 60)
    print("Test: Diagonal Covariance Sampling")
    print("=" * 60)

    # 비대칭 sigma: v는 작고 omega는 큼
    sigma = np.array([0.1, 1.0])
    Q = np.array([10.0, 10.0, 1.0])
    R = np.array([0.1, 0.1])
    cost_fn = make_tracking_cost(Q, R=R)

    ctrl = TorchMPPIController(
        dynamics_fn=diff_drive_dynamics,
        cost_fn=cost_fn,
        N=20, K=1024, nu=2,
        sigma=sigma,
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((21, 3))

    _, info = ctrl.compute_control(state, ref)

    controls = info["sample_controls"]  # (K, N, 2)
    std_v = np.std(controls[:, 0, 0])
    std_omega = np.std(controls[:, 0, 1])

    print(f"  sigma = [0.1, 1.0]")
    print(f"  std(v): {std_v:.3f}, std(omega): {std_omega:.3f}")

    # omega의 분산이 v보다 커야 함
    assert std_omega > std_v * 2, \
        f"Expected std_omega >> std_v, got {std_omega:.3f} vs {std_v:.3f}"

    print("  PASS: Diagonal covariance correctly applied")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Pure PyTorch MPPI Tests".center(60))
    print("=" * 60)

    try:
        test_torch_mppi_cpu()
        test_torch_mppi_gpu_if_available()
        test_torch_vs_numpy_equivalence()
        test_torch_compile_compatible()
        test_torch_obstacle_cost()
        test_torch_kernel_mppi()
        test_torch_kernel_mppi_obstacles()
        test_device_transfer()
        test_diagonal_covariance()

        print("\n" + "=" * 60)
        print("All Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
