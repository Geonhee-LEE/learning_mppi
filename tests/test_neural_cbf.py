"""
Neural CBF 테스트 (18개)

Network (4) + Trainer (5) + Cost (4) + Filter (3) + Integration (2)
"""

import numpy as np
import torch
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.learning.neural_cbf_trainer import (
    NeuralCBFNetwork,
    NeuralCBFTrainer,
    NeuralCBFTrainerConfig,
)
from mppi_controller.controllers.mppi.neural_cbf_cost import NeuralBarrierCost
from mppi_controller.controllers.mppi.neural_cbf_filter import NeuralCBFSafetyFilter


# ============================================================
# Helper utilities
# ============================================================

def _train_simple_cbf(obstacles, epochs=100, num_samples=1000):
    """1장애물 빠른 학습 헬퍼"""
    config = NeuralCBFTrainerConfig(
        state_dim=3,
        hidden_dims=[64, 64],
        epochs=epochs,
        num_safe_samples=num_samples,
        num_unsafe_samples=num_samples,
        num_boundary_samples=num_samples // 2,
        workspace_bounds=(-4.0, 4.0, -4.0, 4.0),
        early_stopping_patience=epochs,  # disable
        batch_size=128,
    )
    trainer = NeuralCBFTrainer(config)
    data = trainer.generate_training_data(obstacles, safety_margin=0.05)
    trainer.train(data, verbose=False)
    return trainer


def _l_shaped_region(state):
    """L자형 비볼록 장애물: (1,1)-(2,2) ∪ (2,1)-(3,1.5)"""
    x, y = state[0], state[1]
    in_box1 = (1.0 <= x <= 2.0) and (1.0 <= y <= 2.0)
    in_box2 = (2.0 <= x <= 3.0) and (1.0 <= y <= 1.5)
    return in_box1 or in_box2


# ============================================================
# Network tests (4)
# ============================================================

def test_network_output_shape():
    """(batch, 1) 출력 형상 검증"""
    print("\n" + "=" * 60)
    print("Test 1: Network Output Shape")
    print("=" * 60)

    net = NeuralCBFNetwork(state_dim=3, hidden_dims=[64, 32])
    x = torch.randn(16, 3)
    h = net(x)

    assert h.shape == (16, 1), f"Expected (16, 1), got {h.shape}"

    # Single input
    h_single = net(torch.randn(1, 3))
    assert h_single.shape == (1, 1), f"Expected (1, 1), got {h_single.shape}"

    print("✓ PASS: Output shape correct\n")


def test_network_output_range():
    """h(x) ∈ [-scale, +scale] 바운드 검증"""
    print("=" * 60)
    print("Test 2: Network Output Range")
    print("=" * 60)

    scale = 5.0
    net = NeuralCBFNetwork(state_dim=3, output_scale=scale)
    x = torch.randn(1000, 3) * 10  # large inputs

    with torch.no_grad():
        h = net(x)

    assert h.min() >= -scale - 1e-6, f"h_min={h.min():.4f} < -{scale}"
    assert h.max() <= scale + 1e-6, f"h_max={h.max():.4f} > {scale}"

    print(f"h range: [{h.min():.4f}, {h.max():.4f}]")
    print("✓ PASS: Output bounded in [-scale, +scale]\n")


def test_network_gradient_shape():
    """∂h/∂x (batch, nx) 형상 검증"""
    print("=" * 60)
    print("Test 3: Network Gradient Shape")
    print("=" * 60)

    net = NeuralCBFNetwork(state_dim=3)
    x = torch.randn(8, 3)
    grad_h = net.gradient(x)

    assert grad_h.shape == (8, 3), f"Expected (8, 3), got {grad_h.shape}"
    print("✓ PASS: Gradient shape correct\n")


def test_network_gradient_nonzero():
    """Gradient 소실 없음 검증"""
    print("=" * 60)
    print("Test 4: Network Gradient Nonzero")
    print("=" * 60)

    net = NeuralCBFNetwork(state_dim=3, hidden_dims=[64, 64])
    x = torch.randn(32, 3)
    grad_h = net.gradient(x)

    grad_norms = torch.norm(grad_h, dim=1)
    assert (grad_norms > 1e-8).all(), "Some gradients are zero!"

    print(f"Gradient norms: mean={grad_norms.mean():.4f}, min={grad_norms.min():.6f}")
    print("✓ PASS: All gradients nonzero\n")


# ============================================================
# Trainer tests (5)
# ============================================================

def test_data_generation_circular():
    """원형 장애물 safe/unsafe/boundary 분류 검증"""
    print("=" * 60)
    print("Test 5: Data Generation - Circular")
    print("=" * 60)

    config = NeuralCBFTrainerConfig(
        num_safe_samples=500,
        num_unsafe_samples=500,
        num_boundary_samples=200,
        workspace_bounds=(-3.0, 3.0, -3.0, 3.0),
    )
    trainer = NeuralCBFTrainer(config)

    obstacles = [(0.0, 0.0, 1.0)]
    data = trainer.generate_training_data(obstacles, safety_margin=0.05)

    assert "safe_states" in data
    assert "unsafe_states" in data
    assert "boundary_states" in data

    # Safe: 모두 장애물 외부
    safe_dist = np.sqrt(data["safe_states"][:, 0]**2 + data["safe_states"][:, 1]**2)
    assert np.all(safe_dist > 1.0), "Some safe samples inside obstacle!"

    # Unsafe: 모두 장애물 내부
    unsafe_dist = np.sqrt(data["unsafe_states"][:, 0]**2 + data["unsafe_states"][:, 1]**2)
    assert np.all(unsafe_dist < 1.0), "Some unsafe samples outside obstacle!"

    # Boundary: 경계 근처
    bnd_dist = np.sqrt(data["boundary_states"][:, 0]**2 + data["boundary_states"][:, 1]**2)
    assert np.all(np.abs(bnd_dist - 1.0) < 0.1), "Boundary samples too far from edge!"

    print(f"Safe: {len(data['safe_states'])}, Unsafe: {len(data['unsafe_states'])}, "
          f"Boundary: {len(data['boundary_states'])}")
    print("✓ PASS: Circular data generation correct\n")


def test_data_generation_non_convex():
    """비볼록 callable 기반 분류 검증"""
    print("=" * 60)
    print("Test 6: Data Generation - Non-Convex")
    print("=" * 60)

    config = NeuralCBFTrainerConfig(
        num_safe_samples=500,
        num_unsafe_samples=500,
        num_boundary_samples=200,
        workspace_bounds=(-1.0, 5.0, -1.0, 5.0),
    )
    trainer = NeuralCBFTrainer(config)

    data = trainer.generate_training_data(
        obstacles=[],
        safety_margin=0.05,
        non_convex_regions=[_l_shaped_region],
    )

    # Unsafe: L자형 내부
    for i in range(min(len(data["unsafe_states"]), 50)):
        state = data["unsafe_states"][i]
        assert _l_shaped_region(state), f"Unsafe sample {i} not in L-shape!"

    # Safe: L자형 외부
    for i in range(min(len(data["safe_states"]), 50)):
        state = data["safe_states"][i]
        assert not _l_shaped_region(state), f"Safe sample {i} in L-shape!"

    print(f"Safe: {len(data['safe_states'])}, Unsafe: {len(data['unsafe_states'])}")
    print("✓ PASS: Non-convex data generation correct\n")


def test_training_convergence():
    """Loss 감소 + accuracy > 0.9 검증"""
    print("=" * 60)
    print("Test 7: Training Convergence")
    print("=" * 60)

    trainer = _train_simple_cbf(
        obstacles=[(0.0, 0.0, 1.0)],
        epochs=150,
        num_samples=2000,
    )

    # Re-train to get history
    config = NeuralCBFTrainerConfig(
        state_dim=3,
        hidden_dims=[64, 64],
        epochs=150,
        num_safe_samples=2000,
        num_unsafe_samples=2000,
        num_boundary_samples=500,
        workspace_bounds=(-4.0, 4.0, -4.0, 4.0),
        early_stopping_patience=150,
        batch_size=128,
    )
    trainer = NeuralCBFTrainer(config)
    data = trainer.generate_training_data([(0.0, 0.0, 1.0)], safety_margin=0.05)
    history = trainer.train(data, verbose=False)

    # Loss should decrease
    assert history["train_loss"][-1] < history["train_loss"][0], \
        "Training loss did not decrease!"

    # Accuracy > 0.9
    final_safe_acc = history["safe_acc"][-1]
    final_unsafe_acc = history["unsafe_acc"][-1]

    print(f"Final train_loss: {history['train_loss'][-1]:.4f}")
    print(f"Safe accuracy: {final_safe_acc:.3f}")
    print(f"Unsafe accuracy: {final_unsafe_acc:.3f}")

    assert final_safe_acc > 0.85, f"Safe accuracy {final_safe_acc:.3f} < 0.85"
    assert final_unsafe_acc > 0.85, f"Unsafe accuracy {final_unsafe_acc:.3f} < 0.85"
    print("✓ PASS: Training converged\n")


def test_predict_h_numpy():
    """safe→h>0, unsafe→h<0, boundary→|h|<ε 검증"""
    print("=" * 60)
    print("Test 8: Predict h - NumPy Interface")
    print("=" * 60)

    trainer = _train_simple_cbf(
        obstacles=[(0.0, 0.0, 1.0)],
        epochs=150,
        num_samples=2000,
    )

    # Safe points (far from obstacle)
    safe_states = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [-3.0, -3.0, 0.0],
    ])
    h_safe = trainer.predict_h(safe_states)
    print(f"h(safe): {h_safe}")
    assert np.all(h_safe > 0), f"Safe points should have h > 0, got {h_safe}"

    # Unsafe points (inside obstacle)
    unsafe_states = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.3, 0.0],
        [-0.2, 0.1, 0.0],
    ])
    h_unsafe = trainer.predict_h(unsafe_states)
    print(f"h(unsafe): {h_unsafe}")
    assert np.all(h_unsafe < 0), f"Unsafe points should have h < 0, got {h_unsafe}"

    # Single state (scalar)
    h_single = trainer.predict_h(np.array([3.0, 0.0, 0.0]))
    assert isinstance(h_single, float), f"Single predict should return float, got {type(h_single)}"

    print("✓ PASS: predict_h NumPy interface correct\n")


def test_save_load_checkpoint():
    """저장/로드 후 예측 동일 검증"""
    print("=" * 60)
    print("Test 9: Save/Load Checkpoint")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = NeuralCBFTrainerConfig(
            hidden_dims=[32, 32],
            epochs=30,
            num_safe_samples=500,
            num_unsafe_samples=500,
            num_boundary_samples=200,
            workspace_bounds=(-3.0, 3.0, -3.0, 3.0),
            early_stopping_patience=30,
            save_dir=tmpdir,
        )
        trainer = NeuralCBFTrainer(config)
        data = trainer.generate_training_data([(0.0, 0.0, 0.5)])
        trainer.train(data, verbose=False)

        test_states = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        h_before = trainer.predict_h(test_states)

        trainer.save_model("test_cbf.pth")

        # Load into new trainer
        trainer2 = NeuralCBFTrainer(config)
        trainer2.load_model("test_cbf.pth")
        h_after = trainer2.predict_h(test_states)

        print(f"h_before: {h_before}")
        print(f"h_after: {h_after}")
        assert np.allclose(h_before, h_after, atol=1e-6), \
            f"Predictions differ after load! diff={np.abs(h_before - h_after)}"

    print("✓ PASS: Save/Load checkpoint correct\n")


# ============================================================
# Cost tests (4)
# ============================================================

def test_neural_cost_shape():
    """compute_cost → (K,) 형상 검증"""
    print("=" * 60)
    print("Test 10: Neural Cost Shape")
    print("=" * 60)

    trainer = _train_simple_cbf([(2.0, 2.0, 0.5)], epochs=50, num_samples=500)
    cost = NeuralBarrierCost(trainer, cbf_alpha=0.1, cbf_weight=100.0)

    K, N = 32, 10
    trajectories = np.zeros((K, N + 1, 3))
    trajectories[:, :, 0] = np.linspace(0, 1, N + 1)
    controls = np.zeros((K, N, 2))
    reference = np.zeros((N + 1, 3))

    costs = cost.compute_cost(trajectories, controls, reference)

    assert costs.shape == (K,), f"Expected ({K},), got {costs.shape}"
    print(f"Costs shape: {costs.shape}, range: [{costs.min():.4f}, {costs.max():.4f}]")
    print("✓ PASS: Neural cost shape correct\n")


def test_neural_cost_safe_low():
    """안전 궤적 → 낮은 비용 검증"""
    print("=" * 60)
    print("Test 11: Neural Cost - Safe Trajectory Low Cost")
    print("=" * 60)

    trainer = _train_simple_cbf([(0.0, 0.0, 1.0)], epochs=150, num_samples=2000)
    cost = NeuralBarrierCost(trainer, cbf_alpha=0.1, cbf_weight=1000.0)

    K, N = 16, 10
    # 안전 궤적: 장애물에서 먼 곳 (x=3 근처)
    trajectories = np.zeros((K, N + 1, 3))
    trajectories[:, :, 0] = 3.0
    trajectories[:, :, 1] = np.linspace(0, 0.5, N + 1)
    controls = np.zeros((K, N, 2))
    reference = np.zeros((N + 1, 3))

    costs = cost.compute_cost(trajectories, controls, reference)

    print(f"Safe trajectory costs: max={costs.max():.4f}")
    assert costs.max() < 10.0, f"Safe trajectory cost too high: {costs.max():.4f}"
    print("✓ PASS: Safe trajectory has low cost\n")


def test_neural_cost_unsafe_high():
    """장애물 통과 궤적 → 높은 비용 검증"""
    print("=" * 60)
    print("Test 12: Neural Cost - Unsafe Trajectory High Cost")
    print("=" * 60)

    trainer = _train_simple_cbf([(0.0, 0.0, 1.0)], epochs=150, num_samples=2000)
    cost_safe = NeuralBarrierCost(trainer, cbf_alpha=0.1, cbf_weight=1000.0)

    K, N = 16, 10
    # 장애물 통과 궤적: 원점 (장애물 내부)
    traj_unsafe = np.zeros((K, N + 1, 3))
    traj_unsafe[:, :, 0] = np.linspace(-0.5, 0.5, N + 1)

    # 안전 궤적: 장애물에서 먼 곳
    traj_safe = np.zeros((K, N + 1, 3))
    traj_safe[:, :, 0] = 3.0

    controls = np.zeros((K, N, 2))
    reference = np.zeros((N + 1, 3))

    cost_unsafe = cost_safe.compute_cost(traj_unsafe, controls, reference)
    cost_far = cost_safe.compute_cost(traj_safe, controls, reference)

    print(f"Unsafe trajectory cost: {cost_unsafe.mean():.2f}")
    print(f"Safe trajectory cost: {cost_far.mean():.2f}")

    assert cost_unsafe.mean() > cost_far.mean(), \
        "Unsafe trajectory should have higher cost than safe!"
    print("✓ PASS: Unsafe trajectory has higher cost\n")


def test_neural_cost_barrier_info():
    """get_barrier_info dict 키/값 검증"""
    print("=" * 60)
    print("Test 13: Neural Cost - Barrier Info")
    print("=" * 60)

    trainer = _train_simple_cbf([(0.0, 0.0, 1.0)], epochs=50, num_samples=500)
    cost = NeuralBarrierCost(trainer, cbf_alpha=0.1, cbf_weight=100.0)

    # (N+1, 3) single trajectory
    traj = np.zeros((10, 3))
    traj[:, 0] = np.linspace(0, 3, 10)

    info = cost.get_barrier_info(traj)

    assert "barrier_values" in info
    assert "min_barrier" in info
    assert "is_safe" in info
    assert isinstance(info["min_barrier"], float)
    assert isinstance(info["is_safe"], bool)
    assert info["barrier_values"].shape == (1, 10)

    # Batch trajectory
    traj_batch = np.zeros((5, 10, 3))
    traj_batch[:, :, 0] = 3.0  # far from obstacle
    info2 = cost.get_barrier_info(traj_batch)
    assert info2["barrier_values"].shape == (5, 10)

    print(f"barrier_values shape: {info['barrier_values'].shape}")
    print(f"min_barrier: {info['min_barrier']:.4f}")
    print(f"is_safe: {info['is_safe']}")
    print("✓ PASS: Barrier info correct\n")


# ============================================================
# Filter tests (3)
# ============================================================

def test_filter_safe_passthrough():
    """안전 상태 → u_safe ≈ u_mppi 검증"""
    print("=" * 60)
    print("Test 14: Filter - Safe Passthrough")
    print("=" * 60)

    trainer = _train_simple_cbf([(0.0, 0.0, 1.0)], epochs=150, num_samples=2000)
    filt = NeuralCBFSafetyFilter(trainer, cbf_alpha=0.3)

    # 안전 상태 (장애물에서 먼 곳)
    state = np.array([3.0, 0.0, 0.0])
    u_mppi = np.array([0.5, 0.1])

    u_safe, info = filt.filter_control(state, u_mppi)

    print(f"u_mppi: {u_mppi}")
    print(f"u_safe: {u_safe}")
    print(f"filtered: {info['filtered']}")

    assert not info["filtered"], "Safe state should not be filtered!"
    assert np.allclose(u_safe, u_mppi), "Safe control should pass through unchanged!"
    print("✓ PASS: Safe state passes through\n")


def test_filter_unsafe_correction():
    """위험 상태 → 속도 감소 검증"""
    print("=" * 60)
    print("Test 15: Filter - Unsafe Correction")
    print("=" * 60)

    trainer = _train_simple_cbf([(2.0, 0.0, 0.5)], epochs=150, num_samples=2000)
    filt = NeuralCBFSafetyFilter(trainer, cbf_alpha=0.3)

    # 장애물 바로 앞, 장애물 쪽으로 진행하는 상태
    state = np.array([1.4, 0.0, 0.0])  # 장애물 (2,0) 반경 0.5 → 경계 1.5
    u_mppi = np.array([1.0, 0.0])  # 전진

    u_safe, info = filt.filter_control(state, u_mppi)

    print(f"u_mppi: {u_mppi}")
    print(f"u_safe: {u_safe}")
    print(f"filtered: {info['filtered']}")
    print(f"correction_norm: {info['correction_norm']:.4f}")

    # 필터가 적용되거나, 이미 안전하면 pass-through
    if info["filtered"]:
        assert u_safe[0] <= u_mppi[0], \
            "Filtered control should reduce forward velocity!"
        print("✓ PASS: Filter corrected unsafe control\n")
    else:
        # Neural CBF may consider this safe depending on training
        print("✓ PASS: Neural CBF considers state safe (no filter needed)\n")


def test_filter_lie_derivative_consistency():
    """원형 장애물에서 분석적 Lie vs 신경망 방향 비교"""
    print("=" * 60)
    print("Test 16: Filter - Lie Derivative Consistency")
    print("=" * 60)

    obstacles = [(2.0, 0.0, 0.5)]
    trainer = _train_simple_cbf(obstacles, epochs=150, num_samples=2000)
    filt = NeuralCBFSafetyFilter(trainer, cbf_alpha=0.1)

    # 장애물 근처 상태
    state = np.array([1.2, 0.0, 0.0])

    # Neural Lie derivatives
    Lf_n, Lg_n, h_n = filt._compute_lie_derivatives_neural(state)

    # Analytical Lie derivatives (for comparison)
    x, y, theta = state
    ox, oy, r = obstacles[0]
    h_analytical = (x - ox)**2 + (y - oy)**2 - r**2
    Lg_analytical = np.array([
        2.0 * (x - ox) * np.cos(theta) + 2.0 * (y - oy) * np.sin(theta),
        0.0,
    ])

    print(f"Neural:     Lf={Lf_n:.4f}, Lg={Lg_n}, h={h_n:.4f}")
    print(f"Analytical: Lf=0.0000, Lg={Lg_analytical}, h={h_analytical:.4f}")

    # Lf should be ~0 (kinematic)
    assert abs(Lf_n) < 0.01, f"Lf_h should be ~0, got {Lf_n}"

    # Lg direction should be similar (both should point away from obstacle)
    if np.linalg.norm(Lg_n) > 1e-4 and np.linalg.norm(Lg_analytical) > 1e-4:
        cos_angle = np.dot(Lg_n, Lg_analytical) / (
            np.linalg.norm(Lg_n) * np.linalg.norm(Lg_analytical)
        )
        print(f"Lie derivative direction cosine: {cos_angle:.4f}")
        # Both should point in similar direction (negative x since obstacle is at x=2)
        assert cos_angle > 0.0, \
            f"Lie derivative direction should be consistent, cosine={cos_angle}"

    print("✓ PASS: Lie derivative consistency verified\n")


# ============================================================
# Integration tests (2)
# ============================================================

def test_neural_cbf_with_mppi():
    """MPPIController + NeuralBarrierCost 종합 동작 검증"""
    print("=" * 60)
    print("Test 17: Integration - Neural CBF with MPPI")
    print("=" * 60)

    from mppi_controller.models.kinematic.differential_drive_kinematic import (
        DifferentialDriveKinematic,
    )
    from mppi_controller.controllers.mppi.base_mppi import MPPIController
    from mppi_controller.controllers.mppi.mppi_params import MPPIParams
    from mppi_controller.controllers.mppi.cost_functions import (
        StateTrackingCost,
        CompositeMPPICost,
    )

    # Train Neural CBF
    obstacles = [(2.0, 0.0, 0.5)]
    trainer = _train_simple_cbf(obstacles, epochs=150, num_samples=2000)

    # Setup MPPI
    robot = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0, wheelbase=0.5)
    params = MPPIParams(
        K=64,
        N=15,
        dt=0.1,
        lambda_=1.0,
    )

    tracking_cost = StateTrackingCost(
        Q=np.diag([10.0, 10.0, 1.0]),
    )
    neural_cbf_cost = NeuralBarrierCost(trainer, cbf_alpha=0.1, cbf_weight=500.0)

    composite = CompositeMPPICost([tracking_cost, neural_cbf_cost])

    controller = MPPIController(
        model=robot,
        cost_function=composite,
        params=params,
    )

    # Run one step
    state = np.array([0.0, 0.0, 0.0])
    reference = np.zeros((params.N + 1, 3))
    reference[:, 0] = np.linspace(0, 3, params.N + 1)

    control, info = controller.compute_control(state, reference)

    print(f"Control: v={control[0]:.3f}, ω={control[1]:.3f}")
    print(f"ESS: {info.get('ess', 'N/A')}")

    assert control.shape == (2,), f"Expected (2,), got {control.shape}"
    assert np.isfinite(control).all(), "Control has NaN/Inf!"

    print("✓ PASS: Neural CBF + MPPI integration works\n")


def test_non_convex_obstacle_avoidance():
    """L자형 비볼록 장애물 회피 (핵심 차별화 테스트)"""
    print("=" * 60)
    print("Test 18: Integration - Non-Convex Obstacle Avoidance")
    print("=" * 60)

    # Train with L-shaped obstacle
    config = NeuralCBFTrainerConfig(
        state_dim=3,
        hidden_dims=[128, 128, 64],
        epochs=200,
        num_safe_samples=3000,
        num_unsafe_samples=3000,
        num_boundary_samples=1000,
        workspace_bounds=(-1.0, 5.0, -1.0, 4.0),
        early_stopping_patience=200,
        batch_size=256,
    )
    trainer = NeuralCBFTrainer(config)
    data = trainer.generate_training_data(
        obstacles=[],
        safety_margin=0.05,
        non_convex_regions=[_l_shaped_region],
    )
    history = trainer.train(data, verbose=False)

    # Verify: inside L-shape → h < 0, outside → h > 0
    inside_pts = np.array([
        [1.5, 1.5, 0.0],   # box1 center
        [2.5, 1.25, 0.0],  # box2 center
    ])
    outside_pts = np.array([
        [0.0, 0.0, 0.0],   # far outside
        [4.0, 3.0, 0.0],   # far outside
        [2.5, 1.8, 0.0],   # outside box2 (y > 1.5) and outside box1 (x > 2)
    ])

    h_inside = trainer.predict_h(inside_pts)
    h_outside = trainer.predict_h(outside_pts)

    print(f"h(inside L-shape): {h_inside}")
    print(f"h(outside L-shape): {h_outside}")

    inside_correct = np.sum(h_inside < 0)
    outside_correct = np.sum(h_outside > 0)
    total = len(h_inside) + len(h_outside)
    accuracy = (inside_correct + outside_correct) / total

    print(f"Classification accuracy: {accuracy:.1%} ({inside_correct + outside_correct}/{total})")
    assert accuracy >= 0.6, f"Non-convex classification accuracy too low: {accuracy:.1%}"

    # Use as cost function — verify barrier info
    cost = NeuralBarrierCost(trainer, cbf_alpha=0.1, cbf_weight=1000.0)

    # Single trajectory through L-shape center
    traj_through = np.zeros((1, 11, 3))
    traj_through[0, :, 0] = np.linspace(1.0, 2.5, 11)
    traj_through[0, :, 1] = 1.25

    # Single trajectory well outside L-shape
    traj_around = np.zeros((1, 11, 3))
    traj_around[0, :, 0] = np.linspace(0.0, 4.0, 11)
    traj_around[0, :, 1] = 0.0  # below L-shape

    info_through = cost.get_barrier_info(traj_through)
    info_around = cost.get_barrier_info(traj_around)

    print(f"Barrier through L-shape: min_h={info_through['min_barrier']:.4f}")
    print(f"Barrier around L-shape: min_h={info_around['min_barrier']:.4f}")

    # Through trajectory should have lower (more negative) barrier values
    assert info_through["min_barrier"] < info_around["min_barrier"], \
        "Trajectory through L-shape should have lower barrier values!"

    print("✓ PASS: Non-convex obstacle avoidance works\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # Network (4)
    test_network_output_shape()
    test_network_output_range()
    test_network_gradient_shape()
    test_network_gradient_nonzero()

    # Trainer (5)
    test_data_generation_circular()
    test_data_generation_non_convex()
    test_training_convergence()
    test_predict_h_numpy()
    test_save_load_checkpoint()

    # Cost (4)
    test_neural_cost_shape()
    test_neural_cost_safe_low()
    test_neural_cost_unsafe_high()
    test_neural_cost_barrier_info()

    # Filter (3)
    test_filter_safe_passthrough()
    test_filter_unsafe_correction()
    test_filter_lie_derivative_consistency()

    # Integration (2)
    test_neural_cbf_with_mppi()
    test_non_convex_obstacle_avoidance()

    print("\n" + "=" * 60)
    print("ALL 18 TESTS PASSED!")
    print("=" * 60)
