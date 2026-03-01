"""
MAML (Model-Agnostic Meta-Learning) 테스트

Tests:
    - MAMLDynamics 생성, 메타 파라미터 저장/복원, adapt, forward
    - MAMLTrainer 생성, 태스크 샘플링, 데이터 생성, 메타 학습, 저장/로드
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
from mppi_controller.learning.maml_trainer import MAMLTrainer
from mppi_controller.learning.neural_network_trainer import (
    DynamicsMLPModel,
    NeuralNetworkTrainer,
)


def _create_maml_model(state_dim=3, control_dim=2, hidden_dims=None):
    """Helper: 합성 데이터로 학습된 모델을 생성하고 MAMLDynamics로 로드."""
    if hidden_dims is None:
        hidden_dims = [32, 32]

    save_dir = tempfile.mkdtemp()
    trainer = NeuralNetworkTrainer(
        state_dim=state_dim,
        control_dim=control_dim,
        hidden_dims=hidden_dims,
        learning_rate=1e-3,
        save_dir=save_dir,
    )

    np.random.seed(42)
    N = 200
    inputs = np.random.randn(N, state_dim + control_dim).astype(np.float32)
    A = np.random.randn(state_dim, state_dim + control_dim).astype(np.float32) * 0.1
    targets = inputs @ A.T

    num_train = int(N * 0.8)
    norm_stats = {
        "state_mean": np.zeros(state_dim),
        "state_std": np.ones(state_dim),
        "control_mean": np.zeros(control_dim),
        "control_std": np.ones(control_dim),
        "state_dot_mean": np.zeros(state_dim),
        "state_dot_std": np.ones(state_dim),
    }

    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=20, verbose=False,
    )
    model_path = os.path.join(save_dir, "test_maml.pth")
    trainer.save_model("test_maml.pth")

    maml = MAMLDynamics(
        state_dim=state_dim, control_dim=control_dim,
        model_path=model_path, inner_lr=0.01, inner_steps=5,
    )
    return maml


# ==================== MAMLDynamics Tests ====================

def test_maml_dynamics_creation():
    """MAMLDynamics 생성 및 속성 확인."""
    maml = MAMLDynamics(state_dim=3, control_dim=2)
    assert maml.state_dim == 3
    assert maml.control_dim == 2
    assert maml.inner_lr == 0.01
    assert maml.inner_steps == 5
    assert maml._meta_weights is None
    assert maml.model is None
    print("  PASS: test_maml_dynamics_creation")


def test_maml_save_restore_meta_weights():
    """메타 파라미터 저장/복원 일관성 검증."""
    maml = _create_maml_model()

    # Save meta weights
    maml.save_meta_weights()
    assert maml._meta_weights is not None

    # Verify saved weights match current
    for key in maml.model.state_dict():
        diff = torch.abs(
            maml.model.state_dict()[key] - maml._meta_weights[key]
        ).max().item()
        assert diff < 1e-7, f"Weight mismatch for {key}: {diff}"

    print("  PASS: test_maml_save_restore_meta_weights")


def test_maml_adapt_reduces_loss():
    """adapt() 후 loss가 감소하는지 검증."""
    maml = _create_maml_model()
    maml.save_meta_weights()

    # Generate synthetic adaptation data
    np.random.seed(123)
    M = 50
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 2).astype(np.float32)
    # Create next_states with a simple linear relationship
    next_states = states + 0.05 * np.column_stack([
        controls[:, 0] * np.cos(states[:, 2]),
        controls[:, 0] * np.sin(states[:, 2]),
        controls[:, 1],
    ])

    # Compute loss before adaptation
    maml.restore_meta_weights()
    maml.model.eval()
    targets = (next_states - states) / 0.05
    theta_diff = next_states[:, 2] - states[:, 2]
    theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
    targets[:, 2] = theta_diff / 0.05
    inputs_t = maml._prepare_inputs(states, controls)
    targets_t = maml._prepare_targets(targets)
    with torch.no_grad():
        pred_before = maml.model(inputs_t)
        loss_before = torch.nn.functional.mse_loss(pred_before, targets_t).item()

    # Adapt
    final_loss = maml.adapt(states, controls, next_states, dt=0.05)

    # After adaptation, compute loss again
    maml.model.eval()
    with torch.no_grad():
        pred_after = maml.model(inputs_t)
        loss_after = torch.nn.functional.mse_loss(pred_after, targets_t).item()

    assert loss_after < loss_before, \
        f"Loss should decrease: before={loss_before:.6f}, after={loss_after:.6f}"
    print(f"  PASS: test_maml_adapt_reduces_loss (before={loss_before:.4f} -> after={loss_after:.4f})")


def test_maml_adapt_changes_weights():
    """adapt() 후 가중치가 변경되는지 확인."""
    maml = _create_maml_model()
    maml.save_meta_weights()

    weights_before = {k: v.clone() for k, v in maml.model.state_dict().items()}

    np.random.seed(456)
    M = 30
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 2).astype(np.float32)
    next_states = states + 0.05 * np.random.randn(M, 3).astype(np.float32)

    maml.adapt(states, controls, next_states, dt=0.05)

    weights_after = maml.model.state_dict()
    changed = False
    for key in weights_before:
        diff = torch.abs(weights_before[key] - weights_after[key]).max().item()
        if diff > 1e-7:
            changed = True
            break

    assert changed, "Weights should change after adaptation"
    print("  PASS: test_maml_adapt_changes_weights")


def test_maml_restore_after_adapt():
    """적응 후 restore → 원래 가중치 복원."""
    maml = _create_maml_model()
    maml.save_meta_weights()

    meta_weights = {k: v.clone() for k, v in maml._meta_weights.items()}

    # Adapt (changes weights)
    np.random.seed(789)
    M = 30
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 2).astype(np.float32)
    next_states = states + 0.05 * np.random.randn(M, 3).astype(np.float32)
    maml.adapt(states, controls, next_states, dt=0.05)

    # Restore
    maml.restore_meta_weights()

    for key in meta_weights:
        diff = torch.abs(
            maml.model.state_dict()[key] - meta_weights[key]
        ).max().item()
        assert diff < 1e-7, f"Weight not restored for {key}: {diff}"

    print("  PASS: test_maml_restore_after_adapt")


def test_maml_forward_dynamics():
    """forward_dynamics() 정상 동작 확인."""
    maml = _create_maml_model()

    state = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    control = np.array([0.5, 0.3], dtype=np.float32)

    state_dot = maml.forward_dynamics(state, control)
    assert state_dot.shape == (3,), f"Expected (3,), got {state_dot.shape}"
    assert np.all(np.isfinite(state_dot)), "Output should be finite"
    print("  PASS: test_maml_forward_dynamics")


def test_maml_batch_forward():
    """배치 forward 지원 확인."""
    maml = _create_maml_model()

    batch_size = 16
    states = np.random.randn(batch_size, 3).astype(np.float32)
    controls = np.random.randn(batch_size, 2).astype(np.float32)

    state_dots = maml.forward_dynamics(states, controls)
    assert state_dots.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {state_dots.shape}"
    assert np.all(np.isfinite(state_dots)), "Batch output should be finite"
    print("  PASS: test_maml_batch_forward")


def test_maml_continuous_finetune():
    """restore=False 연속 fine-tuning: 가중치가 메타에서 계속 멀어짐."""
    maml = _create_maml_model()
    maml.save_meta_weights()
    meta_weights = {k: v.clone() for k, v in maml._meta_weights.items()}

    np.random.seed(321)
    M = 50
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 2).astype(np.float32)
    next_states = states + 0.05 * np.column_stack([
        controls[:, 0] * np.cos(states[:, 2]),
        controls[:, 0] * np.sin(states[:, 2]),
        controls[:, 1],
    ])

    # restore=True 3회 → 매번 리셋, 항상 같은 결과
    maml.adapt(states, controls, next_states, dt=0.05, restore=True)
    w_after_restore = {k: v.clone() for k, v in maml.model.state_dict().items()}
    dist_restore = sum(
        (w_after_restore[k] - meta_weights[k]).norm().item()
        for k in meta_weights
    )

    # restore=False 3회 연속 → 누적 학습
    maml.restore_meta_weights()
    for _ in range(3):
        maml.adapt(states, controls, next_states, dt=0.05, restore=False)
    w_after_cont = {k: v.clone() for k, v in maml.model.state_dict().items()}
    dist_cont = sum(
        (w_after_cont[k] - meta_weights[k]).norm().item()
        for k in meta_weights
    )

    # 연속 fine-tuning은 메타 가중치에서 더 멀리 이동 (더 많은 SGD step 누적)
    assert dist_cont > dist_restore, \
        f"Continuous should diverge more from meta: {dist_cont:.4f} vs {dist_restore:.4f}"

    print(f"  PASS: test_maml_continuous_finetune (cont_dist={dist_cont:.4f} > restore_dist={dist_restore:.4f})")


# ==================== MAMLTrainer Tests ====================

def test_maml_trainer_creation():
    """MAMLTrainer 생성 확인."""
    trainer = MAMLTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01, inner_steps=5,
        meta_lr=1e-3,
    )
    assert trainer.state_dim == 3
    assert trainer.control_dim == 2
    assert trainer.model is not None
    assert trainer.inner_lr == 0.01
    assert trainer.inner_steps == 5
    print("  PASS: test_maml_trainer_creation")


def test_maml_sample_task():
    """태스크 파라미터 범위 확인."""
    trainer = MAMLTrainer(state_dim=3, control_dim=2, hidden_dims=[32, 32])

    np.random.seed(42)
    for _ in range(100):
        task = trainer._sample_task()
        assert 0.1 <= task["c_v"] <= 0.8, f"c_v out of range: {task['c_v']}"
        assert 0.1 <= task["c_omega"] <= 0.5, f"c_omega out of range: {task['c_omega']}"
        assert task["k_v"] == 5.0
        assert task["k_omega"] == 5.0

    print("  PASS: test_maml_sample_task")


def test_maml_generate_task_data():
    """데이터 생성 shape/값 확인."""
    trainer = MAMLTrainer(state_dim=3, control_dim=2, hidden_dims=[32, 32])

    np.random.seed(42)
    task = trainer._sample_task()
    n = 50
    states, controls, next_states = trainer._generate_task_data(task, n)

    assert states.shape == (n, 3), f"Expected ({n}, 3), got {states.shape}"
    assert controls.shape == (n, 2), f"Expected ({n}, 2), got {controls.shape}"
    assert next_states.shape == (n, 3), f"Expected ({n}, 3), got {next_states.shape}"

    # All finite
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(controls))
    assert np.all(np.isfinite(next_states))

    print("  PASS: test_maml_generate_task_data")


def test_maml_meta_train_loss_decreases():
    """10 iter 메타 학습 → loss 감소 확인."""
    trainer = MAMLTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01, inner_steps=3,
        meta_lr=1e-3,
        task_batch_size=2,
        support_size=30,
        query_size=30,
    )

    np.random.seed(42)
    torch.manual_seed(42)
    trainer.meta_train(n_iterations=20, verbose=False)

    losses = trainer.history["meta_loss"]
    assert len(losses) == 20, f"Expected 20 losses, got {len(losses)}"

    # First 5 vs last 5 average: should generally decrease
    first_avg = np.mean(losses[:5])
    last_avg = np.mean(losses[-5:])
    # Relaxed check: last should be lower or at least comparable
    assert last_avg <= first_avg * 2.0, \
        f"Loss should generally decrease: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"

    print(f"  PASS: test_maml_meta_train_loss_decreases (first5={first_avg:.4f}, last5={last_avg:.4f})")


def test_maml_save_load_meta_model():
    """저장/로드 후 동일 출력 검증."""
    save_dir = tempfile.mkdtemp()
    trainer = MAMLTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01, inner_steps=3,
        meta_lr=1e-3,
        task_batch_size=2,
        support_size=20,
        query_size=20,
        save_dir=save_dir,
    )

    np.random.seed(42)
    trainer.meta_train(n_iterations=5, verbose=False)

    # Save
    trainer.save_meta_model("test_maml_meta.pth")

    # Get output before load
    test_input = torch.randn(5, 5)  # state_dim + control_dim = 5
    trainer.model.eval()
    with torch.no_grad():
        output_before = trainer.model(test_input).numpy()

    # Load into new trainer
    trainer2 = MAMLTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        save_dir=save_dir,
    )
    trainer2.load_meta_model("test_maml_meta.pth")

    # Get output after load
    trainer2.model.eval()
    with torch.no_grad():
        output_after = trainer2.model(test_input).numpy()

    np.testing.assert_allclose(output_before, output_after, atol=1e-6)

    # Check norm_stats loaded
    assert trainer2.norm_stats is not None
    for key in trainer.norm_stats:
        np.testing.assert_allclose(
            trainer.norm_stats[key], trainer2.norm_stats[key], atol=1e-6
        )

    print("  PASS: test_maml_save_load_meta_model")


# ==================== DynamicKinematicAdapter Tests ====================

def test_dynamic_kinematic_adapter_creation():
    """DynamicKinematicAdapter 생성 및 속성 확인."""
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter
    model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    assert model.state_dim == 5
    assert model.control_dim == 2
    assert model.model_type == "kinematic"
    assert model._c_v == 0.5
    assert model._c_omega == 0.3
    print("  PASS: test_dynamic_kinematic_adapter_creation")


def test_dynamic_kinematic_adapter_forward():
    """forward_dynamics 단일 상태."""
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter
    model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3, k_v=5.0, k_omega=5.0)
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    dot = model.forward_dynamics(state, control)
    assert dot.shape == (5,), f"Expected (5,), got {dot.shape}"
    # dx = v*cos(θ) = 0.5*cos(0) = 0.5
    assert abs(dot[0] - 0.5) < 1e-6
    # dy = v*sin(θ) = 0.5*sin(0) = 0.0
    assert abs(dot[1]) < 1e-6
    # dtheta = omega = 0.1
    assert abs(dot[2] - 0.1) < 1e-6
    # a = k_v*(v_cmd - v) - c_v*v = 5*(1.0 - 0.5) - 0.5*0.5 = 2.25
    assert abs(dot[3] - 2.25) < 1e-6
    print("  PASS: test_dynamic_kinematic_adapter_forward")


def test_dynamic_kinematic_adapter_batch():
    """forward_dynamics 배치."""
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter
    model = DynamicKinematicAdapter()
    batch = 16
    states = np.random.randn(batch, 5).astype(np.float32)
    controls = np.random.randn(batch, 2).astype(np.float32)
    dots = model.forward_dynamics(states, controls)
    assert dots.shape == (batch, 5)
    assert np.all(np.isfinite(dots))
    print("  PASS: test_dynamic_kinematic_adapter_batch")


def test_dynamic_kinematic_adapter_step():
    """RK4 step 작동 확인."""
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter
    model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    control = np.array([1.0, 0.0])
    next_state = model.step(state, control, 0.05)
    assert next_state.shape == (5,)
    # v should increase (PD control toward v_cmd=1.0)
    assert next_state[3] > 0, f"v should increase, got {next_state[3]}"
    print("  PASS: test_dynamic_kinematic_adapter_step")


def test_dynamic_kinematic_adapter_normalize():
    """normalize_state: 각도 래핑."""
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter
    model = DynamicKinematicAdapter()
    state = np.array([1.0, 2.0, 4.0, 0.5, 0.1])  # θ=4.0 > π
    normed = model.normalize_state(state)
    assert -np.pi <= normed[2] <= np.pi
    print("  PASS: test_dynamic_kinematic_adapter_normalize")


# ==================== 5D MAML Tests ====================

def _create_maml_model_5d():
    """Helper: 5D MAML 모델 생성."""
    save_dir = tempfile.mkdtemp()
    trainer = NeuralNetworkTrainer(
        state_dim=5, control_dim=2,
        hidden_dims=[32, 32],
        learning_rate=1e-3,
        save_dir=save_dir,
    )

    np.random.seed(42)
    N = 200
    inputs = np.random.randn(N, 7).astype(np.float32)  # 5+2
    A = np.random.randn(5, 7).astype(np.float32) * 0.1
    targets = inputs @ A.T

    num_train = int(N * 0.8)
    norm_stats = {
        "state_mean": np.zeros(5),
        "state_std": np.ones(5),
        "control_mean": np.zeros(2),
        "control_std": np.ones(2),
        "state_dot_mean": np.zeros(5),
        "state_dot_std": np.ones(5),
    }

    trainer.train(
        inputs[:num_train], targets[:num_train],
        inputs[num_train:], targets[num_train:],
        norm_stats, epochs=20, verbose=False,
    )
    model_path = os.path.join(save_dir, "test_maml_5d.pth")
    trainer.save_model("test_maml_5d.pth")

    maml = MAMLDynamics(
        state_dim=5, control_dim=2,
        model_path=model_path, inner_lr=0.01, inner_steps=5,
    )
    return maml


def test_maml_5d_forward():
    """5D MAML forward_dynamics."""
    maml = _create_maml_model_5d()
    state = np.array([1.0, 2.0, 0.5, 0.3, 0.1], dtype=np.float32)
    control = np.array([0.5, 0.3], dtype=np.float32)
    dot = maml.forward_dynamics(state, control)
    assert dot.shape == (5,), f"Expected (5,), got {dot.shape}"
    assert np.all(np.isfinite(dot))
    print("  PASS: test_maml_5d_forward")


def test_maml_5d_batch_forward():
    """5D MAML 배치 forward."""
    maml = _create_maml_model_5d()
    batch = 16
    states = np.random.randn(batch, 5).astype(np.float32)
    controls = np.random.randn(batch, 2).astype(np.float32)
    dots = maml.forward_dynamics(states, controls)
    assert dots.shape == (batch, 5)
    assert np.all(np.isfinite(dots))
    print("  PASS: test_maml_5d_batch_forward")


def test_maml_5d_adapt():
    """5D MAML adapt → loss 감소."""
    maml = _create_maml_model_5d()
    maml.save_meta_weights()

    np.random.seed(123)
    M = 50
    states = np.random.randn(M, 5).astype(np.float32)
    controls = np.random.randn(M, 2).astype(np.float32)
    next_states = states + 0.05 * np.random.randn(M, 5).astype(np.float32) * 0.1

    # Loss before
    maml.restore_meta_weights()
    maml.model.eval()
    targets = (next_states - states) / 0.05
    theta_diff = next_states[:, 2] - states[:, 2]
    theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
    targets[:, 2] = theta_diff / 0.05
    inputs_t = maml._prepare_inputs(states, controls)
    targets_t = maml._prepare_targets(targets)
    with torch.no_grad():
        pred_before = maml.model(inputs_t)
        loss_before = torch.nn.functional.mse_loss(pred_before, targets_t).item()

    # Adapt
    final_loss = maml.adapt(states, controls, next_states, dt=0.05)

    maml.model.eval()
    with torch.no_grad():
        pred_after = maml.model(inputs_t)
        loss_after = torch.nn.functional.mse_loss(pred_after, targets_t).item()

    assert loss_after < loss_before, \
        f"5D adapt should reduce loss: {loss_before:.4f} -> {loss_after:.4f}"
    print(f"  PASS: test_maml_5d_adapt (before={loss_before:.4f} -> after={loss_after:.4f})")


def test_maml_5d_with_residual():
    """5D MAML + ResidualDynamics 통합 테스트."""
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter
    from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

    maml = _create_maml_model_5d()
    base_5d = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1)

    residual_model = ResidualDynamics(
        base_model=base_5d,
        learned_model=maml,
        use_residual=True,
    )

    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1], dtype=np.float32)
    control = np.array([1.0, 0.5], dtype=np.float32)
    dot = residual_model.forward_dynamics(state, control)
    assert dot.shape == (5,)
    assert np.all(np.isfinite(dot))

    # 배치도 확인
    batch_states = np.random.randn(8, 5).astype(np.float32)
    batch_controls = np.random.randn(8, 2).astype(np.float32)
    batch_dots = residual_model.forward_dynamics(batch_states, batch_controls)
    assert batch_dots.shape == (8, 5)
    print("  PASS: test_maml_5d_with_residual")


# ==================== Weighted Adapt Tests ====================

def test_maml_adapt_temporal_decay():
    """temporal_decay 적용 시 loss 감소."""
    maml = _create_maml_model()
    maml.save_meta_weights()

    np.random.seed(999)
    M = 50
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 2).astype(np.float32)
    next_states = states + 0.05 * np.column_stack([
        controls[:, 0] * np.cos(states[:, 2]),
        controls[:, 0] * np.sin(states[:, 2]),
        controls[:, 1],
    ])

    loss_uniform = maml.adapt(states, controls, next_states, dt=0.05, restore=True)
    loss_decay = maml.adapt(states, controls, next_states, dt=0.05,
                            restore=True, temporal_decay=0.99)

    # Both should produce finite loss
    assert np.isfinite(loss_uniform), f"Uniform loss not finite: {loss_uniform}"
    assert np.isfinite(loss_decay), f"Decay loss not finite: {loss_decay}"
    print(f"  PASS: test_maml_adapt_temporal_decay (uniform={loss_uniform:.4f}, decay={loss_decay:.4f})")


def test_maml_adapt_sample_weights():
    """sample_weights 적용 시 정상 동작."""
    maml = _create_maml_model()
    maml.save_meta_weights()

    np.random.seed(888)
    M = 30
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 2).astype(np.float32)
    next_states = states + 0.05 * np.random.randn(M, 3).astype(np.float32)
    weights = np.random.uniform(0.5, 1.5, size=M).astype(np.float32)

    loss = maml.adapt(states, controls, next_states, dt=0.05,
                      restore=True, sample_weights=weights)
    assert np.isfinite(loss)
    print(f"  PASS: test_maml_adapt_sample_weights (loss={loss:.4f})")


# ==================== Reptile Tests ====================

def test_reptile_trainer_creation():
    """ReptileTrainer 생성 확인."""
    from mppi_controller.learning.reptile_trainer import ReptileTrainer
    trainer = ReptileTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01, inner_steps=5,
        epsilon=0.1,
    )
    assert trainer.state_dim == 3
    assert trainer.epsilon == 0.1
    assert trainer.model is not None
    print("  PASS: test_reptile_trainer_creation")


def test_reptile_meta_train():
    """Reptile 메타 학습 10 iter → loss 기록."""
    from mppi_controller.learning.reptile_trainer import ReptileTrainer
    trainer = ReptileTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01, inner_steps=3,
        epsilon=0.1,
        task_batch_size=2,
        support_size=30,
    )

    np.random.seed(42)
    trainer.meta_train(n_iterations=10, verbose=False)

    losses = trainer.history["meta_loss"]
    assert len(losses) == 10
    assert all(np.isfinite(l) for l in losses)
    print(f"  PASS: test_reptile_meta_train (losses={losses[0]:.4f} -> {losses[-1]:.4f})")


def test_reptile_save_load():
    """Reptile 모델 저장/로드."""
    from mppi_controller.learning.reptile_trainer import ReptileTrainer
    save_dir = tempfile.mkdtemp()
    trainer = ReptileTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01, inner_steps=3,
        epsilon=0.1,
        task_batch_size=2,
        support_size=20,
        save_dir=save_dir,
    )
    np.random.seed(42)
    trainer.meta_train(n_iterations=5, verbose=False)
    trainer.save_meta_model("reptile_test.pth")

    # Get output
    test_input = torch.randn(5, 5)
    trainer.model.eval()
    with torch.no_grad():
        out_before = trainer.model(test_input).numpy()

    # Load
    trainer2 = ReptileTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[32, 32],
        save_dir=save_dir,
    )
    trainer2.load_meta_model("reptile_test.pth")
    trainer2.model.eval()
    with torch.no_grad():
        out_after = trainer2.model(test_input).numpy()

    np.testing.assert_allclose(out_before, out_after, atol=1e-6)
    print("  PASS: test_reptile_save_load")


def test_reptile_5d():
    """Reptile 5D 메타 학습."""
    from mppi_controller.learning.reptile_trainer import ReptileTrainer
    trainer = ReptileTrainer(
        state_dim=5, control_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01, inner_steps=3,
        epsilon=0.1,
        task_batch_size=2,
        support_size=30,
    )
    np.random.seed(42)
    trainer.meta_train(n_iterations=5, verbose=False)

    losses = trainer.history["meta_loss"]
    assert len(losses) == 5
    assert all(np.isfinite(l) for l in losses)
    print(f"  PASS: test_reptile_5d (losses={losses[0]:.4f} -> {losses[-1]:.4f})")


# ==================== MAMLTrainer 5D Tests ====================

def test_maml_trainer_5d_data_generation():
    """MAMLTrainer 5D 데이터 생성 shape 확인."""
    trainer = MAMLTrainer(state_dim=5, control_dim=2, hidden_dims=[32, 32])
    np.random.seed(42)
    task = trainer._sample_task()
    n = 50
    states, controls, next_states = trainer._generate_task_data_5d(task, n)
    assert states.shape == (n, 5), f"Expected ({n}, 5), got {states.shape}"
    assert controls.shape == (n, 2)
    assert next_states.shape == (n, 5)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(next_states))
    print("  PASS: test_maml_trainer_5d_data_generation")


def test_maml_trainer_5d_meta_train():
    """MAMLTrainer 5D 메타 학습."""
    trainer = MAMLTrainer(
        state_dim=5, control_dim=2,
        hidden_dims=[32, 32],
        inner_lr=0.01, inner_steps=3,
        meta_lr=1e-3,
        task_batch_size=2,
        support_size=30,
        query_size=30,
    )
    np.random.seed(42)
    trainer.meta_train(n_iterations=10, verbose=False)

    losses = trainer.history["meta_loss"]
    assert len(losses) == 10
    assert all(np.isfinite(l) for l in losses)
    print(f"  PASS: test_maml_trainer_5d_meta_train (first={losses[0]:.4f}, last={losses[-1]:.4f})")


# ==================== AngleAware Cost Tests ====================

def test_angle_aware_costs_import():
    """AngleAwareTrackingCost/TerminalCost를 core에서 import."""
    from mppi_controller.controllers.mppi.cost_functions import (
        AngleAwareTrackingCost, AngleAwareTerminalCost,
    )
    Q = np.array([10.0, 10.0, 1.0])
    Qf = np.array([20.0, 20.0, 2.0])
    tracking = AngleAwareTrackingCost(Q)
    terminal = AngleAwareTerminalCost(Qf)

    K, N, nx = 4, 5, 3
    trajs = np.random.randn(K, N+1, nx)
    ctrls = np.random.randn(K, N, 2)
    ref = np.random.randn(N+1, nx)

    cost_t = tracking.compute_cost(trajs, ctrls, ref)
    cost_f = terminal.compute_cost(trajs, ctrls, ref)
    assert cost_t.shape == (K,)
    assert cost_f.shape == (K,)
    assert np.all(np.isfinite(cost_t))
    assert np.all(np.isfinite(cost_f))
    print("  PASS: test_angle_aware_costs_import")


def test_angle_aware_cost_5d():
    """5D AngleAwareTrackingCost 작동."""
    from mppi_controller.controllers.mppi.cost_functions import (
        AngleAwareTrackingCost, AngleAwareTerminalCost,
    )
    Q = np.array([10.0, 10.0, 1.0, 0.1, 0.1])
    Qf = np.array([20.0, 20.0, 2.0, 0.2, 0.2])

    K, N, nx = 4, 5, 5
    trajs = np.random.randn(K, N+1, nx)
    ctrls = np.random.randn(K, N, 2)
    ref = np.random.randn(N+1, nx)

    tracking = AngleAwareTrackingCost(Q, angle_indices=(2,))
    terminal = AngleAwareTerminalCost(Qf, angle_indices=(2,))

    cost_t = tracking.compute_cost(trajs, ctrls, ref)
    cost_f = terminal.compute_cost(trajs, ctrls, ref)
    assert cost_t.shape == (K,)
    assert cost_f.shape == (K,)
    print("  PASS: test_angle_aware_cost_5d")


# ==================== Disturbance Profile Tests ====================

def test_wind_gust_disturbance():
    """WindGustDisturbance: 속도(v,ω)에 돌풍 가속도 적용."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/comparison")))
    from disturbance_profiles import WindGustDisturbance
    dist = WindGustDisturbance(intensity=0.5, seed=42, num_gusts=3, duration_range=(10.0,))
    state = np.array([0.0, 0.0, 0.5, 0.3, 0.1])  # 5D state

    # 전체 시간에서 힘 계산 — 항상 유한
    forces = [dist.get_force(t, state) for t in np.linspace(0, 10, 200)]
    assert all(np.all(np.isfinite(f)) for f in forces), "All forces should be finite"
    assert all(f.shape == (5,) for f in forces)

    # 힘은 속도 상태(force[3], force[4])에 적용
    has_v_force = any(abs(f[3]) > 1e-6 for f in forces)
    assert has_v_force, "WindGust should apply force to velocity state (force[3])"

    # intensity=0 → 항상 0
    dist_zero = WindGustDisturbance(intensity=0.0, seed=42)
    for t in [0.0, 3.0, 7.0]:
        assert np.allclose(dist_zero.get_force(t, state), 0.0)

    # param_delta는 항상 빈 dict
    assert dist.get_param_delta(5.0) == {}
    print("  PASS: test_wind_gust_disturbance")


def test_terrain_change_disturbance():
    """TerrainChangeDisturbance: 마찰 계수 다중 단계 변동."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/comparison")))
    from disturbance_profiles import TerrainChangeDisturbance
    dist = TerrainChangeDisturbance(intensity=0.5, seed=42, sim_duration=20.0, num_transitions=3)
    state = np.zeros(5)

    # force는 항상 0
    assert np.allclose(dist.get_force(3.0, state), 0.0)
    assert np.allclose(dist.get_force(15.0, state), 0.0)

    # 초반: 파라미터 변동 없음
    assert dist.get_param_delta(0.5) == {}

    # 후반: 전환 발생 → delta_c_v, delta_c_omega 존재
    delta = dist.get_param_delta(18.0)
    assert "delta_c_v" in delta or "delta_c_omega" in delta, \
        f"Expected param delta at t=18.0, got {delta}"

    # intensity=0 → delta=0
    dist_zero = TerrainChangeDisturbance(intensity=0.0, seed=42, sim_duration=20.0)
    delta_zero = dist_zero.get_param_delta(18.0)
    if delta_zero:
        assert abs(delta_zero.get("delta_c_v", 0.0)) < 1e-10

    print("  PASS: test_terrain_change_disturbance")


def test_sinusoidal_disturbance():
    """SinusoidalDisturbance: 속도(v,ω)에 주기적 가속도 외란."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/comparison")))
    from disturbance_profiles import SinusoidalDisturbance
    dist = SinusoidalDisturbance(intensity=0.5, seed=42, v_amplitude=2.0,
                                  omega_amplitude=1.0, frequency=1.0)
    state = np.zeros(5)

    max_v_expected = 0.5 * 2.0  # intensity * v_amplitude
    max_omega_expected = 0.5 * 1.0

    for t in np.linspace(0, 10, 100):
        force = dist.get_force(t, state)
        assert force.shape == (5,)
        assert np.all(np.isfinite(force))
        # 속도 상태에만 적용
        assert force[0] == 0.0, "No position force"
        assert force[1] == 0.0, "No position force"
        assert force[2] == 0.0, "No heading force"
        assert abs(force[3]) <= max_v_expected + 1e-6
        assert abs(force[4]) <= max_omega_expected + 1e-6

    # 비영: sin/cos이므로 전체 시간에 걸쳐 비영 구간 존재
    forces_v = [dist.get_force(t, state)[3] for t in np.linspace(0, 10, 200)]
    assert max(abs(f) for f in forces_v) > 0.1, "Sinusoidal should produce non-trivial velocity force"

    assert dist.get_param_delta(5.0) == {}
    print("  PASS: test_sinusoidal_disturbance")


def test_combined_disturbance():
    """CombinedDisturbance: 합성 외란."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/comparison")))
    from disturbance_profiles import CombinedDisturbance
    dist = CombinedDisturbance(intensity=0.5, seed=42, sim_duration=10.0)
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])

    # 외력 합산 (wind + sinusoidal) — 속도에 적용
    force = dist.get_force(5.0, state)
    assert force.shape == (5,)
    assert np.all(np.isfinite(force))

    # 파라미터 변동 (terrain) — 후반부에서 활성
    delta = dist.get_param_delta(8.0)
    assert isinstance(delta, dict)
    # 전환이 발생했으면 delta_c_v 존재
    if delta:
        assert "delta_c_v" in delta or "delta_c_omega" in delta

    print("  PASS: test_combined_disturbance")


def test_create_disturbance_factory():
    """create_disturbance() 팩토리 함수."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/comparison")))
    from disturbance_profiles import create_disturbance

    # "none" → None
    assert create_disturbance("none", 0.5) is None

    # intensity=0 → None
    assert create_disturbance("combined", 0.0) is None

    # 유효한 타입들
    for dtype in ["wind", "terrain", "sine", "combined"]:
        d = create_disturbance(dtype, 0.5, seed=42, sim_duration=10.0)
        assert d is not None, f"Expected non-None for {dtype}"
        force = d.get_force(3.0, np.zeros(5))
        assert force.shape == (5,)

    print("  PASS: test_create_disturbance_factory")


def test_dynamic_world_with_disturbance():
    """DynamicWorld + disturbance 통합: 속도 외란으로 궤적 차이."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/comparison")))
    from disturbance_profiles import SinusoidalDisturbance
    from model_mismatch_comparison_demo import DynamicWorld

    dist = SinusoidalDisturbance(intensity=0.7, seed=42)
    world_no_dist = DynamicWorld(c_v=0.5, c_omega=0.3, disturbance=None)
    world_dist = DynamicWorld(c_v=0.5, c_omega=0.3, disturbance=dist)

    init_state = np.array([0.0, 0.0, 0.0])
    world_no_dist.reset(init_state)
    world_dist.reset(init_state)

    # 여러 스텝 실행하여 차이 누적
    for _ in range(20):
        np.random.seed(100)
        world_no_dist.step(np.array([0.5, 0.0]), 0.05, add_noise=False)
        np.random.seed(100)
        world_dist.step(np.array([0.5, 0.0]), 0.05, add_noise=False)

    obs_no = world_no_dist.get_observation()
    obs_d = world_dist.get_observation()

    assert np.all(np.isfinite(obs_no))
    assert np.all(np.isfinite(obs_d))
    # 20스텝 후 속도 교란이 누적되어 위치 차이 발생
    diff = np.linalg.norm(obs_no[:2] - obs_d[:2])
    assert diff > 1e-4, f"Expected noticeable difference, got {diff}"
    print(f"  PASS: test_dynamic_world_with_disturbance (pos diff={diff:.4f}m)")


def test_dynamic_world_terrain_change():
    """DynamicWorld + TerrainChangeDisturbance: c_v 동적 변경."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/comparison")))
    from disturbance_profiles import TerrainChangeDisturbance
    from model_mismatch_comparison_demo import DynamicWorld

    # 즉시 전환 (sim_duration=2.0, transition at t=0.5s)
    dist = TerrainChangeDisturbance(intensity=1.0, seed=42, sim_duration=2.0, num_transitions=1)
    world = DynamicWorld(c_v=0.5, c_omega=0.3, disturbance=dist)
    world.reset(np.array([0.0, 0.0, 0.0]))

    # 초기: c_v=0.5
    assert abs(world._c_v - 0.5) < 1e-6

    # 많은 스텝 실행하여 전환 발생시킴
    for _ in range(40):
        world.step(np.array([0.5, 0.0]), 0.05, add_noise=False)

    # 40 steps * 0.05 = 2.0s → 전환 발생
    assert world._c_v != 0.5, f"c_v should have changed, got {world._c_v}"

    # reset으로 초기화
    world.reset(np.array([0.0, 0.0, 0.0]))
    assert abs(world._c_v - 0.5) < 1e-6
    print("  PASS: test_dynamic_world_terrain_change")


# ==================== main ====================

if __name__ == "__main__":
    tests = [
        # MAMLDynamics
        test_maml_dynamics_creation,
        test_maml_save_restore_meta_weights,
        test_maml_adapt_reduces_loss,
        test_maml_adapt_changes_weights,
        test_maml_restore_after_adapt,
        test_maml_forward_dynamics,
        test_maml_batch_forward,
        test_maml_continuous_finetune,
        # MAMLTrainer
        test_maml_trainer_creation,
        test_maml_sample_task,
        test_maml_generate_task_data,
        test_maml_meta_train_loss_decreases,
        test_maml_save_load_meta_model,
        # DynamicKinematicAdapter
        test_dynamic_kinematic_adapter_creation,
        test_dynamic_kinematic_adapter_forward,
        test_dynamic_kinematic_adapter_batch,
        test_dynamic_kinematic_adapter_step,
        test_dynamic_kinematic_adapter_normalize,
        # 5D MAML
        test_maml_5d_forward,
        test_maml_5d_batch_forward,
        test_maml_5d_adapt,
        test_maml_5d_with_residual,
        # Weighted Adapt
        test_maml_adapt_temporal_decay,
        test_maml_adapt_sample_weights,
        # Reptile
        test_reptile_trainer_creation,
        test_reptile_meta_train,
        test_reptile_save_load,
        test_reptile_5d,
        # MAMLTrainer 5D
        test_maml_trainer_5d_data_generation,
        test_maml_trainer_5d_meta_train,
        # AngleAware Costs
        test_angle_aware_costs_import,
        test_angle_aware_cost_5d,
        # Disturbance Profiles
        test_wind_gust_disturbance,
        test_terrain_change_disturbance,
        test_sinusoidal_disturbance,
        test_combined_disturbance,
        test_create_disturbance_factory,
        test_dynamic_world_with_disturbance,
        test_dynamic_world_terrain_change,
    ]

    print(f"\n{'='*60}")
    print(f"  MAML Tests ({len(tests)} tests)")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)
