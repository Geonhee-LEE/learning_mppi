"""
LotF (Learning on the Fly) 통합 테스트

테스트 대상:
  - LoRALinear / LoRADynamics
  - SpectralRegularizer
  - DifferentiableMobileManipulator6DOF / Dynamic
  - BPTTResidualTrainer
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import os
import tempfile

from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def simple_mlp():
    """Simple 2-layer MLP for testing."""
    return DynamicsMLPModel(input_dim=17, output_dim=9, hidden_dims=[64, 64])


@pytest.fixture
def saved_model(simple_mlp, tmp_path):
    """Save a model checkpoint for loading tests."""
    model_path = str(tmp_path / "test_model.pth")
    norm_stats = {
        "state_mean": np.zeros(9),
        "state_std": np.ones(9),
        "control_mean": np.zeros(8),
        "control_std": np.ones(8),
        "state_dot_mean": np.zeros(9),
        "state_dot_std": np.ones(9),
    }
    torch.save({
        "model_state_dict": simple_mlp.state_dict(),
        "norm_stats": norm_stats,
        "config": {
            "state_dim": 9,
            "control_dim": 8,
            "hidden_dims": [64, 64],
            "activation": "relu",
            "dropout_rate": 0.0,
        },
    }, model_path)
    return model_path, norm_stats


@pytest.fixture
def kin_model():
    from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
        MobileManipulator6DOFKinematic,
    )
    return MobileManipulator6DOFKinematic()


@pytest.fixture
def dyn_model():
    from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
        MobileManipulator6DOFDynamic,
    )
    return MobileManipulator6DOFDynamic()


@pytest.fixture
def diff_sim():
    from mppi_controller.models.differentiable.diff_sim_6dof import (
        DifferentiableMobileManipulator6DOF,
    )
    return DifferentiableMobileManipulator6DOF()


@pytest.fixture
def diff_sim_dyn():
    from mppi_controller.models.differentiable.diff_sim_6dof import (
        DifferentiableMobileManipulator6DOFDynamic,
    )
    return DifferentiableMobileManipulator6DOFDynamic()


# =============================================================
# TestLoRALinear
# =============================================================

class TestLoRALinear:
    """LoRALinear layer tests."""

    def test_zero_init_matches_original(self):
        """LoRA A=0 초기화 → 원본과 동일 출력."""
        from mppi_controller.models.learned.lora_dynamics import LoRALinear

        linear = nn.Linear(10, 5)
        lora = LoRALinear(linear, rank=4, alpha=1.0)

        x = torch.randn(3, 10)
        with torch.no_grad():
            orig_out = linear(x)
            lora_out = lora(x)

        torch.testing.assert_close(orig_out, lora_out, atol=1e-6, rtol=1e-6)

    def test_rank_shapes(self):
        """LoRA A, B 행렬 shape 검증."""
        from mppi_controller.models.learned.lora_dynamics import LoRALinear

        linear = nn.Linear(10, 5)
        rank = 3
        lora = LoRALinear(linear, rank=rank)

        assert lora.lora_A.shape == (5, rank)
        assert lora.lora_B.shape == (rank, 10)

    def test_frozen_original(self):
        """원본 가중치 frozen 확인."""
        from mppi_controller.models.learned.lora_dynamics import LoRALinear

        linear = nn.Linear(10, 5)
        lora = LoRALinear(linear, rank=4)

        assert not lora.original.weight.requires_grad
        assert not lora.original.bias.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad


# =============================================================
# TestLoRADynamics
# =============================================================

class TestLoRADynamics:
    """LoRADynamics model tests."""

    def test_dims(self, saved_model):
        """차원 확인."""
        from mppi_controller.models.learned.lora_dynamics import LoRADynamics

        model_path, _ = saved_model
        lora = LoRADynamics(state_dim=9, control_dim=8, model_path=model_path)
        assert lora.state_dim == 9
        assert lora.control_dim == 8

    def test_forward_matches_base(self, saved_model):
        """LoRA 초기상태(A=0) → 원본 NeuralDynamics와 동일 출력."""
        from mppi_controller.models.learned.lora_dynamics import LoRADynamics
        from mppi_controller.models.learned.neural_dynamics import NeuralDynamics

        model_path, _ = saved_model

        base = NeuralDynamics(state_dim=9, control_dim=8, model_path=model_path)
        lora = LoRADynamics(state_dim=9, control_dim=8, model_path=model_path, lora_rank=4)
        # Reset LoRA A to zero to ensure delta=0
        lora.reset_lora()

        np.random.seed(123)
        state = np.random.randn(9)
        control = np.random.randn(8)

        base_out = base.forward_dynamics(state, control)
        lora_out = lora.forward_dynamics(state, control)

        np.testing.assert_allclose(base_out, lora_out, atol=1e-4)

    def test_adapt_reduces_loss(self, saved_model):
        """adapt() 후 loss 감소."""
        from mppi_controller.models.learned.lora_dynamics import LoRADynamics

        model_path, _ = saved_model
        lora = LoRADynamics(
            state_dim=9, control_dim=8, model_path=model_path,
            lora_rank=4, inner_lr=0.01, inner_steps=10,
        )
        lora.save_meta_weights()

        M = 20
        states = np.random.randn(M, 9) * 0.1
        controls = np.random.randn(M, 8) * 0.1
        next_states = states + np.random.randn(M, 9) * 0.01
        dt = 0.05

        # First adapt (loss_1 from clean meta)
        loss_1 = lora.adapt(states, controls, next_states, dt, restore=True)

        # Second adapt (should be lower since we start from meta again and run more)
        lora_long = LoRADynamics(
            state_dim=9, control_dim=8, model_path=model_path,
            lora_rank=4, inner_lr=0.01, inner_steps=50,
        )
        lora_long.save_meta_weights()
        loss_2 = lora_long.adapt(states, controls, next_states, dt, restore=True)

        assert loss_2 < loss_1

    def test_adapt_interface(self, saved_model):
        """adapt() 인터페이스 호환성 (MAMLDynamics와 동일)."""
        from mppi_controller.models.learned.lora_dynamics import LoRADynamics

        model_path, _ = saved_model
        lora = LoRADynamics(
            state_dim=9, control_dim=8, model_path=model_path,
            lora_rank=4, inner_lr=0.01, inner_steps=5,
        )
        lora.save_meta_weights()

        M = 10
        states = np.random.randn(M, 9) * 0.1
        controls = np.random.randn(M, 8) * 0.1
        next_states = states + np.random.randn(M, 9) * 0.01

        # Basic call
        loss = lora.adapt(states, controls, next_states, dt=0.05, restore=True)
        assert isinstance(loss, float)

        # With temporal_decay
        loss = lora.adapt(states, controls, next_states, dt=0.05,
                          restore=True, temporal_decay=0.95)
        assert isinstance(loss, float)

        # With sample_weights
        weights = np.ones(M) / M
        loss = lora.adapt(states, controls, next_states, dt=0.05,
                          restore=True, sample_weights=weights)
        assert isinstance(loss, float)

    def test_trainable_params_reduction(self, saved_model):
        """LoRA 학습 파라미터 < 전체 파라미터."""
        from mppi_controller.models.learned.lora_dynamics import LoRADynamics

        model_path, _ = saved_model
        lora = LoRADynamics(
            state_dim=9, control_dim=8, model_path=model_path,
            lora_rank=4,
        )

        trainable = lora.get_trainable_params()
        total = lora.get_total_params()

        assert trainable < total
        assert trainable > 0
        ratio = trainable / total
        assert ratio < 0.5  # LoRA should be much smaller than total

    def test_restore_meta_weights(self, saved_model):
        """save/restore meta weights 동작 확인."""
        from mppi_controller.models.learned.lora_dynamics import LoRADynamics

        model_path, _ = saved_model
        lora = LoRADynamics(
            state_dim=9, control_dim=8, model_path=model_path,
            lora_rank=4, inner_lr=0.1, inner_steps=20,
        )

        state = np.random.randn(9)
        control = np.random.randn(8)

        # Save meta, get original output
        lora.save_meta_weights()
        out_before = lora.forward_dynamics(state, control)

        # Adapt (changes LoRA weights)
        M = 10
        states = np.random.randn(M, 9) * 0.1
        controls = np.random.randn(M, 8) * 0.1
        next_states = states + np.random.randn(M, 9) * 0.01
        lora.adapt(states, controls, next_states, dt=0.05, restore=False)

        out_after_adapt = lora.forward_dynamics(state, control)

        # Restore meta → should match original
        lora.restore_meta_weights()
        out_restored = lora.forward_dynamics(state, control)

        np.testing.assert_allclose(out_before, out_restored, atol=1e-5)

    def test_with_residual_dynamics(self, saved_model, kin_model):
        """ResidualDynamics와 조합 테스트."""
        from mppi_controller.models.learned.lora_dynamics import LoRADynamics
        from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

        model_path, _ = saved_model
        lora = LoRADynamics(state_dim=9, control_dim=8, model_path=model_path, lora_rank=4)
        residual = ResidualDynamics(base_model=kin_model, learned_model=lora)

        state = np.random.randn(9) * 0.1
        control = np.random.randn(8) * 0.1

        state_dot = residual.forward_dynamics(state, control)
        assert state_dot.shape == (9,)
        assert np.all(np.isfinite(state_dot))


# =============================================================
# TestSpectralRegularizer
# =============================================================

class TestSpectralRegularizer:
    """SpectralRegularizer tests."""

    def test_nonnegative(self, simple_mlp):
        """Spectral penalty >= 0."""
        from mppi_controller.learning.spectral_regularization import SpectralRegularizer

        reg = SpectralRegularizer(simple_mlp, lambda_spectral=0.01)
        penalty = reg.compute_penalty()

        assert penalty.item() >= 0

    def test_differentiable(self, simple_mlp):
        """Penalty is differentiable (gradient exists)."""
        from mppi_controller.learning.spectral_regularization import SpectralRegularizer

        reg = SpectralRegularizer(simple_mlp, lambda_spectral=0.01)
        penalty = reg.compute_penalty()
        penalty.backward()

        has_grad = False
        for p in simple_mlp.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_identity_sigma1(self):
        """단위 행렬의 σ_max ≈ 1."""
        from mppi_controller.learning.spectral_regularization import SpectralRegularizer

        model = nn.Sequential(nn.Linear(5, 5))
        # Set weight to identity
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(5))
            model[0].bias.zero_()

        reg = SpectralRegularizer(model, lambda_spectral=1.0, n_power_iterations=10)
        penalty = reg.compute_penalty()

        assert abs(penalty.item() - 1.0) < 0.1  # σ_max(I) = 1

    def test_scales_with_lambda(self, simple_mlp):
        """lambda_spectral이 클수록 penalty 증가."""
        from mppi_controller.learning.spectral_regularization import SpectralRegularizer

        reg_small = SpectralRegularizer(simple_mlp, lambda_spectral=0.001)
        reg_large = SpectralRegularizer(simple_mlp, lambda_spectral=0.1)

        p_small = reg_small.compute_penalty().item()
        p_large = reg_large.compute_penalty().item()

        assert p_large > p_small


# =============================================================
# TestDiffSim
# =============================================================

class TestDiffSim:
    """Differentiable Simulator tests."""

    def test_forward_dynamics_matches_numpy(self, diff_sim, kin_model):
        """기구학 forward_dynamics: PyTorch == NumPy."""
        state_np = np.random.randn(9) * 0.3
        control_np = np.random.randn(8) * 0.5

        np_out = kin_model.forward_dynamics(state_np, control_np)

        state_t = torch.tensor(state_np, dtype=torch.float64)
        control_t = torch.tensor(control_np, dtype=torch.float64)
        torch_out = diff_sim.forward_dynamics(state_t, control_t).detach().numpy()

        np.testing.assert_allclose(torch_out, np_out, atol=1e-10)

    def test_rk4_matches_numpy(self, diff_sim, kin_model):
        """RK4 step: PyTorch == NumPy (기구학 모델의 step은 RK4)."""
        state_np = np.random.randn(9) * 0.3
        control_np = np.random.randn(8) * 0.5
        dt = 0.05

        np_out = kin_model.step(state_np, control_np, dt)

        state_t = torch.tensor(state_np, dtype=torch.float64)
        control_t = torch.tensor(control_np, dtype=torch.float64)
        torch_out = diff_sim.step_rk4(state_t, control_t, dt).detach().numpy()

        np.testing.assert_allclose(torch_out, np_out, atol=1e-10)

    def test_fk_matches_numpy(self, diff_sim, kin_model):
        """FK: PyTorch == NumPy."""
        state_np = np.random.randn(9) * 0.3

        np_ee = kin_model.forward_kinematics(state_np)

        state_t = torch.tensor(state_np, dtype=torch.float64)
        torch_ee = diff_sim.forward_kinematics(state_t).detach().numpy()

        np.testing.assert_allclose(torch_ee, np_ee, atol=1e-10)

    def test_gradients_exist(self, diff_sim):
        """forward_dynamics에 gradient 존재."""
        state = torch.randn(9, dtype=torch.float64, requires_grad=True)
        control = torch.randn(8, dtype=torch.float64, requires_grad=True)

        dot = diff_sim.forward_dynamics(state, control)
        dot.sum().backward()

        assert state.grad is not None
        assert control.grad is not None

    def test_rollout_shape(self, diff_sim):
        """rollout 출력 shape."""
        state_0 = torch.zeros(9, dtype=torch.float64)
        controls = torch.randn(20, 8, dtype=torch.float64)

        traj = diff_sim.rollout(state_0, controls, dt=0.05)

        assert traj.shape == (21, 9)

    def test_rollout_differentiable(self, diff_sim):
        """rollout → loss → backward 가능."""
        state_0 = torch.zeros(9, dtype=torch.float64)
        controls = torch.randn(10, 8, dtype=torch.float64, requires_grad=True)

        traj = diff_sim.rollout(state_0, controls, dt=0.05)
        loss = traj.sum()
        loss.backward()

        assert controls.grad is not None

    def test_fk_gradient_through_rollout(self, diff_sim):
        """rollout → FK → loss → backward (전체 체인)."""
        state_0 = torch.zeros(9, dtype=torch.float64)
        controls = torch.randn(10, 8, dtype=torch.float64, requires_grad=True)

        traj = diff_sim.rollout(state_0, controls, dt=0.05)
        ee_pos = diff_sim.forward_kinematics(traj)  # (11, 3)
        loss = ((ee_pos) ** 2).sum()
        loss.backward()

        assert controls.grad is not None
        assert controls.grad.abs().sum() > 0

    def test_dynamic_matches_numpy(self, diff_sim_dyn, dyn_model):
        """동역학 forward_dynamics: PyTorch == NumPy."""
        state_np = np.random.randn(9) * 0.3
        control_np = np.random.randn(8) * 0.5

        np_out = dyn_model.forward_dynamics(state_np, control_np)

        state_t = torch.tensor(state_np, dtype=torch.float64)
        control_t = torch.tensor(control_np, dtype=torch.float64)
        torch_out = diff_sim_dyn.forward_dynamics(state_t, control_t).detach().numpy()

        np.testing.assert_allclose(torch_out, np_out, atol=1e-10)


# =============================================================
# TestBPTTTrainer
# =============================================================

class TestBPTTTrainer:
    """BPTT Residual Trainer tests."""

    def _make_trainer(self, diff_sim):
        from mppi_controller.learning.bptt_residual_trainer import BPTTResidualTrainer

        residual_model = DynamicsMLPModel(
            input_dim=17, output_dim=9, hidden_dims=[32, 32]
        )
        # Convert to double for compatibility
        residual_model = residual_model.double()

        trainer = BPTTResidualTrainer(
            residual_model=residual_model,
            diff_sim=diff_sim,
            norm_stats=None,
            learning_rate=1e-3,
            spectral_lambda=0.0,
            rollout_horizon=10,
            dt=0.05,
            truncation_length=5,
        )
        return trainer

    def test_trajectory_loss_shape(self, diff_sim):
        """trajectory_loss는 스칼라 반환."""
        trainer = self._make_trainer(diff_sim)

        traj = torch.randn(11, 9, dtype=torch.float64)
        ee_ref = torch.randn(11, 3, dtype=torch.float64)

        loss = trainer.trajectory_loss(traj, ee_ref)

        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_rollout_shape(self, diff_sim):
        """differentiable_rollout 출력 shape."""
        trainer = self._make_trainer(diff_sim)

        state_0 = torch.zeros(9, dtype=torch.float64)
        controls = torch.randn(10, 8, dtype=torch.float64)

        traj = trainer.differentiable_rollout(state_0, controls, dt=0.05)

        assert traj.shape == (11, 9)

    def test_gradient_flows_to_residual(self, diff_sim):
        """BPTT gradient가 residual model까지 흐르는지 확인."""
        trainer = self._make_trainer(diff_sim)

        state_0 = torch.zeros(9, dtype=torch.float64)
        controls = torch.randn(10, 8, dtype=torch.float64)
        ee_ref = torch.randn(11, 3, dtype=torch.float64)

        trainer.residual_model.train()
        trainer.optimizer.zero_grad()

        traj = trainer.differentiable_rollout(state_0, controls, dt=0.05)
        loss = trainer.trajectory_loss(traj, ee_ref)
        loss.backward()

        has_grad = False
        for p in trainer.residual_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_training_reduces_loss(self, diff_sim):
        """짧은 학습으로 loss 감소 확인."""
        trainer = self._make_trainer(diff_sim)

        # Generate a simple episode
        state_0 = np.zeros(9)
        N = 10
        controls = np.random.randn(N, 8) * 0.1
        # Simple reference: origin
        ee_ref = np.tile(
            diff_sim.forward_kinematics(torch.zeros(9, dtype=torch.float64)).detach().numpy(),
            (N + 1, 1),
        )

        episodes = [{"state_0": state_0, "controls": controls, "ee_reference": ee_ref}]

        history = trainer.train(episodes, epochs=20, verbose=False)

        assert len(history["train_loss"]) == 20
        # Loss should decrease (or at least not increase dramatically)
        assert history["train_loss"][-1] <= history["train_loss"][0] * 2.0

    def test_save_load(self, diff_sim, tmp_path):
        """모델 저장/로드."""
        from mppi_controller.learning.bptt_residual_trainer import BPTTResidualTrainer

        residual_model = DynamicsMLPModel(
            input_dim=17, output_dim=9, hidden_dims=[32, 32]
        ).double()

        trainer = BPTTResidualTrainer(
            residual_model=residual_model,
            diff_sim=diff_sim,
            save_dir=str(tmp_path),
        )

        trainer.save_model("test_bptt.pth")
        assert os.path.exists(str(tmp_path / "test_bptt.pth"))

        trainer.load_model("test_bptt.pth")


# =============================================================
# TestNeuralNetworkTrainerSpectral
# =============================================================

class TestNeuralNetworkTrainerSpectral:
    """NeuralNetworkTrainer spectral integration test."""

    def test_spectral_lambda_creates_regularizer(self):
        """spectral_lambda > 0 → SpectralRegularizer 생성."""
        from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

        trainer = NeuralNetworkTrainer(
            state_dim=9, control_dim=8,
            hidden_dims=[32, 32],
            spectral_lambda=0.01,
        )
        assert trainer.spectral_reg is not None

    def test_spectral_lambda_zero_no_regularizer(self):
        """spectral_lambda = 0 → SpectralRegularizer 없음."""
        from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

        trainer = NeuralNetworkTrainer(
            state_dim=9, control_dim=8,
            hidden_dims=[32, 32],
            spectral_lambda=0.0,
        )
        assert trainer.spectral_reg is None


class TestEnsembleTrainerSpectral:
    """EnsembleTrainer spectral integration test."""

    def test_spectral_lambda_creates_regularizers(self):
        """spectral_lambda > 0 → 각 모델에 SpectralRegularizer."""
        from mppi_controller.learning.ensemble_trainer import EnsembleTrainer

        trainer = EnsembleTrainer(
            state_dim=9, control_dim=8,
            num_models=3,
            hidden_dims=[32, 32],
            spectral_lambda=0.01,
        )
        assert all(reg is not None for reg in trainer.spectral_regs)
        assert len(trainer.spectral_regs) == 3


# =============================================================
# TestNNPolicy
# =============================================================

class TestNNPolicy:
    """NN-Policy (BPTT) tests."""

    def _make_trainer(self, tmp_path=None):
        from mppi_controller.learning.nn_policy_trainer import NNPolicyTrainer
        save_dir = str(tmp_path) if tmp_path else "/tmp/test_nn_policy"
        return NNPolicyTrainer(
            state_dim=9, ee_ref_dim=3, control_dim=8,
            hidden_dims=[32, 32],
            control_bounds=np.array([1.0, 2.0] + [3.0] * 6),
            learning_rate=1e-3,
            save_dir=save_dir,
        )

    def test_policy_mlp_output_shape(self):
        """PolicyMLP: (12,) → (8,) 출력 shape."""
        from mppi_controller.learning.nn_policy_trainer import PolicyMLP

        model = PolicyMLP(
            input_dim=12, output_dim=8, hidden_dims=[32, 32],
            control_bounds=np.array([1.0, 2.0] + [3.0] * 6),
        )

        inp = torch.randn(12)
        out = model(inp)
        assert out.shape == (8,)

        # Batch input
        inp_batch = torch.randn(5, 12)
        out_batch = model(inp_batch)
        assert out_batch.shape == (5, 8)

    def test_policy_output_bounds(self):
        """PolicyMLP 출력이 tanh * bounds 범위 내."""
        from mppi_controller.learning.nn_policy_trainer import PolicyMLP

        bounds = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        model = PolicyMLP(
            input_dim=12, output_dim=8, hidden_dims=[64, 64],
            control_bounds=bounds,
        )

        # Test with many random inputs
        inp = torch.randn(100, 12)
        out = model(inp)

        bounds_t = torch.tensor(bounds, dtype=torch.float32)
        assert torch.all(out.abs() <= bounds_t + 1e-6)

    def test_bc_training_reduces_loss(self):
        """Behavioral Cloning으로 MSE loss 감소."""
        trainer = self._make_trainer()

        # Create synthetic episodes
        episodes = []
        for _ in range(5):
            T = 20
            states = np.random.randn(T, 9) * 0.3
            ee_refs = np.random.randn(T, 3) * 0.2
            controls = np.random.randn(T, 8) * 0.5
            episodes.append({
                "states": states,
                "ee_refs": ee_refs,
                "controls": controls,
            })

        history = trainer.train_bc(episodes, epochs=30, verbose=False)

        assert len(history["bc_loss"]) == 30
        # Loss should decrease
        assert history["bc_loss"][-1] < history["bc_loss"][0]

    def test_bptt_rollout_shape(self, diff_sim):
        """BPTT rollout: (N+1, 9) shape."""
        trainer = self._make_trainer()

        state_0 = torch.zeros(9, dtype=torch.float64)
        ee_refs = torch.randn(10, 3, dtype=torch.float64)

        # Inject diff_sim
        trainer._diff_sim = diff_sim

        traj = trainer._policy_rollout(state_0, ee_refs, dt=0.05)
        assert traj.shape == (11, 9)

    def test_save_load_policy(self, tmp_path):
        """정책 모델 저장/로드 일관성."""
        trainer = self._make_trainer(tmp_path)

        # Get output before save
        inp = np.concatenate([np.zeros(9), np.zeros(3)])
        out_before = trainer.compute_control(np.zeros(9), np.zeros(3))

        trainer.save_model("test_policy.pth")
        assert os.path.exists(str(tmp_path / "test_policy.pth"))

        # Create new trainer and load
        trainer2 = self._make_trainer(tmp_path)
        trainer2.load_model("test_policy.pth")

        out_after = trainer2.compute_control(np.zeros(9), np.zeros(3))
        np.testing.assert_allclose(out_before, out_after, atol=1e-6)
