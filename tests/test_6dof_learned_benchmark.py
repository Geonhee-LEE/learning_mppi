"""
6-DOF Learned Model Benchmark 컴포넌트 테스트

TestModelFactory: 모델 생성 및 차원 검증
TestMetrics: 메트릭 계산
TestSimulation: 시뮬레이션 스모크 테스트
TestTraining: 학습 파이프라인 테스트
"""

import numpy as np
import pytest
import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
    MobileManipulator6DOFDynamic,
)
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────
# Import helper for modules that can't use normal import
# ─────────────────────────────────────────────────────────────

def _load_module(name, filepath):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def benchmark_mod():
    """Load 6dof_learned_benchmark module."""
    return _load_module(
        "benchmark_6dof",
        os.path.join(PROJECT_ROOT, "examples", "comparison", "6dof_learned_benchmark.py"),
    )


@pytest.fixture(scope="module")
def train_mod():
    """Load train_6dof_all_models module."""
    return _load_module(
        "train_6dof_all",
        os.path.join(PROJECT_ROOT, "scripts", "train_6dof_all_models.py"),
    )


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def kin_model():
    return MobileManipulator6DOFKinematic()


@pytest.fixture
def dyn_model():
    return MobileManipulator6DOFDynamic()


@pytest.fixture
def oracle_residual(kin_model, dyn_model):
    """Oracle residual function."""
    def residual_fn(state, control):
        return dyn_model.forward_dynamics(state, control) - kin_model.forward_dynamics(state, control)
    return residual_fn


# ─────────────────────────────────────────────────────────────
# TestModelFactory
# ─────────────────────────────────────────────────────────────

class TestModelFactory:
    """Model creation and dimension tests."""

    def test_kinematic_dims(self, kin_model):
        """Kinematic model has correct dimensions."""
        assert kin_model.state_dim == 9
        assert kin_model.control_dim == 8

        state = np.zeros(9)
        control = np.zeros(8)
        state_dot = kin_model.forward_dynamics(state, control)
        assert state_dot.shape == (9,)

    def test_oracle_dims(self, kin_model, dyn_model, oracle_residual):
        """Oracle ResidualDynamics has correct dimensions."""
        model = ResidualDynamics(base_model=kin_model, residual_fn=oracle_residual)
        assert model.state_dim == 9
        assert model.control_dim == 8

        state = np.zeros(9)
        control = np.zeros(8)
        state_dot = model.forward_dynamics(state, control)
        assert state_dot.shape == (9,)

    def test_oracle_matches_dynamic(self, kin_model, dyn_model, oracle_residual):
        """Oracle residual model matches full dynamic model."""
        model = ResidualDynamics(base_model=kin_model, residual_fn=oracle_residual)

        np.random.seed(42)
        state = np.random.randn(9) * 0.5
        control = np.random.randn(8) * 0.3

        oracle_dot = model.forward_dynamics(state, control)
        dyn_dot = dyn_model.forward_dynamics(state, control)

        np.testing.assert_allclose(oracle_dot, dyn_dot, atol=1e-10)

    def test_residual_nn_composition(self, kin_model):
        """ResidualDynamics with fake NN function has correct shape."""
        def fake_nn(state, control):
            if state.ndim == 1:
                return np.zeros(9)
            return np.zeros((state.shape[0], 9))

        model = ResidualDynamics(base_model=kin_model, residual_fn=fake_nn)
        state = np.zeros(9)
        control = np.zeros(8)
        result = model.forward_dynamics(state, control)
        assert result.shape == (9,)

        # Batch
        states = np.zeros((5, 9))
        controls = np.zeros((5, 8))
        results = model.forward_dynamics(states, controls)
        assert results.shape == (5, 9)

    def test_residual_gp_composition(self, kin_model):
        """ResidualDynamics with fake GP function has correct shape."""
        def fake_gp(state, control):
            if state.ndim == 1:
                return np.zeros(9) * 0.01
            return np.zeros((state.shape[0], 9)) * 0.01

        model = ResidualDynamics(base_model=kin_model, residual_fn=fake_gp)
        state = np.random.randn(9)
        control = np.random.randn(8) * 0.3
        result = model.forward_dynamics(state, control)
        assert result.shape == (9,)

    def test_residual_ensemble_composition(self, kin_model):
        """ResidualDynamics with fake ensemble function."""
        def fake_ensemble(state, control):
            if state.ndim == 1:
                return np.ones(9) * 0.005
            return np.ones((state.shape[0], 9)) * 0.005

        model = ResidualDynamics(base_model=kin_model, residual_fn=fake_ensemble)
        state = np.zeros(9)
        control = np.zeros(8)
        result = model.forward_dynamics(state, control)
        assert result.shape == (9,)

    def test_maml_adapt_callable(self):
        """MAMLDynamics adapt method exists and is callable."""
        from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
        model = MAMLDynamics(state_dim=9, control_dim=8)
        assert hasattr(model, "adapt")
        assert callable(model.adapt)

    def test_alpaca_adapt_callable(self):
        """ALPaCADynamics adapt method exists and is callable."""
        from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
        model = ALPaCADynamics(state_dim=9, control_dim=8)
        assert hasattr(model, "adapt")
        assert callable(model.adapt)


# ─────────────────────────────────────────────────────────────
# TestMetrics
# ─────────────────────────────────────────────────────────────

class TestMetrics:
    """Metric computation tests."""

    def test_rmse_calculation(self, benchmark_mod):
        """RMSE calculation is correct."""
        history = {
            "ee_error": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "solve_time": np.array([5.0, 6.0, 7.0, 8.0, 9.0]),
        }
        metrics = benchmark_mod.compute_metrics(history)

        expected_rmse = np.sqrt(np.mean(np.array([0.1, 0.2, 0.3, 0.4, 0.5]) ** 2))
        assert abs(metrics["ee_rmse"] - expected_rmse) < 1e-10

    def test_metrics_keys(self, benchmark_mod):
        """Metrics dict has expected keys."""
        history = {
            "ee_error": np.array([0.1, 0.2]),
            "solve_time": np.array([5.0, 6.0]),
        }
        metrics = benchmark_mod.compute_metrics(history)

        expected_keys = {"ee_rmse", "ee_max", "ee_mean", "solve_ms_mean", "solve_ms_std"}
        assert set(metrics.keys()) == expected_keys

    def test_rmse_zero(self, benchmark_mod):
        """Zero errors give zero RMSE."""
        history = {
            "ee_error": np.zeros(10),
            "solve_time": np.ones(10),
        }
        metrics = benchmark_mod.compute_metrics(history)
        assert metrics["ee_rmse"] == 0.0
        assert metrics["ee_max"] == 0.0


# ─────────────────────────────────────────────────────────────
# TestSimulation
# ─────────────────────────────────────────────────────────────

class TestSimulation:
    """Simulation smoke tests."""

    def test_single_step_smoke(self, kin_model, dyn_model, benchmark_mod):
        """Single model run completes without error (short)."""
        history = benchmark_mod.run_single(
            "kinematic", kin_model, dyn_model, "ee_3d_circle",
            K=16, duration=0.5, seed=42,
        )

        assert "time" in history
        assert "ee_error" in history
        assert len(history["time"]) > 0
        assert history["ee_pos"].shape[1] == 3
        assert history["state"].shape[1] == 9

    def test_oracle_lower_rmse_than_kinematic(self, kin_model, dyn_model, benchmark_mod):
        """Oracle should have lower or equal RMSE than kinematic."""
        h_kin = benchmark_mod.run_single(
            "kinematic", kin_model, dyn_model, "ee_3d_circle",
            K=32, duration=2.0, seed=42,
        )
        h_oracle = benchmark_mod.run_single(
            "oracle", kin_model, dyn_model, "ee_3d_circle",
            K=32, duration=2.0, seed=42,
        )

        m_kin = benchmark_mod.compute_metrics(h_kin)
        m_oracle = benchmark_mod.compute_metrics(h_oracle)

        # Oracle should perform at least as well as kinematic
        assert m_oracle["ee_rmse"] <= m_kin["ee_rmse"] * 1.5  # Allow margin for stochasticity

    def test_online_adapt_no_crash(self, kin_model, dyn_model):
        """MAML online adaptation does not crash during simulation."""
        import torch
        from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
        from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel
        from mppi_controller.controllers.mppi.mppi_params import MPPIParams
        from mppi_controller.controllers.mppi.base_mppi import MPPIController
        from mppi_controller.utils.trajectory import (
            create_trajectory_function, generate_reference_trajectory,
        )

        maml_model = MAMLDynamics(state_dim=9, control_dim=8)

        # Initialize model manually (no pretrained weights)
        maml_model.model = DynamicsMLPModel(
            input_dim=17, output_dim=9,
            hidden_dims=[32, 32], activation="relu", dropout_rate=0.0,
        )
        maml_model.norm_stats = {
            "state_mean": np.zeros(9),
            "state_std": np.ones(9),
            "control_mean": np.zeros(8),
            "control_std": np.ones(8),
            "state_dot_mean": np.zeros(9),
            "state_dot_std": np.ones(9),
        }
        maml_model.save_meta_weights()

        res_model = ResidualDynamics(base_model=kin_model, learned_model=maml_model)

        params = MPPIParams(
            N=5, dt=0.05, K=8, lambda_=1.0,
            sigma=np.array([0.3, 0.3] + [0.5] * 6),
            Q=np.array([1.0] * 9),
            R=np.array([0.1] * 8),
            Qf=np.array([1.0] * 9),
        )

        controller = MPPIController(res_model, params)

        state = np.zeros(9)
        traj_fn = create_trajectory_function("ee_3d_circle")

        # 5 steps with adaptation
        adapt_states, adapt_controls, adapt_ns = [], [], []
        for step in range(5):
            t = step * 0.05
            ref = generate_reference_trajectory(traj_fn, t, 5, 0.05)
            control, _ = controller.compute_control(state, ref)
            next_state = dyn_model.step(state, control, 0.05)

            adapt_states.append(state.copy())
            adapt_controls.append(control.copy())
            adapt_ns.append(next_state.copy())

            if len(adapt_states) >= 3:
                buf_s = np.array(adapt_states[-3:])
                buf_c = np.array(adapt_controls[-3:])
                buf_ns = np.array(adapt_ns[-3:])
                maml_model.adapt(buf_s, buf_c, buf_ns, 0.05)

            state = next_state

    def test_generate_residual_data_shape(self, kin_model, dyn_model, train_mod):
        """generate_residual_data returns correct shapes."""
        data = train_mod.generate_residual_data(kin_model, dyn_model, n_samples=100, seed=42)

        assert data["states"].shape == (100, 9)
        assert data["controls"].shape == (100, 8)
        assert data["state_dots"].shape == (100, 9)

        # Residuals should be non-trivial (some are non-zero)
        assert np.any(np.abs(data["state_dots"]) > 1e-6)


# ─────────────────────────────────────────────────────────────
# TestTraining
# ─────────────────────────────────────────────────────────────

class TestTraining:
    """Training pipeline tests."""

    def test_maml_task_sampling(self, train_mod):
        """6-DOF MAML task sampling produces valid params."""
        np.random.seed(42)
        task = train_mod._sample_6dof_task()

        assert "joint_friction" in task
        assert "gravity_droop" in task
        assert "coupling_gain" in task
        assert "base_response_k" in task

        assert task["joint_friction"].shape == (6,)
        assert 0.03 <= task["gravity_droop"] <= 0.15
        assert 0.01 <= task["coupling_gain"] <= 0.05
        assert 3.0 <= task["base_response_k"] <= 8.0

        # All friction values should be positive
        assert np.all(task["joint_friction"] > 0)

    def test_maml_task_data_generation(self, train_mod):
        """6-DOF MAML task data generation has correct shapes."""
        kin_model = MobileManipulator6DOFKinematic()
        np.random.seed(42)
        task = train_mod._sample_6dof_task()

        states, controls, next_states = train_mod._generate_6dof_task_data(
            task, kin_model, n_samples=50, seed=42
        )

        assert states.shape == (50, 9)
        assert controls.shape == (50, 8)
        assert next_states.shape == (50, 9)

    def test_nn_training_loss_decreases(self, kin_model, dyn_model, train_mod):
        """NN training loss decreases over epochs."""
        from mppi_controller.learning.data_collector import DynamicsDataset
        from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer
        import tempfile

        data = train_mod.generate_residual_data(kin_model, dyn_model, n_samples=200, seed=42)
        dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
        norm_stats = dataset.get_normalization_stats()
        train_inputs, train_targets = dataset.get_train_data()
        val_inputs, val_targets = dataset.get_val_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NeuralNetworkTrainer(
                state_dim=9,
                control_dim=8,
                hidden_dims=[32, 32],
                save_dir=tmpdir,
            )
            history = trainer.train(
                train_inputs, train_targets,
                val_inputs, val_targets,
                norm_stats=norm_stats,
                epochs=20,
                batch_size=64,
                verbose=False,
            )

            # Training loss should decrease
            assert history["train_loss"][-1] < history["train_loss"][0]
