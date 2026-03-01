#!/usr/bin/env python3
"""
6-DOF Mobile Manipulator 전체 학습 모델 통합 학습 스크립트

6종 학습 모델(NN, GP, Ensemble, MC-Dropout, MAML, ALPaCA)을
6-DOF 모바일 매니퓰레이터의 residual dynamics에 대해 학습.

단계:
  1. Kinematic/Dynamic 모델에서 residual 데이터 생성
  2. 각 모델 타입별 학습
  3. 모델 저장

Usage:
    PYTHONPATH=. python scripts/train_6dof_all_models.py --samples 10000 --epochs 200
    PYTHONPATH=. python scripts/train_6dof_all_models.py --quick
    PYTHONPATH=. python scripts/train_6dof_all_models.py --models nn,gp
"""

import numpy as np
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
    MobileManipulator6DOFDynamic,
)
from mppi_controller.learning.data_collector import DynamicsDataset


# ─────────────────────────────────────────────────────────────
# Data generation (reused from train_6dof_residual.py pattern)
# ─────────────────────────────────────────────────────────────

def generate_residual_data(kin_model, dyn_model, n_samples, seed=42):
    """
    랜덤 (state, control) 쌍에서 residual 데이터 생성.

    Returns:
        data: {states, controls, state_dots} — residual = dynamic - kinematic
    """
    np.random.seed(seed)

    states = np.zeros((n_samples, 9))
    controls = np.zeros((n_samples, 8))

    # Base: x, y ~ U(-2, 2), θ ~ U(-π, π)
    states[:, 0] = np.random.uniform(-2.0, 2.0, n_samples)
    states[:, 1] = np.random.uniform(-2.0, 2.0, n_samples)
    states[:, 2] = np.random.uniform(-np.pi, np.pi, n_samples)

    # Joints: q1..q6 ~ U(-π, π)
    for i in range(6):
        states[:, 3 + i] = np.random.uniform(-np.pi, np.pi, n_samples)

    # Controls: uniform within bounds
    bounds = kin_model.get_control_bounds()
    if bounds is not None:
        lower, upper = bounds
        for i in range(8):
            controls[:, i] = np.random.uniform(lower[i], upper[i], n_samples)
    else:
        controls[:, 0] = np.random.uniform(-1.0, 1.0, n_samples)
        controls[:, 1] = np.random.uniform(-1.0, 1.0, n_samples)
        for i in range(6):
            controls[:, 2 + i] = np.random.uniform(-2.0, 2.0, n_samples)

    # Residual 계산 (배치 연산)
    kin_dots = kin_model.forward_dynamics(states, controls)
    dyn_dots = dyn_model.forward_dynamics(states, controls)
    residuals = dyn_dots - kin_dots

    print(f"  Residual stats:")
    print(f"    Mean abs: {np.mean(np.abs(residuals), axis=0)}")
    print(f"    Max abs:  {np.max(np.abs(residuals), axis=0)}")
    print(f"    Norm mean: {np.mean(np.linalg.norm(residuals, axis=1)):.4f}")

    return {
        "states": states,
        "controls": controls,
        "state_dots": residuals,
    }


# ─────────────────────────────────────────────────────────────
# MAML/ALPaCA 서브클래스 — 6-DOF 태스크 분포
# ─────────────────────────────────────────────────────────────

def _generate_6dof_task_data(task_params, kin_model, n_samples, dt=0.05, seed=None):
    """
    6-DOF 매니퓰레이터용 태스크 데이터 생성.

    task_params로 커스텀 동역학 모델 생성 → residual 계산.

    Returns:
        states: (n_samples, 9)
        controls: (n_samples, 8)
        next_states: (n_samples, 9) — kin + residual * dt
    """
    if seed is not None:
        np.random.seed(seed)

    dyn = MobileManipulator6DOFDynamic(
        joint_friction=task_params["joint_friction"],
        gravity_droop=task_params["gravity_droop"],
        coupling_gain=task_params["coupling_gain"],
        base_response_k=task_params["base_response_k"],
    )

    states = np.zeros((n_samples, 9))
    controls = np.zeros((n_samples, 8))

    # Random states
    states[:, 0] = np.random.uniform(-2.0, 2.0, n_samples)
    states[:, 1] = np.random.uniform(-2.0, 2.0, n_samples)
    states[:, 2] = np.random.uniform(-np.pi, np.pi, n_samples)
    for i in range(6):
        states[:, 3 + i] = np.random.uniform(-np.pi, np.pi, n_samples)

    # Random controls within bounds
    bounds = kin_model.get_control_bounds()
    if bounds is not None:
        lower, upper = bounds
        for i in range(8):
            controls[:, i] = np.random.uniform(lower[i], upper[i], n_samples)
    else:
        controls[:, 0] = np.random.uniform(-1.0, 1.0, n_samples)
        controls[:, 1] = np.random.uniform(-1.0, 1.0, n_samples)
        for i in range(6):
            controls[:, 2 + i] = np.random.uniform(-2.0, 2.0, n_samples)

    # Compute residual-based next_states
    kin_dots = kin_model.forward_dynamics(states, controls)
    dyn_dots = dyn.forward_dynamics(states, controls)
    residual_dots = dyn_dots - kin_dots

    # next_states = states + residual_dot * dt  (MAML learns residual)
    next_states = states + residual_dots * dt

    return states, controls, next_states


def _sample_6dof_task():
    """6-DOF 매니퓰레이터 태스크 파라미터 샘플링."""
    default_friction = np.array([0.15, 0.12, 0.10, 0.08, 0.06, 0.05])
    return {
        "joint_friction": default_friction * np.random.uniform(0.5, 2.0, 6),
        "gravity_droop": np.random.uniform(0.03, 0.15),
        "coupling_gain": np.random.uniform(0.01, 0.05),
        "base_response_k": np.random.uniform(3.0, 8.0),
    }


class Manipulator6DOFMAMLTrainer:
    """MAML trainer with 6-DOF manipulator task distribution."""

    def __init__(self, state_dim=9, control_dim=8, save_dir="models/learned_models/6dof_benchmark/maml",
                 **kwargs):
        from mppi_controller.learning.maml_trainer import MAMLTrainer

        self._kin_model = MobileManipulator6DOFKinematic()
        self._trainer = MAMLTrainer(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dims=kwargs.get("hidden_dims", [128, 128, 64]),
            inner_lr=kwargs.get("inner_lr", 0.01),
            inner_steps=kwargs.get("inner_steps", 5),
            meta_lr=kwargs.get("meta_lr", 1e-3),
            task_batch_size=kwargs.get("task_batch_size", 4),
            support_size=kwargs.get("support_size", 50),
            query_size=kwargs.get("query_size", 50),
            save_dir=save_dir,
        )
        # Override task generation methods
        self._trainer._sample_task = _sample_6dof_task
        self._trainer._generate_task_data = self._generate_task_data

    def _generate_task_data(self, task_params, n_samples):
        return _generate_6dof_task_data(task_params, self._kin_model, n_samples)

    def meta_train(self, n_iterations=1000, verbose=True):
        self._trainer.meta_train(n_iterations=n_iterations, verbose=verbose)

    def save_meta_model(self, filename="maml_meta_model.pth"):
        self._trainer.save_meta_model(filename)

    @property
    def trainer(self):
        return self._trainer


class Manipulator6DOFALPaCATrainer:
    """ALPaCA trainer with 6-DOF manipulator task distribution."""

    def __init__(self, state_dim=9, control_dim=8,
                 save_dir="models/learned_models/6dof_benchmark/alpaca", **kwargs):
        from mppi_controller.learning.alpaca_trainer import ALPaCATrainer

        self._kin_model = MobileManipulator6DOFKinematic()
        self._trainer = ALPaCATrainer(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dims=kwargs.get("hidden_dims", [128, 128]),
            feature_dim=kwargs.get("feature_dim", 64),
            meta_lr=kwargs.get("meta_lr", 1e-3),
            task_batch_size=kwargs.get("task_batch_size", 4),
            support_size=kwargs.get("support_size", 50),
            query_size=kwargs.get("query_size", 50),
            save_dir=save_dir,
        )
        # Override task generation methods
        self._trainer._sample_task = _sample_6dof_task
        self._trainer._generate_task_data = self._generate_task_data

    def _generate_task_data(self, task_params, n_samples):
        return _generate_6dof_task_data(task_params, self._kin_model, n_samples)

    def meta_train(self, n_iterations=1000, verbose=True):
        self._trainer.meta_train(n_iterations=n_iterations, verbose=verbose)

    def save_meta_model(self, filename="alpaca_meta_model.pth"):
        self._trainer.save_meta_model(filename)

    @property
    def trainer(self):
        return self._trainer


# ─────────────────────────────────────────────────────────────
# Training functions for each model type
# ─────────────────────────────────────────────────────────────

SAVE_DIR = "models/learned_models/6dof_benchmark"


def train_nn(dataset, norm_stats, epochs, batch_size, save_dir, verbose=True):
    """Train Neural Network residual model."""
    from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

    print("\n" + "─" * 50)
    print("  [NN] Neural Network Training")
    print("─" * 50)

    trainer = NeuralNetworkTrainer(
        state_dim=9,
        control_dim=8,
        hidden_dims=[128, 128, 64],
        activation="relu",
        dropout_rate=0.05,
        learning_rate=1e-3,
        weight_decay=1e-5,
        save_dir=os.path.join(save_dir, "nn"),
    )

    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()

    t0 = time.time()
    history = trainer.train(
        train_inputs, train_targets,
        val_inputs, val_targets,
        norm_stats=norm_stats,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=30,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    trainer.save_model("best_model.pth")
    print(f"  [NN] Final val loss: {history['val_loss'][-1]:.6f} ({elapsed:.1f}s)")
    return history


def train_gp(dataset, norm_stats, num_iterations, save_dir, verbose=True):
    """Train Gaussian Process residual model."""
    from mppi_controller.learning.gaussian_process_trainer import GaussianProcessTrainer

    print("\n" + "─" * 50)
    print("  [GP] Gaussian Process Training")
    print("─" * 50)

    trainer = GaussianProcessTrainer(
        state_dim=9,
        control_dim=8,
        kernel_type="rbf",
        use_sparse=True,
        num_inducing_points=200,
        use_ard=True,
        save_dir=os.path.join(save_dir, "gp"),
    )

    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()

    t0 = time.time()
    history = trainer.train(
        train_inputs, train_targets,
        val_inputs, val_targets,
        norm_stats=norm_stats,
        num_iterations=num_iterations,
        learning_rate=0.1,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    trainer.save_model("gp_model.pth")
    print(f"  [GP] Training complete ({elapsed:.1f}s)")
    return history


def train_ensemble(dataset, norm_stats, epochs, batch_size, save_dir, verbose=True):
    """Train Ensemble Neural Network residual model."""
    from mppi_controller.learning.ensemble_trainer import EnsembleTrainer

    print("\n" + "─" * 50)
    print("  [Ensemble] Ensemble NN Training (5 models)")
    print("─" * 50)

    trainer = EnsembleTrainer(
        state_dim=9,
        control_dim=8,
        num_models=5,
        hidden_dims=[128, 128, 64],
        activation="relu",
        dropout_rate=0.0,
        learning_rate=1e-3,
        weight_decay=1e-5,
        save_dir=os.path.join(save_dir, "ensemble"),
    )

    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()

    t0 = time.time()
    history = trainer.train(
        train_inputs, train_targets,
        val_inputs, val_targets,
        norm_stats=norm_stats,
        epochs=epochs,
        batch_size=batch_size,
        bootstrap=True,
        early_stopping_patience=30,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    trainer.save_model("ensemble.pth")
    print(f"  [Ensemble] Final val loss: {history['val_loss'][-1]:.6f} ({elapsed:.1f}s)")
    return history


def train_mcdropout(dataset, norm_stats, epochs, batch_size, save_dir, verbose=True):
    """Train MC-Dropout Neural Network residual model."""
    from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

    print("\n" + "─" * 50)
    print("  [MCDropout] MC-Dropout Training (dropout=0.1)")
    print("─" * 50)

    trainer = NeuralNetworkTrainer(
        state_dim=9,
        control_dim=8,
        hidden_dims=[128, 128, 64],
        activation="relu",
        dropout_rate=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        save_dir=os.path.join(save_dir, "mcdropout"),
    )

    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()

    t0 = time.time()
    history = trainer.train(
        train_inputs, train_targets,
        val_inputs, val_targets,
        norm_stats=norm_stats,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=30,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    trainer.save_model("best_model.pth")
    print(f"  [MCDropout] Final val loss: {history['val_loss'][-1]:.6f} ({elapsed:.1f}s)")
    return history


def train_maml(n_iterations, save_dir, verbose=True):
    """Train MAML meta-model for 6-DOF residual."""
    print("\n" + "─" * 50)
    print("  [MAML] Meta-Learning Training")
    print("─" * 50)

    maml_trainer = Manipulator6DOFMAMLTrainer(
        save_dir=os.path.join(save_dir, "maml"),
    )

    t0 = time.time()
    maml_trainer.meta_train(n_iterations=n_iterations, verbose=verbose)
    elapsed = time.time() - t0

    maml_trainer.save_meta_model("maml_meta_model.pth")
    print(f"  [MAML] Meta-training complete ({elapsed:.1f}s)")
    return maml_trainer


def train_alpaca(n_iterations, save_dir, verbose=True):
    """Train ALPaCA meta-model for 6-DOF residual."""
    print("\n" + "─" * 50)
    print("  [ALPaCA] Meta-Learning Training")
    print("─" * 50)

    alpaca_trainer = Manipulator6DOFALPaCATrainer(
        save_dir=os.path.join(save_dir, "alpaca"),
    )

    t0 = time.time()
    alpaca_trainer.meta_train(n_iterations=n_iterations, verbose=verbose)
    elapsed = time.time() - t0

    alpaca_trainer.save_meta_model("alpaca_meta_model.pth")
    print(f"  [ALPaCA] Meta-training complete ({elapsed:.1f}s)")
    return alpaca_trainer


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

ALL_MODELS = ["nn", "gp", "ensemble", "mcdropout", "maml", "alpaca"]


def main():
    parser = argparse.ArgumentParser(
        description="Train all 6-DOF learned models for benchmark"
    )
    parser.add_argument("--samples", type=int, default=10000,
                        help="Number of residual data samples")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs (NN/Ensemble/MCDropout)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gp-iterations", type=int, default=100,
                        help="GP optimization iterations")
    parser.add_argument("--meta-iterations", type=int, default=500,
                        help="MAML/ALPaCA meta-training iterations")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model list (e.g., nn,gp,maml)")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (reduced samples/epochs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.samples = 2000
        args.epochs = 30
        args.gp_iterations = 30
        args.meta_iterations = 100

    # Parse model list
    if args.models:
        models_to_train = [m.strip() for m in args.models.split(",")]
        for m in models_to_train:
            if m not in ALL_MODELS:
                print(f"Unknown model: {m}. Available: {ALL_MODELS}")
                sys.exit(1)
    else:
        models_to_train = ALL_MODELS

    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("6-DOF All Models Training".center(60))
    print("=" * 60)
    print(f"  Models:          {models_to_train}")
    print(f"  Samples:         {args.samples}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  GP iterations:   {args.gp_iterations}")
    print(f"  Meta iterations: {args.meta_iterations}")
    print(f"  Save dir:        {args.save_dir}")
    print(f"  Quick mode:      {args.quick}")
    print("=" * 60)

    # 1. 모델 생성 & 데이터 생성 (NN/GP/Ensemble/MCDropout에 공유)
    offline_models = {"nn", "gp", "ensemble", "mcdropout"}
    needs_data = bool(offline_models & set(models_to_train))

    dataset = None
    norm_stats = None

    if needs_data:
        print("\n[1/2] Generating residual data...")
        kin_model = MobileManipulator6DOFKinematic()
        dyn_model = MobileManipulator6DOFDynamic()
        data = generate_residual_data(kin_model, dyn_model, args.samples, seed=args.seed)

        dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
        norm_stats = dataset.get_normalization_stats()

        train_inputs, train_targets = dataset.get_train_data()
        print(f"  Train: {train_inputs.shape[0]} samples")
        print(f"  Val:   {dataset.get_val_data()[0].shape[0]} samples")
        print(f"  Input dim:  {train_inputs.shape[1]} (state=9 + control=8)")
        print(f"  Output dim: {train_targets.shape[1]} (residual_dot=9)")
    else:
        print("\n[1/2] No offline data needed (meta-learning only)")

    # 2. 각 모델 학습
    print("\n[2/2] Training models...")
    results = {}
    total_t0 = time.time()

    for model_name in models_to_train:
        try:
            if model_name == "nn":
                results["nn"] = train_nn(
                    dataset, norm_stats, args.epochs, args.batch_size,
                    args.save_dir, verbose=True,
                )
            elif model_name == "gp":
                results["gp"] = train_gp(
                    dataset, norm_stats, args.gp_iterations,
                    args.save_dir, verbose=True,
                )
            elif model_name == "ensemble":
                results["ensemble"] = train_ensemble(
                    dataset, norm_stats, args.epochs, args.batch_size,
                    args.save_dir, verbose=True,
                )
            elif model_name == "mcdropout":
                results["mcdropout"] = train_mcdropout(
                    dataset, norm_stats, args.epochs, args.batch_size,
                    args.save_dir, verbose=True,
                )
            elif model_name == "maml":
                results["maml"] = train_maml(
                    args.meta_iterations, args.save_dir, verbose=True,
                )
            elif model_name == "alpaca":
                results["alpaca"] = train_alpaca(
                    args.meta_iterations, args.save_dir, verbose=True,
                )
        except Exception as e:
            print(f"\n  [ERROR] {model_name} training failed: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_t0

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary".center(60))
    print("=" * 60)
    for model_name in models_to_train:
        status = "OK" if model_name in results else "FAILED"
        save_path = os.path.join(args.save_dir, model_name)
        print(f"  [{model_name:<10}] {status:>6} → {save_path}/")

    print(f"\n  Total time: {total_elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
