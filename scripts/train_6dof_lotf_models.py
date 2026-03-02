#!/usr/bin/env python3
"""
6-DOF LotF 모델 학습: NN+Spectral, BPTT, NN-Policy

Usage:
    PYTHONPATH=. python scripts/train_6dof_lotf_models.py
    PYTHONPATH=. python scripts/train_6dof_lotf_models.py --quick
    PYTHONPATH=. python scripts/train_6dof_lotf_models.py --models nn_spectral
    PYTHONPATH=. python scripts/train_6dof_lotf_models.py --models bptt
    PYTHONPATH=. python scripts/train_6dof_lotf_models.py --models nn_policy
"""

import numpy as np
import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
    MobileManipulator6DOFDynamic,
)
from mppi_controller.learning.data_collector import DynamicsDataset


SAVE_DIR = "models/learned_models/6dof_benchmark"


def generate_residual_data(kin_model, dyn_model, n_samples, seed=42):
    """Generate residual data (dynamic - kinematic)."""
    np.random.seed(seed)

    states = np.zeros((n_samples, 9))
    controls = np.zeros((n_samples, 8))

    states[:, 0] = np.random.uniform(-2.0, 2.0, n_samples)
    states[:, 1] = np.random.uniform(-2.0, 2.0, n_samples)
    states[:, 2] = np.random.uniform(-np.pi, np.pi, n_samples)
    for i in range(6):
        states[:, 3 + i] = np.random.uniform(-np.pi, np.pi, n_samples)

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

    kin_dots = kin_model.forward_dynamics(states, controls)
    dyn_dots = dyn_model.forward_dynamics(states, controls)
    residuals = dyn_dots - kin_dots

    print(f"  Residual norm mean: {np.mean(np.linalg.norm(residuals, axis=1)):.4f}")
    return {"states": states, "controls": controls, "state_dots": residuals}


# ─────────────────────────────────────────────────────────────
# 1. NN + Spectral Regularization
# ─────────────────────────────────────────────────────────────

def train_nn_spectral(dataset, norm_stats, epochs, batch_size, save_dir,
                      spectral_lambda=0.01, verbose=True):
    """Train NN with Spectral Regularization."""
    from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

    print("\n" + "─" * 60)
    print(f"  [NN+Spectral] Training (λ_spectral={spectral_lambda})")
    print("─" * 60)

    out_dir = os.path.join(save_dir, "nn_spectral")
    trainer = NeuralNetworkTrainer(
        state_dim=9,
        control_dim=8,
        hidden_dims=[128, 128, 64],
        activation="relu",
        dropout_rate=0.05,
        learning_rate=1e-3,
        weight_decay=1e-5,
        spectral_lambda=spectral_lambda,
        save_dir=out_dir,
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
    print(f"  [NN+Spectral] Final val loss: {history['val_loss'][-1]:.6f} ({elapsed:.1f}s)")

    # Show spectral norms
    if trainer.spectral_reg is not None:
        norms = trainer.spectral_reg.get_spectral_norms()
        print(f"  Spectral norms: { {k: f'{v:.2f}' for k, v in norms.items()} }")

    return history


# ─────────────────────────────────────────────────────────────
# 2. BPTT Residual Trainer
# ─────────────────────────────────────────────────────────────

def train_bptt(dataset, norm_stats, save_dir, n_episodes=50, epochs=100,
               mse_pretrain_epochs=50,
               rollout_horizon=20, truncation_length=20, spectral_lambda=0.01,
               learning_rate=1e-4, duration=3.0, batch_size=256, verbose=True):
    """
    Train residual model: MSE pre-train → BPTT fine-tune.

    LotF 논문 방식: 먼저 per-sample MSE로 좋은 초기화를 얻은 후,
    궤적 수준 BPTT로 파인튜닝하여 누적 오차를 직접 최소화.
    """
    from mppi_controller.learning.bptt_residual_trainer import BPTTResidualTrainer
    from mppi_controller.learning.neural_network_trainer import (
        NeuralNetworkTrainer, DynamicsMLPModel,
    )
    from mppi_controller.models.differentiable.diff_sim_6dof import (
        DifferentiableMobileManipulator6DOF,
    )
    from mppi_controller.utils.trajectory import create_trajectory_function

    print("\n" + "─" * 60)
    print(f"  [BPTT] MSE Pre-train ({mse_pretrain_epochs} ep) → BPTT Fine-tune ({epochs} ep)")
    print(f"    Episodes: {n_episodes}, Horizon: {rollout_horizon}")
    print(f"    Spectral λ: {spectral_lambda}, BPTT LR: {learning_rate}")
    print("─" * 60)

    out_dir = os.path.join(save_dir, "bptt")
    os.makedirs(out_dir, exist_ok=True)

    # ── Phase 1: MSE Pre-training ──
    print("\n  [Phase 1] MSE Pre-training...")
    mse_trainer = NeuralNetworkTrainer(
        state_dim=9,
        control_dim=8,
        hidden_dims=[128, 128, 64],
        activation="relu",
        dropout_rate=0.0,
        learning_rate=1e-3,
        weight_decay=1e-5,
        spectral_lambda=spectral_lambda,
        save_dir=out_dir,
    )

    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()

    t0 = time.time()
    mse_history = mse_trainer.train(
        train_inputs, train_targets,
        val_inputs, val_targets,
        norm_stats=norm_stats,
        epochs=mse_pretrain_epochs,
        batch_size=batch_size,
        early_stopping_patience=30,
        verbose=verbose,
    )
    mse_elapsed = time.time() - t0
    print(f"  [Phase 1] MSE val loss: {mse_history['val_loss'][-1]:.6f} ({mse_elapsed:.1f}s)")

    # Save MSE pre-trained model separately (for comparison / fallback)
    mse_trainer.save_model("mse_pretrained.pth")
    print(f"  MSE pre-trained saved: {os.path.join(out_dir, 'mse_pretrained.pth')}")

    # ── Phase 2: BPTT Fine-tuning ──
    print("\n  [Phase 2] BPTT Fine-tuning...")

    # Convert pre-trained model to float64 for differentiable sim
    residual_model = mse_trainer.model.double()

    diff_sim = DifferentiableMobileManipulator6DOF()

    bptt_trainer = BPTTResidualTrainer(
        residual_model=residual_model,
        diff_sim=diff_sim,
        norm_stats=norm_stats,
        learning_rate=learning_rate,
        spectral_lambda=spectral_lambda,
        ee_weight=1.0,
        rollout_horizon=rollout_horizon,
        truncation_length=truncation_length,
        dt=0.05,
        device="cpu",
        save_dir=out_dir,
    )

    # Generate episodes using MPPI with oracle dynamics
    print(f"\n  Generating {n_episodes} episodes ({duration}s each)...")
    kin_model = MobileManipulator6DOFKinematic()
    dyn_model = MobileManipulator6DOFDynamic()

    traj_types = ["ee_3d_circle", "ee_3d_helix"]
    all_episodes = []

    for traj_name in traj_types:
        traj_fn = create_trajectory_function(traj_name)
        n_ep = n_episodes // len(traj_types)
        print(f"    {traj_name}: {n_ep} episodes...")
        episodes = bptt_trainer.generate_episodes(
            dyn_model=dyn_model,
            kin_model=kin_model,
            traj_fn=traj_fn,
            n_episodes=n_ep,
            duration=duration,
        )
        all_episodes.extend(episodes)

    np.random.shuffle(all_episodes)

    n_train = int(len(all_episodes) * 0.8)
    train_episodes = all_episodes[:n_train]
    val_episodes = all_episodes[n_train:]
    print(f"  Train: {len(train_episodes)} episodes, Val: {len(val_episodes)} episodes")

    t0 = time.time()
    bptt_history = bptt_trainer.train(
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        epochs=epochs,
        verbose=verbose,
    )
    bptt_elapsed = time.time() - t0

    print(f"\n  [Phase 2] BPTT complete ({bptt_elapsed:.1f}s)")
    print(f"    Final train loss: {bptt_history['train_loss'][-1]:.6f}")
    if bptt_history['val_loss']:
        print(f"    Final val loss:   {bptt_history['val_loss'][-1]:.6f}")

    # Save as NeuralDynamics-compatible format
    _save_bptt_as_neural_dynamics(bptt_trainer, out_dir, norm_stats)

    return bptt_history


def _save_bptt_as_neural_dynamics(trainer, out_dir, norm_stats):
    """
    BPTT 모델을 NeuralDynamics가 로드할 수 있는 형식으로 저장.
    lotf_benchmark.py의 create_model()이 NeuralDynamics로 로드.
    """
    model = trainer.residual_model

    # Convert back to float32 for inference compatibility
    model_f32 = model.float()

    config = {
        "state_dim": 9,
        "control_dim": 8,
        "hidden_dims": [128, 128, 64],
        "activation": "relu",
        "dropout_rate": 0.0,
    }

    filepath = os.path.join(out_dir, "best_model.pth")
    torch.save({
        "model_state_dict": model_f32.state_dict(),
        "norm_stats": norm_stats,
        "history": trainer.history,
        "config": config,
    }, filepath)
    print(f"  Saved: {filepath}")


# ─────────────────────────────────────────────────────────────
# 3. NN-Policy (Behavioral Cloning + BPTT)
# ─────────────────────────────────────────────────────────────

def train_nn_policy(save_dir, n_episodes=80, bc_epochs=100, bptt_epochs=50,
                    rollout_horizon=20, duration=4.0, verbose=True):
    """
    NN-Policy: MPPI 없이 직접 (state, ee_ref) → control 정책 학습.

    Phase 1: MPPI oracle로 시연 데이터 수집
    Phase 2: Behavioral Cloning (MSE)
    Phase 3: BPTT Fine-tune (궤적 loss)
    """
    from mppi_controller.learning.nn_policy_trainer import NNPolicyTrainer
    from mppi_controller.utils.trajectory import create_trajectory_function

    print("\n" + "─" * 60)
    print(f"  [NN-Policy] BC ({bc_epochs} ep) → BPTT Fine-tune ({bptt_epochs} ep)")
    print(f"    Episodes: {n_episodes}, Horizon: {rollout_horizon}, Duration: {duration}s")
    print("─" * 60)

    out_dir = os.path.join(save_dir, "nn_policy")
    os.makedirs(out_dir, exist_ok=True)

    kin_model = MobileManipulator6DOFKinematic()
    dyn_model = MobileManipulator6DOFDynamic()

    # Get control bounds from kinematic model
    bounds = kin_model.get_control_bounds()
    if bounds is not None:
        control_bounds = bounds[1]  # upper bounds
    else:
        control_bounds = np.array([1.0, 2.0] + [3.0] * 6)

    trainer = NNPolicyTrainer(
        state_dim=9, ee_ref_dim=3, control_dim=8,
        hidden_dims=[128, 128, 64],
        control_bounds=control_bounds,
        learning_rate=1e-3,
        save_dir=out_dir,
    )

    # ── Phase 1: Generate demonstrations ──
    traj_types = ["ee_3d_circle", "ee_3d_helix"]
    all_episodes = []

    for traj_name in traj_types:
        traj_fn = create_trajectory_function(traj_name)
        n_ep = n_episodes // len(traj_types)
        print(f"\n  Generating {n_ep} episodes ({traj_name}, {duration}s each)...")
        episodes = trainer.generate_demonstrations(
            dyn_model=dyn_model,
            kin_model=kin_model,
            traj_fn=traj_fn,
            n_episodes=n_ep,
            duration=duration,
        )
        all_episodes.extend(episodes)

    np.random.shuffle(all_episodes)
    print(f"  Total episodes: {len(all_episodes)}")

    # ── Phase 2: Behavioral Cloning ──
    print("\n  [Phase 2] Behavioral Cloning...")
    t0 = time.time()
    bc_history = trainer.train_bc(all_episodes, epochs=bc_epochs, verbose=verbose)
    bc_elapsed = time.time() - t0
    print(f"  [Phase 2] BC complete ({bc_elapsed:.1f}s)")

    # ── Phase 3: BPTT Fine-tune ──
    print("\n  [Phase 3] BPTT Fine-tuning...")
    t0 = time.time()
    bptt_history = trainer.train_bptt(
        all_episodes, epochs=bptt_epochs,
        rollout_horizon=rollout_horizon,
        verbose=verbose,
    )
    bptt_elapsed = time.time() - t0
    print(f"  [Phase 3] BPTT complete ({bptt_elapsed:.1f}s)")

    trainer.save_model("best_model.pth")
    print(f"  Saved: {os.path.join(out_dir, 'best_model.pth')}")

    return {"bc": bc_history, "bptt": bptt_history}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train NN+Spectral, BPTT, and NN-Policy models for LotF benchmark"
    )
    parser.add_argument("--samples", type=int, default=10000,
                        help="Residual data samples (for NN+Spectral)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs (NN+Spectral)")
    parser.add_argument("--bptt-epochs", type=int, default=100,
                        help="BPTT training epochs")
    parser.add_argument("--bptt-episodes", type=int, default=50,
                        help="BPTT training episodes")
    parser.add_argument("--bptt-duration", type=float, default=3.0,
                        help="BPTT episode duration (seconds)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--spectral-lambda", type=float, default=0.01)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--truncation-length", type=int, default=20)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated: nn_spectral,bptt (default: both)")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduced samples/epochs/episodes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.samples = 2000
        args.epochs = 30
        args.bptt_epochs = 30
        args.bptt_episodes = 20
        args.bptt_duration = 2.0

    available = ["nn_spectral", "bptt", "nn_policy"]
    if args.models:
        models_to_train = [m.strip() for m in args.models.split(",")]
        for m in models_to_train:
            if m not in available:
                print(f"Unknown model: {m}. Available: {available}")
                sys.exit(1)
    else:
        models_to_train = available

    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("LotF Model Training (NN+Spectral & BPTT & NN-Policy)".center(60))
    print("=" * 60)
    print(f"  Models:       {models_to_train}")
    print(f"  Quick mode:   {args.quick}")
    print(f"  Save dir:     {args.save_dir}")
    print("=" * 60)

    kin_model = MobileManipulator6DOFKinematic()
    dyn_model = MobileManipulator6DOFDynamic()

    # Generate data for NN+Spectral (also provides norm_stats for BPTT)
    print("\n[Step 1] Generating residual data...")
    data = generate_residual_data(kin_model, dyn_model, args.samples, seed=args.seed)
    dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
    norm_stats = dataset.get_normalization_stats()

    train_inputs, _ = dataset.get_train_data()
    val_inputs, _ = dataset.get_val_data()
    print(f"  Train: {train_inputs.shape[0]}, Val: {val_inputs.shape[0]}")

    # Train
    print("\n[Step 2] Training...")
    total_t0 = time.time()

    if "nn_spectral" in models_to_train:
        train_nn_spectral(
            dataset, norm_stats, args.epochs, args.batch_size,
            args.save_dir, spectral_lambda=args.spectral_lambda,
        )

    if "bptt" in models_to_train:
        train_bptt(
            dataset, norm_stats, args.save_dir,
            n_episodes=args.bptt_episodes,
            epochs=args.bptt_epochs,
            mse_pretrain_epochs=args.epochs // 2,
            rollout_horizon=args.rollout_horizon,
            truncation_length=args.truncation_length,
            spectral_lambda=args.spectral_lambda,
            learning_rate=1e-4,
            duration=args.bptt_duration,
            batch_size=args.batch_size,
        )

    if "nn_policy" in models_to_train:
        nn_policy_episodes = args.bptt_episodes * 2 if not args.quick else 20
        nn_policy_bc_epochs = args.epochs if not args.quick else 30
        nn_policy_bptt_epochs = args.bptt_epochs if not args.quick else 20
        nn_policy_duration = args.bptt_duration + 1.0 if not args.quick else 2.0
        train_nn_policy(
            args.save_dir,
            n_episodes=nn_policy_episodes,
            bc_epochs=nn_policy_bc_epochs,
            bptt_epochs=nn_policy_bptt_epochs,
            rollout_horizon=args.rollout_horizon,
            duration=nn_policy_duration,
        )

    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 60)
    print(f"  Total training time: {total_elapsed:.1f}s")
    print("=" * 60)
    print("\nRun benchmark:")
    print("  PYTHONPATH=. python examples/comparison/lotf_benchmark.py --duration 10")


if __name__ == "__main__":
    main()
