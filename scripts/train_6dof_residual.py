#!/usr/bin/env python3
"""
6-DOF Mobile Manipulator Residual Dynamics 학습 스크립트

기구학-동역학 차이(residual)를 NN으로 학습.

단계:
  1. Kinematic/Dynamic 모델 생성
  2. 랜덤 (state, control) 쌍 생성
  3. Residual 계산: dynamic - kinematic
  4. DynamicsDataset → NeuralNetworkTrainer 학습
  5. 모델 저장

Usage:
    PYTHONPATH=. python scripts/train_6dof_residual.py --samples 10000 --epochs 200
    PYTHONPATH=. python scripts/train_6dof_residual.py --samples 2000 --epochs 50  # 빠른 테스트
"""

import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
    MobileManipulator6DOFDynamic,
)
from mppi_controller.learning.data_collector import DynamicsDataset
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer


def generate_residual_data(kin_model, dyn_model, n_samples, seed=42):
    """
    랜덤 (state, control) 쌍에서 residual 데이터 생성.

    Args:
        kin_model: 기구학 모델
        dyn_model: 동역학 모델 (ground-truth)
        n_samples: 샘플 수
        seed: 랜덤 시드

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


def main():
    parser = argparse.ArgumentParser(
        description="Train 6-DOF Residual Dynamics NN"
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 64])
    parser.add_argument("--save-dir", type=str, default="models/learned_models/6dof_residual")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("6-DOF Residual Dynamics Training".center(60))
    print("=" * 60)
    print(f"  Samples:    {args.samples}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Hidden:     {args.hidden}")
    print(f"  Save dir:   {args.save_dir}")
    print("=" * 60 + "\n")

    # 1. 모델 생성
    kin_model = MobileManipulator6DOFKinematic()
    dyn_model = MobileManipulator6DOFDynamic()

    print("[1/4] Generating residual data...")
    data = generate_residual_data(kin_model, dyn_model, args.samples, seed=args.seed)

    # 2. Dataset
    print("\n[2/4] Creating dataset...")
    dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()
    norm_stats = dataset.get_normalization_stats()

    print(f"  Train: {train_inputs.shape[0]} samples")
    print(f"  Val:   {val_inputs.shape[0]} samples")
    print(f"  Input dim:  {train_inputs.shape[1]} (state=9 + control=8)")
    print(f"  Output dim: {train_targets.shape[1]} (residual_dot=9)")

    # 3. 학습
    print(f"\n[3/4] Training MLP {args.hidden}...")
    trainer = NeuralNetworkTrainer(
        state_dim=9,
        control_dim=8,
        hidden_dims=args.hidden,
        activation="relu",
        dropout_rate=0.05,
        learning_rate=args.lr,
        weight_decay=1e-5,
        save_dir=args.save_dir,
    )

    history = trainer.train(
        train_inputs, train_targets,
        val_inputs, val_targets,
        norm_stats=norm_stats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=30,
        verbose=True,
    )

    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("Training Results".center(60))
    print("=" * 60)
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss:   {history['val_loss'][-1]:.6f}")
    print(f"  Best val loss:    {min(history['val_loss']):.6f}")
    print(f"  Epochs trained:   {len(history['train_loss'])}")
    print(f"  Model saved to:   {args.save_dir}/best_model.pth")
    print("=" * 60 + "\n")

    # 5. 검증: 몇 개 샘플로 예측 확인
    print("Validation samples:")
    np.random.seed(123)
    for i in range(5):
        state = data["states"][i]
        control = data["controls"][i]
        true_residual = data["state_dots"][i]
        pred_residual = trainer.predict(state, control)

        error = np.linalg.norm(true_residual - pred_residual)
        print(f"  Sample {i}: true_norm={np.linalg.norm(true_residual):.4f}, "
              f"pred_norm={np.linalg.norm(pred_residual):.4f}, error={error:.4f}")

    # 6. Loss 플롯
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs("plots", exist_ok=True)
    trainer.plot_training_history(save_path="plots/6dof_residual_training.png")
    print("Training plot saved to plots/6dof_residual_training.png")


if __name__ == "__main__":
    main()
