#!/usr/bin/env python3
"""
Neural Dynamics Learning Demo

전체 학습 파이프라인 시연:
1. 데이터 수집 (Vanilla MPPI로 시뮬레이션)
2. 신경망 학습
3. 학습된 모델로 MPPI 제어
4. 성능 비교 (물리 모델 vs 학습 모델)
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if "--no-plot" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# MPPI components
from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics
from mppi_controller.utils.trajectory import create_trajectory_function, generate_reference_trajectory

# Learning components
from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer


def collect_data_from_simulation(
    duration: float = 30.0,
    trajectory_type: str = "circle",
) -> DataCollector:
    """
    Step 1: Vanilla MPPI 시뮬레이션으로 데이터 수집

    Args:
        duration: 시뮬레이션 시간 (초)
        trajectory_type: 궤적 타입

    Returns:
        collector: 수집된 데이터
    """
    print("\n" + "="*60)
    print("Step 1: Data Collection from Simulation")
    print("="*60)

    # Model
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # MPPI params
    params = MPPIParams(
        N=30,
        dt=0.05,
        K=1024,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )

    # Controller
    controller = MPPIController(model, params)

    # Simulator
    simulator = Simulator(model, controller, params.dt)

    # Reference trajectory
    trajectory_fn = create_trajectory_function(trajectory_type)

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    # Data collector
    collector = DataCollector(state_dim=3, control_dim=2)

    # Run simulation and collect data
    initial_state = trajectory_fn(0.0)
    simulator.reset(initial_state)

    num_steps = int(duration / params.dt)
    print(f"\nRunning simulation: {duration}s ({num_steps} steps)")

    for step in range(num_steps):
        t = step * params.dt
        ref_traj = reference_fn(t)

        # Compute control
        control, info = controller.compute_control(simulator.state, ref_traj)

        # Record current state and control
        current_state = simulator.state.copy()

        # Step simulation
        next_state = model.step(current_state, control, params.dt)
        simulator.state = next_state

        # Add data sample
        collector.add_sample(current_state, control, next_state, params.dt)

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{num_steps}: state={current_state[:2]}")

    # End episode
    collector.end_episode()

    # Save data
    collector.save("neural_dynamics_training_data.pkl")

    # Statistics
    stats = collector.get_statistics()
    print(f"\nData collection completed:")
    print(f"  Samples: {stats['num_samples']}")
    print(f"  State mean: {stats['state_mean']}")
    print(f"  State std: {stats['state_std']}")

    return collector


def train_neural_network(
    collector: DataCollector,
    epochs: int = 100,
    batch_size: int = 64,
) -> NeuralNetworkTrainer:
    """
    Step 2: 신경망 학습

    Args:
        collector: 수집된 데이터
        epochs: 에폭 수
        batch_size: 배치 크기

    Returns:
        trainer: 학습된 모델
    """
    print("\n" + "="*60)
    print("Step 2: Neural Network Training")
    print("="*60)

    # Prepare dataset
    data = collector.get_data()
    dataset = DynamicsDataset(
        data,
        train_ratio=0.8,
        normalize=True,
        shuffle=True,
    )

    print(f"\nDataset prepared:")
    print(f"  {dataset}")
    print(f"  Train samples: {len(dataset.train_states)}")
    print(f"  Val samples: {len(dataset.val_states)}")

    # Create trainer
    trainer = NeuralNetworkTrainer(
        state_dim=3,
        control_dim=2,
        hidden_dims=[128, 128, 64],
        activation="relu",
        dropout_rate=0.1,
        learning_rate=1e-3,
        device="cpu",
    )

    print(f"\nModel architecture:")
    print(trainer.get_model_summary())

    # Get training data
    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()
    norm_stats = dataset.get_normalization_stats()

    # Train
    print(f"\nStarting training: {epochs} epochs, batch_size={batch_size}")
    history = trainer.train(
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        norm_stats,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=20,
        verbose=True,
    )

    # Plot training history
    trainer.plot_training_history("plots/neural_dynamics_training_history.png")

    return trainer


def evaluate_learned_model(
    trainer: NeuralNetworkTrainer,
    trajectory_type: str = "circle",
    duration: float = 15.0,
):
    """
    Step 3: 학습된 모델 평가 및 비교

    Args:
        trainer: 학습된 모델
        trajectory_type: 궤적 타입
        duration: 평가 시간

    Returns:
        results: 평가 결과
    """
    print("\n" + "="*60)
    print("Step 3: Learned Model Evaluation")
    print("="*60)

    # 1. Kinematic model (ground truth)
    kinematic_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # 2. Neural model (will load config from checkpoint)
    neural_model = NeuralDynamics(
        state_dim=3,
        control_dim=2,
        model_path="models/learned_models/best_model.pth",
    )

    # 3. Residual model (kinematic + neural correction)
    def neural_residual_fn(state, control):
        return trainer.predict(state, control, denormalize=True) - \
               kinematic_model.forward_dynamics(state, control)

    residual_model = ResidualDynamics(
        base_model=kinematic_model,
        residual_fn=neural_residual_fn,
        use_residual=True,
    )

    models = {
        "Kinematic (Physics)": kinematic_model,
        "Neural (Learned)": neural_model,
        "Residual (Physics+Learned)": residual_model,
    }

    # MPPI params
    params = MPPIParams(
        N=30,
        dt=0.05,
        K=1024,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )

    # Reference trajectory
    trajectory_fn = create_trajectory_function(trajectory_type)

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    # Run simulations for each model
    results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        controller = MPPIController(model, params)
        simulator = Simulator(model, controller, params.dt)

        initial_state = trajectory_fn(0.0)
        simulator.reset(initial_state)

        history = simulator.run(reference_fn, duration=duration)
        metrics = compute_metrics(history)

        results[model_name] = {
            "history": history,
            "metrics": metrics,
        }

        print(f"  RMSE: {metrics['position_rmse']:.4f}m")
        print(f"  Mean solve time: {metrics['mean_solve_time']*1000:.2f}ms")

    # Plot comparison
    plot_model_comparison(results, trajectory_type)

    return results


def plot_model_comparison(results: dict, trajectory_type: str):
    """모델 비교 플롯"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    model_names = list(results.keys())
    colors = ['blue', 'red', 'green']

    # 1. XY Trajectories
    ax1 = fig.add_subplot(gs[0, :2])
    for i, model_name in enumerate(model_names):
        history = results[model_name]["history"]
        states = history["state"]
        references = history["reference"]

        ax1.plot(states[:, 0], states[:, 1], label=f"{model_name}", linewidth=2, color=colors[i])

        if i == 0:
            ax1.plot(references[:, 0], references[:, 1], 'k--', label='Reference', linewidth=1, alpha=0.5)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(f"XY Trajectory - {trajectory_type.title()}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Metrics comparison
    ax2 = fig.add_subplot(gs[0, 2])
    rmse_values = [results[name]["metrics"]["position_rmse"] for name in model_names]
    bars = ax2.bar(range(len(model_names)), rmse_values, color=colors)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([name.split()[0] for name in model_names], rotation=0)
    ax2.set_ylabel("RMSE (m)")
    ax2.set_title("Position RMSE")
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, rmse_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # 3-5. Position errors over time
    for i, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[1, i])
        history = results[model_name]["history"]
        states = history["state"]
        references = history["reference"]
        time = history["time"]

        pos_errors = np.linalg.norm(states[:, :2] - references[:, :2], axis=1)

        ax.plot(time, pos_errors, linewidth=2, color=colors[i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title(f"{model_name.split()[0]}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(pos_errors) * 1.1])

    # 6-8. Control inputs
    for i, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[2, i])
        history = results[model_name]["history"]
        controls = history["control"]
        time = history["time"]

        ax.plot(time, controls[:, 0], label='v', linewidth=2)
        ax.plot(time, controls[:, 1], label='ω', linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control")
        ax.set_title(f"{model_name.split()[0]} Control")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Neural Dynamics Learning: Model Comparison", fontsize=14, fontweight='bold')

    # Save
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/neural_dynamics_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to plots/neural_dynamics_comparison.png")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Neural Dynamics Learning Demo")
    parser.add_argument("--collect-data", action="store_true", help="Collect training data")
    parser.add_argument("--train", action="store_true", help="Train neural network")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate learned model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--trajectory", type=str, default="circle", help="Trajectory type")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration (s)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot display")

    args = parser.parse_args()

    # Create directories
    Path("data/learned_models").mkdir(parents=True, exist_ok=True)
    Path("models/learned_models").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)

    collector = None
    trainer = None

    # Step 1: Collect data
    if args.collect_data or args.all:
        collector = collect_data_from_simulation(
            duration=args.duration,
            trajectory_type=args.trajectory,
        )

    # Step 2: Train
    if args.train or args.all:
        if collector is None:
            # Load existing data
            collector = DataCollector(state_dim=3, control_dim=2)
            collector.load("neural_dynamics_training_data.pkl")

        trainer = train_neural_network(
            collector,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    # Step 3: Evaluate
    if args.evaluate or args.all:
        if trainer is None:
            # Load trained model
            trainer = NeuralNetworkTrainer(state_dim=3, control_dim=2)
            trainer.load_model("best_model.pth")

        results = evaluate_learned_model(
            trainer,
            trajectory_type=args.trajectory,
            duration=15.0,
        )

        # Print final summary
        print("\n" + "="*60)
        print("Final Summary")
        print("="*60)
        for model_name, data in results.items():
            metrics = data["metrics"]
            print(f"\n{model_name}:")
            print(f"  Position RMSE: {metrics['position_rmse']:.4f}m")
            print(f"  Max error: {metrics['max_position_error']:.4f}m")
            print(f"  Mean solve time: {metrics['mean_solve_time']*1000:.2f}ms")


if __name__ == "__main__":
    main()
