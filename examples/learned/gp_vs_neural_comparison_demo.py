#!/usr/bin/env python3
"""
Gaussian Process vs Neural Network Comparison Demo

GP와 Neural Network 학습 모델 비교:
1. 데이터 효율성 (적은 데이터로 학습)
2. 불확실성 정량화
3. 계산 시간
4. MPPI 제어 성능
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
import time

# MPPI components
from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
from mppi_controller.models.learned.gaussian_process_dynamics import GaussianProcessDynamics
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics
from mppi_controller.utils.trajectory import create_trajectory_function, generate_reference_trajectory

# Learning components
from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer
from mppi_controller.learning.gaussian_process_trainer import GaussianProcessTrainer


def collect_data(duration: float = 30.0, trajectory_type: str = "circle") -> DataCollector:
    """데이터 수집"""
    print("\n" + "="*60)
    print("Step 1: Data Collection")
    print("="*60)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    controller = MPPIController(model, params)
    simulator = Simulator(model, controller, params.dt)
    trajectory_fn = create_trajectory_function(trajectory_type)

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    collector = DataCollector(state_dim=3, control_dim=2)
    initial_state = trajectory_fn(0.0)
    simulator.reset(initial_state)

    num_steps = int(duration / params.dt)
    print(f"\nCollecting {num_steps} samples ({duration}s)...")

    for step in range(num_steps):
        t = step * params.dt
        ref_traj = reference_fn(t)
        control, info = controller.compute_control(simulator.state, ref_traj)
        current_state = simulator.state.copy()
        next_state = model.step(current_state, control, params.dt)
        simulator.state = next_state
        collector.add_sample(current_state, control, next_state, params.dt)

    collector.end_episode()
    collector.save("gp_neural_comparison_data.pkl")

    stats = collector.get_statistics()
    print(f"Data collection completed: {stats['num_samples']} samples")

    return collector


def train_models(
    collector: DataCollector,
    neural_epochs: int = 100,
    gp_iterations: int = 100,
    data_fraction: float = 1.0,
):
    """
    GP와 Neural Network 학습

    Args:
        collector: 수집된 데이터
        neural_epochs: Neural network epochs
        gp_iterations: GP optimization iterations
        data_fraction: 사용할 데이터 비율 (데이터 효율성 테스트)

    Returns:
        trainers: {"neural": neural_trainer, "gp": gp_trainer}
    """
    print("\n" + "="*60)
    print(f"Step 2: Model Training (using {data_fraction*100:.0f}% of data)")
    print("="*60)

    # Prepare dataset
    data = collector.get_data()

    # Limit data if testing data efficiency
    if data_fraction < 1.0:
        num_samples = int(len(data["states"]) * data_fraction)
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                data[key] = data[key][:num_samples]

    dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True, shuffle=True)

    print(f"\nDataset: {dataset}")

    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()
    norm_stats = dataset.get_normalization_stats()

    # ===== 1. Neural Network Training =====
    print("\n" + "-"*60)
    print("Training Neural Network...")
    print("-"*60)

    neural_trainer = NeuralNetworkTrainer(
        state_dim=3,
        control_dim=2,
        hidden_dims=[128, 128],
        activation="relu",
        dropout_rate=0.1,
        learning_rate=1e-3,
        device="cpu",
    )

    print(neural_trainer.get_model_summary())

    neural_start = time.time()
    neural_trainer.train(
        train_inputs, train_targets, val_inputs, val_targets, norm_stats,
        epochs=neural_epochs, batch_size=64, early_stopping_patience=20, verbose=True
    )
    neural_train_time = time.time() - neural_start

    neural_trainer.save_model("neural_comparison.pth")
    print(f"\nNeural training time: {neural_train_time:.2f}s")

    # ===== 2. Gaussian Process Training =====
    print("\n" + "-"*60)
    print("Training Gaussian Process...")
    print("-"*60)

    # Use sparse GP if data is large
    use_sparse = len(train_inputs) > 500

    gp_trainer = GaussianProcessTrainer(
        state_dim=3,
        control_dim=2,
        kernel_type="rbf",
        use_sparse=use_sparse,
        num_inducing_points=min(200, len(train_inputs) // 2),
        use_ard=True,
        device="cpu",
    )

    print(gp_trainer.get_model_summary())

    gp_start = time.time()
    gp_trainer.train(
        train_inputs, train_targets, val_inputs, val_targets, norm_stats,
        num_iterations=gp_iterations, learning_rate=0.1, verbose=True
    )
    gp_train_time = time.time() - gp_start

    gp_trainer.save_model("gp_comparison.pth")
    print(f"\nGP training time: {gp_train_time:.2f}s")

    # ===== Summary =====
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Neural Network:")
    print(f"  Training time: {neural_train_time:.2f}s")
    print(f"  Val loss (final): {neural_trainer.history['val_loss'][-1]:.6f}")

    print(f"\nGaussian Process:")
    print(f"  Training time: {gp_train_time:.2f}s")
    print(f"  Val loss (final): {gp_trainer.history['val_loss'][-1]:.6f}")
    print(f"  Sparse GP: {use_sparse}")

    return {
        "neural": neural_trainer,
        "gp": gp_trainer,
        "train_time": {"neural": neural_train_time, "gp": gp_train_time},
    }


def evaluate_uncertainty_calibration(trainers: dict, val_data: dict):
    """불확실성 보정 평가"""
    print("\n" + "="*60)
    print("Step 3: Uncertainty Calibration Evaluation")
    print("="*60)

    val_states = val_data["states"][:100]  # 샘플 100개
    val_controls = val_data["controls"][:100]
    val_targets = val_data["state_dots"][:100]

    gp_trainer = trainers["gp"]

    # GP prediction with uncertainty
    gp_means, gp_stds = gp_trainer.predict(
        val_states, val_controls, denormalize=True, return_uncertainty=True
    )

    # Compute calibration metrics
    errors = np.abs(gp_means - val_targets)
    normalized_errors = errors / (gp_stds + 1e-6)

    # Expected: ~68% within 1-sigma, ~95% within 2-sigma
    within_1sigma = np.mean(normalized_errors < 1.0, axis=0)
    within_2sigma = np.mean(normalized_errors < 2.0, axis=0)

    print("\nGP Uncertainty Calibration:")
    print(f"  Within 1-sigma: {within_1sigma * 100}")
    print(f"  Within 2-sigma: {within_2sigma * 100}")
    print(f"  Expected: 68% within 1-sigma, 95% within 2-sigma")

    return {
        "within_1sigma": within_1sigma,
        "within_2sigma": within_2sigma,
        "mean_uncertainty": np.mean(gp_stds, axis=0),
    }


def evaluate_mppi_control(
    trainers: dict,
    trajectory_type: str = "circle",
    duration: float = 15.0,
):
    """MPPI 제어 성능 평가"""
    print("\n" + "="*60)
    print("Step 4: MPPI Control Performance Evaluation")
    print("="*60)

    # Models
    kinematic_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    neural_model = NeuralDynamics(state_dim=3, control_dim=2)
    neural_model.load_model("models/learned_models/neural_comparison.pth")

    gp_model = GaussianProcessDynamics(state_dim=3, control_dim=2)
    gp_model.load_model("models/learned_models/gp_comparison.pth")

    models = {
        "Kinematic (Physics)": kinematic_model,
        "Neural Network": neural_model,
        "Gaussian Process": gp_model,
    }

    # MPPI params
    params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )

    trajectory_fn = create_trajectory_function(trajectory_type)

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    # Run simulations
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

    return results


def plot_comprehensive_comparison(
    trainers: dict,
    uncertainty_metrics: dict,
    control_results: dict,
    data_fraction: float,
):
    """종합 비교 플롯"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    model_names = list(control_results.keys())
    colors = ['blue', 'red', 'green']

    # 1. Training Loss Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    neural_trainer = trainers["neural"]
    ax1.plot(neural_trainer.history["train_loss"], label="Neural Train", linewidth=2)
    ax1.plot(neural_trainer.history["val_loss"], label="Neural Val", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Loss (Neural Network)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. GP Loss
    ax2 = fig.add_subplot(gs[0, 1])
    gp_trainer = trainers["gp"]
    if len(gp_trainer.history["val_loss"]) > 0:
        ax2.bar([0], [gp_trainer.history["val_loss"][-1]], color='green')
        ax2.set_xticks([0])
        ax2.set_xticklabels(['GP'])
        ax2.set_ylabel("Val MSE")
        ax2.set_title("GP Validation Loss")
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. Training Time Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    train_times = [trainers["train_time"]["neural"], trainers["train_time"]["gp"]]
    bars = ax3.bar(["Neural", "GP"], train_times, color=['red', 'green'])
    ax3.set_ylabel("Time (s)")
    ax3.set_title(f"Training Time ({data_fraction*100:.0f}% data)")
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, train_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom')

    # 4. Uncertainty Calibration (GP only)
    ax4 = fig.add_subplot(gs[1, 0])
    dims = ['x', 'y', 'θ']
    x_pos = np.arange(len(dims))
    width = 0.35
    ax4.bar(x_pos - width/2, uncertainty_metrics["within_1sigma"] * 100,
            width, label='Within 1σ', color='skyblue')
    ax4.bar(x_pos + width/2, uncertainty_metrics["within_2sigma"] * 100,
            width, label='Within 2σ', color='lightgreen')
    ax4.axhline(y=68, color='r', linestyle='--', label='Expected 1σ')
    ax4.axhline(y=95, color='g', linestyle='--', label='Expected 2σ')
    ax4.set_ylabel("Percentage (%)")
    ax4.set_title("GP Uncertainty Calibration")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dims)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Mean Uncertainty (GP)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(dims, uncertainty_metrics["mean_uncertainty"], color='orange')
    ax5.set_ylabel("Uncertainty (std)")
    ax5.set_title("GP Mean Uncertainty")
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. MPPI Performance Metrics
    ax6 = fig.add_subplot(gs[1, 2])
    rmse_values = [control_results[name]["metrics"]["position_rmse"] for name in model_names]
    bars = ax6.bar(range(len(model_names)), rmse_values, color=colors)
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels([name.split()[0] for name in model_names], rotation=0)
    ax6.set_ylabel("RMSE (m)")
    ax6.set_title("Control Position RMSE")
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # 7-9. XY Trajectories for each model
    for i, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[2, i])
        history = control_results[model_name]["history"]
        states = history["state"]
        references = history["reference"]

        ax.plot(states[:, 0], states[:, 1], label=model_name.split()[0],
                linewidth=2, color=colors[i])
        ax.plot(references[:, 0], references[:, 1], 'k--', label='Reference',
                linewidth=1, alpha=0.5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{model_name.split()[0]} Trajectory")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # 10-12. Position Errors
    for i, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[3, i])
        history = control_results[model_name]["history"]
        states = history["state"]
        references = history["reference"]
        time_arr = history["time"]

        pos_errors = np.linalg.norm(states[:, :2] - references[:, :2], axis=1)

        ax.plot(time_arr, pos_errors, linewidth=2, color=colors[i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title(f"{model_name.split()[0]} Error")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(pos_errors) * 1.1])

    plt.suptitle(f"GP vs Neural Network Comparison ({data_fraction*100:.0f}% data)",
                 fontsize=14, fontweight='bold')

    # Save
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/gp_neural_comparison_{int(data_fraction*100)}pct.png",
                dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to plots/gp_neural_comparison_{int(data_fraction*100)}pct.png")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="GP vs Neural Network Comparison")
    parser.add_argument("--collect-data", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--trajectory", type=str, default="circle")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--neural-epochs", type=int, default=100)
    parser.add_argument("--gp-iterations", type=int, default=100)
    parser.add_argument("--data-fraction", type=float, default=1.0,
                       help="Fraction of data to use (test data efficiency)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot display")

    args = parser.parse_args()

    # Create directories
    Path("data/learned_models").mkdir(parents=True, exist_ok=True)
    Path("models/learned_models").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)

    collector = None
    trainers = None
    uncertainty_metrics = None
    control_results = None

    # Step 1: Collect data
    if args.collect_data or args.all:
        collector = collect_data(duration=args.duration, trajectory_type=args.trajectory)

    # Step 2: Train
    if args.train or args.all:
        if collector is None:
            collector = DataCollector(state_dim=3, control_dim=2)
            collector.load("gp_neural_comparison_data.pkl")

        trainers = train_models(
            collector,
            neural_epochs=args.neural_epochs,
            gp_iterations=args.gp_iterations,
            data_fraction=args.data_fraction,
        )

    # Step 3-4: Evaluate
    if args.evaluate or args.all:
        if trainers is None:
            # Load trained models
            from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer
            from mppi_controller.learning.gaussian_process_trainer import GaussianProcessTrainer

            neural_trainer = NeuralNetworkTrainer(state_dim=3, control_dim=2)
            neural_trainer.load_model("neural_comparison.pth")

            gp_trainer = GaussianProcessTrainer(state_dim=3, control_dim=2)
            gp_trainer.load_model("gp_comparison.pth")

            trainers = {"neural": neural_trainer, "gp": gp_trainer,
                       "train_time": {"neural": 0, "gp": 0}}

        # Uncertainty calibration
        if collector is None:
            collector = DataCollector(state_dim=3, control_dim=2)
            collector.load("gp_neural_comparison_data.pkl")

        val_data = collector.get_data()
        uncertainty_metrics = evaluate_uncertainty_calibration(trainers, val_data)

        # Control performance
        control_results = evaluate_mppi_control(trainers, trajectory_type=args.trajectory)

        # Plot
        plot_comprehensive_comparison(
            trainers, uncertainty_metrics, control_results, args.data_fraction
        )

        # Final summary
        print("\n" + "="*60)
        print("Final Summary")
        print("="*60)
        for model_name, data in control_results.items():
            metrics = data["metrics"]
            print(f"\n{model_name}:")
            print(f"  Position RMSE: {metrics['position_rmse']:.4f}m")
            print(f"  Max error: {metrics['max_position_error']:.4f}m")
            print(f"  Mean solve time: {metrics['mean_solve_time']*1000:.2f}ms")


if __name__ == "__main__":
    main()
