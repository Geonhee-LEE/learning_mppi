#!/usr/bin/env python3
"""
Online Learning Demo

온라인 학습 시연:
1. 초기 모델 학습 (적은 데이터)
2. 실시간 데이터 수집 및 모델 업데이트
3. Domain adaptation (시뮬레이션 → 실제)
4. 성능 모니터링 및 시각화
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
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics
from mppi_controller.utils.trajectory import create_trajectory_function, generate_reference_trajectory

# Learning components
from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer
from mppi_controller.learning.online_learner import OnlineLearner


def collect_initial_data(duration: float = 10.0, trajectory_type: str = "circle") -> DataCollector:
    """Step 1: 초기 데이터 수집 (적은 데이터)"""
    print("\n" + "="*60)
    print("Step 1: Initial Data Collection (Limited Data)")
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
    stats = collector.get_statistics()
    print(f"Initial data collected: {stats['num_samples']} samples")

    return collector


def train_initial_model(collector: DataCollector, epochs: int = 50):
    """Step 2: 초기 모델 학습"""
    print("\n" + "="*60)
    print("Step 2: Initial Model Training")
    print("="*60)

    data = collector.get_data()
    dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True, shuffle=True)

    print(f"Dataset: {dataset}")

    train_inputs, train_targets = dataset.get_train_data()
    val_inputs, val_targets = dataset.get_val_data()
    norm_stats = dataset.get_normalization_stats()

    # Neural Network
    trainer = NeuralNetworkTrainer(
        state_dim=3,
        control_dim=2,
        hidden_dims=[64, 64],  # Smaller network for online learning
        activation="relu",
        dropout_rate=0.1,
        learning_rate=1e-3,
        device="cpu",
    )

    print(f"\n{trainer.get_model_summary()}")

    trainer.train(
        train_inputs, train_targets, val_inputs, val_targets, norm_stats,
        epochs=epochs, batch_size=32, early_stopping_patience=10, verbose=True
    )

    trainer.save_model("online_initial_model.pth")

    print(f"\nInitial model trained")
    print(f"  Val loss: {trainer.history['val_loss'][-1]:.6f}")

    return trainer


def simulate_domain_shift(
    state: np.ndarray,
    control: np.ndarray,
    base_model: DifferentialDriveKinematic,
    dt: float,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Domain shift 시뮬레이션 (시뮬레이션 → 실제)

    실제 로봇에서는:
    - 마찰 증가
    - 액추에이터 지연
    - 측정 노이즈
    - 비선형성

    Args:
        state: (nx,) 현재 상태
        control: (nu,) 제어 입력
        base_model: 기본 물리 모델
        dt: 시간 간격
        noise_std: 노이즈 표준편차

    Returns:
        next_state: (nx,) 다음 상태 (domain shift 포함)
    """
    # 기본 동역학
    next_state = base_model.step(state, control, dt)

    # Domain shift effects:

    # 1. 마찰 증가 (속도 감쇠)
    friction_factor = 0.95
    # theta는 유지, 위치만 마찰 적용
    next_state[:2] = state[:2] + (next_state[:2] - state[:2]) * friction_factor

    # 2. 액추에이터 bias (제어 입력에 일정한 bias)
    actuator_bias = np.array([0.05, 0.02])  # v, omega bias
    biased_control = control + actuator_bias
    biased_next_state = base_model.step(state, biased_control, dt)
    next_state = (next_state + biased_next_state) / 2  # 혼합

    # 3. 측정 노이즈
    measurement_noise = np.random.normal(0, noise_std, next_state.shape)
    next_state += measurement_noise

    return next_state


def online_learning_simulation(
    trainer: NeuralNetworkTrainer,
    duration: float = 60.0,
    trajectory_type: str = "circle",
):
    """
    Step 3: 온라인 학습 시뮬레이션

    실시간으로 데이터를 수집하고 모델을 업데이트하면서 제어.
    """
    print("\n" + "="*60)
    print("Step 3: Online Learning Simulation")
    print("="*60)

    # Models
    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    neural_model = NeuralDynamics(state_dim=3, control_dim=2)
    neural_model.load_model("models/learned_models/online_initial_model.pth")

    # Online learner
    online_learner = OnlineLearner(
        model=neural_model,
        trainer=trainer,
        buffer_size=500,
        batch_size=32,
        min_samples_for_update=50,
        update_interval=100,  # 100 samples마다 업데이트
        verbose=True,
    )

    # MPPI params
    params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )

    controller = MPPIController(neural_model, params)

    trajectory_fn = create_trajectory_function(trajectory_type)

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    # Simulation
    initial_state = trajectory_fn(0.0)
    current_state = initial_state.copy()

    num_steps = int(duration / params.dt)
    print(f"\nRunning online learning simulation: {duration}s ({num_steps} steps)")

    # History
    history = {
        "time": [],
        "state": [],
        "control": [],
        "reference": [],
        "model_error": [],
        "num_updates": [],
        "buffer_size": [],
    }

    for step in range(num_steps):
        t = step * params.dt
        ref_traj = reference_fn(t)

        # Compute control with current model
        control, info = controller.compute_control(current_state, ref_traj)

        # Step environment (with domain shift!)
        next_state = simulate_domain_shift(
            current_state, control, base_model, params.dt, noise_std=0.02
        )

        # Add sample to online learner
        online_learner.add_sample(current_state, control, next_state, params.dt)

        # Model prediction error
        predicted_state_dot = neural_model.forward_dynamics(current_state, control)
        actual_state_dot = (next_state - current_state) / params.dt
        model_error = np.linalg.norm(predicted_state_dot - actual_state_dot)

        # Record history
        history["time"].append(t)
        history["state"].append(current_state.copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_traj[0].copy())
        history["model_error"].append(model_error)
        history["num_updates"].append(online_learner.performance_history["num_updates"])
        history["buffer_size"].append(len(online_learner.buffer))

        current_state = next_state

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{num_steps} | "
                  f"Model error: {model_error:.6f} | "
                  f"Updates: {online_learner.performance_history['num_updates']} | "
                  f"Buffer: {len(online_learner.buffer)}")

    # Convert to numpy
    for key in history.keys():
        history[key] = np.array(history[key])

    print(f"\nOnline learning completed")
    print(f"  Total updates: {online_learner.performance_history['num_updates']}")
    print(f"  Final buffer size: {len(online_learner.buffer)}")

    return history, online_learner


def plot_online_learning_results(history: dict, online_learner: OnlineLearner):
    """온라인 학습 결과 플롯"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    time_arr = history["time"]
    states = history["state"]
    references = history["reference"]
    controls = history["control"]
    model_errors = history["model_error"]
    num_updates = history["num_updates"]
    buffer_sizes = history["buffer_size"]

    # 1. XY Trajectory
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(states[:, 0], states[:, 1], label="Actual", linewidth=2, color='blue')
    ax1.plot(references[:, 0], references[:, 1], 'r--', label="Reference", linewidth=1, alpha=0.7)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("XY Trajectory (with Online Adaptation)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Position Error
    ax2 = fig.add_subplot(gs[0, 2])
    pos_errors = np.linalg.norm(states[:, :2] - references[:, :2], axis=1)
    ax2.plot(time_arr, pos_errors, linewidth=2, color='orange')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position Error (m)")
    ax2.set_title("Tracking Error")
    ax2.grid(True, alpha=0.3)

    # 3. Model Error (Prediction Error)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_arr, model_errors, linewidth=2, color='red')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Model Error (state_dot)")
    ax3.set_title("Model Prediction Error")
    ax3.grid(True, alpha=0.3)

    # 4. Number of Updates
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_arr, num_updates, linewidth=2, color='green')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Num Updates")
    ax4.set_title("Online Learning Updates")
    ax4.grid(True, alpha=0.3)

    # 5. Buffer Size
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(time_arr, buffer_sizes, linewidth=2, color='purple')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Buffer Size")
    ax5.set_title("Data Buffer Size")
    ax5.grid(True, alpha=0.3)

    # 6. Control Inputs
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(time_arr, controls[:, 0], label='v (linear)', linewidth=2)
    ax6.plot(time_arr, controls[:, 1], label='ω (angular)', linewidth=2)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Control")
    ax6.set_title("Control Inputs")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Model Error Distribution (Before/After)
    ax7 = fig.add_subplot(gs[2, 1])
    split_idx = len(model_errors) // 2
    before_errors = model_errors[:split_idx]
    after_errors = model_errors[split_idx:]

    ax7.hist([before_errors, after_errors], bins=30, label=['Before Adaptation', 'After Adaptation'],
             alpha=0.7, color=['red', 'green'])
    ax7.set_xlabel("Model Error")
    ax7.set_ylabel("Frequency")
    ax7.set_title("Model Error Distribution")
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Performance Summary (Text)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    summary_text = f"""
Online Learning Summary
━━━━━━━━━━━━━━━━━━━━━━━━

Total Updates: {online_learner.performance_history['num_updates']}

Buffer Size: {len(online_learner.buffer)}

Total Samples: {online_learner.buffer.num_samples}

Avg Model Error:
  Before: {np.mean(before_errors):.6f}
  After:  {np.mean(after_errors):.6f}
  Improvement: {(np.mean(before_errors) - np.mean(after_errors)) / np.mean(before_errors) * 100:.1f}%

Avg Tracking Error:
  {np.mean(pos_errors):.4f} m

Final Val Loss:
  {online_learner.performance_history['validation_losses'][-1]:.6f if len(online_learner.performance_history['validation_losses']) > 0 else 'N/A'}
"""

    ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle("Online Learning with Domain Adaptation", fontsize=14, fontweight='bold')

    # Save
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/online_learning_results.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to plots/online_learning_results.png")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Online Learning Demo")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--collect-initial", action="store_true")
    parser.add_argument("--train-initial", action="store_true")
    parser.add_argument("--online-learning", action="store_true")
    parser.add_argument("--trajectory", type=str, default="circle")
    parser.add_argument("--initial-duration", type=float, default=10.0)
    parser.add_argument("--online-duration", type=float, default=60.0)
    parser.add_argument("--initial-epochs", type=int, default=50)
    parser.add_argument("--no-plot", action="store_true", help="Disable plot display")

    args = parser.parse_args()

    # Create directories
    Path("data/learned_models").mkdir(parents=True, exist_ok=True)
    Path("models/learned_models").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)

    collector = None
    trainer = None

    # Step 1: Collect initial data
    if args.collect_initial or args.all:
        collector = collect_initial_data(
            duration=args.initial_duration,
            trajectory_type=args.trajectory
        )

    # Step 2: Train initial model
    if args.train_initial or args.all:
        if collector is None:
            collector = DataCollector(state_dim=3, control_dim=2)
            try:
                collector.load("online_initial_data.pkl")
            except:
                print("Initial data not found. Collecting...")
                collector = collect_initial_data(duration=args.initial_duration)

        trainer = train_initial_model(collector, epochs=args.initial_epochs)

    # Step 3: Online learning
    if args.online_learning or args.all:
        if trainer is None:
            trainer = NeuralNetworkTrainer(state_dim=3, control_dim=2)
            trainer.load_model("online_initial_model.pth")

        history, online_learner = online_learning_simulation(
            trainer,
            duration=args.online_duration,
            trajectory_type=args.trajectory
        )

        # Plot results
        plot_online_learning_results(history, online_learner)


if __name__ == "__main__":
    main()
