#!/usr/bin/env python3
"""
Physics vs Learned 모델 비교 데모

기구학, 동역학, Residual 동역학 3-way 비교.

Usage:
    python physics_vs_learned_demo.py --trajectory circle --duration 30
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.models.learned.residual_dynamics import (
    ResidualDynamics,
    create_constant_residual,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import (
    compute_metrics,
    print_metrics,
    compare_metrics,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Physics vs Learned Model Comparison")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="Simulation duration (s)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("Physics vs Learned Model Comparison (3-way)".center(80))
    print("=" * 80)
    print(f"Trajectory: {args.trajectory}")
    print(f"Duration: {args.duration}s")
    print("=" * 80 + "\n")

    # ==================== 1. 기구학 모델 ====================
    print("Setting up Kinematic Model...")
    kinematic_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    kinematic_params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )
    kinematic_controller = MPPIController(kinematic_model, kinematic_params)
    kinematic_sim = Simulator(kinematic_model, kinematic_controller, kinematic_params.dt)

    trajectory_fn_kinematic = create_trajectory_function(args.trajectory)
    def kinematic_reference_fn(t):
        return generate_reference_trajectory(
            trajectory_fn_kinematic, t, kinematic_params.N, kinematic_params.dt
        )

    initial_state_kinematic = trajectory_fn_kinematic(0.0)
    kinematic_sim.reset(initial_state_kinematic)
    print(f"  Initial: x={initial_state_kinematic[0]:.2f}, y={initial_state_kinematic[1]:.2f}")
    print("  Running simulation...")
    kinematic_history = kinematic_sim.run(kinematic_reference_fn, args.duration, realtime=False)
    kinematic_metrics = compute_metrics(kinematic_history)
    print("  ✓ Kinematic completed\n")

    # ==================== 2. Residual 동역학 모델 ====================
    print("Setting up Residual Dynamics Model (Kinematic + Learned)...")

    # 더미 residual: 슬립 효과 모방
    residual_value = np.array([0.01, 0.01, 0.0])  # 작은 drift
    residual_fn = create_constant_residual(residual_value)

    residual_model = ResidualDynamics(
        base_model=DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
        residual_fn=residual_fn,
        use_residual=True,
    )

    residual_params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )
    residual_controller = MPPIController(residual_model, residual_params)
    residual_sim = Simulator(residual_model, residual_controller, residual_params.dt)

    def residual_reference_fn(t):
        return generate_reference_trajectory(
            trajectory_fn_kinematic, t, residual_params.N, residual_params.dt
        )

    initial_state_residual = trajectory_fn_kinematic(0.0)
    residual_sim.reset(initial_state_residual)
    print(f"  Residual: {residual_value}")
    print("  Running simulation...")
    residual_history = residual_sim.run(residual_reference_fn, args.duration, realtime=False)
    residual_metrics = compute_metrics(residual_history)
    print("  ✓ Residual completed\n")

    # ==================== 3. 동역학 모델 (참조) ====================
    print("Setting up Dynamic Model...")
    dynamic_model = DifferentialDriveDynamic(
        mass=10.0, inertia=1.0, c_v=0.1, c_omega=0.1,
        a_max=2.0, alpha_max=2.0, v_max=2.0, omega_max=2.0,
    )
    dynamic_params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([1.0, 1.0]),
        Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2, 0.2]),
    )
    dynamic_controller = MPPIController(dynamic_model, dynamic_params)
    dynamic_sim = Simulator(dynamic_model, dynamic_controller, dynamic_params.dt)

    def trajectory_fn_dynamic(t):
        kinematic_state = trajectory_fn_kinematic(t)
        if args.trajectory == "circle":
            v_ref, omega_ref = 0.5, 0.1
        else:
            v_ref, omega_ref = 0.5, 0.0
        return np.array([
            kinematic_state[0], kinematic_state[1], kinematic_state[2],
            v_ref, omega_ref,
        ])

    def dynamic_reference_fn(t):
        return generate_reference_trajectory(
            trajectory_fn_dynamic, t, dynamic_params.N, dynamic_params.dt
        )

    initial_state_dynamic = trajectory_fn_dynamic(0.0)
    dynamic_sim.reset(initial_state_dynamic)
    print(f"  Initial: x={initial_state_dynamic[0]:.2f}, y={initial_state_dynamic[1]:.2f}")
    print("  Running simulation...")
    dynamic_history = dynamic_sim.run(dynamic_reference_fn, args.duration, realtime=False)
    dynamic_metrics = compute_metrics(dynamic_history)
    print("  ✓ Dynamic completed\n")

    # ==================== 비교 ====================
    print_metrics(kinematic_metrics, title="Kinematic (Pure Physics)")
    print_metrics(residual_metrics, title="Residual (Physics + Learned)")
    print_metrics(dynamic_metrics, title="Dynamic (Physics with Friction)")

    compare_metrics(
        [kinematic_metrics, residual_metrics, dynamic_metrics],
        ["Kinematic", "Residual", "Dynamic"],
        title="Physics vs Learned Comparison",
    )

    # ==================== 시각화 ====================
    print("Generating comparison plots...\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Physics vs Learned - {args.trajectory.capitalize()}", fontsize=16)

    # 1. XY 궤적 비교
    ax = axes[0, 0]
    kinematic_states = kinematic_history["state"]
    residual_states = residual_history["state"]
    dynamic_states = dynamic_history["state"]
    kinematic_refs = kinematic_history["reference"]

    ax.plot(kinematic_states[:, 0], kinematic_states[:, 1], "b-", label="Kinematic", linewidth=2)
    ax.plot(residual_states[:, 0], residual_states[:, 1], "g-", label="Residual", linewidth=2)
    ax.plot(dynamic_states[:, 0], dynamic_states[:, 1], "m-", label="Dynamic", linewidth=2)
    ax.plot(kinematic_refs[:, 0], kinematic_refs[:, 1], "r--", label="Reference", linewidth=2, alpha=0.7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 위치 오차 비교
    ax = axes[0, 1]
    kinematic_times = kinematic_history["time"]
    residual_times = residual_history["time"]
    dynamic_times = dynamic_history["time"]

    kinematic_errors = np.linalg.norm(kinematic_states[:, :2] - kinematic_refs[:, :2], axis=1)
    residual_errors = np.linalg.norm(residual_states[:, :2] - kinematic_refs[:, :2], axis=1)
    dynamic_errors = np.linalg.norm(dynamic_states[:, :2] - dynamic_states[:, :2], axis=1)

    ax.plot(kinematic_times, kinematic_errors, "b-", label="Kinematic", linewidth=2)
    ax.plot(residual_times, residual_errors, "g-", label="Residual", linewidth=2)
    ax.plot(dynamic_times, dynamic_errors, "m-", label="Dynamic", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. RMSE 바 차트
    ax = axes[0, 2]
    models = ["Kinematic", "Residual", "Dynamic"]
    rmses = [
        kinematic_metrics["position_rmse"],
        residual_metrics["position_rmse"],
        dynamic_metrics["position_rmse"],
    ]
    colors = ["blue", "green", "magenta"]
    bars = ax.bar(models, rmses, color=colors, alpha=0.7)
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    # 값 표시
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.4f}m', ha='center', va='bottom')

    # 4. 제어 입력 비교
    ax = axes[1, 0]
    kinematic_controls = kinematic_history["control"]
    residual_controls = residual_history["control"]
    dynamic_controls = dynamic_history["control"]

    ax.plot(kinematic_times, kinematic_controls[:, 0], "b-", label="Kinematic v", linewidth=2)
    ax.plot(residual_times, residual_controls[:, 0], "g-", label="Residual v", linewidth=2)
    ax.plot(dynamic_times, dynamic_controls[:, 0], "m-", label="Dynamic a", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input (Linear)")
    ax.set_title("Control Inputs Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 계산 시간 비교
    ax = axes[1, 1]
    kinematic_solve_times = kinematic_history["solve_time"] * 1000
    residual_solve_times = residual_history["solve_time"] * 1000
    dynamic_solve_times = dynamic_history["solve_time"] * 1000

    ax.plot(kinematic_times, kinematic_solve_times, "b-", label="Kinematic", linewidth=2)
    ax.plot(residual_times, residual_solve_times, "g-", label="Residual", linewidth=2)
    ax.plot(dynamic_times, dynamic_solve_times, "m-", label="Dynamic", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 메트릭 요약
    ax = axes[1, 2]
    ax.axis("off")

    summary_text = f"""
    3-Way Comparison Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Kinematic (Pure Physics):
      RMSE: {kinematic_metrics['position_rmse']:.4f} m
      Time: {kinematic_metrics['mean_solve_time']:.2f} ms

    Residual (Physics + Learned):
      RMSE: {residual_metrics['position_rmse']:.4f} m
      Time: {residual_metrics['mean_solve_time']:.2f} ms
      Residual: {residual_value}

    Dynamic (Physics with Friction):
      RMSE: {dynamic_metrics['position_rmse']:.4f} m
      Time: {dynamic_metrics['mean_solve_time']:.2f} ms

    Trajectory: {args.trajectory.capitalize()}
    Duration: {args.duration}s
    Samples: K={kinematic_params.K}, N={kinematic_params.N}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment="center",
            family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    plt.savefig("physics_vs_learned_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: physics_vs_learned_comparison.png")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
