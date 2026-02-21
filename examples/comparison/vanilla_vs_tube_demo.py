#!/usr/bin/env python3
"""
Vanilla MPPI vs Tube-MPPI 비교 데모

외란 환경에서 두 컨트롤러의 강건성 비교.

Usage:
    python vanilla_vs_tube_demo.py --trajectory circle --noise 0.05 --duration 30
    python vanilla_vs_tube_demo.py --trajectory figure8 --noise 0.1
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, TubeMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
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
    parser = argparse.ArgumentParser(description="Vanilla vs Tube-MPPI Comparison")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--noise", type=float, default=0.05, help="Process noise std (m)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Simulation duration (s)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("Vanilla MPPI vs Tube-MPPI Comparison".center(80))
    print("=" * 80)
    print(f"Trajectory: {args.trajectory}")
    print(f"Process Noise: {args.noise}m (외란 강도)")
    print(f"Duration: {args.duration}s")
    print("=" * 80 + "\n")

    # ==================== 1. Vanilla MPPI ====================
    print("Setting up Vanilla MPPI...")
    vanilla_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    vanilla_params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )
    vanilla_controller = MPPIController(vanilla_model, vanilla_params)

    # 외란 주입
    process_noise_std = np.array([args.noise, args.noise, args.noise * 0.1])
    vanilla_sim = Simulator(
        vanilla_model, vanilla_controller, vanilla_params.dt, process_noise_std
    )

    trajectory_fn = create_trajectory_function(args.trajectory)

    def reference_fn(t):
        return generate_reference_trajectory(
            trajectory_fn, t, vanilla_params.N, vanilla_params.dt
        )

    initial_state = trajectory_fn(0.0)
    vanilla_sim.reset(initial_state)
    print(f"  Initial: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}")
    print(f"  Process noise: {process_noise_std}")
    print("  Running simulation...")
    vanilla_history = vanilla_sim.run(reference_fn, args.duration, realtime=False)
    vanilla_metrics = compute_metrics(vanilla_history)
    print("  ✓ Vanilla MPPI completed\n")

    # ==================== 2. Tube-MPPI ====================
    print("Setting up Tube-MPPI...")
    tube_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    tube_params = TubeMPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
        tube_enabled=True,
        tube_margin=0.1,
    )
    tube_controller = TubeMPPIController(tube_model, tube_params)

    tube_sim = Simulator(
        tube_model, tube_controller, tube_params.dt, process_noise_std
    )

    tube_sim.reset(initial_state)
    print(f"  Tube enabled: {tube_params.tube_enabled}")
    print(f"  Tube margin: {tube_params.tube_margin}m")
    print("  Running simulation...")
    tube_history = tube_sim.run(reference_fn, args.duration, realtime=False)
    tube_metrics = compute_metrics(tube_history)
    print("  ✓ Tube-MPPI completed\n")

    # Tube 통계
    tube_stats = tube_controller.get_tube_statistics()
    print("=" * 60)
    print("Tube-MPPI Statistics".center(60))
    print("=" * 60)
    print(f"Mean Tube Width: {tube_stats['mean_tube_width']:.4f} m")
    print(f"Max Tube Width:  {tube_stats['max_tube_width']:.4f} m")
    print("=" * 60 + "\n")

    # ==================== 비교 ====================
    print_metrics(vanilla_metrics, title="Vanilla MPPI (No Tube)")
    print_metrics(tube_metrics, title="Tube-MPPI (With Feedback)")

    compare_metrics(
        [vanilla_metrics, tube_metrics],
        ["Vanilla", "Tube"],
        title="Vanilla vs Tube-MPPI Comparison",
    )

    # 강건성 개선율
    rmse_improvement = (
        (vanilla_metrics["position_rmse"] - tube_metrics["position_rmse"])
        / vanilla_metrics["position_rmse"]
        * 100
    )
    print(f"\n{'=' * 60}")
    print(f"Robustness Improvement (Tube over Vanilla)".center(60))
    print(f"{'=' * 60}")
    print(f"Position RMSE: {rmse_improvement:+.1f}% improvement")
    print(f"{'=' * 60}\n")

    # ==================== 시각화 ====================
    print("Generating comparison plots...\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Vanilla vs Tube-MPPI - {args.trajectory.capitalize()} "
        f"(Noise={args.noise}m)",
        fontsize=16,
    )

    vanilla_states = vanilla_history["state"]
    tube_states = tube_history["state"]
    vanilla_refs = vanilla_history["reference"]
    vanilla_times = vanilla_history["time"]
    tube_times = tube_history["time"]

    # 1. XY 궤적 비교
    ax = axes[0, 0]
    ax.plot(
        vanilla_states[:, 0],
        vanilla_states[:, 1],
        "b-",
        label="Vanilla",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        tube_states[:, 0], tube_states[:, 1], "g-", label="Tube", linewidth=2, alpha=0.7
    )
    ax.plot(
        vanilla_refs[:, 0],
        vanilla_refs[:, 1],
        "r--",
        label="Reference",
        linewidth=2,
        alpha=0.5,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 위치 오차 비교
    ax = axes[0, 1]
    vanilla_errors = np.linalg.norm(vanilla_states[:, :2] - vanilla_refs[:, :2], axis=1)
    tube_errors = np.linalg.norm(tube_states[:, :2] - vanilla_refs[:, :2], axis=1)

    ax.plot(vanilla_times, vanilla_errors, "b-", label="Vanilla", linewidth=2)
    ax.plot(tube_times, tube_errors, "g-", label="Tube", linewidth=2)
    ax.axhline(
        y=vanilla_metrics["position_rmse"],
        color="b",
        linestyle="--",
        alpha=0.5,
        label=f'Vanilla RMSE={vanilla_metrics["position_rmse"]:.3f}m',
    )
    ax.axhline(
        y=tube_metrics["position_rmse"],
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f'Tube RMSE={tube_metrics["position_rmse"]:.3f}m',
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Tube Width 히스토리
    ax = axes[0, 2]
    tube_widths = tube_stats["tube_width_history"]
    ax.plot(tube_times, tube_widths, "g-", linewidth=2)
    ax.axhline(
        y=tube_stats["mean_tube_width"],
        color="orange",
        linestyle="--",
        label=f'Mean={tube_stats["mean_tube_width"]:.3f}m',
    )
    ax.axhline(
        y=tube_params.tube_margin,
        color="r",
        linestyle="--",
        label=f"Target Margin={tube_params.tube_margin}m",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tube Width (m)")
    ax.set_title("Tube Width ||x - x_nominal||")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 제어 입력 비교
    ax = axes[1, 0]
    vanilla_controls = vanilla_history["control"]
    tube_controls = tube_history["control"]

    ax.plot(vanilla_times, vanilla_controls[:, 0], "b-", label="Vanilla v", linewidth=2)
    ax.plot(tube_times, tube_controls[:, 0], "g-", label="Tube v", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_title("Control Inputs (v)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. RMSE 바 차트
    ax = axes[1, 1]
    models = ["Vanilla", "Tube"]
    rmses = [vanilla_metrics["position_rmse"], tube_metrics["position_rmse"]]
    colors = ["blue", "green"]
    bars = ax.bar(models, rmses, color=colors, alpha=0.7)
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rmse:.4f}m",
            ha="center",
            va="bottom",
        )

    # 6. 요약
    ax = axes[1, 2]
    ax.axis("off")

    summary_text = f"""
    Comparison Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Vanilla MPPI:
      RMSE: {vanilla_metrics['position_rmse']:.4f} m
      Time: {vanilla_metrics['mean_solve_time']:.2f} ms

    Tube-MPPI:
      RMSE: {tube_metrics['position_rmse']:.4f} m
      Time: {tube_metrics['mean_solve_time']:.2f} ms
      Mean Tube Width: {tube_stats['mean_tube_width']:.4f} m
      Max Tube Width: {tube_stats['max_tube_width']:.4f} m

    Improvement:
      RMSE: {rmse_improvement:+.1f}%

    Settings:
      Trajectory: {args.trajectory.capitalize()}
      Noise: {args.noise}m
      Duration: {args.duration}s
    """

    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=9,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig("vanilla_vs_tube_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: vanilla_vs_tube_comparison.png")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
