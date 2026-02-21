#!/usr/bin/env python3
"""
Multi-Model Comparison Demo

Compare all robot models (Differential Drive, Ackermann, Swerve Drive)
on the same trajectory with MPPI.

Usage:
    python model_comparison_demo.py --trajectory circle --duration 20
    python model_comparison_demo.py --trajectory figure8 --duration 30
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.models.kinematic.swerve_drive_kinematic import SwerveDriveKinematic
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


def run_model(name, model, params, ref_fn, initial_state, duration):
    """Run a single model simulation and return history + metrics."""
    print(f"  Running {name}...")
    controller = MPPIController(model, params)
    sim = Simulator(model, controller, params.dt)
    sim.reset(initial_state)
    history = sim.run(ref_fn, duration, realtime=False)
    metrics = compute_metrics(history)
    print(f"    RMSE: {metrics['position_rmse']:.4f}m, "
          f"Solve: {metrics['mean_solve_time']:.2f}ms")
    return history, metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Comparison Demo")
    parser.add_argument(
        "--trajectory", type=str, default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("Multi-Model Comparison: DD vs Ackermann vs Swerve".center(70))
    print("=" * 70)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print("=" * 70 + "\n")

    # Shared reference trajectory (3D)
    trajectory_fn = create_trajectory_function(args.trajectory)
    N = 30
    dt = 0.05

    # ─── 1. Differential Drive Kinematic ───
    dd_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    dd_params = MPPIParams(
        N=N, dt=dt, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )
    dd_initial = trajectory_fn(0.0)

    def dd_ref_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, N, dt)

    dd_history, dd_metrics = run_model(
        "Differential Drive", dd_model, dd_params, dd_ref_fn, dd_initial, args.duration
    )

    # ─── 2. Ackermann Kinematic ───
    ack_model = AckermannKinematic(wheelbase=0.5, v_max=1.0, max_steer=0.5)
    ack_params = MPPIParams(
        N=N, dt=dt, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0, 0.1]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2]),
    )

    def ack_traj_fn(t):
        s3 = trajectory_fn(t)
        return np.array([s3[0], s3[1], s3[2], 0.0])

    ack_initial = ack_traj_fn(0.0)

    def ack_ref_fn(t):
        return generate_reference_trajectory(ack_traj_fn, t, N, dt)

    ack_history, ack_metrics = run_model(
        "Ackermann", ack_model, ack_params, ack_ref_fn, ack_initial, args.duration
    )

    # ─── 3. Swerve Drive Kinematic ───
    swerve_model = SwerveDriveKinematic(vx_max=1.0, vy_max=1.0, omega_max=1.0)
    swerve_params = MPPIParams(
        N=N, dt=dt, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )
    swerve_initial = trajectory_fn(0.0)

    def swerve_ref_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, N, dt)

    swerve_history, swerve_metrics = run_model(
        "Swerve Drive", swerve_model, swerve_params, swerve_ref_fn,
        swerve_initial, args.duration
    )

    # ─── Comparison Table ───
    print()
    compare_metrics(
        [dd_metrics, ack_metrics, swerve_metrics],
        ["Diff. Drive", "Ackermann", "Swerve Drive"],
        title="Model Comparison",
    )

    # ─── Visualization ───
    print("\nGenerating comparison plots...\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Model Comparison - {args.trajectory.capitalize()} Trajectory",
        fontsize=16,
    )

    colors = {"DD": "tab:blue", "Ack": "tab:orange", "Swerve": "tab:green"}

    # 1. XY Trajectory
    ax = axes[0, 0]
    dd_s = dd_history["state"]
    ack_s = ack_history["state"]
    swerve_s = swerve_history["state"]
    ref_s = dd_history["reference"]

    ax.plot(dd_s[:, 0], dd_s[:, 1], color=colors["DD"], label="Diff. Drive", lw=2)
    ax.plot(ack_s[:, 0], ack_s[:, 1], color=colors["Ack"], label="Ackermann", lw=2)
    ax.plot(swerve_s[:, 0], swerve_s[:, 1], color=colors["Swerve"], label="Swerve", lw=2)
    ax.plot(ref_s[:, 0], ref_s[:, 1], "r--", label="Reference", lw=2, alpha=0.7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. Position Error
    ax = axes[0, 1]
    dd_t = dd_history["time"]
    ack_t = ack_history["time"]
    swerve_t = swerve_history["time"]

    dd_err = np.linalg.norm(dd_s[:, :2] - ref_s[:, :2], axis=1)
    ack_ref = ack_history["reference"]
    ack_err = np.linalg.norm(ack_s[:, :2] - ack_ref[:, :2], axis=1)
    swerve_ref = swerve_history["reference"]
    swerve_err = np.linalg.norm(swerve_s[:, :2] - swerve_ref[:, :2], axis=1)

    ax.plot(dd_t, dd_err, color=colors["DD"], label="Diff. Drive", lw=2)
    ax.plot(ack_t, ack_err, color=colors["Ack"], label="Ackermann", lw=2)
    ax.plot(swerve_t, swerve_err, color=colors["Swerve"], label="Swerve", lw=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Solve Time
    ax = axes[1, 0]
    ax.plot(dd_t, dd_history["solve_time"] * 1000, color=colors["DD"], label="DD", lw=1.5)
    ax.plot(ack_t, ack_history["solve_time"] * 1000, color=colors["Ack"], label="Ack", lw=1.5)
    ax.plot(swerve_t, swerve_history["solve_time"] * 1000, color=colors["Swerve"], label="Swerve", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary Table
    ax = axes[1, 1]
    ax.axis("off")
    summary = f"""
    Comparison Summary
    {'='*45}

    Differential Drive:
      RMSE: {dd_metrics['position_rmse']:.4f} m
      Solve: {dd_metrics['mean_solve_time']:.2f} ms
      State dim: {dd_model.state_dim}, Ctrl dim: {dd_model.control_dim}

    Ackermann (Bicycle):
      RMSE: {ack_metrics['position_rmse']:.4f} m
      Solve: {ack_metrics['mean_solve_time']:.2f} ms
      State dim: {ack_model.state_dim}, Ctrl dim: {ack_model.control_dim}

    Swerve Drive (Omni):
      RMSE: {swerve_metrics['position_rmse']:.4f} m
      Solve: {swerve_metrics['mean_solve_time']:.2f} ms
      State dim: {swerve_model.state_dim}, Ctrl dim: {swerve_model.control_dim}

    Trajectory: {args.trajectory}, Duration: {args.duration}s
    K={dd_params.K}, N={N}
    """
    ax.text(
        0.05, 0.5, summary, fontsize=10, va="center", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.show()

    print("Comparison complete!")


if __name__ == "__main__":
    main()
