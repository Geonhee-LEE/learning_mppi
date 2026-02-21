#!/usr/bin/env python3
"""
MPPI Swerve Drive (Omnidirectional) Demo

Holonomic robot trajectory tracking with MPPI.
Swerve drive can translate and rotate independently.

Usage:
    python mppi_swerve_drive_demo.py --trajectory circle --duration 20
    python mppi_swerve_drive_demo.py --trajectory figure8 --live
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.swerve_drive_kinematic import SwerveDriveKinematic
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.visualizer import SimulationVisualizer
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


def main():
    parser = argparse.ArgumentParser(description="MPPI Swerve Drive Demo")
    parser.add_argument(
        "--trajectory", type=str, default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("MPPI Swerve Drive (Omnidirectional) Demo".center(60))
    print("=" * 60)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Live Mode:  {args.live}")
    print("=" * 60 + "\n")

    # Model: Swerve kinematic (3D state, 3D control)
    model = SwerveDriveKinematic(vx_max=1.0, vy_max=1.0, omega_max=1.0)

    # MPPI params (same state_dim=3 as DD kinematic)
    params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
        device="cpu",
    )

    controller = MPPIController(model, params)
    simulator = Simulator(model, controller, params.dt)

    # Reference trajectory (3D, matches state_dim)
    trajectory_fn = create_trajectory_function(args.trajectory)

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    initial_state = trajectory_fn(0.0)
    simulator.reset(initial_state)

    print(f"Initial: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}, "
          f"theta={np.rad2deg(initial_state[2]):.1f} deg\n")

    if args.live:
        visualizer = SimulationVisualizer()
        visualizer.animate_live(simulator, reference_fn, args.duration)
    else:
        history = simulator.run(reference_fn, args.duration, realtime=False)
        print(f"Completed. Steps: {len(history['time'])}\n")

        metrics = compute_metrics(history)
        print_metrics(metrics, title="Swerve Drive MPPI Performance")

        print("\n" + "=" * 60)
        rmse = metrics["position_rmse"]
        status = "PASS" if rmse < 0.2 else "FAIL"
        print(f"Position RMSE: {rmse:.4f}m [{status}]")
        print(f"Mean Solve Time: {metrics['mean_solve_time']:.2f}ms")
        print("=" * 60 + "\n")

        visualizer = SimulationVisualizer()
        fig = visualizer.plot_results(
            history, metrics,
            title=f"Swerve Drive MPPI - {args.trajectory.capitalize()}",
        )

        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
