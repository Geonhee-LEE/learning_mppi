#!/usr/bin/env python3
"""
MPPI Ackermann Steering Demo

Ackermann (bicycle model) trajectory tracking with MPPI.

Usage:
    python mppi_ackermann_demo.py --trajectory circle --duration 20
    python mppi_ackermann_demo.py --trajectory figure8 --live
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
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
    parser = argparse.ArgumentParser(description="MPPI Ackermann Steering Demo")
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
    print("MPPI Ackermann Steering Demo".center(60))
    print("=" * 60)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Live Mode:  {args.live}")
    print("=" * 60 + "\n")

    # Model: Ackermann kinematic (bicycle)
    model = AckermannKinematic(
        wheelbase=0.5, v_max=1.0, max_steer=0.5, steer_rate_max=1.0
    )

    # MPPI params (state_dim=4: [x,y,theta,delta])
    params = MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0, 0.1]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2]),
        device="cpu",
    )

    controller = MPPIController(model, params)
    simulator = Simulator(model, controller, params.dt)

    # Reference trajectory: 3D -> 4D (append delta=0)
    trajectory_fn_3d = create_trajectory_function(args.trajectory)

    def trajectory_fn_4d(t):
        s3 = trajectory_fn_3d(t)
        return np.array([s3[0], s3[1], s3[2], 0.0])

    def reference_fn(t):
        return generate_reference_trajectory(trajectory_fn_4d, t, params.N, params.dt)

    initial_state = trajectory_fn_4d(0.0)
    simulator.reset(initial_state)

    print(f"Initial: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}, "
          f"theta={np.rad2deg(initial_state[2]):.1f} deg, delta={initial_state[3]:.2f}\n")

    if args.live:
        visualizer = SimulationVisualizer()
        visualizer.animate_live(simulator, reference_fn, args.duration)
    else:
        history = simulator.run(reference_fn, args.duration, realtime=False)
        print(f"Completed. Steps: {len(history['time'])}\n")

        metrics = compute_metrics(history)
        print_metrics(metrics, title="Ackermann MPPI Performance")

        print("\n" + "=" * 60)
        rmse = metrics["position_rmse"]
        status = "PASS" if rmse < 0.3 else "FAIL"
        print(f"Position RMSE: {rmse:.4f}m [{status}]")
        print(f"Mean Solve Time: {metrics['mean_solve_time']:.2f}ms")
        print("=" * 60 + "\n")

        visualizer = SimulationVisualizer()
        fig = visualizer.plot_results(
            history, metrics,
            title=f"Ackermann MPPI - {args.trajectory.capitalize()}",
        )

        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
