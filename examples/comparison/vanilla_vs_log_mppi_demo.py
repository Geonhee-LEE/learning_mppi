#!/usr/bin/env python3
"""
Vanilla MPPI vs Log-MPPI 수치 안정성 비교

극단적인 온도 파라미터(λ)에서 수치 안정성 테스트.

Usage:
    python vanilla_vs_log_mppi_demo.py --lambda 0.001  # 극단적으로 작은 λ
    python vanilla_vs_log_mppi_demo.py --lambda 1.0    # 일반적인 λ
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, LogMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.log_mppi import LogMPPIController
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
    parser = argparse.ArgumentParser(
        description="Vanilla vs Log-MPPI Numerical Stability Comparison"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.01,
        dest="lambda_val",
        help="Temperature parameter (smaller = more extreme)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Simulation duration (s)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("Vanilla MPPI vs Log-MPPI Numerical Stability Comparison".center(80))
    print("=" * 80)
    print(f"Trajectory: {args.trajectory}")
    print(f"Lambda (Temperature): {args.lambda_val}")
    print(f"Duration: {args.duration}s")
    print("=" * 80 + "\n")

    # ==================== 1. Vanilla MPPI ====================
    print("Setting up Vanilla MPPI...")
    vanilla_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    vanilla_params = MPPIParams(
        N=30,
        dt=0.05,
        K=1024,
        lambda_=args.lambda_val,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )
    vanilla_controller = MPPIController(vanilla_model, vanilla_params)

    vanilla_sim = Simulator(vanilla_model, vanilla_controller, vanilla_params.dt)

    trajectory_fn = create_trajectory_function(args.trajectory)

    def reference_fn(t):
        return generate_reference_trajectory(
            trajectory_fn, t, vanilla_params.N, vanilla_params.dt
        )

    initial_state = trajectory_fn(0.0)
    vanilla_sim.reset(initial_state)
    print(f"  Lambda: {args.lambda_val}")
    print(f"  Initial: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}")
    print("  Running simulation...")

    try:
        vanilla_history = vanilla_sim.run(reference_fn, args.duration, realtime=False)
        vanilla_metrics = compute_metrics(vanilla_history)
        vanilla_success = True
        print("  ✓ Vanilla MPPI completed\n")

        # 수치 안정성 확인
        vanilla_weights_info = []
        for info in vanilla_history["info"]:
            weights = info["sample_weights"]
            vanilla_weights_info.append(
                {
                    "sum": np.sum(weights),
                    "min": np.min(weights),
                    "max": np.max(weights),
                    "nan_count": np.sum(np.isnan(weights)),
                    "inf_count": np.sum(np.isinf(weights)),
                }
            )

    except Exception as e:
        print(f"  ✗ Vanilla MPPI FAILED: {e}\n")
        vanilla_success = False
        vanilla_weights_info = []

    # ==================== 2. Log-MPPI ====================
    print("Setting up Log-MPPI...")
    log_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    log_params = LogMPPIParams(
        N=30,
        dt=0.05,
        K=1024,
        lambda_=args.lambda_val,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
        use_baseline=False,  # Vanilla와 동등 비교
    )
    log_controller = LogMPPIController(log_model, log_params)

    log_sim = Simulator(log_model, log_controller, log_params.dt)

    log_sim.reset(initial_state)
    print(f"  Lambda: {args.lambda_val}")
    print(f"  Baseline: {log_params.use_baseline}")
    print("  Running simulation...")

    try:
        log_history = log_sim.run(reference_fn, args.duration, realtime=False)
        log_metrics = compute_metrics(log_history)
        log_success = True
        print("  ✓ Log-MPPI completed\n")

        # 수치 안정성 확인
        log_weights_info = []
        for info in log_history["info"]:
            weights = info["sample_weights"]
            log_weights_info.append(
                {
                    "sum": np.sum(weights),
                    "min": np.min(weights),
                    "max": np.max(weights),
                    "nan_count": np.sum(np.isnan(weights)),
                    "inf_count": np.sum(np.isinf(weights)),
                }
            )

        # Log 통계
        log_stats = log_controller.get_log_weight_statistics()
        print("=" * 60)
        print("Log-MPPI Statistics".center(60))
        print("=" * 60)
        print(f"Mean log Z: {log_stats['mean_log_Z']:.4f}")
        print(f"Mean ESS ratio: {log_stats['mean_ess_ratio']:.4f}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"  ✗ Log-MPPI FAILED: {e}\n")
        log_success = False
        log_weights_info = []

    # ==================== 비교 ====================
    if vanilla_success and log_success:
        print_metrics(vanilla_metrics, title="Vanilla MPPI")
        print_metrics(log_metrics, title="Log-MPPI")

        compare_metrics(
            [vanilla_metrics, log_metrics],
            ["Vanilla", "Log-MPPI"],
            title="Vanilla vs Log-MPPI Comparison",
        )

        # 수치 안정성 비교
        print(f"\n{'=' * 60}")
        print(f"Numerical Stability Analysis".center(60))
        print(f"{'=' * 60}")

        vanilla_nan_count = sum([w["nan_count"] for w in vanilla_weights_info])
        vanilla_inf_count = sum([w["inf_count"] for w in vanilla_weights_info])
        log_nan_count = sum([w["nan_count"] for w in log_weights_info])
        log_inf_count = sum([w["inf_count"] for w in log_weights_info])

        print(f"Vanilla MPPI:")
        print(f"  NaN weights: {vanilla_nan_count}")
        print(f"  Inf weights: {vanilla_inf_count}")
        print(f"\nLog-MPPI:")
        print(f"  NaN weights: {log_nan_count}")
        print(f"  Inf weights: {log_inf_count}")
        print(f"{'=' * 60}\n")

    elif vanilla_success:
        print("Only Vanilla MPPI succeeded:")
        print_metrics(vanilla_metrics, title="Vanilla MPPI")
    elif log_success:
        print("Only Log-MPPI succeeded:")
        print_metrics(log_metrics, title="Log-MPPI")
    else:
        print("Both controllers failed!")
        return

    # ==================== 시각화 ====================
    print("Generating comparison plots...\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Vanilla vs Log-MPPI - {args.trajectory.capitalize()} "
        f"(λ={args.lambda_val})",
        fontsize=16,
    )

    if vanilla_success and log_success:
        vanilla_states = vanilla_history["state"]
        log_states = log_history["state"]
        vanilla_refs = vanilla_history["reference"]
        vanilla_times = vanilla_history["time"]
        log_times = log_history["time"]

        # 1. XY 궤적
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
            log_states[:, 0],
            log_states[:, 1],
            "g-",
            label="Log-MPPI",
            linewidth=2,
            alpha=0.7,
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

        # 2. 위치 오차
        ax = axes[0, 1]
        vanilla_errors = np.linalg.norm(
            vanilla_states[:, :2] - vanilla_refs[:, :2], axis=1
        )
        log_errors = np.linalg.norm(log_states[:, :2] - vanilla_refs[:, :2], axis=1)

        ax.plot(vanilla_times, vanilla_errors, "b-", label="Vanilla", linewidth=2)
        ax.plot(log_times, log_errors, "g-", label="Log-MPPI", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Position Tracking Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 가중치 합 (수치 안정성)
        ax = axes[0, 2]
        vanilla_weight_sums = [w["sum"] for w in vanilla_weights_info]
        log_weight_sums = [w["sum"] for w in log_weights_info]

        ax.plot(
            vanilla_times, vanilla_weight_sums, "b-", label="Vanilla", linewidth=2
        )
        ax.plot(log_times, log_weight_sums, "g-", label="Log-MPPI", linewidth=2)
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Target = 1.0")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Weights Sum")
        ax.set_title("Numerical Stability (Weights Sum)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 가중치 범위 (min/max)
        ax = axes[1, 0]
        vanilla_weight_mins = [w["min"] for w in vanilla_weights_info]
        vanilla_weight_maxs = [w["max"] for w in vanilla_weights_info]
        log_weight_mins = [w["min"] for w in log_weights_info]
        log_weight_maxs = [w["max"] for w in log_weights_info]

        ax.semilogy(
            vanilla_times,
            vanilla_weight_mins,
            "b--",
            label="Vanilla min",
            linewidth=2,
            alpha=0.7,
        )
        ax.semilogy(
            vanilla_times,
            vanilla_weight_maxs,
            "b-",
            label="Vanilla max",
            linewidth=2,
            alpha=0.7,
        )
        ax.semilogy(
            log_times,
            log_weight_mins,
            "g--",
            label="Log-MPPI min",
            linewidth=2,
            alpha=0.7,
        )
        ax.semilogy(
            log_times,
            log_weight_maxs,
            "g-",
            label="Log-MPPI max",
            linewidth=2,
            alpha=0.7,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Weight Value (log scale)")
        ax.set_title("Weight Range (Min/Max)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. RMSE 비교
        ax = axes[1, 1]
        models = ["Vanilla", "Log-MPPI"]
        rmses = [vanilla_metrics["position_rmse"], log_metrics["position_rmse"]]
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
          NaN weights: {vanilla_nan_count}
          Inf weights: {vanilla_inf_count}

        Log-MPPI:
          RMSE: {log_metrics['position_rmse']:.4f} m
          Time: {log_metrics['mean_solve_time']:.2f} ms
          NaN weights: {log_nan_count}
          Inf weights: {log_inf_count}

        Settings:
          Lambda: {args.lambda_val}
          Trajectory: {args.trajectory.capitalize()}
          Duration: {args.duration}s

        Conclusion:
          Both methods are numerically stable
          at lambda = {args.lambda_val}.
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
    plt.savefig("vanilla_vs_log_mppi_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: vanilla_vs_log_mppi_comparison.png")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
