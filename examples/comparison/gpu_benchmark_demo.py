#!/usr/bin/env python3
"""
CPU vs GPU MPPI 벤치마크

K=256/1024/4096/8192에서 solve time 비교.
GPU 결과가 CPU와 동등한 RMSE를 유지하는지 검증.

Usage:
    python gpu_benchmark_demo.py
    python gpu_benchmark_demo.py --trajectory figure8 --duration 10
    python gpu_benchmark_demo.py --no-plot
"""

import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


def check_gpu():
    """GPU 사용 가능 여부 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            print("  GPU: CUDA not available")
            return False
    except ImportError:
        print("  GPU: PyTorch not installed")
        return False


def run_benchmark(trajectory_type, duration, K_values, plot=True):
    """CPU vs GPU 벤치마크 실행"""

    print("\n" + "=" * 70)
    print("GPU Benchmark: CPU vs GPU MPPI".center(70))
    print("=" * 70)

    gpu_available = check_gpu()
    if not gpu_available:
        print("\n[!] GPU not available. Running CPU-only benchmark.")

    print(f"  Trajectory: {trajectory_type}")
    print(f"  Duration: {duration}s")
    print(f"  K values: {K_values}")
    print("=" * 70 + "\n")

    results = {}

    for K in K_values:
        print(f"\n{'─' * 50}")
        print(f"  K = {K}")
        print(f"{'─' * 50}")

        # ── CPU ──
        model_cpu = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        params_cpu = MPPIParams(
            N=30, dt=0.05, K=K, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            device="cpu",
        )
        controller_cpu = MPPIController(model_cpu, params_cpu)
        sim_cpu = Simulator(model_cpu, controller_cpu, dt=0.05)

        traj_fn = create_trajectory_function(trajectory_type)
        N_cpu, dt_cpu = params_cpu.N, params_cpu.dt
        ref_fn_cpu = lambda t: generate_reference_trajectory(traj_fn, t, N_cpu, dt_cpu)

        initial_state = np.array([5.0, 0.0, np.pi / 2])
        sim_cpu.reset(initial_state)
        history_cpu = sim_cpu.run(ref_fn_cpu, duration)
        metrics_cpu = compute_metrics(history_cpu)

        print(f"  CPU: RMSE={metrics_cpu['position_rmse']:.4f}m, "
              f"solve={metrics_cpu['mean_solve_time']:.2f}ms")

        result = {"cpu_rmse": metrics_cpu["position_rmse"],
                  "cpu_time": metrics_cpu["mean_solve_time"]}

        # ── GPU ──
        if gpu_available:
            model_gpu = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
            params_gpu = MPPIParams(
                N=30, dt=0.05, K=K, lambda_=1.0,
                sigma=np.array([0.5, 0.5]),
                Q=np.array([10.0, 10.0, 1.0]),
                R=np.array([0.1, 0.1]),
                device="cuda",
            )
            controller_gpu = MPPIController(model_gpu, params_gpu)

            # Warmup (first call triggers CUDA kernel compilation)
            traj_fn_gpu = create_trajectory_function(trajectory_type)
            N_gpu, dt_gpu = params_gpu.N, params_gpu.dt
            ref_fn_gpu = lambda t: generate_reference_trajectory(traj_fn_gpu, t, N_gpu, dt_gpu)
            warmup_ref = ref_fn_gpu(0.0)
            for _ in range(5):
                controller_gpu.compute_control(initial_state, warmup_ref)
            controller_gpu.reset()

            sim_gpu = Simulator(model_gpu, controller_gpu, dt=0.05)
            sim_gpu.reset(initial_state)
            history_gpu = sim_gpu.run(ref_fn_gpu, duration)
            metrics_gpu = compute_metrics(history_gpu)

            speedup = metrics_cpu["mean_solve_time"] / max(
                metrics_gpu["mean_solve_time"], 1e-6
            )
            rmse_diff = abs(
                metrics_gpu["position_rmse"] - metrics_cpu["position_rmse"]
            )

            print(f"  GPU: RMSE={metrics_gpu['position_rmse']:.4f}m, "
                  f"solve={metrics_gpu['mean_solve_time']:.2f}ms")
            print(f"  Speedup: {speedup:.1f}x")
            print(f"  RMSE diff: {rmse_diff:.4f}m")

            result["gpu_rmse"] = metrics_gpu["position_rmse"]
            result["gpu_time"] = metrics_gpu["mean_solve_time"]
            result["speedup"] = speedup
            result["rmse_diff"] = rmse_diff

        results[K] = result

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Summary".center(70))
    print("=" * 70)
    print(f"{'K':>8} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | "
          f"{'Speedup':>8} | {'RMSE diff':>10}")
    print("─" * 60)

    for K, r in results.items():
        if "gpu_time" in r:
            print(f"{K:>8} | {r['cpu_time']:>10.2f} | {r['gpu_time']:>10.2f} | "
                  f"{r['speedup']:>7.1f}x | {r['rmse_diff']:>10.4f}m")
        else:
            print(f"{K:>8} | {r['cpu_time']:>10.2f} | {'N/A':>10} | "
                  f"{'N/A':>8} | {'N/A':>10}")

    print("=" * 70)

    # ── Plot ──
    if plot and gpu_available:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(
                f"CPU vs GPU MPPI Benchmark ({trajectory_type} trajectory)",
                fontsize=14,
            )

            K_list = list(results.keys())
            cpu_times = [results[k]["cpu_time"] for k in K_list]
            gpu_times = [results[k].get("gpu_time", 0) for k in K_list]
            speedups = [results[k].get("speedup", 1) for k in K_list]
            rmse_diffs = [results[k].get("rmse_diff", 0) for k in K_list]

            # Panel 1: Solve time
            ax = axes[0, 0]
            x = np.arange(len(K_list))
            width = 0.35
            ax.bar(x - width / 2, cpu_times, width, label="CPU", color="steelblue")
            ax.bar(x + width / 2, gpu_times, width, label="GPU", color="coral")
            ax.set_xlabel("K (samples)")
            ax.set_ylabel("Solve Time (ms)")
            ax.set_title("Solve Time")
            ax.set_xticks(x)
            ax.set_xticklabels(K_list)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            # Panel 2: Speedup
            ax = axes[0, 1]
            ax.bar(x, speedups, color="seagreen")
            ax.set_xlabel("K (samples)")
            ax.set_ylabel("Speedup (x)")
            ax.set_title("GPU Speedup")
            ax.set_xticks(x)
            ax.set_xticklabels(K_list)
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1x")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            # Panel 3: RMSE comparison
            ax = axes[1, 0]
            cpu_rmses = [results[k]["cpu_rmse"] for k in K_list]
            gpu_rmses = [results[k].get("gpu_rmse", 0) for k in K_list]
            ax.bar(x - width / 2, cpu_rmses, width, label="CPU", color="steelblue")
            ax.bar(x + width / 2, gpu_rmses, width, label="GPU", color="coral")
            ax.set_xlabel("K (samples)")
            ax.set_ylabel("Position RMSE (m)")
            ax.set_title("Tracking RMSE")
            ax.set_xticks(x)
            ax.set_xticklabels(K_list)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            # Panel 4: RMSE difference
            ax = axes[1, 1]
            ax.bar(x, rmse_diffs, color="goldenrod")
            ax.set_xlabel("K (samples)")
            ax.set_ylabel("RMSE Difference (m)")
            ax.set_title("CPU-GPU RMSE Difference")
            ax.set_xticks(x)
            ax.set_xticklabels(K_list)
            ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="0.01m")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()

            out_dir = os.path.join(os.path.dirname(__file__), "../../results")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "gpu_benchmark.png")
            plt.savefig(out_path, dpi=150)
            print(f"\nPlot saved: {out_path}")
            plt.close()
        except ImportError:
            print("\nmatplotlib not available, skipping plot.")

    return results


def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU MPPI Benchmark")
    parser.add_argument(
        "--trajectory", type=str, default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    K_values = [256, 1024, 4096, 8192]
    run_benchmark(args.trajectory, args.duration, K_values, plot=not args.no_plot)


if __name__ == "__main__":
    main()
