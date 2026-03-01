#!/usr/bin/env python3
"""
Vanilla MPPI vs DIAL-MPPI 비교 데모

DIAL-MPC (ICRA 2025) 기반 확산 어닐링 MPPI와 Vanilla MPPI의
원형 궤적 추적 성능을 비교합니다.

Usage:
    PYTHONPATH=. python examples/comparison/vanilla_vs_dial_mppi_demo.py
    PYTHONPATH=. python examples/comparison/vanilla_vs_dial_mppi_demo.py --trajectory figure8
    PYTHONPATH=. python examples/comparison/vanilla_vs_dial_mppi_demo.py --duration 15 --K 512
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, DIALMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
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
        description="Vanilla MPPI vs DIAL-MPPI Comparison"
    )
    parser.add_argument(
        "--trajectory", type=str, default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Simulation duration (s)"
    )
    parser.add_argument("--K", type=int, default=1024, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-diffuse-init", type=int, default=10,
        help="DIAL cold start iterations",
    )
    parser.add_argument(
        "--n-diffuse", type=int, default=3,
        help="DIAL warm start iterations",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Vanilla MPPI vs DIAL-MPPI Comparison".center(80))
    print("=" * 80)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  K:          {args.K}")
    print(f"  DIAL init:  {args.n_diffuse_init} iters (cold)")
    print(f"  DIAL run:   {args.n_diffuse} iters (warm)")
    print("=" * 80 + "\n")

    # ==================== 공통 설정 ====================
    common = dict(
        N=30, dt=0.05, K=args.K,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )

    trajectory_fn = create_trajectory_function(args.trajectory)
    initial_state = trajectory_fn(0.0)

    # ==================== 1. Vanilla MPPI ====================
    print("Running Vanilla MPPI...")
    np.random.seed(args.seed)
    vanilla_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    vanilla_params = MPPIParams(**common)
    vanilla_ctrl = MPPIController(vanilla_model, vanilla_params)
    vanilla_sim = Simulator(vanilla_model, vanilla_ctrl, common["dt"])

    def ref_fn_v(t):
        return generate_reference_trajectory(trajectory_fn, t, common["N"], common["dt"])

    vanilla_sim.reset(initial_state.copy())
    vanilla_history = vanilla_sim.run(ref_fn_v, args.duration, realtime=False)
    vanilla_metrics = compute_metrics(vanilla_history)
    print(f"  RMSE: {vanilla_metrics['position_rmse']:.4f}m")
    print(f"  Time: {vanilla_metrics['mean_solve_time']:.2f}ms\n")

    # ==================== 2. DIAL-MPPI ====================
    print("Running DIAL-MPPI...")
    np.random.seed(args.seed)
    dial_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    dial_params = DIALMPPIParams(
        **common,
        n_diffuse_init=args.n_diffuse_init,
        n_diffuse=args.n_diffuse,
        traj_diffuse_factor=0.5,
        horizon_diffuse_factor=0.5,
        sigma_scale=1.0,
        use_reward_normalization=True,
    )
    dial_ctrl = DIALMPPIController(dial_model, dial_params)
    dial_sim = Simulator(dial_model, dial_ctrl, common["dt"])

    def ref_fn_d(t):
        return generate_reference_trajectory(trajectory_fn, t, common["N"], common["dt"])

    dial_sim.reset(initial_state.copy())
    dial_history = dial_sim.run(ref_fn_d, args.duration, realtime=False)
    dial_metrics = compute_metrics(dial_history)
    print(f"  RMSE: {dial_metrics['position_rmse']:.4f}m")
    print(f"  Time: {dial_metrics['mean_solve_time']:.2f}ms\n")

    # ==================== 비교 출력 ====================
    print_metrics(vanilla_metrics, title="Vanilla MPPI")
    print_metrics(dial_metrics, title="DIAL-MPPI")
    compare_metrics(
        [vanilla_metrics, dial_metrics],
        ["Vanilla", "DIAL-MPPI"],
        title="Vanilla vs DIAL-MPPI",
    )

    # DIAL 통계
    dial_stats = dial_ctrl.get_dial_statistics()
    print(f"\n{'=' * 60}")
    print("DIAL-MPPI Statistics".center(60))
    print(f"{'=' * 60}")
    print(f"  Mean cost improvement: {dial_stats['mean_cost_improvement']:.4f}")
    print(f"  Mean iterations:       {dial_stats['mean_n_iters']:.1f}")
    print(f"{'=' * 60}\n")

    # ==================== 시각화 ====================
    print("Generating comparison plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Vanilla MPPI vs DIAL-MPPI - {args.trajectory.capitalize()} Trajectory",
        fontsize=16,
    )

    v_states = vanilla_history["state"]
    d_states = dial_history["state"]
    v_refs = vanilla_history["reference"]
    v_times = vanilla_history["time"]
    d_times = dial_history["time"]

    # 1. XY 궤적
    ax = axes[0, 0]
    ax.plot(v_refs[:, 0], v_refs[:, 1], "r--", label="Reference", linewidth=2, alpha=0.5)
    ax.plot(v_states[:, 0], v_states[:, 1], "b-", label="Vanilla", linewidth=2, alpha=0.7)
    ax.plot(d_states[:, 0], d_states[:, 1], "g-", label="DIAL-MPPI", linewidth=2, alpha=0.7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 위치 오차
    ax = axes[0, 1]
    v_errors = np.linalg.norm(v_states[:, :2] - v_refs[:, :2], axis=1)
    d_errors = np.linalg.norm(d_states[:, :2] - v_refs[:, :2], axis=1)
    ax.plot(v_times, v_errors, "b-", label="Vanilla", linewidth=2)
    ax.plot(d_times, d_errors, "g-", label="DIAL-MPPI", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 반복별 비용 개선 (DIAL 마지막 호출의 반복별 비용)
    ax = axes[0, 2]
    last_costs = dial_stats["last_iteration_costs"]
    if last_costs:
        ax.plot(range(1, len(last_costs) + 1), last_costs, "go-", linewidth=2, markersize=6)
        ax.set_xlabel("Diffusion Iteration")
        ax.set_ylabel("Best Sample Cost")
        ax.set_title("DIAL-MPPI: Per-Iteration Cost (Last Call)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No iteration data", ha="center", va="center")
        ax.set_title("DIAL-MPPI: Per-Iteration Cost")

    # 4. 제어 입력
    ax = axes[1, 0]
    v_controls = np.array(vanilla_history["control"])
    d_controls = np.array(dial_history["control"])
    ax.plot(v_times, v_controls[:, 0], "b-", label="Vanilla v", linewidth=1.5, alpha=0.7)
    ax.plot(d_times, d_controls[:, 0], "g-", label="DIAL v", linewidth=1.5, alpha=0.7)
    ax.plot(v_times, v_controls[:, 1], "b--", label="Vanilla ω", linewidth=1.5, alpha=0.5)
    ax.plot(d_times, d_controls[:, 1], "g--", label="DIAL ω", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Inputs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. RMSE 바 차트
    ax = axes[1, 1]
    labels = ["Vanilla", "DIAL-MPPI"]
    rmses = [vanilla_metrics["position_rmse"], dial_metrics["position_rmse"]]
    colors = ["#4A90D9", "#7CB342"]
    bars = ax.bar(labels, rmses, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, rmse in zip(bars, rmses):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
            f"{rmse:.4f}m", ha="center", va="bottom", fontsize=11,
        )

    # 6. 요약 텍스트
    ax = axes[1, 2]
    ax.axis("off")
    improvement = (
        (vanilla_metrics["position_rmse"] - dial_metrics["position_rmse"])
        / vanilla_metrics["position_rmse"] * 100
        if vanilla_metrics["position_rmse"] > 0 else 0
    )
    summary = (
        f"  Comparison Summary\n"
        f"  {'=' * 40}\n\n"
        f"  Vanilla MPPI:\n"
        f"    RMSE:       {vanilla_metrics['position_rmse']:.4f} m\n"
        f"    Solve time: {vanilla_metrics['mean_solve_time']:.2f} ms\n\n"
        f"  DIAL-MPPI:\n"
        f"    RMSE:       {dial_metrics['position_rmse']:.4f} m\n"
        f"    Solve time: {dial_metrics['mean_solve_time']:.2f} ms\n"
        f"    Init iters: {args.n_diffuse_init}\n"
        f"    Run iters:  {args.n_diffuse}\n\n"
        f"  RMSE improvement: {improvement:+.1f}%\n"
        f"  Trajectory: {args.trajectory}\n"
        f"  K={args.K}, N={common['N']}, dt={common['dt']}"
    )
    ax.text(
        0.05, 0.5, summary, fontsize=10, verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    save_path = "plots/vanilla_vs_dial_mppi_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
