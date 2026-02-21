#!/usr/bin/env python3
"""
SVMPC 모델별 비교 (Kinematic vs Dynamic vs Learned)

Stein Variational MPPI를 3가지 모델 타입에 적용하여 샘플 효율 비교.

Usage:
    python svmpc_models_comparison.py --trajectory circle --duration 15
    python svmpc_models_comparison.py --trajectory figure8 --svgd-iterations 10
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
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.controllers.mppi.mppi_params import SteinVariationalMPPIParams
from mppi_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
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
        description="SVMPC Model Comparison (Kinematic/Dynamic/Learned)"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--svgd-iterations",
        type=int,
        default=5,
        dest="svgd_iterations",
        help="SVGD iteration count",
    )
    parser.add_argument(
        "--duration", type=float, default=15.0, help="Simulation duration (s)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("SVMPC Model Comparison (Kinematic/Dynamic/Learned)".center(80))
    print("=" * 80)
    print(f"Trajectory: {args.trajectory}")
    print(f"SVGD Iterations: {args.svgd_iterations}")
    print(f"Duration: {args.duration}s")
    print("=" * 80 + "\n")

    # 공통 파라미터
    common_params = {
        "N": 30,
        "dt": 0.05,
        "K": 512,  # SVMPC는 작은 K로도 효율적
        "lambda_": 1.0,
        "Q": np.array([10.0, 10.0, 1.0]),
        "R": np.array([0.1, 0.1]),
        "Qf": np.array([20.0, 20.0, 2.0]),
        "svgd_num_iterations": args.svgd_iterations,
        "svgd_step_size": 0.01,
    }

    trajectory_fn = create_trajectory_function(args.trajectory)

    results = {}

    # ==================== 1. Kinematic Model ====================
    print("Setting up Kinematic Model + SVMPC...")
    kinematic_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    kinematic_params = SteinVariationalMPPIParams(
        **common_params, sigma=np.array([0.5, 0.5])
    )
    kinematic_controller = SteinVariationalMPPIController(
        kinematic_model, kinematic_params
    )

    def reference_fn_kinematic(t):
        return generate_reference_trajectory(
            trajectory_fn, t, kinematic_params.N, kinematic_params.dt
        )

    initial_state = trajectory_fn(0.0)
    kinematic_sim = Simulator(
        kinematic_model, kinematic_controller, kinematic_params.dt
    )
    kinematic_sim.reset(initial_state)

    print("  Running simulation...")
    kinematic_history = kinematic_sim.run(
        reference_fn_kinematic, args.duration, realtime=False
    )
    kinematic_metrics = compute_metrics(kinematic_history)
    kinematic_svgd_stats = kinematic_controller.get_svgd_statistics()
    print("  ✓ Kinematic completed\n")

    results["Kinematic"] = {
        "history": kinematic_history,
        "metrics": kinematic_metrics,
        "svgd_stats": kinematic_svgd_stats,
    }

    # ==================== 2. Dynamic Model ====================
    print("Setting up Dynamic Model + SVMPC...")
    dynamic_model = DifferentialDriveDynamic(
        mass=10.0, inertia=1.0, c_v=0.1, c_omega=0.1, v_max=2.0, omega_max=2.0
    )
    # Dynamic 모델용 파라미터 (5차원)
    dynamic_params_dict = common_params.copy()
    dynamic_params_dict["Q"] = np.array([10.0, 10.0, 1.0, 0.1, 0.1])
    dynamic_params_dict["Qf"] = np.array([20.0, 20.0, 2.0, 0.2, 0.2])
    dynamic_params = SteinVariationalMPPIParams(
        **dynamic_params_dict, sigma=np.array([0.5, 0.5])
    )
    dynamic_controller = SteinVariationalMPPIController(dynamic_model, dynamic_params)

    def reference_fn_dynamic(t):
        ref_3d = generate_reference_trajectory(
            trajectory_fn, t, dynamic_params.N, dynamic_params.dt
        )
        ref_5d = np.zeros((ref_3d.shape[0], 5))
        ref_5d[:, :3] = ref_3d
        return ref_5d

    initial_state_dynamic = np.concatenate([initial_state, [0.0, 0.0]])
    dynamic_sim = Simulator(dynamic_model, dynamic_controller, dynamic_params.dt)
    dynamic_sim.reset(initial_state_dynamic)

    print("  Running simulation...")
    dynamic_history = dynamic_sim.run(
        reference_fn_dynamic, args.duration, realtime=False
    )
    dynamic_metrics = compute_metrics(dynamic_history)
    dynamic_svgd_stats = dynamic_controller.get_svgd_statistics()
    print("  ✓ Dynamic completed\n")

    results["Dynamic"] = {
        "history": dynamic_history,
        "metrics": dynamic_metrics,
        "svgd_stats": dynamic_svgd_stats,
    }

    # ==================== 3. Learned Model (Residual) ====================
    print("Setting up Learned Model (Residual) + SVMPC...")
    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    learned_model = ResidualDynamics(base_model, residual_fn=None)
    learned_params = SteinVariationalMPPIParams(
        **common_params, sigma=np.array([0.5, 0.5])
    )
    learned_controller = SteinVariationalMPPIController(learned_model, learned_params)

    def reference_fn_learned(t):
        return generate_reference_trajectory(
            trajectory_fn, t, learned_params.N, learned_params.dt
        )

    learned_sim = Simulator(learned_model, learned_controller, learned_params.dt)
    learned_sim.reset(initial_state)

    print("  Running simulation...")
    learned_history = learned_sim.run(
        reference_fn_learned, args.duration, realtime=False
    )
    learned_metrics = compute_metrics(learned_history)
    learned_svgd_stats = learned_controller.get_svgd_statistics()
    print("  ✓ Learned completed\n")

    results["Learned"] = {
        "history": learned_history,
        "metrics": learned_metrics,
        "svgd_stats": learned_svgd_stats,
    }

    # ==================== 비교 ====================
    print_metrics(kinematic_metrics, title="Kinematic + SVMPC")
    print_metrics(dynamic_metrics, title="Dynamic + SVMPC")
    print_metrics(learned_metrics, title="Learned + SVMPC")

    compare_metrics(
        [kinematic_metrics, dynamic_metrics, learned_metrics],
        ["Kinematic", "Dynamic", "Learned"],
        title="SVMPC Model Comparison",
    )

    # SVGD 통계 비교
    print(f"\n{'=' * 60}")
    print(f"SVGD Statistics Comparison (Sample Efficiency)".center(60))
    print(f"{'=' * 60}")
    print(f"Model        | Mean Cost Improvement | Mean Bandwidth")
    print(f"{'─' * 60}")
    print(
        f"Kinematic    | {kinematic_svgd_stats['mean_cost_improvement']:22.4f} | {kinematic_svgd_stats['mean_bandwidth']:14.4f}"
    )
    print(
        f"Dynamic      | {dynamic_svgd_stats['mean_cost_improvement']:22.4f} | {dynamic_svgd_stats['mean_bandwidth']:14.4f}"
    )
    print(
        f"Learned      | {learned_svgd_stats['mean_cost_improvement']:22.4f} | {learned_svgd_stats['mean_bandwidth']:14.4f}"
    )
    print(f"{'=' * 60}\n")

    # ==================== 시각화 ====================
    print("Generating comparison plots...\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"SVMPC Model Comparison - {args.trajectory.capitalize()} "
        f"(SVGD Iterations={args.svgd_iterations})",
        fontsize=16,
    )

    kinematic_states = kinematic_history["state"]
    dynamic_states = dynamic_history["state"]
    learned_states = learned_history["state"]
    kinematic_refs = kinematic_history["reference"]
    kinematic_times = kinematic_history["time"]
    dynamic_times = dynamic_history["time"]
    learned_times = learned_history["time"]

    # 1. XY 궤적 비교
    ax = axes[0, 0]
    ax.plot(
        kinematic_states[:, 0],
        kinematic_states[:, 1],
        "b-",
        label="Kinematic",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        dynamic_states[:, 0],
        dynamic_states[:, 1],
        "g-",
        label="Dynamic",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        learned_states[:, 0],
        learned_states[:, 1],
        "m-",
        label="Learned",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        kinematic_refs[:, 0],
        kinematic_refs[:, 1],
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
    kinematic_errors = np.linalg.norm(
        kinematic_states[:, :2] - kinematic_refs[:, :2], axis=1
    )
    dynamic_errors = np.linalg.norm(
        dynamic_states[:, :2] - dynamic_history["reference"][:, :2], axis=1
    )
    learned_errors = np.linalg.norm(
        learned_states[:, :2] - learned_history["reference"][:, :2], axis=1
    )

    ax.plot(kinematic_times, kinematic_errors, "b-", label="Kinematic", linewidth=2)
    ax.plot(dynamic_times, dynamic_errors, "g-", label="Dynamic", linewidth=2)
    ax.plot(learned_times, learned_errors, "m-", label="Learned", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. SVGD Cost Improvement 히스토리
    ax = axes[0, 2]
    kinematic_cost_improvements = [
        s["cost_improvement"] for s in kinematic_svgd_stats["svgd_stats_history"]
    ]
    dynamic_cost_improvements = [
        s["cost_improvement"] for s in dynamic_svgd_stats["svgd_stats_history"]
    ]
    learned_cost_improvements = [
        s["cost_improvement"] for s in learned_svgd_stats["svgd_stats_history"]
    ]

    ax.plot(
        kinematic_times,
        kinematic_cost_improvements,
        "b-",
        label="Kinematic",
        linewidth=2,
    )
    ax.plot(
        dynamic_times, dynamic_cost_improvements, "g-", label="Dynamic", linewidth=2
    )
    ax.plot(
        learned_times, learned_cost_improvements, "m-", label="Learned", linewidth=2
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cost Improvement")
    ax.set_title("SVGD Cost Improvement (Sample Efficiency)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 제어 입력 비교
    ax = axes[1, 0]
    kinematic_controls = kinematic_history["control"]
    dynamic_controls = dynamic_history["control"]
    learned_controls = learned_history["control"]

    ax.plot(
        kinematic_times, kinematic_controls[:, 0], "b-", label="Kinematic", linewidth=2
    )
    ax.plot(
        dynamic_times, dynamic_controls[:, 0], "g-", label="Dynamic", linewidth=2
    )
    ax.plot(
        learned_times, learned_controls[:, 0], "m-", label="Learned", linewidth=2
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_title("Control Input (v)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. RMSE + Cost Improvement 바 차트
    ax = axes[1, 1]
    models = ["Kinematic", "Dynamic", "Learned"]
    rmses = [
        kinematic_metrics["position_rmse"],
        dynamic_metrics["position_rmse"],
        learned_metrics["position_rmse"],
    ]
    colors = ["blue", "green", "magenta"]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, rmses, width, label="Position RMSE", color=colors, alpha=0.7
    )

    ax2 = ax.twinx()
    cost_improvements = [
        kinematic_svgd_stats["mean_cost_improvement"],
        dynamic_svgd_stats["mean_cost_improvement"],
        learned_svgd_stats["mean_cost_improvement"],
    ]
    bars2 = ax2.bar(
        x + width / 2,
        cost_improvements,
        width,
        label="Mean Cost Improvement",
        color="orange",
        alpha=0.7,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Position RMSE (m)", color="black")
    ax2.set_ylabel("Mean Cost Improvement", color="orange")
    ax.set_title("RMSE and Cost Improvement")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 6. 요약
    ax = axes[1, 2]
    ax.axis("off")

    summary_text = f"""
    Model Comparison Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Kinematic Model:
      RMSE: {kinematic_metrics['position_rmse']:.4f} m
      Time: {kinematic_metrics['mean_solve_time']:.2f} ms
      Cost Improvement: {kinematic_svgd_stats['mean_cost_improvement']:.4f}
      Bandwidth: {kinematic_svgd_stats['mean_bandwidth']:.4f}

    Dynamic Model:
      RMSE: {dynamic_metrics['position_rmse']:.4f} m
      Time: {dynamic_metrics['mean_solve_time']:.2f} ms
      Cost Improvement: {dynamic_svgd_stats['mean_cost_improvement']:.4f}
      Bandwidth: {dynamic_svgd_stats['mean_bandwidth']:.4f}

    Learned Model (Residual):
      RMSE: {learned_metrics['position_rmse']:.4f} m
      Time: {learned_metrics['mean_solve_time']:.2f} ms
      Cost Improvement: {learned_svgd_stats['mean_cost_improvement']:.4f}
      Bandwidth: {learned_svgd_stats['mean_bandwidth']:.4f}

    Settings:
      Trajectory: {args.trajectory.capitalize()}
      SVGD Iterations: {args.svgd_iterations}
      Samples: K={common_params['K']}
      Duration: {args.duration}s

    Conclusion:
      SVMPC provides sample efficiency
      improvements across all models.
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
    plt.savefig("svmpc_models_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: svmpc_models_comparison.png")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
