#!/usr/bin/env python3
"""
Smooth MPPI 모델별 비교 (Kinematic vs Dynamic vs Learned)

Input-Lifting MPPI를 3가지 모델 타입에 적용하여 제어 부드러움 비교.

Usage:
    python smooth_mppi_models_comparison.py --trajectory circle --duration 20
    python smooth_mppi_models_comparison.py --trajectory figure8 --jerk-weight 5.0
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
from mppi_controller.controllers.mppi.mppi_params import SmoothMPPIParams
from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
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


def dummy_residual(state, control):
    """더미 residual (작은 모델링 오차)"""
    return np.array([0.01 * np.sin(state[2]), 0.01 * np.cos(state[2]), 0.0])


def main():
    parser = argparse.ArgumentParser(
        description="Smooth MPPI Model Comparison (Kinematic/Dynamic/Learned)"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--jerk-weight",
        type=float,
        default=1.0,
        dest="jerk_weight",
        help="Jerk cost weight",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="Simulation duration (s)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("Smooth MPPI Model Comparison (Kinematic/Dynamic/Learned)".center(80))
    print("=" * 80)
    print(f"Trajectory: {args.trajectory}")
    print(f"Jerk Weight: {args.jerk_weight}")
    print(f"Duration: {args.duration}s")
    print("=" * 80 + "\n")

    # 공통 파라미터
    common_params = {
        "N": 30,
        "dt": 0.05,
        "K": 1024,
        "lambda_": 1.0,
        "Q": np.array([10.0, 10.0, 1.0]),
        "R": np.array([0.1, 0.1]),
        "Qf": np.array([20.0, 20.0, 2.0]),
        "jerk_weight": args.jerk_weight,
    }

    trajectory_fn = create_trajectory_function(args.trajectory)

    results = {}

    # ==================== 1. Kinematic Model ====================
    print("Setting up Kinematic Model + Smooth MPPI...")
    kinematic_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    kinematic_params = SmoothMPPIParams(
        **common_params, sigma=np.array([0.5, 0.5])
    )
    kinematic_controller = SmoothMPPIController(kinematic_model, kinematic_params)

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
    kinematic_jerk_stats = kinematic_controller.get_jerk_statistics()
    print("  ✓ Kinematic completed\n")

    results["Kinematic"] = {
        "history": kinematic_history,
        "metrics": kinematic_metrics,
        "jerk_stats": kinematic_jerk_stats,
    }

    # ==================== 2. Dynamic Model ====================
    print("Setting up Dynamic Model + Smooth MPPI...")
    dynamic_model = DifferentialDriveDynamic(
        mass=10.0, inertia=1.0, c_v=0.1, c_omega=0.1, v_max=2.0, omega_max=2.0
    )
    # Dynamic 모델용 파라미터 (5차원 상태)
    dynamic_params_dict = common_params.copy()
    dynamic_params_dict["Q"] = np.array([10.0, 10.0, 1.0, 0.1, 0.1])  # [x, y, θ, v, ω]
    dynamic_params_dict["Qf"] = np.array([20.0, 20.0, 2.0, 0.2, 0.2])
    dynamic_params = SmoothMPPIParams(
        **dynamic_params_dict, sigma=np.array([0.5, 0.5])
    )
    dynamic_controller = SmoothMPPIController(dynamic_model, dynamic_params)

    def reference_fn_dynamic(t):
        # Dynamic 모델은 5차원 ([x, y, θ, v, ω])
        ref_3d = generate_reference_trajectory(
            trajectory_fn, t, dynamic_params.N, dynamic_params.dt
        )
        # v, ω는 0으로 설정 (목표 속도)
        ref_5d = np.zeros((ref_3d.shape[0], 5))
        ref_5d[:, :3] = ref_3d  # [x, y, θ]
        return ref_5d

    # Dynamic 모델 초기 상태 (v, ω 추가)
    initial_state_dynamic = np.concatenate([initial_state, [0.0, 0.0]])
    dynamic_sim = Simulator(dynamic_model, dynamic_controller, dynamic_params.dt)
    dynamic_sim.reset(initial_state_dynamic)

    print("  Running simulation...")
    dynamic_history = dynamic_sim.run(
        reference_fn_dynamic, args.duration, realtime=False
    )
    dynamic_metrics = compute_metrics(dynamic_history)
    dynamic_jerk_stats = dynamic_controller.get_jerk_statistics()
    print("  ✓ Dynamic completed\n")

    results["Dynamic"] = {
        "history": dynamic_history,
        "metrics": dynamic_metrics,
        "jerk_stats": dynamic_jerk_stats,
    }

    # ==================== 3. Learned Model (Residual) ====================
    print("Setting up Learned Model (Residual) + Smooth MPPI...")
    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    # Residual 없이 (Physics와 동일, 구조만 시연)
    learned_model = ResidualDynamics(base_model, residual_fn=None)
    learned_params = SmoothMPPIParams(
        **common_params, sigma=np.array([0.5, 0.5])
    )
    learned_controller = SmoothMPPIController(learned_model, learned_params)

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
    learned_jerk_stats = learned_controller.get_jerk_statistics()
    print("  ✓ Learned completed\n")

    results["Learned"] = {
        "history": learned_history,
        "metrics": learned_metrics,
        "jerk_stats": learned_jerk_stats,
    }

    # ==================== 비교 ====================
    print_metrics(kinematic_metrics, title="Kinematic + Smooth MPPI")
    print_metrics(dynamic_metrics, title="Dynamic + Smooth MPPI")
    print_metrics(learned_metrics, title="Learned + Smooth MPPI")

    compare_metrics(
        [kinematic_metrics, dynamic_metrics, learned_metrics],
        ["Kinematic", "Dynamic", "Learned"],
        title="Smooth MPPI Model Comparison",
    )

    # Jerk 통계 비교
    print(f"\n{'=' * 60}")
    print(f"Jerk Statistics Comparison (Smoothness)".center(60))
    print(f"{'=' * 60}")
    print(f"Model        | Mean Jerk | Max Jerk")
    print(f"{'─' * 60}")
    print(
        f"Kinematic    | {kinematic_jerk_stats['mean_jerk']:9.4f} | {kinematic_jerk_stats['max_jerk']:8.4f}"
    )
    print(
        f"Dynamic      | {dynamic_jerk_stats['mean_jerk']:9.4f} | {dynamic_jerk_stats['max_jerk']:8.4f}"
    )
    print(
        f"Learned      | {learned_jerk_stats['mean_jerk']:9.4f} | {learned_jerk_stats['max_jerk']:8.4f}"
    )
    print(f"{'=' * 60}\n")

    # ==================== 시각화 ====================
    print("Generating comparison plots...\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Smooth MPPI Model Comparison - {args.trajectory.capitalize()} "
        f"(Jerk Weight={args.jerk_weight})",
        fontsize=16,
    )

    # 1. XY 궤적 비교
    ax = axes[0, 0]
    kinematic_states = kinematic_history["state"]
    dynamic_states = dynamic_history["state"]
    learned_states = learned_history["state"]
    kinematic_refs = kinematic_history["reference"]

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
    kinematic_times = kinematic_history["time"]
    dynamic_times = dynamic_history["time"]
    learned_times = learned_history["time"]

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

    # 3. Jerk 히스토리 비교
    ax = axes[0, 2]
    kinematic_jerk_hist = kinematic_jerk_stats["jerk_history"]
    dynamic_jerk_hist = dynamic_jerk_stats["jerk_history"]
    learned_jerk_hist = learned_jerk_stats["jerk_history"]

    ax.plot(kinematic_times, kinematic_jerk_hist, "b-", label="Kinematic", linewidth=2)
    ax.plot(dynamic_times, dynamic_jerk_hist, "g-", label="Dynamic", linewidth=2)
    ax.plot(learned_times, learned_jerk_hist, "m-", label="Learned", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Jerk Cost")
    ax.set_title("Jerk Cost Over Time (Smoothness)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 제어 입력 비교 (선속도)
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

    # 5. RMSE + Jerk 바 차트
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
    jerks = [
        kinematic_jerk_stats["mean_jerk"],
        dynamic_jerk_stats["mean_jerk"],
        learned_jerk_stats["mean_jerk"],
    ]
    bars2 = ax2.bar(
        x + width / 2, jerks, width, label="Mean Jerk", color="orange", alpha=0.7
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Position RMSE (m)", color="black")
    ax2.set_ylabel("Mean Jerk Cost", color="orange")
    ax.set_title("RMSE and Jerk Comparison")
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
      Mean Jerk: {kinematic_jerk_stats['mean_jerk']:.4f}
      Max Jerk: {kinematic_jerk_stats['max_jerk']:.4f}

    Dynamic Model:
      RMSE: {dynamic_metrics['position_rmse']:.4f} m
      Time: {dynamic_metrics['mean_solve_time']:.2f} ms
      Mean Jerk: {dynamic_jerk_stats['mean_jerk']:.4f}
      Max Jerk: {dynamic_jerk_stats['max_jerk']:.4f}

    Learned Model (Residual):
      RMSE: {learned_metrics['position_rmse']:.4f} m
      Time: {learned_metrics['mean_solve_time']:.2f} ms
      Mean Jerk: {learned_jerk_stats['mean_jerk']:.4f}
      Max Jerk: {learned_jerk_stats['max_jerk']:.4f}

    Settings:
      Trajectory: {args.trajectory.capitalize()}
      Jerk Weight: {args.jerk_weight}
      Duration: {args.duration}s

    Conclusion:
      Smooth MPPI provides consistent
      control smoothness across all models.
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
    plt.savefig("smooth_mppi_models_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: smooth_mppi_models_comparison.png")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
