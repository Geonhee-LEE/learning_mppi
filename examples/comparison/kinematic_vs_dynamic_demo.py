#!/usr/bin/env python3
"""
기구학 vs 동역학 모델 비교 데모

동일한 레퍼런스 궤적에서 두 모델의 성능 비교.

Usage:
    python kinematic_vs_dynamic_demo.py --trajectory circle --duration 30
    python kinematic_vs_dynamic_demo.py --trajectory figure8 --live
"""

import numpy as np
import argparse
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.visualizer import SimulationVisualizer
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
    parser = argparse.ArgumentParser(description="Kinematic vs Dynamic Model Comparison")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Simulation duration (s)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 랜덤 시드 설정
    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("Kinematic vs Dynamic Model Comparison".center(80))
    print("=" * 80)
    print(f"Trajectory: {args.trajectory}")
    print(f"Duration: {args.duration}s")
    print("=" * 80 + "\n")

    # ==================== 기구학 모델 ====================
    print("Setting up Kinematic Model...")

    # 1. 기구학 모델
    kinematic_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # 2. MPPI 파라미터 (기구학용)
    kinematic_params = MPPIParams(
        N=30,
        dt=0.05,
        K=1024,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),  # [v, ω]
        Q=np.array([10.0, 10.0, 1.0]),  # [x, y, θ]
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )

    # 3. 컨트롤러 및 시뮬레이터
    kinematic_controller = MPPIController(kinematic_model, kinematic_params)
    kinematic_sim = Simulator(kinematic_model, kinematic_controller, kinematic_params.dt)

    # 4. 레퍼런스 궤적 (기구학: 3차원)
    trajectory_fn_kinematic = create_trajectory_function(args.trajectory)

    def kinematic_reference_fn(t):
        return generate_reference_trajectory(
            trajectory_fn_kinematic, t, kinematic_params.N, kinematic_params.dt
        )

    # 5. 초기 상태 및 시뮬레이션
    initial_state_kinematic = trajectory_fn_kinematic(0.0)
    kinematic_sim.reset(initial_state_kinematic)

    print(f"  Initial state: x={initial_state_kinematic[0]:.2f}, "
          f"y={initial_state_kinematic[1]:.2f}, "
          f"θ={np.rad2deg(initial_state_kinematic[2]):.1f}°")
    print("  Running simulation...")

    kinematic_history = kinematic_sim.run(
        kinematic_reference_fn, args.duration, realtime=False
    )

    kinematic_metrics = compute_metrics(kinematic_history)
    print("  ✓ Kinematic simulation completed\n")

    # ==================== 동역학 모델 ====================
    print("Setting up Dynamic Model...")

    # 1. 동역학 모델
    dynamic_model = DifferentialDriveDynamic(
        mass=10.0,
        inertia=1.0,
        c_v=0.1,
        c_omega=0.1,
        a_max=2.0,
        alpha_max=2.0,
        v_max=2.0,
        omega_max=2.0,
    )

    # 2. MPPI 파라미터 (동역학용)
    dynamic_params = MPPIParams(
        N=30,
        dt=0.05,
        K=1024,
        lambda_=1.0,
        sigma=np.array([1.0, 1.0]),  # [a, α]
        Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),  # [x, y, θ, v, ω]
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2, 0.2]),
    )

    # 3. 컨트롤러 및 시뮬레이터
    dynamic_controller = MPPIController(dynamic_model, dynamic_params)
    dynamic_sim = Simulator(dynamic_model, dynamic_controller, dynamic_params.dt)

    # 4. 레퍼런스 궤적 (동역학: 5차원)
    def trajectory_fn_dynamic(t):
        kinematic_state = trajectory_fn_kinematic(t)
        if args.trajectory == "circle":
            v_ref = 0.5
            omega_ref = 0.1
        else:
            v_ref = 0.5
            omega_ref = 0.0
        return np.array(
            [
                kinematic_state[0],
                kinematic_state[1],
                kinematic_state[2],
                v_ref,
                omega_ref,
            ]
        )

    def dynamic_reference_fn(t):
        return generate_reference_trajectory(
            trajectory_fn_dynamic, t, dynamic_params.N, dynamic_params.dt
        )

    # 5. 초기 상태 및 시뮬레이션
    initial_state_dynamic = trajectory_fn_dynamic(0.0)
    dynamic_sim.reset(initial_state_dynamic)

    print(f"  Initial state: x={initial_state_dynamic[0]:.2f}, "
          f"y={initial_state_dynamic[1]:.2f}, "
          f"θ={np.rad2deg(initial_state_dynamic[2]):.1f}°, "
          f"v={initial_state_dynamic[3]:.2f}, ω={initial_state_dynamic[4]:.2f}")
    print("  Running simulation...")

    dynamic_history = dynamic_sim.run(dynamic_reference_fn, args.duration, realtime=False)

    dynamic_metrics = compute_metrics(dynamic_history)
    print("  ✓ Dynamic simulation completed\n")

    # ==================== 비교 ====================
    print_metrics(kinematic_metrics, title="Kinematic Model Performance")
    print_metrics(dynamic_metrics, title="Dynamic Model Performance")

    compare_metrics(
        [kinematic_metrics, dynamic_metrics],
        ["Kinematic", "Dynamic"],
        title="Kinematic vs Dynamic Comparison",
    )

    # ==================== 시각화 ====================
    print("Generating comparison plots...\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Kinematic vs Dynamic - {args.trajectory.capitalize()} Trajectory", fontsize=16)

    # 1. XY 궤적 비교
    ax = axes[0, 0]
    kinematic_states = kinematic_history["state"]
    dynamic_states = dynamic_history["state"]
    kinematic_refs = kinematic_history["reference"]

    ax.plot(kinematic_states[:, 0], kinematic_states[:, 1], "b-", label="Kinematic", linewidth=2)
    ax.plot(dynamic_states[:, 0], dynamic_states[:, 1], "g-", label="Dynamic", linewidth=2)
    ax.plot(
        kinematic_refs[:, 0],
        kinematic_refs[:, 1],
        "r--",
        label="Reference",
        linewidth=2,
        alpha=0.7,
    )
    ax.scatter(kinematic_states[0, 0], kinematic_states[0, 1], c="blue", s=100, marker="o")
    ax.scatter(kinematic_states[-1, 0], kinematic_states[-1, 1], c="blue", s=100, marker="X")
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
    kinematic_errors = np.linalg.norm(
        kinematic_states[:, :2] - kinematic_refs[:, :2], axis=1
    )
    dynamic_errors = np.linalg.norm(dynamic_states[:, :2] - dynamic_states[:, :2], axis=1)

    ax.plot(kinematic_times, kinematic_errors, "b-", label="Kinematic", linewidth=2)
    ax.plot(dynamic_times, dynamic_errors, "g-", label="Dynamic", linewidth=2)
    ax.axhline(
        y=kinematic_metrics["position_rmse"],
        color="b",
        linestyle="--",
        alpha=0.5,
        label=f'Kin RMSE={kinematic_metrics["position_rmse"]:.3f}m',
    )
    ax.axhline(
        y=dynamic_metrics["position_rmse"],
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f'Dyn RMSE={dynamic_metrics["position_rmse"]:.3f}m',
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 제어 입력 비교
    ax = axes[0, 2]
    kinematic_controls = kinematic_history["control"]
    dynamic_controls = dynamic_history["control"]

    ax.plot(
        kinematic_times,
        kinematic_controls[:, 0],
        "b-",
        label="Kinematic v (m/s)",
        linewidth=2,
    )
    ax.plot(
        dynamic_times, dynamic_controls[:, 0], "g-", label="Dynamic a (m/s²)", linewidth=2
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Inputs (Linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 각속도/각가속도 비교
    ax = axes[1, 0]
    ax.plot(
        kinematic_times,
        kinematic_controls[:, 1],
        "b-",
        label="Kinematic ω (rad/s)",
        linewidth=2,
    )
    ax.plot(
        dynamic_times,
        dynamic_controls[:, 1],
        "g-",
        label="Dynamic α (rad/s²)",
        linewidth=2,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Inputs (Angular)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 계산 시간 비교
    ax = axes[1, 1]
    kinematic_solve_times = kinematic_history["solve_time"] * 1000  # ms
    dynamic_solve_times = dynamic_history["solve_time"] * 1000  # ms

    ax.plot(kinematic_times, kinematic_solve_times, "b-", label="Kinematic", linewidth=2)
    ax.plot(dynamic_times, dynamic_solve_times, "g-", label="Dynamic", linewidth=2)
    ax.axhline(
        y=kinematic_metrics["mean_solve_time"],
        color="b",
        linestyle="--",
        alpha=0.5,
        label=f'Kin Mean={kinematic_metrics["mean_solve_time"]:.2f}ms',
    )
    ax.axhline(
        y=dynamic_metrics["mean_solve_time"],
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f'Dyn Mean={dynamic_metrics["mean_solve_time"]:.2f}ms',
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 메트릭 요약 (텍스트)
    ax = axes[1, 2]
    ax.axis("off")

    summary_text = f"""
    Comparison Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Kinematic Model:
      Position RMSE: {kinematic_metrics['position_rmse']:.4f} m
      Solve Time: {kinematic_metrics['mean_solve_time']:.2f} ms

    Dynamic Model:
      Position RMSE: {dynamic_metrics['position_rmse']:.4f} m
      Solve Time: {dynamic_metrics['mean_solve_time']:.2f} ms

    Difference:
      RMSE Δ: {abs(kinematic_metrics['position_rmse'] - dynamic_metrics['position_rmse']):.4f} m
      Time Δ: {abs(kinematic_metrics['mean_solve_time'] - dynamic_metrics['mean_solve_time']):.2f} ms

    Trajectory: {args.trajectory.capitalize()}
    Duration: {args.duration}s
    Samples: K={kinematic_params.K}, N={kinematic_params.N}
    """

    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.show()

    print("Comparison complete!")


if __name__ == "__main__":
    main()
