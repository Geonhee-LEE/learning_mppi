#!/usr/bin/env python3
"""
MPPI 전체 변형 벤치마크

9가지 MPPI 변형을 동시에 비교하여 성능을 평가합니다.

Usage:
    python mppi_all_variants_benchmark.py --trajectory circle --duration 20
    python mppi_all_variants_benchmark.py --trajectory figure8 --duration 30 --live
"""

import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    TubeMPPIParams,
    LogMPPIParams,
    TsallisMPPIParams,
    RiskAwareMPPIParams,
    SmoothMPPIParams,
    SteinVariationalMPPIParams,
    SplineMPPIParams,
    SVGMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.log_mppi import LogMPPIController
from mppi_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController
from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
from mppi_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
from mppi_controller.controllers.mppi.spline_mppi import SplineMPPIController
from mppi_controller.controllers.mppi.svg_mppi import SVGMPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="MPPI All Variants Benchmark")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="Simulation duration (s)"
    )
    parser.add_argument(
        "--live", action="store_true", help="Realtime simulation mode"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 80)
    print("MPPI All Variants Benchmark (9 variants)".center(80))
    print("=" * 80)
    print(f"Trajectory: {args.trajectory}")
    print(f"Duration: {args.duration}s")
    print(f"Realtime: {args.live}")
    print("=" * 80 + "\n")

    # 공통 모델
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    trajectory_fn = create_trajectory_function(args.trajectory)
    initial_state = trajectory_fn(0.0)

    # 공통 파라미터
    common_params = {
        "N": 30,
        "dt": 0.05,
        "K": 1024,
        "lambda_": 1.0,
        "sigma": np.array([0.5, 0.5]),
        "Q": np.array([10.0, 10.0, 1.0]),
        "R": np.array([0.1, 0.1]),
        "Qf": np.array([20.0, 20.0, 2.0]),
    }

    # 변형 정의
    variants = [
        {
            "name": "Vanilla MPPI",
            "controller_class": MPPIController,
            "params_class": MPPIParams,
            "params": common_params,
            "color": "blue",
        },
        {
            "name": "Tube-MPPI",
            "controller_class": TubeMPPIController,
            "params_class": TubeMPPIParams,
            "params": {
                **common_params,
                "tube_enabled": True,
                "tube_margin": 0.3,
                "K_fb": np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            },
            "color": "green",
        },
        {
            "name": "Log-MPPI",
            "controller_class": LogMPPIController,
            "params_class": LogMPPIParams,
            "params": {**common_params, "use_baseline": True},
            "color": "red",
        },
        {
            "name": "Tsallis-MPPI",
            "controller_class": TsallisMPPIController,
            "params_class": TsallisMPPIParams,
            "params": {**common_params, "tsallis_q": 1.2},
            "color": "purple",
        },
        {
            "name": "Risk-Aware",
            "controller_class": RiskAwareMPPIController,
            "params_class": RiskAwareMPPIParams,
            "params": {**common_params, "cvar_alpha": 0.7},
            "color": "orange",
        },
        {
            "name": "Smooth MPPI",
            "controller_class": SmoothMPPIController,
            "params_class": SmoothMPPIParams,
            "params": {**common_params, "jerk_weight": 1.0},
            "color": "cyan",
        },
        {
            "name": "SVMPC",
            "controller_class": SteinVariationalMPPIController,
            "params_class": SteinVariationalMPPIParams,
            "params": {
                **common_params,
                "svgd_num_iterations": 3,
                "svgd_step_size": 0.01,
            },
            "color": "brown",
        },
        {
            "name": "Spline-MPPI",
            "controller_class": SplineMPPIController,
            "params_class": SplineMPPIParams,
            "params": {**common_params, "spline_num_knots": 8, "spline_degree": 3},
            "color": "pink",
        },
        {
            "name": "SVG-MPPI",
            "controller_class": SVGMPPIController,
            "params_class": SVGMPPIParams,
            "params": {
                **common_params,
                "svg_num_guide_particles": 32,
                "svgd_num_iterations": 3,
                "svg_guide_step_size": 0.01,
            },
            "color": "magenta",
        },
    ]

    results = []

    # 각 변형 시뮬레이션
    for i, variant in enumerate(variants):
        print(f"\n[{i+1}/9] Running {variant['name']}...")

        # 파라미터 생성
        params = variant["params_class"](**variant["params"])

        # 컨트롤러 생성
        controller = variant["controller_class"](model, params)

        # 레퍼런스 함수
        def reference_fn(t):
            return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

        # 시뮬레이터
        sim = Simulator(model, controller, params.dt)
        sim.reset(initial_state)

        # 시뮬레이션
        t_start = time.time()
        history = sim.run(reference_fn, args.duration, realtime=args.live)
        total_time = time.time() - t_start

        # 메트릭
        metrics = compute_metrics(history)

        # 결과 저장
        results.append(
            {
                "name": variant["name"],
                "color": variant["color"],
                "history": history,
                "metrics": metrics,
                "total_time": total_time,
            }
        )

        print(f"  ✓ RMSE: {metrics['position_rmse']:.4f}m, "
              f"Solve: {metrics['mean_solve_time']:.2f}ms")

    # ==================== 메트릭 비교 테이블 ====================
    print("\n" + "=" * 100)
    print("MPPI Variants Performance Comparison".center(100))
    print("=" * 100)
    print(
        f"{'Variant':<15} | {'Pos RMSE':<10} | {'Max Error':<10} | "
        f"{'Control Rate':<12} | {'Solve Time':<12} | {'Total Time':<12}"
    )
    print("─" * 100)

    for result in results:
        m = result["metrics"]
        print(
            f"{result['name']:<15} | "
            f"{m['position_rmse']:>9.4f}m | "
            f"{m['max_position_error']:>9.4f}m | "
            f"{m['control_rate']:>11.4f} | "
            f"{m['mean_solve_time']:>10.2f}ms | "
            f"{result['total_time']:>10.2f}s"
        )

    print("=" * 100 + "\n")

    # ==================== 시각화 ====================
    print("Generating benchmark plots...\n")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(
        f"MPPI All Variants Benchmark - {args.trajectory.capitalize()} "
        f"(K=1024, N=30, T={args.duration}s)",
        fontsize=16,
    )

    # 1. XY 궤적 (모든 변형)
    ax = fig.add_subplot(gs[0, 0])
    for result in results:
        states = result["history"]["state"]
        ax.plot(
            states[:, 0],
            states[:, 1],
            label=result["name"],
            color=result["color"],
            linewidth=1.5,
            alpha=0.7,
        )

    # Reference
    ref = results[0]["history"]["reference"]
    ax.plot(ref[:, 0], ref[:, 1], "k--", label="Reference", linewidth=2, alpha=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory Comparison")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. Position RMSE 바 차트
    ax = fig.add_subplot(gs[0, 1])
    names = [r["name"] for r in results]
    rmses = [r["metrics"]["position_rmse"] for r in results]
    colors = [r["color"] for r in results]

    bars = ax.barh(names, rmses, color=colors, alpha=0.7)
    ax.set_xlabel("Position RMSE (m)")
    ax.set_title("Tracking Accuracy Comparison")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, rmse in zip(bars, rmses):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2.0, f"{rmse:.4f}m",
                va="center", fontsize=8)

    # 3. Solve Time 바 차트
    ax = fig.add_subplot(gs[0, 2])
    solve_times = [r["metrics"]["mean_solve_time"] for r in results]

    bars = ax.barh(names, solve_times, color=colors, alpha=0.7)
    ax.set_xlabel("Mean Solve Time (ms)")
    ax.set_title("Computational Performance")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, time_ms in zip(bars, solve_times):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2.0, f"{time_ms:.1f}ms",
                va="center", fontsize=8)

    # 4. 위치 오차 시계열 (상위 3개만)
    ax = fig.add_subplot(gs[1, 0])
    top3_by_rmse = sorted(results, key=lambda x: x["metrics"]["position_rmse"])[:3]

    for result in top3_by_rmse:
        states = result["history"]["state"]
        refs = result["history"]["reference"]
        times = result["history"]["time"]
        errors = np.linalg.norm(states[:, :2] - refs[:, :2], axis=1)

        ax.plot(times, errors, label=result["name"], color=result["color"], linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Error over Time (Top 3)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. Control Rate 비교
    ax = fig.add_subplot(gs[1, 1])
    control_rates = [r["metrics"]["control_rate"] for r in results]

    bars = ax.barh(names, control_rates, color=colors, alpha=0.7)
    ax.set_xlabel("Control Rate (avg Δu)")
    ax.set_title("Control Smoothness")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, rate in zip(bars, control_rates):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2.0, f"{rate:.4f}",
                va="center", fontsize=8)

    # 6. RMSE vs Solve Time 산점도
    ax = fig.add_subplot(gs[1, 2])

    for result in results:
        ax.scatter(
            result["metrics"]["mean_solve_time"],
            result["metrics"]["position_rmse"],
            s=200,
            color=result["color"],
            alpha=0.7,
            label=result["name"],
        )

    ax.set_xlabel("Mean Solve Time (ms)")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Accuracy vs Speed Trade-off")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # 7. 제어 입력 비교 (상위 3개)
    ax = fig.add_subplot(gs[2, 0])

    for result in top3_by_rmse:
        controls = result["history"]["control"]
        times = result["history"]["time"]
        ax.plot(times, controls[:, 0], label=result["name"],
                color=result["color"], linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_title("Control Input (v) - Top 3")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 8. 종합 레이더 차트 (상위 5개)
    ax = fig.add_subplot(gs[2, 1], projection="polar")
    top5 = sorted(results, key=lambda x: x["metrics"]["position_rmse"])[:5]

    categories = ["Accuracy", "Speed", "Smoothness"]
    N_cat = len(categories)

    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]

    for result in top5:
        # 정규화 (역수로 변환하여 높을수록 좋게)
        accuracy = 1.0 / (result["metrics"]["position_rmse"] + 0.001)
        speed = 1.0 / (result["metrics"]["mean_solve_time"] + 1.0)
        smoothness = 1.0 / (result["metrics"]["control_rate"] + 0.01)

        values = [accuracy, speed, smoothness]
        # 정규화 (0-1 범위)
        max_vals = [
            max([1.0 / (r["metrics"]["position_rmse"] + 0.001) for r in results]),
            max([1.0 / (r["metrics"]["mean_solve_time"] + 1.0) for r in results]),
            max([1.0 / (r["metrics"]["control_rate"] + 0.01) for r in results]),
        ]
        values = [v / m for v, m in zip(values, max_vals)]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=result["name"],
                color=result["color"])
        ax.fill(angles, values, alpha=0.15, color=result["color"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("Overall Performance (Top 5)", pad=20)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    # 9. 요약 텍스트
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")

    # 베스트 변형 찾기
    best_accuracy = min(results, key=lambda x: x["metrics"]["position_rmse"])
    best_speed = min(results, key=lambda x: x["metrics"]["mean_solve_time"])
    best_smoothness = min(results, key=lambda x: x["metrics"]["control_rate"])

    summary_text = f"""
    Benchmark Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Best Accuracy:
      {best_accuracy['name']}
      RMSE: {best_accuracy['metrics']['position_rmse']:.4f}m

    Best Speed:
      {best_speed['name']}
      Solve: {best_speed['metrics']['mean_solve_time']:.2f}ms

    Best Smoothness:
      {best_smoothness['name']}
      Rate: {best_smoothness['metrics']['control_rate']:.4f}

    ───────────────────────────────────────

    Recommendations:
    • Real-time: Vanilla, Tube, Spline
    • Accuracy: SVG, SVMPC, Smooth
    • Robustness: Tube, Risk-Aware
    • Memory: Spline (73% reduction)
    • Exploration: Tsallis

    Settings:
      K: {common_params['K']}
      N: {common_params['N']}
      λ: {common_params['lambda_']}
      Trajectory: {args.trajectory}
      Duration: {args.duration}s
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

    plt.savefig("mppi_all_variants_benchmark.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: mppi_all_variants_benchmark.png")
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
