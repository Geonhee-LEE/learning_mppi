#!/usr/bin/env python3
"""
MPPI Variant x Trajectory M×N 그리드 비교 데모

9가지 MPPI 변형 × 5가지 궤적 = 45셀 그리드를 실행하고,
히트맵 + XY 궤적 비교 + ASCII 요약을 제공합니다.

Usage:
    # 빠른 테스트 (2변형 × 2궤적)
    python mppi_variant_trajectory_grid_demo.py --variants vanilla tube --trajectories circle slalom

    # 전체 그리드 (9×5 = 45셀)
    python mppi_variant_trajectory_grid_demo.py

    # 실시간 + 출력 파일
    python mppi_variant_trajectory_grid_demo.py --live --output grid_results.png
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

# ── 공통 파라미터 ──────────────────────────────────────────────
COMMON_PARAMS = {
    "N": 30,
    "dt": 0.05,
    "K": 1024,
    "lambda_": 1.0,
    "sigma": np.array([0.5, 0.5]),
    "Q": np.array([10.0, 10.0, 1.0]),
    "R": np.array([0.1, 0.1]),
    "Qf": np.array([20.0, 20.0, 2.0]),
}

# ── 9 MPPI 변형 정의 ──────────────────────────────────────────
VARIANT_DEFS = {
    "vanilla": {
        "name": "Vanilla",
        "controller_class": MPPIController,
        "params_class": MPPIParams,
        "extra_params": {},
        "color": "blue",
    },
    "tube": {
        "name": "Tube",
        "controller_class": TubeMPPIController,
        "params_class": TubeMPPIParams,
        "extra_params": {
            "tube_enabled": True,
            "tube_margin": 0.3,
            "K_fb": np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        },
        "color": "green",
    },
    "log": {
        "name": "Log",
        "controller_class": LogMPPIController,
        "params_class": LogMPPIParams,
        "extra_params": {"use_baseline": True},
        "color": "red",
    },
    "tsallis": {
        "name": "Tsallis",
        "controller_class": TsallisMPPIController,
        "params_class": TsallisMPPIParams,
        "extra_params": {"tsallis_q": 1.2},
        "color": "purple",
    },
    "risk": {
        "name": "Risk-Aware",
        "controller_class": RiskAwareMPPIController,
        "params_class": RiskAwareMPPIParams,
        "extra_params": {"cvar_alpha": 0.7},
        "color": "orange",
    },
    "smooth": {
        "name": "Smooth",
        "controller_class": SmoothMPPIController,
        "params_class": SmoothMPPIParams,
        "extra_params": {"jerk_weight": 1.0},
        "color": "cyan",
    },
    "svmpc": {
        "name": "SVMPC",
        "controller_class": SteinVariationalMPPIController,
        "params_class": SteinVariationalMPPIParams,
        "extra_params": {"svgd_num_iterations": 3, "svgd_step_size": 0.01},
        "color": "brown",
    },
    "spline": {
        "name": "Spline",
        "controller_class": SplineMPPIController,
        "params_class": SplineMPPIParams,
        "extra_params": {"spline_num_knots": 8, "spline_degree": 3},
        "color": "pink",
    },
    "svg": {
        "name": "SVG",
        "controller_class": SVGMPPIController,
        "params_class": SVGMPPIParams,
        "extra_params": {
            "svg_num_guide_particles": 32,
            "svgd_num_iterations": 3,
            "svg_guide_step_size": 0.01,
        },
        "color": "magenta",
    },
}

# ── 5 궤적 정의 ───────────────────────────────────────────────
TRAJECTORY_DEFS = {
    "circle": {"name": "Circle", "duration": 20.0},
    "figure8": {"name": "Figure-8", "duration": 20.0},
    "sine": {"name": "Sine", "duration": 20.0},
    "slalom": {"name": "Slalom", "duration": 25.0},
    "straight": {"name": "Straight", "duration": 20.0},
}

VARIANT_ORDER = ["vanilla", "tube", "log", "tsallis", "risk", "smooth", "svmpc", "spline", "svg"]
TRAJECTORY_ORDER = ["circle", "figure8", "sine", "slalom", "straight"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPPI Variant x Trajectory Grid Comparison"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=list(VARIANT_DEFS.keys()),
        help="Subset of variants (default: all 9)",
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        default=None,
        choices=list(TRAJECTORY_DEFS.keys()),
        help="Subset of trajectories (default: all 5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override duration for all trajectories (s)",
    )
    parser.add_argument("--live", action="store_true", help="Realtime simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="mppi_variant_trajectory_grid.png",
        help="Output plot filename",
    )
    return parser.parse_args()


def run_grid(variant_keys, trajectory_keys, duration_override, seed, live):
    """M×N 그리드 시뮬레이션 실행."""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    results = {}
    total_cells = len(variant_keys) * len(trajectory_keys)
    cell_idx = 0

    for traj_key in trajectory_keys:
        traj_def = TRAJECTORY_DEFS[traj_key]
        duration = duration_override if duration_override else traj_def["duration"]
        trajectory_fn = create_trajectory_function(traj_key)
        initial_state = trajectory_fn(0.0)

        for var_key in variant_keys:
            cell_idx += 1
            var_def = VARIANT_DEFS[var_key]

            # 공정한 비교를 위해 셀마다 시드 리셋
            np.random.seed(seed)

            # 파라미터 & 컨트롤러 생성
            params = var_def["params_class"](**COMMON_PARAMS, **var_def["extra_params"])
            controller = var_def["controller_class"](model, params)

            # 클로저 캡처를 위한 지역 변수
            _traj_fn = trajectory_fn
            _N = params.N
            _dt = params.dt

            def reference_fn(t, tf=_traj_fn, n=_N, d=_dt):
                return generate_reference_trajectory(tf, t, n, d)

            # 시뮬레이션
            sim = Simulator(model, controller, params.dt)
            sim.reset(initial_state.copy())

            t_start = time.time()
            history = sim.run(reference_fn, duration, realtime=live)
            wall_time = time.time() - t_start

            metrics = compute_metrics(history)

            # 메모리 절약: info (sample_trajectories 포함)를 제거하고
            # 시각화에 필요한 state/reference만 보존
            slim_history = {
                "state": history["state"],
                "reference": history["reference"],
            }
            del history

            results[(var_key, traj_key)] = {
                "metrics": metrics,
                "history": slim_history,
                "wall_time": wall_time,
            }

            print(
                f"  [{cell_idx:2d}/{total_cells}] {var_def['name']:>10s} x {traj_def['name']:<10s} "
                f"| RMSE {metrics['position_rmse']:.4f}m "
                f"| Rate {metrics['control_rate']:.4f} "
                f"| Solve {metrics['mean_solve_time']:.2f}ms"
            )

    return results


def print_ascii_summary(results, variant_keys, trajectory_keys):
    """3개 M×N ASCII 표 출력 (Position RMSE / Control Rate / Solve Time)."""
    var_names = [VARIANT_DEFS[k]["name"] for k in variant_keys]
    traj_names = [TRAJECTORY_DEFS[k]["name"] for k in trajectory_keys]
    n_var = len(variant_keys)
    n_traj = len(trajectory_keys)

    metrics_info = [
        ("Position RMSE (m)", "position_rmse", ".4f"),
        ("Control Rate", "control_rate", ".4f"),
        ("Solve Time (ms)", "mean_solve_time", ".2f"),
    ]

    for title, metric_key, fmt in metrics_info:
        # 데이터 수집
        data = np.zeros((n_var, n_traj))
        for i, vk in enumerate(variant_keys):
            for j, tk in enumerate(trajectory_keys):
                data[i, j] = results[(vk, tk)]["metrics"][metric_key]

        # 열 폭 계산
        val_width = max(len(f"{v:{fmt}}") for v in data.flatten())
        col_width = max(val_width, max(len(n) for n in traj_names)) + 2
        name_width = max(len(n) for n in var_names) + 2

        # 헤더
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")
        header = f"{'':>{name_width}}"
        for tn in traj_names:
            header += f"{tn:>{col_width}}"
        header += f"{'Best':>{col_width}}"
        print(header)
        print(f"{'':>{name_width}}" + "─" * (col_width * (n_traj + 1)))

        # 행별 Best 마킹
        col_bests = data.argmin(axis=0)  # 각 궤적에서 best 변형
        row_bests = data.argmin(axis=1)  # 각 변형에서 best 궤적

        for i, vn in enumerate(var_names):
            row = f"{vn:>{name_width}}"
            row_best_val = f"{data[i, row_bests[i]]:{fmt}}"
            for j in range(n_traj):
                val_str = f"{data[i, j]:{fmt}}"
                if i == col_bests[j]:
                    val_str = f"*{val_str}"
                row += f"{val_str:>{col_width}}"
            row += f"{row_best_val:>{col_width}}"
            print(row)

        # 궤적별 best 변형
        print(f"{'':>{name_width}}" + "─" * (col_width * (n_traj + 1)))
        best_row = f"{'Best':>{name_width}}"
        for j in range(n_traj):
            best_row += f"{var_names[col_bests[j]]:>{col_width}}"
        # 전체 best
        global_best_idx = np.unravel_index(data.argmin(), data.shape)
        global_best_var = var_names[global_best_idx[0]]
        best_row += f"{global_best_var:>{col_width}}"
        print(best_row)

    print(f"\n{'─' * 70}")
    print("  * = best in column (trajectory)")
    print(f"{'─' * 70}\n")


def visualize_grid(results, variant_keys, trajectory_keys, output_file):
    """히트맵 3개 + 궤적별 XY top-3 + 요약 텍스트."""
    import matplotlib.pyplot as plt
    import matplotlib

    var_names = [VARIANT_DEFS[k]["name"] for k in variant_keys]
    traj_names = [TRAJECTORY_DEFS[k]["name"] for k in trajectory_keys]
    n_var = len(variant_keys)
    n_traj = len(trajectory_keys)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    fig.suptitle(
        f"MPPI Variant x Trajectory Grid ({n_var} variants x {n_traj} trajectories)",
        fontsize=16,
        fontweight="bold",
    )

    # ── Row 0: 히트맵 3개 ──────────────────────────────────────
    heatmap_configs = [
        ("Position RMSE (m)", "position_rmse", "Reds", ".4f"),
        ("Control Rate", "control_rate", "Oranges", ".4f"),
        ("Mean Solve Time (ms)", "mean_solve_time", "Blues", ".2f"),
    ]

    for col, (title, metric_key, cmap_name, fmt) in enumerate(heatmap_configs):
        ax = fig.add_subplot(gs[0, col])

        data = np.zeros((n_var, n_traj))
        for i, vk in enumerate(variant_keys):
            for j, tk in enumerate(trajectory_keys):
                data[i, j] = results[(vk, tk)]["metrics"][metric_key]

        im = ax.imshow(data, cmap=cmap_name, aspect="auto")

        # 셀 내 수치 표기 (배경 밝기에 따라 흑/백 전환)
        norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
        for i in range(n_var):
            for j in range(n_traj):
                val = data[i, j]
                rgba = matplotlib.colormaps[cmap_name](norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    j, i, f"{val:{fmt}}",
                    ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold",
                )

        ax.set_xticks(range(n_traj))
        ax.set_xticklabels(traj_names, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(n_var))
        ax.set_yticklabels(var_names, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # ── Row 1-2: 궤적별 XY top-3 + 요약 ──────────────────────
    xy_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

    for plot_idx, traj_key in enumerate(trajectory_keys):
        if plot_idx >= len(xy_positions):
            break
        row, col = xy_positions[plot_idx]
        ax = fig.add_subplot(gs[row, col])

        traj_def = TRAJECTORY_DEFS[traj_key]

        # RMSE 기준 top-3 변형 선택
        rmse_by_var = []
        for vk in variant_keys:
            rmse = results[(vk, traj_key)]["metrics"]["position_rmse"]
            rmse_by_var.append((rmse, vk))
        rmse_by_var.sort()
        top3_keys = [vk for _, vk in rmse_by_var[:3]]

        # Reference 궤적 (첫 번째 결과에서)
        first_key = variant_keys[0]
        ref = results[(first_key, traj_key)]["history"]["reference"]
        ax.plot(
            ref[:, 0], ref[:, 1], "k--",
            linewidth=2, alpha=0.4, label="Reference",
        )

        # Top-3 변형 궤적
        for rank, vk in enumerate(top3_keys):
            var_def = VARIANT_DEFS[vk]
            states = results[(vk, traj_key)]["history"]["state"]
            rmse = results[(vk, traj_key)]["metrics"]["position_rmse"]
            ax.plot(
                states[:, 0], states[:, 1],
                color=var_def["color"],
                linewidth=2.0 - rank * 0.4,
                alpha=0.9 - rank * 0.15,
                label=f"{var_def['name']} ({rmse:.4f}m)",
            )

        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_title(f"{traj_def['name']} — Top 3 by RMSE", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    # ── 요약 텍스트 (Row 2, Col 2) ────────────────────────────
    if len(trajectory_keys) <= 5:
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")

        # 메트릭별 전체 best 찾기
        all_rmse = {
            (vk, tk): results[(vk, tk)]["metrics"]["position_rmse"]
            for vk in variant_keys for tk in trajectory_keys
        }
        all_rate = {
            (vk, tk): results[(vk, tk)]["metrics"]["control_rate"]
            for vk in variant_keys for tk in trajectory_keys
        }
        all_solve = {
            (vk, tk): results[(vk, tk)]["metrics"]["mean_solve_time"]
            for vk in variant_keys for tk in trajectory_keys
        }

        best_rmse_key = min(all_rmse, key=all_rmse.get)
        best_rate_key = min(all_rate, key=all_rate.get)
        best_solve_key = min(all_solve, key=all_solve.get)

        # 변형별 평균 RMSE 순위
        avg_rmse = {}
        for vk in variant_keys:
            vals = [results[(vk, tk)]["metrics"]["position_rmse"] for tk in trajectory_keys]
            avg_rmse[vk] = np.mean(vals)
        ranked = sorted(avg_rmse.items(), key=lambda x: x[1])

        ranking_text = "\n".join(
            f"  {i+1}. {VARIANT_DEFS[vk]['name']:<12s} {avg:.4f}m"
            for i, (vk, avg) in enumerate(ranked[:5])
        )

        summary = (
            f"Grid Summary\n"
            f"{'=' * 38}\n\n"
            f"Best Accuracy:\n"
            f"  {VARIANT_DEFS[best_rmse_key[0]]['name']} x {TRAJECTORY_DEFS[best_rmse_key[1]]['name']}\n"
            f"  RMSE: {all_rmse[best_rmse_key]:.4f}m\n\n"
            f"Best Smoothness:\n"
            f"  {VARIANT_DEFS[best_rate_key[0]]['name']} x {TRAJECTORY_DEFS[best_rate_key[1]]['name']}\n"
            f"  Rate: {all_rate[best_rate_key]:.4f}\n\n"
            f"Best Speed:\n"
            f"  {VARIANT_DEFS[best_solve_key[0]]['name']} x {TRAJECTORY_DEFS[best_solve_key[1]]['name']}\n"
            f"  Solve: {all_solve[best_solve_key]:.2f}ms\n\n"
            f"{'─' * 38}\n"
            f"Avg RMSE Ranking (top 5):\n"
            f"{ranking_text}\n\n"
            f"{'─' * 38}\n"
            f"Cells: {n_var}x{n_traj} = {n_var * n_traj}\n"
            f"K={COMMON_PARAMS['K']}, N={COMMON_PARAMS['N']}"
        )

        ax.text(
            0.05, 0.5, summary,
            fontsize=9, verticalalignment="center", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            transform=ax.transAxes,
        )

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")


def main():
    args = parse_args()

    variant_keys = args.variants if args.variants else list(VARIANT_ORDER)
    trajectory_keys = args.trajectories if args.trajectories else list(TRAJECTORY_ORDER)

    n_var = len(variant_keys)
    n_traj = len(trajectory_keys)
    total = n_var * n_traj

    print("\n" + "=" * 70)
    print("MPPI Variant x Trajectory Grid Comparison".center(70))
    print("=" * 70)
    print(f"  Variants:     {n_var} — {', '.join(VARIANT_DEFS[k]['name'] for k in variant_keys)}")
    print(f"  Trajectories: {n_traj} — {', '.join(TRAJECTORY_DEFS[k]['name'] for k in trajectory_keys)}")
    print(f"  Total cells:  {total}")
    print(f"  Seed:         {args.seed}")
    if args.duration:
        print(f"  Duration:     {args.duration}s (override)")
    print("=" * 70 + "\n")

    # ── 그리드 실행 ────────────────────────────────────────────
    t0 = time.time()
    results = run_grid(variant_keys, trajectory_keys, args.duration, args.seed, args.live)
    elapsed = time.time() - t0
    print(f"\nGrid completed in {elapsed:.1f}s ({elapsed/total:.1f}s/cell)")

    # ── ASCII 요약 ─────────────────────────────────────────────
    print_ascii_summary(results, variant_keys, trajectory_keys)

    # ── 시각화 ─────────────────────────────────────────────────
    print("Generating grid visualization...")
    visualize_grid(results, variant_keys, trajectory_keys, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
