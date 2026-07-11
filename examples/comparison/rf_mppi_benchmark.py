#!/usr/bin/env python3
"""
RF-MPPI (Reference-Free Spline MPPI) 벤치마크: 4-Way x 4 시나리오

핵심 비교 포인트: 매끄러움(SMOOTHNESS) + 적은 샘플(FEW SAMPLES)
  RF-MPPI는 제어 시퀀스를 저차원 Hermite 스플라인으로 파라미터화하여,
  소수 샘플(K≈48)로도 매끄럽고 다양한 모션을 탐색한다.

방법:
  1. Vanilla MPPI (K=512)     — 충분한 샘플의 기준선
  2. Vanilla MPPI (K=48)      — 적은 샘플 -> 성능/매끄러움 저하 (대조군)
  3. RF (Hermite, K=48, pos)  — 위치 knot만 샘플 (sample_velocity_knots=False)
  4. RF (dual, K=48)          — 위치+속도 knot 샘플 (dual-space, =True)

시나리오 4개:
  A. simple       — 원형 궤적, 장애물 없음 (기준선 비교)
  B. obstacles    — 3개 장애물, 원형 궤적 (안전성)
  C. aggressive   — figure8 궤적 (매끄러움 vs 민첩성)
  D. few_sample   — 적은 K 강조 (RF 구조적 매끄러움 이점)

Usage:
    PYTHONPATH=. python examples/comparison/rf_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/rf_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/rf_mppi_benchmark.py --live --scenario obstacles
    PYTHONPATH=. python examples/comparison/rf_mppi_benchmark.py --no-plot
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    RFMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.rf_mppi import RFMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
    figure_eight_trajectory,
)


# -- Common settings --

COMMON = dict(
    N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)

K_FULL = 512   # Vanilla 충분 샘플
K_FEW = 48     # 적은 샘플 (RF + Vanilla few)

COLORS = {
    "Vanilla (K=512)": "#2196F3",
    "Vanilla (K=48)": "#9E9E9E",
    "RF (pos, K=48)": "#FF9800",
    "RF (dual, K=48)": "#E91E63",
}

SHORT = {
    "Vanilla (K=512)": "Van512",
    "Vanilla (K=48)": "Van48",
    "RF (pos, K=48)": "RF-pos",
    "RF (dual, K=48)": "RF-dual",
}


# -- Scenario definitions --

def get_scenarios():
    """4 benchmark scenarios."""
    return {
        "simple": {
            "name": "A. Simple Tracking (Baseline)",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "k_few": K_FEW,
            "description": "No obstacles. Smoothness + tracking baseline.",
        },
        "obstacles": {
            "name": "B. Obstacle Avoidance",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.5),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "k_few": K_FEW,
            "description": "3 obstacles, circle. Safety with few samples.",
        },
        "aggressive": {
            "name": "C. Aggressive Figure-8",
            "obstacles": [],
            "trajectory_fn": lambda t: figure_eight_trajectory(t, scale=3.0),
            "initial_state": np.array([0.0, 0.0, 0.0]),
            "duration": 18.0,
            "k_few": K_FEW,
            "description": "Figure-8: smoothness vs agility trade-off.",
        },
        "few_sample": {
            "name": "D. Few-Sample Stress (K=24)",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (-1.0, 2.5, 0.4),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "k_few": 24,
            "description": "Very low K=24. RF structural smoothness advantage.",
        },
    }


# -- Cost / Controller creation --

def _make_cost(params, obstacles):
    """Base cost + obstacle cost."""
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0))
    return CompositeMPPICost(costs)


def _make_controllers(model, scenario):
    """Create 4 controllers."""
    obstacles = scenario["obstacles"]
    k_few = scenario.get("k_few", K_FEW)

    # 1. Vanilla MPPI (K=512)
    v_full_params = MPPIParams(K=K_FULL, **COMMON)
    vanilla_full = MPPIController(
        model, v_full_params,
        cost_function=_make_cost(v_full_params, obstacles),
    )

    # 2. Vanilla MPPI (적은 샘플 K=48)
    v_few_params = MPPIParams(K=k_few, **COMMON)
    vanilla_few = MPPIController(
        model, v_few_params,
        cost_function=_make_cost(v_few_params, obstacles),
    )

    # 3. RF-MPPI (Hermite, 위치 knot만)
    rf_pos_params = RFMPPIParams(
        K=k_few, **COMMON,
        n_knots=6,
        sample_velocity_knots=False,
        knot_sigma_vel=0.3,
        spline_warm_shift=True,
    )
    rf_pos = RFMPPIController(
        model, rf_pos_params,
        cost_function=_make_cost(rf_pos_params, obstacles),
    )

    # 4. RF-MPPI (dual-space, 위치+속도 knot)
    rf_dual_params = RFMPPIParams(
        K=k_few, **COMMON,
        n_knots=6,
        sample_velocity_knots=True,
        knot_sigma_vel=0.3,
        spline_warm_shift=True,
    )
    rf_dual = RFMPPIController(
        model, rf_dual_params,
        cost_function=_make_cost(rf_dual_params, obstacles),
    )

    return {
        "Vanilla (K=512)": vanilla_full,
        "Vanilla (K=48)": vanilla_few,
        "RF (pos, K=48)": rf_pos,
        "RF (dual, K=48)": rf_dual,
    }


# -- Simulation --

def run_single_simulation(model, controller, scenario, seed=42):
    """단일 컨트롤러 시뮬레이션."""
    np.random.seed(seed)

    dt_val = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt_val)
    trajectory_fn = scenario["trajectory_fn"]

    state = scenario["initial_state"].copy()

    states = [state.copy()]
    controls_hist = []
    solve_times = []
    infos = []

    for step in range(num_steps):
        t = step * dt_val
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt_val)

        t_start = time.time()
        control, info = controller.compute_control(state, ref)
        solve_time = time.time() - t_start

        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt_val

        states.append(state.copy())
        controls_hist.append(control.copy())
        solve_times.append(solve_time)
        infos.append(info)

    return {
        "states": np.array(states),
        "controls": np.array(controls_hist) if controls_hist else np.array([]),
        "solve_times": np.array(solve_times),
        "infos": infos,
    }


def compute_metrics(history, scenario):
    """평가 지표 계산 (매끄러움 MSSD/Jerk 포함)."""
    states = history["states"]
    controls = history["controls"]
    trajectory_fn = scenario["trajectory_fn"]
    dt_val = COMMON["dt"]
    obstacles = scenario["obstacles"]

    # RMSE + MaxError
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt_val)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    max_error = float(np.max(errors))

    # 매끄러움: 적용된 제어 시퀀스의 MSSD + 평균 |Δu| (jerk proxy)
    if controls.ndim == 2 and len(controls) > 1:
        du = np.diff(controls, axis=0)            # (T-1, nu)
        mssd = float(np.mean(du ** 2))
        mean_abs_du = float(np.mean(np.abs(du)))
    else:
        mssd = 0.0
        mean_abs_du = 0.0

    # Collisions / min clearance
    n_collisions = 0
    min_clearance = float("inf")
    for st in states:
        for ox, oy, r in obstacles:
            dist = np.sqrt((st[0] - ox) ** 2 + (st[1] - oy) ** 2)
            clearance = dist - r
            min_clearance = min(min_clearance, clearance)
            if clearance < 0:
                n_collisions += 1

    # ESS
    ess_list = [
        info.get("ess", 0.0) for info in history["infos"]
        if isinstance(info, dict) and "ess" in info
    ]

    return {
        "rmse": rmse,
        "max_error": max_error,
        "mssd": mssd,
        "mean_abs_du": mean_abs_du,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "errors": errors,
    }


# -- Benchmark main --

def run_benchmark(args):
    """정적 벤치마크 실행."""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    dt_val = COMMON["dt"]

    print(f"\n{'=' * 80}")
    print(f"  RF-MPPI Benchmark: 4-Way Comparison (Smoothness + Few Samples)")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {scenario['duration']}s | dt: {dt_val}s | Seed: {args.seed}")
    print(f"{'=' * 80}")

    model = DifferentialDriveKinematic(wheelbase=0.5)
    controllers = _make_controllers(model, scenario)

    all_results = []
    for i, (name, ctrl) in enumerate(controllers.items()):
        np.random.seed(args.seed)

        print(f"\n  [{i+1}/{len(controllers)}] {name:<20}", end=" ", flush=True)
        t_start = time.time()

        history = run_single_simulation(model, ctrl, scenario, seed=args.seed)
        elapsed = time.time() - t_start

        metrics = compute_metrics(history, scenario)

        all_results.append({
            "name": name,
            "short": SHORT[name],
            "color": COLORS[name],
            "states": history["states"],
            "controls": history["controls"],
            "infos": history["infos"],
            "elapsed": elapsed,
            "solve_times_ms": history["solve_times"] * 1000,
            **metrics,
        })

        print(f"done ({elapsed:.1f}s)")

    # Results table
    has_obstacles = len(scenario["obstacles"]) > 0
    print(f"\n{'=' * 104}")
    header = (
        f"{'Method':<18} {'RMSE':>8} {'MaxErr':>8} "
        f"{'MSSD':>9} {'Mean|du|':>9} {'MeanESS':>9} {'SolveMs':>9}"
    )
    if has_obstacles:
        header += f" {'Collis':>7} {'MinClear':>9}"
    print(header)
    print(f"{'=' * 104}")
    for r in all_results:
        line = (
            f"{r['name']:<18} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>8.4f} "
            f"{r['mssd']:>9.5f} "
            f"{r['mean_abs_du']:>9.4f} "
            f"{r['mean_ess']:>9.1f} "
            f"{r['mean_solve_ms']:>9.2f}"
        )
        if has_obstacles:
            line += f" {r['n_collisions']:>7d} {r['min_clearance']:>9.3f}"
        print(line)
    print(f"{'=' * 104}")

    # 매끄러움 요약
    van_few = next((r for r in all_results if r["name"] == "Vanilla (K=48)"), None)
    rf_dual = next((r for r in all_results if r["name"] == "RF (dual, K=48)"), None)
    if van_few and rf_dual and van_few["mssd"] > 0:
        ratio = van_few["mssd"] / max(rf_dual["mssd"], 1e-12)
        print(f"  매끄러움: RF-dual MSSD {rf_dual['mssd']:.5f} vs "
              f"Van48 {van_few['mssd']:.5f}  (~{ratio:.1f}x smoother)")
        print(f"{'=' * 104}")

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-panel result plot (2x4)."""
    dt_val = COMMON["dt"]
    trajectory_fn = scenario["trajectory_fn"]
    duration = scenario["duration"]
    has_obstacles = len(scenario["obstacles"]) > 0

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # (0,0) XY trajectories
    ax = axes[0, 0]
    t_arr = np.linspace(0, duration, 500)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, label="Ref", linewidth=1)
    for r in results:
        ax.plot(r["states"][:, 0], r["states"][:, 1], color=r["color"],
                label=r["short"], linewidth=1.5, alpha=0.8)
    for ox, oy, rad in scenario["obstacles"]:
        ax.add_patch(Circle((ox, oy), rad, facecolor="#FF5252", edgecolor="red",
                            alpha=0.3, linewidth=1.5))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # (0,1) Tracking error
    ax = axes[0, 1]
    for r in results:
        t_plot = np.arange(len(r["errors"])) * dt_val
        ax.plot(t_plot, r["errors"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Control signal v (속도) — 매끄러움 시각화
    ax = axes[0, 2]
    for r in results:
        if r["controls"].ndim == 2 and len(r["controls"]) > 0:
            t_u = np.arange(len(r["controls"])) * dt_val
            ax.plot(t_u, r["controls"][:, 0], color=r["color"],
                    label=r["short"], linewidth=1, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control v (m/s)")
    ax.set_title("Linear Velocity (smoothness)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,3) ESS
    ax = axes[0, 3]
    for r in results:
        if r["ess_list"]:
            t_ess = np.arange(len(r["ess_list"])) * dt_val
            ax.plot(t_ess, r["ess_list"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    names = [r["short"] for r in results]
    colors = [r["color"] for r in results]

    # (1,0) RMSE bar chart
    ax = axes[1, 0]
    rmses = [r["rmse"] for r in results]
    bars = ax.bar(names, rmses, color=colors, alpha=0.8)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1) MSSD bar chart (매끄러움 핵심 지표)
    ax = axes[1, 1]
    mssds = [r["mssd"] for r in results]
    bars_m = ax.bar(names, mssds, color=colors, alpha=0.8)
    for bar, val in zip(bars_m, mssds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Control MSSD")
    ax.set_title("Smoothness (lower = smoother)")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,2) SolveMs bar chart
    ax = axes[1, 2]
    solve_ms_list = [r["mean_solve_ms"] for r in results]
    bars_sm = ax.bar(names, solve_ms_list, color=colors, alpha=0.8)
    for bar, val in zip(bars_sm, solve_ms_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Avg Solve (ms)")
    ax.set_title("Average Solve Time")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) MinClearance (장애물) / MeanESS
    ax = axes[1, 3]
    if has_obstacles:
        min_clears = [r["min_clearance"] for r in results]
        bars_mc = ax.bar(names, min_clears, color=colors, alpha=0.8)
        for bar, val in zip(bars_mc, min_clears):
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Min Clearance (m)")
        ax.set_title("Min Obstacle Clearance")
    else:
        ess_means = [r["mean_ess"] for r in results]
        bars_e = ax.bar(names, ess_means, color=colors, alpha=0.8)
        for bar, val in zip(bars_e, ess_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Mean ESS")
        ax.set_title("Mean Effective Sample Size")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"RF-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla(512) vs Vanilla(48) vs RF-pos vs RF-dual",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/rf_mppi_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# -- Live animation --

def run_live(args):
    """실시간 4-way 비교 애니메이션 -> GIF/MP4 저장."""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    trajectory_fn = scenario["trajectory_fn"]
    dt_val = COMMON["dt"]

    model = DifferentialDriveKinematic(wheelbase=0.5)
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt_val)

    np.random.seed(args.seed)
    controllers = _make_controllers(model, scenario)

    print(f"\n{'=' * 60}")
    print(f"  RF-MPPI Live -- {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | dt: {dt_val}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "v": []}
        for k in controllers
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"RF-MPPI Live -- {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")
    for ox, oy, r in scenario["obstacles"]:
        ax_xy.add_patch(Circle((ox, oy), r, color="red", alpha=0.3))
    ref_t_arr = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t) for t in ref_t_arr])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")
    lines_xy = {}
    dots = {}
    for name, color in COLORS.items():
        lines_xy[name], = ax_xy.plot([], [], color=color, linewidth=2, label=SHORT[name])
        dots[name], = ax_xy.plot([], [], "o", color=color, markersize=8)
    ax_xy.legend(loc="upper left", fontsize=7)

    # [0,1] Error
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, color in COLORS.items():
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5, label=SHORT[name])
    ax_err.legend(fontsize=7)

    # [0,2] Control v (smoothness)
    ax_v = axes[0, 2]
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Control v (m/s)")
    ax_v.set_title("Linear Velocity (smoothness)")
    ax_v.grid(True, alpha=0.3)
    lines_v = {}
    for name, color in COLORS.items():
        lines_v[name], = ax_v.plot([], [], color=color, linewidth=1, alpha=0.8, label=SHORT[name])
    ax_v.legend(fontsize=7)

    # [1,0] RMSE bars
    ax_rmse = axes[1, 0]
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Running RMSE")
    ax_rmse.grid(True, alpha=0.3, axis="y")
    bar_names = list(COLORS.keys())
    bar_colors = [COLORS[n] for n in bar_names]
    bar_labels = [SHORT[n] for n in bar_names]
    bars_rmse = ax_rmse.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_rmse.set_xticks(range(len(bar_names)))
    ax_rmse.set_xticklabels(bar_labels, fontsize=8)

    # [1,1] MSSD bars
    ax_mssd = axes[1, 1]
    ax_mssd.set_ylabel("Control MSSD")
    ax_mssd.set_title("Smoothness (lower=smoother)")
    ax_mssd.grid(True, alpha=0.3, axis="y")
    bars_mssd = ax_mssd.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_mssd.set_xticks(range(len(bar_names)))
    ax_mssd.set_xticklabels(bar_labels, fontsize=8)

    # [1,2] Stats text
    ax_info = axes[1, 2]
    ax_info.axis("off")
    ax_info.set_title("Statistics")
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        va="top", fontsize=10, family="monospace",
    )

    plt.tight_layout()

    def update(frame):
        if frame >= num_steps:
            return
        t = sim_t[0]
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt_val)

        for name, ctrl in controllers.items():
            control, info = ctrl.compute_control(states[name], ref)
            state_dot = model.forward_dynamics(states[name], control)
            states[name] = states[name] + state_dot * dt_val

            ref_pt = trajectory_fn(t)[:2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))
            data[name]["v"].append(float(control[0]))

        sim_t[0] += dt_val
        times = np.array(data["Vanilla (K=512)"]["times"])

        for name in controllers:
            xy = np.array(data[name]["xy"])
            if len(xy) > 0:
                lines_xy[name].set_data(xy[:, 0], xy[:, 1])
                dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
                lines_err[name].set_data(
                    times[:len(data[name]["errors"])], data[name]["errors"])
                lines_v[name].set_data(
                    times[:len(data[name]["v"])], data[name]["v"])

        for a in (ax_xy, ax_err, ax_v):
            a.relim()
            a.autoscale_view()
        ax_xy.set_aspect("equal")

        rmses, mssds = [], []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            rmses.append(rmse)
            bars_rmse[i].set_height(rmse)
            v_arr = np.array(data[name]["v"])
            mssd = float(np.mean(np.diff(v_arr) ** 2)) if len(v_arr) > 1 else 0.0
            mssds.append(mssd)
            bars_mssd[i].set_height(mssd)
        if rmses:
            ax_rmse.set_ylim(0, max(rmses) * 1.3 + 0.01)
        if mssds:
            ax_mssd.set_ylim(0, max(mssds) * 1.3 + 1e-4)

        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            v_arr = np.array(data[name]["v"])
            mssd = float(np.mean(np.diff(v_arr) ** 2)) if len(v_arr) > 1 else 0.0
            lines.append(f"{SHORT[name]:>8}: RMSE={rmse:.3f} ESS={ess:.0f} MSSD={mssd:.4f}")
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/rf_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/rf_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="RF-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "aggressive", "few_sample"],
    )
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true", help="Realtime animation")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    scenarios = get_scenarios()

    if args.live:
        if args.all_scenarios:
            for scenario_name in scenarios:
                args.scenario = scenario_name
                run_live(args)
        else:
            run_live(args)
    elif args.all_scenarios:
        for scenario_name in scenarios:
            args.scenario = scenario_name
            run_benchmark(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
