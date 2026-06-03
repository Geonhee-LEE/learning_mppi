#!/usr/bin/env python3
"""
TR-MPPI (Trust Region MPPI) 벤치마크: 4-Way x 4 시나리오 — 41번째 변형

방법:
  1. Vanilla MPPI        — 표준 MPPI (baseline)
  2. TR (KL stochastic)  — KL 신뢰 영역 투영 + 가우시안 샘플링
  3. TR (deterministic)  — Halton LCD 결정론적 샘플링
  4. TR (adaptive cov)   — 공분산 적응 + 엔트로피 하한

시나리오 4개:
  A. simple        — 원형 궤적, 장애물 없음 (기준선)
  B. obstacles     — 3개 장애물, 원형 궤적
  C. dense_slalom  — 다수 장애물 슬라럼
  D. figure8       — 8자 궤적 (공격적 기동)

Usage:
    PYTHONPATH=. python examples/comparison/tr_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/tr_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/tr_mppi_benchmark.py --live --scenario obstacles
    PYTHONPATH=. python examples/comparison/tr_mppi_benchmark.py --no-plot
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
    TRMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tr_mppi import TRMPPIController
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
    K=512, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)

COLORS = {
    "Vanilla MPPI": "#2196F3",
    "TR (KL stochastic)": "#FF9800",
    "TR (deterministic)": "#4CAF50",
    "TR (adaptive cov)": "#E91E63",
}

SHORT = {
    "Vanilla MPPI": "Van",
    "TR (KL stochastic)": "TR-KL",
    "TR (deterministic)": "TR-LCD",
    "TR (adaptive cov)": "TR-Cov",
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
            "description": "No obstacles. Tracking + KL trust region baseline.",
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
            "description": "3 obstacles, circle trajectory. Safety comparison.",
        },
        "dense_slalom": {
            "name": "C. Dense Slalom",
            "obstacles": [
                (3.0, 0.8, 0.35),
                (1.5, 2.6, 0.35),
                (-1.5, 2.6, 0.35),
                (-3.0, 0.8, 0.35),
                (-1.5, -2.6, 0.35),
                (1.5, -2.6, 0.35),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "description": "6 dense obstacles along circle. Stress test.",
        },
        "figure8": {
            "name": "D. Figure-8 (Aggressive)",
            "obstacles": [],
            "trajectory_fn": lambda t: figure_eight_trajectory(
                t, scale=4.0, period=16.0
            ),
            "initial_state": np.array([4.0, 0.0, np.pi / 2]),
            "duration": 16.0,
            "description": "Figure-8 lemniscate. Aggressive maneuver tracking.",
        },
    }


# -- Controller creation --

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
    common = dict(COMMON)

    # 1. Vanilla MPPI
    v_params = MPPIParams(**common)
    vanilla = MPPIController(
        model, v_params, cost_function=_make_cost(v_params, obstacles)
    )

    # 2. TR (KL trust region, stochastic) — 작은 신뢰 영역으로 보수적 업데이트
    tr_kl_params = TRMPPIParams(
        **common,
        trust_region_radius=0.5,
        use_kl_bound=True,
        n_iters=1,
        use_deterministic_sampling=False,
        adapt_covariance=False,
    )
    tr_kl = TRMPPIController(
        model, tr_kl_params, cost_function=_make_cost(tr_kl_params, obstacles)
    )

    # 3. TR (deterministic LCD) — Halton 저불일치 결정론적 샘플링
    tr_det_params = TRMPPIParams(
        **common,
        trust_region_radius=1.0,
        use_kl_bound=True,
        n_iters=1,
        use_deterministic_sampling=True,
        adapt_covariance=False,
    )
    tr_det = TRMPPIController(
        model, tr_det_params, cost_function=_make_cost(tr_det_params, obstacles)
    )

    # 4. TR (adaptive cov + entropy floor) — 공분산 적응
    tr_cov_params = TRMPPIParams(
        **common,
        trust_region_radius=1.0,
        use_kl_bound=True,
        n_iters=1,
        use_deterministic_sampling=False,
        adapt_covariance=True,
        cov_step_size=0.2,
        entropy_floor_scale=0.3,
        cov_max_scale=4.0,
    )
    tr_cov = TRMPPIController(
        model, tr_cov_params, cost_function=_make_cost(tr_cov_params, obstacles)
    )

    return {
        "Vanilla MPPI": vanilla,
        "TR (KL stochastic)": tr_kl,
        "TR (deterministic)": tr_det,
        "TR (adaptive cov)": tr_cov,
    }


# -- Simulation --

def run_single_simulation(model, controller, scenario, seed=42):
    """Run single controller simulation."""
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
    """Compute evaluation metrics."""
    states = history["states"]
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

    # Collisions + MinClearance
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

    # TR stats (KL divergence + scaling)
    kl_list = []
    scaled_count = 0
    for info in history["infos"]:
        if isinstance(info, dict) and "tr_stats" in info:
            kl_list.append(info["tr_stats"]["kl_divergence"])
            if info["tr_stats"]["step_scaled"]:
                scaled_count += 1

    total_steps = len(history["infos"])
    scaled_fraction = scaled_count / total_steps if total_steps > 0 else 0.0
    mean_kl = float(np.mean(kl_list)) if kl_list else 0.0

    return {
        "rmse": rmse,
        "max_error": max_error,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "mean_kl": mean_kl,
        "scaled_fraction": scaled_fraction,
        "errors": errors,
    }


# -- Live animation --

def run_live(args):
    """Realtime 4-way comparison animation -> GIF/MP4 save."""
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
    print(f"  TR-MPPI Live -- {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | dt: {dt_val}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "solve_ms": []}
        for k in controllers
    }

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"TR-MPPI Live -- {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY trajectories
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
    ax_xy.legend(loc="upper left", fontsize=6)

    # [0,1] Tracking error
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, color in COLORS.items():
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5, label=SHORT[name])
    ax_err.legend(fontsize=6)

    # [0,2] Solve time
    ax_st = axes[0, 2]
    ax_st.set_xlabel("Time (s)")
    ax_st.set_ylabel("Solve Time (ms)")
    ax_st.set_title("Solve Time per Step")
    ax_st.grid(True, alpha=0.3)
    lines_st = {}
    for name, color in COLORS.items():
        lines_st[name], = ax_st.plot([], [], color=color, linewidth=1, alpha=0.7, label=SHORT[name])
    ax_st.legend(fontsize=6)

    # [0,3] ESS
    ax_ess = axes[0, 3]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5, label=SHORT[name])
    ax_ess.legend(fontsize=6)

    bar_names = list(COLORS.keys())
    bar_colors = [COLORS[n] for n in bar_names]
    bar_labels = [SHORT[n] for n in bar_names]

    # [1,0] RMSE bar
    ax_rmse = axes[1, 0]
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Running RMSE")
    ax_rmse.grid(True, alpha=0.3, axis="y")
    bars_rmse = ax_rmse.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_rmse.set_xticks(range(len(bar_names)))
    ax_rmse.set_xticklabels(bar_labels, fontsize=8)
    bar_rmse_texts = [
        ax_rmse.text(b.get_x() + b.get_width() / 2, 0, "", ha="center",
                     va="bottom", fontsize=8)
        for b in bars_rmse
    ]

    # [1,1] SolveMs bar
    ax_sm = axes[1, 1]
    ax_sm.set_ylabel("Mean Solve (ms)")
    ax_sm.set_title("Avg Solve Time")
    ax_sm.grid(True, alpha=0.3, axis="y")
    bars_sm = ax_sm.bar(range(len(bar_names)), [0] * len(bar_names),
                        color=bar_colors, alpha=0.8)
    ax_sm.set_xticks(range(len(bar_names)))
    ax_sm.set_xticklabels(bar_labels, fontsize=8)

    # [1,2] MaxError bar
    ax_me = axes[1, 2]
    ax_me.set_ylabel("Max Error (m)")
    ax_me.set_title("Max Tracking Error")
    ax_me.grid(True, alpha=0.3, axis="y")
    bars_me = ax_me.bar(range(len(bar_names)), [0] * len(bar_names),
                        color=bar_colors, alpha=0.8)
    ax_me.set_xticks(range(len(bar_names)))
    ax_me.set_xticklabels(bar_labels, fontsize=8)

    # [1,3] Statistics text
    ax_info = axes[1, 3]
    ax_info.axis("off")
    ax_info.set_title("Statistics")
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        va="top", fontsize=9, family="monospace",
    )

    plt.tight_layout()

    def update(frame):
        if frame >= num_steps:
            return

        t = sim_t[0]
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt_val)

        for name, ctrl in controllers.items():
            t_start_solve = time.time()
            control, info = ctrl.compute_control(states[name], ref)
            solve_ms = (time.time() - t_start_solve) * 1000

            state_dot = model.forward_dynamics(states[name], control)
            states[name] = states[name] + state_dot * dt_val

            ref_pt = trajectory_fn(t)[:2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))
            data[name]["solve_ms"].append(solve_ms)

        sim_t[0] += dt_val

        times = np.array(data["Vanilla MPPI"]["times"])
        for name in controllers:
            xy = np.array(data[name]["xy"])
            if len(xy) > 0:
                lines_xy[name].set_data(xy[:, 0], xy[:, 1])
                dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
                lines_err[name].set_data(
                    times[:len(data[name]["errors"])], data[name]["errors"])
                lines_st[name].set_data(
                    times[:len(data[name]["solve_ms"])], data[name]["solve_ms"])
                lines_ess[name].set_data(
                    times[:len(data[name]["ess"])], data[name]["ess"])

        for ax in (ax_xy, ax_err, ax_st, ax_ess):
            ax.relim()
            ax.autoscale_view()
        ax_xy.set_aspect("equal")

        rmses = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            rmses.append(rmse)
            bars_rmse[i].set_height(rmse)
            bar_rmse_texts[i].set_position(
                (bars_rmse[i].get_x() + bars_rmse[i].get_width() / 2, rmse))
            bar_rmse_texts[i].set_text(f"{rmse:.3f}")
        if rmses:
            ax_rmse.set_ylim(0, max(rmses) * 1.3 + 0.01)

        solve_avgs = []
        for i, name in enumerate(bar_names):
            ms_arr = data[name]["solve_ms"]
            avg = np.mean(ms_arr) if ms_arr else 0
            solve_avgs.append(avg)
            bars_sm[i].set_height(avg)
        if solve_avgs:
            ax_sm.set_ylim(0, max(solve_avgs) * 1.3 + 0.01)

        max_errs = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            me = max(errs) if errs else 0
            max_errs.append(me)
            bars_me[i].set_height(me)
        if max_errs:
            ax_me.set_ylim(0, max(max_errs) * 1.3 + 0.01)

        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            avg_ms = np.mean(data[name]["solve_ms"]) if data[name]["solve_ms"] else 0
            lines.append(f"{SHORT[name]:>7}: RMSE={rmse:.3f} ESS={ess:.0f} ms={avg_ms:.1f}")
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/tr_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/tr_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    print(f"\n{'=' * 72}")
    print(f"  Final Statistics -- {scenario['name']}")
    print(f"{'=' * 72}")
    print(f"  {'Method':<20} {'RMSE':>8} {'MaxError':>10} {'MeanESS':>10} {'AvgMs':>10}")
    print(f"  {'-' * 60}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        me = max(errs) if errs else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        avg_ms = np.mean(data[name]["solve_ms"]) if data[name]["solve_ms"] else 0
        print(f"  {name:<20} {rmse:>8.4f} {me:>10.4f} {mean_ess:>10.1f} {avg_ms:>10.2f}")
    print(f"{'=' * 72}\n")


# -- Benchmark main --

def run_benchmark(args):
    """Static benchmark execution."""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    dt_val = COMMON["dt"]

    print(f"\n{'=' * 80}")
    print(f"  TR-MPPI Benchmark: 4-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {scenario['duration']}s | dt: {dt_val}s | Seed: {args.seed}")
    print(f"{'=' * 80}")

    model = DifferentialDriveKinematic(wheelbase=0.5)
    controllers = _make_controllers(model, scenario)

    all_results = []
    for i, (name, ctrl) in enumerate(controllers.items()):
        np.random.seed(args.seed)

        print(f"\n  [{i+1}/{len(controllers)}] {name:<22}", end=" ", flush=True)
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

    has_obstacles = len(scenario["obstacles"]) > 0
    print(f"\n{'=' * 104}")
    header = (
        f"{'Method':<20} {'RMSE':>8} {'MaxError':>10} "
        f"{'MeanESS':>10} {'SolveMs':>10} {'MeanKL':>10} {'ScaledFr':>10}"
    )
    if has_obstacles:
        header += f" {'Collisions':>10} {'MinClear':>10}"
    print(header)
    print(f"{'=' * 104}")
    for r in all_results:
        line = (
            f"{r['name']:<20} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>10.4f} "
            f"{r['mean_ess']:>10.1f} "
            f"{r['mean_solve_ms']:>10.2f} "
            f"{r['mean_kl']:>10.4f} "
            f"{r['scaled_fraction']:>10.2f}"
        )
        if has_obstacles:
            line += f" {r['n_collisions']:>10d} {r['min_clearance']:>10.3f}"
        print(line)
    print(f"{'=' * 104}")

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-panel result plot (2x4)."""
    dt_val = COMMON["dt"]
    trajectory_fn = scenario["trajectory_fn"]
    duration = scenario["duration"]

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

    # (0,2) Solve time
    ax = axes[0, 2]
    for r in results:
        t_st = np.arange(len(r["solve_times_ms"])) * dt_val
        ax.plot(t_st, r["solve_times_ms"], color=r["color"],
                label=r["short"], linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Per-Step Solve Time")
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

    # (1,0) RMSE bar
    ax = axes[1, 0]
    rmses = [r["rmse"] for r in results]
    bars = ax.bar(names, rmses, color=colors, alpha=0.8)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1) SolveMs bar
    ax = axes[1, 1]
    solve_ms_list = [r["mean_solve_ms"] for r in results]
    bars_sm = ax.bar(names, solve_ms_list, color=colors, alpha=0.8)
    for bar, val in zip(bars_sm, solve_ms_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Avg Solve (ms)")
    ax.set_title("Average Solve Time")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,2) MaxError / MinClearance bar
    ax = axes[1, 2]
    if scenario["obstacles"]:
        min_clears = [r["min_clearance"] for r in results]
        bars_mc = ax.bar(names, min_clears, color=colors, alpha=0.8)
        for bar, val in zip(bars_mc, min_clears):
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Min Clearance (m)")
        ax.set_title("Min Obstacle Clearance")
    else:
        max_errors = [r["max_error"] for r in results]
        bars_me = ax.bar(names, max_errors, color=colors, alpha=0.8)
        for bar, val in zip(bars_me, max_errors):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Max Error (m)")
        ax.set_title("Max Tracking Error")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) Mean KL / scaled fraction bar
    ax = axes[1, 3]
    mean_kls = [r["mean_kl"] for r in results]
    bars_kl = ax.bar(names, mean_kls, color=colors, alpha=0.8)
    for r, bar, val in zip(results, bars_kl, mean_kls):
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.001,
                f"{val:.3f}\n({r['scaled_fraction']:.0%})",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean KL divergence")
    ax.set_title("Trust Region KL (scaled frac)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"TR-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs TR-KL vs TR-LCD vs TR-Cov",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/tr_mppi_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="TR-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "dense_slalom", "figure8"],
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
