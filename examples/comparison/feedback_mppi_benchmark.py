#!/usr/bin/env python3
"""
Feedback-MPPI (F-MPPI) 벤치마크: 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI   — 매 스텝 재최적화 (baseline)
  2. Tube-MPPI      — 분리된 피드백 (보수적)
  3. Robust MPPI    — 통합 피드백 + 외란 샘플링
  4. Feedback MPPI  — Riccati 피드백으로 MPPI 해 재사용

시나리오 4개:
  A. simple          — 원형 궤적, 장애물 없음 (기준선 비교)
  B. obstacles       — 3개 장애물, 원형 궤적
  C. high_frequency  — dt=0.02, 재사용으로 고주파 제어
  D. perturbation    — 프로세스 노이즈 + 피드백 보정

Usage:
    PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --live --scenario obstacles
    PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --no-plot
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
    TubeMPPIParams,
    RobustMPPIParams,
    FeedbackMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.robust_mppi import RobustMPPIController
from mppi_controller.controllers.mppi.feedback_mppi import FeedbackMPPIController
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
    "Tube-MPPI": "#FF9800",
    "Robust MPPI": "#4CAF50",
    "Feedback MPPI": "#E91E63",
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
            "process_noise_std": np.array([0.0, 0.0, 0.0]),
            "dt_override": None,
            "description": "No obstacles, no noise. Timing baseline comparison.",
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
            "process_noise_std": np.array([0.0, 0.0, 0.0]),
            "dt_override": None,
            "description": "3 obstacles, circle trajectory. Safety comparison.",
        },
        "high_frequency": {
            "name": "C. High-Frequency Control (dt=0.02)",
            "obstacles": [
                (2.5, 1.5, 0.4),
                (-1.0, 2.5, 0.4),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 10.0,
            "process_noise_std": np.array([0.0, 0.0, 0.0]),
            "dt_override": 0.02,
            "description": "dt=0.02 (50Hz). F-MPPI reuse advantage for faster control.",
        },
        "perturbation": {
            "name": "D. Process Noise + Perturbation",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.5),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "process_noise_std": np.array([0.05, 0.05, 0.02]),
            "dt_override": None,
            "description": "Process noise + obstacles. Feedback correction advantage.",
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
    noise_std = scenario["process_noise_std"]
    dt_val = scenario.get("dt_override") or COMMON["dt"]

    common = dict(COMMON)
    common["dt"] = dt_val

    # Vanilla MPPI
    v_params = MPPIParams(**common)
    vanilla = MPPIController(model, v_params, cost_function=_make_cost(v_params, obstacles))

    # Tube-MPPI
    t_params = TubeMPPIParams(**common, tube_enabled=True)
    tube = TubeMPPIController(model, t_params)
    if obstacles:
        tube.cost_function = _make_cost(t_params, obstacles)

    # Robust MPPI
    dist_std = noise_std.tolist() if np.any(noise_std > 0) else [0.03, 0.03, 0.01]
    rob_params = RobustMPPIParams(
        **common,
        disturbance_std=dist_std,
        feedback_gain_scale=1.0,
        disturbance_mode="gaussian",
        robust_alpha=0.8,
        use_feedback=True,
        n_disturbance_samples=1,
    )
    robust = RobustMPPIController(
        model, rob_params,
        cost_function=_make_cost(rob_params, obstacles),
    )

    # Feedback MPPI
    fb_params = FeedbackMPPIParams(
        **common,
        reuse_steps=3,
        jacobian_eps=1e-4,
        feedback_weight_Q=10.0,
        feedback_weight_R=0.1,
        use_feedback=True,
        feedback_gain_clip=10.0,
        use_warm_start=True,
    )
    feedback = FeedbackMPPIController(
        model, fb_params,
        cost_function=_make_cost(fb_params, obstacles),
    )

    return {
        "Vanilla MPPI": vanilla,
        "Tube-MPPI": tube,
        "Robust MPPI": robust,
        "Feedback MPPI": feedback,
    }


# -- Simulation --

def run_single_simulation(model, controller, scenario, seed=42):
    """Run single controller simulation."""
    np.random.seed(seed)

    dt_val = scenario.get("dt_override") or COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt_val)
    trajectory_fn = scenario["trajectory_fn"]
    process_noise_std = scenario["process_noise_std"]

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

        # Forward dynamics + process noise
        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt_val
        if np.any(process_noise_std > 0):
            state = state + np.random.randn(len(state)) * process_noise_std

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
    dt_val = scenario.get("dt_override") or COMMON["dt"]
    obstacles = scenario["obstacles"]

    # RMSE + MaxError
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt_val)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    max_error = float(np.max(errors))

    # Collisions
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

    # Feedback stats (F-MPPI only)
    reuse_fraction = 0.0
    mean_feedback_gain = 0.0
    full_solve_count = 0
    reuse_count = 0
    for info in history["infos"]:
        if isinstance(info, dict) and "feedback_stats" in info:
            fb = info["feedback_stats"]
            if fb["mode"] == "feedback_reuse":
                reuse_count += 1
            else:
                full_solve_count += 1
            mean_feedback_gain += fb.get("mean_gain", 0.0)

    total_steps = len(history["infos"])
    reuse_fraction = reuse_count / total_steps if total_steps > 0 else 0.0
    mean_feedback_gain = mean_feedback_gain / total_steps if total_steps > 0 else 0.0

    return {
        "rmse": rmse,
        "max_error": max_error,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "reuse_fraction": reuse_fraction,
        "mean_feedback_gain": mean_feedback_gain,
        "full_solve_count": full_solve_count,
        "reuse_count": reuse_count,
        "errors": errors,
    }


# -- Live animation --

def run_live(args):
    """Realtime 4-way comparison animation -> GIF/MP4 save."""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    trajectory_fn = scenario["trajectory_fn"]
    process_noise_std = scenario["process_noise_std"]
    dt_val = scenario.get("dt_override") or COMMON["dt"]

    model = DifferentialDriveKinematic(wheelbase=0.5)
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt_val)

    np.random.seed(args.seed)
    controllers = _make_controllers(model, scenario)

    print(f"\n{'=' * 60}")
    print(f"  F-MPPI Live -- {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | dt: {dt_val}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # State init
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "solve_ms": []}
        for k in controllers
    }

    # Figure setup (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"F-MPPI Live -- {scenario['name']}",
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
        lines_xy[name], = ax_xy.plot([], [], color=color, linewidth=2, label=name)
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
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5, label=name)
    ax_err.legend(fontsize=6)

    # [0,2] Solve time
    ax_st = axes[0, 2]
    ax_st.set_xlabel("Time (s)")
    ax_st.set_ylabel("Solve Time (ms)")
    ax_st.set_title("Solve Time per Step")
    ax_st.grid(True, alpha=0.3)
    lines_st = {}
    for name, color in COLORS.items():
        lines_st[name], = ax_st.plot([], [], color=color, linewidth=1, alpha=0.7, label=name)
    ax_st.legend(fontsize=6)

    # [0,3] ESS
    ax_ess = axes[0, 3]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5, label=name)
    ax_ess.legend(fontsize=6)

    # [1,0] RMSE bar chart
    ax_rmse = axes[1, 0]
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Running RMSE")
    ax_rmse.grid(True, alpha=0.3, axis="y")
    bar_names = list(COLORS.keys())
    bar_colors = [COLORS[n] for n in bar_names]
    bars_rmse = ax_rmse.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_rmse.set_xticks(range(len(bar_names)))
    ax_rmse.set_xticklabels(["Van", "Tube", "Rob", "F-MPPI"], fontsize=8)
    bar_rmse_texts = [
        ax_rmse.text(b.get_x() + b.get_width() / 2, 0, "", ha="center",
                     va="bottom", fontsize=8)
        for b in bars_rmse
    ]

    # [1,1] SolveMs bar chart
    ax_sm = axes[1, 1]
    ax_sm.set_ylabel("Mean Solve (ms)")
    ax_sm.set_title("Avg Solve Time")
    ax_sm.grid(True, alpha=0.3, axis="y")
    bars_sm = ax_sm.bar(range(len(bar_names)), [0] * len(bar_names),
                        color=bar_colors, alpha=0.8)
    ax_sm.set_xticks(range(len(bar_names)))
    ax_sm.set_xticklabels(["Van", "Tube", "Rob", "F-MPPI"], fontsize=8)

    # [1,2] MaxError bar chart
    ax_me = axes[1, 2]
    ax_me.set_ylabel("Max Error (m)")
    ax_me.set_title("Max Tracking Error")
    ax_me.grid(True, alpha=0.3, axis="y")
    bars_me = ax_me.bar(range(len(bar_names)), [0] * len(bar_names),
                        color=bar_colors, alpha=0.8)
    ax_me.set_xticks(range(len(bar_names)))
    ax_me.set_xticklabels(["Van", "Tube", "Rob", "F-MPPI"], fontsize=8)

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
            if np.any(process_noise_std > 0):
                states[name] = states[name] + np.random.randn(3) * process_noise_std

            ref_pt = trajectory_fn(t)[:2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))
            data[name]["solve_ms"].append(solve_ms)

        sim_t[0] += dt_val

        # Update plots
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

        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal")
        ax_err.relim()
        ax_err.autoscale_view()
        ax_st.relim()
        ax_st.autoscale_view()
        ax_ess.relim()
        ax_ess.autoscale_view()

        # RMSE bars
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

        # SolveMs bars
        solve_avgs = []
        for i, name in enumerate(bar_names):
            ms_arr = data[name]["solve_ms"]
            avg = np.mean(ms_arr) if ms_arr else 0
            solve_avgs.append(avg)
            bars_sm[i].set_height(avg)
        if solve_avgs:
            ax_sm.set_ylim(0, max(solve_avgs) * 1.3 + 0.01)

        # MaxError bars
        max_errs = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            me = max(errs) if errs else 0
            max_errs.append(me)
            bars_me[i].set_height(me)
        if max_errs:
            ax_me.set_ylim(0, max(max_errs) * 1.3 + 0.01)

        # Stats text
        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            avg_ms = np.mean(data[name]["solve_ms"]) if data[name]["solve_ms"] else 0
            short = name.replace(" MPPI", "").replace("-MPPI", "")
            lines.append(f"{short:>10}: RMSE={rmse:.3f} ESS={ess:.0f} ms={avg_ms:.1f}")
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/feedback_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/feedback_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # Final statistics
    print(f"\n{'=' * 72}")
    print(f"  Final Statistics -- {scenario['name']}")
    print(f"{'=' * 72}")
    print(f"  {'Method':<18} {'RMSE':>8} {'MaxError':>10} {'MeanESS':>10} {'AvgMs':>10}")
    print(f"  {'-' * 58}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        me = max(errs) if errs else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        avg_ms = np.mean(data[name]["solve_ms"]) if data[name]["solve_ms"] else 0
        print(f"  {name:<18} {rmse:>8.4f} {me:>10.4f} {mean_ess:>10.1f} {avg_ms:>10.2f}")
    print(f"{'=' * 72}\n")


# -- Benchmark main --

def run_benchmark(args):
    """Static benchmark execution."""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    dt_val = scenario.get("dt_override") or COMMON["dt"]

    print(f"\n{'=' * 80}")
    print(f"  F-MPPI Benchmark: 4-Way Comparison")
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
            "short": name.replace(" MPPI", "").replace("-MPPI", ""),
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
    print(f"\n{'=' * 100}")
    header = (
        f"{'Method':<18} {'RMSE':>8} {'MaxError':>10} "
        f"{'MeanESS':>10} {'SolveMs':>10} {'ReuseFrac':>10} {'MeanGain':>10}"
    )
    if has_obstacles:
        header += f" {'Collisions':>10} {'MinClear':>10}"
    print(header)
    print(f"{'=' * 100}")
    for r in all_results:
        line = (
            f"{r['name']:<18} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>10.4f} "
            f"{r['mean_ess']:>10.1f} "
            f"{r['mean_solve_ms']:>10.2f} "
            f"{r['reuse_fraction']:>10.2f} "
            f"{r['mean_feedback_gain']:>10.3f}"
        )
        if has_obstacles:
            line += f" {r['n_collisions']:>10d} {r['min_clearance']:>10.3f}"
        print(line)
    print(f"{'=' * 100}")

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-panel result plot (2x4)."""
    dt_val = scenario.get("dt_override") or COMMON["dt"]
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

    # (1,0) RMSE bar chart
    ax = axes[1, 0]
    names = [r["short"] for r in results]
    rmses = [r["rmse"] for r in results]
    colors = [r["color"] for r in results]
    bars = ax.bar(names, rmses, color=colors, alpha=0.8)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1) SolveMs bar chart
    ax = axes[1, 1]
    solve_ms_list = [r["mean_solve_ms"] for r in results]
    bars_sm = ax.bar(names, solve_ms_list, color=colors, alpha=0.8)
    for bar, val in zip(bars_sm, solve_ms_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Avg Solve (ms)")
    ax.set_title("Average Solve Time")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,2) MaxError / MinClearance bar chart
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

    # (1,3) Reuse fraction / MeanGain bar
    ax = axes[1, 3]
    reuse_fracs = [r["reuse_fraction"] for r in results]
    bars_rf = ax.bar(names, reuse_fracs, color=colors, alpha=0.8)
    for bar, val in zip(bars_rf, reuse_fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Reuse Fraction")
    ax.set_title("Feedback Reuse Fraction")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"F-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs Tube vs Robust vs Feedback MPPI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/feedback_mppi_benchmark_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="F-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "high_frequency", "perturbation"],
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
