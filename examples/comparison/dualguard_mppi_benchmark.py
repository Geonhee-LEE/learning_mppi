#!/usr/bin/env python3
"""
DualGuard-MPPI 벤치마크: 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI  — 단일 반복, 등방 노이즈 + ObstacleCost
  2. CBF-MPPI      — CBF 비용 함수 (barrier function 기반)
  3. DBaS-MPPI     — barrier state + 적응적 탐색
  4. DualGuard-MPPI — HJ 안전 가치 함수 + 이중 가드

시나리오 4개:
  A. simple         — 장애물 없음 (기준선)
  B. obstacles      — 3개 정적 장애물
  C. dense_obstacles — 6개 밀집 장애물 (stress test)
  D. velocity_aware  — 경로 근처 장애물 + 속도 페널티

Usage:
    PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --live --scenario obstacles
    PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --no-plot
"""

import numpy as np
import argparse
import time
import sys
import os
from functools import partial

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
    CBFMPPIParams,
    DBaSMPPIParams,
    DualGuardMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.dbas_mppi import DBaSMPPIController
from mppi_controller.controllers.mppi.dualguard_mppi import DualGuardMPPIController
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


# -- Common config --

COMMON = dict(
    K=512, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)

COLORS = {
    "Vanilla MPPI": "#2196F3",
    "CBF-MPPI": "#FF9800",
    "DBaS-MPPI": "#4CAF50",
    "DualGuard-MPPI": "#E91E63",
}


# -- Scenario definition --

def get_scenarios():
    """4 benchmark scenarios"""
    return {
        "simple": {
            "name": "A. Simple (No Obstacles)",
            "obstacles": [],
            "trajectory_fn": partial(circle_trajectory, radius=3.0, angular_velocity=0.15),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 10.0,
            "description": "Baseline tracking, no obstacles",
        },
        "obstacles": {
            "name": "B. Three Obstacles",
            "obstacles": [
                (2.5, 1.5, 0.4),
                (-1.0, 2.5, 0.4),
                (1.5, -2.0, 0.35),
            ],
            "trajectory_fn": partial(circle_trajectory, radius=3.0, angular_velocity=0.15),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 12.0,
            "description": "3 static obstacles near circle path",
        },
        "dense_obstacles": {
            "name": "C. Dense Obstacles (Stress Test)",
            "obstacles": [
                (2.5, 1.5, 0.3),
                (0.0, 2.8, 0.3),
                (-2.5, 0.5, 0.3),
                (-1.0, -2.5, 0.3),
                (1.5, -2.0, 0.3),
                (2.2, 2.2, 0.25),
            ],
            "trajectory_fn": partial(circle_trajectory, radius=3.0, angular_velocity=0.15),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 12.0,
            "description": "6 obstacles near circle path (stress test)",
        },
        "velocity_aware": {
            "name": "D. Velocity-Aware (Obstacle Near Path)",
            "obstacles": [
                (3.0, 1.0, 0.4),
                (0.0, 3.2, 0.35),
                (-2.5, 1.0, 0.3),
            ],
            "trajectory_fn": partial(circle_trajectory, radius=3.0, angular_velocity=0.2),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 12.0,
            "description": "Obstacles near path + faster speed (velocity penalty test)",
        },
    }


# -- Controller creation --

def _make_cost(params, obstacles):
    """Base cost + obstacle cost"""
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.3, cost_weight=2000.0))
    return CompositeMPPICost(costs)


def _make_controllers(model, scenario):
    """Create 4 controllers"""
    obstacles = scenario["obstacles"]

    # 1. Vanilla MPPI
    v_params = MPPIParams(**COMMON)
    vanilla = MPPIController(
        model, v_params,
        cost_function=_make_cost(v_params, obstacles),
    )

    # 2. CBF-MPPI
    cbf_params = CBFMPPIParams(
        **COMMON,
        cbf_obstacles=obstacles,
        cbf_weight=1000.0,
        cbf_alpha=0.3,
        cbf_safety_margin=0.15,
        cbf_use_safety_filter=False,
    )
    # CBF adds its own cost, so only provide base cost (no ObstacleCost)
    cbf_cost = CompositeMPPICost([
        StateTrackingCost(cbf_params.Q),
        TerminalCost(cbf_params.Qf),
        ControlEffortCost(cbf_params.R),
    ])
    cbf = CBFMPPIController(model, cbf_params, cost_function=cbf_cost)

    # 3. DBaS-MPPI
    dbas_params = DBaSMPPIParams(
        **COMMON,
        dbas_obstacles=obstacles,
        barrier_weight=20.0,
        barrier_gamma=0.5,
        exploration_coeff=1.0,
        h_min=1e-6,
        safety_margin=0.15,
        use_adaptive_exploration=True,
    )
    dbas_cost = CompositeMPPICost([
        StateTrackingCost(dbas_params.Q),
        TerminalCost(dbas_params.Qf),
        ControlEffortCost(dbas_params.R),
    ])
    dbas = DBaSMPPIController(model, dbas_params, cost_function=dbas_cost)

    # 4. DualGuard-MPPI
    dg_params = DualGuardMPPIParams(
        **COMMON,
        obstacles=obstacles,
        safety_margin=0.2,
        safety_mode="soft",
        safety_penalty=5000.0,
        safety_decay=8.0,
        use_velocity_penalty=True,
        velocity_penalty_weight=50.0,
        ttc_horizon=1.0,
        use_nominal_guard=True,
        use_sample_guard=True,
        min_safe_fraction=0.1,
        noise_boost_factor=1.5,
    )
    dg_cost = CompositeMPPICost([
        StateTrackingCost(dg_params.Q),
        TerminalCost(dg_params.Qf),
        ControlEffortCost(dg_params.R),
    ])
    dualguard = DualGuardMPPIController(model, dg_params, cost_function=dg_cost)

    return {
        "Vanilla MPPI": vanilla,
        "CBF-MPPI": cbf,
        "DBaS-MPPI": dbas,
        "DualGuard-MPPI": dualguard,
    }


# -- Simulation --

def run_single_simulation(model, controller, scenario, seed=42):
    """Single controller simulation"""
    np.random.seed(seed)

    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)
    trajectory_fn = scenario["trajectory_fn"]

    state = scenario["initial_state"].copy()

    states = [state.copy()]
    controls_hist = []
    solve_times = []
    infos = []

    for step in range(num_steps):
        t = step * dt
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt)

        t_start = time.time()
        control, info = controller.compute_control(state, ref)
        solve_time = time.time() - t_start

        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt

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
    """Compute metrics"""
    states = history["states"]
    trajectory_fn = scenario["trajectory_fn"]
    dt = COMMON["dt"]
    obstacles = scenario["obstacles"]

    # RMSE
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    # Max error
    max_error = float(np.max(errors))

    # Collisions & clearance
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

    # Safe fraction (DualGuard only)
    safe_fractions = []
    mean_safety_values = []
    for info in history["infos"]:
        if isinstance(info, dict) and "guard_stats" in info:
            safe_fractions.append(info["guard_stats"]["safe_fraction"])
            mean_safety_values.append(info["guard_stats"]["mean_safety_value"])

    return {
        "rmse": rmse,
        "max_error": max_error,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "safe_fractions": safe_fractions,
        "mean_safe_fraction": float(np.mean(safe_fractions)) if safe_fractions else 1.0,
        "mean_safety_values": mean_safety_values,
        "mean_safety_value": float(np.mean(mean_safety_values)) if mean_safety_values else 0.0,
        "errors": errors,
    }


# -- Live animation --

def run_live(args):
    """Realtime 4-way comparison animation -> GIF/MP4"""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    trajectory_fn = scenario["trajectory_fn"]
    obstacles = scenario["obstacles"]

    model = DifferentialDriveKinematic(wheelbase=0.5)

    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)

    np.random.seed(args.seed)
    controllers = _make_controllers(model, scenario)

    print(f"\n{'=' * 60}")
    print(f"  DualGuard-MPPI Live -- {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # State init
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [],
             "safe_fraction": [], "safety_value": []}
        for k in controllers
    }

    # Figure: 2x4 panels
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"DualGuard-MPPI Live -- {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY trajectories
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    for ox, oy, r in obstacles:
        ax_xy.add_patch(Circle((ox, oy), r, facecolor="#FF5252", edgecolor="red",
                               alpha=0.3, linewidth=1.5))

    ref_t = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t) for t in ref_t])
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

    # [0,2] Safe fraction (DualGuard)
    ax_sf = axes[0, 2]
    ax_sf.set_xlabel("Time (s)")
    ax_sf.set_ylabel("Safe Fraction")
    ax_sf.set_title("Safe Sample Fraction (DualGuard)")
    ax_sf.grid(True, alpha=0.3)
    line_sf, = ax_sf.plot([], [], color=COLORS["DualGuard-MPPI"], linewidth=1.5)
    ax_sf.set_ylim(-0.05, 1.05)

    # [0,3] Safety value (DualGuard)
    ax_sv = axes[0, 3]
    ax_sv.set_xlabel("Time (s)")
    ax_sv.set_ylabel("Mean Safety Value")
    ax_sv.set_title("Mean Safety Value (DualGuard)")
    ax_sv.grid(True, alpha=0.3)
    line_sv, = ax_sv.plot([], [], color=COLORS["DualGuard-MPPI"], linewidth=1.5)
    ax_sv.axhline(0, color="red", linestyle="--", alpha=0.5)

    # [1,0] ESS
    ax_ess = axes[1, 0]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5, label=name)
    ax_ess.legend(fontsize=6)

    # [1,1] RMSE bar
    ax_rmse = axes[1, 1]
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Running RMSE")
    ax_rmse.grid(True, alpha=0.3, axis="y")
    bar_names = list(COLORS.keys())
    bar_colors = [COLORS[n] for n in bar_names]
    bars_rmse = ax_rmse.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_rmse.set_xticks(range(len(bar_names)))
    ax_rmse.set_xticklabels(["Van", "CBF", "DBaS", "DG"], fontsize=8)
    bar_rmse_texts = [
        ax_rmse.text(b.get_x() + b.get_width() / 2, 0, "", ha="center",
                     va="bottom", fontsize=8)
        for b in bars_rmse
    ]

    # [1,2] Collision bar
    ax_col = axes[1, 2]
    ax_col.set_ylabel("Collisions")
    ax_col.set_title("Collision Count")
    ax_col.grid(True, alpha=0.3, axis="y")
    bars_col = ax_col.bar(range(len(bar_names)), [0] * len(bar_names),
                          color=bar_colors, alpha=0.8)
    ax_col.set_xticks(range(len(bar_names)))
    ax_col.set_xticklabels(["Van", "CBF", "DBaS", "DG"], fontsize=8)
    collision_counts = {k: 0 for k in COLORS}

    # [1,3] Stats text
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
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt)

        for name, ctrl in controllers.items():
            control, info = ctrl.compute_control(states[name], ref)

            state_dot = model.forward_dynamics(states[name], control)
            states[name] = states[name] + state_dot * dt

            ref_pt = trajectory_fn(t)[:2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))

            # Collision check
            for ox, oy, r in obstacles:
                if np.sqrt((states[name][0] - ox) ** 2 + (states[name][1] - oy) ** 2) < r:
                    collision_counts[name] += 1

            # DualGuard stats
            if name == "DualGuard-MPPI" and "guard_stats" in info:
                data[name]["safe_fraction"].append(info["guard_stats"]["safe_fraction"])
                data[name]["safety_value"].append(info["guard_stats"]["mean_safety_value"])

        sim_t[0] += dt

        # Update plots
        times = np.array(data["Vanilla MPPI"]["times"])
        for name in controllers:
            xy = np.array(data[name]["xy"])
            if len(xy) > 0:
                lines_xy[name].set_data(xy[:, 0], xy[:, 1])
                dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
                lines_err[name].set_data(times[:len(data[name]["errors"])],
                                         data[name]["errors"])
                lines_ess[name].set_data(times[:len(data[name]["ess"])],
                                         data[name]["ess"])

        # DualGuard-specific
        dg_data = data["DualGuard-MPPI"]
        if dg_data["safe_fraction"]:
            sf_times = times[:len(dg_data["safe_fraction"])]
            line_sf.set_data(sf_times, dg_data["safe_fraction"])
        if dg_data["safety_value"]:
            sv_times = times[:len(dg_data["safety_value"])]
            line_sv.set_data(sv_times, dg_data["safety_value"])
            ax_sv.relim()
            ax_sv.autoscale_view()

        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal")
        ax_err.relim()
        ax_err.autoscale_view()
        ax_ess.relim()
        ax_ess.autoscale_view()

        # RMSE bar
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

        # Collision bar
        for i, name in enumerate(bar_names):
            bars_col[i].set_height(collision_counts[name])
        max_col = max(collision_counts.values()) if collision_counts else 1
        ax_col.set_ylim(0, max(max_col * 1.3, 1))

        # Stats text
        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            short = name.replace(" MPPI", "").replace("-MPPI", "")
            cols = collision_counts[name]
            lines.append(f"{short:>10}: RMSE={rmse:.3f} ESS={ess:.0f} Col={cols}")
        if dg_data["safe_fraction"]:
            lines.append(f"\nDG safe: {dg_data['safe_fraction'][-1]:.2f}")
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/dualguard_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/dualguard_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # Final stats
    print(f"\n{'=' * 72}")
    print(f"  Final Statistics -- {scenario['name']}")
    print(f"{'=' * 72}")
    print(f"  {'Method':<18} {'RMSE':>8} {'Collisions':>10} {'Mean ESS':>10}")
    print(f"  {'-' * 48}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        cols = collision_counts[name]
        print(f"  {name:<18} {rmse:>8.4f} {cols:>10d} {mean_ess:>10.1f}")
    print(f"{'=' * 72}\n")


# -- Benchmark main --

def run_benchmark(args):
    """Static benchmark + results + plot"""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]

    print(f"\n{'=' * 80}")
    print(f"  DualGuard-MPPI Benchmark: 4-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {scenario['duration']}s | Seed: {args.seed}")
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
            "infos": history["infos"],
            "elapsed": elapsed,
            **metrics,
        })

        print(f"done ({elapsed:.1f}s)")

    # Results table
    print(f"\n{'=' * 90}")
    print(f"{'Method':<18} {'RMSE':>8} {'MaxErr':>8} {'Collisions':>10} "
          f"{'MinClear':>10} {'MeanESS':>10} {'SolveMs':>10} {'SafeFrac':>10}")
    print(f"{'=' * 90}")
    for r in all_results:
        print(
            f"{r['name']:<18} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>8.4f} "
            f"{r['n_collisions']:>10d} "
            f"{r['min_clearance']:>10.3f} "
            f"{r['mean_ess']:>10.1f} "
            f"{r['mean_solve_ms']:>10.1f} "
            f"{r['mean_safe_fraction']:>10.2f}"
        )
    print(f"{'=' * 90}")

    # DualGuard specific stats
    for r in all_results:
        if r["name"] == "DualGuard-MPPI" and r["safe_fractions"]:
            print(f"\n  DualGuard safe_fraction: mean={r['mean_safe_fraction']:.3f}, "
                  f"min={np.min(r['safe_fractions']):.3f}")
            print(f"  DualGuard safety_value: mean={r['mean_safety_value']:.3f}")

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-panel results plot (2x4)"""
    dt = COMMON["dt"]
    trajectory_fn = scenario["trajectory_fn"]
    duration = scenario["duration"]
    obstacles = scenario["obstacles"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # (0,0) XY trajectories
    ax = axes[0, 0]
    t_arr = np.linspace(0, duration, 500)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, label="Ref", linewidth=1)

    for r in results:
        ax.plot(r["states"][:, 0], r["states"][:, 1], color=r["color"],
                label=r["short"], linewidth=1.5, alpha=0.8)

    for ox, oy, rad in obstacles:
        ax.add_patch(Circle((ox, oy), rad, facecolor="#FF5252", edgecolor="red",
                            alpha=0.3, linewidth=1.5))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # (0,1) Position error
    ax = axes[0, 1]
    for r in results:
        t_plot = np.arange(len(r["errors"])) * dt
        ax.plot(t_plot, r["errors"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Safe fraction (DualGuard)
    ax = axes[0, 2]
    has_sf = False
    for r in results:
        if r["safe_fractions"]:
            t_sf = np.arange(len(r["safe_fractions"])) * dt
            ax.plot(t_sf, r["safe_fractions"], color=r["color"],
                    label=r["short"], linewidth=1.5)
            has_sf = True
    if not has_sf:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center",
                fontsize=12, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Safe Fraction")
    ax.set_title("Safe Sample Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,3) Safety value (DualGuard)
    ax = axes[0, 3]
    has_sv = False
    for r in results:
        if r["mean_safety_values"]:
            t_sv = np.arange(len(r["mean_safety_values"])) * dt
            ax.plot(t_sv, r["mean_safety_values"], color=r["color"],
                    label=r["short"], linewidth=1.5)
            has_sv = True
    if not has_sv:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center",
                fontsize=12, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Safety Value")
    ax.set_title("Safety Value (V>0: safe)")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,0) ESS
    ax = axes[1, 0]
    for r in results:
        if r["ess_list"]:
            t_ess = np.arange(len(r["ess_list"])) * dt
            ax.plot(t_ess, r["ess_list"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) RMSE bar
    ax = axes[1, 1]
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

    # (1,2) Collision count bar
    ax = axes[1, 2]
    collisions = [r["n_collisions"] for r in results]
    bars_c = ax.bar(names, collisions, color=colors, alpha=0.8)
    for bar, val in zip(bars_c, collisions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Collisions")
    ax.set_title("Collision Count")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) Min clearance bar
    ax = axes[1, 3]
    clearances = [r["min_clearance"] for r in results]
    bars_cl = ax.bar(names, clearances, color=colors, alpha=0.8)
    for bar, val in zip(bars_cl, clearances):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("Minimum Clearance")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)

    plt.suptitle(
        f"DualGuard-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs CBF vs DBaS vs DualGuard",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/dualguard_mppi_benchmark_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="DualGuard-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "dense_obstacles", "velocity_aware"],
    )
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override scenario duration")
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
