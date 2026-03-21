#!/usr/bin/env python3
"""
C-MPPI (Contingency-Constrained MPPI) 벤치마크: 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI  -- 단일 반복, 등방 노이즈 + ObstacleCost
  2. CBF-MPPI      -- CBF 비용 함수 + 선택적 안전 필터
  3. DBaS-MPPI     -- barrier state + 적응적 탐색
  4. C-MPPI        -- nested MPPI, 체크포인트 contingency 평가

시나리오 4개:
  A. simple          -- 장애물 없음, circle 기준선
  B. obstacles       -- 3개 원형 장애물
  C. narrow_passage  -- 2개 장애물로 좁은 통로 형성 (탈출 계획 테스트)
  D. dynamic_risk    -- 궤적 근처 장애물 (contingency가 미리 계획해야 함)

Usage:
    PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --live --scenario obstacles
    PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --no-plot
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
    CBFMPPIParams,
    DBaSMPPIParams,
    ContingencyMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.dbas_mppi import DBaSMPPIController
from mppi_controller.controllers.mppi.contingency_mppi import ContingencyMPPIController
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
    K=256, N=20, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)

COLORS = {
    "Vanilla MPPI": "#2196F3",
    "CBF-MPPI": "#FF9800",
    "DBaS-MPPI": "#4CAF50",
    "C-MPPI": "#E91E63",
}


# -- Scenarios --

def get_scenarios():
    """4 benchmark scenarios"""
    return {
        "simple": {
            "name": "A. Simple (No Obstacles)",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 10.0,
            "description": "Circle tracking, no obstacles (baseline)",
        },
        "obstacles": {
            "name": "B. Three Obstacles",
            "obstacles": [
                (2.0, 2.0, 0.5),
                (-1.5, 2.5, 0.4),
                (0.0, -3.0, 0.6),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 12.0,
            "description": "3 circular obstacles on circle trajectory",
        },
        "narrow_passage": {
            "name": "C. Narrow Passage",
            "obstacles": [
                (1.5, 1.2, 0.6),
                (1.5, -1.2, 0.6),
                (3.5, 0.0, 0.4),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 12.0,
            "description": "2 close obstacles forming gap + end obstacle (escape planning)",
        },
        "dynamic_risk": {
            "name": "D. Dynamic Risk (Obstacles Near Trajectory)",
            "obstacles": [
                (2.8, 0.5, 0.3),
                (-2.5, 1.5, 0.35),
                (0.5, -2.8, 0.3),
                (-0.5, 2.8, 0.3),
                (2.0, -2.0, 0.3),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 12.0,
            "description": "5 obstacles close to trajectory (contingency must plan ahead)",
        },
    }


# -- Controller Factory --

def _make_cost(params, obstacles):
    """Base cost + obstacle cost"""
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0))
    return CompositeMPPICost(costs)


def _make_controllers(model, scenario):
    """Build 4 controllers"""
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
        cbf_alpha=0.2,
        cbf_safety_margin=0.15,
        cbf_use_safety_filter=False,
    )
    # CBF uses its own composite cost internally
    cbf = CBFMPPIController(model, cbf_params)

    # 3. DBaS-MPPI
    dbas_params = DBaSMPPIParams(
        **COMMON,
        dbas_obstacles=obstacles,
        dbas_walls=[],
        barrier_weight=20.0,
        barrier_gamma=0.5,
        exploration_coeff=1.0,
        h_min=1e-6,
        safety_margin=0.15,
        use_adaptive_exploration=True,
    )
    dbas_cost = _make_cost(dbas_params, None)  # no obstacle cost (barrier replaces it)
    dbas = DBaSMPPIController(model, dbas_params, cost_function=dbas_cost)

    # 4. C-MPPI (reduced inner MPPI for efficiency)
    c_params = ContingencyMPPIParams(
        **COMMON,
        contingency_weight=50.0,
        contingency_horizon=8,
        contingency_samples=16,
        contingency_lambda=1.0,
        n_checkpoints=3,
        safe_cost_threshold=100.0,
        safety_cost_weight=200.0,
        use_braking_contingency=True,
        use_mppi_contingency=True,
        contingency_sigma_scale=1.0,
    )
    main_cost = _make_cost(c_params, obstacles)
    safety_cost = _make_cost(c_params, obstacles)
    c_mppi = ContingencyMPPIController(
        model, c_params,
        cost_function=main_cost,
        safety_cost_function=safety_cost,
    )

    return {
        "Vanilla MPPI": vanilla,
        "CBF-MPPI": cbf,
        "DBaS-MPPI": dbas,
        "C-MPPI": c_mppi,
    }


# -- Simulation --

def run_single_simulation(model, controller, scenario, seed=42):
    """Run simulation for single controller"""
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
    """Compute performance metrics"""
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

    # Obstacle collisions & min clearance
    n_collisions = 0
    min_clearance = float("inf")
    for st in states:
        for ox, oy, r in obstacles:
            dist = np.sqrt((st[0] - ox) ** 2 + (st[1] - oy) ** 2)
            clearance = dist - r
            min_clearance = min(min_clearance, clearance)
            if clearance < 0:
                n_collisions += 1

    if min_clearance == float("inf"):
        min_clearance = 0.0

    # ESS
    ess_list = [
        info.get("ess", 0.0)
        for info in history["infos"]
        if isinstance(info, dict) and "ess" in info
    ]

    # Contingency stats (C-MPPI specific)
    cont_costs = []
    for info in history["infos"]:
        if isinstance(info, dict) and "contingency_stats" in info:
            cont_costs.append(info["contingency_stats"]["mean_contingency_cost"])

    return {
        "rmse": rmse,
        "max_error": max_error,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "cont_costs": cont_costs,
        "mean_cont_cost": float(np.mean(cont_costs)) if cont_costs else 0.0,
        "errors": errors,
    }


# -- Live Animation --

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
    print(f"  C-MPPI Live -- {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # State init
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "cont_cost": []}
        for k in controllers
    }

    # Figure setup (2x4 = 8 panels)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"C-MPPI Live -- {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY trajectories
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    # Static obstacles
    for ox, oy, r in obstacles:
        ax_xy.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))

    # Reference
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

    # [0,2] Contingency cost (C-MPPI only)
    ax_cc = axes[0, 2]
    ax_cc.set_xlabel("Time (s)")
    ax_cc.set_ylabel("Contingency Cost")
    ax_cc.set_title("Contingency Cost (C-MPPI)")
    ax_cc.grid(True, alpha=0.3)
    line_cc, = ax_cc.plot([], [], color=COLORS["C-MPPI"], linewidth=1.5)

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
    ax_rmse.set_xticklabels(["Van", "CBF", "DBaS", "C-MPPI"], fontsize=8)
    bar_rmse_texts = [
        ax_rmse.text(b.get_x() + b.get_width() / 2, 0, "", ha="center",
                     va="bottom", fontsize=8)
        for b in bars_rmse
    ]

    # [1,1] Collision count bar chart
    ax_col = axes[1, 1]
    ax_col.set_ylabel("Collisions")
    ax_col.set_title("Collision Count")
    ax_col.grid(True, alpha=0.3, axis="y")
    bars_col = ax_col.bar(range(len(bar_names)), [0] * len(bar_names),
                          color=bar_colors, alpha=0.8)
    ax_col.set_xticks(range(len(bar_names)))
    ax_col.set_xticklabels(["Van", "CBF", "DBaS", "C-MPPI"], fontsize=8)
    collision_counts = {k: 0 for k in COLORS}

    # [1,2] Min clearance bar chart
    ax_clear = axes[1, 2]
    ax_clear.set_ylabel("Min Clearance (m)")
    ax_clear.set_title("Minimum Clearance")
    ax_clear.grid(True, alpha=0.3, axis="y")
    bars_clear = ax_clear.bar(range(len(bar_names)), [0] * len(bar_names),
                              color=bar_colors, alpha=0.8)
    ax_clear.set_xticks(range(len(bar_names)))
    ax_clear.set_xticklabels(["Van", "CBF", "DBaS", "C-MPPI"], fontsize=8)
    min_clearances = {k: float("inf") for k in COLORS}

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
                if np.sqrt((states[name][0] - ox)**2 + (states[name][1] - oy)**2) < r:
                    collision_counts[name] += 1

            # Min clearance
            for ox, oy, r in obstacles:
                d = np.sqrt((states[name][0] - ox)**2 + (states[name][1] - oy)**2) - r
                min_clearances[name] = min(min_clearances[name], d)

            # C-MPPI contingency cost
            if name == "C-MPPI" and "contingency_stats" in info:
                data[name]["cont_cost"].append(
                    info["contingency_stats"]["mean_contingency_cost"]
                )

        sim_t[0] += dt

        # Update plots
        times = np.array(data["Vanilla MPPI"]["times"])

        for name in controllers:
            xy = np.array(data[name]["xy"])
            if len(xy) > 0:
                lines_xy[name].set_data(xy[:, 0], xy[:, 1])
                dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
                lines_err[name].set_data(
                    times[:len(data[name]["errors"])],
                    data[name]["errors"],
                )
                lines_ess[name].set_data(
                    times[:len(data[name]["ess"])],
                    data[name]["ess"],
                )

        # Contingency cost line
        cc_data = data["C-MPPI"]["cont_cost"]
        if cc_data:
            cc_times = times[:len(cc_data)]
            line_cc.set_data(cc_times, cc_data)
            ax_cc.relim()
            ax_cc.autoscale_view()

        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal")
        ax_err.relim()
        ax_err.autoscale_view()
        ax_ess.relim()
        ax_ess.autoscale_view()

        # RMSE bar chart
        rmses = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            rmses.append(rmse)
            bars_rmse[i].set_height(rmse)
            bar_rmse_texts[i].set_position(
                (bars_rmse[i].get_x() + bars_rmse[i].get_width() / 2, rmse)
            )
            bar_rmse_texts[i].set_text(f"{rmse:.3f}")
        if rmses:
            ax_rmse.set_ylim(0, max(rmses) * 1.3 + 0.01)

        # Collision bar chart
        for i, name in enumerate(bar_names):
            bars_col[i].set_height(collision_counts[name])
        max_col = max(collision_counts.values()) if collision_counts else 1
        ax_col.set_ylim(0, max(max_col * 1.3, 1))

        # Min clearance bar chart
        for i, name in enumerate(bar_names):
            cl = min_clearances[name] if min_clearances[name] != float("inf") else 0
            bars_clear[i].set_height(cl)
        all_cl = [
            min_clearances[n] for n in bar_names
            if min_clearances[n] != float("inf")
        ]
        if all_cl:
            ax_clear.set_ylim(min(min(all_cl) - 0.1, 0), max(all_cl) * 1.3 + 0.01)
        ax_clear.axhline(0, color="red", linestyle="--", alpha=0.5)

        # Statistics text
        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            short = name.replace(" MPPI", "").replace("-MPPI", "")
            cols = collision_counts[name]
            cl = min_clearances[name] if min_clearances[name] != float("inf") else 0
            lines.append(
                f"{short:>8}: RMSE={rmse:.3f} ESS={ess:.0f} "
                f"Col={cols} MinCl={cl:.3f}"
            )
        cc_data = data["C-MPPI"]["cont_cost"]
        if cc_data:
            lines.append(f"\nC-MPPI cont_cost: {cc_data[-1]:.1f}")
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/contingency_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/contingency_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # Final statistics
    print(f"\n{'=' * 72}")
    print(f"  Final Statistics -- {scenario['name']}")
    print(f"{'=' * 72}")
    print(f"  {'Method':<16} {'RMSE':>8} {'Collisions':>10} {'Mean ESS':>10}")
    print(f"  {'-' * 46}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        cols = collision_counts[name]
        print(f"  {name:<16} {rmse:>8.4f} {cols:>10d} {mean_ess:>10.1f}")
    print(f"{'=' * 72}\n")


# -- Static Benchmark --

def run_benchmark(args):
    """Run static benchmark + results table + plots"""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]

    print(f"\n{'=' * 80}")
    print(f"  C-MPPI Benchmark: 4-Way Comparison")
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
    print(
        f"{'Method':<18} {'RMSE':>8} {'MaxErr':>8} {'Collisions':>10} "
        f"{'MinClear':>10} {'MeanESS':>10} {'SolveMs':>10} {'MeanCont':>10}"
    )
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
            f"{r['mean_cont_cost']:>10.1f}"
        )
    print(f"{'=' * 90}")

    # C-MPPI specific stats
    for r in all_results:
        if r["cont_costs"]:
            print(
                f"\n  C-MPPI contingency: mean={np.mean(r['cont_costs']):.1f}, "
                f"max={np.max(r['cont_costs']):.1f}, "
                f"min={np.min(r['cont_costs']):.1f}"
            )

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-panel result plot (2x4)"""
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

    # (0,1) Tracking error
    ax = axes[0, 1]
    for r in results:
        t_plot = np.arange(len(r["errors"])) * dt
        ax.plot(t_plot, r["errors"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Contingency cost (C-MPPI only)
    ax = axes[0, 2]
    for r in results:
        if r["cont_costs"]:
            t_cc = np.arange(len(r["cont_costs"])) * dt
            ax.plot(t_cc, r["cont_costs"], color=r["color"],
                    label=r["short"], linewidth=1.5)
    if not any(r["cont_costs"] for r in results):
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", fontsize=12, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contingency Cost")
    ax.set_title("Contingency Cost (C-MPPI)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,3) ESS
    ax = axes[0, 3]
    for r in results:
        if r["ess_list"]:
            t_ess = np.arange(len(r["ess_list"])) * dt
            ax.plot(t_ess, r["ess_list"], color=r["color"],
                    label=r["short"], linewidth=1)
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

    # (1,1) Collision count bar chart
    ax = axes[1, 1]
    collisions = [r["n_collisions"] for r in results]
    bars_c = ax.bar(names, collisions, color=colors, alpha=0.8)
    for bar, val in zip(bars_c, collisions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Collisions")
    ax.set_title("Collision Count")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,2) Min clearance bar chart
    ax = axes[1, 2]
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

    # (1,3) Solve time bar chart
    ax = axes[1, 3]
    solve_times = [r["mean_solve_ms"] for r in results]
    bars_st = ax.bar(names, solve_times, color=colors, alpha=0.8)
    for bar, val in zip(bars_st, solve_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"C-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs CBF vs DBaS vs C-MPPI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/contingency_mppi_benchmark_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="C-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "narrow_passage", "dynamic_risk"],
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
