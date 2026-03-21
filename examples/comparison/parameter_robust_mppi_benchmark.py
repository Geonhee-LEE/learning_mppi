#!/usr/bin/env python3
"""
Parameter-Robust MPPI (PR-MPPI) 벤치마크: 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI      -- 명목 모델 사용 (파라미터 적응 없음)
  2. Tube-MPPI         -- 분리된 피드백 (보수적)
  3. Robust MPPI       -- 통합 피드백 + 외란 샘플링
  4. PR-MPPI           -- 파라미터 입자 필터 + 다중 모델 rollout + 온라인 학습

시나리오 4개:
  A. simple            -- 파라미터 불일치 없음 (기준선)
  B. mild_mismatch     -- wheelbase 0.5->0.6 (10% 불일치)
  C. severe_mismatch   -- wheelbase 0.5->0.8 (60% 불일치)
  D. mismatch_obstacles-- wheelbase 불일치 + 3개 장애물

Usage:
    PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --live --scenario mild_mismatch
    PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --no-plot
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
    ParameterRobustMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.robust_mppi import RobustMPPIController
from mppi_controller.controllers.mppi.parameter_robust_mppi import (
    ParameterRobustMPPIController,
    _ParametricModel,
)
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


# -- Common Settings -------------------------------------------------------

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
    "PR-MPPI": "#E91E63",
}


# -- Scenario Definitions -------------------------------------------------

def get_scenarios():
    """4 benchmark scenarios"""
    return {
        "simple": {
            "name": "A. Simple (No Mismatch)",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.5,
            "process_noise_std": np.array([0.02, 0.02, 0.01]),
            "description": "No parameter mismatch, circle, baseline",
        },
        "mild_mismatch": {
            "name": "B. Mild Mismatch (wb: 0.5->0.6)",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.6,
            "process_noise_std": np.array([0.02, 0.02, 0.01]),
            "description": "Wheelbase mismatch 0.5->0.6 (20%), circle",
        },
        "severe_mismatch": {
            "name": "C. Severe Mismatch (wb: 0.5->0.8)",
            "obstacles": [],
            "trajectory_fn": figure_eight_trajectory,
            "initial_state": np.array([0.0, 0.0, 0.0]),
            "duration": 12.0,
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.8,
            "process_noise_std": np.array([0.02, 0.02, 0.01]),
            "description": "Wheelbase mismatch 0.5->0.8 (60%), figure8",
        },
        "mismatch_obstacles": {
            "name": "D. Mismatch + Obstacles",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.5),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.65,
            "process_noise_std": np.array([0.03, 0.03, 0.01]),
            "description": "Wheelbase mismatch + 3 obstacles, circle",
        },
    }


# -- Controller Factory ----------------------------------------------------

def _make_cost(params, obstacles):
    """Build cost function with optional obstacles"""
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0))
    return CompositeMPPICost(costs)


def _make_controllers(model, scenario):
    """Create 4 controllers for comparison"""
    obstacles = scenario["obstacles"]
    noise_std = scenario["process_noise_std"]

    # 1. Vanilla MPPI
    v_params = MPPIParams(**COMMON)
    vanilla = MPPIController(model, v_params, cost_function=_make_cost(v_params, obstacles))

    # 2. Tube-MPPI
    t_params = TubeMPPIParams(**COMMON, tube_enabled=True)
    tube = TubeMPPIController(model, t_params)
    if obstacles:
        tube.cost_function = _make_cost(t_params, obstacles)

    # 3. Robust MPPI
    rob_params = RobustMPPIParams(
        **COMMON,
        disturbance_std=noise_std.tolist(),
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

    # 4. PR-MPPI
    pr_params = ParameterRobustMPPIParams(
        **COMMON,
        n_particles=8,
        param_name="wheelbase",
        param_nominal=scenario["planner_wheelbase"],
        param_std=0.15,
        param_min=0.2,
        param_max=1.2,
        aggregation_mode="weighted_mean",
        online_learning=True,
        learning_rate=0.01,
        observation_window=10,
        min_observations=3,
        use_resampling=True,
        resample_threshold=0.3,
    )
    pr_mppi = ParameterRobustMPPIController(
        model, pr_params,
        cost_function=_make_cost(pr_params, obstacles),
    )

    return {
        "Vanilla MPPI": vanilla,
        "Tube-MPPI": tube,
        "Robust MPPI": robust,
        "PR-MPPI": pr_mppi,
    }


# -- Simulation ------------------------------------------------------------

def run_single_simulation(model, controller, scenario, seed=42):
    """Single controller simulation"""
    np.random.seed(seed)

    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)
    trajectory_fn = scenario["trajectory_fn"]
    process_noise_std = scenario["process_noise_std"]

    state = scenario["initial_state"].copy()

    # True model (may have different wheelbase)
    real_wb = scenario.get("real_wheelbase", 0.5)
    planner_wb = scenario.get("planner_wheelbase", 0.5)
    if real_wb != planner_wb:
        # Use parametric model to actually scale omega by wheelbase ratio
        real_model = _ParametricModel(model, "wheelbase", real_wb, planner_wb)
    else:
        real_model = model

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

        # Propagate with TRUE model + process noise
        state = real_model.step(state, control, dt)
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
    """Compute metrics from simulation history"""
    states = history["states"]
    trajectory_fn = scenario["trajectory_fn"]
    dt = COMMON["dt"]
    obstacles = scenario["obstacles"]

    # RMSE + MaxError
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt)
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

    # Parameter estimation (PR-MPPI only)
    estimated_params = []
    param_stds = []
    for info in history["infos"]:
        if isinstance(info, dict) and "parameter_robust_stats" in info:
            pr = info["parameter_robust_stats"]
            estimated_params.append(pr["param_mean"])
            param_stds.append(pr["param_std"])

    return {
        "rmse": rmse,
        "max_error": max_error,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "errors": errors,
        "estimated_params": estimated_params,
        "param_stds": param_stds,
        "estimated_param": estimated_params[-1] if estimated_params else None,
        "param_error": abs(estimated_params[-1] - scenario["real_wheelbase"])
            if estimated_params else None,
    }


# -- Live Animation --------------------------------------------------------

def run_live(args):
    """Realtime 4-way comparison animation -> GIF/MP4"""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    trajectory_fn = scenario["trajectory_fn"]
    process_noise_std = scenario["process_noise_std"]

    wb = scenario.get("planner_wheelbase", 0.5)
    model = DifferentialDriveKinematic(wheelbase=wb)
    real_wb = scenario.get("real_wheelbase", wb)
    if real_wb != wb:
        real_model = _ParametricModel(model, "wheelbase", real_wb, wb)
    else:
        real_model = model

    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)

    np.random.seed(args.seed)
    controllers = _make_controllers(model, scenario)

    print(f"\n{'=' * 60}")
    print(f"  PR-MPPI Live -- {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Planner wb={wb}, True wb={real_wb}")
    print(f"  Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # State initialization
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "param_est": []}
        for k in controllers
    }

    # Figure setup (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"PR-MPPI Live -- {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY Trajectories
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    for ox, oy, r in scenario["obstacles"]:
        ax_xy.add_patch(Circle((ox, oy), r, color="red", alpha=0.3))

    ref_t = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t) for t in ref_t])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

    lines_xy = {}
    dots = {}
    for name, color in COLORS.items():
        lines_xy[name], = ax_xy.plot([], [], color=color, linewidth=2, label=name)
        dots[name], = ax_xy.plot([], [], "o", color=color, markersize=8)
    ax_xy.legend(loc="upper left", fontsize=6)

    # [0,1] Tracking Error
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, color in COLORS.items():
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5, label=name)
    ax_err.legend(fontsize=6)

    # [0,2] Parameter Estimation (PR-MPPI only)
    ax_param = axes[0, 2]
    ax_param.set_xlabel("Time (s)")
    ax_param.set_ylabel("Wheelbase Estimate")
    ax_param.set_title("Parameter Estimation (PR-MPPI)")
    ax_param.grid(True, alpha=0.3)
    ax_param.axhline(y=real_wb, color="red", linestyle="--", alpha=0.5, label="True")
    ax_param.axhline(y=wb, color="blue", linestyle=":", alpha=0.5, label="Nominal")
    line_param, = ax_param.plot([], [], color=COLORS["PR-MPPI"], linewidth=2, label="PR-MPPI Est.")
    ax_param.legend(fontsize=6)

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
    ax_rmse.set_xticklabels(["Van", "Tube", "Robust", "PR"], fontsize=8)
    bar_rmse_texts = [
        ax_rmse.text(b.get_x() + b.get_width() / 2, 0, "", ha="center",
                     va="bottom", fontsize=8)
        for b in bars_rmse
    ]

    # [1,1] MaxError bar chart
    ax_me = axes[1, 1]
    ax_me.set_ylabel("Max Error (m)")
    ax_me.set_title("Max Tracking Error")
    ax_me.grid(True, alpha=0.3, axis="y")
    bars_me = ax_me.bar(range(len(bar_names)), [0] * len(bar_names),
                        color=bar_colors, alpha=0.8)
    ax_me.set_xticks(range(len(bar_names)))
    ax_me.set_xticklabels(["Van", "Tube", "Robust", "PR"], fontsize=8)

    # [1,2] MeanESS bar chart
    ax_ess_bar = axes[1, 2]
    ax_ess_bar.set_ylabel("Mean ESS")
    ax_ess_bar.set_title("Mean ESS")
    ax_ess_bar.grid(True, alpha=0.3, axis="y")
    bars_ess = ax_ess_bar.bar(range(len(bar_names)), [0] * len(bar_names),
                              color=bar_colors, alpha=0.8)
    ax_ess_bar.set_xticks(range(len(bar_names)))
    ax_ess_bar.set_xticklabels(["Van", "Tube", "Robust", "PR"], fontsize=8)

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

            state_new = real_model.step(states[name], control, dt)
            states[name] = state_new + np.random.randn(3) * process_noise_std

            ref_pt = trajectory_fn(t)[:2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))

            if "parameter_robust_stats" in info:
                data[name]["param_est"].append(
                    info["parameter_robust_stats"]["param_mean"]
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
                    data[name]["errors"]
                )
                lines_ess[name].set_data(
                    times[:len(data[name]["ess"])],
                    data[name]["ess"]
                )

        # Parameter estimation line
        pe = data["PR-MPPI"]["param_est"]
        if pe:
            line_param.set_data(times[:len(pe)], pe)
            ax_param.relim()
            ax_param.autoscale_view()

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
                (bars_rmse[i].get_x() + bars_rmse[i].get_width() / 2, rmse))
            bar_rmse_texts[i].set_text(f"{rmse:.3f}")
        if rmses:
            ax_rmse.set_ylim(0, max(rmses) * 1.3 + 0.01)

        # MaxError bar chart
        max_errs = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            me = max(errs) if errs else 0
            max_errs.append(me)
            bars_me[i].set_height(me)
        if max_errs:
            ax_me.set_ylim(0, max(max_errs) * 1.3 + 0.01)

        # ESS bar chart
        mean_ess_vals = []
        for i, name in enumerate(bar_names):
            ess_vals = data[name]["ess"]
            me_ess = np.mean(ess_vals) if ess_vals else 0
            mean_ess_vals.append(me_ess)
            bars_ess[i].set_height(me_ess)
        if mean_ess_vals:
            ax_ess_bar.set_ylim(0, max(mean_ess_vals) * 1.3 + 1)

        # Statistics text
        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s"]
        lines.append(f"True wb = {real_wb}, Nominal wb = {wb}\n")
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            short = name.replace(" MPPI", "").replace("-MPPI", "")
            line = f"{short:>10}: RMSE={rmse:.3f} ESS={ess:.0f}"
            if data[name]["param_est"]:
                line += f" wb={data[name]['param_est'][-1]:.3f}"
            lines.append(line)
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/pr_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/pr_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # Final statistics
    print(f"\n{'=' * 72}")
    print(f"  Final Statistics -- {scenario['name']}")
    print(f"{'=' * 72}")
    print(f"  {'Method':<18} {'RMSE':>8} {'MaxError':>10} {'MeanESS':>10} {'ParamEst':>10}")
    print(f"  {'-' * 58}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        me = max(errs) if errs else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        pe = data[name]["param_est"]
        pe_str = f"{pe[-1]:.3f}" if pe else "N/A"
        print(f"  {name:<18} {rmse:>8.4f} {me:>10.4f} {mean_ess:>10.1f} {pe_str:>10}")
    print(f"{'=' * 72}\n")


# -- Static Benchmark ------------------------------------------------------

def run_benchmark(args):
    """Static benchmark execution"""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]

    print(f"\n{'=' * 90}")
    print(f"  PR-MPPI Benchmark: 4-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Planner wb={scenario['planner_wheelbase']}, "
          f"True wb={scenario['real_wheelbase']}")
    print(f"  Duration: {scenario['duration']}s | Seed: {args.seed}")
    print(f"{'=' * 90}")

    wb = scenario.get("planner_wheelbase", 0.5)
    model = DifferentialDriveKinematic(wheelbase=wb)
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
    has_obstacles = len(scenario["obstacles"]) > 0
    print(f"\n{'=' * 100}")
    header = (
        f"{'Method':<18} {'RMSE':>8} {'MaxError':>10} "
        f"{'MeanESS':>10} {'SolveMs':>10} {'EstParam':>10} {'ParamErr':>10}"
    )
    if has_obstacles:
        header += f" {'Collisions':>10} {'MinClear':>10}"
    print(header)
    print(f"{'=' * 100}")
    for r in all_results:
        est = f"{r['estimated_param']:.3f}" if r.get('estimated_param') is not None else "N/A"
        perr = f"{r['param_error']:.4f}" if r.get('param_error') is not None else "N/A"
        line = (
            f"{r['name']:<18} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>10.4f} "
            f"{r['mean_ess']:>10.1f} "
            f"{r['mean_solve_ms']:>10.1f} "
            f"{est:>10} "
            f"{perr:>10}"
        )
        if has_obstacles:
            line += f" {r['n_collisions']:>10d} {r['min_clearance']:>10.3f}"
        print(line)
    print(f"{'=' * 100}")

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-panel result plot (2x4)"""
    dt = COMMON["dt"]
    trajectory_fn = scenario["trajectory_fn"]
    duration = scenario["duration"]
    real_wb = scenario["real_wheelbase"]
    planner_wb = scenario["planner_wheelbase"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # (0,0) XY Trajectories
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

    # (0,1) Tracking Error
    ax = axes[0, 1]
    for r in results:
        t_plot = np.arange(len(r["errors"])) * dt
        ax.plot(t_plot, r["errors"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Parameter Estimation
    ax = axes[0, 2]
    ax.axhline(y=real_wb, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"True wb={real_wb}")
    ax.axhline(y=planner_wb, color="blue", linestyle=":", alpha=0.5, linewidth=1,
               label=f"Nominal wb={planner_wb}")

    for r in results:
        if r["estimated_params"]:
            t_param = np.arange(len(r["estimated_params"])) * dt
            ax.plot(t_param, r["estimated_params"], color=r["color"],
                    label=f"{r['short']} Est.", linewidth=2)
            # Also show +/- 1 std band
            if r["param_stds"]:
                means = np.array(r["estimated_params"])
                stds = np.array(r["param_stds"])
                ax.fill_between(t_param, means - stds, means + stds,
                                color=r["color"], alpha=0.15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wheelbase Estimate")
    ax.set_title("Online Parameter Estimation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,3) ESS
    ax = axes[0, 3]
    for r in results:
        if r["ess_list"]:
            t_ess = np.arange(len(r["ess_list"])) * dt
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

    # (1,1) MaxError bar chart
    ax = axes[1, 1]
    max_errors = [r["max_error"] for r in results]
    bars_me = ax.bar(names, max_errors, color=colors, alpha=0.8)
    for bar, val in zip(bars_me, max_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Max Error (m)")
    ax.set_title("Max Tracking Error")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,2) Mean ESS bar chart
    ax = axes[1, 2]
    mean_ess_vals = [r["mean_ess"] for r in results]
    bars_ess = ax.bar(names, mean_ess_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars_ess, mean_ess_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean ESS")
    ax.set_title("Mean ESS Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) SolveTime bar chart + ParamError annotation
    ax = axes[1, 3]
    solve_ms = [r["mean_solve_ms"] for r in results]
    bars_st = ax.bar(names, solve_ms, color=colors, alpha=0.8)
    for bar, val in zip(bars_st, solve_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Mean Solve Time")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"PR-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs Tube vs Robust vs PR-MPPI  |  "
        f"wb: {planner_wb} (nominal) -> {real_wb} (true)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/pr_mppi_benchmark_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# -- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PR-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="mild_mismatch",
        choices=["simple", "mild_mismatch", "severe_mismatch", "mismatch_obstacles"],
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
