#!/usr/bin/env python3
"""
Biased-MPPI (Mixture Sampling MPPI) 벤치마크: 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI    — 단일 가우시안 (기준선)
  2. DIAL-MPPI       — 다중 반복 + 어닐링
  3. CMA-MPPI        — 공분산 적응
  4. Biased-MPPI     — 혼합 분포 (정책 + 가우시안)

시나리오 4개:
  A. simple          — 기준선 (장애물 없음, circle)
  B. obstacles       — 3개 원형 장애물 (정책이 안전 회피 제안)
  C. local_minima    — 큰 장애물 (r=0.8, 정책 다양성으로 탈출)
  D. dense_obstacles — 8개 밀집 장애물 (다양한 탐색 필요)

Usage:
    PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py --live --scenario local_minima
    PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py --no-plot
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
    DIALMPPIParams,
    CMAMPPIParams,
    BiasedMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.cma_mppi import CMAMPPIController
from mppi_controller.controllers.mppi.biased_mppi import BiasedMPPIController
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


# ── 공통 설정 ──────────────────────────────────────────────────

COMMON = dict(
    K=512, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)

COLORS = {
    "Vanilla MPPI": "#2196F3",
    "DIAL-MPPI": "#FF9800",
    "CMA-MPPI": "#4CAF50",
    "Biased-MPPI": "#E91E63",
}


# ── 시나리오 정의 ──────────────────────────────────────────────

def get_scenarios():
    """4개 벤치마크 시나리오"""
    return {
        "simple": {
            "name": "A. Simple (Baseline)",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "description": "No obstacles, circle — baseline comparison",
        },
        "obstacles": {
            "name": "B. Obstacles",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.5),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "description": "3 obstacles — policy-guided safe avoidance",
        },
        "local_minima": {
            "name": "C. Local Minima",
            "obstacles": [
                (0.0, 3.0, 0.8),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "description": "Large obstacle (r=0.8) on path — diversity for escape",
        },
        "dense_obstacles": {
            "name": "D. Dense Obstacles",
            "obstacles": [
                (2.5, 1.5, 0.4),
                (1.5, 2.5, 0.3),
                (0.0, 3.0, 0.4),
                (-1.5, 2.5, 0.3),
                (-2.5, 1.0, 0.4),
                (-2.0, -1.0, 0.3),
                (0.0, -3.0, 0.4),
                (2.0, -1.5, 0.3),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "description": "8 dense obstacles — diverse exploration needed",
        },
    }


# ── 비용 함수 ─────────────────────────────────────────────────

def _make_cost(params, obstacles):
    """기본 비용 + 장애물 비용"""
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0))
    return CompositeMPPICost(costs)


# ── 컨트롤러 생성 ─────────────────────────────────────────────

def _make_controllers(model, scenario):
    """4가지 컨트롤러 생성"""
    obstacles = scenario["obstacles"]
    common = {**COMMON}

    # 1. Vanilla
    v_params = MPPIParams(**common)
    vanilla = MPPIController(
        model, v_params, cost_function=_make_cost(v_params, obstacles)
    )

    # 2. DIAL-MPPI
    d_params = DIALMPPIParams(
        **common, n_diffuse_init=8, n_diffuse=3,
        use_reward_normalization=True,
    )
    dial = DIALMPPIController(
        model, d_params, cost_function=_make_cost(d_params, obstacles)
    )

    # 3. CMA-MPPI
    c_params = CMAMPPIParams(
        **common, n_iters_init=8, n_iters=3,
        use_mean_shift=True, use_reward_normalization=True,
    )
    cma = CMAMPPIController(
        model, c_params, cost_function=_make_cost(c_params, obstacles)
    )

    # 4. Biased-MPPI
    b_params = BiasedMPPIParams(
        **common,
        ancillary_types=["pure_pursuit", "braking", "max_speed"],
        samples_per_policy=10,
        policy_noise_scale=0.3,
        use_adaptive_lambda=True,
    )
    biased = BiasedMPPIController(
        model, b_params, cost_function=_make_cost(b_params, obstacles)
    )

    return {
        "Vanilla MPPI": vanilla,
        "DIAL-MPPI": dial,
        "CMA-MPPI": cma,
        "Biased-MPPI": biased,
    }


# ── 시뮬레이션 ─────────────────────────────────────────────────

def run_single_simulation(model, controller, scenario, seed=42):
    """단일 컨트롤러 시뮬레이션"""
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
    """메트릭 계산"""
    states = history["states"]
    controls = history["controls"]
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

    # 장애물 충돌
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

    # PolicyBestRatio (Biased-MPPI 전용)
    policy_best_ratio = 0.0
    for info in history["infos"]:
        if isinstance(info, dict) and "biased_stats" in info:
            policy_best_ratio = info["biased_stats"].get("policy_best_ratio", 0.0)

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
        "policy_best_ratio": policy_best_ratio,
    }


# ── Live 애니메이션 ─────────────────────────────────────────────

def run_live(args):
    """실시간 4-way 비교 → GIF/MP4 저장"""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    trajectory_fn = scenario["trajectory_fn"]

    model = DifferentialDriveKinematic(wheelbase=0.5)
    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)

    np.random.seed(args.seed)
    controllers = _make_controllers(model, scenario)

    print(f"\n{'=' * 60}")
    print(f"  Biased-MPPI Live — {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # 상태 초기화
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "controls": []}
        for k in controllers
    }

    # Figure (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"Biased-MPPI Live — {scenario['name']}",
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

    ref_t = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t) for t in ref_t])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

    lines_xy = {}
    dots = {}
    for name, color in COLORS.items():
        lines_xy[name], = ax_xy.plot([], [], color=color, linewidth=2, label=name)
        dots[name], = ax_xy.plot([], [], "o", color=color, markersize=8)
    ax_xy.legend(loc="upper left", fontsize=6)

    # [0,1] Error
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, color in COLORS.items():
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5)
    ax_err.legend([n.replace(" MPPI", "") for n in COLORS], fontsize=6)

    # [0,2] ESS
    ax_ess = axes[0, 2]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5)

    # [0,3] Lambda (Biased only)
    ax_lam = axes[0, 3]
    ax_lam.set_xlabel("Time (s)")
    ax_lam.set_ylabel("Lambda")
    ax_lam.set_title("Adaptive Lambda (Biased)")
    ax_lam.grid(True, alpha=0.3)
    lambda_data = []
    line_lam, = ax_lam.plot([], [], color=COLORS["Biased-MPPI"], linewidth=2)

    # [1,0] Controls
    ax_ctrl = axes[1, 0]
    ax_ctrl.set_xlabel("Time (s)")
    ax_ctrl.set_ylabel("Control (v)")
    ax_ctrl.set_title("Control Input (v)")
    ax_ctrl.grid(True, alpha=0.3)
    lines_ctrl = {}
    for name, color in COLORS.items():
        lines_ctrl[name], = ax_ctrl.plot([], [], color=color, linewidth=1, alpha=0.7)

    # [1,1] RMSE bars
    ax_rmse = axes[1, 1]
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Running RMSE")
    ax_rmse.grid(True, alpha=0.3, axis="y")
    bar_names = list(COLORS.keys())
    bar_colors = [COLORS[n] for n in bar_names]
    bars_rmse = ax_rmse.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_rmse.set_xticks(range(len(bar_names)))
    ax_rmse.set_xticklabels(["Van", "DIAL", "CMA", "Biased"], fontsize=8)

    # [1,2] Collision/Clearance bars
    ax_safe = axes[1, 2]
    ax_safe.set_ylabel("Min Clearance (m)")
    ax_safe.set_title("Safety (MinClearance)")
    ax_safe.grid(True, alpha=0.3, axis="y")
    bars_safe = ax_safe.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_safe.set_xticks(range(len(bar_names)))
    ax_safe.set_xticklabels(["Van", "DIAL", "CMA", "Biased"], fontsize=8)
    clearance_data = {k: float("inf") for k in COLORS}

    # [1,3] Stats
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
            data[name]["controls"].append(control.copy())

            # Clearance
            for ox, oy, r in scenario["obstacles"]:
                dist = np.sqrt(
                    (states[name][0] - ox) ** 2 + (states[name][1] - oy) ** 2
                )
                clearance_data[name] = min(clearance_data[name], dist - r)

            # Lambda (Biased only)
            if name == "Biased-MPPI" and "biased_stats" in info:
                lambda_data.append(info["biased_stats"]["current_lambda"])

        sim_t[0] += dt

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
                ctrl_arr = np.array(data[name]["controls"])
                if len(ctrl_arr) > 0:
                    lines_ctrl[name].set_data(times[:len(ctrl_arr)], ctrl_arr[:, 0])

        if lambda_data:
            line_lam.set_data(times[:len(lambda_data)], lambda_data)
            ax_lam.relim()
            ax_lam.autoscale_view()

        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal")
        ax_err.relim()
        ax_err.autoscale_view()
        ax_ess.relim()
        ax_ess.autoscale_view()
        ax_ctrl.relim()
        ax_ctrl.autoscale_view()

        # RMSE bars
        rmses = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            rmses.append(rmse)
            bars_rmse[i].set_height(rmse)
        if rmses:
            ax_rmse.set_ylim(0, max(rmses) * 1.3 + 0.01)

        # Clearance bars
        for i, name in enumerate(bar_names):
            c = clearance_data[name]
            bars_safe[i].set_height(max(c, 0) if c != float("inf") else 0)
        vals = [max(clearance_data[n], 0) for n in bar_names
                if clearance_data[n] != float("inf")]
        if vals:
            ax_safe.set_ylim(0, max(vals) * 1.3 + 0.01)

        # Stats
        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            short = name.replace(" MPPI", "").replace("-MPPI", "")
            lines.append(f"{short:>8}: RMSE={rmse:.3f} ESS={ess:.0f}")
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/biased_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/biased_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # Final stats
    print(f"\n{'=' * 80}")
    print(f"  Final Statistics — {scenario['name']}")
    print(f"{'=' * 80}")
    has_obstacles = len(scenario["obstacles"]) > 0
    header = f"  {'Method':<18} {'RMSE':>8} {'MeanESS':>10}"
    if has_obstacles:
        header += f" {'MinClear':>10}"
    print(header)
    print(f"  {'-' * 50}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        ess_vals = data[name]["ess"]
        mean_ess = float(np.mean(ess_vals)) if ess_vals else 0
        line = f"  {name:<18} {rmse:>8.4f} {mean_ess:>10.1f}"
        if has_obstacles:
            c = clearance_data[name]
            line += f" {c:>10.3f}" if c != float("inf") else f" {'N/A':>10}"
        print(line)
    print(f"{'=' * 80}\n")


# ── 벤치마크 메인 ─────────────────────────────────────────────

def run_benchmark(args):
    """정적 벤치마크 실행"""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]

    print(f"\n{'=' * 100}")
    print(f"  Biased-MPPI Benchmark: 4-Way Mixture Sampling Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {scenario['duration']}s | Seed: {args.seed}")
    print(f"{'=' * 100}")

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
            **metrics,
        })

        print(f"done ({elapsed:.1f}s)")

    # 결과 테이블
    has_obstacles = len(scenario["obstacles"]) > 0
    print(f"\n{'=' * 110}")
    header = (
        f"{'Method':<18} {'RMSE':>8} {'MaxError':>10} {'MeanESS':>10} "
        f"{'SolveMs':>10} {'PolicyBest':>12}"
    )
    if has_obstacles:
        header += f" {'Collisions':>10} {'MinClear':>10}"
    print(header)
    print(f"{'=' * 110}")
    for r in all_results:
        line = (
            f"{r['name']:<18} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>10.4f} "
            f"{r['mean_ess']:>10.1f} "
            f"{r['mean_solve_ms']:>10.1f} "
            f"{r['policy_best_ratio']:>12.1%}"
        )
        if has_obstacles:
            line += f" {r['n_collisions']:>10d} {r['min_clearance']:>10.3f}"
        print(line)
    print(f"{'=' * 110}")

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-panel 결과 플롯 (2x4)"""
    dt = COMMON["dt"]
    trajectory_fn = scenario["trajectory_fn"]
    duration = scenario["duration"]
    has_obstacles = len(scenario["obstacles"]) > 0

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # (0,0) XY
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

    # (0,1) Error
    ax = axes[0, 1]
    for r in results:
        t_plot = np.arange(len(r["errors"])) * dt
        ax.plot(t_plot, r["errors"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) ESS
    ax = axes[0, 2]
    for r in results:
        ess = r["ess_list"]
        if ess:
            t_plot = np.arange(len(ess)) * dt
            ax.plot(t_plot, ess, color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,3) Lambda (Biased only)
    ax = axes[0, 3]
    for r in results:
        lambdas = []
        for info in r["infos"]:
            if isinstance(info, dict) and "biased_stats" in info:
                lambdas.append(info["biased_stats"]["current_lambda"])
            else:
                lambdas.append(info.get("temperature", 1.0) if isinstance(info, dict) else 1.0)
        if lambdas:
            t_plot = np.arange(len(lambdas)) * dt
            ax.plot(t_plot, lambdas, color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lambda")
    ax.set_title("Temperature Adaptation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,0) Controls
    ax = axes[1, 0]
    for r in results:
        controls = r["controls"]
        if len(controls) > 0:
            t_plot = np.arange(len(controls)) * dt
            ax.plot(t_plot, controls[:, 0], color=r["color"], label=r["short"],
                    linewidth=1, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control (v)")
    ax.set_title("Control Input (v)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) RMSE bars
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

    # (1,2) Collision/Clearance bars
    ax = axes[1, 2]
    if has_obstacles:
        clearances = [r["min_clearance"] for r in results]
        bars_c = ax.bar(names, clearances, color=colors, alpha=0.8)
        for bar, val in zip(bars_c, clearances):
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Min Clearance (m)")
        ax.set_title("Safety (MinClearance)")
    else:
        ess_vals = [r["mean_ess"] for r in results]
        bars_e = ax.bar(names, ess_vals, color=colors, alpha=0.8)
        for bar, val in zip(bars_e, ess_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mean ESS")
        ax.set_title("Sampling Efficiency")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) Stats
    ax = axes[1, 3]
    ax.axis("off")
    ax.set_title("Summary Statistics")
    lines = []
    for r in results:
        line_txt = (
            f"{r['short']:>8}: RMSE={r['rmse']:.4f}  ESS={r['mean_ess']:.0f}  "
            f"Solve={r['mean_solve_ms']:.1f}ms"
        )
        if has_obstacles:
            line_txt += f"  Col={r['n_collisions']}  Clear={r['min_clearance']:.3f}"
        if r["policy_best_ratio"] > 0:
            line_txt += f"  PolBest={r['policy_best_ratio']:.1%}"
        lines.append(line_txt)
    ax.text(
        0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va="top", fontsize=8, family="monospace",
    )

    plt.suptitle(
        f"Biased-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs DIAL vs CMA vs Biased-MPPI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/biased_mppi_benchmark_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Biased-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "local_minima", "dense_obstacles"],
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
