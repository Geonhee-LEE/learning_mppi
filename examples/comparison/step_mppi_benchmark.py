#!/usr/bin/env python3
"""
Step-MPPI (Single-Step MPPI via Differentiable Predictive Control) 벤치마크
— 43번째 변형 (arXiv:2604.01539), 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI (K large)  — 표준 MPPI, 큰 K (baseline)
  2. Vanilla MPPI (small K)   — 작은 K (단일-스텝 유사 저예산)
  3. Step-MPPI (no train)     — 학습 proposal, 온라인 학습 끔 (zero-init ≈ vanilla)
  4. Step-MPPI (online train)  — 학습 proposal + 온라인 자기지도 학습

시나리오 4개:
  A. simple          — 원형 궤적, 장애물 없음 (기준선)
  B. obstacles       — 3개 장애물, 원형 궤적
  C. online_learning — 더 긴 주행으로 온라인 학습이 작동하도록 (3개 장애물)
  D. figure8         — figure-8 궤적, 장애물 없음 (민첩성)

Usage:
    PYTHONPATH=. python examples/comparison/step_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/step_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/step_mppi_benchmark.py --live --scenario obstacles
    PYTHONPATH=. python examples/comparison/step_mppi_benchmark.py --no-plot
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
    StepMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.step_mppi import StepMPPIController
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


# -- 공통 설정 --

COMMON = dict(
    K=512, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)

SMALL_K = 64  # 저예산(단일-스텝 유사) Vanilla

COLORS = {
    "Vanilla (K=512)": "#2196F3",
    "Vanilla (K=64)": "#FF9800",
    "Step-MPPI (no train)": "#4CAF50",
    "Step-MPPI (online)": "#E91E63",
}

SHORT = {
    "Vanilla (K=512)": "Van512",
    "Vanilla (K=64)": "Van64",
    "Step-MPPI (no train)": "StepNT",
    "Step-MPPI (online)": "StepON",
}


# -- 시나리오 정의 --

def get_scenarios():
    """4개 벤치마크 시나리오."""
    return {
        "simple": {
            "name": "A. Simple Tracking (Baseline)",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "description": "No obstacles. Tracking + timing baseline.",
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
        "online_learning": {
            "name": "C. Online Learning (Long Run)",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.5),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 30.0,
            "description": "Long run lets Step-MPPI online training kick in.",
        },
        "figure8": {
            "name": "D. Figure-8 (Agility)",
            "obstacles": [],
            "trajectory_fn": lambda t: figure_eight_trajectory(t, scale=3.0),
            "initial_state": np.array([0.0, 0.0, 0.0]),
            "duration": 20.0,
            "description": "Figure-8 trajectory. Agility comparison.",
        },
    }


# -- 컨트롤러 생성 --

def _make_cost(params, obstacles):
    """기본 비용 + 장애물 비용."""
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0))
    return CompositeMPPICost(costs)


def _make_controllers(model, scenario):
    """4개 컨트롤러 생성."""
    obstacles = scenario["obstacles"]

    # 1. Vanilla MPPI (K=512)
    v_params = MPPIParams(**COMMON)
    vanilla_big = MPPIController(
        model, v_params, cost_function=_make_cost(v_params, obstacles),
    )

    # 2. Vanilla MPPI (K=64, 저예산)
    common_small = dict(COMMON)
    common_small["K"] = SMALL_K
    vs_params = MPPIParams(**common_small)
    vanilla_small = MPPIController(
        model, vs_params, cost_function=_make_cost(vs_params, obstacles),
    )

    # 3. Step-MPPI (학습 proposal, 온라인 학습 끔) — 저예산 K
    sn_params = StepMPPIParams(
        **common_small,
        use_learned_proposal=True,
        online_training=False,
        blend_ratio=0.7,
        learn_covariance=True,
        lookahead_steps=1,
    )
    step_notrain = StepMPPIController(
        model, sn_params, cost_function=_make_cost(sn_params, obstacles),
    )

    # 4. Step-MPPI (학습 proposal + 온라인 학습) — 저예산 K
    # 보수적 온라인 학습 설정: 단일-스텝 잔차 타깃은 노이지하므로 낮은
    # blend/lr + 큰 min_train_samples로 graceful하게 학습 (loss 감소 확인).
    so_params = StepMPPIParams(
        **common_small,
        use_learned_proposal=True,
        online_training=True,
        blend_ratio=0.3,
        learn_covariance=True,
        lookahead_steps=1,
        min_train_samples=32,
        train_interval=8,
        train_batch_size=32,
        buffer_size=2000,
        proposal_lr=3e-4,
        entropy_weight=0.01,
    )
    step_online = StepMPPIController(
        model, so_params, cost_function=_make_cost(so_params, obstacles),
    )

    return {
        "Vanilla (K=512)": vanilla_big,
        "Vanilla (K=64)": vanilla_small,
        "Step-MPPI (no train)": step_notrain,
        "Step-MPPI (online)": step_online,
    }


# -- 시뮬레이션 --

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
    """평가 지표 계산."""
    states = history["states"]
    trajectory_fn = scenario["trajectory_fn"]
    dt_val = COMMON["dt"]
    obstacles = scenario["obstacles"]

    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt_val)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    max_error = float(np.max(errors))

    n_collisions = 0
    min_clearance = float("inf")
    for st in states:
        for ox, oy, r in obstacles:
            dist = np.sqrt((st[0] - ox) ** 2 + (st[1] - oy) ** 2)
            clearance = dist - r
            min_clearance = min(min_clearance, clearance)
            if clearance < 0:
                n_collisions += 1

    ess_list = [
        info.get("ess", 0.0) for info in history["infos"]
        if isinstance(info, dict) and "ess" in info
    ]

    # Step-MPPI 통계
    train_count = 0
    buffer_size = 0
    last_loss = None
    final_delta_norm = 0.0
    use_net = False
    for info in history["infos"]:
        if isinstance(info, dict) and "step_stats" in info:
            ss = info["step_stats"]
            train_count = max(train_count, ss.get("train_count", 0))
            buffer_size = max(buffer_size, ss.get("buffer_size", 0))
            final_delta_norm = ss.get("proposal_delta_norm", 0.0)
            use_net = ss.get("use_net", False)

    return {
        "rmse": rmse,
        "max_error": max_error,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "train_count": train_count,
        "buffer_size": buffer_size,
        "final_delta_norm": final_delta_norm,
        "use_net": use_net,
        "errors": errors,
    }


# -- 정적 벤치마크 --

def run_benchmark(args):
    """정적 벤치마크 실행."""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    dt_val = COMMON["dt"]

    print(f"\n{'=' * 80}")
    print(f"  Step-MPPI Benchmark: 4-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {scenario['duration']}s | dt: {dt_val}s | Seed: {args.seed}")
    print(f"{'=' * 80}")

    model = DifferentialDriveKinematic(wheelbase=0.5)
    controllers = _make_controllers(model, scenario)

    all_results = []
    for i, (name, ctrl) in enumerate(controllers.items()):
        np.random.seed(args.seed)
        print(f"\n  [{i+1}/{len(controllers)}] {name:<24}", end=" ", flush=True)
        t_start = time.time()
        history = run_single_simulation(model, ctrl, scenario, seed=args.seed)
        elapsed = time.time() - t_start
        metrics = compute_metrics(history, scenario)

        # loss 추이 조회 (Step-MPPI online): 초기/최종 평균으로 감소 여부 표시
        last_loss = None
        init_loss = None
        if hasattr(ctrl, "trainer") and ctrl.trainer is not None:
            lh = ctrl.trainer._loss_history
            if lh:
                k = max(1, min(10, len(lh) // 2))
                init_loss = float(np.mean(lh[:k]))
                last_loss = float(np.mean(lh[-k:]))

        all_results.append({
            "name": name,
            "short": SHORT[name],
            "color": COLORS[name],
            "states": history["states"],
            "controls": history["controls"],
            "infos": history["infos"],
            "elapsed": elapsed,
            "solve_times_ms": history["solve_times"] * 1000,
            "last_loss": last_loss,
            "init_loss": init_loss,
            **metrics,
        })
        print(f"done ({elapsed:.1f}s)")

    # 결과 테이블
    has_obstacles = len(scenario["obstacles"]) > 0
    print(f"\n{'=' * 108}")
    header = (
        f"{'Method':<22} {'RMSE':>8} {'MaxErr':>8} "
        f"{'MeanESS':>9} {'SolveMs':>9} {'Train':>7} {'Loss(0->F)':>16}"
    )
    if has_obstacles:
        header += f" {'Collis':>7} {'MinClr':>8}"
    print(header)
    print(f"{'=' * 108}")
    for r in all_results:
        if r["last_loss"] is not None and r["init_loss"] is not None:
            loss_str = f"{r['init_loss']:.3f}->{r['last_loss']:.3f}"
        else:
            loss_str = "-"
        line = (
            f"{r['name']:<22} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>8.4f} "
            f"{r['mean_ess']:>9.1f} "
            f"{r['mean_solve_ms']:>9.2f} "
            f"{r['train_count']:>7d} "
            f"{loss_str:>16}"
        )
        if has_obstacles:
            line += f" {r['n_collisions']:>7d} {r['min_clearance']:>8.3f}"
        print(line)
    print(f"{'=' * 108}")

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-패널 결과 플롯 (2x4)."""
    dt_val = COMMON["dt"]
    trajectory_fn = scenario["trajectory_fn"]
    duration = scenario["duration"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # (0,0) XY 궤적
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

    # (0,1) 추적 오차
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
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1) SolveMs bar
    ax = axes[1, 1]
    solve_ms_list = [r["mean_solve_ms"] for r in results]
    bars_sm = ax.bar(names, solve_ms_list, color=colors, alpha=0.8)
    for bar, val in zip(bars_sm, solve_ms_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
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
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Min Clearance (m)")
        ax.set_title("Min Obstacle Clearance")
    else:
        max_errors = [r["max_error"] for r in results]
        bars_me = ax.bar(names, max_errors, color=colors, alpha=0.8)
        for bar, val in zip(bars_me, max_errors):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Max Error (m)")
        ax.set_title("Max Tracking Error")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) Train count / Loss text
    ax = axes[1, 3]
    ax.axis("off")
    ax.set_title("Step-MPPI Learning Stats")
    lines = []
    for r in results:
        if r["last_loss"] is not None and r["init_loss"] is not None:
            loss_str = f"{r['init_loss']:.3f}->{r['last_loss']:.3f}"
        else:
            loss_str = "-"
        lines.append(
            f"{r['short']:>8}: net={int(r['use_net'])} "
            f"train={r['train_count']} buf={r['buffer_size']}\n"
            f"          loss={loss_str} dNorm={r['final_delta_norm']:.3f}"
        )
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", fontsize=8, family="monospace")

    plt.suptitle(
        f"Step-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla(K=512) vs Vanilla(K=64) vs Step(no train) vs Step(online)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/step_mppi_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# -- 라이브 애니메이션 --

def run_live(args):
    """실시간 4-way 비교 애니메이션 → GIF/MP4 저장."""
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
    print(f"  Step-MPPI Live -- {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | dt: {dt_val}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "solve_ms": []}
        for k in controllers
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Step-MPPI Live -- {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")
    for ox, oy, r in scenario["obstacles"]:
        ax_xy.add_patch(Circle((ox, oy), r, color="red", alpha=0.3))
    ref_t_arr = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t)[:2] for t in ref_t_arr])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

    lines_xy = {}
    dots = {}
    for name, color in COLORS.items():
        lines_xy[name], = ax_xy.plot([], [], color=color, linewidth=2, label=SHORT[name])
        dots[name], = ax_xy.plot([], [], "o", color=color, markersize=8)
    ax_xy.legend(loc="upper left", fontsize=7)

    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, color in COLORS.items():
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5, label=SHORT[name])
    ax_err.legend(fontsize=7)

    ax_ess = axes[1, 0]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5, label=SHORT[name])
    ax_ess.legend(fontsize=7)

    ax_info = axes[1, 1]
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
        times = np.array(data["Vanilla (K=512)"]["times"])

        for name in controllers:
            xy = np.array(data[name]["xy"])
            if len(xy) > 0:
                lines_xy[name].set_data(xy[:, 0], xy[:, 1])
                dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
                lines_err[name].set_data(
                    times[:len(data[name]["errors"])], data[name]["errors"])
                lines_ess[name].set_data(
                    times[:len(data[name]["ess"])], data[name]["ess"])

        ax_xy.relim(); ax_xy.autoscale_view(); ax_xy.set_aspect("equal")
        ax_err.relim(); ax_err.autoscale_view()
        ax_ess.relim(); ax_ess.autoscale_view()

        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            avg_ms = np.mean(data[name]["solve_ms"]) if data[name]["solve_ms"] else 0
            tc = 0
            if hasattr(controllers[name], "trainer") and controllers[name].trainer:
                tc = len(controllers[name].trainer._loss_history)
            lines.append(
                f"{SHORT[name]:>8}: RMSE={rmse:.3f} ESS={ess:.0f} "
                f"ms={avg_ms:.1f} train={tc}"
            )
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario
    gif_path = f"plots/step_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/step_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="Step-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "online_learning", "figure8"],
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
