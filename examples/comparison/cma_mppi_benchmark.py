#!/usr/bin/env python3
"""
CMA-MPPI (Covariance Matrix Adaptation MPPI) 벤치마크: 3-Way x 2 시나리오

방법:
  1. Vanilla MPPI  — 단일 반복, 등방 노이즈
  2. DIAL-MPPI     — 다중 반복, 고정 기하급수 어닐링
  3. CMA-MPPI      — 다중 반복, 적응적 공분산 학습

시나리오:
  A. simple   — 장애물 없음, 원형 추적 (등방적 비용)
  B. obstacle — 비대칭 장애물 배치 (공분산 적응 시각화)

측정:
  - 위치 추적 RMSE
  - 계산 시간 (min/max/mean)
  - CMA 공분산 히트맵 (핵심 시각화)
  - 반복별 비용 개선
  - ESS (Effective Sample Size)

Usage:
    PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py
    PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --scenario obstacle
    PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --trajectory figure8
    PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --no-plot
    PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --live
    PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --live --scenario obstacle
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
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.cma_mppi import CMAMPPIController
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
from mppi_controller.simulation.harness import SimulationHarness


# ── 시나리오 설정 ────────────────────────────────────────────

OBSTACLES = [
    (2.5, 2.0, 0.4),
    (-1.5, 3.0, 0.5),
    (1.0, -3.0, 0.3),
]


def get_scenarios():
    return {
        "simple": {
            "name": "Simple (no obstacles)",
            "obstacles": None,
        },
        "obstacle": {
            "name": "Obstacle (asymmetric layout)",
            "obstacles": OBSTACLES,
        },
    }


def create_trajectory_fn(name):
    if name == "circle":
        return lambda t: circle_trajectory(t, radius=3.0)
    elif name == "figure8":
        return figure_eight_trajectory
    return lambda t: circle_trajectory(t, radius=3.0)


# ── 시뮬레이션 실행 ──────────────────────────────────────────

def run_simulation(model, controller, reference_fn, initial_state, dt, duration):
    """SimulationHarness 기반 시뮬레이션"""
    harness = SimulationHarness(dt=dt, headless=True, seed=42)
    harness.add_controller("ctrl", controller, model)
    results = harness.run(reference_fn, initial_state, duration)
    r = results["ctrl"]
    h = r["history"]

    return {
        "states": np.vstack([initial_state[None, :], h["state"]]),
        "controls": h["control"],
        "solve_times": h["solve_time"],
        "infos": h.get("info", []),
    }


def compute_tracking_rmse(states, trajectory_fn, dt):
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    return np.sqrt(np.mean(np.array(errors) ** 2))


def compute_obstacle_metrics(states, obstacles):
    if not obstacles:
        return {"n_collisions": 0, "min_clearance": float("inf")}
    n_collisions = 0
    min_dist = float("inf")
    for st in states:
        x, y = st[0], st[1]
        for ox, oy, r in obstacles:
            clearance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2) - r
            min_dist = min(min_dist, clearance)
            if clearance < 0:
                n_collisions += 1
    return {"n_collisions": n_collisions, "min_clearance": min_dist}


# ── 컨트롤러/비용 공통 셋업 ──────────────────────────────────

COLORS = {
    "Vanilla MPPI": "#2196F3",
    "DIAL-MPPI": "#FF9800",
    "CMA-MPPI": "#4CAF50",
}

COMMON = dict(
    K=512, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)


def _make_cost(params, obstacles):
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.3, cost_weight=2000.0))
    return CompositeMPPICost(costs)


def _make_controllers(model, obstacles):
    """3가지 컨트롤러 생성 (dict 반환)"""
    vanilla_p = MPPIParams(**COMMON)
    vanilla = MPPIController(model, vanilla_p, cost_function=_make_cost(vanilla_p, obstacles))

    dial_p = DIALMPPIParams(
        **COMMON, n_diffuse_init=8, n_diffuse=3,
        traj_diffuse_factor=0.5, horizon_diffuse_factor=0.5,
        use_reward_normalization=True,
    )
    dial = DIALMPPIController(model, dial_p, cost_function=_make_cost(dial_p, obstacles))

    cma_p = CMAMPPIParams(
        **COMMON, n_iters_init=8, n_iters=3,
        cov_learning_rate=0.5, sigma_min=0.05, sigma_max=3.0,
        use_mean_shift=True, use_reward_normalization=True,
    )
    cma = CMAMPPIController(model, cma_p, cost_function=_make_cost(cma_p, obstacles))

    return {
        "Vanilla MPPI": vanilla,
        "DIAL-MPPI": dial,
        "CMA-MPPI": cma,
    }


# ── Live 애니메이션 ──────────────────────────────────────────

def run_live(args):
    """실시간 3-way 비교 애니메이션 (FuncAnimation) → GIF/MP4 저장"""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    obstacles = scenario["obstacles"] or []
    trajectory_fn = create_trajectory_fn(args.trajectory)

    model = DifferentialDriveKinematic(wheelbase=0.5)
    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = args.duration
    num_steps = int(duration / dt)

    np.random.seed(args.seed)
    controllers = _make_controllers(model, scenario["obstacles"])

    print(f"\n{'=' * 60}")
    print(f"  CMA-MPPI Live — {scenario['name']}")
    print(f"  Trajectory: {args.trajectory} | Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # 시뮬레이션 상태
    states = {k: np.array([0.0, 0.0, 0.0]) for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "cov_v": [], "cov_w": []}
        for k in controllers
    }

    # ── Figure 설정 (2x3 6패널) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"CMA-MPPI Live — {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY 궤적
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")
    for ox, oy, r in obstacles:
        ax_xy.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))
        ax_xy.add_patch(plt.Circle((ox, oy), r + 0.3, fill=False,
                                    edgecolor="red", alpha=0.3, linestyle="--"))
    ref_t = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t) for t in ref_t])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

    lines_xy = {}
    dots = {}
    for name, color in COLORS.items():
        lines_xy[name], = ax_xy.plot([], [], color=color, linewidth=2, label=name)
        dots[name], = ax_xy.plot([], [], "o", color=color, markersize=8)
    ax_xy.legend(loc="upper left", fontsize=7)

    # [0,1] 위치 오차
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Position Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, color in COLORS.items():
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5, label=name)
    ax_err.legend(fontsize=7)

    # [0,2] CMA 공분산 진화 (핵심!)
    ax_cov = axes[0, 2]
    ax_cov.set_xlabel("Time (s)")
    ax_cov.set_ylabel("Adapted sigma")
    ax_cov.set_title("CMA Covariance Evolution")
    ax_cov.grid(True, alpha=0.3)
    line_cov_v, = ax_cov.plot([], [], color="#E91E63", linewidth=2, label="sigma_v (linear)")
    line_cov_w, = ax_cov.plot([], [], color="#9C27B0", linewidth=2, label="sigma_w (angular)")
    ax_cov.legend(fontsize=7)

    # [1,0] ESS
    ax_ess = axes[1, 0]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5, label=name)
    ax_ess.legend(fontsize=7)

    # [1,1] RMSE 바 차트 (실시간 갱신)
    ax_bar = axes[1, 1]
    ax_bar.set_ylabel("RMSE (m)")
    ax_bar.set_title("Running RMSE")
    ax_bar.grid(True, alpha=0.3, axis="y")
    bar_names = list(COLORS.keys())
    bar_colors = [COLORS[n] for n in bar_names]
    bars = ax_bar.bar(range(len(bar_names)), [0] * len(bar_names), color=bar_colors, alpha=0.8)
    ax_bar.set_xticks(range(len(bar_names)))
    ax_bar.set_xticklabels(["Vanilla", "DIAL", "CMA"], fontsize=9)
    bar_texts = [
        ax_bar.text(b.get_x() + b.get_width() / 2, 0, "", ha="center", va="bottom", fontsize=9)
        for b in bars
    ]

    # [1,2] 상태 텍스트 패널
    ax_info = axes[1, 2]
    ax_info.axis("off")
    ax_info.set_title("Statistics")
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        va="top", fontsize=10, family="monospace",
    )

    plt.tight_layout()

    all_artists = (
        list(lines_xy.values()) + list(dots.values())
        + list(lines_err.values()) + list(lines_ess.values())
        + [line_cov_v, line_cov_w, info_text]
    )

    def init():
        for a in all_artists:
            if hasattr(a, "set_data"):
                a.set_data([], [])
        info_text.set_text("")
        return all_artists

    def update(frame):
        if frame >= num_steps:
            return all_artists

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

            # CMA 공분산 추출
            if name == "CMA-MPPI" and "cma_stats" in info:
                cov_per_dim = info["cma_stats"]["cov_per_dim"]
                data[name]["cov_v"].append(np.sqrt(cov_per_dim[0]))
                data[name]["cov_w"].append(np.sqrt(cov_per_dim[1]))

        sim_t[0] += dt

        # 그래프 업데이트
        times = np.array(data["Vanilla MPPI"]["times"])

        for name in controllers:
            xy = np.array(data[name]["xy"])
            lines_xy[name].set_data(xy[:, 0], xy[:, 1])
            dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
            lines_err[name].set_data(times, data[name]["errors"])
            lines_ess[name].set_data(times, data[name]["ess"])

        # CMA 공분산
        cma_data = data["CMA-MPPI"]
        if cma_data["cov_v"]:
            cov_times = times[:len(cma_data["cov_v"])]
            line_cov_v.set_data(cov_times, cma_data["cov_v"])
            line_cov_w.set_data(cov_times, cma_data["cov_w"])
            ax_cov.relim()
            ax_cov.autoscale_view()

        ax_err.relim(); ax_err.autoscale_view()
        ax_ess.relim(); ax_ess.autoscale_view()

        # RMSE 바 차트 업데이트
        rmses = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            rmses.append(rmse)
            bars[i].set_height(rmse)
            bar_texts[i].set_position((bars[i].get_x() + bars[i].get_width() / 2, rmse))
            bar_texts[i].set_text(f"{rmse:.3f}")
        if rmses:
            ax_bar.set_ylim(0, max(rmses) * 1.3 + 0.01)

        # 정보 텍스트
        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            short = name.replace(" MPPI", "").replace("-MPPI", "")
            lines.append(f"{short:>8}: RMSE={rmse:.4f}m  ESS={ess:.0f}")
        if cma_data["cov_v"]:
            lines.append(f"\nCMA sigma_v: {cma_data['cov_v'][-1]:.3f}")
            lines.append(f"CMA sigma_w: {cma_data['cov_w'][-1]:.3f}")
        info_text.set_text("\n".join(lines))

        return all_artists

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    # GIF 저장
    gif_path = f"plots/cma_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    # MP4 저장
    try:
        mp4_path = f"plots/cma_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # 종료 통계
    print(f"\n{'=' * 60}")
    print(f"  Final Statistics — {scenario['name']}")
    print(f"{'=' * 60}")
    print(f"\n  {'Method':<20} {'RMSE':>8} {'Mean ESS':>10}")
    print(f"  {'-' * 40}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        print(f"  {name:<20} {rmse:>8.4f} {mean_ess:>10.1f}")
    cma_d = data["CMA-MPPI"]
    if cma_d["cov_v"]:
        print(f"\n  CMA final sigma_v: {cma_d['cov_v'][-1]:.4f}")
        print(f"  CMA final sigma_w: {cma_d['cov_w'][-1]:.4f}")
    print(f"{'=' * 60}\n")


# ── 벤치마크 메인 ─────────────────────────────────────────────

def run_benchmark(args):
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    has_obstacles = scenario["obstacles"] is not None

    trajectory_fn = create_trajectory_fn(args.trajectory)

    print(f"\n{'=' * 72}")
    print(f"  CMA-MPPI Benchmark: 3-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  Trajectory: {args.trajectory} | Duration: {args.duration}s | Seed: {args.seed}")
    print(f"{'=' * 72}")

    model = DifferentialDriveKinematic(wheelbase=0.5)
    initial_state = np.array([0.0, 0.0, 0.0])

    common = dict(
        K=512, N=30, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )

    # ── 3가지 방법 ──

    def make_cost(params):
        costs = [
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
            ControlEffortCost(params.R),
        ]
        if has_obstacles:
            costs.append(ObstacleCost(scenario["obstacles"], safety_margin=0.3, cost_weight=2000.0))
        return CompositeMPPICost(costs)

    def make_vanilla():
        params = MPPIParams(**common)
        cost = make_cost(params)
        return MPPIController(model, params, cost_function=cost)

    def make_dial():
        params = DIALMPPIParams(
            **common,
            n_diffuse_init=8,
            n_diffuse=3,
            traj_diffuse_factor=0.5,
            horizon_diffuse_factor=0.5,
            use_reward_normalization=True,
        )
        cost = make_cost(params)
        return DIALMPPIController(model, params, cost_function=cost)

    def make_cma():
        params = CMAMPPIParams(
            **common,
            n_iters_init=8,
            n_iters=3,
            cov_learning_rate=0.5,
            sigma_min=0.05,
            sigma_max=3.0,
            use_mean_shift=True,
            use_reward_normalization=True,
        )
        cost = make_cost(params)
        return CMAMPPIController(model, params, cost_function=cost)

    variants = [
        {"name": "Vanilla MPPI", "short": "Vanilla", "make": make_vanilla, "color": "#2196F3"},
        {"name": "DIAL-MPPI",    "short": "DIAL",    "make": make_dial,    "color": "#FF9800"},
        {"name": "CMA-MPPI",     "short": "CMA",     "make": make_cma,     "color": "#4CAF50"},
    ]

    ref_fn = lambda t, _fn=trajectory_fn, _N=common["N"], _dt=common["dt"]: \
        generate_reference_trajectory(_fn, t, _N, _dt)

    # ── 실행 + 수집 ──
    all_results = []
    for i, var in enumerate(variants):
        np.random.seed(args.seed)

        print(f"\n  [{i+1}/{len(variants)}] {var['name']:<22}", end=" ", flush=True)
        t_start = time.time()

        controller = var["make"]()
        history = run_simulation(
            model, controller, ref_fn, initial_state, common["dt"], args.duration,
        )
        elapsed = time.time() - t_start

        rmse = compute_tracking_rmse(history["states"], trajectory_fn, common["dt"])
        solve_times = history["solve_times"]
        mean_solve = np.mean(solve_times) * 1000
        max_solve = np.max(solve_times) * 1000
        min_solve = np.min(solve_times) * 1000

        obs_metrics = compute_obstacle_metrics(
            history["states"], scenario["obstacles"] if has_obstacles else None
        )

        # ESS 추출
        ess_list = []
        for info in history["infos"]:
            if isinstance(info, dict) and "ess" in info:
                ess_list.append(info["ess"])

        # CMA 고유 메트릭
        cma_cov_history = []
        cma_iteration_costs_history = []
        if var["short"] == "CMA" and history["infos"]:
            for info in history["infos"]:
                if isinstance(info, dict) and "cma_stats" in info:
                    cma_stats = info["cma_stats"]
                    cma_cov_history.append(cma_stats["cov_per_dim"])
                    cma_iteration_costs_history.append(cma_stats["iteration_costs"])

        # DIAL 고유 메트릭
        dial_iteration_costs_history = []
        if var["short"] == "DIAL" and history["infos"]:
            for info in history["infos"]:
                if isinstance(info, dict) and "dial_stats" in info:
                    dial_iteration_costs_history.append(info["dial_stats"]["iteration_costs"])

        all_results.append({
            "name": var["name"],
            "short": var["short"],
            "color": var["color"],
            "rmse": rmse,
            "mean_solve_ms": mean_solve,
            "max_solve_ms": max_solve,
            "min_solve_ms": min_solve,
            "elapsed": elapsed,
            "states": history["states"],
            "infos": history["infos"],
            "obs_metrics": obs_metrics,
            "ess_list": ess_list,
            "cma_cov_history": cma_cov_history,
            "cma_iteration_costs_history": cma_iteration_costs_history,
            "dial_iteration_costs_history": dial_iteration_costs_history,
        })

        print(f"done ({elapsed:.1f}s)")

    # ── 결과 출력 ──
    print(f"\n{'─' * 72}")
    if has_obstacles:
        print(f"{'Method':<22} {'RMSE':>8} {'Collisions':>10} {'MinClear':>10} {'MeanTime':>10} {'MaxTime':>10}")
    else:
        print(f"{'Method':<22} {'RMSE':>8} {'MeanTime(ms)':>14} {'MinTime(ms)':>14} {'MaxTime(ms)':>14}")
    print(f"{'─' * 72}")

    for r in all_results:
        if has_obstacles:
            print(
                f"{r['name']:<22} "
                f"{r['rmse']:>8.4f} "
                f"{r['obs_metrics']['n_collisions']:>10d} "
                f"{r['obs_metrics']['min_clearance']:>10.3f} "
                f"{r['mean_solve_ms']:>10.1f} "
                f"{r['max_solve_ms']:>10.1f}"
            )
        else:
            print(
                f"{r['name']:<22} "
                f"{r['rmse']:>8.4f} "
                f"{r['mean_solve_ms']:>14.1f} "
                f"{r['min_solve_ms']:>14.1f} "
                f"{r['max_solve_ms']:>14.1f}"
            )

    print(f"{'─' * 72}")

    # ESS 출력
    for r in all_results:
        if r["ess_list"]:
            print(f"  {r['short']} ESS: mean={np.mean(r['ess_list']):.1f}, "
                  f"min={np.min(r['ess_list']):.1f}")

    # ── 플롯 생성 ──
    if not args.no_plot:
        _plot_results(all_results, common["dt"], args.duration, trajectory_fn,
                      args.scenario, scenario)

    print()
    return all_results


def _plot_results(results, dt, duration, trajectory_fn, scenario_name, scenario):
    """6-panel 결과 플롯 (2x3)"""
    has_obstacles = scenario["obstacles"] is not None
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. (0,0) XY 궤적
    ax = axes[0, 0]
    t_arr = np.arange(0, duration + dt, dt)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, label="Reference", linewidth=1)

    for r in results:
        states = r["states"]
        ax.plot(states[:, 0], states[:, 1], color=r["color"],
                label=r["short"], linewidth=1.5)

    if has_obstacles:
        for ox, oy, radius in scenario["obstacles"]:
            circle = Circle((ox, oy), radius, fill=True, facecolor="#FF5252",
                            edgecolor="red", alpha=0.3, linewidth=1.5)
            ax.add_patch(circle)
            margin_circle = Circle((ox, oy), radius + 0.3, fill=False,
                                   edgecolor="red", alpha=0.3, linestyle="--")
            ax.add_patch(margin_circle)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. (0,1) 위치 오차
    ax = axes[0, 1]
    for r in results:
        states = r["states"]
        errors = []
        for i, st in enumerate(states):
            ref_pt = trajectory_fn(i * dt)
            err = np.sqrt((st[0] - ref_pt[0]) ** 2 + (st[1] - ref_pt[1]) ** 2)
            errors.append(err)
        t_plot = np.arange(len(errors)) * dt
        ax.plot(t_plot, errors, color=r["color"], label=r["short"], linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. (0,2) CMA 공분산 히트맵 (핵심 시각화)
    ax = axes[0, 2]
    cma_result = next((r for r in results if r["short"] == "CMA"), None)
    if cma_result and cma_result["cma_cov_history"]:
        cov_data = np.array(cma_result["cma_cov_history"])  # (T, nu)
        t_steps = np.arange(len(cov_data)) * dt
        sigma_data = np.sqrt(cov_data)  # 표준편차로 변환

        dim_labels = ["v (linear)", "w (angular)"]
        for d in range(sigma_data.shape[1]):
            ax.plot(t_steps, sigma_data[:, d], label=f"sigma_{dim_labels[d]}",
                    linewidth=1.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Adapted sigma")
        ax.set_title("CMA Covariance Evolution")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No CMA data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, alpha=0.5)
        ax.set_title("CMA Covariance Evolution")
    ax.grid(True, alpha=0.3)

    # 4. (1,0) 반복별 비용 개선
    ax = axes[1, 0]
    for r in results:
        if r["short"] == "CMA" and r["cma_iteration_costs_history"]:
            # 마지막 10스텝의 평균 반복별 비용
            recent = r["cma_iteration_costs_history"][-min(10, len(r["cma_iteration_costs_history"])):]
            # 각 반복의 비용을 평균
            max_iters = max(len(c) for c in recent)
            avg_costs = []
            for it in range(max_iters):
                vals = [c[it] for c in recent if len(c) > it]
                avg_costs.append(np.mean(vals))
            ax.plot(range(1, len(avg_costs) + 1), avg_costs, color=r["color"],
                    label="CMA", linewidth=1.5, marker="o", markersize=4)

        if r["short"] == "DIAL" and r["dial_iteration_costs_history"]:
            recent = r["dial_iteration_costs_history"][-min(10, len(r["dial_iteration_costs_history"])):]
            max_iters = max(len(c) for c in recent)
            avg_costs = []
            for it in range(max_iters):
                vals = [c[it] for c in recent if len(c) > it]
                avg_costs.append(np.mean(vals))
            ax.plot(range(1, len(avg_costs) + 1), avg_costs, color=r["color"],
                    label="DIAL", linewidth=1.5, marker="s", markersize=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost (avg)")
    ax.set_title("Per-Iteration Cost Improvement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. (1,1) ESS
    ax = axes[1, 1]
    for r in results:
        if r["ess_list"]:
            t_ess = np.arange(len(r["ess_list"])) * dt
            ax.plot(t_ess, r["ess_list"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. (1,2) RMSE 바 차트
    ax = axes[1, 2]
    names = [r["short"] for r in results]
    rmses = [r["rmse"] for r in results]
    colors = [r["color"] for r in results]
    bars = ax.bar(names, rmses, color=colors, alpha=0.8)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Tracking RMSE Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"CMA-MPPI Benchmark [{scenario_name}]: Vanilla vs DIAL vs CMA-MPPI",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/cma_mppi_benchmark_{scenario_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CMA-MPPI Benchmark")
    parser.add_argument("--scenario", default="simple",
                        choices=["simple", "obstacle"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--trajectory", default="circle",
                        choices=["circle", "figure8"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true", help="Realtime animation")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.live:
        if args.all_scenarios:
            for scenario_name in get_scenarios():
                args.scenario = scenario_name
                run_live(args)
        else:
            run_live(args)
    elif args.all_scenarios:
        for scenario_name in get_scenarios():
            args.scenario = scenario_name
            run_benchmark(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
