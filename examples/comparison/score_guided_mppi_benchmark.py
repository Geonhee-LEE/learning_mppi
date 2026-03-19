#!/usr/bin/env python3
"""
SG-MPPI (Score-Guided MPPI) 벤치마크: 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI    — 기준선
  2. DIAL-MPPI       — 다단계 어닐링 (비학습)
  3. Flow-MPPI       — CFM 분포 학습 (완전 대체)
  4. SG-MPPI         — Score-guided bias (하이브리드)

시나리오 4개:
  A. simple          — 기준선 비교 (장애물 없음)
  B. obstacles       — 3개 원형 장애물
  C. multimodal      — 대칭 장애물 (2경로 선택)
  D. online_learning — 3개 장애물 + 온라인 학습

Usage:
    PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --scenario simple
    PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --live --scenario obstacles
    PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --no-plot
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
    FlowMPPIParams,
    SGMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.flow_mppi import FlowMPPIController
from mppi_controller.controllers.mppi.score_guided_mppi import SGMPPIController
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
    "Flow-MPPI": "#4CAF50",
    "SG-MPPI": "#E91E63",
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
            "description": "No obstacles, circle — all methods equivalent",
            "use_online": False,
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
            "description": "3 circular obstacles — score learns avoidance direction",
            "use_online": False,
        },
        "multimodal": {
            "name": "C. Multimodal Cost",
            "obstacles": [
                (1.0, 1.0, 0.4),
                (-1.0, 1.0, 0.4),
                (1.0, -1.0, 0.4),
                (-1.0, -1.0, 0.4),
            ],
            "trajectory_fn": figure_eight_trajectory,
            "initial_state": np.array([0.0, 0.0, 0.0]),
            "duration": 15.0,
            "description": "Symmetric obstacles (2 paths), figure8 — path selection",
            "use_online": False,
        },
        "online_learning": {
            "name": "D. Online Learning",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.5),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 20.0,
            "description": "3 obstacles + online training — real-time adaptation",
            "use_online": True,
        },
    }


# ── 컨트롤러 생성 ─────────────────────────────────────────────

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


def _make_controllers(model, scenario):
    """4가지 컨트롤러 생성"""
    obstacles = scenario["obstacles"]
    use_online = scenario.get("use_online", False)

    # Vanilla
    v_params = MPPIParams(**COMMON)
    vanilla = MPPIController(model, v_params, cost_function=_make_cost(v_params, obstacles))

    # DIAL-MPPI
    d_params = DIALMPPIParams(**COMMON, n_diffuse_init=8, n_diffuse=3)
    dial = DIALMPPIController(model, d_params,
                              cost_function=_make_cost(d_params, obstacles))

    # Flow-MPPI
    f_params = FlowMPPIParams(**COMMON, flow_mode="replace_mean",
                              flow_online_training=use_online,
                              flow_training_interval=20,
                              flow_min_samples=50,
                              flow_buffer_size=2000)
    flow = FlowMPPIController(model, f_params,
                              cost_function=_make_cost(f_params, obstacles))

    # SG-MPPI
    sg_kwargs = dict(
        guidance_scale=0.5,
        n_guide_iters=3,
        use_annealing=True,
        guidance_decay=0.95,
        score_hidden_dims=[128, 128],
    )
    if use_online:
        sg_kwargs.update(
            score_online_training=True,
            score_training_interval=20,
            score_min_samples=50,
            score_buffer_size=2000,
        )
    sg_params = SGMPPIParams(**COMMON, **sg_kwargs)
    sg = SGMPPIController(model, sg_params,
                          cost_function=_make_cost(sg_params, obstacles))

    return {
        "Vanilla MPPI": vanilla,
        "DIAL-MPPI": dial,
        "Flow-MPPI": flow,
        "SG-MPPI": sg,
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
    }


# ── Live 애니메이션 ─────────────────────────────────────────────

def run_live(args):
    """실시간 4-way 비교 애니메이션 → GIF/MP4 저장"""
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
    print(f"  SG-MPPI Live — {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # 상태 초기화
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": []}
        for k in controllers
    }

    # Figure 설정 (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"SG-MPPI Live — {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY 궤적
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

    # [0,1] 위치 오차
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, color in COLORS.items():
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5, label=name)
    ax_err.legend(fontsize=6)

    # [0,2] ESS
    ax_ess = axes[0, 2]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5, label=name)
    ax_ess.legend(fontsize=6)

    # [0,3] Score magnitude (SG-MPPI only)
    ax_score = axes[0, 3]
    ax_score.set_xlabel("Time (s)")
    ax_score.set_ylabel("Score Magnitude")
    ax_score.set_title("Score Magnitude (SG-MPPI)")
    ax_score.grid(True, alpha=0.3)
    score_data = {"times": [], "magnitudes": []}
    line_score, = ax_score.plot([], [], color=COLORS["SG-MPPI"], linewidth=1.5)

    # [1,0] 가중치 분포
    ax_wt = axes[1, 0]
    ax_wt.set_xlabel("Weight")
    ax_wt.set_ylabel("Count")
    ax_wt.set_title("Weight Distribution (last step)")
    ax_wt.grid(True, alpha=0.3)

    # [1,1] RMSE 바 차트
    ax_rmse = axes[1, 1]
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Running RMSE")
    ax_rmse.grid(True, alpha=0.3, axis="y")
    bar_names = list(COLORS.keys())
    bar_colors = [COLORS[n] for n in bar_names]
    bars_rmse = ax_rmse.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_rmse.set_xticks(range(len(bar_names)))
    ax_rmse.set_xticklabels(["Van", "DIAL", "Flow", "SG"], fontsize=8)
    bar_rmse_texts = [
        ax_rmse.text(b.get_x() + b.get_width() / 2, 0, "", ha="center",
                     va="bottom", fontsize=8)
        for b in bars_rmse
    ]

    # [1,2] MeanESS 바 차트
    ax_mc = axes[1, 2]
    ax_mc.set_ylabel("Mean ESS")
    ax_mc.set_title("Mean ESS")
    ax_mc.grid(True, alpha=0.3, axis="y")
    bars_mc = ax_mc.bar(range(len(bar_names)), [0] * len(bar_names),
                        color=bar_colors, alpha=0.8)
    ax_mc.set_xticks(range(len(bar_names)))
    ax_mc.set_xticklabels(["Van", "DIAL", "Flow", "SG"], fontsize=8)

    # [1,3] 통계 텍스트
    ax_info = axes[1, 3]
    ax_info.axis("off")
    ax_info.set_title("Statistics")
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        va="top", fontsize=9, family="monospace",
    )

    plt.tight_layout()

    last_infos = {}

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
            last_infos[name] = info

            # Score magnitude 추적
            if name == "SG-MPPI":
                score_stats = info.get("score_stats", {})
                score_data["times"].append(t)
                score_data["magnitudes"].append(
                    score_stats.get("mean_score_magnitude", 0.0)
                )

        sim_t[0] += dt

        # 그래프 업데이트
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

        # Score magnitude
        if score_data["times"]:
            line_score.set_data(score_data["times"], score_data["magnitudes"])
            ax_score.relim()
            ax_score.autoscale_view()

        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal")
        ax_err.relim()
        ax_err.autoscale_view()
        ax_ess.relim()
        ax_ess.autoscale_view()

        # RMSE 바 차트
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

        # ESS 바 차트
        ess_vals = []
        for i, name in enumerate(bar_names):
            ess_d = data[name]["ess"]
            mean_ess = np.mean(ess_d) if ess_d else 0
            ess_vals.append(mean_ess)
            bars_mc[i].set_height(mean_ess)
        if ess_vals:
            ax_mc.set_ylim(0, max(ess_vals) * 1.3 + 1)

        # 가중치 분포
        if frame % 10 == 0 and last_infos:
            ax_wt.cla()
            ax_wt.set_xlabel("Weight")
            ax_wt.set_ylabel("Count")
            ax_wt.set_title("Weight Distribution (last)")
            ax_wt.grid(True, alpha=0.3)
            for name, color in COLORS.items():
                if name in last_infos:
                    w = last_infos[name].get("sample_weights", np.array([]))
                    if len(w) > 0:
                        ax_wt.hist(w, bins=30, alpha=0.5, color=color,
                                   label=name.replace(" MPPI", ""))
            ax_wt.legend(fontsize=6)

        # 통계 텍스트
        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            short = name.replace(" MPPI", "").replace("-MPPI", "")
            lines.append(f"{short:>10}: RMSE={rmse:.3f} ESS={ess:.0f}")
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/score_guided_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/score_guided_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # 종료 통계
    print(f"\n{'=' * 72}")
    print(f"  Final Statistics — {scenario['name']}")
    print(f"{'=' * 72}")
    print(f"  {'Method':<18} {'RMSE':>8} {'MaxError':>10} {'MeanESS':>10}")
    print(f"  {'-' * 48}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        me = max(errs) if errs else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        print(f"  {name:<18} {rmse:>8.4f} {me:>10.4f} {mean_ess:>10.1f}")
    print(f"{'=' * 72}\n")


# ── 벤치마크 메인 ─────────────────────────────────────────────

def run_benchmark(args):
    """정적 벤치마크 실행"""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]

    print(f"\n{'=' * 80}")
    print(f"  SG-MPPI Benchmark: 4-Way Comparison")
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

        # SG-MPPI score stats
        score_mag = 0.0
        if hasattr(ctrl, 'get_score_statistics'):
            stats = ctrl.get_score_statistics()
            score_mag = stats.get("mean_score_magnitude", 0.0)

        all_results.append({
            "name": name,
            "short": name.replace(" MPPI", "").replace("-MPPI", ""),
            "color": COLORS[name],
            "states": history["states"],
            "infos": history["infos"],
            "elapsed": elapsed,
            "score_magnitude": score_mag,
            **metrics,
        })

        print(f"done ({elapsed:.1f}s)")

    # 결과 테이블
    has_obstacles = len(scenario["obstacles"]) > 0
    print(f"\n{'=' * 90}")
    header = f"{'Method':<18} {'RMSE':>8} {'MaxError':>10} {'MeanESS':>10} {'SolveMs':>10}"
    if has_obstacles:
        header += f" {'Collisions':>10} {'MinClear':>10}"
    print(header)
    print(f"{'=' * 90}")
    for r in all_results:
        line = (
            f"{r['name']:<18} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>10.4f} "
            f"{r['mean_ess']:>10.1f} "
            f"{r['mean_solve_ms']:>10.1f}"
        )
        if has_obstacles:
            line += f" {r['n_collisions']:>10d} {r['min_clearance']:>10.3f}"
        print(line)
    print(f"{'=' * 90}")

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

    # (0,1) 위치 오차
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
        if r["ess_list"]:
            t_ess = np.arange(len(r["ess_list"])) * dt
            ax.plot(t_ess, r["ess_list"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,3) Score magnitude
    ax = axes[0, 3]
    for r in results:
        if r["name"] == "SG-MPPI":
            # Extract score magnitude from infos
            mags = []
            for info in r["infos"]:
                ss = info.get("score_stats", {})
                mags.append(ss.get("mean_score_magnitude", 0.0))
            if mags:
                t_plot = np.arange(len(mags)) * dt
                ax.plot(t_plot, mags, color=r["color"], linewidth=1.5, label="SG-MPPI")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score Magnitude")
    ax.set_title("Score Magnitude (SG-MPPI)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,0) 가중치 분포 히스토그램
    ax = axes[1, 0]
    for r in results:
        if r["infos"]:
            last_info = r["infos"][-1]
            w = last_info.get("sample_weights", np.array([]))
            if len(w) > 0:
                ax.hist(w, bins=30, alpha=0.5, color=r["color"],
                        label=r["short"])
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title("Weight Distribution (last step)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) RMSE 바 차트
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

    # (1,2) MinClearance 바 차트
    ax = axes[1, 2]
    if has_obstacles:
        clears = [r["min_clearance"] for r in results]
        bars_mc = ax.bar(names, clears, color=colors, alpha=0.8)
        for bar, val in zip(bars_mc, clears):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height(), 0) + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Min Clearance (m)")
        ax.set_title("Minimum Obstacle Clearance")
    else:
        ax.text(0.5, 0.5, "No Obstacles", transform=ax.transAxes,
                ha="center", fontsize=12, alpha=0.5)
        ax.set_title("Min Clearance (N/A)")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) MeanESS 바 차트
    ax = axes[1, 3]
    ess_vals = [r["mean_ess"] for r in results]
    bars_ess = ax.bar(names, ess_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars_ess, ess_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean ESS")
    ax.set_title("Mean Effective Sample Size")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"SG-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs DIAL vs Flow vs SG-MPPI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/score_guided_mppi_benchmark_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SG-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "multimodal", "online_learning"],
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
