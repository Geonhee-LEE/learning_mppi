#!/usr/bin/env python3
"""
Robust MPPI (R-MPPI) 벤치마크: 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI   — 외란 무시
  2. Tube-MPPI      — 분리된 피드백 (보수적)
  3. Risk-Aware MPPI — CVaR 필터링 (위험 회피)
  4. Robust MPPI    — 통합 피드백 + 외란 샘플링

시나리오 4개:
  A. mild_noise      — 약한 프로세스 노이즈 (기준선)
  B. strong_noise     — 강한 프로세스 노이즈
  C. model_mismatch   — 모델 불일치 + 노이즈
  D. obstacle_noise   — 장애물 + 프로세스 노이즈

Usage:
    PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --scenario mild_noise
    PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --live --scenario strong_noise
    PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --no-plot
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
    RiskAwareMPPIParams,
    RobustMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from mppi_controller.controllers.mppi.robust_mppi import RobustMPPIController
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
    "Tube-MPPI": "#FF9800",
    "Risk-Aware MPPI": "#4CAF50",
    "Robust MPPI": "#E91E63",
}


# ── 시나리오 정의 ──────────────────────────────────────────────

def get_scenarios():
    """4개 벤치마크 시나리오"""
    return {
        "mild_noise": {
            "name": "A. Mild Process Noise",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "process_noise_std": np.array([0.03, 0.03, 0.01]),
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.5,
            "description": "Weak noise baseline, circle trajectory",
        },
        "strong_noise": {
            "name": "B. Strong Process Noise",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "process_noise_std": np.array([0.1, 0.1, 0.05]),
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.5,
            "description": "Strong noise, Vanilla tracking degradation",
        },
        "model_mismatch": {
            "name": "C. Model Mismatch + Noise",
            "obstacles": [],
            "trajectory_fn": figure_eight_trajectory,
            "initial_state": np.array([0.0, 0.0, 0.0]),
            "duration": 12.0,
            "process_noise_std": np.array([0.05, 0.05, 0.02]),
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.6,
            "description": "Wheelbase 0.5->0.6 + process noise, figure8",
        },
        "obstacle_noise": {
            "name": "D. Obstacles + Process Noise",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.5),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "process_noise_std": np.array([0.05, 0.05, 0.02]),
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.5,
            "description": "3 obstacles + process noise, safety + robustness",
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
    noise_std = scenario["process_noise_std"]

    # Vanilla
    v_params = MPPIParams(**COMMON)
    vanilla = MPPIController(model, v_params, cost_function=_make_cost(v_params, obstacles))

    # Tube-MPPI
    t_params = TubeMPPIParams(**COMMON, tube_enabled=True)
    tube = TubeMPPIController(model, t_params)
    # Tube는 장애물 비용 별도 설정
    if obstacles:
        tube.cost_function = _make_cost(t_params, obstacles)

    # Risk-Aware MPPI
    r_params = RiskAwareMPPIParams(**COMMON, cvar_alpha=0.7)
    risk = RiskAwareMPPIController(model, r_params)
    if obstacles:
        risk.cost_function = _make_cost(r_params, obstacles)

    # Robust MPPI
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

    return {
        "Vanilla MPPI": vanilla,
        "Tube-MPPI": tube,
        "Risk-Aware MPPI": risk,
        "Robust MPPI": robust,
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
    process_noise_std = scenario["process_noise_std"]

    state = scenario["initial_state"].copy()

    # 모델 불일치 시뮬레이션용 별도 모델
    real_wheelbase = scenario.get("real_wheelbase", 0.5)
    planner_wheelbase = scenario.get("planner_wheelbase", 0.5)
    if real_wheelbase != planner_wheelbase:
        real_model = DifferentialDriveKinematic(wheelbase=real_wheelbase)
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

        # 실제 모델로 전파 + 프로세스 노이즈
        state_dot = real_model.forward_dynamics(state, control)
        state = state + state_dot * dt
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

    # Tube 폭 (Tube/Robust MPPI만)
    tube_widths = []
    feedback_norms = []
    for info in history["infos"]:
        if isinstance(info, dict):
            if "robust_stats" in info:
                tube_widths.append(info["robust_stats"]["mean_tube_width"])
                feedback_norms.append(info["robust_stats"]["mean_feedback_norm"])
            elif "tube_width" in info:
                tube_widths.append(info["tube_width"])

    return {
        "rmse": rmse,
        "max_error": max_error,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "tube_widths": tube_widths,
        "mean_tube_width": float(np.mean(tube_widths)) if tube_widths else 0.0,
        "feedback_norms": feedback_norms,
        "errors": errors,
    }


# ── Live 애니메이션 ─────────────────────────────────────────────

def run_live(args):
    """실시간 4-way 비교 애니메이션 → GIF/MP4 저장"""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    trajectory_fn = scenario["trajectory_fn"]
    process_noise_std = scenario["process_noise_std"]

    wb = scenario.get("planner_wheelbase", 0.5)
    model = DifferentialDriveKinematic(wheelbase=wb)
    real_wb = scenario.get("real_wheelbase", wb)
    real_model = DifferentialDriveKinematic(wheelbase=real_wb)

    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)

    np.random.seed(args.seed)
    controllers = _make_controllers(model, scenario)

    print(f"\n{'=' * 60}")
    print(f"  R-MPPI Live — {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # 상태 초기화
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "tube_width": []}
        for k in controllers
    }

    # Figure 설정 (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"R-MPPI Live — {scenario['name']}",
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

    # [0,2] Tube 폭
    ax_tw = axes[0, 2]
    ax_tw.set_xlabel("Time (s)")
    ax_tw.set_ylabel("Tube Width (m)")
    ax_tw.set_title("Tube Width (Tube/R-MPPI)")
    ax_tw.grid(True, alpha=0.3)
    lines_tw = {}
    for name in ["Tube-MPPI", "Robust MPPI"]:
        lines_tw[name], = ax_tw.plot([], [], color=COLORS[name], linewidth=1.5, label=name)
    ax_tw.legend(fontsize=6)

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

    # [1,0] 피드백 크기
    ax_fb = axes[1, 0]
    ax_fb.set_xlabel("Time (s)")
    ax_fb.set_ylabel("Feedback Norm")
    ax_fb.set_title("Feedback Magnitude (Tube/R-MPPI)")
    ax_fb.grid(True, alpha=0.3)

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
    ax_rmse.set_xticklabels(["Van", "Tube", "Risk", "R-MPPI"], fontsize=8)
    bar_rmse_texts = [
        ax_rmse.text(b.get_x() + b.get_width() / 2, 0, "", ha="center",
                     va="bottom", fontsize=8)
        for b in bars_rmse
    ]

    # [1,2] MaxError 바 차트
    ax_me = axes[1, 2]
    ax_me.set_ylabel("Max Error (m)")
    ax_me.set_title("Max Tracking Error")
    ax_me.grid(True, alpha=0.3, axis="y")
    bars_me = ax_me.bar(range(len(bar_names)), [0] * len(bar_names),
                        color=bar_colors, alpha=0.8)
    ax_me.set_xticks(range(len(bar_names)))
    ax_me.set_xticklabels(["Van", "Tube", "Risk", "R-MPPI"], fontsize=8)

    # [1,3] 통계 텍스트
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

            state_dot = real_model.forward_dynamics(states[name], control)
            states[name] = states[name] + state_dot * dt
            states[name] = states[name] + np.random.randn(3) * process_noise_std

            ref_pt = trajectory_fn(t)[:2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))

            tw = 0.0
            if "robust_stats" in info:
                tw = info["robust_stats"]["mean_tube_width"]
            elif "tube_width" in info:
                tw = info["tube_width"]
            data[name]["tube_width"].append(tw)

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

        for name in ["Tube-MPPI", "Robust MPPI"]:
            tw_data = data[name]["tube_width"]
            if tw_data:
                lines_tw[name].set_data(times[:len(tw_data)], tw_data)

        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal")
        ax_err.relim()
        ax_err.autoscale_view()
        ax_tw.relim()
        ax_tw.autoscale_view()
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

        # MaxError 바 차트
        max_errs = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            me = max(errs) if errs else 0
            max_errs.append(me)
            bars_me[i].set_height(me)
        if max_errs:
            ax_me.set_ylim(0, max(max_errs) * 1.3 + 0.01)

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

    gif_path = f"plots/robust_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/robust_mppi_live_{scenario_key}.mp4"
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
    print(f"  R-MPPI Benchmark: 4-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {scenario['duration']}s | Seed: {args.seed}")
    print(f"{'=' * 80}")

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

    # 결과 테이블
    has_obstacles = len(scenario["obstacles"]) > 0
    print(f"\n{'=' * 90}")
    header = f"{'Method':<18} {'RMSE':>8} {'MaxError':>10} {'MeanTube':>10} {'MeanESS':>10} {'SolveMs':>10}"
    if has_obstacles:
        header += f" {'Collisions':>10} {'MinClear':>10}"
    print(header)
    print(f"{'=' * 90}")
    for r in all_results:
        line = (
            f"{r['name']:<18} "
            f"{r['rmse']:>8.4f} "
            f"{r['max_error']:>10.4f} "
            f"{r['mean_tube_width']:>10.4f} "
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

    # (0,2) Tube 폭
    ax = axes[0, 2]
    for r in results:
        if r["tube_widths"]:
            t_tw = np.arange(len(r["tube_widths"])) * dt
            ax.plot(t_tw, r["tube_widths"], color=r["color"],
                    label=r["short"], linewidth=1.5)
    if not any(r["tube_widths"] for r in results):
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", fontsize=12, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tube Width (m)")
    ax.set_title("Tube Width (Tube/R-MPPI)")
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

    # (1,0) 피드백 크기
    ax = axes[1, 0]
    for r in results:
        if r["feedback_norms"]:
            t_fb = np.arange(len(r["feedback_norms"])) * dt
            ax.plot(t_fb, r["feedback_norms"], color=r["color"],
                    label=r["short"], linewidth=1.5)
    if not any(r["feedback_norms"] for r in results):
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", fontsize=12, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Feedback Norm")
    ax.set_title("Feedback Magnitude (R-MPPI)")
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

    # (1,2) MaxError 바 차트
    ax = axes[1, 2]
    max_errors = [r["max_error"] for r in results]
    bars_me = ax.bar(names, max_errors, color=colors, alpha=0.8)
    for bar, val in zip(bars_me, max_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Max Error (m)")
    ax.set_title("Max Tracking Error")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) MeanTubeWidth 바 차트
    ax = axes[1, 3]
    tube_ws = [r["mean_tube_width"] for r in results]
    bars_tw = ax.bar(names, tube_ws, color=colors, alpha=0.8)
    for bar, val in zip(bars_tw, tube_ws):
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Tube Width (m)")
    ax.set_title("Mean Tube Width")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"R-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs Tube vs Risk-Aware vs Robust MPPI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/robust_mppi_benchmark_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="R-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="mild_noise",
        choices=["mild_noise", "strong_noise", "model_mismatch", "obstacle_noise"],
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
