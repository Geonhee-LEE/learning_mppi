#!/usr/bin/env python3
"""
Flow-MPPI (Conditional Flow Matching + MPPI) 벤치마크: 3-Way × 2 시나리오

방법:
  1. Vanilla MPPI          — 고정 가우시안 샘플링
  2. DIAL-MPPI             — 확산 어닐링 (multi-iter + noise decay)
  3. Flow-MPPI             — CFM 학습 기반 다중 모달 샘플링

시나리오:
  A. simple    — 직선 경로, 장애물 없음 (기준선)
  B. obstacles — 장애물 코스 (다중 모달 최적 경로)

측정:
  - 목표 도달 RMSE
  - 계산 시간
  - ESS (Effective Sample Size)
  - 총 비용

Usage:
    PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py
    PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --scenario obstacles
    PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --live
    PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --no-plot
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    FlowMPPIParams,
    DIALMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.flow_mppi import FlowMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import (
    circle_trajectory,
    generate_reference_trajectory,
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
            "obstacles": [],
        },
        "obstacles": {
            "name": "Obstacles (multi-modal)",
            "obstacles": OBSTACLES,
        },
    }


# ── 시뮬레이션 루프 (SimulationHarness 기반) ──────────────────

def run_simulation(model, controller, reference_fn, initial_state, dt, duration,
                   realtime=False):
    """SimulationHarness 기반 시뮬레이션 (호환 인터페이스)"""
    N_horizon = controller.params.N

    # reference_fn 래핑 (3인자 → 1인자)
    def ref_1arg(t):
        return reference_fn(t, N_horizon + 1, dt)

    harness = SimulationHarness(dt=dt, headless=True, seed=42)
    harness.add_controller("ctrl", controller, model)
    results = harness.run(ref_1arg, initial_state, duration)
    r = results["ctrl"]
    h = r["history"]

    # ESS / cost 추출 (info에서)
    ess_list = [info.get("ess", 0.0) for info in h.get("info", [])]
    costs = [info.get("best_cost", 0.0) for info in h.get("info", [])]

    return {
        "states": np.vstack([initial_state[None, :], h["state"]]),
        "controls": h["control"],
        "compute_times": h["solve_time"],
        "ess": np.array(ess_list),
        "costs": np.array(costs),
    }


def compute_rmse(states, reference_fn, dt):
    """위치 추적 RMSE"""
    errors = []
    for i, s in enumerate(states):
        t = i * dt
        ref = reference_fn(t, 2, dt)
        errors.append(np.linalg.norm(s[:2] - ref[0, :2]))
    return np.mean(errors)


# ── 벤치마크 실행 ──────────────────────────────────────────

COLORS = {
    "Vanilla MPPI": "#1f77b4",
    "DIAL-MPPI": "#ff7f0e",
    "Flow-MPPI": "#2ca02c",
}


def _setup_common(scenario_key):
    """시나리오별 공통 설정 반환"""
    scenarios = get_scenarios()
    scenario = scenarios[scenario_key]

    model = DifferentialDriveKinematic(wheelbase=0.5)
    dt = 0.05
    duration = 10.0
    initial_state = np.array([0.0, 0.0, 0.0])
    N, K = 30, 512

    Q = np.array([10.0, 10.0, 1.0])
    R = np.array([0.1, 0.1])
    sigma = np.array([0.8, 0.8])

    traj_fn = lambda t: circle_trajectory(t, radius=3.0)

    def reference_fn(t, n_points, dt_):
        return generate_reference_trajectory(traj_fn, t, n_points - 1, dt_)

    obs = scenario["obstacles"]

    def make_cost(obstacles):
        costs = [StateTrackingCost(Q), TerminalCost(Q), ControlEffortCost(R)]
        if obstacles:
            costs.append(ObstacleCost(obstacles, cost_weight=2000.0, safety_margin=0.3))
        return CompositeMPPICost(costs)

    # 컨트롤러 생성
    vanilla_params = MPPIParams(N=N, K=K, dt=dt, sigma=sigma, Q=Q, R=R)
    vanilla_ctrl = MPPIController(model, vanilla_params, make_cost(obs))

    dial_sigma = np.array([0.5, 0.5]) if obs else sigma
    dial_params = DIALMPPIParams(
        N=N, K=K, dt=dt, sigma=dial_sigma, Q=Q, R=R,
        lambda_=0.5,
        n_diffuse_init=10, n_diffuse=3,
        traj_diffuse_factor=0.7,
        sigma_scale=0.8,
        use_reward_normalization=False,
    )
    dial_ctrl = DIALMPPIController(model, dial_params, make_cost(obs))

    flow_params = FlowMPPIParams(
        N=N, K=K, dt=dt, sigma=sigma, Q=Q, R=R,
        flow_hidden_dims=[128, 128],
        flow_num_steps=5,
        flow_mode="blend",
        flow_blend_ratio=0.5,
        flow_min_samples=20,
    )
    flow_ctrl = FlowMPPIController(model, flow_params, make_cost(obs))

    # Flow bootstrap
    bootstrap_state = initial_state.copy()
    t_boot = 0.0
    for _ in range(50):
        ref = reference_fn(t_boot, N + 1, dt)
        control, _ = flow_ctrl.compute_control(bootstrap_state, ref)
        state_dot = model.forward_dynamics(bootstrap_state, control)
        bootstrap_state = bootstrap_state + state_dot * dt
        t_boot += dt
    flow_ctrl.train_flow_model(epochs=50)
    flow_ctrl.reset()

    controllers = {
        "Vanilla MPPI": vanilla_ctrl,
        "DIAL-MPPI": dial_ctrl,
        "Flow-MPPI": flow_ctrl,
    }

    return {
        "model": model, "dt": dt, "duration": duration,
        "initial_state": initial_state, "N": N,
        "reference_fn": reference_fn, "obs": obs,
        "scenario": scenario, "controllers": controllers,
    }


# ── Live 애니메이션 ──────────────────────────────────────────

def run_live(scenario_key):
    """실시간 3-way 비교 애니메이션 (FuncAnimation)"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    cfg = _setup_common(scenario_key)
    model = cfg["model"]
    dt = cfg["dt"]
    duration = cfg["duration"]
    reference_fn = cfg["reference_fn"]
    obs = cfg["obs"]
    N = cfg["N"]
    num_steps = int(duration / dt)

    # 시뮬레이션 상태
    states = {k: cfg["initial_state"].copy() for k in cfg["controllers"]}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [], "costs": []}
        for k in cfg["controllers"]
    }

    # ===== Figure 설정 (2x2 4패널) =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Flow-MPPI Live — {cfg['scenario']['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY 궤적
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")
    # 장애물
    for ox, oy, r in obs:
        ax_xy.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))
        ax_xy.plot(ox, oy, "rx", markersize=8)
    # 레퍼런스
    ref_t = np.linspace(0, duration, 500)
    ref_pts = np.array([circle_trajectory(t, radius=3.0) for t in ref_t])
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
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=2, label=name)
    ax_err.legend(fontsize=7)

    # [1,0] ESS
    ax_ess = axes[1, 0]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=2, label=name)
    ax_ess.legend(fontsize=7)

    # [1,1] 비용
    ax_cost = axes[1, 1]
    ax_cost.set_xlabel("Time (s)")
    ax_cost.set_ylabel("Cost")
    ax_cost.set_title("Step Cost")
    ax_cost.grid(True, alpha=0.3)
    lines_cost = {}
    for name, color in COLORS.items():
        lines_cost[name], = ax_cost.plot([], [], color=color, linewidth=2, label=name)
    ax_cost.legend(fontsize=7)

    # 상태 텍스트
    time_text = ax_xy.text(
        0.5, -0.12, "", transform=ax_xy.transAxes,
        ha="center", fontsize=9, family="monospace",
    )

    plt.tight_layout()

    all_artists = (
        list(lines_xy.values()) + list(dots.values())
        + list(lines_err.values()) + list(lines_ess.values())
        + list(lines_cost.values()) + [time_text]
    )

    def init():
        for a in all_artists:
            if hasattr(a, "set_data"):
                a.set_data([], [])
        time_text.set_text("")
        return all_artists

    def update(frame):
        if frame >= num_steps:
            return all_artists

        t = sim_t[0]

        for name, ctrl in cfg["controllers"].items():
            ref = reference_fn(t, N + 1, dt)
            control, info = ctrl.compute_control(states[name], ref)

            state_dot = model.forward_dynamics(states[name], control)
            states[name] = states[name] + state_dot * dt

            ref_pt = ref[0, :2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))
            data[name]["costs"].append(info.get("best_cost", 0.0))

        sim_t[0] += dt

        # 그래프 업데이트
        times = np.array(data["Vanilla MPPI"]["times"])

        for name in cfg["controllers"]:
            xy = np.array(data[name]["xy"])
            lines_xy[name].set_data(xy[:, 0], xy[:, 1])
            dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
            lines_err[name].set_data(times, data[name]["errors"])
            lines_ess[name].set_data(times, data[name]["ess"])
            lines_cost[name].set_data(times, data[name]["costs"])

        ax_err.relim(); ax_err.autoscale_view()
        ax_ess.relim(); ax_ess.autoscale_view()
        ax_cost.relim(); ax_cost.autoscale_view()

        # RMSE 텍스트
        rmses = []
        for name in cfg["controllers"]:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs)**2)) if errs else 0
            rmses.append(f"{name}:{rmse:.3f}m")
        time_text.set_text(f"t={sim_t[0]:.1f}s | RMSE  " + "  ".join(rmses))

        return all_artists

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)

    # GIF 저장
    gif_path = f"plots/flow_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    # MP4 저장 (ffmpeg 있는 경우)
    try:
        mp4_path = f"plots/flow_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # 종료 후 통계
    print(f"\n{'='*60}")
    print(f"  Final Statistics — {cfg['scenario']['name']}")
    print(f"{'='*60}")
    print(f"\n  {'Method':<20} {'RMSE':>8} {'Mean Cost':>10} {'Mean ESS':>10}")
    print(f"  {'-'*50}")
    for name in cfg["controllers"]:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs)**2)) if errs else 0
        mean_cost = np.mean(data[name]["costs"]) if data[name]["costs"] else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        print(f"  {name:<20} {rmse:>8.4f} {mean_cost:>10.2f} {mean_ess:>10.1f}")
    print(f"{'='*60}\n")


# ── 벤치마크 실행 (기존 배치 모드) ─────────────────────────────

def run_benchmark(scenario_key, show_plot=True, live=False):
    if live:
        run_live(scenario_key)
        return

    cfg = _setup_common(scenario_key)
    model = cfg["model"]
    dt = cfg["dt"]
    duration = cfg["duration"]
    reference_fn = cfg["reference_fn"]
    obs = cfg["obs"]
    scenario = cfg["scenario"]

    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario['name']}")
    print(f"{'='*60}")

    results = {}
    for i, (name, ctrl) in enumerate(cfg["controllers"].items(), 1):
        total = len(cfg["controllers"])
        print(f"\n  [{i}/{total}] {name} ...")
        results[name] = run_simulation(
            model, ctrl, reference_fn, cfg["initial_state"], dt, duration,
        )

    print(f"\n  {'Method':<20} {'RMSE':>8} {'Mean Cost':>10} {'Mean ESS':>10} {'Time/step':>12}")
    print(f"  {'-'*60}")
    for name, res in results.items():
        rmse = compute_rmse(res["states"], reference_fn, dt)
        mean_cost = np.mean(res["costs"])
        mean_ess = np.mean(res["ess"])
        mean_time = np.mean(res["compute_times"]) * 1000
        print(f"  {name:<20} {rmse:>8.4f} {mean_cost:>10.2f} {mean_ess:>10.1f} {mean_time:>10.1f}ms")

    # ── Plot ──
    if show_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # XY trajectory
            ax = axes[0]
            for name, res in results.items():
                ax.plot(res["states"][:, 0], res["states"][:, 1], label=name)
            # Reference
            t_ref = np.linspace(0, duration, 200)
            x_ref = 3.0 * np.cos(t_ref)
            y_ref = 3.0 * np.sin(t_ref)
            ax.plot(x_ref, y_ref, "k--", alpha=0.3, label="Reference")
            # Obstacles
            for ox, oy, r in obs:
                circle = plt.Circle((ox, oy), r, color="red", alpha=0.3)
                ax.add_patch(circle)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("XY Trajectory")
            ax.legend(fontsize=8)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # ESS
            ax = axes[1]
            for name, res in results.items():
                ax.plot(res["ess"], label=name, alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_ylabel("ESS")
            ax.set_title("Effective Sample Size")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Compute time
            ax = axes[2]
            for name, res in results.items():
                ax.plot(res["compute_times"] * 1000, label=name, alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_ylabel("Time (ms)")
            ax.set_title("Compute Time")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            os.makedirs("plots", exist_ok=True)
            plt.savefig(f"plots/flow_mppi_benchmark_{scenario_key}.png", dpi=150)
            print(f"\n  Plot saved: plots/flow_mppi_benchmark_{scenario_key}.png")
            plt.close()
        except ImportError:
            print("  (matplotlib not available, skipping plot)")


def main():
    parser = argparse.ArgumentParser(description="Flow-MPPI Benchmark")
    parser.add_argument("--scenario", type=str, default="simple",
                        choices=["simple", "obstacles"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--live", action="store_true", help="Realtime simulation")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    show_plot = not args.no_plot

    if args.all_scenarios:
        for key in get_scenarios():
            run_benchmark(key, show_plot=show_plot, live=args.live)
    else:
        run_benchmark(args.scenario, show_plot=show_plot, live=args.live)

    print("\nDone.")


if __name__ == "__main__":
    main()
