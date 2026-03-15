#!/usr/bin/env python3
"""
BNN-MPPI (Bayesian Neural Network Surrogate MPPI) 벤치마크: 3-Way × 3 시나리오

방법:
  1. Vanilla MPPI          — 기본 (불확실성 무시)
  2. Uncertainty-Aware MPPI — 불확실성 기반 적응 샘플링
  3. BNN-MPPI              — 불확실성 기반 feasibility 필터링 + 비용

시나리오:
  A. clean    — 외란 없음 (기준선)
  B. noisy    — 프로세스 노이즈 추가: BNN 보수적 제어 우위
  C. obstacle — 장애물 회피: BNN 안전 영역 선호

측정:
  - 위치 추적 RMSE
  - 계산 시간 (min/max/mean)
  - Feasibility 통계 (BNN-MPPI only)
  - 충돌 수 / 최소 클리어런스 (obstacle)
  - ESS (Effective Sample Size)

Usage:
    PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py
    PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py --scenario noisy
    PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py --scenario obstacle
    PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py --trajectory figure8
    PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py --no-plot
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
    UncertaintyMPPIParams,
    BNNMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.uncertainty_mppi import (
    UncertaintyMPPIController,
)
from mppi_controller.controllers.mppi.bnn_mppi import BNNMPPIController
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


# ── 불확실성 모델 (거리 기반) ─────────────────────────────────

class DistanceBasedUncertainty:
    """원점에서 멀수록 불확실성 증가 (학습 없이 시뮬레이션)"""

    def __init__(self, base_std=0.02, distance_scale=0.03):
        self.base_std = base_std
        self.distance_scale = distance_scale

    def __call__(self, states, controls):
        if states.ndim == 1:
            states = states[None, :]
        dist = np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)
        nx = states.shape[-1]
        std = self.base_std + self.distance_scale * dist[:, None]
        return np.broadcast_to(std, (states.shape[0], nx)).copy()


# ── 시나리오 설정 ────────────────────────────────────────────

OBSTACLES = [
    (4.0, 2.5, 0.4),
    (-2.0, 3.5, 0.5),
    (1.0, -4.5, 0.3),
]


def get_scenarios():
    return {
        "clean": {
            "name": "Clean (no disturbance)",
            "process_noise_std": None,
            "unc_model": DistanceBasedUncertainty(base_std=0.02, distance_scale=0.03),
            "obstacles": None,
        },
        "noisy": {
            "name": "Noisy (process noise)",
            "process_noise_std": np.array([0.04, 0.04, 0.01]),
            "unc_model": DistanceBasedUncertainty(base_std=0.06, distance_scale=0.03),
            "obstacles": None,
        },
        "obstacle": {
            "name": "Obstacle avoidance",
            "process_noise_std": None,
            "unc_model": DistanceBasedUncertainty(base_std=0.03, distance_scale=0.03),
            "obstacles": OBSTACLES,
        },
    }


def create_trajectory_fn(name):
    if name == "circle":
        return circle_trajectory
    elif name == "figure8":
        return figure_eight_trajectory
    return circle_trajectory


# ── 시뮬레이션 실행 (SimulationHarness 기반) ──────────────────

def run_simulation(model, controller, reference_fn, initial_state, dt, duration,
                   process_noise_std=None):
    """SimulationHarness 기반 시뮬레이션"""
    harness = SimulationHarness(dt=dt, headless=True, seed=42)
    harness.add_controller("ctrl", controller, model,
                           process_noise_std=process_noise_std)
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
    """궤적 추적 RMSE"""
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    return np.sqrt(np.mean(np.array(errors) ** 2))


def compute_obstacle_metrics(states, obstacles):
    """장애물 관련 메트릭 계산"""
    if not obstacles:
        return {"n_collisions": 0, "min_clearance": float("inf"), "mean_min_clearance": 0}
    n_collisions = 0
    min_dist = float("inf")
    min_distances = []

    for st in states:
        x, y = st[0], st[1]
        for ox, oy, r in obstacles:
            dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            clearance = dist - r
            min_dist = min(min_dist, clearance)
            if clearance < 0:
                n_collisions += 1
        closest = min(np.sqrt((x - ox) ** 2 + (y - oy) ** 2) - r for ox, oy, r in obstacles)
        min_distances.append(closest)

    return {
        "n_collisions": n_collisions,
        "min_clearance": min_dist,
        "mean_min_clearance": float(np.mean(min_distances)),
    }


# ── 벤치마크 메인 ─────────────────────────────────────────────

def run_benchmark(args):
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    has_obstacles = scenario["obstacles"] is not None

    trajectory_fn = create_trajectory_fn(args.trajectory)

    print(f"\n{'=' * 72}")
    print(f"  BNN-MPPI Benchmark: 3-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  Trajectory: {args.trajectory} | Duration: {args.duration}s | Seed: {args.seed}")
    print(f"{'=' * 72}")

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    initial_state = trajectory_fn(0.0)
    unc_model = scenario["unc_model"]

    common = dict(
        K=256, N=20, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )

    # ── 3가지 방법 ──

    def make_vanilla():
        params = MPPIParams(**common)
        if has_obstacles:
            cost = CompositeMPPICost([
                StateTrackingCost(params.Q),
                TerminalCost(params.Qf),
                ControlEffortCost(params.R),
                ObstacleCost(scenario["obstacles"], safety_margin=0.1, cost_weight=200.0),
            ])
            return MPPIController(model, params, cost_function=cost)
        return MPPIController(model, params)

    def make_uncertainty():
        params = UncertaintyMPPIParams(
            **common,
            exploration_factor=1.5,
            uncertainty_strategy="previous_trajectory",
        )
        if has_obstacles:
            cost = CompositeMPPICost([
                StateTrackingCost(params.Q),
                TerminalCost(params.Qf),
                ControlEffortCost(params.R),
                ObstacleCost(scenario["obstacles"], safety_margin=0.2, cost_weight=200.0),
            ])
            return UncertaintyMPPIController(
                model, params, cost_function=cost, uncertainty_fn=unc_model,
            )
        return UncertaintyMPPIController(
            model, params, uncertainty_fn=unc_model,
        )

    def make_bnn():
        params = BNNMPPIParams(
            **common,
            feasibility_weight=50.0,
            uncertainty_reduce="sum",
            feasibility_threshold=0.3,
            max_filter_ratio=0.5,
        )
        if has_obstacles:
            cost = CompositeMPPICost([
                StateTrackingCost(params.Q),
                TerminalCost(params.Qf),
                ControlEffortCost(params.R),
                ObstacleCost(scenario["obstacles"], safety_margin=0.15, cost_weight=200.0),
            ])
            return BNNMPPIController(
                model, params, cost_function=cost, uncertainty_fn=unc_model,
            )
        return BNNMPPIController(
            model, params, uncertainty_fn=unc_model,
        )

    variants = [
        {"name": "Vanilla MPPI",      "short": "Vanilla",  "make": make_vanilla,      "color": "#2196F3"},
        {"name": "Uncertainty MPPI",   "short": "UncMPPI",  "make": make_uncertainty,  "color": "#FF9800"},
        {"name": "BNN-MPPI",          "short": "BNN",      "make": make_bnn,          "color": "#4CAF50"},
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
            process_noise_std=scenario["process_noise_std"],
        )
        elapsed = time.time() - t_start

        rmse = compute_tracking_rmse(
            history["states"], trajectory_fn, common["dt"],
        )
        solve_times = history["solve_times"]
        mean_solve = np.mean(solve_times) * 1000
        max_solve = np.max(solve_times) * 1000
        min_solve = np.min(solve_times) * 1000

        # 장애물 메트릭
        obs_metrics = compute_obstacle_metrics(
            history["states"], scenario["obstacles"] if has_obstacles else None
        )

        # ESS 추출
        ess_list = []
        for info in history["infos"]:
            if isinstance(info, dict) and "ess" in info:
                ess_list.append(info["ess"])

        # BNN 고유 메트릭
        bnn_stats = {}
        if var["short"] == "BNN" and history["infos"]:
            feas_list = [
                info.get("bnn_stats", {}).get("mean_feasibility", 0)
                for info in history["infos"]
                if isinstance(info, dict)
            ]
            filter_list = [
                info.get("bnn_stats", {}).get("filter_ratio", 0)
                for info in history["infos"]
                if isinstance(info, dict)
            ]
            if feas_list:
                bnn_stats["mean_feasibility"] = float(np.mean(feas_list))
                bnn_stats["min_feasibility"] = float(np.min(feas_list))
            if filter_list:
                bnn_stats["mean_filter_ratio"] = float(np.mean(filter_list))

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
            "bnn_stats": bnn_stats,
            "obs_metrics": obs_metrics,
            "ess_list": ess_list,
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

    # BNN 고유 메트릭
    for r in all_results:
        if r["bnn_stats"]:
            stats = r["bnn_stats"]
            print(f"\n  BNN-MPPI 고유 메트릭:")
            print(f"    평균 feasibility: {stats.get('mean_feasibility', 0):.4f}")
            print(f"    최소 feasibility: {stats.get('min_feasibility', 0):.4f}")
            print(f"    평균 필터 비율:   {stats.get('mean_filter_ratio', 0):.4f}")

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

    # 1. XY 궤적
    ax = axes[0, 0]
    t_arr = np.arange(0, duration + dt, dt)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, label="Reference", linewidth=1)

    for r in results:
        states = r["states"]
        ax.plot(states[:, 0], states[:, 1], color=r["color"],
                label=r["short"], linewidth=1.5)

    # 장애물 표시
    if has_obstacles:
        for ox, oy, radius in scenario["obstacles"]:
            circle = Circle((ox, oy), radius, fill=True, facecolor="#FF5252",
                            edgecolor="red", alpha=0.3, linewidth=1.5)
            ax.add_patch(circle)
            # 안전 마진 표시
            margin_circle = Circle((ox, oy), radius + 0.2, fill=False,
                                   edgecolor="red", alpha=0.3, linestyle="--")
            ax.add_patch(margin_circle)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. 위치 오차
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

    # 3. Feasibility 점수 (BNN only)
    ax = axes[0, 2]
    bnn_result = next((r for r in results if r["short"] == "BNN"), None)
    if bnn_result and bnn_result["infos"]:
        feas_scores = []
        for info in bnn_result["infos"]:
            if isinstance(info, dict) and "bnn_stats" in info:
                feas_scores.append(info["bnn_stats"]["mean_feasibility"])
        if feas_scores:
            t_feas = np.arange(len(feas_scores)) * dt
            ax.plot(t_feas, feas_scores, color=bnn_result["color"], linewidth=1)
            ax.fill_between(t_feas, feas_scores, alpha=0.2, color=bnn_result["color"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Feasibility Score")
    ax.set_title("BNN-MPPI Feasibility")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 4. 필터 비율 (BNN only)
    ax = axes[1, 0]
    if bnn_result and bnn_result["infos"]:
        filter_ratios = []
        for info in bnn_result["infos"]:
            if isinstance(info, dict) and "bnn_stats" in info:
                filter_ratios.append(info["bnn_stats"]["filter_ratio"])
        if filter_ratios:
            t_filt = np.arange(len(filter_ratios)) * dt
            ax.plot(t_filt, filter_ratios, color=bnn_result["color"], linewidth=1)
            ax.fill_between(t_filt, filter_ratios, alpha=0.2, color=bnn_result["color"])
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="max_filter_ratio")
            ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Filter Ratio")
    ax.set_title("BNN-MPPI Filter Ratio")
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3)

    # 5. ESS
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

    # 6. 메트릭 바 차트
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

    plt.suptitle(f"BNN-MPPI Benchmark [{scenario_name}]: Vanilla vs Uncertainty vs BNN-MPPI",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/bnn_mppi_benchmark_{scenario_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BNN-MPPI Benchmark")
    parser.add_argument("--scenario", default="clean",
                        choices=["clean", "noisy", "obstacle"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--trajectory", default="circle",
                        choices=["circle", "figure8"])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.all_scenarios:
        for scenario_name in get_scenarios():
            args.scenario = scenario_name
            run_benchmark(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
