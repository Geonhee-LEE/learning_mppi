#!/usr/bin/env python3
"""
MPPI 전변형 종합 벤치마크

Scenario A: 경로 추종 (circle, 장애물 없음)
Scenario B: 장애물 회피 (circle + 3 obstacles)

비교 대상 (9종):
  1. Vanilla MPPI          — 기준선
  2. Kernel MPPI           — RBF 차원 축소
  3. DIAL-MPPI             — 반복 어닐링
  4. CBF-MPPI              — 소프트 안전 제약
  5. Shield-MPPI           — 하드 안전 제약
  6. C2U-MPPI              — UT 공분산 + 기회 제약
  7. Uncertainty-MPPI       — 적응 샘플링
  8. Flow-MPPI             — CFM 학습 샘플링
  9. Conformal CBF-MPPI    — CP 동적 마진

Usage:
    PYTHONPATH=. python examples/comparison/all_mppi_variants_benchmark.py
    PYTHONPATH=. python examples/comparison/all_mppi_variants_benchmark.py --scenario obstacles
    PYTHONPATH=. python examples/comparison/all_mppi_variants_benchmark.py --scenario all
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

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.kernel_mppi import KernelMPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.c2u_mppi import C2UMPPIController
from mppi_controller.controllers.mppi.uncertainty_mppi import UncertaintyMPPIController
from mppi_controller.controllers.mppi.flow_mppi import FlowMPPIController
from mppi_controller.controllers.mppi.conformal_cbf_mppi import ConformalCBFMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    KernelMPPIParams,
    DIALMPPIParams,
    CBFMPPIParams,
    ShieldMPPIParams,
    C2UMPPIParams,
    UncertaintyMPPIParams,
    FlowMPPIParams,
    ConformalCBFMPPIParams,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ── 상수 ──────────────────────────────────────────────────────

OBSTACLES = [(2.5, 2.0, 0.4), (-1.5, 3.0, 0.5), (1.0, -3.0, 0.3)]

N = 20
K = 512
DT = 0.05
SIGMA = np.array([0.5, 0.5])
Q = np.array([10.0, 10.0, 1.0])
R = np.array([0.1, 0.1])
DURATION = 10.0
NUM_STEPS = int(DURATION / DT)


# ── 컨트롤러 팩토리 ─────────────────────────────────────────

def make_cost(obstacles=None):
    costs = [StateTrackingCost(Q), TerminalCost(Q), ControlEffortCost(R)]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=500.0))
    return CompositeMPPICost(costs)


def build_controllers(obstacles=None):
    """9종 MPPI 컨트롤러 생성"""
    common = dict(N=N, dt=DT, K=K, lambda_=1.0, sigma=SIGMA, Q=Q, R=R)
    obs_list = obstacles or []

    controllers = {}

    # 1. Vanilla
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = MPPIParams(**common)
    controllers["Vanilla"] = (m, MPPIController(m, p, cost_function=make_cost(obstacles)))

    # 2. Kernel MPPI
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = KernelMPPIParams(**common, num_support_pts=8, kernel_bandwidth=2.0)
    controllers["Kernel"] = (m, KernelMPPIController(m, p, cost_function=make_cost(obstacles)))

    # 3. DIAL-MPPI
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = DIALMPPIParams(**common, n_diffuse_init=5, n_diffuse=3, traj_diffuse_factor=0.5)
    controllers["DIAL"] = (m, DIALMPPIController(m, p, cost_function=make_cost(obstacles)))

    # 4. CBF-MPPI
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = CBFMPPIParams(**common, cbf_obstacles=obs_list, cbf_weight=1000.0,
                       cbf_alpha=0.3, cbf_safety_margin=0.1)
    controllers["CBF"] = (m, CBFMPPIController(m, p, cost_function=make_cost(obstacles)))

    # 5. Shield-MPPI
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = ShieldMPPIParams(**common, cbf_obstacles=obs_list, cbf_weight=1000.0,
                          cbf_alpha=0.3, cbf_safety_margin=0.1, shield_enabled=True)
    controllers["Shield"] = (m, ShieldMPPIController(m, p, cost_function=make_cost(obstacles)))

    # 6. C2U-MPPI
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = C2UMPPIParams(**common, cc_obstacles=obs_list, chance_alpha=0.05,
                       chance_cost_weight=500.0, process_noise_scale=0.01)
    controllers["C2U"] = (m, C2UMPPIController(m, p, cost_function=make_cost(obstacles)))

    # 7. Uncertainty-MPPI
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = UncertaintyMPPIParams(**common, uncertainty_strategy="previous_trajectory",
                               exploration_factor=1.0)
    controllers["Unc-MPPI"] = (m, UncertaintyMPPIController(m, p, cost_function=make_cost(obstacles)))

    # 8. Flow-MPPI (Gaussian fallback — 학습 없이)
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = FlowMPPIParams(**common, flow_hidden_dims=[128, 128],
                        flow_mode="blend", flow_blend_ratio=0.5)
    controllers["Flow"] = (m, FlowMPPIController(m, p, cost_function=make_cost(obstacles)))

    # 9. Conformal CBF-MPPI
    m = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    p = ConformalCBFMPPIParams(**common, cbf_obstacles=obs_list, cbf_weight=1000.0,
                                cbf_alpha=0.3, cbf_safety_margin=0.1,
                                cp_alpha=0.1, cp_enabled=True)
    controllers["Conformal"] = (m, ConformalCBFMPPIController(m, p, cost_function=make_cost(obstacles)))

    return controllers


# ── 시뮬레이션 ────────────────────────────────────────────────

def run_simulation(name, model, ctrl, traj_fn, obstacles=None):
    """단일 컨트롤러 시뮬레이션"""
    init_pos = traj_fn(0.0)
    state = init_pos.copy()

    xy_list, err_list, ctrl_list = [], [], []
    ess_list, cost_list, compute_times, clearance_list = [], [], [], []

    for step in range(NUM_STEPS):
        t = step * DT
        ref = generate_reference_trajectory(traj_fn, t, N, DT)

        t0 = time.perf_counter()
        control, info = ctrl.compute_control(state, ref)
        compute_times.append(time.perf_counter() - t0)

        state = model.step(state, control, DT)

        xy_list.append(state[:2].copy())
        err_list.append(np.linalg.norm(state[:2] - ref[0, :2]))
        ctrl_list.append(control.copy())
        ess_list.append(info.get("ess", 0.0))
        cost_list.append(info.get("best_cost", 0.0))

        if obstacles:
            clearances = [
                np.sqrt((state[0] - ox)**2 + (state[1] - oy)**2) - r
                for ox, oy, r in obstacles
            ]
            clearance_list.append(min(clearances))

    controls = np.array(ctrl_list)
    jerk = np.mean(np.abs(np.diff(controls, axis=0))) if len(controls) > 1 else 0.0

    return {
        "xy": np.array(xy_list),
        "errors": np.array(err_list),
        "controls": controls,
        "ess": np.array(ess_list),
        "costs": np.array(cost_list),
        "compute_times": np.array(compute_times),
        "clearance": np.array(clearance_list) if clearance_list else None,
        "rmse": np.sqrt(np.mean(np.array(err_list) ** 2)),
        "jerk": jerk,
        "mean_ess": np.mean(ess_list),
        "mean_time_ms": np.mean(compute_times) * 1000,
        "min_clearance": min(clearance_list) if clearance_list else None,
        "collisions": sum(1 for c in clearance_list if c < 0) if clearance_list else 0,
    }


def run_scenario(scenario_name, obstacles=None):
    """하나의 시나리오 실행"""
    traj_fn = create_trajectory_function("circle", radius=3.0)
    controllers = build_controllers(obstacles)

    tag = "Obstacles" if obstacles else "Tracking"
    print(f"\n{'='*80}")
    print(f"  Scenario: {scenario_name} ({tag})")
    print(f"  N={N}, K={K}, dt={DT}, duration={DURATION}s")
    if obstacles:
        print(f"  Obstacles: {len(obstacles)}")
    print(f"{'='*80}\n")

    results = {}
    for name, (model, ctrl) in controllers.items():
        print(f"  Running {name:12s} ...", end="", flush=True)
        try:
            data = run_simulation(name, model, ctrl, traj_fn, obstacles)
            results[name] = data
            print(f" RMSE={data['rmse']:.4f}  Jerk={data['jerk']:.4f}  "
                  f"ESS={data['mean_ess']:.1f}  {data['mean_time_ms']:.1f}ms", end="")
            if data['min_clearance'] is not None:
                print(f"  MinClr={data['min_clearance']:.3f}", end="")
            print()
        except Exception as e:
            print(f" ERROR: {e}")

    return results


# ── 결과 출력 ─────────────────────────────────────────────────

def print_results_table(results, scenario_name, obstacles=None):
    """결과 테이블 출력"""
    has_obs = obstacles is not None

    print(f"\n{'='*90}")
    print(f"  Results: {scenario_name}")
    print(f"{'='*90}")

    header = f"  {'Method':<14} {'RMSE(m)':>8} {'Jerk':>8} {'ESS':>7} {'ms/step':>8}"
    if has_obs:
        header += f" {'MinClr':>8} {'Collis':>7}"
    print(header)
    print(f"  {'-'*76}")

    # RMSE 기준 정렬
    sorted_names = sorted(results.keys(), key=lambda n: results[n]["rmse"])

    for name in sorted_names:
        d = results[name]
        line = (f"  {name:<14} {d['rmse']:>8.4f} {d['jerk']:>8.4f} "
                f"{d['mean_ess']:>7.1f} {d['mean_time_ms']:>7.1f}ms")
        if has_obs:
            mc = d['min_clearance'] if d['min_clearance'] is not None else float('nan')
            line += f" {mc:>8.3f} {d['collisions']:>7d}"
        print(line)

    # Vanilla 대비 비교
    if "Vanilla" in results:
        v = results["Vanilla"]
        print(f"\n  --- vs Vanilla ---")
        for name in sorted_names:
            if name == "Vanilla":
                continue
            d = results[name]
            rmse_diff = (d['rmse'] - v['rmse']) / v['rmse'] * 100
            jerk_diff = (d['jerk'] - v['jerk']) / v['jerk'] * 100
            speed_ratio = d['mean_time_ms'] / v['mean_time_ms']
            sign_r = "+" if rmse_diff > 0 else ""
            sign_j = "+" if jerk_diff > 0 else ""
            print(f"  {name:<14} RMSE {sign_r}{rmse_diff:>5.1f}%  "
                  f"Jerk {sign_j}{jerk_diff:>5.1f}%  "
                  f"Speed {speed_ratio:.2f}x")

    print(f"{'='*90}")


# ── 플롯 ─────────────────────────────────────────────────────

COLORS = {
    "Vanilla":   "#1f77b4",
    "Kernel":    "#d62728",
    "DIAL":      "#2ca02c",
    "CBF":       "#ff7f0e",
    "Shield":    "#9467bd",
    "C2U":       "#8c564b",
    "Unc-MPPI":  "#e377c2",
    "Flow":      "#7f7f7f",
    "Conformal": "#17becf",
}


def plot_comparison(results_tracking, results_obstacles):
    """2 시나리오 비교 플롯"""

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "MPPI Variants Comprehensive Benchmark\n"
        "DifferentialDrive | Circle Trajectory | N=20, K=512",
        fontsize=15, fontweight="bold",
    )
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    traj_fn = create_trajectory_function("circle", radius=3.0)
    ref_t = np.linspace(0, DURATION, 500)
    ref_pts = np.array([traj_fn(t) for t in ref_t])

    # ── Row 0: XY 궤적 ──
    # Col 0: Tracking
    ax = fig.add_subplot(gs[0, 0:2])
    ax.set_title("A. Path Tracking (No Obstacles)", fontsize=12, fontweight="bold")
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")
    for name, d in results_tracking.items():
        ax.plot(d["xy"][:, 0], d["xy"][:, 1], color=COLORS.get(name, "gray"),
                linewidth=1.5, alpha=0.8, label=name)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper left")

    # Col 1: Obstacles
    ax = fig.add_subplot(gs[0, 2:4])
    ax.set_title("B. Obstacle Avoidance", fontsize=12, fontweight="bold")
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")
    for ox, oy, r in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))
        ax.add_patch(plt.Circle((ox, oy), r + 0.2, color="red", alpha=0.1,
                                 fill=False, linestyle="--"))
    for name, d in results_obstacles.items():
        ax.plot(d["xy"][:, 0], d["xy"][:, 1], color=COLORS.get(name, "gray"),
                linewidth=1.5, alpha=0.8, label=name)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper left")

    # ── Row 1: 바 차트 ──
    names_t = sorted(results_tracking.keys(), key=lambda n: results_tracking[n]["rmse"])
    names_o = sorted(results_obstacles.keys(), key=lambda n: results_obstacles[n]["rmse"])

    # RMSE - Tracking
    ax = fig.add_subplot(gs[1, 0])
    rmses = [results_tracking[n]["rmse"] for n in names_t]
    colors = [COLORS.get(n, "gray") for n in names_t]
    bars = ax.bar(range(len(names_t)), rmses, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names_t)))
    ax.set_xticklabels(names_t, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("A. Tracking RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, r in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{r:.3f}", ha="center", va="bottom", fontsize=6)

    # RMSE - Obstacles
    ax = fig.add_subplot(gs[1, 1])
    rmses = [results_obstacles[n]["rmse"] for n in names_o]
    colors = [COLORS.get(n, "gray") for n in names_o]
    bars = ax.bar(range(len(names_o)), rmses, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names_o)))
    ax.set_xticklabels(names_o, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("B. Obstacle RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, r in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{r:.3f}", ha="center", va="bottom", fontsize=6)

    # Jerk - Tracking
    ax = fig.add_subplot(gs[1, 2])
    jerks = [results_tracking[n]["jerk"] for n in names_t]
    colors = [COLORS.get(n, "gray") for n in names_t]
    bars = ax.bar(range(len(names_t)), jerks, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names_t)))
    ax.set_xticklabels(names_t, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean |delta_u|")
    ax.set_title("A. Smoothness (lower=better)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, j in zip(bars, jerks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{j:.3f}", ha="center", va="bottom", fontsize=6)

    # Compute Time
    ax = fig.add_subplot(gs[1, 3])
    times_t = [results_tracking[n]["mean_time_ms"] for n in names_t]
    colors = [COLORS.get(n, "gray") for n in names_t]
    bars = ax.bar(range(len(names_t)), times_t, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names_t)))
    ax.set_xticklabels(names_t, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ms/step")
    ax.set_title("Compute Time")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, t in zip(bars, times_t):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{t:.1f}", ha="center", va="bottom", fontsize=6)

    # ── Row 2: 시계열 + 안전 메트릭 ──
    # Error time series - Tracking
    ax = fig.add_subplot(gs[2, 0:2])
    ax.set_title("Position Error over Time")
    time_axis = np.arange(NUM_STEPS) * DT
    for name in ["Vanilla", "Kernel", "DIAL", "Flow"]:
        if name in results_tracking:
            ax.plot(time_axis, results_tracking[name]["errors"],
                    color=COLORS.get(name, "gray"), linewidth=1.2,
                    alpha=0.8, label=f"{name} (tracking)")
    for name in ["Vanilla", "Shield", "CBF", "C2U"]:
        if name in results_obstacles:
            ax.plot(time_axis, results_obstacles[name]["errors"],
                    color=COLORS.get(name, "gray"), linewidth=1.2,
                    alpha=0.5, linestyle="--", label=f"{name} (obs)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=4)

    # Safety metrics bar (obstacle scenario)
    ax = fig.add_subplot(gs[2, 2])
    safe_names = [n for n in names_o if results_obstacles[n]["min_clearance"] is not None]
    min_clrs = [results_obstacles[n]["min_clearance"] for n in safe_names]
    colors = [COLORS.get(n, "gray") for n in safe_names]
    bars = ax.bar(range(len(safe_names)), min_clrs, color=colors, alpha=0.7)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Collision")
    ax.set_xticks(range(len(safe_names)))
    ax.set_xticklabels(safe_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("B. Safety (min obstacle dist)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    for bar, c in zip(bars, min_clrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{c:.3f}", ha="center", va="bottom", fontsize=6)

    # Summary text
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    lines = ["MPPI Variants Summary", "=" * 35, ""]

    # Tracking 1위
    best_t = min(results_tracking, key=lambda n: results_tracking[n]["rmse"])
    lines.append(f"Tracking best:  {best_t} ({results_tracking[best_t]['rmse']:.4f}m)")

    # Obstacle 1위
    best_o = min(results_obstacles, key=lambda n: results_obstacles[n]["rmse"])
    lines.append(f"Obstacle best:  {best_o} ({results_obstacles[best_o]['rmse']:.4f}m)")

    # Smoothest
    smooth = min(results_tracking, key=lambda n: results_tracking[n]["jerk"])
    lines.append(f"Smoothest:      {smooth} ({results_tracking[smooth]['jerk']:.4f})")

    # Fastest
    fast = min(results_tracking, key=lambda n: results_tracking[n]["mean_time_ms"])
    lines.append(f"Fastest:        {fast} ({results_tracking[fast]['mean_time_ms']:.1f}ms)")

    # Safest
    safe_results = {n: d for n, d in results_obstacles.items() if d["min_clearance"] is not None}
    if safe_results:
        safest = max(safe_results, key=lambda n: safe_results[n]["min_clearance"])
        lines.append(f"Safest:         {safest} ({safe_results[safest]['min_clearance']:.3f}m)")

    lines.extend(["", f"N={N}, K={K}, dt={DT}", f"Duration={DURATION}s"])

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    os.makedirs("plots", exist_ok=True)
    save_path = "plots/all_mppi_variants_benchmark.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MPPI All Variants Benchmark")
    parser.add_argument("--scenario", default="all",
                        choices=["tracking", "obstacles", "all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    results_tracking = None
    results_obstacles = None

    if args.scenario in ("tracking", "all"):
        results_tracking = run_scenario("A. Path Tracking", obstacles=None)
        print_results_table(results_tracking, "A. Path Tracking", obstacles=None)

    if args.scenario in ("obstacles", "all"):
        results_obstacles = run_scenario("B. Obstacle Avoidance", obstacles=OBSTACLES)
        print_results_table(results_obstacles, "B. Obstacle Avoidance", obstacles=OBSTACLES)

    if results_tracking and results_obstacles:
        plot_comparison(results_tracking, results_obstacles)


if __name__ == "__main__":
    main()
