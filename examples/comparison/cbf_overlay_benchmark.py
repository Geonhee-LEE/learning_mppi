#!/usr/bin/env python3
"""
CBF 오버레이 평가 벤치마크

CBF/Shield는 기존 MPPI에 더해지는 안전 레이어.
각 base MPPI 변형에 CBF를 추가했을 때의 효과를 평가.

매트릭스:
  Base × Safety = {Vanilla, PI, Kernel, DIAL} × {None, +ObsCost, +CBF, +Shield*}
  *Shield는 Vanilla 기반만 가능 (rollout 오버라이드 때문)

비교 메트릭:
  - RMSE (경로 추종 정확도)
  - 최소 장애물 클리어런스
  - 충돌 횟수
  - ESS (샘플 효율)
  - 계산 시간

Usage:
    PYTHONPATH=. python examples/comparison/cbf_overlay_benchmark.py
    PYTHONPATH=. python examples/comparison/cbf_overlay_benchmark.py --base vanilla kernel
    PYTHONPATH=. python examples/comparison/cbf_overlay_benchmark.py --safety none cbf shield
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
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.kernel_mppi import KernelMPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.pi_mppi import PIMPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    KernelMPPIParams,
    DIALMPPIParams,
    CBFMPPIParams,
    ShieldMPPIParams,
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


# ── 비용 함수 팩토리 ─────────────────────────────────────────

def make_base_cost():
    """기본 비용: 추적 + 터미널 + 제어"""
    return CompositeMPPICost([
        StateTrackingCost(Q),
        TerminalCost(Q),
        ControlEffortCost(R),
    ])


def make_obstacle_cost():
    """기본 비용 + 장애물 비용 (soft penalty)"""
    return CompositeMPPICost([
        StateTrackingCost(Q),
        TerminalCost(Q),
        ControlEffortCost(R),
        ObstacleCost(OBSTACLES, safety_margin=0.2, cost_weight=500.0),
    ])


def make_cbf_cost():
    """기본 비용 + CBF 비용 (discrete-time CBF penalty)"""
    return CompositeMPPICost([
        StateTrackingCost(Q),
        TerminalCost(Q),
        ControlEffortCost(R),
        ControlBarrierCost(OBSTACLES, cbf_alpha=0.3, cbf_weight=1000.0,
                           safety_margin=0.1),
    ])


# ── 컨트롤러 팩토리 ─────────────────────────────────────────

def build_controller(base_name, safety_name):
    """
    (base × safety) 조합으로 컨트롤러 생성

    Args:
        base_name: "vanilla" | "pi" | "kernel" | "dial"
        safety_name: "none" | "obs_cost" | "cbf" | "shield"

    Returns:
        (model, controller, label)
    """
    common = dict(N=N, dt=DT, K=K, lambda_=1.0, sigma=SIGMA, Q=Q, R=R)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # Safety 비용 결정
    if safety_name == "none":
        cost = make_base_cost()
    elif safety_name == "obs_cost":
        cost = make_obstacle_cost()
    elif safety_name in ("cbf", "shield"):
        cost = make_cbf_cost()
    else:
        raise ValueError(f"Unknown safety: {safety_name}")

    # Shield는 Vanilla 기반만 지원
    if safety_name == "shield":
        if base_name != "vanilla":
            return None  # 지원하지 않는 조합
        params = ShieldMPPIParams(
            **common,
            cbf_obstacles=OBSTACLES,
            cbf_weight=1000.0,
            cbf_alpha=0.3,
            cbf_safety_margin=0.1,
            shield_enabled=True,
        )
        ctrl = ShieldMPPIController(model, params)
        return model, ctrl, f"Vanilla+Shield"

    # CBF-MPPI 래퍼 (Vanilla 전용, CBF cost + QP filter)
    if safety_name == "cbf" and base_name == "vanilla":
        params = CBFMPPIParams(
            **common,
            cbf_obstacles=OBSTACLES,
            cbf_weight=1000.0,
            cbf_alpha=0.3,
            cbf_safety_margin=0.1,
        )
        ctrl = CBFMPPIController(model, params)
        return model, ctrl, f"Vanilla+CBF"

    # Base 컨트롤러 생성
    if base_name == "vanilla":
        params = MPPIParams(**common)
        ctrl = MPPIController(model, params, cost_function=cost)
        label = "Vanilla"

    elif base_name == "pi":
        params = MPPIParams(**common)
        ctrl = PIMPPIController(model, params, cost_function=cost)
        label = "PI-MPPI"

    elif base_name == "kernel":
        params = KernelMPPIParams(
            **common, num_support_pts=8, kernel_bandwidth=2.0
        )
        ctrl = KernelMPPIController(model, params, cost_function=cost)
        label = "Kernel"

    elif base_name == "dial":
        params = DIALMPPIParams(
            **common, n_diffuse_init=5, n_diffuse=3,
            traj_diffuse_factor=0.5,
        )
        ctrl = DIALMPPIController(model, params, cost_function=cost)
        label = "DIAL"

    else:
        raise ValueError(f"Unknown base: {base_name}")

    # Safety suffix
    if safety_name == "obs_cost":
        label += "+ObsCost"
    elif safety_name == "cbf":
        label += "+CBF"

    return model, ctrl, label


# ── 시뮬레이션 ────────────────────────────────────────────────

def run_simulation(model, ctrl, traj_fn):
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

        clearances = [
            np.sqrt((state[0] - ox)**2 + (state[1] - oy)**2) - r
            for ox, oy, r in OBSTACLES
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
        "clearance": np.array(clearance_list),
        "rmse": np.sqrt(np.mean(np.array(err_list) ** 2)),
        "jerk": jerk,
        "mean_ess": np.mean(ess_list),
        "mean_time_ms": np.mean(compute_times) * 1000,
        "min_clearance": min(clearance_list),
        "collisions": sum(1 for c in clearance_list if c < 0),
    }


# ── 벤치마크 실행 ────────────────────────────────────────────

def run_benchmark(base_variants, safety_modes):
    """전체 매트릭스 벤치마크"""
    traj_fn = create_trajectory_function("circle", radius=3.0)

    print(f"\n{'='*90}")
    print(f"  CBF Overlay Benchmark")
    print(f"  N={N}, K={K}, dt={DT}, duration={DURATION}s")
    print(f"  Base variants: {base_variants}")
    print(f"  Safety modes:  {safety_modes}")
    print(f"  Obstacles: {len(OBSTACLES)}")
    print(f"{'='*90}\n")

    results = {}
    for base in base_variants:
        for safety in safety_modes:
            result = build_controller(base, safety)
            if result is None:
                print(f"  {base:8s} × {safety:10s}  SKIP (unsupported)")
                continue

            model, ctrl, label = result
            print(f"  Running {label:22s} ...", end="", flush=True)
            try:
                data = run_simulation(model, ctrl, traj_fn)
                results[label] = data
                print(f" RMSE={data['rmse']:.4f}  MinClr={data['min_clearance']:.3f}  "
                      f"Collis={data['collisions']:3d}  "
                      f"ESS={data['mean_ess']:.1f}  {data['mean_time_ms']:.1f}ms")
            except Exception as e:
                print(f" ERROR: {e}")

    return results


# ── 결과 출력 ─────────────────────────────────────────────────

def print_results_table(results):
    """결과 테이블"""
    print(f"\n{'='*100}")
    print(f"  Results: CBF Overlay Benchmark")
    print(f"{'='*100}")

    header = (f"  {'Method':<22} {'RMSE(m)':>8} {'MinClr(m)':>10} {'Collis':>7} "
              f"{'Jerk':>8} {'ESS':>7} {'ms/step':>8}")
    print(header)
    print(f"  {'-'*88}")

    sorted_names = sorted(results.keys(), key=lambda n: results[n]["rmse"])
    for name in sorted_names:
        d = results[name]
        print(f"  {name:<22} {d['rmse']:>8.4f} {d['min_clearance']:>10.3f} "
              f"{d['collisions']:>7d} {d['jerk']:>8.4f} "
              f"{d['mean_ess']:>7.1f} {d['mean_time_ms']:>7.1f}ms")

    # Base별 CBF 효과 분석
    print(f"\n  === CBF/Safety 효과 분석 ===")
    bases = set()
    for name in results:
        base = name.split("+")[0]
        bases.add(base)

    for base in sorted(bases):
        base_only = results.get(base)
        if base_only is None:
            continue

        print(f"\n  [{base}]")
        for name in sorted_names:
            if not name.startswith(base):
                continue
            d = results[name]
            suffix = name.replace(base, "").lstrip("+") or "(none)"
            rmse_change = (d['rmse'] - base_only['rmse']) / base_only['rmse'] * 100 if base_only['rmse'] > 0 else 0
            collision_change = d['collisions'] - base_only['collisions']
            clr_change = d['min_clearance'] - base_only['min_clearance']
            print(f"    {suffix:12s}: RMSE {rmse_change:+.1f}%  "
                  f"Collisions {collision_change:+d}  "
                  f"MinClr {clr_change:+.3f}m  "
                  f"ESS {d['mean_ess']:.1f}")


# ── 시각화 ────────────────────────────────────────────────────

def plot_results(results, save_path="plots/cbf_overlay_benchmark.png"):
    """결과 시각화"""
    traj_fn = create_trajectory_function("circle", radius=3.0)
    t_ref = np.linspace(0, 2 * np.pi, 200)
    ref_xy = np.column_stack([3.0 * np.cos(t_ref), 3.0 * np.sin(t_ref)])

    n_results = len(results)
    if n_results == 0:
        return

    # 그룹별 색상
    base_colors = {
        "Vanilla": "#1f77b4",
        "PI-MPPI": "#ff7f0e",
        "Kernel": "#2ca02c",
        "DIAL": "#d62728",
    }
    safety_styles = {
        "": "-",         # no safety
        "ObsCost": "--",  # obstacle cost
        "CBF": "-.",      # CBF
        "Shield": ":",    # Shield
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── 1. XY 궤적 ──
    ax = axes[0, 0]
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.3, label="Reference")
    for ox, oy, r in OBSTACLES:
        circle = Circle((ox, oy), r, fill=True, color="red", alpha=0.3)
        ax.add_patch(circle)
        circle_margin = Circle((ox, oy), r + 0.2, fill=False,
                                edgecolor="red", linestyle="--", alpha=0.3)
        ax.add_patch(circle_margin)

    for name, data in results.items():
        base = name.split("+")[0]
        safety = name.replace(base, "").lstrip("+")
        color = base_colors.get(base, "gray")
        style = safety_styles.get(safety, "-")
        ax.plot(data["xy"][:, 0], data["xy"][:, 1],
                color=color, linestyle=style, label=name, alpha=0.8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── 2. RMSE 비교 (그룹별 바 차트) ──
    ax = axes[0, 1]
    names = sorted(results.keys())
    rmses = [results[n]["rmse"] for n in names]
    colors_bar = []
    for n in names:
        base = n.split("+")[0]
        colors_bar.append(base_colors.get(base, "gray"))
    bars = ax.barh(range(len(names)), rmses, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("RMSE (m)")
    ax.set_title("Tracking RMSE")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 3. 최소 클리어런스 비교 ──
    ax = axes[0, 2]
    clearances = [results[n]["min_clearance"] for n in names]
    colors_clr = ["red" if c < 0 else "green" for c in clearances]
    ax.barh(range(len(names)), clearances, color=colors_clr, alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Min Clearance (m)")
    ax.set_title("Min Obstacle Clearance (< 0 = collision)")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 4. 충돌 횟수 ──
    ax = axes[1, 0]
    collisions = [results[n]["collisions"] for n in names]
    ax.barh(range(len(names)), collisions, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Collision Count")
    ax.set_title("Collision Count (safety metric)")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 5. ESS 비교 ──
    ax = axes[1, 1]
    ess_vals = [results[n]["mean_ess"] for n in names]
    ax.barh(range(len(names)), ess_vals, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean ESS")
    ax.set_title("Effective Sample Size")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 6. 계산 시간 ──
    ax = axes[1, 2]
    times_ms = [results[n]["mean_time_ms"] for n in names]
    ax.barh(range(len(names)), times_ms, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Time (ms/step)")
    ax.set_title("Computation Time")
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("CBF Overlay Benchmark: {Base MPPI} × {Safety Mode}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")
    plt.close(fig)


def plot_clearance_timeline(results, save_path="plots/cbf_overlay_clearance_timeline.png"):
    """장애물 클리어런스 시간별 추이"""
    base_colors = {
        "Vanilla": "#1f77b4", "PI-MPPI": "#ff7f0e",
        "Kernel": "#2ca02c", "DIAL": "#d62728",
    }
    safety_styles = {"": "-", "ObsCost": "--", "CBF": "-.", "Shield": ":"}

    fig, ax = plt.subplots(figsize=(14, 6))
    t_axis = np.arange(NUM_STEPS) * DT

    for name, data in results.items():
        base = name.split("+")[0]
        safety = name.replace(base, "").lstrip("+")
        color = base_colors.get(base, "gray")
        style = safety_styles.get(safety, "-")
        ax.plot(t_axis, data["clearance"], color=color, linestyle=style,
                label=name, alpha=0.8, linewidth=1.5)

    ax.axhline(0, color="red", linestyle="--", linewidth=2, alpha=0.5,
               label="Collision boundary")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("Obstacle Clearance Timeline")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save_path}")
    plt.close(fig)


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CBF Overlay Benchmark: {base MPPI} × {safety mode}"
    )
    parser.add_argument(
        "--base", nargs="+",
        default=["vanilla", "pi", "kernel", "dial"],
        choices=["vanilla", "pi", "kernel", "dial"],
        help="Base MPPI variants to test",
    )
    parser.add_argument(
        "--safety", nargs="+",
        default=["none", "obs_cost", "cbf", "shield"],
        choices=["none", "obs_cost", "cbf", "shield"],
        help="Safety modes to test",
    )
    args = parser.parse_args()

    results = run_benchmark(args.base, args.safety)

    if results:
        print_results_table(results)
        plot_results(results)
        plot_clearance_timeline(results)


if __name__ == "__main__":
    main()
