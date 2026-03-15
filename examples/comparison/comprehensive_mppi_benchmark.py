#!/usr/bin/env python3
"""
종합 MPPI 안전 벤치마크

8종 MPPI 변형 × 6종 안전 메커니즘 × 3종 장애물 시나리오 매트릭스 벤치마크.
Gatekeeper, MPS 등 post-filter 안전 쉴드 포함.

유효 조합: 8 base × 6 safety - 7 invalid = 41/시나리오 × 3 = 123 총 실행

Usage:
    # 전체 실행 (~10분)
    PYTHONPATH=. python examples/comparison/comprehensive_mppi_benchmark.py

    # 특정 조합
    PYTHONPATH=. python examples/comparison/comprehensive_mppi_benchmark.py \
      --base vanilla kernel dial --safety none cbf_cost gatekeeper --scenario static

    # 빠른 테스트 (4 base × 4 safety × 1 scenario)
    PYTHONPATH=. python examples/comparison/comprehensive_mppi_benchmark.py \
      --base vanilla pi kernel dial \
      --safety none cbf_cost shield gatekeeper --scenario static
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
from mppi_controller.controllers.mppi.pi_mppi import PIMPPIController
from mppi_controller.controllers.mppi.kernel_mppi import KernelMPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
from mppi_controller.controllers.mppi.spline_mppi import SplineMPPIController
from mppi_controller.controllers.mppi.log_mppi import LogMPPIController
from mppi_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.gatekeeper import Gatekeeper
from mppi_controller.controllers.mppi.mps_controller import MPSController
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
    SmoothMPPIParams,
    SplineMPPIParams,
    LogMPPIParams,
    TsallisMPPIParams,
    ShieldMPPIParams,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ── 상수 ──────────────────────────────────────────────────────

N = 20
K = 512
DT = 0.05
SIGMA = np.array([0.5, 0.5])
Q = np.array([10.0, 10.0, 1.0])
R = np.array([0.1, 0.1])
DURATION = 10.0
NUM_STEPS = int(DURATION / DT)

ALL_BASES = ["vanilla", "pi", "kernel", "dial", "smooth", "spline", "log", "tsallis"]
ALL_SAFETIES = ["none", "obs_cost", "cbf_cost", "shield", "gatekeeper", "mps"]
ALL_SCENARIOS = ["static", "corridor", "dynamic"]


# ── Section 1: 시나리오 정의 ──────────────────────────────────

def make_static_field_scenario():
    """Static Field: seed=42 랜덤 장애물 + circle 궤적 (난이도: 중)"""
    rng = np.random.RandomState(42)
    obstacles = []
    radius = 3.0
    for _ in range(15):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(1.5, 4.5)
        ox = dist * np.cos(angle)
        oy = dist * np.sin(angle)
        r = rng.uniform(0.2, 0.5)
        # 원점(시작점) 근처와 너무 먼 것 제외
        if np.sqrt(ox**2 + oy**2) < 1.0 or np.sqrt(ox**2 + oy**2) > 5.5:
            continue
        obstacles.append((ox, oy, r))

    # 최소 12개 보장
    while len(obstacles) < 12:
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(2.0, 4.0)
        ox = dist * np.cos(angle)
        oy = dist * np.sin(angle)
        r = rng.uniform(0.2, 0.4)
        if np.sqrt(ox**2 + oy**2) > 1.0:
            obstacles.append((ox, oy, r))

    traj_fn = create_trajectory_function("circle", radius=radius)
    return obstacles, traj_fn, False, "Static Field"


def make_corridor_scenario():
    """Narrow Corridor: L자 벽 + 깔때기 (난이도: 상)"""
    obstacles = []

    # 직선 복도 벽 (x: 0→6, y=±0.8)
    for x in np.arange(0.0, 6.5, 0.5):
        obstacles.append((x, 1.0, 0.25))
        obstacles.append((x, -1.0, 0.25))

    # L턴 코너 (x=6, y: 0→4)
    for y in np.arange(0.0, 4.5, 0.5):
        obstacles.append((5.0, y, 0.25))
        obstacles.append((7.0, y, 0.25))

    # 깔때기 (x=6, y=4 → 좁아짐)
    obstacles.append((5.5, 4.5, 0.3))
    obstacles.append((6.5, 4.5, 0.3))

    # 웨이포인트 궤적: 시작 → 직진 → L턴 → 직진
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.5, 0.0, np.pi / 4],
        [6.0, 1.5, np.pi / 2],
        [6.0, 3.0, np.pi / 2],
        [6.0, 4.0, np.pi / 2],
    ])

    total_dist = 0.0
    segment_dists = []
    for i in range(len(waypoints) - 1):
        d = np.linalg.norm(waypoints[i + 1, :2] - waypoints[i, :2])
        segment_dists.append(d)
        total_dist += d

    speed = total_dist / DURATION

    def corridor_traj(t):
        dist_traveled = speed * t
        cum = 0.0
        for i, sd in enumerate(segment_dists):
            if cum + sd >= dist_traveled or i == len(segment_dists) - 1:
                frac = min(1.0, (dist_traveled - cum) / max(sd, 1e-6))
                pos = waypoints[i] + frac * (waypoints[i + 1] - waypoints[i])
                return pos
            cum += sd
        return waypoints[-1].copy()

    return obstacles, corridor_traj, False, "Narrow Corridor"


def make_dynamic_scenario():
    """Dynamic Bouncing: 4개 반사 장애물 (난이도: 상)"""
    # 장애물 초기 위치, 속도, 반사 경계
    dynamic_obs_config = [
        {"pos": np.array([2.5, 0.0]), "vel": np.array([0.0, 0.8]), "r": 0.4,
         "bounds": (-4.0, 4.0, -4.0, 4.0)},
        {"pos": np.array([-1.5, 2.5]), "vel": np.array([0.6, -0.3]), "r": 0.35,
         "bounds": (-5.0, 5.0, -5.0, 5.0)},
        {"pos": np.array([0.0, -3.0]), "vel": np.array([-0.5, 0.5]), "r": 0.3,
         "bounds": (-5.0, 5.0, -5.0, 5.0)},
        {"pos": np.array([-2.0, -1.0]), "vel": np.array([0.4, 0.6]), "r": 0.45,
         "bounds": (-5.0, 5.0, -5.0, 5.0)},
    ]

    # 시간별 장애물 위치 계산
    def get_obstacles(t):
        obs = []
        for cfg in dynamic_obs_config:
            pos = cfg["pos"].copy()
            vel = cfg["vel"].copy()
            # 간단한 반사 시뮬레이션
            x = pos[0] + vel[0] * t
            y = pos[1] + vel[1] * t
            xmin, xmax, ymin, ymax = cfg["bounds"]
            # 반사 처리 (주기적)
            rx = xmax - xmin
            ry = ymax - ymin
            x_rel = (x - xmin) % (2 * rx)
            y_rel = (y - ymin) % (2 * ry)
            if x_rel > rx:
                x_rel = 2 * rx - x_rel
            if y_rel > ry:
                y_rel = 2 * ry - y_rel
            x = xmin + x_rel
            y = ymin + y_rel
            obs.append((x, y, cfg["r"]))
        return obs

    initial_obs = get_obstacles(0.0)
    traj_fn = create_trajectory_function("circle", radius=3.0)
    return initial_obs, traj_fn, get_obstacles, "Dynamic Bouncing"


SCENARIO_BUILDERS = {
    "static": make_static_field_scenario,
    "corridor": make_corridor_scenario,
    "dynamic": make_dynamic_scenario,
}


# ── Section 2: 비용 함수 팩토리 ────────────────────────────────

def make_cost(safety_name, obstacles):
    """안전 모드에 따른 비용 함수 생성"""
    base_costs = [
        StateTrackingCost(Q),
        TerminalCost(Q),
        ControlEffortCost(R),
    ]

    if safety_name == "none":
        return CompositeMPPICost(base_costs)
    elif safety_name == "obs_cost":
        return CompositeMPPICost(
            base_costs + [ObstacleCost(obstacles, safety_margin=0.2, cost_weight=500.0)]
        )
    elif safety_name in ("cbf_cost", "shield"):
        return CompositeMPPICost(
            base_costs + [ControlBarrierCost(
                obstacles, cbf_alpha=0.3, cbf_weight=1000.0, safety_margin=0.1
            )]
        )
    elif safety_name in ("gatekeeper", "mps"):
        # post-filter에서 안전 보장, 샘플링 가이드용 장애물 비용 추가
        return CompositeMPPICost(
            base_costs + [ObstacleCost(obstacles, safety_margin=0.2, cost_weight=300.0)]
        )
    else:
        raise ValueError(f"Unknown safety: {safety_name}")


# ── Section 3: 컨트롤러 팩토리 ─────────────────────────────────

def build_controller(base_name, safety_name, obstacles):
    """
    (base × safety) 조합으로 컨트롤러 + post_filter 생성

    Returns:
        (model, ctrl, post_filter, label) | None (미지원 조합)
    """
    # Shield는 Vanilla만 지원
    if safety_name == "shield" and base_name != "vanilla":
        return None

    common = dict(N=N, dt=DT, K=K, lambda_=1.0, sigma=SIGMA, Q=Q, R=R)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    cost = make_cost(safety_name, obstacles)

    # Shield-MPPI: 별도 처리
    if safety_name == "shield":
        params = ShieldMPPIParams(
            **common,
            cbf_obstacles=obstacles,
            cbf_weight=1000.0,
            cbf_alpha=0.3,
            cbf_safety_margin=0.1,
            shield_enabled=True,
        )
        ctrl = ShieldMPPIController(model, params)
        return model, ctrl, None, "Vanilla+Shield"

    # Base 컨트롤러 생성
    if base_name == "vanilla":
        params = MPPIParams(**common)
        ctrl = MPPIController(model, params, cost_function=cost)
        label = "Vanilla"

    elif base_name == "pi":
        params = MPPIParams(**common)
        ctrl = PIMPPIController(model, params, cost_function=cost)
        label = "PI"

    elif base_name == "kernel":
        params = KernelMPPIParams(
            **common, num_support_pts=8, kernel_bandwidth=2.0
        )
        ctrl = KernelMPPIController(model, params, cost_function=cost)
        label = "Kernel"

    elif base_name == "dial":
        params = DIALMPPIParams(
            **common, n_diffuse_init=5, n_diffuse=3, traj_diffuse_factor=0.5
        )
        ctrl = DIALMPPIController(model, params, cost_function=cost)
        label = "DIAL"

    elif base_name == "smooth":
        params = SmoothMPPIParams(**common, jerk_weight=1.0)
        ctrl = SmoothMPPIController(model, params)
        ctrl.cost_function = cost
        label = "Smooth"

    elif base_name == "spline":
        params = SplineMPPIParams(**common, spline_num_knots=8, spline_degree=3)
        ctrl = SplineMPPIController(model, params)
        ctrl.cost_function = cost
        label = "Spline"

    elif base_name == "log":
        params = LogMPPIParams(**common, use_baseline=True)
        ctrl = LogMPPIController(model, params)
        ctrl.cost_function = cost
        label = "Log"

    elif base_name == "tsallis":
        params = TsallisMPPIParams(**common, tsallis_q=0.8)
        ctrl = TsallisMPPIController(model, params)
        ctrl.cost_function = cost
        label = "Tsallis"

    else:
        raise ValueError(f"Unknown base: {base_name}")

    # Safety suffix + post-filter 생성
    post_filter = None
    if safety_name == "obs_cost":
        label += "+ObsCost"
    elif safety_name == "cbf_cost":
        label += "+CBF"
    elif safety_name == "gatekeeper":
        label += "+GK"
        post_filter = Gatekeeper(
            model=model, obstacles=obstacles,
            safety_margin=0.15, backup_horizon=30, dt=DT,
        )
    elif safety_name == "mps":
        label += "+MPS"
        post_filter = MPSController(
            obstacles=obstacles,
            safety_margin=0.15, backup_horizon=20, dt=DT,
        )

    return model, ctrl, post_filter, label


# ── Section 4: 시뮬레이션 루프 ─────────────────────────────────

def run_simulation(model, ctrl, post_filter, traj_fn, obstacles,
                   get_dynamic_obs=False, label=""):
    """단일 컨트롤러 시뮬레이션"""
    init_pos = traj_fn(0.0)
    state = init_pos.copy()

    xy_list, err_list, ctrl_list = [], [], []
    ess_list, compute_times, clearance_list = [], [], []
    gate_stats = []

    for step in range(NUM_STEPS):
        t = step * DT
        ref = generate_reference_trajectory(traj_fn, t, N, DT)

        # 동적 장애물 업데이트
        if get_dynamic_obs:
            current_obs = get_dynamic_obs(t)
            # cost_function 내부 장애물 업데이트
            if hasattr(ctrl, 'cost_function') and hasattr(ctrl.cost_function, 'cost_functions'):
                for cf in ctrl.cost_function.cost_functions:
                    if hasattr(cf, 'obstacles'):
                        cf.obstacles = current_obs
                    if hasattr(cf, 'update_obstacles'):
                        cf.update_obstacles(current_obs)
            if post_filter is not None:
                if hasattr(post_filter, 'obstacles'):
                    post_filter.obstacles = current_obs
                if hasattr(post_filter, 'update_obstacles'):
                    post_filter.update_obstacles(current_obs)
        else:
            current_obs = obstacles

        t0 = time.perf_counter()
        control, info = ctrl.compute_control(state, ref)

        # Post-filter 적용
        pf_info = {}
        if isinstance(post_filter, Gatekeeper):
            control, pf_info = post_filter.filter(state, control)
        elif isinstance(post_filter, MPSController):
            control, pf_info = post_filter.shield(state, control, model)

        compute_times.append(time.perf_counter() - t0)

        state = model.step(state, control, DT)

        xy_list.append(state[:2].copy())
        err_list.append(np.linalg.norm(state[:2] - ref[0, :2]))
        ctrl_list.append(control.copy())
        ess_list.append(info.get("ess", 0.0))

        # 클리어런스 계산
        clearances = [
            np.sqrt((state[0] - ox)**2 + (state[1] - oy)**2) - r
            for ox, oy, r in current_obs
        ]
        clearance_list.append(min(clearances) if clearances else float("inf"))

        if pf_info:
            gate_stats.append(pf_info)

    controls = np.array(ctrl_list)
    jerk = np.mean(np.abs(np.diff(controls, axis=0))) if len(controls) > 1 else 0.0

    result = {
        "xy": np.array(xy_list),
        "errors": np.array(err_list),
        "controls": controls,
        "ess": np.array(ess_list),
        "compute_times": np.array(compute_times),
        "clearance": np.array(clearance_list),
        "rmse": np.sqrt(np.mean(np.array(err_list) ** 2)),
        "jerk": jerk,
        "mean_ess": np.mean(ess_list),
        "mean_time_ms": np.mean(compute_times) * 1000,
        "min_clearance": min(clearance_list) if clearance_list else float("inf"),
        "collisions": sum(1 for c in clearance_list if c < 0),
    }

    # Gatekeeper/MPS 통계 추가
    if isinstance(post_filter, Gatekeeper):
        stats = post_filter.get_statistics()
        result["gk_open_rate"] = stats.get("gate_open_rate", 1.0)
        result["gk_closed"] = stats.get("gate_closed_count", 0)
    elif isinstance(post_filter, MPSController):
        stats = post_filter.get_statistics()
        result["mps_shield_rate"] = stats.get("shield_rate", 0.0)
        result["mps_shielded"] = stats.get("shielded_count", 0)

    return result


# ── Section 5: 벤치마크 러너 ──────────────────────────────────

def run_scenario_benchmark(scenario_name, base_variants, safety_modes):
    """단일 시나리오 벤치마크"""
    builder = SCENARIO_BUILDERS[scenario_name]
    obstacles, traj_fn, dynamic_fn, scenario_title = builder()

    print(f"\n  {'─'*80}")
    print(f"  Scenario: {scenario_title}")
    print(f"  Obstacles: {len(obstacles)}, Dynamic: {bool(dynamic_fn)}")
    print(f"  {'─'*80}")

    results = {}
    for base in base_variants:
        for safety in safety_modes:
            result = build_controller(base, safety, obstacles)
            if result is None:
                print(f"    {base:8s} × {safety:10s}  SKIP")
                continue

            model, ctrl, post_filter, label = result
            print(f"    {label:22s} ...", end="", flush=True)

            try:
                data = run_simulation(
                    model, ctrl, post_filter, traj_fn, obstacles,
                    get_dynamic_obs=dynamic_fn, label=label,
                )
                results[label] = data

                extra = ""
                if "gk_open_rate" in data:
                    extra = f"  GK_open={data['gk_open_rate']:.1%}"
                elif "mps_shield_rate" in data:
                    extra = f"  MPS_shld={data['mps_shield_rate']:.1%}"

                print(f" RMSE={data['rmse']:.4f}  MinClr={data['min_clearance']:.3f}  "
                      f"Col={data['collisions']:3d}  "
                      f"ESS={data['mean_ess']:.1f}  "
                      f"{data['mean_time_ms']:.1f}ms{extra}")
            except Exception as e:
                print(f" ERROR: {e}")

    return results, obstacles, traj_fn, dynamic_fn, scenario_title


def run_benchmark(base_variants, safety_modes, scenarios):
    """전체 매트릭스 벤치마크"""
    total_combos = len(base_variants) * len(safety_modes) * len(scenarios)

    print(f"\n{'='*90}")
    print(f"  Comprehensive MPPI Safety Benchmark")
    print(f"  N={N}, K={K}, dt={DT}, duration={DURATION}s")
    print(f"  Base variants: {base_variants}")
    print(f"  Safety modes:  {safety_modes}")
    print(f"  Scenarios:     {scenarios}")
    print(f"  Max combos:    {total_combos}")
    print(f"{'='*90}")

    all_results = {}
    scenario_meta = {}

    for sc in scenarios:
        results, obstacles, traj_fn, dynamic_fn, title = run_scenario_benchmark(
            sc, base_variants, safety_modes
        )
        all_results[sc] = results
        scenario_meta[sc] = {
            "obstacles": obstacles,
            "traj_fn": traj_fn,
            "dynamic_fn": dynamic_fn,
            "title": title,
        }

    return all_results, scenario_meta


# ── Section 6: 결과 보고 ──────────────────────────────────────

def print_scenario_table(scenario_name, results):
    """시나리오별 결과 테이블"""
    if not results:
        return

    print(f"\n{'='*110}")
    print(f"  Results: {scenario_name}")
    print(f"{'='*110}")

    header = (f"  {'Method':<22} {'RMSE(m)':>8} {'MinClr(m)':>10} {'Collis':>7} "
              f"{'Jerk':>8} {'ESS':>7} {'ms/step':>8} {'Extra':>15}")
    print(header)
    print(f"  {'-'*105}")

    sorted_names = sorted(results.keys(), key=lambda n: results[n]["rmse"])
    for name in sorted_names:
        d = results[name]
        extra = ""
        if "gk_open_rate" in d:
            extra = f"GK:{d['gk_open_rate']:.0%}"
        elif "mps_shield_rate" in d:
            extra = f"MPS:{d['mps_shield_rate']:.0%}"

        print(f"  {name:<22} {d['rmse']:>8.4f} {d['min_clearance']:>10.3f} "
              f"{d['collisions']:>7d} {d['jerk']:>8.4f} "
              f"{d['mean_ess']:>7.1f} {d['mean_time_ms']:>7.1f}ms {extra:>15}")


def print_category_winners(results):
    """카테고리 우승자"""
    if not results:
        return

    print(f"\n  === Category Winners ===")

    # Best RMSE
    best_rmse = min(results.items(), key=lambda x: x[1]["rmse"])
    print(f"    Best RMSE:      {best_rmse[0]:22s}  ({best_rmse[1]['rmse']:.4f}m)")

    # Safest (max min_clearance)
    best_safe = max(results.items(), key=lambda x: x[1]["min_clearance"])
    print(f"    Safest:         {best_safe[0]:22s}  (min_clr={best_safe[1]['min_clearance']:.3f}m)")

    # Fewest collisions (then best RMSE as tiebreaker)
    best_no_col = min(results.items(), key=lambda x: (x[1]["collisions"], x[1]["rmse"]))
    print(f"    Fewest Collis:  {best_no_col[0]:22s}  ({best_no_col[1]['collisions']})")

    # Fastest
    best_fast = min(results.items(), key=lambda x: x[1]["mean_time_ms"])
    print(f"    Fastest:        {best_fast[0]:22s}  ({best_fast[1]['mean_time_ms']:.1f}ms)")

    # Smoothest (lowest jerk)
    best_smooth = min(results.items(), key=lambda x: x[1]["jerk"])
    print(f"    Smoothest:      {best_smooth[0]:22s}  (jerk={best_smooth[1]['jerk']:.4f})")


def print_cross_scenario_comparison(all_results):
    """시나리오 간 교차 비교"""
    if len(all_results) < 2:
        return

    print(f"\n{'='*110}")
    print(f"  Cross-Scenario Comparison")
    print(f"{'='*110}")

    # 모든 시나리오에 등장하는 method 찾기
    all_methods = set()
    for results in all_results.values():
        all_methods.update(results.keys())

    common_methods = set()
    for method in all_methods:
        if all(method in results for results in all_results.values()):
            common_methods.add(method)

    if not common_methods:
        print("  No common methods across all scenarios.")
        return

    scenarios = list(all_results.keys())
    header = f"  {'Method':<22}"
    for sc in scenarios:
        header += f" {'RMSE_' + sc:>12} {'Col_' + sc:>8}"
    print(header)
    print(f"  {'-'*(22 + len(scenarios) * 22)}")

    sorted_methods = sorted(common_methods)
    for method in sorted_methods:
        line = f"  {method:<22}"
        for sc in scenarios:
            d = all_results[sc][method]
            line += f" {d['rmse']:>12.4f} {d['collisions']:>8d}"
        print(line)

    # 시나리오별 best method
    print(f"\n  === Best per Scenario ===")
    for sc in scenarios:
        best = min(all_results[sc].items(), key=lambda x: x[1]["rmse"])
        safest = min(all_results[sc].items(), key=lambda x: (x[1]["collisions"], -x[1]["min_clearance"]))
        print(f"    {sc:12s}: Best RMSE={best[0]} ({best[1]['rmse']:.4f}), "
              f"Safest={safest[0]} (col={safest[1]['collisions']})")


def print_all_results(all_results):
    """모든 결과 출력"""
    for sc_name, results in all_results.items():
        print_scenario_table(sc_name, results)
        print_category_winners(results)

    print_cross_scenario_comparison(all_results)


# ── Section 7: 시각화 ─────────────────────────────────────────

BASE_COLORS = {
    "Vanilla": "#1f77b4", "PI": "#ff7f0e", "Kernel": "#2ca02c",
    "DIAL": "#d62728", "Smooth": "#9467bd", "Spline": "#8c564b",
    "Log": "#e377c2", "Tsallis": "#7f7f7f",
}

SAFETY_STYLES = {
    "": "-", "ObsCost": "--", "CBF": "-.",
    "Shield": ":", "GK": (0, (3, 1, 1, 1)), "MPS": (0, (5, 2)),
}


def _get_base_and_safety(name):
    """'Kernel+CBF' → ('Kernel', 'CBF')"""
    parts = name.split("+", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def plot_scenario_results(results, obstacles, scenario_title,
                          save_path, traj_fn=None, ref_radius=3.0):
    """시나리오별 2×3 결과 플롯"""
    if not results:
        return

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # ── 1. XY 궤적 ──
    ax = axes[0, 0]
    if traj_fn is not None:
        t_ref = np.linspace(0, DURATION, 300)
        ref_xy = np.array([traj_fn(t)[:2] for t in t_ref])
        ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.3, label="Reference", lw=1)

    for ox, oy, r in obstacles:
        circle = Circle((ox, oy), r, fill=True, color="red", alpha=0.2)
        ax.add_patch(circle)

    for name, data in results.items():
        base, safety = _get_base_and_safety(name)
        color = BASE_COLORS.get(base, "gray")
        style = SAFETY_STYLES.get(safety, "-")
        ax.plot(data["xy"][:, 0], data["xy"][:, 1],
                color=color, linestyle=style, label=name, alpha=0.7, lw=1.2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=5, loc="upper right", ncol=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── 2. RMSE 비교 ──
    ax = axes[0, 1]
    sorted_names = sorted(results.keys(), key=lambda n: results[n]["rmse"])
    rmses = [results[n]["rmse"] for n in sorted_names]
    colors_bar = [BASE_COLORS.get(_get_base_and_safety(n)[0], "gray") for n in sorted_names]
    ax.barh(range(len(sorted_names)), rmses, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_xlabel("RMSE (m)")
    ax.set_title("Tracking RMSE")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 3. 최소 클리어런스 ──
    ax = axes[0, 2]
    clr_vals = [results[n]["min_clearance"] for n in sorted_names]
    colors_clr = ["red" if c < 0 else "green" for c in clr_vals]
    ax.barh(range(len(sorted_names)), clr_vals, color=colors_clr, alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_xlabel("Min Clearance (m)")
    ax.set_title("Min Obstacle Clearance (< 0 = collision)")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 4. 클리어런스 타임라인 ──
    ax = axes[1, 0]
    t_axis = np.arange(NUM_STEPS) * DT
    for name, data in results.items():
        base, safety = _get_base_and_safety(name)
        color = BASE_COLORS.get(base, "gray")
        style = SAFETY_STYLES.get(safety, "-")
        ax.plot(t_axis, data["clearance"], color=color, linestyle=style,
                label=name, alpha=0.6, lw=1)
    ax.axhline(0, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("Clearance Timeline")
    ax.legend(fontsize=5, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    # ── 5. ESS 비교 ──
    ax = axes[1, 1]
    ess_vals = [results[n]["mean_ess"] for n in sorted_names]
    ax.barh(range(len(sorted_names)), ess_vals, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_xlabel("Mean ESS")
    ax.set_title("Effective Sample Size")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 6. 계산 시간 ──
    ax = axes[1, 2]
    times_ms = [results[n]["mean_time_ms"] for n in sorted_names]
    ax.barh(range(len(sorted_names)), times_ms, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_xlabel("Time (ms/step)")
    ax.set_title("Computation Time")
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"Comprehensive MPPI Benchmark: {scenario_title}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")
    plt.close(fig)


def plot_all_results(all_results, scenario_meta):
    """모든 시나리오 플롯"""
    for sc_name, results in all_results.items():
        meta = scenario_meta[sc_name]
        save_path = f"plots/comprehensive_benchmark_{sc_name}.png"
        plot_scenario_results(
            results,
            obstacles=meta["obstacles"],
            scenario_title=meta["title"],
            save_path=save_path,
            traj_fn=meta["traj_fn"],
        )


# ── Section 8: 메인 / CLI ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive MPPI Safety Benchmark: "
                    "{8 base MPPI} × {6 safety} × {3 scenarios}"
    )
    parser.add_argument(
        "--base", nargs="+", default=ALL_BASES,
        choices=ALL_BASES,
        help="Base MPPI variants",
    )
    parser.add_argument(
        "--safety", nargs="+", default=ALL_SAFETIES,
        choices=ALL_SAFETIES,
        help="Safety modes",
    )
    parser.add_argument(
        "--scenario", nargs="+", default=ALL_SCENARIOS,
        choices=ALL_SCENARIOS,
        help="Obstacle scenarios",
    )
    args = parser.parse_args()

    all_results, scenario_meta = run_benchmark(args.base, args.safety, args.scenario)

    # 결과 출력
    print_all_results(all_results)

    # 시각화
    plot_all_results(all_results, scenario_meta)

    # 총 실행 수
    total = sum(len(r) for r in all_results.values())
    print(f"\n  Total runs: {total}")


if __name__ == "__main__":
    main()
