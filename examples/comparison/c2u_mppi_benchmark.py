#!/usr/bin/env python3
"""
C2U-MPPI (Chance-Constrained Unscented MPPI) 벤치마크: 3-Way × 2 시나리오

방법:
  1. Vanilla MPPI          — 고정 sigma, 장애물 비용 없음
  2. Uncertainty-Aware MPPI — 적응 샘플링 + ObstacleCost
  3. C2U-MPPI              — UT 공분산 전파 + 확률적 기회 제약

시나리오:
  A. clean    — 외란 없음 (기준선): 3자 동등, C2U 약간 보수적
  B. noisy    — 프로세스 노이즈 추가: C2U > UncMPPI > Vanilla 안전성

측정:
  - 충돌 수 (최소 장애물 거리 < radius)
  - 최소 장애물 거리
  - 목표 도달 RMSE
  - 계산 시간
  - 유효 반경 변화 (C2U only)

Usage:
    PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py
    PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py --scenario noisy
    PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py --all-scenarios
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
    UncertaintyMPPIParams,
    C2UMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.uncertainty_mppi import (
    UncertaintyMPPIController,
)
from mppi_controller.controllers.mppi.c2u_mppi import C2UMPPIController
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


# ── 불확실성 모델 ────────────────────────────────────────────

class ConstantUncertainty:
    """균일 불확실성"""

    def __init__(self, std_val=0.05):
        self.std_val = std_val

    def __call__(self, states, controls):
        if states.ndim == 1:
            states = states[None, :]
        nx = states.shape[-1]
        return np.full((states.shape[0], nx), self.std_val)


# ── 시나리오 설정 ────────────────────────────────────────────

# 장애물 배치 (원형 궤적 경로에 겹치도록)
OBSTACLES = [
    (4.0, 2.5, 0.4),   # 오른쪽 위
    (-2.0, 3.5, 0.5),  # 왼쪽 위
    (1.0, -4.5, 0.3),  # 아래
]


def get_scenarios():
    return {
        "clean": {
            "name": "Clean (no disturbance)",
            "process_noise_std": None,
            "unc_model": ConstantUncertainty(0.01),
        },
        "noisy": {
            "name": "Noisy (process noise)",
            "process_noise_std": np.array([0.04, 0.04, 0.01]),
            "unc_model": ConstantUncertainty(0.06),
        },
    }


def create_trajectory_fn(name):
    if name == "circle":
        return circle_trajectory
    elif name == "figure8":
        return figure_eight_trajectory
    return circle_trajectory


# ── 시뮬레이션 루프 ───────────────────────────────────────────

def run_simulation(model, controller, reference_fn, initial_state, dt, duration,
                   process_noise_std=None):
    """모델 + 컨트롤러 시뮬레이션 실행"""
    num_steps = int(duration / dt)
    state = initial_state.copy()
    t = 0.0

    states = [state.copy()]
    controls_list = []
    solve_times = []
    infos = []

    for _ in range(num_steps):
        ref = reference_fn(t)

        t0 = time.time()
        control, info = controller.compute_control(state, ref)
        solve_times.append(time.time() - t0)

        next_state = model.step(state, control, dt)
        if process_noise_std is not None:
            next_state = next_state + np.random.normal(0, process_noise_std)
        next_state = model.normalize_state(next_state)

        states.append(next_state.copy())
        controls_list.append(control.copy())
        infos.append(info)

        state = next_state
        t += dt

    return {
        "states": np.array(states),
        "controls": np.array(controls_list),
        "solve_times": np.array(solve_times),
        "infos": infos,
    }


def compute_obstacle_metrics(states, obstacles):
    """장애물 관련 메트릭 계산"""
    n_collisions = 0
    min_dist = float("inf")
    min_distances = []

    for st in states:
        x, y = st[0], st[1]
        for ox, oy, r in obstacles:
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            clearance = dist - r
            min_dist = min(min_dist, clearance)
            if clearance < 0:
                n_collisions += 1
        # 가장 가까운 장애물과의 거리
        closest = min(np.sqrt((x - ox)**2 + (y - oy)**2) - r for ox, oy, r in obstacles)
        min_distances.append(closest)

    return {
        "n_collisions": n_collisions,
        "min_clearance": min_dist,
        "mean_min_clearance": np.mean(min_distances),
    }


def compute_tracking_rmse(states, reference_fn, dt):
    """궤적 추적 RMSE 계산"""
    errors = []
    for i, st in enumerate(states):
        ref = reference_fn(i * dt) if callable(reference_fn) else reference_fn
        if isinstance(ref, np.ndarray) and ref.ndim == 1:
            err = np.sqrt((st[0] - ref[0])**2 + (st[1] - ref[1])**2)
            errors.append(err)
    return np.mean(errors) if errors else 0.0


# ── 벤치마크 메인 ─────────────────────────────────────────────

def run_benchmark(args):
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]

    print(f"\n{'=' * 72}")
    print(f"  C2U-MPPI Benchmark: 3-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  Trajectory: {args.trajectory} | Duration: {args.duration}s | Seed: {args.seed}")
    print(f"{'=' * 72}")

    # 모델
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    trajectory_fn = create_trajectory_fn(args.trajectory)
    initial_state = trajectory_fn(0.0)
    unc_model = scenario["unc_model"]

    # 공통 MPPI 파라미터
    common = dict(
        K=256, N=20, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )

    # ── 3가지 방법 ──

    # 1. Vanilla MPPI + ObstacleCost
    def make_vanilla():
        params = MPPIParams(**common)
        cost = CompositeMPPICost([
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
            ControlEffortCost(params.R),
            ObstacleCost(OBSTACLES, safety_margin=0.1, cost_weight=200.0),
        ])
        return MPPIController(model, params, cost_function=cost)

    # 2. Uncertainty-Aware MPPI + ObstacleCost
    def make_uncertainty():
        params = UncertaintyMPPIParams(
            **common, exploration_factor=1.5,
            uncertainty_strategy="previous_trajectory",
        )
        cost = CompositeMPPICost([
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
            ControlEffortCost(params.R),
            ObstacleCost(OBSTACLES, safety_margin=0.2, cost_weight=200.0),
        ])
        return UncertaintyMPPIController(
            model, params, cost_function=cost, uncertainty_fn=unc_model,
        )

    # 3. C2U-MPPI (ChanceConstraintCost 자동 포함)
    def make_c2u():
        params = C2UMPPIParams(
            **common,
            cc_obstacles=OBSTACLES,
            chance_alpha=0.05,
            chance_cost_weight=500.0,
            process_noise_scale=0.02,
            cc_margin_factor=1.0,
        )
        cost = CompositeMPPICost([
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
            ControlEffortCost(params.R),
        ])
        return C2UMPPIController(model, params, cost_function=cost)

    variants = [
        {"name": "Vanilla MPPI",      "short": "Vanilla",  "make": make_vanilla},
        {"name": "Uncertainty MPPI",   "short": "UncMPPI",  "make": make_uncertainty},
        {"name": "C2U-MPPI",          "short": "C2U",      "make": make_c2u},
    ]

    ref_fn = lambda t, _fn=trajectory_fn, _N=common["N"], _dt=common["dt"]: \
        generate_reference_trajectory(_fn, t, _N, _dt)

    # ── 실행 ──
    results = []
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

        # 메트릭 계산
        obs_metrics = compute_obstacle_metrics(history["states"], OBSTACLES)
        rmse = compute_tracking_rmse(
            history["states"],
            lambda t, _fn=trajectory_fn: _fn(t),
            common["dt"],
        )

        mean_solve = np.mean(history["solve_times"]) * 1000
        max_solve = np.max(history["solve_times"]) * 1000

        # C2U 고유 메트릭
        c2u_stats = {}
        if var["short"] == "C2U" and history["infos"]:
            last_info = history["infos"][-1]
            r_eff = last_info.get("effective_radii")
            if r_eff is not None:
                c2u_stats["mean_r_eff"] = float(np.mean(r_eff))
                c2u_stats["max_r_eff"] = float(np.max(r_eff))
            cov_stats = last_info.get("covariance_stats", {})
            c2u_stats["final_trace"] = cov_stats.get("final_trace", 0)

        results.append({
            "name": var["name"],
            "short": var["short"],
            "obs_metrics": obs_metrics,
            "rmse": rmse,
            "mean_solve_ms": mean_solve,
            "max_solve_ms": max_solve,
            "elapsed": elapsed,
            "c2u_stats": c2u_stats,
        })

        print(f"done ({elapsed:.1f}s)")

    # ── 결과 출력 ──
    print(f"\n{'─' * 72}")
    print(f"{'Method':<22} {'Collisions':>10} {'MinClear':>10} {'RMSE':>10} {'Time(ms)':>10}")
    print(f"{'─' * 72}")

    for r in results:
        print(
            f"{r['name']:<22} "
            f"{r['obs_metrics']['n_collisions']:>10d} "
            f"{r['obs_metrics']['min_clearance']:>10.3f} "
            f"{r['rmse']:>10.3f} "
            f"{r['mean_solve_ms']:>10.1f}"
        )

    print(f"{'─' * 72}")

    # C2U 고유 메트릭
    for r in results:
        if r["c2u_stats"]:
            stats = r["c2u_stats"]
            print(f"\n  C2U-MPPI 고유 메트릭:")
            print(f"    유효 반경: mean={stats.get('mean_r_eff', 0):.4f}, "
                  f"max={stats.get('max_r_eff', 0):.4f}")
            print(f"    최종 공분산 trace={stats.get('final_trace', 0):.6f}")

    print()
    return results


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C2U-MPPI Benchmark")
    parser.add_argument("--scenario", default="clean", choices=["clean", "noisy"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--trajectory", default="circle", choices=["circle", "figure8"])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.all_scenarios:
        for scenario_name in get_scenarios():
            args.scenario = scenario_name
            run_benchmark(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
