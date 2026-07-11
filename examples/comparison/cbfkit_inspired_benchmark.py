#!/usr/bin/env python3
"""
cbfkit-inspired 안전 기법 벤치마크: 10-Way x 4 시나리오

방법 (10):
  1. Vanilla MPPI       — ObstacleCost (soft penalty)
  2. CBF-MPPI           — ControlBarrierCost (discrete zeroing CBF)
  3. Shield-MPPI        — rollout per-step 해석적 CBF 클리핑
  4. HOCBF-MPPI         — HOCBFCost (exponential high-order CBF, rd=2 대응)
  5. HOCBF-Filter       — Vanilla+ObstacleCost 후 HOCBFFilter 사후 필터
  6. StochasticCBF-MPPI — Itô 보정 stochastic CBF 비용
  7. RiskAwareCBF-MPPI  — P(min_t h<0) ≤ ρ 경로 적분 위험 마진
  8. RobustCBF-MPPI     — 유계 외란 worst-case 마진 강화
  9. CLF-CBF-QP         — 독립 QP 베이스라인 (CLF + hard CBF)
 10. CBF-QP             — CLF 없는 pure-pursuit + CBF-QP 필터

시나리오 4개:
  A. static_kin   — 기구학 3D, 원형 r=2.0 궤적 + 경로상 장애물 3개, 무노이즈
  B. dynamic_rd2  — 동역학 5D (가속도 제어, relative degree 2 — HOCBF 홈그라운드)
  C. stochastic   — 기구학 + 강한 프로세스 노이즈, 3 seeds mean±std
  D. risk_sweep   — RiskAwareCBF-MPPI ρ ∈ {0.5,0.2,0.1,0.05,0.01} 스윕 (3 seeds)

Usage:
    PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario static_kin
    PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario stochastic --no-plot
    PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario static_kin --duration 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    ShieldMPPIParams,
)
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.sampling import GaussianSampler
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.controllers.mppi import (
    HOCBFCost,
    HOCBFFilter,
    StochasticCBFCost,
    RiskAwareCBFCost,
    RobustCBFCost,
    CLFCBFQPParams,
    CLFCBFQPController,
    CBFOnlyQPController,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory


# ── 공통 설정 ──────────────────────────────────────────────────

K = 512
N = 20
DT = 0.05
LAMBDA = 1.0

CIRCLE_R = 2.0
CIRCLE_W = 0.5   # v_ref = R·ω = 1.0 m/s, ω_ref = 0.5 rad/s
V_REF = CIRCLE_R * CIRCLE_W

# 경로상 장애물 3개 (r=0.3), 원 r=2.0 위 90° / 210° / 330°
def get_obstacles():
    angs = np.deg2rad([90.0, 210.0, 330.0])
    return [
        (float(CIRCLE_R * np.cos(a)), float(CIRCLE_R * np.sin(a)), 0.3)
        for a in angs
    ]


def kin_circle(t):
    th = CIRCLE_W * t
    return np.array([
        CIRCLE_R * np.cos(th),
        CIRCLE_R * np.sin(th),
        th + np.pi / 2,
    ])


def dyn_circle(t):
    """속도 확장 레퍼런스: [x, y, θ, v_ref, ω_ref]"""
    base = kin_circle(t)
    return np.concatenate([base, [V_REF, CIRCLE_W]])


METHODS = [
    "Vanilla",
    "CBF-MPPI",
    "Shield-MPPI",
    "HOCBF-MPPI",
    "HOCBF-Filter",
    "StochasticCBF-MPPI",
    "RiskAwareCBF-MPPI",
    "RobustCBF-MPPI",
    "CLF-CBF-QP",
    "CBF-QP",
]

COLORS = {
    "Vanilla": "#1f77b4",
    "CBF-MPPI": "#ff7f0e",
    "Shield-MPPI": "#2ca02c",
    "HOCBF-MPPI": "#d62728",
    "HOCBF-Filter": "#9467bd",
    "StochasticCBF-MPPI": "#8c564b",
    "RiskAwareCBF-MPPI": "#e377c2",
    "RobustCBF-MPPI": "#7f7f7f",
    "CLF-CBF-QP": "#bcbd22",
    "CBF-QP": "#17becf",
}

SHORT = {
    "Vanilla": "Van",
    "CBF-MPPI": "CBF",
    "Shield-MPPI": "Shld",
    "HOCBF-MPPI": "HOCBF",
    "HOCBF-Filter": "HOFlt",
    "StochasticCBF-MPPI": "StoCBF",
    "RiskAwareCBF-MPPI": "RiskCBF",
    "RobustCBF-MPPI": "RobCBF",
    "CLF-CBF-QP": "CLFQP",
    "CBF-QP": "CBFQP",
}

RESULTS_DIR = "results/cbfkit_inspired"
PLOTS_DIR = "plots"


# ── 시나리오 정의 ──────────────────────────────────────────────

def get_scenarios():
    obstacles = get_obstacles()
    return {
        "static_kin": {
            "name": "A. Static Obstacles (Kinematic, rd=1)",
            "model_type": "kinematic",
            "obstacles": obstacles,
            "trajectory_fn": kin_circle,
            "initial_state": np.array([CIRCLE_R, 0.0, np.pi / 2]),
            "duration": 20.0,
            "process_noise_std": None,
            "seeds": [42],
            "description": "diffdrive kinematic, circle r=2.0, 3 on-path obstacles, no noise",
        },
        "dynamic_rd2": {
            "name": "B. Dynamic 5D (Acceleration Control, rd=2)",
            "model_type": "dynamic",
            "obstacles": obstacles,
            "trajectory_fn": dyn_circle,
            "initial_state": np.array([CIRCLE_R, 0.0, np.pi / 2, 0.0, 0.0]),
            "duration": 20.0,
            "process_noise_std": None,
            "seeds": [42],
            "description": "diffdrive dynamic 5D, velocity-augmented reference (v=1.0, w=0.5)",
        },
        "stochastic": {
            "name": "C. Stochastic (Strong Process Noise)",
            "model_type": "kinematic",
            "obstacles": obstacles,
            "trajectory_fn": kin_circle,
            "initial_state": np.array([CIRCLE_R, 0.0, np.pi / 2]),
            "duration": 20.0,
            "process_noise_std": np.array([0.05, 0.05, 0.02]),
            "seeds": [42, 43, 44],
            "description": "kinematic + per-step noise std [0.05,0.05,0.02], 3 seeds",
        },
        "risk_sweep": {
            "name": "D. Risk Budget Sweep (RiskAwareCBF)",
            "model_type": "kinematic",
            "obstacles": obstacles,
            "trajectory_fn": kin_circle,
            "initial_state": np.array([CIRCLE_R, 0.0, np.pi / 2]),
            "duration": 20.0,
            "process_noise_std": np.array([0.05, 0.05, 0.02]),
            "seeds": [42, 43, 44],
            "rho_values": [0.5, 0.2, 0.1, 0.05, 0.01],
            "description": "RiskAwareCBF rho sweep under stochastic setup + Vanilla/StochasticCBF refs",
        },
    }


# ── 모델 / 컨트롤러 팩토리 ─────────────────────────────────────

def make_model(model_type):
    if model_type == "dynamic":
        return DifferentialDriveDynamic()  # v_max=2.0, a_max=2.0
    return DifferentialDriveKinematic(v_max=1.5, omega_max=2.0)


def _base_params(model):
    is_dyn = model.state_dim >= 5
    if is_dyn:
        Q = np.array([10.0, 10.0, 1.0, 1.0, 0.5])
        sigma = np.array([1.0, 1.0])
    else:
        Q = np.array([10.0, 10.0, 1.0])
        sigma = np.array([0.5, 0.5])
    R = np.array([0.1, 0.1])
    return dict(K=K, N=N, dt=DT, lambda_=LAMBDA, sigma=sigma, Q=Q, R=R)


def _tracking_costs(params_dict):
    p = MPPIParams(**params_dict)
    return [
        StateTrackingCost(p.Q),
        TerminalCost(p.Qf),
        ControlEffortCost(p.R),
    ]


class PostFilterController:
    """Vanilla MPPI 출력에 HOCBFFilter 사후 필터 적용 래퍼"""

    def __init__(self, base_controller, hocbf_filter):
        self.base = base_controller
        self.filter = hocbf_filter

    def compute_control(self, state, reference_trajectory):
        u, info = self.base.compute_control(state, reference_trajectory)
        u_safe, finfo = self.filter.filter_control(state, u)
        info["hocbf_filter"] = finfo
        return u_safe, info


def make_controller(name, scenario, rho_override=None, seed=None):
    """단일 컨트롤러 생성 (매 run마다 새로 생성, seed 로 샘플러 고정)"""
    model = make_model(scenario["model_type"])
    is_dyn = model.state_dim >= 5
    obstacles = scenario["obstacles"]
    pd = _base_params(model)
    base_costs = _tracking_costs(pd)
    # 기본 GaussianSampler(seed=None) 는 OS 엔트로피 시드 → 재현성 위해 명시 시드
    sampler = GaussianSampler(pd["sigma"], seed=seed)

    noise_std = scenario.get("process_noise_std")
    # 이산 노이즈 std → 연속시간 확산 σ (per-step std = σ·√dt)
    sigma_process = (
        np.asarray(noise_std) / np.sqrt(DT) if noise_std is not None else None
    )
    is_noisy = noise_std is not None
    # 유계 외란 상한: 노이즈 시 1σ 등가 속도 (0.05m / 0.05s = 1.0 m/s)
    w_max = 1.0 if is_noisy else 0.2

    if name == "Vanilla":
        cost = CompositeMPPICost(
            base_costs + [ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0)]
        )
        return MPPIController(model, MPPIParams(**pd), cost, noise_sampler=sampler)

    if name == "CBF-MPPI":
        cost = CompositeMPPICost(
            base_costs
            + [ControlBarrierCost(obstacles, cbf_alpha=0.3, cbf_weight=1000.0,
                                  safety_margin=0.1)]
        )
        return MPPIController(model, MPPIParams(**pd), cost, noise_sampler=sampler)

    if name == "Shield-MPPI":
        sp = ShieldMPPIParams(
            **pd,
            cbf_obstacles=list(obstacles),
            cbf_alpha=0.3, cbf_weight=1000.0, cbf_safety_margin=0.1,
            shield_enabled=True,
        )
        # CBF 비용은 내부에서 자동 추가
        return ShieldMPPIController(model, sp, CompositeMPPICost(base_costs),
                                    noise_sampler=sampler)

    if name == "HOCBF-MPPI":
        rd = 2 if is_dyn else 1
        cost = CompositeMPPICost(
            base_costs
            + [HOCBFCost(obstacles, lambda1=2.0, lambda2=2.0, weight=1000.0,
                         safety_margin=0.1, dt=DT, relative_degree=rd)]
        )
        return MPPIController(model, MPPIParams(**pd), cost, noise_sampler=sampler)

    if name == "HOCBF-Filter":
        cost = CompositeMPPICost(
            base_costs + [ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0)]
        )
        base = MPPIController(model, MPPIParams(**pd), cost, noise_sampler=sampler)
        filt = HOCBFFilter(
            model, obstacles, lambda1=2.0, lambda2=2.0,
            safety_margin=0.1, relative_degree=None,  # auto-detect
        )
        return PostFilterController(base, filt)

    if name == "StochasticCBF-MPPI":
        # alpha=6.0, weight=50 ≡ ControlBarrierCost(alpha=0.3, weight=1000) @ σ=0
        cost = CompositeMPPICost(
            base_costs
            + [StochasticCBFCost(obstacles, alpha=6.0, beta=0.5,
                                 sigma_process=sigma_process,
                                 weight=50.0, safety_margin=0.1, dt=DT)]
        )
        return MPPIController(model, MPPIParams(**pd), cost, noise_sampler=sampler)

    if name == "RiskAwareCBF-MPPI":
        rho = rho_override if rho_override is not None else 0.1
        cost = CompositeMPPICost(
            base_costs
            + [RiskAwareCBFCost(obstacles, rho=rho,
                                sigma_process=sigma_process,
                                weight=1000.0, safety_margin=0.1, dt=DT)]
        )
        return MPPIController(model, MPPIParams(**pd), cost, noise_sampler=sampler)

    if name == "RobustCBF-MPPI":
        cost = CompositeMPPICost(
            base_costs
            + [RobustCBFCost(obstacles, w_max=w_max, alpha=0.3, weight=1000.0,
                             safety_margin=0.1, dt=DT)]
        )
        return MPPIController(model, MPPIParams(**pd), cost, noise_sampler=sampler)

    if name == "CLF-CBF-QP":
        qp = CLFCBFQPParams(dt=DT, safety_margin=0.15)
        return CLFCBFQPController(model, qp, obstacles)

    if name == "CBF-QP":
        qp = CLFCBFQPParams(dt=DT, safety_margin=0.15)
        return CBFOnlyQPController(model, qp, obstacles)

    raise ValueError(f"Unknown method: {name}")


# ── 시뮬레이션 ─────────────────────────────────────────────────

def run_single(name, scenario, seed, duration=None, rho_override=None):
    """단일 (method, seed) 시뮬레이션. 스칼라 info 만 저장 (메모리 절약)."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 10000)

    controller = make_controller(name, scenario, rho_override=rho_override, seed=seed)
    model = make_model(scenario["model_type"])
    trajectory_fn = scenario["trajectory_fn"]
    dur = duration if duration is not None else scenario["duration"]
    num_steps = int(dur / DT)
    noise_std = scenario.get("process_noise_std")

    state = scenario["initial_state"].copy()
    nx = state.shape[0]

    states = [state.copy()]
    solve_times = []
    ess_list = []
    filter_rate = []
    qp_feasible = []

    for step in range(num_steps):
        t = step * DT
        ref = generate_reference_trajectory(trajectory_fn, t, N, DT)

        t0 = time.perf_counter()
        control, info = controller.compute_control(state, ref)
        solve_times.append(time.perf_counter() - t0)

        if isinstance(info, dict):
            if "ess" in info:
                ess_list.append(float(info["ess"]))
            if "hocbf_filter" in info:
                filter_rate.append(1.0 if info["hocbf_filter"]["filtered"] else 0.0)
            if "qp_feasible" in info:
                qp_feasible.append(1.0 if info["qp_feasible"] else 0.0)

        state = model.step(state, control, DT)
        if noise_std is not None:
            state = state + rng.normal(size=nx) * np.asarray(noise_std)[:nx]

        states.append(state.copy())

    return {
        "states": np.array(states),
        "solve_times": np.array(solve_times),
        "ess_list": ess_list,
        "filter_rate": float(np.mean(filter_rate)) if filter_rate else None,
        "qp_feasible_rate": float(np.mean(qp_feasible)) if qp_feasible else None,
    }


def compute_metrics(history, scenario):
    states = history["states"]
    trajectory_fn = scenario["trajectory_fn"]
    obstacles = scenario["obstacles"]

    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * DT)
        errors.append(float(np.hypot(st[0] - ref[0], st[1] - ref[1])))
    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))

    # 클리어런스 (마진 없이 dist - r_obs), 충돌 = dist < r_obs 인 타임스텝 수
    pos = states[:, :2]
    clear_ts = np.full(len(states), np.inf)
    for ox, oy, r in obstacles:
        d = np.hypot(pos[:, 0] - ox, pos[:, 1] - oy) - r
        clear_ts = np.minimum(clear_ts, d)
    min_clearance = float(np.min(clear_ts))
    collisions = int(np.sum(clear_ts < 0.0))

    mean_solve_ms = float(np.mean(history["solve_times"])) * 1000.0
    return {
        "position_rmse": rmse,
        "min_clearance": min_clearance,
        "collisions": collisions,
        "mean_solve_time_ms": mean_solve_ms,
        "control_rate_hz": float(1000.0 / mean_solve_ms) if mean_solve_ms > 0 else 0.0,
        "errors": errors,
        "clearance_ts": clear_ts.tolist(),
        "mean_ess": float(np.mean(history["ess_list"])) if history["ess_list"] else None,
        "filter_rate": history["filter_rate"],
        "qp_feasible_rate": history["qp_feasible_rate"],
    }


# ── 결과 저장 ─────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(scenario_key, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{scenario_key}.json")
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    print(f"  JSON saved: {path}")
    return path


# ── 표준 시나리오 (static_kin / dynamic_rd2 / stochastic) ──────

def run_standard_scenario(scenario_key, args):
    scenario = get_scenarios()[scenario_key]
    seeds = scenario["seeds"]
    dur = args.duration if args.duration else scenario["duration"]

    print(f"\n{'=' * 78}")
    print(f"  cbfkit-inspired Benchmark — {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  K={K}, N={N}, dt={DT} | duration={dur}s | seeds={seeds}")
    print(f"{'=' * 78}")

    results = {}
    for i, name in enumerate(METHODS):
        print(f"  [{i + 1}/{len(METHODS)}] {name:<20}", end=" ", flush=True)
        t0 = time.time()
        per_seed = []
        histories = []
        for seed in seeds:
            hist = run_single(name, scenario, seed, duration=dur)
            m = compute_metrics(hist, scenario)
            per_seed.append(m)
            histories.append(hist)
        elapsed = time.time() - t0

        agg = {}
        for key in ["position_rmse", "min_clearance", "collisions",
                    "mean_solve_time_ms", "control_rate_hz"]:
            vals = np.array([m[key] for m in per_seed], dtype=float)
            agg[key] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        agg["mean_ess"] = per_seed[0]["mean_ess"]
        agg["filter_rate"] = per_seed[0]["filter_rate"]
        agg["qp_feasible_rate"] = per_seed[0]["qp_feasible_rate"]
        agg["per_seed"] = [
            {k: m[k] for k in ["position_rmse", "min_clearance", "collisions",
                               "mean_solve_time_ms"]}
            for m in per_seed
        ]

        results[name] = {
            "agg": agg,
            "states": histories[0]["states"],       # seed 0 (플롯용)
            "errors": per_seed[0]["errors"],
            "clearance_ts": per_seed[0]["clearance_ts"],
        }
        print(f"done ({elapsed:.1f}s)  RMSE={agg['position_rmse']:.3f} "
              f"MinClear={agg['min_clearance']:.3f} Col={agg['collisions']:.1f}")

    multi_seed = len(seeds) > 1
    _print_table(results, multi_seed)

    payload = {
        "scenario": scenario_key,
        "name": scenario["name"],
        "config": {
            "K": K, "N": N, "dt": DT, "duration": dur, "seeds": seeds,
            "obstacles": scenario["obstacles"],
            "process_noise_std": scenario["process_noise_std"],
        },
        "methods": {name: results[name]["agg"] for name in METHODS},
    }
    save_json(scenario_key, payload)

    if not args.no_plot:
        _plot_standard(results, scenario, scenario_key, dur, multi_seed)

    return results


def _print_table(results, multi_seed):
    print(f"\n{'=' * 110}")
    hdr = (f"{'Method':<20} {'RMSE':>12} {'MinClear':>14} {'Collisions':>12} "
           f"{'SolveMs':>9} {'RateHz':>8} {'MeanESS':>9} {'Extra':>18}")
    print(hdr)
    print("-" * 110)
    for name in METHODS:
        if name not in results:
            continue
        a = results[name]["agg"]
        if multi_seed:
            rmse = f"{a['position_rmse']:.3f}±{a['position_rmse_std']:.3f}"
            mc = f"{a['min_clearance']:.3f}±{a['min_clearance_std']:.3f}"
            col = f"{a['collisions']:.1f}±{a['collisions_std']:.1f}"
        else:
            rmse = f"{a['position_rmse']:.4f}"
            mc = f"{a['min_clearance']:.3f}"
            col = f"{a['collisions']:.0f}"
        ess = f"{a['mean_ess']:.0f}" if a["mean_ess"] is not None else "-"
        extra = ""
        if a["filter_rate"] is not None:
            extra = f"filt={a['filter_rate'] * 100:.0f}%"
        if a["qp_feasible_rate"] is not None:
            extra = f"feas={a['qp_feasible_rate'] * 100:.0f}%"
        print(f"{name:<20} {rmse:>12} {mc:>14} {col:>12} "
              f"{a['mean_solve_time_ms']:>9.1f} {a['control_rate_hz']:>8.0f} "
              f"{ess:>9} {extra:>18}")
    print(f"{'=' * 110}")


def _plot_standard(results, scenario, scenario_key, dur, multi_seed):
    trajectory_fn = scenario["trajectory_fn"]
    obstacles = scenario["obstacles"]

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    # (0,0) XY 궤적
    ax = axes[0, 0]
    t_arr = np.linspace(0, dur, 400)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, linewidth=1, label="Ref")
    for name in METHODS:
        st = results[name]["states"]
        ax.plot(st[:, 0], st[:, 1], color=COLORS[name], linewidth=1.2,
                alpha=0.85, label=SHORT[name])
    for ox, oy, r in obstacles:
        ax.add_patch(Circle((ox, oy), r, facecolor="#FF5252", edgecolor="red",
                            alpha=0.4, linewidth=1.5))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories (seed 0)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # (0,1) 추적 오차
    ax = axes[0, 1]
    for name in METHODS:
        errs = results[name]["errors"]
        ax.plot(np.arange(len(errs)) * DT, errs, color=COLORS[name],
                linewidth=1, label=SHORT[name])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error"); ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (0,2) 클리어런스 시계열
    ax = axes[0, 2]
    for name in METHODS:
        c = results[name]["clearance_ts"]
        ax.plot(np.arange(len(c)) * DT, c, color=COLORS[name],
                linewidth=1, label=SHORT[name])
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Clearance (m)")
    ax.set_title("Min Obstacle Clearance over Time")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    names_s = [SHORT[n] for n in METHODS]
    colors = [COLORS[n] for n in METHODS]

    def _bar(ax, key, title, ylabel, fmt="{:.3f}"):
        vals = [results[n]["agg"][key] for n in METHODS]
        errs = ([results[n]["agg"][key + "_std"] for n in METHODS]
                if multi_seed else None)
        bars = ax.bar(names_s, vals, color=colors, alpha=0.85,
                      yerr=errs, capsize=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0),
                    fmt.format(v), ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=7, rotation=45)

    _bar(axes[0, 3], "position_rmse", "Position RMSE", "RMSE (m)")
    _bar(axes[1, 0], "min_clearance", "Minimum Clearance", "Min Clearance (m)")
    axes[1, 0].axhline(0.0, color="red", linestyle="--", alpha=0.6)
    _bar(axes[1, 1], "collisions", "Collision Steps (dist < r_obs)",
         "Collision steps", fmt="{:.0f}")
    _bar(axes[1, 2], "mean_solve_time_ms", "Mean Solve Time", "ms", fmt="{:.1f}")

    # (1,3) 요약 텍스트
    ax = axes[1, 3]
    ax.axis("off")
    lines = [f"{scenario['name']}", f"K={K} N={N} dt={DT} dur={dur}s", ""]
    lines.append(f"{'Method':<10} {'RMSE':>7} {'MinCl':>7} {'Col':>5}")
    for name in METHODS:
        a = results[name]["agg"]
        lines.append(f"{SHORT[name]:<10} {a['position_rmse']:>7.3f} "
                     f"{a['min_clearance']:>7.3f} {a['collisions']:>5.0f}")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", fontsize=9, family="monospace")

    plt.suptitle(f"cbfkit-inspired Safety Benchmark [{scenario_key}]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"cbfkit_inspired_benchmark_{scenario_key}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out}")


# ── risk_sweep 시나리오 ────────────────────────────────────────

def run_risk_sweep(args):
    scenario = get_scenarios()["risk_sweep"]
    seeds = scenario["seeds"]
    rho_values = scenario["rho_values"]
    dur = args.duration if args.duration else scenario["duration"]

    print(f"\n{'=' * 78}")
    print(f"  cbfkit-inspired Benchmark — {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  rho values: {rho_values} | seeds={seeds} | duration={dur}s")
    print(f"{'=' * 78}")

    def _run_multi(name, rho=None):
        per_seed = []
        first_hist = None
        for seed in seeds:
            hist = run_single(name, scenario, seed, duration=dur, rho_override=rho)
            m = compute_metrics(hist, scenario)
            per_seed.append(m)
            if first_hist is None:
                first_hist = hist
        agg = {}
        for key in ["position_rmse", "min_clearance", "collisions",
                    "mean_solve_time_ms"]:
            vals = np.array([m[key] for m in per_seed], dtype=float)
            agg[key] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        return agg, first_hist

    sweep = {}
    for i, rho in enumerate(rho_values):
        print(f"  [{i + 1}/{len(rho_values)}] RiskAwareCBF rho={rho:<5}",
              end=" ", flush=True)
        t0 = time.time()
        agg, hist = _run_multi("RiskAwareCBF-MPPI", rho=rho)
        sweep[rho] = {"agg": agg, "states": hist["states"]}
        print(f"done ({time.time() - t0:.1f}s)  "
              f"RMSE={agg['position_rmse']:.3f}±{agg['position_rmse_std']:.3f} "
              f"MinClear={agg['min_clearance']:.3f}±{agg['min_clearance_std']:.3f} "
              f"Col={agg['collisions']:.1f}")

    refs = {}
    for name in ["Vanilla", "StochasticCBF-MPPI"]:
        print(f"  [ref] {name:<24}", end=" ", flush=True)
        t0 = time.time()
        agg, _ = _run_multi(name)
        refs[name] = agg
        print(f"done ({time.time() - t0:.1f}s)  "
              f"RMSE={agg['position_rmse']:.3f} "
              f"MinClear={agg['min_clearance']:.3f}±{agg['min_clearance_std']:.3f} "
              f"Col={agg['collisions']:.1f}")

    # 테이블
    print(f"\n{'=' * 92}")
    print(f"{'rho':>8} {'RMSE':>16} {'MinClearance':>18} {'Collisions':>14} "
          f"{'SolveMs':>10}")
    print("-" * 92)
    for rho in rho_values:
        a = sweep[rho]["agg"]
        print(f"{rho:>8} "
              f"{a['position_rmse']:>9.3f}±{a['position_rmse_std']:<5.3f} "
              f"{a['min_clearance']:>11.3f}±{a['min_clearance_std']:<5.3f} "
              f"{a['collisions']:>8.1f}±{a['collisions_std']:<4.1f} "
              f"{a['mean_solve_time_ms']:>10.1f}")
    print("-" * 92)
    for name, a in refs.items():
        print(f"{SHORT[name]:>8} "
              f"{a['position_rmse']:>9.3f}±{a['position_rmse_std']:<5.3f} "
              f"{a['min_clearance']:>11.3f}±{a['min_clearance_std']:<5.3f} "
              f"{a['collisions']:>8.1f}±{a['collisions_std']:<4.1f} "
              f"{a['mean_solve_time_ms']:>10.1f}")
    print(f"{'=' * 92}")

    payload = {
        "scenario": "risk_sweep",
        "name": scenario["name"],
        "config": {
            "K": K, "N": N, "dt": DT, "duration": dur, "seeds": seeds,
            "obstacles": scenario["obstacles"],
            "process_noise_std": scenario["process_noise_std"],
            "rho_values": rho_values,
        },
        "sweep": {str(rho): sweep[rho]["agg"] for rho in rho_values},
        "references": refs,
    }
    save_json("risk_sweep", payload)

    if not args.no_plot:
        _plot_risk_sweep(sweep, refs, scenario, rho_values, dur)

    return sweep, refs


def _plot_risk_sweep(sweep, refs, scenario, rho_values, dur):
    obstacles = scenario["obstacles"]
    trajectory_fn = scenario["trajectory_fn"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho_values)))

    # (0,0) 궤적 (seed 0)
    ax = axes[0, 0]
    t_arr = np.linspace(0, dur, 400)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, linewidth=1, label="Ref")
    for c, rho in zip(cmap, rho_values):
        st = sweep[rho]["states"]
        ax.plot(st[:, 0], st[:, 1], color=c, linewidth=1.3, alpha=0.9,
                label=f"rho={rho}")
    for ox, oy, r in obstacles:
        ax.add_patch(Circle((ox, oy), r, facecolor="#FF5252", edgecolor="red",
                            alpha=0.4))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("RiskAwareCBF Trajectories by rho (seed 42)")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    def _curve(ax, key, title, ylabel):
        vals = [sweep[r]["agg"][key] for r in rho_values]
        errs = [sweep[r]["agg"][key + "_std"] for r in rho_values]
        ax.errorbar(rho_values, vals, yerr=errs, marker="o", capsize=4,
                    color="#e377c2", linewidth=2, label="RiskAwareCBF")
        for name, ls in [("Vanilla", ":"), ("StochasticCBF-MPPI", "--")]:
            a = refs[name]
            ax.axhline(a[key], color=COLORS[name], linestyle=ls, alpha=0.8,
                       label=f"{SHORT[name]} ref")
            ax.fill_between([min(rho_values), max(rho_values)],
                            a[key] - a[key + "_std"], a[key] + a[key + "_std"],
                            color=COLORS[name], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("risk budget rho (log)")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.invert_xaxis()  # 왼→오 = 보수적
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    _curve(axes[0, 1], "min_clearance", "rho vs Min Clearance (mean±std)",
           "Min Clearance (m)")
    axes[0, 1].axhline(0.0, color="red", linestyle="-", alpha=0.5)
    _curve(axes[1, 0], "position_rmse", "rho vs Position RMSE (mean±std)",
           "RMSE (m)")
    _curve(axes[1, 1], "collisions", "rho vs Collision Steps (mean±std)",
           "Collision steps")

    plt.suptitle("cbfkit-inspired Benchmark [risk_sweep] — "
                 "Risk Budget vs Safety/Performance Tradeoff",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "cbfkit_inspired_benchmark_risk_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="cbfkit-inspired safety techniques benchmark")
    parser.add_argument(
        "--scenario", default="static_kin",
        choices=["static_kin", "dynamic_rd2", "stochastic", "risk_sweep"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override scenario duration (s)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    keys = (["static_kin", "dynamic_rd2", "stochastic", "risk_sweep"]
            if args.all_scenarios else [args.scenario])

    for key in keys:
        if key == "risk_sweep":
            run_risk_sweep(args)
        else:
            run_standard_scenario(key, args)


if __name__ == "__main__":
    main()
