#!/usr/bin/env python3
"""
Uncertainty-Aware MPPI 벤치마크: 5-Way × 4 시나리오

방법:
  1. Vanilla MPPI                     — 고정 sigma (기준선)
  2. Uncertainty MPPI (previous_traj)  — 직전 궤적 기반 적응 (비용 0)
  3. Uncertainty MPPI (current_state)  — 현재 상태 기반 전역 스케일
  4. Uncertainty MPPI (two_pass)       — 2-패스 적응 (최고 정확도)
  5. Uncertainty MPPI + UncertaintyCost — 이중 효과 (적응 샘플링 + 비용)

시나리오:
  1. clean     — 외란 없음 (기준선)
  2. noisy     — 중간 프로세스 노이즈
  3. mismatch  — 위치 의존 체계적 모델 오차 (핵심 시나리오)
  4. combined  — 모델 오차 + 프로세스 노이즈

Usage:
    PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py
    PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py --live
    PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py --scenario mismatch
    PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py --trajectory figure8
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
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.uncertainty_mppi import (
    UncertaintyMPPIController,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi.uncertainty_cost import UncertaintyAwareCost
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
    figure_eight_trajectory,
)
from mppi_controller.simulation.harness import SimulationHarness


# ── 모델 불일치 시뮬레이션 ───────────────────────────────────

class MismatchModel:
    """
    위치 의존 모델 불일치를 가진 로봇 모델 래퍼 (실제 환경 시뮬레이션)

    MPPI 컨트롤러는 순수 기구학 모델을 사용하지만,
    실제 환경에서는 위치에 따라 체계적 바이어스가 존재:
    - y > 0 영역: "바람" 효과로 x 방향 드리프트 증가
    - 원점에서 멀어질수록: "노면 마찰" 변화로 속도 감쇠
    - 특정 각도 범위: heading 바이어스
    """

    def __init__(self, nominal_model, drift_scale=0.15, friction_scale=0.08):
        self._nominal = nominal_model
        self.drift_scale = drift_scale
        self.friction_scale = friction_scale
        self.state_dim = nominal_model.state_dim
        self.control_dim = nominal_model.control_dim

    def step(self, state, control, dt):
        next_state = self._nominal.step(state, control, dt).copy()

        x, y = state[0], state[1]
        dist = np.sqrt(x**2 + y**2)

        # 1. 위치 의존 바람 (y > 0에서 x방향 drift)
        wind_x = self.drift_scale * max(0, y) * 0.15 * dt
        wind_y = self.drift_scale * np.sin(x * 0.3) * 0.1 * dt

        # 2. 거리 기반 마찰 (멀수록 속도 감쇠)
        friction = self.friction_scale * dist * 0.05 * dt
        v = control[0] if len(control) > 0 else 0
        friction_x = -friction * np.cos(state[2]) * np.sign(v)
        friction_y = -friction * np.sin(state[2]) * np.sign(v)

        # 3. heading 바이어스 (특정 사분면)
        heading_bias = self.drift_scale * 0.02 * np.sin(2 * state[2]) * dt

        next_state[0] += wind_x + friction_x
        next_state[1] += wind_y + friction_y
        next_state[2] += heading_bias

        return next_state

    def get_control_bounds(self):
        return self._nominal.get_control_bounds()

    def normalize_state(self, state):
        return self._nominal.normalize_state(state)


# ── 불확실성 모델 ────────────────────────────────────────────

class PositionDependentUncertainty:
    """
    MismatchModel의 바이어스 패턴을 반영하는 불확실성 모델

    실제로는 학습된 모델(Ensemble/GP)이 데이터로부터 이 패턴을 학습하지만,
    벤치마크에서는 ground truth 바이어스 패턴의 근사치를 사용.
    """

    def __init__(self, drift_scale=0.15, friction_scale=0.08):
        self.drift_scale = drift_scale
        self.friction_scale = friction_scale

    def __call__(self, states, controls):
        if states.ndim == 1:
            states = states[None, :]
        if controls.ndim == 1:
            controls = controls[None, :]

        nx = states.shape[-1]
        batch = states.shape[0]
        std = np.zeros((batch, nx))

        x, y = states[:, 0], states[:, 1]
        dist = np.sqrt(x**2 + y**2)

        # x 불확실성: y > 0 영역 + 거리 기반
        std[:, 0] = (
            0.01
            + self.drift_scale * np.maximum(0, y) * 0.15
            + self.friction_scale * dist * 0.05
        )
        # y 불확실성: sin 패턴 + 거리 기반
        std[:, 1] = (
            0.01
            + self.drift_scale * np.abs(np.sin(x * 0.3)) * 0.1
            + self.friction_scale * dist * 0.05
        )
        # heading 불확실성
        std[:, 2] = 0.005 + self.drift_scale * 0.02 * np.abs(np.sin(2 * states[:, 2]))

        return std


class ConstantUncertainty:
    """균일 불확실성 (noisy 시나리오용)"""

    def __init__(self, std_val=0.05):
        self.std_val = std_val

    def __call__(self, states, controls):
        if states.ndim == 1:
            states = states[None, :]
        nx = states.shape[-1]
        return np.full((states.shape[0], nx), self.std_val)


# ── 시나리오 설정 ────────────────────────────────────────────

def get_scenarios():
    return {
        "clean": {
            "name": "Clean (no disturbance)",
            "use_mismatch": False,
            "process_noise_std": None,
            "unc_model": ConstantUncertainty(0.02),
        },
        "noisy": {
            "name": "Noisy (i.i.d. noise)",
            "use_mismatch": False,
            "process_noise_std": np.array([0.03, 0.03, 0.01]),
            "unc_model": ConstantUncertainty(0.05),
        },
        "mismatch": {
            "name": "Model Mismatch (systematic bias)",
            "use_mismatch": True,
            "mismatch_kwargs": {"drift_scale": 0.20, "friction_scale": 0.10},
            "process_noise_std": None,
            "unc_model": PositionDependentUncertainty(0.20, 0.10),
        },
        "combined": {
            "name": "Combined (mismatch + noise)",
            "use_mismatch": True,
            "mismatch_kwargs": {"drift_scale": 0.15, "friction_scale": 0.08},
            "process_noise_std": np.array([0.02, 0.02, 0.005]),
            "unc_model": PositionDependentUncertainty(0.15, 0.08),
        },
    }


def create_trajectory_fn(name):
    if name == "circle":
        return circle_trajectory
    elif name == "figure8":
        return figure_eight_trajectory
    return circle_trajectory


# ── 커스텀 시뮬레이터 (모델 불일치 지원) ─────────────────────

class MismatchSimulator:
    """
    MPPI 컨트롤러에는 명목 모델을, 실제 전파에는 mismatch 모델을 사용하는 시뮬레이터.
    Simulator 클래스와 동일 인터페이스, 다만 propagation용 모델이 분리됨.
    """

    def __init__(self, real_model, controller, dt, process_noise_std=None):
        self.real_model = real_model
        self.controller = controller
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.history = {k: [] for k in ["time", "state", "control", "reference", "solve_time", "info"]}
        self.state = None
        self.t = 0.0

    def reset(self, initial_state):
        self.state = initial_state.copy()
        self.t = 0.0
        self.history = {k: [] for k in self.history}
        self.controller.reset()

    def run(self, reference_fn, duration, realtime=False):
        num_steps = int(duration / self.dt)
        for _ in range(num_steps):
            ref = reference_fn(self.t)

            t0 = time.time()
            control, info = self.controller.compute_control(self.state, ref)
            solve_time = time.time() - t0

            # 실제 모델로 전파 (MPPI가 모르는 바이어스 포함)
            next_state = self.real_model.step(self.state, control, self.dt)

            if self.process_noise_std is not None:
                next_state += np.random.normal(0, self.process_noise_std)

            next_state = self.real_model.normalize_state(next_state)

            self.history["time"].append(self.t)
            self.history["state"].append(self.state.copy())
            self.history["control"].append(control.copy())
            self.history["reference"].append(ref[0].copy())
            self.history["solve_time"].append(solve_time)
            self.history["info"].append(info)

            self.state = next_state
            self.t += self.dt

            if realtime:
                time.sleep(max(0, self.dt - solve_time))

        return {k: (np.array(v) if k != "info" and v else v) for k, v in self.history.items()}


# ── 벤치마크 실행 ────────────────────────────────────────────

def run_benchmark(args):
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    print(f"\n{'=' * 72}")
    print(f"  Uncertainty-Aware MPPI Benchmark")
    print(f"  Scenario: {scenario['name']}")
    print(f"  Trajectory: {args.trajectory} | Duration: {args.duration}s | Seed: {args.seed}")
    print(f"{'=' * 72}")

    # 모델
    nominal_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    trajectory_fn = create_trajectory_fn(args.trajectory)
    initial_state = trajectory_fn(0.0)
    unc_model = scenario["unc_model"]

    # 실제 모델 (모델 불일치 포함 여부)
    if scenario["use_mismatch"]:
        real_model = MismatchModel(nominal_model, **scenario.get("mismatch_kwargs", {}))
    else:
        real_model = nominal_model

    # 공통 MPPI 파라미터
    common = dict(
        K=256, N=20, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )

    # ── 5가지 방법 ──
    variants = [
        {
            "name": "Vanilla MPPI",
            "short": "Vanilla",
            "color": "#4285f4",
            "make": lambda: MPPIController(nominal_model, MPPIParams(**common)),
        },
        {
            "name": "Unc (prev_traj)",
            "short": "PrevTraj",
            "color": "#ea4335",
            "make": lambda: UncertaintyMPPIController(
                nominal_model,
                UncertaintyMPPIParams(**common, exploration_factor=1.5,
                                      uncertainty_strategy="previous_trajectory"),
                uncertainty_fn=unc_model,
            ),
        },
        {
            "name": "Unc (cur_state)",
            "short": "CurState",
            "color": "#fbbc04",
            "make": lambda: UncertaintyMPPIController(
                nominal_model,
                UncertaintyMPPIParams(**common, exploration_factor=1.5,
                                      uncertainty_strategy="current_state"),
                uncertainty_fn=unc_model,
            ),
        },
        {
            "name": "Unc (two_pass)",
            "short": "TwoPass",
            "color": "#34a853",
            "make": lambda: UncertaintyMPPIController(
                nominal_model,
                UncertaintyMPPIParams(**common, exploration_factor=1.5,
                                      uncertainty_strategy="two_pass"),
                uncertainty_fn=unc_model,
            ),
        },
    ]

    # Dual: 적응 샘플링 + 불확실성 비용
    def make_dual():
        p = UncertaintyMPPIParams(**common, exploration_factor=1.5,
                                  uncertainty_strategy="previous_trajectory")
        cost = CompositeMPPICost([
            StateTrackingCost(p.Q), TerminalCost(p.Qf), ControlEffortCost(p.R),
            UncertaintyAwareCost(uncertainty_fn=unc_model, beta=5.0),
        ])
        return UncertaintyMPPIController(
            nominal_model, p, cost_function=cost, uncertainty_fn=unc_model,
        )

    variants.append({
        "name": "Unc + UncCost",
        "short": "Dual",
        "color": "#9c27b0",
        "make": make_dual,
    })

    # ── 실행 ──
    results = []
    for i, var in enumerate(variants):
        np.random.seed(args.seed)

        print(f"  [{i+1}/{len(variants)}] {var['name']:<22}", end=" ", flush=True)
        t_start = time.time()

        controller = var["make"]()

        if scenario["use_mismatch"]:
            sim = MismatchSimulator(
                real_model, controller, common["dt"],
                process_noise_std=scenario["process_noise_std"],
            )
        else:
            sim = Simulator(
                nominal_model, controller, common["dt"],
                process_noise_std=scenario["process_noise_std"],
                store_info=True,
            )
        sim.reset(initial_state.copy())

        ref_fn = lambda t, _fn=trajectory_fn, _N=common["N"], _dt=common["dt"]: \
            generate_reference_trajectory(_fn, t, _N, _dt)

        history = sim.run(ref_fn, args.duration, realtime=args.live)
        elapsed = time.time() - t_start
        metrics = compute_metrics(history)

        unc_stats = getattr(controller, "get_uncertainty_statistics", lambda: None)()

        results.append({
            "name": var["name"],
            "short": var["short"],
            "color": var["color"],
            "history": history,
            "metrics": metrics,
            "unc_stats": unc_stats,
            "elapsed": elapsed,
        })

        print(
            f"RMSE={metrics['position_rmse']:.4f}m  "
            f"MaxErr={metrics['max_position_error']:.4f}m  "
            f"solve={metrics['mean_solve_time']:.1f}ms  "
            f"({elapsed:.1f}s)"
        )

    print_results_table(results, scenario["name"])

    if not args.no_plot:
        plot_results(results, scenario, args)

    return results


# ── 결과 출력 ────────────────────────────────────────────────

def print_results_table(results, scenario_name):
    w = 92
    print(f"\n{'=' * w}")
    print(f"  Result — {scenario_name}")
    print(f"{'=' * w}")
    print(
        f"  {'Method':<22} | {'RMSE':>9} | {'MaxErr':>9} | "
        f"{'Heading':>9} | {'CtrlRate':>9} | {'Solve':>9}"
    )
    print(f"  {'-' * (w - 2)}")

    best_rmse = min(r["metrics"]["position_rmse"] for r in results)

    for r in results:
        m = r["metrics"]
        star = "*" if abs(m["position_rmse"] - best_rmse) < 1e-6 else " "
        print(
            f" {star}{r['name']:<22} | {m['position_rmse']:>8.4f}m | "
            f"{m['max_position_error']:>8.4f}m | {m['heading_rmse']:>8.4f}r | "
            f"{m['control_rate']:>9.4f} | {m['mean_solve_time']:>8.1f}ms"
        )

    vanilla = results[0]["metrics"]
    print(f"\n  {'Improvement vs Vanilla':^{w - 2}}")
    print(f"  {'-' * (w - 2)}")
    for r in results[1:]:
        m = r["metrics"]
        dr = (vanilla["position_rmse"] - m["position_rmse"]) / vanilla["position_rmse"] * 100
        dm = (vanilla["max_position_error"] - m["max_position_error"]) / vanilla["max_position_error"] * 100
        sr = m["mean_solve_time"] / max(vanilla["mean_solve_time"], 0.01)
        print(f"  {r['name']:<22} | RMSE {dr:>+6.1f}% | MaxErr {dm:>+6.1f}% | Solve {sr:.2f}x")

    print(f"{'=' * w}")


# ── 시각화 ────────────────────────────────────────────────────

def plot_results(results, scenario, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)
    fig.suptitle(
        f"Uncertainty-Aware MPPI — {scenario['name']} / {args.trajectory}",
        fontsize=14, fontweight="bold",
    )

    # 1. XY 궤적
    ax = fig.add_subplot(gs[0, 0])
    ref = results[0]["history"]["reference"]
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.5, alpha=0.4, label="Ref")
    for r in results:
        s = r["history"]["state"]
        ax.plot(s[:, 0], s[:, 1], color=r["color"], lw=1.3, alpha=0.85, label=r["short"])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("XY Trajectory"); ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # 2. RMSE 바
    ax = fig.add_subplot(gs[0, 1])
    names = [r["short"] for r in results]
    rmses = [r["metrics"]["position_rmse"] for r in results]
    colors = [r["color"] for r in results]
    bars = ax.barh(names, rmses, color=colors, alpha=0.8)
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_width() + max(rmses) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}m", va="center", fontsize=8)
    ax.set_xlabel("Position RMSE (m)"); ax.set_title("Tracking Accuracy"); ax.grid(alpha=0.3, axis="x")

    # 3. Solve Time 바
    ax = fig.add_subplot(gs[0, 2])
    st = [r["metrics"]["mean_solve_time"] for r in results]
    bars = ax.barh(names, st, color=colors, alpha=0.8)
    for bar, v in zip(bars, st):
        ax.text(bar.get_width() + max(st) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}ms", va="center", fontsize=8)
    ax.set_xlabel("Mean Solve Time (ms)"); ax.set_title("Computational Cost"); ax.grid(alpha=0.3, axis="x")

    # 4. Error 시계열
    ax = fig.add_subplot(gs[1, 0])
    for r in results:
        s, rf, t = r["history"]["state"], r["history"]["reference"], r["history"]["time"]
        ax.plot(t, np.linalg.norm(s[:, :2] - rf[:, :2], axis=1),
                color=r["color"], lw=1.0, alpha=0.8, label=r["short"])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pos Error (m)")
    ax.set_title("Position Error Over Time"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 5. Sigma Ratio 시계열
    ax = fig.add_subplot(gs[1, 1])
    has_data = False
    for r in results:
        infos = r["history"].get("info", [])
        if not infos:
            continue
        ratios = [i.get("sigma_stats", {}).get("mean_ratio")
                  for i in infos if i.get("sigma_stats", {}).get("has_profile")]
        if ratios:
            has_data = True
            ax.plot(r["history"]["time"][:len(ratios)], ratios,
                    color=r["color"], lw=1.0, alpha=0.8, label=r["short"])
    if has_data:
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Sigma Ratio")
        ax.set_title("Adaptive Noise Scale"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No profile data", ha="center", va="center",
                transform=ax.transAxes, color="gray"); ax.set_title("Adaptive Noise Scale")

    # 6. Uncertainty 시계열
    ax = fig.add_subplot(gs[1, 2])
    has_data = False
    for r in results:
        infos = r["history"].get("info", [])
        if not infos:
            continue
        vals = [i.get("uncertainty_stats", {}).get("mean_uncertainty", 0)
                for i in infos]
        nonzero = [v for v in vals if v > 0]
        if nonzero:
            has_data = True
            ax.plot(r["history"]["time"][:len(vals)], vals,
                    color=r["color"], lw=1.0, alpha=0.8, label=r["short"])
    if has_data:
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Mean Uncertainty")
        ax.set_title("Model Uncertainty"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray"); ax.set_title("Model Uncertainty")

    # 7. Control v
    ax = fig.add_subplot(gs[2, 0])
    for r in results:
        ax.plot(r["history"]["time"], r["history"]["control"][:, 0],
                color=r["color"], lw=0.8, alpha=0.7)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("v (m/s)"); ax.set_title("Linear Velocity"); ax.grid(alpha=0.3)

    # 8. Accuracy vs Speed
    ax = fig.add_subplot(gs[2, 1])
    for r in results:
        m = r["metrics"]
        ax.scatter(m["mean_solve_time"], m["position_rmse"],
                   s=200, color=r["color"], alpha=0.8, edgecolors="k", lw=0.5, zorder=5)
        ax.annotate(r["short"], (m["mean_solve_time"], m["position_rmse"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7)
    ax.set_xlabel("Solve Time (ms)"); ax.set_ylabel("RMSE (m)")
    ax.set_title("Accuracy vs Speed"); ax.grid(alpha=0.3)

    # 9. Summary
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    best_i = int(np.argmin(rmses))
    van_rmse = results[0]["metrics"]["position_rmse"]
    lines = [
        f"Scenario: {scenario['name']}",
        f"Trajectory: {args.trajectory}  Duration: {args.duration}s",
        f"K={256}, N={20}, seed={args.seed}",
        "",
        f"Best: {results[best_i]['name']}",
        f"  RMSE = {rmses[best_i]:.4f}m",
        f"  vs Vanilla: {(van_rmse - rmses[best_i]) / van_rmse * 100:+.1f}%",
        "",
        "Strategy cost:",
        "  prev_traj:  0 extra rollout",
        "  cur_state:  1 forward pass",
        "  two_pass:   2x rollout",
        "  dual:       adaptive + cost penalty",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, family="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fname = f"uncertainty_mppi_{args.scenario}_{args.trajectory}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {fname}")
    plt.close()


# ── 전체 시나리오 요약 ───────────────────────────────────────

def print_overall_summary(all_results):
    w = 100
    print(f"\n{'=' * w}")
    print(f"  Overall Summary (Position RMSE)")
    print(f"{'=' * w}")

    header = f"  {'Method':<22}"
    for s in all_results:
        header += f" | {s:>12}"
    header += f" | {'Average':>10}"
    print(header)
    print(f"  {'-' * (w - 2)}")

    method_names = [r["name"] for r in list(all_results.values())[0]]
    for i, name in enumerate(method_names):
        row = f"  {name:<22}"
        vals = []
        for results in all_results.values():
            v = results[i]["metrics"]["position_rmse"]
            vals.append(v)
            row += f" | {v:>11.4f}m"
        row += f" | {np.mean(vals):>9.4f}m"
        print(row)

    # 개선율
    print(f"\n  {'Improvement vs Vanilla (%)':^{w - 2}}")
    print(f"  {'-' * (w - 2)}")
    for i in range(1, len(method_names)):
        row = f"  {method_names[i]:<22}"
        for results in all_results.values():
            van = results[0]["metrics"]["position_rmse"]
            cur = results[i]["metrics"]["position_rmse"]
            pct = (van - cur) / van * 100
            row += f" | {pct:>+11.1f}%"
        print(row)

    print(f"{'=' * w}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Uncertainty-Aware MPPI Benchmark")
    parser.add_argument("--scenario", choices=["clean", "noisy", "mismatch", "combined"],
                        default="mismatch")
    parser.add_argument("--trajectory", choices=["circle", "figure8"], default="circle")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--all-scenarios", action="store_true")

    args = parser.parse_args()

    if args.all_scenarios:
        all_results = {}
        for key in get_scenarios():
            args.scenario = key
            all_results[key] = run_benchmark(args)
        print_overall_summary(all_results)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
