#!/usr/bin/env python3
"""
Shield-DIAL-MPPI 4종 비교 벤치마크

Vanilla MPPI / DIAL-MPPI / Shield-DIAL-MPPI / AdaptiveShield-DIAL-MPPI
밀집 장애물 슬라럼 + 바람 외란 → Shield 안전 보장의 가치를 검증.

시나리오:
  - 직선 레퍼런스 (y=0) 위에 8개 장애물 밀집 배치
  - 횡방향 바람 외란 (sin 패턴, 로봇을 장애물 쪽으로 밀어냄)
  - 컨트롤러는 외란을 모름 (model mismatch)
  → Vanilla/DIAL: 비용 기반 회피 시도하지만 외란으로 안전 위반 발생
  → Shield/Adaptive: CBF가 hard 제약으로 안전 보장

cbf_safety_margin 파라미터:
  장애물 표면에서 최소 이격 거리 (m)
  effective_r = obstacle_r + safety_margin
  h(x) = dist² - effective_r² ≥ 0 보장

Usage:
    PYTHONPATH=. python examples/comparison/shield_dial_mppi_benchmark.py --live
    PYTHONPATH=. python examples/comparison/shield_dial_mppi_benchmark.py --live --K 512 --duration 20
    PYTHONPATH=. python examples/comparison/shield_dial_mppi_benchmark.py --live --margin 0.3
"""

import matplotlib
if "--live" not in __import__("sys").argv:
    matplotlib.use("Agg")

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    DIALMPPIParams,
    ShieldDIALMPPIParams,
    AdaptiveShieldDIALMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.shield_dial_mppi import ShieldDIALMPPIController
from mppi_controller.controllers.mppi.adaptive_shield_dial_mppi import (
    AdaptiveShieldDIALMPPIController,
)
from mppi_controller.controllers.mppi.cost_functions import (
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
    CompositeMPPICost,
)
import matplotlib.pyplot as plt


# ── 궤적 ─────────────────────────────────────────────────

def straight_line(t: float) -> np.ndarray:
    return np.array([0.5 * t, 0.0, 0.0])


def generate_ref(t: float, N: int, dt: float) -> np.ndarray:
    ref = np.zeros((N + 1, 3))
    for i in range(N + 1):
        ref[i] = straight_line(t + i * dt)
    return ref


# ── 외란 모델 ────────────────────────────────────────────

def wind_disturbance(state: np.ndarray, t: float, strength: float) -> np.ndarray:
    """
    횡방향 바람 외란 (컨트롤러가 모르는 외부 힘)

    y 방향으로 sin 패턴 + 노이즈 → 로봇을 장애물 쪽으로 밀어냄
    """
    dx = 0.0
    dy = strength * np.sin(0.8 * t) + strength * 0.3 * np.random.randn()
    dtheta = strength * 0.1 * np.random.randn()
    return np.array([dx, dy, dtheta])


# ── 비용 함수 ────────────────────────────────────────────

def make_cost(obstacles, Q, R, Qf):
    return CompositeMPPICost([
        StateTrackingCost(Q),
        TerminalCost(Qf),
        ControlEffortCost(R),
        ObstacleCost(obstacles, safety_margin=0.15, cost_weight=100.0),
    ])


# ── 시뮬레이션 ───────────────────────────────────────────

def precompute_wind(duration, dt, strength, seed=42):
    """외란 시퀀스 사전 생성 (모든 컨트롤러에 동일 적용)"""
    rng = np.random.RandomState(seed + 1000)  # 별도 시드
    n_steps = int(duration / dt)
    winds = np.zeros((n_steps, 3))
    for step in range(n_steps):
        t = step * dt
        dy = strength * np.sin(0.8 * t) + strength * 0.3 * rng.randn()
        dtheta = strength * 0.1 * rng.randn()
        winds[step] = np.array([0.0, dy, dtheta]) * dt
    return winds


def run_sim(model, controller, state0, obstacles, duration, dt, N,
            winds=None):
    """외란 포함 시뮬레이션 (사전 생성된 외란 사용)"""
    n_steps = int(duration / dt)
    states = [state0.copy()]
    controls = []
    state = state0.copy()

    for step in range(n_steps):
        t = step * dt
        ref = generate_ref(t, N, dt)
        control, info = controller.compute_control(state, ref)

        # 모델 전파 + 외란 (컨트롤러는 이 외란을 모름)
        state = model.step(state, control, dt)
        if winds is not None:
            state += winds[step]

        states.append(state.copy())
        controls.append(control.copy())

    states = np.array(states)
    controls = np.array(controls)
    times = np.arange(len(states)) * dt
    refs = np.array([straight_line(t) for t in times])
    errors = np.linalg.norm(states[:, :2] - refs[:, :2], axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    # barrier & violations
    min_barrier = np.inf
    violations = 0
    for ox, oy, orr in obstacles:
        h = (states[:, 0] - ox)**2 + (states[:, 1] - oy)**2 - orr**2
        min_barrier = min(min_barrier, np.min(h))
        violations += int(np.sum(np.sqrt((states[:, 0]-ox)**2 + (states[:, 1]-oy)**2) < orr))

    return dict(states=states, controls=controls, times=times, refs=refs,
                errors=errors, rmse=rmse, min_barrier=min_barrier,
                violations=violations)


# ── 라이브 시뮬레이션 ────────────────────────────────────

def run_live(models, ctrls, names, colors, state0, obstacles, duration, dt, N,
             winds, safety_margin):
    n_steps = int(duration / dt)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Shield-DIAL-MPPI Benchmark  |  wind disturbance  "
        f"safety_margin={safety_margin:.2f}m",
        fontsize=15, fontweight="bold",
    )
    plt.ion()

    S = {n: [state0.copy()] for n in names}
    C = {n: [] for n in names}
    E = {n: [] for n in names}
    B = {n: [] for n in names}
    st = {n: state0.copy() for n in names}

    for step in range(n_steps):
        t = step * dt
        wind = winds[step] if winds is not None else np.zeros(3)

        for name, mdl, ctrl in zip(names, models, ctrls):
            ref = generate_ref(t, N, dt)
            u, _ = ctrl.compute_control(st[name], ref)
            st[name] = mdl.step(st[name], u, dt) + wind
            S[name].append(st[name].copy())
            C[name].append(u.copy())
            E[name].append(np.linalg.norm(st[name][:2] - straight_line(t+dt)[:2]))
            min_h = min(
                (st[name][0]-ox)**2 + (st[name][1]-oy)**2 - orr**2
                for ox, oy, orr in obstacles
            )
            B[name].append(min_h)

        if step % 10 == 0 or step == n_steps - 1:
            for row in axes:
                for ax in row:
                    ax.clear()
            et = np.arange(1, step+2) * dt

            # 1. XY
            ax = axes[0, 0]
            for ox, oy, orr in obstacles:
                ax.add_patch(plt.Circle((ox, oy), orr, color="gray", alpha=0.5))
                ax.add_patch(plt.Circle((ox, oy), orr + safety_margin,
                             color="red", alpha=0.1, linestyle="--"))
            rl = np.array([straight_line(i*dt) for i in range(step+2)])
            ax.plot(rl[:, 0], rl[:, 1], "k--", lw=2, alpha=.4, label="Ref")
            for n, c in zip(names, colors):
                s = np.array(S[n])
                ax.plot(s[:, 0], s[:, 1], color=c, lw=2, alpha=.8, label=n)
                ax.plot(s[-1, 0], s[-1, 1], "o", color=c, ms=7)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_title(f"XY Trajectory (t={t:.1f}s)")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=.3); ax.set_aspect("equal")
            ax.set_xlim(-0.5, max(12, t*0.5+2)); ax.set_ylim(-2.5, 2.5)

            # 2. Error
            ax = axes[0, 1]
            for n, c in zip(names, colors):
                ax.plot(et, E[n], color=c, lw=1.5, label=n)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
            ax.set_title("Position Tracking Error"); ax.legend(fontsize=7)
            ax.grid(True, alpha=.3)

            # 3. Barrier
            ax = axes[0, 2]
            for n, c in zip(names, colors):
                ax.plot(et, B[n], color=c, lw=1.5, label=n)
            ax.axhline(0, color="red", ls="--", alpha=.5, label="h=0 UNSAFE")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("min h(x)")
            ax.set_title("CBF Barrier Value"); ax.legend(fontsize=7)
            ax.grid(True, alpha=.3)

            # 4. Control
            ax = axes[1, 0]
            for n, c in zip(names, colors):
                cc = np.array(C[n])
                ax.plot(et, cc[:, 0], color=c, lw=1.2, alpha=.8, label=f"{n} v")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("v (m/s)")
            ax.set_title("Linear Velocity"); ax.legend(fontsize=7)
            ax.grid(True, alpha=.3)

            # 5. RMSE bar
            ax = axes[1, 1]
            rmses = [np.sqrt(np.mean(np.array(E[n])**2)) for n in names]
            bars = ax.bar(names, rmses, color=colors, alpha=.8, ec="black")
            ax.set_ylabel("RMSE (m)"); ax.set_title("RMSE"); ax.grid(True, axis="y", alpha=.3)
            for b, r in zip(bars, rmses):
                ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                        f"{r:.3f}", ha="center", va="bottom", fontsize=9)

            # 6. Summary
            ax = axes[1, 2]; ax.axis("off")
            lines = [f"  Status  (step {step+1}/{n_steps})",
                     f"  margin={safety_margin:.2f}m",
                     "  " + "="*42, ""]
            for n in names:
                e = np.array(E[n])
                viol = sum(1 for b in B[n] if b < 0)
                lines.append(
                    f"  {n}:\n"
                    f"    RMSE:       {np.sqrt(np.mean(e**2)):.4f} m\n"
                    f"    Violations: {viol}\n"
                    f"    Min h(x):   {min(B[n]):.4f}\n")
            ax.text(0.02, 0.5, "\n".join(lines), fontsize=9, va="center",
                    family="monospace",
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=.5))
            plt.tight_layout(); plt.pause(0.001)

    plt.ioff()
    os.makedirs("plots", exist_ok=True)
    path = "plots/shield_dial_mppi_benchmark.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shield-DIAL-MPPI Benchmark")
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--wind", type=float, default=0.6,
                        help="Wind disturbance strength (0=none)")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="CBF safety margin (min distance from obstacle surface)")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Shield-DIAL-MPPI Benchmark (Complex Scenario)".center(80))
    print("="*80)

    # ── 설정 ──
    N, dt = 20, 0.05

    # 밀집 장애물 슬라럼 (8개, 경로 양쪽 교대 배치)
    obstacles = [
        (1.5,  0.35, 0.30),
        (3.0, -0.30, 0.25),
        (4.5,  0.40, 0.30),
        (6.0, -0.35, 0.30),
        (7.5,  0.30, 0.25),
        (9.0, -0.40, 0.30),
        (10.5, 0.35, 0.25),
        (12.0,-0.30, 0.30),
    ]

    common = dict(
        N=N, dt=dt, K=args.K, lambda_=1.0,
        sigma=np.array([0.4, 0.8]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.05]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )
    dial_kw = dict(
        n_diffuse_init=10, n_diffuse=5,
        traj_diffuse_factor=0.6, horizon_diffuse_factor=0.5,
        sigma_scale=1.0, use_reward_normalization=True,
    )
    shield_kw = dict(
        cbf_obstacles=obstacles, cbf_alpha=0.3,
        cbf_safety_margin=args.margin,
        shield_enabled=True, shield_cbf_alpha=2.0,
    )
    adaptive_kw = dict(
        alpha_base=1.5, alpha_dist=0.5, alpha_vel=0.1,
        k_dist=2.0, d_safe=0.4,
    )

    cost_fn = make_cost(obstacles, common["Q"], common["R"], common["Qf"])
    winds = precompute_wind(args.duration, dt, args.wind, args.seed)
    state0 = np.array([0.0, 0.0, 0.0])
    names = ["Vanilla", "DIAL", "Shield-DIAL", "Adaptive-DIAL"]
    colors = ["#4A90D9", "#7CB342", "#E53935", "#FF9800"]

    print(f"  Wind:    {args.wind:.2f}")
    print(f"  Margin:  {args.margin:.2f}m")
    print(f"  K={args.K}, N={N}, duration={args.duration}s")
    print(f"  Obstacles: {len(obstacles)}")
    print("="*80 + "\n")

    # ── Live ──
    if args.live:
        np.random.seed(args.seed)
        mdls = [DifferentialDriveKinematic(v_max=1.0, omega_max=1.0) for _ in range(4)]
        ctrls = [
            MPPIController(mdls[0], MPPIParams(**common), cost_function=cost_fn),
            DIALMPPIController(mdls[1], DIALMPPIParams(**common, **dial_kw),
                               cost_function=cost_fn),
            ShieldDIALMPPIController(
                mdls[2], ShieldDIALMPPIParams(**common, **dial_kw, **shield_kw),
                cost_function=cost_fn),
            AdaptiveShieldDIALMPPIController(
                mdls[3], AdaptiveShieldDIALMPPIParams(
                    **common, **dial_kw, **shield_kw, **adaptive_kw),
                cost_function=cost_fn),
        ]
        run_live(mdls, ctrls, names, colors, state0, obstacles,
                 args.duration, dt, N, winds, args.margin)
        return

    # ── Batch ──
    results = {}
    for i, (name, color) in enumerate(zip(names, colors)):
        print(f"[{i+1}/4] Running {name}...")
        np.random.seed(args.seed)
        mdl = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        if name == "Vanilla":
            ctrl = MPPIController(mdl, MPPIParams(**common), cost_function=cost_fn)
        elif name == "DIAL":
            ctrl = DIALMPPIController(mdl, DIALMPPIParams(**common, **dial_kw),
                                      cost_function=cost_fn)
        elif name == "Shield-DIAL":
            ctrl = ShieldDIALMPPIController(
                mdl, ShieldDIALMPPIParams(**common, **dial_kw, **shield_kw),
                cost_function=cost_fn)
        else:
            ctrl = AdaptiveShieldDIALMPPIController(
                mdl, AdaptiveShieldDIALMPPIParams(
                    **common, **dial_kw, **shield_kw, **adaptive_kw),
                cost_function=cost_fn)

        r = run_sim(mdl, ctrl, state0.copy(), obstacles,
                    args.duration, dt, N, winds)
        results[name] = r
        extra = ""
        if hasattr(ctrl, "get_shield_statistics"):
            ss = ctrl.get_shield_statistics()
            extra = f" | Intervention: {ss['mean_intervention_rate']:.1%}"
        print(f"  RMSE: {r['rmse']:.4f}m | Violations: {r['violations']} | "
              f"Min h: {r['min_barrier']:.4f}{extra}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Shield-DIAL-MPPI Benchmark  |  wind={args.wind:.2f}  "
        f"margin={args.margin:.2f}m", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    for ox, oy, orr in obstacles:
        ax.add_patch(plt.Circle((ox, oy), orr, color="gray", alpha=.5))
        ax.add_patch(plt.Circle((ox, oy), orr+args.margin, color="red",
                                alpha=.1, ls="--"))
    r0 = results["Vanilla"]
    ax.plot(r0["refs"][:, 0], r0["refs"][:, 1], "k--", lw=2, alpha=.4, label="Ref")
    for n, c in zip(names, colors):
        r = results[n]
        ax.plot(r["states"][:, 0], r["states"][:, 1], color=c, lw=2, alpha=.8, label=n)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title("XY + Obstacles")
    ax.legend(fontsize=7); ax.grid(True, alpha=.3); ax.set_aspect("equal")
    ax.set_xlim(-0.5, 14); ax.set_ylim(-2.5, 2.5)

    ax = axes[0, 1]
    for n, c in zip(names, colors):
        ax.plot(results[n]["times"], results[n]["errors"], color=c, lw=1.5, label=n)
    ax.set_xlabel("Time"); ax.set_ylabel("Error (m)"); ax.set_title("Tracking Error")
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    ax = axes[0, 2]
    for n, c in zip(names, colors):
        r = results[n]
        bv = []
        for ti in range(len(r["states"])):
            bv.append(min((r["states"][ti, 0]-ox)**2+(r["states"][ti, 1]-oy)**2-orr**2
                          for ox, oy, orr in obstacles))
        ax.plot(r["times"], bv, color=c, lw=1.5, label=n)
    ax.axhline(0, color="red", ls="--", alpha=.5, label="h=0")
    ax.set_xlabel("Time"); ax.set_ylabel("min h(x)"); ax.set_title("Barrier")
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    ax = axes[1, 0]
    for n, c in zip(names, colors):
        ax.plot(results[n]["times"][:-1], results[n]["controls"][:, 0],
                color=c, lw=1.2, alpha=.8, label=f"{n}")
    ax.set_xlabel("Time"); ax.set_ylabel("v (m/s)"); ax.set_title("Linear Velocity")
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)

    ax = axes[1, 1]
    rmses = [results[n]["rmse"] for n in names]
    viols = [results[n]["violations"] for n in names]
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, rmses, color=colors, alpha=.8, ec="black")
    ax.set_xticks(x_pos); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("RMSE (m)"); ax.set_title("RMSE + Violations")
    ax.grid(True, axis="y", alpha=.3)
    for b, r, v in zip(bars, rmses, viols):
        label = f"{r:.3f}\n({v} viol)" if v > 0 else f"{r:.3f}\n(safe)"
        col = "red" if v > 0 else "green"
        ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                label, ha="center", va="bottom", fontsize=8, color=col)

    ax = axes[1, 2]; ax.axis("off")
    lines = ["  Benchmark Summary", "  " + "="*44, ""]
    for n in names:
        r = results[n]
        safe_str = "SAFE" if r["violations"] == 0 else f"UNSAFE ({r['violations']})"
        lines.append(f"  {n}:\n    RMSE: {r['rmse']:.4f}m\n"
                     f"    Safety: {safe_str}\n    Min h: {r['min_barrier']:.4f}\n")
    lines.append(f"  wind={args.wind}, margin={args.margin}m")
    lines.append(f"  K={args.K}, N={N}, obstacles={len(obstacles)}")
    ax.text(0.02, 0.5, "\n".join(lines), fontsize=9, va="center",
            family="monospace", bbox=dict(boxstyle="round", fc="lightyellow", alpha=.5))

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    path = "plots/shield_dial_mppi_benchmark.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {path}")
    plt.show()


if __name__ == "__main__":
    main()
