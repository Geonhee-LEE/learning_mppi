#!/usr/bin/env python3
"""
Learned Model + Shield-DIAL-MPPI 4종 비교 벤치마크

Shield-DIAL-MPPI가 바람 외란 하에서 안전을 보장하지만,
kinematic 모델이 바람을 모르므로 오실레이션 발생 (RMSE ~1.0m).
L1 Adaptive(외란 추정) / EKF Adaptive(파라미터 추정)를 결합하여 오실레이션 감소를 검증.

4종 컨트롤러:
  1. Vanilla (3D)       : Kinematic       | 안전 없음 | 적응 없음
  2. Shield-DIAL (3D)   : Kinematic       | CBF Shield | 적응 없음
  3. Shield-DIAL+EKF(5D): EKFAdaptive     | CBF Shield | 파라미터 추정
  4. Shield-DIAL+L1 (5D): L1Adaptive      | CBF Shield | 외란 추정

아키텍처:
            Real World (5D DynamicKinematicAdapter)
            c_v=0.5, c_omega=0.3 + wind
                       |
         +------+------+------+------+
         |      |             |      |
    [Vanilla] [Shield]    [+EKF]  [+L1]
     3D kin   3D kin      5D EKF  5D L1
     [:3]     [:3]        full    full

Usage:
    PYTHONPATH=. python examples/comparison/learned_shield_dial_benchmark.py --K 512 --duration 40 --wind 0.6
    PYTHONPATH=. python examples/comparison/learned_shield_dial_benchmark.py --live --K 512 --duration 40
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
from mppi_controller.models.kinematic.dynamic_kinematic_adapter import (
    DynamicKinematicAdapter,
)
from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    ShieldDIALMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.shield_dial_mppi import ShieldDIALMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
    CompositeMPPICost,
    AngleAwareTrackingCost,
    AngleAwareTerminalCost,
)
import matplotlib.pyplot as plt


# ── 상수 ─────────────────────────────────────────────────

REAL_C_V = 0.5
REAL_C_OMEGA = 0.3
NOM_C_V = 0.1
NOM_C_OMEGA = 0.1


# ── DynamicWorld (5D 실제 세계) ──────────────────────────

class DynamicWorld:
    """Real world with friction mismatch (5D DynamicKinematicAdapter)."""

    def __init__(self, c_v=REAL_C_V, c_omega=REAL_C_OMEGA):
        self._adapter = DynamicKinematicAdapter(
            c_v=c_v, c_omega=c_omega, k_v=5.0, k_omega=5.0,
        )
        self.state_5d = np.zeros(5)

    def reset(self, state_3d):
        self.state_5d = np.array([state_3d[0], state_3d[1], state_3d[2], 0.0, 0.0])

    def step(self, control, dt):
        self.state_5d = self._adapter.step(self.state_5d, control, dt)
        self.state_5d[2] = np.arctan2(np.sin(self.state_5d[2]),
                                       np.cos(self.state_5d[2]))
        return self.state_5d[:3].copy()

    def get_full_state(self):
        return self.state_5d.copy()


# ── 궤적 ─────────────────────────────────────────────────

def straight_line(t: float) -> np.ndarray:
    return np.array([0.5 * t, 0.0, 0.0])


def generate_ref_3d(t: float, N: int, dt: float) -> np.ndarray:
    ref = np.zeros((N + 1, 3))
    for i in range(N + 1):
        ref[i] = straight_line(t + i * dt)
    return ref


def make_5d_reference(ref_3d: np.ndarray, dt: float) -> np.ndarray:
    """3D reference (N+1, 3) -> 5D reference (N+1, 5) with v/omega estimation."""
    N_plus_1 = ref_3d.shape[0]
    ref_5d = np.zeros((N_plus_1, 5))
    ref_5d[:, :3] = ref_3d
    if N_plus_1 > 1:
        dx = np.diff(ref_3d[:, 0])
        dy = np.diff(ref_3d[:, 1])
        v_ref = np.sqrt(dx**2 + dy**2) / dt
        ref_5d[:-1, 3] = v_ref
        ref_5d[-1, 3] = v_ref[-1] if len(v_ref) > 0 else 0.0
        dtheta = np.diff(ref_3d[:, 2])
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        omega_ref = dtheta / dt
        ref_5d[:-1, 4] = omega_ref
        ref_5d[-1, 4] = omega_ref[-1] if len(omega_ref) > 0 else 0.0
    return ref_5d


# ── 외란 모델 ────────────────────────────────────────────

def precompute_wind(duration, dt, strength, seed=42):
    """외란 시퀀스 사전 생성 (모든 컨트롤러에 동일 적용)."""
    rng = np.random.RandomState(seed + 1000)
    n_steps = int(duration / dt)
    winds = np.zeros((n_steps, 3))
    for step in range(n_steps):
        t = step * dt
        dy = strength * np.sin(0.8 * t) + strength * 0.3 * rng.randn()
        dtheta = strength * 0.1 * rng.randn()
        winds[step] = np.array([0.0, dy, dtheta]) * dt
    return winds


# ── 비용 함수 ────────────────────────────────────────────

def make_cost_3d(obstacles):
    Q = np.array([10.0, 10.0, 1.0])
    Qf = np.array([20.0, 20.0, 2.0])
    R = np.array([0.1, 0.05])
    return CompositeMPPICost([
        StateTrackingCost(Q),
        TerminalCost(Qf),
        ControlEffortCost(R),
        ObstacleCost(obstacles, safety_margin=0.15, cost_weight=100.0),
    ])


def make_cost_5d(obstacles):
    Q = np.array([10.0, 10.0, 1.0, 0.1, 0.1])
    Qf = np.array([20.0, 20.0, 2.0, 0.2, 0.2])
    R = np.array([0.1, 0.05])
    return CompositeMPPICost([
        AngleAwareTrackingCost(Q, angle_indices=(2,)),
        AngleAwareTerminalCost(Qf, angle_indices=(2,)),
        ControlEffortCost(R),
        ObstacleCost(obstacles, safety_margin=0.15, cost_weight=100.0),
    ])


# ── 컨트롤러 팩토리 ─────────────────────────────────────

NAMES = ["Vanilla (3D)", "Shield-DIAL (3D)", "Shield-DIAL+EKF (5D)", "Shield-DIAL+L1 (5D)"]
COLORS = ["#4A90D9", "#E53935", "#FF9800", "#7CB342"]


def create_controller(name, K, N, dt, obstacles, margin):
    """단일 컨트롤러 생성. (model, controller, state_dim, needs_adaptation) 튜플."""

    common_3d = dict(
        N=N, dt=dt, K=K, lambda_=1.0,
        sigma=np.array([0.4, 0.8]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.05]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )
    common_5d = dict(
        N=N, dt=dt, K=K, lambda_=1.0,
        sigma=np.array([0.4, 0.8]),
        Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
        R=np.array([0.1, 0.05]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2, 0.2]),
    )
    dial_kw = dict(
        n_diffuse_init=10, n_diffuse=5,
        traj_diffuse_factor=0.6, horizon_diffuse_factor=0.5,
        sigma_scale=1.0, use_reward_normalization=True,
    )
    shield_kw = dict(
        cbf_obstacles=obstacles, cbf_alpha=0.3,
        cbf_safety_margin=margin,
        shield_enabled=True, shield_cbf_alpha=2.0,
    )
    cost_3d = make_cost_3d(obstacles)
    cost_5d = make_cost_5d(obstacles)

    if name == NAMES[0]:  # Vanilla (3D)
        mdl = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        ctrl = MPPIController(mdl, MPPIParams(**common_3d), cost_function=cost_3d)
        return (mdl, ctrl, 3, False)

    elif name == NAMES[1]:  # Shield-DIAL (3D)
        mdl = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        ctrl = ShieldDIALMPPIController(
            mdl, ShieldDIALMPPIParams(**common_3d, **dial_kw, **shield_kw),
            cost_function=cost_3d)
        return (mdl, ctrl, 3, False)

    elif name == NAMES[2]:  # Shield-DIAL+EKF (5D)
        mdl = EKFAdaptiveDynamics(
            c_v_init=NOM_C_V, c_omega_init=NOM_C_OMEGA, k_v=5.0, k_omega=5.0)
        ctrl = ShieldDIALMPPIController(
            mdl, ShieldDIALMPPIParams(**common_5d, **dial_kw, **shield_kw),
            cost_function=cost_5d)
        return (mdl, ctrl, 5, True)

    else:  # Shield-DIAL+L1 (5D)
        # 위치 am_gains ≈ 0: 바람(시변)을 무시, 마찰(상수)만 추정
        # cutoff_freq=0.3: 저역통과로 잔여 노이즈 제거
        mdl = L1AdaptiveDynamics(
            c_v_nom=NOM_C_V, c_omega_nom=NOM_C_OMEGA,
            adaptation_gain=50.0, cutoff_freq=0.3,
            am_gains=np.array([-0.2, -0.2, -0.2, -10.0, -10.0]))
        ctrl = ShieldDIALMPPIController(
            mdl, ShieldDIALMPPIParams(**common_5d, **dial_kw, **shield_kw),
            cost_function=cost_5d)
        return (mdl, ctrl, 5, True)


# ── 시뮬레이션 ───────────────────────────────────────────

def run_sim(K, obstacles, duration, dt, N, winds, state0, margin, seed):
    """4종 순차 시뮬레이션. 각각 독립 DynamicWorld + 시드 리셋."""
    results = {}

    for i, name in enumerate(NAMES):
        print(f"  [{i+1}/4] Running {name}...")
        np.random.seed(seed)
        mdl, ctrl, sdim, adapt = create_controller(name, K, N, dt, obstacles, margin)
        world = DynamicWorld()
        world.reset(state0)

        states_3d = [state0.copy()]
        controls_list = []
        adapt_diag = []  # 적응 진단 기록

        prev_state_5d = None
        prev_control = None

        for step in range(int(duration / dt)):
            t = step * dt
            state_5d = world.get_full_state()

            # 적응 업데이트 (5D 모델만)
            if adapt and step > 0 and prev_state_5d is not None:
                mdl.update_step(prev_state_5d, prev_control, state_5d, dt)

            # 컨트롤러 입력: 3D는 [:3], 5D는 전체
            ctrl_state = state_5d[:3] if sdim == 3 else state_5d
            ref_3d = generate_ref_3d(t, N, dt)
            ref = ref_3d if sdim == 3 else make_5d_reference(ref_3d, dt)

            control, info = ctrl.compute_control(ctrl_state, ref)

            prev_state_5d = state_5d.copy()
            prev_control = control.copy()

            # 실제 세계 전파
            world.step(control, dt)
            # 바람 외란 (위치만 영향)
            world.state_5d[:3] += winds[step]

            states_3d.append(world.state_5d[:3].copy())
            controls_list.append(control.copy())

            # 적응 진단 기록
            if adapt:
                if hasattr(mdl, 'get_disturbance_estimate'):
                    diag = mdl.get_disturbance_estimate()
                    adapt_diag.append(diag['sigma_norm'])
                elif hasattr(mdl, 'get_parameter_estimates'):
                    est = mdl.get_parameter_estimates()
                    adapt_diag.append((est['c_v'], est['c_omega']))

        states = np.array(states_3d)
        controls = np.array(controls_list)
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
            dists = np.sqrt((states[:, 0]-ox)**2 + (states[:, 1]-oy)**2)
            violations += int(np.sum(dists < orr))

        extra = ""
        if hasattr(ctrl, "get_shield_statistics"):
            ss = ctrl.get_shield_statistics()
            extra = f" | Intervention: {ss['mean_intervention_rate']:.1%}"

        print(f"    RMSE: {rmse:.4f}m | Violations: {violations} | "
              f"Min h: {min_barrier:.4f}{extra}")

        results[name] = dict(
            states=states, controls=controls, times=times, refs=refs,
            errors=errors, rmse=rmse, min_barrier=min_barrier,
            violations=violations, adapt_diag=adapt_diag,
        )

    return results


# ── 시각화 (Batch) ────────────────────────────────────────

def plot_results(results, obstacles, args):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Learned Model + Shield-DIAL-MPPI Benchmark  |  "
        f"wind={args.wind:.2f}  margin={args.margin:.2f}m  "
        f"real c_v={REAL_C_V}, c_ω={REAL_C_OMEGA}",
        fontsize=14, fontweight="bold")

    # 1. XY Trajectory
    ax = axes[0, 0]
    for ox, oy, orr in obstacles:
        ax.add_patch(plt.Circle((ox, oy), orr, color="gray", alpha=0.5))
        ax.add_patch(plt.Circle((ox, oy), orr + args.margin,
                     color="red", alpha=0.1, linestyle="--"))
    r0 = results[NAMES[0]]
    ax.plot(r0["refs"][:, 0], r0["refs"][:, 1], "k--", lw=2, alpha=0.4, label="Ref")
    for name, color in zip(NAMES, COLORS):
        r = results[name]
        ax.plot(r["states"][:, 0], r["states"][:, 1], color=color, lw=2, alpha=0.8, label=name)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory + Obstacles")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, min(22, args.duration * 0.5 + 2))
    ax.set_ylim(-2.5, 2.5)

    # 2. Tracking Error
    ax = axes[0, 1]
    for name, color in zip(NAMES, COLORS):
        ax.plot(results[name]["times"], results[name]["errors"],
                color=color, lw=1.5, label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 3. Barrier Value
    ax = axes[0, 2]
    for name, color in zip(NAMES, COLORS):
        r = results[name]
        bv = []
        for ti in range(len(r["states"])):
            bv.append(min((r["states"][ti, 0]-ox)**2 + (r["states"][ti, 1]-oy)**2 - orr**2
                          for ox, oy, orr in obstacles))
        ax.plot(r["times"], bv, color=color, lw=1.5, label=name)
    ax.axhline(0, color="red", ls="--", alpha=0.5, label="h=0 UNSAFE")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("min h(x)")
    ax.set_title("CBF Barrier Value")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 4. Linear Velocity
    ax = axes[1, 0]
    for name, color in zip(NAMES, COLORS):
        r = results[name]
        ax.plot(r["times"][:-1], r["controls"][:, 0],
                color=color, lw=1.2, alpha=0.8, label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("v (m/s)")
    ax.set_title("Linear Velocity")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 5. RMSE Bar
    ax = axes[1, 1]
    rmses = [results[n]["rmse"] for n in NAMES]
    viols = [results[n]["violations"] for n in NAMES]
    x_pos = np.arange(len(NAMES))
    bars = ax.bar(x_pos, rmses, color=COLORS, alpha=0.8, ec="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in NAMES], fontsize=7)
    ax.set_ylabel("RMSE (m)"); ax.set_title("RMSE + Safety")
    ax.grid(True, axis="y", alpha=0.3)
    for b, r, v in zip(bars, rmses, viols):
        label = f"{r:.3f}\n({v} viol)" if v > 0 else f"{r:.3f}\n(safe)"
        col = "red" if v > 0 else "green"
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                label, ha="center", va="bottom", fontsize=8, color=col)

    # 6. Summary + Adaptation Diagnostics
    ax = axes[1, 2]; ax.axis("off")
    lines = ["  Learned Model + Shield-DIAL Benchmark",
             f"  wind={args.wind}, margin={args.margin}m",
             f"  real: c_v={REAL_C_V}, c_omega={REAL_C_OMEGA}",
             f"  nominal: c_v={NOM_C_V}, c_omega={NOM_C_OMEGA}",
             f"  K={args.K}, duration={args.duration}s",
             "  " + "=" * 46, ""]

    for name in NAMES:
        r = results[name]
        safe = "SAFE" if r["violations"] == 0 else f"UNSAFE ({r['violations']})"
        block = (f"  {name}:\n"
                 f"    RMSE:   {r['rmse']:.4f} m\n"
                 f"    Safety: {safe}\n"
                 f"    Min h:  {r['min_barrier']:.4f}\n")

        # 적응 진단
        diag = r["adapt_diag"]
        if diag:
            if isinstance(diag[-1], tuple):
                # EKF: (c_v, c_omega)
                cv_final, co_final = diag[-1]
                block += f"    EKF: c_v={cv_final:.3f}, c_omega={co_final:.3f}\n"
            else:
                # L1: sigma_norm
                block += f"    L1: |sigma_f|={diag[-1]:.4f}\n"

        lines.append(block)

    ax.text(0.02, 0.5, "\n".join(lines), fontsize=8, va="center",
            family="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.5))

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    path = "plots/learned_shield_dial_benchmark.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {path}")
    plt.show()


# ── 라이브 시뮬레이션 ────────────────────────────────────

def run_live(K, obstacles, duration, dt, N, winds, state0, margin, seed):
    n_steps = int(duration / dt)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Learned + Shield-DIAL-MPPI  |  wind  "
        f"margin={margin:.2f}m  real c_v={REAL_C_V}, c_ω={REAL_C_OMEGA}",
        fontsize=14, fontweight="bold")
    plt.ion()

    # 각 컨트롤러별 독립 생성 (시드 리셋)
    entries = []
    worlds = []
    for name in NAMES:
        np.random.seed(seed)
        entry = create_controller(name, K, N, dt, obstacles, margin)
        entries.append(entry)
        w = DynamicWorld()
        w.reset(state0)
        worlds.append(w)

    S = {n: [state0.copy()] for n in NAMES}
    C = {n: [] for n in NAMES}
    E = {n: [] for n in NAMES}
    B = {n: [] for n in NAMES}
    AD = {n: [] for n in NAMES}  # 적응 진단
    prev_states = {n: None for n in NAMES}
    prev_controls = {n: None for n in NAMES}

    for step in range(n_steps):
        t = step * dt
        wind = winds[step] if winds is not None else np.zeros(3)

        for idx, (name, (mdl, ctrl, sdim, adapt)) in enumerate(zip(NAMES, entries)):
            world = worlds[idx]
            state_5d = world.get_full_state()

            # 적응 업데이트
            if adapt and step > 0 and prev_states[name] is not None:
                mdl.update_step(prev_states[name], prev_controls[name], state_5d, dt)

            ctrl_state = state_5d[:3] if sdim == 3 else state_5d
            ref_3d = generate_ref_3d(t, N, dt)
            ref = ref_3d if sdim == 3 else make_5d_reference(ref_3d, dt)

            u, _ = ctrl.compute_control(ctrl_state, ref)

            prev_states[name] = state_5d.copy()
            prev_controls[name] = u.copy()

            world.step(u, dt)
            world.state_5d[:3] += wind

            state_3d = world.state_5d[:3].copy()
            S[name].append(state_3d)
            C[name].append(u.copy())
            E[name].append(np.linalg.norm(state_3d[:2] - straight_line(t + dt)[:2]))

            min_h = min(
                (state_3d[0]-ox)**2 + (state_3d[1]-oy)**2 - orr**2
                for ox, oy, orr in obstacles
            )
            B[name].append(min_h)

            # 적응 진단
            if adapt:
                if hasattr(mdl, 'get_disturbance_estimate'):
                    AD[name].append(mdl.get_disturbance_estimate()['sigma_norm'])
                elif hasattr(mdl, 'get_parameter_estimates'):
                    est = mdl.get_parameter_estimates()
                    AD[name].append((est['c_v'], est['c_omega']))

        # 10 스텝마다 그래프 갱신
        if step % 10 == 0 or step == n_steps - 1:
            for row in axes:
                for ax in row:
                    ax.clear()
            et = np.arange(1, step + 2) * dt

            # 1. XY
            ax = axes[0, 0]
            for ox, oy, orr in obstacles:
                ax.add_patch(plt.Circle((ox, oy), orr, color="gray", alpha=0.5))
                ax.add_patch(plt.Circle((ox, oy), orr + margin,
                             color="red", alpha=0.1, linestyle="--"))
            rl = np.array([straight_line(i * dt) for i in range(step + 2)])
            ax.plot(rl[:, 0], rl[:, 1], "k--", lw=2, alpha=0.4, label="Ref")
            for n, c in zip(NAMES, COLORS):
                s = np.array(S[n])
                ax.plot(s[:, 0], s[:, 1], color=c, lw=2, alpha=0.8, label=n)
                ax.plot(s[-1, 0], s[-1, 1], "o", color=c, ms=7)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_title(f"XY Trajectory (t={t:.1f}s)")
            ax.legend(fontsize=6, loc="upper left")
            ax.grid(True, alpha=0.3); ax.set_aspect("equal")
            ax.set_xlim(-0.5, max(12, t * 0.5 + 2)); ax.set_ylim(-2.5, 2.5)

            # 2. Error
            ax = axes[0, 1]
            for n, c in zip(NAMES, COLORS):
                ax.plot(et, E[n], color=c, lw=1.5, label=n)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
            ax.set_title("Tracking Error"); ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

            # 3. Barrier
            ax = axes[0, 2]
            for n, c in zip(NAMES, COLORS):
                ax.plot(et, B[n], color=c, lw=1.5, label=n)
            ax.axhline(0, color="red", ls="--", alpha=0.5, label="h=0 UNSAFE")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("min h(x)")
            ax.set_title("CBF Barrier Value"); ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

            # 4. Control
            ax = axes[1, 0]
            for n, c in zip(NAMES, COLORS):
                cc = np.array(C[n])
                ax.plot(et, cc[:, 0], color=c, lw=1.2, alpha=0.8, label=f"{n} v")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("v (m/s)")
            ax.set_title("Linear Velocity"); ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

            # 5. RMSE bar
            ax = axes[1, 1]
            rmses = [np.sqrt(np.mean(np.array(E[n])**2)) for n in NAMES]
            bars = ax.bar(range(len(NAMES)), rmses, color=COLORS, alpha=0.8, ec="black")
            ax.set_xticks(range(len(NAMES)))
            ax.set_xticklabels([n.replace(" (", "\n(") for n in NAMES], fontsize=6)
            ax.set_ylabel("RMSE (m)"); ax.set_title("RMSE")
            ax.grid(True, axis="y", alpha=0.3)
            for b, r in zip(bars, rmses):
                ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                        f"{r:.3f}", ha="center", va="bottom", fontsize=8)

            # 6. Summary + Adaptation Diagnostics
            ax = axes[1, 2]; ax.axis("off")
            lines = [f"  Status (step {step+1}/{n_steps})",
                     f"  margin={margin:.2f}m, real c_v={REAL_C_V}",
                     "  " + "=" * 44, ""]
            for n in NAMES:
                e = np.array(E[n])
                viol = sum(1 for b in B[n] if b < 0)
                block = (f"  {n}:\n"
                         f"    RMSE:       {np.sqrt(np.mean(e**2)):.4f} m\n"
                         f"    Violations: {viol}\n"
                         f"    Min h(x):   {min(B[n]):.4f}\n")
                diag = AD[n]
                if diag:
                    if isinstance(diag[-1], tuple):
                        cv, co = diag[-1]
                        block += f"    EKF: c_v={cv:.3f}, c_omega={co:.3f}\n"
                    else:
                        block += f"    L1: |sigma_f|={diag[-1]:.4f}\n"
                lines.append(block)

            ax.text(0.02, 0.5, "\n".join(lines), fontsize=8, va="center",
                    family="monospace",
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.5))
            plt.tight_layout(); plt.pause(0.001)

    plt.ioff()
    os.makedirs("plots", exist_ok=True)
    path = "plots/learned_shield_dial_benchmark.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Learned Model + Shield-DIAL-MPPI Benchmark")
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--wind", type=float, default=0.6,
                        help="Wind disturbance strength (0=none)")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="CBF safety margin (min distance from obstacle surface)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Learned Model + Shield-DIAL-MPPI Benchmark".center(80))
    print("=" * 80)

    N, dt = 20, 0.05

    # 8개 장애물 (원래 shield_dial_mppi_benchmark.py와 동일)
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

    winds = precompute_wind(args.duration, dt, args.wind, args.seed)
    state0 = np.array([0.0, 0.0, 0.0])

    print(f"  Wind:      {args.wind:.2f}")
    print(f"  Margin:    {args.margin:.2f}m")
    print(f"  K={args.K}, N={N}, duration={args.duration}s")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Real:      c_v={REAL_C_V}, c_omega={REAL_C_OMEGA}")
    print(f"  Nominal:   c_v={NOM_C_V}, c_omega={NOM_C_OMEGA}")
    print("=" * 80 + "\n")

    if args.live:
        run_live(args.K, obstacles, args.duration, dt, N, winds, state0,
                 args.margin, args.seed)
    else:
        results = run_sim(args.K, obstacles, args.duration, dt, N, winds, state0,
                          args.margin, args.seed)
        plot_results(results, obstacles, args)


if __name__ == "__main__":
    main()
