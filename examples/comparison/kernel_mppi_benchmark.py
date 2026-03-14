#!/usr/bin/env python3
"""
Kernel MPPI 벤치마크: 3 모델 × Vanilla vs KMPPI 비교

모델:
  1. DifferentialDrive  (3D state, 2D control) — circle body
  2. Ackermann           (4D state, 2D control) — car body
  3. SwerveDrive         (3D state, 3D control) — rectangle body

비교:
  - Vanilla MPPI  (전체 N차원 샘플링)
  - Kernel MPPI   (S개 서포트 + RBF 보간)

측정:
  - 위치 추적 RMSE
  - 제어 평활도 (jerk = mean |delta_u|)
  - 샘플링 차원 축소율
  - ESS

Usage:
    PYTHONPATH=. python examples/comparison/kernel_mppi_benchmark.py --live
    PYTHONPATH=. python examples/comparison/kernel_mppi_benchmark.py --live --trajectory figure8
    PYTHONPATH=. python examples/comparison/kernel_mppi_benchmark.py --no-plot
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
from matplotlib.animation import FuncAnimation

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.models.kinematic.swerve_drive_kinematic import (
    SwerveDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, KernelMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.kernel_mppi import KernelMPPIController
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ── 모델 설정 ──────────────────────────────────────────────────

MODEL_CONFIGS = {
    "DiffDrive": {
        "model_fn": lambda: DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
        "state_dim": 3,
        "control_dim": 2,
        "sigma": np.array([0.5, 0.5]),
        "Q": np.array([10.0, 10.0, 1.0]),
        "R": np.array([0.1, 0.1]),
        "color_vanilla": "#1f77b4",
        "color_kernel": "#d62728",
        "marker": "o",
    },
    "Ackermann": {
        "model_fn": lambda: AckermannKinematic(wheelbase=0.3, v_max=1.0, max_steer=0.5),
        "state_dim": 4,
        "control_dim": 2,
        "sigma": np.array([0.5, 0.3]),
        "Q": np.array([10.0, 10.0, 1.0, 0.1]),
        "R": np.array([0.1, 0.1]),
        "color_vanilla": "#2ca02c",
        "color_kernel": "#ff7f0e",
        "marker": "s",
    },
    "Swerve": {
        "model_fn": lambda: SwerveDriveKinematic(vx_max=1.0, vy_max=1.0, omega_max=1.0),
        "state_dim": 3,
        "control_dim": 3,
        "sigma": np.array([0.5, 0.5, 0.5]),
        "Q": np.array([10.0, 10.0, 1.0]),
        "R": np.array([0.1, 0.1, 0.1]),
        "color_vanilla": "#9467bd",
        "color_kernel": "#e377c2",
        "marker": "D",
    },
}


def make_reference_fn(traj_fn, N, dt, state_dim):
    """state_dim에 맞는 reference function 생성"""
    def ref_fn(t):
        ref_3d = generate_reference_trajectory(traj_fn, t, N, dt)  # (N+1, 3)
        if state_dim == 3:
            return ref_3d
        # 4D (Ackermann): pad with zeros for delta
        ref_nd = np.zeros((ref_3d.shape[0], state_dim))
        ref_nd[:, :3] = ref_3d
        return ref_nd
    return ref_fn


def setup_controllers(model_name, cfg, N, dt, K, S, bandwidth):
    """모델별 Vanilla + Kernel 컨트롤러 생성"""
    model_v = cfg["model_fn"]()
    model_k = cfg["model_fn"]()

    common = dict(N=N, dt=dt, K=K, lambda_=1.0, sigma=cfg["sigma"],
                  Q=cfg["Q"], R=cfg["R"])

    vanilla_params = MPPIParams(**common)
    kernel_params = KernelMPPIParams(**common, num_support_pts=S,
                                     kernel_bandwidth=bandwidth)

    ctrl_v = MPPIController(model_v, vanilla_params)
    ctrl_k = KernelMPPIController(model_k, kernel_params)

    return {
        f"{model_name} Vanilla": (model_v, ctrl_v, cfg["color_vanilla"]),
        f"{model_name} KMPPI": (model_k, ctrl_k, cfg["color_kernel"]),
    }


# ── Live 애니메이션 ────────────────────────────────────────────

def run_live(args):
    """3모델 × 2방법 = 6개 라이브 비교 애니메이션"""

    N, dt, K = args.N, args.dt, args.K
    S, bw = args.support_pts, args.bandwidth
    duration = args.duration
    num_steps = int(duration / dt)

    traj_kwargs = {"radius": 3.0} if args.trajectory == "circle" else {}
    traj_fn = create_trajectory_function(args.trajectory, **traj_kwargs)

    # 모든 컨트롤러 설정
    all_entries = {}  # name -> (model, ctrl, color)
    initial_states = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        entries = setup_controllers(model_name, cfg, N, dt, K, S, bw)
        all_entries.update(entries)
        # 초기 상태: 궤적 위
        init_3d = traj_fn(0.0)
        for name in entries:
            if cfg["state_dim"] == 4:
                initial_states[name] = np.concatenate([init_3d, [0.0]])
            else:
                initial_states[name] = init_3d.copy()

    # 시뮬레이션 상태
    states = {name: initial_states[name].copy() for name in all_entries}
    data = {
        name: {"xy": [], "times": [], "errors": [], "ess": [],
               "costs": [], "controls": []}
        for name in all_entries
    }
    sim_t = [0.0]

    # ===== Figure (3행 2열: 모델별 XY + 오른쪽 metrics) =====
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"Kernel MPPI Live Benchmark — {args.trajectory.capitalize()} trajectory\n"
        f"(N={N}, K={K}, S={S}, bandwidth={bw})",
        fontsize=14, fontweight="bold",
    )

    model_names = list(MODEL_CONFIGS.keys())

    # XY 패널 (3행, 좌측 2/3)
    ax_xys = {}
    lines_xy = {}
    dots = {}
    for i, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_title(f"{model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # 레퍼런스 궤적
        ref_t = np.linspace(0, duration, 500)
        ref_pts = np.array([traj_fn(t) for t in ref_t])
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

        ax_xys[model_name] = ax

        # 이 모델의 Vanilla + KMPPI 라인
        for name, (_, _, color) in all_entries.items():
            if name.startswith(model_name):
                short = "Vanilla" if "Vanilla" in name else "KMPPI"
                lines_xy[name], = ax.plot([], [], color=color, linewidth=2,
                                          label=short, alpha=0.8)
                dots[name], = ax.plot([], [], "o", color=color, markersize=7)

        ax.legend(loc="upper left", fontsize=8)

    # Tracking Error 패널 (우측 상단)
    ax_err = fig.add_subplot(gs[0, 1:])
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Position Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, (_, _, color) in all_entries.items():
        short = name.replace(" Vanilla", " V").replace(" KMPPI", " K")
        ls = "-" if "KMPPI" in name else "--"
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5,
                                        linestyle=ls, label=short)
    ax_err.legend(fontsize=7, ncol=3)

    # ESS 패널 (우측 중간)
    ax_ess = fig.add_subplot(gs[1, 1:])
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, (_, _, color) in all_entries.items():
        short = name.replace(" Vanilla", " V").replace(" KMPPI", " K")
        ls = "-" if "KMPPI" in name else "--"
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5,
                                        linestyle=ls, label=short)
    ax_ess.legend(fontsize=7, ncol=3)

    # 요약 텍스트 패널 (우측 하단)
    ax_summary = fig.add_subplot(gs[2, 1:])
    ax_summary.axis("off")
    summary_text = ax_summary.text(
        0.05, 0.95, "", transform=ax_summary.transAxes,
        fontsize=9, family="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # 상태 텍스트
    time_text = fig.text(
        0.5, 0.01, "", ha="center", fontsize=10, family="monospace",
    )

    def init():
        for name in all_entries:
            lines_xy[name].set_data([], [])
            dots[name].set_data([], [])
            lines_err[name].set_data([], [])
            lines_ess[name].set_data([], [])
        time_text.set_text("")
        summary_text.set_text("")
        return []

    def update(frame):
        if frame >= num_steps:
            return []

        t = sim_t[0]

        for name, (model, ctrl, _) in all_entries.items():
            cfg = None
            for mn, c in MODEL_CONFIGS.items():
                if name.startswith(mn):
                    cfg = c
                    break

            ref = make_reference_fn(traj_fn, N, dt, cfg["state_dim"])(t)
            control, info = ctrl.compute_control(states[name], ref)

            states[name] = model.step(states[name], control, dt)

            ref_pt = ref[0, :2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))
            data[name]["costs"].append(info.get("best_cost", 0.0))
            data[name]["controls"].append(control.copy())

        sim_t[0] += dt

        # 그래프 업데이트
        for name in all_entries:
            xy = np.array(data[name]["xy"])
            times = np.array(data[name]["times"])
            lines_xy[name].set_data(xy[:, 0], xy[:, 1])
            dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
            lines_err[name].set_data(times, data[name]["errors"])
            lines_ess[name].set_data(times, data[name]["ess"])

        # 축 자동 조정 (XY)
        for model_name in model_names:
            ax = ax_xys[model_name]
            ax.relim()
            ax.autoscale_view()

        ax_err.relim(); ax_err.autoscale_view()
        ax_ess.relim(); ax_ess.autoscale_view()

        # 요약 텍스트 업데이트 (매 20프레임)
        if frame % 20 == 0 and frame > 0:
            lines = ["Model          Method    RMSE     Jerk    ESS"]
            lines.append("-" * 50)
            for model_name in model_names:
                for suffix in ["Vanilla", "KMPPI"]:
                    name = f"{model_name} {suffix}"
                    errs = data[name]["errors"]
                    rmse = np.sqrt(np.mean(np.array(errs)**2))
                    ctrls = np.array(data[name]["controls"])
                    jerk = np.mean(np.abs(np.diff(ctrls[:, :2], axis=0))) if len(ctrls) > 1 else 0
                    ess = np.mean(data[name]["ess"])
                    tag = "V" if suffix == "Vanilla" else "K"
                    lines.append(f"{model_name:14s} {tag:5s}  {rmse:6.3f}  {jerk:6.3f}  {ess:6.1f}")
            dim_reduction = (1 - S / N) * 100
            lines.append(f"\nDimension reduction: {dim_reduction:.0f}% (S={S}, N={N})")
            summary_text.set_text("\n".join(lines))

        time_text.set_text(f"t = {sim_t[0]:.1f}s / {duration:.0f}s")

        return []

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=num_steps, interval=20, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)

    # GIF
    gif_path = f"plots/kernel_mppi_live_{args.trajectory}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    # MP4
    try:
        mp4_path = f"plots/kernel_mppi_live_{args.trajectory}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # 최종 통계
    print_final_stats(data, model_names, N, S)


# ── 배치 모드 ──────────────────────────────────────────────────

def run_batch(args):
    """배치 모드: 3모델 각각 시뮬레이션 → 정적 비교 플롯"""

    N, dt, K = args.N, args.dt, args.K
    S, bw = args.support_pts, args.bandwidth
    duration = args.duration
    num_steps = int(duration / dt)

    traj_kwargs = {"radius": 3.0} if args.trajectory == "circle" else {}
    traj_fn = create_trajectory_function(args.trajectory, **traj_kwargs)

    print(f"\n{'='*70}")
    print(f"  Kernel MPPI Benchmark — {args.trajectory.capitalize()}")
    print(f"  N={N}, K={K}, S={S}, bandwidth={bw}, duration={duration}s")
    print(f"{'='*70}\n")

    all_data = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        entries = setup_controllers(model_name, cfg, N, dt, K, S, bw)

        init_3d = traj_fn(0.0)
        for name, (model, ctrl, color) in entries.items():
            sd = cfg["state_dim"]
            state = np.concatenate([init_3d, [0.0]]) if sd == 4 else init_3d.copy()
            ref_fn = make_reference_fn(traj_fn, N, dt, sd)

            xy_list, err_list, ctrl_list, time_list = [], [], [], []
            ess_list, cost_list, compute_times = [], [], []

            print(f"  Running {name} ...")
            for step in range(num_steps):
                t = step * dt
                ref = ref_fn(t)

                t0 = time.perf_counter()
                control, info = ctrl.compute_control(state, ref)
                compute_times.append(time.perf_counter() - t0)

                state = model.step(state, control, dt)

                xy_list.append(state[:2].copy())
                err_list.append(np.linalg.norm(state[:2] - ref[0, :2]))
                ctrl_list.append(control.copy())
                time_list.append(t)
                ess_list.append(info.get("ess", 0.0))
                cost_list.append(info.get("best_cost", 0.0))

            all_data[name] = {
                "xy": np.array(xy_list),
                "errors": np.array(err_list),
                "controls": np.array(ctrl_list),
                "times": np.array(time_list),
                "ess": np.array(ess_list),
                "costs": np.array(cost_list),
                "compute_times": np.array(compute_times),
                "color": color,
            }

    # 결과 출력
    model_names = list(MODEL_CONFIGS.keys())
    print(f"\n{'='*75}")
    print(f"  {'Method':<22} {'RMSE':>7} {'Jerk':>7} {'ESS':>7} {'Time/step':>10}")
    print(f"  {'-'*65}")
    for name, d in all_data.items():
        rmse = np.sqrt(np.mean(d["errors"]**2))
        jerk = np.mean(np.abs(np.diff(d["controls"][:, :2], axis=0)))
        ess = np.mean(d["ess"])
        t_ms = np.mean(d["compute_times"]) * 1000
        print(f"  {name:<22} {rmse:>7.4f} {jerk:>7.4f} {ess:>7.1f} {t_ms:>8.1f}ms")
    print(f"{'='*75}\n")

    # 정적 플롯
    if not args.no_plot:
        plot_batch_results(all_data, model_names, args, N, S)


def plot_batch_results(all_data, model_names, args, N, S):
    """배치 결과 정적 플롯"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35)
    fig.suptitle(
        f"Kernel MPPI Benchmark — {args.trajectory.capitalize()}\n"
        f"N={N}, K={args.K}, S={S}, bandwidth={args.bandwidth}",
        fontsize=14, fontweight="bold",
    )

    traj_kwargs = {"radius": 3.0} if args.trajectory == "circle" else {}
    traj_fn = create_trajectory_function(args.trajectory, **traj_kwargs)

    # Row 0: 모델별 XY
    for i, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Reference
        ref_t = np.linspace(0, args.duration, 500)
        ref_pts = np.array([traj_fn(t) for t in ref_t])
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

        for name, d in all_data.items():
            if name.startswith(model_name):
                short = "Vanilla" if "Vanilla" in name else "KMPPI"
                ls = "--" if "Vanilla" in name else "-"
                ax.plot(d["xy"][:, 0], d["xy"][:, 1], color=d["color"],
                        linewidth=2, linestyle=ls, label=short, alpha=0.8)
        ax.legend(fontsize=8)

    # Row 1, Col 0: RMSE 바 차트
    ax = fig.add_subplot(gs[1, 0])
    names, rmses, colors = [], [], []
    for name, d in all_data.items():
        short = name.replace("DiffDrive ", "DD ").replace("Ackermann ", "Ack ").replace("Swerve ", "Sw ")
        names.append(short)
        rmses.append(np.sqrt(np.mean(d["errors"]**2)))
        colors.append(d["color"])
    bars = ax.bar(range(len(names)), rmses, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Position RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, r in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{r:.3f}", ha="center", va="bottom", fontsize=7)

    # Row 1, Col 1: Jerk 바 차트
    ax = fig.add_subplot(gs[1, 1])
    jerks = []
    for name, d in all_data.items():
        j = np.mean(np.abs(np.diff(d["controls"][:, :2], axis=0)))
        jerks.append(j)
    bars = ax.bar(range(len(names)), jerks, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean |delta_u|")
    ax.set_title("Control Smoothness (lower=smoother)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, j in zip(bars, jerks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{j:.3f}", ha="center", va="bottom", fontsize=7)

    # Row 1, Col 2: 요약
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    dim_red = (1 - S / N) * 100
    lines = [
        f"Kernel MPPI Summary",
        f"{'='*40}",
        f"",
        f"{'Model':<14} {'Method':<8} {'RMSE':>6} {'Jerk':>6} {'ms':>6}",
        f"{'-'*40}",
    ]
    for name, d in all_data.items():
        mn = name.split()[0][:6]
        tag = "V" if "Vanilla" in name else "K"
        rmse = np.sqrt(np.mean(d["errors"]**2))
        jerk = np.mean(np.abs(np.diff(d["controls"][:, :2], axis=0)))
        t_ms = np.mean(d["compute_times"]) * 1000
        lines.append(f"{mn:<14} {tag:<8} {rmse:>6.3f} {jerk:>6.3f} {t_ms:>5.1f}")

    lines.extend([
        f"",
        f"Dimension reduction: {dim_red:.0f}%",
        f"Support points: S={S}, Horizon: N={N}",
        f"Bandwidth: {args.bandwidth}",
    ])
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/kernel_mppi_benchmark_{args.trajectory}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save_path}")
    plt.close()


def print_final_stats(data, model_names, N, S):
    """최종 통계 출력"""
    dim_reduction = (1 - S / N) * 100

    print(f"\n{'='*70}")
    print(f"  Final Statistics")
    print(f"{'='*70}")
    print(f"\n  {'Method':<22} {'RMSE':>7} {'Jerk':>7} {'Avg ESS':>8}")
    print(f"  {'-'*50}")
    for name in data:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs)**2))
        ctrls = np.array(data[name]["controls"])
        jerk = np.mean(np.abs(np.diff(ctrls[:, :2], axis=0))) if len(ctrls) > 1 else 0
        ess = np.mean(data[name]["ess"])
        print(f"  {name:<22} {rmse:>7.4f} {jerk:>7.4f} {ess:>8.1f}")

    print(f"\n  Dimension reduction: {dim_reduction:.0f}% (S={S}, N={N})")
    print(f"{'='*70}\n")


# ── main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kernel MPPI Benchmark: 3 models x Vanilla vs KMPPI"
    )
    parser.add_argument(
        "--trajectory", type=str, default="circle",
        choices=["circle", "figure8", "sine", "slalom"],
    )
    parser.add_argument("--live", action="store_true", help="Live animation mode")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--N", type=int, default=30, help="Horizon length")
    parser.add_argument("--K", type=int, default=512, help="Number of samples")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--support-pts", type=int, default=8, help="S (support points)")
    parser.add_argument("--bandwidth", type=float, default=2.0, help="RBF bandwidth")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.live:
        run_live(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
