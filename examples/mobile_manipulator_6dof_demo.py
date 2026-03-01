#!/usr/bin/env python3
"""
Mobile Manipulator 6-DOF End-Effector 3D 추적 데모

UR5-style 6-DOF Arm + DiffDrive Base로 EE 3D 궤적 추적.

개선사항:
  - DiffDrive 바퀴 시각화: 좌우 2개 구동 바퀴 (속도 비례 회전 표시) + 캐스터
  - Smooth 모션: 시뮬레이션 사전 계산 → 렌더링 분리

Usage:
    python examples/mobile_manipulator_6dof_demo.py --trajectory ee_3d_circle --live
    python examples/mobile_manipulator_6dof_demo.py --trajectory ee_3d_helix --live
    python examples/mobile_manipulator_6dof_demo.py --trajectory ee_3d_helix --pose
    python examples/mobile_manipulator_6dof_demo.py --duration 20
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, DIALMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    EndEffector3DTrackingCost,
    EndEffector3DTerminalCost,
    EndEffectorPoseTrackingCost,
    EndEffectorPoseTerminalCost,
    ControlEffortCost,
    ControlRateCost,
    JointLimitCost,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ═══════════════════════════════════════════════════════
#  DiffDrive 바퀴 속도 계산
# ═══════════════════════════════════════════════════════

_BASE_SIZE = 0.15
_WHEEL_TRACK = 2 * _BASE_SIZE  # 좌우 바퀴 간격


def compute_diffdrive_wheel_speeds(control):
    """
    DiffDrive 좌우 바퀴 속도 계산.

    v_left  = v - ω * track/2
    v_right = v + ω * track/2

    Args:
        control: (8,) [v, ω, dq1..6]

    Returns:
        (v_left, v_right)
    """
    v, omega = control[0], control[1]
    v_left = v - omega * _WHEEL_TRACK / 2
    v_right = v + omega * _WHEEL_TRACK / 2
    return v_left, v_right


# ═══════════════════════════════════════════════════════
#  시각화 함수
# ═══════════════════════════════════════════════════════

def draw_base_3d(ax, state, control=None, base_size=0.15, alpha=1.0):
    """3D DiffDrive 베이스 그리기 (사각형 + 좌우 2바퀴 + 속도 표시 + 캐스터 + heading)"""
    x, y, theta = state[0], state[1], state[2]
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

    half = base_size
    corners_body = np.array([
        [-half, -half, 0],
        [half, -half, 0],
        [half, half, 0],
        [-half, half, 0],
    ])
    corners_world = (R @ corners_body.T).T + np.array([x, y, 0])

    # 바닥 사각형
    for i in range(4):
        j = (i + 1) % 4
        ax.plot3D(
            [corners_world[i, 0], corners_world[j, 0]],
            [corners_world[i, 1], corners_world[j, 1]],
            [corners_world[i, 2], corners_world[j, 2]],
            "-", color="gray", linewidth=2, alpha=alpha,
        )

    # 바퀴 속도 계산
    if control is not None:
        v_left, v_right = compute_diffdrive_wheel_speeds(control)
    else:
        v_left, v_right = 0.0, 0.0

    # DiffDrive 구동 바퀴 2개 (좌우, heading 방향 고정)
    wheel_r = base_size * 0.45
    wz = 0.005
    for side, v_wheel in [(-1, v_left), (1, v_right)]:
        wc_body = np.array([0, side * half, wz])
        wc_world = R @ wc_body + np.array([x, y, 0])

        # 바퀴 (항상 heading 방향 — DiffDrive는 조향 없음)
        front = R @ (wc_body + np.array([wheel_r, 0, 0])) + np.array([x, y, 0])
        rear = R @ (wc_body + np.array([-wheel_r, 0, 0])) + np.array([x, y, 0])
        ax.plot3D(
            [rear[0], front[0]], [rear[1], front[1]], [rear[2], front[2]],
            "-", color="black", linewidth=5, alpha=alpha, solid_capstyle="round",
        )

        # 바퀴 축 (짧은 수직선)
        axle_half = wheel_r * 0.35
        top = R @ (wc_body + np.array([0, 0, axle_half])) + np.array([x, y, 0])
        bot = R @ (wc_body + np.array([0, 0, -axle_half])) + np.array([x, y, 0])
        ax.plot3D(
            [bot[0], top[0]], [bot[1], top[1]], [bot[2], top[2]],
            "-", color="black", linewidth=2, alpha=alpha * 0.6,
        )

        # 바퀴 속도 화살표 (heading 방향으로 v_wheel 크기만큼)
        if abs(v_wheel) > 0.05:
            arrow_scale = 0.15
            arrow_end = R @ (wc_body + np.array([v_wheel * arrow_scale, 0, 0])) + np.array([x, y, 0])
            color = "blue" if v_wheel > 0 else "red"
            ax.plot3D(
                [wc_world[0], arrow_end[0]],
                [wc_world[1], arrow_end[1]],
                [wc_world[2], arrow_end[2]],
                "-", color=color, linewidth=2.5, alpha=alpha,
            )

        ax.scatter3D([wc_world[0]], [wc_world[1]], [wc_world[2]],
                      color="black", s=20, alpha=alpha)

    # 전방 캐스터
    caster_body = np.array([half * 0.8, 0, wz])
    caster_world = R @ caster_body + np.array([x, y, 0])
    ax.scatter3D([caster_world[0]], [caster_world[1]], [caster_world[2]],
                  color="dimgray", s=18, marker="o", alpha=alpha)

    # heading 화살표
    arrow_len = base_size * 1.2
    ax.plot3D(
        [x, x + arrow_len * ct],
        [y, y + arrow_len * st],
        [0, 0],
        "-", color="darkgreen", linewidth=2.5, alpha=alpha,
    )
    ax.scatter3D([x], [y], [0], color="gray", s=40, alpha=alpha)


def draw_arm_3d(ax, state, model, control=None, color="blue", alpha=1.0, linewidth=2.0):
    """3D 로봇 팔 + DiffDrive 베이스 그리기"""
    draw_base_3d(ax, state, control=control, alpha=alpha)

    positions = model.get_joint_positions(state)  # (7, 3)
    for i in range(6):
        lw = linewidth * (1.0 - 0.1 * i)
        ax.plot3D(
            [positions[i, 0], positions[i + 1, 0]],
            [positions[i, 1], positions[i + 1, 1]],
            [positions[i, 2], positions[i + 1, 2]],
            "-", color=color, linewidth=lw, alpha=alpha,
        )
    ax.scatter3D(
        positions[:-1, 0], positions[:-1, 1], positions[:-1, 2],
        color=color, s=20, alpha=alpha,
    )
    ax.scatter3D(
        [positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
        color="red", s=60, marker="*", alpha=alpha,
    )


# ═══════════════════════════════════════════════════════
#  시뮬레이션
# ═══════════════════════════════════════════════════════

def run_simulation(model, controller, traj_fn, initial_state, params, duration, use_pose,
                   ema_alpha=0.4):
    """시뮬레이션 실행 (EMA 필터 적용)"""
    dt = params.dt
    n_steps = int(duration / dt)
    state = initial_state.copy()

    history = {
        "time": [], "state": [], "control": [],
        "ee_pos": [], "ee_ref": [], "ee_error": [], "solve_time": [],
    }
    if use_pose:
        history["ee_pose"] = []
        history["ori_error"] = []

    filtered_ctrl = np.zeros(model.control_dim)

    for step in range(n_steps):
        t = step * dt
        ref = generate_reference_trajectory(traj_fn, t, params.N, dt)

        t0 = time.perf_counter()
        raw_control, info = controller.compute_control(state, ref)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # EMA 필터
        filtered_ctrl = ema_alpha * raw_control + (1 - ema_alpha) * filtered_ctrl
        control = filtered_ctrl

        ee_pos = model.forward_kinematics(state)
        ee_ref = ref[0, :3]
        ee_error = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))

        history["time"].append(t)
        history["state"].append(state.copy())
        history["control"].append(control.copy())
        history["ee_pos"].append(ee_pos.copy())
        history["ee_ref"].append(ee_ref.copy())
        history["ee_error"].append(ee_error)
        history["solve_time"].append(solve_ms)

        if use_pose:
            ee_pose = model.forward_kinematics_pose(state)
            history["ee_pose"].append(ee_pose.copy())
            ori_err = ee_pose[3:6] - ref[0, 3:6]
            ori_err = np.arctan2(np.sin(ori_err), np.cos(ori_err))
            history["ori_error"].append(np.sqrt(np.sum(ori_err**2)))

        state = model.step(state, control, dt)
        state = model.normalize_state(state)

        if step % 50 == 0:
            q_str = ", ".join(f"{state[3+i]:.2f}" for i in range(6))
            print(
                f"  t={t:5.1f}s | EE err={ee_error:.4f}m | "
                f"solve={solve_ms:.1f}ms | q=({q_str})"
            )

    for key in history:
        history[key] = np.array(history[key])
    return history


# ═══════════════════════════════════════════════════════
#  플롯
# ═══════════════════════════════════════════════════════

def plot_results(history, model, title="6-DOF Mobile Manipulator EE 3D Tracking", use_pose=False):
    """2x3 그리드 시각화"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    times = history["time"]
    states = history["state"]
    controls = history["control"]
    ee_pos = history["ee_pos"]
    ee_ref = history["ee_ref"]
    ee_error = history["ee_error"]
    solve_time = history["solve_time"]

    # [0,0] 3D 궤적 + 팔 스냅샷
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot3D(states[:, 0], states[:, 1], np.zeros(len(states)), "b-", alpha=0.3, label="Base")
    ax.plot3D(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], "r-", linewidth=1.5, label="EE path")
    ax.plot3D(ee_ref[:, 0], ee_ref[:, 1], ee_ref[:, 2], "g--", linewidth=1.5, label="EE target")
    arm_step = max(1, len(times) // 6)
    for i in range(0, len(times), arm_step):
        a = 0.3 + 0.7 * (i / len(times))
        draw_arm_3d(ax, states[i], model, control=controls[i], alpha=a, linewidth=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory + Arm (DiffDrive)")
    ax.legend(fontsize=7, loc="upper left")

    # [0,1] EE 3D 추적 오차
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(times, ee_error, "r-", linewidth=1.5)
    ax.axhline(y=0.15, color="g", linestyle="--", alpha=0.5, label="Target (0.15m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE Error (m)")
    ax.set_title("EE 3D Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,2] 관절 각도 6개
    ax = fig.add_subplot(2, 3, 3)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i in range(6):
        ax.plot(times, np.rad2deg(states[:, 3 + i]), color=colors[i], label=f"q{i+1}")
    ax.axhline(y=166, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=-166, color="gray", linestyle=":", alpha=0.5, label="Limit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (deg)")
    ax.set_title("Joint Angles (6-DOF)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # [1,0] 제어 입력 8개
    ax = fig.add_subplot(2, 3, 4)
    ctrl_labels = ["v", "omega"] + [f"dq{i}" for i in range(1, 7)]
    for i, label in enumerate(ctrl_labels):
        ax.plot(times, controls[:, i], label=label, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Inputs (8D: DiffDrive + 6-DOF)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # [1,1] 계산 시간
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(times, solve_time, "b-", alpha=0.5)
    ax.axhline(
        y=np.mean(solve_time), color="r", linestyle="--",
        label=f"Mean: {np.mean(solve_time):.1f}ms",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,2] 메트릭 요약
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    ee_rmse = np.sqrt(np.mean(ee_error**2))
    ee_max = np.max(ee_error)
    mean_solve = np.mean(solve_time)
    max_solve = np.max(solve_time)

    metrics_text = (
        f"Base Type:     DiffDrive (Non-holonomic)\n"
        f"EE RMSE:       {ee_rmse:.4f} m\n"
        f"EE Max Error:  {ee_max:.4f} m\n"
        f"Mean Solve:    {mean_solve:.1f} ms\n"
        f"Max Solve:     {max_solve:.1f} ms\n"
    )

    if use_pose and "ori_error" in history:
        ori_rmse = np.sqrt(np.mean(history["ori_error"]**2))
        metrics_text += f"Ori RMSE:      {ori_rmse:.4f} rad\n"

    metrics_text += (
        f"\n"
        f"{'PASS' if ee_rmse < 0.15 else 'FAIL'}: EE RMSE < 0.15m\n"
        f"{'PASS' if mean_solve < 80 else 'FAIL'}: Solve < 80ms"
    )

    ax.text(
        0.1, 0.5, metrics_text, transform=ax.transAxes,
        fontsize=11, fontfamily="monospace", verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Performance Summary")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════
#  라이브 애니메이션 (사전 계산 → 부드러운 재생)
# ═══════════════════════════════════════════════════════

def run_live(model, controller, traj_fn, initial_state, params, duration,
             ema_alpha=0.4):
    """사전 계산 후 부드러운 3D 애니메이션 재생 (EMA 필터 적용)"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # ── 1단계: 시뮬레이션 사전 계산 ──
    print(f"  [Pre-computing simulation (EMA α={ema_alpha})...]")
    dt = params.dt
    n_steps = int(duration / dt)
    state = initial_state.copy()

    all_states = [state.copy()]
    all_controls = [np.zeros(model.control_dim)]
    all_ee_pos = [model.forward_kinematics(state).copy()]

    ref_positions = np.array([traj_fn(t)[:3] for t in np.arange(n_steps) * dt])

    filtered_ctrl = np.zeros(model.control_dim)

    for step in range(n_steps):
        t = step * dt
        ref = generate_reference_trajectory(traj_fn, t, params.N, dt)
        raw_control, _ = controller.compute_control(state, ref)

        # EMA 필터: 고주파 제어 노이즈 제거 → 부드러운 움직임
        filtered_ctrl = ema_alpha * raw_control + (1 - ema_alpha) * filtered_ctrl
        control = filtered_ctrl

        state = model.step(state, control, dt)
        state = model.normalize_state(state)

        all_states.append(state.copy())
        all_controls.append(control.copy())
        all_ee_pos.append(model.forward_kinematics(state).copy())

        if step % 100 == 0:
            print(f"    step {step}/{n_steps}")

    all_states = np.array(all_states)
    all_controls = np.array(all_controls)
    all_ee_pos = np.array(all_ee_pos)

    print(f"  [Done! {n_steps} steps pre-computed. Starting animation...]\n")

    # ── 2단계: 부드러운 애니메이션 재생 ──
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    margin = 0.3
    xlim = [ref_positions[:, 0].min() - margin, ref_positions[:, 0].max() + margin]
    ylim = [ref_positions[:, 1].min() - margin, ref_positions[:, 1].max() + margin]
    zlim = [-0.05, ref_positions[:, 2].max() + margin]

    def update(frame):
        ax.cla()

        s = all_states[frame]
        ctrl = all_controls[frame]
        ee = all_ee_pos[frame]
        t = frame * dt

        ee_ref = traj_fn(t)[:3]
        ee_err = np.sqrt(np.sum((ee - ee_ref) ** 2))

        # 바퀴 속도 정보
        v_l, v_r = compute_diffdrive_wheel_speeds(ctrl)

        ax.set_title(
            f"6-DOF DiffDrive | t={t:.1f}s | err={ee_err:.3f}m\n"
            f"v={ctrl[0]:.2f} m/s  omega={ctrl[1]:.2f} rad/s  "
            f"wheels=[L:{v_l:+.2f}, R:{v_r:+.2f}]",
            fontsize=10,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # 목표 궤적
        ax.plot3D(
            ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2],
            "g--", alpha=0.4, linewidth=1.5,
        )
        # EE 히스토리
        ax.plot3D(
            all_ee_pos[:frame + 1, 0],
            all_ee_pos[:frame + 1, 1],
            all_ee_pos[:frame + 1, 2],
            "r-", linewidth=1.5,
        )
        # 현재 목표점
        ax.scatter3D([ee_ref[0]], [ee_ref[1]], [ee_ref[2]],
                      color="green", s=80, marker="s")

        # 로봇 (베이스 + 팔) — 바퀴 속도 반영
        draw_arm_3d(ax, s, model, control=ctrl, alpha=0.9, linewidth=2.0)

    anim = FuncAnimation(  # noqa: F841
        fig, update, frames=n_steps,
        interval=dt * 1000,
        repeat=False,
    )
    plt.show()


# ═══════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Mobile Manipulator 6-DOF EE 3D Tracking Demo"
    )
    parser.add_argument(
        "--trajectory", type=str, default="ee_3d_circle",
        choices=["ee_3d_circle", "ee_3d_helix"],
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--ema", type=float, default=0.2,
                        help="EMA filter alpha (0=max smooth, 1=no filter)")
    parser.add_argument("--dial", action="store_true",
                        help="Use DIAL-MPPI (Diffusion Annealing) for smoother control")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("6-DOF DiffDrive Mobile Manipulator EE 3D Tracking".center(60))
    print("=" * 60)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Samples:    K={args.K}")
    print(f"  Live Mode:  {args.live}")
    print(f"  Pose Mode:  {args.pose}")
    print(f"  EMA alpha:  {args.ema}")
    print(f"  DIAL-MPPI:  {args.dial}")
    print(f"  Base Type:  DiffDrive (Non-holonomic)")
    print("=" * 60 + "\n")

    model = MobileManipulator6DOFKinematic()

    if args.dial:
        params = DIALMPPIParams(
            N=40,
            dt=0.05,
            K=args.K,
            lambda_=0.3,
            sigma=np.array([0.3, 0.3] + [0.8] * 6),
            Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
            R=np.array([0.1, 0.1] + [0.05] * 6),
            Qf=np.array([20.0, 20.0, 2.0] + [0.2] * 6),
            n_diffuse_init=8,
            n_diffuse=3,
            traj_diffuse_factor=0.5,
            horizon_diffuse_factor=0.9,
            device="cpu",
        )
    else:
        params = MPPIParams(
            N=40,
            dt=0.05,
            K=args.K,
            lambda_=0.3,
            sigma=np.array([0.3, 0.3] + [0.8] * 6),
            Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
            R=np.array([0.1, 0.1] + [0.05] * 6),
            Qf=np.array([20.0, 20.0, 2.0] + [0.2] * 6),
            device="cpu",
        )

    if args.pose:
        cost_fn = CompositeMPPICost([
            EndEffectorPoseTrackingCost(model, pos_weight=150.0, ori_weight=15.0),
            EndEffectorPoseTerminalCost(model, pos_weight=300.0, ori_weight=30.0),
            ControlEffortCost(R=np.array([0.1, 0.1] + [0.05] * 6)),
            ControlRateCost(R_rate=np.array([0.3, 0.3] + [0.1] * 6)),
            JointLimitCost(joint_indices=tuple(range(3, 9)), joint_limits=((-2.9, 2.9),) * 6, weight=5.0),
        ])
    else:
        cost_fn = CompositeMPPICost([
            EndEffector3DTrackingCost(model, weight=200.0),
            EndEffector3DTerminalCost(model, weight=400.0),
            ControlEffortCost(R=np.array([0.05, 0.05] + [0.02] * 6)),
            ControlRateCost(R_rate=np.array([0.3, 0.3] + [0.1] * 6)),
            JointLimitCost(joint_indices=tuple(range(3, 9)), joint_limits=((-2.9, 2.9),) * 6, weight=5.0),
        ])

    if args.dial:
        controller = DIALMPPIController(model, params, cost_function=cost_fn)
        print(f"  Controller: DIAL-MPPI (Diffusion Annealing) + EMA (α={args.ema})")
    else:
        controller = MPPIController(model, params, cost_function=cost_fn)
        print(f"  Controller: Vanilla MPPI + EMA (α={args.ema})")
    traj_fn = create_trajectory_function(args.trajectory)

    initial_state = np.zeros(9)
    ee_init = model.forward_kinematics(initial_state)
    print(f"  Initial base: ({initial_state[0]:.2f}, {initial_state[1]:.2f})")
    print(f"  Initial EE:   ({ee_init[0]:.3f}, {ee_init[1]:.3f}, {ee_init[2]:.3f})")
    print(f"  z_mount={model.z_mount}m\n")

    if args.live:
        run_live(model, controller, traj_fn, initial_state, params, args.duration,
                 ema_alpha=args.ema)
    else:
        history = run_simulation(
            model, controller, traj_fn, initial_state, params, args.duration, args.pose,
            ema_alpha=args.ema,
        )

        ee_rmse = np.sqrt(np.mean(history["ee_error"] ** 2))
        ee_max = np.max(history["ee_error"])
        mean_solve = np.mean(history["solve_time"])

        print("\n" + "=" * 60)
        print("Results".center(60))
        print("=" * 60)
        print(f"  EE RMSE:       {ee_rmse:.4f} m")
        print(f"  EE Max Error:  {ee_max:.4f} m")
        print(f"  Mean Solve:    {mean_solve:.1f} ms")

        if args.pose and "ori_error" in history:
            ori_rmse = np.sqrt(np.mean(history["ori_error"] ** 2))
            print(f"  Ori RMSE:      {ori_rmse:.4f} rad")

        rmse_status = "PASS" if ee_rmse < 0.15 else "FAIL"
        solve_status = "PASS" if mean_solve < 80 else "FAIL"
        print(f"  [{rmse_status}] EE RMSE < 0.15m")
        print(f"  [{solve_status}] Mean Solve < 80ms")
        print("=" * 60 + "\n")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_results(
            history, model,
            title=f"6-DOF DiffDrive Mobile Manipulator - {args.trajectory}",
            use_pose=args.pose,
        )
        os.makedirs("plots", exist_ok=True)
        fig.savefig("plots/mobile_manipulator_6dof_tracking.png", dpi=150, bbox_inches="tight")
        print("Plot saved to plots/mobile_manipulator_6dof_tracking.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
