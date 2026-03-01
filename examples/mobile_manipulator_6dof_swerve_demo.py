#!/usr/bin/env python3
"""
Mobile Manipulator 6-DOF Swerve EE 3D 추적 데모

UR5-style 6-DOF Arm + Swerve (Holonomic) Base로 EE 3D 궤적 추적.
DiffDrive 대비 횡방향 이동이 가능하여 EE 추적 성능이 향상됨.

개선사항:
  - 스티어링 각도 시각화: 제어 (vx, vy, ω) → 각 바퀴 조향각 계산
  - Smooth 모션: 시뮬레이션 사전 계산 → 렌더링 분리

Usage:
    python examples/mobile_manipulator_6dof_swerve_demo.py --trajectory ee_3d_circle --live
    python examples/mobile_manipulator_6dof_swerve_demo.py --trajectory ee_3d_helix --live
    python examples/mobile_manipulator_6dof_swerve_demo.py --trajectory ee_3d_helix --pose
    python examples/mobile_manipulator_6dof_swerve_demo.py --duration 20
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.mobile_manipulator_6dof_swerve_kinematic import (
    MobileManipulator6DOFSwerveKinematic,
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
#  Swerve 바퀴 조향각 계산
# ═══════════════════════════════════════════════════════

# 바퀴 위치 (body frame), base_size=0.15 기준
_BASE_SIZE = 0.15
_WHEEL_POSITIONS_BODY = np.array([
    [-_BASE_SIZE, -_BASE_SIZE],  # rear-left
    [_BASE_SIZE, -_BASE_SIZE],   # front-left
    [_BASE_SIZE, _BASE_SIZE],    # front-right
    [-_BASE_SIZE, _BASE_SIZE],   # rear-right
])


def compute_wheel_steering_angles(control):
    """
    Swerve 각 바퀴의 조향각 계산.

    v_wheel_i = [vx, vy] + ω × r_i
    steering_i = atan2(v_wheel_y, v_wheel_x)

    Args:
        control: (9,) [vx, vy, ω, dq1..6]

    Returns:
        steering_angles: (4,) 각 바퀴 조향각 (rad)
    """
    vx, vy, omega = control[0], control[1], control[2]
    angles = np.zeros(4)
    for i, (rx, ry) in enumerate(_WHEEL_POSITIONS_BODY):
        # ω × r = [-ω*ry, ω*rx]
        vwx = vx - omega * ry
        vwy = vy + omega * rx
        speed = np.sqrt(vwx**2 + vwy**2)
        if speed > 1e-4:
            angles[i] = np.arctan2(vwy, vwx)
        else:
            angles[i] = 0.0
    return angles


# ═══════════════════════════════════════════════════════
#  시각화 함수
# ═══════════════════════════════════════════════════════

def draw_base_3d(ax, state, control=None, base_size=0.15, alpha=1.0):
    """3D Swerve 베이스 그리기 (사각형 + 4개 조향 바퀴 + heading)"""
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

    # 조향각 계산
    if control is not None:
        steer_angles = compute_wheel_steering_angles(control)
    else:
        steer_angles = np.zeros(4)

    # 4개 바퀴: 각 바퀴를 조향각으로 회전시켜 표시
    wheel_r = base_size * 0.45
    wz = 0.005
    for i, corner_body in enumerate(corners_body):
        cb = corner_body.copy()
        cb[2] = wz
        cw = R @ cb + np.array([x, y, 0])

        # 바퀴 방향 = base heading(θ) + 각 바퀴 조향각(δ_i)
        wheel_angle = theta + steer_angles[i]
        cw_cos = np.cos(wheel_angle)
        cw_sin = np.sin(wheel_angle)

        # 바퀴 선분 (조향 방향)
        front = cw + np.array([wheel_r * cw_cos, wheel_r * cw_sin, 0])
        rear = cw - np.array([wheel_r * cw_cos, wheel_r * cw_sin, 0])
        ax.plot3D(
            [rear[0], front[0]], [rear[1], front[1]], [rear[2], front[2]],
            "-", color="darkorange", linewidth=5, alpha=alpha, solid_capstyle="round",
        )

        # 바퀴 축 (조향 직교 방향, 짧은 선)
        perp_cos = np.cos(wheel_angle + np.pi / 2)
        perp_sin = np.sin(wheel_angle + np.pi / 2)
        left = cw + np.array([wheel_r * 0.4 * perp_cos, wheel_r * 0.4 * perp_sin, 0])
        right = cw - np.array([wheel_r * 0.4 * perp_cos, wheel_r * 0.4 * perp_sin, 0])
        ax.plot3D(
            [left[0], right[0]], [left[1], right[1]], [left[2], right[2]],
            "-", color="darkorange", linewidth=2.5, alpha=alpha * 0.7,
        )

        ax.scatter3D([cw[0]], [cw[1]], [cw[2]],
                      color="darkorange", s=30, alpha=alpha)

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
    """3D 로봇 팔 + Swerve 베이스 그리기"""
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

def plot_results(history, model, title="6-DOF Swerve Mobile Manipulator", use_pose=False):
    """2x3 시각화"""
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

    # [0,0] 3D 궤적
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
    ax.set_title("3D Trajectory + Arm (Swerve)")
    ax.legend(fontsize=7, loc="upper left")

    # [0,1] EE 오차
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(times, ee_error, "r-", linewidth=1.5)
    ax.axhline(y=0.15, color="g", linestyle="--", alpha=0.5, label="Target (0.15m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE Error (m)")
    ax.set_title("EE 3D Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,2] 관절 각도
    ax = fig.add_subplot(2, 3, 3)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i in range(6):
        ax.plot(times, np.rad2deg(states[:, 3 + i]), color=colors[i], label=f"q{i+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (deg)")
    ax.set_title("Joint Angles (6-DOF)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # [1,0] 제어 입력 (9D)
    ax = fig.add_subplot(2, 3, 4)
    ctrl_labels = ["vx", "vy", "omega"] + [f"dq{i}" for i in range(1, 7)]
    for i, label in enumerate(ctrl_labels):
        ax.plot(times, controls[:, i], label=label, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Inputs (9D: Swerve + 6-DOF)")
    ax.legend(fontsize=6, ncol=3)
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

    # [1,2] 메트릭
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    ee_rmse = np.sqrt(np.mean(ee_error**2))
    ee_max = np.max(ee_error)
    mean_solve = np.mean(solve_time)
    max_solve = np.max(solve_time)

    metrics_text = (
        f"Base Type:     Swerve (Holonomic)\n"
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

        # EMA 필터
        filtered_ctrl = ema_alpha * raw_control + (1 - ema_alpha) * filtered_ctrl
        control = filtered_ctrl

        state = model.step(state, control, dt)
        state = model.normalize_state(state)

        all_states.append(state.copy())
        all_controls.append(control.copy())
        all_ee_pos.append(model.forward_kinematics(state).copy())

        if step % 100 == 0:
            print(f"    step {step}/{n_steps}")

    all_states = np.array(all_states)    # (n_steps+1, 9)
    all_controls = np.array(all_controls)  # (n_steps+1, 9)
    all_ee_pos = np.array(all_ee_pos)    # (n_steps+1, 3)

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

        # 스티어링 각도 정보
        steer = compute_wheel_steering_angles(ctrl)
        steer_str = ", ".join(f"{np.rad2deg(a):+.0f}" for a in steer)

        ax.set_title(
            f"6-DOF Swerve | t={t:.1f}s | err={ee_err:.3f}m\n"
            f"steer=[{steer_str}] deg",
            fontsize=11,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # 목표 궤적 (전체)
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

        # 로봇 (베이스 + 팔) — 조향각 반영
        draw_arm_3d(ax, s, model, control=ctrl, alpha=0.9, linewidth=2.0)

    anim = FuncAnimation(  # noqa: F841
        fig, update, frames=n_steps,
        interval=dt * 1000,  # 실시간 속도
        repeat=False,
    )
    plt.show()


# ═══════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="6-DOF Swerve Mobile Manipulator EE 3D Tracking Demo"
    )
    parser.add_argument(
        "--trajectory", type=str, default="ee_3d_circle",
        choices=["ee_3d_circle", "ee_3d_helix"],
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--ema", type=float, default=0.3,
                        help="EMA filter alpha (0=max smooth, 1=no filter)")
    parser.add_argument("--dial", action="store_true",
                        help="Use DIAL-MPPI (Diffusion Annealing) for smoother control")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("6-DOF Swerve Mobile Manipulator EE 3D Tracking".center(60))
    print("=" * 60)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Samples:    K={args.K}")
    print(f"  Live Mode:  {args.live}")
    print(f"  Pose Mode:  {args.pose}")
    print(f"  EMA alpha:  {args.ema}")
    print(f"  DIAL-MPPI:  {args.dial}")
    print(f"  Base Type:  Swerve (Holonomic)")
    print("=" * 60 + "\n")

    model = MobileManipulator6DOFSwerveKinematic()

    if args.dial:
        params = DIALMPPIParams(
            N=40,
            dt=0.05,
            K=args.K,
            lambda_=0.3,
            sigma=np.array([0.3, 0.3, 0.3] + [0.8] * 6),
            Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
            R=np.array([0.1, 0.1, 0.1] + [0.05] * 6),
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
            sigma=np.array([0.3, 0.3, 0.3] + [0.8] * 6),
            Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
            R=np.array([0.1, 0.1, 0.1] + [0.05] * 6),
            Qf=np.array([20.0, 20.0, 2.0] + [0.2] * 6),
            device="cpu",
        )

    if args.pose:
        cost_fn = CompositeMPPICost([
            EndEffectorPoseTrackingCost(model, pos_weight=150.0, ori_weight=15.0),
            EndEffectorPoseTerminalCost(model, pos_weight=300.0, ori_weight=30.0),
            ControlEffortCost(R=np.array([0.1, 0.1, 0.1] + [0.05] * 6)),
            ControlRateCost(R_rate=np.array([0.3, 0.3, 0.3] + [0.1] * 6)),
            JointLimitCost(joint_indices=tuple(range(3, 9)), joint_limits=((-2.9, 2.9),) * 6, weight=5.0),
        ])
    else:
        cost_fn = CompositeMPPICost([
            EndEffector3DTrackingCost(model, weight=200.0),
            EndEffector3DTerminalCost(model, weight=400.0),
            ControlEffortCost(R=np.array([0.05, 0.05, 0.05] + [0.02] * 6)),
            ControlRateCost(R_rate=np.array([0.3, 0.3, 0.3] + [0.1] * 6)),
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
            title=f"6-DOF Swerve Mobile Manipulator - {args.trajectory}",
            use_pose=args.pose,
        )
        os.makedirs("plots", exist_ok=True)
        fig.savefig("plots/mobile_manipulator_6dof_swerve_tracking.png", dpi=150, bbox_inches="tight")
        print("Plot saved to plots/mobile_manipulator_6dof_swerve_tracking.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
