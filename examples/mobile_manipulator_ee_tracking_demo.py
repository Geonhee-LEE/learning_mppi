#!/usr/bin/env python3
"""
Mobile Manipulator End-Effector 추적 데모

2-DOF Planar Arm + DiffDrive Base로 EE 궤적 추적.

Usage:
    python examples/mobile_manipulator_ee_tracking_demo.py --trajectory ee_circle --live
    python examples/mobile_manipulator_ee_tracking_demo.py --trajectory ee_figure8 --duration 30
    python examples/mobile_manipulator_ee_tracking_demo.py --K 2048 --seed 42
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.mobile_manipulator_kinematic import (
    MobileManipulatorKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    EndEffectorTrackingCost,
    EndEffectorTerminalCost,
    ControlEffortCost,
    ControlRateCost,
    JointLimitCost,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


def draw_arm(ax, state, model, color="blue", alpha=1.0, linewidth=2.0):
    """로봇 팔 그리기 (shoulder → elbow → EE)"""
    x, y, theta, q1, q2 = state[:5]

    # 관절 위치 계산
    shoulder = np.array([x, y])
    phi1 = theta + q1
    elbow = shoulder + model.L1 * np.array([np.cos(phi1), np.sin(phi1)])
    phi12 = phi1 + q2
    ee = elbow + model.L2 * np.array([np.cos(phi12), np.sin(phi12)])

    # 링크 그리기
    ax.plot(
        [shoulder[0], elbow[0]], [shoulder[1], elbow[1]],
        "-", color=color, linewidth=linewidth, alpha=alpha,
    )
    ax.plot(
        [elbow[0], ee[0]], [elbow[1], ee[1]],
        "-", color=color, linewidth=linewidth * 0.8, alpha=alpha,
    )

    # 관절/EE 마커
    ax.plot(*shoulder, "o", color=color, markersize=6, alpha=alpha)
    ax.plot(*elbow, "o", color=color, markersize=5, alpha=alpha)
    ax.plot(*ee, "*", color="red", markersize=10, alpha=alpha)


def run_simulation(model, controller, traj_fn, initial_state, params, duration):
    """시뮬레이션 실행 및 히스토리 반환"""
    dt = params.dt
    n_steps = int(duration / dt)
    state = initial_state.copy()

    history = {
        "time": [],
        "state": [],
        "control": [],
        "ee_pos": [],
        "ee_ref": [],
        "ee_error": [],
        "solve_time": [],
    }

    for step in range(n_steps):
        t = step * dt
        ref = generate_reference_trajectory(traj_fn, t, params.N, dt)

        t0 = time.perf_counter()
        control, info = controller.compute_control(state, ref)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # EE 위치/오차
        ee_pos = model.forward_kinematics(state)
        ee_ref = ref[0, :2]
        ee_error = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))

        # 기록
        history["time"].append(t)
        history["state"].append(state.copy())
        history["control"].append(control.copy())
        history["ee_pos"].append(ee_pos.copy())
        history["ee_ref"].append(ee_ref.copy())
        history["ee_error"].append(ee_error)
        history["solve_time"].append(solve_ms)

        # 스텝
        state = model.step(state, control, dt)
        state = model.normalize_state(state)

        if step % 50 == 0:
            print(
                f"  t={t:5.1f}s | EE err={ee_error:.4f}m | "
                f"solve={solve_ms:.1f}ms | q=({state[3]:.2f}, {state[4]:.2f})"
            )

    # numpy 배열로 변환
    for key in history:
        history[key] = np.array(history[key])

    return history


def plot_results(history, model, title="Mobile Manipulator EE Tracking"):
    """2x3 그리드 시각화"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    times = history["time"]
    states = history["state"]
    controls = history["control"]
    ee_pos = history["ee_pos"]
    ee_ref = history["ee_ref"]
    ee_error = history["ee_error"]
    solve_time = history["solve_time"]

    # [0,0] XY 궤적 + 팔
    ax = axes[0, 0]
    ax.plot(states[:, 0], states[:, 1], "b-", alpha=0.5, label="Base path")
    ax.plot(ee_pos[:, 0], ee_pos[:, 1], "r-", linewidth=1.5, label="EE path")
    ax.plot(ee_ref[:, 0], ee_ref[:, 1], "g--", linewidth=1.5, label="EE target")
    # 팔 그리기 (매 N스텝)
    arm_step = max(1, len(times) // 8)
    for i in range(0, len(times), arm_step):
        alpha = 0.3 + 0.7 * (i / len(times))
        draw_arm(ax, states[i], model, color="blue", alpha=alpha, linewidth=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory + Arm")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # [0,1] EE 추적 오차
    ax = axes[0, 1]
    ax.plot(times, ee_error, "r-", linewidth=1.5)
    ax.axhline(y=0.1, color="g", linestyle="--", alpha=0.5, label="Target (0.1m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE Error (m)")
    ax.set_title("End-Effector Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,2] 관절 각도
    ax = axes[0, 2]
    ax.plot(times, np.rad2deg(states[:, 3]), "b-", label="q1")
    ax.plot(times, np.rad2deg(states[:, 4]), "r-", label="q2")
    ax.axhline(y=166, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=-166, color="gray", linestyle=":", alpha=0.5, label="Joint limit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (deg)")
    ax.set_title("Joint Angles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] 제어 입력
    ax = axes[1, 0]
    labels = ["v (m/s)", "omega (rad/s)", "dq1 (rad/s)", "dq2 (rad/s)"]
    colors = ["blue", "orange", "green", "red"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(times, controls[:, i], color=color, label=label, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Inputs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,1] 계산 시간
    ax = axes[1, 1]
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
    ax = axes[1, 2]
    ax.axis("off")
    ee_rmse = np.sqrt(np.mean(ee_error**2))
    ee_max = np.max(ee_error)
    mean_solve = np.mean(solve_time)
    max_solve = np.max(solve_time)

    metrics_text = (
        f"EE RMSE:       {ee_rmse:.4f} m\n"
        f"EE Max Error:  {ee_max:.4f} m\n"
        f"Mean Solve:    {mean_solve:.1f} ms\n"
        f"Max Solve:     {max_solve:.1f} ms\n"
        f"\n"
        f"{'PASS' if ee_rmse < 0.1 else 'FAIL'}: EE RMSE < 0.1m\n"
        f"{'PASS' if mean_solve < 50 else 'FAIL'}: Solve < 50ms"
    )
    ax.text(
        0.1, 0.5, metrics_text, transform=ax.transAxes,
        fontsize=12, fontfamily="monospace", verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Performance Summary")

    plt.tight_layout()
    return fig


def run_live(model, controller, traj_fn, initial_state, params, duration):
    """실시간 애니메이션"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    dt = params.dt
    n_steps = int(duration / dt)
    state = initial_state.copy()

    # 사전 계산: 전체 EE 목표 궤적
    ref_times = np.arange(n_steps) * dt
    ref_positions = np.array([traj_fn(t)[:2] for t in ref_times])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Mobile Manipulator EE Tracking (Live)")

    # 전체 목표 궤적 그리기
    ax.plot(ref_positions[:, 0], ref_positions[:, 1], "g--", alpha=0.5, label="EE target")

    # 동적 요소
    (base_line,) = ax.plot([], [], "b-", alpha=0.4, linewidth=1, label="Base path")
    (ee_line,) = ax.plot([], [], "r-", linewidth=1.5, label="EE path")
    (link1_line,) = ax.plot([], [], "b-", linewidth=3)
    (link2_line,) = ax.plot([], [], "c-", linewidth=2.5)
    (shoulder_dot,) = ax.plot([], [], "ko", markersize=7)
    (elbow_dot,) = ax.plot([], [], "bo", markersize=5)
    (ee_dot,) = ax.plot([], [], "r*", markersize=12)
    (target_dot,) = ax.plot([], [], "gs", markersize=8, alpha=0.7)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    base_xs, base_ys = [], []
    ee_xs, ee_ys = [], []

    # 축 범위 설정
    margin = 0.3
    x_range = [
        ref_positions[:, 0].min() - model.L1 - model.L2 - margin,
        ref_positions[:, 0].max() + model.L1 + model.L2 + margin,
    ]
    y_range = [
        ref_positions[:, 1].min() - model.L1 - model.L2 - margin,
        ref_positions[:, 1].max() + model.L1 + model.L2 + margin,
    ]
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.legend(loc="upper right", fontsize=8)

    nonlocal_state = {"state": state}

    def update(frame):
        s = nonlocal_state["state"]
        t = frame * dt
        ref = generate_reference_trajectory(traj_fn, t, params.N, dt)
        control, _ = controller.compute_control(s, ref)

        ee_pos = model.forward_kinematics(s)
        ee_ref = ref[0, :2]
        ee_err = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))

        # 팔 관절 계산
        x, y, theta, q1, q2 = s[:5]
        shoulder = np.array([x, y])
        phi1 = theta + q1
        elbow = shoulder + model.L1 * np.array([np.cos(phi1), np.sin(phi1)])
        phi12 = phi1 + q2
        ee = elbow + model.L2 * np.array([np.cos(phi12), np.sin(phi12)])

        # 업데이트
        base_xs.append(x)
        base_ys.append(y)
        ee_xs.append(ee[0])
        ee_ys.append(ee[1])

        base_line.set_data(base_xs, base_ys)
        ee_line.set_data(ee_xs, ee_ys)
        link1_line.set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
        link2_line.set_data([elbow[0], ee[0]], [elbow[1], ee[1]])
        shoulder_dot.set_data([shoulder[0]], [shoulder[1]])
        elbow_dot.set_data([elbow[0]], [elbow[1]])
        ee_dot.set_data([ee[0]], [ee[1]])
        target_dot.set_data([ee_ref[0]], [ee_ref[1]])

        time_text.set_text(
            f"t={t:.1f}s | EE err={ee_err:.3f}m | "
            f"q=({np.rad2deg(q1):.0f}, {np.rad2deg(q2):.0f}) deg"
        )

        # 스텝
        s = model.step(s, control, dt)
        s = model.normalize_state(s)
        nonlocal_state["state"] = s

        return (
            base_line, ee_line, link1_line, link2_line,
            shoulder_dot, elbow_dot, ee_dot, target_dot, time_text,
        )

    anim = FuncAnimation(
        fig, update, frames=n_steps, interval=dt * 1000, blit=True, repeat=False,
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Mobile Manipulator EE Tracking Demo"
    )
    parser.add_argument(
        "--trajectory", type=str, default="ee_circle",
        choices=["ee_circle", "ee_figure8"],
        help="EE reference trajectory type",
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Duration (s)")
    parser.add_argument("--live", action="store_true", help="Live animation mode")
    parser.add_argument("--K", type=int, default=1024, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot display")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("Mobile Manipulator EE Tracking Demo".center(60))
    print("=" * 60)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Samples:    K={args.K}")
    print(f"  Live Mode:  {args.live}")
    print("=" * 60 + "\n")

    # 1. 모델
    model = MobileManipulatorKinematic(L1=0.3, L2=0.25)

    # 2. MPPI 파라미터
    params = MPPIParams(
        N=30,
        dt=0.05,
        K=args.K,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5, 1.5, 1.5]),
        Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
        R=np.array([0.1, 0.1, 0.05, 0.05]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2, 0.2]),
        device="cpu",
    )

    # 3. EE 비용 함수
    cost_fn = CompositeMPPICost([
        EndEffectorTrackingCost(model, weight=100.0),
        EndEffectorTerminalCost(model, weight=200.0),
        ControlEffortCost(R=np.array([0.1, 0.1, 0.05, 0.05])),
        ControlRateCost(R_rate=np.array([0.05, 0.05, 0.02, 0.02])),
        JointLimitCost(joint_indices=(3, 4), weight=5.0),
    ])

    # 4. MPPI 컨트롤러
    controller = MPPIController(model, params, cost_function=cost_fn)

    # 5. 궤적 함수
    traj_fn = create_trajectory_function(args.trajectory)

    # 6. 초기 상태
    initial_state = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    ee_init = model.forward_kinematics(initial_state)
    print(f"  Initial base: ({initial_state[0]:.2f}, {initial_state[1]:.2f})")
    print(f"  Initial EE:   ({ee_init[0]:.2f}, {ee_init[1]:.2f})")
    print(f"  L1={model.L1}m, L2={model.L2}m, reach={model.L1+model.L2}m\n")

    if args.live and not args.no_plot:
        run_live(model, controller, traj_fn, initial_state, params, args.duration)
    else:
        # 시뮬레이션 실행
        history = run_simulation(
            model, controller, traj_fn, initial_state, params, args.duration
        )

        # 결과 요약
        ee_rmse = np.sqrt(np.mean(history["ee_error"] ** 2))
        ee_max = np.max(history["ee_error"])
        mean_solve = np.mean(history["solve_time"])

        print("\n" + "=" * 60)
        print("Results".center(60))
        print("=" * 60)
        print(f"  EE RMSE:       {ee_rmse:.4f} m")
        print(f"  EE Max Error:  {ee_max:.4f} m")
        print(f"  Mean Solve:    {mean_solve:.1f} ms")
        rmse_status = "PASS" if ee_rmse < 0.1 else "FAIL"
        solve_status = "PASS" if mean_solve < 50 else "FAIL"
        print(f"  [{rmse_status}] EE RMSE < 0.1m")
        print(f"  [{solve_status}] Mean Solve < 50ms")
        print("=" * 60 + "\n")

        # 시각화
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_results(history, model, title=f"Mobile Manipulator - {args.trajectory}")
        fig.savefig("plots/mobile_manipulator_ee_tracking.png", dpi=150, bbox_inches="tight")
        print("Plot saved to plots/mobile_manipulator_ee_tracking.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
