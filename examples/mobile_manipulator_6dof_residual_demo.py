#!/usr/bin/env python3
"""
6-DOF Mobile Manipulator Residual Dynamics 3-Way 비교 데모

3-way 비교:
  1. Kinematic MPPI:  기구학 모델로 계획, Dynamic 환경에서 실행 (모델 미스매치)
  2. Residual MPPI:   기구학+NN 모델로 계획, Dynamic 환경에서 실행 (보정됨)
  3. Oracle MPPI:     Dynamic 모델로 계획, Dynamic 환경에서 실행 (이론적 최적)

Usage:
    # 학습된 모델이 필요 (먼저 train_6dof_residual.py 실행)
    PYTHONPATH=. python examples/mobile_manipulator_6dof_residual_demo.py
    PYTHONPATH=. python examples/mobile_manipulator_6dof_residual_demo.py --duration 15
    PYTHONPATH=. python examples/mobile_manipulator_6dof_residual_demo.py --dial
    PYTHONPATH=. python examples/mobile_manipulator_6dof_residual_demo.py --no-trained
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
from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
    MobileManipulator6DOFDynamic,
)
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, DIALMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    EndEffector3DTrackingCost,
    EndEffector3DTerminalCost,
    ControlEffortCost,
    ControlRateCost,
    JointLimitCost,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


def create_oracle_residual(kin_model, dyn_model):
    """Oracle residual: dynamic - kinematic (치트용)"""
    def residual_fn(state, control):
        kin_dot = kin_model.forward_dynamics(state, control)
        dyn_dot = dyn_model.forward_dynamics(state, control)
        return dyn_dot - kin_dot
    return residual_fn


def create_params(K, use_dial):
    """MPPI 파라미터 생성"""
    if use_dial:
        return DIALMPPIParams(
            N=40,
            dt=0.05,
            K=K,
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
        return MPPIParams(
            N=40,
            dt=0.05,
            K=K,
            lambda_=0.3,
            sigma=np.array([0.3, 0.3] + [0.8] * 6),
            Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
            R=np.array([0.1, 0.1] + [0.05] * 6),
            Qf=np.array([20.0, 20.0, 2.0] + [0.2] * 6),
            device="cpu",
        )


def create_cost_fn(model):
    """비용 함수 생성"""
    return CompositeMPPICost([
        EndEffector3DTrackingCost(model, weight=200.0),
        EndEffector3DTerminalCost(model, weight=400.0),
        ControlEffortCost(R=np.array([0.05, 0.05] + [0.02] * 6)),
        ControlRateCost(R_rate=np.array([0.3, 0.3] + [0.1] * 6)),
        JointLimitCost(
            joint_indices=tuple(range(3, 9)),
            joint_limits=((-2.9, 2.9),) * 6,
            weight=5.0,
        ),
    ])


def run_3way_simulation(controllers, fk_model, env_model, traj_fn, params, duration,
                        ema_alpha=0.2):
    """
    3-way 시뮬레이션 실행.

    모든 컨트롤러가 동일한 env_model(dynamic)에서 실행됨.

    Args:
        controllers: dict {name: controller}
        fk_model: FK 계산용 모델 (kinematic)
        env_model: 환경 동역학 모델 (dynamic)
        traj_fn: 궤적 함수
        params: MPPI 파라미터
        duration: 시뮬레이션 시간

    Returns:
        histories: dict {name: history}
    """
    dt = params.dt
    n_steps = int(duration / dt)

    histories = {}
    for name in controllers:
        histories[name] = {
            "time": [], "state": [], "control": [],
            "ee_pos": [], "ee_ref": [], "ee_error": [], "solve_time": [],
        }

    # 각 컨트롤러별 독립적인 상태
    states = {name: np.zeros(9) for name in controllers}
    filtered_ctrls = {name: np.zeros(8) for name in controllers}

    for step in range(n_steps):
        t = step * dt
        ref = generate_reference_trajectory(traj_fn, t, params.N, dt)

        for name, controller in controllers.items():
            state = states[name]

            t0 = time.perf_counter()
            raw_control, _ = controller.compute_control(state, ref)
            solve_ms = (time.perf_counter() - t0) * 1000.0

            # EMA 필터
            filtered_ctrls[name] = (
                ema_alpha * raw_control + (1 - ema_alpha) * filtered_ctrls[name]
            )
            control = filtered_ctrls[name]

            ee_pos = fk_model.forward_kinematics(state)
            ee_ref = ref[0, :3]
            ee_error = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))

            histories[name]["time"].append(t)
            histories[name]["state"].append(state.copy())
            histories[name]["control"].append(control.copy())
            histories[name]["ee_pos"].append(ee_pos.copy())
            histories[name]["ee_ref"].append(ee_ref.copy())
            histories[name]["ee_error"].append(ee_error)
            histories[name]["solve_time"].append(solve_ms)

            # 환경에서 실행 (dynamic)
            next_state = env_model.step(state, control, dt)
            states[name] = env_model.normalize_state(next_state)

        if step % 50 == 0:
            errors = {n: histories[n]["ee_error"][-1] for n in controllers}
            err_str = " | ".join(f"{n}: {e:.4f}m" for n, e in errors.items())
            print(f"  t={t:5.1f}s | {err_str}")

    # 리스트 → numpy 변환
    for name in controllers:
        for key in histories[name]:
            histories[name][key] = np.array(histories[name][key])

    return histories


def plot_3way_comparison(histories, fk_model, title="6-DOF Residual 3-Way Comparison"):
    """2×3 비교 플롯"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = {"Kinematic": "tab:blue", "Residual": "tab:orange", "Oracle": "tab:green"}
    names = list(histories.keys())

    # [0,0] 3D 궤적 비교
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    for name in names:
        h = histories[name]
        ax.plot3D(
            h["ee_pos"][:, 0], h["ee_pos"][:, 1], h["ee_pos"][:, 2],
            "-", color=colors.get(name, "gray"), linewidth=1.5, label=f"{name} EE",
        )
    # Reference (from first controller)
    ref = histories[names[0]]
    ax.plot3D(
        ref["ee_ref"][:, 0], ref["ee_ref"][:, 1], ref["ee_ref"][:, 2],
        "k--", linewidth=1.0, alpha=0.5, label="Reference",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("EE 3D Trajectory")
    ax.legend(fontsize=7)

    # [0,1] EE 추적 오차 비교
    ax = fig.add_subplot(2, 3, 2)
    for name in names:
        h = histories[name]
        ax.plot(h["time"], h["ee_error"], "-", color=colors.get(name, "gray"),
                linewidth=1.5, label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE Error (m)")
    ax.set_title("EE 3D Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,2] Residual 분석 (Kinematic vs Dynamic 차이 시각화)
    ax = fig.add_subplot(2, 3, 3)
    kin_err = histories[names[0]]["ee_error"]
    if len(names) > 1:
        res_err = histories[names[1]]["ee_error"]
    else:
        res_err = kin_err
    improvement = kin_err - res_err
    times = histories[names[0]]["time"]
    ax.fill_between(times, 0, improvement, alpha=0.3, color="tab:orange",
                     label="Residual improvement")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error Reduction (m)")
    ax.set_title("Residual vs Kinematic Improvement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] 제어 입력 비교 (base v)
    ax = fig.add_subplot(2, 3, 4)
    for name in names:
        h = histories[name]
        ax.plot(h["time"], h["control"][:, 0], "-", color=colors.get(name, "gray"),
                linewidth=1.0, alpha=0.7, label=f"{name} v")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Base v (m/s)")
    ax.set_title("Base Linear Velocity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [1,1] 관절 궤적 비교 (q2 — 중력 영향 큰 관절)
    ax = fig.add_subplot(2, 3, 5)
    for name in names:
        h = histories[name]
        ax.plot(h["time"], np.rad2deg(h["state"][:, 4]), "-",
                color=colors.get(name, "gray"), linewidth=1.0, label=f"{name} q2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("q2 (deg)")
    ax.set_title("Shoulder Pitch (q2) — Gravity Effect")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,2] 메트릭 요약 표
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")

    lines = [f"{'Model':<12} {'RMSE':>8} {'MaxErr':>8} {'MeanSolve':>10}"]
    lines.append("-" * 42)
    for name in names:
        h = histories[name]
        rmse = np.sqrt(np.mean(h["ee_error"] ** 2))
        max_err = np.max(h["ee_error"])
        mean_solve = np.mean(h["solve_time"])
        lines.append(f"{name:<12} {rmse:>8.4f} {max_err:>8.4f} {mean_solve:>8.1f}ms")

    # Improvement 계산
    if len(names) >= 2:
        rmse_kin = np.sqrt(np.mean(histories[names[0]]["ee_error"] ** 2))
        rmse_res = np.sqrt(np.mean(histories[names[1]]["ee_error"] ** 2))
        improvement_pct = (1 - rmse_res / rmse_kin) * 100
        lines.append("")
        lines.append(f"Residual improvement: {improvement_pct:.1f}%")

    metrics_text = "\n".join(lines)
    ax.text(
        0.05, 0.5, metrics_text, transform=ax.transAxes,
        fontsize=10, fontfamily="monospace", verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Performance Summary")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="6-DOF Residual Dynamics 3-Way Comparison Demo"
    )
    parser.add_argument("--trajectory", type=str, default="ee_3d_circle",
                        choices=["ee_3d_circle", "ee_3d_helix"])
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--ema", type=float, default=0.2)
    parser.add_argument("--dial", action="store_true",
                        help="Use DIAL-MPPI instead of Vanilla MPPI")
    parser.add_argument("--model-path", type=str,
                        default="models/learned_models/6dof_residual/best_model.pth",
                        help="Path to trained residual NN model")
    parser.add_argument("--no-trained", action="store_true",
                        help="Use oracle residual instead of trained NN")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("6-DOF Residual 3-Way Comparison".center(60))
    print("=" * 60)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  K:          {args.K}")
    print(f"  DIAL-MPPI:  {args.dial}")
    print(f"  EMA:        {args.ema}")
    print("=" * 60 + "\n")

    # 모델 생성
    kin_model = MobileManipulator6DOFKinematic()
    dyn_model = MobileManipulator6DOFDynamic()

    # Residual 모델 구성
    if args.no_trained:
        print("  [Using oracle residual (ground-truth)]")
        oracle_residual = create_oracle_residual(kin_model, dyn_model)
        res_model = ResidualDynamics(
            base_model=kin_model,
            residual_fn=oracle_residual,
        )
    else:
        if os.path.exists(args.model_path):
            print(f"  [Loading trained NN from {args.model_path}]")
            neural_model = NeuralDynamics(
                state_dim=9, control_dim=8,
                model_path=args.model_path,
            )
            res_model = ResidualDynamics(
                base_model=kin_model,
                learned_model=neural_model,
            )
        else:
            print(f"  [WARNING] Model not found: {args.model_path}")
            print(f"  → Falling back to oracle residual")
            print(f"  → Train first: PYTHONPATH=. python scripts/train_6dof_residual.py")
            oracle_residual = create_oracle_residual(kin_model, dyn_model)
            res_model = ResidualDynamics(
                base_model=kin_model,
                residual_fn=oracle_residual,
            )

    # 파라미터
    params = create_params(args.K, args.dial)

    # 컨트롤러 생성
    ControllerClass = DIALMPPIController if args.dial else MPPIController

    # FK 모델은 kinematic 사용 (cost function에서)
    # 각 컨트롤러에 맞는 비용 함수 생성
    controllers = {}
    models = {
        "Kinematic": kin_model,
        "Residual": res_model,
        "Oracle": dyn_model,
    }

    for name, model in models.items():
        # cost_fn은 EE position FK가 필요 → kinematic의 FK 사용
        cost_fn = create_cost_fn(kin_model)
        controllers[name] = ControllerClass(model, params, cost_function=cost_fn)
        print(f"  Controller [{name}]: {model.__class__.__name__}")

    # 궤적 생성
    traj_fn = create_trajectory_function(args.trajectory)

    # 시뮬레이션 실행
    print("\n  Running 3-way simulation...")
    histories = run_3way_simulation(
        controllers, kin_model, dyn_model, traj_fn, params, args.duration,
        ema_alpha=args.ema,
    )

    # 결과 요약
    print("\n" + "=" * 60)
    print("Results".center(60))
    print("=" * 60)
    for name in controllers:
        h = histories[name]
        rmse = np.sqrt(np.mean(h["ee_error"] ** 2))
        max_err = np.max(h["ee_error"])
        mean_solve = np.mean(h["solve_time"])
        print(f"  [{name:<10}] RMSE={rmse:.4f}m, MaxErr={max_err:.4f}m, "
              f"MeanSolve={mean_solve:.1f}ms")

    rmse_kin = np.sqrt(np.mean(histories["Kinematic"]["ee_error"] ** 2))
    rmse_res = np.sqrt(np.mean(histories["Residual"]["ee_error"] ** 2))
    improvement = (1 - rmse_res / rmse_kin) * 100
    print(f"\n  Residual improvement: {improvement:.1f}%")
    print("=" * 60 + "\n")

    # 플롯
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controller_type = "DIAL-MPPI" if args.dial else "Vanilla MPPI"
    fig = plot_3way_comparison(
        histories, kin_model,
        title=f"6-DOF Residual 3-Way ({controller_type}, K={args.K})",
    )

    os.makedirs("plots", exist_ok=True)
    save_path = "plots/mobile_manipulator_6dof_residual_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
