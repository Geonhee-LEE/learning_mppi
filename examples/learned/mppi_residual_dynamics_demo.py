#!/usr/bin/env python3
"""
MPPI Residual Dynamics 데모

더미 residual 함수로 학습 모델 효과 검증.

Usage:
    python mppi_residual_dynamics_demo.py --trajectory circle --duration 30
    python mppi_residual_dynamics_demo.py --residual constant --live
"""

import numpy as np
import argparse
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.learned.residual_dynamics import (
    ResidualDynamics,
    create_constant_residual,
    create_state_dependent_residual,
    create_control_dependent_residual,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.visualizer import SimulationVisualizer
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


def main():
    parser = argparse.ArgumentParser(description="MPPI Residual Dynamics Demo")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--residual",
        type=str,
        default="constant",
        choices=["constant", "state", "control", "none"],
        help="Residual function type",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Simulation duration (s)"
    )
    parser.add_argument("--live", action="store_true", help="Live animation mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 랜덤 시드 설정
    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("MPPI Residual Dynamics Demo".center(60))
    print("=" * 60)
    print(f"Trajectory: {args.trajectory}")
    print(f"Residual Type: {args.residual}")
    print(f"Duration: {args.duration}s")
    print(f"Live Mode: {args.live}")
    print("=" * 60 + "\n")

    # 1. 베이스 모델 (기구학)
    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # 2. Residual 함수 생성
    if args.residual == "constant":
        # 상수 residual: 약간의 슬립 효과
        residual_value = np.array([0.02, 0.01, 0.0])  # [x_dot, y_dot, theta_dot]
        residual_fn = create_constant_residual(residual_value)
        print(f"Residual: Constant drift = {residual_value}")

    elif args.residual == "state":
        # 상태 의존 residual: 위치에 비례하는 보정
        correction_gain = np.array(
            [
                [0.001, 0.0, 0.0],  # x_dot 보정
                [0.0, 0.001, 0.0],  # y_dot 보정
                [0.0, 0.0, 0.0],  # theta_dot 보정
            ]
        )
        residual_fn = create_state_dependent_residual(correction_gain)
        print(f"Residual: State-dependent (gain diagonal)")

    elif args.residual == "control":
        # 제어 의존 residual: 액추에이터 bias
        correction_matrix = np.array(
            [
                [0.05, 0.0],  # v → x_dot bias
                [0.0, 0.05],  # ω → y_dot bias
                [0.0, 0.0],  # no theta bias
            ]
        )
        residual_fn = create_control_dependent_residual(correction_matrix)
        print(f"Residual: Control-dependent (actuator bias)")

    else:  # none
        residual_fn = None
        print("Residual: None (pure physics)")

    # 3. Residual Dynamics 모델 생성
    model = ResidualDynamics(
        base_model=base_model, residual_fn=residual_fn, use_residual=(residual_fn is not None)
    )

    # 4. MPPI 파라미터 설정
    params = MPPIParams(
        N=30,
        dt=0.05,
        K=1024,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
        device="cpu",
    )

    # 5. MPPI 컨트롤러 생성
    controller = MPPIController(model, params)

    # 6. 시뮬레이터 생성
    simulator = Simulator(model, controller, params.dt)

    # 7. 레퍼런스 궤적 함수 생성
    trajectory_fn = create_trajectory_function(args.trajectory)

    def reference_trajectory_fn(t):
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    # 8. 초기 상태 설정
    initial_state = trajectory_fn(0.0)
    simulator.reset(initial_state)

    print("Starting simulation...")
    print(
        f"Initial state: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}, "
        f"θ={np.rad2deg(initial_state[2]):.1f}°\n"
    )

    # 9. 시뮬레이션 실행
    if args.live:
        # 실시간 애니메이션 모드
        visualizer = SimulationVisualizer()
        visualizer.animate_live(simulator, reference_trajectory_fn, args.duration)
    else:
        # 일반 시뮬레이션
        history = simulator.run(reference_trajectory_fn, args.duration, realtime=False)

        print(f"Simulation completed. Total steps: {len(history['time'])}\n")

        # 10. 메트릭 계산
        metrics = compute_metrics(history)

        # 11. 메트릭 출력
        print_metrics(metrics, title=f"MPPI Residual ({args.residual}) Performance")

        # 12. Residual 통계 출력
        if residual_fn is not None:
            stats = model.get_stats()
            print("\n" + "=" * 60)
            print("Residual Statistics".center(60))
            print("=" * 60)
            if stats["residual_mean"] is not None:
                print(f"Mean Residual: {stats['residual_mean']}")
                print(f"Std Residual:  {stats['residual_std']}")
                print(f"Num Calls:     {stats['num_calls']}")
            print("=" * 60 + "\n")

        # 13. Residual 기여도 분석 (샘플)
        if residual_fn is not None:
            sample_state = history["state"][len(history["state"]) // 2]
            sample_control = history["control"][len(history["control"]) // 2]
            contribution = model.get_residual_contribution(sample_state, sample_control)

            print("=" * 60)
            print("Residual Contribution (Mid-trajectory Sample)".center(60))
            print("=" * 60)
            print(f"Physics dot:   {contribution['physics_dot']}")
            print(f"Residual dot:  {contribution['residual_dot']}")
            print(f"Total dot:     {contribution['total_dot']}")
            print(f"Residual %:    {contribution['residual_ratio'] * 100}")
            print("=" * 60 + "\n")

        # 14. 시각화
        visualizer = SimulationVisualizer()
        fig = visualizer.plot_results(
            history,
            metrics,
            title=f"MPPI Residual ({args.residual}) - {args.trajectory.capitalize()}",
        )

        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
