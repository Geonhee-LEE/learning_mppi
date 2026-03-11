#!/usr/bin/env python3
"""
MPPI Differential Drive 동역학 모델 데모

원형 궤적 추적 데모. Phase 2 (M2) 검증용.

Usage:
    python mppi_differential_drive_dynamic_demo.py --trajectory circle --duration 30
    python mppi_differential_drive_dynamic_demo.py --trajectory figure8 --live
"""

import numpy as np
import argparse
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
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
    parser = argparse.ArgumentParser(
        description="MPPI Differential Drive Dynamic Demo"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Simulation duration (s)"
    )
    parser.add_argument("--live", action="store_true", help="Live animation mode")
    parser.add_argument("--noise", type=float, default=0.0, help="Process noise std")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot display")

    args = parser.parse_args()

    # 랜덤 시드 설정
    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("MPPI Differential Drive Dynamic Demo".center(60))
    print("=" * 60)
    print(f"Trajectory: {args.trajectory}")
    print(f"Duration: {args.duration}s")
    print(f"Live Mode: {args.live}")
    print(f"Process Noise: {args.noise}")
    print("=" * 60 + "\n")

    # 1. 로봇 모델 생성 (Differential Drive Dynamic)
    model = DifferentialDriveDynamic(
        mass=10.0,
        inertia=1.0,
        c_v=0.1,
        c_omega=0.1,
        a_max=2.0,
        alpha_max=2.0,
        v_max=2.0,
        omega_max=2.0,
    )

    # 2. MPPI 파라미터 설정 (동역학 모델용)
    params = MPPIParams(
        N=30,  # 호라이즌
        dt=0.05,  # 50ms
        K=1024,  # 샘플 수
        lambda_=1.0,  # 온도
        sigma=np.array([1.0, 1.0]),  # 제어 노이즈 [a, α] (가속도)
        Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),  # 상태 가중치 [x, y, θ, v, ω]
        R=np.array([0.1, 0.1]),  # 제어 가중치 [a, α]
        Qf=np.array([20.0, 20.0, 2.0, 0.2, 0.2]),  # 터미널 가중치
        device="cpu",
    )

    # 3. MPPI 컨트롤러 생성
    controller = MPPIController(model, params)

    # 4. 시뮬레이터 생성
    process_noise_std = (
        np.array([0.01, 0.01, 0.01, 0.01, 0.01]) if args.noise > 0 else None
    )
    simulator = Simulator(model, controller, params.dt, process_noise_std)

    # 5. 레퍼런스 궤적 함수 생성
    # 주의: 동역학 모델은 상태 차원이 5이므로 레퍼런스도 5차원으로 확장 필요
    trajectory_fn_kinematic = create_trajectory_function(args.trajectory)

    def trajectory_fn(t):
        """기구학 궤적 (3차원)을 동역학 궤적 (5차원)으로 확장"""
        kinematic_state = trajectory_fn_kinematic(t)  # [x, y, θ]
        # 레퍼런스 속도 계산 (원형 궤적 가정)
        if args.trajectory == "circle":
            v_ref = 0.5  # 일정 선속도
            omega_ref = 0.1  # 일정 각속도
        else:
            v_ref = 0.5
            omega_ref = 0.0
        return np.array(
            [
                kinematic_state[0],  # x
                kinematic_state[1],  # y
                kinematic_state[2],  # θ
                v_ref,  # v
                omega_ref,  # ω
            ]
        )

    def reference_trajectory_fn(t):
        """t 시각의 레퍼런스 궤적 생성 (N+1, nx)"""
        return generate_reference_trajectory(trajectory_fn, t, params.N, params.dt)

    # 6. 초기 상태 설정
    initial_state = trajectory_fn(0.0)
    simulator.reset(initial_state)

    print("Starting simulation...")
    print(
        f"Initial state: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}, "
        f"θ={np.rad2deg(initial_state[2]):.1f}°, "
        f"v={initial_state[3]:.2f}, ω={initial_state[4]:.2f}\n"
    )

    # 7. 시뮬레이션 실행
    if args.live:
        # 실시간 애니메이션 모드
        visualizer = SimulationVisualizer()
        visualizer.animate_live(simulator, reference_trajectory_fn, args.duration)
    else:
        # 일반 시뮬레이션
        history = simulator.run(reference_trajectory_fn, args.duration, realtime=False)

        print(f"Simulation completed. Total steps: {len(history['time'])}\n")

        # 8. 메트릭 계산
        metrics = compute_metrics(history)

        # 9. 메트릭 출력
        print_metrics(metrics, title="MPPI Dynamic Model Performance")

        # 10. 목표 달성 검증
        print("\n" + "=" * 60)
        print("Validation (Phase 2 Goals)".center(60))
        print("=" * 60)

        position_rmse = metrics["position_rmse"]
        mean_solve_time = metrics["mean_solve_time"]

        # 목표: Position RMSE < 0.5m (동역학 모델은 더 어려움)
        position_pass = position_rmse < 0.5
        position_status = "✓ PASS" if position_pass else "✗ FAIL"
        print(f"Position RMSE < 0.5m: {position_rmse:.4f}m [{position_status}]")

        # 목표: Mean Solve Time < 100ms (K=1024, N=30)
        time_pass = mean_solve_time < 100.0
        time_status = "✓ PASS" if time_pass else "✗ FAIL"
        print(f"Solve Time < 100ms: {mean_solve_time:.2f}ms [{time_status}]")

        print("=" * 60 + "\n")

        # 11. 시각화
        if args.no_plot:
            import matplotlib
            matplotlib.use("Agg")

        visualizer = SimulationVisualizer()
        fig = visualizer.plot_results(
            history,
            metrics,
            title=f"MPPI Dynamic - {args.trajectory.capitalize()} Trajectory",
        )

        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    main()
