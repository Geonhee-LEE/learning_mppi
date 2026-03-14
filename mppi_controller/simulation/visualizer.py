"""
시뮬레이션 시각화 도구

MPPI 시뮬레이션 결과를 시각화하는 도구.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, Optional, Callable

from mppi_controller.simulation.rendering.robot_renderer import RobotRenderer, make_render_config


class SimulationVisualizer:
    """
    시뮬레이션 결과 시각화

    - 6패널 정적 플롯
    - 실시간 애니메이션 (--live 모드)
    - GIF export (mppi_playground 참고)
    """

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize

    def plot_results(self, history: Dict, metrics: Dict, title: str = "MPPI Results"):
        """
        6패널 정적 플롯

        패널:
            1. XY 궤적
            2. 위치 오차
            3. 제어 입력
            4. 각도 오차
            5. 계산 시간
            6. 메트릭 요약 (텍스트)

        Args:
            history: 시뮬레이션 히스토리
            metrics: 메트릭 딕셔너리
            title: 플롯 제목
        """
        states = history["state"]
        controls = history["control"]
        references = history["reference"]
        times = history["time"]
        solve_times = history["solve_time"] * 1000  # ms

        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)

        # 1. XY 궤적
        ax = axes[0, 0]
        ax.plot(states[:, 0], states[:, 1], "b-", label="Actual", linewidth=2)
        ax.plot(
            references[:, 0],
            references[:, 1],
            "r--",
            label="Reference",
            linewidth=2,
            alpha=0.7,
        )
        ax.scatter(states[0, 0], states[0, 1], c="g", s=100, marker="o", label="Start")
        ax.scatter(states[-1, 0], states[-1, 1], c="r", s=100, marker="X", label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("XY Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        # 2. 위치 오차
        ax = axes[0, 1]
        position_errors = np.linalg.norm(states[:, :2] - references[:, :2], axis=1)
        ax.plot(times, position_errors, "b-", linewidth=2)
        ax.axhline(
            y=metrics["position_rmse"],
            color="r",
            linestyle="--",
            label=f'RMSE={metrics["position_rmse"]:.3f}m',
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Position Tracking Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 제어 입력
        ax = axes[0, 2]
        ax.plot(times, controls[:, 0], "b-", label="v (m/s)", linewidth=2)
        ax.plot(times, controls[:, 1], "r-", label="ω (rad/s)", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control Input")
        ax.set_title("Control Inputs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 각도 오차
        ax = axes[1, 0]
        if states.shape[1] >= 3:
            # 각도 차이를 [-π, π]로 정규화
            heading_errors = np.arctan2(
                np.sin(states[:, 2] - references[:, 2]),
                np.cos(states[:, 2] - references[:, 2]),
            )
            ax.plot(times, np.rad2deg(heading_errors), "b-", linewidth=2)
            ax.axhline(
                y=np.rad2deg(metrics["heading_rmse"]),
                color="r",
                linestyle="--",
                label=f'RMSE={np.rad2deg(metrics["heading_rmse"]):.2f}°',
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Heading Error (deg)")
            ax.set_title("Heading Tracking Error")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 5. 계산 시간
        ax = axes[1, 1]
        ax.plot(times, solve_times, "b-", linewidth=2)
        ax.axhline(
            y=metrics["mean_solve_time"],
            color="r",
            linestyle="--",
            label=f'Mean={metrics["mean_solve_time"]:.2f}ms',
        )
        ax.axhline(
            y=metrics["max_solve_time"],
            color="orange",
            linestyle="--",
            label=f'Max={metrics["max_solve_time"]:.2f}ms',
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Solve Time (ms)")
        ax.set_title("Computation Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. 메트릭 요약 (텍스트)
        ax = axes[1, 2]
        ax.axis("off")

        summary_text = f"""
        Metrics Summary
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Position RMSE: {metrics['position_rmse']:.4f} m
        Max Position Error: {metrics['max_position_error']:.4f} m

        Heading RMSE: {np.rad2deg(metrics['heading_rmse']):.2f}°
        Max Heading Error: {np.rad2deg(metrics['max_heading_error']):.2f}°

        Control Rate (avg): {metrics['control_rate']:.4f}
        Control Rate (max): {metrics['max_control_rate']:.4f}

        Mean Solve Time: {metrics['mean_solve_time']:.2f} ms
        Max Solve Time: {metrics['max_solve_time']:.2f} ms
        Std Solve Time: {metrics['std_solve_time']:.2f} ms
        """

        ax.text(
            0.1,
            0.5,
            summary_text,
            fontsize=10,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        return fig

    def animate_live(
        self,
        simulator,
        reference_trajectory_fn: Callable,
        duration: float,
        interval: int = 50,
    ):
        """
        실시간 애니메이션 (--live 모드)

        Args:
            simulator: Simulator 인스턴스
            reference_trajectory_fn: t → (N+1, nx) 레퍼런스 궤적 함수
            duration: 시뮬레이션 시간 (초)
            interval: 프레임 간격 (ms)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # XY 궤적 축
        ax_xy = axes[0]
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_title("XY Trajectory")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.axis("equal")

        # 위치 오차 축
        ax_error = axes[1]
        ax_error.set_xlabel("Time (s)")
        ax_error.set_ylabel("Position Error (m)")
        ax_error.set_title("Position Tracking Error")
        ax_error.grid(True, alpha=0.3)

        # 로봇 body 렌더러
        model = getattr(simulator, "model", None)
        rc = None
        if model and hasattr(model, "render_config"):
            rc = model.render_config()
        robot_renderer = RobotRenderer(rc)

        # 데이터 저장
        xy_actual = []
        xy_ref = []
        times_list = []
        errors_list = []

        # 플롯 핸들
        (line_actual,) = ax_xy.plot([], [], "b-", label="Actual", linewidth=2)
        (line_ref,) = ax_xy.plot(
            [], [], "r--", label="Reference", linewidth=2, alpha=0.7
        )
        (line_error,) = ax_error.plot([], [], "b-", linewidth=2)

        ax_xy.legend()

        num_steps = int(duration / simulator.dt)

        def init():
            line_actual.set_data([], [])
            line_ref.set_data([], [])
            line_error.set_data([], [])
            return line_actual, line_ref, line_error

        def update(frame):
            if frame >= num_steps:
                return line_actual, line_ref, line_error

            # 레퍼런스 궤적 생성
            ref_traj = reference_trajectory_fn(simulator.t)

            # 한 스텝 실행
            step_info = simulator.step(ref_traj)

            # 데이터 수집
            xy_actual.append(simulator.state[:2])
            xy_ref.append(ref_traj[0, :2])
            times_list.append(simulator.t)

            # 위치 오차 계산
            error = np.linalg.norm(simulator.state[:2] - ref_traj[0, :2])
            errors_list.append(error)

            # 플롯 업데이트
            xy_actual_np = np.array(xy_actual)
            xy_ref_np = np.array(xy_ref)
            times_np = np.array(times_list)
            errors_np = np.array(errors_list)

            line_actual.set_data(xy_actual_np[:, 0], xy_actual_np[:, 1])
            line_ref.set_data(xy_ref_np[:, 0], xy_ref_np[:, 1])
            line_error.set_data(times_np, errors_np)

            # 로봇 body 렌더링
            robot_renderer.render(ax_xy, simulator.state)

            # 축 범위 자동 조정
            ax_xy.relim()
            ax_xy.autoscale_view()
            ax_error.relim()
            ax_error.autoscale_view()

            return line_actual, line_ref, line_error

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=num_steps,
            interval=interval,
            blit=True,
            repeat=False,
        )

        plt.tight_layout()
        plt.show()

        return anim

    def export_gif(
        self,
        history: Dict,
        filename: str = "mppi_animation.gif",
        fps: int = 20,
    ):
        """
        GIF/MP4 파일 생성.

        Args:
            history: 시뮬레이션 히스토리
            filename: 출력 파일명 (.gif 또는 .mp4)
            fps: 프레임 속도
        """
        from mppi_controller.simulation.rendering.animation_saver import AnimationSaver
        import matplotlib
        matplotlib.use("Agg")

        states = history["state"]
        references = history["reference"]
        T = len(states)
        frame_skip = max(1, int(1.0 / (fps * 0.05)))  # assume dt=0.05

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        saver = AnimationSaver(filename, fps=fps)

        for i in range(0, T, frame_skip):
            ax.clear()
            ax.plot(references[:i+1, 0], references[:i+1, 1],
                    "r--", alpha=0.5, label="Reference")
            ax.plot(states[:i+1, 0], states[:i+1, 1],
                    "b-", linewidth=2, label="Actual")
            ax.scatter(states[i, 0], states[i, 1], c="blue", s=50, zorder=5)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            saver.capture_frame(fig)

        saver.finalize()
        plt.close(fig)
