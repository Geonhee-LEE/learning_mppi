"""
EnvVisualizer — 시뮬레이션 환경 시각화.

라이브 애니메이션, 배치 플롯, GIF/MP4 내보내기를 지원.
SimulationEnvironment ABC와 함께 사용.
"""

import numpy as np
import os
from typing import Dict, List, Optional, Callable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mppi_controller.simulation.rendering.robot_renderer import RobotRenderer, make_render_config
from mppi_controller.simulation.rendering.safety_overlay import SafetyOverlay
from mppi_controller.simulation.rendering.animation_saver import AnimationSaver


COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


class EnvVisualizer:
    """
    시뮬레이션 환경 통합 시각화.

    Args:
        environment: SimulationEnvironment 인스턴스
        figsize: 기본 Figure 크기
    """

    def __init__(self, environment, figsize=(16, 10),
                 enable_robot_body: bool = True,
                 safety_overlay: Optional[SafetyOverlay] = None):
        self.env = environment
        self.figsize = figsize
        self.enable_robot_body = enable_robot_body
        self.safety_overlay = safety_overlay

    def run_and_animate(
        self,
        simulators: Dict[str, object],
        reference_fns: Dict[str, Callable],
        duration: float,
        interval: int = 20,
        controller_colors: Optional[Dict[str, str]] = None,
    ):
        """
        실시간 애니메이션.

        Args:
            simulators: {name: Simulator}
            reference_fns: {name: t -> (N+1, nx)}
            duration: 시뮬레이션 시간
            interval: 애니메이션 갱신 간격 (ms)
            controller_colors: {name: color}
        """
        dt = self.env.config.dt
        num_steps = int(duration / dt)
        names = list(simulators.keys())
        colors = controller_colors or {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f"{self.env.name}", fontsize=14, fontweight="bold")

        ax_xy = axes[0, 0]
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_title("Trajectories")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.set_aspect("equal")

        ax_err = axes[0, 1]
        ax_err.set_xlabel("Time (s)")
        ax_err.set_ylabel("Position Error (m)")
        ax_err.set_title("Tracking Error")
        ax_err.grid(True, alpha=0.3)

        ax_dist = axes[1, 0]
        ax_dist.set_xlabel("Time (s)")
        ax_dist.set_ylabel("Min Clearance (m)")
        ax_dist.set_title("Obstacle Clearance")
        ax_dist.grid(True, alpha=0.3)
        ax_dist.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

        ax_info = axes[1, 1]
        ax_info.set_xlabel("Time (s)")
        ax_info.set_ylabel("Solve Time (ms)")
        ax_info.set_title("Computation Time")
        ax_info.grid(True, alpha=0.3)

        # 궤적 라인
        lines_xy = {}
        dots_xy = {}
        lines_err = {}
        lines_dist = {}
        lines_solve = {}
        robot_renderers = {}
        for name in names:
            c = colors[name]
            lines_xy[name], = ax_xy.plot([], [], color=c, linewidth=2, label=name)
            dots_xy[name], = ax_xy.plot([], [], "o", color=c, markersize=6)
            lines_err[name], = ax_err.plot([], [], color=c, linewidth=2, label=name)
            lines_dist[name], = ax_dist.plot([], [], color=c, linewidth=2, label=name)
            lines_solve[name], = ax_info.plot([], [], color=c, linewidth=2, label=name)

            # 로봇 body 렌더러 생성
            if self.enable_robot_body:
                sim = simulators[name]
                model = getattr(sim, "model", None)
                rc = None
                if model and hasattr(model, "render_config"):
                    rc = model.render_config()
                if rc is None:
                    rc = {"shape": "circle", "radius": 0.15}
                rc["color"] = c
                robot_renderers[name] = RobotRenderer(rc)

        for ax in [ax_err, ax_dist, ax_info]:
            ax.legend(fontsize=7)
        ax_xy.legend(fontsize=7)

        time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=10, family="monospace")

        # 장애물 패치 관리
        obs_patches = []
        vel_arrows = []

        data = {n: {"xy": [], "times": [], "errors": [], "clearances": [], "solve_times": []}
                for n in names}

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        def init():
            for name in names:
                lines_xy[name].set_data([], [])
                dots_xy[name].set_data([], [])
                lines_err[name].set_data([], [])
                lines_dist[name].set_data([], [])
                lines_solve[name].set_data([], [])
            return []

        def update(frame):
            if frame >= num_steps:
                return []

            t_current = frame * dt

            # 환경 장애물
            current_obstacles = self.env.get_obstacles(t_current)

            # 장애물 패치 업데이트
            for p in obs_patches:
                try:
                    p.remove()
                except (ValueError, AttributeError):
                    pass
            obs_patches.clear()
            for p in vel_arrows:
                try:
                    p.remove()
                except (ValueError, AttributeError):
                    pass
            vel_arrows.clear()

            for obs in current_obstacles:
                ox, oy = obs[0], obs[1]
                r = obs[2] if len(obs) >= 3 else 0.2
                patch = plt.Circle((ox, oy), r, color="red", alpha=0.25)
                ax_xy.add_patch(patch)
                obs_patches.append(patch)
                # 동적 장애물 속도 화살표 (5-tuple)
                if len(obs) >= 5:
                    vx, vy = obs[3], obs[4]
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > 0.01:
                        arrow = ax_xy.annotate(
                            "", xy=(ox + vx * 0.8, oy + vy * 0.8),
                            xytext=(ox, oy),
                            arrowprops=dict(arrowstyle="->", color="darkred",
                                            lw=1.5, alpha=0.7),
                        )
                        vel_arrows.append(arrow)

            states_dict = {}
            controls_dict = {}
            infos_dict = {}

            for name in names:
                sim = simulators[name]
                ref_fn = reference_fns[name]
                ref_traj = ref_fn(sim.t)
                step_info = sim.step(ref_traj)

                state = sim.state
                states_dict[name] = state
                controls_dict[name] = step_info["control"]
                infos_dict[name] = step_info["info"]

                ref_pt = ref_traj[0, :2]
                data[name]["xy"].append(state[:2].copy())
                data[name]["times"].append(sim.t)
                data[name]["errors"].append(float(np.linalg.norm(state[:2] - ref_pt)))
                data[name]["solve_times"].append(step_info["solve_time"] * 1000)

                if current_obstacles:
                    min_d = min(
                        np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2) - r
                        for ox, oy, r in current_obstacles
                    )
                    data[name]["clearances"].append(float(min_d))
                else:
                    data[name]["clearances"].append(float("inf"))

            # 환경 콜백
            self.env.on_step(t_current, states_dict, controls_dict, infos_dict)

            # 플롯 업데이트
            for name in names:
                xy = np.array(data[name]["xy"])
                times = np.array(data[name]["times"])
                lines_xy[name].set_data(xy[:, 0], xy[:, 1])

                # 로봇 body 렌더링 (dot 대신)
                if self.enable_robot_body and name in robot_renderers:
                    robot_renderers[name].render(ax_xy, states_dict[name])
                    dots_xy[name].set_data([], [])  # dot 숨김
                else:
                    dots_xy[name].set_data([xy[-1, 0]], [xy[-1, 1]])

                lines_err[name].set_data(times, data[name]["errors"])
                lines_solve[name].set_data(times, data[name]["solve_times"])
                if data[name]["clearances"] and data[name]["clearances"][-1] < 1e6:
                    finite_clear = [c for c in data[name]["clearances"] if c < 1e6]
                    lines_dist[name].set_data(
                        times[:len(finite_clear)], finite_clear
                    )

            # 안전 오버레이 갱신
            if self.safety_overlay:
                first_name = names[0]
                first_info = infos_dict.get(first_name, {})
                self.safety_overlay.clear()
                obs_3 = [(o[0], o[1], o[2]) for o in current_obstacles if len(o) >= 3]
                self.safety_overlay.draw_all(
                    ax_xy, states_dict[first_name], obs_3, first_info
                )

            for ax in [ax_err, ax_dist, ax_info]:
                ax.relim()
                ax.autoscale_view()

            # XY 자동 범위
            all_pts = []
            for name in names:
                if data[name]["xy"]:
                    all_pts.extend(data[name]["xy"])
            if all_pts:
                all_pts = np.array(all_pts)
                margin = 2.0
                ax_xy.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
                ax_xy.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

            time_text.set_text(f"t = {t_current:.1f}s / {duration:.0f}s")
            return []

        anim = FuncAnimation(
            fig, update, init_func=init,
            frames=num_steps, interval=interval, blit=False, repeat=False,
        )
        plt.show()

    def run_and_plot(
        self,
        histories: Dict[str, dict],
        metrics: Optional[Dict[str, dict]] = None,
        env_metrics: Optional[Dict[str, dict]] = None,
        controller_colors: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None,
    ):
        """
        배치 모드 정적 플롯 생성.

        Args:
            histories: {name: history_dict}
            metrics: {name: compute_metrics() 결과} (Optional)
            env_metrics: {name: compute_env_metrics() 결과} (Optional)
            controller_colors: {name: color}
            save_path: 저장 경로 (None이면 저장 안함)

        Returns:
            matplotlib Figure
        """
        names = list(histories.keys())
        colors = controller_colors or {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}

        n_rows = 3 if env_metrics else 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows))
        fig.suptitle(f"{self.env.name}", fontsize=16, fontweight="bold")

        # [0,0] XY 궤적
        ax = axes[0, 0]
        first_h = histories[names[0]]
        ref = first_h["reference"]
        ax.plot(ref[:, 0], ref[:, 1], "k--", alpha=0.3, linewidth=1.5, label="Reference")

        # 장애물 (t=0)
        self.env.draw_environment(ax, t=0.0)

        for name in names:
            h = histories[name]
            ax.plot(h["state"][:, 0], h["state"][:, 1],
                    color=colors[name], linewidth=2, label=name)

        ax.plot(first_h["state"][0, 0], first_h["state"][0, 1],
                "ko", markersize=8, label="Start")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("XY Trajectories")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # [0,1] 추적 오차
        ax = axes[0, 1]
        for name in names:
            h = histories[name]
            err = np.linalg.norm(h["state"][:, :2] - h["reference"][:, :2], axis=1)
            ax.plot(h["time"], err, color=colors[name], linewidth=2, label=name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Tracking Error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [1,0] 제어 입력
        ax = axes[1, 0]
        for name in names:
            h = histories[name]
            ax.plot(h["time"], h["control"][:, 0],
                    color=colors[name], linewidth=2, label=f"{name} v")
            ax.plot(h["time"], h["control"][:, 1],
                    color=colors[name], linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control")
        ax.set_title("Control Inputs (solid=v, dashed=w)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # [1,1] 계산 시간
        ax = axes[1, 1]
        for name in names:
            h = histories[name]
            solve_ms = h["solve_time"] * 1000
            ax.plot(h["time"], solve_ms, color=colors[name], linewidth=2, label=name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Solve Time (ms)")
        ax.set_title("Computation Time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 환경 메트릭 패널
        if env_metrics and n_rows > 2:
            # [2,0] 장애물 간격
            ax = axes[2, 0]
            for name in names:
                em = env_metrics[name]
                clearances = em.get("clearances", np.array([]))
                if len(clearances) > 0:
                    h = histories[name]
                    finite_mask = clearances < 1e6
                    if np.any(finite_mask):
                        ax.plot(h["time"][finite_mask], clearances[finite_mask],
                                color=colors[name], linewidth=2, label=name)
            ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Min Clearance (m)")
            ax.set_title("Obstacle Clearance")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # [2,1] 요약 테이블
            ax = axes[2, 1]
            ax.axis("off")
            summary_lines = [f"{'Controller':>15s} | {'RMSE':>7s} | {'Collision':>9s} | {'Safety':>7s} | {'Path':>6s}"]
            summary_lines.append("-" * 55)
            for name in names:
                m = metrics.get(name, {}) if metrics else {}
                em = env_metrics.get(name, {})
                rmse = m.get("position_rmse", 0)
                coll = em.get("collision_count", 0)
                safety = em.get("safety_rate", 1.0)
                path_len = em.get("path_length", 0)
                summary_lines.append(
                    f"{name:>15s} | {rmse:>7.4f} | {coll:>9d} | {safety:>6.1%} | {path_len:>6.2f}"
                )
            summary_text = "\n".join(summary_lines)
            ax.text(0.05, 0.5, summary_text, fontsize=10, va="center",
                    family="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        return fig

    def export_gif(
        self,
        simulators: Dict[str, object],
        reference_fns: Dict[str, Callable],
        duration: float,
        filename: str = "simulation.gif",
        fps: int = 20,
        controller_colors: Optional[Dict[str, str]] = None,
    ):
        """
        GIF 애니메이션 내보내기.

        Args:
            simulators: {name: Simulator}
            reference_fns: {name: t -> (N+1, nx)}
            duration: 시뮬레이션 시간
            filename: 출력 파일명
            fps: 프레임 레이트
            controller_colors: {name: color}
        """
        matplotlib.use("Agg")

        dt = self.env.config.dt
        num_steps = int(duration / dt)
        frame_skip = max(1, int(1.0 / (fps * dt)))
        names = list(simulators.keys())
        colors = controller_colors or {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_title(self.env.name, fontsize=14)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        lines = {}
        dots = {}
        for name in names:
            c = colors[name]
            lines[name], = ax.plot([], [], color=c, linewidth=2, label=name)
            dots[name], = ax.plot([], [], "o", color=c, markersize=6)
        ax.legend(fontsize=8)

        obs_patches = []
        data = {n: {"xy": []} for n in names}

        def init():
            return []

        def update(frame_idx):
            # 여러 스텝 진행
            actual_frame = frame_idx * frame_skip
            if actual_frame >= num_steps:
                return []

            t_current = actual_frame * dt
            obstacles = self.env.get_obstacles(t_current)

            for p in obs_patches:
                p.remove()
            obs_patches.clear()
            for ox, oy, r in obstacles:
                patch = plt.Circle((ox, oy), r, color="red", alpha=0.25)
                ax.add_patch(patch)
                obs_patches.append(patch)

            states_dict = {}
            for name in names:
                sim = simulators[name]
                ref_fn = reference_fns[name]
                # 진행 (frame_skip 스텝)
                for _ in range(frame_skip):
                    if sim.t < duration:
                        ref_traj = ref_fn(sim.t)
                        sim.step(ref_traj)

                states_dict[name] = sim.state
                data[name]["xy"].append(sim.state[:2].copy())

            self.env.on_step(t_current, states_dict, {}, {})

            for name in names:
                xy = np.array(data[name]["xy"])
                lines[name].set_data(xy[:, 0], xy[:, 1])
                dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])

            all_pts = []
            for name in names:
                all_pts.extend(data[name]["xy"])
            if all_pts:
                all_pts = np.array(all_pts)
                m = 2.0
                ax.set_xlim(all_pts[:, 0].min() - m, all_pts[:, 0].max() + m)
                ax.set_ylim(all_pts[:, 1].min() - m, all_pts[:, 1].max() + m)

            return []

        n_frames = num_steps // frame_skip
        anim = FuncAnimation(fig, update, init_func=init,
                             frames=n_frames, blit=False, repeat=False)

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        anim.save(filename, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"GIF saved to: {filename}")
