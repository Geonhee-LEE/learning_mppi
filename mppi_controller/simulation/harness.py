"""
통합 시뮬레이션 하네스.

여러 컨트롤러를 동일 조건으로 시뮬레이션하고 비교하는 통합 프레임워크.
기존 Simulator + compute_metrics + compute_env_metrics를 내부적으로 재사용.

Usage:
    harness = SimulationHarness(dt=0.05)
    harness.add_controller("Vanilla", ctrl_v, model_v, "blue")
    harness.add_controller("Flow", ctrl_f, model_f, "red")
    results = harness.run(ref_fn, x0, duration=15.0)
    harness.plot(results, save_path="plots/comparison.png")
    harness.animate(ref_fn, x0, duration=15.0, save_path="plots/comparison.mp4")
"""

import os
import time
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics, compare_metrics
from mppi_controller.simulation.rendering.headless import create_figure, NullFigure
from mppi_controller.simulation.rendering.robot_renderer import RobotRenderer, make_render_config
from mppi_controller.simulation.rendering.animation_saver import AnimationSaver


@dataclass
class ControllerEntry:
    """비교 대상 컨트롤러 엔트리."""
    name: str
    controller: Any
    model: Any
    color: str = "blue"
    linestyle: str = "-"
    process_noise_std: Optional[np.ndarray] = None
    real_model: Optional[Any] = None  # 모델 불일치 시 실제 전파용


class SimulationHarness:
    """
    통합 시뮬레이션 하네스.

    여러 컨트롤러를 동일한 초기 조건과 레퍼런스로 시뮬레이션하고,
    메트릭 비교 + 시각화를 한곳에서 수행한다.

    Args:
        dt: 시뮬레이션 타임스텝
        headless: True면 GUI 없이 실행
        seed: 난수 시드
    """

    def __init__(self, dt: float = 0.05, headless: bool = False, seed: int = 42):
        self.dt = dt
        self.headless = headless
        self.seed = seed
        self._entries: List[ControllerEntry] = []

    def add_controller(
        self,
        name: str,
        controller,
        model,
        color: str = "blue",
        linestyle: str = "-",
        process_noise_std: Optional[np.ndarray] = None,
        real_model=None,
    ):
        """
        비교 대상 컨트롤러 등록.

        Args:
            name: 컨트롤러 이름 (플롯 범례)
            controller: MPPI 컨트롤러 인스턴스
            model: 로봇 모델 (제어용)
            color: 플롯 색상
            linestyle: 선 스타일
            process_noise_std: 프로세스 노이즈 (nx,)
            real_model: 모델 불일치 테스트용 실제 모델
        """
        self._entries.append(ControllerEntry(
            name=name,
            controller=controller,
            model=model,
            color=color,
            linestyle=linestyle,
            process_noise_std=process_noise_std,
            real_model=real_model,
        ))

    def run(
        self,
        reference_fn: Callable[[float], np.ndarray],
        initial_state: np.ndarray,
        duration: float,
        obstacles_fn: Optional[Callable[[float], list]] = None,
        on_step: Optional[Callable] = None,
    ) -> Dict[str, dict]:
        """
        모든 등록된 컨트롤러로 시뮬레이션 실행.

        Args:
            reference_fn: t -> (N+1, nx) 레퍼런스 궤적 함수
            initial_state: (nx,) 초기 상태
            duration: 시뮬레이션 시간 (초)
            obstacles_fn: t -> [(x, y, r), ...] 장애물 함수 (메트릭용)
            on_step: 매 스텝 콜백(t, states_dict, controls_dict, infos_dict)

        Returns:
            {name: {"history": ..., "metrics": ..., "env_metrics": ...}}
        """
        np.random.seed(self.seed)

        results = {}

        for entry in self._entries:
            # 시뮬레이터 생성 (모델 불일치 지원)
            sim_model = entry.real_model or entry.model
            sim = Simulator(
                model=sim_model,
                controller=entry.controller,
                dt=self.dt,
                process_noise_std=entry.process_noise_std,
            )
            sim.reset(initial_state.copy())

            # 시뮬레이션 실행
            history = sim.run(reference_fn, duration)

            # 메트릭 계산
            metrics = compute_metrics(history)

            # 환경 메트릭 계산
            env_metrics = None
            if obstacles_fn is not None:
                from examples.simulation_environments.common.env_metrics import compute_env_metrics
                env_metrics = compute_env_metrics(history, obstacles_fn)

            results[entry.name] = {
                "history": history,
                "metrics": metrics,
                "env_metrics": env_metrics,
                "entry": entry,
            }

        return results

    def run_with_callback(
        self,
        reference_fn: Callable[[float], np.ndarray],
        initial_state: np.ndarray,
        duration: float,
        obstacles_fn: Optional[Callable[[float], list]] = None,
        on_step: Optional[Callable] = None,
    ) -> Dict[str, dict]:
        """
        스텝별 콜백을 지원하는 시뮬레이션 실행.

        on_step(t, states_dict, controls_dict, infos_dict) 호출.
        """
        np.random.seed(self.seed)
        num_steps = int(duration / self.dt)

        # 시뮬레이터들 생성
        sims = {}
        for entry in self._entries:
            sim_model = entry.real_model or entry.model
            sim = Simulator(
                model=sim_model,
                controller=entry.controller,
                dt=self.dt,
                process_noise_std=entry.process_noise_std,
            )
            sim.reset(initial_state.copy())
            sims[entry.name] = sim

        # 스텝별 실행
        for step in range(num_steps):
            t = step * self.dt
            states_dict = {}
            controls_dict = {}
            infos_dict = {}

            for entry in self._entries:
                sim = sims[entry.name]
                ref_traj = reference_fn(sim.t)
                step_info = sim.step(ref_traj)

                states_dict[entry.name] = sim.state
                controls_dict[entry.name] = step_info["control"]
                infos_dict[entry.name] = step_info["info"]

            if on_step is not None:
                on_step(t, states_dict, controls_dict, infos_dict)

        # 결과 수집
        results = {}
        for entry in self._entries:
            sim = sims[entry.name]
            history = sim.get_history()
            metrics = compute_metrics(history)

            env_metrics = None
            if obstacles_fn is not None:
                from examples.simulation_environments.common.env_metrics import compute_env_metrics
                env_metrics = compute_env_metrics(history, obstacles_fn)

            results[entry.name] = {
                "history": history,
                "metrics": metrics,
                "env_metrics": env_metrics,
                "entry": entry,
            }

        return results

    def plot(
        self,
        results: Dict[str, dict],
        save_path: Optional[str] = None,
        title: str = "Controller Comparison",
        obstacles: Optional[list] = None,
    ):
        """
        정적 결과 플롯.

        Args:
            results: run()의 반환값
            save_path: 저장 경로
            title: 플롯 제목
            obstacles: 정적 장애물 [(x,y,r), ...]
        """
        names = list(results.keys())
        has_env_metrics = any(r.get("env_metrics") is not None for r in results.values())
        n_rows = 3 if has_env_metrics else 2

        fig, axes = create_figure(
            headless=self.headless,
            nrows=n_rows, ncols=2,
            figsize=(16, 6 * n_rows),
        )
        if isinstance(fig, NullFigure):
            return fig

        import matplotlib.pyplot as plt

        fig.suptitle(title, fontsize=16, fontweight="bold")

        colors = {r["entry"].name: r["entry"].color for r in results.values()}

        # [0,0] XY 궤적
        ax = axes[0][0] if n_rows > 1 else axes[0]
        first = results[names[0]]
        ref = first["history"]["reference"]
        ax.plot(ref[:, 0], ref[:, 1], "k--", alpha=0.3, linewidth=1.5, label="Reference")

        # 정적 장애물
        if obstacles:
            for ox, oy, r in obstacles:
                ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.25))

        for name in names:
            h = results[name]["history"]
            c = colors[name]
            ls = results[name]["entry"].linestyle
            ax.plot(h["state"][:, 0], h["state"][:, 1],
                    color=c, linewidth=2, linestyle=ls, label=name)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("XY Trajectories")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # [0,1] 추적 오차
        ax = axes[0][1] if n_rows > 1 else axes[1]
        for name in names:
            h = results[name]["history"]
            err = np.linalg.norm(h["state"][:, :2] - h["reference"][:, :2], axis=1)
            ax.plot(h["time"], err, color=colors[name], linewidth=2, label=name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Tracking Error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [1,0] 제어 입력
        ax = axes[1][0]
        for name in names:
            h = results[name]["history"]
            ax.plot(h["time"], h["control"][:, 0],
                    color=colors[name], linewidth=2, label=f"{name} v")
            ax.plot(h["time"], h["control"][:, 1],
                    color=colors[name], linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control")
        ax.set_title("Control Inputs (solid=v, dashed=ω)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # [1,1] 계산 시간
        ax = axes[1][1]
        for name in names:
            h = results[name]["history"]
            ax.plot(h["time"], h["solve_time"] * 1000,
                    color=colors[name], linewidth=2, label=name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Solve Time (ms)")
        ax.set_title("Computation Time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 환경 메트릭 패널
        if has_env_metrics:
            ax = axes[2][0]
            for name in names:
                em = results[name].get("env_metrics")
                if em and "clearances" in em:
                    h = results[name]["history"]
                    c = em["clearances"]
                    mask = c < 1e6
                    if np.any(mask):
                        ax.plot(h["time"][mask], c[mask],
                                color=colors[name], linewidth=2, label=name)
            ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Min Clearance (m)")
            ax.set_title("Obstacle Clearance")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # [2,1] 요약 테이블
            ax = axes[2][1]
            ax.axis("off")
            lines = [f"{'Controller':>15s} | {'RMSE':>7s} | {'Collision':>9s} | {'Safety':>7s} | {'SolveMs':>7s}"]
            lines.append("-" * 55)
            for name in names:
                m = results[name]["metrics"]
                em = results[name].get("env_metrics") or {}
                lines.append(
                    f"{name:>15s} | {m['position_rmse']:>7.4f} | "
                    f"{em.get('collision_count', 0):>9d} | "
                    f"{em.get('safety_rate', 1.0):>6.1%} | "
                    f"{m['mean_solve_time']:>7.2f}"
                )
            ax.text(0.05, 0.5, "\n".join(lines), fontsize=10, va="center",
                    family="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        return fig

    def animate(
        self,
        reference_fn: Callable[[float], np.ndarray],
        initial_state: np.ndarray,
        duration: float,
        obstacles_fn: Optional[Callable[[float], list]] = None,
        save_path: Optional[str] = None,
        fps: int = 20,
        interval: int = 20,
        title: str = "Simulation",
        on_step: Optional[Callable] = None,
    ) -> Optional[dict]:
        """
        실시간 애니메이션 + MP4/GIF 저장.

        Args:
            reference_fn: t -> (N+1, nx) 레퍼런스 궤적
            initial_state: 초기 상태
            duration: 시뮬레이션 시간
            obstacles_fn: 장애물 함수
            save_path: 저장 경로 (.mp4 또는 .gif)
            fps: 프레임 레이트
            interval: 프레임 간격 (ms, 화면 표시용)
            title: 제목
            on_step: 스텝 콜백

        Returns:
            results dict (run과 동일 형태)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_steps = int(duration / self.dt)
        frame_skip = max(1, int(1.0 / (fps * self.dt)))

        # 시뮬레이터 생성
        sims = {}
        renderers = {}
        for entry in self._entries:
            sim_model = entry.real_model or entry.model
            sim = Simulator(
                model=sim_model,
                controller=entry.controller,
                dt=self.dt,
                process_noise_std=entry.process_noise_std,
            )
            sim.reset(initial_state.copy())
            sims[entry.name] = sim

            # 로봇 렌더러
            rc = None
            if hasattr(entry.model, "render_config"):
                rc = entry.model.render_config()
            if rc is None:
                rc = {"shape": "circle", "radius": 0.15}
            rc["color"] = entry.color
            renderers[entry.name] = RobotRenderer(rc)

        # Figure 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")

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

        ax_time = axes[1, 1]
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Solve Time (ms)")
        ax_time.set_title("Computation Time")
        ax_time.grid(True, alpha=0.3)

        # 라인 핸들
        lines_xy = {}
        lines_err = {}
        lines_dist = {}
        lines_time = {}
        for entry in self._entries:
            c = entry.color
            n = entry.name
            lines_xy[n], = ax_xy.plot([], [], color=c, linewidth=2, label=n)
            lines_err[n], = ax_err.plot([], [], color=c, linewidth=2, label=n)
            lines_dist[n], = ax_dist.plot([], [], color=c, linewidth=2, label=n)
            lines_time[n], = ax_time.plot([], [], color=c, linewidth=2, label=n)
        for ax in [ax_xy, ax_err, ax_dist, ax_time]:
            ax.legend(fontsize=7)

        time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=10, family="monospace")

        data = {e.name: {"xy": [], "times": [], "errors": [],
                         "clearances": [], "solve_times": []}
                for e in self._entries}
        obs_patches = []

        # 애니메이션 세이버
        saver = None
        if save_path:
            saver = AnimationSaver(save_path, fps=fps)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 프레임 생성
        n_frames = num_steps // frame_skip + 1
        for frame_idx in range(n_frames):
            # 이전 장애물 패치 제거
            for p in obs_patches:
                try:
                    p.remove()
                except (ValueError, AttributeError):
                    pass
            obs_patches.clear()

            # 이전 로봇 렌더러 클리어
            for renderer in renderers.values():
                renderer.clear()

            # 스텝 진행
            steps_to_run = min(frame_skip, num_steps - frame_idx * frame_skip)
            if steps_to_run <= 0:
                break

            states_dict = {}
            controls_dict = {}
            infos_dict = {}

            for entry in self._entries:
                sim = sims[entry.name]
                for _ in range(steps_to_run):
                    if sim.t < duration:
                        ref_traj = reference_fn(sim.t)
                        step_info = sim.step(ref_traj)

                states_dict[entry.name] = sim.state
                controls_dict[entry.name] = step_info["control"]
                infos_dict[entry.name] = step_info["info"]

                t_now = sim.t
                data[entry.name]["xy"].append(sim.state[:2].copy())
                data[entry.name]["times"].append(t_now)
                ref_pt = reference_fn(t_now)[0, :2]
                data[entry.name]["errors"].append(
                    float(np.linalg.norm(sim.state[:2] - ref_pt)))
                data[entry.name]["solve_times"].append(
                    step_info["solve_time"] * 1000)

                # 장애물 간격
                if obstacles_fn:
                    obs = obstacles_fn(t_now)
                    if obs:
                        min_d = min(
                            np.sqrt((sim.state[0] - o[0])**2 +
                                    (sim.state[1] - o[1])**2) - o[2]
                            for o in obs)
                        data[entry.name]["clearances"].append(float(min_d))
                    else:
                        data[entry.name]["clearances"].append(float("inf"))

            if on_step:
                t_now = frame_idx * frame_skip * self.dt
                on_step(t_now, states_dict, controls_dict, infos_dict)

            # 장애물 그리기
            if obstacles_fn:
                t_now = sims[self._entries[0].name].t
                for ox, oy, r in obstacles_fn(t_now):
                    patch = plt.Circle((ox, oy), r, color="red", alpha=0.25)
                    ax_xy.add_patch(patch)
                    obs_patches.append(patch)

            # 플롯 업데이트
            for entry in self._entries:
                n = entry.name
                xy = np.array(data[n]["xy"])
                times = np.array(data[n]["times"])
                lines_xy[n].set_data(xy[:, 0], xy[:, 1])
                lines_err[n].set_data(times, data[n]["errors"])
                lines_time[n].set_data(times, data[n]["solve_times"])

                if data[n]["clearances"]:
                    finite = [c for c in data[n]["clearances"] if c < 1e6]
                    if finite:
                        lines_dist[n].set_data(times[:len(finite)], finite)

                # 로봇 body 렌더링
                renderers[n].render(ax_xy, sims[n].state)

            # 축 범위 조정
            all_pts = []
            for entry in self._entries:
                all_pts.extend(data[entry.name]["xy"])
            if all_pts:
                all_pts = np.array(all_pts)
                m = 2.0
                ax_xy.set_xlim(all_pts[:, 0].min() - m, all_pts[:, 0].max() + m)
                ax_xy.set_ylim(all_pts[:, 1].min() - m, all_pts[:, 1].max() + m)

            for ax in [ax_err, ax_dist, ax_time]:
                ax.relim()
                ax.autoscale_view()

            t_display = frame_idx * frame_skip * self.dt
            time_text.set_text(f"t = {t_display:.1f}s / {duration:.0f}s")

            if saver:
                saver.capture_frame(fig)

        if saver:
            saver.finalize()

        plt.close(fig)

        # 결과 수집
        results = {}
        for entry in self._entries:
            sim = sims[entry.name]
            history = sim.get_history()
            metrics = compute_metrics(history)

            env_metrics = None
            if obstacles_fn is not None:
                from examples.simulation_environments.common.env_metrics import compute_env_metrics
                env_metrics = compute_env_metrics(history, obstacles_fn)

            results[entry.name] = {
                "history": history,
                "metrics": metrics,
                "env_metrics": env_metrics,
                "entry": entry,
            }

        return results

    def print_comparison(self, results: Dict[str, dict], title: str = "Comparison"):
        """결과 비교 테이블 출력."""
        names = list(results.keys())
        metrics_list = [results[n]["metrics"] for n in names]
        compare_metrics(metrics_list, names, title)

        # 환경 메트릭
        env_results = {n: results[n]["env_metrics"]
                       for n in names if results[n].get("env_metrics")}
        if env_results:
            from examples.simulation_environments.common.env_metrics import print_env_comparison
            print_env_comparison(env_results, f"{title} — Environment")

    @property
    def entries(self) -> List[ControllerEntry]:
        return self._entries

    def clear_controllers(self):
        """등록된 컨트롤러 초기화."""
        self._entries.clear()

    def __repr__(self) -> str:
        names = [e.name for e in self._entries]
        return f"SimulationHarness(dt={self.dt}, controllers={names})"
