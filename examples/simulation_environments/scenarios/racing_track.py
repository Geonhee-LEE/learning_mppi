#!/usr/bin/env python3
"""
Racing Track Environment — 레이싱 트랙 환경.

직선, 타원, L자형 트랙과 마찰 변화 모델 불일치 시뮬레이션.

Usage:
    PYTHONPATH=. python examples/simulation_environments/scenarios/racing_track.py
    PYTHONPATH=. python examples/simulation_environments/scenarios/racing_track.py --track oval
    PYTHONPATH=. python examples/simulation_environments/scenarios/racing_track.py --track l_shape --mismatch
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.obstacle_field import generate_corridor
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class RacingTrackEnv(SimulationEnvironment):
    """
    레이싱 트랙 환경.

    트랙 종류:
    - "straight": 직선 트랙
    - "oval": 타원형 트랙
    - "l_shape": L자형 트랙

    Args:
        track: 트랙 종류
        track_width: 트랙 폭 (m)
        friction_scale: 마찰 스케일 (1.0=정상, <1.0=미끄러운)
        seed: 랜덤 시드
    """

    def __init__(
        self,
        track: str = "oval",
        track_width: float = 1.5,
        friction_scale: float = 1.0,
        seed: int = 42,
    ):
        config = EnvironmentConfig(
            name=f"Racing Track ({track})",
            duration=20.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.05, 0.05]),
            seed=seed,
        )
        super().__init__(config)
        self.track = track
        self.track_width = track_width
        self.friction_scale = friction_scale
        self._track_points = self._generate_track_points()
        self._obstacles = self._generate_track_walls()

    def _generate_track_points(self):
        """트랙 중심선 점 생성."""
        if self.track == "straight":
            return [(x, 0.0) for x in np.linspace(-8, 8, 50)]

        elif self.track == "oval":
            t = np.linspace(0, 2 * np.pi, 100)
            a, b = 6.0, 4.0
            return [(a * np.cos(ti), b * np.sin(ti)) for ti in t]

        elif self.track == "l_shape":
            points = []
            # 아래쪽 직선 (좌→우)
            for x in np.linspace(-5, 5, 25):
                points.append((x, -4.0))
            # 오른쪽 직선 (아래→위)
            for y in np.linspace(-4, 4, 25):
                points.append((5.0, y))
            # 위쪽 직선 (우→좌)
            for x in np.linspace(5, 0, 15):
                points.append((x, 4.0))
            return points

        else:
            # 기본: 타원
            t = np.linspace(0, 2 * np.pi, 100)
            return [(6 * np.cos(ti), 4 * np.sin(ti)) for ti in t]

    def _generate_track_walls(self):
        """트랙 벽 장애물 생성."""
        return generate_corridor(
            self._track_points,
            width=self.track_width,
            thickness=0.15,
            spacing=0.3,
        )

    def get_initial_state(self) -> np.ndarray:
        p0 = self._track_points[0]
        p1 = self._track_points[1]
        theta = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        return np.array([p0[0], p0[1], theta])

    def get_obstacles(self, t: float = 0.0):
        return self._obstacles

    def get_reference_fn(self):
        N = self.config.N
        dt = self.config.dt
        points = self._track_points
        n_pts = len(points)
        is_closed = self.track == "oval"

        # 누적 거리
        dists = [0.0]
        for i in range(1, n_pts):
            d = np.sqrt((points[i][0] - points[i-1][0])**2 +
                        (points[i][1] - points[i-1][1])**2)
            dists.append(dists[-1] + d)
        total_dist = dists[-1]

        speed = 0.8  # m/s 기준 속도

        def reference_fn(t):
            refs = np.zeros((N + 1, 3))
            for i in range(N + 1):
                ti = t + i * dt
                s = (ti * speed) % total_dist if is_closed else min(ti * speed, total_dist * 0.99)

                # 거리 → 점 인덱스 보간
                idx = 0
                for j in range(1, n_pts):
                    if dists[j] >= s:
                        idx = j - 1
                        break
                else:
                    idx = n_pts - 2

                frac = (s - dists[idx]) / max(dists[idx + 1] - dists[idx], 1e-6)
                p0 = points[idx]
                p1 = points[min(idx + 1, n_pts - 1)]
                refs[i, 0] = p0[0] + frac * (p1[0] - p0[0])
                refs[i, 1] = p0[1] + frac * (p1[1] - p0[1])
                refs[i, 2] = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            return refs

        return reference_fn

    def get_controller_configs(self):
        model = DifferentialDriveKinematic(v_max=1.5, omega_max=2.0)
        params = MPPIParams(
            N=self.config.N, K=self.config.K, dt=self.config.dt,
            lambda_=self.config.lambda_, sigma=self.config.sigma,
            Q=self.config.Q, R=self.config.R,
        )
        controller = MPPIController(model, params)
        return [ControllerConfig("MPPI", controller, model, "#1f77b4")]

    def draw_environment(self, ax, t: float = 0.0):
        super().draw_environment(ax, t)
        # 트랙 중심선
        pts = np.array(self._track_points)
        ax.plot(pts[:, 0], pts[:, 1], "g--", alpha=0.3, linewidth=1, label="Track center")


class FrictionMismatchModel(DifferentialDriveKinematic):
    """
    마찰 변화 모델 불일치 래퍼.

    실제 환경에서 속도가 friction_scale만큼 감소하는 효과를 시뮬레이션.
    """

    def __init__(self, friction_scale: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.friction_scale = friction_scale

    def forward_dynamics(self, state, control):
        # 마찰에 의한 속도 감쇠
        scaled_control = control.copy()
        scaled_control[..., 0] = control[..., 0] * self.friction_scale
        return super().forward_dynamics(state, scaled_control)


def main():
    parser = argparse.ArgumentParser(description="Racing Track Environment")
    parser.add_argument("--track", default="oval", choices=["straight", "oval", "l_shape"])
    parser.add_argument("--mismatch", action="store_true", help="Enable friction mismatch")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    friction = 0.7 if args.mismatch else 1.0
    env = RacingTrackEnv(track=args.track, friction_scale=friction)
    configs = env.get_controller_configs()
    ref_fn = env.get_reference_fn()
    x0 = env.get_initial_state()

    simulators = {}
    for cc in configs:
        # 모델 불일치 시: controller는 nominal model, 시뮬레이션은 mismatch model
        if args.mismatch:
            real_model = FrictionMismatchModel(
                friction_scale=friction,
                v_max=cc.model.v_max,
                omega_max=cc.model.omega_max,
            )
            sim = Simulator(real_model, cc.controller, env.config.dt)
        else:
            sim = Simulator(cc.model, cc.controller, env.config.dt)
        sim.reset(x0.copy())
        simulators[cc.name] = sim

    ref_fns = {cc.name: ref_fn for cc in configs}
    colors = {cc.name: cc.color for cc in configs}

    if args.no_plot:
        for name, sim in simulators.items():
            history = sim.run(ref_fn, env.config.duration)
            metrics = compute_metrics(history)
            em = compute_env_metrics(history, env.get_obstacles)
            print(f"\n{name}: RMSE={metrics['position_rmse']:.4f}m, "
                  f"Collisions={em['collision_count']}, Safety={em['safety_rate']:.1%}")
    else:
        viz = EnvVisualizer(env)
        if args.live:
            viz.run_and_animate(simulators, ref_fns, env.config.duration,
                                controller_colors=colors)
        else:
            histories = {}
            metrics_all = {}
            em_all = {}
            for name, sim in simulators.items():
                h = sim.run(ref_fn, env.config.duration)
                histories[name] = h
                metrics_all[name] = compute_metrics(h)
                em_all[name] = compute_env_metrics(h, env.get_obstacles)

            suffix = "_mismatch" if args.mismatch else ""
            save_path = f"plots/racing_{args.track}{suffix}.png"
            viz.run_and_plot(histories, metrics_all, em_all,
                            controller_colors=colors, save_path=save_path)
            print_env_comparison(em_all, "Racing Track Results")


if __name__ == "__main__":
    main()
