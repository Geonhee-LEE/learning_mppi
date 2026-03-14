#!/usr/bin/env python3
"""
Warehouse Environment — 창고 환경 시뮬레이션.

레벨 0-5 점진적 난이도 (정적 장애물 수 증가, 좁은 통로).
동적 bouncing 장애물 포함 (레벨 3 이상).

Usage:
    PYTHONPATH=. python examples/simulation_environments/scenarios/warehouse.py
    PYTHONPATH=. python examples/simulation_environments/scenarios/warehouse.py --level 3
    PYTHONPATH=. python examples/simulation_environments/scenarios/warehouse.py --level 5 --live
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class WarehouseEnv(SimulationEnvironment):
    """
    창고 환경 — 레벨별 난이도.

    Level 0: 빈 창고 (장애물 없음)
    Level 1: 선반 4개 (넓은 통로)
    Level 2: 선반 8개 (좁은 통로)
    Level 3: 선반 8개 + 동적 장애물 1개
    Level 4: 선반 12개 + 동적 장애물 2개
    Level 5: 선반 16개 + 동적 장애물 3개 (좁은 통로)

    Args:
        level: 난이도 레벨 (0-5)
        seed: 랜덤 시드
    """

    def __init__(self, level: int = 1, seed: int = 42):
        level = max(0, min(5, level))
        config = EnvironmentConfig(
            name=f"Warehouse (Level {level})",
            duration=25.0,
            dt=0.05,
            N=30,
            K=1024,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            seed=seed,
        )
        super().__init__(config)
        self.level = level
        self._shelves = self._create_shelves()
        self._dynamic_obstacles = self._create_dynamic_obstacles()
        self._bounds = (-8.0, 8.0, -8.0, 8.0)  # xmin, xmax, ymin, ymax

    def _create_shelves(self):
        """레벨에 따른 선반 (벽) 장애물 생성."""
        shelves = []

        if self.level == 0:
            return shelves

        # 레벨 1-2: 기본 선반 배치
        shelf_configs = [
            # (x1, y1, x2, y2) 벽 좌표
            (-3.0, -2.0, -3.0, 2.0),   # 왼쪽 세로 선반
            (3.0, -2.0, 3.0, 2.0),     # 오른쪽 세로 선반
            (-1.0, -5.0, -1.0, -1.0),  # 하단 왼쪽
            (1.0, 1.0, 1.0, 5.0),      # 상단 오른쪽
        ]

        if self.level >= 2:
            shelf_configs.extend([
                (-5.0, -4.0, -5.0, 0.0),
                (5.0, 0.0, 5.0, 4.0),
                (-2.0, 4.0, 2.0, 4.0),   # 상단 가로 선반
                (-2.0, -4.0, 2.0, -4.0),  # 하단 가로 선반
            ])

        if self.level >= 4:
            shelf_configs.extend([
                (-6.0, 2.0, -4.0, 2.0),
                (4.0, -2.0, 6.0, -2.0),
                (0.0, -3.0, 0.0, -1.0),
                (0.0, 1.0, 0.0, 3.0),
            ])

        if self.level >= 5:
            shelf_configs.extend([
                (-4.0, -6.0, -4.0, -4.0),
                (4.0, 4.0, 4.0, 6.0),
                (-6.0, -2.0, -4.0, -2.0),
                (4.0, 2.0, 6.0, 2.0),
            ])

        for x1, y1, x2, y2 in shelf_configs[:min(len(shelf_configs), 4 * self.level)]:
            self.add_wall(x1, y1, x2, y2, thickness=0.2)

        return shelf_configs

    def _create_dynamic_obstacles(self):
        """레벨 3 이상: bouncing 동적 장애물."""
        obstacles = []
        if self.level < 3:
            return obstacles

        n_dynamic = self.level - 2  # level 3: 1개, 4: 2개, 5: 3개
        configs = [
            {"pos": np.array([0.0, 0.0]), "vel": np.array([0.3, 0.2]), "r": 0.3},
            {"pos": np.array([-2.0, 3.0]), "vel": np.array([-0.2, 0.3]), "r": 0.25},
            {"pos": np.array([2.0, -3.0]), "vel": np.array([0.25, -0.15]), "r": 0.35},
        ]

        for i in range(min(n_dynamic, len(configs))):
            obstacles.append(configs[i].copy())
            obstacles[-1]["pos"] = configs[i]["pos"].copy()
            obstacles[-1]["vel"] = configs[i]["vel"].copy()

        return obstacles

    def get_initial_state(self) -> np.ndarray:
        return np.array([6.0, 0.0, np.pi])

    def get_obstacles(self, t: float = 0.0):
        # 정적 장애물 (벽)
        obstacles = self.get_wall_obstacles()

        # 동적 장애물 (bouncing)
        for dyn in self._dynamic_obstacles:
            pos = dyn["pos"] + dyn["vel"] * t
            # 바운딩 처리
            xmin, xmax, ymin, ymax = self._bounds
            r = dyn["r"]
            # 주기적 바운싱
            period_x = 2 * (xmax - xmin - 2 * r)
            period_y = 2 * (ymax - ymin - 2 * r)
            if period_x > 0:
                px = (pos[0] - xmin - r) % period_x
                if px > period_x / 2:
                    px = period_x - px
                pos[0] = xmin + r + px
            if period_y > 0:
                py = (pos[1] - ymin - r) % period_y
                if py > period_y / 2:
                    py = period_y - py
                pos[1] = ymin + r + py
            obstacles.append((pos[0], pos[1], r))

        return obstacles

    def get_obstacles_with_velocity(self, t: float = 0.0):
        result = [(x, y, r, 0.0, 0.0) for x, y, r in self.get_wall_obstacles()]

        for dyn in self._dynamic_obstacles:
            pos = dyn["pos"] + dyn["vel"] * t
            r = dyn["r"]
            xmin, xmax, ymin, ymax = self._bounds
            period_x = 2 * (xmax - xmin - 2 * r)
            period_y = 2 * (ymax - ymin - 2 * r)
            vx, vy = dyn["vel"][0], dyn["vel"][1]
            if period_x > 0:
                px = (pos[0] - xmin - r) % period_x
                if px > period_x / 2:
                    vx = -vx
                    px = period_x - px
                pos[0] = xmin + r + px
            if period_y > 0:
                py = (pos[1] - ymin - r) % period_y
                if py > period_y / 2:
                    vy = -vy
                    py = period_y - py
                pos[1] = ymin + r + py
            result.append((pos[0], pos[1], r, vx, vy))

        return result

    def get_reference_fn(self):
        N = self.config.N
        dt = self.config.dt
        # 창고 내 사각형 순회 궤적
        waypoints = [
            (6.0, 0.0), (-6.0, 0.0), (-6.0, 6.0),
            (0.0, 6.0), (0.0, -6.0), (6.0, -6.0), (6.0, 0.0),
        ]

        total_time = self.config.duration

        def reference_fn(t):
            # 웨이포인트 간 보간
            segment_time = total_time / (len(waypoints) - 1)
            refs = np.zeros((N + 1, 3))
            for i in range(N + 1):
                ti = t + i * dt
                seg_idx = min(int(ti / segment_time), len(waypoints) - 2)
                frac = (ti - seg_idx * segment_time) / segment_time
                frac = min(max(frac, 0.0), 1.0)
                wp0 = waypoints[seg_idx]
                wp1 = waypoints[min(seg_idx + 1, len(waypoints) - 1)]
                refs[i, 0] = wp0[0] + frac * (wp1[0] - wp0[0])
                refs[i, 1] = wp0[1] + frac * (wp1[1] - wp0[1])
                dx = wp1[0] - wp0[0]
                dy = wp1[1] - wp0[1]
                refs[i, 2] = np.arctan2(dy, dx)
            return refs

        return reference_fn

    def get_controller_configs(self):
        model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        params = MPPIParams(
            N=self.config.N, K=self.config.K, dt=self.config.dt,
            lambda_=self.config.lambda_, sigma=self.config.sigma,
            Q=self.config.Q, R=self.config.R,
        )
        controller = MPPIController(model, params)
        return [ControllerConfig("MPPI", controller, model, "#1f77b4")]

    def draw_environment(self, ax, t: float = 0.0):
        super().draw_environment(ax, t)
        # 창고 경계
        xmin, xmax, ymin, ymax = self._bounds
        ax.plot([xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                "k-", linewidth=2, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description="Warehouse Environment")
    parser.add_argument("--level", type=int, default=2, choices=range(6))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    env = WarehouseEnv(level=args.level)
    configs = env.get_controller_configs()
    ref_fn = env.get_reference_fn()
    x0 = env.get_initial_state()

    simulators = {}
    for cc in configs:
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
                  f"Collisions={em['collision_count']}, "
                  f"Safety={em['safety_rate']:.1%}")
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

            save_path = f"plots/warehouse_level{args.level}.png"
            viz.run_and_plot(histories, metrics_all, em_all,
                            controller_colors=colors, save_path=save_path)


if __name__ == "__main__":
    main()
