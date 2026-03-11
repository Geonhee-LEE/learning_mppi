#!/usr/bin/env python3
"""
S11: C2U Obstacle Field — Chance-Constrained Unscented MPPI

프로세스 노이즈가 있는 장애물 필드에서 확률적 안전성 비교:
  Vanilla MPPI vs Uncertainty-Aware MPPI vs C2U-MPPI (3-Way)

C2U-MPPI는 UT 공분산 전파 + 기회 제약으로 불확실성 하에서 안전 마진을 동적 확장.

Usage:
    python c2u_obstacle_field.py
    python c2u_obstacle_field.py --live
    python c2u_obstacle_field.py --noise high
    python c2u_obstacle_field.py --no-plot
"""

import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    UncertaintyMPPIParams,
    C2UMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.uncertainty_mppi import (
    UncertaintyMPPIController,
)
from mppi_controller.controllers.mppi.c2u_mppi import C2UMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, print_metrics
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig
from common.obstacle_field import generate_random_field
from common.env_metrics import compute_env_metrics, print_env_comparison
from common.env_visualizer import EnvVisualizer

import matplotlib
import matplotlib.pyplot as plt


# ── Uncertainty model ────────────────────────────────────────────────────────

class ConstantUncertainty:
    """균일 불확실성 모델"""

    def __init__(self, std_val=0.05):
        self.std_val = std_val

    def __call__(self, states, controls):
        if states.ndim == 1:
            states = states[None, :]
        nx = states.shape[-1]
        return np.full((states.shape[0], nx), self.std_val)


# ── Noise presets ────────────────────────────────────────────────────────────

NOISE_PRESETS = {
    "low": np.array([0.02, 0.02, 0.005]),
    "medium": np.array([0.04, 0.04, 0.01]),
    "high": np.array([0.08, 0.08, 0.02]),
}


# ── Simulation Environment ───────────────────────────────────────────────────

class C2UObstacleFieldEnv(SimulationEnvironment):
    """C2U-MPPI 확률적 안전성 비교 시나리오"""

    def __init__(self, noise_level: str = "medium", seed: int = 42):
        self.noise_level = noise_level
        self.process_noise_std = NOISE_PRESETS.get(noise_level, NOISE_PRESETS["medium"])

        config = EnvironmentConfig(
            name=f"S11: C2U Obstacle Field (noise={noise_level})",
            duration=15.0,
            dt=0.05,
            N=20,
            K=512,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            process_noise_std=self.process_noise_std,
            seed=seed,
        )
        super().__init__(config)
        self._obstacles = self._generate_obstacles(seed)
        self._trajectory_fn = create_trajectory_function("circle", radius=4.0)

    def _generate_obstacles(self, seed):
        return generate_random_field(
            n=10,
            x_range=(-5.0, 5.0),
            y_range=(-5.0, 5.0),
            radius_range=(0.3, 0.5),
            exclusion_zones=[
                (4.0, 0.0, 1.5),
                (-4.0, 0.0, 1.5),
                (0.0, 4.0, 1.5),
                (0.0, -4.0, 1.5),
            ],
            seed=seed,
        )

    def get_initial_state(self):
        return self._trajectory_fn(0.0)

    def get_obstacles(self, t=0.0):
        return self._obstacles

    def get_reference_fn(self):
        N = self.config.N
        dt = self.config.dt
        traj_fn = self._trajectory_fn

        def ref_fn(t):
            return generate_reference_trajectory(traj_fn, t, N, dt)
        return ref_fn

    def get_controller_configs(self):
        c = self.config
        common = dict(
            N=c.N, dt=c.dt, K=c.K, lambda_=c.lambda_,
            sigma=c.sigma, Q=c.Q, R=c.R,
        )
        obstacles = self._obstacles
        unc_model = ConstantUncertainty(0.05)

        configs = []

        # 1. Vanilla MPPI + ObstacleCost
        m1 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p1 = MPPIParams(**common)
        cost1 = CompositeMPPICost([
            StateTrackingCost(p1.Q),
            TerminalCost(p1.Qf),
            ControlEffortCost(p1.R),
            ObstacleCost(obstacles, safety_margin=0.1, cost_weight=200.0),
        ])
        c1 = MPPIController(m1, p1, cost_function=cost1)
        configs.append(ControllerConfig("Vanilla MPPI", c1, m1, "#1f77b4"))

        # 2. Uncertainty-Aware MPPI + ObstacleCost
        m2 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p2 = UncertaintyMPPIParams(
            **common, exploration_factor=1.5,
            uncertainty_strategy="previous_trajectory",
        )
        cost2 = CompositeMPPICost([
            StateTrackingCost(p2.Q),
            TerminalCost(p2.Qf),
            ControlEffortCost(p2.R),
            ObstacleCost(obstacles, safety_margin=0.2, cost_weight=200.0),
        ])
        c2 = UncertaintyMPPIController(
            m2, p2, cost_function=cost2, uncertainty_fn=unc_model,
        )
        configs.append(ControllerConfig("Uncertainty MPPI", c2, m2, "#ff7f0e"))

        # 3. C2U-MPPI (ChanceConstraintCost 자동 포함)
        m3 = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
        p3 = C2UMPPIParams(
            **common,
            cc_obstacles=obstacles,
            chance_alpha=0.05,
            chance_cost_weight=500.0,
            process_noise_scale=0.02,
            cc_margin_factor=1.0,
        )
        cost3 = CompositeMPPICost([
            StateTrackingCost(p3.Q),
            TerminalCost(p3.Qf),
            ControlEffortCost(p3.R),
        ])
        c3 = C2UMPPIController(m3, p3, cost_function=cost3)
        configs.append(ControllerConfig("C2U-MPPI", c3, m3, "#2ca02c"))

        return configs

    def draw_environment(self, ax, t=0.0):
        super().draw_environment(ax, t)
        # 궤적 경로 표시
        ts = np.linspace(0, 2 * np.pi / 0.1, 300)
        pts = np.array([self._trajectory_fn(t_) for t_ in ts])
        ax.plot(pts[:, 0], pts[:, 1], "k--", alpha=0.2, linewidth=1)


# ── Run Scenario ─────────────────────────────────────────────────────────────

def run_scenario(noise="medium", live=False, no_plot=False, seed=42):
    env = C2UObstacleFieldEnv(noise_level=noise, seed=seed)
    configs = env.get_controller_configs()
    ref_fn = env.get_reference_fn()
    initial_state = env.get_initial_state()
    duration = env.config.duration
    process_noise_std = env.process_noise_std

    if live:
        simulators = {}
        ref_fns = {}
        colors = {}
        for cc in configs:
            sim = Simulator(cc.model, cc.controller, env.config.dt,
                            process_noise_std=process_noise_std)
            sim.reset(initial_state)
            simulators[cc.name] = sim
            ref_fns[cc.name] = ref_fn
            colors[cc.name] = cc.color

        viz = EnvVisualizer(env)
        viz.run_and_animate(simulators, ref_fns, duration,
                            controller_colors=colors)
    else:
        if no_plot:
            matplotlib.use("Agg")

        histories = {}
        all_metrics = {}
        all_env_metrics = {}
        colors = {}

        obstacles = env.get_obstacles()
        obstacles_fn = lambda t: obstacles

        for cc in configs:
            np.random.seed(seed)
            print(f"  Running {cc.name}...")
            sim = Simulator(cc.model, cc.controller, env.config.dt,
                            process_noise_std=process_noise_std)
            sim.reset(initial_state)
            history = sim.run(ref_fn, duration)

            histories[cc.name] = history
            all_metrics[cc.name] = compute_metrics(history)
            all_env_metrics[cc.name] = compute_env_metrics(
                history, obstacles_fn=obstacles_fn
            )
            colors[cc.name] = cc.color

        # 결과 출력
        for name in histories:
            print_metrics(all_metrics[name], title=name)
        print_env_comparison(all_env_metrics, title=env.name)

        if not no_plot:
            viz = EnvVisualizer(env)
            fig = viz.run_and_plot(
                histories, all_metrics, all_env_metrics,
                controller_colors=colors,
                save_path=f"plots/s11_c2u_obstacle_{noise}.png",
            )
            plt.show()

    return True


def main():
    parser = argparse.ArgumentParser(description="S11: C2U Obstacle Field")
    parser.add_argument("--noise", choices=["low", "medium", "high"],
                        default="medium")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print(f"S11: C2U Obstacle Field (noise={args.noise})".center(70))
    print("Vanilla MPPI vs Uncertainty MPPI vs C2U-MPPI".center(70))
    print("=" * 70 + "\n")

    run_scenario(noise=args.noise, live=args.live,
                 no_plot=args.no_plot, seed=args.seed)


if __name__ == "__main__":
    main()
