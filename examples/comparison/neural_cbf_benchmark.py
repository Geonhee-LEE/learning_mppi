#!/usr/bin/env python3
"""
Neural CBF vs Analytical CBF 벤치마크

3-Way 비교:
  1. Analytical CBF (ControlBarrierCost)
  2. Neural CBF (NeuralBarrierCost)
  3. Neural CBF + Safety Filter (NeuralCBFSafetyFilter)

2 시나리오:
  A. 원형 장애물 (동등 성능 확인)
  B. 비볼록 L자형 (Neural CBF 우위)

Usage:
    PYTHONPATH=. python examples/comparison/neural_cbf_benchmark.py --scenario circular
    PYTHONPATH=. python examples/comparison/neural_cbf_benchmark.py --scenario non_convex
    PYTHONPATH=. python examples/comparison/neural_cbf_benchmark.py --all-scenarios
"""

import argparse
import numpy as np
import torch
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.cost_functions import (
    StateTrackingCost,
    CompositeMPPICost,
)
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost
from mppi_controller.controllers.mppi.cbf_safety_filter import CBFSafetyFilter
from mppi_controller.learning.neural_cbf_trainer import (
    NeuralCBFTrainer,
    NeuralCBFTrainerConfig,
)
from mppi_controller.controllers.mppi.neural_cbf_cost import NeuralBarrierCost
from mppi_controller.controllers.mppi.neural_cbf_filter import NeuralCBFSafetyFilter


def l_shaped_region(state):
    """L자형 비볼록 장애물"""
    x, y = state[0], state[1]
    in_box1 = (1.5 <= x <= 2.5) and (0.5 <= y <= 2.0)
    in_box2 = (2.5 <= x <= 3.5) and (0.5 <= y <= 1.2)
    return in_box1 or in_box2


def create_reference_trajectory(start, goal, N, dt):
    """직선 레퍼런스 궤적 생성"""
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = np.linspace(start[0], goal[0], N + 1)
    ref[:, 1] = np.linspace(start[1], goal[1], N + 1)
    ref[:, 2] = np.arctan2(goal[1] - start[1], goal[0] - start[0])
    return ref


def simulate(controller, robot, state, reference, dt, steps, safety_filter=None):
    """시뮬레이션 실행"""
    N = controller.params.N
    trajectory = [state.copy()]
    controls = []
    times = []

    for i in range(steps):
        # Reference window: exactly (N+1, nx)
        start_idx = min(i, len(reference) - N - 1)
        ref_window = reference[start_idx: start_idx + N + 1]

        t0 = time.time()
        control, info = controller.compute_control(state, ref_window)
        comp_time = time.time() - t0

        if safety_filter is not None:
            control, _ = safety_filter.filter_control(state, control)

        controls.append(control)
        times.append(comp_time)

        state = robot.step(state, control, dt)
        trajectory.append(state.copy())

    return np.array(trajectory), np.array(controls), np.array(times)


def check_collision_circular(trajectory, obstacles):
    """원형 장애물 충돌 검사"""
    collisions = 0
    for state in trajectory:
        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox)**2 + (state[1] - oy)**2)
            if dist < r:
                collisions += 1
                break
    return collisions


def check_collision_nonconvex(trajectory, region_fn):
    """비볼록 장애물 충돌 검사"""
    collisions = 0
    for state in trajectory:
        if region_fn(state):
            collisions += 1
    return collisions


def run_circular_scenario():
    """시나리오 A: 원형 장애물"""
    print("\n" + "=" * 70)
    print("SCENARIO A: Circular Obstacles — Analytical vs Neural CBF")
    print("=" * 70)

    obstacles = [(2.0, 0.3, 0.5), (3.5, -0.2, 0.4)]
    robot = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0, wheelbase=0.5)

    dt = 0.1
    N = 20
    steps = 60

    params = MPPIParams(
        K=128,
        N=N,
        dt=dt,
        lambda_=1.0,
    )

    tracking_cost = StateTrackingCost(
        Q=np.diag([10.0, 10.0, 1.0]),
    )

    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([5.0, 0.0, 0.0])
    reference = create_reference_trajectory(start, goal, steps + N, dt)

    results = {}

    # --- 1. Analytical CBF ---
    print("\n[1/3] Analytical CBF ...")
    analytical_cost = ControlBarrierCost(
        obstacles=obstacles, cbf_alpha=0.1, cbf_weight=1000.0, safety_margin=0.1
    )
    composite_a = CompositeMPPICost([tracking_cost, analytical_cost])
    ctrl_a = MPPIController(model=robot, cost_function=composite_a, params=params)

    np.random.seed(42)
    traj_a, ctrl_a_out, times_a = simulate(ctrl_a, robot, start.copy(), reference, dt, steps)
    col_a = check_collision_circular(traj_a, obstacles)

    results["Analytical CBF"] = {
        "trajectory": traj_a,
        "collisions": col_a,
        "mean_time": np.mean(times_a) * 1000,
        "goal_dist": np.linalg.norm(traj_a[-1, :2] - goal[:2]),
    }

    # --- 2. Neural CBF ---
    print("[2/3] Neural CBF (training + simulation) ...")
    config = NeuralCBFTrainerConfig(
        hidden_dims=[128, 128, 64],
        epochs=200,
        num_safe_samples=5000,
        num_unsafe_samples=5000,
        num_boundary_samples=2000,
        workspace_bounds=(-1.0, 6.0, -2.0, 2.0),
        early_stopping_patience=50,
    )
    trainer = NeuralCBFTrainer(config)
    data = trainer.generate_training_data(obstacles, safety_margin=0.1)
    history = trainer.train(data, verbose=False)

    neural_cost = NeuralBarrierCost(trainer, cbf_alpha=0.1, cbf_weight=1000.0)
    composite_n = CompositeMPPICost([tracking_cost, neural_cost])
    ctrl_n = MPPIController(model=robot, cost_function=composite_n, params=params)

    np.random.seed(42)
    traj_n, ctrl_n_out, times_n = simulate(ctrl_n, robot, start.copy(), reference, dt, steps)
    col_n = check_collision_circular(traj_n, obstacles)

    results["Neural CBF"] = {
        "trajectory": traj_n,
        "collisions": col_n,
        "mean_time": np.mean(times_n) * 1000,
        "goal_dist": np.linalg.norm(traj_n[-1, :2] - goal[:2]),
        "train_acc": f"{history['safe_acc'][-1]:.1%}/{history['unsafe_acc'][-1]:.1%}",
    }

    # --- 3. Neural CBF + Filter ---
    print("[3/3] Neural CBF + Safety Filter ...")
    neural_filter = NeuralCBFSafetyFilter(trainer, cbf_alpha=0.3)
    composite_nf = CompositeMPPICost([tracking_cost, neural_cost])
    ctrl_nf = MPPIController(model=robot, cost_function=composite_nf, params=params)

    np.random.seed(42)
    traj_nf, ctrl_nf_out, times_nf = simulate(
        ctrl_nf, robot, start.copy(), reference, dt, steps, safety_filter=neural_filter
    )
    col_nf = check_collision_circular(traj_nf, obstacles)

    stats = neural_filter.get_filter_statistics()
    results["Neural CBF+Filter"] = {
        "trajectory": traj_nf,
        "collisions": col_nf,
        "mean_time": np.mean(times_nf) * 1000,
        "goal_dist": np.linalg.norm(traj_nf[-1, :2] - goal[:2]),
        "filter_rate": f"{stats['filter_rate']:.1%}",
    }

    # --- Results ---
    print("\n" + "-" * 70)
    print(f"{'Method':<22} {'Collisions':>10} {'Goal Dist':>10} {'Time(ms)':>10} {'Extra':>20}")
    print("-" * 70)
    for name, r in results.items():
        extra = r.get("train_acc", r.get("filter_rate", "-"))
        print(f"{name:<22} {r['collisions']:>10} {r['goal_dist']:>10.3f} "
              f"{r['mean_time']:>10.1f} {extra:>20}")
    print("-" * 70)


def run_non_convex_scenario():
    """시나리오 B: 비볼록 L자형 장애물 (Neural CBF only)"""
    print("\n" + "=" * 70)
    print("SCENARIO B: Non-Convex L-Shaped Obstacle — Neural CBF")
    print("=" * 70)

    robot = DifferentialDriveKinematic(v_max=1.0, omega_max=2.0, wheelbase=0.5)
    dt = 0.1
    N = 20
    steps = 80

    params = MPPIParams(
        K=256,
        N=N,
        dt=dt,
        lambda_=1.0,
    )

    tracking_cost = StateTrackingCost(
        Q=np.diag([10.0, 10.0, 1.0]),
    )

    start = np.array([0.0, 1.0, 0.0])
    goal = np.array([5.0, 1.0, 0.0])
    reference = create_reference_trajectory(start, goal, steps + N, dt)

    # Train Neural CBF for L-shaped obstacle
    print("\n[1/2] Training Neural CBF for L-shaped obstacle ...")
    config = NeuralCBFTrainerConfig(
        hidden_dims=[128, 128, 64],
        epochs=300,
        num_safe_samples=5000,
        num_unsafe_samples=5000,
        num_boundary_samples=2000,
        workspace_bounds=(-1.0, 6.0, -1.0, 4.0),
        early_stopping_patience=80,
        batch_size=256,
    )
    trainer = NeuralCBFTrainer(config)
    data = trainer.generate_training_data(
        obstacles=[],
        safety_margin=0.05,
        non_convex_regions=[l_shaped_region],
    )
    history = trainer.train(data, verbose=False)

    print(f"  Training complete: safe_acc={history['safe_acc'][-1]:.1%}, "
          f"unsafe_acc={history['unsafe_acc'][-1]:.1%}")

    # --- Neural CBF MPPI ---
    print("[2/2] Neural CBF MPPI simulation ...")
    neural_cost = NeuralBarrierCost(trainer, cbf_alpha=0.1, cbf_weight=1000.0)
    composite = CompositeMPPICost([tracking_cost, neural_cost])
    controller = MPPIController(model=robot, cost_function=composite, params=params)

    np.random.seed(42)
    traj, controls, times = simulate(controller, robot, start.copy(), reference, dt, steps)
    collisions = check_collision_nonconvex(traj, l_shaped_region)

    # Barrier landscape
    x_grid = np.linspace(-0.5, 5.5, 50)
    y_grid = np.linspace(-0.5, 3.5, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    states = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    h_vals = trainer.predict_h(states).reshape(X.shape)

    # Results
    print("\n" + "-" * 50)
    print(f"Collisions: {collisions}/{len(traj)}")
    print(f"Goal distance: {np.linalg.norm(traj[-1, :2] - goal[:2]):.3f}m")
    print(f"Mean compute time: {np.mean(times) * 1000:.1f}ms")
    print(f"Barrier landscape: h_min={h_vals.min():.3f}, h_max={h_vals.max():.3f}")
    print("-" * 50)

    # Simple ASCII trajectory visualization
    print("\nTrajectory (top-down view):")
    w, h_size = 60, 20
    canvas = [['·'] * w for _ in range(h_size)]

    # Draw L-shape
    for ix in range(w):
        for iy in range(h_size):
            x = -0.5 + (ix / w) * 6.0
            y = -0.5 + (iy / h_size) * 4.0
            s = np.array([x, y, 0.0])
            if l_shaped_region(s):
                canvas[h_size - 1 - iy][ix] = '█'

    # Draw trajectory
    for state in traj[::3]:
        ix = int((state[0] + 0.5) / 6.0 * w)
        iy = int((state[1] + 0.5) / 4.0 * h_size)
        if 0 <= ix < w and 0 <= iy < h_size:
            canvas[h_size - 1 - iy][ix] = '●'

    # Start and goal
    ix_s = int((start[0] + 0.5) / 6.0 * w)
    iy_s = int((start[1] + 0.5) / 4.0 * h_size)
    ix_g = int((goal[0] + 0.5) / 6.0 * w)
    iy_g = int((goal[1] + 0.5) / 4.0 * h_size)
    if 0 <= ix_s < w and 0 <= iy_s < h_size:
        canvas[h_size - 1 - iy_s][ix_s] = 'S'
    if 0 <= ix_g < w and 0 <= iy_g < h_size:
        canvas[h_size - 1 - iy_g][ix_g] = 'G'

    for row in canvas:
        print(''.join(row))


def main():
    parser = argparse.ArgumentParser(description="Neural CBF Benchmark")
    parser.add_argument(
        "--scenario", choices=["circular", "non_convex"], default="circular",
        help="Scenario to run"
    )
    parser.add_argument(
        "--all-scenarios", action="store_true", help="Run all scenarios"
    )
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("  Neural CBF Benchmark: Analytical vs Neural CBF")
    print("=" * 70)

    if args.all_scenarios:
        run_circular_scenario()
        run_non_convex_scenario()
    elif args.scenario == "circular":
        run_circular_scenario()
    elif args.scenario == "non_convex":
        run_non_convex_scenario()

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
