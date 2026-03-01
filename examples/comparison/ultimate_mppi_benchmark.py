#!/usr/bin/env python3
"""
Ultimate MPPI Benchmark

8개 MPPI 컨트롤러 x 3 적응 기법 x 3 시나리오 종합 비교.

컨트롤러:
  Vanilla, Shield, AdaptiveShield, SVG, ShieldSVG, AdaptiveShieldSVG, Spline, Smooth

적응:
  None (nominal mismatch), EKF, L1

시나리오:
  1. Circle tracking — 장애물 없음, 모델 정확 → 순수 추적
  2. Obstacle slalom — 6개 장애물, 모델 정확 → 안전+추적
  3. Mismatch + obstacles — 6개 장애물 + c_v mismatch → 궁극 테스트

Usage:
    python ultimate_mppi_benchmark.py                         # all
    python ultimate_mppi_benchmark.py --no-plot               # table only
    python ultimate_mppi_benchmark.py --scenario mismatch_obstacle
    python ultimate_mppi_benchmark.py --variant adaptive_shield_svg --adaptation ekf
    python ultimate_mppi_benchmark.py --live --scenario obstacle_slalom
"""

import matplotlib
if "--live" not in __import__("sys").argv:
    matplotlib.use("Agg")

import numpy as np
import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- Controllers ---
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.adaptive_shield_mppi import AdaptiveShieldMPPIController
from mppi_controller.controllers.mppi.svg_mppi import SVGMPPIController
from mppi_controller.controllers.mppi.shield_svg_mppi import ShieldSVGMPPIController
from mppi_controller.controllers.mppi.adaptive_shield_svg_mppi import AdaptiveShieldSVGMPPIController
from mppi_controller.controllers.mppi.spline_mppi import SplineMPPIController
from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController

# --- Params ---
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams, ShieldMPPIParams, SVGMPPIParams, SplineMPPIParams, SmoothMPPIParams,
)
from mppi_controller.controllers.mppi.adaptive_shield_mppi import AdaptiveShieldParams
from mppi_controller.controllers.mppi.shield_svg_mppi import ShieldSVGMPPIParams
from mppi_controller.controllers.mppi.adaptive_shield_svg_mppi import AdaptiveShieldSVGMPPIParams

# --- Cost ---
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost, ControlEffortCost, ObstacleCost,
    AngleAwareTrackingCost, AngleAwareTerminalCost,
)
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost

# --- Models ---
from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter
from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

# --- Trajectory ---
from mppi_controller.utils.trajectory import (
    circle_trajectory, straight_line_trajectory, generate_reference_trajectory,
)

import matplotlib.pyplot as plt


# ============================================================================
#  Constants
# ============================================================================

DT = 0.05
TOTAL_TIME = 12.0
NUM_STEPS = int(TOTAL_TIME / DT)
MPPI_K = 256
MPPI_N = 20

REAL_C_V = 0.5
REAL_C_OMEGA = 0.3
NOM_C_V = 0.1
NOM_C_OMEGA = 0.1
NOISE_STD = np.array([0.005, 0.005, 0.002, 0.01, 0.005])

# Controller variants
VARIANTS = [
    "vanilla", "shield", "adaptive_shield",
    "svg", "shield_svg", "adaptive_shield_svg",
    "spline", "smooth",
]

ADAPTATIONS = ["none", "ekf", "l1"]

SCENARIOS = ["circle_tracking", "obstacle_slalom", "mismatch_obstacle"]

VARIANT_LABELS = {
    "vanilla": "Vanilla",
    "shield": "Shield",
    "adaptive_shield": "AdaptiveShield",
    "svg": "SVG",
    "shield_svg": "ShieldSVG",
    "adaptive_shield_svg": "AdaptShieldSVG",
    "spline": "Spline",
    "smooth": "Smooth",
}

ADAPTATION_LABELS = {
    "none": "None",
    "ekf": "EKF",
    "l1": "L1",
}

VARIANT_COLORS = {
    "vanilla": "#95a5a6",
    "shield": "#2ecc71",
    "adaptive_shield": "#27ae60",
    "svg": "#3498db",
    "shield_svg": "#2980b9",
    "adaptive_shield_svg": "#e74c3c",
    "spline": "#9b59b6",
    "smooth": "#f39c12",
}


# ============================================================================
#  DynamicWorld
# ============================================================================

class DynamicWorld:
    """Real world with friction mismatch + process noise."""

    def __init__(self, c_v=REAL_C_V, c_omega=REAL_C_OMEGA,
                 process_noise_std=NOISE_STD, mismatch=True):
        if not mismatch:
            c_v, c_omega = NOM_C_V, NOM_C_OMEGA
        self._adapter = DynamicKinematicAdapter(
            c_v=c_v, c_omega=c_omega, k_v=5.0, k_omega=5.0,
        )
        self.process_noise_std = process_noise_std
        self.state_5d = np.zeros(5)

    def reset(self, state_3d):
        self.state_5d = np.array([state_3d[0], state_3d[1], state_3d[2], 0.0, 0.0])

    def step(self, control, dt, add_noise=True):
        next_state = self._adapter.step(self.state_5d, control, dt)
        if add_noise:
            next_state += np.random.normal(0.0, self.process_noise_std)
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        self.state_5d = next_state
        return next_state[:3].copy()

    def get_full_state(self):
        return self.state_5d.copy()


# ============================================================================
#  5D Utilities
# ============================================================================

def make_5d_reference(ref_3d):
    N_plus_1 = ref_3d.shape[0]
    ref_5d = np.zeros((N_plus_1, 5))
    ref_5d[:, :3] = ref_3d
    if N_plus_1 > 1:
        dx = np.diff(ref_3d[:, 0])
        dy = np.diff(ref_3d[:, 1])
        v_ref = np.sqrt(dx**2 + dy**2) / DT
        ref_5d[:-1, 3] = v_ref
        ref_5d[-1, 3] = v_ref[-1] if len(v_ref) > 0 else 0.0
        dtheta = np.diff(ref_3d[:, 2])
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        omega_ref = dtheta / DT
        ref_5d[:-1, 4] = omega_ref
        ref_5d[-1, 4] = omega_ref[-1] if len(omega_ref) > 0 else 0.0
    return ref_5d


def create_5d_cost(obstacles, include_obstacle_cost=True):
    Q = np.array([10.0, 10.0, 1.0, 0.1, 0.1])
    Qf = np.array([20.0, 20.0, 2.0, 0.2, 0.2])
    R = np.array([0.1, 0.1])
    costs = [
        AngleAwareTrackingCost(Q, angle_indices=(2,)),
        AngleAwareTerminalCost(Qf, angle_indices=(2,)),
        ControlEffortCost(R),
    ]
    if include_obstacle_cost and obstacles:
        costs.append(ObstacleCost(obstacles=obstacles, cost_weight=500.0))
    return CompositeMPPICost(costs)


# ============================================================================
#  Scenarios
# ============================================================================

def create_scenario(name):
    """Return (obstacles, initial_state_3d, traj_fn, mismatch, description)."""
    if name == "circle_tracking":
        return (
            [],
            np.array([3.0, 0.0, np.pi / 2]),
            lambda t: circle_trajectory(t, radius=3.0, angular_velocity=0.15, center=(0.0, 0.0)),
            False,
            "Circle tracking (no obstacles, no mismatch)",
        )
    elif name == "obstacle_slalom":
        obstacles = [
            (2.0, 0.6, 0.3), (4.0, -0.6, 0.3), (6.0, 0.5, 0.3),
            (8.0, -0.5, 0.3), (10.0, 0.4, 0.3), (12.0, -0.4, 0.3),
        ]
        return (
            obstacles,
            np.array([0.0, 0.0, 0.0]),
            lambda t: straight_line_trajectory(t, velocity=0.6, heading=0.0, start=(0.0, 0.0)),
            False,
            "Obstacle slalom (6 obstacles, no mismatch)",
        )
    elif name == "mismatch_obstacle":
        obstacles = [
            (2.0, 0.6, 0.3), (4.0, -0.6, 0.3), (6.0, 0.5, 0.3),
            (8.0, -0.5, 0.3), (10.0, 0.4, 0.3), (12.0, -0.4, 0.3),
        ]
        return (
            obstacles,
            np.array([0.0, 0.0, 0.0]),
            lambda t: straight_line_trajectory(t, velocity=0.6, heading=0.0, start=(0.0, 0.0)),
            True,
            "Mismatch + obstacles (c_v mismatch + 6 obstacles)",
        )
    else:
        raise ValueError(f"Unknown scenario: {name}")


# ============================================================================
#  Controller Factory
# ============================================================================

def create_controller(variant, model, obstacles):
    """Create controller + params for given variant."""
    base_kwargs = dict(
        K=MPPI_K, N=MPPI_N, dt=DT, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2, 0.2]),
    )
    cbf_kwargs = dict(
        cbf_obstacles=list(obstacles) if obstacles else [],
        cbf_weight=1000.0,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )
    shield_kwargs = dict(shield_enabled=True, shield_cbf_alpha=0.3)
    adaptive_kwargs = dict(alpha_base=0.3, alpha_dist=0.1, alpha_vel=0.5, k_dist=2.0, d_safe=0.5)
    svg_kwargs = dict(
        svgd_num_iterations=2, svgd_step_size=0.01,
        svg_num_guide_particles=16, svg_guide_step_size=0.01,
    )
    cost_fn = create_5d_cost(obstacles)

    if variant == "vanilla":
        params = MPPIParams(**base_kwargs)
        return MPPIController(model, params, cost_fn)

    elif variant == "shield":
        params = ShieldMPPIParams(**base_kwargs, **cbf_kwargs, **shield_kwargs)
        return ShieldMPPIController(model, params, cost_fn)

    elif variant == "adaptive_shield":
        params = AdaptiveShieldParams(**base_kwargs, **cbf_kwargs, **shield_kwargs, **adaptive_kwargs)
        return AdaptiveShieldMPPIController(model, params, cost_fn)

    elif variant == "svg":
        params = SVGMPPIParams(**base_kwargs, **svg_kwargs)
        return SVGMPPIController(model, params)

    elif variant == "shield_svg":
        params = ShieldSVGMPPIParams(**base_kwargs, **svg_kwargs,
                                      **shield_kwargs, cbf_obstacles=list(obstacles) if obstacles else [],
                                      cbf_safety_margin=0.1)
        return ShieldSVGMPPIController(model, params)

    elif variant == "adaptive_shield_svg":
        params = AdaptiveShieldSVGMPPIParams(**base_kwargs, **svg_kwargs,
                                              **shield_kwargs, cbf_obstacles=list(obstacles) if obstacles else [],
                                              cbf_safety_margin=0.1, **adaptive_kwargs)
        return AdaptiveShieldSVGMPPIController(model, params)

    elif variant == "spline":
        params = SplineMPPIParams(**base_kwargs, spline_num_knots=8, spline_degree=3)
        return SplineMPPIController(model, params)

    elif variant == "smooth":
        params = SmoothMPPIParams(**base_kwargs, jerk_weight=1.0)
        return SmoothMPPIController(model, params)

    else:
        raise ValueError(f"Unknown variant: {variant}")


def create_model(adaptation):
    """Create model for given adaptation type."""
    if adaptation == "none":
        return DynamicKinematicAdapter(c_v=NOM_C_V, c_omega=NOM_C_OMEGA, k_v=5.0, k_omega=5.0)
    elif adaptation == "ekf":
        return EKFAdaptiveDynamics(c_v_init=NOM_C_V, c_omega_init=NOM_C_OMEGA, k_v=5.0, k_omega=5.0)
    elif adaptation == "l1":
        return L1AdaptiveDynamics(c_v_nom=NOM_C_V, c_omega_nom=NOM_C_OMEGA, k_v=5.0, k_omega=5.0)
    else:
        raise ValueError(f"Unknown adaptation: {adaptation}")


# ============================================================================
#  Metrics
# ============================================================================

def compute_metrics(states_3d, solve_times, obstacles, ref_states, controls):
    T = len(states_3d)
    if T == 0:
        return {"collision_count": 0, "safety_rate": 1.0, "tracking_rmse": float("inf"),
                "smoothness": 0.0, "compute_time_ms": 0.0}

    states = np.array(states_3d)
    refs = np.array(ref_states)

    pos_errors = np.linalg.norm(states[:, :2] - refs[:, :2], axis=1)
    tracking_rmse = float(np.sqrt(np.mean(pos_errors**2)))

    collision_count = 0
    safe_steps = 0
    robot_radius = 0.25

    for i in range(T):
        step_safe = True
        for ox, oy, r in obstacles:
            d = np.sqrt((states[i, 0] - ox)**2 + (states[i, 1] - oy)**2) - r - robot_radius
            if d < 0:
                step_safe = False
                break
        if step_safe:
            safe_steps += 1
        else:
            collision_count += 1

    # Smoothness: mean |delta u|
    ctrl_arr = np.array(controls)
    if len(ctrl_arr) > 1:
        du = np.diff(ctrl_arr, axis=0)
        smoothness = float(np.mean(np.linalg.norm(du, axis=1)))
    else:
        smoothness = 0.0

    return {
        "collision_count": collision_count,
        "safety_rate": safe_steps / T if T > 0 else 1.0,
        "tracking_rmse": tracking_rmse,
        "smoothness": smoothness,
        "compute_time_ms": float(np.mean(solve_times)) * 1000.0,
    }


# ============================================================================
#  Simulation Runner
# ============================================================================

def run_single(variant, adaptation, scenario_name, seed=42):
    """Run a single (variant, adaptation, scenario) combo."""
    np.random.seed(seed)

    obstacles, initial_state, traj_fn, mismatch, desc = create_scenario(scenario_name)

    model = create_model(adaptation)
    controller = create_controller(variant, model, obstacles)

    world = DynamicWorld(mismatch=mismatch)
    world.reset(initial_state)
    state_5d = np.array([*initial_state, 0.0, 0.0])

    states_3d = []
    solve_times = []
    ref_states = []
    controls = []
    prev_state = state_5d.copy()
    prev_control = np.zeros(2)

    t = 0.0
    for step_i in range(NUM_STEPS):
        ref_3d = generate_reference_trajectory(traj_fn, t, MPPI_N, DT)
        ref_5d = make_5d_reference(ref_3d)

        t_start = time.time()
        control, info = controller.compute_control(state_5d, ref_5d)
        solve_time = time.time() - t_start

        states_3d.append(state_5d[:3].copy())
        ref_states.append(ref_3d[0].copy())
        solve_times.append(solve_time)
        controls.append(control.copy())

        world.step(control, DT, add_noise=True)
        next_5d = world.get_full_state()

        # Adaptation updates
        if adaptation in ("ekf", "l1") and step_i > 0:
            model.update_step(prev_state, prev_control, state_5d, DT)

        prev_state = state_5d.copy()
        prev_control = control.copy()
        state_5d = next_5d
        t += DT

    metrics = compute_metrics(states_3d, solve_times, obstacles, ref_states, controls)
    return metrics, np.array(states_3d)


# ============================================================================
#  Benchmark Runner
# ============================================================================

def make_method_key(variant, adaptation):
    return f"{variant}__{adaptation}"


def run_benchmark(scenarios, variants, adaptations):
    """Run full benchmark grid."""
    all_results = {}  # {scenario: {method_key: metrics}}
    all_trajectories = {}  # {scenario: {method_key: (T,3)}}
    all_obstacles = {}

    for scenario in scenarios:
        obstacles, _, _, _, desc = create_scenario(scenario)
        all_obstacles[scenario] = obstacles
        all_results[scenario] = {}
        all_trajectories[scenario] = {}

        print(f"\n{'=' * 80}")
        print(f"  Scenario: {scenario} — {desc}")
        print(f"{'=' * 80}")

        for variant in variants:
            for adaptation in adaptations:
                key = make_method_key(variant, adaptation)
                label = f"{VARIANT_LABELS[variant]:>16s} + {ADAPTATION_LABELS[adaptation]:<4s}"
                print(f"  [{label}] running...", end="", flush=True)

                try:
                    metrics, traj = run_single(variant, adaptation, scenario)
                    all_results[scenario][key] = metrics
                    all_trajectories[scenario][key] = traj

                    col = metrics["collision_count"]
                    sr = metrics["safety_rate"]
                    rmse = metrics["tracking_rmse"]
                    sm = metrics["smoothness"]
                    ms = metrics["compute_time_ms"]
                    print(f" col={col:3d} safe={sr:.3f} rmse={rmse:.3f}m "
                          f"smooth={sm:.3f} {ms:.1f}ms")
                except Exception as e:
                    print(f" ERROR: {e}")

    return all_results, all_trajectories, all_obstacles


# ============================================================================
#  Console Output
# ============================================================================

def print_results_table(scenario, results, variants, adaptations):
    """Print formatted comparison table for one scenario."""
    if not results:
        return

    print(f"\n{'=' * 100}")
    print(f"  Results: {scenario}")
    print(f"{'=' * 100}")
    header = (f"  {'Variant':<18s} {'Adapt':>5s} {'Col':>5s} {'Safe%':>6s} "
              f"{'RMSE':>7s} {'Smooth':>7s} {'ms':>7s}")
    print(header)
    print(f"  {'-' * 96}")

    for variant in variants:
        for adaptation in adaptations:
            key = make_method_key(variant, adaptation)
            if key not in results:
                continue
            m = results[key]
            vl = VARIANT_LABELS[variant]
            al = ADAPTATION_LABELS[adaptation]
            col = m["collision_count"]
            sr = m["safety_rate"] * 100
            rmse = m["tracking_rmse"]
            sm = m["smoothness"]
            ms = m["compute_time_ms"]
            col_mark = " **" if col > 0 else ""
            print(f"  {vl:<18s} {al:>5s} {col:>5d}{col_mark:<3s}"
                  f"{sr:>5.1f}% {rmse:>7.3f} {sm:>7.3f} {ms:>7.1f}")

    print(f"  {'-' * 96}")

    # Best safe combo
    safe = {k: v for k, v in results.items() if v["collision_count"] == 0}
    if safe:
        best = min(safe.items(), key=lambda x: x[1]["tracking_rmse"])
        v, a = best[0].split("__")
        print(f"\n  Best (collision-free): {VARIANT_LABELS[v]} + {ADAPTATION_LABELS[a]} "
              f"(RMSE={best[1]['tracking_rmse']:.3f}m)")


def print_cross_scenario_summary(all_results, variants, adaptations, scenarios):
    """Print cross-scenario summary table."""
    print(f"\n{'=' * 100}")
    print(f"  Cross-Scenario Summary")
    print(f"{'=' * 100}")
    header = (f"  {'Variant':<18s} {'Adapt':>5s} {'AvgCol':>7s} {'AvgSafe%':>8s} "
              f"{'AvgRMSE':>8s} {'AvgSmooth':>9s} {'AvgMs':>7s}")
    print(header)
    print(f"  {'-' * 96}")

    for variant in variants:
        for adaptation in adaptations:
            key = make_method_key(variant, adaptation)
            mvals = [all_results[s][key] for s in scenarios
                     if key in all_results.get(s, {})]
            if not mvals:
                continue
            vl = VARIANT_LABELS[variant]
            al = ADAPTATION_LABELS[adaptation]
            avg_col = np.mean([v["collision_count"] for v in mvals])
            avg_sr = np.mean([v["safety_rate"] for v in mvals]) * 100
            avg_rmse = np.mean([v["tracking_rmse"] for v in mvals])
            avg_sm = np.mean([v["smoothness"] for v in mvals])
            avg_ms = np.mean([v["compute_time_ms"] for v in mvals])
            col_mark = " **" if avg_col > 0 else ""
            print(f"  {vl:<18s} {al:>5s} {avg_col:>7.1f}{col_mark:<3s}"
                  f"{avg_sr:>7.1f}% {avg_rmse:>8.3f} {avg_sm:>9.3f} {avg_ms:>7.1f}")

    print(f"  {'-' * 96}")

    # Overall best
    all_keys = set()
    for s in scenarios:
        all_keys.update(all_results.get(s, {}).keys())

    best_key = None
    best_score = float("inf")
    for key in all_keys:
        mvals = [all_results[s][key] for s in scenarios if key in all_results.get(s, {})]
        if not mvals:
            continue
        avg_col = np.mean([v["collision_count"] for v in mvals])
        if avg_col > 0:
            continue
        avg_rmse = np.mean([v["tracking_rmse"] for v in mvals])
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_key = key

    if best_key:
        v, a = best_key.split("__")
        print(f"\n  Overall Best (0 collisions): {VARIANT_LABELS[v]} + {ADAPTATION_LABELS[a]} "
              f"(avg RMSE={best_score:.3f}m)")


# ============================================================================
#  Visualization
# ============================================================================

def plot_results(all_results, all_trajectories, all_obstacles, scenarios,
                 variants, adaptations, save_path):
    """Multi-panel comparison figure."""
    n_scenarios = len(scenarios)
    fig = plt.figure(figsize=(7 * n_scenarios, 14))
    fig.suptitle("Ultimate MPPI Benchmark: 8 Variants x 3 Adaptations x 3 Scenarios",
                 fontsize=14, fontweight="bold")

    # Row 1: XY trajectories per scenario
    for si, scenario in enumerate(scenarios):
        ax = fig.add_subplot(3, n_scenarios, si + 1)
        obstacles = all_obstacles.get(scenario, [])
        trajectories = all_trajectories.get(scenario, {})

        for ox, oy, r in obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))
            ax.add_patch(plt.Circle((ox, oy), r + 0.25,
                                     color="red", alpha=0.1, linestyle="--", fill=False))

        # Only plot "none" adaptation for clarity
        for variant in variants:
            key = make_method_key(variant, "none")
            if key not in trajectories:
                continue
            traj = trajectories[key]
            ax.plot(traj[:, 0], traj[:, 1],
                    color=VARIANT_COLORS[variant], linewidth=1.5,
                    label=VARIANT_LABELS[variant], alpha=0.8)

        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if si == 0:
            ax.legend(fontsize=5, loc="upper left", ncol=2)

    # Row 2: RMSE bar chart per scenario
    for si, scenario in enumerate(scenarios):
        ax = fig.add_subplot(3, n_scenarios, n_scenarios + si + 1)
        results = all_results.get(scenario, {})

        keys = []
        labels = []
        rmse_vals = []
        colors = []

        for variant in variants:
            for adaptation in adaptations:
                key = make_method_key(variant, adaptation)
                if key not in results:
                    continue
                keys.append(key)
                labels.append(f"{VARIANT_LABELS[variant][:8]}\n{ADAPTATION_LABELS[adaptation]}")
                rmse_vals.append(results[key]["tracking_rmse"])
                colors.append(VARIANT_COLORS[variant])

        x_pos = np.arange(len(keys))
        bars = ax.bar(x_pos, rmse_vals, color=colors, alpha=0.8,
                       edgecolor="black", linewidth=0.5)
        ax.set_ylabel("RMSE (m)")
        ax.set_title(f"Tracking RMSE — {scenario}", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.grid(axis="y", alpha=0.3)

    # Row 3: Safety rate bar chart per scenario
    for si, scenario in enumerate(scenarios):
        ax = fig.add_subplot(3, n_scenarios, 2 * n_scenarios + si + 1)
        results = all_results.get(scenario, {})

        keys = []
        labels = []
        safety_vals = []
        colors = []

        for variant in variants:
            for adaptation in adaptations:
                key = make_method_key(variant, adaptation)
                if key not in results:
                    continue
                keys.append(key)
                labels.append(f"{VARIANT_LABELS[variant][:8]}\n{ADAPTATION_LABELS[adaptation]}")
                safety_vals.append(results[key]["safety_rate"] * 100)
                colors.append(VARIANT_COLORS[variant])

        x_pos = np.arange(len(keys))
        bars = ax.bar(x_pos, safety_vals, color=colors, alpha=0.8,
                       edgecolor="black", linewidth=0.5)
        ax.axhline(y=100, color="green", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylabel("Safety Rate (%)")
        ax.set_title(f"Safety — {scenario}", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Plot saved to: {save_path}")


# ============================================================================
#  Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Ultimate MPPI Benchmark")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=SCENARIOS)
    parser.add_argument("--variant", type=str, default=None,
                        choices=VARIANTS)
    parser.add_argument("--adaptation", type=str, default=None,
                        choices=ADAPTATIONS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--live", action="store_true", help="Live animation (not yet)")
    return parser.parse_args()


def main():
    args = parse_args()

    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    variants = [args.variant] if args.variant else list(VARIANTS)
    adaptations = [args.adaptation] if args.adaptation else list(ADAPTATIONS)

    total_combos = len(scenarios) * len(variants) * len(adaptations)

    print("=" * 80)
    print("  Ultimate MPPI Benchmark")
    print("=" * 80)
    print(f"  Scenarios:    {scenarios}")
    print(f"  Variants:     {len(variants)} — {variants}")
    print(f"  Adaptations:  {len(adaptations)} — {adaptations}")
    print(f"  Total combos: {total_combos}")
    print(f"  MPPI: K={MPPI_K}, N={MPPI_N}, dt={DT}, T={TOTAL_TIME}s")

    t_start = time.time()
    all_results, all_trajectories, all_obstacles = run_benchmark(
        scenarios, variants, adaptations
    )
    total_time = time.time() - t_start

    for scenario in scenarios:
        if scenario in all_results:
            print_results_table(scenario, all_results[scenario], variants, adaptations)

    if len(scenarios) > 1:
        print_cross_scenario_summary(all_results, variants, adaptations, scenarios)

    if not args.no_plot:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../plots/ultimate_mppi_benchmark.png"
        )
        save_path = os.path.abspath(save_path)
        plot_results(all_results, all_trajectories, all_obstacles,
                     scenarios, variants, adaptations, save_path)

    print(f"\nBenchmark complete in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
