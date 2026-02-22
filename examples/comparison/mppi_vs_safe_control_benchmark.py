#!/usr/bin/env python3
"""
MPPI vs safe_control Benchmark

Compares our MPPI safety methods against the safe_control package (tkkim-robot/safe_control).

Methods compared:
  safe_control:
    1. CBF-QP       — CVXPY quadratic program
    2. MPC-CBF      — CasADi do_mpc

  Our MPPI (6):
    3. Vanilla MPPI        — no safety (baseline)
    4. CBF-MPPI            — CBF cost penalty
    5. Shield-MPPI         — per-step CBF enforcement
    6. Adaptive Shield     — distance-adaptive alpha
    7. CBF-Guided Sampling — gradient-biased resampling
    8. Shield-SVG-MPPI     — Shield + SVG-MPPI

Scenarios:
  - circle_obstacle: Circular trajectory with 4 obstacles on path
  - gauntlet:        Straight-line through corridor of 6 obstacles

Usage:
    python mppi_vs_safe_control_benchmark.py                              # all scenarios
    python mppi_vs_safe_control_benchmark.py --scenario circle_obstacle
    python mppi_vs_safe_control_benchmark.py --live                       # live animation
    python mppi_vs_safe_control_benchmark.py --live --scenario gauntlet
    python mppi_vs_safe_control_benchmark.py --no-plot                    # table only
    python mppi_vs_safe_control_benchmark.py --methods cbf_qp,shield,adaptive_shield
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

# --- Our MPPI imports ---
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    CBFMPPIParams,
    ShieldMPPIParams,
    SVGMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.adaptive_shield_mppi import (
    AdaptiveShieldMPPIController,
    AdaptiveShieldParams,
)
from mppi_controller.controllers.mppi.cbf_guided_sampling_mppi import (
    CBFGuidedSamplingMPPIController,
    CBFGuidedSamplingParams,
)
from mppi_controller.controllers.mppi.shield_svg_mppi import (
    ShieldSVGMPPIController,
    ShieldSVGMPPIParams,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.utils.trajectory import (
    circle_trajectory,
    straight_line_trajectory,
    generate_reference_trajectory,
)

# --- safe_control imports (optional) ---
SAFE_CONTROL_AVAILABLE = False
try:
    import matplotlib.pyplot as plt

    _fig_tmp, _ax_tmp = plt.subplots()
    from safe_control.robots.robot import BaseRobot
    from safe_control.position_control.cbf_qp import CBFQP
    from safe_control.position_control.mpc_cbf import MPCCBF

    SAFE_CONTROL_AVAILABLE = True
    plt.close(_fig_tmp)
except ImportError:
    print("[WARN] safe_control package not installed. External methods will be skipped.")

import matplotlib.pyplot as plt

# ============================================================================
#  Constants
# ============================================================================

DT = 0.05
TOTAL_TIME = 15.0
NUM_STEPS = int(TOTAL_TIME / DT)
ROBOT_RADIUS = 0.25
V_MAX = 1.0
W_MAX = 0.5

MPPI_K = 512
MPPI_N = 30
MPPI_LAMBDA = 1.0
MPPI_SIGMA = np.array([0.5, 0.5])

ALL_METHODS = [
    "cbf_qp", "mpc_cbf",
    "vanilla", "cbf_mppi", "shield", "adaptive_shield",
    "cbf_guided", "shield_svg",
]

METHOD_LABELS = {
    "cbf_qp": "CBF-QP (safe_control)",
    "mpc_cbf": "MPC-CBF (safe_control)",
    "vanilla": "Vanilla MPPI",
    "cbf_mppi": "CBF-MPPI",
    "shield": "Shield-MPPI",
    "adaptive_shield": "Adaptive Shield",
    "cbf_guided": "CBF-Guided Sampling",
    "shield_svg": "Shield-SVG-MPPI",
}

METHOD_COLORS = {
    "cbf_qp": "#e74c3c",
    "mpc_cbf": "#c0392b",
    "vanilla": "#95a5a6",
    "cbf_mppi": "#3498db",
    "shield": "#2ecc71",
    "adaptive_shield": "#27ae60",
    "cbf_guided": "#9b59b6",
    "shield_svg": "#f39c12",
}


# ============================================================================
#  Scenarios
# ============================================================================

def create_scenario(name):
    """Return (obstacles, initial_state, traj_fn, description)."""
    if name == "circle_obstacle":
        # ω=0.3 rad/s → T=15s covers 4.5 rad ≈ 258° of the circle
        # v_linear = r·ω = 3.0·0.3 = 0.9 m/s (< V_MAX=1.0)
        r_path = 3.0
        omega = 0.3
        # Diverse obstacles: ON/near path, inside/outside, various sizes
        # Robot traverses 0° → 258° on the circle
        obstacles = [
            # ON path at 45° — large, requires significant detour
            (r_path * np.cos(np.radians(45)), r_path * np.sin(np.radians(45)), 0.6),
            # Inside at 110° (r=2.5) — medium, forces outward
            (2.5 * np.cos(np.radians(110)), 2.5 * np.sin(np.radians(110)), 0.40),
            # ON path at 170° — large
            (r_path * np.cos(np.radians(170)), r_path * np.sin(np.radians(170)), 0.50),
            # Outside at 220° (r=3.4) — small, forces inward
            (3.4 * np.cos(np.radians(220)), 3.4 * np.sin(np.radians(220)), 0.30),
        ]
        initial_state = np.array([3.0, 0.0, np.pi / 2])
        traj_fn = lambda t: circle_trajectory(
            t, radius=r_path, angular_velocity=omega, center=(0.0, 0.0)
        )
        desc = f"Circle (r=3, ω={omega}) with 4 diverse obstacles"
        return obstacles, initial_state, traj_fn, desc

    elif name == "gauntlet":
        # Diverse obstacles: centered, left, right, various sizes
        # Straight line y=0, forcing zigzag avoidance
        obstacles = [
            (2.0, 0.15, 0.30),    # Slightly right — medium
            (4.0, -0.2, 0.35),    # Left — medium-large
            (6.0, 0.3, 0.25),     # Right — small
            (8.0, -0.1, 0.45),    # Slightly left — large (hard)
            (10.0, 0.25, 0.30),   # Right — medium
            (12.0, -0.15, 0.25),  # Slightly left — small
        ]
        initial_state = np.array([0.0, 0.0, 0.0])
        traj_fn = lambda t: straight_line_trajectory(
            t, velocity=0.6, heading=0.0, start=(0.0, 0.0)
        )
        desc = "Straight-line gauntlet with diverse obstacles"
        return obstacles, initial_state, traj_fn, desc

    else:
        raise ValueError(f"Unknown scenario: {name}")


# ============================================================================
#  Pure Pursuit Trajectory Tracking (for safe_control nominal controller)
# ============================================================================

def find_closest_trajectory_time(state, traj_fn, t_min, t_max, n_samples=50):
    """Find the time parameter of the closest trajectory point to current state."""
    times = np.linspace(t_min, t_max, n_samples)
    pts = np.array([traj_fn(t)[:2] for t in times])
    dists = np.linalg.norm(pts - state[:2], axis=1)
    return times[np.argmin(dists)]


# ============================================================================
#  Metrics
# ============================================================================

def compute_benchmark_metrics(states, controls, solve_times, obstacles, robot_radius,
                              reference_states, traj_fn=None):
    """Compute the five benchmark metrics from recorded trajectories."""
    T = len(states)
    if T == 0:
        return {
            "collision_count": 0,
            "min_obstacle_dist": float("inf"),
            "tracking_rmse": float("inf"),
            "mean_solve_time_ms": 0.0,
            "safety_rate": 1.0,
        }

    states_arr = np.array(states)
    ref_arr = np.array(reference_states)

    # Path-following RMSE: distance to nearest point on trajectory
    # This is fairer than time-based RMSE when obstacles force detours
    if traj_fn is not None:
        # Sample many points along the trajectory for nearest-point search
        t_samples = np.linspace(0, TOTAL_TIME, 1000)
        traj_pts = np.array([traj_fn(t)[:2] for t in t_samples])
        pos_errors = np.array([
            np.min(np.linalg.norm(traj_pts - s[:2], axis=1))
            for s in states_arr
        ])
    else:
        pos_errors = np.linalg.norm(states_arr[:, :2] - ref_arr[:, :2], axis=1)
    tracking_rmse = float(np.sqrt(np.mean(pos_errors ** 2)))

    # Obstacle distances -- distance to surface (center-to-center minus radii)
    collision_count = 0
    min_dist = float("inf")
    safe_steps = 0

    for i in range(T):
        rx, ry = states_arr[i, 0], states_arr[i, 1]
        step_safe = True
        for ox, oy, obs_r in obstacles:
            d = np.sqrt((rx - ox) ** 2 + (ry - oy) ** 2) - obs_r - robot_radius
            if d < min_dist:
                min_dist = d
            if d < 0:
                step_safe = False
        if not step_safe:
            collision_count += 1
        else:
            safe_steps += 1

    safety_rate = safe_steps / T if T > 0 else 1.0
    mean_solve_time_ms = float(np.mean(solve_times)) * 1000.0

    return {
        "collision_count": collision_count,
        "min_obstacle_dist": float(min_dist),
        "tracking_rmse": tracking_rmse,
        "mean_solve_time_ms": mean_solve_time_ms,
        "safety_rate": safety_rate,
    }


# ============================================================================
#  MPPI Runner
# ============================================================================

def run_mppi_method(method_name, obstacles, initial_state, traj_fn):
    """Run an MPPI-based method and return (states, controls, solve_times, refs)."""
    model = DifferentialDriveKinematic(v_max=V_MAX, omega_max=W_MAX)

    # Build cost function (shared by all MPPI variants except SVG)
    costs = [
        StateTrackingCost(Q=np.diag([10.0, 10.0, 1.0])),
        TerminalCost(Qf=np.diag([20.0, 20.0, 2.0])),
        ControlEffortCost(R=np.diag([0.1, 0.1])),
        ObstacleCost(obstacles=obstacles, cost_weight=2000.0),
    ]
    cost_fn = CompositeMPPICost(costs)

    # CBF obstacle list for safety methods
    cbf_obs = list(obstacles)

    if method_name == "vanilla":
        params = MPPIParams(K=MPPI_K, N=MPPI_N, dt=DT, lambda_=MPPI_LAMBDA,
                            sigma=MPPI_SIGMA.copy())
        controller = MPPIController(model, params, cost_fn)

    elif method_name == "cbf_mppi":
        params = CBFMPPIParams(
            K=MPPI_K, N=MPPI_N, dt=DT, lambda_=MPPI_LAMBDA,
            sigma=MPPI_SIGMA.copy(),
            cbf_obstacles=cbf_obs, cbf_weight=1000.0, cbf_alpha=0.3,
            cbf_safety_margin=0.1,
        )
        controller = CBFMPPIController(model, params, cost_fn)

    elif method_name == "shield":
        params = ShieldMPPIParams(
            K=MPPI_K, N=MPPI_N, dt=DT, lambda_=MPPI_LAMBDA,
            sigma=MPPI_SIGMA.copy(),
            cbf_obstacles=cbf_obs, cbf_weight=1000.0, cbf_alpha=0.3,
            cbf_safety_margin=0.1, shield_enabled=True,
        )
        controller = ShieldMPPIController(model, params, cost_fn)

    elif method_name == "adaptive_shield":
        params = AdaptiveShieldParams(
            K=MPPI_K, N=MPPI_N, dt=DT, lambda_=MPPI_LAMBDA,
            sigma=MPPI_SIGMA.copy(),
            cbf_obstacles=cbf_obs, cbf_weight=1000.0, cbf_alpha=0.3,
            cbf_safety_margin=0.1, shield_enabled=True,
            alpha_base=0.3, alpha_dist=0.1, alpha_vel=0.5,
        )
        controller = AdaptiveShieldMPPIController(model, params, cost_fn)

    elif method_name == "cbf_guided":
        params = CBFGuidedSamplingParams(
            K=MPPI_K, N=MPPI_N, dt=DT, lambda_=MPPI_LAMBDA,
            sigma=MPPI_SIGMA.copy(),
            cbf_obstacles=cbf_obs, cbf_weight=1000.0, cbf_alpha=0.3,
            cbf_safety_margin=0.1,
            rejection_ratio=0.3, gradient_bias_weight=0.1,
        )
        controller = CBFGuidedSamplingMPPIController(model, params, cost_fn)

    elif method_name == "shield_svg":
        params = ShieldSVGMPPIParams(
            K=MPPI_K, N=MPPI_N, dt=DT, lambda_=MPPI_LAMBDA,
            sigma=MPPI_SIGMA.copy(),
            shield_enabled=True, shield_cbf_alpha=0.3,
            cbf_obstacles=cbf_obs, cbf_safety_margin=0.1,
        )
        # SVG-MPPI takes (model, params) only -- cost built internally
        controller = ShieldSVGMPPIController(model, params)

    else:
        raise ValueError(f"Unknown MPPI method: {method_name}")

    # Run simulation with pure-pursuit reference (re-join after obstacle detour)
    sim = Simulator(model, controller, dt=DT, store_info=False)
    sim.reset(initial_state.copy())

    states, controls, solve_times, refs = [], [], [], []
    for step_i in range(NUM_STEPS):
        t = sim.t
        # Pure pursuit: find closest point on trajectory, generate reference from there
        t_closest = find_closest_trajectory_time(
            sim.state, traj_fn, max(0, t - 1.0), min(t + 3.0, TOTAL_TIME)
        )
        ref_traj = generate_reference_trajectory(traj_fn, t_closest, MPPI_N, DT)
        step_info = sim.step(ref_traj)

        states.append(sim.history["state"][-1].tolist())
        controls.append(step_info["control"].tolist())
        solve_times.append(step_info["solve_time"])
        refs.append(traj_fn(t).tolist())  # actual time for RMSE

    return states, controls, solve_times, refs


# ============================================================================
#  safe_control Runner
# ============================================================================

def run_safe_control_method(method_name, obstacles, initial_state, traj_fn):
    """Run a safe_control method. Returns (states, controls, solve_times, refs)."""
    if not SAFE_CONTROL_AVAILABLE:
        return [], [], [], []

    fig, ax = plt.subplots()
    robot_spec = {"model": "Unicycle2D", "v_max": V_MAX, "w_max": W_MAX,
                  "radius": ROBOT_RADIUS}
    X0 = initial_state.reshape(-1, 1)
    robot = BaseRobot(X0, robot_spec, dt=DT, ax=ax)

    num_obs = len(obstacles)
    if method_name == "cbf_qp":
        solver = CBFQP(robot, robot_spec, num_obs=max(num_obs, 1))
    elif method_name == "mpc_cbf":
        solver = MPCCBF(robot, robot_spec, num_obs=max(num_obs, 1))
    else:
        raise ValueError(f"Unknown safe_control method: {method_name}")

    # Build obstacle arrays -- CBF-QP needs (7,1) columns, MPC-CBF needs (7,) 1D
    # NOTE: safe_control internally adds robot_radius to obs radius in agent_barrier,
    # so we pass just the obstacle radius (NOT r + ROBOT_RADIUS)
    if method_name == "cbf_qp":
        obs_list = [
            np.array([ox, oy, r, 0, 0, 0, 0]).reshape(-1, 1)
            for ox, oy, r in obstacles
        ]
    else:
        obs_list = [
            np.array([ox, oy, r, 0, 0, 0, 0])
            for ox, oy, r in obstacles
        ]

    states, controls, solve_times, refs = [], [], [], []

    # Warmup for MPC-CBF (first solve is slow due to CasADi compilation)
    if method_name == "mpc_cbf":
        goal_w = np.array([traj_fn(0.0)[0], traj_fn(0.0)[1]])
        u_w = robot.nominal_input(goal_w)
        cr_w = {"state_machine": "track", "u_ref": u_w, "goal": goal_w}
        try:
            solver.solve_control_problem(robot.X, cr_w, obs_list)
        except Exception:
            pass
        # Reset robot state after warmup
        robot.X = initial_state.reshape(-1, 1).copy()

    # Pure pursuit: lookahead distance (not time) for better path re-joining
    lookahead_dist = 0.5 if method_name == "cbf_qp" else 0.4

    t = 0.0
    for step_i in range(NUM_STEPS):
        # Pure pursuit: find closest point on trajectory, then lookahead from there
        cur_state = robot.X.flatten()
        t_closest = find_closest_trajectory_time(
            cur_state, traj_fn, max(0, t - 1.0), min(t + 3.0, TOTAL_TIME)
        )
        # Lookahead from closest point (ensures re-joining after detour)
        t_goal = min(t_closest + lookahead_dist, TOTAL_TIME)
        ref_point = traj_fn(t_goal)
        goal = ref_point[:2].copy()

        # Current reference for tracking error (use actual time for fair RMSE)
        ref_now = traj_fn(t)
        refs.append(ref_now.tolist())

        # Record current state
        cur_state = robot.X.flatten().tolist()
        states.append(cur_state)

        # Nominal control
        u_ref = robot.nominal_input(goal)
        control_ref = {"state_machine": "track", "u_ref": u_ref, "goal": goal}

        # Solve
        t_start = time.time()
        try:
            u = solver.solve_control_problem(robot.X, control_ref, obs_list)
        except Exception:
            u = None
        solve_time = time.time() - t_start

        # Fallback if solver returns None (infeasible)
        if u is None:
            u = u_ref

        # Step robot
        robot.step(u)

        controls.append(np.array(u).flatten().tolist())
        solve_times.append(solve_time)
        t += DT

    plt.close(fig)
    return states, controls, solve_times, refs


# ============================================================================
#  Benchmark Runner
# ============================================================================

def run_benchmark(scenario_name, methods):
    """Run all methods on a scenario. Returns dict of {method: metrics}."""
    obstacles, initial_state, traj_fn, desc = create_scenario(scenario_name)
    print(f"\n{'=' * 70}")
    print(f"  Scenario: {scenario_name} -- {desc}")
    print(f"  Obstacles: {len(obstacles)}, Steps: {NUM_STEPS}, dt={DT}s")
    print(f"{'=' * 70}")

    results = {}
    trajectories = {}

    for method in methods:
        is_external = method in ("cbf_qp", "mpc_cbf")
        if is_external and not SAFE_CONTROL_AVAILABLE:
            print(f"  [{method:>20s}] SKIPPED (safe_control not installed)")
            continue

        label = METHOD_LABELS[method]
        print(f"  [{label:>28s}] running...", end="", flush=True)

        try:
            if is_external:
                states, controls, solve_times, ref_states = run_safe_control_method(
                    method, obstacles, initial_state, traj_fn
                )
            else:
                states, controls, solve_times, ref_states = run_mppi_method(
                    method, obstacles, initial_state, traj_fn
                )

            if len(states) == 0:
                print(" FAILED (no data)")
                continue

            metrics = compute_benchmark_metrics(
                states, controls, solve_times, obstacles, ROBOT_RADIUS, ref_states,
                traj_fn=traj_fn,
            )
            results[method] = metrics
            trajectories[method] = np.array(states)

            col = metrics["collision_count"]
            sr = metrics["safety_rate"]
            rmse = metrics["tracking_rmse"]
            ms = metrics["mean_solve_time_ms"]
            print(f" done  col={col:3d}  safety={sr:.3f}  "
                  f"rmse={rmse:.3f}m  solve={ms:.1f}ms")

        except Exception as e:
            print(f" ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results, trajectories, obstacles


# ============================================================================
#  Console Output
# ============================================================================

def print_results_table(scenario_name, results):
    """Print a formatted comparison table."""
    if not results:
        return

    print(f"\n{'=' * 90}")
    print(f"  Results: {scenario_name}")
    print(f"{'=' * 90}")
    header = (f"  {'Method':<28s} {'Collisions':>10s} {'Safety%':>8s} "
              f"{'MinDist':>8s} {'RMSE(m)':>8s} {'Time(ms)':>9s}")
    print(header)
    print(f"  {'-' * 86}")

    for method in ALL_METHODS:
        if method not in results:
            continue
        m = results[method]
        label = METHOD_LABELS[method]
        col = m["collision_count"]
        sr = m["safety_rate"] * 100
        md = m["min_obstacle_dist"]
        rmse = m["tracking_rmse"]
        ms = m["mean_solve_time_ms"]

        col_marker = " **" if col > 0 else ""
        print(f"  {label:<28s} {col:>10d}{col_marker:<3s} {sr:>7.1f}% "
              f"{md:>8.3f} {rmse:>8.3f} {ms:>9.2f}")

    print(f"  {'-' * 86}")

    # Best method summary
    safe_methods = {k: v for k, v in results.items() if v["collision_count"] == 0}
    if safe_methods:
        best_tracking = min(safe_methods.items(), key=lambda x: x[1]["tracking_rmse"])
        best_speed = min(safe_methods.items(), key=lambda x: x[1]["mean_solve_time_ms"])
        print(f"\n  Best tracking (collision-free): {METHOD_LABELS[best_tracking[0]]} "
              f"(RMSE={best_tracking[1]['tracking_rmse']:.3f}m)")
        print(f"  Fastest (collision-free):       {METHOD_LABELS[best_speed[0]]} "
              f"({best_speed[1]['mean_solve_time_ms']:.2f}ms)")
    else:
        print("\n  WARNING: No collision-free method found!")


# ============================================================================
#  Visualization
# ============================================================================

def plot_results(all_results, all_trajectories, all_obstacles, scenarios, save_path):
    """Create a 6-panel comparison figure."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("MPPI vs safe_control Safety Benchmark", fontsize=16, fontweight="bold")

    # ---- Panels 1-2: XY Trajectories (one per scenario) ----
    for si, scenario in enumerate(scenarios):
        ax = fig.add_subplot(2, 3, si + 1)
        trajectories = all_trajectories.get(scenario, {})
        obstacles = all_obstacles.get(scenario, [])

        # Draw obstacles
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color="red", alpha=0.3)
            ax.add_patch(circle)
            danger = plt.Circle((ox, oy), r + ROBOT_RADIUS,
                                color="red", alpha=0.1, linestyle="--", fill=False)
            ax.add_patch(danger)

        # Draw trajectories
        for method in ALL_METHODS:
            if method not in trajectories:
                continue
            traj = trajectories[method]
            ax.plot(traj[:, 0], traj[:, 1],
                    color=METHOD_COLORS[method], linewidth=1.5,
                    label=METHOD_LABELS[method], alpha=0.8)

        ax.set_title(f"{scenario}", fontsize=11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if si == 0:
            ax.legend(fontsize=6, loc="upper left", ncol=2)

    # Collect all method data across scenarios for bar charts
    method_order = [m for m in ALL_METHODS
                    if any(m in all_results.get(s, {}) for s in scenarios)]

    def gather(metric):
        """Average metric across scenarios for each method."""
        vals = []
        for m in method_order:
            mvals = []
            for s in scenarios:
                if m in all_results.get(s, {}):
                    mvals.append(all_results[s][m][metric])
            vals.append(np.mean(mvals) if mvals else 0.0)
        return vals

    labels = [METHOD_LABELS[m] for m in method_order]
    colors = [METHOD_COLORS[m] for m in method_order]
    x_pos = np.arange(len(method_order))

    # ---- Panel 3: Tracking RMSE ----
    ax3 = fig.add_subplot(2, 3, 3)
    rmse_vals = gather("tracking_rmse")
    bars = ax3.bar(x_pos, rmse_vals, color=colors, alpha=0.8, edgecolor="black",
                   linewidth=0.5)
    ax3.set_ylabel("Tracking RMSE (m)")
    ax3.set_title("Tracking RMSE (avg across scenarios)")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, rmse_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # ---- Panel 4: Min Obstacle Distance ----
    ax4 = fig.add_subplot(2, 3, 4)
    min_dist_vals = gather("min_obstacle_dist")
    bars = ax4.bar(x_pos, min_dist_vals, color=colors, alpha=0.8, edgecolor="black",
                   linewidth=0.5)
    ax4.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Collision zone")
    ax4.set_ylabel("Min distance to obstacle surface (m)")
    ax4.set_title("Min Obstacle Distance (avg)")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax4.grid(axis="y", alpha=0.3)
    ax4.legend(fontsize=8)
    for bar, val in zip(bars, min_dist_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # ---- Panel 5: Safety Rate ----
    ax5 = fig.add_subplot(2, 3, 5)
    safety_vals = [v * 100 for v in gather("safety_rate")]
    bars = ax5.bar(x_pos, safety_vals, color=colors, alpha=0.8, edgecolor="black",
                   linewidth=0.5)
    ax5.axhline(y=100, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax5.set_ylabel("Safety Rate (%)")
    ax5.set_title("Safety Rate (avg)")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax5.set_ylim(0, 105)
    ax5.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, safety_vals):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    # ---- Panel 6: Solve Time (log scale) ----
    ax6 = fig.add_subplot(2, 3, 6)
    time_vals = gather("mean_solve_time_ms")
    bars = ax6.bar(x_pos, time_vals, color=colors, alpha=0.8, edgecolor="black",
                   linewidth=0.5)
    ax6.set_ylabel("Mean Solve Time (ms)")
    ax6.set_title("Computation Time (avg, log scale)")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax6.set_yscale("log")
    ax6.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, time_vals):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Plot saved to: {save_path}")


# ============================================================================
#  Live Animation
# ============================================================================

def compute_min_distance_single(state, obstacles, robot_radius=ROBOT_RADIUS):
    """Compute min surface distance from a single state to all obstacles."""
    min_d = float("inf")
    for ox, oy, r in obstacles:
        d = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2) - r - robot_radius
        min_d = min(min_d, d)
    return min_d


def run_live(scenario_name, methods):
    """Live animation: all methods running simultaneously, 4-panel display."""
    from matplotlib.animation import FuncAnimation

    obstacles, initial_state, traj_fn, desc = create_scenario(scenario_name)
    print(f"\n  Live mode: {scenario_name} — {desc}")
    print(f"  Methods: {[METHOD_LABELS[m] for m in methods]}")

    # --- Build controllers ---
    controllers = {}
    for method in methods:
        is_external = method in ("cbf_qp", "mpc_cbf")
        if is_external and not SAFE_CONTROL_AVAILABLE:
            print(f"  SKIP {method} (safe_control not installed)")
            continue
        try:
            if is_external:
                fig_tmp, ax_tmp = plt.subplots()
                robot_spec = {"model": "Unicycle2D", "v_max": V_MAX, "w_max": W_MAX,
                              "radius": ROBOT_RADIUS}
                X0 = initial_state.reshape(-1, 1)
                robot = BaseRobot(X0, robot_spec, dt=DT, ax=ax_tmp)
                plt.close(fig_tmp)

                num_obs = len(obstacles)
                if method == "cbf_qp":
                    solver = CBFQP(robot, robot_spec, num_obs=max(num_obs, 1))
                    obs_list = [np.array([ox, oy, r, 0, 0, 0, 0]).reshape(-1, 1)
                                for ox, oy, r in obstacles]
                else:
                    solver = MPCCBF(robot, robot_spec, num_obs=max(num_obs, 1))
                    obs_list = [np.array([ox, oy, r, 0, 0, 0, 0])
                                for ox, oy, r in obstacles]
                    # Warmup MPC-CBF
                    goal_w = np.array([traj_fn(0.0)[0], traj_fn(0.0)[1]])
                    u_w = robot.nominal_input(goal_w)
                    cr_w = {"state_machine": "track", "u_ref": u_w, "goal": goal_w}
                    try:
                        solver.solve_control_problem(robot.X, cr_w, obs_list)
                    except Exception:
                        pass
                    # Reset robot state after warmup
                    robot.X = initial_state.reshape(-1, 1).copy()

                controllers[method] = {
                    "type": "external", "robot": robot, "solver": solver,
                    "obs_list": obs_list, "state": initial_state.copy(),
                }
            else:
                model = DifferentialDriveKinematic(v_max=V_MAX, omega_max=W_MAX)
                costs = [
                    StateTrackingCost(Q=np.diag([10.0, 10.0, 1.0])),
                    TerminalCost(Qf=np.diag([20.0, 20.0, 2.0])),
                    ControlEffortCost(R=np.diag([0.1, 0.1])),
                    ObstacleCost(obstacles=obstacles, cost_weight=2000.0),
                ]
                cost_fn = CompositeMPPICost(costs)
                cbf_obs = list(obstacles)

                if method == "vanilla":
                    params = MPPIParams(K=MPPI_K, N=MPPI_N, dt=DT,
                                        lambda_=MPPI_LAMBDA, sigma=MPPI_SIGMA.copy())
                    ctrl = MPPIController(model, params, cost_fn)
                elif method == "cbf_mppi":
                    params = CBFMPPIParams(K=MPPI_K, N=MPPI_N, dt=DT,
                                            lambda_=MPPI_LAMBDA, sigma=MPPI_SIGMA.copy(),
                                            cbf_obstacles=cbf_obs, cbf_weight=1000.0,
                                            cbf_alpha=0.3, cbf_safety_margin=0.1)
                    ctrl = CBFMPPIController(model, params, cost_fn)
                elif method == "shield":
                    params = ShieldMPPIParams(K=MPPI_K, N=MPPI_N, dt=DT,
                                              lambda_=MPPI_LAMBDA, sigma=MPPI_SIGMA.copy(),
                                              cbf_obstacles=cbf_obs, cbf_weight=1000.0,
                                              cbf_alpha=0.3, cbf_safety_margin=0.1,
                                              shield_enabled=True)
                    ctrl = ShieldMPPIController(model, params, cost_fn)
                elif method == "adaptive_shield":
                    params = AdaptiveShieldParams(K=MPPI_K, N=MPPI_N, dt=DT,
                                                  lambda_=MPPI_LAMBDA, sigma=MPPI_SIGMA.copy(),
                                                  cbf_obstacles=cbf_obs, cbf_weight=1000.0,
                                                  cbf_alpha=0.3, cbf_safety_margin=0.1,
                                                  shield_enabled=True,
                                                  alpha_base=0.3, alpha_dist=0.1, alpha_vel=0.5)
                    ctrl = AdaptiveShieldMPPIController(model, params, cost_fn)
                elif method == "cbf_guided":
                    params = CBFGuidedSamplingParams(K=MPPI_K, N=MPPI_N, dt=DT,
                                                     lambda_=MPPI_LAMBDA, sigma=MPPI_SIGMA.copy(),
                                                     cbf_obstacles=cbf_obs, cbf_weight=1000.0,
                                                     cbf_alpha=0.3, cbf_safety_margin=0.1,
                                                     rejection_ratio=0.3, gradient_bias_weight=0.1)
                    ctrl = CBFGuidedSamplingMPPIController(model, params, cost_fn)
                elif method == "shield_svg":
                    params = ShieldSVGMPPIParams(K=MPPI_K, N=MPPI_N, dt=DT,
                                                 lambda_=MPPI_LAMBDA, sigma=MPPI_SIGMA.copy(),
                                                 shield_enabled=True, shield_cbf_alpha=0.3,
                                                 cbf_obstacles=cbf_obs, cbf_safety_margin=0.1)
                    ctrl = ShieldSVGMPPIController(model, params)
                else:
                    continue

                sim = Simulator(model, ctrl, dt=DT, store_info=False)
                sim.reset(initial_state.copy())
                controllers[method] = {
                    "type": "mppi", "sim": sim, "model": model,
                    "state": initial_state.copy(),
                }
        except Exception as e:
            print(f"  SKIP {method}: {e}")

    if not controllers:
        print("  No controllers available!")
        return

    # --- Data storage ---
    data = {m: {"xy": [], "times": [], "errors": [], "min_dist": []}
            for m in controllers}

    # --- Figure setup (2x2 panels) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"MPPI vs safe_control Live — {scenario_name}",
                 fontsize=14, fontweight="bold")

    # [0,0] XY Trajectories
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories + Obstacles")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    for ox, oy, r in obstacles:
        ax_xy.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))
        ax_xy.add_patch(plt.Circle((ox, oy), r + ROBOT_RADIUS,
                                    color="red", alpha=0.1, linestyle="--", fill=False))

    # Reference path (also used for path-following RMSE)
    ref_t = np.linspace(0, TOTAL_TIME, 1000)
    ref_pts = np.array([traj_fn(t) for t in ref_t])
    ref_pts_2d = ref_pts[:, :2]  # (1000, 2) for nearest-point search
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

    # Lines and dots per method
    lines_xy = {}
    dots = {}
    for m in controllers:
        c = METHOD_COLORS[m]
        lines_xy[m], = ax_xy.plot([], [], color=c, linewidth=1.5,
                                   label=METHOD_LABELS[m], alpha=0.8)
        dots[m], = ax_xy.plot([], [], "o", color=c, markersize=7)
    ax_xy.legend(loc="upper left", fontsize=6, ncol=2)

    # [0,1] Path-Following Error
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Path Error (m)")
    ax_err.set_title("Path-Following Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for m in controllers:
        lines_err[m], = ax_err.plot([], [], color=METHOD_COLORS[m],
                                     linewidth=1.5, label=METHOD_LABELS[m])
    ax_err.legend(fontsize=6, ncol=2)

    # [1,0] Min Obstacle Distance
    ax_dist = axes[1, 0]
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("Min Distance (m)")
    ax_dist.set_title("Min Distance to Obstacle Surface")
    ax_dist.grid(True, alpha=0.3)
    ax_dist.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    lines_dist = {}
    for m in controllers:
        lines_dist[m], = ax_dist.plot([], [], color=METHOD_COLORS[m],
                                       linewidth=1.5, label=METHOD_LABELS[m])
    ax_dist.legend(fontsize=6, ncol=2)

    # [1,1] Stats text
    ax_stats = axes[1, 1]
    ax_stats.axis("off")
    ax_stats.set_title("Live Statistics")
    stats_text = ax_stats.text(0.05, 0.95, "", transform=ax_stats.transAxes,
                                fontsize=9, family="monospace", va="top")

    time_text = ax_xy.text(0.5, -0.08, "", transform=ax_xy.transAxes,
                            ha="center", fontsize=9, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    all_artists = list(lines_xy.values()) + list(dots.values()) + \
                  list(lines_err.values()) + list(lines_dist.values()) + \
                  [stats_text, time_text]

    def init():
        for a in all_artists:
            if hasattr(a, "set_data"):
                a.set_data([], [])
        stats_text.set_text("")
        time_text.set_text("")
        return all_artists

    def update(frame):
        if frame >= NUM_STEPS:
            return all_artists

        for m, ctrl_info in controllers.items():
            if ctrl_info["type"] == "mppi":
                sim = ctrl_info["sim"]
                t = sim.t
                # Pure pursuit reference: re-join after obstacle detour
                t_closest = find_closest_trajectory_time(
                    sim.state, traj_fn, max(0, t - 1.0), min(t + 3.0, TOTAL_TIME)
                )
                ref_traj = generate_reference_trajectory(traj_fn, t_closest, MPPI_N, DT)
                sim.step(ref_traj)
                state = sim.state.copy()
                ref_pt = traj_fn(t)[:2]
            else:
                robot = ctrl_info["robot"]
                solver = ctrl_info["solver"]
                obs_list = ctrl_info["obs_list"]
                t = frame * DT
                cur_state = robot.X.flatten()
                # Pure pursuit: find closest traj point, lookahead from there
                la_dist = 0.5 if m == "cbf_qp" else 0.4
                t_closest = find_closest_trajectory_time(
                    cur_state, traj_fn, max(0, t - 1.0), min(t + 3.0, TOTAL_TIME)
                )
                t_goal = min(t_closest + la_dist, TOTAL_TIME)
                ref_point = traj_fn(t_goal)
                goal = ref_point[:2].copy()
                ref_pt = traj_fn(t)[:2]

                u_ref = robot.nominal_input(goal)
                control_ref = {"state_machine": "track", "u_ref": u_ref, "goal": goal}
                try:
                    u = solver.solve_control_problem(robot.X, control_ref, obs_list)
                except Exception:
                    u = u_ref
                if u is None:
                    u = u_ref
                robot.step(u)
                state = robot.X.flatten()

            data[m]["xy"].append(state[:2].copy())
            data[m]["times"].append(frame * DT)
            # Path-following error: distance to nearest trajectory point
            path_err = float(np.min(np.linalg.norm(ref_pts_2d - state[:2], axis=1)))
            data[m]["errors"].append(path_err)
            data[m]["min_dist"].append(
                compute_min_distance_single(state, obstacles))

        # Update plots
        for m in controllers:
            xy = np.array(data[m]["xy"])
            times = np.array(data[m]["times"])
            if len(xy) == 0:
                continue
            lines_xy[m].set_data(xy[:, 0], xy[:, 1])
            dots[m].set_data([xy[-1, 0]], [xy[-1, 1]])
            lines_err[m].set_data(times, data[m]["errors"])
            lines_dist[m].set_data(times, data[m]["min_dist"])

        ax_err.relim()
        ax_err.autoscale_view()
        ax_dist.relim()
        ax_dist.autoscale_view()

        # Stats text
        t_now = frame * DT
        lines = [f"t = {t_now:.1f}s / {TOTAL_TIME:.1f}s\n"]
        lines.append(f"{'Method':<24s} {'RMSE':>7s} {'MinD':>7s} {'Col':>4s}")
        lines.append("-" * 46)
        for m in controllers:
            errs = data[m]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            min_d = min(data[m]["min_dist"]) if data[m]["min_dist"] else 0
            col = sum(1 for d in data[m]["min_dist"] if d < 0)
            lines.append(f"{METHOD_LABELS[m]:<24s} {rmse:>7.3f} {min_d:>7.3f} {col:>4d}")
        stats_text.set_text("\n".join(lines))
        time_text.set_text(f"t = {t_now:.1f}s")

        return all_artists

    anim = FuncAnimation(fig, update, init_func=init,
                          frames=NUM_STEPS, interval=1, blit=False, repeat=False)
    plt.show()

    # Print final stats
    print(f"\n{'=' * 60}")
    print("  Live Simulation Complete")
    print(f"{'=' * 60}")
    for m in controllers:
        errs = data[m]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        min_d = min(data[m]["min_dist"]) if data[m]["min_dist"] else 0
        col = sum(1 for d in data[m]["min_dist"] if d < 0)
        print(f"  {METHOD_LABELS[m]:<28s} RMSE={rmse:.3f}m  MinDist={min_d:.3f}m  Col={col}")
    print(f"{'=' * 60}\n")


# ============================================================================
#  Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MPPI vs safe_control Safety Benchmark"
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=["circle_obstacle", "gauntlet"],
        help="Run a single scenario (default: all)",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated method list (e.g. cbf_qp,shield,adaptive_shield)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation, print table only",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live animation mode (single scenario, real-time)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine scenarios
    if args.scenario:
        scenarios = [args.scenario]
    else:
        scenarios = ["circle_obstacle", "gauntlet"]

    # Determine methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        for m in methods:
            if m not in ALL_METHODS:
                print(f"ERROR: Unknown method '{m}'. Available: {ALL_METHODS}")
                sys.exit(1)
    else:
        methods = list(ALL_METHODS)

    print("=" * 70)
    print("  MPPI vs safe_control Safety Benchmark")
    print("=" * 70)
    print(f"  Scenarios: {scenarios}")
    print(f"  Methods:   {[METHOD_LABELS[m] for m in methods]}")
    print(f"  safe_control available: {SAFE_CONTROL_AVAILABLE}")
    print(f"  Sim: dt={DT}s, T={TOTAL_TIME}s, steps={NUM_STEPS}")
    print(f"  MPPI: K={MPPI_K}, N={MPPI_N}, lambda={MPPI_LAMBDA}")

    # --- Live mode ---
    if args.live:
        scenario = scenarios[0]
        run_live(scenario, methods)
        return

    all_results = {}
    all_trajectories = {}
    all_obstacles = {}

    for scenario in scenarios:
        results, trajectories, obstacles = run_benchmark(scenario, methods)
        all_results[scenario] = results
        all_trajectories[scenario] = trajectories
        all_obstacles[scenario] = obstacles
        print_results_table(scenario, results)

    # Cross-scenario summary
    if len(scenarios) > 1:
        print(f"\n{'=' * 90}")
        print(f"  Cross-Scenario Summary (averaged)")
        print(f"{'=' * 90}")
        header = (f"  {'Method':<28s} {'Collisions':>10s} {'Safety%':>8s} "
                  f"{'MinDist':>8s} {'RMSE(m)':>8s} {'Time(ms)':>9s}")
        print(header)
        print(f"  {'-' * 86}")

        for method in methods:
            mvals = [all_results[s][method] for s in scenarios
                     if method in all_results.get(s, {})]
            if not mvals:
                continue
            label = METHOD_LABELS[method]
            avg_col = np.mean([v["collision_count"] for v in mvals])
            avg_sr = np.mean([v["safety_rate"] for v in mvals]) * 100
            avg_md = np.mean([v["min_obstacle_dist"] for v in mvals])
            avg_rmse = np.mean([v["tracking_rmse"] for v in mvals])
            avg_ms = np.mean([v["mean_solve_time_ms"] for v in mvals])
            col_marker = " **" if avg_col > 0 else ""
            print(f"  {label:<28s} {avg_col:>10.1f}{col_marker:<3s} {avg_sr:>7.1f}% "
                  f"{avg_md:>8.3f} {avg_rmse:>8.3f} {avg_ms:>9.2f}")

        print(f"  {'-' * 86}")

    # Visualization
    if not args.no_plot:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../plots/mppi_vs_safe_control_benchmark.png"
        )
        save_path = os.path.abspath(save_path)
        plot_results(all_results, all_trajectories, all_obstacles, scenarios, save_path)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
