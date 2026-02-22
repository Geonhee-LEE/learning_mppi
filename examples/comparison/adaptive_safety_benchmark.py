#!/usr/bin/env python3
"""
Adaptive Dynamics + Obstacle Avoidance Safety Benchmark

모델 부정확(mismatch) 환경에서 적응 기법(EKF, L1, ALPaCA)이
안전 제어(CBF, Shield)와 조합될 때 장애물 회피 성능을 비교합니다.

Layer 구조:
  Layer 1 (Model)  : Nominal / EKF / L1 / ALPaCA — 동역학 적응
  Layer 2 (Safety) : None / CBF-MPPI / Shield-MPPI — 안전 제약
  Layer 3 (Optim)  : MPPI 가중 평균

비교 조합 (9종):
  1. Nominal (no adaptation, no safety)     — baseline
  2. Nominal + CBF-MPPI                     — safety only
  3. Nominal + Shield-MPPI                  — safety only
  4. EKF (no safety)                        — adaptation only
  5. EKF + Shield-MPPI                      — adaptation + safety
  6. L1 (no safety)                         — adaptation only
  7. L1 + Shield-MPPI                       — adaptation + safety
  8. ALPaCA (no safety)                     — adaptation only
  9. ALPaCA + Shield-MPPI                   — adaptation + safety

Usage:
    python adaptive_safety_benchmark.py                         # all methods
    python adaptive_safety_benchmark.py --live                  # live animation
    python adaptive_safety_benchmark.py --live --scenario gauntlet
    python adaptive_safety_benchmark.py --methods 1,5,7,9       # subset
    python adaptive_safety_benchmark.py --no-plot               # table only
    python adaptive_safety_benchmark.py --no-mismatch           # perfect model
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
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- Imports ---
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams, CBFMPPIParams, ShieldMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost, ControlEffortCost, ObstacleCost,
)
from mppi_controller.controllers.mppi.cbf_cost import ControlBarrierCost
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.kinematic.dynamic_kinematic_adapter import (
    DynamicKinematicAdapter,
)
from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
from mppi_controller.controllers.mppi.cost_functions import AngleAwareTrackingCost, AngleAwareTerminalCost
from mppi_controller.utils.trajectory import (
    circle_trajectory, straight_line_trajectory, generate_reference_trajectory,
)

# ALPaCA (optional — needs trained model)
ALPACA_AVAILABLE = False
try:
    from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
    from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
    ALPACA_AVAILABLE = True
except ImportError:
    pass

import matplotlib.pyplot as plt

# ============================================================================
#  Constants
# ============================================================================

DT = 0.05
TOTAL_TIME = 15.0
NUM_STEPS = int(TOTAL_TIME / DT)
V_MAX = 1.0
W_MAX = 0.5

# MPPI parameters
MPPI_K = 512
MPPI_N = 20

# Model mismatch: real world has higher friction than nominal
REAL_C_V = 0.5       # real friction (velocity)
REAL_C_OMEGA = 0.3   # real friction (angular)
NOM_C_V = 0.1        # nominal (what controller believes)
NOM_C_OMEGA = 0.1
NOISE_STD = np.array([0.005, 0.005, 0.002, 0.01, 0.005])

ALL_METHODS = [
    "nominal",         # 1
    "nominal_cbf",     # 2
    "nominal_shield",  # 3
    "ekf",             # 4
    "ekf_shield",      # 5
    "l1",              # 6
    "l1_shield",       # 7
    "alpaca",          # 8
    "alpaca_shield",   # 9
]

METHOD_LABELS = {
    "nominal":        "Nominal (no adapt)",
    "nominal_cbf":    "Nominal + CBF",
    "nominal_shield": "Nominal + Shield",
    "ekf":            "EKF",
    "ekf_shield":     "EKF + Shield",
    "l1":             "L1 Adaptive",
    "l1_shield":      "L1 + Shield",
    "alpaca":         "ALPaCA",
    "alpaca_shield":  "ALPaCA + Shield",
}

METHOD_COLORS = {
    "nominal":        "#95a5a6",
    "nominal_cbf":    "#3498db",
    "nominal_shield": "#2ecc71",
    "ekf":            "#e67e22",
    "ekf_shield":     "#d35400",
    "l1":             "#9b59b6",
    "l1_shield":      "#8e44ad",
    "alpaca":         "#e74c3c",
    "alpaca_shield":  "#c0392b",
}


# ============================================================================
#  DynamicWorld (real world with model mismatch)
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
#  5D utilities
# ============================================================================

def make_5d_reference(ref_3d):
    """3D reference (N+1, 3) -> 5D reference (N+1, 5) with v/omega estimation."""
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
    """5D angle-aware cost with optional obstacle cost."""
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


def create_5d_params(safety_type="none", obstacles=None):
    """Create 5D MPPI params with optional safety layer."""
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

    if safety_type == "cbf":
        return CBFMPPIParams(**base_kwargs, **cbf_kwargs)
    elif safety_type == "shield":
        return ShieldMPPIParams(**base_kwargs, **cbf_kwargs,
                                shield_enabled=True, shield_cbf_alpha=0.3)
    else:
        return MPPIParams(**base_kwargs)


# ============================================================================
#  Scenarios
# ============================================================================

def create_scenario(name):
    """Return (obstacles, initial_state_3d, traj_fn, description)."""
    if name == "circle_obstacle":
        obstacles = [
            (2.3, 0.0, 0.35),
            (0.0, 3.7, 0.35),
            (-2.5, -0.5, 0.35),
            (0.5, -3.5, 0.35),
        ]
        initial_state = np.array([3.0, 0.0, np.pi / 2])
        traj_fn = lambda t: circle_trajectory(
            t, radius=3.0, angular_velocity=0.1, center=(0.0, 0.0)
        )
        return obstacles, initial_state, traj_fn, "Circle with 4 nearby obstacles"

    elif name == "gauntlet":
        obstacles = [
            (2.0, 0.6, 0.3),
            (4.0, -0.6, 0.3),
            (6.0, 0.5, 0.3),
            (8.0, -0.5, 0.3),
            (10.0, 0.4, 0.3),
            (12.0, -0.4, 0.3),
        ]
        initial_state = np.array([0.0, 0.0, 0.0])
        traj_fn = lambda t: straight_line_trajectory(
            t, velocity=0.6, heading=0.0, start=(0.0, 0.0)
        )
        return obstacles, initial_state, traj_fn, "Straight-line gauntlet"
    else:
        raise ValueError(f"Unknown scenario: {name}")


# ============================================================================
#  Metrics
# ============================================================================

def compute_metrics(states_3d, solve_times, obstacles, ref_states):
    """Compute safety/tracking/compute metrics."""
    T = len(states_3d)
    if T == 0:
        return {"collision_count": 0, "min_obstacle_dist": float("inf"),
                "tracking_rmse": float("inf"), "mean_solve_time_ms": 0.0,
                "safety_rate": 1.0}

    states = np.array(states_3d)
    refs = np.array(ref_states)

    pos_errors = np.linalg.norm(states[:, :2] - refs[:, :2], axis=1)
    tracking_rmse = float(np.sqrt(np.mean(pos_errors**2)))

    collision_count = 0
    min_dist = float("inf")
    safe_steps = 0
    robot_radius = 0.25

    for i in range(T):
        step_safe = True
        for ox, oy, r in obstacles:
            d = np.sqrt((states[i, 0] - ox)**2 + (states[i, 1] - oy)**2) - r - robot_radius
            min_dist = min(min_dist, d)
            if d < 0:
                step_safe = False
        if step_safe:
            safe_steps += 1
        else:
            collision_count += 1

    return {
        "collision_count": collision_count,
        "min_obstacle_dist": float(min_dist),
        "tracking_rmse": tracking_rmse,
        "mean_solve_time_ms": float(np.mean(solve_times)) * 1000.0,
        "safety_rate": safe_steps / T if T > 0 else 1.0,
    }


# ============================================================================
#  Simulation Runner
# ============================================================================

def find_alpaca_model():
    """Find a trained ALPaCA model file."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "../../trained_models/alpaca_5d/best_model.pth"),
        os.path.join(os.path.dirname(__file__), "../../trained_models/alpaca_model.pth"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return None


def run_method(method_name, obstacles, initial_state_3d, traj_fn, mismatch=True, seed=42):
    """Run a single method. Returns (states_3d, solve_times, ref_states)."""
    np.random.seed(seed)

    # Determine safety type
    if method_name.endswith("_cbf"):
        safety_type = "cbf"
    elif method_name.endswith("_shield"):
        safety_type = "shield"
    else:
        safety_type = "none"

    # Determine adaptation type
    adapt_type = method_name.replace("_cbf", "").replace("_shield", "")

    params = create_5d_params(safety_type, obstacles)
    cost_fn = create_5d_cost(obstacles, include_obstacle_cost=True)

    # Create model + controller
    if adapt_type == "nominal":
        model = DynamicKinematicAdapter(c_v=NOM_C_V, c_omega=NOM_C_OMEGA,
                                         k_v=5.0, k_omega=5.0)
        if safety_type == "cbf":
            controller = CBFMPPIController(model, params, cost_fn)
        elif safety_type == "shield":
            controller = ShieldMPPIController(model, params, cost_fn)
        else:
            controller = MPPIController(model, params, cost_fn)

    elif adapt_type == "ekf":
        model = EKFAdaptiveDynamics(c_v_init=NOM_C_V, c_omega_init=NOM_C_OMEGA,
                                     k_v=5.0, k_omega=5.0)
        if safety_type == "shield":
            controller = ShieldMPPIController(model, params, cost_fn)
        else:
            controller = MPPIController(model, params, cost_fn)

    elif adapt_type == "l1":
        model = L1AdaptiveDynamics(c_v_nom=NOM_C_V, c_omega_nom=NOM_C_OMEGA,
                                    k_v=5.0, k_omega=5.0)
        if safety_type == "shield":
            controller = ShieldMPPIController(model, params, cost_fn)
        else:
            controller = MPPIController(model, params, cost_fn)

    elif adapt_type == "alpaca":
        if not ALPACA_AVAILABLE:
            print(f"    ALPaCA not available (import failed)")
            return [], [], []
        alpaca_path = find_alpaca_model()
        if alpaca_path is None:
            print(f"    ALPaCA model not found (run meta-training first)")
            return [], [], []

        base_model = DynamicKinematicAdapter(c_v=NOM_C_V, c_omega=NOM_C_OMEGA,
                                              k_v=5.0, k_omega=5.0)
        alpaca_model = ALPaCADynamics(state_dim=5, control_dim=2, model_path=alpaca_path)
        residual_model = ResidualDynamics(
            base_model=base_model, learned_model=alpaca_model, use_residual=True,
        )

        # Phase 1 controller (base only), Phase 2 controller (residual)
        if safety_type == "shield":
            ctrl_phase1 = ShieldMPPIController(base_model, params, cost_fn)
            ctrl_phase2 = ShieldMPPIController(residual_model, params, cost_fn)
        else:
            ctrl_phase1 = MPPIController(base_model, params, cost_fn)
            ctrl_phase2 = MPPIController(residual_model, params, cost_fn)
    else:
        raise ValueError(f"Unknown adapt type: {adapt_type}")

    # Create world
    world = DynamicWorld(mismatch=mismatch)
    world.reset(initial_state_3d)
    state_5d = np.array([*initial_state_3d, 0.0, 0.0])

    states_3d = []
    solve_times = []
    ref_states = []
    prev_state = state_5d.copy()
    prev_control = np.zeros(2)

    # ALPaCA buffers
    if adapt_type == "alpaca" and ALPACA_AVAILABLE:
        warmup_steps = 10
        adapt_interval = 20
        buffer_size = 50
        buf_s = deque(maxlen=buffer_size)
        buf_c = deque(maxlen=buffer_size)
        buf_n = deque(maxlen=buffer_size)
        adapted = False

    t = 0.0
    for step_i in range(NUM_STEPS):
        ref_3d = generate_reference_trajectory(traj_fn, t, MPPI_N, DT)
        ref_5d = make_5d_reference(ref_3d)

        t_start = time.time()
        if adapt_type == "alpaca" and ALPACA_AVAILABLE:
            if not adapted:
                control, info = ctrl_phase1.compute_control(state_5d, ref_5d)
            else:
                control, info = ctrl_phase2.compute_control(state_5d, ref_5d)
        else:
            control, info = controller.compute_control(state_5d, ref_5d)
        solve_time = time.time() - t_start

        states_3d.append(state_5d[:3].copy())
        ref_states.append(ref_3d[0].copy())
        solve_times.append(solve_time)

        # Step world
        world.step(control, DT, add_noise=True)
        next_5d = world.get_full_state()

        # Adaptation updates
        if adapt_type == "ekf" and step_i > 0:
            model.update_step(prev_state, prev_control, state_5d, DT)
        elif adapt_type == "l1" and step_i > 0:
            model.update_step(prev_state, prev_control, state_5d, DT)
        elif adapt_type == "alpaca" and ALPACA_AVAILABLE:
            buf_s.append(state_5d.copy())
            buf_c.append(control.copy())
            buf_n.append(next_5d.copy())

            if step_i >= warmup_steps and step_i % adapt_interval == 0:
                s_arr = np.array(buf_s)
                c_arr = np.array(buf_c)
                n_arr = np.array(buf_n)
                kin_next = s_arr + base_model.forward_dynamics(s_arr, c_arr) * DT
                residual = n_arr - kin_next
                alpaca_model.adapt(s_arr, c_arr, residual, DT)
                adapted = True

        prev_state = state_5d.copy()
        prev_control = control.copy()
        state_5d = next_5d
        t += DT

    return states_3d, solve_times, ref_states


# ============================================================================
#  Benchmark Runner
# ============================================================================

def run_benchmark(scenario_name, methods, mismatch=True):
    """Run all methods on a scenario."""
    obstacles, initial_state, traj_fn, desc = create_scenario(scenario_name)
    mismatch_str = "WITH" if mismatch else "WITHOUT"
    print(f"\n{'=' * 70}")
    print(f"  Scenario: {scenario_name} — {desc}")
    print(f"  Model mismatch: {mismatch_str} (real c_v={REAL_C_V}, nom c_v={NOM_C_V})")
    print(f"  Obstacles: {len(obstacles)}, Steps: {NUM_STEPS}")
    print(f"{'=' * 70}")

    results = {}
    trajectories = {}

    for method in methods:
        if method.startswith("alpaca") and not ALPACA_AVAILABLE:
            print(f"  [{METHOD_LABELS[method]:>28s}] SKIPPED (ALPaCA not available)")
            continue

        label = METHOD_LABELS[method]
        print(f"  [{label:>28s}] running...", end="", flush=True)

        try:
            states, solve_times, refs = run_method(
                method, obstacles, initial_state, traj_fn,
                mismatch=mismatch, seed=42,
            )
            if len(states) == 0:
                print(" FAILED")
                continue

            metrics = compute_metrics(states, solve_times, obstacles, refs)
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
    """Print formatted comparison table."""
    if not results:
        return

    print(f"\n{'=' * 92}")
    print(f"  Results: {scenario_name}")
    print(f"{'=' * 92}")
    header = (f"  {'Method':<28s} {'Adapt':>6s} {'Safety':>7s} "
              f"{'Col':>5s} {'Safe%':>6s} {'MinD':>7s} {'RMSE':>7s} {'ms':>7s}")
    print(header)
    print(f"  {'-' * 88}")

    for method in ALL_METHODS:
        if method not in results:
            continue
        m = results[method]
        label = METHOD_LABELS[method]
        adapt = method.replace("_cbf", "").replace("_shield", "")
        safety = "Shield" if "_shield" in method else ("CBF" if "_cbf" in method else "—")
        col = m["collision_count"]
        sr = m["safety_rate"] * 100
        md = m["min_obstacle_dist"]
        rmse = m["tracking_rmse"]
        ms = m["mean_solve_time_ms"]
        col_mark = " **" if col > 0 else ""
        print(f"  {label:<28s} {adapt:>6s} {safety:>7s} "
              f"{col:>5d}{col_mark:<3s}{sr:>5.1f}% {md:>7.3f} {rmse:>7.3f} {ms:>7.1f}")

    print(f"  {'-' * 88}")

    # Best combo
    safe_methods = {k: v for k, v in results.items() if v["collision_count"] == 0}
    if safe_methods:
        best = min(safe_methods.items(), key=lambda x: x[1]["tracking_rmse"])
        print(f"\n  Best (collision-free): {METHOD_LABELS[best[0]]} "
              f"(RMSE={best[1]['tracking_rmse']:.3f}m)")


# ============================================================================
#  Visualization
# ============================================================================

def plot_results(all_results, all_trajectories, all_obstacles, scenarios, save_path):
    """6-panel comparison figure."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Adaptive Dynamics + Safety: Obstacle Avoidance Benchmark",
                 fontsize=16, fontweight="bold")

    for si, scenario in enumerate(scenarios):
        ax = fig.add_subplot(2, 3, si + 1)
        trajectories = all_trajectories.get(scenario, {})
        obstacles = all_obstacles.get(scenario, [])

        for ox, oy, r in obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))
            ax.add_patch(plt.Circle((ox, oy), r + 0.25,
                                     color="red", alpha=0.1, linestyle="--", fill=False))

        for method in ALL_METHODS:
            if method not in trajectories:
                continue
            traj = trajectories[method]
            ax.plot(traj[:, 0], traj[:, 1],
                    color=METHOD_COLORS[method], linewidth=1.5,
                    label=METHOD_LABELS[method], alpha=0.8)

        ax.set_title(scenario, fontsize=11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if si == 0:
            ax.legend(fontsize=5, loc="upper left", ncol=2)

    # Gather data for bar charts
    method_order = [m for m in ALL_METHODS
                    if any(m in all_results.get(s, {}) for s in scenarios)]

    def gather(metric):
        vals = []
        for m in method_order:
            mvals = [all_results[s][m][metric] for s in scenarios
                     if m in all_results.get(s, {})]
            vals.append(np.mean(mvals) if mvals else 0.0)
        return vals

    labels = [METHOD_LABELS[m] for m in method_order]
    colors = [METHOD_COLORS[m] for m in method_order]
    x_pos = np.arange(len(method_order))

    # Panel 3: RMSE
    ax3 = fig.add_subplot(2, 3, 3)
    rmse_vals = gather("tracking_rmse")
    bars = ax3.bar(x_pos, rmse_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Tracking RMSE (m)")
    ax3.set_title("Tracking RMSE (avg)")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, rmse_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    # Panel 4: Min Dist
    ax4 = fig.add_subplot(2, 3, 4)
    dist_vals = gather("min_obstacle_dist")
    bars = ax4.bar(x_pos, dist_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax4.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    ax4.set_ylabel("Min obstacle dist (m)")
    ax4.set_title("Min Obstacle Distance (avg)")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax4.grid(axis="y", alpha=0.3)

    # Panel 5: Safety Rate
    ax5 = fig.add_subplot(2, 3, 5)
    safety_vals = [v * 100 for v in gather("safety_rate")]
    bars = ax5.bar(x_pos, safety_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax5.axhline(y=100, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax5.set_ylabel("Safety Rate (%)")
    ax5.set_title("Safety Rate (avg)")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax5.set_ylim(0, 105)
    ax5.grid(axis="y", alpha=0.3)

    # Panel 6: Solve Time
    ax6 = fig.add_subplot(2, 3, 6)
    time_vals = gather("mean_solve_time_ms")
    bars = ax6.bar(x_pos, time_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax6.set_ylabel("Solve Time (ms)")
    ax6.set_title("Computation Time (avg)")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax6.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Plot saved to: {save_path}")


# ============================================================================
#  Live Animation
# ============================================================================

def run_live(scenario_name, methods, mismatch=True):
    """Live animation with all methods running simultaneously."""
    from matplotlib.animation import FuncAnimation

    obstacles, initial_state_3d, traj_fn, desc = create_scenario(scenario_name)
    print(f"\n  Live: {scenario_name} — {desc}")

    # Build controllers (same structure as run_method but keep references)
    controllers = {}
    for method in methods:
        if method.startswith("alpaca") and not ALPACA_AVAILABLE:
            continue

        safety_type = "shield" if "_shield" in method else ("cbf" if "_cbf" in method else "none")
        adapt_type = method.replace("_cbf", "").replace("_shield", "")

        try:
            params = create_5d_params(safety_type, obstacles)
            cost_fn = create_5d_cost(obstacles)
            world = DynamicWorld(mismatch=mismatch)
            world.reset(initial_state_3d)
            state_5d = np.array([*initial_state_3d, 0.0, 0.0])

            if adapt_type == "nominal":
                model = DynamicKinematicAdapter(c_v=NOM_C_V, c_omega=NOM_C_OMEGA, k_v=5.0, k_omega=5.0)
                if safety_type == "cbf":
                    ctrl = CBFMPPIController(model, params, cost_fn)
                elif safety_type == "shield":
                    ctrl = ShieldMPPIController(model, params, cost_fn)
                else:
                    ctrl = MPPIController(model, params, cost_fn)
            elif adapt_type == "ekf":
                model = EKFAdaptiveDynamics(c_v_init=NOM_C_V, c_omega_init=NOM_C_OMEGA, k_v=5.0, k_omega=5.0)
                ctrl = ShieldMPPIController(model, params, cost_fn) if safety_type == "shield" else MPPIController(model, params, cost_fn)
            elif adapt_type == "l1":
                model = L1AdaptiveDynamics(c_v_nom=NOM_C_V, c_omega_nom=NOM_C_OMEGA, k_v=5.0, k_omega=5.0)
                ctrl = ShieldMPPIController(model, params, cost_fn) if safety_type == "shield" else MPPIController(model, params, cost_fn)
            elif adapt_type == "alpaca":
                alpaca_path = find_alpaca_model()
                if not alpaca_path:
                    continue
                base_model = DynamicKinematicAdapter(c_v=NOM_C_V, c_omega=NOM_C_OMEGA, k_v=5.0, k_omega=5.0)
                alpaca_model = ALPaCADynamics(state_dim=5, control_dim=2, model_path=alpaca_path)
                residual_model = ResidualDynamics(base_model=base_model, learned_model=alpaca_model, use_residual=True)
                ctrl = ShieldMPPIController(residual_model, params, cost_fn) if safety_type == "shield" else MPPIController(residual_model, params, cost_fn)
                # Store for ALPaCA adaptation
                model = None  # no update_step
            else:
                continue

            controllers[method] = {
                "ctrl": ctrl, "model": model, "world": world,
                "state_5d": state_5d.copy(), "adapt_type": adapt_type,
                "prev_state": state_5d.copy(), "prev_control": np.zeros(2),
            }
            if adapt_type == "alpaca":
                controllers[method]["base_model"] = base_model
                controllers[method]["alpaca_model"] = alpaca_model
                controllers[method]["buf_s"] = deque(maxlen=50)
                controllers[method]["buf_c"] = deque(maxlen=50)
                controllers[method]["buf_n"] = deque(maxlen=50)
                controllers[method]["adapted"] = False
                controllers[method]["ctrl_phase1"] = ctrl  # starts with base or residual

        except Exception as e:
            print(f"  SKIP {method}: {e}")

    if not controllers:
        print("  No controllers!")
        return

    data = {m: {"xy": [], "times": [], "errors": [], "min_dist": []}
            for m in controllers}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Adaptive + Safety Live — {scenario_name} "
                 f"(mismatch={'ON' if mismatch else 'OFF'})",
                 fontsize=14, fontweight="bold")

    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    for ox, oy, r in obstacles:
        ax_xy.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))
        ax_xy.add_patch(plt.Circle((ox, oy), r + 0.25,
                                    color="red", alpha=0.1, linestyle="--", fill=False))

    ref_t = np.linspace(0, TOTAL_TIME, 500)
    ref_pts = np.array([traj_fn(t) for t in ref_t])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

    lines_xy = {}
    dots = {}
    for m in controllers:
        c = METHOD_COLORS[m]
        lines_xy[m], = ax_xy.plot([], [], color=c, linewidth=1.5,
                                   label=METHOD_LABELS[m], alpha=0.8)
        dots[m], = ax_xy.plot([], [], "o", color=c, markersize=7)
    ax_xy.legend(loc="upper left", fontsize=5, ncol=2)

    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for m in controllers:
        lines_err[m], = ax_err.plot([], [], color=METHOD_COLORS[m], linewidth=1.5)

    ax_dist = axes[1, 0]
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("Min Distance (m)")
    ax_dist.set_title("Min Obstacle Distance")
    ax_dist.grid(True, alpha=0.3)
    ax_dist.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    lines_dist = {}
    for m in controllers:
        lines_dist[m], = ax_dist.plot([], [], color=METHOD_COLORS[m], linewidth=1.5)

    ax_stats = axes[1, 1]
    ax_stats.axis("off")
    ax_stats.set_title("Live Statistics")
    stats_text = ax_stats.text(0.05, 0.95, "", transform=ax_stats.transAxes,
                                fontsize=9, family="monospace", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    all_artists = list(lines_xy.values()) + list(dots.values()) + \
                  list(lines_err.values()) + list(lines_dist.values()) + [stats_text]

    def init():
        for a in all_artists:
            if hasattr(a, "set_data"):
                a.set_data([], [])
        stats_text.set_text("")
        return all_artists

    def update(frame):
        if frame >= NUM_STEPS:
            return all_artists

        t = frame * DT
        for m, ci in controllers.items():
            state_5d = ci["state_5d"]
            ref_3d = generate_reference_trajectory(traj_fn, t, MPPI_N, DT)
            ref_5d = make_5d_reference(ref_3d)

            control, _ = ci["ctrl"].compute_control(state_5d, ref_5d)

            ci["world"].step(control, DT, add_noise=True)
            next_5d = ci["world"].get_full_state()

            # Adaptation
            if ci["adapt_type"] == "ekf" and frame > 0:
                ci["model"].update_step(ci["prev_state"], ci["prev_control"], state_5d, DT)
            elif ci["adapt_type"] == "l1" and frame > 0:
                ci["model"].update_step(ci["prev_state"], ci["prev_control"], state_5d, DT)

            ci["prev_state"] = state_5d.copy()
            ci["prev_control"] = control.copy()
            ci["state_5d"] = next_5d

            ref_pt = ref_3d[0, :2]
            pos = next_5d[:2]
            data[m]["xy"].append(pos.copy())
            data[m]["times"].append(t)
            data[m]["errors"].append(float(np.linalg.norm(pos - ref_pt)))
            min_d = float("inf")
            for ox, oy, r in obstacles:
                d = np.sqrt((pos[0]-ox)**2 + (pos[1]-oy)**2) - r - 0.25
                min_d = min(min_d, d)
            data[m]["min_dist"].append(min_d)

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

        lines = [f"t = {t:.1f}s / {TOTAL_TIME:.1f}s\n"]
        lines.append(f"{'Method':<24s} {'RMSE':>7s} {'MinD':>7s} {'Col':>4s}")
        lines.append("-" * 46)
        for m in controllers:
            errs = data[m]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs)**2)) if errs else 0
            min_d = min(data[m]["min_dist"]) if data[m]["min_dist"] else 0
            col = sum(1 for d in data[m]["min_dist"] if d < 0)
            lines.append(f"{METHOD_LABELS[m]:<24s} {rmse:>7.3f} {min_d:>7.3f} {col:>4d}")
        stats_text.set_text("\n".join(lines))

        return all_artists

    anim = FuncAnimation(fig, update, init_func=init,
                          frames=NUM_STEPS, interval=1, blit=False, repeat=False)
    plt.show()

    print(f"\n{'=' * 60}")
    print("  Live Complete")
    print(f"{'=' * 60}")
    for m in controllers:
        errs = data[m]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs)**2)) if errs else 0
        min_d = min(data[m]["min_dist"]) if data[m]["min_dist"] else 0
        col = sum(1 for d in data[m]["min_dist"] if d < 0)
        print(f"  {METHOD_LABELS[m]:<28s} RMSE={rmse:.3f}m  MinD={min_d:.3f}m  Col={col}")
    print(f"{'=' * 60}\n")


# ============================================================================
#  Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Dynamics + Safety Obstacle Avoidance Benchmark"
    )
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["circle_obstacle", "gauntlet"])
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated: nominal,ekf_shield,l1_shield,...")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--live", action="store_true", help="Live animation")
    parser.add_argument("--no-mismatch", action="store_true",
                        help="Disable model mismatch (perfect model)")
    return parser.parse_args()


def main():
    args = parse_args()

    scenarios = [args.scenario] if args.scenario else ["circle_obstacle", "gauntlet"]

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        for m in methods:
            if m not in ALL_METHODS:
                print(f"ERROR: Unknown method '{m}'. Available: {ALL_METHODS}")
                sys.exit(1)
    else:
        methods = list(ALL_METHODS)

    mismatch = not args.no_mismatch

    print("=" * 70)
    print("  Adaptive Dynamics + Safety Benchmark")
    print("=" * 70)
    print(f"  Scenarios: {scenarios}")
    print(f"  Methods:   {len(methods)} selected")
    print(f"  Mismatch:  {'ON (c_v=0.5 vs nom=0.1)' if mismatch else 'OFF'}")
    print(f"  MPPI: K={MPPI_K}, N={MPPI_N}, dt={DT}")

    if args.live:
        run_live(scenarios[0], methods, mismatch=mismatch)
        return

    all_results = {}
    all_trajectories = {}
    all_obstacles = {}

    for scenario in scenarios:
        results, trajectories, obstacles = run_benchmark(scenario, methods, mismatch)
        all_results[scenario] = results
        all_trajectories[scenario] = trajectories
        all_obstacles[scenario] = obstacles
        print_results_table(scenario, results)

    if len(scenarios) > 1:
        print(f"\n{'=' * 92}")
        print(f"  Cross-Scenario Summary")
        print(f"{'=' * 92}")
        header = (f"  {'Method':<28s} {'Col':>5s} {'Safe%':>6s} "
                  f"{'MinD':>7s} {'RMSE':>7s} {'ms':>7s}")
        print(header)
        print(f"  {'-' * 65}")

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
            col_mark = " **" if avg_col > 0 else ""
            print(f"  {label:<28s} {avg_col:>5.1f}{col_mark:<3s}"
                  f"{avg_sr:>5.1f}% {avg_md:>7.3f} {avg_rmse:>7.3f} {avg_ms:>7.1f}")
        print(f"  {'-' * 65}")

    if not args.no_plot:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../plots/adaptive_safety_benchmark.png"
        )
        save_path = os.path.abspath(save_path)
        plot_results(all_results, all_trajectories, all_obstacles, scenarios, save_path)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
