#!/usr/bin/env python3
"""
Safety-Critical Control Comprehensive Benchmark Demo (14 Methods)

Compares 14 safety-critical control methods across 4 scenarios:
  dense_static, dynamic_bounce, narrow_dynamic, mixed

8 Existing methods:
  1. Vanilla MPPI (no safety — baseline)
  2. CBF-MPPI (ControlBarrierCost + optional QP filter)
  3. Shield-MPPI (per-step CBF enforcement in rollout)
  4. C3BF (Collision Cone CBF — velocity-aware cost)
  5. DPCBF (Dynamic Parabolic CBF — LoS adaptive boundary)
  6. Optimal-Decay CBF (relaxable safety filter)
  7. Gatekeeper (backup trajectory-based infinite-time safety)
  8. Backup CBF Filter (multi-step sensitivity chain QP)

6 New methods:
  9. Horizon-Weighted CBF (time-discounted CBF cost)
 10. Hard CBF (binary rejection cost)
 11. MPS (Model Predictive Shield — stateless gatekeeper)
 12. Adaptive Shield-MPPI (distance/velocity-adaptive alpha)
 13. CBF-Guided Sampling (gradient-biased rejection resampling)
 14. Shield-SVG-MPPI (Shield + SVG-MPPI guide particles)

Usage:
    python safety_novel_benchmark_demo.py
    python safety_novel_benchmark_demo.py --scenario dense_static
    python safety_novel_benchmark_demo.py --scenario mixed --methods 1,3,12,14
    python safety_novel_benchmark_demo.py --no-plot
"""

import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    CBFMPPIParams,
    ShieldMPPIParams,
    SVGMPPIParams,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.c3bf_cost import CollisionConeCBFCost
from mppi_controller.controllers.mppi.dpcbf_cost import DynamicParabolicCBFCost
from mppi_controller.controllers.mppi.optimal_decay_cbf_filter import (
    OptimalDecayCBFSafetyFilter,
)
from mppi_controller.controllers.mppi.gatekeeper import Gatekeeper
from mppi_controller.controllers.mppi.backup_controller import BrakeBackupController
from mppi_controller.controllers.mppi.backup_cbf_filter import BackupCBFSafetyFilter
from mppi_controller.controllers.mppi.horizon_cbf_cost import HorizonWeightedCBFCost
from mppi_controller.controllers.mppi.hard_cbf_cost import HardCBFCost
from mppi_controller.controllers.mppi.mps_controller import MPSController
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


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Dynamic obstacle definitions: (x0, y0, r, vx, vy, x_min, x_max, y_min, y_max)
# Bouncing happens within bounding box [x_min..x_max, y_min..y_max]

def get_scenario(name):
    """Return (static_obstacles, dynamic_defs, initial_state, goal, duration).

    static_obstacles: [(x, y, r), ...]
    dynamic_defs:     [(x0, y0, r, vx, vy, x_lo, x_hi, y_lo, y_hi), ...]
    """
    if name == "dense_static":
        # 8 static obstacles in a field the robot must navigate through
        static = [
            (1.5, 0.6, 0.25),
            (2.5, -0.4, 0.3),
            (3.5, 0.3, 0.25),
            (4.5, -0.5, 0.2),
            (5.0, 0.7, 0.3),
            (6.0, -0.2, 0.25),
            (7.0, 0.5, 0.2),
            (7.5, -0.6, 0.25),
        ]
        dynamic = []
        x0 = np.array([0.0, 0.0, 0.0])
        goal = np.array([9.0, 0.0, 0.0])
        return static, dynamic, x0, goal, 14.0

    elif name == "dynamic_bounce":
        # 3 bouncing dynamic obstacles
        static = []
        dynamic = [
            (3.0, 1.5, 0.3, 0.1, -0.4, 2.0, 4.0, -2.0, 2.0),
            (5.0, -1.0, 0.25, -0.05, 0.35, 4.0, 6.0, -2.0, 2.0),
            (7.0, 0.5, 0.3, -0.1, -0.3, 6.0, 8.0, -2.0, 2.0),
        ]
        x0 = np.array([0.0, 0.0, 0.0])
        goal = np.array([9.0, 0.0, 0.0])
        return static, dynamic, x0, goal, 14.0

    elif name == "narrow_dynamic":
        # Narrow corridor walls + 1 dynamic obstacle blocking the passage
        static = [
            (3.0, 0.9, 0.3),
            (3.0, -0.9, 0.3),
            (5.0, 0.85, 0.25),
            (5.0, -0.85, 0.25),
            (7.0, 0.9, 0.3),
            (7.0, -0.9, 0.3),
        ]
        dynamic = [
            (5.0, 0.0, 0.2, 0.0, 0.25, 4.0, 6.0, -0.6, 0.6),
        ]
        x0 = np.array([0.0, 0.0, 0.0])
        goal = np.array([9.0, 0.0, 0.0])
        return static, dynamic, x0, goal, 16.0

    elif name == "mixed":
        # 4 static + 2 dynamic
        static = [
            (2.0, 0.5, 0.3),
            (4.0, -0.5, 0.25),
            (6.0, 0.6, 0.3),
            (8.0, -0.3, 0.25),
        ]
        dynamic = [
            (3.0, 1.2, 0.25, 0.1, -0.3, 2.0, 5.0, -1.5, 1.5),
            (7.0, -1.0, 0.3, -0.08, 0.25, 5.5, 8.0, -1.5, 1.5),
        ]
        x0 = np.array([0.0, 0.0, 0.0])
        goal = np.array([9.0, 0.0, 0.0])
        return static, dynamic, x0, goal, 14.0

    else:
        raise ValueError(f"Unknown scenario: {name}")


def compute_dynamic_positions(t, dynamic_defs):
    """Compute bouncing dynamic obstacle positions at time t.

    Returns list of (x, y, r) tuples.
    """
    result = []
    for x0, y0, r, vx, vy, xlo, xhi, ylo, yhi in dynamic_defs:
        # Bounce in x
        if abs(vx) > 1e-10:
            period_x = 2.0 * (xhi - xlo) / abs(vx)
            phase_x = (t * abs(vx)) % (2.0 * (xhi - xlo))
            if phase_x <= (xhi - xlo):
                x = xlo + phase_x if vx > 0 else xhi - phase_x
            else:
                x = xhi - (phase_x - (xhi - xlo)) if vx > 0 else xlo + (phase_x - (xhi - xlo))
        else:
            x = x0

        # Bounce in y
        if abs(vy) > 1e-10:
            period_y = 2.0 * (yhi - ylo) / abs(vy)
            phase_y = (t * abs(vy)) % (2.0 * (yhi - ylo))
            if phase_y <= (yhi - ylo):
                y = ylo + phase_y if vy > 0 else yhi - phase_y
            else:
                y = yhi - (phase_y - (yhi - ylo)) if vy > 0 else ylo + (phase_y - (yhi - ylo))
        else:
            y = y0

        result.append((x, y, r))
    return result


def get_all_obstacles_at(t, static_obs, dynamic_defs):
    """Return combined obstacle list at time t as [(x, y, r), ...]."""
    dynamic = compute_dynamic_positions(t, dynamic_defs)
    return static_obs + dynamic


def get_all_obstacles_5d_at(t, static_obs, dynamic_defs, dt_est=0.05):
    """Return combined obstacle list at time t as [(x, y, r, vx, vy), ...].

    For static obstacles, vx=vy=0. For dynamic, estimate velocity from dt_est.
    """
    result = [(ox, oy, r, 0.0, 0.0) for (ox, oy, r) in static_obs]
    dyn_now = compute_dynamic_positions(t, dynamic_defs)
    dyn_next = compute_dynamic_positions(t + dt_est, dynamic_defs)
    for i in range(len(dyn_now)):
        x, y, r = dyn_now[i]
        xn, yn, _ = dyn_next[i]
        vx = (xn - x) / dt_est
        vy = (yn - y) / dt_est
        result.append((x, y, r, vx, vy))
    return result


# ---------------------------------------------------------------------------
# Reference trajectory generator
# ---------------------------------------------------------------------------

def make_reference(goal, N, dt, speed=0.5):
    """Generate straight-line reference trajectory toward goal at given speed."""
    goal_dist = max(np.linalg.norm(goal[:2]), 1e-6)
    heading = np.arctan2(goal[1], goal[0])

    def ref_fn(t):
        ref = np.zeros((N + 1, 3))
        for i in range(N + 1):
            ti = t + i * dt
            progress = min(ti * speed / goal_dist, 1.0)
            ref[i, :2] = progress * goal[:2]
            ref[i, 2] = heading
        return ref
    return ref_fn


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

# Shared parameters
DT = 0.05
N_HORIZON = 15
K_SAMPLES = 256
BASE_Q = np.array([10.0, 10.0, 1.0])
BASE_R = np.array([0.1, 0.1])
BASE_SIGMA = np.array([0.5, 0.5])

ALL_METHOD_NAMES = [
    "Vanilla",           # 1
    "CBF-MPPI",          # 2
    "Shield-MPPI",       # 3
    "C3BF",              # 4
    "DPCBF",             # 5
    "OptimalDecay",      # 6
    "Gatekeeper",        # 7
    "BackupCBF",         # 8
    "HorizonCBF",        # 9
    "HardCBF",           # 10
    "MPS",               # 11
    "AdaptiveShield",    # 12
    "CBFGuidedSamp",     # 13
    "ShieldSVG",         # 14
]

ALL_METHOD_COLORS = [
    "#555555",  # 1  Vanilla: gray
    "#1f77b4",  # 2  CBF-MPPI: blue
    "#ff7f0e",  # 3  Shield: orange
    "#2ca02c",  # 4  C3BF: green
    "#d62728",  # 5  DPCBF: red
    "#9467bd",  # 6  OptimalDecay: purple
    "#8c564b",  # 7  Gatekeeper: brown
    "#e377c2",  # 8  BackupCBF: pink
    "#17becf",  # 9  HorizonCBF: cyan
    "#bcbd22",  # 10 HardCBF: olive
    "#ff9896",  # 11 MPS: light red
    "#aec7e8",  # 12 AdaptiveShield: light blue
    "#98df8a",  # 13 CBFGuidedSamp: light green
    "#c5b0d5",  # 14 ShieldSVG: light purple
]


def make_model():
    """Create a fresh DifferentialDriveKinematic model."""
    return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)


def _base_cost(obstacles_3d):
    """Create base composite cost with obstacle avoidance."""
    cost_fns = [
        StateTrackingCost(BASE_Q),
        TerminalCost(BASE_Q),
        ControlEffortCost(BASE_R),
    ]
    if obstacles_3d:
        cost_fns.append(ObstacleCost(obstacles_3d, safety_margin=0.1, cost_weight=100.0))
    return CompositeMPPICost(cost_fns)


def create_method(method_idx, obstacles_3d, obstacles_5d):
    """Create a single method controller.

    Returns dict with keys: name, color, controller, model, post_filter, post_filter_type
    post_filter: None, Gatekeeper, MPSController, BackupCBFSafetyFilter, or
                 OptimalDecayCBFSafetyFilter
    post_filter_type: None, 'gatekeeper', 'mps', 'backup_cbf', 'optimal_decay'
    """
    idx = method_idx  # 1-based
    model = make_model()
    result = {
        "name": ALL_METHOD_NAMES[idx - 1],
        "color": ALL_METHOD_COLORS[idx - 1],
        "model": model,
        "post_filter": None,
        "post_filter_type": None,
    }

    # 1. Vanilla MPPI (no safety)
    if idx == 1:
        params = MPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
        )
        cost = _base_cost(obstacles_3d)
        result["controller"] = MPPIController(model, params, cost)

    # 2. CBF-MPPI
    elif idx == 2:
        params = CBFMPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
            cbf_obstacles=obstacles_3d, cbf_weight=1000.0, cbf_alpha=0.15,
            cbf_safety_margin=0.1, cbf_use_safety_filter=False,
        )
        result["controller"] = CBFMPPIController(model, params)

    # 3. Shield-MPPI
    elif idx == 3:
        params = ShieldMPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
            cbf_obstacles=obstacles_3d, cbf_weight=500.0, cbf_alpha=0.15,
            cbf_safety_margin=0.1, cbf_use_safety_filter=False,
            shield_enabled=True, shield_cbf_alpha=0.3,
        )
        result["controller"] = ShieldMPPIController(model, params)

    # 4. C3BF (Collision Cone CBF)
    elif idx == 4:
        c3bf_cost = CollisionConeCBFCost(
            obstacles=obstacles_5d, cbf_weight=1000.0, safety_margin=0.1, dt=DT,
        )
        params = MPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
        )
        cost = CompositeMPPICost([
            StateTrackingCost(BASE_Q), TerminalCost(BASE_Q),
            ControlEffortCost(BASE_R), c3bf_cost,
        ])
        result["controller"] = MPPIController(model, params, cost)

    # 5. DPCBF (Dynamic Parabolic CBF)
    elif idx == 5:
        dpcbf_cost = DynamicParabolicCBFCost(
            obstacles=obstacles_5d, cbf_weight=1000.0,
            safety_margin=0.1, a_base=0.3, a_vel=0.5, dt=DT,
        )
        params = MPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
        )
        cost = CompositeMPPICost([
            StateTrackingCost(BASE_Q), TerminalCost(BASE_Q),
            ControlEffortCost(BASE_R), dpcbf_cost,
        ])
        result["controller"] = MPPIController(model, params, cost)

    # 6. Optimal-Decay CBF
    elif idx == 6:
        params = CBFMPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
            cbf_obstacles=obstacles_3d, cbf_weight=1000.0, cbf_alpha=0.1,
            cbf_safety_margin=0.1, cbf_use_safety_filter=True,
        )
        ctrl = CBFMPPIController(model, params)
        ctrl.safety_filter = OptimalDecayCBFSafetyFilter(
            obstacles=obstacles_3d, cbf_alpha=0.1, safety_margin=0.1,
            penalty_weight=1e4,
        )
        result["controller"] = ctrl

    # 7. Gatekeeper
    elif idx == 7:
        params = CBFMPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
            cbf_obstacles=obstacles_3d, cbf_weight=1000.0, cbf_alpha=0.15,
            cbf_safety_margin=0.1, cbf_use_safety_filter=False,
        )
        gk_model = make_model()
        gatekeeper = Gatekeeper(
            backup_controller=BrakeBackupController(),
            model=gk_model, obstacles=obstacles_3d,
            safety_margin=0.15, backup_horizon=20, dt=DT,
        )
        result["controller"] = CBFMPPIController(model, params)
        result["post_filter"] = gatekeeper
        result["post_filter_type"] = "gatekeeper"

    # 8. Backup CBF Filter
    elif idx == 8:
        params = CBFMPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
            cbf_obstacles=obstacles_3d, cbf_weight=500.0, cbf_alpha=0.15,
            cbf_safety_margin=0.1, cbf_use_safety_filter=False,
        )
        backup_filter = BackupCBFSafetyFilter(
            backup_controller=BrakeBackupController(),
            model=make_model(), obstacles=obstacles_3d,
            dt=DT, backup_horizon=10, cbf_alpha=0.3,
            safety_margin=0.1, decay_rate=0.95,
        )
        result["controller"] = CBFMPPIController(model, params)
        result["post_filter"] = backup_filter
        result["post_filter_type"] = "backup_cbf"

    # 9. Horizon-Weighted CBF
    elif idx == 9:
        hw_cost = HorizonWeightedCBFCost(
            obstacles=obstacles_3d, weight=500.0,
            cbf_alpha=0.3, discount_gamma=0.85, safety_margin=0.05,
        )
        params = MPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
        )
        cost = CompositeMPPICost([
            StateTrackingCost(BASE_Q), TerminalCost(BASE_Q),
            ControlEffortCost(BASE_R), hw_cost,
        ])
        result["controller"] = MPPIController(model, params, cost)

    # 10. Hard CBF
    elif idx == 10:
        hard_cost = HardCBFCost(
            obstacles=obstacles_3d, rejection_cost=1e6, safety_margin=0.05,
        )
        params = MPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
        )
        cost = CompositeMPPICost([
            StateTrackingCost(BASE_Q), TerminalCost(BASE_Q),
            ControlEffortCost(BASE_R), hard_cost,
        ])
        result["controller"] = MPPIController(model, params, cost)

    # 11. MPS (Model Predictive Shield)
    elif idx == 11:
        params = MPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
        )
        cost = _base_cost(obstacles_3d)
        mps = MPSController(
            backup_controller=BrakeBackupController(),
            obstacles=obstacles_3d,
            safety_margin=0.15, backup_horizon=20, dt=DT,
        )
        result["controller"] = MPPIController(model, params, cost)
        result["post_filter"] = mps
        result["post_filter_type"] = "mps"

    # 12. Adaptive Shield-MPPI
    elif idx == 12:
        params = AdaptiveShieldParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
            cbf_obstacles=obstacles_3d, cbf_weight=500.0, cbf_alpha=0.15,
            cbf_safety_margin=0.1, cbf_use_safety_filter=False,
            shield_enabled=True, shield_cbf_alpha=0.3,
            alpha_base=0.2, alpha_dist=0.5, alpha_vel=0.15,
            k_dist=2.0, d_safe=0.4,
        )
        result["controller"] = AdaptiveShieldMPPIController(model, params)

    # 13. CBF-Guided Sampling
    elif idx == 13:
        params = CBFGuidedSamplingParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
            cbf_obstacles=obstacles_3d, cbf_weight=1000.0, cbf_alpha=0.15,
            cbf_safety_margin=0.1, cbf_use_safety_filter=False,
            rejection_ratio=0.3, gradient_bias_weight=0.1,
            max_resample_iters=2,
        )
        result["controller"] = CBFGuidedSamplingMPPIController(model, params)

    # 14. Shield-SVG-MPPI
    elif idx == 14:
        params = ShieldSVGMPPIParams(
            N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=1.0,
            sigma=BASE_SIGMA, Q=BASE_Q, R=BASE_R,
            svgd_num_iterations=3, svgd_step_size=0.01,
            svg_num_guide_particles=8, svg_guide_step_size=0.01,
            shield_enabled=True, shield_cbf_alpha=0.3,
            cbf_obstacles=obstacles_3d, cbf_safety_margin=0.1,
        )
        cost = _base_cost(obstacles_3d)
        # SVGMPPIController (and its subclass ShieldSVGMPPIController) take
        # (model, params) — cost is set via the base MPPIController path.
        # We set cost_function manually after construction.
        ctrl = ShieldSVGMPPIController(model, params)
        ctrl.cost_function = cost
        result["controller"] = ctrl

    else:
        raise ValueError(f"Unknown method index: {idx}")

    return result


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_single_method(method_dict, static_obs, dynamic_defs, initial_state,
                      goal, duration, ref_fn):
    """Run one method through the scenario and collect metrics.

    Returns dict with: name, states, collision_count, min_obstacle_dist,
    safety_rate, rmse, solve_time_ms
    """
    controller = method_dict["controller"]
    model = method_dict["model"]
    post_filter = method_dict["post_filter"]
    pf_type = method_dict["post_filter_type"]

    n_steps = int(duration / DT)
    state = initial_state.copy()
    states = [state.copy()]
    solve_times = []
    min_dists = []
    collision_count = 0

    controller.reset()
    if post_filter is not None and hasattr(post_filter, "reset"):
        post_filter.reset()

    for step in range(n_steps):
        t = step * DT

        # Current obstacles
        obs_3d = get_all_obstacles_at(t, static_obs, dynamic_defs)
        obs_5d = get_all_obstacles_5d_at(t, static_obs, dynamic_defs, dt_est=DT)

        # Update dynamic obstacles in controller/filter
        _update_obstacles(controller, obs_3d, obs_5d)
        if post_filter is not None:
            if hasattr(post_filter, "update_obstacles"):
                post_filter.update_obstacles(obs_3d)

        # Reference trajectory
        ref = ref_fn(t)

        # Compute control
        t0 = time.time()
        control, info = controller.compute_control(state, ref)

        # Apply post-filter
        if pf_type == "gatekeeper":
            control, _ = post_filter.filter(state, control)
        elif pf_type == "mps":
            control, _ = post_filter.shield(state, control, model)
        elif pf_type == "backup_cbf":
            u_min = np.array([-model.v_max, -model.omega_max])
            u_max = np.array([model.v_max, model.omega_max])
            control, _ = post_filter.filter_control(state, control, u_min, u_max)
        elif pf_type == "optimal_decay":
            # Already integrated inside CBFMPPIController via safety_filter
            pass

        solve_times.append(time.time() - t0)

        # Step dynamics
        state = model.step(state, control, DT)
        states.append(state.copy())

        # Min obstacle distance
        dists = []
        for ox, oy, r in obs_3d:
            d = np.sqrt((state[0] - ox)**2 + (state[1] - oy)**2) - r
            dists.append(d)
        if dists:
            md = min(dists)
            min_dists.append(md)
            if md < 0:
                collision_count += 1

    states_arr = np.array(states)

    # RMSE: distance from goal
    ref_final = ref_fn(duration)  # final reference
    n_total = len(states_arr)
    tracking_errors = []
    for i in range(n_total):
        t_i = min(i * DT, duration)
        ref_i = ref_fn(t_i)
        tracking_errors.append(np.linalg.norm(states_arr[i, :2] - ref_i[0, :2]))
    rmse = np.sqrt(np.mean(np.array(tracking_errors)**2))

    # Safety rate (fraction of steps without collision)
    safety_rate = 1.0 - collision_count / max(n_steps, 1)

    return {
        "name": method_dict["name"],
        "color": method_dict["color"],
        "states": states_arr,
        "collision_count": collision_count,
        "min_obstacle_dist": min(min_dists) if min_dists else float("inf"),
        "safety_rate": safety_rate,
        "rmse": rmse,
        "solve_time_ms": np.mean(solve_times) * 1000,
    }


def _update_obstacles(controller, obs_3d, obs_5d):
    """Update obstacles inside a controller, handling different types."""
    # CBF-based controllers (CBFMPPIController, ShieldMPPI, AdaptiveShield, CBFGuidedSampling)
    if hasattr(controller, "update_obstacles"):
        controller.update_obstacles(obs_3d)

    # Shield-SVG-MPPI uses a different attribute
    if hasattr(controller, "shield_svg_params") and hasattr(controller.shield_svg_params, "cbf_obstacles"):
        controller.shield_svg_params.cbf_obstacles = obs_3d

    # Cost-based methods: update obstacle costs inside CompositeMPPICost
    if hasattr(controller, "cost_function"):
        cf = controller.cost_function
        if isinstance(cf, CompositeMPPICost):
            for sub_cost in cf.cost_functions:
                if hasattr(sub_cost, "update_obstacles"):
                    if isinstance(sub_cost, (CollisionConeCBFCost, DynamicParabolicCBFCost)):
                        sub_cost.update_obstacles(obs_5d)
                    else:
                        sub_cost.update_obstacles(obs_3d)
                # ObstacleCost has obstacles attribute directly
                if isinstance(sub_cost, ObstacleCost):
                    sub_cost.obstacles = obs_3d


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(scenario_name, method_indices, plot=True):
    """Run the full benchmark and print results table."""
    static_obs, dynamic_defs, initial_state, goal, duration = get_scenario(scenario_name)

    # Initial obstacle snapshot for controller creation
    obs_3d_init = get_all_obstacles_at(0.0, static_obs, dynamic_defs)
    obs_5d_init = get_all_obstacles_5d_at(0.0, static_obs, dynamic_defs)

    ref_fn = make_reference(goal, N_HORIZON, DT, speed=0.5)

    print()
    print("=" * 78)
    print("  Safety-Critical Control Benchmark (14 Methods)".center(78))
    print("=" * 78)
    print(f"  Scenario: {scenario_name}")
    print(f"  Methods:  {len(method_indices)} selected")
    print(f"  K={K_SAMPLES}, N={N_HORIZON}, dt={DT}s, duration={duration}s")
    print(f"  Static obstacles: {len(static_obs)}, Dynamic obstacles: {len(dynamic_defs)}")
    print("=" * 78)

    results = []
    for i, midx in enumerate(method_indices):
        method = create_method(midx, obs_3d_init, obs_5d_init)
        print(f"  [{i+1}/{len(method_indices)}] Running {method['name']}...", end="", flush=True)

        res = run_single_method(
            method, static_obs, dynamic_defs, initial_state, goal, duration, ref_fn,
        )
        print(f"  done ({res['solve_time_ms']:.1f}ms/step)")
        results.append(res)

    # ── Results table ──
    print()
    print("=" * 78)
    print("  Results".center(78))
    print("=" * 78)
    header = (f"{'#':>2s} {'Method':>16s} | {'Collisions':>10s} | {'MinDist':>8s} | "
              f"{'SafeRate':>8s} | {'RMSE':>7s} | {'ms/step':>8s}")
    print(header)
    print("-" * 78)

    for i, r in enumerate(results):
        coll_str = f"{r['collision_count']:>10d}"
        if r["collision_count"] > 0:
            coll_str = f"{'!' + str(r['collision_count']):>10s}"
        print(
            f"{method_indices[i]:>2d} {r['name']:>16s} | {coll_str} | "
            f"{r['min_obstacle_dist']:>8.3f} | "
            f"{r['safety_rate']:>8.1%} | "
            f"{r['rmse']:>7.3f} | "
            f"{r['solve_time_ms']:>8.1f}"
        )

    print("=" * 78)

    # ── Plot ──
    if plot:
        _plot_results(results, scenario_name, static_obs, dynamic_defs, goal, duration)

    return results


def _plot_results(results, scenario_name, static_obs, dynamic_defs, goal, duration):
    """Generate 6-panel figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot.")
        return

    n_methods = len(results)

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Safety-Critical Control Benchmark -- {scenario_name}\n"
        f"K={K_SAMPLES}, N={N_HORIZON}, dt={DT}s, {n_methods} methods",
        fontsize=14, fontweight="bold",
    )

    # Layout: 2 rows x 3 cols
    # Panel 1: All trajectories overlaid
    # Panel 2: Min obstacle distance over time
    # Panel 3: Collision count bar chart
    # Panel 4: Safety rate bar chart
    # Panel 5: RMSE bar chart
    # Panel 6: Solve time bar chart

    # --- Panel 1: Trajectory overlay ---
    ax1 = fig.add_subplot(2, 3, 1)
    # Draw obstacles at t=0 and t=duration/2
    obs_t0 = get_all_obstacles_at(0.0, static_obs, dynamic_defs)
    obs_tmid = get_all_obstacles_at(duration / 2, static_obs, dynamic_defs)

    for obs in obs_t0:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color="red", alpha=0.25)
        ax1.add_patch(circle)
    # Draw dynamic obstacles at midpoint (ghosted)
    for obs in obs_tmid:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color="red",
                            alpha=0.1, linestyle="--", fill=False)
        ax1.add_patch(circle)

    for r in results:
        ax1.plot(r["states"][:, 0], r["states"][:, 1],
                 color=r["color"], linewidth=1.5, alpha=0.8, label=r["name"])
        ax1.plot(r["states"][0, 0], r["states"][0, 1], "o",
                 color=r["color"], markersize=5)
        ax1.plot(r["states"][-1, 0], r["states"][-1, 1], "s",
                 color=r["color"], markersize=4)

    ax1.plot(goal[0], goal[1], "r*", markersize=15, zorder=5)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Trajectories (all methods)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=5, loc="upper left", ncol=2)
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-2.5, 2.5)

    # --- Panel 2: Min obstacle distance over time ---
    ax2 = fig.add_subplot(2, 3, 2)
    for r in results:
        n_pts = len(r["states"])
        times = np.arange(n_pts) * DT
        min_dists_over_time = []
        for i in range(n_pts):
            t_i = times[i]
            obs_at_t = get_all_obstacles_at(t_i, static_obs, dynamic_defs)
            dists = [np.sqrt((r["states"][i, 0] - ox)**2 + (r["states"][i, 1] - oy)**2) - rad
                     for ox, oy, rad in obs_at_t]
            min_dists_over_time.append(min(dists) if dists else float("inf"))
        ax2.plot(times, min_dists_over_time, color=r["color"], linewidth=1.0,
                 alpha=0.8, label=r["name"])
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Min Distance to Obstacle (m)")
    ax2.set_title("Obstacle Clearance Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=5, loc="lower right", ncol=2)

    # --- Panel 3: Collision count ---
    ax3 = fig.add_subplot(2, 3, 3)
    names = [r["name"] for r in results]
    collisions = [r["collision_count"] for r in results]
    colors = [r["color"] for r in results]
    bars = ax3.barh(names, collisions, color=colors, alpha=0.7)
    ax3.set_xlabel("Collision Count (timesteps)")
    ax3.set_title("Collisions")
    ax3.grid(axis="x", alpha=0.3)
    # Annotate bars with values
    for bar, val in zip(bars, collisions):
        if val > 0:
            ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=7, color="red", fontweight="bold")

    # --- Panel 4: Safety rate ---
    ax4 = fig.add_subplot(2, 3, 4)
    safety_rates = [r["safety_rate"] * 100 for r in results]
    bars4 = ax4.barh(names, safety_rates, color=colors, alpha=0.7)
    ax4.set_xlabel("Safety Rate (%)")
    ax4.set_title("Safety Rate (collision-free steps)")
    ax4.set_xlim(90, 101)
    ax4.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars4, safety_rates):
        ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=7)

    # --- Panel 5: RMSE ---
    ax5 = fig.add_subplot(2, 3, 5)
    rmses = [r["rmse"] for r in results]
    bars5 = ax5.barh(names, rmses, color=colors, alpha=0.7)
    ax5.set_xlabel("Tracking RMSE (m)")
    ax5.set_title("Tracking Accuracy (RMSE)")
    ax5.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars5, rmses):
        ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=7)

    # --- Panel 6: Solve time ---
    ax6 = fig.add_subplot(2, 3, 6)
    solve_times = [r["solve_time_ms"] for r in results]
    bars6 = ax6.barh(names, solve_times, color=colors, alpha=0.7)
    ax6.set_xlabel("Mean Solve Time (ms/step)")
    ax6.set_title("Computational Cost")
    ax6.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars6, solve_times):
        ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = os.path.join(os.path.dirname(__file__), "../../results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"safety_novel_benchmark_{scenario_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Safety-Critical Control Comprehensive Benchmark (14 Methods)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  dense_static   -- 8 static obstacles in a field
  dynamic_bounce  -- 3 bouncing dynamic obstacles
  narrow_dynamic  -- narrow corridor + 1 dynamic obstacle
  mixed           -- 4 static + 2 dynamic obstacles

Methods (1-14):
   1: Vanilla          2: CBF-MPPI        3: Shield-MPPI
   4: C3BF             5: DPCBF           6: OptimalDecay
   7: Gatekeeper       8: BackupCBF       9: HorizonCBF
  10: HardCBF         11: MPS            12: AdaptiveShield
  13: CBFGuidedSamp   14: ShieldSVG

Examples:
  python safety_novel_benchmark_demo.py
  python safety_novel_benchmark_demo.py --scenario mixed --methods 1,3,12,14
  python safety_novel_benchmark_demo.py --all-scenarios
        """,
    )
    parser.add_argument(
        "--scenario", type=str, default="dense_static",
        choices=["dense_static", "dynamic_bounce", "narrow_dynamic", "mixed"],
        help="Scenario to run (default: dense_static)",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated method indices (1-14), default: all 14",
    )
    parser.add_argument(
        "--all-scenarios", action="store_true",
        help="Run all 4 scenarios sequentially",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    # Parse method indices
    if args.methods is not None:
        method_indices = [int(x.strip()) for x in args.methods.split(",")]
        for idx in method_indices:
            if idx < 1 or idx > 14:
                print(f"Error: method index {idx} out of range [1, 14]")
                sys.exit(1)
    else:
        method_indices = list(range(1, 15))

    scenarios = (
        ["dense_static", "dynamic_bounce", "narrow_dynamic", "mixed"]
        if args.all_scenarios
        else [args.scenario]
    )

    all_results = {}
    for scenario in scenarios:
        results = run_benchmark(scenario, method_indices, plot=not args.no_plot)
        all_results[scenario] = results

    # Multi-scenario summary
    if len(scenarios) > 1:
        print()
        print("=" * 78)
        print("  Multi-Scenario Summary".center(78))
        print("=" * 78)
        for scenario in scenarios:
            results = all_results[scenario]
            best_safety = max(results, key=lambda r: r["safety_rate"])
            best_rmse = min(results, key=lambda r: r["rmse"])
            best_speed = min(results, key=lambda r: r["solve_time_ms"])
            print(f"\n  [{scenario}]")
            print(f"    Best safety:  {best_safety['name']} "
                  f"({best_safety['safety_rate']:.1%}, "
                  f"{best_safety['collision_count']} collisions)")
            print(f"    Best RMSE:    {best_rmse['name']} ({best_rmse['rmse']:.3f}m)")
            print(f"    Fastest:      {best_speed['name']} ({best_speed['solve_time_ms']:.1f}ms)")
        print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
