#!/usr/bin/env python3
"""
DBaS-MPPI (Discrete Barrier States MPPI) 벤치마크: 4-Way x 4 시나리오

방법:
  1. Vanilla MPPI  — 단일 반복, 등방 노이즈 + ObstacleCost
  2. DIAL-MPPI     — 다중 반복, 고정 기하급수 어닐링 + ObstacleCost
  3. CMA-MPPI      — 다중 반복, 적응적 공분산 + ObstacleCost
  4. DBaS-MPPI     — barrier state + 적응적 탐색

시나리오 4개:
  A. dense_static     — 밀집 정적 장애물 (warehouse)
  B. dynamic_crossing  — 동적 교차 장애물
  C. narrow_passage    — 좁은 통로 + 벽 제약
  D. noisy_mismatch    — 모델 불일치 + 프로세스 노이즈

Usage:
    PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --scenario dense_static
    PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --live --scenario dynamic_crossing
    PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --no-plot
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    DIALMPPIParams,
    CMAMPPIParams,
    DBaSMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.cma_mppi import CMAMPPIController
from mppi_controller.controllers.mppi.dbas_mppi import DBaSMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
    figure_eight_trajectory,
)


# ── 공통 설정 ──────────────────────────────────────────────────

COMMON = dict(
    K=512, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)

COLORS = {
    "Vanilla MPPI": "#2196F3",
    "DIAL-MPPI": "#FF9800",
    "CMA-MPPI": "#4CAF50",
    "DBaS-MPPI": "#E91E63",
}


# ── 동적 장애물 클래스 ─────────────────────────────────────────

class DynamicObstacle:
    """동적 장애물 (교차/반사 운동)"""
    def __init__(self, x0, y0, radius, motion_type, **kwargs):
        self.x0, self.y0 = x0, y0
        self.radius = radius
        self.motion_type = motion_type
        self.kwargs = kwargs

    def position(self, t):
        if self.motion_type == "crossing":
            vx = self.kwargs.get("vx", 0.0)
            vy = self.kwargs.get("vy", 0.0)
            return self.x0 + vx * t, self.y0 + vy * t
        elif self.motion_type == "bouncing":
            vx = self.kwargs.get("vx", 0.5)
            vy = self.kwargs.get("vy", 0.3)
            x_min = self.kwargs.get("x_min", -4.0)
            x_max = self.kwargs.get("x_max", 4.0)
            y_min = self.kwargs.get("y_min", -4.0)
            y_max = self.kwargs.get("y_max", 4.0)

            x = self.x0 + vx * t
            y = self.y0 + vy * t

            # 반사
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_rel = (x - x_min) % (2 * x_range)
            y_rel = (y - y_min) % (2 * y_range)

            if x_rel > x_range:
                x_rel = 2 * x_range - x_rel
            if y_rel > y_range:
                y_rel = 2 * y_range - y_rel

            return x_min + x_rel, y_min + y_rel
        return self.x0, self.y0

    def as_3tuple(self, t):
        x, y = self.position(t)
        return (x, y, self.radius)


# ── 시나리오 정의 ──────────────────────────────────────────────

def generate_obstacle_field(seed=42, n_obs=12, x_range=(-4, 4), y_range=(-4, 4)):
    """Poisson-disk 근사로 밀집 장애물 생성"""
    rng = np.random.RandomState(seed)
    obstacles = []
    attempts = 0
    while len(obstacles) < n_obs and attempts < 500:
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        r = rng.uniform(0.3, 0.6)

        # 최소 간격 확인
        ok = True
        for ox, oy, orad in obstacles:
            if np.sqrt((x - ox) ** 2 + (y - oy) ** 2) < r + orad + 0.5:
                ok = False
                break
        if ok:
            obstacles.append((x, y, r))
        attempts += 1
    return obstacles


def slalom_trajectory(t, obstacles, speed=0.4, y_amp=2.0):
    """장애물 사이를 통과하는 slalom 궤적"""
    x = speed * t
    y = y_amp * np.sin(0.5 * t)
    theta = np.arctan2(y_amp * 0.5 * np.cos(0.5 * t), speed)
    return np.array([x, y, theta])


def waypoint_trajectory(t, waypoints, speed=0.5):
    """Waypoint 기반 직선 궤적"""
    dist = speed * t
    cumulative = 0.0
    for i in range(len(waypoints) - 1):
        wp1 = np.array(waypoints[i])
        wp2 = np.array(waypoints[i + 1])
        seg_len = np.linalg.norm(wp2 - wp1)
        if cumulative + seg_len >= dist:
            frac = (dist - cumulative) / (seg_len + 1e-8)
            pos = wp1 + frac * (wp2 - wp1)
            theta = np.arctan2(wp2[1] - wp1[1], wp2[0] - wp1[0])
            return np.array([pos[0], pos[1], theta])
        cumulative += seg_len
    # 마지막 waypoint
    last = np.array(waypoints[-1])
    if len(waypoints) >= 2:
        prev = np.array(waypoints[-2])
        theta = np.arctan2(last[1] - prev[1], last[0] - prev[0])
    else:
        theta = 0.0
    return np.array([last[0], last[1], theta])


def get_scenarios():
    """4개 벤치마크 시나리오"""
    dense_obstacles = generate_obstacle_field(seed=42, n_obs=12)

    return {
        "dense_static": {
            "name": "A. Dense Static Obstacles",
            "static_obstacles": dense_obstacles,
            "dynamic_obstacles": [],
            "walls": [],
            "trajectory_fn": lambda t: slalom_trajectory(t, dense_obstacles),
            "initial_state": np.array([-4.0, 0.0, 0.0]),
            "duration": 15.0,
            "description": "12+ random obstacles, slalom trajectory",
        },
        "dynamic_crossing": {
            "name": "B. Dynamic Crossing Obstacles",
            "static_obstacles": [],
            "dynamic_obstacles": [
                DynamicObstacle(3.0, -3.0, 0.4, "crossing", vx=-0.3, vy=0.4),
                DynamicObstacle(-3.0, 2.0, 0.35, "crossing", vx=0.35, vy=-0.2),
                DynamicObstacle(0.0, -4.0, 0.3, "bouncing", vx=0.4, vy=0.5),
                DynamicObstacle(2.0, 3.0, 0.35, "bouncing", vx=-0.3, vy=-0.4),
            ],
            "walls": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "description": "4 dynamic obstacles (2 crossing + 2 bouncing), circle",
        },
        "narrow_passage": {
            "name": "C. Narrow Passage with Walls",
            "static_obstacles": [
                # 입구 병목
                (-1.0, 2.8, 0.3),
                (-1.0, 0.2, 0.3),
            ],
            "dynamic_obstacles": [],
            "walls": [
                ('y', 0.5, 1),    # y >= 0.5 (하단 벽)
                ('y', 2.5, -1),   # y <= 2.5 (상단 벽)
                ('x', -5.0, 1),   # x >= -5  (좌측 벽)
                ('x', 8.0, -1),   # x <= 8   (우측 벽)
            ],
            "trajectory_fn": lambda t: waypoint_trajectory(
                t,
                [(-4.0, 1.5), (-1.5, 1.5), (0.0, 1.5), (2.0, 1.5), (4.0, 1.5), (7.0, 1.5)],
                speed=0.5,
            ),
            "initial_state": np.array([-4.0, 1.5, 0.0]),
            "duration": 20.0,
            "description": "Narrow passage (width ~2m) + bottleneck",
        },
        "noisy_mismatch": {
            "name": "D. Model Mismatch + Process Noise",
            "static_obstacles": [
                (2.0, 2.0, 0.5),
                (-2.0, -1.5, 0.4),
                (0.0, 3.0, 0.3),
            ],
            "dynamic_obstacles": [],
            "walls": [],
            "trajectory_fn": figure_eight_trajectory,
            "initial_state": np.array([0.0, 0.0, 0.0]),
            "duration": 12.0,
            "description": "3 static obstacles + process noise + wheelbase mismatch",
            "planner_wheelbase": 0.5,
            "real_wheelbase": 0.6,
            "process_noise_std": np.array([0.05, 0.05, 0.02]),
        },
    }


# ── 컨트롤러 생성 ─────────────────────────────────────────────

def _make_cost(params, obstacles):
    """기본 비용 + 장애물 비용"""
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        costs.append(ObstacleCost(obstacles, safety_margin=0.3, cost_weight=2000.0))
    return CompositeMPPICost(costs)


def _make_controllers(model, scenario):
    """4가지 컨트롤러 생성"""
    static_obs = scenario["static_obstacles"]
    walls = scenario.get("walls", [])

    # Vanilla
    v_params = MPPIParams(**COMMON)
    vanilla = MPPIController(model, v_params, cost_function=_make_cost(v_params, static_obs))

    # DIAL
    d_params = DIALMPPIParams(
        **COMMON, n_diffuse_init=8, n_diffuse=3,
        traj_diffuse_factor=0.5, horizon_diffuse_factor=0.5,
        use_reward_normalization=True,
    )
    dial = DIALMPPIController(model, d_params, cost_function=_make_cost(d_params, static_obs))

    # CMA
    c_params = CMAMPPIParams(
        **COMMON, n_iters_init=8, n_iters=3,
        cov_learning_rate=0.5, sigma_min=0.05, sigma_max=3.0,
        use_mean_shift=True, use_reward_normalization=True,
    )
    cma = CMAMPPIController(model, c_params, cost_function=_make_cost(c_params, static_obs))

    # DBaS
    dbas_params = DBaSMPPIParams(
        **COMMON,
        dbas_obstacles=static_obs,
        dbas_walls=walls,
        barrier_weight=20.0,
        barrier_gamma=0.5,
        exploration_coeff=1.0,
        h_min=1e-6,
        safety_margin=0.15,
        use_adaptive_exploration=True,
    )
    # DBaS는 base cost만 사용 (barrier는 내부 처리)
    dbas_cost = _make_cost(dbas_params, None)  # 장애물 비용 제외 (barrier가 대체)
    dbas = DBaSMPPIController(model, dbas_params, cost_function=dbas_cost)

    return {
        "Vanilla MPPI": vanilla,
        "DIAL-MPPI": dial,
        "CMA-MPPI": cma,
        "DBaS-MPPI": dbas,
    }


# ── 시뮬레이션 ─────────────────────────────────────────────────

def run_single_simulation(model, controller, scenario, seed=42):
    """단일 컨트롤러 시뮬레이션"""
    np.random.seed(seed)

    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)
    trajectory_fn = scenario["trajectory_fn"]

    state = scenario["initial_state"].copy()
    dyn_obs = scenario.get("dynamic_obstacles", [])
    process_noise_std = scenario.get("process_noise_std", None)

    # 모델 불일치 시뮬레이션용 별도 모델
    real_wheelbase = scenario.get("real_wheelbase", None)
    if real_wheelbase is not None:
        real_model = DifferentialDriveKinematic(wheelbase=real_wheelbase)
    else:
        real_model = model

    states = [state.copy()]
    controls_hist = []
    solve_times = []
    infos = []

    for step in range(num_steps):
        t = step * dt
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt)

        # 동적 장애물 업데이트
        if dyn_obs and hasattr(controller, 'update_obstacles'):
            current_dyn_obs = [d.as_3tuple(t) for d in dyn_obs]
            all_obs = list(scenario["static_obstacles"]) + current_dyn_obs
            controller.update_obstacles(all_obs)
        elif dyn_obs:
            # Vanilla/DIAL/CMA: 동적 장애물은 비용 함수에서 처리 불가 (정적만)
            pass

        t_start = time.time()
        control, info = controller.compute_control(state, ref)
        solve_time = time.time() - t_start

        # 프로세스 노이즈
        state_dot = real_model.forward_dynamics(state, control)
        state = state + state_dot * dt
        if process_noise_std is not None:
            state = state + np.random.randn(len(state)) * process_noise_std

        states.append(state.copy())
        controls_hist.append(control.copy())
        solve_times.append(solve_time)
        infos.append(info)

    return {
        "states": np.array(states),
        "controls": np.array(controls_hist) if controls_hist else np.array([]),
        "solve_times": np.array(solve_times),
        "infos": infos,
    }


def compute_metrics(history, scenario):
    """메트릭 계산"""
    states = history["states"]
    trajectory_fn = scenario["trajectory_fn"]
    dt = COMMON["dt"]

    # RMSE
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    # 장애물 충돌
    all_obstacles = list(scenario["static_obstacles"])
    dyn_obs = scenario.get("dynamic_obstacles", [])
    n_collisions = 0
    min_clearance = float("inf")

    for i, st in enumerate(states):
        t = i * dt
        obstacles = all_obstacles + [d.as_3tuple(t) for d in dyn_obs]
        for ox, oy, r in obstacles:
            dist = np.sqrt((st[0] - ox) ** 2 + (st[1] - oy) ** 2)
            clearance = dist - r
            min_clearance = min(min_clearance, clearance)
            if clearance < 0:
                n_collisions += 1

    # 벽 위반
    walls = scenario.get("walls", [])
    wall_violations = 0
    for st in states:
        for axis_name, value, direction in walls:
            axis_idx = 0 if axis_name == 'x' else 1
            if direction * (st[axis_idx] - value) < -0.05:
                wall_violations += 1

    # ESS
    ess_list = [
        info.get("ess", 0.0) for info in history["infos"]
        if isinstance(info, dict) and "ess" in info
    ]

    # DBaS 고유
    dbas_adaptive_scales = []
    dbas_barrier_costs = []
    for info in history["infos"]:
        if isinstance(info, dict) and "dbas_stats" in info:
            dbas_adaptive_scales.append(info["dbas_stats"]["adaptive_scale"])
            dbas_barrier_costs.append(info["dbas_stats"]["barrier_cost_best"])

    return {
        "rmse": rmse,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "wall_violations": wall_violations,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "max_solve_ms": float(np.max(history["solve_times"])) * 1000,
        "ess_list": ess_list,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "dbas_adaptive_scales": dbas_adaptive_scales,
        "dbas_barrier_costs": dbas_barrier_costs,
        "errors": errors,
    }


# ── Live 애니메이션 ─────────────────────────────────────────────

def run_live(args):
    """실시간 4-way 비교 애니메이션 → GIF/MP4 저장"""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    trajectory_fn = scenario["trajectory_fn"]
    dyn_obs = scenario.get("dynamic_obstacles", [])

    wb = scenario.get("planner_wheelbase", 0.5)
    model = DifferentialDriveKinematic(wheelbase=wb)
    real_wb = scenario.get("real_wheelbase", wb)
    real_model = DifferentialDriveKinematic(wheelbase=real_wb)
    process_noise_std = scenario.get("process_noise_std", None)

    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)

    np.random.seed(args.seed)
    controllers = _make_controllers(model, scenario)

    print(f"\n{'=' * 60}")
    print(f"  DBaS-MPPI Live — {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # 상태 초기화
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "times": [], "errors": [], "ess": [],
             "barrier_cost": [], "adaptive_scale": []}
        for k in controllers
    }

    # Figure 설정 (2x4 8패널)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"DBaS-MPPI Live — {scenario['name']}",
        fontsize=14, fontweight="bold",
    )

    # [0,0] XY 궤적
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    # 정적 장애물 그리기
    for ox, oy, r in scenario["static_obstacles"]:
        ax_xy.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.3))

    # 벽 그리기
    for wall in scenario.get("walls", []):
        axis_name, val, direction = wall
        if axis_name == 'x':
            ax_xy.axvline(val, color="brown", linestyle="--", alpha=0.5)
        else:
            ax_xy.axhline(val, color="brown", linestyle="--", alpha=0.5)

    # 레퍼런스
    ref_t = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t) for t in ref_t])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1, label="Ref")

    lines_xy = {}
    dots = {}
    dyn_patches = []
    for name, color in COLORS.items():
        lines_xy[name], = ax_xy.plot([], [], color=color, linewidth=2, label=name)
        dots[name], = ax_xy.plot([], [], "o", color=color, markersize=8)
    # 동적 장애물 패치
    for d in dyn_obs:
        p = plt.Circle((d.x0, d.y0), d.radius, color="orange", alpha=0.5)
        ax_xy.add_patch(p)
        dyn_patches.append(p)
    ax_xy.legend(loc="upper left", fontsize=6)

    # [0,1] 위치 오차
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Tracking Error")
    ax_err.grid(True, alpha=0.3)
    lines_err = {}
    for name, color in COLORS.items():
        lines_err[name], = ax_err.plot([], [], color=color, linewidth=1.5, label=name)
    ax_err.legend(fontsize=6)

    # [0,2] Barrier cost (DBaS only)
    ax_bc = axes[0, 2]
    ax_bc.set_xlabel("Time (s)")
    ax_bc.set_ylabel("Barrier Cost")
    ax_bc.set_title("Barrier Cost (DBaS)")
    ax_bc.grid(True, alpha=0.3)
    line_bc, = ax_bc.plot([], [], color=COLORS["DBaS-MPPI"], linewidth=1.5)

    # [0,3] Adaptive scale (DBaS only)
    ax_as = axes[0, 3]
    ax_as.set_xlabel("Time (s)")
    ax_as.set_ylabel("Adaptive Scale")
    ax_as.set_title("Exploration Scale (DBaS)")
    ax_as.grid(True, alpha=0.3)
    line_as, = ax_as.plot([], [], color=COLORS["DBaS-MPPI"], linewidth=1.5)

    # [1,0] ESS
    ax_ess = axes[1, 0]
    ax_ess.set_xlabel("Time (s)")
    ax_ess.set_ylabel("ESS")
    ax_ess.set_title("Effective Sample Size")
    ax_ess.grid(True, alpha=0.3)
    lines_ess = {}
    for name, color in COLORS.items():
        lines_ess[name], = ax_ess.plot([], [], color=color, linewidth=1.5, label=name)
    ax_ess.legend(fontsize=6)

    # [1,1] RMSE 바 차트
    ax_rmse = axes[1, 1]
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Running RMSE")
    ax_rmse.grid(True, alpha=0.3, axis="y")
    bar_names = list(COLORS.keys())
    bar_colors = [COLORS[n] for n in bar_names]
    bars_rmse = ax_rmse.bar(range(len(bar_names)), [0] * len(bar_names),
                            color=bar_colors, alpha=0.8)
    ax_rmse.set_xticks(range(len(bar_names)))
    ax_rmse.set_xticklabels(["Van", "DIAL", "CMA", "DBaS"], fontsize=8)
    bar_rmse_texts = [
        ax_rmse.text(b.get_x() + b.get_width() / 2, 0, "", ha="center",
                     va="bottom", fontsize=8)
        for b in bars_rmse
    ]

    # [1,2] 충돌 수 바 차트
    ax_col = axes[1, 2]
    ax_col.set_ylabel("Collisions")
    ax_col.set_title("Collision Count")
    ax_col.grid(True, alpha=0.3, axis="y")
    bars_col = ax_col.bar(range(len(bar_names)), [0] * len(bar_names),
                          color=bar_colors, alpha=0.8)
    ax_col.set_xticks(range(len(bar_names)))
    ax_col.set_xticklabels(["Van", "DIAL", "CMA", "DBaS"], fontsize=8)
    collision_counts = {k: 0 for k in COLORS}

    # [1,3] 통계 텍스트
    ax_info = axes[1, 3]
    ax_info.axis("off")
    ax_info.set_title("Statistics")
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        va="top", fontsize=9, family="monospace",
    )

    plt.tight_layout()

    def update(frame):
        if frame >= num_steps:
            return

        t = sim_t[0]
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt)

        # 동적 장애물 업데이트 (시각적)
        for i, d in enumerate(dyn_obs):
            x, y = d.position(t)
            dyn_patches[i].center = (x, y)

        for name, ctrl in controllers.items():
            # 동적 장애물 업데이트 (DBaS)
            if dyn_obs and hasattr(ctrl, 'update_obstacles'):
                current_dyn = [d.as_3tuple(t) for d in dyn_obs]
                ctrl.update_obstacles(list(scenario["static_obstacles"]) + current_dyn)

            control, info = ctrl.compute_control(states[name], ref)

            state_dot = real_model.forward_dynamics(states[name], control)
            states[name] = states[name] + state_dot * dt
            if process_noise_std is not None:
                states[name] = states[name] + np.random.randn(3) * process_noise_std

            ref_pt = trajectory_fn(t)[:2]
            data[name]["xy"].append(states[name][:2].copy())
            data[name]["times"].append(t)
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["ess"].append(info.get("ess", 0.0))

            # 충돌 체크
            all_obs = list(scenario["static_obstacles"])
            if dyn_obs:
                all_obs += [d.as_3tuple(t) for d in dyn_obs]
            for ox, oy, r in all_obs:
                if np.sqrt((states[name][0] - ox) ** 2 + (states[name][1] - oy) ** 2) < r:
                    collision_counts[name] += 1

            # DBaS 고유 데이터
            if name == "DBaS-MPPI" and "dbas_stats" in info:
                data[name]["barrier_cost"].append(info["dbas_stats"]["barrier_cost_best"])
                data[name]["adaptive_scale"].append(info["dbas_stats"]["adaptive_scale"])

        sim_t[0] += dt

        # 그래프 업데이트
        times = np.array(data["Vanilla MPPI"]["times"])

        for name in controllers:
            xy = np.array(data[name]["xy"])
            if len(xy) > 0:
                lines_xy[name].set_data(xy[:, 0], xy[:, 1])
                dots[name].set_data([xy[-1, 0]], [xy[-1, 1]])
                lines_err[name].set_data(times[:len(data[name]["errors"])],
                                         data[name]["errors"])
                lines_ess[name].set_data(times[:len(data[name]["ess"])],
                                         data[name]["ess"])

        # DBaS barrier cost / adaptive scale
        dbas_data = data["DBaS-MPPI"]
        if dbas_data["barrier_cost"]:
            bc_times = times[:len(dbas_data["barrier_cost"])]
            line_bc.set_data(bc_times, dbas_data["barrier_cost"])
            ax_bc.relim()
            ax_bc.autoscale_view()
        if dbas_data["adaptive_scale"]:
            as_times = times[:len(dbas_data["adaptive_scale"])]
            line_as.set_data(as_times, dbas_data["adaptive_scale"])
            ax_as.relim()
            ax_as.autoscale_view()

        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal")
        ax_err.relim()
        ax_err.autoscale_view()
        ax_ess.relim()
        ax_ess.autoscale_view()

        # RMSE 바 차트
        rmses = []
        for i, name in enumerate(bar_names):
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            rmses.append(rmse)
            bars_rmse[i].set_height(rmse)
            bar_rmse_texts[i].set_position(
                (bars_rmse[i].get_x() + bars_rmse[i].get_width() / 2, rmse))
            bar_rmse_texts[i].set_text(f"{rmse:.3f}")
        if rmses:
            ax_rmse.set_ylim(0, max(rmses) * 1.3 + 0.01)

        # 충돌 바 차트
        for i, name in enumerate(bar_names):
            bars_col[i].set_height(collision_counts[name])
        max_col = max(collision_counts.values()) if collision_counts else 1
        ax_col.set_ylim(0, max(max_col * 1.3, 1))

        # 통계 텍스트
        lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
        for name in controllers:
            errs = data[name]["errors"]
            rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
            ess = data[name]["ess"][-1] if data[name]["ess"] else 0
            short = name.replace(" MPPI", "").replace("-MPPI", "")
            cols = collision_counts[name]
            lines.append(f"{short:>8}: RMSE={rmse:.3f} ESS={ess:.0f} Col={cols}")
        if dbas_data["adaptive_scale"]:
            lines.append(f"\nDBaS scale: {dbas_data['adaptive_scale'][-1]:.3f}")
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/dbas_mppi_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=100)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/dbas_mppi_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=100)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # 종료 통계
    print(f"\n{'=' * 72}")
    print(f"  Final Statistics — {scenario['name']}")
    print(f"{'=' * 72}")
    print(f"  {'Method':<16} {'RMSE':>8} {'Collisions':>10} {'Mean ESS':>10}")
    print(f"  {'-' * 46}")
    for name in controllers:
        errs = data[name]["errors"]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 0
        mean_ess = np.mean(data[name]["ess"]) if data[name]["ess"] else 0
        cols = collision_counts[name]
        print(f"  {name:<16} {rmse:>8.4f} {cols:>10d} {mean_ess:>10.1f}")
    print(f"{'=' * 72}\n")


# ── 벤치마크 메인 ─────────────────────────────────────────────

def run_benchmark(args):
    """정적 벤치마크 실행 + 결과 출력 + 플롯"""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]

    print(f"\n{'=' * 72}")
    print(f"  DBaS-MPPI Benchmark: 4-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  Duration: {scenario['duration']}s | Seed: {args.seed}")
    print(f"{'=' * 72}")

    wb = scenario.get("planner_wheelbase", 0.5)
    model = DifferentialDriveKinematic(wheelbase=wb)
    controllers = _make_controllers(model, scenario)

    all_results = []
    for i, (name, ctrl) in enumerate(controllers.items()):
        np.random.seed(args.seed)

        print(f"\n  [{i+1}/{len(controllers)}] {name:<22}", end=" ", flush=True)
        t_start = time.time()

        history = run_single_simulation(model, ctrl, scenario, seed=args.seed)
        elapsed = time.time() - t_start

        metrics = compute_metrics(history, scenario)

        all_results.append({
            "name": name,
            "short": name.replace(" MPPI", "").replace("-MPPI", ""),
            "color": COLORS[name],
            "states": history["states"],
            "infos": history["infos"],
            "elapsed": elapsed,
            **metrics,
        })

        print(f"done ({elapsed:.1f}s)")

    # 결과 테이블
    print(f"\n{'=' * 80}")
    print(f"{'Method':<18} {'RMSE':>8} {'Collisions':>10} {'WallViol':>10} "
          f"{'MinClear':>10} {'MeanESS':>10} {'SolveMs':>10}")
    print(f"{'=' * 80}")
    for r in all_results:
        print(
            f"{r['name']:<18} "
            f"{r['rmse']:>8.4f} "
            f"{r['n_collisions']:>10d} "
            f"{r['wall_violations']:>10d} "
            f"{r['min_clearance']:>10.3f} "
            f"{r['mean_ess']:>10.1f} "
            f"{r['mean_solve_ms']:>10.1f}"
        )
    print(f"{'=' * 80}")

    # DBaS 고유 통계
    for r in all_results:
        if r["dbas_adaptive_scales"]:
            mean_scale = np.mean(r["dbas_adaptive_scales"])
            mean_bc = np.mean(r["dbas_barrier_costs"])
            print(f"\n  DBaS adaptive_scale: mean={mean_scale:.3f}, "
                  f"max={np.max(r['dbas_adaptive_scales']):.3f}")
            print(f"  DBaS barrier_cost:   mean={mean_bc:.3f}, "
                  f"max={np.max(r['dbas_barrier_costs']):.3f}")

    if not args.no_plot:
        _plot_results(all_results, scenario, args.scenario)

    return all_results


def _plot_results(results, scenario, scenario_key):
    """8-panel 결과 플롯 (2x4)"""
    dt = COMMON["dt"]
    trajectory_fn = scenario["trajectory_fn"]
    duration = scenario["duration"]
    dyn_obs = scenario.get("dynamic_obstacles", [])

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # (0,0) XY 궤적
    ax = axes[0, 0]
    t_arr = np.linspace(0, duration, 500)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, label="Ref", linewidth=1)

    for r in results:
        ax.plot(r["states"][:, 0], r["states"][:, 1], color=r["color"],
                label=r["short"], linewidth=1.5, alpha=0.8)

    for ox, oy, rad in scenario["static_obstacles"]:
        ax.add_patch(Circle((ox, oy), rad, facecolor="#FF5252", edgecolor="red",
                            alpha=0.3, linewidth=1.5))
    for wall in scenario.get("walls", []):
        axis_name, val, direction = wall
        if axis_name == 'x':
            ax.axvline(val, color="brown", linestyle="--", alpha=0.5)
        else:
            ax.axhline(val, color="brown", linestyle="--", alpha=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # (0,1) 위치 오차
    ax = axes[0, 1]
    for r in results:
        t_plot = np.arange(len(r["errors"])) * dt
        ax.plot(t_plot, r["errors"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Barrier cost (DBaS only)
    ax = axes[0, 2]
    for r in results:
        if r["dbas_barrier_costs"]:
            t_bc = np.arange(len(r["dbas_barrier_costs"])) * dt
            ax.plot(t_bc, r["dbas_barrier_costs"], color=r["color"],
                    label=r["short"], linewidth=1.5)
    if not any(r["dbas_barrier_costs"] for r in results):
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", fontsize=12, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Barrier Cost")
    ax.set_title("Barrier Cost (DBaS)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,3) Adaptive scale (DBaS only)
    ax = axes[0, 3]
    for r in results:
        if r["dbas_adaptive_scales"]:
            t_as = np.arange(len(r["dbas_adaptive_scales"])) * dt
            ax.plot(t_as, r["dbas_adaptive_scales"], color=r["color"],
                    label=r["short"], linewidth=1.5)
    if not any(r["dbas_adaptive_scales"] for r in results):
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", fontsize=12, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Adaptive Scale")
    ax.set_title("Exploration Scale (DBaS)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,0) ESS
    ax = axes[1, 0]
    for r in results:
        if r["ess_list"]:
            t_ess = np.arange(len(r["ess_list"])) * dt
            ax.plot(t_ess, r["ess_list"], color=r["color"], label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) RMSE 바 차트
    ax = axes[1, 1]
    names = [r["short"] for r in results]
    rmses = [r["rmse"] for r in results]
    colors = [r["color"] for r in results]
    bars = ax.bar(names, rmses, color=colors, alpha=0.8)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,2) 충돌 수 바 차트
    ax = axes[1, 2]
    collisions = [r["n_collisions"] for r in results]
    bars_c = ax.bar(names, collisions, color=colors, alpha=0.8)
    for bar, val in zip(bars_c, collisions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Collisions")
    ax.set_title("Collision Count")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,3) 최소 클리어런스 바 차트
    ax = axes[1, 3]
    clearances = [r["min_clearance"] for r in results]
    bars_cl = ax.bar(names, clearances, color=colors, alpha=0.8)
    for bar, val in zip(bars_cl, clearances):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("Minimum Clearance")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)

    plt.suptitle(
        f"DBaS-MPPI Benchmark [{scenario_key}]: "
        f"Vanilla vs DIAL vs CMA vs DBaS-MPPI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/dbas_mppi_benchmark_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DBaS-MPPI Benchmark")
    parser.add_argument(
        "--scenario", default="dense_static",
        choices=["dense_static", "dynamic_crossing", "narrow_passage", "noisy_mismatch"],
    )
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override scenario duration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true", help="Realtime animation")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    scenarios = get_scenarios()

    if args.live:
        if args.all_scenarios:
            for scenario_name in scenarios:
                args.scenario = scenario_name
                run_live(args)
        else:
            run_live(args)
    elif args.all_scenarios:
        for scenario_name in scenarios:
            args.scenario = scenario_name
            run_benchmark(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
