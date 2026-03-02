#!/usr/bin/env python3
"""
Conformal Prediction + CBF-MPPI 벤치마크: 5-Way × 5 시나리오

방법:
  1. Vanilla MPPI              — 안전 제어 없음 (기준선)
  2. CBF-MPPI (고정 0.01m)     — 공격적 마진 (좋은 추적, 외란에 취약)
  3. CBF-MPPI (고정 0.10m)     — 보수적 마진 (안전하지만 추적 저하)
  4. CP-CBF-MPPI (표준, γ=1.0) — CP 동적 마진 → 최적 트레이드오프
  5. ACP-CBF-MPPI (적응, γ=0.95) — 적응형 CP → 빠른 외란 대응

시나리오:
  1. accurate      — 정확한 모델, 외란 없음
  2. mismatch      — 마찰 기반 모델 불일치
  3. nonstationary — 시변 바람 + 급격한 변화
  4. dynamic       — 동적 장애물 (경로 횡단) + 경미한 노이즈
  5. corridor      — 좁은 L자 통로 + 경미한 마찰

Usage:
    PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py
    PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --live
    PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --live --scenario dynamic
    PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --live --scenario corridor
    PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --scenario dynamic --duration 12
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    CBFMPPIParams,
    ConformalCBFMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.conformal_cbf_mppi import (
    ConformalCBFMPPIController,
)
from examples.simulation_environments.common.dynamic_obstacle import (
    DynamicObstacle,
    CrossingMotion,
    BouncingMotion,
)
from examples.simulation_environments.common.obstacle_field import generate_corridor

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DT = 0.05
N_HORIZON = 20
K_SAMPLES = 512
LAMBDA = 1.0
SIGMA = np.array([0.5, 0.5])
RADIUS = 2.0
SPEED = 0.5

# 장애물: 원 궤적에 가깝게 배치 → 안전 마진이 성능에 직접 영향
CIRCLE_OBSTACLES = [
    (1.50, 1.50, 0.15),   # 45° 방향, r=0.15
    (-1.40, 1.80, 0.15),  # 128° 방향
    (0.30, -2.20, 0.15),  # 278° 방향
]

# 동적 시나리오: 정적 3 + 동적 2
DYNAMIC_STATIC_OBS = [
    (1.50, 1.50, 0.15),
    (-1.40, 1.80, 0.15),
    (0.30, -2.20, 0.15),
]
DYNAMIC_OBS_RADIUS = 0.20

# 좁은 통로: L자형 코리도
CORRIDOR_PATH = [(0.0, 0.0), (3.5, 0.0), (3.5, 3.0)]
CORRIDOR_WIDTH = 0.9
CORRIDOR_WALL_THICKNESS = 0.10
CORRIDOR_WALL_SPACING = 0.40
CORRIDOR_OBSTACLES = generate_corridor(
    CORRIDOR_PATH,
    width=CORRIDOR_WIDTH,
    thickness=CORRIDOR_WALL_THICKNESS,
    spacing=CORRIDOR_WALL_SPACING,
)

METHOD_DISPLAY = {
    "vanilla": "Vanilla MPPI",
    "cbf_small": "CBF (0.01m)",
    "cbf_large": "CBF (0.10m)",
    "cp_cbf": "CP-CBF (std)",
    "acp_cbf": "ACP-CBF (γ=0.95)",
}
METHOD_COLORS = {
    "vanilla": "#7F8C8D",    # gray
    "cbf_small": "#E74C3C",  # red (위험)
    "cbf_large": "#3498DB",  # blue (보수적)
    "cp_cbf": "#27AE60",     # green (적응형)
    "acp_cbf": "#8E44AD",    # purple (빠른 적응)
}

ALL_METHODS = ["vanilla", "cbf_small", "cbf_large", "cp_cbf", "acp_cbf"]
ALL_SCENARIOS = ["accurate", "mismatch", "nonstationary", "dynamic", "corridor"]

# CP 파라미터
CP_MARGIN_MIN = 0.005
CP_MARGIN_MAX = 0.5
CP_ALPHA = 0.1
CP_MIN_SAMPLES = 5


# ─────────────────────────────────────────────────────────────
# Scenario configuration
# ─────────────────────────────────────────────────────────────
def _create_dynamic_obstacles():
    """동적 장애물 인스턴스 생성 (느린 속도 → CBF로 회피 가능)"""
    return [
        DynamicObstacle(
            CrossingMotion(
                center=(0.0, 2.0), amplitude=1.5, period=8.0, direction=0.0,
            ),
            radius=DYNAMIC_OBS_RADIUS,
        ),
        DynamicObstacle(
            BouncingMotion(
                start=(-1.8, -0.5), velocity=(0.25, 0.18),
                bounds=(-3.0, -3.0, 3.0, 3.0),
            ),
            radius=DYNAMIC_OBS_RADIUS,
        ),
    ]


def get_scenario_config(scenario):
    """시나리오별 설정 반환"""
    if scenario in ("accurate", "mismatch", "nonstationary"):
        return {
            "static_obstacles": list(CIRCLE_OBSTACLES),
            "dynamic_obstacles": [],
            "ref_type": "circle",
            "init_state": np.array([RADIUS, 0.0, np.pi / 2]),
        }
    elif scenario == "dynamic":
        return {
            "static_obstacles": list(DYNAMIC_STATIC_OBS),
            "dynamic_obstacles": _create_dynamic_obstacles(),
            "ref_type": "circle",
            "init_state": np.array([RADIUS, 0.0, np.pi / 2]),
        }
    elif scenario == "corridor":
        return {
            "static_obstacles": list(CORRIDOR_OBSTACLES),
            "dynamic_obstacles": [],
            "ref_type": "waypoint",
            "waypoints": CORRIDOR_PATH,
            "init_state": np.array([0.2, 0.0, 0.0]),
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def get_obstacles_at_time(config, t):
    """현재 시간의 전체 장애물 목록 (정적 + 동적)"""
    obs = list(config["static_obstacles"])
    for dyn in config.get("dynamic_obstacles", []):
        obs.append(dyn.as_3tuple(t))
    return obs


# ─────────────────────────────────────────────────────────────
# Reference trajectories
# ─────────────────────────────────────────────────────────────
def circle_reference(N, dt, t_offset=0.0):
    """원형 레퍼런스 궤적"""
    t = np.arange(N + 1) * dt + t_offset
    omega = SPEED / RADIUS
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = RADIUS * np.cos(omega * t)
    ref[:, 1] = RADIUS * np.sin(omega * t)
    ref[:, 2] = omega * t + np.pi / 2
    return ref


def waypoint_reference(waypoints, N, dt, t_offset=0.0, speed=0.5):
    """웨이포인트 기반 레퍼런스 궤적 (등속 이동)"""
    wp = np.array(waypoints)
    segments = np.diff(wp, axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_dist = cum_dist[-1]
    seg_headings = np.arctan2(segments[:, 1], segments[:, 0])

    ref = np.zeros((N + 1, 3))
    for i in range(N + 1):
        t = t_offset + i * dt
        d = min(speed * t, total_dist - 0.01)
        d = max(d, 0.0)

        seg_idx = int(np.searchsorted(cum_dist, d, side="right")) - 1
        seg_idx = np.clip(seg_idx, 0, len(seg_lengths) - 1)

        d_in_seg = d - cum_dist[seg_idx]
        frac = d_in_seg / max(seg_lengths[seg_idx], 1e-6)
        frac = np.clip(frac, 0.0, 1.0)

        ref[i, 0] = wp[seg_idx, 0] + frac * segments[seg_idx, 0]
        ref[i, 1] = wp[seg_idx, 1] + frac * segments[seg_idx, 1]
        ref[i, 2] = seg_headings[seg_idx]

    return ref


def get_reference(config, N, dt, t_offset):
    """시나리오에 맞는 레퍼런스 궤적 반환"""
    if config["ref_type"] == "circle":
        return circle_reference(N, dt, t_offset)
    else:
        return waypoint_reference(
            config["waypoints"], N, dt, t_offset, speed=SPEED,
        )


# ─────────────────────────────────────────────────────────────
# Learned model simulator (prediction_fn for CP)
# ─────────────────────────────────────────────────────────────
def _create_learned_model_predictor(model, dt):
    """
    학습 모델 시뮬레이션: 상태 의존적 예측 편향.

    실제 학습 모델은 훈련 데이터 분포 밖에서 정확도가 떨어짐.
    이를 시뮬레이션하여 CP가 추적할 의미 있는 예측 오차를 생성.
    - 장애물 근처: 데이터 부족 → 큰 편향
    - 고속 영역: 비선형 효과 → 중간 편향
    - 저속/오픈 영역: 정확한 예측
    """
    def predict(state, control):
        pred = model.step(state, control, dt)
        v = abs(control[0])
        theta = state[2]
        # 속도 비례 오차: v=0.5 → ~0.04m (NN 모델의 전형적 오차)
        speed_err = 0.08 * v
        # 위치 의존 오차: ~0.01-0.02m (훈련 분포 밖 → 정확도 저하)
        pos_err = 0.02 * np.sin(state[0] * 2.0) * np.cos(state[1] * 2.0)
        # 편향 방향: 약간 빗나간 방향 (학습 모델의 체계적 오류)
        pred[0] += (speed_err + pos_err) * np.cos(theta + 0.3)
        pred[1] += (speed_err + pos_err) * np.sin(theta + 0.3)
        return pred
    return predict


# ─────────────────────────────────────────────────────────────
# Controllers
# ─────────────────────────────────────────────────────────────
def create_controller(method, model, obstacles):
    """방법별 컨트롤러 생성"""
    common = dict(N=N_HORIZON, dt=DT, K=K_SAMPLES, lambda_=LAMBDA, sigma=SIGMA)

    if method == "vanilla":
        params = MPPIParams(**common)
        return MPPIController(model, params)

    elif method == "cbf_small":
        params = CBFMPPIParams(
            **common,
            cbf_obstacles=obstacles,
            cbf_alpha=0.3,
            cbf_safety_margin=0.01,
            cbf_weight=2000.0,
        )
        return CBFMPPIController(model, params)

    elif method == "cbf_large":
        params = CBFMPPIParams(
            **common,
            cbf_obstacles=obstacles,
            cbf_alpha=0.3,
            cbf_safety_margin=0.10,
            cbf_weight=2000.0,
        )
        return CBFMPPIController(model, params)

    elif method == "cp_cbf":
        params = ConformalCBFMPPIParams(
            **common,
            cbf_obstacles=obstacles,
            cbf_alpha=0.3,
            cbf_safety_margin=0.02,  # cold start
            cbf_weight=2000.0,
            shield_enabled=False,
            cp_alpha=CP_ALPHA,
            cp_gamma=1.0,
            cp_min_samples=CP_MIN_SAMPLES,
            cp_margin_min=CP_MARGIN_MIN,
            cp_margin_max=CP_MARGIN_MAX,
        )
        pred_fn = _create_learned_model_predictor(model, DT)
        return ConformalCBFMPPIController(model, params, prediction_fn=pred_fn)

    elif method == "acp_cbf":
        params = ConformalCBFMPPIParams(
            **common,
            cbf_obstacles=obstacles,
            cbf_alpha=0.3,
            cbf_safety_margin=0.02,  # cold start
            cbf_weight=2000.0,
            shield_enabled=False,
            cp_alpha=CP_ALPHA,
            cp_gamma=0.95,
            cp_min_samples=CP_MIN_SAMPLES,
            cp_margin_min=CP_MARGIN_MIN,
            cp_margin_max=CP_MARGIN_MAX,
        )
        pred_fn = _create_learned_model_predictor(model, DT)
        return ConformalCBFMPPIController(model, params, prediction_fn=pred_fn)

    else:
        raise ValueError(f"Unknown method: {method}")


# ─────────────────────────────────────────────────────────────
# Simulation step with disturbance
# ─────────────────────────────────────────────────────────────
def sim_step(model, state, control, dt, scenario, step):
    """시나리오별 시뮬레이션 스텝"""
    next_state = model.step(state, control, dt)

    if scenario == "accurate":
        pass

    elif scenario == "mismatch":
        friction = 0.3
        v = control[0]
        next_state[0] -= friction * abs(v) * np.cos(state[2]) * dt
        next_state[1] -= friction * abs(v) * np.sin(state[2]) * dt

    elif scenario == "nonstationary":
        t = step * dt
        wind_x = 0.15 * np.sin(0.5 * t)
        wind_y = 0.10 * np.cos(0.3 * t)
        if t > 4.0:
            wind_x += 0.3
            wind_y += 0.15
        if t > 7.0:
            wind_x -= 0.2
        next_state[0] += wind_x * dt
        next_state[1] += wind_y * dt

    elif scenario == "dynamic":
        # 동적 장애물 + 2-Phase 환경 변화
        # Phase 1 (t<5s): 경미한 마찰만 → CP 마진 최소화 → CBF-small 수준 추적
        # Phase 2 (t≥5s): 강한 바람+노이즈 → CP 마진 확대 → CBF-large 수준 안전
        # 고정 마진은 한 Phase에서만 최적, CP는 양쪽 적응
        t = step * dt
        friction = 0.08
        v = control[0]
        next_state[0] -= friction * abs(v) * np.cos(state[2]) * dt
        next_state[1] -= friction * abs(v) * np.sin(state[2]) * dt
        if t >= 5.0:
            ramp = min(1.0, (t - 5.0) / 2.0)  # 2초에 걸쳐 점진적 증가
            # 바람: 궤적 바깥쪽으로 밀어냄 → 장애물에 더 가깝게
            next_state[0] += 0.30 * ramp * np.sin(0.5 * t) * dt
            next_state[1] += 0.25 * ramp * dt
            next_state[0] += np.random.normal(0, 0.025 * ramp)
            next_state[1] += np.random.normal(0, 0.025 * ramp)

    elif scenario == "corridor":
        # 좁은 L-통로 + 2-Phase 환경 변화
        # Phase 1 (t<4s): 경미한 마찰 → 순조로운 통행
        # Phase 2 (t≥4s): 횡방향 바람 → 벽 쪽으로 밀림 + 노이즈
        t = step * dt
        friction = 0.08
        v = control[0]
        next_state[0] -= friction * abs(v) * np.cos(state[2]) * dt
        next_state[1] -= friction * abs(v) * np.sin(state[2]) * dt
        if t >= 4.0:
            ramp = min(1.0, (t - 4.0) / 1.5)
            # 횡방향 바람 (로봇 진행방향의 좌측으로 밀림 → 벽 쪽)
            wind = 0.25 * ramp
            next_state[0] += wind * np.sin(state[2]) * dt
            next_state[1] -= wind * np.cos(state[2]) * dt
            next_state[0] += np.random.normal(0, 0.015 * ramp)
            next_state[1] += np.random.normal(0, 0.015 * ramp)

    return next_state


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def compute_min_obstacle_dists(positions, obstacles):
    """각 스텝별 최소 장애물 거리 (m)"""
    dists = np.full(len(positions), np.inf)
    for obs_x, obs_y, obs_r in obstacles:
        d = np.sqrt(
            (positions[:, 0] - obs_x) ** 2 + (positions[:, 1] - obs_y) ** 2
        ) - obs_r
        dists = np.minimum(dists, d)
    return dists


def compute_min_dist_at_step(pos, obstacles):
    """단일 스텝 최소 장애물 거리"""
    min_d = float("inf")
    for obs_x, obs_y, obs_r in obstacles:
        d = np.sqrt((pos[0] - obs_x) ** 2 + (pos[1] - obs_y) ** 2) - obs_r
        min_d = min(min_d, d)
    return min_d


def summarize_metrics(positions, ref_positions, obstacles, margins):
    """최종 집계 지표"""
    # 추적 오차: reference positions 대비
    errors = np.sqrt(
        (positions[:, 0] - ref_positions[:, 0]) ** 2
        + (positions[:, 1] - ref_positions[:, 1]) ** 2
    )

    # 장애물 거리
    dists = compute_min_obstacle_dists(positions, obstacles)
    collision_steps = int(np.sum(dists < 0))
    total = len(dists)

    result = {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "safety_rate": float(1.0 - collision_steps / total),
        "min_obstacle_dist": float(np.min(dists)),
        "collision_steps": collision_steps,
    }

    if margins:
        result["margin_mean"] = float(np.mean(margins))
        result["margin_min"] = float(np.min(margins))
        result["margin_max"] = float(np.max(margins))

    return result


# ─────────────────────────────────────────────────────────────
# Run single experiment
# ─────────────────────────────────────────────────────────────
def run_experiment(method, scenario, duration, model, seed=42, n_trials=5):
    """다중 시행 평균"""
    all_metrics = []
    for trial in range(n_trials):
        np.random.seed(seed + trial)
        m = _run_single(method, scenario, duration, model)
        all_metrics.append(m)

    avg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        avg[key] = float(np.mean(vals))
    return avg


def _run_single(method, scenario, duration, model):
    """단일 실험"""
    config = get_scenario_config(scenario)
    initial_obstacles = get_obstacles_at_time(config, 0.0)
    ctrl = create_controller(method, model, initial_obstacles)
    n_steps = int(duration / DT)

    state = config["init_state"].copy()
    positions = [state[:2].copy()]
    ref_positions = [state[:2].copy()]  # 첫 스텝 ref = 현재 위치
    margins = []
    times = []
    min_dists_per_step = []

    has_dynamic = len(config.get("dynamic_obstacles", [])) > 0

    for step in range(n_steps):
        t_current = step * DT
        ref = get_reference(config, N_HORIZON, DT, t_current)
        ref_positions.append(ref[1, :2].copy())  # next ref position

        # 동적 장애물 업데이트
        if has_dynamic:
            current_obs = get_obstacles_at_time(config, t_current)
            if hasattr(ctrl, "update_obstacles"):
                ctrl.update_obstacles(current_obs)
        else:
            current_obs = config["static_obstacles"]

        t0 = time.time()
        control, info = ctrl.compute_control(state, ref)
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)

        if "cp_margin" in info:
            margins.append(info["cp_margin"])
        elif hasattr(ctrl, "cbf_params"):
            margins.append(getattr(ctrl.cbf_params, "cbf_safety_margin", 0.0))

        state = sim_step(model, state, control, DT, scenario, step)
        positions.append(state[:2].copy())

        # 장애물 거리 (현재 시간의 장애물 기준)
        min_d = compute_min_dist_at_step(state[:2], current_obs)
        min_dists_per_step.append(min_d)

    positions = np.array(positions)
    ref_positions = np.array(ref_positions)
    # align lengths
    min_len = min(len(positions), len(ref_positions))
    positions = positions[:min_len]
    ref_positions = ref_positions[:min_len]

    # 장애물 거리 기반 메트릭
    dists = np.array(min_dists_per_step)
    errors = np.sqrt(
        (positions[1:, 0] - ref_positions[1:, 0]) ** 2
        + (positions[1:, 1] - ref_positions[1:, 1]) ** 2
    )
    collision_steps = int(np.sum(dists < 0))
    total = len(dists)

    metrics = {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "safety_rate": float(1.0 - collision_steps / total),
        "min_obstacle_dist": float(np.min(dists)) if len(dists) > 0 else 0.0,
        "collision_steps": collision_steps,
        "mean_time_ms": float(np.mean(times)),
    }
    if margins:
        metrics["margin_mean"] = float(np.mean(margins))
        metrics["margin_min"] = float(np.min(margins))
        metrics["margin_max"] = float(np.max(margins))

    return metrics


# ─────────────────────────────────────────────────────────────
# Live animation
# ─────────────────────────────────────────────────────────────
def run_live(methods, scenario, duration, model):
    """Live 2×3 animation"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as mpatches

    np.random.seed(42)
    config = get_scenario_config(scenario)
    n_methods = len(methods)
    n_steps = int(duration / DT)
    has_dynamic = len(config.get("dynamic_obstacles", [])) > 0
    is_corridor = scenario == "corridor"

    # 컨트롤러 생성
    initial_obs = get_obstacles_at_time(config, 0.0)
    ctrls = []
    for m in methods:
        ctrls.append({
            "name": METHOD_DISPLAY.get(m, m),
            "short": m,
            "color": METHOD_COLORS.get(m, "#333"),
            "ctrl": create_controller(m, model, initial_obs),
        })

    # 시뮬레이션 상태 초기화
    sim = []
    for _ in ctrls:
        init_state = config["init_state"].copy()
        sim.append({
            "state": init_state.copy(),
            "positions": [init_state[:2].copy()],
            "margins": [],
            "min_dists": [],
            "errors": [],
        })

    # ===== Figure =====
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    scenario_label = {
        "accurate": "Accurate Model",
        "mismatch": "Model Mismatch (friction=0.3)",
        "nonstationary": "Non-stationary Wind",
        "dynamic": "Dynamic Obstacles + Noise",
        "corridor": "Narrow L-Corridor + Mismatch",
    }
    fig.suptitle(
        f"CP + CBF-MPPI — {scenario_label.get(scenario, scenario)} (Live)",
        fontsize=15, fontweight="bold",
    )

    # 플롯 범위 설정
    if is_corridor:
        xlim = (-1.5, 5.5)
        ylim = (-1.5, 5.0)
    else:
        lim = RADIUS + 1.5
        xlim = (-lim, lim)
        ylim = (-lim, lim)

    # 레퍼런스 경로 (플롯용)
    if is_corridor:
        wp = np.array(CORRIDOR_PATH)
        ref_path_x = wp[:, 0]
        ref_path_y = wp[:, 1]
    else:
        theta_ref = np.linspace(0, 2 * np.pi, 200)
        ref_path_x = RADIUS * np.cos(theta_ref)
        ref_path_y = RADIUS * np.sin(theta_ref)

    traj_lines = []
    pos_markers = []
    dyn_obstacle_patches = []  # 동적 장애물 패치 (per subplot)

    for idx in range(min(n_methods, 5)):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]
        c = ctrls[idx]

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # 레퍼런스 경로
        if is_corridor:
            ax.plot(ref_path_x, ref_path_y, "k--", alpha=0.5, linewidth=1.5,
                    marker="o", markersize=4)
        else:
            ax.plot(ref_path_x, ref_path_y, "k--", alpha=0.25, linewidth=1)

        # 정적 장애물
        for obs_x, obs_y, obs_r in config["static_obstacles"]:
            if is_corridor:
                # 통로 벽: 작고 빽빽한 원
                circle = plt.Circle(
                    (obs_x, obs_y), obs_r, color="#555", alpha=0.6, zorder=2,
                )
            else:
                circle = plt.Circle(
                    (obs_x, obs_y), obs_r, color="red", alpha=0.4, zorder=2,
                )
            ax.add_patch(circle)

        # 동적 장애물 패치 (초기 위치)
        dyn_patches = []
        if has_dynamic:
            for dyn in config["dynamic_obstacles"]:
                px, py = dyn.get_position(0.0)
                dp = plt.Circle(
                    (px, py), dyn.radius, facecolor="orange", alpha=0.5,
                    zorder=3, linewidth=1.5, edgecolor="darkorange",
                )
                ax.add_patch(dp)
                dyn_patches.append(dp)
        dyn_obstacle_patches.append(dyn_patches)

        # 고정 마진 표시 (CBF 컨트롤러만, 코리도 제외)
        if not is_corridor and not has_dynamic:
            if c["short"] in ("cp_cbf", "acp_cbf"):
                for obs_x, obs_y, obs_r in config["static_obstacles"]:
                    mc = plt.Circle(
                        (obs_x, obs_y), obs_r + 0.1,
                        color=c["color"], alpha=0.2, linestyle="--",
                        fill=False, linewidth=1.5,
                    )
                    ax.add_patch(mc)
            elif c["short"] == "cbf_small":
                for obs_x, obs_y, obs_r in config["static_obstacles"]:
                    mc = plt.Circle(
                        (obs_x, obs_y), obs_r + 0.01,
                        color="red", alpha=0.1, linestyle=":", fill=False,
                    )
                    ax.add_patch(mc)
            elif c["short"] == "cbf_large":
                for obs_x, obs_y, obs_r in config["static_obstacles"]:
                    mc = plt.Circle(
                        (obs_x, obs_y), obs_r + 0.10,
                        color="blue", alpha=0.1, linestyle=":", fill=False,
                    )
                    ax.add_patch(mc)

        line, = ax.plot([], [], color=c["color"], linewidth=2.0, alpha=0.85)
        dot, = ax.plot([], [], "o", color=c["color"], markersize=8, zorder=5)
        traj_lines.append(line)
        pos_markers.append(dot)

        # 시작점
        sx, sy = config["init_state"][0], config["init_state"][1]
        ax.plot(sx, sy, "ks", markersize=6, zorder=3)
        ax.set_title(f"{c['name']}\nRMSE: --- | Safety: ---", fontsize=10)

    # 빈 패널 숨기기
    for idx in range(n_methods, 5):
        row, col = idx // 3, idx % 3
        axes[row][col].set_visible(False)

    # 6th panel: 실시간 비교
    ax_cmp = axes[1][2]
    ax_cmp.set_xlabel("Time (s)")
    ax_cmp.set_ylabel("Distance (m)")
    ax_cmp.set_title("Min Obstacle Distance", fontsize=10)
    ax_cmp.grid(True, alpha=0.3)
    ax_cmp.axhline(
        y=0, color="red", linestyle="--", linewidth=2, alpha=0.6,
        label="Collision!",
    )

    dist_lines = []
    margin_plot_lines = []
    for c in ctrls:
        dl, = ax_cmp.plot(
            [], [], color=c["color"], linewidth=1.5,
            label=METHOD_DISPLAY.get(c["short"], c["short"]),
        )
        dist_lines.append(dl)
        if c["short"] in ("cp_cbf", "acp_cbf"):
            ml, = ax_cmp.plot(
                [], [], color=c["color"], linewidth=1.2,
                linestyle=":", alpha=0.5,
            )
            margin_plot_lines.append(ml)
        else:
            margin_plot_lines.append(None)

    ax_cmp.legend(fontsize=7, loc="upper right")

    time_text = fig.text(
        0.5, 0.01, "", ha="center", fontsize=10, family="monospace",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    times_log = []

    def update(frame):
        if frame >= n_steps:
            return []

        t_current = frame * DT
        times_log.append(t_current)

        # 현재 시간의 장애물 목록
        current_obs = get_obstacles_at_time(config, t_current)
        ref = get_reference(config, N_HORIZON, DT, t_current)
        current_ref_pos = ref[0, :2]

        for idx, c in enumerate(ctrls):
            state = sim[idx]["state"]

            # 동적 장애물 업데이트
            if has_dynamic and hasattr(c["ctrl"], "update_obstacles"):
                c["ctrl"].update_obstacles(current_obs)

            control, info = c["ctrl"].compute_control(state, ref)

            margin = info.get("cp_margin", None)
            if margin is None and hasattr(c["ctrl"], "cbf_params"):
                margin = getattr(
                    c["ctrl"].cbf_params, "cbf_safety_margin", 0.0,
                )
            sim[idx]["margins"].append(margin if margin else 0.0)

            state = sim_step(model, state, control, DT, scenario, frame)
            sim[idx]["state"] = state
            sim[idx]["positions"].append(state[:2].copy())

            # 최소 장애물 거리 (현재 시간 장애물 기준)
            pos = state[:2]
            min_d = compute_min_dist_at_step(pos, current_obs)
            sim[idx]["min_dists"].append(min_d)

            # 추적 오차
            err = np.linalg.norm(pos - current_ref_pos)
            sim[idx]["errors"].append(err)

            # 궤적 업데이트
            xy = np.array(sim[idx]["positions"])
            traj_lines[idx].set_data(xy[:, 0], xy[:, 1])
            pos_markers[idx].set_data([state[0]], [state[1]])

            # 동적 장애물 패치 위치 업데이트
            if has_dynamic and dyn_obstacle_patches[idx]:
                for j, dyn in enumerate(config["dynamic_obstacles"]):
                    px, py = dyn.get_position(t_current)
                    dyn_obstacle_patches[idx][j].set_center((px, py))

            # 서브플롯 타이틀 업데이트
            rmse = np.sqrt(np.mean(np.array(sim[idx]["errors"]) ** 2))
            n_collision = sum(1 for d in sim[idx]["min_dists"] if d < 0)
            safety = 1.0 - n_collision / max(len(sim[idx]["min_dists"]), 1)
            margin_str = ""
            if c["short"] in ("cp_cbf", "acp_cbf") and margin:
                margin_str = f" | M={margin:.3f}"
            row, col = idx // 3, idx % 3
            axes[row][col].set_title(
                f"{c['name']}\n"
                f"RMSE: {rmse:.3f}m | Safety: {safety:.0%}"
                f" | Dist: {min_d:.3f}{margin_str}",
                fontsize=9,
            )

        # 비교 패널 업데이트
        t_arr = np.array(times_log)
        for idx in range(len(ctrls)):
            dist_lines[idx].set_data(t_arr, sim[idx]["min_dists"])
            if margin_plot_lines[idx] is not None:
                margin_plot_lines[idx].set_data(t_arr, sim[idx]["margins"])

        ax_cmp.relim()
        ax_cmp.autoscale_view()

        # 상태 바
        parts = []
        for idx, c in enumerate(ctrls):
            n_coll = sum(1 for d in sim[idx]["min_dists"] if d < 0)
            rmse = np.sqrt(np.mean(np.array(sim[idx]["errors"]) ** 2))
            parts.append(f"{c['short']}:RMSE={rmse:.3f},coll={n_coll}")
        time_text.set_text(
            f"t={t_current:.1f}s/{duration:.0f}s  |  " + "  ".join(parts)
        )

        return []

    anim = FuncAnimation(  # noqa: F841
        fig, update, frames=n_steps, interval=1, blit=False, repeat=False,
    )
    plt.show()

    # Summary
    print_live_summary(ctrls, sim, scenario, config)

    plot_dir = os.path.join(os.path.dirname(__file__), "../../plots")
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"conformal_cbf_{scenario}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")


def print_live_summary(ctrls, sim, scenario, config):
    """라이브 모드 결과 요약"""
    print("\n" + "=" * 90)
    print(f"  Results — {scenario}")
    print("=" * 90)
    header = (
        f"{'Method':<22} | {'RMSE':>7} | {'Safety':>7} | "
        f"{'MinDist':>8} | {'Collisions':>10} | {'CP Margin':>16}"
    )
    print(header)
    print("-" * len(header))

    for idx, c in enumerate(ctrls):
        errors = np.array(sim[idx]["errors"])
        min_dists = np.array(sim[idx]["min_dists"])
        margins = sim[idx]["margins"]

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        n_coll = int(np.sum(min_dists < 0))
        safety = 1.0 - n_coll / max(len(min_dists), 1)
        min_dist = float(np.min(min_dists)) if len(min_dists) > 0 else 0.0

        margin_str = "N/A"
        if c["short"] in ("cp_cbf", "acp_cbf") and margins:
            m_arr = np.array(margins)
            margin_str = (
                f"{np.mean(m_arr):.4f}"
                f"({np.min(m_arr):.3f}-{np.max(m_arr):.3f})"
            )
        elif c["short"] == "cbf_small":
            margin_str = "0.010 (fixed)"
        elif c["short"] == "cbf_large":
            margin_str = "0.100 (fixed)"

        print(
            f"{c['name']:<22} | "
            f"{rmse:>6.4f}m | "
            f"{safety:>6.1%} | "
            f"{min_dist:>7.4f}m | "
            f"{n_coll:>10} | "
            f"{margin_str:>16}"
        )

    print("=" * 90)


# ─────────────────────────────────────────────────────────────
# Batch mode
# ─────────────────────────────────────────────────────────────
def run_batch(methods, scenarios, duration, model):
    """배치 모드"""
    print("=" * 90)
    print("  Conformal Prediction + CBF-MPPI Benchmark")
    print("=" * 90)
    print(f"  Methods:   {[METHOD_DISPLAY.get(m, m) for m in methods]}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Duration:  {duration}s ({int(duration / DT)} steps), {5} trials avg")
    print("=" * 90)

    for scenario in scenarios:
        print(f"\n{'─' * 90}")
        print(f"  Scenario: {scenario}")
        print(f"{'─' * 90}")

        header = (
            f"{'Method':<22} | {'RMSE':>7} | {'Safety':>7} | "
            f"{'MinDist':>8} | {'CP Margin':>16} | {'Time':>7}"
        )
        print(header)
        print("-" * len(header))

        for method in methods:
            try:
                metrics = run_experiment(method, scenario, duration, model)

                margin_str = "N/A"
                if "margin_mean" in metrics:
                    margin_str = (
                        f"{metrics['margin_mean']:.4f}"
                        f"({metrics['margin_min']:.3f}-{metrics['margin_max']:.3f})"
                    )
                elif method == "cbf_small":
                    margin_str = "0.010 (fixed)"
                elif method == "cbf_large":
                    margin_str = "0.100 (fixed)"

                print(
                    f"{METHOD_DISPLAY.get(method, method):<22} | "
                    f"{metrics['rmse']:>6.4f}m | "
                    f"{metrics['safety_rate']:>6.1%} | "
                    f"{metrics['min_obstacle_dist']:>7.4f}m | "
                    f"{margin_str:>16} | "
                    f"{metrics['mean_time_ms']:>5.1f}ms"
                )
            except Exception as e:
                print(
                    f"{METHOD_DISPLAY.get(method, method):<22} | ERROR: {e}"
                )
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 90}")
    print("  Benchmark Complete")
    print(f"{'=' * 90}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CP + CBF-MPPI Benchmark")
    parser.add_argument("--live", action="store_true", help="Live animation")
    parser.add_argument(
        "--scenario", type=str, default=None, choices=ALL_SCENARIOS,
        help="시나리오 (None=전체, live시 기본=nonstationary)",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="방법 (콤마 구분)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="시간 (초)",
    )
    args = parser.parse_args()

    methods = args.methods.split(",") if args.methods else ALL_METHODS
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    if args.live:
        scenario = args.scenario or "nonstationary"
        run_live(methods, scenario, args.duration, model)
    else:
        scenarios = [args.scenario] if args.scenario else ALL_SCENARIOS
        run_batch(methods, scenarios, args.duration, model)


if __name__ == "__main__":
    main()
