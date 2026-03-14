#!/usr/bin/env python3
"""
Flow-MPPI 종합 벤치마크: 7 로봇 모델 × 4 환경 × 3 컨트롤러

로봇 모델:
  1. DifferentialDrive Kinematic  (3D state, 2D control)
  2. Ackermann Kinematic          (4D state, 2D control)
  3. Swerve Drive Kinematic       (3D state, 3D control)
  4. DifferentialDrive Dynamic     (5D state, 2D control)
  5. Mobile Manipulator 2-DOF     (5D state, 4D control) — EE 추적
  6. Mobile Manipulator 6-DOF     (9D state, 8D control) — EE 3D 추적
  7. Quadruped (body-level)       (5D state, 5D control)

환경 시나리오:
  A. clean      — 외란 없음 (기준선)
  B. noisy      — 프로세스 노이즈 추가
  C. obstacles  — 정적 장애물 배치
  D. mismatch   — 위치 의존 모델 오류 (바람/마찰)

컨트롤러:
  1. Vanilla MPPI     — 기본 가우시안 샘플링
  2. DIAL-MPPI        — 확산 어닐링 (multi-iter)
  3. Flow-MPPI        — CFM 학습 기반 다중 모달 샘플링

궤적: circle (기본 모델), ee_circle (2-DOF EE), ee_3d_circle (6-DOF EE)

측정: RMSE, Mean Cost, ESS, Compute Time

Usage:
    PYTHONPATH=. python examples/comparison/flow_mppi_multi_benchmark.py --no-plot
    PYTHONPATH=. python examples/comparison/flow_mppi_multi_benchmark.py --model diffdrive
    PYTHONPATH=. python examples/comparison/flow_mppi_multi_benchmark.py --model mobile_manip --no-plot
    PYTHONPATH=. python examples/comparison/flow_mppi_multi_benchmark.py --model quadruped --no-plot
    PYTHONPATH=. python examples/comparison/flow_mppi_multi_benchmark.py --scenario obstacles
    PYTHONPATH=. python examples/comparison/flow_mppi_multi_benchmark.py --all --no-plot
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
from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.models.kinematic.swerve_drive_kinematic import SwerveDriveKinematic
from mppi_controller.models.kinematic.mobile_manipulator_kinematic import (
    MobileManipulatorKinematic,
)
from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.kinematic.quadruped_kinematic import QuadrupedKinematic
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    FlowMPPIParams,
    DIALMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.flow_mppi import FlowMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
    EndEffectorTrackingCost,
    EndEffectorTerminalCost,
    EndEffector3DTrackingCost,
    EndEffector3DTerminalCost,
)
from mppi_controller.utils.trajectory import (
    circle_trajectory,
    figure_eight_trajectory,
    ee_circle_trajectory,
    ee_3d_circle_trajectory,
    generate_reference_trajectory,
)
from mppi_controller.simulation.harness import SimulationHarness


# ══════════════════════════════════════════════════════════════
#  로봇 모델 설정
# ══════════════════════════════════════════════════════════════

def get_model_configs():
    """각 모델의 설정 반환"""
    return {
        "diffdrive": {
            "name": "DiffDrive Kinematic",
            "model": DifferentialDriveKinematic(wheelbase=0.5),
            "state_dim": 3,
            "control_dim": 2,
            "Q": np.array([10.0, 10.0, 1.0]),
            "R": np.array([0.1, 0.1]),
            "sigma": np.array([0.5, 0.5]),
            "initial_state": np.array([0.0, 0.0, 0.0]),
            "noise_std": np.array([0.03, 0.03, 0.01]),
        },
        "ackermann": {
            "name": "Ackermann Kinematic",
            "model": AckermannKinematic(wheelbase=0.5),
            "state_dim": 4,
            "control_dim": 2,
            "Q": np.array([10.0, 10.0, 1.0, 0.1]),
            "R": np.array([0.1, 0.1]),
            "sigma": np.array([0.5, 0.5]),
            "initial_state": np.array([0.0, 0.0, 0.0, 0.0]),
            "noise_std": np.array([0.03, 0.03, 0.01, 0.005]),
        },
        "swerve": {
            "name": "Swerve Kinematic",
            "model": SwerveDriveKinematic(),
            "state_dim": 3,
            "control_dim": 3,
            "Q": np.array([10.0, 10.0, 1.0]),
            "R": np.array([0.1, 0.1, 0.1]),
            "sigma": np.array([0.5, 0.5, 0.5]),
            "initial_state": np.array([0.0, 0.0, 0.0]),
            "noise_std": np.array([0.03, 0.03, 0.01]),
        },
        "diffdrive_dyn": {
            "name": "DiffDrive Dynamic",
            "model": DifferentialDriveDynamic(
                mass=10.0, inertia=1.0, c_v=0.1, c_omega=0.1
            ),
            "state_dim": 5,
            "control_dim": 2,
            "Q": np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
            "R": np.array([0.1, 0.1]),
            "sigma": np.array([1.0, 1.0]),
            "initial_state": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "noise_std": np.array([0.02, 0.02, 0.005, 0.02, 0.01]),
        },
        "mobile_manip": {
            "name": "MobileManip 2-DOF",
            "model": MobileManipulatorKinematic(L1=0.3, L2=0.25),
            "state_dim": 5,
            "control_dim": 4,
            "ee_model": True,
            "ee_dim": 2,
            "Q": np.array([5.0, 5.0, 1.0, 2.0, 2.0]),
            "R": np.array([0.1, 0.1, 0.05, 0.05]),
            "sigma": np.array([0.5, 0.5, 0.8, 0.8]),
            "initial_state": np.array([0.0, 0.0, 0.0, 0.5, -0.5]),
            "noise_std": np.array([0.02, 0.02, 0.005, 0.01, 0.01]),
        },
        "mobile_manip_6dof": {
            "name": "MobileManip 6-DOF",
            "model": MobileManipulator6DOFKinematic(),
            "state_dim": 9,
            "control_dim": 8,
            "ee_model": True,
            "ee_dim": 3,
            "Q": np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "R": np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
            "sigma": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            "initial_state": np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0]),
            "noise_std": np.array([0.02, 0.02, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
        },
        "quadruped": {
            "name": "Quadruped Kinematic",
            "model": QuadrupedKinematic(),
            "state_dim": 5,
            "control_dim": 5,
            "Q": np.array([10.0, 10.0, 1.0, 2.0, 1.0]),
            "R": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            "sigma": np.array([0.5, 0.3, 0.5, 0.2, 0.3]),
            "initial_state": np.array([0.0, 0.0, 0.0, 0.28, 0.0]),
            "noise_std": np.array([0.03, 0.03, 0.01, 0.005, 0.005]),
        },
    }


# ══════════════════════════════════════════════════════════════
#  장애물 & 모델 미스매치
# ══════════════════════════════════════════════════════════════

OBSTACLES = [
    (2.5, 2.0, 0.4),
    (-1.5, 3.0, 0.5),
    (1.0, -3.0, 0.35),
    (-3.0, -1.0, 0.4),
]


class MismatchWrapper:
    """위치 의존 모델 오류 래퍼 (바람 + 마찰 효과)"""

    def __init__(self, model, drift_scale=0.12, friction_scale=0.06):
        self.model = model
        self.drift_scale = drift_scale
        self.friction_scale = friction_scale

    def step(self, state, control, dt):
        """nominal step + mismatch perturbation"""
        next_state = self.model.step(state, control, dt)

        # 위치 기반 드리프트 (바람 효과)
        if state[1] > 0:
            next_state[0] += self.drift_scale * state[1] * dt
        # 거리 기반 마찰
        dist = np.sqrt(state[0] ** 2 + state[1] ** 2)
        if dist > 1.0:
            friction = self.friction_scale * (dist - 1.0) * dt
            next_state[0] -= friction * np.cos(state[2])
            next_state[1] -= friction * np.sin(state[2])

        return next_state


# ══════════════════════════════════════════════════════════════
#  레퍼런스 궤적 생성
# ══════════════════════════════════════════════════════════════

def make_reference_fn(model_key, trajectory="circle"):
    """모델에 맞는 레퍼런스 궤적 함수 생성"""

    configs = get_model_configs()
    cfg = configs[model_key]
    state_dim = cfg["state_dim"]
    ee_model = cfg.get("ee_model", False)
    ee_dim = cfg.get("ee_dim", 0)

    # EE 모델: EE 궤적 함수 사용
    if ee_model and ee_dim == 2:
        return lambda t: ee_circle_trajectory(
            t, radius=0.5, angular_velocity=0.3,
            center=(1.0, 0.0), state_dim=state_dim,
        )
    elif ee_model and ee_dim == 3:
        return lambda t: ee_3d_circle_trajectory(
            t, radius=0.3, angular_velocity=0.3,
            center=(0.4, 0.0, 0.4), state_dim=state_dim,
        )

    # 기본 모델: 베이스 궤적
    if trajectory == "circle":
        base_fn = lambda t: circle_trajectory(t, radius=3.0, angular_velocity=0.15)
    else:
        base_fn = lambda t: figure_eight_trajectory(t, scale=3.0, period=25.0)

    def traj_fn(t):
        base = base_fn(t)  # (3,) = [x, y, θ]
        if state_dim == 3:
            return base
        elif state_dim == 4:
            return np.array([base[0], base[1], base[2], 0.0])
        elif state_dim == 5:
            return np.array([base[0], base[1], base[2], 0.0, 0.0])
        return base

    return traj_fn


# ══════════════════════════════════════════════════════════════
#  시뮬레이션
# ══════════════════════════════════════════════════════════════

def run_simulation(
    model, controller, traj_fn, initial_state, dt, duration,
    process_noise_std=None, mismatch_wrapper=None
):
    """SimulationHarness 기반 시뮬레이션 (호환 인터페이스)"""
    N_horizon = controller.params.N

    def ref_fn(t):
        return generate_reference_trajectory(traj_fn, t, N_horizon, dt)

    harness = SimulationHarness(dt=dt, headless=True, seed=42)
    harness.add_controller(
        "ctrl", controller, model,
        process_noise_std=process_noise_std,
        real_model=mismatch_wrapper if mismatch_wrapper else None,
    )
    results = harness.run(ref_fn, initial_state, duration)
    r = results["ctrl"]
    h = r["history"]

    # ESS / cost 추출
    ess_list = [info.get("ess", 0.0) for info in h.get("info", [])]
    costs = [info.get("best_cost", 0.0) for info in h.get("info", [])]

    return {
        "states": np.vstack([initial_state[None, :], h["state"]]),
        "controls": h["control"],
        "compute_times": h["solve_time"],
        "ess": np.array(ess_list),
        "costs": np.array(costs),
    }


def compute_rmse(states, traj_fn, dt, model=None, ee_dim=0):
    """위치 추적 RMSE (EE 모델이면 FK 기반 EE RMSE)"""
    errors = []
    for i, s in enumerate(states):
        t = i * dt
        ref = traj_fn(t)
        if ee_dim > 0 and model is not None:
            ee_pos = model.forward_kinematics(s)
            errors.append(np.linalg.norm(ee_pos[:ee_dim] - ref[:ee_dim]))
        else:
            errors.append(np.linalg.norm(s[:2] - ref[:2]))
    return np.sqrt(np.mean(np.array(errors) ** 2))


def check_collisions(states, obstacles):
    """충돌 횟수 & 최소 거리"""
    if not obstacles:
        return 0, float("inf")
    min_dist = float("inf")
    collisions = 0
    for s in states:
        for ox, oy, r in obstacles:
            d = np.sqrt((s[0] - ox) ** 2 + (s[1] - oy) ** 2) - r
            min_dist = min(min_dist, d)
            if d < 0:
                collisions += 1
    return collisions, min_dist


# ══════════════════════════════════════════════════════════════
#  컨트롤러 생성
# ══════════════════════════════════════════════════════════════

def create_controllers(model, cfg, obstacles=None, N=20, K=128):
    """3개 컨트롤러 (Vanilla, DIAL, Flow) 생성"""
    Q, R, sigma = cfg["Q"], cfg["R"], cfg["sigma"]
    dt = 0.05
    ee_model = cfg.get("ee_model", False)
    ee_dim = cfg.get("ee_dim", 0)

    def make_cost(obs_list):
        if ee_model and ee_dim == 2:
            cost_parts = [
                EndEffectorTrackingCost(model, weight=100.0),
                EndEffectorTerminalCost(model, weight=200.0),
                ControlEffortCost(R),
            ]
        elif ee_model and ee_dim == 3:
            cost_parts = [
                EndEffector3DTrackingCost(model, weight=100.0),
                EndEffector3DTerminalCost(model, weight=200.0),
                ControlEffortCost(R),
            ]
        else:
            cost_parts = [StateTrackingCost(Q), TerminalCost(Q), ControlEffortCost(R)]
        if obs_list:
            cost_parts.append(ObstacleCost(obs_list, cost_weight=500.0))
        return CompositeMPPICost(cost_parts)

    obs = obstacles or []
    cost = make_cost(obs)

    # Vanilla
    vanilla_params = MPPIParams(N=N, K=K, dt=dt, sigma=sigma, Q=Q, R=R)
    vanilla = MPPIController(model, vanilla_params, cost)

    # DIAL
    dial_params = DIALMPPIParams(
        N=N, K=K, dt=dt, sigma=sigma, Q=Q, R=R,
        n_diffuse_init=5, n_diffuse=2,
    )
    dial = DIALMPPIController(model, dial_params, make_cost(obs))

    # Flow
    flow_params = FlowMPPIParams(
        N=N, K=K, dt=dt, sigma=sigma, Q=Q, R=R,
        flow_hidden_dims=[128, 128],
        flow_num_steps=5,
        flow_mode="blend",
        flow_blend_ratio=0.5,
        flow_min_samples=20,
    )
    flow = FlowMPPIController(model, flow_params, make_cost(obs))

    return {
        "Vanilla": vanilla,
        "DIAL": dial,
        "Flow": flow,
    }


def bootstrap_flow(ctrl, model, traj_fn, initial_state, dt, n_bootstrap=50,
                    process_noise_std=None, mismatch_wrapper=None):
    """Flow 모델 bootstrap: 가우시안으로 데이터 수집 → 학습"""
    state = initial_state.copy()
    rng = np.random.default_rng(123)
    t = 0.0
    N = ctrl.params.N
    for _ in range(n_bootstrap):
        ref = generate_reference_trajectory(traj_fn, t, N, dt)
        control, _ = ctrl.compute_control(state, ref)
        if mismatch_wrapper is not None:
            state = mismatch_wrapper.step(state, control, dt)
        else:
            state = model.step(state, control, dt)
        if process_noise_std is not None:
            state = state + rng.normal(0, process_noise_std)
        state = model.normalize_state(state)
        t += dt
    ctrl.train_flow_model(epochs=50)
    ctrl.reset()


# ══════════════════════════════════════════════════════════════
#  시나리오 정의
# ══════════════════════════════════════════════════════════════

def get_scenarios():
    return {
        "clean": {
            "name": "Clean (no disturbance)",
            "process_noise_std": None,
            "obstacles": [],
            "mismatch": False,
        },
        "noisy": {
            "name": "Noisy (process noise)",
            "process_noise_std": True,  # model-specific
            "obstacles": [],
            "mismatch": False,
        },
        "obstacles": {
            "name": "Obstacles",
            "process_noise_std": None,
            "obstacles": OBSTACLES,
            "mismatch": False,
        },
        "mismatch": {
            "name": "Model Mismatch (wind+friction)",
            "process_noise_std": None,
            "obstacles": [],
            "mismatch": True,
        },
    }


# ══════════════════════════════════════════════════════════════
#  벤치마크 실행
# ══════════════════════════════════════════════════════════════

def run_single_benchmark(model_key, scenario_key, show_plot=True):
    """단일 (모델, 시나리오) 조합 벤치마크"""
    model_configs = get_model_configs()
    scenarios = get_scenarios()

    cfg = model_configs[model_key]
    scen = scenarios[scenario_key]

    model = cfg["model"]
    initial_state = cfg["initial_state"]
    dt = 0.05
    duration = 6.0
    N, K = 20, 128

    # 프로세스 노이즈
    noise_std = cfg["noise_std"] if scen["process_noise_std"] else None

    # 모델 미스매치
    mismatch = MismatchWrapper(model) if scen["mismatch"] else None

    # 장애물
    obstacles = scen["obstacles"]

    # 궤적
    traj_fn = make_reference_fn(model_key, "circle")

    print(f"\n  ┌─ {cfg['name']} × {scen['name']}")

    # 컨트롤러 생성
    controllers = create_controllers(model, cfg, obstacles, N=N, K=K)

    results = {}
    for ctrl_name, ctrl in controllers.items():
        # Flow bootstrap
        if ctrl_name == "Flow":
            bootstrap_flow(
                ctrl, model, traj_fn, initial_state, dt,
                process_noise_std=noise_std, mismatch_wrapper=mismatch,
            )

        res = run_simulation(
            model, ctrl, traj_fn, initial_state, dt, duration,
            process_noise_std=noise_std, mismatch_wrapper=mismatch,
        )
        res["rmse"] = compute_rmse(
            res["states"], traj_fn, dt,
            model=model, ee_dim=cfg.get("ee_dim", 0),
        )
        res["collisions"], res["min_dist"] = check_collisions(
            res["states"], obstacles
        )
        results[ctrl_name] = res

    # 결과 출력
    has_obs = bool(obstacles)
    if has_obs:
        header = f"  │ {'Method':<10} {'RMSE':>7} {'Cost':>9} {'ESS':>7} {'Time':>9} {'Col':>4} {'MinD':>6}"
    else:
        header = f"  │ {'Method':<10} {'RMSE':>7} {'Cost':>9} {'ESS':>7} {'Time':>9}"
    print(header)
    print(f"  │ {'-' * (len(header) - 4)}")

    for name, res in results.items():
        rmse = res["rmse"]
        cost = np.mean(res["costs"])
        ess = np.mean(res["ess"])
        ms = np.mean(res["compute_times"]) * 1000
        if has_obs:
            col = res["collisions"]
            md = res["min_dist"]
            print(f"  │ {name:<10} {rmse:>7.4f} {cost:>9.1f} {ess:>7.1f} {ms:>7.1f}ms {col:>4d} {md:>6.3f}")
        else:
            print(f"  │ {name:<10} {rmse:>7.4f} {cost:>9.1f} {ess:>7.1f} {ms:>7.1f}ms")
    print(f"  └─")

    # Plot
    if show_plot:
        _save_plot(model_key, scenario_key, results, traj_fn, obstacles, dt, duration, cfg)

    return results


def _save_plot(model_key, scenario_key, results, traj_fn, obstacles, dt, duration, cfg):
    """결과 플롯 저장"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"{cfg['name']} — {scenario_key}", fontsize=14)

        colors = {"Vanilla": "#2196F3", "DIAL": "#FF9800", "Flow": "#4CAF50"}

        # XY trajectory
        ax = axes[0]
        t_ref = np.linspace(0, duration, 300)
        ref_pts = np.array([traj_fn(t)[:2] for t in t_ref])
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, label="Reference", lw=1)
        for name, res in results.items():
            ax.plot(res["states"][:, 0], res["states"][:, 1],
                    color=colors[name], label=name, lw=1.5, alpha=0.85)
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color="red", alpha=0.3)
            ax.add_patch(circle)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("XY Trajectory")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Tracking error over time
        ax = axes[1]
        ee_dim = cfg.get("ee_dim", 0)
        ee_model_flag = cfg.get("ee_model", False)
        model = cfg["model"]
        for name, res in results.items():
            errors = []
            for i, s in enumerate(res["states"]):
                t = i * dt
                ref = traj_fn(t)
                if ee_model_flag and ee_dim > 0:
                    ee_pos = model.forward_kinematics(s)
                    errors.append(np.linalg.norm(ee_pos[:ee_dim] - ref[:ee_dim]))
                else:
                    errors.append(np.linalg.norm(s[:2] - ref[:2]))
            ax.plot(np.arange(len(errors)) * dt, errors,
                    color=colors[name], label=name, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Tracking Error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ESS
        ax = axes[2]
        for name, res in results.items():
            ax.plot(res["ess"], color=colors[name], label=name, alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("ESS")
        ax.set_title("Effective Sample Size")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        path = f"plots/flow_multi_{model_key}_{scenario_key}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"    Plot: {path}")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════
#  종합 결과 요약
# ══════════════════════════════════════════════════════════════

def print_summary(all_results):
    """모든 결과의 종합 요약 테이블"""
    print("\n" + "=" * 72)
    print("  종합 결과 요약 (RMSE)")
    print("=" * 72)

    models = list(get_model_configs().keys())
    scenarios = list(get_scenarios().keys())
    ctrl_names = ["Vanilla", "DIAL", "Flow"]

    # 모델별 × 시나리오별 RMSE 테이블
    for model_key in models:
        cfg = get_model_configs()[model_key]
        print(f"\n  {cfg['name']}:")
        print(f"  {'Scenario':<15}", end="")
        for cn in ctrl_names:
            print(f" {cn:>10}", end="")
        print(f" {'Best':>10}")
        print(f"  {'-' * 55}")

        for scen_key in scenarios:
            key = (model_key, scen_key)
            if key not in all_results:
                continue
            res = all_results[key]
            scen_name = get_scenarios()[scen_key]["name"].split("(")[0].strip()
            print(f"  {scen_name:<15}", end="")
            rmses = {}
            for cn in ctrl_names:
                if cn in res:
                    rmse = res[cn]["rmse"]
                    rmses[cn] = rmse
                    print(f" {rmse:>10.4f}", end="")
                else:
                    print(f" {'N/A':>10}", end="")
            if rmses:
                best = min(rmses, key=rmses.get)
                print(f" {best:>10}")
            else:
                print()

    # Flow 승률 집계
    total = 0
    flow_wins = 0
    flow_best_or_close = 0
    for key, res in all_results.items():
        if "Vanilla" in res and "Flow" in res:
            total += 1
            rmses = {cn: res[cn]["rmse"] for cn in ctrl_names if cn in res}
            best = min(rmses.values())
            if rmses.get("Flow", float("inf")) <= best * 1.001:
                flow_wins += 1
            if rmses.get("Flow", float("inf")) <= best * 1.05:
                flow_best_or_close += 1

    print(f"\n  Flow-MPPI 최저 RMSE: {flow_wins}/{total}")
    print(f"  Flow-MPPI ≤ 5% 격차: {flow_best_or_close}/{total}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Flow-MPPI Multi-Model Benchmark")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(get_model_configs().keys()),
                        help="Single model to test")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(get_scenarios().keys()),
                        help="Single scenario to test")
    parser.add_argument("--all", action="store_true",
                        help="Run all model × scenario combinations")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    show_plot = not args.no_plot
    all_results = {}

    models = [args.model] if args.model else list(get_model_configs().keys())
    scenarios = [args.scenario] if args.scenario else list(get_scenarios().keys())

    if not args.all and args.model is None and args.scenario is None:
        # 기본: diffdrive × 모든 시나리오
        models = ["diffdrive"]

    print("=" * 72)
    print("  Flow-MPPI 종합 벤치마크")
    print(f"  Models: {', '.join(models)}")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print("=" * 72)

    for model_key in models:
        for scen_key in scenarios:
            res = run_single_benchmark(model_key, scen_key, show_plot)
            all_results[(model_key, scen_key)] = res

    # 종합 요약 (2개 이상 조합이면)
    if len(all_results) > 1:
        print_summary(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
