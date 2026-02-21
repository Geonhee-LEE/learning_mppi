#!/usr/bin/env python3
"""
Model Mismatch Comparison Demo

시뮬레이터와 컨트롤러의 모델이 다를 때 (model mismatch) 학습 모델의 진가를 입증하는 데모.

두 가지 세계 모드를 지원:

  --world perturbed (기본값): 섭동된 기구학 세계 → 4-Way 비교
    1. Kinematic (Mismatched): 순수 기구학 — 미스매치로 큰 오차
    2. Neural (End-to-End): 학습된 NN — 전체 동역학 근사
    3. Residual (Hybrid): 기구학 + NN 보정 — 물리+데이터 융합
    4. Oracle (Perfect): 섭동 동역학 — 이론적 상한

  --world dynamic: DifferentialDriveDynamic(5D) 세계 → 10-Way 비교
    1. Kinematic (3D): 관성/마찰 모름
    2. Neural (3D): 데이터에서 전체 학습
    3. Residual (3D): 물리+데이터 융합
    4. Dynamic (5D): 구조 알지만 파라미터 틀림 (c_v=0.1 vs 실제 0.5)
    5. MAML (3D): FOMAML 메타 학습 + 실시간 few-shot 적응
    6. MAML-5D (5D): 5D 잔차 MAML 적응
    7. EKF (5D): 확장 칼만 필터 파라미터 추정
    8. L1 (5D): L1 적응 제어 외란 추정
    9. ALPaCA (5D): Bayesian last-layer 적응
    10. Oracle (5D): 정확한 파라미터

Usage:
    # Perturbed world (기본, 4-way)
    python model_mismatch_comparison_demo.py --all
    python model_mismatch_comparison_demo.py --live --trajectory circle --duration 20

    # Dynamic world (6-way)
    python model_mismatch_comparison_demo.py --all --world dynamic --trajectory circle --duration 20
    python model_mismatch_comparison_demo.py --live --world dynamic --trajectory circle --duration 20

    # Dynamic world with disturbance (MAML 이점 부각)
    python model_mismatch_comparison_demo.py --evaluate --world dynamic --noise 0.5 --disturbance combined
    python model_mismatch_comparison_demo.py --live --world dynamic --noise 0.7 --disturbance wind
"""

import numpy as np
import argparse
import sys
import os
import time
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.base_model import RobotModel
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
from mppi_controller.learning.maml_trainer import MAMLTrainer
from mppi_controller.learning.alpaca_trainer import ALPaCATrainer
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    ControlEffortCost,
    AngleAwareTrackingCost,
    AngleAwareTerminalCost,
)
from mppi_controller.models.kinematic.dynamic_kinematic_adapter import (
    DynamicKinematicAdapter,
)
from disturbance_profiles import (
    DisturbanceProfile,
    create_disturbance,
)
from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer
from mppi_controller.simulation.metrics import (
    compute_metrics,
    print_metrics,
    compare_metrics,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)

# ==================== 경로 설정 ====================
DATA_DIR = "data/learned_models"
MODEL_DIR = "models/learned_models"
PLOT_DIR = "plots"
DATA_FILE = "mismatch_data.pkl"
NEURAL_MODEL_FILE = "mismatch_neural_model.pth"
RESIDUAL_MODEL_FILE = "mismatch_residual_model.pth"

# Dynamic world 전용 파일
DYNAMIC_DATA_FILE = "dynamic_mismatch_data.pkl"
DYNAMIC_NEURAL_MODEL_FILE = "dynamic_neural_model.pth"
DYNAMIC_RESIDUAL_MODEL_FILE = "dynamic_residual_model.pth"
DYNAMIC_MAML_MODEL_FILE = "dynamic_maml_meta_model.pth"
DYNAMIC_MAML_5D_MODEL_FILE = "dynamic_maml_5d_meta_model.pth"
DYNAMIC_ALPACA_MODEL_FILE = "dynamic_alpaca_meta_model.pth"

# ==================== 섭동 파라미터 ====================
# 강한 미스매치: 실제 로봇의 바닥 마찰 + 모터 바이어스 + 비대칭 마찰
FRICTION_FACTOR = 0.55        # 위치 변화량 * 0.55 (45% 속도 감쇠)
ACTUATOR_BIAS = np.array([0.12, -0.05])  # [v, omega] 바이어스
HEADING_FRICTION = 0.80       # 각속도 감쇠 (20% 회전 마찰)
PROCESS_NOISE_STD = np.array([0.005, 0.005, 0.002])  # 평가 시 작은 노이즈


# ==================== 섭동된 현실 세계 ====================

def perturbed_step(state, control, base_model, dt, add_noise=True):
    """
    섭동된 "현실 세계" step 함수.

    실제 로봇에서 발생하는 모델 미스매치를 시뮬레이션:
    1. 액추에이터 바이어스: [+0.12, -0.05] on [v, omega]
    2. 마찰: 위치 변화량 * 0.55 (45% 선속도 감쇠)
    3. 회전 마찰: 각속도 변화량 * 0.80 (20% 회전 감쇠)
    4. 프로세스 노이즈 (선택적)
    """
    # 1. 액추에이터 바이어스 적용
    biased_control = control + ACTUATOR_BIAS

    # 2. 기구학 적분 (RK4)
    nominal_next = base_model.step(state, biased_control, dt)

    # 3. 마찰: 위치/각도 변화량 감쇠
    delta = nominal_next - state
    delta[:2] *= FRICTION_FACTOR   # x, y 변화량 감쇠
    delta[2] *= HEADING_FRICTION   # theta 변화량 감쇠
    next_state = state + delta

    # 4. 프로세스 노이즈
    if add_noise:
        noise = np.random.normal(0.0, PROCESS_NOISE_STD)
        next_state += noise

    # 5. 각도 정규화
    next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))

    return next_state


class PerturbedDiffDriveModel(RobotModel):
    """
    섭동된 Differential Drive 모델 (Oracle).

    MPPI 롤아웃에서 perturbed_step과 동일한 섭동 적용 (노이즈 제외).
    컨트롤러가 "현실 세계"의 동역학을 완벽히 알고 있는 이상적 상한.

    step()을 직접 오버라이드하여 perturbed_step()과 정확히 동일한 계산 수행.
    """

    def __init__(self, v_max=1.0, omega_max=1.0):
        self.base_model = DifferentialDriveKinematic(v_max=v_max, omega_max=omega_max)

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(self, state, control):
        """섭동 동역학 (배치 지원) — MPPI의 배치 rollout에서 사용."""
        # 액추에이터 바이어스
        if control.ndim == 1:
            biased_control = control + ACTUATOR_BIAS
        else:
            biased_control = control + ACTUATOR_BIAS[np.newaxis, :]

        # 기구학 state_dot
        state_dot = self.base_model.forward_dynamics(state, biased_control)

        # 마찰: state_dot 감쇠
        result = state_dot.copy()
        result[..., :2] *= FRICTION_FACTOR
        result[..., 2] *= HEADING_FRICTION

        return result

    def step(self, state, control, dt):
        """
        perturbed_step과 동일한 적분 (노이즈 제외, 배치 지원).

        perturbed_step은 RK4 적분 후 delta에 마찰을 곱하므로,
        Oracle도 동일한 순서를 따라야 한다.
        """
        # 액추에이터 바이어스
        if control.ndim == 1:
            biased_control = control + ACTUATOR_BIAS
        else:
            biased_control = control + ACTUATOR_BIAS[np.newaxis, :]

        # 기구학 RK4 적분 (base_model의 step)
        nominal_next = self.base_model.step(state, biased_control, dt)

        # 마찰: delta에 적용
        delta = nominal_next - state
        delta[..., :2] *= FRICTION_FACTOR
        delta[..., 2] *= HEADING_FRICTION

        return state + delta

    def get_control_bounds(self):
        return self.base_model.get_control_bounds()

    def state_to_dict(self, state):
        return self.base_model.state_to_dict(state)

    def normalize_state(self, state):
        return self.base_model.normalize_state(state)


# ==================== Dynamic World (5D 현실 세계) ====================

# DynamicKinematicAdapter: [x,y,θ,v,ω], 제어=[v_cmd,ω_cmd]
# PD + friction이 RK4 매 substep에서 일관되게 재계산됨.
DYNAMIC_WORLD_PARAMS = {
    "c_v": 0.5,          # 선형 마찰 계수
    "c_omega": 0.3,      # 각속도 마찰 계수
    "k_v": 5.0,          # PD 선형 속도 게인
    "k_omega": 5.0,      # PD 각속도 게인
    "process_noise_std": np.array([0.005, 0.005, 0.002, 0.01, 0.005]),
}


class DynamicWorld:
    """
    DynamicKinematicAdapter를 "현실 세계"로 감싸는 래퍼.

    velocity command [v_cmd, ω_cmd] → DynamicKinematicAdapter(RK4) → 5D step.
    외부에는 3D observation [x, y, θ]만 반환.

    DynamicKinematicAdapter를 직접 사용하여 Oracle과 동일한 RK4 적분 구조 보장.
    (PD + friction이 RK4 매 substep에서 일관되게 재계산됨)

    disturbance가 설정되면 시간에 따라 변하는 외란을 주입:
    - get_force(): additive 5D force
    - get_param_delta(): c_v, c_omega 동적 변경
    """

    def __init__(self, c_v=0.5, c_omega=0.3, k_v=5.0, k_omega=5.0,
                 process_noise_std=np.array([0.005, 0.005, 0.002, 0.01, 0.005]),
                 disturbance=None):
        self._base_c_v = c_v
        self._base_c_omega = c_omega
        self._c_v = c_v
        self._c_omega = c_omega
        self._adapter = DynamicKinematicAdapter(
            c_v=c_v, c_omega=c_omega, k_v=k_v, k_omega=k_omega,
        )
        self.process_noise_std = process_noise_std
        self.state_5d = np.zeros(5)  # [x, y, θ, v, ω]
        self.disturbance = disturbance
        self._time = 0.0

    def reset(self, state_3d):
        """3D 상태로 리셋 → 5D [x, y, θ, 0, 0]."""
        self.state_5d = np.array([state_3d[0], state_3d[1], state_3d[2], 0.0, 0.0])
        self._time = 0.0
        # 마찰 계수 초기화
        self._c_v = self._base_c_v
        self._c_omega = self._base_c_omega
        self._adapter._c_v = self._base_c_v
        self._adapter._c_omega = self._base_c_omega
        return self.get_observation()

    def step(self, velocity_cmd, dt, add_noise=True):
        """
        velocity_cmd = [v_cmd, ω_cmd] → DynamicKinematicAdapter.step() → 3D obs.
        """
        # 외란에 의한 파라미터 변동 적용
        if self.disturbance is not None:
            param_delta = self.disturbance.get_param_delta(self._time)
            if param_delta:
                self._c_v = self._base_c_v + param_delta.get("delta_c_v", 0.0)
                self._c_omega = self._base_c_omega + param_delta.get("delta_c_omega", 0.0)
                self._adapter._c_v = self._c_v
                self._adapter._c_omega = self._c_omega

        next_state = self._adapter.step(self.state_5d, velocity_cmd, dt)

        # 외란 힘 적용 (additive)
        if self.disturbance is not None:
            force = self.disturbance.get_force(self._time, self.state_5d)
            next_state += force * dt

        # 프로세스 노이즈
        if add_noise:
            noise = np.random.normal(0.0, self.process_noise_std)
            next_state += noise

        # 각도 정규화
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))

        self.state_5d = next_state
        self._time += dt
        return self.get_observation()

    def get_observation(self):
        """3D observation [x, y, θ]."""
        return self.state_5d[:3].copy()

    def get_full_state(self):
        """5D full state [x, y, θ, v, ω]."""
        return self.state_5d.copy()


# ==================== MPPI 파라미터 ====================

def create_mppi_params():
    return MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0]),
    )


def create_angle_aware_cost(params):
    """Angle-aware 비용 함수 생성 (heading wrapping 적용)."""
    return CompositeMPPICost([
        AngleAwareTrackingCost(params.Q),
        AngleAwareTerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ])


# ==================== 5D MPPI 파라미터 & 유틸리티 ====================

def create_5d_mppi_params():
    """5D state [x,y,θ,v,ω] 용 MPPI 파라미터."""
    return MPPIParams(
        N=30, dt=0.05, K=1024, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0, 0.1, 0.1]),
        R=np.array([0.1, 0.1]),
        Qf=np.array([20.0, 20.0, 2.0, 0.2, 0.2]),
    )


def create_5d_angle_aware_cost(params):
    """5D용 Angle-aware 비용 함수 (θ=index 2)."""
    return CompositeMPPICost([
        AngleAwareTrackingCost(params.Q, angle_indices=(2,)),
        AngleAwareTerminalCost(params.Qf, angle_indices=(2,)),
        ControlEffortCost(params.R),
    ])


def make_5d_reference(ref_3d):
    """
    3D reference (N+1, 3) → 5D reference (N+1, 5).

    v_ref: 연속 위치 차이로 추정, ω_ref: heading 차이로 추정.
    """
    N_plus_1 = ref_3d.shape[0]
    ref_5d = np.zeros((N_plus_1, 5))
    ref_5d[:, :3] = ref_3d

    if N_plus_1 > 1:
        # v_ref: 위치 변화량 / dt
        dx = np.diff(ref_3d[:, 0])
        dy = np.diff(ref_3d[:, 1])
        v_ref = np.sqrt(dx**2 + dy**2) / 0.05  # dt=0.05
        ref_5d[:-1, 3] = v_ref
        ref_5d[-1, 3] = v_ref[-1] if len(v_ref) > 0 else 0.0

        # ω_ref: heading 변화량 / dt
        dtheta = np.diff(ref_3d[:, 2])
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        omega_ref = dtheta / 0.05
        ref_5d[:-1, 4] = omega_ref
        ref_5d[-1, 4] = omega_ref[-1] if len(omega_ref) > 0 else 0.0

    return ref_5d


# ==================== 커스텀 시뮬레이션 루프 ====================

def run_with_perturbed_world(controller, base_model, params, trajectory_fn, duration, seed):
    """
    섭동된 세계에서 컨트롤러를 실행하는 커스텀 시뮬 루프.

    Simulator 클래스 대신 직접 루프를 돌려서 perturbed_step() 사용.
    """
    np.random.seed(seed)
    controller.reset()

    state = trajectory_fn(0.0).copy()
    t = 0.0
    dt = params.dt
    num_steps = int(duration / dt)

    history = {
        "time": [],
        "state": [],
        "control": [],
        "reference": [],
        "solve_time": [],
    }

    for _ in range(num_steps):
        ref_traj = generate_reference_trajectory(trajectory_fn, t, params.N, dt)

        t_start = time.time()
        control, info = controller.compute_control(state, ref_traj)
        solve_time = time.time() - t_start

        history["time"].append(t)
        history["state"].append(state.copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_traj[0].copy())
        history["solve_time"].append(solve_time)

        state = perturbed_step(state, control, base_model, dt, add_noise=True)
        t += dt

    for key in history:
        history[key] = np.array(history[key])

    return history


def run_with_dynamic_world(controller, world, params, trajectory_fn, duration, seed, is_5d=False):
    """
    DynamicWorld에서 컨트롤러를 실행하는 커스텀 시뮬 루프.

    3D 컨트롤러(Kinematic/Neural/Residual): 3D state, 3D ref.
    5D 컨트롤러(Dynamic/Oracle): 5D state (world.get_full_state()), 5D ref.
    메트릭은 3D(x,y,θ) 기준으로 통일.
    """
    np.random.seed(seed)
    controller.reset()

    state_3d = trajectory_fn(0.0)[:3].copy()
    world.reset(state_3d)
    state = np.array([*state_3d, 0.0, 0.0]) if is_5d else state_3d.copy()

    t = 0.0
    dt = params.dt
    num_steps = int(duration / dt)

    history = {
        "time": [],
        "state": [],
        "control": [],
        "reference": [],
        "solve_time": [],
    }

    for _ in range(num_steps):
        ref_3d = generate_reference_trajectory(trajectory_fn, t, params.N, dt)
        ref = make_5d_reference(ref_3d) if is_5d else ref_3d

        t_start = time.time()
        control, info = controller.compute_control(state, ref)
        solve_time = time.time() - t_start

        # 메트릭 기록 (항상 3D)
        history["time"].append(t)
        history["state"].append(state[:3].copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_3d[0].copy())
        history["solve_time"].append(solve_time)

        # DynamicWorld step (velocity command → PD → 5D → 3D obs)
        obs_3d = world.step(control, dt, add_noise=True)

        if is_5d:
            state = world.get_full_state()
        else:
            state = obs_3d.copy()

        t += dt

    for key in history:
        history[key] = np.array(history[key])

    return history


# ==================== Stage 1: 데이터 수집 ====================

def collect_data(args):
    """섭동된/동적 세계에서 학습 데이터 수집."""
    world_type = getattr(args, "world", "perturbed")
    use_dynamic = (world_type == "dynamic")

    print("\n" + "=" * 80)
    title = "Stage 1: Data Collection (Dynamic World)" if use_dynamic else "Stage 1: Data Collection (Perturbed World)"
    print(title.center(80))
    print("=" * 80)

    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = create_mppi_params()
    controller = MPPIController(base_model, params)

    collector = DataCollector(state_dim=3, control_dim=2, save_dir=DATA_DIR)

    # Dynamic world 인스턴스 (필요시)
    if use_dynamic:
        dyn_world = DynamicWorld(**DYNAMIC_WORLD_PARAMS)

    # 데이터 수집: 노이즈 OFF로 깨끗한 데이터 확보
    trajectories = ["circle", "figure8", "sine", "slalom", "straight"]
    episodes_per_traj = 5
    episode_duration = 20.0  # seconds

    for traj_type in trajectories:
        trajectory_fn = create_trajectory_function(traj_type)

        for ep in range(episodes_per_traj):
            np.random.seed(ep * 100 + hash(traj_type) % 1000)

            # Domain randomization: 에피소드별 다른 마찰 파라미터
            if use_dynamic:
                c_v_ep = np.random.uniform(0.1, 0.8)
                c_omega_ep = np.random.uniform(0.1, 0.5)
                dyn_world_ep = DynamicWorld(
                    c_v=c_v_ep, c_omega=c_omega_ep,
                    k_v=DYNAMIC_WORLD_PARAMS["k_v"],
                    k_omega=DYNAMIC_WORLD_PARAMS["k_omega"],
                    process_noise_std=np.zeros(5),
                )
                print(f"  Collecting: {traj_type} ep {ep+1}/{episodes_per_traj} (c_v={c_v_ep:.2f}, c_omega={c_omega_ep:.2f})")
            else:
                dyn_world_ep = None
                print(f"  Collecting: {traj_type} episode {ep+1}/{episodes_per_traj}")

            state = trajectory_fn(0.0).copy()
            controller.reset()
            if use_dynamic:
                dyn_world_ep.reset(state)
            t = 0.0
            dt = params.dt
            num_steps = int(episode_duration / dt)

            for _ in range(num_steps):
                ref_traj = generate_reference_trajectory(trajectory_fn, t, params.N, dt)
                control, _ = controller.compute_control(state, ref_traj)

                if use_dynamic:
                    obs_3d = dyn_world_ep.step(control, dt, add_noise=False)
                    next_state = obs_3d
                else:
                    next_state = perturbed_step(state, control, base_model, dt, add_noise=False)

                collector.add_sample(state, control, next_state, dt)

                state = next_state
                t += dt

            collector.end_episode()

    # 랜덤 탐색 에피소드 추가 (다양한 제어 입력 + domain randomization)
    print("  Collecting: random exploration episodes")
    for ep in range(10):
        np.random.seed(9000 + ep)
        state = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            np.random.uniform(-np.pi, np.pi),
        ])
        if use_dynamic:
            c_v_ep = np.random.uniform(0.1, 0.8)
            c_omega_ep = np.random.uniform(0.1, 0.5)
            dyn_world_ep = DynamicWorld(
                c_v=c_v_ep, c_omega=c_omega_ep,
                k_v=DYNAMIC_WORLD_PARAMS["k_v"],
                k_omega=DYNAMIC_WORLD_PARAMS["k_omega"],
                process_noise_std=np.zeros(5),
            )
            dyn_world_ep.reset(state)
        dt = params.dt
        for _ in range(200):
            control = np.array([
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
            ])
            if use_dynamic:
                obs_3d = dyn_world_ep.step(control, dt, add_noise=False)
                next_state = obs_3d
            else:
                next_state = perturbed_step(state, control, base_model, dt, add_noise=False)
            collector.add_sample(state, control, next_state, dt)
            state = next_state
        collector.end_episode()

    # 통계 출력
    stats = collector.get_statistics()
    print(f"\n  Collected {stats['num_samples']} samples across {stats['num_episodes']} episodes")
    print(f"  State mean: {stats['state_mean']}")
    print(f"  State std:  {stats['state_std']}")

    data_file = DYNAMIC_DATA_FILE if use_dynamic else DATA_FILE
    collector.save(data_file)
    print(f"  Saved to: {os.path.join(DATA_DIR, data_file)}")


# ==================== Stage 2: 모델 학습 ====================

def train_models(args):
    """Neural 및 Residual 모델 학습."""
    world_type = getattr(args, "world", "perturbed")
    use_dynamic = (world_type == "dynamic")

    print("\n" + "=" * 80)
    print("Stage 2: Model Training".center(80))
    print("=" * 80)

    # 데이터/모델 파일 선택
    data_file = DYNAMIC_DATA_FILE if use_dynamic else DATA_FILE
    neural_file = DYNAMIC_NEURAL_MODEL_FILE if use_dynamic else NEURAL_MODEL_FILE
    residual_file = DYNAMIC_RESIDUAL_MODEL_FILE if use_dynamic else RESIDUAL_MODEL_FILE

    # 데이터 로드
    data_path = os.path.join(DATA_DIR, data_file)
    if not os.path.exists(data_path):
        print(f"  Error: Data file not found: {data_path}")
        print("  Run with --collect-data first.")
        return

    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)

    print(f"  Loaded {len(raw_data['states'])} samples")

    # 각도 래핑으로 인한 state_dot 보정
    # (next_theta - theta)/dt 에서 angle wrapping으로 +-2pi 스파이크 발생 방지
    states = raw_data["states"]
    next_states = raw_data["next_states"]
    dt_arr = raw_data["dt"]
    corrected_dots = (next_states - states) / dt_arr[:, np.newaxis]
    # theta 차이만 angle-aware로 재계산
    theta_diff = next_states[:, 2] - states[:, 2]
    theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
    corrected_dots[:, 2] = theta_diff / dt_arr
    raw_data["state_dots"] = corrected_dots

    # ---- Neural Model (End-to-End) ----
    print("\n  Training Neural Model (End-to-End)...")
    neural_dataset = DynamicsDataset(raw_data, train_ratio=0.8, normalize=True)
    train_inputs, train_targets = neural_dataset.get_train_data()
    val_inputs, val_targets = neural_dataset.get_val_data()
    norm_stats = neural_dataset.get_normalization_stats()

    neural_trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[256, 256],
        activation="relu",
        dropout_rate=0.0,
        learning_rate=1e-3,
        device="cpu",
        save_dir=MODEL_DIR,
    )

    neural_trainer.train(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        norm_stats=norm_stats,
        epochs=300,
        batch_size=128,
        early_stopping_patience=40,
        verbose=True,
    )
    neural_trainer.save_model(neural_file)

    # ---- Residual Model (보정량만 학습) ----
    print("\n  Training Residual Model (Correction Only)...")

    # residual_dot = actual_dot - kinematic_dot 계산
    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    states = raw_data["states"]
    controls = raw_data["controls"]
    actual_dots = raw_data["state_dots"]

    # 기구학 예측 (벡터화)
    kinematic_dots = base_model.forward_dynamics(states, controls)

    # Residual = actual - kinematic
    residual_dots = actual_dots - kinematic_dots

    print(f"  Residual stats:")
    print(f"    mean: {np.mean(residual_dots, axis=0)}")
    print(f"    std:  {np.std(residual_dots, axis=0)}")

    # Residual 데이터셋 구성
    residual_data = {
        "states": states,
        "controls": controls,
        "state_dots": residual_dots,
        "next_states": raw_data["next_states"],
        "dt": raw_data["dt"],
    }

    residual_dataset = DynamicsDataset(residual_data, train_ratio=0.8, normalize=True)
    res_train_inputs, res_train_targets = residual_dataset.get_train_data()
    res_val_inputs, res_val_targets = residual_dataset.get_val_data()
    res_norm_stats = residual_dataset.get_normalization_stats()

    residual_trainer = NeuralNetworkTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[128, 128],
        activation="relu",
        dropout_rate=0.0,
        learning_rate=1e-3,
        device="cpu",
        save_dir=MODEL_DIR,
    )

    residual_trainer.train(
        train_inputs=res_train_inputs,
        train_targets=res_train_targets,
        val_inputs=res_val_inputs,
        val_targets=res_val_targets,
        norm_stats=res_norm_stats,
        epochs=300,
        batch_size=128,
        early_stopping_patience=40,
        verbose=True,
    )
    residual_trainer.save_model(residual_file)

    print("\n  Models saved:")
    print(f"    Neural:   {os.path.join(MODEL_DIR, neural_file)}")
    print(f"    Residual: {os.path.join(MODEL_DIR, residual_file)}")


# ==================== Stage 2b: MAML 메타 학습 ====================

def meta_train_maml(args):
    """MAML 메타 학습 (다양한 DynamicWorld 설정으로 학습)."""
    print("\n" + "=" * 80)
    print("Stage 2b: MAML Meta-Training".center(80))
    print("=" * 80)

    trainer = MAMLTrainer(
        state_dim=3, control_dim=2,
        hidden_dims=[128, 128],
        inner_lr=0.005, inner_steps=10,
        meta_lr=5e-4,
        task_batch_size=8,
        support_size=100,
        query_size=100,
        device="cpu",
        save_dir=MODEL_DIR,
    )
    trainer.meta_train(n_iterations=1000, verbose=True)
    trainer.save_meta_model(DYNAMIC_MAML_MODEL_FILE)


def meta_train_maml_5d(args):
    """5D MAML 메타 학습 — 잔차(residual) 타겟으로 학습.

    메타 학습 시 base_model(c_v=0.1) 대비 잔차를 타겟으로 사용하여,
    온라인 적응 시 동일한 잔차 분포에서 적응 가능하게 함.
    (기존: total dynamics 학습 → residual 적응 = 분포 불일치)
    """
    meta_algo = getattr(args, "meta_algo", "fomaml")

    print("\n" + "=" * 80)
    print(f"Stage 2c: 5D MAML Meta-Training — Residual ({meta_algo})".center(80))
    print("=" * 80)

    # 잔차 계산용 고정 base model (온라인과 동일)
    base_5d = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
    dt = 0.05

    if meta_algo == "reptile":
        from mppi_controller.learning.reptile_trainer import ReptileTrainer
        trainer = ReptileTrainer(
            state_dim=5, control_dim=2,
            hidden_dims=[128, 128],
            inner_lr=0.005, inner_steps=10,
            epsilon=0.1,
            task_batch_size=8,
            support_size=100,
            device="cpu",
            save_dir=MODEL_DIR,
        )
    else:
        trainer = MAMLTrainer(
            state_dim=5, control_dim=2,
            hidden_dims=[128, 128],
            inner_lr=0.005, inner_steps=10,
            meta_lr=5e-4,
            task_batch_size=8,
            support_size=100,
            query_size=100,
            device="cpu",
            save_dir=MODEL_DIR,
        )

    # 데이터 생성을 잔차 타겟으로 오버라이드
    original_gen = trainer._generate_task_data_5d

    def residual_gen(task_params, n_samples):
        states, controls, next_states = original_gen(task_params, n_samples)
        # base model 예측
        base_dots = base_5d.forward_dynamics(states, controls)
        base_next = states + base_dots * dt
        # 잔차 타겟: state + (actual - predicted)
        residual_next = states + (next_states - base_next)
        return states, controls, residual_next

    trainer._generate_task_data_5d = residual_gen

    trainer.meta_train(n_iterations=1000, verbose=True)
    trainer.save_meta_model(DYNAMIC_MAML_5D_MODEL_FILE)


def run_with_dynamic_world_maml(_, maml_model, world, params,
                                 trajectory_fn, duration, seed,
                                 warmup_steps=20, adapt_interval=20,
                                 buffer_size=50):
    """
    MAML 컨트롤러 전용 시뮬 루프 — ResidualDynamics + MAML 온라인 적응.

    Phase 1 (warm-up, ~2초): 기구학 모델로 제어 + 데이터 수집
    Phase 2 (adapted): ResidualDynamics(kinematic + MAML residual)로 전환,
                        주기적으로 MAML residual을 메타에서 재적응

    핵심: 기구학이 base dynamics 제공 → 안정성 보장,
          MAML은 잔차(마찰/관성 보정)만 학습 → 빠른 적응.
    """
    from collections import deque

    np.random.seed(seed)

    # 기구학 모델 (base)
    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    cost_fn = create_angle_aware_cost(params)

    # Phase 1: 기구학 컨트롤러
    kin_controller = MPPIController(base_model, params, cost_function=cost_fn)

    # Phase 2: ResidualDynamics(kinematic + MAML)
    residual_model = ResidualDynamics(
        base_model=base_model,
        learned_model=maml_model,
        use_residual=True,
    )
    maml_residual_controller = MPPIController(residual_model, params, cost_function=cost_fn)

    state_3d = trajectory_fn(0.0)[:3].copy()
    world.reset(state_3d)
    state = state_3d.copy()

    t = 0.0
    dt = params.dt
    num_steps = int(duration / dt)

    history = {
        "time": [],
        "state": [],
        "control": [],
        "reference": [],
        "solve_time": [],
    }

    buffer_states = deque(maxlen=buffer_size)
    buffer_controls = deque(maxlen=buffer_size)
    buffer_next = deque(maxlen=buffer_size)
    adapted = False

    for step_i in range(num_steps):
        ref_3d = generate_reference_trajectory(trajectory_fn, t, params.N, dt)

        t_start = time.time()
        if not adapted:
            control, info = kin_controller.compute_control(state, ref_3d)
        else:
            control, info = maml_residual_controller.compute_control(state, ref_3d)
        solve_time = time.time() - t_start

        history["time"].append(t)
        history["state"].append(state.copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_3d[0].copy())
        history["solve_time"].append(solve_time)

        obs_3d = world.step(control, dt, add_noise=True)

        buffer_states.append(state.copy())
        buffer_controls.append(control.copy())
        buffer_next.append(obs_3d.copy())

        # Phase 1→2 전환: 잔차 학습
        if not adapted and step_i >= warmup_steps:
            buf_s = np.array(buffer_states)
            buf_c = np.array(buffer_controls)
            buf_n = np.array(buffer_next)

            # 잔차 target: actual_next - kinematic_next
            kin_dots = np.array([
                base_model.forward_dynamics(s, c) for s, c in zip(buf_s, buf_c)
            ])
            kin_next = buf_s + kin_dots * dt
            # MAML target: (residual_next - state) / dt = residual_dot
            # residual_next = actual_next - kin_dot * dt  (state cancels)
            # We need target state such that (target - state)/dt = residual_dot
            residual_target = buf_s + (buf_n - kin_next)

            maml_model.adapt(buf_s, buf_c, residual_target, dt,
                            restore=True, temporal_decay=0.95)
            maml_residual_controller.reset()
            adapted = True

        # Phase 2: 주기적 잔차 재적응
        elif adapted and step_i % adapt_interval == 0:
            buf_s = np.array(buffer_states)
            buf_c = np.array(buffer_controls)
            buf_n = np.array(buffer_next)
            kin_dots = np.array([
                base_model.forward_dynamics(s, c) for s, c in zip(buf_s, buf_c)
            ])
            kin_next = buf_s + kin_dots * dt
            residual_target = buf_s + (buf_n - kin_next)
            maml_model.adapt(buf_s, buf_c, residual_target, dt,
                            restore=True, temporal_decay=0.95)

        state = obs_3d.copy()
        t += dt

    for key in history:
        history[key] = np.array(history[key])

    return history


def run_with_dynamic_world_maml_5d(maml_model_5d, world, params_5d,
                                    trajectory_fn, duration, seed,
                                    warmup_steps=10, adapt_interval=20,
                                    buffer_size=50, error_threshold=0.15):
    """
    5D MAML 컨트롤러 시뮬 루프 — DynamicKinematicAdapter + 5D MAML residual.

    Phase 1 (warm-up, ~1초): DynamicKinematicAdapter(wrong params)로 제어 + 5D 데이터 수집
    Phase 2 (adapted): ResidualDynamics(DynamicKinematicAdapter + MAML-5D)로 전환,
                        주기적/오차 기반 재적응

    핵심: 5D state로 v/ω를 직접 관측 → 관성/마찰 보정 정확도 대폭 향상.
    """
    from collections import deque

    np.random.seed(seed)

    # 5D base model (잘못된 파라미터 — warmup용)
    base_5d = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
    cost_fn_5d = create_5d_angle_aware_cost(params_5d)

    # Phase 1: DynamicKinematicAdapter 컨트롤러 (wrong params, warmup)
    adapter_controller = MPPIController(base_5d, params_5d, cost_function=cost_fn_5d)

    # Phase 2: ResidualDynamics(DynamicKinematicAdapter + MAML-5D)
    residual_model_5d = ResidualDynamics(
        base_model=base_5d,
        learned_model=maml_model_5d,
        use_residual=True,
    )
    maml_5d_controller = MPPIController(residual_model_5d, params_5d, cost_function=cost_fn_5d)

    state_3d = trajectory_fn(0.0)[:3].copy()
    world.reset(state_3d)
    state = np.array([*state_3d, 0.0, 0.0])  # 5D

    t = 0.0
    dt = params_5d.dt
    num_steps = int(duration / dt)

    history = {
        "time": [],
        "state": [],
        "control": [],
        "reference": [],
        "solve_time": [],
    }

    buffer_states = deque(maxlen=buffer_size)
    buffer_controls = deque(maxlen=buffer_size)
    buffer_next = deque(maxlen=buffer_size)
    adapted = False
    recent_errors = deque(maxlen=5)

    for step_i in range(num_steps):
        ref_3d = generate_reference_trajectory(trajectory_fn, t, params_5d.N, dt)
        ref_5d = make_5d_reference(ref_3d)

        t_start = time.time()
        if not adapted:
            control, info = adapter_controller.compute_control(state, ref_5d)
        else:
            control, info = maml_5d_controller.compute_control(state, ref_5d)
        solve_time = time.time() - t_start

        # 메트릭 기록 (3D로 통일)
        history["time"].append(t)
        history["state"].append(state[:3].copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_3d[0].copy())
        history["solve_time"].append(solve_time)

        # 오차 추적 (적응 트리거용)
        pos_error = float(np.linalg.norm(state[:2] - ref_3d[0, :2]))
        recent_errors.append(pos_error)

        obs_3d = world.step(control, dt, add_noise=True)
        next_state_5d = world.get_full_state()

        buffer_states.append(state.copy())
        buffer_controls.append(control.copy())
        buffer_next.append(next_state_5d.copy())

        def _adapt_5d_residual():
            """5D residual target 계산 + MAML adapt (exponential weighting)."""
            buf_s = np.array(buffer_states)
            buf_c = np.array(buffer_controls)
            buf_n = np.array(buffer_next)
            # 5D 기구학 예측 (잘못된 파라미터)
            kin_dots = base_5d.forward_dynamics(buf_s, buf_c)
            kin_next = buf_s + kin_dots * dt
            # Residual target: actual - kinematic prediction
            residual_target = buf_s + (buf_n - kin_next)
            maml_model_5d.adapt(buf_s, buf_c, residual_target, dt,
                                restore=True, temporal_decay=0.95)

        # Phase 1→2 전환
        if not adapted and step_i >= warmup_steps:
            _adapt_5d_residual()
            maml_5d_controller.reset()
            adapted = True

        # Phase 2: 주기적 + 오차 기반 재적응
        elif adapted:
            avg_error = np.mean(recent_errors) if recent_errors else 0.0
            if step_i % adapt_interval == 0 or avg_error > error_threshold:
                _adapt_5d_residual()

        state = next_state_5d
        t += dt

    for key in history:
        history[key] = np.array(history[key])

    return history


# ==================== EKF / L1 / ALPaCA 시뮬 루프 ====================

def run_with_dynamic_world_ekf(world, params_5d, trajectory_fn, duration, seed,
                                c_v_init=0.1, c_omega_init=0.1):
    """
    EKF 적응 컨트롤러 시뮬 루프 — 1-Phase (warmup 불필요).

    EKF가 매 스텝 파라미터를 추정하므로 즉시 적응 가능.
    DynamicKinematicAdapter를 내부 모델로 사용, 추정된 c_v/c_omega로 갱신.
    """
    np.random.seed(seed)

    ekf_model = EKFAdaptiveDynamics(
        c_v_init=c_v_init, c_omega_init=c_omega_init,
        k_v=5.0, k_omega=5.0,
    )
    cost_fn_5d = create_5d_angle_aware_cost(params_5d)
    controller = MPPIController(ekf_model, params_5d, cost_function=cost_fn_5d)

    state_3d = trajectory_fn(0.0)[:3].copy()
    world.reset(state_3d)
    state = np.array([*state_3d, 0.0, 0.0])  # 5D

    t = 0.0
    dt = params_5d.dt
    num_steps = int(duration / dt)

    history = {
        "time": [], "state": [], "control": [],
        "reference": [], "solve_time": [],
    }

    prev_state = state.copy()
    prev_control = np.zeros(2)

    for step_i in range(num_steps):
        ref_3d = generate_reference_trajectory(trajectory_fn, t, params_5d.N, dt)
        ref_5d = make_5d_reference(ref_3d)

        t_start = time.time()
        control, info = controller.compute_control(state, ref_5d)
        solve_time = time.time() - t_start

        history["time"].append(t)
        history["state"].append(state[:3].copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_3d[0].copy())
        history["solve_time"].append(solve_time)

        obs_3d = world.step(control, dt, add_noise=True)
        next_state_5d = world.get_full_state()

        # EKF 업데이트 (매 스텝)
        if step_i > 0:
            ekf_model.update_step(prev_state, prev_control, state, dt)

        prev_state = state.copy()
        prev_control = control.copy()
        state = next_state_5d
        t += dt

    for key in history:
        history[key] = np.array(history[key])
    return history


def run_with_dynamic_world_l1(world, params_5d, trajectory_fn, duration, seed,
                               c_v_nom=0.1, c_omega_nom=0.1):
    """
    L1 적응 컨트롤러 시뮬 루프 — 1-Phase (warmup 불필요).

    L1이 매 스텝 외란을 추정하여 forward_dynamics에 보정 적용.
    """
    np.random.seed(seed)

    l1_model = L1AdaptiveDynamics(
        c_v_nom=c_v_nom, c_omega_nom=c_omega_nom,
        k_v=5.0, k_omega=5.0,
    )
    cost_fn_5d = create_5d_angle_aware_cost(params_5d)
    controller = MPPIController(l1_model, params_5d, cost_function=cost_fn_5d)

    state_3d = trajectory_fn(0.0)[:3].copy()
    world.reset(state_3d)
    state = np.array([*state_3d, 0.0, 0.0])  # 5D

    t = 0.0
    dt = params_5d.dt
    num_steps = int(duration / dt)

    history = {
        "time": [], "state": [], "control": [],
        "reference": [], "solve_time": [],
    }

    prev_state = state.copy()
    prev_control = np.zeros(2)

    for step_i in range(num_steps):
        ref_3d = generate_reference_trajectory(trajectory_fn, t, params_5d.N, dt)
        ref_5d = make_5d_reference(ref_3d)

        t_start = time.time()
        control, info = controller.compute_control(state, ref_5d)
        solve_time = time.time() - t_start

        history["time"].append(t)
        history["state"].append(state[:3].copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_3d[0].copy())
        history["solve_time"].append(solve_time)

        obs_3d = world.step(control, dt, add_noise=True)
        next_state_5d = world.get_full_state()

        # L1 업데이트 (매 스텝)
        if step_i > 0:
            l1_model.update_step(prev_state, prev_control, state, dt)

        prev_state = state.copy()
        prev_control = control.copy()
        state = next_state_5d
        t += dt

    for key in history:
        history[key] = np.array(history[key])
    return history


def run_with_dynamic_world_alpaca(alpaca_model, world, params_5d,
                                   trajectory_fn, duration, seed,
                                   warmup_steps=10, adapt_interval=20,
                                   buffer_size=50):
    """
    ALPaCA 컨트롤러 시뮬 루프 — 2-Phase (MAML-5D 패턴).

    Phase 1: DynamicKinematicAdapter(wrong params)로 warmup + 데이터 수집
    Phase 2: ResidualDynamics(DynamicKinematicAdapter + ALPaCA)로 전환,
             주기적 closed-form Bayesian 적응
    """
    from collections import deque

    np.random.seed(seed)

    base_5d = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
    cost_fn_5d = create_5d_angle_aware_cost(params_5d)

    # Phase 1: adapter 컨트롤러
    adapter_controller = MPPIController(base_5d, params_5d, cost_function=cost_fn_5d)

    # Phase 2: Residual(adapter + ALPaCA)
    residual_model = ResidualDynamics(
        base_model=base_5d,
        learned_model=alpaca_model,
        use_residual=True,
    )
    alpaca_controller = MPPIController(residual_model, params_5d, cost_function=cost_fn_5d)

    state_3d = trajectory_fn(0.0)[:3].copy()
    world.reset(state_3d)
    state = np.array([*state_3d, 0.0, 0.0])

    t = 0.0
    dt = params_5d.dt
    num_steps = int(duration / dt)

    history = {
        "time": [], "state": [], "control": [],
        "reference": [], "solve_time": [],
    }

    buffer_states = deque(maxlen=buffer_size)
    buffer_controls = deque(maxlen=buffer_size)
    buffer_next = deque(maxlen=buffer_size)
    adapted = False

    for step_i in range(num_steps):
        ref_3d = generate_reference_trajectory(trajectory_fn, t, params_5d.N, dt)
        ref_5d = make_5d_reference(ref_3d)

        t_start = time.time()
        if not adapted:
            control, info = adapter_controller.compute_control(state, ref_5d)
        else:
            control, info = alpaca_controller.compute_control(state, ref_5d)
        solve_time = time.time() - t_start

        history["time"].append(t)
        history["state"].append(state[:3].copy())
        history["control"].append(control.copy())
        history["reference"].append(ref_3d[0].copy())
        history["solve_time"].append(solve_time)

        obs_3d = world.step(control, dt, add_noise=True)
        next_state_5d = world.get_full_state()

        buffer_states.append(state.copy())
        buffer_controls.append(control.copy())
        buffer_next.append(next_state_5d.copy())

        def _adapt_alpaca_residual():
            buf_s = np.array(buffer_states)
            buf_c = np.array(buffer_controls)
            buf_n = np.array(buffer_next)
            kin_dots = base_5d.forward_dynamics(buf_s, buf_c)
            kin_next = buf_s + kin_dots * dt
            residual_target = buf_s + (buf_n - kin_next)
            alpaca_model.adapt(buf_s, buf_c, residual_target, dt, restore=True)

        if not adapted and step_i >= warmup_steps:
            _adapt_alpaca_residual()
            alpaca_controller.reset()
            adapted = True
        elif adapted and step_i % adapt_interval == 0:
            _adapt_alpaca_residual()

        state = next_state_5d
        t += dt

    for key in history:
        history[key] = np.array(history[key])
    return history


# ==================== ALPaCA 메타 학습 ====================

def meta_train_alpaca(args):
    """ALPaCA 메타 학습 (5D residual)."""
    print("\n" + "=" * 80)
    print("Stage 2d: ALPaCA Meta-Training (5D Residual)".center(80))
    print("=" * 80)

    base_5d = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
    dt = 0.05

    trainer = ALPaCATrainer(
        state_dim=5, control_dim=2,
        hidden_dims=[128, 128],
        feature_dim=64,
        meta_lr=1e-3,
        task_batch_size=8,
        support_size=100,
        query_size=100,
        device="cpu",
        save_dir=MODEL_DIR,
    )

    # Residual meta-training: override data gen
    original_gen = trainer._generate_task_data_5d

    def residual_gen(task_params, n_samples):
        states, controls, next_states = original_gen(task_params, n_samples)
        base_dots = base_5d.forward_dynamics(states, controls)
        base_next = states + base_dots * dt
        residual_next = states + (next_states - base_next)
        return states, controls, residual_next

    trainer._generate_task_data_5d = residual_gen

    trainer.meta_train(n_iterations=1000, verbose=True)
    trainer.save_meta_model(DYNAMIC_ALPACA_MODEL_FILE)


# ==================== Stage 3: 평가 ====================

def _make_disturbance(args):
    """CLI args에서 DisturbanceProfile 생성."""
    noise_val = getattr(args, "noise", 0.0)
    dist_type = getattr(args, "disturbance", "combined")
    if noise_val <= 0.0:
        return None
    return create_disturbance(
        disturbance_type=dist_type,
        intensity=noise_val,
        seed=getattr(args, "seed", 42),
        sim_duration=getattr(args, "duration", 20.0),
    )


def _make_dynamic_world(args, disturbance=None):
    """DynamicWorld 생성 (외란 포함)."""
    return DynamicWorld(
        **DYNAMIC_WORLD_PARAMS,
        disturbance=disturbance,
    )


def evaluate(args):
    """4-Way (perturbed) 또는 5-Way (dynamic) 비교 평가."""
    world_type = getattr(args, "world", "perturbed")
    use_dynamic = (world_type == "dynamic")

    noise_val = getattr(args, "noise", 0.0)
    dist_type = getattr(args, "disturbance", "combined")

    n_controllers = 10 if use_dynamic else 4
    print("\n" + "=" * 80)
    print(f"Stage 3: Evaluation ({n_controllers}-Way Comparison, {world_type})".center(80))
    print("=" * 80)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Seed:       {args.seed}")
    print(f"  World:      {world_type}")
    if noise_val > 0.0:
        print(f"  Disturbance: {dist_type} (intensity={noise_val:.2f})")
    print("=" * 80)

    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params_3d = create_mppi_params()
    trajectory_fn = create_trajectory_function(args.trajectory)

    # 모델 파일 존재 확인
    neural_file = DYNAMIC_NEURAL_MODEL_FILE if use_dynamic else NEURAL_MODEL_FILE
    residual_file = DYNAMIC_RESIDUAL_MODEL_FILE if use_dynamic else RESIDUAL_MODEL_FILE
    neural_path = os.path.join(MODEL_DIR, neural_file)
    residual_path = os.path.join(MODEL_DIR, residual_file)

    if not os.path.exists(neural_path) or not os.path.exists(residual_path):
        print(f"\n  Error: Model files not found.")
        print(f"    Neural:   {neural_path} {'(found)' if os.path.exists(neural_path) else '(MISSING)'}")
        print(f"    Residual: {residual_path} {'(found)' if os.path.exists(residual_path) else '(MISSING)'}")
        print("  Run with --collect-data --train first.")
        return

    cost_fn_3d = create_angle_aware_cost(params_3d)

    all_histories = {}
    all_metrics = {}

    if use_dynamic:
        # ---- Dynamic World: 5-Way 비교 ----
        params_5d = create_5d_mppi_params()
        cost_fn_5d = create_5d_angle_aware_cost(params_5d)

        # 외란 프로필 (각 world마다 동일한 외란 적용)
        disturbance_factory = lambda: _make_disturbance(args)

        # 1. Kinematic (3D, 관성/마찰 모름)
        print(f"\n  [1/{n_controllers}] Kinematic (Mismatched, 3D)...")
        kin_controller = MPPIController(base_model, params_3d, cost_function=cost_fn_3d)
        world_kin = _make_dynamic_world(args, disturbance_factory())
        all_histories["Kinematic"] = run_with_dynamic_world(
            kin_controller, world_kin, params_3d, trajectory_fn, args.duration, args.seed, is_5d=False
        )
        all_metrics["Kinematic"] = compute_metrics(all_histories["Kinematic"])
        print(f"        RMSE: {all_metrics['Kinematic']['position_rmse']:.4f}m")

        # 2. Neural (3D, 학습)
        print(f"  [2/{n_controllers}] Neural (End-to-End, 3D)...")
        neural_model = NeuralDynamics(state_dim=3, control_dim=2, model_path=neural_path)
        neural_controller = MPPIController(neural_model, params_3d, cost_function=cost_fn_3d)
        world_neural = _make_dynamic_world(args, disturbance_factory())
        all_histories["Neural"] = run_with_dynamic_world(
            neural_controller, world_neural, params_3d, trajectory_fn, args.duration, args.seed, is_5d=False
        )
        all_metrics["Neural"] = compute_metrics(all_histories["Neural"])
        print(f"        RMSE: {all_metrics['Neural']['position_rmse']:.4f}m")

        # 3. Residual (3D, 물리+NN)
        print(f"  [3/{n_controllers}] Residual (Hybrid, 3D)...")
        residual_nn = NeuralDynamics(state_dim=3, control_dim=2, model_path=residual_path)
        residual_model = ResidualDynamics(
            base_model=DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
            learned_model=residual_nn,
        )
        residual_controller = MPPIController(residual_model, params_3d, cost_function=cost_fn_3d)
        world_residual = _make_dynamic_world(args, disturbance_factory())
        all_histories["Residual"] = run_with_dynamic_world(
            residual_controller, world_residual, params_3d, trajectory_fn, args.duration, args.seed, is_5d=False
        )
        all_metrics["Residual"] = compute_metrics(all_histories["Residual"])
        print(f"        RMSE: {all_metrics['Residual']['position_rmse']:.4f}m")

        # 4. Dynamic (5D, 구조 알지만 파라미터 틀림)
        print(f"  [4/{n_controllers}] Dynamic (5D, mismatched params: c_v=0.1)...")
        dynamic_model = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
        dynamic_controller = MPPIController(dynamic_model, params_5d, cost_function=cost_fn_5d)
        world_dynamic = _make_dynamic_world(args, disturbance_factory())
        all_histories["Dynamic"] = run_with_dynamic_world(
            dynamic_controller, world_dynamic, params_5d, trajectory_fn, args.duration, args.seed, is_5d=True
        )
        all_metrics["Dynamic"] = compute_metrics(all_histories["Dynamic"])
        print(f"        RMSE: {all_metrics['Dynamic']['position_rmse']:.4f}m")

        # 5. MAML (3D, Residual MAML 적응)
        maml_path = os.path.join(MODEL_DIR, DYNAMIC_MAML_MODEL_FILE)
        if os.path.exists(maml_path):
            print(f"  [5/{n_controllers}] MAML (Residual-Adaptive, 3D)...")
            maml_model = MAMLDynamics(
                state_dim=3, control_dim=2,
                model_path=maml_path, inner_lr=0.005, inner_steps=100,
            )
            maml_model.save_meta_weights()
            world_maml = _make_dynamic_world(args, disturbance_factory())
            all_histories["MAML"] = run_with_dynamic_world_maml(
                None, maml_model, world_maml, params_3d,
                trajectory_fn, args.duration, args.seed,
            )
            all_metrics["MAML"] = compute_metrics(all_histories["MAML"])
            print(f"        RMSE: {all_metrics['MAML']['position_rmse']:.4f}m")
        else:
            print(f"  [5/{n_controllers}] MAML — SKIPPED (no meta model, run --meta-train)")

        # 6. MAML-5D (5D, Residual MAML 적응)
        maml_5d_path = os.path.join(MODEL_DIR, DYNAMIC_MAML_5D_MODEL_FILE)
        if os.path.exists(maml_5d_path):
            print(f"  [6/{n_controllers}] MAML-5D (Residual-Adaptive, 5D)...")
            maml_model_5d = MAMLDynamics(
                state_dim=5, control_dim=2,
                model_path=maml_5d_path, inner_lr=0.005, inner_steps=100,
            )
            maml_model_5d.save_meta_weights()
            world_maml_5d = _make_dynamic_world(args, disturbance_factory())
            all_histories["MAML-5D"] = run_with_dynamic_world_maml_5d(
                maml_model_5d, world_maml_5d, params_5d,
                trajectory_fn, args.duration, args.seed,
            )
            all_metrics["MAML-5D"] = compute_metrics(all_histories["MAML-5D"])
            print(f"        RMSE: {all_metrics['MAML-5D']['position_rmse']:.4f}m")
        else:
            print(f"  [6/{n_controllers}] MAML-5D — SKIPPED (no 5D meta model, run --meta-train-5d)")

        # 7. EKF (5D, 파라미터 추정)
        print(f"  [7/{n_controllers}] EKF (Parameter Estimation, 5D)...")
        world_ekf = _make_dynamic_world(args, disturbance_factory())
        all_histories["EKF"] = run_with_dynamic_world_ekf(
            world_ekf, params_5d, trajectory_fn, args.duration, args.seed,
        )
        all_metrics["EKF"] = compute_metrics(all_histories["EKF"])
        print(f"        RMSE: {all_metrics['EKF']['position_rmse']:.4f}m")

        # 8. L1 (5D, 외란 추정)
        print(f"  [8/{n_controllers}] L1 (Disturbance Estimation, 5D)...")
        world_l1 = _make_dynamic_world(args, disturbance_factory())
        all_histories["L1"] = run_with_dynamic_world_l1(
            world_l1, params_5d, trajectory_fn, args.duration, args.seed,
        )
        all_metrics["L1"] = compute_metrics(all_histories["L1"])
        print(f"        RMSE: {all_metrics['L1']['position_rmse']:.4f}m")

        # 9. ALPaCA (5D, Bayesian 적응)
        alpaca_path = os.path.join(MODEL_DIR, DYNAMIC_ALPACA_MODEL_FILE)
        if os.path.exists(alpaca_path):
            print(f"  [9/{n_controllers}] ALPaCA (Bayesian Adaptive, 5D)...")
            alpaca_model = ALPaCADynamics(
                state_dim=5, control_dim=2, model_path=alpaca_path,
            )
            world_alpaca = _make_dynamic_world(args, disturbance_factory())
            all_histories["ALPaCA"] = run_with_dynamic_world_alpaca(
                alpaca_model, world_alpaca, params_5d,
                trajectory_fn, args.duration, args.seed,
            )
            all_metrics["ALPaCA"] = compute_metrics(all_histories["ALPaCA"])
            print(f"        RMSE: {all_metrics['ALPaCA']['position_rmse']:.4f}m")
        else:
            print(f"  [9/{n_controllers}] ALPaCA — SKIPPED (no meta model, run --meta-train-alpaca)")

        # 10. Oracle (5D, 정확한 파라미터)
        print(f"  [10/{n_controllers}] Oracle (5D, exact params: c_v=0.5)...")
        oracle_model = DynamicKinematicAdapter(
            c_v=DYNAMIC_WORLD_PARAMS["c_v"],
            c_omega=DYNAMIC_WORLD_PARAMS["c_omega"],
            k_v=DYNAMIC_WORLD_PARAMS["k_v"],
            k_omega=DYNAMIC_WORLD_PARAMS["k_omega"],
        )
        oracle_controller = MPPIController(oracle_model, params_5d, cost_function=cost_fn_5d)
        world_oracle = _make_dynamic_world(args, disturbance_factory())
        all_histories["Oracle"] = run_with_dynamic_world(
            oracle_controller, world_oracle, params_5d, trajectory_fn, args.duration, args.seed, is_5d=True
        )
        all_metrics["Oracle"] = compute_metrics(all_histories["Oracle"])
        print(f"        RMSE: {all_metrics['Oracle']['position_rmse']:.4f}m")

        label_order = ["Kinematic", "Neural", "Residual", "Dynamic"]
        if "MAML" in all_histories:
            label_order.append("MAML")
        if "MAML-5D" in all_histories:
            label_order.append("MAML-5D")
        label_order.append("EKF")
        label_order.append("L1")
        if "ALPaCA" in all_histories:
            label_order.append("ALPaCA")
        label_order.append("Oracle")

    else:
        # ---- Perturbed World: 4-Way 비교 (기존 로직) ----

        # 1. Kinematic (Mismatched)
        print(f"\n  [1/{n_controllers}] Kinematic (Mismatched)...")
        kin_controller = MPPIController(base_model, params_3d, cost_function=cost_fn_3d)
        all_histories["Kinematic"] = run_with_perturbed_world(
            kin_controller, base_model, params_3d, trajectory_fn, args.duration, args.seed
        )
        all_metrics["Kinematic"] = compute_metrics(all_histories["Kinematic"])
        print(f"        RMSE: {all_metrics['Kinematic']['position_rmse']:.4f}m")

        # 2. Neural (End-to-End)
        print(f"  [2/{n_controllers}] Neural (End-to-End)...")
        neural_model = NeuralDynamics(state_dim=3, control_dim=2, model_path=neural_path)
        neural_controller = MPPIController(neural_model, params_3d, cost_function=cost_fn_3d)
        all_histories["Neural"] = run_with_perturbed_world(
            neural_controller, base_model, params_3d, trajectory_fn, args.duration, args.seed
        )
        all_metrics["Neural"] = compute_metrics(all_histories["Neural"])
        print(f"        RMSE: {all_metrics['Neural']['position_rmse']:.4f}m")

        # 3. Residual (Hybrid)
        print(f"  [3/{n_controllers}] Residual (Hybrid)...")
        residual_nn = NeuralDynamics(state_dim=3, control_dim=2, model_path=residual_path)
        residual_model = ResidualDynamics(
            base_model=DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
            learned_model=residual_nn,
        )
        residual_controller = MPPIController(residual_model, params_3d, cost_function=cost_fn_3d)
        all_histories["Residual"] = run_with_perturbed_world(
            residual_controller, base_model, params_3d, trajectory_fn, args.duration, args.seed
        )
        all_metrics["Residual"] = compute_metrics(all_histories["Residual"])
        print(f"        RMSE: {all_metrics['Residual']['position_rmse']:.4f}m")

        # 4. Oracle (Perfect)
        print(f"  [4/{n_controllers}] Oracle (Perfect)...")
        oracle_model = PerturbedDiffDriveModel(v_max=1.0, omega_max=1.0)
        oracle_controller = MPPIController(oracle_model, params_3d, cost_function=cost_fn_3d)
        all_histories["Oracle"] = run_with_perturbed_world(
            oracle_controller, base_model, params_3d, trajectory_fn, args.duration, args.seed
        )
        all_metrics["Oracle"] = compute_metrics(all_histories["Oracle"])
        print(f"        RMSE: {all_metrics['Oracle']['position_rmse']:.4f}m")

        label_order = ["Kinematic", "Neural", "Residual", "Oracle"]

    # ---- 비교 출력 ----
    for label in label_order:
        print_metrics(all_metrics[label], title=label)

    metrics_list = [all_metrics[l] for l in label_order]
    compare_metrics(metrics_list, label_order, title=f"Model Mismatch Comparison ({world_type})")

    # ---- 순서 검증 ----
    print("\n  RMSE ranking:")
    sorted_labels = sorted(label_order, key=lambda l: all_metrics[l]["position_rmse"])
    for i, label in enumerate(sorted_labels):
        rmse = all_metrics[label]["position_rmse"]
        marker = " (best)" if i == 0 else ""
        print(f"    {i+1}. {label}: {rmse:.4f}m{marker}")

    if use_dynamic:
        if noise_val > 0:
            print(f"\n  Expected (noise={noise_val:.1f}): MAML-5D ~ L1 ~ ALPaCA < EKF ~ Dynamic < Kinematic")
        else:
            print("\n  Expected: Oracle < MAML-5D ~ EKF < Dynamic < Kinematic")
    else:
        print("\n  Expected: Oracle < Residual <= Neural << Kinematic")

    # ---- 시각화 ----
    visualize_general(
        args, params_3d, label_order, all_histories, all_metrics, world_type
    )


# ==================== Stage 4: 시각화 ====================

COLORS = {
    "Kinematic": "#e74c3c",   # red
    "Neural": "#3498db",       # blue
    "Residual": "#2ecc71",     # green
    "Dynamic": "#e67e22",      # orange
    "MAML": "#00bcd4",         # cyan
    "MAML-5D": "#ff6f00",      # amber
    "EKF": "#43a047",          # dark green
    "L1": "#d81b60",           # pink
    "ALPaCA": "#8e24aa",       # purple-violet
    "Oracle": "#9b59b6",       # purple
    "Reference": "#7f8c8d",    # gray
}


def _detect_outliers(label_order, all_metrics, threshold=5.0):
    """RMSE가 중앙값의 threshold배 이상인 메소드를 outlier로 판정."""
    rmses = {n: all_metrics[n]["position_rmse"] for n in label_order}
    values = list(rmses.values())
    if len(values) < 3:
        return set()
    median = np.median(values)
    if median < 1e-6:
        return set()
    return {name for name, v in rmses.items() if v > median * threshold}


def visualize_general(args, params, label_order, all_histories, all_metrics, world_type):
    """N-Way 비교 시각화 (4-Way perturbed / 5-Way dynamic)."""
    import matplotlib.pyplot as plt

    print("\nGenerating comparison plots...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    noise_val = getattr(args, "noise", 0.0)
    dist_type = getattr(args, "disturbance", "combined")
    dist_str = f", disturbance={dist_type}@{noise_val:.1f}" if noise_val > 0 else ""

    if world_type == "dynamic":
        subtitle = f"(Dynamic World: c_v={DYNAMIC_WORLD_PARAMS['c_v']}, c_omega={DYNAMIC_WORLD_PARAMS['c_omega']}{dist_str})"
    else:
        subtitle = f"(Perturbed: friction={FRICTION_FACTOR}, bias={ACTUATOR_BIAS})"

    fig.suptitle(
        f"Model Mismatch Comparison - {args.trajectory.capitalize()} {subtitle}",
        fontsize=16, fontweight="bold",
    )

    first_label = label_order[0]

    # 1. XY 궤적 비교
    ax = axes[0, 0]
    refs = all_histories[first_label]["reference"]
    ax.plot(refs[:, 0], refs[:, 1], "--", color=COLORS["Reference"],
            label="Reference", linewidth=2, alpha=0.7)
    for name in label_order:
        states = all_histories[name]["state"]
        ax.plot(states[:, 0], states[:, 1], "-", color=COLORS[name],
                label=name, linewidth=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 2. 위치 오차 시계열 (outlier auto-clip)
    ax = axes[0, 1]
    outliers = _detect_outliers(label_order, all_metrics)
    all_errors = {}
    for name in label_order:
        hist = all_histories[name]
        states = hist["state"]
        ref = hist["reference"]
        errors = np.linalg.norm(states[:, :2] - ref[:, :2], axis=1)
        all_errors[name] = errors
        is_outlier = name in outliers
        rmse = all_metrics[name]["position_rmse"]
        lbl = f"{name} ({rmse:.2f}m)" if is_outlier else name
        ax.plot(hist["time"], errors, "-", color=COLORS[name], label=lbl,
                linewidth=2, alpha=0.3 if is_outlier else 1.0,
                linestyle=":" if is_outlier else "-")
    if outliers:
        non_outlier_max = max(
            all_errors[n].max() for n in label_order if n not in outliers
        ) if any(n not in outliers for n in label_order) else None
        if non_outlier_max is not None:
            ax.set_ylim(top=non_outlier_max * 1.3)
            ax.text(0.98, 0.97, r"$\uparrow$ outliers clipped",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="gray", style="italic")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. RMSE 바 차트 (outlier auto-clip)
    ax = axes[0, 2]
    rmses = [all_metrics[n]["position_rmse"] for n in label_order]
    bar_colors = [COLORS[n] for n in label_order]
    non_outlier_rmses = [r for n, r in zip(label_order, rmses) if n not in outliers]
    y_max = max(non_outlier_rmses) * 1.5 if (outliers and non_outlier_rmses) else None
    display_rmses = [min(r, y_max) if (y_max and n in outliers) else r
                     for n, r in zip(label_order, rmses)]
    hatches = ["///" if n in outliers else "" for n in label_order]
    bars = ax.bar(label_order, display_rmses, color=bar_colors, alpha=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for bar, rmse, name in zip(bars, rmses, label_order):
        if name in outliers:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"\u2191{rmse:.2f}m", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="red")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{rmse:.4f}m", ha="center", va="bottom", fontsize=9)
    if y_max:
        ax.set_ylim(top=y_max)
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("RMSE Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    if len(label_order) >= 5:
        ax.tick_params(axis="x", labelsize=8)

    # 4. 제어 입력 비교 (v only)
    ax = axes[1, 0]
    for name in label_order:
        hist = all_histories[name]
        controls = hist["control"]
        ax.plot(hist["time"], controls[:, 0], "-", color=COLORS[name],
                label=f"{name} v", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_title("Control Input (Linear Velocity)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. 계산 시간 비교
    ax = axes[1, 1]
    for name in label_order:
        hist = all_histories[name]
        solve_times_ms = hist["solve_time"] * 1000
        ax.plot(hist["time"], solve_times_ms, "-", color=COLORS[name],
                label=name, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. 요약 텍스트
    ax = axes[1, 2]
    ax.axis("off")

    n_way = len(label_order)
    lines = [
        f"  {n_way}-Way Model Mismatch Comparison ({world_type})",
        f"  {'=' * 48}",
        "",
    ]

    if world_type == "dynamic":
        lines += [
            f"  Dynamic World Parameters:",
            f"    c_v={DYNAMIC_WORLD_PARAMS['c_v']}, c_omega={DYNAMIC_WORLD_PARAMS['c_omega']}",
            f"    k_v={DYNAMIC_WORLD_PARAMS['k_v']}, k_omega={DYNAMIC_WORLD_PARAMS['k_omega']}",
            f"    noise={DYNAMIC_WORLD_PARAMS['process_noise_std']}",
        ]
        if noise_val > 0:
            lines.append(f"    disturbance: {dist_type} @ {noise_val:.2f}")
    else:
        lines += [
            f"  Perturbation Parameters:",
            f"    Friction factor:    {FRICTION_FACTOR}",
            f"    Heading friction:   {HEADING_FRICTION}",
            f"    Actuator bias:      {ACTUATOR_BIAS}",
            f"    Process noise:      {PROCESS_NOISE_STD}",
        ]

    lines += ["", "  Results (Position RMSE):"]
    for label in label_order:
        rmse = all_metrics[label]["position_rmse"]
        solve_ms = all_metrics[label]["mean_solve_time"]
        lines.append(f"    {label:20s}: {rmse:.4f} m  ({solve_ms:.1f} ms)")

    lines += [
        "",
        f"  Config:",
        f"    Trajectory: {args.trajectory}",
        f"    Duration:   {args.duration}s",
        f"    Samples:    K={params.K}, N={params.N}",
    ]

    summary_text = "\n".join(lines)
    ax.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    suffix = f"_{world_type}" if world_type == "dynamic" else ""
    if noise_val > 0:
        suffix += f"_noise{noise_val:.1f}"
    save_path = os.path.join(PLOT_DIR, f"model_mismatch_comparison_{args.trajectory}{suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save_path}")
    plt.close()


# ==================== Live 실시간 비교 ====================

def run_live_comparison(args):
    """
    N-Way 실시간 비교 애니메이션.

    FuncAnimation으로 프레임마다 컨트롤러들을 전진시키며
    XY 궤적, 위치 오차, 제어 입력, RMSE 바 차트를 실시간 갱신.

    --world perturbed: 4-Way (Kinematic/Neural/Residual/Oracle)
    --world dynamic:   6-Way (Kinematic/Neural/Residual/Dynamic/MAML/Oracle)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from collections import deque

    world_type = getattr(args, "world", "perturbed")
    use_dynamic = (world_type == "dynamic")
    n_way = 10 if use_dynamic else 4

    print("\n" + "=" * 80)
    print(f"Live Comparison ({n_way}-Way, {world_type}, Real-Time)".center(80))
    print("=" * 80)
    noise_val = getattr(args, "noise", 0.0)
    dist_type = getattr(args, "disturbance", "combined")

    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Seed:       {args.seed}")
    print(f"  World:      {world_type}")
    if noise_val > 0.0:
        print(f"  Disturbance: {dist_type} (intensity={noise_val:.2f})")
    print("=" * 80)

    base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params_3d = create_mppi_params()
    trajectory_fn = create_trajectory_function(args.trajectory)

    # 모델 파일 존재 확인 — 없으면 자동으로 데이터 수집 + 학습
    neural_file = DYNAMIC_NEURAL_MODEL_FILE if use_dynamic else NEURAL_MODEL_FILE
    residual_file = DYNAMIC_RESIDUAL_MODEL_FILE if use_dynamic else RESIDUAL_MODEL_FILE
    neural_path = os.path.join(MODEL_DIR, neural_file)
    residual_path = os.path.join(MODEL_DIR, residual_file)

    if not os.path.exists(neural_path) or not os.path.exists(residual_path):
        print(f"\n  Model files not found — auto-running collect-data + train...")
        collect_data(args)
        train_models(args)

    # 공통 모델 생성
    neural_model = NeuralDynamics(state_dim=3, control_dim=2, model_path=neural_path)
    residual_nn = NeuralDynamics(state_dim=3, control_dim=2, model_path=residual_path)
    residual_model = ResidualDynamics(
        base_model=DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
        learned_model=residual_nn,
    )
    cost_fn_3d = create_angle_aware_cost(params_3d)

    # 컨트롤러 + 메타데이터
    controllers = {}
    is_5d_map = {}
    params_map = {}

    controllers["Kinematic"] = MPPIController(base_model, params_3d, cost_function=cost_fn_3d)
    controllers["Neural"] = MPPIController(neural_model, params_3d, cost_function=cost_fn_3d)
    controllers["Residual"] = MPPIController(residual_model, params_3d, cost_function=cost_fn_3d)
    is_5d_map["Kinematic"] = False
    is_5d_map["Neural"] = False
    is_5d_map["Residual"] = False
    params_map["Kinematic"] = params_3d
    params_map["Neural"] = params_3d
    params_map["Residual"] = params_3d

    if use_dynamic:
        params_5d = create_5d_mppi_params()
        cost_fn_5d = create_5d_angle_aware_cost(params_5d)

        dynamic_model = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
        controllers["Dynamic"] = MPPIController(dynamic_model, params_5d, cost_function=cost_fn_5d)
        is_5d_map["Dynamic"] = True
        params_map["Dynamic"] = params_5d

        # MAML (3D, Residual MAML 아키텍처 — 2-Phase)
        maml_path = os.path.join(MODEL_DIR, DYNAMIC_MAML_MODEL_FILE)
        maml_model_live = None
        maml_residual_controller = None
        maml_kin_controller = None
        if not os.path.exists(maml_path):
            print(f"\n  MAML meta model not found — auto-running meta-train...")
            meta_train_maml(args)
        if os.path.exists(maml_path):
            maml_model_live = MAMLDynamics(
                state_dim=3, control_dim=2,
                model_path=maml_path, inner_lr=0.005, inner_steps=100,
            )
            maml_model_live.save_meta_weights()
            # Phase 1: 기구학 컨트롤러 (warmup)
            maml_kin_controller = MPPIController(
                DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
                params_3d, cost_function=cost_fn_3d,
            )
            # Phase 2: ResidualDynamics(kinematic + MAML)
            maml_residual_model = ResidualDynamics(
                base_model=DifferentialDriveKinematic(v_max=1.0, omega_max=1.0),
                learned_model=maml_model_live,
                use_residual=True,
            )
            maml_residual_controller = MPPIController(
                maml_residual_model, params_3d, cost_function=cost_fn_3d,
            )
            # Phase 1 시작: 기구학 컨트롤러로 초기화
            controllers["MAML"] = maml_kin_controller
            is_5d_map["MAML"] = False
            params_map["MAML"] = params_3d

        # MAML-5D (5D, Residual MAML 아키텍처 — 2-Phase)
        maml_5d_path = os.path.join(MODEL_DIR, DYNAMIC_MAML_5D_MODEL_FILE)
        maml_model_5d_live = None
        maml_5d_residual_controller = None
        maml_5d_adapter_controller = None
        if not os.path.exists(maml_5d_path):
            print(f"\n  MAML-5D meta model not found — auto-running meta-train-5d...")
            meta_train_maml_5d(args)
        if os.path.exists(maml_5d_path):
            maml_model_5d_live = MAMLDynamics(
                state_dim=5, control_dim=2,
                model_path=maml_5d_path, inner_lr=0.005, inner_steps=100,
            )
            maml_model_5d_live.save_meta_weights()
            base_5d_wrong = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
            maml_5d_adapter_controller = MPPIController(
                base_5d_wrong, params_5d, cost_function=cost_fn_5d,
            )
            maml_5d_residual_model = ResidualDynamics(
                base_model=base_5d_wrong,
                learned_model=maml_model_5d_live,
                use_residual=True,
            )
            maml_5d_residual_controller = MPPIController(
                maml_5d_residual_model, params_5d, cost_function=cost_fn_5d,
            )
            controllers["MAML-5D"] = maml_5d_adapter_controller
            is_5d_map["MAML-5D"] = True
            params_map["MAML-5D"] = params_5d

        # EKF (5D, 파라미터 추정 — 1-Phase)
        ekf_model_live = EKFAdaptiveDynamics(
            c_v_init=0.1, c_omega_init=0.1, k_v=5.0, k_omega=5.0,
        )
        controllers["EKF"] = MPPIController(ekf_model_live, params_5d, cost_function=cost_fn_5d)
        is_5d_map["EKF"] = True
        params_map["EKF"] = params_5d

        # L1 (5D, 외란 추정 — 1-Phase)
        l1_model_live = L1AdaptiveDynamics(
            c_v_nom=0.1, c_omega_nom=0.1, k_v=5.0, k_omega=5.0,
        )
        controllers["L1"] = MPPIController(l1_model_live, params_5d, cost_function=cost_fn_5d)
        is_5d_map["L1"] = True
        params_map["L1"] = params_5d

        # ALPaCA (5D, Bayesian — 2-Phase)
        alpaca_path = os.path.join(MODEL_DIR, DYNAMIC_ALPACA_MODEL_FILE)
        alpaca_model_live = None
        alpaca_residual_controller = None
        alpaca_adapter_controller = None
        if not os.path.exists(alpaca_path):
            print(f"\n  ALPaCA meta model not found — auto-running meta-train-alpaca...")
            meta_train_alpaca(args)
        if os.path.exists(alpaca_path):
            alpaca_model_live = ALPaCADynamics(
                state_dim=5, control_dim=2, model_path=alpaca_path,
            )
            alpaca_base_5d = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
            alpaca_adapter_controller = MPPIController(
                alpaca_base_5d, params_5d, cost_function=cost_fn_5d,
            )
            alpaca_residual_model = ResidualDynamics(
                base_model=alpaca_base_5d,
                learned_model=alpaca_model_live,
                use_residual=True,
            )
            alpaca_residual_controller = MPPIController(
                alpaca_residual_model, params_5d, cost_function=cost_fn_5d,
            )
            controllers["ALPaCA"] = alpaca_adapter_controller
            is_5d_map["ALPaCA"] = True
            params_map["ALPaCA"] = params_5d

        oracle_model = DynamicKinematicAdapter(
            c_v=DYNAMIC_WORLD_PARAMS["c_v"],
            c_omega=DYNAMIC_WORLD_PARAMS["c_omega"],
            k_v=DYNAMIC_WORLD_PARAMS["k_v"],
            k_omega=DYNAMIC_WORLD_PARAMS["k_omega"],
        )
        controllers["Oracle"] = MPPIController(oracle_model, params_5d, cost_function=cost_fn_5d)
        is_5d_map["Oracle"] = True
        params_map["Oracle"] = params_5d
    else:
        oracle_model = PerturbedDiffDriveModel(v_max=1.0, omega_max=1.0)
        controllers["Oracle"] = MPPIController(oracle_model, params_3d, cost_function=cost_fn_3d)
        is_5d_map["Oracle"] = False
        params_map["Oracle"] = params_3d

    dt = params_3d.dt
    num_steps = int(args.duration / dt)
    names = list(controllers.keys())

    # 초기 상태 (모두 동일)
    init_state_3d = trajectory_fn(0.0)[:3].copy()
    np.random.seed(args.seed)

    # 각 컨트롤러별 상태/히스토리
    states = {}
    worlds = {}  # dynamic world 인스턴스 (dynamic 모드)
    for name in names:
        if is_5d_map[name]:
            states[name] = np.array([*init_state_3d, 0.0, 0.0])
        else:
            states[name] = init_state_3d.copy()
        if use_dynamic:
            worlds[name] = _make_dynamic_world(args, _make_disturbance(args))
            worlds[name].reset(init_state_3d)

    for ctrl in controllers.values():
        ctrl.reset()

    # MAML adaptation buffers + phase tracking
    maml_buffer_states = deque(maxlen=50)
    maml_buffer_controls = deque(maxlen=50)
    maml_buffer_next = deque(maxlen=50)
    maml_warmup_steps = 20
    maml_adapt_interval = 20
    maml_adapted = [False]  # mutable for closure
    maml_base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    # MAML-5D adaptation buffers + phase tracking
    maml_5d_buffer_states = deque(maxlen=50)
    maml_5d_buffer_controls = deque(maxlen=50)
    maml_5d_buffer_next = deque(maxlen=50)
    maml_5d_warmup_steps = 10
    maml_5d_adapt_interval = 20
    maml_5d_adapted = [False]
    maml_5d_base = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
    maml_5d_recent_errors = deque(maxlen=5)

    # EKF / L1 previous state tracking
    ekf_prev_state = [None]
    ekf_prev_control = [None]
    l1_prev_state = [None]
    l1_prev_control = [None]

    # ALPaCA adaptation buffers
    alpaca_buffer_states = deque(maxlen=50)
    alpaca_buffer_controls = deque(maxlen=50)
    alpaca_buffer_next = deque(maxlen=50)
    alpaca_warmup_steps = 10
    alpaca_adapt_interval = 20
    alpaca_adapted = [False]
    alpaca_base_5d_live = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)

    data = {
        n: {"xy": [], "times": [], "errors": [], "controls_v": [], "solve_times": []}
        for n in names
    }
    noise_seeds = np.random.randint(0, 100000, size=num_steps)

    # ---- Figure 구성: 2x2 패널 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    dist_info = f", {dist_type}@{noise_val:.1f}" if noise_val > 0 else ""
    if use_dynamic:
        subtitle = f"(Dynamic: c_v={DYNAMIC_WORLD_PARAMS['c_v']}{dist_info})"
    else:
        subtitle = f"(Perturbed: friction={FRICTION_FACTOR})"
    fig.suptitle(
        f"Model Mismatch Live - {args.trajectory.capitalize()} {subtitle}",
        fontsize=14, fontweight="bold",
    )

    ax_xy = axes[0, 0]
    ax_err = axes[0, 1]
    ax_ctrl = axes[1, 0]
    ax_bar = axes[1, 1]

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectory")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    ref_times = np.arange(0, args.duration, dt)
    ref_pts = np.array([trajectory_fn(t) for t in ref_times])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "--", color=COLORS["Reference"],
               label="Reference", linewidth=2, alpha=0.5)

    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Position Tracking Error")
    ax_err.grid(True, alpha=0.3)

    ax_ctrl.set_xlabel("Time (s)")
    ax_ctrl.set_ylabel("Linear Velocity (m/s)")
    ax_ctrl.set_title("Control Input (v)")
    ax_ctrl.grid(True, alpha=0.3)

    ax_bar.set_ylabel("Position RMSE (m)")
    ax_bar.set_title("Running RMSE")
    ax_bar.grid(True, axis="y", alpha=0.3)

    lines_xy = {}
    dots_xy = {}
    lines_err = {}
    lines_ctrl = {}
    for name in names:
        c = COLORS[name]
        lines_xy[name], = ax_xy.plot([], [], color=c, linewidth=2, label=name)
        dots_xy[name], = ax_xy.plot([], [], "o", color=c, markersize=7)
        lines_err[name], = ax_err.plot([], [], color=c, linewidth=2, label=name)
        lines_ctrl[name], = ax_ctrl.plot([], [], color=c, linewidth=1.5, label=name, alpha=0.8)

    ax_xy.legend(fontsize=8)
    ax_err.legend(fontsize=8)
    ax_ctrl.legend(fontsize=8)

    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=11, family="monospace")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def init():
        for name in names:
            lines_xy[name].set_data([], [])
            dots_xy[name].set_data([], [])
            lines_err[name].set_data([], [])
            lines_ctrl[name].set_data([], [])
        return []

    def update(frame):
        if frame >= num_steps:
            return []

        t_current = frame * dt

        for name in names:
            p = params_map[name]
            ref_3d = generate_reference_trajectory(trajectory_fn, t_current, p.N, dt)
            ref = make_5d_reference(ref_3d) if is_5d_map[name] else ref_3d

            # MAML: 2-Phase 적응 (Residual MAML 아키텍처)
            if name == "MAML" and maml_model_live is not None:
                # Phase 1→2 전환: warmup 후 잔차 학습 + 컨트롤러 교체
                if not maml_adapted[0] and frame >= maml_warmup_steps and len(maml_buffer_states) >= maml_warmup_steps:
                    buf_s = np.array(maml_buffer_states)
                    buf_c = np.array(maml_buffer_controls)
                    buf_n = np.array(maml_buffer_next)
                    kin_dots = np.array([
                        maml_base_model.forward_dynamics(s, c) for s, c in zip(buf_s, buf_c)
                    ])
                    kin_next = buf_s + kin_dots * dt
                    residual_target = buf_s + (buf_n - kin_next)
                    maml_model_live.adapt(buf_s, buf_c, residual_target, dt, restore=True, temporal_decay=0.95)
                    controllers["MAML"] = maml_residual_controller
                    maml_residual_controller.reset()
                    maml_adapted[0] = True
                # Phase 2: 주기적 재적응
                elif maml_adapted[0] and frame % maml_adapt_interval == 0:
                    buf_s = np.array(maml_buffer_states)
                    buf_c = np.array(maml_buffer_controls)
                    buf_n = np.array(maml_buffer_next)
                    kin_dots = np.array([
                        maml_base_model.forward_dynamics(s, c) for s, c in zip(buf_s, buf_c)
                    ])
                    kin_next = buf_s + kin_dots * dt
                    residual_target = buf_s + (buf_n - kin_next)
                    maml_model_live.adapt(buf_s, buf_c, residual_target, dt, restore=True, temporal_decay=0.95)

            # MAML-5D: 2-Phase 적응 (5D Residual MAML 아키텍처)
            if name == "MAML-5D" and maml_model_5d_live is not None:
                if not maml_5d_adapted[0] and frame >= maml_5d_warmup_steps and len(maml_5d_buffer_states) >= maml_5d_warmup_steps:
                    buf_s = np.array(maml_5d_buffer_states)
                    buf_c = np.array(maml_5d_buffer_controls)
                    buf_n = np.array(maml_5d_buffer_next)
                    kin_dots = maml_5d_base.forward_dynamics(buf_s, buf_c)
                    kin_next = buf_s + kin_dots * dt
                    residual_target = buf_s + (buf_n - kin_next)
                    maml_model_5d_live.adapt(buf_s, buf_c, residual_target, dt, restore=True, temporal_decay=0.95)
                    controllers["MAML-5D"] = maml_5d_residual_controller
                    maml_5d_residual_controller.reset()
                    maml_5d_adapted[0] = True
                elif maml_5d_adapted[0]:
                    avg_err = np.mean(maml_5d_recent_errors) if maml_5d_recent_errors else 0.0
                    if frame % maml_5d_adapt_interval == 0 or avg_err > 0.15:
                        buf_s = np.array(maml_5d_buffer_states)
                        buf_c = np.array(maml_5d_buffer_controls)
                        buf_n = np.array(maml_5d_buffer_next)
                        kin_dots = maml_5d_base.forward_dynamics(buf_s, buf_c)
                        kin_next = buf_s + kin_dots * dt
                        residual_target = buf_s + (buf_n - kin_next)
                        maml_model_5d_live.adapt(buf_s, buf_c, residual_target, dt, restore=True, temporal_decay=0.95)

            # ALPaCA: 2-Phase Bayesian 적응
            if name == "ALPaCA" and alpaca_model_live is not None:
                if not alpaca_adapted[0] and frame >= alpaca_warmup_steps and len(alpaca_buffer_states) >= alpaca_warmup_steps:
                    buf_s = np.array(alpaca_buffer_states)
                    buf_c = np.array(alpaca_buffer_controls)
                    buf_n = np.array(alpaca_buffer_next)
                    kin_dots = alpaca_base_5d_live.forward_dynamics(buf_s, buf_c)
                    kin_next = buf_s + kin_dots * dt
                    residual_target = buf_s + (buf_n - kin_next)
                    alpaca_model_live.adapt(buf_s, buf_c, residual_target, dt, restore=True)
                    controllers["ALPaCA"] = alpaca_residual_controller
                    alpaca_residual_controller.reset()
                    alpaca_adapted[0] = True
                elif alpaca_adapted[0] and frame % alpaca_adapt_interval == 0:
                    buf_s = np.array(alpaca_buffer_states)
                    buf_c = np.array(alpaca_buffer_controls)
                    buf_n = np.array(alpaca_buffer_next)
                    kin_dots = alpaca_base_5d_live.forward_dynamics(buf_s, buf_c)
                    kin_next = buf_s + kin_dots * dt
                    residual_target = buf_s + (buf_n - kin_next)
                    alpaca_model_live.adapt(buf_s, buf_c, residual_target, dt, restore=True)

            t_start = time.time()
            control, info = controllers[name].compute_control(states[name], ref)
            solve_time = time.time() - t_start

            # 3D 위치로 오차 계산
            state_3d = states[name][:3]
            ref_pt = ref_3d[0, :2]
            error = float(np.linalg.norm(state_3d[:2] - ref_pt))

            data[name]["xy"].append(state_3d[:2].copy())
            data[name]["times"].append(t_current)
            data[name]["errors"].append(error)
            data[name]["controls_v"].append(float(control[0]))
            data[name]["solve_times"].append(solve_time * 1000)

            # Step
            prev_state = states[name].copy()  # save before step
            np.random.seed(noise_seeds[frame])
            if use_dynamic:
                obs_3d = worlds[name].step(control, dt, add_noise=True)
                if is_5d_map[name]:
                    states[name] = worlds[name].get_full_state()
                else:
                    states[name] = obs_3d.copy()

                # MAML: 전이 데이터 버퍼에 기록
                if name == "MAML" and maml_model_live is not None:
                    maml_buffer_states.append(state_3d.copy())
                    maml_buffer_controls.append(control.copy())
                    maml_buffer_next.append(obs_3d.copy())

                # MAML-5D: 5D 전이 데이터 버퍼에 기록
                if name == "MAML-5D" and maml_model_5d_live is not None:
                    maml_5d_buffer_states.append(prev_state.copy())
                    maml_5d_buffer_controls.append(control.copy())
                    maml_5d_buffer_next.append(worlds[name].get_full_state().copy())
                    maml_5d_recent_errors.append(error)

                # EKF: 매 스텝 업데이트
                if name == "EKF" and ekf_prev_state[0] is not None:
                    ekf_model_live.update_step(
                        ekf_prev_state[0], ekf_prev_control[0], prev_state, dt
                    )
                if name == "EKF":
                    ekf_prev_state[0] = prev_state.copy()
                    ekf_prev_control[0] = control.copy()

                # L1: 매 스텝 업데이트
                if name == "L1" and l1_prev_state[0] is not None:
                    l1_model_live.update_step(
                        l1_prev_state[0], l1_prev_control[0], prev_state, dt
                    )
                if name == "L1":
                    l1_prev_state[0] = prev_state.copy()
                    l1_prev_control[0] = control.copy()

                # ALPaCA: 5D 전이 데이터 버퍼에 기록
                if name == "ALPaCA" and alpaca_model_live is not None:
                    alpaca_buffer_states.append(prev_state.copy())
                    alpaca_buffer_controls.append(control.copy())
                    alpaca_buffer_next.append(worlds[name].get_full_state().copy())
            else:
                next_state = perturbed_step(states[name], control, base_model, dt, add_noise=True)
                states[name] = next_state

        # 플롯 업데이트
        for name in names:
            xy = np.array(data[name]["xy"])
            times = np.array(data[name]["times"])
            lines_xy[name].set_data(xy[:, 0], xy[:, 1])
            dots_xy[name].set_data([xy[-1, 0]], [xy[-1, 1]])
            lines_err[name].set_data(times, data[name]["errors"])
            lines_ctrl[name].set_data(times, data[name]["controls_v"])

        all_pts = []
        for name in names:
            if data[name]["xy"]:
                all_pts.extend(data[name]["xy"])
        if all_pts:
            all_pts_arr = np.array(all_pts)
            margin = 1.5
            ax_xy.set_xlim(all_pts_arr[:, 0].min() - margin, all_pts_arr[:, 0].max() + margin)
            ax_xy.set_ylim(all_pts_arr[:, 1].min() - margin, all_pts_arr[:, 1].max() + margin)

        for ax in [ax_err, ax_ctrl]:
            ax.relim()
            ax.autoscale_view()

        if frame % 20 == 0 or frame == num_steps - 1:
            ax_bar.clear()
            ax_bar.set_ylabel("Position RMSE (m)")
            ax_bar.set_title("Running RMSE")
            ax_bar.grid(True, axis="y", alpha=0.3)
            rmses = []
            bar_names = []
            bar_colors = []
            for name in names:
                errs = data[name]["errors"]
                if errs:
                    rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
                    rmses.append(rmse)
                    bar_names.append(name)
                    bar_colors.append(COLORS[name])
            if rmses:
                # Outlier auto-clip for live bar chart
                live_median = float(np.median(rmses))
                live_outlier_threshold = live_median * 5.0 if live_median > 1e-6 else None
                live_outlier_set = set()
                if live_outlier_threshold and len(rmses) >= 3:
                    live_outlier_set = {n for n, r in zip(bar_names, rmses) if r > live_outlier_threshold}
                non_outlier_rmses_live = [r for n, r in zip(bar_names, rmses) if n not in live_outlier_set]
                y_max_live = max(non_outlier_rmses_live) * 1.5 if (live_outlier_set and non_outlier_rmses_live) else None
                display_rmses_live = [min(r, y_max_live) if (y_max_live and n in live_outlier_set) else r
                                      for n, r in zip(bar_names, rmses)]
                bars = ax_bar.bar(bar_names, display_rmses_live, color=bar_colors, alpha=0.8)
                for bar, rmse, bname in zip(bars, rmses, bar_names):
                    if bname in live_outlier_set:
                        bar.set_hatch("///")
                        ax_bar.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height(),
                            f"\u2191{rmse:.2f}",
                            ha="center", va="bottom", fontsize=9,
                            fontweight="bold", color="red",
                        )
                    else:
                        ax_bar.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height(),
                            f"{rmse:.4f}",
                            ha="center", va="bottom", fontsize=9,
                        )
                if y_max_live:
                    ax_bar.set_ylim(top=y_max_live)

        time_text.set_text(f"t = {t_current:.1f}s / {args.duration:.0f}s")
        return []

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=num_steps, interval=1, blit=False, repeat=False,
    )
    plt.show()
    print("\nLive comparison finished.")


# ==================== main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Model Mismatch Comparison Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Perturbed world (기본, 4-way)
  python model_mismatch_comparison_demo.py --all
  python model_mismatch_comparison_demo.py --live --trajectory circle --duration 20

  # Dynamic world (6-way: Kinematic/Neural/Residual/Dynamic/MAML/Oracle)
  python model_mismatch_comparison_demo.py --all --world dynamic --trajectory circle --duration 20
  python model_mismatch_comparison_demo.py --live --world dynamic --trajectory circle --duration 20
        """,
    )
    parser.add_argument("--collect-data", action="store_true", help="Stage 1: Collect training data")
    parser.add_argument("--train", action="store_true", help="Stage 2: Train neural/residual models")
    parser.add_argument("--evaluate", action="store_true", help="Stage 3: Run evaluation")
    parser.add_argument("--meta-train", action="store_true", help="Stage 2b: MAML 3D meta-training (dynamic world only)")
    parser.add_argument("--meta-train-5d", action="store_true", help="Stage 2c: MAML 5D meta-training (dynamic world only)")
    parser.add_argument("--meta-train-alpaca", action="store_true", help="Stage 2d: ALPaCA meta-training (dynamic world only)")
    parser.add_argument("--meta-algo", type=str, default="fomaml", choices=["fomaml", "reptile"],
                        help="Meta-learning algorithm (default: fomaml)")
    parser.add_argument("--live", action="store_true", help="Live real-time comparison visualization")
    parser.add_argument("--all", action="store_true", help="Run all stages (collect-data + train + evaluate)")
    parser.add_argument(
        "--world", type=str, default="perturbed",
        choices=["perturbed", "dynamic"],
        help="World type: perturbed (4-way) or dynamic (6-way with DifferentialDriveDynamic) (default: perturbed)",
    )
    parser.add_argument(
        "--trajectory", type=str, default="circle",
        choices=["circle", "figure8", "sine", "slalom", "straight"],
        help="Reference trajectory type (default: circle)",
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Evaluation duration in seconds (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Disturbance intensity 0.0~1.0 (default: 0.0, no disturbance)")
    parser.add_argument("--disturbance", type=str, default="combined",
                        choices=["none", "wind", "terrain", "sine", "combined"],
                        help="Disturbance type (default: combined)")

    args = parser.parse_args()

    # --all이면 전체 파이프라인 실행
    if args.all:
        args.collect_data = True
        args.train = True
        args.evaluate = True
        if args.world == "dynamic":
            args.meta_train = True
            args.meta_train_5d = True
            args.meta_train_alpaca = True

    # 아무 옵션도 없으면 도움말 출력
    if not (args.collect_data or args.train or args.evaluate or args.live or args.meta_train or args.meta_train_5d or args.meta_train_alpaca):
        parser.print_help()
        return

    n_way = 10 if args.world == "dynamic" else 4
    print("\n" + "#" * 80)
    print("#" + f"Model Mismatch Comparison Demo ({n_way}-Way, {args.world})".center(78) + "#")
    print("#" * 80)
    if args.world == "dynamic":
        print(f"\n  Dynamic world: DifferentialDriveDynamic (5D, inertia+friction)")
        print(f"  10-way: Kinematic / Neural / Residual / Dynamic / MAML / MAML-5D / EKF / L1 / ALPaCA / Oracle")
    else:
        print(f"\n  Perturbed world: kinematic + friction, bias, noise")
        print(f"  4-way comparison: Kinematic / Neural / Residual / Oracle")
    print()

    if args.collect_data:
        collect_data(args)

    if args.train:
        train_models(args)

    if args.meta_train:
        meta_train_maml(args)

    if args.meta_train_5d:
        meta_train_maml_5d(args)

    if args.meta_train_alpaca:
        meta_train_alpaca(args)

    if args.evaluate:
        evaluate(args)

    if args.live:
        run_live_comparison(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
