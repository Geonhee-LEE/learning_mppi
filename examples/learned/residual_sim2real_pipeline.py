#!/usr/bin/env python3
"""
Residual Dynamics Sim-to-Real Pipeline (End-to-End Demo)
=========================================================

전체 residual-dynamics 시뮬레이션-투-리얼(sim-to-real) 워크플로우를
기존 저장소 인프라만 사용해 하나의 실행 가능한 스크립트로 시연한다.

파이프라인 단계:
    1. Collect  : "실제 로봇"(perturbed model)에서 탐험 노이즈를 준 제어로
                  궤적을 롤아웃하며 (state, control, next_state, dt) 전이 수집.
    2. Train    : 물리 모델(base) forward_dynamics 대비 residual 타깃
                  (state_dot - physics_dot)을 신경망으로 학습.
    3. Evaluate : 동일한 "실제" 세계(real_model=perturbed)에서
                  Baseline(무지) / Residual(학습 보정) / Oracle(완전 지식)
                  세 컨트롤러의 추적 성능을 정량 비교.
    4. Deploy   : residual 모델을 MPPI에 꽂고, 저장/로드하고, ROS2 노드에
                  매핑하는 배포 골격(skeleton)을 출력.

지원 모델(--model):
    - diffdrive : DifferentialDriveKinematic  (state[x,y,θ], ctrl[v,ω])
    - ackermann : AckermannKinematic          (state[x,y,θ,δ], ctrl[v,φ])
    - swerve    : SwerveDriveKinematic        (state[x,y,θ], ctrl[vx,vy,ω])

실행 예시:
    PYTHONPATH=. python examples/learned/residual_sim2real_pipeline.py \\
        --model diffdrive --episodes 5 --epochs 30 --no-plot
"""

import argparse
import json
import os
import time
from typing import Callable, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 헤드리스: plt.show() 금지, 파일 저장만
import matplotlib.pyplot as plt
import numpy as np

from mppi_controller.models.base_model import RobotModel
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.models.kinematic.swerve_drive_kinematic import SwerveDriveKinematic
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.learning.data_collector import DataCollector
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer
from mppi_controller.learning.model_validator import ModelValidator
from mppi_controller.simulation.harness import SimulationHarness


# ======================================================================
# 1. PerturbedModel — "실제 로봇" 래퍼 (임의의 base 모델/차원에 동작)
# ======================================================================
class PerturbedModel(RobotModel):
    """
    물리 base 모델을 감싸 sim-to-real 갭을 인위적으로 도입한다.

        f_real(x, u) = slip * f_base(x, u) + bias

    - slip (스칼라, ~0.85): 바퀴 미끄러짐/모델 게인 오차를 흉내.
    - bias ((nx,) 벡터): 정렬 오차/드리프트 등 상수 편향.

    이때 학습해야 할 residual 은:
        residual(x, u) = f_real - f_base = (slip - 1) * f_base(x, u) + bias

    step()은 RobotModel 기본 RK4 구현을 그대로 상속하므로,
    forward_dynamics 의 perturbation 이 이산 전이에 자연히 전파된다.
    나머지 인터페이스(state_dim/control_dim/... )는 base 로 위임한다.
    """

    def __init__(self, base_model: RobotModel, slip: float = 0.85, bias: Optional[np.ndarray] = None):
        self.base_model = base_model
        self.slip = float(slip)
        if bias is None:
            # 위치 x,y 에 작은 드리프트, heading 에 더 작은 편향. 나머지 차원은 0.
            bias = np.zeros(base_model.state_dim)
            if base_model.state_dim >= 1:
                bias[0] = 0.03
            if base_model.state_dim >= 2:
                bias[1] = 0.02
            if base_model.state_dim >= 3:
                bias[2] = 0.01
        self.bias = np.asarray(bias, dtype=float)

    @property
    def state_dim(self) -> int:
        return self.base_model.state_dim

    @property
    def control_dim(self) -> int:
        return self.base_model.control_dim

    @property
    def model_type(self) -> str:
        return self.base_model.model_type

    def forward_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        base_dot = self.base_model.forward_dynamics(state, control)
        # bias 는 (nx,) — 배치((B,nx)) 입력에도 브로드캐스트된다.
        return self.slip * base_dot + self.bias

    def get_control_bounds(self):
        return self.base_model.get_control_bounds()

    def state_to_dict(self, state: np.ndarray) -> dict:
        return self.base_model.state_to_dict(state)

    def render_config(self) -> dict:
        return self.base_model.render_config()

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return self.base_model.normalize_state(state)

    def true_residual(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """분석용: 실제 residual = (slip-1)*f_base + bias (신경망 학습 타깃의 정답)."""
        base_dot = self.base_model.forward_dynamics(state, control)
        return (self.slip - 1.0) * base_dot + self.bias


# ======================================================================
# 2. MODEL_REGISTRY — 모델별 factory + Q/Qf/R/sigma + 원형 레퍼런스
# ======================================================================
CIRCLE_RADIUS = 2.0
CIRCLE_OMEGA = 0.5


def _circle_xytheta(t: float) -> Tuple[float, float, float]:
    """반경 R, 각속도 ω 원 궤적의 (x, y, θ). θ는 접선 방향."""
    ang = CIRCLE_OMEGA * t
    x = CIRCLE_RADIUS * np.cos(ang)
    y = CIRCLE_RADIUS * np.sin(ang)
    theta = ang + np.pi / 2.0  # 반시계 접선 heading
    return x, y, theta


def _make_diffdrive():
    base = DifferentialDriveKinematic(v_max=1.5, omega_max=2.0)
    params = MPPIParams(
        N=25, dt=0.05, K=512, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([12.0, 12.0, 1.0]),
        R=np.array([0.05, 0.05]),
        Qf=np.array([24.0, 24.0, 2.0]),
    )

    def traj_fn(t: float) -> np.ndarray:
        x, y, th = _circle_xytheta(t)
        return np.array([x, y, th])

    return base, params, traj_fn


def _make_ackermann():
    base = AckermannKinematic(wheelbase=0.5, v_max=1.5, max_steer=0.6, steer_rate_max=2.0)
    params = MPPIParams(
        N=25, dt=0.05, K=512, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([12.0, 12.0, 1.0, 0.1]),
        R=np.array([0.05, 0.05]),
        Qf=np.array([24.0, 24.0, 2.0, 0.2]),
    )

    def traj_fn(t: float) -> np.ndarray:
        x, y, th = _circle_xytheta(t)
        return np.array([x, y, th, 0.0])  # δ 레퍼런스는 0

    return base, params, traj_fn


def _make_swerve():
    base = SwerveDriveKinematic(vx_max=1.5, vy_max=1.5, omega_max=2.0)
    params = MPPIParams(
        N=25, dt=0.05, K=512, lambda_=1.0,
        sigma=np.array([0.5, 0.5, 0.5]),
        Q=np.array([12.0, 12.0, 1.0]),
        R=np.array([0.05, 0.05, 0.05]),
        Qf=np.array([24.0, 24.0, 2.0]),
    )

    def traj_fn(t: float) -> np.ndarray:
        x, y, th = _circle_xytheta(t)
        return np.array([x, y, th])

    return base, params, traj_fn


MODEL_REGISTRY: Dict[str, Callable] = {
    "diffdrive": _make_diffdrive,
    "ackermann": _make_ackermann,
    "swerve": _make_swerve,
}

# 각도(heading θ) 상태 인덱스 — normalize_state 가 [-π,π] 로 래핑하므로
# 유한차분 (next-state)/dt 계산 시 반드시 언랩(unwrap)해야 한다. (모든 모델에서 θ=index 2)
ANGLE_INDICES: Dict[str, Tuple[int, ...]] = {
    "diffdrive": (2,),
    "ackermann": (2,),  # δ(=3)는 클리핑만 되고 래핑되지 않으므로 제외
    "swerve": (2,),
}


def wrapped_state_dot(states: np.ndarray, next_states: np.ndarray, dt: np.ndarray,
                      angle_indices: Tuple[int, ...]) -> np.ndarray:
    """
    유한차분 state_dot = (next - state)/dt.
    각도 차원은 최소각 차이(atan2(sin Δ, cos Δ))로 언랩하여 ±π 경계에서의
    가짜 대형 미분(spurious huge derivative)을 제거한다.
    """
    delta = next_states - states
    for idx in angle_indices:
        raw = next_states[:, idx] - states[:, idx]
        delta[:, idx] = np.arctan2(np.sin(raw), np.cos(raw))
    return delta / dt[:, None]


def build_reference_fn(traj_fn: Callable[[float], np.ndarray], N: int, dt: float) -> Callable[[float], np.ndarray]:
    """t -> (N+1, nx) 레퍼런스 궤적 함수 생성."""

    def reference_fn(t: float) -> np.ndarray:
        times = np.arange(N + 1) * dt + t
        return np.array([traj_fn(tt) for tt in times])

    return reference_fn


# ======================================================================
# 3. Collect — perturbed(real) 세계에서 탐험 롤아웃하며 전이 수집
# ======================================================================
def collect_data(
    base_model: RobotModel,
    real_model: PerturbedModel,
    params: MPPIParams,
    traj_fn: Callable[[float], np.ndarray],
    episodes: int,
    duration: float,
    seed: int,
) -> DataCollector:
    """
    base 모델을 사용하는 Vanilla MPPI 로 "실제" perturbed 세계를 주행하며
    (state, control, next_state, dt) 전이를 수집한다.

    - 컨트롤러는 base 모델(무지)을 사용 → 실제로 배포될 조건과 일치.
    - 실제 전이는 real_model.step 으로 진행 → sim-to-real 갭 포함.
    - 제어에 탐험 노이즈를 주입해 상태-제어 공간을 넓게 커버.
    """
    nx, nu = base_model.state_dim, base_model.control_dim
    dt = params.dt
    collector = DataCollector(state_dim=nx, control_dim=nu)
    reference_fn = build_reference_fn(traj_fn, params.N, dt)
    steps = int(duration / dt)

    bounds = base_model.get_control_bounds()
    u_lo, u_hi = bounds if bounds is not None else (None, None)

    rng = np.random.default_rng(seed)
    # 탐험 노이즈 스케일: 제어 범위의 ~12%
    if u_hi is not None:
        explore_scale = 0.12 * (u_hi - u_lo) / 2.0
    else:
        explore_scale = 0.15 * np.ones(nu)

    for ep in range(episodes):
        controller = MPPIController(base_model, params)
        controller.reset()
        # 에피소드마다 초기 위상을 살짝 흔들어 다양성 확보
        t0 = ep * 0.7
        state = traj_fn(t0).astype(float)
        # 초기 상태에도 작은 섭동
        state = state + rng.normal(0.0, 0.03, size=nx)

        for k in range(steps):
            t = k * dt
            ref = reference_fn(t)
            control, _ = controller.compute_control(state, ref)
            # 탐험 노이즈 주입
            control = control + rng.normal(0.0, 1.0, size=nu) * explore_scale
            if u_hi is not None:
                control = np.clip(control, u_lo, u_hi)

            next_state = real_model.step(state, control, dt)
            next_state = real_model.normalize_state(next_state)

            collector.add_sample(state.astype(float), control.astype(float),
                                 next_state.astype(float), dt)
            state = next_state

        collector.end_episode()

    return collector


# ======================================================================
# 4. Train — residual 신경망 학습
# ======================================================================
def train_residual(
    base_model: RobotModel,
    collector: DataCollector,
    epochs: int,
    seed: int,
    angle_indices: Tuple[int, ...] = (2,),
) -> Tuple[NeuralNetworkTrainer, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    수집 데이터로 residual = (real_dot - physics_dot) 을 학습.

    핵심: trainer.train() 은 넘겨준 배열을 그대로(정규화 없이) 학습하므로,
          입력/타깃을 미리 정규화해서 넘기고 norm_stats 를 함께 전달한다.
          predict() 는 norm_stats 로 입력 정규화 + 출력 역정규화를 수행한다.

    Returns:
        trainer, history, (held-out) test_states, test_controls, test_residual_targets
    """
    np.random.seed(seed)
    nx, nu = base_model.state_dim, base_model.control_dim

    data = collector.get_data()
    states = data["states"]              # (N, nx)
    controls = data["controls"]          # (N, nu)
    # 각도 언랩된 유한차분 dx/dt (DataCollector 기본값은 θ 래핑으로 인해 오염됨)
    state_dots = wrapped_state_dot(states, data["next_states"], data["dt"], angle_indices)

    # residual 타깃 = 실제 state_dot - 물리 모델 예측 (배치)
    physics_dot = base_model.forward_dynamics(states, controls)  # (N, nx)
    residual_target = state_dots - physics_dot                   # (N, nx)

    # 셔플 후 80/20 분할
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(states))
    states, controls = states[perm], controls[perm]
    residual_target = residual_target[perm]

    n_train = int(0.8 * len(states))
    tr_s, tr_c, tr_y = states[:n_train], controls[:n_train], residual_target[:n_train]
    te_s, te_c, te_y = states[n_train:], controls[n_train:], residual_target[n_train:]

    # 정규화 통계 (학습셋 기준). std 하한으로 0-분산 차원 보호.
    def _stats(arr):
        return arr.mean(axis=0), arr.std(axis=0) + 1e-6

    state_mean, state_std = _stats(tr_s)
    control_mean, control_std = _stats(tr_c)
    dot_mean, dot_std = _stats(tr_y)

    norm_stats = {
        "state_mean": state_mean, "state_std": state_std,
        "control_mean": control_mean, "control_std": control_std,
        "state_dot_mean": dot_mean, "state_dot_std": dot_std,
    }

    # 정규화된 입력/타깃 구성 (train() 이 그대로 학습)
    def _norm_inputs(s, c):
        s_n = (s - state_mean) / state_std
        c_n = (c - control_mean) / control_std
        return np.concatenate([s_n, c_n], axis=1)

    def _norm_targets(y):
        return (y - dot_mean) / dot_std

    train_inputs = _norm_inputs(tr_s, tr_c)
    train_targets = _norm_targets(tr_y)
    val_inputs = _norm_inputs(te_s, te_c)
    val_targets = _norm_targets(te_y)

    trainer = NeuralNetworkTrainer(
        state_dim=nx, control_dim=nu,
        hidden_dims=[128, 128], activation="relu",
        dropout_rate=0.0, learning_rate=1e-3,
        save_dir="models/residual_sim2real",
    )
    history = trainer.train(
        train_inputs, train_targets, val_inputs, val_targets,
        norm_stats=norm_stats, epochs=epochs, batch_size=64,
        early_stopping_patience=max(15, epochs // 3), verbose=False,
    )

    return trainer, history, te_s, te_c, te_y


def make_residual_fn(trainer: NeuralNetworkTrainer) -> Callable:
    """
    trainer.predict 를 ResidualDynamics 가 요구하는 배치-인지 residual_fn 으로 감싼다.
    ((nx,)->(nx,), (B,nx)->(B,nx)) 모두 지원.
    """

    def residual_fn(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return trainer.predict(state, control, denormalize=True)

    return residual_fn


# ======================================================================
# 5. Evaluate — Baseline vs Residual vs Oracle 추적 갭
# ======================================================================
def evaluate_gap(
    base_model: RobotModel,
    residual_model: ResidualDynamics,
    real_model: PerturbedModel,
    params: MPPIParams,
    traj_fn: Callable[[float], np.ndarray],
    duration: float,
    seed: int,
) -> Dict[str, dict]:
    """
    동일한 real_model(perturbed) 세계에서 세 컨트롤러를 비교:
        - Baseline : MPPIController(base)     — perturbation 무지
        - Residual : MPPIController(residual) — 학습 보정
        - Oracle   : MPPIController(real)     — 완전 지식 (상한)
    """
    reference_fn = build_reference_fn(traj_fn, params.N, params.dt)
    x0 = traj_fn(0.0).astype(float)

    harness = SimulationHarness(dt=params.dt, headless=True, seed=seed)
    harness.add_controller(
        "Baseline", MPPIController(base_model, params), base_model,
        color="tab:red", real_model=real_model,
    )
    harness.add_controller(
        "Residual", MPPIController(residual_model, params), residual_model,
        color="tab:blue", real_model=real_model,
    )
    harness.add_controller(
        "Oracle", MPPIController(real_model, params), real_model,
        color="tab:green", real_model=real_model,
    )

    return harness.run(reference_fn, x0, duration)


# ======================================================================
# 6. Deployment skeleton (기본 미실행 — 참고용 출력)
# ======================================================================
def deployment_example(base_model, trainer, norm_stats_available=True):
    """
    배포 골격: (a) residual → MPPI, (b) 저장/로드, (c) ROS2 매핑.
    실제 실행은 하지 않고, 배포 절차를 코드/주석으로 보여준다.
    """
    notes = """
================ DEPLOYMENT NOTES (residual sim-to-real) ================
(a) residual 모델을 MPPI 에 장착:
      residual_fn    = lambda s, u: trainer.predict(s, u, denormalize=True)
      residual_model = ResidualDynamics(base_model, residual_fn=residual_fn)
      controller     = MPPIController(residual_model, params)
      # 이후 controller.compute_control(state, reference) 는 물리+학습 보정 동역학으로 롤아웃.

(b) residual 신경망 저장/로드 (norm_stats 포함):
      trainer.save_model("residual_<model>.pth")   # weights + norm_stats + config
      # 재시작 시:
      #   new = NeuralNetworkTrainer(state_dim, control_dim, hidden_dims=[128,128])
      #   new.load_model("residual_<model>.pth")
      #   residual_fn = lambda s,u: new.predict(s,u, denormalize=True)

(c) ROS2 노드 매핑:
      - ros2/mppi_controller_node.py 의 컨트롤러 생성부에서 base 모델 대신
        ResidualDynamics(base, residual_fn) 를 MPPIController 에 주입하면 된다.
      - norm_stats 는 학습 시점 데이터 분포에 종속 → 배포 로봇의 센서 스케일과
        일치하는지 확인. 분포 이동(distribution shift) 시 온라인 재학습 권장.
      - 실시간 제약: predict() 는 배치 (K*N, nx) 호출이 롤아웃마다 발생하므로
        hidden_dims 를 작게(예: [64,64]) 유지하거나 TorchScript 로 컴파일 권장.
========================================================================
"""
    print(notes)


# ======================================================================
# 결과 출력 / JSON / plot
# ======================================================================
def print_results_table(model_name: str, gap: Dict[str, dict], residual_rmse_1step: float):
    def m(name, key):
        return gap[name]["metrics"][key]

    base_rmse = m("Baseline", "position_rmse")
    res_rmse = m("Residual", "position_rmse")
    orc_rmse = m("Oracle", "position_rmse")
    improve = 100.0 * (base_rmse - res_rmse) / base_rmse if base_rmse > 0 else 0.0

    print("\n" + "=" * 78)
    print(f"  RESIDUAL SIM-TO-REAL RESULTS — model={model_name}".center(78))
    print("=" * 78)
    header = (f"{'Controller':<12} | {'PosRMSE(m)':>11} | {'HeadRMSE(rad)':>13} | "
              f"{'SolveMs':>8}")
    print(header)
    print("-" * 78)
    for name in ["Baseline", "Residual", "Oracle"]:
        print(f"{name:<12} | {m(name,'position_rmse'):>11.4f} | "
              f"{m(name,'heading_rmse'):>13.4f} | {m(name,'mean_solve_time'):>8.2f}")
    print("-" * 78)
    print(f"1-step residual prediction RMSE (held-out): {residual_rmse_1step:.5f}")
    print(f"Residual vs Baseline position-RMSE improvement: {improve:+.1f}%")
    ok = "PASS" if res_rmse < base_rmse else "FAIL"
    print(f"[{ok}] Residual < Baseline : {res_rmse:.4f} < {base_rmse:.4f}")
    print("=" * 78 + "\n")
    return base_rmse, res_rmse, orc_rmse, improve


def save_json(model_name: str, gap, residual_rmse_1step, per_dim_rmse,
              base_rmse, res_rmse, orc_rmse, improve, args):
    out_dir = "results/residual_sim2real"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}.json")

    def metrics_of(name):
        mm = gap[name]["metrics"]
        return {
            "position_rmse": float(mm["position_rmse"]),
            "heading_rmse": float(mm["heading_rmse"]),
            "mean_solve_time_ms": float(mm["mean_solve_time"]),
            "max_position_error": float(mm["max_position_error"]),
        }

    payload = {
        "model": model_name,
        "config": {
            "episodes": args.episodes, "epochs": args.epochs,
            "duration": args.duration, "seed": args.seed,
        },
        "controllers": {n: metrics_of(n) for n in ["Baseline", "Residual", "Oracle"]},
        "residual_1step_rmse": float(residual_rmse_1step),
        "residual_1step_per_dim_rmse": [float(v) for v in per_dim_rmse],
        "baseline_position_rmse": float(base_rmse),
        "residual_position_rmse": float(res_rmse),
        "oracle_position_rmse": float(orc_rmse),
        "improvement_percent": float(improve),
        "residual_helps": bool(res_rmse < base_rmse),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON saved: {path}")
    return path


def save_plot(model_name: str, gap: Dict[str, dict]):
    os.makedirs("plots", exist_ok=True)
    path = f"plots/residual_sim2real_{model_name}.png"

    colors = {"Baseline": "tab:red", "Residual": "tab:blue", "Oracle": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Residual Sim-to-Real — {model_name}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ref = gap["Baseline"]["history"]["reference"]
    ax.plot(ref[:, 0], ref[:, 1], "k--", alpha=0.4, label="Reference")
    for name in ["Baseline", "Residual", "Oracle"]:
        s = gap[name]["history"]["state"]
        ax.plot(s[:, 0], s[:, 1], color=colors[name], linewidth=2, label=name)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories (real/perturbed world)")
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    for name in ["Baseline", "Residual", "Oracle"]:
        h = gap[name]["history"]
        err = np.linalg.norm(h["state"][:, :2] - h["reference"][:, :2], axis=1)
        ax.plot(h["time"], err, color=colors[name], linewidth=2, label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error"); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")
    return path


# ======================================================================
# main
# ======================================================================
def run_pipeline(args):
    model_name = args.model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    print("\n" + "#" * 78)
    print(f"#  RESIDUAL SIM-TO-REAL PIPELINE — {model_name}".ljust(77) + "#")
    print(f"#  episodes={args.episodes} epochs={args.epochs} "
          f"duration={args.duration}s seed={args.seed}".ljust(77) + "#")
    print("#" * 78)

    t_start = time.time()

    base_model, params, traj_fn = MODEL_REGISTRY[model_name]()
    real_model = PerturbedModel(base_model, slip=0.85)

    # --- 1. Collect ---
    print("\n[1/4] Collecting data from perturbed (real) world ...")
    t0 = time.time()
    collector = collect_data(base_model, real_model, params, traj_fn,
                             episodes=args.episodes, duration=args.duration, seed=args.seed)
    print(f"      collected {len(collector)} transitions "
          f"({len(collector.metadata['episodes'])} episodes) in {time.time()-t0:.1f}s")

    # --- 2. Train ---
    print("\n[2/4] Training residual network ...")
    t0 = time.time()
    trainer, history, te_s, te_c, te_y = train_residual(
        base_model, collector, epochs=args.epochs, seed=args.seed,
        angle_indices=ANGLE_INDICES[model_name])
    final_val = history["val_loss"][-1] if history.get("val_loss") else float("nan")
    print(f"      trained {len(history.get('train_loss', []))} epochs "
          f"(final val_loss={final_val:.5f}) in {time.time()-t0:.1f}s")

    residual_fn = make_residual_fn(trainer)
    residual_model = ResidualDynamics(base_model, residual_fn=residual_fn)

    # 1-step residual 예측 RMSE (held-out) via ModelValidator
    validator = ModelValidator()
    val_metrics = validator.evaluate(
        lambda s, c: trainer.predict(s, c, denormalize=True), te_s, te_c, te_y)
    residual_rmse_1step = val_metrics["rmse"]
    per_dim_rmse = val_metrics["per_dim_rmse"]

    # --- 3. Evaluate ---
    print("\n[3/4] Evaluating tracking gap (Baseline vs Residual vs Oracle) ...")
    t0 = time.time()
    gap = evaluate_gap(base_model, residual_model, real_model, params,
                       traj_fn, duration=args.duration, seed=args.seed)
    print(f"      evaluated 3 controllers in {time.time()-t0:.1f}s")

    base_rmse, res_rmse, orc_rmse, improve = print_results_table(
        model_name, gap, residual_rmse_1step)

    # 결과 검증 (모두 유한)
    for name in ["Baseline", "Residual", "Oracle"]:
        pr = gap[name]["metrics"]["position_rmse"]
        assert np.isfinite(pr), f"{name} position_rmse not finite: {pr}"
    assert np.isfinite(residual_rmse_1step), "residual 1-step rmse not finite"

    json_path = save_json(model_name, gap, residual_rmse_1step, per_dim_rmse,
                          base_rmse, res_rmse, orc_rmse, improve, args)

    plot_path = None
    if not args.no_plot:
        plot_path = save_plot(model_name, gap)

    # --- 4. Deploy skeleton ---
    print("\n[4/4] Deployment skeleton:")
    deployment_example(base_model, trainer)

    total = time.time() - t_start
    print(f"Total pipeline runtime ({model_name}): {total:.1f}s")

    return {
        "model": model_name,
        "baseline_rmse": base_rmse,
        "residual_rmse": res_rmse,
        "oracle_rmse": orc_rmse,
        "improvement_percent": improve,
        "residual_1step_rmse": residual_rmse_1step,
        "json": json_path,
        "plot": plot_path,
        "runtime_s": total,
        "residual_helps": res_rmse < base_rmse,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Residual Dynamics Sim-to-Real End-to-End Pipeline")
    parser.add_argument("--model", type=str, default="diffdrive",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_pipeline(args)

    if not summary["residual_helps"]:
        print("\n[WARNING] Residual did NOT improve over Baseline for this run.")


if __name__ == "__main__":
    main()
