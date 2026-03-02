#!/usr/bin/env python3
"""
Learning on the Fly (LotF) 벤치마크: 8-Way 모델 비교

모델:
  1. Kinematic       — 기준선 (모델 미스매치)
  2. Res-NN (MSE)    — MSE 오프라인 학습
  3. Res-NN (MSE+Spec) — MSE + Spectral 정규화
  4. Res-NN (BPTT)   — 궤적 BPTT 학습
  5. Res-LoRA        — MSE pretrain + LoRA 온라인 적응
  6. Res-MAML        — Meta pretrain + SGD 온라인 적응
  7. NN-Policy (BPTT) — MPPI 없이 직접 제어 (BC + BPTT)
  8. Oracle          — 완벽 모델 (upper bound)

시나리오: ee_3d_circle, ee_3d_helix

Usage:
    PYTHONPATH=. python examples/comparison/lotf_benchmark.py
    PYTHONPATH=. python examples/comparison/lotf_benchmark.py --scenario ee_3d_helix
    PYTHONPATH=. python examples/comparison/lotf_benchmark.py --models kinematic,bptt,lora,oracle
    PYTHONPATH=. python examples/comparison/lotf_benchmark.py --duration 10
    PYTHONPATH=. python examples/comparison/lotf_benchmark.py --live --models kinematic,oracle
    PYTHONPATH=. python examples/comparison/lotf_benchmark.py --live --scenario ee_3d_helix --duration 15
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
    MobileManipulator6DOFDynamic,
)
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    EndEffector3DTrackingCost,
    EndEffector3DTerminalCost,
    ControlEffortCost,
    ControlRateCost,
    JointLimitCost,
)
from mppi_controller.utils.trajectory import (
    create_trajectory_function,
    generate_reference_trajectory,
)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SAVE_DIR = "models/learned_models/6dof_benchmark"

ALL_MODELS = [
    "kinematic", "residual_nn", "residual_nn_spectral",
    "bptt", "lora", "maml", "nn_policy", "oracle",
]

MODEL_LABELS = {
    "kinematic": "Kinematic",
    "residual_nn": "Res-NN (MSE)",
    "residual_nn_spectral": "Res-NN (MSE+Spec)",
    "bptt": "Res-NN (BPTT)",
    "lora": "Res-LoRA",
    "maml": "Res-MAML",
    "nn_policy": "NN-Policy (BPTT)",
    "oracle": "Oracle",
}

MODEL_COLORS = {
    "kinematic": "tab:gray",
    "residual_nn": "tab:blue",
    "residual_nn_spectral": "tab:cyan",
    "bptt": "tab:purple",
    "lora": "tab:orange",
    "maml": "tab:red",
    "nn_policy": "tab:green",
    "oracle": "tab:olive",
}

SCENARIOS = ["ee_3d_circle", "ee_3d_helix"]


# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

def create_oracle_residual(kin_model, dyn_model):
    """Oracle residual: dynamic - kinematic."""
    def residual_fn(state, control):
        return dyn_model.forward_dynamics(state, control) - kin_model.forward_dynamics(state, control)
    return residual_fn


def create_model(name, kin_model, dyn_model, model_dir=SAVE_DIR):
    """
    Create planning model by name.

    Returns:
        (planning_model, needs_adapt, adapt_model_or_None)
    """
    if name == "kinematic":
        return kin_model, False, None

    elif name == "oracle":
        oracle_fn = create_oracle_residual(kin_model, dyn_model)
        model = ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn)
        return model, False, None

    elif name == "residual_nn":
        from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
        model_path = os.path.join(model_dir, "nn", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] NN model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = NeuralDynamics(state_dim=9, control_dim=8, model_path=model_path)
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, False, None

    elif name == "residual_nn_spectral":
        from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
        model_path = os.path.join(model_dir, "nn_spectral", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] NN+Spectral model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = NeuralDynamics(state_dim=9, control_dim=8, model_path=model_path)
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, False, None

    elif name == "bptt":
        from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
        model_path = os.path.join(model_dir, "bptt", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] BPTT model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = NeuralDynamics(state_dim=9, control_dim=8, model_path=model_path)
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, False, None

    elif name == "lora":
        from mppi_controller.models.learned.lora_dynamics import LoRADynamics
        model_path = os.path.join(model_dir, "nn", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] LoRA base model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = LoRADynamics(
            state_dim=9, control_dim=8, model_path=model_path,
            lora_rank=4, lora_alpha=1.0, inner_lr=0.01, inner_steps=5,
        )
        learned.save_meta_weights()
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, True, learned

    elif name == "maml":
        from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
        model_path = os.path.join(model_dir, "maml", "maml_meta_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] MAML model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = MAMLDynamics(
            state_dim=9, control_dim=8, model_path=model_path,
            inner_lr=0.01, inner_steps=5,
        )
        learned.save_meta_weights()
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, True, learned

    elif name == "nn_policy":
        from mppi_controller.learning.nn_policy_trainer import NNPolicyTrainer
        model_path = os.path.join(model_dir, "nn_policy", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] NN-Policy model not found: {model_path}, using random policy")
        trainer = NNPolicyTrainer(
            state_dim=9, ee_ref_dim=3, control_dim=8,
            hidden_dims=[128, 128, 64],
            save_dir=os.path.join(model_dir, "nn_policy"),
        )
        if os.path.exists(model_path):
            trainer.load_model("best_model.pth")
        # Return trainer as model; is_nn_policy flag set on trainer.model
        return trainer, False, None

    else:
        raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────────────────────
# MPPI setup
# ─────────────────────────────────────────────────────────────

def create_params(K=512, N=40, dt=0.05):
    """MPPI 파라미터 생성."""
    return MPPIParams(
        N=N, dt=dt, K=K, lambda_=0.3,
        sigma=np.array([0.3, 0.3] + [0.8] * 6),
        Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
        R=np.array([0.1, 0.1] + [0.05] * 6),
        Qf=np.array([20.0, 20.0, 2.0] + [0.2] * 6),
        device="cpu",
    )


def create_cost_fn(fk_model):
    """비용 함수 생성."""
    return CompositeMPPICost([
        EndEffector3DTrackingCost(fk_model, weight=200.0),
        EndEffector3DTerminalCost(fk_model, weight=400.0),
        ControlEffortCost(R=np.array([0.05, 0.05] + [0.02] * 6)),
        ControlRateCost(R_rate=np.array([0.3, 0.3] + [0.1] * 6)),
        JointLimitCost(
            joint_indices=tuple(range(3, 9)),
            joint_limits=((-2.9, 2.9),) * 6,
            weight=5.0,
        ),
    ])


# ─────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────

def run_single(model_name, kin_model, dyn_model, scenario_name,
               K=512, duration=20.0, ema_alpha=0.2, adapt_buffer_size=100,
               adapt_every=5,
               model_dir=SAVE_DIR, seed=42):
    """
    단일 모델 × 시나리오 시뮬레이션.

    Returns:
        history: dict with time, state, control, ee_pos, ee_ref, ee_error, solve_time
    """
    np.random.seed(seed)

    planning_model, needs_adapt, adapt_model = create_model(
        model_name, kin_model, dyn_model, model_dir
    )

    # Check if this is a NN Policy (direct control, no MPPI)
    is_nn_policy = hasattr(planning_model, 'model') and hasattr(planning_model.model, 'is_nn_policy')

    params = create_params(K=K)
    cost_fn = create_cost_fn(kin_model)
    controller = None if is_nn_policy else MPPIController(planning_model, params, cost_function=cost_fn)

    traj_fn = create_trajectory_function(scenario_name)
    dt = params.dt
    n_steps = int(duration / dt)

    history = {
        "time": [], "state": [], "control": [],
        "ee_pos": [], "ee_ref": [], "ee_error": [], "solve_time": [],
    }

    state = np.zeros(9)
    filtered_ctrl = np.zeros(8)

    # Adaptation buffer
    adapt_states = []
    adapt_controls = []
    adapt_dyn_dots = []  # ground truth state_dots from dynamic model

    for step in range(n_steps):
        t = step * dt
        ref = generate_reference_trajectory(traj_fn, t, params.N, dt)

        t0 = time.perf_counter()
        if is_nn_policy:
            # NN Policy: direct control (no MPPI)
            ee_ref_3d = ref[0, :3]
            raw_control = planning_model.compute_control(state, ee_ref_3d)
        else:
            raw_control, _ = controller.compute_control(state, ref)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        filtered_ctrl = ema_alpha * raw_control + (1 - ema_alpha) * filtered_ctrl
        control = filtered_ctrl

        ee_pos = kin_model.forward_kinematics(state)
        ee_ref = ref[0, :3]
        ee_error = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))

        history["time"].append(t)
        history["state"].append(state.copy())
        history["control"].append(control.copy())
        history["ee_pos"].append(ee_pos.copy())
        history["ee_ref"].append(ee_ref.copy())
        history["ee_error"].append(ee_error)
        history["solve_time"].append(solve_ms)

        # Environment step
        next_state = dyn_model.step(state, control, dt)
        next_state = dyn_model.normalize_state(next_state)

        # Online adaptation for LoRA / MAML
        if needs_adapt and adapt_model is not None:
            # Bug fix P0-1: Use forward_dynamics directly (not Euler reverse diff)
            # This avoids RK4 vs Euler numerical mismatch
            dyn_dot = dyn_model.forward_dynamics(
                state.reshape(1, -1), control.reshape(1, -1)
            ).flatten()
            adapt_states.append(state.copy())
            adapt_controls.append(control.copy())
            adapt_dyn_dots.append(dyn_dot.copy())

            # Bug fix P0-2: Larger buffer (100) + adapt every N steps
            if (len(adapt_states) >= adapt_buffer_size
                    and step % adapt_every == 0):
                buf_s = np.array(adapt_states[-adapt_buffer_size:])
                buf_c = np.array(adapt_controls[-adapt_buffer_size:])
                buf_dyn_dots = np.array(adapt_dyn_dots[-adapt_buffer_size:])

                # Residual target = dyn_dot - kin_dot (exact, no Euler)
                kin_dots = kin_model.forward_dynamics(buf_s, buf_c)
                residual_dots = buf_dyn_dots - kin_dots

                # Construct next_states for adapt() interface:
                # adapt() computes target = (next_states - states) / dt
                # So set next_states = states + residual_dots * dt
                residual_ns = buf_s + residual_dots * dt

                # Use temporal decay to weight recent samples more
                adapt_model.adapt(
                    buf_s, buf_c, residual_ns, dt,
                    restore=True,
                    temporal_decay=0.98,
                )

        state = next_state

    for key in history:
        history[key] = np.array(history[key])

    return history


def compute_metrics(history):
    """Compute metrics from history."""
    ee_errors = history["ee_error"]
    solve_times = history["solve_time"]

    return {
        "ee_rmse": float(np.sqrt(np.mean(ee_errors ** 2))),
        "ee_max": float(np.max(ee_errors)),
        "ee_mean": float(np.mean(ee_errors)),
        "solve_ms_mean": float(np.mean(solve_times)),
        "solve_ms_std": float(np.std(solve_times)),
    }


# ─────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────

def run_benchmark(models, scenarios, K=512, duration=20.0, model_dir=SAVE_DIR, seed=42):
    """Run benchmark for all model x scenario combinations."""
    all_results = {}

    kin_model = MobileManipulator6DOFKinematic()
    dyn_model = MobileManipulator6DOFDynamic()

    total = len(models) * len(scenarios)
    idx = 0

    for scenario in scenarios:
        for model_name in models:
            idx += 1
            label = MODEL_LABELS.get(model_name, model_name)
            print(f"\n  [{idx}/{total}] {label} x {scenario}")

            history = run_single(
                model_name, kin_model, dyn_model, scenario,
                K=K, duration=duration, model_dir=model_dir, seed=seed,
            )
            metrics = compute_metrics(history)

            all_results[(model_name, scenario)] = {
                "metrics": metrics,
                "history": history,
            }

            print(f"    RMSE={metrics['ee_rmse']:.4f}m  Max={metrics['ee_max']:.4f}m  "
                  f"Solve={metrics['solve_ms_mean']:.1f}ms")

    return all_results


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_results(all_results, models, scenarios, save_path=None):
    """
    2x3 벤치마크 시각화.

    [0,0] EE 3D 궤적
    [0,1] EE 오차 시계열
    [0,2] RMSE 바 차트
    [1,0] 학습 가능 파라미터 수 비교
    [1,1] 적응 속도 비교 (LoRA vs MAML)
    [1,2] 메트릭 요약 테이블
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_scenarios = len(scenarios)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"LotF Benchmark: {len(models)}-Way Residual Model Comparison",
        fontsize=14, fontweight="bold",
    )

    sc0 = scenarios[0]

    # [0,0] 3D EE Trajectory
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    for model_name in models:
        key = (model_name, sc0)
        if key not in all_results:
            continue
        h = all_results[key]["history"]
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot3D(
            h["ee_pos"][:, 0], h["ee_pos"][:, 1], h["ee_pos"][:, 2],
            "-", color=color, linewidth=1.2, label=label, alpha=0.8,
        )
    first_key = (models[0], sc0)
    if first_key in all_results:
        ref = all_results[first_key]["history"]["ee_ref"]
        ax.plot3D(ref[:, 0], ref[:, 1], ref[:, 2],
                  "k--", linewidth=1.0, alpha=0.5, label="Reference")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"EE 3D Trajectory ({sc0})")
    ax.legend(fontsize=6, loc="upper left")

    # [0,1] EE error time series
    ax = fig.add_subplot(2, 3, 2)
    for model_name in models:
        key = (model_name, sc0)
        if key not in all_results:
            continue
        h = all_results[key]["history"]
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(h["time"], h["ee_error"], "-", color=color,
                linewidth=1.0, label=label, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE Error (m)")
    ax.set_title(f"EE Tracking Error ({sc0})")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # [0,2] RMSE bar chart
    ax = fig.add_subplot(2, 3, 3)
    x = np.arange(len(models))
    width = 0.35
    for si, scenario in enumerate(scenarios):
        rmses = []
        for model_name in models:
            key = (model_name, scenario)
            if key in all_results:
                rmses.append(all_results[key]["metrics"]["ee_rmse"])
            else:
                rmses.append(0.0)
        offset = (si - (n_scenarios - 1) / 2) * width
        bars = ax.bar(x + offset, rmses, width * 0.9, label=scenario, alpha=0.8)
        for bar, val in zip(bars, rmses):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=5)
    labels = [MODEL_LABELS.get(m, m) for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("EE RMSE (m)")
    ax.set_title("EE RMSE Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # [1,0] Trainable parameters comparison
    ax = fig.add_subplot(2, 3, 4)
    param_counts = {
        "kinematic": 0,
        "residual_nn": 27000,
        "residual_nn_spectral": 27000,
        "bptt": 27000,
        "lora": 2700,
        "maml": 27000,
        "nn_policy": 27000,
        "oracle": 0,
    }
    param_vals = [param_counts.get(m, 0) for m in models]
    colors = [MODEL_COLORS.get(m, "gray") for m in models]
    ax.bar(x, param_vals, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("Trainable Parameters")
    ax.grid(True, alpha=0.3, axis="y")

    # [1,1] Solve time
    ax = fig.add_subplot(2, 3, 5)
    for si, scenario in enumerate(scenarios):
        solve_times = []
        for model_name in models:
            key = (model_name, scenario)
            if key in all_results:
                solve_times.append(all_results[key]["metrics"]["solve_ms_mean"])
            else:
                solve_times.append(0.0)
        offset = (si - (n_scenarios - 1) / 2) * width
        ax.bar(x + offset, solve_times, width * 0.9, label=scenario, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # [1,2] Metrics summary table
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")

    lines = [f"{'Model':<18} {'Scenario':<14} {'RMSE':>7} {'MaxErr':>7} {'Solve':>8}"]
    lines.append("-" * 60)
    for scenario in scenarios:
        for model_name in models:
            key = (model_name, scenario)
            if key not in all_results:
                continue
            m = all_results[key]["metrics"]
            label = MODEL_LABELS.get(model_name, model_name)
            lines.append(
                f"{label:<18} {scenario:<14} "
                f"{m['ee_rmse']:>7.4f} {m['ee_max']:>7.4f} {m['solve_ms_mean']:>6.1f}ms"
            )
        lines.append("")

    metrics_text = "\n".join(lines)
    ax.text(
        0.02, 0.5, metrics_text, transform=ax.transAxes,
        fontsize=7, fontfamily="monospace", verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Performance Summary")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")

    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Live animation
# ─────────────────────────────────────────────────────────────

def run_live(models, scenario, K=512, duration=20.0, ema_alpha=0.2,
             adapt_buffer_size=20, model_dir=SAVE_DIR, seed=42):
    """
    실시간 애니메이션: 모든 모델을 한 프레임마다 1 step씩 동시 실행.

    Layout (2x2):
      [0,0] 3D EE 궤적 (모델 겹침 + reference)
      [0,1] EE 오차 시계열
      [1,0] RMSE 바 차트 (실시간 갱신)
      [1,1] 메트릭 텍스트 (실시간 갱신)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    kin_model = MobileManipulator6DOFKinematic()
    dyn_model = MobileManipulator6DOFDynamic()

    dt = 0.05
    n_steps = int(duration / dt)

    # ── Initialize controllers for each model ──
    agents = {}
    for model_name in models:
        np.random.seed(seed)
        planning_model, needs_adapt, adapt_model = create_model(
            model_name, kin_model, dyn_model, model_dir
        )
        is_nn_pol = hasattr(planning_model, 'model') and hasattr(planning_model.model, 'is_nn_policy')
        params = create_params(K=K)
        cost_fn = create_cost_fn(kin_model)
        controller = None if is_nn_pol else MPPIController(planning_model, params, cost_function=cost_fn)

        agents[model_name] = {
            "controller": controller,
            "planning_model": planning_model,
            "is_nn_policy": is_nn_pol,
            "needs_adapt": needs_adapt,
            "adapt_model": adapt_model,
            "state": np.zeros(9),
            "filtered_ctrl": np.zeros(8),
            "adapt_states": [],
            "adapt_controls": [],
            "adapt_next_states": [],
            # Accumulators
            "ee_xs": [], "ee_ys": [], "ee_zs": [],
            "times": [],
            "ee_errors": [],
            "solve_times": [],
        }

    traj_fn = create_trajectory_function(scenario)

    # ── Pre-compute reference for axis limits ──
    ref_preview = np.array([
        generate_reference_trajectory(traj_fn, t * dt, 30, dt)[0, :3]
        for t in range(n_steps)
    ])

    # ── Figure setup ──
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"LotF Live: {len(models)}-Way Comparison ({scenario})",
        fontsize=14, fontweight="bold",
    )

    # [0,0] 3D EE Trajectory
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.set_xlabel("X (m)", fontsize=8)
    ax3d.set_ylabel("Y (m)", fontsize=8)
    ax3d.set_zlabel("Z (m)", fontsize=8)
    ax3d.set_title("EE 3D Trajectory")

    # Reference path (static)
    ax3d.plot3D(ref_preview[:, 0], ref_preview[:, 1], ref_preview[:, 2],
                "k--", linewidth=0.8, alpha=0.4, label="Reference")

    # Set 3D axis limits from reference + margin
    margin = 0.15
    ax3d.set_xlim(ref_preview[:, 0].min() - margin, ref_preview[:, 0].max() + margin)
    ax3d.set_ylim(ref_preview[:, 1].min() - margin, ref_preview[:, 1].max() + margin)
    ax3d.set_zlim(ref_preview[:, 2].min() - margin, ref_preview[:, 2].max() + margin)

    # Line handles per model
    lines_3d = {}
    dots_3d = {}
    for model_name in models:
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)
        (line,) = ax3d.plot3D([], [], [], "-", color=color, linewidth=1.5,
                              label=label, alpha=0.85)
        (dot,) = ax3d.plot3D([], [], [], "o", color=color, markersize=5)
        lines_3d[model_name] = line
        dots_3d[model_name] = dot
    ax3d.legend(fontsize=6, loc="upper left")

    # [0,1] EE Error time series
    ax_err = fig.add_subplot(2, 2, 2)
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("EE Error (m)")
    ax_err.set_title("EE Tracking Error")
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xlim(0, duration)
    ax_err.set_ylim(0, 0.3)

    lines_err = {}
    for model_name in models:
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)
        (line,) = ax_err.plot([], [], "-", color=color, linewidth=1.0,
                              label=label, alpha=0.8)
        lines_err[model_name] = line
    ax_err.legend(fontsize=6, loc="upper right")

    # [1,0] RMSE bar chart (updated each frame)
    ax_bar = fig.add_subplot(2, 2, 3)
    ax_bar.set_title("Current RMSE")
    ax_bar.set_ylabel("EE RMSE (m)")
    ax_bar.grid(True, alpha=0.3, axis="y")
    x_pos = np.arange(len(models))
    bar_colors = [MODEL_COLORS.get(m, "gray") for m in models]
    bar_labels = [MODEL_LABELS.get(m, m) for m in models]
    bars = ax_bar.bar(x_pos, [0.0] * len(models), color=bar_colors, alpha=0.8)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=7)
    ax_bar.set_ylim(0, 0.2)
    bar_texts = []
    for b in bars:
        txt = ax_bar.text(b.get_x() + b.get_width() / 2, 0, "",
                          ha="center", va="bottom", fontsize=7)
        bar_texts.append(txt)

    # [1,1] Metrics text
    ax_txt = fig.add_subplot(2, 2, 4)
    ax_txt.axis("off")
    ax_txt.set_title("Live Metrics")
    metrics_text_obj = ax_txt.text(
        0.05, 0.5, "", transform=ax_txt.transAxes,
        fontsize=8, fontfamily="monospace", verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()

    # ── Animation update ──
    def update(frame):
        if frame >= n_steps:
            return []

        t = frame * dt
        ref = generate_reference_trajectory(traj_fn, t, 40, dt)
        ee_ref = ref[0, :3]

        for model_name in models:
            ag = agents[model_name]
            state = ag["state"]

            # Compute control
            t0 = time.perf_counter()
            if ag["is_nn_policy"]:
                raw_ctrl = ag["planning_model"].compute_control(state, ee_ref)
            else:
                raw_ctrl, _ = ag["controller"].compute_control(state, ref)
            solve_ms = (time.perf_counter() - t0) * 1000.0

            ag["filtered_ctrl"] = ema_alpha * raw_ctrl + (1 - ema_alpha) * ag["filtered_ctrl"]
            control = ag["filtered_ctrl"]

            # EE evaluation
            ee_pos = kin_model.forward_kinematics(state)
            ee_error = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))

            # Accumulate
            ag["ee_xs"].append(ee_pos[0])
            ag["ee_ys"].append(ee_pos[1])
            ag["ee_zs"].append(ee_pos[2])
            ag["times"].append(t)
            ag["ee_errors"].append(ee_error)
            ag["solve_times"].append(solve_ms)

            # Environment step (ground-truth)
            next_state = dyn_model.step(state, control, dt)
            next_state = dyn_model.normalize_state(next_state)

            # Online adaptation
            if ag["needs_adapt"] and ag["adapt_model"] is not None:
                ag["adapt_states"].append(state.copy())
                ag["adapt_controls"].append(control.copy())
                ag["adapt_next_states"].append(next_state.copy())

                if len(ag["adapt_states"]) >= adapt_buffer_size:
                    buf_s = np.array(ag["adapt_states"][-adapt_buffer_size:])
                    buf_c = np.array(ag["adapt_controls"][-adapt_buffer_size:])
                    buf_ns = np.array(ag["adapt_next_states"][-adapt_buffer_size:])

                    kin_dots = kin_model.forward_dynamics(buf_s, buf_c)
                    actual_dots = (buf_ns - buf_s) / dt
                    residual_dots = actual_dots - kin_dots
                    residual_ns = buf_s + residual_dots * dt
                    ag["adapt_model"].adapt(buf_s, buf_c, residual_ns, dt)

            ag["state"] = next_state

            # ── Update 3D lines ──
            lines_3d[model_name].set_data_3d(ag["ee_xs"], ag["ee_ys"], ag["ee_zs"])
            dots_3d[model_name].set_data_3d(
                [ag["ee_xs"][-1]], [ag["ee_ys"][-1]], [ag["ee_zs"][-1]]
            )

            # ── Update error lines ──
            lines_err[model_name].set_data(ag["times"], ag["ee_errors"])

        # ── Update error axis limits ──
        all_errs = [e for ag in agents.values() for e in ag["ee_errors"]]
        if all_errs:
            max_err = max(all_errs) * 1.15
            ax_err.set_ylim(0, max(0.05, max_err))

        # ── Update RMSE bars ──
        rmses = []
        for i, model_name in enumerate(models):
            ag = agents[model_name]
            errs = np.array(ag["ee_errors"])
            rmse = float(np.sqrt(np.mean(errs ** 2))) if len(errs) > 0 else 0.0
            rmses.append(rmse)
            bars[i].set_height(rmse)
            bar_texts[i].set_text(f"{rmse:.3f}")
            bar_texts[i].set_y(rmse)

        if rmses:
            ax_bar.set_ylim(0, max(0.05, max(rmses) * 1.2))

        # ── Update metrics text ──
        lines_txt = [
            f"t = {t:.1f}s / {duration:.0f}s",
            "",
            f"{'Model':<18} {'RMSE':>7} {'Curr':>7} {'Solve':>8}",
            "-" * 44,
        ]
        for model_name in models:
            ag = agents[model_name]
            errs = np.array(ag["ee_errors"])
            rmse = float(np.sqrt(np.mean(errs ** 2))) if len(errs) > 0 else 0.0
            curr = ag["ee_errors"][-1] if ag["ee_errors"] else 0.0
            solve = np.mean(ag["solve_times"]) if ag["solve_times"] else 0.0
            label = MODEL_LABELS.get(model_name, model_name)
            lines_txt.append(f"{label:<18} {rmse:>7.4f} {curr:>7.4f} {solve:>6.1f}ms")
        metrics_text_obj.set_text("\n".join(lines_txt))

        return []

    anim = FuncAnimation(
        fig, update, frames=n_steps, interval=1, blit=False, repeat=False,
    )
    plt.show()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LotF: Learning on the Fly Benchmark"
    )
    parser.add_argument("--scenario", type=str, default=None,
                        choices=SCENARIOS,
                        help="Single scenario (default: all)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model list (default: all)")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--model-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--live", action="store_true", help="Live animation mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        for m in models:
            if m not in ALL_MODELS:
                print(f"Unknown model: {m}. Available: {ALL_MODELS}")
                sys.exit(1)
    else:
        models = ALL_MODELS

    print("\n" + "=" * 60)
    print("LotF: Learning on the Fly Benchmark".center(60))
    print("=" * 60)
    print(f"  Models:    {[MODEL_LABELS.get(m, m) for m in models]}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Duration:  {args.duration}s")
    print(f"  K:         {args.K}")
    print(f"  Live Mode: {args.live}")
    print(f"  Model dir: {args.model_dir}")
    print("=" * 60)

    # ── Live mode ──
    if args.live:
        sc = scenarios[0]
        print(f"\n  Live scenario: {sc}")
        run_live(
            models, sc,
            K=args.K, duration=args.duration,
            model_dir=args.model_dir, seed=args.seed,
        )
        return

    # ── Batch mode ──
    t0 = time.time()
    all_results = run_benchmark(
        models, scenarios,
        K=args.K, duration=args.duration,
        model_dir=args.model_dir, seed=args.seed,
    )
    total_time = time.time() - t0

    print("\n" + "=" * 70)
    print("Results Summary".center(70))
    print("=" * 70)
    print(f"{'Model':<20} {'Scenario':<16} {'RMSE':>8} {'MaxErr':>8} {'Solve(ms)':>10}")
    print("-" * 70)

    for scenario in scenarios:
        for model_name in models:
            key = (model_name, scenario)
            if key not in all_results:
                continue
            m = all_results[key]["metrics"]
            label = MODEL_LABELS.get(model_name, model_name)
            print(f"{label:<20} {scenario:<16} "
                  f"{m['ee_rmse']:>8.4f} {m['ee_max']:>8.4f} "
                  f"{m['solve_ms_mean']:>8.1f}")
        print()

    for scenario in scenarios:
        ranking = []
        for model_name in models:
            key = (model_name, scenario)
            if key in all_results:
                ranking.append((model_name, all_results[key]["metrics"]["ee_rmse"]))
        ranking.sort(key=lambda x: x[1])
        labels_ranked = [f"{MODEL_LABELS.get(m, m)}({r:.4f})" for m, r in ranking]
        print(f"  Ranking ({scenario}): {' > '.join(labels_ranked)}")

    print(f"\n  Total benchmark time: {total_time:.1f}s")
    print("=" * 70 + "\n")

    if not args.no_plot:
        save_path = "plots/lotf_benchmark.png"
        os.makedirs("plots", exist_ok=True)
        plot_results(all_results, models, scenarios, save_path=save_path)


if __name__ == "__main__":
    main()
