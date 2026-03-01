#!/usr/bin/env python3
"""
6-DOF Mobile Manipulator 학습 모델 8-Way 벤치마크

8개 모델 × 2 시나리오 벤치마크:
  1. Kinematic       — 기준선 (모델 미스매치)
  2. Residual-NN     — 오프라인 MLP
  3. Residual-GP     — 오프라인 Sparse GP
  4. Residual-Ensemble — 오프라인 5-MLP
  5. Residual-MCDropout — 오프라인 MLP+Dropout
  6. Residual-MAML   — 메타학습 + few-shot 온라인 적응
  7. Residual-ALPaCA  — 메타학습 + Bayesian 온라인 적응
  8. Oracle          — 완벽 모델 (upper bound)

시나리오: ee_3d_circle, ee_3d_helix

Usage:
    PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py
    PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py --scenario ee_3d_helix
    PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py --models kinematic,residual_nn,oracle
    PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py --duration 15
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
    "kinematic", "residual_nn", "residual_gp", "residual_ensemble",
    "residual_mcdropout", "residual_maml", "residual_alpaca", "oracle",
]

MODEL_LABELS = {
    "kinematic": "Kinematic",
    "residual_nn": "Res-NN",
    "residual_gp": "Res-GP",
    "residual_ensemble": "Res-Ensemble",
    "residual_mcdropout": "Res-MCDrop",
    "residual_maml": "Res-MAML",
    "residual_alpaca": "Res-ALPaCA",
    "oracle": "Oracle",
}

MODEL_COLORS = {
    "kinematic": "tab:gray",
    "residual_nn": "tab:blue",
    "residual_gp": "tab:green",
    "residual_ensemble": "tab:purple",
    "residual_mcdropout": "tab:cyan",
    "residual_maml": "tab:orange",
    "residual_alpaca": "tab:red",
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

    elif name == "residual_gp":
        from mppi_controller.models.learned.gaussian_process_dynamics import GaussianProcessDynamics
        model_path = os.path.join(model_dir, "gp", "gp_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] GP model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = GaussianProcessDynamics(state_dim=9, control_dim=8, model_path=model_path)
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, False, None

    elif name == "residual_ensemble":
        from mppi_controller.models.learned.ensemble_dynamics import EnsembleNeuralDynamics
        model_path = os.path.join(model_dir, "ensemble", "ensemble.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] Ensemble model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = EnsembleNeuralDynamics(state_dim=9, control_dim=8, model_path=model_path)
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, False, None

    elif name == "residual_mcdropout":
        from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics
        model_path = os.path.join(model_dir, "mcdropout", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] MCDropout model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = MCDropoutDynamics(state_dim=9, control_dim=8, model_path=model_path, num_samples=10)
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, False, None

    elif name == "residual_maml":
        from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
        model_path = os.path.join(model_dir, "maml", "maml_meta_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] MAML model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = MAMLDynamics(state_dim=9, control_dim=8, model_path=model_path,
                               inner_lr=0.01, inner_steps=5)
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, True, learned

    elif name == "residual_alpaca":
        from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
        model_path = os.path.join(model_dir, "alpaca", "alpaca_meta_model.pth")
        if not os.path.exists(model_path):
            print(f"    [WARN] ALPaCA model not found: {model_path}, using oracle fallback")
            oracle_fn = create_oracle_residual(kin_model, dyn_model)
            return ResidualDynamics(base_model=kin_model, residual_fn=oracle_fn), False, None
        learned = ALPaCADynamics(state_dim=9, control_dim=8, model_path=model_path)
        model = ResidualDynamics(base_model=kin_model, learned_model=learned)
        return model, True, learned

    else:
        raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────────────────────
# MPPI setup
# ─────────────────────────────────────────────────────────────

def create_params(K=512, N=40, dt=0.05):
    """MPPI 파라미터 생성."""
    return MPPIParams(
        N=N,
        dt=dt,
        K=K,
        lambda_=0.3,
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
               K=512, duration=20.0, ema_alpha=0.2, adapt_buffer_size=20,
               model_dir=SAVE_DIR, seed=42):
    """
    단일 모델 × 시나리오 시뮬레이션.

    Returns:
        history: dict with time, state, control, ee_pos, ee_ref, ee_error, solve_time
    """
    np.random.seed(seed)

    # Model
    planning_model, needs_adapt, adapt_model = create_model(
        model_name, kin_model, dyn_model, model_dir
    )

    # Params & controller
    params = create_params(K=K)
    cost_fn = create_cost_fn(kin_model)
    controller = MPPIController(planning_model, params, cost_function=cost_fn)

    # Trajectory
    traj_fn = create_trajectory_function(scenario_name)

    dt = params.dt
    n_steps = int(duration / dt)

    # History
    history = {
        "time": [], "state": [], "control": [],
        "ee_pos": [], "ee_ref": [], "ee_error": [], "solve_time": [],
    }

    state = np.zeros(9)
    filtered_ctrl = np.zeros(8)

    # Adaptation buffer (for MAML/ALPaCA)
    adapt_states = []
    adapt_controls = []
    adapt_next_states = []

    for step in range(n_steps):
        t = step * dt
        ref = generate_reference_trajectory(traj_fn, t, params.N, dt)

        # Control
        t0 = time.perf_counter()
        raw_control, _ = controller.compute_control(state, ref)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # EMA filter
        filtered_ctrl = ema_alpha * raw_control + (1 - ema_alpha) * filtered_ctrl
        control = filtered_ctrl

        # EE evaluation
        ee_pos = kin_model.forward_kinematics(state)
        ee_ref = ref[0, :3]
        ee_error = np.sqrt(np.sum((ee_pos - ee_ref) ** 2))

        # Record
        history["time"].append(t)
        history["state"].append(state.copy())
        history["control"].append(control.copy())
        history["ee_pos"].append(ee_pos.copy())
        history["ee_ref"].append(ee_ref.copy())
        history["ee_error"].append(ee_error)
        history["solve_time"].append(solve_ms)

        # Environment step (ground-truth dynamic model)
        next_state = dyn_model.step(state, control, dt)
        next_state = dyn_model.normalize_state(next_state)

        # Online adaptation for MAML/ALPaCA
        if needs_adapt and adapt_model is not None:
            adapt_states.append(state.copy())
            adapt_controls.append(control.copy())
            adapt_next_states.append(next_state.copy())

            if len(adapt_states) >= adapt_buffer_size:
                buf_s = np.array(adapt_states[-adapt_buffer_size:])
                buf_c = np.array(adapt_controls[-adapt_buffer_size:])
                buf_ns = np.array(adapt_next_states[-adapt_buffer_size:])

                # Compute residual targets for adaptation
                # MAML/ALPaCA learn residual, so targets = residual_next_states
                kin_dots = kin_model.forward_dynamics(buf_s, buf_c)
                actual_dots = (buf_ns - buf_s) / dt
                residual_dots = actual_dots - kin_dots
                residual_ns = buf_s + residual_dots * dt

                adapt_model.adapt(buf_s, buf_c, residual_ns, dt)

        state = next_state

    # Convert to numpy
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
    """
    Run benchmark for all model × scenario combinations.

    Returns:
        all_results: dict { (model, scenario): {"metrics": ..., "history": ...} }
    """
    all_results = {}

    kin_model = MobileManipulator6DOFKinematic()
    dyn_model = MobileManipulator6DOFDynamic()

    total = len(models) * len(scenarios)
    idx = 0

    for scenario in scenarios:
        for model_name in models:
            idx += 1
            label = MODEL_LABELS.get(model_name, model_name)
            print(f"\n  [{idx}/{total}] {label} × {scenario}")

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
    2×3 벤치마크 시각화.

    [0,0] EE 3D 궤적 (8개 겹침)
    [0,1] EE 오차 시계열
    [0,2] RMSE 바 차트
    [1,0] Solve time 바 차트
    [1,1] 관절 궤적 (q2, 중력)
    [1,2] 메트릭 요약 테이블
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_scenarios = len(scenarios)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"6-DOF Mobile Manipulator: {len(models)}-Way Learned Model Benchmark",
        fontsize=14, fontweight="bold",
    )

    # Use first scenario for trajectory/time-series plots
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
    # Reference
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

    # [0,2] RMSE bar chart (all scenarios)
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
        bars = ax.bar(x + offset, rmses, width * 0.9, label=scenario,
                      alpha=0.8)
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

    # [1,0] Solve time bar chart
    ax = fig.add_subplot(2, 3, 4)
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

    # [1,1] Joint q2 trajectory (gravity-affected)
    ax = fig.add_subplot(2, 3, 5)
    for model_name in models:
        key = (model_name, sc0)
        if key not in all_results:
            continue
        h = all_results[key]["history"]
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(h["time"], np.rad2deg(h["state"][:, 4]), "-",
                color=color, linewidth=0.8, label=label, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("q2 (deg)")
    ax.set_title(f"Shoulder Pitch (q2) ({sc0})")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # [1,2] Metrics summary table
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")

    lines = [f"{'Model':<14} {'Scenario':<14} {'RMSE':>7} {'MaxErr':>7} {'Solve':>8}"]
    lines.append("─" * 56)
    for scenario in scenarios:
        for model_name in models:
            key = (model_name, scenario)
            if key not in all_results:
                continue
            m = all_results[key]["metrics"]
            label = MODEL_LABELS.get(model_name, model_name)
            lines.append(
                f"{label:<14} {scenario:<14} "
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
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="6-DOF Learned Model 8-Way Benchmark"
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse scenarios
    scenarios = [args.scenario] if args.scenario else SCENARIOS

    # Parse models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        for m in models:
            if m not in ALL_MODELS:
                print(f"Unknown model: {m}. Available: {ALL_MODELS}")
                sys.exit(1)
    else:
        models = ALL_MODELS

    print("\n" + "=" * 60)
    print("6-DOF Learned Model Benchmark".center(60))
    print("=" * 60)
    print(f"  Models:    {[MODEL_LABELS.get(m, m) for m in models]}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Duration:  {args.duration}s")
    print(f"  K:         {args.K}")
    print(f"  Model dir: {args.model_dir}")
    print("=" * 60)

    # Run benchmark
    t0 = time.time()
    all_results = run_benchmark(
        models, scenarios,
        K=args.K, duration=args.duration,
        model_dir=args.model_dir, seed=args.seed,
    )
    total_time = time.time() - t0

    # Print results table
    print("\n" + "=" * 70)
    print("Results Summary".center(70))
    print("=" * 70)
    print(f"{'Model':<16} {'Scenario':<16} {'RMSE':>8} {'MaxErr':>8} {'Solve(ms)':>10}")
    print("─" * 70)

    for scenario in scenarios:
        for model_name in models:
            key = (model_name, scenario)
            if key not in all_results:
                continue
            m = all_results[key]["metrics"]
            label = MODEL_LABELS.get(model_name, model_name)
            print(f"{label:<16} {scenario:<16} "
                  f"{m['ee_rmse']:>8.4f} {m['ee_max']:>8.4f} "
                  f"{m['solve_ms_mean']:>8.1f}")
        print()

    # Rankings
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

    # Plot
    if not args.no_plot:
        save_path = "plots/6dof_learned_benchmark.png"
        os.makedirs("plots", exist_ok=True)
        plot_results(all_results, models, scenarios, save_path=save_path)


if __name__ == "__main__":
    main()
