#!/usr/bin/env python3
"""
MPPI Autotune 벤치마크: Default vs Autotuned vs Online 비교

비교 3종:
  1. Default   — 기본 파라미터
  2. Autotuned — 오프라인 최적화 (differential_evolution)
  3. Online    — AutotunedMPPIController (sigma + lambda 적응)

대상: Vanilla MPPI (기본), 선택적으로 Kernel MPPI 추가
시나리오: circle 궤적
출력: plots/autotune_benchmark.png + 콘솔 메트릭 테이블

Usage:
    PYTHONPATH=. python examples/comparison/autotune_benchmark.py
    PYTHONPATH=. python examples/comparison/autotune_benchmark.py --variants vanilla kernel
    PYTHONPATH=. python examples/comparison/autotune_benchmark.py --skip-offline
"""

import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, KernelMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.kernel_mppi import KernelMPPIController
from mppi_controller.controllers.mppi.adaptive_temperature import AdaptiveTemperature
from mppi_controller.controllers.mppi.autotune import (
    AutotuneObjective,
    AutotuneConfig,
    MPPIAutotuner,
    OnlineSigmaAdapter,
    AutotunedMPPIController,
)
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics, compare_metrics
from mppi_controller.utils.trajectory import (
    circle_trajectory,
    generate_reference_trajectory,
)


# ── 설정 ──────────────────────────────────────────────────────────

RADIUS = 3.0
ANGULAR_VEL = 0.3
SIM_DURATION = 15.0
DT = 0.05
K = 256
N = 20

INITIAL_STATE = np.array([RADIUS, 0.0, np.pi / 2])


def make_model():
    return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)


def make_reference_fn(params):
    def ref_fn(t):
        return generate_reference_trajectory(
            lambda t_: circle_trajectory(t_, radius=RADIUS, angular_velocity=ANGULAR_VEL),
            t, params.N, params.dt,
        )
    return ref_fn


# ── 변형 설정 ─────────────────────────────────────────────────────

VARIANT_CONFIGS = {
    "vanilla": {
        "name": "Vanilla MPPI",
        "params_cls": MPPIParams,
        "controller_cls": MPPIController,
        "extra_params": {},
    },
    "kernel": {
        "name": "Kernel MPPI",
        "params_cls": KernelMPPIParams,
        "controller_cls": KernelMPPIController,
        "extra_params": {"num_support_pts": 6, "kernel_bandwidth": 1.0},
    },
}


def run_simulation(controller, model, params, label=""):
    """단일 시뮬레이션 실행 → (history, metrics)"""
    ref_fn = make_reference_fn(params)
    sim = Simulator(model, controller, params.dt, store_info=False)
    sim.reset(INITIAL_STATE.copy())

    t_start = time.time()
    history = sim.run(ref_fn, SIM_DURATION)
    wall_time = time.time() - t_start

    metrics = compute_metrics(history)
    if label:
        print(f"  {label}: RMSE={metrics['position_rmse']:.4f}m, "
              f"solve={metrics['mean_solve_time']:.1f}ms, "
              f"wall={wall_time:.1f}s")
    return history, metrics


def run_variant_benchmark(variant_key, args):
    """단일 변형에 대해 3종 비교"""
    cfg = VARIANT_CONFIGS[variant_key]
    print(f"\n{'='*60}")
    print(f"  {cfg['name']} Autotune Benchmark")
    print(f"{'='*60}")

    base_params_kwargs = dict(
        K=K, N=N, dt=DT,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    base_params_kwargs.update(cfg["extra_params"])
    base_params = cfg["params_cls"](**base_params_kwargs)

    results = {}
    histories = {}

    # ── 1. Default ────────────────────────────────────────────────
    print("\n[1/3] Default parameters...")
    model = make_model()
    controller = cfg["controller_cls"](model, base_params)
    hist, met = run_simulation(controller, model, base_params, "Default")
    results["Default"] = met
    histories["Default"] = hist

    # ── 2. Autotuned (오프라인) ───────────────────────────────────
    if not args.skip_offline:
        print("\n[2/3] Offline autotuning...")

        # 의도적으로 나쁜 초기 파라미터로 시작
        bad_params_kwargs = dict(base_params_kwargs)
        bad_params_kwargs["lambda_"] = 20.0
        bad_params_kwargs["Q"] = np.array([1.0, 1.0, 0.1])
        bad_params = cfg["params_cls"](**bad_params_kwargs)

        tune_config = AutotuneConfig(
            tunable_params=["lambda_", "Q_scale", "sigma_scale"],
            param_bounds={
                "lambda_": (0.1, 50.0),
                "Q_scale": (0.1, 50.0),
                "sigma_scale": (0.1, 5.0),
            },
            optimizer="differential_evolution",
            max_iterations=args.tune_iters,
            sim_duration=min(SIM_DURATION, 8.0),
            seed=42,
            verbose=True,
        )

        tuner = MPPIAutotuner(
            model_fn=make_model,
            controller_cls=cfg["controller_cls"],
            base_params=bad_params,
            reference_fn=make_reference_fn(bad_params),
            initial_state=INITIAL_STATE.copy(),
            objective=AutotuneObjective.balanced(),
            config=tune_config,
        )

        t_tune_start = time.time()
        tuned_params, tune_info = tuner.tune()
        tune_time = time.time() - t_tune_start

        print(f"  Tuning completed: {tune_info['n_evaluations']} evals in {tune_time:.1f}s")
        print(f"  Best score: {tune_info['best_score']:.4f}")
        print(f"  Tuned lambda={tuned_params.lambda_:.2f}, "
              f"sigma={tuned_params.sigma}, Q={tuned_params.Q}")

        model = make_model()
        controller = cfg["controller_cls"](model, tuned_params)
        hist, met = run_simulation(controller, model, tuned_params, "Autotuned")
        results["Autotuned"] = met
        histories["Autotuned"] = hist
    else:
        print("\n[2/3] Skipping offline autotuning (--skip-offline)")

    # ── 3. Online adaptation ──────────────────────────────────────
    print("\n[3/3] Online adaptation...")
    model = make_model()
    controller = cfg["controller_cls"](model, cfg["params_cls"](**base_params_kwargs))
    sigma_adapter = OnlineSigmaAdapter(
        base_sigma=np.array([0.5, 0.5]),
        adaptation_rate=0.05,
        min_sigma_ratio=0.5,
        max_sigma_ratio=2.0,
        seed=42,
    )
    temp_adapter = AdaptiveTemperature(
        initial_lambda=1.0,
        adaptation_rate=0.05,
        lambda_min=0.1,
        lambda_max=20.0,
    )
    wrapped = AutotunedMPPIController(
        controller,
        sigma_adapter=sigma_adapter,
        temperature_adapter=temp_adapter,
    )

    hist, met = run_simulation(wrapped, model, wrapped.params, "Online")
    results["Online"] = met
    histories["Online"] = hist

    # σ 적응 통계
    sigma_stats = sigma_adapter.get_statistics()
    if sigma_stats["has_adapted"]:
        print(f"  Final sigma ratio: mean={sigma_stats['mean_ratio']:.3f}, "
              f"range=[{sigma_stats['min_ratio']:.3f}, {sigma_stats['max_ratio']:.3f}]")
    print(f"  Final lambda: {controller.params.lambda_:.3f}")

    return results, histories


def plot_results(all_results, all_histories, save_path):
    """결과 시각화"""
    n_variants = len(all_results)
    fig, axes = plt.subplots(n_variants, 3, figsize=(15, 5 * n_variants))
    if n_variants == 1:
        axes = axes[np.newaxis, :]

    colors = {"Default": "#1f77b4", "Autotuned": "#2ca02c", "Online": "#d62728"}

    for row, (variant_key, results) in enumerate(all_results.items()):
        histories = all_histories[variant_key]
        cfg = VARIANT_CONFIGS[variant_key]

        # 1. XY 궤적
        ax = axes[row, 0]
        # 레퍼런스 궤적
        t_ref = np.linspace(0, SIM_DURATION, 200)
        ref_xy = np.array([
            circle_trajectory(t, radius=RADIUS, angular_velocity=ANGULAR_VEL)[:2]
            for t in t_ref
        ])
        ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.3, label="Reference")

        for label, hist in histories.items():
            states = hist["state"]
            ax.plot(states[:, 0], states[:, 1],
                    color=colors.get(label, "gray"), label=label, linewidth=1.5)

        ax.set_title(f"{cfg['name']} — XY Trajectory")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. 위치 오차
        ax = axes[row, 1]
        for label, hist in histories.items():
            states = hist["state"]
            refs = hist["reference"]
            errors = np.linalg.norm(states[:, :2] - refs[:, :2], axis=1)
            times = hist["time"]
            ax.plot(times, errors, color=colors.get(label, "gray"),
                    label=label, linewidth=1.0)

        ax.set_title(f"{cfg['name']} — Position Error")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. 메트릭 바 차트
        ax = axes[row, 2]
        metric_keys = ["position_rmse", "control_rate", "mean_solve_time"]
        metric_labels = ["RMSE (m)", "Ctrl Rate", "Solve (ms)"]
        x = np.arange(len(metric_keys))
        width = 0.25

        for i, (label, met) in enumerate(results.items()):
            values = [met[k] for k in metric_keys]
            ax.bar(x + i * width, values, width, label=label,
                   color=colors.get(label, "gray"), alpha=0.8)

        ax.set_title(f"{cfg['name']} — Metrics")
        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MPPI Autotune Benchmark")
    parser.add_argument(
        "--variants", nargs="+", default=["vanilla"],
        choices=list(VARIANT_CONFIGS.keys()),
        help="MPPI variants to benchmark",
    )
    parser.add_argument(
        "--skip-offline", action="store_true",
        help="Skip offline autotuning (faster)",
    )
    parser.add_argument(
        "--tune-iters", type=int, default=15,
        help="Max iterations for offline tuning (default: 15)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    print("MPPI Autotune Benchmark")
    print(f"Variants: {args.variants}")
    print(f"Duration: {SIM_DURATION}s, K={K}, N={N}")

    all_results = {}
    all_histories = {}

    for variant_key in args.variants:
        results, histories = run_variant_benchmark(variant_key, args)
        all_results[variant_key] = results
        all_histories[variant_key] = histories

        # 메트릭 테이블
        labels = list(results.keys())
        metrics_list = [results[l] for l in labels]
        compare_metrics(metrics_list, labels, f"{VARIANT_CONFIGS[variant_key]['name']} Comparison")

    # 플롯
    if not args.no_plot:
        plot_results(all_results, all_histories, "plots/autotune_benchmark.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
