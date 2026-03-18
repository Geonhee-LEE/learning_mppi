#!/usr/bin/env python3
"""
Latent-Space MPPI 벤치마크: 3-Way 비교

방법:
  1. Vanilla MPPI   — 물리 모델 직접 rollout
  2. Flow-MPPI      — CFM 기반 다중 모달 샘플링
  3. Latent-MPPI    — VAE 잠재 공간 rollout + 하이브리드 비용

시나리오:
  A. simple    — 장애물 없음, 순수 추적 성능
  B. obstacles — 장애물 회피, 잠재 공간 안전성

흐름:
  1. DifferentialDriveKinematic에서 학습 데이터 생성
  2. WorldModelTrainer로 VAE 학습
  3. WorldModelDynamics + LatentMPPIController 생성
  4. SimulationHarness 기반 3-way 비교
  5. 6-panel 플롯

Usage:
    PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py
    PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py --scenario obstacles
    PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py --trajectory figure8
    PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py --no-plot
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
from matplotlib.patches import Circle

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    FlowMPPIParams,
    LatentMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.flow_mppi import FlowMPPIController
from mppi_controller.controllers.mppi.latent_mppi import LatentMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.learning.world_model_trainer import WorldModelTrainer
from mppi_controller.models.learned.world_model_dynamics import WorldModelDynamics
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
    figure_eight_trajectory,
)
from mppi_controller.simulation.harness import SimulationHarness


# ── 시나리오 설정 ────────────────────────────────────────────

OBSTACLES = [
    (2.5, 2.0, 0.4),
    (-1.5, 3.0, 0.5),
    (1.0, -3.0, 0.3),
]


def get_scenarios():
    return {
        "simple": {
            "name": "Simple (no obstacles)",
            "obstacles": None,
        },
        "obstacles": {
            "name": "Obstacle avoidance",
            "obstacles": OBSTACLES,
        },
    }


def create_trajectory_fn(name):
    if name == "circle":
        return lambda t: circle_trajectory(t, radius=3.0)
    elif name == "figure8":
        return figure_eight_trajectory
    return lambda t: circle_trajectory(t, radius=3.0)


# ── 학습 데이터 생성 ─────────────────────────────────────────

def generate_training_data(model, dt=0.05, n_samples=2000, seed=42):
    """
    물리 모델에서 학습 데이터 생성

    3가지 소스로 넓은 상태 공간 커버:
    1. 랜덤 롤아웃 (다양한 시작점)
    2. 원형 궤적 추종 데이터 (실제 운용 분포)
    3. 그리드 샘플링 (OOD 영역 커버)
    """
    np.random.seed(seed)

    states = []
    controls = []
    next_states = []

    # 1. 랜덤 롤아웃 — 다양한 시작점에서 (60%)
    n_random = int(n_samples * 0.6)
    state = np.array([0.0, 0.0, 0.0])
    for _ in range(n_random):
        control = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
        ])
        next_state = model.step(state, control, dt)
        states.append(state.copy())
        controls.append(control.copy())
        next_states.append(next_state.copy())
        state = next_state

        # 넓은 범위의 시작점 리셋
        if np.random.random() < 0.05:
            state = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(-np.pi, np.pi),
            ])

    # 2. 원형 궤적 추종 데이터 (25%)
    n_circle = int(n_samples * 0.25)
    for radius in [2.0, 3.0, 4.0]:
        state = np.array([radius, 0.0, np.pi / 2])
        for i in range(n_circle // 3):
            t = i * dt
            ref = circle_trajectory(t, radius=radius)
            # P 제어로 궤적 추종
            dx = ref[0] - state[0]
            dy = ref[1] - state[1]
            target_theta = np.arctan2(dy, dx)
            heading_err = target_theta - state[2]
            heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
            control = np.array([
                np.clip(0.5 + np.random.randn() * 0.2, -1, 1),
                np.clip(heading_err * 2.0 + np.random.randn() * 0.3, -1, 1),
            ])
            next_state = model.step(state, control, dt)
            states.append(state.copy())
            controls.append(control.copy())
            next_states.append(next_state.copy())
            state = next_state

    # 3. 그리드 샘플링 — OOD 영역 커버 (15%)
    n_grid = n_samples - len(states)
    for _ in range(n_grid):
        state = np.array([
            np.random.uniform(-6, 6),
            np.random.uniform(-6, 6),
            np.random.uniform(-np.pi, np.pi),
        ])
        control = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
        ])
        next_state = model.step(state, control, dt)
        states.append(state.copy())
        controls.append(control.copy())
        next_states.append(next_state.copy())

    return np.array(states), np.array(controls), np.array(next_states)


# ── 시뮬레이션 실행 ──────────────────────────────────────────

def run_simulation(model, controller, reference_fn, initial_state, dt, duration):
    """SimulationHarness 기반 시뮬레이션"""
    harness = SimulationHarness(dt=dt, headless=True, seed=42)
    harness.add_controller("ctrl", controller, model)
    results = harness.run(reference_fn, initial_state, duration)
    r = results["ctrl"]
    h = r["history"]

    return {
        "states": np.vstack([initial_state[None, :], h["state"]]),
        "controls": h["control"],
        "solve_times": h["solve_time"],
        "infos": h.get("info", []),
    }


def compute_tracking_rmse(states, trajectory_fn, dt):
    """궤적 추적 RMSE"""
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    return np.sqrt(np.mean(np.array(errors) ** 2))


def compute_obstacle_metrics(states, obstacles):
    """장애물 관련 메트릭 계산"""
    if not obstacles:
        return {"n_collisions": 0, "min_clearance": float("inf"), "mean_min_clearance": 0}
    n_collisions = 0
    min_dist = float("inf")
    min_distances = []

    for st in states:
        x, y = st[0], st[1]
        for ox, oy, r in obstacles:
            dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            clearance = dist - r
            min_dist = min(min_dist, clearance)
            if clearance < 0:
                n_collisions += 1
        closest = min(np.sqrt((x - ox) ** 2 + (y - oy) ** 2) - r for ox, oy, r in obstacles)
        min_distances.append(closest)

    return {
        "n_collisions": n_collisions,
        "min_clearance": min_dist,
        "mean_min_clearance": float(np.mean(min_distances)),
    }


# ── 벤치마크 메인 ─────────────────────────────────────────────

def run_benchmark(args):
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    has_obstacles = scenario["obstacles"] is not None

    trajectory_fn = create_trajectory_fn(args.trajectory)

    print(f"\n{'=' * 72}")
    print(f"  Latent-MPPI Benchmark: 3-Way Comparison")
    print(f"  Scenario: {scenario['name']}")
    print(f"  Trajectory: {args.trajectory} | Duration: {args.duration}s | Seed: {args.seed}")
    print(f"{'=' * 72}")

    model = DifferentialDriveKinematic(wheelbase=0.5)
    initial_state = np.array([0.0, 0.0, 0.0])

    common = dict(
        K=512, N=30, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )

    # ── 1. VAE 학습 ──
    print("\n  [0/3] VAE 학습 중...", end=" ", flush=True)
    t_train_start = time.time()

    # latent_dim=4: state_dim=3이므로 약간만 확장 (과파라미터화 방지)
    latent_dim = 4

    train_states, train_controls, train_next = generate_training_data(
        model, dt=common["dt"], n_samples=5000, seed=args.seed,
    )

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=latent_dim, hidden_dims=[128, 128],
        beta=0.001, alpha_dyn=2.0,
        multistep_horizon=10, alpha_multistep=1.0,
        save_dir="models/learned_models",
    )
    trainer.train(
        train_states, train_controls, train_next,
        epochs=300, batch_size=128, verbose=False,
    )

    # WorldModelDynamics 생성
    wm_dyn = WorldModelDynamics(state_dim=3, control_dim=2, latent_dim=latent_dim)
    wm_dyn.set_vae(trainer.model)

    train_time = time.time() - t_train_start
    print(f"done ({train_time:.1f}s)")

    # ── 2. 컨트롤러 생성 함수 ──

    def make_cost(params):
        costs = [
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
            ControlEffortCost(params.R),
        ]
        if has_obstacles:
            costs.append(ObstacleCost(scenario["obstacles"],
                                      safety_margin=0.3, cost_weight=2000.0))
        return CompositeMPPICost(cost_functions=costs)

    def make_vanilla():
        params = MPPIParams(**common)
        cost = make_cost(params)
        return MPPIController(model, params, cost_function=cost)

    def make_flow():
        params = FlowMPPIParams(
            **common,
            flow_mode="replace_mean",
            flow_exploration_sigma=0.5,
        )
        cost = make_cost(params)
        return FlowMPPIController(model, params, cost_function=cost)

    def make_latent():
        params = LatentMPPIParams(
            **common,
            latent_dim=latent_dim,
            vae_hidden_dims=[128, 128],
            decode_interval=5,  # 5스텝마다 re-encode로 drift 보정
        )
        cost = make_cost(params)
        return LatentMPPIController(
            model, params, cost_function=cost, world_model=wm_dyn,
        )

    variants = [
        {"name": "Vanilla MPPI",   "short": "Vanilla",  "make": make_vanilla,  "color": "#2196F3"},
        {"name": "Flow-MPPI",      "short": "Flow",     "make": make_flow,     "color": "#FF9800"},
        {"name": "Latent-MPPI",    "short": "Latent",   "make": make_latent,   "color": "#9C27B0"},
    ]

    ref_fn = lambda t, _fn=trajectory_fn, _N=common["N"], _dt=common["dt"]: \
        generate_reference_trajectory(_fn, t, _N, _dt)

    # ── 3. 실행 + 수집 ──
    all_results = []
    for i, var in enumerate(variants):
        np.random.seed(args.seed)

        print(f"\n  [{i+1}/{len(variants)}] {var['name']:<22}", end=" ", flush=True)
        t_start = time.time()

        controller = var["make"]()
        history = run_simulation(
            model, controller, ref_fn, initial_state, common["dt"], args.duration,
        )
        elapsed = time.time() - t_start

        rmse = compute_tracking_rmse(history["states"], trajectory_fn, common["dt"])
        solve_times = history["solve_times"]
        mean_solve = np.mean(solve_times) * 1000
        max_solve = np.max(solve_times) * 1000
        min_solve = np.min(solve_times) * 1000

        obs_metrics = compute_obstacle_metrics(
            history["states"], scenario["obstacles"] if has_obstacles else None
        )

        ess_list = []
        for info in history["infos"]:
            if isinstance(info, dict) and "ess" in info:
                ess_list.append(info["ess"])

        # Latent 고유 메트릭
        latent_stats = {}
        if var["short"] == "Latent" and history["infos"]:
            norms = [
                info.get("latent_stats", {}).get("mean_latent_norm", 0)
                for info in history["infos"]
                if isinstance(info, dict) and "latent_stats" in info
            ]
            if norms:
                latent_stats["mean_latent_norm"] = float(np.mean(norms))
                latent_stats["max_latent_norm"] = float(np.max([
                    info.get("latent_stats", {}).get("max_latent_norm", 0)
                    for info in history["infos"]
                    if isinstance(info, dict) and "latent_stats" in info
                ]))

        all_results.append({
            "name": var["name"],
            "short": var["short"],
            "color": var["color"],
            "rmse": rmse,
            "mean_solve_ms": mean_solve,
            "max_solve_ms": max_solve,
            "min_solve_ms": min_solve,
            "elapsed": elapsed,
            "states": history["states"],
            "infos": history["infos"],
            "latent_stats": latent_stats,
            "obs_metrics": obs_metrics,
            "ess_list": ess_list,
        })

        print(f"done ({elapsed:.1f}s)")

    # ── 4. 결과 출력 ──
    print(f"\n{'─' * 72}")
    if has_obstacles:
        print(f"{'Method':<22} {'RMSE':>8} {'Collisions':>10} {'MinClear':>10} {'MeanTime':>10} {'MaxTime':>10}")
    else:
        print(f"{'Method':<22} {'RMSE':>8} {'MeanTime(ms)':>14} {'MinTime(ms)':>14} {'MaxTime(ms)':>14}")
    print(f"{'─' * 72}")

    for r in all_results:
        if has_obstacles:
            print(
                f"{r['name']:<22} "
                f"{r['rmse']:>8.4f} "
                f"{r['obs_metrics']['n_collisions']:>10d} "
                f"{r['obs_metrics']['min_clearance']:>10.3f} "
                f"{r['mean_solve_ms']:>10.1f} "
                f"{r['max_solve_ms']:>10.1f}"
            )
        else:
            print(
                f"{r['name']:<22} "
                f"{r['rmse']:>8.4f} "
                f"{r['mean_solve_ms']:>14.1f} "
                f"{r['min_solve_ms']:>14.1f} "
                f"{r['max_solve_ms']:>14.1f}"
            )

    print(f"{'─' * 72}")

    # Latent 고유 메트릭
    for r in all_results:
        if r["latent_stats"]:
            stats = r["latent_stats"]
            print(f"\n  Latent-MPPI 고유 메트릭:")
            print(f"    평균 latent norm: {stats.get('mean_latent_norm', 0):.4f}")
            print(f"    최대 latent norm: {stats.get('max_latent_norm', 0):.4f}")

    # ESS 출력
    for r in all_results:
        if r["ess_list"]:
            print(f"  {r['short']} ESS: mean={np.mean(r['ess_list']):.1f}, "
                  f"min={np.min(r['ess_list']):.1f}")

    # ── 5. 플롯 ──
    if not args.no_plot:
        _plot_results(all_results, common["dt"], args.duration, trajectory_fn,
                      args.scenario, scenario, wm_dyn)

    print()
    return all_results


def _plot_results(results, dt, duration, trajectory_fn, scenario_name, scenario, wm_dyn):
    """6-panel 결과 플롯"""
    has_obstacles = scenario["obstacles"] is not None
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. XY 궤적
    ax = axes[0, 0]
    t_arr = np.arange(0, duration + dt, dt)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, label="Reference", linewidth=1)

    for r in results:
        states = r["states"]
        ax.plot(states[:, 0], states[:, 1], color=r["color"],
                label=r["short"], linewidth=1.5)

    if has_obstacles:
        for ox, oy, radius in scenario["obstacles"]:
            circle = Circle((ox, oy), radius, fill=True, facecolor="#FF5252",
                            edgecolor="red", alpha=0.3, linewidth=1.5)
            ax.add_patch(circle)
            margin_circle = Circle((ox, oy), radius + 0.2, fill=False,
                                   edgecolor="red", alpha=0.3, linestyle="--")
            ax.add_patch(margin_circle)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. 위치 오차
    ax = axes[0, 1]
    for r in results:
        states = r["states"]
        errors = []
        for i, st in enumerate(states):
            ref_pt = trajectory_fn(i * dt)
            err = np.sqrt((st[0] - ref_pt[0]) ** 2 + (st[1] - ref_pt[1]) ** 2)
            errors.append(err)
        t_plot = np.arange(len(errors)) * dt
        ax.plot(t_plot, errors, color=r["color"], label=r["short"], linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. ESS
    ax = axes[0, 2]
    for r in results:
        if r["ess_list"]:
            t_ess = np.arange(len(r["ess_list"])) * dt
            ax.plot(t_ess, r["ess_list"], color=r["color"],
                    label=r["short"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 계산 시간 비교 (bar chart)
    ax = axes[1, 0]
    names = [r["short"] for r in results]
    solve_means = [r["mean_solve_ms"] for r in results]
    colors = [r["color"] for r in results]
    bars = ax.bar(names, solve_means, color=colors, alpha=0.8)
    for bar, val in zip(bars, solve_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Solve Time (ms)")
    ax.set_title("Computation Time")
    ax.grid(True, alpha=0.3, axis="y")

    # 5. 잠재 공간 시각화 (PCA 2D 프로젝션)
    ax = axes[1, 1]
    latent_result = next((r for r in results if r["short"] == "Latent"), None)
    if latent_result and latent_result["infos"]:
        # 전체 궤적의 잠재 벡터 수집
        try:
            latent_states = wm_dyn.encode(latent_result["states"])

            if latent_states.shape[1] >= 2:
                # 간단한 PCA: 상위 2 주성분
                centered = latent_states - latent_states.mean(axis=0)
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1][:2]
                pc = centered @ eigenvectors[:, idx]

                scatter = ax.scatter(
                    pc[:, 0], pc[:, 1],
                    c=np.arange(len(pc)), cmap="viridis", s=5, alpha=0.6,
                )
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.colorbar(scatter, ax=ax, label="Time step")
        except Exception:
            pass

    ax.set_title("Latent Space (PCA)")
    ax.grid(True, alpha=0.3)

    # 6. RMSE 비교 (bar chart)
    ax = axes[1, 2]
    rmses = [r["rmse"] for r in results]
    bars = ax.bar(names, rmses, color=colors, alpha=0.8)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Tracking RMSE Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Latent-MPPI Benchmark [{scenario_name}]: Vanilla vs Flow vs Latent-MPPI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/latent_mppi_benchmark_{scenario_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Latent-MPPI Benchmark")
    parser.add_argument("--scenario", default="simple",
                        choices=["simple", "obstacles"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--trajectory", default="circle",
                        choices=["circle", "figure8"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.all_scenarios:
        for scenario_name in get_scenarios():
            args.scenario = scenario_name
            run_benchmark(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
