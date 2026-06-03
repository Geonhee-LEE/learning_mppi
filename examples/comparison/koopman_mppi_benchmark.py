"""
Koopman MPPI vs Vanilla MPPI 벤치마크

Koopman 연산자 기반 선형 동역학 예측이 MPPI rollout에 미치는 효과 비교.

실험 설정:
    - 로봇: DifferentialDrive (기구학)
    - 환경: Clean / Noisy (센서 잡음) / Model Mismatch
    - 비교: Vanilla MPPI (fallback) vs Koopman MPPI (EDMD 학습 후)
    - 메트릭: RMSE, 계산 시간, rollout 모드

사용법:
    PYTHONPATH=. python examples/comparison/koopman_mppi_benchmark.py
    PYTHONPATH=. python examples/comparison/koopman_mppi_benchmark.py --no-plot
    PYTHONPATH=. python examples/comparison/koopman_mppi_benchmark.py --scenario mismatch
"""

import sys
import time
import argparse
import numpy as np

sys.path.insert(0, ".")

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.koopman_mppi import KoopmanMPPIController
from mppi_controller.controllers.mppi.mppi_params import KoopmanMPPIParams


# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

N_STEPS = 100     # 시뮬레이션 스텝 수
DT = 0.05         # 타임스텝 (s)
N_HORIZON = 20    # MPPI 호라이즌
K_SAMPLES = 256   # 샘플 수
N_TRAIN = 500     # EDMD 학습 데이터 수


def make_reference(n_steps: int, dt: float) -> np.ndarray:
    """원형 참조 궤적 생성."""
    t = np.arange(n_steps + 1) * dt
    ref = np.zeros((n_steps + 1, 3))
    ref[:, 0] = 2.0 * np.cos(0.3 * t) - 2.0
    ref[:, 1] = 2.0 * np.sin(0.3 * t)
    return ref


def make_params(controller_type: str, **kwargs) -> MPPIParams:
    base = dict(
        N=N_HORIZON,
        K=K_SAMPLES,
        dt=DT,
        lambda_=1.0,
        sigma=np.array([0.5, 0.3]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    if controller_type == "vanilla":
        return MPPIParams(**base)
    else:
        return KoopmanMPPIParams(
            **base,
            koopman_lift_fn=kwargs.get("lift_fn", "rbf"),
            koopman_lift_dim=kwargs.get("lift_dim", 32),
            koopman_rbf_gamma=kwargs.get("rbf_gamma", 1.0),
        )


def collect_training_data(model, n: int, rng) -> tuple:
    """합성 전이 데이터 수집."""
    nx, nu = model.state_dim, model.control_dim
    X = rng.uniform([-3, -3, -np.pi], [3, 3, np.pi], (n, nx))
    U = rng.uniform([-0.5, -1.0], [0.5, 1.0], (n, nu))
    X_next = np.array([
        X[i] + model.forward_dynamics(X[i], U[i]) * DT
        for i in range(n)
    ])
    return X, U, X_next


def run_simulation(
    model,
    ctrl,
    reference: np.ndarray,
    n_steps: int,
    noise_std: float = 0.0,
    rng=None,
) -> dict:
    """단일 시뮬레이션 실행."""
    if rng is None:
        rng = np.random.default_rng(0)

    state = np.array([0.0, 0.0, 0.0])
    states = [state.copy()]
    times = []

    for i in range(n_steps):
        ref_window = reference[i: i + N_HORIZON + 1]
        if len(ref_window) < N_HORIZON + 1:
            ref_window = np.pad(
                ref_window,
                ((0, N_HORIZON + 1 - len(ref_window)), (0, 0)),
                mode="edge",
            )

        t0 = time.perf_counter()
        u, info = ctrl.compute_control(state, ref_window)
        times.append(time.perf_counter() - t0)

        # 상태 전이 (+ 선택적 잡음)
        state_dot = model.forward_dynamics(state, u)
        state = state + state_dot * DT
        if noise_std > 0:
            state += rng.normal(0, noise_std, state.shape)
        states.append(state.copy())

    states = np.array(states)
    ref_states = reference[:n_steps + 1]
    errors = np.sqrt(np.mean((states[:, :2] - ref_states[:, :2]) ** 2, axis=1))

    return {
        "states": states,
        "rmse": float(np.mean(errors)),
        "rmse_final": float(errors[-1]),
        "mean_time_ms": float(np.mean(times) * 1000),
        "rollout_mode": info.get("rollout_mode", "unknown"),
        "koopman_rmse": info.get("koopman_last_rmse"),
    }


def run_scenario(name: str, noise_std: float = 0.0) -> dict:
    """시나리오 실행 및 비교."""
    rng = np.random.default_rng(42)
    model = DifferentialDriveKinematic()
    reference = make_reference(N_STEPS, DT)

    print(f"\n{'─'*60}")
    print(f"  시나리오: {name}")
    print(f"{'─'*60}")

    results = {}

    # ── Vanilla MPPI ──────────────────────────────────────────────
    vanilla_params = make_params("vanilla")
    vanilla_ctrl = MPPIController(model, vanilla_params)
    vanilla_result = run_simulation(model, vanilla_ctrl, reference, N_STEPS,
                                    noise_std=noise_std, rng=rng)
    results["Vanilla MPPI"] = vanilla_result
    print(f"  Vanilla MPPI  │ RMSE: {vanilla_result['rmse']:.4f}m"
          f" │ {vanilla_result['mean_time_ms']:.1f}ms/step")

    # ── Koopman MPPI (RBF, 미학습) ─────────────────────────────────
    koopman_params = make_params("koopman", lift_fn="rbf", lift_dim=32)
    koopman_ctrl = KoopmanMPPIController(model, koopman_params)
    k_untrained = run_simulation(model, koopman_ctrl, reference, N_STEPS,
                                 noise_std=noise_std, rng=rng)
    results["Koopman MPPI (미학습)"] = k_untrained
    print(f"  Koopman (미학습) │ RMSE: {k_untrained['rmse']:.4f}m"
          f" │ {k_untrained['mean_time_ms']:.1f}ms/step"
          f" │ mode: {k_untrained['rollout_mode']}")

    # ── Koopman MPPI (RBF, EDMD 학습) ─────────────────────────────
    koopman_params2 = make_params("koopman", lift_fn="rbf", lift_dim=32)
    koopman_trained_ctrl = KoopmanMPPIController(model, koopman_params2)

    # 학습 데이터 수집 + EDMD 학습
    X, U, X_next = collect_training_data(model, N_TRAIN, rng)
    t_fit_start = time.perf_counter()
    fit_rmse = koopman_trained_ctrl.fit_edmd(X, U, X_next)
    fit_time = time.perf_counter() - t_fit_start
    print(f"  [EDMD 학습] n={N_TRAIN}, fit_rmse={fit_rmse:.6f}, time={fit_time*1000:.1f}ms")

    k_trained = run_simulation(model, koopman_trained_ctrl, reference, N_STEPS,
                               noise_std=noise_std, rng=rng)
    results["Koopman MPPI (EDMD학습)"] = k_trained
    print(f"  Koopman (학습) │ RMSE: {k_trained['rmse']:.4f}m"
          f" │ {k_trained['mean_time_ms']:.1f}ms/step"
          f" │ mode: {k_trained['rollout_mode']}")

    # ── Koopman MPPI (Poly) ───────────────────────────────────────
    koopman_poly_params = make_params("koopman", lift_fn="poly")
    koopman_poly_ctrl = KoopmanMPPIController(model, koopman_poly_params)
    X2, U2, X2_next = collect_training_data(model, N_TRAIN, rng)
    koopman_poly_ctrl.fit_edmd(X2, U2, X2_next)
    k_poly = run_simulation(model, koopman_poly_ctrl, reference, N_STEPS,
                            noise_std=noise_std, rng=rng)
    results["Koopman MPPI (Poly)"] = k_poly
    print(f"  Koopman (Poly)│ RMSE: {k_poly['rmse']:.4f}m"
          f" │ {k_poly['mean_time_ms']:.1f}ms/step"
          f" │ phi_dim={koopman_poly_ctrl.phi_dim}")

    return results


def plot_results(all_results: dict, reference: np.ndarray, show: bool = True):
    """결과 시각화."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib 미설치 — 플롯 스킵")
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Koopman MPPI vs Vanilla MPPI 벤치마크", fontsize=14, fontweight="bold")

    n_scenarios = len(all_results)
    gs = gridspec.GridSpec(2, n_scenarios, hspace=0.4, wspace=0.35)

    colors = {
        "Vanilla MPPI": "tab:blue",
        "Koopman MPPI (미학습)": "tab:orange",
        "Koopman MPPI (EDMD학습)": "tab:green",
        "Koopman MPPI (Poly)": "tab:red",
    }

    for col, (scenario_name, results) in enumerate(all_results.items()):
        # 궤적 플롯
        ax_traj = fig.add_subplot(gs[0, col])
        ax_traj.plot(reference[:, 0], reference[:, 1], "k--", lw=2,
                     label="Reference", alpha=0.6)
        for ctrl_name, r in results.items():
            ax_traj.plot(r["states"][:, 0], r["states"][:, 1],
                         color=colors[ctrl_name], label=ctrl_name, lw=1.5)
        ax_traj.set_title(f"{scenario_name}\n궤적 비교")
        ax_traj.set_xlabel("X (m)")
        ax_traj.set_ylabel("Y (m)")
        ax_traj.legend(fontsize=6)
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.3)

        # RMSE 바 차트
        ax_bar = fig.add_subplot(gs[1, col])
        ctrl_names = list(results.keys())
        rmses = [results[n]["rmse"] for n in ctrl_names]
        bar_colors = [colors[n] for n in ctrl_names]
        bars = ax_bar.bar(range(len(ctrl_names)), rmses, color=bar_colors, alpha=0.8)
        ax_bar.set_xticks(range(len(ctrl_names)))
        ax_bar.set_xticklabels(
            [n.replace(" MPPI", "").replace("Koopman ", "K.") for n in ctrl_names],
            rotation=15, fontsize=7
        )
        ax_bar.set_ylabel("평균 RMSE (m)")
        ax_bar.set_title("RMSE 비교")
        ax_bar.grid(True, alpha=0.3, axis="y")

        for bar, rmse in zip(bars, rmses):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{rmse:.3f}", ha="center", va="bottom", fontsize=7)

    import os
    os.makedirs("plots", exist_ok=True)
    save_path = "plots/koopman_mppi_benchmark.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\n  플롯 저장: {save_path}")

    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Koopman MPPI 벤치마크")
    parser.add_argument("--no-plot", action="store_true", help="플롯 비활성화")
    parser.add_argument("--scenario", default="all",
                        choices=["clean", "noisy", "mismatch", "all"],
                        help="실행할 시나리오")
    args = parser.parse_args()

    print("=" * 60)
    print("  Koopman MPPI 벤치마크")
    print("=" * 60)
    print(f"  N_TRAIN={N_TRAIN}, K={K_SAMPLES}, N={N_HORIZON}, steps={N_STEPS}")

    scenarios = {
        "clean": ("Clean (잡음 없음)", 0.0),
        "noisy": ("Noisy (σ=0.02)", 0.02),
    }

    if args.scenario == "all":
        selected = scenarios
    else:
        selected = {args.scenario: scenarios[args.scenario]}

    reference = make_reference(N_STEPS, DT)
    all_results = {}

    for key, (name, noise) in selected.items():
        results = run_scenario(name, noise_std=noise)
        all_results[name] = results

    # 요약 테이블
    print(f"\n{'='*60}")
    print("  최종 요약")
    print(f"{'='*60}")
    print(f"  {'시나리오':<20} {'컨트롤러':<28} {'RMSE':>8} {'시간(ms)':>10}")
    print(f"  {'-'*68}")
    for scenario_name, results in all_results.items():
        for ctrl_name, r in results.items():
            print(f"  {scenario_name:<20} {ctrl_name:<28} {r['rmse']:>8.4f} {r['mean_time_ms']:>10.1f}")

    if not args.no_plot:
        plot_results(all_results, reference, show=True)


if __name__ == "__main__":
    main()
