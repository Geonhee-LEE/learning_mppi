#!/usr/bin/env python3
"""
Evidential Deep Learning (EDL) vs Ensemble vs MC-Dropout 벤치마크

단일 forward pass 불확실성 정량화 비교:
  1. Ensemble (M=5)     — M개 모델 평균/분산
  2. MC-Dropout (M=20)  — 단일 모델, M회 dropout 샘플
  3. EDL (1-pass)        — 단일 forward pass, NIG 파라미터

시나리오:
  A. clean — 학습 분포 내 평가 (기준선)
  B. noisy — 관측 노이즈 추가 (aleatoric 테스트)
  C. ood   — 학습 범위 외 extrapolation (epistemic 테스트)

측정:
  - Prediction RMSE
  - Uncertainty Calibration
  - OOD Detection
  - Aleatoric/Epistemic 분해 (EDL only)
  - Inference Speed 비교

Usage:
    PYTHONPATH=. python examples/comparison/edl_benchmark.py
    PYTHONPATH=. python examples/comparison/edl_benchmark.py --scenario noisy
    PYTHONPATH=. python examples/comparison/edl_benchmark.py --scenario ood
    PYTHONPATH=. python examples/comparison/edl_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/edl_benchmark.py --no-plot
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

from mppi_controller.learning.evidential_trainer import EvidentialTrainer
from mppi_controller.learning.ensemble_trainer import EnsembleTrainer
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer


# ── 데이터 생성 ────────────────────────────────────────────────


def generate_dynamics_data(N=1000, state_dim=3, control_dim=2, seed=42,
                           noise_std=0.01):
    """비선형 동역학 합성 데이터"""
    np.random.seed(seed)
    states = np.random.randn(N, state_dim).astype(np.float32) * 2.0
    controls = np.random.randn(N, control_dim).astype(np.float32) * 1.0

    # Nonlinear dynamics: dx = A*[s,u] + sin(s) * 0.1
    input_dim = state_dim + control_dim
    A = np.random.RandomState(0).randn(state_dim, input_dim).astype(np.float32) * 0.3
    inputs = np.concatenate([states, controls], axis=1)
    targets = inputs @ A.T + np.sin(states) * 0.1
    targets += np.random.randn(N, state_dim).astype(np.float32) * noise_std

    return states, controls, inputs, targets


def split_data(inputs, targets, train_ratio=0.8):
    """학습/검증 분리"""
    N = len(inputs)
    idx = np.random.permutation(N)
    n_train = int(N * train_ratio)
    return (
        inputs[idx[:n_train]], targets[idx[:n_train]],
        inputs[idx[n_train:]], targets[idx[n_train:]],
    )


def get_norm_stats(state_dim=3, control_dim=2):
    return {
        "state_mean": np.zeros(state_dim, dtype=np.float32),
        "state_std": np.ones(state_dim, dtype=np.float32),
        "control_mean": np.zeros(control_dim, dtype=np.float32),
        "control_std": np.ones(control_dim, dtype=np.float32),
        "state_dot_mean": np.zeros(state_dim, dtype=np.float32),
        "state_dot_std": np.ones(state_dim, dtype=np.float32),
    }


# ── 시나리오 ────────────────────────────────────────────────────


def get_scenarios():
    return {
        "clean": {
            "name": "Clean (in-distribution)",
            "train_noise": 0.01,
            "test_noise": 0.01,
            "ood": False,
        },
        "noisy": {
            "name": "Noisy (high aleatoric)",
            "train_noise": 0.05,
            "test_noise": 0.1,
            "ood": False,
        },
        "ood": {
            "name": "OOD (extrapolation)",
            "train_noise": 0.01,
            "test_noise": 0.01,
            "ood": True,
        },
    }


# ── 학습 + 추론 ────────────────────────────────────────────────


def train_and_evaluate(scenario_cfg, epochs=100, seed=42, verbose=True):
    """3개 모델 학습 + 평가"""
    import tempfile
    np.random.seed(seed)

    state_dim, control_dim = 3, 2
    hidden_dims = [64, 64]
    norm_stats = get_norm_stats()

    # 학습 데이터
    states_tr, controls_tr, inputs_tr, targets_tr = generate_dynamics_data(
        N=800, noise_std=scenario_cfg["train_noise"], seed=seed,
    )
    tr_in, tr_tgt, val_in, val_tgt = split_data(inputs_tr, targets_tr)

    # 테스트 데이터
    if scenario_cfg["ood"]:
        # OOD: 학습 범위의 3~5배
        states_te = np.random.randn(200, state_dim).astype(np.float32) * 8.0
        controls_te = np.random.randn(200, control_dim).astype(np.float32) * 4.0
        inputs_te = np.concatenate([states_te, controls_te], axis=1)
        A = np.random.RandomState(0).randn(state_dim, state_dim + control_dim).astype(np.float32) * 0.3
        targets_te = inputs_te @ A.T + np.sin(states_te) * 0.1
        targets_te += np.random.randn(200, state_dim).astype(np.float32) * scenario_cfg["test_noise"]
        # Also keep ID test data for comparison
        states_id, controls_id, inputs_id, targets_id = generate_dynamics_data(
            N=200, noise_std=scenario_cfg["test_noise"], seed=seed + 100,
        )
    else:
        states_te, controls_te, inputs_te, targets_te = generate_dynamics_data(
            N=200, noise_std=scenario_cfg["test_noise"], seed=seed + 100,
        )
        states_id = states_te
        controls_id = controls_te
        inputs_id = inputs_te
        targets_id = targets_te

    results = {}

    # ─── 1. Ensemble (M=5) ───
    if verbose:
        print("\n[1/3] Training Ensemble (M=5)...")
    tmpdir = tempfile.mkdtemp()
    ens_trainer = EnsembleTrainer(
        state_dim=state_dim, control_dim=control_dim,
        num_models=5, hidden_dims=hidden_dims,
        save_dir=tmpdir,
    )
    ens_trainer.train(tr_in, tr_tgt, val_in, val_tgt, norm_stats,
                      epochs=epochs, verbose=False)

    # Inference timing
    t0 = time.perf_counter()
    n_eval = 100
    for i in range(n_eval):
        ens_trainer.predict(states_te[i % len(states_te)], controls_te[i % len(controls_te)],
                            return_uncertainty=True)
    ens_time = (time.perf_counter() - t0) / n_eval * 1000  # ms

    # Predictions
    ens_preds, ens_stds = [], []
    for i in range(len(states_te)):
        mean, std = ens_trainer.predict(states_te[i], controls_te[i], return_uncertainty=True)
        ens_preds.append(mean)
        ens_stds.append(std)
    ens_preds = np.array(ens_preds)
    ens_stds = np.array(ens_stds)

    results["Ensemble (M=5)"] = {
        "preds": ens_preds, "stds": ens_stds,
        "time_ms": ens_time,
    }

    # ─── 2. MC-Dropout (M=20) ───
    if verbose:
        print("[2/3] Training MC-Dropout (M=20)...")
    tmpdir2 = tempfile.mkdtemp()
    mc_trainer = NeuralNetworkTrainer(
        state_dim=state_dim, control_dim=control_dim,
        hidden_dims=hidden_dims, dropout_rate=0.2,
        save_dir=tmpdir2,
    )
    mc_trainer.train(tr_in, tr_tgt, val_in, val_tgt, norm_stats,
                     epochs=epochs, verbose=False)

    mc_model = mc_trainer.model
    mc_num_samples = 20

    def mc_predict_with_uncertainty(state, control):
        """MC-Dropout 예측"""
        if mc_trainer.norm_stats is not None:
            s_n = (state - mc_trainer.norm_stats["state_mean"]) / mc_trainer.norm_stats["state_std"]
            c_n = (control - mc_trainer.norm_stats["control_mean"]) / mc_trainer.norm_stats["control_std"]
        else:
            s_n, c_n = state, control
        inp = np.concatenate([s_n, c_n])
        import torch
        inp_t = torch.FloatTensor(inp).unsqueeze(0)
        mc_model.train()
        preds_list = []
        with torch.no_grad():
            for _ in range(mc_num_samples):
                preds_list.append(mc_model(inp_t).cpu().numpy())
        preds_arr = np.array(preds_list).squeeze(1)
        return preds_arr.mean(0), preds_arr.std(0)

    t0 = time.perf_counter()
    for i in range(n_eval):
        mc_predict_with_uncertainty(states_te[i % len(states_te)], controls_te[i % len(controls_te)])
    mc_time = (time.perf_counter() - t0) / n_eval * 1000

    mc_preds, mc_stds = [], []
    for i in range(len(states_te)):
        mean, std = mc_predict_with_uncertainty(states_te[i], controls_te[i])
        mc_preds.append(mean)
        mc_stds.append(std)
    mc_preds = np.array(mc_preds)
    mc_stds = np.array(mc_stds)

    results["MC-Dropout (M=20)"] = {
        "preds": mc_preds, "stds": mc_stds,
        "time_ms": mc_time,
    }

    # ─── 3. EDL (1-pass) ───
    if verbose:
        print("[3/3] Training EDL (1-pass)...")
    tmpdir3 = tempfile.mkdtemp()
    edl_trainer = EvidentialTrainer(
        state_dim=state_dim, control_dim=control_dim,
        hidden_dims=hidden_dims, save_dir=tmpdir3,
        lambda_reg=0.01, annealing_epochs=epochs // 2,
    )
    edl_trainer.train(tr_in, tr_tgt, val_in, val_tgt, norm_stats,
                      epochs=epochs, verbose=False)

    t0 = time.perf_counter()
    for i in range(n_eval):
        edl_trainer.predict_with_uncertainty(
            states_te[i % len(states_te)], controls_te[i % len(controls_te)],
        )
    edl_time = (time.perf_counter() - t0) / n_eval * 1000

    edl_preds, edl_stds, edl_ale, edl_epi = [], [], [], []
    for i in range(len(states_te)):
        mean, ale, epi = edl_trainer.predict_with_decomposed_uncertainty(
            states_te[i], controls_te[i],
        )
        _, std = edl_trainer.predict_with_uncertainty(states_te[i], controls_te[i])
        edl_preds.append(mean)
        edl_stds.append(std)
        edl_ale.append(ale)
        edl_epi.append(epi)
    edl_preds = np.array(edl_preds)
    edl_stds = np.array(edl_stds)
    edl_ale = np.array(edl_ale)
    edl_epi = np.array(edl_epi)

    results["EDL (1-pass)"] = {
        "preds": edl_preds, "stds": edl_stds,
        "aleatoric": edl_ale, "epistemic": edl_epi,
        "time_ms": edl_time,
    }

    # OOD ID preds for EDL (for OOD detection plot)
    if scenario_cfg["ood"]:
        edl_stds_id = []
        for i in range(len(states_id)):
            _, std_id = edl_trainer.predict_with_uncertainty(states_id[i], controls_id[i])
            edl_stds_id.append(std_id)
        results["EDL (1-pass)"]["stds_id"] = np.array(edl_stds_id)

    return results, targets_te


# ── 플롯 ────────────────────────────────────────────────────────


def plot_results(results, targets, scenario_name, save_path):
    """6-panel 플롯"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"EDL Benchmark — {scenario_name}", fontsize=14, fontweight="bold")

    colors = {"Ensemble (M=5)": "#e74c3c", "MC-Dropout (M=20)": "#3498db", "EDL (1-pass)": "#2ecc71"}

    # 1. Prediction vs Ground Truth (dim 0)
    ax = axes[0, 0]
    for name, r in results.items():
        errors = np.sqrt(np.mean((r["preds"] - targets) ** 2, axis=1))
        rmse = np.sqrt(np.mean(errors ** 2))
        ax.scatter(targets[:, 0], r["preds"][:, 0], alpha=0.3, s=10,
                   color=colors[name], label=f"{name} (RMSE={rmse:.4f})")
    mn, mx = targets[:, 0].min(), targets[:, 0].max()
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label="Perfect")
    ax.set_xlabel("Ground Truth (dim 0)")
    ax.set_ylabel("Prediction (dim 0)")
    ax.set_title("Prediction vs Ground Truth")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Uncertainty Calibration (expected vs observed)
    ax = axes[0, 1]
    quantiles = np.linspace(0.05, 0.95, 19)
    for name, r in results.items():
        errors = np.abs(r["preds"] - targets)
        # Mean std per sample
        mean_std = r["stds"].mean(axis=1)
        mean_err = errors.mean(axis=1)
        observed = []
        for q in quantiles:
            threshold = np.quantile(mean_std, q)
            mask = mean_std <= threshold
            if mask.sum() > 0:
                observed.append(np.mean(mean_err[mask] <= threshold))
            else:
                observed.append(q)
        ax.plot(quantiles, observed, '-o', markersize=3, color=colors[name], label=name)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Ideal")
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title("Uncertainty Calibration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Uncertainty vs Prediction Error
    ax = axes[0, 2]
    for name, r in results.items():
        mean_err = np.sqrt(np.mean((r["preds"] - targets) ** 2, axis=1))
        mean_std = r["stds"].mean(axis=1)
        ax.scatter(mean_std, mean_err, alpha=0.3, s=10, color=colors[name], label=name)
    ax.set_xlabel("Mean Uncertainty (std)")
    ax.set_ylabel("Prediction Error (RMSE)")
    ax.set_title("Uncertainty vs Error Correlation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. OOD Detection (uncertainty histogram)
    ax = axes[1, 0]
    edl = results.get("EDL (1-pass)", {})
    if "stds_id" in edl:
        std_id = edl["stds_id"].mean(axis=1)
        std_ood = edl["stds"].mean(axis=1)
        ax.hist(std_id, bins=30, alpha=0.6, color="#2ecc71", label="In-Distribution", density=True)
        ax.hist(std_ood, bins=30, alpha=0.6, color="#e74c3c", label="OOD", density=True)
        ax.set_title("OOD Detection (EDL uncertainty)")
    else:
        for name, r in results.items():
            ax.hist(r["stds"].mean(axis=1), bins=30, alpha=0.5, color=colors[name],
                    label=name, density=True)
        ax.set_title("Uncertainty Distribution")
    ax.set_xlabel("Mean Uncertainty")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. Aleatoric/Epistemic Decomposition (EDL only)
    ax = axes[1, 1]
    if "aleatoric" in edl:
        ale_mean = edl["aleatoric"].mean(axis=1)
        epi_mean = edl["epistemic"].mean(axis=1)
        ax.scatter(ale_mean, epi_mean, alpha=0.4, s=15, color="#2ecc71")
        ax.set_xlabel("Aleatoric Uncertainty")
        ax.set_ylabel("Epistemic Uncertainty")
        ax.set_title("EDL: Aleatoric vs Epistemic")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No decomposition data", ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Aleatoric vs Epistemic")

    # 6. Inference Speed
    ax = axes[1, 2]
    names = list(results.keys())
    times = [results[n]["time_ms"] for n in names]
    bars = ax.bar(range(len(names)), times,
                  color=[colors[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" ")[0] for n in names], fontsize=9)
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Speed Comparison")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{t:.2f}ms", ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")


def print_summary(results, targets, scenario_name):
    """콘솔 요약"""
    print(f"\n{'=' * 60}")
    print(f"  {scenario_name}")
    print(f"{'=' * 60}")
    print(f"{'Method':<22} {'RMSE':>8} {'Mean Std':>10} {'Time (ms)':>10}")
    print(f"{'-' * 52}")
    for name, r in results.items():
        rmse = np.sqrt(np.mean((r["preds"] - targets) ** 2))
        mean_std = r["stds"].mean()
        print(f"{name:<22} {rmse:>8.4f} {mean_std:>10.4f} {r['time_ms']:>10.2f}")
    print(f"{'=' * 60}")


# ── 메인 ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="EDL vs Ensemble vs MC-Dropout Benchmark")
    parser.add_argument("--scenario", choices=["clean", "noisy", "ood"], default="clean")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    scenarios = get_scenarios()
    run_list = list(scenarios.keys()) if args.all_scenarios else [args.scenario]

    for scenario_key in run_list:
        cfg = scenarios[scenario_key]
        print(f"\n{'#' * 60}")
        print(f"  Scenario: {cfg['name']}")
        print(f"{'#' * 60}")

        results, targets = train_and_evaluate(
            cfg, epochs=args.epochs, seed=args.seed,
        )

        print_summary(results, targets, cfg["name"])

        if not args.no_plot:
            save_path = f"plots/edl_benchmark_{scenario_key}.png"
            plot_results(results, targets, cfg["name"], save_path)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
