"""
ALPaCA Trainer — Feature Extractor + Bayesian Prior 메타 학습

MAMLTrainer를 상속하여 태스크 생성, 데이터 생성, norm_stats를 재사용.
MLP를 feature_extractor + linear last layer로 분리하여
support set에서 Bayesian update → query loss → backprop through features.

Usage:
    trainer = ALPaCATrainer(state_dim=5, control_dim=2)
    trainer.meta_train(n_iterations=1000)
    trainer.save_meta_model("alpaca_meta_model.pth")
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mppi_controller.learning.maml_trainer import MAMLTrainer
from mppi_controller.models.learned.alpaca_dynamics import FeatureExtractor


class ALPaCATrainer(MAMLTrainer):
    """
    ALPaCA 메타 학습 파이프라인.

    MAMLTrainer를 상속하되 inner loop를 closed-form Bayesian update로 대체.

    구조:
    - Feature extractor φ(x, u): MLP (메타 학습)
    - Last layer W: Bayesian linear regression (closed-form 적응)
    - Prior: μ₀, Λ₀, β (메타 학습)

    메타 학습:
    1. Support set에서 φ 추출
    2. Bayesian update → posterior μ_n
    3. Query set에서 loss = ||y - μ_n·φ||²
    4. Backprop through φ (feature extractor 업데이트)
    5. μ₀, Λ₀, β도 end-to-end 업데이트

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        hidden_dims: feature extractor hidden layers
        feature_dim: 특징 벡터 차원
        meta_lr: 메타 학습률
        task_batch_size: 메타 배치당 태스크 수
        support_size: support set 크기
        query_size: query set 크기
        beta_init: 초기 noise precision
        device: 'cpu' or 'cuda'
        save_dir: 모델 저장 경로
    """

    def __init__(
        self,
        state_dim: int = 5,
        control_dim: int = 2,
        hidden_dims: List[int] = None,
        feature_dim: int = 64,
        meta_lr: float = 1e-3,
        task_batch_size: int = 4,
        support_size: int = 50,
        query_size: int = 50,
        beta_init: float = 1.0,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    ):
        if hidden_dims is None:
            hidden_dims = [128, 128]

        # MAMLTrainer의 __init__을 우회하고 필요한 속성만 설정
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dims = hidden_dims
        self.feature_dim = feature_dim
        self.meta_lr = meta_lr
        self.task_batch_size = task_batch_size
        self.support_size = support_size
        self.query_size = query_size
        self.device = torch.device(device)
        self.save_dir = save_dir
        self.norm_stats = None
        self.history = {"meta_loss": []}

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        input_dim = state_dim + control_dim

        # Feature extractor (메타 학습 대상)
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Bayesian prior (메타 학습 대상)
        self._mu_0 = nn.Parameter(
            torch.zeros(state_dim, feature_dim, device=self.device)
        )
        self._Lambda_0_raw = nn.Parameter(
            torch.eye(feature_dim, device=self.device)
        )
        self._log_beta = nn.Parameter(
            torch.tensor(np.log(beta_init), dtype=torch.float32, device=self.device)
        )

        # Optimizer: feature extractor + prior
        self.meta_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            [self._mu_0, self._Lambda_0_raw, self._log_beta],
            lr=meta_lr,
        )

        # MAMLTrainer's model is not used; set to None
        self.model = None
        self.inner_lr = None
        self.inner_steps = None

    @property
    def _Lambda_0(self):
        """Λ₀ = L·Lᵀ + εI (양정치 보장)."""
        return self._Lambda_0_raw @ self._Lambda_0_raw.T + 1e-4 * torch.eye(
            self.feature_dim, device=self.device
        )

    @property
    def _beta(self):
        """β = exp(log_beta) (양수 보장)."""
        return torch.exp(self._log_beta)

    def _bayesian_update(self, phi, targets):
        """
        Closed-form Bayesian update (differentiable).

        Args:
            phi: (M, d) features (torch, requires_grad)
            targets: (M, nx) normalized targets (torch)

        Returns:
            mu_n: (nx, d) posterior mean
            Lambda_n: (d, d) posterior precision
        """
        Lambda_0 = self._Lambda_0
        mu_0 = self._mu_0
        beta = self._beta

        # Λ_n = Λ₀ + β·Φᵀ·Φ
        Lambda_n = Lambda_0 + beta * (phi.T @ phi)

        # Λ_n⁻¹
        Lambda_n_inv = torch.linalg.inv(Lambda_n + 1e-6 * torch.eye(
            self.feature_dim, device=self.device
        ))

        # μ_n = (Λ_n⁻¹ · (Λ₀·μ₀ᵀ + β·Φᵀ·Y))ᵀ
        prior_term = Lambda_0 @ mu_0.T  # (d, nx)
        data_term = beta * (phi.T @ targets)  # (d, nx)
        mu_n = (Lambda_n_inv @ (prior_term + data_term)).T  # (nx, d)

        return mu_n, Lambda_n

    def meta_train(self, n_iterations: int = 1000, verbose: bool = True):
        """
        ALPaCA 메타 학습.

        각 iteration에서:
        1. 태스크 샘플링
        2. Support → Bayesian update (differentiable)
        3. Query → loss (μ_n·φ vs y)
        4. Backprop through feature extractor + prior

        Args:
            n_iterations: 반복 횟수
            verbose: 진행 상황 출력
        """
        if verbose:
            print(f"\n  ALPaCA Meta-Training")
            print(f"    Iterations: {n_iterations}")
            print(f"    Feature dim: {self.feature_dim}")
            print(f"    Task batch: {self.task_batch_size}")
            print(f"    Support/Query: {self.support_size}/{self.query_size}")
            print(f"    Meta LR: {self.meta_lr}")

        # norm_stats 계산
        n_total = self.support_size + self.query_size
        gen_fn = self._generate_task_data_5d if self.state_dim == 5 else self._generate_task_data

        if verbose:
            print("    Computing normalization stats...")

        pre_data = []
        for _ in range(self.task_batch_size * 2):
            task = self._sample_task()
            data = gen_fn(task, n_total)
            pre_data.append(data)
        self.norm_stats = self._compute_norm_stats(pre_data)

        if verbose:
            print("    Starting meta-training...")

        self.feature_extractor.train()

        for iteration in range(n_iterations):
            total_query_loss = 0.0
            self.meta_optimizer.zero_grad()

            for _ in range(self.task_batch_size):
                task = self._sample_task()
                states, controls, next_states = gen_fn(task, n_total)

                # Support/Query split
                support_s = states[:self.support_size]
                support_c = controls[:self.support_size]
                support_ns = next_states[:self.support_size]
                query_s = states[self.support_size:]
                query_c = controls[self.support_size:]
                query_ns = next_states[self.support_size:]

                # Prepare inputs and targets
                support_inputs, support_targets = self._prepare_batch_alpaca(
                    support_s, support_c, support_ns
                )
                query_inputs, query_targets = self._prepare_batch_alpaca(
                    query_s, query_c, query_ns
                )

                # Feature extraction (differentiable)
                phi_support = self.feature_extractor(support_inputs)
                phi_query = self.feature_extractor(query_inputs)

                # Bayesian update on support (differentiable)
                mu_n, _ = self._bayesian_update(phi_support, support_targets)

                # Query prediction: y = φ · μ_nᵀ
                query_pred = phi_query @ mu_n.T  # (M, nx)
                query_loss = F.mse_loss(query_pred, query_targets)

                query_loss.backward()
                total_query_loss += query_loss.item()

            # Scale gradients
            for param in list(self.feature_extractor.parameters()) + [self._mu_0, self._Lambda_0_raw, self._log_beta]:
                if param.grad is not None:
                    param.grad /= self.task_batch_size

            torch.nn.utils.clip_grad_norm_(
                list(self.feature_extractor.parameters()) +
                [self._mu_0, self._Lambda_0_raw, self._log_beta],
                max_norm=1.0,
            )
            self.meta_optimizer.step()

            avg_loss = total_query_loss / self.task_batch_size
            self.history["meta_loss"].append(avg_loss)

            if verbose and (iteration + 1) % 50 == 0:
                print(
                    f"    Iter {iteration + 1}/{n_iterations} | "
                    f"Meta Loss: {avg_loss:.6f} | "
                    f"β={self._beta.item():.3f}"
                )

        if verbose:
            final_loss = self.history["meta_loss"][-1] if self.history["meta_loss"] else float("nan")
            print(f"\n    ALPaCA meta-training complete. Final loss: {final_loss:.6f}")

        self.feature_extractor.eval()

    def _prepare_batch_alpaca(self, states, controls, next_states, dt=0.05):
        """데이터를 normalized tensor로 변환 (ALPaCA용)."""
        targets = (next_states - states) / dt
        # Angle wrapping
        if states.shape[1] >= 3:
            theta_diff = next_states[:, 2] - states[:, 2]
            theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
            targets[:, 2] = theta_diff / dt

        if self.norm_stats is not None:
            state_norm = (states - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_norm = (controls - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
            targets_norm = (targets - self.norm_stats["state_dot_mean"]) / self.norm_stats["state_dot_std"]
        else:
            state_norm = states
            control_norm = controls
            targets_norm = targets

        inputs = np.concatenate([state_norm, control_norm], axis=1)
        inputs_t = torch.FloatTensor(inputs).to(self.device)
        targets_t = torch.FloatTensor(targets_norm).to(self.device)

        return inputs_t, targets_t

    def save_meta_model(self, filename: str = "alpaca_meta_model.pth"):
        """ALPaCA 메타 모델 저장."""
        filepath = os.path.join(self.save_dir, filename)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # Prior를 numpy로 변환
        Lambda_0_np = self._Lambda_0.detach().cpu().numpy()
        mu_0_np = self._mu_0.detach().cpu().numpy()
        beta_np = float(self._beta.detach().cpu().item())

        torch.save({
            "feature_extractor_state_dict": self.feature_extractor.state_dict(),
            "mu_0": mu_0_np,
            "Lambda_0": Lambda_0_np,
            "beta": beta_np,
            "norm_stats": self.norm_stats,
            "history": self.history,
            "config": {
                "state_dim": self.state_dim,
                "control_dim": self.control_dim,
                "hidden_dims": self.hidden_dims,
                "feature_dim": self.feature_dim,
            },
        }, filepath)

        print(f"  [ALPaCATrainer] Meta model saved to {filepath}")

    def load_meta_model(self, filename: str = "alpaca_meta_model.pth"):
        """ALPaCA 메타 모델 로드."""
        filepath = os.path.join(self.save_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ALPaCA meta model not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        self.feature_extractor.eval()

        # Prior 복원
        mu_0_np = checkpoint["mu_0"]
        Lambda_0_np = checkpoint["Lambda_0"]

        with torch.no_grad():
            self._mu_0.copy_(torch.FloatTensor(mu_0_np).to(self.device))
            # Lambda_0_raw를 Cholesky factor로 설정
            L = torch.linalg.cholesky(
                torch.FloatTensor(Lambda_0_np).to(self.device) +
                1e-4 * torch.eye(self.feature_dim, device=self.device)
            )
            self._Lambda_0_raw.copy_(L)
            self._log_beta.copy_(
                torch.tensor(np.log(checkpoint["beta"]), device=self.device)
            )

        self.norm_stats = checkpoint.get("norm_stats")
        self.history = checkpoint.get("history", {"meta_loss": []})

        print(f"  [ALPaCATrainer] Meta model loaded from {filepath}")
