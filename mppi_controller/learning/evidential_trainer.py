#!/usr/bin/env python3
"""
Evidential Deep Learning (EDL) 동역학 모델 학습 파이프라인

단일 forward pass로 aleatoric + epistemic 불확실성 분리.
Normal-Inverse-Gamma (NIG) 분포로 예측 분포 파라미터화.

References:
    Amini et al. "Deep Evidential Regression" (NeurIPS 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


class EvidentialMLPModel(nn.Module):
    """
    Evidential MLP: 4개 출력 head로 NIG 분포 파라미터 예측

    Architecture:
        Input: [state, control] (nx + nu)
        Trunk: shared hidden layers
        Heads: gamma (mean), nu (virtual obs), alpha (IG shape), beta (IG scale)

    Args:
        input_dim: nx + nu
        output_dim: nx
        hidden_dims: 히든 레이어 차원
        activation: 'relu', 'tanh', 'elu'
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Activation
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Shared trunk
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # 4 output heads
        self.gamma_head = nn.Linear(prev_dim, output_dim)   # mean (unconstrained)
        self.nu_head = nn.Linear(prev_dim, output_dim)      # >0 (softplus + eps)
        self.alpha_head = nn.Linear(prev_dim, output_dim)   # >1 (softplus + 1)
        self.beta_head = nn.Linear(prev_dim, output_dim)    # >0 (softplus + eps)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass → NIG 파라미터

        Args:
            x: (batch, input_dim)

        Returns:
            gamma: (batch, nx) — 예측 평균
            nu: (batch, nx) — >0, 가상 관측 수
            alpha: (batch, nx) — >1, IG shape
            beta: (batch, nx) — >0, IG scale
        """
        h = self.trunk(x)

        gamma = self.gamma_head(h)
        nu = nn.functional.softplus(self.nu_head(h)) + 1e-6
        alpha = nn.functional.softplus(self.alpha_head(h)) + 1.0 + 1e-4
        beta = nn.functional.softplus(self.beta_head(h)) + 1e-6

        return gamma, nu, alpha, beta


def nig_nll_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Normal-Inverse-Gamma 음의 로그 우도

    NLL = 0.5 log(π/ν) - α log(Ω)
          + (α + 0.5) log((y-γ)²ν + Ω)
          + log(Γ(α) / Γ(α + 0.5))

    where Ω = 2β(1 + ν)

    Args:
        y: (batch, nx) 관측 타겟
        gamma, nu, alpha, beta: NIG 파라미터

    Returns:
        nll: (batch,) 배치별 NLL
    """
    omega = 2.0 * beta * (1.0 + nu)

    nll = (
        0.5 * torch.log(torch.tensor(np.pi, device=y.device) / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    )

    # sum over output dims, keep batch dim
    return nll.sum(dim=-1)


def nig_kl_regularizer(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    KL-inspired evidence regularizer: 틀린 예측에 evidence 페널티

    L_reg = |y - γ| · (2ν + α)

    Args:
        y, gamma, nu, alpha, beta: NIG 분포 파라미터

    Returns:
        reg: (batch,) 배치별 정규화 항
    """
    error = torch.abs(y - gamma)
    reg = error * (2.0 * nu + alpha)
    return reg.sum(dim=-1)


class EvidentialLoss(nn.Module):
    """
    EDL 총 손실: NIG NLL + λ × KL regularizer

    λ 어닐링 지원: λ_eff = λ × min(epoch / annealing_epochs, 1)

    Args:
        lambda_reg: KL 정규화 가중치
        annealing: 어닐링 사용 여부
        annealing_epochs: 어닐링 완료 에폭
    """

    def __init__(
        self,
        lambda_reg: float = 0.01,
        annealing: bool = True,
        annealing_epochs: int = 50,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.annealing = annealing
        self.annealing_epochs = annealing_epochs

    def forward(
        self,
        y: torch.Tensor,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            y: (batch, nx) 타겟
            gamma, nu, alpha, beta: NIG 파라미터
            epoch: 현재 에폭 (어닐링 계수 계산)

        Returns:
            loss: scalar
        """
        nll = nig_nll_loss(y, gamma, nu, alpha, beta)
        reg = nig_kl_regularizer(y, gamma, nu, alpha, beta)

        if self.annealing and self.annealing_epochs > 0:
            coeff = min(epoch / self.annealing_epochs, 1.0)
        else:
            coeff = 1.0

        loss = nll + self.lambda_reg * coeff * reg
        return loss.mean()


class EvidentialTrainer:
    """
    Evidential Deep Learning 학습 파이프라인

    NeuralNetworkTrainer API 준수, 단일 forward pass로
    aleatoric + epistemic 불확실성 분리.

    사용 예시:
        trainer = EvidentialTrainer(state_dim=3, control_dim=2)
        trainer.train(train_inputs, train_targets, val_inputs, val_targets, norm_stats)
        mean, std = trainer.predict_with_uncertainty(state, control)
        mean, aleatoric, epistemic = trainer.predict_with_decomposed_uncertainty(state, control)
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        lambda_reg: float = 0.01,
        annealing: bool = True,
        annealing_epochs: int = 50,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.lambda_reg = lambda_reg
        self.annealing = annealing
        self.annealing_epochs = annealing_epochs
        self.device = torch.device(device)

        input_dim = state_dim + control_dim
        output_dim = state_dim

        self.model = EvidentialMLPModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = EvidentialLoss(
            lambda_reg=lambda_reg,
            annealing=annealing,
            annealing_epochs=annealing_epochs,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10,
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        self.norm_stats: Optional[Dict[str, np.ndarray]] = None
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        val_inputs: np.ndarray,
        val_targets: np.ndarray,
        norm_stats: Dict[str, np.ndarray],
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        EDL 모델 학습

        Args:
            train_inputs: (N_train, nx + nu) 정규화된 입력
            train_targets: (N_train, nx) 정규화된 타겟
            val_inputs: (N_val, nx + nu)
            val_targets: (N_val, nx)
            norm_stats: 정규화 통계
            epochs: 에폭 수
            batch_size: 배치 크기
            early_stopping_patience: 조기 종료 인내
            verbose: 진행 출력

        Returns:
            history: 학습 이력
        """
        self.norm_stats = norm_stats

        train_inputs_t = torch.FloatTensor(train_inputs).to(self.device)
        train_targets_t = torch.FloatTensor(train_targets).to(self.device)
        val_inputs_t = torch.FloatTensor(val_inputs).to(self.device)
        val_targets_t = torch.FloatTensor(val_targets).to(self.device)

        train_dataset = TensorDataset(train_inputs_t, train_targets_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_inputs, batch_targets in train_loader:
                self.optimizer.zero_grad()
                gamma, nu, alpha, beta = self.model(batch_inputs)
                loss = self.criterion(batch_targets, gamma, nu, alpha, beta, epoch=epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                gamma_v, nu_v, alpha_v, beta_v = self.model(val_inputs_t)
                val_loss = self.criterion(
                    val_targets_t, gamma_v, nu_v, alpha_v, beta_v, epoch=epoch
                ).item()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]['lr'])

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("best_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

        if verbose:
            print(f"\nTraining completed. Best val loss: {best_val_loss:.6f}")

        return self.history

    def _prepare_input(
        self, state: np.ndarray, control: np.ndarray
    ) -> Tuple[torch.Tensor, bool]:
        """입력 정규화 + 텐서 변환"""
        if self.norm_stats is not None:
            state_n = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_n = (control - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
        else:
            state_n = state
            control_n = control

        single = state.ndim == 1
        if single:
            inputs = np.concatenate([state_n, control_n])
            inputs_t = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)
        else:
            inputs = np.concatenate([state_n, control_n], axis=1)
            inputs_t = torch.FloatTensor(inputs).to(self.device)

        return inputs_t, single

    def predict(
        self,
        state: np.ndarray,
        control: np.ndarray,
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        예측 평균 (gamma)

        Args:
            state: (nx,) or (batch, nx)
            control: (nu,) or (batch, nu)
            denormalize: 역정규화 여부

        Returns:
            state_dot: (nx,) or (batch, nx)
        """
        self.model.eval()
        inputs_t, single = self._prepare_input(state, control)

        with torch.no_grad():
            gamma, _, _, _ = self.model(inputs_t)
            outputs = gamma.cpu().numpy()

        if denormalize and self.norm_stats is not None:
            outputs = outputs * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]

        if single:
            outputs = outputs.squeeze(0)
        return outputs

    def predict_with_uncertainty(
        self, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 + epistemic 불확실성

        epistemic std = sqrt(beta / (nu * (alpha - 1)))

        Args:
            state: (nx,) or (batch, nx)
            control: (nu,) or (batch, nu)

        Returns:
            mean: (nx,) or (batch, nx)
            std: (nx,) or (batch, nx) — epistemic 표준편차
        """
        self.model.eval()
        inputs_t, single = self._prepare_input(state, control)

        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(inputs_t)

            mean = gamma.cpu().numpy()
            # Epistemic variance: β / (ν(α-1))
            epistemic_var = (beta / (nu * (alpha - 1.0))).cpu().numpy()
            std = np.sqrt(np.maximum(epistemic_var, 0.0))

        if self.norm_stats is not None:
            mean = mean * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]
            std = std * self.norm_stats["state_dot_std"]

        if single:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
        return mean, std

    def predict_with_decomposed_uncertainty(
        self, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        예측 + aleatoric/epistemic 불확실성 분해

        aleatoric = sqrt(beta / (alpha - 1))       — 데이터 내재 노이즈
        epistemic = sqrt(beta / (nu * (alpha - 1))) — 모델 불확실성

        Args:
            state: (nx,) or (batch, nx)
            control: (nu,) or (batch, nu)

        Returns:
            mean: (nx,) or (batch, nx)
            aleatoric_std: (nx,) or (batch, nx)
            epistemic_std: (nx,) or (batch, nx)
        """
        self.model.eval()
        inputs_t, single = self._prepare_input(state, control)

        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(inputs_t)

            mean = gamma.cpu().numpy()
            alpha_np = alpha.cpu().numpy()
            beta_np = beta.cpu().numpy()
            nu_np = nu.cpu().numpy()

            # Aleatoric: β / (α - 1)
            aleatoric_var = beta_np / (alpha_np - 1.0)
            aleatoric_std = np.sqrt(np.maximum(aleatoric_var, 0.0))

            # Epistemic: β / (ν(α - 1))
            epistemic_var = beta_np / (nu_np * (alpha_np - 1.0))
            epistemic_std = np.sqrt(np.maximum(epistemic_var, 0.0))

        if self.norm_stats is not None:
            mean = mean * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]
            aleatoric_std = aleatoric_std * self.norm_stats["state_dot_std"]
            epistemic_std = epistemic_std * self.norm_stats["state_dot_std"]

        if single:
            mean = mean.squeeze(0)
            aleatoric_std = aleatoric_std.squeeze(0)
            epistemic_std = epistemic_std.squeeze(0)

        return mean, aleatoric_std, epistemic_std

    def save_model(self, filename: str):
        """모델 + 설정 저장"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "norm_stats": self.norm_stats,
            "history": self.history,
            "config": {
                "model_type": "evidential",
                "state_dim": self.state_dim,
                "control_dim": self.control_dim,
                "hidden_dims": self.hidden_dims,
                "activation": self.activation,
                "lambda_reg": self.lambda_reg,
                "annealing": self.annealing,
                "annealing_epochs": self.annealing_epochs,
            },
        }, filepath)

    def load_model(self, filename: str):
        """모델 로드"""
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.norm_stats = checkpoint["norm_stats"]
        self.history = checkpoint.get("history", {})
        self.model.eval()

    def get_model_summary(self) -> str:
        """모델 요약"""
        num_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"EvidentialMLPModel(\n"
            f"  Input dim: {self.model.input_dim}\n"
            f"  Output dim: {self.model.output_dim}\n"
            f"  Hidden dims: {self.model.hidden_dims}\n"
            f"  Total parameters: {num_params:,}\n"
            f"  λ_reg: {self.lambda_reg}, annealing: {self.annealing}\n"
            f"  Device: {self.device}\n"
            f")"
        )
