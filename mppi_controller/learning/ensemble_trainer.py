"""
앙상블 신경망 학습 파이프라인

M개의 독립 MLP를 부트스트랩 학습하여 불확실성 정량화.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel


class EnsembleTrainer:
    """
    앙상블 NN 학습 파이프라인

    M개의 MLP를 독립적으로 부트스트랩 학습:
    1. 각 멤버마다 데이터 리샘플링 (bootstrap)
    2. 다른 초기화로 학습
    3. 앙상블 분산으로 불확실성 추정

    사용 예시:
        trainer = EnsembleTrainer(state_dim=3, control_dim=2, num_models=5)
        trainer.train(train_inputs, train_targets, val_inputs, val_targets, norm_stats)
        trainer.save_model("ensemble.pth")
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        num_models: int = 5,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
        spectral_lambda: float = 0.0,
    ):
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.num_models = num_models
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        input_dim = state_dim + control_dim
        output_dim = state_dim

        # M개의 독립 MLP
        self.models = []
        self.optimizers = []
        self.spectral_regs = []
        for _ in range(num_models):
            model = DynamicsMLPModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout_rate=dropout_rate,
            ).to(self.device)
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            self.models.append(model)
            self.optimizers.append(optimizer)

            # Spectral regularization per model
            if spectral_lambda > 0:
                from mppi_controller.learning.spectral_regularization import SpectralRegularizer
                self.spectral_regs.append(SpectralRegularizer(model, spectral_lambda))
            else:
                self.spectral_regs.append(None)

        self.criterion = nn.MSELoss()
        self.norm_stats: Optional[Dict[str, np.ndarray]] = None

        self.history = {
            "train_loss": [],  # 앙상블 평균 loss
            "val_loss": [],
            "individual_val_losses": [],  # (M,) per epoch
        }

    def train(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        val_inputs: np.ndarray,
        val_targets: np.ndarray,
        norm_stats: Dict[str, np.ndarray],
        epochs: int = 100,
        batch_size: int = 64,
        bootstrap: bool = True,
        early_stopping_patience: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List]:
        """
        앙상블 학습

        Args:
            train_inputs: (N_train, nx+nu)
            train_targets: (N_train, nx)
            val_inputs: (N_val, nx+nu)
            val_targets: (N_val, nx)
            norm_stats: 정규화 통계
            epochs: 학습 에폭
            batch_size: 배치 크기
            bootstrap: 부트스트랩 리샘플링 여부
            early_stopping_patience: Early stopping patience
            verbose: 출력

        Returns:
            history: 학습 히스토리
        """
        self.norm_stats = norm_stats
        N_train = len(train_inputs)

        val_x = torch.FloatTensor(val_inputs).to(self.device)
        val_y = torch.FloatTensor(val_targets).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_train_losses = []
            epoch_val_losses = []

            for m_idx, (model, optimizer) in enumerate(
                zip(self.models, self.optimizers)
            ):
                # Bootstrap: 리샘플링
                if bootstrap:
                    indices = np.random.choice(N_train, N_train, replace=True)
                    batch_inputs = train_inputs[indices]
                    batch_targets = train_targets[indices]
                else:
                    batch_inputs = train_inputs
                    batch_targets = train_targets

                train_x = torch.FloatTensor(batch_inputs).to(self.device)
                train_y = torch.FloatTensor(batch_targets).to(self.device)

                dataset = TensorDataset(train_x, train_y)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                spectral_reg = self.spectral_regs[m_idx] if m_idx < len(self.spectral_regs) else None

                # Train
                model.train()
                train_loss = 0.0
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = self.criterion(pred, by)
                    if spectral_reg is not None:
                        loss = loss + spectral_reg.compute_penalty()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(loader)
                epoch_train_losses.append(train_loss)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_x)
                    val_loss = self.criterion(val_pred, val_y).item()
                epoch_val_losses.append(val_loss)

            avg_train = np.mean(epoch_train_losses)
            avg_val = np.mean(epoch_val_losses)

            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)
            self.history["individual_val_losses"].append(epoch_val_losses)

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                    f"Val spread: {np.std(epoch_val_losses):.6f}"
                )

        if verbose:
            print(f"\nEnsemble training completed. Best val loss: {best_val_loss:.6f}")

        return self.history

    def predict(
        self,
        state: np.ndarray,
        control: np.ndarray,
        denormalize: bool = True,
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        앙상블 예측

        Returns:
            mean: 앙상블 평균
            std: 앙상블 표준편차 (return_uncertainty=True일 때)
        """
        # Normalize
        if self.norm_stats is not None:
            state_n = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_n = (control - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
        else:
            state_n = state
            control_n = control

        if state.ndim == 1:
            inputs = np.concatenate([state_n, control_n])
            inputs_t = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)
        else:
            inputs = np.concatenate([state_n, control_n], axis=1)
            inputs_t = torch.FloatTensor(inputs).to(self.device)

        preds = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                preds.append(model(inputs_t))

        preds_t = torch.stack(preds, dim=0)
        mean = preds_t.mean(dim=0).cpu().numpy()
        std = preds_t.std(dim=0).cpu().numpy() if self.num_models > 1 else np.zeros_like(mean)

        if denormalize and self.norm_stats is not None:
            mean = mean * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]
            if return_uncertainty:
                std = std * self.norm_stats["state_dot_std"]

        if state.ndim == 1:
            mean = mean.squeeze(0)
            if return_uncertainty:
                std = std.squeeze(0)

        if return_uncertainty:
            return mean, std
        return mean, None

    def save_model(self, filename: str):
        """Save ensemble"""
        filepath = os.path.join(self.save_dir, filename)

        models_state = [m.state_dict() for m in self.models]

        torch.save({
            "models_state_dict": models_state,
            "norm_stats": self.norm_stats,
            "history": self.history,
            "config": {
                "state_dim": self.state_dim,
                "control_dim": self.control_dim,
                "num_models": self.num_models,
                "hidden_dims": self.hidden_dims,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
            },
        }, filepath)

    def load_model(self, filename: str):
        """Load ensemble"""
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        input_dim = config["state_dim"] + config["control_dim"]
        output_dim = config["state_dim"]

        models_state = checkpoint["models_state_dict"]
        self.models = []
        self.optimizers = []

        for sd in models_state:
            model = DynamicsMLPModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=config["hidden_dims"],
                activation=config.get("activation", "relu"),
                dropout_rate=config.get("dropout_rate", 0.0),
            ).to(self.device)
            model.load_state_dict(sd)
            model.eval()
            self.models.append(model)
            self.optimizers.append(
                optim.Adam(model.parameters(), lr=self.learning_rate)
            )

        self.num_models = len(self.models)
        self.norm_stats = checkpoint.get("norm_stats")
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
