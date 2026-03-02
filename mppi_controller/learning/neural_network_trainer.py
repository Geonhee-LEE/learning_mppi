#!/usr/bin/env python3
"""
Neural Network 동역학 모델 학습 파이프라인

PyTorch 기반 MLP 학습 및 MPPI 통합.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import matplotlib.pyplot as plt


class DynamicsMLPModel(nn.Module):
    """
    Multi-Layer Perceptron for dynamics learning

    Architecture:
        Input: [state, control] (nx + nu)
        Hidden layers: MLP with ReLU
        Output: state_dot (nx)

    Args:
        input_dim: nx + nu
        output_dim: nx
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'tanh', 'elu')
        dropout_rate: Dropout rate (0.0 = no dropout)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, input_dim) - [state, control]

        Returns:
            state_dot: (batch, output_dim)
        """
        return self.network(x)


class NeuralNetworkTrainer:
    """
    Neural Network 학습 파이프라인

    사용 예시:
        trainer = NeuralNetworkTrainer(state_dim=3, control_dim=2)
        trainer.train(train_data, val_data, epochs=100)
        trainer.save_model("dynamics_model.pth")
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
        spectral_lambda: float = 0.0,
    ):
        """
        Args:
            state_dim: 상태 벡터 차원
            control_dim: 제어 벡터 차원
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            weight_decay: L2 regularization
            device: 'cpu' or 'cuda'
            save_dir: Model save directory
            spectral_lambda: Spectral regularization strength (0 = disabled)
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.device = torch.device(device)

        # Build model
        input_dim = state_dim + control_dim
        output_dim = state_dim

        self.model = DynamicsMLPModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Spectral regularization
        self.spectral_reg = None
        if spectral_lambda > 0:
            from mppi_controller.learning.spectral_regularization import SpectralRegularizer
            self.spectral_reg = SpectralRegularizer(self.model, spectral_lambda)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Normalization statistics
        self.norm_stats: Optional[Dict[str, np.ndarray]] = None

        # Save directory
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
        Train the neural network

        Args:
            train_inputs: (N_train, nx + nu)
            train_targets: (N_train, nx)
            val_inputs: (N_val, nx + nu)
            val_targets: (N_val, nx)
            norm_stats: Normalization statistics from DynamicsDataset
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Early stopping patience
            verbose: Print progress

        Returns:
            history: Training history
        """
        self.norm_stats = norm_stats

        # Convert to tensors
        train_inputs_tensor = torch.FloatTensor(train_inputs).to(self.device)
        train_targets_tensor = torch.FloatTensor(train_targets).to(self.device)
        val_inputs_tensor = torch.FloatTensor(val_inputs).to(self.device)
        val_targets_tensor = torch.FloatTensor(val_targets).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(train_inputs_tensor, train_targets_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_inputs, batch_targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                if self.spectral_reg is not None:
                    loss = loss + self.spectral_reg.compute_penalty()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_inputs_tensor)
                val_loss = self.criterion(val_outputs, val_targets_tensor).item()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]['lr'])

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model("best_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

        if verbose:
            print(f"\nTraining completed. Best val loss: {best_val_loss:.6f}")

        return self.history

    def predict(
        self,
        state: np.ndarray,
        control: np.ndarray,
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Predict state_dot

        Args:
            state: (nx,) or (batch, nx)
            control: (nu,) or (batch, nu)
            denormalize: Denormalize output

        Returns:
            state_dot: (nx,) or (batch, nx)
        """
        self.model.eval()

        # Normalize inputs
        if self.norm_stats is not None:
            state_norm = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_norm = (control - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
        else:
            state_norm = state
            control_norm = control

        # Concatenate
        if state.ndim == 1:
            inputs = np.concatenate([state_norm, control_norm])
            inputs_tensor = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)
        else:
            inputs = np.concatenate([state_norm, control_norm], axis=1)
            inputs_tensor = torch.FloatTensor(inputs).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs_tensor = self.model(inputs_tensor)
            outputs = outputs_tensor.cpu().numpy()

        # Denormalize outputs
        if denormalize and self.norm_stats is not None:
            outputs = outputs * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]

        # Squeeze if single sample
        if state.ndim == 1:
            outputs = outputs.squeeze(0)

        return outputs

    def save_model(self, filename: str):
        """Save model and normalization stats"""
        filepath = os.path.join(self.save_dir, filename)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "norm_stats": self.norm_stats,
            "history": self.history,
            "config": {
                "state_dim": self.state_dim,
                "control_dim": self.control_dim,
                "hidden_dims": self.hidden_dims,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
            },
        }, filepath)

        print(f"[NeuralNetworkTrainer] Model saved to {filepath}")

    def load_model(self, filename: str):
        """Load model and normalization stats"""
        filepath = os.path.join(self.save_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.norm_stats = checkpoint["norm_stats"]
        self.history = checkpoint.get("history", {})

        self.model.eval()

        print(f"[NeuralNetworkTrainer] Model loaded from {filepath}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(self.history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(self.history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Learning rate
        axes[1].plot(self.history["learning_rate"], linewidth=2, color='orange')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()

    def get_model_summary(self) -> str:
        """Get model summary"""
        num_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"DynamicsMLPModel(\n"
            f"  Input dim: {self.model.input_dim}\n"
            f"  Output dim: {self.model.output_dim}\n"
            f"  Hidden dims: {self.model.hidden_dims}\n"
            f"  Total parameters: {num_params:,}\n"
            f"  Device: {self.device}\n"
            f")"
        )
