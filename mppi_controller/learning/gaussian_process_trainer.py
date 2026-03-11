#!/usr/bin/env python3
"""
Gaussian Process 동역학 모델 학습 파이프라인

GPyTorch 기반 GP 학습 및 불확실성 정량화.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


# GP model classes are defined only when gpytorch is available
if HAS_GPYTORCH:

    class ExactGPModel(gpytorch.models.ExactGP):
        """
        Exact GP Model for single output dimension

        Uses RBF kernel with automatic relevance determination (ARD).
        """

        def __init__(
            self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            likelihood: gpytorch.likelihoods.GaussianLikelihood,
            kernel_type: str = "rbf",
            use_ard: bool = True,
        ):
            super().__init__(train_x, train_y, likelihood)

            input_dim = train_x.shape[-1]

            # Mean
            self.mean_module = gpytorch.means.ConstantMean()

            # Kernel
            if kernel_type == "rbf":
                if use_ard:
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
                    )
                else:
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel()
                    )
            elif kernel_type == "matern":
                if use_ard:
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
                    )
                else:
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(nu=2.5)
                    )
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class SparseGPModel(gpytorch.models.ApproximateGP):
        """
        Sparse GP Model using variational inference

        More efficient for large datasets.
        """

        def __init__(
            self,
            inducing_points: torch.Tensor,
            kernel_type: str = "rbf",
            use_ard: bool = True,
        ):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )

            super().__init__(variational_strategy)

            input_dim = inducing_points.shape[-1]

            # Mean
            self.mean_module = gpytorch.means.ConstantMean()

            # Kernel
            if kernel_type == "rbf":
                if use_ard:
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
                    )
                else:
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel()
                    )
            elif kernel_type == "matern":
                if use_ard:
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
                    )
                else:
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(nu=2.5)
                    )
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessTrainer:
    """
    Gaussian Process 학습 파이프라인

    Multi-output GP: 각 출력 차원마다 독립적인 GP 학습

    사용 예시:
        trainer = GaussianProcessTrainer(state_dim=3, control_dim=2)
        trainer.train(train_data, val_data, num_iterations=100)
        trainer.save_model("gp_model.pth")
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        kernel_type: str = "rbf",
        use_sparse: bool = False,
        num_inducing_points: int = 100,
        use_ard: bool = True,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    ):
        """
        Args:
            state_dim: 상태 벡터 차원
            control_dim: 제어 벡터 차원
            kernel_type: 커널 종류 ('rbf', 'matern')
            use_sparse: Sparse GP 사용 여부
            num_inducing_points: Sparse GP 유도점 수
            use_ard: Automatic Relevance Determination 사용
            device: 'cpu' or 'cuda'
            save_dir: Model save directory
        """
        if not HAS_GPYTORCH:
            raise ImportError(
                "gpytorch is required for GP training. "
                "Install with: pip install gpytorch>=1.11.0"
            )

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.kernel_type = kernel_type
        self.use_sparse = use_sparse
        self.num_inducing_points = num_inducing_points
        self.use_ard = use_ard
        self.device = torch.device(device)

        # Multi-output GP: 각 출력 차원마다 독립 GP
        self.gp_models: List = []
        self.likelihoods: List = []

        # Normalization statistics
        self.norm_stats: Optional[Dict[str, np.ndarray]] = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

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
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train multi-output GP

        Args:
            train_inputs: (N_train, nx + nu)
            train_targets: (N_train, nx)
            val_inputs: (N_val, nx + nu)
            val_targets: (N_val, nx)
            norm_stats: Normalization statistics
            num_iterations: Optimization iterations
            learning_rate: Learning rate
            verbose: Print progress

        Returns:
            history: Training history
        """
        self.norm_stats = norm_stats

        # Clear previous models to prevent accumulation on repeated train() calls
        self.gp_models = []
        self.likelihoods = []

        # Convert to tensors
        train_x = torch.FloatTensor(train_inputs).to(self.device)
        train_y = torch.FloatTensor(train_targets).to(self.device)
        val_x = torch.FloatTensor(val_inputs).to(self.device)
        val_y = torch.FloatTensor(val_targets).to(self.device)

        # Train GP for each output dimension
        for output_dim in range(self.state_dim):
            if verbose:
                print(f"\nTraining GP for output dimension {output_dim+1}/{self.state_dim}")

            train_y_dim = train_y[:, output_dim]

            # Create likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

            # Create GP model
            if self.use_sparse:
                # Select inducing points (k-means)
                inducing_indices = self._select_inducing_points(
                    train_x, self.num_inducing_points
                )
                inducing_points = train_x[inducing_indices].clone()

                model = SparseGPModel(
                    inducing_points=inducing_points,
                    kernel_type=self.kernel_type,
                    use_ard=self.use_ard,
                ).to(self.device)

                # MLL for sparse GP
                mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y_dim.size(0))
            else:
                model = ExactGPModel(
                    train_x=train_x,
                    train_y=train_y_dim,
                    likelihood=likelihood,
                    kernel_type=self.kernel_type,
                    use_ard=self.use_ard,
                ).to(self.device)

                # MLL for exact GP
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Optimizer
            optimizer = torch.optim.Adam(
                [
                    {"params": model.parameters()},
                    {"params": likelihood.parameters()},
                ],
                lr=learning_rate,
            )

            # Training mode
            model.train()
            likelihood.train()

            # Training loop
            for i in range(num_iterations):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y_dim)
                loss.backward()
                optimizer.step()

                if verbose and (i + 1) % 20 == 0:
                    # Validation
                    model.eval()
                    likelihood.eval()
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        val_output = model(val_x)
                        val_mean = val_output.mean
                        val_loss = torch.mean((val_mean - val_y[:, output_dim]) ** 2)

                    print(
                        f"  Iter {i+1}/{num_iterations} | "
                        f"Train Loss: {loss.item():.4f} | "
                        f"Val MSE: {val_loss.item():.6f}"
                    )

                    model.train()
                    likelihood.train()

            # Store model
            self.gp_models.append(model)
            self.likelihoods.append(likelihood)

        if verbose:
            print(f"\nGP training completed for all {self.state_dim} dimensions")

        # Set eval mode for inference after training
        for model in self.gp_models:
            model.eval()
        for likelihood in self.likelihoods:
            likelihood.eval()

        # Compute overall validation loss
        self._compute_validation_loss(val_x, val_y)

        return self.history

    def _select_inducing_points(
        self, train_x: torch.Tensor, num_inducing: int
    ) -> torch.Tensor:
        """Select inducing points using simple uniform sampling"""
        if train_x.size(0) <= num_inducing:
            return torch.arange(train_x.size(0))
        else:
            indices = torch.linspace(0, train_x.size(0) - 1, num_inducing, dtype=torch.long)
            return indices

    def _compute_validation_loss(self, val_x: torch.Tensor, val_y: torch.Tensor):
        """Compute validation loss across all dimensions"""
        total_val_loss = 0.0

        for output_dim in range(self.state_dim):
            model = self.gp_models[output_dim]
            model.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                val_output = model(val_x)
                val_mean = val_output.mean
                val_loss = torch.mean((val_mean - val_y[:, output_dim]) ** 2)
                total_val_loss += val_loss.item()

        self.history["val_loss"].append(total_val_loss / self.state_dim)

    def predict(
        self,
        state: np.ndarray,
        control: np.ndarray,
        denormalize: bool = True,
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict state_dot with uncertainty

        Args:
            state: (nx,) or (batch, nx)
            control: (nu,) or (batch, nu)
            denormalize: Denormalize output
            return_uncertainty: Return uncertainty (std)

        Returns:
            mean: (nx,) or (batch, nx)
            std: (nx,) or (batch, nx) if return_uncertainty=True
        """
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

        # Predict for each dimension
        means = []
        stds = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for output_dim in range(self.state_dim):
                model = self.gp_models[output_dim]
                likelihood = self.likelihoods[output_dim]

                output = model(inputs_tensor)
                pred = likelihood(output)

                means.append(pred.mean.cpu().numpy())
                if return_uncertainty:
                    stds.append(pred.stddev.cpu().numpy())

        # Stack predictions
        mean = np.stack(means, axis=-1)
        if return_uncertainty:
            std = np.stack(stds, axis=-1)
        else:
            std = None

        # Denormalize outputs
        if denormalize and self.norm_stats is not None:
            mean = mean * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]
            if return_uncertainty:
                std = std * self.norm_stats["state_dot_std"]

        # Squeeze if single sample
        if state.ndim == 1:
            mean = mean.squeeze(0)
            if return_uncertainty:
                std = std.squeeze(0)

        return mean, std

    def save_model(self, filename: str):
        """Save GP models and normalization stats"""
        filepath = os.path.join(self.save_dir, filename)

        # Save all models
        models_state = []
        likelihoods_state = []

        for model, likelihood in zip(self.gp_models, self.likelihoods):
            models_state.append(model.state_dict())
            likelihoods_state.append(likelihood.state_dict())

        torch.save({
            "models_state_dict": models_state,
            "likelihoods_state_dict": likelihoods_state,
            "norm_stats": self.norm_stats,
            "history": self.history,
            "config": {
                "state_dim": self.state_dim,
                "control_dim": self.control_dim,
                "kernel_type": self.kernel_type,
                "use_sparse": self.use_sparse,
                "num_inducing_points": self.num_inducing_points,
                "use_ard": self.use_ard,
            },
        }, filepath)

        print(f"[GaussianProcessTrainer] Models saved to {filepath}")

    def load_model(self, filename: str):
        """Load GP models and normalization stats"""
        filepath = os.path.join(self.save_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Load config
        config = checkpoint["config"]
        self.state_dim = config["state_dim"]
        self.control_dim = config["control_dim"]
        self.kernel_type = config["kernel_type"]
        self.use_sparse = config["use_sparse"]
        self.num_inducing_points = config.get("num_inducing_points", 100)
        self.use_ard = config.get("use_ard", True)

        self.norm_stats = checkpoint["norm_stats"]
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})

        # Reconstruct GP models from saved state dicts
        models_state = checkpoint["models_state_dict"]
        likelihoods_state = checkpoint["likelihoods_state_dict"]

        input_dim = self.state_dim + self.control_dim
        dummy_train_x = torch.randn(10, input_dim).to(self.device)
        dummy_train_y = torch.randn(10).to(self.device)

        self.gp_models = []
        self.likelihoods = []

        for output_dim in range(self.state_dim):
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

            if self.use_sparse:
                num_inducing = self.num_inducing_points
                inducing_points = torch.randn(num_inducing, input_dim).to(self.device)
                model = SparseGPModel(
                    inducing_points=inducing_points,
                    kernel_type=self.kernel_type,
                    use_ard=self.use_ard,
                ).to(self.device)
            else:
                model = ExactGPModel(
                    train_x=dummy_train_x,
                    train_y=dummy_train_y,
                    likelihood=likelihood,
                    kernel_type=self.kernel_type,
                    use_ard=self.use_ard,
                ).to(self.device)

            model.load_state_dict(models_state[output_dim])
            likelihood.load_state_dict(likelihoods_state[output_dim])
            model.eval()
            likelihood.eval()

            self.gp_models.append(model)
            self.likelihoods.append(likelihood)

        print(f"[GaussianProcessTrainer] Models loaded from {filepath}")
        print(f"  Restored {len(self.gp_models)} GP models")

    def get_model_summary(self) -> str:
        """Get model summary"""
        if len(self.gp_models) == 0:
            return "No models trained yet"

        total_params = 0
        for model in self.gp_models:
            total_params += sum(p.numel() for p in model.parameters())

        return (
            f"GaussianProcessTrainer(\n"
            f"  State dim: {self.state_dim}\n"
            f"  Control dim: {self.control_dim}\n"
            f"  Kernel: {self.kernel_type}\n"
            f"  Sparse: {self.use_sparse}\n"
            f"  ARD: {self.use_ard}\n"
            f"  Num models: {len(self.gp_models)}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Device: {self.device}\n"
            f")"
        )
