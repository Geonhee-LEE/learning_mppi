"""
Gaussian Process 기반 동역학 모델

GPyTorch 학습 모델 통합 및 불확실성 정량화.
"""

import numpy as np
import torch
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple, Dict

try:
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


class GaussianProcessDynamics(RobotModel):
    """
    Gaussian Process 기반 동역학 모델

    확률적 동역학 학습:
        dx/dt ~ GP(x, u)

    장점:
        - 불확실성 정량화 (평균 + 분산)
        - 데이터 효율성 (적은 데이터로 학습 가능)
        - 커널 선택으로 귀납적 편향 가능

    단점:
        - 계산 비용 O(N³) (Exact) 또는 O(NM²) (Sparse)
        - 고차원에서 확장성 문제
        - 실시간 추론 어려움 (근사 필요)

    활용 사례:
        - 안전 보장 제어 (불확실성 고려)
        - Risk-aware MPPI 연동
        - Active Learning (탐색-활용)
        - 데이터 효율적 학습

    사용 예시:
        # 학습된 GP 모델 로드
        gp_model = GaussianProcessDynamics(
            state_dim=3,
            control_dim=2,
            model_path="models/learned_models/gp_model.pth"
        )

        # 불확실성과 함께 예측
        mean, std = gp_model.predict_with_uncertainty(state, control)

    Args:
        state_dim: 상태 벡터 차원
        control_dim: 제어 벡터 차원
        model_path: 학습된 모델 경로 (GPyTorch checkpoint)
        device: 'cpu' or 'cuda'
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        if not HAS_GPYTORCH:
            raise ImportError(
                "gpytorch is required for GaussianProcessDynamics. "
                "Install with: pip install gpytorch>=1.11.0"
            )

        self._state_dim = state_dim
        self._control_dim = control_dim
        self.device = torch.device(device)
        self.model_path = model_path

        # GP models (one per output dimension)
        self.gp_models = None
        self.likelihoods = None
        self.norm_stats = None

        # Load model if path provided
        if model_path is not None:
            self._load_model(model_path)
        else:
            print(
                "[GaussianProcessDynamics] No model loaded. "
                "Call load_model() or use GaussianProcessTrainer to train."
            )

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def control_dim(self) -> int:
        return self._control_dim

    @property
    def model_type(self) -> str:
        return "learned"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        GP 예측: dx/dt = E[GP(x, u)]

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            state_dot: (nx,) 또는 (batch, nx) - GP 평균
        """
        if self.gp_models is None:
            raise RuntimeError("Models not loaded. Call load_model() first.")

        mean, _ = self.predict_with_uncertainty(state, control)
        return mean

    def predict_with_uncertainty(
        self, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        불확실성과 함께 예측

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            mean: (nx,) 또는 (batch, nx) - GP 평균
            std: (nx,) 또는 (batch, nx) - GP 표준편차
        """
        if self.gp_models is None:
            raise RuntimeError("Models not loaded. Call load_model() first.")

        # Normalize inputs
        if self.norm_stats is not None:
            state_norm = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
            control_norm = (control - self.norm_stats["control_mean"]) / self.norm_stats["control_std"]
        else:
            state_norm = state
            control_norm = control

        # Concatenate [state, control]
        if state.ndim == 1:
            inputs = np.concatenate([state_norm, control_norm])
            inputs_tensor = torch.FloatTensor(inputs).unsqueeze(0).to(self.device)
        else:
            inputs = np.concatenate([state_norm, control_norm], axis=1)
            inputs_tensor = torch.FloatTensor(inputs).to(self.device)

        # Predict for each output dimension
        means = []
        stds = []

        try:
            import gpytorch

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for output_dim in range(self._state_dim):
                    model = self.gp_models[output_dim]
                    likelihood = self.likelihoods[output_dim]

                    output = model(inputs_tensor)
                    pred = likelihood(output)

                    means.append(pred.mean.cpu().numpy())
                    stds.append(pred.stddev.cpu().numpy())

        except ImportError:
            raise ImportError(
                "GPyTorch not installed. Install with: pip install gpytorch"
            )

        # Stack predictions
        mean = np.stack(means, axis=-1)
        std = np.stack(stds, axis=-1)

        # Denormalize outputs
        if self.norm_stats is not None:
            mean = mean * self.norm_stats["state_dot_std"] + self.norm_stats["state_dot_mean"]
            std = std * self.norm_stats["state_dot_std"]

        # Squeeze if single sample
        if state.ndim == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std

    def _load_model(self, model_path: str):
        """
        Load trained GP models

        Args:
            model_path: Path to GPyTorch checkpoint
        """
        try:
            # Import here to avoid dependency if not using GP models
            from mppi_controller.learning.gaussian_process_trainer import (
                ExactGPModel,
                SparseGPModel,
            )
            import gpytorch

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Extract config
            config = checkpoint["config"]
            state_dim = config["state_dim"]
            control_dim = config["control_dim"]
            kernel_type = config["kernel_type"]
            use_sparse = config["use_sparse"]
            use_ard = config.get("use_ard", True)

            # Load normalization stats
            self.norm_stats = checkpoint.get("norm_stats")

            # Reconstruct models
            self.gp_models = []
            self.likelihoods = []

            models_state = checkpoint["models_state_dict"]
            likelihoods_state = checkpoint["likelihoods_state_dict"]

            # Create dummy data for model initialization
            # (GP models need train data in constructor for Exact GP)
            input_dim = state_dim + control_dim
            dummy_train_x = torch.randn(10, input_dim).to(self.device)
            dummy_train_y = torch.randn(10).to(self.device)

            for output_dim in range(state_dim):
                # Create likelihood
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

                # Create model
                if use_sparse:
                    # Sparse GP
                    num_inducing = config.get("num_inducing_points", 100)
                    inducing_points = torch.randn(num_inducing, input_dim).to(self.device)

                    model = SparseGPModel(
                        inducing_points=inducing_points,
                        kernel_type=kernel_type,
                        use_ard=use_ard,
                    ).to(self.device)
                else:
                    # Exact GP
                    model = ExactGPModel(
                        train_x=dummy_train_x,
                        train_y=dummy_train_y,
                        likelihood=likelihood,
                        kernel_type=kernel_type,
                        use_ard=use_ard,
                    ).to(self.device)

                # Load state dict
                model.load_state_dict(models_state[output_dim])
                likelihood.load_state_dict(likelihoods_state[output_dim])

                model.eval()
                likelihood.eval()

                self.gp_models.append(model)
                self.likelihoods.append(likelihood)

            print(f"[GaussianProcessDynamics] Models loaded from {model_path}")
            print(f"  State dim: {state_dim}, Control dim: {control_dim}")
            print(f"  Kernel: {kernel_type}, Sparse: {use_sparse}, ARD: {use_ard}")
            print(f"  Num models: {len(self.gp_models)}")

        except ImportError as e:
            raise ImportError(
                f"GPyTorch not installed or import error: {e}\n"
                "Install with: pip install gpytorch"
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def _set_eval_mode(self):
        """Set all GP models and likelihoods to eval mode (called once after load)"""
        if self.gp_models is not None:
            for model in self.gp_models:
                model.eval()
        if self.likelihoods is not None:
            for likelihood in self.likelihoods:
                likelihood.eval()

    def load_model(self, model_path: str):
        """Load model (public method)"""
        self.model_path = model_path
        self._load_model(model_path)

    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.gp_models is None:
            return {"loaded": False}

        total_params = 0
        for model in self.gp_models:
            total_params += sum(p.numel() for p in model.parameters())

        return {
            "loaded": True,
            "state_dim": self._state_dim,
            "control_dim": self._control_dim,
            "num_models": len(self.gp_models),
            "num_parameters": total_params,
            "device": str(self.device),
            "normalized": self.norm_stats is not None,
        }

    def get_lengthscales(self) -> Dict[int, np.ndarray]:
        """
        Get learned lengthscales for each output dimension

        Returns:
            dict: {output_dim: lengthscales (input_dim,)}
        """
        if self.gp_models is None:
            raise RuntimeError("Models not loaded")

        lengthscales = {}
        for output_dim, model in enumerate(self.gp_models):
            # Extract lengthscales from RBF/Matern kernel
            kernel = model.covar_module.base_kernel
            ls = kernel.lengthscale.detach().cpu().numpy().squeeze()
            lengthscales[output_dim] = ls

        return lengthscales

    def get_noise_variance(self) -> Dict[int, float]:
        """
        Get learned noise variance for each output dimension

        Returns:
            dict: {output_dim: noise_variance}
        """
        if self.likelihoods is None:
            raise RuntimeError("Models not loaded")

        noise_variances = {}
        for output_dim, likelihood in enumerate(self.likelihoods):
            noise_var = likelihood.noise.item()
            noise_variances[output_dim] = noise_var

        return noise_variances

    def __repr__(self) -> str:
        if self.gp_models is not None:
            total_params = sum(
                sum(p.numel() for p in model.parameters())
                for model in self.gp_models
            )
            return (
                f"GaussianProcessDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"num_models={len(self.gp_models)}, "
                f"params={total_params:,}, "
                f"loaded=True)"
            )
        else:
            return (
                f"GaussianProcessDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"loaded=False)"
            )
