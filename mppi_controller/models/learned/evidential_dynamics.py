"""
Evidential Deep Learning 기반 동역학 모델

단일 forward pass로 aleatoric + epistemic 불확실성 분리.
BNN-MPPI / Uncertainty-MPPI와 predict_with_uncertainty() 인터페이스로 자동 연동.
"""

import numpy as np
import torch
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple, Dict


class EvidentialNeuralDynamics(RobotModel):
    """
    Evidential Deep Learning 동역학 모델

    NIG (Normal-Inverse-Gamma) 분포 파라미터를 단일 forward pass로 출력:
        gamma: 예측 평균
        nu: 가상 관측 수 (epistemic 신뢰도)
        alpha, beta: Inverse-Gamma 파라미터

    불확실성 분해:
        epistemic = sqrt(β / (ν(α-1)))  — 모델 불확실성
        aleatoric = sqrt(β / (α-1))     — 데이터 노이즈

    장점:
        - 단일 forward pass (Ensemble M-pass, MCDropout M-pass 대비 O(1))
        - aleatoric/epistemic 분리
        - predict_with_uncertainty() → BNN-MPPI 자동 연동

    사용 예시:
        model = EvidentialNeuralDynamics(
            state_dim=3, control_dim=2,
            model_path="models/learned_models/evidential_best.pth"
        )
        mean = model.forward_dynamics(state, control)
        mean, std = model.predict_with_uncertainty(state, control)
        mean, aleatoric, epistemic = model.predict_with_decomposed_uncertainty(state, control)
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        uncertainty_type: str = "epistemic",
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: 상태 벡터 차원
            control_dim: 제어 벡터 차원
            model_path: 학습된 모델 경로
            uncertainty_type: predict_with_uncertainty의 std 타입
                            ("epistemic" | "aleatoric" | "total")
            device: 'cpu' or 'cuda'
        """
        self._state_dim = state_dim
        self._control_dim = control_dim
        self.uncertainty_type = uncertainty_type
        self.device = torch.device(device)

        self.model = None
        self.norm_stats: Optional[Dict] = None

        if model_path is not None:
            self._load_model(model_path)

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
        EDL 평균 예측: dx/dt = γ(x, u)

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            state_dot: (nx,) 또는 (batch, nx)
        """
        mean, _ = self.predict_with_uncertainty(state, control)
        return mean

    def predict_with_uncertainty(
        self, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        EDL 예측 + 불확실성 (기본: epistemic)

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            mean: (nx,) 또는 (batch, nx)
            std: (nx,) 또는 (batch, nx)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs_t, single = self._prepare_input(state, control)

        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(inputs_t)

            mean = gamma.cpu().numpy()
            alpha_np = alpha.cpu().numpy()
            beta_np = beta.cpu().numpy()
            nu_np = nu.cpu().numpy()

            # Compute requested uncertainty type
            if self.uncertainty_type == "aleatoric":
                var = beta_np / (alpha_np - 1.0)
            elif self.uncertainty_type == "total":
                var = (beta_np * (1.0 + nu_np)) / (nu_np * (alpha_np - 1.0))
            else:  # epistemic (default)
                var = beta_np / (nu_np * (alpha_np - 1.0))

            std = np.sqrt(np.maximum(var, 0.0))

        # Denormalize
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
        예측 + aleatoric/epistemic 분해

        Returns:
            mean, aleatoric_std, epistemic_std
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs_t, single = self._prepare_input(state, control)

        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(inputs_t)

            mean = gamma.cpu().numpy()
            alpha_np = alpha.cpu().numpy()
            beta_np = beta.cpu().numpy()
            nu_np = nu.cpu().numpy()

            aleatoric_var = beta_np / (alpha_np - 1.0)
            epistemic_var = beta_np / (nu_np * (alpha_np - 1.0))

            aleatoric_std = np.sqrt(np.maximum(aleatoric_var, 0.0))
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

    def get_evidence(
        self, state: np.ndarray, control: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        원시 NIG 파라미터 반환

        Returns:
            {"gamma", "nu", "alpha", "beta"}: 각 (nx,) 또는 (batch, nx)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs_t, single = self._prepare_input(state, control)

        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(inputs_t)

        result = {
            "gamma": gamma.cpu().numpy(),
            "nu": nu.cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
            "beta": beta.cpu().numpy(),
        }

        if single:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

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

    def _load_model(self, model_path: str):
        """체크포인트에서 모델 로드"""
        from mppi_controller.learning.evidential_trainer import EvidentialMLPModel

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        input_dim = config["state_dim"] + config["control_dim"]
        output_dim = config["state_dim"]
        hidden_dims = config["hidden_dims"]
        activation = config.get("activation", "relu")

        self.model = EvidentialMLPModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.norm_stats = checkpoint.get("norm_stats")

    def load_model(self, model_path: str):
        """모델 로드 (public)"""
        self._load_model(model_path)

    def get_model_info(self) -> Dict:
        if self.model is None:
            return {"loaded": False}

        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "loaded": True,
            "num_parameters": num_params,
            "uncertainty_type": self.uncertainty_type,
            "device": str(self.device),
            "normalized": self.norm_stats is not None,
        }

    def __repr__(self) -> str:
        if self.model is not None:
            num_params = sum(p.numel() for p in self.model.parameters())
            return (
                f"EvidentialNeuralDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"params={num_params:,}, "
                f"uncertainty={self.uncertainty_type}, "
                f"loaded=True)"
            )
        return (
            f"EvidentialNeuralDynamics("
            f"state_dim={self._state_dim}, "
            f"control_dim={self._control_dim}, "
            f"loaded=False)"
        )
