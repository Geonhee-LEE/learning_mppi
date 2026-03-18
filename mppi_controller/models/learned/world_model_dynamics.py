"""
World Model VAE 기반 동역학 모델

잠재 공간에서 상태 전이: encode → latent_step → decode.
RK4 대신 VAE 잠재 전파로 이산시간 step() 오버라이드.
LatentMPPIController에서 잠재 공간 직접 접근 인터페이스 제공.
"""

import numpy as np
import torch
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Dict


class WorldModelDynamics(RobotModel):
    """
    World Model VAE 동역학 모델

    핵심: step() 오버라이드로 RK4 대신 잠재 공간 전파.
    BatchDynamicsWrapper.rollout()이 model.step() 호출하므로 자동 통합.

    잠재 공간 직접 접근 인터페이스:
    - encode(state) → z
    - decode(z) → state
    - latent_dynamics(z, control) → z_next

    사용 예시:
        model = WorldModelDynamics(state_dim=3, control_dim=2,
                                   model_path="models/learned_models/wm_best.pth")
        next_state = model.step(state, control, dt=0.05)

        # 잠재 공간 직접 접근
        z = model.encode(state)
        z_next = model.latent_dynamics(z, control)
        state_pred = model.decode(z_next)
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        latent_dim: int = 16,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self._state_dim = state_dim
        self._control_dim = control_dim
        self._latent_dim = latent_dim
        self.device = torch.device(device)

        self.vae = None
        self.norm_stats: Optional[Dict] = None

        if model_path is not None:
            self.load_model(model_path)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def control_dim(self) -> int:
        return self._control_dim

    @property
    def model_type(self) -> str:
        return "learned"

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        연속시간 동역학 근사: (next_state - state) / dt

        VAE는 이산시간 모델이므로, forward_dynamics는 dt=1 기준 근사.
        step()을 직접 사용하는 것이 권장됨.
        """
        next_state = self._predict(state, control)
        return next_state - state

    def step(
        self, state: np.ndarray, control: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        이산 시간 전이: encode → latent_step → decode

        RK4 대신 VAE 잠재 공간 전파.
        BatchDynamicsWrapper.rollout()에서 자동 호출.

        Args:
            state: (nx,) 또는 (batch, nx) 현재 상태
            control: (nu,) 또는 (batch, nu) 제어 입력
            dt: 시간 간격 (VAE에서는 무시 — 학습 시 dt 고정)

        Returns:
            next_state: (nx,) 또는 (batch, nx) 다음 상태
        """
        return self._predict(state, control)

    def _predict(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """encode → latent_step → decode"""
        if self.vae is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        single = state.ndim == 1
        if single:
            state = state[np.newaxis, :]
            control = control[np.newaxis, :]

        state_t = torch.FloatTensor(state).to(self.device)
        control_t = torch.FloatTensor(control).to(self.device)

        with torch.no_grad():
            mu, _ = self.vae.encode(state_t)
            z_next = self.vae.latent_step(mu, control_t)
            pred = self.vae.decode(z_next)

        result = pred.cpu().numpy()
        if single:
            result = result.squeeze(0)
        return result

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        상태를 잠재 벡터로 인코딩

        Args:
            state: (nx,) 또는 (batch, nx)

        Returns:
            z: (latent_dim,) 또는 (batch, latent_dim)
        """
        if self.vae is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        single = state.ndim == 1
        if single:
            state = state[np.newaxis, :]

        state_t = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            mu, _ = self.vae.encode(state_t)

        result = mu.cpu().numpy()
        if single:
            result = result.squeeze(0)
        return result

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        잠재 벡터를 상태로 디코딩

        Args:
            z: (latent_dim,) 또는 (batch, latent_dim)

        Returns:
            state: (nx,) 또는 (batch, nx)
        """
        if self.vae is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        single = z.ndim == 1
        if single:
            z = z[np.newaxis, :]

        z_t = torch.FloatTensor(z).to(self.device)

        with torch.no_grad():
            pred = self.vae.decode(z_t)

        result = pred.cpu().numpy()
        if single:
            result = result.squeeze(0)
        return result

    def latent_dynamics(
        self, z: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        잠재 공간 동역학: (z, u) → z_next

        Args:
            z: (latent_dim,) 또는 (batch, latent_dim)
            control: (nu,) 또는 (batch, nu)

        Returns:
            z_next: (latent_dim,) 또는 (batch, latent_dim)
        """
        if self.vae is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        single = z.ndim == 1
        if single:
            z = z[np.newaxis, :]
            control = control[np.newaxis, :]

        z_t = torch.FloatTensor(z).to(self.device)
        control_t = torch.FloatTensor(control).to(self.device)

        with torch.no_grad():
            z_next = self.vae.latent_step(z_t, control_t)

        result = z_next.cpu().numpy()
        if single:
            result = result.squeeze(0)
        return result

    def load_model(self, model_path: str):
        """체크포인트에서 VAE 모델 로드"""
        from mppi_controller.learning.world_model_trainer import WorldModelVAE

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        self._state_dim = config["state_dim"]
        self._control_dim = config["control_dim"]
        self._latent_dim = config["latent_dim"]

        self.vae = WorldModelVAE(
            state_dim=config["state_dim"],
            control_dim=config["control_dim"],
            latent_dim=config["latent_dim"],
            hidden_dims=config["hidden_dims"],
        ).to(self.device)

        self.vae.load_state_dict(checkpoint["model_state_dict"])
        self.vae.eval()
        self.norm_stats = checkpoint.get("norm_stats")

    def set_vae(self, vae, norm_stats=None):
        """직접 VAE 모델 설정 (학습 후 바로 사용)"""
        self.vae = vae
        self.vae.eval()
        self.norm_stats = norm_stats
        self._latent_dim = vae.latent_dim

    def get_control_bounds(self):
        """제어 제약 — 없음 (상위에서 설정)"""
        return None

    def get_model_info(self) -> Dict:
        if self.vae is None:
            return {"loaded": False}

        num_params = sum(p.numel() for p in self.vae.parameters())
        return {
            "loaded": True,
            "num_parameters": num_params,
            "latent_dim": self._latent_dim,
            "device": str(self.device),
            "normalized": self.norm_stats is not None,
        }

    def __repr__(self) -> str:
        if self.vae is not None:
            num_params = sum(p.numel() for p in self.vae.parameters())
            return (
                f"WorldModelDynamics("
                f"state_dim={self._state_dim}, "
                f"control_dim={self._control_dim}, "
                f"latent_dim={self._latent_dim}, "
                f"params={num_params:,}, "
                f"loaded=True)"
            )
        return (
            f"WorldModelDynamics("
            f"state_dim={self._state_dim}, "
            f"control_dim={self._control_dim}, "
            f"latent_dim={self._latent_dim}, "
            f"loaded=False)"
        )
