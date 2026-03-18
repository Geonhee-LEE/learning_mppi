"""
Latent-Space MPPI 컨트롤러

VAE 잠재 공간에서 K×N 롤아웃 수행 후 일괄 디코딩하여
기존 비용 함수로 비용 평가. 물리 공간 비용 함수 재사용 가능.

핵심 알고리즘:
    x_0 → [Encoder] → z_0
    z_0 → [Latent Dynamics ×N] → z_0..z_N
    z_0..z_N → [Decoder] → x_0..x_N
    costs = cost_function(x_0..x_N, controls, reference)

이점:
    - 저차원 잠재 공간에서의 빠른 롤아웃
    - 기존 CompositeMPPICost 그대로 재사용
    - 잠재 공간 시각화 가능 (PCA 등)

References:
    Hafner et al. "Dream to Control" (ICLR 2020)
    Watter et al. "Embed to Control" (NeurIPS 2015)
"""

import numpy as np
from typing import Dict, Tuple, Optional

from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import LatentMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.models.base_model import RobotModel


class LatentMPPIController(MPPIController):
    """
    Latent-Space MPPI: VAE 잠재 공간에서의 MPPI 계획

    world_model (encode/decode/latent_dynamics 인터페이스)이 설정되면
    잠재 공간에서 K×N 롤아웃 → 일괄 디코딩 → 기존 비용 함수 평가.
    world_model이 없으면 표준 MPPI 폴백 (BNN-MPPI 패턴).

    Args:
        model: RobotModel 인스턴스 (제어 경계/차원 정보용)
        params: LatentMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
        world_model: encode/decode/latent_dynamics 메서드를 가진 객체
            None이면 model에서 자동 감지 시도
    """

    def __init__(
        self,
        model: RobotModel,
        params: LatentMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        world_model=None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)

        self.latent_params = params

        # world_model 자동 감지
        if world_model is not None:
            self.world_model = world_model
        elif self._has_world_model_interface(model):
            self.world_model = model
        else:
            self.world_model = None

        # 통계 히스토리
        self._latent_history = []

    @staticmethod
    def _has_world_model_interface(model) -> bool:
        """encode/decode/latent_dynamics 인터페이스 확인"""
        return (
            hasattr(model, 'encode')
            and hasattr(model, 'decode')
            and hasattr(model, 'latent_dynamics')
        )

    def set_world_model(self, world_model):
        """World Model 설정/변경"""
        self.world_model = world_model

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Latent-Space MPPI 제어 계산

        1. z_0 = encode(state)
        2. z_0 → tile(K)
        3. 잠재 롤아웃: z_{t+1} = latent_dynamics(z_t, u_t)
        4. 배치 디코딩: z → x
        5. costs = cost_function(x, controls, reference)
        6. MPPI 가중치 계산 + 제어 업데이트

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + latent_stats
        """
        # world_model이 없거나 use_latent_rollout이 False면 표준 MPPI 폴백
        if self.world_model is None or not self.latent_params.use_latent_rollout:
            return super().compute_control(state, reference_trajectory)

        K = self.params.K
        N = self.params.N
        nx = state.shape[0]

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 잠재 공간 인코딩
        z_0 = self.world_model.encode(state)  # (latent_dim,)
        latent_dim = z_0.shape[0]

        # 4. 잠재 롤아웃
        z_batch = np.tile(z_0, (K, 1))  # (K, latent_dim)

        # 잠재 궤적 수집: (K, N+1, latent_dim)
        latent_trajectories = np.zeros((K, N + 1, latent_dim))
        latent_trajectories[:, 0, :] = z_batch

        decode_interval = self.latent_params.decode_interval

        for t in range(N):
            z_batch = self.world_model.latent_dynamics(
                z_batch, sampled_controls[:, t, :]
            )  # (K, latent_dim)

            # 주기적 re-encoding: decode → re-encode로 drift 보정
            if decode_interval > 1 and (t + 1) % decode_interval == 0 and t < N - 1:
                x_corrected = self.world_model.decode(z_batch)  # (K, nx)
                z_batch = self.world_model.encode(x_corrected)  # (K, latent_dim)

            latent_trajectories[:, t + 1, :] = z_batch

        # 5. 배치 디코딩: (K*(N+1), latent_dim) → (K*(N+1), nx) → (K, N+1, nx)
        z_flat = latent_trajectories.reshape(-1, latent_dim)  # (K*(N+1), latent_dim)
        x_flat = self.world_model.decode(z_flat)  # (K*(N+1), nx)
        sample_trajectories = x_flat.reshape(K, N + 1, nx)

        # 6. 비용 계산 (K,)
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 7. MPPI 가중치 — 적응적 온도 스케일링
        # VAE 디코딩 노이즈로 비용 분산이 크므로, IQR(사분위 범위) 기반 λ 스케일링.
        # IQR은 극단값에 강건하며, λ_eff = λ * IQR/3으로 ESS ≈ K*0.2~0.4 유지.
        q75, q25 = np.percentile(costs, [75, 25])
        iqr = q75 - q25
        if iqr > 1e-10:
            lambda_eff = self.params.lambda_ * iqr / 3.0
        else:
            lambda_eff = self.params.lambda_
        weights = self._compute_weights(costs, lambda_eff)

        # 8. 가중 평균으로 제어 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 9. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        optimal_control = self.U[0, :]

        # 10. Info
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        latent_stats = {
            "latent_dim": latent_dim,
            "mean_latent_norm": float(np.mean(np.linalg.norm(
                latent_trajectories.reshape(-1, latent_dim), axis=-1
            ))),
            "max_latent_norm": float(np.max(np.linalg.norm(
                latent_trajectories.reshape(-1, latent_dim), axis=-1
            ))),
        }
        self._latent_history.append(latent_stats)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": lambda_eff,
            "ess": ess,
            "num_samples": K,
            "latent_trajectories": latent_trajectories,
            "latent_stats": latent_stats,
        }
        self.last_info = info

        return optimal_control, info

    def get_latent_statistics(self) -> Dict:
        """누적된 잠재 공간 통계 반환"""
        if not self._latent_history:
            return {"num_steps": 0}

        norms = [h["mean_latent_norm"] for h in self._latent_history]

        return {
            "num_steps": len(self._latent_history),
            "overall_mean_latent_norm": float(np.mean(norms)),
            "overall_max_latent_norm": float(
                max(h["max_latent_norm"] for h in self._latent_history)
            ),
            "history": self._latent_history,
        }

    def reset(self):
        """상태 초기화"""
        super().reset()
        self._latent_history = []

    def __repr__(self) -> str:
        wm_status = "active" if self.world_model is not None else "none"
        return (
            f"LatentMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"latent_dim={self.latent_params.latent_dim}, "
            f"world_model={wm_status})"
        )
