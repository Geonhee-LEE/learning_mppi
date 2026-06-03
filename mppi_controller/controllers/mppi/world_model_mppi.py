"""
World Model MPPI 컨트롤러

학습된 세계 모델(RSSM/LinearWorldModel)의 잠재 공간에서 MPPI 수행.

핵심 아이디어:
    1. 세계 모델 o_t → z_t (인코딩)
    2. 잠재 공간 rollout: z_{t+k} = f(z_t, a_t...a_{t+k})  (빠른 상상)
    3. 잠재 비용 계산: J(Z) = Σ c(C @ z_t, ref_t)
    4. MPPI 가중치 → 제어 업데이트

Dreamer (Hafner 2019) 방식으로 잠재 공간에서 MPPI 수행.
실제 동역학 rollout 대신 세계 모델 상상으로 계산량 절감.

References:
    Hafner et al. (2019) — Dream to Control (Dreamer)
    Hansen et al. (2022) — TD-MPC
    Williams et al. (2018) — MPPI
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import GaussianSampler, NoiseSampler
from mppi_controller.models.learned.rssm import LinearWorldModel


@dataclass
class WorldModelMPPIParams(MPPIParams):
    """
    World Model MPPI 파라미터

    Attributes:
        wm_latent_dim: 잠재 공간 차원
        wm_reg: 세계 모델 학습 정규화 계수
        wm_min_samples: 세계 모델 학습 최소 데이터 수
        wm_buffer_size: 경험 버퍼 최대 크기
        wm_online_training: 온라인 학습 활성화
        wm_train_interval: 온라인 학습 주기 (스텝)
        wm_use_latent_cost: 잠재 공간 비용 사용 여부
                            (False이면 decode 후 obs 비용)
        wm_imagination_horizon: 잠재 상상 호라이즌 (None이면 N과 동일)
    """

    wm_latent_dim: int = 32
    wm_reg: float = 1e-4
    wm_min_samples: int = 50
    wm_buffer_size: int = 5000
    wm_online_training: bool = False
    wm_train_interval: int = 50
    wm_use_latent_cost: bool = False
    wm_imagination_horizon: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.wm_latent_dim > 0, "wm_latent_dim must be positive"
        assert self.wm_reg > 0, "wm_reg must be positive"
        assert self.wm_min_samples >= 2, "wm_min_samples must be >= 2"
        assert self.wm_buffer_size >= self.wm_min_samples
        assert self.wm_train_interval >= 1


class WorldModelMPPIController(MPPIController):
    """
    World Model MPPI 컨트롤러.

    LinearWorldModel의 잠재 공간에서 rollout → MPPI 비용 계산.
    세계 모델이 미학습 상태이면 기본 dynamics_wrapper로 fallback.

    워크플로우:
        ctrl = WorldModelMPPIController(model, params)
        ctrl.fit_world_model(observations, actions)  # 학습
        u, info = ctrl.compute_control(state, reference)  # 잠재 MPPI

    Args:
        model: RobotModel (fallback + 학습 데이터 수집용)
        params: WorldModelMPPIParams
        cost_function: CostFunction (None이면 기본 비용)
        noise_sampler: NoiseSampler
    """

    def __init__(
        self,
        model: RobotModel,
        params: WorldModelMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(
            model=model,
            params=params,
            cost_function=cost_function,
            noise_sampler=noise_sampler,
        )
        self.wm_params = params

        # 선형 세계 모델
        self.world_model = LinearWorldModel(
            obs_dim=model.state_dim,
            action_dim=model.control_dim,
            latent_dim=params.wm_latent_dim,
        )

        # 경험 버퍼
        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[np.ndarray] = []

        self._step_count = 0
        self._wm_fit_count = 0
        self._wm_last_rmse: Optional[float] = None
        self._last_latent: Optional[np.ndarray] = None  # 최근 인코딩된 잠재 상태

    # ── 세계 모델 학습 ─────────────────────────────────────────────────────────

    @property
    def is_world_model_trained(self) -> bool:
        return self.world_model._is_fitted

    def fit_world_model(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        reg: Optional[float] = None,
    ) -> float:
        """
        관측/행동 시퀀스로 세계 모델 학습.

        Args:
            observations: (T, nx) 관측 시퀀스
            actions: (T-1, nu) 행동 시퀀스
            reg: 정규화 (None이면 params 기본값)

        Returns:
            rmse: 재구성 오차
        """
        r = reg if reg is not None else self.wm_params.wm_reg
        rmse = self.world_model.fit(observations, actions, reg=r)
        self._wm_last_rmse = rmse
        self._wm_fit_count += 1
        return rmse

    def add_observation(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray] = None,
    ) -> None:
        """
        단일 관측 (+ 행동) 버퍼 추가.

        obs_t 추가, action_{t-1} 추가 (obs_t와 쌍).
        """
        self._obs_buf.append(obs.copy())
        if action is not None:
            self._act_buf.append(action.copy())

        max_buf = self.wm_params.wm_buffer_size
        if len(self._obs_buf) > max_buf:
            self._obs_buf.pop(0)
        if len(self._act_buf) > max_buf:
            self._act_buf.pop(0)

    def fit_from_buffer(self) -> Optional[float]:
        """버퍼 데이터로 세계 모델 학습."""
        min_s = self.wm_params.wm_min_samples
        n_obs = len(self._obs_buf)
        n_act = len(self._act_buf)
        if n_obs < min_s or n_act < min_s - 1:
            return None

        T = min(n_obs, n_act + 1)
        obs = np.array(self._obs_buf[:T])
        act = np.array(self._act_buf[:T - 1])
        return self.fit_world_model(obs, act)

    def bootstrap_world_model(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """오프라인 부트스트랩 학습."""
        return self.fit_world_model(observations, actions)

    # ── 잠재 공간 rollout ──────────────────────────────────────────────────────

    def _latent_rollout(
        self,
        state: np.ndarray,
        sampled_controls: np.ndarray,
    ) -> np.ndarray:
        """
        세계 모델 잠재 공간 rollout.

        z_0 = encode(state)
        Z = imagine(z_0, controls)
        obs_traj = decode(Z)

        Args:
            state: (nx,)
            sampled_controls: (K, N, nu)

        Returns:
            trajectories: (K, N+1, nx) — 디코딩된 관측 궤적
        """
        K, N = sampled_controls.shape[:2]

        # 인코딩: state → z0
        z0 = self.world_model.encode(state[None])  # (1, latent)
        z0_batch = np.tile(z0, (K, 1))              # (K, latent)

        # 잠재 공간 상상
        Zs = self.world_model.imagine(z0_batch, sampled_controls)  # (K, N+1, latent)

        # 디코딩: z → obs
        K_, Np1, d = Zs.shape
        trajs = self.world_model.decode(Zs.reshape(-1, d)).reshape(K_, Np1, -1)

        return trajs  # (K, N+1, nx)

    # ── MPPI 제어 계산 ─────────────────────────────────────────────────────────

    def compute_control(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        World Model MPPI 제어 계산.

        세계 모델 학습 완료 시 잠재 rollout 사용,
        미학습 시 기본 dynamics_wrapper fallback.

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 참조 궤적

        Returns:
            control: (nu,)
            info: dict (standard MPPI + world_model_* 메타데이터)
        """
        K = self.params.K
        N = self.params.N

        # 노이즈 샘플링
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # rollout: 세계 모델 또는 fallback
        if self.is_world_model_trained:
            sample_trajectories = self._latent_rollout(state, sampled_controls)
            rollout_mode = "world_model_latent"
        else:
            sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
            rollout_mode = "fallback_dynamics"

        # 비용 계산
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # MPPI 가중치 + 업데이트
        weights = self._compute_weights(costs, self.params.lambda_)
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        optimal_control = self.U[0, :]

        # 버퍼 추가 (온라인 학습)
        self._step_count += 1
        self.add_observation(state, self.U[0, :] if len(self._act_buf) > 0 else None)

        if self.wm_params.wm_online_training:
            if self._step_count % self.wm_params.wm_train_interval == 0:
                self.fit_from_buffer()

        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "world_model_is_trained": self.is_world_model_trained,
            "world_model_fit_count": self._wm_fit_count,
            "world_model_latent_dim": self.wm_params.wm_latent_dim,
            "world_model_last_rmse": self._wm_last_rmse,
            "rollout_mode": rollout_mode,
        }
        self.last_info = info
        return optimal_control, info
