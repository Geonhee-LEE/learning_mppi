"""
Koopman MPPI 컨트롤러

Koopman 연산자 기반 선형 동역학을 사용한 MPPI.
비선형 동역학을 고차원 특징 공간에서 선형으로 표현하여
배치 rollout을 대폭 가속화.

핵심 아이디어:
    z_{t+1} = K @ φ(x_t) + B @ u_t  (선형 예측)
    x_{t+1} ≈ C @ z_{t+1}            (역투영)

배치 rollout:
    전체 호라이즌을 for-loop 없이 행렬 연산으로 처리.

References:
    Koopman (1931) — Original operator theory
    Brunton et al. (2016) — DMD and Koopman modes
    Li et al. (2017) — Extended DMD (EDMD)
    Lusch et al. (2018) — Deep learning for Koopman
    Bevanda et al. (2021) — Koopman operator dynamical models
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams, KoopmanMPPIParams
from mppi_controller.controllers.mppi.cost_functions import (
    CostFunction,
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi.sampling import GaussianSampler, NoiseSampler
from mppi_controller.models.learned.koopman_dynamics import KoopmanDynamics


class KoopmanBatchRollout:
    """
    KoopmanDynamics를 사용한 배치 rollout 헬퍼.

    MPPIController의 dynamics_wrapper를 대체.
    선형 Koopman 예측으로 O(K*N) 대신 벡터화된 배치 계산.

    Args:
        koopman_model: 학습된 KoopmanDynamics 인스턴스
        fallback_wrapper: Koopman 미학습 시 사용할 기본 동역학 래퍼
    """

    def __init__(
        self,
        koopman_model: KoopmanDynamics,
        fallback_wrapper=None,
    ):
        self.koopman = koopman_model
        self.fallback = fallback_wrapper

    @property
    def is_trained(self) -> bool:
        return self.koopman._is_fitted

    def rollout(
        self,
        state: np.ndarray,
        sampled_controls: np.ndarray,
    ) -> np.ndarray:
        """
        배치 궤적 rollout.

        학습된 경우 Koopman 선형 rollout, 미학습 시 fallback.

        Args:
            state: (nx,) 초기 상태
            sampled_controls: (K, N, nu) 샘플 제어 시퀀스

        Returns:
            trajectories: (K, N+1, nx)
        """
        if self.is_trained:
            return self._koopman_rollout(state, sampled_controls)
        elif self.fallback is not None:
            return self.fallback.rollout(state, sampled_controls)
        else:
            raise RuntimeError(
                "Koopman 모델이 미학습 상태이고 fallback이 없습니다. "
                "fit_edmd() 또는 bootstrap_koopman()을 호출하세요."
            )

    def _koopman_rollout(
        self,
        state: np.ndarray,
        sampled_controls: np.ndarray,
    ) -> np.ndarray:
        """
        Koopman 선형 배치 rollout.

        K개 샘플 초기 상태는 모두 동일하므로:
            states: (K, nx) — state 반복
            controls: (K, N, nu) — sampled_controls

        Args:
            state: (nx,)
            sampled_controls: (K, N, nu)

        Returns:
            trajectories: (K, N+1, nx)
        """
        K = sampled_controls.shape[0]
        states_batch = np.tile(state[None, :], (K, 1))  # (K, nx)
        return self.koopman.batch_rollout_linear(states_batch, sampled_controls)


class KoopmanMPPIController(MPPIController):
    """
    Koopman MPPI 컨트롤러.

    Koopman 연산자 기반 선형 동역학으로 rollout을 가속화한 MPPI.

    워크플로우:
        1. KoopmanMPPIController 생성 (내부에 KoopmanDynamics 보유)
        2. 데이터 수집 후 fit_edmd() 또는 bootstrap_koopman() 호출
        3. 학습 후 compute_control() 호출 → 선형 rollout 사용
        4. 미학습 시 일반 dynamics_wrapper fallback 사용

    Args:
        model: RobotModel 인스턴스 (기본 동역학 — fallback + EDMD 데이터 수집용)
        params: KoopmanMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: KoopmanMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(
            model=model,
            params=params,
            cost_function=cost_function,
            noise_sampler=noise_sampler,
        )

        self.koopman_params = params

        # Koopman 동역학 모델 생성
        self.koopman_model = KoopmanDynamics(
            nx=model.state_dim,
            nu=model.control_dim,
            lift_fn=params.koopman_lift_fn,
            lift_dim=params.koopman_lift_dim,
            rbf_gamma=params.koopman_rbf_gamma,
            poly_degree=params.koopman_poly_degree,
            use_bias=params.koopman_use_bias,
        )

        # 배치 rollout 헬퍼 (fallback: 기본 dynamics_wrapper)
        self._koopman_rollout = KoopmanBatchRollout(
            koopman_model=self.koopman_model,
            fallback_wrapper=self.dynamics_wrapper,
        )

        # 데이터 수집 버퍼 (온라인 학습용)
        self._X_buf: List[np.ndarray] = []
        self._U_buf: List[np.ndarray] = []
        self._Xnext_buf: List[np.ndarray] = []

        # 통계
        self._step_count = 0
        self._fit_count = 0
        self._last_fit_rmse: Optional[float] = None

    @property
    def is_koopman_trained(self) -> bool:
        """Koopman 모델 학습 여부"""
        return self.koopman_model._is_fitted

    @property
    def phi_dim(self) -> int:
        """특징 공간 차원"""
        return self.koopman_model._phi_dim

    # ── Koopman 학습 ──────────────────────────────────────────────────────────

    def fit_edmd(
        self,
        X: np.ndarray,
        U: np.ndarray,
        X_next: np.ndarray,
        reg: float = 1e-4,
    ) -> float:
        """
        외부 데이터로 EDMD 학습.

        Args:
            X: (M, nx) 현재 상태
            U: (M, nu) 제어
            X_next: (M, nx) 다음 상태
            reg: 정규화 계수

        Returns:
            rmse: 예측 오차 (RMSE)
        """
        rmse = self.koopman_model.fit_edmd(X, U, X_next, reg=reg)
        self._last_fit_rmse = rmse
        self._fit_count += 1
        return rmse

    def add_transition(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """온라인 데이터 추가."""
        self._X_buf.append(state.copy())
        self._U_buf.append(control.copy())
        self._Xnext_buf.append(next_state.copy())

        # 버퍼 크기 제한
        max_buf = self.koopman_params.koopman_buffer_size
        if len(self._X_buf) > max_buf:
            self._X_buf.pop(0)
            self._U_buf.pop(0)
            self._Xnext_buf.pop(0)

    def bootstrap_koopman(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        reg: Optional[float] = None,
    ) -> float:
        """
        오프라인 데이터로 Koopman 부트스트랩 학습.

        Args:
            states: (T+1, nx) 상태 궤적
            controls: (T, nu) 제어 시퀀스
            reg: 정규화 계수 (None이면 params 기본값)

        Returns:
            rmse: 예측 오차
        """
        T = controls.shape[0]
        X = states[:T]
        X_next = states[1:T + 1]
        U = controls
        reg = reg if reg is not None else self.koopman_params.koopman_reg
        return self.fit_edmd(X, U, X_next, reg=reg)

    def fit_from_buffer(self) -> Optional[float]:
        """
        버퍼 데이터로 EDMD 학습.

        Returns:
            rmse: 예측 오차 (데이터 부족 시 None)
        """
        min_samples = self.koopman_params.koopman_min_samples
        if len(self._X_buf) < min_samples:
            return None

        X = np.array(self._X_buf)
        U = np.array(self._U_buf)
        X_next = np.array(self._Xnext_buf)
        return self.fit_edmd(X, U, X_next, reg=self.koopman_params.koopman_reg)

    # ── MPPI 제어 계산 ────────────────────────────────────────────────────────

    def compute_control(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Koopman MPPI 제어 계산.

        Koopman 학습 완료 시 선형 rollout 사용,
        미학습 시 기본 dynamics_wrapper fallback.

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어
            info: dict (sample_trajectories, sample_weights, best_trajectory,
                        koopman_is_trained, koopman_fit_count, koopman_phi_dim,
                        koopman_last_rmse, rollout_mode)
        """
        K = self.params.K
        N = self.params.N

        # 1. 노이즈 샘플링
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. Koopman 또는 fallback rollout
        sample_trajectories = self._koopman_rollout.rollout(state, sampled_controls)

        # 4. 비용 계산
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. MPPI 가중치
        weights = self._compute_weights(costs, self.params.lambda_)

        # 6. 제어 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 7. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        optimal_control = self.U[0, :]

        # 8. 온라인 학습 (옵션)
        self._step_count += 1
        if self.koopman_params.koopman_online_fitting:
            interval = self.koopman_params.koopman_fit_interval
            if self._step_count % interval == 0:
                self.fit_from_buffer()

        # 9. 정보 반환
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)
        rollout_mode = "koopman_linear" if self.is_koopman_trained else "fallback_dynamics"

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "koopman_is_trained": self.is_koopman_trained,
            "koopman_fit_count": self._fit_count,
            "koopman_phi_dim": self.phi_dim,
            "koopman_last_rmse": self._last_fit_rmse,
            "rollout_mode": rollout_mode,
        }
        self.last_info = info
        return optimal_control, info
