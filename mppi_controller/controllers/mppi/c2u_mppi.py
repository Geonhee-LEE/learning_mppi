"""
C2U-MPPI (Chance-Constrained Unscented MPPI) 컨트롤러

Unscented Transform으로 비선형 공분산을 전파하고
확률적 기회 제약조건(Chance Constraint)을 MPPI 비용에 직접 반영.

핵심 아이디어:
  1. UT: σ-point → 비선형 전파 → 공분산 복원 (EKF 선형화보다 정확)
  2. CC: P(collision) ≤ α → r_eff = r + κ_α * √(trace(Σ_pos))
  3. 불확실할수록 장애물이 커지는 효과 → 보수적 경로 선택

논문: Excess Risk Approach for Planning Under Stochastic Dynamics

propagation_mode:
  - "nominal": 이전 제어 시퀀스 기반 1회 UT (O(N), 기본)
  - "per_sample": K개 샘플 각각 UT 전파 (O(K*N), 정확하지만 느림)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable, List

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import C2UMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction, CompositeMPPICost
from mppi_controller.controllers.mppi.chance_constraint_cost import ChanceConstraintCost
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class UnscentedTransform:
    """
    Unscented Transform (UT)

    비선형 함수를 통한 가우시안 분포의 공분산 전파.
    EKF의 야코비안 선형화보다 2차 항까지 정확.

    Args:
        n: 상태 차원
        alpha: σ-point 분산 스케일 (기본 1e-3)
        beta: 사전 분포 정보 (가우시안=2)
        kappa: 2차 스케일링 (기본 0)
    """

    def __init__(
        self,
        n: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # λ = α²(n + κ) - n
        self.lam = alpha**2 * (n + kappa) - n

        # 가중치 사전 계산
        self._Wm, self._Wc = self._compute_weights()

    def _compute_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """평균/공분산 가중치 계산 (2n+1,)"""
        n = self.n
        lam = self.lam
        num_sigma = 2 * n + 1

        Wm = np.full(num_sigma, 1.0 / (2.0 * (n + lam)))
        Wc = np.full(num_sigma, 1.0 / (2.0 * (n + lam)))

        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)

        return Wm, Wc

    @property
    def weights_mean(self) -> np.ndarray:
        """평균 가중치 (2n+1,)"""
        return self._Wm

    @property
    def weights_cov(self) -> np.ndarray:
        """공분산 가중치 (2n+1,)"""
        return self._Wc

    def compute_sigma_points(
        self, mean: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        """
        σ-point 생성

        Args:
            mean: (n,) 평균
            cov: (n, n) 공분산

        Returns:
            sigma_points: (2n+1, n) σ-points
        """
        n = self.n
        sigma_points = np.zeros((2 * n + 1, n))

        # σ_0 = μ
        sigma_points[0] = mean

        # √((n + λ) * P) — Cholesky 분해
        try:
            L = np.linalg.cholesky((n + self.lam) * cov)
        except np.linalg.LinAlgError:
            # 양정치 보정
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = min(eigvals)
            if min_eig < 0:
                cov_fixed = cov + (-min_eig + 1e-10) * np.eye(n)
            else:
                cov_fixed = cov + 1e-10 * np.eye(n)
            L = np.linalg.cholesky((n + self.lam) * cov_fixed)

        for i in range(n):
            sigma_points[1 + i] = mean + L[i]
            sigma_points[1 + n + i] = mean - L[i]

        return sigma_points

    def propagate(
        self,
        sigma_points: np.ndarray,
        transform_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        σ-point를 비선형 함수로 전파하고 평균/공분산 복원

        Args:
            sigma_points: (2n+1, n) σ-points
            transform_fn: (n,) → (m,) 비선형 변환

        Returns:
            mean_pred: (m,) 전파된 평균
            cov_pred: (m, m) 전파된 공분산
        """
        num_sigma = sigma_points.shape[0]

        # 각 σ-point 전파
        transformed = np.array([transform_fn(sp) for sp in sigma_points])
        m = transformed.shape[1]

        # 가중 평균
        mean_pred = np.sum(self._Wm[:, None] * transformed, axis=0)

        # 가중 공분산
        diff = transformed - mean_pred  # (2n+1, m)
        cov_pred = np.zeros((m, m))
        for i in range(num_sigma):
            cov_pred += self._Wc[i] * np.outer(diff[i], diff[i])

        return mean_pred, cov_pred

    def propagate_trajectory(
        self,
        initial_mean: np.ndarray,
        initial_cov: np.ndarray,
        controls: np.ndarray,
        dynamics_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        process_noise: Optional[np.ndarray] = None,
        dt: float = 0.1,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        제어 시퀀스에 따라 궤적의 공분산을 UT로 전파

        Args:
            initial_mean: (nx,) 초기 평균
            initial_cov: (nx, nx) 초기 공분산
            controls: (N, nu) 제어 시퀀스
            dynamics_fn: (state, control, dt) → next_state
            process_noise: (nx, nx) 프로세스 노이즈 Q (None → 0)
            dt: 타임스텝

        Returns:
            mean_traj: (N+1, nx) 평균 궤적
            cov_traj: [(nx, nx)] * (N+1) 공분산 궤적
        """
        N = controls.shape[0]
        nx = initial_mean.shape[0]

        mean_traj = np.zeros((N + 1, nx))
        cov_traj: List[np.ndarray] = []

        mean_traj[0] = initial_mean
        cov_traj.append(initial_cov.copy())

        mu = initial_mean.copy()
        P = initial_cov.copy()

        for t in range(N):
            u_t = controls[t]

            # dynamics 한 스텝 전파 함수 (상태만 입력)
            def step_fn(state, _u=u_t, _dt=dt):
                return dynamics_fn(state, _u, _dt)

            # σ-point 생성 + 전파
            sigma_pts = self.compute_sigma_points(mu, P)
            mu, P = self.propagate(sigma_pts, step_fn)

            # 프로세스 노이즈 추가
            if process_noise is not None:
                P = P + process_noise

            # 대칭성 보장
            P = 0.5 * (P + P.T)

            mean_traj[t + 1] = mu
            cov_traj.append(P.copy())

        return mean_traj, cov_traj


class C2UMPPIController(MPPIController):
    """
    C2U-MPPI (Chance-Constrained Unscented MPPI) 컨트롤러

    Unscented Transform으로 비선형 공분산 전파 +
    확률적 기회 제약조건을 MPPI 비용에 직접 반영.

    Args:
        model: RobotModel 인스턴스
        params: C2UMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
        uncertainty_fn: (states, controls) → covariance (batch, nx, nx)
            None이면 process_noise_scale * I 사용
    """

    def __init__(
        self,
        model: RobotModel,
        params: C2UMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        uncertainty_fn: Optional[Callable] = None,
    ):
        # ChanceConstraintCost를 cost_function에 추가
        self.chance_cost = ChanceConstraintCost(
            obstacles=params.cc_obstacles,
            chance_alpha=params.chance_alpha,
            weight=params.chance_cost_weight,
            margin_factor=params.cc_margin_factor,
        )

        if cost_function is not None:
            # 기존 비용 함수에 ChanceConstraintCost 추가
            if isinstance(cost_function, CompositeMPPICost):
                cost_function.cost_functions.append(self.chance_cost)
                combined_cost = cost_function
            else:
                combined_cost = CompositeMPPICost([cost_function, self.chance_cost])
        else:
            combined_cost = None  # 기본 비용 함수 사용 (super에서 생성)

        super().__init__(model, params, combined_cost, noise_sampler)

        # 기본 비용 함수로 생성된 경우 ChanceConstraintCost 추가
        if cost_function is None:
            if isinstance(self.cost_function, CompositeMPPICost):
                self.cost_function.cost_functions.append(self.chance_cost)
            else:
                self.cost_function = CompositeMPPICost(
                    [self.cost_function, self.chance_cost]
                )

        self.c2u_params = params

        # Unscented Transform
        self.ut = UnscentedTransform(
            n=model.state_dim,
            alpha=params.ut_alpha,
            beta=params.ut_beta,
            kappa=params.ut_kappa,
        )

        # 불확실성 함수 (외부 제공 또는 자동 감지)
        self.uncertainty_fn = self._resolve_uncertainty_fn(uncertainty_fn, model)

        # 공분산 궤적 저장
        self._cov_trajectory: Optional[List[np.ndarray]] = None

    @staticmethod
    def _resolve_uncertainty_fn(
        uncertainty_fn: Optional[Callable], model: RobotModel
    ) -> Optional[Callable]:
        """불확실성 함수 결정"""
        if uncertainty_fn is not None:
            return uncertainty_fn
        if hasattr(model, "predict_with_uncertainty"):
            return model.predict_with_uncertainty
        return None

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        C2U-MPPI 제어 계산

        1. UT로 nominal 공분산 전파
        2. ChanceConstraintCost에 공분산 전달
        3. 기본 MPPI compute_control 호출
        4. info에 공분산 정보 추가

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 info + covariance_trajectory, effective_radii
        """
        # 1. UT 공분산 전파
        P0 = self._get_initial_covariance()
        Q = self._get_process_noise()

        def dynamics_fn(s, u, dt):
            return self.model.step(s, u, dt)

        mean_traj, cov_traj = self.ut.propagate_trajectory(
            initial_mean=state,
            initial_cov=P0,
            controls=self.U,
            dynamics_fn=dynamics_fn,
            process_noise=Q,
            dt=self.params.dt,
        )

        # 2. ChanceConstraintCost에 공분산 설정
        self.chance_cost.set_covariance_trajectory(cov_traj)
        self._cov_trajectory = cov_traj

        # 3. 기본 MPPI 파이프라인
        control, info = super().compute_control(state, reference_trajectory)

        # 4. 추가 정보
        info["covariance_trajectory"] = cov_traj
        info["mean_trajectory_ut"] = mean_traj
        info["effective_radii"] = self.chance_cost.get_effective_radii()

        # 공분산 통계
        trace_vals = [np.trace(c[:2, :2]) for c in cov_traj]
        info["covariance_stats"] = {
            "initial_trace": float(trace_vals[0]),
            "final_trace": float(trace_vals[-1]),
            "max_trace": float(max(trace_vals)),
            "mean_trace": float(np.mean(trace_vals)),
        }

        return control, info

    def _get_initial_covariance(self) -> np.ndarray:
        """초기 공분산 (작은 값)"""
        nx = self.model.state_dim
        return 1e-4 * np.eye(nx)

    def _get_process_noise(self) -> np.ndarray:
        """프로세스 노이즈 Q 행렬"""
        nx = self.model.state_dim
        if self.uncertainty_fn is not None:
            # uncertainty_fn이 있으면 Q를 동적으로 추정할 수 있지만,
            # 기본적으로는 process_noise_scale을 사용
            pass
        return self.c2u_params.process_noise_scale * np.eye(nx)

    def get_covariance_trajectory(self) -> Optional[List[np.ndarray]]:
        """마지막 compute_control의 공분산 궤적 반환"""
        return self._cov_trajectory

    def reset(self):
        """상태 초기화"""
        super().reset()
        self._cov_trajectory = None
        self.chance_cost.reset_covariance()

    def __repr__(self) -> str:
        return (
            f"C2UMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"chance_α={self.c2u_params.chance_alpha}, "
            f"mode={self.c2u_params.propagation_mode})"
        )
