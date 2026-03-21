"""
Parameter-Robust MPPI Controller (PR-MPPI)

미지 파라미터에 대한 입자 기반 belief를 유지하고,
다수의 파라미터 가설 하에서 MPPI 비용을 평가하여
파라미터 불확실성에 robust한 제어 수행.
온라인으로 파라미터를 학습하면서 점진적으로 성능 향상.

핵심 수식:
    theta_particles = [theta_1, ..., theta_M]
    w_theta_i = likelihood(observation | theta_i)

    For each MPPI sample k:
      cost_k = Sigma_i w_theta_i * cost(rollout(state, u_k, model(theta_i)))  (weighted_mean)
      cost_k = max_i cost(rollout(state, u_k, model(theta_i)))                (worst_case)
      cost_k = mean of top alpha% costs across particles                       (cvar)

    Parameter update (online):
      w_theta_i proportional to p(observation | theta_i) * w_theta_i_prior   (Bayesian)

Reference: Vahs et al., 2026, arXiv:2601.02948
"""

import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, Optional, List

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import ParameterRobustMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper


class ParameterParticleFilter:
    """
    파라미터 추정을 위한 입자 필터

    M개의 파라미터 가설(입자)을 유지하고, 관측에 기반하여
    베이지안 가중치를 업데이트하며, ESS가 낮으면 재샘플링.

    Args:
        n_particles: 입자 수 M
        param_nominal: 명목 파라미터 값
        param_std: 초기 불확실성 표준편차
        param_min: 파라미터 하한
        param_max: 파라미터 상한
    """

    def __init__(
        self,
        n_particles: int,
        param_nominal: float,
        param_std: float,
        param_min: float,
        param_max: float,
    ):
        self.n_particles = n_particles
        self.param_min = param_min
        self.param_max = param_max

        # 초기 입자: 명목값 중심 가우시안 (bounds 내)
        self.particles = np.clip(
            param_nominal + param_std * np.random.randn(n_particles),
            param_min, param_max,
        )
        # 균일 초기 가중치
        self.weights = np.ones(n_particles) / n_particles

    def update_weights(
        self,
        observation_errors: np.ndarray,
        observation_noise_std: float = 0.05,
    ):
        """
        베이지안 가중치 업데이트

        likelihood proportional to exp(-0.5 * ||error||^2 / sigma^2)

        Args:
            observation_errors: (M,) 각 입자의 예측 오차 노름
            observation_noise_std: 관측 노이즈 표준편차
        """
        var = observation_noise_std ** 2
        if var < 1e-12:
            var = 1e-12

        # Log-likelihood (수치 안정)
        log_likes = -0.5 * observation_errors ** 2 / var
        log_likes -= np.max(log_likes)  # overflow 방지

        # Bayesian update: w_new proportional to likelihood * w_prior
        likes = np.exp(log_likes)
        new_weights = self.weights * likes

        # 정규화
        total = np.sum(new_weights)
        if total < 1e-30:
            # 모든 입자 가중치 퇴화 -> 균일 리셋
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights = new_weights / total

    def resample_if_needed(self, threshold: float):
        """
        체계적 재샘플링 (Systematic Resampling)

        ESS < threshold * M 이면 재샘플링 수행.

        Args:
            threshold: ESS/M 재샘플링 임계값
        """
        if self.ess < threshold * self.n_particles:
            self._systematic_resample()

    def _systematic_resample(self):
        """체계적 재샘플링 알고리즘"""
        M = self.n_particles
        positions = (np.arange(M) + np.random.uniform()) / M
        cumsum = np.cumsum(self.weights)

        indices = np.zeros(M, dtype=int)
        i, j = 0, 0
        while i < M:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
                if j >= M:
                    # 안전장치: 남은 인덱스를 마지막 입자로 채움
                    indices[i:] = M - 1
                    break

        self.particles = self.particles[indices].copy()
        self.weights = np.ones(M) / M

        # 약간의 Jitter 추가 (다양성 유지)
        jitter_std = 0.01 * (self.param_max - self.param_min)
        self.particles += jitter_std * np.random.randn(M)
        self.particles = np.clip(self.particles, self.param_min, self.param_max)

    def get_weighted_mean(self) -> float:
        """가중 평균 파라미터 추정: E[theta] = Sigma w_i * theta_i"""
        return float(np.sum(self.weights * self.particles))

    def get_best_particle(self) -> float:
        """MAP 추정: 최대 가중치 입자"""
        return float(self.particles[np.argmax(self.weights)])

    def get_weighted_std(self) -> float:
        """가중 표준편차"""
        mean = self.get_weighted_mean()
        var = np.sum(self.weights * (self.particles - mean) ** 2)
        return float(np.sqrt(max(var, 0.0)))

    @property
    def ess(self) -> float:
        """Effective Sample Size: 1 / Sigma w_i^2"""
        return float(1.0 / np.sum(self.weights ** 2))


class _ParametricModel(RobotModel):
    """
    파라미터 변형 모델 래퍼

    base model을 감싸면서, 특정 파라미터가 forward_dynamics에 영향을
    미치도록 변환. 예: wheelbase가 변하면 angular velocity가 스케일링.

    물리적 근거: Differential drive에서 omega = (v_r - v_l) / L.
    제어 입력이 (v, omega)로 주어질 때, 실제 angular velocity는
    omega_actual = omega_commanded * L_nominal / L_actual 관계.

    param_name이 'wheelbase'이면 omega를 L_nominal/L_actual로 스케일링.
    다른 파라미터는 setattr로 직접 설정 (모델이 사용하는 경우).

    Args:
        base_model: 원본 RobotModel
        param_name: 변경 파라미터 이름
        param_value: 파라미터 값
        param_nominal: 명목 파라미터 값 (스케일링 기준)
    """

    def __init__(
        self,
        base_model: RobotModel,
        param_name: str,
        param_value: float,
        param_nominal: float,
    ):
        self._base = base_model
        self._param_name = param_name
        self._param_value = param_value
        self._param_nominal = param_nominal

        # 속성 복사 (control bounds 등)
        for attr in ['v_max', 'omega_max', 'wheelbase',
                      '_control_lower', '_control_upper']:
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

        # 파라미터 설정
        setattr(self, param_name, param_value)

    @property
    def state_dim(self) -> int:
        return self._base.state_dim

    @property
    def control_dim(self) -> int:
        return self._base.control_dim

    @property
    def model_type(self) -> str:
        return self._base.model_type

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        파라미터 변형된 forward dynamics

        wheelbase 변경 시: omega를 nominal/actual 비율로 스케일링.
        이는 "컨트롤러가 생각하는 wheelbase와 실제가 다를 때" 효과를 모델링.
        """
        if self._param_name == "wheelbase" and self._param_nominal > 0:
            # omega scaling: omega_actual = omega * L_nominal / L_actual
            ratio = self._param_nominal / max(self._param_value, 1e-6)
            # 제어 변환
            modified_control = control.copy()
            modified_control[..., 1] = control[..., 1] * ratio
            return self._base.forward_dynamics(state, modified_control)
        else:
            return self._base.forward_dynamics(state, control)

    def step(
        self, state: np.ndarray, control: np.ndarray, dt: float
    ) -> np.ndarray:
        """파라미터 변형된 RK4 step"""
        if self._param_name == "wheelbase" and self._param_nominal > 0:
            ratio = self._param_nominal / max(self._param_value, 1e-6)
            modified_control = control.copy()
            modified_control[..., 1] = control[..., 1] * ratio
            return self._base.step(state, modified_control, dt)
        else:
            return self._base.step(state, control, dt)

    def get_control_bounds(self):
        return self._base.get_control_bounds()

    def set_param_value(self, value: float):
        """파라미터 값 업데이트"""
        self._param_value = value
        setattr(self, self._param_name, value)


class ParameterRobustMPPIController(MPPIController):
    """
    PR-MPPI (Parameter-Robust MPPI) Controller (37th variant)

    미지 파라미터에 대한 입자 기반 belief를 유지하고,
    다수의 파라미터 가설 하에서 MPPI 비용을 평가하여
    파라미터 불확실성에 robust한 제어 수행.

    Vanilla MPPI 대비 핵심 차이:
        1. M개 모델 변형 (서로 다른 파라미터) 유지
        2. 각 MPPI 샘플에 대해 M개 모델로 rollout -> M개 비용
        3. 비용 집계: weighted_mean / worst_case / cvar
        4. 온라인: 관측과 예측 비교로 입자 가중치 업데이트

    Args:
        model: RobotModel 인스턴스 (명목 모델)
        params: ParameterRobustMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: ParameterRobustMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.pr_params = params

        # 파라미터 입자 필터 생성
        self._particle_filter = ParameterParticleFilter(
            n_particles=params.n_particles,
            param_nominal=params.param_nominal,
            param_std=params.param_std,
            param_min=params.param_min,
            param_max=params.param_max,
        )

        # 각 입자에 대한 모델 변형 + dynamics wrapper 생성
        self._models = []
        self._dynamics_wrappers = []
        self._create_model_variants()

        # 온라인 학습용 히스토리
        self._state_history: List[np.ndarray] = []
        self._control_history: List[np.ndarray] = []
        self._step_count = 0

        # PR 통계
        self._pr_stats_history: List[Dict] = []

    def _create_model_variants(self):
        """M개 모델 변형 생성 (입자별 파라미터 적용)"""
        self._models = []
        self._dynamics_wrappers = []

        for theta in self._particle_filter.particles:
            m = _ParametricModel(
                self.model,
                self.pr_params.param_name,
                theta,
                self.pr_params.param_nominal,
            )
            self._models.append(m)
            self._dynamics_wrappers.append(
                BatchDynamicsWrapper(m, self.params.dt)
            )

    def _update_model_variants(self):
        """입자 값이 변경된 후 모델 파라미터 갱신"""
        for i, theta in enumerate(self._particle_filter.particles):
            self._models[i].set_param_value(theta)

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        PR-MPPI 제어 계산

        1. 노이즈 샘플링 (K, N, nu)
        2. 각 입자 m에 대해 rollout -> 비용 계산 (M, K)
        3. 비용 집계 (M개 -> 1개): weighted_mean / worst_case / cvar
        4. MPPI 가중치 + 업데이트
        5. 온라인: 입자 가중치 업데이트

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict
        """
        K = self.params.K
        N = self.params.N

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 모든 입자에 대해 rollout + 비용 계산
        particle_costs = self._rollout_all_particles(
            state, sampled_controls, reference_trajectory
        )  # (M, K)

        # 명목 모델로 궤적 (시각화용)
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 비용 집계
        costs = self._aggregate_costs(particle_costs)  # (K,)

        # 5. MPPI 가중치 계산
        weights = self._compute_weights(costs, self.params.lambda_)

        # 6. 가중 평균으로 제어 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 7. 첫 제어 추출
        optimal_control = self.U[0].copy()

        # 8. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 9. 온라인 파라미터 학습
        if self.pr_params.online_learning:
            self._update_history(state, optimal_control)
            if len(self._state_history) >= self.pr_params.min_observations + 1:
                self._update_parameters()

        self._step_count += 1

        # 10. ESS 및 통계
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        pr_stats = {
            "param_mean": self._particle_filter.get_weighted_mean(),
            "param_best": self._particle_filter.get_best_particle(),
            "param_std": self._particle_filter.get_weighted_std(),
            "param_ess": self._particle_filter.ess,
            "particles": self._particle_filter.particles.copy(),
            "particle_weights": self._particle_filter.weights.copy(),
            "aggregation_mode": self.pr_params.aggregation_mode,
            "n_particles": self.pr_params.n_particles,
        }
        self._pr_stats_history.append(pr_stats)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "parameter_robust_stats": pr_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _rollout_all_particles(
        self,
        state: np.ndarray,
        sampled_controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        모든 입자 모델로 rollout + 비용 계산

        Args:
            state: (nx,) 초기 상태
            sampled_controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            particle_costs: (M, K) 각 입자 x 샘플 비용
        """
        M = self.pr_params.n_particles
        K = sampled_controls.shape[0]
        particle_costs = np.zeros((M, K))

        for m in range(M):
            # 입자 m의 모델로 rollout
            trajectories_m = self._dynamics_wrappers[m].rollout(
                state, sampled_controls
            )  # (K, N+1, nx)

            # 비용 계산
            costs_m = self.cost_function.compute_cost(
                trajectories_m, sampled_controls, reference_trajectory
            )  # (K,)

            particle_costs[m] = costs_m

        return particle_costs

    def _aggregate_costs(self, particle_costs: np.ndarray) -> np.ndarray:
        """
        입자 간 비용 집계

        Args:
            particle_costs: (M, K) 각 입자 x 샘플 비용

        Returns:
            costs: (K,) 집계된 비용
        """
        mode = self.pr_params.aggregation_mode
        weights = self._particle_filter.weights  # (M,)

        if mode == "weighted_mean":
            # cost_k = Sigma_i w_i * cost_k_i
            costs = np.sum(weights[:, None] * particle_costs, axis=0)  # (K,)

        elif mode == "worst_case":
            # cost_k = max_i cost_k_i
            costs = np.max(particle_costs, axis=0)  # (K,)

        elif mode == "cvar":
            # cost_k = mean of top alpha fraction across particles
            M = particle_costs.shape[0]
            n_top = max(1, int(np.ceil(self.pr_params.cvar_alpha * M)))

            # 각 샘플 k에 대해, 입자별 비용 정렬 후 상위 alpha 평균
            sorted_costs = np.sort(particle_costs, axis=0)[::-1]  # 내림차순 (M, K)
            costs = np.mean(sorted_costs[:n_top], axis=0)  # (K,)

        else:
            # 폴백: weighted_mean
            costs = np.sum(weights[:, None] * particle_costs, axis=0)

        return costs

    def _update_history(self, state: np.ndarray, control: np.ndarray):
        """상태/제어 히스토리 업데이트"""
        self._state_history.append(state.copy())
        self._control_history.append(control.copy())

        # 윈도우 크기 제한
        max_len = self.pr_params.observation_window + 1
        if len(self._state_history) > max_len:
            self._state_history = self._state_history[-max_len:]
            self._control_history = self._control_history[-max_len:]

    def _update_parameters(self):
        """
        온라인 파라미터 학습: 관측과 예측 비교

        각 입자 m에 대해:
            1. 이전 상태 + 제어로 다음 상태 예측
            2. 실제 다음 상태와 예측 비교
            3. 가중치 업데이트: w_m proportional to exp(-error^2) * w_m
        """
        n_obs = min(
            len(self._state_history) - 1,
            self.pr_params.observation_window,
        )
        if n_obs < self.pr_params.min_observations:
            return

        M = self.pr_params.n_particles

        # 최근 n_obs개 관측에 대한 평균 예측 오차
        total_errors = np.zeros(M)

        for i in range(-n_obs, 0):
            prev_state = self._state_history[i]
            control = self._control_history[i]
            actual_next = self._state_history[i + 1]

            for m in range(M):
                predicted = self._models[m].step(
                    prev_state, control, self.params.dt
                )
                error = np.linalg.norm(predicted[:2] - actual_next[:2])
                total_errors[m] += error

        # 평균 오차
        mean_errors = total_errors / n_obs

        # 관측 노이즈 std (적응적: 오차 스케일 기반)
        obs_noise_std = max(0.01, float(np.median(mean_errors)) * 0.5)

        # 가중치 업데이트
        self._particle_filter.update_weights(mean_errors, obs_noise_std)

        # 재샘플링
        if self.pr_params.use_resampling:
            self._particle_filter.resample_if_needed(
                self.pr_params.resample_threshold
            )

        # 모델 변형 갱신
        self._update_model_variants()

    def get_parameter_statistics(self) -> Dict:
        """
        파라미터 추정 통계 반환

        Returns:
            dict:
                - param_mean: 가중 평균 추정
                - param_best: MAP 추정
                - param_std: 가중 표준편차
                - param_ess: 입자 ESS
                - particles: 입자 값 배열
                - particle_weights: 입자 가중치 배열
                - history: 전체 PR 통계 히스토리
        """
        return {
            "param_mean": self._particle_filter.get_weighted_mean(),
            "param_best": self._particle_filter.get_best_particle(),
            "param_std": self._particle_filter.get_weighted_std(),
            "param_ess": self._particle_filter.ess,
            "particles": self._particle_filter.particles.copy(),
            "particle_weights": self._particle_filter.weights.copy(),
            "history": self._pr_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 + 히스토리 + 입자 필터 초기화"""
        super().reset()

        # 입자 필터 재초기화
        self._particle_filter = ParameterParticleFilter(
            n_particles=self.pr_params.n_particles,
            param_nominal=self.pr_params.param_nominal,
            param_std=self.pr_params.param_std,
            param_min=self.pr_params.param_min,
            param_max=self.pr_params.param_max,
        )

        # 모델 변형 재생성
        self._create_model_variants()

        # 히스토리 초기화
        self._state_history = []
        self._control_history = []
        self._step_count = 0
        self._pr_stats_history = []

    def __repr__(self) -> str:
        return (
            f"ParameterRobustMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"param_name={self.pr_params.param_name}, "
            f"n_particles={self.pr_params.n_particles}, "
            f"aggregation={self.pr_params.aggregation_mode}, "
            f"K={self.params.K})"
        )
