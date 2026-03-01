"""
DIAL-MPPI Controller (Diffusion Annealing MPPI)

DIAL-MPC (ICRA 2025 Best Paper Finalist) 기반 구현.
표준 MPPI가 단일 확산 단계와 수학적으로 동치임을 이용,
다단계 확산 어닐링으로 local minima를 회피하면서 정밀한 솔루션에 수렴.

핵심: 큰 노이즈(전역 탐색) → 작은 노이즈(지역 정밀화)를 반복.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import DIALMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class DIALMPPIController(MPPIController):
    """
    DIAL-MPPI Controller

    다단계 확산 어닐링을 통해 MPPI의 sample efficiency를 향상.

    Vanilla MPPI 대비 핵심 차이:
        1. 다중 반복: 1회 → n_diffuse회 sample-evaluate-update 루프
        2. 어닐링 노이즈: 고정 sigma → 반복마다 감소 + 호라이즌 의존
        3. 전체 교체 업데이트: U = Σw·(U+ε) (다중 반복에서 더 안정)
        4. Cold/Warm start 구분: 첫 호출 10회 vs 이후 3회
        5. 보상 정규화: (r-mean)/std/λ 로 수치 안정성 향상

    Args:
        model: RobotModel 인스턴스
        params: DIALMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: DIALMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.dial_params = params

        # 첫 호출 여부 추적
        self._is_first_call = True

        # 호라이즌 노이즈 프로파일 사전 계산 (N, nu)
        # horizon_profile[t] = horizon_diffuse_factor^(N-1-t) * sigma_scale
        # t=0(가까운 미래): 작은 노이즈, t=N-1(먼 미래): 큰 노이즈
        self._horizon_profile = self._compute_horizon_profile()

        # 통계 추적
        self._iteration_costs = []  # 반복별 비용 기록
        self._dial_stats_history = []

    def _compute_horizon_profile(self) -> np.ndarray:
        """호라이즌 의존 노이즈 프로파일 계산 (N,)

        선형 보간: t=0에서 factor, t=N-1에서 1.0
        → 가까운 미래는 작은 노이즈(정밀), 먼 미래는 큰 노이즈(탐색)
        """
        N = self.dial_params.N
        factor = self.dial_params.horizon_diffuse_factor
        scale = self.dial_params.sigma_scale
        # 선형 보간: factor → 1.0
        if N == 1:
            profile = np.array([1.0])
        else:
            profile = factor + (1.0 - factor) * np.arange(N) / (N - 1)
        return profile * scale

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        DIAL-MPPI 제어 계산

        다단계 확산 어닐링으로 제어 시퀀스를 최적화.

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K
        N = self.params.N
        nu = self.model.control_dim

        # Cold start vs Warm start
        if self._is_first_call:
            n_iters = self.dial_params.n_diffuse_init
            self._is_first_call = False
        else:
            n_iters = self.dial_params.n_diffuse

        iteration_costs = []

        # 마지막 반복의 정보를 저장할 변수
        last_trajectories = None
        last_weights = None
        last_costs = None

        for i in range(n_iters):
            # 1. 어닐링된 노이즈 스케일 계산
            traj_scale = self.dial_params.traj_diffuse_factor ** i
            # annealed_sigma: (N, nu) = horizon_profile (N,) * sigma (nu,) * traj_scale
            annealed_sigma = (
                self._horizon_profile[:, None] * self.params.sigma[None, :] * traj_scale
            )

            # 2. 샘플링: W ~ N(0, annealed_sigma)
            rng_noise = np.random.standard_normal((K, N, nu))
            W = rng_noise * annealed_sigma[None, :, :]  # (K, N, nu)

            # 3. 샘플 제어 생성 + 클리핑
            sampled_controls = self.U[None, :, :] + W  # (K, N, nu)
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

            # 4. Rollout + 비용 계산 (기존 인프라 재사용)
            trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
            costs = self.cost_function.compute_cost(
                trajectories, sampled_controls, reference_trajectory
            )

            # 5. 가중치 계산
            if self.dial_params.use_reward_normalization:
                weights = self._compute_weights_normalized(costs)
            else:
                weights = self._compute_weights(costs, self.params.lambda_)

            # 6. 전체 교체 업데이트: U = Σ w_k * sampled_controls_k
            self.U = np.sum(
                weights[:, None, None] * sampled_controls, axis=0
            )  # (N, nu)

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                self.U = np.clip(self.U, self.u_min, self.u_max)

            # 반복별 비용 기록
            iteration_costs.append(float(np.min(costs)))

            # 마지막 반복 데이터 저장
            last_trajectories = trajectories
            last_weights = weights
            last_costs = costs

        # 첫 제어 추출 → shift → 반환
        optimal_control = self.U[0].copy()

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 통계 저장
        self._iteration_costs = iteration_costs
        ess = self._compute_ess(last_weights)
        best_idx = np.argmin(last_costs)

        dial_stats = {
            "n_iters": n_iters,
            "iteration_costs": iteration_costs,
            "cost_improvement": iteration_costs[0] - iteration_costs[-1] if len(iteration_costs) > 1 else 0.0,
        }
        self._dial_stats_history.append(dial_stats)

        info = {
            "sample_trajectories": last_trajectories,
            "sample_weights": last_weights,
            "best_trajectory": last_trajectories[best_idx],
            "best_cost": last_costs[best_idx],
            "mean_cost": np.mean(last_costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "dial_stats": dial_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _compute_weights_normalized(self, costs: np.ndarray) -> np.ndarray:
        """
        보상 정규화 기반 가중치 계산

        rewards = -costs
        normalized = (rewards - mean) / (std + eps)
        weights = softmax(normalized / lambda)

        Args:
            costs: (K,) 비용 배열

        Returns:
            weights: (K,) 정규화된 가중치
        """
        rewards = -costs
        std = np.std(rewards)
        if std < 1e-10:
            # 모든 비용이 동일하면 균등 가중치
            return np.ones(len(costs)) / len(costs)

        normalized = (rewards - np.mean(rewards)) / (std + 1e-10)
        scaled = normalized / self.params.lambda_

        # 수치 안정성을 위한 max-shift
        scaled -= np.max(scaled)
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / np.sum(exp_scaled)

        return weights

    def get_dial_statistics(self) -> Dict:
        """
        DIAL-MPPI 통계 반환

        Returns:
            dict:
                - mean_cost_improvement: 평균 반복 비용 개선
                - mean_n_iters: 평균 반복 횟수
                - last_iteration_costs: 마지막 호출의 반복별 비용
                - dial_stats_history: 전체 통계 히스토리
        """
        if len(self._dial_stats_history) == 0:
            return {
                "mean_cost_improvement": 0.0,
                "mean_n_iters": 0.0,
                "last_iteration_costs": [],
                "dial_stats_history": [],
            }

        improvements = [s["cost_improvement"] for s in self._dial_stats_history]
        n_iters_list = [s["n_iters"] for s in self._dial_stats_history]

        return {
            "mean_cost_improvement": float(np.mean(improvements)),
            "mean_n_iters": float(np.mean(n_iters_list)),
            "last_iteration_costs": self._iteration_costs,
            "dial_stats_history": self._dial_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 초기화 및 첫 호출 상태 복원"""
        super().reset()
        self._is_first_call = True
        self._iteration_costs = []
        self._dial_stats_history = []

    def __repr__(self) -> str:
        return (
            f"DIALMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"n_diffuse_init={self.dial_params.n_diffuse_init}, "
            f"n_diffuse={self.dial_params.n_diffuse}, "
            f"traj_factor={self.dial_params.traj_diffuse_factor}, "
            f"K={self.params.K})"
        )
