"""
CMA-MPPI Controller (Covariance Matrix Adaptation MPPI)

CMA-ES에서 영감받아 보상 가중 샘플로부터 per-timestep 대각 공분산을 학습.
DIAL-MPPI의 등방적(isotropic) 고정 감쇠 대신, 비용 지형의 형상에 맞는
탐색 공분산을 자동 적응.

핵심: σ(i)=σ₀·f^i (DIAL 고정 스케줄) → Σ(t) ← 가중 샘플 분산 (CMA 비용 적응)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import CMAMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class CMAMPPIController(MPPIController):
    """
    CMA-MPPI Controller

    Per-timestep 대각 공분산 적응을 통해 비용 지형에 맞는 탐색 형상을 학습.

    Vanilla/DIAL MPPI 대비 핵심 차이:
        1. 다중 반복: DIAL처럼 n_iters 반복 sample-evaluate-update
        2. 공분산 적응: 반복마다 보상 가중 분산으로 per-timestep Σ(t) 업데이트
        3. EMA 안정화: Σ = (1-α)·Σ_old + α·Σ_est, 단일 반복 노이즈 방지
        4. 공분산 지속: 제어 스텝 간 학습된 Σ를 warm start 전달
        5. Receding horizon shift: U와 Σ 동시 shift

    Args:
        model: RobotModel 인스턴스
        params: CMAMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: CMAMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.cma_params = params

        # Per-timestep 대각 공분산 초기화: (N, nu)
        initial_sigma = params.sigma * params.cov_init_scale
        self.cov = np.outer(np.ones(params.N), initial_sigma ** 2)  # (N, nu)
        self._initial_cov = self.cov.copy()

        # 첫 호출 여부 추적
        self._is_first_call = True

        # 통계 추적
        self._iteration_costs = []
        self._cma_stats_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        CMA-MPPI 제어 계산

        다중 반복 + 공분산 적응으로 제어 시퀀스를 최적화.

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
            n_iters = self.cma_params.n_iters_init
            self._is_first_call = False
        else:
            n_iters = self.cma_params.n_iters

        iteration_costs = []
        last_trajectories = None
        last_weights = None
        last_costs = None

        for i in range(n_iters):
            # 1. Per-timestep 공분산 샘플링
            std = np.sqrt(self.cov)  # (N, nu)
            W = np.random.standard_normal((K, N, nu)) * std[None, :, :]  # (K, N, nu)

            # 2. 샘플 제어 생성 + 클리핑
            sampled_controls = self.U[None, :, :] + W  # (K, N, nu)
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

            # 3. Rollout + 비용 계산
            trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
            costs = self.cost_function.compute_cost(
                trajectories, sampled_controls, reference_trajectory
            )

            # 4. 가중치 계산
            if self.cma_params.use_reward_normalization:
                weights = self._compute_weights_normalized(costs)
            else:
                weights = self._compute_weights(costs, self.params.lambda_)

            # 5. Elite selection (선택적)
            if self.cma_params.elite_ratio > 0:
                weights = self._apply_elite_selection(weights, costs)

            # 6. 평균 업데이트
            if self.cma_params.use_mean_shift:
                # 전체 교체 (DIAL식): U = Σ w_k · U_sampled_k
                self.U = np.sum(
                    weights[:, None, None] * sampled_controls, axis=0
                )  # (N, nu)
            else:
                # 증분 업데이트 (Vanilla식): U += Σ w_k · noise_k
                weighted_noise = np.sum(
                    weights[:, None, None] * W, axis=0
                )  # (N, nu)
                self.U = self.U + weighted_noise

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                self.U = np.clip(self.U, self.u_min, self.u_max)

            # 7. 공분산 적응 (CMA 핵심!)
            diff = sampled_controls - self.U[None, :, :]  # (K, N, nu)
            cov_est = np.sum(
                weights[:, None, None] * (diff ** 2), axis=0
            )  # (N, nu) — 가중 분산

            # EMA 안정화
            alpha = self.cma_params.cov_learning_rate
            self.cov = (1 - alpha) * self.cov + alpha * cov_est

            # 클램핑
            sigma_min_sq = self.cma_params.sigma_min ** 2
            sigma_max_sq = self.cma_params.sigma_max ** 2
            self.cov = np.clip(self.cov, sigma_min_sq, sigma_max_sq)

            # 반복별 비용 기록
            iteration_costs.append(float(np.min(costs)))

            last_trajectories = trajectories
            last_weights = weights
            last_costs = costs

        # 첫 제어 추출
        optimal_control = self.U[0].copy()

        # Receding horizon shift (U와 Σ 모두)
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0
        self.cov = np.roll(self.cov, -1, axis=0)
        self.cov[-1, :] = self._initial_cov[-1]  # 마지막 timestep 리셋

        # 통계 저장
        self._iteration_costs = iteration_costs
        ess = self._compute_ess(last_weights)
        best_idx = np.argmin(last_costs)

        cma_stats = {
            "n_iters": n_iters,
            "iteration_costs": iteration_costs,
            "cost_improvement": (
                iteration_costs[0] - iteration_costs[-1]
                if len(iteration_costs) > 1
                else 0.0
            ),
            "cov_mean": float(np.mean(self.cov)),
            "cov_std": float(np.std(self.cov)),
            "cov_per_dim": [float(np.mean(self.cov[:, d])) for d in range(nu)],
        }
        self._cma_stats_history.append(cma_stats)

        info = {
            "sample_trajectories": last_trajectories,
            "sample_weights": last_weights,
            "best_trajectory": last_trajectories[best_idx],
            "best_cost": last_costs[best_idx],
            "mean_cost": np.mean(last_costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "cma_stats": cma_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _compute_weights_normalized(self, costs: np.ndarray) -> np.ndarray:
        """
        보상 정규화 기반 가중치 계산 (DIAL과 동일)

        rewards = -costs
        normalized = (rewards - mean) / (std + eps)
        weights = softmax(normalized / lambda)
        """
        rewards = -costs
        std = np.std(rewards)
        if std < 1e-10:
            return np.ones(len(costs)) / len(costs)

        normalized = (rewards - np.mean(rewards)) / (std + 1e-10)
        scaled = normalized / self.params.lambda_

        scaled -= np.max(scaled)
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / np.sum(exp_scaled)

        return weights

    def _apply_elite_selection(
        self, weights: np.ndarray, costs: np.ndarray
    ) -> np.ndarray:
        """
        상위 elite_ratio 비율의 샘플만 사용

        나머지 샘플의 가중치를 0으로 설정하고 재정규화.
        """
        K = len(costs)
        n_elite = max(1, int(K * self.cma_params.elite_ratio))

        # 비용 기준 상위 n_elite 인덱스
        elite_indices = np.argsort(costs)[:n_elite]
        mask = np.zeros(K)
        mask[elite_indices] = 1.0

        elite_weights = weights * mask
        total = np.sum(elite_weights)
        if total < 1e-10:
            elite_weights[elite_indices] = 1.0 / n_elite
        else:
            elite_weights = elite_weights / total

        return elite_weights

    def get_cma_statistics(self) -> Dict:
        """
        CMA-MPPI 통계 반환

        Returns:
            dict:
                - mean_cost_improvement: 평균 반복 비용 개선
                - mean_n_iters: 평균 반복 횟수
                - last_iteration_costs: 마지막 호출의 반복별 비용
                - cma_stats_history: 전체 통계 히스토리
                - current_cov_mean: 현재 공분산 평균
        """
        if len(self._cma_stats_history) == 0:
            return {
                "mean_cost_improvement": 0.0,
                "mean_n_iters": 0.0,
                "last_iteration_costs": [],
                "cma_stats_history": [],
                "current_cov_mean": 0.0,
            }

        improvements = [s["cost_improvement"] for s in self._cma_stats_history]
        n_iters_list = [s["n_iters"] for s in self._cma_stats_history]

        return {
            "mean_cost_improvement": float(np.mean(improvements)),
            "mean_n_iters": float(np.mean(n_iters_list)),
            "last_iteration_costs": self._iteration_costs,
            "cma_stats_history": self._cma_stats_history.copy(),
            "current_cov_mean": float(np.mean(self.cov)),
        }

    def reset(self):
        """제어 시퀀스 + 공분산 초기화 + 첫 호출 상태 복원"""
        super().reset()
        self.cov = self._initial_cov.copy()
        self._is_first_call = True
        self._iteration_costs = []
        self._cma_stats_history = []

    def __repr__(self) -> str:
        return (
            f"CMAMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"n_iters_init={self.cma_params.n_iters_init}, "
            f"n_iters={self.cma_params.n_iters}, "
            f"cov_lr={self.cma_params.cov_learning_rate}, "
            f"K={self.params.K})"
        )
