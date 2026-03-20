"""
Biased-MPPI Controller (Mixture Sampling MPPI)

혼합 분포 샘플링: J개 보조 정책 제안 + (K-J)개 가우시안 샘플을 결합.
핵심 정리: importance weight에서 샘플링 분포 q_s가 소거되어,
가중치 = softmax(-S/λ)로 표준 MPPI와 동일.

이점:
  - 도메인 지식을 보조 정책으로 주입 (pure pursuit, braking 등)
  - 기존 컨트롤러/학습 모델을 보조 정책으로 재사용 가능
  - 정책 다양성으로 local minima 탈출 용이
  - 전체 교체 업데이트 U = Σ ω_k V_k

Reference: Trevisan & Alonso-Mora, RA-L 2024, arXiv:2401.09241
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import BiasedMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.controllers.mppi.ancillary_policies import (
    AncillaryPolicy,
    PreviousSolutionPolicy,
    create_policies_from_names,
)


class BiasedMPPIController(MPPIController):
    """
    Biased-MPPI Controller (24번째 MPPI 변형)

    혼합 분포 샘플링 + 전체 교체 업데이트.

    Vanilla MPPI 대비 핵심 차이:
        1. 혼합 샘플링: J개 정책 제안 + (K-J)개 가우시안
        2. 전체 교체 업데이트: U = Σ ω_k V_k (증분 아님)
        3. 적응적 λ: ESS 기반 온도 자동 조절
        4. 도메인 지식: 보조 정책으로 탐색 방향 유도

    Args:
        model: RobotModel 인스턴스
        params: BiasedMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
        policies: 보조 정책 리스트 (None이면 params.ancillary_types로 자동 생성)
    """

    def __init__(
        self,
        model: RobotModel,
        params: BiasedMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        policies: Optional[List[AncillaryPolicy]] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.biased_params = params

        # 보조 정책 설정
        if policies is not None:
            self.policies = policies
        else:
            self.policies = create_policies_from_names(params.ancillary_types)

        # 적응적 λ 상태
        self._current_lambda = params.lambda_

        # 통계 추적
        self._biased_history = []
        self._policy_best_count = 0
        self._total_steps = 0

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Biased-MPPI 제어 계산

        1. 보조 정책 샘플 생성 (J개)
        2. 가우시안 샘플 생성 (K-J개)
        3. 혼합 → rollout → 비용 → 가중치
        4. 전체 교체 업데이트: U = Σ ω_k V_k
        5. 적응적 λ 조절

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어
            info: dict 디버깅 정보
        """
        K = self.params.K
        N = self.params.N
        nu = self.model.control_dim

        # PreviousSolutionPolicy에 현재 U 전달
        for p in self.policies:
            if isinstance(p, PreviousSolutionPolicy):
                p.set_previous_solution(self.U)

        # 1. 정책 샘플 생성
        policy_controls = self._generate_policy_samples(
            state, reference_trajectory, N, nu
        )
        n_policy = policy_controls.shape[0]

        # 2. 가우시안 샘플 생성
        n_gaussian = K - n_policy
        gaussian_controls = self._generate_gaussian_samples(N, nu, n_gaussian)

        # 3. 혼합: (K, N, nu)
        all_controls = np.concatenate([policy_controls, gaussian_controls], axis=0)

        # 제어 제약
        if self.u_min is not None and self.u_max is not None:
            all_controls = np.clip(all_controls, self.u_min, self.u_max)

        # 4. Rollout + 비용
        trajectories = self.dynamics_wrapper.rollout(state, all_controls)
        costs = self.cost_function.compute_cost(
            trajectories, all_controls, reference_trajectory
        )

        # 5. 가중치 계산 (q_s 소거 — 표준 MPPI 가중치와 동일)
        if self.biased_params.use_reward_normalization:
            weights = self._compute_weights_normalized(costs)
        else:
            weights = self._compute_weights(costs, self._current_lambda)

        # 6. 전체 교체 업데이트: U = Σ ω_k V_k
        self.U = np.sum(weights[:, None, None] * all_controls, axis=0)  # (N, nu)

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 첫 제어 추출
        optimal_control = self.U[0].copy()

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 7. ESS 계산 + 적응적 λ
        ess = self._compute_ess(weights)
        if self.biased_params.use_adaptive_lambda:
            self._adapt_lambda(ess, K)

        # 통계
        best_idx = np.argmin(costs)
        policy_best = best_idx < n_policy
        self._total_steps += 1
        if policy_best:
            self._policy_best_count += 1

        biased_stats = {
            "n_policy_samples": n_policy,
            "n_gaussian_samples": n_gaussian,
            "best_is_policy": bool(policy_best),
            "policy_best_ratio": (
                self._policy_best_count / self._total_steps
                if self._total_steps > 0
                else 0.0
            ),
            "current_lambda": self._current_lambda,
            "policy_names": [p.name for p in self.policies],
        }
        self._biased_history.append(biased_stats)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self._current_lambda,
            "ess": ess,
            "num_samples": K,
            "biased_stats": biased_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _generate_policy_samples(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
        N: int,
        nu: int,
    ) -> np.ndarray:
        """
        보조 정책 샘플 생성

        각 정책에서 base_seq 생성 후:
          - 첫 샘플: pure proposal
          - 나머지: proposal + noise (다양성)

        Returns:
            (J, N, nu) 정책 제안 제어 시퀀스
        """
        samples_per = self.biased_params.samples_per_policy
        noise_scale = self.biased_params.policy_noise_scale

        all_policy_samples = []

        for policy in self.policies:
            # 기본 제안
            base_seq = policy.propose_sequence(
                state, reference_trajectory, N, self.params.dt, self.model
            )
            assert base_seq.shape == (N, nu), (
                f"Policy {policy.name} returned shape {base_seq.shape}, "
                f"expected ({N}, {nu})"
            )

            # 첫 샘플: 순수 제안
            all_policy_samples.append(base_seq)

            # 나머지: 제안 + 노이즈
            for _ in range(samples_per - 1):
                noise = noise_scale * self.params.sigma * np.random.standard_normal(
                    (N, nu)
                )
                all_policy_samples.append(base_seq + noise)

        return np.array(all_policy_samples)  # (J, N, nu)

    def _generate_gaussian_samples(
        self, N: int, nu: int, n_samples: int
    ) -> np.ndarray:
        """
        가우시안 샘플 생성 (표준 MPPI 방식)

        Returns:
            (n_samples, N, nu) 가우시안 제어 시퀀스
        """
        noise = self.noise_sampler.sample(self.U, n_samples, self.u_min, self.u_max)
        return self.U[None, :, :] + noise  # (n_samples, N, nu)

    def _compute_weights_normalized(self, costs: np.ndarray) -> np.ndarray:
        """DIAL-style 보상 정규화 가중치"""
        rewards = -costs
        std = np.std(rewards)
        if std < 1e-10:
            return np.ones(len(costs)) / len(costs)

        normalized = (rewards - np.mean(rewards)) / (std + 1e-10)
        scaled = normalized / self._current_lambda

        scaled -= np.max(scaled)
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / np.sum(exp_scaled)
        return weights

    def _adapt_lambda(self, ess: float, K: int):
        """
        ESS 기반 적응적 λ 조절

        ESS/K < ess_min_ratio → λ 증가 (더 균등한 가중치)
        ESS/K > ess_max_ratio → λ 감소 (더 집중된 가중치)
        """
        ess_ratio = ess / K
        p = self.biased_params

        if ess_ratio < p.ess_min_ratio:
            self._current_lambda = min(
                self._current_lambda * p.lambda_increase_rate,
                p.lambda_max,
            )
        elif ess_ratio > p.ess_max_ratio:
            self._current_lambda = max(
                self._current_lambda * p.lambda_decrease_rate,
                p.lambda_min,
            )

    def get_biased_statistics(self) -> Dict:
        """누적 통계 반환"""
        if not self._biased_history:
            return {
                "total_steps": 0,
                "policy_best_ratio": 0.0,
                "mean_lambda": self._current_lambda,
                "history": [],
            }

        lambdas = [h["current_lambda"] for h in self._biased_history]
        return {
            "total_steps": self._total_steps,
            "policy_best_ratio": (
                self._policy_best_count / self._total_steps
                if self._total_steps > 0
                else 0.0
            ),
            "mean_lambda": float(np.mean(lambdas)),
            "current_lambda": self._current_lambda,
            "history": self._biased_history.copy(),
        }

    def reset(self):
        """초기화"""
        super().reset()
        self._current_lambda = self.biased_params.lambda_
        self._biased_history = []
        self._policy_best_count = 0
        self._total_steps = 0

        for p in self.policies:
            if isinstance(p, PreviousSolutionPolicy):
                p.set_previous_solution(np.zeros_like(self.U))

    def __repr__(self) -> str:
        policy_names = [p.name for p in self.policies]
        return (
            f"BiasedMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"policies={policy_names}, "
            f"samples_per_policy={self.biased_params.samples_per_policy}, "
            f"K={self.params.K})"
        )
