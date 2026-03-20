"""
Residual-MPPI Controller (사전 정책 + 잔차 최적화)

사전 정책 π의 출력을 명목 시퀀스로 사용하고,
MPPI는 잔차 δu만 최적화. 증강 비용으로 정책 근처에서 탐색.

핵심 수식:
    U_nominal = π(state, ref)           # 사전 정책 출력
    ε ~ N(0, σ²)                        # 가우시안 노이즈
    V_k = U_nominal + residual_scale * ε_k  # 후보 제어
    C_aug(V_k) = C(τ_k) + kl_weight * ||V_k - U_nominal||²  # 증강 비용
    U = U_nominal + Σ ω_k * ε_k        # 가중 잔차 업데이트

Vanilla MPPI와의 핵심 차이:
    1. 명목 시퀀스가 이전 최적 해 U가 아닌 사전 정책 π(state) 출력
    2. 증강 비용으로 정책 근처 탐색 유도 (KL 페널티)
    3. 잔차만 최적화하므로 정책이 좋을수록 빠르게 수렴

이점:
    - 좋은 사전 정책이 있으면 샘플 효율 대폭 향상
    - 정책 지식을 잃지 않으면서 미세 조정
    - 정책이 나빠도 kl_weight=0으로 Vanilla로 폴백 가능

Reference: Wang et al., ICLR 2025, arXiv:2407.00898
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import ResidualMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class ResidualMPPIController(MPPIController):
    """
    Residual-MPPI Controller (25번째 MPPI 변형)

    사전 정책 + 잔차 최적화.

    Vanilla MPPI 대비 핵심 차이:
        1. 명목 시퀀스: 사전 정책 π(state, ref) 출력 사용
        2. 증강 비용: C(τ) + kl_weight * ||U - U_nominal||²
        3. 잔차 업데이트: U = U_nominal + Σ ω_k * ε_k
        4. 정책 업데이트 주기: 매 N 스텝마다 정책 재평가

    Args:
        model: RobotModel 인스턴스
        params: ResidualMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
        base_policy: 사전 정책 (callable 또는 AncillaryPolicy)
            - callable(state, ref, N, dt, model) -> (N, nu)
            - AncillaryPolicy: propose_sequence() 사용
            - None이면 policy_type에 따라 자동 생성
    """

    def __init__(
        self,
        model: RobotModel,
        params: ResidualMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        base_policy=None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.residual_params = params

        # 사전 정책 설정
        if base_policy is not None:
            self._base_policy = base_policy
        else:
            self._base_policy = self._create_default_policy(params.policy_type)

        # 정책 명목 시퀀스 캐시
        self._policy_nominal = None

        # 통계 추적
        self._step_count = 0
        self._residual_history = []

    def _create_default_policy(self, policy_type: str):
        """policy_type에 따라 기본 정책 생성"""
        if policy_type == "feedback":
            from mppi_controller.controllers.mppi.ancillary_policies import (
                PurePursuitPolicy,
            )
            return PurePursuitPolicy(lookahead=0.5, v_gain=1.0)
        elif policy_type == "zero":
            return None  # 제로 정책 — _get_policy_nominal에서 처리
        elif policy_type == "custom":
            return None  # 사용자가 set_base_policy()로 설정
        else:
            return None

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Residual-MPPI 제어 계산

        1. 사전 정책에서 명목 시퀀스 생성
        2. 명목 시퀀스를 중심으로 노이즈 샘플링
        3. Rollout + 비용 계산
        4. 증강 비용 (KL 페널티) 추가
        5. 가중 잔차 업데이트

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

        # 1. 사전 정책에서 명목 시퀀스 생성 (주기적 업데이트)
        if self._step_count % self.residual_params.policy_update_interval == 0:
            self._policy_nominal = self._get_policy_nominal(
                state, reference_trajectory, N, nu
            )

        # 2. 샘플링 중심 결정
        if self.residual_params.use_policy_nominal and self._policy_nominal is not None:
            center = self._policy_nominal
        else:
            center = self.U

        # 3. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(center, K, self.u_min, self.u_max)
        noise = noise * self.residual_params.residual_scale

        # 4. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = center[None, :, :] + noise

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 5. Rollout + 비용
        trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
        costs = self.cost_function.compute_cost(
            trajectories, sampled_controls, reference_trajectory
        )

        # 6. 증강 비용: KL 페널티 (정책에서 벗어나는 것에 페널티)
        if (
            self.residual_params.use_augmented_cost
            and self._policy_nominal is not None
            and self.residual_params.kl_weight > 0
        ):
            residuals = sampled_controls - self._policy_nominal[None, :, :]
            kl_cost = self.residual_params.kl_weight * np.sum(
                residuals ** 2, axis=(1, 2)
            )
            costs = costs + kl_cost

        # 7. 가중치 계산
        weights = self._compute_weights(costs, self.params.lambda_)

        # 8. 가중 잔차 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)  # (N, nu)
        self.U = center + weighted_noise

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 첫 제어 추출
        optimal_control = self.U[0].copy()

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 통계
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        residual_norm = 0.0
        if self._policy_nominal is not None:
            # self.U는 shift 후이므로 shift 전 값 비교 대신 weighted_noise 사용
            residual_norm = float(np.linalg.norm(weighted_noise))

        self._step_count += 1

        residual_stats = {
            "residual_norm": residual_norm,
            "policy_cost": float(costs[0]) if len(costs) > 0 else 0.0,
            "best_cost": float(costs[best_idx]),
            "kl_weight": self.residual_params.kl_weight,
            "residual_scale": self.residual_params.residual_scale,
        }
        self._residual_history.append(residual_stats)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "residual_stats": residual_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _get_policy_nominal(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
        N: int,
        nu: int,
    ) -> np.ndarray:
        """
        사전 정책에서 명목 시퀀스 생성

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적
            N: 호라이즌
            nu: 제어 차원

        Returns:
            nominal: (N, nu) 명목 제어 시퀀스
        """
        if self._base_policy is None:
            return np.zeros((N, nu))

        if hasattr(self._base_policy, "propose_sequence"):
            return self._base_policy.propose_sequence(
                state, reference_trajectory, N, self.params.dt, self.model
            )
        elif callable(self._base_policy):
            return self._base_policy(
                state, reference_trajectory, N, self.params.dt, self.model
            )
        else:
            return np.zeros((N, nu))

    def set_base_policy(self, policy):
        """
        사전 정책 설정/변경

        Args:
            policy: callable 또는 AncillaryPolicy
        """
        self._base_policy = policy
        self._policy_nominal = None

    def get_residual_statistics(self) -> Dict:
        """누적 잔차 통계 반환"""
        if not self._residual_history:
            return {
                "total_steps": 0,
                "mean_residual_norm": 0.0,
                "mean_best_cost": 0.0,
                "history": [],
            }

        norms = [h["residual_norm"] for h in self._residual_history]
        costs = [h["best_cost"] for h in self._residual_history]

        return {
            "total_steps": self._step_count,
            "mean_residual_norm": float(np.mean(norms)),
            "std_residual_norm": float(np.std(norms)),
            "mean_best_cost": float(np.mean(costs)),
            "min_best_cost": float(np.min(costs)),
            "history": self._residual_history.copy(),
        }

    def reset(self):
        """초기화"""
        super().reset()
        self._step_count = 0
        self._policy_nominal = None
        self._residual_history = []

    def __repr__(self) -> str:
        policy_name = "None"
        if self._base_policy is not None:
            if hasattr(self._base_policy, "name"):
                policy_name = self._base_policy.name
            elif callable(self._base_policy):
                policy_name = "callable"
        return (
            f"ResidualMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"policy={policy_name}, "
            f"kl_weight={self.residual_params.kl_weight}, "
            f"K={self.params.K})"
        )
