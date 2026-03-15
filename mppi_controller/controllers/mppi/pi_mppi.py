"""
PI-MPPI (Path Integral MPPI) Controller

원래 Path Integral 프레임워크에 기반한 MPPI.
가중치 계산에서 제어 비용을 분리하여 순수 상태 비용만 사용.

핵심 차이점 (vs Vanilla MPPI):
    Vanilla: w_k = exp(-S_total / lambda) / Z, S_total = S_state + S_control
    PI-MPPI: w_k = exp(-S_state / lambda) / Z, S_state = state cost only

이론적 근거:
    Path Integral 프레임워크에서 비용 함수:
        J = phi(x_T) + integral[q(x) + 0.5 u^T R u] dt

    제어 비용 0.5 u^T R u 는 샘플링 분포에 흡수:
        Sigma = lambda * R^{-1}

    따라서 가중치는 상태 비용만 사용:
        w_k = exp(-S_state(tau_k) / lambda) / Z

참조:
    - Theodorou et al., "A Generalized Path Integral Control Approach to
      Reinforcement Learning" (JMLR, 2010)
    - Williams et al., "Information Theoretic MPC for Model-Based
      Reinforcement Learning" (ICRA, 2017)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CostFunction,
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class PIMPPIController(MPPIController):
    """
    Path Integral MPPI Controller

    Vanilla MPPI와의 핵심 차이:
        1. 가중치 계산에 상태 비용만 사용 (제어 비용 제외)
        2. 노이즈 공분산이 Sigma = lambda * R^{-1} 관계를 만족
           (enforce_pi_covariance=True일 때)

    이로 인한 효과:
        - 제어 비용이 가중치를 왜곡하지 않음
        - 상태 비용이 작은 궤적에 더 집중
        - 탐색(exploration) 향상 (제어 비용이 가중치를 억제하지 않으므로)

    Args:
        model: RobotModel 인스턴스
        params: MPPIParams 파라미터
        cost_function: 전체 비용 함수 (상태+제어, None이면 기본)
        state_cost_function: 상태 비용만 포함 (None이면 자동 분리)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
        enforce_pi_covariance: Sigma = lambda * R^{-1} 강제
    """

    def __init__(
        self,
        model: RobotModel,
        params: MPPIParams,
        cost_function: Optional[CostFunction] = None,
        state_cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        enforce_pi_covariance: bool = False,
        **kwargs,
    ):
        # enforce_pi_covariance 시 sigma를 먼저 조정
        if enforce_pi_covariance:
            params = self._adjust_covariance(params)

        super().__init__(
            model, params,
            cost_function=cost_function,
            noise_sampler=noise_sampler,
        )

        self.enforce_pi_covariance = enforce_pi_covariance

        # 상태 비용 함수 설정
        if state_cost_function is not None:
            self.state_cost_function = state_cost_function
        elif cost_function is not None:
            # 사용자 비용에서 ControlEffortCost만 제외
            self.state_cost_function = self._filter_state_costs(cost_function)
        else:
            # 기본 비용에서 제어 비용을 제외한 상태 비용만 추출
            self.state_cost_function = self._extract_state_costs(params)

    @staticmethod
    def _adjust_covariance(params: MPPIParams) -> MPPIParams:
        """
        PI 관계식 Sigma = lambda * R^{-1} 적용

        sigma_i = sqrt(lambda / R_i) for diagonal R
        """
        R = params.R
        if R.ndim == 1:
            # 대각 R: sigma_i = sqrt(lambda / R_i)
            safe_R = np.maximum(R, 1e-8)
            new_sigma = np.sqrt(params.lambda_ / safe_R)
            params.sigma = new_sigma
        return params

    @staticmethod
    def _filter_state_costs(cost_function: CostFunction) -> CostFunction:
        """
        비용 함수에서 ControlEffortCost만 제거하여 상태 비용 추출

        CompositeMPPICost인 경우 ControlEffortCost를 제외한 나머지를 반환.
        단일 비용 함수인 경우 그대로 반환 (제어 비용이 아니라고 가정).
        """
        if isinstance(cost_function, CompositeMPPICost):
            state_costs = [
                cf for cf in cost_function.cost_functions
                if not isinstance(cf, ControlEffortCost)
            ]
            if len(state_costs) == 0:
                # 모든 비용이 제어 비용인 경우 (비현실적이지만 방어)
                return cost_function
            return CompositeMPPICost(state_costs)
        return cost_function

    @staticmethod
    def _extract_state_costs(params: MPPIParams) -> CostFunction:
        """
        기본 파라미터에서 상태 비용만 추출

        StateTrackingCost + TerminalCost (ControlEffortCost 제외)
        """
        return CompositeMPPICost([
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
        ])

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        PI-MPPI 제어 계산

        Vanilla MPPI와 동일한 파이프라인이지만,
        가중치 계산에 상태 비용만 사용.

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

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 샘플 궤적 rollout (K, N+1, nx)
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 비용 계산
        # 전체 비용 (보고용)
        total_costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )
        # 상태 비용만 (가중치 계산용) — PI-MPPI 핵심
        state_costs = self.state_cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. PI-MPPI 가중치: 상태 비용만 사용
        weights = self._compute_weights(state_costs, self.params.lambda_)

        # 6. 가중 평균으로 제어 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 7. Receding horizon 시프트
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 8. 최적 제어
        optimal_control = self.U[0, :]

        # 9. 정보 저장
        ess = self._compute_ess(weights)
        best_idx = np.argmin(total_costs)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_controls": sampled_controls,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": total_costs[best_idx],
            "mean_cost": np.mean(total_costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            # PI-MPPI 고유 정보
            "state_cost_mean": float(np.mean(state_costs)),
            "state_cost_best": float(state_costs[best_idx]),
            "control_cost_mean": float(np.mean(total_costs - state_costs)),
        }
        self.last_info = info

        return optimal_control, info

    def __repr__(self) -> str:
        return (
            f"PIMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"enforce_pi_cov={self.enforce_pi_covariance}, "
            f"params={self.params})"
        )
