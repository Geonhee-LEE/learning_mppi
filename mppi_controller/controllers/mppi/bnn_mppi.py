"""
BNN Surrogate MPPI 컨트롤러

앙상블 불확실성 기반 궤적 feasibility 평가 + 필터링.

UncertaintyMPPI와의 차이:
- UncertaintyMPPI: σ를 sampling noise에 반영 (탐색 적응)
- BNN-MPPI: σ를 cost에 반영 + 저신뢰 궤적 필터링 (feasibility)

Ezeji et al. 2025 + CoVO-MPC 영감.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable

from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import BNNMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.models.base_model import RobotModel


class FeasibilityCost(CostFunction):
    """
    앙상블 불확실성 기반 궤적 feasibility 비용

    J_feas(k) = β × Σ_t reduce(σ(x_{k,t}, u_{k,t})²)
    feasibility_score(k) = exp(-J_feas(k) / (β × N))  ∈ (0, 1]

    Args:
        uncertainty_fn: (states, controls) → (batch, nx) 불확실성 (표준편차)
        weight: 불확실성 비용 가중치 β
        reduce: 상태 차원 축소 ("sum" | "max" | "mean")
    """

    def __init__(
        self,
        uncertainty_fn: Callable,
        weight: float = 50.0,
        reduce: str = "sum",
    ):
        self.uncertainty_fn = uncertainty_fn
        self.weight = weight
        self.reduce = reduce

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        불확실성 페널티 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx) — 미사용

        Returns:
            costs: (K,) 불확실성 페널티
        """
        K, N_plus_1, nx = trajectories.shape
        N = controls.shape[1]
        total = np.zeros(K)

        for t in range(N):
            states_t = trajectories[:, t, :]   # (K, nx)
            controls_t = controls[:, t, :]     # (K, nu)

            # 불확실성 추정 (K, nx)
            sigma_t = self.uncertainty_fn(states_t, controls_t)
            if sigma_t.ndim == 1:
                sigma_t = np.broadcast_to(sigma_t, (K, sigma_t.shape[0]))

            # σ² → 스칼라 (K,)
            var_t = sigma_t ** 2
            if self.reduce == "sum":
                penalty_t = np.sum(var_t, axis=-1)
            elif self.reduce == "max":
                penalty_t = np.max(var_t, axis=-1)
            else:  # mean
                penalty_t = np.mean(var_t, axis=-1)

            total += penalty_t

        return self.weight * total

    def compute_feasibility(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        """
        Feasibility 점수 계산 [0, 1]

        feasibility = exp(-cost / (weight × N))

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)

        Returns:
            scores: (K,) feasibility 점수
        """
        N = controls.shape[1]
        # reference는 미사용이지만 인터페이스 호환
        dummy_ref = np.zeros_like(trajectories[:1, :, :]).squeeze(0)
        costs = self.compute_cost(trajectories, controls, dummy_ref)

        # 정규화: weight * N으로 나눠서 scale-independent
        denom = max(self.weight * N, 1e-10)
        return np.exp(-costs / denom)


class BNNMPPIController(MPPIController):
    """
    BNN Surrogate MPPI: 앙상블 불확실성으로 궤적 feasibility 평가 + 필터링

    UncertaintyMPPI와의 차이:
    - UncertaintyMPPI: σ를 sampling noise에 반영 (탐색 적응)
    - BNN-MPPI: σ를 cost에 반영 + 저신뢰 궤적 필터링 (feasibility)

    Args:
        model: RobotModel 인스턴스
        params: BNNMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
        uncertainty_fn: (states, controls) → (batch, nx) 불확실성
            None이면 model.predict_with_uncertainty 자동 감지
    """

    def __init__(
        self,
        model: RobotModel,
        params: BNNMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        uncertainty_fn: Optional[Callable] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)

        self.bnn_params = params

        # 불확실성 함수 설정
        self.uncertainty_fn = self._resolve_uncertainty_fn(uncertainty_fn, model)

        # FeasibilityCost 생성 (uncertainty_fn이 있을 때만)
        self.feasibility_cost: Optional[FeasibilityCost] = None
        if self.uncertainty_fn is not None:
            self.feasibility_cost = FeasibilityCost(
                uncertainty_fn=self.uncertainty_fn,
                weight=params.feasibility_weight,
                reduce=params.uncertainty_reduce,
            )

        # 통계 히스토리
        self._bnn_history = []

    @staticmethod
    def _resolve_uncertainty_fn(
        uncertainty_fn: Optional[Callable], model: RobotModel
    ) -> Optional[Callable]:
        """불확실성 함수 결정 (명시적 > model 자동 감지 > None)"""
        if uncertainty_fn is not None:
            return uncertainty_fn

        if hasattr(model, "predict_with_uncertainty"):
            def _model_uncertainty(states, controls):
                _, std = model.predict_with_uncertainty(states, controls)
                return std
            return _model_uncertainty

        return None

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        BNN-MPPI 제어 계산

        1. 노이즈 샘플링 + rollout
        2. 기본 비용 계산
        3. FeasibilityCost 계산 + 합산
        4. 저신뢰 궤적 필터링 (feasibility < threshold)
        5. 필터된 비용으로 MPPI 가중치 계산
        6. info에 bnn_stats 추가

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + bnn_stats
        """
        # uncertainty_fn이 없으면 표준 MPPI 폴백
        if self.feasibility_cost is None:
            return super().compute_control(state, reference_trajectory)

        K = self.params.K
        N = self.params.N

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 궤적 rollout (K, N+1, nx)
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 기본 비용 (K,)
        base_costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. Feasibility 비용 + 점수
        feas_costs = self.feasibility_cost.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )
        feasibility_scores = self.feasibility_cost.compute_feasibility(
            sample_trajectories, sampled_controls
        )

        total_costs = base_costs + feas_costs

        # 6. 궤적 필터링
        threshold = self.bnn_params.feasibility_threshold
        num_filtered = 0
        valid_mask = np.ones(K, dtype=bool)

        if threshold > 0:
            valid_mask = feasibility_scores >= threshold
            min_keep = max(int(K * (1 - self.bnn_params.max_filter_ratio)), 1)

            if np.sum(valid_mask) < min_keep:
                # threshold 완화: top-(min_keep)개 선택
                sorted_idx = np.argsort(feasibility_scores)[::-1]
                valid_mask = np.zeros(K, dtype=bool)
                valid_mask[sorted_idx[:min_keep]] = True

            # 필터된 궤적 비용을 큰 값으로
            costs_filtered = total_costs.copy()
            costs_filtered[~valid_mask] = 1e10
            num_filtered = int(np.sum(~valid_mask))
        else:
            costs_filtered = total_costs

        # 7. MPPI 가중치
        weights = self._compute_weights(costs_filtered, self.params.lambda_)

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
        best_idx = np.argmin(costs_filtered)

        bnn_stats = {
            "mean_feasibility": float(np.mean(feasibility_scores)),
            "min_feasibility": float(np.min(feasibility_scores)),
            "max_feasibility": float(np.max(feasibility_scores)),
            "num_filtered": num_filtered,
            "filter_ratio": num_filtered / K,
            "mean_uncertainty_cost": float(np.mean(feas_costs)),
            "mean_base_cost": float(np.mean(base_costs)),
        }
        self._bnn_history.append(bnn_stats)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs_filtered[best_idx]),
            "mean_cost": float(np.mean(total_costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "feasibility_scores": feasibility_scores,
            "bnn_stats": bnn_stats,
        }
        self.last_info = info

        return optimal_control, info

    def get_bnn_statistics(self) -> Dict:
        """누적된 BNN 통계 반환"""
        if not self._bnn_history:
            return {"num_steps": 0}

        feas_means = [h["mean_feasibility"] for h in self._bnn_history]
        filter_ratios = [h["filter_ratio"] for h in self._bnn_history]

        return {
            "num_steps": len(self._bnn_history),
            "overall_mean_feasibility": float(np.mean(feas_means)),
            "overall_min_feasibility": float(
                min(h["min_feasibility"] for h in self._bnn_history)
            ),
            "overall_mean_filter_ratio": float(np.mean(filter_ratios)),
            "history": self._bnn_history,
        }

    def reset(self):
        """상태 초기화"""
        super().reset()
        self._bnn_history = []

    def __repr__(self) -> str:
        return (
            f"BNNMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"weight={self.bnn_params.feasibility_weight}, "
            f"threshold={self.bnn_params.feasibility_threshold})"
        )
