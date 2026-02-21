"""
Stein Variational MPPI Controller

SVGD로 샘플 다양성을 강화하는 MPPI 변형.
"""

import numpy as np
from typing import Dict, Tuple
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import SteinVariationalMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.utils.stein_variational import (
    rbf_kernel,
    compute_svgd_update_efficient,
    median_bandwidth,
    rbf_kernel_with_bandwidth,
)


class SteinVariationalMPPIController(MPPIController):
    """
    Stein Variational MPPI Controller (SVMPC)

    SVGD (Stein Variational Gradient Descent)로 샘플 다양성을 강화하는 MPPI 변형.

    동작 원리:
        1. Vanilla MPPI로 초기 샘플링
        2. 비용 계산
        3. SVGD 반복:
           - RBF 커널로 샘플 간 상호작용
           - Φ(x) = (1/K) Σ [K(x_i,x_j)∇log p(x_j) + ∇K(x_i,x_j)]
           - x ← x + ε * Φ(x)
        4. 업데이트된 샘플로 MPPI 가중치 재계산
        5. 가중 평균 제어

    SVGD 핵심 아이디어:
        - RBF 커널: 유사한 샘플끼리 밀어냄 (다양성)
        - Gradient term: 좋은 방향으로 끌어당김 (최적화)
        - 균형: 탐색 + 활용

    장점:
        - ✅ 샘플 효율 2배 향상 (K=512 SVMPC ≈ K=1024 Vanilla)
        - ✅ 모드 커버리지 개선 (multimodal 문제)
        - ✅ 수렴 속도 향상

    단점:
        - ⚠️ 계산 복잡도 O(K²) (RBF kernel)
        - ⚠️ 메모리 사용량 증가

    적용 사례:
        - 복잡한 비용 함수 (local minima 다수)
        - 제약이 많은 환경
        - 높은 품질 요구

    참고 논문:
        - Lambert et al. (2020) - "Stein Variational MPC"
        - Liu & Wang (2016) - "Stein Variational Gradient Descent"

    Args:
        model: RobotModel 인스턴스
        params: SteinVariationalMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: SteinVariationalMPPIParams):
        # 부모 클래스 초기화
        super().__init__(model, params)

        # SVGD 파라미터
        self.svgd_params = params
        self.svgd_num_iterations = params.svgd_num_iterations
        self.svgd_step_size = params.svgd_step_size

        # 통계 (디버깅용)
        self.svgd_stats_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        SVMPC 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K

        # 1. 초기 샘플링 (Vanilla MPPI)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
        sampled_controls = self.U + noise  # (K, N, nu)

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 2. 초기 비용 계산
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
        initial_costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 3. SVGD 반복
        svgd_costs_history = [initial_costs.copy()]

        for iteration in range(self.svgd_num_iterations):
            # 3.1. RBF 커널 + bandwidth 동시 계산 (K² 거리 1회만)
            kernel, bandwidth = rbf_kernel_with_bandwidth(sampled_controls)

            # 3.2. 비용 gradient 추정 (SPSA)
            grad_costs = self._estimate_cost_gradient(
                state, sampled_controls, reference_trajectory
            )

            # grad log p(x) = -grad cost(x) (비용 최소화)
            grad_log_prob = -grad_costs

            # 3.3. SVGD 업데이트 (메모리 효율적 — (K,K,N,nu) 텐서 없음)
            phi = compute_svgd_update_efficient(
                sampled_controls, grad_log_prob, kernel, bandwidth
            )

            # 3.5. 제어 업데이트
            sampled_controls = sampled_controls + self.svgd_step_size * phi

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

            # 3.6. 비용 재계산
            sample_trajectories = self.dynamics_wrapper.rollout(
                state, sampled_controls
            )
            costs = self.cost_function.compute_cost(
                sample_trajectories, sampled_controls, reference_trajectory
            )
            svgd_costs_history.append(costs.copy())

        # 4. MPPI 가중치 (최종 비용으로)
        weights = self._compute_weights(costs, self.params.lambda_)

        # 5. ESS 계산
        ess = 1.0 / np.sum(weights**2)

        # 6. 제어 업데이트
        # 가중 평균: u* = Σ w_k * u_k
        weighted_controls = np.sum(
            weights[:, None, None] * sampled_controls, axis=0
        )  # (N, nu)
        self.U = weighted_controls

        # 7. Receding horizon
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 8. 최적 제어
        optimal_control = self.U[0, :]

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            optimal_control = np.clip(optimal_control, self.u_min, self.u_max)

        # 9. SVGD 통계 저장
        cost_improvement = initial_costs.mean() - costs.mean()
        svgd_stats = {
            "svgd_iterations": self.svgd_num_iterations,
            "initial_mean_cost": initial_costs.mean(),
            "final_mean_cost": costs.mean(),
            "cost_improvement": cost_improvement,
            "bandwidth": bandwidth,
        }
        self.svgd_stats_history.append(svgd_stats)

        # 10. 정보 저장
        info = {
            "sample_trajectories": sample_trajectories,
            "sample_controls": sampled_controls,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[np.argmax(weights)],
            "best_cost": np.min(costs),
            "mean_cost": np.mean(costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "svgd_stats": svgd_stats,
            "svgd_costs_history": svgd_costs_history,
        }

        return optimal_control, info

    def _estimate_cost_gradient(
        self,
        state: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
        epsilon: float = 1e-3,
    ) -> np.ndarray:
        """
        SPSA (Simultaneous Perturbation Stochastic Approximation)로 비용 gradient 추정

        전 차원을 동시 섭동하여 2회 rollout만으로 gradient를 추정.
        기존 dimension-wise finite difference (N×nu회 rollout) 대비 ~30배 빠름.

        ∇C(u) ≈ [C(u + ε·Δ) - C(u - ε·Δ)] / (2ε·Δ)

        Args:
            state: (nx,) 현재 상태
            controls: (K, N, nu) 제어 시퀀스
            reference_trajectory: (N+1, nx) 레퍼런스
            epsilon: Perturbation step

        Returns:
            grad_costs: (K, N, nu) 비용 gradient
        """
        K, N, nu = controls.shape

        # Rademacher 랜덤 방향 (±1)
        delta = np.random.choice([-1.0, 1.0], size=(K, N, nu))

        # 양방향 섭동
        controls_plus = controls + epsilon * delta
        controls_minus = controls - epsilon * delta

        if self.u_min is not None and self.u_max is not None:
            controls_plus = np.clip(controls_plus, self.u_min, self.u_max)
            controls_minus = np.clip(controls_minus, self.u_min, self.u_max)

        # 2회 rollout (기존 N×nu=60회 → 2회)
        traj_plus = self.dynamics_wrapper.rollout(state, controls_plus)
        traj_minus = self.dynamics_wrapper.rollout(state, controls_minus)

        cost_plus = self.cost_function.compute_cost(
            traj_plus, controls_plus, reference_trajectory
        )
        cost_minus = self.cost_function.compute_cost(
            traj_minus, controls_minus, reference_trajectory
        )

        # SPSA gradient: g_i = (C+ - C-) / (2·ε·Δ_i)
        grad_costs = (
            (cost_plus - cost_minus)[:, None, None] / (2.0 * epsilon * delta)
        )

        return grad_costs

    def get_svgd_statistics(self) -> Dict:
        """
        SVGD 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_cost_improvement: float 평균 비용 개선
                - mean_bandwidth: float 평균 bandwidth
                - svgd_stats_history: List[dict] 통계 히스토리
        """
        if len(self.svgd_stats_history) == 0:
            return {
                "mean_cost_improvement": 0.0,
                "mean_bandwidth": 0.0,
                "svgd_stats_history": [],
            }

        cost_improvements = [s["cost_improvement"] for s in self.svgd_stats_history]
        bandwidths = [s["bandwidth"] for s in self.svgd_stats_history]

        return {
            "mean_cost_improvement": np.mean(cost_improvements),
            "mean_bandwidth": np.mean(bandwidths),
            "svgd_stats_history": self.svgd_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.svgd_stats_history = []

    def __repr__(self) -> str:
        return (
            f"SteinVariationalMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"svgd_iterations={self.svgd_num_iterations}, "
            f"params={self.params})"
        )
