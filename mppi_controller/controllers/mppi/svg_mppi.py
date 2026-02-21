"""
SVG-MPPI Controller (Stochastic Value Gradient MPPI)

Guide particle 기반 효율적인 Stein Variational MPPI.
"""

import numpy as np
from typing import Dict, Tuple
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import SVGMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.utils.stein_variational import (
    rbf_kernel_with_bandwidth,
    compute_svgd_update_efficient,
)


class SVGMPPIController(MPPIController):
    """
    SVG-MPPI Controller

    Guide particle 기반 Stein Variational MPPI로 SVMPC의 고속 근사.

    동작 원리:
        1. K개 샘플 초기화
        2. G개 guide particle 선택 (최저 비용)
        3. Guide에 SVGD 적용 (O(G²D) 복잡도)
        4. (K-G)개 follower를 guide 주변 재샘플링
        5. MPPI 가중치 계산 및 업데이트

    SVMPC 대비 장점:
        - 계산량: O(K²D) → O(G²D), G << K
        - 고속화: SVGD를 G개만 적용
        - 다양성 유지: Follower가 guide 분포 따라감
        - 실시간성: G=64, K=1024에서 ~5배 속도 향상

    참고 논문:
        - Kondo et al. (2024) - "SVG-MPPI: Efficient Stein Variational Guidance for MPPI"

    Args:
        model: RobotModel 인스턴스
        params: SVGMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: SVGMPPIParams):
        super().__init__(model, params)
        self.svg_params = params

        # Guide particle 파라미터
        self.G = params.svg_num_guide_particles
        self.svg_step_size = params.svg_guide_step_size
        self.svgd_iterations = params.svgd_num_iterations

        # 통계 (디버깅용)
        self.svg_stats_history = []

        # 검증
        if self.G >= self.params.K:
            raise ValueError(
                f"svg_num_guide_particles ({self.G}) must be < K ({self.params.K})"
            )

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        SVG-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K
        G = self.G

        # 1. 초기 샘플 생성 (노이즈 추가)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
        sampled_controls = self.U + noise

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 2. 초기 롤아웃 및 비용 계산 (guide 선택용)
        initial_trajectories = self.dynamics_wrapper.rollout(
            state, sampled_controls
        )
        initial_costs = self.cost_function.compute_cost(
            initial_trajectories, sampled_controls, reference_trajectory
        )

        # 3. **Guide particle 선택** (최저 비용 G개)
        guide_indices = np.argsort(initial_costs)[:G]
        guide_controls = sampled_controls[guide_indices].copy()  # (G, N, nu)

        # 4. **Guide에 SVGD 적용** (O(G²D) 복잡도)
        initial_guide_cost = np.mean(initial_costs[guide_indices])

        for iteration in range(self.svgd_iterations):
            # SVGD 파라미터
            kernel, bandwidth = rbf_kernel_with_bandwidth(guide_controls)

            # 비용 그래디언트 추정
            grad_costs = self._estimate_cost_gradient(
                state, guide_controls, reference_trajectory
            )

            # log probability gradient (negative cost gradient)
            grad_log_prob = -grad_costs

            # SVGD 업데이트 (메모리 효율적)
            phi = compute_svgd_update_efficient(
                guide_controls, grad_log_prob, kernel, bandwidth
            )

            # Guide 업데이트
            guide_controls = guide_controls + self.svg_step_size * phi

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                guide_controls = np.clip(guide_controls, self.u_min, self.u_max)

        # 5. Guide 최종 비용
        guide_trajectories = self.dynamics_wrapper.rollout(state, guide_controls)
        guide_costs = self.cost_function.compute_cost(
            guide_trajectories, guide_controls, reference_trajectory
        )
        final_guide_cost = np.mean(guide_costs)

        # 6. **Follower 재샘플링** (guide 주변)
        # Gaussian mixture: 각 guide 주변에 follower 분포
        num_followers = K - G
        follower_controls = np.zeros((num_followers, self.params.N, self.model.control_dim))

        # 각 follower를 임의의 guide 주변에 샘플링
        guide_assignments = np.random.choice(G, size=num_followers)

        for i, guide_idx in enumerate(guide_assignments):
            # Guide 주변 Gaussian 샘플링
            follower_noise = np.random.normal(
                0,
                self.params.sigma * 0.5,  # Guide 주변은 작은 분산
                (self.params.N, self.model.control_dim),
            )
            follower_controls[i] = guide_controls[guide_idx] + follower_noise

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                follower_controls[i] = np.clip(
                    follower_controls[i], self.u_min, self.u_max
                )

        # 7. **전체 샘플 결합** (Guide + Follower)
        all_controls = np.concatenate([guide_controls, follower_controls], axis=0)

        # 8. 최종 롤아웃 및 비용
        all_trajectories = self.dynamics_wrapper.rollout(state, all_controls)
        all_costs = self.cost_function.compute_cost(
            all_trajectories, all_controls, reference_trajectory
        )

        # 9. MPPI 가중치
        weights = self._compute_weights(all_costs, self.params.lambda_)

        # 10. ESS 계산
        ess = 1.0 / np.sum(weights**2)

        # 11. 제어 업데이트 (가중 평균 노이즈)
        all_noise = all_controls - self.U
        weighted_noise = np.sum(weights[:, None, None] * all_noise, axis=0)
        self.U = self.U + weighted_noise

        # 12. Receding horizon
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 13. 최적 제어
        optimal_control = self.U[0, :]

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            optimal_control = np.clip(optimal_control, self.u_min, self.u_max)

        # 14. SVG 통계 저장
        svg_stats = {
            "num_guides": G,
            "num_followers": num_followers,
            "svgd_iterations": self.svgd_iterations,
            "guide_step_size": self.svg_step_size,
            "initial_guide_cost": initial_guide_cost,
            "final_guide_cost": final_guide_cost,
            "guide_cost_improvement": initial_guide_cost - final_guide_cost,
            "guide_mean_cost": final_guide_cost,
            "follower_mean_cost": np.mean(all_costs[G:]),
            "bandwidth": bandwidth,
        }
        self.svg_stats_history.append(svg_stats)

        # 15. 정보 저장
        info = {
            "sample_trajectories": all_trajectories,
            "sample_controls": all_controls,
            "sample_weights": weights,
            "guide_indices": guide_indices,
            "guide_controls": guide_controls,
            "best_trajectory": all_trajectories[np.argmax(weights)],
            "best_cost": np.min(all_costs),
            "mean_cost": np.mean(all_costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "svg_stats": svg_stats,
        }

        return optimal_control, info

    def _estimate_cost_gradient(
        self,
        state: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """
        유한 차분으로 비용 그래디언트 추정

        Args:
            state: (nx,) 현재 상태
            controls: (K, N, nu) 제어 시퀀스
            reference_trajectory: (N+1, nx) 레퍼런스
            epsilon: 유한 차분 스텝

        Returns:
            grad_costs: (K, N, nu) 비용 그래디언트
        """
        K, N, nu = controls.shape
        grad_costs = np.zeros((K, N, nu))

        # 각 타임스텝, 각 제어 차원에 대해 유한 차분
        for t in range(N):
            for u_dim in range(nu):
                # 양방향 유한 차분
                controls_plus = controls.copy()
                controls_plus[:, t, u_dim] += epsilon

                controls_minus = controls.copy()
                controls_minus[:, t, u_dim] -= epsilon

                # 롤아웃
                traj_plus = self.dynamics_wrapper.rollout(state, controls_plus)
                traj_minus = self.dynamics_wrapper.rollout(state, controls_minus)

                # 비용
                cost_plus = self.cost_function.compute_cost(
                    traj_plus, controls_plus, reference_trajectory
                )
                cost_minus = self.cost_function.compute_cost(
                    traj_minus, controls_minus, reference_trajectory
                )

                # 그래디언트
                grad_costs[:, t, u_dim] = (cost_plus - cost_minus) / (2 * epsilon)

        return grad_costs

    def get_svg_statistics(self) -> Dict:
        """
        SVG 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_guide_cost_improvement: float 평균 guide 비용 개선
                - mean_bandwidth: float 평균 bandwidth
                - svg_stats_history: List[dict] 통계 히스토리
        """
        if len(self.svg_stats_history) == 0:
            return {
                "mean_guide_cost_improvement": 0.0,
                "mean_bandwidth": 0.0,
                "svg_stats_history": [],
            }

        improvements = [s["guide_cost_improvement"] for s in self.svg_stats_history]
        bandwidths = [s["bandwidth"] for s in self.svg_stats_history]

        return {
            "mean_guide_cost_improvement": np.mean(improvements),
            "mean_bandwidth": np.mean(bandwidths),
            "svg_stats_history": self.svg_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 초기화"""
        super().reset()
        self.svg_stats_history = []

    def __repr__(self) -> str:
        return (
            f"SVGMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"G={self.G}, K={self.params.K}, "
            f"params={self.params})"
        )
