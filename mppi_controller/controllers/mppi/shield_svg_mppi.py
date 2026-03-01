"""
Shield-SVG-MPPI Controller

Shield-MPPI의 안전 보장 + SVG-MPPI의 고품질 샘플링 결합.

SVGMPPIController 상속:
  - Guide particle SVGD 유지 (G guides + SVGD + followers)
  - Shielded rollout 추가: per-step CBF enforcement

동작 원리:
  1. SVG-MPPI 가이드 선택 + SVGD 업데이트
  2. Follower 리샘플
  3. Shielded rollout: 매 스텝 CBF shield 적용
  4. 안전한 궤적에 대해 비용 계산 + 가중 평균
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import SVGMPPIParams
from mppi_controller.controllers.mppi.svg_mppi import SVGMPPIController
from mppi_controller.utils.stein_variational import (
    rbf_kernel_with_bandwidth,
    compute_svgd_update_efficient,
)


@dataclass
class ShieldSVGMPPIParams(SVGMPPIParams):
    """
    Shield-SVG-MPPI 파라미터

    SVGMPPIParams + Shield 파라미터 결합.

    Attributes:
        shield_enabled: Shield 활성화
        shield_cbf_alpha: CBF alpha (0 < alpha <= 1)
        cbf_obstacles: 장애물 리스트 [(x, y, r), ...]
        cbf_safety_margin: 추가 안전 마진 (m)
    """

    shield_enabled: bool = True
    shield_cbf_alpha: float = 0.3
    cbf_obstacles: List[tuple] = field(default_factory=list)
    cbf_safety_margin: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.shield_cbf_alpha <= 1.0, "shield_cbf_alpha must be in (0, 1]"
        assert self.cbf_safety_margin >= 0, "cbf_safety_margin must be non-negative"


class ShieldSVGMPPIController(SVGMPPIController):
    """
    Shield-SVG-MPPI Controller

    SVGMPPIController의 고품질 SVGD 샘플링 +
    Shield-MPPI의 per-step CBF enforcement 결합.

    동작 원리:
        1. K개 초기 샘플 + shielded rollout으로 guide 선택
        2. Guide에 SVGD 적용 (다양성 + 최적화)
        3. Follower 리샘플 (guide 주변)
        4. 전체 결합 후 shielded rollout
        5. MPPI 가중치 계산 (shielded noise 사용, 편향 방지)

    해석적 CBF (Diff Drive Closed-Form):
        h(x) = (x-xo)^2 + (y-yo)^2 - r_eff^2
        Lg_h[0] = 2(x-xo)*cos(theta) + 2(y-yo)*sin(theta)
        접근 시 (Lg_h[0] < 0): v_ceiling = alpha*h / |Lg_h[0]|
        이탈 시 (Lg_h[0] >= 0): 무조건 만족
        omega: 항상 자유 (Lg_h[1] = 0)

    Args:
        model: RobotModel 인스턴스
        params: ShieldSVGMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: ShieldSVGMPPIParams):
        super().__init__(model, params)
        self.shield_svg_params = params
        self._shield_enabled = params.shield_enabled
        self._shield_alpha = params.shield_cbf_alpha

        # Shield 통계
        self.shield_stats_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Shield + SVG-MPPI 제어 계산

        Shield 비활성화 시 기본 SVG-MPPI로 폴백.

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - SVG 통계 + Shield 통계
        """
        if not self._shield_enabled:
            return super().compute_control(state, reference_trajectory)

        K = self.params.K
        G = self.G

        # 1. 초기 샘플 생성 (노이즈 추가)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
        sampled_controls = self.U + noise

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 2. Shielded rollout for initial guide selection
        initial_trajectories, initial_shielded, shield_info = (
            self._shielded_rollout(state, sampled_controls)
        )
        initial_costs = self.cost_function.compute_cost(
            initial_trajectories, initial_shielded, reference_trajectory
        )

        # 3. Guide particle 선택 (최저 비용 G개)
        guide_indices = np.argsort(initial_costs)[:G]
        guide_controls = initial_shielded[guide_indices].copy()  # (G, N, nu)

        # 4. Guide에 SVGD 적용 (O(G^2 D) 복잡도)
        initial_guide_cost = np.mean(initial_costs[guide_indices])
        bandwidth = 1.0  # 기본값 (SVGD 0회 시)

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

        # 5. Guide 최종 비용 (shielded rollout)
        guide_traj, guide_shielded, _ = self._shielded_rollout(
            state, guide_controls
        )
        guide_costs = self.cost_function.compute_cost(
            guide_traj, guide_shielded, reference_trajectory
        )
        final_guide_cost = np.mean(guide_costs)

        # 6. Follower 리샘플링 (guide 주변)
        num_followers = K - G
        follower_controls = np.zeros(
            (num_followers, self.params.N, self.model.control_dim)
        )

        # 각 follower를 임의의 guide 주변에 샘플링
        guide_assignments = np.random.choice(G, size=num_followers)

        for i, g_idx in enumerate(guide_assignments):
            # Guide 주변 Gaussian 샘플링 (작은 분산)
            follower_noise = np.random.normal(
                0,
                self.params.sigma * 0.5,
                (self.params.N, self.model.control_dim),
            )
            follower_controls[i] = guide_shielded[g_idx] + follower_noise

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                follower_controls[i] = np.clip(
                    follower_controls[i], self.u_min, self.u_max
                )

        # 7. 전체 샘플 결합 (Guide + Follower) + shielded rollout
        all_controls = np.concatenate(
            [guide_shielded, follower_controls], axis=0
        )
        all_traj, all_shielded, all_shield_info = self._shielded_rollout(
            state, all_controls
        )
        all_costs = self.cost_function.compute_cost(
            all_traj, all_shielded, reference_trajectory
        )

        # 8. MPPI 가중치 계산
        weights = self._compute_weights(all_costs, self.params.lambda_)

        # 9. ESS 계산
        ess = self._compute_ess(weights)

        # 10. 제어 업데이트 (shielded noise로 편향 방지)
        shielded_noise = all_shielded - self.U
        weighted_noise = np.sum(
            weights[:, None, None] * shielded_noise, axis=0
        )
        self.U = self.U + weighted_noise

        # 11. Receding horizon 시프트
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 12. 최적 제어
        optimal_control = self.U[0, :]

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            optimal_control = np.clip(optimal_control, self.u_min, self.u_max)

        # 13. SVG 통계 저장
        svg_stats = {
            "num_guides": G,
            "num_followers": num_followers,
            "svgd_iterations": self.svgd_iterations,
            "guide_step_size": self.svg_step_size,
            "initial_guide_cost": initial_guide_cost,
            "final_guide_cost": final_guide_cost,
            "guide_cost_improvement": initial_guide_cost - final_guide_cost,
            "guide_mean_cost": final_guide_cost,
            "follower_mean_cost": float(np.mean(all_costs[G:])) if num_followers > 0 else 0.0,
            "bandwidth": bandwidth,
        }
        self.svg_stats_history.append(svg_stats)

        # Shield 통계 저장
        self.shield_stats_history.append(all_shield_info)

        # 14. info 구성
        best_idx = np.argmin(all_costs)
        info = {
            "sample_trajectories": all_traj,
            "sample_controls": all_shielded,
            "sample_weights": weights,
            "guide_indices": guide_indices,
            "guide_controls": guide_shielded,
            "best_trajectory": all_traj[best_idx],
            "best_cost": all_costs[best_idx],
            "mean_cost": np.mean(all_costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "svg_stats": svg_stats,
            "shield_info": all_shield_info,
        }

        return optimal_control, info

    def _shielded_rollout(
        self, initial_state: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Shielded rollout: 매 timestep CBF shield 적용

        Shield-MPPI의 _shielded_rollout() 패턴 동일.

        Args:
            initial_state: (nx,) 초기 상태
            controls: (K, N, nu) 원본 제어 시퀀스

        Returns:
            trajectories: (K, N+1, nx) 안전한 궤적
            shielded_controls: (K, N, nu) CBF 적용된 제어
            info: dict - shield 통계
        """
        K, N, nu = controls.shape
        nx = self.model.state_dim

        trajectories = np.zeros((K, N + 1, nx))
        shielded_controls = controls.copy()
        trajectories[:, 0, :] = initial_state

        total_interventions = 0
        total_vel_reduction = 0.0
        total_steps = K * N

        for t in range(N):
            states_t = trajectories[:, t, :]  # (K, nx)
            controls_t = controls[:, t, :]  # (K, nu)

            # CBF shield 적용
            safe_controls_t, intervened, vel_reduction = (
                self._cbf_shield_batch(states_t, controls_t)
            )

            shielded_controls[:, t, :] = safe_controls_t
            total_interventions += np.sum(intervened)
            total_vel_reduction += np.sum(vel_reduction)

            # 안전한 제어로 다음 상태 전파
            trajectories[:, t + 1, :] = self.model.step(
                states_t, safe_controls_t, self.params.dt
            )

        info = {
            "intervention_rate": float(total_interventions / total_steps) if total_steps > 0 else 0.0,
            "mean_vel_reduction": float(total_vel_reduction / max(total_interventions, 1)),
            "total_interventions": int(total_interventions),
            "total_steps": total_steps,
        }

        return trajectories, shielded_controls, info

    def _cbf_shield_batch(
        self, states: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        해석적 CBF shield (완전 벡터화)

        Differential drive closed-form:
            h(x) = (x-xo)^2 + (y-yo)^2 - r_eff^2
            Lg_h[0] = 2(x-xo)*cos(theta) + 2(y-yo)*sin(theta)

            Lg_h[0] < 0 (접근): v_ceiling = alpha*h / |Lg_h[0]|
            Lg_h[0] >= 0 (이탈): 무조건 만족
            omega: 항상 자유

        Args:
            states: (K, nx) 현재 상태 [x, y, theta]
            controls: (K, nu) 원본 제어 [v, omega]

        Returns:
            safe_controls: (K, nu) CBF 적용된 제어
            intervened: (K,) bool, 개입 여부
            vel_reduction: (K,) 속도 감소량
        """
        K = states.shape[0]
        safe_controls = controls.copy()
        v_original = controls[:, 0]  # (K,)

        x = states[:, 0]  # (K,)
        y = states[:, 1]  # (K,)
        theta = states[:, 2]  # (K,)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        obstacles = self.shield_svg_params.cbf_obstacles
        safety_margin = self.shield_svg_params.cbf_safety_margin
        alpha = self._shield_alpha

        # 모든 장애물에 대해 v_ceiling 계산 후 최보수적(최소) 적용
        v_ceiling = np.full(K, np.inf)

        for obs_x, obs_y, obs_r in obstacles:
            effective_r = obs_r + safety_margin

            dx = x - obs_x  # (K,)
            dy = y - obs_y  # (K,)

            # Barrier value
            h = dx**2 + dy**2 - effective_r**2  # (K,)

            # Lg_h[0] = 2*dx*cos(theta) + 2*dy*sin(theta)
            Lg_h_v = 2.0 * dx * cos_theta + 2.0 * dy * sin_theta  # (K,)

            # CBF 제약: Lg_h_v * v + alpha * h >= 0
            # 접근 시 (Lg_h_v < 0): v <= alpha * h / |Lg_h_v|
            # 이탈 시 (Lg_h_v >= 0): 무조건 만족 -> ceiling = inf
            approaching = Lg_h_v < -1e-10  # 수치 안정성

            v_ceiling_obs = np.where(
                approaching,
                alpha * h / np.maximum(np.abs(Lg_h_v), 1e-10),
                np.inf,
            )

            v_ceiling = np.minimum(v_ceiling, v_ceiling_obs)

        # 속도 클리핑: v_safe = min(v_original, v_ceiling)
        v_safe = np.minimum(v_original, v_ceiling)

        # 제어 제약 하한 적용
        if self.u_min is not None:
            v_safe = np.maximum(v_safe, self.u_min[0])

        safe_controls[:, 0] = v_safe

        # 개입 여부 및 속도 감소량
        intervened = v_safe < v_original - 1e-10
        vel_reduction = np.where(intervened, v_original - v_safe, 0.0)

        return safe_controls, intervened, vel_reduction

    def update_obstacles(self, obstacles: List[tuple]):
        """
        동적 장애물 업데이트

        Args:
            obstacles: [(x, y, radius), ...] 장애물 리스트
        """
        self.shield_svg_params.cbf_obstacles = obstacles

    def set_shield_enabled(self, enabled: bool):
        """Shield 활성화/비활성화"""
        self._shield_enabled = enabled

    def get_shield_statistics(self) -> Dict:
        """
        Shield 통계 반환

        Returns:
            dict:
                - mean_intervention_rate: float 평균 개입률
                - total_interventions: int 총 개입 횟수
                - num_steps: int 기록된 스텝 수
        """
        if not self.shield_stats_history:
            return {
                "mean_intervention_rate": 0.0,
                "total_interventions": 0,
                "num_steps": 0,
            }

        rates = [s["intervention_rate"] for s in self.shield_stats_history]
        total_int = sum(
            s["total_interventions"] for s in self.shield_stats_history
        )

        return {
            "mean_intervention_rate": float(np.mean(rates)),
            "total_interventions": total_int,
            "num_steps": len(self.shield_stats_history),
        }

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.shield_stats_history = []

    def __repr__(self) -> str:
        return (
            f"ShieldSVGMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"G={self.G}, K={self.params.K}, "
            f"shield_alpha={self._shield_alpha})"
        )
