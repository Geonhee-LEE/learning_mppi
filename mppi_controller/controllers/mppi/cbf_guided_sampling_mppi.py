"""
CBF-Guided Sampling MPPI Controller

CBF 그래디언트 기반 거부 샘플링 + 편향 노이즈.
위반 궤적을 리샘플하여 안전한 궤적의 비율을 높임.

핵심 아이디어:
  1. 기본 노이즈 샘플링 (K개)
  2. rollout -> CBF 위반 궤적 식별
  3. 거부 샘플링: 위반 궤적을 grad(h) 방향 편향 노이즈로 리샘플
  4. 재 rollout -> 비용 계산 -> 가중 평균
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import CBFMPPIParams
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


@dataclass
class CBFGuidedSamplingParams(CBFMPPIParams):
    """
    CBF-Guided Sampling MPPI 파라미터

    Attributes:
        rejection_ratio: 리샘플할 위반 궤적 최대 비율
        gradient_bias_weight: grad(h) 방향 편향 강도
        max_resample_iters: 최대 리샘플 반복 횟수
    """

    rejection_ratio: float = 0.3
    gradient_bias_weight: float = 0.1
    max_resample_iters: int = 3

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.rejection_ratio <= 1.0, "rejection_ratio must be in (0, 1]"
        assert self.gradient_bias_weight >= 0, "gradient_bias_weight must be non-negative"
        assert self.max_resample_iters >= 1, "max_resample_iters must be >= 1"


class CBFGuidedSamplingMPPIController(CBFMPPIController):
    """
    CBF-Guided Sampling MPPI Controller

    기존 CBFMPPIController를 확장하여 샘플 품질을 개선:
    - 위험 궤적(h<0) 식별 -> grad(h) 방향으로 편향된 노이즈로 리샘플
    - 최대 max_resample_iters 반복
    - 리샘플 후에도 남은 위반 궤적은 CBF 비용 페널티로 처리 (기존 방식)
    """

    def __init__(
        self,
        model: RobotModel,
        params: CBFGuidedSamplingParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.guided_params = params
        self.resample_stats = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        CBF-Guided Sampling MPPI 제어 계산

        1. 기본 노이즈 샘플링
        2. rollout + CBF 위반 식별
        3. 위반 궤적 리샘플 (grad(h) 편향)
        4. 최종 rollout + 비용 + 가중치
        5. 제어 업데이트
        """
        K = self.params.K
        N = self.params.N

        # 1. 기본 노이즈 샘플링
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 2. 초기 rollout + 위반 식별 + 리샘플 루프
        total_resampled = 0
        for resample_iter in range(self.guided_params.max_resample_iters):
            trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

            # CBF 위반 여부 확인 (h < 0인 궤적)
            violated_mask = self._identify_violations(trajectories)  # (K,) bool
            num_violated = np.sum(violated_mask)

            if num_violated == 0:
                break

            # 리샘플 대상: violated 중 최대 rejection_ratio * K 개
            max_resample = int(self.guided_params.rejection_ratio * K)
            num_to_resample = min(num_violated, max_resample)

            if num_to_resample == 0:
                break

            # 위반 인덱스 (랜덤 샘플링으로 편향 방지)
            all_violated_idx = np.where(violated_mask)[0]
            violated_indices = np.random.choice(
                all_violated_idx, size=num_to_resample, replace=False
            )

            # 3. grad(h) 편향 노이즈로 리샘플
            biased_controls = self._resample_with_gradient_bias(
                state, sampled_controls[violated_indices],
                trajectories[violated_indices]
            )
            sampled_controls[violated_indices] = biased_controls
            total_resampled += num_to_resample

        # 4. 최종 rollout
        trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
        costs = self.cost_function.compute_cost(
            trajectories, sampled_controls, reference_trajectory
        )

        # 5. MPPI 가중치 계산
        weights = self._compute_weights(costs, self.params.lambda_)

        # 6. 제어 업데이트
        all_noise = sampled_controls - self.U
        weighted_noise = np.sum(weights[:, None, None] * all_noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # Receding horizon 시프트
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        optimal_control = self.U[0, :]

        # Layer 2: 안전 필터 (optional, from CBFMPPIController)
        filter_info = {"filtered": False, "correction_norm": 0.0}
        if self.safety_filter is not None:
            optimal_control, filter_info = self.safety_filter.filter_control(
                state, optimal_control, self.u_min, self.u_max
            )

        # Barrier info
        best_idx = np.argmin(costs)
        best_traj = trajectories[best_idx]
        barrier_info = self.cbf_cost.get_barrier_info(best_traj)

        # ESS (from base MPPIController)
        ess = self._compute_ess(weights)

        # 리샘플 통계
        final_violated = np.sum(self._identify_violations(trajectories))
        resample_stat = {
            "total_resampled": total_resampled,
            "final_violated": int(final_violated),
            "safe_ratio": float(1.0 - final_violated / K),
        }
        self.resample_stats.append(resample_stat)

        # CBF 통계 저장
        cbf_stats = {
            "min_barrier": barrier_info["min_barrier"],
            "is_safe": barrier_info["is_safe"],
            "filtered": filter_info["filtered"],
            "correction_norm": filter_info.get("correction_norm", 0.0),
        }
        self.cbf_stats_history.append(cbf_stats)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": best_traj,
            "best_cost": costs[best_idx],
            "mean_cost": np.mean(costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "barrier_values": barrier_info["barrier_values"],
            "min_barrier": barrier_info["min_barrier"],
            "is_safe": barrier_info["is_safe"],
            "cbf_filtered": filter_info["filtered"],
            "cbf_correction_norm": filter_info.get("correction_norm", 0.0),
            "resample_stats": resample_stat,
        }
        self.last_info = info

        return optimal_control, info

    def _identify_violations(self, trajectories: np.ndarray) -> np.ndarray:
        """
        CBF 위반 궤적 식별 (h < 0)

        Args:
            trajectories: (K, N+1, nx)

        Returns:
            violated: (K,) bool
        """
        K = trajectories.shape[0]
        positions = trajectories[:, :, :2]
        violated = np.zeros(K, dtype=bool)

        for obs_x, obs_y, obs_r in self.cbf_params.cbf_obstacles:
            effective_r = obs_r + self.cbf_params.cbf_safety_margin
            dx = positions[:, :, 0] - obs_x
            dy = positions[:, :, 1] - obs_y
            h = dx**2 + dy**2 - effective_r**2
            violated |= np.any(h < 0, axis=1)

        return violated

    def _resample_with_gradient_bias(
        self,
        state: np.ndarray,
        violated_controls: np.ndarray,
        violated_trajectories: np.ndarray,
    ) -> np.ndarray:
        """
        grad(h) 방향 편향 노이즈로 리샘플

        h(x) = ||p - p_obs||^2 - r_eff^2 이므로
        grad_p h = 2 * (p - p_obs) — 장애물에서 멀어지는 방향.

        각 위반 궤적에 대해:
        1. 최소 h인 시간스텝 t* 찾기
        2. t*에서 grad(h) 방향 계산
        3. 새 노이즈 + grad(h) 편향으로 리샘플

        Args:
            state: (nx,) current state
            violated_controls: (M, N, nu) 위반 제어 시퀀스
            violated_trajectories: (M, N+1, nx) 위반 궤적

        Returns:
            new_controls: (M, N, nu) 편향된 새 제어 시퀀스
        """
        M, N, nu = violated_controls.shape
        new_controls = violated_controls.copy()

        # 각 궤적의 최소 h 시간스텝에서의 grad(h) 계산
        positions = violated_trajectories[:, :, :2]  # (M, N+1, 2)

        for obs_x, obs_y, obs_r in self.cbf_params.cbf_obstacles:
            effective_r = obs_r + self.cbf_params.cbf_safety_margin
            dx = positions[:, :, 0] - obs_x  # (M, N+1)
            dy = positions[:, :, 1] - obs_y  # (M, N+1)
            h = dx**2 + dy**2 - effective_r**2  # (M, N+1)

            # 최소 h인 시간스텝 찾기
            min_t = np.argmin(h, axis=1)  # (M,)

            # 해당 시간스텝에서의 grad(h) 방향 (장애물에서 멀어지는 방향)
            # grad_p h = 2 * [dx, dy]
            for m in range(M):
                t = min(min_t[m], N - 1)  # control index (clamp)
                grad_x = 2.0 * dx[m, min_t[m]]
                grad_y = 2.0 * dy[m, min_t[m]]
                grad_norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-10

                # 정규화된 grad(h) 방향 (위치 공간)
                grad_dir = np.array([grad_x / grad_norm, grad_y / grad_norm])

                # 위치 공간 그래디언트를 제어 공간으로 투영
                # diff-drive: v는 [cos(θ), sin(θ)] 방향으로 이동
                # 궤적에서 현재 heading 추출
                theta_m = violated_trajectories[m, min_t[m], 2]
                heading = np.array([np.cos(theta_m), np.sin(theta_m)])

                # v 편향: grad(h)를 heading에 투영 (양이면 전진 = 멀어짐)
                v_bias = self.guided_params.gradient_bias_weight * np.dot(
                    grad_dir, heading
                )

                # ω 편향: heading과 grad(h) 간의 cross product → 회전 방향
                cross = heading[0] * grad_dir[1] - heading[1] * grad_dir[0]
                omega_bias = self.guided_params.gradient_bias_weight * cross

                # 기존 제어에 편향 혼합 (50% 기존 + 50% 새 노이즈)
                fresh_noise = np.random.normal(0, self.params.sigma, (N, nu))
                new_controls[m] = 0.5 * new_controls[m] + 0.5 * (self.U + fresh_noise)

                # t 이후 시간스텝에 grad(h) 편향 적용
                new_controls[m, t:, 0] += v_bias
                if nu > 1:
                    new_controls[m, t:, 1] += omega_bias

        if self.u_min is not None and self.u_max is not None:
            new_controls = np.clip(new_controls, self.u_min, self.u_max)

        return new_controls

    def get_resample_statistics(self) -> Dict:
        """리샘플 통계 반환"""
        if not self.resample_stats:
            return {"mean_resampled": 0, "mean_safe_ratio": 1.0}
        return {
            "mean_resampled": np.mean(
                [s["total_resampled"] for s in self.resample_stats]
            ),
            "mean_safe_ratio": np.mean(
                [s["safe_ratio"] for s in self.resample_stats]
            ),
            "mean_final_violated": np.mean(
                [s["final_violated"] for s in self.resample_stats]
            ),
        }

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.resample_stats = []

    def __repr__(self) -> str:
        return (
            f"CBFGuidedSamplingMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"rejection_ratio={self.guided_params.rejection_ratio}, "
            f"bias_weight={self.guided_params.gradient_bias_weight})"
        )
