"""
Robust MPPI Controller (R-MPPI)

피드백을 MPPI 샘플링 루프 내부에 통합하여, 외란 하에서도
추적 가능한 제어를 학습. Tube-MPPI의 분리된 2계층(MPPI→피드백) 대신
MPPI가 외란 보상 능력을 직접 인지.

핵심 수식:
    x_nom(t+1) = F(x_nom(t), v(t))                          # 명목 (외란 없음)
    x_real(t+1) = F(x_real(t), v(t) + K·(x_real-x_nom)) + w # 실제 (피드백+외란)
    cost_k = Σ_t q(x_real_k(t), ref(t))                      # 실제 궤적으로 비용

Tube-MPPI vs R-MPPI:
    - Tube: MPPI → 피드백 (분리, MPPI는 피드백 능력 미인지, 보수적)
    - R-MPPI: MPPI 내부에 피드백 통합 (외란 보상 인지, 덜 보수적)

Reference: Gandhi et al., RAL 2021, arXiv:2102.09027
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import RobustMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.controllers.mppi.ancillary_controller import (
    AncillaryController,
    create_default_ancillary_controller,
)


class RobustMPPIController(MPPIController):
    """
    Robust MPPI Controller (R-MPPI)

    각 샘플 k에 대해 명목 + 실제 궤적을 동시 rollout하여,
    MPPI가 외란 하에서도 추적 가능한 제어를 학습.

    Vanilla MPPI 대비 핵심 차이:
        1. 명목 rollout: x_nom — BatchDynamicsWrapper.rollout()
        2. 실제 rollout: x_real — 피드백 + 외란 포함, step-by-step
        3. 비용은 실제 궤적(x_real)으로 계산 — 핵심!
        4. 3가지 외란 모드: gaussian, adversarial, none

    Args:
        model: RobotModel 인스턴스
        params: RobustMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
        ancillary_controller: AncillaryController (None이면 기본값 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: RobustMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        ancillary_controller: Optional[AncillaryController] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.robust_params = params

        # Ancillary controller 설정
        if ancillary_controller is not None:
            self.ancillary_controller = ancillary_controller
        else:
            self.ancillary_controller = create_default_ancillary_controller(
                model.model_type, params.feedback_gain_scale
            )

        # 외란 표준편차 (nx,)
        self._disturbance_std = np.array(params.disturbance_std)
        # 상태 차원과 외란 차원 맞추기
        nx = model.state_dim
        if len(self._disturbance_std) < nx:
            self._disturbance_std = np.concatenate([
                self._disturbance_std,
                np.zeros(nx - len(self._disturbance_std)),
            ])
        elif len(self._disturbance_std) > nx:
            self._disturbance_std = self._disturbance_std[:nx]

        # 통계 히스토리
        self._robust_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        R-MPPI 제어 계산

        1. 노이즈 샘플링 (K, N, nu)
        2. 명목 rollout: x_nom (K, N+1, nx)
        3. 실제 rollout (피드백 + 외란): x_real (K, N+1, nx)
        4. 비용: cost_function(x_real, controls, reference)
        5. MPPI 가중치 + 업데이트
        6. Receding horizon shift

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K
        N = self.params.N
        nx = self.model.state_dim
        nu = self.model.control_dim

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 명목 rollout (K, N+1, nx) — 외란 없음
        nominal_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 실제 rollout (피드백 + 외란 통합)
        if self.robust_params.use_feedback or \
                self.robust_params.disturbance_mode != "none":
            real_trajectories = self._rollout_real(
                state, sampled_controls, nominal_trajectories
            )
        else:
            # 피드백 없음 + 외란 없음 → 명목 = 실제
            real_trajectories = nominal_trajectories

        # 5. 비용 계산 — 실제 궤적으로! (핵심)
        costs = self.cost_function.compute_cost(
            real_trajectories, sampled_controls, reference_trajectory
        )

        # 6. MPPI 가중치 계산
        weights = self._compute_weights(costs, self.params.lambda_)

        # 7. 가중 평균으로 제어 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 8. 첫 제어 추출
        optimal_control = self.U[0].copy()

        # 9. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 10. Tube 폭 + 피드백 크기 계산
        best_idx = np.argmin(costs)
        tube_widths = np.linalg.norm(
            real_trajectories[best_idx, :, :2] - nominal_trajectories[best_idx, :, :2],
            axis=1,
        )
        mean_tube_width = float(np.mean(tube_widths))
        max_tube_width = float(np.max(tube_widths))

        # 피드백 크기 (모든 샘플 평균)
        feedback_norms = np.linalg.norm(
            real_trajectories[:, 1:, :2] - nominal_trajectories[:, 1:, :2],
            axis=2,
        )
        mean_feedback_norm = float(np.mean(feedback_norms))

        # 외란 에너지
        disturbance_energy = float(np.sum(self._disturbance_std ** 2))

        # 11. ESS 및 통계
        ess = self._compute_ess(weights)

        robust_stats = {
            "mean_tube_width": mean_tube_width,
            "max_tube_width": max_tube_width,
            "mean_feedback_norm": mean_feedback_norm,
            "disturbance_energy": disturbance_energy,
            "disturbance_mode": self.robust_params.disturbance_mode,
            "use_feedback": self.robust_params.use_feedback,
        }
        self._robust_history.append(robust_stats)

        info = {
            "sample_trajectories": real_trajectories,
            "nominal_trajectories": nominal_trajectories,
            "sample_weights": weights,
            "best_trajectory": real_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "robust_stats": robust_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _rollout_real(
        self,
        initial_state: np.ndarray,
        sampled_controls: np.ndarray,
        nominal_trajectories: np.ndarray,
    ) -> np.ndarray:
        """
        실제 궤적 rollout (피드백 + 외란 통합)

        각 타임스텝에서:
            error = x_real - x_nom
            feedback = _batch_feedback(x_real, x_nom)
            u_real = v(t) + feedback
            x_real(t+1) = model.step(x_real, u_real) + disturbance

        Args:
            initial_state: (nx,) 초기 상태
            sampled_controls: (K, N, nu) 샘플 제어
            nominal_trajectories: (K, N+1, nx) 명목 궤적

        Returns:
            real_trajectories: (K, N+1, nx) 실제 궤적
        """
        K, N, nu = sampled_controls.shape
        nx = self.model.state_dim

        real_trajectories = np.zeros((K, N + 1, nx))
        real_trajectories[:, 0, :] = initial_state

        for t in range(N):
            x_real_t = real_trajectories[:, t, :]     # (K, nx)
            x_nom_t = nominal_trajectories[:, t, :]   # (K, nx)
            v_t = sampled_controls[:, t, :]            # (K, nu)

            # 피드백 계산
            if self.robust_params.use_feedback:
                feedback = self._batch_feedback(x_real_t, x_nom_t)  # (K, nu)
                u_real = v_t + feedback
            else:
                u_real = v_t

            # 제어 제약
            if self.u_min is not None and self.u_max is not None:
                u_real = np.clip(u_real, self.u_min, self.u_max)

            # Forward dynamics
            next_state = self.model.step(x_real_t, u_real, self.params.dt)

            # 외란 추가
            disturbance = self._sample_disturbance(K)  # (K, nx)
            next_state = next_state + disturbance

            real_trajectories[:, t + 1, :] = next_state

        return real_trajectories

    def _batch_feedback(
        self,
        states: np.ndarray,
        nominal_states: np.ndarray,
    ) -> np.ndarray:
        """
        벡터화된 body-frame 피드백 계산

        AncillaryController의 로직을 (K, nx) 배치로 처리.

        Args:
            states: (K, nx) 실제 상태
            nominal_states: (K, nx) 명목 상태

        Returns:
            feedback: (K, nu) 피드백 제어
        """
        K = states.shape[0]
        nx = states.shape[1]
        K_fb = self.ancillary_controller.K_fb  # (nu, nx)

        # World frame 오차
        error_world = states - nominal_states  # (K, nx)

        # Body frame 변환 (heading 회전)
        if nx >= 3:
            theta = nominal_states[:, 2]  # (K,)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            error_body = error_world.copy()
            e_x = error_world[:, 0]
            e_y = error_world[:, 1]
            error_body[:, 0] = cos_t * e_x + sin_t * e_y   # longitudinal
            error_body[:, 1] = -sin_t * e_x + cos_t * e_y  # lateral
        else:
            error_body = error_world

        # 피드백: u_fb = -K_fb @ e_body (배치)
        # K_fb: (nu, nx), error_body: (K, nx) -> feedback: (K, nu)
        feedback = -error_body @ K_fb.T  # (K, nu)

        # 최대 보정값 제한
        if self.ancillary_controller.max_correction is not None:
            mc = self.ancillary_controller.max_correction
            feedback = np.clip(feedback, -mc, mc)

        return feedback

    def _sample_disturbance(self, K: int) -> np.ndarray:
        """
        외란 샘플 생성

        Args:
            K: 샘플 수

        Returns:
            disturbance: (K, nx) 외란 벡터
        """
        nx = len(self._disturbance_std)

        if self.robust_params.disturbance_mode == "none":
            return np.zeros((K, nx))

        elif self.robust_params.disturbance_mode == "gaussian":
            if self.robust_params.n_disturbance_samples == 1:
                return np.random.randn(K, nx) * self._disturbance_std
            else:
                # 다중 외란 샘플 → 평균
                n_ds = self.robust_params.n_disturbance_samples
                samples = np.random.randn(n_ds, K, nx) * self._disturbance_std
                return np.mean(samples, axis=0)

        elif self.robust_params.disturbance_mode == "adversarial":
            # 여러 외란 후보 생성 → 비용 최대화 방향 (최악) 선택
            n_candidates = max(5, self.robust_params.n_disturbance_samples)
            candidates = np.random.randn(n_candidates, K, nx) * self._disturbance_std

            # 최악 선택: L2 노름 기준 상위 alpha
            norms = np.linalg.norm(candidates, axis=2)  # (n_candidates, K)
            # 각 샘플별 최악 후보 선택
            alpha = self.robust_params.robust_alpha
            threshold_idx = max(0, int(n_candidates * alpha) - 1)
            sorted_idx = np.argsort(norms, axis=0)  # (n_candidates, K)
            worst_idx = sorted_idx[threshold_idx]     # (K,)

            result = np.zeros((K, nx))
            for k in range(K):
                result[k] = candidates[worst_idx[k], k]
            return result

        return np.zeros((K, nx))

    def get_robust_statistics(self) -> Dict:
        """
        누적 Robust 통계 반환

        Returns:
            dict:
                - mean_tube_width: 평균 tube 폭
                - max_tube_width: 최대 tube 폭
                - mean_feedback_norm: 평균 피드백 크기
                - disturbance_energy: 외란 에너지
                - history: 전체 히스토리
        """
        if len(self._robust_history) == 0:
            return {
                "mean_tube_width": 0.0,
                "max_tube_width": 0.0,
                "mean_feedback_norm": 0.0,
                "disturbance_energy": 0.0,
                "history": [],
            }

        tube_widths = [s["mean_tube_width"] for s in self._robust_history]
        max_tubes = [s["max_tube_width"] for s in self._robust_history]
        fb_norms = [s["mean_feedback_norm"] for s in self._robust_history]

        return {
            "mean_tube_width": float(np.mean(tube_widths)),
            "max_tube_width": float(np.max(max_tubes)),
            "mean_feedback_norm": float(np.mean(fb_norms)),
            "disturbance_energy": float(np.sum(self._disturbance_std ** 2)),
            "history": self._robust_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 + history 초기화"""
        super().reset()
        self._robust_history = []

    def __repr__(self) -> str:
        return (
            f"RobustMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"disturbance_mode={self.robust_params.disturbance_mode}, "
            f"use_feedback={self.robust_params.use_feedback}, "
            f"disturbance_std={self.robust_params.disturbance_std}, "
            f"K={self.params.K})"
        )
