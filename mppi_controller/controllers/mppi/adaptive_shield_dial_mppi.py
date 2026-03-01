"""
Adaptive Shield-DIAL-MPPI Controller

ShieldDIALMPPI의 확산 어닐링 + 적응형 α(d,v) CBF shield 결합.

기존 ShieldDIALMPPIController는 고정 α 사용:
  → 모든 거리·속도에서 동일한 보수성

이 컨트롤러는 AdaptiveShieldMPPI의 α(d,v) 로직을 이식:
  → 가까울수록 α 감소 → v_ceiling 감소 → 더 보수적
  → 빠를수록 α 감소 → v_ceiling 감소 → 더 보수적

상속 체인:
    MPPIController → DIALMPPIController → ShieldDIALMPPIController
                                            └── AdaptiveShieldDIALMPPIController
"""

import numpy as np
from typing import Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import AdaptiveShieldDIALMPPIParams
from mppi_controller.controllers.mppi.shield_dial_mppi import ShieldDIALMPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class AdaptiveShieldDIALMPPIController(ShieldDIALMPPIController):
    """
    Adaptive Shield-DIAL-MPPI Controller

    ShieldDIALMPPIController의 _cbf_shield_batch()를 오버라이드하여
    고정 α 대신 거리/속도 기반 적응형 α(d,v) 사용.

    α(d,v) = α_base · (α_dist + (1-α_dist)·σ(k·(d-d_safe))) / (1 + α_vel·|v|)
    → 가까울수록 (d 작을수록) α 감소 → v_ceiling 감소 → 더 보수적
    → 빠를수록 (|v| 클수록) α 감소 → v_ceiling 감소 → 더 보수적
    → clip(0.01, 0.99)
    """

    def __init__(
        self,
        model: RobotModel,
        params: AdaptiveShieldDIALMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.adaptive_params = params

    def _cbf_shield_batch(
        self, states: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        적응형 CBF shield (완전 벡터화)

        ShieldDIALMPPI의 _cbf_shield_batch()와 동일 로직이지만
        고정 α 대신 장애물별·샘플별 적응형 α(d,v) 사용.
        """
        K = states.shape[0]
        safe_controls = controls.copy()
        v_original = controls[:, 0]  # (K,)

        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        obstacles = self.shield_dial_params.cbf_obstacles
        safety_margin = self.shield_dial_params.cbf_safety_margin
        p = self.adaptive_params

        v_ceiling = np.full(K, np.inf)

        for obs_x, obs_y, obs_r in obstacles:
            effective_r = obs_r + safety_margin

            dx = x - obs_x
            dy = y - obs_y

            # 거리 계산
            dist = np.sqrt(dx**2 + dy**2)
            d_surface = dist - obs_r  # 장애물 표면까지 거리

            # 적응형 alpha 계산
            # α(d,v) = α_base · (α_dist + (1-α_dist)·σ(k·(d-d_safe))) / (1 + α_vel·|v|)
            sigmoid_val = 1.0 / (1.0 + np.exp(-p.k_dist * (d_surface - p.d_safe)))
            alpha = (p.alpha_base
                     * (p.alpha_dist + (1.0 - p.alpha_dist) * sigmoid_val)
                     / (1.0 + p.alpha_vel * np.abs(v_original)))
            alpha = np.clip(alpha, 0.01, 0.99)

            # Barrier value
            h = dx**2 + dy**2 - effective_r**2

            # Lg_h[0]
            Lg_h_v = 2.0 * dx * cos_theta + 2.0 * dy * sin_theta

            # 접근 시만 제약
            approaching = Lg_h_v < -1e-10

            # v_ceiling = α·h / |Lg_h_v| (per-sample adaptive α)
            v_ceiling_obs = np.where(
                approaching,
                alpha * h / np.maximum(np.abs(Lg_h_v), 1e-10),
                np.inf,
            )

            v_ceiling = np.minimum(v_ceiling, v_ceiling_obs)

        v_safe = np.minimum(v_original, v_ceiling)
        if self.u_min is not None:
            v_safe = np.maximum(v_safe, self.u_min[0])

        safe_controls[:, 0] = v_safe
        intervened = v_safe < v_original - 1e-10
        vel_reduction = np.where(intervened, v_original - v_safe, 0.0)

        return safe_controls, intervened, vel_reduction

    def __repr__(self) -> str:
        p = self.adaptive_params
        return (
            f"AdaptiveShieldDIALMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"n_diffuse_init={self.dial_params.n_diffuse_init}, "
            f"n_diffuse={self.dial_params.n_diffuse}, "
            f"alpha_base={p.alpha_base}, alpha_dist={p.alpha_dist}, "
            f"alpha_vel={p.alpha_vel}, d_safe={p.d_safe}, "
            f"K={self.params.K})"
        )
