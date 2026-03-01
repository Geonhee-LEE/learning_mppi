"""
Adaptive Shield-SVG-MPPI Controller

ShieldSVGMPPI의 고품질 SVGD 샘플링 + 적응형 α(d,v) CBF shield 결합.

기존 ShieldSVGMPPIController는 고정 α 사용:
  → 모든 거리·속도에서 동일한 보수성

이 컨트롤러는 AdaptiveShieldMPPI의 α(d,v) 로직을 이식:
  → 가까울수록 α 감소 → v_ceiling 감소 → 더 보수적
  → 빠를수록 α 감소 → v_ceiling 감소 → 더 보수적

상속 체인:
    SVGMPPIController
      └── ShieldSVGMPPIController (SVG + 고정α shield)
            └── AdaptiveShieldSVGMPPIController (SVG + 적응형α shield)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.shield_svg_mppi import (
    ShieldSVGMPPIController, ShieldSVGMPPIParams,
)


@dataclass
class AdaptiveShieldSVGMPPIParams(ShieldSVGMPPIParams):
    """
    Adaptive Shield-SVG-MPPI 파라미터

    ShieldSVGMPPIParams + 적응형 alpha 파라미터.

    Attributes:
        alpha_base: 기본 CBF alpha (최대값, d >> d_safe일 때)
        alpha_dist: 최소 alpha 비율 (d << d_safe일 때 α = α_base × α_dist)
        alpha_vel: 속도 반응 계수 (α /= (1 + α_vel·|v|))
        k_dist: sigmoid 경사도 (클수록 전환 급격)
        d_safe: 안전 거리 기준 (m)
    """

    alpha_base: float = 0.3
    alpha_dist: float = 0.1
    alpha_vel: float = 0.5
    k_dist: float = 2.0
    d_safe: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        assert self.alpha_base > 0, "alpha_base must be positive"
        assert self.alpha_dist >= 0, "alpha_dist must be non-negative"
        assert self.alpha_vel >= 0, "alpha_vel must be non-negative"
        assert self.k_dist > 0, "k_dist must be positive"
        assert self.d_safe > 0, "d_safe must be positive"


class AdaptiveShieldSVGMPPIController(ShieldSVGMPPIController):
    """
    Adaptive Shield-SVG-MPPI Controller

    ShieldSVGMPPIController의 _cbf_shield_batch()를 오버라이드하여
    고정 α 대신 거리/속도 기반 적응형 α(d,v) 사용.

    α(d,v) = α_base · (α_dist + (1-α_dist)·σ(k·(d-d_safe))) / (1 + α_vel·|v|)
    → 가까울수록 (d 작을수록) α 감소 → v_ceiling 감소 → 더 보수적
    → 빠를수록 (|v| 클수록) α 감소 → v_ceiling 감소 → 더 보수적
    → clip(0.01, 0.99)
    """

    def __init__(self, model: RobotModel, params: AdaptiveShieldSVGMPPIParams):
        super().__init__(model, params)
        self.adaptive_params = params

    def _cbf_shield_batch(
        self, states: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        적응형 CBF shield (완전 벡터화)

        ShieldSVGMPPI의 _cbf_shield_batch()와 동일 로직이지만
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

        obstacles = self.shield_svg_params.cbf_obstacles
        safety_margin = self.shield_svg_params.cbf_safety_margin
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
            f"AdaptiveShieldSVGMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"G={self.G}, K={self.params.K}, "
            f"alpha_base={p.alpha_base}, alpha_dist={p.alpha_dist}, "
            f"alpha_vel={p.alpha_vel}, d_safe={p.d_safe})"
        )
