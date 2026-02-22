"""
Adaptive Shield-MPPI Controller

거리/속도 기반 적응형 α(d,v) Shield-MPPI.
가까운 장애물일수록, 속도가 빠를수록 더 보수적인 CBF 제약 적용.

기존 ShieldMPPIController 상속, _cbf_shield_batch()만 오버라이드.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import ShieldMPPIParams
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


@dataclass
class AdaptiveShieldParams(ShieldMPPIParams):
    """
    Adaptive Shield-MPPI 파라미터

    Attributes:
        alpha_base: 기본 CBF alpha (최대값, d >> d_safe일 때)
        alpha_dist: 최소 alpha 비율 (d << d_safe일 때 α = α_base × α_dist)
        alpha_vel: 속도 반응 계수 (α /= (1 + α_vel·|v|))
        k_dist: sigmoid 경사도 (클수록 전환 급격)
        d_safe: 안전 거리 기준 (m)
    """
    alpha_base: float = 0.3
    alpha_dist: float = 0.1    # 가까울 때 α → α_base × 0.1 = 0.03 (매우 보수적)
    alpha_vel: float = 0.5     # 속도 반응 강화
    k_dist: float = 2.0
    d_safe: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        assert self.alpha_base > 0, "alpha_base must be positive"
        assert self.alpha_dist >= 0, "alpha_dist must be non-negative"
        assert self.alpha_vel >= 0, "alpha_vel must be non-negative"
        assert self.k_dist > 0, "k_dist must be positive"
        assert self.d_safe > 0, "d_safe must be positive"


class AdaptiveShieldMPPIController(ShieldMPPIController):
    """
    Adaptive Shield-MPPI Controller

    ShieldMPPI의 _cbf_shield_batch()를 오버라이드하여
    고정 α 대신 거리/속도 기반 적응형 α(d,v) 사용.

    α(d,v) = α_base · σ(k_dist · (d - d_safe)) / (1 + α_vel · |v|)
    → 가까울수록 (d 작을수록) α 감소 → v_ceiling 감소 → 더 보수적
    → 빠를수록 (|v| 클수록) α 감소 → v_ceiling 감소 → 더 보수적
    → clip(0.01, 0.99)
    """

    def __init__(
        self,
        model: RobotModel,
        params: AdaptiveShieldParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.adaptive_params = params

    def _adaptive_alpha_batch(
        self, states: np.ndarray, obs_x: float, obs_y: float, obs_r: float
    ) -> np.ndarray:
        """
        적응형 α 계산 (벡터화)

        Args:
            states: (K, nx) 현재 상태 [x, y, θ, ...]
            obs_x, obs_y, obs_r: 장애물 위치 및 반경

        Returns:
            alpha: (K,) 적응형 CBF alpha
        """
        p = self.adaptive_params
        x = states[:, 0]
        y = states[:, 1]

        # 거리 계산
        dx = x - obs_x
        dy = y - obs_y
        d = np.sqrt(dx**2 + dy**2) - obs_r  # 표면까지 거리

        # 속도 크기 (diff-drive: control[0] = v 이지만, 상태에서 추정 불가하므로
        # 여기서는 d의 변화로 근사하지 않고 단순히 거리만 사용)
        # → velocity는 _cbf_shield_batch에서 controls[:, 0]으로 접근 가능
        # 이 함수에서는 거리 기반 alpha만 계산, velocity 부분은 _cbf_shield_batch에서 추가

        # 거리 기반 alpha: sigmoid로 가까울수록 α 감소 (더 보수적)
        # σ(k·(d - d_safe)) → d >> d_safe: ~1, d << d_safe: ~0
        sigmoid_val = 1.0 / (1.0 + np.exp(-p.k_dist * (d - p.d_safe)))
        alpha = p.alpha_base * (p.alpha_dist + (1.0 - p.alpha_dist) * sigmoid_val)

        return alpha

    def _cbf_shield_batch(
        self, states: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        적응형 CBF shield (완전 벡터화)

        ShieldMPPI의 _cbf_shield_batch()와 동일 로직이지만
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

        obstacles = self.cbf_params.cbf_obstacles
        safety_margin = self.cbf_params.cbf_safety_margin
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
            # α(d,v) = α_base · σ(k·(d - d_safe)) / (1 + α_vel·|v|)
            # 가까울수록 σ→0 → α 감소 → v_ceiling 감소 → 더 보수적
            # 빠를수록 분모 증가 → α 감소 → 더 보수적
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
        return (f"AdaptiveShieldMPPIController("
                f"model={self.model.__class__.__name__}, "
                f"α_base={p.alpha_base}, α_dist={p.alpha_dist}, "
                f"α_vel={p.alpha_vel}, d_safe={p.d_safe})")
