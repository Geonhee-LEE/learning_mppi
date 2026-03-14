"""
Differential Drive 기구학 모델

상태: [x, y, θ] (3차원)
제어: [v, ω] (2차원) - 선속도, 각속도
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class DifferentialDriveKinematic(RobotModel):
    """
    Differential Drive 기구학 모델

    차동 구동 로봇의 기구학 모델. 질량/관성을 고려하지 않고 순수 기하학적 관계만 사용.

    상태:
        x (m): 월드 프레임 x 위치
        y (m): 월드 프레임 y 위치
        θ (rad): 월드 프레임 heading 각도

    제어:
        v (m/s): 선속도 (로봇 body frame x 방향)
        ω (rad/s): 각속도 (로봇 body frame z 축 회전)

    동역학:
        dx/dt = v * cos(θ)
        dy/dt = v * sin(θ)
        dθ/dt = ω

    Args:
        v_max: 최대 선속도 (m/s)
        omega_max: 최대 각속도 (rad/s)
        wheelbase: 바퀴 간격 (m) - 선택적, 현재 미사용
    """

    def __init__(
        self,
        v_max: float = 1.0,
        omega_max: float = 1.0,
        wheelbase: Optional[float] = None,
    ):
        self.v_max = v_max
        self.omega_max = omega_max
        self.wheelbase = wheelbase

        # 제어 제약
        self._control_lower = np.array([-v_max, -omega_max])
        self._control_upper = np.array([v_max, omega_max])

    @property
    def state_dim(self) -> int:
        return 3  # [x, y, θ]

    @property
    def control_dim(self) -> int:
        return 2  # [v, ω]

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        기구학 동역학: dx/dt = f(x, u)

        Args:
            state: (3,) 또는 (batch, 3) - [x, y, θ]
            control: (2,) 또는 (batch, 2) - [v, ω]

        Returns:
            state_dot: (3,) 또는 (batch, 3) - [dx/dt, dy/dt, dθ/dt]
        """
        # 벡터화 지원: state[..., i]는 마지막 차원 인덱싱
        theta = state[..., 2]
        v = control[..., 0]
        omega = control[..., 1]

        # 기구학 방정식
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega

        # 스택하여 반환 (마지막 차원으로)
        return np.stack([x_dot, y_dot, theta_dot], axis=-1)

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """제어 제약 반환"""
        return (self._control_lower, self._control_upper)

    def state_to_dict(self, state: np.ndarray) -> dict:
        """
        상태를 딕셔너리로 변환 (디버깅/시각화용)

        Args:
            state: (3,) - [x, y, θ]

        Returns:
            {"x": x, "y": y, "theta": theta}
        """
        return {"x": state[0], "y": state[1], "theta": state[2]}

    def render_config(self) -> dict:
        """Circle body (DiffDrive)."""
        return {"shape": "circle", "radius": 0.2, "color": "#1f77b4"}

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        상태 정규화 (θ를 [-π, π] 범위로)

        Args:
            state: (3,) 또는 (batch, 3)

        Returns:
            normalized_state: (3,) 또는 (batch, 3)
        """
        normalized = state.copy()
        # θ를 [-π, π] 범위로 정규화
        normalized[..., 2] = np.arctan2(
            np.sin(state[..., 2]), np.cos(state[..., 2])
        )
        return normalized

    def __repr__(self) -> str:
        return (
            f"DifferentialDriveKinematic("
            f"v_max={self.v_max}, omega_max={self.omega_max})"
        )
