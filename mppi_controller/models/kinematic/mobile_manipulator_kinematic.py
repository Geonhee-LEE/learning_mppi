"""
Mobile Manipulator 기구학 모델 (2-DOF Planar Arm + DiffDrive Base)

상태: [x, y, θ, q1, q2] (5차원)
  - x, y: 베이스 위치 (월드 프레임)
  - θ: 베이스 heading
  - q1: 상완 관절 각도 (베이스 프레임 기준)
  - q2: 전완 관절 각도 (상완 프레임 기준)

제어: [v, ω, dq1, dq2] (4차원)
  - v: 베이스 선속도
  - ω: 베이스 각속도
  - dq1: 상완 관절 속도
  - dq2: 전완 관절 속도

End-Effector FK:
  ee_x = x + L1·cos(θ+q1) + L2·cos(θ+q1+q2)
  ee_y = y + L1·sin(θ+q1) + L2·sin(θ+q1+q2)
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class MobileManipulatorKinematic(RobotModel):
    """
    Mobile Manipulator 기구학 모델 (2-DOF Planar Arm + DiffDrive Base)

    디커플링 기구학: 베이스와 팔이 독립적으로 움직임.
    CBF 호환: state[0]=x, state[1]=y, state[2]=θ → 기존 ObstacleCost/Shield 작동.

    Args:
        L1: 상완(shoulder→elbow) 길이 (m)
        L2: 전완(elbow→end-effector) 길이 (m)
        v_max: 최대 베이스 선속도 (m/s)
        omega_max: 최대 베이스 각속도 (rad/s)
        dq_max: 최대 관절 속도 (rad/s)
    """

    def __init__(
        self,
        L1: float = 0.3,
        L2: float = 0.25,
        v_max: float = 1.0,
        omega_max: float = 1.0,
        dq_max: float = 2.0,
    ):
        self.L1 = L1
        self.L2 = L2
        self.v_max = v_max
        self.omega_max = omega_max
        self.dq_max = dq_max

        self._control_lower = np.array([-v_max, -omega_max, -dq_max, -dq_max])
        self._control_upper = np.array([v_max, omega_max, dq_max, dq_max])

    @property
    def state_dim(self) -> int:
        return 5  # [x, y, θ, q1, q2]

    @property
    def control_dim(self) -> int:
        return 4  # [v, ω, dq1, dq2]

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        디커플링 기구학 동역학: dx/dt = f(x, u)

        베이스: ẋ = v·cos(θ), ẏ = v·sin(θ), θ̇ = ω
        팔:    q̇1 = dq1,     q̇2 = dq2

        Args:
            state: (5,) 또는 (batch, 5) - [x, y, θ, q1, q2]
            control: (4,) 또는 (batch, 4) - [v, ω, dq1, dq2]

        Returns:
            state_dot: (5,) 또는 (batch, 5)
        """
        theta = state[..., 2]
        v = control[..., 0]
        omega = control[..., 1]
        dq1 = control[..., 2]
        dq2 = control[..., 3]

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega
        q1_dot = dq1
        q2_dot = dq2

        return np.stack([x_dot, y_dot, theta_dot, q1_dot, q2_dot], axis=-1)

    def forward_kinematics(self, state: np.ndarray) -> np.ndarray:
        """
        End-Effector Forward Kinematics

        ee_x = x + L1·cos(θ+q1) + L2·cos(θ+q1+q2)
        ee_y = y + L1·sin(θ+q1) + L2·sin(θ+q1+q2)

        Args:
            state: (..., 5) - 임의 배치 차원 지원

        Returns:
            ee_pos: (..., 2) - [ee_x, ee_y]
        """
        phi1 = state[..., 2] + state[..., 3]       # θ + q1
        phi12 = phi1 + state[..., 4]                 # θ + q1 + q2

        ee_x = state[..., 0] + self.L1 * np.cos(phi1) + self.L2 * np.cos(phi12)
        ee_y = state[..., 1] + self.L1 * np.sin(phi1) + self.L2 * np.sin(phi12)

        return np.stack([ee_x, ee_y], axis=-1)

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """제어 제약 반환"""
        return (self._control_lower, self._control_upper)

    def state_to_dict(self, state: np.ndarray) -> dict:
        """상태를 딕셔너리로 변환"""
        return {
            "x": state[0],
            "y": state[1],
            "theta": state[2],
            "q1": state[3],
            "q2": state[4],
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        상태 정규화: θ, q1, q2를 [-π, π] 범위로

        Args:
            state: (5,) 또는 (batch, 5)

        Returns:
            normalized_state: (5,) 또는 (batch, 5)
        """
        normalized = state.copy()
        for idx in (2, 3, 4):  # θ, q1, q2
            normalized[..., idx] = np.arctan2(
                np.sin(state[..., idx]), np.cos(state[..., idx])
            )
        return normalized

    def __repr__(self) -> str:
        return (
            f"MobileManipulatorKinematic("
            f"L1={self.L1}, L2={self.L2}, "
            f"v_max={self.v_max}, omega_max={self.omega_max}, "
            f"dq_max={self.dq_max})"
        )
