"""
Quadruped Robot 기구학 모델 (Body-Level Kinematics)

하위 보행 컨트롤러가 다리 동역학을 처리한다고 가정.
몸체 수준에서 (vx, vy, ω, vz, pitch_rate)를 제어 입력으로 받음.

상태: [x, y, θ, z, pitch] (5차원)
  - x, y: 몸체 위치 (월드 프레임)
  - θ: heading (yaw)
  - z: 몸체 높이
  - pitch: 몸체 pitch 각도

제어: [vx, vy, ω, vz, pitch_rate] (5차원)
  - vx: body-frame 전진 속도
  - vy: body-frame 횡방향 속도
  - ω: yaw rate
  - vz: 수직 속도
  - pitch_rate: pitch 변화율

동역학:
  dx/dt = vx·cos(θ) - vy·sin(θ)
  dy/dt = vx·sin(θ) + vy·cos(θ)
  dθ/dt = ω
  dz/dt = vz
  dpitch/dt = pitch_rate

CBF 호환: state[0]=x, state[1]=y, state[2]=θ → 기존 ObstacleCost/Shield 작동.
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class QuadrupedKinematic(RobotModel):
    """
    Quadruped Robot 기구학 모델 (Body-Level)

    전방향 이동 가능 (홀로노믹 베이스) + 높이/pitch 제어.

    Args:
        vx_max: 최대 전진 속도 (m/s)
        vy_max: 최대 횡방향 속도 (m/s)
        omega_max: 최대 yaw rate (rad/s)
        vz_max: 최대 수직 속도 (m/s)
        pitch_rate_max: 최대 pitch rate (rad/s)
        z_min: 최소 높이 (m)
        z_max: 최대 높이 (m)
        z_nom: 공칭 높이 (m)
        pitch_nom: 공칭 pitch (rad)
    """

    def __init__(
        self,
        vx_max: float = 0.8,
        vy_max: float = 0.5,
        omega_max: float = 1.0,
        vz_max: float = 0.3,
        pitch_rate_max: float = 0.5,
        z_min: float = 0.15,
        z_max: float = 0.40,
        z_nom: float = 0.28,
        pitch_nom: float = 0.0,
    ):
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.omega_max = omega_max
        self.vz_max = vz_max
        self.pitch_rate_max = pitch_rate_max
        self.z_min = z_min
        self.z_max = z_max
        self.z_nom = z_nom
        self.pitch_nom = pitch_nom

        self._control_lower = np.array([
            -vx_max, -vy_max, -omega_max, -vz_max, -pitch_rate_max
        ])
        self._control_upper = np.array([
            vx_max, vy_max, omega_max, vz_max, pitch_rate_max
        ])

    @property
    def state_dim(self) -> int:
        return 5  # [x, y, θ, z, pitch]

    @property
    def control_dim(self) -> int:
        return 5  # [vx, vy, ω, vz, pitch_rate]

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        Body-level 기구학: dx/dt = f(x, u)

        Args:
            state: (..., 5) - [x, y, θ, z, pitch]
            control: (..., 5) - [vx, vy, ω, vz, pitch_rate]

        Returns:
            state_dot: (..., 5)
        """
        theta = state[..., 2]
        vx = control[..., 0]
        vy = control[..., 1]
        omega = control[..., 2]
        vz = control[..., 3]
        pitch_rate = control[..., 4]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_dot = vx * cos_theta - vy * sin_theta
        y_dot = vx * sin_theta + vy * cos_theta
        theta_dot = omega
        z_dot = vz
        pitch_dot = pitch_rate

        return np.stack([x_dot, y_dot, theta_dot, z_dot, pitch_dot], axis=-1)

    def forward_kinematics(self, state: np.ndarray) -> np.ndarray:
        """
        몸체 위치 반환 (CBF/장애물 비용 호환)

        Args:
            state: (..., 5)

        Returns:
            position: (..., 2) - [x, y]
        """
        return state[..., :2]

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return (self._control_lower, self._control_upper)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        상태 정규화: θ, pitch → [-π, π], z → [z_min, z_max]

        Args:
            state: (..., 5)

        Returns:
            normalized_state: (..., 5)
        """
        normalized = state.copy()
        # θ (heading) wrapping
        normalized[..., 2] = np.arctan2(
            np.sin(state[..., 2]), np.cos(state[..., 2])
        )
        # z clamping
        normalized[..., 3] = np.clip(state[..., 3], self.z_min, self.z_max)
        # pitch wrapping
        normalized[..., 4] = np.arctan2(
            np.sin(state[..., 4]), np.cos(state[..., 4])
        )
        return normalized

    def state_to_dict(self, state: np.ndarray) -> dict:
        return {
            "x": state[0],
            "y": state[1],
            "theta": state[2],
            "z": state[3],
            "pitch": state[4],
        }

    def __repr__(self) -> str:
        return (
            f"QuadrupedKinematic("
            f"vx_max={self.vx_max}, vy_max={self.vy_max}, "
            f"omega_max={self.omega_max}, vz_max={self.vz_max}, "
            f"pitch_rate_max={self.pitch_rate_max})"
        )
