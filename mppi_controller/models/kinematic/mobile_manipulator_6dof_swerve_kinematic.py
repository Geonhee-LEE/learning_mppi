"""
Mobile Manipulator 6-DOF Swerve 기구학 모델 (UR5-style 6-DOF Arm + Swerve Base)

Swerve Drive는 holonomic (전방향 이동) 베이스로,
DiffDrive와 달리 횡방향(vy) 이동이 가능하여 EE 추적에 유리.

상태: [x, y, θ, q1, q2, q3, q4, q5, q6] (9차원)
  - x, y: 베이스 위치 (월드 프레임)
  - θ: 베이스 heading
  - q1~q6: 6-DOF 팔 관절 각도

제어: [vx, vy, ω, dq1, dq2, dq3, dq4, dq5, dq6] (9차원)
  - vx: 베이스 body-frame 전방 속도
  - vy: 베이스 body-frame 횡방 속도
  - ω: 베이스 각속도
  - dq1~dq6: 관절 속도

DH Parameters: DiffDrive 6-DOF 버전과 동일 (UR5-style)
EE FK: T_world = T_base(x,y,θ) @ T_mount(z=z_mount) @ T_dh_chain(q1..q6)
"""

import numpy as np
from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
    DEFAULT_DH_PARAMS,
)
from typing import Optional, Tuple


class MobileManipulator6DOFSwerveKinematic(MobileManipulator6DOFKinematic):
    """
    Mobile Manipulator 6-DOF Swerve 기구학 모델 (Holonomic Base + UR5-style Arm)

    DiffDrive 버전(MobileManipulator6DOFKinematic)을 상속하여
    base dynamics만 holonomic swerve로 교체. DH chain FK는 그대로 재사용.

    CBF 호환: state[0]=x, state[1]=y, state[2]=θ 유지.

    Args:
        dh_params: (6, 4) DH 파라미터 [a, d, alpha, theta_offset]
        z_mount: 팔 마운트 높이 (m)
        vx_max: 최대 전방 속도 (m/s)
        vy_max: 최대 횡방 속도 (m/s)
        omega_max: 최대 각속도 (rad/s)
        dq_max: 최대 관절 속도 (rad/s)
    """

    def __init__(
        self,
        dh_params: Optional[np.ndarray] = None,
        z_mount: float = 0.1,
        vx_max: float = 1.0,
        vy_max: float = 1.0,
        omega_max: float = 1.0,
        dq_max: float = 2.0,
    ):
        # Initialize parent (sets up DH chain, FK, etc.)
        super().__init__(
            dh_params=dh_params,
            z_mount=z_mount,
            v_max=vx_max,  # reuse parent's v_max slot
            omega_max=omega_max,
            dq_max=dq_max,
        )
        self.vx_max = vx_max
        self.vy_max = vy_max

        # Override control bounds: 9D [vx, vy, ω, dq1~6]
        self._control_lower = np.array(
            [-vx_max, -vy_max, -omega_max] + [-dq_max] * 6
        )
        self._control_upper = np.array(
            [vx_max, vy_max, omega_max] + [dq_max] * 6
        )

    @property
    def control_dim(self) -> int:
        return 9  # [vx, vy, ω, dq1, dq2, dq3, dq4, dq5, dq6]

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        Holonomic swerve base + decoupled arm dynamics.

        Base (body-to-world rotation):
            ẋ  = vx·cos(θ) - vy·sin(θ)
            ẏ  = vx·sin(θ) + vy·cos(θ)
            θ̇  = ω
        Arm:
            q̇_i = dq_i  (i=1..6)

        Args:
            state: (..., 9) - [x, y, θ, q1~q6]
            control: (..., 9) - [vx, vy, ω, dq1~dq6]

        Returns:
            state_dot: (..., 9)
        """
        theta = state[..., 2]
        vx = control[..., 0]
        vy = control[..., 1]
        omega = control[..., 2]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_dot = vx * cos_theta - vy * sin_theta
        y_dot = vx * sin_theta + vy * cos_theta
        theta_dot = omega

        # Joint velocities: dq1..dq6 = control[3..8]
        joint_dots = [control[..., i] for i in range(3, 9)]

        return np.stack(
            [x_dot, y_dot, theta_dot] + joint_dots, axis=-1
        )

    def state_to_dict(self, state: np.ndarray) -> dict:
        """상태를 딕셔너리로 변환"""
        d = super().state_to_dict(state)
        d["base_type"] = "swerve"
        return d

    def __repr__(self) -> str:
        return (
            f"MobileManipulator6DOFSwerveKinematic("
            f"z_mount={self.z_mount}, "
            f"vx_max={self.vx_max}, vy_max={self.vy_max}, "
            f"omega_max={self.omega_max}, dq_max={self.dq_max})"
        )
