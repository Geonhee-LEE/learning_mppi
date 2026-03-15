"""
Mobile Manipulator 6-DOF 기구학 모델 (UR5-style 6-DOF Arm + DiffDrive Base)

상태: [x, y, θ, q1, q2, q3, q4, q5, q6] (9차원)
  - x, y: 베이스 위치 (월드 프레임)
  - θ: 베이스 heading
  - q1~q6: 6-DOF 팔 관절 각도

제어: [v, ω, dq1, dq2, dq3, dq4, dq5, dq6] (8차원)
  - v: 베이스 선속도
  - ω: 베이스 각속도
  - dq1~dq6: 관절 속도

DH Parameters (UR5-style, scaled):
  Joint | a(m)  | d(m)  | α(rad)  | 설명
  ------+-------+-------+---------+--------
  1     | 0     | 0.15  | π/2     | shoulder yaw
  2     | 0.30  | 0     | 0       | shoulder pitch
  3     | 0.25  | 0     | 0       | elbow pitch
  4     | 0     | 0.10  | π/2     | wrist 1
  5     | 0     | 0.10  | -π/2    | wrist 2
  6     | 0     | 0.05  | 0       | wrist 3

EE FK: T_world = T_base(x,y,θ) @ T_mount(z=z_mount) @ T_dh_chain(q1..q6)
  → position (3D) + orientation (RPY)
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


# UR5-style DH parameters: [a, d, alpha, theta_offset]
DEFAULT_DH_PARAMS = np.array([
    [0.0,  0.15,  np.pi / 2,  0.0],   # J1: shoulder yaw
    [0.30, 0.0,   0.0,        0.0],   # J2: shoulder pitch
    [0.25, 0.0,   0.0,        0.0],   # J3: elbow pitch
    [0.0,  0.10,  np.pi / 2,  0.0],   # J4: wrist 1
    [0.0,  0.10, -np.pi / 2,  0.0],   # J5: wrist 2
    [0.0,  0.05,  0.0,        0.0],   # J6: wrist 3
])


class MobileManipulator6DOFKinematic(RobotModel):
    """
    Mobile Manipulator 6-DOF 기구학 모델 (UR5-style Arm + DiffDrive Base)

    디커플링 기구학: 베이스와 팔이 독립적으로 움직임.
    CBF 호환: state[0]=x, state[1]=y, state[2]=θ → 기존 ObstacleCost/Shield 작동.

    Args:
        dh_params: (6, 4) DH 파라미터 [a, d, alpha, theta_offset]
        z_mount: 팔 마운트 높이 (m)
        v_max: 최대 베이스 선속도 (m/s)
        omega_max: 최대 베이스 각속도 (rad/s)
        dq_max: 최대 관절 속도 (rad/s)
    """

    def __init__(
        self,
        dh_params: Optional[np.ndarray] = None,
        z_mount: float = 0.1,
        v_max: float = 1.0,
        omega_max: float = 1.0,
        dq_max: float = 2.0,
    ):
        if dh_params is None:
            dh_params = DEFAULT_DH_PARAMS.copy()
        self.dh_params = np.asarray(dh_params, dtype=np.float64)
        assert self.dh_params.shape == (6, 4), f"DH params must be (6,4), got {self.dh_params.shape}"

        self.z_mount = z_mount
        self.v_max = v_max
        self.omega_max = omega_max
        self.dq_max = dq_max

        # Pre-compute cos/sin of alpha for performance
        self._cos_alpha = np.cos(self.dh_params[:, 2])  # (6,)
        self._sin_alpha = np.sin(self.dh_params[:, 2])  # (6,)
        self._a = self.dh_params[:, 0]  # (6,)
        self._d = self.dh_params[:, 1]  # (6,)
        self._theta_offset = self.dh_params[:, 3]  # (6,)

        self._control_lower = np.array(
            [-v_max, -omega_max] + [-dq_max] * 6
        )
        self._control_upper = np.array(
            [v_max, omega_max] + [dq_max] * 6
        )

    @property
    def state_dim(self) -> int:
        return 9  # [x, y, θ, q1, q2, q3, q4, q5, q6]

    @property
    def control_dim(self) -> int:
        return 8  # [v, ω, dq1, dq2, dq3, dq4, dq5, dq6]

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        디커플링 기구학 동역학: dx/dt = f(x, u)

        베이스: ẋ = v·cos(θ), ẏ = v·sin(θ), θ̇ = ω
        팔:    q̇_i = dq_i  (i=1..6)

        Args:
            state: (..., 9) - [x, y, θ, q1, q2, q3, q4, q5, q6]
            control: (..., 8) - [v, ω, dq1, dq2, dq3, dq4, dq5, dq6]

        Returns:
            state_dot: (..., 9)
        """
        theta = state[..., 2]
        v = control[..., 0]
        omega = control[..., 1]

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega

        # Joint velocities: dq1..dq6 = control[2..7]
        joint_dots = [control[..., i] for i in range(2, 8)]

        return np.stack(
            [x_dot, y_dot, theta_dot] + joint_dots, axis=-1
        )

    def _dh_transform(self, theta_i, i):
        """
        Standard DH transform for joint i.

        T_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)

        Args:
            theta_i: (...,) joint angle + offset
            i: joint index (0..5)

        Returns:
            T: (..., 4, 4) homogeneous transform
        """
        q = theta_i + self._theta_offset[i]
        cq = np.cos(q)
        sq = np.sin(q)
        ca = self._cos_alpha[i]
        sa = self._sin_alpha[i]
        a = self._a[i]
        d = self._d[i]

        batch_shape = theta_i.shape
        T = np.zeros(batch_shape + (4, 4), dtype=np.float64)

        T[..., 0, 0] = cq
        T[..., 0, 1] = -sq * ca
        T[..., 0, 2] = sq * sa
        T[..., 0, 3] = a * cq

        T[..., 1, 0] = sq
        T[..., 1, 1] = cq * ca
        T[..., 1, 2] = -cq * sa
        T[..., 1, 3] = a * sq

        T[..., 2, 1] = sa
        T[..., 2, 2] = ca
        T[..., 2, 3] = d

        T[..., 3, 3] = 1.0

        return T

    def _base_transform(self, state: np.ndarray) -> np.ndarray:
        """
        Base-to-mount transform: Rz(θ) + translate(x, y, z_mount)

        Args:
            state: (..., 9)

        Returns:
            T: (..., 4, 4)
        """
        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 2]

        batch_shape = x.shape
        T = np.zeros(batch_shape + (4, 4), dtype=np.float64)

        ct = np.cos(theta)
        st = np.sin(theta)

        T[..., 0, 0] = ct
        T[..., 0, 1] = -st
        T[..., 1, 0] = st
        T[..., 1, 1] = ct
        T[..., 2, 2] = 1.0
        T[..., 3, 3] = 1.0

        T[..., 0, 3] = x
        T[..., 1, 3] = y
        T[..., 2, 3] = self.z_mount

        return T

    def _dh_chain(self, q_joints: np.ndarray) -> np.ndarray:
        """
        Compute full DH chain: T_0^6 = T_1 @ T_2 @ ... @ T_6

        Args:
            q_joints: (..., 6) joint angles

        Returns:
            T: (..., 4, 4) end-effector transform in arm frame
        """
        T = self._dh_transform(q_joints[..., 0], 0)
        for i in range(1, 6):
            T_i = self._dh_transform(q_joints[..., i], i)
            T = np.einsum("...ij,...jk->...ik", T, T_i)
        return T

    def forward_kinematics(self, state: np.ndarray) -> np.ndarray:
        """
        End-Effector 3D position via FK.

        T_world = T_base(x,y,θ) @ T_dh_chain(q1..q6)

        Args:
            state: (..., 9)

        Returns:
            ee_pos: (..., 3) [ee_x, ee_y, ee_z]
        """
        T = self.forward_kinematics_full(state)
        return T[..., :3, 3]

    def forward_kinematics_full(self, state: np.ndarray) -> np.ndarray:
        """
        Full 4x4 transform of end-effector in world frame.

        Args:
            state: (..., 9)

        Returns:
            T: (..., 4, 4)
        """
        T_base = self._base_transform(state)
        q_joints = state[..., 3:9]
        T_arm = self._dh_chain(q_joints)
        return np.einsum("...ij,...jk->...ik", T_base, T_arm)

    def forward_kinematics_pose(self, state: np.ndarray) -> np.ndarray:
        """
        End-Effector pose: position (3D) + orientation (RPY).

        Args:
            state: (..., 9)

        Returns:
            pose: (..., 6) [x, y, z, roll, pitch, yaw]
        """
        T = self.forward_kinematics_full(state)
        pos = T[..., :3, 3]
        R = T[..., :3, :3]
        rpy = self._rotation_matrix_to_rpy(R)
        return np.concatenate([pos, rpy], axis=-1)

    @staticmethod
    def _rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
        """
        Rotation matrix to ZYX Euler angles (roll, pitch, yaw).

        pitch = atan2(-R20, sqrt(R00^2 + R10^2))
        yaw   = atan2(R10, R00)
        roll  = atan2(R21, R22)

        Args:
            R: (..., 3, 3)

        Returns:
            rpy: (..., 3) [roll, pitch, yaw]
        """
        sy = np.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
        pitch = np.arctan2(-R[..., 2, 0], sy)
        yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
        roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
        return np.stack([roll, pitch, yaw], axis=-1)

    def get_joint_positions(self, state: np.ndarray) -> np.ndarray:
        """
        Intermediate joint positions for visualization.

        Returns mount + 6 joint positions (7 total).

        Args:
            state: (..., 9)

        Returns:
            positions: (..., 7, 3) — [mount, j1, j2, j3, j4, j5, ee]
        """
        T_base = self._base_transform(state)
        q_joints = state[..., 3:9]
        batch_shape = state.shape[:-1]

        positions = np.zeros(batch_shape + (7, 3), dtype=np.float64)
        # Mount position = base origin + z_mount
        positions[..., 0, :] = T_base[..., :3, 3]

        # Chain through each DH transform
        T_current = T_base.copy()
        for i in range(6):
            T_i = self._dh_transform(q_joints[..., i], i)
            T_current = np.einsum("...ij,...jk->...ik", T_current, T_i)
            positions[..., i + 1, :] = T_current[..., :3, 3]

        return positions

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        상태 정규화: θ, q1~q6을 [-π, π] 범위로

        Args:
            state: (..., 9)

        Returns:
            normalized_state: (..., 9)
        """
        normalized = state.copy()
        for idx in range(2, 9):  # θ, q1, q2, q3, q4, q5, q6
            normalized[..., idx] = np.arctan2(
                np.sin(state[..., idx]), np.cos(state[..., idx])
            )
        return normalized

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
            "q3": state[5],
            "q4": state[6],
            "q5": state[7],
            "q6": state[8],
        }

    def __repr__(self) -> str:
        return (
            f"MobileManipulator6DOFKinematic("
            f"z_mount={self.z_mount}, "
            f"v_max={self.v_max}, omega_max={self.omega_max}, "
            f"dq_max={self.dq_max})"
        )
