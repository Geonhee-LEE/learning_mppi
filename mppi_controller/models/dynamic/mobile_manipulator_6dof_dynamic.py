"""
Mobile Manipulator 6-DOF 동역학 모델 (Ground-truth)

기구학 모델을 상속하여 현실적 물리 효과 추가.
Residual Dynamics 학습 데이터의 "정답" 역할.

State:  [x, y, θ, q1, q2, q3, q4, q5, q6] (9D, kinematic과 동일)
Control: [v, ω, dq1, dq2, dq3, dq4, dq5, dq6] (8D, kinematic과 동일)

물리 효과 (forward_dynamics 오버라이드):
  1. Base velocity lag: v_eff = v * (1 - e^(-k*|v|))  (저속 비선형 마찰)
  2. Joint friction: dq_eff = dq_i - c_i * sign(dq_i) * |dq_i|^0.8  (비선형)
  3. Arm-base coupling: θ̇ += k_coupling * Σ|dq_i * I_i|  (관성 반작용)
  4. Gravity droop: q̇_2,3 -= g_droop * sin(q2+q3)  (어깨/팔꿈치 처짐)
"""

import numpy as np
from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from typing import Optional, Tuple


# 기본 관절 마찰 계수: 어깨(큰 토크) → 손목(작은 토크)
DEFAULT_JOINT_FRICTION = np.array([0.15, 0.12, 0.10, 0.08, 0.06, 0.05])

# 기본 관절 관성: 어깨(무거움) → 손목(가벼움)
DEFAULT_JOINT_INERTIA = np.array([0.5, 0.4, 0.3, 0.15, 0.1, 0.05])


class MobileManipulator6DOFDynamic(MobileManipulator6DOFKinematic):
    """
    Mobile Manipulator 6-DOF 동역학 모델 (UR5-style Arm + DiffDrive Base)

    기구학 모델에 마찰, 중력, 커플링 효과를 추가한 ground-truth 모델.
    9D 상태 유지 (velocity expansion 없이 steady-state 보정).

    Args:
        dh_params: (6, 4) DH 파라미터 [a, d, alpha, theta_offset]
        z_mount: 팔 마운트 높이 (m)
        v_max: 최대 베이스 선속도 (m/s)
        omega_max: 최대 베이스 각속도 (rad/s)
        dq_max: 최대 관절 속도 (rad/s)
        base_friction: 베이스 마찰 계수 (속도 감쇠)
        base_response_k: 베이스 비선형 응답 상수
        joint_friction: (6,) 관절별 마찰 계수
        coupling_gain: 팔→베이스 커플링 게인
        gravity_droop: 중력 처짐 계수
        joint_inertia: (6,) 관절 관성
    """

    def __init__(
        self,
        dh_params=None,
        z_mount: float = 0.1,
        v_max: float = 1.0,
        omega_max: float = 1.0,
        dq_max: float = 2.0,
        base_friction: float = 0.3,
        base_response_k: float = 5.0,
        joint_friction: Optional[np.ndarray] = None,
        coupling_gain: float = 0.02,
        gravity_droop: float = 0.08,
        joint_inertia: Optional[np.ndarray] = None,
    ):
        super().__init__(dh_params, z_mount, v_max, omega_max, dq_max)

        self.base_friction = base_friction
        self.base_response_k = base_response_k
        self.joint_friction = (
            np.asarray(joint_friction, dtype=np.float64)
            if joint_friction is not None
            else DEFAULT_JOINT_FRICTION.copy()
        )
        self.coupling_gain = coupling_gain
        self.gravity_droop = gravity_droop
        self.joint_inertia = (
            np.asarray(joint_inertia, dtype=np.float64)
            if joint_inertia is not None
            else DEFAULT_JOINT_INERTIA.copy()
        )

        assert self.joint_friction.shape == (6,), (
            f"joint_friction must be (6,), got {self.joint_friction.shape}"
        )
        assert self.joint_inertia.shape == (6,), (
            f"joint_inertia must be (6,), got {self.joint_inertia.shape}"
        )

    @property
    def model_type(self) -> str:
        return "dynamic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        동역학: dx/dt = f_kinematic(x, u) + f_correction(x, u)

        Args:
            state: (..., 9) - [x, y, θ, q1..q6]
            control: (..., 8) - [v, ω, dq1..dq6]

        Returns:
            state_dot: (..., 9)
        """
        kin_dot = super().forward_dynamics(state, control)
        correction = self._compute_physics_correction(state, control)
        return kin_dot + correction

    def _compute_physics_correction(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        물리 보정 벡터 계산.

        Args:
            state: (..., 9)
            control: (..., 8)

        Returns:
            correction: (..., 9)
        """
        batch_shape = state.shape[:-1]
        correction = np.zeros_like(state)

        v = control[..., 0]
        omega = control[..., 1]
        theta = state[..., 2]

        # 1. Base velocity lag: 저속에서 비선형 마찰
        #    v_eff = v * (1 - e^(-k*|v|))
        #    correction = v_eff - v (항상 음수, 속도 감소)
        response = 1.0 - np.exp(-self.base_response_k * np.abs(v))
        v_eff = v * response
        v_correction = v_eff - v  # 음수 보정

        # x_dot, y_dot 보정
        correction[..., 0] = v_correction * np.cos(theta)
        correction[..., 1] = v_correction * np.sin(theta)

        # 2. Joint friction: dq_eff = dq - c * sign(dq) * |dq|^0.8
        for i in range(6):
            dq_i = control[..., 2 + i]
            c_i = self.joint_friction[i]
            friction = c_i * np.sign(dq_i) * np.abs(dq_i) ** 0.8
            correction[..., 3 + i] = -friction

        # 3. Arm-base coupling: 팔의 관절 가속도가 베이스 heading에 반작용
        #    θ̇ += k_coupling * Σ|dq_i * I_i|
        coupling_torque = np.zeros(batch_shape) if batch_shape else 0.0
        for i in range(6):
            dq_i = control[..., 2 + i]
            coupling_torque = coupling_torque + np.abs(dq_i) * self.joint_inertia[i]
        correction[..., 2] += self.coupling_gain * coupling_torque

        # 4. Gravity droop: 어깨(q2)/팔꿈치(q3)에 중력 처짐
        #    q̇_2 -= g_droop * sin(q2)
        #    q̇_3 -= g_droop * sin(q2 + q3)
        q2 = state[..., 4]  # q2 (index 4 in 9D state)
        q3 = state[..., 5]  # q3 (index 5 in 9D state)
        correction[..., 4] -= self.gravity_droop * np.sin(q2)
        correction[..., 5] -= self.gravity_droop * np.sin(q2 + q3)

        return correction

    def get_physics_correction(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """물리 보정 벡터 반환 (디버깅/분석용)."""
        return self._compute_physics_correction(state, control)

    def __repr__(self) -> str:
        return (
            f"MobileManipulator6DOFDynamic("
            f"z_mount={self.z_mount}, "
            f"base_friction={self.base_friction}, "
            f"coupling_gain={self.coupling_gain}, "
            f"gravity_droop={self.gravity_droop})"
        )
