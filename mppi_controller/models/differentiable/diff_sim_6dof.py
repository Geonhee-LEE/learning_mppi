"""
PyTorch 미분가능 6-DOF Mobile Manipulator 시뮬레이터

NumPy MobileManipulator6DOFKinematic/Dynamic을 PyTorch로 포팅하여
autograd를 통한 역전파(BPTT)를 지원.

State:  [x, y, θ, q1, q2, q3, q4, q5, q6] (9D)
Control: [v, ω, dq1, dq2, dq3, dq4, dq5, dq6] (8D)

Usage:
    sim = DifferentiableMobileManipulator6DOF()
    traj = sim.rollout(state_0, controls, dt=0.05)  # (N+1, 9)
    ee = sim.forward_kinematics(traj)                # (N+1, 3)
    loss = ((ee - ee_ref) ** 2).sum()
    loss.backward()  # gradients through rollout!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# UR5-style DH parameters: [a, d, alpha, theta_offset]
DEFAULT_DH_PARAMS = np.array([
    [0.0,  0.15,  np.pi / 2,  0.0],
    [0.30, 0.0,   0.0,        0.0],
    [0.25, 0.0,   0.0,        0.0],
    [0.0,  0.10,  np.pi / 2,  0.0],
    [0.0,  0.10, -np.pi / 2,  0.0],
    [0.0,  0.05,  0.0,        0.0],
], dtype=np.float64)

DEFAULT_JOINT_FRICTION = np.array([0.15, 0.12, 0.10, 0.08, 0.06, 0.05])
DEFAULT_JOINT_INERTIA = np.array([0.5, 0.4, 0.3, 0.15, 0.1, 0.05])


class DifferentiableMobileManipulator6DOF(nn.Module):
    """
    PyTorch 미분가능 6-DOF 시뮬레이터 (기구학).

    NumPy MobileManipulator6DOFKinematic과 동일 로직을 torch 텐서로 구현.
    float64로 동작하여 NumPy 모델과의 수치 일치 보장.

    Args:
        dh_params: (6, 4) DH 파라미터 [a, d, alpha, theta_offset]
        z_mount: 팔 마운트 높이 (m)
    """

    def __init__(
        self,
        dh_params: Optional[np.ndarray] = None,
        z_mount: float = 0.1,
    ):
        super().__init__()

        if dh_params is None:
            dh_params = DEFAULT_DH_PARAMS.copy()
        dh_params = np.asarray(dh_params, dtype=np.float64)
        assert dh_params.shape == (6, 4)

        self.z_mount = z_mount

        # Register DH parameters as buffers (not trainable, but move with .to())
        self.register_buffer("_a", torch.tensor(dh_params[:, 0], dtype=torch.float64))
        self.register_buffer("_d", torch.tensor(dh_params[:, 1], dtype=torch.float64))
        self.register_buffer("_cos_alpha", torch.tensor(np.cos(dh_params[:, 2]), dtype=torch.float64))
        self.register_buffer("_sin_alpha", torch.tensor(np.sin(dh_params[:, 2]), dtype=torch.float64))
        self.register_buffer("_theta_offset", torch.tensor(dh_params[:, 3], dtype=torch.float64))

    def forward_dynamics(
        self, state: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        """
        기구학 동역학: dx/dt = f(x, u)

        Args:
            state: (..., 9)
            control: (..., 8)

        Returns:
            state_dot: (..., 9)
        """
        theta = state[..., 2]
        v = control[..., 0]
        omega = control[..., 1]

        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = omega

        # Joint velocities
        joint_dots = [control[..., i] for i in range(2, 8)]

        return torch.stack(
            [x_dot, y_dot, theta_dot] + joint_dots, dim=-1
        )

    def step_euler(
        self, state: torch.Tensor, control: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Euler integration step."""
        return state + self.forward_dynamics(state, control) * dt

    def step_rk4(
        self, state: torch.Tensor, control: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """RK4 integration step (미분가능)."""
        k1 = self.forward_dynamics(state, control)
        k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.forward_dynamics(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _dh_transform(
        self, theta_i: torch.Tensor, i: int
    ) -> torch.Tensor:
        """
        Standard DH transform for joint i.

        Args:
            theta_i: (...,) joint angle
            i: joint index (0..5)

        Returns:
            T: (..., 4, 4)
        """
        q = theta_i + self._theta_offset[i]
        cq = torch.cos(q)
        sq = torch.sin(q)
        ca = self._cos_alpha[i]
        sa = self._sin_alpha[i]
        a = self._a[i]
        d = self._d[i]

        batch_shape = theta_i.shape
        T = torch.zeros(batch_shape + (4, 4), dtype=theta_i.dtype, device=theta_i.device)

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

    def _base_transform(self, state: torch.Tensor) -> torch.Tensor:
        """
        Base-to-mount transform.

        Args:
            state: (..., 9)

        Returns:
            T: (..., 4, 4)
        """
        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 2]

        batch_shape = x.shape
        T = torch.zeros(
            batch_shape + (4, 4), dtype=state.dtype, device=state.device
        )

        ct = torch.cos(theta)
        st = torch.sin(theta)

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

    def _dh_chain(self, q_joints: torch.Tensor) -> torch.Tensor:
        """
        Full DH chain: T_0^6 = T_1 @ T_2 @ ... @ T_6

        Args:
            q_joints: (..., 6)

        Returns:
            T: (..., 4, 4)
        """
        T = self._dh_transform(q_joints[..., 0], 0)
        for i in range(1, 6):
            T_i = self._dh_transform(q_joints[..., i], i)
            T = torch.einsum("...ij,...jk->...ik", T, T_i)
        return T

    def forward_kinematics(self, state: torch.Tensor) -> torch.Tensor:
        """
        End-Effector 3D position via FK (미분가능).

        Args:
            state: (..., 9)

        Returns:
            ee_pos: (..., 3)
        """
        T_base = self._base_transform(state)
        q_joints = state[..., 3:9]
        T_arm = self._dh_chain(q_joints)
        T_world = torch.einsum("...ij,...jk->...ik", T_base, T_arm)
        return T_world[..., :3, 3]

    def rollout(
        self,
        state_0: torch.Tensor,
        controls: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Differentiable trajectory rollout.

        Args:
            state_0: (..., 9) initial state
            controls: (..., N, 8) control sequence
            dt: time step

        Returns:
            trajectory: (..., N+1, 9)
        """
        # Determine N from controls shape
        N = controls.shape[-2]
        batch_shape = state_0.shape[:-1]

        states = [state_0]
        state = state_0

        for t in range(N):
            ctrl = controls[..., t, :]
            state = self.step_rk4(state, ctrl, dt)
            states.append(state)

        return torch.stack(states, dim=-2)


class DifferentiableMobileManipulator6DOFDynamic(DifferentiableMobileManipulator6DOF):
    """
    기구학 + 물리 보정 (마찰/중력/커플링).

    MobileManipulator6DOFDynamic의 PyTorch 포팅.

    Args:
        dh_params: DH 파라미터
        z_mount: 마운트 높이
        base_friction: 베이스 마찰 계수
        base_response_k: 비선형 응답 상수
        joint_friction: (6,) 관절 마찰 계수
        coupling_gain: 팔→베이스 커플링 게인
        gravity_droop: 중력 처짐 계수
        joint_inertia: (6,) 관절 관성
    """

    def __init__(
        self,
        dh_params: Optional[np.ndarray] = None,
        z_mount: float = 0.1,
        base_friction: float = 0.3,
        base_response_k: float = 5.0,
        joint_friction: Optional[np.ndarray] = None,
        coupling_gain: float = 0.02,
        gravity_droop: float = 0.08,
        joint_inertia: Optional[np.ndarray] = None,
    ):
        super().__init__(dh_params, z_mount)

        self.base_friction = base_friction
        self.base_response_k = base_response_k
        self.coupling_gain = coupling_gain
        self.gravity_droop = gravity_droop

        jf = joint_friction if joint_friction is not None else DEFAULT_JOINT_FRICTION.copy()
        ji = joint_inertia if joint_inertia is not None else DEFAULT_JOINT_INERTIA.copy()

        self.register_buffer("_joint_friction", torch.tensor(jf, dtype=torch.float64))
        self.register_buffer("_joint_inertia", torch.tensor(ji, dtype=torch.float64))

    def forward_dynamics(
        self, state: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        """동역학: f_kinematic + f_correction"""
        kin_dot = super().forward_dynamics(state, control)
        correction = self._compute_physics_correction(state, control)
        return kin_dot + correction

    def _compute_physics_correction(
        self, state: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        """물리 보정 벡터 (미분가능)."""
        correction = torch.zeros_like(state)

        v = control[..., 0]
        theta = state[..., 2]

        # 1. Base velocity lag
        response = 1.0 - torch.exp(-self.base_response_k * torch.abs(v))
        v_eff = v * response
        v_correction = v_eff - v

        correction[..., 0] = v_correction * torch.cos(theta)
        correction[..., 1] = v_correction * torch.sin(theta)

        # 2. Joint friction: dq_eff = dq - c * sign(dq) * |dq|^0.8
        for i in range(6):
            dq_i = control[..., 2 + i]
            c_i = self._joint_friction[i]
            friction = c_i * torch.sign(dq_i) * torch.abs(dq_i) ** 0.8
            correction[..., 3 + i] = -friction

        # 3. Arm-base coupling
        coupling_torque = torch.zeros_like(v)
        for i in range(6):
            dq_i = control[..., 2 + i]
            coupling_torque = coupling_torque + torch.abs(dq_i) * self._joint_inertia[i]
        correction[..., 2] = correction[..., 2] + self.coupling_gain * coupling_torque

        # 4. Gravity droop
        q2 = state[..., 4]
        q3 = state[..., 5]
        correction[..., 4] = correction[..., 4] - self.gravity_droop * torch.sin(q2)
        correction[..., 5] = correction[..., 5] - self.gravity_droop * torch.sin(q2 + q3)

        return correction
