"""
Swerve Drive Kinematic Model (Omnidirectional)

State: [x, y, theta] (3D)
Control: [vx, vy, omega] (3D) - body-frame velocities

Dynamics (body-to-world rotation):
    dx/dt = vx * cos(theta) - vy * sin(theta)
    dy/dt = vx * sin(theta) + vy * cos(theta)
    dtheta/dt = omega
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class SwerveDriveKinematic(RobotModel):
    """
    Swerve Drive Kinematic Model (Omnidirectional)

    A holonomic robot that can translate and rotate independently.
    Control inputs are body-frame velocities transformed to world frame.

    State:
        x (m): world frame x position
        y (m): world frame y position
        theta (rad): heading angle

    Control:
        vx (m/s): body-frame x velocity (forward)
        vy (m/s): body-frame y velocity (lateral)
        omega (rad/s): angular velocity

    Args:
        vx_max: maximum forward velocity (m/s)
        vy_max: maximum lateral velocity (m/s)
        omega_max: maximum angular velocity (rad/s)
    """

    def __init__(
        self,
        vx_max: float = 1.0,
        vy_max: float = 1.0,
        omega_max: float = 1.0,
    ):
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.omega_max = omega_max

        # Control bounds: [vx, vy, omega]
        self._control_lower = np.array([-vx_max, -vy_max, -omega_max])
        self._control_upper = np.array([vx_max, vy_max, omega_max])

    @property
    def state_dim(self) -> int:
        return 3  # [x, y, theta]

    @property
    def control_dim(self) -> int:
        return 3  # [vx, vy, omega]

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        Kinematic dynamics with body-to-world rotation.

        Args:
            state: (3,) or (batch, 3) - [x, y, theta]
            control: (3,) or (batch, 3) - [vx, vy, omega]

        Returns:
            state_dot: (3,) or (batch, 3)
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

        return np.stack([x_dot, y_dot, theta_dot], axis=-1)

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return control bounds [vx, vy, omega]"""
        return (self._control_lower, self._control_upper)

    def state_to_dict(self, state: np.ndarray) -> dict:
        """Convert state to dict for debugging."""
        return {
            "x": state[0],
            "y": state[1],
            "theta": state[2],
        }

    def render_config(self) -> dict:
        """Rectangle body (Swerve)."""
        return {
            "shape": "rectangle",
            "length": 0.4,
            "width": 0.35,
            "color": "#ff7f0e",
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize theta to [-pi, pi]."""
        normalized = state.copy()
        normalized[..., 2] = np.arctan2(
            np.sin(state[..., 2]), np.cos(state[..., 2])
        )
        return normalized

    def __repr__(self) -> str:
        return (
            f"SwerveDriveKinematic("
            f"vx_max={self.vx_max}, vy_max={self.vy_max}, "
            f"omega_max={self.omega_max})"
        )
