"""
Ackermann Steering Kinematic Model (Bicycle Model)

State: [x, y, theta, delta] (4D)
Control: [v, phi] (2D) - velocity, steering rate

Dynamics:
    dx/dt = v * cos(theta)
    dy/dt = v * sin(theta)
    dtheta/dt = v * tan(delta) / L
    ddelta/dt = phi
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from typing import Optional, Tuple


class AckermannKinematic(RobotModel):
    """
    Ackermann Steering Kinematic Model (Bicycle Approximation)

    Uses the bicycle model where front and rear axle are collapsed into
    a single-track representation with wheelbase L.

    State:
        x (m): world frame x position
        y (m): world frame y position
        theta (rad): heading angle
        delta (rad): front steering angle

    Control:
        v (m/s): longitudinal velocity
        phi (rad/s): steering rate

    Args:
        wheelbase: distance between front and rear axle (m)
        v_max: maximum velocity (m/s)
        max_steer: maximum steering angle (rad), default 0.5 (~29 deg)
        steer_rate_max: maximum steering rate (rad/s)
    """

    def __init__(
        self,
        wheelbase: float = 0.5,
        v_max: float = 1.0,
        max_steer: float = 0.5,
        steer_rate_max: float = 1.0,
    ):
        self.wheelbase = wheelbase
        self.v_max = v_max
        self.max_steer = max_steer
        self.steer_rate_max = steer_rate_max

        # Control bounds: [v, phi]
        self._control_lower = np.array([-v_max, -steer_rate_max])
        self._control_upper = np.array([v_max, steer_rate_max])

    @property
    def state_dim(self) -> int:
        return 4  # [x, y, theta, delta]

    @property
    def control_dim(self) -> int:
        return 2  # [v, phi]

    @property
    def model_type(self) -> str:
        return "kinematic"

    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        Kinematic dynamics: dx/dt = f(x, u)

        Args:
            state: (4,) or (batch, 4) - [x, y, theta, delta]
            control: (2,) or (batch, 2) - [v, phi]

        Returns:
            state_dot: (4,) or (batch, 4)
        """
        theta = state[..., 2]
        delta = state[..., 3]
        v = control[..., 0]
        phi = control[..., 1]

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = v * np.tan(delta) / self.wheelbase
        delta_dot = phi

        return np.stack([x_dot, y_dot, theta_dot, delta_dot], axis=-1)

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return control bounds [v, phi]"""
        return (self._control_lower, self._control_upper)

    def state_to_dict(self, state: np.ndarray) -> dict:
        """Convert state to dict for debugging."""
        return {
            "x": state[0],
            "y": state[1],
            "theta": state[2],
            "delta": state[3],
        }

    def render_config(self) -> dict:
        """Car body (Ackermann)."""
        return {
            "shape": "car",
            "length": self.wheelbase * 1.2,
            "width": self.wheelbase * 0.5,
            "color": "#2ca02c",
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state:
        - theta to [-pi, pi]
        - delta clipped to [-max_steer, max_steer]
        """
        normalized = state.copy()
        normalized[..., 2] = np.arctan2(
            np.sin(state[..., 2]), np.cos(state[..., 2])
        )
        normalized[..., 3] = np.clip(
            state[..., 3], -self.max_steer, self.max_steer
        )
        return normalized

    def __repr__(self) -> str:
        return (
            f"AckermannKinematic("
            f"wheelbase={self.wheelbase}, v_max={self.v_max}, "
            f"max_steer={self.max_steer})"
        )
