"""Goal reached checker for nav2 FollowPath action.

Checks if the robot has reached the goal pose within xy and yaw tolerances.
Mirrors nav2 SimpleGoalChecker behavior.
"""

import numpy as np


class GoalChecker:
    """Check if robot has reached the goal pose."""

    def __init__(self, xy_tolerance: float = 0.25,
                 yaw_tolerance: float = 0.25,
                 stateful: bool = True):
        """
        Args:
            xy_tolerance: XY distance tolerance [m].
            yaw_tolerance: Yaw angle tolerance [rad].
            stateful: If True, once goal is reached it stays reached
                until reset().
        """
        self.xy_tolerance = xy_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.stateful = stateful
        self._goal_reached = False

    def is_goal_reached(self, current_state: np.ndarray,
                        goal_pose: np.ndarray) -> bool:
        """Check if robot is within tolerance of goal.

        Args:
            current_state: Robot state [x, y, theta, ...].
            goal_pose: Goal [x, y, theta].

        Returns:
            True if goal is reached.
        """
        if self.stateful and self._goal_reached:
            return True

        xy_dist = self.get_distance_to_goal(current_state, goal_pose)
        yaw_diff = abs(self._angle_diff(current_state[2], goal_pose[2]))

        reached = xy_dist <= self.xy_tolerance and yaw_diff <= self.yaw_tolerance
        if reached and self.stateful:
            self._goal_reached = True
        return reached

    def get_distance_to_goal(self, current_state: np.ndarray,
                             goal_pose: np.ndarray) -> float:
        """Euclidean XY distance to goal."""
        dx = current_state[0] - goal_pose[0]
        dy = current_state[1] - goal_pose[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def reset(self):
        """Reset stateful goal reached flag."""
        self._goal_reached = False

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Shortest signed angle difference a - b, wrapped to [-pi, pi]."""
        diff = a - b
        return float(np.arctan2(np.sin(diff), np.cos(diff)))
