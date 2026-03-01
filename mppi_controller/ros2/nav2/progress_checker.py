"""Progress checker for stuck detection in nav2 FollowPath action.

Monitors whether the robot is making sufficient forward progress.
If the robot fails to move required_movement meters within time_allowance
seconds, it is considered stuck.
"""

import numpy as np


class ProgressChecker:
    """Detect if robot is stuck (insufficient movement over time)."""

    def __init__(self, required_movement: float = 0.5,
                 time_allowance: float = 10.0):
        """
        Args:
            required_movement: Minimum distance to move within
                time_allowance to be considered making progress [m].
            time_allowance: Time window for progress check [s].
        """
        self.required_movement = required_movement
        self.time_allowance = time_allowance
        self._reference_position = None
        self._reference_time = None

    def check_progress(self, current_state: np.ndarray,
                       current_time: float) -> bool:
        """Check if robot is making progress.

        Args:
            current_state: Robot state [x, y, ...].
            current_time: Current time [s].

        Returns:
            True if robot is making progress (not stuck).
        """
        current_pos = current_state[:2].copy()

        # First call: initialize reference
        if self._reference_position is None:
            self._reference_position = current_pos
            self._reference_time = current_time
            return True

        # Check distance from reference
        dx = current_pos[0] - self._reference_position[0]
        dy = current_pos[1] - self._reference_position[1]
        dist = np.sqrt(dx * dx + dy * dy)

        # If moved enough, reset reference
        if dist >= self.required_movement:
            self._reference_position = current_pos
            self._reference_time = current_time
            return True

        # Check time elapsed
        elapsed = current_time - self._reference_time
        if elapsed > self.time_allowance:
            return False  # Stuck!

        return True  # Still within time allowance

    def reset(self):
        """Reset progress checker state."""
        self._reference_position = None
        self._reference_time = None
