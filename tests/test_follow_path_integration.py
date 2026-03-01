"""Integration test for MPPI FollowPath pipeline.

Tests the full pipeline without ROS2:
PathWindower + CostmapConverter + GoalChecker + ProgressChecker + MPPI controller.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mppi_controller.ros2.nav2.path_windower import PathWindower
from mppi_controller.ros2.nav2.costmap_converter import CostmapConverter
from mppi_controller.ros2.nav2.goal_checker import GoalChecker
from mppi_controller.ros2.nav2.progress_checker import ProgressChecker
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams


def _make_straight_path(start, end, num_points=50):
    """Generate a straight-line path from start to end."""
    path = np.zeros((num_points, 3))
    path[:, 0] = np.linspace(start[0], end[0], num_points)
    path[:, 1] = np.linspace(start[1], end[1], num_points)
    path[:, 2] = np.arctan2(end[1] - start[1], end[0] - start[0])
    return path


def _make_circle_path(center=(0, 0), radius=3.0, num_points=100):
    """Generate a circular path."""
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    path = np.zeros((num_points, 3))
    path[:, 0] = center[0] + radius * np.cos(t)
    path[:, 1] = center[1] + radius * np.sin(t)
    path[:, 2] = t + np.pi / 2  # tangent direction
    path[:, 2] = np.arctan2(np.sin(path[:, 2]), np.cos(path[:, 2]))
    return path


class TestFollowPathIntegration:
    """Full pipeline integration tests (no ROS2)."""

    def setup_method(self):
        """Set up common test fixtures."""
        self.model = DifferentialDriveKinematic(v_max=0.5, omega_max=1.9)
        self.params = MPPIParams(
            N=20, dt=0.05, K=128, lambda_=1.0,
            sigma=np.array([0.3, 0.3]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            Qf=np.array([20.0, 20.0, 2.0]),
        )
        self.controller = MPPIController(self.model, self.params)

        self.windower = PathWindower(
            horizon=20, dt=0.05, state_dim=3)
        self.costmap_converter = CostmapConverter()
        self.goal_checker = GoalChecker(
            xy_tolerance=0.3, yaw_tolerance=0.5)
        self.progress_checker = ProgressChecker(
            required_movement=0.5, time_allowance=10.0)

    def test_straight_line_goal_reached(self):
        """Test full pipeline: straight path → MPPI → goal reached."""
        path = _make_straight_path([0, 0], [3, 0], num_points=80)
        state = np.array([0.0, 0.0, 0.0])
        goal = path[-1]

        dt = 0.05
        max_steps = 500

        for step in range(max_steps):
            # Check goal
            if self.goal_checker.is_goal_reached(state, goal):
                break

            # Extract reference
            reference, _ = self.windower.extract_reference(path, state)

            # Compute control
            control, info = self.controller.compute_control(state, reference)

            # Step model
            state = self.model.step(state, control, dt)

        dist = self.goal_checker.get_distance_to_goal(state, goal)
        assert dist < 1.0, f"Robot too far from goal: {dist:.2f}m"

    def test_costmap_to_obstacles_pipeline(self):
        """Test costmap → obstacles → controller update pipeline."""
        # Create a simple costmap with an obstacle
        width, height = 40, 40
        resolution = 0.1
        origin_x, origin_y = -2.0, -2.0

        data = np.zeros(width * height, dtype=np.int16)
        # Place obstacle cluster at cells (25,20)-(28,23) = world (0.5,0)-(0.8,0.3)
        for row in range(20, 24):
            for col in range(25, 29):
                data[row * width + col] = 254

        obstacles = self.costmap_converter.convert(
            data, width, height, resolution,
            origin_x, origin_y, 0.0, 0.0)

        assert len(obstacles) >= 1, "Should detect at least one obstacle"
        ox, oy, oradius = obstacles[0]
        assert 0.0 < ox < 1.5, f"Obstacle X should be near 0.5-0.8, got {ox}"
        assert -0.5 < oy < 0.8, f"Obstacle Y should be near 0-0.3, got {oy}"
        assert oradius > 0.22, "Radius should include robot_radius inflation"

    def test_progress_checker_integration(self):
        """Test stuck detection during navigation."""
        path = _make_straight_path([0, 0], [5, 0])
        state = np.array([0.0, 0.0, 0.0])
        goal = path[-1]

        # Simulate robot stuck (not moving)
        for step in range(300):
            t = step * 0.05
            progress = self.progress_checker.check_progress(state, t)

            if not progress:
                # Stuck detected
                assert t > 5.0, "Should take at least 5s to detect stuck"
                return

        assert False, "Should have detected stuck robot"

    def test_windower_with_controller(self):
        """Test PathWindower output is compatible with MPPI controller."""
        path = _make_circle_path(radius=3.0, num_points=200)
        state = np.array([3.0, 0.0, np.pi / 2])

        reference, closest_idx = self.windower.extract_reference(path, state)

        # Verify shape
        assert reference.shape == (21, 3), f"Expected (21,3), got {reference.shape}"

        # Should be usable by controller
        control, info = self.controller.compute_control(state, reference)
        assert control.shape == (2,), f"Control shape: {control.shape}"
        assert np.all(np.isfinite(control)), "Control should be finite"

    def test_goal_checker_with_model_stepping(self):
        """Test GoalChecker with model integration."""
        goal = np.array([1.0, 0.0, 0.0])
        state = np.array([0.9, 0.05, 0.01])

        # Should be within tolerance
        assert self.goal_checker.is_goal_reached(state, goal)

        # Reset and check far state
        self.goal_checker.reset()
        far_state = np.array([0.0, 0.0, 0.0])
        assert not self.goal_checker.is_goal_reached(far_state, goal)

    def test_full_pipeline_components_init(self):
        """Test all pipeline components can be initialized together."""
        assert self.model is not None
        assert self.controller is not None
        assert self.windower is not None
        assert self.costmap_converter is not None
        assert self.goal_checker is not None
        assert self.progress_checker is not None

    def test_empty_costmap_no_obstacles(self):
        """Test pipeline with empty costmap (no obstacles)."""
        data = np.zeros(100, dtype=np.int16)
        obstacles = self.costmap_converter.convert(
            data, 10, 10, 0.1, 0.0, 0.0)
        assert len(obstacles) == 0

    def test_multiple_control_cycles(self):
        """Test running multiple MPPI control cycles in sequence."""
        path = _make_straight_path([0, 0], [2, 0], num_points=60)
        state = np.array([0.0, 0.0, 0.0])

        for _ in range(10):
            reference, _ = self.windower.extract_reference(path, state)
            control, info = self.controller.compute_control(state, reference)
            state = self.model.step(state, control, 0.05)

        # Robot should have moved forward
        assert state[0] > 0.01, f"Robot should move forward, x={state[0]:.4f}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
