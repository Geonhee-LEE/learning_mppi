"""Global path → local MPPI reference trajectory extraction.

Extracts a (N+1, state_dim) local reference window from a global path,
handling closest-point warm-start, lookahead, linear interpolation,
angle wrapping, and end-of-path padding.
"""

import numpy as np


class PathWindower:
    """Extract a local reference window from a global path for MPPI."""

    def __init__(self, horizon: int, dt: float,
                 lookahead_distance: float = 0.1,
                 state_dim: int = 3):
        """
        Args:
            horizon: MPPI horizon length N (output will be N+1 poses).
            dt: MPPI time step [s].
            lookahead_distance: Advance past closest point to avoid
                tracking points already behind the robot [m].
            state_dim: State dimension (3 for kinematic, 5 for dynamic).
        """
        self.horizon = horizon
        self.dt = dt
        self.lookahead_distance = lookahead_distance
        self.state_dim = state_dim
        self._closest_idx = 0  # warm-start index

    def extract_reference(self, global_path: np.ndarray,
                          robot_state: np.ndarray):
        """Extract (N+1, state_dim) reference from global path.

        Args:
            global_path: (M, 3) array of [x, y, theta] waypoints.
            robot_state: Current robot state (at least [x, y, theta]).

        Returns:
            reference: (N+1, state_dim) local reference trajectory.
            closest_idx: Index of the closest point in global_path.
        """
        if global_path.shape[0] == 0:
            ref = np.tile(robot_state[:self.state_dim],
                          (self.horizon + 1, 1))
            return ref, 0

        # 1. Find closest point with warm-start window
        closest_idx = self._find_closest(global_path, robot_state)
        self._closest_idx = closest_idx

        # 2. Advance by lookahead_distance
        start_idx = self._advance_by_distance(
            global_path, closest_idx, self.lookahead_distance)

        # 3. Compute arc-length distances from start_idx
        path_from_start = global_path[start_idx:]
        if len(path_from_start) < 2:
            ref = np.tile(global_path[-1], (self.horizon + 1, 1))
            return self._pad_state_dim(ref), closest_idx

        arc_lengths = self._compute_arc_lengths(path_from_start)

        # 4. Interpolate N+1 poses at uniform dt spacing
        desired_distances = np.arange(self.horizon + 1) * self.dt * self._estimate_speed(global_path, closest_idx)
        # Minimum spacing: use path resolution
        if desired_distances[-1] < 1e-6:
            desired_distances = np.linspace(0, arc_lengths[-1],
                                            self.horizon + 1)

        reference = self._interpolate_path(
            path_from_start, arc_lengths, desired_distances)

        return self._pad_state_dim(reference), closest_idx

    def _find_closest(self, global_path: np.ndarray,
                      robot_state: np.ndarray) -> int:
        """Find closest point with forward-biased warm-start search."""
        rx, ry = robot_state[0], robot_state[1]
        M = len(global_path)

        # Search window: from previous closest to end
        search_start = max(0, self._closest_idx - 5)
        search_end = min(M, search_start + max(100, M - search_start))

        dists = ((global_path[search_start:search_end, 0] - rx) ** 2 +
                 (global_path[search_start:search_end, 1] - ry) ** 2)
        local_idx = np.argmin(dists)
        return search_start + int(local_idx)

    def _advance_by_distance(self, global_path: np.ndarray,
                             start_idx: int,
                             distance: float) -> int:
        """Advance along path by a given distance from start_idx."""
        accumulated = 0.0
        idx = start_idx
        M = len(global_path)
        while idx < M - 1 and accumulated < distance:
            dx = global_path[idx + 1, 0] - global_path[idx, 0]
            dy = global_path[idx + 1, 1] - global_path[idx, 1]
            accumulated += np.sqrt(dx * dx + dy * dy)
            idx += 1
        return idx

    def _compute_arc_lengths(self, path: np.ndarray) -> np.ndarray:
        """Compute cumulative arc-length distances along path."""
        diffs = np.diff(path[:, :2], axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        arc_lengths = np.zeros(len(path))
        arc_lengths[1:] = np.cumsum(segment_lengths)
        return arc_lengths

    def _estimate_speed(self, global_path: np.ndarray,
                        idx: int) -> float:
        """Estimate desired speed from path point spacing and dt."""
        if idx >= len(global_path) - 1:
            idx = len(global_path) - 2
        if idx < 0:
            return 0.3  # default
        dx = global_path[idx + 1, 0] - global_path[idx, 0]
        dy = global_path[idx + 1, 1] - global_path[idx, 1]
        point_dist = np.sqrt(dx * dx + dy * dy)
        # Assume points are spaced at some nominal interval
        speed = max(point_dist / self.dt, 0.1)
        return min(speed, 1.0)  # cap at 1 m/s

    def _interpolate_path(self, path: np.ndarray,
                          arc_lengths: np.ndarray,
                          desired_distances: np.ndarray) -> np.ndarray:
        """Interpolate path poses at desired arc-length distances."""
        N1 = len(desired_distances)
        reference = np.zeros((N1, 3))
        total_length = arc_lengths[-1]

        # Clamp desired distances to path length
        clamped = np.clip(desired_distances, 0, total_length)

        # x, y: linear interpolation
        reference[:, 0] = np.interp(clamped, arc_lengths, path[:, 0])
        reference[:, 1] = np.interp(clamped, arc_lengths, path[:, 1])

        # theta: angle-aware interpolation using sin/cos decomposition
        sin_theta = np.sin(path[:, 2])
        cos_theta = np.cos(path[:, 2])
        interp_sin = np.interp(clamped, arc_lengths, sin_theta)
        interp_cos = np.interp(clamped, arc_lengths, cos_theta)
        reference[:, 2] = np.arctan2(interp_sin, interp_cos)

        return reference

    def _pad_state_dim(self, reference: np.ndarray) -> np.ndarray:
        """Pad reference to state_dim if needed (e.g., 3→5 for dynamic)."""
        N1 = self.horizon + 1
        # Ensure correct length
        if len(reference) < N1:
            pad = np.tile(reference[-1:], (N1 - len(reference), 1))
            reference = np.vstack([reference, pad])
        elif len(reference) > N1:
            reference = reference[:N1]

        if reference.shape[1] >= self.state_dim:
            return reference[:, :self.state_dim]

        # Pad extra dims with zeros (v, omega for dynamic model)
        padded = np.zeros((N1, self.state_dim))
        padded[:, :reference.shape[1]] = reference
        return padded

    def reset(self):
        """Reset warm-start index (e.g., on new path)."""
        self._closest_idx = 0
