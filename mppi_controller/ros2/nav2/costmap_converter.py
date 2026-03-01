"""OccupancyGrid → circle obstacle list converter.

Converts a nav2 costmap (2D occupancy grid) into a list of (x, y, radius)
circle obstacles compatible with MPPI ObstacleCost / CBF / Shield controllers.
Uses grid-based connected component clustering (O(N)) for speed.
"""

import numpy as np


class CostmapConverter:
    """Convert costmap occupied cells to circle obstacle list."""

    def __init__(self, lethal_threshold: int = 253,
                 robot_radius: float = 0.22,
                 safety_margin: float = 0.05,
                 max_obstacle_radius: float = 2.0,
                 max_detection_range: float = 5.0,
                 min_cluster_size: int = 2,
                 cluster_resolution: float = 0.1):
        """
        Args:
            lethal_threshold: Costmap values >= this are occupied (253=LETHAL).
            robot_radius: Robot footprint radius for inflation [m].
            safety_margin: Extra margin added to obstacle radius [m].
            max_obstacle_radius: Ignore clusters larger than this [m].
            max_detection_range: Ignore obstacles beyond this range [m].
            min_cluster_size: Minimum cells in a cluster to be an obstacle.
            cluster_resolution: Grid resolution for clustering [m].
        """
        self.lethal_threshold = lethal_threshold
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.max_obstacle_radius = max_obstacle_radius
        self.max_detection_range = max_detection_range
        self.min_cluster_size = min_cluster_size
        self.cluster_resolution = cluster_resolution

    def convert(self, data: np.ndarray, width: int, height: int,
                resolution: float, origin_x: float, origin_y: float,
                robot_x: float = 0.0, robot_y: float = 0.0):
        """Convert costmap data to circle obstacle list.

        Args:
            data: 1D array of costmap values (row-major, size=width*height).
            width: Costmap width in cells.
            height: Costmap height in cells.
            resolution: Cell size [m/cell].
            origin_x: Costmap origin X in world frame [m].
            origin_y: Costmap origin Y in world frame [m].
            robot_x: Robot X position for range filtering [m].
            robot_y: Robot Y position for range filtering [m].

        Returns:
            List of (x, y, radius) tuples in world frame.
        """
        # 1. Find occupied cells
        grid = np.array(data).reshape(height, width)
        occupied_mask = grid >= self.lethal_threshold
        rows, cols = np.where(occupied_mask)

        if len(rows) == 0:
            return []

        # 2. Convert cell indices to world coordinates
        world_x = origin_x + (cols + 0.5) * resolution
        world_y = origin_y + (rows + 0.5) * resolution
        points = np.column_stack([world_x, world_y])

        # 3. Range filter
        dists_to_robot = np.sqrt(
            (points[:, 0] - robot_x) ** 2 +
            (points[:, 1] - robot_y) ** 2)
        in_range = dists_to_robot <= self.max_detection_range
        points = points[in_range]

        if len(points) == 0:
            return []

        # 4. Grid-based connected component clustering
        clusters = self._grid_cluster(points)

        # 5. Fit enclosing circles
        obstacles = []
        for cluster_points in clusters:
            if len(cluster_points) < self.min_cluster_size:
                continue

            cx, cy, radius = self._fit_enclosing_circle(cluster_points)

            # Inflate by robot radius + safety margin
            inflated_radius = radius + self.robot_radius + self.safety_margin

            if inflated_radius > self.max_obstacle_radius:
                continue

            # Range check on cluster center
            dist = np.sqrt((cx - robot_x) ** 2 + (cy - robot_y) ** 2)
            if dist > self.max_detection_range:
                continue

            obstacles.append((float(cx), float(cy), float(inflated_radius)))

        return obstacles

    def _grid_cluster(self, points: np.ndarray):
        """Grid-based connected component clustering (O(N)).

        Assigns each point to a grid cell, then groups adjacent cells
        using union-find for fast O(N) clustering.
        """
        res = self.cluster_resolution
        # Quantize points to grid cells
        cell_x = np.floor(points[:, 0] / res).astype(int)
        cell_y = np.floor(points[:, 1] / res).astype(int)

        # Build cell → point indices map
        cell_to_indices = {}
        for i in range(len(points)):
            key = (cell_x[i], cell_y[i])
            if key not in cell_to_indices:
                cell_to_indices[key] = []
            cell_to_indices[key].append(i)

        # Union-Find
        parent = {}

        def find(k):
            while parent.get(k, k) != k:
                parent[k] = parent.get(parent[k], parent[k])
                k = parent[k]
            return k

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Connect adjacent cells (8-connectivity)
        for (cx, cy) in cell_to_indices:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in cell_to_indices:
                        union((cx, cy), neighbor)

        # Group points by cluster root
        cluster_map = {}
        for key, indices in cell_to_indices.items():
            root = find(key)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].extend(indices)

        return [points[indices] for indices in cluster_map.values()]

    @staticmethod
    def _fit_enclosing_circle(cluster_points: np.ndarray):
        """Fit minimum enclosing circle (centroid + max distance)."""
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        radius = float(np.max(distances))
        radius = max(radius, 0.05)  # minimum radius
        return float(centroid[0]), float(centroid[1]), radius
