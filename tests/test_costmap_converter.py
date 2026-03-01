"""
CostmapConverter (costmap_converter.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.ros2.nav2.costmap_converter import CostmapConverter


# ── Tests ──────────────────────────────────────────────────


def test_empty_costmap():
    print("\n" + "=" * 60)
    print("Test: empty costmap (all zeros) returns empty list")
    print("=" * 60)

    cc = CostmapConverter(lethal_threshold=253)

    width, height = 20, 20
    resolution = 0.05
    data = np.zeros(width * height, dtype=np.int32)

    obstacles = cc.convert(data, width, height, resolution,
                           origin_x=0.0, origin_y=0.0,
                           robot_x=0.5, robot_y=0.5)

    assert len(obstacles) == 0, f"Expected 0 obstacles, got {len(obstacles)}"
    print("PASS")


def test_single_obstacle():
    print("\n" + "=" * 60)
    print("Test: single cluster of lethal cells produces one obstacle")
    print("=" * 60)

    cc = CostmapConverter(lethal_threshold=253, robot_radius=0.1,
                          safety_margin=0.0, min_cluster_size=2,
                          max_detection_range=10.0)

    width, height = 20, 20
    resolution = 0.05
    data = np.zeros(width * height, dtype=np.int32)

    # Place a 3x3 block of lethal cells at grid center (row=10, col=10)
    for r in range(9, 12):
        for c in range(9, 12):
            data[r * width + c] = 254

    obstacles = cc.convert(data, width, height, resolution,
                           origin_x=0.0, origin_y=0.0,
                           robot_x=0.5, robot_y=0.5)

    assert len(obstacles) == 1, f"Expected 1 obstacle, got {len(obstacles)}"
    x, y, radius = obstacles[0]
    assert radius > 0, f"Radius should be positive, got {radius}"
    print(f"  obstacle=({x:.3f}, {y:.3f}, r={radius:.3f})")
    print("PASS")


def test_two_separate_clusters():
    print("\n" + "=" * 60)
    print("Test: two separated groups produce two obstacles")
    print("=" * 60)

    cc = CostmapConverter(lethal_threshold=253, robot_radius=0.0,
                          safety_margin=0.0, min_cluster_size=2,
                          max_detection_range=20.0,
                          max_obstacle_radius=5.0,
                          cluster_resolution=0.05)

    width, height = 40, 40
    resolution = 0.05
    data = np.zeros(width * height, dtype=np.int32)

    # Cluster 1: top-left corner (rows 2-4, cols 2-4)
    for r in range(2, 5):
        for c in range(2, 5):
            data[r * width + c] = 254

    # Cluster 2: bottom-right corner (rows 35-37, cols 35-37)
    for r in range(35, 38):
        for c in range(35, 38):
            data[r * width + c] = 254

    obstacles = cc.convert(data, width, height, resolution,
                           origin_x=0.0, origin_y=0.0,
                           robot_x=1.0, robot_y=1.0)

    assert len(obstacles) == 2, f"Expected 2 obstacles, got {len(obstacles)}"
    # Verify they are at different locations
    d = np.sqrt((obstacles[0][0] - obstacles[1][0]) ** 2 +
                (obstacles[0][1] - obstacles[1][1]) ** 2)
    assert d > 1.0, f"Obstacles too close: {d:.3f}m"
    print(f"  obstacle_0=({obstacles[0][0]:.2f}, {obstacles[0][1]:.2f})")
    print(f"  obstacle_1=({obstacles[1][0]:.2f}, {obstacles[1][1]:.2f})")
    print(f"  distance={d:.3f}m")
    print("PASS")


def test_range_filtering():
    print("\n" + "=" * 60)
    print("Test: obstacle beyond max_detection_range is filtered out")
    print("=" * 60)

    cc = CostmapConverter(lethal_threshold=253, robot_radius=0.0,
                          safety_margin=0.0, min_cluster_size=2,
                          max_detection_range=1.0)

    width, height = 100, 100
    resolution = 0.1
    data = np.zeros(width * height, dtype=np.int32)

    # Place obstacle far from robot (at ~9m away)
    for r in range(90, 93):
        for c in range(90, 93):
            data[r * width + c] = 254

    # Robot at origin
    obstacles = cc.convert(data, width, height, resolution,
                           origin_x=0.0, origin_y=0.0,
                           robot_x=0.0, robot_y=0.0)

    assert len(obstacles) == 0, \
        f"Expected 0 obstacles (out of range), got {len(obstacles)}"
    print("PASS")


def test_max_radius_filtering():
    print("\n" + "=" * 60)
    print("Test: very large cluster filtered by max_obstacle_radius")
    print("=" * 60)

    cc = CostmapConverter(lethal_threshold=253, robot_radius=0.1,
                          safety_margin=0.05, max_obstacle_radius=0.3,
                          max_detection_range=20.0, min_cluster_size=2,
                          cluster_resolution=0.1)

    width, height = 50, 50
    resolution = 0.05
    data = np.zeros(width * height, dtype=np.int32)

    # Large 20x20 block (1m x 1m at resolution=0.05)
    for r in range(10, 30):
        for c in range(10, 30):
            data[r * width + c] = 254

    obstacles = cc.convert(data, width, height, resolution,
                           origin_x=0.0, origin_y=0.0,
                           robot_x=1.0, robot_y=1.0)

    assert len(obstacles) == 0, \
        f"Expected 0 obstacles (too large), got {len(obstacles)}"
    print("PASS")


def test_inflation():
    print("\n" + "=" * 60)
    print("Test: radius includes robot_radius + safety_margin")
    print("=" * 60)

    robot_radius = 0.22
    safety_margin = 0.05
    cc = CostmapConverter(lethal_threshold=253, robot_radius=robot_radius,
                          safety_margin=safety_margin, min_cluster_size=2,
                          max_detection_range=10.0, max_obstacle_radius=5.0)

    width, height = 20, 20
    resolution = 0.05
    data = np.zeros(width * height, dtype=np.int32)

    # Small 2x2 cluster
    for r in range(10, 12):
        for c in range(10, 12):
            data[r * width + c] = 254

    obstacles = cc.convert(data, width, height, resolution,
                           origin_x=0.0, origin_y=0.0,
                           robot_x=0.5, robot_y=0.5)

    assert len(obstacles) == 1, f"Expected 1 obstacle, got {len(obstacles)}"
    _, _, radius = obstacles[0]
    # Radius should be at least robot_radius + safety_margin + raw_radius
    min_expected = robot_radius + safety_margin
    assert radius >= min_expected, \
        f"Inflated radius {radius:.3f} < min expected {min_expected:.3f}"
    print(f"  inflated radius={radius:.3f} >= {min_expected:.3f}")
    print("PASS")


def test_world_coordinates():
    print("\n" + "=" * 60)
    print("Test: non-zero origin produces correct world frame coords")
    print("=" * 60)

    cc = CostmapConverter(lethal_threshold=253, robot_radius=0.0,
                          safety_margin=0.0, min_cluster_size=2,
                          max_detection_range=20.0, max_obstacle_radius=5.0)

    width, height = 20, 20
    resolution = 0.1
    origin_x, origin_y = 5.0, 3.0
    data = np.zeros(width * height, dtype=np.int32)

    # Place 2x2 cluster at grid (0,0)-(1,1)
    for r in range(0, 2):
        for c in range(0, 2):
            data[r * width + c] = 254

    obstacles = cc.convert(data, width, height, resolution,
                           origin_x=origin_x, origin_y=origin_y,
                           robot_x=origin_x, robot_y=origin_y)

    assert len(obstacles) == 1, f"Expected 1 obstacle, got {len(obstacles)}"
    x, y, _ = obstacles[0]

    # Expected world coords: origin + (col+0.5)*res, origin + (row+0.5)*res
    # Cluster at cols [0,1], rows [0,1] => centroid ~ (0.5, 0.5) in cells
    # World: (5.0 + 0.5*0.1, 3.0 + 0.5*0.1) = (5.05, 3.05) approximately
    assert abs(x - (origin_x + 0.5 * resolution)) < 0.15, \
        f"x={x:.3f}, expected near {origin_x + 0.5 * resolution:.3f}"
    assert abs(y - (origin_y + 0.5 * resolution)) < 0.15, \
        f"y={y:.3f}, expected near {origin_y + 0.5 * resolution:.3f}"
    print(f"  obstacle world pos=({x:.3f}, {y:.3f})")
    print("PASS")


def test_threshold():
    print("\n" + "=" * 60)
    print("Test: cells at lethal_threshold included, below excluded")
    print("=" * 60)

    threshold = 200
    cc = CostmapConverter(lethal_threshold=threshold, robot_radius=0.0,
                          safety_margin=0.0, min_cluster_size=2,
                          max_detection_range=10.0, max_obstacle_radius=5.0)

    width, height = 20, 20
    resolution = 0.05
    data = np.zeros(width * height, dtype=np.int32)

    # Cells exactly at threshold (should be included)
    for r in range(5, 8):
        for c in range(5, 8):
            data[r * width + c] = threshold

    # Cells just below threshold (should be excluded)
    for r in range(15, 18):
        for c in range(15, 18):
            data[r * width + c] = threshold - 1

    obstacles = cc.convert(data, width, height, resolution,
                           origin_x=0.0, origin_y=0.0,
                           robot_x=0.5, robot_y=0.5)

    assert len(obstacles) == 1, \
        f"Expected 1 obstacle (at threshold), got {len(obstacles)}"
    print(f"  threshold={threshold}: at-threshold included, below excluded")
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CostmapConverter Unit Tests")
    print("=" * 60)

    tests = [
        test_empty_costmap,
        test_single_obstacle,
        test_two_separate_clusters,
        test_range_filtering,
        test_max_radius_filtering,
        test_inflation,
        test_world_coordinates,
        test_threshold,
    ]

    try:
        for t in tests:
            t()
        print(f"\n{'=' * 60}")
        print(f"  All {len(tests)} Tests Passed!")
        print(f"{'=' * 60}")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
