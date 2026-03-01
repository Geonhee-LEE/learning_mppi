"""
PathWindower (path_windower.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.ros2.nav2.path_windower import PathWindower


# ── Tests ──────────────────────────────────────────────────


def test_basic_extraction():
    print("\n" + "=" * 60)
    print("Test: basic extraction shape (N+1, state_dim)")
    print("=" * 60)

    horizon = 20
    dt = 0.05
    pw = PathWindower(horizon=horizon, dt=dt, lookahead_distance=0.0, state_dim=3)

    # Straight path along x-axis, 100 points spaced 0.1m apart
    M = 100
    path = np.zeros((M, 3))
    path[:, 0] = np.linspace(0, 10, M)
    path[:, 2] = 0.0  # heading = 0

    robot_state = np.array([0.0, 0.0, 0.0])
    ref, closest_idx = pw.extract_reference(path, robot_state)

    assert ref.shape == (horizon + 1, 3), f"Expected ({horizon+1}, 3), got {ref.shape}"
    assert isinstance(closest_idx, int), f"closest_idx should be int, got {type(closest_idx)}"
    print(f"  ref shape={ref.shape}, closest_idx={closest_idx}")
    print("PASS")


def test_empty_path():
    print("\n" + "=" * 60)
    print("Test: empty global_path fills with robot_state")
    print("=" * 60)

    horizon = 10
    pw = PathWindower(horizon=horizon, dt=0.05, state_dim=3)

    empty_path = np.zeros((0, 3))
    robot_state = np.array([1.0, 2.0, 0.5])
    ref, closest_idx = pw.extract_reference(empty_path, robot_state)

    assert ref.shape == (horizon + 1, 3), f"shape: {ref.shape}"
    assert closest_idx == 0
    # All rows should be robot_state
    for i in range(horizon + 1):
        assert np.allclose(ref[i], robot_state), \
            f"Row {i} mismatch: {ref[i]} vs {robot_state}"
    print(f"  All {horizon+1} rows match robot_state")
    print("PASS")


def test_single_point_path():
    print("\n" + "=" * 60)
    print("Test: single-point path fills all same pose")
    print("=" * 60)

    horizon = 15
    pw = PathWindower(horizon=horizon, dt=0.05, state_dim=3)

    single_path = np.array([[3.0, 4.0, 1.0]])
    robot_state = np.array([3.0, 4.0, 1.0])
    ref, closest_idx = pw.extract_reference(single_path, robot_state)

    assert ref.shape == (horizon + 1, 3), f"shape: {ref.shape}"
    # All rows should be the single point
    for i in range(horizon + 1):
        assert np.allclose(ref[i, :3], single_path[0], atol=1e-6), \
            f"Row {i}: {ref[i]} != {single_path[0]}"
    print("PASS")


def test_closest_point_tracking():
    print("\n" + "=" * 60)
    print("Test: closest_idx tracks robot position along path")
    print("=" * 60)

    horizon = 10
    pw = PathWindower(horizon=horizon, dt=0.05, lookahead_distance=0.0, state_dim=3)

    M = 50
    path = np.zeros((M, 3))
    path[:, 0] = np.linspace(0, 5, M)

    # Robot at start
    _, idx0 = pw.extract_reference(path, np.array([0.0, 0.0, 0.0]))
    pw.reset()

    # Robot at mid-path
    _, idx_mid = pw.extract_reference(path, np.array([2.5, 0.0, 0.0]))
    pw.reset()

    # Robot near end
    _, idx_end = pw.extract_reference(path, np.array([4.8, 0.0, 0.0]))

    assert idx0 < idx_mid < idx_end, \
        f"Indices should increase: {idx0} < {idx_mid} < {idx_end}"
    print(f"  idx_start={idx0}, idx_mid={idx_mid}, idx_end={idx_end}")
    print("PASS")


def test_warm_start():
    print("\n" + "=" * 60)
    print("Test: warm-start _closest_idx advances across calls")
    print("=" * 60)

    horizon = 10
    pw = PathWindower(horizon=horizon, dt=0.05, lookahead_distance=0.0, state_dim=3)

    M = 100
    path = np.zeros((M, 3))
    path[:, 0] = np.linspace(0, 10, M)

    # First call at start
    _, idx1 = pw.extract_reference(path, np.array([0.0, 0.0, 0.0]))
    assert pw._closest_idx == idx1

    # Second call further along - warm start should advance
    _, idx2 = pw.extract_reference(path, np.array([3.0, 0.0, 0.0]))
    assert idx2 >= idx1, f"warm-start should advance: {idx2} >= {idx1}"
    assert pw._closest_idx == idx2
    print(f"  idx1={idx1}, idx2={idx2}")
    print("PASS")


def test_angle_wrapping():
    print("\n" + "=" * 60)
    print("Test: angle wrapping across +/- pi boundary")
    print("=" * 60)

    horizon = 20
    pw = PathWindower(horizon=horizon, dt=0.05, lookahead_distance=0.0, state_dim=3)

    # Path that crosses pi/-pi boundary
    M = 40
    path = np.zeros((M, 3))
    path[:, 0] = np.linspace(0, 4, M)
    # Angles go from 2.5 to 4.0 (crossing pi=3.14...)
    raw_angles = np.linspace(2.5, 4.0, M)
    path[:, 2] = np.arctan2(np.sin(raw_angles), np.cos(raw_angles))

    robot_state = np.array([0.0, 0.0, 2.5])
    ref, _ = pw.extract_reference(path, robot_state)

    # Check no large angle jumps (discontinuity)
    angle_diffs = np.abs(np.diff(ref[:, 2]))
    # Wrap diffs
    angle_diffs = np.abs(np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs)))
    max_jump = np.max(angle_diffs)

    assert max_jump < 1.0, f"Angle discontinuity detected: max_jump={max_jump:.3f} rad"
    print(f"  max angle jump={max_jump:.4f} rad (< 1.0 rad)")
    print("PASS")


def test_end_of_path_padding():
    print("\n" + "=" * 60)
    print("Test: end-of-path padding with last pose")
    print("=" * 60)

    horizon = 30
    pw = PathWindower(horizon=horizon, dt=0.05, lookahead_distance=0.0, state_dim=3)

    # Short path - only 5 points
    M = 5
    path = np.zeros((M, 3))
    path[:, 0] = np.linspace(0, 0.4, M)
    last_pose = path[-1].copy()

    # Robot near the end
    robot_state = np.array([0.35, 0.0, 0.0])
    ref, _ = pw.extract_reference(path, robot_state)

    assert ref.shape == (horizon + 1, 3), f"shape: {ref.shape}"
    # Last several rows should be at/near the final path pose
    # (because path runs out and gets padded)
    end_ref = ref[-1]
    assert np.allclose(end_ref[:2], last_pose[:2], atol=0.1), \
        f"End reference {end_ref} should be near last pose {last_pose}"
    print(f"  ref[-1]={ref[-1]}, last_pose={last_pose}")
    print("PASS")


def test_state_dim_5():
    print("\n" + "=" * 60)
    print("Test: state_dim=5 (dynamic model) output shape (N+1, 5)")
    print("=" * 60)

    horizon = 15
    pw = PathWindower(horizon=horizon, dt=0.05, state_dim=5)

    M = 50
    path = np.zeros((M, 3))
    path[:, 0] = np.linspace(0, 5, M)

    robot_state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    ref, _ = pw.extract_reference(path, robot_state)

    assert ref.shape == (horizon + 1, 5), f"Expected ({horizon+1}, 5), got {ref.shape}"
    # Extra dims (v, omega) should be zero-padded
    assert np.allclose(ref[:, 3], 0.0), "Dim 3 (v) should be zero-padded"
    assert np.allclose(ref[:, 4], 0.0), "Dim 4 (omega) should be zero-padded"
    print(f"  ref shape={ref.shape}, extra dims zero-padded")
    print("PASS")


def test_reset():
    print("\n" + "=" * 60)
    print("Test: reset() clears _closest_idx to 0")
    print("=" * 60)

    horizon = 10
    pw = PathWindower(horizon=horizon, dt=0.05, state_dim=3)

    M = 50
    path = np.zeros((M, 3))
    path[:, 0] = np.linspace(0, 5, M)

    # Move along path
    pw.extract_reference(path, np.array([3.0, 0.0, 0.0]))
    assert pw._closest_idx > 0, "Should have advanced"

    pw.reset()
    assert pw._closest_idx == 0, f"After reset, _closest_idx should be 0, got {pw._closest_idx}"
    print("PASS")


def test_circular_path():
    print("\n" + "=" * 60)
    print("Test: circular path smooth reference extraction")
    print("=" * 60)

    horizon = 20
    pw = PathWindower(horizon=horizon, dt=0.05, lookahead_distance=0.0, state_dim=3)

    # Full circle path
    M = 100
    t = np.linspace(0, 2 * np.pi, M, endpoint=False)
    radius = 2.0
    path = np.zeros((M, 3))
    path[:, 0] = radius * np.cos(t)
    path[:, 1] = radius * np.sin(t)
    path[:, 2] = t + np.pi / 2  # tangent direction

    # Robot at (2, 0), heading north
    robot_state = np.array([radius, 0.0, np.pi / 2])
    ref, closest_idx = pw.extract_reference(path, robot_state)

    assert ref.shape == (horizon + 1, 3), f"shape: {ref.shape}"
    # Reference should be near robot and smooth
    dist_to_robot = np.sqrt((ref[0, 0] - robot_state[0]) ** 2 +
                            (ref[0, 1] - robot_state[1]) ** 2)
    assert dist_to_robot < 1.0, f"First ref point too far from robot: {dist_to_robot:.3f}m"

    # Check smoothness: no large position jumps
    pos_diffs = np.sqrt(np.sum(np.diff(ref[:, :2], axis=0) ** 2, axis=1))
    max_pos_jump = np.max(pos_diffs)
    assert max_pos_jump < 1.0, f"Position jump too large: {max_pos_jump:.3f}m"
    print(f"  closest_idx={closest_idx}, dist_to_robot={dist_to_robot:.3f}m")
    print(f"  max position jump={max_pos_jump:.4f}m")
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PathWindower Unit Tests")
    print("=" * 60)

    tests = [
        test_basic_extraction,
        test_empty_path,
        test_single_point_path,
        test_closest_point_tracking,
        test_warm_start,
        test_angle_wrapping,
        test_end_of_path_padding,
        test_state_dim_5,
        test_reset,
        test_circular_path,
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
