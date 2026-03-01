"""
GoalChecker (goal_checker.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.ros2.nav2.goal_checker import GoalChecker


# ── Tests ──────────────────────────────────────────────────


def test_goal_reached():
    print("\n" + "=" * 60)
    print("Test: goal reached when within both tolerances")
    print("=" * 60)

    gc = GoalChecker(xy_tolerance=0.25, yaw_tolerance=0.25, stateful=False)

    current = np.array([1.0, 2.0, 0.5])
    goal = np.array([1.1, 2.1, 0.6])

    reached = gc.is_goal_reached(current, goal)

    xy_dist = gc.get_distance_to_goal(current, goal)
    yaw_diff = abs(current[2] - goal[2])
    assert reached is True, \
        f"Should be reached: xy_dist={xy_dist:.3f}, yaw_diff={yaw_diff:.3f}"
    print(f"  xy_dist={xy_dist:.3f} <= 0.25, yaw_diff={yaw_diff:.3f} <= 0.25")
    print("PASS")


def test_goal_not_reached_xy():
    print("\n" + "=" * 60)
    print("Test: goal not reached when xy distance too far")
    print("=" * 60)

    gc = GoalChecker(xy_tolerance=0.25, yaw_tolerance=0.25, stateful=False)

    current = np.array([0.0, 0.0, 0.0])
    goal = np.array([1.0, 0.0, 0.0])  # 1m away

    reached = gc.is_goal_reached(current, goal)
    xy_dist = gc.get_distance_to_goal(current, goal)

    assert reached is False, f"Should NOT be reached: xy_dist={xy_dist:.3f}"
    assert abs(xy_dist - 1.0) < 1e-6, f"Distance should be 1.0, got {xy_dist}"
    print(f"  xy_dist={xy_dist:.3f} > 0.25 -> not reached")
    print("PASS")


def test_goal_not_reached_yaw():
    print("\n" + "=" * 60)
    print("Test: goal not reached when yaw difference too large")
    print("=" * 60)

    gc = GoalChecker(xy_tolerance=0.25, yaw_tolerance=0.1, stateful=False)

    current = np.array([1.0, 2.0, 0.0])
    goal = np.array([1.0, 2.0, 0.5])  # same position, yaw diff=0.5

    reached = gc.is_goal_reached(current, goal)
    xy_dist = gc.get_distance_to_goal(current, goal)

    assert reached is False, \
        f"Should NOT be reached: yaw_diff=0.5 > tolerance=0.1"
    assert xy_dist < 0.01, f"XY should be close: {xy_dist}"
    print(f"  xy_dist={xy_dist:.3f} OK, yaw_diff=0.5 > 0.1 -> not reached")
    print("PASS")


def test_stateful():
    print("\n" + "=" * 60)
    print("Test: stateful mode stays reached even if robot moves away")
    print("=" * 60)

    gc = GoalChecker(xy_tolerance=0.25, yaw_tolerance=0.25, stateful=True)

    goal = np.array([1.0, 1.0, 0.0])

    # First: at goal
    at_goal = np.array([1.0, 1.0, 0.0])
    reached1 = gc.is_goal_reached(at_goal, goal)
    assert reached1 is True, "Should be reached at goal"

    # Second: move far away
    far_away = np.array([10.0, 10.0, 3.14])
    reached2 = gc.is_goal_reached(far_away, goal)
    assert reached2 is True, "Stateful: should STILL be reached after moving away"
    print("  At goal -> reached=True, far away -> still reached=True (stateful)")
    print("PASS")


def test_reset():
    print("\n" + "=" * 60)
    print("Test: reset() clears stateful flag")
    print("=" * 60)

    gc = GoalChecker(xy_tolerance=0.25, yaw_tolerance=0.25, stateful=True)

    goal = np.array([1.0, 1.0, 0.0])

    # Reach the goal
    gc.is_goal_reached(np.array([1.0, 1.0, 0.0]), goal)
    assert gc._goal_reached is True

    # Reset
    gc.reset()
    assert gc._goal_reached is False, "After reset, _goal_reached should be False"

    # Now far away should not be reached
    far_away = np.array([10.0, 10.0, 0.0])
    reached = gc.is_goal_reached(far_away, goal)
    assert reached is False, "After reset, far away should NOT be reached"
    print("  After reset: stateful flag cleared, far away -> not reached")
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GoalChecker Unit Tests")
    print("=" * 60)

    tests = [
        test_goal_reached,
        test_goal_not_reached_xy,
        test_goal_not_reached_yaw,
        test_stateful,
        test_reset,
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
