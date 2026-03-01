"""
ProgressChecker (progress_checker.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.ros2.nav2.progress_checker import ProgressChecker


# ── Tests ──────────────────────────────────────────────────


def test_initial_progress():
    print("\n" + "=" * 60)
    print("Test: first call always returns True")
    print("=" * 60)

    pc = ProgressChecker(required_movement=0.5, time_allowance=10.0)

    state = np.array([0.0, 0.0, 0.0])
    result = pc.check_progress(state, current_time=0.0)

    assert result is True, "First call should always return True"
    print("  first call -> True (reference initialized)")
    print("PASS")


def test_making_progress():
    print("\n" + "=" * 60)
    print("Test: moving enough distance returns True")
    print("=" * 60)

    required = 0.5
    pc = ProgressChecker(required_movement=required, time_allowance=10.0)

    # Initialize
    pc.check_progress(np.array([0.0, 0.0, 0.0]), current_time=0.0)

    # Move more than required_movement
    result = pc.check_progress(np.array([0.6, 0.0, 0.0]), current_time=5.0)

    assert result is True, \
        f"Moved 0.6m > {required}m required, should be making progress"
    print(f"  moved 0.6m > {required}m -> True (making progress)")
    print("PASS")


def test_stuck_detection():
    print("\n" + "=" * 60)
    print("Test: staying still past time_allowance returns False")
    print("=" * 60)

    required = 0.5
    allowance = 5.0
    pc = ProgressChecker(required_movement=required, time_allowance=allowance)

    # Initialize at t=0
    pc.check_progress(np.array([0.0, 0.0, 0.0]), current_time=0.0)

    # Stay still but within time
    result1 = pc.check_progress(np.array([0.0, 0.0, 0.0]), current_time=3.0)
    assert result1 is True, "Within time allowance, should still be True"

    # Stay still past time allowance
    result2 = pc.check_progress(np.array([0.0, 0.0, 0.0]), current_time=6.0)
    assert result2 is False, \
        f"Still at origin after {6.0}s > {allowance}s allowance -> stuck"
    print(f"  at t=3.0s -> True (within {allowance}s)")
    print(f"  at t=6.0s -> False (past {allowance}s, stuck!)")
    print("PASS")


def test_within_allowance():
    print("\n" + "=" * 60)
    print("Test: staying still but within time_allowance returns True")
    print("=" * 60)

    allowance = 10.0
    pc = ProgressChecker(required_movement=0.5, time_allowance=allowance)

    # Initialize at t=0
    pc.check_progress(np.array([1.0, 1.0, 0.0]), current_time=0.0)

    # Barely move, within time
    result = pc.check_progress(np.array([1.01, 1.01, 0.0]), current_time=5.0)
    assert result is True, \
        f"Within time allowance ({5.0}s < {allowance}s), should be True"
    print(f"  barely moved at t=5.0s < {allowance}s -> True")
    print("PASS")


def test_reset():
    print("\n" + "=" * 60)
    print("Test: reset() clears state, starts fresh")
    print("=" * 60)

    pc = ProgressChecker(required_movement=0.5, time_allowance=5.0)

    # Initialize and get stuck
    pc.check_progress(np.array([0.0, 0.0, 0.0]), current_time=0.0)
    stuck = pc.check_progress(np.array([0.0, 0.0, 0.0]), current_time=10.0)
    assert stuck is False, "Should be stuck"

    # Reset
    pc.reset()
    assert pc._reference_position is None, "After reset, _reference_position should be None"
    assert pc._reference_time is None, "After reset, _reference_time should be None"

    # First call after reset should return True
    result = pc.check_progress(np.array([0.0, 0.0, 0.0]), current_time=20.0)
    assert result is True, "After reset, first call should return True"
    print("  stuck -> reset -> first call True (fresh start)")
    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ProgressChecker Unit Tests")
    print("=" * 60)

    tests = [
        test_initial_progress,
        test_making_progress,
        test_stuck_detection,
        test_within_allowance,
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
