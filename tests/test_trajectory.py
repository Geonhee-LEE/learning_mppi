"""
궤적 생성 유틸리티 (trajectory.py) 유닛 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.utils.trajectory import (
    circle_trajectory,
    figure_eight_trajectory,
    sine_wave_trajectory,
    slalom_trajectory,
    straight_line_trajectory,
    generate_reference_trajectory,
    create_trajectory_function,
)


# ── Tests ──────────────────────────────────────────────────


def test_circle_shape():
    print("\n" + "=" * 60)
    print("Test: circle_trajectory returns (3,) = [x, y, theta]")
    print("=" * 60)

    state = circle_trajectory(0.0)
    assert state.shape == (3,), f"shape: {state.shape}"
    state2 = circle_trajectory(5.0)
    assert state2.shape == (3,), f"shape at t=5: {state2.shape}"
    print("PASS")


def test_circle_radius():
    print("\n" + "=" * 60)
    print("Test: circle sqrt(x^2 + y^2) ~ radius")
    print("=" * 60)

    radius = 3.0
    for t in [0, 1, 2, 5, 10]:
        state = circle_trajectory(float(t), radius=radius)
        r = np.sqrt(state[0] ** 2 + state[1] ** 2)
        assert abs(r - radius) < 1e-10, f"t={t}: r={r}, expected={radius}"
    print("PASS")


def test_figure8_shape():
    print("\n" + "=" * 60)
    print("Test: figure_eight_trajectory returns (3,)")
    print("=" * 60)

    state = figure_eight_trajectory(0.0)
    assert state.shape == (3,), f"shape: {state.shape}"
    state2 = figure_eight_trajectory(5.0)
    assert state2.shape == (3,), f"shape at t=5: {state2.shape}"
    # Check no NaN
    assert not np.any(np.isnan(state)), f"NaN at t=0: {state}"
    assert not np.any(np.isnan(state2)), f"NaN at t=5: {state2}"
    print("PASS")


def test_sine_shape():
    print("\n" + "=" * 60)
    print("Test: sine_wave_trajectory returns (3,)")
    print("=" * 60)

    state = sine_wave_trajectory(0.0)
    assert state.shape == (3,), f"shape: {state.shape}"
    state2 = sine_wave_trajectory(3.0)
    assert state2.shape == (3,), f"shape at t=3: {state2.shape}"
    print("PASS")


def test_straight_line():
    print("\n" + "=" * 60)
    print("Test: straight_line x = v*t, y = 0 (heading=0)")
    print("=" * 60)

    v = 2.0
    for t in [0, 1, 3, 5]:
        state = straight_line_trajectory(float(t), velocity=v, heading=0.0)
        assert abs(state[0] - v * t) < 1e-10, f"x={state[0]}, expected={v*t}"
        assert abs(state[1]) < 1e-10, f"y={state[1]}, expected=0"
        assert abs(state[2]) < 1e-10, f"theta={state[2]}, expected=0"
    print("PASS")


def test_generate_reference_shape():
    print("\n" + "=" * 60)
    print("Test: generate_reference_trajectory returns (N+1, 3)")
    print("=" * 60)

    N = 20
    dt = 0.05
    ref = generate_reference_trajectory(circle_trajectory, 0.0, N, dt)

    assert ref.shape == (N + 1, 3), f"shape: {ref.shape}"
    # First point should match circle_trajectory(0.0)
    expected_first = circle_trajectory(0.0)
    assert np.allclose(ref[0], expected_first), \
        f"first: {ref[0]} != {expected_first}"
    print("PASS")


def test_create_trajectory_function():
    print("\n" + "=" * 60)
    print("Test: create_trajectory_function for all types")
    print("=" * 60)

    for ttype in ["circle", "figure8", "sine", "slalom", "straight"]:
        fn = create_trajectory_function(ttype)
        state = fn(1.0)
        assert state.shape == (3,), f"{ttype}: shape={state.shape}"
        assert not np.any(np.isnan(state)), f"{ttype}: NaN"
        print(f"  {ttype}: {state}")
    print("PASS")


def test_unknown_trajectory_type():
    print("\n" + "=" * 60)
    print("Test: unknown type -> ValueError")
    print("=" * 60)

    try:
        create_trajectory_function("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  caught: {e}")
    print("PASS")


def test_slalom_shape():
    print("\n" + "=" * 60)
    print("Test: slalom_trajectory returns (3,) with valid heading")
    print("=" * 60)

    for t in [0.0, 5.0, 10.0, 20.0]:
        state = slalom_trajectory(t)
        assert state.shape == (3,), f"shape at t={t}: {state.shape}"
        assert not np.any(np.isnan(state)), f"NaN at t={t}: {state}"
        # heading should be within [-pi, pi]
        assert -np.pi <= state[2] <= np.pi, f"heading out of range: {state[2]}"

    # x should advance with velocity
    s0 = slalom_trajectory(0.0)
    s10 = slalom_trajectory(10.0)
    assert s10[0] > s0[0], "x should increase with time"
    # y should oscillate (not always 0)
    y_vals = [slalom_trajectory(t)[1] for t in np.linspace(0, 20, 100)]
    assert max(y_vals) > 0.1, f"y should have positive excursion: max={max(y_vals)}"
    assert min(y_vals) < -0.1, f"y should have negative excursion: min={min(y_vals)}"
    print("PASS")


def test_slalom_chirp():
    print("\n" + "=" * 60)
    print("Test: slalom chirp — frequency increases over time")
    print("=" * 60)

    # Measure zero-crossing intervals: later intervals should be shorter
    y_vals = []
    times = np.linspace(0, 25, 5000)
    for t in times:
        y_vals.append(slalom_trajectory(t)[1])
    y_vals = np.array(y_vals)

    # Find zero crossings
    crossings = []
    for i in range(len(y_vals) - 1):
        if y_vals[i] * y_vals[i + 1] < 0:
            crossings.append(times[i])

    assert len(crossings) >= 4, f"Need at least 4 zero crossings, got {len(crossings)}"

    # Compare early vs late half-periods
    early_interval = crossings[1] - crossings[0]
    late_interval = crossings[-1] - crossings[-2]
    assert late_interval < early_interval, \
        f"Late interval ({late_interval:.3f}) should be shorter than early ({early_interval:.3f})"
    print(f"  Early half-period: {early_interval:.3f}s, Late: {late_interval:.3f}s")
    print("PASS")


def test_slalom_kinematic_feasibility():
    print("\n" + "=" * 60)
    print("Test: slalom trajectory respects v_max/omega_max constraints")
    print("=" * 60)

    v_max = 1.0
    omega_max = 1.0
    dt_check = 0.01
    times = np.arange(0, 25, dt_check)
    violations_v = 0
    violations_omega = 0

    for i in range(len(times) - 1):
        s0 = slalom_trajectory(times[i])
        s1 = slalom_trajectory(times[i + 1])
        dx = s1[0] - s0[0]
        dy = s1[1] - s0[1]
        v = np.sqrt(dx ** 2 + dy ** 2) / dt_check
        dtheta = np.arctan2(np.sin(s1[2] - s0[2]), np.cos(s1[2] - s0[2]))
        omega = abs(dtheta) / dt_check

        if v > v_max * 1.05:  # 5% tolerance
            violations_v += 1
        if omega > omega_max * 1.5:  # 50% tolerance for omega
            violations_omega += 1

    violation_rate_v = violations_v / len(times)
    violation_rate_omega = violations_omega / len(times)
    print(f"  v violations: {violations_v}/{len(times)} ({violation_rate_v:.1%})")
    print(f"  omega violations: {violations_omega}/{len(times)} ({violation_rate_omega:.1%})")

    assert violation_rate_v < 0.05, \
        f"Too many v_max violations: {violation_rate_v:.1%}"
    assert violation_rate_omega < 0.10, \
        f"Too many omega_max violations: {violation_rate_omega:.1%}"
    print("PASS")


def test_heading_consistency():
    print("\n" + "=" * 60)
    print("Test: heading ~ atan2(dy, dx) for circle trajectory")
    print("=" * 60)

    dt_check = 0.001
    for t in [0.0, 1.0, 3.0, 5.0]:
        s0 = circle_trajectory(t)
        s1 = circle_trajectory(t + dt_check)
        dx = s1[0] - s0[0]
        dy = s1[1] - s0[1]
        numerical_heading = np.arctan2(dy, dx)

        # Compare with reported heading
        heading = s0[2]
        diff = np.arctan2(np.sin(heading - numerical_heading),
                          np.cos(heading - numerical_heading))
        assert abs(diff) < 0.05, \
            f"t={t}: heading={heading:.4f}, numerical={numerical_heading:.4f}, diff={diff:.4f}"

    print("PASS")


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Trajectory Utils Unit Tests")
    print("=" * 60)

    tests = [
        test_circle_shape,
        test_circle_radius,
        test_figure8_shape,
        test_sine_shape,
        test_straight_line,
        test_slalom_shape,
        test_slalom_chirp,
        test_slalom_kinematic_feasibility,
        test_generate_reference_shape,
        test_create_trajectory_function,
        test_unknown_trajectory_type,
        test_heading_consistency,
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
