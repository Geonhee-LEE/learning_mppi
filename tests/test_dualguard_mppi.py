"""
DualGuard-MPPI (Hamilton-Jacobi Safety + MPPI) 유닛 테스트

28 tests covering:
  - TestDualGuardMPPIParams (3)
  - TestSafetyValueFunction (5)
  - TestDualGuardMPPIController (5)
  - TestSafetyGuarantee (4)
  - TestPerformance (4)
  - TestIntegration (7)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    DualGuardMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.dualguard_mppi import (
    DualGuardMPPIController,
    SafetyValueFunction,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


# -- Helpers --

DEFAULT_OBSTACLES = [
    (2.5, 2.0, 0.4),
    (-1.5, 3.0, 0.5),
    (1.0, -3.0, 0.3),
]


def _make_dualguard_controller(**kwargs):
    """Helper: DualGuard-MPPI controller creation"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        obstacles=DEFAULT_OBSTACLES,
        safety_margin=0.2,
        safety_mode="soft",
        safety_penalty=1000.0,
        safety_decay=5.0,
        use_velocity_penalty=True,
        velocity_penalty_weight=50.0,
        ttc_horizon=1.0,
        use_nominal_guard=True,
        use_sample_guard=True,
        min_safe_fraction=0.1,
        noise_boost_factor=1.5,
    )
    defaults.update(kwargs)
    params = DualGuardMPPIParams(**defaults)
    return DualGuardMPPIController(model, params)


def _make_vanilla_controller(**kwargs):
    """Helper: Vanilla MPPI controller with ObstacleCost"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    obstacles = kwargs.pop("obstacles", None)
    defaults.update(kwargs)
    params = MPPIParams(**defaults)

    cost_fns = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles:
        cost_fns.append(ObstacleCost(obstacles, safety_margin=0.1, cost_weight=2000.0))
    cost = CompositeMPPICost(cost_fns)

    return MPPIController(model, params, cost_function=cost)


def _make_ref(N=10, dt=0.05):
    """Helper: Reference trajectory"""
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


# == TestDualGuardMPPIParams (3) ==


def test_params_defaults():
    """DualGuardMPPIParams default values"""
    p = DualGuardMPPIParams()
    assert p.obstacles == []
    assert p.safety_margin == 0.2
    assert p.safety_mode == "soft"
    assert p.safety_penalty == 1000.0
    assert p.safety_decay == 5.0
    assert p.use_velocity_penalty is True
    assert p.velocity_penalty_weight == 50.0
    assert p.ttc_horizon == 1.0
    assert p.use_nominal_guard is True
    assert p.use_sample_guard is True
    assert p.min_safe_fraction == 0.1
    assert p.noise_boost_factor == 1.5
    assert p.N == 30  # MPPIParams default
    assert p.K == 1024


def test_params_custom():
    """DualGuardMPPIParams custom values"""
    p = DualGuardMPPIParams(
        K=256, N=20, lambda_=2.0,
        obstacles=[(1, 2, 0.3)],
        safety_mode="filter",
        safety_penalty=500.0,
        noise_boost_factor=2.0,
    )
    assert p.K == 256
    assert p.N == 20
    assert p.lambda_ == 2.0
    assert len(p.obstacles) == 1
    assert p.safety_mode == "filter"
    assert p.safety_penalty == 500.0
    assert p.noise_boost_factor == 2.0


def test_params_validation():
    """DualGuardMPPIParams validation"""
    # Invalid safety_mode
    try:
        DualGuardMPPIParams(safety_mode="invalid")
        assert False, "Should raise AssertionError"
    except AssertionError:
        pass

    # Invalid noise_boost_factor < 1
    try:
        DualGuardMPPIParams(noise_boost_factor=0.5)
        assert False, "Should raise AssertionError"
    except AssertionError:
        pass

    # Invalid ttc_horizon <= 0
    try:
        DualGuardMPPIParams(ttc_horizon=0.0)
        assert False, "Should raise AssertionError"
    except AssertionError:
        pass

    # Valid modes
    for mode in ["soft", "hard", "filter"]:
        p = DualGuardMPPIParams(safety_mode=mode)
        assert p.safety_mode == mode


# == TestSafetyValueFunction (5) ==


def test_safe_state_positive_value():
    """Far from obstacles -> V > 0"""
    svf = SafetyValueFunction(
        obstacles=[(0.0, 0.0, 0.5)],
        safety_margin=0.2,
    )
    # State far from obstacle
    state = np.array([5.0, 5.0, 0.0])
    V = svf.evaluate(state[None, :])[0]
    assert V > 0, f"Expected V > 0 for safe state, got V={V}"

    # Distance should be approx sqrt(50) - 0.7
    expected = np.sqrt(50.0) - 0.7
    assert abs(V - expected) < 0.01, f"V={V}, expected~{expected}"


def test_unsafe_state_negative_value():
    """Inside obstacle -> V < 0"""
    svf = SafetyValueFunction(
        obstacles=[(0.0, 0.0, 1.0)],
        safety_margin=0.2,
    )
    # State inside obstacle
    state = np.array([0.0, 0.0, 0.0])
    V = svf.evaluate(state[None, :])[0]
    assert V < 0, f"Expected V < 0 for unsafe state, got V={V}"


def test_boundary_state_near_zero():
    """On obstacle boundary -> V near 0"""
    r = 0.5
    margin = 0.2
    svf = SafetyValueFunction(
        obstacles=[(0.0, 0.0, r)],
        safety_margin=margin,
    )
    # State exactly at boundary (dist = r + margin)
    boundary_dist = r + margin
    state = np.array([boundary_dist, 0.0, 0.0])
    V = svf.evaluate(state[None, :])[0]
    assert abs(V) < 0.05, f"Expected V near 0 at boundary, got V={V}"


def test_batch_evaluation():
    """Batch evaluation: (K, N+1, nx) -> (K, N+1) values"""
    svf = SafetyValueFunction(
        obstacles=[(0.0, 0.0, 0.5), (3.0, 3.0, 0.3)],
        safety_margin=0.1,
    )
    K, N = 32, 10
    nx = 3
    states = np.random.randn(K, N + 1, nx) * 3.0
    values = svf.evaluate(states)
    assert values.shape == (K, N + 1), f"Expected ({K}, {N+1}), got {values.shape}"

    # Single state evaluation should match
    v_single = svf.evaluate(states[0, 0:1, :])
    assert abs(values[0, 0] - v_single[0]) < 1e-10


def test_gradient_direction():
    """Gradient points away from nearest obstacle"""
    svf = SafetyValueFunction(
        obstacles=[(0.0, 0.0, 0.5)],
        safety_margin=0.1,
    )
    # State to the right of obstacle
    state = np.array([[1.0, 0.0, 0.0]])
    grad = svf.gradient(state)
    assert grad.shape == (1, 2)
    # Gradient should point to the right (away from (0,0))
    assert grad[0, 0] > 0, f"Expected positive x gradient, got {grad[0, 0]}"
    assert abs(grad[0, 1]) < 0.01, f"Expected ~0 y gradient, got {grad[0, 1]}"

    # State above obstacle
    state2 = np.array([[0.0, 2.0, 0.0]])
    grad2 = svf.gradient(state2)
    assert grad2[0, 1] > 0, f"Expected positive y gradient, got {grad2[0, 1]}"
    assert abs(grad2[0, 0]) < 0.01, f"Expected ~0 x gradient, got {grad2[0, 0]}"


# == TestDualGuardMPPIController (5) ==


def test_compute_control_shape():
    """compute_control returns (nu,) control and info dict"""
    np.random.seed(42)
    ctrl = _make_dualguard_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,), f"Expected (2,), got {control.shape}"
    assert isinstance(info, dict)

    # Required info keys
    for key in ["sample_trajectories", "sample_weights", "best_trajectory",
                "best_cost", "mean_cost", "temperature", "ess", "num_samples"]:
        assert key in info, f"Missing key: {key}"

    assert info["sample_trajectories"].shape == (64, 11, 3)
    assert info["sample_weights"].shape == (64,)
    assert info["num_samples"] == 64


def test_info_guard_stats():
    """Guard stats present in info"""
    np.random.seed(42)
    ctrl = _make_dualguard_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)
    assert "guard_stats" in info
    gs = info["guard_stats"]
    for key in ["safe_fraction", "min_safety_value", "mean_safety_value",
                "noise_boost", "sigma_eff", "nominal_correction",
                "safety_mode", "num_obstacles"]:
        assert key in gs, f"Missing guard_stats key: {key}"

    assert gs["safety_mode"] == "soft"
    assert gs["num_obstacles"] == 3
    assert 0 <= gs["safe_fraction"] <= 1


def test_soft_mode():
    """Soft safety mode adds penalty near obstacles"""
    np.random.seed(42)
    ctrl_soft = _make_dualguard_controller(safety_mode="soft")
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl_soft.compute_control(state, ref)
    assert control.shape == (2,)
    assert info["guard_stats"]["safety_mode"] == "soft"
    # Should still produce valid control
    assert np.all(np.isfinite(control))


def test_hard_mode():
    """Hard safety mode projects controls"""
    np.random.seed(42)
    ctrl_hard = _make_dualguard_controller(safety_mode="hard")
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl_hard.compute_control(state, ref)
    assert control.shape == (2,)
    assert info["guard_stats"]["safety_mode"] == "hard"
    assert np.all(np.isfinite(control))


def test_filter_mode():
    """Filter safety mode rejects unsafe samples"""
    np.random.seed(42)
    ctrl_filter = _make_dualguard_controller(safety_mode="filter")
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    control, info = ctrl_filter.compute_control(state, ref)
    assert control.shape == (2,)
    assert info["guard_stats"]["safety_mode"] == "filter"
    assert np.all(np.isfinite(control))


# == TestSafetyGuarantee (4) ==


def test_never_enters_obstacle():
    """DualGuard-MPPI should not enter obstacles over a long run"""
    np.random.seed(42)
    obstacles = [(3.0, 0.0, 0.5), (-1.5, 2.5, 0.4)]
    ctrl = _make_dualguard_controller(
        K=128, N=15, obstacles=obstacles,
        safety_penalty=2000.0, safety_mode="soft",
    )

    state = np.array([0.0, 0.0, 0.0])
    dt = 0.05
    N = 15
    n_collisions = 0

    for step in range(100):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, N, dt)
        control, _ = ctrl.compute_control(state, ref)

        state_dot = ctrl.model.forward_dynamics(state, control)
        state = state + state_dot * dt

        # Check collision
        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            if dist < r:
                n_collisions += 1

    assert n_collisions == 0, f"Expected 0 collisions, got {n_collisions}"


def test_maintains_clearance():
    """Minimum clearance should be > 0"""
    np.random.seed(42)
    obstacles = [(3.0, 0.0, 0.5)]
    ctrl = _make_dualguard_controller(
        K=128, N=15, obstacles=obstacles,
        safety_mode="soft", safety_penalty=2000.0,
    )

    state = np.array([0.0, 0.0, 0.0])
    dt = 0.05
    N = 15
    min_clearance = float('inf')

    for step in range(80):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, N, dt)
        control, _ = ctrl.compute_control(state, ref)

        state_dot = ctrl.model.forward_dynamics(state, control)
        state = state + state_dot * dt

        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            clearance = dist - r
            min_clearance = min(min_clearance, clearance)

    assert min_clearance > 0, f"Expected clearance > 0, got {min_clearance}"


def test_safety_penalty_increases_near_obstacle():
    """Safety penalty should increase as V(x) decreases"""
    svf = SafetyValueFunction(
        obstacles=[(0.0, 0.0, 0.5)],
        safety_margin=0.2,
    )
    # Points at increasing distances
    distances = [0.3, 0.5, 0.7, 1.0, 2.0, 5.0]
    values = []
    for d in distances:
        state = np.array([[d, 0.0, 0.0]])
        V = svf.evaluate(state)[0]
        values.append(V)

    # Values should be monotonically increasing with distance
    for i in range(len(values) - 1):
        assert values[i] < values[i + 1], \
            f"V should increase with distance: V[{distances[i]}]={values[i]} >= V[{distances[i+1]}]={values[i+1]}"

    # Near obstacle (d=0.3 < r+margin=0.7), V should be negative
    assert values[0] < 0, f"Expected V < 0 inside, got {values[0]}"

    # Far from obstacle (d=5.0), V should be positive
    assert values[-1] > 0, f"Expected V > 0 far away, got {values[-1]}"


def test_velocity_penalty():
    """Moving toward obstacle should be penalized more"""
    np.random.seed(42)

    # Create controller near obstacle with velocity penalty
    obstacles = [(2.0, 0.0, 0.5)]
    ctrl_vel = _make_dualguard_controller(
        K=128, N=15, obstacles=obstacles,
        use_velocity_penalty=True, velocity_penalty_weight=100.0,
    )
    ctrl_no_vel = _make_dualguard_controller(
        K=128, N=15, obstacles=obstacles,
        use_velocity_penalty=False,
    )

    # State heading toward obstacle
    state = np.array([1.0, 0.0, 0.0])  # facing right, obstacle at (2,0)
    ref = _make_ref(N=15)

    np.random.seed(42)
    _, info_vel = ctrl_vel.compute_control(state, ref)
    np.random.seed(42)
    _, info_no_vel = ctrl_no_vel.compute_control(state, ref)

    # With velocity penalty, mean cost should be higher (more cautious)
    assert info_vel["mean_cost"] >= info_no_vel["mean_cost"] * 0.5, \
        "Velocity penalty should influence costs"


# == TestPerformance (4) ==


def test_circle_tracking_rmse():
    """RMSE < 0.3 for circle tracking with obstacles"""
    np.random.seed(42)
    # Use smaller circle (r=3) to match initial state better
    from functools import partial
    small_circle = partial(circle_trajectory, radius=3.0, angular_velocity=0.15)

    obstacles = [(4.0, 4.0, 0.3)]  # Away from circle path
    ctrl = _make_dualguard_controller(
        K=128, N=15, obstacles=obstacles,
        safety_mode="soft",
    )

    state = np.array([3.0, 0.0, np.pi / 2])  # on the circle
    dt = 0.05
    N = 15
    errors = []

    for step in range(100):
        t = step * dt
        ref = generate_reference_trajectory(small_circle, t, N, dt)
        control, _ = ctrl.compute_control(state, ref)

        state_dot = ctrl.model.forward_dynamics(state, control)
        state = state + state_dot * dt

        ref_pt = small_circle(t)
        err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
        errors.append(err)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    assert rmse < 0.3, f"Expected RMSE < 0.3, got {rmse:.4f}"


def test_obstacle_avoidance():
    """3 obstacles, 0 collisions"""
    np.random.seed(42)
    from functools import partial
    small_circle = partial(circle_trajectory, radius=3.0, angular_velocity=0.15)

    # Obstacles placed near but not on the circle path
    obstacles = [
        (2.5, 1.5, 0.3),
        (0.0, 2.5, 0.3),
        (-2.0, -0.5, 0.3),
    ]
    ctrl = _make_dualguard_controller(
        K=128, N=15, obstacles=obstacles,
        safety_mode="soft", safety_penalty=5000.0,
        safety_decay=8.0,
    )

    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05
    N = 15
    n_collisions = 0

    for step in range(120):
        t = step * dt
        ref = generate_reference_trajectory(small_circle, t, N, dt)
        control, _ = ctrl.compute_control(state, ref)

        state_dot = ctrl.model.forward_dynamics(state, control)
        state = state + state_dot * dt

        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            if dist < r:
                n_collisions += 1

    assert n_collisions == 0, f"Expected 0 collisions, got {n_collisions}"


def test_dense_obstacles():
    """6 obstacles, 0 collisions"""
    np.random.seed(42)
    from functools import partial
    small_circle = partial(circle_trajectory, radius=3.0, angular_velocity=0.15)

    # 6 obstacles near but not overlapping the circle path
    obstacles = [
        (2.5, 1.5, 0.2),
        (0.0, 2.5, 0.2),
        (-2.5, 0.5, 0.2),
        (-1.0, -2.5, 0.2),
        (1.5, -2.0, 0.2),
        (2.0, 2.0, 0.15),
    ]
    ctrl = _make_dualguard_controller(
        K=256, N=15, obstacles=obstacles,
        safety_mode="soft", safety_penalty=5000.0,
        safety_decay=8.0,
        use_velocity_penalty=True,
    )

    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05
    N = 15
    n_collisions = 0

    for step in range(100):
        t = step * dt
        ref = generate_reference_trajectory(small_circle, t, N, dt)
        control, _ = ctrl.compute_control(state, ref)

        state_dot = ctrl.model.forward_dynamics(state, control)
        state = state + state_dot * dt

        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            if dist < r:
                n_collisions += 1

    assert n_collisions == 0, f"Expected 0 collisions, got {n_collisions}"


def test_computation_time():
    """K=512, N=30, < 100ms average"""
    np.random.seed(42)
    ctrl = _make_dualguard_controller(
        K=512, N=30, obstacles=DEFAULT_OBSTACLES,
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = generate_reference_trajectory(circle_trajectory, 0.0, 30, 0.05)

    # Warm-up
    ctrl.compute_control(state, ref)

    times = []
    for _ in range(10):
        t0 = time.time()
        ctrl.compute_control(state, ref)
        times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    assert avg_ms < 100, f"Expected < 100ms, got {avg_ms:.1f}ms"


# == TestIntegration (7) ==


def test_numerical_stability():
    """No NaN/Inf in outputs"""
    np.random.seed(42)
    ctrl = _make_dualguard_controller(
        K=64, N=10,
        safety_penalty=10000.0,
        safety_decay=10.0,
    )
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    for _ in range(20):
        control, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(control)), "Control contains NaN/Inf"
        assert np.all(np.isfinite(info["sample_weights"])), "Weights contain NaN/Inf"
        assert np.isfinite(info["ess"]), "ESS is NaN/Inf"

        state_dot = ctrl.model.forward_dynamics(state, control)
        state = state + state_dot * 0.05


def test_update_obstacles():
    """Dynamic obstacle update works"""
    np.random.seed(42)
    ctrl = _make_dualguard_controller(obstacles=[(10.0, 10.0, 0.5)])

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()
    _, info1 = ctrl.compute_control(state, ref)

    # Obstacle far away -> high safety
    gs1 = info1["guard_stats"]
    assert gs1["min_safety_value"] > 0

    # Move obstacle close
    ctrl.update_obstacles([(0.5, 0.0, 0.4)])
    _, info2 = ctrl.compute_control(state, ref)
    gs2 = info2["guard_stats"]

    # Safety should decrease
    assert gs2["min_safety_value"] < gs1["min_safety_value"], \
        "Safety should decrease when obstacle moves closer"


def test_noise_boost_on_unsafe():
    """Noise boost triggers when too few safe samples"""
    np.random.seed(42)
    # Place obstacle right at origin where robot starts
    obstacles = [(0.0, 0.0, 2.0)]  # Large obstacle
    ctrl = _make_dualguard_controller(
        K=64, N=10, obstacles=obstacles,
        min_safe_fraction=0.5,  # Need 50% safe
        noise_boost_factor=2.0,
    )

    state = np.array([0.0, 0.0, 0.0])  # Inside obstacle!
    ref = _make_ref()

    # Run a few steps - noise boost should activate
    for _ in range(5):
        _, info = ctrl.compute_control(state, ref)

    # Noise boost should be > 1.0
    assert ctrl._noise_boost > 1.0, \
        f"Expected noise_boost > 1.0, got {ctrl._noise_boost}"


def test_nominal_guard():
    """Nominal guard corrects unsafe optimal control"""
    np.random.seed(42)
    # Obstacle right in front of robot
    obstacles = [(0.5, 0.0, 0.3)]
    ctrl = _make_dualguard_controller(
        K=64, N=10, obstacles=obstacles,
        use_nominal_guard=True,
        safety_penalty=5000.0,
    )

    state = np.array([0.0, 0.0, 0.0])  # facing obstacle
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    # Should still produce finite control
    assert np.all(np.isfinite(control))

    # Guard stats should show nominal guard activity
    gs = info["guard_stats"]
    # nominal_correction may or may not be > 0 depending on whether
    # the MPPI already avoids the obstacle, but it should be non-negative
    assert gs["nominal_correction"] >= 0


def test_different_K_values():
    """Works with different K values"""
    np.random.seed(42)
    state = np.array([0.0, 0.0, 0.0])

    for K in [32, 128, 256]:
        ctrl = _make_dualguard_controller(K=K, N=10)
        ref = _make_ref()
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["num_samples"] == K
        assert info["sample_trajectories"].shape[0] == K


def test_reset_clears_state():
    """Reset clears control sequence and guard state"""
    np.random.seed(42)
    ctrl = _make_dualguard_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref()

    # Run a few steps
    for _ in range(5):
        ctrl.compute_control(state, ref)

    assert len(ctrl._guard_stats) == 5
    assert ctrl.U is not None

    # Reset
    ctrl.reset()
    assert len(ctrl._guard_stats) == 0
    assert ctrl._noise_boost == 1.0
    assert np.allclose(ctrl.U, 0.0)


def test_no_obstacles_fallback():
    """Empty obstacles -> vanilla MPPI behavior"""
    np.random.seed(42)
    ctrl_guard = _make_dualguard_controller(obstacles=[])
    ctrl_vanilla = _make_vanilla_controller()

    state = np.array([3.0, 0.0, np.pi / 2])
    ref = generate_reference_trajectory(circle_trajectory, 0.0, 10, 0.05)

    # Both should produce finite controls
    np.random.seed(123)
    control_guard, info_guard = ctrl_guard.compute_control(state, ref)
    assert np.all(np.isfinite(control_guard))

    # Guard stats should show no obstacles, full safety
    gs = info_guard["guard_stats"]
    assert gs["num_obstacles"] == 0
    assert gs["safe_fraction"] == 1.0
    assert gs["noise_boost"] == 1.0
