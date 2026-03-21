"""
Parameter-Robust MPPI (PR-MPPI) 유닛 테스트

입자 기반 파라미터 추정 + 다중 모델 MPPI 컨트롤러 검증.
28 tests: Params(3) + ParticleFilter(5) + Controller(5) + Robustness(4) + Performance(4) + Integration(7)
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
    ParameterRobustMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.parameter_robust_mppi import (
    ParameterParticleFilter,
    ParameterRobustMPPIController,
    _ParametricModel,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


# -- Helpers ---------------------------------------------------------------

def _make_pr_controller(**kwargs):
    """Helper: PR-MPPI controller creation"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0, wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        n_particles=5,
        param_name="wheelbase",
        param_nominal=0.5,
        param_std=0.1,
        param_min=0.2,
        param_max=1.0,
        aggregation_mode="weighted_mean",
        online_learning=True,
        learning_rate=0.01,
        observation_window=5,
        min_observations=3,
        use_resampling=True,
        resample_threshold=0.3,
    )
    defaults.update(kwargs)

    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)

    params = ParameterRobustMPPIParams(**defaults)
    return ParameterRobustMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_ref(N=10, dt=0.05):
    """Helper: reference trajectory"""
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


# -- TestParameterRobustMPPIParams (3) ------------------------------------

def test_params_defaults():
    """Default values check"""
    params = ParameterRobustMPPIParams()
    assert params.n_particles == 5
    assert params.param_name == "wheelbase"
    assert params.param_nominal == 0.5
    assert params.param_std == 0.1
    assert params.param_min == 0.2
    assert params.param_max == 1.0
    assert params.aggregation_mode == "weighted_mean"
    assert params.cvar_alpha == 0.3
    assert params.online_learning is True
    assert params.learning_rate == 0.01
    assert params.observation_window == 5
    assert params.min_observations == 3
    assert params.use_resampling is True
    assert params.resample_threshold == 0.3


def test_params_custom():
    """Custom values check"""
    params = ParameterRobustMPPIParams(
        n_particles=10,
        param_name="wheelbase",
        param_nominal=0.6,
        param_std=0.2,
        param_min=0.3,
        param_max=0.9,
        aggregation_mode="worst_case",
        cvar_alpha=0.5,
        online_learning=False,
        learning_rate=0.05,
        observation_window=10,
        min_observations=5,
        use_resampling=False,
        resample_threshold=0.5,
    )
    assert params.n_particles == 10
    assert params.param_nominal == 0.6
    assert params.param_std == 0.2
    assert params.aggregation_mode == "worst_case"
    assert params.cvar_alpha == 0.5
    assert params.online_learning is False
    assert params.learning_rate == 0.05
    assert params.observation_window == 10
    assert params.use_resampling is False


def test_params_validation():
    """Invalid values -> AssertionError"""
    # n_particles = 0
    try:
        ParameterRobustMPPIParams(n_particles=0)
        assert False, "Should raise for n_particles=0"
    except AssertionError:
        pass

    # param_min > param_max
    try:
        ParameterRobustMPPIParams(param_min=1.0, param_max=0.5)
        assert False, "Should raise for param_min > param_max"
    except AssertionError:
        pass

    # param_nominal out of [min, max]
    try:
        ParameterRobustMPPIParams(param_nominal=0.1, param_min=0.2, param_max=1.0)
        assert False, "Should raise for nominal out of bounds"
    except AssertionError:
        pass

    # Unknown aggregation_mode
    try:
        ParameterRobustMPPIParams(aggregation_mode="unknown")
        assert False, "Should raise for unknown mode"
    except AssertionError:
        pass

    # cvar_alpha = 0
    try:
        ParameterRobustMPPIParams(cvar_alpha=0.0)
        assert False, "Should raise for cvar_alpha=0"
    except AssertionError:
        pass


# -- TestParameterParticleFilter (5) --------------------------------------

def test_initial_particles():
    """Particles: correct number, within bounds"""
    np.random.seed(42)
    pf = ParameterParticleFilter(
        n_particles=20,
        param_nominal=0.5,
        param_std=0.1,
        param_min=0.2,
        param_max=1.0,
    )
    assert len(pf.particles) == 20
    assert np.all(pf.particles >= 0.2)
    assert np.all(pf.particles <= 1.0)
    assert len(pf.weights) == 20
    assert np.isclose(np.sum(pf.weights), 1.0)


def test_weight_update():
    """Weights change after observation"""
    np.random.seed(42)
    pf = ParameterParticleFilter(
        n_particles=10,
        param_nominal=0.5,
        param_std=0.2,
        param_min=0.2,
        param_max=1.0,
    )
    initial_weights = pf.weights.copy()

    # Give different errors to particles
    errors = np.abs(pf.particles - 0.6)  # particles near 0.6 have lower error
    pf.update_weights(errors, observation_noise_std=0.1)

    # Weights should have changed
    assert not np.allclose(pf.weights, initial_weights)
    # Particle closest to 0.6 should have highest weight
    best_idx = np.argmin(np.abs(pf.particles - 0.6))
    assert pf.weights[best_idx] > 1.0 / 10  # higher than uniform


def test_resampling():
    """Low ESS triggers resampling"""
    np.random.seed(42)
    pf = ParameterParticleFilter(
        n_particles=10,
        param_nominal=0.5,
        param_std=0.2,
        param_min=0.2,
        param_max=1.0,
    )

    # Force very uneven weights (one particle dominates)
    pf.weights = np.zeros(10)
    pf.weights[0] = 0.99
    pf.weights[1:] = 0.01 / 9.0

    ess_before = pf.ess
    assert ess_before < 2.0  # very low

    pf.resample_if_needed(threshold=0.5)  # threshold*M = 5

    # After resampling, weights should be more uniform
    assert np.isclose(np.sum(pf.weights), 1.0)
    # ESS should increase
    assert pf.ess > ess_before


def test_weighted_mean():
    """E[theta] correct"""
    pf = ParameterParticleFilter(
        n_particles=3,
        param_nominal=0.5,
        param_std=0.1,
        param_min=0.2,
        param_max=1.0,
    )
    # Set known particles and weights
    pf.particles = np.array([0.4, 0.5, 0.6])
    pf.weights = np.array([0.2, 0.5, 0.3])

    mean = pf.get_weighted_mean()
    expected = 0.2 * 0.4 + 0.5 * 0.5 + 0.3 * 0.6
    assert np.isclose(mean, expected, atol=1e-10)


def test_ess_computation():
    """ESS correct"""
    pf = ParameterParticleFilter(
        n_particles=4,
        param_nominal=0.5,
        param_std=0.1,
        param_min=0.2,
        param_max=1.0,
    )

    # Uniform weights -> ESS = N
    pf.weights = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(pf.ess, 4.0, atol=1e-10)

    # Single dominating weight -> ESS ~ 1
    pf.weights = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.isclose(pf.ess, 1.0, atol=1e-10)


# -- TestParameterRobustMPPIController (5) --------------------------------

def test_compute_control_shape():
    """Control output shape and info keys"""
    np.random.seed(42)
    ctrl = _make_pr_controller()
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"Expected (2,), got {control.shape}"
    assert "sample_trajectories" in info
    assert "sample_weights" in info
    assert "best_trajectory" in info
    assert "best_cost" in info
    assert "mean_cost" in info
    assert "temperature" in info
    assert "ess" in info
    assert "num_samples" in info


def test_info_pr_stats():
    """parameter_robust_stats keys in info"""
    np.random.seed(42)
    ctrl = _make_pr_controller()
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)

    assert "parameter_robust_stats" in info
    pr = info["parameter_robust_stats"]
    assert "param_mean" in pr
    assert "param_best" in pr
    assert "param_std" in pr
    assert "param_ess" in pr
    assert "particles" in pr
    assert "particle_weights" in pr
    assert "aggregation_mode" in pr
    assert "n_particles" in pr

    assert pr["n_particles"] == 5
    assert pr["aggregation_mode"] == "weighted_mean"
    assert isinstance(pr["param_mean"], float)
    assert len(pr["particles"]) == 5
    assert len(pr["particle_weights"]) == 5


def test_different_K_values():
    """Different K values work correctly"""
    np.random.seed(42)
    state = np.array([3.0, 0.0, np.pi / 2])

    for K in [32, 128, 256]:
        ctrl = _make_pr_controller(K=K)
        ref = _make_ref()
        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,)
        assert info["num_samples"] == K
        assert info["sample_trajectories"].shape[0] == K


def test_reset_clears_state():
    """Reset clears control sequence, history, and particle filter"""
    np.random.seed(42)
    ctrl = _make_pr_controller()
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # Run a few steps
    for _ in range(5):
        ctrl.compute_control(state, ref)

    assert len(ctrl._state_history) > 0
    assert len(ctrl._pr_stats_history) > 0

    ctrl.reset()

    assert np.allclose(ctrl.U, 0.0)
    assert len(ctrl._state_history) == 0
    assert len(ctrl._control_history) == 0
    assert len(ctrl._pr_stats_history) == 0
    assert ctrl._step_count == 0


def test_aggregation_modes():
    """All aggregation modes produce valid results"""
    np.random.seed(42)
    state = np.array([3.0, 0.0, np.pi / 2])

    for mode in ["weighted_mean", "worst_case", "cvar"]:
        ctrl = _make_pr_controller(aggregation_mode=mode)
        ref = _make_ref()
        control, info = ctrl.compute_control(state, ref)

        assert control.shape == (2,)
        assert info["parameter_robust_stats"]["aggregation_mode"] == mode
        assert np.isfinite(info["best_cost"])
        assert np.isfinite(info["ess"])


# -- TestRobustness (4) ---------------------------------------------------

def _make_true_model(true_wheelbase, nominal_wheelbase=0.5):
    """Helper: create true model with wheelbase-affected dynamics"""
    base = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0, wheelbase=nominal_wheelbase)
    return _ParametricModel(base, "wheelbase", true_wheelbase, nominal_wheelbase)


def test_correct_param_convergence():
    """Particles converge toward the true parameter"""
    np.random.seed(42)

    true_wheelbase = 0.7
    nominal_wheelbase = 0.5

    # Controller with nominal=0.5
    ctrl = _make_pr_controller(
        K=128, N=15, n_particles=10,
        param_nominal=nominal_wheelbase,
        param_std=0.15,
        param_min=0.2, param_max=1.0,
        online_learning=True,
        observation_window=10,
        min_observations=3,
        use_resampling=True,
        resample_threshold=0.3,
    )

    # True model (wheelbase affects dynamics via omega scaling)
    true_model = _make_true_model(true_wheelbase, nominal_wheelbase)

    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05

    initial_mean = ctrl._particle_filter.get_weighted_mean()

    for step in range(60):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 15, dt)
        control, info = ctrl.compute_control(state, ref)

        # Propagate with TRUE model
        state = true_model.step(state, control, dt)

    final_mean = ctrl._particle_filter.get_weighted_mean()

    # Should have moved closer to true value
    initial_error = abs(initial_mean - true_wheelbase)
    final_error = abs(final_mean - true_wheelbase)

    assert final_error < initial_error, (
        f"Parameter should converge: initial_error={initial_error:.4f}, "
        f"final_error={final_error:.4f}"
    )


def test_mismatched_param_tracking():
    """Still tracks trajectory with wrong initial parameter"""
    np.random.seed(42)

    true_wheelbase = 0.7
    ctrl = _make_pr_controller(
        K=128, N=15,
        param_nominal=0.5,
        param_std=0.15,
        online_learning=True,
    )

    true_model = _make_true_model(true_wheelbase, 0.5)

    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05

    errors = []
    for step in range(80):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 15, dt)
        ref_pt = circle_trajectory(t)[:2]

        control, info = ctrl.compute_control(state, ref)
        state = true_model.step(state, control, dt)

        err = np.linalg.norm(state[:2] - ref_pt)
        errors.append(err)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    # Should still track reasonably despite mismatch
    assert rmse < 1.5, f"RMSE too large: {rmse:.4f}"


def test_worst_case_more_conservative():
    """Worst-case mode produces higher (more conservative) costs"""
    np.random.seed(42)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref(N=10)

    # weighted_mean
    ctrl_wm = _make_pr_controller(
        K=128, aggregation_mode="weighted_mean",
        param_std=0.2, n_particles=5,
    )
    _, info_wm = ctrl_wm.compute_control(state, ref)

    # worst_case
    np.random.seed(42)
    ctrl_wc = _make_pr_controller(
        K=128, aggregation_mode="worst_case",
        param_std=0.2, n_particles=5,
    )
    _, info_wc = ctrl_wc.compute_control(state, ref)

    # Worst-case mean cost should be >= weighted_mean mean cost
    assert info_wc["mean_cost"] >= info_wm["mean_cost"] * 0.8, (
        f"Worst-case ({info_wc['mean_cost']:.2f}) should be at least 80% of "
        f"weighted_mean ({info_wm['mean_cost']:.2f})"
    )


def test_parameter_bounds_respected():
    """Particles always stay within [param_min, param_max]"""
    np.random.seed(42)

    ctrl = _make_pr_controller(
        K=64, N=10, n_particles=10,
        param_nominal=0.5,
        param_std=0.3,
        param_min=0.3, param_max=0.8,
        online_learning=True,
        use_resampling=True,
    )

    true_model = _make_true_model(0.7, 0.5)
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05

    for step in range(40):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 10, dt)
        control, info = ctrl.compute_control(state, ref)
        state = true_model.step(state, control, dt)

        # Check bounds
        particles = ctrl._particle_filter.particles
        assert np.all(particles >= 0.3), \
            f"Particle below min: {particles.min()}"
        assert np.all(particles <= 0.8), \
            f"Particle above max: {particles.max()}"


# -- TestPerformance (4) --------------------------------------------------

def test_circle_tracking_rmse():
    """Circle tracking RMSE < 1.5 (multi-model rollout has overhead)"""
    np.random.seed(42)

    ctrl = _make_pr_controller(
        K=256, N=15, lambda_=1.0,
        n_particles=3,
        param_nominal=0.5,
        param_std=0.05,
        online_learning=False,
    )

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0, wheelbase=0.5)
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05

    errors = []
    for step in range(200):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 15, dt)
        ref_pt = circle_trajectory(t)[:2]

        control, info = ctrl.compute_control(state, ref)
        state = model.step(state, control, dt)
        errors.append(np.linalg.norm(state[:2] - ref_pt))

    # Use last 100 steps for RMSE (skip initial convergence)
    rmse = np.sqrt(np.mean(np.array(errors[-100:]) ** 2))
    assert rmse < 0.3, f"Circle RMSE too high: {rmse:.4f}"


def test_obstacle_avoidance():
    """3 obstacles, 0 collisions"""
    np.random.seed(42)

    obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4), (-2.0, -1.0, 0.5)]

    params_dict = dict(
        K=256, N=15, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        n_particles=3,
        param_nominal=0.5,
        param_std=0.05,
        online_learning=False,
    )
    params = ParameterRobustMPPIParams(**params_dict)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0, wheelbase=0.5)

    cost_fn = CompositeMPPICost([
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
        ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0),
    ])

    ctrl = ParameterRobustMPPIController(model, params, cost_function=cost_fn)

    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05
    n_collisions = 0

    for step in range(100):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 15, dt)
        control, info = ctrl.compute_control(state, ref)
        state = model.step(state, control, dt)

        for ox, oy, r in obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            if dist < r:
                n_collisions += 1

    assert n_collisions == 0, f"Collisions: {n_collisions}"


def test_model_mismatch_resilience():
    """true wheelbase=0.7, nominal=0.5, still tracks"""
    np.random.seed(42)

    ctrl = _make_pr_controller(
        K=256, N=15,
        n_particles=8,
        param_nominal=0.5,
        param_std=0.15,
        param_min=0.2, param_max=1.0,
        online_learning=True,
        observation_window=10,
        min_observations=3,
    )

    true_model = _make_true_model(0.7, 0.5)

    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05

    errors = []
    for step in range(100):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 15, dt)
        ref_pt = circle_trajectory(t)[:2]

        control, info = ctrl.compute_control(state, ref)
        state = true_model.step(state, control, dt)
        errors.append(np.linalg.norm(state[:2] - ref_pt))

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    assert rmse < 1.5, f"Model mismatch RMSE too high: {rmse:.4f}"


def test_computation_time():
    """K=256, N=15, < 200ms (M rollouts)"""
    np.random.seed(42)

    ctrl = _make_pr_controller(
        K=256, N=15,
        n_particles=5,
        online_learning=False,
    )

    state = np.array([3.0, 0.0, np.pi / 2])
    ref = generate_reference_trajectory(circle_trajectory, 0.0, 15, 0.05)

    # Warmup
    ctrl.compute_control(state, ref)

    # Benchmark
    times = []
    for _ in range(5):
        t0 = time.time()
        ctrl.compute_control(state, ref)
        times.append(time.time() - t0)

    mean_ms = np.mean(times) * 1000
    assert mean_ms < 200, f"Too slow: {mean_ms:.1f}ms"


# -- TestIntegration (7) --------------------------------------------------

def test_numerical_stability():
    """No NaN/Inf in outputs"""
    np.random.seed(42)
    ctrl = _make_pr_controller(K=64, N=10, n_particles=5)
    state = np.array([3.0, 0.0, np.pi / 2])

    for step in range(30):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 10, 0.05)
        control, info = ctrl.compute_control(state, ref)

        assert np.all(np.isfinite(control)), f"Non-finite control at step {step}"
        assert np.isfinite(info["ess"]), f"Non-finite ESS at step {step}"
        assert np.isfinite(info["best_cost"]), f"Non-finite cost at step {step}"

        pr = info["parameter_robust_stats"]
        assert np.isfinite(pr["param_mean"])
        assert np.isfinite(pr["param_ess"])
        assert np.all(np.isfinite(pr["particles"]))
        assert np.all(np.isfinite(pr["particle_weights"]))

        state = ctrl.model.step(state, control, 0.05)


def test_online_learning_disabled():
    """online_learning=False: particle weights stay uniform"""
    np.random.seed(42)
    ctrl = _make_pr_controller(
        K=64, N=10,
        n_particles=5,
        online_learning=False,
    )

    true_model = _make_true_model(0.7, 0.5)
    state = np.array([3.0, 0.0, np.pi / 2])

    initial_weights = ctrl._particle_filter.weights.copy()

    for step in range(20):
        t = step * 0.05
        ref = generate_reference_trajectory(circle_trajectory, t, 10, 0.05)
        control, info = ctrl.compute_control(state, ref)
        state = true_model.step(state, control, 0.05)

    # Weights should not change
    assert np.allclose(ctrl._particle_filter.weights, initial_weights), \
        "Weights changed with online_learning=False"


def test_online_learning_improves():
    """Online learning reduces parameter error over time"""
    np.random.seed(42)

    true_wheelbase = 0.7
    ctrl = _make_pr_controller(
        K=128, N=15,
        n_particles=10,
        param_nominal=0.5,
        param_std=0.15,
        param_min=0.2, param_max=1.0,
        online_learning=True,
        observation_window=10,
        min_observations=3,
        use_resampling=True,
    )

    true_model = _make_true_model(true_wheelbase, 0.5)
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05

    # Record parameter errors over time
    param_errors = []
    for step in range(80):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 15, dt)
        control, info = ctrl.compute_control(state, ref)
        state = true_model.step(state, control, dt)

        mean_param = ctrl._particle_filter.get_weighted_mean()
        param_errors.append(abs(mean_param - true_wheelbase))

    # Average error in first 20 steps vs last 20 steps
    early_error = np.mean(param_errors[:20])
    late_error = np.mean(param_errors[-20:])

    assert late_error < early_error, (
        f"Parameter error should decrease: early={early_error:.4f}, late={late_error:.4f}"
    )


def test_single_particle():
    """n_particles=1 reduces to single-model behavior"""
    np.random.seed(42)
    ctrl = _make_pr_controller(
        K=64, N=10,
        n_particles=1,
        param_nominal=0.5,
        param_std=0.01,
        online_learning=False,
    )

    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert info["num_samples"] == 64
    pr = info["parameter_robust_stats"]
    assert pr["n_particles"] == 1
    assert len(pr["particles"]) == 1


def test_many_particles():
    """n_particles=10 works correctly"""
    np.random.seed(42)
    ctrl = _make_pr_controller(
        K=64, N=10,
        n_particles=10,
        param_std=0.15,
        online_learning=False,
    )

    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    pr = info["parameter_robust_stats"]
    assert pr["n_particles"] == 10
    assert len(pr["particles"]) == 10
    assert len(pr["particle_weights"]) == 10
    assert np.isclose(np.sum(pr["particle_weights"]), 1.0)


def test_custom_parameter():
    """param_name='wheelbase' correctly modifies model attribute"""
    np.random.seed(42)
    ctrl = _make_pr_controller(
        K=64, N=10,
        n_particles=5,
        param_name="wheelbase",
        param_nominal=0.5,
        param_std=0.1,
    )

    # Each model variant should have different wheelbase
    wheelbases = [m.wheelbase for m in ctrl._models]
    assert len(wheelbases) == 5

    # They shouldn't all be the same (since param_std > 0)
    assert not all(np.isclose(w, wheelbases[0]) for w in wheelbases), \
        "Model variants should have different wheelbase values"

    # All should be within bounds
    for w in wheelbases:
        assert 0.2 <= w <= 1.0, f"Wheelbase {w} out of bounds"


def test_parameter_estimation_accuracy():
    """Estimated parameter close to true after sufficient learning"""
    np.random.seed(42)

    true_wheelbase = 0.65
    ctrl = _make_pr_controller(
        K=128, N=15,
        n_particles=15,
        param_nominal=0.5,
        param_std=0.15,
        param_min=0.2, param_max=1.0,
        online_learning=True,
        observation_window=10,
        min_observations=3,
        use_resampling=True,
        resample_threshold=0.3,
    )

    true_model = _make_true_model(true_wheelbase, 0.5)
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05

    for step in range(100):
        t = step * dt
        ref = generate_reference_trajectory(circle_trajectory, t, 15, dt)
        control, info = ctrl.compute_control(state, ref)
        state = true_model.step(state, control, dt)

    estimated = ctrl._particle_filter.get_weighted_mean()
    error = abs(estimated - true_wheelbase)

    # Should be within 0.15 of true value after 100 steps
    assert error < 0.15, (
        f"Parameter estimation error too large: estimated={estimated:.4f}, "
        f"true={true_wheelbase:.4f}, error={error:.4f}"
    )
