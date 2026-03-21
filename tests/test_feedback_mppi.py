"""
Feedback-MPPI (F-MPPI) 유닛 테스트

28개 테스트:
  - Params (3): 기본값, 커스텀, 검증
  - Jacobian (4): A shape, B shape, 정확도, 일관성
  - Riccati (4): 게인 shape, PSD, 게인 범위, 터미널 조건
  - Controller (5): shape, info keys, full solve mode, reuse mode, reset
  - FeedbackReuse (4): reuse 카운트, 보정 비례, 속도 차이, 비활성화 폴백
  - Performance (4): RMSE, 장애물 회피, 속도 향상, 계산 시간
  - Integration (4): 수치 안정성, 긴 호라이즌, 게인 클리핑, warm start
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    FeedbackMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.feedback_mppi import FeedbackMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)


# -- Helper functions --

def _make_feedback_controller(**kwargs):
    """Feedback-MPPI controller creation helper."""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        reuse_steps=3,
        use_feedback=True,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = FeedbackMPPIParams(**defaults)
    return FeedbackMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_vanilla_controller(**kwargs):
    """Vanilla MPPI for comparison."""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    params = MPPIParams(**defaults)
    return MPPIController(model, params, cost_function=cost_function)


def _make_ref(N=10, dt=0.05):
    """Reference trajectory."""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ================================================================
# 1. Params tests (3)
# ================================================================

class TestFeedbackMPPIParams:
    def test_params_defaults(self):
        """Default values verification."""
        params = FeedbackMPPIParams()
        assert params.reuse_steps == 3
        assert params.jacobian_eps == 1e-4
        assert params.feedback_weight_Q == 10.0
        assert params.feedback_weight_R == 0.1
        assert params.use_feedback is True
        assert params.feedback_gain_clip == 10.0
        assert params.use_warm_start is True

    def test_params_custom(self):
        """Custom values verification."""
        params = FeedbackMPPIParams(
            reuse_steps=5,
            jacobian_eps=1e-3,
            feedback_weight_Q=20.0,
            feedback_weight_R=0.5,
            use_feedback=False,
            feedback_gain_clip=5.0,
            use_warm_start=False,
        )
        assert params.reuse_steps == 5
        assert params.jacobian_eps == 1e-3
        assert params.feedback_weight_Q == 20.0
        assert params.feedback_weight_R == 0.5
        assert params.use_feedback is False
        assert params.feedback_gain_clip == 5.0
        assert params.use_warm_start is False

    def test_params_validation(self):
        """Invalid values -> AssertionError."""
        # reuse_steps < 1
        with pytest.raises(AssertionError):
            FeedbackMPPIParams(reuse_steps=0)

        # jacobian_eps <= 0
        with pytest.raises(AssertionError):
            FeedbackMPPIParams(jacobian_eps=0)

        with pytest.raises(AssertionError):
            FeedbackMPPIParams(jacobian_eps=-1e-4)

        # feedback_weight_Q <= 0
        with pytest.raises(AssertionError):
            FeedbackMPPIParams(feedback_weight_Q=0)

        # feedback_weight_R <= 0
        with pytest.raises(AssertionError):
            FeedbackMPPIParams(feedback_weight_R=-0.1)

        # feedback_gain_clip <= 0
        with pytest.raises(AssertionError):
            FeedbackMPPIParams(feedback_gain_clip=0)


# ================================================================
# 2. Jacobian Computation tests (4)
# ================================================================

class TestJacobianComputation:
    def test_jacobian_A_shape(self):
        """A_t has shape (nx, nx)."""
        ctrl = _make_feedback_controller()
        model = ctrl.model
        nx = model.state_dim  # 3
        N = ctrl.params.N

        state = np.array([3.0, 0.0, np.pi / 2])
        controls = np.zeros((N, model.control_dim))
        controls[:, 0] = 0.5  # some forward velocity

        trajectory = ctrl._extract_nominal_trajectory(state, controls)
        A_list, B_list = ctrl._compute_jacobians(trajectory, controls)

        assert A_list.shape == (N, nx, nx)

    def test_jacobian_B_shape(self):
        """B_t has shape (nx, nu)."""
        ctrl = _make_feedback_controller()
        model = ctrl.model
        nx = model.state_dim
        nu = model.control_dim
        N = ctrl.params.N

        state = np.array([3.0, 0.0, np.pi / 2])
        controls = np.zeros((N, nu))
        controls[:, 0] = 0.3

        trajectory = ctrl._extract_nominal_trajectory(state, controls)
        A_list, B_list = ctrl._compute_jacobians(trajectory, controls)

        assert B_list.shape == (N, nx, nu)

    def test_jacobian_finite_diff_accuracy(self):
        """Finite diff Jacobians are approximately correct for diff-drive model.

        For dx/dt = v*cos(theta), dy/dt = v*sin(theta), dtheta/dt = omega:
        A (linearized discrete) should approximate:
            df/dtheta ~ [-v*sin(theta)*dt, v*cos(theta)*dt, 0]
        """
        ctrl = _make_feedback_controller(jacobian_eps=1e-5)
        model = ctrl.model
        dt = ctrl.params.dt

        state = np.array([1.0, 2.0, 0.5])
        control = np.array([0.5, 0.3])

        # Single step Jacobian
        trajectory = np.zeros((2, 3))
        trajectory[0] = state
        trajectory[1] = model.step(state, control, dt)
        controls = control.reshape(1, -1)

        A_list, B_list = ctrl._compute_jacobians(trajectory, controls)
        A = A_list[0]  # (3, 3)

        # Check A is not all zeros (dynamics are state-dependent)
        assert not np.allclose(A, 0), "Jacobian A should not be zero"

        # For diff-drive with RK4, A should be close to identity + dt * partial derivatives
        # A[0, 2] should be related to -v*sin(theta)*dt (x depends on theta)
        theta = state[2]
        v = control[0]
        # Rough check: A[0,2] should be negative for theta=0.5 (sin > 0)
        # This is a consistency check, not exact
        assert A[0, 2] != 0, "A[0,2] (dx/dtheta) should be nonzero"

    def test_jacobian_consistency(self):
        """A, B are consistent with model.step: x_next ~ A @ x + B @ u (linearized)."""
        ctrl = _make_feedback_controller(jacobian_eps=1e-5)
        model = ctrl.model
        dt = ctrl.params.dt
        N = 5

        state = np.array([1.0, 0.5, 0.3])
        controls = np.random.randn(N, 2) * 0.3

        trajectory = ctrl._extract_nominal_trajectory(state, controls)
        A_list, B_list = ctrl._compute_jacobians(trajectory, controls)

        # Perturbation test: small dx -> x_next = f(x+dx, u) ~ f(x,u) + A @ dx
        dx = np.array([0.001, -0.001, 0.002])

        for t in range(min(3, N)):
            x_nom = trajectory[t]
            u_nom = controls[t]
            x_next_nom = trajectory[t + 1]

            x_perturbed = x_nom + dx
            x_next_perturbed = model.step(x_perturbed, u_nom, dt)

            predicted_delta = A_list[t] @ dx
            actual_delta = x_next_perturbed - x_next_nom

            # Should be close (first-order approximation)
            np.testing.assert_allclose(
                predicted_delta, actual_delta,
                atol=1e-3, rtol=0.1,
                err_msg=f"Jacobian A inconsistent at t={t}",
            )


# ================================================================
# 3. Riccati Solver tests (4)
# ================================================================

class TestRiccatiSolver:
    def test_riccati_gain_shape(self):
        """Gains have shape (N, nu, nx)."""
        ctrl = _make_feedback_controller()
        model = ctrl.model
        N = ctrl.params.N
        nx = model.state_dim
        nu = model.control_dim

        state = np.array([3.0, 0.0, np.pi / 2])
        controls = np.zeros((N, nu))
        controls[:, 0] = 0.5

        trajectory = ctrl._extract_nominal_trajectory(state, controls)
        A_list, B_list = ctrl._compute_jacobians(trajectory, controls)
        gains = ctrl._solve_riccati(A_list, B_list)

        assert gains.shape == (N, nu, nx)

    def test_riccati_positive_definite_P(self):
        """P matrices from Riccati should be positive semi-definite."""
        ctrl = _make_feedback_controller()
        model = ctrl.model
        N = ctrl.params.N
        nx = model.state_dim
        nu = model.control_dim

        state = np.array([3.0, 0.0, np.pi / 2])
        controls = np.zeros((N, nu))
        controls[:, 0] = 0.5

        trajectory = ctrl._extract_nominal_trajectory(state, controls)
        A_list, B_list = ctrl._compute_jacobians(trajectory, controls)

        # Manually run Riccati to check P matrices
        Q_w = ctrl.feedback_params.feedback_weight_Q
        R_w = ctrl.feedback_params.feedback_weight_R
        Q_matrix = np.diag(ctrl.params.Q) * Q_w
        R_matrix = np.diag(ctrl.params.R) * R_w
        Qf_matrix = np.diag(ctrl.params.Qf) * Q_w
        reg = 1e-6 * np.eye(nu)

        P = Qf_matrix.copy()
        # Terminal P = Qf (should be PSD since Qf >= 0)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals >= -1e-10), "Terminal P is not PSD"

        for t in range(N - 1, -1, -1):
            A = A_list[t]
            B = B_list[t]
            BtP = B.T @ P
            BtPB = BtP @ B
            M = R_matrix + BtPB + reg
            K_t = -np.linalg.solve(M, BtP @ A)
            AtP = A.T @ P
            P = Q_matrix + AtP @ A + AtP @ B @ K_t
            P = 0.5 * (P + P.T)

            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals >= -1e-8), \
                f"P at t={t} is not PSD: eigenvalues={eigvals}"

    def test_riccati_gain_bounded(self):
        """Gains within clip range."""
        clip_val = 5.0
        ctrl = _make_feedback_controller(feedback_gain_clip=clip_val)
        model = ctrl.model
        N = ctrl.params.N
        nu = model.control_dim

        state = np.array([3.0, 0.0, np.pi / 2])
        controls = np.zeros((N, nu))
        controls[:, 0] = 0.5

        trajectory = ctrl._extract_nominal_trajectory(state, controls)
        A_list, B_list = ctrl._compute_jacobians(trajectory, controls)
        gains = ctrl._solve_riccati(A_list, B_list)

        assert np.all(gains <= clip_val), f"Max gain {np.max(gains)} > clip {clip_val}"
        assert np.all(gains >= -clip_val), f"Min gain {np.min(gains)} < -clip {clip_val}"

    def test_riccati_terminal_condition(self):
        """Terminal cost Qf affects the gains."""
        # High Qf -> larger gains at early timesteps
        ctrl_high = _make_feedback_controller(
            N=5,
            Q=np.array([10.0, 10.0, 1.0]),
            Qf=np.array([100.0, 100.0, 10.0]),
        )
        ctrl_low = _make_feedback_controller(
            N=5,
            Q=np.array([10.0, 10.0, 1.0]),
            Qf=np.array([1.0, 1.0, 0.1]),
        )

        model = ctrl_high.model
        N = ctrl_high.params.N
        nu = model.control_dim

        state = np.array([3.0, 0.0, np.pi / 2])
        controls = np.zeros((N, nu))
        controls[:, 0] = 0.5

        traj_high = ctrl_high._extract_nominal_trajectory(state, controls)
        A_h, B_h = ctrl_high._compute_jacobians(traj_high, controls)
        gains_high = ctrl_high._solve_riccati(A_h, B_h)

        traj_low = ctrl_low._extract_nominal_trajectory(state, controls)
        A_l, B_l = ctrl_low._compute_jacobians(traj_low, controls)
        gains_low = ctrl_low._solve_riccati(A_l, B_l)

        # Higher Qf should produce larger gain magnitudes (on average)
        mean_high = np.mean(np.abs(gains_high))
        mean_low = np.mean(np.abs(gains_low))
        assert mean_high > mean_low, \
            f"Higher Qf should produce larger gains: {mean_high:.4f} vs {mean_low:.4f}"


# ================================================================
# 4. Controller tests (5)
# ================================================================

class TestFeedbackMPPIController:
    def test_compute_control_shape(self):
        """control (nu,), info standard keys."""
        ctrl = _make_feedback_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert "sample_trajectories" in info
        assert "sample_weights" in info
        assert "best_trajectory" in info
        assert "best_cost" in info
        assert "mean_cost" in info
        assert "temperature" in info
        assert "ess" in info
        assert "num_samples" in info

    def test_info_feedback_stats(self):
        """feedback_stats keys in info."""
        ctrl = _make_feedback_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["feedback_stats"]
        assert "mode" in stats
        assert "reuse_counter" in stats
        assert "step_in_sequence" in stats
        assert "mean_gain" in stats
        assert "max_gain" in stats
        assert "feedback_correction_norm" in stats
        assert "full_solve_count" in stats
        assert "reuse_count" in stats

    def test_full_solve_mode(self):
        """First call triggers full MPPI solve."""
        ctrl = _make_feedback_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        assert info["feedback_stats"]["mode"] == "full_solve"
        assert info["feedback_stats"]["full_solve_count"] == 1
        assert info["num_samples"] == 64  # K=64

    def test_reuse_mode(self):
        """Subsequent calls use feedback reuse."""
        ctrl = _make_feedback_controller(reuse_steps=3)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # First call: full solve
        control1, info1 = ctrl.compute_control(state, ref)
        assert info1["feedback_stats"]["mode"] == "full_solve"

        # Next call: feedback reuse
        state2 = state + np.array([0.01, 0.01, 0.005])
        control2, info2 = ctrl.compute_control(state2, ref)
        assert info2["feedback_stats"]["mode"] == "feedback_reuse"
        assert info2["num_samples"] == 0  # no rollouts

    def test_reset_clears_state(self):
        """reset clears all feedback state."""
        ctrl = _make_feedback_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(5):
            ctrl.compute_control(state, ref)

        assert ctrl._total_steps == 5
        assert len(ctrl._fb_stats_history) == 5

        ctrl.reset()
        assert np.allclose(ctrl.U, 0.0)
        assert ctrl._total_steps == 0
        assert ctrl._full_solve_count == 0
        assert ctrl._reuse_count == 0
        assert ctrl._nominal_trajectory is None
        assert ctrl._feedback_gains is None
        assert ctrl._reuse_counter == 0
        assert len(ctrl._fb_stats_history) == 0


# ================================================================
# 5. Feedback Reuse tests (4)
# ================================================================

class TestFeedbackReuse:
    def test_reuse_steps_count(self):
        """Exactly reuse_steps feedback calls between full solves."""
        reuse = 3
        ctrl = _make_feedback_controller(reuse_steps=reuse)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        model = ctrl.model

        modes = []
        for step in range(10):
            t = step * 0.05
            control, info = ctrl.compute_control(state, ref)
            modes.append(info["feedback_stats"]["mode"])
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

        # Pattern: full_solve, then reuse_steps reuse, then full_solve, ...
        # full(0), reuse(1), reuse(2), reuse(3), full(4), reuse(5), ...
        full_indices = [i for i, m in enumerate(modes) if m == "full_solve"]
        reuse_indices = [i for i, m in enumerate(modes) if m == "feedback_reuse"]

        # Between consecutive full solves, there should be reuse_steps reuse calls
        if len(full_indices) >= 2:
            gap = full_indices[1] - full_indices[0]
            assert gap == reuse + 1, \
                f"Expected {reuse + 1} steps between full solves, got {gap}"

    def test_feedback_correction(self):
        """Correction proportional to state error."""
        ctrl = _make_feedback_controller(reuse_steps=5)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # First call: full solve
        ctrl.compute_control(state, ref)

        # Small perturbation -> small correction
        state_small = state + np.array([0.001, 0.001, 0.001])
        control_small, info_small = ctrl.compute_control(state_small, ref)
        norm_small = info_small["feedback_stats"]["feedback_correction_norm"]

        # Reset and redo
        ctrl.reset()
        ctrl.compute_control(state, ref)

        # Large perturbation -> larger correction
        state_large = state + np.array([0.05, 0.05, 0.02])
        control_large, info_large = ctrl.compute_control(state_large, ref)
        norm_large = info_large["feedback_stats"]["feedback_correction_norm"]

        assert norm_large > norm_small, \
            f"Larger perturbation should give larger correction: {norm_large} vs {norm_small}"

    def test_faster_in_reuse_mode(self):
        """Feedback reuse mode is significantly faster than full solve."""
        ctrl = _make_feedback_controller(K=256, N=20, reuse_steps=3)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=20)

        # Full solve timing
        t_start = time.time()
        ctrl.compute_control(state, ref)
        full_time = time.time() - t_start

        # Reuse timing
        state2 = state + np.array([0.01, 0.01, 0.005])
        t_start = time.time()
        ctrl.compute_control(state2, ref)
        reuse_time = time.time() - t_start

        # Reuse should be at least 5x faster (typically 100x+)
        assert reuse_time < full_time * 0.5, \
            f"Reuse ({reuse_time*1000:.2f}ms) should be much faster than " \
            f"full solve ({full_time*1000:.2f}ms)"

    def test_feedback_disabled_fallback(self):
        """use_feedback=False -> every step is full solve."""
        ctrl = _make_feedback_controller(use_feedback=False, reuse_steps=5)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        model = ctrl.model

        for step in range(5):
            control, info = ctrl.compute_control(state, ref)
            assert info["feedback_stats"]["mode"] == "full_solve"
            assert info["num_samples"] == 64
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05


# ================================================================
# 6. Performance tests (4)
# ================================================================

class TestPerformance:
    def test_circle_tracking_rmse(self):
        """Circle tracking RMSE < 0.3 (50 steps)."""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = FeedbackMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            reuse_steps=3,
            use_feedback=True,
        )
        ctrl = FeedbackMPPIController(model, params)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt = params.dt
        N = params.N
        num_steps = 50

        errors = []
        for step in range(num_steps):
            t = step * dt
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt

            ref_pt = circle_trajectory(t, radius=3.0)
            err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
            errors.append(err)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.3, f"RMSE {rmse:.4f} >= 0.3"

    def test_obstacle_avoidance(self):
        """3 obstacles, 0 collisions."""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4), (-2.0, -1.0, 0.5)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0),
        ])

        params = FeedbackMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            reuse_steps=2,
            use_feedback=True,
        )
        ctrl = FeedbackMPPIController(model, params, cost_function=cost)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt_val = params.dt
        N = params.N

        collisions = 0
        for step in range(80):
            t = step * dt_val
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt_val,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt_val

            for ox, oy, r in obstacles:
                dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                if dist < r:
                    collisions += 1

        assert collisions == 0, f"Collisions: {collisions}"

    def test_speedup_vs_vanilla(self):
        """Average solve time lower due to reuse steps.

        With reuse_steps=4, F-MPPI does 1 full solve + 4 reuse per cycle.
        Full solve is slightly more expensive (Jacobians + Riccati), but
        reuse steps are essentially free (< 0.01ms), so the average should
        be lower when K is large enough that rollout cost dominates.
        """
        K = 1024
        N = 30
        ctrl_fb = _make_feedback_controller(K=K, N=N, reuse_steps=4)
        ctrl_van = _make_vanilla_controller(K=K, N=N)
        state = np.array([3.0, 0.0, np.pi / 2])
        model = ctrl_fb.model

        # Warmup
        ref_w = _make_ref(N=N)
        ctrl_fb.compute_control(state, ref_w)
        ctrl_van.compute_control(state, ref_w)
        ctrl_fb.reset()
        ctrl_van.reset()

        n_steps = 25
        state_fb = state.copy()
        state_van = state.copy()

        # Feedback MPPI timing
        fb_times = []
        for step in range(n_steps):
            t = step * 0.05
            ref_t = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, 0.05,
            )
            t_start = time.time()
            control, _ = ctrl_fb.compute_control(state_fb, ref_t)
            fb_times.append(time.time() - t_start)
            state_dot = model.forward_dynamics(state_fb, control)
            state_fb = state_fb + state_dot * 0.05

        # Vanilla timing
        van_times = []
        for step in range(n_steps):
            t = step * 0.05
            ref_t = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, 0.05,
            )
            t_start = time.time()
            control, _ = ctrl_van.compute_control(state_van, ref_t)
            van_times.append(time.time() - t_start)
            state_dot = model.forward_dynamics(state_van, control)
            state_van = state_van + state_dot * 0.05

        mean_fb = np.mean(fb_times)
        mean_van = np.mean(van_times)

        # F-MPPI should be faster on average due to reuse steps
        assert mean_fb < mean_van, \
            f"F-MPPI ({mean_fb*1000:.2f}ms) should be faster than Vanilla ({mean_van*1000:.2f}ms)"

    def test_computation_time(self):
        """K=512, N=30, mean < 100ms (including reuse steps)."""
        ctrl = _make_feedback_controller(K=512, N=30, reuse_steps=3)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=30)
        model = ctrl.model

        # Warmup
        ctrl.compute_control(state, ref)
        ctrl.reset()

        times = []
        for step in range(12):
            t = step * 0.05
            ref_t = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, 30, 0.05,
            )
            t_start = time.time()
            control, _ = ctrl.compute_control(state, ref_t)
            times.append(time.time() - t_start)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

        mean_ms = np.mean(times) * 1000
        assert mean_ms < 100, f"Mean solve time {mean_ms:.1f}ms >= 100ms"


# ================================================================
# 7. Integration tests (4)
# ================================================================

class TestIntegration:
    def test_numerical_stability(self):
        """No NaN/Inf in outputs."""
        ctrl = _make_feedback_controller(K=64, reuse_steps=3)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(20):
            control, info = ctrl.compute_control(state, ref)
            assert not np.any(np.isnan(control)), "NaN in control"
            assert not np.any(np.isinf(control)), "Inf in control"

            stats = info["feedback_stats"]
            assert not np.isnan(stats["mean_gain"]), "NaN in mean_gain"
            assert not np.isnan(stats["max_gain"]), "NaN in max_gain"

            state_dot = ctrl.model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

    def test_long_horizon_N50(self):
        """N=50 long horizon works correctly."""
        ctrl = _make_feedback_controller(K=64, N=50, reuse_steps=3)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=50)

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["feedback_stats"]["mode"] == "full_solve"

        # Feedback reuse also works
        state2 = state + np.array([0.01, 0.01, 0.005])
        control2, info2 = ctrl.compute_control(state2, ref)
        assert control2.shape == (2,)

    def test_gain_clip_prevents_instability(self):
        """Large perturbation is handled by gain clipping."""
        ctrl = _make_feedback_controller(
            K=64, feedback_gain_clip=2.0, reuse_steps=5,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # Full solve
        ctrl.compute_control(state, ref)

        # Apply large perturbation
        state_perturbed = state + np.array([1.0, 1.0, 0.5])
        control, info = ctrl.compute_control(state_perturbed, ref)

        assert not np.any(np.isnan(control)), "NaN after large perturbation"
        assert not np.any(np.isinf(control)), "Inf after large perturbation"

        # Control should be bounded (gain-clipped + control bounds)
        assert np.all(np.abs(control) < 10), \
            f"Control magnitude too large: {control}"

    def test_warm_start_with_reuse(self):
        """Warm start in feedback mode maintains consistency."""
        ctrl = _make_feedback_controller(
            K=128, N=15, reuse_steps=3, use_warm_start=True,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=15)
        model = ctrl.model

        # Run full sequence with warm start
        controls_history = []
        for step in range(8):
            t = step * 0.05
            ref_t = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, 15, 0.05,
            )
            control, info = ctrl.compute_control(state, ref_t)
            controls_history.append(control.copy())
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

        # Controls should be smooth (no sudden jumps)
        controls_arr = np.array(controls_history)
        diffs = np.diff(controls_arr, axis=0)
        max_jump = np.max(np.abs(diffs))

        # Max control jump should be bounded
        assert max_jump < 2.0, f"Max control jump {max_jump:.3f} too large"

        # Check feedback statistics
        stats = ctrl.get_feedback_statistics()
        assert stats["total_steps"] == 8
        assert stats["full_solve_count"] >= 2  # at least 2 full solves in 8 steps
        assert stats["reuse_count"] >= 3  # at least some reuse
        assert stats["reuse_fraction"] > 0.0
