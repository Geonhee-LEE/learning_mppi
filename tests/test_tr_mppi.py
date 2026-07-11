"""
TR-MPPI (Trust Region MPPI) 유닛 테스트 — 41번째 변형

~26개 테스트:
  - Params (5): 기본값, 커스텀, 검증(trust_region_radius/n_iters/entropy_floor/cov_max)
  - HaltonLCDSampler / icdf (4): 결정론성, shape, 통계, 역정규 CDF 단조성
  - Controller (5): construction/repr, shape, info keys, tr_stats subkeys, sample 통계
  - Trust Region (5): KL 캡, 큰 radius 미적용, use_kl_bound=False, n_iters>1, 결정론적 재현성
  - Covariance (3): adapt_covariance 엔트로피 하한, 상한, reset
  - Integration (4): receding horizon shift, 제어 바운드 클리핑, 수치 안정성, 원형 추적 RMSE
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    TRMPPIParams,
)
from mppi_controller.controllers.mppi.tr_mppi import (
    TRMPPIController,
    HaltonLCDSampler,
    _inverse_normal_cdf,
    _van_der_corput,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)


# -- Helper functions --

def _make_tr_controller(**kwargs):
    """TR-MPPI controller creation helper."""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = TRMPPIParams(**defaults)
    return TRMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_ref(N=10, dt=0.05, t0=0.0):
    """Reference trajectory."""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        t0, N, dt,
    )


# ================================================================
# 1. Params tests (5)
# ================================================================

class TestTRMPPIParams:
    def test_params_defaults(self):
        """Default values verification."""
        params = TRMPPIParams()
        assert params.trust_region_radius == 1.0
        assert params.use_kl_bound is True
        assert params.n_iters == 1
        assert params.use_deterministic_sampling is False
        assert params.adapt_covariance is False
        assert params.cov_step_size == 0.2
        assert params.entropy_floor_scale == 0.3
        assert params.cov_max_scale == 4.0

    def test_params_custom(self):
        """Custom values verification."""
        params = TRMPPIParams(
            trust_region_radius=0.05,
            use_kl_bound=False,
            n_iters=3,
            use_deterministic_sampling=True,
            adapt_covariance=True,
            cov_step_size=0.5,
            entropy_floor_scale=0.5,
            cov_max_scale=2.0,
        )
        assert params.trust_region_radius == 0.05
        assert params.use_kl_bound is False
        assert params.n_iters == 3
        assert params.use_deterministic_sampling is True
        assert params.adapt_covariance is True
        assert params.cov_step_size == 0.5
        assert params.entropy_floor_scale == 0.5
        assert params.cov_max_scale == 2.0

    def test_params_validation_radius(self):
        """trust_region_radius <= 0 -> AssertionError."""
        with pytest.raises(AssertionError):
            TRMPPIParams(trust_region_radius=0.0)
        with pytest.raises(AssertionError):
            TRMPPIParams(trust_region_radius=-1.0)

    def test_params_validation_iters_covstep(self):
        """n_iters < 1, cov_step_size < 0 -> AssertionError."""
        with pytest.raises(AssertionError):
            TRMPPIParams(n_iters=0)
        with pytest.raises(AssertionError):
            TRMPPIParams(cov_step_size=-0.1)

    def test_params_validation_entropy_covmax(self):
        """entropy_floor_scale out of (0,1], cov_max_scale < 1 -> AssertionError."""
        # entropy_floor_scale > 1
        with pytest.raises(AssertionError):
            TRMPPIParams(entropy_floor_scale=1.5)
        # entropy_floor_scale <= 0
        with pytest.raises(AssertionError):
            TRMPPIParams(entropy_floor_scale=0.0)
        # cov_max_scale < 1
        with pytest.raises(AssertionError):
            TRMPPIParams(cov_max_scale=0.5)


# ================================================================
# 2. HaltonLCDSampler / inverse normal CDF tests (4)
# ================================================================

class TestHaltonLCDSampler:
    def test_unit_samples_deterministic(self):
        """unit_samples is deterministic: two calls -> identical output."""
        sampler = HaltonLCDSampler(np.array([0.5, 0.5]))
        z1 = sampler.unit_samples(64, 10, 2)
        z2 = sampler.unit_samples(64, 10, 2)
        assert np.allclose(z1, z2), "LCD samples must be reproducible"

    def test_unit_samples_shape(self):
        """unit_samples returns (K, N, nu)."""
        sampler = HaltonLCDSampler(np.array([0.5, 0.5]))
        z = sampler.unit_samples(128, 20, 2)
        assert z.shape == (128, 20, 2)

    def test_unit_samples_statistics(self):
        """Roughly zero-mean, unit-std standard-normal-like samples."""
        sampler = HaltonLCDSampler(np.array([0.5, 0.5]))
        z = sampler.unit_samples(512, 15, 2)
        assert abs(float(np.mean(z))) < 0.15, f"mean {np.mean(z)} not ~0"
        assert abs(float(np.std(z)) - 1.0) < 0.2, f"std {np.std(z)} not ~1"
        assert not np.any(np.isnan(z)) and not np.any(np.isinf(z))

    def test_inverse_normal_cdf(self):
        """Phi^-1 is monotonic and Phi^-1(0.5) ~ 0."""
        # midpoint -> 0
        mid = _inverse_normal_cdf(np.array([0.5]))
        assert abs(float(mid[0])) < 1e-6, f"Phi^-1(0.5)={mid[0]} not ~0"

        # monotonic increasing
        u = np.linspace(0.01, 0.99, 50)
        q = _inverse_normal_cdf(u)
        assert np.all(np.diff(q) > 0), "Phi^-1 must be strictly increasing"

        # symmetry: Phi^-1(0.1) ~ -Phi^-1(0.9)
        lo = _inverse_normal_cdf(np.array([0.1]))
        hi = _inverse_normal_cdf(np.array([0.9]))
        assert abs(float(lo[0]) + float(hi[0])) < 1e-6

        # van der Corput sanity (values in [0,1))
        vdc = _van_der_corput(32, 2, skip=1)
        assert np.all(vdc >= 0) and np.all(vdc < 1)


# ================================================================
# 3. Controller construction / basic IO tests (5)
# ================================================================

class TestTRMPPIController:
    def test_construction_and_repr(self):
        """Controller constructs and repr contains key info."""
        ctrl = _make_tr_controller(trust_region_radius=0.5, n_iters=2)
        r = repr(ctrl)
        assert "TRMPPIController" in r
        assert "trust_region_radius" in r
        assert "deterministic" in r
        # internal attrs
        assert np.allclose(ctrl._sigma_scale, 1.0)
        assert ctrl._base_sigma.shape == (2,)
        assert isinstance(ctrl._lcd, HaltonLCDSampler)
        assert ctrl._scaled_count == 0
        assert ctrl._total_iters == 0

    def test_compute_control_shape_finite(self):
        """control (2,), finite."""
        ctrl = _make_tr_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))

    def test_info_standard_keys(self):
        """info has standard MPPI keys + valid weights/ess/trajectories."""
        ctrl = _make_tr_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        _, info = ctrl.compute_control(state, ref)
        for key in [
            "sample_trajectories", "sample_weights", "best_trajectory",
            "best_cost", "mean_cost", "temperature", "ess", "num_samples",
        ]:
            assert key in info, f"missing info key {key}"

        # sample_trajectories shape (K, N+1, nx)
        assert info["sample_trajectories"].shape == (64, 11, 3)
        # weights sum to 1
        w = info["sample_weights"]
        assert w.shape == (64,)
        assert abs(float(np.sum(w)) - 1.0) < 1e-6
        # ess within (0, K]
        assert 0 < info["ess"] <= 64 + 1e-6
        assert info["num_samples"] == 64

    def test_info_tr_stats_subkeys(self):
        """info['tr_stats'] has all expected subkeys."""
        ctrl = _make_tr_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        _, info = ctrl.compute_control(state, ref)
        assert "tr_stats" in info
        stats = info["tr_stats"]
        for key in [
            "kl_divergence", "trust_region_radius",
            "step_scaled", "deterministic", "sigma_scale",
        ]:
            assert key in stats, f"missing tr_stats key {key}"
        assert isinstance(stats["step_scaled"], (bool, np.bool_))
        assert isinstance(stats["deterministic"], (bool, np.bool_))
        assert stats["sigma_scale"].shape == (2,)
        assert stats["kl_divergence"] >= 0.0

    def test_get_tr_statistics(self):
        """get_tr_statistics aggregates KL history."""
        ctrl = _make_tr_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        for step in range(5):
            ctrl.compute_control(state, _make_ref(t0=step * 0.05))
        stats = ctrl.get_tr_statistics()
        assert stats["n_steps"] == 5
        assert stats["mean_kl"] >= 0.0
        assert stats["max_kl"] >= stats["mean_kl"] - 1e-9
        assert 0.0 <= stats["scaled_fraction"] <= 1.0
        assert stats["sigma_scale"].shape == (2,)


# ================================================================
# 4. Trust Region behavior tests (5)
# ================================================================

class TestTrustRegion:
    def test_kl_bound_caps_step(self):
        """Very small radius -> KL bound active, step_scaled True at least once."""
        ctrl = _make_tr_controller(trust_region_radius=1e-4, use_kl_bound=True)
        state = np.array([3.0, 0.0, np.pi / 2])

        scaled_any = False
        for step in range(8):
            _, info = ctrl.compute_control(state, _make_ref(t0=step * 0.05))
            kl = info["tr_stats"]["kl_divergence"]
            assert kl >= 0.0
            # When scaling is applied the recorded KL exceeds the (tiny) radius
            if info["tr_stats"]["step_scaled"]:
                scaled_any = True
                assert kl > ctrl.tr_params.trust_region_radius
        assert scaled_any, "small radius should trigger step scaling"
        assert ctrl._scaled_count > 0

    def test_large_radius_no_scaling(self):
        """Very large radius -> step_scaled stays False."""
        ctrl = _make_tr_controller(trust_region_radius=1e6, use_kl_bound=True)
        state = np.array([3.0, 0.0, np.pi / 2])
        for step in range(6):
            _, info = ctrl.compute_control(state, _make_ref(t0=step * 0.05))
            assert info["tr_stats"]["step_scaled"] is False
        assert ctrl._scaled_count == 0

    def test_kl_bound_disabled(self):
        """use_kl_bound=False -> never scale even with tiny radius."""
        ctrl = _make_tr_controller(trust_region_radius=1e-6, use_kl_bound=False)
        state = np.array([3.0, 0.0, np.pi / 2])
        for step in range(6):
            _, info = ctrl.compute_control(state, _make_ref(t0=step * 0.05))
            assert info["tr_stats"]["step_scaled"] is False
        assert ctrl._scaled_count == 0

    def test_small_radius_smaller_update(self):
        """KL bound caps update: tiny radius -> first control much smaller."""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        common = dict(
            K=64, N=10, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]), R=np.array([0.1, 0.1]),
            use_kl_bound=True, use_deterministic_sampling=True,
        )

        # Tiny radius (heavily capped)
        c_small = TRMPPIController(model, TRMPPIParams(trust_region_radius=1e-4, **common))
        u_small, _ = c_small.compute_control(state, ref)

        # Huge radius (effectively no cap) — fresh model to avoid shared U
        model2 = DifferentialDriveKinematic(wheelbase=0.5)
        c_big = TRMPPIController(model2, TRMPPIParams(trust_region_radius=1e6, **common))
        u_big, _ = c_big.compute_control(state, ref)

        # Tiny radius must scale the step; huge radius must not.
        assert c_small._scaled_count >= 1
        assert c_big._scaled_count == 0
        # The capped first control (Δμ[0]) should be strictly smaller in magnitude.
        assert np.linalg.norm(u_small) < np.linalg.norm(u_big), \
            f"capped update {np.linalg.norm(u_small):.4f} not < " \
            f"uncapped {np.linalg.norm(u_big):.4f}"

    def test_multi_iter_runs(self):
        """n_iters > 1 executes multiple inner iterations."""
        ctrl = _make_tr_controller(n_iters=3, trust_region_radius=0.5)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        # one compute_control with n_iters=3 -> 3 inner iterations
        assert ctrl._total_iters == 3
        # second call adds 3 more
        ctrl.compute_control(state, ref)
        assert ctrl._total_iters == 6


# ================================================================
# 5. Deterministic sampling & reproducibility tests (2)
# ================================================================

class TestDeterministicSampling:
    def test_deterministic_reproducible(self):
        """use_deterministic_sampling=True -> two fresh controllers agree."""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        c1 = _make_tr_controller(use_deterministic_sampling=True)
        c2 = _make_tr_controller(use_deterministic_sampling=True)

        seq1, seq2 = [], []
        s1, s2 = state.copy(), state.copy()
        for step in range(5):
            r = _make_ref(t0=step * 0.05)
            u1, _ = c1.compute_control(s1, r)
            u2, _ = c2.compute_control(s2, r)
            seq1.append(u1.copy())
            seq2.append(u2.copy())
            s1 = s1 + c1.model.forward_dynamics(s1, u1) * 0.05
            s2 = s2 + c2.model.forward_dynamics(s2, u2) * 0.05
        assert np.allclose(np.array(seq1), np.array(seq2)), \
            "Deterministic LCD must be reproducible across controllers"

    def test_stochastic_differs(self):
        """Stochastic mode (RNG seeded None) differs across fresh controllers."""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        c1 = _make_tr_controller(use_deterministic_sampling=False)
        c2 = _make_tr_controller(use_deterministic_sampling=False)
        u1, _ = c1.compute_control(state.copy(), ref)
        u2, _ = c2.compute_control(state.copy(), ref)
        # Extremely unlikely to coincide with independent RNGs
        assert not np.allclose(u1, u2), \
            "Stochastic controllers should differ across seeds"


# ================================================================
# 6. Covariance adaptation tests (3)
# ================================================================

class TestCovarianceAdaptation:
    def test_adapt_respects_entropy_floor(self):
        """adapt_covariance keeps sigma_scale within [floor, max], never below floor."""
        floor = 0.3
        cov_max = 4.0
        ctrl = _make_tr_controller(
            adapt_covariance=True,
            cov_step_size=0.5,
            entropy_floor_scale=floor,
            cov_max_scale=cov_max,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        for step in range(30):
            ctrl.compute_control(state, _make_ref(t0=step * 0.05))
            assert np.all(ctrl._sigma_scale >= floor - 1e-9), \
                f"sigma_scale {ctrl._sigma_scale} below floor {floor}"
            assert np.all(ctrl._sigma_scale <= cov_max + 1e-9), \
                f"sigma_scale {ctrl._sigma_scale} above max {cov_max}"

    def test_no_adapt_keeps_unit_scale(self):
        """adapt_covariance=False -> sigma_scale stays ones."""
        ctrl = _make_tr_controller(adapt_covariance=False)
        state = np.array([3.0, 0.0, np.pi / 2])
        for step in range(5):
            ctrl.compute_control(state, _make_ref(t0=step * 0.05))
        assert np.allclose(ctrl._sigma_scale, 1.0)

    def test_reset_restores_state(self):
        """reset restores sigma_scale to ones and clears counters."""
        ctrl = _make_tr_controller(
            adapt_covariance=True, trust_region_radius=1e-4,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        for step in range(8):
            ctrl.compute_control(state, _make_ref(t0=step * 0.05))
        # state accumulated
        assert ctrl._total_iters > 0
        assert len(ctrl._kl_history) == 8

        ctrl.reset()
        assert np.allclose(ctrl.U, 0.0)
        assert np.allclose(ctrl._sigma_scale, 1.0)
        assert ctrl._scaled_count == 0
        assert ctrl._total_iters == 0
        assert ctrl._kl_history == []


# ================================================================
# 7. Integration tests (4)
# ================================================================

class TestIntegration:
    def test_receding_horizon_shift(self):
        """After compute_control, U is shifted with last row zeroed."""
        ctrl = _make_tr_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        ctrl.compute_control(state, ref)
        assert np.allclose(ctrl.U[-1], 0.0), "Last control row should be zero after shift"

    def test_control_bounds_clipping(self):
        """Control respects u_min/u_max bounds."""
        u_min = np.array([-1.0, -1.0])
        u_max = np.array([1.0, 1.0])
        ctrl = _make_tr_controller(
            u_min=u_min, u_max=u_max,
            sigma=np.array([2.0, 2.0]),
            trust_region_radius=100.0,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        for step in range(10):
            control, _ = ctrl.compute_control(state, _make_ref(t0=step * 0.05))
            assert np.all(control >= u_min - 1e-9)
            assert np.all(control <= u_max + 1e-9)

    def test_numerical_stability(self):
        """No NaN/Inf over many steps."""
        ctrl = _make_tr_controller(
            n_iters=2, adapt_covariance=True, trust_region_radius=0.2,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        for step in range(20):
            control, info = ctrl.compute_control(state, _make_ref(t0=step * 0.05))
            assert not np.any(np.isnan(control))
            assert not np.any(np.isinf(control))
            assert not np.isnan(info["tr_stats"]["kl_divergence"])
            state = state + ctrl.model.forward_dynamics(state, control) * 0.05

    def test_circle_tracking_rmse(self):
        """Circle tracking RMSE < 0.3 (50 steps)."""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = TRMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            trust_region_radius=1.0,
        )
        ctrl = TRMPPIController(model, params)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt = params.dt
        N = params.N

        errors = []
        for step in range(50):
            t = step * dt
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
            )
            control, _ = ctrl.compute_control(state, ref)
            state = state + model.forward_dynamics(state, control) * dt
            ref_pt = circle_trajectory(t, radius=3.0)
            err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
            errors.append(err)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.3, f"RMSE {rmse:.4f} >= 0.3"
