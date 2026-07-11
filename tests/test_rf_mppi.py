"""
RF-MPPI (Reference-Free Spline MPPI) 유닛 테스트 — 42번째 변형

~26개 테스트:
  - Params (4): 기본값, 커스텀, 검증(n_knots<2, n_knots>N+1, knot_sigma_vel<=0)
  - Controller (7): 생성/repr, shape, info keys, sample_trajectories,
                    weights, ess, rf_stats, reset, bounds
  - HermiteSplineSampler (6): 기저 attrs, partition-of-unity, reconstruct shape,
                              reconstruct_batch, eval_shifted_knots
  - Smoothness (3): Hermite 샘플이 Gaussian 샘플보다 매끄러움(MSSD), n_knots DOF
  - Dual-space (3): velocity 샘플 on/off, clamp_endpoints_vel, warm_shift
  - Performance (3): 적은 K ESS, 수치 안정성, 원형 추적 RMSE
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import RFMPPIParams
from mppi_controller.controllers.mppi.rf_mppi import (
    RFMPPIController,
    HermiteSplineSampler,
)
from mppi_controller.controllers.mppi.sampling import GaussianSampler
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)


# -- Helper functions --

def _make_rf_controller(**kwargs):
    """RF-MPPI controller 생성 헬퍼."""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=48, N=20, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        n_knots=6,
        sample_velocity_knots=True,
        knot_sigma_vel=0.3,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = RFMPPIParams(**defaults)
    return RFMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_ref(N=20, dt=0.05):
    """원형 기준 궤적."""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ================================================================
# 1. Params tests (4)
# ================================================================

class TestRFMPPIParams:
    def test_params_defaults(self):
        """기본값 검증."""
        params = RFMPPIParams()
        assert params.n_knots == 6
        assert params.sample_velocity_knots is True
        assert params.knot_sigma_pos is None
        assert params.knot_sigma_vel == 0.3
        assert params.clamp_endpoints_vel is False
        assert params.spline_warm_shift is True

    def test_params_custom(self):
        """커스텀값 검증."""
        params = RFMPPIParams(
            n_knots=8,
            sample_velocity_knots=False,
            knot_sigma_pos=np.array([0.4, 0.4]),
            knot_sigma_vel=0.5,
            clamp_endpoints_vel=True,
            spline_warm_shift=False,
        )
        assert params.n_knots == 8
        assert params.sample_velocity_knots is False
        assert np.allclose(params.knot_sigma_pos, [0.4, 0.4])
        assert params.knot_sigma_vel == 0.5
        assert params.clamp_endpoints_vel is True
        assert params.spline_warm_shift is False

    def test_params_validation_n_knots_too_small(self):
        """n_knots < 2 -> AssertionError."""
        with pytest.raises(AssertionError):
            RFMPPIParams(n_knots=1)

    def test_params_validation_n_knots_and_sigma_vel(self):
        """n_knots > N+1, knot_sigma_vel <= 0 -> AssertionError."""
        # n_knots > N + 1
        with pytest.raises(AssertionError):
            RFMPPIParams(N=10, n_knots=12)

        # knot_sigma_vel <= 0
        with pytest.raises(AssertionError):
            RFMPPIParams(knot_sigma_vel=0.0)

        with pytest.raises(AssertionError):
            RFMPPIParams(knot_sigma_vel=-0.1)


# ================================================================
# 2. Controller tests (7)
# ================================================================

class TestRFMPPIController:
    def test_construction_and_repr(self):
        """생성 + repr 문자열."""
        ctrl = _make_rf_controller(n_knots=5, K=32)
        assert ctrl.M == 5
        assert ctrl.nu == 2
        assert isinstance(ctrl.spline, HermiteSplineSampler)
        assert ctrl._P.shape == (5, 2)
        assert ctrl._V.shape == (5, 2)
        r = repr(ctrl)
        assert "RFMPPIController" in r
        assert "n_knots=5" in r
        assert "K=32" in r

    def test_compute_control_shape_and_info_keys(self):
        """제어 shape (2,) + 표준 info 키."""
        ctrl = _make_rf_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))
        for key in [
            "sample_trajectories", "sample_weights", "best_trajectory",
            "best_cost", "mean_cost", "temperature", "ess", "num_samples",
            "rf_stats",
        ]:
            assert key in info, f"missing info key: {key}"

    def test_sample_trajectories_shape(self):
        """sample_trajectories shape (K, N+1, 3)."""
        ctrl = _make_rf_controller(K=48, N=20)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        assert info["sample_trajectories"].shape == (48, 21, 3)

    def test_weights_sum_to_one(self):
        """가중치 합 = 1, ess 유효."""
        ctrl = _make_rf_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        weights = info["sample_weights"]
        assert weights.shape == (ctrl.params.K,)
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-8)
        assert np.all(weights >= 0)
        assert info["ess"] > 0

    def test_rf_stats_subkeys(self):
        """rf_stats 하위 키."""
        ctrl = _make_rf_controller(n_knots=6, sample_velocity_knots=True)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["rf_stats"]
        assert stats["n_knots"] == 6
        assert stats["dual_space"] is True
        assert np.isfinite(stats["knot_pos_norm"])
        assert np.isfinite(stats["knot_vel_norm"])

    def test_reset_zeros_knots(self):
        """reset() -> _P, _V 0으로, U shape 유지."""
        ctrl = _make_rf_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(5):
            ctrl.compute_control(state, ref)

        # 갱신 후 knot은 비영
        assert not np.allclose(ctrl._P, 0.0) or not np.allclose(ctrl._V, 0.0)

        ctrl.reset()
        assert np.allclose(ctrl._P, 0.0)
        assert np.allclose(ctrl._V, 0.0)
        assert np.allclose(ctrl.U, 0.0)

    def test_control_bounds_clipping(self):
        """제어 제약 클리핑 (출력 + warm-shift 미적용 U 시퀀스)."""
        # warm_shift=False면 self.U는 최종 클리핑된 명목 시퀀스를 유지
        ctrl = _make_rf_controller(
            u_min=np.array([-0.5, -0.5]),
            u_max=np.array([0.5, 0.5]),
            knot_sigma_pos=np.array([5.0, 5.0]),
            knot_sigma_vel=5.0,
            spline_warm_shift=False,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, _ = ctrl.compute_control(state, ref)
        # 출력 제어는 항상 제약 내
        assert np.all(control >= -0.5 - 1e-9)
        assert np.all(control <= 0.5 + 1e-9)
        # warm-shift 미적용 -> U 시퀀스 전체도 제약 내
        assert np.all(ctrl.U >= -0.5 - 1e-9)
        assert np.all(ctrl.U <= 0.5 + 1e-9)


# ================================================================
# 3. HermiteSplineSampler tests (6)
# ================================================================

class TestHermiteSplineSampler:
    def test_basis_attrs_and_shapes(self):
        """기저 attr 존재 + shape."""
        N, M = 20, 6
        spline = HermiteSplineSampler(N, M)
        assert spline.knot_times.shape == (M,)
        assert spline.B_p.shape == (N, M)
        assert spline.B_v.shape == (N, M)
        assert spline.dB_p.shape == (N, M)
        assert spline.dB_v.shape == (N, M)
        # knot_times 단조 증가, 끝점 [0, N-1]
        assert spline.knot_times[0] == 0.0
        assert spline.knot_times[-1] == N - 1

    def test_partition_of_unity_constant_spline(self):
        """상수 스플라인(P=ones, V=zeros) -> 모든 제어 = 1 (partition of unity)."""
        N, M, nu = 20, 6, 2
        spline = HermiteSplineSampler(N, M)
        P = np.ones((M, nu))
        V = np.zeros((M, nu))
        U = spline.reconstruct(P, V)
        assert U.shape == (N, nu)
        np.testing.assert_allclose(U, np.ones((N, nu)), atol=1e-10)
        # B_p 행 합 = 1
        np.testing.assert_allclose(spline.B_p.sum(axis=1), np.ones(N), atol=1e-10)

    def test_reconstruct_shape(self):
        """reconstruct (M,nu) -> (N,nu)."""
        N, M, nu = 15, 4, 2
        spline = HermiteSplineSampler(N, M)
        P = np.random.randn(M, nu)
        V = np.random.randn(M, nu)
        U = spline.reconstruct(P, V)
        assert U.shape == (N, nu)
        assert np.all(np.isfinite(U))

    def test_reconstruct_batch_matches_loop(self):
        """reconstruct_batch == 루핑 reconstruct."""
        N, M, nu, K = 18, 5, 2, 7
        spline = HermiteSplineSampler(N, M)
        P = np.random.randn(K, M, nu)
        V = np.random.randn(K, M, nu)
        batch = spline.reconstruct_batch(P, V)
        assert batch.shape == (K, N, nu)
        for k in range(K):
            single = spline.reconstruct(P[k], V[k])
            np.testing.assert_allclose(batch[k], single, atol=1e-10)

    def test_eval_shifted_knots_shapes(self):
        """eval_shifted_knots -> (M,nu) shape 두개."""
        N, M, nu = 20, 6, 2
        spline = HermiteSplineSampler(N, M)
        P = np.random.randn(M, nu)
        V = np.random.randn(M, nu)
        P_new, V_new = spline.eval_shifted_knots(P, V, shift=1.0)
        assert P_new.shape == (M, nu)
        assert V_new.shape == (M, nu)
        assert np.all(np.isfinite(P_new))
        assert np.all(np.isfinite(V_new))

    def test_eval_shifted_preserves_constant(self):
        """상수 스플라인은 시프트해도 위치가 보존(매끄러운 warm-start)."""
        N, M, nu = 20, 6, 2
        spline = HermiteSplineSampler(N, M)
        P = np.full((M, nu), 0.7)
        V = np.zeros((M, nu))
        P_new, V_new = spline.eval_shifted_knots(P, V, shift=1.0)
        # 상수 스플라인 -> 위치 그대로, 속도(도함수) ~ 0
        np.testing.assert_allclose(P_new, np.full((M, nu), 0.7), atol=1e-8)
        np.testing.assert_allclose(V_new, np.zeros((M, nu)), atol=1e-8)


# ================================================================
# 4. Smoothness tests (3)
# ================================================================

def _mssd(controls):
    """평균 제곱 연속 차분 (Mean Squared Successive Difference)."""
    d = np.diff(controls, axis=-2)  # 시간축
    return float(np.mean(d ** 2))


class TestSmoothness:
    def test_hermite_smoother_than_gaussian(self):
        """동일 sigma에서 Hermite 스플라인 샘플이 Gaussian 샘플보다 MSSD 훨씬 낮음."""
        N, M, nu, K = 30, 6, 2, 32
        sigma = np.array([0.5, 0.5])
        rng = np.random.default_rng(0)

        spline = HermiteSplineSampler(N, M)
        # knot 공간 섭동 -> 매끄러운 제어 재구성
        P = rng.normal(0.0, sigma, (K, M, nu))
        V = rng.normal(0.0, 0.3, (K, M, nu))
        hermite_ctrls = spline.reconstruct_batch(P, V)

        # 동일 sigma의 plain Gaussian 샘플 (U=0)
        gsampler = GaussianSampler(sigma=sigma, seed=0)
        gauss_ctrls = gsampler.sample(np.zeros((N, nu)), K)

        mssd_hermite = _mssd(hermite_ctrls)
        mssd_gauss = _mssd(gauss_ctrls)

        assert mssd_hermite < mssd_gauss, (
            f"Hermite MSSD ({mssd_hermite:.4f}) should be < "
            f"Gaussian MSSD ({mssd_gauss:.4f})"
        )
        # "훨씬" 낮음: 최소 5배
        assert mssd_hermite < mssd_gauss / 5.0, (
            f"Hermite should be much smoother: {mssd_hermite:.4f} vs {mssd_gauss:.4f}"
        )

    def test_more_knots_more_dof_still_smooth(self):
        """knot 많을수록 DOF 증가하나 여전히 Gaussian보다 매끄러움."""
        N, nu, K = 30, 2, 32
        sigma = np.array([0.5, 0.5])
        rng = np.random.default_rng(1)

        def hermite_mssd(M):
            spline = HermiteSplineSampler(N, M)
            P = rng.normal(0.0, sigma, (K, M, nu))
            V = rng.normal(0.0, 0.3, (K, M, nu))
            return _mssd(spline.reconstruct_batch(P, V))

        mssd_few = hermite_mssd(3)
        mssd_many = hermite_mssd(10)

        # 더 많은 knot -> 더 많은 DOF -> 일반적으로 MSSD 증가 (더 가변적)
        assert mssd_many > mssd_few, (
            f"More knots should give more DOF (higher MSSD): "
            f"{mssd_many:.5f} vs {mssd_few:.5f}"
        )

        # 그래도 Gaussian보다는 매끄러움
        gauss_ctrls = GaussianSampler(sigma=sigma, seed=1).sample(
            np.zeros((N, nu)), K
        )
        assert mssd_many < _mssd(gauss_ctrls)

    def test_controller_samples_smooth(self):
        """컨트롤러가 생성한 샘플 제어가 매끄러움(연속 차분 작음)."""
        ctrl = _make_rf_controller(K=48, N=30)
        spline = ctrl.spline
        # knot 명목 + 작은 섭동 직접 재구성
        rng = np.random.default_rng(2)
        P = ctrl._P[None] + rng.normal(0.0, 0.5, (48, ctrl.M, ctrl.nu))
        V = ctrl._V[None] + rng.normal(0.0, 0.3, (48, ctrl.M, ctrl.nu))
        ctrls = spline.reconstruct_batch(P, V)
        # 매끄러움 정량: 제어 표준편차 대비 MSSD가 작음
        assert _mssd(ctrls) < 0.05


# ================================================================
# 5. Dual-space tests (3)
# ================================================================

class TestDualSpace:
    def test_velocity_sampling_off_keeps_vel_zero(self):
        """sample_velocity_knots=False -> _V 섭동 0, knot_vel_norm ~ 0 유지."""
        ctrl = _make_rf_controller(
            sample_velocity_knots=False, spline_warm_shift=False,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(5):
            _, info = ctrl.compute_control(state, ref)

        # dV=0 이므로 _V는 갱신되지 않아 0 유지
        assert np.allclose(ctrl._V, 0.0)
        assert info["rf_stats"]["knot_vel_norm"] == pytest.approx(0.0, abs=1e-12)
        assert info["rf_stats"]["dual_space"] is False

    def test_velocity_sampling_on_nonzero(self):
        """sample_velocity_knots=True -> 속도 knot 비영 갱신."""
        ctrl = _make_rf_controller(
            sample_velocity_knots=True, spline_warm_shift=False,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(8):
            _, info = ctrl.compute_control(state, ref)

        assert info["rf_stats"]["dual_space"] is True
        # 속도 knot이 일반적으로 비영 (dual-space 활성)
        assert info["rf_stats"]["knot_vel_norm"] > 0.0

    def test_clamp_endpoints_vel(self):
        """clamp_endpoints_vel=True -> 갱신 후 _V[0], _V[-1] = 0."""
        ctrl = _make_rf_controller(
            sample_velocity_knots=True,
            clamp_endpoints_vel=True,
            spline_warm_shift=False,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(5):
            ctrl.compute_control(state, ref)

        np.testing.assert_allclose(ctrl._V[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(ctrl._V[-1], 0.0, atol=1e-12)


# ================================================================
# 6. Performance / Integration tests (3+)
# ================================================================

class TestPerformance:
    def test_warm_shift_both_run(self):
        """spline_warm_shift True/False 모두 정상 동작."""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for warm in (True, False):
            ctrl = _make_rf_controller(spline_warm_shift=warm)
            for _ in range(5):
                control, _ = ctrl.compute_control(state, ref)
                assert control.shape == (2,)
                assert np.all(np.isfinite(control))
            # RF는 vanilla처럼 U[-1]을 0으로 만들지 않음 (스플라인 재구성)
            assert ctrl.U.shape == (ctrl.params.N, ctrl.nu)

    def test_small_K_reasonable_ess(self):
        """적은 K(=32)에서도 ESS가 reasonably 높음 (>1)."""
        ctrl = _make_rf_controller(K=32)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        ess_vals = []
        for _ in range(5):
            _, info = ctrl.compute_control(state, ref)
            ess_vals.append(info["ess"])

        mean_ess = np.mean(ess_vals)
        assert mean_ess > 1.0, f"Mean ESS {mean_ess:.2f} too low for K=32"

    def test_numerical_stability(self):
        """장기 실행 NaN/Inf 없음."""
        ctrl = _make_rf_controller(K=48, N=20)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        model = ctrl.model

        for _ in range(30):
            control, info = ctrl.compute_control(state, ref)
            assert not np.any(np.isnan(control))
            assert not np.any(np.isinf(control))
            assert np.isfinite(info["rf_stats"]["knot_pos_norm"])
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

    def test_optimal_control_finite_and_U_shape(self):
        """receding horizon: optimal control 유한, self.U shape (N,nu)."""
        ctrl = _make_rf_controller(N=20)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, _ = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(control))
        assert ctrl.U.shape == (20, ctrl.nu)
        assert np.all(np.isfinite(ctrl.U))

    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.35."""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = RFMPPIParams(
            K=64, N=20, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            n_knots=6,
            sample_velocity_knots=True,
            knot_sigma_vel=0.3,
        )
        ctrl = RFMPPIController(model, params)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt = params.dt
        N = params.N
        num_steps = 60

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
        assert rmse < 0.35, f"RMSE {rmse:.4f} >= 0.35"
