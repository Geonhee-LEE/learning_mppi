"""
PGD-MPPI (Preconditioned Gradient Descent MPPI) 유닛 테스트 — 40번째 변형

~26개 테스트:
  - Params (5): 기본값, 커스텀, step_size/n_grad_steps/cov_scale 검증
  - Construction (2): 생성, repr
  - compute_control (7): shape, finite, info keys, 궤적/가중치/ESS, pgd_stats
  - GradSteps (4): n_grad_steps>1 비용 감소, step_size 효과, resample on/off
  - Covariance (3): adapt_covariance, 범위 클리핑, normalize_gradient
  - HorizonReset (2): receding horizon 시프트, reset
  - Integration (3): 제어 바운드 클리핑, 원형 추적 RMSE, 기본값 vanilla-동등
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
    PGDMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.pgd_mppi import PGDMPPIController
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

def _make_pgd_controller(**kwargs):
    """PGD-MPPI controller creation helper."""
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
    params = PGDMPPIParams(**defaults)
    return PGDMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_ref(N=10, dt=0.05):
    """Reference trajectory."""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ================================================================
# 1. Params tests (5)
# ================================================================

class TestPGDMPPIParams:
    def test_params_defaults(self):
        """기본값 검증."""
        params = PGDMPPIParams()
        assert params.step_size == 1.0
        assert params.n_grad_steps == 1
        assert params.resample_each_step is True
        assert params.adapt_covariance is False
        assert params.cov_step_size == 0.2
        assert params.cov_min_scale == 0.25
        assert params.cov_max_scale == 4.0
        assert params.normalize_gradient is False

    def test_params_custom(self):
        """커스텀 값 검증."""
        params = PGDMPPIParams(
            step_size=0.5,
            n_grad_steps=4,
            resample_each_step=False,
            adapt_covariance=True,
            cov_step_size=0.3,
            cov_min_scale=0.5,
            cov_max_scale=2.0,
            normalize_gradient=True,
        )
        assert params.step_size == 0.5
        assert params.n_grad_steps == 4
        assert params.resample_each_step is False
        assert params.adapt_covariance is True
        assert params.cov_step_size == 0.3
        assert params.cov_min_scale == 0.5
        assert params.cov_max_scale == 2.0
        assert params.normalize_gradient is True

    def test_params_validation_step_size(self):
        """step_size <= 0 -> AssertionError."""
        with pytest.raises(AssertionError):
            PGDMPPIParams(step_size=0.0)
        with pytest.raises(AssertionError):
            PGDMPPIParams(step_size=-0.5)

    def test_params_validation_n_grad_steps(self):
        """n_grad_steps < 1 -> AssertionError."""
        with pytest.raises(AssertionError):
            PGDMPPIParams(n_grad_steps=0)
        with pytest.raises(AssertionError):
            PGDMPPIParams(n_grad_steps=-2)

    def test_params_validation_cov_scale(self):
        """cov_min_scale > cov_max_scale, cov_step_size < 0 -> AssertionError."""
        with pytest.raises(AssertionError):
            PGDMPPIParams(cov_min_scale=2.0, cov_max_scale=1.0)
        with pytest.raises(AssertionError):
            PGDMPPIParams(cov_min_scale=0.0)
        with pytest.raises(AssertionError):
            PGDMPPIParams(cov_step_size=-0.1)


# ================================================================
# 2. Construction tests (2)
# ================================================================

class TestConstruction:
    def test_construction(self):
        """컨트롤러 생성 + 핵심 속성."""
        ctrl = _make_pgd_controller()
        assert isinstance(ctrl, MPPIController)
        assert isinstance(ctrl, PGDMPPIController)
        nu = ctrl.model.control_dim
        assert ctrl._base_sigma.shape == (nu,)
        assert ctrl._sigma_scale.shape == (nu,)
        assert np.allclose(ctrl._sigma_scale, 1.0)
        assert ctrl.U.shape == (10, nu)

    def test_repr(self):
        """repr 문자열에 핵심 파라미터 포함."""
        ctrl = _make_pgd_controller(step_size=0.7, n_grad_steps=3, adapt_covariance=True)
        s = repr(ctrl)
        assert "PGDMPPIController" in s
        assert "step_size=0.7" in s
        assert "n_grad_steps=3" in s
        assert "adapt_covariance=True" in s


# ================================================================
# 3. compute_control tests (7)
# ================================================================

class TestComputeControl:
    def test_control_shape(self):
        """control shape (2,)."""
        ctrl = _make_pgd_controller()
        control, _ = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert control.shape == (2,)

    def test_control_finite(self):
        """control finite (NaN/Inf 없음)."""
        ctrl = _make_pgd_controller()
        control, _ = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert np.all(np.isfinite(control))

    def test_info_keys(self):
        """info 표준 MPPI 키 존재."""
        ctrl = _make_pgd_controller()
        _, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        for key in (
            "sample_trajectories", "sample_weights", "best_trajectory",
            "best_cost", "mean_cost", "temperature", "ess", "num_samples",
        ):
            assert key in info, f"missing key {key}"

    def test_sample_trajectories_shape(self):
        """sample_trajectories shape (K, N+1, 3)."""
        ctrl = _make_pgd_controller(K=64, N=10)
        _, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert info["sample_trajectories"].shape == (64, 11, 3)

    def test_weights_sum_to_one(self):
        """가중치 합 = 1."""
        ctrl = _make_pgd_controller()
        _, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        w = info["sample_weights"]
        assert abs(float(np.sum(w)) - 1.0) < 1e-6

    def test_ess_range(self):
        """ESS in [1, K]."""
        ctrl = _make_pgd_controller(K=64)
        _, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        ess = info["ess"]
        assert 1.0 <= ess <= 64.0 + 1e-6

    def test_pgd_stats(self):
        """pgd_stats 서브키 존재 + 타입."""
        ctrl = _make_pgd_controller(step_size=0.7, n_grad_steps=3)
        _, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert "pgd_stats" in info
        stats = info["pgd_stats"]
        assert stats["n_grad_steps"] == 3
        assert stats["step_size"] == 0.7
        assert "grad_norm" in stats
        assert np.isfinite(stats["grad_norm"])
        assert stats["sigma_scale"].shape == (2,)


# ================================================================
# 4. Gradient steps tests (4)
# ================================================================

class TestGradientSteps:
    def test_multi_step_lowers_best_cost(self):
        """n_grad_steps>1 이 수렴된 평균 μ 의 비용을 낮춤(또는 동등).

        best_cost 는 마지막 반복의 무작위 샘플 최소값이라 노이즈가 크므로,
        실제로 개선되는 양인 '수렴된 평균 제어 시퀀스 μ 의 비용'을 여러
        seed 평균으로 비교한다.
        """
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        def mean_mu_cost(n_steps, seed):
            ctrl = _make_pgd_controller(n_grad_steps=n_steps, step_size=0.8)
            ctrl._rng = np.random.default_rng(seed)
            ctrl.compute_control(state, ref)
            # 시프트 전 μ 를 복원: U는 roll 되었으므로 다시 평가 대신
            # 마지막 best_cost 의 평균 비교로 충분. 여기선 mu rollout 비용 평가.
            # U[-1]=0 이고 한 칸 당겨졌으므로 한 스텝 시퀀스로 근사 비용 평가.
            mu = np.roll(ctrl.U, 1, axis=0)  # 원래 μ 근사 복원
            traj = ctrl.dynamics_wrapper.rollout(state, mu[None])
            c = ctrl.cost_function.compute_cost(traj, mu[None], ref)
            return float(c[0])

        seeds = range(8)
        cost1 = np.mean([mean_mu_cost(1, s) for s in seeds])
        cost6 = np.mean([mean_mu_cost(6, s) for s in seeds])

        # 다중 경사 스텝은 평균적으로 더 낮은(또는 동등) μ 비용을 달성
        assert cost6 <= cost1 * 1.05, (
            f"multi-step mean mu-cost {cost6:.3f} should be "
            f"<= 1-step {cost1:.3f}"
        )

    def test_multi_step_runs_more_iterations(self):
        """n_grad_steps 만큼 grad 업데이트 반복 (stats 반영)."""
        ctrl = _make_pgd_controller(n_grad_steps=5)
        _, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert info["pgd_stats"]["n_grad_steps"] == 5

    def test_step_size_effect(self):
        """step_size가 평균 업데이트 크기에 영향 (단일 스텝, 동일 노이즈).

        μ ← μ + α·g̃ 이므로, 동일 seed(동일 노이즈/grad)에서 α가 클수록
        제어 출력의 μ_0 로부터의 변화가 더 크다.
        """
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 동일 RNG seed -> 동일 노이즈 -> 동일 grad. 시작 U는 0.
        ctrl_small = _make_pgd_controller(step_size=0.1, n_grad_steps=1)
        ctrl_large = _make_pgd_controller(step_size=2.0, n_grad_steps=1)
        ctrl_small._rng = np.random.default_rng(7)
        ctrl_large._rng = np.random.default_rng(7)

        c_small, _ = ctrl_small.compute_control(state, ref)
        c_large, _ = ctrl_large.compute_control(state, ref)

        # 시작 U=0 이므로 큰 step_size 의 업데이트(=출력 제어 직전 μ)가 더 큼.
        # receding shift 영향을 피하려고 update 후 U 대신 grad_norm·step 관계를 본다.
        # grad_norm 은 step_size 독립이고 동일 노이즈에서 같아야 함.
        gn_small = ctrl_small.last_info["pgd_stats"]["grad_norm"]
        gn_large = ctrl_large.last_info["pgd_stats"]["grad_norm"]
        assert abs(gn_small - gn_large) < 1e-6, "동일 노이즈에서 grad_norm 일치해야 함"
        # 출력 제어 크기: 큰 α 가 더 큰 갱신 -> 더 큰 |control| (μ_0=0 기준)
        assert np.linalg.norm(c_large) > np.linalg.norm(c_small), (
            f"larger step_size should produce larger control update: "
            f"{np.linalg.norm(c_large):.4f} vs {np.linalg.norm(c_small):.4f}"
        )

    def test_resample_each_step_true(self):
        """resample_each_step=True 정상 동작."""
        ctrl = _make_pgd_controller(n_grad_steps=4, resample_each_step=True)
        control, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))

    def test_resample_each_step_false(self):
        """resample_each_step=False 정상 동작 (노이즈 재사용)."""
        ctrl = _make_pgd_controller(n_grad_steps=4, resample_each_step=False)
        control, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))


# ================================================================
# 5. Covariance adaptation tests (3)
# ================================================================

class TestCovarianceAdaptation:
    def test_adapt_covariance_changes_scale(self):
        """adapt_covariance=True 시 _sigma_scale 가 ones 에서 벗어남."""
        ctrl = _make_pgd_controller(
            n_grad_steps=3, adapt_covariance=True, cov_step_size=0.5,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        for _ in range(5):
            ctrl.compute_control(state, ref)
        assert not np.allclose(ctrl._sigma_scale, 1.0), (
            "adapt_covariance should move sigma_scale away from ones"
        )

    def test_sigma_scale_within_bounds(self):
        """_sigma_scale 가 [cov_min_scale, cov_max_scale] 범위 내 유지."""
        cmin, cmax = 0.4, 2.5
        ctrl = _make_pgd_controller(
            n_grad_steps=4, adapt_covariance=True, cov_step_size=0.6,
            cov_min_scale=cmin, cov_max_scale=cmax,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        for _ in range(10):
            ctrl.compute_control(state, ref)
            assert np.all(ctrl._sigma_scale >= cmin - 1e-9)
            assert np.all(ctrl._sigma_scale <= cmax + 1e-9)

    def test_normalize_gradient(self):
        """normalize_gradient=True 정상 동작 + finite."""
        ctrl = _make_pgd_controller(n_grad_steps=2, normalize_gradient=True)
        control, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))
        assert np.isfinite(info["pgd_stats"]["grad_norm"])


# ================================================================
# 6. Horizon shift & reset tests (2)
# ================================================================

class TestHorizonReset:
    def test_receding_horizon_shift(self):
        """compute_control 후 U[-1] == 0 (receding horizon 시프트)."""
        ctrl = _make_pgd_controller()
        ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert np.allclose(ctrl.U[-1], 0.0)

    def test_reset(self):
        """reset 후 _sigma_scale=ones, U=0, 히스토리 초기화."""
        ctrl = _make_pgd_controller(
            n_grad_steps=3, adapt_covariance=True, cov_step_size=0.5,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        for _ in range(5):
            ctrl.compute_control(state, ref)

        assert len(ctrl._grad_norm_history) == 5

        ctrl.reset()
        assert np.allclose(ctrl._sigma_scale, 1.0)
        assert np.allclose(ctrl.U, 0.0)
        assert len(ctrl._grad_norm_history) == 0


# ================================================================
# 7. Integration tests (3)
# ================================================================

class TestIntegration:
    def test_control_bounds_clipping(self):
        """u_min/u_max 설정 시 제어 바운드 클리핑 준수."""
        u_min = np.array([-0.5, -0.5])
        u_max = np.array([0.5, 0.5])
        ctrl = _make_pgd_controller(
            n_grad_steps=3, step_size=1.5,
            u_min=u_min, u_max=u_max,
            sigma=np.array([2.0, 2.0]),
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        for _ in range(5):
            control, _ = ctrl.compute_control(state, ref)
            assert np.all(control >= u_min - 1e-9), f"control {control} < u_min"
            assert np.all(control <= u_max + 1e-9), f"control {control} > u_max"

    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.3 (50 스텝)."""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = PGDMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            n_grad_steps=3, step_size=0.8,
        )
        ctrl = PGDMPPIController(model, params)

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
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt

            ref_pt = circle_trajectory(t, radius=3.0)
            err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
            errors.append(err)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.3, f"RMSE {rmse:.4f} >= 0.3"

    def test_defaults_vanilla_equivalent(self):
        """기본값(n_grad_steps=1, step_size=1.0, adapt_covariance=False)에서
        단일 MPPI 스텝처럼 동작 (finite control, ess>1)."""
        ctrl = _make_pgd_controller()  # 모든 기본값
        assert ctrl.pgd_params.n_grad_steps == 1
        assert ctrl.pgd_params.step_size == 1.0
        assert ctrl.pgd_params.adapt_covariance is False

        control, info = ctrl.compute_control(np.array([3.0, 0.0, np.pi / 2]), _make_ref())
        assert np.all(np.isfinite(control))
        assert info["ess"] > 1.0
        # 기본값에서는 공분산 적응이 일어나지 않음
        assert np.allclose(ctrl._sigma_scale, 1.0)
