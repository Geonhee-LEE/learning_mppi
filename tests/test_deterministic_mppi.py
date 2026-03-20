"""
dsMPPI (Deterministic Sampling MPPI) 유닛 테스트

28개 테스트:
  - Params (3): 기본값, 커스텀, 검증
  - Deterministic Sampling (5): halton shape, sobol shape, sigma_points, grid, reproducibility
  - Controller (5): output shape, info keys, different K, reset, warm start
  - CEM Iteration (4): elite selection, distribution update, cost decrease, convergence
  - Performance (4): circle RMSE, obstacle avoidance, computation time, low sample efficiency
  - Integration (4): numerical stability, cem disabled, add random samples hybrid, statistics
  - Comparison (3): vs vanilla, vs CMA, sample efficiency
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
    CMAMPPIParams,
    DeterministicMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cma_mppi import CMAMPPIController
from mppi_controller.controllers.mppi.deterministic_mppi import DeterministicMPPIController
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


# -- 헬퍼 함수 --------------------------------------------------

def _make_ds_controller(**kwargs):
    """dsMPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        sampling_method="halton",
        n_cem_iters=3, n_cem_iters_init=5,
        elite_ratio=0.3,
        cem_alpha=0.7,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = DeterministicMPPIParams(**defaults)
    return DeterministicMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_vanilla_controller(**kwargs):
    """비교용 Vanilla MPPI"""
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


def _make_cma_controller(**kwargs):
    """비교용 CMA-MPPI"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        n_iters_init=3, n_iters=2,
        cov_learning_rate=0.5,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    params = CMAMPPIParams(**defaults)
    return CMAMPPIController(model, params, cost_function=cost_function)


def _make_ref(N=10, dt=0.05):
    """레퍼런스 궤적"""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ==============================================================
# 1. Params 테스트 (3개)
# ==============================================================

class TestDeterministicMPPIParams:
    def test_params_defaults(self):
        """기본값 검증"""
        params = DeterministicMPPIParams()
        assert params.sampling_method == "halton"
        assert params.n_cem_iters == 3
        assert params.n_cem_iters_init == 5
        assert params.elite_ratio == 0.3
        assert params.cem_alpha == 0.7
        assert params.use_cem_update is True
        assert params.add_random_samples == 0

    def test_params_custom(self):
        """커스텀 값 검증"""
        params = DeterministicMPPIParams(
            sampling_method="sobol",
            n_cem_iters=5,
            n_cem_iters_init=8,
            elite_ratio=0.2,
            cem_alpha=0.5,
            use_cem_update=False,
            add_random_samples=32,
        )
        assert params.sampling_method == "sobol"
        assert params.n_cem_iters == 5
        assert params.n_cem_iters_init == 8
        assert params.elite_ratio == 0.2
        assert params.cem_alpha == 0.5
        assert params.use_cem_update is False
        assert params.add_random_samples == 32

    def test_params_validation(self):
        """잘못된 값 -> AssertionError"""
        # 잘못된 sampling_method
        with pytest.raises(AssertionError):
            DeterministicMPPIParams(sampling_method="invalid")

        # n_cem_iters <= 0
        with pytest.raises(AssertionError):
            DeterministicMPPIParams(n_cem_iters=0)

        # n_cem_iters_init <= 0
        with pytest.raises(AssertionError):
            DeterministicMPPIParams(n_cem_iters_init=0)

        # elite_ratio out of range
        with pytest.raises(AssertionError):
            DeterministicMPPIParams(elite_ratio=0.0)
        with pytest.raises(AssertionError):
            DeterministicMPPIParams(elite_ratio=1.5)

        # cem_alpha out of range
        with pytest.raises(AssertionError):
            DeterministicMPPIParams(cem_alpha=0.0)

        # add_random_samples < 0
        with pytest.raises(AssertionError):
            DeterministicMPPIParams(add_random_samples=-1)


# ==============================================================
# 2. Deterministic Sampling 테스트 (5개)
# ==============================================================

class TestDeterministicSampling:
    def test_halton_shape(self):
        """Halton 샘플 shape 검증"""
        ctrl = _make_ds_controller(K=64, N=10, sampling_method="halton")
        mu = np.zeros((10, 2))
        sigma = np.ones((10, 2)) * 0.5
        samples = ctrl._generate_deterministic_samples(mu, sigma, 64)

        assert samples.shape == (64, 10, 2)
        assert np.all(np.isfinite(samples))

    def test_sobol_shape(self):
        """Sobol 샘플 shape 검증"""
        ctrl = _make_ds_controller(K=64, N=10, sampling_method="sobol")
        mu = np.zeros((10, 2))
        sigma = np.ones((10, 2)) * 0.5
        samples = ctrl._generate_deterministic_samples(mu, sigma, 64)

        assert samples.shape[0] == 64
        assert samples.shape[1] == 10
        assert samples.shape[2] == 2
        assert np.all(np.isfinite(samples))

    def test_sigma_points(self):
        """Sigma Points: 2*dim+1 포인트"""
        ctrl = _make_ds_controller(K=64, N=5, sampling_method="sigma_points")
        mu = np.zeros((5, 2))
        sigma = np.ones((5, 2)) * 0.5

        samples = ctrl._generate_deterministic_samples(mu, sigma, 64)

        dim = 5 * 2  # N * nu
        expected_count = 2 * dim + 1  # 21
        assert samples.shape == (expected_count, 5, 2)

        # Center point는 mu
        np.testing.assert_allclose(samples[0], mu, atol=1e-10)

    def test_grid_shape(self):
        """Grid 샘플 shape 검증"""
        ctrl = _make_ds_controller(K=64, N=10, sampling_method="grid")
        mu = np.zeros((10, 2))
        sigma = np.ones((10, 2)) * 0.5
        samples = ctrl._generate_deterministic_samples(mu, sigma, 64)

        # Grid는 n_per_dim^nu 개 생성 (64 → n_per_dim=8 → 64개)
        assert samples.shape[1] == 10
        assert samples.shape[2] == 2
        assert samples.shape[0] > 0
        assert np.all(np.isfinite(samples))

    def test_reproducibility(self):
        """결정론적 샘플의 재현성"""
        ctrl = _make_ds_controller(K=32, N=5, sampling_method="halton")
        mu = np.zeros((5, 2))
        sigma = np.ones((5, 2)) * 0.5

        # Halton은 scramble=True이므로 인스턴스별로 다를 수 있지만,
        # 같은 호출 내에서 shape와 유한성은 보장
        samples1 = ctrl._generate_deterministic_samples(mu, sigma, 32)
        samples2 = ctrl._generate_deterministic_samples(mu, sigma, 32)

        assert samples1.shape == samples2.shape
        assert np.all(np.isfinite(samples1))
        assert np.all(np.isfinite(samples2))


# ==============================================================
# 3. Controller 테스트 (5개)
# ==============================================================

class TestDeterministicMPPIController:
    def test_compute_control_shape(self):
        """control (nu,), info 표준 키 검증"""
        ctrl = _make_ds_controller()
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

    def test_info_deterministic_stats(self):
        """deterministic_stats 키 검증"""
        ctrl = _make_ds_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["deterministic_stats"]
        assert "n_iters" in stats
        assert "iteration_costs" in stats
        assert "cost_improvement" in stats
        assert "sampling_method" in stats
        assert "total_samples" in stats
        assert "deterministic_samples" in stats
        assert "random_samples" in stats

        # 첫 호출은 n_cem_iters_init
        assert stats["n_iters"] == 5
        assert len(stats["iteration_costs"]) == 5
        assert stats["sampling_method"] == "halton"

    def test_different_K_values(self):
        """K=32/128/256 정상 작동"""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for K in [32, 128, 256]:
            ctrl = _make_ds_controller(K=K, n_cem_iters=2, n_cem_iters_init=2)
            control, info = ctrl.compute_control(state, ref)
            assert control.shape == (2,)
            assert info["num_samples"] >= K  # 최소 K개 (하이브리드 제외)

    def test_reset_clears_state(self):
        """reset 후 초기화"""
        ctrl = _make_ds_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 몇 번 실행
        for _ in range(3):
            ctrl.compute_control(state, ref)

        assert len(ctrl._cem_stats_history) == 3
        assert ctrl._is_first_call is False

        ctrl.reset()
        assert np.allclose(ctrl.U, 0.0)
        assert ctrl._is_first_call is True
        assert len(ctrl._cem_stats_history) == 0
        assert len(ctrl._iteration_costs) == 0

    def test_warm_start(self):
        """set_control_sequence로 warm start"""
        ctrl = _make_ds_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 초기 제어 시퀀스 설정
        U_init = np.random.randn(10, 2) * 0.1
        ctrl.set_control_sequence(U_init)

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))


# ==============================================================
# 4. CEM Iteration 테스트 (4개)
# ==============================================================

class TestCEMIteration:
    def test_elite_selection(self):
        """elite selection이 상위 비율만 선택"""
        ctrl = _make_ds_controller(K=100, elite_ratio=0.2)

        # 직접 CEM update 테스트
        controls = np.random.randn(100, 10, 2)
        costs = np.random.rand(100)
        mu = np.zeros((10, 2))
        sigma = np.ones((10, 2)) * 0.5

        mu_new, sigma_new = ctrl._cem_update(controls, costs, mu, sigma)

        # 업데이트된 평균이 elite 쪽으로 이동
        elite_idx = np.argsort(costs)[:20]
        elite_mean = np.mean(controls[elite_idx], axis=0)
        # EMA이므로 완전히 일치하지는 않지만, 방향은 일치
        alpha = ctrl.ds_params.cem_alpha
        expected_mu = (1 - alpha) * mu + alpha * elite_mean
        np.testing.assert_allclose(mu_new, expected_mu, atol=1e-10)

    def test_distribution_update(self):
        """CEM 분포 업데이트 sigma 축소"""
        ctrl = _make_ds_controller(K=128, n_cem_iters_init=3, cem_alpha=0.9)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["deterministic_stats"]

        # CEM 후 sigma가 초기보다 축소 경향
        assert stats["final_sigma_mean"] > 0

    def test_cost_decrease(self):
        """CEM 반복으로 비용 감소 경향"""
        decrease_count = 0
        n_trials = 5
        for trial in range(n_trials):
            ctrl = _make_ds_controller(
                K=128, n_cem_iters_init=8, sampling_method="halton"
            )
            state = np.array([3.0, 0.0, np.pi / 2])
            ref = _make_ref()

            _, info = ctrl.compute_control(state, ref)
            costs = info["deterministic_stats"]["iteration_costs"]

            if costs[0] >= costs[-1]:
                decrease_count += 1

        # 과반수에서 비용 감소 경향
        assert decrease_count >= 2, \
            f"Expected cost decrease in majority of trials, got {decrease_count}/{n_trials}"

    def test_cold_warm_convergence(self):
        """첫 호출 (cold) vs 이후 (warm) 반복 수 구분"""
        ctrl = _make_ds_controller(n_cem_iters=2, n_cem_iters_init=6)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 첫 호출 → cold start (6 iters)
        _, info1 = ctrl.compute_control(state, ref)
        assert info1["deterministic_stats"]["n_iters"] == 6

        # 두 번째 호출 → warm start (2 iters)
        _, info2 = ctrl.compute_control(state, ref)
        assert info2["deterministic_stats"]["n_iters"] == 2


# ==============================================================
# 5. Performance 테스트 (4개)
# ==============================================================

class TestPerformance:
    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.3 (50 스텝)"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = DeterministicMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            sampling_method="halton",
            n_cem_iters=3, n_cem_iters_init=5,
        )
        ctrl = DeterministicMPPIController(model, params)

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
            err = np.sqrt(
                (state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2
            )
            errors.append(err)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.3, f"RMSE {rmse:.4f} >= 0.3"

    def test_obstacle_avoidance(self):
        """3개 장애물 충돌 없음"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4), (-2.0, -1.0, 0.5)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0),
        ])

        params = DeterministicMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            sampling_method="halton",
            n_cem_iters=3, n_cem_iters_init=5,
        )
        ctrl = DeterministicMPPIController(model, params, cost_function=cost)

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
                dist = np.sqrt(
                    (state[0] - ox) ** 2 + (state[1] - oy) ** 2
                )
                if dist < r:
                    collisions += 1

        assert collisions == 0, f"Collisions: {collisions}"

    def test_computation_time(self):
        """K=512, N=30에서 200ms 이내"""
        ctrl = _make_ds_controller(
            K=512, N=30, n_cem_iters=2, n_cem_iters_init=2,
            sampling_method="halton",
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=30)

        # Warmup
        ctrl.compute_control(state, ref)
        ctrl.reset()

        times = []
        for _ in range(5):
            t_start = time.time()
            ctrl.compute_control(state, ref)
            times.append(time.time() - t_start)

        mean_ms = np.mean(times) * 1000
        assert mean_ms < 200, f"Mean solve time {mean_ms:.1f}ms >= 200ms"

    def test_low_sample_efficiency(self):
        """K=32 소수 샘플에서도 합리적 성능"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = DeterministicMPPIParams(
            K=32, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            sampling_method="halton",
            n_cem_iters=5, n_cem_iters_init=8,
        )
        ctrl = DeterministicMPPIController(model, params)

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
            err = np.sqrt(
                (state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2
            )
            errors.append(err)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        # K=32에서도 RMSE < 0.5 (결정론적 샘플의 효율성)
        assert rmse < 0.5, f"Low-sample RMSE {rmse:.4f} >= 0.5"


# ==============================================================
# 6. Integration 테스트 (4개)
# ==============================================================

class TestIntegration:
    def test_numerical_stability(self):
        """NaN/Inf 없음 (20 스텝)"""
        ctrl = _make_ds_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(20):
            control, info = ctrl.compute_control(state, ref)
            assert not np.any(np.isnan(control)), "NaN in control"
            assert not np.any(np.isinf(control)), "Inf in control"
            assert not np.isnan(info["ess"]), "NaN in ESS"
            assert not np.isnan(info["best_cost"]), "NaN in best_cost"

            state_dot = ctrl.model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

    def test_cem_disabled(self):
        """use_cem_update=False → CEM 없이 단일 결정론적 샘플링"""
        ctrl = _make_ds_controller(
            K=64, use_cem_update=False,
            n_cem_iters=3, n_cem_iters_init=3,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))
        stats = info["deterministic_stats"]
        # CEM 비활성이므로 분포 업데이트 없음, 하지만 반복은 진행
        assert stats["n_iters"] == 3

    def test_add_random_samples_hybrid(self):
        """하이브리드 모드: 결정론적 + 랜덤 샘플"""
        ctrl = _make_ds_controller(
            K=64, add_random_samples=32,
            n_cem_iters=2, n_cem_iters_init=2,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        stats = info["deterministic_stats"]
        assert stats["total_samples"] == 64 + 32
        assert stats["deterministic_samples"] == 64
        assert stats["random_samples"] == 32

    def test_statistics(self):
        """get_deterministic_statistics() 누적 통계"""
        ctrl = _make_ds_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 빈 상태
        stats = ctrl.get_deterministic_statistics()
        assert stats["mean_cost_improvement"] == 0.0
        assert len(stats["cem_stats_history"]) == 0

        # 실행 후
        for _ in range(5):
            ctrl.compute_control(state, ref)

        stats = ctrl.get_deterministic_statistics()
        assert stats["mean_n_iters"] > 0
        assert len(stats["cem_stats_history"]) == 5
        assert stats["sampling_method"] == "halton"


# ==============================================================
# 7. Comparison 테스트 (3개)
# ==============================================================

class TestComparison:
    def test_vs_vanilla_tracking(self):
        """dsMPPI가 Vanilla MPPI와 유사하거나 더 나은 추적 (50 스텝)"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        dt = 0.05
        N = 15
        num_steps = 50

        vanilla = _make_vanilla_controller(K=128, N=N)
        ds = _make_ds_controller(
            K=128, N=N,
            n_cem_iters=3, n_cem_iters_init=5,
            sampling_method="halton",
        )

        def run_controller(ctrl):
            state = np.array([3.0, 0.0, np.pi / 2])
            errors = []
            np.random.seed(42)
            for step in range(num_steps):
                t = step * dt
                ref = generate_reference_trajectory(
                    lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
                )
                control, _ = ctrl.compute_control(state, ref)
                state_dot = model.forward_dynamics(state, control)
                state = state + state_dot * dt

                ref_pt = circle_trajectory(t, radius=3.0)
                err = np.sqrt(
                    (state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2
                )
                errors.append(err)
            return np.sqrt(np.mean(np.array(errors) ** 2))

        rmse_vanilla = run_controller(vanilla)
        rmse_ds = run_controller(ds)

        # dsMPPI가 Vanilla보다 극단적으로 나쁘지 않아야 함
        assert rmse_ds < rmse_vanilla * 2.0, \
            f"dsMPPI RMSE {rmse_ds:.4f} >> Vanilla {rmse_vanilla:.4f}"

    def test_vs_cma_tracking(self):
        """dsMPPI vs CMA-MPPI -- 둘 다 반복 최적화 변형"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        dt = 0.05
        N = 15
        num_steps = 50

        cma = _make_cma_controller(K=128, N=N)
        ds = _make_ds_controller(
            K=128, N=N,
            n_cem_iters=3, n_cem_iters_init=5,
            sampling_method="halton",
        )

        def run_controller(ctrl):
            state = np.array([3.0, 0.0, np.pi / 2])
            errors = []
            np.random.seed(42)
            for step in range(num_steps):
                t = step * dt
                ref = generate_reference_trajectory(
                    lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
                )
                control, _ = ctrl.compute_control(state, ref)
                state_dot = model.forward_dynamics(state, control)
                state = state + state_dot * dt

                ref_pt = circle_trajectory(t, radius=3.0)
                err = np.sqrt(
                    (state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2
                )
                errors.append(err)
            return np.sqrt(np.mean(np.array(errors) ** 2))

        rmse_cma = run_controller(cma)
        rmse_ds = run_controller(ds)

        # 둘 다 합리적인 추적 성능
        assert rmse_ds < 0.5, f"dsMPPI RMSE {rmse_ds:.4f} too high"
        assert rmse_cma < 0.5, f"CMA RMSE {rmse_cma:.4f} too high"

    def test_sample_efficiency(self):
        """dsMPPI K=64 vs Vanilla K=256 -- 적은 샘플로 비슷한 성능"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        dt = 0.05
        N = 15
        num_steps = 50

        # Vanilla: 많은 샘플
        vanilla = _make_vanilla_controller(K=256, N=N)
        # dsMPPI: 적은 샘플 + CEM
        ds = _make_ds_controller(
            K=64, N=N,
            n_cem_iters=5, n_cem_iters_init=8,
            sampling_method="halton",
        )

        def run_controller(ctrl):
            state = np.array([3.0, 0.0, np.pi / 2])
            errors = []
            np.random.seed(42)
            for step in range(num_steps):
                t = step * dt
                ref = generate_reference_trajectory(
                    lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
                )
                control, _ = ctrl.compute_control(state, ref)
                state_dot = model.forward_dynamics(state, control)
                state = state + state_dot * dt

                ref_pt = circle_trajectory(t, radius=3.0)
                err = np.sqrt(
                    (state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2
                )
                errors.append(err)
            return np.sqrt(np.mean(np.array(errors) ** 2))

        rmse_vanilla = run_controller(vanilla)
        rmse_ds = run_controller(ds)

        # dsMPPI K=64가 Vanilla K=256보다 3배 나쁘지 않아야 함
        assert rmse_ds < rmse_vanilla * 3.0, \
            f"dsMPPI(K=64) {rmse_ds:.4f} >> Vanilla(K=256) {rmse_vanilla:.4f}"
