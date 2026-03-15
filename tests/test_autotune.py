"""
MPPI Autotune 모듈 테스트

테스트 대상:
    - AutotuneObjective: 메트릭 → 스칼라 변환, 프리셋, 제약 페널티
    - AutotuneConfig: 기본값, 커스텀 설정
    - MPPIAutotuner: 벡터↔파라미터 변환, 최적화
    - OnlineSigmaAdapter: 샘플링 shape, 적응 동작, 클리핑, reset
    - AutotunedMPPIController: 래핑, 적응 통합, 원형 추적
"""

import numpy as np
import pytest

from mppi_controller.controllers.mppi.autotune import (
    AutotuneObjective,
    AutotuneConfig,
    MPPIAutotuner,
    OnlineSigmaAdapter,
    AutotunedMPPIController,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.adaptive_temperature import AdaptiveTemperature
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.utils.trajectory import (
    circle_trajectory,
    generate_reference_trajectory,
)

# ── 테스트 헬퍼 ───────────────────────────────────────────────────

# 빠른 실행을 위한 경량 파라미터
FAST_K = 32
FAST_N = 10
FAST_DT = 0.05
FAST_DURATION = 3.0


def make_model():
    return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)


def make_params(**overrides):
    defaults = dict(
        K=FAST_K,
        N=FAST_N,
        dt=FAST_DT,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(overrides)
    return MPPIParams(**defaults)


def make_reference_fn(params):
    """레퍼런스 궤적 함수 (Simulator.run 호환)"""
    def ref_fn(t):
        return generate_reference_trajectory(
            lambda t_: circle_trajectory(t_, radius=2.0, angular_velocity=0.3),
            t, params.N, params.dt,
        )
    return ref_fn


# ═══════════════════════════════════════════════════════════════════
# TestAutotuneObjective
# ═══════════════════════════════════════════════════════════════════


class TestAutotuneObjective:
    def test_evaluate_basic(self):
        """가중 정규화 합산 테스트"""
        obj = AutotuneObjective(
            metric_weights={"position_rmse": 1.0, "control_rate": 0.5},
            metric_targets={"position_rmse": 0.1, "control_rate": 1.0},
        )
        metrics = {"position_rmse": 0.05, "control_rate": 0.3}
        score = obj.evaluate(metrics)

        # 1.0 * 0.05/0.1 + 0.5 * 0.3/1.0 = 0.5 + 0.15 = 0.65
        assert abs(score - 0.65) < 1e-6

    def test_evaluate_missing_metric(self):
        """존재하지 않는 메트릭은 무시"""
        obj = AutotuneObjective(
            metric_weights={"position_rmse": 1.0, "nonexistent": 2.0},
        )
        metrics = {"position_rmse": 0.1}
        score = obj.evaluate(metrics)
        assert abs(score - 0.1) < 1e-6

    def test_evaluate_max_constraint_violation(self):
        """max 제약 위반 시 큰 페널티"""
        obj = AutotuneObjective(
            metric_weights={"position_rmse": 1.0},
            penalty_constraints={"mean_solve_time": ("max", 50.0)},
        )
        metrics = {"position_rmse": 0.1, "mean_solve_time": 100.0}
        score = obj.evaluate(metrics)
        assert score > 1e5  # 페널티가 매우 큼

    def test_evaluate_min_constraint_violation(self):
        """min 제약 위반 시 큰 페널티"""
        obj = AutotuneObjective(
            metric_weights={},
            penalty_constraints={"position_rmse": ("min", 0.5)},
        )
        metrics = {"position_rmse": 0.1}
        score = obj.evaluate(metrics)
        assert score > 1e5

    def test_evaluate_constraint_satisfied(self):
        """제약 만족 시 페널티 없음"""
        obj = AutotuneObjective(
            metric_weights={"position_rmse": 1.0},
            penalty_constraints={"mean_solve_time": ("max", 100.0)},
        )
        metrics = {"position_rmse": 0.1, "mean_solve_time": 50.0}
        score = obj.evaluate(metrics)
        assert score < 1.0  # 페널티 없음

    def test_presets(self):
        """프리셋 팩토리 메서드 테스트"""
        tracking = AutotuneObjective.tracking_focused()
        balanced = AutotuneObjective.balanced()
        safety = AutotuneObjective.safety_focused()

        metrics = {
            "position_rmse": 0.1,
            "heading_rmse": 0.05,
            "max_position_error": 0.3,
            "control_rate": 0.2,
            "mean_solve_time": 30.0,
        }

        # 모든 프리셋이 유한한 스칼라 반환
        for obj in [tracking, balanced, safety]:
            score = obj.evaluate(metrics)
            assert np.isfinite(score)
            assert score > 0


# ═══════════════════════════════════════════════════════════════════
# TestAutotuneConfig
# ═══════════════════════════════════════════════════════════════════


class TestAutotuneConfig:
    def test_defaults(self):
        """기본 설정 검증"""
        config = AutotuneConfig()
        assert "lambda_" in config.tunable_params
        assert "sigma_scale" in config.tunable_params
        assert config.optimizer == "differential_evolution"
        assert config.max_iterations == 50

    def test_custom(self):
        """커스텀 설정"""
        config = AutotuneConfig(
            tunable_params=["lambda_"],
            param_bounds={"lambda_": (0.5, 10.0)},
            optimizer="nelder_mead",
            max_iterations=20,
        )
        assert config.tunable_params == ["lambda_"]
        assert config.param_bounds["lambda_"] == (0.5, 10.0)
        assert config.optimizer == "nelder_mead"


# ═══════════════════════════════════════════════════════════════════
# TestMPPIAutotuner
# ═══════════════════════════════════════════════════════════════════


class TestMPPIAutotuner:
    def test_vector_to_params_lambda(self):
        """lambda_ 벡터↔파라미터 변환"""
        params = make_params()
        config = AutotuneConfig(
            tunable_params=["lambda_"],
            param_bounds={"lambda_": (0.1, 10.0)},
        )
        tuner = MPPIAutotuner(
            model_fn=make_model,
            controller_cls=MPPIController,
            base_params=params,
            reference_fn=make_reference_fn(params),
            initial_state=np.array([2.0, 0.0, np.pi / 2]),
            objective=AutotuneObjective.tracking_focused(),
            config=config,
        )

        vec = np.array([5.0])
        new_params = tuner._vector_to_params(vec)
        assert abs(new_params.lambda_ - 5.0) < 1e-6

    def test_vector_to_params_sigma_scale(self):
        """sigma_scale 변환: sigma = base_sigma * scale"""
        params = make_params(sigma=np.array([0.5, 0.3]))
        config = AutotuneConfig(
            tunable_params=["sigma_scale"],
            param_bounds={"sigma_scale": (0.1, 5.0)},
        )
        tuner = MPPIAutotuner(
            model_fn=make_model,
            controller_cls=MPPIController,
            base_params=params,
            reference_fn=make_reference_fn(params),
            initial_state=np.array([2.0, 0.0, np.pi / 2]),
            objective=AutotuneObjective.tracking_focused(),
            config=config,
        )

        new_params = tuner._vector_to_params(np.array([2.0]))
        np.testing.assert_allclose(new_params.sigma, [1.0, 0.6])

    def test_vector_to_params_Q_R_scale(self):
        """Q_scale, R_scale 변환"""
        params = make_params()
        config = AutotuneConfig(
            tunable_params=["Q_scale", "R_scale"],
            param_bounds={"Q_scale": (0.1, 50.0), "R_scale": (0.01, 10.0)},
        )
        tuner = MPPIAutotuner(
            model_fn=make_model,
            controller_cls=MPPIController,
            base_params=params,
            reference_fn=make_reference_fn(params),
            initial_state=np.array([2.0, 0.0, np.pi / 2]),
            objective=AutotuneObjective.tracking_focused(),
            config=config,
        )

        new_params = tuner._vector_to_params(np.array([2.0, 3.0]))
        np.testing.assert_allclose(new_params.Q, params.Q * 2.0)
        np.testing.assert_allclose(new_params.R, params.R * 3.0)

    def test_params_to_vector(self):
        """파라미터 → 벡터 초기점 생성"""
        params = make_params(lambda_=5.0)
        config = AutotuneConfig(
            tunable_params=["lambda_", "sigma_scale"],
            param_bounds={"lambda_": (0.1, 10.0), "sigma_scale": (0.1, 5.0)},
        )
        tuner = MPPIAutotuner(
            model_fn=make_model,
            controller_cls=MPPIController,
            base_params=params,
            reference_fn=make_reference_fn(params),
            initial_state=np.array([2.0, 0.0, np.pi / 2]),
            objective=AutotuneObjective.tracking_focused(),
            config=config,
        )

        vec = tuner._params_to_vector(params)
        assert abs(vec[0] - 5.0) < 1e-6  # lambda_
        assert abs(vec[1] - 1.0) < 1e-6  # sigma_scale (default 1.0)

    def test_evaluate_params_returns_finite(self):
        """단일 평가가 유한 스칼라 반환"""
        params = make_params()
        config = AutotuneConfig(
            tunable_params=["lambda_"],
            param_bounds={"lambda_": (0.1, 10.0)},
            sim_duration=FAST_DURATION,
        )
        tuner = MPPIAutotuner(
            model_fn=make_model,
            controller_cls=MPPIController,
            base_params=params,
            reference_fn=make_reference_fn(params),
            initial_state=np.array([2.0, 0.0, np.pi / 2]),
            objective=AutotuneObjective.tracking_focused(),
            config=config,
        )

        score = tuner._evaluate_params(np.array([1.0]))
        assert np.isfinite(score)
        assert score > 0

    @pytest.mark.timeout(120)
    def test_tune_improves_over_bad_params(self):
        """의도적으로 나쁜 초기 파라미터 → 튜닝 후 개선 확인"""
        # 나쁜 파라미터: 매우 높은 lambda (탐색 과다), 낮은 Q (추적 약)
        bad_params = make_params(lambda_=30.0, Q=np.array([0.5, 0.5, 0.1]))
        ref_fn = make_reference_fn(bad_params)
        initial_state = np.array([2.0, 0.0, np.pi / 2])

        config = AutotuneConfig(
            tunable_params=["lambda_", "Q_scale"],
            param_bounds={
                "lambda_": (0.1, 50.0),
                "Q_scale": (0.1, 50.0),
            },
            optimizer="differential_evolution",
            max_iterations=5,
            sim_duration=FAST_DURATION,
            seed=42,
            verbose=False,
        )

        tuner = MPPIAutotuner(
            model_fn=make_model,
            controller_cls=MPPIController,
            base_params=bad_params,
            reference_fn=ref_fn,
            initial_state=initial_state,
            objective=AutotuneObjective.tracking_focused(),
            config=config,
        )

        # 나쁜 초기 파라미터의 점수
        initial_vec = tuner._params_to_vector(bad_params)
        initial_score = tuner._evaluate_params(initial_vec)

        # 튜닝 실행
        tuner._eval_count = 0
        best_params, result_info = tuner.tune()

        # 튜닝된 파라미터가 더 나은 점수
        assert result_info["best_score"] <= initial_score


# ═══════════════════════════════════════════════════════════════════
# TestOnlineSigmaAdapter
# ═══════════════════════════════════════════════════════════════════


class TestOnlineSigmaAdapter:
    def test_sample_shape(self):
        """샘플 shape 검증: (K, N, nu)"""
        adapter = OnlineSigmaAdapter(
            base_sigma=np.array([0.5, 0.5]), seed=42
        )
        U = np.zeros((10, 2))
        noise = adapter.sample(U, K=16)
        assert noise.shape == (16, 10, 2)

    def test_sample_with_control_bounds(self):
        """제어 제약 클리핑"""
        adapter = OnlineSigmaAdapter(
            base_sigma=np.array([1.0, 1.0]), seed=42
        )
        U = np.zeros((10, 2))
        noise = adapter.sample(
            U, K=100,
            control_min=np.array([-0.5, -0.5]),
            control_max=np.array([0.5, 0.5]),
        )
        sampled = U + noise
        assert np.all(sampled >= -0.5 - 1e-10)
        assert np.all(sampled <= 0.5 + 1e-10)

    def test_update_low_ess_increases_sigma(self):
        """ESS 낮을 때 σ 증가"""
        adapter = OnlineSigmaAdapter(
            base_sigma=np.array([0.5, 0.5]),
            adaptation_rate=0.2,
        )

        # 낮은 ESS (소수 샘플에 집중)
        info = {
            "ess": 3.0,
            "num_samples": 100,
            "best_cost": 1.0,
            "mean_cost": 5.0,
            "sample_trajectories": np.zeros((100, 11, 3)),
        }
        adapter.update(info)

        stats = adapter.get_statistics()
        assert stats["has_adapted"]
        assert stats["mean_ratio"] > 1.0  # σ 증가

    def test_update_high_ess_low_gap_decreases_sigma(self):
        """ESS 높고 cost gap 작을 때 σ 감소"""
        adapter = OnlineSigmaAdapter(
            base_sigma=np.array([0.5, 0.5]),
            adaptation_rate=0.2,
        )

        info = {
            "ess": 80.0,
            "num_samples": 100,
            "best_cost": 1.0,
            "mean_cost": 1.05,  # gap = 0.05/1.05 < 0.1
            "sample_trajectories": np.zeros((100, 11, 3)),
        }
        adapter.update(info)

        stats = adapter.get_statistics()
        assert stats["has_adapted"]
        assert stats["mean_ratio"] < 1.0  # σ 감소

    def test_update_clipping(self):
        """σ 비율이 min/max 범위 내로 클리핑"""
        adapter = OnlineSigmaAdapter(
            base_sigma=np.array([0.5, 0.5]),
            adaptation_rate=0.5,
            min_sigma_ratio=0.5,
            max_sigma_ratio=2.0,
        )

        # 반복적으로 낮은 ESS → σ 계속 증가 시도
        for _ in range(50):
            adapter.update({
                "ess": 1.0,
                "num_samples": 100,
                "best_cost": 1.0,
                "mean_cost": 10.0,
                "sample_trajectories": np.zeros((100, 11, 3)),
            })

        stats = adapter.get_statistics()
        assert stats["max_ratio"] <= 2.0 + 1e-10  # max 클리핑

    def test_reset(self):
        """reset 후 초기 상태 복원"""
        adapter = OnlineSigmaAdapter(
            base_sigma=np.array([0.5, 0.5]),
        )

        adapter.update({
            "ess": 3.0,
            "num_samples": 100,
            "best_cost": 1.0,
            "mean_cost": 5.0,
            "sample_trajectories": np.zeros((100, 11, 3)),
        })
        assert adapter.get_statistics()["has_adapted"]

        adapter.reset()
        assert not adapter.get_statistics()["has_adapted"]
        assert adapter.get_statistics()["mean_ratio"] == 1.0

    def test_adapted_sample_differs(self):
        """적응 후 노이즈 분포가 변경됨"""
        adapter = OnlineSigmaAdapter(
            base_sigma=np.array([0.5, 0.5]),
            adaptation_rate=0.3,
            seed=42,
        )

        U = np.zeros((10, 2))

        # 적응 전 샘플
        noise_before = adapter.sample(U, K=1000)
        std_before = np.std(noise_before, axis=0).mean()

        # σ 증가 유도 (낮은 ESS)
        adapter.update({
            "ess": 2.0,
            "num_samples": 100,
            "best_cost": 1.0,
            "mean_cost": 10.0,
            "sample_trajectories": np.zeros((100, 11, 3)),
        })

        # 적응 후 샘플
        adapter.rng = np.random.default_rng(42)  # 동일 시드
        noise_after = adapter.sample(U, K=1000)
        std_after = np.std(noise_after, axis=0).mean()

        # 적응 후 노이즈 표준편차 증가
        assert std_after > std_before


# ═══════════════════════════════════════════════════════════════════
# TestAutotunedMPPIController
# ═══════════════════════════════════════════════════════════════════


class TestAutotunedMPPIController:
    def test_wrapping(self):
        """래퍼가 기존 컨트롤러를 올바르게 감쌈"""
        model = make_model()
        params = make_params()
        controller = MPPIController(model, params)
        wrapped = AutotunedMPPIController(controller)

        assert wrapped.controller is controller
        assert wrapped.params is params
        assert wrapped.model is model

    def test_compute_control_returns_valid(self):
        """compute_control이 올바른 반환값 제공"""
        model = make_model()
        params = make_params()
        controller = MPPIController(model, params)
        wrapped = AutotunedMPPIController(controller)

        state = np.array([2.0, 0.0, np.pi / 2])
        ref = generate_reference_trajectory(
            lambda t: circle_trajectory(t, radius=2.0, angular_velocity=0.3),
            0.0, params.N, params.dt,
        )

        control, info = wrapped.compute_control(state, ref)
        assert control.shape == (2,)
        assert "autotune_stats" in info

    def test_sigma_adapter_integration(self):
        """sigma adapter가 compute_control에서 업데이트됨"""
        model = make_model()
        params = make_params()
        controller = MPPIController(model, params)
        adapter = OnlineSigmaAdapter(params.sigma.copy(), seed=42)
        wrapped = AutotunedMPPIController(controller, sigma_adapter=adapter)

        state = np.array([2.0, 0.0, np.pi / 2])
        ref = generate_reference_trajectory(
            lambda t: circle_trajectory(t, radius=2.0, angular_velocity=0.3),
            0.0, params.N, params.dt,
        )

        # 여러 번 호출하여 적응 동작 확인
        for _ in range(5):
            control, info = wrapped.compute_control(state, ref)

        stats = adapter.get_statistics()
        assert stats["has_adapted"]
        assert "sigma_mean_ratio" in info["autotune_stats"]

    def test_temperature_adapter_integration(self):
        """temperature adapter가 compute_control에서 업데이트됨"""
        model = make_model()
        params = make_params(lambda_=1.0)
        controller = MPPIController(model, params)
        temp_adapter = AdaptiveTemperature(initial_lambda=1.0)
        wrapped = AutotunedMPPIController(
            controller, temperature_adapter=temp_adapter
        )

        state = np.array([2.0, 0.0, np.pi / 2])
        ref = generate_reference_trajectory(
            lambda t: circle_trajectory(t, radius=2.0, angular_velocity=0.3),
            0.0, params.N, params.dt,
        )

        initial_lambda = params.lambda_
        for _ in range(10):
            control, info = wrapped.compute_control(state, ref)

        # λ가 적응으로 변경됨 (ESS에 따라 변화할 수 있음)
        assert "lambda" in info["autotune_stats"]

    def test_reset(self):
        """reset이 컨트롤러와 어댑터 모두 초기화"""
        model = make_model()
        params = make_params()
        controller = MPPIController(model, params)
        adapter = OnlineSigmaAdapter(params.sigma.copy())
        temp = AdaptiveTemperature()
        wrapped = AutotunedMPPIController(
            controller, sigma_adapter=adapter, temperature_adapter=temp
        )

        wrapped.reset()
        assert not adapter.get_statistics()["has_adapted"]

    def test_circle_tracking_with_online_adaptation(self):
        """온라인 적응으로 원형 궤적 추적"""
        from mppi_controller.simulation.simulator import Simulator
        from mppi_controller.simulation.metrics import compute_metrics

        model = make_model()
        params = make_params()
        controller = MPPIController(model, params)
        adapter = OnlineSigmaAdapter(params.sigma.copy(), seed=42)
        temp = AdaptiveTemperature(initial_lambda=params.lambda_)
        wrapped = AutotunedMPPIController(
            controller, sigma_adapter=adapter, temperature_adapter=temp
        )

        ref_fn = make_reference_fn(params)
        initial_state = np.array([2.0, 0.0, np.pi / 2])

        sim = Simulator(model, wrapped, params.dt)
        sim.reset(initial_state)
        history = sim.run(ref_fn, FAST_DURATION)
        metrics = compute_metrics(history)

        # 기본적인 추적 성능
        assert metrics["position_rmse"] < 1.0  # 합리적 추적
        assert metrics["mean_solve_time"] < 500.0  # 합리적 시간
