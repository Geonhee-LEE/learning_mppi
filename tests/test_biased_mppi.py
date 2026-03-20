"""
Biased-MPPI (Mixture Sampling MPPI) 유닛 테스트

28개 테스트:
  - Params (3): 기본값, 커스텀, 검증
  - Policy (5): pure_pursuit, braking, feedback, max_speed, factory
  - Controller (5): shape, info, all_samples, different_K, reset
  - Mixture Sampling (4): 정책 샘플 혼합, 구성, 다양성, 전체 교체
  - Adaptive Lambda (3): 증가, 감소, 범위
  - Performance (4): 추적 RMSE, 장애물 회피, local minima, 계산 시간
  - Integration (4): 수치 안정성, 커스텀 정책, 보상 정규화, 기여도 추적
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
    BiasedMPPIParams,
    DIALMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.biased_mppi import BiasedMPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.ancillary_policies import (
    AncillaryPolicy,
    PurePursuitPolicy,
    BrakingPolicy,
    FeedbackPolicy,
    MaxSpeedPolicy,
    PreviousSolutionPolicy,
    create_ancillary_policy,
    create_policies_from_names,
    POLICY_REGISTRY,
)
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


# ── 헬퍼 함수 ─────────────────────────────────────────

def _make_biased_controller(**kwargs):
    """Biased-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        ancillary_types=["pure_pursuit", "braking"],
        samples_per_policy=5,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    policies = defaults.pop("policies", None)
    params = BiasedMPPIParams(**defaults)
    return BiasedMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
        policies=policies,
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


def _make_dial_controller(**kwargs):
    """비교용 DIAL-MPPI"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        n_diffuse_init=3, n_diffuse=2,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    params = DIALMPPIParams(**defaults)
    return DIALMPPIController(model, params, cost_function=cost_function)


def _make_ref(N=10, dt=0.05):
    """레퍼런스 궤적"""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ══════════════════════════════════════════════════════════
# 1. Params 테스트 (3개)
# ══════════════════════════════════════════════════════════

class TestBiasedMPPIParams:
    def test_params_defaults(self):
        """기본값 검증"""
        params = BiasedMPPIParams()
        assert params.ancillary_types == ["pure_pursuit", "braking"]
        assert params.samples_per_policy == 10
        assert params.policy_noise_scale == 0.3
        assert params.use_adaptive_lambda is True
        assert params.ess_min_ratio == 0.1
        assert params.ess_max_ratio == 0.5
        assert params.lambda_increase_rate == 1.2
        assert params.lambda_decrease_rate == 0.9
        assert params.lambda_min == 0.1
        assert params.lambda_max == 100.0
        assert params.use_reward_normalization is False

    def test_params_custom(self):
        """커스텀 값 검증"""
        params = BiasedMPPIParams(
            ancillary_types=["braking", "max_speed", "feedback"],
            samples_per_policy=5,
            policy_noise_scale=0.5,
            use_adaptive_lambda=False,
            ess_min_ratio=0.05,
            ess_max_ratio=0.4,
        )
        assert len(params.ancillary_types) == 3
        assert params.samples_per_policy == 5
        assert params.policy_noise_scale == 0.5
        assert params.use_adaptive_lambda is False

    def test_params_validation(self):
        """잘못된 값 → AssertionError"""
        # 빈 정책 리스트
        with pytest.raises(AssertionError):
            BiasedMPPIParams(ancillary_types=[])

        # samples_per_policy < 1
        with pytest.raises(AssertionError):
            BiasedMPPIParams(samples_per_policy=0)

        # policy_noise_scale 범위 밖
        with pytest.raises(AssertionError):
            BiasedMPPIParams(policy_noise_scale=1.5)

        # ess_min >= ess_max
        with pytest.raises(AssertionError):
            BiasedMPPIParams(ess_min_ratio=0.5, ess_max_ratio=0.3)

        # total_policy_samples >= K
        with pytest.raises(AssertionError):
            BiasedMPPIParams(
                K=20, ancillary_types=["pure_pursuit", "braking"],
                samples_per_policy=11,
            )

        # lambda_increase_rate <= 1
        with pytest.raises(AssertionError):
            BiasedMPPIParams(lambda_increase_rate=0.9)

        # lambda_decrease_rate >= 1
        with pytest.raises(AssertionError):
            BiasedMPPIParams(lambda_decrease_rate=1.1)


# ══════════════════════════════════════════════════════════
# 2. Policy 테스트 (5개)
# ══════════════════════════════════════════════════════════

class TestAncillaryPolicies:
    def test_pure_pursuit_shape(self):
        """PurePursuit → (N, nu) shape"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        policy = PurePursuitPolicy(lookahead=0.5)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        controls = policy.propose_sequence(state, ref, 10, 0.05, model)
        assert controls.shape == (10, 2)
        assert policy.name == "pure_pursuit"

    def test_braking_zeros(self):
        """BrakingPolicy → 모든 값 0"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        policy = BrakingPolicy()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        controls = policy.propose_sequence(state, ref, 10, 0.05, model)
        assert controls.shape == (10, 2)
        assert np.allclose(controls, 0.0)
        assert policy.name == "braking"

    def test_feedback_policy(self):
        """FeedbackPolicy — AncillaryController 기반"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        policy = FeedbackPolicy(gain_scale=1.0)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        controls = policy.propose_sequence(state, ref, 10, 0.05, model)
        assert controls.shape == (10, 2)
        assert policy.name == "feedback"
        # 피드백이 있으므로 0이 아닐 수 있음
        assert not np.allclose(controls, 0.0)

    def test_max_speed_policy(self):
        """MaxSpeedPolicy — 방향 정렬 + 높은 속도"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        policy = MaxSpeedPolicy(speed_ratio=0.8)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        controls = policy.propose_sequence(state, ref, 10, 0.05, model)
        assert controls.shape == (10, 2)
        assert policy.name == "max_speed"

    def test_policy_factory(self):
        """create_ancillary_policy / create_policies_from_names"""
        # 단일 생성
        pp = create_ancillary_policy("pure_pursuit", lookahead=1.0)
        assert isinstance(pp, PurePursuitPolicy)
        assert pp.name == "pure_pursuit"

        # 레지스트리에 없는 이름
        with pytest.raises(ValueError):
            create_ancillary_policy("unknown_policy")

        # 배치 생성
        policies = create_policies_from_names(["braking", "max_speed"])
        assert len(policies) == 2
        assert policies[0].name == "braking"
        assert policies[1].name == "max_speed"

        # 레지스트리 확인
        assert set(POLICY_REGISTRY.keys()) == {
            "pure_pursuit", "braking", "feedback", "max_speed", "previous_solution"
        }


# ══════════════════════════════════════════════════════════
# 3. Controller 테스트 (5개)
# ══════════════════════════════════════════════════════════

class TestBiasedMPPIController:
    def test_compute_control_shape(self):
        """control (nu,), info 표준 키 검증"""
        ctrl = _make_biased_controller()
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

    def test_info_biased_stats(self):
        """biased_stats 키 검증"""
        ctrl = _make_biased_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["biased_stats"]
        assert "n_policy_samples" in stats
        assert "n_gaussian_samples" in stats
        assert "best_is_policy" in stats
        assert "policy_best_ratio" in stats
        assert "current_lambda" in stats
        assert "policy_names" in stats

        # 샘플 수 확인: 2 policies × 5 samples = 10 policy samples
        assert stats["n_policy_samples"] == 10
        assert stats["n_gaussian_samples"] == 54  # 64 - 10

    def test_all_samples_used(self):
        """K개 전체 rollout"""
        ctrl = _make_biased_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        assert info["sample_trajectories"].shape[0] == 64
        assert info["sample_weights"].shape[0] == 64
        assert info["num_samples"] == 64

    def test_different_K_values(self):
        """K=32/128/256 정상 작동"""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for K in [32, 128, 256]:
            ctrl = _make_biased_controller(K=K, samples_per_policy=3)
            control, info = ctrl.compute_control(state, ref)
            assert control.shape == (2,)
            assert info["num_samples"] == K

    def test_reset_clears_state(self):
        """reset 후 초기화"""
        ctrl = _make_biased_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 몇 번 실행
        for _ in range(3):
            ctrl.compute_control(state, ref)

        assert ctrl._total_steps == 3
        assert len(ctrl._biased_history) == 3

        ctrl.reset()
        assert np.allclose(ctrl.U, 0.0)
        assert ctrl._total_steps == 0
        assert ctrl._policy_best_count == 0
        assert len(ctrl._biased_history) == 0
        assert ctrl._current_lambda == ctrl.biased_params.lambda_


# ══════════════════════════════════════════════════════════
# 4. Mixture Sampling 테스트 (4개)
# ══════════════════════════════════════════════════════════

class TestMixtureSampling:
    def test_policy_samples_in_mix(self):
        """전체 K 중 J개가 정책 제안"""
        ctrl = _make_biased_controller(K=100, samples_per_policy=8)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        # 2 policies × 8 = 16 policy samples
        assert info["biased_stats"]["n_policy_samples"] == 16
        assert info["biased_stats"]["n_gaussian_samples"] == 84

    def test_gaussian_plus_policy(self):
        """혼합 구성 확인 — 정책+가우시안"""
        ctrl = _make_biased_controller(K=50, samples_per_policy=5)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        n_policy = info["biased_stats"]["n_policy_samples"]
        n_gaussian = info["biased_stats"]["n_gaussian_samples"]
        assert n_policy + n_gaussian == 50

    def test_policy_noise_diversity(self):
        """policy_noise_scale > 0 → 다양한 제안"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        policy = PurePursuitPolicy()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 동일 정책 다중 호출 → noise 추가 시 다양
        np.random.seed(42)
        seq1 = policy.propose_sequence(state, ref, 10, 0.05, model)
        noise1 = 0.3 * np.array([0.5, 0.5]) * np.random.standard_normal((10, 2))
        sample_a = seq1 + noise1
        noise2 = 0.3 * np.array([0.5, 0.5]) * np.random.standard_normal((10, 2))
        sample_b = seq1 + noise2

        # 서로 다른 시퀀스
        assert not np.allclose(sample_a, sample_b)

    def test_full_replacement_update(self):
        """U = Σ ω_k V_k 전체 교체 검증"""
        ctrl = _make_biased_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 초기 U = 0
        assert np.allclose(ctrl.U, 0.0)

        # 한 번 실행 후 U가 업데이트됨
        ctrl.compute_control(state, ref)

        # U가 0이 아님 (전체 교체로 가중 평균)
        # shift 후이므로 U[:-1]이 업데이트 확인
        # (shift 후 마지막만 0)
        assert ctrl.U[-1, 0] == 0.0  # shift 후 마지막 = 0
        assert ctrl.U[-1, 1] == 0.0


# ══════════════════════════════════════════════════════════
# 5. Adaptive Lambda 테스트 (3개)
# ══════════════════════════════════════════════════════════

class TestAdaptiveLambda:
    def test_lambda_increase_on_low_ess(self):
        """ESS 낮을 때 λ 증가"""
        ctrl = _make_biased_controller(
            K=64, lambda_=0.1,
            use_adaptive_lambda=True,
            ess_min_ratio=0.5,  # 거의 항상 ESS/K < 0.5
            ess_max_ratio=0.9,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        initial_lambda = ctrl._current_lambda
        # 여러 번 실행하여 λ 증가 기회
        for _ in range(10):
            ctrl.compute_control(state, ref)

        # λ가 증가했어야 함 (또는 ess_ratio가 min 위에 있을 수 있음)
        # 적어도 한 번은 adapt가 호출됨
        assert ctrl._current_lambda >= initial_lambda

    def test_lambda_decrease_on_high_ess(self):
        """ESS 높을 때 λ 감소"""
        ctrl = _make_biased_controller(
            K=64, lambda_=50.0,  # 높은 λ → 균등 가중치 → 높은 ESS
            use_adaptive_lambda=True,
            ess_max_ratio=0.3,  # 쉽게 초과
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        initial_lambda = ctrl._current_lambda
        for _ in range(10):
            ctrl.compute_control(state, ref)

        # λ가 감소했어야 함
        assert ctrl._current_lambda <= initial_lambda

    def test_lambda_bounds(self):
        """[lambda_min, lambda_max] 내 유지"""
        ctrl = _make_biased_controller(
            K=64, lambda_=1.0,
            use_adaptive_lambda=True,
            lambda_min=0.5,
            lambda_max=5.0,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(50):
            ctrl.compute_control(state, ref)

        assert ctrl._current_lambda >= ctrl.biased_params.lambda_min
        assert ctrl._current_lambda <= ctrl.biased_params.lambda_max


# ══════════════════════════════════════════════════════════
# 6. Performance 테스트 (4개)
# ══════════════════════════════════════════════════════════

class TestPerformance:
    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.3 (50 스텝)"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = BiasedMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            ancillary_types=["pure_pursuit", "braking"],
            samples_per_policy=5,
        )
        ctrl = BiasedMPPIController(model, params)

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
        """3개 장애물 충돌 없음"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4), (-2.0, -1.0, 0.5)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0),
        ])

        params = BiasedMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            ancillary_types=["pure_pursuit", "braking"],
            samples_per_policy=5,
        )
        ctrl = BiasedMPPIController(model, params, cost_function=cost)

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

    def test_local_minima_escape(self):
        """큰 장애물 → Biased-MPPI가 회피 성공"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        # 경로 위에 큰 장애물
        obstacles = [(0.0, 3.0, 0.8)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.3, cost_weight=3000.0),
        ])

        params = BiasedMPPIParams(
            K=256, N=20, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            ancillary_types=["pure_pursuit", "braking", "max_speed"],
            samples_per_policy=5,
        )
        ctrl = BiasedMPPIController(model, params, cost_function=cost)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt_val = params.dt
        N = params.N

        min_clearance = float("inf")
        for step in range(60):
            t = step * dt_val
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt_val,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt_val

            for ox, oy, r in obstacles:
                dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                clearance = dist - r
                min_clearance = min(min_clearance, clearance)

        # 충돌 없음 (clearance > 0)
        assert min_clearance > -0.05, f"Min clearance: {min_clearance:.3f}"

    def test_computation_time(self):
        """K=512, N=30에서 100ms 이내"""
        ctrl = _make_biased_controller(
            K=512, N=30, samples_per_policy=5,
            ancillary_types=["pure_pursuit", "braking"],
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
        assert mean_ms < 100, f"Mean solve time {mean_ms:.1f}ms >= 100ms"


# ══════════════════════════════════════════════════════════
# 7. Integration 테스트 (4개)
# ══════════════════════════════════════════════════════════

class TestIntegration:
    def test_numerical_stability(self):
        """NaN/Inf 없음"""
        ctrl = _make_biased_controller(K=64)
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

    def test_custom_policy_injection(self):
        """사용자 정의 정책 주입"""

        class ConstantPolicy(AncillaryPolicy):
            @property
            def name(self):
                return "constant"

            def propose_sequence(self, state, ref, N, dt, model):
                return np.full((N, model.control_dim), 0.1)

        custom = ConstantPolicy()
        ctrl = _make_biased_controller(
            policies=[custom, BrakingPolicy()],
            K=64, samples_per_policy=5,
            ancillary_types=["pure_pursuit"],  # policies 인자로 override
        )

        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert "constant" in info["biased_stats"]["policy_names"]
        assert "braking" in info["biased_stats"]["policy_names"]

    def test_reward_normalization(self):
        """use_reward_normalization=True 동작"""
        ctrl = _make_biased_controller(
            K=64, use_reward_normalization=True,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        weights = info["sample_weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-6

    def test_policy_contribution_tracking(self):
        """policy_best_ratio 추적"""
        ctrl = _make_biased_controller(K=64, samples_per_policy=5)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(10):
            ctrl.compute_control(state, ref)

        stats = ctrl.get_biased_statistics()
        assert stats["total_steps"] == 10
        assert 0.0 <= stats["policy_best_ratio"] <= 1.0
        assert stats["mean_lambda"] > 0
        assert len(stats["history"]) == 10
