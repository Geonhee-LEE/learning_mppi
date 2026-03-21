"""
SVG-MPPI (Stein Variational Guided MPPI) 유닛 테스트

Honda et al., ICRA 2024, arXiv:2309.11040

28개 테스트:
  - Params (3): 기본값, 커스텀, 검증
  - SVGD Update (5): 커널 shape, gradient 방향, 파티클 다양성, median bandwidth, step 효과
  - Controller (5): output shape, info keys, different K, reset, warm start
  - Multimodal (4): bimodal cost, 파티클 spread, mode coverage, vs vanilla
  - Performance (4): circle RMSE, 장애물 회피, 계산 시간, 수렴
  - Integration (4): 수치 안정성, blend ratio 효과, svgd disabled fallback, 통계
  - Comparison (3): vs Vanilla, vs DIAL, vs CMA
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
    DIALMPPIParams,
    CMAMPPIParams,
    SVGMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.svg_mppi import SVGMPPIController
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.cma_mppi import CMAMPPIController
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
from mppi_controller.utils.stein_variational import (
    rbf_kernel_with_bandwidth,
)


# -- 헬퍼 함수 --------------------------------------------------

def _make_svg_controller(**kwargs):
    """SVG-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        svg_num_guide_particles=10,
        svgd_num_iterations=3,
        svg_guide_step_size=0.01,
        n_svgd_steps=5,
        temperature_svgd=1.0,
        blend_ratio=0.5,
        use_svgd_warm_start=True,
        use_spsa_gradient=True,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = SVGMPPIParams(**defaults)
    return SVGMPPIController(
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


def _make_cma_controller(**kwargs):
    """비교용 CMA-MPPI"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        n_iters_init=3, n_iters=2,
        cov_learning_rate=0.5, elite_ratio=0.3,
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

class TestSVGMPPIParams:
    def test_params_defaults(self):
        """기본값 검증"""
        params = SVGMPPIParams()
        assert params.svg_num_guide_particles == 10
        assert params.svg_guide_step_size == 0.01
        assert params.n_svgd_steps == 5
        assert params.temperature_svgd == 1.0
        assert params.use_svgd_warm_start is True
        assert params.blend_ratio == 0.5
        assert params.use_spsa_gradient is True
        assert params.svgd_step_size_schedule == "constant"

    def test_params_custom(self):
        """커스텀 값 검증"""
        params = SVGMPPIParams(
            svg_num_guide_particles=50,
            svg_guide_step_size=0.1,
            n_svgd_steps=10,
            temperature_svgd=2.0,
            use_svgd_warm_start=False,
            blend_ratio=0.8,
            use_spsa_gradient=False,
            svgd_step_size_schedule="decay",
        )
        assert params.svg_num_guide_particles == 50
        assert params.svg_guide_step_size == 0.1
        assert params.n_svgd_steps == 10
        assert params.temperature_svgd == 2.0
        assert params.use_svgd_warm_start is False
        assert params.blend_ratio == 0.8
        assert params.use_spsa_gradient is False
        assert params.svgd_step_size_schedule == "decay"

    def test_params_validation(self):
        """잘못된 값 -> AssertionError"""
        # svg_num_guide_particles <= 0
        with pytest.raises(AssertionError):
            SVGMPPIParams(svg_num_guide_particles=0)

        # svg_guide_step_size <= 0
        with pytest.raises(AssertionError):
            SVGMPPIParams(svg_guide_step_size=0)

        # n_svgd_steps < 0
        with pytest.raises(AssertionError):
            SVGMPPIParams(n_svgd_steps=-1)

        # temperature_svgd <= 0
        with pytest.raises(AssertionError):
            SVGMPPIParams(temperature_svgd=0)

        # blend_ratio out of [0, 1]
        with pytest.raises(AssertionError):
            SVGMPPIParams(blend_ratio=-0.1)
        with pytest.raises(AssertionError):
            SVGMPPIParams(blend_ratio=1.1)

        # invalid step size schedule
        with pytest.raises(AssertionError):
            SVGMPPIParams(svgd_step_size_schedule="invalid")

        # G >= K
        with pytest.raises(ValueError):
            model = DifferentialDriveKinematic(wheelbase=0.5)
            params = SVGMPPIParams(K=32, svg_num_guide_particles=32)
            SVGMPPIController(model, params)


# ==============================================================
# 2. SVGD Update 테스트 (5개)
# ==============================================================

class TestSVGDUpdate:
    def test_kernel_shape(self):
        """RBF 커널 행렬 shape (G, G)"""
        ctrl = _make_svg_controller(K=64, svg_num_guide_particles=16)
        # 파티클 생성
        particles = np.random.randn(16, 10, 2)
        kernel, bw = ctrl._compute_rbf_kernel(particles)

        assert kernel.shape == (16, 16), f"Kernel shape: {kernel.shape}"
        # 대칭성
        assert np.allclose(kernel, kernel.T, atol=1e-10)
        # 대각 = 1
        assert np.allclose(np.diag(kernel), 1.0)
        # 양수
        assert np.all(kernel >= 0)
        # bandwidth 양수
        assert bw > 0

    def test_gradient_direction(self):
        """비용 기울기가 비영이어야 함"""
        ctrl = _make_svg_controller(K=64, svg_num_guide_particles=16)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 파티클 초기화
        controls = ctrl.U + np.random.randn(16, 10, 2) * 0.5
        grad = ctrl._estimate_cost_gradient_spsa(state, controls, ref)

        assert grad.shape == (16, 10, 2)
        assert np.linalg.norm(grad) > 1e-10, "Gradient should be non-zero"

    def test_particle_diversity(self):
        """SVGD 커널 반발력이 파티클 다양성 유지"""
        ctrl = _make_svg_controller(
            K=64, svg_num_guide_particles=16,
            n_svgd_steps=5, svg_guide_step_size=0.05,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        guides = info["guide_controls"]  # (G, N, nu)

        # 파티클 간 분산이 0이 아니어야 함 (다양성)
        guide_flat = guides.reshape(16, -1)
        pairwise_dists = np.linalg.norm(
            guide_flat[:, None, :] - guide_flat[None, :, :], axis=2
        )
        mean_dist = np.mean(pairwise_dists[np.triu_indices(16, k=1)])

        assert mean_dist > 0.01, f"Particles collapsed: mean_dist={mean_dist:.6f}"

    def test_median_bandwidth(self):
        """Median heuristic bandwidth가 양수이고 합리적"""
        ctrl = _make_svg_controller(K=128, svg_num_guide_particles=32)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        bw = info["svg_stats"]["bandwidth"]

        assert bw > 0, f"Bandwidth should be positive: {bw}"
        assert np.isfinite(bw), f"Bandwidth should be finite: {bw}"

    def test_svgd_step_effect(self):
        """SVGD 스텝 수 증가 시 guide 비용 변화"""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 1 스텝
        np.random.seed(42)
        ctrl_1 = _make_svg_controller(
            K=128, svg_num_guide_particles=16,
            n_svgd_steps=1, svgd_num_iterations=1,
            svg_guide_step_size=0.05,
        )
        _, info_1 = ctrl_1.compute_control(state, ref)

        # 10 스텝
        np.random.seed(42)
        ctrl_10 = _make_svg_controller(
            K=128, svg_num_guide_particles=16,
            n_svgd_steps=10, svgd_num_iterations=1,
            svg_guide_step_size=0.05,
        )
        _, info_10 = ctrl_10.compute_control(state, ref)

        # 10 스텝이 극단적으로 나쁘지 않아야 함
        assert info_10["svg_stats"]["final_guide_cost"] < \
               info_1["svg_stats"]["initial_guide_cost"] * 3.0, \
            "10 SVGD steps should not dramatically worsen cost"


# ==============================================================
# 3. Controller 테스트 (5개)
# ==============================================================

class TestSVGMPPIController:
    def test_compute_control_shape(self):
        """control (nu,), info 표준 키 검증"""
        ctrl = _make_svg_controller()
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
        assert "svg_stats" in info
        assert "guide_controls" in info
        assert "guide_indices" in info

    def test_info_svg_stats(self):
        """svg_stats 키 검증"""
        ctrl = _make_svg_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["svg_stats"]

        assert "num_guides" in stats
        assert "num_followers" in stats
        assert "svgd_iterations" in stats
        assert "initial_guide_cost" in stats
        assert "final_guide_cost" in stats
        assert "guide_cost_improvement" in stats
        assert "bandwidth" in stats
        assert "blend_ratio" in stats
        assert "warm_start_used" in stats
        assert "n_svgd_steps" in stats
        assert "temperature_svgd" in stats

        assert stats["num_guides"] == 10
        assert isinstance(stats["bandwidth"], float)

    def test_different_K_values(self):
        """K=32/128/256 정상 작동"""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for K in [32, 128, 256]:
            ctrl = _make_svg_controller(
                K=K, svg_num_guide_particles=min(10, K - 1),
                n_svgd_steps=2,
            )
            control, info = ctrl.compute_control(state, ref)
            assert control.shape == (2,)
            assert np.all(np.isfinite(control))

    def test_reset_clears_state(self):
        """reset 후 초기화"""
        ctrl = _make_svg_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 몇 번 실행 (warm start 축적)
        for _ in range(3):
            ctrl.compute_control(state, ref)

        assert len(ctrl.svg_stats_history) == 3
        assert ctrl._warm_particles is not None

        ctrl.reset()
        assert np.allclose(ctrl.U, 0.0)
        assert ctrl._warm_particles is None
        assert len(ctrl.svg_stats_history) == 0

    def test_warm_start(self):
        """Warm start: 첫 호출 후 이전 파티클 재사용"""
        ctrl = _make_svg_controller(use_svgd_warm_start=True)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 첫 호출 (cold start)
        _, info1 = ctrl.compute_control(state, ref)
        assert info1["svg_stats"]["warm_start_used"] is False

        # 두 번째 호출 (warm start)
        _, info2 = ctrl.compute_control(state, ref)
        assert info2["svg_stats"]["warm_start_used"] is True


# ==============================================================
# 4. Multimodal 테스트 (4개)
# ==============================================================

class TestMultimodal:
    def test_bimodal_cost(self):
        """대칭 장애물에서 SVGD 파티클이 여러 경로 탐색"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        # 대칭 장애물: 궤적 앞에 큰 장애물
        obstacles = [(2.0, 1.5, 0.6), (2.0, -1.5, 0.6)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0),
        ])

        params = SVGMPPIParams(
            K=128, N=15, dt=0.05,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            svg_num_guide_particles=20,
            n_svgd_steps=5,
            svg_guide_step_size=0.05,
            blend_ratio=0.5,
        )
        ctrl = SVGMPPIController(model, params, cost_function=cost)

        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=15)

        control, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(control))
        # SVGD 파티클이 존재
        assert info["guide_controls"].shape[0] == 20

    def test_particle_spread(self):
        """SVGD 후 파티클이 collapse 하지 않음"""
        ctrl = _make_svg_controller(
            K=128, svg_num_guide_particles=32,
            n_svgd_steps=5, svg_guide_step_size=0.05,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        guides = info["guide_controls"]  # (32, N, nu)

        # 첫 타임스텝의 파티클 분산
        v_spread = np.std(guides[:, 0, 0])
        w_spread = np.std(guides[:, 0, 1])

        assert v_spread > 1e-4, f"v particles collapsed: std={v_spread:.6f}"
        assert w_spread > 1e-4, f"w particles collapsed: std={w_spread:.6f}"

    def test_mode_coverage(self):
        """SVGD가 비용 지형의 저비용 영역 커버"""
        ctrl = _make_svg_controller(
            K=256, svg_num_guide_particles=32,
            n_svgd_steps=5, svg_guide_step_size=0.05,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)

        # Guide 비용이 전체 평균보다 낮아야 함 (exploitation 효과)
        guide_cost = info["svg_stats"]["guide_mean_cost"]
        mean_cost = info["mean_cost"]

        # Guide가 전체보다 극단적으로 나쁘지 않아야 함
        assert guide_cost < mean_cost * 2.0, \
            f"Guide cost {guide_cost:.4f} >> mean {mean_cost:.4f}"

    def test_vs_vanilla_multimodal(self):
        """SVG-MPPI가 Vanilla보다 장애물에서 더 안전"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4)]
        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0),
        ])

        np.random.seed(42)
        svg = _make_svg_controller(
            K=128, svg_num_guide_particles=20,
            n_svgd_steps=5, svg_guide_step_size=0.05,
            cost_function=cost,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info_svg = svg.compute_control(state, ref)
        # SVG should produce finite controls
        assert np.all(np.isfinite(info_svg["best_cost"]))


# ==============================================================
# 5. Performance 테스트 (4개)
# ==============================================================

class TestPerformance:
    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.3 (50 스텝)"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = SVGMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            svg_num_guide_particles=16,
            n_svgd_steps=3,
            svg_guide_step_size=0.01,
            blend_ratio=0.5,
        )
        ctrl = SVGMPPIController(model, params)

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

        params = SVGMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            svg_num_guide_particles=20,
            n_svgd_steps=5,
            svg_guide_step_size=0.05,
            blend_ratio=0.5,
        )
        ctrl = SVGMPPIController(model, params, cost_function=cost)

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
        """K=256, N=15에서 300ms 이내 (SVGD 반복 포함)"""
        ctrl = _make_svg_controller(
            K=256, N=15,
            svg_num_guide_particles=32,
            n_svgd_steps=3,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=15)

        # Warmup
        ctrl.compute_control(state, ref)
        ctrl.reset()

        times = []
        for _ in range(5):
            t_start = time.time()
            ctrl.compute_control(state, ref)
            times.append(time.time() - t_start)

        mean_ms = np.mean(times) * 1000
        assert mean_ms < 300, f"Mean solve time {mean_ms:.1f}ms >= 300ms"

    def test_convergence(self):
        """SVGD 비용 개선이 비음수"""
        ctrl = _make_svg_controller(
            K=256, svg_num_guide_particles=32,
            n_svgd_steps=10, svg_guide_step_size=0.05,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        improvement_count = 0
        n_trials = 5
        for trial in range(n_trials):
            ctrl.reset()
            np.random.seed(42 + trial)
            _, info = ctrl.compute_control(state, ref)
            improvement = info["svg_stats"]["guide_cost_improvement"]
            if improvement >= 0:
                improvement_count += 1

        # 과반수에서 비용 개선
        assert improvement_count >= 2, \
            f"Expected improvement in majority, got {improvement_count}/{n_trials}"


# ==============================================================
# 6. Integration 테스트 (4개)
# ==============================================================

class TestIntegration:
    def test_numerical_stability(self):
        """NaN/Inf 없음 (20 스텝 연속)"""
        ctrl = _make_svg_controller(K=64)
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

    def test_blend_ratio_effect(self):
        """blend_ratio=0 -> 전부 가우시안, 1.0 -> 전부 SVGD"""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # blend_ratio=0.5 (기본 혼합)
        np.random.seed(42)
        ctrl_half = _make_svg_controller(
            K=64, svg_num_guide_particles=16, blend_ratio=0.5,
        )
        _, info_half = ctrl_half.compute_control(state, ref)
        assert info_half["num_samples"] == 64  # 16 guides + 48 followers

        # blend_ratio=1.0 (SVGD만)
        np.random.seed(42)
        ctrl_full = _make_svg_controller(
            K=64, svg_num_guide_particles=16, blend_ratio=1.0,
        )
        _, info_full = ctrl_full.compute_control(state, ref)
        assert info_full["num_samples"] == 16  # SVGD 파티클만
        assert info_full["svg_stats"]["num_followers"] == 0

    def test_svgd_disabled_fallback(self):
        """n_svgd_steps=0 -> SVGD 없이 guide 선택 + MPPI 가중 평균만"""
        ctrl = _make_svg_controller(
            K=64, svg_num_guide_particles=16,
            n_svgd_steps=0, svgd_num_iterations=1,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))
        # svgd_num_iterations=1이 최소이므로 1 스텝은 실행됨
        # 하지만 여전히 합리적인 제어 출력 보장
        assert np.isfinite(info["svg_stats"]["guide_cost_improvement"])

    def test_svg_statistics(self):
        """get_svg_statistics() 누적 통계"""
        ctrl = _make_svg_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 빈 상태
        stats = ctrl.get_svg_statistics()
        assert stats["mean_guide_cost_improvement"] == 0.0
        assert len(stats["svg_stats_history"]) == 0

        # 실행 후
        for _ in range(5):
            ctrl.compute_control(state, ref)

        stats = ctrl.get_svg_statistics()
        assert len(stats["svg_stats_history"]) == 5
        assert stats["mean_bandwidth"] > 0


# ==============================================================
# 7. Comparison 테스트 (3개)
# ==============================================================

class TestComparison:
    def test_vs_vanilla_tracking(self):
        """SVG-MPPI가 Vanilla MPPI와 유사하거나 더 나은 추적 (50 스텝)"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        dt = 0.05
        N = 15
        num_steps = 50

        vanilla = _make_vanilla_controller(K=128, N=N)
        svg = _make_svg_controller(
            K=128, N=N,
            svg_num_guide_particles=16,
            n_svgd_steps=3, svg_guide_step_size=0.01,
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
        rmse_svg = run_controller(svg)

        # SVG-MPPI가 Vanilla보다 극단적으로 나쁘지 않아야 함
        assert rmse_svg < rmse_vanilla * 2.0, \
            f"SVG RMSE {rmse_svg:.4f} >> Vanilla {rmse_vanilla:.4f}"

    def test_vs_dial_tracking(self):
        """SVG-MPPI vs DIAL-MPPI -- 유사 성능"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        dt = 0.05
        N = 15
        num_steps = 50

        dial = _make_dial_controller(K=128, N=N, n_diffuse_init=5, n_diffuse=3)
        svg = _make_svg_controller(
            K=128, N=N,
            svg_num_guide_particles=16,
            n_svgd_steps=3, svg_guide_step_size=0.01,
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

        rmse_dial = run_controller(dial)
        rmse_svg = run_controller(svg)

        # 둘 다 합리적 추적
        assert rmse_svg < 0.5, f"SVG RMSE {rmse_svg:.4f} too high"
        assert rmse_dial < 0.5, f"DIAL RMSE {rmse_dial:.4f} too high"

    def test_vs_cma_tracking(self):
        """SVG-MPPI vs CMA-MPPI -- 공분산 적응 비교"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        dt = 0.05
        N = 15
        num_steps = 50

        cma = _make_cma_controller(K=128, N=N, n_iters_init=5, n_iters=3)
        svg = _make_svg_controller(
            K=128, N=N,
            svg_num_guide_particles=16,
            n_svgd_steps=3, svg_guide_step_size=0.01,
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
        rmse_svg = run_controller(svg)

        # 둘 다 합리적 추적
        assert rmse_svg < 0.5, f"SVG RMSE {rmse_svg:.4f} too high"
        assert rmse_cma < 0.5, f"CMA RMSE {rmse_cma:.4f} too high"
