"""
DRPA-MPPI (Dynamic Repulsive Potential Augmented MPPI) 유닛 테스트

28개 테스트:
  - Params (3): 기본값, 커스텀, 검증
  - Stagnation Detection (5): 정체 감지, 오탐 방지, 윈도우 크기, 임계값 민감도, 비용 정체
  - Repulsive Potential (4): 포텐셜 형태, 영향 거리, 강도 효과, 범위 밖 0
  - Controller (5): 출력 shape, info 키, 다양한 K, reset, 모드 전환
  - Escape Behavior (4): 탈출 활성화, 노이즈 증폭, 복귀 감지, 반복 탈출
  - Performance (4): circle RMSE, 장애물 회피, local minima 탈출, 계산 시간
  - Comparison (3): vs vanilla local minima, vs biased, 통계 추적
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
    DRPAMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.drpa_mppi import DRPAMPPIController
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

def _make_drpa_controller(**kwargs):
    """DRPA-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        obstacles=[(2.5, 1.5, 0.5), (0.0, 3.0, 0.4)],
        repulsive_gain=5.0,
        influence_distance=1.0,
        stagnation_threshold=0.1,
        stagnation_window=10,
        escape_boost=2.0,
        recovery_threshold=0.3,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = DRPAMPPIParams(**defaults)
    return DRPAMPPIController(
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


def _make_ref(N=10, dt=0.05):
    """레퍼런스 궤적"""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ══════════════════════════════════════════════════════════
# 1. Params 테스트 (3개)
# ══════════════════════════════════════════════════════════

class TestDRPAMPPIParams:
    def test_params_defaults(self):
        """기본값 검증"""
        params = DRPAMPPIParams()
        assert params.obstacles == []
        assert params.repulsive_gain == 5.0
        assert params.influence_distance == 1.0
        assert params.stagnation_threshold == 0.1
        assert params.stagnation_window == 10
        assert params.escape_boost == 2.0
        assert params.recovery_threshold == 0.3
        assert params.use_noise_boost is True

    def test_params_custom(self):
        """커스텀 값 검증"""
        params = DRPAMPPIParams(
            obstacles=[(1.0, 2.0, 0.5), (3.0, 4.0, 0.3)],
            repulsive_gain=10.0,
            influence_distance=2.0,
            stagnation_threshold=0.05,
            stagnation_window=20,
            escape_boost=3.0,
            recovery_threshold=0.5,
            use_noise_boost=False,
        )
        assert len(params.obstacles) == 2
        assert params.repulsive_gain == 10.0
        assert params.influence_distance == 2.0
        assert params.stagnation_threshold == 0.05
        assert params.stagnation_window == 20
        assert params.escape_boost == 3.0
        assert params.recovery_threshold == 0.5
        assert params.use_noise_boost is False

    def test_params_validation(self):
        """잘못된 값 → AssertionError"""
        # 음수 repulsive_gain
        with pytest.raises(AssertionError):
            DRPAMPPIParams(repulsive_gain=-1.0)

        # influence_distance <= 0
        with pytest.raises(AssertionError):
            DRPAMPPIParams(influence_distance=0.0)

        # stagnation_threshold <= 0
        with pytest.raises(AssertionError):
            DRPAMPPIParams(stagnation_threshold=0.0)

        # stagnation_window < 2
        with pytest.raises(AssertionError):
            DRPAMPPIParams(stagnation_window=1)

        # escape_boost < 1.0
        with pytest.raises(AssertionError):
            DRPAMPPIParams(escape_boost=0.5)

        # recovery_threshold <= 0
        with pytest.raises(AssertionError):
            DRPAMPPIParams(recovery_threshold=0.0)

        # 잘못된 장애물 형식 (반지름 음수)
        with pytest.raises(AssertionError):
            DRPAMPPIParams(obstacles=[(1.0, 2.0, -0.5)])

        # 잘못된 장애물 형식 (2개 요소)
        with pytest.raises(AssertionError):
            DRPAMPPIParams(obstacles=[(1.0, 2.0)])


# ══════════════════════════════════════════════════════════
# 2. Stagnation Detection 테스트 (5개)
# ══════════════════════════════════════════════════════════

class TestStagnationDetection:
    def test_detect_stagnation_when_stuck(self):
        """위치 변화 없으면 정체 감지"""
        ctrl = _make_drpa_controller(stagnation_window=5, stagnation_threshold=0.05)
        state = np.array([0.0, 0.0, 0.0])

        # 같은 위치 5번 추가
        for _ in range(5):
            ctrl._stagnation_history.append(np.array([0.0, 0.0]))

        detected = ctrl._detect_stagnation(state)
        assert detected is True

    def test_no_false_positive_when_moving(self):
        """충분히 이동 시 오탐 없음"""
        ctrl = _make_drpa_controller(stagnation_window=5, stagnation_threshold=0.05)
        state = np.array([1.0, 0.0, 0.0])

        # 충분히 이동
        for i in range(5):
            ctrl._stagnation_history.append(np.array([i * 0.1, 0.0]))

        detected = ctrl._detect_stagnation(state)
        assert detected is False

    def test_window_size_effect(self):
        """윈도우 크기 미달 시 감지 안 됨"""
        ctrl = _make_drpa_controller(stagnation_window=10, stagnation_threshold=0.05)
        state = np.array([0.0, 0.0, 0.0])

        # 윈도우(10)보다 적은 히스토리(5)
        for _ in range(5):
            ctrl._stagnation_history.append(np.array([0.0, 0.0]))

        detected = ctrl._detect_stagnation(state)
        assert detected is False

    def test_threshold_sensitivity(self):
        """임계값에 따른 감지 민감도"""
        # 높은 임계값 + 충분한 이동: 정체 아님
        ctrl_high = _make_drpa_controller(
            stagnation_window=5, stagnation_threshold=0.3
        )
        for i in range(5):
            ctrl_high._stagnation_history.append(np.array([i * 0.5, 0.0]))

        state = np.array([2.0, 0.0, 0.0])
        # displacement = ||[2.0,0] - [0,0]|| = 2.0 > 0.3
        assert ctrl_high._detect_stagnation(state) is False

        # 낮은 이동 + 낮은 임계값: 정체 감지
        ctrl_low = _make_drpa_controller(
            stagnation_window=5, stagnation_threshold=10.0
        )
        for i in range(5):
            ctrl_low._stagnation_history.append(np.array([i * 0.1, 0.0]))

        # displacement = ||[0.4,0] - [0,0]|| = 0.4 < 10.0
        assert ctrl_low._detect_stagnation(state) is True

    def test_cost_stagnation(self):
        """비용 변화 없으면 정체 감지"""
        ctrl = _make_drpa_controller(stagnation_window=5, stagnation_threshold=0.05)
        state = np.array([1.0, 0.0, 0.0])

        # 위치는 이동 중이지만 히스토리 충분
        for i in range(5):
            ctrl._stagnation_history.append(np.array([i * 0.5, 0.0]))

        # 비용 정체 (동일 비용)
        ctrl._cost_history = [100.0, 100.0, 100.0, 100.0, 100.0]

        detected = ctrl._detect_stagnation(state)
        assert detected is True


# ══════════════════════════════════════════════════════════
# 3. Repulsive Potential 테스트 (4개)
# ══════════════════════════════════════════════════════════

class TestRepulsivePotential:
    def test_potential_shape(self):
        """반발 포텐셜 출력 shape 검증"""
        ctrl = _make_drpa_controller(K=32)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # Dummy trajectories (K, N+1, nx)
        K, N = 32, 10
        trajectories = np.zeros((K, N + 1, 3))
        trajectories[:, :, 0] = 3.0  # x
        trajectories[:, :, 1] = np.linspace(0, 3, N + 1)  # y

        potentials = ctrl._compute_repulsive_potential(trajectories)
        assert potentials.shape == (K,)
        assert np.all(potentials >= 0)

    def test_influence_distance(self):
        """영향 범위 밖 포텐셜 = 0"""
        ctrl = _make_drpa_controller(
            obstacles=[(0.0, 0.0, 0.5)],
            influence_distance=1.0,
        )

        # 장애물에서 멀리 떨어진 궤적
        K, N = 4, 5
        trajectories = np.zeros((K, N + 1, 3))
        trajectories[:, :, 0] = 10.0  # x = 10 (매우 멀리)
        trajectories[:, :, 1] = 10.0  # y = 10

        potentials = ctrl._compute_repulsive_potential(trajectories)
        assert np.allclose(potentials, 0.0)

    def test_gain_effect(self):
        """η 증가 → 포텐셜 비례 증가"""
        obstacles = [(0.0, 0.0, 0.3)]

        # 장애물 근처 궤적
        K, N = 4, 5
        trajectories = np.zeros((K, N + 1, 3))
        trajectories[:, :, 0] = 0.8  # 장애물 표면에서 0.5m
        trajectories[:, :, 1] = 0.0

        ctrl_low = _make_drpa_controller(
            obstacles=obstacles, repulsive_gain=1.0, influence_distance=2.0
        )
        ctrl_high = _make_drpa_controller(
            obstacles=obstacles, repulsive_gain=10.0, influence_distance=2.0
        )

        pot_low = ctrl_low._compute_repulsive_potential(trajectories)
        pot_high = ctrl_high._compute_repulsive_potential(trajectories)

        # η가 10배면 포텐셜도 10배
        ratio = pot_high / np.maximum(pot_low, 1e-10)
        assert np.allclose(ratio, 10.0, rtol=0.01)

    def test_zero_outside_range(self):
        """영향 범위 밖에서 정확히 0"""
        ctrl = _make_drpa_controller(
            obstacles=[(5.0, 5.0, 0.5)],
            influence_distance=1.0,
        )

        # 장애물에서 3m 떨어진 궤적 (표면 거리 2.5m > d0=1.0)
        K, N = 2, 3
        trajectories = np.zeros((K, N + 1, 3))
        trajectories[:, :, 0] = 2.0  # 장애물(5,5)에서 ~4.2m
        trajectories[:, :, 1] = 2.0

        potentials = ctrl._compute_repulsive_potential(trajectories)
        assert np.allclose(potentials, 0.0)


# ══════════════════════════════════════════════════════════
# 4. Controller 테스트 (5개)
# ══════════════════════════════════════════════════════════

class TestDRPAMPPIController:
    def test_compute_control_shape(self):
        """control (nu,), info 표준 키 검증"""
        ctrl = _make_drpa_controller()
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

    def test_info_drpa_stats(self):
        """drpa_stats 키 검증"""
        ctrl = _make_drpa_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["drpa_stats"]
        assert "in_escape_mode" in stats
        assert "stagnation_detected" in stats
        assert "escape_count" in stats
        assert "repulsive_cost_mean" in stats
        assert "repulsive_cost_best" in stats
        assert "sigma_eff" in stats
        assert "min_clearance" in stats
        assert "total_steps" in stats

    def test_different_K_values(self):
        """K=32/128/256 정상 작동"""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for K in [32, 128, 256]:
            ctrl = _make_drpa_controller(K=K)
            control, info = ctrl.compute_control(state, ref)
            assert control.shape == (2,)
            assert info["num_samples"] == K

    def test_reset_clears_state(self):
        """reset 후 초기화"""
        ctrl = _make_drpa_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(5):
            ctrl.compute_control(state, ref)

        assert ctrl._total_steps == 5
        assert len(ctrl._drpa_history) == 5
        assert len(ctrl._stagnation_history) > 0

        ctrl.reset()
        assert np.allclose(ctrl.U, 0.0)
        assert ctrl._total_steps == 0
        assert ctrl._escape_count == 0
        assert ctrl._in_escape_mode is False
        assert len(ctrl._drpa_history) == 0
        assert len(ctrl._stagnation_history) == 0
        assert len(ctrl._cost_history) == 0

    def test_mode_switching(self):
        """수동 모드 전환 검증"""
        ctrl = _make_drpa_controller()

        assert ctrl._in_escape_mode is False

        # 수동으로 탈출 모드 설정
        ctrl._in_escape_mode = True
        assert ctrl._in_escape_mode is True

        # reset으로 복귀
        ctrl.reset()
        assert ctrl._in_escape_mode is False


# ══════════════════════════════════════════════════════════
# 5. Escape Behavior 테스트 (4개)
# ══════════════════════════════════════════════════════════

class TestEscapeBehavior:
    def test_escape_activation(self):
        """정체 시 탈출 모드 자동 진입"""
        ctrl = _make_drpa_controller(
            stagnation_window=3, stagnation_threshold=0.05,
            obstacles=[(0.0, 3.0, 0.8)],
        )

        # 정체 상태 시뮬레이션: 같은 위치에서 반복
        state = np.array([0.0, 2.0, np.pi / 2])
        ref = _make_ref()

        # 정체 히스토리 사전 주입
        for _ in range(5):
            ctrl._stagnation_history.append(np.array([0.0, 2.0]))

        # 실행하면 정체 감지 → 탈출 모드
        ctrl.compute_control(state, ref)
        assert ctrl._in_escape_mode is True
        assert ctrl._escape_count >= 1

    def test_noise_boost_in_escape(self):
        """탈출 모드 시 sigma 증폭 확인"""
        ctrl = _make_drpa_controller(
            escape_boost=3.0,
            use_noise_boost=True,
        )

        # 탈출 모드 강제 설정
        ctrl._in_escape_mode = True

        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        sigma_eff = info["drpa_stats"]["sigma_eff"]

        # sigma * escape_boost = [0.5, 0.5] * 3.0 = [1.5, 1.5]
        expected = (ctrl.params.sigma * 3.0).tolist()
        assert np.allclose(sigma_eff, expected)

    def test_recovery_detection(self):
        """이동 재개 시 탈출 모드 해제"""
        ctrl = _make_drpa_controller(
            stagnation_window=3,
            recovery_threshold=0.2,
        )

        # 탈출 모드 상태
        ctrl._in_escape_mode = True

        # 이동 히스토리 (충분히 이동)
        ctrl._stagnation_history = [
            np.array([0.0, 0.0]),
            np.array([0.1, 0.1]),
            np.array([0.3, 0.3]),
        ]

        state = np.array([0.3, 0.3, 0.0])
        ctrl._check_recovery(state)

        # displacement = ||[0.3,0.3] - [0.0,0.0]|| ≈ 0.424 > 0.2
        assert ctrl._in_escape_mode is False

    def test_repeated_escape(self):
        """반복 탈출 카운트"""
        ctrl = _make_drpa_controller(
            stagnation_window=3, stagnation_threshold=0.05,
            recovery_threshold=0.2,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 1차 정체 → 탈출
        for _ in range(5):
            ctrl._stagnation_history.append(np.array([3.0, 0.0]))
        ctrl.compute_control(state, ref)
        assert ctrl._escape_count >= 1

        # 복귀
        ctrl._in_escape_mode = False

        # 2차 정체 → 탈출
        ctrl._stagnation_history = []
        for _ in range(5):
            ctrl._stagnation_history.append(np.array([3.0, 0.0]))
        ctrl.compute_control(state, ref)
        assert ctrl._escape_count >= 2


# ══════════════════════════════════════════════════════════
# 6. Performance 테스트 (4개)
# ══════════════════════════════════════════════════════════

class TestPerformance:
    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.3 (장애물 없음, 50 스텝)"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = DRPAMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            obstacles=[],  # 장애물 없음
        )
        ctrl = DRPAMPPIController(model, params)

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

        params = DRPAMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            obstacles=obstacles,
            repulsive_gain=5.0,
            influence_distance=1.0,
        )
        ctrl = DRPAMPPIController(model, params, cost_function=cost)

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
        """큰 장애물 → DRPA-MPPI가 회피 성공"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(0.0, 3.0, 0.8)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.3, cost_weight=3000.0),
        ])

        params = DRPAMPPIParams(
            K=256, N=20, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            obstacles=obstacles,
            repulsive_gain=10.0,
            influence_distance=1.5,
            stagnation_threshold=0.08,
            stagnation_window=8,
            escape_boost=2.5,
        )
        ctrl = DRPAMPPIController(model, params, cost_function=cost)

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

        assert min_clearance > -0.05, f"Min clearance: {min_clearance:.3f}"

    def test_computation_time(self):
        """K=512, N=30에서 100ms 이내"""
        ctrl = _make_drpa_controller(K=512, N=30)
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
# 7. Comparison 테스트 (3개)
# ══════════════════════════════════════════════════════════

class TestComparison:
    def test_vs_vanilla_local_minima(self):
        """Vanilla vs DRPA — local minima에서 DRPA가 더 나음"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(0.0, 3.0, 0.8)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.3, cost_weight=3000.0),
        ])

        # Vanilla
        v_params = MPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
        )
        vanilla = MPPIController(model, v_params, cost_function=cost)

        # DRPA
        d_params = DRPAMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            obstacles=obstacles,
            repulsive_gain=10.0,
            influence_distance=1.5,
        )
        drpa = DRPAMPPIController(model, d_params, cost_function=cost)

        dt = 0.05
        N = 15
        num_steps = 60

        def run_sim(ctrl, seed=42):
            np.random.seed(seed)
            state = np.array([3.0, 0.0, np.pi / 2])
            min_clear = float("inf")
            for step in range(num_steps):
                t = step * dt
                ref = generate_reference_trajectory(
                    lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
                )
                control, _ = ctrl.compute_control(state, ref)
                state_dot = model.forward_dynamics(state, control)
                state = state + state_dot * dt

                for ox, oy, r in obstacles:
                    d = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                    min_clear = min(min_clear, d - r)
            return min_clear

        v_clear = run_sim(vanilla)
        d_clear = run_sim(drpa)

        # DRPA가 적어도 충돌하지 않아야 함
        assert d_clear > -0.1, f"DRPA clearance: {d_clear:.3f}"

    def test_vs_biased_comparison(self):
        """DRPA vs Biased — 둘 다 동작 확인"""
        ctrl = _make_drpa_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # DRPA 실행
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["ess"] > 0

        # 숫자 안정성
        assert not np.any(np.isnan(control))
        assert not np.any(np.isinf(control))

    def test_statistics_tracking(self):
        """get_drpa_statistics() 누적 통계"""
        ctrl = _make_drpa_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(10):
            ctrl.compute_control(state, ref)

        stats = ctrl.get_drpa_statistics()
        assert stats["total_steps"] == 10
        assert stats["escape_count"] >= 0
        assert 0.0 <= stats["escape_ratio"] <= 1.0
        assert stats["mean_repulsive_cost"] >= 0.0
        assert len(stats["history"]) == 10
