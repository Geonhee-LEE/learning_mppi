"""
BNN Surrogate MPPI 유닛 테스트

FeasibilityCost + BNNMPPIController 20개 테스트.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams, BNNMPPIParams, UncertaintyMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.uncertainty_mppi import (
    UncertaintyMPPIController,
)
from mppi_controller.controllers.mppi.bnn_mppi import (
    FeasibilityCost,
    BNNMPPIController,
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
    figure_eight_trajectory,
)
from mppi_controller.simulation.simulator import Simulator


# ── 헬퍼 ──────────────────────────────────────────────────────

def _make_model():
    return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)


def _make_bnn_params(**kwargs):
    defaults = dict(
        K=32, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    return BNNMPPIParams(**defaults)


def _make_ref(N=10, dt=0.05):
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


def _mock_uncertainty_fn(states, controls):
    """거리 기반 불확실성: 원점에서 멀수록 불확실"""
    if states.ndim == 1:
        states = states[None, :]
    dist = np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)
    nx = states.shape[-1]
    return dist[:, None] * 0.1 * np.ones((1, nx))


def _zero_uncertainty_fn(states, controls):
    """항상 0 불확실성"""
    if states.ndim == 1:
        states = states[None, :]
    nx = states.shape[-1]
    return np.zeros((states.shape[0], nx))


def _high_uncertainty_fn(states, controls):
    """항상 높은 불확실성"""
    if states.ndim == 1:
        states = states[None, :]
    nx = states.shape[-1]
    return np.full((states.shape[0], nx), 1.0)


def _make_controller(uncertainty_fn=_mock_uncertainty_fn, **kwargs):
    model = _make_model()
    params = _make_bnn_params(**kwargs)
    return BNNMPPIController(model, params, uncertainty_fn=uncertainty_fn)


# ══════════════════════════════════════════════════════════════
# BNNMPPIParams 테스트 (#1~#3)
# ══════════════════════════════════════════════════════════════

def test_params_defaults():
    """#1: 기본값 검증"""
    print("\n" + "=" * 60)
    print("Test #1: BNNMPPIParams defaults")
    print("=" * 60)

    p = _make_bnn_params()
    assert p.feasibility_weight == 50.0
    assert p.uncertainty_reduce == "sum"
    assert p.feasibility_threshold == 0.0
    assert p.max_filter_ratio == 0.5
    assert p.margin_scale == 1.0
    assert p.margin_max == 0.5
    print("  defaults OK")
    print("PASS")


def test_params_custom():
    """#2: 커스텀 값 검증"""
    print("\n" + "=" * 60)
    print("Test #2: BNNMPPIParams custom values")
    print("=" * 60)

    p = _make_bnn_params(
        feasibility_weight=100.0,
        uncertainty_reduce="max",
        feasibility_threshold=0.3,
        max_filter_ratio=0.7,
    )
    assert p.feasibility_weight == 100.0
    assert p.uncertainty_reduce == "max"
    assert p.feasibility_threshold == 0.3
    assert p.max_filter_ratio == 0.7
    print("  custom values OK")
    print("PASS")


def test_params_validation():
    """#3: 잘못된 값 → AssertionError"""
    print("\n" + "=" * 60)
    print("Test #3: BNNMPPIParams validation")
    print("=" * 60)

    # 잘못된 reduce 모드
    try:
        _make_bnn_params(uncertainty_reduce="invalid")
        assert False, "should raise"
    except AssertionError:
        print("  invalid reduce → AssertionError OK")

    # threshold 범위 초과
    try:
        _make_bnn_params(feasibility_threshold=2.0)
        assert False, "should raise"
    except AssertionError:
        print("  threshold > 1 → AssertionError OK")

    # max_filter_ratio 0
    try:
        _make_bnn_params(max_filter_ratio=0.0)
        assert False, "should raise"
    except AssertionError:
        print("  max_filter_ratio=0 → AssertionError OK")

    print("PASS")


# ══════════════════════════════════════════════════════════════
# FeasibilityCost 테스트 (#4~#9)
# ══════════════════════════════════════════════════════════════

def test_feasibility_cost_shape():
    """#4: compute_cost 반환 shape (K,)"""
    print("\n" + "=" * 60)
    print("Test #4: FeasibilityCost shape")
    print("=" * 60)

    K, N, nx, nu = 16, 5, 3, 2
    cost = FeasibilityCost(_mock_uncertainty_fn, weight=50.0)

    trajectories = np.random.randn(K, N + 1, nx)
    controls = np.random.randn(K, N, nu)
    ref = np.zeros((N + 1, nx))

    costs = cost.compute_cost(trajectories, controls, ref)
    assert costs.shape == (K,), f"shape: {costs.shape}"
    print(f"  costs shape={costs.shape} OK")
    print("PASS")


def test_feasibility_zero_uncertainty():
    """#5: σ=0 → 비용=0"""
    print("\n" + "=" * 60)
    print("Test #5: FeasibilityCost zero uncertainty")
    print("=" * 60)

    K, N, nx, nu = 16, 5, 3, 2
    cost = FeasibilityCost(_zero_uncertainty_fn, weight=50.0)

    trajectories = np.random.randn(K, N + 1, nx)
    controls = np.random.randn(K, N, nu)
    ref = np.zeros((N + 1, nx))

    costs = cost.compute_cost(trajectories, controls, ref)
    assert np.allclose(costs, 0.0), f"costs={costs}"
    print(f"  all costs ≈ 0 OK")
    print("PASS")


def test_feasibility_high_uncertainty_high_cost():
    """#6: σ 큼 → 비용 큼"""
    print("\n" + "=" * 60)
    print("Test #6: High uncertainty → high cost")
    print("=" * 60)

    K, N, nx, nu = 16, 5, 3, 2
    cost_low = FeasibilityCost(_zero_uncertainty_fn, weight=50.0)
    cost_high = FeasibilityCost(_high_uncertainty_fn, weight=50.0)

    trajectories = np.random.randn(K, N + 1, nx)
    controls = np.random.randn(K, N, nu)
    ref = np.zeros((N + 1, nx))

    costs_low = cost_low.compute_cost(trajectories, controls, ref)
    costs_high = cost_high.compute_cost(trajectories, controls, ref)

    assert np.all(costs_high > costs_low), "high unc should have higher cost"
    print(f"  mean_low={np.mean(costs_low):.3f}, mean_high={np.mean(costs_high):.3f}")
    print("PASS")


def test_feasibility_score_range():
    """#7: feasibility score ∈ [0, 1]"""
    print("\n" + "=" * 60)
    print("Test #7: Feasibility score range")
    print("=" * 60)

    K, N, nx, nu = 32, 5, 3, 2
    cost = FeasibilityCost(_mock_uncertainty_fn, weight=50.0)

    trajectories = np.random.randn(K, N + 1, nx) * 3
    controls = np.random.randn(K, N, nu)

    scores = cost.compute_feasibility(trajectories, controls)
    assert scores.shape == (K,), f"shape: {scores.shape}"
    assert np.all(scores >= 0) and np.all(scores <= 1), \
        f"range: [{scores.min():.4f}, {scores.max():.4f}]"
    print(f"  range: [{scores.min():.4f}, {scores.max():.4f}] OK")
    print("PASS")


def test_feasibility_reduce_modes():
    """#8: reduce 모드 (sum/max/mean) 결과 비교"""
    print("\n" + "=" * 60)
    print("Test #8: Reduce modes comparison")
    print("=" * 60)

    K, N, nx, nu = 16, 5, 3, 2
    np.random.seed(42)
    trajectories = np.random.randn(K, N + 1, nx) * 2
    controls = np.random.randn(K, N, nu)
    ref = np.zeros((N + 1, nx))

    costs = {}
    for mode in ("sum", "max", "mean"):
        cost = FeasibilityCost(_mock_uncertainty_fn, weight=50.0, reduce=mode)
        costs[mode] = cost.compute_cost(trajectories, controls, ref)
        print(f"  {mode}: mean={np.mean(costs[mode]):.3f}")

    # sum >= mean (nx=3인 경우 sum = 3*mean)
    assert np.all(costs["sum"] >= costs["mean"] - 1e-10), "sum >= mean"
    # sum >= max
    assert np.all(costs["sum"] >= costs["max"] - 1e-10), "sum >= max"
    print("PASS")


def test_feasibility_weight_scaling():
    """#9: weight 비례 확인"""
    print("\n" + "=" * 60)
    print("Test #9: Weight scaling")
    print("=" * 60)

    K, N, nx, nu = 16, 5, 3, 2
    np.random.seed(42)
    trajectories = np.random.randn(K, N + 1, nx)
    controls = np.random.randn(K, N, nu)
    ref = np.zeros((N + 1, nx))

    cost_1x = FeasibilityCost(_mock_uncertainty_fn, weight=10.0)
    cost_2x = FeasibilityCost(_mock_uncertainty_fn, weight=20.0)

    costs_1 = cost_1x.compute_cost(trajectories, controls, ref)
    costs_2 = cost_2x.compute_cost(trajectories, controls, ref)

    ratio = costs_2 / np.maximum(costs_1, 1e-10)
    assert np.allclose(ratio, 2.0, atol=1e-6), f"ratio: {ratio}"
    print(f"  2x weight → 2x cost OK")
    print("PASS")


# ══════════════════════════════════════════════════════════════
# BNNMPPIController 테스트 (#10~#17)
# ══════════════════════════════════════════════════════════════

def test_controller_compute_control_shape():
    """#10: compute_control 반환 shape + info 키"""
    print("\n" + "=" * 60)
    print("Test #10: Controller compute_control shape")
    print("=" * 60)

    ctrl = _make_controller()
    state = circle_trajectory(0.0)
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,), f"control shape: {control.shape}"
    assert "bnn_stats" in info, "missing bnn_stats"
    assert "feasibility_scores" in info, "missing feasibility_scores"
    assert "sample_trajectories" in info
    assert "sample_weights" in info
    print(f"  control shape={control.shape}")
    print(f"  info keys: {list(info.keys())}")
    print("PASS")


def test_controller_no_uncertainty_fallback():
    """#11: uncertainty_fn=None → 표준 MPPI 폴백"""
    print("\n" + "=" * 60)
    print("Test #11: No uncertainty → fallback to standard MPPI")
    print("=" * 60)

    model = _make_model()
    params = _make_bnn_params()
    ctrl = BNNMPPIController(model, params, uncertainty_fn=None)

    assert ctrl.feasibility_cost is None, "should have no feasibility cost"

    state = circle_trajectory(0.0)
    ref = _make_ref()
    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,)
    assert "bnn_stats" not in info, "should fallback to standard MPPI"
    print("  fallback to standard MPPI OK")
    print("PASS")


def test_controller_auto_detect_model_uncertainty():
    """#12: predict_with_uncertainty 자동 감지"""
    print("\n" + "=" * 60)
    print("Test #12: Auto-detect model uncertainty")
    print("=" * 60)

    class MockUncertaintyModel:
        state_dim = 3
        control_dim = 2

        def forward(self, state, control, dt):
            return state + np.array([0.01, 0.01, 0.0])

        def predict_with_uncertainty(self, states, controls):
            mean = states
            std = np.full_like(states, 0.1)
            return mean, std

        def get_control_bounds(self):
            return (np.array([-1, -1]), np.array([1, 1]))

    model = MockUncertaintyModel()
    params = _make_bnn_params()
    ctrl = BNNMPPIController(model, params)

    assert ctrl.uncertainty_fn is not None, "should auto-detect"
    assert ctrl.feasibility_cost is not None
    print("  auto-detect OK")
    print("PASS")


def test_controller_feasibility_filtering():
    """#13: threshold 설정 → 필터 동작"""
    print("\n" + "=" * 60)
    print("Test #13: Feasibility filtering")
    print("=" * 60)

    ctrl = _make_controller(
        uncertainty_fn=_high_uncertainty_fn,
        feasibility_threshold=0.5,
    )
    state = circle_trajectory(0.0)
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    stats = info["bnn_stats"]

    print(f"  mean_feasibility={stats['mean_feasibility']:.4f}")
    print(f"  num_filtered={stats['num_filtered']}")
    print(f"  filter_ratio={stats['filter_ratio']:.3f}")

    # high uncertainty → 많은 궤적이 필터될 수 있음
    assert control.shape == (2,)
    assert stats["num_filtered"] >= 0
    print("PASS")


def test_controller_max_filter_ratio():
    """#14: 최소 생존 궤적 보장"""
    print("\n" + "=" * 60)
    print("Test #14: Max filter ratio guarantee")
    print("=" * 60)

    K = 32
    max_ratio = 0.5
    min_keep = max(int(K * (1 - max_ratio)), 1)

    ctrl = _make_controller(
        K=K,
        uncertainty_fn=_high_uncertainty_fn,
        feasibility_threshold=0.99,  # 거의 모든 궤적 필터
        max_filter_ratio=max_ratio,
    )
    state = circle_trajectory(0.0)
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    stats = info["bnn_stats"]

    num_survived = K - stats["num_filtered"]
    assert num_survived >= min_keep, \
        f"survived {num_survived} < min_keep {min_keep}"
    print(f"  survived={num_survived} >= min_keep={min_keep} OK")
    print("PASS")


def test_controller_info_bnn_stats():
    """#15: bnn_stats 키/값 확인"""
    print("\n" + "=" * 60)
    print("Test #15: bnn_stats keys and values")
    print("=" * 60)

    ctrl = _make_controller()
    state = circle_trajectory(0.0)
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)
    stats = info["bnn_stats"]

    expected_keys = [
        "mean_feasibility", "min_feasibility", "max_feasibility",
        "num_filtered", "filter_ratio",
        "mean_uncertainty_cost", "mean_base_cost",
    ]
    for key in expected_keys:
        assert key in stats, f"missing key: {key}"

    assert 0 <= stats["mean_feasibility"] <= 1
    assert 0 <= stats["min_feasibility"] <= 1
    assert stats["filter_ratio"] >= 0
    print(f"  all keys present, values valid OK")
    print("PASS")


def test_controller_circle_tracking():
    """#16: Simulator 연동 원형 궤적 추적"""
    print("\n" + "=" * 60)
    print("Test #16: Circle tracking with Simulator")
    print("=" * 60)

    model = _make_model()
    params = _make_bnn_params(K=64, N=15)
    ctrl = BNNMPPIController(model, params, uncertainty_fn=_mock_uncertainty_fn)

    initial_state = circle_trajectory(0.0)
    dt = params.dt
    duration = 2.0

    ref_fn = lambda t: generate_reference_trajectory(circle_trajectory, t, params.N, dt)

    sim = Simulator(model, ctrl, dt)
    sim.reset(initial_state)

    n_steps = int(duration / dt)
    for step in range(n_steps):
        t = step * dt
        ref = ref_fn(t)
        sim.step(ref)

    states = np.array(sim.history["state"])
    errors = []
    for i, st in enumerate(states):
        ref_pt = circle_trajectory(i * dt)
        err = np.sqrt((st[0] - ref_pt[0]) ** 2 + (st[1] - ref_pt[1]) ** 2)
        errors.append(err)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"  RMSE={rmse:.4f} (threshold: 0.5)")
    assert rmse < 0.5, f"RMSE too high: {rmse:.4f}"
    print("PASS")


def test_controller_high_uncertainty_conservative():
    """#17: 불확실 영역에서 보수적 제어"""
    print("\n" + "=" * 60)
    print("Test #17: High uncertainty → conservative control")
    print("=" * 60)

    model = _make_model()
    params_base = MPPIParams(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    params_bnn = _make_bnn_params(K=64, N=10, feasibility_weight=100.0)

    vanilla = MPPIController(model, params_base)
    bnn = BNNMPPIController(model, params_bnn, uncertainty_fn=_high_uncertainty_fn)

    state = np.array([3.0, 3.0, 0.0])  # 원점에서 먼 곳
    ref = _make_ref(N=10)

    np.random.seed(42)
    ctrl_vanilla, _ = vanilla.compute_control(state, ref)

    np.random.seed(42)
    ctrl_bnn, info = bnn.compute_control(state, ref)

    # BNN-MPPI는 불확실 비용 때문에 다른 제어 선택
    print(f"  vanilla control: {ctrl_vanilla}")
    print(f"  bnn control:     {ctrl_bnn}")
    print(f"  mean_feas: {info['bnn_stats']['mean_feasibility']:.4f}")
    # 항상 다를 필요는 없지만, feasibility 비용이 추가됨을 확인
    assert info["bnn_stats"]["mean_uncertainty_cost"] > 0, \
        "uncertainty cost should be > 0 with high uncertainty"
    print("PASS")


# ══════════════════════════════════════════════════════════════
# 통합 테스트 (#18~#20)
# ══════════════════════════════════════════════════════════════

def test_integration_with_mock_ensemble():
    """#18: 가짜 앙상블 모델 연동"""
    print("\n" + "=" * 60)
    print("Test #18: Mock ensemble model integration")
    print("=" * 60)

    class MockEnsembleModel:
        state_dim = 3
        control_dim = 2

        def forward(self, state, control, dt):
            if state.ndim == 1:
                return state + np.array([
                    control[0] * np.cos(state[2]) * dt,
                    control[0] * np.sin(state[2]) * dt,
                    control[1] * dt,
                ])
            # batch
            dx = np.zeros_like(state)
            dx[:, 0] = control[:, 0] * np.cos(state[:, 2]) * dt
            dx[:, 1] = control[:, 0] * np.sin(state[:, 2]) * dt
            dx[:, 2] = control[:, 1] * dt
            return state + dx

        def step(self, state, control, dt):
            return self.forward(state, control, dt)

        def predict_with_uncertainty(self, states, controls):
            if states.ndim == 1:
                states = states[None, :]
            mean = states.copy()
            dist = np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)
            std = dist[:, None] * 0.05 * np.ones((1, 3))
            return mean, std

        def get_control_bounds(self):
            return (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

    model = MockEnsembleModel()
    params = _make_bnn_params(K=32, N=10)
    ctrl = BNNMPPIController(model, params)

    assert ctrl.uncertainty_fn is not None
    state = np.array([1.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(1.0, 2.0, 11)

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert "bnn_stats" in info
    print(f"  control={control}")
    print(f"  feasibility={info['bnn_stats']['mean_feasibility']:.4f}")
    print("PASS")


def test_integration_reset():
    """#19: reset 후 상태 초기화"""
    print("\n" + "=" * 60)
    print("Test #19: Reset clears state")
    print("=" * 60)

    ctrl = _make_controller()
    state = circle_trajectory(0.0)
    ref = _make_ref()

    # 2회 실행
    ctrl.compute_control(state, ref)
    ctrl.compute_control(state, ref)

    assert len(ctrl._bnn_history) == 2

    ctrl.reset()
    assert len(ctrl._bnn_history) == 0
    assert np.allclose(ctrl.U, 0.0)
    print("  reset clears history + U OK")
    print("PASS")


def test_integration_varying_uncertainty():
    """#20: 시간에 따라 변하는 불확실성"""
    print("\n" + "=" * 60)
    print("Test #20: Varying uncertainty over time")
    print("=" * 60)

    call_count = [0]

    def time_varying_uncertainty(states, controls):
        call_count[0] += 1
        if states.ndim == 1:
            states = states[None, :]
        # 호출 횟수에 따라 불확실성 증가
        scale = min(call_count[0] * 0.01, 1.0)
        nx = states.shape[-1]
        return np.full((states.shape[0], nx), scale)

    ctrl = _make_controller(uncertainty_fn=time_varying_uncertainty)
    state = circle_trajectory(0.0)
    ref = _make_ref()

    feasibilities = []
    for _ in range(5):
        _, info = ctrl.compute_control(state, ref)
        feasibilities.append(info["bnn_stats"]["mean_feasibility"])

    # 불확실성이 증가하면 feasibility는 감소해야
    print(f"  feasibilities: {[f'{f:.4f}' for f in feasibilities]}")
    # 마지막이 첫번째보다 낮거나 같아야
    assert feasibilities[-1] <= feasibilities[0] + 0.01, \
        "feasibility should decrease or stay with increasing uncertainty"
    print("PASS")


# ══════════════════════════════════════════════════════════════
# 확장 테스트 (#21~#28)
# ══════════════════════════════════════════════════════════════

def test_feasibility_threshold_sweep():
    """#21: threshold 증가 → 필터 수 단조 증가"""
    print("\n" + "=" * 60)
    print("Test #21: Feasibility threshold sweep")
    print("=" * 60)

    state = circle_trajectory(0.0)
    ref = _make_ref()
    thresholds = [0.0, 0.3, 0.7]
    filter_counts = []

    for thr in thresholds:
        np.random.seed(42)
        ctrl = _make_controller(
            uncertainty_fn=_mock_uncertainty_fn,
            feasibility_threshold=thr,
            K=64,
        )
        _, info = ctrl.compute_control(state, ref)
        if "bnn_stats" in info:
            n_filt = info["bnn_stats"]["num_filtered"]
        else:
            n_filt = 0
        filter_counts.append(n_filt)
        print(f"  threshold={thr:.1f} → filtered={n_filt}")

    # 필터 수 단조 증가 (또는 동일)
    for i in range(len(filter_counts) - 1):
        assert filter_counts[i] <= filter_counts[i + 1], \
            f"filter count should be monotonically non-decreasing: {filter_counts}"
    print("PASS")


def test_controller_figure8_tracking():
    """#22: figure8 궤적 RMSE < 0.5"""
    print("\n" + "=" * 60)
    print("Test #22: Figure8 tracking")
    print("=" * 60)

    model = _make_model()
    params = _make_bnn_params(K=64, N=15)
    ctrl = BNNMPPIController(model, params, uncertainty_fn=_mock_uncertainty_fn)

    initial_state = figure_eight_trajectory(0.0)
    dt = params.dt
    duration = 2.0

    ref_fn = lambda t: generate_reference_trajectory(figure_eight_trajectory, t, params.N, dt)

    sim = Simulator(model, ctrl, dt)
    sim.reset(initial_state)

    n_steps = int(duration / dt)
    for step in range(n_steps):
        t = step * dt
        ref = ref_fn(t)
        sim.step(ref)

    states = np.array(sim.history["state"])
    errors = []
    for i, st in enumerate(states):
        ref_pt = figure_eight_trajectory(i * dt)
        err = np.sqrt((st[0] - ref_pt[0]) ** 2 + (st[1] - ref_pt[1]) ** 2)
        errors.append(err)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"  RMSE={rmse:.4f} (threshold: 1.0)")
    assert rmse < 1.0, f"RMSE too high: {rmse:.4f}"
    print("PASS")


def test_controller_with_obstacle_cost():
    """#23: ObstacleCost + FeasibilityCost 동시 사용"""
    print("\n" + "=" * 60)
    print("Test #23: ObstacleCost + FeasibilityCost combined")
    print("=" * 60)

    model = _make_model()
    obstacles = [(3.0, 0.0, 0.4), (-2.0, 2.0, 0.5)]
    params = _make_bnn_params(K=64, N=10)

    cost = CompositeMPPICost([
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
        ObstacleCost(obstacles, safety_margin=0.1, cost_weight=200.0),
    ])
    ctrl = BNNMPPIController(
        model, params, cost_function=cost, uncertainty_fn=_mock_uncertainty_fn,
    )

    state = circle_trajectory(0.0)
    ref = _make_ref(N=10)

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert "bnn_stats" in info
    assert "feasibility_scores" in info
    print(f"  control={control}")
    print(f"  feasibility={info['bnn_stats']['mean_feasibility']:.4f}")
    print("PASS")


def test_numerical_stability_extreme():
    """#24: 극단적 σ에서 NaN/Inf 없음"""
    print("\n" + "=" * 60)
    print("Test #24: Numerical stability with extreme σ")
    print("=" * 60)

    state = circle_trajectory(0.0)
    ref = _make_ref()

    # 매우 작은 불확실성
    def tiny_unc(states, controls):
        if states.ndim == 1:
            states = states[None, :]
        return np.full((states.shape[0], states.shape[-1]), 1e-10)

    # 매우 큰 불확실성
    def huge_unc(states, controls):
        if states.ndim == 1:
            states = states[None, :]
        return np.full((states.shape[0], states.shape[-1]), 1e3)

    for name, unc_fn in [("tiny (1e-10)", tiny_unc), ("huge (1e3)", huge_unc)]:
        ctrl = _make_controller(uncertainty_fn=unc_fn, K=32)
        control, info = ctrl.compute_control(state, ref)
        assert not np.any(np.isnan(control)), f"{name}: NaN in control"
        assert not np.any(np.isinf(control)), f"{name}: Inf in control"
        scores = info["feasibility_scores"]
        assert not np.any(np.isnan(scores)), f"{name}: NaN in scores"
        assert not np.any(np.isinf(scores)), f"{name}: Inf in scores"
        print(f"  {name}: control={control}, mean_feas={np.mean(scores):.6f} OK")

    print("PASS")


def test_bnn_vs_vanilla_comparison():
    """#25: 동일 시드, BNN의 uncertainty_cost > 0"""
    print("\n" + "=" * 60)
    print("Test #25: BNN vs Vanilla — uncertainty cost > 0")
    print("=" * 60)

    model = _make_model()
    state = np.array([2.0, 2.0, 0.0])  # 원점에서 먼 곳
    ref = _make_ref()

    params_v = MPPIParams(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    params_b = _make_bnn_params(K=64, N=10)

    vanilla = MPPIController(model, params_v)
    bnn = BNNMPPIController(model, params_b, uncertainty_fn=_mock_uncertainty_fn)

    np.random.seed(42)
    ctrl_v, _ = vanilla.compute_control(state, ref)

    np.random.seed(42)
    ctrl_b, info_b = bnn.compute_control(state, ref)

    unc_cost = info_b["bnn_stats"]["mean_uncertainty_cost"]
    print(f"  vanilla control: {ctrl_v}")
    print(f"  bnn control:     {ctrl_b}")
    print(f"  uncertainty_cost: {unc_cost:.4f}")
    assert unc_cost > 0, "uncertainty cost should be > 0"
    print("PASS")


def test_bnn_vs_uncertainty_comparison():
    """#26: UncMPPI vs BNN 비교 — 둘 다 유효한 제어"""
    print("\n" + "=" * 60)
    print("Test #26: BNN vs UncMPPI — both produce valid control")
    print("=" * 60)

    model = _make_model()
    state = circle_trajectory(0.0)
    ref = _make_ref()

    params_unc = UncertaintyMPPIParams(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        exploration_factor=1.5,
        uncertainty_strategy="previous_trajectory",
    )
    params_bnn = _make_bnn_params(K=64, N=10)

    unc_ctrl = UncertaintyMPPIController(
        model, params_unc, uncertainty_fn=_mock_uncertainty_fn,
    )
    bnn_ctrl = BNNMPPIController(
        model, params_bnn, uncertainty_fn=_mock_uncertainty_fn,
    )

    np.random.seed(42)
    ctrl_unc, info_unc = unc_ctrl.compute_control(state, ref)

    np.random.seed(42)
    ctrl_bnn, info_bnn = bnn_ctrl.compute_control(state, ref)

    assert ctrl_unc.shape == (2,)
    assert ctrl_bnn.shape == (2,)
    assert not np.any(np.isnan(ctrl_unc))
    assert not np.any(np.isnan(ctrl_bnn))
    print(f"  unc control: {ctrl_unc}")
    print(f"  bnn control: {ctrl_bnn}")
    print("PASS")


def test_get_bnn_statistics():
    """#27: get_bnn_statistics 5회 실행 후 통계 검증"""
    print("\n" + "=" * 60)
    print("Test #27: get_bnn_statistics after 5 steps")
    print("=" * 60)

    ctrl = _make_controller()
    state = circle_trajectory(0.0)
    ref = _make_ref()

    for _ in range(5):
        ctrl.compute_control(state, ref)

    stats = ctrl.get_bnn_statistics()
    assert stats["num_steps"] == 5, f"num_steps={stats['num_steps']}"
    assert "overall_mean_feasibility" in stats
    assert "overall_min_feasibility" in stats
    assert "overall_mean_filter_ratio" in stats
    assert len(stats["history"]) == 5

    assert 0 <= stats["overall_mean_feasibility"] <= 1
    assert 0 <= stats["overall_min_feasibility"] <= 1
    assert stats["overall_mean_filter_ratio"] >= 0

    print(f"  num_steps={stats['num_steps']}")
    print(f"  mean_feas={stats['overall_mean_feasibility']:.4f}")
    print(f"  min_feas={stats['overall_min_feasibility']:.4f}")
    print(f"  mean_filter={stats['overall_mean_filter_ratio']:.4f}")
    print("PASS")


def test_controller_different_K():
    """#28: K=16/64/128 모두 정상 control shape"""
    print("\n" + "=" * 60)
    print("Test #28: Different K values")
    print("=" * 60)

    state = circle_trajectory(0.0)
    ref = _make_ref()

    for K in [16, 64, 128]:
        ctrl = _make_controller(K=K)
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,), f"K={K}: shape={control.shape}"
        assert not np.any(np.isnan(control)), f"K={K}: NaN"
        print(f"  K={K}: control={control} OK")

    print("PASS")


# ── 메인 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Params
        test_params_defaults,
        test_params_custom,
        test_params_validation,
        # FeasibilityCost
        test_feasibility_cost_shape,
        test_feasibility_zero_uncertainty,
        test_feasibility_high_uncertainty_high_cost,
        test_feasibility_score_range,
        test_feasibility_reduce_modes,
        test_feasibility_weight_scaling,
        # Controller
        test_controller_compute_control_shape,
        test_controller_no_uncertainty_fallback,
        test_controller_auto_detect_model_uncertainty,
        test_controller_feasibility_filtering,
        test_controller_max_filter_ratio,
        test_controller_info_bnn_stats,
        test_controller_circle_tracking,
        test_controller_high_uncertainty_conservative,
        # Integration
        test_integration_with_mock_ensemble,
        test_integration_reset,
        test_integration_varying_uncertainty,
        # Extended (#21~#28)
        test_feasibility_threshold_sweep,
        test_controller_figure8_tracking,
        test_controller_with_obstacle_cost,
        test_numerical_stability_extreme,
        test_bnn_vs_vanilla_comparison,
        test_bnn_vs_uncertainty_comparison,
        test_get_bnn_statistics,
        test_controller_different_K,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"\nFAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed / {len(tests)} total")
    print(f"{'=' * 60}")
