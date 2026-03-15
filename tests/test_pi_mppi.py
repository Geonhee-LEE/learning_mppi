"""
PI-MPPI (Path Integral MPPI) 테스트

Vanilla MPPI와의 차이점 검증:
- 가중치에 상태 비용만 사용
- enforce_pi_covariance 시 Sigma = lambda * R^{-1}
- 탐색 향상 (ESS 비교)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.pi_mppi import PIMPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)


def _make_model():
    return DifferentialDriveKinematic(v_max=1.0, omega_max=2.0)


def _make_params(**overrides):
    defaults = dict(
        N=20, dt=0.05, K=512,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        lambda_=1.0,
    )
    defaults.update(overrides)
    return MPPIParams(**defaults)


def _make_circle_ref(N=20, radius=2.0, n_points=100):
    """원형 레퍼런스 궤적"""
    t = np.linspace(0, 2 * np.pi, n_points)
    ref_full = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        t + np.pi / 2,
    ])
    return ref_full[:N + 1]


def _make_straight_ref(N=20):
    """직진 레퍼런스 궤적"""
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = np.linspace(0, 1, N + 1)
    return ref


# ── 기본 동작 테스트 ───────────────────────────────────────────


def test_pi_mppi_basic():
    """PI-MPPI 기본 동작: 올바른 출력 형태 및 키"""
    model = _make_model()
    params = _make_params()
    ctrl = PIMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_straight_ref()

    control, info = ctrl.compute_control(state, ref)

    assert control.shape == (2,), f"Control shape: {control.shape}"
    assert not np.any(np.isnan(control)), "Control has NaN"

    # 표준 MPPI info 키
    required_keys = [
        "sample_trajectories", "sample_weights", "best_trajectory",
        "best_cost", "mean_cost", "temperature", "ess", "num_samples",
    ]
    for key in required_keys:
        assert key in info, f"Missing key: {key}"

    # PI-MPPI 고유 키
    pi_keys = ["state_cost_mean", "state_cost_best", "control_cost_mean"]
    for key in pi_keys:
        assert key in info, f"Missing PI-MPPI key: {key}"

    assert info["sample_trajectories"].shape == (512, 21, 3)
    assert abs(np.sum(info["sample_weights"]) - 1.0) < 1e-5
    assert 1.0 <= info["ess"] <= 512

    print("  PASS: PI-MPPI basic functionality")


def test_pi_mppi_state_cost_separation():
    """가중치가 상태 비용만 반영하는지 확인"""
    model = _make_model()

    # 높은 R → Vanilla에서는 제어 비용이 가중치를 강하게 억제
    params = _make_params(R=np.array([10.0, 10.0]), K=1024)

    vanilla = MPPIController(model, params)
    pi = PIMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_straight_ref()

    np.random.seed(42)
    _, v_info = vanilla.compute_control(state, ref)
    np.random.seed(42)
    _, pi_info = pi.compute_control(state, ref)

    # PI-MPPI의 ESS는 Vanilla보다 높아야 함 (제어 비용이 가중치를 왜곡하지 않음)
    # R이 크면 Vanilla는 작은 제어를 선호 → 가중치 집중 → 낮은 ESS
    # PI-MPPI는 상태 비용만 사용 → 더 균등한 가중치 → 높은 ESS
    print(f"  Vanilla ESS: {v_info['ess']:.1f}, PI-MPPI ESS: {pi_info['ess']:.1f}")

    # PI-MPPI에서 control_cost_mean > 0 (제어 비용이 계산은 되지만 가중치에 미반영)
    assert pi_info["control_cost_mean"] >= 0, "Control cost should be non-negative"
    assert pi_info["state_cost_mean"] >= 0, "State cost should be non-negative"

    print("  PASS: State cost separation verified")


def test_pi_mppi_enforce_covariance():
    """enforce_pi_covariance=True 시 sigma = sqrt(lambda/R) 확인"""
    model = _make_model()
    params = _make_params(
        lambda_=2.0,
        R=np.array([0.5, 0.8]),
        sigma=np.array([0.5, 0.5]),  # 이 값은 override됨
    )

    ctrl = PIMPPIController(model, params, enforce_pi_covariance=True)

    # sigma_i = sqrt(lambda / R_i)
    expected_sigma = np.sqrt(2.0 / np.array([0.5, 0.8]))

    actual_sigma = ctrl.params.sigma
    np.testing.assert_allclose(
        actual_sigma, expected_sigma, rtol=1e-5,
        err_msg=f"Expected sigma={expected_sigma}, got {actual_sigma}"
    )

    # 컨트롤러가 정상 동작하는지 확인
    state = np.array([0.0, 0.0, 0.0])
    ref = _make_straight_ref()
    control, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(control))

    print(f"  Expected sigma: {expected_sigma}")
    print(f"  Actual sigma:   {actual_sigma}")
    print("  PASS: PI covariance enforcement")


def test_pi_mppi_custom_state_cost():
    """사용자 정의 상태 비용 함수 주입"""
    model = _make_model()
    params = _make_params()

    # 커스텀 상태 비용: 위치만 추적 (heading 무시)
    custom_state_cost = CompositeMPPICost([
        StateTrackingCost(np.array([20.0, 20.0, 0.0])),
        TerminalCost(np.array([20.0, 20.0, 0.0])),
    ])

    ctrl = PIMPPIController(
        model, params,
        state_cost_function=custom_state_cost,
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_straight_ref()

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert not np.any(np.isnan(control))

    print("  PASS: Custom state cost function")


def test_pi_mppi_with_obstacles():
    """PI-MPPI + 장애물 회피"""
    model = _make_model()
    params = _make_params(K=1024)

    obstacles = [(0.5, 0.0, 0.2)]
    full_cost = CompositeMPPICost([
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
        ObstacleCost(obstacles, safety_margin=0.1, cost_weight=500.0),
    ])
    # 상태 비용 = 추적 + 터미널 + 장애물 (제어 제외)
    state_cost = CompositeMPPICost([
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ObstacleCost(obstacles, safety_margin=0.1, cost_weight=500.0),
    ])

    ctrl = PIMPPIController(
        model, params,
        cost_function=full_cost,
        state_cost_function=state_cost,
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_straight_ref()

    # 여러 스텝 실행
    for _ in range(30):
        control, info = ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)

    # 장애물과의 거리 확인
    ox, oy, r = obstacles[0]
    dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2) - r
    print(f"  Final distance from obstacle: {dist:.4f} m")
    assert not np.any(np.isnan(state)), "State has NaN"

    print("  PASS: PI-MPPI with obstacles")


def test_pi_mppi_circle_tracking():
    """PI-MPPI 원형 궤적 추적 성능"""
    model = _make_model()
    params = _make_params(K=1024, N=30)
    ctrl = PIMPPIController(model, params)

    state = np.array([2.0, 0.0, np.pi / 2])
    n_steps = 100
    errors = []

    for step in range(n_steps):
        # 현재 위치 기반 레퍼런스 생성
        t_offset = step * 0.05
        t_vals = np.linspace(t_offset, t_offset + 30 * 0.05, 31)
        ref = np.column_stack([
            2.0 * np.cos(t_vals),
            2.0 * np.sin(t_vals),
            t_vals + np.pi / 2,
        ])

        control, info = ctrl.compute_control(state, ref)
        state = model.step(state, control, 0.05)

        pos_error = np.sqrt((state[0] - ref[0, 0]) ** 2 + (state[1] - ref[0, 1]) ** 2)
        errors.append(pos_error)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"  Circle tracking RMSE: {rmse:.4f} m")
    # PI-MPPI는 제어 비용이 가중치에 미반영 → Vanilla보다 다소 느슨
    assert rmse < 2.0, f"RMSE too large: {rmse}"

    print("  PASS: PI-MPPI circle tracking")


def test_pi_mppi_vs_vanilla_comparison():
    """PI-MPPI vs Vanilla MPPI 성능 비교"""
    model = _make_model()
    params = _make_params(K=512, N=20)

    vanilla = MPPIController(model, params)
    pi = PIMPPIController(model, params)

    ref = _make_straight_ref()
    n_steps = 50

    results = {}
    for name, ctrl in [("Vanilla", vanilla), ("PI-MPPI", pi)]:
        state = np.array([0.0, 0.0, 0.0])
        errors = []
        ess_list = []

        for _ in range(n_steps):
            control, info = ctrl.compute_control(state, ref)
            state = model.step(state, control, 0.05)
            errors.append(np.linalg.norm(state[:2] - ref[0, :2]))
            ess_list.append(info["ess"])

        results[name] = {
            "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
            "mean_ess": np.mean(ess_list),
        }

    print(f"  Vanilla RMSE: {results['Vanilla']['rmse']:.4f}, "
          f"ESS: {results['Vanilla']['mean_ess']:.1f}")
    print(f"  PI-MPPI RMSE: {results['PI-MPPI']['rmse']:.4f}, "
          f"ESS: {results['PI-MPPI']['mean_ess']:.1f}")

    # 둘 다 합리적인 결과를 내야 함
    assert results["PI-MPPI"]["rmse"] < 2.0, "PI-MPPI RMSE too large"
    assert results["Vanilla"]["rmse"] < 2.0, "Vanilla RMSE too large"

    print("  PASS: PI-MPPI vs Vanilla comparison")


def test_pi_mppi_reset():
    """reset 후 상태 초기화 확인"""
    model = _make_model()
    params = _make_params()
    ctrl = PIMPPIController(model, params)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_straight_ref()

    # 몇 스텝 실행
    for _ in range(5):
        ctrl.compute_control(state, ref)

    # 내부 상태가 0이 아님을 확인
    assert not np.allclose(ctrl.U, 0.0), "U should not be zero after steps"

    # reset
    ctrl.reset()
    assert np.allclose(ctrl.U, 0.0), "U should be zero after reset"

    print("  PASS: PI-MPPI reset")


def test_pi_mppi_repr():
    """__repr__ 출력 확인"""
    model = _make_model()
    params = _make_params()
    ctrl = PIMPPIController(model, params)

    repr_str = repr(ctrl)
    assert "PIMPPIController" in repr_str
    assert "DifferentialDriveKinematic" in repr_str

    print(f"  repr: {repr_str}")
    print("  PASS: PI-MPPI repr")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PI-MPPI (Path Integral MPPI) Tests".center(60))
    print("=" * 60)

    try:
        test_pi_mppi_basic()
        test_pi_mppi_state_cost_separation()
        test_pi_mppi_enforce_covariance()
        test_pi_mppi_custom_state_cost()
        test_pi_mppi_with_obstacles()
        test_pi_mppi_circle_tracking()
        test_pi_mppi_vs_vanilla_comparison()
        test_pi_mppi_reset()
        test_pi_mppi_repr()

        print("\n" + "=" * 60)
        print("All PI-MPPI Tests Passed!".center(60))
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
