"""
EKF Adaptive Dynamics 테스트

EKFAdaptiveDynamics의 파라미터 추정, 야코비안, 수렴, 적응을 검증.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ==================== 테스트 ====================


def test_creation():
    """EKFAdaptiveDynamics 생성 및 기본 속성."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics

    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    assert ekf.state_dim == 5
    assert ekf.control_dim == 2
    assert ekf.model_type == "learned"
    est = ekf.get_parameter_estimates()
    assert abs(est["c_v"] - 0.1) < 1e-6
    assert abs(est["c_omega"] - 0.1) < 1e-6
    print("  PASS: test_creation")


def test_forward_dynamics_single():
    """단일 상태 forward dynamics."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics

    ekf = EKFAdaptiveDynamics()
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    dot = ekf.forward_dynamics(state, control)
    assert dot.shape == (5,)
    assert not np.any(np.isnan(dot))
    print("  PASS: test_forward_dynamics_single")


def test_forward_dynamics_batch():
    """배치 forward dynamics."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics

    ekf = EKFAdaptiveDynamics()
    states = np.random.randn(10, 5)
    controls = np.random.randn(10, 2)
    dots = ekf.forward_dynamics(states, controls)
    assert dots.shape == (10, 5)
    assert not np.any(np.isnan(dots))
    print("  PASS: test_forward_dynamics_batch")


def test_single_update():
    """단일 EKF 업데이트가 오류 없이 실행."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics

    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    next_state = np.array([0.025, 0.0, 0.005, 0.6, 0.12])
    ekf.update_step(state, control, next_state, dt=0.05)

    est = ekf.get_parameter_estimates()
    assert est["c_v"] >= 0.01
    assert est["c_omega"] >= 0.01
    print("  PASS: test_single_update")


def test_parameter_convergence():
    """알려진 파라미터(c_v=0.5, c_omega=0.3)로 수렴 확인."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_c_v = 0.5
    true_c_omega = 0.3
    true_model = DynamicKinematicAdapter(c_v=true_c_v, c_omega=true_c_omega)

    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    dt = 0.05

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)

    for _ in range(500):
        control = np.array([
            np.random.uniform(0.2, 1.0),
            np.random.uniform(-0.8, 0.8),
        ])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        ekf.update_step(state, control, next_state, dt)
        state = next_state

    est = ekf.get_parameter_estimates()
    # 수렴 오차: 0.2 이내
    assert abs(est["c_v"] - true_c_v) < 0.2, f"c_v={est['c_v']:.3f}, expected ~{true_c_v}"
    assert abs(est["c_omega"] - true_c_omega) < 0.2, f"c_omega={est['c_omega']:.3f}, expected ~{true_c_omega}"
    print(f"  PASS: test_parameter_convergence (ĉ_v={est['c_v']:.3f}, ĉ_ω={est['c_omega']:.3f})")


def test_adapt_batch():
    """adapt() 배치 인터페이스."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    dt = 0.05

    M = 50
    states = np.zeros((M, 5))
    controls = np.zeros((M, 2))
    next_states = np.zeros((M, 5))

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(123)
    for i in range(M):
        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.5, 0.5)])
        dot = true_model.forward_dynamics(state, control)
        ns = state + dot * dt
        states[i] = state
        controls[i] = control
        next_states[i] = ns
        state = ns

    result = ekf.adapt(states, controls, next_states, dt)
    assert isinstance(result, float)
    est = ekf.get_parameter_estimates()
    assert est["c_v"] > 0.01
    print("  PASS: test_adapt_batch")


def test_covariance_positive_definite():
    """공분산 행렬이 양정치(positive definite) 유지."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    dt = 0.05

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)

    for _ in range(100):
        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.5, 0.5)])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        ekf.update_step(state, control, next_state, dt)
        state = next_state

        P = ekf.get_covariance()
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > -1e-10), f"Non-PD covariance: min eigval = {eigvals.min()}"

    print("  PASS: test_covariance_positive_definite")


def test_parameter_clipping():
    """파라미터 범위 클리핑 확인."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics

    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    # 강제로 범위 밖 값 설정
    ekf._ekf_state[5] = -1.0
    ekf._ekf_state[6] = 5.0
    ekf._clip_parameters()
    assert ekf._ekf_state[5] >= 0.01
    assert ekf._ekf_state[6] <= 1.5
    print("  PASS: test_parameter_clipping")


def test_jacobian_analytical_vs_numerical():
    """해석적 야코비안 vs 수치적 야코비안 비교."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics

    ekf = EKFAdaptiveDynamics()
    ekf_state = np.array([1.0, 0.5, 0.3, 0.7, 0.2, 0.4, 0.25])
    control = np.array([0.8, 0.3])
    dt = 0.05

    F_analytical = ekf._jacobian_F(ekf_state, control, dt)

    # 수치적 야코비안
    eps = 1e-5
    F_numerical = np.zeros((7, 7))
    for j in range(7):
        ekf_plus = ekf_state.copy()
        ekf_minus = ekf_state.copy()
        ekf_plus[j] += eps
        ekf_minus[j] -= eps
        f_plus = ekf._f_7d(ekf_plus, control, dt)
        f_minus = ekf._f_7d(ekf_minus, control, dt)
        F_numerical[:, j] = (f_plus - f_minus) / (2 * eps)

    max_err = np.max(np.abs(F_analytical - F_numerical))
    assert max_err < 1e-4, f"Jacobian error: {max_err}"
    print(f"  PASS: test_jacobian_analytical_vs_numerical (max_err={max_err:.2e})")


def test_reset():
    """EKF 리셋 동작."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics

    ekf = EKFAdaptiveDynamics(c_v_init=0.3, c_omega_init=0.2)
    # 몇 번 업데이트 후 리셋
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    next_state = np.array([0.025, 0.0, 0.005, 0.6, 0.12])
    ekf.update_step(state, control, next_state, 0.05)

    ekf.reset(c_v_init=0.5, c_omega_init=0.4)
    est = ekf.get_parameter_estimates()
    assert abs(est["c_v"] - 0.5) < 1e-6
    assert abs(est["c_omega"] - 0.4) < 1e-6
    print("  PASS: test_reset")


def test_uncertainty_decreases():
    """업데이트 후 파라미터 불확실성이 감소."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    dt = 0.05

    est_before = ekf.get_parameter_estimates()
    std_before = est_before["c_v_std"]

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)
    for _ in range(50):
        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.5, 0.5)])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        ekf.update_step(state, control, next_state, dt)
        state = next_state

    est_after = ekf.get_parameter_estimates()
    std_after = est_after["c_v_std"]
    assert std_after < std_before, f"Uncertainty did not decrease: {std_before:.4f} → {std_after:.4f}"
    print(f"  PASS: test_uncertainty_decreases ({std_before:.4f} → {std_after:.4f})")


def test_adapt_error_decreases():
    """adapt() 후 예측 오차가 감소."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    dt = 0.05

    # 데이터 생성
    M = 100
    states = np.zeros((M, 5))
    controls = np.zeros((M, 2))
    next_states = np.zeros((M, 5))
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(99)
    for i in range(M):
        c = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.5, 0.5)])
        dot = true_model.forward_dynamics(state, c)
        ns = state + dot * dt
        states[i] = state
        controls[i] = c
        next_states[i] = ns
        state = ns

    # 적응 전 오차
    ekf_before = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    errors_before = []
    for i in range(20):
        pred = states[i] + ekf_before.forward_dynamics(states[i], controls[i]) * dt
        err = np.linalg.norm(next_states[i] - pred)
        errors_before.append(err)

    # 적응
    ekf_after = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)
    ekf_after.adapt(states[:80], controls[:80], next_states[:80], dt)

    errors_after = []
    for i in range(80, M):
        pred = states[i] + ekf_after.forward_dynamics(states[i], controls[i]) * dt
        err = np.linalg.norm(next_states[i] - pred)
        errors_after.append(err)

    mean_before = np.mean(errors_before)
    mean_after = np.mean(errors_after)
    assert mean_after < mean_before, f"Error did not decrease: {mean_before:.4f} → {mean_after:.4f}"
    print(f"  PASS: test_adapt_error_decreases ({mean_before:.4f} → {mean_after:.4f})")


def test_repr():
    """__repr__ 출력."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    ekf = EKFAdaptiveDynamics()
    s = repr(ekf)
    assert "EKFAdaptiveDynamics" in s
    assert "ĉ_v" in s
    print(f"  PASS: test_repr → {s}")


def test_step():
    """step() 메서드 동작."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    ekf = EKFAdaptiveDynamics()
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    next_state = ekf.step(state, control, 0.05)
    assert next_state.shape == (5,)
    assert not np.any(np.isnan(next_state))
    print("  PASS: test_step")


def test_get_control_bounds():
    """get_control_bounds 위임."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    ekf = EKFAdaptiveDynamics()
    lower, upper = ekf.get_control_bounds()
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    print("  PASS: test_get_control_bounds")


def test_state_to_dict():
    """state_to_dict 위임."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    ekf = EKFAdaptiveDynamics()
    d = ekf.state_to_dict(np.array([1.0, 2.0, 0.5, 0.3, 0.1]))
    assert "x" in d and "v" in d
    print("  PASS: test_state_to_dict")


def test_normalize_state():
    """normalize_state 위임."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    ekf = EKFAdaptiveDynamics()
    state = np.array([0.0, 0.0, 4.0, 0.0, 0.0])
    normed = ekf.normalize_state(state)
    assert abs(normed[2]) < np.pi + 0.01
    print("  PASS: test_normalize_state")


def test_disturbance_tracking():
    """시변 외란 환경에서 EKF 파라미터 추적."""
    from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    dt = 0.05
    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)

    # c_v가 점진적으로 변하는 환경
    for step in range(200):
        # 시간에 따라 c_v 변화
        true_c_v = 0.3 + 0.2 * np.sin(step * dt * 0.5)
        true_c_omega = 0.2
        true_model = DynamicKinematicAdapter(c_v=true_c_v, c_omega=true_c_omega)

        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.3, 0.3)])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        ekf.update_step(state, control, next_state, dt)
        state = next_state

    # EKF가 합리적인 범위 내에서 추적하는지 확인
    est = ekf.get_parameter_estimates()
    assert 0.01 < est["c_v"] < 2.0
    assert 0.01 < est["c_omega"] < 1.5
    print(f"  PASS: test_disturbance_tracking (ĉ_v={est['c_v']:.3f}, ĉ_ω={est['c_omega']:.3f})")


# ==================== 실행 ====================

if __name__ == "__main__":
    tests = [
        test_creation,
        test_forward_dynamics_single,
        test_forward_dynamics_batch,
        test_single_update,
        test_parameter_convergence,
        test_adapt_batch,
        test_covariance_positive_definite,
        test_parameter_clipping,
        test_jacobian_analytical_vs_numerical,
        test_reset,
        test_uncertainty_decreases,
        test_adapt_error_decreases,
        test_repr,
        test_step,
        test_get_control_bounds,
        test_state_to_dict,
        test_normalize_state,
        test_disturbance_tracking,
    ]

    print(f"\n{'=' * 60}")
    print(f"  EKF Adaptive Dynamics Tests ({len(tests)} tests)")
    print(f"{'=' * 60}")

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_fn.__name__} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_fn.__name__} — {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
