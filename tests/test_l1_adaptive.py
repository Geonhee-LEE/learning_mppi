"""
L1 Adaptive Dynamics 테스트

L1AdaptiveDynamics의 외란 추정, 보정, 안정성, 적응을 검증.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ==================== 테스트 ====================


def test_creation():
    """L1AdaptiveDynamics 생성 및 기본 속성."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)
    assert l1.state_dim == 5
    assert l1.control_dim == 2
    assert l1.model_type == "learned"
    dist = l1.get_disturbance_estimate()
    assert np.allclose(dist["sigma_filtered"], 0.0)
    print("  PASS: test_creation")


def test_forward_dynamics_single():
    """단일 상태 forward dynamics (보정 전)."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

    l1 = L1AdaptiveDynamics()
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    dot = l1.forward_dynamics(state, control)
    assert dot.shape == (5,)
    assert not np.any(np.isnan(dot))
    print("  PASS: test_forward_dynamics_single")


def test_forward_dynamics_batch():
    """배치 forward dynamics."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

    l1 = L1AdaptiveDynamics()
    states = np.random.randn(10, 5)
    controls = np.random.randn(10, 2)
    dots = l1.forward_dynamics(states, controls)
    assert dots.shape == (10, 5)
    assert not np.any(np.isnan(dots))
    print("  PASS: test_forward_dynamics_batch")


def test_update_step():
    """단일 update_step이 오류 없이 실행."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

    l1 = L1AdaptiveDynamics()
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    next_state = np.array([0.025, 0.0, 0.005, 0.6, 0.12])
    l1.update_step(state, control, next_state, dt=0.05)

    dist = l1.get_disturbance_estimate()
    assert dist["sigma_norm"] >= 0.0
    print("  PASS: test_update_step")


def test_sigma_convergence():
    """외란 추정이 실제 모델 미스매치를 추적."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)
    dt = 0.05

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)

    for _ in range(200):
        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.3, 0.3)])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        l1.update_step(state, control, next_state, dt)
        state = next_state

    dist = l1.get_disturbance_estimate()
    assert dist["sigma_norm"] > 0.0, "Sigma should be non-zero for mismatched model"
    print(f"  PASS: test_sigma_convergence (|σ_f|={dist['sigma_norm']:.4f})")


def test_filter_smoothness():
    """저역통과 필터가 σ_hat를 평활화."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)
    dt = 0.05

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)

    sigma_hats = []
    sigma_filtereds = []
    for _ in range(100):
        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.3, 0.3)])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        l1.update_step(state, control, next_state, dt)
        state = next_state
        dist = l1.get_disturbance_estimate()
        sigma_hats.append(dist["sigma_hat"].copy())
        sigma_filtereds.append(dist["sigma_filtered"].copy())

    # 필터링된 σ는 원시 σ_hat보다 변동이 적어야 함
    sigma_hat_var = np.var(np.array(sigma_hats), axis=0)
    sigma_f_var = np.var(np.array(sigma_filtereds), axis=0)
    # 최소 하나의 차원에서 필터링 효과 확인
    assert np.any(sigma_f_var <= sigma_hat_var + 1e-6), "Filter should smooth sigma"
    print("  PASS: test_filter_smoothness")


def test_hurwitz_stability():
    """A_m의 Hurwitz 안정성 (모든 고유값 < 0)."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

    l1 = L1AdaptiveDynamics()
    assert l1.is_stable(), "A_m should be Hurwitz stable"

    # 불안정한 A_m
    l1_unstable = L1AdaptiveDynamics(am_gains=np.array([1.0, -5.0, -5.0, -10.0, -10.0]))
    assert not l1_unstable.is_stable(), "Positive eigenvalue should be unstable"
    print("  PASS: test_hurwitz_stability")


def test_adapt_batch():
    """adapt() 배치 인터페이스."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)
    dt = 0.05

    M = 50
    states = np.zeros((M, 5))
    controls = np.zeros((M, 2))
    next_states = np.zeros((M, 5))

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(123)
    for i in range(M):
        c = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.5, 0.5)])
        dot = true_model.forward_dynamics(state, c)
        ns = state + dot * dt
        states[i] = state
        controls[i] = c
        next_states[i] = ns
        state = ns

    result = l1.adapt(states, controls, next_states, dt)
    assert isinstance(result, float)
    print("  PASS: test_adapt_batch")


def test_periodic_disturbance():
    """주기적 외란 환경에서 L1 추적."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    dt = 0.05
    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)

    errors = []
    for step in range(300):
        # 시변 마찰
        true_c_v = 0.5 + 0.3 * np.sin(step * dt * 2.0)
        true_model = DynamicKinematicAdapter(c_v=true_c_v, c_omega=0.3)

        control = np.array([np.random.uniform(0.3, 0.8), np.random.uniform(-0.2, 0.2)])
        true_dot = true_model.forward_dynamics(state, control)
        next_state = state + true_dot * dt
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))

        l1.update_step(state, control, next_state, dt)

        # L1 예측 오차
        pred_dot = l1.forward_dynamics(state, control)
        pred_next = state + pred_dot * dt
        err = np.linalg.norm(next_state[:3] - pred_next[:3])
        errors.append(err)

        state = next_state

    # 외란이 없는 경우와 비교: L1 보정된 모델이 최소한 동작해야 함
    # 에러가 매우 작으면 (< 1e-6) 이미 잘 추적 중
    mean_err = np.mean(errors)
    assert mean_err < 0.5, f"L1 prediction error too large: {mean_err:.4f}"
    print(f"  PASS: test_periodic_disturbance (mean_err={mean_err:.6f})")


def test_reset():
    """L1 리셋 동작."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)
    dt = 0.05

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)
    for _ in range(20):
        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.3, 0.3)])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        l1.update_step(state, control, next_state, dt)
        state = next_state

    dist_before = l1.get_disturbance_estimate()
    assert dist_before["sigma_norm"] > 0.0, f"sigma_norm should be > 0 after updates: {dist_before['sigma_norm']}"

    l1.reset()
    dist_after = l1.get_disturbance_estimate()
    assert np.allclose(dist_after["sigma_filtered"], 0.0)
    print("  PASS: test_reset")


def test_corrected_forward():
    """보정된 forward_dynamics가 공칭 모델보다 정확."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    nominal_model = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1)
    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)
    dt = 0.05

    # 학습 데이터
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)
    for _ in range(100):
        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.3, 0.3)])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        l1.update_step(state, control, next_state, dt)
        state = next_state

    # 평가
    test_state = np.array([1.0, 0.5, 0.3, 0.7, 0.2])
    test_control = np.array([0.8, 0.3])
    true_dot = true_model.forward_dynamics(test_state, test_control)
    nom_dot = nominal_model.forward_dynamics(test_state, test_control)
    l1_dot = l1.forward_dynamics(test_state, test_control)

    nom_err = np.linalg.norm(true_dot - nom_dot)
    l1_err = np.linalg.norm(true_dot - l1_dot)
    assert l1_err < nom_err, f"L1 should be more accurate than nominal: {l1_err:.4f} vs {nom_err:.4f}"
    print(f"  PASS: test_corrected_forward (L1={l1_err:.4f} < nom={nom_err:.4f})")


def test_repr():
    """__repr__ 출력."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    l1 = L1AdaptiveDynamics()
    s = repr(l1)
    assert "L1AdaptiveDynamics" in s
    print(f"  PASS: test_repr → {s}")


def test_get_control_bounds():
    """get_control_bounds 위임."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    l1 = L1AdaptiveDynamics()
    lower, upper = l1.get_control_bounds()
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    print("  PASS: test_get_control_bounds")


def test_state_to_dict():
    """state_to_dict 위임."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    l1 = L1AdaptiveDynamics()
    d = l1.state_to_dict(np.array([1.0, 2.0, 0.5, 0.3, 0.1]))
    assert "x" in d and "v" in d
    print("  PASS: test_state_to_dict")


def test_normalize_state():
    """normalize_state 위임."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    l1 = L1AdaptiveDynamics()
    state = np.array([0.0, 0.0, 4.0, 0.0, 0.0])
    normed = l1.normalize_state(state)
    assert abs(normed[2]) < np.pi + 0.01
    print("  PASS: test_normalize_state")


def test_custom_am_gains():
    """커스텀 A_m 게인으로 생성."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

    am_gains = np.array([-10.0, -10.0, -10.0, -20.0, -20.0])
    l1 = L1AdaptiveDynamics(am_gains=am_gains)
    assert l1.is_stable()
    assert np.allclose(np.diag(l1._A_m), am_gains)
    print("  PASS: test_custom_am_gains")


def test_multiple_update_steps():
    """여러 스텝 연속 업데이트."""
    from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    true_model = DynamicKinematicAdapter(c_v=0.5, c_omega=0.3)
    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)
    dt = 0.05

    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.random.seed(42)

    for _ in range(500):
        control = np.array([np.random.uniform(0.2, 1.0), np.random.uniform(-0.5, 0.5)])
        dot = true_model.forward_dynamics(state, control)
        next_state = state + dot * dt
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        l1.update_step(state, control, next_state, dt)
        state = next_state

    # NaN 없음
    dist = l1.get_disturbance_estimate()
    assert not np.any(np.isnan(dist["sigma_filtered"]))
    assert not np.any(np.isnan(dist["sigma_hat"]))
    print("  PASS: test_multiple_update_steps")


# ==================== 실행 ====================

if __name__ == "__main__":
    tests = [
        test_creation,
        test_forward_dynamics_single,
        test_forward_dynamics_batch,
        test_update_step,
        test_sigma_convergence,
        test_filter_smoothness,
        test_hurwitz_stability,
        test_adapt_batch,
        test_periodic_disturbance,
        test_reset,
        test_corrected_forward,
        test_repr,
        test_get_control_bounds,
        test_state_to_dict,
        test_normalize_state,
        test_custom_am_gains,
        test_multiple_update_steps,
    ]

    print(f"\n{'=' * 60}")
    print(f"  L1 Adaptive Dynamics Tests ({len(tests)} tests)")
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
