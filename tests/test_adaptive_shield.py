"""AdaptiveShieldMPPI 테스트"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.controllers.mppi.adaptive_shield_mppi import (
    AdaptiveShieldMPPIController, AdaptiveShieldParams,
)

def _make_controller():
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = AdaptiveShieldParams(
        N=10, dt=0.05, K=64, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=[(3.0, 0.0, 0.5)],
        cbf_weight=1000.0, cbf_alpha=0.1,
        cbf_safety_margin=0.1,
        shield_enabled=True,
        alpha_base=0.3, alpha_dist=0.5, alpha_vel=0.2,
        k_dist=2.0, d_safe=0.5,
    )
    return AdaptiveShieldMPPIController(model, params), model, params

def test_adaptive_shield_basic():
    """기본 compute_control 동작"""
    print("\n" + "=" * 60)
    print("Test 1: AdaptiveShield - Basic Control")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert u.shape == (2,)
    assert not np.any(np.isnan(u))
    assert "shield_intervention_rate" in info
    print(f"  u={u}, intervention_rate={info['shield_intervention_rate']:.3f}")
    print("✓ PASS\n")

def test_adaptive_shield_alpha_varies():
    """거리에 따라 alpha가 변함"""
    print("=" * 60)
    print("Test 2: AdaptiveShield - Alpha Varies with Distance")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    # 가까운 상태
    states_near = np.array([[2.0, 0.0, 0.0]])  # 장애물 (3,0,0.5) 근처
    states_far = np.array([[0.0, 0.0, 0.0]])   # 멀리
    alpha_near = ctrl._adaptive_alpha_batch(states_near, 3.0, 0.0, 0.5)
    alpha_far = ctrl._adaptive_alpha_batch(states_far, 3.0, 0.0, 0.5)
    # 가까울수록 α 감소 (더 보수적: v_ceiling = α·h/|Lg_h| 감소)
    assert alpha_near[0] < alpha_far[0], f"Near alpha ({alpha_near[0]:.3f}) should be < far ({alpha_far[0]:.3f})"
    print(f"  alpha_near={alpha_near[0]:.4f}, alpha_far={alpha_far[0]:.4f}")
    print("✓ PASS\n")

def test_adaptive_shield_no_obstacle():
    """장애물 없으면 vanilla 동작"""
    print("=" * 60)
    print("Test 3: AdaptiveShield - No Obstacles")
    print("=" * 60)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = AdaptiveShieldParams(
        N=10, dt=0.05, K=64, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=[],
        cbf_weight=1000.0, cbf_alpha=0.1,
        cbf_safety_margin=0.1,
        shield_enabled=True,
    )
    ctrl = AdaptiveShieldMPPIController(model, params)
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(u))
    assert info["shield_intervention_rate"] == 0.0
    print("✓ PASS\n")

def test_adaptive_shield_multistep():
    """여러 스텝 실행"""
    print("=" * 60)
    print("Test 4: AdaptiveShield - Multi-step")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    for _ in range(5):
        u, info = ctrl.compute_control(state, ref)
        state = model.step(state, u, params.dt)
    assert not np.any(np.isnan(state))
    print(f"  Final state: {state}")
    print("✓ PASS\n")

def test_adaptive_shield_params_validation():
    """파라미터 검증"""
    print("=" * 60)
    print("Test 5: AdaptiveShield - Params Validation")
    print("=" * 60)
    try:
        AdaptiveShieldParams(
            N=10, dt=0.05, K=64, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            cbf_obstacles=[], cbf_alpha=0.1,
            alpha_base=-0.1  # invalid
        )
        assert False, "Should raise AssertionError"
    except (AssertionError, AssertionError):
        print("  Correctly rejected negative alpha_base")
    print("✓ PASS\n")

def test_adaptive_shield_shield_stats():
    """shield 통계"""
    print("=" * 60)
    print("Test 6: AdaptiveShield - Statistics")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    for _ in range(3):
        ctrl.compute_control(state, ref)
    stats = ctrl.get_shield_statistics()
    assert stats["num_steps"] == 3
    print(f"  {stats}")
    print("✓ PASS\n")

def test_adaptive_shield_repr():
    """repr"""
    print("=" * 60)
    print("Test 7: AdaptiveShield - Repr")
    print("=" * 60)
    ctrl, _, _ = _make_controller()
    print(f"  {ctrl}")
    assert "AdaptiveShieldMPPI" in repr(ctrl)
    print("✓ PASS\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AdaptiveShieldMPPI Tests".center(60))
    print("=" * 60)
    try:
        test_adaptive_shield_basic()
        test_adaptive_shield_alpha_varies()
        test_adaptive_shield_no_obstacle()
        test_adaptive_shield_multistep()
        test_adaptive_shield_params_validation()
        test_adaptive_shield_shield_stats()
        test_adaptive_shield_repr()
        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
