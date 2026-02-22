"""CBFGuidedSamplingMPPI 테스트"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.controllers.mppi.cbf_guided_sampling_mppi import (
    CBFGuidedSamplingMPPIController, CBFGuidedSamplingParams,
)

def _make_controller(obstacles=None):
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    if obstacles is None:
        obstacles = [(3.0, 0.0, 0.5)]
    params = CBFGuidedSamplingParams(
        N=10, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        cbf_obstacles=obstacles,
        cbf_weight=1000.0, cbf_alpha=0.1,
        cbf_safety_margin=0.1,
        rejection_ratio=0.3,
        gradient_bias_weight=0.1,
        max_resample_iters=3,
    )
    return CBFGuidedSamplingMPPIController(model, params), model, params

def test_guided_basic():
    """기본 동작"""
    print("\n" + "=" * 60)
    print("Test 1: CBFGuided - Basic Control")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert u.shape == (2,)
    assert not np.any(np.isnan(u))
    assert "resample_stats" in info
    print(f"  u={u}, resampled={info['resample_stats']['total_resampled']}")
    print("✓ PASS\n")

def test_guided_no_obstacles():
    """장애물 없으면 리샘플 없음"""
    print("=" * 60)
    print("Test 2: CBFGuided - No Obstacles")
    print("=" * 60)
    ctrl, model, params = _make_controller(obstacles=[])
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert info['resample_stats']['total_resampled'] == 0
    print("✓ PASS\n")

def test_guided_violation_detection():
    """위반 궤적 탐지"""
    print("=" * 60)
    print("Test 3: CBFGuided - Violation Detection")
    print("=" * 60)
    ctrl, _, _ = _make_controller(obstacles=[(2.0, 0.0, 0.5)])
    K, N = 32, 10
    traj = np.zeros((K, N+1, 3))
    traj[:, :, 0] = np.linspace(0, 4, N+1)  # 관통
    violated = ctrl._identify_violations(traj)
    assert np.all(violated), "All should violate"
    traj_safe = np.zeros((K, N+1, 3))
    traj_safe[:, :, 0] = np.linspace(0, 0.5, N+1)
    violated_safe = ctrl._identify_violations(traj_safe)
    assert not np.any(violated_safe), "None should violate"
    print("✓ PASS\n")

def test_guided_resample():
    """리샘플 동작"""
    print("=" * 60)
    print("Test 4: CBFGuided - Resample")
    print("=" * 60)
    ctrl, model, params = _make_controller(obstacles=[(1.5, 0.0, 0.3)])
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 3, 11)
    u, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(u))
    print(f"  Resampled: {info['resample_stats']['total_resampled']}")
    print(f"  Safe ratio: {info['resample_stats']['safe_ratio']:.2%}")
    print("✓ PASS\n")

def test_guided_multistep():
    """여러 스텝"""
    print("=" * 60)
    print("Test 5: CBFGuided - Multi-step")
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

def test_guided_statistics():
    """통계"""
    print("=" * 60)
    print("Test 6: CBFGuided - Statistics")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    for _ in range(3):
        ctrl.compute_control(state, ref)
    stats = ctrl.get_resample_statistics()
    assert "mean_safe_ratio" in stats
    cbf_stats = ctrl.get_cbf_statistics()
    assert cbf_stats["safety_rate"] >= 0
    print(f"  Resample: {stats}")
    print(f"  CBF: {cbf_stats}")
    print("✓ PASS\n")

def test_guided_reset():
    """리셋"""
    print("=" * 60)
    print("Test 7: CBFGuided - Reset")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    ctrl.compute_control(state, ref)
    ctrl.reset()
    stats = ctrl.get_resample_statistics()
    assert stats["mean_safe_ratio"] == 1.0
    print("✓ PASS\n")

def test_guided_repr():
    """repr"""
    print("=" * 60)
    print("Test 8: CBFGuided - Repr")
    print("=" * 60)
    ctrl, _, _ = _make_controller()
    print(f"  {ctrl}")
    assert "CBFGuidedSampling" in repr(ctrl)
    print("✓ PASS\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CBFGuidedSamplingMPPI Tests".center(60))
    print("=" * 60)
    try:
        test_guided_basic()
        test_guided_no_obstacles()
        test_guided_violation_detection()
        test_guided_resample()
        test_guided_multistep()
        test_guided_statistics()
        test_guided_reset()
        test_guided_repr()
        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
