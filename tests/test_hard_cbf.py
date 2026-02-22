"""HardCBFCost 테스트"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.controllers.mppi.hard_cbf_cost import HardCBFCost

def test_hard_cbf_no_violation():
    """안전 궤적은 비용 0"""
    print("\n" + "=" * 60)
    print("Test 1: HardCBF - No Violation")
    print("=" * 60)
    cost_fn = HardCBFCost(obstacles=[(10.0, 10.0, 0.5)])
    K, N = 32, 10
    traj = np.zeros((K, N+1, 3))
    traj[:, :, 0] = np.linspace(0, 1, N+1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N+1, 3))
    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,)
    assert np.allclose(costs, 0.0)
    print("✓ PASS\n")

def test_hard_cbf_rejection():
    """위반 궤적은 rejection_cost"""
    print("=" * 60)
    print("Test 2: HardCBF - Rejection Cost")
    print("=" * 60)
    cost_fn = HardCBFCost(obstacles=[(2.0, 0.0, 0.5)], rejection_cost=1e6)
    K, N = 32, 10
    traj = np.zeros((K, N+1, 3))
    traj[:, :, 0] = np.linspace(0, 4, N+1)  # 장애물 관통
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N+1, 3))
    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.all(costs == 1e6), f"Expected 1e6, got {costs[0]}"
    print(f"  All costs = {costs[0]:.0f} (rejection)")
    print("✓ PASS\n")

def test_hard_cbf_mixed():
    """일부 안전 / 일부 위반 혼합"""
    print("=" * 60)
    print("Test 3: HardCBF - Mixed Safe/Unsafe")
    print("=" * 60)
    cost_fn = HardCBFCost(obstacles=[(2.0, 0.0, 0.5)], rejection_cost=1e6, safety_margin=0.05)
    K, N = 64, 10
    traj = np.zeros((K, N+1, 3))
    # half go through obstacle, half stay safe
    traj[:32, :, 0] = np.linspace(0, 4, N+1)
    traj[32:, :, 0] = np.linspace(0, 0.5, N+1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N+1, 3))
    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert np.all(costs[:32] == 1e6)
    assert np.all(costs[32:] == 0.0)
    print(f"  Unsafe half: {costs[0]:.0f}, Safe half: {costs[32]:.0f}")
    print("✓ PASS\n")

def test_hard_cbf_barrier_info():
    """barrier info 확인"""
    print("=" * 60)
    print("Test 4: HardCBF - Barrier Info")
    print("=" * 60)
    cost_fn = HardCBFCost(obstacles=[(2.0, 0.0, 0.5)])
    traj_safe = np.zeros((11, 3))
    info = cost_fn.get_barrier_info(traj_safe)
    assert "min_barrier" in info
    assert "is_safe" in info
    print(f"  min_barrier={info['min_barrier']:.4f}")
    # empty obstacles
    cost_fn2 = HardCBFCost(obstacles=[])
    info2 = cost_fn2.get_barrier_info(traj_safe)
    assert info2["is_safe"] == True
    print("✓ PASS\n")

def test_hard_cbf_repr():
    """repr and update_obstacles"""
    print("=" * 60)
    print("Test 5: HardCBF - Repr and Update")
    print("=" * 60)
    cost_fn = HardCBFCost(obstacles=[(1.0, 0.0, 0.3)])
    print(f"  {cost_fn}")
    cost_fn.update_obstacles([(5.0, 5.0, 1.0)])
    assert cost_fn.obstacles[0] == (5.0, 5.0, 1.0)
    print("✓ PASS\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HardCBFCost Tests".center(60))
    print("=" * 60)
    try:
        test_hard_cbf_no_violation()
        test_hard_cbf_rejection()
        test_hard_cbf_mixed()
        test_hard_cbf_barrier_info()
        test_hard_cbf_repr()
        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
