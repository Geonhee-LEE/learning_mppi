"""HorizonWeightedCBFCost 테스트"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.controllers.mppi.horizon_cbf_cost import HorizonWeightedCBFCost

def test_horizon_cbf_no_obstacle():
    """장애물 멀리 있을 때 비용 0"""
    print("\n" + "=" * 60)
    print("Test 1: HorizonWeightedCBF - No Obstacle Nearby")
    print("=" * 60)
    cost_fn = HorizonWeightedCBFCost(obstacles=[(10.0, 10.0, 0.5)])
    K, N = 32, 10
    traj = np.zeros((K, N+1, 3))
    traj[:, :, 0] = np.linspace(0, 1, N+1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N+1, 3))
    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,)
    assert np.allclose(costs, 0.0), f"Expected zero cost, got max={np.max(costs)}"
    print("✓ PASS\n")

def test_horizon_cbf_violation():
    """장애물 관통 시 높은 비용"""
    print("=" * 60)
    print("Test 2: HorizonWeightedCBF - Violation Cost")
    print("=" * 60)
    cost_fn = HorizonWeightedCBFCost(obstacles=[(2.0, 0.0, 0.5)], weight=100.0, cbf_alpha=0.3)
    K, N = 32, 10
    traj_unsafe = np.zeros((K, N+1, 3))
    traj_unsafe[:, :, 0] = np.linspace(0, 4, N+1)
    traj_safe = np.zeros((K, N+1, 3))
    traj_safe[:, :, 0] = np.linspace(0, 0.5, N+1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N+1, 3))
    c_unsafe = cost_fn.compute_cost(traj_unsafe, ctrl, ref)
    c_safe = cost_fn.compute_cost(traj_safe, ctrl, ref)
    assert np.mean(c_unsafe) > np.mean(c_safe)
    assert np.mean(c_unsafe) > 0
    print(f"  Unsafe: {np.mean(c_unsafe):.2f}, Safe: {np.mean(c_safe):.2f}")
    print("✓ PASS\n")

def test_horizon_cbf_discount():
    """γ < 1 이 γ = 1 보다 가까운 미래 위반에 민감"""
    print("=" * 60)
    print("Test 3: HorizonWeightedCBF - Discount Effect")
    print("=" * 60)
    obs = [(2.0, 0.0, 0.5)]
    cost_disc = HorizonWeightedCBFCost(obstacles=obs, discount_gamma=0.5, weight=100.0)
    cost_full = HorizonWeightedCBFCost(obstacles=obs, discount_gamma=1.0, weight=100.0)
    K, N = 32, 10
    traj = np.zeros((K, N+1, 3))
    traj[:, :, 0] = np.linspace(0, 4, N+1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N+1, 3))
    c_disc = cost_disc.compute_cost(traj, ctrl, ref)
    c_full = cost_full.compute_cost(traj, ctrl, ref)
    # discount < 1 → 총 비용 감소 (할인)
    assert np.mean(c_disc) < np.mean(c_full), "Discounted should be less than full"
    print(f"  Discounted(γ=0.5): {np.mean(c_disc):.2f}")
    print(f"  Full(γ=1.0): {np.mean(c_full):.2f}")
    print("✓ PASS\n")

def test_horizon_cbf_gamma_one_equals_standard():
    """γ=1 일 때 기존 CBF와 동일 (가중치 차이 제외)"""
    print("=" * 60)
    print("Test 4: HorizonWeightedCBF - γ=1 Equivalence")
    print("=" * 60)
    obs = [(2.0, 0.0, 0.5)]
    cost_fn = HorizonWeightedCBFCost(obstacles=obs, discount_gamma=1.0, weight=100.0, cbf_alpha=0.1)
    K, N = 16, 10
    traj = np.zeros((K, N+1, 3))
    traj[:, :, 0] = np.linspace(0, 4, N+1)
    ctrl = np.zeros((K, N, 2))
    ref = np.zeros((N+1, 3))
    costs = cost_fn.compute_cost(traj, ctrl, ref)
    assert costs.shape == (K,)
    assert np.all(costs > 0)
    print("✓ PASS\n")

def test_horizon_cbf_barrier_info():
    """get_barrier_info 동작 확인"""
    print("=" * 60)
    print("Test 5: HorizonWeightedCBF - Barrier Info")
    print("=" * 60)
    cost_fn = HorizonWeightedCBFCost(obstacles=[(2.0, 0.0, 0.5)])
    traj = np.zeros((1, 11, 3))
    info = cost_fn.get_barrier_info(traj)
    assert "min_barrier" in info
    assert "is_safe" in info
    assert "barrier_values" in info
    print(f"  min_barrier={info['min_barrier']:.4f}, is_safe={info['is_safe']}")
    # empty obstacles
    cost_fn2 = HorizonWeightedCBFCost(obstacles=[])
    info2 = cost_fn2.get_barrier_info(traj)
    assert info2["is_safe"] == True
    print("✓ PASS\n")

def test_horizon_cbf_update_obstacles():
    """동적 장애물 업데이트"""
    print("=" * 60)
    print("Test 6: HorizonWeightedCBF - Update Obstacles")
    print("=" * 60)
    cost_fn = HorizonWeightedCBFCost(obstacles=[(1.0, 0.0, 0.3)])
    cost_fn.update_obstacles([(5.0, 5.0, 1.0)])
    assert len(cost_fn.obstacles) == 1
    assert cost_fn.obstacles[0] == (5.0, 5.0, 1.0)
    print(f"  repr: {cost_fn}")
    print("✓ PASS\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HorizonWeightedCBFCost Tests".center(60))
    print("=" * 60)
    try:
        test_horizon_cbf_no_obstacle()
        test_horizon_cbf_violation()
        test_horizon_cbf_discount()
        test_horizon_cbf_gamma_one_equals_standard()
        test_horizon_cbf_barrier_info()
        test_horizon_cbf_update_obstacles()
        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
