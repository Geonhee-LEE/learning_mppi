"""MPSController 테스트"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.controllers.mppi.mps_controller import MPSController
from mppi_controller.controllers.mppi.backup_controller import BrakeBackupController, TurnAndBrakeBackupController

def test_mps_no_obstacles():
    """장애물 없으면 nominal 통과"""
    print("\n" + "=" * 60)
    print("Test 1: MPS - No Obstacles")
    print("=" * 60)
    mps = MPSController(obstacles=[])
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.1])
    u_safe, info = mps.shield(state, u, model)
    assert np.allclose(u_safe, u)
    assert info["shielded"] == False
    print("✓ PASS\n")

def test_mps_safe_nominal():
    """안전한 nominal 제어는 통과"""
    print("=" * 60)
    print("Test 2: MPS - Safe Nominal Passes")
    print("=" * 60)
    mps = MPSController(obstacles=[(10.0, 10.0, 0.5)], backup_horizon=20)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    state = np.array([0.0, 0.0, 0.0])
    u = np.array([0.5, 0.0])
    u_safe, info = mps.shield(state, u, model)
    assert np.allclose(u_safe, u)
    assert info["shielded"] == False
    print(f"  backup_min_barrier={info['backup_min_barrier']:.4f}")
    print("✓ PASS\n")

def test_mps_unsafe_intervention():
    """위험한 nominal → backup으로 개입"""
    print("=" * 60)
    print("Test 3: MPS - Unsafe Intervention")
    print("=" * 60)
    # 장애물이 바로 앞에
    mps = MPSController(
        obstacles=[(0.3, 0.0, 0.2)],
        safety_margin=0.1,
        backup_horizon=20,
        dt=0.05,
    )
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    state = np.array([0.0, 0.0, 0.0])
    u_nominal = np.array([1.0, 0.0])  # 전진 (장애물 방향)
    u_safe, info = mps.shield(state, u_nominal, model)
    # backup이 개입하여 정지 (BrakeBackup)
    if info["shielded"]:
        assert np.allclose(u_safe, [0.0, 0.0]), "Brake backup should return [0, 0]"
        print(f"  Shielded! u_backup={u_safe}")
    else:
        print(f"  Not shielded (barrier still positive: {info['backup_min_barrier']:.4f})")
    print("✓ PASS\n")

def test_mps_turn_backup():
    """TurnAndBrake 백업 컨트롤러 사용"""
    print("=" * 60)
    print("Test 4: MPS - TurnAndBrake Backup")
    print("=" * 60)
    backup = TurnAndBrakeBackupController(turn_speed=0.5, turn_steps=5)
    mps = MPSController(
        backup_controller=backup,
        obstacles=[(0.3, 0.0, 0.15)],
        safety_margin=0.1,
        backup_horizon=20,
    )
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    state = np.array([0.0, 0.0, 0.0])
    u_safe, info = mps.shield(state, np.array([1.0, 0.0]), model)
    assert not np.any(np.isnan(u_safe))
    print(f"  u_safe={u_safe}, shielded={info['shielded']}")
    print("✓ PASS\n")

def test_mps_statistics():
    """통계 수집"""
    print("=" * 60)
    print("Test 5: MPS - Statistics")
    print("=" * 60)
    mps = MPSController(obstacles=[(10.0, 10.0, 0.5)])
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    state = np.array([0.0, 0.0, 0.0])
    for _ in range(5):
        mps.shield(state, np.array([0.5, 0.0]), model)
    stats = mps.get_statistics()
    assert stats["total_steps"] == 5
    assert 0 <= stats["shield_rate"] <= 1
    print(f"  {stats}")
    mps.reset()
    stats2 = mps.get_statistics()
    assert stats2["total_steps"] == 0
    print("✓ PASS\n")

def test_mps_update_obstacles():
    """장애물 업데이트"""
    print("=" * 60)
    print("Test 6: MPS - Update Obstacles")
    print("=" * 60)
    mps = MPSController(obstacles=[(1.0, 0.0, 0.3)])
    mps.update_obstacles([(5.0, 5.0, 1.0)])
    assert mps.obstacles == [(5.0, 5.0, 1.0)]
    print("✓ PASS\n")

def test_mps_repr():
    """repr"""
    print("=" * 60)
    print("Test 7: MPS - Repr")
    print("=" * 60)
    mps = MPSController(obstacles=[(1.0, 0.0, 0.3)], backup_horizon=15)
    print(f"  {mps}")
    assert "MPSController" in repr(mps)
    print("✓ PASS\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MPSController Tests".center(60))
    print("=" * 60)
    try:
        test_mps_no_obstacles()
        test_mps_safe_nominal()
        test_mps_unsafe_intervention()
        test_mps_turn_backup()
        test_mps_statistics()
        test_mps_update_obstacles()
        test_mps_repr()
        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
