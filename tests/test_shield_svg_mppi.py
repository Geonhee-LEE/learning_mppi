"""ShieldSVGMPPI 테스트"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.controllers.mppi.shield_svg_mppi import (
    ShieldSVGMPPIController, ShieldSVGMPPIParams,
)

def _make_controller(obstacles=None, shield_enabled=True):
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    if obstacles is None:
        obstacles = [(3.0, 0.0, 0.5)]
    params = ShieldSVGMPPIParams(
        N=10, dt=0.05, K=64, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        svgd_num_iterations=3,
        svgd_step_size=0.01,
        svg_num_guide_particles=8,
        svg_guide_step_size=0.01,
        shield_enabled=shield_enabled,
        shield_cbf_alpha=0.3,
        cbf_obstacles=obstacles,
        cbf_safety_margin=0.1,
    )
    return ShieldSVGMPPIController(model, params), model, params

def test_shield_svg_basic():
    """기본 동작"""
    print("\n" + "=" * 60)
    print("Test 1: ShieldSVG - Basic Control")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert u.shape == (2,)
    assert not np.any(np.isnan(u))
    assert "shield_info" in info
    assert "svg_stats" in info
    print(f"  u={u}")
    print(f"  intervention_rate={info['shield_info']['intervention_rate']:.3f}")
    print(f"  guide_improvement={info['svg_stats']['guide_cost_improvement']:.2f}")
    print("✓ PASS\n")

def test_shield_svg_no_obstacles():
    """장애물 없을 때"""
    print("=" * 60)
    print("Test 2: ShieldSVG - No Obstacles")
    print("=" * 60)
    ctrl, model, params = _make_controller(obstacles=[])
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(u))
    assert info['shield_info']['intervention_rate'] == 0.0
    print("✓ PASS\n")

def test_shield_svg_disabled():
    """shield 비활성화시 SVG-MPPI 동작"""
    print("=" * 60)
    print("Test 3: ShieldSVG - Disabled (fallback to SVG)")
    print("=" * 60)
    ctrl, model, params = _make_controller(shield_enabled=False)
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(u))
    assert "svg_stats" in info  # SVG 통계는 있어야 함
    print("✓ PASS\n")

def test_shield_svg_multistep():
    """여러 스텝"""
    print("=" * 60)
    print("Test 4: ShieldSVG - Multi-step")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    for _ in range(3):
        u, info = ctrl.compute_control(state, ref)
        state = model.step(state, u, params.dt)
    assert not np.any(np.isnan(state))
    print(f"  Final state: {state}")
    print("✓ PASS\n")

def test_shield_svg_update_obstacles():
    """장애물 업데이트"""
    print("=" * 60)
    print("Test 5: ShieldSVG - Update Obstacles")
    print("=" * 60)
    ctrl, _, _ = _make_controller()
    ctrl.update_obstacles([(5.0, 5.0, 1.0)])
    assert ctrl.shield_svg_params.cbf_obstacles == [(5.0, 5.0, 1.0)]
    print("✓ PASS\n")

def test_shield_svg_statistics():
    """통계"""
    print("=" * 60)
    print("Test 6: ShieldSVG - Statistics")
    print("=" * 60)
    ctrl, model, params = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    for _ in range(3):
        ctrl.compute_control(state, ref)
    shield_stats = ctrl.get_shield_statistics()
    svg_stats = ctrl.get_svg_statistics()
    assert shield_stats["num_steps"] == 3
    assert len(svg_stats["svg_stats_history"]) == 3
    print(f"  Shield: {shield_stats}")
    print("✓ PASS\n")

def test_shield_svg_repr():
    """repr"""
    print("=" * 60)
    print("Test 7: ShieldSVG - Repr")
    print("=" * 60)
    ctrl, _, _ = _make_controller()
    print(f"  {ctrl}")
    assert "ShieldSVGMPPI" in repr(ctrl)
    ctrl.reset()
    assert len(ctrl.shield_stats_history) == 0
    print("✓ PASS\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ShieldSVGMPPI Tests".center(60))
    print("=" * 60)
    try:
        test_shield_svg_basic()
        test_shield_svg_no_obstacles()
        test_shield_svg_disabled()
        test_shield_svg_multistep()
        test_shield_svg_update_obstacles()
        test_shield_svg_statistics()
        test_shield_svg_repr()
        print("=" * 60)
        print("All Tests Passed! ✓".center(60))
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}\n")
        sys.exit(1)
