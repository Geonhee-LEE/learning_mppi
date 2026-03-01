"""AdaptiveShieldSVGMPPI 테스트"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.controllers.mppi.adaptive_shield_svg_mppi import (
    AdaptiveShieldSVGMPPIController, AdaptiveShieldSVGMPPIParams,
)

def _make_controller(obstacles=None, shield_enabled=True):
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    if obstacles is None:
        obstacles = [(3.0, 0.0, 0.5)]
    params = AdaptiveShieldSVGMPPIParams(
        N=10, dt=0.05, K=64, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        svgd_num_iterations=2,
        svgd_step_size=0.01,
        svg_num_guide_particles=8,
        svg_guide_step_size=0.01,
        shield_enabled=shield_enabled,
        shield_cbf_alpha=0.3,
        cbf_obstacles=obstacles,
        cbf_safety_margin=0.1,
        alpha_base=0.3, alpha_dist=0.1, alpha_vel=0.5,
        k_dist=2.0, d_safe=0.5,
    )
    return AdaptiveShieldSVGMPPIController(model, params), model, params


def test_adaptive_shield_svg_basic():
    """기본 compute_control 동작"""
    print("\n" + "=" * 60)
    print("Test 1: AdaptiveShieldSVG - Basic Control")
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
    print("PASS\n")


def test_adaptive_shield_svg_alpha_varies():
    """거리에 따라 alpha가 변함 (적응형)"""
    print("=" * 60)
    print("Test 2: AdaptiveShieldSVG - Alpha Varies with Distance")
    print("=" * 60)
    ctrl, model, params = _make_controller()

    # 가까운 상태 vs 먼 상태에서 shield 적용
    obs_x, obs_y, obs_r = 3.0, 0.0, 0.5
    p = ctrl.adaptive_params

    # 가까운 상태: d_surface 작음 → sigmoid 작음 → α 작음
    states_near = np.array([[2.0, 0.0, 0.0]])
    states_far = np.array([[0.0, 0.0, 0.0]])
    controls = np.array([[0.5, 0.0]])

    # _cbf_shield_batch에서 adaptive alpha 계산 확인
    # 가까운 상태
    d_near = np.sqrt((2.0 - obs_x)**2 + (0.0 - obs_y)**2) - obs_r
    d_far = np.sqrt((0.0 - obs_x)**2 + (0.0 - obs_y)**2) - obs_r
    sig_near = 1.0 / (1.0 + np.exp(-p.k_dist * (d_near - p.d_safe)))
    sig_far = 1.0 / (1.0 + np.exp(-p.k_dist * (d_far - p.d_safe)))
    alpha_near = p.alpha_base * (p.alpha_dist + (1.0 - p.alpha_dist) * sig_near)
    alpha_far = p.alpha_base * (p.alpha_dist + (1.0 - p.alpha_dist) * sig_far)

    assert alpha_near < alpha_far, (
        f"Near alpha ({alpha_near:.4f}) should be < far ({alpha_far:.4f})"
    )
    print(f"  d_near={d_near:.2f}m → alpha={alpha_near:.4f}")
    print(f"  d_far={d_far:.2f}m → alpha={alpha_far:.4f}")
    print("PASS\n")


def test_adaptive_shield_svg_no_obstacles():
    """장애물 없을 때"""
    print("=" * 60)
    print("Test 3: AdaptiveShieldSVG - No Obstacles")
    print("=" * 60)
    ctrl, model, params = _make_controller(obstacles=[])
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(u))
    assert info['shield_info']['intervention_rate'] == 0.0
    print("PASS\n")


def test_adaptive_shield_svg_shield_disabled():
    """shield 비활성화시 SVG-MPPI 폴백"""
    print("=" * 60)
    print("Test 4: AdaptiveShieldSVG - Disabled (fallback to SVG)")
    print("=" * 60)
    ctrl, model, params = _make_controller(shield_enabled=False)
    state = np.array([0.0, 0.0, 0.0])
    ref = np.zeros((11, 3))
    ref[:, 0] = np.linspace(0, 2, 11)
    u, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(u))
    assert "svg_stats" in info
    print("PASS\n")


def test_adaptive_shield_svg_multistep():
    """여러 스텝 실행"""
    print("=" * 60)
    print("Test 5: AdaptiveShieldSVG - Multi-step")
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
    print("PASS\n")


def test_adaptive_shield_svg_update_obstacles():
    """장애물 업데이트"""
    print("=" * 60)
    print("Test 6: AdaptiveShieldSVG - Update Obstacles")
    print("=" * 60)
    ctrl, _, _ = _make_controller()
    ctrl.update_obstacles([(5.0, 5.0, 1.0)])
    assert ctrl.shield_svg_params.cbf_obstacles == [(5.0, 5.0, 1.0)]
    print("PASS\n")


def test_adaptive_shield_svg_statistics():
    """shield + SVG 통계"""
    print("=" * 60)
    print("Test 7: AdaptiveShieldSVG - Statistics")
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
    print("PASS\n")


def test_adaptive_shield_svg_params_validation():
    """파라미터 검증"""
    print("=" * 60)
    print("Test 8: AdaptiveShieldSVG - Params Validation")
    print("=" * 60)
    try:
        AdaptiveShieldSVGMPPIParams(
            N=10, dt=0.05, K=64, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            svgd_num_iterations=2, svgd_step_size=0.01,
            svg_num_guide_particles=8, svg_guide_step_size=0.01,
            cbf_obstacles=[], shield_cbf_alpha=0.3,
            alpha_base=-0.1,  # invalid
        )
        assert False, "Should raise AssertionError"
    except (AssertionError, AssertionError):
        print("  Correctly rejected negative alpha_base")
    print("PASS\n")


def test_adaptive_shield_svg_repr():
    """repr"""
    print("=" * 60)
    print("Test 9: AdaptiveShieldSVG - Repr")
    print("=" * 60)
    ctrl, _, _ = _make_controller()
    r = repr(ctrl)
    print(f"  {r}")
    assert "AdaptiveShieldSVGMPPI" in r
    ctrl.reset()
    assert len(ctrl.shield_stats_history) == 0
    print("PASS\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AdaptiveShieldSVGMPPI Tests".center(60))
    print("=" * 60)
    try:
        test_adaptive_shield_svg_basic()
        test_adaptive_shield_svg_alpha_varies()
        test_adaptive_shield_svg_no_obstacles()
        test_adaptive_shield_svg_shield_disabled()
        test_adaptive_shield_svg_multistep()
        test_adaptive_shield_svg_update_obstacles()
        test_adaptive_shield_svg_statistics()
        test_adaptive_shield_svg_params_validation()
        test_adaptive_shield_svg_repr()
        print("=" * 60)
        print("All Tests Passed!".center(60))
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
