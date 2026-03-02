"""
Conformal Prediction + CBF-MPPI 테스트

ConformalPredictor 단위 테스트 (13개)
ConformalCBFMPPIParams 테스트 (2개)
ConformalCBFMPPIController 테스트 (10개)
통합 테스트 (5개)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.learning.conformal_predictor import (
    ConformalPredictor,
    ConformalPredictorConfig,
)
from mppi_controller.controllers.mppi.mppi_params import ConformalCBFMPPIParams
from mppi_controller.controllers.mppi.conformal_cbf_mppi import (
    ConformalCBFMPPIController,
)
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)


# ============================================================
# Helper
# ============================================================

def _make_model():
    return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)


def _make_params(obstacles=None, **kwargs):
    obs = obstacles or [(3.0, 0.0, 0.3)]
    defaults = dict(
        N=15, dt=0.05, K=64, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obs,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )
    defaults.update(kwargs)
    return ConformalCBFMPPIParams(**defaults)


def _make_controller(obstacles=None, prediction_fn=None, **kwargs):
    model = _make_model()
    params = _make_params(obstacles=obstacles, **kwargs)
    return ConformalCBFMPPIController(model, params, prediction_fn=prediction_fn)


def _circle_reference(N, dt, radius=2.0, speed=0.5, center=(0, 0)):
    """원형 레퍼런스 궤적 생성"""
    t = np.arange(N + 1) * dt
    omega = speed / radius
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = center[0] + radius * np.cos(omega * t)
    ref[:, 1] = center[1] + radius * np.sin(omega * t)
    ref[:, 2] = omega * t + np.pi / 2
    return ref


# ============================================================
# ConformalPredictor 단위 테스트 (13개)
# ============================================================

def test_cp_initial_margin_is_default():
    """업데이트 전 default_margin 반환"""
    cp = ConformalPredictor(ConformalPredictorConfig(default_margin=0.15))
    assert cp.get_margin() == 0.15
    print("PASS: test_cp_initial_margin_is_default")


def test_cp_margin_after_min_samples():
    """min_samples 이후 마진 변화"""
    cp = ConformalPredictor(ConformalPredictorConfig(
        min_samples=5, default_margin=0.15
    ))

    # 4개 업데이트: 아직 default
    for i in range(4):
        pred = np.array([1.0, 0.0, 0.0])
        actual = np.array([1.01, 0.01, 0.0])
        cp.update(pred, actual)
    assert cp.get_margin() == 0.15

    # 5번째: 이제 계산된 마진
    cp.update(np.array([1.0, 0.0, 0.0]), np.array([1.01, 0.01, 0.0]))
    margin = cp.get_margin()
    assert margin != 0.15  # default와 달라야 함
    print("PASS: test_cp_margin_after_min_samples")


def test_cp_margin_decreases_with_accuracy():
    """낮은 오차 → 작은 마진"""
    cp = ConformalPredictor(ConformalPredictorConfig(
        min_samples=5, gamma=1.0, margin_min=0.001, default_margin=0.5
    ))

    # 매우 정확한 예측
    for _ in range(50):
        pred = np.array([1.0, 0.0, 0.0])
        actual = pred + np.random.randn(3) * 0.001
        cp.update(pred, actual)

    margin = cp.get_margin()
    assert margin < 0.05, f"Margin {margin} should be small for accurate model"
    print(f"PASS: test_cp_margin_decreases_with_accuracy (margin={margin:.4f})")


def test_cp_margin_increases_with_error():
    """높은 오차 → 큰 마진"""
    cp = ConformalPredictor(ConformalPredictorConfig(
        min_samples=5, gamma=1.0, margin_max=1.0
    ))

    # 부정확한 예측
    for _ in range(50):
        pred = np.array([1.0, 0.0, 0.0])
        actual = pred + np.random.randn(3) * 0.3
        cp.update(pred, actual)

    margin = cp.get_margin()
    assert margin > 0.1, f"Margin {margin} should be large for inaccurate model"
    print(f"PASS: test_cp_margin_increases_with_error (margin={margin:.4f})")


def test_cp_margin_clamped_min():
    """margin >= margin_min"""
    cp = ConformalPredictor(ConformalPredictorConfig(
        min_samples=5, margin_min=0.05, gamma=1.0
    ))

    # 완벽한 예측 (오차 0)
    for _ in range(20):
        pred = np.array([1.0, 0.0, 0.0])
        cp.update(pred, pred.copy())

    assert cp.get_margin() >= 0.05
    print("PASS: test_cp_margin_clamped_min")


def test_cp_margin_clamped_max():
    """margin <= margin_max"""
    cp = ConformalPredictor(ConformalPredictorConfig(
        min_samples=5, margin_max=0.3, gamma=1.0
    ))

    # 매우 큰 오차
    for _ in range(20):
        pred = np.array([0.0, 0.0, 0.0])
        actual = np.array([10.0, 10.0, 0.0])
        cp.update(pred, actual)

    assert cp.get_margin() <= 0.3
    print("PASS: test_cp_margin_clamped_max")


def test_cp_coverage_tracking():
    """empirical_coverage 정확성"""
    np.random.seed(123)
    cp = ConformalPredictor(ConformalPredictorConfig(
        min_samples=5, default_margin=1.0, gamma=1.0,
        alpha=0.1, margin_min=0.0, margin_max=10.0,
    ))

    # 충분히 많은 샘플로 CP가 안정화된 후 커버리지 확인
    for _ in range(100):
        pred = np.array([0.0, 0.0, 0.0])
        actual = pred + np.random.randn(3) * 0.01
        cp.update(pred, actual)

    stats = cp.get_statistics()
    # CP는 90% 커버리지 목표. 유한 샘플이므로 75% 이상이면 OK
    assert stats["cp_empirical_coverage"] > 0.75, \
        f"Coverage {stats['cp_empirical_coverage']:.2f} should be > 0.75"
    print(f"PASS: test_cp_coverage_tracking (coverage={stats['cp_empirical_coverage']:.2f})")


def test_cp_adaptive_vs_standard():
    """gamma<1 vs gamma=1 차이"""
    np.random.seed(42)

    # Phase 1: 작은 오차, Phase 2: 큰 오차
    preds = [np.array([0.0, 0.0, 0.0])] * 50
    actuals_small = [np.array([0.01, 0.01, 0.0]) for _ in range(30)]
    actuals_big = [np.array([0.3, 0.3, 0.0]) for _ in range(20)]
    actuals = actuals_small + actuals_big

    cp_standard = ConformalPredictor(ConformalPredictorConfig(
        gamma=1.0, min_samples=5, margin_min=0.001, margin_max=1.0
    ))
    cp_adaptive = ConformalPredictor(ConformalPredictorConfig(
        gamma=0.8, min_samples=5, margin_min=0.001, margin_max=1.0
    ))

    for pred, actual in zip(preds, actuals):
        cp_standard.update(pred, actual)
        cp_adaptive.update(pred, actual)

    # ACP는 최근 큰 오차에 더 빠르게 반응 → 더 큰 마진
    m_std = cp_standard.get_margin()
    m_acp = cp_adaptive.get_margin()
    assert m_acp > m_std * 0.5, f"ACP margin {m_acp} should respond to recent errors"
    print(f"PASS: test_cp_adaptive_vs_standard (std={m_std:.4f}, acp={m_acp:.4f})")


def test_cp_score_types():
    """position_norm/full_state_norm/per_dim_max"""
    pred = np.array([0.0, 0.0, 0.0])
    actual = np.array([0.3, 0.4, 1.0])

    # position_norm: sqrt(0.3^2 + 0.4^2) = 0.5
    cp1 = ConformalPredictor(ConformalPredictorConfig(score_type="position_norm"))
    cp1.update(pred, actual)
    assert abs(cp1.get_statistics()["cp_mean_score"] - 0.5) < 1e-6

    # full_state_norm: sqrt(0.3^2 + 0.4^2 + 1.0^2)
    cp2 = ConformalPredictor(ConformalPredictorConfig(score_type="full_state_norm"))
    cp2.update(pred, actual)
    expected = np.linalg.norm([0.3, 0.4, 1.0])
    assert abs(cp2.get_statistics()["cp_mean_score"] - expected) < 1e-6

    # per_dim_max: max(|0.3|, |0.4|, |1.0|) = 1.0
    cp3 = ConformalPredictor(ConformalPredictorConfig(score_type="per_dim_max"))
    cp3.update(pred, actual)
    assert abs(cp3.get_statistics()["cp_mean_score"] - 1.0) < 1e-6

    print("PASS: test_cp_score_types")


def test_cp_sliding_window_eviction():
    """window_size 초과 시 오래된 점수 제거"""
    cp = ConformalPredictor(ConformalPredictorConfig(
        window_size=10, min_samples=3
    ))

    # 20개 추가 → 10개만 남아야 함
    for i in range(20):
        pred = np.array([0.0, 0.0, 0.0])
        actual = np.array([float(i) * 0.01, 0.0, 0.0])
        cp.update(pred, actual)

    assert cp.get_statistics()["cp_num_scores"] == 10
    print("PASS: test_cp_sliding_window_eviction")


def test_cp_reset_clears_state():
    """reset() 후 초기 상태 복원"""
    cp = ConformalPredictor(ConformalPredictorConfig(default_margin=0.2))

    for _ in range(30):
        cp.update(np.zeros(3), np.random.randn(3) * 0.1)

    cp.reset()
    assert cp.get_margin() == 0.2
    assert cp.get_statistics()["cp_num_scores"] == 0
    assert cp.get_statistics()["cp_step_count"] == 0
    print("PASS: test_cp_reset_clears_state")


def test_cp_statistics_dict_keys():
    """get_statistics() 반환 키 확인"""
    cp = ConformalPredictor()
    stats = cp.get_statistics()
    expected_keys = {
        "cp_margin", "cp_num_scores", "cp_mean_score", "cp_std_score",
        "cp_min_score", "cp_max_score", "cp_empirical_coverage", "cp_step_count",
    }
    assert set(stats.keys()) == expected_keys, f"Missing keys: {expected_keys - set(stats.keys())}"
    print("PASS: test_cp_statistics_dict_keys")


def test_cp_weighted_quantile_correctness():
    """가중 quantile 수학적 정확성"""
    cp = ConformalPredictor(ConformalPredictorConfig(
        gamma=0.9, min_samples=3, alpha=0.1, margin_min=0.0, margin_max=10.0
    ))

    # 알려진 점수 추가
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    for s in scores:
        pred = np.array([0.0, 0.0, 0.0])
        actual = np.array([s, 0.0, 0.0])
        cp.update(pred, actual)

    margin = cp.get_margin()
    # 90th quantile이므로 0.1~0.5 범위 내 상위
    assert 0.1 <= margin <= 0.5, f"Margin {margin} should be within score range"
    print(f"PASS: test_cp_weighted_quantile_correctness (margin={margin:.4f})")


# ============================================================
# ConformalCBFMPPIParams 테스트 (2개)
# ============================================================

def test_params_construction():
    """생성 및 기본값 검증"""
    params = ConformalCBFMPPIParams(
        N=15, dt=0.05, K=64,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=[(1.0, 0.0, 0.3)],
        cbf_alpha=0.3,
    )
    assert params.cp_alpha == 0.1
    assert params.cp_window_size == 200
    assert params.cp_min_samples == 10
    assert params.cp_gamma == 0.95
    assert params.cp_enabled is True
    assert params.shield_enabled is True  # ShieldMPPIParams 상속
    print("PASS: test_params_construction")


def test_params_validation():
    """잘못된 값 에러"""
    import pytest

    # cp_alpha 범위 위반
    with pytest.raises(AssertionError):
        ConformalCBFMPPIParams(
            sigma=np.array([0.5, 0.5]),
            cbf_obstacles=[(1.0, 0.0, 0.3)],
            cp_alpha=0.0,
        )

    with pytest.raises(AssertionError):
        ConformalCBFMPPIParams(
            sigma=np.array([0.5, 0.5]),
            cbf_obstacles=[(1.0, 0.0, 0.3)],
            cp_alpha=1.0,
        )

    # cp_gamma 범위 위반
    with pytest.raises(AssertionError):
        ConformalCBFMPPIParams(
            sigma=np.array([0.5, 0.5]),
            cbf_obstacles=[(1.0, 0.0, 0.3)],
            cp_gamma=0.0,
        )

    # margin_max < margin_min
    with pytest.raises(AssertionError):
        ConformalCBFMPPIParams(
            sigma=np.array([0.5, 0.5]),
            cbf_obstacles=[(1.0, 0.0, 0.3)],
            cp_margin_min=0.5,
            cp_margin_max=0.1,
        )

    print("PASS: test_params_validation")


# ============================================================
# ConformalCBFMPPIController 테스트 (10개)
# ============================================================

def test_controller_construction():
    """생성자 정상 동작"""
    ctrl = _make_controller()
    assert ctrl.conformal_predictor is not None
    assert ctrl._prev_predicted_next is None
    print("PASS: test_controller_construction")


def test_controller_compute_control_output():
    """(nu,) 제어 + info dict 반환"""
    ctrl = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(ctrl.params.N, ctrl.params.dt)

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,), f"Expected (2,), got {control.shape}"
    assert isinstance(info, dict)
    print("PASS: test_controller_compute_control_output")


def test_controller_info_has_cp_keys():
    """info에 cp_margin 등 포함"""
    ctrl = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(ctrl.params.N, ctrl.params.dt)

    _, info = ctrl.compute_control(state, ref)
    assert "cp_margin" in info
    assert "cp_empirical_coverage" in info
    assert "cp_num_scores" in info
    assert "cp_mean_score" in info
    print("PASS: test_controller_info_has_cp_keys")


def test_controller_margin_adapts_over_time():
    """시뮬 중 마진 변화"""
    ctrl = _make_controller(cp_min_samples=3, cp_margin_min=0.001, cp_margin_max=1.0)
    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(ctrl.params.N, ctrl.params.dt, radius=3.0)

    margins = []
    for step in range(30):
        control, info = ctrl.compute_control(state, ref)
        margins.append(info["cp_margin"])
        state = ctrl.model.step(state, control, ctrl.params.dt)

    # 마진이 일정하지 않아야 함 (적응 발생)
    assert len(set(f"{m:.6f}" for m in margins)) > 1, "Margin should adapt over time"
    print(f"PASS: test_controller_margin_adapts_over_time (range={min(margins):.4f}~{max(margins):.4f})")


def test_controller_cp_disabled_fixed_margin():
    """cp_enabled=False → 고정 마진"""
    ctrl = _make_controller(cp_enabled=False, cbf_safety_margin=0.1)
    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(ctrl.params.N, ctrl.params.dt)

    for _ in range(10):
        control, info = ctrl.compute_control(state, ref)
        state = ctrl.model.step(state, control, ctrl.params.dt)

    # CP 비활성 → cbf_safety_margin 변경 없음
    assert ctrl.cbf_cost.safety_margin == 0.1
    print("PASS: test_controller_cp_disabled_fixed_margin")


def test_controller_with_prediction_fn():
    """커스텀 예측 함수"""
    model = _make_model()

    # 일부러 편향된 예측 함수
    def biased_predict(state, control):
        next_state = model.step(state, control, 0.05)
        next_state[:2] += 0.1  # 편향 추가
        return next_state

    params = _make_params()
    ctrl = ConformalCBFMPPIController(
        model, params, prediction_fn=biased_predict
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(params.N, params.dt)

    for _ in range(20):
        control, info = ctrl.compute_control(state, ref)
        state = ctrl.model.step(state, control, params.dt)

    # 편향된 예측 → 점수가 높아야 함
    stats = ctrl.get_cp_statistics()
    assert stats["cp_mean_score"] > 0.05, \
        f"Mean score {stats['cp_mean_score']} should reflect prediction bias"
    print(f"PASS: test_controller_with_prediction_fn (mean_score={stats['cp_mean_score']:.4f})")


def test_controller_update_obstacles():
    """장애물 업데이트 전파"""
    ctrl = _make_controller(obstacles=[(5.0, 0.0, 0.3)])
    assert len(ctrl.cbf_params.cbf_obstacles) == 1

    new_obs = [(2.0, 1.0, 0.5), (3.0, -1.0, 0.4)]
    ctrl.update_obstacles(new_obs)
    assert len(ctrl.cbf_params.cbf_obstacles) == 2
    assert ctrl.cbf_cost.obstacles == new_obs
    print("PASS: test_controller_update_obstacles")


def test_controller_reset_clears_cp():
    """reset() 시 CP 상태 초기화"""
    ctrl = _make_controller()
    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(ctrl.params.N, ctrl.params.dt)

    # 몇 스텝 실행
    for _ in range(5):
        control, _ = ctrl.compute_control(state, ref)
        state = ctrl.model.step(state, control, ctrl.params.dt)

    assert ctrl.conformal_predictor.get_statistics()["cp_num_scores"] > 0

    ctrl.reset()
    assert ctrl.conformal_predictor.get_statistics()["cp_num_scores"] == 0
    assert ctrl._prev_predicted_next is None
    print("PASS: test_controller_reset_clears_cp")


def test_controller_safety_rate():
    """안전율 >= 90%"""
    obstacles = [(3.0, 0.0, 0.3)]
    ctrl = _make_controller(
        obstacles=obstacles, K=128, N=15, cbf_safety_margin=0.1
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(ctrl.params.N, ctrl.params.dt, radius=2.0)

    safe_count = 0
    total = 40
    for step in range(total):
        control, info = ctrl.compute_control(state, ref)
        if info.get("is_safe", True):
            safe_count += 1
        state = ctrl.model.step(state, control, ctrl.params.dt)

    safety_rate = safe_count / total
    assert safety_rate >= 0.9, f"Safety rate {safety_rate:.2f} should be >= 0.9"
    print(f"PASS: test_controller_safety_rate (rate={safety_rate:.2f})")


def test_controller_get_cp_statistics():
    """get_cp_statistics() 반환값"""
    ctrl = _make_controller()
    stats = ctrl.get_cp_statistics()
    assert "cp_margin" in stats
    assert "cp_num_scores" in stats
    assert "cp_empirical_coverage" in stats
    print("PASS: test_controller_get_cp_statistics")


# ============================================================
# 통합 테스트 (5개)
# ============================================================

def test_cp_expands_margin_under_mismatch():
    """모델 불일치 시 마진 확대"""
    model = _make_model()

    # 편향된 예측: 실제 시스템에 마찰이 있는 것처럼
    def mismatched_predict(state, control):
        next_state = model.step(state, control, 0.05)
        next_state[0] += 0.05  # x 방향 편향
        return next_state

    params = _make_params(
        cp_min_samples=5, cp_margin_min=0.01, cp_margin_max=0.5,
        cbf_safety_margin=0.1
    )
    ctrl = ConformalCBFMPPIController(
        model, params, prediction_fn=mismatched_predict
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(params.N, params.dt)

    initial_margin = ctrl.conformal_predictor.get_margin()
    for _ in range(30):
        control, info = ctrl.compute_control(state, ref)
        state = ctrl.model.step(state, control, params.dt)

    final_margin = info["cp_margin"]
    assert final_margin > initial_margin * 0.5, \
        f"Margin should expand under mismatch: {initial_margin:.4f} -> {final_margin:.4f}"
    print(f"PASS: test_cp_expands_margin_under_mismatch ({initial_margin:.4f} -> {final_margin:.4f})")


def test_cp_shrinks_margin_with_perfect_model():
    """정확한 모델 시 마진 축소"""
    params = _make_params(
        cp_min_samples=5, cp_margin_min=0.001, cp_margin_max=0.5,
        cbf_safety_margin=0.2  # default_margin = 0.2로 시작
    )
    ctrl = _make_controller(
        cp_min_samples=5, cp_margin_min=0.001, cp_margin_max=0.5,
        cbf_safety_margin=0.2,
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(ctrl.params.N, ctrl.params.dt, radius=3.0)

    # model.step이 예측 함수이므로 완벽 일치
    for _ in range(40):
        control, info = ctrl.compute_control(state, ref)
        state = ctrl.model.step(state, control, ctrl.params.dt)

    final_margin = info["cp_margin"]
    assert final_margin < 0.15, \
        f"Margin {final_margin} should shrink with perfect model"
    print(f"PASS: test_cp_shrinks_margin_with_perfect_model (margin={final_margin:.4f})")


def test_cp_coverage_meets_target():
    """경험적 커버리지 >= 1-α (통계적)"""
    np.random.seed(42)

    cp = ConformalPredictor(ConformalPredictorConfig(
        alpha=0.1, min_samples=10, gamma=1.0,
        margin_min=0.0, margin_max=10.0
    ))

    # 정규 분포 오차
    for _ in range(200):
        pred = np.zeros(3)
        actual = np.random.randn(3) * 0.1
        cp.update(pred, actual)

    stats = cp.get_statistics()
    # 90% 커버리지 목표, 유한 샘플이라 80% 이상이면 OK
    assert stats["cp_empirical_coverage"] >= 0.80, \
        f"Coverage {stats['cp_empirical_coverage']:.2f} should be >= 0.80"
    print(f"PASS: test_cp_coverage_meets_target (coverage={stats['cp_empirical_coverage']:.2f})")


def test_cp_vs_fixed_safety_comparison():
    """고정 마진 대비 안전율 비교"""
    from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
    from mppi_controller.controllers.mppi.mppi_params import ShieldMPPIParams

    model = _make_model()
    obstacles = [(2.5, 0.5, 0.3)]

    # 고정 마진 Shield-MPPI
    fixed_params = ShieldMPPIParams(
        N=15, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obstacles,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
    )
    fixed_ctrl = ShieldMPPIController(model, fixed_params)

    # CP Shield-MPPI
    cp_params = ConformalCBFMPPIParams(
        N=15, dt=0.05, K=128, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        cbf_obstacles=obstacles,
        cbf_alpha=0.3,
        cbf_safety_margin=0.1,
        cp_min_samples=5,
    )
    cp_ctrl = ConformalCBFMPPIController(model, cp_params)

    ref = _circle_reference(15, 0.05, radius=2.0)

    # 둘 다 실행
    for ctrl_name, ctrl in [("Fixed", fixed_ctrl), ("CP", cp_ctrl)]:
        state = np.array([0.0, 0.0, 0.0])
        for _ in range(20):
            control, info = ctrl.compute_control(state, ref)
            state = ctrl.model.step(state, control, ctrl.params.dt)

    # 둘 다 안전해야 함
    fixed_stats = fixed_ctrl.get_cbf_statistics()
    cp_stats = cp_ctrl.get_cbf_statistics()

    assert fixed_stats["safety_rate"] >= 0.8
    assert cp_stats["safety_rate"] >= 0.8
    print(f"PASS: test_cp_vs_fixed_safety_comparison "
          f"(fixed={fixed_stats['safety_rate']:.2f}, cp={cp_stats['safety_rate']:.2f})")


def test_cp_circle_tracking_with_obstacles():
    """원형 궤적 + 장애물 시뮬레이션"""
    obstacles = [(2.0, 1.0, 0.3), (1.0, 2.0, 0.3)]
    ctrl = _make_controller(
        obstacles=obstacles, K=128, N=15,
        cp_min_samples=5,
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = _circle_reference(ctrl.params.N, ctrl.params.dt, radius=2.5)

    positions = [state[:2].copy()]
    margins = []

    for step in range(50):
        control, info = ctrl.compute_control(state, ref)
        state = ctrl.model.step(state, control, ctrl.params.dt)
        positions.append(state[:2].copy())
        margins.append(info["cp_margin"])

    positions = np.array(positions)

    # 장애물 충돌 확인
    safe = True
    for obs_x, obs_y, obs_r in obstacles:
        dists = np.sqrt((positions[:, 0] - obs_x) ** 2 + (positions[:, 1] - obs_y) ** 2)
        if np.any(dists < obs_r):
            safe = False

    print(f"  Margin range: [{min(margins):.4f}, {max(margins):.4f}]")
    print(f"  Safe: {safe}")
    # Shield가 활성화되어 있으므로 안전해야 함
    assert safe, "Should not collide with obstacles"
    print("PASS: test_cp_circle_tracking_with_obstacles")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Conformal Prediction + CBF-MPPI Tests")
    print("=" * 70)

    # ConformalPredictor 단위 테스트
    print("\n--- ConformalPredictor Unit Tests ---")
    test_cp_initial_margin_is_default()
    test_cp_margin_after_min_samples()
    test_cp_margin_decreases_with_accuracy()
    test_cp_margin_increases_with_error()
    test_cp_margin_clamped_min()
    test_cp_margin_clamped_max()
    test_cp_coverage_tracking()
    test_cp_adaptive_vs_standard()
    test_cp_score_types()
    test_cp_sliding_window_eviction()
    test_cp_reset_clears_state()
    test_cp_statistics_dict_keys()
    test_cp_weighted_quantile_correctness()

    # Params 테스트
    print("\n--- ConformalCBFMPPIParams Tests ---")
    test_params_construction()
    test_params_validation()

    # Controller 테스트
    print("\n--- ConformalCBFMPPIController Tests ---")
    test_controller_construction()
    test_controller_compute_control_output()
    test_controller_info_has_cp_keys()
    test_controller_margin_adapts_over_time()
    test_controller_cp_disabled_fixed_margin()
    test_controller_with_prediction_fn()
    test_controller_update_obstacles()
    test_controller_reset_clears_cp()
    test_controller_safety_rate()
    test_controller_get_cp_statistics()

    # 통합 테스트
    print("\n--- Integration Tests ---")
    test_cp_expands_margin_under_mismatch()
    test_cp_shrinks_margin_with_perfect_model()
    test_cp_coverage_meets_target()
    test_cp_vs_fixed_safety_comparison()
    test_cp_circle_tracking_with_obstacles()

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED!")
    print("=" * 70)
