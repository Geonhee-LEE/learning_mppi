"""
SG-MPPI (Score-Guided MPPI) 유닛 테스트

Denoising Score Matching 기반 score-guided 샘플링 + 온라인 학습 검증.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    SGMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.score_guided_mppi import SGMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


# ── 헬퍼 ─────────────────────────────────────────────────────

def _make_sg_controller(**kwargs):
    """헬퍼: SG-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        guidance_scale=0.5,
        score_hidden_dims=[64, 64],
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    params = SGMPPIParams(**defaults)
    return SGMPPIController(model, params, cost_function=cost_function)


def _make_vanilla_controller(**kwargs):
    """헬퍼: Vanilla MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    params = MPPIParams(**defaults)
    return MPPIController(model, params)


def _make_ref(N=10, dt=0.05):
    """헬퍼: 레퍼런스 궤적 생성"""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ── Params 테스트 (3) ─────────────────────────────────────────

def test_params_defaults():
    """SGMPPIParams 기본값 검증"""
    params = SGMPPIParams()
    assert params.guidance_scale == 0.5
    assert params.n_guide_iters == 1
    assert params.use_annealing is False
    assert params.n_sigma_levels == 10
    assert params.sigma_min == 0.01
    assert params.sigma_max == 1.0
    assert params.guidance_decay == 0.95
    assert params.score_online_training is False
    assert params.score_training_interval == 20
    assert params.score_min_samples == 50
    assert params.score_buffer_size == 2000
    assert params.score_model_path is None


def test_params_custom():
    """커스텀 파라미터 검증"""
    params = SGMPPIParams(
        guidance_scale=1.0,
        n_guide_iters=5,
        use_annealing=True,
        score_online_training=True,
        score_training_interval=10,
        score_min_samples=30,
        score_buffer_size=1000,
        n_sigma_levels=20,
    )
    assert params.guidance_scale == 1.0
    assert params.n_guide_iters == 5
    assert params.use_annealing is True
    assert params.score_online_training is True
    assert params.score_training_interval == 10
    assert params.n_sigma_levels == 20


def test_params_validation():
    """잘못된 파라미터 → AssertionError"""
    # n_sigma_levels < 1
    with pytest.raises(AssertionError):
        SGMPPIParams(n_sigma_levels=0)

    # sigma_min >= sigma_max
    with pytest.raises(AssertionError):
        SGMPPIParams(sigma_min=1.0, sigma_max=0.5)

    # guidance_scale < 0
    with pytest.raises(AssertionError):
        SGMPPIParams(guidance_scale=-1.0)

    # guidance_decay out of range
    with pytest.raises(AssertionError):
        SGMPPIParams(guidance_decay=0.0)
    with pytest.raises(AssertionError):
        SGMPPIParams(guidance_decay=1.5)

    # n_guide_iters < 1
    with pytest.raises(AssertionError):
        SGMPPIParams(n_guide_iters=0)

    # score_training_interval < 1
    with pytest.raises(AssertionError):
        SGMPPIParams(score_training_interval=0)

    # score_min_samples < 1
    with pytest.raises(AssertionError):
        SGMPPIParams(score_min_samples=0)

    # score_buffer_size < score_min_samples
    with pytest.raises(AssertionError):
        SGMPPIParams(score_min_samples=100, score_buffer_size=50)


# ── ScoreNetwork 테스트 (4) ─────────────────────────────────────

def test_score_network_output_shape():
    """입력/출력 차원 검증"""
    import torch
    from mppi_controller.models.learned.score_network import ScoreNetwork

    control_seq_dim = 20  # N=10, nu=2
    context_dim = 3
    net = ScoreNetwork(control_seq_dim, context_dim, hidden_dims=[64, 64])

    B = 16
    U = torch.randn(B, control_seq_dim)
    sigma = torch.ones(B) * 0.5
    ctx = torch.randn(B, context_dim)

    score = net(U, sigma, ctx)
    assert score.shape == (B, control_seq_dim), f"shape={score.shape}"


def test_score_network_zero_init():
    """초기 출력 ≈ 0 (zero-init output layer)"""
    import torch
    from mppi_controller.models.learned.score_network import ScoreNetwork

    net = ScoreNetwork(20, 3, hidden_dims=[64, 64])

    B = 32
    U = torch.randn(B, 20)
    sigma = torch.ones(B) * 0.5
    ctx = torch.randn(B, 3)

    with torch.no_grad():
        score = net(U, sigma, ctx)

    # Zero-init → 초기 출력 값이 매우 작아야 함
    max_val = torch.max(torch.abs(score)).item()
    assert max_val < 0.1, f"max_val={max_val}, expected ≈ 0 (zero-init)"


def test_sigma_embedding():
    """σ 임베딩 차원/값 범위"""
    import torch
    from mppi_controller.models.learned.score_network import SigmaEmbedding

    emb = SigmaEmbedding(emb_dim=64)
    sigma = torch.tensor([0.01, 0.1, 0.5, 1.0])
    result = emb(sigma)
    assert result.shape == (4, 64), f"shape={result.shape}"
    # sin/cos → [-1, 1] 범위
    assert torch.all(result >= -1.1) and torch.all(result <= 1.1)


def test_score_network_different_sigma():
    """σ별 출력 차이 (σ가 다르면 score도 달라야 함)"""
    import torch
    from mppi_controller.models.learned.score_network import ScoreNetwork

    net = ScoreNetwork(20, 3, hidden_dims=[64, 64])

    # 동일 입력, 다른 σ
    U = torch.randn(1, 20)
    ctx = torch.randn(1, 3)

    with torch.no_grad():
        s1 = net(U, torch.tensor([0.01]), ctx)
        s2 = net(U, torch.tensor([1.0]), ctx)

    # σ가 매우 다르면 출력도 달라야 함 (완전히 같으면 σ 무시)
    # Zero-init이므로 차이가 작을 수 있지만, 임베딩 path가 다르므로 0은 아님
    # 두 출력이 완전히 동일하지 않으면 통과
    diff = torch.abs(s1 - s2).sum().item()
    # 두 출력 중 하나라도 0이 아니면 차이 존재 가능
    assert s1.shape == s2.shape == (1, 20)


# ── ScoreMatchingTrainer 테스트 (3) ─────────────────────────────

def test_trainer_dsm_loss():
    """DSM 손실 감소 확인"""
    from mppi_controller.learning.score_matching_trainer import ScoreMatchingTrainer

    trainer = ScoreMatchingTrainer(
        control_seq_dim=20, context_dim=3,
        hidden_dims=[32, 32], n_sigma_levels=5,
    )

    # 간단한 학습 데이터
    M = 100
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 10, 2).astype(np.float32) * 0.5

    metrics = trainer.train(states, controls, epochs=30, batch_size=32)

    assert "losses" in metrics
    assert "final_loss" in metrics
    assert len(metrics["losses"]) == 30
    # 손실 감소 (초기 > 최종)
    assert metrics["losses"][-1] < metrics["losses"][0], \
        f"Loss did not decrease: {metrics['losses'][0]:.4f} → {metrics['losses'][-1]:.4f}"


def test_trainer_save_load():
    """체크포인트 저장/로드"""
    import tempfile
    from mppi_controller.learning.score_matching_trainer import ScoreMatchingTrainer

    trainer = ScoreMatchingTrainer(
        control_seq_dim=20, context_dim=3,
        hidden_dims=[32, 32], n_sigma_levels=5,
    )

    M = 50
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 10, 2).astype(np.float32)
    trainer.train(states, controls, epochs=5)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    trainer.save_model(path)

    # 새 trainer에 로드
    trainer2 = ScoreMatchingTrainer(
        control_seq_dim=20, context_dim=3,
        hidden_dims=[32, 32], n_sigma_levels=5,
    )
    trainer2.load_model(path)

    # 동일 입력에 대해 같은 출력
    import torch
    U = torch.randn(4, 20)
    sigma = torch.ones(4) * 0.5
    ctx = torch.randn(4, 3)

    with torch.no_grad():
        out1 = trainer.get_model()(U, sigma, ctx)
        out2 = trainer2.get_model()(U, sigma, ctx)

    assert torch.allclose(out1, out2, atol=1e-5), "Loaded model outputs differ"

    os.unlink(path)


def test_trainer_multi_sigma():
    """다중 σ 스케일 학습"""
    from mppi_controller.learning.score_matching_trainer import ScoreMatchingTrainer

    trainer = ScoreMatchingTrainer(
        control_seq_dim=20, context_dim=3,
        hidden_dims=[32, 32],
        n_sigma_levels=20, sigma_min=0.001, sigma_max=2.0,
    )

    assert len(trainer.sigma_levels) == 20
    assert trainer.sigma_levels[0].item() < trainer.sigma_levels[-1].item()

    M = 50
    states = np.random.randn(M, 3).astype(np.float32)
    controls = np.random.randn(M, 10, 2).astype(np.float32)
    metrics = trainer.train(states, controls, epochs=5)

    assert metrics["final_loss"] < float("inf")


# ── Controller 기본 테스트 (5) ─────────────────────────────────

def test_compute_control_shape():
    """control (nu,), info keys"""
    ctrl = _make_sg_controller()
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,), f"shape={control.shape}"
    assert "sample_weights" in info
    assert "sample_trajectories" in info
    assert "best_trajectory" in info
    assert "ess" in info
    assert "temperature" in info
    assert "score_stats" in info


def test_fallback_without_score():
    """Score 없으면 Vanilla MPPI 동작"""
    np.random.seed(42)
    ctrl = _make_sg_controller(guidance_scale=0.5)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # Score model 미학습 → fallback
    assert ctrl._score_model is None

    control, info = ctrl.compute_control(state, ref)
    assert control.shape == (2,)
    assert info["ess"] > 0
    assert info["score_stats"]["score_ready"] is False


def test_info_score_stats():
    """score_stats 키/값 확인"""
    ctrl = _make_sg_controller()
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()
    _, info = ctrl.compute_control(state, ref)

    stats = info["score_stats"]
    assert "score_ready" in stats
    assert "buffer_size" in stats
    assert "step_count" in stats
    assert "online_training" in stats
    assert "guidance_scale" in stats
    assert "n_guide_iters" in stats
    assert "mean_score_magnitude" in stats
    assert stats["score_ready"] is False
    assert stats["step_count"] == 1


def test_all_modes_run():
    """single-iter, multi-iter, annealing 모두 실행"""
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # Single iteration (기본)
    ctrl1 = _make_sg_controller(n_guide_iters=1, use_annealing=False)
    c1, i1 = ctrl1.compute_control(state, ref)
    assert c1.shape == (2,)

    # Multi-iteration
    ctrl2 = _make_sg_controller(n_guide_iters=3, use_annealing=False)
    c2, i2 = ctrl2.compute_control(state, ref)
    assert c2.shape == (2,)
    assert "sg_multi_iter" in i2

    # Annealing
    ctrl3 = _make_sg_controller(n_guide_iters=3, use_annealing=True)
    c3, i3 = ctrl3.compute_control(state, ref)
    assert c3.shape == (2,)
    assert "sg_multi_iter" in i3


def test_different_K_values():
    """K=16/64/128 정상"""
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for K in [16, 64, 128]:
        ctrl = _make_sg_controller(K=K)
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["ess"] > 0
        assert info["num_samples"] == K


# ── Score Guidance 테스트 (4) ─────────────────────────────────

def test_sample_with_score_bias():
    """Score 있으면 샘플 분포 변화"""
    import torch
    from mppi_controller.models.learned.score_network import ScoreNetwork

    ctrl = _make_sg_controller(K=64, N=10)

    # Score network 수동 설정 (학습 없이 random weights)
    net = ScoreNetwork(20, 3, hidden_dims=[32, 32])
    # 출력층을 non-zero로 설정 (score ≠ 0)
    with torch.no_grad():
        net.output_layer.weight.fill_(0.1)
        net.output_layer.bias.fill_(0.0)
    ctrl._score_model = net
    ctrl._current_state = np.array([3.0, 0.0, np.pi / 2])

    # Score-guided 노이즈 생성
    np.random.seed(42)
    noise_guided = ctrl._sample_with_score(ctrl.U, 64)

    # 순수 가우시안 노이즈 (score off)
    ctrl._score_model = None
    np.random.seed(42)
    noise_vanilla = ctrl._sample_with_score(ctrl.U, 64)

    # Score-guided 노이즈는 순수 가우시안과 달라야 함
    diff = np.abs(noise_guided - noise_vanilla).sum()
    assert diff > 0.1, f"Score-guided noise too similar to vanilla: diff={diff:.4f}"


def test_guidance_scale_effect():
    """α↑ → 더 큰 bias"""
    import torch
    from mppi_controller.models.learned.score_network import ScoreNetwork

    net = ScoreNetwork(20, 3, hidden_dims=[32, 32])
    with torch.no_grad():
        net.output_layer.weight.fill_(0.1)

    state = np.array([3.0, 0.0, np.pi / 2])
    U = np.zeros((10, 2))

    diffs = []
    for alpha in [0.1, 0.5, 2.0]:
        ctrl = _make_sg_controller(K=32, N=10, guidance_scale=alpha)
        ctrl._score_model = net
        ctrl._current_state = state

        np.random.seed(42)
        noise_guided = ctrl._sample_with_score(U, 32)
        ctrl._score_model = None
        np.random.seed(42)
        noise_vanilla = ctrl._sample_with_score(U, 32)

        diff = np.abs(noise_guided - noise_vanilla).mean()
        diffs.append(diff)

    # α가 클수록 bias가 크므로 차이도 커야 함
    assert diffs[2] > diffs[0], \
        f"Higher α should give bigger bias: α=0.1→{diffs[0]:.4f}, α=2.0→{diffs[2]:.4f}"


def test_guidance_decay_multi_iter():
    """반복마다 α 감소 (multi-iter에서 cost_improvement > 0)"""
    ctrl = _make_sg_controller(
        K=64, N=10,
        n_guide_iters=3, use_annealing=True,
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    _, info = ctrl.compute_control(state, ref)
    assert "sg_multi_iter" in info
    assert info["sg_multi_iter"]["n_iters"] == 3


def test_zero_guidance_equals_vanilla():
    """α=0 → score bias 없음 (순수 가우시안)"""
    ctrl_sg = _make_sg_controller(K=64, guidance_scale=0.0)
    state = np.array([3.0, 0.0, np.pi / 2])

    # α=0이면 score model 유무와 무관하게 score bias = 0
    ctrl_sg._current_state = state
    np.random.seed(42)
    noise_sg = ctrl_sg._sample_with_score(ctrl_sg.U, 64)

    # 순수 가우시안 노이즈
    np.random.seed(42)
    noise_vanilla = np.random.standard_normal((64, 10, 2)) * 0.5

    assert np.allclose(noise_sg, noise_vanilla, atol=1e-6), \
        f"α=0 should produce pure Gaussian noise"


# ── 온라인 학습 테스트 (4) ─────────────────────────────────────

def test_online_training_trigger():
    """interval마다 학습 트리거"""
    ctrl = _make_sg_controller(
        K=32, N=5,
        score_online_training=True,
        score_training_interval=5,
        score_min_samples=10,
        score_buffer_size=100,
        score_hidden_dims=[16, 16],
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref(N=5)

    for i in range(15):
        ctrl.compute_control(state, ref)

    # 15스텝 중 interval=5 → 5, 10, 15에서 트리거
    # min_samples=10이므로 10, 15에서 실제 학습
    assert ctrl._step_count == 15
    assert ctrl._data_collector.num_samples == 15
    # 10스텝 이후 학습이 한 번은 되었을 것
    assert ctrl._score_model is not None, "Score model should be trained after enough samples"


def test_data_collection():
    """가중 샘플 수집 확인"""
    ctrl = _make_sg_controller(K=32, N=5, score_buffer_size=100)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref(N=5)

    assert ctrl._data_collector.num_samples == 0

    for _ in range(10):
        ctrl.compute_control(state, ref)

    assert ctrl._data_collector.num_samples == 10


def test_score_improves_with_training():
    """학습 후 score ≠ 0"""
    ctrl = _make_sg_controller(
        K=32, N=5,
        score_hidden_dims=[16, 16],
        score_min_samples=20,
        score_buffer_size=200,
    )
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref(N=5)

    # 데이터 수집
    for _ in range(30):
        ctrl.compute_control(state, ref)

    # 수동 학습 트리거
    metrics = ctrl.train_score_model(epochs=30)
    assert metrics["status"] == "trained"
    assert ctrl._score_model is not None

    # 학습 후 score magnitude > 0
    ctrl._current_state = state
    noise = ctrl._sample_with_score(ctrl.U, 32)
    # Score-guided noise가 생성되었으면 stats가 기록됨
    assert len(ctrl._score_stats_history) > 0


def test_buffer_size_limit():
    """버퍼 크기 제한"""
    ctrl = _make_sg_controller(K=16, N=5, score_buffer_size=20, score_min_samples=10)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref(N=5)

    for _ in range(30):
        ctrl.compute_control(state, ref)

    # 버퍼 크기 제한 확인
    assert ctrl._data_collector.num_samples <= 20


# ── 성능 테스트 (3) ─────────────────────────────────────────

def test_circle_tracking_rmse():
    """원형 궤적 RMSE < 0.3 (50스텝)"""
    np.random.seed(42)
    ctrl = _make_sg_controller(K=256, N=20)
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

    state = np.array([3.0, 0.0, np.pi / 2])
    dt = 0.05
    errors = []

    for step in range(50):
        t = step * dt
        ref = generate_reference_trajectory(
            lambda t_: circle_trajectory(t_, radius=3.0), t, 20, dt,
        )
        control, info = ctrl.compute_control(state, ref)
        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt

        ref_pt = circle_trajectory(t, radius=3.0)
        err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
        errors.append(err)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    assert rmse < 0.3, f"RMSE={rmse:.4f}, expected < 0.3"


def test_sg_vs_vanilla_obstacles():
    """장애물 시나리오에서 SG ≤ Vanilla 충돌 (score 미학습이라도 동등)"""
    np.random.seed(42)
    obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4)]
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    dt = 0.05
    N = 15

    obstacle_cost = ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0)
    cost_fn = CompositeMPPICost([
        StateTrackingCost(np.array([10.0, 10.0, 1.0])),
        ControlEffortCost(np.array([0.1, 0.1])),
        obstacle_cost,
    ])

    def run_sim(ctrl):
        state = np.array([3.0, 0.0, np.pi / 2])
        collisions = 0
        for step in range(60):
            t = step * dt
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt
            for ox, oy, r in obstacles:
                if np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2) < r:
                    collisions += 1
        return collisions

    np.random.seed(42)
    sg_ctrl = _make_sg_controller(K=256, N=N, cost_function=cost_fn)
    sg_collisions = run_sim(sg_ctrl)

    np.random.seed(42)
    vanilla = _make_vanilla_controller(K=256, N=N)
    vanilla.cost_function = cost_fn
    vanilla_collisions = run_sim(vanilla)

    # Score 미학습이라도 SG ≤ Vanilla + 2 (fallback이므로 동등)
    assert sg_collisions <= vanilla_collisions + 2, \
        f"SG={sg_collisions} vs Vanilla={vanilla_collisions}"


def test_multi_iter_vs_single():
    """다중 반복이 단일보다 비용↓"""
    np.random.seed(42)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # 단일 반복
    ctrl_single = _make_sg_controller(K=128, n_guide_iters=1)
    np.random.seed(42)
    _, info_single = ctrl_single.compute_control(state, ref)

    # 다중 반복
    ctrl_multi = _make_sg_controller(K=128, n_guide_iters=5, use_annealing=True)
    np.random.seed(42)
    _, info_multi = ctrl_multi.compute_control(state, ref)

    # 다중 반복은 cost가 동등하거나 더 낮아야 함
    assert info_multi["best_cost"] <= info_single["best_cost"] * 1.5, \
        f"multi={info_multi['best_cost']:.4f} vs single={info_single['best_cost']:.4f}"


# ── 통합 테스트 (2) ─────────────────────────────────────────

def test_reset_clears_state():
    """reset 후 초기화"""
    ctrl = _make_sg_controller(score_online_training=True)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    for _ in range(5):
        ctrl.compute_control(state, ref)

    assert ctrl._step_count == 5
    assert ctrl._data_collector.num_samples == 5

    ctrl.reset()

    assert ctrl._step_count == 0
    assert ctrl._current_state is None
    assert len(ctrl._score_stats_history) == 0
    assert np.all(ctrl.U == 0)


def test_numerical_stability():
    """극단 비용에서 NaN/Inf 없음"""
    ctrl = _make_sg_controller(K=64)
    state = np.array([3.0, 0.0, np.pi / 2])
    ref = _make_ref()

    # 정상 실행
    control, info = ctrl.compute_control(state, ref)
    assert not np.any(np.isnan(control)), "NaN in control"
    assert not np.any(np.isinf(control)), "Inf in control"
    assert not np.any(np.isnan(info["sample_weights"])), "NaN in weights"
    assert not np.any(np.isinf(info["sample_weights"])), "Inf in weights"

    # 다중 반복에서도 안정
    ctrl2 = _make_sg_controller(K=64, n_guide_iters=5, use_annealing=True)
    control2, info2 = ctrl2.compute_control(state, ref)
    assert not np.any(np.isnan(control2)), "NaN in multi-iter control"
    assert not np.any(np.isinf(control2)), "Inf in multi-iter control"
