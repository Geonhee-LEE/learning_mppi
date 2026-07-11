"""
Step-MPPI (Single-Step MPPI via Differentiable Predictive Control) 유닛 테스트
— 43번째 변형 (arXiv:2604.01539)

~26개 테스트:
  - Params (10): 기본값, 커스텀, 검증 (8개 잘못된 값)
  - Controller (8): 생성/repr, 출력 shape, info keys, step_stats, 바운드,
                    receding horizon, reset, lookahead>1
  - Graceful degradation / zero-init (3): use_learned_proposal=False,
                    zero-init 첫 호출 Δμ≈0, learn_covariance=False
  - Online training (2): 버퍼 성장 + train_count 증가, blend_ratio 스케일
  - Buffer / Network (4): 링 버퍼 capacity, sample_batch shape,
                    ProposalNetwork forward shape + zero-init
  - Performance (1): circle tracking RMSE < 0.35
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import StepMPPIParams
from mppi_controller.controllers.mppi.step_mppi import (
    StepMPPIController,
    StepExperienceBuffer,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)

try:
    import torch
    from mppi_controller.controllers.mppi.step_mppi import (
        ProposalNetwork,
        ProposalTrainer,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── 헬퍼 함수 ─────────────────────────────────────────

def _make_step_controller(**kwargs):
    """Step-MPPI 컨트롤러 생성 헬퍼."""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = StepMPPIParams(**defaults)
    return StepMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_ref(N=10, dt=0.05, t=0.0):
    """원형 참조 궤적."""
    return generate_reference_trajectory(
        lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
    )


# ================================================================
# 1. Params 테스트 (10)
# ================================================================

class TestStepMPPIParams:
    def test_params_defaults(self):
        """기본값 검증."""
        p = StepMPPIParams()
        assert p.lookahead_steps == 1
        assert p.proposal_hidden_dim == 64
        assert p.proposal_n_layers == 2
        assert p.use_learned_proposal is True
        assert p.blend_ratio == 0.7
        assert p.learn_covariance is True
        assert p.online_training is True
        assert p.proposal_lr == 1e-3
        assert p.train_interval == 10
        assert p.train_batch_size == 64
        assert p.buffer_size == 2000
        assert p.min_train_samples == 64
        assert p.entropy_weight == 0.01
        assert p.constraint_weight == 1.0
        assert p.elite_frac == 0.1

    def test_params_custom(self):
        """커스텀 값 검증."""
        p = StepMPPIParams(
            lookahead_steps=3,
            proposal_hidden_dim=128,
            proposal_n_layers=3,
            use_learned_proposal=False,
            blend_ratio=0.5,
            learn_covariance=False,
            online_training=False,
            proposal_lr=5e-4,
            train_interval=5,
            buffer_size=500,
            min_train_samples=32,
            entropy_weight=0.05,
            elite_frac=0.2,
        )
        assert p.lookahead_steps == 3
        assert p.proposal_hidden_dim == 128
        assert p.proposal_n_layers == 3
        assert p.use_learned_proposal is False
        assert p.blend_ratio == 0.5
        assert p.learn_covariance is False
        assert p.online_training is False
        assert p.proposal_lr == 5e-4
        assert p.train_interval == 5
        assert p.buffer_size == 500
        assert p.min_train_samples == 32
        assert p.elite_frac == 0.2

    def test_validation_lookahead_steps(self):
        with pytest.raises(AssertionError):
            StepMPPIParams(lookahead_steps=0)

    def test_validation_hidden_dim(self):
        with pytest.raises(AssertionError):
            StepMPPIParams(proposal_hidden_dim=0)

    def test_validation_n_layers(self):
        with pytest.raises(AssertionError):
            StepMPPIParams(proposal_n_layers=0)

    def test_validation_blend_ratio(self):
        with pytest.raises(AssertionError):
            StepMPPIParams(blend_ratio=1.5)
        with pytest.raises(AssertionError):
            StepMPPIParams(blend_ratio=-0.1)

    def test_validation_proposal_lr(self):
        with pytest.raises(AssertionError):
            StepMPPIParams(proposal_lr=0.0)

    def test_validation_train_interval(self):
        with pytest.raises(AssertionError):
            StepMPPIParams(train_interval=0)

    def test_validation_buffer_vs_min_samples(self):
        with pytest.raises(AssertionError):
            StepMPPIParams(buffer_size=10, min_train_samples=64)

    def test_validation_elite_frac(self):
        with pytest.raises(AssertionError):
            StepMPPIParams(elite_frac=1.5)
        with pytest.raises(AssertionError):
            StepMPPIParams(elite_frac=0.0)


# ================================================================
# 2. Controller 테스트 (8)
# ================================================================

class TestStepMPPIController:
    def test_construction_and_repr(self):
        """생성 및 repr 문자열."""
        ctrl = _make_step_controller()
        assert ctrl.nx == 3
        assert ctrl.nu == 2
        assert ctrl.N == 10
        assert ctrl.input_dim == 9  # 3 * nx
        r = repr(ctrl)
        assert "StepMPPIController" in r
        assert "use_net" in r
        assert "lookahead_steps" in r

    def test_compute_control_shape(self):
        """control (2,), 유한, info 표준 키."""
        ctrl = _make_step_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert np.all(np.isfinite(control))
        for key in (
            "sample_trajectories", "sample_weights", "best_trajectory",
            "best_cost", "mean_cost", "temperature", "ess", "num_samples",
        ):
            assert key in info

    def test_sample_trajectories_shape(self):
        """sample_trajectories shape (K, N+1, 3), weights 합=1."""
        ctrl = _make_step_controller(K=64, N=10)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        traj = info["sample_trajectories"]
        assert traj.shape == (64, 11, 3)

        w = info["sample_weights"]
        assert np.isclose(np.sum(w), 1.0)
        assert info["ess"] > 0.0

    def test_step_stats_subkeys(self):
        """info['step_stats'] 서브키 존재."""
        ctrl = _make_step_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["step_stats"]
        for key in (
            "use_net", "lookahead_steps", "proposal_delta_norm",
            "sigma_eff", "buffer_size", "train_count",
        ):
            assert key in stats
        assert stats["lookahead_steps"] == 1
        assert stats["sigma_eff"].shape == (2,)

    def test_control_bounds_clipping(self):
        """제어 바운드가 적용되면 출력이 범위 내."""
        model = DifferentialDriveKinematic(
            v_max=1.0, omega_max=1.0, wheelbase=0.5,
        )
        params = StepMPPIParams(
            K=64, N=10, dt=0.05, lambda_=1.0,
            sigma=np.array([2.0, 2.0]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
        )
        ctrl = StepMPPIController(model, params)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(5):
            control, _ = ctrl.compute_control(state, ref)
            assert np.all(control >= -1.0 - 1e-9)
            assert np.all(control <= 1.0 + 1e-9)

    def test_receding_horizon_shift(self):
        """receding horizon: 마지막 제어 0으로 시프트."""
        ctrl = _make_step_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        ctrl.compute_control(state, ref)
        assert np.allclose(ctrl.U[-1], 0.0)

    def test_reset(self):
        """reset() 시 step_count 리셋, U=0."""
        ctrl = _make_step_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        for _ in range(4):
            ctrl.compute_control(state, ref)
        assert ctrl._step_count == 4
        ctrl.reset()
        assert ctrl._step_count == 0
        assert np.allclose(ctrl.U, 0.0)

    def test_lookahead_steps_multiple(self):
        """lookahead_steps > 1 시 여러 짧은 업데이트 실행."""
        ctrl = _make_step_controller(lookahead_steps=3)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["step_stats"]["lookahead_steps"] == 3
        assert np.all(np.isfinite(control))


# ================================================================
# 3. Graceful degradation / zero-init (3)
# ================================================================

class TestGracefulDegradation:
    def test_use_learned_proposal_false_is_vanilla(self):
        """use_learned_proposal=False → Vanilla 동작 (use_net False, delta_norm 0)."""
        ctrl = _make_step_controller(use_learned_proposal=False)
        assert ctrl._use_net is False
        assert ctrl.net is None
        assert ctrl.buffer is None
        assert ctrl.trainer is None

        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        control, info = ctrl.compute_control(state, ref)
        stats = info["step_stats"]
        assert stats["use_net"] is False
        assert stats["proposal_delta_norm"] == 0.0
        assert stats["buffer_size"] == 0
        assert stats["train_count"] == 0
        assert np.all(np.isfinite(control))

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_zero_init_first_call_near_vanilla(self):
        """zero-init 네트워크: 첫 호출(학습 전) proposal_delta_norm ≈ 0."""
        ctrl = _make_step_controller(use_learned_proposal=True)
        assert ctrl._use_net is True
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        _, info = ctrl.compute_control(state, ref)
        # mean_delta ≈ 0 (zero-init 출력층) → μ ≈ U_warm
        assert info["step_stats"]["proposal_delta_norm"] < 1e-5
        assert info["step_stats"]["use_net"] is True

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_learn_covariance_false_keeps_base_sigma(self):
        """learn_covariance=False → sigma_eff == base_sigma."""
        ctrl = _make_step_controller(
            use_learned_proposal=True, learn_covariance=False,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        _, info = ctrl.compute_control(state, ref)
        np.testing.assert_allclose(
            info["step_stats"]["sigma_eff"], ctrl._base_sigma,
        )


# ================================================================
# 4. Online training (2)
# ================================================================

class TestOnlineTraining:
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_online_training_grows_buffer_and_trains(self):
        """작은 min_train_samples/train_interval로 버퍼 성장 + train_count>0."""
        ctrl = _make_step_controller(
            use_learned_proposal=True,
            online_training=True,
            min_train_samples=4,
            train_interval=2,
            train_batch_size=8,
            buffer_size=200,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        model = ctrl.model
        dt = ctrl.params.dt
        N = ctrl.N

        for step in range(30):
            t = step * dt
            ref = _make_ref(N=N, dt=dt, t=t)
            control, info = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt

        assert len(ctrl.buffer) > 4
        assert len(ctrl.trainer._loss_history) > 0
        stats = ctrl.get_step_statistics()
        assert stats["train_count"] > 0
        assert stats["buffer_size"] > 4

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_blend_ratio_scales_residual(self):
        """blend_ratio가 학습된 잔차 스케일에 비례."""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # zero-init 출력층을 인위적으로 비-0으로 만들어 잔차 효과 비교
        ctrl_hi = _make_step_controller(use_learned_proposal=True, blend_ratio=1.0)
        ctrl_lo = _make_step_controller(use_learned_proposal=True, blend_ratio=0.2)

        # 동일한 가짜 mean_head bias 주입 (양쪽 동일 네트워크 상태)
        with torch.no_grad():
            for ctrl in (ctrl_hi, ctrl_lo):
                ctrl.net.mean_head.bias.fill_(0.1)

        feats = ctrl_hi._build_features(state, ref)
        md_hi, _ = ctrl_hi._net_proposal(feats)
        md_lo, _ = ctrl_lo._net_proposal(feats)
        # mean_delta 자체는 동일, blend_ratio는 mu 계산에서 적용
        np.testing.assert_allclose(md_hi, md_lo)

        U_warm_hi = ctrl_hi.U.copy()
        U_warm_lo = ctrl_lo.U.copy()
        mu_hi = U_warm_hi + ctrl_hi.step_params.blend_ratio * md_hi
        mu_lo = U_warm_lo + ctrl_lo.step_params.blend_ratio * md_lo
        # blend가 클수록 잔차 기여가 큼
        assert np.linalg.norm(mu_hi - U_warm_hi) > np.linalg.norm(mu_lo - U_warm_lo)


# ================================================================
# 5. Buffer / Network (4)
# ================================================================

class TestStepExperienceBuffer:
    def test_ring_buffer_capacity_cap(self):
        """링 버퍼: capacity 초과 시 len이 capacity로 cap."""
        buf = StepExperienceBuffer(capacity=5)
        rng = np.random.default_rng(0)
        for i in range(12):
            feat = np.ones(9) * i
            target = np.ones((10, 2)) * i
            buf.add(feat, target, float(i))
        assert len(buf) == 5

    def test_sample_batch_shapes(self):
        """sample_batch 반환 shape 검증."""
        buf = StepExperienceBuffer(capacity=50)
        rng = np.random.default_rng(1)
        for i in range(20):
            buf.add(np.ones(9) * i, np.ones((10, 2)) * i, float(i))
        F, T = buf.sample_batch(8, rng)
        assert F.shape == (8, 9)
        assert T.shape == (8, 10, 2)

    def test_sample_batch_caps_at_len(self):
        """배치 크기가 버퍼보다 크면 len으로 제한."""
        buf = StepExperienceBuffer(capacity=50)
        rng = np.random.default_rng(2)
        for i in range(3):
            buf.add(np.ones(9), np.ones((10, 2)), float(i))
        F, T = buf.sample_batch(16, rng)
        assert F.shape[0] == 3
        assert T.shape[0] == 3


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestProposalNetwork:
    def test_forward_shapes_and_zero_init(self):
        """forward 출력 shape + zero-init 첫 forward Δμ≈0."""
        net = ProposalNetwork(
            input_dim=9, N=10, nu=2, hidden_dim=32, n_layers=2,
        )
        x = torch.randn(4, 9)
        mean_delta, log_std = net(x)
        assert mean_delta.shape == (4, 10, 2)
        assert log_std.shape == (4, 2)
        # zero-init 출력층 → mean_delta, log_std 모두 0
        assert torch.allclose(mean_delta, torch.zeros_like(mean_delta))
        assert torch.allclose(log_std, torch.zeros_like(log_std))


# ================================================================
# 6. Performance (1)
# ================================================================

class TestPerformance:
    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.35 (50 스텝)."""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = StepMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            use_learned_proposal=True,
            online_training=True,
            min_train_samples=16,
            train_interval=5,
        )
        ctrl = StepMPPIController(model, params)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt = params.dt
        N = params.N
        num_steps = 50

        errors = []
        for step in range(num_steps):
            t = step * dt
            ref = _make_ref(N=N, dt=dt, t=t)
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt

            ref_pt = circle_trajectory(t, radius=3.0)
            err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
            errors.append(err)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.35, f"RMSE {rmse:.4f} >= 0.35"
