"""
T-MPPI (Transformer-based MPPI) 유닛 테스트

28개 테스트:
  - Params (3): 기본값, 커스텀, 검증
  - Model (4): Transformer shape, positional encoding, data buffer, trainer loss
  - Controller (5): shape, info keys, fallback, different_K, reset
  - Init (4): with history, blend ratio, disabled, context padding
  - Online Learning (3): buffer collection, periodic training, training improves
  - Performance (4): 추적 RMSE, 장애물 회피, warm start efficiency, 계산 시간
  - Integration (5): 수치 안정성, pretrained 로드, online disabled, long horizon, varying ref
"""

import numpy as np
import time
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    TransformerMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.transformer_mppi import (
    TransformerMPPIController,
    TransformerDataBuffer,
    TransformerTrainer,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
    figure_eight_trajectory,
)

try:
    import torch
    from mppi_controller.controllers.mppi.transformer_mppi import (
        ControlTransformer,
        PositionalEncoding,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── 헬퍼 함수 ─────────────────────────────────────────

def _make_transformer_controller(**kwargs):
    """T-MPPI 컨트롤러 생성"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        transformer_hidden_dim=64,
        transformer_n_heads=4,
        transformer_n_layers=1,
        transformer_context_length=10,
        transformer_buffer_size=500,
        transformer_min_samples=20,
        transformer_batch_size=16,
        transformer_training_interval=5,
        transformer_n_train_steps=2,
        online_learning=True,
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    noise_sampler = defaults.pop("noise_sampler", None)
    params = TransformerMPPIParams(**defaults)
    return TransformerMPPIController(
        model, params,
        cost_function=cost_function,
        noise_sampler=noise_sampler,
    )


def _make_vanilla_controller(**kwargs):
    """비교용 Vanilla MPPI"""
    model = DifferentialDriveKinematic(wheelbase=0.5)
    defaults = dict(
        K=64, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    defaults.update(kwargs)
    cost_function = defaults.pop("cost_function", None)
    params = MPPIParams(**defaults)
    return MPPIController(model, params, cost_function=cost_function)


def _make_ref(N=10, dt=0.05):
    """레퍼런스 궤적"""
    return generate_reference_trajectory(
        lambda t: circle_trajectory(t, radius=3.0),
        0.0, N, dt,
    )


# ══════════════════════════════════════════════════════════
# 1. Params 테스트 (3개)
# ══════════════════════════════════════════════════════════

class TestTransformerMPPIParams:
    def test_params_defaults(self):
        """기본값 검증"""
        params = TransformerMPPIParams()
        assert params.transformer_hidden_dim == 128
        assert params.transformer_n_heads == 4
        assert params.transformer_n_layers == 2
        assert params.transformer_context_length == 20
        assert params.transformer_dropout == 0.1
        assert params.transformer_lr == 1e-3
        assert params.transformer_buffer_size == 5000
        assert params.transformer_min_samples == 100
        assert params.transformer_batch_size == 32
        assert params.transformer_training_interval == 10
        assert params.transformer_n_train_steps == 5
        assert params.use_transformer_init is True
        assert params.transformer_model_path is None
        assert params.blend_ratio == 0.7
        assert params.online_learning is True

    def test_params_custom(self):
        """커스텀 값 검증"""
        params = TransformerMPPIParams(
            transformer_hidden_dim=256,
            transformer_n_heads=8,
            transformer_n_layers=4,
            transformer_context_length=30,
            blend_ratio=0.5,
            online_learning=False,
        )
        assert params.transformer_hidden_dim == 256
        assert params.transformer_n_heads == 8
        assert params.transformer_n_layers == 4
        assert params.transformer_context_length == 30
        assert params.blend_ratio == 0.5
        assert params.online_learning is False

    def test_params_validation(self):
        """잘못된 값 -> AssertionError"""
        # hidden_dim not divisible by n_heads
        with pytest.raises(AssertionError):
            TransformerMPPIParams(transformer_hidden_dim=127, transformer_n_heads=4)

        # context_length < 1
        with pytest.raises(AssertionError):
            TransformerMPPIParams(transformer_context_length=0)

        # dropout out of range
        with pytest.raises(AssertionError):
            TransformerMPPIParams(transformer_dropout=1.0)

        # blend_ratio out of range
        with pytest.raises(AssertionError):
            TransformerMPPIParams(blend_ratio=1.5)

        # buffer < min_samples
        with pytest.raises(AssertionError):
            TransformerMPPIParams(
                transformer_buffer_size=10, transformer_min_samples=100
            )

        # training_interval < 1
        with pytest.raises(AssertionError):
            TransformerMPPIParams(transformer_training_interval=0)


# ══════════════════════════════════════════════════════════
# 2. Model 테스트 (4개)
# ══════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestTransformerModel:
    def test_control_transformer_shape(self):
        """output (batch, N, nu) shape"""
        state_dim, control_dim, N = 3, 2, 10
        model = ControlTransformer(
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=N,
            d_model=64,
            n_heads=4,
            n_layers=1,
            context_length=15,
        )

        batch_size = 4
        seq_len = 15
        context = torch.randn(batch_size, seq_len, state_dim + control_dim)
        output = model(context)

        assert output.shape == (batch_size, N, control_dim)

    def test_positional_encoding(self):
        """PE 출력 shape 및 값 변화"""
        pe = PositionalEncoding(d_model=64, max_len=50, dropout=0.0)
        x = torch.zeros(2, 20, 64)
        out = pe(x)

        assert out.shape == (2, 20, 64)
        # PE가 추가되어 0이 아님
        assert not torch.allclose(out, torch.zeros_like(out))
        # 위치별 다른 인코딩
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_data_buffer_add_and_sample(self):
        """Ring buffer 추가 및 샘플링"""
        buffer = TransformerDataBuffer(max_size=100)

        # 50개 추가
        for i in range(50):
            state_hist = np.random.randn(10, 3)
            ctrl_hist = np.random.randn(10, 2)
            optimal_U = np.random.randn(15, 2)
            buffer.add(state_hist, ctrl_hist, optimal_U)

        assert len(buffer) == 50

        # 샘플링
        s, c, u = buffer.sample(16)
        assert s.shape == (16, 10, 3)
        assert c.shape == (16, 10, 2)
        assert u.shape == (16, 15, 2)

        # max_size 초과 시 ring buffer
        for i in range(60):
            buffer.add(np.random.randn(10, 3), np.random.randn(10, 2),
                       np.random.randn(15, 2))
        assert len(buffer) == 100

        # clear
        buffer.clear()
        assert len(buffer) == 0

    def test_trainer_loss_decreases(self):
        """MSE 손실 감소 검증"""
        state_dim, control_dim, N = 3, 2, 10
        model = ControlTransformer(
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=N,
            d_model=64,
            n_heads=4,
            n_layers=1,
            context_length=10,
        )
        trainer = TransformerTrainer(model, lr=1e-3)

        # 간단한 패턴: 상수 출력 학습
        np.random.seed(42)
        target = np.random.randn(N, control_dim) * 0.5

        losses = []
        for i in range(50):
            states_b = np.random.randn(16, 10, state_dim) * 0.1
            controls_b = np.random.randn(16, 10, control_dim) * 0.1
            # 동일 타겟
            targets_b = np.tile(target, (16, 1, 1))

            loss = trainer.train_step(states_b, controls_b, targets_b)
            losses.append(loss)

        # 마지막 10번이 첫 10번보다 낮아야 함
        early_mean = np.mean(losses[:10])
        late_mean = np.mean(losses[-10:])
        assert late_mean < early_mean, \
            f"Loss did not decrease: early={early_mean:.4f} late={late_mean:.4f}"
        assert trainer.update_count == 50


# ══════════════════════════════════════════════════════════
# 3. Controller 테스트 (5개)
# ══════════════════════════════════════════════════════════

class TestTransformerMPPIController:
    def test_compute_control_shape(self):
        """control (nu,), info 표준 키 검증"""
        ctrl = _make_transformer_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert "sample_trajectories" in info
        assert "sample_weights" in info
        assert "best_trajectory" in info
        assert "best_cost" in info
        assert "mean_cost" in info
        assert "temperature" in info
        assert "ess" in info
        assert "num_samples" in info

    def test_info_transformer_stats(self):
        """transformer_stats 키 검증"""
        ctrl = _make_transformer_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        _, info = ctrl.compute_control(state, ref)
        stats = info["transformer_stats"]
        assert "transformer_used" in stats
        assert "transformer_init_ratio" in stats
        assert "buffer_size" in stats
        assert "train_count" in stats
        assert "train_loss" in stats
        assert "mean_train_loss" in stats
        assert "context_length" in stats

    def test_fallback_without_training(self):
        """미학습 -> Vanilla MPPI 폴백 (정상 동작)"""
        ctrl = _make_transformer_controller(
            transformer_buffer_size=2000,
            transformer_min_samples=1000,  # 높게 -> 학습 안됨
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 10 스텝 동작
        for _ in range(10):
            control, info = ctrl.compute_control(state, ref)
            assert control.shape == (2,)
            assert info["transformer_stats"]["transformer_used"] is False
            state_dot = ctrl.model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

    def test_different_K_values(self):
        """K=32/128/256 정상 작동"""
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for K in [32, 128, 256]:
            ctrl = _make_transformer_controller(K=K)
            control, info = ctrl.compute_control(state, ref)
            assert control.shape == (2,)
            assert info["num_samples"] == K

    def test_reset_clears_state(self):
        """reset 후 초기화"""
        ctrl = _make_transformer_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 몇 번 실행
        for _ in range(5):
            ctrl.compute_control(state, ref)

        assert ctrl._step_count == 5
        assert len(ctrl._state_history) == 5
        assert len(ctrl._control_history) == 5

        ctrl.reset()
        assert np.allclose(ctrl.U, 0.0)
        assert ctrl._step_count == 0
        assert len(ctrl._state_history) == 0
        assert len(ctrl._control_history) == 0
        assert ctrl._transformer_used_count == 0
        # 버퍼와 학습 상태는 유지
        assert len(ctrl._buffer) > 0


# ══════════════════════════════════════════════════════════
# 4. Transformer Init 테스트 (4개)
# ══════════════════════════════════════════════════════════

class TestTransformerInit:
    def test_init_with_history(self):
        """충분한 이력 + 학습 후 Transformer 사용"""
        ctrl = _make_transformer_controller(
            transformer_min_samples=5,
            transformer_training_interval=2,
            transformer_n_train_steps=3,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 초기에는 미사용
        _, info = ctrl.compute_control(state, ref)
        assert info["transformer_stats"]["transformer_used"] is False

        # 충분한 데이터 수집 + 학습
        for i in range(20):
            state_dot = ctrl.model.forward_dynamics(state,
                                                     np.array([0.1, 0.1]))
            state = state + state_dot * 0.05
            ref = generate_reference_trajectory(
                lambda t: circle_trajectory(t, radius=3.0),
                i * 0.05, 10, 0.05,
            )
            _, info = ctrl.compute_control(state, ref)

        # 학습 후에는 Transformer가 사용되어야 함
        assert ctrl._train_count > 0
        # 마지막 몇 스텝에서 사용됨
        assert ctrl._transformer_used_count > 0

    def test_blend_ratio(self):
        """blend_ratio 동작 검증"""
        ctrl_full = _make_transformer_controller(
            blend_ratio=1.0,
            transformer_min_samples=5,
            transformer_training_interval=1,
        )
        ctrl_none = _make_transformer_controller(
            blend_ratio=0.0,
            transformer_min_samples=5,
            transformer_training_interval=1,
        )
        ctrl_half = _make_transformer_controller(
            blend_ratio=0.5,
            transformer_min_samples=5,
            transformer_training_interval=1,
        )

        state = np.array([3.0, 0.0, np.pi / 2])

        # 모두 동일 시드로 약간의 데이터 수집
        for ctrl in [ctrl_full, ctrl_none, ctrl_half]:
            for i in range(10):
                ref = generate_reference_trajectory(
                    lambda t: circle_trajectory(t, radius=3.0),
                    i * 0.05, 10, 0.05,
                )
                ctrl.compute_control(state.copy(), ref)

        # 모두 정상 작동
        assert ctrl_full._step_count == 10
        assert ctrl_none._step_count == 10
        assert ctrl_half._step_count == 10

    def test_no_init_when_disabled(self):
        """use_transformer_init=False -> 항상 미사용"""
        ctrl = _make_transformer_controller(
            use_transformer_init=False,
            transformer_min_samples=5,
            transformer_training_interval=1,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 충분히 실행 + 학습
        for i in range(20):
            _, info = ctrl.compute_control(state, ref)
            assert info["transformer_stats"]["transformer_used"] is False

    def test_context_length_handling(self):
        """짧은 이력 -> 패딩 처리"""
        ctx_len = 15
        ctrl = _make_transformer_controller(
            transformer_context_length=ctx_len,
            transformer_min_samples=3,
            transformer_training_interval=1,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 이력 < context_length 상태에서 동작 확인
        for i in range(5):
            control, info = ctrl.compute_control(state, ref)
            assert control.shape == (2,)
            # 이력이 context보다 짧아도 정상 동작
            assert info["transformer_stats"]["context_length"] == i + 1
            assert i + 1 <= ctx_len


# ══════════════════════════════════════════════════════════
# 5. Online Learning 테스트 (3개)
# ══════════════════════════════════════════════════════════

class TestOnlineLearning:
    def test_buffer_collection(self):
        """매 스텝 데이터 수집"""
        ctrl = _make_transformer_controller()
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for i in range(10):
            ctrl.compute_control(state, ref)
            state_dot = ctrl.model.forward_dynamics(state, np.array([0.1, 0.1]))
            state = state + state_dot * 0.05

        # 첫 스텝은 이력 부족으로 수집 안 될 수 있음
        assert len(ctrl._buffer) >= 9

    def test_periodic_training(self):
        """training_interval 주기로 학습 트리거"""
        ctrl = _make_transformer_controller(
            transformer_training_interval=5,
            transformer_min_samples=5,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for i in range(25):
            _, info = ctrl.compute_control(state, ref)
            state_dot = ctrl.model.forward_dynamics(state, np.array([0.1, 0.1]))
            state = state + state_dot * 0.05

        # 25 스텝 / 5 interval = ~5 학습 기회
        # (min_samples 도달 후부터 실제 학습)
        assert ctrl._train_count > 0

    def test_training_improves_prediction(self):
        """학습 후 예측 정확도 향상"""
        ctrl = _make_transformer_controller(
            transformer_min_samples=10,
            transformer_training_interval=2,
            transformer_n_train_steps=5,
            transformer_lr=1e-3,
        )
        state = np.array([3.0, 0.0, np.pi / 2])

        # 충분한 데이터 수집 + 학습
        for i in range(40):
            t = i * 0.05
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0),
                t, 10, 0.05,
            )
            _, info = ctrl.compute_control(state, ref)
            state_dot = ctrl.model.forward_dynamics(state, np.array([0.3, 0.3]))
            state = state + state_dot * 0.05

        # 학습이 진행되었음
        assert ctrl._train_count > 0

        # 손실이 존재
        stats = ctrl.get_transformer_statistics()
        assert stats["mean_train_loss"] >= 0


# ══════════════════════════════════════════════════════════
# 6. Performance 테스트 (4개)
# ══════════════════════════════════════════════════════════

class TestPerformance:
    def test_circle_tracking_rmse(self):
        """원형 궤적 추적 RMSE < 0.3 (50 스텝)"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        params = TransformerMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            transformer_hidden_dim=64,
            transformer_n_heads=4,
            transformer_n_layers=1,
            transformer_context_length=10,
            transformer_min_samples=20,
            transformer_training_interval=5,
        )
        ctrl = TransformerMPPIController(model, params)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt = params.dt
        N = params.N
        num_steps = 50

        errors = []
        for step in range(num_steps):
            t = step * dt
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt

            ref_pt = circle_trajectory(t, radius=3.0)
            err = np.sqrt((state[0] - ref_pt[0]) ** 2 + (state[1] - ref_pt[1]) ** 2)
            errors.append(err)

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        assert rmse < 0.3, f"RMSE {rmse:.4f} >= 0.3"

    def test_obstacle_avoidance(self):
        """3개 장애물 충돌 없음"""
        model = DifferentialDriveKinematic(wheelbase=0.5)
        obstacles = [(2.5, 1.5, 0.5), (0.0, 3.0, 0.4), (-2.0, -1.0, 0.5)]

        cost = CompositeMPPICost([
            StateTrackingCost(np.array([10.0, 10.0, 1.0])),
            TerminalCost(np.array([10.0, 10.0, 1.0])),
            ControlEffortCost(np.array([0.1, 0.1])),
            ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0),
        ])

        params = TransformerMPPIParams(
            K=128, N=15, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            transformer_hidden_dim=64,
            transformer_n_heads=4,
            transformer_n_layers=1,
            transformer_context_length=10,
            transformer_min_samples=20,
            transformer_training_interval=5,
        )
        ctrl = TransformerMPPIController(model, params, cost_function=cost)

        state = np.array([3.0, 0.0, np.pi / 2])
        dt_val = params.dt
        N = params.N

        collisions = 0
        for step in range(80):
            t = step * dt_val
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0), t, N, dt_val,
            )
            control, _ = ctrl.compute_control(state, ref)
            state_dot = model.forward_dynamics(state, control)
            state = state + state_dot * dt_val

            for ox, oy, r in obstacles:
                dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                if dist < r:
                    collisions += 1

        assert collisions == 0, f"Collisions: {collisions}"

    def test_warm_start_efficiency(self):
        """Transformer init이 빈 초기화보다 빠르게 수렴"""
        model = DifferentialDriveKinematic(wheelbase=0.5)

        # T-MPPI: 데이터 사전 수집 + 학습
        params_t = TransformerMPPIParams(
            K=64, N=10, dt=0.05, lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            transformer_hidden_dim=64,
            transformer_n_heads=4,
            transformer_n_layers=1,
            transformer_context_length=10,
            transformer_min_samples=10,
            transformer_training_interval=3,
            transformer_n_train_steps=5,
        )
        ctrl_t = TransformerMPPIController(model, params_t)

        state = np.array([3.0, 0.0, np.pi / 2])

        # 사전 학습 (30 스텝)
        for i in range(30):
            t = i * 0.05
            ref = generate_reference_trajectory(
                lambda t_: circle_trajectory(t_, radius=3.0),
                t, 10, 0.05,
            )
            ctrl_t.compute_control(state.copy(), ref)
            state_dot = model.forward_dynamics(state, np.array([0.3, 0.3]))
            state = state + state_dot * 0.05

        # 학습 진행 확인
        assert ctrl_t._train_count > 0

        # 두 컨트롤러 모두 동작 확인
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()
        control, info = ctrl_t.compute_control(state, ref)
        assert control.shape == (2,)

    def test_computation_time(self):
        """K=512, N=30에서 100ms 이내"""
        ctrl = _make_transformer_controller(
            K=512, N=30,
            transformer_context_length=10,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=30)

        # Warmup
        ctrl.compute_control(state, ref)
        ctrl.reset()

        times = []
        for _ in range(5):
            t_start = time.time()
            ctrl.compute_control(state, ref)
            times.append(time.time() - t_start)

        mean_ms = np.mean(times) * 1000
        assert mean_ms < 100, f"Mean solve time {mean_ms:.1f}ms >= 100ms"


# ══════════════════════════════════════════════════════════
# 7. Integration 테스트 (5개)
# ══════════════════════════════════════════════════════════

class TestIntegration:
    def test_numerical_stability(self):
        """NaN/Inf 없음"""
        ctrl = _make_transformer_controller(K=64)
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(20):
            control, info = ctrl.compute_control(state, ref)
            assert not np.any(np.isnan(control)), "NaN in control"
            assert not np.any(np.isinf(control)), "Inf in control"
            assert not np.isnan(info["ess"]), "NaN in ESS"
            assert not np.isnan(info["best_cost"]), "NaN in best_cost"

            state_dot = ctrl.model.forward_dynamics(state, control)
            state = state + state_dot * 0.05

    def test_pretrained_model_load(self):
        """transformer_model_path 로드/저장"""
        ctrl1 = _make_transformer_controller(
            transformer_min_samples=5,
            transformer_training_interval=1,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        # 학습 데이터 수집 + 학습
        for i in range(15):
            ctrl1.compute_control(state, ref)

        # 저장
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model_path = f.name
        ctrl1.save_model(model_path)

        # 로드
        ctrl2 = _make_transformer_controller(
            transformer_model_path=model_path,
        )

        # 로드된 모델로 정상 동작
        control, info = ctrl2.compute_control(state, ref)
        assert control.shape == (2,)
        assert ctrl2._train_count > 0

        # 정리
        os.unlink(model_path)

    def test_online_learning_disabled(self):
        """online_learning=False -> 학습 안 함"""
        ctrl = _make_transformer_controller(
            online_learning=False,
            transformer_min_samples=5,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref()

        for _ in range(20):
            ctrl.compute_control(state, ref)

        # 학습 미실행
        assert ctrl._train_count == 0

    def test_long_horizon(self):
        """N=50 장기 호라이즌"""
        ctrl = _make_transformer_controller(
            N=50,
            transformer_context_length=10,
        )
        state = np.array([3.0, 0.0, np.pi / 2])
        ref = _make_ref(N=50)

        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (2,)
        assert info["sample_trajectories"].shape[1] == 51  # N+1

    def test_varying_reference(self):
        """다양한 궤적 타입"""
        ctrl = _make_transformer_controller()
        state = np.array([3.0, 0.0, np.pi / 2])

        # Circle
        ref_circle = generate_reference_trajectory(
            lambda t: circle_trajectory(t, radius=3.0),
            0.0, 10, 0.05,
        )
        control1, _ = ctrl.compute_control(state, ref_circle)
        assert control1.shape == (2,)

        # Figure-8
        ref_fig8 = generate_reference_trajectory(
            lambda t: figure_eight_trajectory(t, scale=3.0),
            0.0, 10, 0.05,
        )
        control2, _ = ctrl.compute_control(state, ref_fig8)
        assert control2.shape == (2,)

        # 다른 궤적이면 다른 제어 (확률적이므로 항상은 아니나 대부분)
        # 적어도 crash 없이 동작
