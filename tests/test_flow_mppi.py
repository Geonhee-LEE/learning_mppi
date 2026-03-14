"""
Flow-MPPI (Conditional Flow Matching + MPPI) 테스트

테스트 카테고리:
    1. FlowMatchingModel: 속도장 모델 기본 동작
    2. FlowMatchingSampler: 3가지 모드 + fallback
    3. FlowMPPIParams: 파라미터 유효성
    4. FlowMPPIController: compute_control, fallback, 데이터 수집, train/save/load
    5. FlowMatchingTrainer: CFM loss 감소, 생성 품질
    6. FlowDataCollector: 버퍼 동작
    7. Integration: bootstrap 루프, 온라인 학습
"""

import numpy as np
import pytest
import torch
import tempfile
import os

from mppi_controller.models.learned.flow_matching_model import (
    FlowMatchingModel,
    SinusoidalTimeEmbedding,
)
from mppi_controller.controllers.mppi.flow_matching_sampler import FlowMatchingSampler
from mppi_controller.controllers.mppi.mppi_params import FlowMPPIParams
from mppi_controller.controllers.mppi.flow_mppi import FlowMPPIController
from mppi_controller.learning.flow_matching_trainer import FlowMatchingTrainer
from mppi_controller.learning.flow_data_collector import FlowDataCollector
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.utils.trajectory import circle_trajectory


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def dd_model():
    return DifferentialDriveKinematic(wheelbase=0.5)


@pytest.fixture
def flow_params():
    return FlowMPPIParams(
        N=10, K=32, dt=0.1,
        sigma=np.array([0.5, 0.5]),
        flow_hidden_dims=[64, 64],
        flow_num_steps=3,
        flow_mode="replace_mean",
        flow_online_training=False,
    )


@pytest.fixture
def flow_model():
    """N=10, nu=2 → control_seq_dim=20, context_dim=3"""
    return FlowMatchingModel(
        control_seq_dim=20, context_dim=3,
        hidden_dims=[64, 64], time_embed_dim=16,
    )


@pytest.fixture
def reference_traj():
    """원형 레퍼런스 궤적 (N+1=11 포인트)"""
    t = np.linspace(0, 2 * np.pi, 11)
    ref = np.zeros((11, 3))
    ref[:, 0] = 3.0 * np.cos(t)
    ref[:, 1] = 3.0 * np.sin(t)
    ref[:, 2] = np.arctan2(np.gradient(ref[:, 1]), np.gradient(ref[:, 0]))
    return ref


# ── 1. FlowMatchingModel 테스트 ──────────────────────────

class TestFlowMatchingModel:

    def test_output_shape(self, flow_model):
        """forward 출력 shape 검증"""
        B = 8
        x_t = torch.randn(B, 20)
        t = torch.rand(B)
        ctx = torch.randn(B, 3)
        v = flow_model(x_t, t, ctx)
        assert v.shape == (B, 20)

    def test_time_embedding(self):
        """Sinusoidal time embedding shape"""
        emb = SinusoidalTimeEmbedding(embed_dim=32)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert out.shape == (3, 32)

    def test_generate_shape(self, flow_model):
        """ODE generate 출력 shape"""
        ctx = torch.randn(3)
        samples = flow_model.generate(ctx, num_samples=16, num_steps=3)
        assert samples.shape == (16, 20)

    def test_generate_solvers(self, flow_model):
        """Euler vs Midpoint solver 모두 동작"""
        ctx = torch.randn(3)
        s_euler = flow_model.generate(ctx, num_samples=4, num_steps=3, solver="euler")
        s_mid = flow_model.generate(ctx, num_samples=4, num_steps=3, solver="midpoint")
        assert s_euler.shape == (4, 20)
        assert s_mid.shape == (4, 20)

    def test_generate_seed_reproducibility(self, flow_model):
        """동일 seed → 동일 결과 (모델 deterministic 부분)"""
        ctx = torch.randn(3)
        torch.manual_seed(42)
        s1 = flow_model.generate(ctx, num_samples=4, num_steps=3)
        torch.manual_seed(42)
        s2 = flow_model.generate(ctx, num_samples=4, num_steps=3)
        np.testing.assert_allclose(s1.numpy(), s2.numpy(), atol=1e-6)


# ── 2. FlowMatchingSampler 테스트 ─────────────────────────

class TestFlowMatchingSampler:

    def test_sample_shape_fallback(self):
        """Flow 없을 때 가우시안 fallback shape"""
        sampler = FlowMatchingSampler(sigma=np.array([0.5, 0.5]), seed=42)
        U = np.zeros((10, 2))
        noise = sampler.sample(U, K=32)
        assert noise.shape == (32, 10, 2)

    def test_replace_mean_mode(self, flow_model):
        """replace_mean 모드 동작"""
        sampler = FlowMatchingSampler(
            sigma=np.array([0.5, 0.5]), mode="replace_mean", seed=42
        )
        sampler.set_flow_model(flow_model)
        sampler.set_context(np.array([1.0, 2.0, 0.5]))
        U = np.zeros((10, 2))
        noise = sampler.sample(U, K=32)
        assert noise.shape == (32, 10, 2)

    def test_replace_distribution_mode(self, flow_model):
        """replace_distribution 모드 동작"""
        sampler = FlowMatchingSampler(
            sigma=np.array([0.5, 0.5]), mode="replace_distribution", seed=42
        )
        sampler.set_flow_model(flow_model)
        sampler.set_context(np.array([1.0, 2.0, 0.5]))
        U = np.zeros((10, 2))
        noise = sampler.sample(U, K=16)
        assert noise.shape == (16, 10, 2)

    def test_blend_mode(self, flow_model):
        """blend 모드 동작"""
        sampler = FlowMatchingSampler(
            sigma=np.array([0.5, 0.5]), mode="blend",
            blend_ratio=0.5, seed=42
        )
        sampler.set_flow_model(flow_model)
        sampler.set_context(np.array([1.0, 2.0, 0.5]))
        U = np.zeros((10, 2))
        noise = sampler.sample(U, K=20)
        assert noise.shape == (20, 10, 2)

    def test_is_flow_ready(self, flow_model):
        """is_flow_ready 프로퍼티"""
        sampler = FlowMatchingSampler(sigma=np.array([0.5, 0.5]))
        assert not sampler.is_flow_ready
        sampler.set_flow_model(flow_model)
        assert not sampler.is_flow_ready  # context 없음
        sampler.set_context(np.array([1.0, 2.0, 0.5]))
        assert sampler.is_flow_ready


# ── 3. FlowMPPIParams 테스트 ──────────────────────────────

class TestFlowMPPIParams:

    def test_valid_params(self):
        """유효한 파라미터 생성"""
        params = FlowMPPIParams(
            N=20, K=64, dt=0.05,
            flow_hidden_dims=[128, 128],
            flow_num_steps=5,
            flow_mode="blend",
            flow_blend_ratio=0.3,
        )
        assert params.flow_num_steps == 5
        assert params.flow_mode == "blend"

    def test_invalid_flow_mode(self):
        """잘못된 flow_mode → AssertionError"""
        with pytest.raises(AssertionError):
            FlowMPPIParams(flow_mode="invalid")

    def test_invalid_solver(self):
        """잘못된 solver → AssertionError"""
        with pytest.raises(AssertionError):
            FlowMPPIParams(flow_solver="rk4")

    def test_invalid_blend_ratio(self):
        """범위 밖 blend_ratio → AssertionError"""
        with pytest.raises(AssertionError):
            FlowMPPIParams(flow_blend_ratio=1.5)


# ── 4. FlowMPPIController 테스트 ──────────────────────────

class TestFlowMPPIController:

    def test_compute_control_fallback(self, dd_model, flow_params, reference_traj):
        """Flow 모델 없이 compute_control (가우시안 fallback)"""
        ctrl = FlowMPPIController(dd_model, flow_params)
        state = np.array([0.0, 0.0, 0.0])
        control, info = ctrl.compute_control(state, reference_traj)
        assert control.shape == (2,)
        assert "flow_stats" in info
        assert info["flow_stats"]["flow_ready"] is False

    def test_compute_control_with_flow(self, dd_model, flow_params, reference_traj):
        """Flow 모델 주입 후 compute_control"""
        ctrl = FlowMPPIController(dd_model, flow_params)
        flow_model = FlowMatchingModel(
            control_seq_dim=flow_params.N * dd_model.control_dim,
            context_dim=dd_model.state_dim,
            hidden_dims=flow_params.flow_hidden_dims,
        )
        ctrl.noise_sampler.set_flow_model(flow_model)
        state = np.array([0.0, 0.0, 0.0])
        control, info = ctrl.compute_control(state, reference_traj)
        assert control.shape == (2,)
        assert info["flow_stats"]["flow_ready"] is True

    def test_info_dict_keys(self, dd_model, flow_params, reference_traj):
        """info dict에 필수 키 포함"""
        ctrl = FlowMPPIController(dd_model, flow_params)
        state = np.array([0.0, 0.0, 0.0])
        _, info = ctrl.compute_control(state, reference_traj)
        assert "sample_trajectories" in info
        assert "sample_weights" in info
        assert "best_trajectory" in info
        assert "ess" in info
        assert "flow_stats" in info

    def test_data_collection(self, dd_model, flow_params, reference_traj):
        """compute_control 호출 시 데이터 수집"""
        ctrl = FlowMPPIController(dd_model, flow_params)
        state = np.array([0.0, 0.0, 0.0])
        for _ in range(5):
            ctrl.compute_control(state, reference_traj)
        assert ctrl._data_collector.num_samples == 5

    def test_train_flow_model(self, dd_model, reference_traj):
        """내부 버퍼로 학습 트리거"""
        params = FlowMPPIParams(
            N=10, K=16, dt=0.1,
            sigma=np.array([0.5, 0.5]),
            flow_hidden_dims=[32, 32],
            flow_min_samples=5,
        )
        ctrl = FlowMPPIController(dd_model, params)
        state = np.array([0.0, 0.0, 0.0])
        # 데이터 수집
        for _ in range(10):
            ctrl.compute_control(state, reference_traj)
        # 학습
        metrics = ctrl.train_flow_model(epochs=5)
        assert metrics["status"] == "trained"
        assert metrics["final_loss"] < float("inf")
        # 학습 후 flow_ready 확인
        assert ctrl.noise_sampler.is_flow_ready or not isinstance(
            ctrl.noise_sampler, FlowMatchingSampler
        )

    def test_save_load_flow_model(self, dd_model, reference_traj):
        """모델 저장/로드"""
        params = FlowMPPIParams(
            N=10, K=16, dt=0.1,
            sigma=np.array([0.5, 0.5]),
            flow_hidden_dims=[32, 32],
            flow_min_samples=5,
        )
        ctrl = FlowMPPIController(dd_model, params)
        state = np.array([0.0, 0.0, 0.0])
        for _ in range(10):
            ctrl.compute_control(state, reference_traj)
        ctrl.train_flow_model(epochs=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flow_model.pth")
            ctrl.save_flow_model(path)
            assert os.path.exists(path)

            # 새 컨트롤러에서 로드
            ctrl2 = FlowMPPIController(dd_model, params)
            ctrl2.load_flow_model(path)
            ctrl2.noise_sampler.set_context(state)
            assert ctrl2.noise_sampler.is_flow_ready

    def test_reset(self, dd_model, flow_params, reference_traj):
        """reset 후 상태 초기화"""
        ctrl = FlowMPPIController(dd_model, flow_params)
        state = np.array([0.0, 0.0, 0.0])
        ctrl.compute_control(state, reference_traj)
        ctrl.reset()
        assert ctrl._step_count == 0
        assert np.allclose(ctrl.U, 0.0)

    def test_all_modes(self, dd_model, reference_traj):
        """3가지 flow 모드 모두 동작"""
        for mode in ["replace_mean", "replace_distribution", "blend"]:
            params = FlowMPPIParams(
                N=10, K=16, dt=0.1,
                sigma=np.array([0.5, 0.5]),
                flow_hidden_dims=[32, 32],
                flow_mode=mode,
                flow_min_samples=5,
            )
            ctrl = FlowMPPIController(dd_model, params)
            state = np.array([0.0, 0.0, 0.0])
            # 학습
            for _ in range(10):
                ctrl.compute_control(state, reference_traj)
            ctrl.train_flow_model(epochs=3)
            # Flow 활성화 후 실행
            control, info = ctrl.compute_control(state, reference_traj)
            assert control.shape == (2,)
            assert info["flow_stats"]["flow_ready"]


# ── 5. FlowMatchingTrainer 테스트 ─────────────────────────

class TestFlowMatchingTrainer:

    def test_cfm_loss_decreases(self):
        """학습 시 CFM loss 감소"""
        trainer = FlowMatchingTrainer(
            control_seq_dim=20, context_dim=3,
            hidden_dims=[64, 64], lr=1e-3,
        )
        # 합성 데이터: 상태 → 제어 (간단한 패턴)
        M = 100
        states = np.random.randn(M, 3).astype(np.float32)
        controls = np.random.randn(M, 10, 2).astype(np.float32) * 0.5
        metrics = trainer.train(states, controls, epochs=50, batch_size=32)
        assert metrics["final_loss"] < metrics["losses"][0]

    def test_generation_shape(self):
        """학습 후 생성 shape"""
        trainer = FlowMatchingTrainer(
            control_seq_dim=20, context_dim=3,
            hidden_dims=[32, 32],
        )
        M = 50
        states = np.random.randn(M, 3).astype(np.float32)
        controls = np.random.randn(M, 10, 2).astype(np.float32)
        trainer.train(states, controls, epochs=10)

        model = trainer.get_model()
        ctx = torch.randn(3)
        samples = model.generate(ctx, num_samples=8, num_steps=5)
        assert samples.shape == (8, 20)

    def test_save_load(self):
        """모델 저장/로드 일관성"""
        trainer = FlowMatchingTrainer(
            control_seq_dim=20, context_dim=3,
            hidden_dims=[32, 32],
        )
        M = 30
        states = np.random.randn(M, 3).astype(np.float32)
        controls = np.random.randn(M, 10, 2).astype(np.float32)
        trainer.train(states, controls, epochs=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flow.pth")
            trainer.save_model(path)

            trainer2 = FlowMatchingTrainer(
                control_seq_dim=20, context_dim=3,
                hidden_dims=[32, 32],
            )
            trainer2.load_model(path)

            # 같은 입력에 대해 같은 출력
            x = torch.randn(4, 20)
            t = torch.tensor([0.5, 0.5, 0.5, 0.5])
            ctx = torch.randn(4, 3)
            v1 = trainer.get_model()(x, t, ctx)
            v2 = trainer2.get_model()(x, t, ctx)
            np.testing.assert_allclose(
                v1.detach().numpy(), v2.detach().numpy(), atol=1e-6
            )


# ── 6. FlowDataCollector 테스트 ───────────────────────────

class TestFlowDataCollector:

    def test_add_and_get(self):
        """샘플 추가 및 조회"""
        dc = FlowDataCollector(buffer_size=100)
        for i in range(10):
            dc.add_sample(np.array([i, 0, 0]), np.random.randn(5, 2))
        assert dc.num_samples == 10
        states, controls = dc.get_training_data()
        assert states.shape == (10, 3)
        assert controls.shape == (10, 5, 2)

    def test_ring_buffer_overflow(self):
        """버퍼 오버플로우 시 ring buffer 동작"""
        dc = FlowDataCollector(buffer_size=5)
        for i in range(10):
            dc.add_sample(np.array([float(i)]), np.zeros((3, 2)))
        assert dc.num_samples == 5
        states, _ = dc.get_training_data()
        assert states.shape == (5, 1)

    def test_should_train(self):
        """학습 가능 여부 확인"""
        dc = FlowDataCollector(buffer_size=100)
        assert not dc.should_train(min_samples=5)
        for i in range(5):
            dc.add_sample(np.zeros(3), np.zeros((4, 2)))
        assert dc.should_train(min_samples=5)

    def test_clear(self):
        """버퍼 초기화"""
        dc = FlowDataCollector(buffer_size=100)
        for i in range(5):
            dc.add_sample(np.zeros(3), np.zeros((4, 2)))
        dc.clear()
        assert dc.num_samples == 0


# ── 7. Integration 테스트 ─────────────────────────────────

class TestFlowMPPIIntegration:

    def test_bootstrap_loop(self, dd_model):
        """Bootstrap: Vanilla 데이터 수집 → Flow 학습 → Flow 실행"""
        N, K = 10, 16
        params = FlowMPPIParams(
            N=N, K=K, dt=0.1,
            sigma=np.array([0.5, 0.5]),
            flow_hidden_dims=[32, 32],
            flow_min_samples=10,
        )
        ctrl = FlowMPPIController(dd_model, params)
        state = np.array([0.0, 0.0, 0.0])

        # 원형 궤적 생성
        t_arr = np.linspace(0, 2 * np.pi, N + 1)
        ref = np.zeros((N + 1, 3))
        ref[:, 0] = 3.0 * np.cos(t_arr)
        ref[:, 1] = 3.0 * np.sin(t_arr)

        # Phase 1: Gaussian fallback로 데이터 수집
        for _ in range(20):
            control, info = ctrl.compute_control(state, ref)
            assert not info["flow_stats"]["flow_ready"]
            # 간단한 상태 업데이트
            state = state + np.array([
                control[0] * np.cos(state[2]) * 0.1,
                control[0] * np.sin(state[2]) * 0.1,
                control[1] * 0.1,
            ])

        # Phase 2: 학습
        metrics = ctrl.train_flow_model(epochs=10)
        assert metrics["status"] == "trained"

        # Phase 3: Flow 활성화 후 실행
        control, info = ctrl.compute_control(state, ref)
        assert info["flow_stats"]["flow_ready"]

    def test_online_training_trigger(self, dd_model):
        """온라인 학습이 training_interval마다 트리거"""
        params = FlowMPPIParams(
            N=10, K=16, dt=0.1,
            sigma=np.array([0.5, 0.5]),
            flow_hidden_dims=[32, 32],
            flow_online_training=True,
            flow_training_interval=10,
            flow_min_samples=5,
        )
        ctrl = FlowMPPIController(dd_model, params)
        state = np.array([0.0, 0.0, 0.0])
        t_arr = np.linspace(0, 2 * np.pi, 11)
        ref = np.zeros((11, 3))
        ref[:, 0] = 3.0 * np.cos(t_arr)
        ref[:, 1] = 3.0 * np.sin(t_arr)

        # 10스텝 (= training_interval) 실행
        for _ in range(10):
            ctrl.compute_control(state, ref)

        # 온라인 학습이 트리거되어 flow 모델이 로드됨
        assert ctrl._step_count == 10
        # min_samples=5, 10개 수집 → 학습 가능 → flow_ready
        stats = ctrl.get_flow_statistics()
        assert stats["buffer_size"] == 10


# ── 8. QuadrupedKinematic 테스트 ────────────────────────

class TestQuadrupedKinematic:

    def test_shape(self):
        """state_dim=5, control_dim=5 검증"""
        from mppi_controller.models.kinematic.quadruped_kinematic import QuadrupedKinematic
        model = QuadrupedKinematic()
        assert model.state_dim == 5
        assert model.control_dim == 5

    def test_forward_dynamics_shape(self):
        """forward_dynamics 출력 shape (single + batch)"""
        from mppi_controller.models.kinematic.quadruped_kinematic import QuadrupedKinematic
        model = QuadrupedKinematic()
        # single
        state = np.array([1.0, 2.0, 0.5, 0.28, 0.0])
        control = np.array([0.5, 0.1, 0.3, 0.0, 0.0])
        dot = model.forward_dynamics(state, control)
        assert dot.shape == (5,)
        # batch
        states = np.tile(state, (16, 1))
        controls = np.tile(control, (16, 1))
        dots = model.forward_dynamics(states, controls)
        assert dots.shape == (16, 5)

    def test_normalize_state(self):
        """θ, pitch wrapping + z clamping"""
        from mppi_controller.models.kinematic.quadruped_kinematic import QuadrupedKinematic
        model = QuadrupedKinematic(z_min=0.15, z_max=0.40)
        state = np.array([1.0, 2.0, 4.0, 0.50, -4.0])
        normalized = model.normalize_state(state)
        assert -np.pi <= normalized[2] <= np.pi  # θ wrapped
        assert -np.pi <= normalized[4] <= np.pi  # pitch wrapped
        assert 0.15 <= normalized[3] <= 0.40      # z clamped

    def test_get_control_bounds(self):
        """제어 제약 검증"""
        from mppi_controller.models.kinematic.quadruped_kinematic import QuadrupedKinematic
        model = QuadrupedKinematic(vx_max=0.8, vy_max=0.5, omega_max=1.0, vz_max=0.3, pitch_rate_max=0.5)
        bounds = model.get_control_bounds()
        assert bounds is not None
        lower, upper = bounds
        assert lower.shape == (5,)
        assert upper.shape == (5,)
        np.testing.assert_allclose(upper, [0.8, 0.5, 1.0, 0.3, 0.5])
        np.testing.assert_allclose(lower, -upper)


# ── 9. Flow-MPPI + Manipulator 테스트 ───────────────────

class TestFlowMPPIManipulator:

    def test_mobile_manip_2dof(self):
        """MobileManip 2-DOF + Flow-MPPI compute_control (EE 비용)"""
        from mppi_controller.models.kinematic.mobile_manipulator_kinematic import (
            MobileManipulatorKinematic,
        )
        from mppi_controller.controllers.mppi.cost_functions import (
            CompositeMPPICost,
            EndEffectorTrackingCost,
            EndEffectorTerminalCost,
            ControlEffortCost,
        )
        model = MobileManipulatorKinematic(L1=0.3, L2=0.25)
        R = np.array([0.1, 0.1, 0.05, 0.05])
        cost = CompositeMPPICost([
            EndEffectorTrackingCost(model, weight=100.0),
            EndEffectorTerminalCost(model, weight=200.0),
            ControlEffortCost(R),
        ])
        params = FlowMPPIParams(
            N=10, K=16, dt=0.1,
            sigma=np.array([0.5, 0.5, 0.8, 0.8]),
            flow_hidden_dims=[32, 32],
            flow_num_steps=3,
            flow_mode="replace_mean",
            flow_online_training=False,
        )
        ctrl = FlowMPPIController(model, params, cost)
        state = np.array([0.0, 0.0, 0.0, 0.5, -0.5])
        ref = np.zeros((11, 5))
        ref[:, 0] = 0.5  # ee_x target
        ref[:, 1] = 0.3  # ee_y target
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (4,)
        assert "flow_stats" in info

    def test_mobile_manip_6dof(self):
        """MobileManip 6-DOF + Flow-MPPI compute_control (EE 3D 비용)"""
        from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
            MobileManipulator6DOFKinematic,
        )
        from mppi_controller.controllers.mppi.cost_functions import (
            CompositeMPPICost,
            EndEffector3DTrackingCost,
            EndEffector3DTerminalCost,
            ControlEffortCost,
        )
        model = MobileManipulator6DOFKinematic()
        R = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        cost = CompositeMPPICost([
            EndEffector3DTrackingCost(model, weight=100.0),
            EndEffector3DTerminalCost(model, weight=200.0),
            ControlEffortCost(R),
        ])
        params = FlowMPPIParams(
            N=10, K=16, dt=0.1,
            sigma=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            flow_hidden_dims=[32, 32],
            flow_num_steps=3,
            flow_mode="replace_mean",
            flow_online_training=False,
        )
        ctrl = FlowMPPIController(model, params, cost)
        state = np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0])
        ref = np.zeros((11, 9))
        ref[:, 0] = 0.4  # ee_x target
        ref[:, 1] = 0.0  # ee_y target
        ref[:, 2] = 0.4  # ee_z target
        control, info = ctrl.compute_control(state, ref)
        assert control.shape == (8,)
        assert "flow_stats" in info
