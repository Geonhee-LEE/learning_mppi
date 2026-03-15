"""
Diffusion-MPPI 테스트
"""

import numpy as np
import pytest
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.diffusion_mppi import DiffusionMPPIController
from mppi_controller.controllers.mppi.mppi_params import DiffusionMPPIParams
from mppi_controller.controllers.mppi.diffusion_sampler import (
    DDIMSampler,
    DDPMSampler,
    _make_cosine_schedule,
    _make_linear_schedule,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    return DifferentialDriveKinematic()


@pytest.fixture
def params():
    return DiffusionMPPIParams(
        N=10, K=32, dt=0.05,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        diff_ddim_steps=3,
        diff_T=100,
        diff_beta_schedule="cosine",
        diff_mode="replace",
        diff_online_training=False,
    )


@pytest.fixture
def state():
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture
def reference():
    N = 10
    ref = np.zeros((N + 1, 3))
    ref[:, 0] = np.linspace(0, 1.0, N + 1)
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionMPPIParams 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffusionMPPIParams:
    def test_defaults(self):
        p = DiffusionMPPIParams(sigma=np.array([0.5, 0.5]), Q=np.array([1.0, 1.0, 0.1]), R=np.array([0.1, 0.1]))
        assert p.diff_ddim_steps == 5
        assert p.diff_T == 1000
        assert p.diff_beta_schedule == "cosine"
        assert p.diff_mode == "replace"

    def test_invalid_beta_schedule(self):
        with pytest.raises(AssertionError):
            DiffusionMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                diff_beta_schedule="invalid",
            )

    def test_invalid_mode(self):
        with pytest.raises(AssertionError):
            DiffusionMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                diff_mode="invalid_mode",
            )

    def test_invalid_blend_ratio(self):
        with pytest.raises(AssertionError):
            DiffusionMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                diff_blend_ratio=1.5,
            )

    def test_all_modes(self):
        for mode in ("replace", "blend"):
            p = DiffusionMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                diff_mode=mode,
            )
            assert p.diff_mode == mode


# ─────────────────────────────────────────────────────────────────────────────
# Noise Schedule 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseSchedule:
    def test_cosine_schedule_shape(self):
        ab = _make_cosine_schedule(1000)
        assert ab.shape == (1001,)

    def test_cosine_schedule_decreasing(self):
        ab = _make_cosine_schedule(100)
        assert ab[0] > ab[-1]
        assert np.all(np.diff(ab) <= 0)

    def test_cosine_schedule_range(self):
        ab = _make_cosine_schedule(100)
        assert np.all(ab >= 1e-5)
        assert np.all(ab <= 1.0 - 1e-5)

    def test_linear_schedule_shape(self):
        ab = _make_linear_schedule(1000)
        assert ab.shape == (1001,)

    def test_linear_schedule_decreasing(self):
        ab = _make_linear_schedule(100)
        assert np.all(np.diff(ab) <= 0)


# ─────────────────────────────────────────────────────────────────────────────
# DDIMSampler 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestDDIMSampler:
    def test_output_shape_fallback(self):
        """미학습 시 가우시안 fallback"""
        sigma = np.array([0.5, 0.5])
        sampler = DDIMSampler(sigma=sigma, ddim_steps=3, T=100)
        U = np.zeros((10, 2))
        noise = sampler.sample(U, K=32)
        assert noise.shape == (32, 10, 2)

    def test_fallback_gaussian_distribution(self):
        """Fallback 노이즈가 대략 가우시안"""
        sigma = np.array([1.0, 1.0])
        sampler = DDIMSampler(sigma=sigma, ddim_steps=3, T=100)
        U = np.zeros((5, 2))
        noises = [sampler.sample(U, K=500) for _ in range(5)]
        all_noise = np.concatenate(noises, axis=0)  # (2500, 5, 2)
        std = np.std(all_noise)
        assert 0.5 < std < 2.0  # 대략 σ=1 근처

    def test_set_context(self):
        sigma = np.array([0.5, 0.5])
        sampler = DDIMSampler(sigma=sigma)
        sampler.set_context(np.array([1.0, 2.0, 0.5]))
        assert sampler._context is not None
        np.testing.assert_array_equal(sampler._context, [1.0, 2.0, 0.5])

    def test_blend_mode_shape(self):
        sigma = np.array([0.5, 0.5])
        sampler = DDIMSampler(sigma=sigma, mode="blend", ddim_steps=2, T=50)
        U = np.zeros((8, 2))
        noise = sampler.sample(U, K=16)
        assert noise.shape == (16, 8, 2)

    def test_ddim_timesteps(self):
        """DDIM 타임스텝이 역순으로 설정됨"""
        sampler = DDIMSampler(sigma=np.array([0.5]), ddim_steps=5, T=100)
        assert len(sampler.ddim_timesteps) == 5
        # 역순 확인 (T→0)
        assert sampler.ddim_timesteps[0] > sampler.ddim_timesteps[-1]


class TestDDPMSampler:
    def test_output_shape(self):
        sigma = np.array([0.5, 0.5])
        sampler = DDPMSampler(sigma=sigma, ddim_steps=3, T=50)
        U = np.zeros((5, 2))
        noise = sampler.sample(U, K=8)
        assert noise.shape == (8, 5, 2)

    def test_probabilistic_sampling(self):
        """DDPM은 확률적 → 같은 입력에도 다른 출력"""
        sigma = np.array([0.5, 0.5])
        sampler = DDPMSampler(sigma=sigma, ddim_steps=2, T=20)
        U = np.zeros((5, 2))
        n1 = sampler.sample(U, K=8)
        n2 = sampler.sample(U, K=8)
        # 동일하지 않아야 함 (확률적)
        # (이론적으로 같을 확률은 0이므로 항상 다름)
        assert not np.allclose(n1, n2, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionMPPIController 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffusionMPPIController:
    def test_instantiation(self, model, params):
        ctrl = DiffusionMPPIController(model, params)
        assert ctrl is not None
        assert not ctrl.is_diffusion_trained

    def test_control_output_shape(self, model, params, state, reference):
        ctrl = DiffusionMPPIController(model, params)
        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)  # [v, ω]

    def test_control_output_finite(self, model, params, state, reference):
        ctrl = DiffusionMPPIController(model, params)
        u, info = ctrl.compute_control(state, reference)
        assert np.all(np.isfinite(u))

    def test_info_keys(self, model, params, state, reference):
        ctrl = DiffusionMPPIController(model, params)
        u, info = ctrl.compute_control(state, reference)
        assert "sample_trajectories" in info
        assert "sample_weights" in info
        assert "best_trajectory" in info
        assert "diffusion_is_trained" in info
        assert "diffusion_step_count" in info

    def test_info_not_trained_initially(self, model, params, state, reference):
        ctrl = DiffusionMPPIController(model, params)
        u, info = ctrl.compute_control(state, reference)
        assert info["diffusion_is_trained"] == False

    def test_sample_trajectories_shape(self, model, params, state, reference):
        params.K = 16
        ctrl = DiffusionMPPIController(model, params)
        u, info = ctrl.compute_control(state, reference)
        assert info["sample_trajectories"].shape == (16, 11, 3)

    def test_sample_weights_sum_to_one(self, model, params, state, reference):
        ctrl = DiffusionMPPIController(model, params)
        u, info = ctrl.compute_control(state, reference)
        assert np.isclose(info["sample_weights"].sum(), 1.0, atol=1e-6)

    def test_sequential_calls(self, model, params, state, reference):
        """연속 호출 정상 동작"""
        ctrl = DiffusionMPPIController(model, params)
        for _ in range(5):
            u, info = ctrl.compute_control(state, reference)
            assert np.all(np.isfinite(u))

    def test_blend_mode(self, model, state, reference):
        """blend 모드 동작"""
        params = DiffusionMPPIParams(
            N=10, K=16,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            diff_mode="blend",
            diff_blend_ratio=0.5,
            diff_ddim_steps=2,
            diff_T=50,
        )
        ctrl = DiffusionMPPIController(model, params)
        u, info = ctrl.compute_control(state, reference)
        assert np.all(np.isfinite(u))


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionTrainer 테스트 (PyTorch 필요)
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffusionTrainer:
    @pytest.fixture(autouse=True)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch 미설치 — Diffusion 학습기 테스트 스킵")

    def test_add_sample(self):
        from mppi_controller.learning.diffusion_trainer import DiffusionTrainer
        trainer = DiffusionTrainer(control_seq_dim=10 * 2, context_dim=3)
        state = np.zeros(3)
        ctrl = np.zeros((10, 2))
        trainer.add_sample(state, ctrl)
        assert len(trainer._control_seqs) == 1

    def test_add_batch(self):
        from mppi_controller.learning.diffusion_trainer import DiffusionTrainer
        trainer = DiffusionTrainer(control_seq_dim=10 * 2, context_dim=3)
        states = np.zeros((5, 3))
        ctrls = np.zeros((5, 10, 2))
        trainer.add_batch(states, ctrls)
        assert len(trainer._control_seqs) == 5

    def test_train_insufficient_data(self):
        """데이터 부족 시 학습 스킵"""
        from mppi_controller.learning.diffusion_trainer import DiffusionTrainer
        trainer = DiffusionTrainer(control_seq_dim=10 * 2, context_dim=3)
        stats = trainer.train(epochs=1)
        assert stats["n_samples"] < 2

    def test_train_with_data(self):
        """충분한 데이터로 학습"""
        from mppi_controller.learning.diffusion_trainer import DiffusionTrainer
        trainer = DiffusionTrainer(
            control_seq_dim=5 * 2,
            context_dim=3,
            hidden_dims=[32, 32],
        )
        # 100개 샘플 추가
        rng = np.random.default_rng(42)
        for _ in range(100):
            trainer.add_sample(
                rng.standard_normal(3),
                rng.standard_normal((5, 2)),
            )
        stats = trainer.train(epochs=3, verbose=False)
        assert stats["n_samples"] == 100
        assert stats["epochs"] == 3
        assert np.isfinite(stats["loss"])

    def test_get_model(self):
        """모델 반환"""
        from mppi_controller.learning.diffusion_trainer import DiffusionTrainer
        trainer = DiffusionTrainer(control_seq_dim=10, context_dim=3)
        model = trainer.get_model()
        assert model is not None


# ─────────────────────────────────────────────────────────────────────────────
# 통합: Bootstrap 후 학습된 모델로 샘플링
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffusionBootstrap:
    @pytest.fixture(autouse=True)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch 미설치")

    def test_bootstrap_and_sample(self, model):
        """부트스트랩 학습 후 샘플링"""
        params = DiffusionMPPIParams(
            N=5, K=16,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([1.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
            diff_ddim_steps=2,
            diff_T=50,
            diff_online_training=False,
            diff_min_samples=10,
        )
        ctrl = DiffusionMPPIController(model, params)

        # 부트스트랩 데이터 생성
        rng = np.random.default_rng(0)
        states = rng.standard_normal((20, 3))
        controls = rng.standard_normal((20, 5, 2)) * 0.3

        stats = ctrl.bootstrap_diffusion(states, controls, epochs=3, verbose=False)
        assert stats["n_samples"] == 20
        assert ctrl.is_diffusion_trained

        # 학습된 모델로 샘플링
        state = np.zeros(3)
        ref = np.zeros((6, 3))
        u, info = ctrl.compute_control(state, ref)
        assert np.all(np.isfinite(u))
        assert info["diffusion_is_trained"] == True
