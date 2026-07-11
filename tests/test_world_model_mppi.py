"""
World Model MPPI 테스트
"""

import numpy as np
import pytest
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.world_model_mppi import (
    WorldModelMPPIController,
    WorldModelMPPIParams,
)
from mppi_controller.models.learned.rssm import LinearWorldModel


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def robot_model():
    return DifferentialDriveKinematic()


@pytest.fixture
def params():
    return WorldModelMPPIParams(
        N=10, K=32, dt=0.05,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        wm_latent_dim=16,
        wm_min_samples=20,
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


def _make_trajectory(model, T=200, rng=None):
    """합성 궤적 데이터 생성."""
    if rng is None:
        rng = np.random.default_rng(42)
    dt = 0.05
    state = np.zeros(model.state_dim)
    obs_list = [state.copy()]
    act_list = []
    for _ in range(T - 1):
        u = rng.uniform(-0.3, 0.3, model.control_dim)
        state = state + model.forward_dynamics(state, u) * dt
        obs_list.append(state.copy())
        act_list.append(u)
    return np.array(obs_list), np.array(act_list)


# ─────────────────────────────────────────────────────────────────────────────
# WorldModelMPPIParams 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldModelMPPIParams:
    def test_defaults(self):
        p = WorldModelMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([1.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
        )
        assert p.wm_latent_dim == 32
        assert p.wm_reg == 1e-4
        assert p.wm_min_samples == 50
        assert not p.wm_online_training

    def test_invalid_latent_dim(self):
        with pytest.raises(AssertionError):
            WorldModelMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                wm_latent_dim=0,
            )

    def test_invalid_reg(self):
        with pytest.raises(AssertionError):
            WorldModelMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                wm_reg=-1e-4,
            )

    def test_buffer_smaller_than_min_samples(self):
        with pytest.raises(AssertionError):
            WorldModelMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                wm_min_samples=100,
                wm_buffer_size=50,
            )


# ─────────────────────────────────────────────────────────────────────────────
# LinearWorldModel 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestLinearWorldModel:
    def test_encode_shape(self):
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=8)
        obs = np.random.randn(10, 3)
        z = wm.encode(obs)
        assert z.shape == (10, 8)

    def test_decode_shape(self):
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=8)
        z = np.random.randn(10, 8)
        obs = wm.decode(z)
        assert obs.shape == (10, 3)

    def test_predict_shape(self):
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=8)
        z = np.random.randn(5, 8)
        a = np.random.randn(5, 2)
        z_next = wm.predict(z, a)
        assert z_next.shape == (5, 8)

    def test_imagine_shape(self):
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=8)
        z0 = np.random.randn(16, 8)
        actions = np.random.randn(16, 10, 2)
        Zs = wm.imagine(z0, actions)
        assert Zs.shape == (16, 11, 8)

    def test_imagine_initial_state(self):
        """초기 잠재 상태 보존"""
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=8)
        z0 = np.random.randn(4, 8)
        actions = np.zeros((4, 5, 2))
        Zs = wm.imagine(z0, actions)
        np.testing.assert_array_equal(Zs[:, 0, :], z0)

    def test_fit_returns_rmse(self, robot_model):
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=8)
        obs, acts = _make_trajectory(robot_model, T=100)
        rmse = wm.fit(obs, acts)
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_fit_marks_fitted(self, robot_model):
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=8)
        assert not wm._is_fitted
        obs, acts = _make_trajectory(robot_model, T=100)
        wm.fit(obs, acts)
        assert wm._is_fitted

    def test_fit_predict_finite(self, robot_model):
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=16)
        obs, acts = _make_trajectory(robot_model, T=200)
        wm.fit(obs, acts)
        z = wm.encode(obs[:10])
        z_next = wm.predict(z, acts[:10])
        assert np.all(np.isfinite(z_next))

    def test_fit_insufficient_data(self):
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=8)
        with pytest.raises(ValueError):
            wm.fit(np.zeros((2, 3)), np.zeros((1, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# WorldModelMPPIController 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldModelMPPIController:
    def test_instantiation(self, robot_model, params):
        ctrl = WorldModelMPPIController(robot_model, params)
        assert ctrl is not None
        assert not ctrl.is_world_model_trained

    def test_control_before_training(self, robot_model, params, state, reference):
        """미학습 시 fallback으로 정상 제어"""
        ctrl = WorldModelMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert np.all(np.isfinite(u))
        assert info["rollout_mode"] == "fallback_dynamics"

    def test_control_after_training(self, robot_model, params, state, reference):
        """학습 후 잠재 공간 rollout 사용"""
        ctrl = WorldModelMPPIController(robot_model, params)
        obs, acts = _make_trajectory(robot_model, T=100)
        ctrl.fit_world_model(obs, acts)
        assert ctrl.is_world_model_trained

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert np.all(np.isfinite(u))
        assert info["rollout_mode"] == "world_model_latent"

    def test_info_keys(self, robot_model, params, state, reference):
        ctrl = WorldModelMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        for key in [
            "sample_trajectories", "sample_weights", "best_trajectory",
            "world_model_is_trained", "world_model_fit_count",
            "world_model_latent_dim", "world_model_last_rmse", "rollout_mode",
        ]:
            assert key in info

    def test_info_not_trained_initially(self, robot_model, params, state, reference):
        ctrl = WorldModelMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert info["world_model_is_trained"] == False
        assert info["world_model_fit_count"] == 0
        assert info["world_model_last_rmse"] is None

    def test_sample_trajectories_shape(self, robot_model, state, reference):
        params = WorldModelMPPIParams(
            N=10, K=16,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            wm_latent_dim=8,
            wm_min_samples=10,
        )
        ctrl = WorldModelMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert info["sample_trajectories"].shape == (16, 11, 3)

    def test_sample_weights_sum_to_one(self, robot_model, params, state, reference):
        ctrl = WorldModelMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert np.isclose(info["sample_weights"].sum(), 1.0, atol=1e-6)

    def test_fit_world_model_rmse(self, robot_model, params):
        ctrl = WorldModelMPPIController(robot_model, params)
        obs, acts = _make_trajectory(robot_model, T=100)
        rmse = ctrl.fit_world_model(obs, acts)
        assert isinstance(rmse, float)
        assert ctrl._wm_fit_count == 1
        assert ctrl._wm_last_rmse == rmse

    def test_bootstrap_world_model(self, robot_model, params):
        ctrl = WorldModelMPPIController(robot_model, params)
        obs, acts = _make_trajectory(robot_model, T=100)
        rmse = ctrl.bootstrap_world_model(obs, acts)
        assert ctrl.is_world_model_trained
        assert isinstance(rmse, float)

    def test_add_observation_buffer(self, robot_model, params):
        ctrl = WorldModelMPPIController(robot_model, params)
        for _ in range(10):
            ctrl.add_observation(np.zeros(3))
        assert len(ctrl._obs_buf) == 10

    def test_fit_from_buffer_insufficient(self, robot_model, params):
        ctrl = WorldModelMPPIController(robot_model, params)
        ctrl.add_observation(np.zeros(3))
        result = ctrl.fit_from_buffer()
        assert result is None

    def test_fit_from_buffer_sufficient(self, robot_model):
        params = WorldModelMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([1.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
            wm_latent_dim=8,
            wm_min_samples=30,
        )
        ctrl = WorldModelMPPIController(robot_model, params)
        obs, acts = _make_trajectory(robot_model, T=80)
        for i in range(len(obs)):
            a = acts[i] if i < len(acts) else None
            ctrl.add_observation(obs[i], a)
        rmse = ctrl.fit_from_buffer()
        assert rmse is not None
        assert ctrl.is_world_model_trained

    def test_sequential_calls(self, robot_model, params, state, reference):
        ctrl = WorldModelMPPIController(robot_model, params)
        for _ in range(5):
            u, info = ctrl.compute_control(state, reference)
            assert np.all(np.isfinite(u))

    def test_sequential_calls_after_training(self, robot_model, params, state, reference):
        ctrl = WorldModelMPPIController(robot_model, params)
        obs, acts = _make_trajectory(robot_model, T=100)
        ctrl.fit_world_model(obs, acts)
        for _ in range(5):
            u, info = ctrl.compute_control(state, reference)
            assert np.all(np.isfinite(u))
            assert info["rollout_mode"] == "world_model_latent"

    def test_latent_dim_reported_in_info(self, robot_model, params, state, reference):
        ctrl = WorldModelMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert info["world_model_latent_dim"] == params.wm_latent_dim


# ─────────────────────────────────────────────────────────────────────────────
# RSSM 테스트 (PyTorch 있는 경우)
# ─────────────────────────────────────────────────────────────────────────────

class TestRSSMCore:
    @pytest.fixture(autouse=True)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch 미설치 — RSSM 테스트 스킵")

    def test_initial_state(self):
        import torch
        from mppi_controller.models.learned.rssm import RSSMCore
        rssm = RSSMCore(obs_dim=3, action_dim=2, deter_dim=64, stoch_dim=16)
        h, z = rssm.initial_state(batch_size=4)
        assert h.shape == (4, 64)
        assert z.shape == (4, 16)

    def test_prior_step_shapes(self):
        import torch
        from mppi_controller.models.learned.rssm import RSSMCore
        rssm = RSSMCore(obs_dim=3, action_dim=2, deter_dim=64, stoch_dim=16)
        h, z = rssm.initial_state(4)
        a = torch.randn(4, 2)
        h2, z2, pm, pv = rssm.prior_step(h, z, a)
        assert h2.shape == (4, 64)
        assert z2.shape == (4, 16)
        assert pm.shape == (4, 16)

    def test_posterior_step_shapes(self):
        import torch
        from mppi_controller.models.learned.rssm import RSSMCore
        rssm = RSSMCore(obs_dim=3, action_dim=2, deter_dim=64, stoch_dim=16)
        h, z = rssm.initial_state(4)
        a = torch.randn(4, 2)
        obs = torch.randn(4, 3)
        h2, z2, qm, qv, pm, pv = rssm.posterior_step(h, z, a, obs)
        assert h2.shape == (4, 64)
        assert z2.shape == (4, 16)

    def test_imagine_shapes(self):
        import torch
        from mppi_controller.models.learned.rssm import RSSMCore
        rssm = RSSMCore(obs_dim=3, action_dim=2, deter_dim=64, stoch_dim=16)
        h, z = rssm.initial_state(8)
        actions = torch.randn(8, 10, 2)
        hs, zs = rssm.imagine(h, z, actions)
        assert len(hs) == 11
        assert len(zs) == 11
        assert hs[0].shape == (8, 64)

    def test_decode_shape(self):
        import torch
        from mppi_controller.models.learned.rssm import RSSMCore
        rssm = RSSMCore(obs_dim=3, action_dim=2, deter_dim=64, stoch_dim=16)
        h = torch.randn(4, 64)
        z = torch.randn(4, 16)
        obs = rssm.decode(h, z)
        assert obs.shape == (4, 3)

    def test_kl_loss(self):
        import torch
        from mppi_controller.models.learned.rssm import RSSMCore
        rssm = RSSMCore(obs_dim=3, action_dim=2, deter_dim=64, stoch_dim=16)
        pm = torch.zeros(8, 16)
        pv = torch.zeros(8, 16)
        qm = torch.zeros(8, 16)
        qv = torch.zeros(8, 16)
        kl = rssm.kl_loss(qm, qv, pm, pv)
        assert isinstance(kl.item(), float)
        assert kl.item() >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 통합 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldModelIntegration:
    def test_latent_rmse_finite(self, robot_model):
        """세계 모델 학습 오차가 유한해야 함"""
        wm = LinearWorldModel(obs_dim=3, action_dim=2, latent_dim=16)
        obs, acts = _make_trajectory(robot_model, T=300)
        rmse = wm.fit(obs, acts)
        assert np.isfinite(rmse)

    def test_tracking_no_crash(self, robot_model):
        """학습 후 여러 스텝 제어 정상"""
        params = WorldModelMPPIParams(
            N=15, K=64, dt=0.05,
            lambda_=1.0,
            sigma=np.array([0.5, 0.3]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            wm_latent_dim=16,
            wm_min_samples=20,
        )
        ctrl = WorldModelMPPIController(robot_model, params)

        obs, acts = _make_trajectory(robot_model, T=200)
        ctrl.fit_world_model(obs, acts)

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((16, 3))
        ref[:, 0] = np.linspace(0, 1.0, 16)

        for _ in range(10):
            u, info = ctrl.compute_control(state, ref)
            assert np.all(np.isfinite(u))
            assert info["rollout_mode"] == "world_model_latent"
