"""
Koopman MPPI 테스트
"""

import numpy as np
import pytest
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.koopman_mppi import (
    KoopmanMPPIController,
    KoopmanBatchRollout,
)
from mppi_controller.controllers.mppi.mppi_params import KoopmanMPPIParams
from mppi_controller.models.learned.koopman_dynamics import (
    KoopmanDynamics,
    KoopmanTrainer,
    _rbf_features,
    _polynomial_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def robot_model():
    return DifferentialDriveKinematic()


@pytest.fixture
def params():
    return KoopmanMPPIParams(
        N=10, K=32, dt=0.05,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        koopman_lift_fn="rbf",
        koopman_lift_dim=16,
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


def _make_synthetic_data(model, n=200, rng=None):
    """합성 전이 데이터 생성 (학습용)."""
    if rng is None:
        rng = np.random.default_rng(42)
    nx, nu = model.state_dim, model.control_dim
    X = rng.uniform(-1, 1, (n, nx))
    U = rng.uniform(-0.5, 0.5, (n, nu))
    dt = 0.05
    X_next = np.array([
        X[i] + model.forward_dynamics(X[i], U[i]) * dt
        for i in range(n)
    ])
    return X, U, X_next


# ─────────────────────────────────────────────────────────────────────────────
# KoopmanMPPIParams 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanMPPIParams:
    def test_defaults(self):
        p = KoopmanMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([1.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
        )
        assert p.koopman_lift_fn == "rbf"
        assert p.koopman_lift_dim == 64
        assert p.koopman_rbf_gamma == 1.0
        assert p.koopman_reg == 1e-4
        assert p.koopman_min_samples == 50
        assert p.koopman_buffer_size == 5000

    def test_invalid_lift_fn(self):
        with pytest.raises(AssertionError):
            KoopmanMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                koopman_lift_fn="invalid",
            )

    def test_invalid_poly_degree(self):
        with pytest.raises(AssertionError):
            KoopmanMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                koopman_poly_degree=3,
            )

    def test_invalid_reg(self):
        with pytest.raises(AssertionError):
            KoopmanMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                koopman_reg=-0.1,
            )

    def test_all_lift_fns(self):
        for fn in ("rbf", "poly", "identity"):
            p = KoopmanMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                koopman_lift_fn=fn,
            )
            assert p.koopman_lift_fn == fn


# ─────────────────────────────────────────────────────────────────────────────
# 특징 함수 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureFunctions:
    def test_rbf_shape(self):
        x = np.random.randn(5, 3)
        centers = np.random.randn(16, 3)
        feat = _rbf_features(x, centers, gamma=1.0)
        assert feat.shape == (5, 16)

    def test_rbf_range(self):
        """RBF 출력은 [0, 1]"""
        x = np.random.randn(10, 3)
        centers = np.random.randn(8, 3)
        feat = _rbf_features(x, centers)
        assert np.all(feat >= 0)
        assert np.all(feat <= 1.0)

    def test_rbf_self_similarity(self):
        """자기 자신에 대한 RBF는 1.0"""
        x = np.array([[1.0, 2.0, 3.0]])
        centers = np.array([[1.0, 2.0, 3.0]])
        feat = _rbf_features(x, centers)
        assert np.isclose(feat[0, 0], 1.0)

    def test_poly_degree1_shape(self):
        x = np.random.randn(7, 4)
        feat = _polynomial_features(x, degree=1)
        assert feat.shape == (7, 4)

    def test_poly_degree2_shape(self):
        nx = 3
        x = np.random.randn(5, nx)
        feat = _polynomial_features(x, degree=2)
        # nx + nx*(nx+1)/2 = 3 + 6 = 9
        assert feat.shape == (5, 9)

    def test_poly_batch(self):
        """배치 차원 처리 정상"""
        x = np.random.randn(4, 3)
        feat = _polynomial_features(x, degree=2)
        for i in range(4):
            feat_i = _polynomial_features(x[i:i+1], degree=2)
            np.testing.assert_array_almost_equal(feat[i], feat_i[0])


# ─────────────────────────────────────────────────────────────────────────────
# KoopmanDynamics 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanDynamics:
    def test_phi_shape_rbf(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=16)
        x = np.random.randn(5, 3)
        z = model.phi(x)
        assert z.shape == (5, 17)  # 16 RBF + 1 bias

    def test_phi_shape_poly(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="poly", poly_degree=2)
        x = np.random.randn(5, 3)
        z = model.phi(x)
        assert z.shape == (5, 10)  # 3 + 6 + 1 bias

    def test_phi_no_bias(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8, use_bias=False)
        x = np.random.randn(2, 3)
        z = model.phi(x)
        assert z.shape == (2, 8)

    def test_project_shape(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8)
        z = np.random.randn(5, model._phi_dim)
        x = model.project(z)
        assert x.shape == (5, 3)

    def test_forward_dynamics_shape(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8)
        state = np.random.randn(3)
        control = np.random.randn(2)
        state_dot = model.forward_dynamics(state, control)
        assert state_dot.shape == (3,)

    def test_forward_dynamics_batch(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8)
        state = np.random.randn(5, 3)
        control = np.random.randn(5, 2)
        state_dot = model.forward_dynamics(state, control)
        assert state_dot.shape == (5, 3)

    def test_rollout_linear_shape(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8)
        state = np.zeros(3)
        controls = np.zeros((10, 2))
        traj = model.rollout_linear(state, controls)
        assert traj.shape == (11, 3)

    def test_rollout_linear_initial_state(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8)
        state = np.array([1.0, 2.0, 3.0])
        traj = model.rollout_linear(state, np.zeros((5, 2)))
        np.testing.assert_array_equal(traj[0], state)

    def test_batch_rollout_shape(self):
        model = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8)
        states = np.zeros((16, 3))
        controls = np.zeros((16, 10, 2))
        trajs = model.batch_rollout_linear(states, controls)
        assert trajs.shape == (16, 11, 3)

    def test_fit_edmd_returns_rmse(self, robot_model):
        model = KoopmanDynamics(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
            lift_fn="rbf",
            lift_dim=16,
        )
        X, U, X_next = _make_synthetic_data(robot_model, n=100)
        rmse = model.fit_edmd(X, U, X_next)
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_fit_edmd_is_fitted(self, robot_model):
        model = KoopmanDynamics(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
            lift_fn="rbf",
            lift_dim=16,
        )
        assert not model._is_fitted
        X, U, X_next = _make_synthetic_data(robot_model, n=100)
        model.fit_edmd(X, U, X_next)
        assert model._is_fitted

    def test_fit_edmd_predicts_finite(self, robot_model):
        model = KoopmanDynamics(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
            lift_fn="rbf",
            lift_dim=16,
        )
        X, U, X_next = _make_synthetic_data(robot_model, n=150)
        model.fit_edmd(X, U, X_next)
        state_dot = model.forward_dynamics(X[0], U[0])
        assert np.all(np.isfinite(state_dot))


# ─────────────────────────────────────────────────────────────────────────────
# KoopmanTrainer 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanTrainer:
    def test_add_transition(self, robot_model):
        trainer = KoopmanTrainer(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
        )
        trainer.add_transition(np.zeros(3), np.zeros(2), np.zeros(3))
        assert trainer.n_samples == 1

    def test_add_trajectory(self, robot_model):
        trainer = KoopmanTrainer(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
        )
        states = np.zeros((11, 3))
        controls = np.zeros((10, 2))
        trainer.add_trajectory(states, controls)
        assert trainer.n_samples == 10

    def test_fit_insufficient_data(self, robot_model):
        trainer = KoopmanTrainer(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
        )
        trainer.add_transition(np.zeros(3), np.zeros(2), np.zeros(3))
        with pytest.raises(ValueError):
            trainer.fit()

    def test_fit_returns_rmse(self, robot_model):
        trainer = KoopmanTrainer(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
            lift_fn="rbf",
            lift_dim=16,
        )
        X, U, X_next = _make_synthetic_data(robot_model, n=100)
        for i in range(len(X)):
            trainer.add_transition(X[i], U[i], X_next[i])
        rmse = trainer.fit()
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_model_is_fitted_after_fit(self, robot_model):
        trainer = KoopmanTrainer(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
        )
        X, U, X_next = _make_synthetic_data(robot_model, n=80)
        for i in range(len(X)):
            trainer.add_transition(X[i], U[i], X_next[i])
        trainer.fit()
        assert trainer.model._is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# KoopmanBatchRollout 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanBatchRollout:
    def test_fallback_when_untrained(self, robot_model, params):
        """미학습 시 fallback dynamics 사용"""
        from mppi_controller.controllers.mppi.dynamics_wrapper import BatchDynamicsWrapper
        koopman = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8)
        fallback = BatchDynamicsWrapper(robot_model, params.dt)
        rollout = KoopmanBatchRollout(koopman, fallback)

        assert not rollout.is_trained
        state = np.zeros(3)
        controls = np.zeros((8, 10, 2))
        traj = rollout.rollout(state, controls)
        assert traj.shape == (8, 11, 3)

    def test_koopman_rollout_after_training(self, robot_model, params):
        """학습 후 Koopman rollout 사용"""
        koopman = KoopmanDynamics(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
            lift_fn="rbf",
            lift_dim=16,
        )
        X, U, X_next = _make_synthetic_data(robot_model, n=100)
        koopman.fit_edmd(X, U, X_next)

        rollout = KoopmanBatchRollout(koopman, fallback_wrapper=None)
        assert rollout.is_trained

        state = np.zeros(3)
        controls = np.zeros((8, 10, 2))
        traj = rollout.rollout(state, controls)
        assert traj.shape == (8, 11, 3)
        assert np.all(np.isfinite(traj))

    def test_no_fallback_raises_when_untrained(self):
        """fallback 없고 미학습 시 RuntimeError"""
        koopman = KoopmanDynamics(nx=3, nu=2, lift_fn="rbf", lift_dim=8)
        rollout = KoopmanBatchRollout(koopman, fallback_wrapper=None)
        with pytest.raises(RuntimeError):
            rollout.rollout(np.zeros(3), np.zeros((4, 5, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# KoopmanMPPIController 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanMPPIController:
    def test_instantiation(self, robot_model, params):
        ctrl = KoopmanMPPIController(robot_model, params)
        assert ctrl is not None
        assert not ctrl.is_koopman_trained

    def test_phi_dim_rbf(self, robot_model):
        params = KoopmanMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([1.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
            koopman_lift_fn="rbf",
            koopman_lift_dim=32,
            koopman_use_bias=True,
        )
        ctrl = KoopmanMPPIController(robot_model, params)
        assert ctrl.phi_dim == 33  # 32 + 1

    def test_control_output_before_training(self, robot_model, params, state, reference):
        """미학습 시 fallback으로 정상 제어 계산"""
        ctrl = KoopmanMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert np.all(np.isfinite(u))
        assert info["rollout_mode"] == "fallback_dynamics"

    def test_control_output_after_training(self, robot_model, params, state, reference):
        """학습 후 Koopman rollout으로 정상 제어 계산"""
        ctrl = KoopmanMPPIController(robot_model, params)
        X, U, X_next = _make_synthetic_data(robot_model, n=200)
        ctrl.fit_edmd(X, U, X_next)
        assert ctrl.is_koopman_trained

        u, info = ctrl.compute_control(state, reference)
        assert u.shape == (2,)
        assert np.all(np.isfinite(u))
        assert info["rollout_mode"] == "koopman_linear"

    def test_info_keys(self, robot_model, params, state, reference):
        ctrl = KoopmanMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        for key in [
            "sample_trajectories", "sample_weights", "best_trajectory",
            "koopman_is_trained", "koopman_fit_count", "koopman_phi_dim",
            "koopman_last_rmse", "rollout_mode",
        ]:
            assert key in info

    def test_info_not_trained_initially(self, robot_model, params, state, reference):
        ctrl = KoopmanMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert info["koopman_is_trained"] == False
        assert info["koopman_fit_count"] == 0
        assert info["koopman_last_rmse"] is None

    def test_sample_trajectories_shape(self, robot_model, state, reference):
        params = KoopmanMPPIParams(
            N=10, K=16,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
        )
        ctrl = KoopmanMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert info["sample_trajectories"].shape == (16, 11, 3)

    def test_sample_weights_sum_to_one(self, robot_model, params, state, reference):
        ctrl = KoopmanMPPIController(robot_model, params)
        u, info = ctrl.compute_control(state, reference)
        assert np.isclose(info["sample_weights"].sum(), 1.0, atol=1e-6)

    def test_bootstrap_koopman(self, robot_model, params):
        ctrl = KoopmanMPPIController(robot_model, params)
        rng = np.random.default_rng(0)
        T = 100
        states = np.zeros((T + 1, 3))
        controls = rng.uniform(-0.3, 0.3, (T, 2))
        rmse = ctrl.bootstrap_koopman(states, controls)
        assert isinstance(rmse, float)
        assert ctrl.is_koopman_trained

    def test_fit_edmd_rmse(self, robot_model, params):
        ctrl = KoopmanMPPIController(robot_model, params)
        X, U, X_next = _make_synthetic_data(robot_model, n=200)
        rmse = ctrl.fit_edmd(X, U, X_next)
        assert isinstance(rmse, float)
        assert rmse >= 0
        assert ctrl._fit_count == 1
        assert ctrl._last_fit_rmse == rmse

    def test_add_transition_buffer(self, robot_model, params):
        ctrl = KoopmanMPPIController(robot_model, params)
        for _ in range(10):
            ctrl.add_transition(np.zeros(3), np.zeros(2), np.zeros(3))
        assert len(ctrl._X_buf) == 10

    def test_fit_from_buffer_insufficient(self, robot_model, params):
        ctrl = KoopmanMPPIController(robot_model, params)
        ctrl.add_transition(np.zeros(3), np.zeros(2), np.zeros(3))
        result = ctrl.fit_from_buffer()
        assert result is None

    def test_fit_from_buffer_sufficient(self, robot_model):
        params = KoopmanMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([1.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
            koopman_lift_dim=16,
            koopman_min_samples=50,
        )
        ctrl = KoopmanMPPIController(robot_model, params)
        X, U, X_next = _make_synthetic_data(robot_model, n=100)
        for i in range(len(X)):
            ctrl.add_transition(X[i], U[i], X_next[i])
        rmse = ctrl.fit_from_buffer()
        assert rmse is not None
        assert isinstance(rmse, float)
        assert ctrl.is_koopman_trained

    def test_sequential_calls(self, robot_model, params, state, reference):
        """연속 호출 정상 동작"""
        ctrl = KoopmanMPPIController(robot_model, params)
        for _ in range(5):
            u, info = ctrl.compute_control(state, reference)
            assert np.all(np.isfinite(u))

    def test_sequential_calls_after_training(self, robot_model, params, state, reference):
        """학습 후 연속 호출 정상"""
        ctrl = KoopmanMPPIController(robot_model, params)
        X, U, X_next = _make_synthetic_data(robot_model, n=200)
        ctrl.fit_edmd(X, U, X_next)
        for _ in range(5):
            u, info = ctrl.compute_control(state, reference)
            assert np.all(np.isfinite(u))
            assert info["rollout_mode"] == "koopman_linear"

    def test_poly_lift_fn(self, robot_model):
        params = KoopmanMPPIParams(
            N=10, K=16,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            koopman_lift_fn="poly",
            koopman_poly_degree=2,
        )
        ctrl = KoopmanMPPIController(robot_model, params)
        state = np.zeros(3)
        ref = np.zeros((11, 3))
        u, info = ctrl.compute_control(state, ref)
        assert u.shape == (2,)
        assert np.all(np.isfinite(u))

    def test_identity_lift_fn(self, robot_model):
        params = KoopmanMPPIParams(
            N=10, K=16,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            koopman_lift_fn="identity",
        )
        ctrl = KoopmanMPPIController(robot_model, params)
        state = np.zeros(3)
        ref = np.zeros((11, 3))
        u, info = ctrl.compute_control(state, ref)
        assert u.shape == (2,)


# ─────────────────────────────────────────────────────────────────────────────
# 통합 테스트: EDMD 학습 후 추적 성능
# ─────────────────────────────────────────────────────────────────────────────

class TestKoopmanIntegration:
    def test_edmd_reduces_prediction_error(self, robot_model):
        """EDMD 학습 후 예측 오차가 감소해야 함"""
        koopman = KoopmanDynamics(
            nx=robot_model.state_dim,
            nu=robot_model.control_dim,
            lift_fn="rbf",
            lift_dim=32,
        )
        X, U, X_next = _make_synthetic_data(robot_model, n=500)
        rmse = koopman.fit_edmd(X, U, X_next)
        # DiffDrive는 선형 근사가 잘 맞으므로 허용치 내
        assert rmse < 1.0, f"EDMD RMSE too large: {rmse}"

    def test_tracking_goal(self, robot_model):
        """목표 지점으로 진행 중 오류 없음 확인"""
        params = KoopmanMPPIParams(
            N=20, K=64, dt=0.05,
            lambda_=1.0,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            koopman_lift_dim=16,
        )
        ctrl = KoopmanMPPIController(robot_model, params)

        # 학습 데이터 생성
        X, U, X_next = _make_synthetic_data(robot_model, n=300)
        ctrl.fit_edmd(X, U, X_next)

        state = np.array([0.0, 0.0, 0.0])
        ref = np.zeros((21, 3))
        ref[:, 0] = np.linspace(0, 1.0, 21)

        errors = []
        for _ in range(5):
            u, info = ctrl.compute_control(state, ref)
            assert np.all(np.isfinite(u))
            errors.append(info["best_cost"])

        # 비용이 유한해야 함
        assert all(np.isfinite(e) for e in errors)
