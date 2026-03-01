"""
Mobile Manipulator 6-DOF Residual Dynamics 통합 테스트

Residual Dynamics = Kinematic + NN 보정의 통합 검증:
  1. ResidualDynamics 조합 확인
  2. 데이터 생성 파이프라인
  3. NN 학습 수렴 테스트
  4. MPPI 컨트롤러 통합
"""

import numpy as np
import pytest
import torch

from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
    MobileManipulator6DOFDynamic,
)
from mppi_controller.models.learned.residual_dynamics import (
    ResidualDynamics,
    create_constant_residual,
)
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset
from mppi_controller.learning.neural_network_trainer import (
    NeuralNetworkTrainer,
    DynamicsMLPModel,
)


class TestResidualIntegration:
    """ResidualDynamics 조합 테스트"""

    def setup_method(self):
        self.kin_model = MobileManipulator6DOFKinematic()
        self.dyn_model = MobileManipulator6DOFDynamic()

    def test_residual_composition(self):
        """base + residual_fn = total"""
        residual_val = np.array([0.01, 0.02, 0.005, -0.1, -0.05, -0.03, 0.0, 0.0, 0.0])
        residual_fn = create_constant_residual(residual_val)

        model = ResidualDynamics(
            base_model=self.kin_model,
            residual_fn=residual_fn,
        )

        state = np.zeros(9)
        control = np.array([0.5, 0.1, 0.3, -0.2, 0.1, 0.0, 0.0, 0.0])

        kin_dot = self.kin_model.forward_dynamics(state, control)
        total_dot = model.forward_dynamics(state, control)

        expected = kin_dot + residual_val
        np.testing.assert_allclose(total_dot, expected, atol=1e-10)

    def test_residual_with_zero_residual_fn(self):
        """residual_fn=None이면 base만 사용"""
        model = ResidualDynamics(base_model=self.kin_model)

        state = np.zeros(9)
        control = np.array([0.5, 0.1, 0.3, -0.2, 0.1, 0.0, 0.0, 0.0])

        kin_dot = self.kin_model.forward_dynamics(state, control)
        total_dot = model.forward_dynamics(state, control)

        np.testing.assert_allclose(total_dot, kin_dot, atol=1e-10)

    def test_residual_dimensions(self):
        """ResidualDynamics 차원이 base와 동일"""
        model = ResidualDynamics(base_model=self.kin_model)
        assert model.state_dim == 9
        assert model.control_dim == 8
        assert model.model_type == "learned"

    def test_residual_contribution_analysis(self):
        """get_residual_contribution 분석"""
        residual_val = np.array([0.01, 0.02, 0.005, -0.1, -0.05, -0.03, 0.0, 0.0, 0.0])
        residual_fn = create_constant_residual(residual_val)

        model = ResidualDynamics(
            base_model=self.kin_model,
            residual_fn=residual_fn,
        )

        state = np.array([0.0, 0.0, 0.3, 0.5, -0.3, 0.2, 0.0, 0.0, 0.0])
        control = np.array([0.5, 0.2, 1.0, -0.5, 0.3, 0.0, 0.0, 0.0])

        contrib = model.get_residual_contribution(state, control)
        assert "physics_dot" in contrib
        assert "residual_dot" in contrib
        assert "total_dot" in contrib
        assert "residual_ratio" in contrib

        # total = physics + residual
        np.testing.assert_allclose(
            contrib["total_dot"],
            contrib["physics_dot"] + contrib["residual_dot"],
            atol=1e-10,
        )

    def test_mppi_with_residual_model(self):
        """ResidualDynamics로 MPPI compute_control 호출"""
        from mppi_controller.controllers.mppi.base_mppi import MPPIController
        from mppi_controller.controllers.mppi.mppi_params import MPPIParams
        from mppi_controller.controllers.mppi.cost_functions import (
            CompositeMPPICost,
            EndEffector3DTrackingCost,
            ControlEffortCost,
        )

        model = ResidualDynamics(base_model=self.kin_model)

        params = MPPIParams(
            N=10,
            dt=0.05,
            K=32,
            lambda_=1.0,
            sigma=np.array([0.3, 0.3] + [0.5] * 6),
            Q=np.array([1.0] * 9),
            R=np.array([0.1] * 8),
            Qf=np.array([1.0] * 9),
            device="cpu",
        )

        cost_fn = CompositeMPPICost([
            EndEffector3DTrackingCost(self.kin_model, weight=10.0),
            ControlEffortCost(R=np.array([0.1] * 8)),
        ])

        controller = MPPIController(model, params, cost_function=cost_fn)

        state = np.zeros(9)
        ref = np.zeros((11, 9))
        ref[:, 0] = 0.5  # x target

        control, info = controller.compute_control(state, ref)
        assert control.shape == (8,)
        assert "sample_weights" in info

    def test_residual_reduces_error(self):
        """Residual 모델이 oracle에 더 가깝게 동작하는지 확인"""
        # dynamic 모델의 보정을 residual로 직접 사용
        def oracle_residual(state, control):
            kin_dot = self.kin_model.forward_dynamics(state, control)
            dyn_dot = self.dyn_model.forward_dynamics(state, control)
            return dyn_dot - kin_dot

        residual_model = ResidualDynamics(
            base_model=self.kin_model,
            residual_fn=oracle_residual,
        )

        # 여러 상태에서 테스트
        np.random.seed(42)
        for _ in range(10):
            state = np.random.randn(9) * 0.5
            control = np.random.randn(8) * 0.5

            oracle_dot = self.dyn_model.forward_dynamics(state, control)
            kin_dot = self.kin_model.forward_dynamics(state, control)
            res_dot = residual_model.forward_dynamics(state, control)

            kin_error = np.linalg.norm(oracle_dot - kin_dot)
            res_error = np.linalg.norm(oracle_dot - res_dot)

            # Residual 모델은 oracle과 완전 일치해야 함
            assert res_error < kin_error + 1e-10


class TestDataGeneration:
    """데이터 생성 파이프라인 테스트"""

    def setup_method(self):
        self.kin_model = MobileManipulator6DOFKinematic()
        self.dyn_model = MobileManipulator6DOFDynamic()

    def test_data_collection_shape(self):
        """DataCollector로 데이터 수집 shape 확인"""
        collector = DataCollector(state_dim=9, control_dim=8)

        n_samples = 100
        dt = 0.05

        np.random.seed(42)
        for i in range(n_samples):
            state = np.random.randn(9) * 0.5
            control = np.random.randn(8) * 0.3
            next_state = self.dyn_model.step(state, control, dt)
            collector.add_sample(state, control, next_state, dt)

        collector.end_episode()

        data = collector.get_data()
        assert data["states"].shape == (n_samples, 9)
        assert data["controls"].shape == (n_samples, 8)
        assert data["next_states"].shape == (n_samples, 9)
        assert data["state_dots"].shape == (n_samples, 9)

    def test_residual_nonzero(self):
        """기구학 ≠ 동역학 → residual 비영"""
        np.random.seed(42)
        state = np.array([0.0, 0.0, 0.3, 0.5, -0.3, 0.2, 0.0, 0.0, 0.0])
        control = np.array([0.5, 0.2, 1.0, -0.5, 0.3, 0.0, 0.0, 0.0])

        kin_dot = self.kin_model.forward_dynamics(state, control)
        dyn_dot = self.dyn_model.forward_dynamics(state, control)
        residual = dyn_dot - kin_dot

        assert np.linalg.norm(residual) > 1e-4

    def test_dataset_split(self):
        """DynamicsDataset train/val 분할"""
        n_samples = 200
        np.random.seed(42)

        states = np.random.randn(n_samples, 9) * 0.5
        controls = np.random.randn(n_samples, 8) * 0.3
        state_dots = np.random.randn(n_samples, 9) * 0.1

        data = {
            "states": states,
            "controls": controls,
            "state_dots": state_dots,
        }

        dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)

        train_inputs, train_targets = dataset.get_train_data()
        val_inputs, val_targets = dataset.get_val_data()

        assert train_inputs.shape[0] == 160  # 80%
        assert val_inputs.shape[0] == 40     # 20%
        assert train_inputs.shape[1] == 17   # 9 + 8
        assert train_targets.shape[1] == 9

    def test_nn_training_converges(self):
        """NN이 수 에포크 안에 loss 감소하는지 확인"""
        np.random.seed(42)
        n_samples = 500

        # Residual 데이터 생성
        states = np.random.randn(n_samples, 9) * 0.5
        controls = np.random.randn(n_samples, 8) * 0.3

        residuals = np.zeros((n_samples, 9))
        for i in range(n_samples):
            kin_dot = self.kin_model.forward_dynamics(states[i], controls[i])
            dyn_dot = self.dyn_model.forward_dynamics(states[i], controls[i])
            residuals[i] = dyn_dot - kin_dot

        data = {
            "states": states,
            "controls": controls,
            "state_dots": residuals,
        }

        dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
        train_inputs, train_targets = dataset.get_train_data()
        val_inputs, val_targets = dataset.get_val_data()
        norm_stats = dataset.get_normalization_stats()

        trainer = NeuralNetworkTrainer(
            state_dim=9,
            control_dim=8,
            hidden_dims=[64, 64],
            learning_rate=1e-3,
            dropout_rate=0.0,
            save_dir="/tmp/test_6dof_residual",
        )

        history = trainer.train(
            train_inputs, train_targets,
            val_inputs, val_targets,
            norm_stats=norm_stats,
            epochs=30,
            batch_size=64,
            verbose=False,
        )

        # Loss가 감소해야 함
        assert history["train_loss"][-1] < history["train_loss"][0]
        assert history["val_loss"][-1] < history["val_loss"][0]
