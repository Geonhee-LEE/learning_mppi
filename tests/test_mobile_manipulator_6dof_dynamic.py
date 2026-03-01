"""
Mobile Manipulator 6-DOF Dynamic 모델 테스트

Ground-truth 동역학 모델의 물리 효과 검증:
  1. 기본 속성 (state_dim, control_dim, model_type)
  2. 기구학 모델 상속 확인
  3. 물리 효과: 베이스 마찰, 관절 마찰, 커플링, 중력 처짐
  4. 배치 연산 지원
"""

import numpy as np
import pytest
from mppi_controller.models.dynamic.mobile_manipulator_6dof_dynamic import (
    MobileManipulator6DOFDynamic,
)
from mppi_controller.models.kinematic.mobile_manipulator_6dof_kinematic import (
    MobileManipulator6DOFKinematic,
)
from mppi_controller.models.base_model import RobotModel


class TestDynamic6DOFModel:
    """동역학 모델 기본 검증"""

    def setup_method(self):
        self.model = MobileManipulator6DOFDynamic()
        self.kin_model = MobileManipulator6DOFKinematic()

    def test_dimensions(self):
        """state_dim=9, control_dim=8"""
        assert self.model.state_dim == 9
        assert self.model.control_dim == 8

    def test_model_type_dynamic(self):
        """model_type이 'dynamic'"""
        assert self.model.model_type == "dynamic"

    def test_is_robot_model(self):
        """RobotModel ABC 상속 확인"""
        assert isinstance(self.model, RobotModel)

    def test_inherits_kinematic(self):
        """MobileManipulator6DOFKinematic 상속 확인"""
        assert isinstance(self.model, MobileManipulator6DOFKinematic)

    def test_forward_dynamics_shape(self):
        """forward_dynamics 출력 shape 확인"""
        state = np.zeros(9)
        control = np.zeros(8)
        dot = self.model.forward_dynamics(state, control)
        assert dot.shape == (9,)

    def test_forward_dynamics_differs_from_kinematic(self):
        """동역학 출력이 기구학과 다른지 확인 (비영 제어)"""
        state = np.array([0.0, 0.0, 0.3, 0.5, -0.3, 0.2, 0.0, 0.0, 0.0])
        control = np.array([0.5, 0.2, 1.0, -0.5, 0.3, 0.0, 0.0, 0.0])

        kin_dot = self.kin_model.forward_dynamics(state, control)
        dyn_dot = self.model.forward_dynamics(state, control)

        # 물리 보정으로 차이가 있어야 함
        assert not np.allclose(kin_dot, dyn_dot, atol=1e-6)

    def test_fk_matches_kinematic(self):
        """FK는 기구학과 동일해야 함 (오버라이드 안 함)"""
        state = np.array([1.0, 0.5, 0.3, 0.2, -0.1, 0.4, 0.0, 0.1, -0.2])

        kin_ee = self.kin_model.forward_kinematics(state)
        dyn_ee = self.model.forward_kinematics(state)

        np.testing.assert_allclose(kin_ee, dyn_ee, atol=1e-10)

    def test_base_friction_effect(self):
        """베이스 속도 마찰: v가 양수면 x_dot이 기구학보다 감소"""
        state = np.zeros(9)  # theta=0 → cos(0)=1
        control = np.zeros(8)
        control[0] = 0.3  # v = 0.3 (저속에서 마찰 효과 큼)

        kin_dot = self.kin_model.forward_dynamics(state, control)
        dyn_dot = self.model.forward_dynamics(state, control)

        # 동역학 x_dot < 기구학 x_dot (마찰로 감소)
        assert dyn_dot[0] < kin_dot[0]
        # y_dot은 theta=0이므로 둘 다 0
        np.testing.assert_allclose(dyn_dot[1], 0.0, atol=1e-10)

    def test_joint_friction_effect(self):
        """관절 마찰: dq가 양수면 q_dot이 기구학보다 감소"""
        state = np.zeros(9)
        control = np.zeros(8)
        control[2] = 1.5  # dq1 = 1.5

        kin_dot = self.kin_model.forward_dynamics(state, control)
        dyn_dot = self.model.forward_dynamics(state, control)

        # 관절 마찰로 q1_dot이 감소
        assert dyn_dot[3] < kin_dot[3]

    def test_gravity_droop_effect(self):
        """중력 처짐: q2에서 sin(q2) 비례 처짐"""
        state = np.zeros(9)
        state[4] = np.pi / 4  # q2 = 45도 (수평 팔)
        control = np.zeros(8)

        kin_dot = self.kin_model.forward_dynamics(state, control)
        dyn_dot = self.model.forward_dynamics(state, control)

        # 기구학은 제어 없으면 q2_dot = 0
        np.testing.assert_allclose(kin_dot[4], 0.0, atol=1e-10)
        # 동역학은 중력으로 q2_dot < 0
        assert dyn_dot[4] < 0

    def test_coupling_effect(self):
        """팔-베이스 커플링: 팔 관절 운동이 베이스 θ에 영향"""
        state = np.zeros(9)
        control = np.zeros(8)
        control[2] = 2.0  # dq1 = 2.0 (큰 관절 속도)

        kin_dot = self.kin_model.forward_dynamics(state, control)
        dyn_dot = self.model.forward_dynamics(state, control)

        # 기구학은 ω=0이면 theta_dot=0
        np.testing.assert_allclose(kin_dot[2], 0.0, atol=1e-10)
        # 동역학은 커플링으로 theta_dot ≠ 0
        assert abs(dyn_dot[2]) > 1e-4

    def test_batch_dynamics(self):
        """배치 입력 처리"""
        K = 64
        states = np.random.randn(K, 9) * 0.3
        controls = np.random.randn(K, 8) * 0.5

        dots = self.model.forward_dynamics(states, controls)
        assert dots.shape == (K, 9)

        # 각 배치가 개별 계산과 일치하는지 확인
        for i in range(min(5, K)):
            dot_single = self.model.forward_dynamics(states[i], controls[i])
            np.testing.assert_allclose(dots[i], dot_single, atol=1e-10)

    def test_step_integration(self):
        """step() (RK4 적분) 동작 확인"""
        state = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 6, 0.0, 0.0, 0.0, 0.0])
        control = np.array([0.5, 0.1, 0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
        dt = 0.05

        next_state = self.model.step(state, control, dt)
        assert next_state.shape == (9,)
        assert not np.allclose(next_state, state)

    def test_zero_control_with_gravity(self):
        """제어 0에서도 중력 효과 발생"""
        state = np.zeros(9)
        state[4] = np.pi / 3  # q2 = 60도
        control = np.zeros(8)

        dot = self.model.forward_dynamics(state, control)

        # q2 중력 처짐만 발생, 나머지는 0에 가까워야
        assert abs(dot[4]) > 0.01  # 중력 효과
        np.testing.assert_allclose(dot[0], 0.0, atol=1e-10)  # x 정지

    def test_custom_parameters(self):
        """커스텀 물리 파라미터"""
        model = MobileManipulator6DOFDynamic(
            base_friction=0.5,
            base_response_k=10.0,
            joint_friction=np.array([0.3, 0.3, 0.2, 0.2, 0.1, 0.1]),
            coupling_gain=0.05,
            gravity_droop=0.15,
        )
        assert model.base_friction == 0.5
        assert model.coupling_gain == 0.05

        state = np.zeros(9)
        control = np.array([0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dot = model.forward_dynamics(state, control)
        assert dot.shape == (9,)

    def test_normalize_state_inherited(self):
        """normalize_state 상속 확인"""
        state = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        normalized = self.model.normalize_state(state)
        assert abs(normalized[2]) <= np.pi + 1e-6

    def test_get_control_bounds(self):
        """제어 제약 상속 확인"""
        bounds = self.model.get_control_bounds()
        assert bounds is not None
        lower, upper = bounds
        assert lower.shape == (8,)
        assert upper.shape == (8,)

    def test_repr(self):
        """__repr__ 동작 확인"""
        r = repr(self.model)
        assert "Dynamic" in r
        assert "base_friction" in r
