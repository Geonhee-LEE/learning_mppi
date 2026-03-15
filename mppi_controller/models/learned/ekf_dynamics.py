"""
EKF Adaptive Dynamics — 확장 칼만 필터 기반 파라미터 추정

7D 확장 상태 [x, y, θ, v, ω, ĉ_v, ĉ_ω]에서 마찰 계수 c_v, c_omega를
실시간으로 추정하여 DynamicKinematicAdapter의 파라미터를 갱신.

오프라인 학습이 전혀 필요 없으며, 관측 데이터만으로 즉시 적응 가능.

Usage:
    ekf = EKFAdaptiveDynamics(c_v_init=0.1, c_omega_init=0.1)

    # 매 스텝 업데이트
    ekf.update_step(state_5d, control, next_state_5d, dt)

    # MPPI rollout에 사용
    state_dot = ekf.forward_dynamics(state, control)

    # 추정 결과
    estimates = ekf.get_parameter_estimates()
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter


class EKFAdaptiveDynamics(RobotModel):
    """
    EKF 기반 적응 동역학 모델.

    7D 확장 상태: [x, y, θ, v, ω, ĉ_v, ĉ_ω]
    - 5D 관측: [x, y, θ, v, ω]
    - 2D 추정 파라미터: [ĉ_v, ĉ_ω]

    파라미터 추정 후 DynamicKinematicAdapter에 반영하여 MPPI rollout 수행.

    Args:
        c_v_init: 초기 c_v 추정값
        c_omega_init: 초기 c_omega 추정값
        k_v: PD 선형 속도 게인
        k_omega: PD 각속도 게인
        Q_process: 7D 프로세스 노이즈 공분산 대각 요소
        R_obs: 5D 관측 노이즈 공분산 대각 요소
        param_bounds: 파라미터 범위 dict {c_v: (min, max), c_omega: (min, max)}
    """

    def __init__(
        self,
        c_v_init=0.1,
        c_omega_init=0.1,
        k_v=5.0,
        k_omega=5.0,
        Q_process=None,
        R_obs=None,
        param_bounds=None,
    ):
        self._k_v = k_v
        self._k_omega = k_omega

        # 7D 확장 상태: [x, y, θ, v, ω, ĉ_v, ĉ_ω]
        self._ekf_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, c_v_init, c_omega_init])

        # 공분산 초기화
        self._P = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 1.0, 1.0])

        # 프로세스 노이즈
        if Q_process is None:
            Q_process = np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.001, 0.001])
        self._Q = np.diag(Q_process)

        # 관측 노이즈
        if R_obs is None:
            R_obs = np.array([0.01, 0.01, 0.005, 0.02, 0.01])
        self._R = np.diag(R_obs)

        # 파라미터 범위
        if param_bounds is None:
            param_bounds = {"c_v": (0.01, 2.0), "c_omega": (0.01, 1.5)}
        self._param_bounds = param_bounds

        # MPPI 내부 모델 (추정 파라미터 사용)
        self._adapter = DynamicKinematicAdapter(
            c_v=c_v_init, c_omega=c_omega_init, k_v=k_v, k_omega=k_omega
        )

        # 관측 행렬: H = [I_5x5 | 0_5x2]
        self._H = np.zeros((5, 7))
        self._H[:5, :5] = np.eye(5)

    @property
    def state_dim(self) -> int:
        return 5

    @property
    def control_dim(self) -> int:
        return 2

    @property
    def model_type(self) -> str:
        return "learned"

    def forward_dynamics(self, state, control):
        """추정된 파라미터의 adapter로 5D forward dynamics."""
        return self._adapter.forward_dynamics(state, control)

    def step(self, state, control, dt):
        """추정된 파라미터의 adapter로 step."""
        return self._adapter.step(state, control, dt)

    def _f_7d(self, ekf_state, control, dt):
        """
        7D 상태 전이 함수.

        state = [x, y, θ, v, ω, ĉ_v, ĉ_ω]
        control = [v_cmd, ω_cmd]
        """
        x, y, theta, v, omega, c_v, c_omega = ekf_state
        v_cmd, omega_cmd = control

        # PD + friction 동역학
        a = self._k_v * (v_cmd - v) - c_v * v
        alpha = self._k_omega * (omega_cmd - omega) - c_omega * omega

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        dv = a
        domega = alpha
        # 파라미터는 상수 (random walk)
        dc_v = 0.0
        dc_omega = 0.0

        state_dot = np.array([dx, dy, dtheta, dv, domega, dc_v, dc_omega])
        return ekf_state + state_dot * dt

    def _jacobian_F(self, ekf_state, control, dt):
        """
        상태 전이 야코비안 ∂f/∂x (7x7).

        해석적 계산: 대부분 0, 핵심은 v, ω, c_v, c_omega 관련 항.
        """
        _, _, theta, v, omega, c_v, c_omega = ekf_state
        v_cmd, omega_cmd = control

        F = np.eye(7)

        # ∂dx/∂θ = -v*sin(θ)*dt, ∂dx/∂v = cos(θ)*dt
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt

        # ∂dy/∂θ = v*cos(θ)*dt, ∂dy/∂v = sin(θ)*dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt

        # ∂dθ/∂ω = dt
        F[2, 4] = dt

        # ∂dv/∂v = 1 + (-k_v - c_v)*dt
        F[3, 3] = 1.0 + (-self._k_v - c_v) * dt
        # ∂dv/∂ĉ_v = -v*dt
        F[3, 5] = -v * dt

        # ∂dω/∂ω = 1 + (-k_ω - c_ω)*dt
        F[4, 4] = 1.0 + (-self._k_omega - c_omega) * dt
        # ∂dω/∂ĉ_ω = -ω*dt
        F[4, 6] = -omega * dt

        return F

    def update_step(self, state_5d, control, next_state_5d, dt):
        """
        단일 EKF 업데이트 (predict → innovate → update).

        Args:
            state_5d: (5,) 현재 관측 [x, y, θ, v, ω]
            control: (2,) 제어 입력 [v_cmd, ω_cmd]
            next_state_5d: (5,) 다음 관측
            dt: 시간 간격
        """
        # EKF 내부 상태의 관측 부분을 현재 관측으로 동기화
        self._ekf_state[:5] = state_5d.copy()

        # === Predict ===
        x_pred = self._f_7d(self._ekf_state, control, dt)
        F = self._jacobian_F(self._ekf_state, control, dt)
        P_pred = F @ self._P @ F.T + self._Q

        # === Update ===
        z = next_state_5d  # 관측
        z_pred = x_pred[:5]  # 예측 관측 (H @ x_pred)

        # Innovation
        innovation = z - z_pred
        # 각도 래핑
        innovation[2] = np.arctan2(np.sin(innovation[2]), np.cos(innovation[2]))

        # Innovation 공분산
        S = self._H @ P_pred @ self._H.T + self._R

        # Kalman gain
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        # 상태 업데이트
        self._ekf_state = x_pred + K @ innovation

        # Joseph form 공분산 업데이트 (수치 안정)
        I_KH = np.eye(7) - K @ self._H
        self._P = I_KH @ P_pred @ I_KH.T + K @ self._R @ K.T

        # 대칭 강제
        self._P = 0.5 * (self._P + self._P.T)

        # 파라미터 범위 클리핑
        self._clip_parameters()

        # Adapter 갱신
        self._sync_adapter()

    def adapt(self, states, controls, next_states, dt, **kwargs):
        """
        배치 적응: M개 관측을 순차적 EKF 업데이트.

        MAML과 동일한 인터페이스 호환.

        Args:
            states: (M, 5) 상태
            controls: (M, 2) 제어
            next_states: (M, 5) 다음 상태
            dt: 시간 간격

        Returns:
            float: 최종 innovation norm
        """
        M = states.shape[0]
        final_innov = 0.0
        for i in range(M):
            self.update_step(states[i], controls[i], next_states[i], dt)
            # innovation norm 기록
            z_pred = self._f_7d(
                np.concatenate([states[i], self._ekf_state[5:]]), controls[i], dt
            )[:5]
            innov = next_states[i] - z_pred
            innov[2] = np.arctan2(np.sin(innov[2]), np.cos(innov[2]))
            final_innov = float(np.linalg.norm(innov))
        return final_innov

    def _clip_parameters(self):
        """파라미터 범위 클리핑."""
        cv_min, cv_max = self._param_bounds["c_v"]
        co_min, co_max = self._param_bounds["c_omega"]
        self._ekf_state[5] = np.clip(self._ekf_state[5], cv_min, cv_max)
        self._ekf_state[6] = np.clip(self._ekf_state[6], co_min, co_max)

    def _sync_adapter(self):
        """추정 파라미터를 adapter에 동기화."""
        self._adapter._c_v = float(self._ekf_state[5])
        self._adapter._c_omega = float(self._ekf_state[6])

    def get_parameter_estimates(self):
        """추정 파라미터 + 불확실성 반환."""
        return {
            "c_v": float(self._ekf_state[5]),
            "c_omega": float(self._ekf_state[6]),
            "c_v_std": float(np.sqrt(max(self._P[5, 5], 0.0))),
            "c_omega_std": float(np.sqrt(max(self._P[6, 6], 0.0))),
        }

    def get_covariance(self):
        """전체 7x7 공분산 행렬 반환."""
        return self._P.copy()

    def reset(self, c_v_init=None, c_omega_init=None):
        """EKF 상태 초기화."""
        if c_v_init is None:
            c_v_init = 0.1
        if c_omega_init is None:
            c_omega_init = 0.1
        self._ekf_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, c_v_init, c_omega_init])
        self._P = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 1.0, 1.0])
        self._adapter._c_v = c_v_init
        self._adapter._c_omega = c_omega_init

    def get_control_bounds(self):
        return self._adapter.get_control_bounds()

    def state_to_dict(self, state):
        return self._adapter.state_to_dict(state)

    def normalize_state(self, state):
        return self._adapter.normalize_state(state)

    def __repr__(self) -> str:
        est = self.get_parameter_estimates()
        return (
            f"EKFAdaptiveDynamics("
            f"ĉ_v={est['c_v']:.3f}±{est['c_v_std']:.3f}, "
            f"ĉ_ω={est['c_omega']:.3f}±{est['c_omega_std']:.3f})"
        )
