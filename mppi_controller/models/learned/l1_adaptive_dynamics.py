"""
L1 Adaptive Control Dynamics — 상태 예측기 + 외란 추정 + 저역통과 필터

L1 적응 제어 이론에 기반한 외란 추정 및 보상 모델.
공칭 모델(DynamicKinematicAdapter)과 실제 관측의 차이를 실시간으로 추정하여
MPPI rollout에서 보정된 동역학을 제공.

오프라인 학습이 전혀 필요 없으며, 매 타임스텝 갱신으로 즉시 동작.

Usage:
    l1 = L1AdaptiveDynamics(c_v_nom=0.1, c_omega_nom=0.1)

    # 매 스텝 업데이트
    l1.update_step(state_5d, control, next_state_5d, dt)

    # MPPI rollout에 사용 (보정된 동역학)
    state_dot = l1.forward_dynamics(state, control)
"""

import numpy as np
from mppi_controller.models.base_model import RobotModel
from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter


class L1AdaptiveDynamics(RobotModel):
    """
    L1 적응 제어 기반 동역학 모델.

    구성:
    1. 상태 예측기: x̂_dot = f_nom(x, u) + A_m·(x̂ - x) + σ̂
    2. 적응 법칙: σ̂ += -Γ · x̃ · dt (예측 오차 기반)
    3. 저역통과 필터: σ_f += ω_c·(σ̂ - σ_f)·dt
    4. 보정 출력: f_total = f_nom + σ_filtered

    A_m은 Hurwitz 안정 행렬로, 예측 오차가 지수적으로 수렴 보장.

    Args:
        c_v_nom: 공칭 c_v (잘못된 파라미터)
        c_omega_nom: 공칭 c_omega
        k_v: PD 선형 속도 게인
        k_omega: PD 각속도 게인
        adaptation_gain: Γ (적응 게인, 클수록 빠른 추적)
        cutoff_freq: ω_c (저역통과 차단 주파수, rad/s)
        am_gains: A_m 대각 요소 (Hurwitz 안정 행렬)
    """

    def __init__(
        self,
        c_v_nom=0.1,
        c_omega_nom=0.1,
        k_v=5.0,
        k_omega=5.0,
        adaptation_gain=100.0,
        cutoff_freq=10.0,
        am_gains=None,
    ):
        # 공칭 모델
        self._nominal = DynamicKinematicAdapter(
            c_v=c_v_nom, c_omega=c_omega_nom, k_v=k_v, k_omega=k_omega
        )

        # 적응 파라미터
        self._Gamma = adaptation_gain
        self._omega_c = cutoff_freq

        # Hurwitz 안정 행렬 A_m (음의 대각)
        if am_gains is None:
            am_gains = np.array([-5.0, -5.0, -5.0, -10.0, -10.0])
        self._A_m = np.diag(am_gains)

        # 상태 예측기
        self._x_hat = np.zeros(5)  # 예측 상태
        self._sigma_hat = np.zeros(5)  # 추정 외란
        self._sigma_filtered = np.zeros(5)  # 필터링된 외란

        self._initialized = False

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
        """
        보정된 동역학: f_nom(x, u) + σ_filtered.

        MPPI rollout에서 사용. 단일/배치 지원.
        """
        nom_dot = self._nominal.forward_dynamics(state, control)

        if state.ndim == 1:
            return nom_dot + self._sigma_filtered
        else:
            return nom_dot + self._sigma_filtered[np.newaxis, :]

    def update_step(self, state_5d, control, next_state_5d, dt):
        """
        단일 L1 적응 업데이트.

        1. 예측 오차 계산: x̃ = x̂ - x_actual
        2. 적응 법칙: σ̂ += -Γ·x̃·dt
        3. 저역통과: σ_f += ω_c·(σ̂ - σ_f)·dt
        4. 상태 예측기 전진

        Args:
            state_5d: (5,) 현재 관측
            control: (2,) 제어 입력
            next_state_5d: (5,) 다음 관측
            dt: 시간 간격
        """
        if not self._initialized:
            self._x_hat = state_5d.copy()
            self._initialized = True

        # 1. 예측 오차
        x_tilde = self._x_hat - state_5d
        # 각도 래핑
        x_tilde[2] = np.arctan2(np.sin(x_tilde[2]), np.cos(x_tilde[2]))

        # 2. 적응 법칙: σ̂ += -Γ · x̃ · dt
        self._sigma_hat += -self._Gamma * x_tilde * dt

        # 3. 저역통과 필터: σ_f += ω_c·(σ̂ - σ_f)·dt
        self._sigma_filtered += self._omega_c * (self._sigma_hat - self._sigma_filtered) * dt

        # 4. 상태 예측기 전진
        # x̂_dot = f_nom(x, u) + A_m·(x̂ - x) + σ̂
        nom_dot = self._nominal.forward_dynamics(state_5d, control)
        x_hat_dot = nom_dot + self._A_m @ x_tilde + self._sigma_hat
        self._x_hat = state_5d + x_hat_dot * dt

        # 각도 정규화
        self._x_hat[2] = np.arctan2(np.sin(self._x_hat[2]), np.cos(self._x_hat[2]))

    def adapt(self, states, controls, next_states, dt, **kwargs):
        """
        배치 적응: M개 관측을 순차적 업데이트.

        MAML/EKF와 동일한 인터페이스.

        Args:
            states: (M, 5) 상태
            controls: (M, 2) 제어
            next_states: (M, 5) 다음 상태
            dt: 시간 간격

        Returns:
            float: 최종 예측 오차 norm
        """
        M = states.shape[0]
        final_error = 0.0
        for i in range(M):
            self.update_step(states[i], controls[i], next_states[i], dt)
            # 오차 기록
            pred_next = states[i] + self.forward_dynamics(states[i], controls[i]) * dt
            err = next_states[i] - pred_next
            err[2] = np.arctan2(np.sin(err[2]), np.cos(err[2]))
            final_error = float(np.linalg.norm(err))
        return final_error

    def get_disturbance_estimate(self):
        """추정 외란 반환."""
        return {
            "sigma_hat": self._sigma_hat.copy(),
            "sigma_filtered": self._sigma_filtered.copy(),
            "sigma_norm": float(np.linalg.norm(self._sigma_filtered)),
        }

    def get_prediction_error(self):
        """현재 예측 오차 norm."""
        return float(np.linalg.norm(self._x_hat[:2]))

    def reset(self):
        """L1 적응 상태 초기화."""
        self._x_hat = np.zeros(5)
        self._sigma_hat = np.zeros(5)
        self._sigma_filtered = np.zeros(5)
        self._initialized = False

    def is_stable(self):
        """A_m의 Hurwitz 안정성 확인 (모든 고유값 실수부 < 0)."""
        eigvals = np.linalg.eigvals(self._A_m)
        return bool(np.all(np.real(eigvals) < 0))

    def get_control_bounds(self):
        return self._nominal.get_control_bounds()

    def state_to_dict(self, state):
        return self._nominal.state_to_dict(state)

    def normalize_state(self, state):
        return self._nominal.normalize_state(state)

    def __repr__(self) -> str:
        dist = self.get_disturbance_estimate()
        return (
            f"L1AdaptiveDynamics("
            f"Γ={self._Gamma}, ω_c={self._omega_c}, "
            f"|σ_f|={dist['sigma_norm']:.4f})"
        )
