"""
Ancillary Policies for Biased-MPPI

Biased-MPPI에서 사용하는 보조 정책 (ancillary policy) 인터페이스 및 내장 정책 구현.
각 정책은 N-step 제어 시퀀스를 제안하여 혼합 분포 샘플링에 참여.

Reference: Trevisan & Alonso-Mora, RA-L 2024, arXiv:2401.09241
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class AncillaryPolicy(ABC):
    """
    보조 정책 추상 인터페이스

    Biased-MPPI의 혼합 분포에 참여하는 정책.
    각 정책은 현재 상태와 레퍼런스로부터 N-step 제어 시퀀스를 제안.
    """

    @abstractmethod
    def propose_sequence(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
        N: int,
        dt: float,
        model,
    ) -> np.ndarray:
        """
        N-step 제어 시퀀스 제안

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적
            N: 호라이즌 길이
            dt: 타임스텝
            model: RobotModel 인스턴스

        Returns:
            controls: (N, nu) 제안된 제어 시퀀스
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """정책 이름"""
        pass


class PurePursuitPolicy(AncillaryPolicy):
    """
    Pure Pursuit 기반 보조 정책

    각 타임스텝에서 lookahead 거리만큼 앞의 레퍼런스 포인트를 향해
    pure pursuit (v, omega) 계산 후 forward simulation.
    """

    def __init__(self, lookahead: float = 0.5, v_gain: float = 1.0):
        self.lookahead = lookahead
        self.v_gain = v_gain

    @property
    def name(self) -> str:
        return "pure_pursuit"

    def propose_sequence(self, state, reference_trajectory, N, dt, model):
        nx = state.shape[0]
        nu = model.control_dim
        controls = np.zeros((N, nu))
        current_state = state.copy()

        # 제어 제약
        bounds = model.get_control_bounds()
        if bounds is not None:
            u_min, u_max = bounds
        else:
            u_min, u_max = None, None

        for t in range(N):
            # lookahead 인덱스 계산
            look_idx = min(t + max(1, int(self.lookahead / dt)), N)
            target = reference_trajectory[look_idx, :2]

            # 현재 위치→목표 방향
            dx = target[0] - current_state[0]
            dy = target[1] - current_state[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 1e-6:
                controls[t] = 0.0
            else:
                # 목표 각도
                target_angle = np.arctan2(dy, dx)

                # 각도 오차 (heading 있는 경우)
                if nx >= 3:
                    angle_error = target_angle - current_state[2]
                    angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
                else:
                    angle_error = 0.0

                # Pure pursuit: v와 omega 계산
                v = self.v_gain * dist
                omega = 2.0 * angle_error / max(dt, 0.01)

                if nu >= 2:
                    controls[t, 0] = v
                    controls[t, 1] = omega
                else:
                    controls[t, 0] = v

            # 제어 제약
            if u_min is not None:
                controls[t] = np.clip(controls[t], u_min, u_max)

            # Forward simulation
            state_dot = model.forward_dynamics(current_state, controls[t])
            current_state = current_state + state_dot * dt

        return controls


class BrakingPolicy(AncillaryPolicy):
    """
    비상 정지 (제로 제어) 정책

    모든 제어 입력을 0으로 설정. 안전한 기본 행동.
    """

    @property
    def name(self) -> str:
        return "braking"

    def propose_sequence(self, state, reference_trajectory, N, dt, model):
        nu = model.control_dim
        return np.zeros((N, nu))


class FeedbackPolicy(AncillaryPolicy):
    """
    피드백 기반 보조 정책

    기존 AncillaryController를 재사용하여 레퍼런스 추적 피드백 제어 생성.
    """

    def __init__(self, gain_scale: float = 1.0):
        self.gain_scale = gain_scale

    @property
    def name(self) -> str:
        return "feedback"

    def propose_sequence(self, state, reference_trajectory, N, dt, model):
        from mppi_controller.controllers.mppi.ancillary_controller import (
            create_default_ancillary_controller,
        )

        nu = model.control_dim
        nx = state.shape[0]
        controls = np.zeros((N, nu))
        current_state = state.copy()

        # 모델 타입 추정
        model_type = "dynamic" if nx > 3 else "kinematic"
        try:
            ac = create_default_ancillary_controller(model_type, self.gain_scale)
        except ValueError:
            ac = create_default_ancillary_controller("kinematic", self.gain_scale)

        bounds = model.get_control_bounds()
        if bounds is not None:
            u_min, u_max = bounds
        else:
            u_min, u_max = None, None

        for t in range(N):
            ref_state = reference_trajectory[min(t + 1, N), :nx]
            fb = ac.compute_feedback(current_state, ref_state)
            controls[t] = fb

            if u_min is not None:
                controls[t] = np.clip(controls[t], u_min, u_max)

            state_dot = model.forward_dynamics(current_state, controls[t])
            current_state = current_state + state_dot * dt

        return controls


class MaxSpeedPolicy(AncillaryPolicy):
    """
    최대 속도 추적 정책

    레퍼런스 방향으로 최대 속도로 이동. 적극적 탐색용.
    """

    def __init__(self, speed_ratio: float = 0.8):
        """
        Args:
            speed_ratio: 최대 속도 대비 비율 (0~1)
        """
        self.speed_ratio = speed_ratio

    @property
    def name(self) -> str:
        return "max_speed"

    def propose_sequence(self, state, reference_trajectory, N, dt, model):
        nx = state.shape[0]
        nu = model.control_dim
        controls = np.zeros((N, nu))
        current_state = state.copy()

        bounds = model.get_control_bounds()
        if bounds is not None:
            u_min, u_max = bounds
        else:
            u_min, u_max = None, None

        for t in range(N):
            target = reference_trajectory[min(t + 1, N), :2]
            dx = target[0] - current_state[0]
            dy = target[1] - current_state[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 1e-6:
                controls[t] = 0.0
            else:
                target_angle = np.arctan2(dy, dx)

                # 최대 속도
                if u_max is not None:
                    v = self.speed_ratio * u_max[0]
                else:
                    v = self.speed_ratio * 1.0

                # 각도 오차 기반 omega
                if nx >= 3:
                    angle_error = target_angle - current_state[2]
                    angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
                    omega = 3.0 * angle_error
                else:
                    omega = 0.0

                if nu >= 2:
                    controls[t, 0] = v
                    controls[t, 1] = omega
                else:
                    controls[t, 0] = v

            if u_min is not None:
                controls[t] = np.clip(controls[t], u_min, u_max)

            state_dot = model.forward_dynamics(current_state, controls[t])
            current_state = current_state + state_dot * dt

        return controls


class PreviousSolutionPolicy(AncillaryPolicy):
    """
    이전 솔루션 정책

    이전 최적 제어 시퀀스 U를 그대로 반환 (warm start).
    BiasedMPPIController가 set_previous_solution()으로 업데이트.
    """

    def __init__(self):
        self._previous_U = None

    @property
    def name(self) -> str:
        return "previous_solution"

    def set_previous_solution(self, U: np.ndarray):
        """이전 솔루션 설정"""
        self._previous_U = U.copy()

    def propose_sequence(self, state, reference_trajectory, N, dt, model):
        nu = model.control_dim
        if self._previous_U is not None and self._previous_U.shape == (N, nu):
            return self._previous_U.copy()
        return np.zeros((N, nu))


# ── 정책 레지스트리 ──────────────────────────────────────────

POLICY_REGISTRY: Dict[str, type] = {
    "pure_pursuit": PurePursuitPolicy,
    "braking": BrakingPolicy,
    "feedback": FeedbackPolicy,
    "max_speed": MaxSpeedPolicy,
    "previous_solution": PreviousSolutionPolicy,
}


def create_ancillary_policy(name: str, **kwargs) -> AncillaryPolicy:
    """
    이름으로 보조 정책 생성

    Args:
        name: 정책 이름 (POLICY_REGISTRY 키)
        **kwargs: 정책별 파라미터

    Returns:
        AncillaryPolicy 인스턴스
    """
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy: {name}. "
            f"Available: {list(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[name](**kwargs)


def create_policies_from_names(names: List[str]) -> List[AncillaryPolicy]:
    """
    이름 리스트로 보조 정책들 생성

    Args:
        names: 정책 이름 리스트

    Returns:
        AncillaryPolicy 인스턴스 리스트
    """
    return [create_ancillary_policy(name) for name in names]
