"""
로봇 모델 추상 베이스 클래스

모든 로봇 모델 (Kinematic/Dynamic/Learned)이 상속하는 추상 클래스.
통일된 인터페이스로 BatchDynamicsWrapper가 모든 타입을 지원.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class RobotModel(ABC):
    """
    로봇 모델 추상 베이스 클래스

    모든 모델 타입 (kinematic/dynamic/learned)이 구현해야 하는 인터페이스 정의.
    통일된 인터페이스로 BatchDynamicsWrapper가 모든 타입 지원.

    Attributes:
        state_dim (int): 상태 벡터 차원 (nx)
        control_dim (int): 제어 벡터 차원 (nu)
        model_type (str): 'kinematic', 'dynamic', 'learned'
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """상태 벡터 차원 (nx)"""
        pass

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """제어 벡터 차원 (nu)"""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """모델 타입: 'kinematic', 'dynamic', 'learned'"""
        pass

    @abstractmethod
    def forward_dynamics(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        연속 시간 동역학: dx/dt = f(x, u)

        Args:
            state: (nx,) 또는 (batch, nx) 상태 벡터
            control: (nu,) 또는 (batch, nu) 제어 벡터

        Returns:
            state_dot: (nx,) 또는 (batch, nx) 상태 미분
        """
        pass

    def step(
        self, state: np.ndarray, control: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        이산 시간 적분: x_{t+1} = x_t + ∫ f(x, u) dt (RK4)

        기본 구현으로 RK4 적분 제공. 서브클래스에서 오버라이드 가능.

        Args:
            state: (nx,) 또는 (batch, nx) 현재 상태
            control: (nu,) 또는 (batch, nu) 제어 입력
            dt: 시간 간격 (초)

        Returns:
            next_state: (nx,) 또는 (batch, nx) 다음 상태
        """
        # RK4 적분
        k1 = self.forward_dynamics(state, control)
        k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.forward_dynamics(state + dt * k3, control)

        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        제어 입력 제약

        Returns:
            (lower_bound, upper_bound): 각각 (nu,) 배열
            None이면 제약 없음
        """
        return None

    def state_to_dict(self, state: np.ndarray) -> dict:
        """
        상태 벡터를 딕셔너리로 변환 (디버깅/시각화용)

        Args:
            state: (nx,) 상태 벡터

        Returns:
            state_dict: {"x": ..., "y": ..., "theta": ...} 형태
        """
        return {"state": state}

    def render_config(self) -> dict:
        """
        시각화용 렌더링 설정 반환.

        Returns:
            {"shape": str, "radius": float, "color": str, ...}
        """
        return {"shape": "circle", "radius": 0.15, "color": "blue"}

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        상태 정규화 (예: 각도 [-π, π] 범위로)

        기본 구현은 그대로 반환. 서브클래스에서 오버라이드 필요시 구현.

        Args:
            state: (nx,) 또는 (batch, nx) 상태 벡터

        Returns:
            normalized_state: (nx,) 또는 (batch, nx)
        """
        return state
