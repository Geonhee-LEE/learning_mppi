"""
레퍼런스 궤적 생성 유틸리티

다양한 테스트 궤적 생성 함수.
"""

import numpy as np
from typing import Callable


def circle_trajectory(
    t: float, radius: float = 5.0, angular_velocity: float = 0.1, center=(0.0, 0.0)
) -> np.ndarray:
    """
    원형 궤적 생성

    Args:
        t: 현재 시간 (초)
        radius: 원 반지름 (m)
        angular_velocity: 각속도 (rad/s)
        center: 원 중심 (x, y)

    Returns:
        state: (3,) [x, y, θ]
    """
    theta = angular_velocity * t
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    heading = theta + np.pi / 2  # 속도 방향

    return np.array([x, y, heading])


def figure_eight_trajectory(
    t: float, scale: float = 5.0, period: float = 20.0
) -> np.ndarray:
    """
    8자 궤적 (Lemniscate) 생성

    Args:
        t: 현재 시간 (초)
        scale: 스케일 (m)
        period: 주기 (초)

    Returns:
        state: (3,) [x, y, θ]
    """
    theta = 2 * np.pi * t / period

    # Lemniscate 파라미터 방정식
    denom = 1 + np.sin(theta) ** 2
    x = scale * np.cos(theta) / denom
    y = scale * np.sin(theta) * np.cos(theta) / denom

    # 속도 방향 (수치 미분)
    dt = 0.01
    theta_next = 2 * np.pi * (t + dt) / period
    denom_next = 1 + np.sin(theta_next) ** 2
    x_next = scale * np.cos(theta_next) / denom_next
    y_next = scale * np.sin(theta_next) * np.cos(theta_next) / denom_next

    heading = np.arctan2(y_next - y, x_next - x)

    return np.array([x, y, heading])


def sine_wave_trajectory(
    t: float, amplitude: float = 2.0, wavelength: float = 10.0, velocity: float = 1.0
) -> np.ndarray:
    """
    사인파 궤적 생성

    Args:
        t: 현재 시간 (초)
        amplitude: 진폭 (m)
        wavelength: 파장 (m)
        velocity: 전진 속도 (m/s)

    Returns:
        state: (3,) [x, y, θ]
    """
    x = velocity * t
    y = amplitude * np.sin(2 * np.pi * x / wavelength)

    # 미분: dy/dx = (2π A / λ) cos(2π x / λ)
    dy_dx = (2 * np.pi * amplitude / wavelength) * np.cos(2 * np.pi * x / wavelength)
    heading = np.arctan2(dy_dx, 1.0)

    return np.array([x, y, heading])


def slalom_trajectory(
    t: float,
    amplitude: float = 2.5,
    base_wavelength: float = 8.0,
    velocity: float = 0.8,
    chirp_rate: float = 0.008,
) -> np.ndarray:
    """
    슬라럼 궤적 생성 (chirp sine 기반)

    후반부로 갈수록 주파수가 증가하여 급격한 곡률 변화를 생성.

    Args:
        t: 현재 시간 (초)
        amplitude: 진폭 (m)
        base_wavelength: 초기 파장 (m)
        velocity: 전진 속도 (m/s)
        chirp_rate: 주파수 증가율 (Hz/s)

    Returns:
        state: (3,) [x, y, θ]
    """
    f0 = velocity / base_wavelength
    x = velocity * t
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t ** 2)
    y = amplitude * np.sin(phase)

    # heading via numerical differentiation
    dt_num = 0.001
    t1 = t + dt_num
    x1 = velocity * t1
    phase1 = 2 * np.pi * (f0 * t1 + 0.5 * chirp_rate * t1 ** 2)
    y1 = amplitude * np.sin(phase1)
    heading = np.arctan2(y1 - y, x1 - x)

    return np.array([x, y, heading])


def straight_line_trajectory(
    t: float, velocity: float = 1.0, heading: float = 0.0, start=(0.0, 0.0)
) -> np.ndarray:
    """
    직선 궤적 생성

    Args:
        t: 현재 시간 (초)
        velocity: 속도 (m/s)
        heading: 방향 (rad)
        start: 시작 위치 (x, y)

    Returns:
        state: (3,) [x, y, θ]
    """
    x = start[0] + velocity * t * np.cos(heading)
    y = start[1] + velocity * t * np.sin(heading)

    return np.array([x, y, heading])


def generate_reference_trajectory(
    trajectory_fn: Callable[[float], np.ndarray],
    t_current: float,
    N: int,
    dt: float,
) -> np.ndarray:
    """
    레퍼런스 궤적 생성 (N+1 스텝)

    Args:
        trajectory_fn: t → (nx,) 궤적 함수
        t_current: 현재 시간 (초)
        N: 호라이즌 길이
        dt: 타임스텝 간격 (초)

    Returns:
        reference: (N+1, nx) 레퍼런스 궤적
    """
    times = np.arange(N + 1) * dt + t_current
    reference = np.array([trajectory_fn(t) for t in times])

    return reference


def create_trajectory_function(
    trajectory_type: str, **kwargs
) -> Callable[[float], np.ndarray]:
    """
    궤적 타입에 따라 함수 생성

    Args:
        trajectory_type: 'circle', 'figure8', 'sine', 'slalom', 'straight'
        **kwargs: 궤적 파라미터

    Returns:
        trajectory_fn: t → (nx,) 함수
    """
    if trajectory_type == "circle":
        return lambda t: circle_trajectory(t, **kwargs)
    elif trajectory_type == "figure8":
        return lambda t: figure_eight_trajectory(t, **kwargs)
    elif trajectory_type == "sine":
        return lambda t: sine_wave_trajectory(t, **kwargs)
    elif trajectory_type == "slalom":
        return lambda t: slalom_trajectory(t, **kwargs)
    elif trajectory_type == "straight":
        return lambda t: straight_line_trajectory(t, **kwargs)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
