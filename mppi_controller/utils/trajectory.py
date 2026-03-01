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
    amplitude: float = 0.8,
    base_wavelength: float = 8.0,
    velocity: float = 0.45,
    chirp_rate: float = 0.003,
    v_max: float = 1.0,
) -> np.ndarray:
    """
    슬라럼 궤적 생성 (chirp sine 기반, 적응형 진폭)

    후반부로 갈수록 주파수가 증가하며, 진폭이 자동 감소하여
    전 구간에서 로봇의 속도 제약(v_max)을 준수.

    Args:
        t: 현재 시간 (초)
        amplitude: 최대 진폭 (m)
        base_wavelength: 초기 파장 (m)
        velocity: 전진 속도 (m/s)
        chirp_rate: 주파수 증가율 (Hz/s)
        v_max: 로봇 최대 속도 (m/s), 적응형 진폭 계산에 사용

    Returns:
        state: (3,) [x, y, θ]
    """
    f0 = velocity / base_wavelength
    f_inst = f0 + chirp_rate * t
    v_budget = 0.9 * v_max
    # 적응형 진폭: 순간 주파수 기반으로 v_max 이내 보장
    max_lateral_v = v_budget - velocity  # 횡방향 속도 예산
    if max_lateral_v > 0 and f_inst > 0:
        a_eff = min(amplitude, max_lateral_v / (2 * np.pi * f_inst))
    else:
        a_eff = amplitude

    x = velocity * t
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t ** 2)
    y = a_eff * np.sin(phase)

    # heading via numerical differentiation
    dt_num = 0.001
    t1 = t + dt_num
    f_inst1 = f0 + chirp_rate * t1
    if max_lateral_v > 0 and f_inst1 > 0:
        a_eff1 = min(amplitude, max_lateral_v / (2 * np.pi * f_inst1))
    else:
        a_eff1 = amplitude
    x1 = velocity * t1
    phase1 = 2 * np.pi * (f0 * t1 + 0.5 * chirp_rate * t1 ** 2)
    y1 = a_eff1 * np.sin(phase1)
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


def ee_circle_trajectory(
    t: float,
    radius: float = 0.5,
    angular_velocity: float = 0.3,
    center: tuple = (1.0, 0.0),
    state_dim: int = 5,
) -> np.ndarray:
    """
    End-Effector 원형 궤적

    ref[0]=ee_x, ref[1]=ee_y, 나머지=0
    MPPI의 EndEffectorTrackingCost가 ref[:2]를 EE 목표로 사용.

    Args:
        t: 현재 시간 (초)
        radius: 원 반지름 (m)
        angular_velocity: 각속도 (rad/s)
        center: 원 중심 (x, y)
        state_dim: 상태 차원 (기본 5: mobile manipulator)

    Returns:
        state: (state_dim,) - [ee_x, ee_y, 0, ...]
    """
    theta = angular_velocity * t
    ee_x = center[0] + radius * np.cos(theta)
    ee_y = center[1] + radius * np.sin(theta)

    state = np.zeros(state_dim)
    state[0] = ee_x
    state[1] = ee_y
    return state


def ee_figure_eight_trajectory(
    t: float,
    scale: float = 0.5,
    period: float = 20.0,
    center: tuple = (1.0, 0.0),
    state_dim: int = 5,
) -> np.ndarray:
    """
    End-Effector 8자 궤적 (Lemniscate)

    Args:
        t: 현재 시간 (초)
        scale: 스케일 (m)
        period: 주기 (초)
        center: 중심 (x, y)
        state_dim: 상태 차원

    Returns:
        state: (state_dim,) - [ee_x, ee_y, 0, ...]
    """
    theta = 2 * np.pi * t / period
    denom = 1 + np.sin(theta) ** 2
    ee_x = center[0] + scale * np.cos(theta) / denom
    ee_y = center[1] + scale * np.sin(theta) * np.cos(theta) / denom

    state = np.zeros(state_dim)
    state[0] = ee_x
    state[1] = ee_y
    return state


def ee_3d_circle_trajectory(
    t: float,
    radius: float = 0.3,
    angular_velocity: float = 0.3,
    center: tuple = (0.4, 0.0, 0.4),
    orientation: tuple = (np.pi, 0.0, 0.0),
    state_dim: int = 9,
) -> np.ndarray:
    """
    End-Effector 3D 원형 궤적 (XY 평면, 고정 z)

    ref[:3] = position (x, y, z)
    ref[3:6] = orientation (roll, pitch, yaw)
    ref[6:] = 0

    Args:
        t: 현재 시간 (초)
        radius: 원 반지름 (m)
        angular_velocity: 각속도 (rad/s)
        center: 원 중심 (x, y, z)
        orientation: 목표 RPY (roll, pitch, yaw)
        state_dim: 상태 차원 (기본 9: 6-DOF mobile manipulator)

    Returns:
        state: (state_dim,) - [ee_x, ee_y, ee_z, roll, pitch, yaw, 0, ...]
    """
    theta = angular_velocity * t
    ee_x = center[0] + radius * np.cos(theta)
    ee_y = center[1] + radius * np.sin(theta)
    ee_z = center[2]

    state = np.zeros(state_dim)
    state[0] = ee_x
    state[1] = ee_y
    state[2] = ee_z
    state[3] = orientation[0]
    state[4] = orientation[1]
    state[5] = orientation[2]
    return state


def ee_3d_helix_trajectory(
    t: float,
    radius: float = 0.3,
    angular_velocity: float = 0.3,
    center: tuple = (0.4, 0.0, 0.3),
    z_amplitude: float = 0.15,
    z_frequency: float = 0.3,
    orientation: tuple = (np.pi, 0.0, 0.0),
    state_dim: int = 9,
) -> np.ndarray:
    """
    End-Effector 3D 나선 궤적 (circle + sinusoidal z)

    Args:
        t: 현재 시간 (초)
        radius: 원 반지름 (m)
        angular_velocity: 각속도 (rad/s)
        center: 중심 (x, y, z_center)
        z_amplitude: z 방향 진폭 (m)
        z_frequency: z 방향 주파수 (Hz)
        orientation: 목표 RPY (roll, pitch, yaw)
        state_dim: 상태 차원

    Returns:
        state: (state_dim,) - [ee_x, ee_y, ee_z, roll, pitch, yaw, 0, ...]
    """
    theta = angular_velocity * t
    ee_x = center[0] + radius * np.cos(theta)
    ee_y = center[1] + radius * np.sin(theta)
    ee_z = center[2] + z_amplitude * np.sin(2 * np.pi * z_frequency * t)

    state = np.zeros(state_dim)
    state[0] = ee_x
    state[1] = ee_y
    state[2] = ee_z
    state[3] = orientation[0]
    state[4] = orientation[1]
    state[5] = orientation[2]
    return state


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
    elif trajectory_type == "ee_circle":
        return lambda t: ee_circle_trajectory(t, **kwargs)
    elif trajectory_type == "ee_figure8":
        return lambda t: ee_figure_eight_trajectory(t, **kwargs)
    elif trajectory_type == "ee_3d_circle":
        return lambda t: ee_3d_circle_trajectory(t, **kwargs)
    elif trajectory_type == "ee_3d_helix":
        return lambda t: ee_3d_helix_trajectory(t, **kwargs)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
