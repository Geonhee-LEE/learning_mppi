"""
Flow-MPPI 데이터 수집기

MPPI 실행 중 (state, optimal_control_sequence) 쌍을 수집하여
Flow Matching 모델 학습에 사용.

기존 DataCollector와 차이: 단일 전이가 아닌 전체 제어 시퀀스 (N, nu) 저장.
"""

import numpy as np
from typing import Tuple, Optional


class FlowDataCollector:
    """
    Flow Matching 학습 데이터 수집기 (ring buffer)

    각 MPPI 스텝에서 (state, U_optimal) 쌍을 저장.

    Args:
        buffer_size: 최대 버퍼 크기
    """

    def __init__(self, buffer_size: int = 5000):
        self.buffer_size = buffer_size
        self._states = []
        self._controls = []
        self._count = 0
        self._idx = 0

    def add_sample(self, state: np.ndarray, optimal_U: np.ndarray):
        """
        샘플 추가

        Args:
            state: (nx,) 현재 상태
            optimal_U: (N, nu) 최적 제어 시퀀스
        """
        if self._count < self.buffer_size:
            self._states.append(state.copy())
            self._controls.append(optimal_U.copy())
            self._count += 1
        else:
            # Ring buffer: 가장 오래된 데이터 덮어쓰기
            self._states[self._idx] = state.copy()
            self._controls[self._idx] = optimal_U.copy()
        self._idx = (self._idx + 1) % self.buffer_size

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터 반환

        Returns:
            states: (M, nx) 상태 배열
            controls: (M, N, nu) 제어 시퀀스 배열
        """
        states = np.array(self._states[:self._count])
        controls = np.array(self._controls[:self._count])
        return states, controls

    def should_train(self, min_samples: int = 200) -> bool:
        """학습에 충분한 데이터가 있는지 확인"""
        return self._count >= min_samples

    @property
    def num_samples(self) -> int:
        return self._count

    def clear(self):
        """버퍼 초기화"""
        self._states.clear()
        self._controls.clear()
        self._count = 0
        self._idx = 0

    def __repr__(self) -> str:
        return f"FlowDataCollector(count={self._count}/{self.buffer_size})"
