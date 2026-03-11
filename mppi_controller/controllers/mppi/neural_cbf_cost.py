"""
Neural CBF 비용 함수

학습된 Neural CBF h(x)를 사용하여 discrete-time CBF 조건 위반을 비용으로 변환.
ControlBarrierCost의 drop-in 대체.
"""

import numpy as np
import torch
from typing import Dict, Optional
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.learning.neural_cbf_trainer import NeuralCBFTrainer


class NeuralBarrierCost(CostFunction):
    """
    Neural CBF 기반 MPPI 비용 함수

    학습된 h(x) 네트워크로 barrier 값 계산.
    ControlBarrierCost와 동일한 discrete CBF 조건 적용:
        h(x_{t+1}) - (1-α)·h(x_t) ≥ 0

    Args:
        neural_cbf_trainer: 학습된 NeuralCBFTrainer 인스턴스
        cbf_alpha: Class-K function 파라미터 (0 < α ≤ 1)
        cbf_weight: CBF 위반 비용 가중치
        state_indices: 사용할 상태 인덱스 (None이면 전체 사용)
    """

    def __init__(
        self,
        neural_cbf_trainer: NeuralCBFTrainer,
        cbf_alpha: float = 0.1,
        cbf_weight: float = 1000.0,
        state_indices: Optional[list] = None,
    ):
        self.trainer = neural_cbf_trainer
        self.cbf_alpha = cbf_alpha
        self.cbf_weight = cbf_weight
        self.state_indices = state_indices

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Neural CBF 위반 비용 계산 (벡터화)

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스

        Returns:
            costs: (K,) 각 샘플의 CBF 비용
        """
        K, N_plus_1, nx = trajectories.shape

        # State selection
        if self.state_indices is not None:
            states = trajectories[:, :, self.state_indices]
        else:
            states = trajectories

        # Flatten → batch NN → reshape
        flat_states = states.reshape(-1, states.shape[-1])  # (K*(N+1), nx)
        h_flat = self.trainer.predict_h(flat_states)  # (K*(N+1),)
        h = h_flat.reshape(K, N_plus_1)  # (K, N+1)

        # Discrete CBF 조건: h[:, 1:] - (1-α)·h[:, :-1] ≥ 0
        cbf_condition = h[:, 1:] - (1.0 - self.cbf_alpha) * h[:, :-1]  # (K, N)
        violation = np.maximum(0.0, -cbf_condition)

        costs = self.cbf_weight * np.sum(violation, axis=1)  # (K,)
        return costs

    def get_barrier_info(self, trajectories: np.ndarray) -> Dict:
        """
        Barrier 정보 반환 (ControlBarrierCost 호환)

        Args:
            trajectories: (K, N+1, nx) 또는 (N+1, nx)

        Returns:
            dict: barrier_values, min_barrier, is_safe
        """
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, :, :]

        if self.state_indices is not None:
            states = trajectories[:, :, self.state_indices]
        else:
            states = trajectories

        K, N_plus_1, sdim = states.shape
        flat_states = states.reshape(-1, sdim)
        h_flat = self.trainer.predict_h(flat_states)
        h = h_flat.reshape(K, N_plus_1)

        min_barrier = float(np.min(h))

        return {
            "barrier_values": h,
            "min_barrier": min_barrier,
            "is_safe": bool(min_barrier > 0),
        }

    def update_network(self, neural_cbf_trainer: NeuralCBFTrainer):
        """네트워크 업데이트 (온라인 재학습 후)"""
        self.trainer = neural_cbf_trainer

    def __repr__(self) -> str:
        return (
            f"NeuralBarrierCost("
            f"alpha={self.cbf_alpha}, "
            f"weight={self.cbf_weight})"
        )
