"""
Conformal Prediction + Shield-MPPI Controller

Shield-MPPI의 고정 안전 마진을 Conformal Prediction 기반 동적 마진으로 대체.
- 모델 정확 시: 마진 축소 → 불필요한 보수성 제거
- 모델 부정확 시: 마진 확대 → 안전성 향상
- 분포-무관 커버리지 보장: P(actual ∈ region) ≥ 1-α

Reference: ACP + Probabilistic CBF (arXiv:2407.03569)
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from mppi_controller.controllers.mppi.mppi_params import ConformalCBFMPPIParams
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.learning.conformal_predictor import (
    ConformalPredictor,
    ConformalPredictorConfig,
)
from mppi_controller.models.base_model import RobotModel


class ConformalCBFMPPIController(ShieldMPPIController):
    """
    CP + Shield-MPPI: 동적 안전 마진 보정

    매 제어 스텝마다:
    1. 이전 예측 vs 실제 상태로 CP 업데이트
    2. CP 마진으로 cbf_cost.safety_margin & cbf_params.cbf_safety_margin 갱신
    3. Shield-MPPI 제어 계산 (갱신된 마진 사용)
    4. 다음 스텝 예측 저장

    Args:
        model: RobotModel 인스턴스
        params: ConformalCBFMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 + CBF)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
        prediction_fn: 상태 예측 함수 (state, control) -> next_state
                       None이면 model.step 사용
    """

    def __init__(
        self,
        model: RobotModel,
        params: ConformalCBFMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        prediction_fn: Optional[Callable] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)

        self.cp_params = params

        # CP 인스턴스 생성
        self.conformal_predictor = ConformalPredictor(
            ConformalPredictorConfig(
                alpha=params.cp_alpha,
                window_size=params.cp_window_size,
                min_samples=params.cp_min_samples,
                gamma=params.cp_gamma,
                margin_min=params.cp_margin_min,
                margin_max=params.cp_margin_max,
                default_margin=params.cbf_safety_margin,
                score_type=params.cp_score_type,
            )
        )

        # 예측 함수: learned model 또는 nominal model
        self._prediction_fn = prediction_fn

        # 상태 추적 (이전 예측 vs 현재 관측)
        self._prev_predicted_next = None

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        CP + Shield-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - Shield-MPPI info + CP info
        """
        # Step 1: CP 업데이트 (이전 예측 vs 현재 실제 상태)
        if self._prev_predicted_next is not None and self.cp_params.cp_enabled:
            self.conformal_predictor.update(self._prev_predicted_next, state)

        # Step 2: 동적 마진 갱신
        if self.cp_params.cp_enabled:
            cp_margin = self.conformal_predictor.get_margin()
            self.cbf_cost.safety_margin = cp_margin
            self.cbf_params.cbf_safety_margin = cp_margin

        # Step 3: Shield-MPPI 제어 (부모 호출 — 갱신된 마진 사용)
        control, info = super().compute_control(state, reference_trajectory)

        # Step 4: 다음 스텝 예측 저장
        if self._prediction_fn is not None:
            self._prev_predicted_next = self._prediction_fn(state, control)
        else:
            self._prev_predicted_next = self.model.step(
                state, control, self.params.dt
            )

        # Step 5: CP 정보 info에 추가
        cp_stats = self.conformal_predictor.get_statistics()
        info["cp_margin"] = cp_stats["cp_margin"]
        info["cp_empirical_coverage"] = cp_stats["cp_empirical_coverage"]
        info["cp_num_scores"] = cp_stats["cp_num_scores"]
        info["cp_mean_score"] = cp_stats["cp_mean_score"]

        return control, info

    def get_cp_statistics(self) -> Dict:
        """CP 통계 반환"""
        return self.conformal_predictor.get_statistics()

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트 (부모 + CP 전파)"""
        super().update_obstacles(obstacles)

    def reset(self):
        """제어 시퀀스, 통계, CP 상태 초기화"""
        super().reset()
        self.conformal_predictor.reset()
        self._prev_predicted_next = None

    def __repr__(self) -> str:
        return (
            f"ConformalCBFMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"obstacles={len(self.cbf_params.cbf_obstacles)}, "
            f"cp_alpha={self.cp_params.cp_alpha}, "
            f"cp_gamma={self.cp_params.cp_gamma})"
        )
