"""
Conformal Prediction 모듈

온라인 적응형 Conformal Prediction으로 모델 예측 불확실성을 정량화.
분포-무관(distribution-free) 커버리지 보장: P(actual ∈ region) ≥ 1-α

Reference: Adaptive Conformal Prediction + Probabilistic CBF (arXiv:2407.03569)
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ConformalPredictorConfig:
    """Conformal Predictor 설정"""

    alpha: float = 0.1  # 실패율 (0.1 → 90% 커버리지)
    window_size: int = 200  # 슬라이딩 윈도우 크기
    min_samples: int = 10  # CP 활성화 최소 샘플 수
    default_margin: float = 0.15  # cold start 기본 마진 (m)
    gamma: float = 0.95  # ACP 지수 감쇠 (1.0=표준CP, <1.0=적응형)
    margin_min: float = 0.02  # 최소 마진 클램프 (m)
    margin_max: float = 0.5  # 최대 마진 클램프 (m)
    score_type: str = "position_norm"  # 비순응 점수 유형


class ConformalPredictor:
    """
    온라인 적응형 Conformal Predictor

    모델 예측과 실제 관측의 비순응 점수(nonconformity score)를 추적하여
    통계적 커버리지 보장을 제공하는 동적 마진을 계산.

    - 표준 CP (gamma=1.0): quantile(scores, (n+1)(1-α)/n)
    - 적응형 CP (gamma<1.0): 지수 가중 quantile로 최근 데이터에 더 반응
    """

    def __init__(self, config: ConformalPredictorConfig = None):
        self.config = config or ConformalPredictorConfig()
        self._scores = deque(maxlen=self.config.window_size)
        self._coverage_hits = deque(maxlen=self.config.window_size)
        self._margin = self.config.default_margin
        self._step_count = 0

    def update(self, predicted_state: np.ndarray, actual_state: np.ndarray) -> None:
        """
        새 관측으로 CP 상태 업데이트

        Args:
            predicted_state: 모델이 예측한 다음 상태 (nx,)
            actual_state: 실제 관측된 상태 (nx,)
        """
        score = self._compute_score(predicted_state, actual_state)
        self._scores.append(score)
        self._step_count += 1

        # 커버리지 추적: 이전 마진 안에 들어왔는지
        self._coverage_hits.append(1.0 if score <= self._margin else 0.0)

        # 마진 재계산
        self._margin = self._compute_margin()

    def get_margin(self) -> float:
        """현재 CP 마진 반환 (클램프 적용)"""
        return self._margin

    def get_statistics(self) -> Dict:
        """CP 통계 반환"""
        scores = np.array(self._scores) if self._scores else np.array([0.0])
        hits = np.array(self._coverage_hits) if self._coverage_hits else np.array([])

        return {
            "cp_margin": self._margin,
            "cp_num_scores": len(self._scores),
            "cp_mean_score": float(np.mean(scores)),
            "cp_std_score": float(np.std(scores)),
            "cp_min_score": float(np.min(scores)),
            "cp_max_score": float(np.max(scores)),
            "cp_empirical_coverage": float(np.mean(hits)) if len(hits) > 0 else 0.0,
            "cp_step_count": self._step_count,
        }

    def reset(self) -> None:
        """모든 상태 초기화"""
        self._scores.clear()
        self._coverage_hits.clear()
        self._margin = self.config.default_margin
        self._step_count = 0

    def _compute_score(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """비순응 점수 계산"""
        if self.config.score_type == "position_norm":
            return float(np.linalg.norm(actual[:2] - predicted[:2]))
        elif self.config.score_type == "full_state_norm":
            return float(np.linalg.norm(actual - predicted))
        elif self.config.score_type == "per_dim_max":
            return float(np.max(np.abs(actual - predicted)))
        else:
            return float(np.linalg.norm(actual[:2] - predicted[:2]))

    def _compute_margin(self) -> float:
        """CP 마진 계산"""
        n = len(self._scores)
        if n < self.config.min_samples:
            return np.clip(
                self.config.default_margin,
                self.config.margin_min,
                self.config.margin_max,
            )

        scores = np.array(self._scores)

        if self.config.gamma < 1.0:
            # Adaptive CP: 지수 가중 quantile
            margin = self._weighted_quantile(scores)
        else:
            # Standard CP: quantile((n+1)(1-α)/n)
            level = min(1.0, (n + 1) * (1.0 - self.config.alpha) / n)
            margin = float(np.quantile(scores, level))

        return np.clip(margin, self.config.margin_min, self.config.margin_max)

    def _weighted_quantile(self, scores: np.ndarray) -> float:
        """지수 가중 quantile (ACP)"""
        n = len(scores)
        # 최근 데이터에 높은 가중치: w_i = gamma^(n-1-i)
        weights = np.array(
            [self.config.gamma ** (n - 1 - i) for i in range(n)]
        )
        weights /= weights.sum()

        # 가중 quantile: 정렬 후 누적 가중치로 분위 계산
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative = np.cumsum(sorted_weights)

        target = 1.0 - self.config.alpha
        idx = np.searchsorted(cumulative, target)
        idx = min(idx, n - 1)

        return float(sorted_scores[idx])
