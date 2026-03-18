"""
ASR-MPPI (Adaptive Spectral Risk MPPI) Controller

Spectral Risk Measure (SRM) 기반 가중치 계산으로 CVaR의 경질 절단을 연속적 왜곡 함수로 일반화.

핵심 수식:
    SRM_φ(S) = ∫₀¹ VaR_q(S) · φ'(q) dq
    가중치: w_k ∝ φ'(q_k) · exp(-S_{(k)} / λ)

왜곡 함수 φ(q):
    - sigmoid: σ(β(q-α)), 부드러운 S-곡선
    - power: q^γ, 꼬리 제어
    - dual_power: 1-(1-q)^γ, 반대 꼬리
    - cvar: 계단 함수 (기존 CVaR 호환)

Reference: Lin et al., ICML 2025 (arXiv:2501.02087) 영감
"""

import numpy as np
from collections import deque
from typing import Dict
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import ASRMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController


class ASRMPPIController(MPPIController):
    """
    ASR-MPPI (Adaptive Spectral Risk MPPI) Controller

    Spectral Risk Measure를 사용하여 비용 분위수를 비균일 가중하는 MPPI 변형.

    동작 원리:
        1. 비용을 오름차순 정렬 → S_{(1)} ≤ ... ≤ S_{(K)}
        2. 분위수 q_k = k/K 계산
        3. 왜곡 함수 도함수 φ'(q_k)로 분위수별 밀도 계산
        4. spectral_weights = φ'(q_k) · exp(-S_{(k)} / λ)
        5. 정규화: w_k = spectral_weights / Σ spectral_weights

    CVaR은 SRM의 특수 경우:
        - φ(q) = 0 (q < 1-α), (q-(1-α))/α (q ≥ 1-α) → 계단 함수
        - distortion_type="cvar"로 기존 Risk-Aware MPPI와 동일 동작

    적응적 위험 (use_adaptive_risk=True):
        - ESS/K가 min_ess_ratio 이하 → β 감소 (가중치 분산)
        - ESS/K가 높으면 → β 증가 (가중치 집중)

    장점:
        - 부드러운 가중치 전환 (CVaR의 불연속 제거)
        - 다양한 왜곡 함수로 표현력 향상
        - ESS 기반 자동 적응으로 안정성 보장

    Args:
        model: RobotModel 인스턴스
        params: ASRMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: ASRMPPIParams, **kwargs):
        super().__init__(model, params, **kwargs)

        self.asr_params = params
        self._current_alpha = params.distortion_alpha
        self._current_beta = params.distortion_beta

        # 적응 히스토리
        self._cost_history = deque(maxlen=params.adaptation_window)
        self._risk_history = []

        # 마지막 스펙트럴 통계
        self._last_spectral_stats = {}

    def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Spectral Risk Measure 기반 가중치 계산

        Args:
            costs: (K,) 샘플 비용
            lambda_: 온도 파라미터

        Returns:
            weights: (K,) 정규화된 가중치
        """
        K = len(costs)

        # 1. 비용 오름차순 정렬
        sorted_indices = np.argsort(costs)
        sorted_costs = costs[sorted_indices]

        # 2. 분위수 계산 (0, 1/K, ..., (K-1)/K)
        quantiles = np.arange(K) / K

        # 3. 왜곡 함수 도함수 (density)
        distortion_weights = self._eval_distortion_derivative(quantiles)

        # 4. Softmax (baseline 적용)
        baseline = sorted_costs[0]
        exp_costs = np.exp(-(sorted_costs - baseline) / lambda_)

        # 5. Spectral weights = φ'(q) · softmax
        spectral_weights = distortion_weights * exp_costs

        # 6. 정규화
        total = np.sum(spectral_weights)
        if total > 0:
            weights = np.zeros(K)
            weights[sorted_indices] = spectral_weights / total
        else:
            weights = np.ones(K) / K

        # 7. ESS
        ess = 1.0 / np.sum(weights ** 2)
        ess_ratio = ess / K

        # 8. Spectral Risk Value (SRM 추정)
        spectral_risk_value = float(np.sum(
            (distortion_weights / np.sum(distortion_weights) if np.sum(distortion_weights) > 0 else np.ones(K) / K)
            * sorted_costs
        ))

        # 9. 통계 저장
        self._last_spectral_stats = {
            "spectral_risk_value": spectral_risk_value,
            "distortion_type": self.asr_params.distortion_type,
            "current_alpha": self._current_alpha,
            "current_beta": self._current_beta,
            "ess": ess,
            "ess_ratio": ess_ratio,
            "mean_spectral_weight": float(np.mean(distortion_weights)),
            "max_weight": float(np.max(weights)),
            "num_zero_weights": int(np.sum(weights == 0)),
        }
        self._risk_history.append(self._last_spectral_stats)

        # 10. 적응
        if self.asr_params.use_adaptive_risk:
            self._adapt_parameters(costs, weights, ess_ratio)

        return weights

    def _eval_distortion(self, q: np.ndarray) -> np.ndarray:
        """
        왜곡 함수 φ(q) 계산

        Args:
            q: (N,) 분위수 [0, 1]

        Returns:
            phi: (N,) φ(q) 값
        """
        dtype = self.asr_params.distortion_type

        if dtype == "sigmoid":
            alpha = self._current_alpha
            beta = self._current_beta
            raw = 1.0 / (1.0 + np.exp(-beta * (q - alpha)))
            # 정규화: φ(0)=0, φ(1)=1
            phi_0 = 1.0 / (1.0 + np.exp(-beta * (0.0 - alpha)))
            phi_1 = 1.0 / (1.0 + np.exp(-beta * (1.0 - alpha)))
            denom = phi_1 - phi_0
            if denom > 1e-10:
                return (raw - phi_0) / denom
            return q  # fallback to identity

        elif dtype == "power":
            gamma = self.asr_params.distortion_gamma
            return np.power(np.maximum(q, 0.0), gamma)

        elif dtype == "dual_power":
            gamma = self.asr_params.distortion_gamma
            return 1.0 - np.power(np.maximum(1.0 - q, 0.0), gamma)

        elif dtype == "cvar":
            alpha = self._current_alpha
            return np.where(q < 1.0 - alpha, 0.0, (q - (1.0 - alpha)) / alpha)

        else:
            return q

    def _eval_distortion_derivative(self, q: np.ndarray) -> np.ndarray:
        """
        왜곡 함수 도함수 φ'(q) 계산 (분위수별 밀도)

        Args:
            q: (N,) 분위수 [0, 1]

        Returns:
            phi_prime: (N,) φ'(q) 값
        """
        dtype = self.asr_params.distortion_type

        if dtype == "sigmoid":
            alpha = self._current_alpha
            beta = self._current_beta
            sig = 1.0 / (1.0 + np.exp(-beta * (q - alpha)))
            raw_deriv = beta * sig * (1.0 - sig)
            # 정규화 상수
            phi_0 = 1.0 / (1.0 + np.exp(-beta * (0.0 - alpha)))
            phi_1 = 1.0 / (1.0 + np.exp(-beta * (1.0 - alpha)))
            denom = phi_1 - phi_0
            if denom > 1e-10:
                return raw_deriv / denom
            return np.ones_like(q)

        elif dtype == "power":
            gamma = self.asr_params.distortion_gamma
            # φ'(q) = γ·q^(γ-1), q=0에서 γ<1이면 발산 → 클리핑
            safe_q = np.maximum(q, 1e-10)
            return gamma * np.power(safe_q, gamma - 1.0)

        elif dtype == "dual_power":
            gamma = self.asr_params.distortion_gamma
            # φ'(q) = γ·(1-q)^(γ-1)
            safe_one_minus_q = np.maximum(1.0 - q, 1e-10)
            return gamma * np.power(safe_one_minus_q, gamma - 1.0)

        elif dtype == "cvar":
            alpha = self._current_alpha
            return np.where(q < 1.0 - alpha, 0.0, 1.0 / alpha)

        else:
            return np.ones_like(q)

    def _adapt_parameters(self, costs: np.ndarray, weights: np.ndarray,
                          ess_ratio: float):
        """
        비용 분포 기반 파라미터 적응

        ESS/K가 낮으면 β를 줄여 가중치를 넓히고,
        ESS/K가 높으면 β를 높여 가중치를 집중시킴.

        Args:
            costs: (K,) 비용 배열
            weights: (K,) 현재 가중치
            ess_ratio: ESS/K 비율
        """
        self._cost_history.append({
            "ess_ratio": ess_ratio,
            "cost_std": float(np.std(costs)),
        })

        if len(self._cost_history) < 2:
            return

        rate = self.asr_params.adaptation_rate
        min_ess = self.asr_params.min_ess_ratio

        # ESS 기반 β 조절
        if ess_ratio < min_ess:
            # ESS 너무 낮음 → β 감소 (더 균일한 가중)
            beta_target = self._current_beta * 0.8
        elif ess_ratio > 0.5:
            # ESS 충분히 높음 → β 약간 증가 (더 집중)
            beta_target = self._current_beta * 1.05
        else:
            beta_target = self._current_beta

        # β 범위 제한
        beta_target = np.clip(beta_target, 0.5, 50.0)

        # EMA 업데이트
        self._current_beta = (1.0 - rate) * self._current_beta + rate * beta_target

    def get_risk_statistics(self) -> Dict:
        """
        Spectral Risk 통계 반환

        Returns:
            dict: spectral_risk_value, distortion_type, current_alpha,
                  current_beta, mean_ess_ratio, 등
        """
        if not self._risk_history:
            return {
                "spectral_risk_value": 0.0,
                "distortion_type": self.asr_params.distortion_type,
                "current_alpha": self._current_alpha,
                "current_beta": self._current_beta,
                "mean_ess_ratio": 0.0,
                "mean_spectral_weight": 0.0,
                "risk_history": [],
            }

        ess_ratios = [s["ess_ratio"] for s in self._risk_history]
        spectral_vals = [s["spectral_risk_value"] for s in self._risk_history]

        return {
            "spectral_risk_value": float(np.mean(spectral_vals)),
            "distortion_type": self.asr_params.distortion_type,
            "current_alpha": self._current_alpha,
            "current_beta": self._current_beta,
            "mean_ess_ratio": float(np.mean(ess_ratios)),
            "mean_spectral_weight": float(np.mean(
                [s["mean_spectral_weight"] for s in self._risk_history]
            )),
            "risk_history": self._risk_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 + 적응 상태 초기화"""
        super().reset()
        self._current_alpha = self.asr_params.distortion_alpha
        self._current_beta = self.asr_params.distortion_beta
        self._cost_history.clear()
        self._risk_history = []
        self._last_spectral_stats = {}

    def __repr__(self) -> str:
        return (
            f"ASRMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"distortion={self.asr_params.distortion_type}, "
            f"alpha={self._current_alpha:.2f}, "
            f"beta={self._current_beta:.2f}, "
            f"params={self.params})"
        )
