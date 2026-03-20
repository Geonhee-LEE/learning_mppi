"""
LP-MPPI (Low-Pass MPPI) Controller — 23번째 MPPI 변형

Butterworth 저역통과 필터를 MPPI 노이즈에 적용하여
주파수 영역에서 직접적인 smoothness 제어.

핵심 수식:
    ε ~ N(0, Σ)
    ε_LP = ButterworthLPF(ε, f_c, o_LPF)
    U_i = U + ε_LP

    |H(f)|² = 1 / (1 + (f/f_c)^(2n))
      f_c = cutoff frequency (Hz)
      n = filter order (o_LPF)

기존 변형 대비 핵심 차이:
    - Smooth-MPPI: 제어 공간에서 ΔU + jerk cost → 사후 평활
    - Colored-Noise: OU 프로세스로 시간 상관 → Python 이중 루프
    - LP-MPPI: Butterworth LPF → 주파수 도메인 직접 제어, scipy 벡터화

Reference: Kicki et al., ICRA 2026, arXiv:2503.11717
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import LPMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler, LowPassSampler


class LPMPPIController(MPPIController):
    """
    LP-MPPI 컨트롤러 (23번째 MPPI 변형)

    noise_sampler 미제공 시 LowPassSampler 자동 생성.
    compute_control()은 super() 호출 후 smoothness_stats를 info에 추가.

    Args:
        model: RobotModel 인스턴스
        params: LPMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 LowPassSampler 자동 생성)
    """

    def __init__(
        self,
        model: RobotModel,
        params: LPMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        # noise_sampler 미제공 시 LowPassSampler 자동 생성
        if noise_sampler is None:
            noise_sampler = LowPassSampler(
                sigma=params.sigma,
                cutoff_freq=params.cutoff_freq,
                filter_order=params.filter_order,
                dt=params.dt,
                normalize_variance=params.normalize_variance,
            )

        super().__init__(model, params, cost_function, noise_sampler)

        self.lp_params = params
        self._smoothness_history: List[Dict] = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        LP-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + smoothness_stats
        """
        control, info = super().compute_control(state, reference_trajectory)

        # Smoothness 통계 계산 및 추가
        smoothness = self._compute_smoothness_stats(info)
        info["smoothness_stats"] = smoothness
        info["cutoff_freq"] = self.lp_params.cutoff_freq
        info["filter_order"] = self.lp_params.filter_order

        self._smoothness_history.append(smoothness)

        return control, info

    def _compute_smoothness_stats(self, info: Dict) -> Dict:
        """
        Smoothness 통계 계산

        MSSD: Mean Squared Second Difference (가속도 지표)
        Jerk: 제어의 3차 미분 크기

        Args:
            info: compute_control에서 반환된 info dict

        Returns:
            smoothness_stats: mssd, mean_jerk, max_jerk
        """
        # best_trajectory의 제어 시퀀스 대신, 현재 U 시퀀스 사용
        U = self.U  # (N, nu) — shift 후이지만 smoothness 지표로 충분

        if U.shape[0] < 3:
            return {"mssd": 0.0, "mean_jerk": 0.0, "max_jerk": 0.0}

        # MSSD: mean(diff(U, n=2, axis=0) ** 2)
        second_diff = np.diff(U, n=2, axis=0)  # (N-2, nu)
        mssd = float(np.mean(second_diff ** 2))

        # Jerk: norm of second difference (≈ acceleration change)
        jerk_norms = np.linalg.norm(second_diff, axis=1)  # (N-2,)
        mean_jerk = float(np.mean(jerk_norms))
        max_jerk = float(np.max(jerk_norms))

        return {
            "mssd": mssd,
            "mean_jerk": mean_jerk,
            "max_jerk": max_jerk,
        }

    def get_smoothness_statistics(self) -> Dict:
        """누적 smoothness 통계 반환"""
        if not self._smoothness_history:
            return {
                "num_steps": 0,
                "mean_mssd": 0.0,
                "mean_jerk": 0.0,
            }

        mssds = [s["mssd"] for s in self._smoothness_history]
        jerks = [s["mean_jerk"] for s in self._smoothness_history]

        return {
            "num_steps": len(self._smoothness_history),
            "mean_mssd": float(np.mean(mssds)),
            "std_mssd": float(np.std(mssds)),
            "mean_jerk": float(np.mean(jerks)),
            "std_jerk": float(np.std(jerks)),
        }

    def reset(self):
        """제어 시퀀스 + smoothness 히스토리 초기화"""
        super().reset()
        self._smoothness_history = []

    def __repr__(self) -> str:
        return (
            f"LPMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"cutoff_freq={self.lp_params.cutoff_freq}, "
            f"filter_order={self.lp_params.filter_order})"
        )
