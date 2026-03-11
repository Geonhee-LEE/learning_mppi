"""
Uncertainty-Aware MPPI 컨트롤러

모델 예측 불확실성에 비례하여 샘플링 노이즈를 적응적으로 조절.
- 불확실한 영역 → 넓은 탐색 (exploration)
- 확실한 영역 → 좁은 활용 (exploitation)

3가지 전략 지원:
- previous_trajectory: 직전 최적 궤적의 불확실성 재사용 (비용 0)
- current_state: 현재 상태 불확실성으로 전역 스케일
- two_pass: 1차 rollout → 불확실성 추정 → 2차 적응 rollout
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable, List

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import UncertaintyMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import (
    NoiseSampler,
    UncertaintyAwareSampler,
)


class UncertaintyMPPIController(MPPIController):
    """
    Uncertainty-Aware MPPI 컨트롤러

    모델 불확실성에 비례하여 MPPI 샘플링 노이즈를 적응 조절.
    불확실한 영역에서는 넓은 탐색, 확실한 영역에서는 좁은 활용.

    Args:
        model: RobotModel 인스턴스
        params: UncertaintyMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 UncertaintyAwareSampler 자동 생성)
        uncertainty_fn: (states, controls) → std (batch, nx)
            None이면 model.predict_with_uncertainty 자동 감지
    """

    def __init__(
        self,
        model: RobotModel,
        params: UncertaintyMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        uncertainty_fn: Optional[Callable] = None,
    ):
        # UncertaintyAwareSampler 자동 생성 (명시적 sampler가 없으면)
        if noise_sampler is None:
            noise_sampler = UncertaintyAwareSampler(
                base_sigma=params.sigma,
                exploration_factor=params.exploration_factor,
                min_sigma_ratio=params.min_sigma_ratio,
                max_sigma_ratio=params.max_sigma_ratio,
            )

        super().__init__(model, params, cost_function, noise_sampler)

        self.uncertainty_params = params

        # 불확실성 함수 설정 (자동 감지)
        self.uncertainty_fn = self._resolve_uncertainty_fn(uncertainty_fn, model)

        # 이전 최적 궤적/제어 저장 (previous_trajectory 전략용)
        self._prev_best_trajectory: Optional[np.ndarray] = None
        self._prev_best_controls: Optional[np.ndarray] = None

        # 통계 히스토리
        self._uncertainty_history: List[dict] = []

    @staticmethod
    def _resolve_uncertainty_fn(
        uncertainty_fn: Optional[Callable], model: RobotModel
    ) -> Optional[Callable]:
        """불확실성 함수 결정 (명시적 > model 자동 감지 > None)"""
        if uncertainty_fn is not None:
            return uncertainty_fn

        # model.predict_with_uncertainty 자동 감지
        if hasattr(model, "predict_with_uncertainty"):
            def _model_uncertainty(states, controls):
                _, std = model.predict_with_uncertainty(states, controls)
                return std
            return _model_uncertainty

        return None

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        불확실성 인식 MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + uncertainty_stats, sigma_stats
        """
        strategy = self.uncertainty_params.uncertainty_strategy

        if strategy == "two_pass":
            return self._compute_control_two_pass(state, reference_trajectory)

        # previous_trajectory / current_state 전략
        unc_profile = self._estimate_uncertainty_profile(state)

        # sampler 업데이트
        if unc_profile is not None and isinstance(
            self.noise_sampler, UncertaintyAwareSampler
        ):
            self.noise_sampler.update_uncertainty_profile(unc_profile)

        # 표준 MPPI 파이프라인
        control, info = super().compute_control(state, reference_trajectory)

        # 최적 궤적 저장 (다음 호출에서 previous_trajectory 전략용)
        if "best_trajectory" in info:
            self._prev_best_trajectory = info["best_trajectory"].copy()
        # 최적 제어 시퀀스 저장 (U는 이미 shift됨, shift 전 값 필요)
        self._prev_best_controls = self.U.copy()

        # 통계 추가
        self._append_stats(info, unc_profile)

        return control, info

    def _compute_control_two_pass(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Two-pass 전략: 1차 표준 rollout → 불확실성 추정 → 2차 적응 rollout

        비용이 2배이지만 정확도가 가장 높음.
        """
        # 1차: 표준 노이즈로 rollout (불확실성 없이)
        if isinstance(self.noise_sampler, UncertaintyAwareSampler):
            # 임시로 프로파일 제거하여 base_sigma 사용
            saved_ratios = self.noise_sampler._sigma_ratios
            self.noise_sampler._sigma_ratios = None

        control_pass1, info_pass1 = super().compute_control(
            state, reference_trajectory
        )

        if isinstance(self.noise_sampler, UncertaintyAwareSampler):
            self.noise_sampler._sigma_ratios = saved_ratios

        # 1차 best trajectory에서 불확실성 추정
        best_traj = info_pass1.get("best_trajectory")
        if best_traj is not None and self.uncertainty_fn is not None:
            unc_profile = self._estimate_uncertainty_along_trajectory(
                best_traj, self.U
            )
            if isinstance(self.noise_sampler, UncertaintyAwareSampler):
                self.noise_sampler.update_uncertainty_profile(unc_profile)

        # U를 1차 이전으로 되돌리기 (shift 취소)
        # super().compute_control()이 U를 이미 shift했으므로 복원
        self.U = np.roll(self.U, 1, axis=0)
        self.U[0, :] = control_pass1

        # 2차: 적응된 노이즈로 rollout
        control, info = super().compute_control(state, reference_trajectory)

        # 최적 궤적 저장
        if "best_trajectory" in info:
            self._prev_best_trajectory = info["best_trajectory"].copy()
        self._prev_best_controls = self.U.copy()

        # 통계
        unc_profile_final = (
            unc_profile if (best_traj is not None and self.uncertainty_fn is not None)
            else None
        )
        self._append_stats(info, unc_profile_final)
        info["two_pass"] = True

        return control, info

    def _estimate_uncertainty_profile(
        self, state: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        불확실성 프로파일 추정 (전략에 따라)

        Args:
            state: (nx,) 현재 상태

        Returns:
            uncertainty: (N, nx) 또는 None
        """
        if self.uncertainty_fn is None:
            return None

        strategy = self.uncertainty_params.uncertainty_strategy
        N = self.uncertainty_params.N

        if strategy == "previous_trajectory":
            # 이전 best_trajectory 재사용
            if self._prev_best_trajectory is not None:
                return self._estimate_uncertainty_along_trajectory(
                    self._prev_best_trajectory,
                    self._prev_best_controls if self._prev_best_controls is not None
                    else self.U,
                )
            # 첫 호출: current_state로 fallback
            return self._estimate_uncertainty_current_state(state, N)

        elif strategy == "current_state":
            return self._estimate_uncertainty_current_state(state, N)

        return None

    def _estimate_uncertainty_current_state(
        self, state: np.ndarray, N: int
    ) -> np.ndarray:
        """
        현재 상태에서의 불확실성을 전역으로 사용

        Args:
            state: (nx,) 현재 상태
            N: 호라이즌 길이

        Returns:
            uncertainty: (N, nx) 동일한 불확실성 N번 반복
        """
        # 단일 상태에 대한 불확실성 추정
        states_batch = state[None, :]  # (1, nx)
        controls_batch = self.U[0:1, :]  # (1, nu)

        std = self.uncertainty_fn(states_batch, controls_batch)  # (1, nx)
        if std.ndim == 1:
            std = std[None, :]

        # N 타임스텝 동안 동일한 불확실성
        return np.broadcast_to(std, (N, std.shape[-1])).copy()

    def _estimate_uncertainty_along_trajectory(
        self, trajectory: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """
        궤적을 따라 각 타임스텝의 불확실성 추정

        Args:
            trajectory: (N+1, nx) 궤적
            controls: (N, nu) 제어 시퀀스

        Returns:
            uncertainty: (N, nx) 시간별 불확실성
        """
        N = min(len(trajectory) - 1, len(controls))
        nx = trajectory.shape[-1]
        uncertainty = np.zeros((N, nx))

        for t in range(N):
            state_t = trajectory[t:t+1, :]  # (1, nx)
            control_t = controls[t:t+1, :]  # (1, nu)
            std = self.uncertainty_fn(state_t, control_t)  # (1, nx) or (nx,)
            if std.ndim == 1:
                uncertainty[t, :] = std
            else:
                uncertainty[t, :] = std[0]

        return uncertainty

    def _append_stats(self, info: Dict, unc_profile: Optional[np.ndarray]):
        """info dict에 불확실성/sigma 통계 추가"""
        # uncertainty 통계
        if unc_profile is not None:
            unc_stats = {
                "mean_uncertainty": float(np.mean(unc_profile)),
                "max_uncertainty": float(np.max(unc_profile)),
                "min_uncertainty": float(np.min(unc_profile)),
                "profile_shape": unc_profile.shape,
            }
        else:
            unc_stats = {
                "mean_uncertainty": 0.0,
                "max_uncertainty": 0.0,
                "min_uncertainty": 0.0,
                "profile_shape": None,
            }
        info["uncertainty_stats"] = unc_stats

        # sigma 통계
        if isinstance(self.noise_sampler, UncertaintyAwareSampler):
            info["sigma_stats"] = self.noise_sampler.get_sigma_statistics()
        else:
            info["sigma_stats"] = {"has_profile": False, "mean_ratio": 1.0}

        # 히스토리에 추가
        self._uncertainty_history.append(unc_stats)

    def get_uncertainty_statistics(self) -> Dict:
        """누적된 불확실성 통계 반환"""
        if not self._uncertainty_history:
            return {"num_steps": 0}

        means = [h["mean_uncertainty"] for h in self._uncertainty_history]
        maxes = [h["max_uncertainty"] for h in self._uncertainty_history]

        return {
            "num_steps": len(self._uncertainty_history),
            "overall_mean_uncertainty": float(np.mean(means)),
            "overall_max_uncertainty": float(np.max(maxes)),
            "history": self._uncertainty_history,
        }

    def reset(self):
        """상태 초기화"""
        super().reset()
        self._prev_best_trajectory = None
        self._prev_best_controls = None
        self._uncertainty_history = []
        if isinstance(self.noise_sampler, UncertaintyAwareSampler):
            self.noise_sampler._uncertainty_profile = None
            self.noise_sampler._sigma_ratios = None

    def __repr__(self) -> str:
        return (
            f"UncertaintyMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"strategy={self.uncertainty_params.uncertainty_strategy}, "
            f"exploration={self.uncertainty_params.exploration_factor})"
        )
