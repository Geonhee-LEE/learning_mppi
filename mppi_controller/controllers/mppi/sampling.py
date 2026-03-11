"""
MPPI 노이즈 샘플러

제어 노이즈 샘플링 전략 정의.
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class NoiseSampler(ABC):
    """노이즈 샘플러 추상 베이스 클래스"""

    @abstractmethod
    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        제어 노이즈 샘플링

        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 개수
            control_min: (nu,) 제어 하한
            control_max: (nu,) 제어 상한

        Returns:
            noise: (K, N, nu) 노이즈 샘플
        """
        pass


class GaussianSampler(NoiseSampler):
    """
    가우시안 노이즈 샘플러

    ε ~ N(0, Σ), Σ = diag(σ²)

    Args:
        sigma: (nu,) 표준편차
        seed: 랜덤 시드 (재현성)
    """

    def __init__(self, sigma: np.ndarray, seed: Optional[int] = None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        가우시안 노이즈 샘플링

        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 개수
            control_min: (nu,) 제어 하한
            control_max: (nu,) 제어 상한

        Returns:
            noise: (K, N, nu) 노이즈 샘플
        """
        N, nu = U.shape

        # 가우시안 노이즈 생성 (K, N, nu)
        noise = self.rng.normal(0.0, self.sigma, (K, N, nu))

        # 제어 제약이 있으면 클리핑
        if control_min is not None and control_max is not None:
            # 샘플 제어 = 명목 + 노이즈
            sampled_controls = U + noise  # 브로드캐스트 (K, N, nu)

            # 클리핑
            sampled_controls = np.clip(sampled_controls, control_min, control_max)

            # 노이즈 = 클리핑된 제어 - 명목 제어
            noise = sampled_controls - U

        return noise

    def __repr__(self) -> str:
        return f"GaussianSampler(sigma={self.sigma})"


class ColoredNoiseSampler(NoiseSampler):
    """
    Colored Noise 샘플러 (Ornstein-Uhlenbeck 프로세스)

    시간 상관 노이즈로 부드러운 제어 생성.

    dε/dt = -θ ε + σ dW

    Args:
        sigma: (nu,) 표준편차
        theta: (nu,) 복원율 (reversion rate)
        dt: 타임스텝 간격
        seed: 랜덤 시드
    """

    def __init__(
        self,
        sigma: np.ndarray,
        theta: np.ndarray,
        dt: float,
        seed: Optional[int] = None,
    ):
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        OU 프로세스 노이즈 샘플링

        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 개수
            control_min: (nu,) 제어 하한
            control_max: (nu,) 제어 상한

        Returns:
            noise: (K, N, nu) 노이즈 샘플
        """
        N, nu = U.shape
        noise = np.zeros((K, N, nu))

        # 각 샘플에 대해 OU 프로세스 시뮬레이션
        for k in range(K):
            epsilon = np.zeros(nu)  # 초기 노이즈
            for t in range(N):
                # OU 프로세스 업데이트 (Euler-Maruyama)
                dW = self.rng.normal(0.0, np.sqrt(self.dt), nu)
                epsilon = (
                    epsilon
                    - self.theta * epsilon * self.dt
                    + self.sigma * dW
                )
                noise[k, t, :] = epsilon

        # 제어 제약이 있으면 클리핑
        if control_min is not None and control_max is not None:
            sampled_controls = U + noise
            sampled_controls = np.clip(sampled_controls, control_min, control_max)
            noise = sampled_controls - U

        return noise

    def __repr__(self) -> str:
        return (
            f"ColoredNoiseSampler(sigma={self.sigma}, theta={self.theta}, dt={self.dt})"
        )


class RectifiedGaussianSampler(NoiseSampler):
    """
    정류 가우시안 샘플러 (pytorch_mppi 스타일)

    제약을 위반하는 샘플을 재샘플링하여 제약 준수 보장.

    Args:
        sigma: (nu,) 표준편차
        max_retries: 최대 재샘플링 횟수
        seed: 랜덤 시드
    """

    def __init__(
        self, sigma: np.ndarray, max_retries: int = 10, seed: Optional[int] = None
    ):
        self.sigma = sigma
        self.max_retries = max_retries
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        정류 가우시안 노이즈 샘플링

        제약 위반 샘플을 재샘플링하여 제약 준수.

        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 개수
            control_min: (nu,) 제어 하한
            control_max: (nu,) 제어 상한

        Returns:
            noise: (K, N, nu) 노이즈 샘플
        """
        N, nu = U.shape

        # 초기 노이즈 생성
        noise = self.rng.normal(0.0, self.sigma, (K, N, nu))

        # 제약이 없으면 바로 반환
        if control_min is None or control_max is None:
            return noise

        # 제약 위반 샘플 재샘플링
        for retry in range(self.max_retries):
            # 샘플 제어 = 명목 + 노이즈
            sampled_controls = U + noise  # (K, N, nu)

            # 제약 위반 체크 (K, N, nu) → (K,)
            violations = np.any(
                (sampled_controls < control_min) | (sampled_controls > control_max),
                axis=(1, 2),
            )

            # 모든 샘플이 제약을 만족하면 종료
            if not np.any(violations):
                break

            # 위반한 샘플만 재샘플링
            num_violations = np.sum(violations)
            noise[violations] = self.rng.normal(
                0.0, self.sigma, (num_violations, N, nu)
            )

        # 최대 재시도 후에도 위반이 있으면 클리핑 (fallback)
        sampled_controls = U + noise
        sampled_controls = np.clip(sampled_controls, control_min, control_max)
        noise = sampled_controls - U

        return noise

    def __repr__(self) -> str:
        return (
            f"RectifiedGaussianSampler(sigma={self.sigma}, "
            f"max_retries={self.max_retries})"
        )


class UncertaintyAwareSampler(NoiseSampler):
    """
    불확실성 인식 노이즈 샘플러

    모델 예측 불확실성에 비례하여 시간별 노이즈 스케일을 적응적으로 조절.
    - 불확실한 타임스텝 → σ 증가 (넓은 탐색)
    - 확실한 타임스텝 → σ 감소 (좁은 활용)

    sigma_t = clip(1 + exploration_factor * mean(std_t) / mean(base_sigma),
                   min_ratio, max_ratio) * base_sigma

    Args:
        base_sigma: (nu,) 기본 표준편차
        exploration_factor: 불확실성→노이즈 변환 계수
        min_sigma_ratio: 최소 sigma 비율 (base 대비)
        max_sigma_ratio: 최대 sigma 비율 (base 대비)
        seed: 랜덤 시드
    """

    def __init__(
        self,
        base_sigma: np.ndarray,
        exploration_factor: float = 1.0,
        min_sigma_ratio: float = 0.3,
        max_sigma_ratio: float = 3.0,
        seed: Optional[int] = None,
    ):
        self.base_sigma = np.asarray(base_sigma, dtype=float)
        self.exploration_factor = exploration_factor
        self.min_sigma_ratio = min_sigma_ratio
        self.max_sigma_ratio = max_sigma_ratio
        self.rng = np.random.default_rng(seed)

        # 시간별 불확실성 프로파일: (N, nx) 또는 None
        self._uncertainty_profile: Optional[np.ndarray] = None
        # 계산된 시간별 sigma 비율: (N,) 또는 None
        self._sigma_ratios: Optional[np.ndarray] = None

    def update_uncertainty_profile(self, uncertainty_per_step: np.ndarray):
        """
        시간별 불확실성 프로파일 업데이트

        Args:
            uncertainty_per_step: (N, nx) 각 타임스텝의 상태 불확실성 (std)
        """
        self._uncertainty_profile = np.asarray(uncertainty_per_step, dtype=float)

        # 시간별 평균 불확실성 → sigma 비율 계산
        mean_unc_per_step = np.mean(self._uncertainty_profile, axis=-1)  # (N,)
        mean_base = np.mean(self.base_sigma)

        if mean_base > 0:
            raw_ratios = 1.0 + self.exploration_factor * mean_unc_per_step / mean_base
        else:
            raw_ratios = np.ones(len(mean_unc_per_step))

        self._sigma_ratios = np.clip(
            raw_ratios, self.min_sigma_ratio, self.max_sigma_ratio
        )

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        불확실성 기반 적응 노이즈 샘플링

        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 개수
            control_min: (nu,) 제어 하한
            control_max: (nu,) 제어 상한

        Returns:
            noise: (K, N, nu) 노이즈 샘플
        """
        N, nu = U.shape

        if self._sigma_ratios is not None and len(self._sigma_ratios) == N:
            # 시간별 적응 sigma: (N, nu)
            sigma_profile = self._sigma_ratios[:, None] * self.base_sigma[None, :]
        else:
            # 프로파일 없으면 base_sigma 사용 (GaussianSampler와 동일)
            sigma_profile = np.broadcast_to(self.base_sigma, (N, nu))

        # 시간별 다른 sigma로 노이즈 생성
        noise = self.rng.normal(0.0, 1.0, (K, N, nu)) * sigma_profile[None, :, :]

        # 제어 제약 클리핑
        if control_min is not None and control_max is not None:
            sampled_controls = U + noise
            sampled_controls = np.clip(sampled_controls, control_min, control_max)
            noise = sampled_controls - U

        return noise

    def get_sigma_statistics(self) -> dict:
        """현재 sigma 프로파일 통계 반환"""
        if self._sigma_ratios is None:
            return {
                "has_profile": False,
                "mean_ratio": 1.0,
                "min_ratio": 1.0,
                "max_ratio": 1.0,
            }
        return {
            "has_profile": True,
            "mean_ratio": float(np.mean(self._sigma_ratios)),
            "min_ratio": float(np.min(self._sigma_ratios)),
            "max_ratio": float(np.max(self._sigma_ratios)),
            "ratios": self._sigma_ratios.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"UncertaintyAwareSampler(base_sigma={self.base_sigma}, "
            f"exploration_factor={self.exploration_factor})"
        )
