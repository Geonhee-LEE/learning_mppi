"""
Flow Matching 기반 MPPI 노이즈 샘플러

학습된 Flow 모델로 다중 모달 제어 시퀀스를 생성하여
MPPI의 가우시안 노이즈를 대체.

3가지 모드:
- replace_mean: flow 출력을 평균으로 사용, 탐색 노이즈 추가
- replace_distribution: flow로 K개 샘플 직접 생성
- blend: flow 샘플과 가우시안 샘플 혼합
"""

import numpy as np
from typing import Optional
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class FlowMatchingSampler(NoiseSampler):
    """
    Flow Matching 기반 노이즈 샘플러

    NoiseSampler ABC 준수: sample(U, K, ...) → (K, N, nu)

    Args:
        sigma: (nu,) 가우시안 fallback 표준편차
        mode: "replace_mean" | "replace_distribution" | "blend"
        blend_ratio: blend 모드의 flow 비율
        exploration_sigma: replace_mean 모드의 탐색 노이즈 스케일
        num_ode_steps: ODE 적분 스텝 수
        solver: "euler" | "midpoint"
        seed: 랜덤 시드
    """

    def __init__(
        self,
        sigma: np.ndarray,
        mode: str = "replace_mean",
        blend_ratio: float = 0.5,
        exploration_sigma: float = 0.5,
        num_ode_steps: int = 5,
        solver: str = "euler",
        seed: Optional[int] = None,
    ):
        self.sigma = np.asarray(sigma, dtype=float)
        self.mode = mode
        self.blend_ratio = blend_ratio
        self.exploration_sigma = exploration_sigma
        self.num_ode_steps = num_ode_steps
        self.solver = solver
        self.rng = np.random.default_rng(seed)

        self._flow_model = None
        self._context: Optional[np.ndarray] = None

    def set_flow_model(self, model):
        """학습된 Flow 모델 주입"""
        self._flow_model = model

    def set_context(self, state: np.ndarray):
        """현재 상태 컨텍스트 설정 (compute_control 전에 호출)"""
        self._context = np.asarray(state, dtype=float)

    @property
    def is_flow_ready(self) -> bool:
        """Flow 모델이 로드되어 사용 가능한지 여부"""
        return self._flow_model is not None and self._context is not None

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        제어 노이즈 샘플링

        Flow 모델이 없으면 가우시안 fallback.

        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 개수
            control_min: (nu,) 제어 하한
            control_max: (nu,) 제어 상한

        Returns:
            noise: (K, N, nu) 노이즈 샘플
        """
        if not self.is_flow_ready:
            return self._gaussian_fallback(U, K, control_min, control_max)

        N, nu = U.shape

        if self.mode == "replace_mean":
            noise = self._sample_replace_mean(U, K, N, nu)
        elif self.mode == "replace_distribution":
            noise = self._sample_replace_distribution(U, K, N, nu)
        elif self.mode == "blend":
            noise = self._sample_blend(U, K, N, nu)
        else:
            noise = self._gaussian_fallback(U, K, control_min, control_max)
            return noise

        # 제어 제약 클리핑
        if control_min is not None and control_max is not None:
            sampled_controls = U + noise
            sampled_controls = np.clip(sampled_controls, control_min, control_max)
            noise = sampled_controls - U

        return noise

    def _sample_replace_mean(
        self, U: np.ndarray, K: int, N: int, nu: int
    ) -> np.ndarray:
        """Flow 출력을 평균으로 사용, 가우시안 탐색 노이즈 추가"""
        import torch

        context = torch.tensor(self._context, dtype=torch.float32)
        # Flow가 1개 평균 시퀀스 생성
        flow_output = self._flow_model.generate(
            context, num_samples=1, num_steps=self.num_ode_steps, solver=self.solver
        )
        # (1, N*nu) → (N, nu)
        mu = flow_output[0].cpu().numpy().reshape(N, nu)

        # noise = (mu - U) + gaussian exploration
        base_offset = mu - U  # (N, nu)
        exploration = self.rng.normal(
            0.0, self.exploration_sigma * self.sigma, (K, N, nu)
        )
        noise = base_offset[None, :, :] + exploration

        return noise

    def _sample_replace_distribution(
        self, U: np.ndarray, K: int, N: int, nu: int
    ) -> np.ndarray:
        """Flow로 K개 샘플 직접 생성"""
        import torch

        context = torch.tensor(self._context, dtype=torch.float32)
        flow_output = self._flow_model.generate(
            context, num_samples=K, num_steps=self.num_ode_steps, solver=self.solver
        )
        # (K, N*nu) → (K, N, nu)
        samples = flow_output.cpu().numpy().reshape(K, N, nu)
        noise = samples - U[None, :, :]

        return noise

    def _sample_blend(
        self, U: np.ndarray, K: int, N: int, nu: int
    ) -> np.ndarray:
        """Flow 샘플과 가우시안 샘플 혼합"""
        import torch

        K_flow = max(1, int(K * self.blend_ratio))
        K_gauss = K - K_flow

        # Flow 샘플
        context = torch.tensor(self._context, dtype=torch.float32)
        flow_output = self._flow_model.generate(
            context, num_samples=K_flow, num_steps=self.num_ode_steps, solver=self.solver
        )
        flow_samples = flow_output.cpu().numpy().reshape(K_flow, N, nu)
        flow_noise = flow_samples - U[None, :, :]

        # 가우시안 샘플
        gauss_noise = self.rng.normal(0.0, self.sigma, (K_gauss, N, nu))

        # 합침
        noise = np.concatenate([flow_noise, gauss_noise], axis=0)

        return noise

    def _gaussian_fallback(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Flow 모델 없을 때 표준 가우시안 노이즈"""
        N, nu = U.shape
        noise = self.rng.normal(0.0, self.sigma, (K, N, nu))

        if control_min is not None and control_max is not None:
            sampled_controls = U + noise
            sampled_controls = np.clip(sampled_controls, control_min, control_max)
            noise = sampled_controls - U

        return noise

    def __repr__(self) -> str:
        status = "ready" if self.is_flow_ready else "fallback"
        return f"FlowMatchingSampler(mode={self.mode}, status={status})"
