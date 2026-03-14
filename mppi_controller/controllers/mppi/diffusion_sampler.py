"""
Diffusion 역확산 샘플러 — MPPI 노이즈 샘플러 인터페이스

DDIM (Denoising Diffusion Implicit Models) 가속 역확산으로
K개 제어 시퀀스를 병렬 생성.

Flow-MPPI의 FlowMatchingSampler와 동일 인터페이스 (NoiseSampler ABC).

수식 (DDIM):
    x_{t-1} = √ᾱ_{t-1} * pred_x0
              + √(1-ᾱ_{t-1}) * ε_θ(x_t, t, c)
    pred_x0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t

References:
    Song et al. (2021) — DDIM
    Ho et al. (2020) — DDPM
    Chi et al. (2023) — Diffusion Policy
"""

import numpy as np
from typing import Optional, Tuple
from mppi_controller.controllers.mppi.sampling import NoiseSampler


def _make_cosine_schedule(T: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine noise schedule.

    Returns:
        alpha_bar: (T+1,) ᾱ_t = Π_{s=1}^{t} αs
        betas: (T,)
    """
    s = 0.008
    steps = np.arange(T + 1)
    f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    alpha_bar = np.clip(alpha_bar, 1e-5, 1.0 - 1e-5)
    return alpha_bar


def _make_linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """
    Linear noise schedule.

    Returns:
        alpha_bar: (T+1,) ᾱ_t
    """
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = np.concatenate([[1.0], np.cumprod(alphas)])
    return np.clip(alpha_bar, 1e-5, 1.0 - 1e-5)


class DDIMSampler(NoiseSampler):
    """
    DDIM 가속 역확산 샘플러 (NoiseSampler ABC 구현).

    학습된 노이즈 예측 모델 ε_θ(x_t, t, context)로
    K개 제어 시퀀스를 ddim_steps 역확산 스텝으로 생성.

    학습 전: 순수 가우시안 noise 반환 (= Vanilla MPPI 동작).

    Args:
        sigma: (nu,) 기본 가우시안 노이즈 표준편차 (fallback용)
        ddim_steps: 역확산 스텝 수 (1~20, 작을수록 빠름)
        T: 학습 시 총 노이즈 스텝 수 (1000)
        beta_schedule: 노이즈 스케줄 ("cosine" | "linear")
        mode: 샘플링 모드
            - "replace": 가우시안 완전 대체 (K개 새 샘플)
            - "blend": α*diffusion + (1-α)*gaussian
        blend_ratio: blend 모드에서 diffusion 비율 [0, 1]
        exploration_sigma: blend/replace 후 추가 탐색 노이즈 스케일
        guidance_scale: Classifier-free guidance 스케일 (1.0 = 비활성화)
    """

    def __init__(
        self,
        sigma: np.ndarray,
        ddim_steps: int = 5,
        T: int = 1000,
        beta_schedule: str = "cosine",
        mode: str = "replace",
        blend_ratio: float = 0.5,
        exploration_sigma: float = 0.5,
        guidance_scale: float = 1.0,
    ):
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.ddim_steps = ddim_steps
        self.T = T
        self.beta_schedule = beta_schedule
        self.mode = mode
        self.blend_ratio = blend_ratio
        self.exploration_sigma = exploration_sigma
        self.guidance_scale = guidance_scale

        # 노이즈 스케줄 사전 계산
        if beta_schedule == "cosine":
            self.alpha_bar = _make_cosine_schedule(T)
        else:
            self.alpha_bar = _make_linear_schedule(T)

        # DDIM 타임스텝 (T → 0 방향)
        step_size = T // ddim_steps
        self.ddim_timesteps = np.arange(0, T, step_size)[::-1]  # T~0 역순

        # 모델 (None이면 가우시안 fallback)
        self._model = None
        self._is_trained = False
        self._context = None  # 현재 상태 컨텍스트

    def set_model(self, model, trained: bool = True) -> None:
        """Diffusion 모델 설정."""
        self._model = model
        self._is_trained = trained

    def set_context(self, state: np.ndarray) -> None:
        """현재 상태 컨텍스트 설정 (compute_control 전에 호출)."""
        self._context = np.asarray(state, dtype=np.float64)

    def _predict_noise(
        self,
        x_t: np.ndarray,
        t_idx: int,
        context: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        노이즈 예측 (NumPy 인터페이스).

        Args:
            x_t: (K, D) 노이즈 섞인 제어 시퀀스
            t_idx: 정수 타임스텝 인덱스 (0~T-1)
            context: (nx,) 상태 컨텍스트

        Returns:
            eps: (K, D) 예측 노이즈
        """
        if self._model is None or not self._is_trained:
            return np.zeros_like(x_t)

        K, D = x_t.shape
        t_norm = np.full(K, t_idx / self.T, dtype=np.float64)

        if context is not None:
            ctx = np.tile(context[None, :], (K, 1))
        else:
            ctx = None

        # PyTorch 모델
        try:
            import torch
            with torch.no_grad():
                x_t_t = torch.tensor(x_t, dtype=torch.float32)
                t_t = torch.tensor(t_norm, dtype=torch.float32)
                ctx_t = torch.tensor(ctx, dtype=torch.float32) if ctx is not None else None
                eps = self._model(x_t_t, t_t, ctx_t)
                return eps.numpy()
        except Exception:
            # NumpyMLPDiffusion fallback
            if hasattr(self._model, 'predict'):
                return self._model.predict(x_t, t_norm, ctx)
            return np.zeros_like(x_t)

    def _ddim_step(
        self,
        x_t: np.ndarray,
        t: int,
        t_prev: int,
        context: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        단일 DDIM 역확산 스텝.

        x_{t_prev} = √ᾱ_{t_prev} * pred_x0
                    + √(1-ᾱ_{t_prev}) * ε_θ(x_t, t, c)

        Args:
            x_t: (K, D)
            t: 현재 스텝
            t_prev: 이전 스텝 (t_prev < t)
            context: (nx,)

        Returns:
            x_prev: (K, D)
        """
        ab_t = self.alpha_bar[t]
        ab_prev = self.alpha_bar[t_prev]

        eps = self._predict_noise(x_t, t, context)  # (K, D)

        # pred x0
        pred_x0 = (x_t - np.sqrt(1.0 - ab_t) * eps) / np.sqrt(ab_t)
        pred_x0 = np.clip(pred_x0, -5.0, 5.0)  # 수치 안정성

        # DDIM update (결정론적 η=0)
        x_prev = np.sqrt(ab_prev) * pred_x0 + np.sqrt(1.0 - ab_prev) * eps

        return x_prev

    def _diffusion_sample(
        self,
        U: np.ndarray,
        K: int,
        context: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        DDIM 역확산으로 K개 제어 시퀀스 생성.

        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 수
            context: (nx,)

        Returns:
            samples: (K, N, nu) 생성 제어 시퀀스
        """
        N, nu = U.shape
        D = N * nu

        # 순수 노이즈에서 시작
        x_t = np.random.randn(K, D) * self.sigma.reshape(-1)[:nu].mean()

        # DDIM 역확산
        prev_t = 0
        for i, t in enumerate(self.ddim_timesteps):
            t_prev = self.ddim_timesteps[i + 1] if i + 1 < len(self.ddim_timesteps) else 0
            x_t = self._ddim_step(x_t, int(t), int(t_prev), context)

        # (K, D) → (K, N, nu)
        samples = x_t.reshape(K, N, nu)

        # 명목 제어 주변으로 이동 (bias correction)
        samples = samples + U[None, :, :]

        return samples

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        K개 제어 노이즈 시퀀스 생성.

        Args:
            U: (N, nu) 명목 제어
            K: 샘플 수

        Returns:
            noise: (K, N, nu) 노이즈 (controls = U + noise)
        """
        N, nu = U.shape

        if not self._is_trained or self._model is None:
            # Fallback: 가우시안 노이즈
            noise = np.random.randn(K, N, nu) * self.sigma[None, None, :nu]
            return noise

        if self.mode == "replace":
            # 완전 대체: 역확산 샘플 → 노이즈로 변환
            samples = self._diffusion_sample(U, K, self._context)
            noise = samples - U[None, :, :]
            # 소규모 탐색 노이즈 추가
            if self.exploration_sigma > 0:
                noise += np.random.randn(K, N, nu) * self.exploration_sigma

        elif self.mode == "blend":
            # 혼합: α*diffusion + (1-α)*gaussian
            diffusion_samples = self._diffusion_sample(U, K, self._context)
            diffusion_noise = diffusion_samples - U[None, :, :]
            gaussian_noise = np.random.randn(K, N, nu) * self.sigma[None, None, :nu]
            noise = self.blend_ratio * diffusion_noise + (1 - self.blend_ratio) * gaussian_noise

        else:
            noise = np.random.randn(K, N, nu) * self.sigma[None, None, :nu]

        return noise


class DDPMSampler(DDIMSampler):
    """
    전체 DDPM 샘플러 (느리지만 이론적으로 정확).

    DDIM 상속 + 확률적 (η=1) 샘플링.
    학습 중 또는 고품질 오프라인 계획에 적합.
    """

    def _ddim_step(
        self,
        x_t: np.ndarray,
        t: int,
        t_prev: int,
        context: Optional[np.ndarray],
    ) -> np.ndarray:
        """DDPM (확률적) 역확산 스텝."""
        ab_t = self.alpha_bar[t]
        ab_prev = self.alpha_bar[t_prev]
        beta_t = 1.0 - ab_t / ab_prev

        eps = self._predict_noise(x_t, t, context)

        # DDPM: x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε) + σ_t * z
        alpha_t = ab_t / ab_prev
        mean = (x_t - beta_t / np.sqrt(1.0 - ab_t) * eps) / np.sqrt(alpha_t)
        sigma_t = np.sqrt(beta_t)
        x_prev = mean + sigma_t * np.random.randn(*x_t.shape)

        return x_prev
