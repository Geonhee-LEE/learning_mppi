"""
TR-MPPI Controller (Trust Region MPPI) — 41번째 변형

샘플링 기반 MPC를 신뢰 영역(trust region) 프레임으로 정식화한다. proposal
분포(가우시안 평균)의 업데이트를 KL 발산 경계 δ로 제약하여 과도한 스텝을
방지하고, 공분산에 엔트로피 하한을 두어 조기 붕괴를 막는다. 추가로 결정론적
LCD(localized cumulative distribution) 샘플링(Halton 저불일치 수열 + 역정규
CDF)으로 샘플 효율과 수렴성을 개선한다.

핵심 수식:
    고정 공분산 가우시안의 평균 간 KL:
        KL(q_new‖q_old) = ½ Σ_t Δμ_t^T Σ^{-1} Δμ_t = ½ Σ ‖Δμ/σ‖²
    신뢰 영역 투영 (Δμ = α Σ_i w_i ε_i):
        KL_prop > δ  ⇒  Δμ ← Δμ · sqrt(δ / KL_prop)
    엔트로피 하한:  대각 σ_d ≥ entropy_floor_scale · σ_base
    LCD 결정론적 샘플:  ε = σ · Φ⁻¹(halton_d)

Reference: arXiv:2605.07801, "Sampling-based Model Predictive Control Using
Trust Regions" (2026)
"""

import numpy as np
from typing import Dict, Tuple, Optional

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import TRMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


def _inverse_normal_cdf(u: np.ndarray) -> np.ndarray:
    """
    역 표준정규 CDF Φ⁻¹(u).

    scipy.special.ndtri 사용, 없으면 Acklam 유리근사로 폴백.
    입력은 (0,1)로 클램프하여 ±inf 방지.
    """
    u = np.clip(u, 1e-9, 1.0 - 1e-9)
    try:
        from scipy.special import ndtri
        return ndtri(u)
    except Exception:
        # Acklam's algorithm (정확도 ~1e-9)
        a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        x = np.zeros_like(u)
        lo = u < plow
        hi = u > phigh
        mid = ~(lo | hi)
        ql = np.sqrt(-2 * np.log(u[lo]))
        x[lo] = (((((c[0]*ql+c[1])*ql+c[2])*ql+c[3])*ql+c[4])*ql+c[5]) / \
                ((((d[0]*ql+d[1])*ql+d[2])*ql+d[3])*ql+1)
        qh = np.sqrt(-2 * np.log(1 - u[hi]))
        x[hi] = -(((((c[0]*qh+c[1])*qh+c[2])*qh+c[3])*qh+c[4])*qh+c[5]) / \
                ((((d[0]*qh+d[1])*qh+d[2])*qh+d[3])*qh+1)
        qm = u[mid] - 0.5
        rm = qm * qm
        x[mid] = (((((a[0]*rm+a[1])*rm+a[2])*rm+a[3])*rm+a[4])*rm+a[5])*qm / \
                 (((((b[0]*rm+b[1])*rm+b[2])*rm+b[3])*rm+b[4])*rm+1)
        return x


_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
           67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
           139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]


def _van_der_corput(n_points: int, base: int, skip: int = 1) -> np.ndarray:
    """base 진법 van der Corput 수열 (n_points,)"""
    idx = np.arange(skip, skip + n_points)
    result = np.zeros(n_points, dtype=float)
    f = 1.0 / base
    i = idx.astype(np.float64).copy()
    bf = f
    while np.any(i > 0):
        result += (i % base) * bf
        i = np.floor(i / base)
        bf /= base
    return result


class HaltonLCDSampler(NoiseSampler):
    """
    Halton 저불일치 수열 기반 결정론적 LCD 샘플러.

    (N·nu) 차원의 Halton 점을 생성하고 역정규 CDF로 변환하여 결정론적
    표준정규 유사 샘플을 만든다. 동일 K에 대해 항상 같은 샘플을 반환하므로
    분산이 낮고 수렴이 안정적이다.

    Args:
        sigma: (nu,) 표준편차
        skip: Halton 초기 항 스킵(상관 완화)
    """

    def __init__(self, sigma: np.ndarray, skip: int = 20):
        self.sigma = np.asarray(sigma, dtype=float)
        self.skip = skip

    def unit_samples(self, K: int, N: int, nu: int) -> np.ndarray:
        """결정론적 표준정규 유사 샘플 (K, N, nu)"""
        d = N * nu
        cols = []
        for j in range(d):
            base = _PRIMES[j % len(_PRIMES)]
            cols.append(_van_der_corput(K, base, skip=self.skip + 1))
        u = np.stack(cols, axis=1)            # (K, d)
        z = _inverse_normal_cdf(u)            # (K, d)
        return z.reshape(K, N, nu)

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        N, nu = U.shape
        z = self.unit_samples(K, N, nu)
        noise = z * self.sigma[None, None, :]
        if control_min is not None and control_max is not None:
            sampled = np.clip(U + noise, control_min, control_max)
            noise = sampled - U
        return noise

    def __repr__(self) -> str:
        return f"HaltonLCDSampler(sigma={self.sigma}, skip={self.skip})"


class TRMPPIController(MPPIController):
    """
    TR-MPPI Controller — 41번째 변형

    신뢰 영역 제약 하에 proposal 평균을 업데이트한다. 매 반복마다:
      1. (Gaussian 또는 Halton-LCD) 노이즈 샘플링
      2. rollout / cost / softmax 가중치
      3. 제안 업데이트 Δμ = α Σ w_i ε_i
      4. KL_prop = ½‖Δμ/σ‖² 이 δ를 넘으면 Δμ를 sqrt(δ/KL_prop)로 축소
      5. μ += Δμ; 선택적 공분산 적응 + 엔트로피 하한
    """

    def __init__(
        self,
        model: RobotModel,
        params: TRMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.tr_params = params

        nu = model.control_dim
        base_sigma = np.asarray(params.sigma, dtype=float)
        if base_sigma.ndim == 0 or base_sigma.size == 1:
            base_sigma = np.full(nu, float(base_sigma.reshape(-1)[0]))
        self._base_sigma = base_sigma.copy()
        self._sigma_scale = np.ones(nu)

        self._rng = np.random.default_rng(getattr(params, "seed", None))
        self._lcd = HaltonLCDSampler(self._base_sigma)

        # 통계
        self._kl_history = []
        self._scaled_count = 0
        self._total_iters = 0

    def _current_sigma(self) -> np.ndarray:
        return self._base_sigma * self._sigma_scale

    def _sample_noise(self, K: int, sigma: np.ndarray) -> np.ndarray:
        N = self.params.N
        nu = self._base_sigma.shape[0]
        if self.tr_params.use_deterministic_sampling:
            z = self._lcd.unit_samples(K, N, nu)
        else:
            z = self._rng.normal(0.0, 1.0, (K, N, nu))
        return z * sigma[None, None, :]

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        K = self.params.K
        lambda_ = self.params.lambda_
        delta = self.tr_params.trust_region_radius

        mu = self.U.copy()
        sample_trajectories = None
        weights = None
        costs = None
        last_kl = 0.0
        last_scaled = False

        for it in range(self.tr_params.n_iters):
            sigma = self._current_sigma()
            noise = self._sample_noise(K, sigma)  # (K, N, nu)

            sampled_controls = mu[None] + noise
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)
                noise = sampled_controls - mu[None]

            sample_trajectories = self.dynamics_wrapper.rollout(
                state, sampled_controls
            )
            costs = self.cost_function.compute_cost(
                sample_trajectories, sampled_controls, reference_trajectory
            )
            weights = self._compute_weights(costs, lambda_)

            delta_mu = np.sum(weights[:, None, None] * noise, axis=0)  # (N, nu)

            # KL_prop = ½ Σ ‖Δμ/σ‖²
            sig = np.maximum(sigma, 1e-9)
            kl_prop = 0.5 * float(np.sum((delta_mu / sig[None, :]) ** 2))
            last_kl = kl_prop

            if self.tr_params.use_kl_bound and kl_prop > delta and kl_prop > 0:
                scale = np.sqrt(delta / kl_prop)
                delta_mu = delta_mu * scale
                last_scaled = True
                self._scaled_count += 1
            else:
                last_scaled = False

            mu = mu + delta_mu
            if self.u_min is not None and self.u_max is not None:
                mu = np.clip(mu, self.u_min, self.u_max)

            if self.tr_params.adapt_covariance:
                self._adapt_covariance(sampled_controls, mu, weights)

            self._total_iters += 1

        self._kl_history.append(last_kl)

        self.U = mu
        optimal_control = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        ess = self._compute_ess(weights)
        best_idx = int(np.argmin(costs))

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": lambda_,
            "ess": ess,
            "num_samples": K,
            "tr_stats": {
                "kl_divergence": last_kl,
                "trust_region_radius": delta,
                "step_scaled": last_scaled,
                "deterministic": self.tr_params.use_deterministic_sampling,
                "sigma_scale": self._sigma_scale.copy(),
            },
        }
        self.last_info = info
        return optimal_control, info

    def _adapt_covariance(
        self, sampled_controls: np.ndarray, mu: np.ndarray, weights: np.ndarray
    ):
        """가중 경험 공분산 적응 + 엔트로피 하한(σ_floor) + 상한."""
        dev = sampled_controls - mu[None]
        var_w = np.sum(weights[:, None, None] * dev ** 2, axis=0)  # (N, nu)
        var_w = np.mean(var_w, axis=0)                             # (nu,)
        std_w = np.sqrt(np.maximum(var_w, 1e-12))

        target_scale = std_w / np.maximum(self._base_sigma, 1e-9)
        beta = self.tr_params.cov_step_size
        self._sigma_scale = (1.0 - beta) * self._sigma_scale + beta * target_scale
        # 엔트로피 하한: σ ≥ entropy_floor_scale·σ_base, 상한: cov_max_scale
        self._sigma_scale = np.clip(
            self._sigma_scale,
            self.tr_params.entropy_floor_scale,
            self.tr_params.cov_max_scale,
        )

    def get_tr_statistics(self) -> Dict:
        hist = self._kl_history
        return {
            "n_steps": len(hist),
            "mean_kl": float(np.mean(hist)) if hist else 0.0,
            "max_kl": float(np.max(hist)) if hist else 0.0,
            "scaled_fraction": (
                self._scaled_count / self._total_iters
                if self._total_iters > 0 else 0.0
            ),
            "sigma_scale": self._sigma_scale.copy(),
        }

    def reset(self):
        super().reset()
        self._sigma_scale = np.ones(self._base_sigma.shape[0])
        self._kl_history = []
        self._scaled_count = 0
        self._total_iters = 0

    def __repr__(self) -> str:
        return (
            f"TRMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"trust_region_radius={self.tr_params.trust_region_radius}, "
            f"deterministic={self.tr_params.use_deterministic_sampling}, "
            f"n_iters={self.tr_params.n_iters}, K={self.params.K})"
        )
