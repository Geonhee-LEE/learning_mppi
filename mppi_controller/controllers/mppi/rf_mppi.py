"""
RF-MPPI Controller (Reference-Free Spline MPPI) — 42번째 변형

제어 시퀀스를 저차원 큐빅 Hermite 스플라인으로 파라미터화한다. 소수의
제어점(knot) 공간에서 섭동을 샘플링하고 매끄러운 N-스텝 제어로 보간하여,
적은 샘플(K≈32~64)로도 매끄럽고 다양한 모션을 탐색한다. "dual-space" —
각 knot이 위치(position) 제어점과 속도(velocity, Hermite 접선) 제어점을
모두 가져 풍부한 동작 공간을 표현한다. 매끄러움이 구조적으로 보장되므로
GPU 없이 CPU에서 소수 샘플로 실시간 동작이 가능하다.

핵심 수식 (큐빅 Hermite 기저):
    s∈[0,1], Δτ = knot 간격(스텝 단위)
    h00 = 2s³-3s²+1, h10 = s³-2s²+s, h01 = -2s³+3s², h11 = s³-s²
    u(t) = h00 p_m + Δτ·h10 v_m + h01 p_{m+1} + Δτ·h11 v_{m+1}
         = (B_p @ P + B_v @ V)[t]          (선형 기저 → 벡터화)
    섭동: P += N(0,σ_p²),  V += N(0,σ_v²)   (저차원 knot 공간)

Reference: arXiv:2511.19204, "Reference-Free Sampling-Based Model Predictive
Control" (2026)
"""

import numpy as np
from typing import Dict, Tuple, Optional

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import RFMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class HermiteSplineSampler:
    """
    큐빅 Hermite 스플라인 기저 (dual-space: 위치 + 속도 제어점).

    N개 타임스텝을 M개 knot으로 파라미터화한다. knot 시간은 [0, N-1]에
    균등 분포. 위치/속도 기저 행렬 B_p, B_v (N×M)을 사전 계산하여
    U = B_p @ P + B_v @ V 의 선형 형태로 매끄러운 제어를 재구성한다.

    Args:
        N: 타임스텝 수
        n_knots: 제어점 수 M (>= 2)
    """

    def __init__(self, N: int, n_knots: int):
        assert n_knots >= 2 and n_knots <= N + 1
        self.N = N
        self.M = n_knots
        self.knot_times = np.linspace(0.0, N - 1, n_knots)  # (M,)
        self.B_p, self.B_v = self._build_basis(np.arange(N, dtype=float))
        # 속도 워밍 시프트용 도함수 기저
        self.dB_p, self.dB_v = self._build_deriv_basis(np.arange(N, dtype=float))

    def _segment_of(self, t: float) -> Tuple[int, float, float]:
        """시간 t가 속한 세그먼트 인덱스 m, 국소 s∈[0,1], 세그먼트 길이 Δτ."""
        kt = self.knot_times
        # 마지막 세그먼트로 클램프
        m = int(np.searchsorted(kt, t, side="right") - 1)
        m = max(0, min(m, self.M - 2))
        tau = kt[m + 1] - kt[m]
        s = (t - kt[m]) / tau if tau > 0 else 0.0
        s = float(np.clip(s, 0.0, 1.0))
        return m, s, tau

    def _build_basis(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """위치/속도 기저 (len(times), M)."""
        T = len(times)
        B_p = np.zeros((T, self.M))
        B_v = np.zeros((T, self.M))
        for i, t in enumerate(times):
            m, s, tau = self._segment_of(float(t))
            s2, s3 = s * s, s * s * s
            h00 = 2 * s3 - 3 * s2 + 1
            h10 = s3 - 2 * s2 + s
            h01 = -2 * s3 + 3 * s2
            h11 = s3 - s2
            B_p[i, m] += h00
            B_p[i, m + 1] += h01
            B_v[i, m] += tau * h10
            B_v[i, m + 1] += tau * h11
        return B_p, B_v

    def _build_deriv_basis(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """위치/속도에 대한 du/dt 기저 (워밍 시프트의 속도 재추정용)."""
        T = len(times)
        dB_p = np.zeros((T, self.M))
        dB_v = np.zeros((T, self.M))
        for i, t in enumerate(times):
            m, s, tau = self._segment_of(float(t))
            s2 = s * s
            # ds/dt = 1/tau
            dh00 = (6 * s2 - 6 * s) / tau if tau > 0 else 0.0
            dh10 = (3 * s2 - 4 * s + 1) / tau if tau > 0 else 0.0
            dh01 = (-6 * s2 + 6 * s) / tau if tau > 0 else 0.0
            dh11 = (3 * s2 - 2 * s) / tau if tau > 0 else 0.0
            dB_p[i, m] += dh00
            dB_p[i, m + 1] += dh01
            dB_v[i, m] += tau * dh10
            dB_v[i, m + 1] += tau * dh11
        return dB_p, dB_v

    def reconstruct(self, P: np.ndarray, V: np.ndarray) -> np.ndarray:
        """(M,nu) 위치/속도 knot → (N,nu) 제어 시퀀스."""
        return self.B_p @ P + self.B_v @ V

    def reconstruct_batch(self, P: np.ndarray, V: np.ndarray) -> np.ndarray:
        """(K,M,nu) 배치 → (K,N,nu) 제어 시퀀스."""
        return np.einsum("nm,kmu->knu", self.B_p, P) + \
            np.einsum("nm,kmu->knu", self.B_v, V)

    def eval_shifted_knots(
        self, P: np.ndarray, V: np.ndarray, shift: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        스플라인을 시간 +shift 만큼 이동하여 새 knot 위치/속도를 재추정.
        (receding horizon warm-start)
        """
        new_times = np.clip(self.knot_times + shift, 0.0, self.N - 1)
        Bp_s, Bv_s = self._build_basis(new_times)
        dBp_s, dBv_s = self._build_deriv_basis(new_times)
        P_new = Bp_s @ P + Bv_s @ V
        V_new = dBp_s @ P + dBv_s @ V
        return P_new, V_new


class RFMPPIController(MPPIController):
    """
    RF-MPPI Controller — 42번째 변형

    위치/속도 knot 공간에서 섭동을 샘플링하고 Hermite 보간으로 매끄러운
    제어 시퀀스를 재구성한다. MPPI 가중치로 knot을 갱신하고, receding
    horizon에서 스플라인을 시간 이동하여 warm-start 한다.
    """

    def __init__(
        self,
        model: RobotModel,
        params: RFMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.rf_params = params

        nu = model.control_dim
        self.nu = nu
        self.M = params.n_knots
        self.spline = HermiteSplineSampler(params.N, params.n_knots)

        # knot 섭동 표준편차
        if params.knot_sigma_pos is not None:
            sp = np.asarray(params.knot_sigma_pos, dtype=float)
            if sp.size == 1:
                sp = np.full(nu, float(sp.reshape(-1)[0]))
        else:
            base = np.asarray(params.sigma, dtype=float)
            if base.size == 1:
                base = np.full(nu, float(base.reshape(-1)[0]))
            sp = base
        self._sigma_p = sp                                  # (nu,)
        self._sigma_v = float(params.knot_sigma_vel)

        # 명목 knot (위치/속도)
        self._P = np.zeros((self.M, nu))
        self._V = np.zeros((self.M, nu))

        self._rng = np.random.default_rng(getattr(params, "seed", None))

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        K = self.params.K
        nu = self.nu
        M = self.M

        # knot 공간 섭동 샘플
        dP = self._rng.normal(0.0, 1.0, (K, M, nu)) * self._sigma_p[None, None, :]
        if self.rf_params.sample_velocity_knots:
            dV = self._rng.normal(0.0, 1.0, (K, M, nu)) * self._sigma_v
        else:
            dV = np.zeros((K, M, nu))

        P_k = self._P[None] + dP                            # (K, M, nu)
        V_k = self._V[None] + dV

        # 매끄러운 제어 시퀀스 재구성 (K, N, nu)
        sampled_controls = self.spline.reconstruct_batch(P_k, V_k)
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )
        weights = self._compute_weights(costs, self.params.lambda_)

        # knot 갱신 (가중 평균 섭동)
        self._P = self._P + np.sum(weights[:, None, None] * dP, axis=0)
        self._V = self._V + np.sum(weights[:, None, None] * dV, axis=0)

        if self.rf_params.clamp_endpoints_vel:
            self._V[0] = 0.0
            self._V[-1] = 0.0

        # 명목 제어 시퀀스 (base 호환 / info)
        self.U = self.spline.reconstruct(self._P, self._V)
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        optimal_control = self.U[0].copy()

        # receding horizon: 스플라인 시간 +1 이동하여 warm-start
        if self.rf_params.spline_warm_shift:
            self._P, self._V = self.spline.eval_shifted_knots(
                self._P, self._V, shift=1.0
            )
            self.U = self.spline.reconstruct(self._P, self._V)

        ess = self._compute_ess(weights)
        best_idx = int(np.argmin(costs))

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "rf_stats": {
                "n_knots": M,
                "dual_space": self.rf_params.sample_velocity_knots,
                "knot_pos_norm": float(np.linalg.norm(self._P)),
                "knot_vel_norm": float(np.linalg.norm(self._V)),
            },
        }
        self.last_info = info
        return optimal_control, info

    def reset(self):
        super().reset()
        self._P = np.zeros((self.M, self.nu))
        self._V = np.zeros((self.M, self.nu))

    def __repr__(self) -> str:
        return (
            f"RFMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"n_knots={self.M}, dual_space={self.rf_params.sample_velocity_knots}, "
            f"K={self.params.K})"
        )
