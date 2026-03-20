"""
dsMPPI (Deterministic Sampling MPPI) Controller

랜덤 샘플링 대신 결정론적 샘플(Halton, Sobol, Sigma Points, Grid)을 사용하여
MPPI의 비용 추정 분산을 줄이고, CEM 반복으로 분포를 최적 영역에 집중.

기존 MPPI와의 핵심 차이:
  1. 확률적 → 결정론적 샘플링: 동일 시드 = 동일 결과 (재현성)
  2. QMC 저불일치 시퀀스: 적은 K로도 제어 공간 균등 커버
  3. CEM 반복: elite selection + 분포 업데이트로 최적 영역 zoom-in
  4. 하이브리드 모드: 결정론적 + 소수 랜덤 샘플 혼합 (탐색/활용 균형)

Reference: Walker et al., "Smooth Sampling-Based MPC Using Deterministic Samples",
           arXiv:2601.03893, 2026
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import qmc, norm

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import DeterministicMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class DeterministicMPPIController(MPPIController):
    """
    dsMPPI (Deterministic Sampling MPPI) Controller

    결정론적 샘플 + CEM 반복 최적화로 적은 샘플에서도 효율적이고
    매끄러운 제어를 생성.

    알고리즘:
        1. 결정론적 샘플 생성 (Halton/Sobol/Sigma Points/Grid)
        2. CEM 반복 (n_cem_iters):
           a. Rollout → 비용 계산
           b. Elite selection (상위 elite_ratio)
           c. 분포 업데이트: μ = EMA(μ, mean(elite)), σ = EMA(σ, std(elite))
           d. 새 결정론적 샘플 생성
        3. 최종 MPPI 지수 가중 평균
        4. Receding horizon shift

    Args:
        model: RobotModel 인스턴스
        params: DeterministicMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 기본 가우시안, 하이브리드에 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: DeterministicMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.ds_params = params

        # Halton/Sobol 시퀀스 캐시
        self._qmc_dim = params.N * model.control_dim
        self._halton_sampler = None
        self._sobol_sampler = None

        # 첫 호출 추적
        self._is_first_call = True

        # 통계
        self._iteration_costs = []
        self._cem_stats_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        dsMPPI 제어 계산

        결정론적 샘플 + CEM 반복 + MPPI 가중 평균.

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K
        N = self.params.N
        nu = self.model.control_dim

        # Cold/Warm start 반복 횟수
        if self._is_first_call:
            n_iters = self.ds_params.n_cem_iters_init
            self._is_first_call = False
        else:
            n_iters = self.ds_params.n_cem_iters

        # CEM 분포 초기화: 현재 명목 시퀀스가 평균
        mu = self.U.copy()  # (N, nu)
        sigma = np.outer(np.ones(N), self.params.sigma)  # (N, nu)

        iteration_costs = []
        last_trajectories = None
        last_weights = None
        last_costs = None
        last_controls = None

        for it in range(n_iters):
            # 1. 결정론적 샘플 생성
            sampled_controls = self._generate_deterministic_samples(
                mu, sigma, K
            )  # (K_det, N, nu)

            # 2. 하이브리드: 추가 랜덤 샘플
            if self.ds_params.add_random_samples > 0:
                n_rand = self.ds_params.add_random_samples
                random_noise = np.random.standard_normal((n_rand, N, nu)) * sigma[None, :, :]
                random_controls = mu[None, :, :] + random_noise
                sampled_controls = np.concatenate(
                    [sampled_controls, random_controls], axis=0
                )

            K_total = sampled_controls.shape[0]

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

            # 3. Rollout + 비용
            trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
            costs = self.cost_function.compute_cost(
                trajectories, sampled_controls, reference_trajectory
            )

            iteration_costs.append(float(np.min(costs)))

            # 4. CEM 분포 업데이트
            if self.ds_params.use_cem_update and it < n_iters - 1:
                mu, sigma = self._cem_update(
                    sampled_controls, costs, mu, sigma
                )

            last_trajectories = trajectories
            last_costs = costs
            last_controls = sampled_controls

        # 5. 최종 MPPI 지수 가중 평균
        weights = self._compute_weights(last_costs, self.params.lambda_)
        last_weights = weights

        # 전체 교체식 가중 평균: U = Σ w_k · u_k
        self.U = np.sum(
            weights[:, None, None] * last_controls, axis=0
        )  # (N, nu)

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 첫 제어 추출
        optimal_control = self.U[0].copy()

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 통계
        ess = self._compute_ess(weights)
        best_idx = np.argmin(last_costs)

        cem_stats = {
            "n_iters": n_iters,
            "iteration_costs": iteration_costs,
            "cost_improvement": (
                iteration_costs[0] - iteration_costs[-1]
                if len(iteration_costs) > 1
                else 0.0
            ),
            "sampling_method": self.ds_params.sampling_method,
            "total_samples": last_controls.shape[0],
            "deterministic_samples": K,
            "random_samples": self.ds_params.add_random_samples,
            "final_sigma_mean": float(np.mean(sigma)) if self.ds_params.use_cem_update else float(np.mean(self.params.sigma)),
        }
        self._iteration_costs = iteration_costs
        self._cem_stats_history.append(cem_stats)

        info = {
            "sample_trajectories": last_trajectories,
            "sample_weights": last_weights,
            "best_trajectory": last_trajectories[best_idx],
            "best_cost": last_costs[best_idx],
            "mean_cost": np.mean(last_costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": last_controls.shape[0],
            "deterministic_stats": cem_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _generate_deterministic_samples(
        self, mu: np.ndarray, sigma: np.ndarray, K: int
    ) -> np.ndarray:
        """
        결정론적 샘플 생성

        Args:
            mu: (N, nu) 평균 (현재 명목 시퀀스)
            sigma: (N, nu) 표준편차
            K: 목표 샘플 수

        Returns:
            samples: (K_actual, N, nu) 결정론적 샘플
        """
        method = self.ds_params.sampling_method
        N = mu.shape[0]
        nu = mu.shape[1]
        dim = N * nu

        if method == "halton":
            return self._halton_samples(mu, sigma, K, N, nu, dim)
        elif method == "sobol":
            return self._sobol_samples(mu, sigma, K, N, nu, dim)
        elif method == "sigma_points":
            return self._sigma_point_samples(mu, sigma, N, nu)
        elif method == "grid":
            return self._grid_samples(mu, sigma, K, N, nu)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _halton_samples(
        self, mu: np.ndarray, sigma: np.ndarray, K: int,
        N: int, nu: int, dim: int
    ) -> np.ndarray:
        """Halton 준난수 시퀀스 → PPF 변환"""
        sampler = qmc.Halton(d=dim, scramble=True)
        uniform = sampler.random(n=K)  # (K, dim) in [0, 1]

        # PPF 변환: Φ^{-1}(u) → 정규분포
        normal = norm.ppf(np.clip(uniform, 1e-6, 1 - 1e-6))  # (K, dim)

        # Reshape to (K, N, nu)
        normal = normal.reshape(K, N, nu)

        # μ + σ * z
        samples = mu[None, :, :] + sigma[None, :, :] * normal
        return samples

    def _sobol_samples(
        self, mu: np.ndarray, sigma: np.ndarray, K: int,
        N: int, nu: int, dim: int
    ) -> np.ndarray:
        """Sobol 준난수 시퀀스 → PPF 변환"""
        # Sobol requires K to be power of 2; generate enough and trim
        m = int(np.ceil(np.log2(max(K, 2))))
        K_sobol = 2 ** m

        sampler = qmc.Sobol(d=dim, scramble=True)
        uniform = sampler.random(n=K_sobol)  # (K_sobol, dim) in [0, 1]

        # 필요한 만큼만 사용
        uniform = uniform[:K]

        # PPF 변환
        normal = norm.ppf(np.clip(uniform, 1e-6, 1 - 1e-6))
        normal = normal.reshape(K, N, nu)

        samples = mu[None, :, :] + sigma[None, :, :] * normal
        return samples

    def _sigma_point_samples(
        self, mu: np.ndarray, sigma: np.ndarray,
        N: int, nu: int
    ) -> np.ndarray:
        """
        Sigma Points 방식: μ + √(dim)·σ·e_i, μ - √(dim)·σ·e_i

        2*dim + 1 포인트 (mean 포함)
        """
        dim = N * nu
        scale = np.sqrt(dim)

        mu_flat = mu.flatten()  # (dim,)
        sigma_flat = sigma.flatten()  # (dim,)

        # 2*dim + 1 포인트
        points = np.zeros((2 * dim + 1, dim))
        points[0] = mu_flat  # center point

        for i in range(dim):
            offset = np.zeros(dim)
            offset[i] = scale * sigma_flat[i]
            points[1 + 2 * i] = mu_flat + offset
            points[2 + 2 * i] = mu_flat - offset

        # Reshape to (2*dim+1, N, nu)
        samples = points.reshape(-1, N, nu)
        return samples

    def _grid_samples(
        self, mu: np.ndarray, sigma: np.ndarray, K: int,
        N: int, nu: int
    ) -> np.ndarray:
        """
        Grid 방식: 각 차원에서 균등 격자점

        차원의 저주를 피하기 위해 per-timestep grid를 사용.
        각 timestep에서 nu 차원 격자를 만들어 조합.
        """
        # K개 표본에 가장 가까운 per-dim grid size
        # dim = N*nu는 너무 크므로 per-timestep 접근
        # per-timestep에서 nu 차원 격자 → n_per_dim^nu 조합
        n_per_dim = max(2, int(np.round(K ** (1.0 / nu))))
        K_actual = n_per_dim ** nu

        # 각 차원의 격자점: PPF(균등분할)
        quantiles = np.linspace(0.05, 0.95, n_per_dim)
        z_vals = norm.ppf(quantiles)  # 정규분포 격자점

        # nu차원 격자 생성
        grids = np.meshgrid(*([z_vals] * nu), indexing='ij')
        z_grid = np.stack([g.flatten() for g in grids], axis=-1)  # (K_actual, nu)

        # 모든 timestep에서 동일한 격자 패턴 적용
        samples = np.zeros((K_actual, N, nu))
        for t in range(N):
            samples[:, t, :] = mu[t] + sigma[t] * z_grid

        return samples

    def _cem_update(
        self, controls: np.ndarray, costs: np.ndarray,
        mu: np.ndarray, sigma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CEM 분포 업데이트: elite selection → EMA 업데이트

        Args:
            controls: (K_total, N, nu) 샘플 제어
            costs: (K_total,) 비용
            mu: (N, nu) 현재 평균
            sigma: (N, nu) 현재 표준편차

        Returns:
            mu_new: (N, nu) 업데이트된 평균
            sigma_new: (N, nu) 업데이트된 표준편차
        """
        K_total = len(costs)
        n_elite = max(1, int(K_total * self.ds_params.elite_ratio))

        # Elite selection
        elite_idx = np.argsort(costs)[:n_elite]
        elite_controls = controls[elite_idx]  # (n_elite, N, nu)

        # Elite 평균 / 표준편차
        elite_mean = np.mean(elite_controls, axis=0)  # (N, nu)
        elite_std = np.std(elite_controls, axis=0)  # (N, nu)

        # 표준편차 하한 (축소 방지)
        sigma_min = self.params.sigma * 0.01
        elite_std = np.maximum(elite_std, sigma_min)

        # EMA 업데이트
        alpha = self.ds_params.cem_alpha
        mu_new = (1 - alpha) * mu + alpha * elite_mean
        sigma_new = (1 - alpha) * sigma + alpha * elite_std

        return mu_new, sigma_new

    def get_deterministic_statistics(self) -> Dict:
        """
        dsMPPI 통계 반환

        Returns:
            dict: 누적 통계 정보
        """
        if len(self._cem_stats_history) == 0:
            return {
                "mean_cost_improvement": 0.0,
                "mean_n_iters": 0.0,
                "last_iteration_costs": [],
                "cem_stats_history": [],
                "sampling_method": self.ds_params.sampling_method,
            }

        improvements = [s["cost_improvement"] for s in self._cem_stats_history]
        n_iters_list = [s["n_iters"] for s in self._cem_stats_history]

        return {
            "mean_cost_improvement": float(np.mean(improvements)),
            "mean_n_iters": float(np.mean(n_iters_list)),
            "last_iteration_costs": self._iteration_costs,
            "cem_stats_history": self._cem_stats_history.copy(),
            "sampling_method": self.ds_params.sampling_method,
        }

    def reset(self):
        """제어 시퀀스 + 통계 초기화"""
        super().reset()
        self._is_first_call = True
        self._iteration_costs = []
        self._cem_stats_history = []

    def __repr__(self) -> str:
        return (
            f"DeterministicMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"method={self.ds_params.sampling_method}, "
            f"n_cem_iters={self.ds_params.n_cem_iters}, "
            f"elite_ratio={self.ds_params.elite_ratio}, "
            f"K={self.params.K})"
        )
