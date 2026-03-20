"""
SVG-MPPI Controller (Stein Variational Guided MPPI)

Honda et al., "Stein Variational Guided Model Predictive Path Integral Control",
ICRA 2024, arXiv:2309.11040

핵심 아이디어:
  - 수정된 SVGD(Stein Variational Gradient Descent)로 최적 분포의 목표 모드 식별
  - SVGD 커널로 학습 없이 실시간 다중 모드 탐색 (비모수적)
  - 샘플링 분포 자체를 SVGD로 최적화하여 multimodal 비용 지형에서 유리

수학적 배경:
  1. 기존 MPPI: U_new = U + Sigma omega_k epsilon_k (단일 모드 가중 평균)
  2. SVG-MPPI: SVGD 파티클 업데이트
     phi*(x) = E_{x'~q}[k(x', x) nabla_{x'} log p(x') + nabla_{x'} k(x', x)]
     - 첫 항: 비용 기울기 방향 (exploitation)
     - 둘째 항: 커널 반발력 (exploration, 다양성 유지)
  3. RBF 커널: k(x, x') = exp(-||x-x'||^2 / (2h^2)), h = median heuristic
  4. SVGD 업데이트 후 MPPI 가중 평균으로 최종 제어 결정

기존 SVG-MPPI(Kondo 2024)의 Guide Particle 패턴 유지하면서
Honda et al. 2024의 blend_ratio, warm start, SVGD step schedule 추가.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import SVGMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.utils.stein_variational import (
    rbf_kernel_with_bandwidth,
    compute_svgd_update_efficient,
)


class SVGMPPIController(MPPIController):
    """
    SVG-MPPI Controller (28번째 MPPI 변형)

    Stein Variational Guided MPPI — SVGD로 파티클을 최적 분포 모드로
    이동시킨 뒤, SVGD 파티클과 가우시안 샘플을 혼합하여 MPPI 업데이트.

    Vanilla MPPI 대비 핵심 차이:
        1. Guide particle 선택: 최저 비용 G개를 SVGD 대상으로 선택
        2. SVGD 업데이트: RBF 커널 + 비용 기울기로 파티클 이동
           - Exploitation: k(x',x) * (-nabla_cost/lambda) — 비용 감소 방향
           - Exploration: nabla_k(x',x) — 파티클 간 반발력으로 다양성 유지
        3. 혼합 샘플링: SVGD 파티클(비율=blend_ratio) + 가우시안(1-blend_ratio)
        4. Warm start: 이전 SVGD 파티클을 다음 스텝 초기화에 재사용
        5. 표준 MPPI 가중 평균으로 최종 제어 결정

    핵심 수식:
        phi_k = (1/G) sum_j [k(x_j, x_k) * (-nabla_cost_j / lambda) + nabla_k(x_j, x_k)]
        particle_k += lr * phi_k

    Args:
        model: RobotModel 인스턴스
        params: SVGMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: SVGMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.svg_params = params

        # Guide particle 파라미터 (기존 호환)
        self.G = params.svg_num_guide_particles
        self.svg_step_size = params.svg_guide_step_size
        self.svgd_iterations = params.svgd_num_iterations

        # Honda et al. 2024 추가 파라미터
        self.n_svgd_steps = params.n_svgd_steps
        self.temperature_svgd = params.temperature_svgd
        self.use_svgd_warm_start = params.use_svgd_warm_start
        self.blend_ratio = params.blend_ratio
        self.use_spsa_gradient = params.use_spsa_gradient
        self.step_size_schedule = params.svgd_step_size_schedule

        # Warm start 파티클 저장
        self._warm_particles = None

        # 통계 (디버깅용)
        self.svg_stats_history = []

        # 검증
        if self.G >= self.params.K:
            raise ValueError(
                f"svg_num_guide_particles ({self.G}) must be < K ({self.params.K})"
            )

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        SVG-MPPI 제어 계산

        1. SVGD 파티클 초기화 (warm start 또는 가우시안)
        2. n_svgd_steps 반복: rollout -> 비용 -> RBF 커널 -> SVGD 업데이트
        3. SVGD 파티클 + 가우시안 샘플 혼합 (blend_ratio)
        4. 최종 rollout -> 비용 -> 표준 MPPI 가중 평균
        5. Receding horizon shift

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K
        G = self.G
        N = self.params.N
        nu = self.model.control_dim

        # ─── 1. SVGD 파티클 초기화 ──────────────────────────
        _warm_start_used = (
            self.use_svgd_warm_start and self._warm_particles is not None
        )
        if _warm_start_used:
            # Warm start: 이전 파티클을 receding horizon shift 후 재사용
            guide_controls = self._warm_particles.copy()
            # 가우시안 교란 추가 (탐색 유지)
            warm_noise = np.random.normal(
                0, self.params.sigma * 0.3, guide_controls.shape
            )
            guide_controls = guide_controls + warm_noise
        else:
            # Cold start: 가우시안 샘플에서 최저 비용 G개 선택
            noise_init = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
            init_controls = self.U + noise_init
            if self.u_min is not None and self.u_max is not None:
                init_controls = np.clip(init_controls, self.u_min, self.u_max)

            init_trajectories = self.dynamics_wrapper.rollout(state, init_controls)
            init_costs = self.cost_function.compute_cost(
                init_trajectories, init_controls, reference_trajectory
            )
            guide_indices_init = np.argsort(init_costs)[:G]
            guide_controls = init_controls[guide_indices_init].copy()

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            guide_controls = np.clip(guide_controls, self.u_min, self.u_max)

        # ─── 2. SVGD 반복 업데이트 ──────────────────────────
        # 유효 SVGD 스텝 수: n_svgd_steps 또는 svgd_num_iterations (기존 호환)
        effective_steps = max(self.n_svgd_steps, self.svgd_iterations)

        initial_guide_cost = None
        bandwidth = 1.0

        for step_idx in range(effective_steps):
            # 2a. 파티클 rollout -> 비용 계산
            guide_traj = self.dynamics_wrapper.rollout(state, guide_controls)
            guide_costs = self.cost_function.compute_cost(
                guide_traj, guide_controls, reference_trajectory
            )

            if step_idx == 0:
                initial_guide_cost = float(np.mean(guide_costs))

            # 2b. RBF 커널 행렬 계산 (median bandwidth)
            kernel, bandwidth = rbf_kernel_with_bandwidth(guide_controls)
            if bandwidth < 1e-10:
                bandwidth = 1.0  # 파티클 collapse 시 기본 bandwidth

            # 2c. 비용 기울기 추정
            if self.use_spsa_gradient:
                grad_costs = self._estimate_cost_gradient_spsa(
                    state, guide_controls, reference_trajectory
                )
            else:
                grad_costs = self._estimate_cost_gradient(
                    state, guide_controls, reference_trajectory
                )

            # grad_log_prob = -grad_cost / temperature_svgd
            # NaN/Inf guard
            if np.any(~np.isfinite(grad_costs)):
                grad_costs = np.nan_to_num(grad_costs, nan=0.0, posinf=0.0, neginf=0.0)
            grad_log_prob = -grad_costs / self.temperature_svgd

            # 2d. SVGD 업데이트: phi = k * grad_log_prob + grad_k
            phi = compute_svgd_update_efficient(
                guide_controls, grad_log_prob, kernel, bandwidth
            )

            # 2e. 스텝 크기 스케줄
            if self.step_size_schedule == "decay":
                lr = self.svg_step_size / (1.0 + step_idx * 0.5)
            else:
                lr = self.svg_step_size

            # 2f. 파티클 업데이트 (NaN guard)
            update = lr * phi
            if np.any(np.isnan(update)):
                break  # SVGD 발산 — 현재 파티클로 중단
            guide_controls = guide_controls + update

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                guide_controls = np.clip(guide_controls, self.u_min, self.u_max)

        # SVGD 후 비용
        guide_traj_final = self.dynamics_wrapper.rollout(state, guide_controls)
        guide_costs_final = self.cost_function.compute_cost(
            guide_traj_final, guide_controls, reference_trajectory
        )
        final_guide_cost = float(np.mean(guide_costs_final))

        if initial_guide_cost is None:
            initial_guide_cost = final_guide_cost

        # ─── 3. SVGD 파티클 + 가우시안 혼합 ────────────────
        n_svgd_samples = G  # SVGD 파티클
        n_gaussian_samples = K - G  # 가우시안 샘플

        if self.blend_ratio < 1.0 and n_gaussian_samples > 0:
            # Follower: 각 가우시안 샘플을 랜덤 guide 주변에 분포
            follower_controls = np.zeros((n_gaussian_samples, N, nu))
            guide_assignments = np.random.choice(G, size=n_gaussian_samples)

            for i, g_idx in enumerate(guide_assignments):
                follower_noise = np.random.normal(
                    0, self.params.sigma * 0.5, (N, nu)
                )
                follower_controls[i] = guide_controls[g_idx] + follower_noise

            if self.u_min is not None and self.u_max is not None:
                follower_controls = np.clip(
                    follower_controls, self.u_min, self.u_max
                )

            # 혼합: SVGD 파티클 + 가우시안 follower
            all_controls = np.concatenate(
                [guide_controls, follower_controls], axis=0
            )
        else:
            # blend_ratio=1.0: SVGD 파티클만 사용
            all_controls = guide_controls

        # ─── 4. 최종 rollout -> 비용 -> MPPI 가중 평균 ─────
        all_trajectories = self.dynamics_wrapper.rollout(state, all_controls)
        all_costs = self.cost_function.compute_cost(
            all_trajectories, all_controls, reference_trajectory
        )

        # MPPI 가중치
        weights = self._compute_weights(all_costs, self.params.lambda_)

        # ESS 계산
        ess = self._compute_ess(weights)

        # 제어 업데이트 (가중 평균 노이즈)
        all_noise = all_controls - self.U
        weighted_noise = np.sum(weights[:, None, None] * all_noise, axis=0)
        self.U = self.U + weighted_noise

        # ─── 5. Warm start 저장 + Receding horizon ────────
        # Warm start: SVGD 파티클을 다음 스텝 용으로 저장
        if self.use_svgd_warm_start:
            self._warm_particles = np.roll(guide_controls.copy(), -1, axis=1)
            self._warm_particles[:, -1, :] = 0.0

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # ─── 6. 최적 제어 추출 ────────────────────────────
        optimal_control = self.U[0, :].copy()
        if self.u_min is not None and self.u_max is not None:
            optimal_control = np.clip(optimal_control, self.u_min, self.u_max)

        # ─── 7. 통계 + info ──────────────────────────────
        best_idx = np.argmin(all_costs)

        svg_stats = {
            "num_guides": G,
            "num_followers": n_gaussian_samples if self.blend_ratio < 1.0 else 0,
            "svgd_iterations": effective_steps,
            "guide_step_size": self.svg_step_size,
            "initial_guide_cost": initial_guide_cost,
            "final_guide_cost": final_guide_cost,
            "guide_cost_improvement": initial_guide_cost - final_guide_cost,
            "guide_mean_cost": final_guide_cost,
            "follower_mean_cost": (
                float(np.mean(all_costs[G:])) if len(all_costs) > G else 0.0
            ),
            "bandwidth": bandwidth,
            "blend_ratio": self.blend_ratio,
            "warm_start_used": _warm_start_used,
            "n_svgd_steps": effective_steps,
            "temperature_svgd": self.temperature_svgd,
        }
        self.svg_stats_history.append(svg_stats)

        info = {
            "sample_trajectories": all_trajectories,
            "sample_controls": all_controls,
            "sample_weights": weights,
            "guide_indices": np.arange(G),  # SVGD 파티클이 첫 G개
            "guide_controls": guide_controls,
            "best_trajectory": all_trajectories[best_idx],
            "best_cost": float(np.min(all_costs)),
            "mean_cost": float(np.mean(all_costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": len(all_controls),
            "svg_stats": svg_stats,
        }

        return optimal_control, info

    def _estimate_cost_gradient_spsa(
        self,
        state: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
        epsilon: float = 1e-3,
    ) -> np.ndarray:
        """
        SPSA (Simultaneous Perturbation Stochastic Approximation)로 비용 gradient 추정

        전 차원을 동시 섭동하여 2회 rollout만으로 gradient를 추정.
        O(2) rollout vs O(N*nu) finite difference.

        Args:
            state: (nx,) 현재 상태
            controls: (G, N, nu) 제어 시퀀스
            reference_trajectory: (N+1, nx) 레퍼런스
            epsilon: Perturbation step

        Returns:
            grad_costs: (G, N, nu) 비용 gradient
        """
        G, N, nu = controls.shape

        # Rademacher 랜덤 방향 (+/-1)
        delta = np.random.choice([-1.0, 1.0], size=(G, N, nu))

        controls_plus = controls + epsilon * delta
        controls_minus = controls - epsilon * delta

        if self.u_min is not None and self.u_max is not None:
            controls_plus = np.clip(controls_plus, self.u_min, self.u_max)
            controls_minus = np.clip(controls_minus, self.u_min, self.u_max)

        traj_plus = self.dynamics_wrapper.rollout(state, controls_plus)
        traj_minus = self.dynamics_wrapper.rollout(state, controls_minus)

        cost_plus = self.cost_function.compute_cost(
            traj_plus, controls_plus, reference_trajectory
        )
        cost_minus = self.cost_function.compute_cost(
            traj_minus, controls_minus, reference_trajectory
        )

        # SPSA gradient: g_i = (C+ - C-) / (2 * epsilon * delta_i)
        grad_costs = (
            (cost_plus - cost_minus)[:, None, None] / (2.0 * epsilon * delta)
        )

        return grad_costs

    def _estimate_cost_gradient(
        self,
        state: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """
        유한 차분으로 비용 그래디언트 추정 (차원별)

        정확하지만 O(N*nu) rollout 필요 — 작은 G에서만 실용적.

        Args:
            state: (nx,) 현재 상태
            controls: (G, N, nu) 제어 시퀀스
            reference_trajectory: (N+1, nx) 레퍼런스
            epsilon: 유한 차분 스텝

        Returns:
            grad_costs: (G, N, nu) 비용 그래디언트
        """
        G, N, nu = controls.shape
        grad_costs = np.zeros((G, N, nu))

        for t in range(N):
            for u_dim in range(nu):
                controls_plus = controls.copy()
                controls_plus[:, t, u_dim] += epsilon

                controls_minus = controls.copy()
                controls_minus[:, t, u_dim] -= epsilon

                traj_plus = self.dynamics_wrapper.rollout(state, controls_plus)
                traj_minus = self.dynamics_wrapper.rollout(state, controls_minus)

                cost_plus = self.cost_function.compute_cost(
                    traj_plus, controls_plus, reference_trajectory
                )
                cost_minus = self.cost_function.compute_cost(
                    traj_minus, controls_minus, reference_trajectory
                )

                grad_costs[:, t, u_dim] = (cost_plus - cost_minus) / (
                    2 * epsilon
                )

        return grad_costs

    def _compute_rbf_kernel(
        self, particles: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        RBF 커널 행렬 계산 (편의 메서드)

        Args:
            particles: (M, N, nu) 파티클 집합

        Returns:
            kernel: (M, M) RBF 커널 행렬
            bandwidth: float median heuristic bandwidth
        """
        return rbf_kernel_with_bandwidth(particles)

    def get_svg_statistics(self) -> Dict:
        """
        SVG 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_guide_cost_improvement: float 평균 guide 비용 개선
                - mean_bandwidth: float 평균 bandwidth
                - svg_stats_history: List[dict] 통계 히스토리
        """
        if len(self.svg_stats_history) == 0:
            return {
                "mean_guide_cost_improvement": 0.0,
                "mean_bandwidth": 0.0,
                "svg_stats_history": [],
            }

        improvements = [s["guide_cost_improvement"] for s in self.svg_stats_history]
        bandwidths = [s["bandwidth"] for s in self.svg_stats_history]

        return {
            "mean_guide_cost_improvement": float(np.mean(improvements)),
            "mean_bandwidth": float(np.mean(bandwidths)),
            "svg_stats_history": self.svg_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 및 warm start 파티클 초기화"""
        super().reset()
        self._warm_particles = None
        self.svg_stats_history = []

    def __repr__(self) -> str:
        return (
            f"SVGMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"G={self.G}, K={self.params.K}, "
            f"n_svgd_steps={self.n_svgd_steps}, "
            f"blend_ratio={self.blend_ratio})"
        )
