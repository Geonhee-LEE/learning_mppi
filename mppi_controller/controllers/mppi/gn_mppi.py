"""
GN-MPPI Controller (Gauss-Newton Accelerated MPPI)

표준 MPPI의 1차 가중 평균 업데이트를 2차 가우스-뉴턴 스텝으로 대체.
기존 K개 샘플의 비용+노이즈 정보로 야코비안을 복원하고,
GGN 헤시안으로 곡률 방향 최적화.

핵심:
  - 가우시안 스무딩으로 비용 기울기 추정: ∇J ≈ E[C·ε] / σ²
  - GGN 대각 헤시안: H ≈ E[C²·ε²] / σ⁴ + reg
  - 뉴턴 스텝: δU = -H^{-1}·∇J
  - 병렬 라인 서치로 최적 스텝 크기 선택
  - 표준 MPPI 업데이트와 GN 업데이트 중 더 좋은 것 선택

Reference: Homburger et al., arXiv:2512.04579
"""

import numpy as np
from typing import Dict, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import GNMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class GNMPPIController(MPPIController):
    """
    GN-MPPI Controller (26번째 MPPI 변형)

    가우스-뉴턴 가속 MPPI — 2차 최적화로 MPPI 수렴 가속.

    Vanilla MPPI 대비 핵심 차이:
        1. 다중 반복: DIAL/CMA처럼 n_iters 반복 sample-evaluate-update
        2. 2차 업데이트: 가우시안 스무딩 기울기 + GGN 헤시안으로 뉴턴 스텝
        3. 병렬 라인 서치: 여러 스텝 크기 후보 중 최적 선택
        4. 표준 MPPI 폴백: GN 스텝이 열등하면 표준 MPPI 업데이트 사용
        5. Cold/Warm start 구분: 첫 호출 5회 vs 이후 3회

    핵심 수식:
        ∇J ≈ (1/K) Σ_k C_k·ε_k / σ²          (가우시안 스무딩 기울기)
        H_diag ≈ (1/K) Σ_k C_k²·ε_k² / σ⁴ + λI (GGN 대각 헤시안)
        δU = -H^{-1}·∇J                          (뉴턴 스텝)
        U_{k+1} = U_k + α·δU   (α = line search)

    Args:
        model: RobotModel 인스턴스
        params: GNMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: GNMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.gn_params = params

        # 첫 호출 여부 추적
        self._is_first_call = True

        # 통계 추적
        self._iteration_costs = []
        self._gn_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        GN-MPPI 제어 계산

        다중 반복 + 가우스-뉴턴 2차 업데이트로 제어 시퀀스를 최적화.

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

        # Cold start vs Warm start
        if self._is_first_call:
            n_iters = self.gn_params.n_gn_iters_init
            self._is_first_call = False
        else:
            n_iters = self.gn_params.n_gn_iters

        iteration_costs = []
        last_trajectories = None
        last_weights = None
        last_costs = None
        gn_used_count = 0

        for i in range(n_iters):
            # 1. 노이즈 샘플링 (K, N, nu)
            noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

            # 2. 샘플 제어 시퀀스 (K, N, nu)
            sampled_controls = self.U[None, :, :] + noise
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

            # 3. Rollout + 비용
            trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
            costs = self.cost_function.compute_cost(
                trajectories, sampled_controls, reference_trajectory
            )

            if self.gn_params.use_gn_update:
                # --- 가우스-뉴턴 업데이트 ---
                gn_update = self._compute_gn_step(noise, costs, K, N, nu)

                # 병렬 라인 서치
                best_gn_cost, best_gn_update = self._line_search(
                    state, reference_trajectory, gn_update, N, nu,
                )

                # 표준 MPPI 업데이트 계산
                if self.gn_params.use_reward_normalization:
                    weights_std = self._compute_weights_normalized(costs)
                else:
                    weights_std = self._compute_weights(costs, self.params.lambda_)
                mppi_update = np.sum(
                    weights_std[:, None, None] * noise, axis=0
                )  # (N, nu)

                # MPPI 업데이트 비용 평가
                U_mppi_candidate = self.U + mppi_update
                if self.u_min is not None and self.u_max is not None:
                    U_mppi_candidate = np.clip(
                        U_mppi_candidate, self.u_min, self.u_max
                    )
                mppi_traj = self.dynamics_wrapper.rollout(
                    state, U_mppi_candidate[None, :, :]
                )
                mppi_cost = self.cost_function.compute_cost(
                    mppi_traj, U_mppi_candidate[None, :, :], reference_trajectory
                )[0]

                # GN vs MPPI 비교: 더 좋은 업데이트 선택
                if best_gn_cost < mppi_cost:
                    self.U = self.U + best_gn_update.reshape(N, nu)
                    gn_used_count += 1
                else:
                    self.U = self.U + mppi_update
            else:
                # 표준 MPPI 폴백
                if self.gn_params.use_reward_normalization:
                    weights = self._compute_weights_normalized(costs)
                else:
                    weights = self._compute_weights(costs, self.params.lambda_)
                weighted_noise = np.sum(
                    weights[:, None, None] * noise, axis=0
                )
                self.U = self.U + weighted_noise

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                self.U = np.clip(self.U, self.u_min, self.u_max)

            # 반복별 비용 기록
            iteration_costs.append(float(np.min(costs)))

            last_trajectories = trajectories
            last_costs = costs

        # 최종 가중치 계산 (info 용)
        if self.gn_params.use_reward_normalization:
            last_weights = self._compute_weights_normalized(last_costs)
        else:
            last_weights = self._compute_weights(last_costs, self.params.lambda_)

        # 첫 제어 추출
        optimal_control = self.U[0].copy()

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 통계 저장
        self._iteration_costs = iteration_costs
        ess = self._compute_ess(last_weights)
        best_idx = np.argmin(last_costs)

        gn_stats = {
            "n_iters": n_iters,
            "iteration_costs": iteration_costs,
            "cost_improvement": (
                iteration_costs[0] - iteration_costs[-1]
                if len(iteration_costs) > 1
                else 0.0
            ),
            "used_gn": self.gn_params.use_gn_update,
            "gn_used_count": gn_used_count,
            "gn_used_ratio": (
                gn_used_count / n_iters if n_iters > 0 else 0.0
            ),
        }
        self._gn_history.append(gn_stats)

        info = {
            "sample_trajectories": last_trajectories,
            "sample_weights": last_weights,
            "best_trajectory": last_trajectories[best_idx],
            "best_cost": float(last_costs[best_idx]),
            "mean_cost": float(np.mean(last_costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "gn_stats": gn_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _compute_gn_step(
        self,
        noise: np.ndarray,
        costs: np.ndarray,
        K: int,
        N: int,
        nu: int,
    ) -> np.ndarray:
        """
        가우스-뉴턴 스텝 계산

        가우시안 스무딩으로 기울기 복원 + GGN 대각 헤시안으로 뉴턴 방향 계산.

        Args:
            noise: (K, N, nu) 노이즈 샘플
            costs: (K,) 비용 배열
            K: 샘플 수
            N: 호라이즌
            nu: 제어 차원

        Returns:
            step: (N*nu,) 뉴턴 스텝 방향
        """
        # noise를 평탄화: (K, N*nu)
        noise_flat = noise.reshape(K, -1)

        # sigma 벡터 확장: (N*nu,)
        sigma_flat = np.tile(self.params.sigma, N)
        sigma_sq = sigma_flat ** 2

        # 비용 중심화 (분산 감소)
        cost_centered = costs - np.mean(costs)

        # 가우시안 스무딩 기울기: ∇J ≈ E[C·ε] / σ²
        gradient = (
            np.mean(cost_centered[:, None] * noise_flat, axis=0)
            / (sigma_sq + 1e-10)
        )

        # GGN 대각 헤시안: H ≈ E[C²·ε²] / σ⁴ + reg
        hessian_diag = (
            np.mean(cost_centered[:, None] ** 2 * noise_flat ** 2, axis=0)
            / (sigma_sq ** 2 + 1e-10)
        )
        hessian_diag += self.gn_params.regularization

        # 뉴턴 스텝: δU = -H^{-1} · ∇J
        step = -gradient / (hessian_diag + 1e-10)

        return step

    def _line_search(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
        step: np.ndarray,
        N: int,
        nu: int,
    ) -> Tuple[float, np.ndarray]:
        """
        병렬 라인 서치 — 여러 스텝 크기 중 최적 선택

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx)
            step: (N*nu,) GN 스텝 방향
            N: 호라이즌
            nu: 제어 차원

        Returns:
            best_cost: 최적 후보의 비용
            best_update: 최적 업데이트 벡터 (N*nu,)
        """
        alphas = [
            self.gn_params.gn_step_size * (self.gn_params.line_search_decay ** j)
            for j in range(self.gn_params.line_search_steps)
        ]

        best_cost = float("inf")
        best_update = np.zeros_like(step)

        for alpha in alphas:
            candidate_update = alpha * step
            U_candidate = self.U + candidate_update.reshape(N, nu)

            if self.u_min is not None and self.u_max is not None:
                U_candidate = np.clip(U_candidate, self.u_min, self.u_max)

            # 후보 평가: 단일 rollout
            cand_traj = self.dynamics_wrapper.rollout(
                state, U_candidate[None, :, :]
            )
            cand_cost = self.cost_function.compute_cost(
                cand_traj, U_candidate[None, :, :], reference_trajectory
            )[0]

            if cand_cost < best_cost:
                best_cost = cand_cost
                best_update = candidate_update.copy()

        return best_cost, best_update

    def _compute_weights_normalized(self, costs: np.ndarray) -> np.ndarray:
        """
        보상 정규화 기반 가중치 계산 (DIAL/CMA와 동일)

        rewards = -costs
        normalized = (rewards - mean) / (std + eps)
        weights = softmax(normalized / lambda)
        """
        rewards = -costs
        std = np.std(rewards)
        if std < 1e-10:
            return np.ones(len(costs)) / len(costs)

        normalized = (rewards - np.mean(rewards)) / (std + 1e-10)
        scaled = normalized / self.params.lambda_

        # 수치 안정성을 위한 max-shift
        scaled -= np.max(scaled)
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / np.sum(exp_scaled)

        return weights

    def get_gn_statistics(self) -> Dict:
        """
        GN-MPPI 통계 반환

        Returns:
            dict:
                - mean_cost_improvement: 평균 반복 비용 개선
                - mean_n_iters: 평균 반복 횟수
                - mean_gn_used_ratio: GN 스텝 사용 비율
                - last_iteration_costs: 마지막 호출의 반복별 비용
                - gn_history: 전체 통계 히스토리
        """
        if len(self._gn_history) == 0:
            return {
                "mean_cost_improvement": 0.0,
                "mean_n_iters": 0.0,
                "mean_gn_used_ratio": 0.0,
                "last_iteration_costs": [],
                "gn_history": [],
            }

        improvements = [s["cost_improvement"] for s in self._gn_history]
        n_iters_list = [s["n_iters"] for s in self._gn_history]
        gn_ratios = [s["gn_used_ratio"] for s in self._gn_history]

        return {
            "mean_cost_improvement": float(np.mean(improvements)),
            "mean_n_iters": float(np.mean(n_iters_list)),
            "mean_gn_used_ratio": float(np.mean(gn_ratios)),
            "last_iteration_costs": self._iteration_costs,
            "gn_history": self._gn_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 초기화 및 첫 호출 상태 복원"""
        super().reset()
        self._is_first_call = True
        self._iteration_costs = []
        self._gn_history = []

    def __repr__(self) -> str:
        return (
            f"GNMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"n_gn_iters_init={self.gn_params.n_gn_iters_init}, "
            f"n_gn_iters={self.gn_params.n_gn_iters}, "
            f"line_search_steps={self.gn_params.line_search_steps}, "
            f"K={self.params.K})"
        )
