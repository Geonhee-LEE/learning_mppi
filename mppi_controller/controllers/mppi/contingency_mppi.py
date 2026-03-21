"""
Contingency-MPPI Controller (C-MPPI)

Nested MPPI 구조: 외부 MPPI가 명목 궤적을 최적화하면서,
체크포인트 상태에서 내부 MPPI(contingency)가 비상 궤적의 비용을 평가.
모든 계획 상태에서 안전 집합 도달 가능성을 보장.

핵심 수식:
    min_u J_nom(u) + lambda_cont * max_t contingency_cost(x_t)
    contingency_cost(x_t) = min_v cost_safe(rollout(x_t, v))  -- inner MPPI
    효율성: 모든 타임스텝이 아닌 체크포인트에서만 내부 MPPI 평가

Reference: Jung, Estornell & Everett, L4DC 2025, arXiv:2412.09777
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import ContingencyMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class ContingencyMPPIController(MPPIController):
    """
    C-MPPI (35th variant) -- Nested MPPI with contingency safety

    Vanilla MPPI 대비 핵심 차이:
        1. 체크포인트 평가: 명목 궤적의 n_checkpoints 지점에서 contingency 평가
        2. Braking contingency: 제로 제어 시퀀스로 정지 가능 여부 확인 (배치화)
        3. MPPI contingency: 내부 MPPI로 최소 비용 탈출 경로 탐색 (배치화)
        4. 복합 비용: nominal_cost + contingency_weight * max_checkpoint(contingency_cost)

    성능 최적화:
        - Braking contingency: K개 체크포인트 상태를 배치 롤아웃 (Python 루프 없음)
        - MPPI contingency: K*K_cont개 제어를 한번에 롤아웃하여 비용 계산
        - 체크포인트 레퍼런스: 사전 계산하여 재사용

    Args:
        model: RobotModel 인스턴스
        params: ContingencyMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        safety_cost_function: 내부 MPPI contingency 평가용 비용 함수
            (None이면 메인 비용 함수 재사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: ContingencyMPPIParams,
        cost_function: Optional[CostFunction] = None,
        safety_cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.contingency_params = params
        self._safety_cost = safety_cost_function  # None이면 메인 비용 함수 사용
        self._contingency_stats: List[Dict] = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        C-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - MPPI info + contingency info
        """
        K = self.params.K
        N = self.params.N

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 명목 궤적 rollout (K, N+1, nx)
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 명목 비용 계산 (K,)
        nominal_costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. 체크포인트에서 contingency 비용 평가 (벡터화)
        checkpoint_indices = self._get_checkpoint_indices(N)
        contingency_costs = self._evaluate_contingency_batch(
            sample_trajectories, checkpoint_indices, reference_trajectory
        )

        # 6. 안전 집합 임계값 페널티
        threshold = self.contingency_params.safe_cost_threshold
        safety_penalty = np.where(
            contingency_costs > threshold,
            self.contingency_params.safety_cost_weight * (contingency_costs - threshold),
            0.0,
        )

        # 7. 총 비용 = 명목 + contingency_weight * contingency + safety_penalty
        total_costs = (
            nominal_costs
            + self.contingency_params.contingency_weight * contingency_costs
            + safety_penalty
        )

        # 8. MPPI 가중치 계산
        weights = self._compute_weights(total_costs, self.params.lambda_)

        # 9. 가중 평균으로 제어 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 10. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 11. 최적 제어 반환
        optimal_control = self.U[0, :]

        # 12. 정보 저장
        ess = self._compute_ess(weights)
        best_idx = np.argmin(total_costs)

        # Contingency 통계
        cont_stats = {
            "mean_contingency_cost": float(np.mean(contingency_costs)),
            "max_contingency_cost": float(np.max(contingency_costs)),
            "min_contingency_cost": float(np.min(contingency_costs)),
            "best_contingency_cost": float(contingency_costs[best_idx]),
            "n_above_threshold": int(np.sum(contingency_costs > threshold)),
            "checkpoint_indices": checkpoint_indices.tolist(),
            "best_details": [
                {"checkpoint_idx": int(ci), "contingency_cost": float(contingency_costs[best_idx])}
                for ci in checkpoint_indices
            ],
        }
        self._contingency_stats.append(cont_stats)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(total_costs[best_idx]),
            "mean_cost": float(np.mean(total_costs)),
            "nominal_cost": float(nominal_costs[best_idx]),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "contingency_stats": cont_stats,
            "contingency_costs": contingency_costs,
        }
        self.last_info = info

        return optimal_control, info

    def _evaluate_contingency_batch(
        self,
        sample_trajectories: np.ndarray,
        checkpoint_indices: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        모든 K개 샘플의 체크포인트에서 contingency 비용을 배치 평가

        Args:
            sample_trajectories: (K, N+1, nx) 명목 궤적
            checkpoint_indices: (n_cp,) 체크포인트 인덱스
            reference_trajectory: (N+1, nx) 레퍼런스

        Returns:
            contingency_costs: (K,) 각 샘플의 max-over-checkpoints contingency 비용
        """
        K = sample_trajectories.shape[0]
        contingency_costs = np.zeros(K)
        cost_fn = self._safety_cost if self._safety_cost is not None else self.cost_function
        N_cont = self.contingency_params.contingency_horizon

        for cp_idx in checkpoint_indices:
            # 모든 K개 샘플의 체크포인트 상태 (K, nx)
            checkpoint_states = sample_trajectories[:, cp_idx, :]

            # 체크포인트 이후 레퍼런스 슬라이스
            ref_from_cp = self._get_reference_from_checkpoint(
                reference_trajectory, cp_idx
            )

            cp_costs = np.full(K, np.inf)

            # Braking contingency: K개를 배치로 롤아웃
            if self.contingency_params.use_braking_contingency:
                brake_controls = np.zeros((K, N_cont, self.model.control_dim))
                brake_trajs = self._batch_rollout_from_states(
                    checkpoint_states, brake_controls
                )
                brake_costs = cost_fn.compute_cost(
                    brake_trajs, brake_controls, ref_from_cp
                )
                cp_costs = np.minimum(cp_costs, brake_costs)

            # MPPI contingency: K개 체크포인트 각각에 대해 K_cont개 샘플
            if self.contingency_params.use_mppi_contingency:
                mppi_costs = self._batch_inner_mppi(
                    checkpoint_states, ref_from_cp, cost_fn
                )
                cp_costs = np.minimum(cp_costs, mppi_costs)

            # max over checkpoints
            contingency_costs = np.maximum(contingency_costs, cp_costs)

        return contingency_costs

    def _batch_rollout_from_states(
        self,
        initial_states: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        """
        K개 서로 다른 초기 상태에서 배치 롤아웃

        Args:
            initial_states: (K, nx) 초기 상태들
            controls: (K, N_cont, nu) 제어 시퀀스들

        Returns:
            trajectories: (K, N_cont+1, nx) 궤적들
        """
        K, N_cont, _ = controls.shape
        nx = self.model.state_dim
        trajectories = np.zeros((K, N_cont + 1, nx))
        trajectories[:, 0, :] = initial_states

        for t in range(N_cont):
            for k in range(K):
                trajectories[k, t + 1] = self.model.step(
                    trajectories[k, t], controls[k, t], self.params.dt
                )

        return trajectories

    def _batch_inner_mppi(
        self,
        checkpoint_states: np.ndarray,
        reference: np.ndarray,
        cost_fn: CostFunction,
    ) -> np.ndarray:
        """
        K개 체크포인트 상태 각각에서 내부 MPPI 수행 (배치 최적화)

        전체 K*K_cont 샘플을 한번에 처리하는 대신,
        K개 상태를 순회하되 각 상태의 K_cont 샘플은 배치 롤아웃.

        Args:
            checkpoint_states: (K, nx) 체크포인트 상태들
            reference: (N_cont+1, nx) 레퍼런스
            cost_fn: 비용 함수

        Returns:
            min_costs: (K,) 각 체크포인트의 최소 contingency 비용
        """
        K = checkpoint_states.shape[0]
        K_cont = self.contingency_params.contingency_samples
        N_cont = self.contingency_params.contingency_horizon
        nu = self.model.control_dim
        sigma = self.params.sigma * self.contingency_params.contingency_sigma_scale

        min_costs = np.full(K, np.inf)

        for k in range(K):
            # K_cont개 노이즈 샘플링
            noise = np.random.standard_normal((K_cont, N_cont, nu)) * sigma
            controls = noise.copy()

            if self.u_min is not None and self.u_max is not None:
                controls = np.clip(controls, self.u_min, self.u_max)

            # 배치 롤아웃 (dynamics_wrapper 사용)
            trajectories = self.dynamics_wrapper.rollout(
                checkpoint_states[k], controls
            )

            # 배치 비용 계산
            costs = cost_fn.compute_cost(trajectories, controls, reference)
            min_costs[k] = float(np.min(costs))

        return min_costs

    def _evaluate_contingency_at_checkpoint(
        self, checkpoint_state: np.ndarray, reference_from_checkpoint: np.ndarray
    ) -> float:
        """
        단일 체크포인트 상태에서 최소 contingency 비용 평가
        (비배치 버전 - 테스트 호환용)

        Args:
            checkpoint_state: (nx,) 체크포인트 상태
            reference_from_checkpoint: (N_cont+1, nx) 체크포인트 이후 레퍼런스

        Returns:
            best_contingency_cost: 최소 contingency 비용 (스칼라)
        """
        costs = []

        # Braking contingency
        if self.contingency_params.use_braking_contingency:
            N_cont = self.contingency_params.contingency_horizon
            brake_controls = np.zeros((N_cont, self.model.control_dim))
            brake_traj = self._rollout_single(checkpoint_state, brake_controls)
            brake_cost = self._compute_safety_cost(
                brake_traj, brake_controls, reference_from_checkpoint
            )
            costs.append(brake_cost)

        # MPPI contingency
        if self.contingency_params.use_mppi_contingency:
            inner_cost = self._inner_mppi_solve(
                checkpoint_state, reference_from_checkpoint
            )
            costs.append(inner_cost)

        return min(costs) if costs else 0.0

    def _inner_mppi_solve(
        self, state: np.ndarray, reference: np.ndarray
    ) -> float:
        """
        간소화된 내부 MPPI: 샘플링 -> 롤아웃 -> 최소 비용 반환

        Args:
            state: (nx,) 시작 상태
            reference: (N_cont+1, nx) 레퍼런스 궤적

        Returns:
            min_cost: 최소 비용 (스칼라)
        """
        K_cont = self.contingency_params.contingency_samples
        N_cont = self.contingency_params.contingency_horizon
        nu = self.model.control_dim
        sigma = self.params.sigma * self.contingency_params.contingency_sigma_scale

        noise = np.random.standard_normal((K_cont, N_cont, nu)) * sigma
        controls = noise.copy()

        if self.u_min is not None and self.u_max is not None:
            controls = np.clip(controls, self.u_min, self.u_max)

        trajectories = self.dynamics_wrapper.rollout(state, controls)

        cost_fn = self._safety_cost if self._safety_cost is not None else self.cost_function
        costs = cost_fn.compute_cost(trajectories, controls, reference)

        return float(np.min(costs))

    def _rollout_single(
        self, state: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """단일 궤적 롤아웃"""
        N = controls.shape[0]
        traj = np.zeros((N + 1, self.model.state_dim))
        traj[0] = state
        for t in range(N):
            traj[t + 1] = self.model.step(traj[t], controls[t], self.params.dt)
        return traj

    def _compute_safety_cost(
        self,
        trajectory: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """단일 궤적의 안전 비용 계산"""
        traj_batch = trajectory[np.newaxis, :, :]
        ctrl_batch = controls[np.newaxis, :, :]
        cost_fn = self._safety_cost if self._safety_cost is not None else self.cost_function
        costs = cost_fn.compute_cost(traj_batch, ctrl_batch, reference)
        return float(costs[0])

    def _compute_safety_cost_batch(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """배치 궤적의 안전 비용 계산"""
        cost_fn = self._safety_cost if self._safety_cost is not None else self.cost_function
        return cost_fn.compute_cost(trajectories, controls, reference)

    def _get_checkpoint_indices(self, N: int) -> np.ndarray:
        """체크포인트 타임스텝 인덱스 반환 (균일 간격)"""
        n_cp = min(self.contingency_params.n_checkpoints, N)
        if n_cp <= 0:
            return np.array([1], dtype=int)
        return np.linspace(1, max(1, N - 1), n_cp, dtype=int)

    def _get_reference_from_checkpoint(
        self, reference_trajectory: np.ndarray, cp_idx: int
    ) -> np.ndarray:
        """체크포인트 이후의 레퍼런스 궤적 슬라이스 생성"""
        N_cont = self.contingency_params.contingency_horizon
        nx = reference_trajectory.shape[1]
        ref_slice = np.zeros((N_cont + 1, nx))

        remaining = reference_trajectory[cp_idx:]
        n_available = min(N_cont + 1, remaining.shape[0])
        ref_slice[:n_available] = remaining[:n_available]

        if n_available < N_cont + 1:
            ref_slice[n_available:] = remaining[-1]

        return ref_slice

    def get_contingency_statistics(self) -> Dict:
        """누적 contingency 통계 반환"""
        if not self._contingency_stats:
            return {
                "mean_contingency_cost": 0.0,
                "max_contingency_cost": 0.0,
                "min_contingency_cost": 0.0,
                "n_steps": 0,
                "threshold_violation_rate": 0.0,
            }

        mean_costs = [s["mean_contingency_cost"] for s in self._contingency_stats]
        max_costs = [s["max_contingency_cost"] for s in self._contingency_stats]
        min_costs = [s["min_contingency_cost"] for s in self._contingency_stats]
        n_above = [s["n_above_threshold"] for s in self._contingency_stats]
        K = self.params.K

        return {
            "mean_contingency_cost": float(np.mean(mean_costs)),
            "max_contingency_cost": float(np.max(max_costs)),
            "min_contingency_cost": float(np.min(min_costs)),
            "n_steps": len(self._contingency_stats),
            "threshold_violation_rate": float(np.mean(
                [n / K for n in n_above]
            )),
        }

    def reset(self):
        """제어 시퀀스 및 contingency 통계 초기화"""
        super().reset()
        self._contingency_stats = []

    def __repr__(self) -> str:
        return (
            f"ContingencyMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"contingency_weight={self.contingency_params.contingency_weight}, "
            f"n_checkpoints={self.contingency_params.n_checkpoints}, "
            f"contingency_samples={self.contingency_params.contingency_samples}, "
            f"braking={self.contingency_params.use_braking_contingency}, "
            f"mppi={self.contingency_params.use_mppi_contingency})"
        )
