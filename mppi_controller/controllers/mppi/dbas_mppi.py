"""
DBaS-MPPI Controller (Discrete Barrier States MPPI)

상태를 barrier state beta(x)로 증강하고, barrier 비용에 비례하여
탐색 노이즈를 적응적 스케일링. 밀집 장애물 환경에서 가중치 퇴화 방지.

Reference: arXiv:2502.14387

핵심 수식:
    h_i(x) = ||p - p_obs||^2 - (r + margin)^2       (원형 장애물)
    h_j(x) = direction * (x[axis] - value)            (벽 제약)
    B(h) = -log(max(h, h_min))                        (log barrier)
    beta(x_{k+1}) = B(h(x_{k+1})) - gamma*(B(h(x_d)) - beta(x_k))
    C_B = RB * sum_t sum_c max(beta_tc, 0)            (barrier 비용)
    Se = mu * log(e + C_B(best))                       (적응적 탐색)
    sigma_eff = sigma * (1 + Se)                       (스케일된 노이즈)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import DBaSMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class DBaSMPPIController(MPPIController):
    """
    DBaS-MPPI Controller (Discrete Barrier States MPPI)

    Barrier state 증강으로 장애물 정보를 상태에 내재화하고,
    적응적 탐색 노이즈로 밀집 장애물에서도 안전한 궤적 생성.

    Vanilla MPPI 대비 핵심 차이:
        1. Log barrier: h(x) -> B(h) = -log(max(h, h_min))
        2. Barrier state dynamics: gamma 항으로 레퍼런스 근처 안정화
        3. Barrier 비용: 기존 비용에 C_B 추가
        4. 적응적 탐색: best 궤적의 barrier cost로 sigma 스케일링

    Args:
        model: RobotModel 인스턴스
        params: DBaSMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: DBaSMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.dbas_params = params

        # 적응적 탐색 스케일 (누적)
        self._adaptive_scale = 0.0

        # 통계 히스토리
        self._dbas_history = []

        # 장애물/벽 캐시
        self._obstacles = list(params.dbas_obstacles)
        self._walls = list(params.dbas_walls)

    def _compute_constraint_values(self, positions: np.ndarray) -> np.ndarray:
        """
        모든 제약 조건의 h(x) 값을 일괄 계산

        Args:
            positions: (..., 2) 위치 배열

        Returns:
            h: (..., num_constraints) 제약값 (양수=안전, 음수=위반)
        """
        orig_shape = positions.shape[:-1]
        margin = self.dbas_params.safety_margin

        constraints = []

        # 원형 장애물: h = ||p - p_obs||^2 - (r + margin)^2
        for ox, oy, r in self._obstacles:
            obs_pos = np.array([ox, oy])
            diff = positions - obs_pos
            dist_sq = np.sum(diff ** 2, axis=-1)
            h = dist_sq - (r + margin) ** 2
            constraints.append(h)

        # 벽 제약: h = direction * (x[axis] - value)
        for wall in self._walls:
            axis_name, value, direction = wall
            axis_idx = 0 if axis_name == 'x' else 1
            h = direction * (positions[..., axis_idx] - value)
            constraints.append(h)

        if len(constraints) == 0:
            return np.zeros(orig_shape + (0,))

        return np.stack(constraints, axis=-1)

    def _barrier_function(self, h: np.ndarray) -> np.ndarray:
        """
        Log barrier 함수: B(h) = -log(max(h, h_min))

        Args:
            h: 제약값 (양수=안전)

        Returns:
            B: barrier 값 (작을수록 안전)
        """
        h_clipped = np.maximum(h, self.dbas_params.h_min)
        return -np.log(h_clipped)

    def _compute_barrier_cost(
        self,
        trajectories: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Barrier state dynamics 전파 + barrier 비용 계산

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            reference_trajectory: (N+1, nx) 레퍼런스

        Returns:
            barrier_costs: (K,) 각 샘플의 barrier 비용
            barrier_states: (K, N+1, C) barrier state 히스토리
        """
        K, N_plus_1, nx = trajectories.shape
        N = N_plus_1 - 1
        gamma = self.dbas_params.barrier_gamma
        RB = self.dbas_params.barrier_weight

        # 모든 위치에서 제약값 계산
        positions = trajectories[:, :, :2]  # (K, N+1, 2)
        h_all = self._compute_constraint_values(positions)  # (K, N+1, C)
        C = h_all.shape[-1]

        if C == 0:
            return np.zeros(K), np.zeros((K, N_plus_1, 0))

        # Barrier 함수 계산
        B_all = self._barrier_function(h_all)  # (K, N+1, C)

        # 레퍼런스의 barrier 값
        ref_positions = reference_trajectory[:, :2]  # (N+1, 2)
        h_ref = self._compute_constraint_values(ref_positions)  # (N+1, C)
        B_ref = self._barrier_function(h_ref)  # (N+1, C)

        # Barrier state dynamics 전파
        # beta(x_{t+1}) = B(h(x_{t+1})) - gamma * (B(h(x_d_t)) - beta(x_t))
        barrier_states = np.zeros((K, N_plus_1, C))
        barrier_states[:, 0, :] = B_all[:, 0, :]  # 초기 barrier state

        for t in range(N):
            ref_t = min(t, len(B_ref) - 1)
            barrier_states[:, t + 1, :] = (
                B_all[:, t + 1, :]
                - gamma * (B_ref[ref_t, :] - barrier_states[:, t, :])
            )

        # Barrier 비용: C_B = RB * sum_t sum_c max(beta_tc, 0)
        positive_beta = np.maximum(barrier_states[:, 1:, :], 0.0)  # 초기 상태 제외
        barrier_costs = RB * np.sum(positive_beta, axis=(1, 2))  # (K,)

        return barrier_costs, barrier_states

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        DBaS-MPPI 제어 계산

        1. sigma_eff = sigma * (1 + Se) (적응적)
        2. 노이즈 샘플링 + rollout
        3. base_cost + barrier_cost
        4. MPPI 가중치 + 업데이트
        5. Se 갱신 (best 궤적의 barrier cost)
        6. Receding horizon shift

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K
        N = self.params.N

        # 1. 적응적 탐색 노이즈 스케일링
        if self.dbas_params.use_adaptive_exploration:
            sigma_scale = 1.0 + self._adaptive_scale
        else:
            sigma_scale = 1.0

        sigma_eff = self.params.sigma * sigma_scale

        # 2. 스케일된 노이즈 샘플링
        noise = np.random.standard_normal(
            (K, N, self.model.control_dim)
        ) * sigma_eff[None, None, :]

        # 3. 샘플 제어 시퀀스
        sampled_controls = self.U[None, :, :] + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 4. Rollout
        trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 5. 기본 비용 계산
        base_costs = self.cost_function.compute_cost(
            trajectories, sampled_controls, reference_trajectory
        )

        # 6. Barrier 비용 계산
        barrier_costs, barrier_states = self._compute_barrier_cost(
            trajectories, reference_trajectory
        )

        # 7. 총 비용
        total_costs = base_costs + barrier_costs

        # 8. MPPI 가중치 계산
        weights = self._compute_weights(total_costs, self.params.lambda_)

        # 9. 가중 평균으로 제어 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 10. 첫 제어 추출
        optimal_control = self.U[0].copy()

        # 11. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 12. 적응적 탐색 스케일 갱신 (best 궤적의 barrier cost)
        best_idx = np.argmin(total_costs)
        best_barrier_cost = barrier_costs[best_idx]

        if self.dbas_params.use_adaptive_exploration:
            mu = self.dbas_params.exploration_coeff
            self._adaptive_scale = mu * np.log(np.e + best_barrier_cost)

        # 13. 제약 최소값 (안전성 지표)
        positions = trajectories[best_idx, :, :2]  # (N+1, 2)
        if len(self._obstacles) > 0 or len(self._walls) > 0:
            h_best = self._compute_constraint_values(positions)
            min_constraint = float(np.min(h_best)) if h_best.size > 0 else float('inf')
        else:
            min_constraint = float('inf')

        # 14. ESS 및 통계
        ess = self._compute_ess(weights)

        dbas_stats = {
            "adaptive_scale": float(self._adaptive_scale),
            "sigma_eff": sigma_eff.tolist(),
            "barrier_cost_mean": float(np.mean(barrier_costs)),
            "barrier_cost_best": float(best_barrier_cost),
            "min_constraint": min_constraint,
            "num_obstacles": len(self._obstacles),
            "num_walls": len(self._walls),
        }
        self._dbas_history.append(dbas_stats)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx],
            "best_cost": float(total_costs[best_idx]),
            "mean_cost": float(np.mean(total_costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "dbas_stats": dbas_stats,
            "barrier_costs": barrier_costs,
            "barrier_states": barrier_states,
        }
        self.last_info = info

        return optimal_control, info

    def update_obstacles(self, obstacles: List[tuple]):
        """
        동적 장애물 실시간 업데이트

        Args:
            obstacles: [(x, y, radius), ...] 새 장애물 목록
        """
        self._obstacles = list(obstacles)

    def update_walls(self, walls: List[tuple]):
        """
        벽 제약 실시간 업데이트

        Args:
            walls: [('x'|'y', value, direction), ...] 새 벽 목록
        """
        self._walls = list(walls)

    def get_dbas_statistics(self) -> Dict:
        """
        누적 DBaS 통계 반환

        Returns:
            dict:
                - history: 전체 스텝별 통계 리스트
                - mean_adaptive_scale: 평균 적응 스케일
                - mean_barrier_cost: 평균 barrier 비용
                - min_constraint_ever: 전체 최소 제약값
        """
        if len(self._dbas_history) == 0:
            return {
                "history": [],
                "mean_adaptive_scale": 0.0,
                "mean_barrier_cost": 0.0,
                "min_constraint_ever": float('inf'),
            }

        scales = [s["adaptive_scale"] for s in self._dbas_history]
        costs = [s["barrier_cost_best"] for s in self._dbas_history]
        min_constraints = [s["min_constraint"] for s in self._dbas_history]

        return {
            "history": self._dbas_history.copy(),
            "mean_adaptive_scale": float(np.mean(scales)),
            "mean_barrier_cost": float(np.mean(costs)),
            "min_constraint_ever": float(np.min(min_constraints)),
        }

    def reset(self):
        """제어 시퀀스 + adaptive_scale + history 초기화"""
        super().reset()
        self._adaptive_scale = 0.0
        self._dbas_history = []
        self._obstacles = list(self.dbas_params.dbas_obstacles)
        self._walls = list(self.dbas_params.dbas_walls)

    def __repr__(self) -> str:
        return (
            f"DBaSMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"obstacles={len(self._obstacles)}, "
            f"walls={len(self._walls)}, "
            f"barrier_weight={self.dbas_params.barrier_weight}, "
            f"gamma={self.dbas_params.barrier_gamma}, "
            f"K={self.params.K})"
        )
