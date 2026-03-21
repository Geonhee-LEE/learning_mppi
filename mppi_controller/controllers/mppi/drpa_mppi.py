"""
DRPA-MPPI Controller (Dynamic Repulsive Potential Augmented MPPI)

예측 궤적에서 local minima trap을 동적으로 감지하고,
반발 포텐셜(repulsive potential)을 자동 추가하여 반응적으로 탈출.
글로벌 경로 탐색 없이 포텐셜 필드 + 감지 로직만으로 동작.

핵심 수식:
    F_rep(x) = η · (1/d(x,o) - 1/d_0)² if d < d_0, else 0
    C_total = C_normal + α · Σ F_rep(x)  (탈출 모드 시)
    진행 정체 감지: ||x_T - x_0|| < threshold (호라이즌 끝 이동량)
    탈출 성공 판정: ||Δx|| > recovery_threshold

Reference: Fuke et al., "DRPA-MPPI: Dynamic Repulsive Potential Augmented
           MPPI for Reactive Navigation", arXiv:2503.20134, 2025
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import DRPAMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class DRPAMPPIController(MPPIController):
    """
    DRPA-MPPI Controller (28번째 MPPI 변형)

    Dynamic Repulsive Potential Augmented MPPI.
    Local minima 동적 감지 + 반발 포텐셜 자동 추가로 반응적 탈출.

    Vanilla MPPI 대비 핵심 차이:
        1. 정체 감지: 위치/비용 기반 stagnation detection
        2. 반발 포텐셜: 장애물 근처에서 밀어내는 포텐셜 필드
        3. 모드 전환: 정상 → 탈출 → 복귀 (자동)
        4. 노이즈 증폭: 탈출 모드 시 탐색 노이즈 증가

    Args:
        model: RobotModel 인스턴스
        params: DRPAMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: DRPAMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.drpa_params = params

        # 모드 상태
        self._in_escape_mode: bool = False

        # 정체 감지 히스토리 (최근 위치)
        self._stagnation_history: List[np.ndarray] = []

        # 비용 히스토리 (정체 감지용)
        self._cost_history: List[float] = []

        # 장애물 목록
        self._obstacles: List[tuple] = list(params.obstacles)

        # 통계 추적
        self._drpa_history: List[Dict] = []
        self._escape_count: int = 0
        self._total_steps: int = 0

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        DRPA-MPPI 제어 계산

        1. 정체 감지 (stagnation detection)
        2. 탈출 모드 시 노이즈 증폭
        3. Rollout + 비용 계산 (반발 포텐셜 추가)
        4. MPPI 가중 평균 업데이트
        5. 복귀 판정
        6. Receding horizon shift

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어
            info: dict 디버깅 정보
        """
        K = self.params.K
        N = self.params.N
        nu = self.model.control_dim

        # 1. 정체 감지
        stagnation_detected = self._detect_stagnation(state)
        if stagnation_detected and not self._in_escape_mode:
            self._in_escape_mode = True
            self._escape_count += 1

        # 2. 노이즈 샘플링 (탈출 모드 시 증폭)
        if self._in_escape_mode and self.drpa_params.use_noise_boost:
            sigma_eff = self.params.sigma * self.drpa_params.escape_boost
        else:
            sigma_eff = self.params.sigma

        noise = np.random.standard_normal(
            (K, N, nu)
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

        # 6. 반발 포텐셜 비용 계산
        repulsive_costs = self._compute_repulsive_potential(trajectories)

        # 7. 증강 비용: 탈출 모드이면 반발 포텐셜 추가
        if self._in_escape_mode:
            total_costs = base_costs + repulsive_costs
        else:
            total_costs = base_costs

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

        # 12. 복귀 판정
        best_idx = np.argmin(total_costs)
        best_cost = float(total_costs[best_idx])

        # 비용 히스토리 업데이트
        self._cost_history.append(best_cost)
        if len(self._cost_history) > self.drpa_params.stagnation_window * 2:
            self._cost_history = self._cost_history[
                -self.drpa_params.stagnation_window * 2:
            ]

        if self._in_escape_mode:
            self._check_recovery(state)

        # 13. 위치 히스토리 업데이트
        self._stagnation_history.append(state[:2].copy())
        if len(self._stagnation_history) > self.drpa_params.stagnation_window * 2:
            self._stagnation_history = self._stagnation_history[
                -self.drpa_params.stagnation_window * 2:
            ]

        self._total_steps += 1

        # 14. ESS 및 통계
        ess = self._compute_ess(weights)

        # 최소 장애물 간극
        min_clearance = self._compute_min_clearance(state)

        drpa_stats = {
            "in_escape_mode": self._in_escape_mode,
            "stagnation_detected": stagnation_detected,
            "escape_count": self._escape_count,
            "repulsive_cost_mean": float(np.mean(repulsive_costs)),
            "repulsive_cost_best": float(repulsive_costs[best_idx]),
            "sigma_eff": sigma_eff.tolist(),
            "min_clearance": min_clearance,
            "total_steps": self._total_steps,
        }
        self._drpa_history.append(drpa_stats)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx],
            "best_cost": best_cost,
            "mean_cost": float(np.mean(total_costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "drpa_stats": drpa_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _detect_stagnation(self, state: np.ndarray) -> bool:
        """
        Local minima 정체 감지

        조건 (OR):
        1. 위치 정체: 최근 window 스텝 이동량 < threshold
        2. 비용 정체: 비용 변화량이 작음

        Args:
            state: (nx,) 현재 상태

        Returns:
            True if stagnation detected
        """
        window = self.drpa_params.stagnation_window

        # 위치 기반 정체 감지
        if len(self._stagnation_history) >= window:
            recent = self._stagnation_history[-window:]
            displacement = np.linalg.norm(
                np.array(recent[-1]) - np.array(recent[0])
            )
            if displacement < self.drpa_params.stagnation_threshold:
                return True

        # 비용 기반 정체 감지
        if len(self._cost_history) >= window:
            recent_costs = self._cost_history[-window:]
            cost_range = max(recent_costs) - min(recent_costs)
            # 비용 변화가 매우 작으면 정체
            if cost_range < self.drpa_params.stagnation_threshold * 0.1:
                return True

        return False

    def _compute_repulsive_potential(
        self, trajectories: np.ndarray
    ) -> np.ndarray:
        """
        반발 포텐셜 비용 계산

        F_rep(x) = η · (1/d(x,o) - 1/d_0)² if d < d_0, else 0

        벡터화된 배치 연산으로 전체 궤적에 대해 일괄 계산.

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적

        Returns:
            repulsive_costs: (K,) 각 샘플의 반발 포텐셜 비용 합
        """
        K = trajectories.shape[0]
        eta = self.drpa_params.repulsive_gain
        d0 = self.drpa_params.influence_distance

        if len(self._obstacles) == 0 or d0 <= 0:
            return np.zeros(K)

        positions = trajectories[:, :, :2]  # (K, N+1, 2)
        total_potential = np.zeros(K)

        for ox, oy, r in self._obstacles:
            obs_pos = np.array([ox, oy])
            diff = positions - obs_pos  # (K, N+1, 2)
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (K, N+1)

            # 표면까지 거리 (반지름 차감)
            surface_dist = dist - r  # (K, N+1)
            surface_dist = np.maximum(surface_dist, 1e-6)  # 0 나누기 방지

            # 영향 범위 내에서만 포텐셜 적용
            in_range = surface_dist < d0  # (K, N+1)

            # F_rep = η · (1/d - 1/d_0)² where d < d_0
            potential = np.zeros_like(surface_dist)
            potential[in_range] = eta * (
                1.0 / surface_dist[in_range] - 1.0 / d0
            ) ** 2

            # 궤적 전체 합산
            total_potential += np.sum(potential, axis=1)  # (K,)

        return total_potential

    def _check_recovery(self, state: np.ndarray):
        """
        탈출 성공 판정

        최근 window 스텝 이동량 > recovery_threshold → 정상 모드 복귀

        Args:
            state: (nx,) 현재 상태
        """
        window = self.drpa_params.stagnation_window
        threshold = self.drpa_params.recovery_threshold

        if len(self._stagnation_history) >= window:
            recent = self._stagnation_history[-window:]
            displacement = np.linalg.norm(
                np.array(recent[-1]) - np.array(recent[0])
            )
            if displacement > threshold:
                self._in_escape_mode = False

    def _compute_min_clearance(self, state: np.ndarray) -> float:
        """
        현재 위치에서 가장 가까운 장애물까지 간극

        Args:
            state: (nx,) 현재 상태

        Returns:
            min_clearance: 최소 간극 (장애물 없으면 inf)
        """
        if len(self._obstacles) == 0:
            return float("inf")

        min_clearance = float("inf")
        for ox, oy, r in self._obstacles:
            dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
            clearance = dist - r
            min_clearance = min(min_clearance, clearance)

        return min_clearance

    def update_obstacles(self, obstacles: List[tuple]):
        """
        장애물 실시간 업데이트

        Args:
            obstacles: [(x, y, radius), ...] 새 장애물 목록
        """
        self._obstacles = list(obstacles)

    def get_drpa_statistics(self) -> Dict:
        """
        누적 DRPA 통계 반환

        Returns:
            dict:
                - total_steps: 총 스텝 수
                - escape_count: 탈출 모드 진입 횟수
                - escape_ratio: 탈출 모드 비율
                - mean_repulsive_cost: 평균 반발 포텐셜
                - history: 전체 스텝별 통계
        """
        if not self._drpa_history:
            return {
                "total_steps": 0,
                "escape_count": 0,
                "escape_ratio": 0.0,
                "mean_repulsive_cost": 0.0,
                "history": [],
            }

        escape_steps = sum(
            1 for h in self._drpa_history if h["in_escape_mode"]
        )
        rep_costs = [h["repulsive_cost_mean"] for h in self._drpa_history]

        return {
            "total_steps": self._total_steps,
            "escape_count": self._escape_count,
            "escape_ratio": escape_steps / self._total_steps
            if self._total_steps > 0
            else 0.0,
            "mean_repulsive_cost": float(np.mean(rep_costs)),
            "history": self._drpa_history.copy(),
        }

    def reset(self):
        """전체 상태 초기화"""
        super().reset()
        self._in_escape_mode = False
        self._stagnation_history = []
        self._cost_history = []
        self._drpa_history = []
        self._escape_count = 0
        self._total_steps = 0
        self._obstacles = list(self.drpa_params.obstacles)

    def __repr__(self) -> str:
        return (
            f"DRPAMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"obstacles={len(self._obstacles)}, "
            f"repulsive_gain={self.drpa_params.repulsive_gain}, "
            f"influence_distance={self.drpa_params.influence_distance}, "
            f"escape_mode={self._in_escape_mode}, "
            f"K={self.params.K})"
        )
