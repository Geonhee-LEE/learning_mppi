"""
CSC-MPPI Controller (Constrained Sampling Cluster MPPI)

Primal-dual 투영으로 샘플을 실행 가능 영역으로 이동시키고,
DBSCAN 클러스터링으로 대표 궤적을 선택. 가중 평균 대신
최적 클러스터 대표를 사용하여 실행 가능성을 보장.

기존 MPPI의 근본 문제:
  - 안전한 궤적들의 가중 평균이 불안전할 수 있음 (convex combination)
  - CSC-MPPI는 클러스터 대표 선택으로 이 문제를 해결

Reference: arXiv:2506.16386, 2025

핵심 수식:
    1. Primal-Dual 투영:
       L(V, λ) = C(V) + λ · g(V)
       V ← V - α_v · ∇_V L
       λ ← max(0, λ + α_λ · g(V))
       g(V) = safety_margin - min_dist(V, obs)

    2. DBSCAN 클러스터링:
       투영된 제어 시퀀스 → 특징 벡터 → 클러스터 분류

    3. 최종 선택:
       각 클러스터 대표 (최저 비용) 중 전체 최적 선택
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import CSCMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler

# DBSCAN import (sklearn 선택적)
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class CSCMPPIController(MPPIController):
    """
    CSC-MPPI Controller (32번째 MPPI 변형)

    Constrained Sampling Cluster MPPI:
      1. Primal-dual gradient로 샘플 궤적을 실행 가능 영역으로 투영
      2. DBSCAN 클러스터링으로 다양한 궤적 모드 분류
      3. 가중 평균 대신 최적 클러스터 대표 궤적 선택

    Vanilla MPPI 대비 핵심 차이:
        1. 제약 투영: 샘플 제어 시퀀스를 primal-dual로 실행 가능 영역에 투영
        2. 클러스터링: 투영된 궤적들을 DBSCAN으로 분류
        3. 대표 선택: 가중 평균 아닌 클러스터 대표 중 최적 선택
        4. 실행 가능성 보장: 안전 궤적들의 평균이 불안전한 문제 해결

    Args:
        model: RobotModel 인스턴스
        params: CSCMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: CSCMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.csc_params = params

        # 장애물 캐시
        self._obstacles = list(params.obstacles)

        # 통계 히스토리
        self._csc_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        CSC-MPPI 제어 계산

        1. 노이즈 샘플링 → 제어 시퀀스 (K, N, nu)
        2. primal-dual 제약 투영 (선택적)
        3. rollout → 비용 계산
        4. DBSCAN 클러스터링 (선택적)
        5. 클러스터 대표 중 최적 선택
        6. 선택된 궤적의 제어를 U로 설정 (전체 교체)
        7. Receding horizon shift

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

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
        sampled_controls = self.U[None, :, :] + noise  # (K, N, nu)

        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 2. Primal-dual 제약 투영
        n_projected = 0
        total_violation_reduction = 0.0
        if self.csc_params.use_projection and len(self._obstacles) > 0:
            sampled_controls, projection_info = self._project_to_feasible(
                sampled_controls, state
            )
            n_projected = projection_info["n_projected"]
            total_violation_reduction = projection_info["violation_reduction"]

            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. Rollout + 비용
        trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
        costs = self.cost_function.compute_cost(
            trajectories, sampled_controls, reference_trajectory
        )

        # 4. 클러스터링 + 대표 선택
        n_clusters = 0
        n_noise_samples = 0
        selected_idx = np.argmin(costs)  # 기본: 최저 비용

        if self.csc_params.use_clustering and K >= self.csc_params.dbscan_min_samples:
            labels, cluster_info = self._cluster_trajectories(
                sampled_controls, costs
            )
            n_clusters = cluster_info["n_clusters"]
            n_noise_samples = cluster_info["n_noise"]

            if n_clusters > 0:
                selected_idx = self._select_best_cluster(
                    sampled_controls, costs, labels
                )
            elif self.csc_params.fallback_to_mppi:
                # 클러스터 없음 → 표준 MPPI 폴백
                selected_idx = self._mppi_fallback(
                    noise, costs, state, reference_trajectory
                )
        elif not self.csc_params.use_clustering:
            # 클러스터링 비활성 → 표준 MPPI
            selected_idx = self._mppi_fallback(
                noise, costs, state, reference_trajectory
            )

        # 5. 선택된 궤적의 제어를 U로 설정 (전체 교체)
        if isinstance(selected_idx, str) and selected_idx == "mppi_fallback":
            # MPPI 폴백: 이미 U가 갱신됨
            pass
        else:
            self.U = sampled_controls[selected_idx].copy()

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 6. 첫 제어 추출
        optimal_control = self.U[0].copy()

        # 7. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 8. ESS 계산
        weights = self._compute_weights(costs, self.params.lambda_)
        ess = self._compute_ess(weights)

        # 9. 안전성 메트릭
        best_idx_for_traj = (
            selected_idx if isinstance(selected_idx, (int, np.integer))
            else np.argmin(costs)
        )
        min_clearance = self._compute_min_clearance(
            trajectories[best_idx_for_traj]
        )

        # 10. 통계
        csc_stats = {
            "n_clusters": n_clusters,
            "n_noise_samples": n_noise_samples,
            "n_projected": n_projected,
            "violation_reduction": float(total_violation_reduction),
            "selected_idx": int(best_idx_for_traj),
            "min_clearance": float(min_clearance),
            "use_projection": self.csc_params.use_projection,
            "use_clustering": self.csc_params.use_clustering,
            "num_obstacles": len(self._obstacles),
        }
        self._csc_history.append(csc_stats)

        best_idx_overall = np.argmin(costs)
        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx_for_traj],
            "best_cost": float(costs[best_idx_for_traj]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "csc_stats": csc_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _project_to_feasible(
        self,
        controls: np.ndarray,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Primal-dual gradient descent로 제어 시퀀스를 실행 가능 영역으로 투영

        L(V, λ) = C(V) + λ · g(V)
        V ← V - α_v · ∇_V L   (primal)
        λ ← max(0, λ + α_λ · g(V))   (dual)

        Args:
            controls: (K, N, nu) 샘플 제어 시퀀스
            state: (nx,) 현재 상태

        Returns:
            projected_controls: (K, N, nu) 투영된 제어 시퀀스
            info: dict 투영 정보
        """
        K, N, nu = controls.shape
        n_obs = len(self._obstacles)
        if n_obs == 0:
            return controls, {"n_projected": 0, "violation_reduction": 0.0}

        projected = controls.copy()
        dual_vars = np.zeros((K, n_obs))  # (K, n_obstacles)

        alpha_v = self.csc_params.projection_lr
        alpha_lambda = self.csc_params.dual_lr
        safety_margin = self.csc_params.safety_margin

        # 초기 위반량 계산
        trajectories_init = self.dynamics_wrapper.rollout(state, projected)
        violations_init = self._compute_violations(trajectories_init)
        initial_total_violation = float(np.sum(np.maximum(violations_init, 0)))

        for step in range(self.csc_params.n_projection_steps):
            # 현재 궤적 rollout
            trajectories = self.dynamics_wrapper.rollout(state, projected)

            # 제약 위반 계산: g(V) = safety_margin - min_dist
            violations = self._compute_violations(trajectories)  # (K, n_obs)

            # Primal 업데이트: 제약 위반을 줄이는 방향으로 제어 이동
            grad = self._compute_constraint_gradient(
                projected, state, violations, dual_vars
            )
            projected = projected - alpha_v * grad

            # Dual 업데이트: λ ← max(0, λ + α_λ · g(V))
            dual_vars = np.maximum(0, dual_vars + alpha_lambda * violations)

        # 최종 위반량 계산
        trajectories_final = self.dynamics_wrapper.rollout(state, projected)
        violations_final = self._compute_violations(trajectories_final)
        final_total_violation = float(np.sum(np.maximum(violations_final, 0)))

        violation_reduction = initial_total_violation - final_total_violation
        n_projected = int(np.sum(np.any(violations_init > 0, axis=-1)))

        return projected, {
            "n_projected": n_projected,
            "violation_reduction": max(0, violation_reduction),
        }

    def _compute_violations(self, trajectories: np.ndarray) -> np.ndarray:
        """
        각 장애물에 대한 제약 위반량 계산

        g(V) = safety_margin - min_dist(trajectory, obstacle)

        Args:
            trajectories: (K, N+1, nx) 궤적

        Returns:
            violations: (K, n_obstacles) 위반량 (양수=위반)
        """
        K = trajectories.shape[0]
        n_obs = len(self._obstacles)
        safety_margin = self.csc_params.safety_margin

        violations = np.zeros((K, n_obs))

        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        for i, (ox, oy, r) in enumerate(self._obstacles):
            obs_pos = np.array([ox, oy])
            # 거리: (K, N+1)
            diff = positions - obs_pos[None, None, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (K, N+1)

            # 최소 거리 (궤적 전체에서) - 장애물 반경
            min_dist = np.min(dist, axis=1) - r  # (K,)

            # 위반: safety_margin - min_dist (양수=위반)
            violations[:, i] = safety_margin - min_dist

        return violations

    def _compute_constraint_gradient(
        self,
        controls: np.ndarray,
        state: np.ndarray,
        violations: np.ndarray,
        dual_vars: np.ndarray,
    ) -> np.ndarray:
        """
        제약 조건의 제어 입력에 대한 벡터화 해석적 기울기 계산

        최소 거리 시점에서 heading-장애물 방향 관계를 이용하여
        속도 감소 + 회전 방향을 벡터화로 일괄 계산.

        Args:
            controls: (K, N, nu) 제어
            state: (nx,) 상태
            violations: (K, n_obs) 위반량
            dual_vars: (K, n_obs) dual 변수

        Returns:
            grad: (K, N, nu) 제어에 대한 기울기
        """
        K, N, nu = controls.shape
        n_obs = len(self._obstacles)

        grad = np.zeros_like(controls)

        # 위반이 있는 샘플에만 기울기 적용
        has_violation = np.any(violations > 0, axis=-1)  # (K,)
        if not np.any(has_violation):
            return grad

        # 현재 궤적 rollout (캐시 가능하지만 투영 루프에서 매번 변경)
        trajectories = self.dynamics_wrapper.rollout(state, controls)
        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        for i, (ox, oy, r) in enumerate(self._obstacles):
            obs_pos = np.array([ox, oy])

            # 활성 샘플 필터링
            active = (violations[:, i] > 0) | (dual_vars[:, i] > 0)  # (K,)
            if not np.any(active):
                continue

            active_idx = np.where(active)[0]
            Ka = len(active_idx)

            # 활성 샘플의 위치 (Ka, N+1, 2)
            pos_a = positions[active_idx]
            diff_a = pos_a - obs_pos[None, None, :]  # (Ka, N+1, 2)
            dist_a = np.sqrt(np.sum(diff_a ** 2, axis=-1))  # (Ka, N+1)

            # 최소 거리 시점 (Ka,)
            min_t_idx = np.argmin(dist_a, axis=1)
            min_dist = dist_a[np.arange(Ka), min_t_idx]  # (Ka,)

            # 방향 벡터 (Ka, 2)
            min_diff = diff_a[np.arange(Ka), min_t_idx, :]
            direction = min_diff / (min_dist[:, None] + 1e-8)

            # 가중치 (Ka,)
            weight = dual_vars[active_idx, i] + np.maximum(violations[active_idx, i], 0)
            grad_scale = weight / (min_dist + 0.1)  # (Ka,)

            # 타겟 시간 스텝 (최소 거리 시점 근처 3개 스텝)
            # min_t_idx를 제어 인덱스로 클리핑 (0 ~ N-1)
            t_center = np.minimum(min_t_idx, N - 1)  # (Ka,)

            # heading at min distance time (Ka,)
            t_for_heading = np.minimum(min_t_idx, trajectories.shape[1] - 1)
            headings = trajectories[active_idx, t_for_heading, 2] if trajectories.shape[-1] > 2 else np.zeros(Ka)
            heading_cos = np.cos(headings)  # (Ka,)
            heading_sin = np.sin(headings)  # (Ka,)

            # cross product: heading x direction → 회전 방향
            cross = heading_cos * direction[:, 1] - heading_sin * direction[:, 0]  # (Ka,)

            # 3 시점에 기울기 분배: t_center-2, t_center-1, t_center
            for dt_offset in [-2, -1, 0]:
                t_apply = t_center + dt_offset  # (Ka,)
                valid = (t_apply >= 0) & (t_apply < N)
                if not np.any(valid):
                    continue

                valid_k = np.where(valid)[0]
                t_valid = t_apply[valid_k]
                global_k = active_idx[valid_k]

                # v 성분: 장애물 접근 시 속도 감소
                grad[global_k, t_valid, 0] += grad_scale[valid_k] * 0.5

                # ω 성분: 장애물 반대 방향 회전
                grad[global_k, t_valid, 1] += grad_scale[valid_k] * cross[valid_k] * 0.5

        # 위반이 없는 샘플은 기울기 0 (이미 0이지만 안전장치)
        grad[~has_violation] = 0.0

        return grad

    def _cluster_trajectories(
        self,
        controls: np.ndarray,
        costs: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        DBSCAN 클러스터링으로 제어 시퀀스 분류

        Args:
            controls: (K, N, nu) 제어 시퀀스
            costs: (K,) 비용

        Returns:
            labels: (K,) 클러스터 라벨 (-1 = 노이즈)
            info: dict 클러스터링 정보
        """
        K = controls.shape[0]
        features = controls.reshape(K, -1)  # (K, N*nu)

        eps = self.csc_params.dbscan_eps
        min_samples = self.csc_params.dbscan_min_samples

        if HAS_SKLEARN:
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clustering.fit_predict(features)
        else:
            # 폴백: 비용 순서 기반 거리 클러스터링
            labels = self._fallback_clustering(features, costs, eps, min_samples)

        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = int(np.sum(labels == -1))

        return labels, {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

    def _fallback_clustering(
        self,
        features: np.ndarray,
        costs: np.ndarray,
        eps: float,
        min_samples: int,
    ) -> np.ndarray:
        """
        sklearn 없을 때 간단한 거리 기반 클러스터링

        비용 순서로 정렬 후 eps 거리 내 그룹화.

        Args:
            features: (K, D) 특징 벡터
            costs: (K,) 비용
            eps: 이웃 거리
            min_samples: 최소 샘플 수

        Returns:
            labels: (K,) 라벨 (-1 = 노이즈)
        """
        K = features.shape[0]
        labels = np.full(K, -1, dtype=int)
        sorted_indices = np.argsort(costs)

        current_label = 0
        assigned = np.zeros(K, dtype=bool)

        for idx in sorted_indices:
            if assigned[idx]:
                continue

            # 이 샘플로부터 eps 이내의 이웃 찾기
            dists = np.linalg.norm(features - features[idx], axis=-1)
            neighbors = np.where((dists < eps) & (~assigned))[0]

            if len(neighbors) >= min_samples:
                labels[neighbors] = current_label
                assigned[neighbors] = True
                current_label += 1
            else:
                # 이웃이 부족하면 노이즈
                if not assigned[idx]:
                    labels[idx] = -1

        return labels

    def _select_best_cluster(
        self,
        controls: np.ndarray,
        costs: np.ndarray,
        labels: np.ndarray,
    ) -> int:
        """
        클러스터 대표 중 최적 선택

        각 클러스터에서 최저 비용 샘플을 대표로 선택하고,
        대표들 중 전체 최저 비용을 반환.

        Args:
            controls: (K, N, nu) 제어 시퀀스
            costs: (K,) 비용
            labels: (K,) 클러스터 라벨

        Returns:
            best_idx: 최적 클러스터 대표의 인덱스
        """
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 노이즈 제외

        if len(unique_labels) == 0:
            return int(np.argmin(costs))

        best_cost = float("inf")
        best_idx = int(np.argmin(costs))

        for label in unique_labels:
            mask = labels == label
            cluster_costs = costs[mask]
            cluster_indices = np.where(mask)[0]

            # 클러스터 내 최저 비용
            local_best = np.argmin(cluster_costs)
            global_idx = cluster_indices[local_best]

            if costs[global_idx] < best_cost:
                best_cost = costs[global_idx]
                best_idx = int(global_idx)

        return best_idx

    def _mppi_fallback(
        self,
        noise: np.ndarray,
        costs: np.ndarray,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> str:
        """
        표준 MPPI 폴백: 가중 평균으로 제어 업데이트

        Args:
            noise: (K, N, nu) 노이즈
            costs: (K,) 비용
            state: (nx,) 상태
            reference_trajectory: (N+1, nx) 레퍼런스

        Returns:
            "mppi_fallback" 문자열 (전체 교체 대신 MPPI 업데이트)
        """
        weights = self._compute_weights(costs, self.params.lambda_)
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        return "mppi_fallback"

    def _compute_min_clearance(self, trajectory: np.ndarray) -> float:
        """
        궤적의 최소 장애물 클리어런스 계산

        Args:
            trajectory: (N+1, nx) 궤적

        Returns:
            min_clearance: 최소 클리어런스 (양수=안전)
        """
        if len(self._obstacles) == 0:
            return float("inf")

        positions = trajectory[:, :2]  # (N+1, 2)
        min_clearance = float("inf")

        for ox, oy, r in self._obstacles:
            obs_pos = np.array([ox, oy])
            dists = np.sqrt(np.sum((positions - obs_pos) ** 2, axis=-1))
            clearance = float(np.min(dists) - r)
            min_clearance = min(min_clearance, clearance)

        return min_clearance

    def update_obstacles(self, obstacles: List[tuple]):
        """
        장애물 실시간 업데이트

        Args:
            obstacles: [(x, y, radius), ...] 새 장애물 목록
        """
        self._obstacles = list(obstacles)

    def get_csc_statistics(self) -> Dict:
        """
        누적 CSC 통계 반환

        Returns:
            dict: 통계 정보
        """
        if len(self._csc_history) == 0:
            return {
                "history": [],
                "mean_clusters": 0.0,
                "mean_violation_reduction": 0.0,
                "min_clearance_ever": float("inf"),
                "total_steps": 0,
            }

        n_clusters = [s["n_clusters"] for s in self._csc_history]
        reductions = [s["violation_reduction"] for s in self._csc_history]
        clearances = [s["min_clearance"] for s in self._csc_history]

        return {
            "history": self._csc_history.copy(),
            "mean_clusters": float(np.mean(n_clusters)),
            "mean_violation_reduction": float(np.mean(reductions)),
            "min_clearance_ever": float(np.min(clearances)),
            "total_steps": len(self._csc_history),
        }

    def reset(self):
        """제어 시퀀스 + 통계 초기화"""
        super().reset()
        self._csc_history = []
        self._obstacles = list(self.csc_params.obstacles)

    def __repr__(self) -> str:
        return (
            f"CSCMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"obstacles={len(self._obstacles)}, "
            f"n_projection_steps={self.csc_params.n_projection_steps}, "
            f"dbscan_eps={self.csc_params.dbscan_eps}, "
            f"K={self.params.K})"
        )
