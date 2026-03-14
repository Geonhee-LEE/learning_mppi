"""
SE(3) 비용 함수 모듈

SO(3) 측지선 거리 기반 자세 추적 비용 함수.
RPY 오차 대신 회전 행렬 로그 맵(geodesic distance)을 사용하여
정확한 orientation 추적 비용을 계산.

수식:
    d(R1, R2) = || log(R1^T R2) ||_F / sqrt(2)
    log(R): R = I + sin(θ)/θ * K + (1-cos(θ))/θ² * K²  →  역으로
    θ = arccos((tr(R) - 1) / 2)
    log(R) = θ / (2 sin(θ)) * (R - R^T)   (θ ≠ 0)
"""

import numpy as np
from typing import Optional, List, Tuple
from mppi_controller.controllers.mppi.cost_functions import CostFunction


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _batch_rotation_log_norm(R: np.ndarray) -> np.ndarray:
    """
    SO(3) 로그 맵 노름 계산: || log(R) ||_F

    geodesic distance = || log(R) ||_F / sqrt(2)

    Args:
        R: (..., 3, 3) 회전 행렬

    Returns:
        log_norm: (...,) 각도 노름 (라디안, [0, π])
    """
    # trace → cos(θ) = (tr(R) - 1) / 2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (...,) ∈ [0, π]
    return theta  # ||log(R)||_F = sqrt(2) * θ  → geodesic = θ


def _geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """
    SO(3) 측지선 거리: d(R1, R2) = arccos((tr(R1^T R2) - 1) / 2)

    Args:
        R1: (..., 3, 3)
        R2: (..., 3, 3)

    Returns:
        dist: (...,) ∈ [0, π] (라디안)
    """
    R_rel = np.einsum("...ij,...ik->...jk", R1, R2)  # R1^T @ R2
    return _batch_rotation_log_norm(R_rel)


def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """
    ZYX Euler (roll, pitch, yaw) → 회전 행렬 R = Rz(yaw) Ry(pitch) Rx(roll)

    Args:
        rpy: (..., 3) [roll, pitch, yaw]

    Returns:
        R: (..., 3, 3)
    """
    r = rpy[..., 0]
    p = rpy[..., 1]
    y = rpy[..., 2]

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    batch = rpy.shape[:-1]
    R = np.zeros(batch + (3, 3), dtype=rpy.dtype)

    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr

    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr

    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr

    return R


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) 비용 함수
# ─────────────────────────────────────────────────────────────────────────────

class GeodesicOrientationCost(CostFunction):
    """
    SO(3) 측지선 거리 기반 자세 추적 비용.

    RPY 오차의 단순 제곱합(wrap 포함) 대신 회전 행렬의 측지선 거리를 사용.
    특이점(gimbal lock)이 없고 전 각도 범위에서 올바른 오차 측도.

    J = ori_weight * Σ_{t=0}^{N-1} d(R_t, R_ref_t)²

    참조 궤적 형식:
        reference_trajectory의 마지막 3열이 [roll, pitch, yaw] 또는
        target_rpy를 직접 지정.

    Args:
        model: forward_kinematics_full(state) → (..., 4, 4) 를 제공하는 모델
        ori_weight: 자세 비용 가중치
        rpy_ref_indices: reference_trajectory에서 RPY가 있는 열 인덱스 (3개)
                         None이면 target_rpy 사용
        target_rpy: 고정 목표 자세 [roll, pitch, yaw] (rpy_ref_indices가 None일 때)
    """

    def __init__(
        self,
        model,
        ori_weight: float = 10.0,
        rpy_ref_indices: Optional[Tuple[int, int, int]] = None,
        target_rpy: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.ori_weight = ori_weight
        self.rpy_ref_indices = rpy_ref_indices
        self.target_rpy = np.asarray(target_rpy, dtype=np.float64) if target_rpy is not None else None

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, ref_dim)

        Returns:
            costs: (K,)
        """
        K, N_plus_1, nx = trajectories.shape
        N = N_plus_1 - 1

        # FK full: (K, N, nx) → (K, N, 4, 4)
        states_run = trajectories[:, :-1, :]  # (K, N, nx)
        T_world = self.model.forward_kinematics_full(states_run)  # (K, N, 4, 4)
        R_traj = T_world[..., :3, :3]  # (K, N, 3, 3)

        # 목표 회전 행렬 (N, 3, 3)
        if self.rpy_ref_indices is not None:
            idx = self.rpy_ref_indices
            ref_rpy = reference_trajectory[:-1, list(idx)]  # (N, 3)
        elif self.target_rpy is not None:
            ref_rpy = np.tile(self.target_rpy, (N, 1))  # (N, 3)
        else:
            raise ValueError("rpy_ref_indices 또는 target_rpy 중 하나를 지정하세요.")

        R_ref = _rpy_to_rotation_matrix(ref_rpy)  # (N, 3, 3)

        # geodesic distance: (K, N) — 브로드캐스트 (K, N, 3, 3) vs (N, 3, 3)
        dist = _geodesic_distance(R_traj, R_ref[None, :, :, :])  # (K, N)

        return self.ori_weight * np.sum(dist**2, axis=1)  # (K,)


class GeodesicOrientationTerminalCost(CostFunction):
    """
    SO(3) 측지선 거리 기반 터미널 자세 비용.

    J = ori_weight * d(R_N, R_goal)²

    Args:
        model: forward_kinematics_full(state) 제공 모델
        target_rpy: 목표 자세 [roll, pitch, yaw]
        ori_weight: 가중치
    """

    def __init__(
        self,
        model,
        target_rpy: np.ndarray,
        ori_weight: float = 20.0,
    ):
        self.model = model
        self.target_rpy = np.asarray(target_rpy, dtype=np.float64)
        self.ori_weight = ori_weight
        # 사전 계산
        self.R_goal = _rpy_to_rotation_matrix(self.target_rpy[None])[0]  # (3, 3)

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        # FK at terminal: (K, nx) → (K, 4, 4)
        T_terminal = self.model.forward_kinematics_full(trajectories[:, -1, :])
        R_terminal = T_terminal[..., :3, :3]  # (K, 3, 3)

        R_goal_batch = self.R_goal[None, :, :]  # (1, 3, 3)
        dist = _geodesic_distance(R_terminal, R_goal_batch)  # (K,)

        return self.ori_weight * dist**2


class SE3TrackingCost(CostFunction):
    """
    SE(3) 추적 비용: 위치(Euclidean) + 자세(SO(3) Geodesic) 통합.

    J = pos_weight * Σ ||p_t - p_ref_t||²
      + ori_weight * Σ d(R_t, R_ref_t)²

    참조 궤적 형식: (N+1, ≥6) — 처음 3열=위치, 다음 3열=RPY
    또는 target_pos, target_rpy를 고정값으로 지정.

    Args:
        model: forward_kinematics_full(state) → (..., 4, 4) 제공 모델
        pos_weight: 위치 비용 가중치
        ori_weight: 자세 비용 가중치
        pos_ref_indices: reference_trajectory에서 위치 열 인덱스 (3개)
        rpy_ref_indices: reference_trajectory에서 RPY 열 인덱스 (3개)
        target_pos: 고정 목표 위치 (pos_ref_indices=None 시)
        target_rpy: 고정 목표 자세 (rpy_ref_indices=None 시)
    """

    def __init__(
        self,
        model,
        pos_weight: float = 100.0,
        ori_weight: float = 10.0,
        pos_ref_indices: Optional[Tuple[int, int, int]] = None,
        rpy_ref_indices: Optional[Tuple[int, int, int]] = None,
        target_pos: Optional[np.ndarray] = None,
        target_rpy: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self.pos_ref_indices = pos_ref_indices
        self.rpy_ref_indices = rpy_ref_indices
        self.target_pos = np.asarray(target_pos, dtype=np.float64) if target_pos is not None else None
        self.target_rpy = np.asarray(target_rpy, dtype=np.float64) if target_rpy is not None else None

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        K, N_plus_1, nx = trajectories.shape
        N = N_plus_1 - 1

        # FK: (K, N, nx) → (K, N, 4, 4)
        states_run = trajectories[:, :-1, :]
        T_world = self.model.forward_kinematics_full(states_run)

        # 위치 오차
        p_traj = T_world[..., :3, 3]  # (K, N, 3)
        if self.pos_ref_indices is not None:
            idx = list(self.pos_ref_indices)
            p_ref = reference_trajectory[:-1, idx]  # (N, 3)
        elif self.target_pos is not None:
            p_ref = np.tile(self.target_pos, (N, 1))
        else:
            # 기본: reference_trajectory 처음 3열이 EE 위치
            p_ref = reference_trajectory[:-1, :3]

        pos_err = p_traj - p_ref[None, :, :]  # (K, N, 3)
        pos_cost = self.pos_weight * np.sum(pos_err**2, axis=(1, 2))  # (K,)

        # 자세 오차 (geodesic)
        R_traj = T_world[..., :3, :3]  # (K, N, 3, 3)
        if self.rpy_ref_indices is not None:
            idx = list(self.rpy_ref_indices)
            ref_rpy = reference_trajectory[:-1, idx]
        elif self.target_rpy is not None:
            ref_rpy = np.tile(self.target_rpy, (N, 1))
        else:
            # 기본: reference_trajectory 열 3~5가 RPY
            ref_rpy = reference_trajectory[:-1, 3:6]

        R_ref = _rpy_to_rotation_matrix(ref_rpy)  # (N, 3, 3)
        dist = _geodesic_distance(R_traj, R_ref[None, :, :, :])  # (K, N)
        ori_cost = self.ori_weight * np.sum(dist**2, axis=1)  # (K,)

        return pos_cost + ori_cost


class SE3TerminalCost(CostFunction):
    """
    SE(3) 터미널 비용: 위치 + 자세(geodesic) 최종 스텝.

    J = pos_weight * ||p_N - p_goal||²
      + ori_weight * d(R_N, R_goal)²

    Args:
        model: forward_kinematics_full(state) 제공 모델
        target_pos: 목표 EE 위치 [x, y, z]
        target_rpy: 목표 EE 자세 [roll, pitch, yaw]
        pos_weight: 위치 가중치
        ori_weight: 자세 가중치
    """

    def __init__(
        self,
        model,
        target_pos: np.ndarray,
        target_rpy: np.ndarray,
        pos_weight: float = 200.0,
        ori_weight: float = 20.0,
    ):
        self.model = model
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.target_rpy = np.asarray(target_rpy, dtype=np.float64)
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self.R_goal = _rpy_to_rotation_matrix(self.target_rpy[None])[0]  # (3, 3)

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        # FK at terminal
        T_terminal = self.model.forward_kinematics_full(trajectories[:, -1, :])

        # 위치
        p_terminal = T_terminal[..., :3, 3]  # (K, 3)
        pos_err = p_terminal - self.target_pos
        pos_cost = self.pos_weight * np.sum(pos_err**2, axis=1)

        # 자세
        R_terminal = T_terminal[..., :3, :3]  # (K, 3, 3)
        dist = _geodesic_distance(R_terminal, self.R_goal[None])  # (K,)
        ori_cost = self.ori_weight * dist**2

        return pos_cost + ori_cost


class ReachabilityMapCost(CostFunction):
    """
    팔 도달가능성 기반 베이스 이동 비용.

    사전에 계산된 도달가능성 구(sphere)를 이용해 베이스가
    EE 목표에 가까이 위치하도록 유도.

    J = reach_weight * max(0, ||p_base - p_goal_xy|| - max_reach + margin)²

    Args:
        target_pos: 목표 EE 위치 [x, y, z]
        max_reach: 팔의 최대 도달 거리 (m) — DH 파라미터 기반 or 실험치
        margin: 최적 도달 거리 마진 (m)
        reach_weight: 비용 가중치
        base_indices: 상태 벡터에서 (x, y) 인덱스
    """

    def __init__(
        self,
        target_pos: np.ndarray,
        max_reach: float = 0.70,
        margin: float = 0.05,
        reach_weight: float = 50.0,
        base_indices: Tuple[int, int] = (0, 1),
    ):
        self.target_xy = np.asarray(target_pos[:2], dtype=np.float64)
        self.max_reach = max_reach
        self.margin = margin
        self.reach_weight = reach_weight
        self.base_indices = base_indices

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        K, N_plus_1, nx = trajectories.shape

        # 베이스 위치: (K, N+1, 2)
        ix, iy = self.base_indices
        base_xy = trajectories[:, :, [ix, iy]]  # (K, N+1, 2)

        # 목표 XY까지의 수평 거리
        diff = base_xy - self.target_xy  # (K, N+1, 2)
        dist_xy = np.sqrt(np.sum(diff**2, axis=-1))  # (K, N+1)

        # 최대 도달 거리 초과분에 패널티
        excess = np.maximum(0.0, dist_xy - (self.max_reach - self.margin))  # (K, N+1)
        cost = self.reach_weight * np.sum(excess**2, axis=1)  # (K,)

        return cost


class SE3ManipulabilityCost(CostFunction):
    """
    Manipulability measure 기반 특이점 회피 비용.

    w(q) = sqrt(det(J(q) J(q)^T)) — 0에 가까울수록 특이점 근처.
    w가 낮은 자세에 패널티를 부과하여 특이점 회피.

    J = manip_weight * Σ max(0, threshold - w(q_t))²

    Jacobian은 수치 미분으로 계산 (모든 모델 호환).

    Args:
        model: forward_kinematics(state) → (..., 3) EE 위치 제공 모델
        joint_indices: 상태에서 관절 각도 인덱스 (기본: 6-DOF [3..8])
        threshold: manipulability 임계값 (이 값 미만에서 비용 발생)
        manip_weight: 비용 가중치
        eps: 수치 미분 스텝
    """

    def __init__(
        self,
        model,
        joint_indices: Optional[List[int]] = None,
        threshold: float = 0.05,
        manip_weight: float = 20.0,
        eps: float = 1e-4,
    ):
        self.model = model
        self.joint_indices = joint_indices if joint_indices is not None else list(range(3, 9))
        self.threshold = threshold
        self.manip_weight = manip_weight
        self.eps = eps
        self._n_joints = len(self.joint_indices)

    def _compute_jacobian(self, states: np.ndarray) -> np.ndarray:
        """
        수치 Jacobian J ∈ R^{3 × n_joints} 계산.

        Args:
            states: (M, nx) — M개 상태

        Returns:
            J: (M, 3, n_joints)
        """
        M, nx = states.shape
        n = self._n_joints
        J = np.zeros((M, 3, n), dtype=np.float64)

        ee_base = self.model.forward_kinematics(states)  # (M, 3)

        for k, ji in enumerate(self.joint_indices):
            s_plus = states.copy()
            s_plus[:, ji] += self.eps
            ee_plus = self.model.forward_kinematics(s_plus)  # (M, 3)
            J[:, :, k] = (ee_plus - ee_base) / self.eps

        return J  # (M, 3, n_joints)

    def _manipulability(self, states: np.ndarray) -> np.ndarray:
        """
        Args:
            states: (M, nx)

        Returns:
            w: (M,) manipulability
        """
        J = self._compute_jacobian(states)  # (M, 3, n)
        JJT = np.einsum("mij,mkj->mik", J, J)  # (M, 3, 3)
        det = np.linalg.det(JJT)  # (M,)
        return np.sqrt(np.maximum(det, 0.0))

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        K, N_plus_1, nx = trajectories.shape
        N = N_plus_1 - 1

        # 중간 스텝만 검사 (계산 절약을 위해 stride=2)
        stride = max(1, N // 10)
        step_indices = list(range(0, N, stride))

        total_cost = np.zeros(K, dtype=np.float64)

        for t in step_indices:
            states_t = trajectories[:, t, :].reshape(K, nx)  # (K, nx)
            w = self._manipulability(states_t)  # (K,)
            deficit = np.maximum(0.0, self.threshold - w)
            total_cost += self.manip_weight * deficit**2

        return total_cost
