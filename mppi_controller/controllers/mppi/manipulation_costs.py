"""
Mobile Manipulation 전용 비용 함수 모듈

WBC-MPPI에서 사용하는 조작(manipulation) 특화 비용 함수.
베이스 이동과 팔 조작을 통합 최적화하기 위한 비용.

포함 클래스:
    ReachabilityWorkspaceCost  — FK 기반 팔 도달가능성 판단
    ArmSingularityAvoidanceCost — manipulability w(q) 기반 특이점 회피
    GraspApproachCost          — EE 접근 방향 정렬 비용
    CollisionFreeSweepCost     — 팔 전체 링크 충돌 비용
    WBCBaseNavigationCost      — 베이스 이동 비용 (팔 작업 중)
    JointVelocitySmoothCost    — 관절 속도 부드러움 비용
"""

import numpy as np
from typing import Optional, List, Tuple
from mppi_controller.controllers.mppi.cost_functions import CostFunction


class ReachabilityWorkspaceCost(CostFunction):
    """
    팔 도달가능성 기반 베이스 위치 비용.

    현재 베이스 위치에서 팔이 목표 EE 위치를 도달할 수 있는지 판단.
    FK를 통해 팔의 최대 도달 거리(max_reach)와 최소 도달 거리(min_reach)를
    기반으로 베이스가 최적 위치에 있도록 유도.

    비용:
        - 베이스~목표 수평 거리가 [min_reach, max_reach] 범위 벗어나면 페널티
        - 목표 높이가 팔의 z 범위를 벗어나면 페널티

    Args:
        target_pos: 목표 EE 위치 [x, y, z]
        max_reach: 팔 최대 도달 반경 (m), UR5-style ≈ 0.70
        min_reach: 팔 최소 도달 반경 (m), 팔 고관절 주변 금지구역
        target_z_range: 목표 z 높이 허용 범위 [z_min, z_max]
        reach_weight: 가로 도달 불가 페널티 가중치
        height_weight: 세로 높이 불가 페널티 가중치
        base_indices: 상태에서 (x, y) 인덱스
    """

    def __init__(
        self,
        target_pos: np.ndarray,
        max_reach: float = 0.70,
        min_reach: float = 0.10,
        target_z_range: Optional[Tuple[float, float]] = None,
        reach_weight: float = 50.0,
        height_weight: float = 20.0,
        base_indices: Tuple[int, int] = (0, 1),
    ):
        self.target_xy = np.asarray(target_pos[:2], dtype=np.float64)
        self.target_z = float(target_pos[2])
        self.max_reach = max_reach
        self.min_reach = min_reach
        self.target_z_range = target_z_range
        self.reach_weight = reach_weight
        self.height_weight = height_weight
        self.base_indices = base_indices

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        K, N_plus_1, nx = trajectories.shape

        ix, iy = self.base_indices
        base_xy = trajectories[:, :, [ix, iy]]  # (K, N+1, 2)

        # 목표까지 수평 거리
        diff = base_xy - self.target_xy  # (K, N+1, 2)
        dist = np.sqrt(np.sum(diff**2, axis=-1))  # (K, N+1)

        # 최대 도달 초과 페널티
        over_max = np.maximum(0.0, dist - self.max_reach)
        # 최소 도달 미달 페널티 (너무 가까우면 팔이 꺾임)
        under_min = np.maximum(0.0, self.min_reach - dist)

        cost = self.reach_weight * np.sum((over_max + under_min) ** 2, axis=1)

        # z 높이 범위 페널티 (지정 시)
        if self.target_z_range is not None:
            z_min, z_max = self.target_z_range
            # 팔 마운트 높이 기준으로 간단히 체크
            if self.target_z < z_min or self.target_z > z_max:
                cost += self.height_weight * (self.target_z - np.clip(self.target_z, z_min, z_max))**2

        return cost  # (K,)


class ArmSingularityAvoidanceCost(CostFunction):
    """
    관절 각도 기반 특이점 회피 비용.

    수치 Jacobian의 최솟값 특이치(minimum singular value)를 사용하여
    특이점 근접 여부를 판단. manipulability measure 대비 더 안정적.

    w(q) = σ_min(J)  — 최솟값 특이치, 0에 가까울수록 특이점

    비용:
        J = weight * Σ max(0, threshold - σ_min(J(q_t)))²

    Args:
        model: forward_kinematics(state) → (..., 3) 을 제공하는 모델
        joint_indices: 관절 인덱스 (기본: [3, 4, 5, 6, 7, 8] — 6-DOF)
        threshold: σ_min 임계값
        weight: 비용 가중치
        eps: 수치 미분 스텝
        check_stride: 몇 스텝마다 검사할지 (계산 절약)
    """

    def __init__(
        self,
        model,
        joint_indices: Optional[List[int]] = None,
        threshold: float = 0.02,
        weight: float = 20.0,
        eps: float = 1e-4,
        check_stride: int = 3,
    ):
        self.model = model
        self.joint_indices = joint_indices if joint_indices is not None else list(range(3, 9))
        self.threshold = threshold
        self.weight = weight
        self.eps = eps
        self.check_stride = check_stride
        self._n_joints = len(self.joint_indices)

    def _compute_min_singular_value(self, states: np.ndarray) -> np.ndarray:
        """
        Args:
            states: (M, nx)

        Returns:
            sigma_min: (M,)
        """
        M, nx = states.shape
        n = self._n_joints
        J = np.zeros((M, 3, n), dtype=np.float64)

        ee_base = self.model.forward_kinematics(states)  # (M, 3)
        for k, ji in enumerate(self.joint_indices):
            s_plus = states.copy()
            s_plus[:, ji] += self.eps
            ee_plus = self.model.forward_kinematics(s_plus)
            J[:, :, k] = (ee_plus - ee_base) / self.eps

        # 최솟값 특이치 계산
        sigma_min = np.zeros(M, dtype=np.float64)
        for m in range(M):
            sv = np.linalg.svd(J[m], compute_uv=False)
            sigma_min[m] = sv[-1]  # 최솟값 특이치

        return sigma_min

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        K, N_plus_1, nx = trajectories.shape
        N = N_plus_1 - 1

        step_indices = list(range(0, N, self.check_stride))
        total_cost = np.zeros(K, dtype=np.float64)

        for t in step_indices:
            states_t = trajectories[:, t, :]  # (K, nx)
            sigma_min = self._compute_min_singular_value(states_t)  # (K,)
            deficit = np.maximum(0.0, self.threshold - sigma_min)
            total_cost += self.weight * deficit**2

        return total_cost


class GraspApproachCost(CostFunction):
    """
    파지(grasp) 접근 방향 정렬 비용.

    EE가 목표 객체를 향해 특정 방향으로 접근해야 할 때 사용.
    EE z축 방향(tool axis)이 접근 방향과 일치하도록 유도.

    비용:
        J = approach_weight * Σ (1 - (z_ee · d_approach))²
        여기서 z_ee = R_ee @ [0, 0, 1]  (EE tool axis)
               d_approach = 목표를 향한 접근 방향 단위 벡터

    Args:
        model: forward_kinematics_full(state) → (..., 4, 4) 모델
        target_pos: 목표 객체 위치 [x, y, z]
        approach_dir: 고정 접근 방향 단위 벡터 (None이면 EE→target 방향 사용)
        approach_weight: 가중치
        ee_axis: EE 공구 축 인덱스 (기본: 2 = z축)
    """

    def __init__(
        self,
        model,
        target_pos: np.ndarray,
        approach_dir: Optional[np.ndarray] = None,
        approach_weight: float = 30.0,
        ee_axis: int = 2,
    ):
        self.model = model
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.approach_dir = (
            np.asarray(approach_dir, dtype=np.float64) / np.linalg.norm(approach_dir)
            if approach_dir is not None
            else None
        )
        self.approach_weight = approach_weight
        self.ee_axis = ee_axis

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        K, N_plus_1, nx = trajectories.shape

        # FK full: (K, N, nx) → (K, N, 4, 4)
        states_run = trajectories[:, :-1, :]  # (K, N, nx)
        T_world = self.model.forward_kinematics_full(states_run)

        # EE tool axis in world frame
        R_ee = T_world[..., :3, :3]  # (K, N, 3, 3)
        # z축 = 세 번째 열
        tool_axis = R_ee[..., :, self.ee_axis]  # (K, N, 3)

        if self.approach_dir is not None:
            # 고정 접근 방향
            d_approach = self.approach_dir  # (3,)
        else:
            # EE → target 방향 (현재 EE 위치 기준)
            ee_pos = T_world[..., :3, 3]  # (K, N, 3)
            diff = self.target_pos - ee_pos  # (K, N, 3)
            norms = np.linalg.norm(diff, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            d_approach = diff / norms  # (K, N, 3)

        # 정렬 비용: 1 - dot(tool_axis, d_approach)  ∈ [0, 2]
        if self.approach_dir is not None:
            dot = np.sum(tool_axis * d_approach, axis=-1)  # (K, N)
        else:
            dot = np.sum(tool_axis * d_approach, axis=-1)  # (K, N)

        alignment_cost = (1.0 - dot) ** 2  # (K, N)
        return self.approach_weight * np.sum(alignment_cost, axis=1)  # (K,)


class CollisionFreeSweepCost(CostFunction):
    """
    팔 전체 링크 충돌 비용.

    EE만이 아닌 모든 팔 링크 위치를 장애물과 검사.
    각 링크 중간점을 FK로 계산하고 장애물과의 거리를 기반으로 비용 부과.

    비용:
        J = Σ_obstacle Σ_link collision_cost(link_pos, obstacle)
        collision_cost = exp(-dist²/(2σ²)) * weight    (dist < margin)

    Args:
        model: MobileManipulator6DOFKinematic 인스턴스
        obstacles: 장애물 목록 [(cx, cy, radius), ...] — 2D 원형 장애물
        safe_margin: 안전 마진 (m)
        collision_weight: 충돌 비용 가중치
        link_check_count: 검사할 링크 수 (1=EE만, 6=모든 링크)
    """

    def __init__(
        self,
        model,
        obstacles: List[Tuple[float, float, float]],
        safe_margin: float = 0.15,
        collision_weight: float = 200.0,
        link_check_count: int = 4,
    ):
        self.model = model
        self.obstacles = obstacles
        self.safe_margin = safe_margin
        self.collision_weight = collision_weight
        self.link_check_count = link_check_count

    def _get_link_positions(self, states: np.ndarray) -> np.ndarray:
        """
        각 링크의 XY 위치 계산.

        Args:
            states: (M, nx)

        Returns:
            link_xy: (M, n_links, 2)
        """
        M, nx = states.shape
        n = self.link_check_count

        # 점진적으로 각 관절 위치 계산 (partial FK)
        # 간단하게: 상태에서 관절 각도 점진적으로 0으로 만들어 partial FK
        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]

        # Base transform: (M, 4, 4)
        T_base = self.model._base_transform(states)

        # 각 링크 중간점 XY 위치
        link_xy = np.zeros((M, n, 2))

        # EE 위치 (마지막 링크)
        T_ee = self.model.forward_kinematics_full(states)
        link_xy[:, -1, :] = T_ee[:, :2, 3]

        # 중간 링크: 관절 i까지만 FK 계산
        for k in range(1, n - 1):
            joint_idx = int(k * 6 / (n - 1))  # 0~6 관절에 균등 분포
            joint_idx = min(joint_idx, 5)
            # 해당 관절까지만 DH chain
            partial_state = states.copy()
            partial_state[:, 3 + joint_idx + 1:9] = 0.0  # 이후 관절 0으로
            T_partial = self.model.forward_kinematics_full(partial_state)
            link_xy[:, k, :] = T_partial[:, :2, 3]

        # 베이스 위치 (첫 번째)
        link_xy[:, 0, :] = states[:, :2]

        return link_xy  # (M, n_links, 2)

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        K, N_plus_1, nx = trajectories.shape
        N = N_plus_1 - 1

        # 계산 절약: 일부 스텝만 검사
        stride = max(1, N // 5)
        step_indices = list(range(0, N, stride))

        total_cost = np.zeros(K, dtype=np.float64)

        for t in step_indices:
            states_t = trajectories[:, t, :]  # (K, nx)
            link_xy = self._get_link_positions(states_t)  # (K, n_links, 2)

            for obs_x, obs_y, obs_r in self.obstacles:
                obs_center = np.array([[obs_x, obs_y]])  # (1, 2)
                # (K, n_links, 2) - (1, 1, 2) → (K, n_links, 2)
                diff = link_xy - obs_center[:, None, :]
                dist = np.sqrt(np.sum(diff**2, axis=-1))  # (K, n_links)

                # 장애물 경계까지의 거리
                dist_to_edge = dist - obs_r

                # 안전 마진 내에서 비용 발생
                violation = np.maximum(0.0, self.safe_margin - dist_to_edge)
                total_cost += self.collision_weight * np.sum(violation**2, axis=1)

        return total_cost


class WBCBaseNavigationCost(CostFunction):
    """
    팔 조작 중 베이스 이동 비용.

    WBC 시나리오에서 베이스는 EE 작업 중 안정적 위치를 유지해야 함.
    베이스 속도가 크면 팔의 EE 추적 정확도가 저하되므로 페널티.

    비용:
        J = Σ (base_vel_weight * v² + base_omega_weight * ω²)

    Args:
        base_vel_weight: 베이스 선속도 페널티 가중치
        base_omega_weight: 베이스 각속도 페널티 가중치
        base_vel_indices: 제어 벡터에서 (v, ω) 인덱스
    """

    def __init__(
        self,
        base_vel_weight: float = 0.5,
        base_omega_weight: float = 0.5,
        base_vel_indices: Tuple[int, int] = (0, 1),
    ):
        self.base_vel_weight = base_vel_weight
        self.base_omega_weight = base_omega_weight
        self.iv, self.iw = base_vel_indices

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        # controls: (K, N, nu)
        v = controls[:, :, self.iv]    # (K, N) 베이스 선속도
        omega = controls[:, :, self.iw]  # (K, N) 베이스 각속도

        cost = (
            self.base_vel_weight * np.sum(v**2, axis=1)
            + self.base_omega_weight * np.sum(omega**2, axis=1)
        )  # (K,)

        return cost


class JointVelocitySmoothCost(CostFunction):
    """
    관절 속도 부드러움 비용.

    연속 타임스텝 간 관절 속도 변화를 최소화하여 부드러운 팔 움직임 유도.

    비용:
        J = smooth_weight * Σ Σ_joint (dq_{t+1} - dq_t)²

    Args:
        joint_ctrl_indices: 제어 벡터에서 관절 속도 인덱스 (기본: [2..7])
        smooth_weight: 가중치
    """

    def __init__(
        self,
        joint_ctrl_indices: Optional[List[int]] = None,
        smooth_weight: float = 1.0,
    ):
        self.joint_ctrl_indices = (
            joint_ctrl_indices if joint_ctrl_indices is not None else list(range(2, 8))
        )
        self.smooth_weight = smooth_weight

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        K, N, nu = controls.shape

        if N < 2:
            return np.zeros(K)

        # 관절 속도만 추출: (K, N, n_joints)
        idx = self.joint_ctrl_indices
        joint_ctrl = controls[:, :, idx]  # (K, N, n_joints)

        # 연속 스텝 간 차이
        delta = joint_ctrl[:, 1:, :] - joint_ctrl[:, :-1, :]  # (K, N-1, n_joints)
        smooth_cost = self.smooth_weight * np.sum(delta**2, axis=(1, 2))  # (K,)

        return smooth_cost
