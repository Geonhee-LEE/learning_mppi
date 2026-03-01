"""
MPPI 비용 함수 모듈

다양한 비용 함수 컴포넌트 정의.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List


class CostFunction(ABC):
    """비용 함수 추상 베이스 클래스"""

    @abstractmethod
    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        비용 계산

        Args:
            trajectories: (K, N+1, nx) 샘플 궤적
            controls: (K, N, nu) 샘플 제어
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            costs: (K,) 각 샘플의 총 비용
        """
        pass


class StateTrackingCost(CostFunction):
    """
    상태 추적 비용

    J_state = Σ_{t=0}^{N-1} (x_t - x_ref_t)^T Q (x_t - x_ref_t)

    Args:
        Q: (nx,) 또는 (nx, nx) 가중치 행렬 (대각 또는 풀 매트릭스)
    """

    def __init__(self, Q: np.ndarray):
        self.Q = Q
        self.is_diagonal = Q.ndim == 1

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        상태 추적 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        K, N_plus_1, nx = trajectories.shape

        # 오차 계산 (K, N, nx) - 터미널 제외
        errors = trajectories[:, :-1, :] - reference_trajectory[:-1, :]

        if self.is_diagonal:
            # 대각 가중치: (K, N, nx) * (nx,) → (K, N, nx) → (K, N) → (K,)
            costs = np.sum(errors**2 * self.Q, axis=(1, 2))
        else:
            # 풀 매트릭스 가중치: errors @ Q @ errors^T
            # (K, N, nx) @ (nx, nx) → (K, N, nx)
            weighted_errors = np.einsum("ktn,nm->ktm", errors, self.Q)
            # (K, N, nx) * (K, N, nx) → (K, N) → (K,)
            costs = np.sum(weighted_errors * errors, axis=(1, 2))

        return costs


class TerminalCost(CostFunction):
    """
    터미널 비용

    J_terminal = (x_N - x_ref_N)^T Qf (x_N - x_ref_N)

    Args:
        Qf: (nx,) 또는 (nx, nx) 터미널 가중치
    """

    def __init__(self, Qf: np.ndarray):
        self.Qf = Qf
        self.is_diagonal = Qf.ndim == 1

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        터미널 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        # 터미널 상태 (K, nx)
        terminal_states = trajectories[:, -1, :]
        reference_terminal = reference_trajectory[-1, :]

        # 오차 (K, nx)
        errors = terminal_states - reference_terminal

        if self.is_diagonal:
            # 대각 가중치: (K, nx) * (nx,) → (K, nx) → (K,)
            costs = np.sum(errors**2 * self.Qf, axis=1)
        else:
            # 풀 매트릭스 가중치
            weighted_errors = errors @ self.Qf  # (K, nx)
            costs = np.sum(weighted_errors * errors, axis=1)  # (K,)

        return costs


class ControlEffortCost(CostFunction):
    """
    제어 노력 비용

    J_control = Σ_{t=0}^{N-1} u_t^T R u_t

    Args:
        R: (nu,) 또는 (nu, nu) 가중치 행렬
    """

    def __init__(self, R: np.ndarray):
        self.R = R
        self.is_diagonal = R.ndim == 1

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        제어 노력 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        if self.is_diagonal:
            # 대각 가중치: (K, N, nu) * (nu,) → (K, N, nu) → (K, N) → (K,)
            costs = np.sum(controls**2 * self.R, axis=(1, 2))
        else:
            # 풀 매트릭스 가중치
            weighted_controls = np.einsum("ktn,nm->ktm", controls, self.R)
            costs = np.sum(weighted_controls * controls, axis=(1, 2))

        return costs


class ControlRateCost(CostFunction):
    """
    제어 변화율 비용 (M2 고급 기능)

    J_rate = Σ_{t=0}^{N-2} (u_{t+1} - u_t)^T R_rate (u_{t+1} - u_t)

    제어 입력의 변화를 페널티하여 부드러운 제어 생성.

    Args:
        R_rate: (nu,) 또는 (nu, nu) 가중치 행렬
    """

    def __init__(self, R_rate: np.ndarray):
        self.R_rate = R_rate
        self.is_diagonal = R_rate.ndim == 1

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        제어 변화율 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        # 제어 변화 (K, N-1, nu)
        control_diffs = np.diff(controls, axis=1)

        if self.is_diagonal:
            costs = np.sum(control_diffs**2 * self.R_rate, axis=(1, 2))
        else:
            weighted_diffs = np.einsum("ktn,nm->ktm", control_diffs, self.R_rate)
            costs = np.sum(weighted_diffs * control_diffs, axis=(1, 2))

        return costs


class ObstacleCost(CostFunction):
    """
    장애물 회피 비용 (M2 고급 기능)

    원형 장애물 회피. 장애물 내부에 들어가면 큰 비용.

    Args:
        obstacles: List of (x, y, radius) 튜플
        safety_margin: 안전 마진 (m)
        cost_weight: 비용 가중치
    """

    def __init__(
        self,
        obstacles: List[tuple],
        safety_margin: float = 0.2,
        cost_weight: float = 100.0,
    ):
        self.obstacles = obstacles
        self.safety_margin = safety_margin
        self.cost_weight = cost_weight

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        장애물 회피 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        K, N_plus_1, nx = trajectories.shape
        costs = np.zeros(K)

        # 각 장애물에 대해 비용 계산
        for obs_x, obs_y, obs_radius in self.obstacles:
            # 궤적 위치 (K, N+1, 2) - x, y만 사용
            positions = trajectories[:, :, :2]  # (K, N+1, 2)

            # 장애물까지 거리 (K, N+1)
            distances = np.sqrt(
                (positions[..., 0] - obs_x) ** 2 + (positions[..., 1] - obs_y) ** 2
            )

            # 침투 깊이 (음수면 안전, 양수면 위험)
            penetrations = (obs_radius + self.safety_margin) - distances

            # 침투한 경우만 비용 부과 (exponential)
            obstacle_costs = np.where(
                penetrations > 0, np.exp(penetrations * 5.0), 0.0
            )

            # 시간 스텝에 대해 합산 (K, N+1) → (K,)
            costs += self.cost_weight * np.sum(obstacle_costs, axis=1)

        return costs


class AngleAwareTrackingCost(CostFunction):
    """
    상태 추적 비용 + heading angle wrapping.

    heading이 [-π, π]를 넘어 무한히 증가하는 reference와의 차이에서
    2π 오차가 발생하는 것을 방지. angle_indices의 차원에 대해
    arctan2로 래핑하여 올바른 비용 계산.

    Args:
        Q: (nx,) 또는 (nx, nx) 가중치 행렬
        angle_indices: 각도 래핑할 state 인덱스 (기본: (2,) = θ)
    """

    def __init__(self, Q: np.ndarray, angle_indices=(2,)):
        self.Q = Q
        self.is_diagonal = Q.ndim == 1
        self.angle_indices = angle_indices

    def compute_cost(self, trajectories, controls, reference_trajectory):
        errors = trajectories[:, :-1, :] - reference_trajectory[:-1, :]
        for idx in self.angle_indices:
            errors[:, :, idx] = np.arctan2(
                np.sin(errors[:, :, idx]), np.cos(errors[:, :, idx])
            )
        if self.is_diagonal:
            return np.sum(errors**2 * self.Q, axis=(1, 2))
        weighted = np.einsum("ktn,nm->ktm", errors, self.Q)
        return np.sum(weighted * errors, axis=(1, 2))


class AngleAwareTerminalCost(CostFunction):
    """
    터미널 비용 + heading angle wrapping.

    Args:
        Qf: (nx,) 또는 (nx, nx) 터미널 가중치
        angle_indices: 각도 래핑할 state 인덱스 (기본: (2,) = θ)
    """

    def __init__(self, Qf: np.ndarray, angle_indices=(2,)):
        self.Qf = Qf
        self.is_diagonal = Qf.ndim == 1
        self.angle_indices = angle_indices

    def compute_cost(self, trajectories, controls, reference_trajectory):
        errors = trajectories[:, -1, :] - reference_trajectory[-1, :]
        for idx in self.angle_indices:
            errors[:, idx] = np.arctan2(
                np.sin(errors[:, idx]), np.cos(errors[:, idx])
            )
        if self.is_diagonal:
            return np.sum(errors**2 * self.Qf, axis=1)
        weighted = errors @ self.Qf
        return np.sum(weighted * errors, axis=1)


class CompositeMPPICost(CostFunction):
    """
    복합 비용 함수

    여러 비용 함수를 합성하여 총 비용 계산.

    Args:
        cost_functions: List[CostFunction]
    """

    def __init__(self, cost_functions: List[CostFunction]):
        self.cost_functions = cost_functions

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        복합 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,) 총 비용
        """
        K = trajectories.shape[0]
        total_costs = np.zeros(K)

        for cost_fn in self.cost_functions:
            total_costs += cost_fn.compute_cost(
                trajectories, controls, reference_trajectory
            )

        return total_costs

    def __repr__(self) -> str:
        return (
            f"CompositeMPPICost("
            f"{[cf.__class__.__name__ for cf in self.cost_functions]})"
        )


class EndEffectorTrackingCost(CostFunction):
    """
    End-Effector 추적 비용

    FK를 통해 EE 위치를 계산하고, reference의 [0:2] 인덱스를 EE 목표로 사용.

    J_ee = weight * Σ_{t=0}^{N-1} ||FK(x_t) - ee_ref_t||²

    Args:
        model: forward_kinematics(state) → (..., 2) 메서드를 가진 모델
        weight: 비용 가중치
    """

    def __init__(self, model, weight: float = 100.0):
        self.model = model
        self.weight = weight

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        EE 추적 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx) - ref[:, :2] = EE 목표 위치

        Returns:
            costs: (K,)
        """
        # FK: (K, N+1, nx) → (K, N+1, 2)
        ee_pos = self.model.forward_kinematics(trajectories)
        # EE reference: (N+1, 2) → 터미널 제외 (N, 2)
        ee_ref = reference_trajectory[:-1, :2]
        # 오차: (K, N, 2)
        errors = ee_pos[:, :-1, :] - ee_ref
        return self.weight * np.sum(errors**2, axis=(1, 2))


class EndEffectorTerminalCost(CostFunction):
    """
    End-Effector 터미널 비용

    J_terminal = weight * ||FK(x_N) - ee_ref_N||²

    Args:
        model: forward_kinematics(state) → (..., 2) 메서드를 가진 모델
        weight: 비용 가중치
    """

    def __init__(self, model, weight: float = 200.0):
        self.model = model
        self.weight = weight

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        EE 터미널 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        # FK at terminal: (K, nx) → (K, 2)
        ee_terminal = self.model.forward_kinematics(trajectories[:, -1, :])
        ee_ref = reference_trajectory[-1, :2]
        errors = ee_terminal - ee_ref
        return self.weight * np.sum(errors**2, axis=1)


class EndEffector3DTrackingCost(CostFunction):
    """
    End-Effector 3D 추적 비용

    FK를 통해 EE 3D 위치를 계산하고, reference의 [0:3] 인덱스를 EE 목표로 사용.

    J_ee = weight * Σ_{t=0}^{N-1} ||FK(x_t) - ee_ref_t||²

    Args:
        model: forward_kinematics(state) → (..., 3) 메서드를 가진 모델
        weight: 비용 가중치
    """

    def __init__(self, model, weight: float = 100.0):
        self.model = model
        self.weight = weight

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        # FK: (K, N+1, nx) → (K, N+1, 3)
        ee_pos = self.model.forward_kinematics(trajectories)
        # EE reference: (N+1, 3) → 터미널 제외 (N, 3)
        ee_ref = reference_trajectory[:-1, :3]
        # 오차: (K, N, 3)
        errors = ee_pos[:, :-1, :] - ee_ref
        return self.weight * np.sum(errors**2, axis=(1, 2))


class EndEffector3DTerminalCost(CostFunction):
    """
    End-Effector 3D 터미널 비용

    J_terminal = weight * ||FK(x_N) - ee_ref_N||²

    Args:
        model: forward_kinematics(state) → (..., 3) 메서드를 가진 모델
        weight: 비용 가중치
    """

    def __init__(self, model, weight: float = 200.0):
        self.model = model
        self.weight = weight

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        # FK at terminal: (K, nx) → (K, 3)
        ee_terminal = self.model.forward_kinematics(trajectories[:, -1, :])
        ee_ref = reference_trajectory[-1, :3]
        errors = ee_terminal - ee_ref
        return self.weight * np.sum(errors**2, axis=1)


class EndEffectorPoseTrackingCost(CostFunction):
    """
    End-Effector Pose 추적 비용 (위치 + 자세)

    FK_pose(trajectories) → (K, N+1, 6) [pos + RPY]
    pos_error = ||pos - ref[:3]||²
    ori_error = ||wrap(rpy - ref[3:6])||²

    J = pos_weight * Σ pos_error + ori_weight * Σ ori_error

    Args:
        model: forward_kinematics_pose(state) → (..., 6) 메서드를 가진 모델
        pos_weight: 위치 비용 가중치
        ori_weight: 자세 비용 가중치
    """

    def __init__(self, model, pos_weight: float = 100.0, ori_weight: float = 10.0):
        self.model = model
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        # FK pose: (K, N+1, nx) → (K, N+1, 6) [x, y, z, roll, pitch, yaw]
        ee_pose = self.model.forward_kinematics_pose(trajectories)

        # Position: (K, N, 3)
        pos_errors = ee_pose[:, :-1, :3] - reference_trajectory[:-1, :3]
        pos_cost = self.pos_weight * np.sum(pos_errors**2, axis=(1, 2))

        # Orientation: (K, N, 3) with angle wrapping
        ori_errors = ee_pose[:, :-1, 3:6] - reference_trajectory[:-1, 3:6]
        ori_errors = np.arctan2(np.sin(ori_errors), np.cos(ori_errors))
        ori_cost = self.ori_weight * np.sum(ori_errors**2, axis=(1, 2))

        return pos_cost + ori_cost


class EndEffectorPoseTerminalCost(CostFunction):
    """
    End-Effector Pose 터미널 비용 (위치 + 자세)

    Args:
        model: forward_kinematics_pose(state) → (..., 6) 메서드를 가진 모델
        pos_weight: 위치 비용 가중치
        ori_weight: 자세 비용 가중치
    """

    def __init__(self, model, pos_weight: float = 200.0, ori_weight: float = 20.0):
        self.model = model
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        # FK pose at terminal: (K, nx) → (K, 6)
        ee_pose = self.model.forward_kinematics_pose(trajectories[:, -1, :])

        # Position error
        pos_errors = ee_pose[:, :3] - reference_trajectory[-1, :3]
        pos_cost = self.pos_weight * np.sum(pos_errors**2, axis=1)

        # Orientation error with wrapping
        ori_errors = ee_pose[:, 3:6] - reference_trajectory[-1, 3:6]
        ori_errors = np.arctan2(np.sin(ori_errors), np.cos(ori_errors))
        ori_cost = self.ori_weight * np.sum(ori_errors**2, axis=1)

        return pos_cost + ori_cost


class JointLimitCost(CostFunction):
    """
    관절 제한 비용 (log-barrier)

    h(q) = (q_max - q)(q - q_min) > 0 일 때 안전
    cost = -weight * Σ log(h)   (h > 0)
    cost = penalty              (h ≤ 0, 제한 위반)

    Args:
        joint_indices: 관절 상태 인덱스 튜플
        joint_limits: ((q1_min, q1_max), (q2_min, q2_max), ...) 튜플
        weight: 비용 가중치
        penalty: 제한 위반 시 페널티
    """

    def __init__(
        self,
        joint_indices: tuple = (3, 4),
        joint_limits: tuple = ((-2.9, 2.9), (-2.9, 2.9)),
        weight: float = 10.0,
        penalty: float = 1e4,
    ):
        self.joint_indices = joint_indices
        self.joint_limits = joint_limits
        self.weight = weight
        self.penalty = penalty

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        관절 제한 비용 계산

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        K = trajectories.shape[0]
        costs = np.zeros(K)

        for idx, (q_min, q_max) in zip(self.joint_indices, self.joint_limits):
            q = trajectories[:, :, idx]  # (K, N+1)
            h = (q_max - q) * (q - q_min)  # (K, N+1)
            # log-barrier: -log(h) for h > 0, penalty for h <= 0
            safe = h > 0
            barrier = np.where(safe, -np.log(np.maximum(h, 1e-10)), self.penalty)
            costs += self.weight * np.sum(barrier, axis=1)

        return costs
