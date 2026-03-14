"""
WBC-MPPI (Whole-Body Control MPPI) 컨트롤러

MobileManipulator6DOFKinematic (9D 상태, 8D 제어)를 사용하는
베이스 + 팔 통합 최적화 MPPI 컨트롤러.

알고리즘 개요:
    1. 기존 Vanilla MPPI 파이프라인 유지 (상속)
    2. 비용 함수: EE 추적(SE3) + 관절 한계 + 특이점 회피 + 도달가능성
    3. 분리된 sigma: 베이스(v,ω)와 팔(dq1..6) 별도 노이즈 수준
    4. 동적 EE 목표 업데이트 (set_ee_target)
    5. 작업 모드: ee_tracking / navigation / both

참조:
    - WBC 개념: Sentis & Khatib (2005) "A whole-body control framework"
    - MPPI: Williams et al. (2016, 2018)
    - SE(3) geodesic: Buss (2004) "Introduction to Inverse Kinematics"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import WBCMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CostFunction,
    CompositeMPPICost,
    ControlEffortCost,
    TerminalCost,
)
from mppi_controller.controllers.mppi.se3_cost import (
    SE3TrackingCost,
    SE3TerminalCost,
    SE3ManipulabilityCost,
)
from mppi_controller.controllers.mppi.manipulation_costs import (
    ArmSingularityAvoidanceCost,
    ReachabilityWorkspaceCost,
    WBCBaseNavigationCost,
    JointVelocitySmoothCost,
    GraspApproachCost,
    CollisionFreeSweepCost,
)
from mppi_controller.controllers.mppi.sampling import GaussianSampler, NoiseSampler


# 6-DOF 모바일 매니퓰레이터 기본 관절 한계 (UR5-style, rad)
DEFAULT_JOINT_LIMITS_6DOF = [
    (-np.pi, np.pi),     # q1: shoulder yaw
    (-np.pi, 0.0),       # q2: shoulder pitch
    (-2.5, 2.5),         # q3: elbow pitch
    (-np.pi, np.pi),     # q4: wrist 1
    (-np.pi, np.pi),     # q5: wrist 2
    (-np.pi, np.pi),     # q6: wrist 3
]


class WBCNoiseSampler(NoiseSampler):
    """
    WBC 전용 노이즈 샘플러.

    베이스(v, ω)와 팔(dq1..6) 관절에 별도의 노이즈 수준 적용.
    EE 작업 중 베이스 노이즈를 줄이거나, 팔 탐색을 강화할 수 있음.

    Args:
        sigma_base: 베이스 제어(v, ω)의 노이즈 표준편차 [σ_v, σ_ω]
        sigma_arm: 팔 관절 속도의 노이즈 표준편차 (스칼라 또는 6차원)
        task_mode: "ee_tracking": 베이스 노이즈 ↓, 팔 노이즈 ↑
                   "navigation": 베이스 노이즈 ↑, 팔 노이즈 ↓
                   "both": 균등 노이즈
    """

    def __init__(
        self,
        sigma_base: np.ndarray = np.array([0.3, 0.3]),
        sigma_arm: np.ndarray = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        task_mode: str = "ee_tracking",
    ):
        self.sigma_base = np.asarray(sigma_base, dtype=np.float64)
        self.sigma_arm = (
            np.ones(6) * sigma_arm
            if np.isscalar(sigma_arm)
            else np.asarray(sigma_arm, dtype=np.float64)
        )
        self.task_mode = task_mode

        # 작업 모드별 노이즈 스케일 조정
        if task_mode == "ee_tracking":
            self.sigma_base = self.sigma_base * 0.5  # 베이스 노이즈 감소
        elif task_mode == "navigation":
            self.sigma_arm = self.sigma_arm * 0.3   # 팔 노이즈 감소

        # 통합 sigma: [σ_v, σ_ω, σ_dq1, ..., σ_dq6]
        self.sigma = np.concatenate([self.sigma_base, self.sigma_arm])

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 수

        Returns:
            noise: (K, N, nu)
        """
        N, nu = U.shape
        noise = np.random.randn(K, N, nu) * self.sigma[None, None, :]
        return noise


class WBCMPPIController(MPPIController):
    """
    WBC-MPPI (Whole-Body Control MPPI) 컨트롤러

    MobileManipulator6DOFKinematic과 함께 사용하는 전신 제어 MPPI.
    EE 위치/자세 추적 + 관절 한계 + 특이점 회피 + 도달가능성을 통합 최적화.

    사용 방법:
        model = MobileManipulator6DOFKinematic()
        params = WBCMPPIParams(N=20, K=512, ee_pos_weight=100.0)
        ctrl = WBCMPPIController(
            model, params,
            ee_target_pos=np.array([1.0, 0.5, 0.8]),
            ee_target_rpy=np.array([0.0, -np.pi/2, 0.0]),
        )

        state = np.zeros(9)
        ref = np.zeros((params.N+1, 9))  # 베이스 참조 궤적
        u, info = ctrl.compute_control(state, ref)

    Args:
        model: MobileManipulator6DOFKinematic 또는 동일 인터페이스 모델
        params: WBCMPPIParams
        ee_target_pos: 목표 EE 위치 [x, y, z]
        ee_target_rpy: 목표 EE 자세 [roll, pitch, yaw]
        joint_limits: 관절 한계 리스트 [(q_min, q_max), ...]
        obstacles: 장애물 리스트 [(cx, cy, radius), ...] — 충돌 검사용
        noise_sampler: 사용자 정의 노이즈 샘플러 (None이면 WBCNoiseSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: WBCMPPIParams,
        ee_target_pos: Optional[np.ndarray] = None,
        ee_target_rpy: Optional[np.ndarray] = None,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        self._wbc_params = params
        self._ee_target_pos = (
            np.asarray(ee_target_pos, dtype=np.float64)
            if ee_target_pos is not None
            else None
        )
        self._ee_target_rpy = (
            np.asarray(ee_target_rpy, dtype=np.float64)
            if ee_target_rpy is not None
            else None
        )
        self._joint_limits = joint_limits if joint_limits is not None else DEFAULT_JOINT_LIMITS_6DOF
        self._obstacles = obstacles or []

        # 비용 함수 구성
        cost_fn = self._build_cost_function(model, params)

        # WBC 전용 노이즈 샘플러
        if noise_sampler is None:
            noise_sampler = WBCNoiseSampler(
                sigma_base=np.array([params.sigma[0], params.sigma[1]])
                if len(params.sigma) >= 2
                else np.array([0.3, 0.3]),
                sigma_arm=params.sigma[2:8]
                if len(params.sigma) >= 8
                else np.ones(6) * 0.5,
                task_mode=params.task_mode,
            )

        super().__init__(model, params, cost_function=cost_fn, noise_sampler=noise_sampler)

    def _build_cost_function(
        self, model: RobotModel, params: WBCMPPIParams
    ) -> CompositeMPPICost:
        """WBC 비용 함수 구성."""
        costs: List[CostFunction] = []

        # 1. EE 위치 + 자세 추적 비용 (SE3)
        if (
            self._ee_target_pos is not None
            and params.task_mode in ("ee_tracking", "both")
        ):
            costs.append(
                SE3TrackingCost(
                    model=model,
                    pos_weight=params.ee_pos_weight,
                    ori_weight=params.ee_ori_weight,
                    target_pos=self._ee_target_pos,
                    target_rpy=self._ee_target_rpy if self._ee_target_rpy is not None
                    else np.zeros(3),
                )
            )
            costs.append(
                SE3TerminalCost(
                    model=model,
                    target_pos=self._ee_target_pos,
                    target_rpy=self._ee_target_rpy if self._ee_target_rpy is not None
                    else np.zeros(3),
                    pos_weight=params.ee_terminal_pos_weight,
                    ori_weight=params.ee_terminal_ori_weight,
                )
            )

        # 2. 관절 한계 비용 (6-DOF 팔: 상태 인덱스 3~8)
        if self._joint_limits:
            from mppi_controller.controllers.mppi.cost_functions import JointLimitCost
            joint_indices = tuple(range(3, 3 + len(self._joint_limits)))
            costs.append(
                JointLimitCost(
                    joint_indices=joint_indices,
                    joint_limits=self._joint_limits,
                    weight=params.joint_limit_weight,
                    penalty=params.joint_penalty,
                )
            )

        # 3. 특이점 회피 비용
        if params.singularity_weight > 0:
            costs.append(
                ArmSingularityAvoidanceCost(
                    model=model,
                    joint_indices=list(range(3, 9)),
                    threshold=params.singularity_threshold,
                    weight=params.singularity_weight,
                )
            )

        # 4. 도달가능성 비용
        if (
            self._ee_target_pos is not None
            and params.reachability_weight > 0
        ):
            costs.append(
                ReachabilityWorkspaceCost(
                    target_pos=self._ee_target_pos,
                    max_reach=params.max_arm_reach,
                    min_reach=params.min_arm_reach,
                    reach_weight=params.reachability_weight,
                )
            )

        # 5. 베이스 속도 비용 (EE 추적 중 베이스 안정화)
        if params.base_vel_weight > 0 and params.task_mode == "ee_tracking":
            costs.append(
                WBCBaseNavigationCost(
                    base_vel_weight=params.base_vel_weight,
                    base_omega_weight=params.base_vel_weight,
                )
            )

        # 6. 팔 관절 속도 부드러움 비용
        if params.smooth_weight > 0:
            costs.append(
                JointVelocitySmoothCost(
                    joint_ctrl_indices=list(range(2, 8)),
                    smooth_weight=params.smooth_weight,
                )
            )

        # 7. 제어 노력 비용 (팔 관절 속도)
        if params.arm_effort_weight > 0:
            R_wbc = np.concatenate([
                np.array([params.base_vel_weight, params.base_vel_weight]),
                np.ones(6) * params.arm_effort_weight,
            ])
            costs.append(ControlEffortCost(R_wbc))

        # 8. 충돌 비용 (장애물 지정 시)
        if self._obstacles:
            costs.append(
                CollisionFreeSweepCost(
                    model=model,
                    obstacles=self._obstacles,
                )
            )

        return CompositeMPPICost(costs)

    def set_ee_target(
        self,
        pos: np.ndarray,
        rpy: Optional[np.ndarray] = None,
    ) -> None:
        """
        동적 EE 목표 업데이트.

        compute_control() 호출 전에 사용하여 실시간으로 EE 목표 변경.

        Args:
            pos: 목표 EE 위치 [x, y, z]
            rpy: 목표 EE 자세 [roll, pitch, yaw] (None이면 자세 미지정)
        """
        self._ee_target_pos = np.asarray(pos, dtype=np.float64)
        if rpy is not None:
            self._ee_target_rpy = np.asarray(rpy, dtype=np.float64)

        # 비용 함수 재구성
        self.cost_function = self._build_cost_function(self.model, self._wbc_params)

    def set_obstacles(
        self,
        obstacles: List[Tuple[float, float, float]],
    ) -> None:
        """
        장애물 목록 업데이트.

        Args:
            obstacles: [(cx, cy, radius), ...] — 2D 원형 장애물
        """
        self._obstacles = obstacles
        self.cost_function = self._build_cost_function(self.model, self._wbc_params)

    def compute_control(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        WBC-MPPI 제어 계산.

        Args:
            state: (9,) 현재 상태 [x, y, θ, q1..q6]
            reference_trajectory: (N+1, ≥3) 베이스 참조 궤적
                첫 2열: (x, y) 베이스 위치 참조
                나머지: 사용하지 않음 (EE 목표는 params에서)

        Returns:
            control: (8,) [v, ω, dq1..dq6]
            info: {
                'sample_trajectories': (K, N+1, 9),
                'sample_weights': (K,),
                'best_trajectory': (N+1, 9),
                'ee_position': (3,) 현재 EE 위치,
                'ee_rpy': (6,) 현재 EE pose,
                'ess': float,
                'temperature': float,
                'task_mode': str,
            }
        """
        control, info = super().compute_control(state, reference_trajectory)

        # 현재 EE 위치/자세 추가
        if hasattr(self.model, "forward_kinematics"):
            info["ee_position"] = self.model.forward_kinematics(state[None])[0]
        if hasattr(self.model, "forward_kinematics_pose"):
            info["ee_pose"] = self.model.forward_kinematics_pose(state[None])[0]
        info["task_mode"] = self._wbc_params.task_mode
        info["ee_target_pos"] = self._ee_target_pos

        return control, info

    @property
    def ee_target_pos(self) -> Optional[np.ndarray]:
        """현재 EE 목표 위치."""
        return self._ee_target_pos

    @property
    def ee_target_rpy(self) -> Optional[np.ndarray]:
        """현재 EE 목표 자세 (RPY)."""
        return self._ee_target_rpy
