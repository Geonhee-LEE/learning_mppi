"""
Hierarchical MPPI — 2단계 계획 컨트롤러

글로벌 A*/Theta* 경로 + 로컬 MPPI 추적을 결합한 장거리 내비게이션.

아키텍처:
    ┌─────────────────────────────────────────┐
    │  Hierarchical MPPI Controller           │
    │                                         │
    │  [글로벌 레이어] A* / Theta*             │
    │    맵 + 목표 → 글로벌 경로               │
    │         ↓ 서브목표 선택 (lookahead)       │
    │  [로컬 레이어] MPPI                      │
    │    서브목표 참조 → 제어 출력              │
    └─────────────────────────────────────────┘

재계획 조건:
    - 주기적 (replan_interval 스텝마다)
    - 현재 경로를 이탈했을 때 (path_deviation_threshold 초과)
    - 목표 변경 시

References:
    Agha-mohammadi et al. (2014) — FIRM: Sampling-based feedback control
    Williams et al. (2018) — Information Theoretic MPC
    Hierarchical planning 일반적 방법론
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.controllers.hierarchical.global_planner import (
    OccupancyGrid,
    AStarPlanner,
    ThetaStarPlanner,
    smooth_path,
    path_to_reference,
)


# ─────────────────────────────────────────────────────────────────────────────
# 파라미터
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HierarchicalMPPIParams(MPPIParams):
    """
    Hierarchical MPPI 파라미터

    글로벌 경로계획 + 로컬 MPPI 추적에 필요한 파라미터.

    Attributes:
        global_planner: 경로계획 알고리즘 ("astar" | "thetastar")
        grid_resolution: 점유 격자 해상도 (m/cell)
        grid_width: 격자 가로 셀 수
        grid_height: 격자 세로 셀 수
        grid_origin: 격자 원점 (x, y) [m]
        robot_radius: 로봇 반경 (장애물 팽창용, m)
        goal_tolerance: 목표 도달 판정 거리 (m)
        lookahead_dist: 로컬 서브목표 lookahead 거리 (m)
        replan_interval: 재계획 주기 (스텝 수)
        path_deviation_threshold: 경로 이탈 판정 거리 (m)
        smooth_path: 글로벌 경로 평활화 여부
        smooth_weight: 평활화 가중치
        v_desired: 경로 추적 목표 속도 (m/s)
        obstacles: 초기 장애물 리스트 [(x, y, radius), ...]
    """

    # 글로벌 계획
    global_planner: str = "thetastar"
    grid_resolution: float = 0.2
    grid_width: int = 100
    grid_height: int = 100
    grid_origin: Tuple[float, float] = (-10.0, -10.0)
    robot_radius: float = 0.2
    goal_tolerance: float = 0.3
    lookahead_dist: float = 1.5
    replan_interval: int = 20
    path_deviation_threshold: float = 0.8
    smooth_path: bool = True
    smooth_weight: float = 0.3
    v_desired: float = 0.5
    obstacles: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert self.global_planner in ("astar", "thetastar"), \
            f"global_planner must be 'astar' or 'thetastar', got {self.global_planner!r}"
        assert self.grid_resolution > 0, "grid_resolution must be positive"
        assert self.grid_width > 0 and self.grid_height > 0
        assert self.goal_tolerance > 0, "goal_tolerance must be positive"
        assert self.lookahead_dist > 0, "lookahead_dist must be positive"
        assert self.replan_interval >= 1, "replan_interval must be >= 1"
        assert self.path_deviation_threshold > 0


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical MPPI 컨트롤러
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalMPPIController:
    """
    Hierarchical MPPI 컨트롤러.

    2단계 구조:
        1. 글로벌 레이어: A*/Theta*로 장애물을 피한 글로벌 경로 생성
        2. 로컬 레이어: MPPI로 lookahead 서브목표까지 최적 제어 계산

    워크플로우:
        ctrl = HierarchicalMPPIController(model, params, goal=[5.0, 3.0])
        ctrl.set_obstacle_map(obstacles)   # 선택적
        u, info = ctrl.compute_control(state, reference)

    Args:
        model: RobotModel 인스턴스
        params: HierarchicalMPPIParams
        goal: (x, y) 또는 (x, y, θ) 최종 목표
        cost_function: 로컬 MPPI 비용함수 (None이면 기본 비용)
        noise_sampler: 로컬 MPPI 노이즈 샘플러
    """

    def __init__(
        self,
        model: RobotModel,
        params: HierarchicalMPPIParams,
        goal: Optional[np.ndarray] = None,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        self.model = model
        self.params = params

        # 로컬 MPPI (서브목표 추적)
        self.local_mppi = MPPIController(
            model=model,
            params=params,
            cost_function=cost_function,
            noise_sampler=noise_sampler,
        )

        # 점유 격자 초기화
        self.grid = OccupancyGrid(
            width=params.grid_width,
            height=params.grid_height,
            resolution=params.grid_resolution,
            origin=params.grid_origin,
        )
        if params.obstacles:
            self.grid.set_obstacles_from_list(params.obstacles)
            if params.robot_radius > 0:
                self.grid.add_inflation(params.robot_radius)

        # 경로계획기
        if params.global_planner == "astar":
            self.planner = AStarPlanner(self.grid)
        else:
            self.planner = ThetaStarPlanner(self.grid)

        # 상태
        self._goal: Optional[np.ndarray] = (
            np.array(goal, dtype=float) if goal is not None else None
        )
        self._global_path: Optional[List[Tuple[float, float]]] = None
        self._path_ref: Optional[np.ndarray] = None  # (N+1, nx)
        self._step_count: int = 0
        self._replan_count: int = 0
        self._goal_reached: bool = False

        # 서브목표 인덱스
        self._subgoal_idx: int = 0

    # ── 상태 설정 ──────────────────────────────────────────────────────────────

    def set_goal(self, goal: np.ndarray) -> None:
        """최종 목표 업데이트 → 강제 재계획."""
        self._goal = np.array(goal, dtype=float)
        self._global_path = None
        self._goal_reached = False

    def set_obstacle_map(self, obstacles: list, robot_radius: float = 0.0) -> None:
        """
        장애물 맵 업데이트 → 격자 초기화 + 재계획.

        Args:
            obstacles: [(x, y, radius), ...] 장애물 리스트
            robot_radius: 로봇 반경 (팽창용)
        """
        self.grid = OccupancyGrid(
            width=self.params.grid_width,
            height=self.params.grid_height,
            resolution=self.params.grid_resolution,
            origin=self.params.grid_origin,
        )
        self.grid.set_obstacles_from_list(obstacles)
        r = robot_radius if robot_radius > 0 else self.params.robot_radius
        if r > 0:
            self.grid.add_inflation(r)

        # 경로계획기 맵 업데이트
        self.planner = (
            AStarPlanner(self.grid)
            if self.params.global_planner == "astar"
            else ThetaStarPlanner(self.grid)
        )
        self._global_path = None  # 강제 재계획

    def add_obstacle(self, x: float, y: float, radius: float = 0.1) -> None:
        """단일 장애물 추가 → 강제 재계획."""
        self.grid.set_obstacle(x, y, radius)
        self._global_path = None

    # ── 글로벌 계획 ────────────────────────────────────────────────────────────

    def _replan(self, state: np.ndarray) -> bool:
        """
        A*/Theta*로 글로벌 경로 재계획.

        Returns:
            success: 경로 탐색 성공 여부
        """
        if self._goal is None:
            return False

        start_xy = (float(state[0]), float(state[1]))
        goal_xy = (float(self._goal[0]), float(self._goal[1]))

        path = self.planner.plan(start_xy, goal_xy)
        if path is None or len(path) < 2:
            return False

        # 경로 평활화
        if self.params.smooth_path and len(path) >= 3:
            path = smooth_path(path, weight_smooth=self.params.smooth_weight)

        self._global_path = path
        self._subgoal_idx = 0
        self._replan_count += 1
        return True

    def _needs_replan(self, state: np.ndarray) -> bool:
        """재계획 필요 여부 판단."""
        if self._global_path is None:
            return True
        if self._step_count % self.params.replan_interval == 0:
            return True
        # 경로 이탈 확인
        deviation = self._path_deviation(state)
        if deviation > self.params.path_deviation_threshold:
            return True
        return False

    def _path_deviation(self, state: np.ndarray) -> float:
        """현재 상태와 글로벌 경로 간 최소 거리."""
        if self._global_path is None:
            return float("inf")
        pos = np.array([state[0], state[1]])
        path_arr = np.array(self._global_path)
        dists = np.sqrt(((path_arr - pos) ** 2).sum(axis=1))
        return float(dists.min())

    # ── 서브목표 선택 ──────────────────────────────────────────────────────────

    def _select_subgoal(self, state: np.ndarray) -> np.ndarray:
        """
        Lookahead 거리 기반 서브목표 선택.

        현재 위치에서 lookahead_dist 앞의 경로점을 서브목표로.

        Returns:
            subgoal: (x, y) 서브목표 좌표
        """
        if self._global_path is None or len(self._global_path) == 0:
            return self._goal[:2] if self._goal is not None else np.zeros(2)

        pos = np.array([state[0], state[1]])
        path_arr = np.array(self._global_path)

        # 가장 가까운 경로점 인덱스
        dists = np.sqrt(((path_arr - pos) ** 2).sum(axis=1))
        nearest_idx = int(np.argmin(dists))

        # Lookahead 거리 이후 경로점 탐색
        lookahead = self.params.lookahead_dist
        for i in range(nearest_idx, len(self._global_path)):
            d = np.sqrt(
                (self._global_path[i][0] - pos[0]) ** 2
                + (self._global_path[i][1] - pos[1]) ** 2
            )
            if d >= lookahead:
                return np.array(self._global_path[i])

        # 마지막 경로점 반환
        return np.array(self._global_path[-1])

    def _build_local_reference(
        self, state: np.ndarray, subgoal: np.ndarray
    ) -> np.ndarray:
        """
        서브목표까지의 로컬 참조 궤적 생성.

        서브목표를 향한 등속 직선 경로를 MPPI 참조로 변환.

        Args:
            state: (nx,) 현재 상태
            subgoal: (2,) 서브목표 좌표

        Returns:
            ref: (N+1, nx) 로컬 참조 궤적
        """
        N = self.params.N
        nx = self.model.state_dim
        dt = self.params.dt
        v = self.params.v_desired

        pos = np.array([state[0], state[1]])
        direction = subgoal - pos
        dist = np.linalg.norm(direction)

        ref = np.zeros((N + 1, nx))

        if dist < 1e-6:
            # 이미 서브목표 도달
            ref[:] = state[:nx]
            return ref

        unit = direction / dist
        theta = np.arctan2(unit[1], unit[0])

        for i in range(N + 1):
            t = i * dt
            travel = min(v * t, dist)
            ref[i, 0] = pos[0] + unit[0] * travel
            ref[i, 1] = pos[1] + unit[1] * travel
            if nx > 2:
                ref[i, 2] = theta

        return ref

    # ── 제어 계산 ──────────────────────────────────────────────────────────────

    def compute_control(
        self,
        state: np.ndarray,
        reference_trajectory: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Hierarchical MPPI 제어 계산.

        1. 글로벌 경로 재계획 (필요 시)
        2. 서브목표 선택 (Lookahead)
        3. 로컬 MPPI로 서브목표 추적 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: 무시됨 (글로벌 경로가 참조 역할)
                                   외부 참조 제공 시 서브목표 오버라이드

        Returns:
            control: (nu,) 최적 제어
            info: dict {
                sample_trajectories, sample_weights, best_trajectory,
                global_path, subgoal, local_reference,
                goal_reached, replan_count, path_deviation,
                planner_type, step_count
            }
        """
        self._step_count += 1
        nu = self.model.control_dim

        # 목표 도달 확인
        if self._goal is not None:
            dist_to_goal = np.sqrt(
                (state[0] - self._goal[0]) ** 2
                + (state[1] - self._goal[1]) ** 2
            )
            if dist_to_goal <= self.params.goal_tolerance:
                self._goal_reached = True

        if self._goal_reached:
            info = {
                "sample_trajectories": np.zeros((self.params.K, self.params.N + 1, self.model.state_dim)),
                "sample_weights": np.ones(self.params.K) / self.params.K,
                "best_trajectory": np.tile(state, (self.params.N + 1, 1)),
                "global_path": self._global_path,
                "subgoal": self._goal[:2] if self._goal is not None else None,
                "local_reference": None,
                "goal_reached": True,
                "replan_count": self._replan_count,
                "path_deviation": 0.0,
                "planner_type": self.params.global_planner,
                "step_count": self._step_count,
                "ess": float(self.params.K),
                "temperature": self.params.lambda_,
                "num_samples": self.params.K,
            }
            return np.zeros(nu), info

        # 글로벌 계획
        plan_success = True
        if self._needs_replan(state):
            plan_success = self._replan(state)

        # 서브목표 선택
        if reference_trajectory is not None:
            # 외부 참조 우선
            local_ref = reference_trajectory
            subgoal = np.array([reference_trajectory[-1, 0], reference_trajectory[-1, 1]])
        elif plan_success and self._global_path is not None:
            subgoal = self._select_subgoal(state)
            local_ref = self._build_local_reference(state, subgoal)
        else:
            # 계획 실패 → 정지
            subgoal = np.array([state[0], state[1]])
            local_ref = np.tile(state[:self.model.state_dim], (self.params.N + 1, 1))

        # 로컬 MPPI 제어 계산
        control, mppi_info = self.local_mppi.compute_control(state, local_ref)

        path_dev = self._path_deviation(state)

        info = {
            **mppi_info,
            "global_path": self._global_path,
            "subgoal": subgoal,
            "local_reference": local_ref,
            "goal_reached": False,
            "replan_count": self._replan_count,
            "path_deviation": path_dev,
            "planner_type": self.params.global_planner,
            "step_count": self._step_count,
            "plan_success": plan_success,
        }

        return control, info

    # ── 편의 메서드 ────────────────────────────────────────────────────────────

    @property
    def is_goal_reached(self) -> bool:
        return self._goal_reached

    @property
    def global_path(self) -> Optional[List[Tuple[float, float]]]:
        return self._global_path

    def reset(self) -> None:
        """컨트롤러 상태 초기화."""
        self._global_path = None
        self._path_ref = None
        self._step_count = 0
        self._replan_count = 0
        self._goal_reached = False
        self._subgoal_idx = 0
        self.local_mppi.U = np.zeros_like(self.local_mppi.U)
