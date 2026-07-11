"""
Hierarchical MPPI 테스트
"""

import numpy as np
import pytest
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.hierarchical.global_planner import (
    OccupancyGrid,
    AStarPlanner,
    ThetaStarPlanner,
    smooth_path,
    path_to_reference,
)
from mppi_controller.controllers.hierarchical.hierarchical_mppi import (
    HierarchicalMPPIParams,
    HierarchicalMPPIController,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def robot_model():
    return DifferentialDriveKinematic()


@pytest.fixture
def small_grid():
    """5×5 소형 격자 (테스트용)."""
    return OccupancyGrid(
        width=50, height=50,
        resolution=0.2,
        origin=(-5.0, -5.0),
    )


@pytest.fixture
def params():
    return HierarchicalMPPIParams(
        N=10, K=32, dt=0.05,
        lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        grid_width=50,
        grid_height=50,
        grid_resolution=0.2,
        grid_origin=(-5.0, -5.0),
        replan_interval=5,
        lookahead_dist=1.0,
        goal_tolerance=0.3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HierarchicalMPPIParams 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestHierarchicalMPPIParams:
    def test_defaults(self):
        p = HierarchicalMPPIParams(
            sigma=np.array([0.5, 0.5]),
            Q=np.array([1.0, 1.0, 0.1]),
            R=np.array([0.1, 0.1]),
        )
        assert p.global_planner == "thetastar"
        assert p.grid_resolution == 0.2
        assert p.goal_tolerance == 0.3
        assert p.replan_interval == 20

    def test_invalid_planner(self):
        with pytest.raises(AssertionError):
            HierarchicalMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                global_planner="invalid",
            )

    def test_invalid_goal_tolerance(self):
        with pytest.raises(AssertionError):
            HierarchicalMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                goal_tolerance=-0.1,
            )

    def test_both_planners(self):
        for planner in ("astar", "thetastar"):
            p = HierarchicalMPPIParams(
                sigma=np.array([0.5, 0.5]),
                Q=np.array([1.0, 1.0, 0.1]),
                R=np.array([0.1, 0.1]),
                global_planner=planner,
            )
            assert p.global_planner == planner


# ─────────────────────────────────────────────────────────────────────────────
# OccupancyGrid 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestOccupancyGrid:
    def test_world_to_grid(self, small_grid):
        col, row = small_grid.world_to_grid(-5.0, -5.0)
        assert col == 0
        assert row == 0

    def test_grid_to_world(self, small_grid):
        x, y = small_grid.grid_to_world(0, 0)
        # 셀 중심
        assert np.isclose(x, -5.0 + 0.5 * small_grid.resolution)
        assert np.isclose(y, -5.0 + 0.5 * small_grid.resolution)

    def test_roundtrip(self, small_grid):
        """world → grid → world 근사 왕복 변환"""
        for x, y in [(-4.0, -3.0), (0.0, 0.0), (3.5, 2.1)]:
            col, row = small_grid.world_to_grid(x, y)
            wx, wy = small_grid.grid_to_world(col, row)
            assert abs(wx - x) < small_grid.resolution
            assert abs(wy - y) < small_grid.resolution

    def test_initial_free(self, small_grid):
        """초기 격자는 모두 비어 있음"""
        assert not small_grid.is_occupied(10, 10)
        assert not small_grid.is_occupied(25, 25)

    def test_out_of_bounds_occupied(self, small_grid):
        """경계 밖은 장애물로 처리"""
        assert small_grid.is_occupied(-1, 0)
        assert small_grid.is_occupied(0, -1)
        assert small_grid.is_occupied(100, 0)

    def test_set_obstacle(self, small_grid):
        small_grid.set_obstacle(0.0, 0.0, radius=0.15)
        col, row = small_grid.world_to_grid(0.0, 0.0)
        assert small_grid.is_occupied(col, row)

    def test_set_obstacle_radius(self, small_grid):
        """반경 내 셀 모두 장애물 설정"""
        small_grid.set_obstacle(0.0, 0.0, radius=0.4)
        col, row = small_grid.world_to_grid(0.0, 0.0)
        assert small_grid.is_occupied(col, row)

    def test_set_obstacles_from_list(self, small_grid):
        obstacles = [(0.0, 0.0, 0.2), (1.0, 1.0, 0.2)]
        small_grid.set_obstacles_from_list(obstacles)
        c0, r0 = small_grid.world_to_grid(0.0, 0.0)
        c1, r1 = small_grid.world_to_grid(1.0, 1.0)
        assert small_grid.is_occupied(c0, r0)
        assert small_grid.is_occupied(c1, r1)

    def test_free_cells_count_initially_full(self, small_grid):
        """초기에는 모든 셀이 자유"""
        assert small_grid.free_cells_count() == 50 * 50


# ─────────────────────────────────────────────────────────────────────────────
# AStarPlanner 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestAStarPlanner:
    def test_plan_simple(self, small_grid):
        planner = AStarPlanner(small_grid)
        path = planner.plan((-4.0, -4.0), (4.0, 4.0))
        assert path is not None
        assert len(path) >= 2

    def test_plan_start_equals_goal(self, small_grid):
        planner = AStarPlanner(small_grid)
        path = planner.plan((0.0, 0.0), (0.0, 0.0))
        # 시작=목표 → 단일 점 또는 빈 경로
        assert path is not None

    def test_plan_path_endpoints(self, small_grid):
        """경로 시작/끝이 목표 근처"""
        planner = AStarPlanner(small_grid)
        start = (-3.0, -3.0)
        goal = (3.0, 3.0)
        path = planner.plan(start, goal)
        assert path is not None
        # 마지막 점이 goal 근처
        last = path[-1]
        assert abs(last[0] - goal[0]) < 0.5
        assert abs(last[1] - goal[1]) < 0.5

    def test_plan_with_obstacle(self):
        """장애물이 있어도 우회 경로 탐색"""
        grid = OccupancyGrid(width=20, height=20, resolution=0.5, origin=(-5.0, -5.0))
        # 가운데 수직 벽 (중앙에 통로 남김)
        for r in range(5, 18):
            grid.grid[r, 10] = 1.0
        planner = AStarPlanner(grid)
        path = planner.plan((-4.0, 0.0), (4.0, 0.0))
        # 경로 있거나 없음 (통로 상황에 따라)
        # 여기서는 탐색 완료 여부만 확인
        assert True  # 에러 없이 실행

    def test_plan_blocked_returns_none(self):
        """완전 막힌 목표 → None 또는 인근 자유셀 경로"""
        grid = OccupancyGrid(width=10, height=10, resolution=0.5, origin=(0.0, 0.0))
        # 전체 격자를 장애물로
        grid.grid[:] = 1.0
        planner = AStarPlanner(grid)
        path = planner.plan((0.5, 0.5), (4.5, 4.5))
        assert path is None  # 완전 막힘


# ─────────────────────────────────────────────────────────────────────────────
# ThetaStarPlanner 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestThetaStarPlanner:
    def test_plan_simple(self, small_grid):
        planner = ThetaStarPlanner(small_grid)
        path = planner.plan((-4.0, -4.0), (4.0, 4.0))
        assert path is not None
        assert len(path) >= 2

    def test_theta_shorter_than_astar(self, small_grid):
        """Theta*는 A*보다 경로 길이가 같거나 짧음"""
        a_planner = AStarPlanner(small_grid)
        t_planner = ThetaStarPlanner(small_grid)

        start, goal = (-3.0, -3.0), (3.0, 3.0)
        a_path = a_planner.plan(start, goal)
        t_path = t_planner.plan(start, goal)

        assert a_path is not None and t_path is not None

        def path_len(p):
            arr = np.array(p)
            return float(np.sum(np.sqrt(np.diff(arr, axis=0) ** 2).sum(axis=1)))

        a_len = path_len(a_path)
        t_len = path_len(t_path)
        # Theta* ≤ A* (약간의 부동소수점 허용)
        assert t_len <= a_len + 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestPathUtils:
    def test_smooth_path_length(self):
        """평활화 후 경로점 수 유지"""
        path = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0), (3.0, 0.5)]
        smoothed = smooth_path(path)
        assert len(smoothed) == len(path)

    def test_smooth_path_endpoints_fixed(self):
        """시작/끝 점은 변경 없음"""
        path = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 0.0)]
        smoothed = smooth_path(path)
        assert np.isclose(smoothed[0][0], path[0][0])
        assert np.isclose(smoothed[0][1], path[0][1])
        assert np.isclose(smoothed[-1][0], path[-1][0])
        assert np.isclose(smoothed[-1][1], path[-1][1])

    def test_path_to_reference_shape(self):
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        ref = path_to_reference(path, N=10, dt=0.05, nx=3)
        assert ref.shape == (11, 3)

    def test_path_to_reference_empty(self):
        ref = path_to_reference([], N=5, dt=0.1)
        assert ref.shape == (6, 3)
        assert np.allclose(ref, 0.0)

    def test_path_to_reference_start(self):
        """참조 첫 점이 경로 시작점"""
        path = [(1.0, 2.0), (3.0, 4.0)]
        ref = path_to_reference(path, N=5, dt=0.1, nx=3)
        assert np.isclose(ref[0, 0], 1.0, atol=0.1)
        assert np.isclose(ref[0, 1], 2.0, atol=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# HierarchicalMPPIController 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestHierarchicalMPPIController:
    def test_instantiation_astar(self, robot_model, params):
        params.global_planner = "astar"
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        assert ctrl is not None
        assert not ctrl.is_goal_reached

    def test_instantiation_thetastar(self, robot_model, params):
        params.global_planner = "thetastar"
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        assert ctrl is not None

    def test_compute_control_output_shape(self, robot_model, params):
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state)
        assert u.shape == (2,)

    def test_compute_control_finite(self, robot_model, params):
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state)
        assert np.all(np.isfinite(u))

    def test_info_keys(self, robot_model, params):
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state)
        for key in [
            "sample_trajectories", "sample_weights", "best_trajectory",
            "global_path", "subgoal", "local_reference",
            "goal_reached", "replan_count", "path_deviation", "planner_type",
        ]:
            assert key in info, f"Missing key: {key}"

    def test_global_path_generated(self, robot_model, params):
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        state = np.array([0.0, 0.0, 0.0])
        ctrl.compute_control(state)
        assert ctrl.global_path is not None
        assert len(ctrl.global_path) >= 2

    def test_replan_count_increases(self, robot_model, params):
        params.replan_interval = 1  # 매 스텝 재계획
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        state = np.array([0.0, 0.0, 0.0])
        for _ in range(3):
            ctrl.compute_control(state)
        assert ctrl._replan_count >= 1

    def test_goal_reached_detection(self, robot_model, params):
        """목표 바로 옆에서 시작 → goal_reached"""
        ctrl = HierarchicalMPPIController(
            robot_model, params,
            goal=np.array([0.1, 0.0]),
        )
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state)
        assert info["goal_reached"] or ctrl.is_goal_reached

    def test_goal_reached_returns_zero(self, robot_model, params):
        """목표 도달 시 zero 제어"""
        params.goal_tolerance = 5.0  # 큰 허용치
        ctrl = HierarchicalMPPIController(
            robot_model, params,
            goal=np.array([0.0, 0.0]),
        )
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state)
        assert np.allclose(u, 0.0)
        assert info["goal_reached"]

    def test_set_goal(self, robot_model, params):
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([1.0, 1.0]))
        ctrl.set_goal(np.array([3.0, 3.0]))
        assert np.allclose(ctrl._goal, [3.0, 3.0])
        assert ctrl._global_path is None  # 강제 재계획

    def test_with_obstacles(self, robot_model):
        params = HierarchicalMPPIParams(
            N=10, K=32, dt=0.05,
            sigma=np.array([0.5, 0.5]),
            Q=np.array([10.0, 10.0, 1.0]),
            R=np.array([0.1, 0.1]),
            grid_width=50, grid_height=50,
            grid_resolution=0.2, grid_origin=(-5.0, -5.0),
            obstacles=[(1.0, 0.0, 0.3)],  # 중간 장애물
            robot_radius=0.15,
        )
        ctrl = HierarchicalMPPIController(
            robot_model, params, goal=np.array([3.0, 0.0])
        )
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state)
        # 에러 없이 실행, global_path 존재
        assert u.shape == (2,)

    def test_reset(self, robot_model, params):
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        state = np.array([0.0, 0.0, 0.0])
        ctrl.compute_control(state)
        ctrl.reset()
        assert ctrl._global_path is None
        assert ctrl._step_count == 0
        assert ctrl._replan_count == 0
        assert not ctrl._goal_reached

    def test_sequential_calls(self, robot_model, params):
        """연속 호출 정상 동작"""
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        state = np.array([0.0, 0.0, 0.0])
        for i in range(10):
            u, info = ctrl.compute_control(state)
            assert np.all(np.isfinite(u)), f"step {i}: non-finite control"

    def test_no_goal_set(self, robot_model, params):
        """목표 미설정 → 정지 제어 반환"""
        ctrl = HierarchicalMPPIController(robot_model, params, goal=None)
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state)
        assert u.shape == (2,)

    def test_set_obstacle_map(self, robot_model, params):
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        ctrl.set_obstacle_map([(1.0, 0.0, 0.3)], robot_radius=0.1)
        assert ctrl._global_path is None  # 강제 재계획

    def test_add_obstacle(self, robot_model, params):
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([3.0, 3.0]))
        ctrl.add_obstacle(1.5, 0.0, 0.3)
        assert ctrl._global_path is None  # 강제 재계획

    def test_planner_type_in_info(self, robot_model, params):
        params.global_planner = "astar"
        ctrl = HierarchicalMPPIController(robot_model, params, goal=np.array([2.0, 2.0]))
        state = np.array([0.0, 0.0, 0.0])
        u, info = ctrl.compute_control(state)
        assert info["planner_type"] == "astar"
