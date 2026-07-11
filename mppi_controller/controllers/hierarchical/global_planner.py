"""
글로벌 경로계획기 — A* / Theta*

2D 점유 격자(Occupancy Grid)에서 최단 경로 탐색.
Hierarchical MPPI의 글로벌 계획 레이어.

A*:     격자 기반 최단 경로 (8-방향)
Theta*: Any-Angle A* — 직선 시야(Line-of-Sight) 최적화로 부드러운 경로

References:
    Hart et al. (1968) — A* Search Algorithm
    Nash et al. (2007) — Theta*: Any-Angle Path Planning
"""

import heapq
import numpy as np
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 점유 격자
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyGrid:
    """
    2D 점유 격자 맵.

    셀 값: 0 = 빈 공간, 1 = 장애물 (또는 [0,1] 확률값).

    Args:
        width: 격자 가로 셀 수
        height: 격자 세로 셀 수
        resolution: 격자 해상도 (m/cell)
        origin: 격자 원점 (x, y) [m]  — 격자 (0,0) 셀의 실제 좌표
        obstacle_threshold: 점유 확률 임계값 (이상이면 장애물 처리)
    """

    def __init__(
        self,
        width: int,
        height: int,
        resolution: float = 0.1,
        origin: Tuple[float, float] = (0.0, 0.0),
        obstacle_threshold: float = 0.5,
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin, dtype=float)
        self.obstacle_threshold = obstacle_threshold

        self.grid = np.zeros((height, width), dtype=float)

    # ── 좌표 변환 ──────────────────────────────────────────────────────────────

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        실세계 좌표 → 격자 인덱스 (col, row).

        Returns:
            (col, row) — 격자 셀 인덱스
        """
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """격자 인덱스 → 실세계 좌표 (셀 중심)."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return x, y

    def is_in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def is_occupied(self, col: int, row: int) -> bool:
        if not self.is_in_bounds(col, row):
            return True  # 경계 밖은 장애물로 처리
        return self.grid[row, col] >= self.obstacle_threshold

    # ── 격자 수정 ──────────────────────────────────────────────────────────────

    def set_obstacle(self, x: float, y: float, radius: float = 0.0) -> None:
        """
        실세계 좌표에 원형 장애물 추가.

        Args:
            x, y: 장애물 중심 (실세계)
            radius: 장애물 반경 (m)
        """
        cx, cy = self.world_to_grid(x, y)
        r_cells = max(1, int(radius / self.resolution))
        for dr in range(-r_cells - 1, r_cells + 2):
            for dc in range(-r_cells - 1, r_cells + 2):
                row, col = cy + dr, cx + dc
                if self.is_in_bounds(col, row):
                    wx, wy = self.grid_to_world(col, row)
                    if (wx - x) ** 2 + (wy - y) ** 2 <= radius ** 2:
                        self.grid[row, col] = 1.0

    def set_obstacles_from_list(self, obstacles: list) -> None:
        """
        장애물 리스트 [(x, y, radius), ...] 일괄 추가.
        """
        for obs in obstacles:
            if len(obs) == 3:
                self.set_obstacle(obs[0], obs[1], obs[2])
            else:
                self.set_obstacle(obs[0], obs[1])

    def add_inflation(self, robot_radius: float) -> None:
        """
        장애물 팽창 (로봇 반경만큼 여유 공간 확보).

        Args:
            robot_radius: 로봇 반경 (m)
        """
        from scipy.ndimage import binary_dilation
        r_cells = max(1, int(robot_radius / self.resolution))
        struct = np.zeros((2 * r_cells + 1, 2 * r_cells + 1))
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr ** 2 + dc ** 2 <= r_cells ** 2:
                    struct[dr + r_cells, dc + r_cells] = 1
        occupied = self.grid >= self.obstacle_threshold
        inflated = binary_dilation(occupied, structure=struct)
        self.grid[inflated] = 1.0

    def free_cells_count(self) -> int:
        return int(np.sum(self.grid < self.obstacle_threshold))


# ─────────────────────────────────────────────────────────────────────────────
# A* 경로계획기
# ─────────────────────────────────────────────────────────────────────────────

class AStarPlanner:
    """
    A* 경로계획기 (8-방향 이동).

    휴리스틱: Euclidean distance (admissible).
    대각선 이동 비용: √2 × resolution.

    Args:
        grid: OccupancyGrid 맵
    """

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
        # 8-방향 이동: (dcol, drow, cost)
        self._moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ]

    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        실세계 좌표에서 경로 탐색.

        Args:
            start_xy: (x, y) 시작점 (실세계, m)
            goal_xy: (x, y) 목표점 (실세계, m)

        Returns:
            path: [(x, y), ...] 실세계 경로점 리스트
                  None이면 경로 없음
        """
        start = self.grid.world_to_grid(*start_xy)
        goal = self.grid.world_to_grid(*goal_xy)
        return self._astar(start, goal)

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[float, float]]]:
        if self.grid.is_occupied(*start):
            # 시작점이 장애물이면 인근 자유 셀로 이동
            start = self._find_nearest_free(start)
            if start is None:
                return None
        if self.grid.is_occupied(*goal):
            goal = self._find_nearest_free(goal)
            if goal is None:
                return None

        # f = g + h
        open_heap = []  # (f, g, node)
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            f, g, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self._reconstruct(came_from, current)

            col, row = current
            for dc, dr, cost in self._moves:
                nc, nr = col + dc, row + dr
                neighbor = (nc, nr)
                if not self.grid.is_in_bounds(nc, nr):
                    continue
                if self.grid.is_occupied(nc, nr):
                    continue
                if neighbor in closed:
                    continue

                new_g = g + cost
                if new_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_g
                    came_from[neighbor] = current
                    f_new = new_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_new, new_g, neighbor))

        return None  # 경로 없음

    def _find_nearest_free(
        self, cell: Tuple[int, int], max_search: int = 20
    ) -> Optional[Tuple[int, int]]:
        """장애물에 막힌 셀 인근의 자유 셀 탐색 (BFS)."""
        q = [cell]
        visited = {cell}
        for _ in range(max_search ** 2):
            if not q:
                break
            c = q.pop(0)
            if not self.grid.is_occupied(*c):
                return c
            for dc, dr, _ in self._moves:
                nc, nr = c[0] + dc, c[1] + dr
                if (nc, nr) not in visited and self.grid.is_in_bounds(nc, nr):
                    visited.add((nc, nr))
                    q.append((nc, nr))
        return None

    def _reconstruct(
        self,
        came_from: dict,
        current: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        path_cells = [current]
        while current in came_from:
            current = came_from[current]
            path_cells.append(current)
        path_cells.reverse()
        return [self.grid.grid_to_world(*c) for c in path_cells]


# ─────────────────────────────────────────────────────────────────────────────
# Theta* 경로계획기
# ─────────────────────────────────────────────────────────────────────────────

class ThetaStarPlanner(AStarPlanner):
    """
    Theta* Any-Angle 경로계획기.

    A*의 확장: 직선 시야(Line-of-Sight) 확인으로
    격자 제약 없이 임의 각도 경로 생성.

    Nash et al. (2007) — 부드럽고 더 짧은 경로.
    """

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[float, float]]]:
        if self.grid.is_occupied(*start):
            start = self._find_nearest_free(start)
            if start is None:
                return None
        if self.grid.is_occupied(*goal):
            goal = self._find_nearest_free(goal)
            if goal is None:
                return None

        open_heap = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start))
        came_from = {start: start}  # Theta*: parent는 자기 자신으로 초기화
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            f, g, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self._reconstruct(came_from, current)

            parent = came_from.get(current, current)
            col, row = current
            for dc, dr, _ in self._moves:
                nc, nr = col + dc, row + dr
                neighbor = (nc, nr)
                if not self.grid.is_in_bounds(nc, nr):
                    continue
                if self.grid.is_occupied(nc, nr):
                    continue
                if neighbor in closed:
                    continue

                # Path 2: parent → neighbor (Line-of-Sight 확인)
                if self._line_of_sight(parent, neighbor):
                    pg = g_score[parent] + self._heuristic(parent, neighbor)
                    if pg < g_score.get(neighbor, float("inf")):
                        g_score[neighbor] = pg
                        came_from[neighbor] = parent
                        heapq.heappush(open_heap,
                                       (pg + self._heuristic(neighbor, goal), pg, neighbor))
                else:
                    # Path 1: current → neighbor
                    move_cost = self._heuristic(current, neighbor)
                    ng = g + move_cost
                    if ng < g_score.get(neighbor, float("inf")):
                        g_score[neighbor] = ng
                        came_from[neighbor] = current
                        heapq.heappush(open_heap,
                                       (ng + self._heuristic(neighbor, goal), ng, neighbor))

        return None

    def _line_of_sight(
        self, a: Tuple[int, int], b: Tuple[int, int]
    ) -> bool:
        """Bresenham 직선 위의 모든 셀이 자유인지 확인."""
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.grid.is_occupied(x, y):
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.grid.is_occupied(x, y):
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        return not self.grid.is_occupied(x1, y1)

    def _reconstruct(
        self,
        came_from: dict,
        current: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        path_cells = [current]
        while came_from.get(current) != current:
            current = came_from[current]
            path_cells.append(current)
        path_cells.reverse()
        return [self.grid.grid_to_world(*c) for c in path_cells]


# ─────────────────────────────────────────────────────────────────────────────
# 경로 후처리 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def smooth_path(
    path: List[Tuple[float, float]],
    weight_data: float = 0.5,
    weight_smooth: float = 0.1,
    iterations: int = 50,
) -> List[Tuple[float, float]]:
    """
    경로 평활화 (gradient descent).

    최소 3개 점 필요. 시작/끝 점은 고정.

    Args:
        path: [(x, y), ...] 원본 경로
        weight_data: 원본 데이터 유지 가중치
        weight_smooth: 평활도 가중치
        iterations: 최적화 반복 수

    Returns:
        smoothed: 평활화된 경로
    """
    if len(path) < 3:
        return path

    smoothed = [list(p) for p in path]
    n = len(smoothed)

    for _ in range(iterations):
        for i in range(1, n - 1):
            for j in range(2):
                smoothed[i][j] += weight_data * (path[i][j] - smoothed[i][j])
                smoothed[i][j] += weight_smooth * (
                    smoothed[i - 1][j] + smoothed[i + 1][j] - 2 * smoothed[i][j]
                )

    return [tuple(p) for p in smoothed]


def path_to_reference(
    path: List[Tuple[float, float]],
    N: int,
    dt: float,
    v_desired: float = 0.5,
    nx: int = 3,
) -> np.ndarray:
    """
    경로점 리스트 → MPPI 참조 궤적 (N+1, nx).

    Args:
        path: [(x, y), ...] 실세계 경로
        N: MPPI 호라이즌
        dt: 타임스텝 (s)
        v_desired: 목표 속도 (m/s)
        nx: 상태 차원 (3: [x, y, θ], 4: [x, y, θ, v] 등)

    Returns:
        ref: (N+1, nx) 참조 궤적
    """
    if len(path) == 0:
        return np.zeros((N + 1, nx))

    # 경로를 등간격 아크길이로 리샘플링
    path_arr = np.array(path)
    diffs = np.diff(path_arr, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    arc_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = arc_len[-1]

    # N+1 개 균등 샘플
    s_sample = np.linspace(0, total_len, N + 1)

    ref = np.zeros((N + 1, nx))
    for i, s in enumerate(s_sample):
        idx = np.searchsorted(arc_len, s, side="right") - 1
        idx = np.clip(idx, 0, len(path) - 2)
        t_frac = (s - arc_len[idx]) / max(seg_lens[idx], 1e-9) if idx < len(seg_lens) else 0.0
        t_frac = np.clip(t_frac, 0, 1)

        x = path_arr[idx, 0] + t_frac * (path_arr[idx + 1, 0] - path_arr[idx, 0])
        y = path_arr[idx, 1] + t_frac * (path_arr[idx + 1, 1] - path_arr[idx, 1])

        # 방향각 계산 (전방 차분)
        if idx < len(path) - 1:
            dx = path_arr[idx + 1, 0] - path_arr[idx, 0]
            dy = path_arr[idx + 1, 1] - path_arr[idx, 1]
            theta = np.arctan2(dy, dx)
        else:
            theta = ref[i - 1, 2] if i > 0 else 0.0

        ref[i, 0] = x
        ref[i, 1] = y
        if nx > 2:
            ref[i, 2] = theta

    return ref
