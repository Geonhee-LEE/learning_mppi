"""
SimulationEnvironment ABC + obstacle types + EnvironmentConfig

시뮬레이션 시나리오의 공통 베이스 클래스와 장애물 정의.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple


# ── Obstacle Types ──────────────────────────────────────────────────────────

@dataclass
class CircleObstacle:
    """원형 장애물"""
    x: float
    y: float
    radius: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.radius)

    def contains(self, px: float, py: float) -> bool:
        return (px - self.x) ** 2 + (py - self.y) ** 2 < self.radius ** 2


@dataclass
class WallObstacle:
    """벽 장애물 — 원형 장애물 체인으로 근사"""
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 0.15

    def to_circles(self, spacing: float = 0.15) -> List[CircleObstacle]:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length < 1e-6:
            return [CircleObstacle(self.x1, self.y1, self.thickness)]
        n = max(2, int(length / spacing) + 1)
        circles = []
        for i in range(n):
            t = i / (n - 1)
            cx = self.x1 + t * dx
            cy = self.y1 + t * dy
            circles.append(CircleObstacle(cx, cy, self.thickness))
        return circles

    def as_tuples(self, spacing: float = 0.15) -> List[Tuple[float, float, float]]:
        return [c.as_tuple() for c in self.to_circles(spacing)]


# ── Environment Config ──────────────────────────────────────────────────────

@dataclass
class EnvironmentConfig:
    """시나리오 공통 설정"""
    name: str = "default"
    duration: float = 15.0
    dt: float = 0.05
    N: int = 30
    K: int = 1024
    lambda_: float = 1.0
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    Q: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1]))
    Qf: Optional[np.ndarray] = None
    robot_model_type: str = "differential_drive"
    v_max: float = 1.0
    omega_max: float = 1.0
    process_noise_std: Optional[np.ndarray] = None
    seed: int = 42


# ── Controller Config ───────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    """시뮬레이션에서 비교할 컨트롤러 설정"""
    name: str
    controller: object
    model: object
    color: str
    linestyle: str = "-"


# ── Simulation Environment ABC ──────────────────────────────────────────────

class SimulationEnvironment(ABC):
    """
    시뮬레이션 시나리오 추상 베이스 클래스.

    각 시나리오는 이 클래스를 상속하여:
    - 초기 상태, 장애물, 레퍼런스 궤적, 컨트롤러 목록을 정의
    - draw_environment()로 시나리오별 시각화 데코레이션 추가
    - on_step()으로 동적 업데이트 수행
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self._static_obstacles: List[CircleObstacle] = []
        self._walls: List[WallObstacle] = []

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """초기 상태 반환 (nx,)"""
        ...

    @abstractmethod
    def get_obstacles(self, t: float = 0.0) -> List[Tuple[float, float, float]]:
        """시간 t에서의 장애물 목록 반환 [(x, y, radius), ...]"""
        ...

    @abstractmethod
    def get_reference_fn(self) -> Callable[[float], np.ndarray]:
        """레퍼런스 궤적 함수 반환: t -> (N+1, nx)"""
        ...

    @abstractmethod
    def get_controller_configs(self) -> List[ControllerConfig]:
        """비교할 컨트롤러 목록 반환"""
        ...

    def get_obstacles_with_velocity(self, t: float = 0.0) -> List[Tuple[float, float, float, float, float]]:
        """
        속도 포함 장애물 반환 [(x, y, r, vx, vy), ...].

        기본 구현은 정적 장애물 (vx=vy=0). 동적 장애물이 있는
        서브클래스에서 오버라이드.
        """
        return [(x, y, r, 0.0, 0.0) for x, y, r in self.get_obstacles(t)]

    def draw_environment(self, ax, t: float = 0.0):
        """시나리오별 시각화 데코레이션 (matplotlib Axes에 그리기)"""
        import matplotlib.pyplot as plt

        # 기본: 원형 장애물 그리기
        obstacles = self.get_obstacles(t)
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color="red", alpha=0.25)
            ax.add_patch(circle)
            margin = plt.Circle((ox, oy), r + 0.05, color="red",
                                alpha=0.08, linestyle="--", fill=False)
            ax.add_patch(margin)

    def on_step(self, t: float, states: Dict[str, np.ndarray],
                controls: Dict[str, np.ndarray], infos: Dict[str, dict]):
        """매 스텝마다 호출되는 콜백 (동적 장애물 업데이트 등)"""
        pass

    def get_extra_metrics(self, histories: Dict[str, dict]) -> Dict[str, dict]:
        """시나리오별 추가 메트릭 반환"""
        return {}

    # ── Utility ──

    def add_static_obstacle(self, x: float, y: float, r: float):
        self._static_obstacles.append(CircleObstacle(x, y, r))

    def add_wall(self, x1: float, y1: float, x2: float, y2: float,
                 thickness: float = 0.15):
        self._walls.append(WallObstacle(x1, y1, x2, y2, thickness))

    def get_wall_obstacles(self) -> List[Tuple[float, float, float]]:
        """벽을 원형 장애물로 변환"""
        result = []
        for wall in self._walls:
            result.extend(wall.as_tuples())
        return result

    def get_static_obstacles(self) -> List[Tuple[float, float, float]]:
        return [c.as_tuple() for c in self._static_obstacles]

    def get_all_static_obstacles(self) -> List[Tuple[float, float, float]]:
        return self.get_static_obstacles() + self.get_wall_obstacles()

    def compute_min_distance(self, state: np.ndarray,
                             obstacles: List[Tuple[float, float, float]]) -> float:
        """로봇과 장애물 간 최소 거리"""
        if not obstacles:
            return float("inf")
        min_d = float("inf")
        px, py = state[0], state[1]
        for ox, oy, r in obstacles:
            d = np.sqrt((px - ox) ** 2 + (py - oy) ** 2) - r
            min_d = min(min_d, d)
        return min_d

    def check_collision(self, state: np.ndarray,
                        obstacles: List[Tuple[float, float, float]],
                        robot_radius: float = 0.15) -> bool:
        """충돌 여부 확인"""
        px, py = state[0], state[1]
        for ox, oy, r in obstacles:
            if (px - ox) ** 2 + (py - oy) ** 2 < (r + robot_radius) ** 2:
                return True
        return False
