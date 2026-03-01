"""
Multi-Robot CBF (#731)

N개 로봇이 각자 MPPI를 실행. 다른 로봇은 동적 장애물(5-tuple)로 취급.
쌍별(pairwise) CBF 제약으로 로봇 간 충돌 회피를 보장.

Layer A: MultiRobotCBFCost — MPPI cost에 장벽 패널티 추가
Layer B: MultiRobotCBFFilter — QP 사후 보정
Coordinator: 여러 에이전트를 순차적으로 계획하고 상태를 공유

수학:
  h_ij = ||p_i - p_j||² - (r_i + r_j + margin)²

  로봇 i의 QP 제약:
    Lg·h_ij·u_i + ḣ_from_j + α·h_ij ≥ 0

    Lg·h_ij = [2(xi-xj)cos(θi) + 2(yi-yj)sin(θi), 0]  (diff-drive)
    ḣ_from_j = 2(xi-xj)·vx_j + 2(yi-yj)·vy_j
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import minimize
from mppi_controller.controllers.mppi.cost_functions import CostFunction


@dataclass
class RobotAgent:
    """
    로봇 에이전트 정의.

    Args:
        id: 로봇 ID
        state: (nx,) 현재 상태
        radius: 로봇 반경 (m)
        model: RobotModel
        controller: MPPIController
        velocity: (2,) [vx, vy] 현재 속도
    """
    id: int
    state: np.ndarray
    radius: float
    model: object
    controller: object
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


class MultiRobotCBFCost(CostFunction):
    """
    Multi-Robot CBF Cost (Layer A)

    다른 로봇과의 쌍별 CBF 패널티를 MPPI cost에 추가.

    Args:
        other_robots: List of (x, y, radius, vx, vy) 다른 로봇 정보
        cbf_alpha: class-K 파라미터
        cbf_weight: 패널티 가중치
        safety_margin: 추가 마진 (m)
    """

    def __init__(
        self,
        other_robots: Optional[List[tuple]] = None,
        cbf_alpha: float = 0.3,
        cbf_weight: float = 2000.0,
        safety_margin: float = 0.2,
    ):
        self.other_robots = other_robots or []
        self.cbf_alpha = cbf_alpha
        self.cbf_weight = cbf_weight
        self.safety_margin = safety_margin

    def compute_cost(
        self,
        trajectories: np.ndarray,
        controls: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Multi-robot CBF cost.

        Args:
            trajectories: (K, N+1, nx)
            controls: (K, N, nu)
            reference_trajectory: (N+1, nx)

        Returns:
            costs: (K,)
        """
        K = trajectories.shape[0]
        costs = np.zeros(K)

        if not self.other_robots:
            return costs

        for robot_info in self.other_robots:
            rx, ry, r_other = robot_info[0], robot_info[1], robot_info[2]
            vx_other = robot_info[3] if len(robot_info) > 3 else 0.0
            vy_other = robot_info[4] if len(robot_info) > 4 else 0.0

            effective_r = r_other + self.safety_margin

            # 궤적 위치 (K, N+1)
            px = trajectories[:, :, 0]
            py = trajectories[:, :, 1]

            # 거리 제곱
            dist_sq = (px - rx) ** 2 + (py - ry) ** 2  # (K, N+1)

            # barrier: h = dist² - r_eff²
            h = dist_sq - effective_r ** 2  # (K, N+1)

            # barrier 위반 시 패널티 (exponential, 오버플로 방지)
            violation = np.where(
                h < 0, np.exp(np.clip(-h * 3.0, 0, 10)), 0.0
            )

            # 접근 속도 고려: 다른 로봇 방향으로 접근하면 추가 패널티
            if abs(vx_other) > 1e-6 or abs(vy_other) > 1e-6:
                # 상대 속도에 의한 위험도 증가
                dp_x = px - rx
                dp_y = py - ry
                # 다른 로봇이 접근하면 h의 시간 도함수가 음
                h_dot_from_j = 2.0 * dp_x * vx_other + 2.0 * dp_y * vy_other
                approach_penalty = np.where(
                    h_dot_from_j < 0,
                    0.1 * np.abs(h_dot_from_j) / (np.sqrt(dist_sq) + 1e-6),
                    0.0,
                )
                violation += approach_penalty

            costs += self.cbf_weight * np.sum(violation, axis=1)

        return costs

    def update_other_robots(self, other_robots: List[tuple]):
        """다른 로봇 상태 업데이트"""
        self.other_robots = other_robots

    def get_barrier_info(self, trajectory: np.ndarray) -> Dict:
        """
        단일 궤적에 대한 barrier 정보.

        Args:
            trajectory: (N+1, nx)

        Returns:
            dict with barrier_values, min_barrier, is_safe
        """
        if not self.other_robots:
            return {
                "barrier_values": [],
                "min_barrier": float("inf"),
                "is_safe": True,
            }

        all_barriers = []
        for robot_info in self.other_robots:
            rx, ry, r_other = robot_info[0], robot_info[1], robot_info[2]
            effective_r = r_other + self.safety_margin

            dist_sq = (
                (trajectory[:, 0] - rx) ** 2 + (trajectory[:, 1] - ry) ** 2
            )
            h = dist_sq - effective_r ** 2
            all_barriers.append(float(np.min(h)))

        min_barrier = min(all_barriers)
        return {
            "barrier_values": all_barriers,
            "min_barrier": min_barrier,
            "is_safe": min_barrier > 0,
        }

    def __repr__(self) -> str:
        return (
            f"MultiRobotCBFCost("
            f"num_robots={len(self.other_robots)}, "
            f"α={self.cbf_alpha}, w={self.cbf_weight})"
        )


class MultiRobotCBFFilter:
    """
    Multi-Robot CBF Safety Filter (Layer B)

    QP 기반 사후 보정: 다른 로봇과의 충돌을 방지.

    Args:
        other_robots: List of (x, y, radius, vx, vy)
        cbf_alpha: class-K 파라미터
        safety_margin: 추가 마진 (m)
    """

    def __init__(
        self,
        other_robots: Optional[List[tuple]] = None,
        cbf_alpha: float = 0.3,
        safety_margin: float = 0.2,
    ):
        self.other_robots = other_robots or []
        self.cbf_alpha = cbf_alpha
        self.safety_margin = safety_margin
        self.filter_stats = []

    def filter_control(
        self,
        state: np.ndarray,
        u_mppi: np.ndarray,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Multi-robot CBF 안전 필터.

        Args:
            state: (nx,) [x, y, θ, ...]
            u_mppi: (nu,) MPPI 제어
            u_min: (nu,) 하한
            u_max: (nu,) 상한

        Returns:
            u_safe: (nu,)
            info: dict
        """
        if not self.other_robots:
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "barrier_values": [],
                "min_barrier": float("inf"),
            }
            self.filter_stats.append(info)
            return u_mppi.copy(), info

        x, y = state[0], state[1]
        theta = state[2] if len(state) >= 3 else 0.0

        constraints = []
        barrier_values = []

        for robot_info in self.other_robots:
            rx, ry, r_other = robot_info[0], robot_info[1], robot_info[2]
            vx_other = robot_info[3] if len(robot_info) > 3 else 0.0
            vy_other = robot_info[4] if len(robot_info) > 4 else 0.0

            effective_r = r_other + self.safety_margin

            # barrier
            h = (x - rx) ** 2 + (y - ry) ** 2 - effective_r ** 2
            barrier_values.append(h)

            # Lie derivatives (diff-drive kinematic)
            # Lg_h = [2(x-rx)cos(θ) + 2(y-ry)sin(θ), 0]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            Lg_h = np.array([
                2.0 * (x - rx) * cos_theta + 2.0 * (y - ry) * sin_theta,
                0.0,
            ])

            # 다른 로봇 속도에 의한 ḣ 기여
            h_dot_from_j = 2.0 * (x - rx) * vx_other + 2.0 * (y - ry) * vy_other

            def cbf_con(u, Lg=Lg_h, hdj=h_dot_from_j, alpha=self.cbf_alpha, hv=h):
                return Lg @ u + hdj + alpha * hv

            constraints.append({"type": "ineq", "fun": cbf_con})

        # 빠른 경로: 모든 제약 만족
        all_safe = all(c["fun"](u_mppi) >= 0 for c in constraints)
        if all_safe:
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "barrier_values": barrier_values,
                "min_barrier": min(barrier_values) if barrier_values else float("inf"),
            }
            self.filter_stats.append(info)
            return u_mppi.copy(), info

        # QP
        def objective(u):
            d = u - u_mppi
            return 0.5 * np.dot(d, d)

        def objective_jac(u):
            return u - u_mppi

        bounds = None
        if u_min is not None and u_max is not None:
            bounds = list(zip(u_min, u_max))

        result = minimize(
            objective,
            x0=u_mppi.copy(),
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-8},
        )

        u_safe = result.x
        correction_norm = float(np.linalg.norm(u_safe - u_mppi))

        info = {
            "filtered": True,
            "correction_norm": correction_norm,
            "barrier_values": barrier_values,
            "min_barrier": min(barrier_values) if barrier_values else float("inf"),
            "optimization_success": result.success,
        }
        self.filter_stats.append(info)
        return u_safe, info

    def update_other_robots(self, other_robots: List[tuple]):
        """다른 로봇 상태 업데이트"""
        self.other_robots = other_robots

    def reset(self):
        self.filter_stats = []

    def __repr__(self) -> str:
        return (
            f"MultiRobotCBFFilter("
            f"num_robots={len(self.other_robots)}, "
            f"α={self.cbf_alpha})"
        )


class MultiRobotCoordinator:
    """
    Multi-Robot Coordinator

    여러 로봇 에이전트를 순차적으로 계획/조율.
    각 에이전트의 MPPI cost에 다른 로봇 CBF cost를 주입하고,
    옵션으로 사후 CBF 필터도 적용.

    Args:
        agents: List[RobotAgent]
        dt: 시간 간격
        cbf_alpha: CBF 파라미터
        safety_margin: 안전 마진 (m)
        use_cost: Layer A (cost) 사용 여부
        use_filter: Layer B (filter) 사용 여부
    """

    def __init__(
        self,
        agents: List[RobotAgent],
        dt: float = 0.05,
        cbf_alpha: float = 0.3,
        safety_margin: float = 0.2,
        use_cost: bool = True,
        use_filter: bool = True,
    ):
        self.agents = {a.id: a for a in agents}
        self.dt = dt
        self.cbf_alpha = cbf_alpha
        self.safety_margin = safety_margin
        self.use_cost = use_cost
        self.use_filter = use_filter

        # 각 에이전트의 CBF cost와 filter
        self._costs: Dict[int, MultiRobotCBFCost] = {}
        self._filters: Dict[int, MultiRobotCBFFilter] = {}

        for agent in agents:
            if use_cost:
                self._costs[agent.id] = MultiRobotCBFCost(
                    cbf_alpha=cbf_alpha, safety_margin=safety_margin,
                )
            if use_filter:
                self._filters[agent.id] = MultiRobotCBFFilter(
                    cbf_alpha=cbf_alpha, safety_margin=safety_margin,
                )

    def _get_other_robots(self, agent_id: int) -> List[tuple]:
        """특정 에이전트를 제외한 다른 로봇 정보"""
        others = []
        for aid, agent in self.agents.items():
            if aid == agent_id:
                continue
            others.append((
                agent.state[0], agent.state[1], agent.radius,
                agent.velocity[0], agent.velocity[1],
            ))
        return others

    def step(
        self, reference_trajectories: Dict[int, np.ndarray]
    ) -> Dict[int, Tuple[np.ndarray, Dict]]:
        """
        모든 에이전트 1-step 조율.

        Args:
            reference_trajectories: {agent_id: (N+1, nx)} 레퍼런스

        Returns:
            results: {agent_id: (control, info)}
        """
        results = {}

        for agent_id, agent in self.agents.items():
            ref = reference_trajectories.get(agent_id)
            if ref is None:
                continue

            # 다른 로봇 정보 업데이트
            others = self._get_other_robots(agent_id)

            if self.use_cost and agent_id in self._costs:
                self._costs[agent_id].update_other_robots(others)

            if self.use_filter and agent_id in self._filters:
                self._filters[agent_id].update_other_robots(others)

            # MPPI 계산
            control, mppi_info = agent.controller.compute_control(agent.state, ref)

            # CBF filter 적용
            info = {"mppi_info": mppi_info, "cbf_filtered": False}
            if self.use_filter and agent_id in self._filters:
                bounds = agent.model.get_control_bounds()
                u_min = bounds[0] if bounds else None
                u_max = bounds[1] if bounds else None
                control, filter_info = self._filters[agent_id].filter_control(
                    agent.state, control, u_min, u_max,
                )
                info["filter_info"] = filter_info
                info["cbf_filtered"] = filter_info.get("filtered", False)

            # 상태 전파
            next_state = agent.model.step(agent.state, control, self.dt)

            # 속도 업데이트 (간단 근사)
            if len(agent.state) >= 3:
                v = control[0] if len(control) >= 1 else 0.0
                theta = agent.state[2]
                agent.velocity = np.array([v * np.cos(theta), v * np.sin(theta)])

            agent.state = next_state
            results[agent_id] = (control, info)

        return results

    def get_states(self) -> Dict[int, np.ndarray]:
        """모든 에이전트 상태 반환"""
        return {aid: a.state.copy() for aid, a in self.agents.items()}

    def __repr__(self) -> str:
        return (
            f"MultiRobotCoordinator("
            f"agents={len(self.agents)}, "
            f"cost={self.use_cost}, filter={self.use_filter})"
        )
