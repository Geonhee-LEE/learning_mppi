"""
안전 시각화 오버레이 — CBF contour, Collision cone, DPCBF parabola 등.

SafetyOverlay 클래스가 다양한 안전 기법의 시각적 표현을 제공한다.

Usage:
    overlay = SafetyOverlay()
    overlay.draw_cbf_contour(ax, obstacles, state)
    overlay.draw_collision_cone(ax, state, obstacle_5tuple)
    overlay.draw_effective_radius(ax, obstacles, r_eff_list)
"""

import numpy as np
from typing import List, Tuple, Optional, Any


class SafetyOverlay:
    """
    안전 관련 시각화 요소를 matplotlib Axes에 그린다.

    각 메서드는 독립적으로 호출 가능하며,
    이전 프레임의 아티팩트를 제거하고 새로 그린다.
    """

    def __init__(self):
        self._artifacts: list = []

    def clear(self):
        """이전 프레임의 모든 아티팩트 제거."""
        for art in self._artifacts:
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self._artifacts.clear()

    # ── CBF Contour (h(x) = 0 등고선) ──────────────────────────────────

    def draw_cbf_contour(
        self,
        ax,
        obstacles: List[Tuple[float, float, float]],
        state: np.ndarray,
        robot_radius: float = 0.15,
        grid_resolution: int = 50,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        """
        CBF h(x) = ||p - p_obs||^2 - (r + r_robot)^2 등고선.

        h > 0: 안전, h < 0: 위험. h=0 경계를 빨간 실선으로 표시.

        Args:
            ax: matplotlib Axes
            obstacles: [(x, y, r), ...]
            state: 로봇 상태 [x, y, ...]
            robot_radius: 로봇 반지름
            grid_resolution: 그리드 해상도
            xlim, ylim: 그리드 범위 (None이면 자동)
        """
        if not obstacles:
            return

        px, py = state[0], state[1]

        # 그리드 범위 설정
        if xlim is None:
            all_x = [o[0] for o in obstacles] + [px]
            xlim = (min(all_x) - 3, max(all_x) + 3)
        if ylim is None:
            all_y = [o[1] for o in obstacles] + [py]
            ylim = (min(all_y) - 3, max(all_y) + 3)

        xs = np.linspace(xlim[0], xlim[1], grid_resolution)
        ys = np.linspace(ylim[0], ylim[1], grid_resolution)
        X, Y = np.meshgrid(xs, ys)

        # 모든 장애물에 대한 CBF 최소값
        H = np.full_like(X, np.inf)
        for ox, oy, r in obstacles:
            h = (X - ox) ** 2 + (Y - oy) ** 2 - (r + robot_radius) ** 2
            H = np.minimum(H, h)

        # 등고선 (h=0 경계)
        contour = ax.contour(
            X, Y, H, levels=[0],
            colors="red", linewidths=2, linestyles="solid",
        )
        # matplotlib >= 3.8 uses contour set directly
        if hasattr(contour, 'collections'):
            self._artifacts.extend(contour.collections)
        else:
            self._artifacts.append(contour)

        # 안전/위험 영역 색상
        fill = ax.contourf(
            X, Y, H, levels=[-np.inf, 0, np.inf],
            colors=["#ff000015", "#00ff0008"],
        )
        if hasattr(fill, 'collections'):
            self._artifacts.extend(fill.collections)
        else:
            self._artifacts.append(fill)

    # ── Collision Cone (C3BF) ──────────────────────────────────────────

    def draw_collision_cone(
        self,
        ax,
        state: np.ndarray,
        obstacle: Tuple[float, float, float, float, float],
        cone_length: float = 3.0,
    ):
        """
        C3BF 충돌 콘 삼각형 시각화.

        5-tuple 장애물 (x, y, vx, vy, r) 기준으로 상대 속도 벡터가
        콘 내부에 있으면 충돌 위험을 표시.

        Args:
            ax: matplotlib Axes
            state: [x, y, theta, ...]
            obstacle: (ox, oy, vx_obs, vy_obs, r)
            cone_length: 콘 가시화 길이
        """
        import matplotlib.patches as mpatches

        px, py = state[0], state[1]
        ox, oy, vx_obs, vy_obs, r = obstacle

        # 상대 위치
        dx, dy = ox - px, oy - py
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist < r:
            return  # 이미 충돌 상태

        # 콘 반각
        half_angle = np.arcsin(min(r / dist, 1.0))
        center_angle = np.arctan2(dy, dx)

        # 콘 삼각형 정점
        angle1 = center_angle - half_angle
        angle2 = center_angle + half_angle

        cone_pts = np.array([
            [px, py],
            [px + cone_length * np.cos(angle1), py + cone_length * np.sin(angle1)],
            [px + cone_length * np.cos(angle2), py + cone_length * np.sin(angle2)],
        ])

        triangle = mpatches.Polygon(
            cone_pts,
            closed=True,
            facecolor="orange",
            alpha=0.15,
            edgecolor="orange",
            linewidth=1.5,
            linestyle="--",
            zorder=3,
        )
        ax.add_patch(triangle)
        self._artifacts.append(triangle)

    # ── DPCBF Parabola (LoS 경계) ─────────────────────────────────────

    def draw_dpcbf_parabola(
        self,
        ax,
        state: np.ndarray,
        obstacle: Tuple[float, float, float, float, float],
        n_points: int = 50,
        t_range: float = 3.0,
    ):
        """
        DPCBF (Discrete-Predictive CBF) Line-of-Sight 포물선 경계.

        Args:
            ax: matplotlib Axes
            state: [x, y, theta, ...]
            obstacle: (ox, oy, vx, vy, r)
            n_points: 포물선 점 수
            t_range: 시간 범위
        """
        px, py = state[0], state[1]
        ox, oy, vx, vy, r = obstacle

        ts = np.linspace(0, t_range, n_points)
        # 장애물 예측 궤적
        ox_t = ox + vx * ts
        oy_t = oy + vy * ts

        # 안전 반경 경계 (상단/하단)
        dx = ox_t - px
        dy = oy_t - py
        dist = np.sqrt(dx ** 2 + dy ** 2)
        dist = np.maximum(dist, 1e-6)

        # 수직 방향 오프셋
        nx_hat = -dy / dist
        ny_hat = dx / dist

        upper_x = ox_t + r * nx_hat
        upper_y = oy_t + r * ny_hat
        lower_x = ox_t - r * nx_hat
        lower_y = oy_t - r * ny_hat

        line_upper, = ax.plot(
            upper_x, upper_y, "m--", linewidth=1.5, alpha=0.6, label="DPCBF boundary"
        )
        line_lower, = ax.plot(
            lower_x, lower_y, "m--", linewidth=1.5, alpha=0.6,
        )
        self._artifacts.extend([line_upper, line_lower])

    # ── Effective Radius (C2U-MPPI) ────────────────────────────────────

    def draw_effective_radius(
        self,
        ax,
        obstacles: List[Tuple[float, float, float]],
        r_eff: Optional[List[float]] = None,
    ):
        """
        C2U-MPPI 확장 반경 시각화.

        원래 장애물 반경 위에 r_eff = r + κ_α√Σ 확장 경계를 점선으로 표시.

        Args:
            ax: matplotlib Axes
            obstacles: [(x, y, r), ...]
            r_eff: 각 장애물의 확장 반경 리스트 (None이면 생략)
        """
        import matplotlib.pyplot as plt

        if r_eff is None:
            return

        for i, (ox, oy, r) in enumerate(obstacles):
            if i < len(r_eff) and r_eff[i] > r:
                circle = plt.Circle(
                    (ox, oy), r_eff[i],
                    color="purple", alpha=0.1,
                    linestyle="--", linewidth=1.5,
                    fill=True,
                    zorder=2,
                )
                ax.add_patch(circle)
                self._artifacts.append(circle)

    # ── Neural CBF Heatmap ─────────────────────────────────────────────

    def draw_neural_cbf_heatmap(
        self,
        ax,
        h_fn,
        xlim: Tuple[float, float] = (-5, 5),
        ylim: Tuple[float, float] = (-5, 5),
        theta: float = 0.0,
        grid_resolution: int = 40,
    ):
        """
        Neural CBF h(x) 히트맵 시각화.

        Args:
            ax: matplotlib Axes
            h_fn: state (N, nx) -> h values (N,) 함수
            xlim, ylim: 그리드 범위
            theta: 고정 heading
            grid_resolution: 해상도
        """
        xs = np.linspace(xlim[0], xlim[1], grid_resolution)
        ys = np.linspace(ylim[0], ylim[1], grid_resolution)
        X, Y = np.meshgrid(xs, ys)

        # 그리드 상태 벡터 생성
        states = np.column_stack([
            X.ravel(), Y.ravel(),
            np.full(X.size, theta),
        ])

        H = h_fn(states).reshape(X.shape)

        # 히트맵
        pcm = ax.pcolormesh(
            X, Y, H, cmap="RdYlGn", shading="auto",
            alpha=0.3, zorder=1,
        )
        self._artifacts.append(pcm)

        # h=0 등고선
        contour = ax.contour(
            X, Y, H, levels=[0],
            colors="black", linewidths=2,
        )
        if hasattr(contour, 'collections'):
            self._artifacts.extend(contour.collections)
        else:
            self._artifacts.append(contour)

    # ── Infeasibility Marker ───────────────────────────────────────────

    def draw_infeasibility_marker(
        self,
        ax,
        state: np.ndarray,
        info: dict,
    ):
        """
        QP 비실행 가능 표시 (빨간 X).

        info dict에 'qp_infeasible': True가 있으면 로봇 위치에 X 표시.

        Args:
            ax: matplotlib Axes
            state: [x, y, ...]
            info: 컨트롤러 info dict
        """
        if not info.get("qp_infeasible", False):
            return

        marker = ax.plot(
            state[0], state[1], "rx",
            markersize=15, markeredgewidth=3,
            zorder=10,
        )
        self._artifacts.extend(marker)

    # ── Shield Intervention Marker ─────────────────────────────────────

    def draw_shield_intervention(
        self,
        ax,
        state: np.ndarray,
        info: dict,
    ):
        """
        Shield 개입 표시 (주황 사각형 마커).

        info dict에 'shield_active': True가 있으면 표시.

        Args:
            ax: matplotlib Axes
            state: [x, y, ...]
            info: 컨트롤러 info dict
        """
        if not info.get("shield_active", False):
            return

        marker = ax.plot(
            state[0], state[1], "s",
            color="orange", markersize=10,
            markeredgecolor="red", markeredgewidth=2,
            alpha=0.7, zorder=9,
        )
        self._artifacts.extend(marker)

    def draw_all(
        self,
        ax,
        state: np.ndarray,
        obstacles: List[Tuple],
        info: Optional[dict] = None,
        robot_radius: float = 0.15,
    ):
        """
        활성화된 모든 안전 시각화를 한번에 그린다.

        Args:
            ax: matplotlib Axes
            state: 로봇 상태
            obstacles: 장애물 리스트
            info: 컨트롤러 info dict
            robot_radius: 로봇 반지름
        """
        self.clear()

        info = info or {}

        # CBF contour
        if obstacles:
            self.draw_cbf_contour(ax, obstacles, state, robot_radius)

        # 확장 반경 (C2U-MPPI)
        r_eff = info.get("effective_radii")
        if r_eff is not None:
            # 3-tuple 장애물만 대상
            obs_3 = [(o[0], o[1], o[2]) for o in obstacles if len(o) >= 3]
            self.draw_effective_radius(ax, obs_3, r_eff)

        # 충돌 콘 (5-tuple 동적 장애물)
        for obs in obstacles:
            if len(obs) >= 5:
                self.draw_collision_cone(ax, state, obs)

        # QP 비실행 가능 / Shield 개입
        self.draw_infeasibility_marker(ax, state, info)
        self.draw_shield_intervention(ax, state, info)
