"""
로봇 body 패치 렌더링.

Circle (DiffDrive), Car (Ackermann), Rectangle (Swerve) 3가지 shape을
matplotlib patches로 렌더링하고, 상태 업데이트마다 위치/회전을 갱신한다.

Usage:
    renderer = RobotRenderer({"shape": "circle", "radius": 0.2, "color": "blue"})
    patches = renderer.render(ax, state)  # state = [x, y, theta, ...]
    renderer.update(state)                # 위치 갱신
    renderer.clear()                      # 패치 제거
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RenderConfig:
    """로봇 렌더링 설정."""
    shape: str = "circle"        # "circle", "car", "rectangle"
    radius: float = 0.15         # circle 반지름
    length: float = 0.5          # car/rectangle 길이
    width: float = 0.3           # car/rectangle 너비
    color: str = "blue"          # body 색상
    alpha: float = 0.4           # body 투명도
    heading_length: float = 0.0  # 방향선 길이 (0이면 자동)
    show_heading: bool = True    # 방향선 표시 여부


def make_render_config(d: Optional[Dict] = None) -> RenderConfig:
    """dict → RenderConfig 변환 (None이면 기본값)."""
    if d is None:
        return RenderConfig()
    if isinstance(d, RenderConfig):
        return d
    return RenderConfig(**{k: v for k, v in d.items() if k in RenderConfig.__dataclass_fields__})


class RobotRenderer:
    """
    로봇 body 패치 렌더링/업데이트/제거.

    Args:
        config: 렌더링 설정 (dict 또는 RenderConfig)
    """

    def __init__(self, config=None):
        self.config = make_render_config(config)
        self._patches: List[Any] = []
        self._ax = None

    def render(self, ax, state: np.ndarray) -> List:
        """
        최초 패치 생성 또는 기존 패치 위치 갱신.

        Args:
            ax: matplotlib Axes
            state: [x, y, theta, ...] 로봇 상태

        Returns:
            patches 리스트
        """
        self._ax = ax

        if self._patches:
            self.update(state)
            return self._patches

        x, y = state[0], state[1]
        theta = state[2] if len(state) > 2 else 0.0

        if self.config.shape == "circle":
            self._patches = self._create_circle(ax, x, y, theta)
        elif self.config.shape == "car":
            self._patches = self._create_car(ax, x, y, theta)
        elif self.config.shape == "rectangle":
            self._patches = self._create_rectangle(ax, x, y, theta)
        else:
            self._patches = self._create_circle(ax, x, y, theta)

        return self._patches

    def update(self, state: np.ndarray):
        """기존 패치의 위치/회전 갱신."""
        if not self._patches:
            return

        x, y = state[0], state[1]
        theta = state[2] if len(state) > 2 else 0.0

        if self.config.shape == "circle":
            self._update_circle(x, y, theta)
        elif self.config.shape == "car":
            self._update_car(x, y, theta)
        elif self.config.shape == "rectangle":
            self._update_rectangle(x, y, theta)

    def clear(self):
        """모든 패치를 Axes에서 제거."""
        for p in self._patches:
            try:
                p.remove()
            except (ValueError, AttributeError):
                pass
        self._patches.clear()

    # ── Circle ──────────────────────────────────────────────────────────

    def _create_circle(self, ax, x, y, theta) -> list:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        r = self.config.radius
        body = plt.Circle(
            (x, y), r,
            color=self.config.color, alpha=self.config.alpha,
            zorder=5,
        )
        ax.add_patch(body)

        patches = [body]

        if self.config.show_heading:
            hl = self.config.heading_length or r * 1.5
            arrow = mpatches.FancyArrow(
                x, y,
                hl * np.cos(theta), hl * np.sin(theta),
                width=r * 0.3,
                head_width=r * 0.6,
                head_length=r * 0.3,
                color=self.config.color, alpha=0.8,
                zorder=6,
            )
            ax.add_patch(arrow)
            patches.append(arrow)

        return patches

    def _update_circle(self, x, y, theta):
        import matplotlib.patches as mpatches

        body = self._patches[0]
        body.center = (x, y)

        if len(self._patches) > 1 and self.config.show_heading:
            old_arrow = self._patches[1]
            old_arrow.remove()

            r = self.config.radius
            hl = self.config.heading_length or r * 1.5
            arrow = mpatches.FancyArrow(
                x, y,
                hl * np.cos(theta), hl * np.sin(theta),
                width=r * 0.3,
                head_width=r * 0.6,
                head_length=r * 0.3,
                color=self.config.color, alpha=0.8,
                zorder=6,
            )
            self._ax.add_patch(arrow)
            self._patches[1] = arrow

    # ── Car (Ackermann) ─────────────────────────────────────────────────

    def _create_car(self, ax, x, y, theta) -> list:
        import matplotlib.patches as mpatches
        import matplotlib.transforms as mtransforms

        L = self.config.length
        W = self.config.width

        # body rectangle (centroid 기준)
        body = mpatches.FancyBboxPatch(
            (-L / 2, -W / 2), L, W,
            boxstyle="round,pad=0.02",
            facecolor=self.config.color, alpha=self.config.alpha,
            edgecolor=self.config.color, linewidth=1.5,
            zorder=5,
        )

        # Transform: rotate + translate
        t = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        body.set_transform(t)
        ax.add_patch(body)

        patches = [body]

        # 전면 표시 (삼각형)
        if self.config.show_heading:
            front_marker = mpatches.RegularPolygon(
                (L / 2 * 0.8, 0), numVertices=3,
                radius=W * 0.25, orientation=0,
                facecolor="white", alpha=0.8,
                zorder=6,
            )
            front_marker.set_transform(t)
            ax.add_patch(front_marker)
            patches.append(front_marker)

        return patches

    def _update_car(self, x, y, theta):
        import matplotlib.transforms as mtransforms

        t = mtransforms.Affine2D().rotate(theta).translate(x, y) + self._ax.transData
        for p in self._patches:
            p.set_transform(t)

    # ── Rectangle (Swerve) ──────────────────────────────────────────────

    def _create_rectangle(self, ax, x, y, theta) -> list:
        import matplotlib.patches as mpatches
        import matplotlib.transforms as mtransforms

        L = self.config.length
        W = self.config.width

        body = mpatches.Rectangle(
            (-L / 2, -W / 2), L, W,
            facecolor=self.config.color, alpha=self.config.alpha,
            edgecolor=self.config.color, linewidth=1.5,
            zorder=5,
        )

        t = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        body.set_transform(t)
        ax.add_patch(body)

        patches = [body]

        if self.config.show_heading:
            # 방향 화살표
            hl = self.config.heading_length or L * 0.6
            arrow = mpatches.FancyArrow(
                0, 0, hl, 0,
                width=W * 0.15,
                head_width=W * 0.3,
                head_length=L * 0.15,
                color="white", alpha=0.8,
                zorder=6,
            )
            arrow.set_transform(t)
            ax.add_patch(arrow)
            patches.append(arrow)

        return patches

    def _update_rectangle(self, x, y, theta):
        import matplotlib.transforms as mtransforms

        t = mtransforms.Affine2D().rotate(theta).translate(x, y) + self._ax.transData
        for p in self._patches:
            p.set_transform(t)
