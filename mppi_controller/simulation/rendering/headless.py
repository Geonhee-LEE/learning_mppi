"""
Headless 렌더링 지원 — NullAxes / NullFigure.

GUI 없는 환경 (CI, SSH, headless 서버)에서 matplotlib 호출을
조용히 흡수하여 코드 분기 없이 동일한 시각화 코드를 실행.

Usage:
    fig, ax = create_figure(headless=True)   # NullFigure, NullAxes 반환
    ax.plot([1, 2], [3, 4])                  # 아무 동작 없음
    fig.savefig("out.png")                   # 아무 동작 없음
"""

from typing import Any, Tuple


class NullAxes:
    """
    matplotlib.axes.Axes 의 모든 메서드를 흡수하는 Null Object.

    plot, scatter, add_patch, set_xlabel, set_title, legend 등
    모든 호출을 조용히 무시한다.
    """

    def __getattr__(self, name: str) -> Any:
        """알려지지 않은 속성/메서드 접근을 자기 자신을 반환하는 callable로 처리."""
        return self._noop

    def _noop(self, *args: Any, **kwargs: Any) -> "NullAxes":
        """모든 호출을 흡수하고 체이닝 지원을 위해 self 반환."""
        return self

    def __iter__(self):
        """빈 iterable (unpack 방지)."""
        return iter([])

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "NullAxes()"


class NullFigure:
    """
    matplotlib.figure.Figure 의 Null Object.

    savefig, tight_layout, suptitle, text, set_size_inches 등
    모든 호출을 조용히 무시한다.
    """

    def __init__(self):
        self._axes = NullAxes()

    @property
    def axes(self):
        return [self._axes]

    def __getattr__(self, name: str) -> Any:
        return self._noop

    def _noop(self, *args: Any, **kwargs: Any) -> "NullFigure":
        return self

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "NullFigure()"


def create_figure(
    headless: bool = False,
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[float, float] = (12, 8),
    **kwargs,
):
    """
    Figure + Axes 생성 (headless 투명 분기).

    Args:
        headless: True면 NullFigure/NullAxes 반환
        nrows: 서브플롯 행 수
        ncols: 서브플롯 열 수
        figsize: Figure 크기
        **kwargs: plt.subplots 추가 인자

    Returns:
        (fig, axes) — headless이면 (NullFigure, NullAxes 또는 2D array)
    """
    if headless:
        fig = NullFigure()
        if nrows == 1 and ncols == 1:
            return fig, NullAxes()
        elif nrows == 1 or ncols == 1:
            return fig, [NullAxes() for _ in range(max(nrows, ncols))]
        else:
            return fig, [[NullAxes() for _ in range(ncols)] for _ in range(nrows)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes
