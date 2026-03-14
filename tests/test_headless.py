"""
NullAxes / NullFigure / create_figure 테스트
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.simulation.rendering.headless import (
    NullAxes,
    NullFigure,
    create_figure,
)


# ── NullAxes ──────────────────────────────────────────────────

def test_null_axes_plot():
    """NullAxes.plot()는 아무것도 하지 않고 반환"""
    ax = NullAxes()
    result = ax.plot([1, 2, 3], [4, 5, 6])
    assert isinstance(result, NullAxes)


def test_null_axes_scatter():
    ax = NullAxes()
    ax.scatter([1], [2], c="r")


def test_null_axes_set_methods():
    ax = NullAxes()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Title")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis("equal")


def test_null_axes_add_patch():
    ax = NullAxes()
    ax.add_patch("dummy_patch")


def test_null_axes_bool():
    ax = NullAxes()
    assert not ax


def test_null_axes_iter():
    ax = NullAxes()
    assert list(ax) == []


def test_null_axes_repr():
    ax = NullAxes()
    assert "NullAxes" in repr(ax)


# ── NullFigure ────────────────────────────────────────────────

def test_null_figure_savefig():
    fig = NullFigure()
    fig.savefig("dummy.png")


def test_null_figure_tight_layout():
    fig = NullFigure()
    fig.tight_layout()


def test_null_figure_suptitle():
    fig = NullFigure()
    fig.suptitle("Title")


def test_null_figure_axes():
    fig = NullFigure()
    axes = fig.axes
    assert len(axes) == 1
    assert isinstance(axes[0], NullAxes)


def test_null_figure_bool():
    fig = NullFigure()
    assert not fig


def test_null_figure_text():
    fig = NullFigure()
    fig.text(0.5, 0.5, "hello")


# ── create_figure ─────────────────────────────────────────────

def test_create_figure_headless_single():
    fig, ax = create_figure(headless=True)
    assert isinstance(fig, NullFigure)
    assert isinstance(ax, NullAxes)


def test_create_figure_headless_1d():
    fig, axes = create_figure(headless=True, nrows=1, ncols=3)
    assert isinstance(fig, NullFigure)
    assert len(axes) == 3
    assert all(isinstance(a, NullAxes) for a in axes)


def test_create_figure_headless_2d():
    fig, axes = create_figure(headless=True, nrows=2, ncols=3)
    assert isinstance(fig, NullFigure)
    assert len(axes) == 2
    assert len(axes[0]) == 3
    assert all(isinstance(a, NullAxes) for a in axes[0])


def test_create_figure_real():
    """실제 matplotlib Figure 생성 (Agg 백엔드)"""
    fig, ax = create_figure(headless=False, nrows=1, ncols=1)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
