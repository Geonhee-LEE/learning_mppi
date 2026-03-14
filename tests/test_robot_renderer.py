"""
RobotRenderer 테스트 — Circle / Car / Rectangle 렌더링
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mppi_controller.simulation.rendering.robot_renderer import (
    RobotRenderer,
    RenderConfig,
    make_render_config,
)


def _make_ax():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    return fig, ax


# ── RenderConfig ──────────────────────────────────────────────

def test_render_config_default():
    rc = RenderConfig()
    assert rc.shape == "circle"
    assert rc.radius == 0.15


def test_make_render_config_from_dict():
    rc = make_render_config({"shape": "car", "length": 0.6, "width": 0.3})
    assert rc.shape == "car"
    assert rc.length == 0.6


def test_make_render_config_none():
    rc = make_render_config(None)
    assert rc.shape == "circle"


def test_make_render_config_passthrough():
    rc = RenderConfig(shape="rectangle")
    result = make_render_config(rc)
    assert result is rc


# ── Circle Renderer ───────────────────────────────────────────

def test_circle_render():
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "circle", "radius": 0.2, "color": "blue"})
    state = np.array([1.0, 2.0, 0.5])
    patches = renderer.render(ax, state)
    assert len(patches) >= 1  # body + heading arrow
    plt.close(fig)


def test_circle_update():
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "circle", "radius": 0.2})
    renderer.render(ax, np.array([0.0, 0.0, 0.0]))
    renderer.update(np.array([1.0, 1.0, 1.0]))
    body = renderer._patches[0]
    assert body.center == (1.0, 1.0)
    plt.close(fig)


def test_circle_clear():
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "circle"})
    renderer.render(ax, np.array([0.0, 0.0, 0.0]))
    assert len(renderer._patches) > 0
    renderer.clear()
    assert len(renderer._patches) == 0
    plt.close(fig)


def test_circle_no_heading():
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "circle", "show_heading": False})
    patches = renderer.render(ax, np.array([0.0, 0.0, 0.0]))
    assert len(patches) == 1  # body만
    plt.close(fig)


# ── Car Renderer ──────────────────────────────────────────────

def test_car_render():
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "car", "length": 0.5, "width": 0.3})
    patches = renderer.render(ax, np.array([0.0, 0.0, np.pi / 4]))
    assert len(patches) >= 1
    plt.close(fig)


def test_car_update():
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "car", "length": 0.5, "width": 0.3})
    renderer.render(ax, np.array([0.0, 0.0, 0.0]))
    renderer.update(np.array([2.0, 3.0, np.pi / 2]))
    assert len(renderer._patches) >= 1
    plt.close(fig)


# ── Rectangle Renderer ───────────────────────────────────────

def test_rectangle_render():
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "rectangle", "length": 0.4, "width": 0.35})
    patches = renderer.render(ax, np.array([1.0, -1.0, 0.0]))
    assert len(patches) >= 1
    plt.close(fig)


def test_rectangle_update():
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "rectangle"})
    renderer.render(ax, np.array([0.0, 0.0, 0.0]))
    renderer.update(np.array([5.0, 5.0, np.pi]))
    plt.close(fig)


# ── Edge Cases ────────────────────────────────────────────────

def test_render_2d_state():
    """2D 상태 (theta 없음) 처리"""
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "circle"})
    patches = renderer.render(ax, np.array([1.0, 2.0]))
    assert len(patches) >= 1
    plt.close(fig)


def test_render_rerender():
    """두 번 render 호출 시 update"""
    fig, ax = _make_ax()
    renderer = RobotRenderer({"shape": "circle"})
    renderer.render(ax, np.array([0.0, 0.0, 0.0]))
    renderer.render(ax, np.array([1.0, 1.0, 0.5]))
    plt.close(fig)
