"""
SafetyOverlay 테스트 — CBF contour, collision cone, DPCBF, effective radius
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mppi_controller.simulation.rendering.safety_overlay import SafetyOverlay


def _make_ax():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    return fig, ax


# ── CBF Contour ───────────────────────────────────────────────

def test_cbf_contour_basic():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    obstacles = [(1.0, 1.0, 0.5), (-2.0, 0.0, 0.3)]
    state = np.array([0.0, 0.0, 0.0])
    overlay.draw_cbf_contour(ax, obstacles, state, grid_resolution=20)
    assert len(overlay._artifacts) > 0
    plt.close(fig)


def test_cbf_contour_no_obstacles():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    overlay.draw_cbf_contour(ax, [], np.array([0.0, 0.0, 0.0]))
    assert len(overlay._artifacts) == 0
    plt.close(fig)


# ── Collision Cone ────────────────────────────────────────────

def test_collision_cone():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    state = np.array([0.0, 0.0, 0.0])
    obstacle = (3.0, 1.0, 0.5, -0.2, 0.1, 0.4)  # x, y, vx, vy, r (5-tuple)
    overlay.draw_collision_cone(ax, state, obstacle[:5])
    assert len(overlay._artifacts) >= 1
    plt.close(fig)


def test_collision_cone_at_obstacle():
    """이미 장애물 내부에 있는 경우"""
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    state = np.array([1.0, 1.0, 0.0])
    obstacle = (1.0, 1.0, 0.0, 0.0, 1.0)  # 로봇이 장애물 내부
    overlay.draw_collision_cone(ax, state, obstacle)
    # 아무것도 그리지 않음
    plt.close(fig)


# ── DPCBF Parabola ────────────────────────────────────────────

def test_dpcbf_parabola():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    state = np.array([0.0, 0.0, 0.0])
    obstacle = (3.0, 2.0, 0.3, -0.1, 0.5)
    overlay.draw_dpcbf_parabola(ax, state, obstacle)
    assert len(overlay._artifacts) == 2  # upper + lower
    plt.close(fig)


# ── Effective Radius ──────────────────────────────────────────

def test_effective_radius():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    obstacles = [(1.0, 1.0, 0.3), (2.0, 0.0, 0.4)]
    r_eff = [0.5, 0.7]  # 확장된 반경
    overlay.draw_effective_radius(ax, obstacles, r_eff)
    assert len(overlay._artifacts) == 2
    plt.close(fig)


def test_effective_radius_none():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    overlay.draw_effective_radius(ax, [(1, 1, 0.3)], None)
    assert len(overlay._artifacts) == 0
    plt.close(fig)


# ── Infeasibility / Shield Markers ────────────────────────────

def test_infeasibility_marker():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    state = np.array([0.0, 0.0])
    overlay.draw_infeasibility_marker(ax, state, {"qp_infeasible": True})
    assert len(overlay._artifacts) > 0
    plt.close(fig)


def test_infeasibility_not_triggered():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    overlay.draw_infeasibility_marker(ax, np.array([0, 0]), {})
    assert len(overlay._artifacts) == 0
    plt.close(fig)


def test_shield_intervention():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    state = np.array([1.0, 2.0])
    overlay.draw_shield_intervention(ax, state, {"shield_active": True})
    assert len(overlay._artifacts) > 0
    plt.close(fig)


# ── Clear ─────────────────────────────────────────────────────

def test_clear():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    overlay.draw_cbf_contour(ax, [(1, 1, 0.3)], np.array([0, 0, 0]),
                             grid_resolution=10)
    assert len(overlay._artifacts) > 0
    overlay.clear()
    assert len(overlay._artifacts) == 0
    plt.close(fig)


# ── draw_all ──────────────────────────────────────────────────

def test_draw_all():
    fig, ax = _make_ax()
    overlay = SafetyOverlay()
    obstacles = [(2.0, 1.0, 0.4)]
    state = np.array([0.0, 0.0, 0.0])
    info = {"shield_active": True}
    overlay.draw_all(ax, state, obstacles, info)
    assert len(overlay._artifacts) > 0
    plt.close(fig)
