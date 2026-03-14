"""
SimulationHarness 테스트 — 단일/다중 컨트롤러, headless, 콜백
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.simulation.harness import SimulationHarness, ControllerEntry
from mppi_controller.utils.trajectory import generate_reference_trajectory, circle_trajectory


def _make_controller():
    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    params = MPPIParams(
        K=32, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    ctrl = MPPIController(model, params)
    return model, ctrl, params


def _make_ref_fn(N=10, dt=0.05):
    def ref_fn(t):
        return generate_reference_trajectory(
            lambda t: circle_trajectory(t, radius=3.0),
            t, N, dt,
        )
    return ref_fn


# ── 기본 기능 ─────────────────────────────────────────────────

def test_add_controller():
    harness = SimulationHarness(dt=0.05)
    model, ctrl, _ = _make_controller()
    harness.add_controller("test", ctrl, model, "blue")
    assert len(harness.entries) == 1
    assert harness.entries[0].name == "test"


def test_clear_controllers():
    harness = SimulationHarness(dt=0.05)
    model, ctrl, _ = _make_controller()
    harness.add_controller("test", ctrl, model)
    harness.clear_controllers()
    assert len(harness.entries) == 0


def test_repr():
    harness = SimulationHarness(dt=0.05)
    model, ctrl, _ = _make_controller()
    harness.add_controller("MyCtrl", ctrl, model)
    assert "MyCtrl" in repr(harness)


# ── 시뮬레이션 실행 ───────────────────────────────────────────

def test_run_single_controller():
    harness = SimulationHarness(dt=0.05, headless=True)
    model, ctrl, _ = _make_controller()
    harness.add_controller("Vanilla", ctrl, model, "blue")

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])
    results = harness.run(ref_fn, x0, duration=1.0)

    assert "Vanilla" in results
    r = results["Vanilla"]
    assert "history" in r
    assert "metrics" in r
    assert r["metrics"]["position_rmse"] >= 0
    assert len(r["history"]["state"]) > 0


def test_run_multiple_controllers():
    harness = SimulationHarness(dt=0.05, headless=True)
    model1, ctrl1, _ = _make_controller()
    model2, ctrl2, _ = _make_controller()
    harness.add_controller("A", ctrl1, model1, "blue")
    harness.add_controller("B", ctrl2, model2, "red")

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])
    results = harness.run(ref_fn, x0, duration=1.0)

    assert len(results) == 2
    assert "A" in results
    assert "B" in results


def test_run_with_noise():
    harness = SimulationHarness(dt=0.05, headless=True)
    model, ctrl, _ = _make_controller()
    noise = np.array([0.01, 0.01, 0.005])
    harness.add_controller("Noisy", ctrl, model, process_noise_std=noise)

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])
    results = harness.run(ref_fn, x0, duration=1.0)
    assert results["Noisy"]["metrics"]["position_rmse"] >= 0


def test_run_with_callback():
    harness = SimulationHarness(dt=0.05, headless=True)
    model, ctrl, _ = _make_controller()
    harness.add_controller("V", ctrl, model)

    callback_calls = []

    def on_step(t, states, controls, infos):
        callback_calls.append(t)

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])
    results = harness.run_with_callback(ref_fn, x0, duration=0.5, on_step=on_step)

    assert len(callback_calls) > 0
    assert "V" in results


# ── 메트릭 ────────────────────────────────────────────────────

def test_metrics_valid():
    harness = SimulationHarness(dt=0.05, headless=True)
    model, ctrl, _ = _make_controller()
    harness.add_controller("V", ctrl, model)

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])
    results = harness.run(ref_fn, x0, duration=1.0)

    m = results["V"]["metrics"]
    assert "position_rmse" in m
    assert "mean_solve_time" in m
    assert m["position_rmse"] >= 0
    assert m["mean_solve_time"] >= 0


# ── 플롯 ─────────────────────────────────────────────────────

def test_plot_headless():
    harness = SimulationHarness(dt=0.05, headless=True)
    model, ctrl, _ = _make_controller()
    harness.add_controller("V", ctrl, model, "blue")

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])
    results = harness.run(ref_fn, x0, duration=0.5)

    # headless plot 반환 NullFigure
    from mppi_controller.simulation.rendering.headless import NullFigure
    fig = harness.plot(results)
    assert isinstance(fig, NullFigure)


def test_plot_save():
    import matplotlib
    matplotlib.use("Agg")

    harness = SimulationHarness(dt=0.05, headless=False)
    model, ctrl, _ = _make_controller()
    harness.add_controller("V", ctrl, model, "blue")

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])
    results = harness.run(ref_fn, x0, duration=0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test.png")
        fig = harness.plot(results, save_path=save_path)
        assert os.path.exists(save_path)
        import matplotlib.pyplot as plt
        plt.close(fig)


# ── 비교 출력 ─────────────────────────────────────────────────

def test_print_comparison(capsys):
    harness = SimulationHarness(dt=0.05, headless=True)
    model, ctrl, _ = _make_controller()
    harness.add_controller("V", ctrl, model)

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])
    results = harness.run(ref_fn, x0, duration=0.5)

    harness.print_comparison(results)
    captured = capsys.readouterr()
    assert "Position RMSE" in captured.out


# ── 애니메이션 ────────────────────────────────────────────────

def test_animate_gif():
    harness = SimulationHarness(dt=0.05, headless=False)
    model, ctrl, _ = _make_controller()
    harness.add_controller("V", ctrl, model, "blue")

    ref_fn = _make_ref_fn()
    x0 = np.array([3.0, 0.0, 0.0])

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test.gif")
        results = harness.animate(
            ref_fn, x0, duration=0.5,
            save_path=save_path, fps=5,
        )
        assert os.path.exists(save_path)
        assert "V" in results


# ── ControllerEntry ───────────────────────────────────────────

def test_controller_entry():
    model, ctrl, _ = _make_controller()
    entry = ControllerEntry(name="Test", controller=ctrl, model=model)
    assert entry.name == "Test"
    assert entry.color == "blue"
    assert entry.process_noise_std is None


def test_controller_entry_with_real_model():
    model, ctrl, _ = _make_controller()
    real = DifferentialDriveKinematic(v_max=0.5, omega_max=0.5)
    entry = ControllerEntry(
        name="Mismatch", controller=ctrl, model=model,
        real_model=real,
    )
    assert entry.real_model is real
