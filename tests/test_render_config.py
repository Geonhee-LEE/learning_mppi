"""
모델별 render_config 검증
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.base_model import RobotModel
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.models.kinematic.swerve_drive_kinematic import SwerveDriveKinematic


def test_base_model_render_config():
    """RobotModel 기본 render_config는 circle"""
    class DummyModel(RobotModel):
        @property
        def state_dim(self): return 3
        @property
        def control_dim(self): return 2
        @property
        def model_type(self): return "kinematic"
        def forward_dynamics(self, state, control):
            return np.zeros_like(state)

    model = DummyModel()
    rc = model.render_config()
    assert rc["shape"] == "circle"
    assert "radius" in rc


def test_diffdrive_render_config():
    model = DifferentialDriveKinematic()
    rc = model.render_config()
    assert rc["shape"] == "circle"
    assert rc["radius"] == 0.2


def test_ackermann_render_config():
    model = AckermannKinematic(wheelbase=0.5)
    rc = model.render_config()
    assert rc["shape"] == "car"
    assert "length" in rc
    assert "width" in rc
    assert rc["length"] > 0
    assert rc["width"] > 0


def test_ackermann_render_config_wheelbase():
    """wheelbase에 따라 크기 변화"""
    model_small = AckermannKinematic(wheelbase=0.3)
    model_large = AckermannKinematic(wheelbase=1.0)
    rc_small = model_small.render_config()
    rc_large = model_large.render_config()
    assert rc_large["length"] > rc_small["length"]


def test_swerve_render_config():
    model = SwerveDriveKinematic()
    rc = model.render_config()
    assert rc["shape"] == "rectangle"
    assert "length" in rc
    assert "width" in rc


def test_all_have_color():
    """모든 모델의 render_config에 color 존재"""
    models = [
        DifferentialDriveKinematic(),
        AckermannKinematic(),
        SwerveDriveKinematic(),
    ]
    for model in models:
        rc = model.render_config()
        assert "color" in rc, f"{model.__class__.__name__} has no color"
