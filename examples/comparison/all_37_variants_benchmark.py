#!/usr/bin/env python3
"""
MPPI 전체 41종 변형 통합 벤치마크

41가지 MPPI 변형을 3개 시나리오에서 비교합니다.

Groups:
  A. Foundational (6): Vanilla, Log, Tsallis, Risk-Aware, ASR, PI
  B. Smoothness   (5): Smooth, Spline, LP, Projection, Deterministic
  C. Robustness   (5): Tube, Robust, Feedback, PR, Uncertainty
  D. Exploration  (6): DIAL, CMA, SVMPC, SVG, Biased, Kernel
  E. Safety       (8): DBaS, DRPA, CSC, DualGuard, Contingency, CBF, Shield, C2U
  F. Learning     (7): Flow, SG, Transformer, TD, GN, Residual, Conformal-CBF
  G. ICRA/IROS 2026 (4): PGD, TR, RF, Step

Usage:
    PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py
    PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --scenario obstacles
    PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --all-scenarios
    PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --live --scenario obstacles
    PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --no-plot
"""

import numpy as np
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
)


# ── 공통 설정 ──────────────────────────────────────────────────

COMMON = dict(
    K=512, N=20, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
)

# 6개 그룹 색상 팔레트
GROUP_COLORS = {
    "A": "#2196F3",  # 파랑 (Foundational)
    "B": "#4CAF50",  # 초록 (Smoothness)
    "C": "#FF9800",  # 주황 (Robustness)
    "D": "#9C27B0",  # 보라 (Exploration)
    "E": "#F44336",  # 빨강 (Safety)
    "F": "#795548",  # 갈색 (Learning)
    "G": "#00838F",  # 청록 (ICRA/IROS 2026)
}

# 그룹 내 구분을 위한 shade 생성
GROUP_SHADES = {
    "A": ["#1565C0", "#1976D2", "#2196F3", "#42A5F5", "#64B5F6", "#90CAF9"],
    "B": ["#2E7D32", "#388E3C", "#4CAF50", "#66BB6A", "#81C784"],
    "C": ["#E65100", "#EF6C00", "#FF9800", "#FFA726", "#FFB74D"],
    "D": ["#6A1B9A", "#7B1FA2", "#9C27B0", "#AB47BC", "#CE93D8", "#E1BEE7"],
    "E": ["#B71C1C", "#C62828", "#D32F2F", "#E53935", "#EF5350", "#EF9A9A",
          "#FFCDD2", "#FF8A80"],
    "F": ["#3E2723", "#4E342E", "#5D4037", "#6D4C41", "#795548", "#8D6E63",
          "#A1887F"],
    "G": ["#006064", "#00838F", "#0097A7", "#00ACC1"],
}


# ── 변형 레지스트리 ──────────────────────────────────────────────

def _get_variant_registry():
    """37종 MPPI 변형 레지스트리 반환 (lazy import)"""

    registry = []

    # ── A. Foundational (6) ──
    from mppi_controller.controllers.mppi.base_mppi import MPPIController
    from mppi_controller.controllers.mppi.mppi_params import MPPIParams
    registry.append(dict(
        name="Vanilla", group="A", idx=0, ctor="standard",
        controller_cls=MPPIController, params_cls=MPPIParams,
        extra_params={},
    ))

    from mppi_controller.controllers.mppi.log_mppi import LogMPPIController
    from mppi_controller.controllers.mppi.mppi_params import LogMPPIParams
    registry.append(dict(
        name="Log", group="A", idx=1, ctor="no_cost",
        controller_cls=LogMPPIController, params_cls=LogMPPIParams,
        extra_params={"use_baseline": True},
    ))

    from mppi_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController
    from mppi_controller.controllers.mppi.mppi_params import TsallisMPPIParams
    registry.append(dict(
        name="Tsallis", group="A", idx=2, ctor="no_cost",
        controller_cls=TsallisMPPIController, params_cls=TsallisMPPIParams,
        extra_params={"tsallis_q": 1.2},
    ))

    from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
    from mppi_controller.controllers.mppi.mppi_params import RiskAwareMPPIParams
    registry.append(dict(
        name="Risk-Aware", group="A", idx=3, ctor="no_cost",
        controller_cls=RiskAwareMPPIController, params_cls=RiskAwareMPPIParams,
        extra_params={"cvar_alpha": 0.7},
    ))

    from mppi_controller.controllers.mppi.spectral_risk_mppi import ASRMPPIController
    from mppi_controller.controllers.mppi.mppi_params import ASRMPPIParams
    registry.append(dict(
        name="ASR", group="A", idx=4, ctor="standard",
        controller_cls=ASRMPPIController, params_cls=ASRMPPIParams,
        extra_params={"distortion_type": "sigmoid", "distortion_beta": 5.0},
    ))

    from mppi_controller.controllers.mppi.pi_mppi import PIMPPIController
    registry.append(dict(
        name="PI", group="A", idx=5, ctor="standard",
        controller_cls=PIMPPIController, params_cls=MPPIParams,
        extra_params={},
    ))

    # ── B. Smoothness (5) ──
    from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
    from mppi_controller.controllers.mppi.mppi_params import SmoothMPPIParams
    registry.append(dict(
        name="Smooth", group="B", idx=0, ctor="smooth",
        controller_cls=SmoothMPPIController, params_cls=SmoothMPPIParams,
        extra_params={"jerk_weight": 1.0},
    ))

    from mppi_controller.controllers.mppi.spline_mppi import SplineMPPIController
    from mppi_controller.controllers.mppi.mppi_params import SplineMPPIParams
    registry.append(dict(
        name="Spline", group="B", idx=1, ctor="no_cost",
        controller_cls=SplineMPPIController, params_cls=SplineMPPIParams,
        extra_params={"spline_num_knots": 8, "spline_degree": 3},
    ))

    from mppi_controller.controllers.mppi.lp_mppi import LPMPPIController
    from mppi_controller.controllers.mppi.mppi_params import LPMPPIParams
    registry.append(dict(
        name="LP", group="B", idx=2, ctor="standard",
        controller_cls=LPMPPIController, params_cls=LPMPPIParams,
        extra_params={"cutoff_freq": 3.0, "filter_order": 3},
    ))

    from mppi_controller.controllers.mppi.projection_mppi import ProjectionMPPIController
    from mppi_controller.controllers.mppi.mppi_params import ProjectionMPPIParams
    registry.append(dict(
        name="Projection", group="B", idx=3, ctor="standard",
        controller_cls=ProjectionMPPIController, params_cls=ProjectionMPPIParams,
        extra_params={"jerk_limit": 5.0, "use_jerk_constraint": True},
    ))

    from mppi_controller.controllers.mppi.deterministic_mppi import DeterministicMPPIController
    from mppi_controller.controllers.mppi.mppi_params import DeterministicMPPIParams
    registry.append(dict(
        name="Deterministic", group="B", idx=4, ctor="standard",
        controller_cls=DeterministicMPPIController, params_cls=DeterministicMPPIParams,
        extra_params={"sampling_method": "halton", "n_cem_iters": 3},
    ))

    # ── C. Robustness (5) ──
    from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
    from mppi_controller.controllers.mppi.mppi_params import TubeMPPIParams
    registry.append(dict(
        name="Tube", group="C", idx=0, ctor="tube",
        controller_cls=TubeMPPIController, params_cls=TubeMPPIParams,
        extra_params={
            "tube_enabled": True, "tube_margin": 0.3,
            "K_fb": np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        },
    ))

    from mppi_controller.controllers.mppi.robust_mppi import RobustMPPIController
    from mppi_controller.controllers.mppi.mppi_params import RobustMPPIParams
    registry.append(dict(
        name="Robust", group="C", idx=1, ctor="standard",
        controller_cls=RobustMPPIController, params_cls=RobustMPPIParams,
        extra_params={"disturbance_mode": "gaussian", "disturbance_std": [0.05, 0.05, 0.02]},
    ))

    from mppi_controller.controllers.mppi.feedback_mppi import FeedbackMPPIController
    from mppi_controller.controllers.mppi.mppi_params import FeedbackMPPIParams
    registry.append(dict(
        name="Feedback", group="C", idx=2, ctor="standard",
        controller_cls=FeedbackMPPIController, params_cls=FeedbackMPPIParams,
        extra_params={"reuse_steps": 3},
    ))

    from mppi_controller.controllers.mppi.parameter_robust_mppi import ParameterRobustMPPIController
    from mppi_controller.controllers.mppi.mppi_params import ParameterRobustMPPIParams
    registry.append(dict(
        name="PR", group="C", idx=3, ctor="standard",
        controller_cls=ParameterRobustMPPIController, params_cls=ParameterRobustMPPIParams,
        extra_params={
            "n_particles": 10, "param_name": "wheelbase",
            "param_nominal": 0.5, "param_std": 0.1,
            "param_min": 0.3, "param_max": 0.7,
        },
    ))

    from mppi_controller.controllers.mppi.uncertainty_mppi import UncertaintyMPPIController
    from mppi_controller.controllers.mppi.mppi_params import UncertaintyMPPIParams
    registry.append(dict(
        name="Uncertainty", group="C", idx=4, ctor="standard",
        controller_cls=UncertaintyMPPIController, params_cls=UncertaintyMPPIParams,
        extra_params={"exploration_factor": 0.5},
    ))

    # ── D. Exploration (6) ──
    from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
    from mppi_controller.controllers.mppi.mppi_params import DIALMPPIParams
    registry.append(dict(
        name="DIAL", group="D", idx=0, ctor="standard",
        controller_cls=DIALMPPIController, params_cls=DIALMPPIParams,
        extra_params={"n_diffuse_init": 5, "n_diffuse": 3, "use_reward_normalization": True},
    ))

    from mppi_controller.controllers.mppi.cma_mppi import CMAMPPIController
    from mppi_controller.controllers.mppi.mppi_params import CMAMPPIParams
    registry.append(dict(
        name="CMA", group="D", idx=1, ctor="standard",
        controller_cls=CMAMPPIController, params_cls=CMAMPPIParams,
        extra_params={"n_iters_init": 5, "n_iters": 3, "elite_ratio": 0.1},
    ))

    from mppi_controller.controllers.mppi.stein_variational_mppi import SteinVariationalMPPIController
    from mppi_controller.controllers.mppi.mppi_params import SteinVariationalMPPIParams
    registry.append(dict(
        name="SVMPC", group="D", idx=2, ctor="no_cost",
        controller_cls=SteinVariationalMPPIController, params_cls=SteinVariationalMPPIParams,
        extra_params={"svgd_num_iterations": 3, "svgd_step_size": 0.01},
    ))

    from mppi_controller.controllers.mppi.svg_mppi import SVGMPPIController
    from mppi_controller.controllers.mppi.mppi_params import SVGMPPIParams
    registry.append(dict(
        name="SVG", group="D", idx=3, ctor="standard",
        controller_cls=SVGMPPIController, params_cls=SVGMPPIParams,
        extra_params={
            "svg_num_guide_particles": 32, "svgd_num_iterations": 3,
            "svg_guide_step_size": 0.01,
        },
    ))

    from mppi_controller.controllers.mppi.biased_mppi import BiasedMPPIController
    from mppi_controller.controllers.mppi.mppi_params import BiasedMPPIParams
    registry.append(dict(
        name="Biased", group="D", idx=4, ctor="standard",
        controller_cls=BiasedMPPIController, params_cls=BiasedMPPIParams,
        extra_params={
            "ancillary_types": ["pure_pursuit", "braking", "max_speed"],
            "samples_per_policy": 10, "policy_noise_scale": 0.3,
        },
    ))

    from mppi_controller.controllers.mppi.kernel_mppi import KernelMPPIController
    from mppi_controller.controllers.mppi.mppi_params import KernelMPPIParams
    registry.append(dict(
        name="Kernel", group="D", idx=5, ctor="standard",
        controller_cls=KernelMPPIController, params_cls=KernelMPPIParams,
        extra_params={"num_support_pts": 16, "kernel_bandwidth": 0.5},
    ))

    # ── E. Safety (8) ──
    from mppi_controller.controllers.mppi.dbas_mppi import DBaSMPPIController
    from mppi_controller.controllers.mppi.mppi_params import DBaSMPPIParams
    registry.append(dict(
        name="DBaS", group="E", idx=0, ctor="obstacle_in_params",
        controller_cls=DBaSMPPIController, params_cls=DBaSMPPIParams,
        extra_params={"barrier_weight": 50.0, "barrier_gamma": 0.9},
        obstacle_field="dbas_obstacles",
        skip_obstacle_cost=True,
    ))

    from mppi_controller.controllers.mppi.drpa_mppi import DRPAMPPIController
    from mppi_controller.controllers.mppi.mppi_params import DRPAMPPIParams
    registry.append(dict(
        name="DRPA", group="E", idx=1, ctor="obstacle_in_params",
        controller_cls=DRPAMPPIController, params_cls=DRPAMPPIParams,
        extra_params={"repulsive_gain": 1.0, "influence_distance": 1.5},
        obstacle_field="obstacles",
    ))

    from mppi_controller.controllers.mppi.csc_mppi import CSCMPPIController
    from mppi_controller.controllers.mppi.mppi_params import CSCMPPIParams
    registry.append(dict(
        name="CSC", group="E", idx=2, ctor="obstacle_in_params",
        controller_cls=CSCMPPIController, params_cls=CSCMPPIParams,
        extra_params={"safety_margin": 0.2, "use_projection": True, "use_clustering": True},
        obstacle_field="obstacles",
    ))

    from mppi_controller.controllers.mppi.dualguard_mppi import DualGuardMPPIController
    from mppi_controller.controllers.mppi.mppi_params import DualGuardMPPIParams
    registry.append(dict(
        name="DualGuard", group="E", idx=3, ctor="obstacle_in_params",
        controller_cls=DualGuardMPPIController, params_cls=DualGuardMPPIParams,
        extra_params={"safety_mode": "soft", "safety_penalty": 100.0},
        obstacle_field="obstacles",
    ))

    from mppi_controller.controllers.mppi.contingency_mppi import ContingencyMPPIController
    from mppi_controller.controllers.mppi.mppi_params import ContingencyMPPIParams
    registry.append(dict(
        name="Contingency", group="E", idx=4, ctor="contingency",
        controller_cls=ContingencyMPPIController, params_cls=ContingencyMPPIParams,
        extra_params={
            "contingency_weight": 1.0, "contingency_horizon": 10,
            "contingency_samples": 64, "n_checkpoints": 3,
        },
    ))

    from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
    from mppi_controller.controllers.mppi.mppi_params import CBFMPPIParams
    registry.append(dict(
        name="CBF", group="E", idx=5, ctor="obstacle_in_params",
        controller_cls=CBFMPPIController, params_cls=CBFMPPIParams,
        extra_params={"cbf_weight": 100.0, "cbf_alpha": 0.5},
        obstacle_field="cbf_obstacles",
    ))

    from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
    from mppi_controller.controllers.mppi.mppi_params import ShieldMPPIParams
    registry.append(dict(
        name="Shield", group="E", idx=6, ctor="obstacle_in_params",
        controller_cls=ShieldMPPIController, params_cls=ShieldMPPIParams,
        extra_params={"shield_enabled": True, "cbf_weight": 100.0, "cbf_alpha": 0.5},
        obstacle_field="cbf_obstacles",
    ))

    from mppi_controller.controllers.mppi.c2u_mppi import C2UMPPIController
    from mppi_controller.controllers.mppi.mppi_params import C2UMPPIParams
    registry.append(dict(
        name="C2U", group="E", idx=7, ctor="obstacle_in_params",
        controller_cls=C2UMPPIController, params_cls=C2UMPPIParams,
        extra_params={"chance_alpha": 0.05, "chance_cost_weight": 500.0},
        obstacle_field="cc_obstacles",
    ))

    # ── F. Learning (7) — torch-dependent ──
    try:
        import torch  # noqa: F401

        from mppi_controller.controllers.mppi.flow_mppi import FlowMPPIController
        from mppi_controller.controllers.mppi.mppi_params import FlowMPPIParams
        registry.append(dict(
            name="Flow", group="F", idx=0, ctor="standard",
            controller_cls=FlowMPPIController, params_cls=FlowMPPIParams,
            extra_params={
                "flow_mode": "blend", "flow_blend_ratio": 0.3,
                "flow_online_training": False,
            },
        ))

        from mppi_controller.controllers.mppi.score_guided_mppi import SGMPPIController
        from mppi_controller.controllers.mppi.mppi_params import SGMPPIParams
        registry.append(dict(
            name="SG", group="F", idx=1, ctor="standard",
            controller_cls=SGMPPIController, params_cls=SGMPPIParams,
            extra_params={
                "guidance_scale": 0.5, "n_guide_iters": 1,
                "score_online_training": False,
            },
        ))

        from mppi_controller.controllers.mppi.transformer_mppi import TransformerMPPIController
        from mppi_controller.controllers.mppi.mppi_params import TransformerMPPIParams
        registry.append(dict(
            name="Transformer", group="F", idx=2, ctor="standard",
            controller_cls=TransformerMPPIController, params_cls=TransformerMPPIParams,
            extra_params={
                "transformer_hidden_dim": 64, "transformer_n_heads": 2,
                "transformer_n_layers": 1, "transformer_context_length": 10,
                "use_transformer_init": True, "blend_ratio": 0.5,
                "online_learning": False,
            },
        ))

        from mppi_controller.controllers.mppi.td_mppi import TDMPPIController
        from mppi_controller.controllers.mppi.mppi_params import TDMPPIParams
        registry.append(dict(
            name="TD", group="F", idx=3, ctor="standard",
            controller_cls=TDMPPIController, params_cls=TDMPPIParams,
            extra_params={
                "use_terminal_value": True, "value_weight": 0.5,
                "td_update_interval": 5,
            },
        ))

        from mppi_controller.controllers.mppi.gn_mppi import GNMPPIController
        from mppi_controller.controllers.mppi.mppi_params import GNMPPIParams
        registry.append(dict(
            name="GN", group="F", idx=4, ctor="standard",
            controller_cls=GNMPPIController, params_cls=GNMPPIParams,
            extra_params={"n_gn_iters": 3, "gn_step_size": 0.1, "use_gn_update": True},
        ))

        from mppi_controller.controllers.mppi.residual_mppi import ResidualMPPIController
        from mppi_controller.controllers.mppi.mppi_params import ResidualMPPIParams
        registry.append(dict(
            name="Residual", group="F", idx=5, ctor="standard",
            controller_cls=ResidualMPPIController, params_cls=ResidualMPPIParams,
            extra_params={"policy_type": "feedback", "residual_scale": 1.0},
        ))

        from mppi_controller.controllers.mppi.conformal_cbf_mppi import ConformalCBFMPPIController
        from mppi_controller.controllers.mppi.mppi_params import ConformalCBFMPPIParams
        registry.append(dict(
            name="Conformal-CBF", group="F", idx=6, ctor="obstacle_in_params",
            controller_cls=ConformalCBFMPPIController, params_cls=ConformalCBFMPPIParams,
            extra_params={
                "cp_enabled": True, "shield_enabled": True,
                "cbf_weight": 100.0, "cbf_alpha": 0.5,
            },
            obstacle_field="cbf_obstacles",
        ))

    except ImportError:
        print("  [WARN] torch not available -- F-Learning group (7 variants) skipped")

    # ── G. ICRA/IROS 2026 (4) ──
    from mppi_controller.controllers.mppi.pgd_mppi import PGDMPPIController
    from mppi_controller.controllers.mppi.mppi_params import PGDMPPIParams
    registry.append(dict(
        name="PGD", group="G", idx=0, ctor="standard",
        controller_cls=PGDMPPIController, params_cls=PGDMPPIParams,
        extra_params={"n_grad_steps": 3, "step_size": 0.8, "adapt_covariance": True},
    ))

    from mppi_controller.controllers.mppi.tr_mppi import TRMPPIController
    from mppi_controller.controllers.mppi.mppi_params import TRMPPIParams
    registry.append(dict(
        name="TR", group="G", idx=1, ctor="standard",
        controller_cls=TRMPPIController, params_cls=TRMPPIParams,
        extra_params={"trust_region_radius": 2.0, "use_deterministic_sampling": True},
    ))

    from mppi_controller.controllers.mppi.rf_mppi import RFMPPIController
    from mppi_controller.controllers.mppi.mppi_params import RFMPPIParams
    registry.append(dict(
        name="RF", group="G", idx=2, ctor="standard",
        controller_cls=RFMPPIController, params_cls=RFMPPIParams,
        extra_params={"n_knots": 6, "sample_velocity_knots": True},
    ))

    from mppi_controller.controllers.mppi.step_mppi import StepMPPIController
    from mppi_controller.controllers.mppi.mppi_params import StepMPPIParams
    registry.append(dict(
        name="Step", group="G", idx=3, ctor="standard",
        controller_cls=StepMPPIController, params_cls=StepMPPIParams,
        extra_params={"use_learned_proposal": True, "online_training": False},
    ))

    return registry


# ── 시나리오 정의 ──────────────────────────────────────────────

def get_scenarios():
    """3개 벤치마크 시나리오"""
    return {
        "simple": {
            "name": "A. Simple (No Obstacles)",
            "obstacles": [],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 12.0,
            "description": "Circle tracking, no obstacles -- baseline",
        },
        "obstacles": {
            "name": "B. Obstacles (3)",
            "obstacles": [
                (2.5, 1.5, 0.5),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.5),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 12.0,
            "description": "Circle tracking with 3 obstacles",
        },
        "dense_obstacles": {
            "name": "C. Dense Obstacles (6)",
            "obstacles": [
                (2.5, 1.5, 0.4),
                (0.0, 3.0, 0.4),
                (-2.0, -1.0, 0.4),
                (-1.0, 2.5, 0.35),
                (1.5, -2.0, 0.35),
                (-2.5, 0.5, 0.35),
            ],
            "trajectory_fn": lambda t: circle_trajectory(t, radius=3.0),
            "initial_state": np.array([3.0, 0.0, np.pi / 2]),
            "duration": 15.0,
            "description": "Circle tracking with 6 dense obstacles",
        },
    }


# ── 비용 함수 팩토리 ─────────────────────────────────────────────

def _make_cost(params, obstacles, skip_obstacle=False):
    """CompositeMPPICost 팩토리"""
    costs = [
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ]
    if obstacles and not skip_obstacle:
        costs.append(ObstacleCost(obstacles, safety_margin=0.2, cost_weight=2000.0))
    return CompositeMPPICost(costs)


# ── 컨트롤러 생성 ─────────────────────────────────────────────

def _build_controller(variant, model, obstacles):
    """5가지 생성자 패턴에 따라 컨트롤러 생성"""
    ctor_type = variant["ctor"]
    params_kw = {**COMMON, **variant["extra_params"]}

    # 장애물을 params에 주입하는 패턴
    obstacle_field = variant.get("obstacle_field")
    if obstacle_field and obstacles:
        params_kw[obstacle_field] = obstacles

    params = variant["params_cls"](**params_kw)
    skip_obstacle = variant.get("skip_obstacle_cost", False)
    cost = _make_cost(params, obstacles, skip_obstacle=skip_obstacle)

    if ctor_type == "standard":
        return variant["controller_cls"](model, params, cost_function=cost)

    elif ctor_type in ("smooth", "tube", "no_cost"):
        # These controllers do not accept cost_function kwarg
        return variant["controller_cls"](model, params)

    elif ctor_type == "contingency":
        # ContingencyMPPIController has extra safety_cost_function
        safety_cost = _make_cost(params, obstacles)
        return variant["controller_cls"](
            model, params, cost_function=cost, safety_cost_function=safety_cost
        )

    elif ctor_type == "obstacle_in_params":
        return variant["controller_cls"](model, params, cost_function=cost)

    else:
        raise ValueError(f"Unknown ctor type: {ctor_type}")


# ── 시뮬레이션 ─────────────────────────────────────────────────

def run_single(model, controller, scenario, seed=42):
    """단일 컨트롤러 시뮬레이션"""
    np.random.seed(seed)

    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)
    trajectory_fn = scenario["trajectory_fn"]

    state = scenario["initial_state"].copy()
    states = [state.copy()]
    controls_hist = []
    solve_times = []
    infos = []

    for step in range(num_steps):
        t = step * dt
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt)

        t_start = time.time()
        control, info = controller.compute_control(state, ref)
        solve_time = time.time() - t_start

        state_dot = model.forward_dynamics(state, control)
        state = state + state_dot * dt

        states.append(state.copy())
        controls_hist.append(control.copy())
        solve_times.append(solve_time)
        infos.append(info)

    return {
        "states": np.array(states),
        "controls": np.array(controls_hist) if controls_hist else np.array([]),
        "solve_times": np.array(solve_times),
        "infos": infos,
    }


# ── 메트릭 추출 ─────────────────────────────────────────────────

def extract_metrics(history, scenario):
    """RMSE/ESS/Collisions/MinClear 등 메트릭 계산"""
    states = history["states"]
    controls = history["controls"]
    trajectory_fn = scenario["trajectory_fn"]
    dt = COMMON["dt"]
    obstacles = scenario["obstacles"]

    # RMSE + MaxError
    errors = []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt)
        err = np.sqrt((st[0] - ref[0]) ** 2 + (st[1] - ref[1]) ** 2)
        errors.append(err)
    errors = np.array(errors)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_error = float(np.max(errors))

    # Collisions & MinClearance
    n_collisions = 0
    min_clearance = float("inf")
    for st in states:
        for ox, oy, r in obstacles:
            dist = np.sqrt((st[0] - ox) ** 2 + (st[1] - oy) ** 2)
            clearance = dist - r
            min_clearance = min(min_clearance, clearance)
            if clearance < 0:
                n_collisions += 1

    # ESS
    ess_list = [
        info.get("ess", 0.0) for info in history["infos"]
        if isinstance(info, dict) and "ess" in info
    ]

    # Control rate (mean |delta u|)
    ctrl_rate = 0.0
    if len(controls) > 1:
        diffs = np.diff(controls, axis=0)
        ctrl_rate = float(np.mean(np.abs(diffs)))

    return {
        "rmse": rmse,
        "max_error": max_error,
        "n_collisions": n_collisions,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "mean_solve_ms": float(np.mean(history["solve_times"])) * 1000,
        "mean_ess": float(np.mean(ess_list)) if ess_list else 0.0,
        "ctrl_rate": ctrl_rate,
        "errors": errors,
    }


# ── 결과 테이블 ─────────────────────────────────────────────────

GROUP_NAMES = {
    "A": "Foundational",
    "B": "Smoothness",
    "C": "Robustness",
    "D": "Exploration",
    "E": "Safety",
    "F": "Learning",
    "G": "ICRA/IROS 2026",
}


def print_results_table(results, scenario):
    """37행 포맷 테이블 출력"""
    has_obstacles = len(scenario["obstacles"]) > 0

    header = (
        f"  {'#':<3} {'Group':<14} {'Variant':<16} {'RMSE':>8} {'MaxErr':>8} "
        f"{'ESS':>8} {'SolveMs':>9} {'CtrlRate':>9}"
    )
    if has_obstacles:
        header += f" {'MinClear':>9} {'Collis':>7}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    current_group = None
    for i, r in enumerate(results):
        if r["group"] != current_group:
            current_group = r["group"]
            if i > 0:
                print(f"  {'─' * (len(header) - 2)}")

        group_label = f"{r['group']}-{GROUP_NAMES[r['group']]}"
        line = (
            f"  {i+1:<3} {group_label:<14} {r['name']:<16} "
            f"{r['rmse']:>8.4f} {r['max_error']:>8.4f} "
            f"{r['mean_ess']:>8.1f} {r['mean_solve_ms']:>9.2f} {r['ctrl_rate']:>9.4f}"
        )
        if has_obstacles:
            line += f" {r['min_clearance']:>9.3f} {r['n_collisions']:>7d}"
        print(line)


# ── 정적 플롯 ─────────────────────────────────────────────────

def plot_static(results, scenario, scenario_key):
    """3x4 PNG 그리드 생성"""
    dt = COMMON["dt"]
    trajectory_fn = scenario["trajectory_fn"]
    duration = scenario["duration"]
    has_obstacles = len(scenario["obstacles"]) > 0

    fig, axes = plt.subplots(3, 4, figsize=(26, 18))
    fig.suptitle(
        f"All 37 MPPI Variants Benchmark [{scenario_key}]: {scenario['name']}\n"
        f"K={COMMON['K']}, N={COMMON['N']}, dt={COMMON['dt']}, T={duration}s",
        fontsize=14, fontweight="bold",
    )

    # Helper: get color for a result
    def _color(r):
        shades = GROUP_SHADES[r["group"]]
        return shades[min(r["idx"], len(shades) - 1)]

    # (0,0) XY trajectories (all, colored by group)
    ax = axes[0, 0]
    t_arr = np.linspace(0, duration, 500)
    ref_xy = np.array([trajectory_fn(t)[:2] for t in t_arr])
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", alpha=0.4, linewidth=1, label="Ref")

    for r in results:
        ax.plot(r["states"][:, 0], r["states"][:, 1], color=_color(r),
                linewidth=1.0, alpha=0.6)

    for ox, oy, rad in scenario["obstacles"]:
        ax.add_patch(Circle((ox, oy), rad, facecolor="#FF5252", edgecolor="red",
                            alpha=0.3, linewidth=1.5))

    # Legend by group
    for g, gname in GROUP_NAMES.items():
        c = GROUP_COLORS[g]
        count = sum(1 for r in results if r["group"] == g)
        if count > 0:
            ax.plot([], [], color=c, linewidth=2, label=f"{g}: {gname} ({count})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectories (by Group)")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # (0,1) RMSE bar chart
    ax = axes[0, 1]
    names = [r["name"] for r in results]
    rmses = [r["rmse"] for r in results]
    colors = [_color(r) for r in results]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, rmses, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Position RMSE (m)")
    ax.set_title("Tracking Accuracy")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # (0,2) Solve Time bar chart
    ax = axes[0, 2]
    solve_times = [r["mean_solve_ms"] for r in results]
    ax.barh(y_pos, solve_times, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Mean Solve Time (ms)")
    ax.set_title("Computational Cost")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # (0,3) Error time-series (top 5 by RMSE)
    ax = axes[0, 3]
    top5 = sorted(results, key=lambda x: x["rmse"])[:5]
    for r in top5:
        t_plot = np.arange(len(r["errors"])) * dt
        ax.plot(t_plot, r["errors"], color=_color(r), label=r["name"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Error Time-Series (Top 5)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (1,0) ESS bar chart
    ax = axes[1, 0]
    ess_vals = [r["mean_ess"] for r in results]
    ax.barh(y_pos, ess_vals, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Mean ESS")
    ax.set_title("Sampling Efficiency (ESS)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # (1,1) Control Rate bar chart
    ax = axes[1, 1]
    ctrl_rates = [r["ctrl_rate"] for r in results]
    ax.barh(y_pos, ctrl_rates, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Control Rate (mean |Δu|)")
    ax.set_title("Control Smoothness")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # (1,2) RMSE vs Solve Time scatter
    ax = axes[1, 2]
    for r in results:
        ax.scatter(r["mean_solve_ms"], r["rmse"], s=60, color=_color(r),
                   alpha=0.7, edgecolors="black", linewidths=0.3)
    # Group legend
    for g, gname in GROUP_NAMES.items():
        c = GROUP_COLORS[g]
        if any(r["group"] == g for r in results):
            ax.scatter([], [], s=80, color=c, label=f"{g}: {gname}")
    ax.set_xlabel("Mean Solve Time (ms)")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Accuracy vs Speed")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    # (1,3) MinClearance/Collisions (if obstacles) or ESS time-series
    ax = axes[1, 3]
    if has_obstacles:
        clearances = [r["min_clearance"] for r in results]
        bar_colors_safe = []
        for r, cl in zip(results, clearances):
            bar_colors_safe.append("red" if r["n_collisions"] > 0 else _color(r))
        ax.barh(y_pos, clearances, color=bar_colors_safe, alpha=0.8, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=5)
        ax.set_xlabel("Min Clearance (m)")
        ax.set_title("Safety (MinClearance)")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()
    else:
        for r in top5:
            ess_ts = [
                info.get("ess", 0.0) for info in r["infos"]
                if isinstance(info, dict) and "ess" in info
            ]
            if ess_ts:
                t_plot = np.arange(len(ess_ts)) * dt
                ax.plot(t_plot, ess_ts, color=_color(r), label=r["name"], linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ESS")
        ax.set_title("ESS Time-Series (Top 5)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # (2,0) Group-averaged RMSE
    ax = axes[2, 0]
    group_keys = ["A", "B", "C", "D", "E", "F"]
    group_rmses = []
    group_labels = []
    group_bar_colors = []
    for g in group_keys:
        group_results = [r for r in results if r["group"] == g]
        if group_results:
            group_rmses.append(np.mean([r["rmse"] for r in group_results]))
            group_labels.append(f"{g}: {GROUP_NAMES[g]}")
            group_bar_colors.append(GROUP_COLORS[g])
    ax.bar(group_labels, group_rmses, color=group_bar_colors, alpha=0.8)
    ax.set_ylabel("Mean RMSE (m)")
    ax.set_title("Group Average RMSE")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # (2,1) Group-averaged Solve Time
    ax = axes[2, 1]
    group_solve = []
    for g in group_keys:
        group_results = [r for r in results if r["group"] == g]
        if group_results:
            group_solve.append(np.mean([r["mean_solve_ms"] for r in group_results]))
    ax.bar(group_labels, group_solve, color=group_bar_colors, alpha=0.8)
    ax.set_ylabel("Mean Solve Time (ms)")
    ax.set_title("Group Average Solve Time")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # (2,2) Top/Bottom per category
    ax = axes[2, 2]
    ax.axis("off")
    ax.set_title("Best by Category")

    best_accuracy = min(results, key=lambda x: x["rmse"])
    best_speed = min(results, key=lambda x: x["mean_solve_ms"])
    best_smoothness = min(results, key=lambda x: x["ctrl_rate"])
    best_ess = max(results, key=lambda x: x["mean_ess"])

    text_lines = [
        f"Best Accuracy:    {best_accuracy['name']:<16} RMSE={best_accuracy['rmse']:.4f}m",
        f"Best Speed:       {best_speed['name']:<16} Solve={best_speed['mean_solve_ms']:.2f}ms",
        f"Best Smoothness:  {best_smoothness['name']:<16} Rate={best_smoothness['ctrl_rate']:.4f}",
        f"Best ESS:         {best_ess['name']:<16} ESS={best_ess['mean_ess']:.1f}",
    ]
    if has_obstacles:
        safe_results = [r for r in results if r["n_collisions"] == 0]
        if safe_results:
            best_safety = max(safe_results, key=lambda x: x["min_clearance"])
            text_lines.append(
                f"Best Safety:      {best_safety['name']:<16} Clear={best_safety['min_clearance']:.3f}m"
            )
        n_crashed = sum(1 for r in results if r["n_collisions"] > 0)
        text_lines.append(f"\nCollision-free: {len(results) - n_crashed}/{len(results)}")

    text_lines.append(f"\n--- Settings ---")
    text_lines.append(f"K={COMMON['K']}  N={COMMON['N']}  dt={COMMON['dt']}  T={duration}s")
    text_lines.append(f"Scenario: {scenario['name']}")
    text_lines.append(f"Variants tested: {len(results)}/37")

    ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            va="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    # (2,3) Collision heatmap (if obstacles) or recommendation
    ax = axes[2, 3]
    if has_obstacles:
        ax.set_title("RMSE vs MinClearance")
        for r in results:
            marker = "x" if r["n_collisions"] > 0 else "o"
            ax.scatter(r["min_clearance"], r["rmse"], s=60, color=_color(r),
                       marker=marker, alpha=0.7, edgecolors="black", linewidths=0.3)
        for g, gname in GROUP_NAMES.items():
            c = GROUP_COLORS[g]
            if any(r["group"] == g for r in results):
                ax.scatter([], [], s=80, color=c, label=f"{g}: {gname}")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Collision boundary")
        ax.set_xlabel("Min Clearance (m)")
        ax.set_ylabel("RMSE (m)")
        ax.legend(fontsize=5, loc="upper right")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")
        ax.set_title("Recommendations")
        recs = [
            "Real-time:     Vanilla, Log, Spline",
            "Accuracy:      SVG, SVMPC, CMA",
            "Smoothness:    LP, Smooth, Projection",
            "Robustness:    Tube, Robust, Feedback",
            "Safety:        DBaS, Shield, DualGuard",
            "Exploration:   DIAL, Biased, SVG",
            "Learning:      GN, TD, Transformer",
        ]
        ax.text(0.05, 0.95, "\n".join(recs), transform=ax.transAxes,
                va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))

    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/all_37_variants_{scenario_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


# ── Live 애니메이션 ─────────────────────────────────────────────

def run_live(args):
    """실시간 37-way 비교 -> GIF/MP4 저장"""
    from matplotlib.animation import FuncAnimation

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]
    trajectory_fn = scenario["trajectory_fn"]

    model = DifferentialDriveKinematic(wheelbase=0.5)
    dt = COMMON["dt"]
    N = COMMON["N"]
    duration = scenario["duration"]
    num_steps = int(duration / dt)

    np.random.seed(args.seed)
    registry = _get_variant_registry()

    # 컨트롤러 생성
    controllers = {}
    ctrl_colors = {}
    for v in registry:
        try:
            ctrl = _build_controller(v, model, scenario["obstacles"])
            controllers[v["name"]] = ctrl
            shades = GROUP_SHADES[v["group"]]
            ctrl_colors[v["name"]] = shades[min(v["idx"], len(shades) - 1)]
        except Exception as e:
            print(f"  [SKIP] {v['name']}: {e}")

    n_variants = len(controllers)
    print(f"\n{'=' * 60}")
    print(f"  All Variants Live -- {scenario['name']}  ({n_variants} variants)")
    print(f"  {scenario['description']}")
    print(f"  Duration: {duration}s | Frames: {num_steps}")
    print(f"{'=' * 60}")

    # 상태 초기화
    states = {k: scenario["initial_state"].copy() for k in controllers}
    sim_t = [0.0]
    data = {
        k: {"xy": [], "errors": [], "times": []}
        for k in controllers
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"All {n_variants} MPPI Variants Live -- {scenario['name']}",
        fontsize=13, fontweight="bold",
    )

    # (0,0) XY
    ax_xy = axes[0, 0]
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Trajectories")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    for ox, oy, r in scenario["obstacles"]:
        ax_xy.add_patch(Circle((ox, oy), r, color="red", alpha=0.3))

    ref_t = np.linspace(0, duration, 500)
    ref_pts = np.array([trajectory_fn(t) for t in ref_t])
    ax_xy.plot(ref_pts[:, 0], ref_pts[:, 1], "k--", alpha=0.3, linewidth=1)

    lines_xy = {}
    for name in controllers:
        lines_xy[name], = ax_xy.plot([], [], color=ctrl_colors[name], linewidth=1, alpha=0.6)

    # (0,1) Error
    ax_err = axes[0, 1]
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Error (m)")
    ax_err.set_title("Tracking Error (Top 5)")
    ax_err.grid(True, alpha=0.3)

    # (0,2) RMSE bars (updated live)
    ax_rmse = axes[0, 2]
    ax_rmse.set_title("Running RMSE")
    ax_rmse.grid(True, alpha=0.3, axis="x")

    # (1,0-2) Stats text
    ax_stats = axes[1, 0]
    ax_stats.axis("off")
    ax_stats.set_title("Statistics")
    stats_text = ax_stats.text(0.02, 0.98, "", transform=ax_stats.transAxes,
                                va="top", fontsize=6, family="monospace")

    axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    plt.tight_layout()

    def update(frame):
        if frame >= num_steps:
            return

        t = sim_t[0]
        ref = generate_reference_trajectory(trajectory_fn, t, N, dt)
        ref_pt = trajectory_fn(t)[:2]

        for name, ctrl in controllers.items():
            try:
                control, info = ctrl.compute_control(states[name], ref)
                state_dot = model.forward_dynamics(states[name], control)
                states[name] = states[name] + state_dot * dt
            except Exception:
                pass

            data[name]["xy"].append(states[name][:2].copy())
            data[name]["errors"].append(np.linalg.norm(states[name][:2] - ref_pt))
            data[name]["times"].append(t)

        sim_t[0] += dt

        # Update XY lines
        for name in controllers:
            xy = np.array(data[name]["xy"])
            if len(xy) > 0:
                lines_xy[name].set_data(xy[:, 0], xy[:, 1])

        ax_xy.relim()
        ax_xy.autoscale_view()
        ax_xy.set_aspect("equal")

        # Update error (top 5)
        if frame % 10 == 0 and frame > 0:
            ax_err.clear()
            ax_err.set_xlabel("Time (s)")
            ax_err.set_ylabel("Error (m)")
            ax_err.set_title("Tracking Error (Top 5)")
            ax_err.grid(True, alpha=0.3)

            current_rmses = {}
            for name in controllers:
                errs = data[name]["errors"]
                if errs:
                    current_rmses[name] = np.sqrt(np.mean(np.array(errs) ** 2))
            top5_names = sorted(current_rmses, key=current_rmses.get)[:5]
            for name in top5_names:
                times = np.array(data[name]["times"])
                errs = np.array(data[name]["errors"])
                ax_err.plot(times, errs, color=ctrl_colors[name], label=name, linewidth=1)
            ax_err.legend(fontsize=5)

        # Update RMSE bars every 20 frames
        if frame % 20 == 0 and frame > 0:
            ax_rmse.clear()
            ax_rmse.set_title("Running RMSE")
            ax_rmse.grid(True, alpha=0.3, axis="x")

            rmse_dict = {}
            for name in controllers:
                errs = data[name]["errors"]
                if errs:
                    rmse_dict[name] = np.sqrt(np.mean(np.array(errs) ** 2))

            sorted_names = sorted(rmse_dict, key=rmse_dict.get)
            bar_y = np.arange(len(sorted_names))
            bar_vals = [rmse_dict[n] for n in sorted_names]
            bar_cols = [ctrl_colors[n] for n in sorted_names]
            ax_rmse.barh(bar_y, bar_vals, color=bar_cols, alpha=0.8, height=0.7)
            ax_rmse.set_yticks(bar_y)
            ax_rmse.set_yticklabels(sorted_names, fontsize=4)
            ax_rmse.set_xlabel("RMSE (m)")
            ax_rmse.invert_yaxis()

        # Stats text
        if frame % 20 == 0:
            lines = [f"t = {sim_t[0]:.1f}s / {duration:.0f}s\n"]
            rmses = {}
            for name in controllers:
                errs = data[name]["errors"]
                if errs:
                    rmses[name] = np.sqrt(np.mean(np.array(errs) ** 2))
            for name in sorted(rmses, key=rmses.get):
                lines.append(f"  {name:<16} RMSE={rmses[name]:.4f}")
            stats_text.set_text("\n".join(lines))

    anim = FuncAnimation(
        fig, update, frames=num_steps, interval=50, blit=False, repeat=False,
    )

    os.makedirs("plots", exist_ok=True)
    scenario_key = args.scenario

    gif_path = f"plots/all_37_variants_live_{scenario_key}.gif"
    print(f"\n  Saving GIF ({num_steps} frames) ...")
    anim.save(gif_path, writer="pillow", fps=20, dpi=80)
    print(f"  GIF saved: {gif_path}")

    try:
        mp4_path = f"plots/all_37_variants_live_{scenario_key}.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=80)
        print(f"  MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  MP4 skip (ffmpeg not available): {e}")

    plt.close()

    # Final ranking
    print(f"\n{'=' * 60}")
    print(f"  Final Ranking -- {scenario['name']}")
    print(f"{'=' * 60}")
    final_rmses = {}
    for name in controllers:
        errs = data[name]["errors"]
        if errs:
            final_rmses[name] = np.sqrt(np.mean(np.array(errs) ** 2))
    for rank, name in enumerate(sorted(final_rmses, key=final_rmses.get), 1):
        print(f"  {rank:>3}. {name:<16} RMSE={final_rmses[name]:.4f}")
    print(f"{'=' * 60}\n")


# ── 벤치마크 메인 ─────────────────────────────────────────────

def run_benchmark(args):
    """정적 벤치마크 실행"""
    scenarios = get_scenarios()
    scenario = scenarios[args.scenario]

    print(f"\n{'=' * 100}")
    print(f"  All 37 MPPI Variants Benchmark")
    print(f"  Scenario: {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"  K={COMMON['K']}, N={COMMON['N']}, dt={COMMON['dt']}, T={scenario['duration']}s")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 100}")

    model = DifferentialDriveKinematic(wheelbase=0.5)
    registry = _get_variant_registry()

    all_results = []
    n_total = len(registry)
    n_skip = 0

    for i, v in enumerate(registry):
        name = v["name"]
        print(f"\n  [{i+1}/{n_total}] {name:<20}", end=" ", flush=True)

        try:
            ctrl = _build_controller(v, model, scenario["obstacles"])
        except Exception as e:
            print(f"BUILD FAIL: {e}")
            n_skip += 1
            continue

        try:
            np.random.seed(args.seed)
            t_start = time.time()
            history = run_single(model, ctrl, scenario, seed=args.seed)
            elapsed = time.time() - t_start

            metrics = extract_metrics(history, scenario)

            all_results.append({
                "name": name,
                "group": v["group"],
                "idx": v["idx"],
                "states": history["states"],
                "controls": history["controls"],
                "infos": history["infos"],
                "elapsed": elapsed,
                **metrics,
            })

            print(f"RMSE={metrics['rmse']:.4f}  Solve={metrics['mean_solve_ms']:.1f}ms  "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            print(f"RUN FAIL: {e}")
            n_skip += 1
            continue

    # 결과 테이블
    print(f"\n{'=' * 120}")
    print(f"  Results: {len(all_results)}/{n_total} succeeded, {n_skip} skipped")
    print(f"  Scenario: {scenario['name']}")
    print(f"{'=' * 120}")
    print_results_table(all_results, scenario)
    print(f"{'=' * 120}")

    # Best-of summary
    if all_results:
        best_acc = min(all_results, key=lambda x: x["rmse"])
        best_spd = min(all_results, key=lambda x: x["mean_solve_ms"])
        print(f"\n  Best Accuracy:  {best_acc['name']} (RMSE={best_acc['rmse']:.4f}m)")
        print(f"  Best Speed:     {best_spd['name']} (Solve={best_spd['mean_solve_ms']:.2f}ms)")

        if scenario["obstacles"]:
            safe = [r for r in all_results if r["n_collisions"] == 0]
            if safe:
                best_safe = max(safe, key=lambda x: x["min_clearance"])
                print(f"  Best Safety:    {best_safe['name']} (Clear={best_safe['min_clearance']:.3f}m)")
            n_crashed = sum(1 for r in all_results if r["n_collisions"] > 0)
            print(f"  Collision-free: {len(all_results) - n_crashed}/{len(all_results)}")

    if not args.no_plot and all_results:
        plot_static(all_results, scenario, args.scenario)

    return all_results


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="All 37 MPPI Variants Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario", default="simple",
        choices=["simple", "obstacles", "dense_obstacles"],
        help="Benchmark scenario (default: simple)",
    )
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run all 3 scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--live", action="store_true", help="Live animation mode")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    scenarios = get_scenarios()

    if args.live:
        if args.all_scenarios:
            for scenario_name in scenarios:
                args.scenario = scenario_name
                run_live(args)
        else:
            run_live(args)
    elif args.all_scenarios:
        for scenario_name in scenarios:
            args.scenario = scenario_name
            run_benchmark(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
