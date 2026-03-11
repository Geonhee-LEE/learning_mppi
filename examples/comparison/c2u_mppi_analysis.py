#!/usr/bin/env python3
"""
C2U-MPPI 심층 분석: 다양한 시나리오 × 파라미터 × 시드

분석 항목:
  1. 장애물 근접 시나리오 (장애물이 경로 위에 배치)
  2. 프로세스 노이즈 스케일 sweep
  3. chance_alpha 민감도 분석
  4. figure8 궤적 추가
  5. 다중 시드 통계 (5회 반복)
  6. 계산 시간 분석

Usage:
    PYTHONPATH=. python examples/comparison/c2u_mppi_analysis.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    UncertaintyMPPIParams,
    C2UMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.uncertainty_mppi import (
    UncertaintyMPPIController,
)
from mppi_controller.controllers.mppi.c2u_mppi import C2UMPPIController
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
    figure_eight_trajectory,
)


# ── 불확실성 모델 ────────────────────────────────────────────

class ConstantUncertainty:
    def __init__(self, std_val=0.05):
        self.std_val = std_val
    def __call__(self, states, controls):
        if states.ndim == 1:
            states = states[None, :]
        return np.full((states.shape[0], states.shape[-1]), self.std_val)


class PositionDependentUncertainty:
    """원점에서 멀수록 불확실성 증가"""
    def __init__(self, base=0.02, scale=0.03):
        self.base = base
        self.scale = scale
    def __call__(self, states, controls):
        if states.ndim == 1:
            states = states[None, :]
        dist = np.sqrt(states[:, 0]**2 + states[:, 1]**2)
        nx = states.shape[-1]
        std = np.full((states.shape[0], nx), self.base)
        std[:, 0] += self.scale * dist
        std[:, 1] += self.scale * dist
        return std


# ── 시뮬레이션 유틸 ──────────────────────────────────────────

def run_simulation(model, controller, ref_fn, initial_state, dt, duration,
                   process_noise_std=None, real_model=None):
    """시뮬레이션 실행 + 상세 메트릭 수집"""
    if real_model is None:
        real_model = model

    n_steps = int(duration / dt)
    state = initial_state.copy()
    t = 0.0

    states = [state.copy()]
    controls_list = []
    solve_times = []
    infos = []

    for _ in range(n_steps):
        ref = ref_fn(t)
        t0 = time.time()
        control, info = controller.compute_control(state, ref)
        solve_times.append(time.time() - t0)

        next_state = real_model.step(state, control, dt)
        if process_noise_std is not None:
            next_state = next_state + np.random.normal(0, process_noise_std)
        next_state = real_model.normalize_state(next_state)

        states.append(next_state.copy())
        controls_list.append(control.copy())
        infos.append(info)

        state = next_state
        t += dt

    return {
        "states": np.array(states),
        "controls": np.array(controls_list),
        "solve_times": np.array(solve_times),
        "infos": infos,
    }


def compute_metrics(states, obstacles, traj_fn, dt):
    """종합 메트릭 계산"""
    n_collisions = 0
    min_clearance = float("inf")
    clearances = []
    tracking_errors = []

    for i, st in enumerate(states):
        x, y = st[0], st[1]

        # 장애물 메트릭
        for ox, oy, r in obstacles:
            d = np.sqrt((x - ox)**2 + (y - oy)**2) - r
            min_clearance = min(min_clearance, d)
            if d < 0:
                n_collisions += 1
        closest = min(np.sqrt((x - ox)**2 + (y - oy)**2) - r for ox, oy, r in obstacles)
        clearances.append(closest)

        # 추적 오차
        ref = traj_fn(i * dt)
        err = np.sqrt((x - ref[0])**2 + (y - ref[1])**2)
        tracking_errors.append(err)

    return {
        "n_collisions": n_collisions,
        "min_clearance": min_clearance,
        "mean_clearance": np.mean(clearances),
        "rmse": np.sqrt(np.mean(np.array(tracking_errors)**2)),
        "max_error": np.max(tracking_errors),
    }


# ── 모델 불일치 래퍼 ─────────────────────────────────────────

class MismatchModel:
    """위치 의존 체계적 바이어스"""
    def __init__(self, nominal, drift=0.15, friction=0.08):
        self._nominal = nominal
        self.drift = drift
        self.friction = friction
        self.state_dim = nominal.state_dim
        self.control_dim = nominal.control_dim

    def step(self, state, control, dt):
        ns = self._nominal.step(state, control, dt).copy()
        x, y = state[0], state[1]
        dist = np.sqrt(x**2 + y**2)
        ns[0] += self.drift * max(0, y) * 0.15 * dt
        ns[1] += self.drift * np.sin(x * 0.3) * 0.1 * dt
        friction = self.friction * dist * 0.05 * dt
        ns[0] -= friction * np.cos(state[2])
        ns[1] -= friction * np.sin(state[2])
        ns[2] += self.drift * 0.02 * np.sin(2 * state[2]) * dt
        return ns

    def get_control_bounds(self):
        return self._nominal.get_control_bounds()

    def normalize_state(self, state):
        return self._nominal.normalize_state(state)


# ── 컨트롤러 팩토리 ──────────────────────────────────────────

def make_common_params():
    return dict(
        K=256, N=20, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )


def make_vanilla(model, obstacles, common):
    params = MPPIParams(**common)
    cost = CompositeMPPICost([
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
        ObstacleCost(obstacles, safety_margin=0.1, cost_weight=200.0),
    ])
    return MPPIController(model, params, cost_function=cost)


def make_uncertainty(model, obstacles, common, unc_model):
    params = UncertaintyMPPIParams(
        **common, exploration_factor=1.5,
        uncertainty_strategy="previous_trajectory",
    )
    cost = CompositeMPPICost([
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
        ObstacleCost(obstacles, safety_margin=0.2, cost_weight=200.0),
    ])
    return UncertaintyMPPIController(model, params, cost_function=cost,
                                     uncertainty_fn=unc_model)


def make_c2u(model, obstacles, common, process_noise_scale=0.01,
             chance_alpha=0.05, chance_cost_weight=500.0, margin_factor=1.0):
    params = C2UMPPIParams(
        **common,
        cc_obstacles=obstacles,
        chance_alpha=chance_alpha,
        chance_cost_weight=chance_cost_weight,
        process_noise_scale=process_noise_scale,
        cc_margin_factor=margin_factor,
    )
    cost = CompositeMPPICost([
        StateTrackingCost(params.Q),
        TerminalCost(params.Qf),
        ControlEffortCost(params.R),
    ])
    return C2UMPPIController(model, params, cost_function=cost)


# ══════════════════════════════════════════════════════════════
# 분석 1: 장애물 난이도별 비교 (근접/중간/원거리)
# ══════════════════════════════════════════════════════════════

def analysis_1_obstacle_proximity():
    print("\n" + "=" * 74)
    print("  분석 1: 장애물 근접도별 비교 (circle 궤적, radius=5)")
    print("  장애물이 궤적에 가까울수록 C2U의 보수성이 효과 발휘")
    print("=" * 74)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    common = make_common_params()

    # circle 궤적은 반지름 5.0, 중심 (0,0)
    obstacle_sets = {
        "원거리 (d≈1.5m)": [(5.0, 3.5, 0.4), (-3.5, 5.0, 0.4), (3.0, -5.0, 0.3)],
        "중거리 (d≈0.8m)": [(5.0, 1.5, 0.4), (-1.5, 5.2, 0.4), (4.5, -2.5, 0.3)],
        "근접   (d≈0.3m)": [(5.0, 0.8, 0.4), (-0.8, 5.2, 0.4), (4.8, -1.5, 0.3)],
    }

    unc = ConstantUncertainty(0.03)
    noise_std = np.array([0.03, 0.03, 0.01])

    for obs_name, obstacles in obstacle_sets.items():
        print(f"\n  ── {obs_name} ──")
        print(f"  {'Method':<20} {'Collisions':>10} {'MinClr':>8} {'MeanClr':>8} {'RMSE':>8} {'ms':>6}")
        print(f"  {'─' * 62}")

        for method_name, make_fn in [
            ("Vanilla", lambda: make_vanilla(model, obstacles, common)),
            ("UncMPPI", lambda: make_uncertainty(model, obstacles, common, unc)),
            ("C2U (α=0.05)", lambda: make_c2u(model, obstacles, common, 0.02, 0.05, 500.0)),
            ("C2U (α=0.01)", lambda: make_c2u(model, obstacles, common, 0.02, 0.01, 500.0)),
        ]:
            # 3시드 평균
            metrics_list = []
            solve_times_all = []
            for seed in [42, 123, 777]:
                np.random.seed(seed)
                ctrl = make_fn()
                initial = circle_trajectory(0.0)
                ref_fn = lambda t: generate_reference_trajectory(circle_trajectory, t, 20, 0.05)
                hist = run_simulation(model, ctrl, ref_fn, initial, 0.05, 8.0,
                                      process_noise_std=noise_std)
                m = compute_metrics(hist["states"], obstacles, circle_trajectory, 0.05)
                metrics_list.append(m)
                solve_times_all.extend(hist["solve_times"])

            avg = {
                "n_collisions": sum(m["n_collisions"] for m in metrics_list),
                "min_clearance": min(m["min_clearance"] for m in metrics_list),
                "mean_clearance": np.mean([m["mean_clearance"] for m in metrics_list]),
                "rmse": np.mean([m["rmse"] for m in metrics_list]),
            }
            avg_ms = np.mean(solve_times_all) * 1000

            print(f"  {method_name:<20} {avg['n_collisions']:>10d} "
                  f"{avg['min_clearance']:>8.3f} {avg['mean_clearance']:>8.3f} "
                  f"{avg['rmse']:>8.3f} {avg_ms:>6.1f}")


# ══════════════════════════════════════════════════════════════
# 분석 2: 프로세스 노이즈 스케일 sweep
# ══════════════════════════════════════════════════════════════

def analysis_2_noise_sweep():
    print("\n" + "=" * 74)
    print("  분석 2: 프로세스 노이즈 강도별 비교")
    print("  노이즈 증가 → C2U의 공분산 전파가 r_eff 확대 → 안전성 향상")
    print("=" * 74)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    common = make_common_params()
    obstacles = [(5.0, 1.0, 0.4), (-1.0, 5.2, 0.4), (4.5, -2.0, 0.3)]
    unc = ConstantUncertainty(0.03)

    noise_levels = [
        ("없음",   None),
        ("약 ",    np.array([0.01, 0.01, 0.005])),
        ("중 ",    np.array([0.04, 0.04, 0.01])),
        ("강 ",    np.array([0.08, 0.08, 0.02])),
        ("극강",   np.array([0.15, 0.15, 0.04])),
    ]

    print(f"\n  {'Noise':<6} {'Method':<16} {'Collisions':>10} {'MinClr':>8} {'RMSE':>8}")
    print(f"  {'─' * 52}")

    for noise_name, noise_std in noise_levels:
        for method_name, make_fn in [
            ("Vanilla", lambda: make_vanilla(model, obstacles, common)),
            ("UncMPPI", lambda: make_uncertainty(model, obstacles, common, unc)),
            ("C2U-MPPI", lambda: make_c2u(model, obstacles, common, 0.02, 0.05, 500.0)),
        ]:
            collisions_total = 0
            min_clr_total = float("inf")
            rmses = []
            for seed in [42, 123, 777]:
                np.random.seed(seed)
                ctrl = make_fn()
                initial = circle_trajectory(0.0)
                ref_fn = lambda t: generate_reference_trajectory(circle_trajectory, t, 20, 0.05)
                hist = run_simulation(model, ctrl, ref_fn, initial, 0.05, 8.0,
                                      process_noise_std=noise_std)
                m = compute_metrics(hist["states"], obstacles, circle_trajectory, 0.05)
                collisions_total += m["n_collisions"]
                min_clr_total = min(min_clr_total, m["min_clearance"])
                rmses.append(m["rmse"])

            prefix = f"  {noise_name:<6}" if method_name == "Vanilla" else f"  {'':6}"
            print(f"{prefix} {method_name:<16} {collisions_total:>10d} "
                  f"{min_clr_total:>8.3f} {np.mean(rmses):>8.3f}")

        print()


# ══════════════════════════════════════════════════════════════
# 분석 3: 모델 불일치 시나리오 (핵심 테스트)
# ══════════════════════════════════════════════════════════════

def analysis_3_model_mismatch():
    print("\n" + "=" * 74)
    print("  분석 3: 모델 불일치 (MPPI는 순수 기구학, 실제는 바이어스 모델)")
    print("  C2U의 공분산 전파가 체계적 오차를 간접 반영")
    print("=" * 74)

    nominal = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    common = make_common_params()
    obstacles = [(5.0, 1.0, 0.4), (-1.0, 5.2, 0.4), (4.5, -2.0, 0.3)]

    mismatch_levels = [
        ("없음",   None,                              None),
        ("약  ",   {"drift": 0.08, "friction": 0.04}, np.array([0.01, 0.01, 0.005])),
        ("중  ",   {"drift": 0.15, "friction": 0.08}, np.array([0.02, 0.02, 0.008])),
        ("강  ",   {"drift": 0.25, "friction": 0.12}, np.array([0.03, 0.03, 0.01])),
    ]

    unc_model = PositionDependentUncertainty(base=0.02, scale=0.03)

    print(f"\n  {'Mismatch':<8} {'Method':<16} {'Collisions':>10} {'MinClr':>8} {'RMSE':>8}")
    print(f"  {'─' * 54}")

    for mm_name, mm_kwargs, noise_std in mismatch_levels:
        if mm_kwargs is not None:
            real_model = MismatchModel(nominal, **mm_kwargs)
        else:
            real_model = nominal

        for method_name, make_fn in [
            ("Vanilla", lambda: make_vanilla(nominal, obstacles, common)),
            ("UncMPPI", lambda: make_uncertainty(nominal, obstacles, common, unc_model)),
            ("C2U-MPPI", lambda: make_c2u(nominal, obstacles, common, 0.02, 0.05, 500.0)),
        ]:
            collisions_total = 0
            min_clr_total = float("inf")
            rmses = []
            for seed in [42, 123, 777]:
                np.random.seed(seed)
                ctrl = make_fn()
                initial = circle_trajectory(0.0)
                ref_fn = lambda t: generate_reference_trajectory(circle_trajectory, t, 20, 0.05)
                hist = run_simulation(nominal, ctrl, ref_fn, initial, 0.05, 8.0,
                                      process_noise_std=noise_std, real_model=real_model)
                m = compute_metrics(hist["states"], obstacles, circle_trajectory, 0.05)
                collisions_total += m["n_collisions"]
                min_clr_total = min(min_clr_total, m["min_clearance"])
                rmses.append(m["rmse"])

            prefix = f"  {mm_name:<8}" if method_name == "Vanilla" else f"  {'':8}"
            print(f"{prefix} {method_name:<16} {collisions_total:>10d} "
                  f"{min_clr_total:>8.3f} {np.mean(rmses):>8.3f}")

        print()


# ══════════════════════════════════════════════════════════════
# 분석 4: Figure-8 궤적 (좁은 공간 연속 회전)
# ══════════════════════════════════════════════════════════════

def analysis_4_figure8():
    print("\n" + "=" * 74)
    print("  분석 4: Figure-8 궤적 (좁은 공간 + 연속 방향 전환)")
    print("=" * 74)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    common = make_common_params()

    # figure8 궤적 범위에 맞는 장애물
    obstacles = [(4.0, 1.5, 0.3), (-4.0, -1.5, 0.3), (0.0, 2.0, 0.3)]
    unc = ConstantUncertainty(0.04)
    noise_std = np.array([0.03, 0.03, 0.01])

    print(f"\n  {'Method':<20} {'Collisions':>10} {'MinClr':>8} {'MeanClr':>8} {'RMSE':>8} {'ms':>6}")
    print(f"  {'─' * 62}")

    for method_name, make_fn in [
        ("Vanilla", lambda: make_vanilla(model, obstacles, common)),
        ("UncMPPI", lambda: make_uncertainty(model, obstacles, common, unc)),
        ("C2U (α=0.05)", lambda: make_c2u(model, obstacles, common, 0.02, 0.05, 500.0)),
        ("C2U (α=0.10)", lambda: make_c2u(model, obstacles, common, 0.02, 0.10, 300.0)),
    ]:
        metrics_list = []
        solve_times_all = []
        for seed in [42, 123, 777]:
            np.random.seed(seed)
            ctrl = make_fn()
            initial = figure_eight_trajectory(0.0)
            ref_fn = lambda t: generate_reference_trajectory(figure_eight_trajectory, t, 20, 0.05)
            hist = run_simulation(model, ctrl, ref_fn, initial, 0.05, 10.0,
                                  process_noise_std=noise_std)
            m = compute_metrics(hist["states"], obstacles, figure_eight_trajectory, 0.05)
            metrics_list.append(m)
            solve_times_all.extend(hist["solve_times"])

        avg = {
            "n_collisions": sum(m["n_collisions"] for m in metrics_list),
            "min_clearance": min(m["min_clearance"] for m in metrics_list),
            "mean_clearance": np.mean([m["mean_clearance"] for m in metrics_list]),
            "rmse": np.mean([m["rmse"] for m in metrics_list]),
        }
        avg_ms = np.mean(solve_times_all) * 1000

        print(f"  {method_name:<20} {avg['n_collisions']:>10d} "
              f"{avg['min_clearance']:>8.3f} {avg['mean_clearance']:>8.3f} "
              f"{avg['rmse']:>8.3f} {avg_ms:>6.1f}")


# ══════════════════════════════════════════════════════════════
# 분석 5: C2U 파라미터 민감도 (chance_alpha × process_noise_scale)
# ══════════════════════════════════════════════════════════════

def analysis_5_c2u_parameter_sensitivity():
    print("\n" + "=" * 74)
    print("  분석 5: C2U 파라미터 민감도 (chance_α × process_noise_scale)")
    print("  안전성-성능 트레이드오프 파레토 분석")
    print("=" * 74)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    common = make_common_params()
    obstacles = [(5.0, 1.0, 0.4), (-1.0, 5.2, 0.4), (4.5, -2.0, 0.3)]
    noise_std = np.array([0.04, 0.04, 0.01])

    alphas = [0.01, 0.05, 0.10, 0.20]
    pn_scales = [0.001, 0.005, 0.01, 0.02, 0.05]

    print(f"\n  {'α':>6} {'pn_scale':>10} {'Collisions':>10} {'MinClr':>8} {'RMSE':>8} {'r_eff_mean':>10}")
    print(f"  {'─' * 56}")

    for alpha in alphas:
        for pns in pn_scales:
            collisions_total = 0
            min_clr_total = float("inf")
            rmses = []
            r_effs = []

            for seed in [42, 123]:
                np.random.seed(seed)
                ctrl = make_c2u(model, obstacles, common, pns, alpha, 500.0)
                initial = circle_trajectory(0.0)
                ref_fn = lambda t: generate_reference_trajectory(circle_trajectory, t, 20, 0.05)
                hist = run_simulation(model, ctrl, ref_fn, initial, 0.05, 8.0,
                                      process_noise_std=noise_std)
                m = compute_metrics(hist["states"], obstacles, circle_trajectory, 0.05)
                collisions_total += m["n_collisions"]
                min_clr_total = min(min_clr_total, m["min_clearance"])
                rmses.append(m["rmse"])

                last_info = hist["infos"][-1]
                r_eff = last_info.get("effective_radii")
                if r_eff is not None:
                    r_effs.append(np.mean(r_eff))

            mean_r_eff = np.mean(r_effs) if r_effs else 0
            print(f"  {alpha:>6.2f} {pns:>10.3f} {collisions_total:>10d} "
                  f"{min_clr_total:>8.3f} {np.mean(rmses):>8.3f} {mean_r_eff:>10.4f}")

        print()


# ══════════════════════════════════════════════════════════════
# 분석 6: 공분산 전파 시각화 (텍스트)
# ══════════════════════════════════════════════════════════════

def analysis_6_covariance_evolution():
    print("\n" + "=" * 74)
    print("  분석 6: 공분산 전파 시각화 (호라이즌 내 Σ trace 변화)")
    print("=" * 74)

    model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
    common = make_common_params()
    obstacles = [(5.0, 1.0, 0.4)]

    for pns_name, pns in [("Low (0.001)", 0.001), ("Mid (0.01)", 0.01), ("High (0.05)", 0.05)]:
        np.random.seed(42)
        ctrl = make_c2u(model, obstacles, common, pns, 0.05, 500.0)
        initial = circle_trajectory(0.0)
        ref_fn = lambda t: generate_reference_trajectory(circle_trajectory, t, 20, 0.05)
        _, info = ctrl.compute_control(initial, ref_fn(0.0))

        cov_traj = info["covariance_trajectory"]
        traces = [np.trace(c[:2, :2]) for c in cov_traj]

        print(f"\n  Process noise: {pns_name}")
        print(f"  t:     ", "  ".join(f"{t:>5d}" for t in range(0, len(traces), 4)))
        print(f"  trace: ", "  ".join(f"{traces[t]:>5.3f}" for t in range(0, len(traces), 4)))

        # ASCII 바 차트
        max_tr = max(traces)
        bar_width = 40
        print(f"  ┌{'─' * bar_width}┐")
        for t in range(0, len(traces), 2):
            bar_len = int(traces[t] / max(max_tr, 1e-10) * bar_width)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            print(f"  │{bar}│ t={t:>2} trace={traces[t]:.4f}")
        print(f"  └{'─' * bar_width}┘")

        # 유효 반경 변화
        r_eff = info["effective_radii"]
        if r_eff is not None:
            print(f"  r_eff: min={np.min(r_eff):.4f}, max={np.max(r_eff):.4f}")


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  C2U-MPPI 심층 성능 분석 (Chance-Constrained Unscented MPPI)       ║")
    print("║  3-Way: Vanilla MPPI vs Uncertainty MPPI vs C2U-MPPI               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    t_start = time.time()

    analysis_1_obstacle_proximity()
    analysis_2_noise_sweep()
    analysis_3_model_mismatch()
    analysis_4_figure8()
    analysis_5_c2u_parameter_sensitivity()
    analysis_6_covariance_evolution()

    elapsed = time.time() - t_start

    print("\n" + "=" * 74)
    print(f"  총 분석 시간: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
