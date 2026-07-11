#!/usr/bin/env python3
"""
MPPI 변형 x 로봇 모델 교차 벤치마크 (TODO #207 모델 타입별 벤치마크)

전체 41종 MPPI 변형을 6종 로봇 모델(기구학/동역학)에서 교차 평가합니다.
변형 레지스트리는 all_37_variants_benchmark._get_variant_registry()를 재사용합니다.

Models:
  diffdrive_kin  : DifferentialDriveKinematic  (3D [x,y,th],           2D [v,w])
  diffdrive_dyn  : DifferentialDriveDynamic    (5D [x,y,th,v,w],       2D [a,alpha])
  ackermann_kin  : AckermannKinematic          (4D [x,y,th,delta],     2D [v,phi])
  ackermann_dyn  : AckermannDynamic            (5D [x,y,th,v,delta],   2D [a,phi])
  swerve_kin     : SwerveDriveKinematic        (3D [x,y,th],           3D [vx,vy,w])
  swerve_dyn     : SwerveDriveDynamic          (6D [x,y,th,vx,vy,w],   3D [ax,ay,alpha])

Scenarios:
  simple    : 원형 궤적 추적 (r=2.0, w=0.5), 장애물 없음
  obstacles : 경로 위 원형 장애물 3개 (45/165/285도, r=0.35)

Usage:
    PYTHONPATH=. python examples/comparison/variants_x_models_benchmark.py --smoke
    PYTHONPATH=. python examples/comparison/variants_x_models_benchmark.py
    PYTHONPATH=. python examples/comparison/variants_x_models_benchmark.py --scenario all
    PYTHONPATH=. python examples/comparison/variants_x_models_benchmark.py \
        --models diffdrive_kin,ackermann_kin --variants Vanilla,DIAL,Tube --duration 6
    PYTHONPATH=. python examples/comparison/variants_x_models_benchmark.py \
        --models diffdrive_kin --scenario simple --duration 4

Outputs:
    results/variants_x_models/{model}_{scenario}.json
    plots/variants_x_models_heatmap_{scenario}.png
    plots/variants_x_models_summary_{scenario}.png
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 변형 레지스트리 / 비용 팩토리 재사용 (all_37_variants_benchmark)
from all_37_variants_benchmark import _get_variant_registry, _make_cost, GROUP_COLORS

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.kinematic.ackermann_kinematic import AckermannKinematic
from mppi_controller.models.kinematic.swerve_drive_kinematic import SwerveDriveKinematic
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.models.dynamic.ackermann_dynamic import AckermannDynamic
from mppi_controller.models.dynamic.swerve_drive_dynamic import SwerveDriveDynamic
from mppi_controller.utils.trajectory import (
    circle_trajectory,
    generate_reference_trajectory,
)


# ── 공통 설정 ──────────────────────────────────────────────────

RADIUS = 2.0            # 원형 궤적 반지름
ANGULAR_VELOCITY = 0.5  # 원형 궤적 각속도
WHEELBASE = 0.5         # Ackermann wheelbase

COMMON_BASE = dict(K=512, N=20, dt=0.05, lambda_=1.0)


# ── 레퍼런스 궤적 빌더 ──────────────────────────────────────────

def _pose_ref(t):
    """(3,) [x, y, theta] 원형 궤적"""
    return circle_trajectory(t, radius=RADIUS, angular_velocity=ANGULAR_VELOCITY)


def _ref_pose3(t):
    return _pose_ref(t)


def _ref_diffdrive_dyn5(t):
    """5D [x, y, th, v, w]: v_ref = r*w, w_ref = w"""
    pose = _pose_ref(t)
    return np.concatenate([pose, [RADIUS * ANGULAR_VELOCITY, ANGULAR_VELOCITY]])


def _ref_ackermann_kin4(t):
    """4D [x, y, th, delta]: delta_ref = atan(L / r)"""
    pose = _pose_ref(t)
    return np.concatenate([pose, [np.arctan(WHEELBASE / RADIUS)]])


def _ref_ackermann_dyn5(t):
    """5D [x, y, th, v, delta]"""
    pose = _pose_ref(t)
    return np.concatenate(
        [pose, [RADIUS * ANGULAR_VELOCITY, np.arctan(WHEELBASE / RADIUS)]]
    )


def _ref_swerve_dyn6(t):
    """6D [x, y, th, vx, vy, w]: body-frame vx = r*w (전진), vy = 0"""
    pose = _pose_ref(t)
    return np.concatenate(
        [pose, [RADIUS * ANGULAR_VELOCITY, 0.0, ANGULAR_VELOCITY]]
    )


# ── 모델 레지스트리 ─────────────────────────────────────────────

def get_model_registry():
    """6종 로봇 모델 레지스트리"""
    return {
        "diffdrive_kin": dict(
            label="DiffDrive Kin (3D)",
            factory=lambda: DifferentialDriveKinematic(
                v_max=2.0, omega_max=2.0, wheelbase=WHEELBASE
            ),
            Q=np.array([10.0, 10.0, 1.0]),
            sigma=np.array([0.5, 0.5]),
            trajectory_fn=_ref_pose3,
        ),
        "diffdrive_dyn": dict(
            label="DiffDrive Dyn (5D)",
            factory=lambda: DifferentialDriveDynamic(v_max=2.0, omega_max=2.0),
            Q=np.array([10.0, 10.0, 1.0, 0.5, 0.5]),
            sigma=np.array([1.0, 1.0]),
            trajectory_fn=_ref_diffdrive_dyn5,
        ),
        "ackermann_kin": dict(
            label="Ackermann Kin (4D)",
            factory=lambda: AckermannKinematic(
                wheelbase=WHEELBASE, v_max=2.0, max_steer=0.6, steer_rate_max=2.0
            ),
            Q=np.array([10.0, 10.0, 1.0, 0.1]),
            sigma=np.array([0.5, 0.5]),
            trajectory_fn=_ref_ackermann_kin4,
        ),
        "ackermann_dyn": dict(
            label="Ackermann Dyn (5D)",
            factory=lambda: AckermannDynamic(
                wheelbase=WHEELBASE, v_max=2.0, a_max=2.0,
                max_steer=0.6, steer_rate_max=2.0,
            ),
            Q=np.array([10.0, 10.0, 1.0, 0.5, 0.1]),
            sigma=np.array([1.0, 0.5]),
            trajectory_fn=_ref_ackermann_dyn5,
        ),
        "swerve_kin": dict(
            label="Swerve Kin (3D)",
            factory=lambda: SwerveDriveKinematic(
                vx_max=2.0, vy_max=2.0, omega_max=2.0
            ),
            Q=np.array([10.0, 10.0, 1.0]),
            sigma=np.array([0.5, 0.5, 0.5]),
            trajectory_fn=_ref_pose3,
        ),
        "swerve_dyn": dict(
            label="Swerve Dyn (6D)",
            factory=lambda: SwerveDriveDynamic(
                vx_max=2.0, vy_max=2.0, omega_max=2.0,
                ax_max=2.0, ay_max=2.0, alpha_max=2.0,
            ),
            Q=np.array([10.0, 10.0, 1.0, 0.5, 0.5, 0.5]),
            sigma=np.array([1.0, 1.0, 1.0]),
            trajectory_fn=_ref_swerve_dyn6,
        ),
    }


# ── 시나리오 ───────────────────────────────────────────────────

def get_scenarios(duration):
    """simple / obstacles 시나리오 (모든 모델 공통 장애물)"""
    # 경로(r=2.0) 위 장애물: 45 / 165 / 285도, 반지름 0.35
    obstacle_angles = np.deg2rad([45.0, 165.0, 285.0])
    obstacles = [
        (RADIUS * np.cos(a), RADIUS * np.sin(a), 0.35) for a in obstacle_angles
    ]
    return {
        "simple": dict(name="Simple (No Obstacles)", obstacles=[], duration=duration),
        "obstacles": dict(name="Obstacles (3 on path)", obstacles=obstacles,
                          duration=duration),
    }


# ── 모델별 변형 파라미터 적응 ────────────────────────────────────

def adapt_variant_for_model(variant, model):
    """차원 의존적 extra_params를 모델 차원에 맞게 오버라이드"""
    variant = dict(variant)
    extra = copy.deepcopy(variant["extra_params"])
    nx, nu = model.state_dim, model.control_dim

    if "K_fb" in extra:
        # Tube: 피드백 게인 (nu, nx)
        extra["K_fb"] = 2.0 * np.eye(nu, nx)

    if "disturbance_std" in extra:
        # Robust: 상태 차원 외란 표준편차 (pose 0.05, 나머지 0.02)
        std = np.full(nx, 0.02)
        std[:min(3, nx)] = 0.05
        extra["disturbance_std"] = std.tolist()

    if variant["name"] == "PR":
        # PR: wheelbase 파라미터 (Ackermann은 실제 사용, 나머지는 setattr 통과)
        extra["param_nominal"] = getattr(model, "wheelbase", None) or WHEELBASE
        extra["param_std"] = 0.1
        extra["param_min"] = extra["param_nominal"] * 0.6
        extra["param_max"] = extra["param_nominal"] * 1.4

    variant["extra_params"] = extra
    return variant


# ── 컨트롤러 생성 (all_37 _build_controller의 모델별 params 버전) ──

def _needs_custom_ancillary(model):
    """기본 AncillaryController가 diffdrive 차원(kin 3x2 / dyn 5x2)을 가정"""
    nx, nu = model.state_dim, model.control_dim
    expected = (3, 2) if model.model_type == "kinematic" else (5, 2)
    return (nx, nu) != expected


def _make_ancillary(model):
    """일반 차원 (nu, nx) 대각 게인 AncillaryController"""
    from mppi_controller.controllers.mppi.ancillary_controller import (
        AncillaryController,
    )
    nx, nu = model.state_dim, model.control_dim
    return AncillaryController(
        K_fb=np.eye(nu, nx),
        max_correction=0.5 * np.ones(nu),
    )


def build_controller(variant, model, model_spec, obstacles):
    """5가지 생성자 패턴에 따라 컨트롤러 생성 (모델별 Q/R/sigma 사용)"""
    variant = adapt_variant_for_model(variant, model)
    nu = model.control_dim

    common = dict(
        COMMON_BASE,
        sigma=model_spec["sigma"].copy(),
        Q=model_spec["Q"].copy(),
        R=0.1 * np.ones(nu),
    )
    params_kw = {**common, **variant["extra_params"]}

    # 장애물을 params에 주입하는 패턴 (빈 리스트면 주입 생략)
    obstacle_field = variant.get("obstacle_field")
    if obstacle_field and obstacles:
        params_kw[obstacle_field] = obstacles

    params = variant["params_cls"](**params_kw)
    skip_obstacle = variant.get("skip_obstacle_cost", False)
    cost = _make_cost(params, obstacles, skip_obstacle=skip_obstacle)

    ctor_type = variant["ctor"]
    if ctor_type in ("standard", "obstacle_in_params"):
        kwargs = dict(cost_function=cost)
        if variant["name"] == "Robust" and _needs_custom_ancillary(model):
            # RobustMPPI 기본 피드백은 diffdrive 차원 가정 → 명시적 주입
            kwargs["ancillary_controller"] = _make_ancillary(model)
        return variant["controller_cls"](model, params, **kwargs)
    elif ctor_type in ("smooth", "tube", "no_cost"):
        if ctor_type == "tube" and _needs_custom_ancillary(model):
            return variant["controller_cls"](
                model, params, ancillary_controller=_make_ancillary(model)
            )
        return variant["controller_cls"](model, params)
    elif ctor_type == "contingency":
        safety_cost = _make_cost(params, obstacles)
        return variant["controller_cls"](
            model, params, cost_function=cost, safety_cost_function=safety_cost
        )
    raise ValueError(f"Unknown ctor type: {ctor_type}")


# ── 시뮬레이션 + 메트릭 ─────────────────────────────────────────

def run_cell(variant, model_key, model_spec, scenario, seed=42):
    """(variant, model, scenario) 단일 셀 실행 → 메트릭 dict"""
    result = dict(status="ok", fail_reason=None)

    model = model_spec["factory"]()
    trajectory_fn = model_spec["trajectory_fn"]
    obstacles = scenario["obstacles"]

    # 1) 컨트롤러 생성 (constructor error 캡처)
    try:
        np.random.seed(seed)
        controller = build_controller(variant, model, model_spec, obstacles)
    except Exception as e:
        result["status"] = "failed"
        result["fail_reason"] = f"constructor: {type(e).__name__}: {e}"
        return result

    # 2) 시뮬레이션 (runtime error 캡처)
    dt = COMMON_BASE["dt"]
    N = COMMON_BASE["N"]
    num_steps = int(scenario["duration"] / dt)

    state = trajectory_fn(0.0).copy()  # 초기 상태 = t=0 레퍼런스
    states = [state.copy()]
    controls_hist = []
    solve_times = []
    ess_list = []

    try:
        for step in range(num_steps):
            t = step * dt
            ref = generate_reference_trajectory(trajectory_fn, t, N, dt)

            t0 = time.time()
            control, info = controller.compute_control(state, ref)
            solve_times.append(time.time() - t0)

            control = np.asarray(control, dtype=float)
            if not np.all(np.isfinite(control)):
                raise ValueError("non-finite control output")

            state = state + model.forward_dynamics(state, control) * dt
            if not np.all(np.isfinite(state)):
                raise ValueError("non-finite state (diverged)")

            states.append(state.copy())
            controls_hist.append(control.copy())
            if isinstance(info, dict) and "ess" in info:
                ess_list.append(float(info["ess"]))
    except Exception as e:
        result["status"] = "failed"
        result["fail_reason"] = f"runtime(step {len(controls_hist)}): " \
                                f"{type(e).__name__}: {e}"
        return result

    # 3) 메트릭
    states = np.array(states)
    controls_hist = np.array(controls_hist)

    pos_err, head_err = [], []
    for i, st in enumerate(states):
        ref = trajectory_fn(i * dt)
        pos_err.append(np.hypot(st[0] - ref[0], st[1] - ref[1]))
        dth = st[2] - ref[2]
        head_err.append(np.arctan2(np.sin(dth), np.cos(dth)))
    pos_err = np.array(pos_err)
    head_err = np.array(head_err)

    n_collisions = 0
    min_clearance = float("inf")
    for st in states:
        for ox, oy, r in obstacles:
            clearance = np.hypot(st[0] - ox, st[1] - oy) - r
            min_clearance = min(min_clearance, clearance)
            if clearance < 0:
                n_collisions += 1

    ctrl_rate = 0.0
    if len(controls_hist) > 1:
        ctrl_rate = float(np.mean(np.abs(np.diff(controls_hist, axis=0))))

    result.update(
        position_rmse=float(np.sqrt(np.mean(pos_err ** 2))),
        heading_rmse=float(np.sqrt(np.mean(head_err ** 2))),
        max_position_error=float(np.max(pos_err)),
        control_rate=ctrl_rate,
        mean_solve_time_ms=float(np.mean(solve_times)) * 1000.0,
        ess=float(np.mean(ess_list)) if ess_list else 0.0,
        min_clearance=float(min_clearance) if obstacles else None,
        collisions=n_collisions,
    )
    return result


# ── 테이블 출력 ─────────────────────────────────────────────────

def print_model_table(model_key, model_label, scenario_key, results):
    """모델별 변형 결과 ASCII 테이블"""
    print(f"\n{'=' * 96}")
    print(f"  Model: {model_label} ({model_key}) | Scenario: {scenario_key}")
    print(f"{'=' * 96}")
    header = (f"  {'Variant':<14} {'Grp':<4} {'RMSE':>7} {'HeadRMSE':>9} "
              f"{'MaxErr':>7} {'CtrlRate':>9} {'Solve(ms)':>10} {'ESS':>7} "
              f"{'MinClr':>7} {'Coll':>5}  Status")
    print(header)
    print(f"  {'-' * 92}")
    for name, r in results.items():
        if r["status"] != "ok":
            reason = (r["fail_reason"] or "")[:48]
            print(f"  {name:<14} {r.get('group', '-'):<4} "
                  f"{'--':>7} {'--':>9} {'--':>7} {'--':>9} {'--':>10} "
                  f"{'--':>7} {'--':>7} {'--':>5}  FAILED ({reason})")
            continue
        clr = f"{r['min_clearance']:.3f}" if r["min_clearance"] is not None else "--"
        print(f"  {name:<14} {r.get('group', '-'):<4} "
              f"{r['position_rmse']:>7.4f} {r['heading_rmse']:>9.4f} "
              f"{r['max_position_error']:>7.3f} {r['control_rate']:>9.4f} "
              f"{r['mean_solve_time_ms']:>10.2f} {r['ess']:>7.1f} "
              f"{clr:>7} {r['collisions']:>5}  ok")


# ── 플롯 ───────────────────────────────────────────────────────

def plot_heatmap(all_results, model_keys, variant_names, scenario_key, plots_dir):
    """변형(행) x 모델(열) position_rmse 히트맵"""
    n_v, n_m = len(variant_names), len(model_keys)
    data = np.full((n_v, n_m), np.nan)
    for j, mk in enumerate(model_keys):
        for i, vn in enumerate(variant_names):
            r = all_results.get(mk, {}).get(vn)
            if r and r["status"] == "ok":
                data[i, j] = r["position_rmse"]

    fig, ax = plt.subplots(figsize=(3 + 1.6 * n_m, 2 + 0.32 * n_v))
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="#BDBDBD")
    vmax = np.nanpercentile(data, 90) if np.any(np.isfinite(data)) else 1.0
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0,
                   vmax=max(vmax, 1e-6))

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(model_keys, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_v))
    ax.set_yticklabels(variant_names, fontsize=8)
    for i in range(n_v):
        for j in range(n_m):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if data[i, j] > 0.6 * vmax else "black")
            else:
                ax.text(j, i, "FAIL", ha="center", va="center", fontsize=7,
                        color="#B71C1C")

    ax.set_title(f"Position RMSE: MPPI Variants x Robot Models "
                 f"[{scenario_key}]", fontsize=12)
    fig.colorbar(im, ax=ax, label="Position RMSE (m)", shrink=0.6)
    fig.tight_layout()
    path = os.path.join(plots_dir, f"variants_x_models_heatmap_{scenario_key}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [PLOT] {path}")


def plot_summary(all_results, model_keys, variant_names, variant_groups,
                 scenario_key, plots_dir):
    """2x2 요약: RMSE bar / solve time bar / best-variant 테이블 / scatter"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"MPPI Variants x Models Summary [{scenario_key}]", fontsize=14)

    model_colors = plt.cm.tab10(np.linspace(0, 1, len(model_keys)))

    # (0,0) 모델별 mean/min RMSE bar
    ax = axes[0, 0]
    means, mins = [], []
    for mk in model_keys:
        vals = [r["position_rmse"] for r in all_results[mk].values()
                if r["status"] == "ok"]
        means.append(np.mean(vals) if vals else np.nan)
        mins.append(np.min(vals) if vals else np.nan)
    x = np.arange(len(model_keys))
    ax.bar(x - 0.2, means, 0.4, label="mean (OK variants)", color="#42A5F5")
    ax.bar(x + 0.2, mins, 0.4, label="best variant", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("RMSE by Model")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (0,1) 모델별 mean solve time bar
    ax = axes[0, 1]
    st_means = []
    for mk in model_keys:
        vals = [r["mean_solve_time_ms"] for r in all_results[mk].values()
                if r["status"] == "ok"]
        st_means.append(np.mean(vals) if vals else np.nan)
    ax.bar(x, st_means, 0.5, color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean solve time (ms)")
    ax.set_title("Solve Time by Model (mean over OK variants)")
    ax.grid(axis="y", alpha=0.3)

    # (1,0) 모델별 best variant 테이블
    ax = axes[1, 0]
    ax.axis("off")
    rows = []
    for mk in model_keys:
        ok = {vn: r for vn, r in all_results[mk].items() if r["status"] == "ok"}
        n_fail = sum(1 for r in all_results[mk].values() if r["status"] != "ok")
        if ok:
            best = min(ok, key=lambda vn: ok[vn]["position_rmse"])
            rows.append([mk, best, f"{ok[best]['position_rmse']:.4f}",
                         f"{ok[best]['mean_solve_time_ms']:.1f}",
                         f"{len(ok)}/{len(all_results[mk])}"])
        else:
            rows.append([mk, "--", "--", "--", f"0/{len(all_results[mk])}"])
        _ = n_fail
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Best Variant", "RMSE (m)", "Solve (ms)", "OK/Total"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax.set_title("Best Variant per Model", fontsize=11)

    # (1,1) RMSE vs solve time scatter (색=모델, 마커 없음 표시로 그룹은 생략)
    ax = axes[1, 1]
    for mk, color in zip(model_keys, model_colors):
        xs, ys = [], []
        for vn in variant_names:
            r = all_results[mk].get(vn)
            if r and r["status"] == "ok":
                xs.append(r["mean_solve_time_ms"])
                ys.append(r["position_rmse"])
        ax.scatter(xs, ys, s=28, color=color, label=mk, alpha=0.75)
    ax.set_xlabel("Mean solve time (ms)")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_xscale("log")
    ax.set_title("RMSE vs Solve Time (color = model)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(plots_dir, f"variants_x_models_summary_{scenario_key}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [PLOT] {path}")


# ── JSON 저장 ──────────────────────────────────────────────────

def save_json(output_dir, model_key, scenario_key, scenario, results, duration):
    os.makedirs(output_dir, exist_ok=True)
    payload = dict(
        model=model_key,
        scenario=scenario_key,
        duration=duration,
        common=dict(COMMON_BASE),
        obstacles=[list(o) for o in scenario["obstacles"]],
        results=results,
    )
    path = os.path.join(output_dir, f"{model_key}_{scenario_key}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  [JSON] {path}")


# ── 메인 ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MPPI variants x robot models cross benchmark")
    parser.add_argument("--models", type=str, default=None,
                        help="comma-separated model keys (default: all 6)")
    parser.add_argument("--variants", type=str, default=None,
                        help="comma-separated variant names (default: all)")
    parser.add_argument("--scenario", type=str, default="simple",
                        choices=["simple", "obstacles", "all"])
    parser.add_argument("--duration", type=float, default=10.0,
                        help="simulation duration per cell (s)")
    parser.add_argument("--smoke", action="store_true",
                        help="quick validation: 2 variants x all models x 2s")
    parser.add_argument("--output-dir", type=str,
                        default="results/variants_x_models")
    args = parser.parse_args()

    model_registry = get_model_registry()
    variant_registry = _get_variant_registry()
    variant_by_name = {v["name"]: v for v in variant_registry}

    # 모델 선택
    if args.models:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in model_keys if m not in model_registry]
        if unknown:
            parser.error(f"unknown models: {unknown} "
                         f"(available: {list(model_registry)})")
    else:
        model_keys = list(model_registry)

    # 변형 선택
    if args.smoke:
        variant_names = ["Vanilla", "Log"]
        args.duration = 2.0
        scenario_keys = ["simple"]
    else:
        if args.variants:
            variant_names = [v.strip() for v in args.variants.split(",")
                             if v.strip()]
            unknown = [v for v in variant_names if v not in variant_by_name]
            if unknown:
                parser.error(f"unknown variants: {unknown} "
                             f"(available: {list(variant_by_name)})")
        else:
            variant_names = list(variant_by_name)
        scenario_keys = (["simple", "obstacles"] if args.scenario == "all"
                         else [args.scenario])

    scenarios = get_scenarios(args.duration)
    plots_dir = os.path.join(_REPO_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    variant_groups = {vn: variant_by_name[vn]["group"] for vn in variant_names}

    print(f"\n{'#' * 96}")
    print(f"#  MPPI Variants x Models Benchmark")
    print(f"#  variants={len(variant_names)}, models={len(model_keys)}, "
          f"scenarios={scenario_keys}, duration={args.duration}s")
    print(f"#  K={COMMON_BASE['K']}, N={COMMON_BASE['N']}, "
          f"dt={COMMON_BASE['dt']}, lambda={COMMON_BASE['lambda_']}")
    print(f"{'#' * 96}")

    t_total = time.time()
    for scenario_key in scenario_keys:
        scenario = scenarios[scenario_key]
        all_results = {}

        print(f"\n{'=' * 96}")
        print(f"  SCENARIO: {scenario['name']}")
        print(f"{'=' * 96}")

        for model_key in model_keys:
            model_spec = model_registry[model_key]
            results = {}
            for vn in variant_names:
                variant = variant_by_name[vn]
                t0 = time.time()
                try:
                    cell = run_cell(variant, model_key, model_spec, scenario)
                except Exception as e:
                    # 마지막 안전망: 셀 실패가 전체 실행을 중단하지 않도록
                    cell = dict(status="failed",
                                fail_reason=f"harness: {type(e).__name__}: {e}")
                    traceback.print_exc()
                cell["group"] = variant["group"]
                cell["cell_time_s"] = round(time.time() - t0, 2)
                results[vn] = cell
                tag = ("ok" if cell["status"] == "ok"
                       else f"FAILED: {cell['fail_reason']}")
                print(f"  [{model_key:<14}] {vn:<14} "
                      f"({cell['cell_time_s']:6.1f}s) {tag}")

            all_results[model_key] = results
            print_model_table(model_key, model_spec["label"], scenario_key,
                              results)
            save_json(args.output_dir, model_key, scenario_key, scenario,
                      results, args.duration)

        plot_heatmap(all_results, model_keys, variant_names, scenario_key,
                     plots_dir)
        plot_summary(all_results, model_keys, variant_names, variant_groups,
                     scenario_key, plots_dir)

    print(f"\n[DONE] total {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
