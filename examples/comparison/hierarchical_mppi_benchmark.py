"""
Hierarchical MPPI vs Vanilla MPPI 벤치마크

글로벌 A*/Theta* 경로계획 + 로컬 MPPI 추적 vs 단순 MPPI 비교.

시나리오:
    A. simple:   장애물 없는 장거리 이동
    B. obstacles: 여러 장애물이 있는 복잡한 환경
    C. maze:     좁은 통로가 있는 미로형 환경

사용법:
    PYTHONPATH=. python examples/comparison/hierarchical_mppi_benchmark.py --no-plot
    PYTHONPATH=. python examples/comparison/hierarchical_mppi_benchmark.py --scenario obstacles
"""

import sys
import time
import argparse
import numpy as np

sys.path.insert(0, ".")

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.hierarchical.hierarchical_mppi import (
    HierarchicalMPPIController,
    HierarchicalMPPIParams,
)


# ─────────────────────────────────────────────────────────────────────────────
# 시나리오 정의
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "simple": {
        "name": "Simple (장애물 없음)",
        "start": np.array([-4.0, -4.0, 0.0]),
        "goal": np.array([4.0, 4.0]),
        "obstacles": [],
    },
    "obstacles": {
        "name": "Obstacles (3개 장애물)",
        "start": np.array([-4.0, 0.0, 0.0]),
        "goal": np.array([4.0, 0.0]),
        "obstacles": [
            (-1.5, 0.5, 0.5),
            (0.0, -0.5, 0.5),
            (1.5, 0.5, 0.5),
        ],
    },
    "maze": {
        "name": "Maze (좁은 통로)",
        "start": np.array([-4.0, 0.0, 0.0]),
        "goal": np.array([4.0, 0.0]),
        "obstacles": [
            (-2.0, 1.5, 0.3), (-2.0, 0.5, 0.3), (-2.0, -0.5, 0.3),
            (0.0, -1.5, 0.3), (0.0, -0.5, 0.3), (0.0, 0.5, 0.3),
            (2.0, 1.5, 0.3), (2.0, 0.5, 0.3), (2.0, -0.5, 0.3),
        ],
    },
}

N_STEPS = 200
DT = 0.05
N_HORIZON = 20
K_SAMPLES = 128


# ─────────────────────────────────────────────────────────────────────────────
# 시뮬레이션
# ─────────────────────────────────────────────────────────────────────────────

def make_vanilla_reference(state, goal, N, dt, v=0.5):
    """목표 방향 직선 참조 궤적."""
    pos = np.array([state[0], state[1]])
    g = np.array([goal[0], goal[1]])
    direction = g - pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.tile(state[:3], (N + 1, 1))
    unit = direction / dist
    theta = np.arctan2(unit[1], unit[0])
    ref = np.zeros((N + 1, 3))
    for i in range(N + 1):
        t = i * dt
        travel = min(v * t, dist)
        ref[i, 0] = pos[0] + unit[0] * travel
        ref[i, 1] = pos[1] + unit[1] * travel
        ref[i, 2] = theta
    return ref


def check_collision(state, obstacles, robot_r=0.15):
    for ox, oy, r in obstacles:
        dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
        if dist < r + robot_r:
            return True
    return False


def run_vanilla(model, scenario, n_steps):
    """Vanilla MPPI 시뮬레이션."""
    start = scenario["start"].copy()
    goal = scenario["goal"]
    obstacles = scenario["obstacles"]

    params = MPPIParams(
        N=N_HORIZON, K=K_SAMPLES, dt=DT,
        lambda_=1.0,
        sigma=np.array([0.5, 0.3]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    ctrl = MPPIController(model, params)

    state = start.copy()
    states = [state.copy()]
    times = []
    collisions = 0

    for _ in range(n_steps):
        ref = make_vanilla_reference(state, goal, N_HORIZON, DT)
        t0 = time.perf_counter()
        u, _ = ctrl.compute_control(state, ref)
        times.append(time.perf_counter() - t0)

        state_dot = model.forward_dynamics(state, u)
        state = state + state_dot * DT
        states.append(state.copy())

        if check_collision(state, obstacles):
            collisions += 1

        # 목표 도달
        if np.linalg.norm(state[:2] - goal[:2]) < 0.3:
            break

    states = np.array(states)
    dist_to_goal = float(np.linalg.norm(states[-1, :2] - goal[:2]))
    return {
        "states": states,
        "dist_to_goal": dist_to_goal,
        "collisions": collisions,
        "mean_time_ms": float(np.mean(times) * 1000),
        "n_steps": len(states) - 1,
        "goal_reached": dist_to_goal < 0.5,
    }


def run_hierarchical(model, scenario, n_steps, planner="thetastar"):
    """Hierarchical MPPI 시뮬레이션."""
    start = scenario["start"].copy()
    goal = scenario["goal"]
    obstacles = scenario["obstacles"]

    params = HierarchicalMPPIParams(
        N=N_HORIZON, K=K_SAMPLES, dt=DT,
        lambda_=1.0,
        sigma=np.array([0.5, 0.3]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        global_planner=planner,
        grid_width=100, grid_height=100,
        grid_resolution=0.1, grid_origin=(-6.0, -6.0),
        robot_radius=0.2,
        goal_tolerance=0.3,
        lookahead_dist=1.5,
        replan_interval=30,
        obstacles=obstacles,
    )
    ctrl = HierarchicalMPPIController(model, params, goal=goal)

    state = start.copy()
    states = [state.copy()]
    times = []
    collisions = 0

    for _ in range(n_steps):
        t0 = time.perf_counter()
        u, info = ctrl.compute_control(state)
        times.append(time.perf_counter() - t0)

        state_dot = model.forward_dynamics(state, u)
        state = state + state_dot * DT
        states.append(state.copy())

        if check_collision(state, obstacles):
            collisions += 1

        if info.get("goal_reached"):
            break

    states = np.array(states)
    dist_to_goal = float(np.linalg.norm(states[-1, :2] - goal[:2]))
    return {
        "states": states,
        "dist_to_goal": dist_to_goal,
        "collisions": collisions,
        "mean_time_ms": float(np.mean(times) * 1000),
        "n_steps": len(states) - 1,
        "goal_reached": dist_to_goal < 0.5,
        "global_path": info.get("global_path"),
    }


def run_scenario(key: str) -> dict:
    scenario = SCENARIOS[key]
    model = DifferentialDriveKinematic()

    print(f"\n{'─'*60}")
    print(f"  시나리오: {scenario['name']}")
    print(f"{'─'*60}")

    results = {}

    # Vanilla
    r = run_vanilla(model, scenario, N_STEPS)
    results["Vanilla MPPI"] = r
    print(f"  Vanilla MPPI    │ goal: {'✓' if r['goal_reached'] else '✗'}"
          f" │ dist: {r['dist_to_goal']:.2f}m"
          f" │ coll: {r['collisions']}"
          f" │ {r['mean_time_ms']:.1f}ms/step")

    # Hierarchical A*
    r = run_hierarchical(model, scenario, N_STEPS, planner="astar")
    results["Hierarchical A*"] = r
    print(f"  Hierarchical A* │ goal: {'✓' if r['goal_reached'] else '✗'}"
          f" │ dist: {r['dist_to_goal']:.2f}m"
          f" │ coll: {r['collisions']}"
          f" │ {r['mean_time_ms']:.1f}ms/step")

    # Hierarchical Theta*
    r = run_hierarchical(model, scenario, N_STEPS, planner="thetastar")
    results["Hierarchical Theta*"] = r
    print(f"  Hierarchical θ* │ goal: {'✓' if r['goal_reached'] else '✗'}"
          f" │ dist: {r['dist_to_goal']:.2f}m"
          f" │ coll: {r['collisions']}"
          f" │ {r['mean_time_ms']:.1f}ms/step")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results: dict, show: bool = True):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib 미설치 — 플롯 스킵")
        return

    n_scenarios = len(all_results)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 6))
    if n_scenarios == 1:
        axes = [axes]

    fig.suptitle("Hierarchical MPPI vs Vanilla MPPI", fontsize=14, fontweight="bold")
    colors = {
        "Vanilla MPPI": "tab:blue",
        "Hierarchical A*": "tab:orange",
        "Hierarchical Theta*": "tab:green",
    }

    for ax, (scenario_key, results) in zip(axes, all_results.items()):
        scenario = SCENARIOS[scenario_key]

        # 장애물
        for ox, oy, r in scenario["obstacles"]:
            circle = patches.Circle((ox, oy), r, color="gray", alpha=0.5)
            ax.add_patch(circle)

        # 시작/목표
        s = scenario["start"]
        g = scenario["goal"]
        ax.plot(s[0], s[1], "ko", ms=10, label="Start")
        ax.plot(g[0], g[1], "r*", ms=12, label="Goal")

        for ctrl_name, r in results.items():
            ax.plot(r["states"][:, 0], r["states"][:, 1],
                    color=colors[ctrl_name], label=ctrl_name, lw=1.5)
            # Theta*의 글로벌 경로 표시
            if ctrl_name == "Hierarchical Theta*" and r.get("global_path"):
                gpath = np.array(r["global_path"])
                ax.plot(gpath[:, 0], gpath[:, 1], "g--", lw=1, alpha=0.4,
                        label="Global Path")

        ax.set_title(f"{scenario['name']}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)

    import os
    os.makedirs("plots", exist_ok=True)
    save_path = "plots/hierarchical_mppi_benchmark.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\n  플롯 저장: {save_path}")

    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hierarchical MPPI 벤치마크")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--scenario", default="all",
                        choices=["simple", "obstacles", "maze", "all"])
    args = parser.parse_args()

    print("=" * 60)
    print("  Hierarchical MPPI 벤치마크")
    print("=" * 60)
    print(f"  K={K_SAMPLES}, N={N_HORIZON}, max_steps={N_STEPS}")

    selected = (
        list(SCENARIOS.keys())
        if args.scenario == "all"
        else [args.scenario]
    )

    all_results = {}
    for key in selected:
        all_results[key] = run_scenario(key)

    # 요약
    print(f"\n{'='*60}")
    print("  최종 요약")
    print(f"{'='*60}")
    print(f"  {'시나리오':<20} {'컨트롤러':<22} {'목표':>5} {'거리':>8} {'충돌':>6} {'시간(ms)':>9}")
    print(f"  {'-'*70}")
    for skey, results in all_results.items():
        for cname, r in results.items():
            print(f"  {SCENARIOS[skey]['name'][:18]:<20} {cname:<22}"
                  f" {'✓' if r['goal_reached'] else '✗':>5}"
                  f" {r['dist_to_goal']:>8.2f}"
                  f" {r['collisions']:>6}"
                  f" {r['mean_time_ms']:>9.1f}")

    if not args.no_plot:
        plot_results(all_results, show=True)


if __name__ == "__main__":
    main()
