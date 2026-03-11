#!/usr/bin/env python3
"""
Run All Simulation Environments

모든 11개 시뮬레이션 시나리오를 순차 실행하고 요약 테이블 출력.

Usage:
    python run_all.py
    python run_all.py --no-plot
    python run_all.py --scenarios s1 s2 s3
"""

import numpy as np
import argparse
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")

# 시나리오 모듈 임포트
from scenarios.static_obstacle_field import run_scenario as run_s1
from scenarios.dynamic_bouncing import run_scenario as run_s2
from scenarios.chasing_evading import run_scenario as run_s3
from scenarios.multi_robot_coordination import run_scenario as run_s4
from scenarios.waypoint_navigation import run_scenario as run_s5
from scenarios.drifting_disturbance import run_scenario as run_s6
from scenarios.parking_precision import run_scenario as run_s7
from scenarios.racing_mpcc import run_scenario as run_s8
from scenarios.narrow_corridor import run_scenario as run_s9
from scenarios.mixed_challenge import run_scenario as run_s10
from scenarios.c2u_obstacle_field import run_scenario as run_s11


SCENARIOS = {
    "s1": ("S1: Static Obstacle Field", run_s1, {"layout": "random", "no_plot": True}),
    "s2": ("S2: Dynamic Bouncing", run_s2, {"no_plot": True}),
    "s3": ("S3: Chasing Evader", run_s3, {"no_plot": True}),
    "s4": ("S4: Multi-Robot Coordination", run_s4, {"no_plot": True}),
    "s5": ("S5: Waypoint Navigation", run_s5, {"no_plot": True}),
    "s6": ("S6: Drifting Disturbance", run_s6, {"no_plot": True}),
    "s7": ("S7: Parking Precision", run_s7, {"no_plot": True}),
    "s8": ("S8: Racing MPCC", run_s8, {"no_plot": True}),
    "s9": ("S9: Narrow Corridor", run_s9, {"no_plot": True}),
    "s10": ("S10: Mixed Challenge", run_s10, {"no_plot": True}),
    "s11": ("S11: C2U Obstacle Field", run_s11, {"noise": "medium", "no_plot": True}),
}


def main():
    parser = argparse.ArgumentParser(description="Run All Simulation Environments")
    parser.add_argument("--no-plot", action="store_true", default=True,
                        help="Skip plot display (default for batch)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Run specific scenarios (e.g., s1 s2 s5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 실행할 시나리오 결정
    if args.scenarios:
        scenario_keys = [s.lower() for s in args.scenarios]
    else:
        scenario_keys = list(SCENARIOS.keys())

    print("\n" + "=" * 78)
    print("MPPI Simulation Environments — Full Suite".center(78))
    print("=" * 78)
    print(f"Scenarios: {len(scenario_keys)} | Seed: {args.seed}")
    print("=" * 78 + "\n")

    results = {}
    total_start = time.time()

    for key in scenario_keys:
        if key not in SCENARIOS:
            print(f"  [SKIP] Unknown scenario: {key}")
            continue

        name, run_fn, kwargs = SCENARIOS[key]
        kwargs = dict(kwargs)
        kwargs["seed"] = args.seed

        print(f"\n{'─' * 78}")
        print(f"  [{key.upper()}] {name}")
        print(f"{'─' * 78}")

        start = time.time()
        try:
            success = run_fn(**kwargs)
            elapsed = time.time() - start
            results[key] = {"name": name, "status": "PASS", "time": elapsed}
            print(f"  [{key.upper()}] PASS ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            results[key] = {"name": name, "status": "FAIL", "time": elapsed, "error": str(e)}
            print(f"  [{key.upper()}] FAIL ({elapsed:.1f}s): {e}")
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    # 요약 테이블
    print("\n\n" + "=" * 78)
    print("Summary".center(78))
    print("=" * 78)
    print(f"{'ID':>5s} | {'Scenario':>35s} | {'Status':>8s} | {'Time':>8s}")
    print("-" * 78)

    passed = 0
    failed = 0
    for key in scenario_keys:
        if key not in results:
            continue
        r = results[key]
        status_marker = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"{key:>5s} | {r['name']:>35s} | {status_marker:>8s} | {r['time']:>7.1f}s")
        if r["status"] == "PASS":
            passed += 1
        else:
            failed += 1

    print("-" * 78)
    print(f"{'Total':>5s} | {'':>35s} | {passed}P/{failed}F  | {total_elapsed:>7.1f}s")
    print("=" * 78)

    if failed > 0:
        print(f"\nFailed scenarios:")
        for key, r in results.items():
            if r["status"] == "FAIL":
                print(f"  {key}: {r.get('error', 'Unknown')}")
        sys.exit(1)

    print(f"\nAll {passed} scenarios passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
