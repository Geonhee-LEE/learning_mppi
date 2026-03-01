# Safety-Critical Control Guide

Mathematical background, design principles, and usage for 7 safety control methods integrated into the MPPI framework.

## Table of Contents

1. [Overview](#1-overview)
2. [Standard CBF-MPPI](#2-standard-cbf-mppi)
3. [C3BF (Collision Cone CBF)](#3-c3bf-collision-cone-cbf)
4. [DPCBF (Dynamic Parabolic CBF)](#4-dpcbf-dynamic-parabolic-cbf)
5. [Optimal-Decay CBF](#5-optimal-decay-cbf)
6. [Gatekeeper](#6-gatekeeper)
7. [Shield-MPPI](#7-shield-mppi)
8. [Superellipsoid Obstacles](#8-superellipsoid-obstacles)
9. [Method Comparison](#9-method-comparison)
10. [Usage Guide](#10-usage-guide)
12. [DIAL-MPPI + Learned Model Benchmark](#12-dial-mppi--learned-model-benchmark)

---

## 1. Overview

### 1.1 Why Safety-Critical Control?

MPPI is a sampling-based optimal controller that avoids obstacles through cost functions, but only provides **probabilistic guarantees**. Insufficient samples or inadequate cost weights can lead to collisions.

Safety-Critical Control answers these questions:
- Is a cost penalty sufficient? (Approach A: CBF Cost)
- Can we mathematically guarantee safety? (Approach B: QP Safety Filter)
- Can all sample trajectories be safe? (Approach C: Shield-MPPI)
- Can we guarantee safety for infinite time? (Gatekeeper)

### 1.2 Control Barrier Function (CBF) Basics

A CBF defines a safe set `C = {x : h(x) >= 0}` through a function `h(x)`.

```
h(x) > 0   ->  Safe
h(x) = 0   ->  Boundary
h(x) < 0   ->  Unsafe
```

**Continuous-time CBF condition:**

```
dh/dt(x) + alpha * h(x) >= 0
```

When satisfied, `h(x(t)) >= 0` remains invariant for all future time.

**Discrete-time CBF condition (used in this project):**

```
h(x_{t+1}) - (1 - alpha) * h(x_t) >= 0
```

Here `alpha in (0, 1]` is the decay rate; larger values are more conservative (faster barrier recovery).

### 1.3 Architecture Overview

```
+------------------------------------------------------+
|                    MPPI Controller                    |
|  +------------------------------------------------+  |
|  | Layer 1: Sampling + Cost                       |  |
|  |  +---------+  +----------+  +--------------+   |  |
|  |  | Gaussian|->| Rollout  |->| Cost Function|   |  |
|  |  | Noise   |  | (K x N)  |  | (CBF/C3BF/  |   |  |
|  |  +---------+  +----------+  |  DPCBF/Super)|   |  |
|  |                              +------+-------+   |  |
|  |                                     v           |  |
|  |                              +--------------+   |  |
|  |                              | Softmax      |   |  |
|  |                              | Weights      |   |  |
|  |                              +------+-------+   |  |
|  +-------------------------------------+-----------+  |
|                                        v              |
|  +------------------------------------------------+   |
|  | Layer 2: Safety Filter (optional)              |   |
|  |  +-------------+  +------------------------+   |   |
|  |  | QP Filter   |  | Optimal-Decay          |   |   |
|  |  | min||u-u*||^2| | min||u-u*||^2+p(w-1)^2 |   |   |
|  |  | s.t. CBF>=0 |  | s.t. CBF*w>=0          |   |   |
|  |  +-------------+  +------------------------+   |   |
|  +------------------------------------------------+   |
|                                        v              |
|  +------------------------------------------------+   |
|  | Layer 3: Gatekeeper (optional)                 |   |
|  |  backup trajectory safe -> gate open / closed  |   |
|  +------------------------------------------------+   |
+------------------------------------------------------+
```

Or the Shield-MPPI path:

```
+------------------------------------------------------+
| Shield-MPPI: per-step CBF enforcement in rollout     |
|                                                      |
|  Noise -> Per-Step CBF Clip -> Safe Rollout -> Cost  |
|                                -> Softmax -> Control |
|  All K samples are always safe                       |
+------------------------------------------------------+
```

---

## 2. Standard CBF-MPPI

> **Files**: `cbf_cost.py`, `cbf_safety_filter.py`, `cbf_mppi.py`
> **Paper**: Zeng et al. (2021) — "Safety-Critical MPC with Discrete-Time CBF"

### 2.1 Barrier Function

For a circular obstacle `(x_o, y_o, r)`:

```
h(x) = (x - x_o)^2 + (y - y_o)^2 - (r + margin)^2
```

- `h > 0`: outside obstacle (safe)
- `h = 0`: obstacle boundary
- `h < 0`: inside obstacle (collision)

### 2.2 Approach A — CBF Cost Penalty

Adds a CBF violation penalty to the MPPI cost function:

```
cost_cbf = w_cbf * sum_t max(0, -(h(x_{t+1}) - (1-alpha)*h(x_t)))
```

Zero cost when no violation; penalty scales with `w_cbf` upon violation.

**Pros**: No change to MPPI structure, naturally combines with other costs
**Cons**: Only probabilistic guarantee (violation possible with insufficient weight)

### 2.3 Approach B — QP Safety Filter

Minimally modifies MPPI output `u_mppi` to satisfy CBF conditions:

```
min   ||u - u_mppi||^2
s.t.  Lf*h + Lg*h*u + alpha*h >= 0    (per obstacle)
      u_min <= u <= u_max
```

**Lie Derivatives (Differential Drive):**

```
f(x) = [0, 0, 0]^T                     (drift-free kinematic)
g(x) = [[cos(th), 0], [sin(th), 0], [0, 1]]^T

Lf*h = 0
Lg*h = [2(x-x_o)*cos(th) + 2(y-y_o)*sin(th),  0]
```

Physical meaning of `Lg*h`: effect of linear velocity `v` on the barrier. Negative when heading toward obstacle.

**Pros**: Mathematical safety guarantee (1-step)
**Cons**: QP solver cost, guarantee only at current state

### 2.4 Two-Layer Integration

`CBFMPPIController` combines both layers:

```python
# Layer 1: MPPI with CBF cost included
control, info = super().compute_control(state, ref)

# Layer 2 (optional): QP Safety Filter
if self.safety_filter:
    control, filter_info = self.safety_filter.filter_control(state, control)
```

### 2.5 Usage

```python
from mppi_controller.controllers.mppi.mppi_params import CBFMPPIParams
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController

params = CBFMPPIParams(
    N=20, dt=0.05, K=512, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
    cbf_obstacles=[(3.0, 0.5, 0.4), (5.0, -0.3, 0.3)],
    cbf_weight=1000.0,         # violation penalty weight
    cbf_alpha=0.1,             # barrier decay rate
    cbf_safety_margin=0.15,    # additional safety margin (m)
    cbf_use_safety_filter=True,  # enable Layer 2 QP
)
controller = CBFMPPIController(model, params)
```

### 2.6 Parameter Tuning

| Parameter | Role | Recommended | Effect |
|-----------|------|-------------|--------|
| `cbf_weight` | Violation penalty | 500~2000 | Higher = stronger avoidance, too high = unable to progress |
| `cbf_alpha` | Decay rate | 0.05~0.3 | Higher = more conservative (faster barrier recovery) |
| `cbf_safety_margin` | Extra buffer | 0.1~0.3m | Higher = maintain greater distance from obstacles |

---

## 3. C3BF (Collision Cone CBF)

> **File**: `c3bf_cost.py`
> **Paper**: Thirugnanam et al. (2024) — "Safety-Critical Control with Collision Cone CBFs"

### 3.1 Key Idea

Standard CBF only considers **distance**. Even when moving away from an obstacle, proximity incurs a penalty.

C3BF also considers the **relative velocity direction**. Moving away = safe, moving toward = unsafe.

```
Standard CBF:  h = ||p||^2 - R^2       (distance only)
C3BF:          h = <p, v> + ||p||*||v||*cos(phi)   (distance + velocity direction)
```

### 3.2 Collision Cone Geometry

```
              Robot
               *-> v_robot
              /|
             / |
            /  |  phi = cone half-angle
           /   |
          /    |  = arcsin(R/||p||)
         /     |
-----------*-----------
         Obstacle
           (R)

cos(phi_safe) = sqrt(||p||^2 - R^2) / ||p||
```

**Barrier function:**

```
p_rel = p_robot - p_obstacle        (relative position)
v_rel = v_robot - v_obstacle        (relative velocity)

h = <p_rel, v_rel> + ||p_rel|| * ||v_rel|| * cos(phi_safe)
```

- `h > 0`: relative velocity vector **outside** collision cone -> safe
- `h <= 0`: relative velocity vector **inside** cone -> collision path

### 3.3 Velocity Estimation

Robot velocity is computed via finite difference of positions:

```
v_robot[t] = (pos[t+1] - pos[t]) / dt
```

Obstacle velocity `(vx, vy)` is provided externally (tracker or scenario config).

### 3.4 Usage

```python
from mppi_controller.controllers.mppi.c3bf_cost import CollisionConeCBFCost
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost, StateTrackingCost, TerminalCost, ControlEffortCost,
)

# Dynamic obstacle: (x, y, radius, vx, vy)
dynamic_obstacles = [(4.0, 2.0, 0.3, 0.0, -0.3)]

c3bf_cost = CollisionConeCBFCost(
    obstacles=dynamic_obstacles,
    cbf_weight=1000.0,
    safety_margin=0.15,
    dt=0.05,
)

composite = CompositeMPPICost([
    StateTrackingCost(Q), TerminalCost(Q), ControlEffortCost(R),
    c3bf_cost,
])
controller = MPPIController(model, params, cost_function=composite)
```

### 3.5 Properties

| Property | Value |
|----------|-------|
| Obstacle repr. | (x, y, r, vx, vy) |
| Dynamic obstacles | Native support |
| Complexity | O(K * N * num_obs) |
| Safety guarantee | Probabilistic (cost-based) |
| Key advantage | Zero cost when receding -> efficient paths |

---

## 4. DPCBF (Dynamic Parabolic CBF)

> **File**: `dpcbf_cost.py`
> **Paper**: Kim et al. (2026) — "Dynamic Parabolic CBFs" (ICRA 2026)

### 4.1 Key Idea

C3BF only considers velocity **direction**. DPCBF dynamically adjusts the safety boundary based on **approach direction and speed magnitude**.

Enlarges the safety margin for head-on approaches, reduces it for lateral passages, enabling efficient navigation through narrow corridors.

### 4.2 Line-of-Sight (LoS) Coordinates

```
                  <- v_approach

    Robot *------------> Obstacle *
          |<- r (dist) ->|

    LoS coordinates:
      r = ||p_robot - p_obs||            (distance)
      beta = angle(v_rel, -p_rel)        (approach angle)
          beta ~ 0   -> head-on approach
          beta ~ pi/2 -> lateral passage
```

### 4.3 Adaptive Boundary

The safety boundary varies with direction in a Gaussian shape:

```
r_safe(beta) = R_eff + a(v_app) * exp(-beta^2 / (2*sigma_beta^2))
```

Where:
- `a(v_app) = a_base + a_vel * max(0, v_approach)` — boundary expands with approach speed
- `sigma_beta` — directional dependency width (default 0.8 rad)

```
r_safe
  ^
  |     ,-.        <- head-on (beta~0): large margin
  |    /   \
  |   /     \
  |--/-------\---- <- R_eff (base radius)
  | /         \
  +--------------> beta
  0    pi/2     pi
  head  lateral  rear
```

**Barrier function:**

```
h = r - r_safe(beta)

h > 0  ->  safe (outside boundary)
h <= 0 ->  unsafe (inside boundary)
```

### 4.4 Usage

```python
from mppi_controller.controllers.mppi.dpcbf_cost import DynamicParabolicCBFCost

dpcbf = DynamicParabolicCBFCost(
    obstacles=[(4.0, 2.0, 0.3, 0.0, -0.3)],
    cbf_weight=1000.0,
    safety_margin=0.15,
    a_base=0.3,    # base Gaussian amplitude (m)
    a_vel=0.5,     # speed coupling coefficient (s)
    sigma_beta=0.8, # directional dependency width (rad)
    dt=0.05,
)
```

### 4.5 Parameter Effects

| Parameter | Effect |
|-----------|--------|
| `a_base` up | Larger head-on margin, earlier avoidance start |
| `a_vel` up | Margin increases sharply with fast approach |
| `sigma_beta` down | Focused on head-on only, lateral margin drops quickly |
| `sigma_beta` up | Uniform margin in all directions (converges to Standard CBF) |

---

## 5. Optimal-Decay CBF

> **File**: `optimal_decay_cbf_filter.py`
> **Paper**: Gurriet et al. (2020) — "Scalable Safety-Critical Control"

### 5.1 Key Idea

Standard QP Safety Filter fails when constraints are **infeasible** (cannot satisfy all simultaneously). Optimal-Decay CBF adds the decay rate `omega` as an optimization variable, ensuring a **solution always exists**.

### 5.2 Mathematical Definition

**Standard CBF QP:**
```
min   ||u - u_mppi||^2
s.t.  Lf*h + Lg*h*u + alpha*h >= 0
```

**Optimal-Decay CBF QP:**
```
min   ||u - u_mppi||^2 + p_sb*(omega - 1)^2
s.t.  Lf*h + Lg*h*u + alpha*omega*h >= 0
      u_min <= u <= u_max
      omega_min <= omega <= omega_max
```

| Variable | Meaning |
|----------|---------|
| `omega = 1` | Same as standard CBF (full safety) |
| `0 < omega < 1` | Relaxed CBF (graceful degradation) |
| `omega = 0` | CBF condition disabled (last resort) |
| `p_sb` | Slack penalty (default 1e4), higher = stronger omega=1 enforcement |

### 5.3 Feasibility Guarantee

When `omega = 0`, the constraint becomes `Lf*h >= 0`, which is always satisfiable (for drift-free systems, `Lf*h = 0`).

Therefore, the Optimal-Decay QP is **always feasible**.

### 5.4 Usage

```python
from mppi_controller.controllers.mppi.optimal_decay_cbf_filter import OptimalDecayCBFSafetyFilter
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController

# Attach Optimal-Decay filter to CBF-MPPI
params = CBFMPPIParams(
    ...,
    cbf_use_safety_filter=True,
)
controller = CBFMPPIController(model, params)

# Replace safety filter with Optimal-Decay version
controller.safety_filter = OptimalDecayCBFSafetyFilter(
    obstacles=obstacles,
    cbf_alpha=0.1,
    safety_margin=0.15,
    penalty_weight=1e4,   # omega != 1 penalty (higher = more conservative)
)
```

### 5.5 Properties

- **Feasibility guarantee**: Valid control output in any situation
- **Most conservative**: `p_sb=1e4` strongly maintains omega=1 -> wide margins
- **Interpretable**: `omega` value quantifies safety margin

---

## 6. Gatekeeper

> **Files**: `gatekeeper.py`, `backup_controller.py`
> **Paper**: Gurriet et al. (2020) — "Scalable Safety-Critical Control of Robotic Systems"

### 6.1 Key Idea

Other methods guarantee only **1-step** safety. Gatekeeper guarantees **infinite-time** safety.

Principle: "After applying the current control, can we return to safety via a backup policy?"

```
MPPI proposes: u_mppi
  -> next state: x_next = f(x, u_mppi, dt)
  -> generate backup trajectory: [x_next, x_{next+1}, ...]
  -> is the entire backup trajectory safe?
    -> Yes (Gate Open):  u_out = u_mppi
    -> No (Gate Closed): u_out = u_backup
```

### 6.2 Backup Controller

Two backup policies are provided:

**BrakeBackupController:**
```
u = [0, 0]  (immediate stop)
```
Valid for kinematic models with no stopping inertia.

**TurnAndBrakeBackupController:**
```
Phase 1 (turn_steps):  u = [0, +/-turn_speed]  (rotate away from obstacle)
Phase 2 (remaining):   u = [0, 0]               (stop)
```
Rotates away from the obstacle, then stops to reach a safer state.

### 6.3 Infinite-Time Safety Principle

```
+------------------------------------------+
| Invariant: "can always stop safely"      |
|                                          |
|  t=0: backup safe -> gate open           |
|  t=1: apply u_mppi -> re-check backup    |
|       -> safe: gate open                 |
|       -> unsafe: gate closed -> backup   |
|  t=2: if in backup, already safe         |
|                                          |
|  -> maintain "can stop" at every t       |
|  -> infinite-time safety                 |
+------------------------------------------+
```

### 6.4 Usage

```python
from mppi_controller.controllers.mppi.gatekeeper import Gatekeeper
from mppi_controller.controllers.mppi.backup_controller import (
    BrakeBackupController,
    TurnAndBrakeBackupController,
)

gatekeeper = Gatekeeper(
    backup_controller=TurnAndBrakeBackupController(turn_speed=0.5, turn_steps=5),
    model=model,
    obstacles=obstacles,
    safety_margin=0.2,
    backup_horizon=30,  # backup trajectory length
    dt=0.05,
)

# Apply in control loop
control, info = mppi_controller.compute_control(state, ref)
safe_control, gk_info = gatekeeper.filter(state, control)
# gk_info["gate_open"] = True/False
```

### 6.5 Properties

| Property | Value |
|----------|-------|
| Safety guarantee | Infinite-time (forward invariance) |
| Compute cost | O(backup_horizon x num_obs) per step |
| Dynamic obstacles | `update_obstacles()` supported |
| Drawback | Conservative backup may slow progress |

---

## 7. Shield-MPPI

> **File**: `shield_mppi.py`

### 7.1 Key Idea

Standard MPPI: reduces weight of unsafe trajectories via cost -> unsafe trajectories still **exist**.

Shield-MPPI: applies CBF constraint at **every timestep** of rollout -> all K samples are **guaranteed safe**.

```
Standard MPPI:   sample -> rollout (unsafe possible) -> cost -> weight
Shield-MPPI:     sample -> CBF clip -> rollout (always safe) -> cost -> weight
```

### 7.2 Analytical CBF Shield

Computed in closed-form without a QP solver:

```
For each sample k, timestep t, obstacle i:
    h = (x-x_o)^2 + (y-y_o)^2 - r_eff^2
    Lg_h = 2(x-x_o)*cos(th) + 2(y-y_o)*sin(th)

    if Lg_h < 0:  (moving toward obstacle)
        v_ceiling = alpha*h / |Lg_h|
        v_safe = min(v_raw, v_ceiling)
    else:          (moving away from obstacle)
        v_safe = v_raw  (no constraint)
```

The most conservative `v_ceiling` across all obstacles is applied.

### 7.3 Shielded Noise Correction

When the shield modifies controls, the noise distribution becomes skewed. This is corrected:

```
Standard MPPI:   noise = sampled_controls - U
Shield-MPPI:     noise = shielded_controls - U  (based on corrected controls)
```

This prevents the weight update from being biased by the shield.

### 7.4 Usage

```python
from mppi_controller.controllers.mppi.mppi_params import ShieldMPPIParams
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController

params = ShieldMPPIParams(
    N=20, dt=0.05, K=512, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]),
    R=np.array([0.1, 0.1]),
    cbf_obstacles=obstacles,
    cbf_weight=1000.0,
    cbf_alpha=0.1,
    cbf_safety_margin=0.15,
    shield_enabled=True,
    shield_cbf_alpha=0.2,  # separate alpha for shield (optional)
)
controller = ShieldMPPIController(model, params)
```

### 7.5 Properties

| Property | Value |
|----------|-------|
| Safety guarantee | Hard (all samples safe) |
| Complexity | O(K x N x num_obs) |
| QP solver | Not required (analytical closed-form) |
| Dynamic obstacles | `update_obstacles()` supported |
| Key advantage | Maintains MPPI exploration + safety guarantee simultaneously |

---

## 8. Superellipsoid Obstacles

> **File**: `superellipsoid_cost.py`

### 8.1 Key Idea

Circular obstacles alone cannot accurately model walls, vehicles, furniture, and other non-circular objects. Superellipsoids represent circles, ellipses, rectangles, and more with a single equation.

### 8.2 Superellipse Equation

```
(|x'/a|)^n + (|y'/b|)^n = 1
```

Local coordinate transform:
```
[x', y'] = R(-theta) * [x - cx, y - cy]
```

**Shape parameter `n` effects:**
```
n = 1    ->  diamond
n = 2    ->  ellipse (circle if a=b)
n = 4    ->  rounded rectangle
n = 10   ->  near-rectangle
n -> inf ->  true rectangle
```

### 8.3 Barrier Function

```
h(x, y) = (|x'/a|)^n + (|y'/b|)^n - 1

h > 0  ->  outside (safe)
h = 0  ->  boundary
h < 0  ->  inside (collision)
```

The discrete-time CBF condition is applied identically to Standard CBF.

### 8.4 Usage

```python
from mppi_controller.controllers.mppi.superellipsoid_cost import (
    SuperellipsoidObstacle, SuperellipsoidCost,
)

obstacles = [
    SuperellipsoidObstacle(cx=3.0, cy=0.0, a=1.0, b=0.3, n=4, theta=0.0),  # wall
    SuperellipsoidObstacle(cx=5.0, cy=1.0, a=0.5, b=0.5, n=2, theta=0.0),  # circle
    SuperellipsoidObstacle(cx=7.0, cy=-0.5, a=0.8, b=0.4, n=6, theta=np.pi/4),  # rotated rect
]

cost = SuperellipsoidCost(
    obstacles=obstacles,
    cbf_alpha=0.1,
    cbf_weight=1000.0,
    safety_margin=0.1,  # added uniformly to a, b
)
```

---

## 9. Method Comparison

### 9.1 Safety Guarantee Levels

```
                    Safety Guarantee Strength
                    ----------------------->

Probabilistic    Conditional       Hard Guarantee    Infinite-Time
(Cost-based)     (QP Filter)       (Shield)          (Gatekeeper)
+---------+    +-------------+    +----------+      +----------+
| CBF Cost|    | CBF Filter  |    | Shield-  |      |Gatekeeper|
| C3BF    |    | Optimal-    |    | MPPI     |      |          |
| DPCBF   |    | Decay       |    |          |      |          |
| Super-  |    |             |    |          |      |          |
| ellip.  |    |             |    |          |      |          |
+---------+    +-------------+    +----------+      +----------+
```

### 9.2 Comprehensive Comparison

| Method | Integration | Safety Guarantee | Dynamic Obs. | QP Required | Key Advantage |
|--------|-------------|-----------------|-------------|-------------|---------------|
| **Standard CBF** | Cost + QP filter | 1-step | Position only | Layer 2 | General-purpose baseline |
| **C3BF** | Cost | Probabilistic | Incl. velocity | No | Direction-aware -> efficient paths |
| **DPCBF** | Cost | Probabilistic | Incl. velocity | No | Adaptive boundary -> narrow passages |
| **Optimal-Decay** | QP filter | Feasibility guaranteed | Position only | Yes | Always-feasible solution |
| **Gatekeeper** | Post-verification | Infinite-time | `update_obstacles()` | No | Strongest safety guarantee |
| **Shield-MPPI** | Rollout modification | Hard (all samples) | Position only | No | Exploration + safety simultaneously |
| **Superellipsoid** | Cost | Probabilistic | Position only | No | Non-circular obstacles |

### 9.3 Benchmark Results

**Static Scenario (3 static obstacles)**

| Method | Solve (ms) | Min Clearance (m) | Collision |
|--------|-----------|-------------------|-----------|
| Standard CBF | 2.1 | 0.22 | No |
| C3BF | 2.5 | 0.15 | No |
| DPCBF | 2.6 | 0.21 | No |
| Optimal-Decay | 2.7 | 1.12 | No |
| Gatekeeper | 2.7 | 0.24 | No |

**Crossing Scenario (2 crossing dynamic obstacles)**

| Method | Solve (ms) | Min Clearance (m) | Collision |
|--------|-----------|-------------------|-----------|
| Standard CBF | 2.0 | 1.70 | No |
| C3BF | 2.3 | 0.37 | No |
| DPCBF | 2.5 | 1.70 | No |
| Optimal-Decay | 2.6 | 1.88 | No |
| Gatekeeper | 2.6 | 1.70 | No |

**Narrow Scenario (4 obstacles forming narrow passage)**

| Method | Solve (ms) | Min Clearance (m) | Collision |
|--------|-----------|-------------------|-----------|
| Standard CBF | 2.1 | 0.50 | No |
| C3BF | 2.5 | 0.50 | No |
| DPCBF | 2.7 | 0.50 | No |
| Optimal-Decay | 2.9 | 1.19 | No |
| Gatekeeper | 2.7 | 0.50 | No |

> All 15 runs (5 methods x 3 scenarios) achieved **zero collisions**.

---

## 10. Usage Guide

### 10.1 Scenario-Based Recommendations

| Scenario | 1st Choice | 2nd Choice | Reason |
|----------|-----------|-----------|--------|
| Static obstacle avoidance | Standard CBF | Shield-MPPI | Simple, fast |
| Dynamic obstacles (slow) | C3BF | DPCBF | Velocity-direction awareness |
| Dynamic obstacles (fast) | DPCBF | Shield-MPPI | Adaptive boundary |
| Narrow passage | DPCBF | C3BF | Reduced lateral margin |
| Dense environment | Optimal-Decay | Gatekeeper | Feasibility guarantee |
| Safety-first | Gatekeeper | Shield-MPPI | Infinite-time / hard guarantee |
| Non-circular obstacles | Superellipsoid | — | Only non-circular support |

### 10.2 Combining Methods

Methods are designed to be **orthogonal** and can be freely combined:

```python
# CBF Cost (Layer 1) + Optimal-Decay (Layer 2) + Gatekeeper (Layer 3)
controller = CBFMPPIController(model, params)
controller.safety_filter = OptimalDecayCBFSafetyFilter(...)
gatekeeper = Gatekeeper(...)

control, info = controller.compute_control(state, ref)
safe_control, gk_info = gatekeeper.filter(state, control)
```

### 10.3 Running Demos

```bash
# Batch mode (saves PNG)
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --scenario static
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --scenario crossing
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --scenario narrow

# Live animation (2x3 layout, 5 methods simultaneously)
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --live
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --live --scenario crossing
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --live --scenario narrow
```

**Live mode layout:**

```
+--------------+--------------+--------------+
| Standard CBF | C3BF (Cone)  | DPCBF        |
|  XY + obs    |  XY + obs    |  XY + obs    |
+--------------+--------------+--------------+
| Optimal-Decay| Gatekeeper   | Min Clearance |
|  XY + obs    |  XY + obs    |  live compare |
+--------------+--------------+--------------+
```

### 10.4 Dynamic Obstacle Updates

When obstacle positions change in a real-time system:

```python
# CBF-MPPI (Standard, Optimal-Decay)
controller.update_obstacles([(x1, y1, r1), (x2, y2, r2)])

# Gatekeeper
gatekeeper.update_obstacles([(x1, y1, r1), (x2, y2, r2)])

# C3BF / DPCBF (including velocity)
c3bf_cost.update_obstacles([(x1, y1, r1, vx1, vy1)])
```

---

## 11. Extended Safety Methods (Phase S4)

Six additional safety-critical control methods were added in Phase S4, bringing the total to **16 methods**.

### 11.1 HorizonWeightedCBFCost — Time-Discounted CBF

Applies temporal discount `γ^t` to CBF violations so near-future violations are penalized more.

```python
from mppi_controller.controllers.mppi.horizon_cbf_cost import HorizonWeightedCBFCost

cost = HorizonWeightedCBFCost(
    obstacles=[(2.0, 0.0, 0.5)],
    weight=100.0,          # CBF violation weight
    cbf_alpha=0.3,         # discrete CBF alpha
    discount_gamma=0.9,    # γ < 1: conservative, γ = 1: standard CBF
    safety_margin=0.05,
)
# Use in CompositeMPPICost
composite = CompositeMPPICost([StateTrackingCost(Q), TerminalCost(Qf), cost])
controller = MPPIController(model, params, composite)
```

### 11.2 HardCBFCost — Binary Rejection

Assigns `rejection_cost` (default 1e6) to any trajectory that penetrates an obstacle.

```python
from mppi_controller.controllers.mppi.hard_cbf_cost import HardCBFCost

cost = HardCBFCost(
    obstacles=[(2.0, 0.0, 0.5)],
    rejection_cost=1e6,    # effectively zero softmax weight
    safety_margin=0.05,
)
```

### 11.3 MPSController — Model Predictive Shield

Lightweight stateless safety shield (simplified Gatekeeper).

```python
from mppi_controller.controllers.mppi.mps_controller import MPSController
from mppi_controller.controllers.mppi.backup_controller import BrakeBackupController

mps = MPSController(
    backup_controller=BrakeBackupController(),
    obstacles=obstacles,
    safety_margin=0.15,
    backup_horizon=20,
    dt=0.05,
)

# In control loop:
u_mppi, info = controller.compute_control(state, ref)
u_safe, mps_info = mps.shield(state, u_mppi, model)  # stateless
```

### 11.4 AdaptiveShieldMPPIController — Distance/Velocity-Adaptive Shield

Shield-MPPI with adaptive alpha that **decreases** near obstacles (more conservative):

```
α(d,v) = α_base · (α_dist + (1 - α_dist) · σ(k·(d - d_safe))) / (1 + α_vel · |v|)
```

- `d >> d_safe` → `σ ≈ 1` → `α ≈ α_base` (relaxed, normal Shield)
- `d << d_safe` → `σ ≈ 0` → `α ≈ α_base · α_dist` (very conservative)
- High `|v|` → denominator grows → `α` decreases (conservative when fast)

Lower α → lower `v_ceiling = α·h/|Lg_h|` → stronger speed reduction near obstacles.

```python
from mppi_controller.controllers.mppi.adaptive_shield_mppi import (
    AdaptiveShieldMPPIController, AdaptiveShieldParams,
)

params = AdaptiveShieldParams(
    N=15, dt=0.05, K=256, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]), R=np.array([0.1, 0.1]),
    cbf_obstacles=obstacles, cbf_alpha=0.1,
    shield_enabled=True,
    alpha_base=0.3,   # max alpha (far from obstacles)
    alpha_dist=0.1,   # min alpha ratio (close: α → α_base × 0.1 = 0.03)
    alpha_vel=0.5,    # velocity reactivity (α /= (1 + 0.5·|v|))
    k_dist=2.0,       # sigmoid steepness
    d_safe=0.5,       # safe distance threshold (m)
)
controller = AdaptiveShieldMPPIController(model, params)
u, info = controller.compute_control(state, ref)
```

### 11.5 CBFGuidedSamplingMPPIController — Rejection + Gradient Bias

Resamples CBF-violating trajectories with `∇h`-biased noise.

```python
from mppi_controller.controllers.mppi.cbf_guided_sampling_mppi import (
    CBFGuidedSamplingMPPIController, CBFGuidedSamplingParams,
)

params = CBFGuidedSamplingParams(
    N=15, dt=0.05, K=256, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]), R=np.array([0.1, 0.1]),
    cbf_obstacles=obstacles, cbf_alpha=0.1, cbf_weight=1000.0,
    rejection_ratio=0.3,         # max fraction to resample
    gradient_bias_weight=0.1,    # ∇h bias strength
    max_resample_iters=3,        # max resample loops
)
controller = CBFGuidedSamplingMPPIController(model, params)
```

### 11.6 ShieldSVGMPPIController — Shield + SVG-MPPI

Combines SVG-MPPI's high-quality SVGD sampling with per-step CBF shield enforcement.

```python
from mppi_controller.controllers.mppi.shield_svg_mppi import (
    ShieldSVGMPPIController, ShieldSVGMPPIParams,
)

params = ShieldSVGMPPIParams(
    N=15, dt=0.05, K=256, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    Q=np.array([10.0, 10.0, 1.0]), R=np.array([0.1, 0.1]),
    svg_num_guide_particles=16,
    svgd_num_iterations=5,
    shield_enabled=True,
    shield_cbf_alpha=0.3,
    cbf_obstacles=obstacles,
    cbf_safety_margin=0.1,
)
controller = ShieldSVGMPPIController(model, params)
```

### 11.7 Shield-DIAL-MPPI — DIAL Annealing + CBF Shield

DIAL-MPPI (Diffusion Annealing)의 multi-iteration 샘플링에 per-step CBF shield를 결합합니다.

```
DIAL-MPPI:  반복 어닐링 (noise decay) → 점진적으로 최적 궤적 수렴
Shield:     매 rollout step에서 CBF constraint → h(x) ≥ 0 보장
Adaptive:   α(d, v) 적응형 — 거리/속도에 따라 shield 강도 조절
```

**계층 구조**:
```
DIALMPPIController             — 확산 어닐링 (cold/warm start)
└── ShieldDIALMPPIController   — + per-step CBF shield
    └── AdaptiveShieldDIALMPPIController — + 적응형 α(d,v)
```

```python
from mppi_controller.controllers.mppi.shield_dial_mppi import (
    ShieldDIALMPPIController,
)
from mppi_controller.controllers.mppi.mppi_params import ShieldDIALMPPIParams

params = ShieldDIALMPPIParams(
    N=20, dt=0.05, K=512, lambda_=1.0,
    sigma=np.array([0.4, 0.8]),
    Q=np.array([10.0, 10.0, 1.0]), R=np.array([0.1, 0.05]),
    # DIAL annealing
    n_diffuse_init=10, n_diffuse=5,
    traj_diffuse_factor=0.6, horizon_diffuse_factor=0.5,
    sigma_scale=1.0, use_reward_normalization=True,
    # Shield CBF
    cbf_obstacles=obstacles, cbf_alpha=0.3,
    cbf_safety_margin=0.15,
    shield_enabled=True, shield_cbf_alpha=2.0,
)
controller = ShieldDIALMPPIController(model, params, cost_function=cost_fn)
```

```bash
# Shield-DIAL 4종 벤치마크 (Vanilla / DIAL / Shield-DIAL / Adaptive-DIAL)
PYTHONPATH=. python examples/comparison/shield_dial_mppi_benchmark.py --live --K 512
```

### 11.8 14-Method Benchmark (+ DIAL variants)

```bash
# Full benchmark (all 14 methods, dense_static scenario)
python examples/comparison/safety_novel_benchmark_demo.py

# Specific scenario
python examples/comparison/safety_novel_benchmark_demo.py --scenario mixed

# Select methods (by number)
python examples/comparison/safety_novel_benchmark_demo.py --methods 1,3,12,14

# All 4 scenarios
python examples/comparison/safety_novel_benchmark_demo.py --all-scenarios

# No plot (console output only)
python examples/comparison/safety_novel_benchmark_demo.py --no-plot
```

### 11.9 Method Selection Guide

| Scenario | Recommended | Why |
|----------|------------|-----|
| Dense static obstacles | AdaptiveShield | Distance-adaptive conservatism |
| Fast dynamic obstacles | C3BF + Shield | Velocity-aware + rollout safety |
| Narrow corridors | Shield-MPPI | Per-step enforcement |
| Mixed challenge | ShieldSVG | High sample quality + safety |
| Real-time critical | HardCBF | Minimal compute overhead |
| Formal guarantee needed | Gatekeeper / MPS | Infinite-time safety proof |
| Wind + model mismatch | Shield-DIAL+EKF | 파라미터 추정 + CBF 안전 |
| Constant external bias | Shield-DIAL+L1 | 외란 추정 + CBF 안전 |

### 11.10 MPPI vs safe_control Benchmark

Compares our MPPI safety methods against [tkkim-robot/safe_control](https://github.com/tkkim-robot/safe_control)
(CBF-QP, MPC-CBF) on identical obstacle avoidance scenarios.

**Scenarios**: Circle (r=3, 4 diverse obstacles) + Gauntlet (6 diverse obstacles)

```bash
# Full benchmark (8 methods, 2 scenarios, saves PNG)
python examples/comparison/mppi_vs_safe_control_benchmark.py

# Live animation
python examples/comparison/mppi_vs_safe_control_benchmark.py --live --scenario circle_obstacle

# Specific methods
python examples/comparison/mppi_vs_safe_control_benchmark.py --methods cbf_qp,shield,adaptive_shield
```

**Results (Cross-Scenario Average, Path-Following RMSE):**

| Method | Collisions | Safety% | PathRMSE | Time(ms) |
|--------|-----------|---------|----------|----------|
| **Adaptive Shield** | 0 | 100% | 0.381m | 7.5ms |
| **Shield-MPPI** | 0 | 100% | 0.429m | 5.7ms |
| **CBF-QP** (safe_control) | 0 | 100% | 0.671m | 1.3ms |
| **MPC-CBF** (safe_control) | 0 | 100% | 0.804m | 8.6ms |
| Vanilla MPPI | 84 | 72% | 0.486m | 3.1ms |

Key findings:
- **Adaptive Shield-MPPI** achieves best path tracking while maintaining 100% safety
- **Shield-MPPI** provides hard safety guarantee with low path error
- **CBF-QP** is fastest (1.3ms) but over-conservative on curved paths
- **MPC-CBF** guarantees safety but has higher path deviation due to MPC horizon

---

## 12. DIAL-MPPI + Learned Model Benchmark

### 12.1 동기

Shield-DIAL-MPPI는 바람 외란 하에서도 CBF로 안전을 보장하지만, kinematic 모델이 실제 마찰/바람을 모르기 때문에 **모델 미스매치로 인한 오실레이션**이 발생합니다.

```
문제:  Shield-DIAL(Kinematic 3D) → 마찰/바람 모름 → 예측≠실제 → RMSE 1.71m
해결:  Shield-DIAL(Learned 5D)   → 마찰 추정     → 예측≈실제 → RMSE 개선
```

### 12.2 아키텍처

```
            Real World (5D DynamicKinematicAdapter)
            c_v=0.5, c_omega=0.3 + wind sin(0.8t)
                        |
         +------+-------+-------+-------+
         |      |               |       |
    [Vanilla] [Shield-DIAL]  [+EKF]   [+L1]
     3D kin    3D kin         5D EKF   5D L1
     [:3]      [:3]           full     full
                              ↓         ↓
                         파라미터 추정  외란 추정
                        (c_v, c_omega) (sigma_f)
```

**CBF Shield 호환성**: `_cbf_shield_batch()`가 `states[:,0:2]`(x,y)와 `states[:,2]`(θ)만 사용하므로 3D/5D 동일하게 동작합니다.

### 12.3 4종 컨트롤러

| # | 이름 | 모델 | state_dim | 안전 | 적응 |
|---|------|------|-----------|------|------|
| 1 | Vanilla | DiffDriveKinematic | 3D | 없음 | 없음 |
| 2 | Shield-DIAL | DiffDriveKinematic | 3D | CBF Shield | 없음 |
| 3 | Shield-DIAL+EKF | EKFAdaptiveDynamics | 5D | CBF Shield | 파라미터 추정 (매 스텝) |
| 4 | Shield-DIAL+L1 | L1AdaptiveDynamics | 5D | CBF Shield | 외란 추정 (매 스텝) |

### 12.4 3D vs 5D 처리

**3D 컨트롤러** (Vanilla, Shield-DIAL):
- `ctrl_state = state_5d[:3]` — 위치+heading만 사용
- `ref = (N+1, 3)` — StateTrackingCost + TerminalCost
- Q = [10, 10, 1], Qf = [20, 20, 2]

**5D 컨트롤러** (EKF, L1):
- `ctrl_state = state_5d` — 위치+heading+선속도+각속도 전부
- `ref = (N+1, 5)` — AngleAwareTrackingCost (heading wrapping)
- Q = [10, 10, 1, 0.1, 0.1], Qf = [20, 20, 2, 0.2, 0.2]
- `make_5d_reference()`: 3D ref → 5D ref (v_ref, ω_ref 추정)

### 12.5 시뮬레이션 루프

```python
for step in range(n_steps):
    state_5d = world.get_full_state()

    # 1. 적응 업데이트 (5D 모델만)
    if adapt and step > 0:
        model.update_step(prev_state_5d, prev_control, state_5d, dt)

    # 2. 컨트롤러 입력 (3D/5D 분기)
    ctrl_state = state_5d[:3] if dim == 3 else state_5d
    ref = ref_3d if dim == 3 else make_5d_reference(ref_3d, dt)
    control, info = controller.compute_control(ctrl_state, ref)

    # 3. 실제 세계 전파 + 바람 외란
    world.step(control, dt)
    world.state_5d[:3] += winds[step]  # 위치만 영향
```

### 12.6 L1 파라미터 튜닝

L1은 friction(상수) + wind(시변)을 단일 σ로 추정합니다. MPPI rollout(20 step) 동안 σ가 상수로 적용되므로, 시변 바람이 포함되면 예측이 악화됩니다.

**해결**: 위치 성분 A_m을 거의 0으로 → 바람(위치 영향)을 무시, 마찰(속도 영향)만 추정

```python
L1AdaptiveDynamics(
    adaptation_gain=50.0,     # Γ
    cutoff_freq=0.3,          # ω_c (저역통과 → 바람 노이즈 제거)
    am_gains=[-0.2, -0.2, -0.2, -10.0, -10.0],  # 위치 ≈ 0, 속도 빠르게
)
```

| 파라미터 | 역할 | 나쁜 값 → 결과 | 좋은 값 |
|---------|------|---------------|---------|
| am_gains[0:3] | 위치 예측기 | -2.0 → 바람 추적 → RMSE 2.16m | -0.2 → 바람 무시 |
| am_gains[3:5] | 속도 예측기 | -0.5 → 마찰 늦게 학습 | -10.0 → 빠른 추정 |
| cutoff_freq | 저역통과 | 10.0 → 바람 그대로 통과 | 0.3 → 바람 필터링 |

### 12.7 EKF vs L1 비교

| 측면 | EKF | L1 |
|------|-----|-----|
| 추정 대상 | 상수 파라미터 (c_v, c_ω) | 외란 벡터 σ (5D) |
| 바람 영향 | innovation 노이즈로 흡수 | σ에 혼입 |
| rollout 정확도 | 높음 (상수 파라미터 정확) | 중간 (시변 성분 남음) |
| 수렴 속도 | ~2-3초 | ~1-2초 |
| 적합 시나리오 | 구조적 미스매치 (마찰, 관성) | 상수 외란 (bias wind, 경사) |

### 12.8 벤치마크 결과

**조건**: K=512, N=20, dt=0.05, 25s, wind=0.6, 8 obstacles, margin=0.15m

```
실제 세계: c_v=0.5, c_omega=0.3  (컨트롤러는 모름)
공칭 모델: c_v=0.1, c_omega=0.1
```

| Method | RMSE | Violations | Min h(x) | Intervention | vs Shield-DIAL |
|--------|------|-----------|----------|-------------|---------------|
| Vanilla (3D) | 0.71m | 1 | -0.002 | — | — |
| **Shield-DIAL (3D)** | **1.71m** | **0** | 0.002 | 6.9% | baseline |
| **Shield-DIAL+EKF (5D)** | **0.69m** | **0** | 0.056 | 7.6% | **-60%** |
| **Shield-DIAL+L1 (5D)** | **0.89m** | **0** | 0.005 | 8.7% | **-48%** |

**핵심 발견**:
- **EKF**: Vanilla보다 낮은 RMSE(0.69m)를 달성하면서 0 violations (최고 성능)
- **L1**: Shield-DIAL 대비 48% RMSE 감소, 안전 유지
- **Shield-DIAL+EKF**가 "안전하면서 정확한" 최적 조합

### 12.9 적응 진단

벤치마크 종료 시 적응 모델의 추정 상태:

- **EKF**: ĉ_v ≈ 0.48 (실제 0.5), ĉ_ω ≈ 0.28 (실제 0.3) — 거의 정확한 추정
- **L1**: |σ_f| ≈ 0.19 — 마찰 보상 위주의 안정된 외란 추정

### 12.10 실행 방법

```bash
# Batch 실행 (플롯 저장)
PYTHONPATH=. python examples/comparison/learned_shield_dial_benchmark.py \
    --K 512 --duration 25 --wind 0.6

# 라이브 애니메이션
PYTHONPATH=. python examples/comparison/learned_shield_dial_benchmark.py \
    --live --K 512 --duration 25

# CLI 옵션
#   --K 512        샘플 수 (기본: 512)
#   --duration 25  시뮬레이션 시간 (기본: 25s)
#   --seed 42      랜덤 시드 (기본: 42)
#   --wind 0.6     바람 강도 (기본: 0.6)
#   --margin 0.15  CBF 안전 마진 (기본: 0.15m)
#   --live         실시간 애니메이션
```

### 12.11 시각화 (2x3 Grid)

| 패널 | 내용 |
|------|------|
| [0,0] XY Trajectory | 4종 궤적 + 8 장애물 + safety margin |
| [0,1] Tracking Error | 시간별 위치 오차 |
| [0,2] CBF Barrier | min h(x) — h<0이면 안전 위반 |
| [1,0] Linear Velocity | 선속도 (Shield 개입 시 감속) |
| [1,1] RMSE Bar | 4종 RMSE + violation 표시 |
| [1,2] Summary | RMSE/Safety/적응 진단 (EKF ĉ, L1 |σ_f|) |

---

## References

1. **Ames et al. (2019)** — "Control Barrier Functions: Theory and Applications" — CBF theory survey
2. **Zeng et al. (2021)** — "Safety-Critical MPC with Discrete-Time CBF" — Discrete-time CBF-MPC
3. **Thirugnanam et al. (2024)** — "Safety-Critical Control with Collision Cone CBFs" — C3BF
4. **Kim et al. (2026)** — "Dynamic Parabolic CBFs" (ICRA 2026) — DPCBF
5. **Gurriet et al. (2020)** — "Scalable Safety-Critical Control of Robotic Systems" — Optimal-Decay + Gatekeeper
6. **Rimon & Koditschek (1992)** — "Exact Robot Navigation Using Artificial Potential Functions"
7. **Yin et al. (2023)** — "Shield Model Predictive Path Integral" — Shield-MPPI
8. **Kondo et al. (2024)** — "SVG-MPPI" — Guide particle SVGD for MPPI
9. **Power et al. (2025)** — "DIAL-MPC: Diffusion-Inspired Annealing for MPC" (ICRA 2025 Best Paper Finalist) — DIAL-MPPI
