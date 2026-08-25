# 안전 제어 스택 코드 워크스루 — 함수 단위로 읽는 실제 구현

> **이 문서의 위치**: 이론은 [SAFETY_THEORY.md](../SAFETY_THEORY.md)와
> [03_CBF_FUNDAMENTALS.md](03_CBF_FUNDAMENTALS.md) /
> [04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md)가 담당합니다.
> 여기서는 **코드 그 자체** — 배열 shape, 브로드캐스팅, 분기 구조, info dict —
> 를 함수 단위로 해설합니다. 각 절은 "코드 발췌 → 해설 → 왜 이렇게 (트레이드오프)
> → 흔한 실수" 순서로 진행합니다.

## 목차

1. [안전 스택 아키텍처 — 3층위가 코드 어디에 끼어드는가](#1-안전-스택-아키텍처)
2. [CBF 비용의 배치 벡터화 — cbf_cost.py](#2-cbf-비용의-배치-벡터화)
3. [HOCBF 구현 — hocbf_cost.py](#3-hocbf-구현)
4. [Stochastic / RiskAware / Robust CBF](#4-stochastic--riskaware--robust-cbf)
5. [CLF-CBF-QP 솔버 — clf_cbf_qp.py](#5-clf-cbf-qp-솔버)
6. [필터 계열 비교 구현](#6-필터-계열-비교-구현)
7. [Gatekeeper / Backup — gatekeeper.py](#7-gatekeeper--backup)
8. [DualGuard의 SafetyValueFunction](#8-dualguard의-safetyvaluefunction)
9. [벤치마크에서 안전 메트릭 계산](#9-벤치마크에서-안전-메트릭-계산)
10. [실습 — 내 안전 비용을 하나 만들어보자](#10-실습--내-안전-비용을-하나-만들어보자)

---

## 1. 안전 스택 아키텍처

### 1.1 콜 그래프 — 3층위의 개입 지점

[04_ADVANCED_SAFETY.md §1](04_ADVANCED_SAFETY.md)의 "3가지 층위"가 코드에서
정확히 어디에 끼어드는지부터 봅니다. 기준은 vanilla MPPI의
`compute_control()` 파이프라인
([base_mppi.py:111-190](../../mppi_controller/controllers/mppi/base_mppi.py))입니다.

```
 Simulator loop (매 제어 주기)
 │
 ├─ controller.compute_control(state, ref) ──────────────────────┐
 │   │                                                            │
 │   │  1. noise = noise_sampler.sample(U, K, ...)   (K, N, nu)   │
 │   │  2. sampled_controls = U + noise → clip                    │
 │   │       │                                                    │
 │   │       │   ┌──────────────────────────────────────┐         │
 │   │       ├──▶│ [층위 1.5] Shield-MPPI               │         │
 │   │       │   │ _shielded_rollout(): rollout 루프    │         │
 │   │       │   │ 안에서 per-step CBF 클리핑            │         │
 │   │       │   └──────────────────────────────────────┘         │
 │   │  3. trajectories = rollout(state, controls) (K, N+1, nx)   │
 │   │  4. costs = cost_function.compute_cost(...)   (K,)         │
 │   │       │                                                    │
 │   │       │   ┌──────────────────────────────────────┐         │
 │   │       └──▶│ [층위 1] CBF 비용                     │         │
 │   │           │ CompositeMPPICost 안의 한 항으로      │         │
 │   │           │ ControlBarrierCost / HOCBFCost /     │         │
 │   │           │ StochasticCBFCost / RobustCBFCost 등 │         │
 │   │           └──────────────────────────────────────┘         │
 │   │  5. weights = _compute_weights(costs, λ)                   │
 │   │  6. U ← U + Σ w_k ε_k → clip → roll                        │
 │   │  7. control = U[0]                                         │
 │   │       │                                                    │
 │   │       │   ┌──────────────────────────────────────┐         │
 │   │       └──▶│ [층위 2] 안전 필터 (반환 직전)          │         │
 │   │           │ safety_filter.filter_control(state,  │         │
 │   │           │   control, u_min, u_max)             │         │
 │   │           │ → CBFSafetyFilter / HOCBFFilter /    │         │
 │   │           │   OptimalDecayCBFSafetyFilter        │         │
 │   │           └──────────────────────────────────────┘         │
 │   └─ return control, info ─────────────────────────────────────┘
 │
 ├─ [층위 3] u_safe, ginfo = gatekeeper.filter(state, control)
 │            (Simulator에 넘기기 직전 — 백업 궤적 검증 게이트)
 │
 └─ state = model.step(state, u_safe, dt)
```

층위 2의 실제 개입 코드가 [cbf_mppi.py:116-128](../../mppi_controller/controllers/mppi/cbf_mppi.py)입니다.
`super().compute_control()`이 끝난 **반환 직전**에 끼어듭니다:

```python
# cbf_mppi.py:116-128 (CBFMPPIController.compute_control)
# Layer 1: MPPI 제어 계산 (CBF 비용 포함)
control, info = super().compute_control(state, reference_trajectory)
...
# Layer 2: 안전 필터 (optional)
filter_info = {"filtered": False, "correction_norm": 0.0}
if self.safety_filter is not None:
    control, filter_info = self.safety_filter.filter_control(
        state, control, self.u_min, self.u_max
    )
```

층위 1의 개입 코드는 생성자에 있습니다 — CBF 비용을 **CompositeMPPICost의
한 항으로 밀어 넣기만** 하면 됩니다
([cbf_mppi.py:66-72](../../mppi_controller/controllers/mppi/cbf_mppi.py)):

```python
# cbf_mppi.py:66-72
composite_cost = CompositeMPPICost([
    StateTrackingCost(params.Q),
    TerminalCost(params.Qf),
    ControlEffortCost(params.R),
    self.cbf_cost,          # ← 층위 1: 비용 항 하나 추가가 전부
])
```

층위 3은 컨트롤러 밖입니다 — [gatekeeper.py:22-27](../../mppi_controller/controllers/mppi/gatekeeper.py)의
사용 예시 그대로, Simulator 호출 직전에 씁니다:

```python
# gatekeeper.py docstring (사용 패턴)
u_safe, info = gatekeeper.filter(state, u_mppi)
```

### 1.2 공통 인터페이스 — 3가지 시그니처

| 층위 | 인터페이스 | 입력 | 출력 | 파일 |
|---|---|---|---|---|
| 1 (비용) | `CostFunction.compute_cost(trajectories, controls, reference_trajectory)` | `(K,N+1,nx)`, `(K,N,nu)`, `(N+1,nx)` | `(K,)` | [cost_functions.py:12-16](../../mppi_controller/controllers/mppi/cost_functions.py) |
| 2 (필터) | `filter_control(state, u_nom, u_min, u_max)` | `(nx,)`, `(nu,)` | `(u_safe, info)` | cbf_safety_filter.py, hocbf_cost.py, optimal_decay_cbf_filter.py |
| 3 (게이트) | `Gatekeeper.filter(state, u_mppi)` | `(nx,)`, `(nu,)` | `(u_safe, info)` | gatekeeper.py |

핵심 관찰 세 가지:

- **층위 1은 배치, 층위 2/3은 단일**. `compute_cost`는 K개 샘플 전체를 한 번에
  받으므로 반드시 벡터화해야 하고 (2절), `filter_control`/`filter`는 최종
  제어 1개만 받으므로 scipy SLSQP 같은 스칼라 솔버도 허용됩니다.
- **층위 2와 3은 시그니처가 거의 같지만 의미가 다릅니다**. 필터는 "u를 최소
  수정"하고, 게이트는 "u를 통째로 채택/기각"합니다 (binary decision).
- 필터 계열은 duck typing입니다 — `CBFSafetyFilter`, `HOCBFFilter`,
  `OptimalDecayCBFSafetyFilter` 모두 같은 `filter_control` 시그니처를
  제공하므로 `CBFMPPIController.safety_filter`에 갈아 끼울 수 있고, 벤치마크의
  `PostFilterController` 래퍼
  ([cbfkit_inspired_benchmark.py:239-250](../../examples/comparison/cbfkit_inspired_benchmark.py))도
  같은 이유로 어떤 필터든 감쌀 수 있습니다:

```python
# cbfkit_inspired_benchmark.py:239-250 (PostFilterController)
def compute_control(self, state, reference_trajectory):
    u, info = self.base.compute_control(state, reference_trajectory)
    u_safe, finfo = self.filter.filter_control(state, u)
    info["hocbf_filter"] = finfo
    return u_safe, info
```

> **왜 이렇게**: 층위마다 인터페이스를 분리하면 어떤 조합도 조립 가능합니다
> (비용 + 필터 + 게이트 3중 스택도 코드 수정 없이 구성). 이 조립성이
> [SAFETY_THEORY.md §21](../SAFETY_THEORY.md)의 "안전 기법 선택 가이드"를
> 실험으로 검증할 수 있게 하는 토대입니다.

> **흔한 실수**: 층위 2 필터를 쓸 때 `info` dict를 버리는 것. `filtered`,
> `correction_norm`이 없으면 필터가 얼마나 자주/세게 개입했는지 모니터링할 수
> 없고, "MPPI가 잘 되는 줄 알았는데 사실 필터가 다 하고 있었다"는 상황을
> 놓칩니다. 필터 개입률(filter_rate)이 높으면 층위 1 비용의 가중치/α 튜닝이
> 잘못됐다는 신호입니다.

---

## 2. CBF 비용의 배치 벡터화

파일: [cbf_cost.py](../../mppi_controller/controllers/mppi/cbf_cost.py) —
이론은 [SAFETY_THEORY.md §2](../SAFETY_THEORY.md),
[03_CBF_FUNDAMENTALS.md §9](03_CBF_FUNDAMENTALS.md).

### 2.1 compute_cost — shape 추적

```python
# cbf_cost.py:65-88 (ControlBarrierCost.compute_cost)
K = trajectories.shape[0]
costs = np.zeros(K)

positions = trajectories[:, :, :2]  # (K, N+1, 2)

for obs_x, obs_y, obs_r in self.obstacles:
    effective_r = obs_r + self.safety_margin

    # 거리 제곱 (K, N+1)
    dx = positions[:, :, 0] - obs_x
    dy = positions[:, :, 1] - obs_y
    dist_sq = dx**2 + dy**2

    # Barrier value: h(x) = ||p - p_obs||^2 - r_eff^2
    h = dist_sq - effective_r**2  # (K, N+1)

    # Discrete CBF 조건: h(x_{t+1}) - (1-alpha)*h(x_t) >= 0
    # 위반 = max(0, -[h(x_{t+1}) - (1-alpha)*h(x_t)])
    cbf_condition = h[:, 1:] - (1.0 - self.cbf_alpha) * h[:, :-1]  # (K, N)
    violation = np.maximum(0.0, -cbf_condition)

    costs += self.cbf_weight * np.sum(violation, axis=1)
```

shape 흐름을 한 줄씩 따라가면:

| 단계 | 표현식 | shape | 설명 |
|---|---|---|---|
| 위치 추출 | `trajectories[:, :, :2]` | `(K, N+1, 2)` | 상태 앞 2차원(x,y)만 사용 — 3D/5D 모델 공용 |
| 장애물별 offset | `positions[:,:,0] - obs_x` | `(K, N+1)` | 스칼라 브로드캐스팅 |
| barrier | `dist_sq - effective_r**2` | `(K, N+1)` | 전 샘플×전 시점의 h를 한 번에 |
| CBF 조건 | `h[:, 1:] - (1-α)·h[:, :-1]` | `(K, N)` | **시프트 슬라이싱**으로 t+1 vs t 쌍 생성 |
| 위반 | `np.maximum(0.0, -cbf_condition)` | `(K, N)` | hinge — 만족 구간은 0 |
| 집계 | `np.sum(violation, axis=1)` | `(K,)` | 시간축만 합산, 샘플축 보존 |

이산 CBF 조건 `h_{t+1} - (1-α)h_t ≥ 0`의 "diff"는 `np.diff`가 아니라
**슬라이싱 쌍 `h[:, 1:]`와 `h[:, :-1]`**로 구현되어 있습니다. `(1-α)` 계수가
붙기 때문에 순수 차분이 아니어서 슬라이싱이 더 자연스럽습니다 (α=0이면
`np.diff(h, axis=1)`와 동일).

주의할 점: 장애물 루프는 남아 있습니다. 이건 의도적입니다 — 장애물 수
`n_obs`는 보통 3~8개로 K=512~1024, N=30에 비해 미미해서, 루프를
`(K, N+1, n_obs)` 3D 텐서로 바꿔봐야 메모리만 더 쓰고 이득이 거의 없습니다.
(`get_barrier_info`는 시각화용으로 `(num_obs, K, N+1)` 스택을 실제로 만듭니다
— [cbf_cost.py:126](../../mppi_controller/controllers/mppi/cbf_cost.py).)

### 2.2 왜 K 루프를 돌리면 100배 느린가

벡터화하지 않은 naive 구현:

```python
# 안티패턴 — 이렇게 쓰지 마세요
for k in range(K):                 # 512회
    for t in range(N):             # 30회
        for obs in self.obstacles: # 3회
            costs[k] += weight * max(0.0, -(h(k,t+1) - (1-alpha)*h(k,t)))
```

K=512, N=30, 장애물 3개면 **파이썬 레벨 반복 46,080회 + 함수 호출 92,160회**
— 매 호출마다 인터프리터 오버헤드(바이트코드 디스패치, 임시 객체 생성)가
붙습니다. 벡터화 버전은 같은 산술을 numpy의 C 루프로 내려보내 장애물당
**대여섯 개의 배열 연산**으로 끝냅니다. 이 repo의 성능 기준 "계산 시간
< 100ms (K=1024, N=30)"은 비용 함수 하나가 수 ms 안에 끝나야 달성됩니다 —
K 루프 하나가 파이프라인 전체 예산을 태워버립니다.

> **왜 이렇게 (트레이드오프)**: h를 유클리드 거리 `||p-p_o|| - r`이 아닌
> **거리 제곱** `||p-p_o||² - r²`으로 정의한 것도 벡터화 친화적 선택입니다.
> `sqrt` 호출이 없고, gradient가 `[2dx, 2dy]`로 다항식이라 이후 HOCBF/Robust
> CBF에서 해석적으로 재사용됩니다 (3, 4절). 대가는 h의 스케일이 거리에
> 비례하지 않고 제곱으로 커진다는 것 — `cbf_weight` 튜닝이 장애물 반경에
> 민감해집니다.

> **흔한 실수 모음**
> 1. `h[:, 1:] - h[:, :-1]`의 부호를 뒤집어 `h[:, :-1] - h[:, 1:]`로 쓰면
>    "멀어지는 것"을 벌점하는 반대 조건이 됩니다. 테스트: 장애물에서 곧장
>    멀어지는 궤적의 비용이 0인지 확인.
> 2. `violation = -cbf_condition` (hinge 없이). 조건을 크게 만족하는 궤적이
>    **음수 비용(보상)**을 받아 장애물 근처를 맴돌게 됩니다.
> 3. `np.sum(violation)` (axis 없이) — `(K,)`가 아닌 스칼라가 되어 모든 샘플에
>    같은 비용이 더해지고, CBF가 가중치에 아무 영향을 못 줍니다. 조용히
>    틀리는 버그라 특히 위험합니다.

---

## 3. HOCBF 구현

파일: [hocbf_cost.py](../../mppi_controller/controllers/mppi/hocbf_cost.py) —
이론은 [SAFETY_THEORY.md §16](../SAFETY_THEORY.md),
[03_CBF_FUNDAMENTALS.md §7](03_CBF_FUNDAMENTALS.md).

### 3.1 HOCBFCost — 유한 차분 캐스케이드

연속시간 exponential HOCBF의 캐스케이드 ψ0 → ψ1 → C를, 이산 궤적 위에서
유한 차분으로 근사합니다:

```python
# hocbf_cost.py:197-220 (HOCBFCost.compute_cost 핵심)
for obs_x, obs_y, obs_r in self.obstacles:
    effective_r = obs_r + self.safety_margin

    dx = positions[:, :, 0] - obs_x
    dy = positions[:, :, 1] - obs_y
    h = dx**2 + dy**2 - effective_r**2  # ψ0: (K, N+1)

    # 1차 cascade: ψ1_t = (ψ0_{t+1} - ψ0_t)/dt + λ1·ψ0_t
    psi1 = (h[:, 1:] - h[:, :-1]) / self.dt + self.lambda1 * h[:, :-1]

    if effective_rd == 1:
        constraint = psi1  # (K, N)
    else:
        # 2차 cascade: C_t = (ψ1_{t+1} - ψ1_t)/dt + λ2·ψ1_t
        constraint = (
            (psi1[:, 1:] - psi1[:, :-1]) / self.dt
            + self.lambda2 * psi1[:, :-1]
        )  # (K, N-1)

    violation = np.maximum(0.0, -constraint)
    if self.penalty == "squared":
        violation = violation**2

    costs += self.weight * np.sum(violation, axis=1)
```

shape가 캐스케이드를 따라 한 칸씩 줄어드는 것에 주목하세요:
`h (K,N+1)` → `psi1 (K,N)` → `constraint (K,N-1)`. 차분 한 번에 시점 하나를
소비하기 때문입니다. 그래서 rd=2는 **최소 3개 시점**이 필요하고, 코드는
`if effective_rd == 2 and n_steps < 3: effective_rd = 1`로 rd=1에 폴백해
가드합니다 ([hocbf_cost.py:192-195](../../mppi_controller/controllers/mppi/hocbf_cost.py)).

**rd=1 축약 분기**가 이 클래스의 정합성 검증 포인트입니다. 모듈 docstring
([hocbf_cost.py:21-25](../../mppi_controller/controllers/mppi/hocbf_cost.py))이
증명하듯, rd=1이면 `C_t = (1/dt)·[h_{t+1} - (1 - λ1·dt)·h_t]`이므로
`penalty="linear"`, `λ1 = α/dt`, `weight' = weight·dt`로 두면 2절의
`ControlBarrierCost(alpha=α, weight=weight)`와 **수치까지 정확히 일치**합니다.
새 안전 비용을 만들 때 이렇게 "기존 구현으로 축약되는 파라미터 조합"을
테스트로 박아두는 것이 이 repo의 패턴입니다 (4절의 σ=0, w_max=0 축약도 동일).

penalty 모드는 3가지: `squared`(기본, 위반 크기에 이차 — 경계 근처에서
부드러운 gradient), `linear`(ControlBarrierCost 호환), 그리고
`use_hard_rejection`(h<0인 궤적에 `rejection_cost=1e6` 일괄 추가 — 이진
거부, [hocbf_cost.py:222-226](../../mppi_controller/controllers/mppi/hocbf_cost.py)).
하드 거부의 `violated |= np.any(h < 0, axis=1)`는 장애물 루프 **바깥**에서
누적되는 `(K,)` bool 마스크입니다 — 어느 한 장애물이라도 침입하면 거부됩니다.

### 3.2 detect_relative_degree — g(x) 열을 유한 차분으로 뽑기

상대 차수를 수동 지정하는 대신, "∂h/∂x·g(x)가 0인가"를 샘플링으로
판정합니다:

```python
# hocbf_cost.py:82-103 (detect_relative_degree 핵심)
for _ in range(n_samples):
    x = generator.normal(size=nx)
    f0 = model.forward_dynamics(x, np.zeros(nu))   # drift f(x) = f(x, 0)
    grad = h_grad_fn(x)
    for j in range(nu):
        uj = np.zeros(nu)
        uj[j] = delta                               # delta = 1e-3
        g_col = (model.forward_dynamics(x, uj) - f0) / delta   # g(x)의 j열
        total_authority += abs(float(grad @ g_col))
...
return 1 if total_authority > tol else 2
```

트릭은 제어-어파인 가정 `ẋ = f(x) + g(x)u`에 있습니다.
`forward_dynamics(x, δ·e_j) - forward_dynamics(x, 0) = g(x)·δ·e_j`이므로
**제어에 대한 forward 차분이 g(x)의 j번째 열을 정확히** 줍니다 (어파인이면
차분 오차 0). 그 열과 ∂h/∂x의 내적 절대값을 20개 무작위 상태에서 누적해,
합이 `tol=1e-8` 미만이면 "제어가 ḣ에 못 들어온다" → rd ≥ 2로 판정합니다.

> **왜 이렇게**: 심볼릭 미분(cbfkit은 JAX autodiff)을 쓰지 않고도 어떤
> `RobotModel`에나 적용되는 범용 판정입니다. 대가는 확률적 판정이라는 것 —
> 특정 상태에서만 rd가 떨어지는(singular) 시스템은 오판할 수 있습니다.
> 그래서 무작위 상태 20개를 쓰고, NaN 발생 시 명시적으로 raise합니다
> ([hocbf_cost.py:97-101](../../mppi_controller/controllers/mppi/hocbf_cost.py)).

> **흔한 실수**: differential drive 기구학에서 h=||p||²는 언뜻 rd=2로 보이지만
> 실제로는 rd=1입니다 — `∂h/∂x·g = [2x·cosθ + 2y·sinθ, 0]`으로 v가 ḣ에 직접
> 들어옵니다. rd=2가 되는 것은 제어가 **가속도**인 5D 동역학 모델입니다.
> 자동 검출(`relative_degree=None`)이 이 실수를 막아줍니다.

### 3.3 HOCBFFilter.filter_control — 해석적 최소 노름 보정

QP 솔버 없이 closed-form으로 단일 제약을 보정하는 필터입니다.
제약을 `aᵀu + b ≥ 0` 형태로 만든 뒤:

```python
# hocbf_cost.py:449-471 (filter_control 핵심 루프)
for _ in range(self.n_passes):
    # 가장 위반이 큰 제약 탐색
    worst_val, worst_a, worst_b = float("inf"), None, None
    for obs in self.obstacles:
        a, b = self._constraint_terms(state, obs)
        val = float(a @ u + b)
        if val < worst_val:
            worst_val, worst_a, worst_b = val, a, b

    if worst_val >= 0.0:
        break                                    # 모든 제약 만족 → 종료

    # closed-form 최소 노름 보정
    correction = max(0.0, -(worst_b + float(worst_a @ u)))
    u = u + correction * worst_a / (float(worst_a @ worst_a) + self.eps)

    if lo is not None and hi is not None:
        u = np.clip(u, lo, hi)                   # 보정 → 클리핑 순서
```

보정 공식 `u* = u + max(0, -(b + aᵀu))·a/(aᵀa + eps)`는 초평면
`aᵀu + b = 0`으로의 유클리드 투영입니다 (유도는
[03_CBF_FUNDAMENTALS.md §5](03_CBF_FUNDAMENTALS.md)). `eps=1e-8`은 `a ≈ 0`
(제약이 제어에 둔감한 상태 — 예: 장애물과 정확히 접선 방향)에서의 0-나눗셈
가드입니다.

`(a, b)` 자체는 rd에 따라 분기합니다
([hocbf_cost.py:392-408](../../mppi_controller/controllers/mppi/hocbf_cost.py)):

```python
# hocbf_cost.py:399-406
if self.relative_degree == 1:
    grad_h = self._h_grad(x, obs)
    a = grad_h @ g
    b = float(grad_h @ f) + self.lambda1 * self._h(x, obs)
else:
    grad_psi1 = self._psi1_grad(x, obs)        # ∇ψ1: 상태 central difference
    a = grad_psi1 @ g
    b = float(grad_psi1 @ f) + self.lambda2 * self._psi1(x, obs)
```

rd=2에서 ∇ψ1은 상태에 대한 **central difference**
(`_psi1_grad`, [hocbf_cost.py:380-390](../../mppi_controller/controllers/mppi/hocbf_cost.py))로
계산합니다 — ψ1이 이미 f(x)를 포함한 합성 함수라 해석적 미분이 번거롭기
때문입니다. g(x)는 3.2절과 같은 제어 forward 차분(`_g`)입니다.

> **왜 이렇게 (다중 장애물 처리)**: SLSQP로 m개 제약을 동시에 푸는 대신,
> "최악 제약 하나 투영 → 재평가"를 `n_passes=3`회 반복합니다. 제약이 대부분
> 한 개만 활성인 실전에서는 1-pass에 끝나고(조기 break), 두 장애물 사이를
> 지날 때만 2~3회 돕니다. 대가: 세 개 이상의 제약이 동시에 활성이고 서로
> 충돌하면 수렴 보장이 없습니다 — 그 경우는 5절의 SLSQP 폴백이 있는
> CLF-CBF-QP를 쓰는 것이 맞습니다.

> **흔한 실수**: 클리핑을 투영 **앞**에 두는 것. 이 코드는 "투영 → 클리핑"
> 순서인데, 클리핑이 투영 결과를 다시 제약 위반 쪽으로 되돌릴 수 있습니다.
> 그래서 루프가 다음 pass에서 **다시 제약을 평가**하는 구조가 중요합니다 —
> 한 번 투영하고 끝내면 클리핑 때문에 위반이 남을 수 있습니다.

---

## 4. Stochastic / RiskAware / Robust CBF

파일: [stochastic_cbf.py](../../mppi_controller/controllers/mppi/stochastic_cbf.py),
[robust_cbf_margin.py](../../mppi_controller/controllers/mppi/robust_cbf_margin.py) —
이론은 [SAFETY_THEORY.md §17~19](../SAFETY_THEORY.md),
[04_ADVANCED_SAFETY.md §4](04_ADVANCED_SAFETY.md).

### 4.1 StochasticCBFCost — Itô 항이 코드 한 줄인 이유

Itô 보정 `0.5·Tr[σᵀ(∇²h)σ]`는 일반적으로 Hessian이 필요하지만, 원형 barrier
`h = ||p-p_o||² - R²`의 위치 블록 Hessian은 **상수 2I**입니다. 그래서 생성자
한 줄로 끝납니다:

```python
# stochastic_cbf.py:101-104 (__init__)
# Itô correction: 0.5·Tr[σᵀ(∇²h)σ], ∇²h = 2I (위치 블록)
#   = 0.5 · Σ_i 2·σ_pos_i² = Σ_i σ_pos_i²
sigma_pos = self.sigma_process[:2]
self.ito_correction = float(np.sum(sigma_pos**2))

# stochastic_cbf.py:142-148 (compute_cost — 상태 무관 스칼라라 그냥 더함)
# C_t = Δh/dt + α·h_t + ito - β ≥ 0
condition = (
    (h[:, 1:] - h[:, :-1]) / self.dt
    + self.alpha * h[:, :-1]
    + self.ito_correction
    - self.beta
)  # (K, N)
```

autodiff도, 상태 의존 계산도 없습니다 — Hessian이 상수라서 얻는 공짜입니다.

**부호에 주의**: 볼록 barrier에서 ∇²h ⪰ 0이므로 Itô 항은 **양수 = 조건 완화**
입니다 (등방성 노이즈는 기대 제곱 거리를 늘림 — docstring
[stochastic_cbf.py:21-24](../../mppi_controller/controllers/mppi/stochastic_cbf.py)가
명시). "노이즈가 있으니 더 보수적이어야지"라는 직관과 반대죠. 보수성은
`beta > 0` buffer나 아래 RiskAwareCBFCost가 담당합니다.

**σ=0, β=0 축약**이 정확한 이유: `ito_correction = 0`이면 조건이
`Δh/dt + α·h_t ≥ 0`, 즉 dt를 곱하면 `h_{t+1} - (1 - α·dt)·h_t ≥ 0`. 따라서
`StochasticCBFCost(alpha=a, weight=w·dt) ≡ ControlBarrierCost(cbf_alpha=a·dt,
cbf_weight=w)` — **부동소수점까지 동일**합니다 (같은 hinge, 같은 슬라이싱).
벤치마크의 파라미터 주석
([cbfkit_inspired_benchmark.py:318](../../examples/comparison/cbfkit_inspired_benchmark.py)
"`alpha=6.0, weight=50 ≡ ControlBarrierCost(alpha=0.3, weight=1000) @ σ=0`")이
이 등가성을 실제로 사용합니다: 0.3 = 6.0·0.05, 1000 = 50/0.05.

### 4.2 RiskAwareCBFCost — erfinv 마진 스케줄

`P(min_t h < 0) ≤ ρ`를 위한 시간 의존 마진
`margin(t) = √(2t)·η·erfinv(1-2ρ)`의 구현:

```python
# stochastic_cbf.py:226 (__init__) — 생성자에서 한 번만 계산
self.erf_factor = float(erfinv(1.0 - 2.0 * rho))

# stochastic_cbf.py:256-263 (get_margin)
if eta is None:
    eta = self._last_eta if self._last_eta is not None else self._eta_default
t_arr = np.asarray(t, dtype=float)
margin = np.sqrt(2.0 * t_arr) * eta * self.erf_factor
```

`scipy.special.erfinv`는 매 스텝 호출하기엔 아까운 특수함수라 생성자에서
캐싱합니다. ρ=0.5면 erfinv(0)=0으로 마진이 정확히 0 — vanilla와 동일해지는
sanity check 포인트입니다.

η(||∇h·σ|| 상한)의 근사가 실용적 타협의 핵심입니다:

```python
# stochastic_cbf.py:302-315 (compute_cost 내부)
# η: ||∇h·σ_pos|| = sqrt((2dx·σx)² + (2dy·σy)²) 의 배치 상한
if self.grad_bound is not None:
    eta = self.grad_bound * sigma_norm
else:
    grad_sigma_norm = np.sqrt(
        (2.0 * dx * self.sigma_pos[0]) ** 2
        + (2.0 * dy * self.sigma_pos[1]) ** 2
    )  # (K, N+1)
    eta = float(np.max(grad_sigma_norm)) if grad_sigma_norm.size else 0.0
...
margin = sqrt_2t * eta * self.erf_factor  # (N+1,)
violation = np.maximum(0.0, margin[np.newaxis, :] - h)  # (K, N+1) 브로드캐스트
```

이론상 η는 안전 집합 전체에서의 sup이지만, 코드는 **현재 샘플 배치가 방문한
영역의 max**로 대체합니다 (배치가 방문한 영역에 대해서는 유효한 상한 —
전역 상한이 필요하면 `grad_bound`로 직접 지정). 위반 계산에서
`margin[np.newaxis, :]`로 `(N+1,)` 마진을 `(K, N+1)` h에 브로드캐스트하는
것도 확인하세요 — 마진은 시간의 함수일 뿐 샘플과 무관합니다.

> **흔한 실수**: 이 비용은 CBF 조건(차분식)이 아니라 **상태 제약**
> `h(x_t) ≥ margin(t)`을 직접 벌점합니다. 2절과 달리 `h[:, 1:]` 시프트가
> 없다는 점을 놓치고 같은 패턴으로 리팩터링하면 의미가 달라집니다.

### 4.3 RobustCBFCost — ∇h·M 노름 마진

유계 외란 `||w|| ≤ w_max`에 대한 최악 마진 `dt·||∇h·M||·w_max`:

```python
# robust_cbf_margin.py:104-109 (_robust_term)
gM = grad_pos @ self.M_pos  # (..., m)
if self.norm == "two":
    rob = np.linalg.norm(gM, axis=-1)
else:  # sup-norm 외란 → 1-노름 쌍대
    rob = np.sum(np.abs(gM), axis=-1)
return rob * self.w_max

# robust_cbf_margin.py:163-172 (compute_cost 내부)
# ∇h 위치 성분 (t = 0..N-1 스텝에서 평가)
grad_pos = np.stack(
    [2.0 * dx[:, :-1], 2.0 * dy[:, :-1]], axis=-1
)  # (K, N, 2)
robust_margin = self.dt * self._robust_term(grad_pos)  # (K, N)

# 강화된 이산 CBF 조건
condition = (
    h[:, 1:] - (1.0 - self.alpha) * h[:, :-1] - robust_margin
)  # (K, N)
```

포인트 세 가지:

1. **∇h가 해석적**: 거리 제곱 barrier 덕분에 `∇h_pos = [2dx, 2dy]`를
   `np.stack`으로 조립합니다. gradient를 `[:, :-1]` (t 스텝)에서 평가하는
   것은 "외란이 스텝 t 동안 h를 얼마나 깎을 수 있나"의 선형화 기준점이
   t이기 때문입니다.
2. **노름 쌍대성**: `||w||₂ ≤ w_max`면 최악 감소는 `||∇h·M||₂·w_max`,
   `||w||∞ ≤ w_max`면 쌍대 노름인 1-노름 `Σ|(∇h·M)_j|·w_max`. 외란 모델과
   노름 선택이 짝이 맞아야 합니다.
3. **w_max=0 축약이 exact**: `rob * self.w_max = 0`이므로
   `robust_margin`이 0 배열이 되고, condition이 2절의 `ControlBarrierCost`와
   **문자 그대로 같은 식**이 됩니다. 근사 없는 축약 — 회귀 테스트의 기준점.

> **왜 이렇게**: 세 클래스(Stochastic/RiskAware/Robust) 모두 vanilla 조건에
> "마진 항 하나 추가"라는 동일한 구조입니다. 불확실성 모델(확산/위험
> 예산/유계 외란)만 다르고 벡터화 골격은 2절 그대로 — 새 변형을 만들 때
> 이 골격을 복사하는 것이 이 repo의 관례입니다 (10절 실습에서 직접 합니다).

---

## 5. CLF-CBF-QP 솔버

파일: [clf_cbf_qp.py](../../mppi_controller/controllers/mppi/clf_cbf_qp.py) —
이론은 [SAFETY_THEORY.md §20](../SAFETY_THEORY.md),
[03_CBF_FUNDAMENTALS.md §5~6](03_CBF_FUNDAMENTALS.md).

### 5.1 solve()의 3-경로 구조

`CBFCLFQPSolver.solve()`
([clf_cbf_qp.py:105-320](../../mppi_controller/controllers/mppi/clf_cbf_qp.py))는
외부 QP 라이브러리(OSQP 등) 없이 3단 폴백으로 풉니다:

```
경로 (a) analytic:            CLF-tradeoff 닫힌형 → 클리핑 → 전 제약 검사 통과 시 반환
경로 (b) analytic_projection: 단일 CBF 위반 → 등식 투영 닫힌형 → 검사 통과 시 반환
경로 (c) slsqp:               다중 제약/경계 활성 → scipy SLSQP (+ best-effort 폴백)
```

**경로 (a)** — soft CLF의 닫힌형
([clf_cbf_qp.py:189-213](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)):

```python
# clf_cbf_qp.py:192-213
s = float(a_clf @ u_nom - b_clf)
if s > 0.0:
    Pinv_a = np.linalg.solve(P, a_clf)
    q = float(a_clf @ Pinv_a)
    u_cand = u_nom - (self.lambda_clf * s / (1.0 + self.lambda_clf * q)) * Pinv_a
    delta_cand = s / (1.0 + self.lambda_clf * q)
...
u_c = _clip(u_cand)
clipped = not np.allclose(u_c, u_cand, atol=1e-12)
P_is_diag = np.allclose(P, np.diag(np.diag(P)))
# 클리핑이 발생하면: P가 대각일 때만 클리핑이 정확한 투영.
# CLF가 활성이었다면 클리핑 후 tradeoff가 달라지므로 SLSQP로.
clip_ok = (not clipped) or (P_is_diag and delta_cand == 0.0)
clf_ok_at_uc = (not have_clf) or (float(a_clf @ u_c - b_clf) <= delta_cand + self.feas_tol)

if clip_ok and clf_ok_at_uc and _cbf_ok(u_c):
    return _finish(u_c, True, "analytic", delta_cand)
```

slack의 최적값 `δ*(u) = max(0, a·u - b)`를 목적함수에 대입하면 1차원 축소
문제가 되고, 그 해가 위 닫힌형입니다 (유도:
[03_CBF_FUNDAMENTALS.md §6](03_CBF_FUNDAMENTALS.md)). 눈여겨볼 것은 **닫힌형이
유효한 조건을 코드가 명시적으로 검사**한다는 점: 클리핑이 일어났는데 P가
비대각이거나 CLF가 활성이었다면 "클리핑 = 박스로의 투영" 등식이 깨지므로
fast path를 포기하고 SLSQP로 넘어갑니다. 이 검사 없이 반환하면 조용히
부정확한 해를 내놓게 됩니다.

**경로 (b)** — 단일 CBF 활성 등식 투영
([clf_cbf_qp.py:215-236](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)):

```python
# clf_cbf_qp.py:219-236
resid_c = _cbf_resid(u_c)
violated = np.where(resid_c < -self.feas_tol * (1.0 + np.abs(b_cbf)))[0]
if violated.size == 1:
    g, c = A_cbf[int(violated[0])], float(b_cbf[int(violated[0])])
    Pinv_g = np.linalg.solve(P, g)
    denom = float(g @ Pinv_g)
    if denom > 1e-12:
        u_proj = u_nom + Pinv_g * (c - float(g @ u_nom)) / denom
        if in_bounds and clf_inactive and _cbf_ok(u_proj):
            return _finish(u_proj, True, "analytic_projection", 0.0)
```

`u* = u_nom + P⁻¹gᵀ(c - g·u_nom)/(g·P⁻¹gᵀ)`은 P-노름 최소 수정으로 초평면
`g·u = c`에 붙이는 투영입니다 (3.3절 HOCBFFilter 공식의 P-가중 일반화;
P=I면 동일식). 반환 전에 **세 가지를 전부 재검사**합니다: (i) 경계 내부인가,
(ii) CLF가 비활성으로 남는가, (iii) 다른 CBF까지 만족하는가. 하나라도
실패하면 활성 집합이 2개 이상이라는 뜻 → SLSQP.

`feas_tol * (1.0 + np.abs(b_cbf))`의 **상대 허용 오차**도 포인트입니다 —
b가 수천 스케일(거리 제곱 barrier)일 때 절대 tol 1e-6은 무의미하게 엄격해서
불필요한 SLSQP 폴백을 유발합니다.

**경로 (c)** — SLSQP 폴백은 결정 변수를 `z = [u, δ]`로 확장하고
([clf_cbf_qp.py:241-296](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)),
실패 시에도 best-effort를 반환합니다:

```python
# clf_cbf_qp.py:309-318
feasible = success and _cbf_ok(u_opt)
if not feasible:
    # best-effort: SLSQP 결과 vs 클리핑된 명목 제어 중 위반이 적은 쪽
    if have_cbf:
        viol_opt = float(max(0.0, -(_cbf_resid(u_opt)).min()))
        viol_nom = float(max(0.0, -(_cbf_resid(u0)).min()))
        if viol_nom < viol_opt:
            u_opt = u0
            delta = _clf_slack(u0)
    return _finish(u_opt, False, "slsqp_infeasible", delta)
```

> **왜 이렇게 (트레이드오프)**: 해석적 fast path는 실전에서 대부분의 스텝을
> 커버합니다 (제약 비활성이거나 하나만 활성). SLSQP는 정확하지만 10~100배
> 느리므로, "흔한 경우는 닫힌형, 드문 경우만 솔버"가 10Hz 실시간성의
> 핵심입니다. `use_analytic=False`로 끄면 항상 SLSQP — fast path 검증용
> 스위치입니다.

### 5.2 look-ahead 맵 M(θ) — 기구학의 가역화

unicycle은 `ṗ = [v·cosθ, v·sinθ]`라 위치 속도가 v 하나로만 결정됩니다
(ω가 위치에 즉시 안 들어옴 = 위치 barrier의 rd 문제). look-ahead 포인트가
이를 해결합니다:

```python
# clf_cbf_qp.py:442-449 (_lookahead_map)
th = float(state[2])
d = self.params.lookahead_d
c, s = np.cos(th), np.sin(th)
p_tilde = state[:2] + d * np.array([c, s])
M = np.array([[c, -d * s], [s, d * c]])
return p_tilde, M
```

`p̃ = p + d·[cosθ, sinθ]`를 미분하면 `ṗ̃ = M(θ)·[v, ω]`이고
`det M = d·(c² + s²) = d > 0`으로 **항상 가역**입니다. 이제 ω도 p̃의 속도에
들어오므로 CBF 행이 두 제어 모두에 걸립니다
([clf_cbf_qp.py:590-598](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)):

```python
# clf_cbf_qp.py:591-597 (_cbf_rows_kinematic)
for ox, oy, orad in self.obstacles:
    e_o = p_tilde - np.array([ox, oy])
    R = orad + p.lookahead_d + p.safety_margin
    h = float(e_o @ e_o - R**2)
    rows.append(2.0 * (M.T @ e_o))     # A행 = ∇h·M = 2·e_oᵀ·M
    rhs.append(-p.alpha_cbf * h)
```

반경에 `+ p.lookahead_d`가 더해지는 이유: 안전을 보장하는 것은 p̃이지 로봇
중심 p가 아니므로, p̃ 기준 반경을 d만큼 부풀려 로봇 본체를 커버합니다.
이걸 빼먹으면 로봇 뒤꽁무니가 장애물을 스칩니다.

### 5.3 5D backstepping-lite 캐스케이드

동역학 모델(제어 = 가속도)에서는 HOCBF 1단 캐스케이드 + 속도 오차 CLF를
씁니다 ([clf_cbf_qp.py:628-689](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)):

```python
# clf_cbf_qp.py:655-663
# 명목 제어: ẇ = u - C·w = -k_vel·e_w 가 되도록
w_d = self._desired_twist(state, ref_pt, v_ff, p_tilde, M)
e_w = w - w_d
u_nom = self._clip_u(C @ w - p.k_vel * e_w)
# CLF: V = 0.5·|e_w|²
V = 0.5 * float(e_w @ e_w)
a_clf = e_w.copy()
b_clf = -p.c_clf * V + float(e_w @ (C @ w))

# clf_cbf_qp.py:673-686 (HOCBF 행 — ḧ에 u가 등장)
for ox, oy, orad in self.obstacles:
    e_o = p_tilde - np.array([ox, oy])
    ...
    h_dot = 2.0 * float(e_o @ Mw)
    h_e = h_dot + p.lambda_hocbf * h
    drift = (
        2.0 * float(Mw @ Mw)
        + 2.0 * float(w[1]) * float(e_o @ (dM_dth @ w))
        - 2.0 * float(e_o @ (M @ (C @ w)))
    )
    rows.append(2.0 * (M.T @ e_o))
    rhs.append(-p.lambda_hocbf * h_dot - p.alpha_cbf * h_e - drift)
```

캐스케이드 구조: 기구학 레벨에서 desired twist `w_d`를 만들고(5.2절 로직
재사용), 동역학 레벨 CLF는 `e_w = w - w_d`만 잡습니다. `ẇ_d ≈ 0` 근사라서
"backstepping-**lite**"입니다 (정식 backstepping은 ẇ_d 항을 보상). drift
3개 항은 ḧ 전개에서 u에 안 걸리는 부분 전부입니다: `2|Mw|²`(속도 자체),
`2ω·e_oᵀ(dM/dθ)w`(회전에 의한 M 변화), `-2e_oᵀM·C·w`(마찰). 어느 하나라도
빠뜨리면 QP가 잘못된 우변으로 풀립니다 — 유도 검증은
[SAFETY_THEORY.md §3](../SAFETY_THEORY.md) "동역학 Differential Drive
(5-state) Lie 미분" 절과 대조하세요.

### 5.4 info dict 설계

```python
# clf_cbf_qp.py:741-752 (compute_control 반환 info)
info = {
    "clf_value": float(V),
    "min_barrier": float(min(h_vals)) if h_vals else float("inf"),
    "qp_feasible": bool(feasible),
    "delta": float(sinfo["delta"]),
    "active_constraints": sinfo["active_constraints"],
    "qp_method": sinfo["method"],           # 어느 경로로 풀렸나
    "u_nominal": u_nom.copy(), ...
}
```

repo 표준 `compute_control(state, ref) -> (u, info)` 인터페이스를 지키면서,
QP 특화 필드를 얹습니다. `qp_method`("analytic"/"analytic_projection"/
"slsqp"/"slsqp_infeasible")로 fast path 적중률을, `qp_feasible`로 실현
불가능률을 모니터링합니다 — 9절 벤치마크의 `qp_feasible_rate`가 바로 이
필드를 집계한 것입니다.

---

## 6. 필터 계열 비교 구현

세 구현은 모두 "MPPI 출력 u를 안전하게 고친다"는 같은 목표를 다른 지점에서
수행합니다. 이론 비교는 [SAFETY_THEORY.md §3, §4, §9](../SAFETY_THEORY.md),
[04_ADVANCED_SAFETY.md §6](04_ADVANCED_SAFETY.md).

### 6.1 CBFSafetyFilter — 기본형 (SLSQP QP)

[cbf_safety_filter.py](../../mppi_controller/controllers/mppi/cbf_safety_filter.py).
구조는 단순합니다: (1) Lie 미분으로 제약 구성 → (2) 빠른 경로 검사 →
(3) SLSQP → (4) 실패 시 정지.

```python
# cbf_safety_filter.py:94-109 (빠른 경로)
all_safe = True
for con in constraints:
    if con["fun"](u_mppi) < 0:
        all_safe = False
        break
if all_safe:
    return u_mppi.copy(), info      # QP를 아예 안 품
```

Lie 미분은 differential drive 기구학 전용 하드코딩입니다
([cbf_safety_filter.py:199-208](../../mppi_controller/controllers/mppi/cbf_safety_filter.py)):

```python
# cbf_safety_filter.py:199-208
Lf_h = 0.0  # kinematic, no drift
Lg_h = np.array([
    2.0 * (x - obs_x) * cos_theta + 2.0 * (y - obs_y) * sin_theta,
    0.0,                                # ω는 ḣ에 미기여
])
```

(손계산 유도는 [03_CBF_FUNDAMENTALS.md §4](03_CBF_FUNDAMENTALS.md).)
최적화 실패 시의 최후 폴백이 **정지 제어**라는 점이 특징입니다:

```python
# cbf_safety_filter.py:143-146
    else:
        # 안전하지 않은 결과 → 정지 제어 (v=0, ω=0)
        u_safe = np.zeros_like(u_mppi)
```

### 6.2 OptimalDecayCBFSafetyFilter — ω 추가 변수 QP

[optimal_decay_cbf_filter.py](../../mppi_controller/controllers/mppi/optimal_decay_cbf_filter.py)는
`CBFSafetyFilter`를 **상속**하고 `filter_control`만 오버라이드합니다.
결정 변수를 `z = [u, ω]`로 한 차원 늘려, 표준 CBF가 실현 불가능할 때
decay rate를 완화합니다:

```python
# optimal_decay_cbf_filter.py:126-144
# 결정 변수: z = [u (nu), ω (1)]
z0 = np.concatenate([u_mppi.copy(), [1.0]])  # 초기: ω=1

# 목적 함수: ||u - u_mppi||² + p_sb·(ω - 1)²
def objective(z):
    u, omega_decay = z[:nu], z[nu]
    control_cost = 0.5 * np.dot(u - u_mppi, u - u_mppi)
    decay_penalty = 0.5 * self.penalty_weight * (omega_decay - 1.0) ** 2
    return control_cost + decay_penalty

# optimal_decay_cbf_filter.py:149-153 (제약: α에 ω가 곱해짐)
def con_fun(z):
    u, omega_decay = z[:nu], z[nu]
    return Lf_h_ + Lg_h_ @ u + self.cbf_alpha * omega_decay * h_
```

`bounds.append((self.omega_min, self.omega_max))`
([optimal_decay_cbf_filter.py:169](../../mppi_controller/controllers/mppi/optimal_decay_cbf_filter.py))로
ω ∈ [0, 1]을 박스 제약으로 겁니다. h > 0(안전 집합 내부)이면 ω를 줄일수록
제약이 **약해지므로** ω=0에서는 항상 실현 가능 — "guaranteed feasibility"의
코드적 실체입니다. `penalty_weight=1e4`가 커서 가능한 한 ω=1(표준 CBF)에
붙어 있고, 불가능할 때만 미끄러집니다. info에 `optimal_omega`와
`decay_relaxed`(ω < 0.99)가 추가되고, `get_filter_statistics()`가
`mean_omega`/`min_omega`/`relaxation_rate`를 집계합니다
([optimal_decay_cbf_filter.py:199-216](../../mppi_controller/controllers/mppi/optimal_decay_cbf_filter.py)).

### 6.3 Shield-MPPI — rollout 내부 per-step 강제

[shield_mppi.py](../../mppi_controller/controllers/mppi/shield_mppi.py)는
필터가 아니라 **rollout을 통째로 교체**합니다. 시간 루프는 남지만 K는 완전
벡터화:

```python
# shield_mppi.py:227-242 (_shielded_rollout)
for t in range(N):
    states_t = trajectories[:, t, :]  # (K, nx)
    safe_controls_t, intervened, vel_reduction = (
        self._cbf_shield_batch(states_t, controls[:, t, :])
    )
    shielded_controls[:, t, :] = safe_controls_t
    # 안전한 제어로 다음 상태 전파
    trajectories[:, t + 1, :] = self.model.step(
        states_t, safe_controls_t, self.params.dt
    )
```

시간 루프를 없앨 수 없는 이유: t+1의 상태가 t의 **클리핑된** 제어에
의존하는 순차 의존성 때문입니다. 대신 스텝당 shield는 QP 없이 해석적
속도 상한으로 K개를 한 번에 처리합니다:

```python
# shield_mppi.py:311-326 (_cbf_shield_batch 핵심)
# Lg_h_v < 0인 경우만 제약 적용
approaching = Lg_h_v < -1e-10  # 수치 안정성

# v_ceiling_obs = α * h / |Lg_h_v| (접근 시만)
# h < 0이면 ceiling도 음수 → 후진만 허용
v_ceiling_obs = np.where(
    approaching,
    alpha * h / np.maximum(np.abs(Lg_h_v), 1e-10),
    np.inf,
)
v_ceiling = np.minimum(v_ceiling, v_ceiling_obs)
...
v_safe = np.minimum(v_original, v_ceiling)
```

`Lg_h = [·, 0]`이라 ω는 건드리지 않고 v에만 상한을 겁니다 — 6.1절 Lie 미분의
직접적 귀결입니다. 가중 업데이트도 달라집니다:

```python
# shield_mppi.py:127-132
# 6. Shielded noise로 제어 업데이트 (편향 방지)
shielded_noise = shielded_controls - self.U  # (K, N, nu)
weighted_noise = np.sum(
    weights[:, None, None] * shielded_noise, axis=0
)
self.U = self.U + weighted_noise
```

원래 `noise`가 아닌 `shielded_controls - U`를 쓰는 이유: 비용은 클리핑된
제어의 궤적으로 계산했으므로, 가중 평균도 **실제 평가된 제어** 기준이어야
분포-비용 대응이 맞습니다. 원래 noise로 업데이트하면 "안전하게 고쳐서 평가한
샘플"의 점수를 "고치기 전 샘플"에 주는 편향이 생깁니다.

**ESS 붕괴가 코드에서 일어나는 지점**: `_cbf_shield_batch`의
`v_safe = np.minimum(v_original, v_ceiling)`입니다. 장애물 접근 시 K개
샘플의 서로 다른 v가 **모두 같은 `v_ceiling`으로 눌리면**, 클리핑된 제어와
그 궤적·비용이 서로 복제본이 되고, softmax 가중치가 남은 미세 차이에
집중되어 ESS(`_compute_ess`, base_mppi.py:296)가 한 자릿수로 떨어집니다.
게다가 클리핑이 rollout의 매 스텝 반복되므로 다양성이 시간축을 따라
누적적으로 죽습니다. 실측: static_kin 시나리오에서 Shield ESS ≈ 1.4/512
(vs Vanilla 62) — 수치와 완화책(Adaptive Shield 등)은
[04_ADVANCED_SAFETY.md §6.2~6.3](04_ADVANCED_SAFETY.md) 참조.

### 6.4 세 구현 비교표

| | CBFSafetyFilter | OptimalDecayCBF | Shield-MPPI |
|---|---|---|---|
| 개입 지점 | `compute_control` 반환 직전 (u 1개) | 동일 | rollout 내부 (K×N 전부) |
| 최적화 | SLSQP, 변수 `u (nu)` | SLSQP, 변수 `[u, ω] (nu+1)` | 없음 — 해석적 `min(v, ceiling)` |
| 실현 불가능 시 | `u = 0` 정지 | ω 완화로 항상 feasible | 항상 feasible (클리핑이므로) |
| 빠른 경로 | 제약 만족 시 QP 생략 | 동일 (ω=1로 검사) | 없음 (매 스텝 실행) |
| 안전 보장 | 1스텝 CBF (QP 성공 시) | 1스텝, 단 ω<1이면 약화 | 전 샘플 궤적 수준 경향 |
| 부작용 | 계획-실행 불일치 | 동일 + 보장 약화 가능 | ESS 붕괴 |
| 공통점 | `filter_control(state, u, u_min, u_max) -> (u_safe, info)` + `filter_stats` 누적 + `get_filter_statistics()` | (상속으로 동일) | 통계는 `get_shield_statistics()` |

> **흔한 실수**: Shield-MPPI에서 동적 장애물 갱신 시
> `update_obstacles`를 부모 것만 호출하는 것. Shield의 `_cbf_shield_batch`는
> `self.cbf_params.cbf_obstacles`를 **직접 참조**하므로 오버라이드된
> `update_obstacles`([shield_mppi.py:363-372](../../mppi_controller/controllers/mppi/shield_mppi.py))가
> 이를 함께 갱신합니다 — 부모(`cbf_cost`/`safety_filter`)만 갱신하면 shield는
> 옛 장애물을 계속 봅니다.

---

## 7. Gatekeeper / Backup

파일: [gatekeeper.py](../../mppi_controller/controllers/mppi/gatekeeper.py),
[backup_controller.py](../../mppi_controller/controllers/mppi/backup_controller.py) —
이론은 [SAFETY_THEORY.md §10](../SAFETY_THEORY.md),
[04_ADVANCED_SAFETY.md §2](04_ADVANCED_SAFETY.md).

### 7.1 filter()의 3단계

```python
# gatekeeper.py:96-131 (filter 핵심)
# 1. u_mppi 적용 후 예측 상태
x_next = self.model.step(state, u_mppi, self.dt)

# 2. x_next에서 백업 궤적 생성
backup_traj = self.backup_controller.generate_backup_trajectory(
    x_next, self.model, self.dt, self.backup_horizon, self.obstacles
)

# 3. 백업 궤적의 안전성 검증
is_safe, min_barrier = self._check_trajectory_safety(backup_traj)

if is_safe:
    return u_mppi.copy(), info      # Gate open: MPPI 제어 허용
else:
    # Gate closed: 백업 제어 적용 (x_next 아닌 '현재 state' 기준!)
    u_backup = self.backup_controller.compute_backup_control(
        state, self.obstacles
    )
    return u_backup, info
```

논리 구조가 핵심입니다: 검증 대상은 u_mppi의 미래 전체가 아니라
**"u_mppi를 1스텝 적용한 뒤에도 백업 정책으로 탈출 가능한가"** 하나입니다.
이 불변식(항상 안전한 백업 궤적이 존재하는 상태에 머문다)이 귀납적으로
유지되면 무한 시간 안전이 따라옵니다 — 증명 스케치는
[04_ADVANCED_SAFETY.md §2](04_ADVANCED_SAFETY.md). 게이트가 닫힐 때 백업
제어를 현재 state에서 계산하는 것도 논리적으로 맞습니다: u_mppi는 적용하지
않기로 했으니까요.

### 7.2 _check_trajectory_safety — 벡터화

```python
# gatekeeper.py:148-163
positions = trajectory[:, :2]  # (T+1, 2)
min_barrier = float("inf")
for obs in self.obstacles:
    effective_r = obs[2] + self.safety_margin
    dx = positions[:, 0] - obs[0]
    dy = positions[:, 1] - obs[1]
    h = dx**2 + dy**2 - effective_r**2
    min_barrier = min(min_barrier, float(np.min(h)))
return min_barrier > 0, min_barrier
```

2절과 같은 패턴이되 궤적이 1개(`(T+1, nx)`)라 K축이 없습니다. 시간축은
`np.min(h)`으로 한 번에 처리 — 백업 horizon 30스텝을 파이썬 루프 없이
검사합니다. 판정은 `min_barrier > 0`의 strict 부등호: 경계 위(h=0)도
불안전으로 간주하는 보수적 선택입니다.

### 7.3 BrakeBackup vs TurnAndBrake — generate_backup_trajectory 차이

```python
# backup_controller.py:88-95 (BrakeBackupController)
u_backup = np.array([0.0, 0.0])
for t in range(horizon):
    trajectory[t + 1] = model.step(trajectory[t], u_backup, dt)

# backup_controller.py:161-170 (TurnAndBrakeBackupController)
u_turn = self.compute_backup_control(state, obstacles)   # [0, ±turn_speed]
u_stop = np.array([0.0, 0.0])
for t in range(horizon):
    u = u_turn if t < self.turn_steps else u_stop
    trajectory[t + 1] = model.step(trajectory[t], u, dt)
```

차이는 단 한 줄 — `u = u_turn if t < self.turn_steps else u_stop`.
TurnAndBrake는 처음 `turn_steps`(기본 5)스텝 동안 가장 가까운 장애물
**반대 방향**으로 회전(`compute_backup_control`이 `-sign(angle_diff) ·
turn_speed`로 방향 결정,
[backup_controller.py:139-151](../../mppi_controller/controllers/mppi/backup_controller.py))한
뒤 정지합니다. 기구학 모델에서 v=0이면 위치가 안 변하므로 Brake의 백업
궤적은 사실상 한 점입니다 — 그럼에도 rollout을 도는 이유는 **동역학 모델
(관성으로 미끄러짐)에서도 같은 코드가 올바르게 동작**하게 하기 위해서입니다.

> **왜 이렇게 (트레이드오프)**: 백업 정책은 단순할수록 좋습니다. 검증이
> 매 제어 주기 실행되므로 백업 rollout 비용 = horizon × model.step 1회이며,
> 백업이 보수적일수록(정지) 게이트가 자주 닫혀 성능이 죽고, 공격적일수록
> (회전 탈출) 검증 커버리지가 넓어져 게이트가 자주 열립니다.

모니터링은 `get_statistics()`
([gatekeeper.py:169-186](../../mppi_controller/controllers/mppi/gatekeeper.py)):
`gate_open_rate`가 1.0에 가깝다면 게이트가 사실상 통과 장치이고, 낮다면
MPPI 계획이 백업 불가능한 상태로 자주 들어간다는 뜻입니다 (층위 1 비용
강화 필요 신호).

> **흔한 실수**: `backup_horizon`을 너무 짧게 잡는 것. 백업 궤적이 정지
> 상태에 도달하기 전에 끝나면 "그 이후"의 안전을 검증하지 않은 것이 되어
> 무한 시간 보장 논리가 깨집니다. 정지(불변 상태) 도달까지 커버해야 합니다.

---

## 8. DualGuard의 SafetyValueFunction

파일: [dualguard_mppi.py](../../mppi_controller/controllers/mppi/dualguard_mppi.py) —
이론(HJ reachability와의 관계)은
[04_ADVANCED_SAFETY.md §3](04_ADVANCED_SAFETY.md).

### 8.1 배치 signed distance 평가

```python
# dualguard_mppi.py:88-107 (SafetyValueFunction.evaluate)
positions = states[..., :2]  # (..., 2)
orig_shape = positions.shape[:-1]
pos_flat = positions.reshape(-1, 2)  # (B, 2) — 임의 선행 차원 flatten

diff = pos_flat[:, None, :] - self._obs_pos[None, :, :]  # (B, M, 2)
dist = np.linalg.norm(diff, axis=-1)  # (B, M)
signed_dist = dist - (self._obs_rad[None, :] + self.safety_margin)  # (B, M)
values = np.min(signed_dist, axis=-1)  # (B,) — 장애물 축 min-reduce
return values.reshape(orig_shape)
```

2절의 CBF 비용과 다른 벡터화 전략을 씁니다: 장애물 루프 대신 **flatten →
(B, M) 브로드캐스트 → min-reduce → reshape**. `(K, N+1, nx)`든 `(nx,)`든
`(N+1, nx)`든 임의 선행 차원을 `B`로 뭉개서 처리하므로, 같은 함수가 궤적
배치 평가(step 5)와 nominal guard의 단일 상태 평가(`_guard_nominal`,
[dualguard_mppi.py:641](../../mppi_controller/controllers/mppi/dualguard_mppi.py))에
그대로 쓰입니다. 장애물 좌표는 생성자에서 `(M,2)`/`(M,)` 배열로 사전 변환해
둡니다 ([dualguard_mppi.py:62-69](../../mppi_controller/controllers/mppi/dualguard_mppi.py)).

CBF 비용과 달리 h가 아니라 **V = 진짜 거리(m 단위)**라는 점도 중요합니다 —
`dist - r`이지 `dist² - r²`이 아니므로 threshold(`safety_margin`)와 penalty
decay가 미터 단위로 해석됩니다.

TTC 근사는 `evaluate_with_velocity`
([dualguard_mppi.py:156-169](../../mppi_controller/controllers/mppi/dualguard_mppi.py))에
있습니다: 연속 위치의 forward 차분으로 속도를 근사하고, 장애물 방향
접근 속도가 음수일 때 `ttc = dist/|approach_speed|`를 계산, `ttc <
ttc_horizon`인 만큼 `ttc_factor`로 페널티를 키워 V에서 **뺍니다** (접근 중인
상태는 같은 거리라도 더 위험).

### 8.2 soft / hard / filter 3모드 분기

분기점은 `compute_control`의 step 7입니다
([dualguard_mppi.py:311-329](../../mppi_controller/controllers/mppi/dualguard_mppi.py)):

```python
# dualguard_mppi.py:312-329
mode = self.guard_params.safety_mode
if mode == "soft":
    costs = self._apply_soft_guard(costs, safety_values)
elif mode == "hard":
    sampled_controls = self._apply_hard_guard(
        sampled_controls, trajectories, safety_values, state
    )
    # Re-rollout with corrected controls
    trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
    costs = self.cost_function.compute_cost(...)
    safety_values = self._safety_value.evaluate(trajectories)
    costs = self._apply_soft_guard(costs, safety_values)
elif mode == "filter":
    costs = self._apply_filter_guard(costs, safety_values)
```

- **soft**: 비용만 증강 —
  `cost += penalty · Σ_t exp(-decay·V(x_t))·[V < threshold]`
  ([dualguard_mppi.py:479-491](../../mppi_controller/controllers/mppi/dualguard_mppi.py)).
  `np.clip(exponent, -50, 50)`의 overflow 가드에 주목 — V가 크게 음수면
  `exp`가 순식간에 inf가 됩니다.
- **hard**: 제어를 gradient 방향으로 직접 수정한 뒤 **re-rollout**합니다.
  세 모드 중 유일하게 rollout을 2번 도는 (2배 비용) 모드이고, 수정 후에도
  soft guard를 한 번 더 겁니다.
- **filter**: 이름은 "reject"지만 실제 구현은 가중치를 0으로 만드는 게
  아니라 **큰 비용을 더하는** 방식입니다:

```python
# dualguard_mppi.py:576-585 (_apply_filter_guard)
min_safety_per_sample = np.min(safety_values, axis=-1)  # (K,)
unsafe_mask = min_safety_per_sample < 0  # (K,)

filtered_costs = costs.copy()
large_penalty = self.guard_params.safety_penalty * self.params.N
filtered_costs[unsafe_mask] += large_penalty
```

> **왜 이렇게**: softmax를 통과하면 `+large_penalty`는 사실상 가중치 0과
> 같지만, **모든 샘플이 unsafe인 경우에도** (전부 같은 페널티를 받아 상대
> 순위가 보존되어) NaN 없이 동작합니다. `w_k = 0`을 문자 그대로 구현하면
> 이 경우 가중치 합이 0이 되어 0-나눗셈이 납니다. 안전 샘플 고갈 자체는
> `_check_and_boost_noise`
> ([dualguard_mppi.py:606-617](../../mppi_controller/controllers/mppi/dualguard_mppi.py))가
> `safe_fraction < min_safe_fraction`일 때 σ를 최대 5배까지 증폭해
> 탐색적으로 해결합니다 — 6.3절 Shield의 ESS 붕괴와 정반대 방향의 대응
> (다양성을 죽이는 대신 늘림)이라는 점이 흥미로운 대비입니다.

---

## 9. 벤치마크에서 안전 메트릭 계산

파일: [cbfkit_inspired_benchmark.py](../../examples/comparison/cbfkit_inspired_benchmark.py) —
결과 해석은 [CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md),
[04_ADVANCED_SAFETY.md §7](04_ADVANCED_SAFETY.md).

### 9.1 min_clearance / collisions

```python
# cbfkit_inspired_benchmark.py:421-428 (compute_metrics)
# 클리어런스 (마진 없이 dist - r_obs), 충돌 = dist < r_obs 인 타임스텝 수
pos = states[:, :2]
clear_ts = np.full(len(states), np.inf)
for ox, oy, r in obstacles:
    d = np.hypot(pos[:, 0] - ox, pos[:, 1] - oy) - r
    clear_ts = np.minimum(clear_ts, d)
min_clearance = float(np.min(clear_ts))
collisions = int(np.sum(clear_ts < 0.0))
```

두 가지 설계 결정을 읽어야 합니다:

1. **safety_margin을 빼지 않습니다** — `d = dist - r_obs`이지
   `- (r_obs + margin)`이 아닙니다. 평가 기준을 컨트롤러 파라미터
   (기법마다 margin이 0.1/0.15/0.2로 다름)와 분리해, 모든 기법을 **물리적
   충돌**이라는 동일 잣대로 비교합니다. 컨트롤러의 margin은 "여유"로,
   메트릭의 clearance는 "사실"로 남습니다.
2. `collisions`는 이진 플래그가 아니라 **침입 타임스텝 수** — 스치듯 1스텝
   침입과 관통을 구분합니다. `clear_ts`는 먼저 시간별 최소(장애물 축
   `np.minimum` 누적)를 만들고, 그 시계열에서 `np.min`(min_clearance)과
   `np.sum(< 0)`(collisions)을 뽑는 2단 reduce입니다.

### 9.2 시드 재현성 — 샘플러 주입

```python
# cbfkit_inspired_benchmark.py:260-261 (make_controller)
# 기본 GaussianSampler(seed=None) 는 OS 엔트로피 시드 → 재현성 위해 명시 시드
sampler = GaussianSampler(pd["sigma"], seed=seed)

# cbfkit_inspired_benchmark.py:358-363 (run_single)
np.random.seed(seed)
rng = np.random.default_rng(seed + 10000)
controller = make_controller(name, scenario, rho_override=rho_override, seed=seed)
```

난수 소스가 **세 갈래**임을 구분해야 합니다:

| 소스 | 용도 | 시드 |
|---|---|---|
| `GaussianSampler(sigma, seed=seed)` | MPPI 탐색 노이즈 | 명시 주입 (컨트롤러 생성 시) |
| `np.random.seed(seed)` | 레거시 전역 RNG를 쓰는 내부 코드 방어 | 전역 |
| `np.random.default_rng(seed + 10000)` | **프로세스 노이즈** (환경 외란) | 별도 스트림 |

프로세스 노이즈에 `seed + 10000`의 독립 `Generator`를 쓰는 이유: 탐색
노이즈와 환경 외란이 같은 스트림을 공유하면, 컨트롤러가 노이즈를 몇 번
뽑느냐(K, N 설정)에 따라 **환경이 달라지는** 교차 오염이 생깁니다. 스트림을
분리하면 "같은 시드 = 같은 외란 시퀀스"가 컨트롤러와 무관하게 보장됩니다.
stochastic 시나리오는 seeds `[42, 43, 44]` 3개로 mean±std를 집계합니다
([cbfkit_inspired_benchmark.py:485-506](../../examples/comparison/cbfkit_inspired_benchmark.py)).

> **흔한 실수**: `GaussianSampler`를 시드 없이 만들고 `np.random.seed`만
> 부르는 것. 주석이 경고하듯 기본 샘플러는 OS 엔트로피로 시드되므로 전역
> 시드의 영향을 받지 않아, "시드 고정했는데 결과가 매번 다른" 미스터리가
> 됩니다. 재현성은 **샘플러 객체에 시드를 주입**해야 확보됩니다.

또한 `run_single`은 매 스텝 `info` dict에서 스칼라만 추출해 저장합니다
(`ess`, `hocbf_filter.filtered`, `qp_feasible` —
[cbfkit_inspired_benchmark.py:387-393](../../examples/comparison/cbfkit_inspired_benchmark.py)).
`sample_trajectories` 같은 `(K, N+1, nx)` 배열을 400스텝 쌓으면 GB 단위가
되므로, "스칼라 info만 저장 (메모리 절약)"이 장시간 벤치마크의 생존
조건입니다.

---

## 10. 실습 — 내 안전 비용을 하나 만들어보자

지금까지의 패턴을 종합해 **축 정렬 사각형(AABB) 장애물의 SDF barrier 비용**을
직접 만들고 MPPI에 결합합니다. 원형만 지원하는 기존 비용들과 달리 벽/책상
같은 박스 장애물을 다룰 수 있습니다.

설계 결정 (2~4절 패턴 재사용):
- barrier: `h = SDF_box(p) - margin` (SDF는 밖에서 양수/안에서 음수 → 그대로
  barrier). 8절 DualGuard처럼 **미터 단위** signed distance를 씁니다.
- 조건: 2절과 동일한 이산 CBF `h_{t+1} - (1-α)h_t ≥ 0` + 침입 벌점.
- 벡터화: `(K, N+1, 2)`에서 K 루프 없이 계산.

전체 코드 (스크래치패드
`custom_box_barrier_demo.py`에서 실행 검증 완료):

```python
import numpy as np
from mppi_controller.controllers.mppi.cost_functions import (
    CostFunction, CompositeMPPICost, StateTrackingCost, TerminalCost, ControlEffortCost,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.sampling import GaussianSampler
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)


class BoxBarrierCost(CostFunction):
    """축 정렬 사각형 장애물 SDF barrier 비용 (이산 CBF 조건)"""

    def __init__(self, boxes, cbf_alpha=0.2, weight=2000.0, margin=0.15):
        # boxes: [(cx, cy, half_w, half_h), ...]
        self.boxes = boxes
        self.cbf_alpha = cbf_alpha
        self.weight = weight
        self.margin = margin

    def _sdf(self, positions, box):
        """AABB signed distance. positions: (..., 2) -> (...)"""
        cx, cy, hw, hh = box
        qx = np.abs(positions[..., 0] - cx) - hw
        qy = np.abs(positions[..., 1] - cy) - hh
        # 바깥: ||max(q,0)||, 안: max(qx,qy) (<0)
        outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside

    def compute_cost(self, trajectories, controls, reference_trajectory):
        K = trajectories.shape[0]
        costs = np.zeros(K)
        positions = trajectories[:, :, :2]                    # (K, N+1, 2)
        for box in self.boxes:
            h = self._sdf(positions, box) - self.margin       # (K, N+1)
            # 이산 CBF 조건: h_{t+1} - (1-alpha) h_t >= 0
            cond = h[:, 1:] - (1.0 - self.cbf_alpha) * h[:, :-1]  # (K, N)
            costs += self.weight * np.sum(np.maximum(0.0, -cond), axis=1)
            # 침입 자체에도 벌점 (h<0 인 스텝)
            costs += self.weight * np.sum(np.maximum(0.0, -h), axis=1)
        return costs
```

`_sdf`가 핵심입니다. AABB SDF의 표준형:
- 박스 중심 좌표계에서 `q = |p - c| - halfsize` — `np.abs`가 4분면 대칭을
  한 번에 처리합니다.
- 바깥 거리 `||max(q, 0)||₂` (모서리 근처에서는 코너까지의 유클리드 거리,
  면 근처에서는 수직 거리로 자동 전환), 안쪽 거리 `min(max(qx,qy), 0)`
  (가장 가까운 면까지의 음수 거리).
- 전부 `(...)` 임의 배치 shape에서 동작 — 2절의 벡터화 규율 그대로입니다.

원형 barrier와 달리 **h_{t+1} - (1-α)h_t 항만으로는 부족해** 침입 벌점
`max(0, -h)`를 추가했습니다: SDF는 박스 내부에서 기울기가 완만해서(면까지
거리) CBF 조건만으로는 이미 침입한 샘플의 탈출 압력이 약하기 때문입니다
(원형의 거리 제곱 h는 내부에서 빠르게 음수가 커져 이 문제가 덜합니다).

MPPI 결합은 1절 콜 그래프의 층위 1 그대로 — `CompositeMPPICost`에 한 항
추가:

```python
model = DifferentialDriveKinematic(v_max=1.5, omega_max=2.0)
boxes = [(0.0, 0.15, 0.4, 0.4)]  # 0.8×0.8 박스가 직선 경로를 가로막음
params = MPPIParams(K=512, N=30, dt=0.05, lambda_=0.1,
                    sigma=np.array([0.6, 0.9]),
                    Q=np.array([10.0, 10.0, 1.0]), R=np.array([0.1, 0.1]))
cost = CompositeMPPICost([
    StateTrackingCost(params.Q), TerminalCost(params.Qf),
    ControlEffortCost(params.R),
    BoxBarrierCost(boxes, cbf_alpha=0.2, weight=2000.0, margin=0.15),
])
sampler = GaussianSampler(params.sigma, seed=42)   # 9.2절: 시드 주입
controller = MPPIController(model, params, cost, noise_sampler=sampler)
```

시나리오: `(-2, 0)`에서 출발해 x축 직선 레퍼런스(0.5 m/s)를 추적하는데,
경로 한가운데를 0.8×0.8 박스가 막고 있습니다. **실행 결과** (10초, 200스텝):

```
$ PYTHONPATH=. python custom_box_barrier_demo.py
steps=200, final_pos=(2.969, 0.017)
RMSE=0.4291 m
min SDF (box clearance)=0.1619 m  -> SAFE
max |y| detour=0.546 m
OK: 사각형 SDF barrier 비용으로 우회 성공
```

읽는 법: 로봇은 박스를 y 방향 최대 0.55m 우회해 통과했고, 궤적 전체에서
박스 표면까지의 최소 거리 0.162m > 0 (침입 0회, `margin=0.15`를 근소하게
웃도는 값 — barrier가 정확히 margin 경계에서 작동했다는 증거). RMSE
0.43m는 우회 구간이 포함된 값입니다.

**실습에서 재현된 흔한 실수 하나**: 처음에는 박스를 y=0 정중앙
`(0.0, 0.0, 0.4, 0.4)`에 두었는데, 로봇이 박스 앞에서 **멈춰버렸습니다**
(final x = -0.56, 우회 실패). 정면 대칭 장애물은 위/아래 우회 비용이 정확히
같아 가중 평균이 0으로 상쇄되는 고전적 local minimum입니다. 박스를 y=0.15로
살짝 오프셋하고 ω 노이즈를 키우자(σ_ω 0.5→0.9) 해결됐습니다 — 실전에서
대칭 정체를 근본적으로 다루는 기법이 DRPA-MPPI(반발 포텐셜)와
CLF-CBF-QP의 접선 투영 휴리스틱
([clf_cbf_qp.py:480-523](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)
`_project_pdot_around_obstacles`)입니다.

**확장 과제**:
1. `BoxBarrierCost`에 `get_barrier_info()`를 추가해
   ([cbf_cost.py:90-133](../../mppi_controller/controllers/mppi/cbf_cost.py)
   패턴) 시각화 오버레이와 연동해 보세요.
2. 회전된 박스(OBB): `_sdf` 앞에서 `p' = Rᵀ(p - c)` 회전 변환 한 줄이면
   됩니다 — 벡터화가 깨지지 않는지 shape를 추적해 보세요.
3. 같은 박스 SDF로 층위 2 필터를 만들어 보세요: `∇h`는 SDF의 gradient
   (바깥에서는 `max(q,0)/||max(q,0)||`에 부호 복원)이고, 이후는 3.3절
   HOCBFFilter의 투영 공식 그대로입니다.
4. 6절 비교를 재현: 같은 시나리오에 `BoxBarrierCost`(층위 1) vs 박스 필터
   (층위 2)를 넣고 RMSE / min SDF / ESS를 비교해 보세요 — ESS는 어느 쪽이
   높을까요?

---

## 정리 — 파일 지도

| 절 | 파일 | 핵심 함수 |
|---|---|---|
| 1 | base_mppi.py, cbf_mppi.py | `compute_control` 파이프라인, Layer 1/2 접합부 |
| 2 | cbf_cost.py | `ControlBarrierCost.compute_cost` |
| 3 | hocbf_cost.py | `HOCBFCost.compute_cost`, `detect_relative_degree`, `HOCBFFilter.filter_control` |
| 4 | stochastic_cbf.py, robust_cbf_margin.py | `ito_correction`, `get_margin`, `_robust_term` |
| 5 | clf_cbf_qp.py | `CBFCLFQPSolver.solve` 3-경로, `_lookahead_map`, `_terms_dynamic` |
| 6 | cbf_safety_filter.py, optimal_decay_cbf_filter.py, shield_mppi.py | `filter_control` ×2, `_shielded_rollout`, `_cbf_shield_batch` |
| 7 | gatekeeper.py, backup_controller.py | `filter`, `_check_trajectory_safety`, `generate_backup_trajectory` |
| 8 | dualguard_mppi.py | `SafetyValueFunction.evaluate`, `_apply_{soft,hard,filter}_guard` |
| 9 | examples/comparison/cbfkit_inspired_benchmark.py | `compute_metrics`, `make_controller`, `run_single` |
| 10 | (실습) | `BoxBarrierCost` — 2절 골격 + AABB SDF |

다음 단계: 이론적 배경이 궁금하면
[03_CBF_FUNDAMENTALS.md](03_CBF_FUNDAMENTALS.md)(CBF 수학)와
[04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md)(층위/트레이드오프)로,
전체 기법 카탈로그와 선택 가이드는
[SAFETY_THEORY.md §21](../SAFETY_THEORY.md)로 이동하세요.
