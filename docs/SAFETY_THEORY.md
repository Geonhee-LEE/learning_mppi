# 안전 제어 심층 이론 가이드

> **대상 독자**: CBF(Control Barrier Function) 이론과 22종 안전 기법의 수학적 기초를 이해하고자 하는 대학원생 및 연구자
>
> **구현 참조**: 모든 수식은 `mppi_controller/controllers/mppi/` 내 실제 코드와 1:1 대응

---

## 목차

1. [안전 제어 기초](#1-안전-제어-기초)
2. [CBF-MPPI (비용 기반)](#2-cbf-mppi-비용-기반)
3. [CBF Safety Filter (QP 기반)](#3-cbf-safety-filter-qp-기반)
4. [Shield-MPPI (per-step 강제)](#4-shield-mppi-per-step-강제)
5. [Adaptive Shield (α 적응)](#5-adaptive-shield-α-적응)
6. [C3BF (Collision Cone)](#6-c3bf-collision-cone)
7. [DPCBF (Dynamic Parabolic)](#7-dpcbf-dynamic-parabolic)
8. [Neural CBF (학습 기반)](#8-neural-cbf-학습-기반)
9. [Optimal-Decay CBF (이완형)](#9-optimal-decay-cbf-이완형)
10. [Backup CBF + Gatekeeper](#10-backup-cbf--gatekeeper)
11. [MPS (Model Predictive Shield)](#11-mps-model-predictive-shield)
12. [Conformal Prediction + CBF](#12-conformal-prediction--cbf)
13. [DIAL + Safety 결합](#13-dial--safety-결합)
14. [Chance Constraint (C2U 연계)](#14-chance-constraint-c2u-연계)
15. [안전 기법 선택 가이드](#15-안전-기법-선택-가이드)

---

## 1. 안전 제어 기초

### 1.1 안전 집합과 Barrier Function

안전 제어의 핵심은 시스템 상태 x를 **안전 집합 C** 내에 유지하는 것이다.

**안전 집합**:
```
C = {x ∈ R^n : h(x) ≥ 0}
```

여기서 h(x)는 **Control Barrier Function (CBF)**으로,
안전 영역에서 양수, 장애물 내부에서 음수, 경계에서 0이다.

**예시: 원형 장애물**:
```
h(x) = ||p - p_obs||² - r²
     = (x - x_obs)² + (y - y_obs)² - r²

h(x) > 0:  안전 (장애물 밖)
h(x) = 0:  경계 (장애물 표면)
h(x) < 0:  위험 (장애물 안)
```

```
ASCII 도식:
┌──────────────────────────────┐
│                              │
│    h > 0 (안전 영역)         │
│         ┌─────┐             │
│         │h < 0│  ← 장애물    │
│         │     │             │
│    ●→   └─────┘  h = 0 경계  │
│   로봇                       │
│                              │
└──────────────────────────────┘
```

### 1.2 다양한 h(x) 설계 예시

**예시 1: 직사각형 장애물**

직사각형 장애물 (중심 (cx, cy), 반폭 wx, 반높이 wy)에 대한 h(x):
```
h_rect(x) = max(|x - cx| - wx, |y - cy| - wy)  를 뒤집으면:

h(x) = min(wx - |x - cx|, wy - |y - cy|)   (장애물 내부가 h < 0)

→ 로봇 관점: h(x) = max(|x - cx| - wx, |y - cy| - wy)
  장애물 밖에서 h > 0, 장애물 안에서 h < 0

→ 또는 매끄러운 근사:
  h(x) = ((x-cx)/wx)^p + ((y-cy)/wy)^p - 1     (p = 4, 6 등)
```

```
ASCII 도식: 직사각형 장애물 h(x) 등고선
┌──────────────────────────────────┐
│  h=2   h=1                       │
│   ╭────────────────╮  h=0 경계   │
│   │ h=-1  h=-2     │             │
│   │   ┌────────┐   │             │
│   │   │obstacle│   │             │
│   │   └────────┘   │             │
│   │                │             │
│   ╰────────────────╯             │
│                          h=3     │
└──────────────────────────────────┘
```

**예시 2: 복도 (양쪽 벽)**

복도 y ∈ [y_min, y_max] 에서의 h(x):
```
h_left(x)  = y - y_min        (왼쪽 벽에서 떨어진 정도)
h_right(x) = y_max - y        (오른쪽 벽에서 떨어진 정도)

결합: h(x) = min(h_left, h_right) = min(y - y_min, y_max - y)

→ 매끄러운 근사 (smooth min):
  h(x) = -1/k · log(exp(-k·h_left) + exp(-k·h_right))     (k > 0)
```

**예시 3: 볼록 다각형 장애물**

n개 면을 가진 볼록 다각형, 각 면의 법선 n_i와 오프셋 d_i:
```
h_i(x) = n_i^T · p - d_i      (i번째 면에서의 부호 거리)

전체: h(x) = min_i h_i(x)

→ 매끄러운 근사:
  h(x) = -1/k · log(Σ_i exp(-k · h_i(x)))
```

**참고**: `min` 함수는 미분 불가능하므로, log-sum-exp 근사를 사용하면
그래디언트 기반 최적화(QP, 학습)에서 안정적이다.

### 1.3 연속시간 CBF 조건

**Control-Affine 시스템**:
```
ẋ = f(x) + g(x)u
```

다양한 로봇에 대한 control-affine 형태:

**Differential Drive (기구학, 3-state)**:
```
x = [x, y, θ],  u = [v, ω]

f(x) = [0, 0, 0]^T     (drift 없음)

g(x) = [cos(θ)  0]
       [sin(θ)  0]
       [0       1]
```

**Differential Drive (동역학, 5-state)**:
```
x = [x, y, θ, v, ω],  u = [F, τ]    (힘, 토크)

f(x) = [v·cos(θ)]      g(x) = [0    0  ]
       [v·sin(θ)]             [0    0  ]
       [ω       ]             [0    0  ]
       [0       ]             [1/m  0  ]
       [0       ]             [0    1/I]
```

**Ackermann (기구학, 4-state)**:
```
x = [x, y, θ, δ],  u = [v, δ_dot]    (속도, 조향 변화율)

f(x) = [0, 0, 0, 0]^T

g(x) = [cos(θ)         0]
       [sin(θ)         0]
       [tan(δ)/L       0]     (L = 축거)
       [0               1]
```

**연속시간 CBF 조건 (Nagumo 정리)**:
```
ḣ(x, u) + α(h(x)) ≥ 0
```

여기서 α는 class-K 함수 (가장 간단한 경우 α(h) = α·h, α > 0).

이 조건이 만족되면 **전방 불변성(forward invariance)** 이 보장된다:
```
h(x(0)) ≥ 0  ⟹  h(x(t)) ≥ 0,  ∀t ≥ 0
```

### 1.3.1 전방 불변성 증명 (Nagumo 정리)

**정리 (Nagumo)**: 닫힌 집합 C = {x : h(x) >= 0}에 대해,
모든 x in dC (경계)에서 ḣ(x,u) + alpha(h(x)) >= 0을 만족하는
제어 u가 존재하면, C는 전방 불변이다.

**증명 스케치**:

```
Step 1: 모순 가정
  h(x(0)) >= 0이지만, 어떤 시각 t* > 0에서
  h(x(t*)) < 0이 된다고 가정하자.

Step 2: 연속성에 의한 경계 시점 존재
  h는 연속이므로, h(x(t_0)) = 0인 시각 t_0 in (0, t*]이 존재.
  이 시점에서 h가 0에서 음수로 전환된다.
  즉, ḣ(x(t_0), u(t_0)) < 0.

Step 3: CBF 조건 적용
  CBF 조건에 의해:
    ḣ(x(t_0), u(t_0)) >= -α(h(x(t_0))) = -α(0) = 0

  따라서 ḣ(x(t_0), u(t_0)) >= 0.

Step 4: 모순
  Step 2에서 ḣ < 0, Step 3에서 ḣ >= 0.
  이는 모순이므로, h(x(t)) >= 0이 모든 t >= 0에서 성립한다.  ∎
```

**직관적 해석**: CBF 조건은 "h가 0에 도달할 때, h의 시간 미분이
비음수여야 한다"는 것을 의미한다. 즉, 경계에서 안쪽으로
밀어내는 힘이 항상 존재한다.

**Lie 미분**:

CBF 조건을 전개하면:
```
ḣ = ∂h/∂x · ẋ = ∂h/∂x · [f(x) + g(x)u]
   = L_f h(x) + L_g h(x) · u
```

여기서:
- `L_f h = ∂h/∂x · f(x)`: f에 대한 Lie 미분 (drift 항)
- `L_g h = ∂h/∂x · g(x)`: g에 대한 Lie 미분 (control 항)

**CBF 제약**:
```
L_f h(x) + L_g h(x) · u + α · h(x) ≥ 0
```

### 1.4 이산시간 CBF 조건

디지털 제어에서는 이산시간 버전을 사용한다:

```
h(x_{t+1}) ≥ (1 - α) · h(x_t)

⟺  h(f(x_t, u_t)) - (1-α) · h(x_t) ≥ 0
```

α ∈ (0, 1]로:
- α = 1: h(x_{t+1}) ≥ 0 (가장 보수적, 즉시 안전 강제)
- α → 0: h 값의 감소만 제한 (점진적)

### 1.5 Class-K 함수

```
α: R≥0 → R≥0, 연속, 순증가, α(0) = 0
```

**다양한 Class-K 함수 예시**:
```
단순 선형:   α(h) = c · h            (c > 0)
이차:       α(h) = c · h²
제곱근:     α(h) = c · √h
Tanh:       α(h) = c · tanh(h)      (포화 특성)
적응형:     α(d, v) = f(d, v) · h
```

```
Class-K 함수 비교 (ASCII plot):

α(h) ↑
  4  │                          ╱ 이차 (h²)
     │                        ╱
  3  │                      ╱
     │                    ╱     ╱─── 선형 (c·h)
  2  │                 ╱      ╱
     │               ╱      ╱   ╱─── 제곱근 (√h)
  1  │            ╱       ╱   ╱ ╱─── tanh (포화)
     │         ╱        ╱  ╱  ╱
  0  ├────────╱───────╱─╱─╱──────→ h
     0       1       2  3  4

 선형:    일정한 감쇠율, 가장 일반적
 이차:    h 클 때 공격적 감쇠 (빠른 복귀)
 제곱근:  h 작을 때 공격적 (경계 근처 민감)
 Tanh:    h 클 때 감쇠 포화 (과도 수정 방지)
```

**Class-K 선택 가이드**:
- **선형 α(h) = c·h**: 대부분의 경우에 적합. c가 클수록 보수적.
  c가 너무 크면 chattering (떨림) 발생 가능.
- **이차 α(h) = c·h²**: 장애물에서 멀 때 빠르게 복귀. 경계 근처에서는
  약해지므로 단독 사용 주의.
- **제곱근 α(h) = c·√h**: 경계(h→0) 근처에서 강한 반발력.
  실제 로봇 제어에서 안전 마진이 중요한 경우 적합.
- **Tanh α(h) = c·tanh(h)**: 포화 특성으로 큰 h에서 과도한 수정을 방지.
  제어 입력 제한이 있는 시스템에 적합.

### 1.6 상대 차수(Relative Degree)와 HOCBF

**상대 차수**: h(x)의 시간 미분에서 제어 u가 처음 나타나는 차수.

```
상대 차수 1:  ḣ = L_f h + L_g h · u     (L_g h ≠ 0)
  → CBF 직접 적용 가능

상대 차수 2:  ḣ = L_f h     (L_g h = 0 — u가 ḣ에 직접 나타나지 않음)
              ḧ = L_f² h + L_g L_f h · u    (L_g L_f h ≠ 0)
  → Higher-Order CBF (HOCBF) 필요
```

**언제 상대 차수 > 1인가?**

원형 장애물 h = ||p - p_obs||² - r² 에 대해:

- **기구학 모델** (u = [v, ω]): ∂h/∂x · g(x)에서 v가 나타남 → 상대 차수 1
- **동역학 모델** (u = [F, τ]): ∂h/∂x가 위치에만 의존하고,
  g(x)가 가속도를 제어하므로 L_g h = 0 → 상대 차수 2

```
예시: 동역학 Differential Drive (5-state)

h = (x-xo)² + (y-yo)² - r²

∂h/∂x = [2(x-xo), 2(y-yo), 0, 0, 0]

g(x) = [0, 0, 0, 1/m, 0]^T (F 열)
       [0, 0, 0, 0, 1/I]^T (τ 열)

L_g h = ∂h/∂x · g = [0, 0] → 상대 차수 > 1!
```

**HOCBF (Higher-Order CBF) 개요**:

상대 차수 r인 경우, r개의 barrier function을 연쇄적으로 정의한다:

```
ψ_0(x) = h(x)
ψ_1(x) = ψ̇_0 + α_1(ψ_0)
ψ_2(x) = ψ̇_1 + α_2(ψ_1)
...
HOCBF 제약: ψ̇_{r-1} + α_r(ψ_{r-1}) ≥ 0
```

이 마지막 조건에서 u가 직접 나타나므로 QP로 풀 수 있다.

**본 프로젝트에서는** 기구학 모델(상대 차수 1)을 주로 사용하므로
표준 CBF를 적용한다. 동역학 모델에서는 HOCBF 또는
가상 제어(v, ω를 제어 입력으로 취급)를 사용한다.

---

## 2. CBF-MPPI (비용 기반)

### 문제

MPPI에서 안전을 가장 쉽게 통합하는 방법은?

### 핵심 아이디어

CBF 위반을 **비용 함수**에 페널티로 추가하여,
위반 궤적의 가중치를 줄인다. (Soft constraint)

### 수학적 정의

**ControlBarrierCost**:
```
h(x) = ||p - p_obs||² - (r + margin)²

CBF 위반량:
  violation_t = max(0, -[h(x_{t+1}) - (1-α)·h(x_t)])

비용:
  J_cbf = weight · Σ_t violation_t
```

**전체 비용 구조**:
```
J_total = J_tracking + J_control + J_cbf
        = CompositeMPPICost(StateTracking, ControlEffort, ControlBarrierCost)
```

### MPPI 가중치에 대한 CBF 비용의 영향

CBF 비용이 MPPI 가중치 분포를 어떻게 변화시키는지 분석한다.

**MPPI 가중치 공식**:
```
w_k = exp(-J_k / λ) / Σ_j exp(-J_j / λ)
```

CBF 비용이 추가되면:
```
J_k = J_tracking_k + J_control_k + J_cbf_k

J_cbf_k = weight · Σ_t max(0, -[h(x_{t+1}) - (1-α)·h(x_t)])
```

**비용 경관 시각화 (2D 평면, 장애물 하나)**:
```
J_total
  ↑
  │           ╱╲
  │          ╱  ╲  ← CBF 비용 벽
  │         ╱    ╲      (weight = 100)
  │        ╱      ╲
  │       ╱        ╲
  │      ╱          ╲─────── CBF 비용 벽 (반대편)
  │─────╱   ○ 장애물  ╲
  │    ╱       (h<0)   ╲─────── 추적 비용 곡선
  │   ╱                 ╲
  │──╱                   ╲
  ├──┼──────┼──────┼──────┼──→ x
  x_start  obs   x_goal
```

**가중치 분석**:
```
weight 작을 때 (예: 1.0):
  위반 궤적도 상당한 가중치를 유지
  → 최종 제어에 위험 궤적이 기여 → 안전 위반 가능

weight 클 때 (예: 1000):
  위반 궤적의 가중치가 0에 수렴
  → 안전 궤적만 기여하지만, ESS(유효 샘플 수) 감소
  → 극단적으로 크면 하나의 샘플만 기여 (사실상 최소 비용 선택)

weight 권장 범위: [10, 200]
  추적 비용(~1.0)의 10~200배 정도가 적절
```

### 다중 장애물 확장

다중 장애물이 있을 때, CBF 비용은 각 장애물에 대해 독립적으로 계산한다:
```
J_cbf = Σ_j (weight_j · Σ_t max(0, -[h_j(x_{t+1}) - (1-α_j)·h_j(x_t)]))

여기서 j = 1,...,M (장애물 인덱스)
```

장애물별로 다른 weight와 α를 설정할 수 있다:
- 큰 장애물: 높은 weight, 높은 α (보수적)
- 작은 장애물: 낮은 weight, 낮은 α (유연)

### 파라미터 민감도

```
┌────────────────────────────────────────────────────────┐
│ weight  │ 효과                     │ 위험               │
├─────────┼──────────────────────────┼────────────────────┤
│ < 1     │ CBF 거의 무시            │ 안전 위반          │
│ 1~10    │ 약한 안전 선호           │ 근접 시 위반       │
│ 10~100  │ 적절한 안전-성능 균형    │ 권장 범위          │
│ 100~1000│ 강한 안전 (보수적)       │ ESS 감소           │
│ > 1000  │ 사실상 Hard constraint   │ 최적화 불안정      │
└─────────┴──────────────────────────┴────────────────────┘

α (CBF 파라미터):
  α 큼 (→1):  h가 빨리 회복되어야 함 → 보수적
  α 작음 (→0): h 감소를 느슨하게 허용 → 유연
  권장: α ∈ [0.1, 0.5]
```

### 한계

**Soft constraint** → 위반 가능:
- 가중치가 0이 아닌 한, 위반 궤적이 최종 제어에 기여
- 충분히 큰 weight가 필요하지만, 너무 크면 최적화 불안정

### 구현

- **파일**: `cbf_cost.py` (라인 13-146)
- **핵심 메서드**:
  - `compute_cost()`: (K,) 비용 반환
  - `get_barrier_info()`: `{barrier_values, min_barrier, is_safe}` 반환
- **파라미터**: `cbf_alpha ∈ (0,1]`, `cbf_weight`, `safety_margin`

### 변형: HorizonWeightedCBFCost

시간 할인으로 가까운 미래의 위반에 더 높은 페널티를 부여:
```
J = Σ_t γ^t · weight · max(0, -[h(x_{t+1}) - (1-α)·h(x_t)])
```
- γ < 1: 가까운 미래 중시 (보수적)
- γ = 1: 기존 CBF와 동일

**구현**: `horizon_cbf_cost.py:HorizonWeightedCBFCost`

### 변형: HardCBFCost

이진 거부 — 궤적 어디서든 h(x) < 0이면 전체 비용 = 1e6:
```
cost = { rejection_cost  if ∃t: h(x_t) < 0
       { 0               otherwise
```

**구현**: `hard_cbf_cost.py:HardCBFCost`

### 언제 사용

- 계산 비용이 최소여야 하는 경우
- 안전 위반이 치명적이지 않은 경우 (soft 허용)
- 다른 안전 메커니즘의 보완으로

---

## 3. CBF Safety Filter (QP 기반)

### 문제

MPPI의 출력 제어가 안전하지 않을 수 있다.
최소한의 수정으로 안전을 보장하는 사후 처리가 필요하다.

### 핵심 아이디어

MPPI 출력 `u_mppi`에 가장 가까우면서 CBF 제약을 만족하는 제어를 QP로 찾는다.

### 수학적 정의

**QP (Quadratic Program)**:
```
min_u   ||u - u_mppi||²

s.t.    L_f h_j + L_g h_j · u + α · h_j ≥ 0    ∀j = 1,...,M (장애물)
        u_min ≤ u ≤ u_max                        (제어 제한)
```

### Differential Drive 전용 Lie 미분 유도

상태: x = [x, y, θ], 제어: u = [v, ω]

**기구학 모델** (drift 없음):
```
ẋ = [v·cos(θ), v·sin(θ), ω]^T

f(x) = 0,  g(x) = [cos(θ)  0]
                   [sin(θ)  0]
                   [0       1]
```

**Barrier**: h = (x-x_o)² + (y-y_o)² - r²

**Lie 미분**:
```
∂h/∂x = [2(x-x_o), 2(y-y_o), 0]

L_f h = ∂h/∂x · f(x) = 0              (kinematic, drift 없음)

L_g h = ∂h/∂x · g(x)
      = [2(x-x_o)·cos(θ) + 2(y-y_o)·sin(θ),  0]
```

**핵심 관찰**: ω(각속도)는 h에 영향을 주지 않음 (L_g h의 2번째 성분 = 0).
따라서 CBF 제약은 v(선속도)에만 적용된다.

### 다중 장애물 QP 정식화

M개 장애물이 있을 때, QP는 M개의 선형 제약을 가진다:

```
min_u   ||u - u_mppi||² = (v - v_mppi)² + (ω - ω_mppi)²

s.t.    L_f h_1 + L_g h_1 · u + α_1 · h_1 ≥ 0    (장애물 1)
        L_f h_2 + L_g h_2 · u + α_2 · h_2 ≥ 0    (장애물 2)
        ...
        L_f h_M + L_g h_M · u + α_M · h_M ≥ 0    (장애물 M)
        u_min ≤ u ≤ u_max

행렬 형태:
  min_u  (u - u_mppi)^T (u - u_mppi)
  s.t.   A·u + b ≥ 0,   u_min ≤ u ≤ u_max

  A = [L_g h_1]     b = [L_f h_1 + α_1·h_1]
      [L_g h_2]         [L_f h_2 + α_2·h_2]
      [  ...  ]         [      ...         ]
      [L_g h_M]         [L_f h_M + α_M·h_M]
```

### 동역학 Differential Drive (5-state) Lie 미분

상태: x = [x, y, θ, v, ω], 제어: u = [F, τ] (힘, 토크)

```
f(x) = [v·cos(θ), v·sin(θ), ω, -μ_v·v, -μ_ω·ω]^T    (마찰 포함)

g(x) = [0    0  ]
       [0    0  ]
       [0    0  ]
       [1/m  0  ]
       [0    1/I]

h = (x - x_o)² + (y - y_o)² - r²

∂h/∂x = [2(x-x_o), 2(y-y_o), 0, 0, 0]

L_f h = ∂h/∂x · f(x)
      = 2(x-x_o)·v·cos(θ) + 2(y-y_o)·v·sin(θ)

L_g h = ∂h/∂x · g(x) = [0, 0]
```

**문제**: L_g h = 0 → 상대 차수 2! 제어 u가 ḣ에 직접 나타나지 않는다.

**해결 1 — HOCBF**: ḧ에서 u가 나타남.
```
ḣ = L_f h = 2(x-x_o)·v·cosθ + 2(y-y_o)·v·sinθ

ḧ = d/dt(ḣ) 에서 F, τ가 나타남 → 상대 차수 2 HOCBF 적용
```

**해결 2 — 가상 제어**: v, ω를 제어 입력으로 취급 (내부 루프가 v,ω를 추종).
이 경우 기구학 모델과 동일한 CBF를 사용할 수 있다.

### Ackermann 모델 Lie 미분

상태: x = [x, y, θ, δ], 제어: u = [v, δ_dot]

```
f(x) = [0, 0, 0, 0]^T

g(x) = [cos(θ)         0]
       [sin(θ)         0]
       [tan(δ)/L       0]     (L = 축거)
       [0               1]

h = (x - x_o)² + (y - y_o)² - r²

∂h/∂x = [2(x-x_o), 2(y-y_o), 0, 0]

L_f h = 0

L_g h = [2(x-x_o)·cos(θ) + 2(y-y_o)·sin(θ),  0]
```

**관찰**: Ackermann 기구학에서도 δ_dot는 h에 직접 영향을 주지 않는다.
v만이 CBF 제약에 관여하며, 이는 Differential Drive와 동일한 구조이다.

### KKT 조건 분석

CBF QP의 KKT (Karush-Kuhn-Tucker) 최적성 조건:

```
Lagrangian: L = ||u - u_mppi||² - Σ_j λ_j (L_g h_j · u + L_f h_j + α_j h_j)
                                 - μ_lo^T (u - u_min) - μ_hi^T (u_max - u)

KKT 조건:
  1. 정상성:    2(u* - u_mppi) - Σ_j λ_j · L_g h_j^T - μ_lo + μ_hi = 0
  2. 주요 실현: L_g h_j · u* + L_f h_j + α_j h_j ≥ 0,  ∀j
  3. 쌍대 실현: λ_j ≥ 0,  μ_lo ≥ 0,  μ_hi ≥ 0
  4. 상보성:    λ_j · (L_g h_j · u* + L_f h_j + α_j h_j) = 0,  ∀j

해석:
  - λ_j > 0:  j번째 CBF 제약이 활성 (등호 성립) → 경계에서 운행
  - λ_j = 0:  j번째 CBF 제약이 비활성 → 해당 장애물 무관
  - 정상성 조건에서: u* = u_mppi + (1/2)·Σ_j λ_j · L_g h_j^T + ...
    → u*는 u_mppi에서 CBF 그래디언트 방향으로 이동한 점
```

### QP 실현 불가능성 분석

QP가 **실현 불가능(infeasible)**한 경우:

```
상황 1: 상충하는 CBF 제약
  두 장애물 사이 좁은 통로에서:
    h_1 제약: v ≥ v_min1 (왼쪽 장애물에서 멀어져야)
    h_2 제약: v ≤ v_max2 (오른쪽 장애물에서 멀어져야)
    v_min1 > v_max2 이면 실현 불가능

상황 2: CBF 제약과 입력 제한 상충
    CBF 요구: v ≤ 0 (후진해야 안전)
    입력 제한: v ≥ 0 (후진 불가 로봇)
    → 실현 불가능

해결:
  → Optimal-Decay CBF (§9)로 이완
  → safety_margin을 줄여 제약 완화
  → 백업 제어 (u = 0) 사용
```

### 알고리즘 의사코드

```
Algorithm: CBF Safety Filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, u_mppi, obstacles, α, u_min, u_max

1. 각 장애물 j에 대해:
   h_j = ||p - p_obs_j||² - r_j²
   ∂h_j/∂x = 2·(p - p_obs_j) 확장
   L_f h_j = ∂h_j/∂x · f(x)
   L_g h_j = ∂h_j/∂x · g(x)

2. 활성 제약 판별:
   active_set = {j : h_j < d_threshold}    (가까운 장애물만)

3. If active_set = ∅:
     return u_mppi    (모든 장애물이 멀리 있음)

4. QP 구성:
     min_u  ||u - u_mppi||²
     s.t.   L_g h_j · u + L_f h_j + α·h_j ≥ 0,  ∀j ∈ active_set
            u_min ≤ u ≤ u_max

5. QP 풀기 (quadprog / scipy.optimize / 해석적)

6. If 실현 가능:
     return u_qp
   Else:
     return u_backup (예: 감속 또는 정지)
```

### 해석적 해 (1D QP)

v만 관련되므로 단순 1D 문제:
```
a = L_g h[0]                           (v의 계수)
b = L_f h + α·h = α·h                  (상수항)

제약: a·v + b ≥ 0

If a·v_mppi + b ≥ 0:
    u_safe = u_mppi                     (이미 안전)
Else:
    v_safe = -b/a = α·h / |L_g h[0]|   (경계로 투영)
    u_safe = [v_safe, ω_mppi]
```

### 구현

- **파일**: `cbf_safety_filter.py` (라인 13-243)
- **핵심 메서드**: `filter_control(state, u_mppi, u_min, u_max)`
- **반환**: `{filtered, correction_norm, barrier_values, min_barrier, optimization_success}`

### 언제 사용

- 확실한 안전 보장이 필요한 경우
- 단일 제어 출력에 대한 사후 필터링
- 볼록 장애물 + 알려진 동역학

---

## 4. Shield-MPPI (per-step 강제)

### 문제

QP Filter는 **최종 제어**만 수정하지만,
MPPI의 K개 **롤아웃 궤적** 자체가 안전하지 않으면
비용 평가가 왜곡된다.

### 핵심 아이디어

롤아웃의 **매 시간 스텝**에서 CBF 제약을 강제하여,
모든 K개 샘플 궤적이 안전하도록 보장한다.

### 수학적 정의

**v_ceiling 계산**:
```
L_g h[0] = 2(x-x_o)cos(θ) + 2(y-y_o)sin(θ)

If L_g h[0] < 0 (장애물에 접근 중):
    v_ceiling = α · h / |L_g h[0]|     (최대 허용 속도)
    v = min(v, v_ceiling)              (클리핑)
Else:
    통과 (멀어지고 있음)
```

### Shield vs Filter 비교

```
┌─────────────────────────────────────────────────────┐
│  Shield-MPPI (내부 안전)                              │
│                                                     │
│  ┌─ 롤아웃 ──────────────────┐                      │
│  │ t=0  t=1  t=2  ...  t=N │  ← 매 스텝 CBF 강제   │
│  │  ✓    ✓    ✓    ...   ✓  │                      │
│  └──────────────────────────┘                      │
│  → 모든 K 샘플 궤적이 안전                           │
│  → 비용 평가가 안전 궤적만 기반                       │
├─────────────────────────────────────────────────────┤
│  CBF Filter (외부 안전)                               │
│                                                     │
│  ┌─ 롤아웃 ──────────────────┐                      │
│  │ t=0  t=1  t=2  ...  t=N │  ← 안전 미보장        │
│  │  ✓    ✗    ✓    ...   ✗  │                      │
│  └──────────────────────────┘                      │
│  → 최종 u*만 QP로 수정                               │
│  → 비용 평가가 위험 궤적 포함                         │
└─────────────────────────────────────────────────────┘
```

### 알고리즘

```
_shielded_rollout(state, controls):
  For k = 1 to K:
    x = state
    For t = 0 to N-1:
      For each obstacle:
        L_g_h0 = 2(x[0]-ox)cos(x[2]) + 2(x[1]-oy)sin(x[2])
        h = (x[0]-ox)² + (x[1]-oy)² - r_eff²
        if L_g_h0 < 0 and h > 0:   (접근 중 + 아직 안전)
          v_max = α · h / |L_g_h0|
          controls[k,t,0] = min(controls[k,t,0], v_max)
      x = dynamics(x, controls[k,t])
```

### 안전 보장 증명 스케치

**정리**: Shield-MPPI에서 롤아웃의 매 스텝에 CBF 제약을 강제하면,
모든 K개 샘플 궤적이 안전 집합 C 내에 유지된다.

```
증명 스케치:

전제:
  - h(x_0) > 0 (초기 상태가 안전)
  - 매 스텝 t에서: v ≤ α·h(x_t) / |L_g h[0]| 강제

귀납법:
  Base: h(x_0) > 0 (가정)

  Step: h(x_t) > 0이면, 이산시간 CBF 조건에 의해:
    h(x_{t+1}) ≈ h(x_t) + ḣ·dt
               ≥ h(x_t) + (-α·h(x_t))·dt    (CBF 제약에 의해 ḣ ≥ -α·h)
               = h(x_t)·(1 - α·dt)
               > 0                            (α·dt < 1일 때)

  따라서 h(x_t) > 0이 모든 t에서 성립.  ∎

주의: 이산화 오차로 인해 α·dt < 1 조건이 필요.
  dt = 0.1, α = 0.5이면 α·dt = 0.05 << 1 → 안전.
```

### 벡터화 배치 구현 주의사항

```
성능 최적화:
  K = 1024 샘플, N = 30 스텝, M = 5 장애물일 때:
  순차 루프: K × N × M = 153,600 CBF 계산 → ~50ms
  벡터화:   (K, N, M) 텐서 연산 → ~2ms (25x 가속)

벡터화 핵심:
  1. 모든 K개 궤적을 (K, N, state_dim) 텐서로 유지
  2. 장애물 좌표를 (1, 1, M, 2)로 브로드캐스트
  3. L_g h, h 를 (K, N, M) 텐서로 한번에 계산
  4. v_ceiling = α·h / |L_g h| 를 텐서 연산으로 계산
  5. np.minimum(v, v_ceiling.min(axis=-1))로 최소값 클리핑
```

### 개입율 vs 추적 오차 분석

```
개입율(intervention rate):  Shield가 속도를 수정한 비율

┌──────────────────────────────────────────────┐
│ RMSE(m) ↑                                     │
│  0.6  │                               ●       │
│  0.5  │                          ●            │
│  0.4  │                    ●                  │
│  0.3  │            ● ●                        │
│  0.2  │      ●                                │
│  0.15 │  ●                                    │
│  0.1  │●                                      │
│       ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──→     │
│       0  10 20 30 40 50 60 70 80 90 100       │
│                     Intervention Rate (%)      │
└──────────────────────────────────────────────┘

관찰:
  - 개입율 < 20%: 추적 성능 거의 영향 없음
  - 개입율 20~50%: 선형적 성능 저하
  - 개입율 > 50%: 급격한 성능 저하 (장애물이 너무 가까이)
```

### 비교: Shield vs Safety Layer (Dalal et al., 2018)

```
┌───────────────────┬─────────────────────┬──────────────────────┐
│ 특성               │ Shield-MPPI          │ Safety Layer          │
├───────────────────┼─────────────────────┼──────────────────────┤
│ 적용 위치          │ 롤아웃 내부 (매 스텝) │ 정책 출력 후 (1회)    │
│ 안전 범위          │ 전체 K 샘플 궤적     │ 최종 제어만            │
│ 비용 평가 영향     │ 안전 궤적 기반 비용   │ 비용 왜곡 가능         │
│ 계산 복잡도        │ O(K·N·M)            │ O(M) per step         │
│ 학습 필요          │ 불필요               │ 불필요                │
│ 미분 가능          │ 아니오               │ 예 (QP layer)         │
│ 다중 모달 지원     │ 예 (K 샘플 분산)     │ 아니오                │
│ 적합한 방법론      │ MPPI (샘플링)        │ RL, MPC 등 범용       │
└───────────────────┴─────────────────────┴──────────────────────┘
```

### 구현

- **파일**: `shield_mppi.py` (라인 28-390)
- **핵심 메서드**:
  - `_shielded_rollout()`: (trajectories, shielded_controls, shield_info)
  - `_cbf_shield_batch()`: 벡터화된 배치 클리핑
- **추가 info**: `intervention_rate`, `mean_vel_reduction`, `total_interventions`

---

## 5. Adaptive Shield (α 적응)

### 문제

고정 α는 모든 상황에서 동일한 보수성을 적용한다.
장애물에 가까울 때는 더 보수적이어야 하고,
멀 때는 덜 보수적이어야 한다.

### 핵심 아이디어

거리와 속도에 따라 α를 동적으로 조절한다:
가까울수록/빠를수록 더 보수적.

### 수학적 정의

```
σ(x) = 1 / (1 + exp(-k · (d - d_safe)))     (시그모이드)

α(d, v) = α_base · [α_dist + (1 - α_dist) · σ(k · (d - d_safe))]
                    / (1 + α_vel · |v|)
```

**해석**:

```
d >> d_safe (멀리):  σ → 1  → α ≈ α_base         (느슨)
d << d_safe (가깝):  σ → 0  → α ≈ α_base · α_dist (보수적)
|v| 클 때:                  → α가 더 감소           (보수적)
```

```
α 값
  ↑ α_base ──────────╮
  │                   ╲
  │                    ╲─── 시그모이드 전이
  │                     ╲
  │ α_base·α_dist ───────── 최소값
  ├─────────┬───────────→ 거리 d
         d_safe
```

### 알고리즘 의사코드

```
Algorithm: Adaptive Shield-MPPI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, controls U[K,N,2], obstacles, params

For k = 1 to K:
  x_curr = x
  For t = 0 to N-1:
    For each obstacle j:
      1. 거리 계산:
         d_j = ||p_curr - p_obs_j|| - r_j
         v_approach_j = (p_curr - p_obs_j)·v_curr / ||p_curr - p_obs_j||

      2. 적응 α 계산:
         σ_j = 1 / (1 + exp(-k_dist · (d_j - d_safe)))
         α_j = α_base · [α_dist + (1-α_dist)·σ_j] / (1 + α_vel·|v_approach_j|)

      3. CBF 클리핑 (α_j 사용):
         h_j = d_j² + 2·r_j·d_j     (= ||p-p_obs||² - r²을 거리로 표현)
         L_g_h = 2(x-x_o)cosθ + 2(y-y_o)sinθ
         if L_g_h < 0 and h_j > 0:
           v_max_j = α_j · h_j / |L_g_h|
           U[k,t,0] = min(U[k,t,0], v_max_j)

    x_curr = dynamics(x_curr, U[k,t])
```

### α(d, v) 표면 시각화

```
α(d,v) 표면 (ASCII 3D 투영):

  α ↑
 0.3│  ╲  ╲  ╲  ╲       d = 2.0 (멀리)
    │   ╲  ╲  ╲  ╲
 0.2│    ╲  ╲  ╲  ╲     d = 1.0 (중간)
    │     ╲  ╲  ╲  ╲
 0.1│      ╲  ╲  ╲  ╲   d = 0.3 (가까이)
    │       ╲  ╲  ╲  ╲
0.03│        ╲  ╲  ╲  ╲  d = 0.1 (매우 가까이)
    ├────┼────┼────┼────┼──→ |v|
    0   0.5  1.0  1.5  2.0

  거리 가까울수록 (d↓): α 감소 → 보수적 (속도 제한 강화)
  속도 빠를수록  (v↑): α 감소 → 보수적 (정지 거리 확보)

  d=2.0, v=0:   α ≈ 0.30  (최대, 느슨)
  d=0.1, v=2.0: α ≈ 0.01  (최소, 매우 보수적)
```

### 적응 α의 안정성 분석

α(d, v)가 시간에 따라 변하므로, 전방 불변성 증명이 달라진다:

```
ḣ ≥ -α(d,v) · h

여기서 α(d,v) ∈ [α_min, α_max]:
  α_min = α_base · α_dist / (1 + α_vel · v_max)
  α_max = α_base

전방 불변성 조건:
  h(x_{t+1}) ≥ h(x_t) · (1 - α_max · dt)

α_max · dt < 1이면 전방 불변성 보장.
  예: α_base = 0.3, dt = 0.1 → α·dt = 0.03 << 1  ✓

핵심: α가 시변이더라도, 유계(bounded)이고 α·dt < 1이면
전방 불변성이 유지된다.
```

### 튜닝 절차 (Step-by-Step)

```
1. α_base 설정 (0.2~0.5):
   장애물 없는 환경에서 추적 성능 확인.
   α_base가 클수록 장애물 근처에서 빠르게 감속.
   → 추적 오차가 수용 가능한 범위 내 최대값 선택.

2. d_safe 설정 (0.3~1.0m):
   장애물과의 최소 허용 거리.
   로봇 크기 + 안전 여유 반영.
   → d_safe = 로봇반경 + 0.1~0.3m.

3. k_dist 설정 (3.0~10.0):
   시그모이드 전이 기울기.
   k 클수록 d_safe에서 급격한 전환.
   → k_dist = 5.0 (기본값)이 대부분 적절.

4. α_dist 설정 (0.05~0.2):
   최소 α 비율. 작을수록 가까이서 더 보수적.
   → α_dist = 0.1 (기본값), 고속 로봇은 0.05.

5. α_vel 설정 (0.3~1.0):
   속도 감쇠 강도. 클수록 고속에서 더 보수적.
   → α_vel = 0.5 (기본값), v_max > 2m/s이면 0.8.

6. 검증:
   다양한 시나리오 (정면 접근, 측면 통과, 좁은 통로)에서
   intervention_rate 확인. 목표: < 30%.
```

### 구현

- **파일**: `adaptive_shield_mppi.py` (라인 46-181)
- **부모 클래스**: ShieldMPPIController
- **파라미터**:

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| alpha_base | 0.3 | 최대 α (멀리서) |
| alpha_dist | 0.1 | 최소 α 비율 (가까이서) |
| alpha_vel | 0.5 | 속도 감쇠 계수 |
| k_dist | 5.0 | 시그모이드 기울기 |
| d_safe | 0.5 | 기준 안전 거리 |

---

## 6. C3BF (Collision Cone)

### 문제

정적 CBF (h = dist² - r²)는 **접근 방향**을 고려하지 않는다.
장애물에서 멀어지고 있어도 페널티를 부과한다.

### 핵심 아이디어

**충돌 원뿔(collision cone)** 기하학을 사용하여,
속도 벡터가 충돌 원뿔 밖이면 안전하다고 판단한다.

### 수학적 정의

**충돌 원뿔 각도**:
```
φ_safe = arccos(√(||p_rel||² - R²) / ||p_rel||)
```

여기서 `p_rel = p_robot - p_obs`, R = 장애물 반경 + 마진.

**CBF 정의**:
```
h_c3bf = ⟨p_rel, v_rel⟩ + ||p_rel|| · ||v_rel|| · cos(φ_safe)
```

**직관**:
- `⟨p_rel, v_rel⟩ > 0`: 멀어지고 있음 → 안전
- `⟨p_rel, v_rel⟩ < 0`: 가까워지고 있음 → 방향 확인 필요

```
         ╱ 충돌 원뿔 ╲
        ╱              ╲
       ╱    ○ 장애물     ╲
      ╱     R             ╲
     ╱       φ_safe        ╲
────●──────────────────────────
   로봇  v_rel →

   속도가 원뿔 안: 충돌 위험 (h < 0)
   속도가 원뿔 밖: 안전 (h > 0)
```

### 충돌 원뿔 기하학 유도

```
로봇 위치: p = [x, y]
장애물 위치: p_obs = [x_o, y_o], 반경 R (= r_obs + safety_margin)

상대 위치: p_rel = p - p_obs
상대 거리: d = ||p_rel||

충돌 원뿔 반각 (half-angle):
  sin(φ_safe) = R / d
  cos(φ_safe) = √(d² - R²) / d
  φ_safe = arcsin(R / d)

기하학적 의미:
  로봇에서 장애물 원의 접선이 LoS와 이루는 각도가 φ_safe.
  속도 벡터가 이 원뿔 밖이면 장애물에 충돌하지 않는다.
```

```
상세 기하 도식:

                    접선
                   ╱
                  ╱  φ_safe
                 ╱───────────╱ LoS (Line of Sight)
                ╱           ╱
               ╱   ┌──────╱──┐
              ╱    │  ○  ╱   │  ← 장애물 (반경 R)
             ╱     │    ╱    │
            ╱      └──╱─────┘
           ╱         ╱
  로봇 ●────────────╱
                   접선
         ╲
          ╲  v_rel (속도 벡터)
           ╲
            ↘
  v_rel이 원뿔 안: h < 0 (충돌 경로)
  v_rel이 원뿔 밖: h > 0 (안전)
```

### 상대 속도 계산 (동적 장애물)

정적 장애물의 경우 v_obs = 0이지만, 동적 장애물에서는:

```
v_rel = v_robot - v_obs

로봇 속도: v_robot = [v·cos(θ), v·sin(θ)]
장애물 속도: v_obs = [vx_obs, vy_obs]

상대 속도: v_rel = v_robot - v_obs
```

### CBF 미분 유도

```
h_c3bf = ⟨p_rel, v_rel⟩ + ||p_rel|| · ||v_rel|| · cos(φ_safe)

φ_safe가 거리에 의존하므로, cos(φ_safe) = √(d² - R²) / d:

h_c3bf = ⟨p_rel, v_rel⟩ + ||v_rel|| · √(||p_rel||² - R²)

ḣ_c3bf = d/dt[⟨p_rel, v_rel⟩] + d/dt[||v_rel|| · √(||p_rel||² - R²)]

       = ⟨v_rel, v_rel⟩ + ⟨p_rel, a_rel⟩
         + d/dt[||v_rel||] · √(d² - R²)
         + ||v_rel|| · ⟨p_rel, v_rel⟩ / √(d² - R²)

여기서 a_rel = dv_rel/dt는 상대 가속도.
```

### 알고리즘 의사코드

```
Algorithm: C3BF Cost Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: trajectories (K, N, 3), obstacles [(x,y,r),...], weight

For k = 1 to K:
  cost_k = 0
  For t = 0 to N-1:
    For each obstacle j:
      1. 상대 벡터:
         p_rel = p[k,t] - p_obs_j
         d = ||p_rel||

      2. 로봇 속도:
         v_robot = [v·cos(θ), v·sin(θ)]
         v_rel = v_robot - v_obs_j     (정적: v_obs = 0)

      3. 원뿔 각도:
         if d ≤ R_j: cost_k += rejection_cost; continue
         cos_phi = √(d² - R_j²) / d

      4. C3BF 값:
         h = ⟨p_rel, v_rel⟩ + ||v_rel|| · d · cos_phi

      5. 접근 판정:
         if ⟨p_rel, v_rel⟩ > 0:
           continue     (멀어지고 있음 → 비용 0)

      6. 이산시간 위반:
         h_next = C3BF(x[k,t+1], obs_j)
         violation = max(0, -[h_next - (1-α)·h])
         cost_k += weight · violation

  costs[k] = cost_k
```

### 비교: C3BF vs Velocity Obstacle vs Dynamic Window

```
┌────────────────┬───────────────┬───────────────┬────────────────┐
│ 특성            │ C3BF           │ Velocity Obs.  │ Dynamic Window │
├────────────────┼───────────────┼───────────────┼────────────────┤
│ 수학적 기초     │ CBF (Lyapunov)│ 기하학적      │ 탐색 기반       │
│ 최적성 보장     │ 있음 (CBF QP) │ 없음          │ 없음            │
│ 동적 장애물     │ 자연스럽게    │ 자연스럽게    │ 예측 필요       │
│ 다체 로봇       │ 확장 가능     │ RVO 확장      │ 어려움          │
│ MPPI 통합       │ 비용 함수     │ 별도 필터     │ 별도 필터       │
│ 연속시간 보장   │ 있음          │ 없음          │ 없음            │
│ 계산 복잡도     │ O(K·M)       │ O(M)          │ O(v·ω 격자)    │
│ 비홀로노믹      │ 고려 필요     │ 고려 안됨     │ 고려됨          │
└────────────────┴───────────────┴───────────────┴────────────────┘
```

### 정적 CBF와 차이

```
정적 CBF:  장애물 옆을 스쳐 지나갈 때도 페널티 (거리만 고려)
C3BF:      옆으로 지나가면 안전 (속도 방향 고려)
```

### 구현

- **파일**: `c3bf_cost.py:CollisionConeCBFCost` (라인 24-195)
- **특징**: 멀어지고 있으면 비용 = 0

---

## 7. DPCBF (Dynamic Parabolic)

### 문제

C3BF는 원형 원뿔만 고려한다. 실제로는 정면 접근 vs 측면 통과에서
필요한 안전 마진이 다르다.

### 핵심 아이디어

**시선각(LoS: Line-of-Sight)** 좌표계에서 방향별 파라볼릭 안전 프로파일을 적용한다.
정면에서는 넓은 마진, 측면에서는 좁은 마진.

### 수학적 정의

**LoS 좌표계**:
- β: 상대 속도와 LoS 방향의 각도
- r: 장애물까지의 거리

**동적 안전 반경**:
```
r_safe(β) = R_eff + a(v) · exp(-β² / (2σ²))
```

여기서:
```
a(v) = a_base + a_vel · max(0, -v_approach)
```

- `v_approach < 0` (접근 중): a 증가 → 안전 마진 확대
- `β ≈ 0` (정면): exp ≈ 1 → 최대 마진
- `β ≈ π/2` (측면): exp ≈ 0 → 최소 마진

```
            ← 넓은 마진 (정면)
          ╱───────────────╲
        ╱                   ╲
       │    ○ 장애물          │
       │    R_eff             │
        ╲                   ╱ ← 좁은 마진 (측면)
          ╲───────────────╱
```

### LoS 좌표계 유도

```
Step 1: 상대 위치 벡터
  p_rel = p_obs - p_robot = [Δx, Δy]
  d = ||p_rel||

Step 2: LoS 방향각
  θ_los = atan2(Δy, Δx)       (로봇에서 장애물까지의 방향)

Step 3: 로봇 속도의 LoS 분해
  v_robot = [v·cos(θ), v·sin(θ)]

  v_approach = v_robot · (p_rel / d)
             = v · cos(θ - θ_los)       (LoS 방향 속도 성분)

  v_lateral  = v · sin(θ - θ_los)       (LoS 수직 속도 성분)

Step 4: 접근 각도 β
  β = θ - θ_los                         (로봇 진행 방향과 LoS의 차이)
  β = 0:   정면 접근
  β = π/2: 측면 통과
  β = π:   멀어지는 중

Step 5: 접근 속도 (부호 규약)
  v_approach = v · cos(β)
  v_approach > 0: 접근 중
  v_approach < 0: 멀어지는 중

  ※ 코드에서는 v_approach = -⟨v_robot, p_rel/d⟩ 로 계산하는 경우도 있음
     (부호 규약이 논문마다 다름)
```

### σ_beta 파라미터 분석

σ_beta는 방향별 안전 마진의 폭을 결정하는 핵심 파라미터이다.

```
r_safe(β) = R_eff + a(v) · exp(-β² / (2σ²))

σ_beta에 따른 안전 마진 프로파일:

r_safe ↑
  R+a │ ●                              σ=0.3 (좁음, 정면만)
      │ ●●                             σ=0.5 (중간)
      │ ●●●●                           σ=1.0 (넓음, 측면까지)
      │ ●●●●●●
      │  ●●●●●●●●
      │    ●●●●●●●●●●
      │       ●●●●●●●●●●●●
  R   │──────────●●●●●●●●●●●●●●●●── R_eff (최소 마진)
      ├──┼──┼──┼──┼──┼──┼──┼──┼──→ β (rad)
      0  0.5 1.0 1.5 2.0 2.5 3.0

σ_beta 작을 때 (0.2~0.3):
  정면 접근만 큰 마진 → 측면 통과가 자유로움
  좁은 통로 통과에 유리

σ_beta 클 때 (1.0~2.0):
  넓은 범위에서 큰 마진 → 전체적으로 보수적
  안전성 우선 시나리오에 적합

권장: σ_beta = 0.5 ~ 1.0
```

### 알고리즘 의사코드

```
Algorithm: DPCBF Cost Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: trajectories (K, N, 3), obstacles, params

For k = 1 to K:
  cost_k = 0
  For t = 0 to N-1:
    For each obstacle j:
      1. 상대 위치:
         Δx, Δy = p_obs_j - p[k,t,:2]
         d = √(Δx² + Δy²)
         θ_los = atan2(Δy, Δx)

      2. 접근 각도 및 속도:
         β = θ[k,t] - θ_los
         v_approach = v[k,t] · cos(β)

      3. 동적 안전 반경:
         a = a_base + a_vel · max(0, v_approach)
         r_safe = R_j + a · exp(-β² / (2σ²))

      4. 거리 기반 DPCBF:
         h = d - r_safe
         if h < 0:
           violation = |h|
         else:
           h_next = d_next - r_safe_next
           violation = max(0, -[h_next - (1-α)·h])

      5. 비용 누적:
         cost_k += weight · violation

  costs[k] = cost_k
```

### 다중 장애물 처리

```
여러 장애물이 있을 때, DPCBF의 핵심은 각 장애물에 대해
독립적으로 r_safe를 계산하는 것이다:

장애물 1: r_safe_1(β_1) = R_1 + a_1(v) · exp(-β_1² / 2σ²)
장애물 2: r_safe_2(β_2) = R_2 + a_2(v) · exp(-β_2² / 2σ²)
...

각 장애물의 β_i와 v_approach_i가 다르므로,
같은 시점에서도 방향마다 다른 안전 마진이 적용된다.

주의: 장애물이 밀집한 경우, 한 장애물을 피하려다
다른 장애물의 DPCBF를 위반할 수 있다.
→ 이 경우 Shield-MPPI + DPCBF 결합을 권장한다.
```

### 구현

- **파일**: `dpcbf_cost.py:DynamicParabolicCBFCost` (라인 37-252)
- **파라미터**: `a_base`, `a_vel`, `sigma_beta`

---

## 8. Neural CBF (학습 기반)

### 문제

해석적 CBF (h = dist² - r²)는 **원형/볼록 장애물**에만 적용 가능하다.
비볼록 장애물(L자형, 복잡한 형상)에는 h(x) 설계가 어렵다.

### 핵심 아이디어

MLP로 h(x)를 **학습**하여 임의 형상의 장애물에 대한 barrier function을 자동 생성한다.

### 수학적 정의

**NeuralCBFNetwork**:
```
h_θ(x) = network(x)     ← MLP: R^{nx} → R

아키텍처: x → Linear(128) → Softplus → Linear(128) → Softplus
                → Linear(64) → Softplus → Linear(1) → output_scale · tanh
```

**학습 데이터**:
```
{(x_i, label_i)} where label_i ∈ {safe, unsafe, boundary}

safe:      h(x) > 0 으로 학습
unsafe:    h(x) < 0 으로 학습
boundary:  h(x) ≈ 0 으로 학습
```

**손실 함수**:
```
L = L_safe + L_unsafe + L_boundary + L_lipschitz

L_safe     = Σ_{x∈safe}     max(0, -h_θ(x))²
L_unsafe   = Σ_{x∈unsafe}   max(0, h_θ(x))²
L_boundary = Σ_{x∈boundary} h_θ(x)²
L_lipschitz = λ · E[||∇_x h_θ(x)||²]
```

**Lipschitz 정규화**: h(x)의 그래디언트 크기를 제한하여 매끄러움을 보장한다.
이는 CBF 조건의 Lie 미분 계산에서 중요하다.

### Autograd Lie 미분

PyTorch의 자동 미분으로 Lie 미분을 계산한다:
```
∂h/∂x = torch.autograd.grad(h_θ(x), x)

L_f h = (∂h/∂x) · f(x)
L_g h = (∂h/∂x) · g(x)
```

→ 어떤 동역학 모델에서든 자동 계산 가능

### 네트워크 아키텍처 상세

```
NeuralCBFNetwork 아키텍처 (ASCII 다이어그램):

  입력 x ∈ R^{nx}
       │
       ▼
  ┌──────────────────┐
  │ Linear(nx, 128)  │  가중치: W₁ ∈ R^{128×nx}
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │   Softplus(β=1)  │  활성화: log(1 + exp(x))
  └────────┬─────────┘  (ReLU보다 매끄러움 → ∇h 연속)
           │
  ┌────────▼─────────┐
  │ Linear(128, 128) │  가중치: W₂ ∈ R^{128×128}
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │   Softplus(β=1)  │
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ Linear(128, 64)  │  가중치: W₃ ∈ R^{64×128}
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │   Softplus(β=1)  │
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  Linear(64, 1)   │  가중치: W₄ ∈ R^{1×64}
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │ output_scale ×   │  h = scale · tanh(z)
  │    tanh(z)       │  → h ∈ [-scale, +scale]
  └────────┬─────────┘
           │
       h_θ(x) ∈ R

  총 파라미터: nx×128 + 128 + 128×128 + 128 + 128×64 + 64 + 64×1 + 1
             = ~25,000 (nx=3일 때)

  Softplus 선택 이유:
    ReLU: ∂h/∂x가 불연속 → Lie 미분 계산 불안정
    Sigmoid: vanishing gradient 문제
    Softplus: 매끄럽고(C^∞), ReLU와 유사한 비선형성
    → autograd Lie 미분이 안정적
```

### 학습 절차 알고리즘

```
Algorithm: Neural CBF Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: obstacles (형상 정의), domain bounds, hyperparams

Phase 1: 데이터 생성
  1. 도메인 내 N_total 점을 균일 샘플링
  2. 각 점 x_i에 대해 라벨링:
     - safe:     모든 장애물에서 margin 이상 떨어짐
     - unsafe:   장애물 내부
     - boundary: 장애물 표면에서 margin 이내
  3. 클래스 균형 조정:
     boundary 데이터가 부족하면, 장애물 표면 근처에서 추가 샘플링

Phase 2: 학습 (supervised)
  For epoch = 1 to max_epochs:
    For each mini-batch B:
      1. Forward: h_θ(x_i)  ∀x_i ∈ B
      2. Loss 계산:
         L_safe     = (1/|B_s|) · Σ max(0, -h_θ(x))²     (x ∈ safe)
         L_unsafe   = (1/|B_u|) · Σ max(0, h_θ(x))²      (x ∈ unsafe)
         L_boundary = (1/|B_b|) · Σ h_θ(x)²               (x ∈ boundary)
         L_lip      = λ · (1/|B|) · Σ ||∇_x h_θ(x)||²
         L = w_s·L_safe + w_u·L_unsafe + w_b·L_boundary + L_lip
      3. Backward: ∂L/∂θ
      4. Optimizer step (Adam, lr=1e-3)

  Early stopping: validation loss 기준

Phase 3: 검증
  1. 테스트 도메인에서 h_θ 등고선 시각화
  2. h_θ(x) ≈ 0 등고선이 장애물 경계와 일치하는지 확인
  3. ∇h 크기가 경계에서 충분한지 확인 (CBF 조건 실현 가능성)
```

### 손실 함수 그래디언트 분석

```
각 손실 항의 그래디언트:

∂L_safe/∂θ:
  x ∈ safe에서 h_θ(x) < 0이면 (잘못 분류):
    ∂/∂θ[max(0, -h_θ)²] = -2·max(0, -h_θ) · ∂h_θ/∂θ
    → h_θ를 증가시키는 방향 (안전 영역에서 양수가 되도록)

∂L_unsafe/∂θ:
  x ∈ unsafe에서 h_θ(x) > 0이면 (잘못 분류):
    ∂/∂θ[max(0, h_θ)²] = 2·max(0, h_θ) · ∂h_θ/∂θ
    → h_θ를 감소시키는 방향 (위험 영역에서 음수가 되도록)

∂L_lip/∂θ:
  ∂/∂θ[||∇_x h_θ||²] = 2·∇_x h_θ · ∂(∇_x h_θ)/∂θ
  → ∇h의 크기를 줄이는 방향 (매끄러움 강제)
  → 이중 미분 필요: torch.autograd.grad(h, x) → grad(grad_norm², θ)

균형 가중치 권장:
  w_safe = 1.0, w_unsafe = 1.0, w_boundary = 5.0, λ = 0.01
  boundary 가중치를 높여 0-등고선의 정확도를 보장한다.
```

### 일반화 분석: h_θ(x)의 외삽 성능

```
Neural CBF의 일반화 한계:

1. 훈련 도메인 내부 (보간): 우수
   - 장애물 형상을 정확히 학습
   - 0-등고선이 실제 경계에 근접

2. 훈련 도메인 외부 (외삽): 위험
   - MLP는 훈련 영역 밖에서 예측 불가
   - tanh 출력: h → ±scale로 포화 → 거짓 안전/위험 판정

3. 새로운 장애물: 재학습 필요
   - 해석적 CBF는 장애물 위치만 업데이트하면 되지만
   - Neural CBF는 새 데이터로 전체 재학습 또는 fine-tuning 필요

대응 전략:
  - 훈련 도메인을 운용 영역보다 넓게 설정 (+20%)
  - output_scale을 적절히 설정 (너무 크면 경계가 불명확)
  - 주기적 재학습 (online fine-tuning)
  - 해석적 CBF를 fallback으로 유지
```

### 비교: Neural CBF vs SOS (Sum of Squares) CBF

```
┌──────────────────┬─────────────────────┬──────────────────────┐
│ 특성              │ Neural CBF           │ SOS CBF               │
├──────────────────┼─────────────────────┼──────────────────────┤
│ 표현력            │ 임의 함수 (MLP)      │ 다항식 (차수 제한)     │
│ 비볼록 대응       │ 자연스럽게           │ 고차 다항식 필요       │
│ 안전 보장         │ 학습 정확도에 의존   │ 수학적 증명 가능       │
│ 계산 (오프라인)   │ SGD 학습 (~분)       │ SDP 풀기 (~시간)      │
│ 계산 (온라인)     │ MLP forward O(1)    │ 다항식 평가 O(1)      │
│ Lie 미분          │ Autograd (자동)      │ 해석적 (수동)         │
│ 확장성            │ 고차원 가능          │ 차원 저주             │
│ 데이터 필요       │ 예 (샘플 라벨)       │ 아니오 (해석적)       │
│ 적합 상황         │ 복잡 형상, 고차원    │ 저차원, 보장 필요     │
└──────────────────┴─────────────────────┴──────────────────────┘
```

### 데이터 효율성 분석

```
학습 데이터 양 vs 분류 정확도:

정확도(%)↑
  100 │                    ●────────●────────●
      │               ●
   95 │          ●
      │      ●
   90 │   ●
      │  ●
   85 │ ●
      │●
   80 ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──→ 데이터 수
      100 500 1K 2K 5K 10K 20K 50K 100K

관찰:
  - ~1K 데이터: ~90% 정확도 (단순 형상에 충분)
  - ~5K 데이터: ~97% 정확도 (복잡 형상에 적합)
  - ~10K+ 데이터: 포화 (~99%)

데이터 효율 향상 전략:
  1. Active learning: h ≈ 0 근처에서 집중 샘플링
  2. 경계 증폭: boundary 데이터 비율 높이기 (30%)
  3. 데이터 증강: 대칭 변환 활용
```

### 두 가지 사용 방법

1. **NeuralBarrierCost**: ControlBarrierCost의 drop-in 대체
   - h(x) = NeuralCBFNetwork(x) 사용
   - `neural_cbf_cost.py`

2. **NeuralCBFSafetyFilter**: QP 필터의 drop-in 대체
   - autograd로 Lie 미분 계산
   - `neural_cbf_filter.py` (라인 36-210)
   - `_compute_lie_derivatives_neural()`: (L_f h, L_g h, h) 반환

### 구현

- **학습**: `learning/neural_cbf_trainer.py:NeuralCBFTrainer`
  - `NeuralCBFNetwork` (라인 18-80): Softplus 활성화, Kaiming 초기화
- **비용**: `neural_cbf_cost.py`
- **필터**: `neural_cbf_filter.py`

### 언제 사용

- 비볼록 장애물 (L자, 미로 등)
- 장애물 형상을 해석적으로 표현하기 어려운 경우
- 데이터에서 안전 영역을 학습해야 하는 경우

---

## 9. Optimal-Decay CBF (이완형)

### 문제

표준 CBF QP는 **실현 불가능(infeasible)**할 수 있다.
특히 제어 제한이 있을 때, CBF 제약과 입력 제약이 동시에 만족 불가능한 경우가 있다.

### 핵심 아이디어

이완 변수 ω ∈ [0, 1]를 도입하여 CBF 제약을 "부드럽게" 완화한다.
ω = 1이면 완전한 CBF, ω = 0이면 제약 없음. 항상 실현 가능하다.

### 수학적 정의

```
min_{u, ω}   ||u - u_nom||² + p_sb · (ω - 1)²

s.t.   L_f h + L_g h · u + α · ω · h ≥ 0
       u_min ≤ u ≤ u_max
       0 ≤ ω ≤ 1
```

**ω 해석**:
```
ω = 1:  완전한 CBF 강제 (표준 QP와 동일)
ω < 1:  완화된 제약 (graceful degradation)
ω = 0:  제약 없음 (제약이 0 · h = 0 → 자동 만족)
```

**왜 항상 실현 가능한가**: ω = 0으로 설정하면 CBF 제약이
`L_f h + L_g h · u ≥ 0`이 되는데, u_nom이 이를 만족하지 않더라도
ω 자체가 자유 변수이므로 해가 항상 존재한다.

### KKT 분석: ω = 0의 실현 가능성

```
확장 QP:
  변수: z = [u; ω] ∈ R^{m+1}
  목적: ||u - u_nom||² + p_sb·(ω - 1)²
  제약: L_f h + L_g h · u + α·ω·h ≥ 0
        u_min ≤ u ≤ u_max
        0 ≤ ω ≤ 1

z = [u_nom; 0]을 대입:
  L_f h + L_g h · u_nom + α·0·h = L_f h + L_g h · u_nom

  이 값이 ≥ 0이 아니더라도, ω를 조정하여:
  L_f h + L_g h · u + α·ω·h ≥ 0

  최악의 경우 ω = 0이면:
    제약: L_f h + L_g h · u ≥ 0
    이는 u의 방향만으로 만족 가능 (제어 범위 내에서)

  만약 그것도 불가능하면 (극단적 상황):
    ω = 0은 CBF 제약을 α·0·h = 0 항으로 무효화
    → L_f h + L_g h · u ≥ 0만 남음
    → u = 0이 이를 만족시키면 (kinematic: L_f h = 0 → 0 ≥ 0 ✓)

  따라서 kinematic 모델에서 (u=0, ω=0)은 항상 실현 가능.  ∎
```

### Slack Variable 방법과의 비교

```
┌─────────────────────┬─────────────────────┬──────────────────────┐
│ 특성                 │ Optimal-Decay (ω)    │ Slack Variable (δ)   │
├─────────────────────┼─────────────────────┼──────────────────────┤
│ 정식화               │ α·ω·h (ω 곱)        │ L_g h·u + b ≥ -δ     │
│ 물리적 의미          │ CBF 감쇠율 조절      │ 제약 위반 허용량      │
│ ω/δ 범위            │ [0, 1]               │ [0, ∞)               │
│ 페널티               │ p·(ω-1)²            │ p·δ² 또는 p·δ        │
│ 안전 해석            │ ω=1이면 완전 안전    │ δ=0이면 완전 안전     │
│ 기본 제약 유지       │ ω→0: 제약 소멸       │ δ→∞: 제약 소멸       │
│ 실현 가능성          │ 항상 (ω=0)           │ 항상 (δ=∞)           │
│ 장점                 │ graceful degradation │ 직관적 위반량         │
│ 단점                 │ 비선형 제약 (ω·h)    │ 큰 δ가 필요할 수 있음 │
└─────────────────────┴─────────────────────┴──────────────────────┘
```

### ω 해석: 안전 신뢰도 지표

```
ω 값을 안전 신뢰도로 활용할 수 있다:

ω ∈ [0, 1] 해석:
  ω = 1.0:  완전히 안전한 제어 가능 (표준 CBF 만족)
  ω = 0.8:  80% 수준의 CBF 유지 (약간의 이완)
  ω = 0.5:  50% 수준 (상당한 이완, 안전 저하)
  ω = 0.1:  거의 제약 없음 (매우 위험한 상황)
  ω = 0.0:  CBF 완전 포기 (비상 상황)

응용:
  1. 경보 시스템: ω < 0.5이면 경고 발생
  2. 속도 제한: ω < 0.3이면 최대 속도를 v_max · ω로 줄임
  3. 재계획 트리거: ω < 0.2이면 전역 경로 재계획 요청
  4. 로깅: ω 시계열을 기록하여 안전 이력 분석
```

### 알고리즘 의사코드

```
Algorithm: Optimal-Decay CBF Filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, u_nom, obstacles, p_sb, α

1. 각 장애물 j에 대해:
   h_j, L_f h_j, L_g h_j 계산 (§3과 동일)

2. 확장 QP 구성:
   변수: z = [u; ω] ∈ R^{m+1}

   목적함수:
     Q = diag(1,...,1, p_sb)   (m+1 × m+1)
     c = [-2u_nom; -2p_sb]    (m+1)
     → min z^T Q z + c^T z

   제약 (각 장애물 j):
     [L_g h_j, α·h_j] · z ≥ -L_f h_j      (CBF 제약)
     u_min ≤ z[:m] ≤ u_max                  (입력 제한)
     0 ≤ z[m] ≤ 1                           (ω 범위)

3. QP 풀기 (scipy.optimize.minimize 또는 cvxpy)

4. Return:
   u_safe = z[:m]
   ω = z[m]
   info = {optimal_omega: ω, correction_norm: ||u_safe - u_nom||}
```

### 구현

- **파일**: `optimal_decay_cbf_filter.py` (라인 31-224)
- **부모**: `CBFSafetyFilter`
- **추가 반환**: `optimal_omega` (1에 가까울수록 안전)
- **파라미터**: `penalty_weight` (p_sb), `omega_min`, `omega_max`

---

## 10. Backup CBF + Gatekeeper

### 10.1 Backup CBF

### 문제

단일 스텝 CBF는 "지금 안전"만 보장하지,
"미래에도 안전한 행동이 존재"하는지 확인하지 않는다.

### 핵심 아이디어

**백업 궤적**(예: 긴급 정지)을 시뮬레이션하고,
현재 제어가 백업 궤적의 안전성에 미치는 **민감도**를 계산한다.

### 수학적 정의

**민감도 전파** (체인 룰):
```
백업 궤적: x_b[0] = x_next(u),  x_b[k+1] = f(x_b[k], u_backup)

민감도:
  dx_b[0]/du = ∂f/∂u|_{state, u} · dt

  dx_b[k+1]/du = ∂f/∂x|_{x_b[k]} · dx_b[k]/du
```

**다단계 CBF 제약**:
```
∀k = 0,...,H-1:
  ∂h/∂x|_{x_b[k]} · (dx_b[k]/du) · u + α_k · h(x_b[k]) ≥ 0

여기서 α_k = α · γ^k  (시간 감쇠)
```

### 민감도 전파 유도 (체인 룰 확장)

```
Step 1: 초기 민감도
  x_next = f(x, u, dt) ≈ x + [f(x) + g(x)·u]·dt

  dx_next/du = g(x) · dt     (x는 현재 상태, 상수 취급)

  Differential Drive 예시:
    dx_next/du = [cos(θ)  0] · dt = [cos(θ)·dt    0    ]
                [sin(θ)  0]        [sin(θ)·dt    0    ]
                [0       1]        [0             dt   ]

Step 2: 재귀 전파 (k = 1, 2, ..., H-1)
  x_b[k+1] = f(x_b[k], u_backup, dt)

  dx_b[k+1]/du = ∂f/∂x|_{x_b[k], u_backup} · dx_b[k]/du

  Jacobian ∂f/∂x:
    DiffDrive (kinematic, u_backup = [0, 0]):
      ∂f/∂x = I + [0  0  -v·sin(θ)] · dt
                  [0  0   v·cos(θ)]
                  [0  0   0       ]
    u_backup = 0이면 v = 0이므로:
      ∂f/∂x = I  (단위 행렬)

  따라서 긴급 정지 백업:
    dx_b[k]/du = dx_b[0]/du = g(x)·dt,  ∀k
    (정지 상태에서 Jacobian이 단위 행렬이므로)

Step 3: CBF 그래디언트
  ∂h/∂x|_{x_b[k]} = [2(x_b[k,0] - x_o), 2(x_b[k,1] - y_o), 0]

Step 4: 다단계 CBF 제약 구성
  c_k = ∂h/∂x|_{x_b[k]} · dx_b[k]/du    (u에 대한 그래디언트)
  b_k = α_k · h(x_b[k])

  → c_k · u + b_k ≥ 0    (선형 제약)
```

### 수치 Jacobian 계산

해석적 Jacobian이 어려운 경우, 유한 차분으로 계산한다:

```
수치 Jacobian (중앙 차분):
  ∂f_i/∂x_j ≈ [f_i(x + ε·e_j, u) - f_i(x - ε·e_j, u)] / (2ε)

  ε = 1e-6 (일반적)

  코드:
    def numerical_jacobian(f, x, u, dt, eps=1e-6):
        n = len(x)
        J = np.zeros((n, n))
        for j in range(n):
            x_plus = x.copy(); x_plus[j] += eps
            x_minus = x.copy(); x_minus[j] -= eps
            J[:, j] = (f(x_plus, u, dt) - f(x_minus, u, dt)) / (2*eps)
        return J

  장점: 임의 동역학 모델에 적용 가능
  단점: n번의 추가 시뮬레이션 (O(n) 비용)
```

### 백업 정책 설계 가이드라인

```
1. 긴급 정지 (가장 단순):
   u_backup = [0, 0]     (v=0, ω=0)
   장점: 항상 안전 (정지 상태에서 장애물과 거리 불변)
   단점: 정지 거리 미고려 (관성이 있으면 즉시 정지 불가)

2. 최대 감속:
   u_backup = [-a_max, 0]    (최대 감속, 직진)
   장점: 현실적 정지 거리 반영
   단점: 방향 변경 없음

3. 회피 감속:
   u_backup = [-a_max, ω_avoid]   (감속 + 장애물에서 멀어지는 회전)
   장점: 가장 보수적 (거리 증가)
   단점: 추가 계산 필요 (ω_avoid 결정)

4. 최소 에너지:
   u_backup = argmin ||u||  s.t. ḣ ≥ 0
   장점: 최소한의 개입
   단점: QP 풀이 필요 (실시간 부담)

권장: 기구학 모델 → 긴급 정지, 동역학 모델 → 최대 감속
```

### 구현

- **파일**: `backup_cbf_filter.py` (라인 26-348)
- **파라미터**: `backup_horizon`, `decay_rate`, `cbf_alpha`

### 10.2 Gatekeeper

### 문제

"현재 제어를 적용한 후에도, 안전하게 정지할 수 있는가?"

### 핵심 아이디어

MPPI 제어를 적용한 후의 상태에서 **백업 궤적**을 시뮬레이션한다.
백업 궤적이 안전하면 게이트를 열고(통과), 아니면 닫고(백업 제어 적용).

### 수학적 정의

```
Algorithm: Gatekeeper
━━━━━━━━━━━━━━━━━━━
Input: state x, u_mppi

1. 다음 상태 예측:
   x_next = f(x, u_mppi, dt)

2. 백업 궤적 시뮬레이션 (x_next에서):
   x_backup[0] = x_next
   for k = 1 to H:
     x_backup[k] = f(x_backup[k-1], u_backup, dt)

3. 안전 검증:
   all_safe = ∀k: h(x_backup[k]) > 0

4. 게이팅:
   if all_safe:
     return u_mppi         (gate OPEN)
   else:
     return u_backup        (gate CLOSED)
```

**전방 불변성**: 항상 안전한 백업 행동(정지)이 가능하므로,
무한 시간 안전이 보장된다.

### 전방 불변성 증명 (Gatekeeper)

```
정리: Gatekeeper가 적용된 시스템은 h(x(t)) > 0, ∀t ≥ 0을 만족한다.

증명:

전제:
  - h(x(0)) > 0 (초기 상태 안전)
  - 백업 정책 u_backup = 0 (긴급 정지)
  - 정지 상태에서 장애물과의 거리 불변 (마찰 없는 기구학)

Base case (t = 0):
  x(0) ∈ C 이고, 정지 시 모든 미래 상태가 안전 (검증 가능).

Inductive step:
  시각 t에서 x(t) ∈ C이고, Gatekeeper가 u(t)를 적용했다고 가정.

  Case 1 — Gate OPEN:
    u(t) = u_mppi를 적용 → x(t+1) = f(x(t), u_mppi, dt)
    Gatekeeper는 x(t+1)에서 백업 궤적이 안전함을 검증했으므로:
      ∀k: h(x_backup[k]) > 0
    x(t+1)에서 다시 Gatekeeper를 적용하면,
    최악의 경우 Gate가 닫혀 백업 정책이 적용됨 → 안전.

  Case 2 — Gate CLOSED:
    u(t) = u_backup를 적용 → x(t+1) = f(x(t), u_backup, dt)
    이전 스텝에서 이미 x(t)에서의 백업 궤적이 안전함을 검증했으므로:
      h(x(t+1)) > 0
    x(t+1)에서 백업 궤적을 재검증하면 여전히 안전.

  두 경우 모두 h(x(t+1)) > 0이고,
  x(t+1)에서 안전한 백업이 존재 → 귀납법에 의해 ∀t: h(x(t)) > 0.  ∎
```

### 비교: Gatekeeper vs MPC 안전

```
┌──────────────────┬────────────────────┬─────────────────────┐
│ 특성              │ Gatekeeper          │ MPC Safety           │
├──────────────────┼────────────────────┼─────────────────────┤
│ 안전 검증 방법    │ 백업 궤적 시뮬레이션│ 최적화 내 제약       │
│ 온라인 계산       │ O(H × dynamics)    │ O(N × QP)           │
│ 안전 보장         │ 강건 (worst-case)  │ 조건부 (실현가능 시) │
│ 성능 영향         │ 최소 (gate 판정만) │ 보수적 궤적          │
│ 구현 복잡도       │ 낮음               │ 높음                 │
│ 모델 정확도 요구  │ 백업 궤적 정확도    │ 전체 모델 정확도     │
│ 임의 플래너 호환  │ 예 (plug-in)       │ 아니오 (MPC 전용)    │
│ 재귀 실현가능     │ 항상 (백업 존재)    │ 보장 안됨            │
└──────────────────┴────────────────────┴─────────────────────┘
```

### 구현

- **파일**: `gatekeeper.py` (라인 38-198)
- **핵심 메서드**: `filter(state, u_mppi)` → (u_safe, info)

---

## 11. MPS (Model Predictive Shield)

### 문제

Gatekeeper는 복잡한 백업 전략이 필요할 수 있다.
더 간단한 안전 검증이 필요한 경우.

### 핵심 아이디어

**정지 궤적**(제어 = 0)의 안전만 확인하는 간소화된 Gatekeeper.
상태 없는(stateless) 설계로 외부 플래너와의 통합이 쉽다.

### 구현

- **파일**: `mps_controller.py` (라인 27-171)
- **API**: `shield(state, nominal_control, model)` → (safe_control, info)
- Gatekeeper보다 단순하지만, 동일한 전방 불변성 보장

### 감속 궤적 계산

MPS에서 정지 궤적은 u = 0을 적용한 시뮬레이션이다:

```
기구학 모델 (마찰 없음):
  u_backup = [0, 0]
  x_{k+1} = x_k     (정지 → 상태 불변)
  → 현재 위치에서 안전하면 영원히 안전

동역학 모델 (마찰 있음):
  u_backup = [0, 0]  (힘 = 0)
  v_{k+1} = v_k · (1 - μ·dt)     (마찰 감속)
  x_{k+1} = x_k + v_k · cos(θ) · dt
  y_{k+1} = y_k + v_k · sin(θ) · dt

  정지까지 소요 시간: t_stop ≈ v_0 / (μ·g)
  정지 거리: d_stop ≈ v_0² / (2·μ·g)

  → H > t_stop / dt 으로 설정해야 감속 완료까지 검증
```

### 외부 플래너 통합

```
MPS는 stateless이므로 어떤 상위 플래너와도 통합이 쉽다:

┌──────────────────────────────────────────────────────┐
│  Global Planner (RRT*, A*, ...)                       │
│       ↓ reference path                               │
│  Local Planner (MPPI, MPC, DWA, ...)                  │
│       ↓ u_nominal                                    │
│  ┌────────────────────────────────┐                  │
│  │  MPS Shield                    │                  │
│  │  shield(state, u_nominal, model)│                  │
│  │       ↓                        │                  │
│  │  if backup_safe: u_nominal     │                  │
│  │  else:           u_backup = 0   │                  │
│  └────────────────────────────────┘                  │
│       ↓ u_safe                                       │
│  Robot Actuators                                      │
└──────────────────────────────────────────────────────┘

장점:
  - 플래너 교체 시 MPS 코드 변경 불필요
  - 단일 함수 호출로 안전 보장
  - 테스트 간단 (unit test에서 model mock 가능)
```

### 비교

| 특성 | Gatekeeper | MPS |
|------|-----------|-----|
| 상태 관리 | Stateful | Stateless |
| 백업 전략 | 사용자 정의 | 정지 (u=0) |
| API | filter(state, u) | shield(state, u, model) |
| 복잡도 | O(H × rollout) | O(H × rollout) |
| 유연성 | 높음 | 낮음 (단순) |

---

## 12. Conformal Prediction + CBF

### 문제

CBF의 안전 마진은 보통 **고정값**이다.
모델이 정확하면 마진이 불필요하게 크고,
모델이 부정확하면 마진이 부족할 수 있다.

### 핵심 아이디어

**Conformal Prediction (CP)**으로 모델 예측 오차를 동적으로 추정하고,
CBF의 안전 마진을 **자동 조절**한다.

### 수학적 정의

**비순응 점수 (Nonconformity Score)**:
```
s_i = ||y_i - ŷ_i||     (예측 오차)
```

**분위수**:
```
q̂ = Quantile(s_1, ..., s_n, ⌈(n+1)(1-α)/n⌉)
```

**분포-무관 보장 (Distribution-Free Coverage)**:
```
P(s_{n+1} ≤ q̂) ≥ 1 - α
```

이것은 예측 오차 분포에 대한 **어떤 가정도 없이** 성립한다.
교환 가능성(exchangeability)만 필요하다.

### CP 이론 유도 (교환 가능성으로부터)

```
정리 (Vovk et al., 2005): 교환 가능한 확률 변수 Z_1, ..., Z_{n+1}에 대해,
  P(Z_{n+1}이 상위 ⌈(n+1)(1-α)⌉/n 분위수 이하) ≥ 1 - α

유도:

Step 1: 교환 가능성 (Exchangeability)
  (Z_1, ..., Z_{n+1})이 교환 가능하다:
    P(Z_{π(1)}, ..., Z_{π(n+1)}) = P(Z_1, ..., Z_{n+1})
    모든 순열 π에 대해.

  → i.i.d.이면 교환 가능 (역은 성립하지 않음)
  → 시계열에서는 조건부 교환 가능성 필요

Step 2: 순위 통계량
  Z_{n+1}의 순위 R = |{i : Z_i ≤ Z_{n+1}}|
  교환 가능성에 의해, R은 {1, ..., n+1}에서 균등 분포.

Step 3: 분위수 보장
  P(R ≤ ⌈(n+1)(1-α)⌉) = ⌈(n+1)(1-α)⌉ / (n+1) ≥ 1 - α

Step 4: 비순응 점수 적용
  s_i = ||y_i - ŷ_i|| (예측 오차)
  q̂ = Quantile(s_1, ..., s_n; ⌈(n+1)(1-α)/n⌉)

  P(s_{n+1} ≤ q̂) ≥ 1 - α   (분포-무관 보장)

핵심: 모델의 정확도와 무관하게, 커버리지 보장은 성립한다.
  모델이 나쁘면 → s가 크고 → q̂가 큼 → 넓은 구간
  모델이 좋으면 → s가 작고 → q̂가 작음 → 좁은 구간
```

### 유한 표본 보장 증명 스케치

```
정리: n개의 교정 데이터에 대해,
  P(s_{n+1} ≤ q̂) ≥ 1 - α
  P(s_{n+1} ≤ q̂) ≤ 1 - α + 1/(n+1)

증명:
  교환 가능성에 의해, Z_{n+1}이 상위 k/n 분위수 이하일 확률:
    P(R_{n+1} ≤ k) = k / (n+1)

  k = ⌈(n+1)(1-α)⌉로 설정하면:
    P ≥ ⌈(n+1)(1-α)⌉ / (n+1) ≥ (n+1)(1-α) / (n+1) = 1 - α

  상한: k ≤ (n+1)(1-α) + 1이므로:
    P ≤ ((n+1)(1-α) + 1) / (n+1) = 1 - α + 1/(n+1)

  n = 49 (window=50)일 때:
    1/(n+1) = 0.02 → 오차 2% 이내
    α = 0.1: 커버리지 ∈ [90%, 92%]
```

### ACP (Adaptive Conformal Prediction)

비정상(non-stationary) 환경에서는 지수 가중을 사용한다:
```
w_i = γ^{n-1-i}     (γ = 0.95)
```

최근 데이터에 더 높은 가중치를 부여하여 분포 변화에 대응한다.

### ACP 수렴 분석

```
ACP의 장기 커버리지:

정의:
  err_t = 1{s_t > q̂_t}     (t 시점에서 커버리지 실패)

ACP 업데이트:
  α_{t+1} = α_t + γ_lr · (err_t - α_target)

장기 평균:
  lim_{T→∞} (1/T) Σ_{t=1}^T err_t → α_target

  즉, 평균적으로 α_target 비율의 실패율을 달성한다.

수렴 속도:
  γ_lr이 클수록 빠른 적응, 높은 분산
  γ_lr이 작을수록 느린 적응, 낮은 분산

  권장: γ_lr = 0.01 ~ 0.05

비정상 환경에서의 추적성:
  분포 변화율 Δ에 대해, ACP의 추적 지연:
    delay ≈ 1 / γ_lr 스텝

  예: γ_lr = 0.05, dt = 0.1s → 추적 지연 ≈ 2s
```

### 마진 적응 시각화 (시계열)

```
시간에 따른 안전 마진 적응:

margin(m) ↑
  0.5  │                        ╭───╮
       │                       ╱     ╲
  0.4  │              ╭──╮   ╱       ╲
       │             ╱    ╲ ╱         ╲
  0.3  │            ╱      ╳           ╲───
       │           ╱       ╲            ╲
  0.2  │     ╭────╱                      ╲───
       │    ╱
  0.1  │───╱
       │
  0.02 │─── margin_min
       ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──→ t (s)
       0  1  2  3  4  5  6  7  8  9  10

  t=0~2: 모델 정확 → 마진 감소 (성능↑)
  t=3~4: 외란 발생 → 예측 오차 증가 → 마진 확대 (안전↑)
  t=5~6: 외란 최대 → 마진 최대 (0.5 클리핑)
  t=7~10: 외란 감소 → 마진 점진적 감소

  핵심: 마진이 자동으로 모델 정확도에 적응한다.
```

### Shield-MPPI 통합

```
Algorithm: Conformal CBF Shield-MPPI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 이전 예측과 실제 상태 비교:
   s = ||x_actual - x_predicted||

2. CP 분위수 업데이트:
   q̂ = ACP_update(s, window)

3. 안전 마진 설정:
   safety_margin = clip(q̂, margin_min, margin_max)

4. Shield-MPPI 실행 (업데이트된 마진으로):
   u = ShieldMPPI(state, ref, margin=safety_margin)
```

**동적 마진 효과**:
```
모델 정확:  s 작음 → q̂ 작음 → 마진↓ → 성능↑
모델 부정확: s 큼 → q̂ 큼 → 마진↑ → 안전↑
```

### 비교: CP vs Bayesian vs Bootstrap 신뢰 구간

```
┌───────────────────┬────────────────┬────────────────┬────────────────┐
│ 특성               │ Conformal Pred. │ Bayesian       │ Bootstrap      │
├───────────────────┼────────────────┼────────────────┼────────────────┤
│ 분포 가정          │ 없음 (무관)     │ 사전/우도 필요 │ i.i.d. 필요    │
│ 커버리지 보장      │ 유한 표본 보장  │ 점근적         │ 점근적         │
│ 계산 비용          │ O(n log n)     │ O(n·MC)       │ O(n·B)        │
│ 적응성             │ ACP로 가능     │ 순차 업데이트  │ 재부트스트랩   │
│ 모델 의존성        │ 임의 모델      │ 확률 모델 필요 │ 임의 추정량    │
│ 구간 형태          │ 대칭 (±q̂)     │ 비대칭 가능    │ 비대칭 가능    │
│ 실시간 적합성      │ 우수           │ 보통           │ 나쁨           │
│ CBF 통합 용이성    │ margin으로 직접│ 분산→margin   │ 분산→margin   │
└───────────────────┴────────────────┴────────────────┴────────────────┘

CP가 로봇 안전 제어에 가장 적합한 이유:
  1. 분포 가정 없음 → 모델 오류 형태 불문
  2. 유한 표본 보장 → 데이터 부족해도 안전
  3. 실시간 계산 → 제어 루프 내 사용 가능
  4. 단순 통합 → margin으로 직접 변환
```

### 알고리즘 의사코드

```
Algorithm: Conformal CBF Shield-MPPI (상세)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
State: score_buffer (size W), prev_predicted_state

Input: state x, reference_traj, model

1. 비순응 점수 업데이트:
   if prev_predicted_state is not None:
     s = ||x - prev_predicted_state||
     score_buffer.append(s)

2. CP 분위수 계산:
   if use_acp:
     weights = [γ^{W-1-i} for i in range(W)]
     q̂ = weighted_quantile(score_buffer, weights, 1-α)
   else:
     k = ceil((W+1)*(1-α)/W)
     q̂ = sorted(score_buffer)[k-1]

3. 안전 마진 설정:
   safety_margin = clip(q̂ · margin_scale, margin_min, margin_max)

4. Shield-MPPI 실행:
   obstacles_adjusted = [(x,y, r + safety_margin) for obs]
   u, info = ShieldMPPI.compute_control(x, ref, obstacles_adjusted)

5. 다음 상태 예측 (CP용):
   prev_predicted_state = model.predict(x, u, dt)

6. Return u, {**info, cp_margin: safety_margin, cp_quantile: q̂}
```

### 구현

- **파일**: `conformal_cbf_mppi.py` (라인 33-149)
- **부모**: ShieldMPPIController
- **파라미터**:

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| cp_alpha | 0.1 | CP 커버리지 레벨 (1-α = 90%) |
| cp_window_size | 50 | 윈도우 크기 |
| cp_margin_min | 0.02 | 최소 마진 |
| cp_margin_max | 0.5 | 최대 마진 |
| cp_gamma | 0.95 | ACP 지수 가중 계수 |

### 언제 사용

- 모델 정확도가 시간에 따라 변하는 경우
- 분포 가정 없이 안전 보장이 필요한 경우
- 적응적 안전 마진이 필요한 경우

---

## 13. DIAL + Safety 결합

### 13.1 Shield-DIAL-MPPI

DIAL-MPPI의 확산 어닐링 + Shield의 per-step CBF를 결합한다.

**핵심 차이**: 각 어닐링 반복에서 `rollout()` 대신 `_shielded_rollout()` 사용.

```
For iteration i = 1 to n_iters:
  1. 어닐링된 노이즈로 샘플링
  2. _shielded_rollout()으로 안전 궤적 생성   ← 핵심
  3. 안전 궤적 기반 가중치 계산
  4. U = Σ w_k · shielded_controls_k
```

### 반복별 안전 분석

DIAL-MPPI는 여러 번의 어닐링 반복을 수행한다. 각 반복에서의 안전 상태:

```
반복 1 (높은 노이즈 σ₁):
  - 탐색적 샘플링 → 다양한 궤적 (일부 위험)
  - Shield가 위험 궤적 클리핑 → 안전 궤적만 비용 평가
  - 넓은 탐색 범위에서 안전한 최적 발견

반복 2 (줄어든 노이즈 σ₂):
  - 1차 최적 근처에서 정밀 탐색
  - Shield 개입 빈도 감소 (이미 안전한 영역 탐색)
  - 안전 영역 내에서 비용 최소화

반복 n (최소 노이즈 σ_n):
  - 최종 정밀 조정
  - Shield 개입 거의 없음
  - 안전 + 최적 제어 동시 달성

핵심 관찰:
  Shield 없는 DIAL: 초기 반복에서 위험 궤적이 비용 평가에 포함
    → 위험 방향으로 수렴할 수 있음
  Shield + DIAL: 모든 반복에서 안전 보장
    → 안전 영역 내 최적으로 수렴
```

### Shield-DIAL 알고리즘 의사코드

```
Algorithm: Shield-DIAL-MPPI
━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference, n_iters, σ_schedule, obstacles

U = warm-start (이전 제어 시퀀스 시프트)

For i = 1 to n_iters:
  1. 어닐링된 노이즈:
     σ_i = σ_max · decay^(i-1)     (지수 감소)
     ε ~ N(0, σ_i²·I)              (K × N × m)

  2. 샘플 제어:
     U_samples[k] = U + ε[k]       (K개 샘플)
     clip(U_samples, u_min, u_max)

  3. Shield 롤아웃:
     trajectories, shielded_controls, info = _shielded_rollout(x, U_samples)
     (매 스텝에서 CBF 클리핑 적용)

  4. 비용 계산 (안전 궤적 기반):
     costs = compute_costs(trajectories, reference)

  5. 가중치:
     w = softmax(-costs / λ)

  6. 제어 업데이트:
     U = Σ_k w_k · shielded_controls[k]

Return U[0], info
```

### 13.2 Adaptive-Shield-DIAL-MPPI

Shield-DIAL에 α(d,v) 적응을 추가:
```
_cbf_shield_batch()에서 고정 α 대신 α(d,v) 사용
```

### 구현

- **Shield-DIAL**: `shield_dial_mppi.py` (라인 22-380)
- **Adaptive-Shield-DIAL**: `adaptive_shield_dial_mppi.py` (라인 27-131)

### 성능

벤치마크 결과에서 **AdaptiveShieldDIAL**이 가장 높은 안전율 (100%)과
가장 좋은 추적 성능 (RMSE 0.38m)을 달성했다.

---

## 14. Chance Constraint (C2U 연계)

### 핵심 아이디어

불확실성에 비례하여 장애물의 유효 반경을 동적으로 확장한다.
(→ MPPI_THEORY.md §13 참조)

### 수학적 정의

```
r_eff = r + κ_α · √(trace(Σ_pos))

κ_α = Φ^{-1}(1 - α)   (정규분포 분위수)
```

**ChanceConstraintCost**에서 UT 공분산 궤적을 받아
시간별 유효 반경을 계산한다:

```
For t = 0 to N:
  σ_pos = Σ_pos[:2, :2]  (2x2 위치 공분산 블록)
  σ_eff = √(trace(σ_pos))
  r_eff[t] = r + margin_factor · κ_α · σ_eff
```

### Chebyshev 부등식과의 비교

가우시안 가정이 아닌 경우, Chebyshev 부등식으로 더 보수적인 r_eff를 계산할 수 있다:

```
가우시안 분위수 (κ_α):
  α = 0.05 (95% 안전): κ = Φ^{-1}(0.95) = 1.645
  α = 0.01 (99% 안전): κ = Φ^{-1}(0.99) = 2.326

Chebyshev 부등식 (분포 무관):
  P(|X - μ| ≥ k·σ) ≤ 1/k²

  α = 0.05: k = √(1/0.05) = 4.47
  α = 0.01: k = √(1/0.01) = 10.0

비교:
  ┌──────────┬─────────────┬─────────────┬────────────────┐
  │ α (위반) │ κ (가우시안) │ k (Chebyshev)│ 비율 k/κ       │
  ├──────────┼─────────────┼─────────────┼────────────────┤
  │ 0.10     │ 1.28        │ 3.16        │ 2.47x 더 보수적 │
  │ 0.05     │ 1.645       │ 4.47        │ 2.72x 더 보수적 │
  │ 0.01     │ 2.326       │ 10.0        │ 4.30x 더 보수적 │
  └──────────┴─────────────┴─────────────┴────────────────┘

  → 가우시안 가정이 성립하면, κ_α를 사용하는 것이
    훨씬 덜 보수적 (성능 유지).
  → 가정이 불확실하면, Chebyshev가 더 안전하지만 과도하게 보수적.
```

### 비가우시안 확장

UT(Unscented Transform)는 공분산 전파에서 비선형성을 2차까지 포착하지만,
분포 형태는 가우시안으로 가정한다. 비가우시안 확장:

```
1. Particle Filter 기반:
   σ-point 대신 N개 파티클로 전파
   r_eff = r + κ_α · 경험적 분위수(파티클 위치)
   장점: 임의 분포, 단점: 계산량 O(N·state_dim)

2. Mixture of Gaussians:
   Σ = Σ_i w_i · Σ_i  (혼합 공분산)
   각 성분별 r_eff 계산 후 최대값 사용
   장점: 다봉 분포, 단점: 혼합 추정 필요

3. Conformal + Chance Constraint:
   CP 분위수를 κ_α 대신 사용
   r_eff = r + q̂_CP · √(trace(Σ_pos))
   장점: 분포 무관, 단점: 과거 데이터 필요
```

### r_eff 시간 변화 시각화

```
시간에 따른 유효 반경 r_eff 변화:

r_eff(m)↑
  1.5  │                              ╱───── (α=0.01, 99%)
       │                            ╱
  1.2  │                          ╱
       │                        ╱   ╱─────── (α=0.05, 95%)
  1.0  │                      ╱   ╱
       │                    ╱   ╱
  0.8  │                  ╱   ╱   ╱───────── (α=0.10, 90%)
       │                ╱   ╱   ╱
  0.6  │              ╱   ╱   ╱
       │            ╱   ╱   ╱
  0.4  │          ╱   ╱   ╱
       │        ╱   ╱   ╱
  r=0.3│───────╱───╱───╱──── 실제 반경 r
       ├──┼──┼──┼──┼──┼──┼──→ t (horizon step)
       0  5  10 15 20 25 30

  관찰:
    - t가 커질수록 불확실성 축적 → Σ 증가 → r_eff 증가
    - α가 작을수록 (안전 요구↑) → κ_α 증가 → r_eff 더 확장
    - 처음 몇 스텝은 r_eff ≈ r (불확실성 작음)
    - 먼 미래는 r_eff >> r (불확실성 큼 → 매우 보수적)
```

### 구현

- **파일**: `chance_constraint_cost.py` (라인 31-148)
- **API**: `set_covariance_trajectory()` → `compute_cost()`

---

## 15. 안전 기법 선택 가이드

### 의사결정 트리

```
안전 기법 선택
│
├─ 안전 보장 수준?
│  ├─ Hard (100%) ─┬─ 다단계 검증? → Gatekeeper / MPS
│  │               ├─ 롤아웃 내부? → Shield-MPPI
│  │               └─ 사후 필터?   → QP Filter
│  ├─ Soft (비용) ─┬─ 이진 거부?   → HardCBFCost
│  │               ├─ 시간 할인?   → HorizonWeightedCBFCost
│  │               └─ 기본?        → ControlBarrierCost
│  └─ 확률적 ─────┬─ 분포 무관?   → Conformal + Shield
│                  └─ 가우시안?    → C2U-MPPI (ChanceConstraint)
│
├─ 장애물 형태?
│  ├─ 비볼록 → Neural CBF
│  ├─ 동적 ──┬─ 속도 인지? → C3BF
│  │         └─ 방향 인지? → DPCBF
│  └─ 정적 볼록 → 기본 CBF
│
├─ QP 실현 가능성 우려?
│  └─ Yes → Optimal-Decay CBF
│
└─ 모델 불확실성?
   ├─ 시변 → Conformal + Shield
   └─ 정적 → C2U-MPPI
```

### 비교 매트릭스

```
┌──────────────────┬────────┬────────┬────────┬────────┬────────┐
│ 기법              │ 보장   │ 계산   │ 비볼록  │ 동적   │ 불확실  │
│                  │ 수준   │ 비용   │ 대응   │ 장애물  │ 성 대응 │
├──────────────────┼────────┼────────┼────────┼────────┼────────┤
│ CBF Cost         │ Soft   │ O(K)   │ ✗     │ ✗      │ ✗     │
│ Hard CBF Cost    │ Soft*  │ O(K)   │ ✗     │ ✗      │ ✗     │
│ Horizon CBF Cost │ Soft   │ O(K)   │ ✗     │ ✗      │ ✗     │
│ QP Filter        │ Hard†  │ O(1)   │ ✗     │ ✓      │ ✗     │
│ Optimal-Decay    │ Soft‡  │ O(1)   │ ✗     │ ✓      │ ✗     │
│ Shield           │ Hard   │ O(K)   │ ✗     │ ✗      │ ✗     │
│ Adaptive Shield  │ Hard   │ O(K)   │ ✗     │ ✗      │ ✗     │
│ Neural CBF       │ Learned│ O(1)   │ ✓     │ ✗      │ ✗     │
│ C3BF             │ Soft   │ O(K)   │ ✗     │ ✓      │ ✗     │
│ DPCBF            │ Soft   │ O(K)   │ ✗     │ ✓      │ ✗     │
│ Gatekeeper       │ Hard   │ O(H)   │ ✗     │ ✓      │ ✗     │
│ MPS              │ Hard   │ O(H)   │ ✗     │ ✓      │ ✗     │
│ Conformal+Shield │ Prob.  │ O(W+K) │ ✗     │ ✗      │ ✓     │
│ C2U (Chance)     │ Prob.  │ O(n²)  │ ✗     │ ✗      │ ✓     │
│ Shield-DIAL      │ Hard   │ O(I·K) │ ✗     │ ✗      │ ✗     │
│ Adaptive-DIAL    │ Hard   │ O(I·K) │ ✗     │ ✗      │ ✗     │
└──────────────────┴────────┴────────┴────────┴────────┴────────┘

† QP 실현 가능 시  ‡ ω=0이면 제약 무효화  * 1e6 비용으로 사실상 거부
```

### 시나리오별 추천

| 시나리오 | 추천 기법 | 이유 |
|---------|----------|------|
| 정적 볼록 장애물 | Shield-MPPI | 100% 안전 + 단순 |
| 비볼록 장애물 | Neural CBF | 학습으로 임의 형상 대응 |
| 움직이는 장애물 | C3BF + Gatekeeper | 속도 인지 + 다단계 검증 |
| 모델 불확실성 | Conformal + Shield | 분포-무관 적응 마진 |
| 실시간 제약 | QP Filter | O(1) 사후 처리 |
| 최고 안전 + 성능 | Adaptive-Shield-DIAL | 어닐링 + 적응 + 안전 |
| QP 실현 불가 우려 | Optimal-Decay | 항상 실현 가능 |

### 복합 안전 전략 추천

단일 기법으로는 모든 시나리오에 대응할 수 없다. 복합 전략을 권장한다:

```
전략 1: 방어 심층 (Defense in Depth)

  Layer 1 — 비용 함수:
    ControlBarrierCost (Soft, 가벼움)
    → 위험 궤적의 가중치를 줄여 "안전 선호" 유도

  Layer 2 — 롤아웃 안전:
    Shield-MPPI (Hard, 중간 비용)
    → 모든 샘플 궤적을 안전하게 강제

  Layer 3 — 사후 필터:
    Gatekeeper (Hard, 최종 방어)
    → 최종 제어의 안전을 다단계 검증

  결과: 3중 방어로 사실상 100% 안전

전략 2: 적응형 안전

  기본: Adaptive-Shield-MPPI (거리/속도 적응 α)
  모델 불확실성: + Conformal Prediction (동적 마진)
  비볼록 장애물: + Neural CBF (학습 h(x))

  결과: 환경 변화에 자동 적응하는 안전 시스템

전략 3: 최소 오버헤드

  기본: CBF Cost (비용 함수만, O(K))
  백업: QP Filter (O(1), 위반 시에만)

  결과: 최소 계산으로 합리적 안전
```

### 안전 검증 체크리스트

```
□ 1. 환경 정의
  □ 장애물 형태 (원형/직사각형/비볼록/동적)
  □ 장애물 최대 속도 (동적 장애물)
  □ 로봇 최대 속도 / 가속도
  □ 최소 허용 거리 (로봇 반경 + 안전 마진)

□ 2. CBF 설계
  □ h(x) 정의 (연속, 미분 가능)
  □ 안전 집합 C = {x : h(x) ≥ 0} 시각화 확인
  □ ∂h/∂x 계산 (해석적 또는 autograd)
  □ 상대 차수 확인 (1이면 직접 CBF, 2이면 HOCBF)

□ 3. 파라미터 튜닝
  □ α 설정 (기본 0.3, α·dt < 1 확인)
  □ safety_margin 설정 (로봇 크기 반영)
  □ weight 설정 (비용 기반 사용 시)

□ 4. 테스트 시나리오
  □ 정면 접근 (head-on)
  □ 측면 통과 (lateral pass)
  □ 좁은 통로 (narrow corridor)
  □ 다중 장애물 (multi-obstacle)
  □ 고속 접근 (high-speed approach)
  □ 코너 케이스 (장애물 경계에서 시작)

□ 5. 성능 검증
  □ 안전 위반율 = 0% (Hard 기법)
  □ 추적 RMSE < 목표값
  □ 개입율 (intervention rate) 기록
  □ 계산 시간 < 제어 주기
```

### 일반적 실패 모드와 대처

```
┌─────────────────────────────────┬──────────────────────────────────┐
│ 실패 모드                        │ 대처 방법                         │
├─────────────────────────────────┼──────────────────────────────────┤
│ 1. 이산화 오차로 h < 0 발생      │ dt 줄이기, margin 늘리기          │
│    (α·dt ≈ 1일 때)              │ α·dt < 0.3 유지                  │
│                                  │                                  │
│ 2. QP 실현 불가능               │ Optimal-Decay 사용, margin 줄이기 │
│    (좁은 통로, 입력 제한)        │ 백업 제어 준비                    │
│                                  │                                  │
│ 3. Chattering (제어 떨림)        │ α 줄이기, 히스테리시스 추가       │
│    (경계 근처에서 on/off 반복)   │ 매끄러운 class-K 함수 사용        │
│                                  │                                  │
│ 4. 데드락 (진행 불가)            │ 재계획 트리거, 백업 궤적 다양화   │
│    (CBF가 모든 방향 차단)        │ 시간 제한 후 긴급 후진            │
│                                  │                                  │
│ 5. 모델 불일치                   │ Conformal Prediction 추가         │
│    (실제 vs 예측 차이 큼)        │ robust margin 사용                │
│                                  │                                  │
│ 6. 고속 충돌                     │ C3BF 또는 DPCBF 사용             │
│    (CBF 반응 시간 부족)          │ 속도 제한 레이어 추가             │
│                                  │                                  │
│ 7. Neural CBF 외삽 실패          │ 훈련 도메인 확장, fallback CBF    │
│    (미학습 영역 진입)            │ 온라인 재학습 트리거              │
└─────────────────────────────────┴──────────────────────────────────┘
```

---

## 참고 문헌

1. Ames, A. et al. (2019). "Control Barrier Functions: Theory and Applications." ECC.
2. Zeng, J. et al. (2021). "Safety-Critical Model Predictive Control with Discrete-Time Control Barrier Function." ACC.
3. Hobbs, K. et al. (2023). "Runtime Assurance via Backup Control Barrier Functions." CDC.
4. Angelopoulos, A. et al. (2023). "Conformal Prediction Under Covariate Shift." NeurIPS.
5. Dawson, C. et al. (2023). "Safe Control with Learned Certificates." IEEE.
6. Nagumo, M. (1942). "Uber die Lage der Integralkurven gewohnlicher Differentialgleichungen." Proc. Phys.-Math. Soc. Japan.
7. Xiao, W. & Belta, C. (2021). "High Order Control Barrier Functions." IEEE TAC.
8. Thirugnanam, A. et al. (2022). "Safety-Critical Control with Control Barrier Functions and Collision Cones." IEEE RA-L.
9. Dalal, G. et al. (2018). "Safe Exploration in Continuous Action Spaces." ICML Workshop.
10. Vovk, V. et al. (2005). "Algorithmic Learning in a Random World." Springer.
11. Lindemann, L. et al. (2023). "Safe Planning in Dynamic Environments using Conformal Prediction." IEEE RA-L.
12. Tonkens, S. & Herbert, S. (2022). "Refining Control Barrier Functions through Hamilton-Jacobi Reachability." IROS.
13. Ferlez, J. & Shao, Y. (2020). "Neural CBF: Learning Barrier Certificates for Safety-Critical Systems." CDC.
14. Williams, G. et al. (2017). "Information Theoretic MPC for Model-Based Reinforcement Learning." ICRA.
