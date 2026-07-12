# MPC 기초 공부 자료 — 최적 제어에서 Model Predictive Control까지

> **대상**: MPPI를 제대로 이해하기 위해 그 아래 깔린 최적 제어/MPC 이론을
> 기초부터 다시 쌓고 싶은 로보틱스 엔지니어 (= 이 repo의 주인)
>
> **성격**: 학습용 문서. 직관 → 유도 → 예제 → 연습문제 순서로 개념을 쌓는다.
> 변형별 레퍼런스는 [docs/MPPI_THEORY.md](../MPPI_THEORY.md) 참조.
>
> **다음 문서**: [02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md) —
> 이 문서의 개념(HJB, LQR, receding horizon)을 그대로 이어받아 MPPI를 유도한다.

---

## 목차

1. [최적 제어 문제의 일반형](#1-최적-제어-문제의-일반형)
2. [동적 계획법과 HJB 방정식](#2-동적-계획법과-hjb-방정식)
3. [LQR: 해석적으로 풀리는 특별한 경우](#3-lqr-해석적으로-풀리는-특별한-경우)
4. [MPC: receding horizon의 아이디어](#4-mpc-receding-horizon의-아이디어)
5. [제약 처리: hard vs soft, QP 정식화](#5-제약-처리-hard-vs-soft-qp-정식화)
6. [NMPC와 실시간 이슈](#6-nmpc와-실시간-이슈)
7. [MPC vs MPPI — 왜 이 repo는 MPPI인가](#7-mpc-vs-mppi--왜-이-repo는-mppi인가)
8. [연습문제](#8-연습문제)
9. [부록 — 더 공부하기 위한 자료](#9-부록--더-공부하기-위한-자료)

---

## 1. 최적 제어 문제의 일반형

### 1.1 표기법 정립 (이 문서와 02 문서에서 공통 사용)

| 기호 | 의미 | 이 repo에서의 대응 |
|------|------|-------------------|
| `x ∈ R^nx` | 상태 (state) | 차동구동: `[x, y, θ]`, nx=3 |
| `u ∈ R^nu` | 제어 입력 (control) | `[v, ω]`, nu=2 |
| `f(x, u)` | 동역학 (dynamics) | `RobotModel.dynamics()` |
| `N` | 예측 호라이즌 길이 | `MPPIParams.N = 30` |
| `dt` | 이산화 스텝 | `MPPIParams.dt = 0.05` (50ms) |
| `l(x, u)` | stage cost (순간 비용) | `StateTrackingCost + ControlEffortCost` |
| `l_f(x)` | terminal cost (종단 비용) | `TerminalCost(Qf)` |
| `J` | 전체 비용 범함수 | 궤적 하나의 총 비용 |
| `V(x)` | value function (최적 cost-to-go) | 02 문서의 free energy와 연결 |
| `U = (u_0, …, u_{N-1})` | 제어 시퀀스 | `MPPIController.U`, shape `(N, nu)` |

표기 규칙:
- 연속시간이면 `x(t)`, `ẋ = f(x, u)`. 이산시간이면 `x_t`, `x_{t+1} = f(x_t, u_t)`.
- 이 repo는 전부 **이산시간** 구현이다 (`dt`로 Euler 이산화,
  `mppi_controller/controllers/mppi/dynamics_wrapper.py`의 `BatchDynamicsWrapper.rollout`).

### 1.2 문제의 일반형

**연속시간 최적 제어 문제 (OCP, Optimal Control Problem)**:

```
            ⌠T
minimize    ⎮  l(x(t), u(t)) dt  +  l_f(x(T))        ← 비용 범함수 J[u(·)]
  u(·)      ⌡0

subject to  ẋ(t) = f(x(t), u(t)),   x(0) = x₀        ← 동역학 (등식 제약)
            u(t) ∈ U                                   ← 입력 제약 (예: |v| ≤ v_max)
            x(t) ∈ X                                   ← 상태 제약 (예: 장애물 회피)
```

**이산시간 버전** (실제로 컴퓨터가 푸는 형태):

```
            N-1
minimize    Σ  l(x_t, u_t)  +  l_f(x_N)
 u_0..u_{N-1} t=0

subject to  x_{t+1} = f(x_t, u_t),   x_0 = x_init
            u_t ∈ U,   x_t ∈ X
```

**핵심 관찰 3가지**:

1. **결정 변수는 함수(시퀀스)다.** 점 하나가 아니라 `u_0, …, u_{N-1}` 전체를
   찾는다. 그래서 "범함수(functional)" 최적화라고 부른다.
   변수 개수 = `N × nu` (repo 기본값이면 30 × 2 = 60차원).

2. **동역학은 등식 제약이다.** `x_{t+1} = f(x_t, u_t)`를 만족하는 궤적만 허용.
   이 제약을 어떻게 다루느냐가 수치 기법을 가른다:
   - **shooting**: `u`만 변수로 두고 `x`는 시뮬레이션으로 소거 (MPPI가 이 방식)
   - **collocation / multiple shooting**: `x`도 변수로 두고 등식 제약 유지

3. **비용의 구조가 풀이 가능성을 결정한다.**
   - `f` 선형 + `l` 이차 + 제약 없음 → **LQR** (해석해 존재, §3)
   - `f` 선형 + `l` 이차 + 선형 제약 → **QP 기반 linear MPC** (§5)
   - `f` 비선형 → **NMPC** (반복 수치 최적화, §6) 또는 **샘플링 (MPPI)** (02 문서)

### 1.3 미니 예제: 1차원 더블 적분기

앞으로 계속 쓸 장난감 문제. 질량 1인 수레를 원점에 세우기:

```
상태:  x = [p, v]ᵀ  (위치, 속도)
제어:  u = 가속도 (스칼라)
동역학: p_{t+1} = p_t + dt·v_t
        v_{t+1} = v_t + dt·u_t
비용:   l(x, u) = p² + v² + 0.1·u²     (원점 추종 + 제어 노력)
```

직관: `u`를 크게 쓰면 빨리 멈추지만 `0.1 u²` 페널티가 커진다.
최적해는 "적당히 세게 감속"이며, 그 "적당히"를 수학적으로 정의하는 것이
최적 제어 이론 전체의 목표다.

---

## 2. 동적 계획법과 HJB 방정식

### 2.1 Bellman의 최적성 원리 — 직관

> "최적 궤적의 어느 중간 지점에서 잘라서 봐도,
> 남은 구간은 그 지점에서 시작하는 최적 궤적이다."

```
x₀ ──────●────────────→ x_N     전체가 최적이면
          ↑
        x_t 에서 잘라도          x_t → x_N 구간도 최적
```

왜 자명한가: 만약 `x_t` 이후에 더 싼 경로가 있다면, 전체 궤적에서 그 부분만
바꿔치기해서 전체 비용을 낮출 수 있다 → 원래 궤적이 최적이라는 가정에 모순.

이 원리를 수식으로 쓰면 **Bellman 방정식** (이산시간):

```
V_t(x) = min_u [ l(x, u) + V_{t+1}(f(x, u)) ]
V_N(x) = l_f(x)
```

- `V_t(x)`: 시각 t에 상태 x에서 시작할 때 **앞으로 낼 최소 비용** (cost-to-go).
- 뒤에서 앞으로 (`t = N-1 → 0`) 계산한다 = **backward recursion**.
- 60차원 시퀀스 최적화가 **매 스텝 nu차원 최적화 N번**으로 쪼개진다.
  이것이 동적 계획법(DP)의 힘이자, LQR Riccati 재귀(§3)의 뼈대다.

### 2.2 차원의 저주

DP가 만능이 아닌 이유: `V_t(x)`를 **모든 x에 대해** 저장해야 한다.
상태를 축당 100칸 격자로 나누면:

```
nx = 1  →  100 개 값
nx = 3  →  10⁶ 개 값        (차동구동 로봇, 아직 가능)
nx = 12 →  10²⁴ 개 값       (쿼드로터, 불가능)
```

그래서 실전에서는:
- 특수 구조를 이용해 해석해 (LQR, §3)
- 현재 상태 주변에서만 국소적으로 풀기 (MPC, §4)
- V를 신경망으로 근사 (이 repo의 TD-MPPI가 terminal value에 사용:
  `mppi_controller/controllers/mppi/td_value.py`)

### 2.3 연속시간으로: HJB 방정식

Bellman 방정식에서 `dt → 0` 극한을 취하면 편미분방정식이 된다.
유도 스케치 (1줄씩):

```
V(x, t) = min_u [ l(x,u)·dt + V(x + f(x,u)·dt, t + dt) ]          Bellman, 한 스텝
        ≈ min_u [ l·dt + V(x,t) + ∂V/∂x · f·dt + ∂V/∂t · dt ]     테일러 1차 전개
0       = min_u [ l·dt + ∂V/∂x · f·dt + ∂V/∂t · dt ]              양변에서 V(x,t) 소거
```

`dt`로 나누면 **Hamilton–Jacobi–Bellman (HJB) 방정식**:

```
-∂V/∂t = min_u [ l(x, u) + (∂V/∂x)ᵀ f(x, u) ]        (결정론적 버전)
 V(x, T) = l_f(x)                                      (종단 조건)
```

**확률적 동역학** `dx = f(x,u)dt + B(x)dw` (w: 브라운 운동)이면 이토 보정항이 추가된다:

```
-∂V/∂t = min_u [ l + (∂V/∂x)ᵀ f + ½ tr(Bᵀ (∂²V/∂x²) B) ]         (확률적 HJB)
```

이 확률적 HJB가 **MPPI의 출발점**이다 — 02 문서 §2에서 log 변환으로
이 비선형 PDE를 선형화하고, Feynman-Kac으로 "기댓값 = 경로적분"으로 바꾼다.
([docs/MPPI_THEORY.md §1.2](../MPPI_THEORY.md)에도 요약이 있다.)

### 2.4 1차원 워크드 예제: HJB를 손으로 확인

가장 단순한 문제로 HJB가 실제로 작동하는지 본다.

```
동역학: ẋ = u          (속도를 직접 제어)
비용:   J = ∫₀^∞ (x² + u²) dt        (무한 시간, 시불변 → ∂V/∂t = 0)
```

시불변이므로 HJB는:

```
0 = min_u [ x² + u² + V'(x)·u ]
```

안쪽 min은 u에 대한 이차식 → 미분해서 0:

```
∂/∂u [u² + V'(x)u] = 2u + V'(x) = 0   →   u* = -½ V'(x)
```

대입:

```
0 = x² + ¼V'² - ½V'² = x² - ¼ V'(x)²
→ V'(x)² = 4x²  →  V'(x) = 2|x|... 후보: V(x) = x²  (V ≥ 0, V(0)=0 이어야 함)
```

검산: `V = x²` → `V' = 2x` → `0 = x² + ¼(2x)² - ½(2x)(2x)·½ ✓`
최적 제어는:

```
u*(x) = -½ V'(x) = -x        ← 단순 비례 피드백!
```

**교훈**: 최적 제어의 답은 "시퀀스"가 아니라 **피드백 법칙 u*(x)** 형태로
나온다. LQR도 마찬가지 (`u = -Kx`). MPC/MPPI는 이 피드백 법칙을 명시적으로
못 구하니까, **매 스텝 open-loop 문제를 다시 풀어서 암묵적 피드백**을 만든다 (§4).

---

## 3. LQR: 해석적으로 풀리는 특별한 경우

### 3.1 문제 설정

동역학이 선형, 비용이 이차이면 Bellman 재귀가 닫힌 형태로 풀린다.

```
동역학: x_{t+1} = A x_t + B u_t
비용:   J = Σ_{t=0}^{N-1} (x_tᵀ Q x_t + u_tᵀ R u_t) + x_Nᵀ Q_f x_N
가정:   Q ⪰ 0 (준정정), R ≻ 0 (양정정 — u에 대한 min이 유일하도록)
```

### 3.2 유한 시간 LQR: Riccati 재귀 유도

**귀납 가설**: value function이 이차 형식 `V_t(x) = xᵀ P_t x` 이다.

**Base case**: `V_N(x) = xᵀ Q_f x` → `P_N = Q_f`. ✓

**귀납 스텝**: `V_{t+1}(x) = xᵀ P_{t+1} x` 가정하고 Bellman 방정식에 대입:

```
V_t(x) = min_u [ xᵀQx + uᵀRu + (Ax + Bu)ᵀ P_{t+1} (Ax + Bu) ]
```

괄호를 전개 (P ≡ P_{t+1}로 줄여 씀):

```
= min_u [ xᵀQx + uᵀRu + xᵀAᵀPAx + 2uᵀBᵀPAx + uᵀBᵀPBu ]
```

u에 대한 이차식이므로 ∂/∂u = 0:

```
2Ru + 2BᵀPAx + 2BᵀPBu = 0
(R + BᵀPB) u = -BᵀPA x
u* = -(R + BᵀPB)⁻¹ BᵀPA x  ≡  -K_t x        ← 선형 피드백!
```

u*를 다시 대입하고 정리하면 (전개 연습은 연습문제 2):

```
P_t = Q + AᵀP_{t+1}A - AᵀP_{t+1}B (R + BᵀP_{t+1}B)⁻¹ BᵀP_{t+1}A
```

이것이 **이산시간 Riccati 재귀 (backward)**. 정리하면:

```
알고리즘 (Backward Riccati Recursion)
─────────────────────────────────────
P_N = Q_f
for t = N-1, N-2, …, 0:
    K_t = (R + Bᵀ P_{t+1} B)⁻¹ Bᵀ P_{t+1} A
    P_t = Q + Aᵀ P_{t+1} A - Aᵀ P_{t+1} B K_t
최적 제어: u_t = -K_t x_t
최적 비용: J* = x₀ᵀ P₀ x₀
```

### 3.3 이 repo와의 연결 — F-MPPI, Tube-MPPI

위 재귀가 이 repo에 **문자 그대로** 구현되어 있다.

**F-MPPI** (`mppi_controller/controllers/mppi/feedback_mppi.py`,
`_solve_riccati()` — 335행 근처):

```python
# P_N = Qf_matrix
# P_t = Q + A^T P_{t+1} A
#       - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A
# K_t = -(R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A
for t in range(N - 1, -1, -1):
    BtP = B.T @ P
    M = R_matrix + BtP @ B + reg
    K_t = -np.linalg.solve(M, BtP @ A)
    P = Q_matrix + A.T @ P @ A + A.T @ P @ B @ K_t
```

차이점: F-MPPI의 A, B는 상수가 아니라 **MPPI 명목 궤적을 따라 유한 차분으로
선형화한 시변 야코비안** `A_t = ∂f/∂x|_{x̄_t,ū_t}, B_t = ∂f/∂u`. 즉
"time-varying LQR around a nominal trajectory" — iLQR의 backward pass와 동일한
구조다. MPPI가 명목 궤적을 만들고, LQR이 그 주변 보정을 담당한다.

**Tube-MPPI** (`mppi_controller/controllers/mppi/tube_mppi.py` +
`ancillary_controller.py`): `AncillaryController(K_fb)`가 고정 게인
`u_fb = K_fb (x_nominal - x_actual)` 피드백을 수행한다. 이 `K_fb`를 설계하는
원칙적 방법이 바로 LQR이다 (repo 구현은 body-frame 변환 후 수동 튜닝 게인도
허용하지만, 이론적 근거는 LQR). Robust-MPPI(`robust_mppi.py`)도 같은
`AncillaryController`를 샘플링 루프 **안**에서 재사용한다.

### 3.4 무한 시간 LQR과 폐루프 안정성

`N → ∞`이고 시불변이면 `P_t`가 상수 `P_∞`로 수렴한다 (조건: (A,B) 가제어,
(A, Q^{1/2}) 가관측). 재귀의 고정점이 **대수 Riccati 방정식 (DARE)**:

```
P = Q + AᵀPA - AᵀPB (R + BᵀPB)⁻¹ BᵀPA
```

고정 게인 `K = (R + BᵀPB)⁻¹BᵀPA`, 폐루프 `x_{t+1} = (A - BK) x_t`.

**안정성 증명 (Lyapunov 논법 — MPC 안정성 증명의 원형이므로 꼭 이해할 것)**:

`V(x) = xᵀPx`를 Lyapunov 함수 후보로 삼는다. Bellman 방정식에서:

```
V(x_t) = x_tᵀQx_t + u_tᵀRu_t + V(x_{t+1})
→ V(x_{t+1}) - V(x_t) = -(x_tᵀQx_t + u_tᵀRu_t) ≤ 0
```

즉 **V는 폐루프 궤적을 따라 단조 감소**하며, 감소량이 stage cost와 같다.
Q ≻ 0이면 `x ≠ 0`에서 강감소 → 점근 안정. 이 "value function =
Lyapunov 함수" 트릭이 §4.4의 MPC 안정성 증명에 그대로 재활용된다.

### 3.5 워크드 예제: 더블 적분기 LQR 손계산 (N=2)

§1.3 문제, `dt = 1`로 단순화:

```
A = [1 1]    B = [0]    Q = I,  R = [1],  Q_f = I,  N = 2
    [0 1]        [1]
```

**t = 2**: `P_2 = Q_f = I`

**t = 1**:
```
BᵀP₂B = [0 1][0;1] = 1        → R + BᵀPB = 2
BᵀP₂A = [0 1][1 1;0 1] = [0 1]
K_1 = ½ [0 1] = [0, 0.5]
P_1 = I + AᵀA - Aᵀ B ½ Bᵀ A
    = I + [1 1;1 2] - ½[0 0;0 1][1 1;0 1]…  (전개하면)
    = [2   1  ]
      [1   2.5]
```

**t = 0** (같은 방식, 숫자만 확인):
```
BᵀP₁B = 2.5 → R + BᵀPB = 3.5
BᵀP₁A = [1, 3.5]
K_0 = (1/3.5)[1, 3.5] ≈ [0.286, 1.0]
```

해석: `u_0 = -0.286·p - 1.0·v`. 속도에 대한 게인이 위치보다 크다 —
"먼저 속도를 죽여야 위치가 잡힌다"는 물리 직관과 일치.
(검산 스니펫은 연습문제 3.)

---

## 4. MPC: receding horizon의 아이디어

### 4.1 왜 open-loop 최적해를 반복 계산하는가

LQR은 피드백 법칙 `u = -Kx`를 **미리** 준다. 하지만:
- 비선형 동역학 → 닫힌 형태의 피드백 법칙이 없다
- 제약 (`u ∈ U`, 장애물) → LQR로 다룰 수 없다
- 전역 피드백 법칙 계산 = DP = 차원의 저주 (§2.2)

**MPC의 타협**: 전역 피드백 법칙을 포기하고, **지금 있는 상태 x에서만**
유한 호라이즌 open-loop 문제를 푼다. 그리고 첫 입력만 쓰고 버린다.

```
매 제어 주기마다:
  1. 현재 상태 x 측정
  2. x에서 시작하는 N-스텝 OCP를 풀어 U* = (u₀*, …, u*_{N-1}) 획득
  3. u₀*만 적용
  4. 한 스텝 후 새 상태에서 1로 복귀 (호라이즌이 한 칸 "물러남" = receding)
```

```
t=0:  [━━━━━━ 계획 (N스텝) ━━━━━━]
       ▲ u₀* 적용
t=1:     [━━━━━━ 재계획 ━━━━━━━━]
          ▲ u₀* 적용
t=2:        [━━━━━━ 재계획 ━━━━━]
             ▲ …
────────────────────────────────────→ 시간
```

**핵심 통찰**: 각각의 풀이는 open-loop지만, **매 스텝 실측 상태로 다시 풀기
때문에** 전체 시스템은 폐루프 피드백이 된다. 외란이 오면 다음 스텝의 초기
조건이 달라지고, 계획이 자동으로 수정된다. 즉:

> MPC = "암묵적으로 정의된 피드백 법칙 u = κ_MPC(x)"
> (κ를 수식으로 못 쓸 뿐, 매 스텝 최적화로 그 값을 계산한다)

이 구조는 MPPI도 **완전히 동일**하다. `base_mppi.py`의 `compute_control()`
마지막을 보라:

```python
# 7. 다음 스텝을 위한 시프트 (receding horizon)
self.U = np.roll(self.U, -1, axis=0)
self.U[-1, :] = 0.0
# 8. 최적 제어 반환 (첫 번째 제어)
optimal_control = self.U[0, :]
```

`np.roll` 시프트 = 이전 해를 다음 스텝의 초기 추정치로 재활용하는
**warm start**. MPC 솔버들도 똑같은 트릭을 쓴다 (해가 스텝 간에 크게 변하지
않는다는 가정).

### 4.2 유한 호라이즌의 함정: 근시안 문제

호라이즌 N이 짧으면 "지금 당장 싼" 행동이 나중에 비싸질 수 있다.

```
        절벽
  ●──→ ─────╲
   로봇      ╲  N스텝 안에는 절벽이 안 보임
              ╲───── → 낭떠러지
```

고전적 예: 감속에 5스텝이 필요한데 N=3이면, 최적화는 절벽 직전까지
전속력을 선택한다. 해결책 두 갈래:

1. **N을 늘린다** → 계산량 증가 (MPPI에선 rollout 길이 증가)
2. **terminal cost `l_f` / terminal set으로 "호라이즌 너머"를 요약한다** ← 정석

이 repo의 **TD-MPPI** (`td_mppi.py`)가 정확히 2번 접근: 학습된 가치함수
`V(x_N)`을 terminal cost로 써서 N=10으로 N=30급 성능을 낸다
([MPPI_THEORY.md §25](../MPPI_THEORY.md) 참조). §2의 DP 관점에서 보면
`l_f = V_∞`가 이상적 — terminal cost가 정확한 cost-to-go면 N=1도 최적이다.

### 4.3 Recursive feasibility (재귀적 실행가능성)

**정의**: 시각 t에 OCP의 해가 존재했다면, 시각 t+1에도 (외란이 없을 때)
해가 존재함이 보장되는 성질.

왜 문제가 되나: 제약이 있는 MPC에서, 이번 스텝의 최적 행동이 다음 스텝에
**해가 아예 없는** 상태로 시스템을 몰고 갈 수 있다 (예: 장애물에 너무
접근해서 어떤 u로도 충돌 회피 불가).

**표준 해법 — terminal set**: 종단 상태를 제어 불변 집합(control invariant
set) `X_f`에 넣도록 강제한다:

```
x_N ∈ X_f    where    ∀x ∈ X_f, ∃u: f(x,u) ∈ X_f  (한번 들어가면 머물 수 있음)
```

**증명 아이디어 (꼬리 붙이기, tail argument)**:

```
시각 t의 해:    (u₀*, u₁*, …, u*_{N-1})  →  x_N* ∈ X_f
시각 t+1 후보:  (u₁*, u₂*, …, u*_{N-1}, κ_f(x_N*))
                 └── 이전 해 시프트 ──┘   └ X_f 안에 머무는 보조 제어 ┘
```

이 후보는 모든 제약을 만족한다 (앞부분은 이미 검증됨, 마지막은 불변성) →
feasible한 해가 최소 하나 존재 → 재귀적 실행가능. ∎

MPPI에서의 대응물: MPPI는 hard 제약 대신 비용 페널티를 쓰므로 "infeasible"이
문법적으로는 없지만, **물리적으로 회피 불가능한 상태**는 존재한다. 이 repo의
대응이 바로 안전 계열 변형들이다:
- **C-MPPI** (`contingency_mppi.py`): 모든 계획 checkpoint에서 비상 탈출
  (braking/inner MPPI) 가능성을 검사 — recursive feasibility의 샘플링 버전
- **DualGuard-MPPI** (`dualguard_mppi.py`): HJ 안전 가치함수 = 최대 안전
  불변 집합의 근사
- **Gatekeeper** (`gatekeeper.py`): 검증된 backup 궤적이 있을 때만 새 계획 수용

### 4.4 안정성: terminal cost가 Lyapunov 함수를 만든다

LQR 증명(§3.4)의 일반화. 조건:

```
(A1) l(x,u) ≥ α(‖x‖)  (양의 stage cost)
(A2) X_f 는 제어 불변, 그 위에서 보조 제어 κ_f 존재
(A3) l_f(f(x, κ_f(x))) - l_f(x) ≤ -l(x, κ_f(x))   ∀x ∈ X_f
     ("terminal cost는 X_f 안에서 local Lyapunov 함수")
```

**정리**: 위 조건에서 MPC value function `V_N(x)` (OCP 최적값)은
폐루프의 Lyapunov 함수이고, 원점은 점근 안정.

**증명 스케치 (한 줄 논리)**: §4.3의 시프트+꼬리 후보 해의 비용은

```
J(후보) = V_N(x_t) - l(x_t, u₀*) + [ l(x_N*, κ_f) + l_f(x_{N+1}) - l_f(x_N*) ]
        ≤ V_N(x_t) - l(x_t, u₀*)              (A3에 의해 대괄호 ≤ 0)
```

최적값은 후보보다 작거나 같으므로:

```
V_N(x_{t+1}) ≤ V_N(x_t) - l(x_t, u₀*)   →   V_N 단조 감소  ∎
```

**기억할 것**: "terminal cost는 장식이 아니라 안정성 증명의 핵심 부품"이다.
`Qf`를 `Q`보다 크게 잡는 관행 (이 repo `MPPIParams.Qf` 기본값도 그렇다)은
(A3)를 근사적으로 만족시키려는 휴리스틱이다 — 이상적으로는 `l_f = xᵀP_∞x`
(무한 시간 LQR의 P)로 두면 (A3)가 등식으로 성립한다.

---

## 5. 제약 처리: hard vs soft, QP 정식화

### 5.1 Hard vs Soft 제약

| | Hard 제약 | Soft 제약 (페널티) |
|---|-----------|-------------------|
| 형태 | `g(x,u) ≤ 0` 을 반드시 만족 | 비용에 `ρ·max(0, g)²` 등 추가 |
| 위반 | 절대 불가 (해가 없으면 infeasible) | 가능하되 비쌈 |
| 적합 | 액추에이터 한계 (물리적으로 불가능한 건 hard) | 장애물 여유, 승차감 |
| 위험 | infeasibility → 솔버 실패 → 제어 공백 | ρ 튜닝 실패 시 위반 발생 |

실무 정석: **입력 제약은 hard** (어차피 하드웨어가 강제),
**상태 제약은 slack 변수로 soft화**해서 infeasibility 방지:

```
minimize  J + ρ‖s‖₁        (ρ 충분히 크면 s=0인 해가 있을 때 hard와 동일 — exact penalty)
s.t.      g(x_t) ≤ s_t,  s_t ≥ 0
```

**MPPI의 선택**: 거의 전부 soft. `base_mppi.py`에서 유일한 hard 제약은
입력 클리핑 `np.clip(sampled_controls, u_min, u_max)` (145행)뿐이고,
장애물은 비용 함수(`cost_functions.py`의 obstacle cost, `cbf_cost.py` 등)로
처리한다. hard 상태 제약이 필요하면 별도 장치가 필요하다:
- **pi-MPPI** (`projection_mppi.py`): 샘플을 제약 집합에 QP/clip **투영** —
  jerk/snap hard 제약 보장
- **CSC-MPPI** (`csc_mppi.py`): primal-dual 투영 + 클러스터링
- **CBF safety filter** (`cbf_safety_filter.py`, `clf_cbf_qp.py`):
  MPPI 출력 뒤에 QP 한 번 더 (최소 수정 필터)

### 5.2 Linear MPC의 QP 정식화 (condensed form)

선형 동역학 + 이차 비용 + 선형 제약이면 MPC OCP는 **QP**다. 유도:

상태를 시뮬레이션으로 소거한다 (`x_{t+1} = Ax_t + Bu_t` 반복 대입):

```
x₁ = Ax₀ + Bu₀
x₂ = A²x₀ + ABu₀ + Bu₁
⋮
┌ x₁ ┐   ┌ A  ┐        ┌ B          0    ⋯ ┐ ┌ u₀   ┐
│ x₂ │ = │ A² │ x₀  +  │ AB         B      │ │ u₁   │
│ ⋮  │   │ ⋮  │        │ ⋮          ⋱      │ │ ⋮    │
└ x_N┘   └ A^N┘        └ A^{N-1}B  ⋯    B ┘ └u_{N-1}┘
  X    =   Φ x₀     +           Γ            U
```

비용에 대입:

```
J = XᵀQ̄X + UᵀR̄U               (Q̄ = blkdiag(Q,…,Q,Q_f), R̄ = blkdiag(R,…,R))
  = (Φx₀ + ΓU)ᵀ Q̄ (Φx₀ + ΓU) + UᵀR̄U
  = Uᵀ(ΓᵀQ̄Γ + R̄)U + 2x₀ᵀΦᵀQ̄Γ U + const
       └── H ──┘      └── qᵀ ──┘
```

결과:

```
minimize  ½ UᵀHU + qᵀ(x₀)U          ← H는 고정, q만 x₀에 선형 의존
s.t.      G U ≤ b(x₀)               ← 입력/상태 제약도 U의 선형 부등식
```

- H ≻ 0 (R ≻ 0이므로) → **볼록 QP** → 전역 최적해, 신뢰성 있는 솔버 다수
  (OSQP, qpOASES). 밀리초 단위로 풀린다.
- 제약이 없으면 `U* = -H⁻¹q(x₀)` — x₀에 **선형** → LQR과 일치 (sanity check).
- 제약이 있으면 `U*(x₀)`는 **구간별 선형(piecewise affine)** — explicit MPC의
  이론적 기반 (Borrelli 교재의 주제).

**MPPI와의 대비 (02 문서 예고)**: MPPI의 업데이트 `U ← U + Σ w_k ε_k`도
`U`에 대한 반복 개선인데, H⁻¹ 같은 곡률 정보 없이 샘플 가중 평균만 쓴다.
PGD-MPPI/GN-MPPI (§7 표 참조)는 바로 이 곡률 정보를 되살리려는 시도다.

### 5.3 미니 예제: 제약이 해를 어떻게 바꾸나

§3.5 더블 적분기, `|u| ≤ 0.5` 제약 추가, 초기 상태 `p=1, v=0`:

```
무제약 LQR:  u₀ = -0.286·1 - 1.0·0 = -0.286   → 제약 안 걸림, 동일
초기 상태 p=3이면: u₀ = -0.857                 → 클리핑 → -0.5
```

단순 클리핑(saturated LQR)과 제약 인지 QP의 차이: 클리핑은 "무제약 최적을
자르는" 것이고, QP는 "잘릴 것을 **알고** 나머지 시퀀스를 재조정"한다.
호라이즌 초반이 포화되면 후반 계획이 달라지므로 QP가 항상 같거나 낫다.
(MPPI도 마찬가지 이유로 클리핑을 샘플에 **먼저** 적용하고 rollout한다 —
`base_mppi.py` 145행이 rollout **전**에 clip하는 이유.)

---

## 6. NMPC와 실시간 이슈

### 6.1 비선형이면 무엇이 깨지나

`f`가 비선형이면 §5.2의 Γ가 상수 행렬이 아니게 되고, J는 U에 대해 비볼록.
→ QP가 아니라 **NLP (비선형 계획)**. 전역 최적 보장 상실, 지역 최적/초기값
민감성 발생.

### 6.2 두 가지 이산화 전략

```
Single shooting                     Multiple shooting
───────────────                     ─────────────────
변수: U만                            변수: U와 각 구간 시작 상태 s_i
x는 x₀에서 끝까지 시뮬레이션          구간별로 짧게 시뮬레이션 +
                                     이음매 등식 제약 x(t_{i+1}; s_i) = s_{i+1}

x₀ ─▶▶▶▶▶▶▶▶▶▶ x_N                  x₀ ─▶▶ s₁ ─▶▶ s₂ ─▶▶ x_N
                                          ‖제약  ‖제약
장점: 변수 적음, 구현 단순             장점: 수치 안정 (긴 적분의 민감도 폭발 방지),
단점: 불안정 시스템에서 발산,                초기 궤적 추정 활용, 희소 구조
      비볼록성 심함                    단점: 변수 많음 (희소 솔버로 상쇄)
```

MPPI는 **single shooting의 샘플링 버전**이다: `dynamics_wrapper.rollout()`이
x₀에서 N스텝을 쭉 적분한다. shooting의 단점(민감도 폭발)을 "미분 안 하고
K개 병렬 시뮬레이션"으로 우회하는 셈.

### 6.3 SQP와 Real-Time Iteration (개요 수준)

**SQP (Sequential Quadratic Programming)**: 현재 추정해 주변에서 NLP를
QP로 근사 → QP 풀기 → 스텝 → 반복.

```
반복 k:
  1. 현재 (X̄, Ū)에서 f를 선형화 (A_t = ∂f/∂x, B_t = ∂f/∂u), 비용을 이차 근사
  2. §5.2 형태의 QP를 풀어 (ΔX, ΔU)
  3. (X̄, Ū) += α(ΔX, ΔU),  수렴까지 반복
```

**실시간 문제**: 제어 주기 (이 repo 기준 100ms) 안에 수렴까지 못 돈다.
**RTI (Real-Time Iteration, Diehl)**: "수렴할 때까지"를 포기하고
**매 제어 주기에 SQP 1회만** 수행. 이전 해 warm start 덕분에 해가 참
최적해를 "추적"한다. 준비 단계(선형화)는 측정 전에 미리 하고, 측정 후에는
QP만 푸는 분할로 지연을 최소화.

**MPPI와의 평행 구조** — 이 대응을 눈에 담아두면 02 문서가 쉬워진다:

```
SQP/RTI                              MPPI
─────────────────────────           ─────────────────────────
매 주기 QP 1회 (수렴 포기)      ↔   매 주기 샘플링 업데이트 1회
이전 해 warm start              ↔   U 시프트 (np.roll)
선형화 (야코비안 필요)           ↔   rollout K개 (야코비안 불필요)
QP 스텝 = 곡률 반영 하강         ↔   가중 평균 = 전처리 경사 하강 (PGD 해석)
```

### 6.4 실시간 체크리스트 (엔지니어링 관점)

- **최악 실행 시간(WCET)** 기준으로 설계 (평균 아님). 이 repo의 성능 기준
  "< 100ms (K=1024, N=30)"도 같은 정신 (CLAUDE.md).
- 솔버 실패/시간 초과 시 **fallback**: 이전 해 시프트 적용, 또는 backup
  컨트롤러 (repo: `backup_controller.py`, `gatekeeper.py`).
- warm start 무효화 주의: 목표가 점프하면 이전 해가 나쁜 초기값이 된다
  (repo: T-MPPI가 transformer로 초기값을 학습하는 동기 —
  `mppi_controller/controllers/mppi/transformer_mppi.py`,
  [MPPI_THEORY.md §32](../MPPI_THEORY.md) 참조).

---

## 7. MPC vs MPPI — 왜 이 repo는 MPPI인가

### 7.1 비교표

| 축 | 그래디언트 MPC (QP/SQP) | MPPI (샘플링) |
|---|---|---|
| **동역학 요구** | 미분 가능 (∂f/∂x, ∂f/∂u) | **블랙박스 시뮬레이터면 충분** (신경망 동역학, 접촉, 불연속 OK) |
| **비용 요구** | 매끄러움 (2차 미분까지 쓰면 더) | **불연속/지표함수 OK** (충돌 = ∞ 페널티 가능) |
| **비볼록/다중모달** | 지역 최적 수렴, 초기값 민감 | 분포로 탐색 — 장애물 좌/우 같은 다중 모달 자연 처리 |
| **제약 처리** | ★ 강점: hard 제약 체계적 (QP/NLP) | 약점: soft 페널티 기본, hard는 투영/필터 추가 필요 |
| **최적성** | 볼록이면 전역 최적, 수렴 증명 | 유한 K에서 근사, 분산 있음 (λ, K에 의존) |
| **병렬화** | 어려움 (순차적 인수분해) | ★ 강점: K개 rollout 완전 병렬 = GPU 친화 |
| **계산 특성** | 저차원·매끈한 문제에서 빠르고 정밀 | 고차원 샘플 수 요구, 대신 스텝당 비용 예측 가능 |
| **구현 복잡도** | 솔버 스택 (CasADi/acados/OSQP) | numpy 몇 십 줄 (`base_mppi.py`가 326줄 전부) |
| **이론 성숙도** | 안정성/feasibility 정리 완비 (§4) | 안정성 이론 상대적으로 미성숙 (연구 전선) |

### 7.2 이 repo가 MPPI 계열인 이유 (요약 논증)

1. **학습 모델과의 결합이 1급 목표다.** repo에 학습 동역학
   (`mppi_controller/models/learned/` — BNN, world model, Koopman)이 14종
   있는데, 신경망 동역학의 야코비안 기반 NMPC는 취약한 반면 MPPI는 forward
   pass만 있으면 된다.
2. **장애물 비용이 비볼록·다중모달이다.** 벤치마크 시나리오(local_minima,
   dense_obstacles)가 정확히 그래디언트 MPC가 지역해에 갇히는 상황.
3. **GPU 병렬화.** `base_mppi.py`의 `_compute_control_gpu()`가 K=1024
   rollout을 통째로 GPU에서 돈다.
4. **약점은 변형으로 보완한다** — 이것이 43개 변형의 존재 이유다:

```
MPC의 강점              →  그걸 흡수한 MPPI 변형 (이 repo)
──────────────────────────────────────────────────────────
hard 제약 (QP)          →  pi-MPPI(투영), CSC-MPPI, CBF-QP 필터
피드백 게인 (LQR)       →  F-MPPI(Riccati), Tube/Robust-MPPI(ancillary)
곡률 활용 (뉴턴/SQP)    →  GN-MPPI(가우스-뉴턴), PGD/TR-MPPI(전처리/신뢰영역)
recursive feasibility  →  C-MPPI(비상 계획), Gatekeeper, DualGuard(HJ)
terminal ingredient    →  TD-MPPI(학습 가치함수)
```

즉 이 repo의 변형 지도를 "MPC 이론의 어떤 조각을 샘플링 프레임워크에
이식했는가"로 읽을 수 있다. 상세 분류는
[02_MPPI_FUNDAMENTALS.md §7](02_MPPI_FUNDAMENTALS.md#7-변형-분류-체계)에서.

---

## 8. 연습문제

**문제 1 (Bellman 재귀, 손계산).**
스칼라 시스템 `x_{t+1} = x_t + u_t`, 비용 `Σ_{t=0}^{1} (x_t² + u_t²) + x_2²`,
`x_0 = 1`. Bellman 재귀를 t=2부터 손으로 돌려 `V_1(x)`, `V_0(x)`와 최적
`u_0, u_1`을 구하라. (힌트: `V_t(x) = p_t x²` 꼴을 유지한다.
답: `p_2 = 1, p_1 = 3/2, p_0 = 8/5`, `u_0 = -3/5`.)

**문제 2 (Riccati 유도 완성).**
§3.2에서 `u* = -Kx`를 Bellman 방정식에 되대입하여
`P_t = Q + AᵀPA - AᵀPB(R + BᵀPB)⁻¹BᵀPA`를 유도하라. 그리고 이 식이
`P_t = Q + KᵀRK + (A - BK)ᵀP(A - BK)` (Joseph form)와 동치임을 보여라 —
후자는 수치적으로 P의 대칭성/준정정성을 보존해서 실무에서 선호된다.
(`feedback_mppi.py`의 `P = 0.5*(P + P.T)` 대칭화가 왜 필요한지와 연결해 볼 것.)

**문제 3 (LQR 검산 코드).**
§3.5의 손계산을 다음 스니펫으로 검증하고, N을 2 → 50으로 늘리며 `K_0`가
어디로 수렴하는지 관찰하라 (무한 시간 게인). `scipy.linalg.solve_discrete_are`
결과와 비교하라.

```python
import numpy as np
A = np.array([[1., 1.], [0., 1.]]); B = np.array([[0.], [1.]])
Q = np.eye(2); R = np.array([[1.]]); P = np.eye(2)   # P_N = Qf
for t in range(50):
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    P = Q + A.T @ P @ A - A.T @ P @ B @ K
    print(t, K.ravel())
```

**문제 4 (recursive feasibility 반례 만들기).**
더블 적분기 `|u| ≤ 1`, 상태 제약 `p ≤ 10` (벽), terminal set 없음, N = 3,
dt = 1. 초기 상태 `p = 0`에서 어떤 속도 `v`부터 "시각 0에는 feasible하지만
최적 입력을 적용하면 시각 1에 infeasible"이 되는지 구성해 보라.
(힌트: 정지거리 `v²/2 > 남은 거리`가 되는 순간을 N스텝 창이 못 보는 경우.)
그런 다음 terminal 제약 `v_N = 0`을 추가하면 왜 문제가 사라지는지 설명하라.

**문제 5 (MPC vs MPPI 실험).**
이 repo에서 직접 확인: 장애물 시나리오에서 그래디언트 방법의 지역해 문제를
MPPI가 어떻게 피하는지 본다.

```bash
PYTHONPATH=. python examples/comparison/drpa_mppi_benchmark.py --all-scenarios
```

local_minima 시나리오에서 Vanilla MPPI조차 갇힐 수 있음을 관찰하고
(DRPA가 탈출시키는 것도), "샘플링이면 비볼록성이 공짜로 해결"이 아니라
**분포가 덮는 범위 안에서만** 전역 탐색이 됨을 확인하라. σ를 키우면
탈출이 쉬워지는 대신 무엇이 나빠지는가? (02 문서 §4의 σ 트레이드오프로 이어짐.)

---

## 9. 부록 — 더 공부하기 위한 자료

> 본문을 다 읽은 뒤의 자습(self-study) 가이드. 외부 링크는 2026-07 기준
> 접근/존재를 확인한 것만 실었다. MPPI 쪽 심화 자료는
> [02_MPPI_FUNDAMENTALS.md §10](02_MPPI_FUNDAMENTALS.md) 부록이 담당한다.

### 9.1 주석 달린 핵심 레퍼런스

읽는 순서 추천: ① → ②(필요한 장만) → ④/⑤(수치 기법) → 이후는 관심 분기
(학습 결합이면 ⑦⑧, 안전이면 ⑨, RL이면 ⑩).

1. **Mayne, Rawlings, Rao, Scokaert, "Constrained model predictive control:
   Stability and optimality," *Automatica*, 2000.**
   terminal cost + terminal set으로 안정성과 recursive feasibility를 증명하는
   §4.3-4.4 프레임의 원논문. §4의 증명 스케치를 엄밀한 버전으로 확인하고
   싶어질 때 읽는다 — MPC 이론의 사실상 "정본"이다.

2. **Rawlings, Mayne, Diehl, *Model Predictive Control: Theory, Computation,
   and Design* (2nd ed., Nob Hill, 2017) —
   [저자 공개 PDF](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-5th-printing.pdf).**
   DP·안정성·강건 MPC·수치 기법까지 덮는 표준 교과서로, 저자가 전체 PDF를
   무료 공개했다. 이 문서 §2/§4의 원전이므로 처음부터 통독하기보다
   해당 장(Ch.1-2)을 사전처럼 찾아 읽는 용도로 좋다.

3. **Borrelli, Bemporad, Morari, *Predictive Control for Linear and Hybrid
   Systems* (Cambridge, 2017).**
   QP 정식화, explicit MPC, 불변 집합 — §5의 원전. hard 제약과 feasibility를
   집합 이론 수준에서 제대로 다루고 싶을 때 읽는다.

4. **Diehl, Bock, Schlöder, "A real-time iteration scheme for nonlinear
   optimization in optimal feedback control," *SIAM J. Control Optim.*, 2005.**
   §6.3 RTI의 원전 — "매 주기 SQP 1회 + warm start"가 참 해를 추적함을 보인다.
   MPPI의 "매 주기 샘플링 1회 + U 시프트" 구조가 왜 작동하는지의 이론적
   사촌이므로, §6.3의 평행 구조 표를 본 뒤 읽으면 좋다.

5. **Andersson, Gillis, Horn, Rawlings, Diehl, "CasADi: a software framework
   for nonlinear optimization and optimal control," *Math. Prog. Comp.*, 2019 —
   [Springer](https://link.springer.com/article/10.1007/s12532-018-0139-4).**
   NMPC 프로토타이핑의 사실상 표준 도구 논문 (자동 미분 + 솔버 인터페이스).
   §5-6의 QP/NLP를 코드로 직접 만들어 보고 싶어지는 시점에 읽는다.

6. **Verschueren et al., "acados — a modular open-source framework for fast
   embedded optimal control," 2019 —
   [arXiv:1910.13753](https://arxiv.org/abs/1910.13753).**
   RTI 계열 실시간 NMPC를 임베디드 C 코드로 뽑아주는 프레임워크 논문.
   "그래디언트 MPC의 실전 성능이 어디까지 왔나"를 가늠할 때 —
   §7 비교표의 MPC 쪽 최전선이다.

7. **Amos, Jimenez, Sacks, Boots, Kolter, "Differentiable MPC for End-to-end
   Planning and Control," NeurIPS 2018 —
   [arXiv:1810.13400](https://arxiv.org/abs/1810.13400).**
   MPC 자체를 미분 가능한 정책 계층으로 만들어 비용/동역학을 end-to-end로
   학습한다 (KKT 조건을 통한 미분). 이 repo의 미분 가능 시뮬레이터
   (`mppi_controller/models/differentiable/`)와 같은 정신이므로 축 5 학습
   변형(02 문서 §7.5)을 본 뒤 읽으면 연결이 보인다.

8. **Hewing, Wabersich, Menner, Zeilinger, "Learning-Based Model Predictive
   Control: Toward Safe Learning in Control," *Annual Review of Control,
   Robotics, and Autonomous Systems*, 2020 —
   [Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev-control-090419-075625).**
   학습 동역학/학습 비용/안전 보증을 아우르는 learning-based MPC 지도 논문.
   이 repo의 학습 모델 14종이 MPC 진영에서는 어떻게 다뤄지는지 조감할 때 읽는다.

9. **Wabersich, Zeilinger, "A predictive safety filter for learning-based
   control of constrained nonlinear dynamical systems" —
   [arXiv:1812.05506](https://arxiv.org/abs/1812.05506).**
   임의의 (학습) 정책 출력을 받아 "안전하면 통과, 아니면 최소 수정"하는
   MPC 기반 safety filter의 원전. repo의 CBF-QP 필터
   (`cbf_safety_filter.py`)와 동일한 필터 아키텍처의 MPC판 — §4.3의
   recursive feasibility가 실전에서 어떻게 쓰이는지 보여준다.

10. **Bertsekas, "Model Predictive Control and Reinforcement Learning:
    A Unified Framework Based on Dynamic Programming," 2024 —
    [arXiv:2406.00592](https://arxiv.org/abs/2406.00592).**
    MPC와 RL을 "Newton 스텝 = 온라인 재계획, 기저 정책 = 오프라인 학습"으로
    통합하는 관점 논문. §2의 DP를 이해한 뒤 "MPC냐 RL이냐" 논쟁을 정리하고
    싶을 때 읽는다.

11. **Kirk, *Optimal Control Theory: An Introduction* (Dover).**
    변분법·Pontryagin 최소 원리·HJB를 다루는 고전 입문서 (저렴한 Dover판).
    §2의 HJB 유도가 압축적으로 느껴졌다면 이 책으로 기초를 보강한다.

12. **Bertsekas, *Dynamic Programming and Optimal Control*, Vol. 1.**
    Bellman 원리와 DP의 정석 — §2의 원전. LQR·확률 DP·근사 DP까지
    체계적으로 쌓고 싶을 때 (TD-MPPI의 가치함수 학습 배경이기도 하다).

### 9.2 최근 연구 동향 (2024–2026)

MPC 진영이 어디로 가고 있는지 5개 흐름. 각 흐름이 이 repo의 어떤 부분과
공명하는지 함께 적었다.

1. **미분 가능 MPC의 성숙 — mpc.pytorch에서 Theseus 계보로.**
   Amos의 mpc.pytorch(2018)에서 시작한 "최적화 계층을 신경망에 삽입" 노선이
   Meta의 Theseus ([arXiv:2207.09442](https://arxiv.org/abs/2207.09442),
   희소 솔버 + 배치/GPU + 암묵적 미분) 같은 범용 라이브러리로 성숙했고,
   최근에는 GPU 위 미분 가능 MPC
   ([arXiv:2510.06179](https://arxiv.org/abs/2510.06179))처럼 대규모 병렬화와
   결합되고 있다. repo 대응: `models/differentiable/`, Step-MPPI의 DPC 학습.

2. **실시간 NMPC 솔버 생태계의 표준화 — acados 중심.**
   acados ([arXiv:1910.13753](https://arxiv.org/abs/1910.13753))가 임베디드
   NMPC의 사실상 표준이 되면서, 학습 모델(GP 등)을 acados 파이프라인에
   꽂는 L4acados ([arXiv:2411.19258](https://arxiv.org/abs/2411.19258)) 같은
   확장이 나오고 있다. §6.3 RTI가 이 생태계의 이론적 핵심이다.

3. **Safety filter로서의 MPC.**
   "성능은 학습 정책이, 안전은 MPC가"라는 분업 —
   predictive safety filter ([arXiv:1812.05506](https://arxiv.org/abs/1812.05506)),
   레이싱 적용 ([arXiv:2102.11907](https://arxiv.org/abs/2102.11907)),
   필터 자체의 안정성 분석
   ([arXiv:2404.05496](https://arxiv.org/abs/2404.05496))으로 이어졌다.
   repo의 `gatekeeper.py`/`cbf_safety_filter.py`가 정확히 이 아키텍처의
   샘플링 진영 대응물이다.

4. **RL + MPC 결합의 체계화.**
   Bertsekas의 통합 프레임 ([arXiv:2406.00592](https://arxiv.org/abs/2406.00592))
   이후, RL로 MPC의 비용/파라미터/기저 정책을 학습하는 아키텍처 비교 연구
   ([arXiv:2510.03354](https://arxiv.org/abs/2510.03354))가 활발하다.
   repo의 Residual-MPPI(사전 정책 + 잔차 최적화), TD-MPPI(학습 terminal
   value)가 같은 질문의 샘플링판 답이다.

5. **Foundation-model-guided MPC.**
   VLM/LLM이 목표·비용·중간 계획을 생성하고 MPC가 저수준 실행을 담당하는
   계층 구조 — VLMPC (RSS 2024), VLM 기반 조작 계획·궤적 생성
   ([arXiv:2504.05225](https://arxiv.org/abs/2504.05225)) 등. "비용 함수를
   손으로 설계하지 않는다"는 점에서 축 5(학습 결합)의 극단이다.

### 9.3 오픈소스 생태계

직접 설치해서 §5-6의 개념을 실험해 볼 수 있는 도구들 (존재/활성 여부
2026-07 확인).

| 이름 | 링크 | 언어 | 특징 | 이 repo와의 관계 |
|------|------|------|------|-----------------|
| acados | [docs.acados.org](https://docs.acados.org/) | C (Python/MATLAB 인터페이스) | 임베디드 실시간 NMPC, RTI, BLASFEO 기반 | 그래디언트 NMPC 성능 기준선 — §7 비교표의 반대편 실물 |
| CasADi | [web.casadi.org](https://web.casadi.org/) | C++ (Python/MATLAB) | 자동 미분 + NLP 모델링, acados/rockit의 기반 | §5.2 QP·§6 NLP를 손으로 짜볼 때의 표준 도구 |
| do-mpc | [github.com/do-mpc/do-mpc](https://github.com/do-mpc/do-mpc) | Python | robust/multi-stage MPC + MHE, 교육 친화적 문서 | Tube-MPPI(§3.3)와 robust MPC를 나란히 실험하기 좋음 |
| rockit | [github.com/meco-group/rockit](https://github.com/meco-group/rockit) | Python | OCP 신속 프로토타이핑 (CasADi Opti 기반), multi-stage/free end-time | 연습문제 4 같은 소형 OCP를 빠르게 정식화해 검산 |
| HILO-MPC | [github.com/hilo-mpc/hilo-mpc](https://github.com/hilo-mpc/hilo-mpc) | Python | 학습 모델(TensorFlow/PyTorch)을 MPC에 직접 삽입 ([arXiv:2203.13671](https://arxiv.org/abs/2203.13671)) | repo `models/learned/`와 같은 목표의 MPC 진영 구현 |
| OCS2 | [github.com/leggedrobotics/ocs2](https://github.com/leggedrobotics/ocs2) | C++ | SLQ/DDP 계열 + SQP/IPM, 사족보행 실기 검증 (ETH RSL) | F-MPPI의 Riccati backward pass(§3.3)와 같은 계열의 산업급 구현 |
| Crocoddyl | [github.com/loco-3d/crocoddyl](https://github.com/loco-3d/crocoddyl) | C++ (Python 바인딩) | 접촉 하 DDP 최적 제어 ([arXiv:1909.04947](https://arxiv.org/abs/1909.04947)), Pinocchio 기반 | GN-MPPI가 흉내 내는 "2차 정보 활용"의 정통 구현 |
| mpc.pytorch | [github.com/locuslab/mpc.pytorch](https://github.com/locuslab/mpc.pytorch) | Python (PyTorch) | 미분 가능 MPC 솔버 (박스 제약 iLQR) | §9.2 동향 1의 출발점 — repo 미분 가능 시뮬레이터와 비교 |
| Theseus | [github.com/facebookresearch/theseus](https://github.com/facebookresearch/theseus) | Python (PyTorch) | 미분 가능 비선형 최소제곱, GPU 배치, 암묵적 미분 | 학습 파이프라인에 최적화 계층을 넣을 때의 현대적 선택지 |

### 9.4 더 공부하기 — 교재·강의·튜토리얼

**교재 (무료 공개 확인)**
- Rawlings, Mayne, Diehl —
  [PDF 무료](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-5th-printing.pdf) (§9.1-②)
- Gros & Diehl, *Numerical Optimal Control* (draft) —
  [PDF 무료](https://www.syscop.de/files/2024ws/NOC/book-NOCSE.pdf).
  shooting/collocation/SQP/RTI — §6 전체의 교과서 버전.
- Boyd & Vandenberghe, *Convex Optimization* —
  [PDF 무료](https://stanford.edu/~boyd/cvxbook/). §5의 QP가 왜 "풀리는
  문제"인지의 기초 체력.
- Borrelli, Bemporad, Morari (Cambridge, 2017) — 유료지만 §5 심화의 정석.
- Kirk (Dover) / Bertsekas Vol.1 / Anderson & Moore *Optimal Control:
  Linear Quadratic Methods* — §2-3 보강용 고전.

**공개 강의**
- Boyd, Stanford EE364A (Convex Optimization) —
  [강의 영상+자료](https://see.stanford.edu/Course/EE364A). LQR 심화는
  Boyd의 EE363 (Linear Dynamical Systems) 강의 노트.
- Tedrake, *Underactuated Robotics* (MIT 6.832) —
  [OCW](https://ocw.mit.edu/courses/6-832-underactuated-robotics-spring-2022/) +
  [GitHub 교재](https://github.com/RussTedrake/underactuated).
  HJB/LQR/궤적 최적화를 로보틱스 관점에서 — 이 문서와 가장 결이 같은 강의.
- Diehl, Numerical Optimal Control (Freiburg) —
  [강의 페이지](https://www.syscop.de/teaching/ss2020/numerical-optimal-control-online)
  (영상 + 위 draft 교재).
- Borrelli, UC Berkeley ME231 자료 — explicit MPC/불변 집합 (Borrelli 교재와 세트).

**튜토리얼/문서**
- [do-mpc 문서](https://github.com/do-mpc/do-mpc) — robust MPC를 코드 예제로
  익히기 가장 빠른 경로.
- [acados 문서](https://docs.acados.org/) — RTI를 실물 파라미터
  (`nlp_solver_type = SQP_RTI`)로 만져볼 수 있다.
- [CasADi 예제집](https://web.casadi.org/) — §5.2 condensed QP를 20줄로 재현 가능.

### 9.5 자주 궁금한 점 → 어디를 볼까

| 질문 | 이 repo에서 | 외부에서 |
|------|------------|----------|
| 제약을 hard로 걸고 싶으면? | §5.1 + pi-MPPI(`projection_mppi.py`), CSC-MPPI, CBF-QP 필터 | Borrelli 교재 (§9.1-③), OSQP/qpOASES 문서 |
| terminal cost는 어떻게 정하나? | §4.4 + TD-MPPI(`td_mppi.py`, 학습 가치함수) | Rawlings 등 교재 Ch.2, Mayne 2000 (§9.1-①) |
| 솔버가 제어 주기 안에 안 끝나면? | §6.3-6.4 + `backup_controller.py`, `gatekeeper.py` | RTI 논문 (§9.1-④), acados `SQP_RTI` |
| 안정성 증명을 처음부터 따라가려면? | §3.4 (LQR Lyapunov) → §4.4 (MPC 일반화) | Mayne 2000, Rawlings 등 교재 §2.4-2.5 |
| recursive feasibility가 왜 깨지나? | §4.3 + 연습문제 4; C-MPPI/DualGuard/Gatekeeper | predictive safety filter ([arXiv:1812.05506](https://arxiv.org/abs/1812.05506)) |
| 학습 모델을 예측 모델로 쓰려면? | 02 문서 §7.5 + `models/learned/` 14종 | Hewing 리뷰 (§9.1-⑧), HILO-MPC, L4acados |
| MPC를 미분해서 비용/모델을 학습하려면? | `models/differentiable/` + Step-MPPI(DPC) | Amos 2018 (§9.1-⑦), Theseus, mpc.pytorch |
| RL과 MPC 중 무엇을 쓰나? | Residual-MPPI(사전 정책+잔차), TD-MPPI(가치 학습) | Bertsekas ([arXiv:2406.00592](https://arxiv.org/abs/2406.00592)) |
| LQR 게인은 어떻게 설계/검산하나? | §3.5 + 연습문제 3, `feedback_mppi.py`, `ancillary_controller.py` | Boyd EE363 노트, Anderson & Moore |
| 샘플링(MPPI)으로 넘어갈 준비가 됐는지? | §7 비교표 + 연습문제 5 | — 다음 문서 [02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md)로 |

### 9.6 이 repo에서 이어서 볼 것

- [docs/study/02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md) — 다음 문서
- [docs/MPPI_THEORY.md](../MPPI_THEORY.md) §1-2 — HJB→MPPI 요약 + Vanilla 해설
- `mppi_controller/controllers/mppi/feedback_mppi.py` — Riccati 재귀 실물
- `mppi_controller/controllers/mppi/tube_mppi.py`, `ancillary_controller.py`
  — LQR식 ancillary 피드백 실물
