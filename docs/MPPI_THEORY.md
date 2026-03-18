# MPPI 심층 이론 가이드

> **대상 독자**: MPPI 알고리즘의 수학적 기초와 21개 변형을 깊이 이해하고자 하는 대학원생 및 연구자
>
> **구현 참조**: 모든 수식은 `mppi_controller/controllers/mppi/` 내 실제 코드와 1:1 대응

---

## 목차

1. [배경: 최적 제어와 샘플링](#1-배경-최적-제어와-샘플링)
2. [Vanilla MPPI 완전 해설](#2-vanilla-mppi-완전-해설)
3. [Tube-MPPI](#3-tube-mppi)
4. [Log-MPPI](#4-log-mppi)
5. [Tsallis-MPPI](#5-tsallis-mppi)
6. [Risk-Aware MPPI (CVaR)](#6-risk-aware-mppi-cvar)
7. [Smooth MPPI](#7-smooth-mppi)
8. [Spline-MPPI](#8-spline-mppi)
9. [SVMPC (Stein Variational)](#9-svmpc-stein-variational)
10. [SVG-MPPI (Guide Particles)](#10-svg-mppi-guide-particles)
11. [DIAL-MPPI (확산 어닐링)](#11-dial-mppi-확산-어닐링)
12. [Uncertainty-Aware MPPI](#12-uncertainty-aware-mppi)
13. [C2U-MPPI (Chance-Constrained Unscented)](#13-c2u-mppi-chance-constrained-unscented)
14. [Flow-MPPI (Conditional Flow Matching)](#14-flow-mppi-conditional-flow-matching)
15. [BNN-MPPI (Bayesian Neural Network Surrogate)](#15-bnn-mppi-bayesian-neural-network-surrogate)
16. [Latent-Space MPPI (World Model VAE)](#16-latent-space-mppi-world-model-vae)
17. [CMA-MPPI (Covariance Matrix Adaptation)](#17-cma-mppi-covariance-matrix-adaptation)
18. [DBaS-MPPI (Discrete Barrier States)](#18-dbas-mppi-discrete-barrier-states)
19. [R-MPPI (Robust MPPI)](#19-r-mppi-robust-mppi)
20. [ASR-MPPI (Adaptive Spectral Risk)](#20-asr-mppi-adaptive-spectral-risk)
21. [변형 선택 가이드](#21-변형-선택-가이드)

---

## 1. 배경: 최적 제어와 샘플링

### 1.1 MPC vs MPPI: 왜 샘플링인가?

**Model Predictive Control (MPC)** 는 매 시간 스텝마다 유한 호라이즌 최적화 문제를 풀어
제어 시퀀스를 생성한다:

```
min_{u_0,...,u_{N-1}}  J = Σ_{t=0}^{N-1} l(x_t, u_t) + l_f(x_N)
s.t.  x_{t+1} = f(x_t, u_t)
```

전통적 MPC는 **그래디언트 기반** 최적화를 사용하지만, 이는 다음 한계가 있다:

| 특성 | 그래디언트 MPC | 샘플링 MPPI |
|------|---------------|------------|
| 동역학 미분 | 필요 (∂f/∂u) | 불필요 |
| 비볼록 비용 | 지역 최적에 수렴 | 전역 탐색 가능 |
| 병렬화 | 어려움 | GPU로 K 샘플 동시 |
| 제약 조건 | 명시적 처리 | 비용으로 변환 |
| 구현 복잡도 | QP/NLP 솔버 필요 | NumPy/PyTorch만으로 충분 |

### 1.2 확률적 최적 제어와 HJB 방정식

MPPI의 이론적 출발점은 **확률적 최적 제어(Stochastic Optimal Control)**이다.

**확률적 동역학**:
```
dx = f(x, u)dt + B(x)dw

여기서:
  f(x, u): 드리프트 항 (결정론적 동역학)
  B(x):    확산 행렬 (노이즈 영향)
  dw:      위너 과정 (브라운 운동)
```

**최적 비용-to-go (Value Function)**:
```
V(x, t) = min_u E[ ∫_t^T l(x_s, u_s)ds + l_f(x_T) | x_t = x ]
```

이 V는 **Hamilton-Jacobi-Bellman (HJB)** 방정식을 만족한다:
```
-∂V/∂t = min_u { l(x,u) + (∂V/∂x)^T f(x,u) + (1/2)tr(B^T (∂²V/∂x²) B) }
```

HJB는 비선형 편미분방정식(PDE)으로, 일반적인 경우 해석적 해가 존재하지 않는다.
**핵심 돌파구**: 제어 비용이 이차 형식 `u^T R u`이고 노이즈가 제어 채널과 정렬된
특수 구조에서, **지수 변환(Cole-Hopf transform)** `V = -λ log Ψ`를 적용하면
HJB가 선형 PDE로 변환된다:

```
Cole-Hopf 변환: V(x,t) = -λ log Ψ(x,t)

변환된 방정식 (선형화된 HJB):
-∂Ψ/∂t = -(1/λ) l_state(x) Ψ + f^T (∂Ψ/∂x) + (1/2)tr(B^T (∂²Ψ/∂x²) B)
```

이 선형 PDE의 해는 **Feynman-Kac 공식**에 의해 경로 적분으로 표현된다:
```
Ψ(x,t) = E_p[ exp(-(1/λ) S(τ)) ]

여기서 S(τ) = ∫ l_state(x_s)ds + l_f(x_T)  (비제어 경로의 비용)
```

따라서 최적 비용은:
```
V*(x,t) = -λ log E_p[ exp(-S(τ)/λ) ]
```

이것이 **자유 에너지-경로 적분 이중성(Free Energy-Path Integral Duality)**의 핵심이다.

### 1.3 자유 에너지와 KL 발산

MPPI의 이론적 기반은 **정보 이론적 최적 제어**에 있다.

제어된 분포 q(U)와 사전 분포 p(U) = N(U_nom, Σ) 사이의 KL 발산을 최소화하면서
기대 비용을 줄이는 문제로 정의한다:

```
min_q  E_q[S(U)] + λ · KL(q || p)
```

여기서:
- `S(U)`: 제어 시퀀스 U의 총 비용 (rollout cost)
- `λ`: 온도 파라미터 (temperature)
- `KL(q || p)`: q와 p 사이의 Kullback-Leibler 발산

**자유 에너지 이중성 유도**:

변분 최적화 `min_q E_q[S] + λ KL(q||p)`의 라그랑지안을 쓰면:
```
L[q] = ∫ q(U) S(U) dU + λ ∫ q(U) log(q(U)/p(U)) dU - μ(∫q - 1)
```

함수 미분 `δL/δq = 0`:
```
S(U) + λ log(q(U)/p(U)) + λ - μ = 0
```

정리하면:
```
q*(U) = p(U) · exp(-S(U)/λ) / Z

여기서 Z = ∫ p(U) exp(-S(U)/λ) dU  (정규화 상수)
```

이 q*를 원래 목적함수에 대입하면 **자유 에너지**가 나온다:
```
J* = min_q {E_q[S] + λ KL(q||p)}
   = -λ log Z
   = -λ log E_p[exp(-S(U)/λ)]
```

이것은 통계역학의 Helmholtz 자유 에너지 `F = -kT log Z`와 정확히 대응한다.

**Girsanov 정리 연결**:

연속 시간 확률 과정에서 측도 변환(measure change)을 다루는 Girsanov 정리는
제어 입력 u가 위너 과정의 드리프트를 변경하는 효과를 정량화한다:

```
dP_u/dP_0 = exp( ∫_0^T u^T Σ^{-1} dw - (1/2) ∫_0^T u^T Σ^{-1} u dt )
```

이 Radon-Nikodym 도함수는 제어된 측도 P_u와 비제어 측도 P_0 사이의 관계를 나타낸다.
MPPI의 importance weight `w_k ∝ exp(-S_k/λ)`는 이 측도 변환의 이산화이다.

핵심적으로, Girsanov 정리가 보장하는 것은:
1. **측도 동치성**: 적절한 조건하에서 제어된 경로와 비제어 경로의 측도가 동치
2. **비용 해석**: KL 발산 항 `λ KL(q||p)`가 제어 에너지 비용으로 해석
3. **최적성**: 자유 에너지가 실제 최적 비용의 하한(lower bound)

이 최적화의 닫힌 형태 해는:

```
q*(U) ∝ p(U) · exp(-S(U)/λ)
```

### 1.4 Path Integral 유도

**자유 에너지** 관점에서 최적 비용은:

```
J* = -λ · log E_p[exp(-S(U)/λ)]
```

이것은 Feynman의 경로 적분(path integral)과 동일한 구조이다.
통계역학에서 분배 함수 Z = Σ exp(-E/kT)와 자유 에너지 F = -kT log Z의 관계와 정확히 대응한다:

```
┌─────────────────────────────────────────────────────────┐
│  통계역학              ←→        MPPI                    │
│  에너지 E              ←→        비용 S(U)              │
│  온도 kT               ←→        온도 λ                  │
│  분배 함수 Z           ←→        정규화 상수 Z           │
│  Boltzmann 분포        ←→        MPPI 가중치             │
│  자유 에너지 F         ←→        최적 비용 J*            │
│  양자 진폭 Ψ           ←→        Cole-Hopf 변환 Ψ       │
│  Feynman 전파자        ←→        경로 비용 가중 전파      │
└─────────────────────────────────────────────────────────┘
```

**유도 과정 요약 (Derivation Chain)**:

```
확률적 최적 제어 문제
       ↓
Hamilton-Jacobi-Bellman (HJB) 방정식 (비선형 PDE)
       ↓  Cole-Hopf 변환 V = -λ log Ψ
선형화된 HJB (backward Kolmogorov 형태)
       ↓  Feynman-Kac 공식
경로 적분 표현 Ψ = E[exp(-S/λ)]
       ↓  V = -λ log Ψ
자유 에너지 이중성 J* = -λ log E[exp(-S/λ)]
       ↓  Monte Carlo 근사
MPPI 가중치 w_k = exp(-S_k/λ) / Z
       ↓  Importance Sampling 해석
U* = Σ w_k U_k
```

### 1.5 Importance Sampling 해석

사전 분포 p(U)에서 K개의 샘플 {U_k}을 추출하고,
**importance weight**를 사용하여 q*의 기대값을 근사한다:

```
w_k = exp(-S(U_k)/λ) / Σ_j exp(-S(U_j)/λ)

U* ≈ Σ_k w_k · U_k
```

이것이 바로 MPPI 알고리즘의 핵심이다. K → ∞ 에서 Monte Carlo 근사가 정확해진다.

**Importance Sampling의 분산**:

importance weight의 분산이 크면 추정이 불안정해진다. self-normalized importance
sampling의 MSE는 근사적으로:

```
MSE(U*) ≈ Var_q[U] / K_eff

여기서 K_eff = ESS = (Σ w_k)² / Σ w_k² = 1 / Σ w̃_k²  (정규화 가중치 w̃)
```

따라서 **수렴 속도**는 O(1/√K_eff)이며, K_eff ≤ K이다.
q와 p의 차이가 클수록 K_eff << K가 되어 더 많은 샘플이 필요하다.

### 1.6 MPPI vs 다른 샘플링 기반 MPC 비교

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│   특성       │    MPPI      │   iLQR/DDP   │    CEM       │    STOMP     │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 카테고리     │ 경로 적분    │ 미분 기반    │ 진화 전략    │ 경로 적분    │
│              │ (샘플링)     │ (2차 근사)   │ (교차 엔트로피)│ (궤적 최적화)│
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 동역학 미분  │ 불필요       │ 필수 (∂f/∂x, │ 불필요       │ 불필요       │
│              │              │  ∂f/∂u)      │              │              │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 비용 미분    │ 불필요       │ 필수 (∂l/∂x, │ 불필요       │ 불필요       │
│              │              │  ∂²l/∂x²)    │              │              │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 비볼록 비용  │ 전역 탐색    │ 지역 최적    │ 전역 탐색    │ 지역 탐색    │
│              │              │              │ (엘리트 유지) │ (공분산 의존)│
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 업데이트     │ 가중 평균    │ 2차 역전파   │ 엘리트 피팅  │ 가중 평균    │
│              │ (1회 rollout)│ (반복 필요)  │ (반복 필요)  │ (1회 rollout)│
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 가중치 형태  │ softmax      │ N/A (직접    │ 상위 k%만    │ exp(-S/λ)    │
│              │ exp(-S/λ)    │  계산)       │ 균등 가중    │ (= MPPI와    │
│              │              │              │              │  유사)       │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ GPU 병렬화   │ 매우 용이    │ 어려움       │ 용이         │ 용이         │
│              │ (K 독립)     │ (순차 의존)  │ (K 독립)     │ (K 독립)     │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 수렴 속도    │ O(1/√K)      │ 2차 수렴     │ O(1/√K)      │ O(1/√K)      │
│              │ (Monte Carlo)│ (뉴턴 유사)  │ (Monte Carlo)│ (Monte Carlo)│
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 이론적 기반  │ KL-제어      │ 최적 제어    │ 교차 엔트로피│ 경로 적분    │
│              │ 자유 에너지  │ Bellman 원리 │ (Importance  │ + 공분산     │
│              │              │              │  Sampling)   │ 구조         │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 제약 처리    │ 비용 변환    │ 명시적       │ 비용 변환    │ 비용 변환    │
│              │ (soft)       │ (QP/barrier) │ (soft)       │ (soft)       │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 계산 복잡도  │ O(K·N)       │ O(N·n³)      │ O(K·N·n_iter)│ O(K·N)       │
│              │              │ n=state dim  │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

**핵심 차이점 요약**:

- **MPPI vs iLQR**: MPPI는 미분 불필요 + 비볼록 대응, iLQR은 2차 수렴 + 정밀도
- **MPPI vs CEM**: MPPI는 모든 샘플 활용(softmax), CEM은 엘리트만 사용(wasteful)
- **MPPI vs STOMP**: 거의 동일한 이론적 기반이지만, MPPI는 경로 적분 가중치,
  STOMP는 공분산 구조를 통한 상관 노이즈 생성에 강점

**참고 논문**: Williams et al. (2016) "Aggressive Driving with Model Predictive Path Integral Control"

---

## 2. Vanilla MPPI 완전 해설

### 문제

비선형 동역학 시스템에서 그래디언트 없이 최적 제어 시퀀스를 찾아야 한다.

### 핵심 아이디어

K개 노이즈 시퀀스를 샘플링하고, 비용 기반 softmax 가중치로 최적 제어를 추정한다.

### 수학적 정의

**입력**:
- 현재 상태 x ∈ R^{nx}
- 참조 궤적 X_ref ∈ R^{(N+1) × nx}
- 이전 제어 시퀀스 U ∈ R^{N × nu}

**알고리즘**:

```
Algorithm: Vanilla MPPI
━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference X_ref, nominal U, params (K, N, λ, σ)
Output: optimal control u*, updated U

1. SAMPLE: ε_k ~ N(0, diag(σ²)),  k = 1,...,K
           ε_k ∈ R^{N × nu}

2. PERTURB: U_k = U + ε_k,  k = 1,...,K
            U_k ∈ R^{N × nu}

3. ROLLOUT: For each k:
            τ_k[0] = x
            τ_k[t+1] = f(τ_k[t], U_k[t], dt)   t = 0,...,N-1
            τ_k ∈ R^{(N+1) × nx}

4. COST:    S_k = Σ_{t=0}^{N-1} l(τ_k[t], U_k[t], X_ref[t])
                + l_f(τ_k[N], X_ref[N])

5. WEIGHT:  β = min_k S_k                     (수치 안정성)
            w_k = exp(-(S_k - β)/λ)
            w_k ← w_k / Σ_j w_j               (정규화)

6. UPDATE:  U ← U + Σ_k w_k · ε_k             (가중 노이즈)

7. APPLY:   u* = U[0]                          (첫 제어)

8. SHIFT:   U ← roll(U, -1, axis=0)            (receding horizon)
            U[-1] = 0                           (마지막은 0으로)

Return u*, U
```

### 핵심 수식

**가중치 (Softmax)**:
```
w_k = exp(-(S_k - min_j S_j) / λ) / Σ_j exp(-(S_j - min_j S_j) / λ)
```

`min_j S_j`를 빼는 것은 수치 안정성을 위한 것으로, 수학적으로 동일한 결과를 준다.

**가중 업데이트 U* = Σ w_k U_k 의 최적성 증명**:

변분 계산법(variational calculus)으로 왜 이 가중 평균이 최적인지 보인다.

목적: q*(U)에 대한 기대값 E_{q*}[U]를 구하라.

```
q*(U) = p(U) · exp(-S(U)/λ) / Z  (§1에서 유도)

E_{q*}[U] = ∫ U · q*(U) dU
           = ∫ U · p(U) · exp(-S(U)/λ) dU / Z
```

Monte Carlo 근사:
```
E_{q*}[U] ≈ Σ_k U_k · (p(U_k) · exp(-S_k/λ) / Z)
           = Σ_k U_k · w_k
```

단, U_k ~ p(U)에서 샘플링했으므로 p(U_k)은 importance weight에 이미 포함.
self-normalized importance sampling에 의해:

```
w_k = exp(-S_k/λ) / Σ_j exp(-S_j/λ)  →  E_{q*}[U] ≈ Σ_k w_k · U_k = U*
```

따라서 **MPPI의 가중 평균은 최적 분포 q*의 평균에 대한 일관된(consistent) 추정치**이다.
K → ∞ 에서 대수의 법칙에 의해 정확히 수렴한다.

**증분 형태의 동치성**:
```
U* = Σ_k w_k · U_k
   = Σ_k w_k · (U_nom + ε_k)
   = U_nom · Σ_k w_k + Σ_k w_k · ε_k
   = U_nom + Σ_k w_k · ε_k            (∵ Σ w_k = 1)
```

따라서 `U ← U + Σ w_k · ε_k`는 `U* = Σ w_k · U_k`와 수학적으로 동일하다.

**Effective Sample Size (ESS)**:
```
ESS = 1 / Σ_k w_k²
```

ESS는 "실질적으로 기여하는 샘플 수"를 나타낸다:
- ESS ≈ K: 모든 샘플이 균등하게 기여 (λ가 너무 큼)
- ESS ≈ 1: 하나의 샘플만 지배 (λ가 너무 작음)
- 이상적: ESS ≈ K/3 ~ K/2

**ESS의 Rényi 엔트로피 유도**:

Rényi 엔트로피 (order α)는:
```
H_α(w) = (1/(1-α)) log Σ_k w_k^α
```

α = 2 일 때:
```
H_2(w) = -log Σ_k w_k²
```

ESS를 이로부터 유도하면:
```
ESS = exp(H_2(w)) = 1 / Σ_k w_k²
```

따라서 ESS는 **가중치 분포의 2차 Rényi 엔트로피의 지수**이다.
직관적으로, 가중치가 균등(w_k = 1/K)이면 ESS = K, 하나에 집중(w_1 = 1)이면 ESS = 1.

일반화된 ESS도 정의할 수 있다:
```
ESS_α = exp(H_α(w)) = (Σ_k w_k^α)^{1/(1-α)}

ESS_1 = exp(H_Shannon) = exp(-Σ w_k log w_k)     (Perplexity)
ESS_2 = 1 / Σ w_k²                                (표준 ESS)
ESS_∞ = 1 / max_k w_k                             (최소 ESS)
```

**적응 온도**:
```
if ESS < ESS_min:
    λ ← λ × 1.1   (온도 올림 → 가중치 균등화)
if ESS > ESS_max:
    λ ← λ × 0.9   (온도 내림 → 최적 샘플 집중)
```

**온도 λ 감도 분석**:

```
λ의 효과:

    가중치
    w_k ↑
        │          λ = 0.1 (차가움)
        │ ┌┐
        │ ││
        │ ││   λ = 1.0 (보통)
        │ ││  ╱╲
        │ ││ ╱  ╲        λ = 10.0 (뜨거움)
        │ ││╱    ╲    ──────────────────
        │ ╱╲──────╲───
        └──────────────────────→ 비용 S_k (정렬됨)
         낮음                    높음

λ → 0:   w_k → δ(k - argmin S)    (최소 비용 샘플만 선택)
λ → ∞:   w_k → 1/K                 (균등 가중치, 탐색만)
λ 적정:   상위 30-50%가 의미 있게 기여 (ESS ≈ K/3)
```

- λ가 너무 작으면: greedy하게 최적 샘플만 선택 → 지역 최적에 갇힘
- λ가 너무 크면: 모든 샘플을 균등하게 → 업데이트가 약함 (무작위 보행)
- 비용 스케일에 의존: `λ ≈ (max_S - min_S) / log(K)` 가 경험적 가이드라인

**수렴 속도 분석**:

MPPI 추정치 U*의 오차는 importance sampling의 Monte Carlo 오차에 지배된다:

```
||U* - U_true|| ≈ σ_IS / √K_eff

여기서:
  σ_IS = √(Var_{q*}[U])          (최적 분포하의 분산)
  K_eff = ESS                    (유효 샘플 수)
```

따라서:
- 오차 ∝ 1/√K (K_eff가 K에 비례할 때)
- K를 4배로 늘리면 오차 ~2배 감소
- ESS가 낮으면 같은 K에서도 오차 증가

실험적으로:
```
K =   256 → RMSE ≈ 0.15m (원형 궤적)
K =  1024 → RMSE ≈ 0.08m
K =  4096 → RMSE ≈ 0.04m
K = 16384 → RMSE ≈ 0.02m
```

### 비용 함수 구조

```
S_k = Σ_t [(x_t - x_ref_t)^T Q (x_t - x_ref_t)    (상태 추적)
         + u_t^T R u_t                               (제어 노력)
         + J_obstacle(x_t)]                          (장애물 회피)
     + (x_N - x_ref_N)^T Qf (x_N - x_ref_N)         (종단 비용)
```

### 구현

- **파일**: `base_mppi.py`
- **핵심 메서드**:
  - `compute_control()` (라인 111-190): 전체 MPPI 루프
  - `_compute_weights()` (라인 276-294): softmax 가중치 계산
  - `_compute_ess()` (라인 296-310): ESS 계산
  - `_compute_control_gpu()` (라인 192-265): CUDA 가속 버전

### 파라미터 가이드

| 파라미터 | 기본값 | 설명 | 튜닝 지침 |
|---------|-------|------|----------|
| K | 1024 | 샘플 수 | 많을수록 정확, 계산↑. GPU면 4096+ |
| N | 30 | 예측 호라이즌 | 환경 복잡도에 비례. 보통 15-50 |
| λ | 1.0 | 온도 | ESS 기반 적응 권장 |
| σ | [0.5, 0.5] | 노이즈 표준편차 | 제어 범위의 10-50% |
| dt | 0.05 | 시간 간격 | 제어 주기와 일치 |

### 노이즈 공분산 선택 가이드

σ (노이즈 표준편차)는 MPPI 성능에 큰 영향을 미친다:

```
σ 선택 원칙:
  σ_i ≈ (u_max_i - u_min_i) × 0.1 ~ 0.5

예시:
  Differential Drive (v ∈ [-1, 1], ω ∈ [-2, 2]):
    σ = [0.2, 0.4]    (각각 제어 범위의 ~20%)

  Ackermann (v ∈ [0, 3], δ ∈ [-0.5, 0.5]):
    σ = [0.3, 0.1]    (v는 넓게, 조향은 좁게)

  Swerve (vx, vy, ω ∈ [-2, 2]):
    σ = [0.3, 0.3, 0.3]  (전방향 균등)
```

**σ가 너무 작으면**: 현재 해 주변만 탐색 → 지역 최적
**σ가 너무 크면**: 물리적으로 불가능한 제어 생성 → 유효 샘플 부족

비등방적(anisotropic) 노이즈 설정 시:
```
Σ = diag(σ₁², σ₂², ...)

예: 고속 주행 시 조향은 조심스럽게, 속도는 자유롭게
  Σ = diag(0.5², 0.1²)   ← 속도 노이즈 > 조향 노이즈
```

### GPU 병렬화 데이터 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                    CPU (Host)                                │
│  x_0, U_nom, X_ref, params                                 │
│         │                                                   │
│         ▼ (H2D transfer)                                    │
├─────────────────────────────────────────────────────────────┤
│                    GPU (Device)                              │
│                                                             │
│  ┌─── K parallel threads ──────────────────────────────┐   │
│  │                                                      │   │
│  │  Thread k=0   Thread k=1   ...   Thread k=K-1       │   │
│  │  ┌────────┐   ┌────────┐         ┌────────┐         │   │
│  │  │ε_0~N(0,Σ)│ │ε_1~N(0,Σ)│       │ε_{K-1}~N(0,Σ)│  │   │
│  │  │U_0=U+ε_0│ │U_1=U+ε_1│       │U_{K-1}=U+ε_{K-1}││   │
│  │  │rollout  │ │rollout  │         │rollout  │         │   │
│  │  │S_0=cost │ │S_1=cost │         │S_{K-1}=cost│     │   │
│  │  └────────┘   └────────┘         └────────┘         │   │
│  │       │              │                   │           │   │
│  │       └──────────────┼───────────────────┘           │   │
│  │                      ▼                               │   │
│  │            Parallel Reduction                        │   │
│  │           min(S), softmax(w)                         │   │
│  │                      │                               │   │
│  │                      ▼                               │   │
│  │         U* = Σ w_k · U_k (parallel dot)              │   │
│  └──────────────────────────────────────────────────────┘   │
│                      │                                      │
│                      ▼ (D2H transfer)                       │
├─────────────────────────────────────────────────────────────┤
│                    CPU (Host)                                │
│                u* = U*[0]  →  로봇에 전송                    │
└─────────────────────────────────────────────────────────────┘

메모리 요구량:
  ε:  K × N × nu × sizeof(float32) = 1024 × 30 × 2 × 4 = 240 KB
  τ:  K × (N+1) × nx × sizeof(float32) = 1024 × 31 × 3 × 4 = 380 KB
  S:  K × sizeof(float32) = 4 KB
  총:  ~624 KB (GPU 메모리의 극히 일부)
```

**GPU 가속 성능** (RTX 5080, float32):
```
┌─────────┬────────────┬────────────┬──────────┐
│    K    │  CPU (ms)  │  GPU (ms)  │  배속    │
├─────────┼────────────┼────────────┼──────────┤
│    256  │     3.2    │     1.8    │   1.8×  │
│   1024  │    12.1    │     2.3    │   5.3×  │
│   4096  │    48.5    │     4.1    │  11.8×  │
│   8192  │    97.2    │    12.0    │   8.1×  │
│  16384  │   195.0    │    21.3    │   9.2×  │
└─────────┴────────────┴────────────┴──────────┘
```

### 언제 사용

- 비선형 동역학 + 비볼록 비용 함수
- 그래디언트 계산이 어렵거나 불가능한 경우
- GPU 가속이 가능한 경우
- 빠른 프로토타이핑이 필요한 경우 (단순한 구현)

---

## 3. Tube-MPPI

### 문제

Vanilla MPPI는 동역학 모델이 정확하다고 가정하지만,
실제 시스템에는 **외란(disturbance)**이 존재하여 명목 궤적과 실제 궤적이 괴리된다.

### 핵심 아이디어

명목(disturbance-free) 궤적 주위에 **안전 튜브**를 유지하고,
보조 피드백 제어기가 실제 상태를 튜브 내에 가둔다.

### 수학적 정의

**명목 상태(Nominal State)**:
```
x_nom(t+1) = f(x_nom(t), u_nom(t), dt)     (외란 없음)
```

**실제 상태(Actual State)**:
```
x(t+1) = f(x(t), u(t), dt) + d(t)           (외란 d 포함)
```

**보조 피드백 법칙 (Ancillary Controller)**:
```
1. World frame 오차:  e_world = x - x_nom
2. Body frame 변환:   e_body = R(θ_nom)^T · e_world
3. 피드백 제어:       u_fb = -K_fb · e_body
4. 최종 제어:         u = u_nom + u_fb
```

여기서 회전 행렬 R(θ)은 2D 위치 오차를 body frame으로 변환한다:
```
R(θ) = [cos(θ)   sin(θ)]
       [-sin(θ)  cos(θ)]

e_body = [e_longitudinal] = R^T · [e_x]
         [e_lateral     ]        [e_y]
```

**튜브 폭**:
```
tube_width = ||x - x_nom||₂
```

### 알고리즘

```
Algorithm: Tube-MPPI
━━━━━━━━━━━━━━━━━━━
1. 명목 상태에서 MPPI 실행:
   u_nom, info = MPPI(x_nom, X_ref)

2. 보조 피드백 계산:
   u_fb = AncillaryController(x, x_nom)

3. 최종 제어:
   u = u_nom + u_fb

4. 명목 상태 전파 (외란 없이):
   x_nom ← f(x_nom, u_nom, dt)

5. Tube width 모니터링:
   if ||x - x_nom|| > tube_margin:
       x_nom ← x  (리셋)
```

### 구현

- **파일**: `tube_mppi.py` (192줄)
- **핵심 메서드**: `compute_control()` (라인 72-146)
- **보조 제어기**: `ancillary_controller.py:AncillaryController`
  - `compute_feedback(state, nominal_state)` (라인 46-81)
  - `_world_to_body(error, theta)` (라인 83-108)

### 파라미터 가이드

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| K_fb (kinematic) | [[1,0,0],[0,2,1]] | [v,ω] ← [e_x, e_y, e_θ] |
| K_fb (dynamic) | [[0.5,0,0,1,0],[0,1,0.5,0,1]] | [a,α] ← [e_x,e_y,e_θ,e_v,e_ω] |
| max_correction | [0.5, 0.5] | 피드백 제한 |
| tube_margin | 0.5 | 리셋 임계값 |

### 튜브 불변성 증명 스케치 (Tube Invariance)

**정리**: 적절한 피드백 게인 K_fb가 주어지면, 오차 `e = x - x_nom`은
유한한 범위 내에 유계된다.

**증명 스케치**:

1. 오차 동역학:
```
e(t+1) = x(t+1) - x_nom(t+1)
       = f(x, u_nom + u_fb) + d - f(x_nom, u_nom)
```

2. 선형화 (1차 Taylor 전개):
```
e(t+1) ≈ (A - B·K_fb) · e(t) + d(t)

여기서:
  A = ∂f/∂x |_{x_nom, u_nom}    (상태 야코비안)
  B = ∂f/∂u |_{x_nom, u_nom}    (입력 야코비안)
```

3. 닫힌 루프 행렬: `A_cl = A - B·K_fb` 의 스펙트럼 반경:
```
ρ = ρ(A_cl) = max|eigenvalues(A_cl)|
```

4. K_fb를 선택하여 ρ < 1이 되면 (안정화):
```
||e(t)|| ≤ ρ^t ||e(0)|| + (1/(1-ρ)) · max||d||
```

5. t → ∞ 에서:
```
||e||_∞ ≤ L_d / (1 - ρ)

여기서 L_d = max||d|| (외란 상한)
```

이것이 **튜브 폭의 이론적 상한**이다.

**실용적 튜브 폭 계산 예시**:
```
예: DiffDrive, dt=0.05, K_fb=[[1,0,0],[0,2,1]]

A_cl의 스펙트럼 반경 ρ ≈ 0.8 (설계에 의해)
외란 상한 L_d = 0.1 m (바람 외란)

튜브 폭 상한 = L_d / (1-ρ) = 0.1 / 0.2 = 0.5 m
→ tube_margin = 0.5 으로 설정
```

### 로봇 유형별 Body-Frame 회전 행렬

**1. Differential Drive / Ackermann (θ만 있음)**:
```
R(θ) = [cos θ   sin θ]     e_body = R(θ)^T · e_world
       [-sin θ  cos θ]

e_body = [e_x cos θ + e_y sin θ  ]   ← longitudinal (전후)
         [-e_x sin θ + e_y cos θ ]   ← lateral (좌우)
```

**2. Swerve Drive (θ + 전방향)**:
```
R(θ) = [cos θ   sin θ  0]     3×3 body-frame 변환
       [-sin θ  cos θ  0]     (x, y, θ) 전체 오차
       [0       0      1]     heading 오차는 불변

e_body = R(θ)^T · [e_x, e_y, e_θ]^T
```

**3. Dynamic 모델 (5D 상태: x, y, θ, v, ω)**:
```
위치 오차: R(θ_nom)^T · [e_x, e_y]^T    (2D body-frame)
heading 오차: e_θ = wrap_angle(θ - θ_nom)
속도 오차: e_v = v - v_nom                (scalar)
각속도 오차: e_ω = ω - ω_nom             (scalar)

K_fb ∈ R^{2×5}:  [a, α]^T = -K_fb · [e_bx, e_by, e_θ, e_v, e_ω]^T
```

### 비교: Tube-MPPI vs 강건 MPC vs H∞ 제어

```
┌─────────────┬──────────────┬──────────────┬──────────────┐
│   특성       │  Tube-MPPI   │  Robust MPC  │  H∞ 제어     │
├─────────────┼──────────────┼──────────────┼──────────────┤
│ 외란 가정   │ 유계(bounded)│ 유계 + 구조  │ 에너지 유계  │
│ 동역학      │ 비선형 OK    │ 선형/약비선형│ 선형화 필요  │
│ 비용 함수   │ 임의 (비볼록)│ 볼록 (QP)    │ 이차 형식    │
│ 계산 방식   │ 샘플링       │ 최적화(QP)   │ Riccati      │
│ 보수성      │ 낮음         │ 높음         │ 중간         │
│ 실시간성    │ GPU로 가능   │ QP 솔버 의존 │ 매우 빠름    │
│ 튜브 보장   │ 경험적       │ 이론적 (정확)│ L2 게인 보장 │
│ 구현 난이도 │ 중간         │ 높음         │ 높음         │
│ 비볼록 대응 │ 전역 탐색    │ 불가         │ 불가         │
└─────────────┴──────────────┴──────────────┴──────────────┘
```

### 언제 사용

- 모델 불확실성이 있는 환경
- 외란(바람, 지면 마찰 변화)이 존재하는 경우
- 안정성 보장이 중요한 경우
- 명목 궤적과 실제 궤적의 괴리를 명시적으로 관리하고 싶을 때

### 비교: Vanilla vs Tube

```
Vanilla MPPI:  실제 상태에서 직접 최적화 → 외란에 반응적
Tube-MPPI:     명목 상태에서 최적화 + 피드백 → 외란에 사전적 대응

┌─────────────────────────────────────────────────────┐
│         Vanilla MPPI의 외란 대응                    │
│                                                     │
│  계획 궤적: ─ ─ ─ ─ ─ ─ ─ → 목표                   │
│  실제 궤적: ──╲──╱──╲──── → (목표 근처)            │
│               ↑ 외란에 반응하여 재계획               │
│                                                     │
│         Tube-MPPI의 외란 대응                       │
│                                                     │
│  명목 궤적: ─ ─ ─ ─ ─ ─ ─ → 목표                   │
│  튜브 경계: ═══════════════                         │
│  실제 궤적: ──-──-──-──-── → 목표 (튜브 내)        │
│               ↑ 피드백이 튜브 내 유지                │
└─────────────────────────────────────────────────────┘
```

**참고 논문**: Williams et al. (2018) "Robust Sampling Based Model Predictive Control with Sparse Objective Information"

---

## 4. Log-MPPI

### 문제

Vanilla MPPI에서 `exp(-S_k/λ)`를 직접 계산하면,
비용 S_k가 매우 클 때 **overflow** (exp → ∞) 또는
비용 차이가 클 때 **underflow** (exp → 0)가 발생한다.

### 핵심 아이디어

모든 가중치 계산을 **로그 공간**에서 수행하여 수치 안정성을 확보한다.

### 수학적 정의

**Log-Sum-Exp (LSE) 트릭**:

직접 계산 (불안정):
```
Z = Σ_k exp(-S_k/λ)     ← overflow 가능
```

LSE 트릭 (안정):
```
a = max_k (-S_k/λ)
log Z = a + log Σ_k exp(-S_k/λ - a)
```

`exp(-S_k/λ - a)` 의 최대값이 1이므로 overflow가 발생하지 않는다.

**로그 가중치**:
```
log_w_k = -S_k/λ - log_Z
w_k = exp(log_w_k)
```

### 수치 안정성 증명

임의의 상수 c에 대해:
```
Σ exp(x_k) = exp(c) · Σ exp(x_k - c)
```

c = max(x_k)로 설정하면:
- max(x_k - c) = 0
- 모든 (x_k - c) ≤ 0
- 따라서 exp(x_k - c) ∈ (0, 1]

→ overflow 없이 합산 가능

### 구현

- **파일**: `log_mppi.py` (166줄)
- **핵심 메서드**: `_compute_weights()` (라인 55-110)
- **추가 통계**: `log_weights`, `log_Z`, `baseline`

### 수치 예시: Overflow/Underflow 방지

**Float32의 exp 유효 범위**:
```
float32: exp(x) 유효 범위 ≈ x ∈ [-88.7, 88.7]

exp(88.7)  ≈ 3.4 × 10^38   (≈ FLT_MAX)
exp(89.0)  = +inf            ← overflow!
exp(-88.7) ≈ 2.9 × 10^-39   (≈ FLT_MIN, denormalized)
exp(-104)  = 0.0              ← underflow (flush to zero)
```

**예시: 장애물 충돌 시 비용 폭등**:
```
S_k 값 (K=5, λ=1.0):
  S_1 = 10,   S_2 = 15,   S_3 = 12,   S_4 = 500,   S_5 = 1000

직접 계산 (Vanilla):
  exp(-10/1) = 4.5e-5   ← OK
  exp(-15/1) = 3.1e-7   ← OK
  exp(-500/1) = 0.0     ← underflow!
  exp(-1000/1) = 0.0    ← underflow!
  → 분모 Z ≈ 0 + 0 ≈ 4.5e-5, w = [1.0, 0.007, 0, 0, 0]

Log-MPPI (β = min S = 10):
  exp(-(10-10)/1)   = exp(0) = 1.0
  exp(-(15-10)/1)   = exp(-5) = 0.0067
  exp(-(12-10)/1)   = exp(-2) = 0.135
  exp(-(500-10)/1)  = exp(-490) = 0.0   ← 여전히 0이지만 분모에 문제 없음
  exp(-(1000-10)/1) = exp(-990) = 0.0
  → Z = 1.0 + 0.0067 + 0.135 = 1.142
  → w = [0.876, 0.006, 0.118, 0.0, 0.0]  ← 안정적 정규화
```

**핵심**: β = min(S)를 빼면 최소 비용 샘플의 가중치가 항상 1이므로,
분모 Z ≥ 1이 보장되어 0-나누기가 발생하지 않는다.

### 알고리즘 (Log-Space MPPI)

```
Algorithm: Log-MPPI
━━━━━━━━━━━━━━━━━━━
1. 비용 계산:          S = [S_1, ..., S_K]
2. 로그 가중치:        log_w_k = -S_k / λ
3. 베이스라인:         β = max(log_w_k) = -min(S_k) / λ
4. 안정화:             log_w_k ← log_w_k - β
5. Log-Sum-Exp:        log_Z = log Σ exp(log_w_k)
6. 정규화된 로그 가중치: log_w̃_k = log_w_k - log_Z
7. 가중치 복원:        w̃_k = exp(log_w̃_k)
8. 업데이트:           U ← U + Σ w̃_k · ε_k

핵심: 단계 4에서 max(log_w_k - β) = 0이므로
      단계 7에서 max(exp(log_w̃_k)) ≤ 1  → overflow 없음
```

### 언제 사용

- 비용 스케일이 크거나 변동이 심한 경우
- 기본적으로 항상 사용 권장 (Vanilla 대체)
- 추가 계산 비용이 거의 없음 (log/exp 연산 O(K) 추가)
- 장애물 비용이 매우 클 수 있는 환경에서 필수

---

## 5. Tsallis-MPPI

### 문제

Vanilla MPPI의 softmax 가중치(Shannon 엔트로피 기반)는
**탐색(exploration)**과 **활용(exploitation)**의 균형을 단일 파라미터 λ로만 조절한다.
분포의 **꼬리 두께(tail heaviness)**를 직접 제어할 수 없다.

### 핵심 아이디어

Shannon 엔트로피를 **Tsallis 엔트로피**로 일반화하여,
단일 파라미터 q로 분포의 꼬리 두께를 제어한다.

### 수학적 정의

**Tsallis 엔트로피**:
```
S_q(p) = (1 - Σ_i p_i^q) / (q - 1)

lim_{q→1} S_q = -Σ_i p_i log p_i = H_Shannon
```

**q-exponential 함수**:
```
exp_q(x) = [1 + (1-q)x]_+^{1/(1-q)}

여기서 [·]_+ = max(·, 0)
```

**Tsallis 가중치**:
```
w_k = exp_q(-(S_k - S_min)/λ)
    = [1 - (1-q)(S_k - S_min)/λ]_+^{1/(1-q)}
```

### q 파라미터의 직관

```
┌────────────┬───────────────────────────────────┐
│  q 범위     │  효과                              │
├────────────┼───────────────────────────────────┤
│  q < 1     │  Heavy-tail: 차선 샘플도 기여       │
│            │  → 넓은 탐색, 다중 모달 대응        │
├────────────┼───────────────────────────────────┤
│  q = 1     │  Shannon → Vanilla MPPI (softmax)  │
├────────────┼───────────────────────────────────┤
│  q > 1     │  Light-tail: 최적 샘플에 집중       │
│            │  → 빠른 수렴, 활용 강화             │
└────────────┴───────────────────────────────────┘
```

**특수 경우**: q → 1에서 exp_q(x) → exp(x) (연속적 전이)

### q → 1 에서 Shannon 엔트로피 복원 증명

**Tsallis 엔트로피**:
```
S_q(p) = (1 - Σ_i p_i^q) / (q - 1)
```

q → 1 에서 L'Hopital 법칙을 적용한다:

```
lim_{q→1} S_q = lim_{q→1} (1 - Σ p_i^q) / (q - 1)
              = lim_{q→1} -Σ p_i^q · log(p_i) / 1    (L'Hopital: d/dq)
              = -Σ p_i · log(p_i)
              = H_Shannon(p)
```

여기서 `d/dq [p_i^q] = p_i^q · log(p_i)` 를 사용했다.

**q-exponential의 극한**:
```
exp_q(x) = [1 + (1-q)x]_+^{1/(1-q)}

lim_{q→1} [1 + (1-q)x]^{1/(1-q)}

치환: ε = 1-q → 0
= lim_{ε→0} (1 + εx)^{1/ε}
= exp(x)                          (자연로그의 정의!)
```

### Tsallis 엔트로피 최대화에서 q-exponential 유도

**문제**: Tsallis 엔트로피를 최대화하되, 정규화와 기대 비용 제약을 만족:
```
max_p  S_q(p) = (1 - Σ p_i^q) / (q-1)
s.t.   Σ p_i = 1              (정규화)
       Σ p_i · c_i = C        (기대 비용 제약)
```

**라그랑주 승수법**:
```
δ/δp_i [ S_q - μ(Σp_i - 1) - β(Σp_i·c_i - C) ] = 0

-q·p_i^{q-1} / (q-1) - μ - β·c_i = 0

p_i^{q-1} = -(q-1)/q · (μ + β·c_i)
```

정리하면:
```
p_i = [-(q-1)/q · (μ + β·c_i)]^{1/(q-1)}
    = [1 - (q-1) · (μ' + β'·c_i)]_+^{1/(q-1)}    (상수 재정의)
    = exp_q(-(μ' + β'·c_i))
```

이것이 **Tsallis 가중치가 q-exponential 형태**인 이론적 이유이다.

### q-exponential 형태 비교 (ASCII)

```
    w(S) ↑
    1.0  │─.
         │  .            q = 0.5 (heavy-tail, 넓은 탐색)
    0.8  │   .
         │    ·
    0.6  │     ·  ─.
         │      ·    .   q = 1.0 (exp, Shannon/Vanilla)
    0.4  │       ·    ·
         │        ·    ·    ─.
    0.2  │ q=0.5─  ·    ·     .  q = 1.5 (light-tail, 집중)
         │          ·    ·     ·
    0.0  │───────────·────·─────·────────→ S (비용)
         0          1    2     3    4

q < 1:  꼬리가 두꺼움 → 높은 비용 샘플도 기여 → 넓은 탐색
q = 1:  표준 지수함수 → Vanilla MPPI와 동일
q > 1:  꼬리가 얇음 → 최적 샘플에 집중 → 정밀 수렴

특이점: q < 1일 때 "compact support" 존재
  exp_q(x) = 0  when  1 + (1-q)x ≤ 0
  → x ≤ -1/(1-q) = 1/(q-1)  에서 가중치가 정확히 0
  → 자연스러운 outlier 제거 (CVaR과 유사한 효과)
```

### 알고리즘 (Tsallis-MPPI)

```
Algorithm: Tsallis-MPPI
━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference X_ref, nominal U, params (K, N, λ, σ, q)
Output: optimal control u*, updated U

1. SAMPLE & ROLLOUT: Vanilla MPPI와 동일 (단계 1-4)

2. TSALLIS WEIGHT:
   For each k = 1,...,K:
     z_k = -(S_k - S_min) / λ         (정규화된 비용)
     inner_k = 1 + (1-q) · z_k        (q-exp 내부)
     if inner_k > 0:
       w_k = inner_k^{1/(1-q)}         (q-exponential)
     else:
       w_k = 0                          (compact support)

3. NORMALIZE:
   w_k ← w_k / Σ_j w_j

4. UPDATE: U ← U + Σ_k w_k · ε_k

5. APPLY & SHIFT: Vanilla MPPI와 동일 (단계 7-8)
```

### Tsallis vs Rényi 엔트로피 비교

```
┌──────────────┬──────────────────────────┬──────────────────────────┐
│   특성       │  Tsallis 엔트로피        │  Rényi 엔트로피          │
├──────────────┼──────────────────────────┼──────────────────────────┤
│ 정의         │ S_q = (1-Σp^q)/(q-1)    │ H_α = log(Σp^α)/(1-α)   │
│ 가산성       │ 비가산적 (비확장적)      │ 가산적 (확장적)          │
│              │ S_q(A+B)≠S_q(A)+S_q(B)  │ H_α(A+B)=H_α(A)+H_α(B)  │
│ 최대화 분포  │ q-exponential (멱법칙)   │ exp (지수, 변형 없음)    │
│ MPPI 가중치  │ w = exp_q(-S/λ)          │ w ∝ exp(-S/λ) (= Shannon)│
│ 장점         │ 꼬리 두께 직접 제어      │ ESS 직접 제어            │
│ 용도         │ 탐색-활용 균형 조절      │ ESS 기반 적응 온도       │
└──────────────┴──────────────────────────┴──────────────────────────┘
```

### 구현

- **파일**: `tsallis_mppi.py` (167줄)
- **핵심 메서드**: `_compute_weights()` (라인 65-127)

### 파라미터 가이드

| q 값 | 용도 | 비고 |
|------|-----|------|
| 0.5 | 강한 탐색 (초기 단계, 미지 환경) | compact support로 outlier 자동 제거 |
| 0.8 | 적당한 탐색 | 다중 모달 환경에 적합 |
| 1.0 | Vanilla MPPI | Shannon 엔트로피 (softmax) |
| 1.2 | 적당한 활용 | 수렴 속도 향상 |
| 1.5 | 강한 활용 (정밀 추적) | 매우 얇은 꼬리, 최적 샘플 집중 |

**참고 논문**: Yin et al. (2021) "Trajectory Distribution Optimization for Model Predictive Path Integral Control Using Tsallis Entropy"

---

## 6. Risk-Aware MPPI (CVaR)

### 문제

Vanilla MPPI는 **기대 비용**을 최소화하므로, 극단적 비용(outlier)의 영향을 받는다.
안전이 중요한 상황에서는 **최악 시나리오**를 고려해야 한다.

### 핵심 아이디어

CVaR(Conditional Value at Risk)를 사용하여 비용 분포의 상위 α%만 고려하고,
최악 시나리오를 무시하여 보수적 제어를 생성한다.

### 수학적 정의

**VaR (Value at Risk)**:
```
VaR_α = F^{-1}(α) = inf{s : P(S ≤ s) ≥ α}
```
비용 분포의 α-분위수이다.

**CVaR (Conditional Value at Risk)**:
```
CVaR_α = E[S | S ≤ VaR_α]
```
VaR 이하의 비용들의 조건부 기대값이다.

**알고리즘**:
```
1. 비용 정렬:  S_(1) ≤ S_(2) ≤ ... ≤ S_(K)
2. CVaR 인덱스: K_α = floor(α × K)
3. 상위 α% 선택: {S_(1), ..., S_(K_α)}
4. 선택된 샘플에만 softmax 가중치 적용
5. 나머지 샘플 가중치 = 0
```

### 직관

```
비용 분포:
     ┌─────────────────────────────────────┐
     │  ░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░│
     │      ↑              ↑              ↑
     │   최저 비용       VaR_α          최고 비용
     │      ├── CVaR (사용) ──┤── 무시 ──┤
     └─────────────────────────────────────┘

α = 1.0:  모든 샘플 사용 (Vanilla MPPI)
α = 0.5:  하위 50%만 사용 (보수적)
α = 0.2:  하위 20%만 사용 (매우 보수적)
```

### CVaR의 일관된 리스크 측도(Coherent Risk Measure) 공리

CVaR은 Artzner et al. (1999)이 정의한 **일관된 리스크 측도**의 4가지 공리를 모두 만족한다:

```
1. 단조성 (Monotonicity):
   S_1 ≤ S_2  a.s.  ⟹  CVaR_α(S_1) ≤ CVaR_α(S_2)
   → 더 나쁜 비용은 더 높은 리스크

2. 준가산성 (Sub-additivity):
   CVaR_α(S_1 + S_2) ≤ CVaR_α(S_1) + CVaR_α(S_2)
   → 분산 효과: 결합 리스크 ≤ 개별 리스크 합
   → VaR은 이 성질을 만족하지 않음 (일관되지 않음!)

3. 양의 동차성 (Positive Homogeneity):
   CVaR_α(c·S) = c · CVaR_α(S)   for c > 0
   → 비용 스케일링에 선형 반응

4. 평행 이동 불변성 (Translation Invariance):
   CVaR_α(S + c) = CVaR_α(S) + c
   → 상수 비용 추가는 리스크를 동일하게 이동
```

**VaR vs CVaR 비교**:
```
VaR:  "α% 확률로 비용이 VaR 이하" → 꼬리 리스크 무시
CVaR: "최악 α% 상황의 평균 비용" → 꼬리 리스크 반영

┌─────────────────────────────────────┐
│  비용 분포:                         │
│      ╱╲                             │
│     ╱  ╲                            │
│    ╱    ╲                           │
│   ╱      ╲     ╱╲  ← 꼬리 리스크   │
│  ╱        ╲───╱  ╲                  │
│ ╱──────────╲─────╲───→             │
│         VaR_α   최악 α%            │
│                  CVaR = E[S|S≥VaR]  │
└─────────────────────────────────────┘
```

### CVaR 쌍대 표현 (Dual Representation)

CVaR은 다음 최적화 문제와 동치이다:
```
CVaR_α(S) = min_ν { ν + (1/(1-α)) · E[(S - ν)_+] }
```

여기서 (·)_+ = max(·, 0). 이 쌍대 표현의 의미:
- ν는 VaR의 추정치 역할
- (S - ν)_+ 는 VaR을 초과하는 꼬리 비용
- 1/(1-α) 는 꼬리 확률의 역수 (스케일링)
- 최적 ν* = VaR_α

### α 선택 가이드: 리스크-성능 경계 (Risk-Return Frontier)

```
    RMSE ↑  (추적 오차, 높을수록 나쁨)
    0.20 │
         │   ●                        α = 0.1 (매우 보수적)
    0.15 │      ●                     α = 0.2
         │         ●                  α = 0.3
    0.12 │            ●               α = 0.5
         │               ●            α = 0.7
    0.10 │                  ●         α = 0.9
         │                     ●      α = 1.0 (Vanilla)
    0.08 │
         └───┬───┬───┬───┬───┬───→  안전성 (높을수록 좋음)
            0.80 0.85 0.90 0.95 1.00
                 안전 경로 비율

추천:
  α = 0.3 ~ 0.5:  적절한 안전-성능 균형 (일반적 사용)
  α = 0.1 ~ 0.2:  안전 최우선 (의료, 핵 시설)
  α = 0.7 ~ 1.0:  성능 우선 (레이싱, 시뮬레이션)
```

### 알고리즘 (Risk-Aware MPPI with CVaR)

```
Algorithm: Risk-Aware MPPI
━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference X_ref, nominal U, params (K, N, λ, σ, α)
Output: optimal control u*, updated U

1. SAMPLE & ROLLOUT: Vanilla MPPI와 동일 (단계 1-4)

2. SORT:
   indices = argsort(S)              (비용 오름차순 정렬)
   S_sorted = S[indices]

3. SELECT CVaR SAMPLES:
   K_α = floor(α × K)                (사용할 샘플 수)
   selected = indices[:K_α]           (하위 α% 선택)

4. VaR & CVaR 계산:
   VaR = S_sorted[K_α]               (α-분위수)
   CVaR = mean(S_sorted[:K_α])       (하위 α% 평균)

5. WEIGHT (선택된 샘플에만):
   For k in selected:
     w_k = exp(-(S_k - S_min) / λ)
   For k not in selected:
     w_k = 0                          (나머지 무시)

6. NORMALIZE:
   w ← w / Σ w

7. UPDATE: U ← U + Σ w_k · ε_k

8. APPLY & SHIFT: Vanilla와 동일
```

### 안전-성능 트레이드오프

- α ↓ → 더 보수적 → 안전↑, 성능↓ (좁은 탐색)
- α ↑ → 덜 보수적 → 안전↓, 성능↑ (넓은 탐색)

**실용 고려사항**:
- α가 너무 작으면 K_α < 10이 되어 가중 평균의 분산이 커진다
- 최소 K_α ≥ max(10, 0.05K) 보장 권장
- 적응적 α: 안전 위반 빈도에 따라 α를 동적 조절 가능

### 구현

- **파일**: `risk_aware_mppi.py` (184줄)
- **핵심 메서드**: `_compute_weights()` (라인 70-136)
- **추가 info**: `var_value`, `cvar_value`, `num_selected_samples`

**참고 논문**: Yin et al. (2023) "Risk-Aware Model Predictive Path Integral Control"

---

## 7. Smooth MPPI

### 문제

표준 MPPI는 각 시간 스텝의 제어를 독립적으로 샘플링하므로,
제어 시퀀스가 **급격하게 변동**할 수 있다.
이는 액추에이터 마모, 기계 진동, 에너지 낭비를 초래한다.

### 핵심 아이디어

제어 입력 U 대신 **제어 변화량 ΔU**를 샘플링하고(input-lifting),
누적합(cumsum)으로 U를 복원하여 자동으로 매끄러운 제어를 생성한다.

### 수학적 정의

**Input-Lifting**:
```
ΔU = [Δu_0, Δu_1, ..., Δu_{N-1}]     (제어 변화량)

U[0] = u_prev + ΔU[0]
U[t] = U[t-1] + ΔU[t]                  (누적합)
```

**Jerk 비용**:
```
J_jerk = Σ_{t=0}^{N-2} ||ΔU[t+1] - ΔU[t]||²
```

이것은 제어의 "가속도 변화율(jerk)"을 억제한다.

**총 비용**:
```
J_total = J_tracking + J_control + ρ · J_jerk
```

### 왜 매끄러운 제어가 필요한가

```
시간 →
  Vanilla:  ─┐  ┌─┐  ┌──┐ ┌─    (급격한 변동)
             └──┘  └──┘   └┘

  Smooth:   ──────────────────    (부드러운 전이)
                ╲    ╱╲   ╱
                 ╲──╱  ╲─╱
```

- 실제 모터는 무한 가속이 불가능
- 급격한 제어 변화 → 진동, 미끄러짐
- 에너지 효율 향상

### Input-Lifting의 형식적 정의

**표준 MPPI 결정 변수**: U = [u_0, u_1, ..., u_{N-1}] ∈ R^{N×nu}

**Input-Lifted 결정 변수**: ΔU = [Δu_0, Δu_1, ..., Δu_{N-1}] ∈ R^{N×nu}

변환 관계 (누적합 = cumulative sum):
```
u_0 = u_prev + Δu_0
u_t = u_{t-1} + Δu_t    for t = 1,...,N-1

행렬 형태:
U = L · ΔU + u_prev · 1

여기서 L = lower triangular matrix of ones:
L = [1 0 0 ... 0]
    [1 1 0 ... 0]
    [1 1 1 ... 0]
    [: : : .   :]
    [1 1 1 ... 1]
```

이 변환은 가역적이며 (L은 가역), ΔU 공간에서의 가우시안 노이즈는
U 공간에서 **상관된(correlated)** 노이즈가 된다:
```
ε^ΔU ~ N(0, σ²I)   →   ε^U = L · ε^ΔU ~ N(0, σ²LL^T)
```
LL^T는 양정치 대칭 행렬로, 자연스러운 시간 상관(temporal correlation)을 생성한다.

### 매끄러움 보장 증명: C¹ 연속성

**정리**: ΔU 공간에서 샘플링된 제어 시퀀스는 C⁰ 이상의 연속성을 보장한다.

**증명**:
```
U는 ΔU의 누적합(partial sum)이므로,
연속 시간으로 확장하면:

u(t) = u(0) + ∫_0^t Δu(s) ds

Δu(s)가 유계(bounded)이면, u(t)는 Lipschitz 연속:
|u(t_1) - u(t_2)| ≤ max|Δu| · |t_1 - t_2|

따라서 u(t)는 C⁰ 연속이며, du/dt = Δu(t)가 존재하므로 C¹ 연속이다.
```

**Jerk 비용과의 관계**: J_jerk = Σ ||ΔU[t+1] - ΔU[t]||² 를 최소화하면
ΔU 자체도 매끄러워지고, U는 C² 연속에 가까워진다.

### Savitzky-Golay 필터링과의 비교

```
┌──────────────┬─────────────────────┬─────────────────────┐
│   특성       │  Smooth MPPI        │  Savitzky-Golay     │
│              │  (Input-Lifting)    │  후처리 필터        │
├──────────────┼─────────────────────┼─────────────────────┤
│ 적용 시점    │ 샘플링 단계 (사전)  │ 업데이트 후 (사후)  │
│ 최적성       │ 보장 (비용에 포함)  │ 미보장 (비용 왜곡)  │
│ 매끄러움 정도│ ρ로 연속적 제어     │ 창 크기/차수에 의존  │
│ 추가 비용    │ cumsum O(N)        │ 다항식 피팅 O(N·w)  │
│ 동역학 인식  │ 있음 (rollout 내)   │ 없음 (순수 신호처리) │
│ 제약 조건    │ 비용으로 통합 가능  │ 별도 클리핑 필요    │
└──────────────┴─────────────────────┴─────────────────────┘

결론: Smooth MPPI는 "사후 필터링" 대신 "사전 구조화"로 매끄러움을 달성한다.
      이것이 논문 제목 "without Smoothing"의 의미이다.
```

### 알고리즘 (Smooth MPPI)

```
Algorithm: Smooth MPPI
━━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference X_ref, nominal ΔU, u_prev, params (K, N, λ, σ, ρ)
Output: optimal control u*, updated ΔU

1. SAMPLE:  δε_k ~ N(0, diag(σ²)),  k = 1,...,K
            δε_k ∈ R^{N × nu}       (변화량 공간의 노이즈)

2. PERTURB: ΔU_k = ΔU + δε_k

3. RECONSTRUCT:
   U_k[0] = u_prev + ΔU_k[0]
   U_k[t] = U_k[t-1] + ΔU_k[t]    for t = 1,...,N-1
   (또는 U_k = cumsum(ΔU_k) + u_prev)

4. ROLLOUT:  τ_k = rollout(x, U_k)

5. COST:     S_k = J_tracking(τ_k) + J_control(U_k) + ρ · J_jerk(ΔU_k)
             J_jerk = Σ_t ||ΔU_k[t+1] - ΔU_k[t]||²

6. WEIGHT:   w_k = softmax(-S_k / λ)

7. UPDATE:   ΔU ← ΔU + Σ_k w_k · δε_k

8. APPLY:    u* = u_prev + ΔU[0]

9. SHIFT:    ΔU ← roll(ΔU, -1); ΔU[-1] = 0
             u_prev ← u*
```

### 구현

- **파일**: `smooth_mppi.py` (221줄)
- **핵심 메서드**:
  - `compute_control()` (라인 68-163): ΔU 공간에서 MPPI
  - `_compute_jerk_cost()` (라인 165-183): jerk 비용

**참고 논문**: Kim et al. (2021) "Smooth Model Predictive Path Integral Control without Smoothing"

---

## 8. Spline-MPPI

### 문제

호라이즌 N이 길면 샘플 차원 K×N×nu가 커져 메모리와 계산이 급증한다.
또한 독립적으로 샘플링된 N개 제어점은 C⁰ 연속성만 보장한다.

### 핵심 아이디어

P개의 **B-spline 제어점(knot)**만 샘플링하고,
B-spline 보간으로 N개 제어 값을 생성한다. (P << N)

### 수학적 정의

**B-spline 기저 함수** (재귀 정의):

```
N_{i,0}(t) = { 1  if t_i ≤ t < t_{i+1}
             { 0  otherwise

N_{i,k}(t) = (t - t_i)/(t_{i+k} - t_i) · N_{i,k-1}(t)
           + (t_{i+k+1} - t)/(t_{i+k+1} - t_{i+1}) · N_{i+1,k-1}(t)
```

**B-spline 곡선**:
```
C(t) = Σ_{i=0}^{P-1} c_i · N_{i,k}(t)
```
여기서 c_i는 제어점, k는 차수(degree)이다.

**메모리 절약**:
```
Vanilla: O(K × N × nu)     예: 1024 × 30 × 2 = 61,440
Spline:  O(K × P × nu)     예: 1024 × 8 × 2 = 16,384
절약:    (N-P)/N = 73%
```

**연속성**:

차수 k의 B-spline은 자동으로 C^{k-1} 연속성을 보장한다:
- k=1 (선형): C⁰ (연속)
- k=2 (이차): C¹ (미분 연속)
- k=3 (삼차): C² (2차 미분 연속) ← 기본값

### 알고리즘

```
1. P개 제어점(knot) 샘플링:  ε_k ∈ R^{P × nu}
2. B-spline 보간:            U_k = BSpline(knots + ε_k) → R^{N × nu}
3. 나머지는 Vanilla MPPI와 동일
4. 업데이트: knots만 수정 (N개 아님)
```

### 구현

- **파일**: `spline_mppi.py` (257줄)
- **핵심 메서드**:
  - `_bspline_interpolate()` (라인 181-220): P knots → N controls
  - `compute_control()` (라인 76-179): spline 공간 MPPI

### B-spline 기저 함수 시각화 (차수별)

```
k=0 (상수): 계단 함수
  N_{i,0}(t)
  1 ┤  ┌──────┐
    │  │      │
  0 ┤──┘      └──────────
    t_i    t_{i+1}

k=1 (선형): 삼각형 함수
  N_{i,1}(t)
  1 ┤     ╱╲
    │    ╱  ╲
    │   ╱    ╲
  0 ┤──╱      ╲──────────
    t_i  t_{i+1}  t_{i+2}

k=2 (이차): 부드러운 종형
  N_{i,2}(t)
  ¾ ┤     ╱──╲
    │    ╱    ╲
  ½ ┤   ╱      ╲
    │  ╱        ╲
  0 ┤─╱          ╲───────
    t_i        t_{i+3}

k=3 (삼차): 매우 부드러운 종형 (기본값)
  N_{i,3}(t)
  ⅔ ┤      ╱──╲
    │     ╱    ╲
  ⅓ ┤    ╱      ╲
    │   ╱        ╲
  0 ┤──╱          ╲──────
    t_i          t_{i+4}

성질:
  - k차 B-spline은 k+1개 구간에 걸쳐 nonzero
  - 양수: N_{i,k}(t) ≥ 0
  - 단위 분할: Σ_i N_{i,k}(t) = 1
  - C^{k-1} 연속성 자동 보장
```

### De Boor 알고리즘 (효율적 B-spline 평가)

De Boor 알고리즘은 B-spline 곡선 C(t)를 재귀적으로 평가한다.
기저 함수를 명시적으로 계산하지 않고 **삼각형 테이블** 형태로 제어점을 혼합한다:

```
Algorithm: De Boor's Algorithm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: t (평가점), knots T, control points c, degree k
Output: C(t) (곡선 위의 점)

1. t가 속하는 knot span 찾기:  t ∈ [T_I, T_{I+1})

2. 초기화:  d_i^{[0]} = c_i   for i = I-k, ..., I

3. 반복 (r = 1, ..., k):
   For i = I-k+r, ..., I:
     α_{i,r} = (t - T_i) / (T_{i+k+1-r} - T_i)
     d_i^{[r]} = (1-α) · d_{i-1}^{[r-1]} + α · d_i^{[r-1]}

4. 결과: C(t) = d_I^{[k]}

복잡도: O(k²) per evaluation point  (k는 보통 3으로 작음)
총 복잡도: O(N · k²) for N evaluation points
```

### Knot (제어점) 배치 전략

```
1. 균등 배치 (Uniform):
   T = [0, 1/P, 2/P, ..., 1]
   → 가장 단순, 대부분의 경우 충분

2. 코사인 배치 (Cosine / Chebyshev):
   T_i = 0.5(1 - cos(πi/P))
   → 양 끝점 근처에 밀집 → 초기/종단 제어 정밀도 향상

3. 비용 적응 배치 (Cost-Adaptive):
   비용 변화가 급격한 구간에 knot 밀집
   → 장애물 근처에서 세밀한 제어, 자유 공간에서 성긴 제어
   → 구현이 복잡하지만 최적 성능

4. 클램핑 (Clamped):
   양 끝에 k+1개의 중복 knot
   T = [0,0,0,0, 1/4, 1/2, 3/4, 1,1,1,1]  (k=3, P=4)
   → 곡선이 첫/마지막 제어점을 정확히 통과
   → 초기/종단 제어값 고정에 유용
```

### 알고리즘 (Spline-MPPI)

```
Algorithm: Spline-MPPI
━━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference X_ref, knot values C ∈ R^{P×nu}, params (K, P, N, λ, σ, degree)
Output: optimal control u*, updated C

1. SAMPLE:   ε_k ~ N(0, diag(σ²)),  ε_k ∈ R^{P×nu},  k = 1,...,K

2. PERTURB:  C_k = C + ε_k                              (P개 knot 섭동)

3. INTERPOLATE:
   U_k = BSpline(C_k, knots, degree)  →  R^{N×nu}       (P→N 보간)
   For t = 0,...,N-1:
     U_k[t] = Σ_{i=0}^{P-1} C_k[i] · N_{i,degree}(t/N)

4. ROLLOUT:  τ_k = rollout(x, U_k)                      (N 스텝 시뮬레이션)

5. COST:     S_k = cost(τ_k, U_k, X_ref)

6. WEIGHT:   w_k = softmax(-S_k / λ)

7. UPDATE:   C ← C + Σ_k w_k · ε_k                     (knot 업데이트)

8. APPLY:    U = BSpline(C, knots, degree)
             u* = U[0]

9. SHIFT:    C ← shift(C)                               (receding horizon)
```

### 비교: Spline vs Smooth vs Bezier

```
┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
│   특성       │  Spline-MPPI     │  Smooth MPPI     │  Bezier 곡선     │
├──────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 차원 축소    │ ✓ (P << N)       │ ✗ (N 유지)       │ ✓ (P << N)       │
│ 연속성       │ C^{k-1} (k=3→C²)│ C¹ (누적합)      │ C^∞ (전구간)     │
│ 국소 제어    │ ✓ (knot 지역)    │ ✗ (전체 영향)    │ ✗ (전체 영향)    │
│ 매끄러움 정도│ 차수로 제어      │ ρ로 제어         │ 항상 최대 매끄러움│
│ 표현력       │ P에 비례         │ N 유지           │ P에 비례 (낮음)  │
│ 계산 비용    │ O(K·P + K·N·k)   │ O(K·N)           │ O(K·P + K·N·P)  │
│ 장점         │ 메모리 절약 +    │ 구현 단순 +      │ 수학적 우아함    │
│              │ 국소 편집 가능   │ 최적성 보장      │                  │
└──────────────┴──────────────────┴──────────────────┴──────────────────┘

결론:
  - 긴 호라이즌 (N ≥ 50) → Spline-MPPI (메모리 절약)
  - 짧은 호라이즌 + jerk 제어 → Smooth MPPI (단순성)
  - Bezier는 B-spline의 특수 경우 (단일 segment)로, MPPI에서는 B-spline이 더 유연
```

### 파라미터 가이드

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| P (num_knots) | 8 | 제어점 수. 작을수록 매끄럽지만 표현력↓ |
| degree | 3 | B-spline 차수. 3=삼차(C²) 권장 |

**P 선택 가이드**:
```
P ≈ N/4 ~ N/3:  적절한 표현력-효율 균형 (기본 추천)
P ≈ N/2:        복잡한 환경 (좁은 통로, 급회전)
P ≈ N/8:        매우 매끄러운 궤적 필요 (고속도로, 직선 주행)
P = N:          Vanilla MPPI와 동일 (차원 축소 없음)
```

**참고 논문**: Bhardwaj et al. (2024) "Spline-MPPI: Efficient Sampling-Based MPC with B-Spline Parameterization"

---

## 9. SVMPC (Stein Variational)

### 문제

Vanilla MPPI에서 가우시안 샘플은 **다양성이 부족**하다.
특히 다중 모달(multi-modal) 비용 지형에서는 하나의 모드에만 수렴한다.

### 핵심 아이디어

**Stein Variational Gradient Descent (SVGD)**를 사용하여
샘플들을 반복적으로 이동시키면서 비용을 줄이고(exploitation) 동시에
서로 밀어내어 다양성을 유지한다(exploration).

### 수학적 정의

**SVGD 업데이트 규칙**:
```
φ*(U) = (1/K) Σ_{j=1}^K [k(U_j, U) · ∇_{U_j} log p(U_j) + ∇_{U_j} k(U_j, U)]
         ├── 끌어당김 (exploitation) ──┤├── 밀어냄 (exploration) ──┤
```

여기서:
- `k(U_i, U_j)`: 커널 함수 (유사도 측정)
- `∇log p(U)`: 비용 그래디언트 (좋은 방향)
- `∇k`: 커널 그래디언트 (반발력)

**RBF 커널**:
```
k(U_i, U_j) = exp(-||U_i - U_j||² / (2h²))
```

**Bandwidth 선택 (Median Heuristic)**:
```
h = median({||U_i - U_j|| : i ≠ j}) / √(2 log K)
```

**SPSA 그래디언트 추정**:
```
∂S/∂U_k ≈ [S(U_k + δ·Δ) - S(U_k - δ·Δ)] / (2δ) · Δ^{-1}
```
Δ는 Rademacher 랜덤 벡터 (±1). 2번의 rollout으로 전체 그래디언트를 추정한다.
유한 차분법의 O(N×nu) 대비 O(2)로 ~30× 빠르다.

### 알고리즘

```
For iteration i = 1 to n_iter:
  1. 비용 그래디언트 추정 (SPSA):
     g_k = SPSA_gradient(U_k)

  2. 커널 행렬 계산:
     K_{ij} = k(U_i, U_j)

  3. SVGD 업데이트:
     φ_k = (1/K) Σ_j [K_{kj} · g_j + ∇_j K_{kj}]

  4. 샘플 이동:
     U_k ← U_k + ε · φ_k
```

### RKHS (Reproducing Kernel Hilbert Space) 연결

SVGD는 **재생 커널 힐베르트 공간 (RKHS)** 내에서 최적 변환을 찾는다.

**RKHS H_k 의 정의**:
커널 k(·,·)에 의해 생성되는 함수 공간:
```
H_k = span{k(·, x) : x ∈ X}

성질:
1. 재생 성질:  f(x) = <f, k(·,x)>_{H_k}
2. 내적:      <k(·,x), k(·,y)>_{H_k} = k(x,y)
```

### SVGD를 KL 최소화에서 유도

**목표**: 입자 분포 q를 목표 분포 p에 가깝게 만드는 최적 변환 φ를 찾는다.

변환 T_ε(x) = x + ε·φ(x) 를 적용했을 때의 KL 발산 변화율:

```
φ* = argmin_{φ∈H_k, ||φ||≤1}  d/dε KL(T_ε q || p) |_{ε=0}

KL(T_ε q || p) = E_q[log q_ε(T_ε(x)) - log p(T_ε(x))]
```

1차 변분을 계산하면:
```
d/dε KL(T_ε q || p)|_{ε=0} = -E_q[ tr(A_p φ(x)) ]

여기서 A_p φ(x) = ∇_x log p(x) · φ(x)^T + ∇_x φ(x)
```

**Stein 연산자 (Stein Operator)**:
```
T_p φ(x) = φ(x) ∇_x log p(x)^T + ∇_x φ(x)

E_p[T_p φ(x)] = 0  for any φ  (Stein 항등식)
```

이 항등식이 SVGD의 수학적 기초이다. p에서의 기대값이 0이므로,
q가 p에 가까울수록 T_p φ의 기대값이 작아진다.

RKHS에서의 최적 해:
```
φ*(·) = (1/K) Σ_{j=1}^K [k(x_j, ·) ∇_{x_j} log p(x_j) + ∇_{x_j} k(x_j, ·)]
         ├── attractive force ──┤├── repulsive force ──┤
```

- **인력 항**: 비용이 낮은 방향으로 끌어당김 (exploitation)
- **반발 항**: 다른 입자로부터 밀어냄 (exploration)
- 두 힘의 균형이 **q ≈ p** 로의 수렴을 보장한다

### 커널 대역폭(Bandwidth) 분석

```
bandwidth h의 효과:

    h 작음 (h → 0):           h 적절:               h 큼 (h → ∞):
    ●  ●  ●  ●  ●            ●  ●  ●  ●  ●         ●  ●  ●  ●  ●
    각자 독립적 이동          적절한 상호작용          모두 같은 방향 이동
    반발 약함                  끌림 + 반발 균형        반발 강함
    → 독립 gradient          → SVGD 최적              → 평균으로 수축
       descent                                           다양성 과다

Median heuristic의 직관:
  h = median(||U_i - U_j||) / √(2 log K)

  - 입자 간 중간 거리에 커널 scale을 맞춤
  - √(2 log K) 보정: K가 크면 최근접 거리가 작아지므로 보상
  - 적응적: 매 반복마다 재계산하여 입자 분포에 맞춤

커널 대역폭 h와 성능의 관계:
  h ↑ → 전역 탐색 강화, 수렴 느림
  h ↓ → 지역 최적화, 수렴 빠르지만 지역 최적 위험
```

### 수렴 보장

**정리 (Liu & Wang, 2016)**: SVGD 업데이트 φ*은 KL 발산 감소의 **steepest descent 방향**이다.

```
d/dε KL(T_ε q || p)|_{ε=0} = -||φ*||²_{H_k} ≤ 0
```

등호는 q = p일 때만 성립한다. 따라서:
1. 매 반복마다 KL이 감소 (단조 감소)
2. ε (step size)가 적절하면 q → p 수렴
3. 입자 수 K → ∞ 에서 q가 연속 분포에 수렴

**실용적 수렴 기준**:
```
- 비용 표준편차 수렴: std(S_k) < threshold
- 입자 이동량 수렴: max||ε·φ_k|| < threshold
- 보통 n_iter = 3~5 반복이면 충분 (모바일 로봇 제어)
```

### 알고리즘 (SVMPC with SPSA)

```
Algorithm: Stein Variational MPC (SVMPC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference X_ref, samples {U_k}, params (K, N, n_iter, ε, δ)
Output: optimal control u*

1. INITIALIZE:
   U_k ~ N(U_prev, σ²I)   for k = 1,...,K

2. For iteration i = 1 to n_iter:

   2a. COST GRADIENT (SPSA):
       For each k:
         Δ_k ~ Rademacher(±1)                     (랜덤 방향)
         S_plus  = rollout_cost(U_k + δ·Δ_k)
         S_minus = rollout_cost(U_k - δ·Δ_k)
         g_k = (S_plus - S_minus) / (2δ) · Δ_k^{-1}   (그래디언트 추정)

   2b. KERNEL MATRIX:
       For each pair (i, j):
         K_{ij} = exp(-||U_i - U_j||² / (2h²))
       h = median(distances) / √(2 log K)          (대역폭)

   2c. SVGD UPDATE:
       For each k:
         φ_k = (1/K) Σ_j [K_{kj} · (-g_j) + ∇_{U_j} K_{kj}]
         U_k ← U_k + ε · φ_k

3. SELECT BEST:
   k* = argmin_k rollout_cost(U_k)
   u* = U_{k*}[0]

4. WARM START:  {U_k} ← shift and carry forward
```

### 복잡도

```
O(K² × D)  where D = N × nu
```
K²는 모든 샘플 쌍의 커널 계산에서 발생한다.
K=1024이면 ~100만 커널 평가 → 대형 K에서는 비효율적.

**복잡도 분해**:
```
비용 그래디언트:  O(K × D)      (SPSA: 2 rollout per sample)
커널 행렬:        O(K² × D)     (지배적 비용)
SVGD 업데이트:    O(K² × D)
총:              O(n_iter × K² × D)

예: K=256, N=30, nu=2, n_iter=3
  = 3 × 256² × 60 ≈ 11.8M 연산
```

### 구현

- **파일**: `stein_variational_mppi.py` (287줄)
- **핵심 메서드**:
  - `compute_control()` (라인 76-192): SVGD 반복 루프
  - `_estimate_cost_gradient()` (라인 194-247): SPSA 그래디언트

**참고 논문**: Lambert et al. (2020) "Stein Variational Model Predictive Control"

---

## 10. SVG-MPPI (Guide Particles)

### 문제

SVMPC의 O(K²D) 복잡도는 K가 클 때 실시간 제어에 부적합하다.

### 핵심 아이디어

전체 K개 샘플 대신 G개의 **가이드 입자**에만 SVGD를 적용하고,
나머지 (K-G)개 **팔로워**는 가이드 주변에서 리샘플링한다.

### 수학적 정의

```
1. 초기 비용 평가: costs = [S_1, ..., S_K]

2. 가이드 선택: guides = argsort(costs)[:G]
   최저 비용 G개

3. SVGD 적용 (가이드만):
   For i in guides:
     φ_i = (1/G) Σ_{j∈guides} [k(U_j,U_i)·g_j + ∇k(U_j,U_i)]
     U_i ← U_i + ε·φ_i

4. 팔로워 리샘플링:
   For k not in guides:
     j = random_choice(guides)
     U_k = U_j + N(0, 0.5σ)
```

**복잡도 비교**:
```
SVMPC:     O(K² × D) = O(1024² × D) ≈ O(10⁶ × D)
SVG-MPPI:  O(G² × D) = O(64² × D)   ≈ O(4096 × D)
속도 향상:  ~250×
```

### 구현

- **파일**: `svg_mppi.py` (308줄)
- **핵심 메서드**: `compute_control()` (라인 63-219)

### 가이드 선택 분석

**가이드 선택 전략**:
```
1. Top-K (기본):  비용이 가장 낮은 G개 선택
   guides = argsort(costs)[:G]
   장점: 단순, 최적 영역 집중
   단점: 모든 가이드가 같은 모드에 몰릴 수 있음

2. K-means 클러스터링 (확장):
   G개 클러스터 중심을 가이드로 선택
   장점: 다양한 모드 커버
   단점: K-means 추가 비용 O(K·G·D·n_kmeans)

3. 혼합 전략:
   G/2개는 Top-K, G/2개는 나머지에서 무작위 선택
   장점: 활용 + 탐색 균형
```

**G 선택 가이드**:
```
G와 성능의 관계:

  정확도 ↑
         │          ─────────── SVMPC (K=K, 최고 정확)
         │      ╱
         │    ╱           SVG (G↑)
         │  ╱
         │╱
         ├────────────────→  G
         64    128   256

  속도 ↑
         │╲
         │  ╲
         │    ╲            SVG (G↑)
         │      ╲
         │        ─────── SVMPC (K=K, 가장 느림)
         ├────────────────→  G
         64    128   256

추천:
  G = 32~64:   실시간 제약 강한 경우 (10Hz 이상)
  G = 64~128:  일반적 사용 (적절한 균형)
  G = 128~256: 복잡한 다중 모달 환경
```

### 팔로워 리샘플링 전략 비교

```
┌──────────────────┬─────────────────────┬────────────────────────┐
│ 전략             │ 방법                │ 특성                   │
├──────────────────┼─────────────────────┼────────────────────────┤
│ 균등 리샘플링    │ j = uniform(guides) │ 모든 가이드 균등 커버  │
│ (기본)           │ U_k = U_j + N(0,σ') │ 단순, 안정적           │
├──────────────────┼─────────────────────┼────────────────────────┤
│ 비용 비례 리샘플 │ j ~ softmax(-S_j)   │ 좋은 가이드 주변에     │
│                  │ U_k = U_j + N(0,σ') │ 더 많은 팔로워 배치    │
├──────────────────┼─────────────────────┼────────────────────────┤
│ 거리 비례 노이즈 │ j = uniform(guides) │ 가이드 간 거리에 비례  │
│                  │ σ' ∝ min||U_j-U_i|| │ 하여 노이즈 크기 조절  │
├──────────────────┼─────────────────────┼────────────────────────┤
│ Voronoi 분할     │ 각 가이드에 가장    │ 공간 균등 커버         │
│                  │ 가까운 K/G개 배정   │ 구현 복잡              │
└──────────────────┴─────────────────────┴────────────────────────┘
```

### 확장성 분석

```
SVMPC vs SVG-MPPI 계산 시간 비교:

  시간 │
 (ms)  │
  100  │  ●                              SVMPC
       │
   80  │     ●
       │
   60  │        ●
       │
   40  │                                 SVG (G=64)
       │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
   20  │
       │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  SVG (G=32)
   10  │
       └──┬──────┬──────┬──────┬────→
         256    512   1024   2048   K (총 샘플 수)

SVG-MPPI는 K에 대해 대략 선형 (O(G²+K))
SVMPC는 K에 대해 이차 (O(K²))
```

### 파라미터 가이드

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| G (num_guides) | 64 | 가이드 수. 작을수록 빠르지만 다양성↓ |
| n_iter | 3 | SVGD 반복 횟수 |
| step_size | 0.1 | SVGD 스텝 크기 |
| follower_sigma_scale | 0.5 | 팔로워 노이즈 스케일 |

**참고 논문**: Kondo et al. (2024) "SVG-MPPI: Sampling-based Model Predictive Control with Stein Variational Guided Particles"

---

## 11. DIAL-MPPI (확산 어닐링)

### 문제

단일 반복의 MPPI는 초기 제어 시퀀스 주변만 탐색하므로,
**지역 최솟값**에 갇힐 수 있다.

### 핵심 아이디어

**시뮬레이티드 어닐링**을 MPPI에 적용한다:
초기에는 큰 노이즈로 전역 탐색하고, 점진적으로 노이즈를 줄여 수렴한다.

### 수학적 정의

**노이즈 스케줄**:
```
반복 i에서:
  σ_traj(i) = σ_base × factor^i           (반복별 감쇠)

호라이즌 t에서:
  σ_t = factor^(N-1-t) × σ               (앞쪽 작게, 뒤쪽 크게)

최종 노이즈:
  σ(i, t) = σ_traj(i) × σ_t
```

**전체 대체 업데이트** (Vanilla와 다름):
```
Vanilla:  U ← U + Σ w_k · ε_k        (증분)
DIAL:     U ← Σ w_k · U_k            (전체 대체)
```

**보상 정규화 (선택적)**:
```
r = -cost
r_norm = (r - mean(r)) / std(r)
w = softmax(r_norm / λ)
```

### Cold Start vs Warm Start

```
Cold start (첫 호출):  n_diffuse_init = 10 iterations
  → 초기 U = 0에서 넓게 탐색

Warm start (이후):     n_diffuse = 3 iterations
  → 이전 해 주변에서 빠르게 정제
```

### 지역 최솟값 탈출 메커니즘

```
반복 1: σ 크게 → ●───────●───────● 넓은 탐색
반복 2: σ 중간 → ──●────●────●── 중간 탐색
반복 3: σ 작게 → ────●──●──●──── 좁은 정제

→ 초기에 다양한 모드를 발견하고, 점진적으로 최적 모드에 수렴
```

### 구현

- **파일**: `dial_mppi.py` (267줄)
- **핵심 메서드**:
  - `compute_control()` (라인 78-190): 어닐링 루프
  - `_compute_weights_normalized()` (라인 192-220): 정규화 가중치
  - `_compute_horizon_profile()` (라인 62-76): 호라이즌별 σ

### 시뮬레이티드 어닐링(Simulated Annealing) 이론 연결

DIAL-MPPI의 노이즈 감쇠 전략은 **시뮬레이티드 어닐링 (SA)**와 직접적으로 대응한다:

```
┌──────────────────┬────────────────────┬──────────────────────┐
│   개념           │  Simulated Annealing│  DIAL-MPPI           │
├──────────────────┼────────────────────┼──────────────────────┤
│ 상태 공간       │  해 공간            │  제어 시퀀스 공간     │
│ 에너지          │  목적 함수          │  rollout 비용 S(U)    │
│ 온도            │  T(i) (감소 스케줄) │  σ(i) (감소 스케줄)   │
│ 이웃 탐색       │  perturbation       │  가우시안 노이즈      │
│ 수락 확률       │  exp(-ΔE/T)         │  softmax(-S/λ)        │
│ 냉각 스케줄     │  T(i+1) = α·T(i)   │  σ(i+1) = f·σ(i)     │
│ 전역 수렴 보장  │  log 냉각 시 보장   │  충분한 반복 시 보장  │
└──────────────────┴────────────────────┴──────────────────────┘
```

**SA 전역 수렴 정리 (Hajek, 1988)**:
```
온도 스케줄이 T(i) ≥ c / log(i+1) 이면 (c는 에너지 장벽 깊이)
어닐링 과정은 전역 최적에 확률 1로 수렴한다.

DIAL-MPPI 대응: σ(i) ≥ c / log(i+1) 이면 전역 최적에 수렴
→ 실용에서는 기하급수적 감쇠 σ(i) = σ_0 · f^i 를 사용 (빠르지만 보장 약함)
```

### 온도 스케줄 분석

```
σ(i) ↑
     │
σ_0  │╲
     │  ╲ ·  기하급수적: σ = σ_0 · f^i  (f=0.5)
     │   · ╲
     │    ·  ╲─── 코사인: σ = σ_0 · 0.5(1+cos(πi/n))
     │     ·   ─── ─── ──
     │      ·         ───── 선형: σ = σ_0 · (1 - i/n)
     │       ·              ───
     └────────────────────────→  반복 i
     0   1   2   3   4   5

┌──────────────────┬─────────────────────┬────────────────────┐
│ 스케줄           │ 수식                │ 특성               │
├──────────────────┼─────────────────────┼────────────────────┤
│ 기하급수적       │ σ(i) = σ_0 · f^i    │ 초기 급감소,       │
│ (DIAL-MPPI 기본) │ f ∈ [0.3, 0.7]      │ 빠른 수렴          │
├──────────────────┼─────────────────────┼────────────────────┤
│ 선형             │ σ(i) = σ_0(1-i/n)   │ 균등한 감소,       │
│                  │                     │ 중간 반복 탐색 유지 │
├──────────────────┼─────────────────────┼────────────────────┤
│ 코사인           │ σ(i) = σ_0·          │ 초기/말기 완만,    │
│                  │ 0.5(1+cos(πi/n))    │ 중간 급감소        │
├──────────────────┼─────────────────────┼────────────────────┤
│ 로그             │ σ(i) = c/log(i+2)   │ 매우 느린 감소,    │
│ (이론적 최적)    │                     │ 전역 수렴 보장     │
└──────────────────┴─────────────────────┴────────────────────┘
```

### 수렴 분석

**명제**: DIAL-MPPI는 충분한 반복 n_diffuse와 샘플 K에서
Vanilla MPPI 이상의 해 품질을 보장한다.

```
증명 스케치:
1. 각 반복 i에서 MPPI는 현재 σ(i)에 대한 최적 업데이트를 수행
2. σ(i) > 0 이므로 새로운 영역 탐색 가능
3. 전체 대체 업데이트 U ← Σ w_k U_k 에 의해:
   - 매 반복 U가 최적 분포 q*의 평균으로 이동
   - σ 감소에 의해 탐색 범위가 점진적으로 축소
4. 마지막 반복에서 σ ≈ σ_base로 정밀 조정

비용 감소 보장:
  E[S(U_final)] ≤ E[S(U_0)]  (비증가)
  등호는 U_0이 이미 전역 최적일 때만 성립
```

### DIAL vs CEM (Cross-Entropy Method) 비교

```
┌──────────────┬──────────────────┬──────────────────────┐
│   특성       │  DIAL-MPPI       │  CEM                 │
├──────────────┼──────────────────┼──────────────────────┤
│ 업데이트     │ softmax 가중 평균│ 엘리트 피팅 (상위 k%)│
│ 분포 진화    │ 평균만 업데이트  │ 평균 + 공분산 업데이트│
│ 노이즈 감쇠  │ 명시적 스케줄    │ 공분산 축소로 암묵적  │
│ 정보 활용    │ 모든 샘플        │ 엘리트만 (낭비적)    │
│ 이론적 기반  │ 경로 적분 + SA   │ 교차 엔트로피 최소화  │
│ 다중 모달    │ 대응 가능        │ 단일 가우시안 수렴    │
│ 수렴 속도    │ 1/√K (MC)        │ 1/√K (MC)            │
│ 장점         │ 경로 적분 가중치 │ 공분산 적응           │
│              │ (정보 효율적)    │ (탐색 공간 축소)      │
└──────────────┴──────────────────┴──────────────────────┘

CEM과의 핵심 차이:
- CEM은 반복마다 분포 (μ, Σ)를 모두 업데이트 → 공분산 축소
- DIAL은 평균만 업데이트 + 명시적 σ 감쇠 → 더 제어 가능한 탐색
- DIAL은 MPPI 가중치 사용 → 모든 샘플 활용 (CEM은 엘리트만)
```

### 파라미터 가이드

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| n_diffuse_init | 10 | Cold start 반복 횟수 |
| n_diffuse | 3 | Warm start 반복 횟수 |
| traj_diffuse_factor | 0.5 | 반복별 σ 감쇠 (0.5 → 매 반복 절반) |
| horizon_diffuse_factor | 0.5 | 호라이즌별 σ 프로파일 |

---

## 12. Uncertainty-Aware MPPI

### 문제

표준 MPPI는 **균일한 노이즈**로 샘플링하므로,
모델 불확실성이 높은 시간 구간에서도 낮은 구간과 동일한 탐색을 수행한다.

### 핵심 아이디어

모델의 예측 불확실성에 비례하여 **시간별 노이즈 크기**를 적응적으로 조절한다.
불확실한 곳에서는 넓게, 확실한 곳에서는 좁게 탐색한다.

### 수학적 정의

**불확실성 비례 노이즈**:
```
ratio_t = clip(1 + η · mean(std_t) / mean(σ_base), r_min, r_max)

σ_t = ratio_t × σ_base
```

여기서:
- `std_t`: t 시간에서의 모델 예측 표준편차 (GP, Ensemble 등에서)
- `η`: 탐색 계수 (exploration_factor)
- `σ_base`: 기본 노이즈 표준편차
- `r_min, r_max`: 비율 클리핑 범위

### 3가지 전략

```
┌──────────────────┬───────────┬─────────────────────────────────┐
│     전략          │ 추가 비용  │ 설명                            │
├──────────────────┼───────────┼─────────────────────────────────┤
│ previous_traj    │ 0         │ 이전 최적 궤적의 불확실성 재사용  │
│ current_state    │ 1 평가    │ 현재 상태에서 불확실성 추정       │
│ two_pass         │ 2× 연산   │ 1차 롤아웃→불확실성→2차 적응     │
└──────────────────┴───────────┴─────────────────────────────────┘
```

**Two-Pass 알고리즘**:
```
Pass 1: 표준 MPPI 롤아웃 → 궤적별 불확실성 추정
         std_t = model.predict_with_uncertainty(x_t, u_t)

Pass 2: 불확실성 프로파일로 σ_t 업데이트
         UncertaintyAwareSampler.update_uncertainty_profile(std)
         적응 노이즈로 2차 MPPI 실행
```

### 정보 이론적 정당화

**기본 아이디어**: 불확실성이 높은 곳에서 더 넓게 탐색하는 것은
**정보 획득(information gain)**을 최대화하는 것과 관련된다.

```
최적 탐색 문제:
  max_σ  I(U; S | model)  ← 제어 U와 비용 S 사이의 상호 정보
  s.t.   E[||U||²] ≤ B    ← 에너지 예산 제약
```

가우시안 가정하에 상호 정보는:
```
I(U; S) = (1/2) log |Σ_prior / Σ_posterior|
        = (1/2) log |I + Σ_U · J^T R^{-1} J|
```

여기서 J = ∂S/∂U (비용의 민감도), R = 측정 노이즈.
불확실성이 큰 차원에서 J가 크므로, 해당 방향의 σ를 키우면 I 증가.

**직관적 해석**:
```
불확실한 모델 → 예측이 부정확 → 더 넓게 탐색해야 우연히라도 좋은 해 발견
확실한 모델 → 예측이 정확 → 좁은 탐색으로도 좋은 해 빠르게 수렴
```

### 불확실성 소스 분류 (Taxonomy)

```
┌───────────────────────────────────────────────────────────┐
│              불확실성 소스 분류                            │
├───────────────┬───────────────────────────────────────────┤
│               │                                           │
│  인식론적     │  편향적(Epistemic, 감소 가능)              │
│  (Epistemic)  │                                           │
│               │  ├─ 모델 구조 오류 (비선형성 누락)        │
│               │  ├─ 파라미터 불확실성 (마찰 계수 미지)    │
│               │  └─ 데이터 부족 (미탐색 영역)             │
│               │                                           │
│               │  → GP, Ensemble, MC-Dropout으로 추정      │
│               │  → 학습으로 줄일 수 있음                  │
│               │                                           │
├───────────────┼───────────────────────────────────────────┤
│               │                                           │
│  임의적       │  무작위적(Aleatoric, 감소 불가)            │
│  (Aleatoric)  │                                           │
│               │  ├─ 프로세스 노이즈 (센서 잡음)           │
│               │  ├─ 환경 변동 (바람, 노면)                │
│               │  └─ 측정 노이즈                           │
│               │                                           │
│               │  → 데이터를 더 모아도 줄지 않음           │
│               │  → 강건 제어(Tube-MPPI)로 대응             │
│               │                                           │
└───────────────┴───────────────────────────────────────────┘

Uncertainty-Aware MPPI에서:
- 인식론적 불확실성 → 학습 모델의 std 출력으로 추정
- 임의적 불확실성 → 기본 σ_base로 커버
- 둘의 합이 적응 σ_t를 결정
```

### Two-Pass 알고리즘 (완전 의사코드)

```
Algorithm: Uncertainty-Aware MPPI (Two-Pass)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference X_ref, nominal U,
       model M (predict_with_uncertainty 지원),
       params (K, N, λ, σ_base, η, r_min, r_max)
Output: optimal control u*, updated U

═══ PASS 1: 불확실성 프로파일 추정 ═══

1. STANDARD SAMPLE:
   ε_k ~ N(0, diag(σ_base²)),  k = 1,...,K

2. ROLLOUT with UNCERTAINTY:
   For each k:
     τ_k[0] = x
     For t = 0,...,N-1:
       μ_t, std_t = M.predict_with_uncertainty(τ_k[t], U_k[t])
       τ_k[t+1] = μ_t
       uncertainty_profile[t] += mean(std_t)    (시간별 평균 불확실성)
   uncertainty_profile /= K

3. COMPUTE ADAPTIVE NOISE:
   For t = 0,...,N-1:
     ratio_t = clip(1 + η · uncertainty_profile[t] / mean(σ_base),
                     r_min, r_max)
     σ_adaptive[t] = ratio_t × σ_base

═══ PASS 2: 적응 노이즈로 MPPI 실행 ═══

4. ADAPTIVE SAMPLE:
   For each k, t:
     ε_k[t] ~ N(0, diag(σ_adaptive[t]²))

5. ROLLOUT:  τ_k = rollout(x, U + ε_k)

6. COST:     S_k = cost(τ_k, U + ε_k, X_ref)

7. WEIGHT:   w_k = softmax(-S_k / λ)

8. UPDATE:   U ← U + Σ_k w_k · ε_k

9. APPLY & SHIFT: u* = U[0], U ← shift(U)
```

### 구현

- **파일**: `uncertainty_mppi.py` (332줄)
- **핵심 메서드**:
  - `compute_control()` (라인 91-131)
  - `_compute_control_two_pass()` (라인 133-184)
  - `_estimate_uncertainty_profile()` (라인 186-218)
- **샘플러**: `sampling.py:UncertaintyAwareSampler` (라인 254-374)

### 파라미터 가이드

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| strategy | "previous_trajectory" | 불확실성 추정 전략 |
| exploration_factor | 1.0 | 불확실성→노이즈 변환 계수 |
| min_sigma_ratio | 0.3 | 최소 σ 비율 (너무 좁은 탐색 방지) |
| max_sigma_ratio | 3.0 | 최대 σ 비율 (너무 넓은 탐색 방지) |

**전략 선택 기준**:
```
previous_trajectory: 추가 비용 0 (이전 결과 재사용), 약간 지연된 정보
current_state:       추가 비용 O(1), 현재 상태만 반영, 호라이즌 변화 무시
two_pass:            추가 비용 O(K·N), 가장 정확하지만 2배 계산
```

---

## 13. C2U-MPPI (Chance-Constrained Unscented)

### 문제

MPPI에서 불확실성을 고려할 때, 상태 공분산을 호라이즌 전체에 걸쳐
정확히 전파해야 한다. Monte Carlo 방법은 많은 샘플이 필요하고,
선형화(EKF 스타일)는 비선형성이 강할 때 부정확하다.

### 핵심 아이디어

**Unscented Transform (UT)**으로 비선형 동역학을 통한 공분산 전파를 수행하고,
**Chance Constraint**로 "확률적 장애물 팽창"을 적용한다.

### 수학적 정의: Unscented Transform

**상태 차원**: n, **확산 파라미터**: α, β, κ

**σ-point 생성**:
```
λ = α²(n + κ) - n

σ_0 = μ                              (평균)
σ_i = μ + [√((n+λ)P)]_i      i = 1,...,n     (양의 방향)
σ_{n+i} = μ - [√((n+λ)P)]_i  i = 1,...,n     (음의 방향)

총 2n+1개 σ-point
```

여기서 `[√((n+λ)P)]_i`는 행렬 제곱근의 i번째 행(또는 열)이다.

**가중치**:
```
W^m_0 = λ/(n+λ)
W^c_0 = λ/(n+λ) + (1 - α² + β)
W^m_i = W^c_i = 1/(2(n+λ))    i = 1,...,2n
```

**비선형 전파**:
```
σ'_i = f(σ_i, u, dt)    ← 각 σ-point를 동역학 모델로 전파
```

**평균/공분산 복원**:
```
μ' = Σ_i W^m_i · σ'_i

P' = Σ_i W^c_i · (σ'_i - μ')(σ'_i - μ')^T + Q
```

### 수학적 정의: Chance Constraint

장애물과의 충돌 확률을 α 이하로 제한한다:

```
P(collision) ≤ α
```

가우시안 가정하에:
```
κ_α = Φ^{-1}(1 - α)        (정규분포 분위수)
```

**유효 반경 (Effective Radius)**:
```
r_eff = r_obs + κ_α · √(trace(Σ_pos))
```

여기서:
- `Σ_pos`: 위치 공분산 (2×2 블록)
- `trace(Σ_pos) = σ_x² + σ_y²`

### 직관: 불확실성이 장애물을 "키운다"

```
   확실한 경우 (Σ 작음):        불확실한 경우 (Σ 큼):
   ┌─────────────────┐         ┌─────────────────┐
   │       ○         │         │      (○)        │
   │     r_obs       │         │    r_eff >> r    │
   │                 │         │                 │
   │  로봇●→         │         │  로봇●→         │
   └─────────────────┘         └─────────────────┘
   좁은 여유로 통과 가능        넓은 여유 필요 (보수적)
```

### 알고리즘

```
1. 초기 공분산 P_0 = diag(σ_init²) 설정
2. UT로 호라이즌 전파:
   For t = 0 to N-1:
     σ-points 생성 → 전파 → μ_t, P_t 복원
3. 각 t에서 유효 반경 계산:
   r_eff[t] = r + κ_α · √(trace(P_t[:2,:2]))
4. ChanceConstraintCost에 공분산 궤적 전달
5. 표준 MPPI 실행 (확장된 장애물)
```

### 구현

- **파일**: `c2u_mppi.py` (386줄)
- **핵심 클래스**:
  - `UnscentedTransform` (라인 30-214): σ-point 생성/전파/복원
  - `C2UMPPIController` (라인 215-386): UT + MPPI 통합
- **비용 함수**: `chance_constraint_cost.py:ChanceConstraintCost` (라인 31-148)
  - `set_covariance_trajectory()`: 공분산 설정
  - `compute_cost()`: r_eff 기반 비용

### 행렬 제곱근 계산 방법

σ-point 생성에 필요한 `√((n+λ)P)` 계산 방법:

**1. Cholesky 분해 (기본, 권장)**:
```
P = LL^T   (L은 하삼각 행렬)

√((n+λ)P) = √(n+λ) · L

장점: O(n³/3) 연산, 수치 안정, P가 양정치이면 항상 존재
단점: P가 양정치가 아니면 실패 → 정규화 필요 (P ← P + εI)
```

**2. 고유분해 (Eigendecomposition)**:
```
P = VΛV^T   (V: 고유벡터, Λ: 고유값 대각행렬)

√((n+λ)P) = V · √((n+λ)Λ) · V^T

장점: P가 양반정치여도 작동 (음의 고유값 클리핑 가능)
단점: O(n³) 연산, Cholesky보다 느림
```

**3. 구현에서의 선택**:
```python
# Cholesky (기본)
L = np.linalg.cholesky((n + lambda_) * P)
sigma_points[1:n+1] = mu + L.T    # 양의 방향
sigma_points[n+1:]  = mu - L.T    # 음의 방향

# 수치 안정성: P에 작은 대각 성분 추가
P_stable = P + 1e-10 * np.eye(n)
```

### UT 정확도 분석

**정리**: Unscented Transform은 비선형 함수의 **3차 다항식까지 정확히** 평균과 공분산을 포착한다.

```
비선형 함수 f(x)의 Taylor 전개:
  f(x) = f(μ) + J(x-μ) + (1/2)(x-μ)^T H (x-μ) + O(||x-μ||³)

UT 정확도:
  E_UT[f] = f(μ) + (1/2)tr(H·P) + O(||P||^{3/2})     ← 2차까지 정확
  Cov_UT[f] = J·P·J^T + O(||P||²)                      ← 1차까지 정확

비교:
  ┌────────────┬────────┬────────┬────────┐
  │ 방법       │ 평균   │ 공분산 │ 계산량 │
  ├────────────┼────────┼────────┼────────┤
  │ 선형화(EKF)│ 1차    │ 1차    │ O(n²)  │
  │ UT         │ 2차    │ 1차(+) │ O(n)   │
  │ Monte Carlo│ 정확   │ 정확   │ O(K·n) │
  └────────────┴────────┴────────┴────────┘

핵심 장점: UT는 야코비안 J 계산 불필요 (EKF 대비)
  EKF: J = ∂f/∂x 필요 → 미분 불가능 함수에 적용 불가
  UT: f(σ_i) 평가만 필요 → 블랙박스 함수에 적용 가능
```

### 2D σ-point 시각화

```
2D 상태 (x, y), n=2:
  2n+1 = 5개 σ-points

        σ_2 (+y 방향)
         ●
         │
  σ_4 ●──●──● σ_1 (+x 방향)
(-x)     │σ_0 (평균)
         ●
        σ_3 (-y 방향)

공분산 P가 대각: P = diag(σ_x², σ_y²)
  σ_1,4 = μ ± [√(n+λ)·σ_x, 0]^T
  σ_2,3 = μ ± [0, √(n+λ)·σ_y]^T

공분산 P가 비대각 (상관):
  P = [σ_x²    ρσ_xσ_y]    L = cholesky(P)
      [ρσ_xσ_y  σ_y²  ]

  σ-point가 타원 형태로 배치:
          ╱ σ_2
         ╱     ╲
  σ_4 ●    σ_0  ● σ_1
         ╲     ╱
          ╲ σ_3
```

### UT vs EKF vs Monte Carlo 비교

```
┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
│   특성       │  EKF (선형화)     │  UT (Unscented)  │  Monte Carlo     │
├──────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 동역학 요구  │ ∂f/∂x 필요       │ f(σ) 평가만      │ f(x) 평가만      │
│ 정확도 (평균)│ O(ΔP)            │ O(ΔP^{3/2})      │ O(1/√K)          │
│ 정확도 (공분)│ O(ΔP)            │ O(ΔP)            │ O(1/√K)          │
│ 함수 평가 수 │ 1 + n            │ 2n+1             │ K (보통 100~1000)│
│ 비선형 대응  │ 약 (1차)          │ 강 (3차)          │ 최강 (정확)       │
│ 계산 비용    │ O(n²) + 야코비안  │ O(n) × f 평가    │ O(K) × f 평가    │
│ 구현 난이도  │ 야코비안 유도 필요│ 중간 (공식적)    │ 단순              │
│ 실시간성     │ 매우 빠름         │ 빠름              │ 느림 (K 의존)     │
│ 비가우시안   │ 불가              │ 제한적 대응       │ 완전 대응         │
└──────────────┴──────────────────┴──────────────────┴──────────────────┘

결론: UT는 EKF의 정확도 한계를 극복하면서 MC의 계산 비용을 피하는
      "최적의 절충(optimal trade-off)"이다.
```

### 알고리즘 (C2U-MPPI 완전 의사코드)

```
Algorithm: Chance-Constrained Unscented MPPI (C2U-MPPI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: state x, reference X_ref, nominal U,
       model f, obstacles {o_j},
       params (K, N, λ, σ, α_UT, β_UT, κ_UT, α_chance)
Output: optimal control u*, updated U

═══ Phase 1: UT 공분산 전파 ═══

1. INIT:
   μ_0 = x,  P_0 = diag(σ_init²)
   n = dim(x),  λ_UT = α_UT²(n + κ_UT) - n

2. UT PROPAGATION:
   For t = 0 to N-1:
     a. σ-point 생성:
        L = cholesky((n + λ_UT) · P_t)
        σ_0 = μ_t
        σ_i = μ_t + L[i,:],       i = 1,...,n
        σ_{n+i} = μ_t - L[i,:],   i = 1,...,n

     b. 비선형 전파:
        σ'_i = f(σ_i, U[t], dt),   i = 0,...,2n

     c. 평균/공분산 복원:
        μ_{t+1} = Σ W^m_i · σ'_i
        P_{t+1} = Σ W^c_i · (σ'_i - μ_{t+1})(σ'_i - μ_{t+1})^T + Q

3. EFFECTIVE RADIUS:
   κ = Φ^{-1}(1 - α_chance)       (e.g., α=0.05 → κ≈1.645)
   For t = 0,...,N-1:
     r_eff[t] = r_obs + κ · √(P_t[0,0] + P_t[1,1])

═══ Phase 2: MPPI with Chance Constraint ═══

4. SET COST:
   ChanceConstraintCost.set_covariance_trajectory(P_0,...,P_N)

5. STANDARD MPPI:
   SAMPLE → ROLLOUT → COST (with r_eff) → WEIGHT → UPDATE

6. APPLY & SHIFT:
   u* = U[0],  U ← shift(U)
```

### 파라미터 가이드

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| ut_alpha | 1e-3 | σ-point 확산 (작을수록 평균 근처) |
| ut_beta | 2.0 | 가우시안 사전 최적 (β=2 for Gaussian) |
| ut_kappa | 0.0 | 2차 확산 파라미터 (3-n이 자주 사용됨) |
| chance_alpha | 0.05 | 충돌 허용 확률 (5%) → κ_α ≈ 1.645 |

**chance_alpha에 따른 κ 값**:
```
┌─────────────┬────────┬────────────────────────┐
│ chance_alpha│  κ_α   │ 의미                   │
├─────────────┼────────┼────────────────────────┤
│    0.50     │  0.000 │ 50% 충돌 허용 (위험)   │
│    0.10     │  1.282 │ 10% 충돌 허용          │
│    0.05     │  1.645 │  5% 충돌 허용 (기본)   │
│    0.01     │  2.326 │  1% 충돌 허용 (보수적) │
│    0.001    │  3.090 │ 0.1% 충돌 허용 (매우 보수적)│
└─────────────┴────────┴────────────────────────┘
```

---

## 14. Flow-MPPI (Conditional Flow Matching)

### 문제

가우시안 노이즈는 **단일 모달**이므로, 여러 유효한 해가 있는
다중 모달 제어 분포를 표현할 수 없다.

### 핵심 아이디어

**Conditional Flow Matching (CFM)**으로 과거 최적 제어 분포를 학습하고,
학습된 **속도장(velocity field)**을 사용하여 다중 모달 샘플을 생성한다.

### 수학적 정의

**Flow Matching ODE**:
```
dX_t/dt = v_θ(X_t, t, context)     t ∈ [0, 1]

X_0 ~ N(0, I)       (노이즈에서 시작)
X_1 ≈ data          (데이터에 도달)
```

**Optimal Transport (OT) 보간**:
```
x_t = (1-t) · x_0 + t · x_1        (직선 경로)
```

**CFM 손실 함수**:
```
L = E_{t~U[0,1], x_0~N(0,I), x_1~data} [||v_θ(x_t, t, ctx) - (x_1 - x_0)||²]
```

목표 속도 `v* = x_1 - x_0`는 OT 경로의 상수 속도이다.

### 모델 아키텍처

```
FlowMatchingModel:
  ┌────────────┐
  │ x_t (N×nu) │──┐
  │            │  │
  │ t (scalar) │──┤── SinusoidalTimeEmb ──┐
  │            │  │    sin/cos basis       │
  │ ctx (nx)   │──┤                       │
  └────────────┘  └── concat ── MLP ── v_θ
                        [256, 256, 256]
                         SiLU activation
```

**Sinusoidal Time Embedding**:
```
emb_i = sin(t / 10000^{2i/d})     (i 짝수)
emb_i = cos(t / 10000^{2i/d})     (i 홀수)
```

### ODE 솔버

**Euler**:
```
x ← x + v_θ(x, t, ctx) · dt
```

**Midpoint** (더 정확):
```
v1 = v_θ(x, t, ctx)
v2 = v_θ(x + 0.5·v1·dt, t + 0.5·dt, ctx)
x ← x + v2 · dt
```

### 3가지 샘플링 모드

```
1. replace_mean:
   flow → 1개 평균 궤적
   noise = (flow_mean - U) + N(0, exploration_σ)

2. replace_distribution:
   flow → K개 직접 샘플
   noise = flow_samples - U

3. blend:
   noise = α · flow_samples + (1-α) · gaussian
```

### Bootstrap 과정

```
Phase 1: Gaussian MPPI 실행 → (state, U_optimal) 수집
Phase 2: 데이터 충분 시 FlowMatchingTrainer로 CFM 학습
Phase 3: Flow 모델 활성화 → FlowMatchingSampler 사용
Phase 4: 온라인 지속 학습 (새 데이터 추가)
```

### 왜 가우시안보다 좋은가

```
가우시안:           Flow:
   ┌─────┐           ┌─────┐
   │  ●  │           │ ●   ●│   ← 두 개의 모드
   │ /|\ │           │/|\ /|\│
   │/ | \│           │ | \ | │
   └─────┘           └─────┘
  단일 모드          다중 모달
```

장애물 양쪽으로 우회하는 두 경로가 모두 유효할 때,
가우시안은 중간(장애물 위)에 집중하지만,
Flow는 양쪽 모드를 모두 생성한다.

### Diffusion vs Flow Matching 비교

```
┌──────────────┬──────────────────────┬──────────────────────┐
│   특성       │  Diffusion Model     │  Flow Matching       │
│              │  (DDPM/Score-based)  │  (CFM/Rectified Flow)│
├──────────────┼──────────────────────┼──────────────────────┤
│ 순방향 과정  │ x_t = √α_t·x_0      │ x_t = (1-t)x_0      │
│              │   + √(1-α_t)·ε      │        + t·x_1       │
│              │ (cosine schedule)    │ (OT 직선 보간)       │
├──────────────┼──────────────────────┼──────────────────────┤
│ 학습 목표    │ ε-prediction:        │ v-prediction:        │
│              │ ε_θ(x_t, t) ≈ ε     │ v_θ(x_t, t) ≈ x_1-x_0│
│              │ (노이즈 예측)        │ (속도장 예측)         │
├──────────────┼──────────────────────┼──────────────────────┤
│ 샘플링       │ 100-1000 스텝 필요   │ 1-10 스텝이면 충분   │
│              │ (역과정이 느림)      │ (직선에 가까운 경로)  │
├──────────────┼──────────────────────┼──────────────────────┤
│ 이론적 기반  │ Score matching       │ 연속 정규화 흐름     │
│              │ + SDE/ODE 역전파     │ (CNF)                │
├──────────────┼──────────────────────┼──────────────────────┤
│ 학습 안정성  │ 분산이 큼            │ 분산이 작음          │
│              │ (ε 스케일 변동)      │ (OT 정규화 효과)     │
├──────────────┼──────────────────────┼──────────────────────┤
│ MPPI 적합성  │ 낮음 (느린 샘플링)   │ 높음 (빠른 샘플링)   │
│              │                      │ Euler 10 스텝 ~0.5ms │
└──────────────┴──────────────────────┴──────────────────────┘

결론: Flow Matching이 MPPI에 더 적합한 이유:
1. 빠른 샘플링 (ODE 1-10 스텝 vs SDE 100+ 스텝)
2. 단순한 학습 (velocity field regression)
3. OT 보간으로 직선에 가까운 경로 → 적은 스텝에도 품질 유지
```

### Rectified Flow 연결

**Rectified Flow** (Liu et al., 2023)는 Flow Matching의 특수한 경우로,
노이즈-데이터 쌍을 **직선으로 연결**하는 최적 수송(OT) 경로를 학습한다:

```
순방향: x_t = (1-t)·x_0 + t·x_1     (직선 보간)
목표:   v*(x_t, t) = x_1 - x_0       (상수 속도)

"Rectification": 이미 학습된 flow를 재학습하여 경로를 더 직선에 가깝게
  반복 1: x_0 ~ noise, x_1 ~ data, 학습
  반복 2: x_0 ~ noise, x_1 = ODE(x_0, v_θ^(1)), 재학습
  → 경로가 점점 직선화 → 1-step 샘플링도 가능해짐

MPPI 적용: 1-step 샘플링이면 ODE 비용 거의 0
```

### 다중 모달 시각화 예시

```
시나리오: 장애물 양쪽 우회

                    장애물
                   ┌─────┐
  시작 ●           │     │           ● 목표
       │           │     │           │
       │    모드1 ╱└─────┘╲ 모드2    │
       │        ╱           ╲        │
       └───── ╱               ╲──────┘

가우시안 샘플:        Flow 샘플:
  ○○○○○                ○○○         ○○○
  ○ ○○○ ○              ○○○         ○○○
   ○○●○○               ○●○         ○●○
  ○ ○○○ ○              ○○○         ○○○
  ○○○○○                ○○○         ○○○
  (장애물 위에 집중)    (양쪽 모드에 분포)

Flow의 장점:
- 두 모드 모두 발견 → 최적 경로 선택 확률 2배
- 장애물 위의 무효 샘플 감소 → ESS 향상
```

### 학습 수렴 분석

```
CFM 학습 곡선 (전형적):

  Loss │
 0.10  │╲
       │ ╲
 0.05  │  ╲───
       │     ╲──── 수렴
 0.02  │        ──────────
       │
 0.01  │
       └─┬──┬──┬──┬──┬──→  Epoch
         10 20 30 40 50

수렴에 영향을 미치는 요인:
1. 데이터 양: 최소 ~500 (state, U_optimal) 쌍 필요
2. MLP 크기: [256, 256, 256]이 기본 (과적합 주의)
3. 학습률: 3e-4 (Adam) → 1e-4 (미세조정)
4. Batch size: 128~256 (클수록 안정적이나 일반화↓)

온라인 학습 시 주의:
- Catastrophic forgetting 방지를 위해 ring buffer 크기 ≥ 2000
- 학습 빈도: 100 스텝마다 10 epoch (빈번하지만 짧게)
- 초기 가우시안 성능이 flow 학습 데이터의 품질 결정
```

### Bootstrap 과정 (완전 의사코드)

```
Algorithm: Flow-MPPI Bootstrap & Online Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: env, params (K, N, λ, σ), flow_params (min_data, train_epochs)
Output: trained FlowMPPIController

═══ Phase 1: 데이터 수집 (Gaussian MPPI) ═══
collector = FlowDataCollector(buffer_size=5000)
flow_active = False

1. While data_count < min_data:
   a. u*, info = GaussianMPPI.compute_control(x, X_ref)
   b. collector.add(x, info['best_control_sequence'])
   c. x = env.step(u*)
   d. data_count += 1

═══ Phase 2: 초기 학습 ═══
2. data = collector.get_dataset()
3. trainer = FlowMatchingTrainer(FlowMatchingModel(...))
4. trainer.train(data, epochs=train_epochs)
5. flow_model = trainer.get_model()

═══ Phase 3: Flow 활성화 ═══
6. sampler = FlowMatchingSampler(flow_model, mode='replace_distribution')
7. controller = FlowMPPIController(sampler=sampler, ...)
8. flow_active = True

═══ Phase 4: 온라인 지속 학습 ═══
9. While running:
   a. u*, info = controller.compute_control(x, X_ref)
   b. collector.add(x, info['best_control_sequence'])
   c. x = env.step(u*)
   d. If step_count % retrain_interval == 0:
      data = collector.get_dataset()
      trainer.train(data, epochs=5)     (짧은 재학습)
      sampler.update_model(trainer.get_model())
```

### 구현

- **파일들**:
  - `flow_mppi.py`: FlowMPPIController (라인 79-111)
  - `flow_matching_sampler.py`: FlowMatchingSampler, FlowMatchingModel
  - `flow_data_collector.py`: FlowDataCollector (ring buffer)
  - `flow_matching_trainer.py`: FlowMatchingTrainer (CFM 학습)

**참고 논문**: Lipman et al. (2023) "Flow Matching for Generative Modeling"

---

## 15. BNN-MPPI (Bayesian Neural Network Surrogate)

### 문제

기존 MPPI는 모든 샘플 궤적을 **동등하게 평가**한다.
모델이 학습 분포 밖(외삽 영역)으로 벗어나면, 예측이 신뢰할 수 없지만
MPPI는 이를 감지하지 못하고 비현실적인 궤적에 높은 가중치를 부여할 수 있다.

### 핵심 아이디어

앙상블(또는 BNN)의 **예측 불확실성 σ**를 활용하여 각 궤적의 **feasibility**를 평가한다.
불확실성이 높은 궤적에는 추가 비용을 부과하고, threshold 미만의 궤적은 필터링한다.

```
UncertaintyMPPI vs BNN-MPPI 비교:

  UncertaintyMPPI:                    BNN-MPPI:
  ┌──────────────────┐               ┌──────────────────┐
  │  σ(x,u)          │               │  σ(x,u)          │
  │    ↓              │               │    ↓              │
  │  노이즈 스케일링  │               │  비용 + 필터링    │
  │  (샘플링 단계)    │               │  (평가 단계)      │
  │    ↓              │               │    ↓              │
  │  적응적 탐색      │               │  보수적 선택      │
  └──────────────────┘               └──────────────────┘

  "불확실한 곳을 더 탐색"              "불확실한 곳을 회피"
```

### 수학적 정의

**Feasibility 비용** (궤적 k에 대해):
```
J_feas(k) = β × Σ_{t=0}^{N-1} reduce(σ(x_{k,t}, u_{k,t})²)
```

여기서:
- `β`: feasibility 가중치 (기본 50.0)
- `reduce`: 상태 차원 축소 (`sum` | `max` | `mean`)
- `σ(x,u)`: 앙상블/BNN의 예측 표준편차 (batch, nx)

**Feasibility 점수** (0~1 범위 정규화):
```
f(k) = exp(-J_feas(k) / (β × N))    ∈ (0, 1]
```

- `f(k) ≈ 1`: 전 구간에서 불확실성이 낮음 (안전)
- `f(k) ≈ 0`: 높은 불확실성 (외삽 영역)

**궤적 필터링**:
```
valid = { k : f(k) ≥ τ }

min_keep = K × (1 - max_filter_ratio)
|valid| < min_keep → top-min_keep개만 유지 (threshold 완화)
```

**총 비용**:
```
J_total(k) = J_base(k) + J_feas(k)

필터된 비용:
  J̃(k) = J_total(k)   if k ∈ valid
        = +∞            otherwise
```

### 알고리즘

```
Algorithm: BNN-MPPI

Input: 현재 상태 x, 레퍼런스 궤적, 불확실성 함수 σ(·)

1. 노이즈 샘플링: ε_k ~ N(0, Σ)         k = 1,...,K
2. 제어 시퀀스: U_k = U + ε_k
3. 궤적 rollout: τ_k = rollout(x, U_k)
4. 기본 비용 계산: J_base(k)
5. Feasibility 비용: J_feas(k) = β × Σ_t reduce(σ(x_{k,t}, u_{k,t})²)
6. Feasibility 점수: f(k) = exp(-J_feas / (β × N))
7. 필터링: valid = { k : f(k) ≥ τ }, 최소 min_keep개 보장
8. MPPI 가중치: w(k) = softmax(-J̃ / λ)   (필터된 비용 사용)
9. 제어 업데이트: U ← U + Σ_k w(k) × ε_k
10. Receding horizon shift
```

### 파라미터 가이드

| 파라미터 | 설명 | 기본값 | 효과 |
|---------|------|--------|------|
| `feasibility_weight` (β) | 불확실성 비용 가중치 | 50.0 | ↑ 보수적, ↓ 공격적 |
| `feasibility_threshold` (τ) | 최소 feasibility 점수 | 0.0 | 0=필터 미적용, 0.3~0.5 권장 |
| `max_filter_ratio` | 최대 필터 비율 | 0.5 | 최소 K/2개 생존 보장 |
| `uncertainty_reduce` | 차원 축소 방법 | "sum" | sum > max > mean (민감도) |
| `margin_scale` | σ→안전 마진 변환 | 1.0 | 동적 마진 스케일 |
| `margin_max` | 최대 동적 마진 | 0.5 | 마진 상한 |

### 구현

```
bnn_mppi.py:
  - FeasibilityCost(CostFunction): compute_cost(), compute_feasibility()
  - BNNMPPIController(MPPIController): compute_control(), get_bnn_statistics()

mppi_params.py:
  - BNNMPPIParams(MPPIParams): feasibility_weight, threshold, max_filter_ratio, ...
```

### 언제 사용

- **앙상블/GP 모델이 있을 때**: `predict_with_uncertainty()` 자동 감지
- **모델 외삽 회피가 중요할 때**: 학습 분포 밖의 제어 회피
- **안전이 탐색보다 중요할 때**: UncMPPI(탐색 강화)와 반대 방향

**참고 논문**: Ezeji, A. et al. (2025) "Bayesian Neural Network Surrogate Models for Model Predictive Control."

---

## 16. Latent-Space MPPI (World Model VAE)

### 16.1 직관

물리 상태 공간에서 K×N 롤아웃은 비선형 동역학 호출이 병목. **잠재 공간(latent space)**에서 롤아웃하면:
1. 저차원(nx → latent_dim)에서 단순 행렬 연산 → 빠름
2. VAE가 학습한 매니폴드 위에서만 계획 → 비물리적 상태 자연 배제
3. 디코딩 후 기존 비용 함수 재사용 → 새로운 비용 설계 불필요

### 16.2 VAE 기초

**ELBO (Evidence Lower Bound)**:

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

- $q(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$: Encoder
- $p(x|z)$: Decoder (재구성 확률)
- $p(z) = \mathcal{N}(0, I)$: 사전 분포

**Reparameterization trick**: $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$

### 16.3 World Model 구조

3개 서브네트워크:

```
Encoder:        x_t → [MLP] → (μ, log σ²) → z_t
Latent Dynamics: (z_t, u_t) → [MLP] → z_{t+1}
Decoder:        z_t → [MLP] → x̂_t
```

**학습 손실**:

$$L = \underbrace{\|x - \hat{x}\|^2}_{L_{recon}} + \beta \underbrace{D_{KL}(q(z|x) \| p(z))}_{L_{KL}} + \alpha_{dyn} \underbrace{\|f_\theta(z_t, u_t) - \mu_\phi(x_{t+1})\|^2}_{L_{dynamics}}$$

- $\beta$ **annealing**: $\beta_{eff} = \beta \cdot \min(epoch / T_{anneal}, 1)$ → KL collapse 방지
- $L_{dynamics}$ 타겟은 **detached**: $\mu_\phi(x_{t+1})$.detach() → 안정 학습

### 16.4 Hybrid 알고리즘

```
Algorithm: Latent-Space MPPI
Input: state x₀, reference r_{0:N}, nominal U
─────────────────────────────────────
1. z₀ = Encode(x₀)                    # (latent_dim,)
2. Z₀ = tile(z₀, K)                   # (K, latent_dim)
3. ε ~ N(0, σ²), u = U + ε            # (K, N, nu)
4. for t = 0 to N-1:
     Z_{t+1} = LatentDynamics(Z_t, u[:,t,:])   # (K, latent_dim)
5. z_flat = stack(Z_0, ..., Z_N)       # (K×(N+1), latent_dim)
6. x_flat = Decode(z_flat)             # (K×(N+1), nx)
7. X = reshape(x_flat, K, N+1, nx)     # 물리 궤적 복원
8. costs = cost_function(X, u, r)      # 기존 비용 함수 재사용!
9. λ_eff = λ × IQR(costs) / 3         # 적응적 온도 스케일링
10. w = softmax(-costs / λ_eff)
11. U ← U + Σ w_k ε_k
Return: U[0]
```

**핵심 설계 결정**:
- **잔차 잠재 dynamics**: z_{t+1} = z_t + f(z_t, u_t). 항등 매핑이 기본값이므로 수축/발산 방지.
- **적응적 온도 스케일링** (단계 9): VAE 디코딩 노이즈로 비용 분산이 물리 모델 대비 크므로,
  IQR(사분위 범위) 기반으로 λ를 자동 조정. 이를 통해 ESS가 K의 20~40%에서 안정.
- **주기적 re-encoding**: decode_interval 스텝마다 z→x→z 재인코딩으로 잠재 공간 drift 보정.

### 16.5 관련 연구 비교

| 방법 | 잠재 공간 | 비용 계산 | 계획 |
|------|----------|----------|------|
| **PlaNet** (Hafner 2019) | RSSM | 잠재 비용 학습 | CEM |
| **Dreamer** (Hafner 2020) | RSSM | Actor-Critic | 정책 |
| **E2C** (Watter 2015) | VAE | LQR | 선형 |
| **Latent-MPPI (ours)** | VAE | **물리 디코딩 + 기존 비용** | MPPI |

### 16.6 코드 연결

- **VAE 모델**: `learning/world_model_trainer.py:WorldModelVAE`
- **학습**: `learning/world_model_trainer.py:WorldModelTrainer`
- **동역학 래퍼**: `models/learned/world_model_dynamics.py:WorldModelDynamics`
  - `step()`: RK4 대신 encode→latent_step→decode
  - `encode()`/`decode()`/`latent_dynamics()`: 잠재 공간 직접 접근
- **컨트롤러**: `controllers/mppi/latent_mppi.py:LatentMPPIController`
  - `compute_control()`: 잠재 롤아웃 + 배치 디코딩 + 기존 비용 평가

**참고 논문**: Hafner et al. (2019) "Learning Latent Dynamics for Planning from Pixels"; Watter et al. (2015) "Embed to Control"

---

## 17. CMA-MPPI (Covariance Matrix Adaptation)

### 17.1 동기: 등방적 감쇠의 한계

DIAL-MPPI는 반복마다 기하급수적으로 노이즈를 감쇠하여 local minima를 회피한다:

```
σ^(i) = σ₀ · f^i,   f ∈ (0, 1)
```

그러나 이 감쇠는 **등방적(isotropic)**이다 — 모든 제어 차원과 타임스텝에 동일한 비율을 적용.
비용 지형이 비대칭인 경우 (예: 장애물 근처에서 선속도는 민감하지만 각속도는 자유로움)
등방적 감쇠는 비효율적이다.

### 17.2 CMA-MPPI 알고리즘

CMA-ES (Covariance Matrix Adaptation Evolution Strategy)에서 영감받아,
**보상 가중 샘플로부터 per-timestep 대각 공분산**을 학습한다.

**핵심 수식:**

```
1. 샘플링:     ε_k ~ N(0, diag(Σ_t)),   k=1,...,K,  t=0,...,N-1
2. 평균 업데이트: U ← Σ_k w_k · U_k^sampled
3. 공분산 적응: Σ̂_t = Σ_k w_k · (U_k - U)_t²        (가중 분산)
4. EMA 안정화:  Σ_t ← (1-α)·Σ_t + α·Σ̂_t
5. 클램핑:      Σ_t = clip(Σ_t, σ_min², σ_max²)
```

여기서 `w_k = softmax(-cost_k / λ)` 는 MPPI 가중치, `α ∈ (0, 1]` 는 EMA 학습률.

### 17.3 CMA-ES vs CMA-MPPI 비교

| 특성 | CMA-ES | CMA-MPPI |
|------|--------|----------|
| 목적 | Black-box 최적화 | 실시간 MPC |
| 공분산 | 풀 공분산 (n×n) | Per-timestep 대각 (N×nu) |
| 적응 | 진화 경로 (p_c, p_σ) | EMA 보상 가중 분산 |
| warm start | 없음 | 이전 스텝 Σ 전달 |
| 샘플 수 | ~4+3ln(n) | K=512+ (MPPI 샘플) |

### 17.4 DIAL-MPPI vs CMA-MPPI

| 특성 | DIAL-MPPI | CMA-MPPI |
|------|-----------|----------|
| 노이즈 스케줄 | 고정: σ₀·f^i | 적응: 가중 분산 |
| 등방성 | 모든 차원 동일 감쇠 | 차원별 독립 적응 |
| 호라이즌 의존 | factor^(N-1-t) | per-timestep Σ_t |
| 하이퍼파라미터 | f, horizon_factor | α, σ_min, σ_max |
| 장점 | 단순, 예측 가능 | 비용 지형 적응 |
| 단점 | 비대칭 비용 비효율 | 노이즈 추정 불안정 가능 |

### 17.5 코드 연결

- **파라미터**: `mppi_params.py:CMAMPPIParams`
  - `n_iters_init/n_iters`: cold/warm start 반복 횟수
  - `cov_learning_rate`: EMA α (0.5 기본, 1.0=즉시 반영)
  - `sigma_min/sigma_max`: 공분산 클램핑 범위
  - `elite_ratio`: 0=전체 가중치, >0=상위 비율만
- **컨트롤러**: `cma_mppi.py:CMAMPPIController`
  - `compute_control()`: 다중 반복 + 공분산 적응 루프
  - `cov`: (N, nu) per-timestep 대각 공분산
  - receding horizon에서 U와 Σ 동시 shift

---

## 18. DBaS-MPPI (Discrete Barrier States)

> **핵심**: 상태를 barrier state β(x)로 증강하여 장애물 정보를 내재화하고,
> barrier 비용에 비례하는 적응적 탐색 노이즈로 밀집 장애물에서도 가중치 퇴화 방지.

### 18.1 동기

기존 MPPI는 장애물 회피를 **비용 함수 페널티**(ObstacleCost)로만 처리:
- 밀집 장애물: K개 샘플 대부분이 충돌 → **가중치 퇴화** (ESS → 1)
- 좁은 통로: 가우시안 노이즈로 좁은 통로 통과 확률 극히 낮음
- 고정 노이즈: 자유 공간에서 과도한 탐색 vs 장애물 근처에서 부족한 탐색

DBaS-MPPI (arXiv:2502.14387)의 해법:
1. **Barrier state 증강**: h(x) → B(h) → β(x)로 장애물 정보를 상태에 내재화
2. **적응적 탐색**: barrier 비용에 비례하여 σ 스케일링

### 18.2 Log Barrier 함수

원형 장애물 $(o_x, o_y, r)$에 대한 제약 함수:

$$h_i(\mathbf{x}) = \|\mathbf{p} - \mathbf{p}_{obs}\|^2 - (r + m)^2$$

여기서 $m$은 안전 마진. $h > 0$이면 안전, $h < 0$이면 충돌.

Log barrier 변환:

$$B(h) = -\log(\max(h, h_{min}))$$

- $h \gg 0$: $B \approx 0$ (안전 → 비용 없음)
- $h \to 0^+$: $B \to \infty$ (경계 → 무한 비용)
- $h_{min}$ 클리핑: 특이점 방지

벽 제약 $(axis, val, dir)$에 대해:

$$h_j(\mathbf{x}) = dir \cdot (x_{axis} - val)$$

### 18.3 Barrier State Dynamics

레퍼런스 궤적 $\mathbf{x}_d$에 대한 barrier state 전파:

$$\beta(\mathbf{x}_{k+1}) = B(h(\mathbf{x}_{k+1})) - \gamma \left( B(h(\mathbf{x}_d)) - \beta(\mathbf{x}_k) \right)$$

- $\gamma \in (0,1)$: 수렴률
  - $\gamma \to 0$: $\beta \approx B(h)$ (순수 barrier)
  - $\gamma \to 1$: 이전 상태 기억 증가 (더 부드러운 전이)
- 레퍼런스 근처에서 $B(h(\mathbf{x})) \approx B(h(\mathbf{x}_d))$ → $\beta$ 안정화

### 18.4 Barrier 비용

$$C_B = R_B \sum_{t=0}^{N} \sum_{c=1}^{C} \max(\beta_{t,c}, 0)$$

- $R_B$: barrier 비용 가중치 (`barrier_weight`)
- 양수 $\beta$만 비용에 기여 (안전 영역의 음수 $\beta$는 무시)

총 비용: $J_k = J_{base,k} + C_{B,k}$

### 18.5 적응적 탐색

Best 궤적의 barrier 비용으로 탐색 노이즈 스케일링:

$$S_e = \mu \cdot \log(e + C_B(\mathbf{X}^*_{best}))$$

$$\sigma_{eff} = \sigma \cdot (1 + S_e)$$

- **자유 공간**: $C_B \approx 0$ → $S_e \approx \mu$ → 정밀 제어
- **장애물 근처**: $C_B \gg 0$ → $S_e \gg 1$ → 확대된 탐색
- $\mu$: 탐색 계수 (`exploration_coeff`)

### 18.6 알고리즘

```
Input: state x, reference x_d, obstacles, walls
       Previous: U (N, nu), adaptive_scale

1. sigma_eff = sigma * (1 + adaptive_scale)
2. noise ~ N(0, sigma_eff²)  →  (K, N, nu)
3. sampled_controls = U + noise
4. trajectories = rollout(x, sampled_controls)  →  (K, N+1, nx)
5. base_costs = cost_function(trajectories, controls, reference)
6. h = constraint_values(trajectories[:,:,:2])  →  (K, N+1, C)
7. B = -log(max(h, h_min))
8. beta dynamics propagation → barrier_states  →  (K, N+1, C)
9. barrier_costs = RB * sum(max(beta, 0))
10. total_costs = base_costs + barrier_costs
11. weights = softmax(-total_costs / lambda)
12. U += sum(weights * noise)
13. optimal_control = U[0]
14. U = shift(U)  [receding horizon]
15. adaptive_scale = mu * log(e + barrier_costs[best])

Return: optimal_control, info
```

### 18.7 Shield-MPPI / CBF-MPPI 와의 비교

| 특성 | CBF-MPPI | Shield-MPPI | DBaS-MPPI |
|------|----------|-------------|-----------|
| 안전 보장 | 소프트 (비용) | 하드 (QP 필터) | 소프트 (barrier) |
| 노이즈 적응 | 없음 | 없음 | 있음 (Se) |
| 벽 제약 | 없음 | CBF로 인코딩 | 네이티브 지원 |
| 계산 복잡도 | O(KC) | O(KNC) QP | O(KNC) 산술 |
| 가중치 퇴화 | 취약 | 불가능 (모두 안전) | 적응적 완화 |
| 동적 장애물 | 정적만 | 정적만 | update_obstacles() |

### 18.8 코드 연결

- **파라미터**: `mppi_params.py:DBaSMPPIParams`
  - `dbas_obstacles`: 원형 장애물 [(x,y,r), ...]
  - `dbas_walls`: 벽 제약 [('x'|'y', val, dir), ...]
  - `barrier_weight`: $R_B$, `barrier_gamma`: $\gamma$
  - `exploration_coeff`: $\mu$, `h_min`: 클리핑
  - `use_adaptive_exploration`: $S_e$ 활성화
- **컨트롤러**: `dbas_mppi.py:DBaSMPPIController`
  - `_compute_constraint_values()`: 원형 + 벽 제약 일괄 계산
  - `_barrier_function()`: $B(h) = -\log(\max(h, h_{min}))$
  - `_compute_barrier_cost()`: barrier state dynamics + 비용
  - `update_obstacles()`: 동적 장애물 실시간 갱신
  - info에 `dbas_stats` (adaptive_scale, barrier_cost, min_constraint)

---

## 19. R-MPPI (Robust MPPI)

> **핵심**: 피드백 게인 K를 MPPI 샘플링 루프 내부에 통합하여,
> 명목(nominal) 궤적과 실제(real) 궤적을 동시에 롤아웃하고 실제 궤적의 비용으로 가중치 계산.

### 19.1 동기: Tube-MPPI의 한계

Tube-MPPI (§3)는 **분리된 2계층** 구조로 외란에 대응한다:

```
1계층: MPPI → 명목 궤적 u_nom (외란 무시)
2계층: 피드백 K·(x_real - x_nom) 보정 (사후 보정)
```

이 분리 구조의 한계:
- 명목 궤적 최적화 시 **피드백 효과를 고려하지 않음**
- 외란이 큰 환경에서 명목 궤적과 실제 궤적의 괴리 증가
- 비용은 명목 궤적 기준으로 평가 → 실제 성능과 괴리

R-MPPI (Gandhi et al., RAL 2021, arXiv:2102.09027)의 해법:
- **피드백을 샘플링 루프에 통합** → 실제 궤적 기반 비용 평가
- 증강 상태 $z = [x_{nom}, x_{real}]$로 명목/실제 동시 전파
- 외란 모델을 명시적으로 샘플링하여 기대 비용 최소화

### 19.2 Tube-MPPI vs R-MPPI 비교

| 특성 | Tube-MPPI | R-MPPI |
|------|-----------|--------|
| 구조 | 분리: MPPI + 피드백 | 통합: 피드백 내재화 |
| 비용 평가 | 명목 궤적 $x_{nom}$ | **실제 궤적 $x_{real}$** |
| 피드백 반영 | 사후 보정 | 최적화 시 고려 |
| 외란 모델링 | 암묵적 (K로 억제) | 명시적 (샘플링) |
| 상태 차원 | $n_x$ | $2n_x$ (증강) |
| 계산 비용 | 1x 롤아웃 | 2x 롤아웃 (nom + real) |

### 19.3 증강 상태 롤아웃

명목 상태와 실제 상태를 동시에 전파:

**명목 동역학** (외란 없음):
$$x_{nom}(t+1) = F(x_{nom}(t), v(t))$$

**실제 동역학** (피드백 + 외란):
$$x_{real}(t+1) = F(x_{real}(t), v(t) + K \cdot (x_{real}(t) - x_{nom}(t))) + w(t)$$

여기서:
- $v(t) = U(t) + \epsilon_k(t)$: 샘플된 제어 입력
- $K$: 피드백 게인 행렬 (`feedback_gain_scale` 기반)
- $w(t)$: 외란 샘플 (모드에 따라 다름)

### 19.4 외란 모델

3가지 외란 모드를 지원:

| 모드 | $w(t)$ 생성 | 적합 상황 |
|------|------------|---------|
| `gaussian` | $w \sim \mathcal{N}(0, \sigma_d^2 I)$ | 일반적 프로세스 노이즈 |
| `adversarial` | $w = \alpha \cdot \nabla_x J$ 방향 | 최악의 경우 강건성 |
| `none` | $w = 0$ | 외란 없는 피드백 통합만 |

- **Gaussian**: 각 샘플 $k$, 타임스텝 $t$에 독립 노이즈 적용
- **Adversarial**: 비용 증가 방향으로 외란 주입 → minimax 강건성
- **None**: 외란 없이 피드백 통합 효과만 평가

### 19.5 비용 계산

핵심 차이: **실제 궤적에 대한 비용**으로 가중치 계산.

$$J_k = \sum_{t=0}^{N-1} q(x_{real,k}(t), ref(t)) + R \cdot v_k(t)^T v_k(t)$$

- $q(\cdot)$: 상태 비용 (기존 MPPI와 동일)
- 실제 궤적 $x_{real,k}$에서 평가 → 피드백 효과 반영
- 가중치: $w_k = \text{softmax}(-J_k / \lambda)$

### 19.6 알고리즘

```
Input: state x, reference ref, feedback gain K
       Previous: U (N, nu)
       Disturbance mode: gaussian / adversarial / none

1. x_nom_0 = x,  x_real_0 = x
2. for k = 1 to K:
   2a. noise_k ~ N(0, sigma²)  → (N, nu)
   2b. v_k = U + noise_k
   2c. if disturbance_mode == "gaussian":
         w_k ~ N(0, sigma_d²)  → (N, nx)
       elif disturbance_mode == "adversarial":
         w_k = alpha * adversarial_direction
       else:
         w_k = 0
   2d. for t = 0 to N-1:
         x_nom_k(t+1) = F(x_nom_k(t), v_k(t))
         u_fb = v_k(t) + K · (x_real_k(t) - x_nom_k(t))
         x_real_k(t+1) = F(x_real_k(t), u_fb) + w_k(t)
3. costs_k = Σ_t q(x_real_k(t), ref(t))  ← REAL trajectory
4. weights = softmax(-costs / lambda)
5. U += Σ_k weights_k · noise_k
6. optimal_control = U[0]
7. U = shift(U)

Return: optimal_control, info
```

### 19.7 Shield-MPPI / Tube-MPPI / R-MPPI 비교

| 특성 | Shield-MPPI | Tube-MPPI | R-MPPI |
|------|-------------|-----------|--------|
| 안전 보장 | 하드 (QP) | 소프트 (피드백) | 소프트 (피드백+외란) |
| 피드백 | 없음 | 사후 보정 | **루프 내 통합** |
| 외란 모델 | 없음 | 암묵적 | **명시적 샘플링** |
| 비용 기준 | 명목 | 명목 | **실제** |
| 학습 필요 | 아니오 | 아니오 | 아니오 |
| 계산 비용 | O(KN) QP | O(KN) | O(2KN) |

### 19.8 코드 연결

- **파라미터**: `mppi_params.py:RobustMPPIParams`
  - `disturbance_std`: 외란 표준편차 $\sigma_d$
  - `feedback_gain_scale`: 피드백 게인 스케일
  - `disturbance_mode`: `"gaussian"` | `"adversarial"` | `"none"`
  - `robust_alpha`: adversarial 외란 크기 $\alpha$
  - `use_feedback`: 피드백 통합 활성화
  - `n_disturbance_samples`: 외란 샘플 수 (기대값 근사)
- **컨트롤러**: `robust_mppi.py:RobustMPPIController`
  - `compute_control()`: 증강 상태 롤아웃 + 실제 궤적 비용
  - info에 `robust_stats` (nominal_trajectories, real_trajectories, disturbance_mode)

---

## 20. ASR-MPPI (Adaptive Spectral Risk)

> **핵심**: Spectral Risk Measure (SRM)의 왜곡 함수 φ(q)로 비용 분위수를
> 비균일 가중하여, CVaR의 경질 절단을 연속적 곡선으로 일반화.

### 20.1 Spectral Risk Measure

**정의**: 왜곡 함수 φ: [0,1] → [0,1] (φ(0)=0, φ(1)=1, 단조 증가)에 대해:

$$
\text{SRM}_\phi(S) = \int_0^1 \text{VaR}_q(S) \cdot \phi'(q) \, dq
$$

여기서 VaR_q(S)는 비용 S의 q-분위수, φ'(q)는 왜곡 밀도(density).

**MPPI 가중치 적용**: 비용 S_{(1)} ≤ ... ≤ S_{(K)} (정렬), q_k = k/K:

$$
w_k \propto \phi'(q_k) \cdot \exp\left(-\frac{S_{(k)}}{\lambda}\right)
$$

### 20.2 왜곡 함수 4종

| 함수 | φ(q) | φ'(q) | 특성 |
|------|-------|--------|------|
| **Sigmoid** | σ(β(q-α))† | β·σ·(1-σ)/Z | 부드러운 S-곡선, 가장 유연 |
| **Power** | q^γ | γ·q^(γ-1) | γ<1: 낮은 비용 강조, γ>1: 높은 비용 |
| **Dual Power** | 1-(1-q)^γ | γ·(1-q)^(γ-1) | Power의 반대 꼬리 |
| **CVaR** | 0 (q<1-α), (q-(1-α))/α | 0 or 1/α | 기존 Risk-Aware 호환 (계단) |

†정규화: (φ(q)-φ(0))/(φ(1)-φ(0))로 φ(0)=0, φ(1)=1 보장.

### 20.3 CVaR/Tsallis는 SRM의 특수 경우

- **CVaR_α**: φ(q) = max(0, (q-(1-α))/α) → 계단 왜곡 → SRM = CVaR
- **Sigmoid β→∞**: σ(β(q-α)) → Heaviside → CVaR 수렴
- **Tsallis q-exponential**: q-지수 가중도 특정 왜곡 함수로 표현 가능

### 20.4 ESS 기반 적응 알고리즘

```
매 제어 스텝마다:
  1. ESS = 1/Σw_k², ess_ratio = ESS/K
  2. if ess_ratio < min_ess_ratio:
       β_target = β · 0.8        (더 균일한 가중)
     elif ess_ratio > 0.5:
       β_target = β · 1.05       (더 집중)
     else:
       β_target = β
  3. β ← (1 - rate)·β + rate·β_target    (EMA 업데이트)
  4. β = clip(β, 0.5, 50.0)
```

### 20.5 코드 매핑

- **파라미터**: `mppi_params.py:ASRMPPIParams`
  - `distortion_type`: sigmoid | power | dual_power | cvar
  - `distortion_alpha/beta/gamma`: 왜곡 함수 파라미터
  - `use_adaptive_risk`: ESS 기반 자동 조절
- **컨트롤러**: `spectral_risk_mppi.py:ASRMPPIController`
  - `_compute_weights()`: 정렬 → φ'(q) → spectral_weights → 정규화
  - `_eval_distortion()` / `_eval_distortion_derivative()`: 왜곡 함수 계산
  - `_adapt_parameters()`: ESS 기반 β 적응
  - `get_risk_statistics()`: SRM 값, ESS, α/β 추적

---

## 21. 변형 선택 가이드

### 의사결정 트리

```
MPPI 변형 선택
│
├─ 외란이 있는가?
│  ├─ Yes ─┬─ 피드백 통합 원함?   → R-MPPI
│  │       └─ 분리 구조 OK?      → Tube-MPPI
│  └─ No ─┐
│          │
├─ 모델 불확실성이 있는가?
│  ├─ Yes ─┬─ 확률적 안전 필요? → C2U-MPPI
│  │       ├─ 적응 탐색 원함?   → Uncertainty-Aware MPPI
│  │       └─ 모델 외삽 회피?   → BNN-MPPI
│  └─ No ─┐
│          │
├─ 다중 모달 해가 있는가?
│  ├─ Yes ─┬─ 학습 데이터 있음? → Flow-MPPI
│  │       ├─ 실시간 필요?      → SVG-MPPI
│  │       └─ 최고 다양성?      → SVMPC
│  └─ No ─┐
│          │
├─ 매끄러운 제어가 필요한가?
│  ├─ Yes ─┬─ 메모리 절약?      → Spline-MPPI
│  │       └─ jerk 최소화?      → Smooth MPPI
│  └─ No ─┐
│          │
├─ 밀집 장애물 / 좁은 통로?
│  ├─ Yes ─┬─ 벽 제약 필요?       → DBaS-MPPI
│  │       └─ 동적 장애물?        → DBaS-MPPI (update_obstacles)
│  └─ No ─┐
│          │
├─ 안전이 최우선인가?
│  ├─ Yes → Risk-Aware MPPI (+ Safety 기법, SAFETY_THEORY.md 참조)
│  └─ No ─┐
│          │
├─ 지역 최적 탈출?
│  ├─ Yes ─┬─ 등방 감쇠 OK?     → DIAL-MPPI
│  │       └─ 비용 지형 적응?   → CMA-MPPI
│  └─ No ─┐
│          │
├─ 수치 안정성 이슈?
│  ├─ Yes → Log-MPPI
│  └─ No ─┐
│          │
├─ 탐색-활용 세밀 제어?
│  ├─ Yes → Tsallis-MPPI
│  └─ No → Vanilla MPPI
```

### 트레이드오프 매트릭스

```
┌──────────────┬────────┬────────┬────────┬────────┬────────┐
│   변형       │ 속도   │ 정확도  │ 강건성  │ 메모리  │ 복잡도  │
├──────────────┼────────┼────────┼────────┼────────┼────────┤
│ Vanilla      │ ★★★★★ │ ★★★   │ ★★    │ ★★★★★ │ ★     │
│ Tube         │ ★★★★  │ ★★★   │ ★★★★★ │ ★★★★  │ ★★    │
│ Log          │ ★★★★★ │ ★★★   │ ★★★   │ ★★★★★ │ ★     │
│ Tsallis      │ ★★★★★ │ ★★★★  │ ★★★   │ ★★★★★ │ ★★    │
│ Risk-Aware   │ ★★★★  │ ★★★   │ ★★★★  │ ★★★★★ │ ★★    │
│ Smooth       │ ★★★★  │ ★★★★  │ ★★★   │ ★★★★  │ ★★    │
│ Spline       │ ★★★★★ │ ★★★   │ ★★★   │ ★★★★★ │ ★★★   │
│ SVMPC        │ ★★    │ ★★★★★ │ ★★★★  │ ★★    │ ★★★★  │
│ SVG-MPPI     │ ★★★★  │ ★★★★  │ ★★★★  │ ★★★   │ ★★★   │
│ DIAL         │ ★★★   │ ★★★★★ │ ★★★★  │ ★★★★  │ ★★★   │
│ Uncertainty  │ ★★★   │ ★★★★  │ ★★★★  │ ★★★★  │ ★★★   │
│ C2U          │ ★★    │ ★★★★  │ ★★★★★ │ ★★★   │ ★★★★  │
│ Flow         │ ★★★   │ ★★★★★ │ ★★★★  │ ★★★   │ ★★★★★ │
│ BNN          │ ★★★   │ ★★★★  │ ★★★★★ │ ★★★★  │ ★★★   │
│ CMA          │ ★★★   │ ★★★★★ │ ★★★★  │ ★★★★  │ ★★★   │
│ DBaS         │ ★★★★  │ ★★★★  │ ★★★★  │ ★★★★  │ ★★★   │
│ R-MPPI       │ ★★★   │ ★★★★  │ ★★★★★ │ ★★★   │ ★★★   │
└──────────────┴────────┴────────┴────────┴────────┴────────┘
```

### 시나리오별 추천

| 시나리오 | 추천 변형 | 이유 |
|---------|----------|------|
| 깨끗한 환경, 기본 | Vanilla / Log | 단순, 빠름 |
| 외란 환경 (바람, 분리) | Tube + DIAL | 강건성 + 전역 탐색 |
| 외란 환경 (통합 피드백) | R-MPPI | 피드백 내재화 + 실제 궤적 비용 |
| 비대칭 장애물 | CMA | 비용 지형 적응 공분산 |
| 밀집 장애물 / 좁은 통로 | DBaS | barrier state + 적응적 탐색 |
| 장애물 + 안전 | Risk-Aware + Shield | CVaR + CBF 보장 |
| 다중 경로 | Flow / SVG | 다중 모달 샘플링 |
| 실시간 제약 | Spline / SVG | 메모리/계산 효율 |
| 매끄러운 제어 | Smooth / Spline | 연속 제어 보장 |
| 모델 불확실성 | C2U / Uncertainty / BNN | 불확실성 인식 |
| 정밀 추적 | Tsallis (q>1) | 활용 강화 |

### 변형 조합 추천

MPPI 변형들은 대부분 **직교적(orthogonal)**으로, 여러 기법을 조합할 수 있다:

```
┌──────────────────────────────────────────────────────────────┐
│                    조합 호환성 매트릭스                       │
│                                                              │
│           Log  Tsallis  Smooth  Spline  Tube  Risk  DIAL    │
│ Log        -    ✗       ✓       ✓       ✓     ✓     ✓      │
│ Tsallis    ✗    -       ✓       ✓       ✓     ✗     ✓      │
│ Smooth     ✓    ✓       -       ✗       ✓     ✓     ✓      │
│ Spline     ✓    ✓       ✗       -       ✓     ✓     ✓      │
│ Tube       ✓    ✓       ✓       ✓       -     ✓     ✓      │
│ Risk       ✓    ✗       ✓       ✓       ✓     -     ✓      │
│ DIAL       ✓    ✓       ✓       ✓       ✓     ✓     -      │
│                                                              │
│ ✓: 직접 조합 가능                                            │
│ ✗: 가중치 계산이 충돌하므로 택 1 (Log vs Tsallis vs Risk)    │
└──────────────────────────────────────────────────────────────┘
```

**추천 조합 예시**:

```
1. 강건 + 매끄러운 제어:
   Tube + Smooth + Log
   → 외란 대응 + 제어 연속성 + 수치 안정성

2. 안전 최우선:
   Tube + Risk-Aware + Shield (SAFETY_THEORY.md 참조)
   → 외란 강건 + CVaR 보수 + CBF 안전 보장

3. 복잡한 환경 (다중 장애물):
   DIAL + Spline + Log
   → 전역 탐색 + 메모리 절약 + 수치 안정

4. 학습 기반 + 안전:
   Flow + C2U + Shield
   → 다중 모달 샘플 + 확률적 안전 + CBF 보장

5. 최고 성능 (계산 여유):
   SVG + DIAL + Log
   → SVGD 다양성 + 어닐링 전역 탐색 + 안정성

6. 최소 계산 (임베디드):
   Spline + Log
   → 차원 축소 + 수치 안정 (가장 가벼운 조합)
```

### 계산 예산 가이드

```
사용 가능한 계산 시간에 따른 추천:

  ┌─────────────────────────────────────────────────────────┐
  │ 예산        │ 추천 변형              │ K    │ 특이사항  │
  ├─────────────┼────────────────────────┼──────┼───────────┤
  │ < 5ms       │ Vanilla + Log          │ 256  │ CPU 전용  │
  │ (200Hz)     │ Spline + Log           │ 512  │           │
  ├─────────────┼────────────────────────┼──────┼───────────┤
  │ 5~20ms      │ Smooth + Log           │ 1024 │ CPU       │
  │ (50~200Hz)  │ Tsallis + Log          │ 1024 │           │
  │             │ Risk-Aware             │ 1024 │           │
  ├─────────────┼────────────────────────┼──────┼───────────┤
  │ 20~50ms     │ Tube + Smooth          │ 2048 │ CPU/GPU   │
  │ (20~50Hz)   │ Uncertainty-Aware      │ 2048 │           │
  │             │ BNN-MPPI               │ 2048 │           │
  │             │ SVG-MPPI               │ 2048 │ G=64      │
  ├─────────────┼────────────────────────┼──────┼───────────┤
  │ 50~100ms    │ DIAL (3 iter)          │ 4096 │ GPU 권장  │
  │ (10~20Hz)   │ C2U-MPPI              │ 4096 │           │
  │             │ Flow-MPPI             │ 4096 │           │
  ├─────────────┼────────────────────────┼──────┼───────────┤
  │ > 100ms     │ SVMPC                  │ 256  │ K² 주의   │
  │ (< 10Hz)    │ DIAL + SVG             │ 8192 │ GPU 필수  │
  └─────────────┴────────────────────────┴──────┴───────────┘
```

### 실세계 배포 체크리스트

```
━━━ MPPI 실세계 배포 전 확인 사항 ━━━

□ 1. 모델 정확도 검증
  □ 실제 로봇에서 open-loop 예측 오차 측정
  □ 오차가 크면: Tube-MPPI 또는 학습 보정 모델 사용
  □ 불확실성 추정 가능 여부 확인 (GP/Ensemble)

□ 2. 실시간성 확인
  □ worst-case 계산 시간 < 제어 주기의 80%
  □ 메모리 사용량 확인 (임베디드 제약)
  □ GPU 사용 가능 여부 → K 결정

□ 3. 안전성 검증
  □ 장애물 회피 테스트 (정적/동적)
  □ 비상 정지(E-stop) 통합 확인
  □ 제어 포화(saturation) 처리
  □ 안전 필터(CBF/Shield) 추가 여부 결정

□ 4. 센서/통신 지연
  □ 상태 추정 지연 보상 (상태 예측)
  □ 제어 명령 지연 보상 (look-ahead)
  □ 통신 드롭아웃 시 안전 모드

□ 5. 파라미터 튜닝
  □ σ: 실제 제어 범위의 10-30%로 시작
  □ λ: ESS 기반 적응 온도 활성화
  □ K: 실시간 제약 내 최대값
  □ N: 최대 속도 × dt × N > 장애물 탐지 거리

□ 6. 모니터링
  □ ESS 실시간 모니터링 (너무 낮으면 경고)
  □ 비용 분포 로깅 (이상 탐지)
  □ 최적 궤적 시각화 (디버깅용)
  □ 계산 시간 히스토그램 (실시간성 확인)

□ 7. 폴백(Fallback) 전략
  □ MPPI 실패 시 안전 제어기로 전환
  □ 무한 비용(NaN/Inf) 감지 및 복구
  □ 궤적 발산 감지 및 재초기화
```

---

## 참고 문헌

### MPPI 핵심 논문
1. Williams, G. et al. (2016). "Aggressive Driving with Model Predictive Path Integral Control." RSS.
2. Williams, G. et al. (2018). "Robust Sampling Based Model Predictive Control with Sparse Objective Information." L4DC.

### MPPI 변형
3. Yin, H. et al. (2021). "Trajectory Distribution Optimization for Model Predictive Path Integral Control Using Tsallis Entropy." arXiv.
4. Yin, H. et al. (2023). "Risk-Aware Model Predictive Path Integral Control." L4DC.
5. Lambert, A. et al. (2020). "Stein Variational Model Predictive Control." CoRL.
6. Kim, J. et al. (2021). "Smooth Model Predictive Path Integral Control without Smoothing." ICRA.
7. Bhardwaj, M. et al. (2024). "Spline-MPPI: Efficient Sampling-Based MPC." ICRA.
8. Kondo, Y. et al. (2024). "SVG-MPPI: Sampling-based MPC with Stein Variational Guided Particles." IROS.

### 생성 모델 및 Flow Matching
9. Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." ICLR.
10. Liu, X. et al. (2023). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR.

### 이론적 기초
11. Kappen, H. J. (2005). "Path integrals and symmetry breaking for optimal control theory." J. Stat. Mech.
12. Theodorou, E. et al. (2010). "A Generalized Path Integral Control Approach to Reinforcement Learning." JMLR.
13. Liu, Q. & Wang, D. (2016). "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm." NeurIPS.
14. Artzner, P. et al. (1999). "Coherent Measures of Risk." Mathematical Finance.
15. Hajek, B. (1988). "Cooling Schedules for Optimal Annealing." Mathematics of Operations Research.

### BNN Surrogate
16. Ezeji, A. et al. (2025). "Bayesian Neural Network Surrogate Models for Model Predictive Control."
17. Le Cleac'h, S. et al. (2024). "CoVO-MPC: Covariance-Optimal Model Predictive Control." RSS.

### Robust MPPI
18. Gandhi, M. et al. (2021). "Robust Model Predictive Path Integral Control: Analysis and Performance Guarantees." RAL. arXiv:2102.09027.
