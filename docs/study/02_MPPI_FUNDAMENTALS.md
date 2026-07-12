# MPPI 기초 공부 자료 — Free Energy에서 43개 변형까지

> **대상**: [01_MPC_FUNDAMENTALS.md](01_MPC_FUNDAMENTALS.md)를 읽은 로보틱스
> 엔지니어. HJB, LQR, receding horizon 개념을 전제한다.
>
> **성격**: 학습용 문서. "왜 softmax 가중 평균이 최적 제어인가"를 밑바닥부터
> 유도하고, 이 repo의 `base_mppi.py`와 한 줄씩 대응시킨다.
> 변형별 상세 레퍼런스는 [docs/MPPI_THEORY.md](../MPPI_THEORY.md)가 담당한다 —
> 여기서는 **분류 축과 대표 예시**만 심화한다.

---

## 목차

1. [확률적 최적 제어와 Free Energy](#1-확률적-최적-제어와-free-energy)
2. [Path Integral Control — Kappen 계열 유도](#2-path-integral-control--kappen-계열-유도)
3. [Information-Theoretic MPPI — Williams 2017 전체 유도](#3-information-theoretic-mppi--williams-2017-전체-유도)
4. [핵심 하이퍼파라미터의 이론적 의미](#4-핵심-하이퍼파라미터의-이론적-의미)
5. [ESS와 weight degeneracy](#5-ess와-weight-degeneracy)
6. [의사코드 ↔ base_mppi.py 대응표](#6-의사코드--base_mppipy-대응표)
7. [변형 분류 체계 — 43개를 5+1개 축으로](#7-변형-분류-체계--43개를-51개-축으로)
8. [알려진 한계와 연구 전선 (2026)](#8-알려진-한계와-연구-전선-2026)
9. [연습문제](#9-연습문제)
10. [부록 — 더 공부하기 위한 자료](#10-부록--더-공부하기-위한-자료)

---

## 1. 확률적 최적 제어와 Free Energy

### 1.1 출발점: min을 계산하기 어렵다

01 문서의 이산시간 OCP를 다시 본다. 제어 시퀀스 `U = (u_0,…,u_{N-1})`,
궤적 비용을

```
S(U) = Σ_{t=0}^{N-1} l(x_t, u_t) + l_f(x_N)     (x는 U로 rollout해서 결정)
```

라 하자. 풀고 싶은 것은 `min_U S(U)` — 하지만 S는 비볼록이고 (장애물),
미분도 어렵다 (블랙박스 동역학). 여기서 관점을 바꾼다.

### 1.2 Free energy: min의 부드러운 버전

**정의** (통계역학에서 빌려온 이름):

```
F = -λ log E_{U~p}[ exp(-S(U)/λ) ] ,       λ > 0  ("온도")
```

p는 어떤 기준 분포 (예: 현재 계획 주변의 가우시안). 이 이상한 변환이 왜
유용한가? 세 가지 성질:

**성질 1 — F는 "soft min"이다.** 유한 표본 버전으로 보면 명확하다:

```
-λ log ( (1/K) Σ_k e^{-S_k/λ} )
    λ → 0   :  → min_k S_k          (최소 비용 항이 지수적으로 지배)
    λ → ∞   :  → (1/K) Σ_k S_k      (평균; 테일러 전개로 확인, 연습문제 1)
```

즉 λ가 **"min과 mean 사이를 보간하는 손잡이"**다. min은 미분 불가능하고
탐욕적이지만, soft min은 매끄럽고 주변 정보를 반영한다.

**성질 2 — 최적화가 기댓값 계산으로 바뀐다.** min_U는 탐색 문제지만,
E_p[·]는 **몬테카를로로 추정 가능한 적분**이다:

```
E_p[e^{-S/λ}] ≈ (1/K) Σ_{k=1}^{K} e^{-S(U_k)/λ},    U_k ~ p    ← K개 병렬 rollout!
```

미분도, 볼록성도 필요 없다. 시뮬레이터에 U_k를 넣고 비용만 받으면 된다.
이것이 MPPI가 GPU 병렬화와 블랙박스 모델에 강한 근본 이유다.

**성질 3 — 변분 등식 (이 문서에서 가장 중요한 수식).** 임의의 분포 q에 대해:

```
F = -λ log E_p[e^{-S/λ}] = min_q { E_q[S] + λ·KL(q ‖ p) }
```

**유도** (한 줄씩):

```
-λ log E_p[e^{-S/λ}]
  = -λ log ∫ q(U) · (p(U)/q(U)) e^{-S(U)/λ} dU          q를 곱하고 나눔 (importance)
  ≤ -λ ∫ q(U) log[ (p/q) e^{-S/λ} ] dU                   Jensen: log E ≥ E log, 부호 반전
  = -λ E_q[ log(p/q) ] + E_q[S]
  = E_q[S] + λ·KL(q ‖ p)                                  KL 정의
```

등호는 `log[(p/q)e^{-S/λ}]`가 상수일 때, 즉:

```
q*(U) = p(U) · exp(-S(U)/λ) / η ,     η = E_p[e^{-S/λ}]    ← "Gibbs 분포"
```

**해석**: free energy 최소화 = "기대 비용 E_q[S]을 낮추되, 기준 분포 p에서
너무 멀어지지 마라 (KL 페널티, 세기 λ)". 그리고 그 최적 분포 q*는
**낮은 비용 궤적에 exp(-S/λ)만큼 확률을 몰아준 분포**다.

이미 MPPI의 형태가 보인다: `exp(-S/λ)` — softmax 가중치의 분자가 바로 q*의
밀도비다. §3에서 이 관찰을 알고리즘으로 완성한다.

### 1.3 미리 보는 그림

```
비용 지형 S(U)          exp(-S/λ)  (λ 작음)        exp(-S/λ)  (λ 큼)
                        ← 최적 분포 q*의 모양 →
   ╲    ╱╲    ╱           ▂█▂                        ▄▄███▄▄▄▆▄▄
    ╲  ╱  ╲  ╱            (min 근처만)               (넓게 퍼짐)
     ╲╱    ╲╱          날카로움 = 활용(exploit)     퍼짐 = 탐색(explore)
   2개 우물(다중모달)
```

다중 모달 비용이면 q*도 다중 모달 — MPPI가 "장애물 왼쪽/오른쪽" 두 모드를
동시에 담을 수 있는 이유. (단, 가우시안 하나로 q*를 근사하는 순간 이 장점이
제한된다 — §8의 한계와 SVMPC/Flow 계열 변형의 동기.)

---

## 2. Path Integral Control — Kappen 계열 유도

역사적으로 MPPI 이전에, "HJB를 선형화하면 경로적분이 나온다"는 발견이
있었다 (Kappen 2005, Theodorou PI² 2010). 직관 수준으로 따라가 본다.
(01 문서 §2.3의 확률적 HJB에서 출발.)

### 2.1 설정

```
동역학:  dx = f(x)dt + G(x)(u dt + dw)      ← 노이즈가 제어와 같은 채널로 들어옴
비용:    E[ ∫ (ℓ(x) + ½ uᵀR u) dt + l_f ]   ← 제어 비용은 이차
핵심 가정: λ G G ᵀ = G R⁻¹ Gᵀ · λ  …  요약하면  R = λ Σ_w⁻¹
          "제어가 비싼 방향 = 노이즈가 작은 방향" (비례 상수 λ)
```

이 가정이 인위적으로 보이지만, 의미는 자연스럽다: **탐색(노이즈)이 공짜인
방향으로는 제어도 싸야** 이론이 맞아떨어진다. MPPI에서 λ와 σ가 독립이
아니라는 사실(§4.1)의 뿌리가 여기다.

### 2.2 HJB 선형화 — log 변환 (Cole-Hopf)

확률적 HJB (01 문서 §2.3)에서 u에 대한 min을 명시적으로 풀면
(`u* = -R⁻¹Gᵀ∂V/∂x`, LQR §3.2와 같은 계산), 남는 PDE는 V에 **이차**인
항 `-½(∂V/∂x)ᵀGR⁻¹Gᵀ(∂V/∂x)`을 가진다 — 비선형 PDE.

**마법의 치환**: `V(x,t) = -λ log Ψ(x,t)` 를 대입한다. 연쇄법칙으로:

```
∂V/∂x = -λ (∂Ψ/∂x)/Ψ
∂²V/∂x² = -λ [ ∂²Ψ/∂x²·Ψ - (∂Ψ/∂x)(∂Ψ/∂x)ᵀ ] / Ψ²
```

이차항(∂V/∂x)² 와 이토항의 (∂Ψ/∂x)² 부분이 **정확히 상쇄**된다 —
바로 R = λΣ_w⁻¹ 가정 덕분이다. 남는 것:

```
-∂Ψ/∂t = -(1/λ) ℓ(x) Ψ + fᵀ(∂Ψ/∂x) + ½ tr(GΣ_wGᵀ ∂²Ψ/∂x²)      ← Ψ에 선형!
```

### 2.3 Feynman-Kac: 선형 PDE = 기댓값

**Feynman-Kac 공식** (직관 수준): 위 형태의 선형 포물형 PDE의 해는
"**제어 없이(u=0)** 노이즈만으로 굴린 경로들에 대한 기댓값"으로 쓸 수 있다:

```
Ψ(x, t) = E_{p(τ|x)}[ exp( -(1/λ) S(τ) ) ] ,
S(τ) = ∫_t^T ℓ(x_s) ds + l_f(x_T)          (비제어 경로의 상태 비용)
```

직관적 이해: 항 `-(1/λ)ℓΨ`는 경로가 비용 ℓ을 지나갈 때마다 Ψ를 지수적으로
"증발"시키는 항이다 (화학의 흡수/붕괴 항과 동일 구조). 살아남은 경로의
질량이 Ψ — 즉 **"비용을 적게 낸 경로일수록 많이 살아남는다."**

되돌리면:

```
V(x,t) = -λ log E_p[ e^{-S(τ)/λ} ]         ← §1.2의 free energy와 동일!
```

**즉 value function 자체가 free energy다.** 그리고 최적 제어는:

```
u*(x,t) dt = E_{q*}[ dw ]   where   q*(τ) ∝ p(τ) e^{-S(τ)/λ}
```

"최적 제어 = **비용 가중치를 준 노이즈의 평균**" — MPPI 업데이트
`U += Σ w_k ε_k`의 연속시간 원형이다.

### 2.4 Kappen 유도의 한계 → Williams의 재유도

Kappen 계열은 우아하지만 제약이 세다:
- R = λΣ_w⁻¹ 구조 가정 (제어-노이즈 정렬)
- 제어-아핀 동역학
- u=0 분포에서 샘플링해야 함 (현재 계획 주변이 아니라!) → 실용상 치명적

Williams et al. (2017)의 **정보 이론적 재유도**는 이 제약을 풀고,
"임의의 명목 제어 U 주변에서 샘플링 + importance weight 보정"이라는
실용 알고리즘을 준다. 이것이 현대 MPPI고, 이 repo의 구현이다. §3에서 유도.

---

## 3. Information-Theoretic MPPI — Williams 2017 전체 유도

### 3.1 설정과 세 가지 분포

이산시간, 노이즈가 제어에 더해지는 구조:

```
x_{t+1} = f(x_t, v_t),    v_t = u_t + ε_t,   ε_t ~ N(0, Σ)
```

- `v_t`: 실제로 시스템에 들어가는 (교란된) 제어
- `V = (v_0,…,v_{N-1})`: 교란 제어 시퀀스 — 이것이 확률 변수

세 가지 분포를 구분하는 것이 유도의 전부다:

| 분포 | 정의 | 의미 |
|------|------|------|
| `p(V)` | `Π_t N(v_t; 0, Σ)` | **비제어(base)** 분포 — u=0 |
| `q_U(V)` | `Π_t N(v_t; u_t, Σ)` | **제어된** 분포 — 평균을 U로 민 것 |
| `q*(V)` | `p(V)e^{-S(V)/λ}/η` | **최적** 분포 (§1.2의 Gibbs) |

가우시안 밀도비 (Girsanov의 이산 버전 — 직접 나눠보면 됨):

```
q_U(V)/p(V) = exp( Σ_t [ u_tᵀΣ⁻¹v_t - ½u_tᵀΣ⁻¹u_t ] )        …(D)
```

### 3.2 Step 1 — 하한: free energy는 어떤 제어로도 못 넘는 벽

§1.2의 변분 등식을 q = q_U로 특수화하면:

```
F = -λ log E_p[e^{-S/λ}]  ≤  E_{q_U}[S(V)] + λ·KL(q_U ‖ p)
```

우변 둘째 항을 (D)로 계산하면 (가우시안 KL):

```
λ·KL(q_U ‖ p) = (λ/2) Σ_t u_tᵀ Σ⁻¹ u_t        ← 이차 제어 비용이 유도에서 "공짜로" 등장!
```

즉: **어떤 U를 골라도 [기대 상태비용 + 이차 제어비용] ≥ F.**
제어 비용 `½λuᵀΣ⁻¹u`는 우리가 집어넣은 게 아니라 KL에서 나온 것이며,
이때 R = λΣ⁻¹ — Kappen의 가정(§2.1)이 여기선 **결론**으로 나온다.

### 3.3 Step 2 — 최적 분포에 붙이기: KL 최소화

하한의 등호는 q_U = q*일 때인데, q*는 일반적으로 가우시안이 아니므로 정확히
맞출 수 없다. 그래서 **q* 에 가장 가까운 q_U**를 찾는다:

```
U* = argmin_U  KL( q* ‖ q_U )
```

(방향 주의: q*를 기준으로 한 KL — q*가 확률을 두는 곳을 q_U가 커버하도록.)
전개:

```
KL(q*‖q_U) = E_{q*}[log q* - log q_U]
           = const - E_{q*}[ log q_U(V) ]
           = const + E_{q*}[ Σ_t ½(v_t - u_t)ᵀΣ⁻¹(v_t - u_t) ] + const'
```

u_t에 대해 미분해서 0 (이차식이므로 유일 최소):

```
∂/∂u_t : E_{q*}[ Σ⁻¹(v_t - u_t) ] = 0    →    u_t* = E_{q*}[ v_t ]     …(★)
```

**결론 (★)**: 최적 제어 = **최적 분포 하에서 교란 제어의 평균.**
"moment matching" — 가우시안의 평균만 q*에 맞추는 것.

### 3.4 Step 3 — Importance Sampling: q*에서 샘플 못 뽑으니까

(★)의 기댓값은 q* 하의 기댓값인데, q*에서 직접 샘플링은 불가능하다
(정규화 상수 η부터 모름). 대신 **지금 갖고 있는 계획 Û 주변**
q_Û에서 샘플을 뽑고 밀도비로 보정한다:

```
u_t* = E_{q*}[v_t] = E_{q_Û}[ (q*(V)/q_Û(V)) · v_t ]
```

밀도비를 한 줄씩 전개:

```
q*(V)/q_Û(V) = [p(V) e^{-S(V)/λ} / η] / q_Û(V)                q* 정의 대입
             = (1/η) e^{-S(V)/λ} · (p(V)/q_Û(V))              항 재배열
             = (1/η) e^{-S(V)/λ} · exp(-Σ_t[û_tᵀΣ⁻¹v_t - ½û_tᵀΣ⁻¹û_t])   (D)의 역수
             = (1/η) exp( -(1/λ)[ S(V) + λΣ_t û_tᵀΣ⁻¹ε_t + ½λΣ_t û_tᵀΣ⁻¹û_t ] )
                                                               v_t = û_t + ε_t 대입, 정리
             ≡ (1/η) exp( -(1/λ) S̃(V) )
```

즉 보정된 비용:

```
S̃(V) = S(V) + λ Σ_t û_tᵀ Σ⁻¹ ε_t + ½λ Σ_t û_tᵀ Σ⁻¹ û_t
        └ 상태비용 ┘ └── importance 보정 (교차항) ──┘ └ U에만 의존 (가중치엔 상수) ┘
```

### 3.5 Step 4 — 몬테카를로: softmax 가중 평균의 탄생

K개 샘플 `ε^k ~ N(0,Σ)`, `V^k = Û + ε^k`, 각 rollout 비용 `S̃_k`:

```
u_t* ≈ Σ_k  w_k · v_t^k ,        w_k = e^{-S̃_k/λ} / Σ_j e^{-S̃_j/λ}     ← softmax(-S̃/λ)!
```

(η는 분모 정규화에 흡수 — 몰라도 됨. 이것이 self-normalized importance
sampling.) `v_t^k = û_t + ε_t^k`를 대입하면 **업데이트 형태**가 나온다:

```
u_t* ≈ Σ_k w_k (û_t + ε_t^k) = û_t + Σ_k w_k ε_t^k        (Σw_k = 1이므로)

⇒   U ← U + Σ_k w_k ε^k        ← MPPI 업데이트 식 완성
```

### 3.6 유도 요약 (한 페이지 지도)

```
free energy F = -λ log E_p[e^{-S/λ}]                                (§1.2)
   │  변분 등식
   ▼
최적 분포 q* ∝ p·e^{-S/λ}                                           (§1.2)
   │  KL(q*‖q_U) 최소화 → moment matching
   ▼
U* = E_{q*}[V]                                                       (★)
   │  importance sampling (q_Û에서 샘플)
   ▼
U* = E_{q_Û}[(q*/q_Û)·V],   비용 보정 S → S̃                        (§3.4)
   │  몬테카를로 K개
   ▼
w_k = softmax(-S̃_k/λ),   U ← U + Σ_k w_k ε^k                        (§3.5)
```

### 3.7 이 repo 구현과의 차이 (정직한 각주)

`base_mppi.py`의 `_compute_weights()`는 S̃가 아니라 **S 그대로** 쓴다
(`costs`에 교차항 `λûᵀΣ⁻¹ε` 없음). 대신:

- `ControlEffortCost(R)`가 `vᵀRv`형 제어 비용을 S 안에 직접 넣는다 —
  Williams 논문도 실전 팁으로 "보정항 없이 총비용에 제어 페널티 포함"을
  자주 쓴다 (γ 파라미터로 보정 세기 조절하는 변종도 있음).
- 수치 안정화 `exp(-(costs - min_cost)/λ)` (276-294행)는 softmax의 표준
  트릭 — 분자·분모에 같은 상수 `e^{min/λ}`를 곱한 것이므로 **수학적으로
  동일**하고 overflow만 막는다.
- 샘플에 `np.clip`을 적용한 뒤의 실제 noise는 ε' = clip(U+ε) - U인데,
  vanilla 구현(141-145행)은 clip 후에도 원래 ε로 업데이트한다 — 경계
  근처에서 약간의 바이어스 (PGD-MPPI `pgd_mppi.py` 117-119행은
  `noise = sampled_controls - mu[None]`로 이걸 정확히 보정한다. 비교해 볼 것).

---

## 4. 핵심 하이퍼파라미터의 이론적 의미

### 4.1 λ — 온도

**세 가지 얼굴** (모두 같은 λ):

1. **soft-min 보간자** (§1.2): λ→0이면 best-sample copy, λ→∞이면 전 샘플 평균.
2. **KL 페널티 세기** (§1.2 변분 등식): λ가 클수록 "기준 분포에서 멀어지지
   마라"가 강함 → 보수적 업데이트.
3. **제어 비용 스케일** (§3.2): 유도상 R = λΣ⁻¹ — λ를 키우는 것은 제어
   페널티를 키우는 것과 이론적으로 등가.

**극한 유도** (연습문제 1에서 직접):

```
λ → 0:   w_k → 1{k = argmin S}     (탐욕 — 노이즈 한 방에 휘둘림, ESS→1)
λ → ∞:   w_k → 1/K                 (정보 무시 — U가 거의 안 움직임, ESS→K)
```

**실무 신호는 ESS다** (§5). 이 repo의 `AdaptiveTemperature`
(`mppi_controller/controllers/mppi/adaptive_temperature.py`)는 비례 제어로
ESS를 목표 비율에 붙인다:

```python
ess = 1.0 / np.sum(weights**2)
ess_error = ess/K - self.target_ess_ratio       # 기본 목표 0.5
delta_lambda = -self.adaptation_rate * ess_error * self.lambda_
# ESS 낮음(집중) → λ 증가(평탄화/탐색), ESS 높음(균등) → λ 감소(집중/활용)
```

repo 기본값: `MPPIParams.lambda_ = 1.0` (`mppi_params.py` 37행).
비용 스케일에 민감하므로 "λ 자체"보다 "S/λ의 분포 폭"이 본질이다 —
비용을 10배 스케일하면 λ도 10배 해야 같은 가중치.

### 4.2 σ (Σ) — 탐색 반경

노이즈 표준편차 σ (repo 기본 `sigma = [0.5, 0.5]`)의 역할:

- **탐색 범위**: q_Û가 덮는 영역 밖의 좋은 해는 **절대 발견 못 한다**
  (importance sampling은 지지집합 안에서만 보정 가능). 01 문서 연습문제 5의
  local minima 탈출 실험이 이것.
- **해의 매끄러움**: 큰 σ → 최종 U에 잔여 노이즈 → 채터링. i.i.d. 가우시안의
  백색 스펙트럼이 원인이고, 이것이 **샘플링 분포 축 변형들의 존재 이유**다
  (§7.1: LP/Colored/Smooth).
- **유도와의 결속**: §3.2에서 제어 비용 = λΣ⁻¹ — σ를 키우면 암묵적 제어
  페널티가 줄어든다. σ와 λ는 독립 손잡이가 아니다.
- **클리핑과의 상호작용**: σ가 u_max에 비해 크면 샘플 다수가 경계에 붙어
  분포가 찌그러진다 (`RectifiedGaussianSampler`가 `sampling.py`에 있는 이유).

### 4.3 K vs N — 샘플 수와 호라이즌의 트레이드오프

계산 예산은 대략 `K × N × (rollout 스텝 비용)`으로 고정이라 치자.

```
K (샘플 수)                         N (호라이즌)
──────────────                      ──────────────
↑ 몬테카를로 분산 ↓ (∝ 1/√K 느낌)   ↑ 근시안 완화 (01 문서 §4.2 절벽 문제)
↑ 다중 모달 커버                    ↑ 탐색 공간 차원 N·nu 증가
                                       → 같은 K로 커버율 지수적 악화!
                                    ↑ 모델 오차 누적 (rollout 길수록 발산)
```

**차원의 저주가 여기서 재등장한다**: 상태 공간이 아니라 **제어 시퀀스 공간**
(N·nu 차원)에서. N=30, nu=2 → 60차원 공간을 K=1024개 점으로 덮는 것은
불가능하고, 실제로는 "U 주변 국소 개선"만 일어난다. 대응하는 변형:
- N을 줄이고 terminal value로 보상: **TD-MPPI** (`td_mppi.py`)
- 시퀀스를 저차원 파라미터화: **Spline-MPPI** (`spline_mppi.py`),
  **RF-MPPI** (`rf_mppi.py` — Hermite 스플라인 dual-space)
- 샘플 효율을 결정론적으로: **dsMPPI** (`deterministic_mppi.py` — K=64로
  Vanilla 동등), **TR-MPPI**의 Halton LCD (`tr_mppi.py`의 `HaltonLCDSampler`)

repo 성능 기준 (CLAUDE.md): K=1024, N=30에서 < 100ms.

---

## 5. ESS와 weight degeneracy

### 5.1 정의와 직관

```
ESS = 1 / Σ_k w_k²        (w는 정규화된 가중치)
```

`base_mppi.py`의 `_compute_ess()` (296-310행) 그대로. 극단값으로 감 잡기:

```
균등 w_k = 1/K          → ESS = 1/(K·(1/K)²) = K       "K개 전부 유효"
집중 w_1 = 1, 나머지 0  → ESS = 1                       "사실상 샘플 1개"
```

**왜 이 공식인가**: importance sampling 추정량 `Σw_k g(V^k)`의 분산이
대략 `Var_q*(g)/ESS`로 스케일된다 — 즉 ESS는 "i.i.d. 샘플 몇 개짜리
추정인가"의 환산치다. ESS = 12면 K = 1024를 뽑아도 12개짜리 정보.

### 5.2 Weight degeneracy — 언제, 왜 무너지나

softmax 가중치는 비용 **차이**에 지수적이다:

```
w_1/w_2 = exp( (S_2 - S_1)/λ )
```

비용 표준편차가 λ의 몇 배만 되어도 최상위 한두 샘플이 다 먹는다. 잘 터지는
상황:

1. **비용 스케일 급변**: 장애물 페널티(수천)가 추적 비용(수십)과 섞일 때 —
   충돌 안 한 샘플 몇 개에 가중치 올인. (repo의 shield/CBF 계열이 페널티
   대신 필터/장벽을 쓰는 이유 중 하나.)
2. **차원 증가** (N·nu 큼): 랜덤 샘플의 비용 분산 자체가 커진다.
3. **λ 너무 작음**: §4.1.

**증상 → 처방 매핑** (repo 안에서):

| 증상 | 처방 | repo 구현 |
|------|------|-----------|
| ESS 만성 저하 | λ 적응 | `AdaptiveTemperature`, Biased-MPPI `_adapt_lambda()` |
| 극단 비용 꼬리 | 가중치 함수 교체 | Log/Tsallis/ASR (§7.2) |
| 분포가 좋은 영역을 못 덮음 | 제안 분포 개선 | Biased/Flow/SG (§7.1) |
| 한 번의 업데이트가 과격 | 신뢰 영역 | TR-MPPI `KL_prop = ½‖Δμ/σ‖² ≤ δ` |

벤치마크 info dict에 `ess`가 항상 들어가는 이유 — MPPI 상태 진단의
1번 계기판이다.

---

## 6. 의사코드 ↔ base_mppi.py 대응표

의사코드 (Williams 2017, Algorithm 2 요약)와
`mppi_controller/controllers/mppi/base_mppi.py`의 `compute_control()`
(111-190행)을 한 줄씩 대응시킨다. **읽으면서 파일을 옆에 띄워 놓고 확인할 것.**

```
의사코드                                   base_mppi.py (CPU 경로)
────────────────────────────────────────  ─────────────────────────────────────────────
입력: x₀, 레퍼런스, 이전 해 U              compute_control(state, reference_trajectory)
                                           self.U : (N, nu), 이전 스텝에서 시프트된 warm start
for k = 1..K:
  ε^k ~ N(0, Σ)                            L138  noise = self.noise_sampler.sample(...)
                                                 → GaussianSampler (sampling.py 38행), (K,N,nu) 한 번에
  V^k = U + ε^k                            L141  sampled_controls = self.U + noise   (브로드캐스트)
  V^k = clip(V^k, u_min, u_max)            L144-145  np.clip(...)     ← rollout "전" 클리핑 (01 §5.3)
  x^k = rollout(x₀, V^k)                   L148  self.dynamics_wrapper.rollout(state, sampled_controls)
                                                 → (K, N+1, nx), for문은 시간축만 (K는 벡터화)
  S_k = Σ l(x,v) + l_f                     L151-153  self.cost_function.compute_cost(...)
                                                 → CompositeMPPICost = StateTracking+Terminal+ControlEffort
w = softmax(-S/λ)                          L156  self._compute_weights(costs, self.params.lambda_)
  (수치 안정: S ← S - min S)               L289-292  min_cost 빼고 exp — §3.7에서 동일성 확인
U ← U + Σ_k w_k ε^k                        L160-161  weighted_noise = np.sum(weights[:,None,None]*noise, 0)
                                                 self.U = self.U + weighted_noise
U ← clip(U)                                L164-165
u_apply = U[0]                             L172  optimal_control = self.U[0, :]   ※ 아래 주의
U ← shift(U); U[-1] = 0                    L168-169  np.roll(self.U, -1, axis=0); self.U[-1,:] = 0
                                                 (receding horizon, 01 문서 §4.1)
진단: ESS = 1/Σw²                          L175, L296-310  _compute_ess()
info dict 반환                             L178-187  sample_trajectories / sample_weights /
                                                 best_trajectory / temperature / ess / num_samples
```

**읽을 때 확인할 미묘한 포인트 3개**:

1. **시프트 순서**: repo는 시프트를 **먼저** 하고 `U[0]`을 반환한다
   (168→172행). 즉 반환되는 것은 시프트 후의 첫 원소 = 원래 U의 **u_1**이
   아니라… 직접 따져보라 (연습문제 4). PGD-MPPI (`pgd_mppi.py` 148-152행)는
   순서가 반대다(`optimal_control = U[0]` 후 roll) — 두 컨벤션의 차이가
   실제 적용 제어에 어떤 영향을 주는지 비교해 볼 것.
2. **업데이트에 쓰는 ε**: clip 후 재계산 안 함 (§3.7 셋째 항목).
3. **GPU 경로** (`_compute_control_gpu`, 192-265행): 같은 수식을 torch로 —
   대응표의 각 줄이 어디로 갔는지 찾아보면 좋은 복습이 된다.

서브클래스 확장 포인트 (CLAUDE.md 인터페이스 규칙): 변형들은 주로
`_compute_weights()` 오버라이드 (가중치 축) 또는 `compute_control()` 전체
교체 (업데이트 축), 혹은 `noise_sampler` 주입 (샘플링 축)으로 만든다 —
이 세 주입점이 §7 분류 축과 정확히 일치한다.

---

## 7. 변형 분류 체계 — 43개를 5+1개 축으로

43개 변형을 외우는 것은 무의미하다. **"Vanilla 파이프라인의 어느 단계를
바꾸는가"**로 분류하면 5+1개 축으로 정리된다:

```
        [샘플링 q_Û]──▶[rollout f]──▶[비용 S]──▶[가중치 w(S)]──▶[업데이트 U←]
             │              │            │            │               │
         축1 샘플링      축5 학습/모델  축4 비용/안전  축2 가중치      축3 업데이트
             └──────────────────┬───────────────────────────────────┘
                          축6 (횡단): 강건성/피드백 — 루프 바깥을 감싼다
```

각 축에서 대표 2-3개만 심화하고, 나머지는
[docs/MPPI_THEORY.md](../MPPI_THEORY.md) 해당 절 링크로 대신한다.

### 7.1 축 1 — 샘플링 분포를 바꾼다 (ε 또는 q_Û 자체)

이론적 근거: importance sampling은 **제안 분포가 q*에 가까울수록 분산이
작다** (§5). i.i.d. 백색 가우시안은 q*와 가장 먼 "정보 없는" 선택이다.

| 변형 | 무엇으로 바꾸나 | 파일 |
|------|----------------|------|
| LP-MPPI | Butterworth 저역통과 노이즈 | `lp_mppi.py`, `sampling.py`의 `LowPassSampler` |
| Colored-noise | OU 과정 (시간 상관) | `sampling.py`의 `ColoredNoiseSampler` |
| **Biased-MPPI** | 보조 정책 혼합 | `biased_mppi.py`, `ancillary_policies.py` |
| dsMPPI | Halton/Sobol/sigma point (결정론) | `deterministic_mppi.py` |
| Flow / Diffusion / SG | 학습된 생성 모델 | `flow_mppi.py`, `diffusion_mppi.py`, `score_guided_mppi.py` |

**심화 1 — LP-MPPI** ([MPPI_THEORY.md §22](../MPPI_THEORY.md)): 백색 노이즈의
파워 스펙트럼은 평탄 → 샘플 제어가 고주파로 진동 → 가중 평균해도 잔여 채터링.
LPF를 노이즈에 걸면 **탐색을 "액추에이터가 따라갈 수 있는 부분공간"으로
제한**하는 것과 같다. 같은 σ 총량으로 저주파에 몰아주니 유효 탐색은 오히려
늘 수 있다. 구현 핵심은 `sosfilt(sos, noise, axis=1)` 한 줄 — (K,N,nu)
전체를 시간축으로 일괄 필터.

**심화 2 — Biased-MPPI** ([MPPI_THEORY.md §23](../MPPI_THEORY.md)): 샘플의
일부를 "손으로 만든 좋은 후보" (pure pursuit, braking, 이전 해…) 주변에서
뽑는다. 유도의 아름다운 포인트: 혼합 제안 분포 q_mix로 importance weight를
쓰면 밀도비의 q_mix가 **분자·분모에서 소거**되어 가중치는 여전히
`softmax(-S/λ)` 그대로다 — 코드가 거의 안 바뀌면서 제안 분포만 좋아진다.
비상 정지 정책이 항상 후보에 있으므로 안전 관점의 부수 효과도 있다.

### 7.2 축 2 — 가중치 함수를 바꾼다 (`_compute_weights` 오버라이드)

이론적 근거: softmax = "q*를 지수족으로 표현"인데, 지수 꼬리는 outlier
비용에 취약 (§5.2). 왜곡 함수를 바꾸면 **리스크 선호**가 바뀐다.

| 변형 | w(S)의 형태 | 파일 |
|------|------------|------|
| Log-MPPI | log 스케일 완화 | `log_mppi.py` |
| **Tsallis-MPPI** | q-지수 (멱법칙 꼬리) | `tsallis` 계열, [MPPI_THEORY.md §5](../MPPI_THEORY.md) |
| **Risk-Aware (CVaR)** | 상위 α-분위만 사용 | `risk_aware_mppi.py` |
| ASR-MPPI | 스펙트럴 리스크 (sigmoid 왜곡) + ESS 적응 | `spectral_risk_mppi.py` |

**심화 — CVaR과 스펙트럴 리스크**: 기대 비용 대신
`CVaR_α(S) = E[S | S ≥ VaR_α]` (나쁜 꼬리의 평균)를 최소화하면 위험 회피
제어가 된다. 샘플 구현은 단순하다 — 비용 정렬 후 좋은 쪽 α-분위 샘플에만
가중치를 준다 (경질 절단). ASR은 이 절단을 매끄러운 왜곡 함수 φ'(q)로
일반화 — CVaR은 φ가 계단 함수인 특수 경우이며, 매끄러운 φ가 ESS를 덜
망가뜨린다. **분류 관점의 교훈**: "가중치 축 변형은 전부 `_compute_weights()`
하나만 갈아 끼운다" — repo 인터페이스 규칙이 이 축을 위해 존재한다.

### 7.3 축 3 — 업데이트 규칙을 바꾼다 (한 번의 가중 평균 → 그 이상)

이론적 근거: §8에서 자세히 — MPPI 업데이트는 **스텝 크기 1의 전처리 경사
하강 1스텝**으로 볼 수 있고, 그러면 최적화 이론의 도구 전부(다중 스텝,
곡률, 신뢰 영역, 라인 서치)를 가져올 수 있다.

| 변형 | 무엇을 추가하나 | 파일 |
|------|----------------|------|
| DIAL-MPPI | 다중 반복 + 노이즈 어닐링 | `dial_mppi.py` |
| **PGD-MPPI** | n회 경사 스텝 + step size + 공분산 전처리 | `pgd_mppi.py` |
| **TR-MPPI** | KL 신뢰 영역으로 Δμ 클리핑 | `tr_mppi.py` |
| GN-MPPI | 가우스-뉴턴 2차 스텝 + 라인 서치 | `gn_mppi.py` |
| CMA-MPPI | 공분산 행렬 적응 (CMA-ES식) | `cma_mppi.py` |
| SVMPC / SVG | 파티클 집합을 SVGD로 밀기 (다중 모달 유지) | `stein_variational_mppi.py`, `svg_mppi.py` |

**심화 1 — PGD-MPPI** ([MPPI_THEORY.md §38](../MPPI_THEORY.md)): Vanilla의
`U += Σw_kε_k`에서 `Σw_kε_k ≡ g̃`를 "전처리된 경사"로 명명하고,
`mu = mu + step_size * grad` (`pgd_mppi.py` 137행)로 일반화. step_size=1,
n_grad_steps=1이면 Vanilla와 동일 — **Vanilla가 특수 경우임을 코드로 확인할
수 있는** 가장 좋은 변형. `normalize_gradient` 옵션은 `K/ESS`를 곱해
degeneracy로 죽은 경사를 되살린다 (§5와 연결).

**심화 2 — TR-MPPI** ([MPPI_THEORY.md §39](../MPPI_THEORY.md)): 업데이트가
과격하면 다음 스텝의 제안 분포가 좋은 영역을 이탈한다 (importance sampling
신뢰도 붕괴). 해법은 자연스럽게 신뢰 영역:
`KL(q_new‖q_old) = ½‖Δμ/σ‖² ≤ δ`, 넘으면 `Δμ *= sqrt(δ/KL)` 축소
(`tr_mppi.py` docstring). TRPO(강화학습)와 정확히 같은 정신이다.

### 7.4 축 4 — 비용 구조를 바꾼다 (S에 무엇을 넣는가)

이론적 근거: MPPI에서 제약은 비용이다 (01 문서 §5.1). "어떤 함수를 페널티로
넣으면 soft 제약이 hard처럼 행동하는가"가 이 축의 질문.

| 변형 | 비용에 추가되는 것 | 파일 |
|------|-------------------|------|
| **CBF-MPPI 계열** | 제어 장벽 함수 위반 페널티/필터 | `cbf_cost.py`, `cbf_mppi.py`, `shield_mppi.py` |
| **DBaS-MPPI** | 장벽 상태를 동역학에 증강 | `dbas_mppi.py` |
| DRPA-MPPI | 반발 포텐셜 (local minima 탈출) | `drpa_mppi.py` |
| DualGuard | HJ 안전 가치함수 | `dualguard_mppi.py` |
| TD-MPPI | 학습된 terminal value V(x_N) | `td_mppi.py`, `td_value.py` |
| MPCC | contouring 비용 (경로 파라미터화) | `mpcc_cost.py` |

**심화 — DBaS-MPPI** ([MPPI_THEORY.md §18](../MPPI_THEORY.md)): 단순 페널티
`ρ·1{충돌}`은 위반 직전까지 그래디언트(비용 차이) 신호가 없다. 장벽 함수
`β = 1/h(x)` (h: 안전 여유)를 **상태로 증강**해 rollout하면, 경계에 접근하는
동안 β가 폭증하며 비용에 연속적으로 반영된다 — "얼마나 위험해지고 있는가"를
샘플 가중치가 미리 감지. 안전 계열 전반의 이론은
docs/SAFETY_THEORY.md 참조 (본 문서에서는 다루지 않음).

### 7.5 축 5 — 학습과 결합한다 (f, S, q, U_init을 데이터로)

| 변형 | 무엇을 학습하나 | 파일 |
|------|----------------|------|
| BNN / Koopman / World-Model / Latent | 동역학 f (불확실성/선형화/잠재공간) | `bnn_mppi.py`, `koopman_mppi.py`, `world_model_mppi` 계열, `latent_mppi.py` |
| T-MPPI | 초기해 U_init (transformer) | `transformer_mppi.py` |
| **Residual-MPPI** | 사전 정책 + 잔차 δu만 MPPI | `residual_mppi.py` |
| TD-MPPI | terminal value | `td_mppi.py` |
| **Step-MPPI** | proposal 분포 자체 (DPC, 단일 스텝) | `step_mppi.py` |
| RF-MPPI | 스플라인 dual-space 파라미터화 | `rf_mppi.py` |

**심화 1 — Residual-MPPI** ([MPPI_THEORY.md §24](../MPPI_THEORY.md)):
샘플링을 `u = π_prior(x) + δu` 주변에서 수행. §7.1 Biased와 동기가 같지만
(제안 분포를 q*에 가깝게), 정책을 혼합 성분이 아니라 **평균 이동**으로 쓰고
δu 공간에서 최적화 + KL 페널티로 prior 근처를 유지한다. "사전 지식은 크게,
탐색은 작게"의 분업.

**심화 2 — Step-MPPI** ([MPPI_THEORY.md §41](../MPPI_THEORY.md)): 극단으로
가면 — proposal이 충분히 좋으면 반복이 필요 없다. DPC(Differentiable
Predictive Control)로 proposal 분포를 오프라인 학습하고 온라인에서는 샘플링
+ 가중 평균 **1회**만. §4.3의 K/N 트레이드오프를 "학습으로 계산을 선불"하는
쪽으로 푸는 노선이며, amortized optimization의 제어판이다.

### 7.6 축 6 (횡단) — 강건성/피드백으로 루프를 감싼다

01 문서 §3.3에서 본 것들: **Tube-MPPI** (`tube_mppi.py` — 명목 MPPI +
ancillary LQR식 피드백), **F-MPPI** (`feedback_mppi.py` — Riccati 게인으로
solve 재사용, 75% 계산 절감), **Robust-MPPI** (`robust_mppi.py` — 피드백을
샘플링 루프 **안**으로), **PR-MPPI** (`parameter_robust_mppi.py` — 파라미터
belief에 대한 기대/worst-case 비용), **C2U** (`c2u_mppi.py` — unscented
불확실성 전파). 이 축은 "MPPI가 만든 계획과 실제 상태의 괴리"를 다루므로
다른 다섯 축과 직교한다 — 예: Tube + LP-MPPI처럼 조합 가능.

### 7.7 분류표 한 장 요약

```
축              바꾸는 것        대표 (심화)              나머지 → MPPI_THEORY.md
─────────────────────────────────────────────────────────────────────────────
1 샘플링        ε, q_Û          LP, Biased               Colored/ds/Flow/Diffusion/SG/Smooth/Spline
2 가중치        w(S)            CVaR, ASR                Log/Tsallis
3 업데이트      U← 규칙          PGD, TR                  DIAL/GN/CMA/SVMPC/SVG/CSC/pi(투영)
4 비용/안전     S               DBaS                     CBF계열/Shield/DRPA/DualGuard/MPCC
5 학습          f,S,q,U_init    Residual, Step           BNN/Koopman/WM/Latent/T/TD/RF
6 강건/피드백   루프 바깥        Tube(01 문서)             F/Robust/PR/C2U/Uncertainty
```

새 논문을 읽을 때도 이 축에 꽂아 넣어 보라 — 대부분 한두 축의 조합이다.

---

## 8. 알려진 한계와 연구 전선 (2026)

### 8.1 정직한 한계 목록

1. **가우시안 moment matching의 정보 손실** (§3.3): q*가 다중 모달이어도
   평균 하나로 요약 → 두 모드의 평균 = 장애물 정중앙이라는 최악의 해가
   가능. (SVMPC/Flow/Diffusion이 공격하는 지점.)
2. **고차원 weight degeneracy** (§5.2): N·nu가 크면 ESS 붕괴가 구조적.
3. **soft 제약의 비보장** (01 문서 §5.1): 페널티는 확률적 회피일 뿐.
   안전 보증은 외부 장치(CBF 필터, gatekeeper, HJ)가 필요.
4. **안정성/수렴 이론 부재**: MPC의 정리들 (01 문서 §4.4)에 대응하는 일반
   결과가 아직 없다. λ, K 유한에서의 성능 보장은 열린 문제.
5. **모델 오차 민감성**: rollout이 전부이므로 f가 틀리면 다 틀린다
   (축 6과 축 5가 대응).

### 8.2 재해석 전선: MPPI = preconditioned gradient descent

이 repo가 40-41번째로 구현한 노선 (PGD-MPPI arXiv:2603.24489, TR-MPPI
arXiv:2605.07801)의 핵심 관찰. 부드러워진 비용(= free energy)

```
J_σ(U) = -λ log E_{ε~N(0,Σ)}[ e^{-S(U+ε)/λ} ]
```

의 그래디언트를 계산해 보면 (log-derivative trick — 연습문제 2에서 유도):

```
∇_U J_σ(U) = -Σ⁻¹ · E_w[ε]      where   E_w[ε] = Σ_k w_k ε_k  (softmax 가중 평균!)
```

즉:

```
MPPI 업데이트  U ← U + Σ_k w_k ε_k  =  U - Σ·∇J_σ(U)
             = "공분산 Σ로 전처리(preconditioning)된 경사 하강, 스텝 크기 1, 1회"
```

이 관점이 열어주는 것들 (repo 구현과 대응):

| 최적화 이론의 도구 | MPPI 이식 | repo |
|-------------------|-----------|------|
| step size / 다중 스텝 | PGD-MPPI `step_size`, `n_grad_steps` | `pgd_mppi.py` |
| preconditioner 적응 | Gibbs-tilted 경험 공분산 | `pgd_mppi.py` `_adapt_covariance()` |
| 신뢰 영역 | KL ≤ δ 투영 | `tr_mppi.py` |
| 곡률 (2차) | 가우스-뉴턴 + 라인 서치 | `gn_mppi.py` |
| 준-몬테카를로 | Halton LCD 샘플 | `tr_mppi.py` `HaltonLCDSampler` |

그리고 σ의 새 해석: **σ는 탐색 반경이자 스무딩 커널 폭** — J_σ는 S를
가우시안으로 컨볼브한 것이므로, σ가 크면 지형이 뭉개져 local minima가
사라지는 대신 최적점 위치가 밀린다 (smoothing bias). DIAL의 어닐링 = σ를
크게 시작해 줄여가는 continuation method로 정확히 재해석된다.

### 8.3 그 외 전선 (2026 기준, repo 구현 관점)

- **학습 proposal / amortization**: Step-MPPI (DPC), T-MPPI, Flow/Diffusion —
  "온라인 계산을 오프라인 학습으로 선불" 노선. §7.5.
- **파라미터화 전환**: RF-MPPI의 dual-space Hermite 스플라인 — 탐색 공간
  자체를 저차원·매끄럽게. few-sample 영역 (K 수십)이 목표.
- **안전 보증의 정식화**: DualGuard(HJ), C-MPPI(contingency), Shield 계열 —
  MPC의 recursive feasibility (01 문서 §4.3)에 대응하는 샘플링 이론 만들기.
- **결정론화**: dsMPPI, Halton LCD — 분산 자체를 없애 재현성/인증 친화적으로.

---

## 9. 연습문제

**문제 1 (λ 극한 유도).**
`w_k = e^{-S_k/λ}/Σ_j e^{-S_j/λ}`에 대해 다음을 엄밀히 보여라.
(a) λ→0⁺에서 w → argmin 지시함수 (유일 최소 가정. 힌트: S_k - S_min ≥ Δ > 0인
항은 e^{-Δ/λ} → 0). (b) λ→∞에서 w_k → 1/K이고, 1차 보정이
`w_k ≈ (1/K)[1 - (S_k - S̄)/λ]`임을 테일러 전개로 보여라 (S̄: 표본 평균).
(c) (b)를 업데이트 식에 대입하면 λ→∞에서 `ΔU ≈ -(1/λ)·Cov(S, ε)`
형태가 됨을 보이고, 이것이 "약한 경사 신호"임을 §8.2와 연결해 해석하라.

**문제 2 (smoothed gradient 유도 — §8.2의 빈칸 채우기).**
`J_σ(U) = -λ log E_{ε~N(0,Σ)}[e^{-S(U+ε)/λ}]`에 대해
`∇_U J_σ(U) = -Σ⁻¹ E_w[ε]`를 유도하라. 힌트: 기댓값을 적분
`∫ N(ε; 0, Σ) e^{-S(U+ε)/λ} dε`로 쓰고 치환 `v = U + ε` 후 U에 대해 미분 —
미분이 S가 아니라 **가우시안 밀도**에 걸리게 만들면
`∇_U N(v-U; 0,Σ) = Σ⁻¹(v-U)·N(...)`에서 결과가 나온다 (S 미분 불필요!).
이 유도가 "S가 미분 불가능해도 J_σ는 미분 가능"을 어떻게 보장하는지 설명하라.

**문제 3 (1D 이중우물 수치 실험).**
정적 최적화로 MPPI 커널을 체험한다 (동역학 없이 가중 평균 업데이트만).

```python
import numpy as np
S = lambda u: (u**2 - 1.0)**2 + 0.3*u          # 이중우물: u≈-1(전역), u≈+1(지역)
u, sigma, lam, K = 1.0, 0.5, 0.1, 256          # 나쁜 우물에서 시작
rng = np.random.default_rng(0)
for it in range(50):
    eps = rng.normal(0, sigma, K)
    costs = S(u + eps)
    w = np.exp(-(costs - costs.min())/lam); w /= w.sum()
    u = u + w @ eps
    print(f"it={it:2d}  u={u:+.3f}  ESS={1/(w**2).sum():6.1f}")
```

(a) σ = 0.5, 0.2, 1.5에서 각각 실행 — 어느 경우 전역 우물(-1 근처)로
탈출하는가? §4.2/§8.2의 "σ = 스무딩 폭" 해석으로 설명하라.
(b) λ = 0.001과 λ = 10.0에서 ESS와 수렴 속도를 관찰하고 §4.1과 대조하라.
(c) 업데이트를 `u += 0.3 * (w @ eps)`로 바꿔 (PGD step_size) 진동이
줄어드는지 확인하라.

**문제 4 (repo 코드 정밀 독해).**
`base_mppi.py` 158-172행: 업데이트 → 클리핑 → **시프트 → U[0] 반환** 순서다.
(a) 시프트 후의 `U[0]`은 시프트 전의 어떤 원소인가? 이 컨벤션에서 "지금
적용되는 제어"는 최적화 관점에서 몇 스텝째 제어인가?
(b) `pgd_mppi.py` 148-152행은 반환 후 시프트한다. 두 방식이 실제 적용
제어에서 어떻게 다른지, 어느 쪽이 Williams 논문의 `u_0` 적용과 일치하는지
논하라. (c) `self.U[-1,:] = 0` 대신 `self.U[-1] = self.U[-2]` (복제)로
바꾸면 어떤 상황에서 더 나을지 추론하고, 원형 궤적 추적으로 실험해 보라:

```bash
python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory circle --no-plot
```

**문제 5 (importance 보정항).**
§3.4의 보정 비용 `S̃ = S + λΣ_t û_tᵀΣ⁻¹ε_t (+ const)`에서:
(a) 교차항 `λûᵀΣ⁻¹ε`이 하는 일을 말로 설명하라 (힌트: 현재 계획 û와 같은
방향의 ε은 비용이 **가산**된다 — 어떤 바이어스를 상쇄하는가?).
(b) û = 0 (첫 스텝, reset 직후)이면 S̃ = S가 됨을 확인하라.
(c) `base_mppi.py`의 `_compute_weights`를 서브클래스에서 오버라이드해
보정항을 추가한 `CorrectedMPPIController`를 작성하고 (CLAUDE.md의
`_compute_weights()` 오버라이드 규칙), 원형 궤적 RMSE로 Vanilla와 비교하라.
차이가 크지 않다면 왜 그런지 (ControlEffortCost가 이미 유사 역할을 함, §3.7)
논하라.

---

## 10. 부록 — 더 공부하기 위한 자료

> 기존 "추천 논문" 절을 확장한 자습(self-study) 부록. 외부 링크는 2026-07
> 기준 존재/접근을 확인한 것만 실었다. MPC 쪽 배경 자료는
> [01_MPC_FUNDAMENTALS.md §9](01_MPC_FUNDAMENTALS.md) 부록이 담당하고,
> 변형별 상세 레퍼런스는 [MPPI_THEORY.md](../MPPI_THEORY.md) 각 절에 있다.

### 10.1 주석 달린 핵심 레퍼런스

읽는 순서 추천: ①②(유도 원전) → ⑤(조감) → 관심 분기 (다중 모달이면 ⑦⑧,
스무딩이면 ⑨, 실전 배포면 ⑪⑫).

1. **Williams, Aldrich, Theodorou, "Model Predictive Path Integral Control:
   From Theory to Parallel Computation," *J. Guidance, Control, and
   Dynamics*, 2017.**
   §3 유도의 원전 — importance sampling 보정과 GPU 병렬 구현을 함께 다룬다.
   §3.6의 유도 지도를 손에 들고 원문의 수식 번호와 대조하며 읽는 것이
   이 문서의 최종 목표다.

2. **Williams et al., "Information-Theoretic Model Predictive Control:
   Theory and Applications to Autonomous Driving," *IEEE T-RO*, 2018 —
   [arXiv:1707.02342](https://arxiv.org/abs/1707.02342).**
   free energy 하한(§1, §3.2)의 원전이자 AutoRally 실차 검증 논문.
   유도가 ①보다 정돈되어 있어, 둘 중 하나만 정독한다면 이쪽을 권한다.

3. **Kappen, "Path integrals and symmetry breaking for optimal control
   theory," *J. Stat. Mech.*, 2005 —
   [arXiv:physics/0505066](https://arxiv.org/abs/physics/0505066).**
   §2의 원전 — log 변환(Cole-Hopf)으로 HJB를 선형화하고 경로적분으로 푼다.
   물리 배경이 있다면 MPPI 계보의 뿌리를 보는 재미가 있고, 없어도 §2를
   읽었다면 도입부와 예제(symmetry breaking = 다중 모달)는 따라갈 수 있다.

4. **Theodorou, Buchli, Schaal, "A Generalized Path Integral Control
   Approach to Reinforcement Learning" (PI²), *JMLR*, 2010.**
   경로적분 제어를 정책 개선(policy improvement)으로 확장한 다리 논문.
   Kappen(이론)과 Williams(알고리즘) 사이의 역사적 연결 고리가 궁금할 때 읽는다.

5. **Kazim, Hong, Kim, Kim, "Recent Advances in Path Integral Control for
   Trajectory Optimization," *Annual Reviews in Control*, 2024 —
   [arXiv:2309.12566](https://arxiv.org/abs/2309.12566).**
   CEM·MPPI·피드백 파라미터화까지 경로적분 계열 전체를 정리한 서베이.
   이 문서 §7의 분류 축을 학계 표준 용어와 맞춰볼 때 — 새 논문을 읽기 전
   좌표계를 세우는 용도로 가장 좋다.

6. **Todorov, "Linearly-solvable Markov decision problems," NeurIPS 2006.**
   이산 MDP에서 같은 선형화(KL 제어)가 성립함을 보인 병렬 이론.
   §1의 변분 등식이 연속/이산을 가리지 않는 구조임을 확인하고 싶을 때 읽는다.

7. **Lambert et al., "Stein Variational Model Predictive Control," CoRL 2020 —
   [arXiv:2011.07641](https://arxiv.org/abs/2011.07641).**
   MPC를 베이지안 추론으로 캐스팅하고 SVGD로 다중 모달 사후분포를 유지한다.
   §8.1-1 (가우시안 moment matching의 정보 손실)을 체감한 뒤 읽으면
   `stein_variational_mppi.py`가 왜 필요한지 명확해진다.

8. **Honda et al., "Stein Variational Guided Model Predictive Path Integral
   Control," ICRA 2024 — [arXiv:2309.11040](https://arxiv.org/abs/2309.11040).**
   SVGD로 제안 분포를 최빈 모드 근처로 유도한 뒤 MPPI를 수행 — repo
   SVG-MPPI(`svg_mppi.py`)의 원전. ⑦의 실용화 버전으로, 고속 주행 실험까지
   포함한다.

9. **Kim et al., "Smooth Model Predictive Path Integral Control without
   Smoothing," *RA-L*, 2022 —
   [arXiv:2112.09988](https://arxiv.org/abs/2112.09988).**
   제어 대신 제어 증분(ΔU)을 샘플링하는 input lifting으로 채터링을 제거 —
   repo Smooth-MPPI의 원전. §4.2의 "σ와 매끄러움" 트레이드오프를 겪은 뒤
   읽으면 좋다 (LP-MPPI [arXiv:2503.11717](https://arxiv.org/abs/2503.11717)와
   접근을 비교해 볼 것).

10. **Trevisan, Alonso-Mora, "Biased-MPPI," *RA-L*, 2024 —
    [arXiv:2401.09241](https://arxiv.org/abs/2401.09241).**
    보조 정책 혼합 제안 분포에서 importance weight의 분모가 소거되는 §7.1의
    정리를 담은 논문. 짧고 자기완결적이라 "MPPI 유도를 이해했는지" 스스로
    시험하는 첫 논문으로 적합하다.

11. **Vlahov, Gibson, Gandhi, Theodorou, "MPPI-Generic: A CUDA Library for
    Stochastic Trajectory Optimization," 2024 —
    [arXiv:2409.07563](https://arxiv.org/abs/2409.07563).**
    Vanilla/Tube/Robust MPPI를 템플릿화한 C++/CUDA 라이브러리 논문 —
    GPU 커널 수준의 성능 공학을 다룬다. 이 repo의 `_compute_control_gpu()`를
    본 뒤 "진짜 프로덕션 GPU 구현"이 궁금할 때 읽는다.

12. **"Model Predictive Control via Probabilistic Inference: A Tutorial and
    Survey" — [arXiv:2511.08019](https://arxiv.org/abs/2511.08019).**
    MPPI·CEM·변분 추론 계열 MPC를 "확률 추론으로서의 제어" 관점에서 통합
    정리한 튜토리얼. §1-3을 다 소화한 뒤 지식을 재배열하는 마무리 독서로 좋다.

**배경 보강**: Botev et al., "The Cross-Entropy Method for Optimization"
(2013) — CEM은 MPPI의 사촌 (elite 평균 vs softmax 평균). Yin et al.,
"Trajectory Distribution Control ... Tsallis" (2021) — 가중치 축(§7.2)의 원전.

### 10.2 최근 연구 동향 (2024–2026)

이 repo의 변형 43종이 이미 따라간 계보이기도 하다 — 각 동향에 repo
구현과 대표 논문을 함께 적는다.

1. **스무딩 계열의 다변화 — 시간 도메인에서 주파수/투영/파라미터화로.**
   input lifting (SMPPI, [arXiv:2112.09988](https://arxiv.org/abs/2112.09988))
   → 주파수 도메인 LPF (LP-MPPI,
   [arXiv:2503.11717](https://arxiv.org/abs/2503.11717)) → QP 투영으로
   jerk/snap hard 보장 (π-MPPI,
   [arXiv:2504.10962](https://arxiv.org/abs/2504.10962)).
   repo: `smooth_mppi.py`, `lp_mppi.py`, `projection_mppi.py` (§7.1, §7.3).

2. **학습 제안 분포 / amortization.**
   백색 가우시안 대신 "좋은 후보"를 학습으로 만든다 — Transformer 초기화
   (T-MPPI, [arXiv:2412.17118](https://arxiv.org/abs/2412.17118)), 샘플링
   분포 자체의 Stein 최적화
   ([arXiv:2511.02015](https://arxiv.org/abs/2511.02015)), proposal을
   오프라인 학습해 단일 스텝으로 (Step-MPPI, arXiv:2604.01539).
   repo: `transformer_mppi.py`, `flow_mppi.py`, `score_guided_mppi.py`,
   `step_mppi.py` (§7.5).

3. **2차 정보·최적화 이론과의 합류 (§8.2의 전선).**
   MPPI = 전처리 경사 하강 재해석 (PGD-MPPI, arXiv:2603.24489) → KL 신뢰
   영역 (TR-MPPI, arXiv:2605.07801) → 가우스-뉴턴 가속 (GN-MPPI,
   [arXiv:2512.04579](https://arxiv.org/abs/2512.04579)) → 이차 모델 근사로
   분산 축소 ([arXiv:2602.03639](https://arxiv.org/abs/2602.03639)).
   repo: `pgd_mppi.py`, `tr_mppi.py`, `gn_mppi.py` (§7.3).

4. **결정론적/저분산 샘플링.**
   Halton/Sobol/sigma point로 몬테카를로 분산 자체를 제거하는 노선 —
   C-Uniform 궤적 샘플러
   ([arXiv:2503.05819](https://arxiv.org/abs/2503.05819)), STL 비용의
   결정론적 경로적분 최적화
   ([arXiv:2503.01476](https://arxiv.org/abs/2503.01476)).
   repo: `deterministic_mppi.py`(dsMPPI), TR-MPPI의 `HaltonLCDSampler` (§4.3).

5. **GPU/시뮬레이터 통합 프레임워크.**
   MPPI-Generic ([arXiv:2409.07563](https://arxiv.org/abs/2409.07563),
   C++/CUDA 템플릿), Isaac Gym 물리 시뮬레이터를 동역학 모델로 직접 쓰는
   mppi-isaac (ICRA 2023 워크숍,
   [github](https://github.com/tud-amr/mppi-isaac)), 매니퓰레이터 실시간
   반응 제어 STORM ([arXiv:2104.13542](https://arxiv.org/abs/2104.13542)).
   "동역학을 코딩하지 않고 시뮬레이터를 f로 쓴다"가 공통 아이디어 —
   §1.2 성질 2의 극단이다.

6. **안전 결합과 필드 배포.**
   CBF/reach-avoid를 MPPI 비용·필터로 통합
   ([arXiv:2407.13693](https://arxiv.org/abs/2407.13693)), chance constraint
   + 안전 실드 ([arXiv:2408.00494](https://arxiv.org/abs/2408.00494)),
   비정형 환경 반발 포텐셜 내비게이션 (DRPA-MPPI,
   [arXiv:2503.20134](https://arxiv.org/abs/2503.20134)), 오프로드 확률
   하이브리드 시스템의 risk-aware MPPI
   ([arXiv:2411.09198](https://arxiv.org/abs/2411.09198)).
   repo: `shield_mppi.py`, `dualguard_mppi.py`, `drpa_mppi.py`,
   `risk_aware_mppi.py` (§7.4).

### 10.3 오픈소스 생태계

| 이름 | 링크 | 언어 | 특징 | 이 repo와의 관계 |
|------|------|------|------|-----------------|
| pytorch_mppi | [github.com/UM-ARM-Lab/pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi) | Python (PyTorch) | 가장 널리 쓰이는 경량 MPPI + SMPPI/KMPPI 변형 | `base_mppi.py`와 가장 비슷한 급 — 구현 대조 학습에 최적 |
| MPPI-Generic | [github.com/ACDSLab/MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) | C++/CUDA | 템플릿 기반 고성능, Vanilla/Tube/Robust 내장 (GaTech) | GPU 경로(`_compute_control_gpu`)의 프로덕션급 상위 호환 |
| mppi-isaac | [github.com/tud-amr/mppi-isaac](https://github.com/tud-amr/mppi-isaac) | Python | Isaac Gym rollout — 동역학 코딩 불필요, 접촉 과제 | `dynamics_wrapper.rollout()`을 물리 시뮬레이터로 치환한 형태 |
| STORM | [github.com/NVlabs/storm](https://github.com/NVlabs/storm) | Python (PyTorch) | 매니퓰레이터 joint-space MPPI, ~125Hz (NVIDIA) | 모바일 로봇 중심인 이 repo의 매니퓰레이터판 참고 구현 |
| AutoRally | [github.com/AutoRally/autorally](https://github.com/AutoRally/autorally) | C++/ROS | MPPI 원 실험 플랫폼 (1/5 스케일 오프로드 차량, GaTech) | §10.1-①② 논문의 실차 코드 — 역사적 원본 |
| pytorch_icem | [github.com/UM-ARM-Lab/pytorch_icem](https://github.com/UM-ARM-Lab/pytorch_icem) | Python (PyTorch) | iCEM (improved Cross-Entropy Method) 병렬 구현 | CEM vs MPPI (elite 평균 vs softmax) 비교 실험용 |
| mppi_playground | [github.com/kohonda/mppi_playground](https://github.com/kohonda/mppi_playground) | Python (PyTorch) | 시각화 중심 MPPI 놀이터 | 예제/벤치마크 구성이 이 repo `examples/`와 유사 — 교차 검증용 |
| smooth-mppi-pytorch | [github.com/tkkim-robot/smooth-mppi-pytorch](https://github.com/tkkim-robot/smooth-mppi-pytorch) | Python (PyTorch) | SMPPI 저자 구현 | `smooth_mppi.py`(ΔU 리프팅)의 원저자 코드와 대조 |
| CBFkit | [github.com/bardhh/cbfkit](https://github.com/bardhh/cbfkit) | Python/ROS2 (JAX) | CBF 안전 제어 툴킷, MPPI 통합 ([arXiv:2404.07158](https://arxiv.org/abs/2404.07158)) | repo 안전 계열의 직접 영감 — [docs/CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) 참조 |
| learning_mppi (이 repo) | `mppi_controller/` | Python (NumPy/PyTorch) | 43종 변형 + 22종 안전 제어 + 학습 모델, 단일 인터페이스 | 위 라이브러리들이 각각 1-3개 변형을 다루는 것과 달리 축 전체를 커버 |

### 10.4 더 공부하기 — 서베이·강의·튜토리얼

- **서베이/튜토리얼 논문**: Kazim et al.
  ([arXiv:2309.12566](https://arxiv.org/abs/2309.12566), §10.1-⑤) →
  확률 추론 관점 통합 튜토리얼
  ([arXiv:2511.08019](https://arxiv.org/abs/2511.08019)). 이 두 편이면
  2026년 기준 지형도가 완성된다.
- **영상**: AutoRally의 [MPPI 설명 영상](https://autorally.github.io/mppi-video/)
  — 원저자 그룹의 직관 설명. Tedrake *Underactuated Robotics*
  ([OCW](https://ocw.mit.edu/courses/6-832-underactuated-robotics-spring-2022/))
  의 궤적 최적화 장은 §4.3(제어 시퀀스 공간의 차원)의 배경.
- **코드로 배우기**: [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi)
  README의 최소 예제를 이 repo `base_mppi.py`와 나란히 놓고 §6 대응표를
  양쪽에 적용해 보라 — 같은 알고리즘의 두 구현을 대조하면 컨벤션(시프트
  순서, 클리핑 시점)의 차이가 선명해진다.
- **최적화 배경**: Boyd & Vandenberghe *Convex Optimization*
  ([무료 PDF](https://stanford.edu/~boyd/cvxbook/)) — §8.2의 "MPPI =
  전처리 경사 하강" 재해석을 제대로 즐기려면 경사법/신뢰영역의 기초가 필요하다.
  MPC 쪽 교재 목록은 [01 문서 §9.4](01_MPC_FUNDAMENTALS.md) 참조.

### 10.5 자주 궁금한 점 → 어디를 볼까

| 질문 | 이 repo에서 | 외부에서 |
|------|------------|----------|
| MPPI가 local minima에 빠지면? | §8.2 (σ = 스무딩 폭) + 01 문서 연습문제 5; DRPA(`drpa_mppi.py`), DIAL, SVMPC | DRPA-MPPI ([arXiv:2503.20134](https://arxiv.org/abs/2503.20134)), SVMPC ([arXiv:2011.07641](https://arxiv.org/abs/2011.07641)) |
| 제어가 덜덜 떨린다 (채터링) | §4.2 + §7.1; LP/Smooth/Colored (`lp_mppi.py`, `smooth_mppi.py`) | SMPPI ([arXiv:2112.09988](https://arxiv.org/abs/2112.09988)) |
| 제약을 hard로 걸고 싶으면? | 01 문서 §5.1 + pi-MPPI(`projection_mppi.py`), CSC-MPPI, CBF-QP 필터 | π-MPPI ([arXiv:2504.10962](https://arxiv.org/abs/2504.10962)) |
| ESS가 계속 낮게 나온다 | §5.2 증상→처방 표; `AdaptiveTemperature`, ASR/Tsallis | Kazim 서베이 §샘플링 전략 ([arXiv:2309.12566](https://arxiv.org/abs/2309.12566)) |
| λ, σ를 어떻게 튜닝하나? | §4.1-4.2 (λ·σ는 독립 손잡이가 아님) + 연습문제 3 | Williams T-RO 2018 ([arXiv:1707.02342](https://arxiv.org/abs/1707.02342)) 실험 절 |
| GPU 가속을 하고 싶다 | `base_mppi.py`의 `_compute_control_gpu()` (§6 포인트 3) | MPPI-Generic ([arXiv:2409.07563](https://arxiv.org/abs/2409.07563)), mppi-isaac, STORM |
| 샘플 수 K를 줄이고 싶다 | §4.3; dsMPPI(`deterministic_mppi.py`), TR-MPPI Halton, RF-MPPI | C-Uniform 샘플러 ([arXiv:2503.05819](https://arxiv.org/abs/2503.05819)) |
| 학습 모델을 동역학으로 쓰면? | §7.5 + `models/learned/`; [LEARNING_THEORY.md](../LEARNING_THEORY.md) | Hewing 리뷰 (01 문서 §9.1-⑧) |
| 안전을 "보장"하고 싶다 | §8.1-3 + §7.4; DualGuard/Shield/C-MPPI; [SAFETY_THEORY.md](../SAFETY_THEORY.md) | CBFkit ([arXiv:2404.07158](https://arxiv.org/abs/2404.07158)), reach-avoid MPPI ([arXiv:2407.13693](https://arxiv.org/abs/2407.13693)) |
| 유도를 처음부터 다시 보고 싶다 | §1→§3 순서로 재독 + §3.6 지도 | Williams T-RO ([arXiv:1707.02342](https://arxiv.org/abs/1707.02342)), 추론 관점은 [arXiv:2511.08019](https://arxiv.org/abs/2511.08019) |

### 10.6 이 repo에서 이어서 볼 것

- [docs/MPPI_THEORY.md](../MPPI_THEORY.md) — 43개 변형 전체 레퍼런스
- [docs/study/01_MPC_FUNDAMENTALS.md](01_MPC_FUNDAMENTALS.md) — 선행 문서
- [docs/LEARNING_THEORY.md](../LEARNING_THEORY.md) — 축 5 (학습 결합)의 이론
- `mppi_controller/controllers/mppi/base_mppi.py` — §6 대응표와 함께 정독
- `examples/comparison/all_37_variants_benchmark.py` — 축별 대표를 직접 비교:

```bash
PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --scenario obstacles
```
