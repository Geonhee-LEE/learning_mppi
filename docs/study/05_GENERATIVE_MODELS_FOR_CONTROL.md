# 05. 제어를 위한 생성 모델 — MPPI 샘플링 분포 학습

> **학습 시리즈 5편**: MPPI의 가우시안 제안 분포를 학습된 생성 모델로 대체/보강하는 이론과
> 이 저장소의 실제 구현(Flow-MPPI, SG-MPPI, Diffusion-MPPI, Latent-MPPI, Step-MPPI)을 잇는 스터디 가이드.
>
> **선수 지식**: [02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md) (information-theoretic MPPI 유도, importance sampling),
> 기초 확률론(가우시안, KL divergence), 신경망 학습 기초.
>
> **관련 구현 파일**:
>
> | 변형 | 컨트롤러 | 생성 모델 / 학습기 |
> |------|---------|------------------|
> | Flow-MPPI | `mppi_controller/controllers/mppi/flow_mppi.py` | `models/learned/flow_matching_model.py`, `learning/flow_matching_trainer.py` |
> | SG-MPPI | `mppi_controller/controllers/mppi/score_guided_mppi.py` | `models/learned/score_network.py`, `learning/score_matching_trainer.py` |
> | Diffusion-MPPI | `mppi_controller/controllers/mppi/diffusion_mppi.py` | `models/learned/diffusion_model.py`, `learning/diffusion_trainer.py` |
> | Latent-MPPI | `mppi_controller/controllers/mppi/latent_mppi.py` | `learning/world_model_trainer.py` (WorldModelVAE) |
> | Step-MPPI | `mppi_controller/controllers/mppi/step_mppi.py` | (내장 ProposalNetwork + ProposalTrainer) |
> | (참고) T-MPPI | `mppi_controller/controllers/mppi/transformer_mppi.py` | ControlTransformer |

---

## 목차

1. [왜 생성 모델인가 — 가우시안 제안 분포의 한계](#1-왜-생성-모델인가)
2. [VAE 복습 — Latent-MPPI의 기반](#2-vae-복습--latent-mppi의-기반)
3. [Score Matching & Diffusion — SG-MPPI와 Diffusion-MPPI](#3-score-matching--diffusion)
4. [Flow Matching / CFM — Flow-MPPI (핵심 챕터)](#4-flow-matching--cfm-핵심-챕터)
5. [학습 제안 분포의 공통 설계 패턴](#5-학습-제안-분포의-공통-설계-패턴)
6. [비교표 — 6가지 학습 제안 분포](#6-비교표--6가지-학습-제안-분포)
7. [연습문제](#7-연습문제)
8. [추천 자료](#8-추천-자료)

---

## 1. 왜 생성 모델인가

### 1.1 MPPI 복습 — 가우시안 제안 분포

Vanilla MPPI는 매 제어 주기에 다음을 수행한다 (`base_mppi.py`):

```
1. 노이즈 샘플링:  ε_k ~ N(0, Σ),          k = 1..K
2. 제어 시퀀스:    V_k = U + ε_k            (U: 이전 해, warm start)
3. 롤아웃:         X_k = rollout(x_0, V_k)
4. 비용:           S_k = cost(X_k, V_k, ref)
5. 가중치:         w_k = softmax(-S_k / λ)
6. 업데이트:       U ← U + Σ_k w_k ε_k
```

여기서 샘플이 뽑히는 분포 `q(V) = N(U, Σ)`를 **제안 분포(proposal distribution)** 라 부른다.
MPPI의 성능은 이 제안 분포가 "좋은 제어 시퀀스가 사는 영역"을 얼마나 잘 덮는지에 결정적으로 좌우된다.

### 1.2 가우시안 제안의 세 가지 한계

**한계 1 — 단봉성(unimodality)**: 가우시안은 봉우리가 하나다.
장애물을 왼쪽으로도 오른쪽으로도 피할 수 있는 상황(다중 모달 해)에서
가우시안 평균은 두 모드의 **중간** — 즉 장애물 정면 — 으로 수렴할 수 있다.

```
비용 지형 (낮을수록 좋음)                    가우시안 샘플의 문제

 cost                                        ●●● : 샘플
  │      ╲       ╱▔▔▔╲       ╱
  │       ╲     ╱ 장애물╲     ╱                     ┌── 평균이 여기로 끌림
  │        ╲   ╱  (高비용)╲   ╱                     ▼
  │   좋은  ╲ ╱           ╲ ╱  좋은          ──●●●●█●●●●──   ← N(U, Σ)
  │   경로A  ▼             ▼  경로B             A  ✕  B
  │  ────────┴─────────────┴────────→ u        두 모드의 평균 = 최악의 지점
  │      모드 A          모드 B
```

**한계 2 — 비용 지형 무시**: `N(U, Σ)`는 비용 함수 `S(·)`에 대해 아무것도 모른다.
저비용 영역이 어디인지 학습할 수 없으므로, 샘플 대부분이 고비용 영역에 낭비된다.
특히 λ가 작으면 유효 샘플 수(ESS)가 급감한다.

**한계 3 — 고차원 비효율**: 제어 시퀀스 공간은 `N × nu` 차원 (예: 30 × 2 = 60차원).
등방성 가우시안으로 60차원 공간에서 좋은 궤적을 우연히 맞추려면 K가 기하급수적으로 필요하다.
실제 좋은 제어 시퀀스는 이 공간의 **저차원 매니폴드** 위에 살고 있다
(부드럽고, 동역학적으로 일관되고, 태스크 구조를 반영).

### 1.3 해법: 제안 분포를 학습하자

아이디어는 단순하다. **과거에 MPPI가 찾아낸 좋은 해들 `(state, U*)`을 데이터 삼아,
"이 상태에서는 이런 제어 시퀀스가 좋더라"를 생성 모델 `q_θ(U | state)`로 학습**한다.

```
                    ┌────────────────────────────────┐
                    │        Self-Supervised 루프      │
                    │                                │
   state ──────────►│  q_θ(U|state) ──► 샘플 V_1..V_K │
                    │      ▲                │        │
                    │      │                ▼        │
                    │   학습 (주기적)      rollout+cost │
                    │      │                │        │
                    │  ring buffer ◄── U* (softmax 해)│
                    │  (state, U*)                   │
                    └────────────────────────────────┘
```

이 저장소의 다섯 변형은 모두 이 그림의 인스턴스이고, `q_θ`의 정체만 다르다:

- **Latent-MPPI**: `q_θ`가 아니라 **롤아웃 자체**를 VAE 잠재 공간에서 수행 (모델 학습)
- **SG-MPPI**: 가우시안 + score 방향 bias (하이브리드)
- **Diffusion-MPPI**: DDPM/DDIM 역확산으로 샘플 생성
- **Flow-MPPI**: CFM 속도장 ODE 적분으로 샘플 생성
- **Step-MPPI**: 신경망이 가우시안의 **(평균, 공분산)** 자체를 출력

### 1.4 제안 분포를 바꿔도 되는가 — Importance Sampling 관점

"샘플을 다른 분포에서 뽑았는데 같은 softmax 가중치를 써도 되나?"가 핵심 질문이다.
답은 Biased-MPPI (Trevisan & Alonso-Mora, RA-L 2024)의 논리와 동일하다.

**Information-theoretic MPPI 복습** (상세 유도는 02편 참조).
최적 분포 `Q*`는 base 분포 `P`에 대해 다음 밀도비를 가진다:

```
dQ*/dP (V) = exp(-S(V)/λ) / E_P[exp(-S/λ)]
```

MPPI 업데이트는 `Q*`의 평균을 구하는 것이다: `U* = E_{Q*}[V]`.
샘플을 임의의 제안 분포 `q_s`에서 뽑으면 self-normalized importance sampling으로:

```
U* = E_{Q*}[V] = E_{q_s}[ (dQ*/dq_s) V ]
              ≈ Σ_k w̄_k V_k,    w̄_k ∝ exp(-S(V_k)/λ) · (dP/dq_s)(V_k)
```

밀도비 `dP/dq_s`가 남아 있는 것처럼 보인다. 그러나:

**핵심 정리 (Biased-MPPI, 유도 스케치)**.
Information-theoretic MPPI에서 실제로 사용하는 비용은 순수 상태 비용이 아니라
자유 에너지 유도 과정에서 나오는 **제어 노이즈의 로그 우도 항이 흡수된 증강 비용** `S̃`이다.
증강 비용을 base 분포와 제안 분포의 로그 밀도비를 포함하도록 정의하면:

```
S̃(V) = S(V) + λ · log( q_s(V) / p(V) )
```

이때 self-normalized 가중치는:

```
w̄_k ∝ exp(-S̃(V_k)/λ) · (p/q_s)(V_k)
    = exp(-S(V_k)/λ) · (q_s/p)^{-1}(V_k) · (p/q_s)(V_k) ... (전개하면)
    = exp(-S(V_k)/λ)                        ← q_s 완전 소거!
```

즉, **비용에 밀도비 페널티를 함께 넣어주는 순간, 가중치 계산에서 제안 분포가 소거**되어
`softmax(-S/λ)`만 남는다. 실무적으로는 (그리고 이 저장소에서는) 밀도비 항을 생략하고
순수 비용의 softmax만 쓰는데, 이것이 "Biased"라는 이름의 유래다 —
추정량에 약간의 편향이 생기지만, **어떤 제안 분포에서 샘플을 가져와도
비용이 낮은 샘플이 큰 가중치를 받는다는 본질은 유지**되고,
좋은 제안 분포일수록 편향은 오히려 유리한 방향(저비용 영역 집중)으로 작동한다.

> **실무 요약**: 샘플 V_k가 flow에서 왔든, diffusion에서 왔든, Pure Pursuit 정책에서 왔든,
> `w_k = softmax(-S_k/λ)`로 평가하고 `U ← Σ w_k V_k`로 섞으면 된다.
> 나쁜 샘플은 자동으로 가중치 ≈ 0이 되므로 **최악의 경우에도 안전**하다.
> 이것이 이 저장소의 모든 생성 모델 변형이 `_compute_weights()`를 그대로 두고
> **샘플러만 교체**하는 이유다 (`flow_matching_sampler.py`가 `NoiseSampler` ABC를 준수).

### 1.5 무엇을 학습하는가 — 세 가지 수준

생성 모델을 MPPI에 결합하는 수준은 세 가지로 분류할 수 있다:

```
수준 1: 평균만 학습               수준 2: 분포 전체 학습          수준 3: 동역학까지 학습
  (T-MPPI, Step-MPPI 일부)        (Flow/Diffusion-MPPI)          (Latent-MPPI)

  q = N(μ_θ(x), Σ_고정)           q = q_θ(U|x) 임의 분포           롤아웃 자체를
                                                                 학습된 공간에서
  ── 가우시안 구조 유지            ── 다중 모달 가능
  ── 붕괴 위험 낮음               ── 학습 실패 시 붕괴 위험         ── 물리 모델 불필요
  ── 다중 모달 불가               ── fallback 필수                ── 재구성 오차 누적 위험
```

SG-MPPI는 수준 1과 2의 중간(가우시안 + score bias)이다. 3장에서 자세히 본다.

---

## 2. VAE 복습 — Latent-MPPI의 기반

### 2.1 잠재 변수 모델과 ELBO 유도

관측 `x` (여기서는 로봇 상태)가 저차원 잠재 변수 `z`로부터 생성된다고 가정한다:

```
z ~ p(z) = N(0, I),    x ~ p_θ(x|z)     (decoder)
```

우리가 원하는 것은 데이터의 로그 우도 `log p_θ(x)` 최대화지만,
`p_θ(x) = ∫ p_θ(x|z) p(z) dz`는 적분이 불가능(intractable)하다.
그래서 **근사 사후분포(encoder)** `q_φ(z|x)`를 도입한다.

**ELBO 유도** (임의의 `q_φ`에 대해 성립):

```
log p_θ(x) = log ∫ p_θ(x,z) dz
           = log ∫ q_φ(z|x) · [p_θ(x,z) / q_φ(z|x)] dz
           = log E_{q_φ}[ p_θ(x,z) / q_φ(z|x) ]
           ≥ E_{q_φ}[ log p_θ(x,z) - log q_φ(z|x) ]        (Jensen 부등식)
           = E_{q_φ}[ log p_θ(x|z) ] - KL( q_φ(z|x) ‖ p(z) )
           ─────────┬──────────────   ──────────┬──────────
              재구성 항                    정규화 항
           =: ELBO(θ, φ; x)
```

부등식의 갭이 정확히 `KL(q_φ(z|x) ‖ p_θ(z|x))`임도 확인할 수 있다:

```
log p_θ(x) = ELBO + KL( q_φ(z|x) ‖ p_θ(z|x) )
```

즉 ELBO 최대화 = (우도 최대화) + (encoder를 진짜 사후분포에 근접시키기)를 동시에 수행한다.

**Reparameterization trick**: `z ~ N(μ_φ(x), σ_φ²(x))`에서 직접 샘플링하면
gradient가 흐르지 않으므로, `z = μ_φ(x) + σ_φ(x) ⊙ ε, ε ~ N(0,I)`로 다시 쓴다.
이 저장소의 `WorldModelVAE.reparameterize()`가 정확히 이것이며,
**eval 모드에서는 μ만 반환**한다는 실무 디테일이 있다 (제어 시 결정론적 인코딩).

### 2.2 잠재 공간 롤아웃 — "Dream to Control" 아이디어

VAE에 **잠재 동역학** `z_{t+1} = f_ψ(z_t, u_t)`을 추가하면 world model이 된다
(Hafner et al. "Dream to Control", Watter et al. "Embed to Control").
MPPI의 롤아웃을 물리 공간이 아닌 잠재 공간에서 수행할 수 있다:

```
Latent-MPPI 파이프라인 (latent_mppi.py):

  x_0 ──[Encoder]──► z_0 ──┬──[f_ψ(·,u_0)]──► z_1 ──► ... ──► z_N   (K개 병렬)
                           │
                           └── K개 복제(tile)

  z_0..z_N ──[Decoder 일괄]──► x̂_0..x̂_N ──► costs = cost_fn(x̂, U, ref)
```

핵심 설계 결정: **비용은 디코딩된 물리 공간에서 평가**한다.
이 덕분에 기존 `CompositeMPPICost`(레퍼런스 추적, 장애물, CBF 비용 등)를
전혀 수정하지 않고 재사용한다. 잠재 공간에서 직접 비용을 정의하는 방식
(Dreamer의 learned reward)보다 공학적으로 훨씬 안전한 선택이다.

`WorldModelDynamics`(`models/learned/world_model_dynamics.py`)는 `RobotModel`의
`step()`을 오버라이드하여 RK4 대신 `encode → latent_step → decode`를 수행하므로,
`BatchDynamicsWrapper.rollout()`과도 자동 호환된다.

### 2.3 이 저장소의 교훈: 잔차(residual) latent dynamics

`WorldModelVAE.latent_step()`의 시그니처를 보면:

```python
def latent_step(self, z, control):
    """Residual latent dynamics: z_next = z + f(z, u)"""
```

잠재 동역학을 `z_next = f(z,u)`가 아니라 **`z_next = z + f(z,u)`** 로 정의한다. 이유:

1. **항등 매핑이 기본값**: 학습 초기에 `f ≈ 0`이면 `z_next ≈ z` — 상태가 갑자기
   튀지 않는다. `z_next = f(z,u)` 방식은 초기에 잠재 상태를 임의 지점으로 던져버린다.
2. **수축(contraction) 방지**: ReLU MLP는 출력 노름을 줄이는 경향이 있어,
   N스텝 롤아웃 시 `z`가 원점으로 수축하며 모든 미래가 같아지는 붕괴가 발생한다.
   잔차 연결은 이를 구조적으로 막는다.
3. **작은 dt 물리와의 정합**: 실제 이산 동역학도 `x_{t+1} = x_t + Δt·f(x,u)` 꼴이므로
   inductive bias가 맞는다.

추가 실무 디테일: encoder의 `log_var`를 `[-20, 2]`로 clamp하여 분산 폭주/소멸을 방지한다.

### 2.4 한계와 다음 장으로의 연결

Latent-MPPI는 **동역학을** 학습하지 **제안 분포는** 여전히 가우시안이다.
즉 1.2절의 한계 1, 2는 그대로 남는다. 다음 두 장에서는 제안 분포 자체를 학습하는
score/diffusion/flow 계열을 다룬다. VAE에서 배운 것 중 다음이 계속 재등장한다:

- "intractable한 분포를 다루기 위해 보조 분포/보조 노이즈를 도입한다"
- "학습 초기에 무해(no-op)하도록 구조를 설계한다" (잔차 연결 → zero-init으로 진화)

---

## 3. Score Matching & Diffusion

### 3.1 Score function이란

분포 `p(x)`의 **score**는 로그 밀도의 기울기다:

```
s(x) := ∇_x log p(x)
```

왜 score인가? 정규화 상수가 사라지기 때문이다. `p(x) = e^{-E(x)}/Z`라면:

```
∇_x log p(x) = -∇_x E(x) - ∇_x log Z = -∇_x E(x)     (Z는 x와 무관)
```

계산 불가능한 `Z`를 몰라도 score는 알 수 있다. 그리고 score만 있으면
**Langevin dynamics**로 `p`에서 샘플링할 수 있다:

```
x_{i+1} = x_i + (η/2)·s(x_i) + √η·ε_i,   ε_i ~ N(0, I)
```

직관: score는 "확률이 높아지는 방향"을 가리키는 벡터장이다.
제어 맥락에서 `p(U|x) ∝ exp(-S(U;x)/λ)`로 두면
**score = 비용이 낮아지는 방향** — 즉 비용 지형의 내리막 방향이다.

```
비용 지형 위의 score 벡터장:

   U₂ ▲
      │   ↘  ↓  ↙        ← score 벡터들이
      │   →  ●  ←           저비용 봉우리(●)를
      │   ↗  ↑  ↖           향해 수렴
      │
      │        ↘ ↓ ↙
      │        → ● ←     ← 다중 모드도 각각 표현 가능
      │        ↗ ↑ ↖        (가우시안과의 결정적 차이)
      └──────────────► U₁
```

### 3.2 DSM Loss 유도 — Vincent (2011) 트릭 전체

score를 신경망 `s_θ`로 회귀하고 싶다. 순진한 목적함수(Explicit Score Matching):

```
J_ESM(θ) = E_{p(x)}[ ‖s_θ(x) - ∇_x log p(x)‖² ]
```

문제: `∇ log p(x)`를 모른다 (그걸 배우려는 거니까). Vincent의 트릭은
**데이터에 노이즈를 섞은 분포의 score는 조건부 커널의 score로 대체 가능**하다는 것.

노이즈 커널 `q_σ(x̃|x) = N(x̃; x, σ²I)`로 흐린(smoothed) 분포를 정의:

```
p_σ(x̃) = ∫ q_σ(x̃|x) p(x) dx
```

**정리 (Vincent 2011)**: 다음 두 목적함수는 θ와 무관한 상수 차이만 난다.

```
J_ESM(θ) = E_{p_σ(x̃)}      [ ‖s_θ(x̃) - ∇_x̃ log p_σ(x̃)‖² ]
J_DSM(θ) = E_{p(x)q_σ(x̃|x)}[ ‖s_θ(x̃) - ∇_x̃ log q_σ(x̃|x)‖² ]
```

**증명**. 두 목적함수를 전개하면 `‖s_θ‖²` 항은 동일하다
(둘 다 `E_{p_σ(x̃)}[‖s_θ(x̃)‖²]` — 조건부 기대값의 탑 법칙).
target 쪽 제곱항은 θ와 무관한 상수. 남는 것은 교차항의 일치 여부다.

ESM의 교차항:

```
E_{p_σ}[ s_θ(x̃)ᵀ ∇ log p_σ(x̃) ]
  = ∫ s_θ(x̃)ᵀ ∇p_σ(x̃) dx̃                          (∇log p = ∇p/p 이므로 p 소거)
  = ∫ s_θ(x̃)ᵀ ∇_x̃ [∫ q_σ(x̃|x) p(x) dx] dx̃          (p_σ 정의 대입)
  = ∫∫ s_θ(x̃)ᵀ [∇_x̃ q_σ(x̃|x)] p(x) dx dx̃            (적분-미분 교환)
  = ∫∫ s_θ(x̃)ᵀ [∇_x̃ log q_σ(x̃|x)] q_σ(x̃|x) p(x) dx dx̃
  = E_{p(x)q_σ(x̃|x)}[ s_θ(x̃)ᵀ ∇_x̃ log q_σ(x̃|x) ]     = DSM의 교차항  ∎
```

**가우시안 커널의 target은 닫힌 형태**:

```
∇_x̃ log q_σ(x̃|x) = ∇_x̃ [ -‖x̃-x‖²/(2σ²) ] = -(x̃-x)/σ²
```

`x̃ = x + σε, ε ~ N(0,I)`로 파라미터화하면 target은 `-ε/σ`. 최종 DSM loss:

```
J_DSM(θ) = E_{x~data, ε~N(0,I)}[ ‖ s_θ(x + σε, σ) + ε/σ ‖² ]
```

**해석**: "노이즈가 섞인 점에서, 섞였던 노이즈를 되돌리는 방향(denoising 방향)을
예측하라". 이것이 diffusion 모델의 노이즈 예측 `ε_θ`와 본질적으로 같다
(`s_θ = -ε_θ/σ` 관계).

이 저장소의 구현이 정확히 이 식이다 (`score_network.py` 헤더):

```
DSM Loss: L(θ) = E[ ‖s_θ(U + σε, σ, state) - (-ε/σ)‖² ]
```

state를 조건으로 넣어 **조건부 score** `∇_U log p(U|state)`를 학습한다는 점만 다르다.

### 3.3 Annealed sampling — σ 스케줄이 필요한 이유

단일 σ의 문제:

- **σ가 작으면**: 데이터 매니폴드 밖(저밀도 영역)에서 score 추정이 엉망 —
  학습 데이터가 거기 없었으니까. Langevin 체인이 매니폴드에 도달하지 못한다.
- **σ가 크면**: 어디서나 score는 잘 정의되지만 흐려진 분포 `p_σ`의 score라서 부정확.

Song & Ermon (2019)의 해법: **여러 σ 수준을 기하급수 스케줄로 두고,
큰 σ에서 시작해 점점 줄이며(annealing) Langevin 샘플링**한다.
멀리서는 대충 방향만 잡고, 가까워질수록 정밀하게 수렴하는 것이다.

이 저장소의 SG-MPPI도 σ 스케줄을 사용한다 (`score_guided_mppi.py`):

```python
self._sigma_levels = np.geomspace(params.sigma_min, params.sigma_max, params.n_sigma_levels)
```

그리고 `ScoreNetwork`는 `SigmaEmbedding`(log σ의 sinusoidal 임베딩)으로
**하나의 네트워크가 모든 노이즈 수준을 조건부로 처리**한다 — NCSN/DDPM과 동일 패턴.

### 3.4 DDPM/DDIM 요약 — Diffusion-MPPI

Diffusion 모델(Ho et al. 2020)은 annealed score matching의 마르코프 체인 버전이다:

```
Forward (고정):   x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε        점점 노이즈化
Reverse (학습):   ε_θ(x_t, t) 로 노이즈 예측 → x_{t-1} 복원

학습:  L = E_{t, x_0, ε}[ ‖ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)‖² ]   ← DSM과 동형!
```

**DDIM** (Song et al. 2021)은 reverse 과정을 비마르코프/결정론적으로 재정의하여
T=1000 스텝 학습 모델을 **5~10 스텝**으로 샘플링할 수 있게 한다.
실시간 제어(10Hz)에서는 이것이 필수다.

`diffusion_mppi.py`는 `DDIMSampler`로 이를 구현한다:
- `diff_ddim_steps=5`: DDIM 가속 역확산 (5~10 스텝)
- `diff_T`, `diff_beta_schedule`: forward 과정 정의
- Flow-MPPI와 **동일한 self-supervised 학습 루프** + 동일한 3모드
  (`replace_mean` / `replace_distribution` / `blend`, 4.6절 참조)
- 미학습 시 가우시안 fallback (= Vanilla MPPI)

### 3.5 SG-MPPI — 왜 "하이브리드"인가

Diffusion/Flow-MPPI는 샘플 분포를 **통째로 대체**한다. 강력하지만 위험하다:
생성 모델이 나쁘게 학습되면 모든 샘플이 나쁜 영역에 몰려 MPPI가 붕괴한다.

SG-MPPI(`score_guided_mppi.py`)의 선택은 다르다.
**가우시안 구조를 유지하고, score 방향으로 노이즈를 살짝 밀기만** 한다:

```
ε ~ N(0, Σ)                                    ← 표준 가우시안 노이즈
ε_guided = ε + α · σ² · s_θ(U + ε, σ, state)   ← score 방향 bias 추가
```

이는 Langevin 스텝 `x + (η/2)∇log p + noise`의 1회 적용과 같은 꼴이다.

```
        가우시안만                    가우시안 + score bias

     ●  ●    ●                        ●●  ●
   ●    ✛U    ●          ──►        ●●✛U ●          ✛: 현재 해 U
     ●     ●   ●                    ●●●●              ★: 저비용 영역
   ★(멀리 있음)                      ★← 샘플 구름이
                                       저비용 쪽으로 이동
```

장점 세 가지:

1. **Graceful degradation**: `ScoreNetwork`의 출력층은 **zero-init**이다.
   학습 전 `s_θ ≈ 0` → `ε_guided = ε` → 정확히 Vanilla MPPI.
   학습이 진행될수록 부드럽게 성능이 올라간다. 붕괴 모드가 없다.
2. **탐색 보존**: 가우시안 분산이 그대로 살아 있어 ESS가 높게 유지된다
   (벤치마크에서 SG-MPPI MeanESS=383.2로 최고 기록).
3. **DIAL과의 결합**: `n_guide_iters>1` 또는 `use_annealing=True`면
   DIAL-style 다중 반복 + σ annealing과 score guidance를 결합한다.
   DIAL이 비용 지형 구조를 **매 스텝 새로 탐색**하는 반면,
   SG는 과거 경험에서 지형의 gradient를 **기억**해 재사용한다는 차이가 있다.

학습 데이터는 Flow-MPPI와 동일하게 `FlowDataCollector`를 재사용한다
(ring buffer에 `(state, U*)` 저장, 주기적 DSM 학습).

---

## 4. Flow Matching / CFM (핵심 챕터)

### 4.1 Continuous Normalizing Flow와 확률 흐름 ODE

생성 모델의 또 다른 관점: **간단한 분포를 복잡한 분포로 "수송(transport)"하는
연속적인 흐름**을 배우자. 시간 `t ∈ [0,1]`에 대해 ODE를 정의한다:

```
dx/dt = v_θ(x, t),      x(0) = x_0 ~ p_0 = N(0, I)
```

이 ODE의 해 `x(1)`의 분포가 데이터 분포 `p_1 = q_data`가 되도록 속도장 `v_θ`를 학습한다.
속도장이 입자들을 밀고 다닐 때 밀도가 어떻게 변하는지는 **연속성 방정식**이 지배한다:

```
∂p_t/∂t + ∇·(p_t · v_t) = 0        (질량 보존 — 유체역학과 동일)
```

속도장 `v_t`가 이 방정식을 통해 확률 경로 `{p_t}`를 "생성한다(generates)"고 말한다.

```
       t=0                t=0.5               t=1
    N(0, I)           중간 분포             데이터 분포
      ⬤        ──►      ◗◖        ──►      ●   ●
    (한 덩어리)        (갈라지는 중)         (다중 모달!)

    입자들이 v_θ(x,t)를 따라 흘러가며 분포가 변형된다
```

전통적 CNF(Chen et al. 2018, Neural ODE)는 log-likelihood로 학습했는데,
이는 **매 학습 스텝마다 ODE를 시뮬레이션**해야 해서 극도로 느렸다.
Flow Matching의 혁신은 이 시뮬레이션을 완전히 제거한 것이다.

### 4.2 Flow Matching 목적함수와 Conditional FM 정리

목표 확률 경로 `p_t`와 그것을 생성하는 목표 속도장 `u_t(x)`를 안다고 치면,
학습은 단순 회귀다:

```
L_FM(θ) = E_{t~U[0,1], x~p_t}[ ‖v_θ(x, t) - u_t(x)‖² ]
```

문제: **marginal** 속도장 `u_t(x)`를 모른다. 그런데 데이터 샘플 `x_1` **하나에
조건부인** 경로는 우리가 마음대로 설계할 수 있다:

```
조건부 경로:    p_t(x | x_1)      (예: x_1을 향해 좁아지는 가우시안)
조건부 속도장:  u_t(x | x_1)      (닫힌 형태로 계산 가능!)
```

marginal은 이들의 혼합이다:

```
p_t(x) = ∫ p_t(x|x_1) q(x_1) dx_1
u_t(x) = ∫ u_t(x|x_1) · [p_t(x|x_1) q(x_1) / p_t(x)] dx_1     (조건부 속도장의 사후 가중 평균)
```

**Conditional Flow Matching 정리 (Lipman et al. 2023, Theorem 2)**:

```
L_CFM(θ) = E_{t, x_1~q, x~p_t(·|x_1)}[ ‖v_θ(x,t) - u_t(x|x_1)‖² ]

이면       ∇_θ L_FM(θ) = ∇_θ L_CFM(θ)
```

**유도 스케치**. 두 loss를 전개한다:

```
L_FM  = E[‖v_θ‖²] - 2·E[v_θᵀ u_t(x)]    + const
L_CFM = E[‖v_θ‖²] - 2·E[v_θᵀ u_t(x|x_1)] + const'
```

(1) `‖v_θ‖²` 항: `L_FM`은 `x~p_t`, `L_CFM`은 `x_1~q, x~p_t(·|x_1)`에서 기대값인데,
marginal 정의상 `p_t(x) = ∫p_t(x|x_1)q(x_1)dx_1`이므로 두 기대값은 동일하다.

(2) 교차항:

```
E_{p_t}[ v_θ(x)ᵀ u_t(x) ]
  = ∫ v_θ(x)ᵀ u_t(x) p_t(x) dx
  = ∫ v_θ(x)ᵀ [ ∫ u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1 ] dx     (u_t 정의 대입 — p_t 소거)
  = ∫∫ v_θ(x)ᵀ u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1 dx
  = E_{x_1~q, x~p_t(·|x_1)}[ v_θ(x)ᵀ u_t(x|x_1) ]              = CFM의 교차항  ∎
```

θ에 의존하는 항이 전부 일치하므로 gradient가 같다.
**3.2절 Vincent 트릭과 구조가 완전히 동일**함에 주목하라 —
"intractable한 marginal target을 tractable한 conditional target으로 바꿔도
회귀 문제의 gradient는 같다". 이 패턴이 score matching과 flow matching을 관통한다.

### 4.3 OT interpolation 경로 — 가장 단순한 조건부 경로

조건부 경로를 자유롭게 설계할 수 있으니, 가장 단순한 것을 고르자:
**노이즈 `x_0`와 데이터 `x_1`을 잇는 직선** (optimal transport displacement):

```
x_t = (1-t)·x_0 + t·x_1,        x_0 ~ N(0, I)
```

이 경로의 속도는 미분하면 바로 나온다:

```
u_t = dx_t/dt = x_1 - x_0        ← 시간에 무관한 상수 속도!
```

최종 학습 알고리즘은 허무할 정도로 단순하다:

```
반복:
  x_1 ~ 데이터,  x_0 ~ N(0,I),  t ~ U[0,1]
  x_t = (1-t)·x_0 + t·x_1
  loss = ‖ v_θ(x_t, t, context) - (x_1 - x_0) ‖²
  gradient step
```

이 저장소의 `flow_matching_trainer.py::_cfm_step()`이 정확히 이 4줄이다:

```python
x0 = torch.randn_like(x1)                    # x₀ ~ N(0, I)
t = torch.rand(B, device=self.device)        # t ~ U[0, 1]
x_t = (1 - t_expand) * x0 + t_expand * x1    # OT interpolation
target_v = x1 - x0                           # 상수 목표 속도
```

> **참고**: Lipman 원논문의 OT 경로는 `σ_min > 0`을 남기는
> `x_t = (1-(1-σ_min)t)x_0 + t·x_1` 꼴이고, 위의 순수 직선 버전은
> Rectified Flow (Liu et al. 2023) / I-CFM (Tong et al. 2023)과 일치한다.
> 실무 성능 차이는 미미하며, 이 저장소는 단순한 직선 버전을 쓴다.

**왜 직선 경로가 좋은가**: 목표 속도장이 상수이므로 학습된 ODE 궤적도 거의 직선이다.
직선 ODE는 **적은 Euler 스텝으로도 정확히 적분**된다 — 4.5절의 "5 스텝 생성"의 비밀.

### 4.4 Diffusion 대비 장점

| 항목 | Diffusion (DDPM) | Flow Matching (OT-CFM) |
|------|-----------------|----------------------|
| 학습 | 시뮬레이션-free (노이즈 예측 회귀) | 시뮬레이션-free (속도 회귀) — 동급 |
| 확률 경로 | forward SDE가 결정 (곡선, 분산 폭발/보존) | **설계 자유** — 직선 OT 선택 가능 |
| 생성 | reverse SDE/ODE, 수십~수천 스텝 (DDIM으로 5~50) | ODE, 직선에 가까워 **5 스텝이면 충분** |
| t=1에서 데이터 도달 | 점근적 (T→∞에서만 정확히 가우시안) | **정확히 t=1에서 도달** |
| 구현 복잡도 | β 스케줄, ᾱ 누적곱, posterior 분산 등 | 보간 한 줄 + 회귀 한 줄 |
| 이론적 관계 | diffusion의 확률 흐름 ODE는 FM의 특수한 경로 선택에 해당 | 상위 일반화 |

제어 관점의 결론: **10Hz 제어 주기 안에 K=1024개 샘플을 생성해야 하는 MPPI에서,
"적분 스텝 수 = NN forward 횟수"이므로 5-스텝 직선 ODE의 가치는 절대적**이다.
같은 이유로 Diffusion-MPPI도 DDIM 가속이 없으면 실용 불가다.

### 4.5 이 저장소의 Flow-MPPI 구현 매핑

**(a) FlowMatchingModel** (`models/learned/flow_matching_model.py`):

```
입력: [ x_t (N·nu 평탄화 제어 시퀀스) | SinusoidalTimeEmbedding(t) | state (context) ]
      → MLP (SiLU, 기본 [256,256,256])
출력: v (N·nu 속도 벡터)
```

- 조건부 생성: state를 concat하여 `v_θ(x_t, t | state)` — "지금 이 상태에서" 좋은
  제어 시퀀스 분포를 낸다.
- `generate()`: `x_0 ~ N(0,I)`에서 시작해 Euler 또는 midpoint로 `num_steps=5` 적분.
  midpoint는 스텝당 forward 2회지만 2차 정확도 — 곡률이 있는 흐름에서 유리하다.

**(b) FlowMatchingSampler 3모드** (`flow_matching_sampler.py`) —
`NoiseSampler` ABC를 준수하므로 MPPI 본체 수정이 전혀 없다:

```
mode="replace_mean":            flow가 평균 1개 생성 → μ 주위 가우시안 탐색
  noise_k = (μ_flow - U) + N(0, exploration_sigma²·σ²)
  ── 가장 보수적. 단봉이지만 평균이 학습된 위치로 이동.

mode="replace_distribution":    flow가 K개 샘플 직접 생성
  noise_k = flow_sample_k - U
  ── 가장 공격적. 완전한 다중 모달. 모델 품질에 전적으로 의존.

mode="blend":                   flow K·ratio개 + 가우시안 K·(1-ratio)개 혼합
  ── 실무 추천. flow가 저비용 영역을 집중 공략하고,
     가우시안이 탐색과 안전망(모델 오류 보험)을 담당.
     1.4절 정리 덕분에 섞어도 softmax 가중치는 그대로 유효.
```

```
        replace_mean          replace_distribution        blend

         ●●●                    ●●        ●●              ●●    ○
        ●●μ●●                   ●●        ●●             ●●   ○ ✛ ○
         ●●●    ✛U               (모드A)   (모드B)          (flow) ○ (gauss)
   flow가 평균만 이동          flow가 분포 전체 담당        반반 혼합
```

**(c) Self-supervised 데이터 수집** (`FlowDataCollector` + `flow_mppi.py`):

전문가 시연이 전혀 필요 없다. **MPPI 스스로가 교사**다:

```
매 스텝: compute_control() 종료 후 → collector.add_sample(state, self.U.copy())
                                      (U = softmax 가중 평균 해 = "elite" 해)
주기적: step_count % flow_training_interval == 0 이고
        buffer ≥ flow_min_samples 이면 → 20 epochs CFM 학습
```

이것이 Kurtz & Burdick (2025) GPC의 핵심 아이디어다: MPPI의 출력(가중 평균 해)은
가우시안 샘플보다 항상 좋으므로, 그것을 데이터로 삼으면 flow는
"MPPI가 결국 도달할 곳"을 한 번에 생성하도록 학습된다 — **자기 증류(self-distillation)**.

**(d) Graceful degradation**:

```python
if not self.is_flow_ready:                     # 모델 없음 or context 없음
    return self._gaussian_fallback(U, K, ...)  # = Vanilla MPPI와 동일
```

Flow-MPPI의 안전망은 SG-MPPI의 zero-init과 다른 방식이다:
**모델이 준비되기 전에는 아예 호출하지 않는다** (fallback 스위치).
zero-init은 "출력이 0인 모델"이고 fallback은 "모델 없음 분기"인데,
효과는 동일하다 — **학습 전 = Vanilla MPPI, 절대 더 나빠지지 않음**.
단, 학습 *후* 모델이 이상해지는 경우는 fallback이 못 잡으므로 blend 모드가 보험이 된다.

**(e) 실행해 보기**:

```bash
PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --live --scenario obstacles
PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --all-scenarios
```

### 4.6 Step-MPPI — 생성 모델의 극한 경량화 (DPC)

스펙트럼의 반대쪽 끝: **분포를 통째로 배우는 대신, 가우시안의 파라미터만 배우면?**
Step-MPPI (`step_mppi.py`, arXiv:2604.01539)는 Differentiable Predictive Control 관점으로
proposal 분포를 신경망으로 파라미터화한다:

```
(Δμ_θ(φ), log σ_θ(φ)) = NN(φ),   φ = 특징(state, ref 요약)

런타임:  μ = U_warm + blend · Δμ_θ                     (평균: warm start + 학습 잔차)
        U_k = μ + diag(σ_base · exp(log σ_θ)) · ε_k     (상태 적응적 공분산!)

자기지도 학습:  L(θ) = E[ ‖Δμ_θ - (U* - U_warm)‖² ] - τ · Σ log σ_θ
```

주목할 설계 3가지:

1. **잔차 예측**: 절대 시퀀스가 아니라 `U* - U_warm`(warm start 대비 잔차)을 회귀.
   2.3절 잔차 latent dynamics와 같은 철학 — 기본값이 무해하도록.
2. **zero-init 출력층**: 학습 전 `Δμ=0, σ=σ_base` → 정확히 Vanilla. (또 그 패턴!)
3. **최대 엔트로피 정규화** `-τ·Σ log σ_θ`: MSE만 쓰면 네트워크가 σ→0으로
   붕괴시키는 것이 이득이므로(불확실성 숨기기), 엔트로피 보너스로 분산을 유지시킨다.
   이것이 5장의 "분포 붕괴 방지"의 가장 명시적인 사례다.

장기 호라이즌 MPC 목적을 **학습 시점에** 흡수하므로, 런타임에는 짧은 최적화만으로
장기 계획 정보를 암묵적으로 활용한다 → 초저지연. 대신 다중 모달성은 포기한다
(출력이 단일 가우시안이므로). T-MPPI(Transformer로 U_init 예측)도 같은 "수준 1" 계열이며,
차이는 특징 추출기(MLP vs 히스토리 attention)다.

---

## 5. 학습 제안 분포의 공통 설계 패턴

다섯 변형을 관통하는 공학 패턴을 정리한다. 새 변형을 만들 때의 체크리스트로 쓰자.

### 5.1 Ring buffer + elite selection

```
FlowDataCollector (Flow/SG 공용), StepExperienceBuffer (Step), TransformerDataBuffer (T)

┌───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ ← buffer_size 고정, 가장 오래된 것부터 덮어쓰기
└───┴───┴─▲─┴───┴───┘
          idx (다음 쓰기 위치)
```

- **왜 ring buffer**: 온라인 제어는 무한 스트림이다. 무한 축적은 메모리 폭발 +
  오래된(다른 정책 시절의) 데이터가 학습을 오염시킨다. 고정 크기 링은
  자연스러운 "최근성 가중"이 된다.
- **왜 elite만 저장**: K개 샘플 전부가 아니라 **softmax 가중 평균 해 `U` 하나만**
  저장한다. 이것이 elite selection — 생성 모델이 "MPPI 샘플 분포"가 아니라
  "MPPI가 수렴하는 좋은 해 분포"를 배우게 한다. (K개 전부 배우면
  가우시안을 다시 배우는 꼴이 된다!)
- 주의점 (SG-MPPI 교훈): `buffer_size < min_samples`로 설정하면 영원히 학습이
  시작되지 않는다. 파라미터 검증 필수.

### 5.2 Zero-init 출력층 / fallback — "학습 전 = Vanilla" 불변식

| 변형 | 메커니즘 | 학습 전 동작 |
|------|---------|------------|
| Flow-MPPI | `is_flow_ready` 분기 → gaussian fallback | Vanilla |
| Diffusion-MPPI | 동일 fallback 패턴 | Vanilla |
| SG-MPPI | ScoreNetwork 출력층 zero-init → `s_θ≈0` | Vanilla |
| Step-MPPI | mean/logstd head 모두 zero-init → `Δμ=0, σ=σ_base` | Vanilla |
| T-MPPI | zero-init + blend_ratio | Vanilla |
| Latent-MPPI | world_model 없으면 표준 MPPI 폴백 | Vanilla |

**불변식**: *생성 모델을 추가해서 성능이 나빠지는 일은 (학습 전에는) 구조적으로 불가능해야 한다.*
안전-필수 제어기에 학습 요소를 넣을 때의 제1원칙이다.

두 방식의 차이도 알아두자:
- **fallback 분기**: 코드가 명시적, 디버깅 쉬움. 단 "모델 있음/없음"의 이진 전환.
- **zero-init**: 학습이 진행되며 **연속적으로** 영향력이 커짐. 전환 충격이 없음.
  단, 학습이 잘못되면 자동으로 꺼지지 않음 → blend/exploration 노이즈로 보완.

### 5.3 온라인 학습 주기 설계

```
if step % training_interval != 0: return       # (1) 매 스텝 학습 금지
if buffer.count < min_samples:    return       # (2) 데이터 부족 시 학습 금지
train(epochs=20)                                # (3) 소량 epoch만
```

- **(1) interval**: 학습은 제어 주기(100ms)를 깨뜨린다. 실시간성이 필요하면
  interval을 길게 하거나 별도 스레드로 뺀다. (벤치마크 기록: 온라인 학습 시
  4.4s vs 기준선 2.1s — 학습 비용은 공짜가 아니다.)
- **(2) min_samples**: 소표본 과적합은 분포 붕괴의 지름길. 초기 구간은 fallback으로 버틴다.
- **(3) 소량 epoch**: 매번 수렴까지 학습하면 최신 데이터에 과적합 + 급격한 분포 변화.
  20 epoch씩 자주가 100 epoch씩 가끔보다 안정적이다 (일종의 EMA 효과).

### 5.4 분포 붕괴 방지

학습 제안 분포의 고유 위험: **자기 데이터로 자기를 학습**하는 폐루프이므로,
분포가 좁아지면 → 좁은 데이터만 수집 → 더 좁아짐의 양성 피드백이 가능하다.

| 방어 수단 | 사용처 | 원리 |
|----------|--------|------|
| 엔트로피 정규화 `-τ·Σlogσ` | Step-MPPI | σ→0 붕괴에 직접 페널티 |
| blend 모드 (가우시안 혼합) | Flow/Diffusion | 탐색 샘플이 항상 일정 비율 존재 |
| exploration_sigma | Flow replace_mean | 학습된 평균 주위 강제 탐색 |
| score bias만 (분포 유지) | SG-MPPI | 가우시안 분산이 애초에 안 줄어듦 |
| ring buffer (오래된 다양성 보존) | 전체 | 붕괴 이전 데이터가 한동안 남음 |
| 입력/출력 정규화 통계 | FlowMatchingTrainer | 스케일 드리프트로 인한 발산 방지 |

### 5.5 검증 시 주의 (repo 교훈)

- **RNG 부작용**: "α=0이면 Vanilla와 동일" 테스트가 실패한 사례 —
  `compute_control()` 내부의 데이터 수집/학습이 RNG 상태를 분기시킨다.
  등가성 테스트는 컨트롤러 레벨이 아니라 **샘플러 함수 레벨**에서 비교하라
  (`_sample_with_score()` 직접 호출).
- **numpy `or` 함정**: `sigma_override or default`는 numpy 배열에서
  ambiguity 에러 — `x if x is not None else default`를 써라.

---

## 6. 비교표 — 6가지 학습 제안 분포

| 축 | VAE (Latent) | Score (SG) | Diffusion | CFM (Flow) | Transformer (T) | DPC (Step) |
|----|-------------|-----------|-----------|-----------|----------------|-----------|
| **학습 대상** | 동역학 (encoder+latent dyn+decoder) | 비용 지형의 ∇log p | 노이즈 예측 ε_θ | 속도장 v_θ | U_init 예측 (히스토리→시퀀스) | 가우시안 (Δμ, logσ) |
| **학습 loss** | ELBO (재구성+KL) | DSM 회귀 | ε 회귀 (DSM 동형) | CFM 회귀 (직선 속도) | MSE (U* 회귀) | MSE − 엔트로피 |
| **학습 비용** | 중 (3개 네트워크) | 중 (σ 스케줄 필요) | 중 (T 스텝 스케줄) | **저 (보간+회귀만)** | 중 (attention) | **최저 (소형 MLP)** |
| **추론 비용/샘플** | rollout 자체가 추론 (N회 latent step) | forward 1회 (bias 계산) | DDIM 5~10 스텝 | ODE 5 스텝 | forward 1회 (평균만) | forward 1회 (평균만) |
| **다중모달성** | ✕ (제안은 가우시안) | △ (모드별 score는 표현, 샘플은 국소 이동) | ◎ | ◎ | ✕ (단일 시퀀스) | ✕ (단일 가우시안) |
| **MPPI 결합 방식** | rollout 대체 (수준 3) | 노이즈 bias (수준 1.5) | 샘플러 대체 3모드 (수준 2) | 샘플러 대체 3모드 (수준 2) | 초기해 U_init (수준 1) | 제안 파라미터 (수준 1) |
| **학습 전 안전망** | 표준 MPPI 폴백 | zero-init | gaussian fallback | gaussian fallback | zero-init + blend | zero-init |
| **붕괴 위험** | 재구성 오차 누적 | **최저** (가우시안 유지) | blend 없으면 있음 | blend 없으면 있음 | 낮음 | 엔트로피 항으로 방어 |
| **적합 상황** | 물리 모델 없음/고차원 관측 | 비용 지형이 험한 장애물 환경 | 오프라인 대량 데이터 (Diffusion Policy류) | **온라인 self-supervised 실시간** | 반복 태스크 warm start | 초저지연/임베디드 |

**선택 가이드 요약**:

```
물리 모델이 없다/관측이 고차원이다        → Latent-MPPI (동역학부터 학습)
안전이 최우선, 점진적 개선 원함           → SG-MPPI (하이브리드)
다중 모달 해가 명확히 존재 (장애물 양쪽)   → Flow-MPPI blend 모드
  └ 오프라인 시연 데이터가 많다           → Diffusion-MPPI
계산 예산이 극도로 빡빡하다               → Step-MPPI / T-MPPI
```

---

## 7. 연습문제

**문제 1 (importance weight 소거).**
1.4절의 증강 비용 `S̃(V) = S(V) + λ·log(q_s(V)/p(V))`을 사용할 때
self-normalized 가중치에서 `q_s`가 소거됨을 손으로 전개하여 보여라.
그리고 밀도비 항을 **생략**했을 때(Biased-MPPI의 실제 구현) 추정량이
어느 방향으로 편향되는지 논하라. (힌트: `q_s`가 저비용 영역에 집중되어 있다면
그 영역의 샘플이 "과다 대표"되는데, 이것이 왜 실무에서 해가 되지 않는가?)

**문제 2 (DSM 유도 완성).**
3.2절 증명에서 `‖s_θ‖²` 항이 ESM과 DSM에서 같음을 조건부 기대값의 탑 법칙
`E_{p_σ(x̃)}[f(x̃)] = E_{p(x)}E_{q_σ(x̃|x)}[f(x̃)]`로 명시적으로 보여라.
또한 가우시안 커널에서 target `-(x̃-x)/σ² = -ε/σ`의 마지막 등식이 성립하려면
`x̃ = x + σε` 파라미터화에서 σ 몇 제곱이 소거되는지 확인하라.

**문제 3 (CFM 정리의 조건).**
4.2절 유도에서 marginal 속도장 `u_t(x)`가 조건부 속도장의
**사후 가중 평균**임을 사용했다. `p_t(x) = 0`인 점에서 이 정의가 왜 문제가 되며,
Lipman et al.이 어떤 조건(`p_t(x) > 0` a.e.)을 가정하는지 설명하라.
그리고 OT 직선 경로에서 서로 다른 `(x_0, x_1)` 쌍의 직선들이 교차하는 점에서
marginal 속도장은 무엇이 되는지 기하적으로 서술하라. (힌트: 교차점에서
개별 직선 속도들의 평균 — 이것이 Rectified Flow의 "reflow"가 필요한 이유다.)

**문제 4 (설계 문제 — 붕괴 시나리오).**
Flow-MPPI를 `replace_distribution` 모드, blend 없이, 온라인 학습으로 돌린다고 하자.
어느 시점에 flow가 "직진만 하는" 시퀀스만 생성하도록 잘못 수렴했다.
(a) 이후 ring buffer에 어떤 데이터가 쌓이고 학습이 어떻게 진행될지 폐루프를 추적하라.
(b) 5.4절의 방어 수단 중 이 시나리오를 끊을 수 있는 것을 모두 고르고 각각의 개입 지점을 표시하라.
(c) `flow_mppi_benchmark.py`로 이 가설을 검증할 실험을 설계하라
(모드/blend_ratio를 바꿔가며 MinClearance와 ESS를 비교).

**문제 5 (구현 문제 — σ-조건부 CFM).**
현재 `FlowMatchingModel`은 state만 조건으로 받는다. SG-MPPI의 `SigmaEmbedding`
패턴을 참고하여, **레퍼런스 궤적 요약**(예: 향후 N스텝 웨이포인트의 로봇 좌표계
상대 위치)을 추가 context로 넣는 확장을 설계하라. 입력 차원, 정규화 통계,
`FlowDataCollector`에 저장해야 할 추가 필드, 그리고 학습 전 fallback 불변식이
유지되는지를 명세로 작성하라. (실제 구현은 `flow_matching_model.py`,
`flow_data_collector.py`, `flow_mppi.py` 3파일 수정으로 가능하다.)

---

## 8. 추천 자료

### 핵심 논문 (이 문서의 유도가 따르는 순서)

1. **Lipman et al. (2023)** — *Flow Matching for Generative Modeling* (ICLR 2023).
   CFM 정리의 원전. Theorem 1-3과 OT 경로 섹션만 읽어도 4장 전체가 커버된다.
2. **Vincent (2011)** — *A Connection Between Score Matching and Denoising Autoencoders*.
   DSM 트릭의 원전. 4페이지로 짧다.
3. **Song & Ermon (2019)** — *Generative Modeling by Estimating Gradients of the Data
   Distribution* (NeurIPS 2019). NCSN — σ 스케줄과 annealed Langevin의 원전.
4. **Ho et al. (2020)** — *Denoising Diffusion Probabilistic Models*.
   DDPM. Song et al. (2021) *DDIM*과 함께 읽을 것 (가속 샘플링).
5. **Kingma & Welling (2014)** — *Auto-Encoding Variational Bayes*. VAE/ELBO 원전.

### 제어 응용

6. **Chi et al. (2023)** — *Diffusion Policy* (RSS 2023). 생성 모델을 정책으로 쓰는
   대표작. 다중 모달 행동 표현의 실증.
7. **Kurtz & Burdick (2025)** — *Generative Predictive Control* (GPC).
   이 저장소 Flow-MPPI의 self-supervised 루프의 직접적 원형.
8. **Trevisan & Alonso-Mora (2024)** — *Biased-MPPI* (RA-L). 제안 분포 교체의
   정당화 (1.4절 정리). arXiv:2401.09241.
9. **Hafner et al. (2020)** — *Dream to Control* (Dreamer, ICLR 2020).
   잠재 공간 계획의 대표작 — Latent-MPPI의 배경.
10. **Liu et al. (2023)** — *Rectified Flow* (ICLR 2023) + Tong et al. (2023)
    *Improved CFM*. 직선 경로 계열의 정리와 reflow 기법.

### 강의/입문 자료

- Yang Song의 블로그 *"Generative Modeling by Estimating Gradients of the Data Distribution"* — score 계열 최고의 입문.
- MIT 6.S184 / Peter Holderrieth의 *Flow Matching and Diffusion Models* 강의 노트 — FM과 diffusion을 통합 관점으로.
- Lilian Weng, *"What are Diffusion Models?"* — DDPM 수식 전개 정리.

### 이 저장소에서 이어서 볼 것

- `docs/MPPI_THEORY.md` — 43종 변형 전체 레퍼런스 (본 문서는 그중 생성 모델 계열의 심화)
- `docs/LEARNING_THEORY.md` — 학습 동역학 모델 (BNN/GP/Ensemble 등) 이론
- 벤치마크 실행: `examples/comparison/flow_mppi_benchmark.py`,
  `score_guided_mppi_benchmark.py`, `latent_mppi_benchmark.py`, `step_mppi_benchmark.py`

---

*이전 문서: [04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md) | 시리즈 인덱스: [README.md](README.md)*
