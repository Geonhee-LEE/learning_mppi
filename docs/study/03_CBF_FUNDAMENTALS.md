# CBF 기초 공부 자료 — 안전을 수학으로 만들기

> **공부 자료 시리즈 3편.** 이 문서는 Control Barrier Function(CBF)을 밑바닥부터
> 쌓아 올리는 **학습용** 문서입니다. 기법별 레퍼런스가 필요하면
> [docs/SAFETY_THEORY.md](../SAFETY_THEORY.md)를 보세요 (특히 §1 기초, §3 CBF-QP,
> §15 선택 가이드). 여기서는 "왜 이렇게 정의하는가"라는 직관과 손계산 유도에
> 집중합니다.
>
> **대상 독자**: 안전 필수 제어(safety-critical control)를 기초부터 공부하려는
> 로보틱스 엔지니어. 선수 지식: 상미분방정식, 기초 선형대수, 라그랑주 승수법의
> 개념 정도.

---

## 목차

1. [안전을 수학으로: 안전 집합과 전방 불변성](#1-안전을-수학으로-안전-집합과-전방-불변성)
2. [Nagumo 정리 — 경계에서 벡터장이 안쪽을 향하면 된다](#2-nagumo-정리--경계에서-벡터장이-안쪽을-향하면-된다)
3. [Class-K 함수와 CBF 정의 — 왜 내부에서도 조건을 거는가](#3-class-k-함수와-cbf-정의--왜-내부에서도-조건을-거는가)
4. [Lie derivative 손계산 — Differential Drive 완전 예제](#4-lie-derivative-손계산--differential-drive-완전-예제)
5. [CBF-QP와 해석해 — 단일 제약 투영 공식 유도](#5-cbf-qp와-해석해--단일-제약-투영-공식-유도)
6. [CLF와의 결합 — 안전 > 수렴의 우선순위 설계](#6-clf와의-결합--안전--수렴의-우선순위-설계)
7. [Relative Degree 문제와 HOCBF 개요](#7-relative-degree-문제와-hocbf-개요)
8. [CBF의 실패 모드와 이 repo의 해법 매핑](#8-cbf의-실패-모드와-이-repo의-해법-매핑)
9. [Discrete-time CBF — MPPI 비용으로 쓸 때](#9-discrete-time-cbf--mppi-비용으로-쓸-때)
10. [연습문제](#10-연습문제)
11. [부록 — 더 공부하기 위한 자료](#11-부록--더-공부하기-위한-자료)

---

## 1. 안전을 수학으로: 안전 집합과 전방 불변성

### 1.1 "안전하다"를 어떻게 수식으로 쓸까

로봇이 "안전하다"는 말은 애매합니다. 이것을 수학으로 바꾸는 첫걸음은
**안전한 상태들의 집합**을 정의하는 것입니다.

연속적으로 미분 가능한 함수 h: ℝⁿ → ℝ 를 하나 잡고,

```
C = { x ∈ ℝⁿ : h(x) ≥ 0 }        (안전 집합, safe set)
∂C = { x : h(x) = 0 }             (경계, boundary)
Int(C) = { x : h(x) > 0 }         (내부, interior)
```

h(x)는 "안전의 여유분"을 재는 자입니다. 예를 들어 반경 r짜리 원형 장애물
(x_o, y_o)를 피하는 문제라면:

```
h(x) = (x - x_o)² + (y - y_o)² - r²
```

- h > 0: 장애물 밖 (안전)
- h = 0: 장애물 경계에 정확히 접촉
- h < 0: 장애물 안 (충돌)

이 repo에서 쓰는 다양한 h(x) 설계(타원, 벽, 속도 제한, 충돌 원뿔 등)는
[SAFETY_THEORY.md §1.2](../SAFETY_THEORY.md)에 정리되어 있습니다.

### 1.2 전방 불변성 (Forward Invariance)

이제 "안전 유지"를 정의할 수 있습니다.

> **정의 (전방 불변성).** 동역학 ẋ = f(x, u)와 제어 정책 u = π(x)에 대해,
> 집합 C가 **전방 불변(forward invariant)** 이라는 것은:
>
> x(0) ∈ C  ⟹  x(t) ∈ C  for all t ≥ 0
>
> 즉, 안전하게 시작하면 영원히 안전하다는 뜻입니다.

핵심 통찰: **안전 = 집합의 불변성**. "충돌하지 않는다"라는 시간 축 전체에 걸친
성질을, "집합에서 나가지 않는다"라는 기하학적 성질로 바꿨습니다. 이제 질문은
하나로 좁혀집니다:

> **어떤 조건을 걸면 x(t)가 C를 빠져나가지 못하는가?**

```
        상태 공간
   ┌─────────────────────────────┐
   │                             │
   │        C  (h ≥ 0)           │
   │     ┌───────────┐           │
   │     │  x(0) ●───┼──→ x(t)   │   x(t)가 C 밖으로 나가면
   │     │      안전  │  ✗ 위험   │   "안전 위반"
   │     └───────────┘           │
   │       h < 0 (위험 영역)      │
   └─────────────────────────────┘
```

### 1.3 왜 "매 순간 h ≥ 0 확인"으로는 부족한가

순진한 접근: "매 스텝 h(x) ≥ 0인지 확인하고, 아니면 멈추자." 문제는 **관성과
반응 지연**입니다. h = 0.01 (거의 접촉)인데 로봇이 장애물을 향해 전속력으로
달리고 있으면, 이미 늦었습니다. 안전은 **현재 위치**가 아니라 **현재 벡터장이
어디를 향하는가**의 문제입니다. 이것이 다음 절의 주제입니다.

---

## 2. Nagumo 정리 — 경계에서 벡터장이 안쪽을 향하면 된다

### 2.1 직관

C를 빠져나가려면 반드시 경계 ∂C를 **통과**해야 합니다 (연속 궤적이므로).
따라서 경계 위의 모든 점에서 "밖으로 나가는 방향의 속도 성분"이 없다면,
빠져나갈 방법이 없습니다.

```
                 ∇h (h가 증가하는 방향 = 안쪽)
                  ↑
   h < 0     ─────●─────────  ∂C = {h = 0}
  (위험)         /|\
                / | \
           ẋ₁ ✗  ẋ₂✓  ẋ₃ ✓        h > 0 (안전, 아래쪽이 안쪽이라 가정)
```

- ẋ₁: 경계에서 밖(h 감소 방향)을 향함 → **탈출 → 위반**
- ẋ₂: 경계에 접선 방향 → 경계를 따라 미끄러짐 → **OK**
- ẋ₃: 안쪽(h 증가 방향)을 향함 → **OK**

"밖으로 나가는 성분"은 ∇h와의 내적으로 잽니다. ḣ = ∇h(x)ᵀ ẋ 이므로:

> **Nagumo 조건 (부분 정리, 직관 버전).**
> 경계 위의 모든 점 x ∈ ∂C에서
>
> ḣ(x) = ∇h(x)ᵀ f(x, u) ≥ 0
>
> 이면 C는 전방 불변이다.

한 줄 요약: **경계에서 h가 감소하지만 않으면 절대 못 나간다.**

### 2.2 왜 "정리"가 필요한가 (미묘한 부분)

직관은 명확하지만 증명에는 함정이 있습니다. ḣ = 0인 접선 궤적이 경계를 무한히
따라가다가 ∇h = 0인 특이점을 만나면? 해가 유일하지 않으면? Nagumo(1942)는
이런 경우까지 처리해 "접촉 원뿔(tangent cone)에 벡터장이 들어 있으면 불변"
이라는 형태로 엄밀화했습니다. 이 repo 문서의 증명 스케치는
[SAFETY_THEORY.md §1.3.1](../SAFETY_THEORY.md)에 있습니다 (비교 보조정리 기반).

실무 관점에서 기억할 것 두 가지:

1. **정칙성 가정**: 경계에서 ∇h(x) ≠ 0 이어야 합니다 (h = 0인 곳에서 기울기가
   죽으면 "안쪽"이 정의가 안 됨). h = dist² - r² 꼴은 dist = r > 0에서 ∇h ≠ 0
   이므로 안전합니다.
2. **Nagumo는 조건을 경계에서만 겁니다.** 이게 실용적으로 왜 문제인지가 다음
   절의 출발점입니다.

---

## 3. Class-K 함수와 CBF 정의 — 왜 내부에서도 조건을 거는가

### 3.1 경계에서만 조건을 걸면 생기는 문제

Nagumo 조건 "∂C에서 ḣ ≥ 0"을 그대로 제어기로 구현한다고 합시다:

```
if h(x) == 0:   # 경계에 닿으면
    ḣ ≥ 0 을 강제
else:
    아무거나 해도 됨
```

문제 세 가지:

1. **bang-bang 개입**: h > 0인 동안 제약이 전혀 없다가 h = 0에서 갑자기
   ḣ ≥ 0을 강제 → 제어 입력이 불연속적으로 튐. 실제 로봇에서는 액추에이터
   한계 때문에 순간적인 방향 전환이 불가능합니다.
2. **측도 0의 조건**: 연속 상태 공간에서 "정확히 h = 0"은 실질적으로 감지
   불가능. 이산 시간 구현에서는 h = 0을 건너뛰고 h < 0으로 바로 떨어질 수
   있습니다.
3. **실현 가능성**: 경계 도달 시점에 접근 속도가 크면, 제어 권한(control
   authority)이 부족해서 ḣ ≥ 0 자체가 불가능할 수 있습니다.

해법: **경계에 가까워질수록 점진적으로 감속을 요구**하자. 이를 위한 도구가
class-K 함수입니다.

### 3.2 Class-K 함수

> **정의.** α: [0, a) → [0, ∞)가 **class-K** 라는 것은 (i) 연속, (ii) 순증가,
> (iii) α(0) = 0 인 것. 정의역이 ℝ 전체이고 α(-s) = -α(s)로 확장하면
> **extended class-K** (h < 0에서도 의미를 가짐).

대표 예: α(h) = γh (선형), α(h) = γh³ (경계 근처에서 관대, 멀리서 엄격).
이 repo의 class-K 상세는 [SAFETY_THEORY.md §1.5](../SAFETY_THEORY.md) 참조.

### 3.3 CBF 정의

> **정의 (Control Barrier Function).** 제어-어파인 시스템
> ẋ = f(x) + g(x)u 에 대해, h가 **CBF**라는 것은 어떤 extended class-K 함수
> α가 존재하여 모든 x에서
>
> **sup_u [ L_f h(x) + L_g h(x) u ] ≥ -α(h(x))**
>
> 를 만족하는 것. (L_f h = ∇hᵀf, L_g h = ∇hᵀg 는 Lie derivative, §4에서 계산)

그리고 실제 제어기는 매 순간 다음 조건을 만족하는 u를 고릅니다:

```
ḣ(x, u) ≥ -α(h(x))          … (CBF 조건)
```

### 3.4 이 조건이 하는 일: "여유 있는 감속" 스케줄

CBF 조건을 읽는 법: **h가 줄어드는 것 자체는 허용하되, 줄어드는 속도의 상한을
현재 여유분 h에 비례시킨다.**

```
  ḣ (h의 변화율)
   │
   │  허용 영역 (ḣ ≥ -αh)
   │ ░░░░░░░░░░░░░░░░░░░░
   │ ░░░░░░░░░░░░░░░░░░░░
 0 ┼──────────────────────────→ h
   │            ╱  기울기 -α
   │      ✗   ╱   ← 금지 영역: h가 작은데도
   │        ╱        빠르게 감소하는 것
   │      ╱
   │    ╱   h가 클 때(멀 때)는 ḣ가 크게 음수여도 됨
```

- **장애물에서 멀 때 (h 큼)**: -α(h)가 큰 음수 → ḣ가 꽤 음수여도 됨 →
  거의 제약 없음, 성능 자유.
- **가까울 때 (h 작음)**: -α(h) ≈ 0 → h가 거의 감소 못 함 → 부드럽게 감속.
- **경계에서 (h = 0)**: ḣ ≥ 0 → 정확히 Nagumo 조건 복원.

즉 CBF 조건은 Nagumo 조건의 **연속적 내부 확장**입니다. 경계에서 급브레이크를
밟는 대신, 접근하면서 미리미리 속도를 줄이는 스케줄을 강제합니다. α가 클수록
경계 근처에서만 개입(공격적), 작을수록 멀리서부터 개입(보수적)합니다.

보너스: h(0) < 0(이미 위반 상태)에서 시작해도, extended class-K와 비교
보조정리에 의해 h(t) → 0으로 **복귀**합니다 (asymptotic recovery). 이것이
zeroing CBF의 자기 회복 성질입니다.

### 3.5 미분 부등식으로 보는 정량적 의미

α(h) = γh (선형)일 때 CBF 조건은 ḣ ≥ -γh. 비교 보조정리를 쓰면:

```
h(t) ≥ h(0) · e^(-γt)
```

지수적으로 감쇠하는 **하한 곡선** 위에 h가 항상 머뭅니다. h(0) > 0이면
e^(-γt) > 0이므로 영원히 h(t) > 0. 이것이 CBF가 주는 전방 불변성 증명의
전부입니다 — 한 줄짜리 미분 부등식.

```
 h(t)
  │●  h(0)
  │ ╲
  │  ╲  실제 h(t)  (이 위 어디든 가능)
  │   ╲_______——————————
  │    ╲
  │     ╲ _ _  하한 h(0)e^(-γt)
  │          ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾
 0┼───────────────────────────→ t
     h(t)는 하한 아래로 절대 못 내려감
```

---

## 4. Lie derivative 손계산 — Differential Drive 완전 예제

CBF 조건을 실제 시스템에 쓰려면 L_f h, L_g h를 계산해야 합니다. 이 repo의
차동 구동(diffdrive) 기구학 모델
([mppi_controller/models/kinematic/differential_drive_kinematic.py](../../mppi_controller/models/kinematic/differential_drive_kinematic.py))로
처음부터 끝까지 손으로 계산해 봅니다.

### 4.1 시스템을 제어-어파인 형태로 쓰기

상태 x = [x, y, θ]ᵀ, 제어 u = [v, ω]ᵀ:

```
ẋ = v cos θ
ẏ = v sin θ
θ̇ = ω
```

제어-어파인 형태 ẋ = f(x) + g(x)u 로 분해하면:

```
f(x) = [0]        g(x) = [cos θ   0]
       [0]               [sin θ   0]
       [0]               [  0     1]
```

**기구학 모델은 drift가 없습니다** (f = 0). u = 0이면 로봇은 그 자리에 멈춰
있습니다. 이 사실이 뒤에서 "정지는 항상 안전한 백업"이라는 직관의 근거가
됩니다 (04_ADVANCED_SAFETY.md의 backup controller 참조).

### 4.2 Barrier 함수와 기울기

장애물 (x_o, y_o), 유효 반경 r_eff = r + margin:

```
h(x) = (x - x_o)² + (y - y_o)² - r_eff²
```

상태에 대한 기울기 (θ에는 의존하지 않음에 주의):

```
∇h = ∂h/∂x = [ 2(x - x_o),  2(y - y_o),  0 ]
```

### 4.3 Lie derivative 계산

**L_f h** — drift 방향으로의 h 변화율:

```
L_f h = ∇h · f(x) = [2(x-x_o), 2(y-y_o), 0] · [0, 0, 0]ᵀ = 0
```

기구학이므로 0. "가만히 있으면 h는 안 변한다."

**L_g h** — 각 제어 채널이 h를 미는 힘:

```
L_g h = ∇h · g(x)
      = [2(x-x_o), 2(y-y_o), 0] · [cos θ  0]
                                  [sin θ  0]
                                  [  0    1]
      = [ 2(x-x_o)cos θ + 2(y-y_o)sin θ ,   0 ]
```

두 가지 관찰:

1. **ω 성분이 0**: 회전은 h를 즉각적으로는 전혀 못 바꿉니다. 위치가 θ의
   함수가 아니기 때문. (회전이 안전에 영향을 주는 것은 *미래의* v 방향을
   바꾸기 때문 — 이것이 §7 relative degree 문제의 씨앗입니다.)
2. **v 성분의 기하학적 의미**: e_o = (x-x_o, y-y_o)를 장애물→로봇 벡터,
   heading 단위벡터를 d = (cos θ, sin θ)라 하면
   L_g h의 v 성분 = 2 e_o·d = 2‖e_o‖cos φ (φ = e_o와 d 사이 각).
   로봇이 장애물 **반대쪽**을 보면(cos φ > 0) 전진이 h를 늘리고,
   장애물 **쪽**을 보면(cos φ < 0) 전진이 h를 줄입니다. 당연한 물리가
   수식에서 그대로 나옵니다.

### 4.4 CBF 조건 완성

α(h) = αh (선형)로 두면:

```
ḣ = L_f h + L_g h · u
  = [2(x-x_o)cos θ + 2(y-y_o)sin θ] · v + 0 · ω  ≥  -α h
```

**숫자 예제.** 로봇 (0, 0, 0) (원점, +x 방향), 장애물 (2, 0), r_eff = 0.5,
α = 1.0:

```
h     = (0-2)² + 0² - 0.25 = 3.75
∇h    = [2(0-2), 0, 0] = [-4, 0, 0]
L_g h = [-4·cos0 + 0·sin0, 0] = [-4, 0]

CBF 조건:  -4v ≥ -1.0 × 3.75  ⟹  v ≤ 0.9375
```

해석: 정면에 2 m 떨어진 장애물을 향해 갈 때, 지금 이 순간 허용되는 최대
전진 속도는 0.9375 m/s. 더 다가가서 h = 1.0이 되면 (dist ≈ 1.118):

```
L_g h = [-2·1.118, 0] ≈ [-2.236, 0]
-2.236 v ≥ -1.0  ⟹  v ≤ 0.447
```

가까워질수록 허용 속도가 자동으로 줄어듭니다 — §3.4의 "여유 있는 감속"이
숫자로 확인됩니다.

이 계산은 repo의
[cbf_safety_filter.py](../../mppi_controller/controllers/mppi/cbf_safety_filter.py)
`_compute_lie_derivatives()`에 그대로 구현되어 있습니다:

```python
# cbf_safety_filter.py (발췌)
h = (x - obs_x) ** 2 + (y - obs_y) ** 2 - effective_r ** 2
Lf_h = 0.0  # kinematic, no drift
Lg_h = np.array([
    2.0 * (x - obs_x) * cos_theta + 2.0 * (y - obs_y) * sin_theta,
    0.0,
])
```

5D 동역학 모델(가속도 제어)과 Ackermann 모델의 Lie 미분 유도는
[SAFETY_THEORY.md §3](../SAFETY_THEORY.md)의 "동역학 Differential Drive (5-state)
Lie 미분", "Ackermann 모델 Lie 미분" 절에 있습니다.

### 4.5 look-ahead 트릭 (미리보기)

위 계산에서 L_g h의 ω 성분이 0이라 QP가 ω를 활용하지 못합니다. repo의
[clf_cbf_qp.py](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)는
**look-ahead 포인트** p̃ = p + d·[cos θ, sin θ]를 대신 제어합니다:

```
ṗ̃ = M(θ) u,   M(θ) = [cos θ  -d sin θ]
                      [sin θ   d cos θ]
```

M(θ)는 d > 0이면 가역(det = d)이므로 p̃는 **완전 구동(fully actuated)** 점이
되고, h를 p̃에 대해 정의하면 L_g h = 2 e_oᵀ M 의 두 성분이 모두 살아나
ω로도 안전에 기여할 수 있습니다 (near-identity diffeomorphism 기법).

---

## 5. CBF-QP와 해석해 — 단일 제약 투영 공식 유도

### 5.1 Safety filter 정식화

CBF 조건은 u에 대한 **선형 부등식**입니다. 성능 컨트롤러(MPPI, PID, 무엇이든)가
낸 u_nom을 "최소로 수정"해 안전하게 만드는 문제는 QP가 됩니다:

```
u* = argmin_u  ‖u - u_nom‖²
     s.t.      L_f h_i + L_g h_i · u ≥ -α h_i     (장애물 i마다)
               u_min ≤ u ≤ u_max
```

이것이 **safety filter** 패턴입니다:

```
              u_nom                u_safe
  [MPPI 등] ────────→ [CBF-QP 필터] ────────→ 로봇
   성능 담당            안전 담당 (최소 개입)
```

- 제약이 이미 만족되면: u* = u_nom (필터가 투명해짐, minimally invasive)
- 위반이면: 제약 경계로 **최소 거리 투영**

### 5.2 단일 제약 투영 공식 유도 (전체 과정)

제약 하나만 활성이라고 합시다. 제약을 g·u ≥ c 로 쓰면
(g = L_g h ∈ ℝ^nu 행벡터, c = -L_f h - αh 스칼라). u_nom이 위반
(g·u_nom < c)일 때, 최적해는 제약 **경계** g·u = c 위에 있습니다
(경계 안쪽으로 더 들어갈 이유가 없음 — 목적 함수가 u_nom에서 멀어질수록
커지므로). 따라서 등식 제약 최소화:

```
min_u  ½‖u - u_nom‖²   s.t.  g·u = c
```

라그랑지안: L(u, μ) = ½‖u - u_nom‖² + μ(c - g·u)

정류 조건:

```
∂L/∂u = (u - u_nom) - μ gᵀ = 0    ⟹  u = u_nom + μ gᵀ
∂L/∂μ = c - g·u = 0               ⟹  g·u = c
```

첫 식을 둘째 식에 대입:

```
g·(u_nom + μ gᵀ) = c
μ = (c - g·u_nom) / (g·gᵀ) = (c - g·u_nom) / ‖g‖²
```

> **단일 제약 투영 공식:**
>
> u* = u_nom + gᵀ · (c - g·u_nom) / ‖g‖²

기하학적으로: u_nom을 초평면 {u : g·u = c}에 **수직으로 투영**한 것.
(c - g·u_nom)/‖g‖가 초평면까지의 부호 있는 거리이고, gᵀ/‖g‖가 법선 방향입니다.

```
        u 공간 (제어 공간)
                            g·u ≥ c  (안전 반평면)
        ░░░░░░░░░░░░░░░░░░
        ░░░░░ u* ░░░░░░░░░
        ──────●──────────────  {g·u = c}
              │  ↑ 법선 방향 gᵀ/‖g‖
              │
              ● u_nom  (위반 상태)
```

**가중 노름 버전.** 목적 함수가 (u-u_nom)ᵀP(u-u_nom)이면 (P ≻ 0, 채널별
수정 비용 차등), 같은 유도로:

```
u* = u_nom + P⁻¹gᵀ · (c - g·u_nom) / (g P⁻¹ gᵀ)
```

### 5.3 repo의 fast path와 대응

이 공식이 그대로
[clf_cbf_qp.py](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)의
`CBFCLFQPSolver.solve()` 해석적 fast path에 구현되어 있습니다:

```python
# clf_cbf_qp.py — "(b) 단일 CBF 활성 → 등식 투영 (closed-form)"
g, c = A_cbf[i], float(b_cbf[i])
Pinv_g = np.linalg.solve(P, g)
denom = float(g @ Pinv_g)
u_proj = u_nom + Pinv_g * (c - float(g @ u_nom)) / denom
```

솔버의 전체 전략 (같은 파일 모듈 docstring):

1. **제약 비활성** → `u* = clip(u_nom)` (계산량 ≈ 0)
2. **CLF만 활성 (soft)** → 닫힌형 tradeoff 해 (§6)
3. **단일 CBF 활성** → 위의 투영 공식
4. **다중 제약 / 제어 경계 동시 활성** → `scipy.optimize.minimize(SLSQP)` 폴백

이 구조 덕분에 벤치마크에서 QP 계열의 평균 solve time이 **0.09–0.15 ms**로,
MPPI(1.8–3.0 ms)의 약 1/20입니다
([docs/CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) §6). 대부분의
스텝에서 제약이 비활성이거나 하나만 활성이라 해석해로 끝나기 때문입니다.

주의할 것: 투영해가 **제어 경계를 벗어나면** fast path를 포기하고 SLSQP로
갑니다 (코드의 `in_bounds` 체크). 경계가 활성인 순간 KKT 구조가 달라져서
단순 클리핑은 최적해가 아니기 때문입니다 (P가 대각일 때만 클리핑 = 투영).
KKT 조건의 전체 분석은 [SAFETY_THEORY.md §3](../SAFETY_THEORY.md) "KKT 조건
분석" 절 참조.

---

## 6. CLF와의 결합 — 안전 > 수렴의 우선순위 설계

### 6.1 CLF: 수렴의 barrier 버전

CBF가 "집합에서 안 나가기"라면, Control Lyapunov Function(CLF)은 "목표로
수렴하기"입니다. V(x) ≥ 0, V(목표) = 0인 함수에 대해:

```
V̇(x, u) = L_f V + L_g V · u ≤ -γ(V)      (γ는 class-K)
```

를 만족하는 u를 계속 쓰면 V → 0, 즉 목표로 수렴합니다. 구조가 CBF와
완전히 대칭입니다:

| | CBF | CLF |
|---|---|---|
| 함수 | h ≥ 0 유지 | V → 0 수렴 |
| 조건 | ḣ ≥ -α(h) (하한) | V̇ ≤ -γ(V) (상한) |
| 의미 | 나쁜 집합 회피 | 좋은 점으로 끌림 |

### 6.2 충돌과 slack — 안전이 항상 이긴다

문제: 목표가 장애물 뒤에 있으면 두 조건이 **동시에 만족 불가능**할 수
있습니다 (수렴하려면 다가가야 하는데 안전이 막음). 해법은 CLF에만 slack
변수 δ를 주는 것:

```
min_{u, δ}   ‖u - u_nom‖²_P + λ_clf · δ²
s.t.         L_f h + L_g h·u ≥ -α(h)            ← hard  (절대 완화 안 됨)
             L_f V + L_g V·u ≤ -γ(V) + δ        ← soft  (δ만큼 완화 가능)
             u_min ≤ u ≤ u_max,  δ ≥ 0
```

- **CBF는 hard**: 어떤 경우에도 완화되지 않음 → 안전 보장 유지.
- **CLF는 soft**: 필요하면 δ > 0으로 수렴을 일시 포기. λ_clf가 클수록
  "웬만하면 수렴도 지켜라"라는 압력.

이것이 "**안전 > 수렴**" 우선순위의 표준 인코딩입니다 (Ames et al. 2017).
직관적 시나리오:

```
   목표 ★
        │        로봇이 장애물을 우회하는 동안 V는 일시적으로
   ┌────┼────┐   증가할 수 있음 (δ > 0). CBF가 hard라서
   │ 장애물  │   "장애물 뚫고 직진"은 어떤 λ_clf에서도 불가.
   └─────────┘
        ●
       로봇
```

### 6.3 repo 구현: soft CLF의 닫힌형 해

[clf_cbf_qp.py](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)는 slack을
목적 함수에 흡수해 CLF-only 케이스를 닫힌형으로 풉니다. δ*(u) = max(0, a·u - b)
를 대입하면 1차원 문제로 축소되고 (a = L_g V 방향), s = a·u_nom - b > 0일 때:

```python
# clf_cbf_qp.py — "(a) CLF-tradeoff 무제약 최적해"
u_cand = u_nom - (lambda_clf * s / (1 + lambda_clf * q)) * Pinv_a   # q = aᵀP⁻¹a
delta_cand = s / (1 + lambda_clf * q)
```

λ_clf → ∞ 극한에서 u_cand는 CLF 등식 제약 투영(§5.2 공식과 동일 형태)으로,
λ_clf → 0 극한에서 u_nom으로 수렴하는 부드러운 보간입니다. slack 방법과
optimal-decay 방법의 비교는 [SAFETY_THEORY.md §9](../SAFETY_THEORY.md) "Slack
Variable 방법과의 비교" 절에 있습니다.

`CLFCBFQPController`(CLF+CBF)와 `CBFOnlyQPController`(명목 pure-pursuit +
CBF만)가 이 solver를 공유하며, 벤치마크에서 두 방식의 성능 차이를 직접 비교할
수 있습니다 ([CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) §5).

---

## 7. Relative Degree 문제와 HOCBF 개요

### 7.1 문제: 제어가 h에 "안 보일" 때

§4.3에서 diffdrive 기구학의 L_g h는 v 채널에 살아 있었습니다. 그런데 **동역학
모델** (5D 상태 [x, y, θ, v, ω], 제어 u = [a, α] 가속도)에서는:

```
h는 위치만의 함수 → ḣ = ∇h · ẋ 는 속도(v, ω)의 함수 → 가속도 u가 안 나타남!
L_g h = 0   (모든 u에 대해)
```

CBF 조건 ḣ + αh ≥ 0에 u가 없으므로 **제약이 제어를 전혀 구속하지 못합니다**.
u가 나타나려면 한 번 더 미분해야 합니다: ḧ에는 가속도가 등장. 이때 h의
**relative degree가 2**라고 합니다 (u가 나타날 때까지 미분해야 하는 횟수).

물리적으로: 가속도로 제어하는 차는 "지금 속도"를 즉시 못 바꾸므로, 위치
barrier만 보고는 늦습니다. **속도까지 포함한 예측적 barrier**가 필요합니다.

### 7.2 HOCBF: barrier의 cascade

Exponential/High-Order CBF (Xiao & Belta 2019)의 아이디어는 barrier를
연쇄적으로 쌓는 것입니다:

```
ψ₀ = h                          (위치 여유)
ψ₁ = ψ̇₀ + λ₁ψ₀                 ("여유 있게 접근 중인가" — 속도 포함)
제약: ψ̇₁ + λ₂ψ₁ ≥ 0            (여기에는 u가 등장 — rd 2 소진)
```

ψ₁ ≥ 0를 유지하면 ψ₀ = h ≥ 0가 따라오는 구조입니다 (각 단계가 §3.5의
지수 하한 논리로 연결됨). "위치가 안전한가"를 "접근 속도가 위치 여유에 비해
과하지 않은가"로 한 단계 리프팅한 것으로 읽으면 됩니다.

repo 구현은
[hocbf_cost.py](../../mppi_controller/controllers/mppi/hocbf_cost.py) (이산
유한차분 cascade, MPPI 비용용)이고, `relative_degree=1`로 두면 표준 이산 CBF로
정확히 축약됩니다 (파일 docstring의 등가성 유도 참조).
[clf_cbf_qp.py](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)의 동역학
분기도 `h_e = ḣ + λ·h` 1단 cascade를 사용합니다.

**실증**: 5D 가속도 제어 벤치마크에서 1차 CBF 비용들은 min clearance
0.039–0.199 m에 그친 반면 HOCBF-MPPI는 0.282 m를 달성했습니다
([CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) §5.2) — "제약이
제어에 실제로 반응하는가"의 차이입니다.

이론 상세(정의, λ 튜닝, 전방 불변성 증명, 벤치마크)는
[SAFETY_THEORY.md §16 "HOCBF (High-Order CBF)"](../SAFETY_THEORY.md)를 보세요
(§1.6에는 상대 차수 개념의 짧은 포인터가 있습니다).

---

## 8. CBF의 실패 모드와 이 repo의 해법 매핑

CBF는 "조건이 만족되는 한" 안전을 보장합니다. 실무에서 깨지는 지점은 그
전제들입니다. 실패 모드 4가지와, 각각을 겨냥한 이 repo의 구현을 매핑합니다.

### 8.1 실패 모드 요약표

| 실패 모드 | 원인 | 증상 | repo 해법 |
|---|---|---|---|
| Deadlock | 목표-장애물-로봇 대칭 정렬 | 안전하지만 전진 못 함 | 접선 투영 휴리스틱 ([clf_cbf_qp.py](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)), DRPA-MPPI ([drpa_mppi.py](../../mppi_controller/controllers/mppi/drpa_mppi.py)) |
| Feasibility 상실 | 제어 한계 + 다중 제약 충돌 | QP infeasible → 필터 무력화 | Optimal-Decay CBF ([optimal_decay_cbf_filter.py](../../mppi_controller/controllers/mppi/optimal_decay_cbf_filter.py)) |
| 이산화 오차 | 연속 조건을 dt 간격으로만 체크 | 스텝 사이 미세 침투 | safety_margin 흡수, 이산 CBF 정식화 (§9, [cbf_cost.py](../../mppi_controller/controllers/mppi/cbf_cost.py)) |
| 모델 불확실성 | f, g 오차 / 프로세스 노이즈 | ḣ 예측이 틀려 위반 | Robust CBF ([robust_cbf_margin.py](../../mppi_controller/controllers/mppi/robust_cbf_margin.py)), Risk-Aware CBF ([stochastic_cbf.py](../../mppi_controller/controllers/mppi/stochastic_cbf.py)), PR-MPPI ([parameter_robust_mppi.py](../../mppi_controller/controllers/mppi/parameter_robust_mppi.py)) |

### 8.2 Deadlock (head-on 대칭)

로봇–장애물–목표가 일직선이면, CBF 필터는 "장애물 방향 성분 제거"만 하므로
u_safe ≈ 0이 되어 멈춥니다. 안전 위반은 아니지만 **liveness 실패**입니다.

```
   ● 로봇 →→  ⊘ 장애물  ★ 목표
      u_nom은 장애물 정면 방향 → 투영하면 남는 성분이 0
      좌우 대칭이라 어느 쪽으로 돌지 결정 근거도 없음
```

- **repo 해법 1 — 접선 투영 + 히스테리시스**:
  [clf_cbf_qp.py](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)의
  `_project_pdot_around_obstacles()`는 명목 속도에서 반경 방향 위반 성분을
  제거한 뒤, 접선 성분이 작으면 `_detour_side`(±1 기억)로 **일관된 우회
  방향의 최소 접선 속도를 주입**합니다. 대칭을 인위적으로 깨는 것.
  중요한 설계 노트: 이 투영은 *명목 제어*에만 적용되는 휴리스틱이고, 안전
  보장은 여전히 QP의 hard CBF 제약이 담당합니다 (docstring에 명시).
- **repo 해법 2 — DRPA-MPPI**:
  [drpa_mppi.py](../../mppi_controller/controllers/mppi/drpa_mppi.py)는 horizon
  끝 이동량으로 **정체를 감지**하고, 반발 포텐셜 F_rep를 비용에 동적으로
  추가 + 탐색 노이즈 증폭으로 탈출합니다. 필터 레벨이 아니라 최적화(샘플링)
  레벨의 해법.

### 8.3 Feasibility 상실

장애물이 밀집하거나 제어 한계가 빡빡하면 {u : 모든 CBF 제약 만족} ∩
[u_min, u_max] = ∅ 가 될 수 있습니다. 이때 표준 QP는 해가 없고, 구현은
"best-effort"로 후퇴합니다 (clf_cbf_qp.py의 `slsqp_infeasible` 분기 — 위반이
가장 적은 후보 반환).

- **repo 해법 — Optimal-Decay CBF**:
  [optimal_decay_cbf_filter.py](../../mppi_controller/controllers/mppi/optimal_decay_cbf_filter.py)
  (Zeng et al. 2021)는 decay rate에 최적화 변수 ω를 곱합니다:

  ```
  min  ‖u - u_mppi‖² + p_sb(ω - 1)²
  s.t. L_f h + L_g h·u + α·ω·h ≥ 0,   0 ≤ ω ≤ 1
  ```

  ω = 1이면 표준 CBF, 부족하면 ω < 1로 **점진적 완화**(graceful degradation).
  h > 0인 한 ω = 0으로 두면 항상 feasible하므로 QP가 절대 실패하지 않고,
  ω 값 자체가 "지금 안전 제약이 얼마나 무리인가"의 진단 지표가 됩니다
  ([SAFETY_THEORY.md §9](../SAFETY_THEORY.md) "ω 해석" 절).

### 8.4 이산화 오차

연속 시간 조건 ḣ ≥ -αh를 만족하는 u를 골라도, 실제로는 dt 동안 **일정하게
유지(zero-order hold)** 하므로 스텝 중간에 h가 예측보다 더 떨어질 수
있습니다 (곡률이 있는 궤적에서 특히).

- **repo 해법**: (i) `safety_margin`으로 r_eff를 부풀려 적분 오차를 흡수
  ([clf_cbf_qp.py](../../mppi_controller/controllers/mppi/clf_cbf_qp.py)
  docstring: "CBF는 연속시간 조건 → 이산 적분 오차는 safety_margin으로 흡수"),
  (ii) 애초에 이산 시간으로 정식화한 CBF를 쓰기 — §9.

### 8.5 모델 불확실성

CBF 조건의 ḣ은 모델 f, g로 계산합니다. 모델이 틀리면 "안전하다고 판정한 u"가
실제로는 위험할 수 있습니다. **벤치마크가 이것을 극적으로 보여줍니다**:
프로세스 노이즈 하에서 순정 discrete CBF-MPPI는 **3/3 시드 모두 충돌**
(min clearance -0.073±0.062 m)한 반면, 명시적 불확실성 마진을 가진 방법들만
무충돌이었습니다 ([CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md)
§5.3, §6).

- **Robust CBF** ([robust_cbf_margin.py](../../mppi_controller/controllers/mppi/robust_cbf_margin.py),
  Jankovic 2018): 유계 외란 ‖w‖ ≤ w_max의 **최악 케이스**만큼 조건을 강화 —
  `ḣ + α(h) - ‖∇h·M‖·w_max ≥ 0`. 결정론적 보장, w_max에 선형인 보수성.
- **Risk-Aware CBF** ([stochastic_cbf.py](../../mppi_controller/controllers/mppi/stochastic_cbf.py)
  `RiskAwareCBFCost`, Black CDC 2023): 확률적 노이즈에 대해
  P(위반) ≤ ρ가 되도록 시간 증가 마진 `margin(t) = √(2t)·η·erfinv(1-2ρ)` 요구.
  ρ가 해석 가능한 "위험 다이얼" (04_ADVANCED_SAFETY.md §7에서 데이터로 논의).
- **PR-MPPI** ([parameter_robust_mppi.py](../../mppi_controller/controllers/mppi/parameter_robust_mppi.py)):
  파라미터 불확실성을 입자 belief로 온라인 추정하며 다중 모델 가설로 rollout —
  barrier가 아니라 예측 모델 자체를 robust하게.

> **공부 포인트**: 네 실패 모드는 서로 독립적이라 해법도 조합됩니다. 예:
> HOCBF 비용(rd 문제) + safety_margin(이산화) + RiskAware 마진(노이즈) +
> DRPA(정체 탈출)를 한 컨트롤러에 겹칠 수 있습니다. 계층화 전략은
> [SAFETY_THEORY.md §15](../SAFETY_THEORY.md) "복합 안전 전략 추천" 참조.

---

## 9. Discrete-time CBF — MPPI 비용으로 쓸 때

### 9.1 왜 이산 버전이 필요한가

MPPI는 궤적을 dt 간격의 이산 rollout으로 평가합니다. 연속 조건 ḣ ≥ -αh를
그대로 쓸 수 없으므로, 이산 대응물을 정의합니다 (Agrawal & Sreenath 2017):

```
h(x_{k+1}) - h(x_k) ≥ -α_d · h(x_k),    0 < α_d ≤ 1

⟺  h(x_{k+1}) ≥ (1 - α_d) · h(x_k)
```

읽는 법: **한 스텝에 h는 최대 α_d 비율만큼만 줄어들 수 있다.** 귀납적으로

```
h(x_k) ≥ (1 - α_d)^k · h(x_0)
```

이고 (1 - α_d)^k > 0이므로 h(x_0) > 0 ⟹ h(x_k) > 0 forever. §3.5의 지수 하한
e^(-γt)의 이산판이 기하 수열 (1-α_d)^k인 셈입니다 (α_d ≈ γ·dt로 대응).

### 9.2 연속시간과의 차이 — 세 가지 함정

1. **α의 스케일이 다릅니다.** 연속 α는 [1/s] 단위의 rate, 이산 α_d는 무차원
   비율(0~1]. 같은 "α=0.1"이라도 dt에 따라 의미가 완전히 다릅니다.
   대응 관계는 α_d ≈ α·dt. repo의
   [hocbf_cost.py](../../mppi_controller/controllers/mppi/hocbf_cost.py)
   docstring이 이 등가성을 정확히 적어 놓았습니다: rd=1 cascade
   `C_t = (1/dt)[h_{t+1} - (1 - λ₁dt)h_t]`에서 λ₁ = α_d/dt로 두면
   `ControlBarrierCost(alpha=α_d)`와 일치.
2. **α_d ≤ 1 이 필수입니다.** α_d > 1이면 (1-α_d) < 0이라 h가 음수로
   "허용"되어 버립니다. 연속 시간에는 없는 제약.
3. **스텝 내부는 보호되지 않습니다.** 조건은 격자점 x_k에서만 성립. 스텝
   사이에서 장애물을 스치는 것은 margin으로 커버해야 합니다 (§8.4와 동일
   논점).

### 9.3 MPPI 비용으로의 변환

MPPI에는 hard 제약이 없으므로, 이산 CBF 조건 **위반량을 벌점**으로 바꿉니다.
repo의 [cbf_cost.py](../../mppi_controller/controllers/mppi/cbf_cost.py)
`ControlBarrierCost`:

```python
# cbf_cost.py — Discrete CBF 조건: h(x_{t+1}) - (1-alpha)*h(x_t) >= 0
cbf_condition = h[:, 1:] - (1.0 - self.cbf_alpha) * h[:, :-1]   # (K, N)
cost = weight * Σ_t max(0, -cbf_condition)
```

포인트:

- **위치가 아니라 "감소율"에 벌점**을 줍니다. 단순 거리 벌점(장애물 가까우면
  비용)과 달리, *멀리 있어도 빠르게 접근하는* 궤적을 초기에 걸러냅니다 —
  §3.4의 예측적 감속 논리가 샘플 가중치에 이식된 것.
- 이것은 **soft** 안전입니다. weight가 유한하면 추적 이득이 벌점을 이길 수
  있어 보장이 아닙니다. 실제로 벤치마크에서 순정 CBF-MPPI는 결정론적
  시나리오에서도 clearance -0.030 m로 스쳤습니다
  ([CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) §5.1). weight/α
  민감도 분석은 [SAFETY_THEORY.md §2](../SAFETY_THEORY.md) "파라미터 민감도" 절.
- 보장이 필요하면 층을 올립니다: per-step 강제(Shield), trajectory 검증
  (Gatekeeper) — 이 스펙트럼이 다음 문서
  [04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md)의 주제입니다.

변형들: `HorizonWeightedCBFCost`(시간 가중), `HardCBFCost`(위반 시 무한대
비용) — [SAFETY_THEORY.md §2](../SAFETY_THEORY.md) 참조.

---

## 10. 연습문제

**문제 1 — h 설계와 불변성.**
반경 0.4 m 장애물이 (1, 1)에 있다. (a) h(x)를 두 가지로 설계하라:
h₁ = dist² - r², h₂ = dist - r. (b) 각각의 ∇h를 구하고, 경계에서 ∇h ≠ 0인지
확인하라. (c) h₂는 dist = 0(장애물 중심)에서 무엇이 문제인가?
*힌트: h₂의 기울기는 dist → 0에서 정의되지 않지만 경계(dist = r)에서는 단위
벡터로 잘 정의된다. h₁은 어디서나 매끄럽다 — repo가 h₁ 꼴을 쓰는 이유
중 하나.*

**문제 2 — Lie derivative 손계산.**
로봇 상태 (1, 0, π/2) (위쪽을 봄), 장애물 (1, 2), r_eff = 0.5, α = 2.0.
§4의 과정을 따라 (a) h, (b) L_g h, (c) 허용되는 최대 전진 속도 v_max를
구하라. (d) 로봇이 (1, 0, 0) (오른쪽을 봄)이면 CBF 조건이 v를 구속하는가?
*답: (a) h = 4 - 0.25 = 3.75. (b) ∇h = [0, -4, 0], L_g h = [-4·sin(π/2)... 주의:
v 성분 = 2(x-x_o)cosθ + 2(y-y_o)sinθ = 0·0 + (-4)·1 = -4. (c) -4v ≥ -2·3.75
⟹ v ≤ 1.875. (d) θ=0이면 v 성분 = 0·1 + (-4)·0 = 0 → 어떤 v도 h를 안 바꿈,
조건은 0 ≥ -7.5로 항상 참 → 구속 없음 (장애물이 정측면이므로 당연).*

**문제 3 — 투영 공식.**
u_nom = [1.0, 0.3], 제약 g·u ≥ c with g = [-2, 0], c = -1 (즉 -2v ≥ -1).
(a) u_nom이 제약을 위반하는지 확인하라. (b) §5.2 공식으로 u*를 계산하라.
(c) ω 성분이 왜 안 바뀌었는지 기하학적으로 설명하라.
*답: (a) -2·1 = -2 < -1 위반. (b) u* = u_nom + gᵀ(c - g·u_nom)/‖g‖²
= [1, 0.3] + [-2, 0]·(-1+2)/4 = [0.5, 0.3]. (c) 법선 gᵀ가 v축 방향이므로 투영은
v만 수정 — ω는 제약면과 평행한 좌표.*

**문제 4 — 이산 CBF의 기하 수열 하한.**
α_d = 0.1, dt = 0.05 s, h(x_0) = 2.0. (a) 이산 CBF 조건이 매 스텝 등식으로
성립할 때(최대 속도로 접근할 때) h(x_k)를 k의 함수로 쓰고, h가 0.1 이하로
떨어지는 데 몇 스텝(몇 초)이 걸리는지 구하라. (b) 같은 감쇠를 연속 조건
ḣ = -αh로 재현하려면 α는 얼마여야 하나?
*답: (a) h_k = 2·(0.9)^k, 2·0.9^k ≤ 0.1 ⟹ k ≥ ln(0.05)/ln(0.9) ≈ 28.4 →
29스텝 ≈ 1.45 s. (b) (1-α_d) = e^(-α·dt) ⟹ α = -ln(0.9)/0.05 ≈ 2.11 /s
(근사 α ≈ α_d/dt = 2.0과 비교).*

**문제 5 — 실패 모드 진단.**
다음 각 관찰에 대해 §8의 실패 모드 중 무엇인지 진단하고, repo의 어느 모듈로
대응할지 답하라.
(a) 시뮬레이션 dt를 0.05→0.2로 키웠더니 가끔 clearance가 살짝 음수가 된다.
(b) 좁은 통로에서 QP solver가 `slsqp_infeasible`을 반환하기 시작했다.
(c) 로봇이 장애물 정면 30 cm 앞에서 멈춘 채 좌우로 미세 진동만 한다.
(d) 실기 바퀴 반경이 모델보다 5% 크고, 시뮬레이션에선 없던 위반이 실기에서
발생한다.
*답: (a) 이산화 오차 → safety_margin 증가 또는 dt 축소, cbf_cost.py의 α_d
재튜닝. (b) feasibility 상실 → optimal_decay_cbf_filter.py. (c) deadlock →
clf_cbf_qp.py 접선 투영 또는 drpa_mppi.py. (d) 모델 불확실성 →
robust_cbf_margin.py (유계 오차) 또는 parameter_robust_mppi.py (온라인 추정).*

---

## 11. 부록 — 더 공부하기 위한 자료

> 본문을 다 읽은 뒤의 자습 가이드. 링크는 2026-07 기준이며, 확신할 수 없는
> arXiv ID는 싣지 않고 제목+학회만 표기했습니다. 고급 안전(HJ/gatekeeper/
> 확률 보장) 쪽 자료는 [04_ADVANCED_SAFETY.md §9](04_ADVANCED_SAFETY.md)
> 부록이 담당합니다.

### 11.1 주석 달린 핵심 레퍼런스

**필독 (이 순서대로):**

1. **Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada,
   "Control Barrier Functions: Theory and Applications", ECC 2019.**
   — 이 분야의 표준 튜토리얼 논문. 본 문서 §1–§6의 원전. Nagumo부터 CBF-QP,
   CLF 결합까지 30페이지에 압축. arXiv:1903.11199.
2. **Ames, Xu, Grizzle, Tabuada, "Control Barrier Function Based Quadratic
   Programs for Safety Critical Systems", IEEE TAC 2017.**
   — CBF-CLF-QP 프레임워크 원 논문. slack 우선순위 설계(§6)의 출처.
3. **Xiao & Belta, "Control Barrier Functions for Systems with High Relative
   Degree", CDC 2019** (확장판: "High-Order Control Barrier Functions",
   IEEE TAC 2022). — §7 HOCBF의 원전.

**주제별 심화:**

4. Agrawal & Sreenath, "Discrete Control Barrier Functions for Safety-Critical
   Control of Discrete Systems with Application to Bipedal Robot Navigation",
   RSS 2017 — §9 이산 CBF의 원전.
5. Zeng, Zhang, Sreenath, "Safety-Critical Model Predictive Control with
   Discrete-Time Control Barrier Function", ACC 2021 — optimal-decay ω 기법.
6. Jankovic, "Robust control barrier functions for constrained stabilization
   of nonlinear systems", Automatica 2018 — robust 마진.
7. Gurriet et al., "Scalable Safety-Critical Control of Robotic Systems",
   2020 — backup set / gatekeeper 계열 (다음 문서에서 상세히).

8. Dawson, Gao, Fan, "Safe Control with Learned Certificates: A Survey of
   Neural Lyapunov, Barrier, and Contraction Methods", IEEE T-RO 2023
   (arXiv:2202.11762) — 학습 기반 인증서(§11.2 동향 1)의 표준 서베이.
   repo의 `neural_cbf_cost.py`가 이 계열.

### 11.2 최근 연구 동향 (2024–2026)

1. **Neural CBF: 합성 + 검증의 결합.** h(x)를 신경망으로 학습하되
   Lipschitz 상수/SMT 솔버로 사후 검증해 "학습했지만 보장은 형식적"인
   인증서를 만드는 흐름. 입문은 Dawson 서베이(위 8번), repo 구현은
   `neural_cbf_cost.py`/`neural_cbf_filter.py`.
2. **생성 모델 정책과 CBF의 결합.** diffusion/flow 정책의 denoising 과정에
   barrier 조건을 주입해 "생성되는 궤적 자체가 안전"하도록 만드는 계열
   (대표: SafeDiffuser, arXiv:2306.00148). 이 repo 관점에선
   [05_GENERATIVE_MODELS_FOR_CONTROL.md](05_GENERATIVE_MODELS_FOR_CONTROL.md)
   부록 B와 만나는 지점.
3. **고차원 시스템으로 확장.** 매니퓰레이터/휴머노이드의 whole-body 안전
   (자기충돌·관절한계를 다수 barrier로), reduced-order model 위에 CBF를 걸고
   전신 제어로 전파하는 Ames 그룹 계열. relative degree 문제(§7)가 실전에서
   왜 중요한지 보여주는 응용처.
4. **데이터 기반 불확실성과 CBF.** conformal prediction 마진(repo의
   Conformal-CBF), GP 잔차 마진, 분포 강건(distributionally robust) CBF 등
   "모델 오차를 데이터로 정량화해 barrier에 반영"하는 흐름 —
   SAFETY_THEORY.md §12, §17–19와 직결.
5. **툴박스 생태계 성숙.** cbfkit(arXiv:2404.07158) 같은 범용 CBF 툴박스가
   등장, 이 repo도 그중 5종을 numpy로 포팅했다
   ([CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md)).

### 11.3 오픈소스 생태계

| 이름 | 링크 | 언어 | 특징 | 이 repo와의 관계 |
|---|---|---|---|---|
| cbfkit | github.com/bardhh/cbfkit | Python/JAX | CBF-CLF-QP, stochastic/risk-aware CBF, ROS2 | 5종 기법의 원본 (§16–20 포팅 출처) |
| safe_control | github.com/tkkim-robot/safe_control | Python | CBF-QP, MPC-CBF 구현체 | `mppi_vs_safe_control_benchmark.py`로 직접 비교 |
| safe-control-gym | github.com/learnsyslab/safe-control-gym | Python/PyBullet | 안전 RL/제어 벤치마크 환경 (quadrotor 등) | 외부 검증 환경 후보 |
| CBF 예제 (Ames 그룹 AMBER Lab) | github.com/HybridRobotics (조직) | MATLAB/Python | CBF-MPC, 이족보행 등 응용 예제 | 응용 사례 참고 |
| 이 repo | — | Python/numpy | CBF 비용/필터/QP 27종 + MPPI 결합 | `mppi_controller/controllers/mppi/` |

### 11.4 강의/영상

- Aaron Ames의 ECC 2019 tutorial 세션 및 Caltech AMBER Lab 강의 영상 —
  "safety as invariance" 관점의 직관 설명이 좋습니다.
- UC Berkeley Hybrid Systems Lab (Claire Tomlin) 강의 노트 — HJ reachability
  관점과의 대비 (04_ADVANCED_SAFETY.md §3의 배경).
- MIT Underactuated Robotics (Tedrake) 중 Lyapunov/verification 장 —
  CLF 쪽 기초를 보강할 때.

### 11.5 자주 궁금한 점 → 어디를 볼까

| 궁금한 점 | 내부 자료 | 외부 자료 |
|---|---|---|
| 움직이는 장애물은 어떻게? | SAFETY_THEORY §6 C3BF, §7 DPCBF | collision cone CBF 논문 계열 |
| 장애물이 여러 개면? | `clf_cbf_qp.py` (제약 행 추가), §5 | Ames TAC 2017 §V |
| 로봇이 장애물 앞에서 멈춰버림 (deadlock) | §8 실패 모드 1 + DRPA-MPPI (`drpa_mppi.py`) | reactive planner deadlock 문헌 |
| QP가 infeasible이 되면? | §8 실패 모드 2 + SAFETY_THEORY §9 optimal-decay ω | Zeng ACC 2021 |
| 모델이 부정확한데 보장이 유효한가? | SAFETY_THEORY §19 Robust CBF, PR-MPPI | Jankovic Automatica 2018 |
| 노이즈 아래 확률적 보장을 원함 | SAFETY_THEORY §17–18 + 04 §4 | Black et al. 계열 |
| 가속도 제어(rd=2) 모델이면? | §7 + SAFETY_THEORY §16 HOCBF | Xiao & Belta TAC 2022 |
| h를 손으로 설계하기 어려움 | `neural_cbf_cost.py` | Dawson 서베이 (arXiv:2202.11762) |
| CBF를 MPC/MPPI 제약으로 넣으려면? | §9 discrete CBF + `cbf_cost.py` | Zeng ACC 2021 |

### 11.6 이 repo 내부

- [docs/SAFETY_THEORY.md](../SAFETY_THEORY.md) — 22종 안전 기법 레퍼런스
  (§1 기초 유도, §3 QP 상세, §15 선택 가이드).
- [docs/CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) — 본 문서에서
  인용한 CBF 계열 벤치마크의 전체 결과와 분석.
- 다음 편: [docs/study/04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md) — 안전
  보장의 스펙트럼 (soft 벌점 → 필터 → 예측 검증 → HJ/확률적 보장).

---

*작성: 2026-07 — learning_mppi 공부 자료 시리즈. 코드 인용은 작성 시점의
repo 상태 기준입니다.*
