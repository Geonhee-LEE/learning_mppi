# 09. 두 개의 "잔차(residual)" — 잔차 제어와 잔차 동역학

> **학습 시리즈 9편**: 이 저장소에는 이름은 같지만 **사는 층(layer)이 다른**
> 두 가지 "residual" 개념이 있다. 하나는 **컨트롤러 층**의 잔차 제어
> (Residual-MPPI, 25번째 변형), 다른 하나는 **모델 층**의 잔차 동역학
> (`ResidualDynamics`). 이 문서는 둘을 명확히 구분하고, 각각을 수식 → 코드
> 발췌 → 왜 이렇게 → 흔한 실수 순서로 해설한다.
>
> **읽는 법**: 06–08편과 같은 **코드 워크스루** 형식이다. 모든 코드 발췌에는
> `파일경로:줄번호`를 붙였고, 작성 시점(2026-07)의 실제 소스에서 그대로 가져와
> 줄번호를 하나하나 검증했다. 이론적 배경은 [docs/MPPI_THEORY.md §24](../MPPI_THEORY.md)에
> 위임하고, 여기서는 "코드가 어떻게 생겼고 왜 그렇게 생겼는지"에 집중한다.
>
> **선행 학습**: [02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md)(MPPI 기본
> 업데이트 식), [05_GENERATIVE_MODELS_FOR_CONTROL.md](05_GENERATIVE_MODELS_FOR_CONTROL.md)
> (학습된 제안 분포), [07_CODE_WALKTHROUGH_VARIANTS.md](07_CODE_WALKTHROUGH_VARIANTS.md)
> (변형 확장 패턴 A~F).

---

## 목차

1. [두 개의 residual — 개념 구분](#1-두-개의-residual--개념-구분)
2. [Residual 동역학 (모델 층)](#2-residual-동역학-모델-층)
3. [Residual 제어 (Residual-MPPI, 컨트롤러 층)](#3-residual-제어-residual-mppi-컨트롤러-층)
4. [수식 ↔ 코드 매핑표](#4-수식--코드-매핑표)
5. [흔한 실수](#5-흔한-실수)
6. [연습문제](#6-연습문제)
7. [부록 — 더 공부하기 위한 자료](#7-부록--더-공부하기-위한-자료)

---

## 1. 두 개의 residual — 개념 구분

### 1.1 왜 헷갈리는가

"residual"은 로보틱스에서 **"이미 알고 있는 것으로 설명되지 않는 나머지"**를
가리키는 범용 단어다. 문제는 이 repo에서 "이미 알고 있는 것"이 두 가지로
등장한다는 것이다:

- **정책(policy)을 이미 안다** → 정책이 못 하는 나머지를 최적화한다 = **잔차 제어**
- **물리 모델을 이미 안다** → 물리가 못 맞추는 나머지를 학습한다 = **잔차 동역학**

둘 다 "베이스라인 + 나머지"라는 구조를 공유하지만, 베이스라인이 **제어 시퀀스
U**냐 **상태 미분 ẋ**냐가 다르다. 이 하나의 차이가 모든 것을 가른다.

### 1.2 개념 대조표

| 구분 | **잔차 제어** (Residual-MPPI) | **잔차 동역학** (ResidualDynamics) |
|------|------------------------------|-----------------------------------|
| **사는 층** | 컨트롤러 (`controllers/mppi/`) | 모델 (`models/learned/`) |
| **베이스라인(사전 지식)** | 사전 정책 π(state, ref) → 명목 제어 시퀀스 U_nom | 물리 모델 f_phys(x, u) → 상태 미분 |
| **잔차의 대상** | 제어 시퀀스 δu = U − U_nom | 상태 미분 f_learned(x, u) = ẋ − f_phys |
| **잔차를 어떻게 얻나** | MPPI로 **온라인 최적화** (샘플링·가중 평균) | 데이터로 **학습** (NN/GP) 또는 온라인 **추정** |
| **"베이스라인"의 역할** | 샘플링 중심 + 비용 정규화 기준 | rollout 시 항상 더해지는 물리 항 |
| **깨질 때 폴백** | kl_weight=0 → Vanilla MPPI | use_residual=False → 순수 물리 모델 |
| **대표 참고** | Wang et al. ICLR 2025 (arXiv:2407.00898) | 물리+학습 하이브리드 / residual GP 계열 |
| **이 repo 파일** | `residual_mppi.py`, `ancillary_policies.py` | `residual_dynamics.py`, (+ MAML/L1/EKF/ALPaCA) |

### 1.3 언제 무엇을 쓰나

두 residual은 **경쟁하지 않는다 — 직교한다**. 같은 실행에서 둘을 동시에 쓸 수
있다: `ResidualDynamics`로 만든 정확한 모델 위에서 `ResidualMPPIController`가
사전 정책 잔차를 최적화한다.

- **좋은 정책(전문가 데모, RL 정책, pure pursuit)이 있는데 미세 조정만 필요**
  → 잔차 제어. 정책이 대부분을 하고 MPPI는 근처만 탐색하니 샘플 효율이 높다.
- **모델이 부정확한데(슬립·마찰·미지 페이로드) 안전한 외삽이 필요**
  → 잔차 동역학. 물리가 큰 틀을 잡고 학습이 보정하니 OOD에서도 발산하지 않는다.
- **실기 배포 파이프라인** → 보통 **둘 다**: sim-to-real 갭은 잔차 동역학으로
  좁히고, 데모 정책은 잔차 제어로 다듬는다. 이 조합의 실습은
  [examples/learned/residual_sim2real_pipeline.py](../../examples/learned/residual_sim2real_pipeline.py)
  에서 다룬다(→ 이 문서 §2.6, §7.5).

이제 각각을 코드 레벨로 파고든다. 모델 층이 개념적으로 더 단순하므로
잔차 동역학부터 시작한다.

---

## 2. Residual 동역학 (모델 층)

### 2.1 수식과 직관

잔차 동역학의 전부는 **한 줄**이다:

```
f_total(x, u) = f_physics(x, u) + f_learned(x, u)
```

- **f_physics**: 우리가 신뢰하는 물리 법칙 (기구학/동역학 방정식). 손으로 유도된,
  검증된 항.
- **f_learned**: 물리가 못 맞추는 **나머지**를 데이터로 채우는 항 (NN/GP). 슬립,
  마찰, 미지 페이로드, 고차 항 등.

왜 총 동역학을 통째로 학습하지 않고 이렇게 쪼개는가? 두 가지 실전 이유:

1. **OOD 안전성 (외삽 안정성).** 순수 신경망 모델은 학습 데이터 밖(out-of-distribution)에서
   임의로 발산할 수 있다. 하지만 f_physics는 어디서든 물리적으로 타당한 값을
   내므로, f_learned가 데이터 밖에서 0에 가깝게만 나와도 전체는 "물리로 외삽"된다.
   즉 최악의 경우 **틀린 물리 모델 수준**으로 우아하게 퇴화한다.
2. **데이터 효율.** 학습해야 할 함수가 "전체 동역학"이 아니라 "물리로 설명 안 되는
   작은 나머지"뿐이다. 타겟의 크기(norm)와 복잡도가 작으니 적은 데이터로 학습된다.

이 직관은 residual RL(Silver·Johannink), residual GP 동역학과 같은 계보다
(→ §7.1).

### 2.2 총 동역학 — `forward_dynamics`

핵심 계산은 정확히 §2.1의 한 줄을 코드로 옮긴 것이다:

```python
# mppi_controller/models/learned/residual_dynamics.py:89-116
def forward_dynamics(
    self, state: np.ndarray, control: np.ndarray
) -> np.ndarray:
    # 1. 물리 기반 동역학
    physics_dot = self.base_model.forward_dynamics(state, control)

    # 2. Residual 추가 (있을 경우)
    if self.use_residual and self.residual_fn is not None:
        residual_dot = self.residual_fn(state, control)

        # 통계 업데이트 (디버깅)
        self._update_stats(residual_dot)

        total_dot = physics_dot + residual_dot
    else:
        total_dot = physics_dot

    return total_dot
```

**해설**:

- `base_model.forward_dynamics()`가 f_physics다. `ResidualDynamics`는
  `RobotModel`을 상속하므로(residual_dynamics.py:14) MPPI의 `rollout()`이 이
  객체를 여느 모델처럼 호출한다 — 컨트롤러는 자기가 잔차 모델을 쓰는지조차
  모른다. 이것이 "모델 층 교체"의 핵심 장점이다.
- `residual_fn`이 f_learned다. state와 control을 받아 **상태 미분의 보정량**을
  낸다. 배치 입력(K개 샘플)도 그대로 지원해야 한다(§2.5의 헬퍼들이 `ndim`으로
  분기하는 이유).
- `use_residual=False` 또는 `residual_fn is None`이면 `total_dot = physics_dot`
  — **순수 물리 모델로 폴백**한다(residual_dynamics.py:113-114). 이것이 §1.2의
  "깨질 때 폴백" 행에 해당한다. 디버깅 시 잔차를 꺼서 물리 단독 성능을 baseline으로
  비교할 수 있다.

**왜 이렇게 — 덧셈이지 합성이 아니다.** f_total은 f_phys와 f_learned를 **더한다**.
곱하거나 함수 합성(f_learned(f_phys(...)))하지 않는다. 덧셈이라야 (a) 물리 항이
그대로 보존되고, (b) f_learned=0이 자연스러운 "물리만" 상태가 되어 폴백/외삽
안전성이 성립한다.

### 2.3 학습된 모델 자동 연결 — `learned_model=`

`residual_fn`을 직접 넘기는 대신 **또 다른 `RobotModel`**(예: NeuralDynamics)을
잔차로 꽂을 수 있다:

```python
# mppi_controller/models/learned/residual_dynamics.py:56-68
# learned_model이 주어지면 자동으로 residual_fn/uncertainty_fn 연결
if learned_model is not None:
    self.learned_model = learned_model
    self.residual_fn = learned_model.forward_dynamics
    # GP 모델이면 uncertainty_fn 자동 연결
    if hasattr(learned_model, 'predict_with_uncertainty'):
        self.uncertainty_fn = lambda s, u: learned_model.predict_with_uncertainty(s, u)[1]
    else:
        self.uncertainty_fn = uncertainty_fn
else:
    self.learned_model = None
    self.residual_fn = residual_fn
    self.uncertainty_fn = uncertainty_fn
```

**해설**:

- `self.residual_fn = learned_model.forward_dynamics` — 학습된 모델의
  `forward_dynamics`를 **잔차 함수로 바인딩**한다. 이 한 줄에 **가장 중요한
  계약**이 숨어 있다: `learned_model.forward_dynamics(x,u)`가 반환하는 값은
  **총 동역학이 아니라 잔차 ẋ − f_phys여야 한다**.
- `predict_with_uncertainty`를 가진 모델(GP 계열)이면, 두 번째 반환값(표준편차)을
  `uncertainty_fn`으로 자동 연결한다(residual_dynamics.py:61-62). 이 불확실성은
  나중에 `get_uncertainty()`(residual_dynamics.py:130-146)로 조회되어 risk-aware
  비용이나 tube 마진에 쓸 수 있다.

> **⚠ 핵심 주의 — 학습 타겟의 정의.** `learned_model`은 반드시 **잔차만** 예측하도록
> 학습되어야 한다. 즉 학습 타겟(supervision signal)은
>
> ```
> target = (x_{next} − x) / dt  −  f_physics(x, u)
> ```
>
> 이다. 여기서 `(x_next − x)/dt`는 실측된 상태 미분(Euler 근사), 여기서
> **물리 예측을 빼준 나머지**가 잔차의 정답이다. 실수로 `(x_next − x)/dt`
> 자체(= 총 동역학)를 타겟으로 학습하면, 추론 시 `f_phys + f_learned ≈
> f_phys + f_total = 2·f_phys`가 되어 물리가 **이중 계산**된다. 이 실수는
> §5의 첫 번째 항목에서 다시 다룬다.

### 2.4 잔차 기여도 분석 — `get_residual_contribution`

잔차가 실제로 얼마나 일하는지(물리가 대부분을 설명하는지, 아니면 잔차가 지배하는지)를
해석하는 진단 메서드다:

```python
# mppi_controller/models/learned/residual_dynamics.py:164-187 (발췌)
physics_dot = self.base_model.forward_dynamics(state, control)

if self.use_residual and self.residual_fn is not None:
    residual_dot = self.residual_fn(state, control)
    total_dot = physics_dot + residual_dot

    # Residual 기여도 비율
    residual_ratio = np.zeros_like(residual_dot)
    nonzero = np.abs(total_dot) > 1e-6
    residual_ratio[nonzero] = (
        residual_dot[nonzero] / total_dot[nonzero]
    )
else:
    residual_dot = np.zeros_like(physics_dot)
    total_dot = physics_dot
    residual_ratio = np.zeros_like(physics_dot)

return {
    "physics_dot": physics_dot,
    "residual_dot": residual_dot,
    "total_dot": total_dot,
    "residual_ratio": residual_ratio,
}
```

**해설**:

- `residual_ratio[i] = residual_dot[i] / total_dot[i]` — 상태 차원별로 "총
  미분 중 잔차가 차지하는 비율"이다. 0에 가까우면 물리가 지배(모델이 이미
  좋다), 1에 가까우면 잔차가 지배(물리 모델이 그 차원에서 크게 틀렸다).
- `nonzero = np.abs(total_dot) > 1e-6` — 0으로 나누기를 막는 마스크. total_dot이
  거의 0인 차원은 비율을 0으로 남긴다(0/0 회피).

**해석 실전 팁**: 잔차 비율이 **꾸준히 크다(>0.5)**면 물리 모델을 다시 봐야
한다는 신호다(파라미터가 틀렸거나 구조가 부족). 반대로 잔차 비율이 **거의 0**인데도
추적이 나쁘면, 문제는 모델이 아니라 컨트롤러/비용에 있다. `get_stats()`
(residual_dynamics.py:207-218)는 Welford 알고리즘으로 잔차의 running mean/std를
누적해 시간에 따른 잔차 크기 추이를 볼 수 있게 한다.

### 2.5 4개 헬퍼 factory

파일 하단(residual_dynamics.py:237-342)에는 **테스트·데모용 잔차 함수 4종**이
있다. 실제 학습 없이도 "잔차가 있으면 어떻게 되는지"를 재현하는 용도다:

| factory | 정의 | 물리적 의미 | 시그니처 (state, control) → residual |
|---------|------|-------------|--------------------------------------|
| `create_constant_residual(v)` | residual = v (상수) | 고정 bias/드리프트 | :239-257 |
| `create_state_dependent_residual(G)` | residual = state @ Gᵀ | 상태 비례 (예: 슬립) | :260-281 |
| `create_control_dependent_residual(M)` | residual = control @ Mᵀ | 제어 비례 (예: 액추에이터 bias) | :284-305 |
| `create_sine_residual(a, f, φ)` | residual = a·sin(2πf·t+φ) | 주기적 외란 | :308-342 |

**공통 패턴**: 각 factory는 클로저(closure)를 반환하며 내부에서 `state.ndim`으로
단일/배치를 분기한다(예: constant는 `state.ndim == 1`이면 그대로,
아니면 `np.tile`로 배치 브로드캐스트, residual_dynamics.py:251-255). 이는 MPPI
rollout이 (K, nx) 배치를 넘기기 때문에 필수다. `create_sine_residual`은 벽시계
시간(`time.time()`)을 쓰므로(residual_dynamics.py:327-332) 재현성이 필요한
테스트에서는 주의한다.

### 2.6 적응형 잔차 — MAML / L1 / EKF / ALPaCA

`residual_dynamics.py`의 잔차는 보통 **오프라인 학습**되거나 헬퍼로 고정된다.
하지만 잔차를 **온라인으로 추정**하는 관점이 이 repo의 `models/learned/`에 여럿
있다. 이들은 "모델 보정을 실행 중에 추정한다"는 점에서 **적응형(adaptive)
잔차**로 통일해서 볼 수 있다:

| 모델 | 파일 | 잔차를 얻는 방법 | 오프라인 학습 |
|------|------|------------------|---------------|
| **MAML** | `maml_dynamics.py` | 메타 파라미터에서 few-shot inner-loop SGD로 적응 | 필요 (메타 학습) |
| **L1-Adaptive** | `l1_adaptive_dynamics.py` | 상태 예측기 오차 → σ̂ 추정 → 저역통과 필터 → f_nom에 가산 | 불필요 |
| **EKF** | `ekf_dynamics.py` | 확장 상태 [.., ĉ_v, ĉ_ω]로 마찰 파라미터를 칼만 추정 | 불필요 |
| **ALPaCA** | `alpaca_dynamics.py` | frozen feature + Bayesian linear regression closed-form 적응 | 필요 (메타 학습) |

**통일된 관점**: L1의 보정 출력 `f_total = f_nom + σ_filtered`
(l1_adaptive_dynamics.py 클래스 docstring)는 §2.1의 `f_total = f_phys + f_learned`와
**정확히 같은 구조**다 — 단지 f_learned가 데이터로 학습된 NN이 아니라
**필터로 추정된 외란 σ**일 뿐이다. EKF도 마찬가지로 "공칭 모델 +
온라인 추정된 파라미터 보정"이다. 즉 잔차 동역학은 **정적 학습 ↔ 온라인 적응**의
스펙트럼이며, `ResidualDynamics`는 그 정적 끝단의 일반 컨테이너다.

이 스펙트럼의 실습(오프라인 잔차 학습 → 온라인 적응 비교)은
[examples/learned/residual_sim2real_pipeline.py](../../examples/learned/residual_sim2real_pipeline.py)를
참고한다.

---

## 3. Residual 제어 (Residual-MPPI, 컨트롤러 층)

### 3.1 수식

이제 **컨트롤러 층**으로 올라간다. Residual-MPPI(25번째 변형)는 Vanilla MPPI의
"이전 최적 해 U를 중심으로 샘플링"을 **"사전 정책 π의 출력 U_nom을 중심으로
샘플링"**으로 바꾼다:

```
U_nom = π(state, ref)                          # 사전 정책 명목 시퀀스
ε_k ~ N(0, σ²)                                 # 가우시안 잔차 노이즈
V_k  = U_nom + residual_scale · ε_k            # 후보 제어 시퀀스
C_aug(V_k) = C(τ_k) + kl_weight · ||V_k − U_nom||²   # 증강 비용
U    = U_nom + Σ_k ω_k · ε_k                   # 가중 잔차 업데이트
```

Vanilla MPPI(→ [02편](02_MPPI_FUNDAMENTALS.md))와 비교하면 딱 세 군데가 다르다:

1. **명목 시퀀스**가 이전 해 U가 아니라 **정책 출력 π(state)**이다.
2. **증강 비용**에 KL 페널티 `kl_weight·||U−U_nom||²`가 붙어 "정책 근처에
   머물라"는 압력을 준다.
3. **업데이트가 잔차 위에서** 일어난다: 최종 해 = 정책 명목 + 가중 노이즈.

정책이 좋을수록 최적해가 명목 근처에 있으므로 **적은 샘플로 빠르게 수렴**한다.
정책이 나쁘면? kl_weight를 낮추면 그냥 Vanilla처럼 동작한다(§3.5).

### 3.2 `compute_control` 흐름

Residual-MPPI는 `_compute_weights`만 오버라이드하는 최소 침습형이 아니라,
`compute_control` 전체를 오버라이드한다(중심을 U가 아닌 π로 바꿔야 하므로).
전체는 residual_mppi.py:97-213이고, 핵심 5구간을 발췌한다:

**(1) 정책 명목 시퀀스 (주기적 캐싱)**

```python
# mppi_controller/controllers/mppi/residual_mppi.py:121-131
# 1. 사전 정책에서 명목 시퀀스 생성 (주기적 업데이트)
if self._step_count % self.residual_params.policy_update_interval == 0:
    self._policy_nominal = self._get_policy_nominal(
        state, reference_trajectory, N, nu
    )

# 2. 샘플링 중심 결정
if self.residual_params.use_policy_nominal and self._policy_nominal is not None:
    center = self._policy_nominal
else:
    center = self.U
```

정책 forward-sim은 비쌀 수 있으므로 `policy_update_interval` 스텝마다만 재평가하고
사이엔 캐시(`self._policy_nominal`)를 쓴다. `use_policy_nominal=False`거나 정책이
없으면 `center = self.U`로 **Vanilla와 동일한 중심**이 된다.

**(2) 잔차 노이즈 샘플링**

```python
# mppi_controller/controllers/mppi/residual_mppi.py:133-138
# 3. 노이즈 샘플링 (K, N, nu)
noise = self.noise_sampler.sample(center, K, self.u_min, self.u_max)
noise = noise * self.residual_params.residual_scale

# 4. 샘플 제어 시퀀스 (K, N, nu)
sampled_controls = center[None, :, :] + noise
```

`residual_scale`로 잔차의 크기를 스케일한다. 좋은 정책일수록 작게 하면 정책
근처만 촘촘히 탐색한다. `sampled_controls = center + noise`가 수식의 `V_k = U_nom
+ residual_scale·ε_k`다.

**(3) rollout + 비용, 그리고 증강 비용(KL 페널티)**

```python
# mppi_controller/controllers/mppi/residual_mppi.py:150-160
# 6. 증강 비용: KL 페널티 (정책에서 벗어나는 것에 페널티)
if (
    self.residual_params.use_augmented_cost
    and self._policy_nominal is not None
    and self.residual_params.kl_weight > 0
):
    residuals = sampled_controls - self._policy_nominal[None, :, :]
    kl_cost = self.residual_params.kl_weight * np.sum(
        residuals ** 2, axis=(1, 2)
    )
    costs = costs + kl_cost
```

`||V_k − U_nom||²`를 시퀀스 전체에 대해 합산(axis=(1,2))해 kl_weight를 곱하고
비용에 더한다. **가우시안 노이즈 하에서 이 이차 페널티가 곧 정책 분포에 대한
KL 근사**라서 이름이 "KL 페널티"다. `kl_weight > 0` 가드 덕분에 페널티를 완전히
끌 수 있다.

**(4) 가중 잔차 업데이트**

```python
# mppi_controller/controllers/mppi/residual_mppi.py:162-167
# 7. 가중치 계산
weights = self._compute_weights(costs, self.params.lambda_)

# 8. 가중 잔차 업데이트
weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)  # (N, nu)
self.U = center + weighted_noise
```

가중치 계산 `_compute_weights`는 **base의 softmax를 그대로 상속**한다
(base_mppi.py:276). 핵심 차이는 업데이트 기준선이 `center`(=정책 명목)라는 것뿐:
`U = U_nom + Σ ω_k ε_k`. 이후 receding-horizon shift(residual_mppi.py:176-178)와
통계 수집(residual_mppi.py:180-198)은 Vanilla와 동일하다.

### 3.3 사전 정책 계약 — `_get_policy_nominal`

정책은 두 가지 형태를 모두 받는다: `propose_sequence`를 가진 `AncillaryPolicy`
객체이거나, 같은 시그니처의 순수 callable이다.

```python
# mppi_controller/controllers/mppi/residual_mppi.py:234-246
if self._base_policy is None:
    return np.zeros((N, nu))

if hasattr(self._base_policy, "propose_sequence"):
    return self._base_policy.propose_sequence(
        state, reference_trajectory, N, self.params.dt, self.model
    )
elif callable(self._base_policy):
    return self._base_policy(
        state, reference_trajectory, N, self.params.dt, self.model
    )
else:
    return np.zeros((N, nu))
```

**계약**: 정책은 `(state, reference_trajectory, N, dt, model) → (N, nu)`를 만족해야
한다. 정책이 없으면 **제로 시퀀스**를 낸다 — 이 경우 명목이 0이라 사실상 Vanilla와
같은 초기화가 된다. `set_base_policy()`(residual_mppi.py:248-256)로 실행 중 정책을
교체할 수 있고, 교체 시 `_policy_nominal` 캐시를 무효화한다.

### 3.4 5개 내장 보조 정책

정책 계약(`AncillaryPolicy`, ancillary_policies.py:15-51)은 원래 Biased-MPPI용으로
만들어졌지만 Residual-MPPI가 그대로 재사용한다. 내장 5종:

| 정책 | 파일 | 아이디어 | 잔차 제어에서의 쓸모 |
|------|------|----------|---------------------|
| **PurePursuitPolicy** | :54-124 | lookahead 목표점 향해 (v, ω) 계산 + forward-sim | 기본값. 부드러운 추적 명목 |
| **BrakingPolicy** | :127-140 | 전부 0 (`np.zeros`) | 비상 정지 명목 (안전 폴백) |
| **FeedbackPolicy** | :143-191 | `AncillaryController` 재사용, 레퍼런스 추적 피드백 | 안정적 추적 명목 |
| **MaxSpeedPolicy** | :194-261 | 레퍼런스 방향 최대 속도 | 적극적 탐색 명목 |
| **PreviousSolutionPolicy** | :264-287 | 이전 U 그대로 반환 (warm start) | 명목=이전 해 → Vanilla 근사 |

`ResidualMPPIController`의 기본값은 `policy_type="feedback"`일 때
`PurePursuitPolicy(lookahead=0.5, v_gain=1.0)`를 자동 생성한다
(residual_mppi.py:83-89). `POLICY_REGISTRY`(ancillary_policies.py:292-298)와
`create_ancillary_policy()`(ancillary_policies.py:301-317)로 이름 기반 생성도
가능하다.

**PurePursuit 정책 내부** (ancillary_policies.py:70-124)는 각 스텝마다 lookahead
인덱스의 레퍼런스 점을 목표로 (v, ω)를 계산하고 `model.forward_dynamics`로
한 스텝 전진하는 **sequential forward-sim**이다. 이 sim이 앞의 `model`을
`ResidualDynamics`로 넘기면, 정책조차 잔차 보정된 모델 위에서 명목을 만든다 —
두 residual이 자연스럽게 합쳐지는 지점이다(§1.3).

### 3.5 kl_weight=0 → Vanilla graceful degradation

Residual-MPPI의 안전장치는 **kl_weight → Vanilla 폴백**이다. §3.2의 KL 가드
(residual_mppi.py:151-154)를 다시 보면, `kl_weight > 0`이 아니면 증강 비용이
아예 붙지 않는다. 그러면 비용은 순수 `C(τ)`가 되고, 명목만 정책 출력이면서
가중치는 표준 softmax다.

- `kl_weight=0` **그리고** `use_policy_nominal=True`: 명목은 정책이되 비용 압력은
  없음 → 정책을 **초기화(warm start)**로만 쓰는 순수 MPPI.
- `kl_weight=0` **그리고** `use_policy_nominal=False`: `center=self.U` →
  **완전한 Vanilla MPPI**. 정책이 나쁘다는 게 판명되면 이렇게 꺼서 손해를 없앤다.
- `kl_weight` 큼: 해가 정책에 강하게 묶임. 정책이 훌륭할 때만 이득.

이 "정책이 나빠도 최악의 경우 Vanilla"라는 성질이 Residual-MPPI를 실전에서
**안전하게 채택 가능**하게 만든다.

### 3.6 이웃 변형과의 차이 — Biased / Feedback

Residual-MPPI는 "정책을 MPPI에 결합"하는 여러 방식 중 하나다. 자매 변형과의
차이는 [07편 §3.3(Biased)](07_CODE_WALKTHROUGH_VARIANTS.md#33-biased-mppi--혼합-샘플링과-q_s-소거)과
[07편 §6(패턴 E, Feedback)](07_CODE_WALKTHROUGH_VARIANTS.md#6-패턴-e--피드백구조-결합)에서
자세히 다룬다. 3줄 요약:

- **Biased-MPPI**(혼합 샘플링, 패턴 B): 정책을 **샘플의 일부**(K개 중 J개)로만
  주입하고 나머지는 가우시안. 정책 품질 의존도가 **낮다**(일부 샘플만 정책).
- **Feedback-MPPI**(Riccati, 패턴 E): 정책이 아니라 **피드백 게인**을 MPPI 해에
  결합해 재계산 없이 빠른 보정. 명목의 성질이 다르다.
- **Residual-MPPI**(정책 중심, 이 문서): 정책을 **명목 시퀀스 전체**로 쓰고
  비용으로 근처 탐색을 유도. 정책 품질 의존도가 **높지만** kl_weight로 조절 가능.

([MPPI_THEORY.md §24](../MPPI_THEORY.md)의 "Vanilla vs Biased vs Residual" 대조표도
이 세 축(명목 시퀀스·정책 역할·정책 품질 의존)으로 정리한다.)

---

## 4. 수식 ↔ 코드 매핑표

### 4.1 잔차 동역학

| 수식 | 코드 | 위치 |
|------|------|------|
| f_physics(x, u) | `self.base_model.forward_dynamics(state, control)` | residual_dynamics.py:103 |
| f_learned(x, u) | `self.residual_fn(state, control)` | residual_dynamics.py:107 |
| f_total = f_phys + f_learned | `total_dot = physics_dot + residual_dot` | residual_dynamics.py:112 |
| f_learned ← learned_model | `self.residual_fn = learned_model.forward_dynamics` | residual_dynamics.py:59 |
| 폴백 (잔차 off) | `total_dot = physics_dot` | residual_dynamics.py:114 |
| 잔차 비율 r/총 | `residual_dot[nonzero] / total_dot[nonzero]` | residual_dynamics.py:174-176 |
| σ(x,u) 불확실성 | `learned_model.predict_with_uncertainty(s,u)[1]` | residual_dynamics.py:62 |

### 4.2 잔차 제어 (Residual-MPPI)

| 수식 | 코드 | 위치 |
|------|------|------|
| U_nom = π(state, ref) | `_get_policy_nominal(...)` → `propose_sequence` | residual_mppi.py:123, :237-239 |
| 샘플링 중심 = U_nom | `center = self._policy_nominal` | residual_mppi.py:129 |
| V_k = U_nom + scale·ε_k | `sampled_controls = center[None] + noise` (`noise *= residual_scale`) | residual_mppi.py:135-138 |
| C(τ_k) | `self.cost_function.compute_cost(...)` | residual_mppi.py:146-148 |
| kl_weight·‖V−U_nom‖² | `kl_weight * np.sum(residuals**2, axis=(1,2))` | residual_mppi.py:156-159 |
| C_aug = C + KL | `costs = costs + kl_cost` | residual_mppi.py:160 |
| ω_k = softmax(−C_aug/λ) | `self._compute_weights(costs, lambda_)` (base 상속) | residual_mppi.py:163 / base_mppi.py:276 |
| U = U_nom + Σ ω_k ε_k | `self.U = center + weighted_noise` | residual_mppi.py:166-167 |
| kl_weight=0 → Vanilla | KL 가드 `kl_weight > 0` 미충족 시 미적용 | residual_mppi.py:154 |

---

## 5. 흔한 실수

이 두 residual을 다룰 때 실제로 반복되는 함정들이다.

### 5.1 잔차 동역학 — 학습 타겟 부호/정의를 반대로

가장 흔하고 치명적인 실수. `learned_model`을 **총 동역학**으로 학습해 놓고
`ResidualDynamics`에 잔차로 꽂는 것이다. 그러면 rollout에서
`f_phys + f_total ≈ 2·f_phys`로 물리가 **이중 계산**되어, 예측 상태가 실제보다
빠르게(대략 2배) 움직인다. 증상: 잔차를 켜자마자 rollout이 발산하거나
로봇이 목표를 지나쳐 오버슛한다.
- **올바른 타겟**: `target = (x_next − x)/dt − f_physics(x, u)` (§2.3 주의).
- **진단**: `get_residual_contribution()`의 `residual_ratio`가 모든 차원에서
  ~1.0에 가깝고 `residual_dot ≈ physics_dot`이면 타겟이 총 동역학인 것.
- 부호 실수(타겟에서 물리를 더해버림)도 같은 증상을 낸다.

### 5.2 잔차 동역학 — 물리/잔차의 상태·제어 차원 불일치

`base_model`이 3D 기구학(x, y, θ)인데 `learned_model`은 5D 동역학
(x, y, θ, v, ω)이라면, `physics_dot + residual_dot`가 shape mismatch로
터지거나(운 좋으면) 조용히 브로드캐스트되어 틀린 값을 낸다.
- **규칙**: `residual_fn`의 출력은 `base_model.state_dim`과 **정확히 같은 차원**의
  상태 미분이어야 한다. control_dim도 두 모델이 일치해야 한다.
- `ResidualDynamics`는 `state_dim`/`control_dim`을 **base_model에 위임**한다
  (residual_dynamics.py:77-83) — 즉 잔차 함수가 그 차원을 지킬 책임은 사용자에게
  있다. 배치 축(K)까지 고려하면 잔차 함수는 (K, nx) → (K, nx)를 지원해야 한다
  (§2.5 헬퍼들이 `ndim` 분기로 이를 처리하는 이유).

### 5.3 잔차 제어 — KL 스케일 과대/과소

`kl_weight`는 "정책 신뢰도" 손잡이다. 양 극단 모두 함정이다.
- **과대**: 해가 정책에 못 박혀 MPPI가 실질적으로 아무것도 최적화하지 못한다.
  정책이 장애물로 향하면 로봇도 장애물로 향한다(정책 오류를 교정 못 함).
- **과소(→0)**: 명목만 정책이고 비용 압력이 없어 **사실상 Vanilla**가 된다
  (§3.5). 정책의 이점(샘플 효율)이 사라진다.
- **처방**: `residual_scale`와 함께 튜닝한다. 좋은 정책이면 kl_weight를 키우고
  residual_scale을 줄여 "정책 근처 촘촘히", 불확실한 정책이면 반대로.

### 5.4 잔차 제어 — 나쁜 정책이 오히려 해가 됨

Residual-MPPI는 정책 품질에 **의존도가 높다**(§3.6 대조). 정책이 명목을 나쁜
방향으로 끌면, 샘플링 중심 자체가 나쁜 곳에 있어 좁은 residual_scale로는
빠져나오지 못한다.
- **증상**: Vanilla보다 성능이 나쁨. 특히 정책이 학습된 방향과 다른 시나리오
  (OOD)에서.
- **처방**: kl_weight를 낮추거나 `use_policy_nominal=False`로 폴백(§3.5). 또는
  `BrakingPolicy`처럼 **안전한 정책**을 명목으로 두어 "최소한 나쁘지 않게".
  정책 품질을 모니터링하려면 `get_residual_statistics()`(residual_mppi.py:258)의
  `mean_best_cost`를 Vanilla baseline과 비교한다.

### 5.5 두 residual을 혼동해서 잘못된 층에 손댐

"모델이 부정확한데" Residual-MPPI의 kl_weight를 만지는 식의 오진.
- 모델 오차(슬립·마찰) → **잔차 동역학** (`residual_dynamics.py`, 또는 L1/EKF 적응).
- 정책 미세 조정 필요 → **잔차 제어** (`residual_mppi.py`).
- §1.2 대조표로 "지금 고치려는 게 어느 층인가"를 먼저 판별하라.

---

## 6. 연습문제

**문제 1 — 폴백 등가성.**
`ResidualDynamics(base, use_residual=False)`와 `ResidualDynamics(base,
residual_fn=None)`는 `forward_dynamics`가 동일한 결과를 낸다. 코드 근거
(residual_dynamics.py의 어느 줄들)를 들어 왜 그런지 설명하라.
*힌트: :106의 `and` 조건 두 항.*

**문제 2 — 학습 타겟 유도.**
샘플 (x_t, u_t, x_{t+1})와 dt=0.05, 물리 모델 f_phys가 주어졌다. 잔차 신경망의
지도 타겟을 식으로 쓰고, 만약 실수로 `(x_{t+1}−x_t)/dt`를 타겟으로 학습하면
추론 시 `forward_dynamics`가 반환하는 값을 f_phys의 함수로 표현하라.
*답: 타겟 = (x_{t+1}−x_t)/dt − f_phys(x_t, u_t). 오학습 시 f_phys +
(x_{t+1}−x_t)/dt ≈ 2·f_phys (물리 이중 계산).*

**문제 3 — kl_weight 극한.**
Residual-MPPI에서 `kl_weight → ∞`, `residual_scale → 0`으로 보내면 컨트롤러의
출력 U는 무엇에 수렴하는가? 반대로 `kl_weight = 0`, `use_policy_nominal = False`면?
코드 위치를 들어 답하라.
*힌트: :167의 `center`가 각각 무엇이 되는가.*

**문제 4 — 정책과 잔차 동역학 결합.**
`ResidualMPPIController`에 `PurePursuitPolicy`를 쓰고 `model`로 `ResidualDynamics`를
넘겼다. 정책의 명목 시퀀스가 만들어질 때 잔차 동역학이 쓰이는가? 아니면 컨트롤러
rollout에서만? ancillary_policies.py의 어느 줄이 근거인가?
*힌트: ancillary_policies.py:121의 forward-sim.*

**문제 5 — 진단.**
다음 관찰을 두 residual 중 어느 것의 문제로 진단하고 어떤 손잡이를 만질지
답하라. (a) 잔차를 켠 직후 rollout이 발산하고 로봇이 2배 속도로 튄다.
(b) 정책을 넣었더니 Vanilla보다 장애물 회피가 나빠졌다. (c) `residual_ratio`가
θ 차원에서만 0.8이고 나머지는 0.05다.
*답: (a) 잔차 동역학 학습 타겟이 총 동역학 → §5.1. (b) 잘못된 정책 →
kl_weight↓ 또는 폴백 §5.4. (c) 물리 모델의 heading 항이 부정확 → 물리 파라미터
재튜닝 신호 §2.4.*

---

## 7. 부록 — 더 공부하기 위한 자료

> 링크는 2026-07 기준. 확신할 수 없는 arXiv ID는 제목+학회만 표기했다.

### 7.1 주석 달린 핵심 레퍼런스

**잔차 제어 (Residual-MPPI 계열):**

1. **Wang et al., "Residual-MPPI: ...", ICLR 2025 (arXiv:2407.00898).**
   — 이 repo `residual_mppi.py`의 원전. 사전 정책 명목 + 잔차 최적화 + KL 정규화.
   본 문서 §3의 수식·구조가 여기서 온다.
2. **Silver et al., "Residual Policy Learning", 2018 (arXiv:1812.06298).**
   — residual RL의 대표. "고정 베이스 정책 + 학습된 잔차 정책"이라는 아이디어의
   제어판. Residual-MPPI가 "학습" 대신 "MPPI 최적화"로 잔차를 얻는 대응물임을
   이해하는 데 좋다.
3. **Johannink et al., "Residual Reinforcement Learning for Robot Control",
   ICRA 2019.** — 손으로 만든 제어기 + RL 잔차로 실기 조립 작업. §5.4의
   "나쁜 베이스가 해가 됨" 직관의 실증 사례.

**잔차 동역학 (물리+학습 하이브리드):**

4. residual/hybrid dynamics 계열 — "physics prior + learned correction"으로
   데이터 효율과 외삽 안정성을 얻는 흐름. Gaussian Process 잔차 모델(GP-MPC
   계열)이 대표적이며, GP의 posterior 분산이 §2.2의 `uncertainty_fn`으로
   자연스럽게 연결된다.
5. **적응 제어 고전** — L1 adaptive control(Hovakimyan & Cao), EKF 파라미터
   추정. 이 repo의 `l1_adaptive_dynamics.py`/`ekf_dynamics.py`가 "온라인 추정된
   잔차"의 구현체다(§2.6).
6. **메타 학습 기반 적응** — MAML(Finn et al., ICML 2017), ALPaCA(Harrison et al.,
   probabilistic meta-learning). `maml_dynamics.py`/`alpaca_dynamics.py`의 배경.

### 7.2 최근 연구 동향 (2024–2026)

1. **학습된 제안 분포로서의 정책.** diffusion/flow 정책을 MPPI 명목이나 샘플러로
   쓰는 흐름 — Residual-MPPI의 "정책=명목" 아이디어가 생성 모델과 만나는 지점.
   이 repo 관점은 [05_GENERATIVE_MODELS_FOR_CONTROL.md](05_GENERATIVE_MODELS_FOR_CONTROL.md)와
   [07편 §7(패턴 F)](07_CODE_WALKTHROUGH_VARIANTS.md#7-패턴-f--학습-결합의-공통-뼈대).
2. **불확실성 인지 잔차 동역학.** GP/앙상블 잔차의 분산을 risk-aware MPPI 비용이나
   tube 마진에 주입하는 흐름. `ResidualDynamics.get_uncertainty()`가 그 진입점.
3. **온라인 적응 ↔ 오프라인 학습의 통합.** 메타 학습(빠른 few-shot 적응)과
   필터 기반 추정(L1/EKF)을 하나의 스펙트럼으로 보는 관점(§2.6)이 실기 배포에서
   표준화되는 중.

### 7.3 오픈소스/도구

| 이름 | 특징 | 이 repo와의 관계 |
|------|------|------------------|
| GP-MPC / safe-control-gym | GP 잔차 동역학 + MPC | `ResidualDynamics` + GP `uncertainty_fn` 대응 |
| pytorch_mppi (UM-ARM Lab) | 미분 가능 MPPI + 학습 모델 rollout | 학습 모델을 MPPI에 꽂는 패턴 참고 |

### 7.4 이 repo 내부

- [docs/MPPI_THEORY.md §24](../MPPI_THEORY.md) — Residual-MPPI 이론 + Vanilla/Biased
  대조표.
- [07_CODE_WALKTHROUGH_VARIANTS.md §3.3, §6, §7](07_CODE_WALKTHROUGH_VARIANTS.md) —
  Biased(패턴 B)·Feedback(패턴 E)·학습 결합(패턴 F)과의 코드 레벨 비교.
- [05_GENERATIVE_MODELS_FOR_CONTROL.md](05_GENERATIVE_MODELS_FOR_CONTROL.md) —
  학습된 제안 분포 일반론.
- **다음 편**: [10_SIM_TO_REAL_DEPLOYMENT.md](10_SIM_TO_REAL_DEPLOYMENT.md) —
  잔차 동역학으로 sim-to-real 갭을 좁히고 실기에 배포하는 플레이북.

### 7.5 실습

- [examples/learned/residual_sim2real_pipeline.py](../../examples/learned/residual_sim2real_pipeline.py)
  — 잔차 동역학 end-to-end: 물리 모델 baseline → 잔차 데이터 수집 → 잔차 학습
  → `ResidualDynamics` rollout 비교. (병렬 작성 중)

### 7.6 자주 궁금한 점 → 어디를 볼까

| 궁금한 점 | 내부 자료 |
|---|---|
| 모델이 부정확한데 어디를 고치나? | §1.2 대조표 + §2 (잔차 동역학) |
| 정책은 있는데 미세 조정만 원함 | §3 (Residual-MPPI) + [07 §3.3](07_CODE_WALKTHROUGH_VARIANTS.md#33-biased-mppi--혼합-샘플링과-q_s-소거) |
| 잔차를 켰더니 발산함 | §5.1 (학습 타겟 정의) |
| 정책이 오히려 해가 됨 | §5.4 + kl_weight/폴백 §3.5 |
| 온라인 적응(학습 없이)이 필요 | §2.6 (L1/EKF), `l1_adaptive_dynamics.py` |
| 불확실성을 비용에 반영하려면 | `get_uncertainty()` (residual_dynamics.py:130) + §7.2-2 |

---

*작성: 2026-07 — learning_mppi 공부 자료 시리즈 9편. 코드 인용은 작성 시점의
repo 상태(브랜치 feature/cbfkit-inspired-safety) 기준이며, 모든 줄번호는 실제
소스에서 검증했다.*
