# 07. 코드 워크스루 — MPPI 변형의 확장 패턴

> **이 문서의 목적**: 이 저장소의 43종 MPPI 변형이 **코드 레벨에서** base
> `MPPIController`를 어떻게 확장하는지, 확장 패턴별 대표 구현을 함수 단위로
> 해설한다. 수식 유도와 이론적 배경은 [docs/MPPI_THEORY.md](../MPPI_THEORY.md)에
> 위임하고, 여기서는 "실제 코드가 어떻게 생겼고 왜 그렇게 생겼는지"에 집중한다.
>
> **읽는 법**: 각 절은 "코드 발췌 → 해설 → 설계 트레이드오프 → 흔한 실수" 순서를
> 따른다. 모든 발췌에는 `파일경로:줄번호`를 표기했으며, 이 문서 작성 시점의
> 실제 소스에서 그대로 가져왔다.

---

## 목차

1. [확장 패턴 지도](#1-확장-패턴-지도)
2. [패턴 A — 가중치 함수 교체 (`_compute_weights` 오버라이드)](#2-패턴-a--가중치-함수-교체)
3. [패턴 B — `compute_control` 전체 교체 (DIAL 패턴)](#3-패턴-b--compute_control-전체-교체)
4. [패턴 C — 샘플러 교체](#4-패턴-c--샘플러-교체)
5. [패턴 D — 최적화 관점 확장](#5-패턴-d--최적화-관점-확장)
6. [패턴 E — 피드백/구조 결합](#6-패턴-e--피드백구조-결합)
7. [패턴 F — 학습 결합의 공통 뼈대](#7-패턴-f--학습-결합의-공통-뼈대)
8. [파라미터/레지스트리 연동 — 새 변형 추가 체크리스트](#8-파라미터레지스트리-연동--새-변형-추가-체크리스트)

---

## 1. 확장 패턴 지도

### 1.1 base `MPPIController`의 구조와 오버라이드 포인트

모든 변형의 출발점은 `mppi_controller/controllers/mppi/base_mppi.py`의
`MPPIController`다. 생성자 시그니처부터 확장 지점이 드러난다:

```python
# mppi_controller/controllers/mppi/base_mppi.py:43-48
def __init__(
    self,
    model: RobotModel,
    params: MPPIParams,
    cost_function: Optional[CostFunction] = None,
    noise_sampler: Optional[NoiseSampler] = None,
):
```

`cost_function=None`이면 `StateTracking + Terminal + ControlEffort` 복합 비용을
기본 생성하고(base_mppi.py:57-65), `noise_sampler=None`이면 `GaussianSampler`를
생성한다(base_mppi.py:70-73). 즉 **비용과 샘플러는 상속 없이 생성자 주입만으로
교체 가능**하다.

`compute_control()`은 9단계 파이프라인이다 (base_mppi.py:111-190):

```python
# base_mppi.py:137-169 (요약 발췌)
# 1. 노이즈 샘플링 (K, N, nu)
noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
# 2. 샘플 제어 시퀀스
sampled_controls = self.U + noise
# 3. 샘플 궤적 rollout (K, N+1, nx)
sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
# 4. 비용 계산 (K,)
costs = self.cost_function.compute_cost(...)
# 5. MPPI 가중치 계산
weights = self._compute_weights(costs, self.params.lambda_)
# 6. 가중 평균 업데이트: U_new = U + Σ w_k ε_k
weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
self.U = self.U + weighted_noise
# 7. receding horizon shift
self.U = np.roll(self.U, -1, axis=0)
self.U[-1, :] = 0.0
```

가중치 계산은 별도 메서드로 분리되어 있어 서브클래스의 1순위 오버라이드
포인트가 된다:

```python
# base_mppi.py:276-294
def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
    # 수치 안정성을 위한 log-space 연산
    min_cost = np.min(costs)
    exp_costs = np.exp(-(costs - min_cost) / lambda_)
    weights = exp_costs / np.sum(exp_costs)
    return weights
```

정리하면, base가 제공하는 **확장 포인트는 5가지**다:

| # | 확장 포인트 | 위치 | 성격 |
|---|------------|------|------|
| 1 | `_compute_weights()` 오버라이드 | base_mppi.py:276 | 최소 침습 — 파이프라인 6단계 중 5번만 교체 |
| 2 | `compute_control()` 전체 교체 | base_mppi.py:111 | 최대 자유도 — 루프 구조 자체를 바꿈 |
| 3 | `noise_sampler` 주입 | base_mppi.py:70-73 | 상속조차 불필요 — 1단계만 교체 |
| 4 | `cost_function` 주입 | base_mppi.py:57-67 | 4단계만 교체 (CBF/장애물/체인 비용) |
| 5 | Params 데이터클래스 확장 | mppi_params.py:12 | 위 4개와 조합 — 변형별 하이퍼파라미터 전달 |

이때 서브클래스가 공짜로 물려받는 인프라는 `dynamics_wrapper.rollout()`(배치
rollout), `_compute_ess()`(base_mppi.py:296-310), `u_min/u_max` 클리핑,
`reset()`/`set_control_sequence()`(warm start, base_mppi.py:312-319)다.

### 1.2 변형 분류표 — 어느 변형이 어느 패턴을 쓰는가

통합 벤치마크 `examples/comparison/all_37_variants_benchmark.py`의 레지스트리
(`_get_variant_registry()`, 같은 파일 89-476행)는 각 변형에 **생성자(ctor)
분류**를 붙여 두었다. `_build_controller()`(538-570행)가 이 분류로 분기한다:

```python
# examples/comparison/all_37_variants_benchmark.py:552-567 (요약)
if ctor_type == "standard":
    return variant["controller_cls"](model, params, cost_function=cost)
elif ctor_type in ("smooth", "tube", "no_cost"):
    # These controllers do not accept cost_function kwarg
    return variant["controller_cls"](model, params)
elif ctor_type == "contingency":
    safety_cost = _make_cost(params, obstacles)
    return variant["controller_cls"](
        model, params, cost_function=cost, safety_cost_function=safety_cost)
elif ctor_type == "obstacle_in_params":
    return variant["controller_cls"](model, params, cost_function=cost)
```

레지스트리에 등록된 41종(저장소 전체 43종 중 Koopman-MPPI, World-Model-MPPI는
전용 벤치마크로만 다룸)을 ctor 분류 × 확장 패턴으로 정리하면:

| ctor 분류 | 변형 (registry name) | 주된 확장 패턴 |
|-----------|---------------------|----------------|
| `standard` (25종) | Vanilla, ASR, PI, LP, Projection, Deterministic, Robust, Feedback, PR, Uncertainty, DIAL, CMA, SVG, Biased, Kernel, Flow, SG, Transformer, TD, GN, Residual, PGD, TR, RF, Step | 다양 (아래 참조) |
| `no_cost` (5종) | Log, Tsallis, Risk-Aware, Spline, SVMPC | 대부분 패턴 A (2-인자 생성자, `cost_function` kwarg 미지원) |
| `smooth` (1종) | Smooth | ΔU 리프팅 (2-인자 생성자) |
| `tube` (1종) | Tube | 패턴 E (AncillaryController 결합) |
| `contingency` (1종) | Contingency | 패턴 B + 내부 MPPI 중첩 (`safety_cost_function` 추가 인자) |
| `obstacle_in_params` (8종) | DBaS, DRPA, CSC, DualGuard, CBF, Shield, C2U, Conformal-CBF | 패턴 (비용 주입) + params에 장애물 필드 |

확장 패턴 관점의 재분류 (이 문서의 A~F):

| 패턴 | 핵심 아이디어 | 대표 변형 | 이 문서 절 |
|------|-------------|-----------|-----------|
| **A. 가중치 교체** | `_compute_weights()`만 오버라이드 | Log, Tsallis, Risk-Aware, ASR | §2 |
| **B. compute_control 전체 교체** | 다중 반복 / 혼합 샘플링 루프 | DIAL, CMA, GN, Biased, SVG, dsMPPI | §3 |
| **C. 샘플러 교체** | `NoiseSampler` 구현체 주입/내장 | LP, TR(HaltonLCD), RF(HermiteSpline), Colored, Uncertainty | §4 |
| **D. 최적화 관점 확장** | MPPI 스텝을 최적화 스텝으로 일반화 | PGD, TR, GN | §5 |
| **E. 피드백/구조 결합** | MPPI 해 + 피드백 제어기 | Tube, Robust, Feedback(F-MPPI) | §6 |
| **F. 학습 결합** | NN proposal/value + 온라인 학습 | Step, Flow, SG, Transformer, TD | §7 |

> 패턴은 배타적이지 않다. 예컨대 TR-MPPI는 C(결정론적 샘플러) + D(신뢰 영역
> 투영)를 동시에 쓰고, GN-MPPI는 B(다중 반복 루프) 위에 D(2차 업데이트)를 얹는다.

**흔한 실수 (ctor 분류를 무시할 때)**: MEMORY에도 기록된 사례로,
`SmoothMPPIController.__init__()`은 `cost_function` 키워드를 받지 않는 2-인자
생성자다. 레지스트리의 `ctor="smooth"`/`"no_cost"` 분기가 정확히 이를 처리한다
(all_37_variants_benchmark.py:555-557). 새 벤치마크에서 모든 변형을
`ctor="standard"`로 가정하면 `TypeError: unexpected keyword argument`로 즉사한다.

---

## 2. 패턴 A — 가중치 함수 교체

가장 저렴한 확장이다. 파이프라인의 5단계(`_compute_weights`)만 바꾸고
샘플링·rollout·업데이트·shift는 전부 base에 위임한다. 서브클래스 파일이
160~180줄에 불과한 이유다 (log_mppi.py 165줄, tsallis_mppi.py 166줄,
risk_aware_mppi.py 183줄).

### 2.1 Log-MPPI — log-space softmax

> 이론: [MPPI_THEORY.md §4 Log-MPPI](../MPPI_THEORY.md)

```python
# mppi_controller/controllers/mppi/log_mppi.py:55-92 (발췌)
def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
    # 1. Baseline 적용 (선택적)
    if self.use_baseline:
        baseline = np.min(costs)
        costs_shifted = costs - baseline
    ...
    # 2. Log-space 가중치 계산
    log_weights_unnorm = -costs_shifted / self.params.lambda_
    # 3. Log-sum-exp trick으로 정규화
    log_Z = self._log_sum_exp(log_weights_unnorm)
    # 4. 정규화된 log 가중치
    log_weights = log_weights_unnorm - log_Z
    # 5. Exp-space 변환
    weights = np.exp(log_weights)
    # 6. 수치 검증 (sum = 1)
    weights = weights / np.sum(weights)  # 추가 정규화 (수치 오차 보정)
```

`_log_sum_exp()`(log_mppi.py:112-125)는 교과서적 trick 그대로다:
`max_log = np.max(log_values)`를 뺀 뒤 `max_log + log(Σ exp(x - max))`.

**왜 수치적으로 안전한가**: `exp(-S/λ)`를 직접 계산하면 비용이 크거나 λ가 작을 때
underflow(전부 0 → 0으로 나눔), 반대 부호에서 overflow가 발생한다.
log-sum-exp trick은 `max(x)`를 밖으로 빼서 `exp()` 인자를 항상 `≤ 0`으로 만든다
— `exp(x_i - max)`의 최댓값이 정확히 1이므로 overflow가 원천 차단되고, 최소한
최고 샘플 하나는 절대 underflow하지 않는다.

**base와의 관계**: 사실 base의 `_compute_weights()`도 `min_cost`를 빼는 shift를
한다(base_mppi.py:290-292). `exp(-(costs-min)/λ)`에서 최솟값 샘플이 `exp(0)=1`이
되므로 base도 overflow는 없다. Log-MPPI의 실익은 (1) **log 가중치 자체를 통계로
보존**한다는 점(log_weights_stats_history, log_mppi.py:98-108 — base에서는 이미
0으로 붕괴된 값밖에 못 봄), (2) `use_baseline=False`로 shift를 끄고도 안전하다는
점이다.

**흔한 실수**: 5번에서 `exp(log_weights)` 후에도 92행에서 한 번 더
`weights / np.sum(weights)`로 재정규화한다. "이론상 이미 정규화됐는데 왜?" —
부동소수점 합이 1.0에서 1e-16 수준 어긋나는데, 이 오차가
`np.random.choice(p=weights)` 같은 후속 소비자에서 `ValueError: probabilities do
not sum to 1`을 일으킬 수 있다. 방어적 재정규화는 관용적 패턴이다.

### 2.2 Tsallis-MPPI — q-exponential과 q→1 극한

> 이론: [MPPI_THEORY.md §5 Tsallis-MPPI](../MPPI_THEORY.md)

q-exponential은 별도 유틸이 아니라 `_compute_weights()` 안에 인라인으로
구현되어 있다 (utils 모듈에 `q_exponential` 함수는 존재하지 않음 — grep 확인):

```python
# mppi_controller/controllers/mppi/tsallis_mppi.py:83-98
# 2. Tsallis q-exponential 가중치
if np.isclose(q, 1.0, atol=1e-6):
    # q=1: Vanilla MPPI (Shannon entropy)
    unnormalized_weights = np.exp(-costs_shifted / lambda_)
else:
    # q≠1: Tsallis entropy
    # w_k = [1 - (1-q) * S_k / λ]_+^(1/(1-q))
    exponent = -costs_shifted / lambda_
    argument = 1.0 + (1.0 - q) * exponent

    # 양수 부분만 (절단)
    argument = np.maximum(argument, 0.0)

    # q-exponential
    power = 1.0 / (1.0 - q)
    unnormalized_weights = np.power(argument, power)
```

**q→1 극한 처리**: 수학적으로 `exp_q(x) = [1+(1-q)x]^{1/(1-q)}`는 q→1에서
`exp(x)`로 수렴하지만, 코드에서 q=1을 그대로 대입하면 `1/(1-q)`가 0으로 나누기가
된다. 그래서 **극한을 해석적으로 계산하지 않고 `np.isclose(q, 1.0, atol=1e-6)`
분기로 `np.exp`를 직접 호출**한다(84행). 이것이 "Vanilla MPPI를 특수 케이스로
포함"한다는 주장의 코드적 실체다.

**절단(truncation)의 부작용과 방어**: q>1이면 `power=1/(1-q)<0`인데
`argument`가 0인 샘플에서 `0^음수 = inf`가 될 것 같지만, 실제로는
`np.maximum(argument, 0.0)` 후 `np.power(0.0, 음수)`가 `inf`를 반환하고 경고를
낸다 — 다행히 q>1에서는 `argument = 1 - (q-1)·S/λ`이 비용 큰 샘플에서 0으로
절단되고, 그 경우 `1/(1-q)<0`이므로 실질적으로는 비용이 임계값을 넘는 샘플의
가중치가 0이 되는 hard cutoff다. 전부 절단되는 극단 상황은
`weights_sum > 0` 체크 후 균등 가중치 `np.ones(K)/K`로 폴백해 방어한다
(tsallis_mppi.py:100-106).

**트레이드오프**: q 하나로 heavy-tail(q<1, 탐색)/light-tail(q>1, 활용)을
연속 조절할 수 있지만, q>1의 절단은 유효 샘플 수를 갑자기 줄인다. 그래서
통계에 `num_zero_weights`를 기록해(tsallis_mppi.py:118) 절단 정도를 모니터링한다.

### 2.3 Risk-Aware MPPI — 정렬 기반 CVaR 가중

> 이론: [MPPI_THEORY.md §6 Risk-Aware MPPI (CVaR)](../MPPI_THEORY.md)

정렬 → 상위 α 분위 선택 → 부분집합 softmax → 나머지 0. 전형적인
"분위수 기반 가중" 구현이다:

```python
# mppi_controller/controllers/mppi/risk_aware_mppi.py:84-107 (발췌)
# 1. 비용을 오름차순 정렬 (낮은 비용이 좋음)
sorted_indices = np.argsort(costs)
# 2. CVaR cutoff (상위 α*100% 샘플 선택)
cvar_count = max(1, int(K * alpha))
cvar_indices = sorted_indices[:cvar_count]
# 3~5. CVaR set 비용 → baseline shift → softmax
cvar_costs = costs[cvar_indices]
baseline = np.min(cvar_costs)
cvar_costs_shifted = cvar_costs - baseline
exp_costs = np.exp(-cvar_costs_shifted / lambda_)
cvar_weights_unnormalized = exp_costs
# 6. 전체 샘플에 대한 가중치 (CVaR 외부는 0)
weights = np.zeros(K)
weights[cvar_indices] = cvar_weights_unnormalized / np.sum(
    cvar_weights_unnormalized
)
```

핵심은 6번: **반환 shape은 항상 (K,)로 유지**하고 절단된 샘플은 0을 채운다.
base의 6단계 `np.sum(weights[:, None, None] * noise, axis=0)`(base_mppi.py:160)이
`(K,)` 가중치를 기대하므로, 부분집합만 반환하면 브로드캐스트가 깨진다.
`max(1, int(K*alpha))`(89행)는 α가 아주 작아도 최소 1개 샘플을 보장하는 가드다.

같은 "정렬 기반" 계열의 일반화가 ASR-MPPI(스펙트럴 리스크)다. CVaR의 계단형
절단 대신 **왜곡 함수 도함수 φ'(q)를 분위수별 연속 가중**으로 곱한다:

```python
# mppi_controller/controllers/mppi/spectral_risk_mppi.py:85-106 (발췌)
sorted_indices = np.argsort(costs)
sorted_costs = costs[sorted_indices]
# 2. 분위수 계산 (0, 1/K, ..., (K-1)/K)
quantiles = np.arange(K) / K
# 3. 왜곡 함수 도함수 (density)
distortion_weights = self._eval_distortion_derivative(quantiles)
# 4. Softmax (baseline 적용)
exp_costs = np.exp(-(sorted_costs - baseline) / lambda_)
# 5. Spectral weights = φ'(q) · softmax
spectral_weights = distortion_weights * exp_costs
...
weights[sorted_indices] = spectral_weights / total
```

**흔한 실수**: 정렬 기반 가중에서 `weights[sorted_indices] = ...` 형태의
**역-인덱싱을 빼먹는 것**. `sorted_costs`에 대한 가중치를 그대로 반환하면
가중치와 노이즈 샘플의 순서가 어긋나서, 최악의 경우 최고 비용 샘플에 최대
가중치를 주게 된다. 시뮬레이션은 돌아가되 성능만 조용히 망가지는 종류의
버그라 특히 위험하다.

**패턴 A 종합 트레이드오프**: 구현 비용이 가장 낮고 base와의 동작 차이를
가중치 하나로 격리할 수 있어 검증이 쉽다(비용 배열만 넣어보면 됨). 반면
샘플 생성 자체를 바꿀 수 없으므로 탐색 분포의 형상(공분산, 시간 상관)에는
손을 못 댄다 — 그건 패턴 B/C의 영역이다.

---

## 3. 패턴 B — `compute_control` 전체 교체

단일 sample-evaluate-update로는 표현할 수 없는 구조(다중 반복, 혼합 분포,
반복 간 상태)를 가지려면 `compute_control()`을 통째로 다시 쓴다. 이 저장소에서
이 패턴의 원형이 DIAL-MPPI라서 "DIAL 패턴"이라 부른다. 핵심 규칙:
**rollout/비용/ESS 등 base 인프라는 재사용하되, 루프와 업데이트 식은 소유한다.**

### 3.1 DIAL-MPPI — 다중 반복 + 어닐링 + 보상 정규화

> 이론: [MPPI_THEORY.md §11 DIAL-MPPI (확산 어닐링)](../MPPI_THEORY.md)

루프 구조 (dial_mppi.py:112-156):

```python
# mppi_controller/controllers/mppi/dial_mppi.py:112-148 (발췌)
for i in range(n_iters):
    # 1. 어닐링된 노이즈 스케일 계산
    traj_scale = self.dial_params.traj_diffuse_factor ** i
    # annealed_sigma: (N, nu) = horizon_profile (N,) * sigma (nu,) * traj_scale
    annealed_sigma = (
        self._horizon_profile[:, None] * self.params.sigma[None, :] * traj_scale
    )
    # 2. 샘플링: W ~ N(0, annealed_sigma)
    rng_noise = np.random.standard_normal((K, N, nu))
    W = rng_noise * annealed_sigma[None, :, :]
    # 3. 샘플 제어 생성 + 클리핑
    sampled_controls = self.U[None, :, :] + W
    ...
    # 4. Rollout + 비용 계산 (기존 인프라 재사용)
    trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
    costs = self.cost_function.compute_cost(...)
    # 5. 가중치 계산
    if self.dial_params.use_reward_normalization:
        weights = self._compute_weights_normalized(costs)
    else:
        weights = self._compute_weights(costs, self.params.lambda_)
    # 6. 전체 교체 업데이트: U = Σ w_k * sampled_controls_k
    self.U = np.sum(weights[:, None, None] * sampled_controls, axis=0)
```

주목할 코드 레벨 차이 3가지:

1. **어닐링 스케줄이 2축**: 반복 축(`traj_diffuse_factor ** i` — 반복마다 노이즈
   감소, 전역→지역)과 호라이즌 축(`_horizon_profile` — t=0 작게, t=N-1 크게;
   dial_mppi.py:62-76의 선형 보간). 두 프로파일의 외적으로 `(N, nu)` sigma
   행렬을 만들어 브로드캐스트한다.
2. **전체 교체 업데이트**: base는 `U += Σ w ε`(증분)인데 DIAL은
   `U = Σ w·(U+ε)`(전체 교체). 수학적으로 Σw=1이면 동치지만, **클리핑이 개입하면
   다르다** — `sampled_controls`는 클리핑된 값이므로 전체 교체는 결과가 항상
   제약 내부의 볼록 조합이 되어 다중 반복에서 더 안정적이다.
3. **warm/cold start 상태**: `self._is_first_call` 플래그로 첫 호출은
   `n_diffuse_init`(예: 10회), 이후는 `n_diffuse`(예: 3회)로 반복 수를 바꾼다
   (dial_mppi.py:98-103). `reset()`에서 반드시 이 플래그를 복원해야 한다
   (dial_mppi.py:251-254) — 복원을 잊으면 reset 후에도 warm start로 동작하는
   미묘한 버그가 된다.

보상 정규화 (r-mean)/std/λ:

```python
# dial_mppi.py:192-218 (발췌)
def _compute_weights_normalized(self, costs: np.ndarray) -> np.ndarray:
    rewards = -costs
    std = np.std(rewards)
    if std < 1e-10:
        return np.ones(len(costs)) / len(costs)  # 균등 가중치 폴백
    normalized = (rewards - np.mean(rewards)) / (std + 1e-10)
    scaled = normalized / self.params.lambda_
    scaled -= np.max(scaled)                     # 수치 안정 max-shift
    exp_scaled = np.exp(scaled)
    weights = exp_scaled / np.sum(exp_scaled)
```

**왜 필요한가**: 다중 반복에서는 반복이 진행될수록 비용 스케일이 줄어든다.
고정 λ의 softmax는 초반엔 지나치게 뾰족하고 후반엔 지나치게 평평해진다.
표준화(z-score)를 먼저 하면 **λ가 비용의 절대 스케일과 무관해져** 반복 전체에서
일관된 선택압을 유지한다. 이 함수는 CMA(cma_mppi.py:202-222),
GN(gn_mppi.py:327-348), Biased(biased_mppi.py:244-257)에 동일하게 복제되어
있다 — 패턴 B 계열의 사실상 공용 유틸이다(현재는 각 파일에 중복 구현).

**흔한 실수**: `std < 1e-10` 가드 없이 나누면 모든 샘플 비용이 같은
(예: 제자리 정지 초기 스텝) 순간 NaN 폭탄이 터진다. 4개 구현 모두 이 가드를
갖고 있다.

### 3.2 CMA-MPPI — 공분산 적응 EMA + persistence + receding shift

> 이론: [MPPI_THEORY.md §17 CMA-MPPI](../MPPI_THEORY.md)

DIAL과 같은 다중 반복 뼈대에 **per-timestep 대각 공분산 상태**를 추가한다.
공분산은 생성자에서 `(N, nu)` 행렬로 초기화되어 제어 스텝을 넘어 지속된다:

```python
# mppi_controller/controllers/mppi/cma_mppi.py:50-53
# Per-timestep 대각 공분산 초기화: (N, nu)
initial_sigma = params.sigma * params.cov_init_scale
self.cov = np.outer(np.ones(params.N), initial_sigma ** 2)  # (N, nu)
self._initial_cov = self.cov.copy()
```

반복 루프 안의 공분산 적응 (CMA 핵심):

```python
# cma_mppi.py:137-150
# 7. 공분산 적응 (CMA 핵심!)
diff = sampled_controls - self.U[None, :, :]  # (K, N, nu)
cov_est = np.sum(
    weights[:, None, None] * (diff ** 2), axis=0
)  # (N, nu) — 가중 분산

# EMA 안정화
alpha = self.cma_params.cov_learning_rate
self.cov = (1 - alpha) * self.cov + alpha * cov_est

# 클램핑
sigma_min_sq = self.cma_params.sigma_min ** 2
sigma_max_sq = self.cma_params.sigma_max ** 2
self.cov = np.clip(self.cov, sigma_min_sq, sigma_max_sq)
```

세 줄이 각각 실패 모드 하나씩을 막는다:

- **가중 분산 추정**(diff²의 w-가중합): 좋은 샘플들이 퍼져 있으면 Σ 확대(탐색),
  모여 있으면 축소(활용). "업데이트된 U" 기준 편차라는 점에 주의 — CMA-ES의
  rank-μ 업데이트와 같은 정신이다.
- **EMA**: 단일 반복의 추정 노이즈가 공분산을 널뛰게 하는 것을
  `cov_learning_rate`로 감쇠.
- **클램핑**: `sigma_min`은 조기 붕괴(모든 샘플이 같아져 ESS→K, 학습 정지),
  `sigma_max`는 발산을 방지.

공분산 persistence + receding shift — U와 Σ를 **함께** 시프트한다:

```python
# cma_mppi.py:162-166
# Receding horizon shift (U와 Σ 모두)
self.U = np.roll(self.U, -1, axis=0)
self.U[-1, :] = 0.0
self.cov = np.roll(self.cov, -1, axis=0)
self.cov[-1, :] = self._initial_cov[-1]  # 마지막 timestep 리셋
```

**왜 Σ도 시프트하나**: 공분산이 per-timestep이므로, 시점 t에서 학습한 형상은
다음 제어 주기에는 t-1 위치에 해당한다. 시프트를 잊으면 학습된 공분산이 매
스텝 한 칸씩 미래로 밀려 무의미해진다. 새로 들어오는 마지막 slot은 학습된 값이
없으므로 `_initial_cov[-1]`(넓은 탐색)로 리셋한다 — `0`이나 직전 값 복사보다
안전한 선택이다. `reset()`도 `self.cov = self._initial_cov.copy()`로 복원한다
(cma_mppi.py:281-284).

### 3.3 Biased-MPPI — 혼합 샘플링과 q_s 소거

> 이론: [MPPI_THEORY.md §23 Biased-MPPI](../MPPI_THEORY.md)

DIAL 패턴의 또 다른 축: 반복이 아니라 **샘플 출처의 혼합**이다. J개는 보조
정책(pure pursuit, braking, ...)의 제안, K-J개는 표준 가우시안:

```python
# mppi_controller/controllers/mppi/biased_mppi.py:105-135 (발췌)
# 1. 정책 샘플 생성
policy_controls = self._generate_policy_samples(state, reference_trajectory, N, nu)
n_policy = policy_controls.shape[0]
# 2. 가우시안 샘플 생성
n_gaussian = K - n_policy
gaussian_controls = self._generate_gaussian_samples(N, nu, n_gaussian)
# 3. 혼합: (K, N, nu)
all_controls = np.concatenate([policy_controls, gaussian_controls], axis=0)
...
# 5. 가중치 계산 (q_s 소거 — 표준 MPPI 가중치와 동일)
weights = self._compute_weights(costs, self._current_lambda)
# 6. 전체 교체 업데이트: U = Σ ω_k V_k
self.U = np.sum(weights[:, None, None] * all_controls, axis=0)  # (N, nu)
```

혼합 분포에서 샘플링했는데 importance weight 보정이 없는 이유가 128행 주석의
"q_s 소거"다 — 논문(Trevisan & Alonso-Mora, RA-L 2024)의 핵심 정리로, 제안
분포 밀도가 가중치 분자·분모에서 상쇄되어 `softmax(-S/λ)` 그대로 쓸 수 있다.
**코드에서는 base의 `_compute_weights()`를 그대로 호출하는 것**이 그 정리의
구현이다. 단, 이때 업데이트는 반드시 전체 교체(`U = Σ ω V`)여야 한다 — 정책
샘플은 `U + ε` 형태가 아니어서 "노이즈"라는 개념 자체가 없기 때문이다. base의
증분 업데이트(`U += Σ w ε`)를 재사용할 수 없는, 패턴 B가 강제되는 사례다.
부가 기능으로 ESS 기반 적응 λ(`_adapt_lambda`, biased_mppi.py:259-278)가 붙는다:
`ess/K < ess_min_ratio`면 λ 증가(평탄화), 반대면 감소하며, `_current_lambda`
상태는 `reset()`에서 초기값 복원된다(biased_mppi.py:303-306).

**패턴 B 공통 체크리스트** (전체 교체 구현 시 빠뜨리기 쉬운 것):
1. `info` dict 계약 준수 — `sample_trajectories`, `sample_weights`,
   `best_trajectory`, `temperature`, `ess`, `num_samples`는 시각화/벤치마크가
   소비하므로 반드시 채운다 (CLAUDE.md 인터페이스 규칙).
2. receding horizon shift를 직접 수행 (`np.roll` + 마지막 0) — base 것을 안
   거치므로 잊기 쉽다.
3. `reset()`에서 루프 상태(`_is_first_call`, `_current_lambda`, 통계) 복원.
4. 다중 반복 시 `optimal_control`은 **shift 전** `self.U[0]`에서 추출
   (dial_mppi.py:158-163 순서 참조).

---

## 4. 패턴 C — 샘플러 교체

### 4.1 `NoiseSampler` 계약

```python
# mppi_controller/controllers/mppi/sampling.py:12-35 (발췌)
class NoiseSampler(ABC):
    @abstractmethod
    def sample(
        self,
        U: np.ndarray,          # (N, nu) 명목 제어 시퀀스
        K: int,                  # 샘플 개수
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns: noise: (K, N, nu) 노이즈 샘플"""
```

계약의 미묘한 부분: **클리핑은 노이즈에 반영해서 반환**해야 한다.
`GaussianSampler`의 구현이 레퍼런스다:

```python
# sampling.py:78-86
if control_min is not None and control_max is not None:
    sampled_controls = U + noise            # 브로드캐스트 (K, N, nu)
    sampled_controls = np.clip(sampled_controls, control_min, control_max)
    # 노이즈 = 클리핑된 제어 - 명목 제어
    noise = sampled_controls - U
```

base의 업데이트 식 `U += Σ w ε`이 사용하는 ε이 바로 이 반환값이므로, 클리핑
후 재계산(`noise = clipped - U`)을 생략하면 제약 밖의 노이즈로 U가 갱신되어
클리핑 경계에서 편향이 생긴다.

### 4.2 LowPassSampler — scipy sosfilt 벡터화 (LP-MPPI)

> 이론: [MPPI_THEORY.md §22 LP-MPPI](../MPPI_THEORY.md)

필터 계수는 생성자에서 사전 계산한다 — `_update_filter()`(sampling.py:415-423)가
`wn = cutoff_freq / nyquist`로 정규화한 뒤
`butter(filter_order, wn, btype='low', output='sos')`를 저장하고,
Nyquist 초과/비양수 차단 주파수면 `self._sos = None`으로 필터를
무효화(bypass)한다. `set_cutoff_freq()`/`set_filter_order()`(425-433행)는 런타임
변경 시 이 함수를 재호출한다.

핵심 한 줄 — **(K, N, nu) 텐서를 Python 루프 없이 일괄 필터링**:

```python
# sampling.py:456-469
noise = self.rng.normal(0.0, self.sigma, (K, N, nu))

# 필터 bypass 조건: SOS 없음 또는 시퀀스가 너무 짧음
if self._sos is not None and N > self.filter_order:
    from scipy.signal import sosfilt
    # axis=1 (시간축)을 따라 벡터화 필터링
    noise = sosfilt(self._sos, noise, axis=1)

    # 분산 정규화: 필터 후 std를 원래 sigma에 맞춤
    if self.normalize_variance:
        current_std = np.std(noise, axis=1, keepdims=True)  # (K, 1, nu)
        current_std = np.where(current_std < 1e-10, 1.0, current_std)
        noise = noise / current_std * self.sigma[None, None, :]
```

`sosfilt(..., axis=1)`은 scipy가 C 레벨에서 K×nu개의 길이 N 시퀀스를 동시에
필터링한다. 같은 목적의 `ColoredNoiseSampler`(OU 프로세스, sampling.py:144-154)가
`for k in range(K): for t in range(N):` 이중 Python 루프인 것과 대조적 —
MEMORY 기록 기준 약 10배 차이다. **시간축이 axis=1이라는 것**이 이 설계의
전부라 해도 과언이 아니다: (K, N, nu) 레이아웃 덕분에 "각 샘플의 각 제어
차원별 시계열"이 연속된 축 하나로 표현된다.

LPF는 노이즈 분산을 (통과 대역 비율만큼) 깎으므로 `normalize_variance` 옵션으로
사후 재정규화할 수 있다. `1e-10` 가드는 짧은 시퀀스에서 std≈0일 때의 0-나눗셈
방지.

컨트롤러 쪽(`LPMPPIController`)은 최소 개입의 모범이다 — **샘플러 자동 생성 +
`super().compute_control()` 호출 + info 확장**이 전부다:

```python
# mppi_controller/controllers/mppi/lp_mppi.py:55-65, 84-88 (발췌)
if noise_sampler is None:
    noise_sampler = LowPassSampler(
        sigma=params.sigma, cutoff_freq=params.cutoff_freq,
        filter_order=params.filter_order, dt=params.dt,
        normalize_variance=params.normalize_variance,
    )
super().__init__(model, params, cost_function, noise_sampler)
...
control, info = super().compute_control(state, reference_trajectory)
smoothness = self._compute_smoothness_stats(info)
info["smoothness_stats"] = smoothness
```

가중치 계산은 Vanilla softmax 그대로 — 변형의 정체성이 전적으로 샘플러에 있다.

**흔한 실수** (MEMORY에 기록된 실전 사례): `sigma_override or self.sigma`처럼
numpy 배열에 `or`를 쓰면 `ValueError: truth value of an array is ambiguous`.
반드시 `x if x is not None else default` 패턴을 쓴다.

### 4.3 HaltonLCDSampler — 결정론적 샘플링 (TR-MPPI)

> 이론: [MPPI_THEORY.md §39 TR-MPPI](../MPPI_THEORY.md)

`tr_mppi.py`는 `NoiseSampler`를 상속한 결정론적 샘플러를 내장한다. 구성은
"저불일치 수열 → 균등분포 (0,1) → 역정규 CDF → 표준정규 유사 샘플":

```python
# mppi_controller/controllers/mppi/tr_mppi.py:110-119
def unit_samples(self, K: int, N: int, nu: int) -> np.ndarray:
    """결정론적 표준정규 유사 샘플 (K, N, nu)"""
    d = N * nu
    cols = []
    for j in range(d):
        base = _PRIMES[j % len(_PRIMES)]
        cols.append(_van_der_corput(K, base, skip=self.skip + 1))
    u = np.stack(cols, axis=1)            # (K, d)
    z = _inverse_normal_cdf(u)            # (K, d)
    return z.reshape(K, N, nu)
```

- 차원마다 다른 소수(prime) 진법의 van der Corput 수열
  (`_van_der_corput`, tr_mppi.py:79-90 — 진법 전개를 numpy 벡터 연산으로 처리)을
  쓰는 것이 Halton 수열의 정의다. `_PRIMES` 목록(tr_mppi.py:74-76)은 46개라
  `j % len(_PRIMES)`로 순환한다 — N·nu가 46을 넘으면 차원 간 상관이 생길 수
  있는 알려진 한계.
- `skip`(기본 20)은 Halton 초기 항의 강한 상관을 버리는 관례.
- `_inverse_normal_cdf`(tr_mppi.py:32-71)는 `scipy.special.ndtri`를 우선 쓰고
  scipy 부재 시 Acklam 유리근사(정확도 ~1e-9)로 폴백한다. 입력은
  `np.clip(u, 1e-9, 1-1e-9)`로 클램프해 ±inf를 차단(39행).

컨트롤러는 플래그로 확률/결정론 샘플링을 스위치한다:

컨트롤러 쪽 `_sample_noise()`(tr_mppi.py:180-187)는
`use_deterministic_sampling` 플래그 하나로 `self._lcd.unit_samples(K, N, nu)`와
`self._rng.normal(0, 1, (K, N, nu))`를 스위치한 뒤 `z * sigma`로 스케일한다 —
결정론/확률 샘플링이 단위 정규 샘플 생성 한 줄만 다르다.

**트레이드오프**: 동일 K에 대해 항상 같은 샘플이므로 (1) 분산이 낮아 적은 K로
수렴 안정, (2) 완전한 재현성. 대신 매 스텝 같은 격자를 재사용하므로 탐색의
"우연한 발견"이 없다 — TR-MPPI는 이를 공분산 적응(`_adapt_covariance`,
tr_mppi.py:274-291)과 신뢰 영역 스텝 제한으로 보완한다(§5 참조).

### 4.4 HermiteSplineSampler — 저차원 knot 파라미터화 (RF-MPPI)

> 이론: [MPPI_THEORY.md §40 RF-MPPI](../MPPI_THEORY.md)

주의: 이름은 Sampler지만 `NoiseSampler` ABC를 상속하지 않는 **별도 기저 유틸**
이다 — 샘플링 공간 자체가 (N, nu)가 아니라 (M, nu) knot 공간이어서 인터페이스가
다르다. 큐빅 Hermite 기저 행렬을 생성자에서 한 번만 만든다:

```python
# mppi_controller/controllers/mppi/rf_mppi.py:70-81 (발췌, _build_basis)
for i, t in enumerate(times):
    m, s, tau = self._segment_of(float(t))
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    B_p[i, m] += h00;        B_p[i, m + 1] += h01
    B_v[i, m] += tau * h10;  B_v[i, m + 1] += tau * h11
```

Hermite 보간 `u(t) = h00·p_m + Δτ·h10·v_m + h01·p_{m+1} + Δτ·h11·v_{m+1}`을
**시점별 기저 계수로 미리 전개**해 두면, 재구성은 순수 행렬 곱이 된다:

```python
# rf_mppi.py:102-109
def reconstruct(self, P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """(M,nu) 위치/속도 knot → (N,nu) 제어 시퀀스."""
    return self.B_p @ P + self.B_v @ V

def reconstruct_batch(self, P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """(K,M,nu) 배치 → (K,N,nu) 제어 시퀀스."""
    return np.einsum("nm,kmu->knu", self.B_p, P) + \
        np.einsum("nm,kmu->knu", self.B_v, V)
```

컨트롤러의 샘플-업데이트 흐름 — **섭동도, 가중 평균도 knot 공간에서**:

```python
# rf_mppi.py:176-199 (발췌)
# knot 공간 섭동 샘플
dP = self._rng.normal(0.0, 1.0, (K, M, nu)) * self._sigma_p[None, None, :]
...
P_k = self._P[None] + dP                            # (K, M, nu)
V_k = self._V[None] + dV
# 매끄러운 제어 시퀀스 재구성 (K, N, nu)
sampled_controls = self.spline.reconstruct_batch(P_k, V_k)
...
weights = self._compute_weights(costs, self.params.lambda_)
# knot 갱신 (가중 평균 섭동)
self._P = self._P + np.sum(weights[:, None, None] * dP, axis=0)
self._V = self._V + np.sum(weights[:, None, None] * dV, axis=0)
```

즉 최적화 변수가 `self.U`(N·nu차원)가 아니라 `self._P/_V`(2·M·nu차원)다.
`self.U`는 base 호환/시각화를 위해 매 스텝 `reconstruct()`로 파생된다
(rf_mppi.py:206-208). 재구성이 선형이라 "knot의 가중 평균 = 재구성 제어의 가중
평균"이 성립하는 점이 이 코드가 수학적으로 성립하는 이유다 (재구성이
비선형이었다면 knot 공간 평균은 제어 공간 평균과 달라진다).

receding horizon warm start도 knot 공간에서: 스플라인을 시간 +1 이동한 위치에서
새 knot 값·미분값을 재평가한다 (`eval_shifted_knots`, rf_mppi.py:112-123 —
도함수 기저 `dB_p/dB_v`는 rf_mppi.py:83-100에서 사전 계산).

**트레이드오프**: 탐색 차원이 N·nu → 2·M·nu로 줄어 K≈32~64의 적은 샘플로도
동작하고 매끄러움이 구조적으로 보장된다(커밋 f94d762 기준 K=24에서 MSSD ~13배
개선). 대신 표현력이 스플라인 공간으로 제한되어 급격한 회피 기동(비상 제동
등)은 knot 수 M을 올리지 않으면 표현 불가.

**흔한 실수**: knot 갱신 후 `clamp_endpoints_vel`(rf_mppi.py:201-203)처럼
끝점 속도를 0으로 클램프하는 후처리를 잊으면 스플라인이 호라이즌 양 끝에서
발산 방향의 접선을 학습해 진동할 수 있다.

---

## 5. 패턴 D — 최적화 관점 확장

MPPI 업데이트 `U += Σ w ε`을 "1차 최적화의 한 스텝"으로 해석하면, 스텝 크기·
반복 수·전처리 행렬·2차 정보로 일반화할 수 있다. 이 패턴의 특징은 **기본값에서
Vanilla로 퇴화하는 상위집합(graceful superset)** 설계다.

### 5.1 PGD-MPPI — 전처리 경사 하강으로 일반화

> 이론: [MPPI_THEORY.md §38 PGD-MPPI](../MPPI_THEORY.md)

`step_size`(α)와 `n_grad_steps`가 어디에 곱해지고 어디를 감싸는지가 전부다:

```python
# mppi_controller/controllers/mppi/pgd_mppi.py:109-144 (발췌)
for it in range(self.pgd_params.n_grad_steps):
    sigma = self._current_sigma()
    # 첫 스텝이거나 재샘플링 옵션이면 새 노이즈
    if it == 0 or self.pgd_params.resample_each_step:
        noise = self._sample_noise(K, sigma)  # (K, N, nu)
    sampled_controls = mu[None] + noise
    ...
    weights = self._compute_weights(costs, lambda_)

    # 전처리 경사 g̃ = Σ w_i ε_i
    grad = np.sum(weights[:, None, None] * noise, axis=0)  # (N, nu)

    if self.pgd_params.normalize_gradient:
        ess = self._compute_ess(weights)
        grad = grad * (K / max(ess, 1.0))

    last_grad_norm = float(np.linalg.norm(grad))
    mu = mu + step_size * grad          # ← α가 곱해지는 유일한 지점
    ...
    if self.pgd_params.adapt_covariance:
        self._adapt_covariance(sampled_controls, mu, weights)
```

- **α의 위치**: `mu = mu + step_size * grad` 한 곳뿐이다. base의
  `self.U = self.U + weighted_noise`(base_mppi.py:161)와 비교하면
  `weighted_noise ≡ grad`, `step_size ≡ 1`이 정확히 대응한다.
- **n_grad_steps**: sample→rollout→weight→update 전체를 감싸는 외곽 루프.
  `resample_each_step=False`면 같은 노이즈를 재사용해 여러 스텝을 내딛는다
  (계산 절약, 대신 편향 위험).
- **공분산 전처리** (`_adapt_covariance`, pgd_mppi.py:176-200): Gibbs-tilted
  경험 분산 `Σ w (U-μ)²`으로 σ 스케일을 EMA 갱신 —
  `(1-β)·scale + β·(std_w/σ_base)` 후 `[cov_min_scale, cov_max_scale]` 클립.
  CMA(§3.2)·TR(tr_mppi.py:274-291)과 동일한 3단 구조(가중 분산→EMA→클립)다.

**Vanilla와의 동등성 — 정확한 의미**: 파일 docstring은 "기본값(step_size=1.0,
n_grad_steps=1, adapt_covariance=False)에서 Vanilla MPPI와 정확히 동일하게
동작"이라 쓴다(pgd_mppi.py:19-20). 이는 **업데이트 수식 경로의 동일성**이다:
루프 1회 + α=1이면 `mu += 1.0 · Σ w ε`으로 base의 6단계와 대수적으로 같고,
가중치도 base `_compute_weights` 그대로다. 단 **난수 스트림까지 같지는 않다**
— PGD는 주입된 `noise_sampler` 대신 자체 `np.random.default_rng`로 샘플링한다
(pgd_mppi.py:71-72, 82-86; docstring도 "noise_sampler ... 사용되지 않음"이라
명시, pgd_mppi.py:50-52). 그래서 회귀 테스트도 bit 비교가 아니라 동작 수준
검증이다 — `test_defaults_vanilla_equivalent`(tests/test_pgd_mppi.py:434-446)는
"기본값에서 단일 MPPI 스텝처럼 동작 (finite control, ess>1)"만 확인하고,
공분산 적응이 안 일어났음을 `np.allclose(ctrl._sigma_scale, 1.0)`으로 잡는다.

**흔한 실수**: "graceful superset"을 "같은 시드에서 같은 출력"으로 오해하는 것.
RNG 소스가 다르면 수식이 같아도 궤적은 다르다. bit-exact 재현이 필요하면 노이즈
텐서를 고정해 `_compute_weights` + 업데이트 식만 직접 비교해야 한다 (SG-MPPI의
α=0 테스트에서 같은 교훈 — compute_control의 부작용이 RNG를 분기시켜 직접
함수 비교로 우회했다, MEMORY 기록).

같은 계열인 TR-MPPI의 신뢰 영역 투영도 여기서 함께 보면 좋다 — PGD가 스텝
크기를 상수 α로 두는 반면 TR은 **스텝의 KL 크기를 측정해 사후 축소**한다:

```python
# tr_mppi.py:220-231
delta_mu = np.sum(weights[:, None, None] * noise, axis=0)  # (N, nu)

# KL_prop = ½ Σ ‖Δμ/σ‖²
sig = np.maximum(sigma, 1e-9)
kl_prop = 0.5 * float(np.sum((delta_mu / sig[None, :]) ** 2))
...
if self.tr_params.use_kl_bound and kl_prop > delta and kl_prop > 0:
    scale = np.sqrt(delta / kl_prop)
    delta_mu = delta_mu * scale
```

고정 공분산 가우시안 사이의 KL이 `½‖Δμ/σ‖²`로 닫힌형이라 투영이 스칼라 곱
하나로 끝난다 — KL이 δ를 넘으면 `sqrt(δ/KL)`배 축소해 정확히 경계 위에
올린다.

### 5.2 GN-MPPI — 가우스-뉴턴 스텝 + 라인서치 + MPPI 폴백

> 이론: [MPPI_THEORY.md §26 GN-MPPI](../MPPI_THEORY.md)

기존 K개 샘플의 (비용, 노이즈) 쌍만으로 기울기와 GGN 대각 헤시안을 복원한다
— 추가 rollout 없이 2차 정보를 얻는 것이 포인트:

```python
# mppi_controller/controllers/mppi/gn_mppi.py:248-274 (발췌)
noise_flat = noise.reshape(K, -1)               # (K, N*nu)
sigma_flat = np.tile(self.params.sigma, N)
sigma_sq = sigma_flat ** 2
# 비용 중심화 (분산 감소)
cost_centered = costs - np.mean(costs)

# 가우시안 스무딩 기울기: ∇J ≈ E[C·ε] / σ²
gradient = (
    np.mean(cost_centered[:, None] * noise_flat, axis=0)
    / (sigma_sq + 1e-10)
)
# GGN 대각 헤시안: H ≈ E[C²·ε²] / σ⁴ + reg
hessian_diag = (
    np.mean(cost_centered[:, None] ** 2 * noise_flat ** 2, axis=0)
    / (sigma_sq ** 2 + 1e-10)
)
hessian_diag += self.gn_params.regularization
# 뉴턴 스텝: δU = -H^{-1} · ∇J
step = -gradient / (hessian_diag + 1e-10)
```

기울기는 가우시안 스무딩(Stein 항등식 `∇J = E[C·ε]/σ²`)의 몬테카를로 추정이고,
헤시안은 대각 GGN 근사라 역행렬이 원소별 나눗셈으로 끝난다.
`cost_centered`(비용에서 평균을 뺌)는 추정량의 분산을 줄이는 control variate.
`regularization` 가산은 Levenberg-Marquardt 스타일 감쇠로, 헤시안 추정이
0에 가까운 차원에서 스텝 폭주를 막는다.

몬테카를로 헤시안은 신뢰도가 낮으므로 **라인서치 + 폴백**의 이중 안전장치를
두었다. 라인서치는 기하 감쇠 후보 α들을 각각 단일 rollout으로 평가하고
(gn_mppi.py:298-325), 최종 채택은 MPPI 업데이트와의 직접 비용 비교다:

```python
# gn_mppi.py:149-154
# GN vs MPPI 비교: 더 좋은 업데이트 선택
if best_gn_cost < mppi_cost:
    self.U = self.U + best_gn_update.reshape(N, nu)
    gn_used_count += 1
else:
    self.U = self.U + mppi_update
```

두 후보(GN 스텝 결과 vs 표준 MPPI 가중 평균 결과)를 **실제 rollout 비용**으로
심판하는 것이라, GN 근사가 나쁜 스텝에서는 자동으로 표준 MPPI로 퇴화한다.
`gn_used_ratio` 통계(gn_mppi.py:204-206)로 이 비율을 추적하며, 장애물
시나리오에서 ~89%가 GN 채택이었다 (MEMORY 기록).

**트레이드오프**: 후보 평가용 추가 rollout이 라인서치 스텝 수 + 1회 필요하다.
K개 배치 rollout에 비하면 (1, N, nu) rollout이라 미미하지만, `n_gn_iters`를
올리면 선형으로 늘어난다. 또 대각 헤시안 근사는 제어 차원 간 커플링(예: v-ω
상호작용)을 못 잡는다 — 그것까지 잡으려면 전체 GGN이 필요하고 (N·nu)² 행렬이
된다.

---

## 6. 패턴 E — 피드백/구조 결합

MPPI가 계산한 명목 해 위에 별도의 피드백 법칙을 얹는 패턴. MPPI 파이프라인
자체는 건드리지 않고 **그 바깥**(명목/실제 상태 분리, 게인 계산)을 추가한다.

### 6.1 Tube-MPPI + AncillaryController

> 이론: [MPPI_THEORY.md §3 Tube-MPPI](../MPPI_THEORY.md)

구조: MPPI는 (외란 없는) **명목 상태**에서 실행하고, 실제 상태와의 편차는
선형 피드백으로 보정한다:

```python
# mppi_controller/controllers/mppi/tube_mppi.py:108-130 (발췌)
# 2. Vanilla MPPI로 명목 제어 계산 — 명목 상태를 사용하여 MPPI 실행
nominal_control, mppi_info = super().compute_control(
    self.nominal_state, reference_trajectory
)
# 3. Ancillary controller로 피드백 보정
feedback_correction = self.ancillary_controller.compute_feedback(
    state, self.nominal_state
)
# 4. 최종 제어: u = u_nominal + u_fb
control = nominal_control + feedback_correction
...
# 5. 명목 상태 전파 (외란 없음)
self.nominal_state = self.model.step(
    self.nominal_state, nominal_control, self.params.dt
)
```

`super().compute_control()`에 **실제 state가 아니라 `self.nominal_state`를
넘기는 것**이 Tube-MPPI의 정체성이다. 외란이 MPPI의 warm start(`self.U`)를
오염시키지 않아 명목 계획이 매끈하게 유지되고, 외란 대응은 전부 피드백 항이
맡는다.

body-frame 오차 변환 — 피드백 게인이 로봇 진행 방향 기준으로 의미를 갖도록
world 오차를 회전시킨다:

```python
# mppi_controller/controllers/mppi/ancillary_controller.py:61-73, 99-108 (발췌)
# 1. World frame 오차
error_world = state - nominal_state
# 2. Body frame으로 변환 (Differential Drive의 경우)
if self.nx >= 3:
    theta = nominal_state[2]  # 명목 상태의 heading 사용
    error_body = self._world_to_body(error_world, theta)
...
# x, y 오차만 회전 (θ 오차는 그대로)
error_body[0] = cos_theta * e_x + sin_theta * e_y   # longitudinal
error_body[1] = -sin_theta * e_x + cos_theta * e_y  # lateral
# 3. 피드백 제어: u_fb = -K_fb @ e_body
feedback_control = -self.K_fb @ error_body
```

K_fb 적용 지점은 (1) body 변환 **후**, (2) `max_correction` 클립 **전**
(ancillary_controller.py:76-79)이다. 보정량 클립은 피드백이 명목 제어를
압도하는 것(피드백 폭주)을 막는 마지막 방어선이다.

**차원 하드코딩 이슈** — 이 모듈의 알려진 한계이자 variants×models 계열
벤치마크에서 드러난 문제다. 기본 게인 팩토리가 differential drive의 상태
레이아웃을 가정한 고정 shape이다:

```python
# ancillary_controller.py:137-146, 159-160 (발췌)
if model_type == "kinematic":
    # Differential Drive Kinematic: (2, 3) — [v, ω] ← [e_x, e_y, e_θ]
    K_fb = gain_scale * np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 1.0],
    ])
...  # "dynamic"이면 (2, 5) 고정 행렬
else:
    raise ValueError(f"Unknown model_type: {model_type}")
```

문제 지점 3곳: (1) `create_default_ancillary_controller`는 `model_type`
문자열 두 개만 알고 나머지는 `ValueError` — swerve/매니퓰레이터 등 다른 모델에
Tube를 붙이면 즉시 실패하거나, (2) K_fb를 직접 넘겨도
`_world_to_body`가 `state[2]=θ, state[0:2]=x,y` 레이아웃을 가정하므로
(ancillary_controller.py:66-67, 101-107) 상태 배치가 다른 모델에서는 **오차
회전이 물리적으로 엉뚱한 축에 적용**된다. (3) `nx >= 3`이면 무조건 body 변환을
타는 휴리스틱(63행)이라 3차원 이상이지만 heading이 index 2가 아닌 상태 공간에서
조용히 틀린다. 그래서 통합 벤치마크는 diff-drive 전제 하에 K_fb를 명시 주입한다
(all_37_variants_benchmark.py:189-192, `"K_fb": np.array([[2,0,0],[0,2,0]])`).
새 모델에 Tube를 이식할 때는 K_fb shape과 상태 레이아웃 가정을 반드시 함께
검토해야 한다.

### 6.2 F-MPPI — Riccati 게인으로 solve 재사용

> 이론: [MPPI_THEORY.md §33 F-MPPI (Feedback Reuse MPPI)](../MPPI_THEORY.md)

Tube가 "매 스텝 MPPI + 피드백"이라면, F-MPPI는 "가끔 MPPI + 나머지는 피드백만"
이다. 분기 조건:

```python
# mppi_controller/controllers/mppi/feedback_mppi.py:104-114
need_full_solve = (
    self._reuse_counter == 0
    or self._nominal_trajectory is None
    or self._feedback_gains is None
    or self._current_step_in_sequence >= self.params.N - 1
)
if need_full_solve:
    return self._full_solve(state, reference_trajectory)
else:
    return self._feedback_step(state, reference_trajectory)
```

`_full_solve()`는 표준 MPPI 파이프라인 수행 후 (1) shift **전의** 최적 시퀀스
`optimal_U`를 따로 보존하고(feedback_mppi.py:160-162), (2) 그 명목 궤적을 따라
유한차분 야코비안을 구하고, (3) backward Riccati로 게인을 만들고, (4)
`_reuse_counter = reuse_steps`를 세팅한다(feedback_mppi.py:182-186).

유한차분 야코비안(`_compute_jacobians`, feedback_mppi.py:277-333)은 모델의
해석적 미분 없이 `model.step()` 호출만으로 국소 선형화를 얻는다 — 상태 축은
`A_list[t,:,j] = (f(x+εe_j,u) - f(x-εe_j,u)) / 2ε` 중앙 차분(309-319행), 제어 축
B도 동일 구조(321-331행)로 시점당 2(nx+nu)회 step 호출이 든다.

backward Riccati 재귀 — 수치 안정 장치가 3중이다 (정칙화 reg, solve 실패 시
lstsq 폴백, 게인 클립 + P 대칭화):

```python
# feedback_mppi.py:391-420 (발췌)
for t in range(N - 1, -1, -1):
    A = A_list[t];  B = B_list[t]
    BtP = B.T @ P
    M = R_matrix + BtP @ B + reg        # (nu, nu)
    try:
        K_t = -np.linalg.solve(M, BtP @ A)
    except np.linalg.LinAlgError:
        K_t = -np.linalg.lstsq(M, BtP @ A, rcond=None)[0]
    K_t = np.clip(K_t, -clip_val, clip_val)
    gains[t] = K_t
    AtP = A.T @ P
    P = Q_matrix + AtP @ A + AtP @ B @ K_t   # P_t = Q + AᵀPA + AᵀPB·K_t
    P = 0.5 * (P + P.T)                       # 대칭화 (수치 안정)
```

reuse 스텝은 rollout·비용 평가가 전혀 없다 — 인덱싱과 행렬-벡터 곱 하나:

```python
# feedback_mppi.py:443-460 (발췌, _apply_feedback)
u_nom = self._nominal_controls[step_idx]
x_nom = self._nominal_trajectory[step_idx]
K_t = self._feedback_gains[step_idx]
state_error = state - x_nom
# Angle wrapping for theta component if present
if len(state_error) >= 3:
    state_error[2] = np.arctan2(np.sin(state_error[2]), np.cos(state_error[2]))
correction = K_t @ state_error
control = u_nom + correction
```

그래서 full solve ~3ms vs reuse ~0.01ms, `reuse_steps=3`이면 계산량 75% 절감
(MEMORY 기록). 각도 오차의 `arctan2` wrapping(451-454행)이 없으면 θ가 ±π
경계를 넘는 순간 오차가 2π만큼 튀어 피드백이 로봇을 반대로 회전시킨다 —
피드백 계열의 고전적 함정이다. 참고로 이 코드에도 `state_error[2]=θ`라는
레이아웃 가정이 남아 있다(§6.1과 동일 계열의 하드코딩).

**트레이드오프**: reuse 중에는 장애물 회피가 갱신되지 않는다(비용 평가 자체가
없음). 명목 궤적이 유효한 짧은 구간(reuse_steps ≤ 3~5)에서만 안전하며,
`_current_step_in_sequence >= N-1` 가드(feedback_mppi.py:108)가 명목 시퀀스
소진을 막는다.

---

## 7. 패턴 F — 학습 결합의 공통 뼈대

Flow/SG/Transformer/TD/Step-MPPI가 모두 같은 4요소 뼈대를 공유한다:
**(1) torch 없음 폴백, (2) zero-init 출력층, (3) ring buffer 데이터 수집,
(4) 주기적 온라인 학습 트리거.** 최신 구현인 Step-MPPI(43번째)를 대표로 해부한다.

> 이론: [MPPI_THEORY.md §41 Step-MPPI](../MPPI_THEORY.md)

### 7.1 torch 없음 폴백

```python
# mppi_controller/controllers/mppi/step_mppi.py:30-35
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch 부재 환경
    _TORCH_AVAILABLE = False
```

모듈 최상단에서 import를 시도하고 실패 플래그만 남긴다. `nn.Module` 서브클래스
정의 자체를 `if _TORCH_AVAILABLE:` 블록 안에 넣어(step_mppi.py:38) torch가 없을
때 클래스 정의부에서 NameError가 나는 것도 막는다. 컨트롤러는
`self._use_net = _TORCH_AVAILABLE and params.use_learned_proposal`
(step_mppi.py:180-183)로 두 조건을 AND 해서 학습 경로를 켜고, 꺼진 경우
`mean_delta = np.zeros((N, nu)); mu = U_warm; sigma_eff = self._base_sigma`
(step_mppi.py:246-248)로 폴백한다.

폴백 시 proposal이 항등(Δμ=0, σ=σ_base)이 되어 **Vanilla MPPI와 동일하게
동작**한다. 벤치마크 레지스트리의 torch 그룹 전체 skip
(all_37_variants_benchmark.py:363-441의 try/except ImportError)과 이중 방어다.

### 7.2 zero-init 출력층 — graceful degradation

```python
# step_mppi.py:68-74
self.mean_head = nn.Linear(d, N * nu)
self.logstd_head = nn.Linear(d, nu)
# zero-init 출력층
nn.init.zeros_(self.mean_head.weight)
nn.init.zeros_(self.mean_head.bias)
nn.init.zeros_(self.logstd_head.weight)
nn.init.zeros_(self.logstd_head.bias)
```

출력층 weight/bias를 0으로 초기화하면 학습 전 네트워크 출력이 항상
`Δμ=0, log σ=0(→scale=1)`이다. 런타임 결합식이
`μ = U_warm + blend·Δμ`, `σ_eff = σ_base·exp(logσ)`(step_mppi.py:241-244)이므로
**미학습 상태 = 정확히 Vanilla warm start + 기본 노이즈**가 된다. 학습이
진행되면서만 점진적으로 개입이 커진다. 백본이 아니라 **출력층만** 0으로 두는
것이 요령이다 — 백본까지 0이면 기울기가 죽어 학습이 안 된다. 같은 기법이
SG-MPPI의 ScoreNetwork에도 쓰였다 (MEMORY 기록: "Zero-init 출력층: 학습 초기
s_θ≈0 → 순수 가우시안").

추론 시 `log_std`를 `np.clip(..., -1.386, 1.386)`(≈ln4)으로 클램프해 σ가
[0.25σ_base, 4σ_base] 밖으로 못 나가게 한다(step_mppi.py:225) — 학습 초기의
공분산 폭주/붕괴 방어.

### 7.3 Ring buffer 데이터 수집

`StepExperienceBuffer.add()`(step_mppi.py:93-102)는 용량 도달 전엔 append,
이후엔 `self._ptr = (self._ptr + 1) % self.capacity` 포인터 순환 덮어쓰기 —
오래된 경험부터 밀려나는 FIFO ring buffer다. `feature.copy()`가 중요하다: MPPI 루프의 배열은 매 스텝
in-place로 갱신되므로 참조만 저장하면 버퍼 전체가 최신 스텝 값으로 알리어싱된다.

타깃 생성은 자기지도(self-supervised)다 — 라벨이 "MPPI가 실제로 찾은 해와
warm start의 잔차":

```python
# step_mppi.py:277-281
# 자기지도 학습 데이터 수집 (잔차 타깃)
if self._use_net and self.step_params.online_training:
    target_residual = solution_U - U_warm
    self.buffer.add(features, target_residual, float(np.min(costs)))
    self._maybe_train()
```

### 7.4 온라인 학습 트리거

```python
# step_mppi.py:314-323
def _maybe_train(self):
    """주기적 자기지도 학습."""
    if len(self.buffer) < self.step_params.min_train_samples:
        return
    if self._step_count % self.step_params.train_interval != 0:
        return
    F, T = self.buffer.sample_batch(
        self.step_params.train_batch_size, self._rng
    )
    self.trainer.train_step(F, T)
```

두 개의 가드(최소 샘플 수, 학습 주기)를 이른 return으로 처리하는 전형적
구조다. 학습은 제어 루프 안에서 실행되므로 **1회 train_step은 미니배치
1개**로 제한해 제어 주기를 지킨다 — 에폭 단위 학습을 여기 넣으면 실시간성이
깨진다. 손실은 `MSE(Δμ, target) − entropy_weight·mean(logσ)`
(ProposalTrainer.train_step, step_mppi.py:130-142)로, 엔트로피 보너스가 σ 붕괴
(과신)를 막는다.

**흔한 실수** (MEMORY의 SG-MPPI 교훈 재인용): (1) buffer_size <
min_train_samples로 설정하면 영원히 학습이 안 시작된다 — 파라미터 검증에서
잡아야 한다. (2) 학습 비활성 비교 실험에서도 데이터 수집 부작용이 RNG 스트림을
바꿔 "학습 없음 = Vanilla 동일"이 성립하지 않을 수 있다 — 통합 벤치마크가
학습 계열을 전부 `online_training/score_online_training=False`로 등록하는
이유이기도 하다 (all_37_variants_benchmark.py:374, 384, 473).

---

## 8. 파라미터/레지스트리 연동 — 새 변형 추가 체크리스트

새 변형 하나가 "완성"되려면 controller 파일 하나로 끝나지 않는다. 실제 커밋
이력이 표준 파일 세트를 보여준다. 40~43번째 변형을 추가한 커밋 f94d762의
diff-stat에서 변형당 반복되는 패턴 (PGD 기준):

```
mppi_controller/controllers/mppi/pgd_mppi.py       |  224 ++++   (1) 컨트롤러
mppi_controller/controllers/mppi/mppi_params.py    |  278 +++    (2) Params 4종 추가
mppi_controller/controllers/mppi/__init__.py       |   52 +      (3) exports
tests/test_pgd_mppi.py                             |  446 +++    (4) 테스트
examples/comparison/pgd_mppi_benchmark.py          |  817 ++++   (5) 전용 벤치마크
examples/comparison/all_37_variants_benchmark.py   | 1317 ++++   (6) 통합 레지스트리
CLAUDE.md                                          |   26 +-     (7) 데모 명령
docs/MPPI_THEORY.md                                |  512 ++++   (8) 이론 문서
docs/TUTORIALS.md / README.md                      |  ...        (8') 튜토리얼/개요
```

단계별로:

### (1) 컨트롤러 — `mppi_controller/controllers/mppi/{name}_mppi.py`

- §1.1의 5개 확장 포인트 중 최소 침습 조합을 고른다.
- `compute_control(state, reference_trajectory) -> (control, info)` 시그니처와
  info dict 계약(§3 끝 체크리스트) 준수 — CLAUDE.md 인터페이스 규칙.
- 파일 상단 docstring에 핵심 수식 + arXiv 레퍼런스 (전 변형 공통 관례,
  예: pgd_mppi.py:1-24).

### (2) Params — `mppi_params.py`에 데이터클래스 추가

`MPPIParams`를 상속하고 `__post_init__`에서 `super().__post_init__()` 호출 후
자체 검증을 추가한다:

```python
# mppi_params.py:1908-1923 (PGDMPPIParams 발췌)
step_size: float = 1.0
n_grad_steps: int = 1
...
def __post_init__(self):
    super().__post_init__()
    assert self.step_size > 0, "step_size must be positive"
    assert self.n_grad_steps >= 1, "n_grad_steps must be >= 1"
    ...
```

base `__post_init__`(mppi_params.py:56-92)가 sigma/Q/R의 ndarray 변환과
`Qf=None → Q.copy()` 기본값을 처리해 주므로, **서브클래스에서 super 호출을
빼먹으면 sigma가 스칼라인 채로 컨트롤러에 들어가 브로드캐스트 shape 버그**가
난다. 기본값은 가능하면 "Vanilla 동등"으로 (graceful superset, §5.1).

### (3) exports — `__init__.py`

컨트롤러 import + Params import + `__all__` 3곳:

```python
# mppi_controller/controllers/mppi/__init__.py:143, 156, 308-310
from mppi_controller.controllers.mppi.pgd_mppi import PGDMPPIController
...
    PGDMPPIParams,
...
    # PGD-MPPI (Preconditioned Gradient Descent, 40th)
    "PGDMPPIController",
    "PGDMPPIParams",
```

### (4) 테스트 — `tests/test_{name}_mppi.py`

기존 파일 하나를 템플릿으로 쓰면 된다. `test_pgd_mppi.py`(446줄, 28 tests)의
구성이 표준 골격이다 — 파일 docstring이 곧 테스트 설계서다:

```python
# tests/test_pgd_mppi.py:1-13
"""
PGD-MPPI ... 유닛 테스트 — 40번째 변형

~26개 테스트:
  - Params (5): 기본값, 커스텀, step_size/n_grad_steps/cov_scale 검증
  - Construction (2): 생성, repr
  - compute_control (7): shape, finite, info keys, 궤적/가중치/ESS, pgd_stats
  - GradSteps (4): n_grad_steps>1 비용 감소, step_size 효과, resample on/off
  - Covariance (3): adapt_covariance, 범위 클리핑, normalize_gradient
  - HorizonReset (2): receding horizon 시프트, reset
  - Integration (3): 제어 바운드 클리핑, 원형 추적 RMSE, 기본값 vanilla-동등
"""
```

최소 세트로 정리하면:

| 그룹 | 반드시 검증할 것 | 템플릿 위치 |
|------|-----------------|------------|
| Params | 기본값, 커스텀, `__post_init__` assert 발동 | test_pgd_mppi.py:79-137 |
| Construction | 생성 성공, `repr` 문자열 | :141-163 |
| compute_control | control shape `(nu,)`, `np.isfinite`, info 필수 키, weights 합=1, 1≤ESS≤K | :167-224 |
| 변형 고유 기능 | 핵심 파라미터의 인과 효과 (예: n_grad_steps↑ → best_cost↓) | :227-352 |
| 상태 관리 | shift 동작, `reset()` 후 상태 복원 | :356-382 |
| Integration | 제어 바운드 준수, 원형 추적 RMSE < 임계값, Vanilla 퇴화 | :385-446 |

헬퍼 함수 패턴도 그대로 복사한다 — `_make_*_controller(**kwargs)`
(작은 K=64/N=10으로 빠르게, test_pgd_mppi.py:46-63)와 `_make_ref()`.
flaky한 확률적 테스트(비용 감소 등)는 5-trial majority voting으로 안정화한
전례가 있다 (MEMORY, test_iteration_cost_decrease).

실행:

```bash
python -m pytest tests/test_pgd_mppi.py -v --override-ini="addopts="
```

### (5) 전용 벤치마크 — `examples/comparison/{name}_mppi_benchmark.py`

Vanilla + 관련 변형 2~3종과의 4-Way 비교, `--all-scenarios`/`--live` 지원,
`plt.show()` 금지 + `plots/` 저장 (CLAUDE.md 데모 출력 규칙:
`matplotlib.use("Agg")`, PNG/MP4/GIF).

### (6) 통합 레지스트리 등록 — `all_37_variants_benchmark.py`

`_get_variant_registry()`에 항목 추가. **ctor 분류를 정확히** 지정한다:

```python
# examples/comparison/all_37_variants_benchmark.py:444-450
from mppi_controller.controllers.mppi.pgd_mppi import PGDMPPIController
from mppi_controller.controllers.mppi.mppi_params import PGDMPPIParams
registry.append(dict(
    name="PGD", group="G", idx=0, ctor="standard",
    controller_cls=PGDMPPIController, params_cls=PGDMPPIParams,
    extra_params={"n_grad_steps": 3, "step_size": 0.8, "adapt_covariance": True},
))
```

- `cost_function` kwarg를 안 받는 2-인자 생성자면 `ctor="no_cost"`.
- 장애물을 params로 받으면 `ctor="obstacle_in_params"` + `obstacle_field="..."`.
- torch 의존이면 F-그룹의 `try: import torch` 블록 안에(363-441행).
- `extra_params`에는 벤치마크에서 변형의 특성이 드러나는 값을 준다 —
  기본값(=Vanilla 동등)으로 넣으면 비교 의미가 없다 (PGD가 step_size=0.8,
  n_grad_steps=3으로 등록된 이유).

### (7) CLAUDE.md 데모 명령 추가

```bash
# PGD-MPPI (40번째, Preconditioned Gradient Descent) 벤치마크
PYTHONPATH=. python examples/comparison/pgd_mppi_benchmark.py --all-scenarios
PYTHONPATH=. python examples/comparison/pgd_mppi_benchmark.py --live --scenario obstacles
```

### (8) 문서

- `docs/MPPI_THEORY.md`에 이론 절 추가 (수식·유도·기존 변형과의 비교 —
  §42 "변형 선택 가이드" 갱신 포함).
- `docs/TUTORIALS.md`에 사용법, `README.md` 변형 수 갱신.

**체크리스트 요약** (커밋 전 확인):

- [ ] 컨트롤러: `compute_control` 시그니처 + info dict 계약
- [ ] Params: `super().__post_init__()` 호출 + assert 검증
- [ ] `__init__.py`: import 2곳 + `__all__` 2항목
- [ ] 테스트: Params/Construction/compute_control/고유기능/reset/Integration
- [ ] 전용 벤치마크: `--all-scenarios`, `--live`, plots/ 저장 (no `plt.show()`)
- [ ] `all_37_variants_benchmark.py` 레지스트리 (올바른 ctor 분류)
- [ ] CLAUDE.md 데모 명령
- [ ] MPPI_THEORY.md / TUTORIALS.md / README.md
- [ ] `python -m pytest tests/ -v --override-ini="addopts="` 전체 통과

---

## 맺음말 — 패턴 선택의 실용적 기준

새 아이디어를 구현할 때의 결정 순서:

1. **가중치만 바꾸면 되는가?** → 패턴 A. 150줄짜리 파일 하나로 끝난다.
2. **탐색 분포의 형상(주파수·상관·저차원성)이 본질인가?** → 패턴 C.
   `NoiseSampler` 구현 + 얇은 컨트롤러 (LP-MPPI가 모범).
3. **반복·혼합·반복 간 상태가 필요한가?** → 패턴 B. `compute_control`을
   소유하되 rollout/cost/ESS는 base 인프라 재사용, shift/reset/info 계약을
   직접 책임진다.
4. **스텝 자체를 다른 최적화로 해석하는가?** → 패턴 D. 기본값 = Vanilla가
   되도록 설계하고 그 사실을 테스트로 고정한다.
5. **MPPI 바깥에 피드백/구조를 얹는가?** → 패턴 E. 명목/실제 상태 분리와
   각도 wrapping, 차원 가정을 조심한다.
6. **학습이 끼는가?** → 패턴 F. torch 폴백 + zero-init + ring buffer +
   `_maybe_train` 4종 세트를 그대로 가져간다.

어느 패턴이든 마지막은 같다: §8의 체크리스트를 완주해야 43+1번째 변형이 된다.
