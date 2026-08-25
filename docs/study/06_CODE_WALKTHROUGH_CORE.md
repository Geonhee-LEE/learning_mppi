# 06. 코드 워크스루 — 코어 파이프라인

> **이 문서의 위치**: 이론 편 [01](01_MPC_FUNDAMENTALS.md)–[05](05_GENERATIVE_MODELS_FOR_CONTROL.md)를
> 읽은 독자가 **실제 코드를 함수 단위로 이해하고 직접 수정**할 수 있게 하는 코드 해설서다.
> [02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md)의 업데이트 법칙
> `U ← Σ softmax(-S/λ)·V`가 numpy 코드 어느 줄에 어떻게 앉아 있는지를 추적한다.
> 변형별 상세 이론은 [../MPPI_THEORY.md](../MPPI_THEORY.md),
> 안전 계열은 [../SAFETY_THEORY.md](../SAFETY_THEORY.md)를 참조.

모든 코드 발췌는 실제 소스에서 그대로 가져왔으며 `파일:라인` 형식으로 위치를 표기한다.
(라인 번호는 현재 브랜치 기준 — 코드가 수정되면 어긋날 수 있으니 함수명으로도 찾을 것.)

---

## 목차

1. [아키텍처 조감도 — 한 제어 사이클의 콜 그래프](#1-아키텍처-조감도)
2. [RobotModel ABC — 모든 모델의 계약](#2-robotmodel-abc)
3. [BatchDynamicsWrapper — K개 샘플의 병렬 rollout](#3-batchdynamicswrapper)
4. [샘플러 — 노이즈가 곧 탐색이다](#4-샘플러)
5. [비용 함수 — (K,) 배치 규약](#5-비용-함수)
6. [MPPIController.compute_control — 심장부 라인별 해설](#6-mppicontrollercompute_control)
7. [Simulator / Harness / Metrics — 폐루프 검증 인프라](#7-simulator--harness--metrics)
8. [파라미터 시스템 — dataclass 상속 트리](#8-파라미터-시스템)
9. [실습: 내 변형을 하나 만들어보자 (Top-K MPPI)](#9-실습-내-변형을-하나-만들어보자)

---

## 1. 아키텍처 조감도

### 1.1 한 제어 사이클의 콜 그래프

10Hz 제어 주기의 한 틱(tick)에서 일어나는 일 전부:

```
Simulator.step(reference)                          simulation/simulator.py:72
 │
 ├─► controller.compute_control(state, ref)        controllers/mppi/base_mppi.py:111
 │    │
 │    ├─ 1. noise_sampler.sample(U, K, u_min, u_max)       sampling.py:53
 │    │       ε ~ N(0, σ²)                       → (K, N, nu)
 │    │
 │    ├─ 2. sampled_controls = U + ε  (+clip)              base_mppi.py:141
 │    │                                          → (K, N, nu)
 │    │
 │    ├─ 3. dynamics_wrapper.rollout(state, controls)      dynamics_wrapper.py:35
 │    │       └─ 매 t: model.step(states_t, controls_t, dt)   base_model.py:60
 │    │            └─ RK4 × model.forward_dynamics(...)      base_model.py:44
 │    │                                          → (K, N+1, nx)
 │    │
 │    ├─ 4. cost_function.compute_cost(traj, ctrl, ref)    cost_functions.py:356
 │    │       └─ Σ [StateTracking + Terminal + Effort + ...]
 │    │                                          → (K,)
 │    │
 │    ├─ 5. _compute_weights(costs, λ)                     base_mppi.py:276
 │    │       w = softmax(-(S - S_min)/λ)        → (K,)
 │    │
 │    ├─ 6. U ← U + Σ_k w_k·ε_k  (+clip)                   base_mppi.py:160
 │    ├─ 7. U ← roll(U, -1);  U[-1] = 0   (warm start)     base_mppi.py:168
 │    └─ 8. return U[0], info dict                         base_mppi.py:172
 │
 ├─► model.step(state, control, dt)     ← "실제 세계" 전파   simulator.py:92
 ├─► + process_noise  (외란 주입)                            simulator.py:95
 ├─► model.normalize_state(next_state)  (각도 래핑)          simulator.py:100
 └─► history 기록 (state/control/solve_time/info)           simulator.py:109
```

주의할 대칭성: `model.step`이 **두 번** 등장한다. 3번(rollout 내부)은 컨트롤러의
*상상 속 미래* K개이고, Simulator의 것은 *실제 세계* 1개다. 모델 불일치(model mismatch)
실험은 이 두 자리에 서로 다른 모델을 꽂아 만든다 (§7.3 `real_model`).

### 1.2 데이터 shape 흐름표

이 표는 문서 전체의 지도다. 모든 절의 shape 주석은 코드로 검증했다.

| 데이터 | shape | 생산자 | 소비자 |
|--------|-------|--------|--------|
| `state` | `(nx,)` | Simulator | `compute_control`, `rollout` 초기값 |
| `reference_trajectory` | `(N+1, nx)` | 벤치마크의 `ref_fn(t)` | 비용 함수 |
| `U` (명목 제어) | `(N, nu)` | 컨트롤러 내부 상태 | 샘플링 중심, warm start |
| `noise` (ε) | `(K, N, nu)` | `NoiseSampler.sample` | 제어 섭동, 가중 평균 |
| `sampled_controls` | `(K, N, nu)` | `U + noise` 브로드캐스트 | rollout, 비용 |
| `sample_trajectories` | `(K, N+1, nx)` | `rollout` | 비용, 시각화 |
| `costs` (S) | `(K,)` | `compute_cost` | `_compute_weights` |
| `weights` (w) | `(K,)` | `_compute_weights` | 가중 평균, ESS, 시각화 알파 |
| `control` (반환값) | `(nu,)` | `U[0]` | Simulator → 실제 전파 |

기억법: **샘플 축 K가 항상 첫 번째, 시간 축이 두 번째, 물리 차원이 마지막**.
`(K, N, nu)` 규약을 지키면 numpy 브로드캐스팅이 거의 공짜로 따라온다.
궤적만 `N+1`인 이유는 초기 상태 `x_0`가 포함되기 때문 (제어는 N개, 상태는 N+1개).

### 1.3 파일 지도

```
mppi_controller/
├── models/
│   ├── base_model.py                  # RobotModel ABC (§2)
│   └── kinematic/differential_drive_kinematic.py   # 대표 구현 예제
├── controllers/mppi/
│   ├── base_mppi.py                   # MPPIController (§6) ← 심장
│   ├── mppi_params.py                 # MPPIParams + 43종 변형 Params (§8)
│   ├── dynamics_wrapper.py            # BatchDynamicsWrapper (§3)
│   ├── sampling.py                    # NoiseSampler 계열 (§4)
│   └── cost_functions.py              # CostFunction 계열 (§5)
└── simulation/
    ├── simulator.py                   # Simulator (§7.1)
    ├── metrics.py                     # compute_metrics (§7.2)
    └── harness.py                     # SimulationHarness (§7.3)
```

---

## 2. RobotModel ABC

`mppi_controller/models/base_model.py`

### 2.1 인터페이스 계약

모든 모델(kinematic/dynamic/learned)이 지켜야 하는 계약은 단 4개의 abstract 멤버다:

```python
# base_model.py:26-58
@property
@abstractmethod
def state_dim(self) -> int:
    """상태 벡터 차원 (nx)"""

@property
@abstractmethod
def control_dim(self) -> int:
    """제어 벡터 차원 (nu)"""

@property
@abstractmethod
def model_type(self) -> str:
    """모델 타입: 'kinematic', 'dynamic', 'learned'"""

@abstractmethod
def forward_dynamics(self, state, control) -> np.ndarray:
    """연속 시간 동역학: dx/dt = f(x, u)
    state: (nx,) 또는 (batch, nx) / control: (nu,) 또는 (batch, nu)
    """
```

그 외 `step`(RK4 기본 구현), `get_control_bounds`(기본 `None`),
`normalize_state`(기본 항등), `state_to_dict`/`render_config`(시각화용)는
**기본 구현이 있는 훅**이라 필요할 때만 오버라이드한다.

핵심 설계: 컨트롤러는 `forward_dynamics`를 직접 부르지 않고 **`step`만** 부른다
(§3의 wrapper 경유). 즉 새 모델 작성자는 "연속 시간 미분방정식 한 줄"만 쓰면
적분·배치·rollout이 전부 상속으로 따라온다.

### 2.2 RK4 기본 구현

```python
# base_model.py:76-82
# RK4 적분
k1 = self.forward_dynamics(state, control)
k2 = self.forward_dynamics(state + 0.5 * dt * k1, control)
k3 = self.forward_dynamics(state + 0.5 * dt * k2, control)
k4 = self.forward_dynamics(state + dt * k3, control)

return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
```

주목할 점 두 가지:

1. **shape 불가지론(shape-agnostic)**: 이 6줄에는 인덱싱이 하나도 없다.
   `state`가 `(nx,)`든 `(K, nx)`든 `forward_dynamics`가 같은 shape를 돌려주기만 하면
   덧셈·스칼라곱은 그대로 성립한다. 배치 지원의 책임이 전부 `forward_dynamics`
   한 곳으로 위임되는 구조다.
2. **ZOH(zero-order hold) 가정**: k1~k4 모두 같은 `control`을 쓴다.
   제어는 스텝 내에서 상수라는 가정 — MPC 계열의 표준 이산화다.

**왜 RK4인가 (트레이드오프)**: Euler(`x + dt·f`) 대비 4배의 `forward_dynamics`
호출 비용이 들지만, `dt=0.05s`에서 비선형 회전 운동의 적분 오차가 크게 줄어
horizon 끝(N=30 → 1.5초 뒤)의 예측 드리프트를 억제한다. 학습 모델처럼
`forward_dynamics`가 비싼 경우, 서브클래스가 `step` 자체를 오버라이드해
1-스텝 예측(discrete model)으로 바꿀 수 있게 `step`을 abstract로 두지 않았다.

### 2.3 벡터화 규약 — 브로드캐스팅이 성립하는 이유

DiffDrive 구현으로 규약을 확인하자:

```python
# models/kinematic/differential_drive_kinematic.py:78-89
# 벡터화 지원: state[..., i]는 마지막 차원 인덱싱
theta = state[..., 2]
v = control[..., 0]
omega = control[..., 1]

# 기구학 방정식
x_dot = v * np.cos(theta)
y_dot = v * np.sin(theta)
theta_dot = omega

# 스택하여 반환 (마지막 차원으로)
return np.stack([x_dot, y_dot, theta_dot], axis=-1)
```

규약은 두 줄로 요약된다:

- **읽을 때 `[..., i]`**: `state[..., 2]`는 입력이 `(3,)`이면 스칼라,
  `(K, 3)`이면 `(K,)`를 준다. `state[2]`라고 쓰면 배치 입력에서
  "2번째 로봇의 상태 전체"를 뽑아버리는 참사가 난다.
- **쓸 때 `np.stack(..., axis=-1)`**: 성분별 `(K,)` 배열들을 마지막 축으로
  쌓아 `(K, 3)`을 복원한다. `np.array([x_dot, y_dot, theta_dot])`로 쓰면
  `(3, K)`가 되어 축이 뒤집힌다 — 배치에서만 터지는 대표적 버그.

이 두 규약만 지키면 같은 함수가 `(nx,)` 단건과 `(K, nx)` 배치를 **동일 코드로**
처리한다. RK4(§2.2)가 shape 불가지론이므로 `step`도 자동으로 배치를 지원하고,
그 결과 §3의 rollout이 샘플 축 Python 루프 없이 돌아간다.

### 2.4 새 모델 작성 체크리스트

1. `state_dim` / `control_dim` / `model_type` 프로퍼티 구현.
2. `forward_dynamics`를 **반드시 `[..., i]` 읽기 + `stack(axis=-1)` 쓰기**로 구현.
3. 배치 스모크 테스트: `model.forward_dynamics(np.zeros((7, nx)), np.zeros((7, nu))).shape == (7, nx)` 확인.
4. 제어 한계가 있으면 `get_control_bounds` 오버라이드 —
   `MPPIParams.u_min/u_max`가 `None`일 때 컨트롤러가 이 값을 폴백으로 쓴다
   (base_mppi.py:76-78).
5. 각도 상태가 있으면 `normalize_state`를 `arctan2(sin, cos)` 패턴으로 오버라이드
   (differential_drive_kinematic.py:121-126 참조). Simulator가 매 스텝 호출한다.
6. (선택) `state_to_dict` / `render_config`로 시각화 지원.

**흔한 실수**: `normalize_state`에서 `state[..., 2] = ...`를 **입력 배열에 직접**
수행하는 것. DiffDrive 구현이 `normalized = state.copy()`를 먼저 하는 이유다 —
호출자의 배열을 조용히 오염시키면 추적이 매우 어려운 버그가 된다.

---

## 3. BatchDynamicsWrapper

`mppi_controller/controllers/mppi/dynamics_wrapper.py`

### 3.1 rollout — 샘플 축은 벡터, 시간 축은 루프

```python
# dynamics_wrapper.py:50-66
K, N, _ = controls.shape

# 궤적 저장 배열 초기화
trajectories = np.zeros((K, N + 1, self.nx))

# 초기 상태 설정 (브로드캐스트)
trajectories[:, 0, :] = initial_state

# 시간 스텝별 전파
for t in range(N):
    state_t = trajectories[:, t, :]  # (K, nx)
    control_t = controls[:, t, :]  # (K, nu)

    # 배치로 한 스텝 전파 (RK4)
    trajectories[:, t + 1, :] = self.model.step(state_t, control_t, self.dt)

return trajectories
```

읽는 법:

- `trajectories[:, 0, :] = initial_state`는 `(nx,)`를 `(K, nx)` 슬라이스에
  브로드캐스트한다 — K개 샘플 전부가 같은 현재 상태에서 출발한다.
- 루프 본체 한 번의 `model.step` 호출이 **K=1024개 상태를 동시에** 한 스텝
  전진시킨다 (§2.3의 벡터화 규약 덕분에 가능).

**왜 시간 축 루프는 불가피한가**: `x_{t+1} = f(x_t, u_t)`라는 **순차 의존성**
때문이다. t+1 시점 상태는 t 시점 결과가 나와야 계산 가능하므로 시간 축은
원리적으로 병렬화할 수 없다. 반면 샘플 축 K는 샘플끼리 완전히 독립이라
전부 벡터화된다. 결과적으로 rollout의 Python 오버헤드는 O(N)이고 (K와 무관),
K를 512→4096으로 늘려도 루프 횟수는 그대로다 — MPPI가 "샘플을 늘려서 이기는"
전략을 쓸 수 있는 계산적 근거다.

> 시간 축까지 없애고 싶다면 동역학 자체를 바꿔야 한다: Koopman MPPI(38th)는
> EDMD 선형화로 `x_{t} = A^t x_0`꼴 닫힌 전파를 쓰고, pi-MPPI(29th)류는
> 적분기 체인이라 행렬 한 방으로 전파한다. 일반 비선형 모델에서는 불가능.

### 3.2 dt 처리

`dt`는 생성 시 한 번 주입된다 (`BatchDynamicsWrapper(model, params.dt)`,
base_mppi.py:54). 즉 **컨트롤러의 예측 dt = params.dt**이며, Simulator의 dt와
반드시 같아야 폐루프가 정합한다 — 벤치마크들이 `Simulator(model, ctrl, dt=params.dt)`
로 맞춰 쓰는 이유다. 예측 dt ≠ 실행 dt이면 horizon 길이(N·dt)가 실제 시간과
어긋나 레퍼런스 인덱싱이 통째로 밀린다.

**흔한 실수**: `rollout` 결과 shape를 `(K, N, nx)`로 착각하는 것.
초기 상태가 포함되어 `(K, N+1, nx)`다. 비용 함수(§5)가 `[:, :-1, :]`(running)과
`[:, -1, :]`(terminal)로 나눠 쓰는 것이 이 규약의 소비 지점이다.

---

## 4. 샘플러

`mppi_controller/controllers/mppi/sampling.py`

### 4.1 NoiseSampler 프로토콜

```python
# sampling.py:12-35
class NoiseSampler(ABC):
    """노이즈 샘플러 추상 베이스 클래스"""

    @abstractmethod
    def sample(self, U, K, control_min=None, control_max=None) -> np.ndarray:
        """
        U: (N, nu) 명목 제어 시퀀스 / K: 샘플 개수
        Returns: noise: (K, N, nu) 노이즈 샘플
        """
```

계약의 미묘한 점: 반환은 **제어가 아니라 노이즈 ε**다. 컨트롤러가
`U + ε`로 제어를 만들고, 업데이트도 `U + Σ w_k ε_k`로 노이즈 공간에서 한다.
따라서 샘플러가 제약 클리핑을 하고 싶으면 "클리핑된 제어 − U"를 다시 노이즈로
환산해 돌려줘야 한다 (아래 4.2).

### 4.2 GaussianSampler — 기본 샘플러

```python
# sampling.py:49-88
def __init__(self, sigma: np.ndarray, seed: Optional[int] = None):
    self.sigma = sigma
    self.rng = np.random.default_rng(seed)

def sample(self, U, K, control_min=None, control_max=None):
    N, nu = U.shape

    # 가우시안 노이즈 생성 (K, N, nu)
    noise = self.rng.normal(0.0, self.sigma, (K, N, nu))

    # 제어 제약이 있으면 클리핑
    if control_min is not None and control_max is not None:
        # 샘플 제어 = 명목 + 노이즈
        sampled_controls = U + noise  # 브로드캐스트 (K, N, nu)
        # 클리핑
        sampled_controls = np.clip(sampled_controls, control_min, control_max)
        # 노이즈 = 클리핑된 제어 - 명목 제어
        noise = sampled_controls - U

    return noise
```

**sigma 브로드캐스팅**: `self.rng.normal(0.0, self.sigma, (K, N, nu))`에서
`sigma`가 `(nu,)`이면 numpy가 출력 shape `(K, N, nu)`의 마지막 축에 맞춰
브로드캐스트한다 — 제어 채널별로 다른 표준편차(예: v는 0.5, ω는 0.3)가
한 줄로 표현된다. `sigma`를 `(N, nu)`로 주면 시간별로도 달라진다
(UncertaintyAwareSampler가 sampling.py:337-343에서 정확히 이 확장을 쓴다:
`noise = rng.normal(0,1,(K,N,nu)) * sigma_profile[None,:,:]`).

**seed=None 함정 (벤치마크 재현성 이슈)**: `MPPIController.__init__`는
샘플러를 안 주면 `GaussianSampler(params.sigma)`를 **시드 없이** 만든다
(base_mppi.py:70-71). `np.random.default_rng(None)`은 OS 엔트로피로 시드되므로,
벤치마크가 `np.random.seed(42)`를 아무리 불러도 (전역 레거시 RNG만 고정될 뿐)
**샘플러의 Generator에는 아무 영향이 없다** — 실행할 때마다 결과가 달라진다.
이 저장소도 같은 함정을 겪고 명시 시드로 해결했다:

```python
# examples/comparison/cbfkit_inspired_benchmark.py:260-261
# 기본 GaussianSampler(seed=None) 는 OS 엔트로피 시드 → 재현성 위해 명시 시드
sampler = GaussianSampler(pd["sigma"], seed=seed)
```

교훈: 재현 가능한 실험은 **샘플러를 직접 만들어 `noise_sampler=`로 주입**해야
한다. 반대로 실기체 운영에서는 무시드(매번 다른 탐색)가 오히려 자연스럽다 —
기본값이 `None`인 것 자체는 합리적 선택이고, 함정은 "전역 시드가 통할 것"이라는
착각 쪽에 있다.

### 4.3 ColoredNoiseSampler — OU 프로세스

백색 가우시안은 인접 타임스텝 노이즈가 독립이라 제어가 지글거린다(chattering).
Ornstein-Uhlenbeck 프로세스는 시간 상관을 주입해 부드러운 탐색을 만든다:

```python
# sampling.py:143-154
# 각 샘플에 대해 OU 프로세스 시뮬레이션
for k in range(K):
    epsilon = np.zeros(nu)  # 초기 노이즈
    for t in range(N):
        # OU 프로세스 업데이트 (Euler-Maruyama)
        dW = self.rng.normal(0.0, np.sqrt(self.dt), nu)
        epsilon = (
            epsilon
            - self.theta * epsilon * self.dt
            + self.sigma * dW
        )
        noise[k, t, :] = epsilon
```

`dε = -θ·ε·dt + σ·dW`의 Euler-Maruyama 이산화 그대로다. `θ`(복원율)가 클수록
백색 노이즈에 가까워지고, 작을수록 저주파(느리게 변하는) 섭동이 된다.
`dW ~ N(0, dt)`라서 표준편차에 `sqrt(self.dt)`가 들어가는 것에 주의 —
브라운 운동 증분의 분산이 dt에 비례하기 때문이다.

**트레이드오프**: 이 구현은 K×N **이중 Python 루프**라 GaussianSampler보다
수십 배 느리다 (OU는 t 방향 재귀라 naive하게는 벡터화가 안 됨). 같은
"저주파 노이즈" 목표를 scipy의 C 구현 Butterworth 필터로 달성한 것이
`LowPassSampler`다 — `sosfilt(self._sos, noise, axis=1)` 한 줄로 (K, N, nu)를
일괄 필터링한다 (sampling.py:460-463, LP-MPPI 23rd의 핵심). 스무딩 샘플러가
필요하면 실전에서는 LowPassSampler를 먼저 고려하라.

### 4.4 커스텀 샘플러 작성법

`NoiseSampler`를 상속해 `sample`만 구현하면 `MPPIController(..., noise_sampler=내것)`
으로 즉시 주입된다. 체크리스트:

1. 반환 shape는 반드시 `(K, N, nu)` — U와 같은 dtype(float).
2. 평균이 0 근처여야 한다. ε의 평균이 크게 치우치면 `U + Σ w_k ε_k` 업데이트가
   매 스텝 그 방향으로 드리프트한다 (의도된 bias가 아니라면 버그).
3. 제약 인자를 받으면 **"클리핑된 제어 − U" 환산 패턴**(4.2)을 그대로 따를 것.
   클리핑을 생략해도 컨트롤러가 한 번 더 클리핑하므로(§6.2) 안전하긴 하다.
4. 재현성이 필요하면 `seed` 인자와 `np.random.default_rng(seed)` 멤버를 둘 것
   (저장소의 5개 샘플러 전부 이 패턴).
5. 상태 의존 샘플링이 필요하면 별도 setter를 두라 — `sample` 시그니처는
   프로토콜이므로 바꾸지 말 것 (UncertaintyAwareSampler의
   `update_uncertainty_profile`이 선례, sampling.py:292).

---

## 5. 비용 함수

`mppi_controller/controllers/mppi/cost_functions.py`

### 5.1 CostFunction 인터페이스

```python
# cost_functions.py:15-32
@abstractmethod
def compute_cost(
    self,
    trajectories: np.ndarray,      # (K, N+1, nx) 샘플 궤적
    controls: np.ndarray,          # (K, N, nu) 샘플 제어
    reference_trajectory: np.ndarray,  # (N+1, nx) 레퍼런스 궤적
) -> np.ndarray:                   # (K,) 각 샘플의 총 비용
```

한 번의 호출로 **K개 샘플 전부**의 비용을 계산한다. 샘플별 Python 루프를
비용 함수 안에 쓰는 순간 MPPI의 실시간성(K=1024, <100ms)이 무너지므로,
모든 내장 비용은 순수 numpy 브로드캐스팅으로 작성돼 있다.

### 5.2 StateTrackingCost — einsum/브로드캐스팅 해설

```python
# cost_functions.py:67-82
K, N_plus_1, nx = trajectories.shape

# 오차 계산 (K, N, nx) - 터미널 제외
errors = trajectories[:, :-1, :] - reference_trajectory[:-1, :]

if self.is_diagonal:
    # 대각 가중치: (K, N, nx) * (nx,) → (K, N, nx) → (K, N) → (K,)
    costs = np.sum(errors**2 * self.Q, axis=(1, 2))
else:
    # 풀 매트릭스 가중치: errors @ Q @ errors^T
    # (K, N, nx) @ (nx, nx) → (K, N, nx)
    weighted_errors = np.einsum("ktn,nm->ktm", errors, self.Q)
    # (K, N, nx) * (K, N, nx) → (K, N) → (K,)
    costs = np.sum(weighted_errors * errors, axis=(1, 2))
```

- `trajectories[:, :-1, :] - reference_trajectory[:-1, :]`:
  `(K, N, nx) - (N, nx)` — 레퍼런스가 샘플 축으로 브로드캐스트된다.
  터미널(`-1`)을 빼는 이유는 그 몫이 `TerminalCost`(Qf)의 것이기 때문 —
  둘 다 켜져 있으므로 여기서도 포함하면 마지막 상태가 이중 과세된다.
- 대각 경로: `e^T diag(Q) e = Σ_i Q_i e_i²`이므로 성분별 곱 후 시간·상태 축을
  한꺼번에 `axis=(1, 2)`로 접는다. 행렬곱이 아예 없다.
- 풀 매트릭스 경로: `einsum("ktn,nm->ktm", errors, Q)`는 "k, t를 고정한 채
  마지막 축 n을 Q의 행 인덱스와 축약"이므로 각 (k, t)마다 `e^T Q`를 계산한 것.
  이어 `weighted * errors` 성분곱 후 합산하면 `e^T Q e`가 된다 —
  `(K·N, nx)` reshape 없이 2차형식 배치 계산을 끝내는 관용구다.

`is_diagonal = Q.ndim == 1` 분기(생성자, cost_functions.py:46-48)는
기본 파라미터가 `Q = np.array([10., 10., 1.])`처럼 대각 벡터인 점을 활용한
성능 최적화다. 같은 패턴이 Terminal/ControlEffort/ControlRate에 반복된다.

### 5.3 ObstacleCost — `[:, :, :2]` 슬라이싱

```python
# cost_functions.py:263-282
for obs_x, obs_y, obs_radius in self.obstacles:
    # 궤적 위치 (K, N+1, 2) - x, y만 사용
    positions = trajectories[:, :, :2]  # (K, N+1, 2)

    # 장애물까지 거리 (K, N+1)
    distances = np.sqrt(
        (positions[..., 0] - obs_x) ** 2 + (positions[..., 1] - obs_y) ** 2
    )

    # 침투 깊이 (음수면 안전, 양수면 위험)
    penetrations = (obs_radius + self.safety_margin) - distances

    # 침투한 경우만 비용 부과 (exponential)
    obstacle_costs = np.where(
        penetrations > 0, np.exp(penetrations * 5.0), 0.0
    )

    # 시간 스텝에 대해 합산 (K, N+1) → (K,)
    costs += self.cost_weight * np.sum(obstacle_costs, axis=1)
```

- `trajectories[:, :, :2]`: 상태의 앞 두 성분이 (x, y)라는 **암묵적 규약**에
  의존한다. 이 규약은 metrics(§7.2)의 `states[:, :2]`, 렌더링, 대부분의 안전
  비용에 공유된다 — 상태 배치가 다른 모델(예: 매니퓰레이터 관절각)에는
  ObstacleCost를 그대로 못 쓰고 EE FK 계열 비용(EndEffectorTrackingCost)을 쓴다.
- 루프는 **장애물 개수**에 대해서만 돈다 (보통 수 개). 샘플·시간 축은 벡터화.
- `np.where(p > 0, exp(5p), 0)`: 안전 구역에서 비용 0, 침투 시 지수 증가 —
  soft constraint다. 지수 안 5.0은 하드코딩된 sharpness. 침투 0.2m에서
  e¹ ≈ 2.7, 0.9m에서 e⁴·⁵ ≈ 90에 `cost_weight=100`이 곱해지므로 사실상
  뚫는 샘플의 softmax 가중치를 0으로 만든다. hard 보장이 필요하면
  CBF/Shield 계열([../SAFETY_THEORY.md](../SAFETY_THEORY.md),
  [03_CBF_FUNDAMENTALS.md](03_CBF_FUNDAMENTALS.md))로 넘어간다.

### 5.4 CompositeMPPICost — 합성 패턴

```python
# cost_functions.py:373-381
K = trajectories.shape[0]
total_costs = np.zeros(K)

for cost_fn in self.cost_functions:
    total_costs += cost_fn.compute_cost(
        trajectories, controls, reference_trajectory
    )

return total_costs
```

단순 합산 Composite 패턴. 상대 중요도는 각 비용의 내부 가중치(Q, R,
cost_weight)로 조절한다. 컨트롤러 기본값은 다음 3종 합성이다:

```python
# base_mppi.py:59-65
self.cost_function = CompositeMPPICost([
    StateTrackingCost(params.Q),
    TerminalCost(params.Qf),
    ControlEffortCost(params.R),
])
```

장애물 회피 실험은 여기에 `ObstacleCost`를 얹은 Composite를 `cost_function=`
인자로 주입하는 식이다. **컨트롤러 코드를 건드리지 않고 비용만 갈아끼우는**
것이 이 저장소 벤치마크 전반의 조립 방식이다.

### 5.5 커스텀 비용 작성 체크리스트 (배치 규약 위반 시 증상)

1. 반환은 반드시 `(K,)` float 배열. **위반 시 증상**:
   - 스칼라(`np.sum` 전축 합산 실수)를 돌려주면 — softmax 입력이 상수가 되어
     `weights`가 브로드캐스트로 (1,) 혹은 스칼라가 되고, `weights[:, None, None] * noise`
     에서 K축 축약이 사라져 U 업데이트가 "모든 노이즈의 평균 방향"으로
     뭉개진다. 에러 없이 성능만 조용히 죽는 최악 유형.
   - `(K, N)`을 돌려주면 — `_compute_weights`의 `np.min`은 통과하지만
     `weights[:, None, None]`이 `(K, N, 1, 1)`이 되어 `* noise (K, N, nu)`에서
     broadcast 에러로 즉사한다. 차라리 낫다.
2. 샘플 축 Python 루프 금지 — K=1024 기준 루프 하나가 수십 ms를 먹는다.
3. 비용은 **낮을수록 좋음** (softmax(−S/λ)). 보상(reward)을 만들었다면 부호 반전.
4. 음수 비용도 수학적으로는 무방하다 — min-shift(§6.3)가 알아서 처리한다.
   단 `np.inf`는 금지: `inf - inf = nan`이 min-shift에서 발생할 수 있다.
   위반 페널티는 큰 유한값(저장소 관례 `1e4`, JointLimitCost 참조)으로.
5. 스케일 감각: 비용 차이가 λ(기본 1.0) 대비 너무 크면 ESS→1 (탐욕), 너무
   작으면 ESS→K (평균화). 새 항을 넣은 뒤 info dict의 `ess`(§6.5)를 관찰하라.
   이론적 배경은 [02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md)의 ESS 절.

---

## 6. MPPIController.compute_control

`mppi_controller/controllers/mppi/base_mppi.py` — 이 문서의 핵심.
전체 메서드(111-190행)를 단계별로 발췌하며 해설한다.
[02편](02_MPPI_FUNDAMENTALS.md)에서 유도한 업데이트 법칙의 numpy 구현이다.

### 6.0 생성자에서 준비되는 것들

```python
# base_mppi.py:85-89
# 명목 제어 시퀀스 초기화 (N, nu)
self.U = np.zeros((params.N, model.control_dim))

# 메트릭 저장
self.last_info = {}
```

`self.U`가 컨트롤러의 **유일한 본질적 상태**다. 매 호출에서 갱신·시프트되어
warm start를 이룬다. `reset()`(312행)은 이걸 0으로 되돌릴 뿐이다.
`set_control_sequence`(316행)로 외부에서 초기해를 심을 수도 있다
(T-MPPI(33rd)가 Transformer 예측을 여기에 꽂는 식의 진입점).

### 6.1 진입: GPU 분기와 샘플링

```python
# base_mppi.py:131-141
if self._use_gpu:
    return self._compute_control_gpu(state, reference_trajectory)

K = self.params.K
N = self.params.N

# 1. 노이즈 샘플링 (K, N, nu)
noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

# 2. 샘플 제어 시퀀스 (K, N, nu)
sampled_controls = self.U + noise  # 브로드캐스트
```

- GPU 분기는 **메서드 최상단의 조기 반환** 하나로 처리된다.
  `params.device == "cuda"`이고 `torch.cuda.is_available()`일 때만
  `_use_gpu=True`가 되며(92행), 이때 생성자가 `gpu/` 서브패키지에서
  `TorchDynamicsWrapper / TorchCompositeCost / TorchGaussianSampler`를
  **지연 import**한다(94-109행) — CPU 사용자에게 torch 의존성을 강요하지
  않기 위해서다. `_compute_control_gpu`(192-265행)는 아래 CPU 경로의
  1:1 torch 미러이며 (같은 9단계, `torch.roll`/`torch.clamp`로 치환),
  info만 마지막에 `.cpu().numpy()`로 내린다. 알고리즘을 고칠 때 **두 경로를
  같이 고쳐야 한다**는 유지보수 비용이 이 구조의 대가다.
- `self.U + noise`: `(N, nu) + (K, N, nu)` 브로드캐스트. K개 샘플 제어
  `V_k = U + ε_k`가 한 줄에 만들어진다.

### 6.2 클리핑 — 제약과 분포 바이어스

```python
# base_mppi.py:143-145
# 제어 제약 클리핑 (safety)
if self.u_min is not None and self.u_max is not None:
    sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)
```

`u_min/u_max`는 `(nu,)`라 `(K, N, nu)`의 마지막 축에 브로드캐스트된다.
샘플러가 이미 클리핑 환산(§4.2)을 했더라도 커스텀 샘플러의 실수를 막는
2차 방어선으로 한 번 더 자른다.

**클리핑이 분포에 주는 바이어스**: MPPI의 importance-sampling 유도는 제안
분포가 `N(U, Σ)`라고 가정한다. 클리핑은 이 가우시안을 **경계에 질량이 쌓인
절단-집적(censored) 분포**로 바꾼다. U가 경계 근처에 있으면 (예: 최고 속도로
주행 중) 노이즈의 실효 평균이 경계 안쪽으로 쏠리고, 이론적 가중치와 실제
샘플 분포가 어긋난다. 실무적 결과: 경계 근처에서 업데이트가 보수적으로
수축한다 — 대체로 무해하지만, 제약 활성 상태가 오래 유지되는 태스크에서
추적 지연으로 나타날 수 있다. 이를 원리적으로 해결하는 변형이
RectifiedGaussianSampler(재샘플링, sampling.py:170)와 pi-MPPI(29th, 투영)다.

### 6.3 rollout · 비용 · 가중치

```python
# base_mppi.py:147-156
# 3. 샘플 궤적 rollout (K, N+1, nx)
sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

# 4. 비용 계산 (K,)
costs = self.cost_function.compute_cost(
    sample_trajectories, sampled_controls, reference_trajectory
)

# 5. MPPI 가중치 계산
weights = self._compute_weights(costs, self.params.lambda_)
```

3·4단계는 §3·§5 그대로다. 5단계가 MPPI의 수학적 심장인데, 별도 메서드로
분리된 것이 이 저장소의 가장 중요한 설계 결정이다:

```python
# base_mppi.py:276-294
def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
    """
    MPPI 가중치 계산 (softmax)

    w_k = exp(-cost_k / λ) / Σ exp(-cost_k / λ)
    """
    # 수치 안정성을 위한 log-space 연산
    min_cost = np.min(costs)
    exp_costs = np.exp(-(costs - min_cost) / lambda_)
    weights = exp_costs / np.sum(exp_costs)

    return weights
```

**min-shift가 결과를 바꾸지 않는 이유 (유도)**: 상수 `c = S_min`을 빼면

```
w_k = exp(-(S_k - c)/λ) / Σ_j exp(-(S_j - c)/λ)
    = [exp(-S_k/λ)·exp(c/λ)] / [exp(c/λ)·Σ_j exp(-S_j/λ)]
    = exp(-S_k/λ) / Σ_j exp(-S_j/λ)                    ← e^{c/λ} 소거
```

softmax는 입력의 평행이동에 불변이므로 **어떤 상수**를 빼도 되지만, 하필
min을 빼는 이유가 수치 안정성이다: `S_k - S_min ≥ 0`이므로 지수 인자가
전부 `≤ 0`이 되어 `exp`는 (0, 1] 범위 — **overflow가 원천 차단**된다.
underflow(아주 나쁜 샘플이 0.0이 되는 것)는 일어나지만, 그 샘플의 가중치는
어차피 ≈0이어야 하므로 무해하다.

**log-sum-exp와의 관계**: LSE 트릭 `log Σ exp(x_j) = M + log Σ exp(x_j - M)`
(M = max x)에서 `x_k = -S_k/λ`로 두면 `M = -S_min/λ` — 위 min-shift와 정확히
같은 연산이다. 즉 이 구현은 "LSE 트릭을 exp 공간에서 수행한 것"이고,
Log-MPPI(`log_mppi.py`)는 같은 계산을 log 공간에 끝까지 머물며 수행해
`log w_k = -S_k/λ - LSE(-S/λ)`를 만든 뒤 마지막에 exp한다. 두 결과는
수학적으로 동일하며, 차이는 극단적 비용 스케일에서의 수치 정밀도뿐이다.

**서브클래스 오버라이드 훅**: 프로젝트 인터페이스 규칙(CLAUDE.md)대로,
가중치 계열 변형은 `_compute_weights`만 갈아끼운다 — Log-MPPI(log-space),
Tsallis-MPPI(q-지수), Risk-Aware(CVaR 절단), ASR-MPPI(spectral risk 왜곡)가
전부 이 한 메서드의 오버라이드다. §9에서 직접 만들어본다.

### 6.4 가중 평균 업데이트와 warm start 시프트

```python
# base_mppi.py:158-172
# 6. 가중 평균으로 제어 업데이트
# U_new = Σ w_k (U + ε_k) = U + Σ w_k ε_k
weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)  # (N, nu)
self.U = self.U + weighted_noise

# 제어 제약 클리핑
if self.u_min is not None and self.u_max is not None:
    self.U = np.clip(self.U, self.u_min, self.u_max)

# 7. 다음 스텝을 위한 시프트 (receding horizon)
self.U = np.roll(self.U, -1, axis=0)
self.U[-1, :] = 0.0  # 마지막 제어는 0으로

# 8. 최적 제어 반환 (첫 번째 제어)
optimal_control = self.U[0, :]
```

- `weights[:, None, None]`: `(K,)` → `(K, 1, 1)`로 늘려 `(K, N, nu)` 노이즈와
  브로드캐스트. `axis=0` 합산으로 K축이 접혀 `(N, nu)` — 기대 노이즈다.
  주석의 항등식 `Σ w_k (U + ε_k) = U + Σ w_k ε_k`은 `Σ w_k = 1`에서 나온다.
  덕분에 제어 전체가 아니라 **노이즈만 가중 평균**하면 되고, 이것이 02편의
  `U ← Σ softmax(-S/λ)·V`와 동치다.
- **warm start 시프트**: `np.roll(U, -1)`은 `[u_0, u_1, ..., u_{N-1}]`을
  `[u_1, ..., u_{N-1}, u_0]`으로 회전시키고, 랩어라운드로 맨 뒤에 온 `u_0`
  자리를 `U[-1] = 0`으로 덮는다. 다음 호출 때 시간이 한 스텝 흘러 있으므로,
  이번에 계산한 계획의 "한 칸 뒤부터"를 초기해로 재사용하는 것 —
  이것이 warm start이고, MPPI가 K개 샘플만으로 매 스텝 수렴하는 실질적 이유다
  (cold start라면 매번 0에서 다시 탐색해야 한다).
- **왜 마지막 스텝을 0으로 채우나**: 시프트로 비는 마지막 칸을 채우는 정책은
  구현마다 갈린다. (a) 0으로 채움 — "새로 보이는 미래는 우선 정지 성향"이라는
  보수적 사전(prior). (b) 마지막 스텝 복제(`U[-1] = U[-2]`) — 정상 주행
  (등속 원 추적 등)에서 더 자연스러운 사전. 이 저장소는 (a)를 쓴다.
  마지막 스텝은 어차피 N번의 재계획을 거치며 노이즈+가중평균으로 다듬어진 뒤에야
  실행 위치(U[0])에 도달하므로 실전 차이는 작지만, σ가 작고 λ가 클 때는
  0-채움의 감속 성향이 horizon 끝 추적 오차로 배어날 수 있다 — TerminalCost(Qf)가
  이를 상쇄하는 역할도 겸한다.
- **주의 — 시프트가 반환보다 먼저다**: 코드 순서상 시프트(7) 후에 `U[0]`을
  반환(8)하므로, 실제 적용되는 제어는 최적화된 시퀀스의 `u_0`이 아니라
  **`u_1`**이다 (roll 후의 U[0] = roll 전의 U[1]). 교과서 순서(u_0 반환 → 시프트)
  와 한 스텝 어긋나며, "계산 지연 1스텝을 선보상하는" 해석도 가능하지만
  dt=0.05s에서 사실상 체감 차이는 없다. 다만 변형을 구현하며 base를 참조할 때
  이 순서를 그대로 복사할지, u_0 반환으로 바꿀지는 의식적으로 선택해야 한다
  (여러 변형이 `compute_control`을 통째로 오버라이드하면서 이 순서를 각자
  다르게 처리한다).

### 6.5 info dict — 각 키의 의미와 소비처

```python
# base_mppi.py:174-188
# 9. 정보 저장
ess = self._compute_ess(weights)
best_idx = np.argmin(costs)

info = {
    "sample_trajectories": sample_trajectories,
    "sample_weights": weights,
    "best_trajectory": sample_trajectories[best_idx],
    "best_cost": costs[best_idx],
    "mean_cost": np.mean(costs),
    "temperature": self.params.lambda_,
    "ess": ess,
    "num_samples": K,
}
self.last_info = info
```

| 키 | shape/type | 의미 | 주요 소비처 |
|----|-----------|------|-------------|
| `sample_trajectories` | `(K, N+1, nx)` | 전체 샘플 궤적 | 라이브 애니메이션의 "스파게티" 렌더링 (`simulation/rendering/`) |
| `sample_weights` | `(K,)` | softmax 가중치 | 궤적 라인 alpha/색 농도 (좋은 샘플일수록 진하게) |
| `best_trajectory` | `(N+1, nx)` | 최저 비용 단일 궤적 | 계획 경로 하이라이트 표시 |
| `best_cost` / `mean_cost` | float | 비용 분포 요약 | 수렴/발산 모니터링, 반복형 변형의 개선 판정 |
| `temperature` | float | 현재 λ | 적응 λ 변형(Biased-MPPI 등)의 추이 로깅 |
| `ess` | float | `1/Σw²` ∈ [1, K] | **핵심 건강 지표** — 아래 참고 |
| `num_samples` | int | K | 벤치마크 표기 |

ESS(`_compute_ess`, 296-310행)는 "가중치가 실질적으로 몇 개 샘플에 퍼져
있는가"다. ESS ≈ 1이면 한 샘플 독식(λ 너무 작거나 비용 스케일 과대 —
사실상 탐욕 선택), ESS ≈ K이면 균등(λ 너무 큼 — 그냥 노이즈 평균).
튜닝 시 목표 대역은 대략 K의 2~30%. 적응형 변형들(Biased-MPPI의
`_adapt_lambda`, ASR-MPPI의 `_adapt_parameters`)이 바로 이 값을 피드백
신호로 쓴다. 이론은 [02편](02_MPPI_FUNDAMENTALS.md)의 ESS 절.

`sample_trajectories`는 K=1024, N=30, nx=3 기준 float64로 ≈ 760KB/스텝이다.
장시간 시뮬레이션에서 history에 전부 쌓으면 메모리가 부풀므로 Simulator에
`store_info=False` 옵션이 있다 (§7.1).

**흔한 실수 모음 (§6 종합)**:

- info dict 키를 빠뜨린 커스텀 변형 — 렌더러가 `info["sample_trajectories"]`를
  기대하므로 KeyError로 애니메이션이 죽는다. 변형이 `compute_control`을 통째로
  오버라이드할 때도 표준 키는 전부 채울 것 (프로젝트 인터페이스 규칙).
- `weights * noise`처럼 `[:, None, None]` 없이 곱하기 — `(K,) * (K, N, nu)`는
  차원이 안 맞아 즉시 에러가 나니 그나마 낫지만, `(K, 1, 1)`로 reshape한다며
  `weights.reshape(-1, 1)`(2D)로 만들면 조용히 잘못 브로드캐스트될 수 있다.
- λ를 `params.lambda_`가 아닌 지역 변수로 하드코딩 — 적응 λ 변형과 조합 시
  온도 불일치.

---

## 7. Simulator / Harness / Metrics

`mppi_controller/simulation/`

### 7.1 Simulator.step / run — 폐루프의 한 바퀴

```python
# simulation/simulator.py:86-119 (step, 주석 축약)
# 1. MPPI 제어 계산
t_start = time.time()
control, info = self.controller.compute_control(self.state, reference_trajectory)
solve_time = time.time() - t_start

# 2. 상태 전파 (모델 사용)
next_state = self.model.step(self.state, control, self.dt)

# 3. 외란 주입 (있을 경우)
if self.process_noise_std is not None:
    noise = np.random.normal(0.0, self.process_noise_std, self.model.state_dim)
    next_state += noise

# 4. 상태 정규화 (각도 등)
next_state = self.model.normalize_state(next_state)
...
# 6. 히스토리 기록
self.history["time"].append(self.t)
self.history["state"].append(self.state.copy())
self.history["control"].append(control.copy())
self.history["reference"].append(reference_trajectory[0].copy())
self.history["solve_time"].append(solve_time)
if self.store_info:
    self.history["info"].append(info)

# 7. 상태 업데이트
self.state = next_state
self.t += self.dt
```

- **process_noise 주입 지점**: 컨트롤러가 모르는 외란을 "실제 세계" 전파
  **직후, 정규화 직전**에 더한다. 컨트롤러의 rollout(§3)에는 이 노이즈가
  없으므로 예측-현실 격차가 생기고, Robust/Tube 계열 변형이 강한지 시험하는
  손잡이가 된다. 이 노이즈는 **레거시 전역 RNG**(`np.random.normal`)를 쓰므로
  `np.random.seed(...)`로 고정된다 — §4.2의 샘플러 시드와는 **별개 RNG**라는
  점을 다시 강조한다 (외란은 재현되는데 샘플링은 재현이 안 되는 "반쪽 재현성"이
  전형적 증상).
- history에 기록되는 `state`는 갱신 **전**(제어를 계산한 시점의) 상태고,
  `reference`는 그 시점 레퍼런스의 첫 행 `reference_trajectory[0]`이다 —
  둘이 같은 시각이므로 §7.2의 추적 오차가 정합하게 계산된다.
- `run(reference_trajectory_fn, duration)`(128-159행)은
  `num_steps = int(duration / dt)`번 `ref_fn(self.t)` → `step()`을 반복하는
  단순 루프이며, `realtime=True`면 `dt - solve_time`만큼 sleep해 실시간 재생한다.
- `get_history()`(161-181행)가 list-of-arrays를 `(T, nx)`, `(T, nu)` 등의
  numpy 배열로 변환한다 (`info`는 dict 리스트 그대로).

### 7.2 compute_metrics — 지표 계산

```python
# simulation/metrics.py:41-49
# 1. 위치 오차 (x, y)
position_errors = np.linalg.norm(states[:, :2] - references[:, :2], axis=1)
position_rmse = np.sqrt(np.mean(position_errors**2))
max_position_error = np.max(position_errors)

# 2. 각도 오차 (θ) - 각도 차이는 [-π, π]로 정규화
if nx >= 3:
    heading_errors = angle_difference(states[:, 2], references[:, 2])
    heading_rmse = np.sqrt(np.mean(heading_errors**2))
```

`(T, nx)` history 배열에서 9개 지표를 만든다: `position_rmse` /
`max_position_error` / `heading_rmse` / `max_heading_error` /
`control_rate`·`max_control_rate`(연속 제어 차분 노름 — 부드러움) /
`mean`·`max`·`std_solve_time`(ms). 여기서도 `[:, :2]` = (x, y) 규약(§5.3)과,
각도 차를 `arctan2(sin, cos)`로 래핑하는 관용구(metrics.py:96-99)가 반복된다 —
래핑 없이 `state[2] - ref[2]`를 쓰면 θ가 ±π를 넘는 순간 2π짜리 유령 오차가
RMSE를 폭파시킨다. 성능 기준(CLAUDE.md): 원형 궤적 position_rmse < 0.2m,
K=1024/N=30에서 solve < 100ms.

### 7.3 SimulationHarness — 다중 컨트롤러 비교 패턴

```python
# simulation/harness.py:7-13 (docstring)
harness = SimulationHarness(dt=0.05)
harness.add_controller("Vanilla", ctrl_v, model_v, "blue")
harness.add_controller("Flow", ctrl_f, model_f, "red")
results = harness.run(ref_fn, x0, duration=15.0)
harness.plot(results, save_path="plots/comparison.png")
harness.animate(ref_fn, x0, duration=15.0, save_path="plots/comparison.mp4")
```

내부 구조: `add_controller`가 `ControllerEntry`(dataclass — name/controller/
model/색상/`process_noise_std`/`real_model`)를 등록하고, `run`이 엔트리마다
Simulator를 만들어 **동일 초기 상태·레퍼런스·duration**으로 돌린 뒤
`{name: {"history", "metrics", "env_metrics"}}`를 돌려준다. 두 가지 포인트:

- `run` 서두의 `np.random.seed(self.seed)`(harness.py:113)로 전역 RNG를
  고정한다 → process noise는 컨트롤러 간 공정 비교가 된다. 단 §4.2의 이유로
  **샘플러까지 재현하려면 각 컨트롤러에 시드 있는 샘플러를 직접 주입**해야 한다.
- `sim_model = entry.real_model or entry.model`(119행): 컨트롤러에게는
  `model`(믿는 모델)을, 시뮬레이터에게는 `real_model`(실제 모델)을 줄 수 있다 —
  wheelbase 불일치 같은 model mismatch 실험(PR-MPPI 벤치마크 등)이 이 한 필드로
  구성된다. §1.1에서 말한 "model.step 두 자리에 다른 모델 꽂기"의 구현이다.

시각화는 헤드리스 규약(plt.show() 금지, `plots/*.png|mp4|gif` 저장)을
`rendering/headless.py`·`animation_saver.py`에 위임한다.

---

## 8. 파라미터 시스템

`mppi_controller/controllers/mppi/mppi_params.py` (2000줄 넘는 파일 —
43종 변형의 Params가 전부 여기 모여 있다)

### 8.1 베이스 MPPIParams

```python
# mppi_params.py:31-54 (발췌)
# 기본 파라미터
N: int = 30  # 호라이즌
dt: float = 0.05  # 50ms
K: int = 1024  # 샘플 수

# 온도 및 노이즈
lambda_: float = 1.0  # 온도 파라미터
sigma: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))

# 비용 함수 가중치
Q: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 1.0]))
R: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1]))
Qf: Optional[np.ndarray] = None  # 터미널 비용 (None이면 Q 사용)

# 제어 제약 (None이면 모델 제약 사용)
u_min: Optional[np.ndarray] = None
u_max: Optional[np.ndarray] = None

# 디바이스 설정
device: str = "cpu"  # 'cpu' or 'cuda'
```

세부 장치들:

- `field(default_factory=...)`: ndarray는 가변 객체라 일반 기본값으로 쓰면
  dataclass가 `ValueError`를 내거나(파이썬이 잡아주지 못하는 타입이면)
  **모든 인스턴스가 같은 배열을 공유**하는 버그가 된다. factory가 정답.
- `__post_init__`(56-91행)이 3중 역할을 한다: (1) list/스칼라 입력을
  ndarray로 승격 — `sigma=0.5`나 `Q=[10,10,1]`처럼 편하게 써도 됨,
  (2) `Qf=None → Q.copy()` 파생 기본값, (3) `assert` 융단 검증
  (`N>0`, `sigma>0`, `u_min<u_max` 등) — 잘못된 설정을 rollout 도중의
  미스터리한 NaN이 아니라 **생성 시점의 명확한 AssertionError**로 바꾼다.
- `lambda_`의 트레일링 언더스코어는 파이썬 키워드 `lambda` 회피.

### 8.2 변형별 Params 서브클래스 패턴

```python
# mppi_params.py:144-157
@dataclass
class TsallisMPPIParams(MPPIParams):
    """
    Tsallis-MPPI 전용 추가 파라미터

    Attributes:
        tsallis_q: Tsallis 엔트로피 파라미터 (1.0이면 Vanilla MPPI)
    """

    tsallis_q: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.tsallis_q > 0, "tsallis_q must be positive"
```

모든 변형이 이 4줄 패턴을 따른다: **`@dataclass` 상속 + 추가 필드 +
`super().__post_init__()` 체인 + 자기 필드 assert**. dataclass 상속은
부모 필드를 앞에, 자식 필드를 뒤에 이어 붙이므로 (파이썬 규칙상 자식 필드도
전부 기본값이 있어야 함) `TsallisMPPIParams(K=512, tsallis_q=1.5)`처럼
부모·자식 파라미터를 한 생성자로 섞어 쓸 수 있다. 다단 상속도 흔하다:
`ShieldMPPIParams(CBFMPPIParams)` → `ConformalCBFMPPIParams(ShieldMPPIParams)`
처럼 기능 누적 계보가 타입 계층으로 그대로 드러난다.

**기본값 철학** — 위 docstring의 "1.0이면 Vanilla MPPI"가 핵심이다.
이 저장소의 관례는 *변형 파라미터의 기본값 = 해당 기능이 베이스와 동등해지는
값 또는 논문 권장값*이다 (`cvar_alpha=1.0` → Vanilla와 동일,
`use_baseline=True` → 안전한 기본, `tube_enabled=True` → 이름값을 하는 기본).
덕분에 "Params만 바꿔 끼우고 아무 것도 안 넣으면" 최소한 망가지지는 않는다.
새 변형의 Params를 설계할 때도 이 철학을 따를 것: **기본값으로 생성하면
그럴듯하게 동작해야 하고, 극단값은 사용자가 명시적으로 선택하게 하라.**

**흔한 실수**: 자식 `__post_init__`에서 `super().__post_init__()` 호출을
빠뜨리는 것 — sigma/Q의 ndarray 승격과 Qf 파생이 통째로 건너뛰어져,
한참 뒤 `'list' object has no attribute 'ndim'` 같은 원거리 에러로 나타난다.

---

## 9. 실습: 내 변형을 하나 만들어보자

이제 §6.3의 훅을 직접 써 본다. 목표: **비용 하위 top-k개 샘플에만 softmax
가중치를 주는 "Top-K MPPI"** — CEM(Cross-Entropy Method)의 엘리트 선택과
MPPI softmax의 하이브리드다. `_compute_weights` 하나만 오버라이드하며,
아래 코드는 실제로 실행해 검증했다 (결과는 §9.3).

### 9.1 전체 코드

```python
"""Top-K MPPI: _compute_weights만 오버라이드하는 최소 변형."""
import numpy as np
from dataclasses import dataclass

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.sampling import GaussianSampler
from mppi_controller.simulation.simulator import Simulator
from mppi_controller.simulation.metrics import compute_metrics


@dataclass
class TopKMPPIParams(MPPIParams):          # §8.2 패턴 그대로
    """Top-K MPPI 파라미터: 엘리트 샘플 수만 추가."""
    top_k: int = 64  # 가중치를 받을 엘리트 샘플 수

    def __post_init__(self):
        super().__post_init__()            # 잊으면 안 되는 체인 (§8.2)
        assert 0 < self.top_k <= self.K, "top_k must be in (0, K]"


class TopKMPPIController(MPPIController):
    """상위 top_k개 저비용 샘플만 softmax 가중치를 받는 MPPI 변형."""

    def _compute_weights(self, costs: np.ndarray, lambda_: float) -> np.ndarray:
        k = self.params.top_k
        # 1) 비용 하위 k개 인덱스 (엘리트) — argpartition은 O(K), 정렬 불필요
        elite_idx = np.argpartition(costs, k - 1)[:k]  # (k,)
        # 2) 엘리트에만 min-shift softmax (§6.3과 동일한 수치 안정화)
        elite_costs = costs[elite_idx]
        shifted = elite_costs - np.min(elite_costs)
        elite_w = np.exp(-shifted / lambda_)
        elite_w /= np.sum(elite_w)
        # 3) 전체 (K,)로 산포 — 비엘리트는 0. 반환 shape 계약 유지!
        weights = np.zeros_like(costs)
        weights[elite_idx] = elite_w
        return weights


def make_circle_ref_fn(N, dt, radius=2.0, omega=0.3):
    def ref_fn(t):
        ts = t + np.arange(N + 1) * dt
        x = radius * np.cos(omega * ts)
        y = radius * np.sin(omega * ts)
        theta = omega * ts + np.pi / 2
        return np.stack([x, y, theta], axis=-1)  # (N+1, 3)
    return ref_fn


def run(controller_cls, params, seed=42, duration=15.0):
    model = DifferentialDriveKinematic(v_max=2.0, omega_max=2.0)
    sampler = GaussianSampler(params.sigma, seed=seed)  # 재현성: 명시 시드 (§4.2)
    ctrl = controller_cls(model, params, noise_sampler=sampler)
    sim = Simulator(model, ctrl, dt=params.dt)
    ref_fn = make_circle_ref_fn(params.N, params.dt)
    sim.reset(np.array([2.0, 0.0, np.pi / 2]))
    history = sim.run(ref_fn, duration=duration)
    return compute_metrics(history), ctrl


if __name__ == "__main__":
    common = dict(N=30, dt=0.05, K=512, lambda_=1.0,
                  sigma=np.array([0.3, 0.3]))

    m_vanilla, _ = run(MPPIController, MPPIParams(**common))
    m_topk, ctrl = run(TopKMPPIController, TopKMPPIParams(top_k=64, **common))

    # 가중치 불변식 검증
    w = ctrl.last_info["sample_weights"]
    assert w.shape == (512,)                    # (K,) 계약
    assert np.isclose(w.sum(), 1.0)             # 정규화
    assert np.count_nonzero(w) <= 64            # 엘리트만 비영
    assert np.all(w >= 0)

    print(f"Vanilla RMSE = {m_vanilla['position_rmse']:.4f} m")
    print(f"Top-K   RMSE = {m_topk['position_rmse']:.4f} m")
    assert m_topk["position_rmse"] < 0.2, "성능 기준 미달"
```

### 9.2 설계 해설

- **오버라이드 지점이 왜 `_compute_weights`뿐인가**: base의 `compute_control`이
  weights를 받아 하는 일(가중 평균, 시프트, info)은 전부 shape `(K,)`와
  `Σw=1`만 가정한다. 이 두 계약만 지키면 샘플링·rollout·warm start를 공짜로
  물려받는다 — Log/Tsallis/Risk-Aware/ASR이 전부 이 방식이다 (§6.3).
  반대로 샘플링 분포 자체를 바꾸는 변형(DIAL, Biased)은 `compute_control`을
  통째로 오버라이드해야 한다 — 그건 다음 편의 주제.
- `np.argpartition(costs, k-1)[:k]`: 하위 k개를 O(K)에 뽑는다. 전체 정렬
  (`argsort`, O(K log K))이 필요 없다 — 엘리트 내부 순서는 softmax에 무의미.
- ESS 관점 예측: 가중치를 k=64개로 제한했으니 ESS ≤ 64. 실측 8.3 (아래) —
  엘리트 안에서도 softmax가 추가로 집중시킨 것. λ를 키우면 ESS가 64에
  다가가고, 이 손잡이 감각이 §6.5에서 말한 튜닝 감각이다.
- info dict의 `sample_weights`에 0이 대량 포함되지만 렌더러는 alpha=0으로
  그릴 뿐이므로 시각화 계약도 그대로 성립한다.

### 9.3 실행과 검증 결과

```bash
PYTHONPATH=. python topk_mppi.py   # 저장소 루트에서
```

실제 실행 결과 (K=512, top_k=64, 원형 궤적 15초, seed=42):

```
[invariants OK] nonzero=64, ess=8.3
Vanilla RMSE = 0.1551 m
Top-K   RMSE = 0.1566 m (top_k=64/512)
PASS
```

Vanilla와 동급 RMSE(둘 다 성능 기준 0.2m 이내)를 내면서 가중치 불변식
(shape `(K,)`, 합=1, 비영 ≤ 64, 비음수)을 모두 통과했다. 여기서 더 나아가는
숙제: (1) `top_k=1`로 두면 greedy(argmin) 선택이 된다 — 장애물 시나리오에서
chattering이 생기는지 확인해 보라. (2) pytest로 승격하려면
`tests/test_base_mppi.py`의 기존 테스트 구조(모델+params+controller 조립 →
`compute_control` 1회 호출 → shape/불변식 assert)를 복제하면 된다.
(3) 정식 변형으로 편입한다면: `controllers/mppi/topk_mppi.py` +
`mppi_params.py`에 Params 추가 + `tests/test_topk_mppi.py` + 벤치마크 —
저장소의 43종이 전부 이 4종 세트로 관리된다.

### 9.4 변형 작성 종합 체크리스트

1. **가중치만 바꾸나?** → `_compute_weights` 오버라이드 (이 절 방식).
   반환 `(K,)`, 합 1, 비음수 유지.
2. **샘플링/반복 구조를 바꾸나?** → `compute_control` 전체 오버라이드.
   시그니처 `(state, reference_trajectory) -> (control, info)`와
   info 표준 키(§6.5)는 반드시 유지.
3. Params는 §8.2 패턴 (상속 + `super().__post_init__()` + assert +
   "기본값 = 무해" 철학).
4. 재현 실험은 시드 있는 샘플러 주입 (§4.2 함정).
5. 검증 순서: 불변식 단위 테스트 → Simulator 폐루프 RMSE(< 0.2m) →
   기존 벤치마크에 4-Way 비교로 편입.

---

## 정리

| 절 | 파일 | 한 줄 요약 |
|----|------|-----------|
| §2 | `models/base_model.py` | `forward_dynamics` 한 줄 + `[..., i]`/`stack(-1)` 규약이면 배치 RK4가 공짜 |
| §3 | `dynamics_wrapper.py` | 샘플 축은 벡터화, 시간 축 루프는 순차 의존성 때문에 불가피 |
| §4 | `sampling.py` | 노이즈(ε)를 반환하는 프로토콜, seed=None은 전역 시드와 무관 |
| §5 | `cost_functions.py` | `(K,N+1,nx) → (K,)` 계약, 위반하면 조용히 또는 시끄럽게 죽는다 |
| §6 | `base_mppi.py` | softmax(-(S-S_min)/λ)와 warm start 시프트가 MPPI의 전부 |
| §7 | `simulation/` | 예측 모델과 실제 모델의 분리, process noise, RMSE 지표 |
| §8 | `mppi_params.py` | dataclass 상속 + post_init 체인 + "기본값 = 무해" |
| §9 | (실습) | `_compute_weights` 오버라이드만으로 44번째 변형 후보 완성 |

다음 단계: 이 코어 파이프라인 위에 각 변형이 어느 지점을 어떻게 갈아끼우는지
([../MPPI_THEORY.md](../MPPI_THEORY.md)의 변형 사전과 함께), 그리고 안전
계열이 비용/필터/rollout에 개입하는 방식([../SAFETY_THEORY.md](../SAFETY_THEORY.md),
[03](03_CBF_FUNDAMENTALS.md)·[04](04_ADVANCED_SAFETY.md)편)을 코드로 추적하는
후속 워크스루로 이어진다.
