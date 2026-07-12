# 잔차 MPPI 실로봇 배포 플레이북 — Sim → 잔차 학습 → 검증 → 실기

> **이 문서의 위치**: [09_RESIDUAL_MPC_MPPI.md](09_RESIDUAL_MPC_MPPI.md)가 잔차 동역학의
> **이론과 코드**(왜 잔차인가, `f_total = f_phys + f_learned` 유도, 잔차 제어 vs 잔차 동역학 구분,
> MPPI 결합)를 담당한다면,
> 이 문서는 그것을 **실제 로봇에 올리는 배포 절차서**다. 시뮬레이션에서 데이터를 뽑고,
> 잔차를 학습하고, sim-to-real 갭을 정량화하고, ROS2로 배선하고, 안전 계층을 씌워
> 실기에 넣기까지 — 엔지니어가 순서대로 따라갈 수 있는 **체크리스트·표·명령어** 중심 문서다.
>
> 이론 유도는 09편, 실시간 파이프라인 코드는 [06_CODE_WALKTHROUGH_CORE.md](06_CODE_WALKTHROUGH_CORE.md),
> 안전 계층은 [04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md) /
> [08_CODE_WALKTHROUGH_SAFETY.md](08_CODE_WALKTHROUGH_SAFETY.md)가 담당한다.
> 여기서는 그 조각들을 **하나의 배포 워크플로우**로 엮는다. 중복 서술 대신 링크로 넘긴다.

## 목차

1. [배포 파이프라인 개요](#1-배포-파이프라인-개요)
2. [데이터 구성 — 무엇을 어떻게 로깅하나](#2-데이터-구성)
3. [학습 방법 — 오프라인 · 온라인 · 무학습 적응](#3-학습-방법)
4. [적용(배포) 방법 — 잔차를 MPPI·ROS2에 꽂기](#4-적용배포-방법)
5. [실로봇 적용 전 체크리스트 & 테스트](#5-실로봇-적용-전-체크리스트--테스트)
6. [모델별 배포 노트 (DiffDrive / Ackermann / Swerve)](#6-모델별-배포-노트)
7. [FAQ 포인터 + 부록](#7-faq-포인터--부록)

---

## 0. 이 문서가 답하는 4가지 축

사용자가 실기 배포에서 반복적으로 부딪히는 4가지 질문을, **로봇 모델별**로 정리한다.

| 축 | 핵심 질문 | 담당 절 |
|----|----------|--------|
| **테스트** | 실기에 넣기 전에 무엇을 통과시켜야 하나? | §5 |
| **데이터 구성** | 어떤 튜플을, 얼마나, 어떤 커버리지로 모으나? | §2 |
| **학습 방법** | 오프라인 MLP? 온라인 재학습? 무학습 실시간 추정? 언제 무엇을? | §3 |
| **적용 방법** | 학습된 잔차를 MPPI·ROS2·안전계층에 어떻게 배선하나? | §4 |

한 줄 요약: **물리 모델(`f_phys`)은 그대로 두고, sim-to-real 갭만 잔차(`f_res`)로 학습해
MPPI 롤아웃 동역학에 투명하게 더한다. 나머지(속도 제한·안전 필터·워치독)는 보수적으로.**

---

## 1. 배포 파이프라인 개요

### 1.1 전체 흐름

```
 [단계 0] 시스템 식별                       산출물
 ────────────────────                      ─────────────
 wheelbase 실측, c_v/c_ω step-response,  →  configs/*.yaml 물리 파라미터
 mass/inertia (동적 모델 쓸 때만)             (§2.4)
        │
        ▼
 [단계 1] 데이터 수집 (sim 또는 실주행)      DataCollector 로그
 ─────────────────────────────────        (state, control,
 명목 컨트롤러로 주행하며 전이 튜플 로깅   →   next_state, dt)
 커버리지: 속도·조향·곡률 골고루              (§2.1–2.3)
        │
        ▼
 [단계 2] 잔차 학습                          residual_fn / uncertainty_fn
 ─────────────────────                     학습된 가중치 (.pt)
 target = (next-state)/dt − f_phys      →   ResidualDynamics 주입 준비
 NeuralNetworkTrainer(MLP) / GP / Ensemble    (§3.1–3.2)
        │
        ▼
 [단계 3] 검증 & 갭 정량화                   ModelValidator: RMSE/MAE/R²
 ─────────────────────────                  ConformalPredictor: 커버리지 마진
 1-step 오차 + rollout 오차 + 갭 벤치마크 →   E2E 스크립트 리포트
 (PerturbedDiffDriveModel / DynamicWorld)     (§3.3, §5)
        │
        ▼
 [단계 4] 실기 배선 & 배포                   실행 중인 노드
 ────────────────────                       cmd_vel 스트림
 ResidualDynamics → MPPI → ROS2 노드     →   안전 필터 + 워치독
 보수적 v_max, shield 기본, 폴백             (§4)
```

각 단계는 **되돌아갈 수 있는 게이트**다. §5의 체크리스트 행 하나가 실패하면
직전 단계로 되돌아간다 (예: rollout RMSE가 크면 단계 1로 → 데이터 커버리지 보강).

### 1.2 한 번에 돌려보는 E2E 도구

단계 1~3을 손으로 엮는 대신, 이 저장소에는 **hands-on 파이프라인 스크립트**가 있다:

```bash
# 데이터 수집 → 잔차 학습 → 갭 평가를 한 번에 (모델별)
PYTHONPATH=. python examples/learned/residual_sim2real_pipeline.py \
    --model diffdrive --episodes 8 --epochs 40 --no-plot
PYTHONPATH=. python examples/learned/residual_sim2real_pipeline.py --model ackermann --no-plot
PYTHONPATH=. python examples/learned/residual_sim2real_pipeline.py --model swerve --no-plot
```

`--model {diffdrive,ackermann,swerve}`로 로봇 모델을 고르고, `--episodes`(수집 에피소드 수),
`--epochs`(학습 에폭), `--duration`(에피소드 길이), `--seed`를 조절한다
([residual_sim2real_pipeline.py:658-667](../../examples/learned/residual_sim2real_pipeline.py)).
이 스크립트는 §2~§3의 워크플로우를 **모델별로 실증**하고, Baseline(물리만) vs Residual(물리+잔차)
vs Oracle(완전 관측 모델)의 rollout RMSE를 리포트한다.

**E2E 실측 갭 수치** (`--episodes 5 --epochs 30 --duration 12 --seed 42`, 검증 실행):

| 모델 | Baseline RMSE (m) | Residual RMSE (m) | Oracle RMSE (m) | 개선율 | 1-step 잔차 RMSE |
|---|---|---|---|---|---|
| diffdrive | 0.5833 | **0.4737** | 0.5740 | +18.8% | 0.0148 |
| ackermann | 1.6234 | **0.7205** | 1.4417 | +55.6% | 0.0947 |
| swerve | 0.1040 | **0.0692** | 0.0701 | +33.5% | 0.0251 |

세 모델 모두 Residual < Baseline이고, swerve는 Residual ≈ Oracle로
학습된 잔차가 사실상 완전 관측 성능을 회복했다. (수치는 대표값 — MPPI 샘플러가
`seed=None`이라 실행마다 소폭 변동한다. 재현성 함정은 [06편](06_CODE_WALKTHROUGH_CORE.md)
§4 참조.) **핵심 교훈**: 첫 실행에서는 잔차 타겟
`(next−state)/dt`가 heading θ의 ±π 경계에서 유한차분이 폭주(−125 rad/s)해 학습이 실패했다
(1-step RMSE 3.36). 각도 차원을 `atan2(sinΔ, cosΔ)`로 unwrap한 뒤 dt로 나누도록 고쳐
(diffdrive 1-step RMSE 3.36→0.0148) 세 모델 모두 통과했다 — §2.2의 "잔차 타겟 부호/wrap
주의"가 실전에서 어떻게 문제가 되는지 보여주는 실제 사례다.

---

## 2. 데이터 구성

### 2.1 무엇을 로깅하나 — 전이 튜플

잔차 학습의 원재료는 **상태 전이 튜플** `(state, control, next_state, dt)`다.
이 저장소의 `DataCollector`가 정확히 이 네 필드를 모은다
([data_collector.py:15](../../mppi_controller/learning/data_collector.py)):

```python
# DataCollector.add_sample(state, control, next_state, dt)
collector = DataCollector(state_dim=nx, control_dim=nu, max_samples=100_000)
collector.add_sample(state, control, next_state, dt)   # 매 제어 스텝마다
...
data = collector.get_data()   # dict: states, controls, next_states, state_dots, dt
```

`get_data()`는 편의상 `state_dots = (next_states - states) / dt`를 함께 계산해 준다.
이것이 잔차 타겟 계산의 절반이다.

**수집 루프 예시** — 명목 컨트롤러로 주행하며 매 제어 스텝 튜플을 남긴다:

```python
collector = DataCollector(state_dim=model.state_dim, control_dim=model.control_dim)
state = env.reset()
for step in range(num_steps):
    control, _ = controller.compute_control(state, reference)   # 명목 컨트롤러
    next_state = plant.step(state, control, dt)                  # 실기/고충실도 플랜트
    collector.add_sample(state, control, next_state, dt)
    state = next_state
collector.end_episode()      # 에피소드 경계 표시 (rollout 검증 분할에 사용)
collector.save("residual_diffdrive.npz")
```

`end_episode()`로 에피소드 경계를 남기면, 나중에 `evaluate_rollout`(§3.3)이 에피소드 단위로
연속 궤적을 재구성해 누적 오차를 잴 수 있다. `max_samples`(기본 100k)를 넘으면 오래된
샘플이 밀려난다.

### 2.2 잔차 타겟 정의 — 부호에 주의

**잔차 타겟은 "관측된 상태 미분"에서 "물리 모델이 예측한 상태 미분"을 뺀 것**이다:

```
f_res_target = (next_state - state) / dt  −  f_phys(state, control)
              └──────── 관측 state_dot ────┘   └── 물리 예측 ──┘
```

- 첫 항 `(next-state)/dt`는 `DataCollector`의 `state_dots`와 동일.
- 둘째 항 `f_phys(state, control)`는 물리 모델의 `forward_dynamics(state, control)` 호출.
- **부호 함정**: 순서를 뒤집으면(`f_phys − 관측`) 잔차가 반대 방향으로 학습되어
  MPPI 롤아웃이 발산한다. 항상 `관측 − 물리` 순서.

이 정의는 09편의 `f_total = f_phys + f_res` (아래 §4.1의 `ResidualDynamics`)와 정확히 짝을 이룬다:
학습이 `f_res ≈ 관측 − 물리`를 맞추면, `f_phys + f_res ≈ 관측`이 되어 갭이 메워진다.
적응 모델들(`MAMLDynamics.adapt`, `ALPaCADynamics.adapt`)도 내부에서 동일하게
`targets = (next_states - states) / dt`로 타겟을 만든다
([maml_dynamics.py:101](../../mppi_controller/models/learned/maml_dynamics.py),
[alpaca_dynamics.py:187](../../mppi_controller/models/learned/alpaca_dynamics.py)).

### 2.3 정규화·커버리지·데이터량

**정규화는 훈련셋에서만** 통계를 뽑아야 한다 (검증셋/실주행 데이터가 새면 리크).
`DynamicsDataset`이 `train_ratio`로 분할한 뒤 **훈련 분할의 mean/std로만** 정규화한다
([data_collector.py:278](../../mppi_controller/learning/data_collector.py)):

```python
dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True, shuffle=True)
train_in, train_tgt = dataset.get_train_data()
val_in, val_tgt = dataset.get_val_data()
norm_stats = dataset.get_normalization_stats()   # state/control/state_dot mean·std
```

`norm_stats`는 학습·추론 양쪽에서 재사용된다 (배포 시 로드해 실기 관측을 같은 통계로 정규화).

**커버리지** — 데이터가 실주행에서 만나는 상태·제어 분포를 골고루 덮어야 한다:

| 항목 | 왜 중요한가 | 실무 팁 |
|------|-----------|--------|
| 속도 분포 | 고속에서만 나타나는 마찰·슬립 갭이 있음 | 정지→최대속까지 램프·스텝 포함 |
| 조향/각속도 분포 | 급회전 갭(Ackermann 슬립, DiffDrive 미끄러짐) | 좌우 대칭, 다양한 곡률 |
| 곡률 분포 | 직선만 모으면 회전 갭 학습 불가 | 8자·슬라럼 궤적 섞기 |
| 정지·재출발 | 데드밴드·정지 마찰 | 저속 구간 별도 수집 |

커버리지가 부족하면 잔차가 **분포 밖(OOD)에서 외삽**하며 위험해진다 → 이때 GP/Ensemble의
불확실성(§3.1)이나 `ConformalPredictor` 마진(§3.3)으로 방어한다.

**데이터량 가이드** (경험칙, 상태·제어 차원에 비례):

| 모델 | 권장 전이 수 (오프라인 MLP) | 비고 |
|------|--------------------------|------|
| DiffDrive (3/2) | 5k–20k | 가장 적게 필요 |
| Ackermann (4/2) | 10k–30k | 조향각 차원 추가 → 커버리지 부담 |
| Swerve (3/3) | 10k–30k | 제어 3D → vx·vy·ω 조합 폭발 |

**엘리트/링버퍼** — 온라인 상황에서는 전량 저장 대신 순환 버퍼를 쓴다.
`OnlineDataBuffer`가 FIFO(deque, maxlen) 순환 버퍼 + `should_retrain()` 트리거를 제공하고
([online_learner.py:16](../../mppi_controller/learning/online_learner.py)),
제안 분포 학습 계열은 `FlowDataCollector`가 (state, optimal_U) 쌍을 elite로 모은다
([flow_data_collector.py:14](../../mppi_controller/learning/flow_data_collector.py)) —
후자는 잔차 동역학이 아니라 05편의 생성 제안 분포용이므로 혼동 주의.

### 2.4 모델별 상태/제어 차원 + 시스템 식별

먼저 각 로봇 모델의 인터페이스를 못박는다 (소스 검증됨):

| 모델 | 클래스 (file) | state_dim | control_dim | 상태 벡터 | 제어 벡터 |
|------|--------------|:--------:|:-----------:|----------|----------|
| DiffDrive kin | `DifferentialDriveKinematic` ([kinematic/](../../mppi_controller/models/kinematic/differential_drive_kinematic.py)) | 3 | 2 | `[x, y, θ]` | `[v, ω]` |
| DiffDrive dyn | `DifferentialDriveDynamic` ([dynamic/](../../mppi_controller/models/dynamic/differential_drive_dynamic.py)) | 5 | 2 | `[x, y, θ, v, ω]` | `[a, α]` |
| Ackermann kin | `AckermannKinematic` ([kinematic/](../../mppi_controller/models/kinematic/ackermann_kinematic.py)) | 4 | 2 | `[x, y, θ, δ]` | `[v, φ]` (φ=조향률) |
| Ackermann dyn | `AckermannDynamic` ([dynamic/](../../mppi_controller/models/dynamic/ackermann_dynamic.py)) | 5 | 2 | `[x, y, θ, v, δ]` | `[a, φ]` |
| Swerve kin | `SwerveDriveKinematic` ([kinematic/](../../mppi_controller/models/kinematic/swerve_drive_kinematic.py)) | 3 | 3 | `[x, y, θ]` | `[vx, vy, ω]` (바디프레임) |
| Swerve dyn | `SwerveDriveDynamic` ([dynamic/](../../mppi_controller/models/dynamic/swerve_drive_dynamic.py)) | 6 | 3 | `[x, y, θ, vx, vy, ω]` | `[ax, ay, α]` |

**시스템 식별 (단계 0)** — 잔차가 메우는 갭을 최소화하려면 물리 파라미터부터 실측한다.
잔차는 "남은 갭"을 메울 뿐, **틀린 물리 파라미터를 잔차로 덮으려 하면 데이터 폭증**한다.

| 파라미터 | 어느 모델 | 식별 방법 |
|----------|----------|----------|
| `wheelbase` | Ackermann(kin/dyn) | 앞·뒷축 거리 **자로 실측** (가장 확실). 생성자 기본값 `0.5` |
| `c_v` (선형 마찰) | Dynamic 계열 | **step-response 피팅**: 일정 가속 명령 후 정상속도 도달 곡선에 `v̇ = a − c_v·v` 피팅. 기본값 `0.1` |
| `c_omega` (각 마찰) | Dynamic 계열 | 각속도 step-response에 `ω̇ = α − c_ω·ω` 피팅. 기본값 `0.1` |
| `mass`, `inertia` | Dynamic 계열 | mass는 저울, inertia는 회전 실험 또는 CAD. 기본값 `10.0`, `1.0` |
| `v_max`, `omega_max` | 전 모델 | 데이터시트 또는 최대 명령 실측. **배포 시 보수적으로 낮춤** (§4.3) |

> **kinematic vs dynamic 선택**: 실기 배포 초기에는 **kinematic 모델 + 잔차**를 권장한다.
> 식별할 물리 파라미터가 적고(`v_max`, `omega_max`, wheelbase만), 갭 대부분을 잔차가
> 흡수하기 때문. dynamic 모델은 `c_v`/`c_omega`/`mass`/`inertia`를 제대로 식별할 수 있을 때만.

**step-response로 `c_v` 피팅 (dynamic 모델용)** — 정지 상태에서 일정 가속 명령 `a`를 주고
속도 로그 `v(t)`를 기록한 뒤, `DifferentialDriveDynamic`의 `v̇ = a − c_v·v`
([differential_drive_dynamic.py](../../mppi_controller/models/dynamic/differential_drive_dynamic.py))의
해석해 `v(t) = (a/c_v)(1 − e^{−c_v·t})`를 피팅한다:

```python
from scipy.optimize import curve_fit
import numpy as np
def v_model(t, c_v, a):
    return (a / c_v) * (1.0 - np.exp(-c_v * t))
(c_v_hat, a_hat), _ = curve_fit(v_model, t_log, v_log, p0=[0.1, 1.0])
```

각속도도 동일하게 `ω̇ = α − c_ω·ω`로 `c_ω`를 피팅. 이렇게 물리를 맞춘 뒤 **남은 갭만**
잔차로 학습하면 데이터량이 크게 준다.

### 2.5 sim 수집 vs 실주행 수집 — 무엇이 다른가

E2E 스크립트(§1.2)는 **sim에서** 갭을 인위 주입(§5 게이트 6~7의 Perturbed/DynamicWorld)해
잔차 워크플로우를 실증한다. 하지만 진짜 배포에서는 **실주행 로그**가 필요하다:

| 축 | sim 수집 | 실주행 수집 |
|----|---------|------------|
| 갭 원천 | 인위 주입(바이어스/마찰/외란) | 실제 미지 동역학·센서 지연·지면 |
| 커버리지 | 궤적 스크립트로 통제 가능 | 안전 범위 내에서만 (저속 우선) |
| 노이즈 | 알려진 분포 | 미지, 상관 있음 → 정규화·필터 주의 |
| 용도 | 파이프라인·안전 로직 검증 | **최종 잔차 학습·CP 마진 보정** |

권장 순서: sim에서 파이프라인·안전 로직을 굳힌 뒤(게이트 1~8), **섀도우 모드(§5)로 실주행
로그를 모아 잔차를 재학습**한다. sim 잔차를 실기에 그대로 올리지 말 것 — sim 갭과 실기 갭은 다르다.

---

## 3. 학습 방법

### 3.1 오프라인 학습 — MLP가 기본, 불확실성 필요하면 GP/Ensemble

**결정론적 잔차만 필요하면 MLP** — `NeuralNetworkTrainer`
([neural_network_trainer.py:100](../../mppi_controller/learning/neural_network_trainer.py)):

```python
trainer = NeuralNetworkTrainer(
    state_dim=nx, control_dim=nu,
    hidden_dims=[128, 128], activation="relu",
    dropout_rate=0.1, learning_rate=1e-3, weight_decay=1e-5,
)
history = trainer.train(
    train_in, train_tgt, val_in, val_tgt, norm_stats,
    epochs=100, batch_size=64, early_stopping_patience=20,
)
# 추론: predict(state, control, denormalize=True) → state_dot 예측
```

입력 `[state, control]` 연결(nx+nu) → 출력 state_dot(nx)의 MLP. `early_stopping_patience`로
과적합 방어, `spectral_lambda`로 Lipschitz 정규화(외삽 안정성)를 켤 수 있다.

주의: `NeuralNetworkTrainer.predict()`가 반환하는 것은 **전체 state_dot**이다. 잔차만
쓰려면 `residual_fn`에서 `predict − f_phys`를 빼거나, **애초에 타겟을 잔차(§2.2)로 학습**해
`predict()`가 곧 잔차가 되게 한다 (후자 권장 — MPPI의 `ResidualDynamics.residual_fn`에 직접 꽂힘).

**모델별 학습 하이퍼파라미터 출발점** (경험칙, 여기서 튜닝 시작):

| 모델 | hidden_dims | epochs | batch_size | 비고 |
|------|-------------|:------:|:----------:|------|
| DiffDrive (3/2) | `[64, 64]` | 40–80 | 64 | 작은 net으로 충분, 실시간 여유 |
| Ackermann (4/2) | `[128, 128]` | 60–100 | 64 | 조향각 비선형성 → 용량 ↑ |
| Swerve (3/3) | `[128, 128]` | 60–100 | 64 | 제어 3D 조합 → 용량 ↑ |

net이 클수록 롤아웃당 `predict` 비용이 커지므로(§4.4), 실시간 예산과 트레이드오프한다.

**불확실성이 필요하면** (OOD 방어, 리스크 인지 MPPI 연동) 두 선택지:

| 트레이너 | 불확실성 | file | 특징 |
|---------|:-------:|------|------|
| `GaussianProcessTrainer` | mean + std | [gaussian_process_trainer.py:132](../../mppi_controller/learning/gaussian_process_trainer.py) | `predict(..., return_uncertainty=True)` → `(mean, std)`. 데이터 적을 때 강함, 대량엔 sparse(`use_sparse`) |
| `EnsembleTrainer` | 앙상블 분산 | [ensemble_trainer.py:19](../../mppi_controller/learning/ensemble_trainer.py) | `num_models`개 MLP + `bootstrap`. GP보다 대량 데이터·고차원에 유리 |

불확실성 `std`는 `ResidualDynamics`의 `uncertainty_fn`으로 주입해 위험한 잔차 예측을
MPPI 비용에서 페널티할 수 있다 (09편 §리스크 인지 참조).

### 3.2 온라인 / 적응 — 재학습 vs 무학습 실시간 추정

실기에서 갭이 **시간에 따라 변할 때**(타이어 마모, 노면 변화, 배터리 저하) 온라인 적응이 필요하다.
크게 두 갈래:

**(A) 온라인 재학습 (SGD 기반)** — `OnlineLearner`
([online_learner.py:220](../../mppi_controller/learning/online_learner.py)):

```python
learner = OnlineLearner(
    model, trainer,
    buffer_size=1000, min_samples_for_update=100, update_interval=500,
    checkpoint_dir="checkpoints/", max_checkpoints=10,
)
learner.add_sample(state, control, next_state, dt)   # 매 스텝
# 내부: buffer.should_retrain() → update_model() 자동 트리거
```

핵심 안전장치는 **자동 롤백**: val_loss가 best의 1.5배를 넘으면 성능 저하로 판단해
최고 체크포인트로 되돌린다 (`rollback(version=None)`). 실기에서 나쁜 배치로 모델이
망가지는 것을 막는 필수 장치다.

**(B) Few-shot 메타 적응** — `MAMLDynamics.adapt()`
([maml_dynamics.py:28](../../mppi_controller/models/learned/maml_dynamics.py)):

```python
loss = maml.adapt(states, controls, next_states, dt, restore=True)
# inner_lr·inner_steps회 gradient step으로 few-shot 적응, restore=True면 메타 가중치 복원
```

메타 학습된 초기 가중치에서 **수 스텝 gradient로 새 조건에 적응**. 조건이 자주 바뀌는
환경(여러 노면 사이 전환)에 적합.

**(C) 무학습 실시간 추정 (gradient 없음)** — 계산 예산이 빠듯하거나 학습 인프라 없이
파라미터/외란만 추정할 때:

| 모델 | file | 추정 대상 | 방식 | 학습 필요? |
|------|------|----------|------|:---------:|
| `EKFAdaptiveDynamics` | [ekf_dynamics.py:27](../../mppi_controller/models/learned/ekf_dynamics.py) | `c_v`, `c_ω` (7D 확장 상태) | EKF predict/update | 무학습 |
| `L1AdaptiveDynamics` | [l1_adaptive_dynamics.py:25](../../mppi_controller/models/learned/l1_adaptive_dynamics.py) | 미지 외란 σ | 상태 예측기 + 저역통과 | 무학습 |
| `ALPaCADynamics` | [alpaca_dynamics.py:52](../../mppi_controller/models/learned/alpaca_dynamics.py) | 마지막 층 (베이지안) | closed-form 사후 (SGD 없음) | 메타 사전학습만 |

EKF/L1은 **완전 무학습** — 물리 구조 위에서 실시간으로 마찰·외란을 추정한다.
ALPaCA는 특징 추출기는 사전 메타학습하되 온라인 적응은 행렬 대수(무 SGD)라 매우 빠르다.

**"언제 무엇을" 결정표**:

| 상황 | 데이터량 | 불확실성 필요 | 계산 예산 | 적응 속도 | → 권장 |
|------|:-------:|:-------------:|:--------:|:--------:|-------|
| 갭 고정, 사전 로그 충분 | 많음 | 아니오 | 여유 | 불필요 | **NeuralNetworkTrainer (MLP)** |
| 갭 고정, 데이터 적음 | 적음 | 예 | 여유 | 불필요 | **GaussianProcessTrainer** |
| 갭 고정, OOD 방어 중요 | 많음 | 예 | 여유 | 불필요 | **EnsembleTrainer** |
| 갭 서서히 변함 | 스트림 | 선택 | 중간 | 느림~중간 | **OnlineLearner (+롤백)** |
| 조건이 자주 전환 | few-shot | 아니오 | 중간 | 빠름 | **MAML.adapt()** |
| 마찰/외란만, 인프라 최소 | 없음 | (내장) | 최소 | 실시간 | **EKF / L1 / ALPaCA** |

### 3.3 검증 — 배포 게이트

학습된 잔차는 **실기에 넣기 전 반드시 검증**한다. 두 도구:

**`ModelValidator`** ([model_validator.py:11](../../mppi_controller/learning/model_validator.py)) —
1-step 정확도 + rollout 누적 오차:

```python
v = ModelValidator()
m = v.evaluate(predict_fn, test_states, test_controls, test_targets)
#   → rmse, mae, r2, per_dim_rmse, max_error ...
r = v.evaluate_rollout(model, init_states, control_seqs, true_trajs, dt)
#   → mean_rollout_rmse, per_step_rmse, worst_case_rmse
```

**1-step RMSE가 낮아도 rollout에서 발산할 수 있으므로** `evaluate_rollout`의
`per_step_rmse` 증가 곡선을 꼭 확인한다 — MPPI는 N스텝 롤아웃을 쓰기 때문.

**`ConformalPredictor`** ([conformal_predictor.py:31](../../mppi_controller/learning/conformal_predictor.py)) —
예측 오차에 **분포 무관 커버리지 보증** 마진 부여:

```python
cp = ConformalPredictor(ConformalPredictorConfig(alpha=0.1, gamma=0.95))
cp.update(predicted_state, actual_state)   # 매 스텝 논컨포미티 점수 갱신
margin = cp.get_margin()                    # 90% 커버리지 안전 마진 (m)
```

`alpha=0.1`이면 90% 커버리지, `gamma<1.0`이면 최근 데이터 강조(적응형 CP) — 실기에서
갭이 변할 때 마진이 따라 넓어진다. 이 마진을 안전 필터의 `safety_margin`에 더해
잔차 불확실성을 안전 여유로 환산할 수 있다.

### 3.4 불확실성을 리스크-인지 MPPI에 연결

GP/Ensemble의 `std` 또는 CP 마진은 단순 참고 지표가 아니라 **제어에 되먹임**할 수 있다:

| 되먹임 경로 | 방법 | 효과 |
|-------------|------|------|
| `ResidualDynamics.uncertainty_fn` | `std`를 반환 → 09편 잔차 기여도 분석 | 불확실 영역 롤아웃 인지 |
| 안전 마진 확장 | `cbf_safety_margin += CP_margin` | 불확실할수록 장애물에서 더 멀리 |
| 리스크-인지 가중 | CVaR/Risk-Aware MPPI (02편) | 최악 시나리오 회피 |
| 속도 상한 하향 | `std` 급증 → `v_max` 자동 축소(§4.6) | OOD 진입 시 감속 |

핵심 원칙은 **"모르면 보수적으로"** — 잔차가 자신 없는(OOD) 영역에서 공격적 제어를 내지 않게
불확실성을 명시적으로 제어에 반영한다. 이는 학습 요소가 있어도 학습 전=보수적 기본동작이라는
저장소 전반의 graceful degradation 철학(README 부록 B 4번)과 일치한다.

---

## 4. 적용(배포) 방법

### 4.1 잔차를 MPPI에 투명하게 꽂기 — ResidualDynamics

MPPI 컨트롤러는 **동역학 모델의 인터페이스만 알 뿐** 내부가 물리인지 물리+잔차인지 모른다.
`ResidualDynamics`가 이 투명성을 제공한다
([residual_dynamics.py:14](../../mppi_controller/models/learned/residual_dynamics.py)):

```python
# f_total(x,u) = f_phys(x,u) + f_res(x,u)
residual_model = ResidualDynamics(
    base_model=DifferentialDriveKinematic(v_max=0.5, omega_max=1.9),
    residual_fn=lambda s, u: trainer.predict(s, u),   # 학습된 잔차
    uncertainty_fn=None,                               # 있으면 GP/Ensemble std 주입
    use_residual=True,
)
controller = MPPIController(residual_model, params)   # 물리 모델처럼 그냥 넘김
```

`forward_dynamics()`가 `physics_dot + residual_dot`를 반환하므로
([residual_dynamics.py:89](../../mppi_controller/models/learned/residual_dynamics.py)),
MPPI의 K-병렬 롤아웃(`BatchDynamicsWrapper`,
[dynamics_wrapper.py:13](../../mppi_controller/controllers/mppi/dynamics_wrapper.py))이
잔차 포함 동역학으로 궤적을 펼친다. 컨트롤러 코드는 **한 줄도 안 바꾼다** — 이것이 잔차
접근의 핵심 장점 (06편 §RobotModel ABC 규약 참조).

### 4.2 Residual-MPPI — 잔차를 "정책"에 적용하는 별도 변형

동역학 잔차(§4.1)와 혼동하기 쉬운 별개 기법이 **Residual-MPPI**
([residual_mppi.py:36](../../mppi_controller/controllers/mppi/residual_mppi.py)) — 이건
사전 정책(base policy) 위에 **제어 잔차 δu**를 최적화한다 (동역학이 아니라 정책 잔차):

```python
controller = ResidualMPPIController(model, params, base_policy=None)
controller.set_base_policy(pure_pursuit_policy)   # 사전 정책 주입/변경
```

`set_base_policy(policy)`로 callable 또는 `AncillaryPolicy`를 주입한다
([residual_mppi.py:248](../../mppi_controller/controllers/mppi/residual_mppi.py)).
**정리**: 학습된 잔차 *동역학*은 §4.1의 `ResidualDynamics`, 학습/규칙 기반 잔차 *정책*은
이 §4.2의 `ResidualMPPIController`. 배포에서 둘을 함께 쓸 수도 있다(잔차 동역학 모델 +
사전 정책 잔차 제어).

### 4.3 ROS2 배선

**단일 노드 (직접 토픽)** — `MPPIControllerNode`
([mppi_controller_node.py](../../mppi_controller/ros2/mppi_controller_node.py)):

| 방향 | 토픽 | 타입 | 비고 |
|------|------|------|------|
| Sub | `/odom` | `nav_msgs/Odometry` | 상태 추출 ([x,y,θ] 또는 dynamic 시 +[v,ω]) |
| Sub | `/reference_path` | `nav_msgs/Path` | (N+1, state_dim) 참조로 변환 |
| Sub | `/scan` | `sensor_msgs/LaserScan` | `scan_enabled` 시 장애물 검출 |
| Pub | `/cmd_vel` | `geometry_msgs/Twist` | `linear.x=v`, `angular.z=ω` |

`control_rate`(기본 10Hz) 타이머로 `compute_control()`를 돌리고 solve 시간을 로깅한다.
주요 파라미터: `model_type`(kinematic/dynamic), `controller_type`(vanilla/shield/...),
`K`, `N`, `dt`, `v_max`, `omega_max`. 잔차 배포 시에는 `_create_model()`에서 만든 물리 모델을
`ResidualDynamics`로 감싸 컨트롤러에 넘기도록 확장한다 (§4.1 패턴).

**Nav2 통합 (권장, 전역 계획 연동)** — `MPPIFollowPathServer`
([follow_path_server.py:101](../../mppi_controller/ros2/nav2/follow_path_server.py))가
`FollowPath` 액션 서버로 동작한다. PathWindower / CostmapConverter / GoalChecker /
ProgressChecker 파이프라인은 [NAV2_INTEGRATION.md](../NAV2_INTEGRATION.md)에 상세하므로
여기서는 중복하지 않는다 (그 문서로 이동).

```bash
ros2 launch learning_mppi mppi_nav2.launch.py controller_type:=shield
```

**보수적 배포 기본값** — `configs/mppi_nav2.yaml`은 실기 안전을 위해 **보수적으로** 세팅돼 있다:

```yaml
# configs/mppi_nav2.yaml (검증됨)
v_max: 0.5                 # 단독 컨트롤러 기본 1.0 → 절반으로 낮춤
controller_type: shield    # safety-first 기본 (per-step CBF 강제)
shield_enabled: true
cbf_alpha: 0.3
robot_radius: 0.22
```

실기 초기 배포는 **이 보수적 기본값에서 시작**해 점진적으로 `v_max`를 올린다.

### 4.4 실시간 예산

| 항목 | 기준 | 근거 |
|------|------|------|
| solve 시간 | < 100ms (K=1024, N=30) | CLAUDE.md 성능 기준 |
| 제어 주기 | 10~20Hz (단독 10Hz / nav2 20Hz) | 노드 `control_rate` / `controller_frequency` |

예산 초과 시 K/N을 줄인다. 잔차 MLP `predict`가 롤아웃마다 K×N회 호출되므로
**잔차 네트워크는 작게**(hidden 128 이하) 유지한다. 실시간 최적화·프로파일링은
[06_CODE_WALKTHROUGH_CORE.md](06_CODE_WALKTHROUGH_CORE.md)(핵심 파이프라인)와
02편 샘플링 축을 참조.

**K/N 축소 가이드** (solve 시간 vs 성능 트레이드오프):

| 플랫폼 | K | N | 비고 |
|--------|:-:|:-:|------|
| 데스크톱/워크스테이션 | 1024 | 30 | 기본 (CLAUDE.md 기준) |
| 임베디드 (Jetson 등) | 512 | 20–30 | NAV2_INTEGRATION.md 성능 튜닝 권고 |
| 저사양 SBC | 256 | 15–20 | dsMPPI 등 결정론 샘플러면 K=64에서도 동작(02편) |

K를 줄이면 ESS가 낮아져 탐색이 약해진다 → `sigma`를 키우거나 warm start를 강화해 보완.
N을 줄이면 계획 지평이 짧아진다 → TD-MPPI류 terminal value로 보완 가능(02편).

### 4.5 안전 계층

잔차 모델은 갭을 줄이지만 **안전을 보증하지 않는다**. 실기에는 반드시 안전 계층을 얹는다:

- **Shield-MPPI / CBF-MPPI** — per-step CBF 클리핑 또는 CBF 비용. nav2 기본 `controller_type: shield`.
- **CBF 안전 필터** — 반환 직전 제어 보정.
- **Gatekeeper** — 백업 궤적 검증 게이트 (Simulator에 넘기기 직전).

3층위(비용/필터/게이트)가 코드 어디에 끼어드는지는
[08_CODE_WALKTHROUGH_SAFETY.md §1](08_CODE_WALKTHROUGH_SAFETY.md), 이론은
[04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md) 참조 (여기서 중복하지 않음).

### 4.6 워치독 & 폴백

실기 안전의 최후 방어선. 노드는 이미 다음 폴백을 갖고 있다
([mppi_controller_node.py:434-436, 467-472](../../mppi_controller/ros2/mppi_controller_node.py)):

- `compute_control()` 예외 → `publish_zero_velocity()` (정지).
- 참조 경로 미수신 → 정지.

배포 시 추가로 권장:

| 조건 | 대응 |
|------|------|
| 제어값 NaN/Inf | 즉시 정지 (`np.isfinite` 체크) |
| solve 시간 예산 초과 반복 | 이전 해 유지 or 정지, 알람 |
| 오도메트리 stale (수신 끊김) | 정지 (throttle 워닝은 이미 존재) |
| 잔차 예측 불확실성 급증(CP 마진 폭증) | 속도 상한 자동 하향 or 정지 |

Nav2 경로에서는 `ProgressChecker`(stuck)와 `NO_VALID_CONTROL`(=103) 에러 코드가
BT Navigator에 리커버리를 트리거한다 ([follow_path_server.py:98,548](../../mppi_controller/ros2/nav2/follow_path_server.py)).

---

## 5. 실로봇 적용 전 체크리스트 & 테스트

실기에 넣기 전 **아래 표를 위→아래 순서로** 통과시킨다. 각 행은 되돌아갈 게이트다.
전체 테스트 카테고리·규칙은 [TESTING.md](../TESTING.md)를 따르며 여기서 중복하지 않는다.

| # | 게이트 | 기준 | 실행 명령 / 파일 |
|---|--------|------|-----------------|
| 1 | **단위 — 모델 forward/배치** | forward_dynamics·rollout shape/값 통과 | `pytest tests/test_robot_models.py -o "addopts=" -q` |
| 2 | **단위 — 잔차 결합** | `f_total = f_phys + f_res` 정확 | `pytest tests/ -k residual -o "addopts=" -q` |
| 3 | **성능 — 추적 RMSE** | 원형 궤적 RMSE < 0.2m | `python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory circle --no-plot` |
| 4 | **성능 — solve 시간** | < 100ms (K=1024, N=30) | 노드 solve 로그 / 벤치마크 SolveMs |
| 5 | **안전 — 무충돌** | 충돌 0 assert | `PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --scenario obstacles` |
| 6 | **모델 불일치** | 갭 존재 시 잔차가 baseline 개선 | `PYTHONPATH=. python examples/comparison/model_mismatch_comparison_demo.py --evaluate --world dynamic` |
| 7 | **외란 강건성** | wind/terrain/sine에서 무충돌·추적 유지 | `... model_mismatch_comparison_demo.py --evaluate --noise 0.5 --disturbance combined` |
| 8 | **sim-to-real 갭 정량화** | Residual < Baseline rollout RMSE | `PYTHONPATH=. python examples/learned/residual_sim2real_pipeline.py --model diffdrive --no-plot` |
| 9 | **검증 지표** | ModelValidator R²↑, rollout per-step 발산 없음 | §3.3 코드 |
| 10 | **커버리지 보증** | ConformalPredictor 경험적 커버리지 ≈ 1−α | §3.3 코드 |

**모델 불일치·외란 도구 (게이트 6~7)** — `model_mismatch_comparison_demo.py`가
sim-to-real 갭을 인위적으로 만든다:

- `PerturbedDiffDriveModel` ([:149](../../examples/comparison/model_mismatch_comparison_demo.py)) —
  액추에이터 바이어스 `[+0.12, −0.05]`, 마찰 감쇠(x,y ×0.55, θ ×0.80), 프로세스 노이즈.
- `DynamicWorld` + `DisturbanceProfile`
  ([disturbance_profiles.py:28](../../examples/comparison/disturbance_profiles.py)) —
  `--noise 0.0~1.0`(강도) · `--disturbance {none,wind,terrain,sine,combined}` CLI로
  바람 돌풍·노면 마찰 변화·주기 외란을 주입.

**섀도우 / HIL 모드 권고** — 실제 액추에이터에 명령을 보내기 전 단계적으로:

1. **HIL(Hardware-in-the-Loop)**: 실 컴퓨트 보드에서 노드를 돌리되 플랜트는 시뮬레이션.
   지연·지터·솔브시간을 실측해 §4.4 예산을 실기 하드웨어에서 재검증. `control_rate`를
   유지할 수 있는지, solve 스파이크(온라인 학습 스텝)가 주기를 깨지 않는지 확인.
2. **섀도우 모드**: 실기 센서(`/odom`, `/scan`)로 `compute_control()`를 돌리되
   `/cmd_vel`은 **발행하지 않고 로깅만** 한다 (사람이 로봇을 수동 주행).
   - 매 스텝 예측 next_state vs 실측 next_state를 비교 → **잔차가 실기 갭을 실제로 줄이는지**
     라이브 검증 (Baseline 물리 예측과 Residual 예측을 나란히 로깅).
   - `ConformalPredictor.update(pred, actual)`를 여기서 돌리면 실기 커버리지 마진이 warm-up된다.
   - 안전 필터가 언제 개입했는지(개입 횟수·보정 크기) 카운트 → 필터가 과도하게 개입하면
     `v_max`나 코스트 가중치 재조정.
3. **저속 실주행**: 통과 후 **낮은 `v_max`(예: 0.3 m/s)**로 실기 첫 자율 주행.
   워치독(§4.6)과 e-stop을 손에 든 채, 개활지에서 시작.
4. **점진 상향**: 무충돌·추적 RMSE 기준을 유지하며 `v_max`를 단계적으로(0.3→0.5→…) 올린다.

**섀도우 → 실주행 승격 판정 예시**:

| 지표 | 승격 기준 |
|------|----------|
| Residual rollout RMSE (섀도우 로그) | < Baseline rollout RMSE (개선 확인) |
| CP 경험적 커버리지 | ≈ 1−α (예: 0.9±0.05) |
| 안전 필터 개입율 | 낮고 안정적 (급증 없음) |
| solve 시간 (HIL) | p99 < 예산 |

---

## 6. 모델별 배포 노트

### 6.1 DiffDrive (예: TurtleBot3)

| 항목 | 노트 |
|------|------|
| Nav2 지원 | **최상** — TurtleBot3 튜토리얼 그대로 ([NAV2_INTEGRATION.md](../NAV2_INTEGRATION.md) §TurtleBot3) |
| 상태/제어 | kin `[x,y,θ]` / `[v,ω]` → `/cmd_vel` `linear.x`, `angular.z` 직접 매핑 |
| 잔차 초기 배포 | kinematic + 잔차 권장 (식별 파라미터 최소) |
| 시스템 식별 | `v_max`, `omega_max`만. wheelbase는 kin에서 미사용 |
| 데이터량 | 가장 적게 필요 (5k–20k) |
| 주의 | 정지·저속 데드밴드/미끄러짐 갭 → 저속 데이터 별도 수집 |

### 6.2 Ackermann (조향 제약)

| 항목 | 노트 |
|------|------|
| 상태/제어 | kin `[x,y,θ,δ]` / `[v,φ]` (φ=조향률). **δ는 상태**라 오도메트리에 조향각 필요 |
| 최소 회전 반경 | `R_min = wheelbase / tan(max_steer)`. 기본 wheelbase=0.5, max_steer=0.5rad → 참조 경로가 R_min 위반하면 추종 불가 |
| 시스템 식별 | **wheelbase 실측 필수** (오차가 곡률 갭으로 직결). `max_steer`·`steer_rate_max` 실측 |
| 잔차 커버리지 | 조향각 차원 추가 → 좌우 대칭·다양한 곡률 필수 (10k–30k) |
| 주의 | 후진·제자리 회전 불가. 참조 경로가 kinematically feasible한지 사전 검사 |
| 홀로노믹 아님 | Swerve와 달리 측방 이동 불가 → 참조가 이를 요구하면 갭 발산 |

### 6.3 Swerve (홀로노믹)

| 항목 | 노트 |
|------|------|
| 상태/제어 | kin `[x,y,θ]` / `[vx,vy,ω]` — 제어 3D, **바디프레임 속도** |
| 좌표계 | 제어가 바디프레임 → `/cmd_vel`도 바디프레임(`linear.x=vx`, `linear.y=vy`, `angular.z=ω`). 하드웨어 드라이버가 vy를 지원하는지 확인 |
| 홀로노믹 | 측방 이동 자유 → 참조 추종은 쉽지만 개별 모듈 조향/구동 갭이 vx·vy·ω 전반에 |
| 잔차 커버리지 | 제어 3D 조합 폭발 → vx·vy·ω 독립·결합 모두 수집 (10k–30k) |
| 주의 | 모듈 간 비대칭(휠 슬립 차이)이 잔차로 흡수됨 → 대칭 데이터 중요 |
| 동역학 모델 | dyn 6D `[x,y,θ,vx,vy,ω]` — vx·vy에 동일 `c_v` 마찰 적용됨 (모듈별 다르면 잔차로) |

### 6.4 모델 공통 배포 순서 (요약)

세 모델 모두 동일한 골격을 따르되 §6.1~6.3의 모델별 주의점을 끼운다:

```
1. 시스템 식별 (§2.4)      — wheelbase 실측(Ackermann), c_v/c_ω 피팅(dynamic)
2. sim 파이프라인 검증      — E2E 스크립트 --model {diffdrive|ackermann|swerve} (§1.2, §5 게이트 1~8)
3. 섀도우 로그 → 잔차 재학습 — 실주행 로그로 최종 잔차 학습 + CP 마진 보정 (§2.5, §5)
4. 보수적 배선              — v_max 낮춤, shield 기본, 워치독 (§4.3, §4.6)
5. 저속 실주행 → 점진 상향   — 승격 판정표 통과 시 v_max 상향 (§5)
```

**모델별 실패 모드 요약**:

| 모델 | 가장 흔한 배포 실패 | 1차 조치 |
|------|--------------------|---------|
| DiffDrive | 저속 데드밴드/미끄러짐 갭 | 저속 데이터 보강, 잔차 재학습 |
| Ackermann | 참조가 R_min 위반 → 추종 불가 | 참조 경로 feasibility 사전 검사, wheelbase 재실측 |
| Swerve | 드라이버가 vy 미지원 / 모듈 비대칭 | `/cmd_vel` vy 배선 확인, 대칭 데이터 수집 |

---

## 7. FAQ 포인터 + 부록

### 7.1 FAQ (증상 → 조치)

| 증상 | 원인 후보 | 조치 → 참조 |
|------|----------|------------|
| **실기에서 발산한다** | 잔차 부호 반대 / rollout 누적 오차 / 물리 파라미터 오식별 | §2.2 부호 확인 → `evaluate_rollout` per-step 곡선(§3.3) → 시스템 식별(§2.4). 안전 계층 확인(§4.5) |
| **데이터가 적다** | 커버리지 부족 / OOD 외삽 | GP 트레이너(§3.1) / MAML few-shot(§3.2) / CP 마진으로 방어(§3.3). 커버리지 보강(§2.3) |
| **속도가 안 나온다** | solve>예산 / 잔차 MLP 과대 / K 과대 | K/N 축소(§4.4), 잔차 hidden 축소, [06편](06_CODE_WALKTHROUGH_CORE.md) 프로파일링 |
| **충돌한다** | 안전 계층 미적용 / margin 부족 | shield 기본 확인(§4.3), CBF margin·CP 마진 상향(§3.3, §4.5), [04편](04_ADVANCED_SAFETY.md) |
| **로봇이 안 움직인다** | 오도메트리/경로 미수신 | [NAV2_INTEGRATION.md](../NAV2_INTEGRATION.md) §Troubleshooting |
| **온라인 학습 후 나빠졌다** | 나쁜 배치로 모델 손상 | `OnlineLearner` 자동 롤백 확인(§3.2), `update_interval` 상향 |
| **갭이 시간에 따라 변한다** | 마모·노면·배터리 | OnlineLearner / EKF·L1·ALPaCA 무학습 추정(§3.2) |

### 7.2 내부 교차 링크

- [09_RESIDUAL_MPC_MPPI.md](09_RESIDUAL_MPC_MPPI.md) — 잔차 이론·코드 (이 문서의 이론 짝)
- [06_CODE_WALKTHROUGH_CORE.md](06_CODE_WALKTHROUGH_CORE.md) — 실시간 파이프라인·성능 최적화
- [04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md) / [08_CODE_WALKTHROUGH_SAFETY.md](08_CODE_WALKTHROUGH_SAFETY.md) — 안전 계층 이론·구현
- [../NAV2_INTEGRATION.md](../NAV2_INTEGRATION.md) — Nav2 FollowPath 서버 상세
- [../TESTING.md](../TESTING.md) — 테스트 카테고리·규칙·성능 기준
- [../SIMULATION_ENVIRONMENTS.md](../SIMULATION_ENVIRONMENTS.md) — 시뮬레이션 시나리오·외란 환경
- [../LEARNING_THEORY.md](../LEARNING_THEORY.md) — 학습 동역학 모델(BNN/GP/Ensemble) 이론

### 7.3 부록 — 외부 자료 (실기/ROS2 배포)

링크는 2026-07 기준 확인:

- [Nav2 공식 문서](https://docs.nav2.org/) — 내비게이션 스택 전반, 컨트롤러 서버·코스트맵·BT.
  DiffDrive/holonomic/car-like 키네마틱 지원.
- [ros2_control 문서](https://control.ros.org/) — 실시간 로봇 제어 프레임워크(하드웨어 인터페이스·
  컨트롤러 매니저). `/cmd_vel` 아래 실제 액추에이터 배선 계층.
- [Nav2 TurtleBot3 튜토리얼](https://docs.nav2.org/getting_started/index.html) — 시뮬→실기 첫걸음.
- [ros2_control diff_drive_controller](https://control.ros.org/rolling/doc/ros2_controllers/diff_drive_controller/doc/userdoc.html) —
  DiffDrive `/cmd_vel` → 휠 명령 표준 컨트롤러.

**시스템 식별·검증 관련 (이 저장소 밖 확장 시)**:

- step-response 피팅으로 `c_v`/`c_ω` 식별: scipy `curve_fit`에 `v̇ = a − c_v·v` 해석해 피팅.
- Conformal Prediction 이론: 03편/09편 부록 및 저장소 `conformal_predictor.py` 주석 참조.

---

> **한 줄 마무리**: 물리 모델은 실측으로 최대한 맞추고, 남은 sim-to-real 갭만 잔차로 학습해
> `ResidualDynamics`로 MPPI에 투명하게 더한다. 보수적 속도·shield·워치독으로 감싸고,
> §5 체크리스트를 위→아래로 통과시킨 뒤, 섀도우 모드 → 저속 실주행 → 점진 상향으로 배포한다.
