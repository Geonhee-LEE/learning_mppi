# MPPI 학습 기반 동역학 모델 아키텍처 가이드

## 목차

1. [전체 아키텍처](#1-전체-아키텍처)
2. [학습 모델 5종 상세](#2-학습-모델-5종-상세)
   - [2.1 NeuralDynamics](#21-neuraldynamics-mlp-신경망)
   - [2.2 GaussianProcessDynamics](#22-gaussianprocessdynamics-가우시안-프로세스)
   - [2.3 EnsembleNeuralDynamics](#23-ensembleneuraldynamics-앙상블-mlp)
   - [2.4 MCDropoutDynamics](#24-mcdropoutdynamics-mc-dropout)
   - [2.5 ResidualDynamics](#25-residualdynamics-하이브리드-잔차-학습)
3. [학습 파이프라인](#3-학습-파이프라인)
   - [3.1 데이터 수집](#31-데이터-수집-datacollector)
   - [3.2 NeuralNetworkTrainer](#32-neuralnetworktrainer)
   - [3.3 GaussianProcessTrainer](#33-gaussianprocesstrainer)
   - [3.4 EnsembleTrainer](#34-ensembletrainer)
4. [온라인 학습](#4-온라인-학습-onlinelearner)
5. [불확실성 활용](#5-불확실성-활용-uncertaintyawarecost)
6. [GPU 가속](#6-gpu-가속)
7. [모델 검증](#7-모델-검증-modelvalidator)
8. [모델 선택 가이드](#8-모델-선택-가이드)
9. [데모 실행법](#9-데모-실행법)
10. [파일 구조 맵](#10-파일-구조-맵)

---

## 1. 전체 아키텍처

### 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────────┐
│                    학습 기반 동역학 모델 시스템                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────────────┐  │
│  │ DataCollector │ ──→ │   Trainer    │ ──→ │   Learned Model    │  │
│  │  (데이터수집)  │     │  (오프라인)   │     │  (동역학 모델)      │  │
│  └──────────────┘     └──────────────┘     └─────────┬──────────┘  │
│                                                       │             │
│                                              ┌────────▼─────────┐  │
│  ┌──────────────┐                            │ MPPI Controller  │  │
│  │OnlineLearner │ ◄──── Simulator ─────────→ │  (제어 계산)      │  │
│  │ (온라인 적응)  │     (자동 데이터 피드)      └──────────────────┘  │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙

모든 학습 모델은 `RobotModel` ABC를 구현합니다:

```python
class RobotModel(ABC):
    state_dim: int       # 상태 차원 (nx)
    control_dim: int     # 제어 차원 (nu)
    model_type: str      # 'kinematic' | 'dynamic' | 'learned'

    def forward_dynamics(state, control) -> state_dot   # dx/dt = f(x, u)
    def step(state, control, dt) -> next_state          # RK4 적분
```

따라서 기존 물리 모델과 **동일한 인터페이스**로 MPPI 컨트롤러에 drop-in 교체 가능합니다:

```python
# 물리 모델 MPPI
controller = MPPIController(DifferentialDriveKinematic(), params)

# 학습 모델 MPPI (코드 변경 없이 모델만 교체)
controller = MPPIController(NeuralDynamics(3, 2, "nn.pth"), params)
```

---

## 2. 학습 모델 5종 상세

### 2.1 NeuralDynamics (MLP 신경망)

**파일**: `mppi_controller/models/learned/neural_dynamics.py`

#### 구조

```
입력: [x, y, θ, v, ω]  ──→  정규화  ──→  MLP  ──→  역정규화  ──→  출력: [ẋ, ẏ, θ̇]
     (state + control)      (z-score)                             (state_dot)
```

```
MLP 내부 구조:
  Linear(nx+nu, 128)  →  ReLU  →  [Dropout]
  Linear(128, 128)    →  ReLU  →  [Dropout]
  Linear(128, nx)
```

#### 주요 API

```python
model = NeuralDynamics(state_dim=3, control_dim=2, model_path="nn_model.pth")

# 기본 예측 (MPPI에서 사용)
state_dot = model.forward_dynamics(state, control)  # (nx,) → (nx,)

# 배치 예측
state_dots = model.forward_dynamics(states, controls)  # (K, nx) → (K, nx)
```

#### 특징
- **Kaiming 초기화**: He initialization으로 안정적 학습 시작
- **입출력 정규화**: z-score 정규화로 학습 안정성 확보 (norm_stats 저장)
- **Early Stopping**: 과적합 방지, 최적 모델 자동 저장
- **GPU 지원**: `TorchNeuralDynamics` 래퍼로 MPPI 루프 내 GPU 추론

#### 장단점
| 장점 | 단점 |
|------|------|
| 빠른 추론 (~0.1ms) | 불확실성 정보 없음 |
| 높은 표현력 | 대량 데이터 필요 |
| GPU 가속 지원 | 외삽(extrapolation) 취약 |

---

### 2.2 GaussianProcessDynamics (가우시안 프로세스)

**파일**: `mppi_controller/models/learned/gaussian_process_dynamics.py`

#### 구조

```
입력: [state, control]  ──→  GP_1  ──→  mean(ẋ), std(ẋ)
                        ──→  GP_2  ──→  mean(ẏ), std(ẏ)    ← 출력 차원별 독립 GP
                        ──→  GP_3  ──→  mean(θ̇), std(θ̇)
```

#### 주요 API

```python
model = GaussianProcessDynamics(state_dim=3, control_dim=2, model_path="gp.pth")

# 평균만 반환 (forward_dynamics 호환)
state_dot = model.forward_dynamics(state, control)  # (nx,)

# 평균 + 불확실성 반환
mean, std = model.predict_with_uncertainty(state, control)  # (nx,), (nx,)

# ARD 커널 길이스케일 확인 (변수 중요도)
lengthscales = model.get_lengthscales()  # List[(input_dim,)]
```

#### 핵심 개념

**커널(Kernel)**: 데이터 포인트 간 유사도를 측정하는 함수
- **RBF**: k(x, x') = exp(-||x-x'||² / 2l²), 가장 일반적
- **Matern**: 더 거친 함수 근사 가능

**ARD (Automatic Relevance Determination)**: 입력 변수별 독립 길이스케일
- 길이스케일이 큰 변수 = 덜 중요, 작은 변수 = 더 중요

**Sparse GP**: 대량 데이터용 근사
- 유도점(inducing points) 100~500개로 N개 데이터 요약
- VariationalELBO 최적화

#### 불확실성의 의미

```
std(ẋ) 가 클 때:
  → 해당 영역에 학습 데이터가 적음
  → 예측이 불확실함
  → MPPI가 해당 영역을 회피해야 함 (UncertaintyAwareCost)
```

#### 장단점
| 장점 | 단점 |
|------|------|
| 해석적 불확실성 추정 | 추론 느림 (~5ms) |
| 적은 데이터로도 학습 가능 | O(N³) 학습 복잡도 |
| 과적합에 강건 | GPU 미지원 (GPyTorch 제약) |
| 외삽 시 불확실성 자동 증가 | 대량 데이터에 비효율 |

---

### 2.3 EnsembleNeuralDynamics (앙상블 MLP)

**파일**: `mppi_controller/models/learned/ensemble_dynamics.py`

#### 구조

```
              ┌── MLP_1(부트스트랩 데이터₁) ──→ pred_1 ─┐
입력 ─────────┤── MLP_2(부트스트랩 데이터₂) ──→ pred_2 ──┤──→ mean = (1/M)Σpred_m
              ├── MLP_3(부트스트랩 데이터₃) ──→ pred_3 ──┤    std  = √Var(pred)
              └── MLP_M(부트스트랩 데이터_M) ──→ pred_M ─┘
```

#### 주요 API

```python
model = EnsembleNeuralDynamics(state_dim=3, control_dim=2, model_path="ensemble.pth")

# 평균 예측
state_dot = model.forward_dynamics(state, control)  # (nx,)

# 평균 + 불확실성
mean, std = model.predict_with_uncertainty(state, control)  # (nx,), (nx,)
```

#### 핵심 개념

**부트스트랩(Bootstrap)**: 각 MLP는 원본 데이터에서 복원추출(replacement)된 서로 다른 데이터셋으로 학습

```python
# EnsembleTrainer 내부
for m in range(M):
    indices = np.random.choice(N_train, N_train, replace=True)  # 복원추출
    train_model_m(data[indices])  # 각 모델이 다른 데이터 분포 학습
```

**불확실성 원리**: M개 모델이 동의하면 std 작음, 의견 분분하면 std 큼

#### 장단점
| 장점 | 단점 |
|------|------|
| NN의 표현력 + 불확실성 | M배 메모리/학습 비용 |
| 병렬 학습 가능 | M배 추론 비용 |
| 구현이 단순 | 불확실성 보정 필요할 수 있음 |

---

### 2.4 MCDropoutDynamics (MC Dropout)

**파일**: `mppi_controller/models/learned/mc_dropout_dynamics.py`

#### 구조

```
                   ┌── forward(mask₁) ──→ pred_1 ─┐
단일 MLP + ────────┤── forward(mask₂) ──→ pred_2 ──┤──→ mean, std
Dropout 활성화     ├── forward(mask₃) ──→ pred_3 ──┤
(model.train())    └── forward(mask_M) ──→ pred_M ─┘
```

#### 주요 API

```python
model = MCDropoutDynamics(state_dim=3, control_dim=2,
                          num_samples=20,        # MC 샘플 수
                          model_path="mc.pth")

mean, std = model.predict_with_uncertainty(state, control)
```

#### 핵심 원리

일반적으로 추론 시에는 dropout을 비활성화하지만,
MC Dropout은 **추론 시에도 dropout을 활성화**하여 베이지안 근사를 수행합니다:

```python
# 핵심 로직
self.model.train()  # dropout 활성화 상태 유지
with torch.no_grad():
    for _ in range(num_samples):
        pred = self.model(inputs)  # 매번 다른 dropout mask
        predictions.append(pred)

mean = torch.stack(predictions).mean(dim=0)
std  = torch.stack(predictions).std(dim=0)
```

**이론적 근거**: Gal & Ghahramani (2016) — Dropout as Bayesian Approximation

#### 장단점
| 장점 | 단점 |
|------|------|
| 모델 1개만 저장 | M회 forward pass 필요 |
| 학습 시 변경 불필요 | dropout_rate > 0 필수 |
| 앙상블 대비 경량 | 불확실성 품질이 dropout_rate에 민감 |

---

### 2.5 ResidualDynamics (하이브리드 잔차 학습)

**파일**: `mppi_controller/models/learned/residual_dynamics.py`

#### 구조

```
                    ┌── 물리 모델: ẋ_physics ────────────┐
state, control ─────┤                                    ├──→ ẋ_total = ẋ_physics + ẋ_residual
                    └── 학습 모델: ẋ_residual (잔차) ────┘
```

#### 주요 API

```python
# 방법 1: learned_model 자동 와이어링 (권장)
physics = DifferentialDriveKinematic()
nn_model = NeuralDynamics(state_dim=3, control_dim=2, model_path="residual.pth")
hybrid = ResidualDynamics(base_model=physics, learned_model=nn_model)

# 방법 2: 수동 잔차 함수
hybrid = ResidualDynamics(
    base_model=physics,
    residual_fn=lambda s, c: np.array([0.01, 0.0, 0.0]),  # 상수 잔차
)

# 사용 (MPPI와 동일 인터페이스)
state_dot = hybrid.forward_dynamics(state, control)

# 잔차 기여도 분석
contribution = hybrid.get_residual_contribution(state, control)
# {'physics_norm': 0.95, 'residual_norm': 0.05, 'ratio': 0.053}

# GP 연결 시 불확실성 자동 추출
uncertainty = hybrid.get_uncertainty(state, control)  # (nx,)
```

#### 핵심 아이디어

물리 모델이 대부분의 동역학을 설명하고, 학습 모델은 **잔차(오차)만 보정**:

```
실제 동역학:  ẋ_true = f_physics(x, u) + f_unmodeled(x, u)
                        ↑ 알려진 부분      ↑ 마찰, 공기저항, 바닥 요철 등

학습 목표:    f_residual ≈ f_unmodeled
```

#### 자동 와이어링 (learned_model 전달 시)

```python
# ResidualDynamics 내부 자동 연결
if learned_model is not None:
    self.residual_fn = learned_model.forward_dynamics

    # GP/Ensemble인 경우 불확실성도 자동 연결
    if hasattr(learned_model, 'predict_with_uncertainty'):
        self.uncertainty_fn = lambda s, c: learned_model.predict_with_uncertainty(s, c)[1]
```

#### 장단점
| 장점 | 단점 |
|------|------|
| 적은 데이터로 학습 (잔차만) | 물리 모델 필요 |
| 물리적 해석 가능 | 물리 모델 정확도에 의존 |
| 데이터 없을 때 물리 모델로 fallback | 구현 복잡도 증가 |
| GP 연결 시 불확실성 자동 지원 | |

---

## 3. 학습 파이프라인

### 3.1 데이터 수집 (DataCollector)

**파일**: `mppi_controller/learning/data_collector.py`

```python
from mppi_controller.learning.data_collector import DataCollector, DynamicsDataset

# 1. 수집기 생성
collector = DataCollector(state_dim=3, control_dim=2, max_samples=100000)

# 2. 시뮬레이션 루프에서 데이터 추가
for step in simulation:
    state_before = simulator.state.copy()
    result = simulator.step(ref)
    collector.add_sample(state_before, result['control'], simulator.state, dt)

# 3. 에피소드 종료 (반드시 호출!)
collector.end_episode()

# 4. 저장/로드
collector.save("training_data.pkl")
collector.load("training_data.pkl")

# 5. 학습용 데이터셋 변환
data = collector.get_data()                 # dict 반환
dataset = DynamicsDataset(data)             # dict를 전달 (DataCollector 객체 아님!)
train_inputs, train_targets = dataset.get_train_data()
val_inputs, val_targets = dataset.get_val_data()
norm_stats = dataset.get_normalization_stats()
```

> **주의**: `DynamicsDataset(data)`에는 `collector.get_data()` 결과(dict)를 전달합니다.
> `DataCollector` 객체를 직접 전달하면 `TypeError`가 발생합니다.
> 또한 `collector.end_episode()`를 호출해야 현재 에피소드의 데이터가 확정됩니다.

**데이터 형식**:

```
수집: (state, control, next_state, dt)
변환: state_dot = (next_state - state) / dt

학습 입력:  inputs  = [state | control]    shape: (N, nx+nu), z-score 정규화
학습 출력:  targets = state_dot            shape: (N, nx), z-score 정규화
정규화:     norm_stats = {state_mean, state_std, control_mean, control_std,
                          state_dot_mean, state_dot_std}
```

### 3.2 NeuralNetworkTrainer

**파일**: `mppi_controller/learning/neural_network_trainer.py`

```python
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

trainer = NeuralNetworkTrainer(
    state_dim=3, control_dim=2,
    hidden_dims=[128, 128],     # MLP 구조
    activation="relu",
    dropout_rate=0.1,           # MC Dropout용
    learning_rate=1e-3,
    weight_decay=1e-5,          # L2 정규화
    device="cpu",
)

# 학습
history = trainer.train(
    train_inputs, train_targets,
    val_inputs, val_targets,
    norm_stats,
    epochs=100,
    batch_size=64,
    early_stopping_patience=20,
)

# 저장 (체크포인트에 포함: model_state_dict, norm_stats, config, history)
trainer.save_model("nn_model.pth")
```

**학습 알고리즘**:

```
1. Adam optimizer (lr=1e-3, weight_decay=1e-5)
2. MSE Loss: L = (1/N) Σ ||NN(x,u) - ẋ_true||²
3. ReduceLROnPlateau: val_loss 정체 시 lr × 0.5
4. Early Stopping: val_loss 20 epoch 연속 개선 없으면 중단
5. Best Model Checkpoint: val_loss 최소 모델 자동 저장
```

### 3.3 GaussianProcessTrainer

**파일**: `mppi_controller/learning/gaussian_process_trainer.py`

```python
from mppi_controller.learning.gaussian_process_trainer import GaussianProcessTrainer

trainer = GaussianProcessTrainer(
    state_dim=3, control_dim=2,
    kernel_type="rbf",          # "rbf" 또는 "matern"
    use_sparse=False,           # True: Sparse GP (대량 데이터용)
    num_inducing_points=100,    # Sparse GP 유도점 수
    use_ard=True,               # 변수별 길이스케일
)

history = trainer.train(
    train_inputs, train_targets,
    val_inputs, val_targets,
    norm_stats,
    num_iterations=100,
    learning_rate=0.1,
)

trainer.save_model("gp_model.pth")
```

**학습 알고리즘**:

```
출력 차원별 독립 GP 학습 (nx개):
  1. GP_d 생성 (ExactGP 또는 SparseGP)
  2. Likelihood 생성 (GaussianLikelihood)
  3. MLL 최대화: max θ  log p(y_d | X, θ)
     - ExactGP:  ExactMarginalLogLikelihood
     - SparseGP: VariationalELBO
  4. Adam optimizer로 커널 파라미터 최적화
```

### 3.4 EnsembleTrainer

**파일**: `mppi_controller/learning/ensemble_trainer.py`

```python
from mppi_controller.learning.ensemble_trainer import EnsembleTrainer

trainer = EnsembleTrainer(
    state_dim=3, control_dim=2,
    num_models=5,               # 앙상블 크기
    hidden_dims=[128, 128],
    bootstrap=True,             # 부트스트랩 복원추출
)

history = trainer.train(
    train_inputs, train_targets,
    val_inputs, val_targets,
    norm_stats,
    epochs=100,
    batch_size=64,
    bootstrap=True,
)

trainer.save_model("ensemble_model.pth")
```

---

## 4. 온라인 학습 (OnlineLearner)

**파일**: `mppi_controller/learning/online_learner.py`

실시간으로 데이터를 수집하고 모델을 적응시키는 시스템입니다.

### 동작 흐름

```
┌──────────────────────────────────────────────────────────────────┐
│                   OnlineLearner 동작 흐름                          │
│                                                                  │
│  Simulator.step()                                                │
│       │                                                          │
│       ├──→ MPPI.compute_control()                                │
│       │                                                          │
│       └──→ OnlineLearner.add_sample(state, ctrl, next_state, dt) │
│                  │                                                │
│                  ▼                                                │
│            OnlineDataBuffer (원형 버퍼, 기본 1000개)                │
│                  │                                                │
│                  │ 트리거 조건: samples ≥ 100 AND count % 500 == 0 │
│                  ▼                                                │
│            update_model()                                         │
│              ├── 데이터 80/20 셔플 분할                            │
│              ├── trainer.train(epochs=10)  ← 소량 fine-tuning     │
│              ├── 체크포인트 저장 (model_v{N}.pth)                  │
│              └── 성능 감시                                        │
│                    │                                              │
│                    └── val_loss > 1.5 × best?                    │
│                         ├── Yes → 자동 롤백 (이전 버전 복원)       │
│                         └── No  → 계속                           │
└──────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
from mppi_controller.learning.online_learner import OnlineLearner

# 사전 학습된 모델 + 트레이너 준비
model = NeuralDynamics(state_dim=3, control_dim=2, model_path="pretrained.pth")
trainer = NeuralNetworkTrainer(state_dim=3, control_dim=2)
trainer.load_model("pretrained.pth")

# 온라인 학습기 생성
learner = OnlineLearner(
    model=model,
    trainer=trainer,
    buffer_size=1000,
    min_samples_for_update=100,
    update_interval=500,
    checkpoint_dir="checkpoints/",
    max_checkpoints=10,
)

# Simulator와 통합 (자동 데이터 피드)
simulator = Simulator(model, controller, dt=0.05, online_learner=learner)

# 시뮬레이션 실행 → 자동으로 데이터 수집 + 모델 업데이트
simulator.run(ref_fn, duration=30.0)
```

### 체크포인트 관리

```python
# 수동 롤백
learner.rollback(version=3)     # 특정 버전으로 복원
learner.rollback()              # 최신 체크포인트로 복원

# 적응 모니터링
learner.monitor_adaptation(test_data)
# 출력: "Adaptation: initial_error=0.05 → current_error=0.02, improvement=60.0%"
```

---

## 5. 불확실성 활용 (UncertaintyAwareCost)

**파일**: `mppi_controller/controllers/mppi/uncertainty_cost.py`

GP / Ensemble / MCDropout 모델의 불확실성을 MPPI 비용함수에 통합하여
**불확실한 영역을 자동 회피**하는 보수적 제어를 구현합니다.

### 비용 공식

```
J_uncertainty = β × Σ_{t=0}^{N-1} reduce(σ_t²)

  β (beta): 불확실성 페널티 가중치 (클수록 보수적)
  σ_t:      시간 t에서의 예측 불확실성 (standard deviation)
  reduce:   "sum" | "max" | "mean"
```

### 사용법

```python
from mppi_controller.controllers.mppi.uncertainty_cost import UncertaintyAwareCost
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost, StateTrackingCost, ControlEffortCost,
)

# GP 모델 준비
gp_model = GaussianProcessDynamics(3, 2, "gp.pth")

# 불확실성 비용 생성
unc_cost = UncertaintyAwareCost(
    uncertainty_fn=lambda s, c: gp_model.predict_with_uncertainty(s, c)[1],
    beta=10.0,       # 페널티 강도
    reduce="sum",    # 차원별 합산
)

# 복합 비용에 추가
composite = CompositeMPPICost([
    StateTrackingCost(Q),
    ControlEffortCost(R),
    unc_cost,             # 불확실성 페널티 추가
])

controller = MPPIController(gp_model, params, composite)
```

### 효과

```
불확실성 높은 영역:
  σ 큼 → J_uncertainty 큼 → 해당 궤적 가중치↓ → MPPI가 회피

불확실성 낮은 영역:
  σ 작음 → J_uncertainty 작음 → 정상 추종 동작
```

---

## 6. GPU 가속

**파일**: `mppi_controller/controllers/mppi/gpu/torch_learned.py`

NeuralDynamics를 GPU에서 실행할 때, numpy↔torch 변환 오버헤드를 제거합니다.

### 자동 활성화

```python
params = MPPIParams(K=4096, N=30, device="cuda")
model = NeuralDynamics(state_dim=3, control_dim=2, model_path="nn.pth")
controller = MPPIController(model, params)
# → 내부적으로 TorchNeuralDynamics 자동 래핑
# → 전체 MPPI 파이프라인이 GPU에서 실행
```

### 성능

```
RTX 5080 기준:
  K=1024  CPU: ~5ms   GPU: ~4ms   (1.3x)
  K=4096  CPU: ~18ms  GPU: ~4ms   (4.3x)
  K=8192  CPU: ~35ms  GPU: ~4ms   (8.1x)  ← GPU는 K에 거의 무관
```

---

## 7. 모델 검증 (ModelValidator)

**파일**: `mppi_controller/learning/model_validator.py`

```python
from mppi_controller.learning.model_validator import ModelValidator

validator = ModelValidator()

# 단일 모델 평가
metrics = validator.evaluate(
    predict_fn=model.forward_dynamics,
    test_states=test_states,       # (N, nx)
    test_controls=test_controls,   # (N, nu)
    test_targets=test_targets,     # (N, nx)
)
# metrics: {rmse, mae, r2, per_dim_rmse, per_dim_mae, per_dim_r2, max_error}

# 다중 모델 비교
results = validator.compare(
    models={
        "Neural": nn_model.forward_dynamics,
        "GP": gp_model.forward_dynamics,
        "Ensemble": ensemble_model.forward_dynamics,
    },
    test_states, test_controls, test_targets,
)
validator.print_comparison(results)

# 롤아웃 정확도 평가
rollout_metrics = validator.evaluate_rollout(
    model=model,
    initial_states=init_states,       # (M, nx)
    control_sequences=ctrl_seqs,      # (M, T, nu)
    true_trajectories=true_trajs,     # (M, T+1, nx)
    dt=0.05,
)
```

---

## 8. 모델 선택 가이드

### 비교표

| 모델 | 불확실성 | 데이터 요구량 | 추론 속도 | GPU | 메모리 | 적합 상황 |
|------|:------:|:----------:|:-------:|:---:|:-----:|----------|
| **NeuralDynamics** | - | 많음 (>1000) | ~0.1ms | O | 낮음 | 대량 데이터, 빠른 추론 필요 |
| **GP** | 해석적 | 적음 (~200) | ~5ms | - | 높음 | 소량 데이터, 안전 최우선 |
| **Ensemble** | 분산 | 많음 (>1000) | ~0.5ms×M | - | M배 | NN + 불확실성 모두 필요 |
| **MCDropout** | MC | 많음 (>1000) | ~0.1ms×M | - | 낮음 | 저장 효율 + 불확실성 |
| **Residual** | 선택적 | 적음 (~100) | 물리+α | 부분 | 낮음 | 물리 모델 보정 |

### 결정 흐름도

```
데이터가 충분한가? (>1000 샘플)
├── Yes
│   ├── 불확실성이 필요한가?
│   │   ├── Yes → Ensemble 또는 MCDropout
│   │   └── No  → NeuralDynamics (+ GPU 가속)
│   │
│   └── 물리 모델이 있는가?
│       ├── Yes → ResidualDynamics + NN
│       └── No  → NeuralDynamics
│
└── No (< 500 샘플)
    ├── 불확실성이 필요한가?
    │   ├── Yes → GP (최선)
    │   └── No  → ResidualDynamics (물리 모델 보정)
    │
    └── 물리 모델 보정만 원하는가?
        └── Yes → ResidualDynamics (constant/state-dependent)
```

---

## 9. 데모 실행법

### 사전 준비

```bash
cd /path/to/learning_mppi

# 필수 패키지
pip install torch matplotlib

# GP 모델 사용 시 추가 (gp_vs_neural_comparison_demo 등)
pip install gpytorch

# PYTHONPATH 설정 (또는 PYTHONPATH=. 접두어 사용)
export PYTHONPATH=$PWD:$PYTHONPATH
```

> **참고**: GP 관련 기능(`GaussianProcessDynamics`, `GaussianProcessTrainer`,
> `gp_vs_neural_comparison_demo.py`)은 `gpytorch` 패키지가 필요합니다.
> NN/Ensemble/MCDropout/Residual은 `torch`만으로 사용 가능합니다.

### 9.1 Neural Dynamics 전체 파이프라인

데이터 수집 → NN 학습 → Kinematic vs Neural vs Residual 3종 비교

```bash
# 기본 실행 (원형 궤적, 30초 데이터)
PYTHONPATH=. python examples/learned/neural_dynamics_learning_demo.py

# 8자 궤적, 60초 데이터, 200 에폭
PYTHONPATH=. python examples/learned/neural_dynamics_learning_demo.py \
    --trajectory figure8 --duration 60 --epochs 200

# 옵션:
#   --trajectory {circle, figure8, sine, straight}
#   --duration   데이터 수집 시간 (초)
#   --epochs     학습 에폭 수
```

**출력**: 학습 곡선 + 3종 MPPI 추종 비교 플롯

### 9.2 GP vs Neural Network 비교

데이터 효율, 불확실성 보정, MPPI 제어 성능 비교

```bash
# 기본 실행
PYTHONPATH=. python examples/learned/gp_vs_neural_comparison_demo.py

# 데이터 일부만 사용 (데이터 효율 비교)
PYTHONPATH=. python examples/learned/gp_vs_neural_comparison_demo.py \
    --data-fraction 0.3 --trajectory circle

# 옵션:
#   --trajectory     궤적 타입
#   --duration       데이터 수집 시간
#   --data-fraction  학습 데이터 비율 (0.1~1.0)
```

**출력**: 12-subplot 비교 (학습 손실, 불확실성 보정, MPPI 성능, 궤적, 오차)

### 9.3 Residual Dynamics 데모

물리+잔차 하이브리드 모델 시연

```bash
# 상수 잔차
PYTHONPATH=. python examples/learned/mppi_residual_dynamics_demo.py \
    --residual constant --trajectory circle

# 상태 의존 잔차
PYTHONPATH=. python examples/learned/mppi_residual_dynamics_demo.py \
    --residual state_dependent --trajectory figure8

# 제어 의존 잔차
PYTHONPATH=. python examples/learned/mppi_residual_dynamics_demo.py \
    --residual control_dependent

# 라이브 애니메이션
PYTHONPATH=. python examples/learned/mppi_residual_dynamics_demo.py --live

# 옵션:
#   --residual  {constant, state_dependent, control_dependent, none}
#   --live      실시간 애니메이션
```

### 9.4 온라인 학습 데모

초기 학습 → 도메인 변화 → 온라인 적응 시연

```bash
# 기본 실행 (도메인 변화: 마찰 증가 + 액추에이터 바이어스)
PYTHONPATH=. python examples/learned/online_learning_demo.py

# 옵션:
#   --trajectory   궤적 타입
#   --duration     온라인 학습 시간
```

**출력**: 8-subplot (예측 오차 변화, 추종 오차, 업데이트 주기, 버퍼 크기 등)

### 9.5 물리 vs 학습 모델 3-way 비교

Kinematic vs Dynamic vs Residual MPPI 비교

```bash
# 기본 실행
PYTHONPATH=. python examples/comparison/physics_vs_learned_demo.py

# 8자 궤적, 30초
PYTHONPATH=. python examples/comparison/physics_vs_learned_demo.py \
    --trajectory figure8 --duration 30

# 옵션:
#   --trajectory {circle, figure8, sine, straight}
#   --duration   시뮬레이션 시간
#   --seed       랜덤 시드
```

**출력**: 6-subplot (XY 궤적, 위치 오차, RMSE 바차트, 제어 입력, 계산 시간, 메트릭 요약)

### 데모 요약표

| 데모 | 비교 대상 | 핵심 관측 포인트 | 소요 시간 |
|------|----------|----------------|----------|
| neural_dynamics_learning_demo | Kinematic vs NN vs Residual | 학습 효과, 잔차 기여도 | ~2분 |
| gp_vs_neural_comparison_demo | GP vs NN | 불확실성 보정, 데이터 효율 | ~3분 |
| mppi_residual_dynamics_demo | 잔차 타입 3종 | 물리 모델 보정 효과 | ~1분 |
| online_learning_demo | 적응 전 vs 후 | 도메인 변화 대응 | ~3분 |
| physics_vs_learned_demo | Kinematic vs Dynamic vs Residual | 모델 정확도 vs 계산 비용 | ~1분 |
| 6dof_learned_benchmark | 8-Way 학습 모델 비교 | 6-DOF EE 추적 (NN/GP/Ensemble/MCDrop/MAML/ALPaCA) | ~2분 |

---

## 10. 파일 구조 맵

```
learning_mppi/
├── mppi_controller/
│   ├── models/
│   │   ├── base_model.py                          # RobotModel ABC
│   │   ├── learned/
│   │   │   ├── neural_dynamics.py                 # MLP 동역학
│   │   │   ├── gaussian_process_dynamics.py        # GP 동역학
│   │   │   ├── ensemble_dynamics.py                # 앙상블 MLP
│   │   │   ├── mc_dropout_dynamics.py              # MC Dropout
│   │   │   └── residual_dynamics.py                # 하이브리드 잔차
│   │   ├── kinematic/                             # 기구학 모델
│   │   └── dynamic/                               # 동역학 모델
│   │
│   ├── learning/
│   │   ├── data_collector.py                       # 데이터 수집기
│   │   ├── neural_network_trainer.py               # NN 학습기
│   │   ├── gaussian_process_trainer.py             # GP 학습기
│   │   ├── ensemble_trainer.py                     # 앙상블 학습기
│   │   ├── maml_trainer.py                         # MAML 메타 학습기
│   │   ├── alpaca_trainer.py                       # ALPaCA 메타 학습기
│   │   ├── online_learner.py                       # 온라인 학습기
│   │   └── model_validator.py                      # 모델 검증기
│   │
│   ├── controllers/mppi/
│   │   ├── base_mppi.py                           # MPPI (GPU 자동 전환)
│   │   ├── uncertainty_cost.py                     # 불확실성 비용
│   │   └── gpu/
│   │       ├── torch_learned.py                    # TorchNeuralDynamics
│   │       └── __init__.py                         # get_torch_model()
│   │
│   └── simulation/
│       └── simulator.py                            # online_learner 통합
│
├── examples/
│   ├── learned/
│   │   ├── neural_dynamics_learning_demo.py        # NN 전체 파이프라인
│   │   ├── gp_vs_neural_comparison_demo.py         # GP vs NN 비교
│   │   ├── mppi_residual_dynamics_demo.py          # 잔차 모델 데모
│   │   └── online_learning_demo.py                 # 온라인 학습 데모
│   └── comparison/
│       ├── physics_vs_learned_demo.py              # 3-way 비교
│       └── 6dof_learned_benchmark.py               # 6-DOF 학습 모델 8-Way 벤치마크
│
├── docs/learned_models/
│   ├── LEARNED_MODELS_GUIDE.md                     # 영문 종합 가이드
│   ├── ONLINE_LEARNING.md                          # 온라인 학습 가이드
│   └── LEARNED_MODELS_ARCHITECTURE_KR.md           # 본 문서 (한국어)
│
├── scripts/
│   └── train_6dof_all_models.py                    # 6종 모델 통합 학습
│
└── tests/
    ├── test_neural_dynamics.py                     # NN 테스트 (8)
    ├── test_gaussian_process_dynamics.py            # GP 테스트 (9)
    ├── test_trainers.py                            # 학습기 테스트 (13)
    ├── test_data_pipeline.py                       # 데이터 파이프라인 (15)
    ├── test_online_learner.py                      # 온라인 학습 (17)
    ├── test_ensemble_validator_uncertainty.py       # 앙상블/검증/불확실성 (14)
    ├── test_mc_dropout_checkpoint.py               # MCDropout/체크포인트 (17)
    └── test_6dof_learned_benchmark.py              # 6-DOF 8-Way 벤치마크 (18)
```
