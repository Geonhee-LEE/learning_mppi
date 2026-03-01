# MPPI ROS2 학습 모델 종합 가이드

## 목차

1. [개요](#개요)
2. [학습 모델 타입](#학습-모델-타입)
3. [데이터 수집 및 전처리](#데이터-수집-및-전처리)
4. [모델별 학습 방법](#모델별-학습-방법)
5. [성능 비교 및 선택 가이드](#성능-비교-및-선택-가이드)
6. [온라인 학습 및 적응](#온라인-학습-및-적응)
7. [MPPI 통합](#mppi-통합)
8. [고급 기능](#고급-기능)
9. [베스트 프랙티스](#베스트-프랙티스)
10. [API 레퍼런스](#api-레퍼런스)

---

## 개요

MPPI ROS2는 **9가지 학습/적응 기반 동역학 모델**을 지원합니다:

1. **Neural Dynamics**: 심층 신경망 기반 end-to-end 학습
2. **Gaussian Process Dynamics**: 베이지안 비모수 모델, 불확실성 정량화
3. **Residual Dynamics**: 물리 모델 + 학습 보정 (하이브리드)
4. **Ensemble Neural Dynamics**: M개 MLP 앙상블, 분산 기반 불확실성
5. **MC-Dropout Dynamics**: 추론 시 dropout으로 베이지안 근사
6. **MAML Dynamics**: 메타 학습 기반 few-shot 실시간 적응
7. **EKF Adaptive Dynamics**: 확장 칼만 필터 기반 파라미터 실시간 추정
8. **L1 Adaptive Dynamics**: L1 적응 제어 기반 외란 추정+보상
9. **ALPaCA Dynamics**: Bayesian last-layer 기반 closed-form 적응

### 핵심 특징

- **통일된 인터페이스**: 모든 학습 모델이 `RobotModel` 베이스 클래스 구현
- **MPPI 호환성**: 기구학/동역학 모델과 동일한 방식으로 사용 가능
- **온라인 학습**: 실시간 데이터 수집 및 모델 업데이트 (체크포인트 버전 관리 포함)
- **불확실성 정량화**: GP / 앙상블 / MC-Dropout / ALPaCA 기반 불확실성 추정
- **불확실성 인식 비용**: `UncertaintyAwareCost`로 Risk-Aware MPPI 연동
- **모델 검증**: `ModelValidator`로 RMSE/MAE/R² 통합 평가
- **Sim-to-Real 전이**: 도메인 적응 지원
- **메타 학습**: MAML / ALPaCA 기반 few-shot 실시간 적응 ([META_LEARNING.md](./META_LEARNING.md) 참조)
- **적응 제어**: EKF / L1 기반 학습 없는 실시간 적응
- **GPU 가속**: `TorchNeuralDynamics`로 MPPI 루프 내 GPU 추론

---

## 학습 모델 타입

### 1. Neural Dynamics

**특징**:
- PyTorch 기반 Multi-Layer Perceptron (MLP)
- End-to-end 학습: (state, control) → state_dot
- 빠른 추론 속도 (~0.1ms)
- 대규모 데이터에 적합

**장점**:
- 높은 표현력 (복잡한 비선형 동역학)
- GPU 가속 가능
- 학습 데이터가 충분하면 높은 정확도

**단점**:
- 불확실성 정량화 어려움 (별도 앙상블 필요)
- 데이터 효율성 낮음 (GP 대비)
- 외삽 영역에서 신뢰도 낮음

**사용 사례**:
- 고속 제어 (~100Hz)
- 충분한 학습 데이터 확보 가능
- 불확실성보다 정확도 우선

### 2. Gaussian Process Dynamics

**특징**:
- GPyTorch 기반 베이지안 비모수 모델
- RBF/Matern 커널 + ARD (Automatic Relevance Determination)
- 평균 + 표준편차 예측 (불확실성)
- Exact GP (소규모) / Sparse GP (대규모)

**장점**:
- 불확실성 정량화 (1σ: 68%, 2σ: 95% 신뢰구간)
- 데이터 효율성 높음 (소량 데이터로 학습)
- 외삽 영역에서 불확실성 자동 증가
- Feature importance (ARD lengthscales)

**단점**:
- 추론 속도 느림 (Exact GP: O(N²), Sparse GP: O(NM))
- 대규모 데이터 처리 어려움 (N > 10,000)
- GPU 가속 제한적

**사용 사례**:
- 안전이 중요한 응용 (불확실성 고려)
- 데이터 수집 비용 높음
- 저속 제어 (~10Hz)
- 능동 학습 (불확실성 기반 데이터 수집)

### 3. Residual Dynamics

**특징**:
- 물리 모델 + 학습 보정의 하이브리드
- f_total = f_physics + f_learned
- f_learned는 Neural 또는 GP

**장점**:
- 물리 모델의 구조적 지식 활용
- 학습 부담 감소 (잔차만 학습)
- 외삽 시 물리 모델로 fallback
- 데이터 효율성 높음

**단점**:
- 물리 모델이 필요 (모델링 비용)
- 물리 모델 오차가 크면 효과 감소

**사용 사례**:
- 물리 모델이 존재하지만 불완전함
- 마찰, 공기저항 등 모델링 어려운 항
- Sim-to-Real 전이 (시뮬레이터 모델 보정)

### 4. Ensemble Neural Dynamics

**특징**:
- M개의 독립 학습된 MLP 앙상블
- 앙상블 평균: 예측, 앙상블 분산: 불확실성
- 부트스트랩 학습 (각 멤버마다 리샘플링)

**장점**:
- NN에서도 불확실성 정량화 가능
- GP보다 빠른 추론 (O(1) vs O(N²))
- Risk-Aware MPPI 연동 가능

**단점**:
- M배 메모리/학습 비용
- MC-Dropout 대비 무거움

**사용 사례**:
- 불확실성이 필요하지만 GP 추론 속도가 부족한 경우
- UncertaintyAwareCost와 연동
- 고속 제어 (~50Hz) + 불확실성

```python
from mppi_controller.models.learned.ensemble_dynamics import EnsembleNeuralDynamics

model = EnsembleNeuralDynamics(
    state_dim=3, control_dim=2,
    model_path="models/learned_models/ensemble.pth",
)
mean, std = model.predict_with_uncertainty(state, control)
```

### 5. MC-Dropout Dynamics

**특징**:
- 단일 MLP에서 추론 시 dropout 활성화
- M회 forward pass의 분산으로 불확실성 추정
- Gal & Ghahramani (2016) 의 Bayesian 근사

**장점**:
- 앙상블 대비 경량 (단일 모델)
- 학습 비용 1회 (앙상블은 M회)
- dropout_rate로 불확실성 크기 조절

**단점**:
- 학습 시 dropout_rate > 0 필수
- 앙상블보다 불확실성 품질 낮을 수 있음

**사용 사례**:
- 경량 불확실성 추정
- 메모리 제약 환경
- 기존 dropout 모델 재활용

```python
from mppi_controller.models.learned.mc_dropout_dynamics import MCDropoutDynamics

model = MCDropoutDynamics(
    state_dim=3, control_dim=2,
    model_path="models/learned_models/best_model.pth",
    num_samples=20,
)
mean, std = model.predict_with_uncertainty(state, control)
```

### 6. MAML Dynamics

**특징**:
- FOMAML (First-Order MAML) 기반 메타 학습
- 다양한 환경에서 사전 학습된 메타 파라미터
- 실행 중 few-shot (20~50개 데이터) 적응
- 매 적응마다 메타 파라미터에서 시작 (안정적)

**장점**:
- 빠른 적응 (SGD 100 step, ~10ms)
- 환경 변화에 즉시 대응 (마찰, 관성, 하중 등)
- 누적 드리프트 없음 (매번 메타 파라미터 기준)
- 3D 상태만으로 5D 환경 적응 가능

**단점**:
- 메타 학습에 다양한 환경 시뮬레이션 필요
- 메타 학습 시간 (~5분)
- 태스크 분포 밖 환경에서는 적응 한계

**사용 사례**:
- 환경이 자주 변하는 로봇 (다른 바닥, 하중 변화 등)
- Sim-to-Real 빠른 적응
- 오프라인 Neural/Residual이 성능 부족한 경우

```python
from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

# Residual MAML: 기구학 base + MAML 잔차 (권장 아키텍처)
maml = MAMLDynamics(
    state_dim=3, control_dim=2,
    model_path="models/learned_models/dynamic_maml_meta_model.pth",
    inner_lr=0.005, inner_steps=100,
)
maml.save_meta_weights()

# 잔차 적응 후 ResidualDynamics로 사용
maml.adapt(states, controls, residual_targets, dt=0.05, restore=True)
residual_model = ResidualDynamics(base_model=kinematic, learned_model=maml, use_residual=True)
```

자세한 내용은 [META_LEARNING.md](./META_LEARNING.md)를 참조하세요.

### 7. EKF Adaptive Dynamics

**특징**:
- 확장 칼만 필터(EKF)로 물리 파라미터 (c_v, c_omega) 실시간 추정
- 7D 확장 상태: `[x, y, θ, v, ω, ĉ_v, ĉ_ω]` (5D 관측 + 2D 파라미터)
- DynamicKinematicAdapter를 내부 모델로 사용, 추정 파라미터로 갱신
- **오프라인 학습 불필요** — 즉시 동작

**장점**:
- 학습 데이터 불필요 (메타 학습/사전 학습 없음)
- 파라미터 물리적 의미 유지 (해석 가능)
- 추정 불확실성 정량화 (공분산 행렬)
- 파라미터 불일치에 매우 강함

**단점**:
- 비모델링 외란 (wind, sine)에는 적응 제한
- 해석적 야코비안 필요 (새 모델마다 유도)
- 파라미터 수에 비례하여 상태 차원 증가

**사용 사례**:
- 마찰/관성 변화가 주된 모델 불확실성
- 학습 인프라 없이 빠른 배포 필요
- 파라미터 모니터링이 중요한 진단 응용

```python
from mppi_controller.models.learned.ekf_dynamics import EKFAdaptiveDynamics

model = EKFAdaptiveDynamics(state_dim=5, control_dim=2)

# 매 타임스텝 업데이트
model.update_step(state_5d, control, next_state_5d, dt=0.05)

# 추정된 파라미터 확인
params = model.get_parameter_estimates()
print(f"c_v={params['c_v']:.3f}, c_omega={params['c_omega']:.3f}")
```

### 8. L1 Adaptive Dynamics

**특징**:
- L1 적응 제어: 상태 예측기 + 외란 추정 + 저역통과 필터
- 연속 외란 σ(t)를 실시간 추정하여 모델 보정
- Hurwitz 안정 행렬 A_m으로 예측 오차 수렴 보장
- **오프라인 학습 불필요** — 즉시 동작

**장점**:
- 시변 외란 (wind, periodic force) 추적에 강함
- 수학적 안정성 보장 (Hurwitz A_m)
- 학습 데이터 불필요
- 저역통과 필터로 고주파 노이즈 제거

**단점**:
- 파라미터 직접 추정 불가 (EKF 대비)
- 필터 대역폭(ω_c)이 추적 성능과 노이즈 억제 간 trade-off
- 급격한 외란 변화 시 응답 지연

**사용 사례**:
- 시간에 따라 변하는 외란이 주된 불확실성
- 풍하중, 경사면 등 외부 힘 보상
- 빠른 배포 (학습 불필요)

```python
from mppi_controller.models.learned.l1_adaptive_dynamics import L1AdaptiveDynamics

model = L1AdaptiveDynamics(state_dim=5, control_dim=2)

# 매 타임스텝 업데이트
model.update_step(state_5d, control, next_state_5d, dt=0.05)

# forward_dynamics는 f_nom + σ_filtered 반환 (MPPI용 보정 모델)
state_dot = model.forward_dynamics(state, control)
```

### 9. ALPaCA Dynamics

**특징**:
- 메타 학습된 feature extractor (frozen MLP) + Bayesian linear regression
- Closed-form 적응: SGD 없이 행렬 연산으로 즉시 적응
- 예측 불확실성 정량화 (posterior predictive variance)
- ResidualDynamics(base + ALPaCA) 패턴으로 사용

**장점**:
- SGD 없는 closed-form 적응 (MAML보다 빠르고 안정적)
- 불확실성 정량화 (GP와 유사한 Bayesian 예측)
- 적응 품질이 데이터양에 단조 증가 (과적합 없음)
- 메타 학습 후 feature extractor 고정 → 경량 적응

**단점**:
- 메타 학습 필요 (~5분)
- Linear last layer의 표현력 한계 (highly nonlinear 잔차에 부적합)
- MAML 대비 적응 유연성 제한

**사용 사례**:
- 빠르고 안정적인 적응이 필요한 실시간 시스템
- 불확실성이 필요한 안전 크리티컬 응용
- SGD 기반 적응(MAML)이 불안정한 환경

```python
from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

# 메타 학습된 모델 로드
alpaca = ALPaCADynamics(
    state_dim=5, control_dim=2,
    model_path="models/learned_models/dynamic_alpaca_meta_model.pth",
)

# Closed-form 적응 (SGD 없음!)
loss = alpaca.adapt(states, controls, next_states, dt=0.05)

# 불확실성 포함 예측
uncertainty = alpaca.get_uncertainty(state, control)

# Residual 모델로 사용 (권장)
residual_model = ResidualDynamics(base_model=adapter_5d, learned_model=alpaca, use_residual=True)
```

자세한 내용은 [META_LEARNING.md](./META_LEARNING.md)를 참조하세요.

---

## 데이터 수집 및 전처리

### DataCollector

실제 로봇 또는 시뮬레이터에서 동역학 데이터를 수집합니다.

```python
from mppi_controller.learning.data_collector import DataCollector

collector = DataCollector(
    state_dim=3,      # [x, y, θ]
    control_dim=2,    # [v, ω]
)

# Episode 시작
collector.start_episode()

# 제어 루프
for t in range(num_steps):
    state = get_current_state()
    control = controller.compute_control(state, ref_trajectory)

    # 제어 적용
    apply_control(control)
    time.sleep(dt)

    next_state = get_current_state()

    # 샘플 추가
    collector.add_sample(state, control, next_state, dt)

# Episode 종료
collector.end_episode()

# 데이터 추출
data = collector.get_data()
# Returns: {"states": (N, nx), "controls": (N, nu),
#           "next_states": (N, nx), "state_dots": (N, nx), "dt": (N,)}
```

### DynamicsDataset

학습을 위한 데이터 전처리 및 분할.

```python
from mppi_controller.learning.data_collector import DynamicsDataset

dataset = DynamicsDataset(
    data=data,
    train_ratio=0.8,     # 80% 학습, 20% 검증
    normalize=True,      # 자동 정규화
)

train_inputs, train_targets = dataset.get_train_data()
val_inputs, val_targets = dataset.get_val_data()

# 정규화 통계 (모델 저장 시 필요)
norm_stats = dataset.get_normalization_stats()
# Returns: {"state_mean", "state_std", "control_mean", "control_std",
#           "state_dot_mean", "state_dot_std"}
```

**정규화 중요성**:
- 신경망: 학습 안정성 및 수렴 속도 향상
- GP: 커널 최적화 효율성

---

## 모델별 학습 방법

### Neural Dynamics 학습

#### 1. 데이터 수집

```python
# 다양한 궤적에서 데이터 수집 (최소 5000 샘플 권장)
collector = DataCollector(state_dim=3, control_dim=2)

for episode in range(50):
    collector.start_episode()

    # 랜덤 레퍼런스 궤적 생성
    ref_fn = generate_random_trajectory()

    # 데이터 수집 (30초)
    collect_episode(ref_fn, duration=30.0, dt=0.05)

    collector.end_episode()

data = collector.get_data()
dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
```

#### 2. 모델 학습

```python
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

trainer = NeuralNetworkTrainer(
    state_dim=3,
    control_dim=2,
    hidden_dims=[128, 128, 128],  # 3-layer MLP
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 학습 실행
history = trainer.train(
    train_inputs=dataset.train_inputs,
    train_targets=dataset.train_targets,
    val_inputs=dataset.val_inputs,
    val_targets=dataset.val_targets,
    norm_stats=dataset.get_normalization_stats(),
    epochs=200,
    batch_size=256,
    early_stopping_patience=20,
    verbose=True,
)

# 모델 저장
trainer.save_model("neural_model.pth")
```

#### 3. 모델 로드 및 사용

```python
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics

model = NeuralDynamics(
    state_dim=3,
    control_dim=2,
    model_path="neural_model.pth",
    device="cpu",
)

# MPPI와 통합
from mppi_controller.controllers.mppi.base_mppi import MPPIController

controller = MPPIController(model=model, params=params)
```

### Gaussian Process 학습

#### 1. 데이터 수집

```python
# GP는 데이터 효율적 (1000-2000 샘플로 충분)
collector = DataCollector(state_dim=3, control_dim=2)

for episode in range(10):
    collector.start_episode()
    collect_episode(ref_fn, duration=30.0, dt=0.05)
    collector.end_episode()

data = collector.get_data()
dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
```

#### 2. 모델 학습

```python
from mppi_controller.learning.gaussian_process_trainer import GaussianProcessTrainer

trainer = GaussianProcessTrainer(
    state_dim=3,
    control_dim=2,
    kernel_type="rbf",        # "rbf" or "matern"
    use_ard=True,             # Automatic Relevance Determination
    use_sparse_gp=False,      # Exact GP (소규모 데이터)
    num_inducing_points=500,  # Sparse GP 사용 시
)

# 학습 실행
history = trainer.train(
    train_inputs=dataset.train_inputs,
    train_targets=dataset.train_targets,
    val_inputs=dataset.val_inputs,
    val_targets=dataset.val_targets,
    norm_stats=dataset.get_normalization_stats(),
    num_iterations=100,
    verbose=True,
)

# 모델 저장
trainer.save_model("gp_model.pth")
```

#### 3. 모델 로드 및 사용

```python
from mppi_controller.models.learned.gaussian_process_dynamics import GaussianProcessDynamics

model = GaussianProcessDynamics(
    state_dim=3,
    control_dim=2,
    model_path="gp_model.pth",
)

# 불확실성 포함 예측
state = np.array([0.0, 0.0, 0.0])
control = np.array([1.0, 0.5])

mean, std = model.predict_with_uncertainty(state, control)
print(f"State_dot: {mean} ± {std}")

# Feature importance 분석
lengthscales = model.get_lengthscales()
print(f"Lengthscales: {lengthscales}")  # 작을수록 중요한 feature
```

### Residual Dynamics 학습

#### 1. 데이터 수집 (실제 시스템)

```python
# 물리 모델과 실제 시스템 차이 수집
collector = DataCollector(state_dim=3, control_dim=2)

for episode in range(20):
    collector.start_episode()
    collect_episode_from_real_robot(duration=30.0)
    collector.end_episode()

data = collector.get_data()
dataset = DynamicsDataset(data, train_ratio=0.8, normalize=True)
```

#### 2. Residual 학습

```python
from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

# 물리 모델
physics_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)

# 물리 모델 예측
physics_predictions = []
for state, control in zip(dataset.train_inputs[:, :3], dataset.train_inputs[:, 3:]):
    physics_dot = physics_model.forward_dynamics(state, control)
    physics_predictions.append(physics_dot)

physics_predictions = np.array(physics_predictions)

# Residual 타겟 계산
residual_targets = dataset.train_targets - physics_predictions

# Residual 학습 (Neural Network)
residual_trainer = NeuralNetworkTrainer(
    state_dim=3,
    control_dim=2,
    hidden_dims=[64, 64],  # Residual은 작은 네트워크로 충분
)

history = residual_trainer.train(
    train_inputs=dataset.train_inputs,
    train_targets=residual_targets,
    val_inputs=dataset.val_inputs,
    val_targets=dataset.val_targets - physics_predictions_val,
    norm_stats=dataset.get_normalization_stats(),
    epochs=100,
)

residual_trainer.save_model("residual_model.pth")
```

#### 3. Residual Dynamics 사용

```python
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

# Residual 함수 정의
def residual_fn(state, control):
    return residual_trainer.predict(state, control, denormalize=True)

# Residual Dynamics 생성 (방법 1: 함수 래핑)
residual_model = ResidualDynamics(
    base_model=physics_model,
    residual_fn=residual_fn,
)

# 방법 2: 학습 모델 직접 연결 (권장)
# NeuralDynamics/GP를 learned_model로 전달하면 자동으로
# forward_dynamics → residual_fn, predict_with_uncertainty → uncertainty_fn 연결
from mppi_controller.models.learned.neural_dynamics import NeuralDynamics
learned = NeuralDynamics(model_path="residual_model.pth", state_dim=3, control_dim=2)
residual_model = ResidualDynamics(
    base_model=physics_model,
    learned_model=learned,  # 자동 와이어링
)

# MPPI와 통합
controller = MPPIController(model=residual_model, params=params)
```

---

## 성능 비교 및 선택 가이드

### 데이터 효율성

| 모델 타입 | 필요 데이터 | 학습 시간 | 추론 속도 | 불확실성 |
|----------|------------|----------|----------|---------|
| Neural | 5,000+ 샘플 | ~5분 (GPU) | 0.1ms | 없음 |
| GP (Exact) | 1,000-2,000 샘플 | ~10분 | 10ms | 높은 품질 |
| GP (Sparse) | 5,000+ 샘플 | ~15분 | 5ms | 높은 품질 |
| Residual | 2,000-3,000 샘플 | ~3분 | 0.2ms | 선택적 |
| Ensemble (M=5) | 5,000+ 샘플 | ~25분 (5x) | 0.5ms | 중간 품질 |
| MC-Dropout (M=20) | 5,000+ 샘플 | ~5분 (1x) | 2ms | 낮은 품질 |
| **MAML** | **40~200 (적응)** | **~5분 (메타)** | **~10ms (적응)** | **없음** |
| **EKF** | **없음** | **없음** | **~0.1ms** | **공분산** |
| **L1 Adaptive** | **없음** | **없음** | **~0.1ms** | **없음** |
| **ALPaCA** | **10~50 (적응)** | **~5분 (메타)** | **~1ms (적응)** | **Bayesian** |

### 정확도 비교 (Differential Drive 예시)

```
┌─────────────────────────────────────────────────────────┐
│          학습 데이터: 5,000 샘플 (50 episodes)         │
├─────────────────────────────────────────────────────────┤
│ Model Type    │  RMSE (m)  │  Max Error (m)  │ 2σ Cov │
├───────────────┼────────────┼─────────────────┼────────┤
│ Physics Only  │   0.150    │      0.450      │   N/A  │
│ Neural        │   0.035    │      0.120      │   N/A  │
│ GP (Exact)    │   0.042    │      0.135      │  96.2% │
│ GP (Sparse)   │   0.048    │      0.145      │  94.8% │
│ Residual+NN   │   0.028    │      0.095      │   N/A  │
│ Residual+GP   │   0.032    │      0.105      │  97.1% │
└───────────────┴────────────┴─────────────────┴────────┘

(2σ Cov: 2-sigma coverage, 이론적으로 95% 목표)
```

### 선택 가이드

#### Neural Dynamics를 선택하세요:
- ✅ 대량의 학습 데이터 확보 가능
- ✅ 고속 제어 필요 (>50Hz)
- ✅ GPU 가속 가능
- ✅ 불확실성보다 정확도 우선
- ❌ 데이터 수집 비용 높음
- ❌ 안전 크리티컬 응용

#### Gaussian Process를 선택하세요:
- ✅ 데이터 수집 비용 높음 (소량 데이터)
- ✅ 불확실성 정량화 필요
- ✅ 안전 크리티컬 응용
- ✅ 능동 학습 (불확실성 기반)
- ❌ 실시간 고속 제어 (>30Hz)
- ❌ 대규모 데이터 (>10,000)

#### Residual Dynamics를 선택하세요:
- ✅ 물리 모델 존재 (불완전함)
- ✅ Sim-to-Real 전이
- ✅ 데이터 효율성 + 정확도 동시 필요
- ✅ 외삽 영역에서 안정성 필요
- ❌ 물리 모델 없음
- ❌ 물리 모델이 매우 정확함 (학습 불필요)

#### MAML Dynamics를 선택하세요:
- ✅ 환경이 자주 변함 (다른 바닥, 하중, 마모 등)
- ✅ 빠른 적응 필요 (수초 이내)
- ✅ 오프라인으로 다양한 환경 시뮬레이션 가능
- ✅ Neural/Residual 오프라인 학습으로 부족할 때
- ❌ 환경이 고정적 (오프라인 학습 충분)
- ❌ 메타 학습용 환경 시뮬레이션 불가

#### EKF Adaptive를 선택하세요:
- ✅ 물리 파라미터 변화가 주된 불확실성 (마찰, 관성)
- ✅ 학습 인프라 없이 즉시 배포 필요
- ✅ 파라미터 모니터링/진단이 중요
- ✅ 추정 불확실성 정량화 필요 (공분산)
- ❌ 비모델링 외란 (wind, periodic force)이 지배적
- ❌ 파라미터 수가 많은 복잡한 모델

#### L1 Adaptive를 선택하세요:
- ✅ 시변 외란이 주된 불확실성 (풍하중, 경사면)
- ✅ 학습 없이 즉시 배포 필요
- ✅ 수학적 안정성 보장이 중요
- ✅ 고주파 노이즈 제거 필요 (저역통과 필터)
- ❌ 파라미터 정확 추정이 필요 (EKF가 적합)
- ❌ 매우 급격한 외란 변화 (필터 지연)

#### ALPaCA Dynamics를 선택하세요:
- ✅ SGD 없는 안정적 적응이 필요 (MAML의 SGD 불안정 우려)
- ✅ 적응 시 불확실성 정량화 필요
- ✅ 실시간 적응 + Bayesian 예측 동시 필요
- ✅ MAML과 동일한 메타 학습 인프라 활용 가능
- ❌ highly nonlinear 잔차 (linear last layer 한계)
- ❌ 메타 학습 불가 (MAML과 동일 제약)

---

## 온라인 학습 및 적응

자세한 내용은 [ONLINE_LEARNING.md](./ONLINE_LEARNING.md)를 참조하세요.

### 온라인 학습 개요

실시간으로 데이터를 수집하고 모델을 업데이트하여 도메인 변화에 적응합니다.

```python
from mppi_controller.learning.online_learner import OnlineLearner

# 사전 학습된 모델 및 트레이너
model = NeuralDynamics(model_path="neural_model.pth")
trainer = NeuralNetworkTrainer(...)

# 온라인 학습 관리자
online_learner = OnlineLearner(
    model=model,
    trainer=trainer,
    buffer_size=1000,          # 순환 버퍼 크기
    min_samples_for_update=100,
    update_interval=500,       # 500 샘플마다 재학습
)

# 제어 루프
for t in range(num_steps):
    state = get_state()
    control = controller.compute_control(state, ref_trajectory)

    apply_control(control)
    next_state = get_state()

    # 온라인 데이터 추가 (자동 재학습 트리거)
    online_learner.add_sample(state, control, next_state, dt)

# 성능 요약
summary = online_learner.get_performance_summary()
print(f"Updates: {summary['num_updates']}")
print(f"Improvement: {summary['adaptation_improvement']:.2f}%")
```

### Sim-to-Real 전이

시뮬레이터에서 학습한 모델을 실제 로봇에 적응:

1. **시뮬레이터에서 사전 학습**
   ```python
   # 시뮬레이터 데이터로 초기 모델 학습
   train_in_simulator(model, trainer, num_samples=10000)
   ```

2. **실제 로봇에서 온라인 적응**
   ```python
   # 실제 로봇 제어 + 온라인 학습
   online_learner = OnlineLearner(model, trainer)

   for episode in range(100):
       run_real_robot_episode(online_learner)
   ```

3. **적응 모니터링**
   ```python
   online_learner.monitor_adaptation(test_data)
   # Improvement: 45.3% (초기 RMSE: 0.250 → 현재: 0.137)
   ```

---

## MPPI 통합

### 기본 통합

모든 학습 모델은 `RobotModel` 인터페이스를 구현하므로 MPPI와 즉시 통합 가능:

```python
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams

# 모델 선택
model = NeuralDynamics(model_path="neural_model.pth")
# 또는
# model = GaussianProcessDynamics(model_path="gp_model.pth")
# 또는
# model = ResidualDynamics(base_model=physics_model, residual_fn=residual_fn)

# MPPI 파라미터
params = MPPIParams(
    num_samples=1024,
    horizon=30,
    dt=0.05,
    lambda_=1.0,
    sigma_u=np.array([0.5, 0.3]),
)

# MPPI 컨트롤러 생성
controller = MPPIController(model=model, params=params)

# 제어 계산
control, info = controller.compute_control(state, reference_trajectory)
```

### GP 불확실성 활용 (Risk-Aware MPPI)

GP의 불확실성을 MPPI 비용 함수에 통합:

```python
from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController

# GP 모델
gp_model = GaussianProcessDynamics(model_path="gp_model.pth")

# Risk-Aware MPPI (CVaR)
controller = RiskAwareMPPIController(
    model=gp_model,
    params=params,
    alpha=0.1,  # CVaR 신뢰수준 (하위 10%)
)

# 제어 계산 (불확실성 높은 샘플 페널티)
control, info = controller.compute_control(state, reference_trajectory)
```

### Tube-MPPI와 학습 모델

Tube-MPPI는 모델 불확실성에 강건합니다:

```python
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController

# 학습 모델 (불완전할 수 있음)
learned_model = NeuralDynamics(model_path="neural_model.pth")

# Tube-MPPI (외란 강건 제어)
controller = TubeMPPIController(
    model=learned_model,
    params=params,
    k_fb=np.array([2.0, 1.5]),  # 피드백 게인
)

# 제어 계산 (외란 보상)
control, info = controller.compute_control(state, reference_trajectory)
```

---

## 고급 기능

### 앙상블 학습 (EnsembleTrainer)

M개의 독립 MLP를 부트스트랩 학습:

```python
from mppi_controller.learning.ensemble_trainer import EnsembleTrainer

trainer = EnsembleTrainer(
    state_dim=3, control_dim=2,
    num_models=5,
    hidden_dims=[128, 128],
)
trainer.train(train_inputs, train_targets, val_inputs, val_targets, norm_stats)
trainer.save_model("ensemble.pth")
```

### 모델 검증 (ModelValidator)

학습 모델의 성능을 RMSE/MAE/R²/롤아웃 오차로 평가:

```python
from mppi_controller.learning.model_validator import ModelValidator

validator = ModelValidator()

# 단일 모델 평가
metrics = validator.evaluate(model.forward_dynamics, test_states, test_controls, test_targets)
print(f"RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.4f}")

# 여러 모델 비교
results = validator.compare(
    {"Neural": nn_model.forward_dynamics, "GP": gp_model.forward_dynamics},
    test_states, test_controls, test_targets,
)
validator.print_comparison(results)
```

### 불확실성 인식 비용 함수 (UncertaintyAwareCost)

GP/앙상블 불확실성에 비례하는 MPPI 비용 함수:

```python
from mppi_controller.controllers.mppi.uncertainty_cost import UncertaintyAwareCost

# GP 모델 불확실성 연동
unc_cost = UncertaintyAwareCost(
    uncertainty_fn=lambda s, c: gp_model.predict_with_uncertainty(s, c)[1],
    beta=10.0,     # 불확실성 페널티 가중치
    reduce="sum",  # "sum", "max", "mean"
)

# 앙상블 모델 불확실성 연동
unc_cost = UncertaintyAwareCost(
    uncertainty_fn=lambda s, c: ensemble.predict_with_uncertainty(s, c)[1],
    beta=5.0,
)
```

### GPU 가속 (TorchNeuralDynamics)

NeuralDynamics를 MPPI GPU 루프 내에서 사용 시 numpy↔torch 변환 오버헤드 제거:

```python
from mppi_controller.controllers.mppi.gpu import get_torch_model

# NeuralDynamics → TorchNeuralDynamics 자동 변환
torch_model = get_torch_model(neural_model, device="cuda")
```

### 체크포인트 버전 관리

OnlineLearner에서 `checkpoint_dir`를 설정하면 자동으로 모델 버전 관리:

```python
online_learner = OnlineLearner(
    model=model, trainer=trainer,
    checkpoint_dir="models/checkpoints",
    max_checkpoints=10,
)

# 자동 동작:
# 1. update_model() 호출 시 model_v{N}.pth 자동 저장
# 2. val_loss가 최적 대비 50% 이상 악화 시 자동 롤백
# 3. 오래된 체크포인트 자동 정리

# 수동 롤백
online_learner.rollback(version=3)

# 체크포인트 히스토리 확인
for cp in online_learner.get_checkpoint_history():
    best = " *BEST*" if cp["is_best"] else ""
    print(f"  v{cp['version']}: val_loss={cp['val_loss']:.6f}{best}")
```

### Simulator 통합

Simulator에서 OnlineLearner 자동 데이터 피드:

```python
from mppi_controller.simulation.simulator import Simulator

sim = Simulator(
    model=model, controller=controller, dt=0.05,
    online_learner=online_learner,  # 자동 add_sample() 호출
)
```

---

## 베스트 프랙티스

### 데이터 수집

1. **다양한 궤적**: 직선, 곡선, 급회전, 정지 등 다양한 시나리오
2. **상태 공간 커버리지**: 전체 동작 범위 탐색
3. **노이즈 주입**: 실제 환경의 불확실성 반영
4. **에피소드 기반**: 긴 연속 궤적보다 짧은 에피소드 여러 개

### 모델 학습

1. **정규화 필수**: 입력/출력 정규화로 학습 안정성 향상
2. **Train/Val 분할**: 80/20 비율로 과적합 방지
3. **Early Stopping**: 검증 손실 모니터링
4. **하이퍼파라미터 튜닝**: 학습률, 레이어 크기, 커널 타입 등

### 모델 검증

1. **RMSE/Max Error**: 기본 정확도 메트릭
2. **불확실성 보정**: GP는 2σ coverage 95% 목표
3. **외삽 테스트**: 학습 범위 밖 데이터로 테스트
4. **실제 제어 테스트**: 시뮬레이터/실제 로봇에서 추적 성능 확인

### 온라인 학습

1. **점진적 적응**: 작은 학습률로 천천히 업데이트
2. **재앙적 망각 방지**: 버퍼에 이전 데이터 유지
3. **성능 모니터링**: 적응 개선도 추적
4. **안전 메커니즘**: 성능 저하 시 이전 모델로 복귀

---

## API 레퍼런스

### NeuralDynamics

```python
class NeuralDynamics(RobotModel):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: str,
        device: str = "cpu",
    )

    def forward_dynamics(
        self,
        state: np.ndarray,    # (nx,) or (batch, nx)
        control: np.ndarray,  # (nu,) or (batch, nu)
    ) -> np.ndarray:          # (nx,) or (batch, nx)
        """신경망 예측"""
```

### GaussianProcessDynamics

```python
class GaussianProcessDynamics(RobotModel):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: str,
    )

    def forward_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """GP 평균 예측"""

    def predict_with_uncertainty(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(평균, 표준편차) 반환"""

    def get_lengthscales(self) -> List[np.ndarray]:
        """Feature importance 반환"""
```

### ResidualDynamics

```python
class ResidualDynamics(RobotModel):
    def __init__(
        self,
        base_model: RobotModel,
        residual_fn: Optional[Callable] = None,
        uncertainty_fn: Optional[Callable] = None,
        learned_model: Optional[RobotModel] = None,  # 자동 와이어링
    )

    def forward_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """f_physics + f_learned"""
```

### EnsembleNeuralDynamics

```python
class EnsembleNeuralDynamics(RobotModel):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
    )

    def forward_dynamics(self, state, control) -> np.ndarray:
        """앙상블 평균"""

    def predict_with_uncertainty(self, state, control) -> Tuple[np.ndarray, np.ndarray]:
        """(평균, 표준편차) 반환"""
```

### MCDropoutDynamics

```python
class MCDropoutDynamics(RobotModel):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        num_samples: int = 20,
        device: str = "cpu",
    )

    def forward_dynamics(self, state, control) -> np.ndarray:
        """MC 평균"""

    def predict_with_uncertainty(self, state, control) -> Tuple[np.ndarray, np.ndarray]:
        """(MC 평균, MC 표준편차) 반환"""
```

### MAMLDynamics

```python
class MAMLDynamics(NeuralDynamics):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        use_adam: bool = False,   # True → Adam, False → SGD
    )

    def save_meta_weights(self):
        """메타 파라미터 스냅샷 저장"""

    def restore_meta_weights(self):
        """메타 파라미터로 복원"""

    def adapt(
        self,
        states: np.ndarray,       # (M, nx)
        controls: np.ndarray,     # (M, nu)
        next_states: np.ndarray,  # (M, nx)
        dt: float,
        restore: bool = True,     # True → 메타 파라미터 복원 후 적응
                                  # False → 현재 파라미터에서 계속 fine-tune
    ) -> float:
        """Few-shot 적응 → 최종 loss 반환"""

    def forward_dynamics(self, state, control) -> np.ndarray:
        """적응된 모델로 예측 (NeuralDynamics 상속)"""
```

### EKFAdaptiveDynamics

```python
class EKFAdaptiveDynamics(RobotModel):
    def __init__(
        self,
        state_dim: int = 5,
        control_dim: int = 2,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        initial_c_v: float = 0.5,
        initial_c_omega: float = 0.3,
    )

    def forward_dynamics(self, state, control) -> np.ndarray:
        """추정 파라미터 기반 5D 동역학 예측"""

    def update_step(self, state, control, next_state, dt) -> dict:
        """단일 EKF 업데이트 (predict→innovate→update)
        Returns: {'c_v': float, 'c_omega': float, 'innovation': np.ndarray}"""

    def adapt(self, states, controls, next_states, dt, restore=True) -> float:
        """배치 적응 (MAML 인터페이스 호환). Returns: MSE."""

    def get_parameter_estimates(self) -> dict:
        """{'c_v': float, 'c_omega': float, 'c_v_std': float, 'c_omega_std': float}"""
```

### L1AdaptiveDynamics

```python
class L1AdaptiveDynamics(RobotModel):
    def __init__(
        self,
        state_dim: int = 5,
        control_dim: int = 2,
        adaptation_gain: float = 100.0,
        filter_bandwidth: float = 10.0,
        am_gains: np.ndarray = None,  # default: [-5,-5,-5,-10,-10]
    )

    def forward_dynamics(self, state, control) -> np.ndarray:
        """f_nom(x,u) + σ_filtered — 외란 보정 모델"""

    def update_step(self, state, control, next_state, dt) -> dict:
        """단일 L1 업데이트 (predictor→adaptation law→filter)
        Returns: {'sigma_raw': np.ndarray, 'sigma_filtered': np.ndarray, 'prediction_error': np.ndarray}"""

    def adapt(self, states, controls, next_states, dt, restore=True) -> float:
        """배치 적응 (MAML 인터페이스 호환). Returns: MSE."""

    def is_stable(self) -> bool:
        """A_m이 Hurwitz 안정인지 확인 (모든 고유값 실부 < 0)"""
```

### ALPaCADynamics

```python
class ALPaCADynamics(RobotModel):
    def __init__(
        self,
        state_dim: int = 5,
        control_dim: int = 2,
        model_path: Optional[str] = None,
        feature_dim: int = 64,
        hidden_dims: List[int] = None,
        device: str = "cpu",
    )

    def forward_dynamics(self, state, control) -> np.ndarray:
        """μ_n @ φ(x,u) — posterior mean 예측"""

    def adapt(self, states, controls, next_states, dt, restore=True) -> float:
        """Closed-form Bayesian update (SGD 없음). Returns: MSE."""

    def get_uncertainty(self, state, control) -> np.ndarray:
        """(1/β)·φᵀ·Λ_n⁻¹·φ — posterior predictive variance"""

    def restore_prior(self):
        """μ₀, Λ₀로 복원 (prior 상태)"""
```

### OnlineLearner

```python
class OnlineLearner:
    def __init__(
        self,
        model: RobotModel,
        trainer: Union[NeuralNetworkTrainer, GaussianProcessTrainer],
        buffer_size: int = 1000,
        min_samples_for_update: int = 100,
        update_interval: int = 500,
    )

    def add_sample(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
        dt: float,
    ):
        """샘플 추가 (자동 재학습 트리거)"""

    def update_model(self, num_epochs: int = 10):
        """모델 fine-tuning"""

    def monitor_adaptation(self, test_data: Dict):
        """적응 성능 모니터링"""
```

---

## 데모 실행

### Neural Dynamics 학습

```bash
python examples/learned/neural_dynamics_learning_demo.py \
    --num_episodes 50 \
    --trajectory circle \
    --plot
```

### GP vs Neural 비교

```bash
python examples/learned/gp_vs_neural_comparison_demo.py \
    --num_episodes 20 \
    --data_fraction 0.5 \
    --plot
```

### 온라인 학습

```bash
python examples/learned/online_learning_demo.py \
    --duration 60.0 \
    --noise_std 0.05 \
    --plot
```

### 10-Way 적응 기법 비교 (Dynamic World)

```bash
# 전체 파이프라인 (데이터 수집 + 학습 + 메타 학습 + 10-Way 평가)
python examples/comparison/model_mismatch_comparison_demo.py \
    --all --world dynamic --trajectory circle --duration 20

# 강한 외란 하 비교 (MAML/EKF/L1/ALPaCA 이점 강조)
python examples/comparison/model_mismatch_comparison_demo.py \
    --all --world dynamic --noise 0.7 --disturbance combined

# 실시간 비교 (메타 모델 사전 학습 필요)
python examples/comparison/model_mismatch_comparison_demo.py \
    --live --world dynamic --trajectory circle --duration 20
```

10-Way 비교 대상:
1. Kinematic (3D), 2. Neural (3D), 3. Residual (3D),
4. Dynamic (5D), 5. MAML-3D, 6. MAML-5D,
7. EKF (5D), 8. L1 Adaptive (5D), 9. ALPaCA (5D), 10. Oracle (5D)

### 6-DOF Mobile Manipulator 8-Way 학습 모델 벤치마크

6-DOF 모바일 매니퓰레이터(9D 상태, 8D 제어)에 대해 8개 학습 모델을 2개 시나리오(`ee_3d_circle`, `ee_3d_helix`)에서 비교합니다.

```bash
# 1. 모델 학습 (6종: NN, GP, Ensemble, MC-Dropout, MAML, ALPaCA)
PYTHONPATH=. python scripts/train_6dof_all_models.py --quick  # 빠른 테스트
PYTHONPATH=. python scripts/train_6dof_all_models.py          # 전체 학습

# 2. 벤치마크 실행
PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py

# 3. 특정 모델/시나리오 선택
PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py \
    --models kinematic,residual_nn,residual_maml,oracle --scenario ee_3d_circle

# 4. 학습 모델 없이 (kinematic + oracle만)
PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py \
    --models kinematic,oracle --duration 15
```

8-Way 비교 대상:
1. Kinematic (기준선), 2. Residual-NN, 3. Residual-GP,
4. Residual-Ensemble, 5. Residual-MCDropout,
6. Residual-MAML (온라인 적응), 7. Residual-ALPaCA (온라인 적응), 8. Oracle

---

## 참고 자료

### 논문
- Nagabandi et al. (2018) - "Neural Network Dynamics for Model-Based RL"
- Deisenroth et al. (2015) - "Gaussian Processes for Data-Efficient Learning"
- Hewing et al. (2020) - "Learning-Based MPC: A Review"
- Cheng et al. (2019) - "End-to-End Safe RL with GP"
- Finn et al. (2017) - "Model-Agnostic Meta-Learning for Fast Adaptation" (MAML)
- Nichol et al. (2018) - "On First-Order Meta-Learning Algorithms" (FOMAML)
- Harrison et al. (2018) - "Meta-Learning Priors for Efficient Online Bayesian Regression" (ALPaCA)
- Hovakimyan & Cao (2010) - "L1 Adaptive Control Theory" (L1 Adaptive)
- Ljung (1979) - "Asymptotic Behavior of the Extended Kalman Filter" (EKF)

### 라이브러리
- PyTorch: https://pytorch.org
- GPyTorch: https://gpytorch.ai
- Scikit-learn: https://scikit-learn.org

---

## 문제 해결

### Q: 신경망 학습이 수렴하지 않아요
**A**:
1. 정규화 확인 (`normalize=True`)
2. 학습률 낮추기 (1e-4 ~ 1e-3)
3. 배치 크기 조정 (64, 128, 256)
4. 레이어 크기 축소 (과적합 가능성)

### Q: GP 학습이 너무 느려요
**A**:
1. Sparse GP 사용 (`use_sparse_gp=True`)
2. Inducing points 줄이기 (500 → 300)
3. 데이터 샘플 줄이기 (GP는 2000 샘플로 충분)

### Q: 온라인 학습 중 성능이 떨어져요
**A**:
1. 학습률 낮추기 (fine-tuning epochs 5 → 3)
2. 버퍼 크기 늘리기 (재앙적 망각 방지)
3. 업데이트 주기 늘리기 (500 → 1000)

### Q: 외삽 영역에서 모델이 이상해요
**A**:
1. Residual Dynamics 사용 (물리 모델 fallback)
2. GP 사용 (불확실성 자동 증가)
3. 데이터 수집 범위 확장

---

**마지막 업데이트**: 2026-02-21
**작성자**: Claude Sonnet 4.5 + Claude Opus 4.6 + Geonhee LEE
