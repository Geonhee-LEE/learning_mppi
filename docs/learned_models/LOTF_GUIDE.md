# Learning on the Fly (LotF) 가이드

## 목차

1. [개요](#개요)
2. [배경: 왜 LotF가 필요한가](#배경-왜-lotf가-필요한가)
3. [핵심 기법 4가지](#핵심-기법-4가지)
4. [아키텍처](#아키텍처)
5. [구현 상세](#구현-상세)
6. [벤치마크](#벤치마크)
7. [API 레퍼런스](#api-레퍼런스)
8. [참고 논문](#참고-논문)

---

## 개요

**Learning on the Fly**는 Pan et al. (2025, UZH)의 논문에서 제안된 기법으로,
미분가능 시뮬레이터(Differentiable Simulator)를 통한 역전파(BPTT)로
잔차 동역학 모델을 **궤적 수준**에서 학습하고,
**LoRA**로 효율적인 온라인 적응을 수행하는 프레임워크입니다.

```
┌─────────────────────────────────────────────────────────────┐
│               Learning on the Fly 파이프라인                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ① 오프라인 학습 (BPTT)                                    │
│  ┌──────────────────────────────────────────┐               │
│  │  미분가능 시뮬레이터                     │               │
│  │  ┌──────┐   ┌──────┐   ┌──────┐         │               │
│  │  │ x(0) │──▶│ x(1) │──▶│ x(2) │──▶ ...  │               │
│  │  └──────┘   └──────┘   └──────┘         │               │
│  │      │           │           │           │               │
│  │      ▼           ▼           ▼           │               │
│  │   FK(x₀)     FK(x₁)     FK(x₂)         │               │
│  │      │           │           │           │               │
│  │      ▼           ▼           ▼           │               │
│  │   ‖ee-ref‖²  ‖ee-ref‖²  ‖ee-ref‖²      │               │
│  │      └─────────┬─────────┘              │               │
│  │                ▼                         │               │
│  │        L = Σₜ ‖FK(xₜ) - refₜ‖²         │               │
│  │                │                         │               │
│  │         ∂L/∂θ  ▼  (역전파)               │               │
│  │        ┌──────────────┐                  │               │
│  │        │ Residual MLP │                  │               │
│  │        │   θ (학습)   │                  │               │
│  │        └──────────────┘                  │               │
│  └──────────────────────────────────────────┘               │
│                                                             │
│  ② 온라인 적응 (LoRA)                                       │
│  ┌──────────────────────────────────────────┐               │
│  │  사전학습 MLP (frozen)                   │               │
│  │  ┌─────────────────────────┐             │               │
│  │  │ W (고정) + ΔW = A·B    │ ← rank=4    │               │
│  │  │         (학습)          │   ~10% 파라미터│              │
│  │  └─────────────────────────┘             │               │
│  │     최근 20개 데이터로 5 step SGD         │               │
│  └──────────────────────────────────────────┘               │
│                                                             │
│  ③ Spectral 정규화                                          │
│  ┌──────────────────────────────────────────┐               │
│  │  각 Linear layer의 σ_max(W) 제한          │               │
│  │  → Lipschitz 상수 bound → 외삽 안정성     │               │
│  └──────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 배경: 왜 LotF가 필요한가

### 기존 방식의 한계

기존 잔차 동역학 학습 파이프라인에는 세 가지 핵심 문제가 있습니다:

#### 문제 1: 1-step MSE의 한계

기존 `NeuralNetworkTrainer`는 **1-step 예측 오차**를 최소화합니다:

```
L_MSE = ‖f_pred(xₜ, uₜ) - f_true(xₜ, uₜ)‖²
```

이는 각 시점의 오차를 독립적으로 줄이지만, **궤적 수준의 누적 오차**를 고려하지 않습니다.
작은 1-step 오차도 N step 누적되면 큰 추적 오차로 이어질 수 있습니다.

```
1-step 오차:    ε = 0.001 rad/s (매우 작음)
20-step 누적:   Σε ≈ 0.02 rad (눈에 보이는 편차)
100-step 누적:  Σε ≈ 0.1 rad (심각한 추적 실패)
```

#### 문제 2: 온라인 적응의 비효율

MAML은 온라인 적응 시 **전체 파라미터**를 SGD로 업데이트합니다.
27,000개 파라미터를 모두 갱신하면:
- 계산 비용이 큼 (backward pass가 전체 네트워크)
- 소수 데이터(20개)로 전체 파라미터를 건드리면 과적합 위험
- 적응 안정성 보장 어려움

#### 문제 3: 외삽 불안정

학습된 MLP는 학습 데이터 분포 밖에서 예측값이 급격히 발산할 수 있습니다.
정규화 없는 모델은 큰 가중치 → 높은 Lipschitz 상수 → 작은 입력 변화에 큰 출력 변화.

### LotF의 해결책

| 문제 | 기존 | LotF 해결 |
|------|------|----------|
| 1-step MSE | `‖f̂-f‖²` per sample | **BPTT**: `Σₜ‖FK(xₜ)-refₜ‖²` 궤적 손실 |
| 전체 파라미터 적응 | MAML: 27K params SGD | **LoRA**: ~2.7K params SGD (10%) |
| 외삽 불안정 | L2 weight decay | **Spectral Reg**: σ_max(W) 직접 제한 |

---

## 핵심 기법 4가지

### 1. 미분가능 시뮬레이터 (Differentiable Simulator)

기존 NumPy 기반 시뮬레이터는 `np.cos`, `np.sin`, `np.einsum`으로 구현되어
gradient 계산이 불가능합니다. 미분가능 시뮬레이터는 이를 PyTorch로 포팅하여
**rollout 전체에 대한 역전파**를 가능하게 합니다.

```python
# NumPy (기존) — gradient 없음
state_dot = np.array([v * np.cos(theta), v * np.sin(theta), omega, ...])

# PyTorch (미분가능) — autograd gradient 자동 계산
state_dot = torch.stack([v * torch.cos(theta), v * torch.sin(theta), omega, ...])
```

**구현**: `DifferentiableMobileManipulator6DOF`

- 기구학 + DH 변환 + FK → 전부 PyTorch `torch.Tensor` 연산
- `float64`로 NumPy 모델과 수치 일치 (atol < 1e-10)
- RK4 적분 포함 (`step_rk4`)
- `rollout(state_0, controls, dt)` → (N+1, 9) 궤적, gradient 유지

**핵심**: rollout → FK → loss → backward가 **하나의 연속적인 계산 그래프**를 형성합니다.

### 2. BPTT 잔차 학습 (Backpropagation Through Time)

BPTT는 RNN 학습에서 유래한 기법으로, 시간 축으로 펼쳐진 계산 그래프를 통해
역전파합니다. 여기서는 **시뮬레이터 rollout 자체**가 "시간 축"입니다.

```
MSE 학습 (기존):
  Loss = Σᵢ ‖MLP(xᵢ, uᵢ) - target_i‖²     ← 각 샘플 독립

BPTT 학습 (LotF):
  x₁ = RK4(x₀, u₀, dt; θ)                   ← θ = MLP 파라미터
  x₂ = RK4(x₁, u₁, dt; θ)                   ← x₁에 의존 (체인)
  ...
  xₙ = RK4(xₙ₋₁, uₙ₋₁, dt; θ)
  Loss = Σₜ ‖FK(xₜ) - refₜ‖²               ← N-step 궤적 손실
  ∂Loss/∂θ = Σₜ ∂‖FK(xₜ)-refₜ‖²/∂xₜ · ∂xₜ/∂θ   ← 체인룰로 역전파
```

**TBPTT (Truncated BPTT)**: 긴 궤적에서는 gradient가 소실/폭발할 수 있습니다.
매 `truncation_length` step마다 `detach()`하여 gradient chain을 끊습니다.

```python
for t in range(N):
    if t > 0 and t % truncation_length == 0:
        state = state.detach().requires_grad_(True)  # gradient 끊기
    state = rk4_step(state, control[t], dt)
```

**핵심 차이**: MSE는 "각 시점의 기울기가 맞는가"를 최적화하고,
BPTT는 "이 기울기로 적분했을 때 궤적이 reference를 잘 따르는가"를 직접 최적화합니다.

### 3. LoRA 온라인 적응 (Low-Rank Adaptation)

LoRA는 대형 언어 모델(LLM) fine-tuning에서 시작된 기법으로,
사전학습된 가중치를 고정하고 **저차원 행렬 분해**로 소수 파라미터만 적응합니다.

```
기존 Linear:  y = W·x + b        (W: d_out × d_in, 전체 파라미터)

LoRA Linear:  y = W·x + b + (α/r)·A·B·x
                  ───────     ───────────
                  고정(frozen)   학습(trainable)

              A: (d_out × r), 초기값 = 0
              B: (r × d_in),  초기값 ~ N(0, 1/r)
              r ≪ min(d_out, d_in)  (예: r=4)
```

**왜 효과적인가**:

1. **A=0 초기화**: 초기에는 LoRA 기여 = 0 → 사전학습 모델 그대로 작동
2. **Low-rank**: 변화량 ΔW = A·B는 rank-r 행렬 → 소수 방향으로만 적응
3. **파라미터 효율**: `r=4, d=128`일 때 LoRA 파라미터 = `4×128 + 4×128 = 1024` vs 원본 `128×128 = 16384` (6%)
4. **과적합 방지**: 적은 파라미터 → 20개 데이터로도 안정적 적응

**MAML과 비교**:

| | MAML | LoRA |
|---|------|------|
| 적응 파라미터 | 전체 27K | ~2.7K (10%) |
| 적응 방식 | 전체 SGD | LoRA params만 SGD |
| 초기화 | 메타 파라미터 복원 | LoRA A=0 복원 (or 메타 LoRA) |
| 과적합 위험 | 중간 (전체 파라미터) | 낮음 (저차원 적응) |
| 인터페이스 | `adapt(states, controls, next_states, dt)` | **동일** |

### 4. Spectral 정규화 (Spectral Regularization)

**동기**: MLP의 Lipschitz 상수가 크면, 학습 데이터 분포 밖에서 출력이 급격히 변합니다.

```
Lipschitz 상수 L 정의:
  ‖f(x₁) - f(x₂)‖ ≤ L · ‖x₁ - x₂‖

MLP의 Lipschitz 상수:
  L(MLP) ≤ Π_i σ_max(Wᵢ)    (각 층의 최대 특이값의 곱)
```

**Spectral Regularization**은 각 Linear layer의 **최대 특이값 σ_max(W)**를
학습 손실에 패널티로 추가합니다:

```
L_total = L_MSE + λ_spectral · Σᵢ σ_max(Wᵢ)
```

**Power Iteration**으로 σ_max를 효율적으로 근사합니다 (O(d) per layer):

```
반복:
  v ← W^T u / ‖W^T u‖
  u ← W v / ‖W v‖
σ_max ≈ u^T W v    (differentiable!)
```

**효과**: 가중치 크기를 직접 제한하여 외삽 영역에서도 출력 변화가 완만해집니다.
L2 weight decay(`‖W‖²_F`)는 모든 특이값을 균등하게 줄이지만,
Spectral은 **최대 특이값만 선택적으로** 제한하여 표현력 손실을 최소화합니다.

---

## 아키텍처

### 전체 구성도

```
┌──────────────────────────────────────────────────────────────┐
│                     LotF 전체 아키텍처                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [오프라인 학습]                                             │
│                                                              │
│    방법 A: MSE (기존)                                        │
│    ┌─────────┐    ┌──────────┐    ┌──────────┐              │
│    │ DataSet │───▶│ MLP(θ)   │───▶│ L=‖ŷ-y‖² │              │
│    └─────────┘    └──────────┘    └──────────┘              │
│                                                              │
│    방법 B: MSE + Spectral                                    │
│    ┌─────────┐    ┌──────────┐    ┌──────────────────────┐  │
│    │ DataSet │───▶│ MLP(θ)   │───▶│ L=‖ŷ-y‖²+λΣσ_max   │  │
│    └─────────┘    └──────────┘    └──────────────────────┘  │
│                                                              │
│    방법 C: BPTT (LotF)                                       │
│    ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│    │ Episodes │───▶│ DiffSim      │───▶│ L=Σ‖FK(x)-ref‖² │  │
│    │ (s,u,ref)│    │ +Residual(θ) │    │ +λΣσ_max       │   │
│    └──────────┘    └──────────────┘    └────────────────┘   │
│                                                              │
│  [온라인 적응]                                               │
│                                                              │
│    방법 D: LoRA                                              │
│    ┌─────────────┐    ┌───────────────────────┐             │
│    │ 최근 20 데이터│───▶│ MLP(W_frozen + A·B)  │             │
│    │ (실시간 수집) │    │ SGD on A,B only      │             │
│    └─────────────┘    └───────────────────────┘             │
│                                                              │
│    방법 E: MAML (기존)                                       │
│    ┌─────────────┐    ┌───────────────────────┐             │
│    │ 최근 20 데이터│───▶│ MLP(θ_meta)          │             │
│    │ (실시간 수집) │    │ SGD on all θ         │             │
│    └─────────────┘    └───────────────────────┘             │
│                                                              │
│  [추론] (모든 방법 공통)                                     │
│                                                              │
│    ẋ_total = f_kinematic(x, u) + MLP_residual(x, u)         │
│              ─────────────────   ──────────────────          │
│              물리 기반 (안전)     학습 보정 (정확도)           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 파일 구조

```
mppi_controller/
├── models/
│   ├── learned/
│   │   ├── lora_dynamics.py           ← LoRALinear + LoRADynamics
│   │   ├── neural_dynamics.py          (LoRA가 상속)
│   │   ├── maml_dynamics.py            (adapt() 인터페이스 참조)
│   │   └── residual_dynamics.py        (base + learned 합성)
│   └── differentiable/
│       ├── __init__.py
│       └── diff_sim_6dof.py           ← PyTorch 미분가능 시뮬레이터
├── learning/
│   ├── spectral_regularization.py     ← SpectralRegularizer
│   ├── bptt_residual_trainer.py       ← BPTT 궤적 학습
│   ├── nn_policy_trainer.py           ← NN-Policy (BC + BPTT) 학습
│   ├── neural_network_trainer.py       (spectral_lambda 통합)
│   └── ensemble_trainer.py             (spectral_lambda 통합)
├── examples/comparison/
│   └── lotf_benchmark.py             ← 8-Way 벤치마크 + Live 모드
└── tests/
    └── test_lotf.py                   ← 35개 통합 테스트
```

---

## 구현 상세

### LoRADynamics

```python
from mppi_controller.models.learned.lora_dynamics import LoRADynamics
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics

# 사전학습된 잔차 MLP 위에 LoRA 적용
lora = LoRADynamics(
    state_dim=9, control_dim=8,
    model_path="models/learned_models/6dof_benchmark/nn/best_model.pth",
    lora_rank=4,       # Low-rank 차원 (작을수록 파라미터 적음)
    lora_alpha=1.0,    # LoRA 스케일
    inner_lr=0.01,     # 적응 학습률
    inner_steps=5,     # 적응 SGD 횟수
)
lora.save_meta_weights()  # 초기 LoRA 상태 저장

# ResidualDynamics로 합성 (기구학 + LoRA 잔차)
model = ResidualDynamics(base_model=kin_model, learned_model=lora)

# 온라인 적응 (MAMLDynamics와 동일 인터페이스)
loss = lora.adapt(
    states, controls, next_states, dt=0.05,
    restore=True,           # 메타 LoRA로 복원 후 적응
    temporal_decay=0.95,    # 최근 데이터 강조
)

# 파라미터 효율 확인
print(f"Trainable: {lora.get_trainable_params():,}")   # ~2,700
print(f"Total:     {lora.get_total_params():,}")        # ~27,000
```

### SpectralRegularizer

```python
from mppi_controller.learning.spectral_regularization import SpectralRegularizer

reg = SpectralRegularizer(model, lambda_spectral=0.01, n_power_iterations=1)

# 학습 루프에서:
loss = mse_loss + reg.compute_penalty()  # differentiable
loss.backward()

# 디버깅: 각 layer의 σ_max 확인
norms = reg.get_spectral_norms()
# {'network.0': 2.34, 'network.2': 1.87, 'network.4': 1.12}
```

### NeuralNetworkTrainer (Spectral 통합)

```python
from mppi_controller.learning.neural_network_trainer import NeuralNetworkTrainer

# spectral_lambda > 0 → 자동으로 SpectralRegularizer 생성
trainer = NeuralNetworkTrainer(
    state_dim=9, control_dim=8,
    hidden_dims=[128, 128],
    spectral_lambda=0.01,   # ← 새 파라미터
)
```

### DifferentiableMobileManipulator6DOF

```python
from mppi_controller.models.differentiable.diff_sim_6dof import (
    DifferentiableMobileManipulator6DOF,
    DifferentiableMobileManipulator6DOFDynamic,
)

# 기구학 (미분가능)
sim = DifferentiableMobileManipulator6DOF()

# 동역학 (마찰/중력 포함, 미분가능)
sim_dyn = DifferentiableMobileManipulator6DOFDynamic()

# Rollout (gradient 유지)
state_0 = torch.zeros(9, dtype=torch.float64)
controls = torch.randn(20, 8, dtype=torch.float64, requires_grad=True)
trajectory = sim.rollout(state_0, controls, dt=0.05)  # (21, 9)

# FK (미분가능)
ee_positions = sim.forward_kinematics(trajectory)  # (21, 3)

# Loss → backward
loss = ((ee_positions - ee_reference) ** 2).sum()
loss.backward()  # controls.grad가 생성됨!
```

### BPTTResidualTrainer

```python
from mppi_controller.learning.bptt_residual_trainer import BPTTResidualTrainer
from mppi_controller.learning.neural_network_trainer import DynamicsMLPModel

# 잔차 MLP (float64 for 미분가능 시뮬레이터 호환)
residual_mlp = DynamicsMLPModel(
    input_dim=17, output_dim=9, hidden_dims=[128, 128]
).double()

trainer = BPTTResidualTrainer(
    residual_model=residual_mlp,
    diff_sim=sim,
    norm_stats=norm_stats,
    learning_rate=1e-4,
    spectral_lambda=0.01,
    rollout_horizon=20,      # rollout 길이
    truncation_length=10,    # TBPTT 절단 간격
    dt=0.05,
)

# 에피소드 생성 (MPPI로 데이터 수집)
episodes = trainer.generate_episodes(dyn_model, kin_model, traj_fn, n_episodes=50)

# BPTT 학습
history = trainer.train(
    train_episodes=episodes[:40],
    val_episodes=episodes[40:],
    epochs=100,
)
```

---

## 벤치마크

### 8-Way 비교

| # | 모델 | 학습 방식 | 온라인 적응 | 파라미터 |
|---|------|----------|-----------|---------|
| 1 | Kinematic | 없음 | X | 0 |
| 2 | Res-NN (MSE) | MSE offline | X | 27K |
| 3 | Res-NN (MSE+Spec) | MSE + λ·σ_max | X | 27K |
| 4 | Res-NN (BPTT) | 궤적 BPTT | X | 27K |
| 5 | Res-LoRA | MSE pretrain + LoRA adapt | O | 2.7K trainable |
| 6 | Res-MAML | Meta pretrain + SGD adapt | O | 27K trainable |
| 7 | NN-Policy (BPTT) | BC + BPTT fine-tune | X | 27K |
| 8 | Oracle | 없음 (완벽 모델) | X | 0 |

**NN-Policy**: MPPI 없이 NN이 직접 `(state, ee_ref) → control`을 출력하는 정책.
Behavioral Cloning (MPPI 시연) + BPTT Fine-tune으로 학습.
잔차+MPPI 방식이 순수 NN 정책보다 우수함을 보여주는 비교 대상.

### 실행 방법

```bash
# 배치 모드 (전체 모델 × 전체 시나리오)
PYTHONPATH=. python examples/comparison/lotf_benchmark.py

# 특정 시나리오
PYTHONPATH=. python examples/comparison/lotf_benchmark.py --scenario ee_3d_helix

# 특정 모델만
PYTHONPATH=. python examples/comparison/lotf_benchmark.py --models kinematic,bptt,lora,nn_policy,oracle

# 짧은 벤치마크
PYTHONPATH=. python examples/comparison/lotf_benchmark.py --duration 10

# 실시간 애니메이션 모드
PYTHONPATH=. python examples/comparison/lotf_benchmark.py --live --models kinematic,nn_policy,oracle
PYTHONPATH=. python examples/comparison/lotf_benchmark.py --live --scenario ee_3d_helix --duration 15

# NN-Policy 학습 (BC + BPTT)
PYTHONPATH=. python scripts/train_6dof_lotf_models.py --models nn_policy
```

### 기대 결과

```
Model                ee_3d_circle   ee_3d_helix   Solve(ms)  Params
─────────────────────────────────────────────────────────────────
Oracle               ~0.063m        ~0.063m       ~43        -
Res-BPTT             ~0.065m        ~0.065m       ~50        27K
Res-LoRA (adapted)   ~0.066m        ~0.066m       ~50        2.7K trainable
Res-MAML (adapted)   ~0.069m        ~0.067m       ~50        27K trainable
Res-NN (MSE+Spec)    ~0.070m        ~0.070m       ~50        27K
Res-NN (MSE)         ~0.072m        ~0.073m       ~50        27K
Kinematic            ~0.066m        ~0.074m       ~27        -
NN-Policy (BPTT)     ~0.10m+        ~0.12m+       ~1         27K
```

> NN-Policy는 MPPI 없이 단일 forward pass로 제어하므로 solve time이 ~1ms로 매우 빠르나,
> 샘플링 기반 최적화 없이 직접 제어하므로 추적 RMSE가 잔차+MPPI 방식보다 높음.

### 시각화 (Live 모드)

```
┌──────────────────────┬──────────────────────┐
│  [3D] EE Trajectory  │  EE Tracking Error   │
│  모델별 색상 구분    │  시계열 그래프       │
│  + Reference (검정)  │  + 범례              │
│  + 현재 위치 마커    │                      │
├──────────────────────┼──────────────────────┤
│  Current RMSE        │  Live Metrics        │
│  바 차트 (실시간)    │  t=12.3s / 20s       │
│  + 수치 라벨        │  Model  RMSE  Solve  │
│                      │  Oracle 0.063 43ms   │
└──────────────────────┴──────────────────────┘
```

---

## API 레퍼런스

### LoRALinear

| 메서드 | 설명 |
|--------|------|
| `__init__(original, rank, alpha)` | nn.Linear을 LoRA로 감싸기 |
| `forward(x)` | `W@x + b + (α/r)·A@B@x` |
| `reset_lora()` | A=0, B~N(0,1/r)으로 리셋 |

### LoRADynamics

| 메서드 | 설명 |
|--------|------|
| `adapt(states, controls, next_states, dt, ...)` | LoRA 온라인 적응 (MAML 호환) |
| `save_meta_weights()` / `restore_meta_weights()` | LoRA 상태 저장/복원 |
| `reset_lora()` | LoRA를 0으로 (= 사전학습 모델) |
| `get_trainable_params()` | LoRA 파라미터 수 |
| `get_total_params()` | 전체 파라미터 수 |

### SpectralRegularizer

| 메서드 | 설명 |
|--------|------|
| `compute_penalty()` | `λ · Σᵢ σ_max(Wᵢ)` (differentiable) |
| `get_spectral_norms()` | 각 layer의 σ_max dict |

### DifferentiableMobileManipulator6DOF

| 메서드 | 설명 |
|--------|------|
| `forward_dynamics(state, control)` | ẋ = f(x, u) (미분가능) |
| `step_rk4(state, control, dt)` | RK4 적분 (미분가능) |
| `forward_kinematics(state)` | FK → EE 3D 위치 (미분가능) |
| `rollout(state_0, controls, dt)` | 궤적 rollout (미분가능) |

### BPTTResidualTrainer

| 메서드 | 설명 |
|--------|------|
| `train(train_episodes, val_episodes, epochs)` | BPTT 학습 |
| `differentiable_rollout(state_0, controls, dt)` | TBPTT rollout |
| `trajectory_loss(trajectory, ee_reference)` | 궤적 추적 손실 |
| `generate_episodes(dyn_model, kin_model, traj_fn)` | 학습 에피소드 생성 |

### NNPolicyTrainer (NN-Policy)

NN이 직접 `(state, ee_ref) → control`을 출력하는 정책 학습기.
MPPI 없이 단일 forward pass로 제어하므로 잔차+MPPI 방식과의 비교용.

```python
from mppi_controller.learning.nn_policy_trainer import NNPolicyTrainer, PolicyMLP

trainer = NNPolicyTrainer(
    state_dim=9, ee_ref_dim=3, control_dim=8,
    hidden_dims=[128, 128, 64],
    control_bounds=np.array([1.0, 2.0] + [3.0] * 6),
    learning_rate=1e-3,
)

# Phase 1: MPPI oracle 데이터 수집
episodes = trainer.generate_demonstrations(dyn_model, kin_model, traj_fn, n_episodes=80)

# Phase 2: Behavioral Cloning (MSE)
trainer.train_bc(episodes, epochs=100)

# Phase 3: BPTT fine-tune (궤적 손실)
trainer.train_bptt(episodes, epochs=50)

# Inference (MPPI 없이 직접 제어)
control = trainer.compute_control(state, ee_ref)
```

| 메서드 | 설명 |
|--------|------|
| `generate_demonstrations(dyn, kin, traj_fn, n_episodes)` | MPPI oracle 데이터 수집 |
| `train_bc(episodes, epochs, batch_size)` | Behavioral Cloning (MSE) |
| `train_bptt(episodes, epochs, rollout_horizon)` | BPTT fine-tune (궤적 손실) |
| `compute_control(state, ee_ref)` | 정책 inference (단일 forward pass) |
| `save_model(filename)` / `load_model(filename)` | 모델 저장/로드 |

---

## 참고 논문

1. **Pan et al. (2025)** — "Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation" (UZH)
   - 미분가능 시뮬레이터 + BPTT 정책 학습의 원본 논문

2. **Hu et al. (2022)** — "LoRA: Low-Rank Adaptation of Large Language Models"
   - Low-Rank Adaptation 원본 논문 (NLP에서 시작, robotics로 확장)

3. **Miyato et al. (2018)** — "Spectral Normalization for Generative Adversarial Networks"
   - Power iteration 기반 spectral normalization 원본

4. **Finn et al. (2017)** — "Model-Agnostic Meta-Learning for Fast Adaptation" (MAML)
   - LoRA 대비 기준인 MAML의 원본 논문

5. **Williams et al. (2016)** — "Aggressive Driving with MPPI"
   - MPPI 프레임워크 원본 (잔차 모델이 이 위에서 동작)
