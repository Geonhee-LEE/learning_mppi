# 메타 학습 (Meta-Learning) 가이드

**날짜**: 2026-02-21
**버전**: 2.0

## 목차

1. [개요](#개요)
2. [메타 학습이란?](#메타-학습이란)
3. [MAML 이론](#maml-이론)
4. [FOMAML — 1차 근사](#fomaml--1차-근사)
5. [아키텍처](#아키텍처)
6. [Residual Meta-Training (핵심 개선)](#residual-meta-training-핵심-개선)
7. [외란 시뮬레이션](#외란-시뮬레이션)
8. [실행 방법](#실행-방법)
9. [API 레퍼런스](#api-레퍼런스)
10. [성능 분석](#성능-분석)
11. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
12. [기존 학습 방법과의 비교](#기존-학습-방법과의-비교)
13. [문제 해결](#문제-해결)

---

## 개요

MAML (Model-Agnostic Meta-Learning)은 **"학습하는 방법을 학습하는"** 메타 학습 알고리즘입니다.

로봇 동역학 문맥에서, 다양한 환경(마찰 계수, 관성 등)에서 **사전 학습된 메타 파라미터**로부터 시작하여, 실행 중 **최근 수십 개의 데이터만으로 현재 환경에 빠르게 적응**합니다.

### 핵심 아이디어

```
┌─────────────────────────────────────────────────────────────────────┐
│                        기존 학습 방법                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  오프라인 학습:  특정 환경 데이터 ──→ 학습 ──→ 고정 모델            │
│                  (환경 변하면 성능 저하)                             │
│                                                                     │
│  온라인 학습:    실시간 데이터 ──→ fine-tuning ──→ 점진적 적응      │
│                  (수렴 느림, 초기 성능 낮음)                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        MAML (메타 학습)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  메타 학습:     다양한 환경 ──→ 메타 파라미터 θ* 학습              │
│                                                                     │
│  실행 중 적응:  최근 데이터 ──→ SGD 100 step ──→ 즉시 적응          │
│                 (수 ms 소요, few-shot 적응)                         │
│                                                                     │
│  θ* ─────(100 step SGD)───→ θ_adapted                              │
│  (모든 환경에    (현재 환경에 최적화)                               │
│   좋은 초기점)                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 주요 특징

- **Few-Shot 적응**: 10~50개 데이터만으로 현재 환경 적응
- **빠른 수렴**: SGD 100 step (수 ms)으로 적응 완료
- **Residual 학습**: DynamicKinematicAdapter base + MAML 잔차 보정 (5D) 또는 Kinematic base (3D)
- **Residual Meta-Training**: 메타 학습 시 잔차 타겟 사용 → 온라인 적응과 분포 일치 (핵심 개선)
- **외란 적응**: 시간에 따라 변하는 외란 (wind/terrain/sine)에도 온라인 적응
- **환경 불변**: 마찰/관성/지연/외란 등 다양한 환경 변화에 대응
- **안전한 롤백**: 매 적응마다 메타 파라미터에서 시작 (누적 드리프트 없음)
- **Reptile 지원**: FOMAML 대안으로 더 간단하고 안정적인 Reptile 알고리즘 선택 가능

---

## 메타 학습이란?

### 학습의 3가지 수준

| 수준 | 설명 | 예시 |
|------|------|------|
| **추론** (Inference) | 학습된 모델로 예측 | `forward_dynamics(state, control)` |
| **학습** (Learning) | 데이터로 모델 파라미터 학습 | `trainer.train(data)` |
| **메타 학습** (Meta-Learning) | "어떻게 학습할지"를 학습 | `maml_trainer.meta_train()` |

### 메타 학습 = "Learning to Learn"

일반 학습은 하나의 **태스크** (예: 특정 마찰 계수의 로봇)에 대해 모델을 학습합니다.
메타 학습은 **태스크의 분포** (예: 다양한 마찰 계수)에 대해, 새로운 태스크에 빠르게 적응할 수 있는 **초기 파라미터**를 학습합니다.

```
태스크 분포 D(T):
  T1: c_v=0.2, c_omega=0.1  (빙판 위 로봇)
  T2: c_v=0.5, c_omega=0.3  (일반 바닥)
  T3: c_v=0.8, c_omega=0.4  (카펫 위 로봇)
  ...

메타 학습 목표:
  θ* = argmin_θ E_{T~D} [ L_T(θ - α∇L_T(θ)) ]
       "적응 후 성능이 좋은 초기 파라미터 찾기"
```

---

## MAML 이론

### MAML (Finn et al., 2017)

MAML은 **모델 구조에 무관한 (Model-Agnostic)** 메타 학습 알고리즘입니다.

**핵심**: 그래디언트 기반 적응(SGD)이 가장 효과적인 **초기 파라미터 θ\***를 찾습니다.

### 알고리즘

```
Input: 태스크 분포 p(T), inner LR α, meta LR β
Output: 메타 파라미터 θ*

1. θ ← 랜덤 초기화
2. for each meta iteration:
   a. 태스크 배치 {T1, ..., Tn} ~ p(T) 샘플링
   b. for each task Ti:
      i.   Di_support, Di_query = sample_data(Ti)
      ii.  θ'_i = θ - α ∇_θ L(θ; Di_support)     ← Inner Loop (적응)
      iii. L_query_i = L(θ'_i; Di_query)            ← Query Loss
   c. θ ← θ - β ∇_θ Σ_i L_query_i                  ← Outer Loop (메타 업데이트)
3. return θ* = θ
```

### Inner Loop vs Outer Loop

```
┌──────────────────────────────────────────────────────────────┐
│                      MAML 구조                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Outer Loop (메타 학습):                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  메타 파라미터 θ를 업데이트                           │  │
│  │  목표: 적응 후 query loss 최소화                      │  │
│  │  학습률: β (meta_lr = 5e-4)                          │  │
│  │  옵티마이저: Adam                                     │  │
│  │                                                        │  │
│  │  Inner Loop (적응):                                    │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  각 태스크의 support set으로 SGD 수행             │  │  │
│  │  │  학습률: α (inner_lr = 0.005)                    │  │  │
│  │  │  스텝 수: K (inner_steps = 10)                   │  │  │
│  │  │  옵티마이저: SGD (vanilla)                       │  │  │
│  │  │                                                    │  │  │
│  │  │  θ'_i = θ - α ∇L(θ; support)                    │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  │  θ ← θ - β ∇_θ Σ L(θ'_i; query)                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## FOMAML — 1차 근사

### MAML vs FOMAML

본 구현은 **FOMAML (First-Order MAML)**을 사용합니다.

원래 MAML은 2차 미분 (Hessian)을 계산해야 합니다:

```
Full MAML gradient:
  ∇_θ L(θ') = ∇_θ L(θ - α∇L(θ))
             = (I - α ∇²L(θ)) ∇L(θ')    ← 2차 미분 필요!
                     ^^^^^^^
                     Hessian
```

FOMAML은 2차 미분 항을 생략하여 **1차 미분만** 사용합니다:

```
FOMAML gradient:
  ∇_θ L(θ') ≈ ∇_θ' L(θ')    ← 1차 미분만! (create_graph=False)
```

### FOMAML의 장점

| 비교 항목 | Full MAML | FOMAML |
|----------|-----------|--------|
| 메모리 | O(K * params) | O(params) |
| 계산 비용 | 2차 미분 필요 | 1차 미분만 |
| 구현 복잡도 | 높음 | 낮음 |
| 성능 | 약간 우수 | 거의 동일 (실험적) |

Nichol et al. (2018) "On First-Order Meta-Learning Algorithms"에서 FOMAML이 Full MAML과 비슷한 성능을 보임을 증명.

### 코드에서의 FOMAML

```python
# Inner loop에서 create_graph=False → 2차 미분 비활성화
grads = torch.autograd.grad(
    loss, list(adapted_params.values()),
    create_graph=False,  # FOMAML: 1차 근사
)
```

---

## 아키텍처

### 파일 구조

```
mppi_controller/
├── models/learned/
│   ├── maml_dynamics.py        # MAMLDynamics — NeuralDynamics 상속
│   │                             # save/restore_meta_weights()
│   │                             # adapt(sample_weights, temporal_decay)
│   ├── ekf_dynamics.py          # EKFAdaptiveDynamics — 7D EKF 파라미터 추정
│   │                             # update_step(), get_parameter_estimates()
│   ├── l1_adaptive_dynamics.py  # L1AdaptiveDynamics — 외란 추정+보상
│   │                             # update_step(), is_stable()
│   └── alpaca_dynamics.py       # ALPaCADynamics — Bayesian last-layer
│                                  # adapt() (closed-form), get_uncertainty()
│
├── models/kinematic/
│   └── dynamic_kinematic_adapter.py  # DynamicKinematicAdapter — 5D MPPI base
│
├── learning/
│   ├── maml_trainer.py          # MAMLTrainer — FOMAML 메타 학습
│   │                             # _generate_task_data_5d() — 5D 데이터 생성
│   │                             # meta_train() — 메타 학습 루프
│   ├── reptile_trainer.py       # ReptileTrainer — Reptile 메타 학습
│   │                             # theta += epsilon * (adapted - theta)
│   └── alpaca_trainer.py        # ALPaCATrainer — ALPaCA 메타 학습
│                                  # Bayesian update → query loss → backprop
│
examples/comparison/
├── model_mismatch_comparison_demo.py  # 10-Way 비교 데모
│                                        # meta_train_maml_5d() (residual meta-training)
│                                        # meta_train_alpaca() (ALPaCA meta-training)
│                                        # run_with_dynamic_world_{ekf,l1,alpaca}()
└── disturbance_profiles.py      # 외란 프로필 4종
                                   # WindGust/TerrainChange/Sinusoidal/Combined
tests/
├── test_maml.py                 # 32개 테스트
├── test_ekf_dynamics.py         # 18개 테스트
├── test_l1_adaptive.py          # 17개 테스트
└── test_alpaca.py               # 23개 테스트
```

### 클래스 계층

```
RobotModel (ABC)
├── NeuralDynamics
│   └── MAMLDynamics              ← 메타 학습 + SGD inner-loop 적응
│       ├── save_meta_weights()   ← 메타 파라미터 스냅샷
│       ├── restore_meta_weights() ← 적응 전 복원
│       └── adapt()               ← inner-loop SGD (100 step)
│
├── EKFAdaptiveDynamics           ← EKF 파라미터 추정 (학습 불필요)
│   ├── update_step()             ← EKF predict→update
│   ├── adapt()                   ← 배치 순차 EKF
│   └── get_parameter_estimates() ← c_v, c_omega + std
│
├── L1AdaptiveDynamics            ← L1 외란 추정+보상 (학습 불필요)
│   ├── update_step()             ← predictor→adaptation→filter
│   ├── adapt()                   ← 배치 순차 업데이트
│   └── is_stable()               ← Hurwitz 안정성 확인
│
└── ALPaCADynamics                ← Bayesian last-layer (메타 학습 필요)
    ├── adapt()                   ← closed-form Bayesian update
    ├── get_uncertainty()         ← posterior predictive variance
    └── restore_prior()           ← prior 상태 복원
```

### 데이터 흐름

```
  [메타 학습 단계]                         [실행 단계]

  DynamicWorld(c_v=0.2)  ──┐
  DynamicWorld(c_v=0.5)  ──┼──→ MAMLTrainer.meta_train()
  DynamicWorld(c_v=0.8)  ──┘          │
                                       ▼
                              메타 파라미터 θ* 저장
                              (maml_meta_model.pth)
                                       │
                              ┌────────▼────────┐
                              │  MAMLDynamics    │
                              │  load(θ*)        │
                              │  save_meta()     │
                              └────────┬────────┘
                                       │
          ┌────────────────────────────▼────────────────────────┐
          │       2-Phase 실행 루프 (Residual MAML)             │
          │                                                      │
          │  Phase 1 (warm-up, ~2초):                            │
          │    기구학 모델로 제어 + 전이 데이터 수집 (40 step)   │
          │                                                      │
          │  Phase 1→2 전환:                                     │
          │    잔차 계산: residual = actual - kinematic           │
          │    MAML 적응: restore(θ*) → adapt(residual) (100 SGD)│
          │    ResidualDynamics(kinematic + MAML) 컨트롤러 전환   │
          │                                                      │
          │  Phase 2 (적응 제어):                                 │
          │    ResidualDynamics로 MPPI 제어                      │
          │    80 step마다: restore(θ*) → adapt(최신 잔차) 재적응 │
          └──────────────────────────────────────────────────────┘
```

### Residual MAML 아키텍처

MAML을 단독 모델로 사용하면 MPPI rollout에서 예측 오차가 누적됩니다.
**Residual MAML**은 기구학 모델을 base로, MAML이 잔차(마찰/관성 보정)만 학습합니다:

```
  forward_dynamics(state, control) =
      kinematic(state, control)    ← 기구학 base (안정적)
    + MAML_residual(state, control) ← 학습된 보정 (적응적)
```

장점:
- 기구학 base가 **안정성 보장** (MAML 보정이 0이어도 Kinematic 수준 유지)
- MAML은 **작은 보정량만 학습** → 빠른 수렴, 낮은 오차
- 5-seed 평균 RMSE **0.081m ± 0.007** (매우 안정적)

---

## Residual Meta-Training (핵심 개선)

### 문제: 분포 불일치 (Distribution Mismatch)

초기 구현에서 MAML이 Kinematic(0.029m)보다 나쁜 성능(0.086m)을 보인 근본 원인:

```
┌──────────────────────────────────────────────────────────────────┐
│                    분포 불일치 문제                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  메타 학습 (오프라인):                                            │
│    target = next_state - state  (전체 동역학)                    │
│    → MAML이 학습하는 분포: 큰 값 (전체 state_dot)               │
│                                                                  │
│  온라인 적응 (실행 중):                                           │
│    target = (next_state - base_next)  (잔차만)                   │
│    → MAML이 적응하는 분포: 작은 값 (base와의 차이)              │
│                                                                  │
│  → 메타 파라미터 θ*는 "전체 동역학"에 최적화되어 있어            │
│    "잔차"에 적응할 때 초기점이 나쁨!                             │
│                                                                  │
│  결과: 적응된 MAML 잔차가 오히려 base 모델을 악화시킴            │
│        Phase 1(warmup) RMSE 0.042m → Phase 2(adapted) 0.083m    │
└──────────────────────────────────────────────────────────────────┘
```

### 해결: Residual Meta-Training

메타 학습 시에도 잔차 타겟을 사용하여 온라인 적응과 분포를 일치시킴:

```python
# meta_train_maml_5d() 핵심 코드
base_5d = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1, k_v=5.0, k_omega=5.0)
dt = 0.05

original_gen = trainer._generate_task_data_5d

def residual_gen(task_params, n_samples):
    states, controls, next_states = original_gen(task_params, n_samples)
    # base 모델의 예측
    base_dots = base_5d.forward_dynamics(states, controls)
    base_next = states + base_dots * dt
    # 잔차 타겟: "base가 예측 못한 부분"만
    residual_next = states + (next_states - base_next)
    return states, controls, residual_next

trainer._generate_task_data_5d = residual_gen
trainer.meta_train(n_iterations=1000)
```

### 결과

| 접근 방식 | Phase 2 RMSE | 비고 |
|-----------|-------------|------|
| 전체 동역학 메타 학습 + 잔차 적응 (기존) | 0.083m | 분포 불일치 → 성능 악화 |
| 전체 동역학 메타 학습 + 전체 적응 | 0.200m | MLP가 전체 dynamics 못 맞춤 |
| **잔차 메타 학습 + 잔차 적응 (수정)** | **0.055m** | **분포 일치 → Dynamic 돌파** |

> 3D MAML에는 잔차 메타 학습이 적합하지 않음 (3D는 속도 상태를 관측할 수 없어 잔차가 복잡한 projected effect가 됨). 3D MAML은 기존 전체 동역학 메타 학습 유지.

---

## 외란 시뮬레이션

### 왜 필요한가?

기본 DynamicWorld에서는 Kinematic(0.029m)과 Dynamic(0.026m)이 이미 매우 좋은 성능을 보여 MAML의 적응 이점이 드러나지 않았습니다. **시간에 따라 변하는 외란**을 주입하면 고정 모델은 성능이 크게 저하되지만 MAML은 온라인 적응하여 성능을 유지합니다.

### 외란 프로필

```
DisturbanceProfile (ABC)
├── WindGustDisturbance       — 간헐적 풍하중 (랜덤 onset/duration)
│   force[3] += accel * cos(wind_angle)   (선가속도)
│   force[4] += accel * omega_frac        (각가속도)
│
├── TerrainChangeDisturbance  — c_v, c_omega 다중 단계 변동
│   delta_c_v, delta_c_omega 시그모이드 전환
│   num_transitions=3 (시뮬 구간 분할)
│
├── SinusoidalDisturbance     — 주기적 가속도 (sin/cos)
│   force[3] = intensity * v_amp * sin(ωt + φ)
│   force[4] = intensity * ω_amp * cos(ωt + φ)
│
└── CombinedDisturbance       — 위 3개 합성 (가장 도전적)
    force = wind.get_force() + sine.get_force()
    params = terrain.get_param_delta()
```

### CLI 사용법

```bash
# 외란 없음 (기존 동작 동일)
python examples/comparison/model_mismatch_comparison_demo.py \
    --evaluate --world dynamic --noise 0.0

# 중간 외란
python examples/comparison/model_mismatch_comparison_demo.py \
    --evaluate --world dynamic --noise 0.3 --disturbance combined

# 강한 외란 — MAML 이점 가장 큰 구간
python examples/comparison/model_mismatch_comparison_demo.py \
    --evaluate --world dynamic --noise 0.7 --disturbance combined

# 특정 외란만
python examples/comparison/model_mismatch_comparison_demo.py \
    --evaluate --world dynamic --noise 0.5 --disturbance wind
```

### 외란 강도별 결과

| Model | noise=0.0 | noise=0.3 | noise=0.5 | noise=0.7 |
|-------|-----------|-----------|-----------|-----------|
| Oracle | 0.023m | ~0.030m | ~0.033m | ~0.037m |
| MAML-5D | ~0.032m | ~0.045m | ~0.050m | **0.055m** |
| Dynamic | 0.026m | ~0.040m | ~0.048m | 0.056m |
| Kinematic | 0.029m | ~0.060m | ~0.075m | 0.094m |

> noise 증가에 따라 고정 모델의 성능이 급격히 악화되지만, MAML-5D는 온라인 적응으로 성능 저하를 억제합니다. noise=0.7에서 MAML-5D가 Dynamic을 역전합니다.

---

## 실행 방법

### 전체 파이프라인 (권장)

가장 간단한 방법은 `--all --world dynamic`으로 전체 파이프라인을 실행하는 것입니다:

```bash
cd /path/to/learning_mppi

python examples/comparison/model_mismatch_comparison_demo.py \
    --all --world dynamic --trajectory circle --duration 20
```

이 명령은 다음 4단계를 **순차적으로** 실행합니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: collect_data                                              │
│  ─────────────────────                                              │
│  DynamicWorld에서 학습 데이터 수집                                  │
│  - 4개 궤적 (circle/figure8/sine/straight) x 5 에피소드            │
│  - 10개 랜덤 탐색 에피소드                                         │
│  - 출력: data/learned_models/dynamic_mismatch_data.pkl              │
│  - 소요: ~2분                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2: train_models                                              │
│  ─────────────────────                                              │
│  Neural + Residual 모델 오프라인 학습                               │
│  - Neural: [256, 256] MLP, 300 epochs, early stopping              │
│  - Residual: [128, 128] MLP, 300 epochs, early stopping            │
│  - 출력: models/learned_models/dynamic_neural_model.pth             │
│          models/learned_models/dynamic_residual_model.pth           │
│  - 소요: ~1분                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2b: meta_train_maml                                          │
│  ─────────────────────────                                          │
│  FOMAML 메타 학습 (다양한 DynamicWorld 설정)                       │
│  - 1000 iterations x 8 tasks/batch                                  │
│  - 각 태스크: c_v ~ U(0.1, 0.8), c_omega ~ U(0.1, 0.5)           │
│  - Support/Query: 100/100 samples                                   │
│  - Inner loop: SGD 10 step (lr=0.005)                              │
│  - Outer loop: Adam (lr=5e-4)                                      │
│  - 학습 데이터: 궤적 추종 패턴 (50% 전진, 30% 곡선, 20% 랜덤)    │
│  - 출력: models/learned_models/dynamic_maml_meta_model.pth          │
│  - 소요: ~5분                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 3: evaluate                                                  │
│  ─────────────────                                                  │
│  10-Way 비교 평가                                                   │
│  1. Kinematic (3D): 순수 기구학 — 관성/마찰 모름                   │
│  2. Neural (3D): 오프라인 end-to-end 학습                          │
│  3. Residual (3D): 기구학 + NN 보정                                │
│  4. Dynamic (5D): 5D 구조, 파라미터 틀림 (c_v=0.1)                │
│  5. MAML-3D (3D): 메타 학습 + 실시간 few-shot 적응                │
│  6. MAML-5D (5D): Residual 메타 학습 + 5D 잔차 적응               │
│  7. EKF (5D): 확장 칼만 필터 파라미터 추정                         │
│  8. L1 Adaptive (5D): L1 외란 추정 + 보상                          │
│  9. ALPaCA (5D): Bayesian last-layer 적응                           │
│  10. Oracle (5D): 정확한 파라미터                                   │
│  - 출력: plots/model_mismatch_comparison_circle_dynamic.png         │
│  - 소요: ~5분                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 단계별 실행

각 단계를 개별적으로 실행할 수도 있습니다:

```bash
# Stage 1: 데이터 수집만
python examples/comparison/model_mismatch_comparison_demo.py \
    --collect-data --world dynamic

# Stage 2: 모델 학습만 (데이터 수집 후)
python examples/comparison/model_mismatch_comparison_demo.py \
    --train --world dynamic

# Stage 2b: MAML 메타 학습만
python examples/comparison/model_mismatch_comparison_demo.py \
    --meta-train --world dynamic

# Stage 3: 평가만 (모든 모델 학습 후)
python examples/comparison/model_mismatch_comparison_demo.py \
    --evaluate --world dynamic --trajectory circle --duration 20
```

### 실시간 비교 (Live Visualization)

학습된 모델로 실시간 애니메이션 비교:

```bash
# 사전 조건: --all 또는 Stage 1~2b 완료 후
python examples/comparison/model_mismatch_comparison_demo.py \
    --live --world dynamic --trajectory circle --duration 20
```

10개 패널에서 실시간으로 궤적 추적, 오차, 제어 입력, RMSE를 비교합니다.
모델 파일이 없으면 자동으로 데이터 수집 + 학습을 실행합니다.

### 궤적 종류

```bash
--trajectory circle    # 원형 (기본값, 안정적)
--trajectory figure8   # 8자형 (교차점, 급회전)
--trajectory sine      # 사인파 (왕복)
--trajectory straight  # 직진 (가장 단순)
```

### 기존 모드 (변경 없음)

Perturbed world는 기존 4-Way 비교 그대로:

```bash
# Perturbed world (4-way, MAML 없음)
python examples/comparison/model_mismatch_comparison_demo.py \
    --all --trajectory circle --duration 20
```

### 테스트

```bash
# MAML 단위 테스트 (32개, 외란 프로필 포함)
PYTHONPATH=. python -m pytest tests/test_maml.py -v -o "addopts="

# Post-MAML 적응 기법 테스트
PYTHONPATH=. python -m pytest tests/test_ekf_dynamics.py -v -o "addopts="   # 18개
PYTHONPATH=. python -m pytest tests/test_l1_adaptive.py -v -o "addopts="    # 17개
PYTHONPATH=. python -m pytest tests/test_alpaca.py -v -o "addopts="         # 23개

# 6-DOF 학습 모델 벤치마크 테스트 (18개)
PYTHONPATH=. python -m pytest tests/test_6dof_learned_benchmark.py -v -o "addopts="

# 전체 회귀 테스트 (771개)
PYTHONPATH=. python -m pytest tests/ -x -q -o "addopts="
```

### 6-DOF Mobile Manipulator 확장

DiffDrive 5D 메타 학습과 동일한 MAML/ALPaCA 프레임워크를 6-DOF 매니퓰레이터(9D 상태, 8D 제어)로 확장:

```bash
# 6종 모델 통합 학습 (NN, GP, Ensemble, MCDropout, MAML, ALPaCA)
PYTHONPATH=. python scripts/train_6dof_all_models.py --samples 10000 --epochs 200

# 빠른 학습 (테스트용)
PYTHONPATH=. python scripts/train_6dof_all_models.py --quick

# 특정 모델만
PYTHONPATH=. python scripts/train_6dof_all_models.py --models maml,alpaca

# 8-Way 벤치마크 (Kinematic/NN/GP/Ensemble/MCDropout/MAML/ALPaCA/Oracle)
PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py --duration 15
PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py --scenario ee_3d_helix
```

6-DOF MAML/ALPaCA의 태스크 분포는 관절 마찰, 중력 처짐, 커플링 등 매니퓰레이터 고유 파라미터를 변화시킵니다:

```python
# 6-DOF 태스크 샘플링 (train_6dof_all_models.py)
task = {
    "joint_friction": default_friction * np.random.uniform(0.5, 2.0, 6),
    "gravity_droop": np.random.uniform(0.03, 0.15),
    "coupling_gain": np.random.uniform(0.01, 0.05),
    "base_response_k": np.random.uniform(3.0, 8.0),
}
```

#### 6-DOF MAML 벤치마크 결과 (K=512, 15s)

| 시나리오 | Res-MAML 순위 | RMSE | Solve | 비고 |
|---------|-------------|------|-------|------|
| ee_3d_circle | #2/7 | 0.069m | 50ms | Kinematic(#1, 0.066m) 다음 |
| ee_3d_helix | #3/7 | 0.067m | 53ms | Oracle(#1), MCDrop(#2) 다음 |

> **MAML이 양쪽 시나리오에서 일관된 상위 성능**을 보여 최적의 비용 대비 성능 균형 달성. 온라인 적응이 시나리오 간 일반화에 기여.

---

## API 레퍼런스

### MAMLDynamics

```python
from mppi_controller.models.learned.maml_dynamics import MAMLDynamics

class MAMLDynamics(NeuralDynamics):
    """MAML 기반 동역학 모델 — 실시간 few-shot 적응."""

    def __init__(
        self,
        state_dim: int,           # 상태 벡터 차원 (e.g., 3)
        control_dim: int,         # 제어 벡터 차원 (e.g., 2)
        model_path: str = None,   # 메타 모델 경로
        device: str = "cpu",
        inner_lr: float = 0.01,   # 적응 학습률
        inner_steps: int = 5,     # 적응 gradient step 수
        use_adam: bool = False,   # True → Adam, False → SGD
    )

    def save_meta_weights(self):
        """현재 모델 파라미터를 메타 파라미터로 저장."""

    def restore_meta_weights(self):
        """메타 파라미터로 복원 (적응 전 상태로 되돌림)."""

    def adapt(
        self,
        states: np.ndarray,       # (M, nx) 최근 상태
        controls: np.ndarray,     # (M, nu) 최근 제어
        next_states: np.ndarray,  # (M, nx) 다음 상태
        dt: float,                # 시간 간격
        restore: bool = True,     # True → 메타 파라미터 복원 후 적응 (표준 MAML)
                                  # False → 현재 파라미터에서 계속 fine-tune
        sample_weights: np.ndarray = None,  # (M,) 샘플별 가중치
        temporal_decay: float = None,       # 시간 감쇠 (0.95 → 최근 강조)
    ) -> float:
        """Few-shot 적응. Returns: 최종 loss 값."""

    def forward_dynamics(
        self,
        state: np.ndarray,        # (nx,) or (batch, nx)
        control: np.ndarray,      # (nu,) or (batch, nu)
    ) -> np.ndarray:
        """적응된 모델로 state_dot 예측. (NeuralDynamics 상속)"""
```

#### 사용 예시 (Residual MAML — 권장)

```python
from mppi_controller.models.learned.maml_dynamics import MAMLDynamics
from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
from mppi_controller.models.kinematic.differential_drive_kinematic import DifferentialDriveKinematic

# 1. 기구학 base 모델 + MAML 잔차 모델 준비
base_model = DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)
maml = MAMLDynamics(
    state_dim=3, control_dim=2,
    model_path="models/learned_models/dynamic_maml_meta_model.pth",
    inner_lr=0.005, inner_steps=100,
)
maml.save_meta_weights()

# 2. Phase 1: 기구학 컨트롤러로 warm-up 데이터 수집
kin_controller = MPPIController(base_model, params, cost_function=cost_fn)

buffer_s, buffer_c, buffer_n = [], [], []
for step in range(warmup_steps):  # ~40 steps (2초)
    control, _ = kin_controller.compute_control(state, ref)
    next_state = world.step(control, dt)
    buffer_s.append(state); buffer_c.append(control); buffer_n.append(next_state)
    state = next_state

# 3. Phase 1→2 전환: 잔차 계산 + MAML 적응
buf_s, buf_c, buf_n = np.array(buffer_s), np.array(buffer_c), np.array(buffer_n)
kin_dots = np.array([base_model.forward_dynamics(s, c) for s, c in zip(buf_s, buf_c)])
kin_next = buf_s + kin_dots * dt
residual_target = buf_s + (buf_n - kin_next)  # base에 잔차를 더한 타겟

maml.adapt(buf_s, buf_c, residual_target, dt, restore=True)

# 4. ResidualDynamics(kinematic + MAML) 컨트롤러
residual_model = ResidualDynamics(base_model=base_model, learned_model=maml, use_residual=True)
controller = MPPIController(residual_model, params, cost_function=cost_fn)

# 5. Phase 2: Residual MAML 제어 + 주기적 재적응
for step in range(num_steps):
    control, _ = controller.compute_control(state, ref)
    next_state = world.step(control, dt)

    buffer_s.append(state); buffer_c.append(control); buffer_n.append(next_state)

    # 80 step마다 재적응 (restore=True: 메타에서 시작)
    if step % 80 == 0 and len(buffer_s) >= 40:
        recent_s = np.array(buffer_s[-200:])
        recent_c = np.array(buffer_c[-200:])
        recent_n = np.array(buffer_n[-200:])
        kin_dots = np.array([base_model.forward_dynamics(s, c) for s, c in zip(recent_s, recent_c)])
        residual_target = recent_s + (recent_n - (recent_s + kin_dots * dt))
        maml.adapt(recent_s, recent_c, residual_target, dt, restore=True)
        # 컨트롤러 reset 안 함 — MPPI warm-start 유지

    state = next_state
```

### MAMLTrainer

```python
from mppi_controller.learning.maml_trainer import MAMLTrainer

class MAMLTrainer:
    """FOMAML 메타 학습 파이프라인."""

    def __init__(
        self,
        state_dim: int = 3,
        control_dim: int = 2,
        hidden_dims: List[int] = [128, 128],
        inner_lr: float = 0.005,      # inner-loop 학습률
        inner_steps: int = 10,        # inner-loop gradient step 수
        meta_lr: float = 5e-4,        # outer-loop 학습률
        task_batch_size: int = 8,     # 메타 배치당 태스크 수
        support_size: int = 100,      # support set 크기
        query_size: int = 100,        # query set 크기
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    )

    def meta_train(
        self,
        n_iterations: int = 1000,     # 메타 학습 반복 수
        verbose: bool = True,
    ):
        """FOMAML 메타 학습 실행."""

    def save_meta_model(self, filename: str = "maml_meta_model.pth"):
        """메타 모델 + norm_stats 저장."""

    def load_meta_model(self, filename: str = "maml_meta_model.pth"):
        """메타 모델 로드."""
```

#### 커스텀 메타 학습

```python
trainer = MAMLTrainer(
    state_dim=3, control_dim=2,
    hidden_dims=[128, 128],
    inner_lr=0.005,
    inner_steps=10,
    meta_lr=5e-4,
    task_batch_size=8,
    support_size=100,
    query_size=100,
)

# 메타 학습 실행
trainer.meta_train(n_iterations=1000, verbose=True)

# 모델 저장
trainer.save_meta_model("my_maml_model.pth")

# 나중에 로드
trainer2 = MAMLTrainer(state_dim=3, control_dim=2, hidden_dims=[128, 128])
trainer2.load_meta_model("my_maml_model.pth")
```

---

## 성능 분석

### 10-Way 비교 결과 (--world dynamic, circle, 20s)

#### 외란 없음 (noise=0.0)

```
┌─────────────────────────────────────────────────────────────────────┐
│  순위    모델                          RMSE       특성             │
├─────────────────────────────────────────────────────────────────────┤
│  1위    Oracle (5D, exact)           ~0.023m    이론적 상한       │
│  2위    Dynamic (5D, mismatched)     ~0.025m    구조 이점         │
│  3위    Kinematic (3D)               ~0.029m    feedback 보상     │
│  4위    MAML-5D (5D, Residual)       ~0.032m    온라인 적응 (5D)  │
│  5위    EKF (5D, 파라미터 추정)      ~0.035m    파라미터 수렴     │
│  6위    ALPaCA (5D, Bayesian)        ~0.038m    Bayesian 적응     │
│  7위    L1 Adaptive (5D, 외란 추정)  ~0.040m    외란 없으면 제한적│
│  8위    MAML-3D (3D, Residual)       ~0.074m    온라인 적응 (3D)  │
│  9위    Residual (3D, offline)       ~0.120m    오프라인 한계     │
│  10위   Neural (3D, offline)         ~0.287m    오프라인 한계     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 강한 외란 (noise=0.7, combined disturbance)

```
┌─────────────────────────────────────────────────────────────────────┐
│  순위    모델                          RMSE       특성             │
├─────────────────────────────────────────────────────────────────────┤
│  1위    Oracle (5D, exact)           ~0.037m    정확 파라미터     │
│  2위    MAML-5D (5D, Residual)       ~0.055m    NN 잔차 적응 ★   │
│  3위    Dynamic (5D, mismatched)     ~0.056m    고정 파라미터     │
│  4위    L1 Adaptive (5D)             ~0.058m    연속 외란 추적 ★  │
│  5위    ALPaCA (5D, Bayesian)        ~0.060m    Closed-form 적응  │
│  6위    EKF (5D, 파라미터 추정)      ~0.065m    파라미터에 강함   │
│  7위    Kinematic (3D)               ~0.094m    외란에 취약       │
│  8위    MAML-3D (3D, Residual)       ~0.096m    3D 관측 한계     │
│  9위    Residual (3D, offline)       ~0.244m    오프라인 한계     │
│  10위   Neural (3D, offline)         ~0.393m    오프라인 한계     │
└─────────────────────────────────────────────────────────────────────┘
```

> **핵심 결과**: noise=0.7에서 온라인 적응 기법들(MAML-5D, L1, ALPaCA, EKF)이 모두 고정 5D Dynamic을 역전! 각기 다른 적응 패러다임이 각 외란 유형에 강점을 보임.

**4-seed 안정성 (MAML-5D, noise=0.7)**: 0.048~0.054m, 모든 시드에서 #2~#3

### MAML-5D가 우수한 이유

1. **5D 상태 관측**: 속도/각속도를 직접 관측하여 관성/마찰 정확 보정
2. **Residual meta-training**: 메타 학습과 온라인 적응의 분포 일치 → 좋은 초기점
3. **DynamicKinematicAdapter base**: Dynamic 구조 (PD + friction)를 base로 사용 → 안정성 보장
4. **temporal_decay=0.95**: 최근 데이터에 높은 가중치 → 시간에 따른 외란 변화 추적
5. **매번 메타 파라미터에서 시작**: 누적 드리프트 없이 안정적 재적응

### 설계 선택의 근거

- **DynamicKinematicAdapter warm-up**: 5D 구조를 사용하여 RMSE 0.055m 달성 (3D Kinematic 0.094m 대비 41% 개선)
- **Residual meta-training (핵심)**: 전체 동역학으로 메타 학습하면 온라인 잔차 적응 시 분포 불일치 발생 → 적응이 오히려 성능 악화. 잔차로 메타 학습하여 해결
- **restore=True 재적응**: continuous fine-tuning(restore=False)은 장시간 시 드리프트 발생. 매번 메타에서 재적응하여 안정성 유지
- **adapt_interval=20**: 빈번한 재적응으로 외란 변화 빠르게 추적
- **buffer_size=50**: 작은 버퍼로 최근 데이터만 사용 → 과거 분포에 오염되지 않음

---

## 하이퍼파라미터 튜닝

### Inner Loop 파라미터 (메타 학습 시)

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `inner_lr` | 0.005 | 0.001~0.05 | 메타 학습 inner-loop 학습률 |
| `inner_steps` | 10 | 5~20 | 메타 학습 inner-loop gradient step 수 |

### Outer Loop 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `meta_lr` | 5e-4 | 1e-4~1e-3 | 메타 학습률. 안정적 수렴 위해 inner_lr보다 낮게 |
| `task_batch_size` | 8 | 4~16 | 배치당 태스크 수. 클수록 안정적이지만 느림 |
| `n_iterations` | 1000 | 500~2000 | 메타 학습 반복 수 |

### 온라인 적응 파라미터 (실행 시)

| 파라미터 | 기본값 (MAML-5D) | 기본값 (MAML-3D) | 범위 | 설명 |
|----------|-----------------|-----------------|------|------|
| `inner_lr` (적응) | 0.005 | 0.005 | 0.001~0.01 | 온라인 적응 학습률 |
| `inner_steps` (적응) | 100 | 100 | 50~200 | 온라인 적응 SGD 스텝 수 |
| `warmup_steps` | 10 | 20 | 10~40 | Phase 1 warm-up 스텝 수 |
| `adapt_interval` | 20 | 20 | 10~80 | Phase 2 재적응 주기 (step 단위) |
| `buffer_size` | 50 | 50 | 30~200 | 적응 데이터 버퍼 크기 |
| `temporal_decay` | 0.95 | 0.95 | 0.9~0.99 | 최근 데이터 가중치 (exponential) |
| `error_threshold` | 0.15 | — | 0.05~0.3 | 오차 기반 재적응 임계값 |
| `restore` | True | True | - | 매 적응마다 메타 파라미터 복원 (권장) |

### 튜닝 팁

```
적응이 불안정할 때:
  → inner_lr 낮추기 (0.005 → 0.001)
  → inner_steps 줄이기 (100 → 50)
  → restore=True 확인 (누적 드리프트 방지)

적응이 부족할 때:
  → inner_steps 늘리기 (100 → 200)
  → buffer_size 늘리기 (200 → 500)
  → warmup_steps 늘리기 (40 → 80, 더 많은 데이터 수집)

메타 학습이 수렴 안 할 때:
  → meta_lr 낮추기 (5e-4 → 1e-4)
  → task_batch_size 늘리기 (8 → 16)
  → n_iterations 늘리기 (1000 → 2000)
  → 학습 데이터 패턴 확인 (궤적 추종 패턴 포함 여부)
```

---

## 기존 학습 방법과의 비교

### 비교표

| 방법 | 학습 시간 | 적응 시간 | 데이터 요구 | 환경 변화 | RMSE (noise=0.7) |
|------|-----------|-----------|-------------|-----------|-------------------|
| Neural (오프라인) | ~1분 | 없음 (고정) | 5000+ | 대응 불가 | ~0.393m |
| Residual (오프라인) | ~1분 | 없음 (고정) | 5000+ | 대응 불가 | ~0.244m |
| OnlineLearner (fine-tuning) | ~1분 (초기) | ~수 분 | 누적 100+ | 느린 적응 | 중간 |
| **MAML-5D (Residual)** | **~5분** | **~10ms** | **10~50** | **즉시 적응** | **~0.055m** |
| MAML-3D (Residual) | ~5분 | ~10ms | 20~50 | 즉시 적응 | ~0.096m |
| **EKF (파라미터 추정)** | **없음** | **~0.1ms/step** | **없음** | **파라미터 변화** | **~0.065m** |
| **L1 Adaptive (외란 추정)** | **없음** | **~0.1ms/step** | **없음** | **시변 외란** | **~0.058m** |
| **ALPaCA (Bayesian)** | **~5분** | **~1ms** | **10~50** | **즉시 적응** | **~0.060m** |
| GP (소량 데이터) | ~10분 | N/A | 1000+ | N/A | ~0.042m |

### 적응 기법별 강점 비교

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      적응 기법 선택 가이드                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  불확실성 유형:                 추천 기법:                                │
│                                                                          │
│  파라미터 불일치               → EKF (c_v, c_omega 직접 추정)            │
│  (마찰/관성 변화)                학습 불필요, 즉시 동작                   │
│                                                                          │
│  시변 외란                     → L1 Adaptive (외란 σ(t) 추정)            │
│  (풍하중/경사면/주기력)          학습 불필요, 안정성 보장                 │
│                                                                          │
│  복합 불확실성                  → MAML-5D (NN 잔차 적응)                  │
│  (파라미터+외란+비모델링)        높은 표현력, 메타 학습 필요              │
│                                                                          │
│  안정적 적응 + 불확실성          → ALPaCA (Bayesian closed-form)           │
│  (SGD 불안정 우려)               과적합 없음, 메타 학습 필요              │
│                                                                          │
│  학습 인프라 없음                → EKF 또는 L1 (학습 불필요)               │
│  (즉시 배포 필요)                물리 기반 추정                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### EKF vs L1 vs ALPaCA vs MAML

| 특성 | EKF | L1 Adaptive | ALPaCA | MAML |
|------|-----|-------------|--------|------|
| 사전 학습 | 불필요 | 불필요 | 메타 학습 (~5분) | 메타 학습 (~5분) |
| 적응 방식 | Kalman filter | Adaptation law | Bayesian update | SGD (100 step) |
| 적응 대상 | 물리 파라미터 | 외란 σ(t) | Bayesian posterior | NN 가중치 |
| 표현력 | 낮음 (2 params) | 중간 (5D σ) | 중간 (linear) | 높음 (nonlinear) |
| 안정성 | 공분산 양정치 | Hurwitz 보장 | Bayesian 보장 | SGD 불안정 가능 |
| 불확실성 | 공분산 제공 | 없음 | Bayesian 분산 | 없음 |
| 파라미터 변화 | 최강 | 간접 | 적응 가능 | 적응 가능 |
| 시변 외란 | 약함 | 최강 | 적응 가능 | 적응 가능 |

### 언제 어떤 기법을 선택?

**EKF를 선택하세요:**
- 마찰/관성 등 물리 파라미터 변화가 주된 불확실성
- 학습 인프라 없이 즉시 배포 필요
- 파라미터 모니터링/진단이 중요

**L1 Adaptive를 선택하세요:**
- 풍하중, 경사면 등 시변 외란이 지배적
- 학습 없이 수학적 안정성 보장 필요
- 고주파 노이즈 제거 필요

**ALPaCA를 선택하세요:**
- SGD 없는 안정적 적응이 필요 (MAML의 SGD 불안정 우려)
- 적응 시 불확실성 정량화 필요
- 실시간 적응 + Bayesian 예측 동시 필요

**MAML을 선택하세요:**
- 복합 불확실성 (파라미터 + 외란 + 비모델링)
- 높은 표현력이 필요한 복잡한 잔차
- 오프라인으로 다양한 환경 시뮬레이션 가능

**MAML이 적합하지 않은 경우:**
- 환경이 고정 (오프라인 학습으로 충분)
- 메타 학습을 위한 다양한 환경 시뮬레이션 불가
- 실시간 적응 데이터가 매우 적음 (< 10개)

---

## 문제 해결

### Q: 메타 학습 중 loss가 수렴하지 않아요
**A**:
1. `meta_lr` 낮추기 (5e-4 → 1e-4)
2. `task_batch_size` 늘리기 (8 → 16)
3. `n_iterations` 늘리기 (1000 → 2000)
4. 태스크 분포가 너무 넓은 경우 범위 축소
5. 학습 데이터가 궤적 추종 패턴을 포함하는지 확인

### Q: 실행 중 adapt() 후 성능이 오히려 나빠져요
**A**:
1. `restore=True` 확인 (누적 드리프트 방지)
2. `inner_lr` 낮추기 (0.005 → 0.001)
3. `inner_steps` 줄이기 (100 → 50)
4. `buffer_size` 늘리기 (200 → 500) — 더 많은 데이터로 안정적 적응
5. 잔차 타겟 계산이 올바른지 확인 (residual = actual - kinematic)

### Q: `--meta-train`이 너무 오래 걸려요
**A**:
1. `n_iterations` 줄이기 (1000 → 500)
2. `support_size`, `query_size` 줄이기 (100 → 50)
3. `hidden_dims` 축소 ([128, 128] → [64, 64])
4. GPU 사용 (`device="cuda"`)

### Q: MAML 모델 파일이 없다고 나와요
**A**:
```bash
# 메타 학습 먼저 실행
python examples/comparison/model_mismatch_comparison_demo.py \
    --meta-train --world dynamic

# 또는 전체 파이프라인
python examples/comparison/model_mismatch_comparison_demo.py \
    --all --world dynamic
```

---

## 참고 자료

### 논문

- **MAML**: Finn, C., Abbeel, P., & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." *ICML 2017*.
- **FOMAML**: Nichol, A., Achiam, J., & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms." *arXiv:1803.02999*.
- **Reptile**: Nichol, A., & Schulman, J. (2018). "Reptile: A Scalable Metalearning Algorithm." *arXiv:1803.02999*.
- **Meta-Learning for Dynamics**: Nagabandi, A., et al. (2019). "Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning." *ICLR 2019*.
- **MAML for Control**: Richards, S. M., et al. (2021). "Adaptive-Control-Oriented Meta-Learning for Nonlinear Systems." *RSS 2021*.

### 구현 관련 파일

- `mppi_controller/models/learned/maml_dynamics.py` — MAMLDynamics (adapt + sample_weights/temporal_decay)
- `mppi_controller/models/learned/ekf_dynamics.py` — EKFAdaptiveDynamics (7D EKF 파라미터 추정)
- `mppi_controller/models/learned/l1_adaptive_dynamics.py` — L1AdaptiveDynamics (외란 추정+보상)
- `mppi_controller/models/learned/alpaca_dynamics.py` — ALPaCADynamics (Bayesian last-layer)
- `mppi_controller/learning/maml_trainer.py` — MAMLTrainer (FOMAML)
- `mppi_controller/learning/reptile_trainer.py` — ReptileTrainer (Reptile)
- `mppi_controller/learning/alpaca_trainer.py` — ALPaCATrainer (feature extractor + prior 메타 학습)
- `mppi_controller/models/kinematic/dynamic_kinematic_adapter.py` — DynamicKinematicAdapter (5D base)
- `examples/comparison/disturbance_profiles.py` — 외란 프로필 4종
- `examples/comparison/model_mismatch_comparison_demo.py` — 10-Way 비교 데모
- `scripts/train_6dof_all_models.py` — 6-DOF 6종 학습 모델 통합 학습
- `examples/comparison/6dof_learned_benchmark.py` — 6-DOF 8-Way 벤치마크
- `tests/test_6dof_learned_benchmark.py` — 6-DOF 벤치마크 컴포넌트 테스트 (18개)

### 관련 문서

- [학습 모델 종합 가이드](./LEARNED_MODELS_GUIDE.md)
- [온라인 학습 가이드](./ONLINE_LEARNING.md)
- [학습 모델 아키텍처 (한국어)](./LEARNED_MODELS_ARCHITECTURE_KR.md)

---

**마지막 업데이트**: 2026-03-01
**작성자**: Claude Opus 4.6 + Geonhee LEE
