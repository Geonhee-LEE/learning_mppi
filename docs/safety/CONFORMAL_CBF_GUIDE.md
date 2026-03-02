# Conformal Prediction + CBF-MPPI Guide

모델 예측 불확실성을 정량화하여 CBF 안전 마진을 동적으로 조절하는 기술 가이드.

## Table of Contents

1. [Motivation](#1-motivation)
2. [Conformal Prediction Theory](#2-conformal-prediction-theory)
3. [Adaptive Conformal Prediction (ACP)](#3-adaptive-conformal-prediction-acp)
4. [CBF-MPPI Integration](#4-cbf-mppi-integration)
5. [Algorithm](#5-algorithm)
6. [Architecture](#6-architecture)
7. [Parameter Tuning Guide](#7-parameter-tuning-guide)
8. [Benchmark Results](#8-benchmark-results)
9. [API Reference](#9-api-reference)
10. [References](#10-references)

---

## 1. Motivation

### 1.1 고정 안전 마진의 한계

기존 CBF-MPPI는 **고정 안전 마진**(`safety_margin`)으로 장애물의 유효 반경을 확장합니다:

```
h(x) = ||p - p_obs||^2 - (r_obs + margin)^2
```

이 고정 마진은 근본적인 **안전 vs 성능 딜레마**를 초래합니다:

```
┌─────────────────────────────────────────────────────────────┐
│  고정 마진의 딜레마                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  작은 마진 (0.01m):                                        │
│    ✓ 좋은 궤적 추적 (낮은 RMSE)                            │
│    ✗ 모델 불확실성에 취약 → 충돌 위험                      │
│                                                             │
│  큰 마진 (0.10m):                                          │
│    ✓ 높은 안전율 (충돌 거의 없음)                          │
│    ✗ 불필요한 보수성 → 추적 성능 저하                      │
│    ✗ 좁은 공간에서 통행 불가                               │
│                                                             │
│  → 환경이 변하면 최적 마진도 변함                          │
│  → 고정 마진은 어떤 환경에서든 차선(suboptimal)            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 CP의 해결 방식

Conformal Prediction(CP)은 모델 예측 품질을 **온라인으로 추적**하여 **필요한 만큼만** 마진을 자동 조절합니다:

```
┌─────────────────────────────────────────────────────────────┐
│  CP 동적 마진의 장점                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  정확한 모델 (예측 오차 작음):                              │
│    → CP 마진 축소 (~0.01m) → 공격적 마진 수준 추적         │
│                                                             │
│  부정확한 모델 (예측 오차 큼):                              │
│    → CP 마진 확대 (~0.07m) → 보수적 마진 수준 안전         │
│                                                             │
│  환경 변화 시:                                              │
│    → CP가 자동 적응 → 수동 튜닝 불필요                     │
│                                                             │
│  핵심: 분포-무관(distribution-free) 커버리지 보장           │
│        P(actual ∈ prediction region) ≥ 1 - α               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Conformal Prediction Theory

### 2.1 비순응 점수 (Nonconformity Score)

모델 예측과 실제 관측의 차이를 정량화하는 스칼라 값:

```
s_t = score(predicted_t, actual_t)
```

**지원하는 점수 유형:**

| Score Type | 수식 | 적용 상황 |
|---|---|---|
| `position_norm` | `\|\|actual[:2] - predicted[:2]\|\|` | 위치 추적 (기본값) |
| `full_state_norm` | `\|\|actual - predicted\|\|` | 전체 상태 (위치+방향) |
| `per_dim_max` | `max(\|actual_i - predicted_i\|)` | 차원별 최악 오차 |

**`position_norm`을 기본으로 사용하는 이유:**
- CBF의 장애물 거리 계산이 xy 위치 기반
- 방향(θ) 오차는 안전 마진에 직접 영향하지 않음
- 노름 기반이므로 스칼라 비교가 직관적

### 2.2 표준 Conformal Prediction

슬라이딩 윈도우 내 비순응 점수의 분위수(quantile)로 마진을 결정:

```
주어진 점수 윈도우: S = {s_1, s_2, ..., s_n}
커버리지 목표: 1 - α (예: 0.9 = 90%)

마진 계산:
  level = min(1, (n+1)(1-α) / n)
  margin = quantile(S, level)
  margin = clip(margin, margin_min, margin_max)
```

**수학적 보장** (교환 가능성 가정):

```
P(s_{n+1} ≤ margin) ≥ 1 - α
```

이는 다음을 의미합니다:
- 미래 예측 오차가 `margin` 이하일 확률이 최소 `1-α`
- `margin`을 안전 마진으로 사용하면 충분한 확률적 커버리지 보장
- **어떤 분포에서든 성립** (분포-무관 보장)

### 2.3 유한 샘플 커버리지

n개 점수로 계산한 quantile의 커버리지 보장:

```
Exact coverage: P(s_{n+1} ≤ q) = ceil((n+1)(1-α)) / (n+1)

예시 (α=0.1, n=100):
  level = min(1, 101 × 0.9 / 100) = 0.909
  quantile(scores, 0.909) → 상위 ~9%를 제외한 최대값
  커버리지 ≥ 90%
```

---

## 3. Adaptive Conformal Prediction (ACP)

### 3.1 표준 CP의 한계

표준 CP는 윈도우 내 모든 점수에 **동일 가중치**를 부여합니다. 비정상(non-stationary) 환경에서:

```
시간  1-50:  오차 작음 (정확한 모델)
시간 51-100: 오차 큼   (환경 변화)

표준 CP: 이전의 작은 오차가 quantile을 낮게 유지
       → 환경 변화에 느리게 반응
```

### 3.2 지수 가중 Quantile

ACP는 최근 데이터에 **지수적으로 높은 가중치**를 부여:

```
가중치: w_i = γ^(n-1-i)    (i=0이 가장 오래된 점수)

γ = 0.95:
  1 스텝 전: w = 1.00
  10 스텝 전: w = 0.60
  50 스텝 전: w = 0.08  ← 거의 무시

γ = 1.00: 표준 CP (모든 점수 동일 가중치)
```

**가중 quantile 계산:**

```python
# 1. 가중치 계산
weights = [γ^(n-1-i) for i in range(n)]
weights /= sum(weights)  # 정규화

# 2. 점수 정렬
sorted_indices = argsort(scores)
sorted_scores = scores[sorted_indices]
sorted_weights = weights[sorted_indices]

# 3. 누적 가중치로 분위 결정
cumulative = cumsum(sorted_weights)
idx = searchsorted(cumulative, 1 - α)
margin = sorted_scores[idx]
```

### 3.3 γ 선택 가이드

| γ 값 | 유효 윈도우 | 특성 | 적용 상황 |
|---|---|---|---|
| 1.00 | 전체 윈도우 | 표준 CP, 느린 적응 | 정적 환경 |
| 0.98 | ~50 스텝 | 준-적응형 | 서서히 변하는 환경 |
| **0.95** | ~20 스텝 | **기본 추천** | 대부분의 동적 환경 |
| 0.90 | ~10 스텝 | 빠른 적응 | 급변하는 환경 |
| 0.80 | ~5 스텝 | 매우 빠른 적응 | 극단적 비정상 환경 |

유효 윈도우 ≈ 1 / (1 - γ). γ=0.95이면 최근 20 스텝이 지배적.

---

## 4. CBF-MPPI Integration

### 4.1 마진 주입 메커니즘

ConformalCBFMPPIController는 ShieldMPPIController를 상속하며, `compute_control()` 호출 전에 두 속성을 동적 갱신합니다:

```python
# 매 제어 스텝마다:
cp_margin = self.conformal_predictor.get_margin()

# 1. CBF 비용 함수의 마진 → 샘플링 시 장애물 유효 반경 결정
self.cbf_cost.safety_margin = cp_margin

# 2. Shield 파라미터의 마진 → per-step 해석적 클리핑 반경
self.cbf_params.cbf_safety_margin = cp_margin
```

**기존 ControlBarrierCost의 `safety_margin`이 mutable이므로**, 별도의 비용 클래스 없이 직접 갱신으로 동작합니다.

### 4.2 예측 함수 (prediction_fn)

CP 업데이트에 사용하는 예측 함수를 교체할 수 있습니다:

```python
# 기본: 명목 기구학 모델 (model.step)
controller = ConformalCBFMPPIController(model, params)
# → 예측 오차 = 순수 외란 (매우 작음, ~O(dt))

# 학습 모델 (NN, GP 등): 의미 있는 예측 불확실성
controller = ConformalCBFMPPIController(
    model, params,
    prediction_fn=lambda s, u: s + learned_model.forward(s, u) * dt
)
# → 예측 오차 = 학습 모델 오차 + 외란 (0.01-0.10m)
# → CP 마진이 의미 있게 변동 → 환경 적응 효과 극대화

# 앙상블 모델 (불확실성 직접 제공)
ensemble = EnsembleNeuralDynamics(...)
controller = ConformalCBFMPPIController(
    model, params,
    prediction_fn=lambda s, u: s + ensemble.forward_dynamics(s, u) * dt
)
```

**핵심 원리:** CP는 prediction_fn이 **얼마나 정확한지**를 추적합니다. 학습 모델이 부정확할수록 CP 마진이 커지고, 정확할수록 줄어듭니다.

### 4.3 제어 루프 흐름

```
┌─────────────────────────────────────────────────────────┐
│  ConformalCBFMPPIController.compute_control(state, ref) │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: CP 업데이트                                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │ if prev_prediction exists:                        │  │
│  │   score = ||prev_predicted[:2] - state[:2]||     │  │
│  │   conformal_predictor.update(prev_pred, state)   │  │
│  └───────────────────────────────────────────────────┘  │
│            ↓                                            │
│  Step 2: 동적 마진 갱신                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │ margin = conformal_predictor.get_margin()         │  │
│  │ cbf_cost.safety_margin = margin                   │  │
│  │ cbf_params.cbf_safety_margin = margin             │  │
│  └───────────────────────────────────────────────────┘  │
│            ↓                                            │
│  Step 3: Shield-MPPI 제어 (부모 클래스)                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │ control, info = super().compute_control(...)      │  │
│  │ (갱신된 margin으로 CBF 비용 계산 + Shield 클립)   │  │
│  └───────────────────────────────────────────────────┘  │
│            ↓                                            │
│  Step 4: 다음 스텝 예측 저장                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ if prediction_fn:                                 │  │
│  │   prev_pred = prediction_fn(state, control)       │  │
│  │ else:                                             │  │
│  │   prev_pred = model.step(state, control, dt)      │  │
│  └───────────────────────────────────────────────────┘  │
│            ↓                                            │
│  Step 5: CP 통계 info에 추가                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ info["cp_margin"] = margin                        │  │
│  │ info["cp_empirical_coverage"] = coverage          │  │
│  │ info["cp_num_scores"] = n_scores                  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  return control, info                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Algorithm

### 5.1 전체 의사코드

```
CONFORMAL-CBF-MPPI(state, reference):
  Input:
    state: 현재 로봇 상태 (x, y, θ)
    reference: N+1 스텝 레퍼런스 궤적
    prev_pred: 이전 스텝에서 저장한 예측 (없으면 None)

  // Phase 1: CP 업데이트
  IF prev_pred is not None AND cp_enabled:
    score = ||prev_pred[:2] - state[:2]||    // 비순응 점수
    scores_window.append(score)
    IF len(scores_window) >= min_samples:
      IF gamma < 1.0:
        margin = weighted_quantile(scores_window, γ, 1-α)
      ELSE:
        level = min(1, (n+1)(1-α)/n)
        margin = quantile(scores_window, level)
      margin = clip(margin, margin_min, margin_max)
    ELSE:
      margin = default_margin    // cold start

  // Phase 2: 마진 주입
  cbf_cost.safety_margin = margin
  cbf_params.cbf_safety_margin = margin

  // Phase 3: MPPI 샘플링 + 제어 계산
  // (ShieldMPPIController가 갱신된 margin으로 CBF 비용 계산)
  control, info = ShieldMPPI.compute_control(state, reference)

  // Phase 4: 다음 스텝 예측
  prev_pred = prediction_fn(state, control)

  return control, {info + cp_margin, cp_coverage, ...}
```

### 5.2 Warm-up 전략

CP는 `min_samples` 이전에는 `default_margin`을 사용합니다:

```
스텝 0 ~ min_samples-1:  margin = default_margin (cold start)
스텝 min_samples ~ :     margin = quantile 기반 (데이터 기반)
```

**Cold start 마진 선택:**
- 너무 크면: 초기 추적 성능 저하
- 너무 작으면: 초기 안전 위험
- 권장: `default_margin = 0.02` (빠르게 CP가 적응)

---

## 6. Architecture

### 6.1 클래스 계층

```
MPPIController (base_mppi.py)
  └── CBFMPPIController (cbf_mppi.py)
        └── ShieldMPPIController (shield_mppi.py)
              └── ConformalCBFMPPIController (conformal_cbf_mppi.py)  ← NEW
                    │
                    ├── ConformalPredictor (learning/conformal_predictor.py)
                    │     ├── ConformalPredictorConfig
                    │     ├── update(predicted, actual)
                    │     ├── get_margin() → float
                    │     └── get_statistics() → Dict
                    │
                    └── prediction_fn: Optional[Callable]
                          └── (state, control) → next_state
```

### 6.2 파일 구성

| 파일 | 역할 | LOC |
|---|---|---|
| `learning/conformal_predictor.py` | CP 알고리즘 핵심 모듈 | ~147 |
| `controllers/mppi/conformal_cbf_mppi.py` | CP+Shield-MPPI 컨트롤러 | ~143 |
| `controllers/mppi/mppi_params.py` | ConformalCBFMPPIParams 추가 | +27 |
| `tests/test_conformal_cbf.py` | 30개 테스트 | ~500 |
| `examples/comparison/conformal_cbf_benchmark.py` | 5-Way × 5 시나리오 벤치마크 | ~600 |

---

## 7. Parameter Tuning Guide

### 7.1 ConformalCBFMPPIParams

```python
@dataclass
class ConformalCBFMPPIParams(ShieldMPPIParams):
    cp_alpha: float = 0.1          # 실패율 (0.1 → 90% 커버리지)
    cp_window_size: int = 200      # 슬라이딩 윈도우 크기
    cp_min_samples: int = 10       # CP 활성화 최소 샘플 수
    cp_gamma: float = 0.95         # ACP 감쇠율 (1.0=표준CP)
    cp_margin_min: float = 0.02    # 최소 마진 클램프 (m)
    cp_margin_max: float = 0.5     # 최대 마진 클램프 (m)
    cp_score_type: str = "position_norm"
    cp_enabled: bool = True
```

### 7.2 파라미터별 상세 설명

#### `cp_alpha` — 실패율

```
α = 0.1  →  90% 커버리지 (기본, 대부분의 경우 충분)
α = 0.05 →  95% 커버리지 (더 보수적, 마진 ~10% 증가)
α = 0.2  →  80% 커버리지 (더 공격적, 마진 ~15% 감소)

권장: 0.1 (안전-성능 균형)
안전 최우선: 0.05
성능 최우선: 0.15-0.2
```

#### `cp_gamma` — ACP 감쇠율

```
γ = 1.00: 표준 CP. 윈도우 내 모든 데이터 동일 가중치.
          장점: 안정적. 단점: 환경 변화에 느림.

γ = 0.95: 기본 추천. 최근 ~20 스텝이 지배적.
          2초 내 환경 변화에 적응 (dt=0.05).

γ = 0.90: 빠른 적응. 최근 ~10 스텝이 지배적.
          주의: 단기 노이즈에도 민감.
```

#### `cp_margin_min` / `cp_margin_max` — 마진 클램프

```
margin_min: CP 마진의 하한. 모델이 완벽해도 이 이하로 내려가지 않음.
  0.005m: 매우 공격적 (정확한 모델 전용)
  0.02m:  일반적 (기본값)
  0.05m:  보수적

margin_max: CP 마진의 상한. 이상치(outlier) 방어.
  0.3m:   일반 로봇
  0.5m:   기본값
  1.0m:   대형 로봇
```

#### `cp_window_size` — 슬라이딩 윈도우

```
window_size = 200 (기본)
  20Hz × 10초 = 200 스텝의 최근 데이터 사용
  너무 작으면: 분산 높음, 마진 불안정
  너무 크면: 오래된 데이터가 적응 방해

경험적 권장: 제어 주파수 × 5-15초
  20Hz: 100-300
  50Hz: 250-750
```

### 7.3 시나리오별 권장 설정

| 시나리오 | α | γ | margin_min | margin_max | 이유 |
|---|---|---|---|---|---|
| 정적 환경 + 정확한 모델 | 0.1 | 1.0 | 0.005 | 0.3 | 안정적, 최소 마진 |
| 동적 장애물 | 0.1 | 0.95 | 0.02 | 0.5 | 빠른 적응 필요 |
| 좁은 통로 | 0.1 | 0.95 | 0.005 | 0.3 | 작은 마진이 중요 |
| 학습 모델 (NN/GP) | 0.1 | 0.95 | 0.02 | 0.5 | 큰 예측 오차 추적 |
| 비정상 환경 (바람) | 0.05 | 0.90 | 0.02 | 0.5 | 높은 커버리지 + 빠른 적응 |

---

## 8. Benchmark Results

### 8.1 5-Way × 5 시나리오 비교

**방법:**
1. Vanilla MPPI — 안전 제어 없음 (기준선)
2. CBF-MPPI (0.01m) — 공격적 고정 마진
3. CBF-MPPI (0.10m) — 보수적 고정 마진
4. CP-CBF-MPPI (γ=1.0) — 표준 Conformal Prediction
5. ACP-CBF-MPPI (γ=0.95) — 적응형 Conformal Prediction

**시나리오:**
1. **accurate** — 정확한 모델, 외란 없음
2. **mismatch** — 마찰 기반 모델 불일치 (friction=0.3)
3. **nonstationary** — 시변 바람 + t>4s 급변 + t>7s 역전
4. **dynamic** — 동적 장애물 (CrossingMotion + BouncingMotion) + 2-Phase 외란
5. **corridor** — L자 좁은 통로 (0.9m 폭) + 후반부 횡방향 바람

### 8.2 핵심 결과

#### Dynamic (동적 장애물 + 2-Phase 환경 변화)

| Method | RMSE | Safety | CP Margin |
|---|---|---|---|
| Vanilla MPPI | 0.270m | 95.4% | N/A |
| CBF (0.01m) | 0.297m | 99.3% | 0.010 (fixed) |
| CBF (0.10m) | 0.385m | **100%** | 0.100 (fixed) |
| CP-CBF (std) | 0.314m | **99.5%** | 0.062 (adaptive) |
| ACP-CBF (γ=0.95) | 0.382m | **100%** | 0.072 (adaptive) |

- **ACP**: CBF-large와 동일한 100% 안전, 더 좋은 RMSE
- **CP**: CBF-small보다 높은 안전율(99.5% vs 99.3%), 합리적 RMSE

#### Corridor (L자 좁은 통로 + 횡방향 바람)

| Method | RMSE | Safety | CP Margin |
|---|---|---|---|
| Vanilla MPPI | 0.334m | 89.9% | N/A |
| CBF (0.01m) | 0.337m | 94.0% | 0.010 (fixed) |
| CBF (0.10m) | 0.342m | 97.8% | 0.100 (fixed) |
| CP-CBF (std) | **0.331m** | **97.8%** | 0.071 (adaptive) |
| ACP-CBF (γ=0.95) | 0.339m | 97.3% | 0.075 (adaptive) |

- **CP-CBF**: **BEST RMSE** (0.331m) + CBF-large급 안전(97.8%)
- CBF-small 대비 안전율 3.8%p 향상 (94.0% → 97.8%)

### 8.3 CP 마진 적응 동작

```
accurate 시나리오:
  CP 마진: 0.049m (0.020-0.057)  ← 학습 모델 편향 추적
  → 100% 안전 + RMSE 0.074 (CBF-small~large 중간)

mismatch 시나리오:
  CP 마진: 0.071m (0.020-0.083)  ← 마찰 + 학습 편향 → 더 큰 마진
  → 100% 안전 + CBF-large에 근접한 안전 거리

dynamic 시나리오 (2-Phase):
  Phase 1 (t<5s): 마진 ~0.020m (정확한 환경)
  Phase 2 (t>5s): 마진 최대 0.105m (환경 악화)
  → ACP가 표준 CP보다 빠르게 적응 (max 0.122 vs 0.105)
```

### 8.4 벤치마크 실행

```bash
# 전체 5 시나리오 배치 실행
PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --duration 12

# 라이브 애니메이션
PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --live --scenario dynamic --duration 12
PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --live --scenario corridor --duration 12

# 특정 시나리오만
PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --scenario mismatch --duration 10
```

---

## 9. API Reference

### 9.1 ConformalPredictor

```python
from mppi_controller.learning.conformal_predictor import (
    ConformalPredictor, ConformalPredictorConfig
)

# 생성
config = ConformalPredictorConfig(
    alpha=0.1,
    window_size=200,
    gamma=0.95,
    margin_min=0.02,
    margin_max=0.5,
)
cp = ConformalPredictor(config)

# 업데이트 (매 스텝)
cp.update(predicted_state, actual_state)

# 마진 조회
margin = cp.get_margin()  # float (m)

# 통계 조회
stats = cp.get_statistics()
# {cp_margin, cp_num_scores, cp_mean_score, cp_std_score,
#  cp_min_score, cp_max_score, cp_empirical_coverage, cp_step_count}

# 초기화
cp.reset()
```

### 9.2 ConformalCBFMPPIController

```python
from mppi_controller.controllers.mppi.conformal_cbf_mppi import (
    ConformalCBFMPPIController
)
from mppi_controller.controllers.mppi.mppi_params import ConformalCBFMPPIParams

# 기본 사용 (명목 모델 예측)
params = ConformalCBFMPPIParams(
    N=20, dt=0.05, K=512, lambda_=1.0, sigma=np.array([0.5, 0.5]),
    cbf_obstacles=[(1.0, 1.0, 0.3)],
    cbf_safety_margin=0.02,  # cold start margin
    cp_alpha=0.1,
    cp_gamma=0.95,
)
ctrl = ConformalCBFMPPIController(model, params)

# 학습 모델과 함께 사용
ctrl = ConformalCBFMPPIController(
    model, params,
    prediction_fn=lambda s, u: learned_model.predict(s, u)
)

# 제어 계산
control, info = ctrl.compute_control(state, reference_trajectory)
# info["cp_margin"]: 현재 CP 마진 (m)
# info["cp_empirical_coverage"]: 경험적 커버리지 (0-1)
# info["cp_num_scores"]: 누적 점수 수

# 장애물 업데이트 (동적 장애물)
ctrl.update_obstacles([(x, y, r), ...])

# CP 통계
stats = ctrl.get_cp_statistics()

# 초기화
ctrl.reset()
```

---

## 10. References

1. **Adaptive Conformal Prediction + Probabilistic CBF** — Lindemann et al. (2024). "Safe Planning through Incremental Decomposition of Signal Temporal Logic Specifications." arXiv:2407.03569
2. **Conformal Prediction** — Vovk et al. (2005). "Algorithmic Learning in a Random World." Springer.
3. **Adaptive Conformal Inference** — Gibbs & Candes (2021). "Adaptive Conformal Inference Under Distribution Shift." NeurIPS.
4. **CBF-MPPI** — 본 프로젝트의 `docs/safety/SAFETY_CRITICAL_CONTROL.md` 참조.
