# MPPI (Model Predictive Path Integral) Control - PRD

## 1. 프로젝트 배경

### 1.1 MPPI란?

MPPI (Model Predictive Path Integral)는 **샘플링 기반 최적 제어 알고리즘**으로, 기존 MPC (Model Predictive Control)와 달리 gradient 계산 없이 확률적 샘플링으로 최적 제어를 찾습니다.

### 1.2 기존 MPC 대비 MPPI 장점

| 항목 | MPC (CasADi/IPOPT) | MPPI |
|------|---------------------|------|
| **최적화 방식** | Gradient-based NLP | Sampling-based (derivative-free) |
| **비용 함수** | 미분 가능해야 함 | 임의 비용 함수 가능 |
| **제약 조건** | Hard/Soft constraint | 비용 함수에 포함 |
| **병렬성** | 직렬 (IPOPT) | K개 샘플 병렬 처리 |
| **실시간성** | 수렴까지 가변 시간 | 고정 계산 시간 (샘플 수 의존) |
| **비선형 동역학** | CasADi symbolic 필요 | NumPy forward simulation |
| **로컬 최적** | 빠질 가능성 높음 | 샘플링으로 탐색 범위 넓음 |
| **GPU 가속** | 어려움 | 매우 용이함 |

### 1.3 MPPI 기본 원리 (Williams et al., 2016)

```
MPPI 알고리즘 흐름:

                            ┌─────────────────────┐
                            │   초기 제어열 U      │
                            │   (N, nu) 크기      │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
            ┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼───────┐
            │ ε₁ ~ N(0, Σ)   │ │ ε₂ ~ N(0, Σ) │ │ εₖ ~ N(0, Σ)   │
            │ (N, nu) 노이즈 │ │ (N, nu) 노이즈│ │ (N, nu) 노이즈 │
            └───────┬────────┘ └──────┬──────┘ └────────┬───────┘
                    │                  │                  │
            ┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼───────┐
            │  U + ε₁        │ │  U + ε₂     │ │  U + εₖ        │
            │  샘플 제어열 1  │ │  샘플 제어열 2│ │  샘플 제어열 K │
            └───────┬────────┘ └──────┬──────┘ └────────┬───────┘
                    │                  │                  │
                    └──── Rollout (RK4 적분) ─────────────┘
                                       │
            ┌───────┬──────────────────┼──────────────────┬───────┐
            │       │                  │                  │       │
    ┌───────▼──┐ ┌─▼────────┐ ┌──────▼──────┐ ┌────────▼─┐ ┌───▼──────┐
    │ 궤적 τ₁  │ │ 궤적 τ₂   │ │ 궤적 τ₃     │ │  ...     │ │ 궤적 τₖ  │
    │ (N+1,nx) │ │ (N+1,nx)  │ │ (N+1,nx)    │ │          │ │ (N+1,nx) │
    └───────┬──┘ └─┬────────┘ └──────┬──────┘ └────────┬─┘ └───┬──────┘
            │      │                  │                  │       │
            └────────── 비용 함수 S(τ) 계산 ───────────────────────┘
                                       │
            ┌───────┬──────────────────┼──────────────────┬───────┐
            │       │                  │                  │       │
    ┌───────▼──┐ ┌─▼────────┐ ┌──────▼──────┐ ┌────────▼─┐ ┌───▼──────┐
    │ S₁ = 10.2│ │ S₂ = 8.5  │ │ S₃ = 15.7   │ │  ...     │ │ Sₖ = 9.1 │
    └───────┬──┘ └─┬────────┘ └──────┬──────┘ └────────┬─┘ └───┬──────┘
            │      │                  │                  │       │
            └────────── Softmax 가중치 계산 ─────────────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  wₖ = exp(-Sₖ / λ)  │
                            │  wₖ = wₖ / Σⱼ wⱼ    │
                            │  (정규화된 가중치)   │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  U* = U + Σₖ wₖ·εₖ  │
                            │  (가중 평균 제어)    │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  u* = U*[0]         │
                            │  (첫 번째 제어 적용) │
                            └─────────────────────┘
```

**핵심 개념:**
1. **샘플링**: 현재 제어열 U에 Gaussian 노이즈 ε를 더해 K개 샘플 생성
2. **Rollout**: 각 샘플 제어로 로봇 동역학을 적분하여 궤적 τ 생성
3. **비용 계산**: 각 궤적의 총 비용 S(τ) 계산
4. **Softmax 가중**: exp(-S/λ)로 비용이 낮은 샘플에 높은 가중치 부여
5. **가중 평균**: 노이즈들의 가중 평균으로 최적 제어 방향 결정
6. **Receding Horizon**: 첫 번째 제어만 적용하고 다음 스텝에서 반복

## 2. 기능 요구사항

### 2.1 Milestone 1: Vanilla MPPI (기본 구현) ✅ 목표

#### 핵심 컴포넌트

**FR-1: MPPIParams 데이터클래스**
- 모든 MPPI 파라미터를 관리하는 데이터클래스
- 필수 필드:
  - `N`: 예측 호라이즌 (기본 30)
  - `dt`: 시간 스텝 (기본 0.05초)
  - `K`: 샘플 수 (기본 1024)
  - `lambda_`: 온도 파라미터 (기본 1.0)
  - `sigma`: 노이즈 표준편차 (기본 [0.5, 0.5])
  - `Q`: 상태 추적 가중치 (3x3 행렬)
  - `R`: 제어 노력 가중치 (2x2 행렬)
  - `Qf`: 최종 상태 가중치 (3x3 행렬)

**FR-2: BatchDynamicsWrapper - 벡터화 동역학**
- DifferentialDriveModel을 K개 샘플로 확장
- RK4 적분 벡터화 (for-loop 금지)
- 입력: `(K, nx)` 상태, `(K, N, nu)` 제어
- 출력: `(K, N+1, nx)` 궤적
- NumPy broadcasting 활용

**FR-3: 비용 함수 모듈**
- `StateTrackingCost`: 경로 추적 오차 비용
  - `cost = (x - x_ref)ᵀ Q (x - x_ref)`
- `TerminalCost`: 최종 상태 오차 비용
  - `cost = (x_N - x_ref_N)ᵀ Qf (x_N - x_ref_N)`
- `ControlEffortCost`: 제어 입력 크기 비용
  - `cost = uᵀ R u`
- `ObstacleCost`: 장애물 회피 비용
  - `cost = exp(-d²/2σ²)` (거리 d 기반)
- `CompositeMPPICost`: 여러 비용 함수 합성

**FR-4: Gaussian 노이즈 샘플러**
- `GaussianSampler` 클래스
- `sample(U, K)` → `(K, N, nu)` 노이즈
- Σ = σ² I 대각 공분산
- 제어 한계 클리핑 지원

**FR-5: Vanilla MPPI 컨트롤러**
- `MPPIController` 클래스
- 인터페이스: `compute_control(state, ref) → (control, info)`
- `info` dict 반환:
  - `sample_trajectories`: `(K, N+1, nx)` 샘플 궤적
  - `sample_weights`: `(K,)` 정규화 가중치
  - `best_trajectory`: `(N+1, nx)` 가중 평균 궤적
  - `temperature`: 현재 λ 값
  - `ess`: Effective Sample Size
- Receding horizon 구현

**FR-6: RVIZ 시각화**
- `MPPIRVizVisualizer` 클래스
- 샘플 궤적 (MarkerArray, 투명도 = 가중치)
- 가중 평균 궤적 (Path, 시안색)
- 비용 히트맵 (Marker)
- 온도/ESS 텍스트 (Marker)

**FR-7: 원형 궤적 추적 데모**
- `examples/mppi_basic_demo.py`
- 반지름 5m 원형 경로
- `--live` 실시간 시뮬레이션 모드
- RMSE < 0.2m 검증

### 2.2 Milestone 2: 고도화 (M2 MPPI)

**FR-8: ControlRateCost - 제어 변화율 비용**
- Δu = u_t - u_{t-1} 제어 변화율
- `cost = Δuᵀ R_rate Δu`
- 부드러운 제어 신호 유도

**FR-9: AdaptiveTemperature - 자동 λ 튜닝**
- ESS (Effective Sample Size) 기반
- `ESS = (Σ w_k)² / Σ w_k²`
- 목표 ESS 비율 (예: 0.5) 유지
- λ ← λ · (ESS_target / ESS_current)

**FR-10: ColoredNoiseSampler - 시간 상관 노이즈**
- OU (Ornstein-Uhlenbeck) 프로세스
- `dε = -θ·ε·dt + σ·dW`
- theta (reversion rate) 파라미터
- 시간적으로 부드러운 제어 탐색

**FR-11: Tube-MPPI - 외란 강건성**
- `AncillaryController`: body frame 피드백
  - world → body 오차 변환
  - `u_fb = K_fb · e_body`
- `TubeMPPIController`: 명목 상태 전파
  - `x_nominal[t+1] = f(x_nominal[t], u_nominal[t])`
  - `u_actual = u_nominal + u_feedback`
- `tube_enabled=False` → Vanilla 동작 (하위 호환)

**FR-12: TubeAwareCost - Tube 고려 장애물 비용**
- 장애물 안전 거리 확장
- `d_safe = safety_margin + tube_margin`
- tube_margin은 불확실성 크기 기반

**FR-13: 비교 데모**
- `examples/mppi_vanilla_vs_m2_demo.py`: Vanilla vs M2 전체 비교
- `examples/mppi_vanilla_vs_tube_demo.py`: Tube-MPPI 효과 검증
- `--live` 실시간 시뮬레이션
- `--noise` 외란 강도 조절

### 2.3 Milestone 3: SOTA 변형 (M3 MPPI)

#### M3a: Log-MPPI (참조 구현) ✅

**FR-14: LogMPPIController**
- log-space softmax 가중치 계산
- 수치 안정성 향상 (NaN/Inf 방지)
- Vanilla와 수학적 동등성 확인

#### M3b: Tsallis-MPPI (일반화 엔트로피) ✅

**FR-15: TsallisMPPIController**
- q-exponential 가중치
- `exp_q(x) = [1 + (1-q)x]^(1/(1-q))`
- q=1.0 → Vanilla (하위 호환)
- q>1 → heavy-tail (탐색 증가)
- q<1 → light-tail (집중 증가)

**FR-16: q-exponential 유틸리티**
- `utils.py`에 `q_exponential(x, q)` 구현
- `q_logarithm(x, q)` 구현
- min-centering 적용 (translation-invariance)

**FR-17: q 파라미터 비교 데모**
- `examples/tsallis_mppi_demo.py`
- q=0.5, 1.0, 1.2, 1.5 비교
- 탐색 vs 집중 trade-off 시각화

#### M3c: Risk-Aware MPPI (CVaR) ✅

**FR-18: RiskAwareMPPIController**
- CVaR (Conditional Value at Risk) 가중치 절단
- 최저 비용 `ceil(alpha*K)` 개만 softmax
- `cvar_alpha` 파라미터 (기본 1.0)
  - α=1.0: risk-neutral (Vanilla)
  - α<1.0: risk-averse (보수적)
  - 실용 범위: [0.1, 1.0]

**FR-19: Risk-Aware 데모**
- `examples/risk_aware_mppi_demo.py`
- 장애물 환경에서 α별 회피 전략 비교
- risk-averse가 더 큰 여유 거리 유지

#### M3d: Stein Variational MPPI (SVMPC) ✅

**FR-20: SteinVariationalMPPIController**
- SVGD (Stein Variational Gradient Descent) 기반
- 샘플 다양성 유도 (커널 반발력)
- rbf_kernel, median_bandwidth 활용
- `svgd_num_iterations` 파라미터
  - 0 → Vanilla (하위 호환)
  - 3-5 → 일반적 설정

**FR-21: SVGD 유틸리티**
- `utils.py`에 `rbf_kernel(X, Y, bandwidth)` 구현
- `rbf_kernel_grad(X, Y, bandwidth)` 구현
- `median_bandwidth(X)` 구현

**FR-22: SVMPC 비교 데모**
- `examples/stein_variational_mppi_demo.py`
- SVGD iteration 수별 성능 비교
- 샘플 분포 시각화

### 2.4 Milestone 3.5: 확장 변형 (M3.5 MPPI)

#### M3.5a: Smooth MPPI (SMPPI) ✅

**FR-23: SmoothMPPIController**
- Δu input-lifting 구조
- 제어 변화량 공간에서 최적화
- cumsum으로 원래 제어 복원
- `u[t] = u[0] + Σ(Δu[0:t])`

**FR-24: Jerk Cost**
- ΔΔu (제어 변화량의 변화량) 페널티
- `cost = ΔΔuᵀ R_jerk ΔΔu`
- 액추에이터 보호

**FR-25: Smooth MPPI 데모**
- `examples/smooth_mppi_demo.py`
- Vanilla vs SMPPI 제어 변화율 비교
- jerk weight별 부드러움 정도 조절

#### M3.5b: Spline-MPPI ✅

**FR-26: SplineMPPIController**
- P개 knot에 노이즈 적용
- B-spline basis로 N개 제어점 보간
- P << N으로 노이즈 차원 축소
- 구조적 smooth 제어

**FR-27: B-spline 유틸리티**
- `utils.py`에 `_bspline_basis(N, P, degree)` 구현
- de Boor 재귀 알고리즘 (순수 NumPy)
- scipy 의존성 없음

**FR-28: Spline-MPPI 파라미터**
- `spline_num_knots`: knot 개수 P (기본 8)
- `spline_degree`: B-spline 차수 (기본 3)
- `spline_knot_sigma`: knot 노이즈 표준편차

**FR-29: Spline-MPPI 데모**
- `examples/spline_mppi_demo.py`
- P=4 vs P=8 비교
- 노이즈 차원 축소 효과 검증

#### M3.5c: SVG-MPPI (Guide Particle) ✅

**FR-30: SVGMPPIController**
- G개 guide particle SVGD 적용
- (K-G)개 follower는 guide 주변 재샘플링
- G << K로 SVGD 계산량 O(G²D) << O(K²D)
- SVMPC 대비 고속화

**FR-31: SVG-MPPI 파라미터**
- `svg_num_guide_particles`: guide 수 G (기본 64)
- `svg_guide_step_size`: SVGD 스텝 크기
- `svg_guide_iterations`: SVGD iteration 수

**FR-32: SVG-MPPI 데모**
- `examples/svg_mppi_demo.py`
- SVG vs SVMPC 계산 속도 비교
- 장애물 환경에서 다중 모드 유지 검증

### 2.5 벤치마크 도구

**FR-33: 전체 변형 벤치마크**
- `examples/mppi_all_variants_benchmark.py`
- 9종 변형 동시 비교:
  1. Vanilla MPPI
  2. Tube-MPPI
  3. Log-MPPI
  4. Tsallis-MPPI
  5. Risk-Aware MPPI
  6. SVMPC
  7. Smooth MPPI
  8. Spline-MPPI
  9. SVG-MPPI

**FR-34: 메트릭 수집**
- Position RMSE (m)
- Max Position Error (m)
- Control Rate (Δu 평균)
- Solve Time (ms)
- ESS (Effective Sample Size)

**FR-35: 벤치마크 옵션**
- `--live`: 실시간 시뮬레이션 모드
- `--trajectory {circle,figure8,sine}`: 궤적 선택
- ASCII 요약 테이블
- 6패널 정적 비교 차트

### 2.6 비기능 요구사항 (NFR)

**NFR-1: 순수 NumPy 구현**
- CasADi 의존성 없음
- ROS2 환경에서 경량 실행

**NFR-2: 실시간 성능**
- K=1024, N=30에서 < 100ms
- 10Hz 제어 주기 유지

**NFR-3: 인터페이스 일관성**
- `compute_control(state, ref) → (control, info)` 시그니처 준수
- 기존 MPC와 호환 가능

**NFR-4: 벡터화 강제**
- for-loop 금지
- NumPy broadcasting 활용
- GPU 가속 준비

**NFR-5: 성능 검증**
- 원형 궤적 Position RMSE < 0.2m
- figure8 궤적 RMSE < 0.5m

**NFR-6: 하위 호환성**
- Tube-MPPI `tube_enabled=False` → Vanilla
- SVMPC `svgd_num_iterations=0` → Vanilla
- Tsallis-MPPI `tsallis_q=1.0` → Vanilla

## 3. 아키텍처

### 3.1 시스템 아키텍처

```
전체 시스템 구조:

┌─────────────────────────────────────────────────────────────────────┐
│                       MPPIController (base_mppi.py)                 │
│                          Vanilla MPPI 핵심 알고리즘                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────┐  │
│  │ MPPIParams   │   │ Sampling     │   │ CompositeMPPICost     │  │
│  │ (파라미터)    │   │ (노이즈 생성) │   │ (비용 함수 합성)       │  │
│  └──────┬───────┘   └──────┬───────┘   └───────────┬───────────┘  │
│         │                  │                        │              │
│         │      ┌───────────▼────────────┐           │              │
│         └─────►│ BatchDynamicsWrapper   │◄──────────┘              │
│                │ (K개 샘플 병렬 rollout) │                          │
│                └───────────┬────────────┘                          │
│                            │                                       │
│                ┌───────────▼────────────┐                          │
│                │ DifferentialDriveModel │ (기존 모델 재사용)        │
│                │ dx/dt = f(x, u)        │                          │
│                └────────────────────────┘                          │
│                                                                     │
│  compute_control(state, ref) → (control, info)                     │
│    info: {sample_trajectories, sample_weights, best_trajectory,    │
│           temperature, ess, ...}                                   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              │ 상속/확장
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
┌──────▼────────┐  ┌──────────▼──────────┐  ┌───────▼────────────┐
│ TubeMPPI      │  │ LogMPPI             │  │ TsallisMPPI        │
│ (M2)          │  │ (M3a)               │  │ (M3b)              │
│ - 명목 상태    │  │ - log-space softmax │  │ - q-exponential    │
│ - 피드백 보정  │  │ - 수치 안정성       │  │ - 탐색/집중 조절   │
└───────────────┘  └─────────────────────┘  └────────────────────┘

┌───────────────┐  ┌─────────────────────┐  ┌────────────────────┐
│ RiskAwareMPPI │  │ SteinVariationalMPPI│  │ SmoothMPPI         │
│ (M3c)         │  │ (M3d / SVMPC)       │  │ (M3.5a)            │
│ - CVaR 절단   │  │ - SVGD 샘플 다양성  │  │ - Δu input-lifting │
│ - α<1 보수적  │  │ - 커널 반발력       │  │ - jerk cost        │
└───────────────┘  └─────────────────────┘  └────────────────────┘

┌───────────────┐  ┌─────────────────────┐
│ SplineMPPI    │  │ SVGMPPIController   │
│ (M3.5b)       │  │ (M3.5c)             │
│ - B-spline    │  │ - Guide SVGD        │
│ - knot 보간   │  │ - Follower resample │
└───────────────┘  └─────────────────────┘

┌───────────────────────────────────────────┐
│         MPPIRVizVisualizer                │
│         (ros2/mppi_rviz_visualizer.py)    │
│  - 샘플 궤적 (MarkerArray, 투명도 가중치) │
│  - 가중 평균 궤적 (Path, 시안)            │
│  - 비용 히트맵 (Marker)                   │
│  - Tube 경계 (Marker, Tube-MPPI)          │
│  - 온도/ESS 텍스트 (Marker)               │
└───────────────────────────────────────────┘
```

### 3.2 파일 구조

```
learning_mppi/
├── mppi_controller/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── differential_drive/
│   │       ├── __init__.py
│   │       └── differential_drive_model.py      # dx/dt = f(x, u)
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── mppi/
│   │       ├── __init__.py                      # MPPIController 등 export
│   │       ├── mppi_params.py                   # MPPIParams 데이터클래스
│   │       ├── dynamics_wrapper.py              # BatchDynamicsWrapper (RK4)
│   │       ├── cost_functions.py                # 비용 함수 모듈
│   │       │   ├── StateTrackingCost
│   │       │   ├── TerminalCost
│   │       │   ├── ControlEffortCost
│   │       │   ├── ControlRateCost              [M2]
│   │       │   ├── ObstacleCost
│   │       │   ├── TubeAwareCost                [M2]
│   │       │   └── CompositeMPPICost
│   │       ├── sampling.py                      # 노이즈 샘플러
│   │       │   ├── GaussianSampler
│   │       │   └── ColoredNoiseSampler          [M2]
│   │       ├── base_mppi.py                     # Vanilla MPPI
│   │       │   └── _compute_weights()           ← 오버라이드 포인트
│   │       ├── adaptive_temperature.py          # AdaptiveTemperature [M2]
│   │       ├── ancillary_controller.py          # AncillaryController [M2]
│   │       ├── tube_mppi.py                     # TubeMPPIController [M2]
│   │       ├── log_mppi.py                      # LogMPPIController [M3a]
│   │       ├── tsallis_mppi.py                  # TsallisMPPIController [M3b]
│   │       ├── risk_aware_mppi.py               # RiskAwareMPPIController [M3c]
│   │       ├── stein_variational_mppi.py        # SteinVariationalMPPIController [M3d]
│   │       ├── smooth_mppi.py                   # SmoothMPPIController [M3.5a]
│   │       ├── spline_mppi.py                   # SplineMPPIController [M3.5b]
│   │       ├── svg_mppi.py                      # SVGMPPIController [M3.5c]
│   │       └── utils.py                         # 유틸리티
│   │           ├── normalize_angle()
│   │           ├── log_sum_exp()
│   │           ├── q_exponential(), q_logarithm()  [M3b]
│   │           ├── rbf_kernel(), median_bandwidth() [M3d]
│   │           └── _bspline_basis()                [M3.5b]
│   ├── ros2/
│   │   ├── __init__.py
│   │   ├── mppi_node.py                         # MPPI ROS2 노드
│   │   └── mppi_rviz_visualizer.py              # RVIZ 시각화
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── simulator.py                         # 시뮬레이터
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                            # 로깅 유틸리티
│       └── trajectory.py                        # 궤적 생성
├── tests/
│   ├── __init__.py
│   ├── test_mppi.py                             # Vanilla MPPI 테스트
│   ├── test_mppi_cost_functions.py              # 비용 함수 테스트
│   ├── test_mppi_sampling.py                    # 샘플링 테스트
│   ├── test_dynamics_wrapper.py                 # 동역학 래퍼 테스트
│   ├── test_ancillary_controller.py             # AncillaryController [M2]
│   ├── test_tube_mppi.py                        # TubeMPPIController [M2]
│   ├── test_log_mppi.py                         # LogMPPIController [M3a]
│   ├── test_tsallis_mppi.py                     # TsallisMPPIController [M3b]
│   ├── test_risk_aware_mppi.py                  # RiskAwareMPPIController [M3c]
│   ├── test_stein_variational_mppi.py           # SteinVariationalMPPIController [M3d]
│   ├── test_smooth_mppi.py                      # SmoothMPPIController [M3.5a]
│   ├── test_spline_mppi.py                      # SplineMPPIController [M3.5b]
│   └── test_svg_mppi.py                         # SVGMPPIController [M3.5c]
├── examples/
│   ├── mppi_basic_demo.py                       # Vanilla MPPI 데모
│   ├── mppi_vanilla_vs_m2_demo.py               # Vanilla vs M2 비교 [M2]
│   ├── mppi_vanilla_vs_tube_demo.py             # Vanilla vs Tube 비교 [M2]
│   ├── log_mppi_demo.py                         # Log-MPPI 비교 [M3a]
│   ├── tsallis_mppi_demo.py                     # Tsallis q 비교 [M3b]
│   ├── risk_aware_mppi_demo.py                  # Risk-Aware α 비교 [M3c]
│   ├── stein_variational_mppi_demo.py           # SVMPC iteration 비교 [M3d]
│   ├── smooth_mppi_demo.py                      # Smooth MPPI jerk weight [M3.5a]
│   ├── spline_mppi_demo.py                      # Spline-MPPI knot 수 [M3.5b]
│   ├── svg_mppi_demo.py                         # SVG-MPPI vs SVMPC [M3.5c]
│   └── mppi_all_variants_benchmark.py           # 9종 변형 벤치마크
├── configs/
│   ├── mppi_params.yaml                         # 기본 설정
│   └── robot_params.yaml                        # 로봇 파라미터
├── docs/
│   ├── mppi/
│   │   ├── PRD.md                               # 본 문서
│   │   └── MPPI_GUIDE.md                        # 기술 가이드
│   └── api/                                     # API 문서
├── .claude/
│   ├── scripts/
│   │   ├── issue-watcher.sh                     # GitHub Issue Watcher
│   │   └── todo-worker.sh                       # TODO Worker
│   └── memory/
│       └── MEMORY.md                            # Claude 메모리
├── pyproject.toml                               # Python 패키지 설정
├── requirements.txt                             # 의존성
├── CLAUDE.md                                    # Claude 개발 가이드
├── TODO.md                                      # 작업 목록
└── README.md                                    # 프로젝트 README
```

## 4. 마일스톤 로드맵

```
마일스톤 진행 계획:

M1: Vanilla MPPI (기본 구현) ← 현재 목표
├── ✅ MPPIParams 데이터클래스
├── ✅ BatchDynamicsWrapper (RK4 벡터화)
├── ✅ 비용 함수 (StateTracking, Terminal, ControlEffort, Obstacle)
├── ✅ GaussianSampler
├── ✅ Vanilla MPPI 컨트롤러
├── ✅ 원형 궤적 추적 테스트 (RMSE < 0.2m)
└── ⬜ RVIZ 시각화 (샘플 궤적, 가중 궤적, 히트맵)

M2: 고도화
├── ⬜ ControlRateCost (제어 변화율 비용)
├── ⬜ AdaptiveTemperature (ESS 기반 λ auto-tuning)
├── ⬜ ColoredNoiseSampler (OU 프로세스)
├── ⬜ AncillaryController (body frame 피드백) [Tube]
├── ⬜ TubeMPPIController (명목 상태 + 피드백) [Tube]
├── ⬜ TubeAwareCost (tube margin 확장) [Tube]
├── ⬜ Vanilla vs M2 / Vanilla vs Tube 비교 데모
└── ⬜ GPU 가속 (CuPy/JAX) — 선택적

M3: SOTA 변형
├── ⬜ M3a: Log-MPPI (log-space softmax 참조 구현)
├── ⬜ M3b: Tsallis-MPPI (q-exponential, min-centering)
├── ⬜ M3c: Risk-Aware MPPI (CVaR 가중치 절단)
└── ⬜ M3d: Stein Variational MPPI (SVGD 커널 다양성)

M3.5: 확장 변형
├── ⬜ M3.5a: Smooth MPPI (Δu input-lifting, jerk cost)
├── ⬜ M3.5b: Spline-MPPI (B-spline basis 보간, P << N)
└── ⬜ M3.5c: SVG-MPPI (Guide particle SVGD + follower)

M4: ROS2 통합
├── ⬜ ROS2 기본 노드 구현
├── ⬜ nav2 Controller 플러그인 (Python prototype)
├── ⬜ 실제 로봇 인터페이스
├── ⬜ 파라미터 서버 통합
├── ⬜ 동적 장애물 회피
├── ⬜ 실시간 경로 재계획
└── ⬜ ROS2 통합 테스트

M5a: C++ MPPI 코어 변환
├── ⬜ Python → C++ 포팅 (실시간 성능)
├── ⬜ Eigen 기반 배치 rollout
├── ⬜ pybind11 Python 바인딩
└── ⬜ 성능 벤치마크 (< 100ms 검증)

M5b: ROS2 nav2 Controller 플러그인
├── ⬜ C++ MPPI nav2 Server 플러그인
├── ⬜ nav2 ComputePathToPose 호환
├── ⬜ 파라미터 YAML 설정
└── ⬜ 실제 로봇 테스트

GPU 가속 (선택적, 고성능 시나리오)
├── ⬜ CuPy/JAX 기반 rollout + cost 병렬화
├── ⬜ SVMPC pairwise kernel CUDA 가속
└── ⬜ K=4096+ 대규모 샘플 벤치마크
```

## 5. 성능 검증 기준

### 5.1 추적 성능

| 궤적 유형 | RMSE 목표 | Max Error 목표 |
|----------|----------|---------------|
| Circle (r=5m) | < 0.2m | < 0.5m |
| Figure-8 | < 0.5m | < 1.0m |
| Sine wave | < 0.3m | < 0.7m |

### 5.2 계산 성능

| 설정 | 시간 목표 | 주파수 |
|-----|----------|-------|
| K=1024, N=30 (Python) | < 100ms | 10Hz |
| K=1024, N=30 (C++) | < 50ms | 20Hz |
| K=4096, N=50 (GPU) | < 100ms | 10Hz |

### 5.3 제어 품질

- **Control Rate**: Vanilla 대비 M2에서 30% 감소
- **ESS**: 0.3 ~ 0.7 범위 유지 (AdaptiveTemperature)
- **Tube Width**: 외란 환경에서 < 0.5m 유지

## 6. 학습 기반 모델 적응 (Milestone 3.6+)

### 6.1 현황 및 문제점

Model Mismatch 시나리오(컨트롤러 모델 ≠ 실제 세계)에서 학습 모델의 성능이 기대에 미치지 못하는 문제가 확인됨.

**Dynamic World 설정**: 실제 세계는 5D DifferentialDriveDynamic (마찰 c_v=0.5, c_omega=0.3, PD 제어, 프로세스 노이즈)이나, 컨트롤러는 3D 기구학 모델만 보유.

```
현재 성능 (circle 20s, --world dynamic):

  Oracle (5D 정확)    ████░░░░░░░░░░░░░░░░  0.022m  ← 이론 상한
  Dynamic (5D 추정)   █████░░░░░░░░░░░░░░░  0.026m
  Kinematic (3D)      █████░░░░░░░░░░░░░░░  0.029m  ← 기준선
  ─────────────────── 목표 라인 ──────────────────────
  MAML (3D Residual)  ██████████░░░░░░░░░░  0.086m  ✗ 기구학보다 열위
  Neural (3D E2E)     ███████████░░░░░░░░░  0.096m  ✗
  Residual (3D)       ████████████████████  0.198m  ✗ 매우 저조
```

### 6.2 근본 원인 분석

**FR-40: 차원 불일치 (Dimension Mismatch)**
- 실제 세계는 5D 상태 (x, y, θ, v, ω)를 가지나, 학습 모델은 3D 관측 (x, y, θ)만 수신
- 관성/속도 상태를 관측하지 못하므로 동역학 잔차를 정확히 근사할 수 없음
- 이는 Partially Observable 문제이며, 3D 잔차 모델의 이론적 한계

**FR-41: Warmup 오차 누적**
- 2-Phase 아키텍처의 Phase 1 (40 step, ~2초)은 순수 기구학 → 초기 오차 큼
- 전체 RMSE에 warmup 구간 오차가 평균으로 반영되어 성능 하락
- 특히 짧은 시뮬레이션 (10-20초)에서 warmup 비중이 10-20%

**FR-42: FOMAML 근사 한계**
- 1차 근사 (create_graph=False)로 Hessian 정보 손실
- inner_steps=100은 실질적으로 fine-tuning에 가까워 메타 학습 이점 약화
- 적응 후에도 잔차 fitting 정확도가 ~70-80%에 머무름

### 6.3 개선 요구사항

**FR-43: 5D Residual MAML**
- MAML 모델을 5D state (x, y, θ, v, ω) + 2D control → 5D residual로 확장
- DynamicKinematicAdapter (5D forward_dynamics)를 base model로 사용
- 기구학 base가 아닌 동역학 base에서 잔차 학습 → 더 작은 잔차, 더 정확한 보정
- 검증 기준: MAML RMSE < Kinematic RMSE (0.029m 이하)

**FR-44: 적응형 Warmup**
- Phase 1 최소화: warmup_steps를 동적으로 조절 (데이터 품질 기반)
- 이전 에피소드의 적응 결과 재활용 (cold start 방지)
- 적응 트리거: 고정 주기 대신 오차 급증 시 즉시 재적응

**FR-45: 대안 메타 학습 알고리즘 탐색**
- Reptile (Nichol et al. 2018): FOMAML보다 안정적, 구현 단순
- CAVIA (Zintgraf et al. 2019): context vector 기반, 파라미터 전체가 아닌 context만 적응
- ALPaCA (Harrison et al. 2018): Bayesian linear regression + neural features → 불확실성 포함
- 검증 기준: 동일 데이터/조건에서 FOMAML 대비 RMSE 30%+ 개선

**FR-46: 오프라인 학습 모델 개선**
- Neural/Residual 오프라인 모델의 학습 데이터 다양성 부족 해소
- Domain randomization: c_v ∈ [0.1, 0.8], c_omega ∈ [0.1, 0.5] 범위에서 랜덤 샘플링
- 다양한 궤적 (circle, figure8, sine, random) + 속도 프로필 조합
- 검증 기준: Residual RMSE < 0.1m (현재 0.198m → 50%+ 개선)

### 6.4 성능 목표

| 모델 | 현재 RMSE | 목표 RMSE | 개선율 |
|------|-----------|-----------|--------|
| MAML (Residual) | 0.086m | < 0.030m | 65%+ |
| Neural (E2E) | 0.096m | < 0.050m | 48%+ |
| Residual (Hybrid) | 0.198m | < 0.080m | 60%+ |

### 6.5 리스크

| 리스크 | 영향 | 완화 전략 |
|--------|------|-----------|
| 5D MAML 학습 불안정 | 높음 | 학습률 스케줄링, gradient clipping, 단계적 확장 |
| Full MAML 메모리 폭발 | 중간 | FOMAML 유지하되 inner_steps 최적화, 또는 Reptile |
| 오프라인 데이터 과적합 | 중간 | Domain randomization, validation split 확대 |
| Warmup 제거 시 발산 | 높음 | 최소 warmup 보장 (10 step), 발산 감지 fallback |

---

## 7. 참고 자료

### 7.1 논문

**Vanilla MPPI:**
- Williams et al. (2016) - "Aggressive Driving with Model Predictive Path Integral Control"
- Williams et al. (2017) - "Information Theoretic MPC for Model-Based Reinforcement Learning"

**M2 고도화:**
- Williams et al. (2018) - "Robust Sampling Based Model Predictive Control with Sparse Objective Information" (Tube-MPPI)
- Bhardwaj et al. (2020) - "Blending MPC & Value Function Approximation for Efficient RL" (Colored Noise)

**M3 SOTA 변형:**
- Yin et al. (2021) - "Trajectory Distribution Control for Model Predictive Path Integral Control Using Covariance Steering" (Tsallis MPPI)
- Yin et al. (2023) - "Risk-Aware Model Predictive Path Integral Control" (CVaR MPPI)
- Lambert et al. (2020) - "Stein Variational Model Predictive Control" (SVMPC)

**M3.5 확장 변형:**
- Kim et al. (2021) - "Smooth Model Predictive Path Integral Control" (SMPPI, input-lifting)
- Bhardwaj et al. (2024, ICRA) - "Spline-MPPI: Continuous-Time Trajectory Optimization via B-Spline Interpolation"
- Kondo et al. (2024, ICRA) - "SVG-MPPI: Efficient Stein Variational Guidance for MPPI"

**메타 학습 / 적응 제어:**
- Finn et al. (2017) - "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (MAML)
- Nichol et al. (2018) - "On First-Order Meta-Learning Algorithms" (Reptile, FOMAML)
- Zintgraf et al. (2019) - "Fast Context Adaptation via Meta-Learning" (CAVIA)
- Harrison et al. (2018) - "Meta-Learning Priors for Efficient Online Bayesian Regression" (ALPaCA)
- Nagabandi et al. (2019) - "Learning to Adapt in Dynamic, Real-World Environments through Meta-Reinforcement Learning"

### 7.2 오픈소스 참조 구현

- **pytorch_mppi** - PyTorch GPU 가속 MPPI ([GitHub](https://github.com/UM-ARM-Lab/pytorch_mppi))
- **mppic** - ROS2 nav2 MPPI Controller C++ 플러그인 ([GitHub](https://github.com/ros-planning/navigation2/tree/main/nav2_mppi_controller))
- **PythonLinearNonlinearControl** - Python 제어 알고리즘 모음 (MPPI 포함)

### 7.3 ROS2 관련

- **nav2_core::Controller** - nav2 Controller 플러그인 인터페이스
- **nav2_costmap_2d** - 동적 장애물 맵
- **visualization_msgs** - RVIZ 마커 메시지

## 8. 개발 우선순위 요약

```
우선순위 매트릭스:

┌─────────────────────┬──────────┬──────────┬─────────┐
│ 항목                │ 중요도   │ 난이도   │ 우선순위 │
├─────────────────────┼──────────┼──────────┼─────────┤
│ M1: Vanilla MPPI    │ ★★★★★   │ ★★★☆☆   │ P0      │
│ M2: Tube-MPPI       │ ★★★★☆   │ ★★★★☆   │ P1      │
│ M2: AdaptiveTemp    │ ★★★☆☆   │ ★★☆☆☆   │ P1      │
│ M3: Tsallis-MPPI    │ ★★★☆☆   │ ★★★☆☆   │ P1      │
│ M3: SVMPC           │ ★★★☆☆   │ ★★★★★   │ P2      │
│ M3.5: Smooth MPPI   │ ★★★☆☆   │ ★★★☆☆   │ P1      │
│ M3.5: Spline-MPPI   │ ★★☆☆☆   │ ★★★★☆   │ P2      │
│ M3.5: SVG-MPPI      │ ★★☆☆☆   │ ★★★★☆   │ P2      │
│ M4: ROS2 통합       │ ★★★★★   │ ★★★☆☆   │ P0      │
│ M5: C++ 포팅        │ ★★★★☆   │ ★★★★★   │ P1      │
│ GPU 가속            │ ★★☆☆☆   │ ★★★★☆   │ P2      │
└─────────────────────┴──────────┴──────────┴─────────┘

추천 순서:
1. M1 Vanilla MPPI (P0) ← 시작점
2. M4 ROS2 기본 통합 (P0) ← 실용성
3. M2 Tube-MPPI (P1) ← 강건성
4. M2 AdaptiveTemp (P1) ← 튜닝 자동화
5. M3 Tsallis-MPPI (P1) ← 탐색/집중 조절
6. M3.5 Smooth MPPI (P1) ← 제어 부드러움
7. M5 C++ 포팅 (P1) ← 실시간 성능
8. M3 SVMPC (P2) ← 고급 기능
9. M3.5 Spline/SVG (P2) ← 연구 목적
10. GPU 가속 (P2) ← 고성능 시나리오
```

## 9. 리스크 및 완화 전략

| 리스크 | 영향 | 확률 | 완화 전략 |
|-------|-----|-----|----------|
| 수치 불안정 (NaN/Inf) | 높음 | 중간 | Log-space 연산, 클리핑, 단위 테스트 |
| 실시간 성능 미달 | 높음 | 중간 | C++ 포팅, GPU 가속, 프로파일링 |
| ROS2 통합 복잡도 | 중간 | 높음 | 단계적 통합, Python prototype 먼저 |
| SVMPC 계산량 폭발 | 중간 | 높음 | SVG-MPPI로 대체, K 제한 |
| 메모리 부족 (대규모 샘플) | 낮음 | 낮음 | 배치 처리, GPU 메모리 관리 |

---

**문서 버전:** 1.1
**최종 업데이트:** 2026-02-18
**작성자:** Claude Sonnet 4.5 / Claude Opus 4.6
**상태:** Active (M1~M3.6 완료, MAML 성능 개선 진행 중)
