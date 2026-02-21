# MPPI 구현 현황

**날짜**: 2026-02-21 (Updated)
**상태**: Phase 4 + Safety + GPU + MAML + Post-MAML + 최적화 완료 ✅

## 구현 완료 변형

### M1: Vanilla MPPI ✅
- **파일**: `mppi_controller/controllers/mppi/base_mppi.py`
- **특징**: 기본 MPPI 알고리즘, softmax 가중치
- **성능**: RMSE 0.012m, 41ms
- **커밋**: 초기 구현

### M2: Tube-MPPI ✅
- **파일**: `mppi_controller/controllers/mppi/tube_mppi.py`
- **특징**: 명목 상태 + 피드백 제어, 외란 강건성
- **성능**: RMSE 0.010m, 41ms
- **커밋**: f9052de
- **추가 컴포넌트**:
  - `ancillary_controller.py`: Body frame 피드백
  - `adaptive_temperature.py`: ESS 기반 λ 자동 조정

### M3a: Log-MPPI ✅
- **파일**: `mppi_controller/controllers/mppi/log_mppi.py`
- **특징**: log-space softmax, 수치 안정성
- **성능**: RMSE 0.012m, 42ms
- **커밋**: cd736f3
- **핵심 기술**: log-sum-exp trick, NaN/Inf 방지

### M3b: Tsallis-MPPI ✅
- **파일**: `mppi_controller/controllers/mppi/tsallis_mppi.py`
- **특징**: q-exponential 가중치, 탐색/집중 조절
- **성능**: RMSE 0.010m, 43ms
- **커밋**: d1790d6
- **파라미터**: `tsallis_q` (q=1.0 → Vanilla)

### M3c: Risk-Aware MPPI ✅
- **파일**: `mppi_controller/controllers/mppi/risk_aware_mppi.py`
- **특징**: CVaR 기반 샘플 선택, 안전성
- **성능**: RMSE 0.013m, 42ms
- **커밋**: 7a01534
- **파라미터**: `cvar_alpha` (α<1.0 → 보수적)

### M3d: Stein Variational MPPI (SVMPC) ✅ + SPSA 최적화
- **파일**: `mppi_controller/controllers/mppi/stein_variational_mppi.py`
- **특징**: SVGD로 샘플 다양성 유지
- **성능**: RMSE 0.009m, **113ms** (SPSA 최적화 후, 기존 1515ms에서 13x 개선)
- **커밋**: 4945838
- **유틸리티**: `utils/stein_variational.py` (RBF 커널, median bandwidth, efficient SVGD)
- **최적화 (2026-02-21)**:
  1. SPSA gradient: per-dim finite diff (N×nu=60 rollouts) → 동시 섭동 (2 rollouts)
  2. Efficient SVGD: (K,K,N,nu) 텐서 제거 → 행렬 연산 (503MB→0MB at K=1024)
  3. Merged kernel+bandwidth: `rbf_kernel_with_bandwidth()` K² 거리 1회 계산

### M3.5a: Smooth MPPI ✅
- **파일**: `mppi_controller/controllers/mppi/smooth_mppi.py`
- **특징**: Δu input-lifting, 제어 부드러움
- **성능**: RMSE 0.009m, 42ms, Control Rate 0.0000
- **커밋**: 399cff6
- **추가 비용**: Jerk Cost (ΔΔu 페널티)

### M3.5b: Spline-MPPI ✅
- **파일**: `mppi_controller/controllers/mppi/spline_mppi.py`
- **특징**: B-spline 보간, 메모리 효율
- **성능**: RMSE 0.018m, 41ms, 메모리 73.3% 감소
- **커밋**: 9c1c7ed
- **파라미터**: `spline_num_knots` (P=8), `spline_degree` (k=3)

### M3.5c: SVG-MPPI (Guide Particle) ✅
- **파일**: `mppi_controller/controllers/mppi/svg_mppi.py`
- **특징**: Guide particle SVGD, 효율성
- **성능**: RMSE 0.007m, 273ms, SVGD 99.9% 복잡도 감소
- **커밋**: bedfec0
- **파라미터**: `svg_num_guide_particles` (G=32)

## Phase 4: 학습 모델 고도화 ✅

### M3.6a: Neural Dynamics ✅
- **파일**: `mppi_controller/models/learned/neural_dynamics.py`
- **특징**: PyTorch MLP 기반 end-to-end 학습
- **성능**: RMSE 0.068m, 추론 24ms
- **커밋**: b2bc212, dcace1b
- **아키텍처**: MLP [128, 128, 64], 25,731 파라미터
- **학습 파이프라인**: `learning/neural_network_trainer.py`
  - 데이터 수집 → 학습 → 평가 전체 파이프라인
  - Normalization, Early stopping, LR scheduling
  - 학습 히스토리 plot 자동 생성

### M3.6b: Gaussian Process Dynamics ✅
- **파일**: `mppi_controller/models/learned/gaussian_process_dynamics.py`
- **특징**: GPyTorch sparse GP, 불확실성 정량화
- **성능**: RMSE ~0.04m, 추론 ~10ms (inducing points: 200)
- **커밋**: ecfe346
- **장점**: 데이터 효율성 (100 샘플), 불확실성 96%+ calibration
- **학습 파이프라인**: `learning/gaussian_process_trainer.py`
  - Sparse GP 학습 (메모리 효율적)
  - Inducing point 자동 선택
  - 불확실성 정량화 (epistemic + aleatoric)

### M3.6c: Residual Dynamics (Hybrid) ✅
- **파일**: `mppi_controller/models/learned/residual_dynamics.py`
- **특징**: Physics model + Learned correction
- **성능**: RMSE 0.092m (NN), 0.032m (GP), 추론 31ms (NN)
- **커밋**: f34753e
- **장점**: 물리 법칙 보장 + 학습 유연성
- **지원**: Neural/GP residual functions

### M3.6d: 온라인 학습 ✅
- **파일**: `mppi_controller/learning/online_dynamics_learner.py`
- **특징**: Sim-to-Real 온라인 적응
- **성능**: Real-time adaptation (10Hz 제어 유지)
- **커밋**: 84b222f
- **기능**: Incremental learning, 새 데이터 추가

### 학습 모델 문서화 ✅
- **LEARNED_MODELS_GUIDE.md** (743 lines): 학습 모델 3종 종합 가이드
- **ONLINE_LEARNING.md** (481 lines): 온라인 학습 알고리즘 상세 설명
- **총 문서**: 1,224 lines

### Plot 갤러리 ✅
- **MPPI 변형 비교** (7개): 전체 벤치마크, Vanilla vs Tube/Log, Smooth/Spline/SVG/SVMPC
- **학습 모델 비교** (2개): Neural Dynamics 9패널 비교, 학습 곡선
- **총 Plot**: 9개 PNG (plots/ 디렉토리)

## 성능 비교 요약

### MPPI 변형 성능

| 변형 | RMSE (m) | Solve Time (ms) | 특징 | 사용 시나리오 |
|------|----------|-----------------|------|---------------|
| **Vanilla** | 0.012 | 41 | 기본, 빠름 | 일반 추적 |
| **Tube** | 0.010 | 41 | 강건 | 외란 환경 |
| **Log** | 0.012 | 42 | 안정 | 수치 안정성 필수 |
| **Tsallis** | 0.010 | 43 | 탐색 조절 | 다중 모드 탐색 |
| **Risk-Aware** | 0.013 | 42 | 안전 | 장애물 회피 |
| **Smooth** | 0.009 | 42 | 부드러움 | 액추에이터 보호 |
| **SVMPC** | 0.009 | 113 | 샘플 품질 (SPSA) | 고품질 제어 |
| **Spline** | 0.018 | 41 | 메모리 효율 | 메모리 제약 |
| **SVG** | 0.007 | 273 | SVGD 고속화 | 품질+속도 균형 |

### 학습 모델 성능

| 모델 | RMSE (m) | 추론 시간 (ms) | 불확실성 | 데이터 요구량 | 사용 시나리오 |
|------|----------|----------------|----------|--------------|---------------|
| **Physics (Kinematic)** | 0.007 | 4.6 | ❌ | 0 (모델 기반) | 정확한 모델 가능 |
| **Neural (Learned)** | 0.068 | 24.0 | ❌ | 600 샘플 | 복잡한 동역학 |
| **Residual (Hybrid)** | 0.092 | 31.0 | ❌ | 600 샘플 | 모델 보정 |
| **Gaussian Process** | ~0.04 | ~10 | ✅ 96% | 100 샘플 | 데이터 효율+불확실성 |

## 복잡도 비교

### 메모리 복잡도
- **Vanilla MPPI**: O(K×N×nu) = 1024×30×2 = 61,440
- **Spline-MPPI**: O(K×P×nu) = 1024×8×2 = 16,384 (73.3% 감소)

### 계산 복잡도
- **Vanilla MPPI**: O(K×N) rollout
- **SVMPC**: O(K×N) rollout + O(K²×N×nu) SVGD
- **SVG-MPPI**: O(K×N) rollout + O(G²×N×nu) SVGD (99.9% 감소)

## 테스트 현황

### MPPI 변형 테스트

```
tests/
├── test_mppi.py                        ✅ Vanilla MPPI
├── test_tube_mppi.py                   ✅ Tube-MPPI (4 tests)
├── test_log_mppi.py                    ✅ Log-MPPI (4 tests)
├── test_tsallis_mppi.py                ✅ Tsallis-MPPI (5 tests)
├── test_risk_aware_mppi.py             ✅ Risk-Aware (6 tests)
├── test_smooth_mppi.py                 ✅ Smooth MPPI (5 tests)
├── test_stein_variational_mppi.py      ✅ SVMPC (6 tests)
├── test_spline_mppi.py                 ✅ Spline-MPPI (6 tests)
└── test_svg_mppi.py                    ✅ SVG-MPPI (6 tests)
```

**MPPI 테스트**: 43개 ✅ All Passing

### 학습 모델 테스트

```
tests/
├── test_neural_dynamics.py             ✅ Neural Dynamics (5 tests)
├── test_gaussian_process_dynamics.py   ✅ GP Dynamics (불확실성 정량화)
├── test_residual_dynamics.py           ✅ Residual Dynamics (하이브리드)
└── test_online_learning.py             ✅ Online Learning (적응 검증)
```

**학습 모델 테스트**: 5개 ✅ All Passing

**총 테스트**: 48개 ✅ All Passing

## 모델별 비교 데모

각 변형에 대해 Kinematic/Dynamic/Learned 모델 비교 완료:

```
examples/comparison/
├── smooth_mppi_models_comparison.py    ✅
├── svmpc_models_comparison.py          ✅
├── spline_mppi_models_comparison.py    ✅
└── svg_mppi_models_comparison.py       ✅
```

## 벤치마크 도구

- **전체 변형 벤치마크**: `examples/mppi_all_variants_benchmark.py` ✅
  - 9개 변형 동시 비교
  - 성능 메트릭 수집
  - 9패널 시각화 (XY 궤적, RMSE, Solve Time, 레이더 차트 등)

## 커밋 히스토리

```
bedfec0 - feat: add SVG-MPPI with Guide Particle SVGD
9c1c7ed - feat: add Spline-MPPI with B-spline control interpolation
4945838 - feat: add Stein Variational MPPI (SVMPC)
399cff6 - feat: add Smooth MPPI with input-lifting
7a01534 - feat: add Risk-Aware MPPI with CVaR
d1790d6 - feat: add Tsallis-MPPI with q-exponential
cd736f3 - feat: add Log-MPPI with log-space softmax
f9052de - feat: add Tube-MPPI with ancillary controller
```

## 참고 논문

### Vanilla MPPI
- Williams et al. (2016) - "Aggressive Driving with MPPI"
- Williams et al. (2017) - "Information Theoretic MPC"

### M2 고도화
- Williams et al. (2018) - "Robust Sampling Based MPPI" (Tube-MPPI)
- Bhardwaj et al. (2020) - "Blending MPC & Value Function"

### M3 SOTA 변형
- Yin et al. (2021) - "Tsallis Entropy for MPPI"
- Yin et al. (2023) - "Risk-Aware MPPI"
- Lambert et al. (2020) - "Stein Variational MPC"

### M3.5 확장 변형
- Kim et al. (2021) - "Smooth MPPI"
- Bhardwaj et al. (2024) - "Spline-MPPI"
- Kondo et al. (2024) - "SVG-MPPI"

## Safety-Critical Control (8종) ✅

| # | Method | 파일 | 핵심 |
|---|--------|------|------|
| 1 | Standard CBF | `cbf_mppi.py` | Distance-based barrier + QP filter |
| 2 | C3BF | `c3bf_cost.py` | Relative velocity-aware collision cone |
| 3 | DPCBF | `dpcbf_cost.py` | LoS + Gaussian-shaped adaptive boundary |
| 4 | Optimal-Decay | `optimal_decay_cbf_filter.py` | Joint (u,ω) optimization |
| 5 | Gatekeeper | `gatekeeper.py` | Backup trajectory infinite-time safety |
| 6 | Backup CBF | `backup_cbf_filter.py` | Sensitivity propagation multi-constraint QP |
| 7 | Multi-Robot CBF | `multi_robot_cbf.py` | Pairwise inter-robot collision avoidance |
| 8 | Shield-MPPI | `shield_mppi.py` | Per-timestep analytical CBF enforcement |

## GPU 가속 ✅

- `gpu/` 패키지: TorchDiffDriveKinematic, TorchCompositeCost, TorchGaussianSampler
- RTX 5080: K=4096→4.4x, K=8192→8.1x speedup
- `device="cuda"` 설정만으로 활성화 (기존 CPU 코드 무수정)

## 학습 모델 9종 ✅

| # | 모델 | 파일 | 특징 |
|---|------|------|------|
| 1 | Neural Network | `neural_dynamics.py` | PyTorch MLP end-to-end |
| 2 | Gaussian Process | `gaussian_process_dynamics.py` | GPyTorch sparse GP + 불확실성 |
| 3 | Residual | `residual_dynamics.py` | Physics + learned correction |
| 4 | Ensemble | `ensemble_dynamics.py` | M개 MLP 앙상블 불확실성 |
| 5 | MC-Dropout | `mc_dropout_dynamics.py` | 추론 시 dropout 불확실성 |
| 6 | MAML | `maml_dynamics.py` | FOMAML/Reptile few-shot adaptation |
| 7 | EKF Adaptive | `ekf_dynamics.py` | 파라미터 실시간 추정 (오프라인 불필요) |
| 8 | L1 Adaptive | `l1_adaptive_dynamics.py` | 외란 추정 + 저역통과 필터 |
| 9 | ALPaCA | `alpaca_dynamics.py` | Bayesian linear regression 적응 |

## 최적화 히스토리

### SVMPC SPSA 최적화 (2026-02-21)
- **Before**: 1464ms/step (per-dim finite diff = 60 rollouts/iteration)
- **After**: 113ms/step (SPSA = 2 rollouts/iteration)
- **Speedup**: 13x
- K=256 최적: 26ms/step, RMSE=0.009m

### Smooth MPPI cumsum 벡터화 (2026-02-21)
- Python for-loop → `np.cumsum` 벡터화

### Simulator 메모리 최적화 (2026-02-21)
- `store_info=False` 파라미터 추가 (300-500MB/cell 절약)

### Slalom 궤적 수정 (2026-02-21)
- 적응형 진폭: `A_eff = min(amp, v_budget / (2π·f_inst))`
- 전 구간 v_max 이내 보장

## 벤치마크 도구

- `examples/mppi_all_variants_benchmark.py`: 9종 벤치마크
- `examples/comparison/mppi_variant_trajectory_grid_demo.py`: 변형×궤적 그리드
  - `--mode obstacle`: 장애물 회피 모드
  - `--with-cbf`: CBF/Shield 변형 추가
- `examples/comparison/model_mismatch_comparison_demo.py`: 10-Way 비교

## 다음 단계 (M4)

### ROS2 통합
- [ ] nav2 Controller 플러그인
- [ ] RVIZ 시각화 고도화
- [ ] 실시간 성능 최적화

### C++ 포팅
- [ ] C++ MPPI 코어 (Eigen)
- [ ] pybind11 바인딩

## 통계

- **총 코드 라인**: ~27,000+ 라인
- **최종 업데이트**: 2026-02-21
- **테스트**: 487개 (37 파일, 모두 통과 ✅)
- **MPPI 변형**: 9개 ✅
- **안전 제어**: 8개 ✅
- **학습 모델**: 9개 ✅ (Neural/GP/Residual/Ensemble/MC-Dropout/MAML/EKF/L1/ALPaCA)
- **로봇 모델**: 5개 ✅ (DiffDrive/Ackermann/Swerve × Kinematic/Dynamic)
- **시뮬레이션**: 10 시나리오 + 4 외란 프로필
- **GPU 가속**: RTX 5080 K=8192→8.1x
