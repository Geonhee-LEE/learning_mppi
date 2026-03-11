# Testing Guide

## Overview

learning_mppi는 870개의 단위 테스트를 통해 모든 MPPI 변형, 안전 제어, 로봇 모델, 학습 모델의 정확성을 검증합니다.

```
┌──────────────────────────────────────────────────┐
│  Test Summary (2026-03-07)                       │
├──────────────────────────────────────────────────┤
│  Total:      890 tests                           │
│  Files:      57 test files                       │
│  Status:     ALL PASSED                          │
│  Duration:   ~12 seconds                         │
│  Python:     3.12.12                             │
│  Framework:  pytest 9.0.2                        │
│  Failures:   0                                   │
│  Skipped:    0                                   │
└──────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 전체 테스트 실행
python -m pytest tests/ -v --override-ini="addopts="

# coverage 포함 (pytest-cov 필요)
python -m pytest tests/ -v

# 특정 파일
python -m pytest tests/test_base_mppi.py -v --override-ini="addopts="

# 특정 함수
python -m pytest tests/test_base_mppi.py::test_circle_tracking -v --override-ini="addopts="

# 키워드 필터
python -m pytest tests/ -k "shield" -v --override-ini="addopts="
```

> **Note**: `pyproject.toml`에 `addopts = "-v --cov=mppi_controller --cov-report=term-missing"`이 설정되어 있어 `pytest-cov`가 미설치된 환경에서는 `--override-ini="addopts="`를 추가해야 합니다.

## Test Categories

### 1. MPPI Controllers (12 files, 97 tests)

12종 MPPI 변형 알고리즘의 핵심 동작을 검증합니다.

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_base_mppi.py` | 12 | Vanilla MPPI: 원형 궤적 추적, 제어 출력 범위, 비용 수렴, 초기화 |
| `test_tube_mppi.py` | 4 | Tube-MPPI: ancillary controller, tube 활성/비활성, 통계 |
| `test_log_mppi.py` | 4 | Log-MPPI: log-space softmax, 수치 안정성, 가중치 비교 |
| `test_tsallis_mppi.py` | 5 | Tsallis-MPPI: q=1 수렴, 탐색/활용 조절, 극한값 |
| `test_risk_aware_mppi.py` | 6 | Risk-Aware: CVaR 절단, 보수적 제어, 위험 레벨 |
| `test_smooth_mppi.py` | 5 | Smooth MPPI: input-lifting, delta-u 최소화, 평활도 |
| `test_spline_mppi.py` | 6 | Spline-MPPI: B-spline 보간, knot 수, 메모리 절감 |
| `test_svg_mppi.py` | 6 | SVG-MPPI: guide particle, SVGD 커널, 정확도 |
| `test_stein_variational_mppi.py` | 6 | SVMPC: Stein variational, SPSA gradient, 다양성 |
| `test_gpu_mppi.py` | 7 | GPU: CUDA 가용성, CPU fallback, 배치 처리 |
| `test_uncertainty_mppi.py` | 16 | Uncertainty MPPI: 적응 샘플링, 3전략, sigma 스케일링, 통계 |
| `test_c2u_mppi.py` | 20 | **C2U-MPPI: Unscented Transform, σ-point 전파, ChanceConstraintCost, 유효 반경** |

```bash
# MPPI 컨트롤러 전체 테스트
python -m pytest tests/test_base_mppi.py tests/test_tube_mppi.py tests/test_log_mppi.py \
  tests/test_tsallis_mppi.py tests/test_risk_aware_mppi.py tests/test_smooth_mppi.py \
  tests/test_spline_mppi.py tests/test_svg_mppi.py tests/test_stein_variational_mppi.py \
  tests/test_gpu_mppi.py tests/test_uncertainty_mppi.py tests/test_c2u_mppi.py \
  -v --override-ini="addopts="
```

### 2. Safety-Critical Control (14 files, 176 tests)

22종 안전 제어 방법의 장애물 회피 및 안전성을 검증합니다.

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_cbf_mppi.py` | 7 | CBF-MPPI: CBF 비용 + QP 필터, 장애물 거리 |
| `test_shield_mppi.py` | 10 | Shield-MPPI: per-step CBF enforcement, 궤적 안전성 |
| `test_adaptive_shield.py` | 7 | Adaptive Shield: α(d,v) 적응, 거리/속도별 α 변화 |
| `test_adaptive_shield_svg.py` | 9 | Adaptive Shield-SVG: Shield+SVGD 결합, 통계 |
| `test_shield_svg_mppi.py` | 7 | Shield-SVG-MPPI: 안전성 + 샘플 품질 |
| `test_cbf_guided_sampling.py` | 8 | CBF-Guided: ∇h 편향 샘플링, 거부 샘플링 |
| `test_hard_cbf.py` | 5 | Hard CBF: 이진 거부 (h<0 → 1e6) |
| `test_horizon_cbf.py` | 6 | Horizon-Weighted: γ^t 시간 할인 |
| `test_safety_advanced.py` | 20 | C3BF, DPCBF, Optimal-Decay CBF 고급 안전 |
| `test_safety_s3.py` | 23 | Backup CBF, Multi-Robot CBF, MPCC, MPS |
| `test_gatekeeper_superellipsoid.py` | 19 | Gatekeeper 안전 Shield + 비원형 장애물 |
| `test_conformal_cbf.py` | 30 | CP/ACP 동적 마진, 커버리지 보장, ConformalCBFMPPIController |
| `test_neural_cbf.py` | 18 | **Neural CBF: MLP h(x) 학습, 비볼록 장애물 분류, Cost/Filter/MPPI 통합** |

```bash
# Safety-Critical 전체 테스트
python -m pytest tests/test_cbf_mppi.py tests/test_shield_mppi.py tests/test_adaptive_shield.py \
  tests/test_adaptive_shield_svg.py tests/test_shield_svg_mppi.py tests/test_cbf_guided_sampling.py \
  tests/test_hard_cbf.py tests/test_horizon_cbf.py tests/test_safety_advanced.py \
  tests/test_safety_s3.py tests/test_gatekeeper_superellipsoid.py tests/test_conformal_cbf.py \
  tests/test_neural_cbf.py -v --override-ini="addopts="
```

### 3. Robot Models (1 file, 69 tests)

3종 로봇 모델 × 기구학/동역학 변형의 정방향 동역학, 속도 제한, 상태 차원을 검증합니다.

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_robot_models.py` | 69 | DiffDrive/Ackermann/Swerve × Kinematic/Dynamic: forward_dynamics, 속도 제한, 상태 차원, 배치 처리 |

```bash
python -m pytest tests/test_robot_models.py -v --override-ini="addopts="
```

### 4. Learning Models (10 files, 180 tests)

10종 학습 기반 동역학 모델의 학습/추론/적응을 검증합니다.

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_neural_dynamics.py` | 8 | Neural NN: MLP 구조, forward, 학습, 저장/로드 |
| `test_gaussian_process_dynamics.py` | 9 | GP: sparse GP 학습, 불확실성 출력, 예측 |
| `test_residual_dynamics.py` | 5 | Residual: 물리+학습 보정, 하이브리드 모델 |
| `test_ensemble_validator_uncertainty.py` | 14 | Ensemble NN: 다중 모델 앙상블, 불확실성 추정 |
| `test_mc_dropout_checkpoint.py` | 17 | MC-Dropout: 베이지안 NN, dropout 불확실성, 체크포인트 |
| `test_maml.py` | 39 | MAML: 메타 학습, few-shot 적응, FOMAML/Reptile, Residual MAML |
| `test_ekf_dynamics.py` | 18 | EKF: 확장 칼만 필터, 파라미터 추정, 온라인 적응 |
| `test_l1_adaptive.py` | 17 | L1 Adaptive: 외란 추정, low-pass 필터, 적응 제어 |
| `test_alpaca.py` | 23 | ALPaCA: 베이지안 선형 회귀, 사후 업데이트, 불확실성 |
| `test_lotf.py` | 35 | LotF: LoRA 적응, Spectral 정규화, 미분가능 시뮬레이터, BPTT 학습, NN-Policy |

```bash
# 학습 모델 전체 테스트
python -m pytest tests/test_neural_dynamics.py tests/test_gaussian_process_dynamics.py \
  tests/test_residual_dynamics.py tests/test_ensemble_validator_uncertainty.py \
  tests/test_mc_dropout_checkpoint.py tests/test_maml.py tests/test_ekf_dynamics.py \
  tests/test_l1_adaptive.py tests/test_alpaca.py tests/test_lotf.py \
  -v --override-ini="addopts="
```

### 5. Core Components (6 files, 59 tests)

비용 함수, 노이즈 샘플링, 동역학 래퍼, 궤적 생성, 시뮬레이터, 메트릭 등 핵심 인프라를 검증합니다.

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_cost_functions.py` | 14 | StateTrackingCost, ObstacleCost, ControlRateCost 등 |
| `test_sampling.py` | 10 | Gaussian, Halton, Colored Noise 샘플러 |
| `test_dynamics_wrapper.py` | 7 | BatchDynamicsWrapper: 배치 rollout, 상태 전파 |
| `test_trajectory.py` | 12 | circle, figure8, sine, slalom, straight 궤적 |
| `test_simulator.py` | 8 | Simulator: reset, run, history 기록 |
| `test_metrics.py` | 8 | RMSE, MAE, R², rollout error 계산 |

```bash
python -m pytest tests/test_cost_functions.py tests/test_sampling.py tests/test_dynamics_wrapper.py \
  tests/test_trajectory.py tests/test_simulator.py tests/test_metrics.py \
  -v --override-ini="addopts="
```

### 6. Perception (2 files, 17 tests)

LaserScan 기반 장애물 감지 및 추적 파이프라인을 검증합니다.

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_obstacle_detector.py` | 8 | LaserScan → 장애물 변환, 클러스터링 |
| `test_obstacle_tracker.py` | 9 | Nearest-neighbor 추적, 속도 추정 |

### 7. Data Pipeline (3 files, 45 tests)

데이터 수집, 학습 파이프라인, 온라인 학습을 검증합니다.

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_data_pipeline.py` | 15 | DataCollector, 데이터 분할, 정규화 |
| `test_trainers.py` | 13 | NNTrainer, GPTrainer: 학습, 저장/로드, early stopping |
| `test_online_learner.py` | 17 | OnlineLearner: 실시간 업데이트, 버퍼, 체크포인트 |

### 8. Nav2 Integration (5 files, 36 tests)

ROS2 Nav2 통합 컴포넌트를 검증합니다 (ROS2 없이 단위 테스트).

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_follow_path_integration.py` | 8 | FollowPath 액션 서버 로직 |
| `test_costmap_converter.py` | 8 | Costmap → 장애물 리스트 변환 |
| `test_path_windower.py` | 10 | 경로 윈도우 추출, lookahead |
| `test_goal_checker.py` | 5 | 목표 도달 판정 |
| `test_progress_checker.py` | 5 | 진행 상태 모니터링 |

### 9. 6-DOF Mobile Manipulator Benchmark (1 file, 18 tests)

6-DOF 모바일 매니퓰레이터에 대한 8-Way 학습 모델 벤치마크 컴포넌트를 검증합니다.

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_6dof_learned_benchmark.py` | 18 | 모델 팩토리 (8), 메트릭 계산 (3), 시뮬레이션 (4), 학습 파이프라인 (3) |

```bash
python -m pytest tests/test_6dof_learned_benchmark.py -v --override-ini="addopts="
```

### 10. Others (2 files, 14 tests)

| Test File | Tests | 검증 항목 |
|-----------|-------|----------|
| `test_dynamic_obstacles.py` | 7 | 동적 장애물 회피 시나리오 |
| `test_gpu_mppi.py` | 7 | GPU 가속 (CUDA 미설치 시 CPU fallback) |

## Test Architecture

### 테스트 구조

```
tests/
├── test_base_mppi.py              # Vanilla MPPI 핵심
├── test_tube_mppi.py              # Tube-MPPI 외란 강건성
├── test_log_mppi.py               # Log-MPPI 수치 안정성
├── test_tsallis_mppi.py           # Tsallis 탐색/활용
├── test_risk_aware_mppi.py        # CVaR 위험 인식
├── test_smooth_mppi.py            # Input-lifting 평활도
├── test_spline_mppi.py            # B-spline 메모리 절감
├── test_svg_mppi.py               # SVG guide particle
├── test_stein_variational_mppi.py # SVMPC SVGD 다양성
├── test_uncertainty_mppi.py      # Uncertainty-Aware 적응 샘플링
├── test_c2u_mppi.py              # C2U-MPPI: UT 공분산 전파 + Chance Constraint
├── test_cbf_mppi.py               # CBF 비용+QP 필터
├── test_shield_mppi.py            # Shield rollout 안전성
├── test_adaptive_shield.py        # α(d,v) 적응형 Shield
├── test_adaptive_shield_svg.py    # Shield+SVG 결합
├── test_shield_svg_mppi.py        # Shield-SVG-MPPI
├── test_cbf_guided_sampling.py    # ∇h 편향 샘플링
├── test_hard_cbf.py               # 이진 거부 CBF
├── test_horizon_cbf.py            # 시간 할인 CBF
├── test_safety_advanced.py        # C3BF/DPCBF/OptDecay
├── test_safety_s3.py              # Backup CBF/Multi-Robot/MPCC
├── test_gatekeeper_superellipsoid.py  # Gatekeeper + 비원형 장애물
├── test_conformal_cbf.py          # CP/ACP 동적 안전 마진
├── test_neural_cbf.py             # Neural CBF: MLP h(x) 학습, 비볼록 장애물
├── test_mps.py                    # Model Predictive Shield
├── test_robot_models.py           # 3종 × 2 로봇 모델
├── test_neural_dynamics.py        # Neural NN 동역학
├── test_gaussian_process_dynamics.py  # GP 동역학
├── test_residual_dynamics.py      # Residual 하이브리드
├── test_ensemble_validator_uncertainty.py  # Ensemble NN
├── test_mc_dropout_checkpoint.py  # MC-Dropout 베이지안
├── test_maml.py                   # MAML 메타 학습
├── test_ekf_dynamics.py           # EKF 적응
├── test_l1_adaptive.py            # L1 적응 제어
├── test_alpaca.py                 # ALPaCA 베이지안
├── test_cost_functions.py         # 비용 함수
├── test_sampling.py               # 노이즈 샘플러
├── test_dynamics_wrapper.py       # 배치 동역학
├── test_trajectory.py             # 궤적 생성
├── test_simulator.py              # 시뮬레이터
├── test_metrics.py                # 평가 메트릭
├── test_obstacle_detector.py      # 장애물 감지
├── test_obstacle_tracker.py       # 장애물 추적
├── test_data_pipeline.py          # 데이터 파이프라인
├── test_trainers.py               # NN/GP 트레이너
├── test_online_learner.py         # 온라인 학습
├── test_6dof_learned_benchmark.py  # 6-DOF 학습 모델 8-Way 벤치마크
├── test_lotf.py                   # LotF: LoRA/Spectral/DiffSim/BPTT/NN-Policy
├── test_dynamic_obstacles.py      # 동적 장애물
├── test_gpu_mppi.py               # GPU 가속
├── test_follow_path_integration.py    # Nav2 FollowPath
├── test_costmap_converter.py      # Nav2 Costmap 변환
├── test_path_windower.py          # Nav2 Path Windower
├── test_goal_checker.py           # Nav2 Goal Checker
└── test_progress_checker.py       # Nav2 Progress Checker
```

### 테스트 규칙

1. **명명 규칙**: `test_<module>.py` 파일, `test_<function>` 함수
2. **외부 의존성 없음**: ROS2, GPU, 외부 서비스 없이 실행 가능
3. **빠른 실행**: 전체 890개 테스트 ~12초 완료
4. **pytest 설정**: `pyproject.toml`의 `[tool.pytest.ini_options]` 참조

### pyproject.toml 설정

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=mppi_controller --cov-report=term-missing"
```

## Performance Criteria

| 항목 | 기준 | 현재 상태 |
|------|------|----------|
| 위치 추적 RMSE | < 0.2m (원형 궤적) | 0.005~0.006m (SVG/Vanilla) |
| 계산 시간 | < 100ms (K=1024, N=30) | 5.0ms (Vanilla) |
| 실시간성 | 10Hz 제어 주기 | 200Hz+ 달성 가능 |
| 안전성 | 0 collision (22종) | 100% 달성 |
| C2U 안전 여유 | MinClr > 0.4m | 0.4~0.97m |
| 테스트 실행 시간 | < 30s | ~12s |

## Troubleshooting

### pytest-cov 미설치 오류

```
ERROR: unrecognized arguments: --cov=mppi_controller --cov-report=term-missing
```

**해결**: `--override-ini="addopts="` 플래그 추가

```bash
python -m pytest tests/ -v --override-ini="addopts="
```

또는 `pytest-cov` 설치:

```bash
pip install pytest-cov
```

### Import 오류

PYTHONPATH가 설정되지 않은 경우:

```bash
# 방법 1: PYTHONPATH 설정
PYTHONPATH=. python -m pytest tests/ -v --override-ini="addopts="

# 방법 2: 패키지 설치 (editable mode)
pip install -e .
python -m pytest tests/ -v --override-ini="addopts="
```

### GPU 테스트 건너뛰기

GPU가 없는 환경에서도 `test_gpu_mppi.py`는 CPU fallback으로 정상 통과합니다. CUDA 관련 테스트는 자동으로 적절히 처리됩니다.
