# MPPI 튜토리얼 가이드

이 문서는 learning_mppi 프로젝트의 전체 기능을 단계별로 안내합니다.
43종 MPPI 변형, 27종 안전 제어, 14종 학습 모델을 포괄하는 실습 가이드입니다.

---

## 목차

1. [환경 설정](#1-환경-설정)
2. [기본 MPPI 제어 (기구학)](#2-기본-mppi-제어-기구학)
3. [동역학 모델 제어](#3-동역학-모델-제어)
4. [MPPI 변형 30종 벤치마크](#4-mppi-변형-30종-벤치마크)
5. [안전 제어 (CBF / Shield / Adaptive)](#5-안전-제어-cbf--shield--adaptive)
6. [모델 학습 (NN / GP / Residual / Ensemble)](#6-모델-학습-nn--gp--residual--ensemble)
7. [메타 학습 및 온라인 적응](#7-메타-학습-및-온라인-적응-maml--lora--ekf--l1--alpaca)
8. [고급: LotF / BPTT / DiffSim / NN-Policy](#8-고급-lotf--bptt--diffsim--nn-policy)
9. [불확실성 기반 제어](#9-불확실성-기반-제어-uncertainty--conformal--c2u-mppi)
10. [시뮬레이션 환경 (S1-S13)](#10-시뮬레이션-환경-s1-s13)
11. [GPU 가속](#11-gpu-가속)
12. [cbfkit-inspired 안전 기법 실습](#12-cbfkit-inspired-안전-기법-실습)

---

## 1. 환경 설정

프로젝트 의존성을 설치하고 실행 환경을 구성합니다.
기본 패키지(NumPy, SciPy, Matplotlib)만으로 MPPI 핵심 기능을 사용할 수 있으며,
학습 모델을 사용하려면 `[ml]` 옵션을 추가합니다.

### 설치

```bash
# 저장소 클론
git clone https://github.com/Geonhee-LEE/learning_mppi.git
cd learning_mppi

# 기본 설치 (MPPI 핵심 + 시뮬레이션)
pip install -e .

# ML 의존성 포함 설치 (PyTorch, GPyTorch)
pip install -e ".[ml]"

# 개발 도구 포함 설치
pip install -e ".[dev]"

# GPU 가속 (CUDA 11.x)
pip install -e ".[gpu]"
```

### PYTHONPATH 설정

데모 실행 시 프로젝트 루트를 PYTHONPATH에 포함해야 합니다.

```bash
# 방법 1: 환경 변수 설정
export PYTHONPATH=/path/to/learning_mppi:$PYTHONPATH

# 방법 2: 실행 시 인라인 지정
PYTHONPATH=. python examples/kinematic/mppi_differential_drive_kinematic_demo.py

# 방법 3: pip install -e . 로 설치한 경우 자동 인식
```

### 테스트 실행

```bash
# 전체 테스트 (1351+개, ~26초)
python -m pytest tests/ -v --override-ini="addopts="

# 특정 카테고리
python -m pytest tests/test_base_mppi.py -v --override-ini="addopts="
```

### 주요 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| numpy | >= 1.21.0 | 배열 연산, 핵심 수치 계산 |
| scipy | >= 1.7.0 | 최적화, QP 솔버 |
| matplotlib | >= 3.4.0 | 시각화 |
| torch | >= 2.0.0 | 신경망 학습 (선택) |
| gpytorch | >= 1.11.0 | 가우시안 프로세스 (선택) |

---

## 2. 기본 MPPI 제어 (기구학)

3종 기구학 모델(DiffDrive, Ackermann, Swerve)로 궤적 추적을 수행합니다.
기구학 모델은 속도 입력을 직접 상태 변화로 변환하며,
마찰/관성 없이 이상적인 모션을 가정합니다.

### 2.1 Differential Drive (차동 구동)

```bash
# 원형 궤적 추적
PYTHONPATH=. python examples/kinematic/mppi_differential_drive_kinematic_demo.py \
    --trajectory circle --duration 30

# Figure-8 궤적 (라이브 애니메이션)
PYTHONPATH=. python examples/kinematic/mppi_differential_drive_kinematic_demo.py \
    --trajectory figure8 --live

# Headless 모드 (서버 환경)
PYTHONPATH=. python examples/kinematic/mppi_differential_drive_kinematic_demo.py \
    --trajectory circle --no-plot
```

### 2.2 Ackermann (아커만 조향)

```bash
PYTHONPATH=. python examples/kinematic/mppi_ackermann_demo.py \
    --trajectory circle --duration 30

PYTHONPATH=. python examples/kinematic/mppi_ackermann_demo.py \
    --trajectory slalom --live
```

### 2.3 Swerve Drive (스워브 구동)

```bash
PYTHONPATH=. python examples/kinematic/mppi_swerve_drive_demo.py \
    --trajectory circle --duration 30

PYTHONPATH=. python examples/kinematic/mppi_swerve_drive_demo.py \
    --trajectory figure8 --live
```

### 주요 파라미터 (MPPIParams)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `K` | 1024 | 샘플 궤적 수 |
| `N` | 30 | 예측 호라이즌 (타임스텝) |
| `lambda_` | 1.0 | 온도 파라미터 (낮을수록 탐욕적) |
| `sigma` | [0.5, 0.3] | 제어 입력 노이즈 표준편차 |
| `dt` | 0.1 | 시간 간격 (초) |

### 기대 결과

- 위치 추적 RMSE: < 0.2m (원형 궤적)
- 계산 시간: < 50ms (K=1024, N=30)
- 지원 궤적: circle, figure8, sine, slalom, straight

---

## 3. 동역학 모델 제어

기구학 모델과 동역학 모델의 차이를 비교합니다.
동역학 모델은 마찰, 관성, 토크 제한 등 물리적 특성을 반영하여
실제 로봇에 더 가까운 동작을 생성합니다.

### 동역학 데모

```bash
# 동역학 모델 단독 실행
PYTHONPATH=. python examples/dynamic/mppi_differential_drive_dynamic_demo.py \
    --trajectory circle --duration 30

# 프로세스 노이즈 추가
PYTHONPATH=. python examples/dynamic/mppi_differential_drive_dynamic_demo.py \
    --trajectory figure8 --noise 0.3 --live
```

### 기구학 vs 동역학 비교

```bash
PYTHONPATH=. python examples/comparison/kinematic_vs_dynamic_demo.py \
    --trajectory circle --duration 20

PYTHONPATH=. python examples/comparison/kinematic_vs_dynamic_demo.py --no-plot
```

### 동역학 추가 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `mass` | 10.0 | 로봇 질량 (kg) |
| `inertia` | 0.5 | 관성 모멘트 (kg*m^2) |
| `friction` | 0.1 | 마찰 계수 |
| `max_torque` | 5.0 | 최대 토크 (Nm) |

### 기대 결과

- 동역학 모델은 기구학 대비 약간 높은 RMSE (마찰/관성 효과)
- 급격한 방향 전환 시 동역학 모델이 더 현실적인 경로 생성
- 노이즈 환경에서 동역학 모델이 더 강건한 추적 성능

---

## 4. MPPI 변형 30종 벤치마크

30가지 MPPI 변형 알고리즘을 동시에 비교하여 성능을 평가합니다.
각 변형은 특정 문제(분포 왜곡, 위험 회피, 샘플 다양성 등)를
해결하기 위해 설계되었습니다.

### 전체 벤치마크 실행

```bash
# 원형 궤적 기준 벤치마크
PYTHONPATH=. python examples/mppi_all_variants_benchmark.py \
    --trajectory circle --duration 20

# Figure-8 궤적 라이브 비교
PYTHONPATH=. python examples/mppi_all_variants_benchmark.py \
    --trajectory figure8 --duration 30 --live

# Headless 벤치마크 (테이블 출력만)
PYTHONPATH=. python examples/mppi_all_variants_benchmark.py --no-plot
```

### MPPI 변형 비교표

| # | 변형 | 핵심 아이디어 | 장점 |
|---|------|-------------|------|
| 1 | **Vanilla** | 표준 MPPI (지수 가중 평균) | 기준선, 단순 구현 |
| 2 | **Log-MPPI** | Log-space softmax 가중치 | 수치 안정성 향상 |
| 3 | **Tsallis** | q-exponential 가중치 (Tsallis 엔트로피) | 탐색/활용 균형 조절 |
| 4 | **Risk-Aware** | CVaR 가중치 절단 | 꼬리 위험 회피 |
| 5 | **SVMPC** | Stein Variational Gradient Descent | 샘플 다양성 극대화 |
| 6 | **Tube** | 명목 궤적 + 피드백 보정 | 외란 강건성 |
| 7 | **Smooth** | 입력 변화량(delta-u) 리프팅 | 부드러운 제어 입력 |
| 8 | **Spline** | B-spline 기저 함수 보간 | 저차원 탐색 공간 |
| 9 | **SVG** | Guide particle + SVGD | SVMPC 대비 효율적 |
| 10 | **DIAL** | 확산 어닐링 (반복 + 노이즈 감쇄) | 수렴 속도 향상 |
| 11 | **Uncertainty** | 불확실성 적응 샘플링 | 모델 오차 적응 |
| 12 | **C2U** | Unscented Transform + 기회 제약 | 확률적 안전 보장 |
| 13 | **Flow** | CFM 속도장 학습 → 다중 모달 샘플링 | 학습된 분포 사전 정보 |
| 14 | **Diffusion** | DDPM/DDIM 역확산 → 제어 시퀀스 생성 | 고품질 다중 모달 샘플 |
| 15 | **WBC** | 모바일 매니퓰레이터 통합 (베이스+팔) | 전신 제어 |
| 16 | **BNN** | 앙상블 불확실성 → feasibility 비용 + 필터링 | 보수적 안전 제어 |
| 17 | **Latent** | VAE 잠재 공간 롤아웃 + 배치 디코딩 | 저차원 계획 가속 |
| 18 | **CMA** | Per-timestep 대각 공분산 적응 | 적응적 탐색 분포 |
| 19 | **DBaS** | Barrier state 증강 + 적응적 탐색 노이즈 | 밀집 장애물 + 좁은 통로 |
| 20 | **Robust** | 피드백 게인 + 외란 모델링 | 외란 강건성 |
| 21 | **ASR** | Spectral Risk Measure + 적응적 왜곡 함수 | 부드러운 위험 가중 |
| 22 | **SG** | Denoising Score Matching + score-guided 샘플링 | 비용 지형 기반 유도 |
| 23 | **LP** | Butterworth 저역통과 필터 노이즈 | 주파수 도메인 smoothness |
| 24 | **Biased** | 보조 정책 혼합 + 가우시안 샘플링 | 도메인 지식 기반 local minima 탈출 |
| 25 | **Residual** | 사전 정책 nominal + 잔차 최적화 | 사전 정보 활용 + 미세 조정 |
| 26 | **TD** | TD-learned terminal value V(x_T) | 짧은 호라이즌에서 장기 계획 |
| 27 | **GN** | 가우스-뉴턴 2차 업데이트 + 라인 서치 | 비용 지형 곡률 활용 정밀 최적화 |
| 28 | **T** | Transformer 기반 초기화 학습 | 수렴 가속 + graceful degradation |
| 29 | **F** | Riccati 피드백 재사용 (75%+ 절감) | 고주파 제어 (50Hz+) |
| 30 | **Koopman** | EDMD Koopman 연산자 → 선형 특징 공간 예측 | 배치 rollout 가속 (행렬 곱 O(K*N)) |

### 변형별 고유 파라미터

```python
# Log-MPPI
LogMPPIParams(K=1024, N=30, lambda_=1.0)

# Tsallis-MPPI
TsallisMPPIParams(K=1024, N=30, q=1.5)  # q: Tsallis 파라미터

# Risk-Aware MPPI
RiskAwareMPPIParams(K=1024, N=30, alpha=0.3)  # alpha: CVaR 수준

# Tube-MPPI
TubeMPPIParams(K=1024, N=30, Q_tube=..., R_tube=...)

# SVMPC
SteinVariationalMPPIParams(K=1024, N=30, n_svgd_steps=5)

# Uncertainty-Aware MPPI
UncertaintyMPPIParams(K=1024, N=30, strategy="two_pass")
```

### 기대 결과

- 25종 알고리즘의 RMSE, 계산 시간, ESS 비교 테이블 출력
- 궤적 비교 플롯 (각 변형의 추적 경로 오버레이)
- Vanilla 대비 각 변형의 상대 성능 비율

---

## 5. 안전 제어 (CBF / Shield / Adaptive)

27종 안전 제어 기법을 장애물 환경에서 비교합니다.
(신규 cbfkit-inspired 5종 — HOCBF / Stochastic / RiskAware / Robust /
CLF-CBF-QP — 은 [12장](#12-cbfkit-inspired-안전-기법-실습)에서 실습합니다.)
CBF(Control Barrier Function) 기반 비용 함수, 안전 필터,
컨트롤러 조합으로 충돌 회피를 보장합니다.

### 5.1 기본 안전 비교 (5-Way)

```bash
# CBF / C3BF / DPCBF / Optimal-Decay / Gatekeeper 비교
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py

# 시나리오 선택
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --scenario crossing
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --scenario narrow --live

PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --no-plot
```

### 5.2 확장 안전 벤치마크 (14-Way)

```bash
# 14종 안전 기법 종합 비교
PYTHONPATH=. python examples/comparison/safety_novel_benchmark_demo.py

# 특정 시나리오 + 특정 기법
PYTHONPATH=. python examples/comparison/safety_novel_benchmark_demo.py \
    --scenario dense_static
PYTHONPATH=. python examples/comparison/safety_novel_benchmark_demo.py \
    --scenario mixed --methods 1,3,12,14

PYTHONPATH=. python examples/comparison/safety_novel_benchmark_demo.py --no-plot
```

### 5.3 적응형 안전 벤치마크 (9-Way)

모델 부정확(mismatch) 환경에서 적응 기법(EKF, L1, ALPaCA)과
안전 제어(CBF, Shield)의 조합 성능을 평가합니다.

```bash
# 전체 9종 조합 비교
PYTHONPATH=. python examples/comparison/adaptive_safety_benchmark.py

# 라이브 + 시나리오 선택
PYTHONPATH=. python examples/comparison/adaptive_safety_benchmark.py \
    --live --scenario gauntlet

# 특정 조합만 실행
PYTHONPATH=. python examples/comparison/adaptive_safety_benchmark.py --methods 1,5,7,9

PYTHONPATH=. python examples/comparison/adaptive_safety_benchmark.py --no-plot
```

### 5.4 DIAL / Shield-DIAL 벤치마크

```bash
# DIAL-MPPI vs Shield-DIAL vs Adaptive Shield-DIAL
PYTHONPATH=. python examples/comparison/shield_dial_mppi_benchmark.py

PYTHONPATH=. python examples/comparison/shield_dial_mppi_benchmark.py --no-plot
```

### 안전 제어 22종 분류

**비용 함수 (7종):**

| 비용 함수 | 특징 |
|-----------|------|
| ControlBarrierCost | 거리 기반 기본 CBF 비용 |
| NeuralBarrierCost | MLP 학습 h(x), 비볼록 장애물 대응 |
| HorizonWeightedCBFCost | 시간 할인 CBF (gamma^t 가중) |
| HardCBFCost | 이진 거부 (h<0 -> 무한 비용) |
| CollisionConeCBFCost | 속도 인지 C3BF |
| DynamicParabolicCBFCost | LoS 적응형 DPCBF |
| ChanceConstraintCost | r_eff = r + kappa * sqrt(Sigma) |

**안전 필터 (6종):**

| 필터 | 특징 |
|------|------|
| CBFSafetyFilter | QP 기반 기본 안전 필터 |
| NeuralCBFSafetyFilter | Neural CBF + autograd Lie 미분 |
| OptimalDecayCBFSafetyFilter | 이완형 CBF (relaxable) |
| BackupCBFSafetyFilter | 민감도 전파 QP |
| Gatekeeper | 백업 궤적 기반 무한시간 안전 |
| MPSController | 간소 Model Predictive Shield |

**컨트롤러 (9종):**

| 컨트롤러 | 특징 |
|---------|------|
| CBFMPPIController | CBF 비용 + QP 필터 |
| ShieldMPPIController | 롤아웃 시 per-step CBF |
| AdaptiveShieldMPPIController | 거리/속도 적응형 alpha |
| CBFGuidedSamplingMPPIController | 거부 샘플링 + 그래디언트 편향 |
| DIALMPPIController | 확산 어닐링 |
| ShieldDIALMPPIController | Shield + DIAL 결합 |
| AdaptiveShieldDIALMPPIController | Adaptive + DIAL 결합 |
| ConformalCBFMPPIController | CP/ACP 동적 마진 |
| ShieldSVGMPPIController | Shield + SVG 결합 |

### 기대 결과

- 충돌률, 최소 장애물 거리, RMSE 비교 테이블
- AdaptiveShield: 100% 안전 + RMSE 0.38m (최고 성능 조합)
- Shield-DIAL: 바람 외란 시나리오에서 100% 안전 보장

---

## 6. 모델 학습 (NN / GP / Residual / Ensemble)

물리 모델의 한계를 보완하기 위해 데이터 기반 학습 모델을 훈련합니다.
학습 파이프라인: 데이터 수집 -> 모델 학습 -> MPPI 제어 적용의 3단계로 구성됩니다.

### 6.1 신경망 학습 파이프라인

```bash
# 전체 파이프라인: 데이터 수집 -> NN 학습 -> 제어 비교
PYTHONPATH=. python examples/learned/neural_dynamics_learning_demo.py

PYTHONPATH=. python examples/learned/neural_dynamics_learning_demo.py --no-plot
```

### 6.2 GP vs Neural 비교

```bash
# 가우시안 프로세스 vs 신경망 성능 비교
PYTHONPATH=. python examples/learned/gp_vs_neural_comparison_demo.py

PYTHONPATH=. python examples/learned/gp_vs_neural_comparison_demo.py --no-plot
```

### 6.3 잔차 동역학 (Residual Dynamics)

물리 모델 + 학습 보정항의 하이브리드 접근법입니다.

```bash
# 잔차 모델 학습 및 비교
PYTHONPATH=. python examples/learned/mppi_residual_dynamics_demo.py

PYTHONPATH=. python examples/learned/mppi_residual_dynamics_demo.py --no-plot
```

### 6.4 온라인 학습

실시간 데이터로 모델을 지속적으로 개선합니다.

```bash
# 온라인 학습 파이프라인
PYTHONPATH=. python examples/learned/online_learning_demo.py

PYTHONPATH=. python examples/learned/online_learning_demo.py --no-plot
```

### 6.5 6-DOF 학습 모델 8-Way 벤치마크

모바일 매니퓰레이터 환경에서 8개 학습 모델을 비교합니다.

```bash
# 전체 8-Way 비교 (ee_3d_circle 시나리오)
PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py

# 헬릭스 시나리오
PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py --scenario ee_3d_helix

# 특정 모델만 실행
PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py \
    --models kinematic,residual_nn,oracle

PYTHONPATH=. python examples/comparison/6dof_learned_benchmark.py --no-plot
```

### 학습 모델 비교표

| 모델 | 학습 방식 | 데이터 효율 | 불확실성 추정 |
|------|---------|-----------|-------------|
| NeuralDynamics | 오프라인 MLP | 중간 | 불가 |
| GaussianProcess | 오프라인 Sparse GP | 높음 | 공분산 출력 |
| ResidualDynamics | 물리+MLP 하이브리드 | 높음 | 불가 |
| EnsembleDynamics | 5-MLP 앙상블 | 중간 | 분산 추정 |
| MCDropoutDynamics | MLP+Dropout | 중간 | MC 샘플링 |

### 기대 결과

- 학습 모델이 물리 모델 대비 RMSE 30-50% 개선
- GP 모델이 소량 데이터에서 가장 효율적
- Ensemble/MCDropout이 불확실성 추정 제공

---

## 7. 메타 학습 및 온라인 적응 (MAML / LoRA / EKF / L1 / ALPaCA)

새로운 환경에 빠르게 적응하는 메타 학습 및 온라인 적응 기법을 다룹니다.
MAML은 few-shot 적응을, EKF/L1/ALPaCA는 실시간 외란 추정을 제공합니다.

### 7.1 적응형 안전 벤치마크 (메타 학습 + 안전 제어)

```bash
# EKF / L1 / ALPaCA + CBF/Shield 조합 9종 비교
PYTHONPATH=. python examples/comparison/adaptive_safety_benchmark.py

# 모델 미스매치 없이 (완벽 모델 기준선)
PYTHONPATH=. python examples/comparison/adaptive_safety_benchmark.py --no-mismatch

PYTHONPATH=. python examples/comparison/adaptive_safety_benchmark.py --no-plot
```

### 온라인 적응 기법 비교표

| 기법 | 원리 | 적응 속도 | 메모리 |
|------|------|---------|--------|
| MAML | 메타 학습 + few-shot SGD | 5-10 스텝 | 모델 전체 |
| LoRA | Low-Rank Adaptation (~10% 파라미터) | 온라인 | 저랭크 행렬 |
| EKF | 확장 칼만 필터 외란 추정 | 1 스텝 | 공분산 행렬 |
| L1 | L1 적응 제어 (저주파 외란) | 1 스텝 | 적응 이득 |
| ALPaCA | Bayesian 선형 적응 (메타 사전분포) | 1 스텝 | 사전분포 |

### 주요 파라미터

| 파라미터 | 기법 | 설명 |
|---------|------|------|
| `inner_lr` | MAML | 내부 루프 학습률 (0.01) |
| `n_inner_steps` | MAML | 적응 스텝 수 (5) |
| `lora_rank` | LoRA | 저랭크 차수 (4-8) |
| `Q_ekf` | EKF | 프로세스 노이즈 공분산 |
| `cutoff_freq` | L1 | 저역 통과 필터 차단 주파수 |

### 기대 결과

- MAML 5-shot 적응: RMSE 0.055m (noise=0.7 환경)
- ALPaCA + Shield: 100% 안전 + 빠른 적응
- EKF + Shield: 안정적 외란 추정 + 안전 보장

---

## 8. 고급: LotF / BPTT / DiffSim / NN-Policy

Learning on the Fly(LotF) 프레임워크: LoRA 적응, Spectral 정규화,
궤적 수준 BPTT 학습, 미분가능 시뮬레이터를 통합한 고급 학습 파이프라인입니다.

### 8.1 LotF 벤치마크 (8-Way)

```bash
# 전체 8-Way 비교 (ee_3d_circle)
PYTHONPATH=. python examples/comparison/lotf_benchmark.py

# 헬릭스 시나리오
PYTHONPATH=. python examples/comparison/lotf_benchmark.py --scenario ee_3d_helix

# 특정 모델만 비교
PYTHONPATH=. python examples/comparison/lotf_benchmark.py \
    --models kinematic,bptt,lora,oracle

# 라이브 모드
PYTHONPATH=. python examples/comparison/lotf_benchmark.py \
    --live --models kinematic,oracle

PYTHONPATH=. python examples/comparison/lotf_benchmark.py --no-plot
```

### LotF 모델 비교표

| # | 모델 | 학습 방식 | 특징 |
|---|------|---------|------|
| 1 | Kinematic | 없음 (기준선) | 모델 미스매치 |
| 2 | Res-NN (MSE) | MSE 오프라인 | 단순 지도학습 |
| 3 | Res-NN (MSE+Spec) | MSE + Spectral 정규화 | 안정적 학습 |
| 4 | Res-NN (BPTT) | 궤적 수준 BPTT | 장기 오차 최소화 |
| 5 | Res-LoRA | MSE pretrain + LoRA 온라인 | ~10% 파라미터 적응 |
| 6 | Res-MAML | Meta pretrain + SGD | few-shot 적응 |
| 7 | NN-Policy (BPTT) | BC + BPTT fine-tune | MPPI 없이 직접 제어 |
| 8 | Oracle | 없음 (완벽 모델) | 성능 상한선 |

### 8.2 모바일 매니퓰레이터 데모

```bash
# End-effector 추적 (3-DOF base + 3-DOF arm)
PYTHONPATH=. python examples/mobile_manipulator_ee_tracking_demo.py

# 6-DOF 전체 제어
PYTHONPATH=. python examples/mobile_manipulator_6dof_demo.py
```

### 기대 결과

- BPTT 학습: MSE 대비 궤적 추적 오차 20-40% 개선
- LoRA 적응: 전체 파라미터의 ~10%만으로 온라인 적응
- NN-Policy: MPPI 없이 NN이 직접 (state, ee_ref) -> control 출력
- Oracle 대비 BPTT/LoRA가 가장 근접한 성능

---

## 9. 불확실성 기반 제어 (Uncertainty / Conformal / C2U-MPPI)

모델 불확실성을 명시적으로 다루는 3가지 접근법을 비교합니다.
불확실성이 클 때 보수적으로, 정확할 때 공격적으로 제어하여
안전성과 성능의 최적 균형을 달성합니다.

### 9.1 Uncertainty-Aware MPPI (5-Way)

```bash
# 기본 벤치마크 (clean 시나리오)
PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py

# 모델 미스매치 시나리오 (핵심)
PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py --scenario mismatch

# 전체 4개 시나리오
PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py --all-scenarios

# Figure-8 궤적
PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py \
    --trajectory figure8

PYTHONPATH=. python examples/comparison/uncertainty_mppi_benchmark.py --no-plot
```

**불확실성 적응 전략:**

| 전략 | 방식 | 적합 상황 |
|------|------|---------|
| `previous_traj` | 직전 궤적 기반 적응 | 저비용, 점진적 변화 |
| `current_state` | 현재 상태 전역 스케일 | 실시간 반응 |
| `two_pass` | 2-패스 적응 (최고 정확도) | 높은 정확도 요구 |

### 9.2 Conformal Prediction + CBF (5-Way)

분포 무관(distribution-free) 보장으로 동적 안전 마진을 조절합니다.
모델이 정확하면 마진을 축소하고, 부정확하면 확대합니다.

```bash
# 기본 벤치마크
PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py

# 동적 장애물 시나리오
PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py \
    --live --scenario dynamic

# 좁은 통로 시나리오
PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py \
    --live --scenario corridor

PYTHONPATH=. python examples/comparison/conformal_cbf_benchmark.py --no-plot
```

**Conformal CBF 시나리오:**

| 시나리오 | 설명 |
|---------|------|
| `accurate` | 정확한 모델, 외란 없음 |
| `mismatch` | 마찰 기반 모델 불일치 |
| `nonstationary` | 시변 바람 + 급격한 변화 |
| `dynamic` | 동적 장애물 횡단 |
| `corridor` | 좁은 L자 통로 |

### 9.3 Neural CBF (3-Way)

MLP로 학습한 h(x) barrier function으로 비볼록 장애물에 대응합니다.

```bash
# 원형 장애물 (동등 성능 확인)
PYTHONPATH=. python examples/comparison/neural_cbf_benchmark.py --scenario circular

# 비볼록 L자형 장애물 (Neural CBF 우위)
PYTHONPATH=. python examples/comparison/neural_cbf_benchmark.py --scenario non_convex

# 전체 시나리오
PYTHONPATH=. python examples/comparison/neural_cbf_benchmark.py --all-scenarios

PYTHONPATH=. python examples/comparison/neural_cbf_benchmark.py --no-plot
```

### 9.4 C2U-MPPI (3-Way)

Unscented Transform으로 공분산을 전파하고,
기회 제약(Chance Constraint)으로 확률적 안전을 보장합니다.

```bash
# 기본 벤치마크 (clean)
PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py

# 노이즈 시나리오 (C2U 우위)
PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py --scenario noisy

# 전체 시나리오
PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py --all-scenarios

PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py --no-plot
```

**C2U-MPPI 핵심 수식:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `alpha_cc` | 허용 충돌 확률 P(collision) <= alpha | 0.05 |
| `kappa_alpha` | 안전 마진 스케일 (정규분위수) | 1.645 |
| `r_eff` | 유효 반경 = r + kappa * sqrt(Sigma) | 동적 계산 |

### 9.5 BNN-MPPI (3-Way)

앙상블 불확실성으로 궤적 feasibility를 평가하고, 저신뢰 궤적을 필터링합니다.

```bash
# 기본 벤치마크 (clean)
PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py

# 노이즈 시나리오 (BNN 보수적 제어 우위)
PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py --scenario noisy

# 장애물 시나리오 (BNN 안전 영역 선호)
PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py --scenario obstacle

# 전체 시나리오
PYTHONPATH=. python examples/comparison/bnn_mppi_benchmark.py --all-scenarios
```

**BNN-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `feasibility_weight` | 불확실성 비용 가중치 β | 50.0 |
| `feasibility_threshold` | 최소 feasibility (0=필터 미적용) | 0.0 |
| `max_filter_ratio` | 최대 필터 비율 | 0.5 |
| `uncertainty_reduce` | 차원 축소 ("sum"\|"max"\|"mean") | "sum" |

### 9.6 Evidential Deep Learning (EDL) 벤치마크

단일 forward pass로 Normal-Inverse-Gamma (NIG) 분포 파라미터를 출력하여
aleatoric(데이터 노이즈)과 epistemic(모델 불확실성)을 분리합니다.
앙상블 대비 M배 빠른 추론이 핵심 장점입니다.

```bash
# 기본 벤치마크 (clean 시나리오)
PYTHONPATH=. python examples/comparison/edl_benchmark.py

# 노이즈 시나리오 (EDL 불확실성 분리 우위)
PYTHONPATH=. python examples/comparison/edl_benchmark.py --scenario noisy

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/edl_benchmark.py --scenario obstacle

# 전체 시나리오
PYTHONPATH=. python examples/comparison/edl_benchmark.py --all-scenarios

# 플롯 없이 (headless)
PYTHONPATH=. python examples/comparison/edl_benchmark.py --no-plot
```

**EDL 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `lambda_reg` | KL 정규화 가중치 (evidence penalty) | 0.1 |
| `annealing` | 정규화 어닐링 활성화 | True |
| `annealing_epochs` | 어닐링 완료 에포크 수 | 50 |
| `hidden_dims` | MLP 히든 레이어 차원 | [128, 128, 64] |

**Ensemble vs MC-Dropout vs EDL 비교:**

| | Ensemble | MC-Dropout | EDL |
|---|---|---|---|
| Forward passes | M | M | 1 |
| 파라미터 수 | M x P | P | ~P |
| 학습 비용 | M배 | 1배 | 1배 |
| 불확실성 분해 | 불가 | 불가 | aleatoric + epistemic |
| 추론 속도 | O(M) | O(M) | O(1) |

### 기대 결과

- Uncertainty-Aware: Clean 시나리오에서 Vanilla 대비 +59% 개선
- Conformal CBF: ACP가 비정상 외란에서 가장 빠른 마진 적응
- C2U-MPPI: 노이즈 환경에서 C2U > UncMPPI > Vanilla 안전성 순서
- Neural CBF: 비볼록 장애물에서 분석적 CBF 대비 명확한 우위
- BNN-MPPI: 불확실 영역 회피, obstacle 시나리오에서 Vanilla보다 안전하고 보수적
- EDL: 단일 패스로 앙상블 수준 불확실성, 추론 속도 M배 향상
- Latent-MPPI: VAE 잠재 공간 롤아웃으로 기존 비용 함수 재사용

### 9.7 Latent-Space MPPI 벤치마크

VAE 잠재 공간에서 K×N 롤아웃 후 디코딩하여 기존 비용 함수로 평가합니다.
물리 모델 직접 rollout 대비 저차원 계획의 특성을 비교합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py --scenario obstacles

# 전체 시나리오
PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py --all-scenarios

# 플롯 없이 (headless)
PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py --no-plot
```

**Latent-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `latent_dim` | VAE 잠재 공간 차원 | 16 |
| `vae_hidden_dims` | VAE MLP 은닉층 | [128, 128] |
| `vae_beta` | KL 발산 가중치 | 0.001 |
| `decode_interval` | 디코딩 간격 | 1 |

---

### 9.8 CMA-MPPI (Covariance Matrix Adaptation) 벤치마크

CMA-ES 영감의 적응적 공분산 학습으로, DIAL-MPPI의 등방적 감쇠를 비용 지형 적응적으로 대체합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py

# 장애물 시나리오 (공분산 적응 시각화)
PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --scenario obstacle

# 전체 시나리오
PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --all-scenarios

# 플롯 없이 (headless)
PYTHONPATH=. python examples/comparison/cma_mppi_benchmark.py --no-plot
```

**CMA-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `n_iters_init` | Cold start 반복 횟수 | 8 |
| `n_iters` | Warm start 반복 횟수 | 3 |
| `cov_learning_rate` | EMA 학습률 α | 0.5 |
| `sigma_min` | 최소 σ (발산 방지) | 0.05 |
| `sigma_max` | 최대 σ | 3.0 |
| `elite_ratio` | 상위 비율만 사용 (0=전체) | 0.0 |

### 9.9 DBaS-MPPI (Discrete Barrier States) 벤치마크

Barrier state 증강 + 적응적 탐색 노이즈로 밀집 장애물/좁은 통로에서 안전한 제어를 수행합니다.

```bash
# 밀집 정적 장애물 (warehouse)
PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --scenario dense_static

# 동적 교차 장애물
PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --scenario dynamic_crossing

# 좁은 통로 + 벽 제약
PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --scenario narrow_passage

# 모델 불일치 + 프로세스 노이즈
PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --scenario noisy_mismatch

# 전체 시나리오
PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --live --scenario dynamic_crossing
```

**DBaS-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `dbas_obstacles` | 원형 장애물 [(x,y,r), ...] | [] |
| `dbas_walls` | 벽 제약 [('x'\|'y', val, dir), ...] | [] |
| `barrier_weight` | Barrier 비용 가중치 $R_B$ | 10.0 |
| `barrier_gamma` | Barrier state 수렴률 $\gamma$ | 0.5 |
| `exploration_coeff` | 적응적 탐색 계수 $\mu$ | 1.0 |
| `h_min` | Barrier 클리핑 (특이점 방지) | 1e-6 |
| `safety_margin` | 추가 안전 마진 (m) | 0.1 |
| `use_adaptive_exploration` | 적응적 탐색 활성화 | True |

### 9.10 R-MPPI (Robust MPPI) 벤치마크

피드백 게인을 MPPI 샘플링 루프 내부에 통합하여, 명목/실제 궤적을 동시에 롤아웃하고
실제 궤적 기반으로 비용을 평가합니다. Tube-MPPI의 분리 구조(사후 피드백)를 개선합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py

# 노이즈 시나리오 (R-MPPI 피드백 통합 우위)
PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --scenario noisy

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --scenario obstacle

# 전체 시나리오
PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --live --scenario noisy

# 플롯 없이 (headless)
PYTHONPATH=. python examples/comparison/robust_mppi_benchmark.py --no-plot
```

**R-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `disturbance_std` | 외란 표준편차 $\sigma_d$ | 0.1 |
| `feedback_gain_scale` | 피드백 게인 스케일 | 1.0 |
| `disturbance_mode` | 외란 모드 (`"gaussian"` \| `"adversarial"` \| `"none"`) | `"gaussian"` |
| `robust_alpha` | adversarial 외란 크기 | 0.1 |
| `use_feedback` | 피드백 통합 활성화 | True |
| `n_disturbance_samples` | 외란 샘플 수 | 1 |

### 9.11 ASR-MPPI (Adaptive Spectral Risk) 벤치마크

Spectral Risk Measure (SRM)의 왜곡 함수 φ(q)로 비용 분위수를 비균일 가중하여,
CVaR의 경질 절단을 연속적 곡선으로 일반화합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/spectral_risk_mppi_benchmark.py

# 장애물 시나리오 (부드러운 가중치 우위)
PYTHONPATH=. python examples/comparison/spectral_risk_mppi_benchmark.py --scenario obstacles

# 밀집 장애물 (SRM 표현력)
PYTHONPATH=. python examples/comparison/spectral_risk_mppi_benchmark.py --scenario dense_slalom

# 전체 시나리오
PYTHONPATH=. python examples/comparison/spectral_risk_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/spectral_risk_mppi_benchmark.py --live --scenario obstacles
```

**ASR-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `distortion_type` | 왜곡 함수 (`"sigmoid"` \| `"power"` \| `"dual_power"` \| `"cvar"`) | `"sigmoid"` |
| `distortion_alpha` | 중심 파라미터 (sigmoid 전환점) | 0.5 |
| `distortion_beta` | 경사도 (sigmoid sharpness) | 5.0 |
| `distortion_gamma` | 지수 (power: q^γ) | 1.0 |
| `use_adaptive_risk` | ESS 기반 자동 β 조절 | False |
| `adaptation_rate` | 적응 속도 (EMA) | 0.1 |

### 9.12 SG-MPPI (Score-Guided) 벤치마크

Denoising Score Matching으로 비용 지형의 score function을 학습하고,
MPPI 가우시안 노이즈에 score 방향 bias를 추가하여 저비용 영역으로 유도합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py

# 장애물 시나리오 (score가 회피 방향 학습)
PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --scenario obstacles

# 다봉 비용 (경로 선택)
PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --scenario multimodal

# 전체 시나리오
PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --live --scenario obstacles
```

**SG-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `guidance_scale` | Score bias 강도 α | 0.5 |
| `guidance_decay` | 다중 반복 시 α 감쇠율 | 0.95 |
| `n_guide_iters` | Score-guided 반복 횟수 | 1 |
| `use_annealing` | DIAL-style σ 어닐링 결합 | False |
| `score_online_training` | 온라인 학습 활성화 | False |
| `score_training_interval` | 학습 주기 (스텝) | 20 |

### 9.13 LP-MPPI (Low-Pass) 벤치마크

Butterworth 저역통과 필터를 MPPI 노이즈에 적용하여
주파수 영역에서 직접적인 smoothness 제어. 2개 파라미터(f_c, order)로
물리적으로 해석 가능한 제어 부드러움 달성.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/lp_mppi_benchmark.py

# 장애물 시나리오 (부드러운 회피 궤적)
PYTHONPATH=. python examples/comparison/lp_mppi_benchmark.py --scenario obstacles

# 급격한 방향 전환 (figure8, smoothness vs agility)
PYTHONPATH=. python examples/comparison/lp_mppi_benchmark.py --scenario aggressive

# 전체 시나리오
PYTHONPATH=. python examples/comparison/lp_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/lp_mppi_benchmark.py --live --scenario obstacles
```

**LP-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `cutoff_freq` | Butterworth 차단 주파수 (Hz) | 3.0 |
| `filter_order` | Butterworth 필터 차수 | 3 |
| `normalize_variance` | 필터 후 분산 정규화 | False |

### 9.14 Biased-MPPI (Mixture Sampling) 벤치마크

보조 정책(ancillary policy) 샘플과 가우시안 샘플을 혼합하여
도메인 지식 기반 local minima 탈출. 학습 없이 정책 설계로 다중 모드 탐색.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py

# 장애물 시나리오 (정책 다양성 + 회피 경로)
PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py --scenario obstacles

# local minima 시나리오 (Biased 핵심 우위)
PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py --scenario local_minima

# 전체 시나리오
PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/biased_mppi_benchmark.py --live --scenario obstacles
```

**Biased-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `biased_policies` | 활성 보조 정책 목록 | `["pure_pursuit", "feedback", "previous_solution"]` |
| `use_adaptive_lambda` | ESS 기반 적응적 온도 | True |
| `min_ess_ratio` | 최소 ESS 비율 (이하면 λ 증가) | 0.3 |
| `max_ess_ratio` | 최대 ESS 비율 (이상이면 λ 감소) | 0.7 |
| `lambda_adaptation_rate` | λ 적응 속도 | 0.1 |
| `lookahead_distance` | PurePursuit lookahead 거리 (m) | 1.0 |

### 9.15 Residual-MPPI (사전 정책 + 잔차 최적화) 벤치마크

사전 정책(PurePursuit 등)의 출력을 명목 시퀀스로 사용하고,
MPPI가 잔차만 최적화. KL 페널티로 정책 근처 탐색 유도.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/residual_mppi_benchmark.py

# 장애물 시나리오 (잔차로 장애물 회피)
PYTHONPATH=. python examples/comparison/residual_mppi_benchmark.py --scenario obstacles

# 정책 품질 비교 (figure-8 + 장애물)
PYTHONPATH=. python examples/comparison/residual_mppi_benchmark.py --scenario policy_quality

# 전체 시나리오
PYTHONPATH=. python examples/comparison/residual_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/residual_mppi_benchmark.py --live --scenario obstacles
```

**Residual-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `policy_weight` | 사전 정책 가중치 | 1.0 |
| `use_policy_nominal` | 정책 출력을 샘플링 중심으로 | True |
| `residual_scale` | 잔차 노이즈 스케일 | 1.0 |
| `policy_type` | 기본 정책 유형 (feedback/zero/custom) | "feedback" |
| `kl_weight` | KL 발산 가중치 | 0.1 |
| `use_augmented_cost` | 증강 비용 활성화 | True |

### 9.16 TD-MPPI (Temporal-Difference) 벤치마크

TD 학습 terminal value function V(x_T)로 짧은 롤아웃에서도
장기 계획 품질 유지. N=10에서 N=30급 성능 접근.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/td_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/td_mppi_benchmark.py --scenario obstacles

# Short horizon 시나리오 (TD 핵심 우위)
PYTHONPATH=. python examples/comparison/td_mppi_benchmark.py --scenario short_horizon

# 전체 시나리오
PYTHONPATH=. python examples/comparison/td_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/td_mppi_benchmark.py --live --scenario short_horizon
```

**TD-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `td_learning_rate` | TD 학습률 | 0.001 |
| `td_gamma` | 할인율 | 0.99 |
| `td_buffer_size` | 경험 버퍼 크기 | 5000 |
| `td_update_interval` | TD 업데이트 주기 (스텝) | 5 |
| `td_min_samples` | 최소 학습 샘플 | 100 |
| `use_terminal_value` | terminal value 사용 여부 | True |
| `value_weight` | V(x_T) 가중치 | 1.0 |

### 9.17 GN-MPPI (Gauss-Newton) 벤치마크

가우스-뉴턴 2차 업데이트로 MPPI 수렴 가속. 가우시안 스무딩 기울기 +
GGN 헤시안으로 비용 곡률 방향 최적화. 표준 MPPI 폴백 안전장치 포함.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/gn_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/gn_mppi_benchmark.py --scenario obstacles

# 수렴 속도 비교 (figure8)
PYTHONPATH=. python examples/comparison/gn_mppi_benchmark.py --scenario convergence

# 전체 시나리오
PYTHONPATH=. python examples/comparison/gn_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/gn_mppi_benchmark.py --live --scenario obstacles
```

**GN-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `n_gn_iters` | GN 반복 횟수 | 3 |
| `n_gn_iters_init` | Cold start 반복 횟수 | 5 |
| `gn_step_size` | 라인 서치 초기 스텝 | 1.0 |
| `line_search_steps` | 라인 서치 후보 수 | 5 |
| `line_search_decay` | 라인 서치 감쇠율 | 0.5 |
| `regularization` | 헤시안 정규화 | 1e-4 |
| `use_gn_update` | GN 업데이트 사용 여부 | True |

### 9.18 SVG-MPPI (Stein Variational Guided) 벤치마크

SVGD 파티클 최적화로 학습 없이 다중 모드 탐색을 수행합니다.
RBF 커널 + median bandwidth로 샘플 간 반발력을 계산하여 분포 다양성을 유지합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/svg_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/svg_mppi_benchmark.py --scenario obstacles

# 다봉 비용 (다중 경로 탐색)
PYTHONPATH=. python examples/comparison/svg_mppi_benchmark.py --scenario multimodal

# 전체 시나리오
PYTHONPATH=. python examples/comparison/svg_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/svg_mppi_benchmark.py --live --scenario obstacles
```

**SVG-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `n_svgd_steps` | SVGD 반복 횟수 | 5 |
| `svgd_step_size` | SVGD 스텝 크기 | 0.1 |
| `temperature_svgd` | SVGD 온도 파라미터 | 1.0 |
| `blend_ratio` | SVGD/MPPI 혼합 비율 | 0.5 |
| `use_svgd_warm_start` | Warm start 활성화 | True |
| `use_spsa_gradient` | SPSA 기울기 추정 사용 | True |

### 9.19 pi-MPPI (Projection-based) 벤치마크

QP/clip projection으로 jerk/snap 하드 제약을 보장합니다.
LP-MPPI의 주파수 영역 필터링과 달리 시간 영역에서 직접 제약을 투영하여 smoothness를 확보합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/projection_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/projection_mppi_benchmark.py --scenario obstacles

# 급격한 방향 전환 (figure8, smoothness vs agility)
PYTHONPATH=. python examples/comparison/projection_mppi_benchmark.py --scenario aggressive

# 전체 시나리오
PYTHONPATH=. python examples/comparison/projection_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/projection_mppi_benchmark.py --live --scenario obstacles
```

**pi-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `jerk_limit` | Jerk 제약 상한 | 5.0 |
| `snap_limit` | Snap 제약 상한 | 50.0 |
| `use_jerk_constraint` | Jerk 제약 활성화 | True |
| `use_snap_constraint` | Snap 제약 활성화 | False |
| `projection_method` | 투영 방법 (`"clip"` \| `"qp"`) | `"clip"` |
| `project_samples` | 샘플 단계 투영 | True |
| `project_output` | 최종 출력 투영 | True |

### 9.20 dsMPPI (Deterministic Sampling) 벤치마크

결정론적 샘플링 + CEM 반복으로 유일한 비확률적 MPPI를 구현합니다.
Halton/Sobol 준난수 시퀀스로 K=64에서도 균일 커버리지를 달성하여 샘플 효율을 극대화합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/deterministic_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/deterministic_mppi_benchmark.py --scenario obstacles

# 저 샘플 시나리오 (K=64, dsMPPI 핵심 우위)
PYTHONPATH=. python examples/comparison/deterministic_mppi_benchmark.py --scenario low_samples

# 전체 시나리오
PYTHONPATH=. python examples/comparison/deterministic_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/deterministic_mppi_benchmark.py --live --scenario obstacles
```

**dsMPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `sampling_method` | 샘플링 방법 (`"halton"` \| `"sobol"` \| `"latin"`) | `"halton"` |
| `n_cem_iters` | CEM 반복 횟수 (warm start) | 3 |
| `n_cem_iters_init` | CEM 반복 횟수 (cold start) | 5 |
| `elite_ratio` | 상위 엘리트 비율 | 0.3 |
| `cem_alpha` | CEM EMA 학습률 | 0.7 |
| `add_random_samples` | 추가 랜덤 샘플 수 | 0 |

### 9.21 DRPA-MPPI (Dynamic Repulsive Potential) 벤치마크

반발 포텐셜 필드로 local minima를 자동 탈출합니다.
정체 감지 로직이 progress 정지를 인식하면 탈출 부스트를 가하여
학습 없이도 U-형 함정 등에서 빠져나옵니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/drpa_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/drpa_mppi_benchmark.py --scenario obstacles

# local minima 시나리오 (DRPA 핵심 우위)
PYTHONPATH=. python examples/comparison/drpa_mppi_benchmark.py --scenario local_minima

# 전체 시나리오
PYTHONPATH=. python examples/comparison/drpa_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/drpa_mppi_benchmark.py --live --scenario obstacles
```

**DRPA-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `repulsive_gain` | 반발 포텐셜 이득 | 5.0 |
| `influence_distance` | 포텐셜 영향 거리 (m) | 1.0 |
| `stagnation_threshold` | 정체 판단 임계값 | 0.1 |
| `stagnation_window` | 정체 판단 윈도우 (스텝) | 10 |
| `escape_boost` | 탈출 부스트 배율 | 2.0 |
| `recovery_threshold` | 복귀 판단 임계값 | 0.3 |

### 9.22 CSC-MPPI (Constrained Sampling Cluster) 벤치마크

Primal-dual 제약 투영 + DBSCAN 클러스터링으로 실행 가능 궤적을 선택합니다.
가중 평균 대신 클러스터별 최적 궤적을 선택하여 장애물 근처에서도 안전한 제어를 보장합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/csc_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/csc_mppi_benchmark.py --scenario obstacles

# 좁은 통로 (제약 투영 + 클러스터링 우위)
PYTHONPATH=. python examples/comparison/csc_mppi_benchmark.py --scenario narrow_passage

# 전체 시나리오
PYTHONPATH=. python examples/comparison/csc_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/csc_mppi_benchmark.py --live --scenario obstacles
```

**CSC-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `safety_margin` | 안전 마진 (m) | 0.2 |
| `n_projection_steps` | Primal-dual 투영 반복 수 | 5 |
| `projection_lr` | 투영 학습률 | 0.1 |
| `dual_lr` | 듀얼 변수 학습률 | 0.01 |
| `dbscan_eps` | DBSCAN 이웃 거리 | 1.0 |
| `dbscan_min_samples` | DBSCAN 최소 클러스터 크기 | 3 |
| `use_projection` | 제약 투영 활성화 | True |
| `use_clustering` | DBSCAN 클러스터링 활성화 | True |

### 9.23 T-MPPI (Transformer-based Initialization) 벤치마크

Transformer 모델로 과거 상태/제어 히스토리로부터 근최적 초기 제어 시퀀스를 예측합니다.
MPPI의 초기화만 개선하여 수렴 속도를 높이며, 학습 실패 시 자연스럽게 표준 MPPI로 폴백합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/transformer_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/transformer_mppi_benchmark.py --scenario obstacles

# 온라인 학습 (실시간 Transformer 학습)
PYTHONPATH=. python examples/comparison/transformer_mppi_benchmark.py --scenario online_learning

# Warm start 수렴 테스트
PYTHONPATH=. python examples/comparison/transformer_mppi_benchmark.py --scenario warm_start

# 전체 시나리오
PYTHONPATH=. python examples/comparison/transformer_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/transformer_mppi_benchmark.py --live --scenario obstacles
```

**T-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `use_transformer_init` | Transformer 초기화 활성화 | True |
| `blend_ratio` | Transformer/warm-start 혼합 비율 | 0.5 |
| `transformer_hidden_dim` | Transformer d_model | 128 |
| `transformer_n_heads` | 어텐션 헤드 수 | 4 |
| `transformer_n_layers` | 인코더 레이어 수 | 2 |
| `transformer_context_length` | 히스토리 윈도우 | 20 |
| `transformer_lr` | 학습률 | 1e-3 |
| `online_learning` | 온라인 학습 활성화 | True |

**4-Way 비교** (Vanilla vs DIAL vs Biased vs T-MPPI):
- **A. simple**: 기준선 (장애물 없음) — Transformer 초기화 효과 측정
- **B. obstacles**: 3개 장애물 — 비용 지형 복잡 환경에서 수렴 속도
- **C. online_learning**: 3개 장애물 + 온라인 학습 — 실시간 학습 효과
- **D. warm_start**: 큰 장애물 1개 — 급변 시 Transformer vs warm start 비교

### 9.24 F-MPPI (Feedback Reuse) 벤치마크

1회 MPPI 풀 솔브 후 Riccati 피드백 게인으로 여러 스텝을 보정합니다.
reuse_steps=3이면 75% 계산 절감 — 고주파 제어(50Hz+)에 적합합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --scenario obstacles

# 고주파 제어 (dt=0.02, 50Hz)
PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --scenario high_frequency

# 외란 + 장애물
PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --scenario perturbation

# 전체 시나리오
PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/feedback_mppi_benchmark.py --live --scenario obstacles
```

**F-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `use_feedback` | 피드백 재사용 활성화 | True |
| `reuse_steps` | 피드백 재사용 횟수 | 3 |
| `jacobian_eps` | 유한 차분 ε | 1e-4 |
| `feedback_weight_Q` | 상태 추적 비용 가중치 | 10.0 |
| `feedback_weight_R` | 제어 비용 가중치 | 0.1 |
| `feedback_gain_clip` | 게인 클리핑 값 | 10.0 |
| `use_warm_start` | Warm start 유지 | True |

**4-Way 비교** (Vanilla vs Tube vs Robust vs F-MPPI):
- **A. simple**: 기준선 — full solve vs reuse 비용/성능 비교
- **B. obstacles**: 3개 장애물 — 피드백 보정으로 장애물 회피 유지
- **C. high_frequency**: dt=0.02 (50Hz) — 고주파에서 계산 시간 절감 효과
- **D. perturbation**: 프로세스 노이즈 + 장애물 — 외란 하 피드백 강건성

### 9.25 C-MPPI (Contingency) 벤치마크

명목 궤적의 체크포인트에서 비상 계획(급정거 + 내부 MPPI)을 평가합니다.
모든 계획 시점에서 안전한 탈출 경로가 보장되는 안전 중심 변형입니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --scenario obstacles

# 좁은 통로 (비상 탈출 계획 필수)
PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --scenario narrow_passage

# 밀집 위험 환경
PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --scenario dynamic_risk

# 전체 시나리오
PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/contingency_mppi_benchmark.py --live --scenario obstacles
```

**C-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `n_checkpoints` | 비상 평가 체크포인트 수 | 3 |
| `contingency_weight` | 비상 비용 가중치 | 50.0 |
| `contingency_samples` | 내부 MPPI 샘플 수 | 16 |
| `contingency_horizon` | 비상 rollout 호라이즌 | 8 |
| `use_braking_contingency` | 급정거 비상 활성화 | True |
| `use_mppi_contingency` | 내부 MPPI 비상 활성화 | True |
| `safe_cost_threshold` | 안전 페널티 임계값 | 100.0 |
| `safety_cost_weight` | 안전 위반 페널티 가중치 | 200.0 |

**4-Way 비교** (Vanilla vs CBF vs DBaS vs C-MPPI):
- **A. simple**: 기준선 — 비상 계획 오버헤드 측정
- **B. obstacles**: 3개 장애물 — 안전 보장 + 추적 성능
- **C. narrow_passage**: 좁은 통로 — 비상 탈출 계획이 필수인 환경
- **D. dynamic_risk**: 5개 밀집 장애물 — 비상 비용이 경로 선택에 미치는 영향

### 9.26 DualGuard-MPPI (HJ Safety Value) 벤치마크

Signed distance + TTC 기반 안전 가치 함수로 이중 가드(sample guard + nominal guard)를 적용합니다.
soft/hard/filter 3가지 모드를 지원하며, 적응적 노이즈 부스트로 안전 샘플 비율을 유지합니다.

```bash
# 기본 벤치마크 (simple 시나리오)
PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py

# 장애물 시나리오
PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --scenario obstacles

# 밀집 장애물 (6개, 스트레스 테스트)
PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --scenario dense_obstacles

# 속도 인식 (빠른 속도 + 장애물)
PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --scenario velocity_aware

# 전체 시나리오
PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --live --scenario obstacles
```

**DualGuard-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `safety_margin` | 안전 마진 (m) | 0.2 |
| `safety_mode` | 가드 모드 (soft/hard/filter) | "soft" |
| `safety_penalty` | soft 가드 페널티 계수 | 5000.0 |
| `safety_decay` | soft 가드 감쇠율 | 8.0 |
| `use_velocity_penalty` | 속도 페널티 활성화 | True |
| `velocity_penalty_weight` | 속도 페널티 가중치 | 50.0 |
| `ttc_horizon` | TTC 호라이즌 (s) | 1.0 |
| `use_nominal_guard` | 명목 가드 활성화 | True |
| `min_safe_fraction` | 최소 안전 비율 | 0.1 |
| `noise_boost_factor` | 노이즈 부스트 배율 | 1.5 |

**4-Way 비교** (Vanilla vs CBF vs DBaS vs DualGuard):
- **A. simple**: 기준선 — 가드 오버헤드 측정
- **B. obstacles**: 3개 장애물 — MinClearance 비교 (DualGuard 우위)
- **C. dense_obstacles**: 6개 밀집 장애물 — 이중 가드 효과 극대화
- **D. velocity_aware**: 빠른 속도 + 장애물 — TTC 속도 인식 장점

### 9.27 PR-MPPI (Parameter-Robust) 벤치마크

입자 필터로 모델 파라미터 belief를 유지하고, M개 모델 가설로 동시 rollout합니다.
파라미터 불일치(wheelbase 등)에 강건하며, 온라인 Bayesian 학습으로 파라미터를 추정합니다.

```bash
# 기본 벤치마크 (mismatch 없음)
PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py

# 약한 불일치 (wheelbase 20%)
PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --scenario mild_mismatch

# 강한 불일치 (wheelbase 60%)
PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --scenario severe_mismatch

# 불일치 + 장애물
PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --scenario mismatch_obstacles

# 전체 시나리오
PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --all-scenarios

# 실시간 애니메이션
PYTHONPATH=. python examples/comparison/parameter_robust_mppi_benchmark.py --live --scenario mild_mismatch
```

**PR-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `n_particles` | 파라미터 입자 수 | 8 |
| `param_name` | 추적 파라미터 이름 | "wheelbase" |
| `param_nominal` | 명목 파라미터 값 | (시나리오 의존) |
| `param_std` | 초기 불확실성 표준편차 | 0.15 |
| `aggregation_mode` | 비용 집계 모드 | "weighted_mean" |
| `online_learning` | 온라인 파라미터 학습 | True |
| `observation_window` | 관측 히스토리 윈도우 | 10 |
| `use_resampling` | ESS 기반 재샘플링 | True |
| `resample_threshold` | 재샘플링 임계값 (ESS/M) | 0.3 |

**4-Way 비교** (Vanilla vs Tube vs Robust vs PR-MPPI):
- **A. simple**: 기준선 — 불일치 없이 PR-MPPI 오버헤드 측정
- **B. mild_mismatch**: wheelbase 0.5→0.6 (20% 불일치) — 적응 학습 효과
- **C. severe_mismatch**: wheelbase 0.5→0.8 (60% 불일치) — RMSE 35% 개선
- **D. mismatch_obstacles**: 불일치 + 3개 장애물 — 안전 + 강건성 동시 요구

### 9.28 Koopman-MPPI (Koopman Operator) 벤치마크

EDMD(Extended Dynamic Mode Decomposition)로 Koopman 연산자를 학습하여,
비선형 동역학을 선형 특징 공간에서 행렬 곱으로 예측합니다.
미학습 시 fallback dynamics를 자동으로 사용하여 graceful degradation을 보장합니다.

```bash
# 기본 벤치마크
PYTHONPATH=. python examples/comparison/koopman_mppi_benchmark.py --no-plot

# 라이브 애니메이션
PYTHONPATH=. python examples/comparison/koopman_mppi_benchmark.py --live

# 전체 시나리오
PYTHONPATH=. python examples/comparison/koopman_mppi_benchmark.py --all-scenarios
```

**Koopman-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `koopman_lift_fn` | 리프팅 함수 종류 | "rbf" |
| `koopman_lift_dim` | 리프팅 특징 차원 | 64 |
| `koopman_reg` | EDMD 정규화 계수 | 1e-4 |

**핵심 파일:**
- `mppi_controller/controllers/mppi/koopman_mppi.py` — Koopman-MPPI 컨트롤러
- `mppi_controller/models/learned/koopman_dynamics.py` — EDMD Koopman 동역학 모델

**핵심 아이디어:**
- 비선형 f(x,u)를 리프팅 함수 ψ(x)로 고차원 특징 공간에 매핑
- EDMD로 선형 연산자 K 학습: ψ(x') ≈ K·[ψ(x); u]
- 배치 rollout이 행렬 곱 O(K*N)으로 가속
- 미학습 상태에서 원본 dynamics fallback 자동 사용

### 9.29 PGD-MPPI (Preconditioned Gradient Descent) 벤치마크

MPPI를 KL 정규화 자유에너지의 전처리 경사 하강으로 재해석합니다.
스텝 크기 α·다중 경사 스텝·공분산 전처리 적응을 제공하며,
기본값에서는 Vanilla MPPI와 정확히 동일하게 동작합니다 (graceful superset).

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi import PGDMPPIController, PGDMPPIParams

model = DifferentialDriveKinematic(wheelbase=0.5)
params = PGDMPPIParams(
    K=512, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    step_size=1.0,          # 전처리 경사 스텝 α (1.0 = 표준 MPPI)
    n_grad_steps=2,         # 제어 주기당 경사 스텝 수
    adapt_covariance=True,  # Gibbs-tilted 공분산 전처리
)
controller = PGDMPPIController(model, params)

state = np.array([0.0, 0.0, 0.0])             # [x, y, θ]
reference = np.tile([2.0, 1.0, 0.0], (params.N + 1, 1))  # (N+1, 3)
control, info = controller.compute_control(state, reference)
print(control, info["pgd_stats"]["grad_norm"])
```

```bash
PYTHONPATH=. python examples/comparison/pgd_mppi_benchmark.py --all-scenarios
PYTHONPATH=. python examples/comparison/pgd_mppi_benchmark.py --live --scenario obstacles
```

**PGD-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `step_size` | 전처리 경사 스텝 크기 α (1.0 = 표준 MPPI) | 1.0 |
| `n_grad_steps` | 제어 주기당 경사 스텝 수 | 1 |
| `adapt_covariance` | Gibbs-tilted 공분산 전처리 활성화 | False |
| `cov_step_size` | 공분산 적응 비율 β (EMA) | 0.2 |
| `cov_min_scale` | 공분산 스케일 하한 (붕괴 방지) | 0.25 |
| `cov_max_scale` | 공분산 스케일 상한 (발산 방지) | 4.0 |
| `normalize_gradient` | 경사를 ESS로 정규화 | False |

### 9.30 TR-MPPI (Trust Region) 벤치마크

proposal 평균 업데이트를 KL 발산 경계 δ로 제약하여 단조롭고 안정적인 수렴을 보장합니다.
공분산 엔트로피 하한으로 조기 붕괴를 막고, Halton 저불일치 수열 + 역정규 CDF 기반
결정론적 LCD 샘플링으로 샘플 효율을 높입니다.

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi import TRMPPIController, TRMPPIParams

model = DifferentialDriveKinematic(wheelbase=0.5)
params = TRMPPIParams(
    K=512, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    trust_region_radius=1.0,         # KL 경계 δ (작을수록 보수적)
    use_kl_bound=True,
    use_deterministic_sampling=True, # Halton-LCD 결정론적 샘플링
    n_iters=2,
)
controller = TRMPPIController(model, params)

state = np.array([0.0, 0.0, 0.0])
reference = np.tile([2.0, 1.0, 0.0], (params.N + 1, 1))
control, info = controller.compute_control(state, reference)
print(control, info["tr_stats"]["kl_divergence"], info["tr_stats"]["step_scaled"])
```

```bash
PYTHONPATH=. python examples/comparison/tr_mppi_benchmark.py --all-scenarios
PYTHONPATH=. python examples/comparison/tr_mppi_benchmark.py --live --scenario obstacles
```

**TR-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `trust_region_radius` | KL 경계 δ (작을수록 보수적) | 1.0 |
| `use_kl_bound` | 신뢰 영역(KL) 평균 투영 활성화 | True |
| `n_iters` | 제어 주기당 신뢰 영역 반복 수 | 1 |
| `use_deterministic_sampling` | LCD(Halton) 결정론적 샘플링 | False |
| `adapt_covariance` | 가중 경험 공분산 적응 | False |
| `entropy_floor_scale` | σ_floor = scale·σ_base (엔트로피 하한) | 0.3 |
| `cov_max_scale` | 공분산 스케일 상한 | 4.0 |

### 9.31 RF-MPPI (Reference-Free Spline) 벤치마크

제어 시퀀스를 저차원 큐빅 Hermite 스플라인(위치+속도 dual-space)으로 파라미터화합니다.
소수의 knot 공간에서 섭동을 샘플링하여 매끄러움을 구조적으로 보장하므로,
적은 샘플(K≈32~64)로도 높은 ESS와 낮은 jerk를 달성하며 CPU 실시간 동작이 가능합니다.

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi import RFMPPIController, RFMPPIParams

model = DifferentialDriveKinematic(wheelbase=0.5)
params = RFMPPIParams(
    K=48, N=30, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    n_knots=6,                   # 스플라인 제어점 수 M (M << N)
    sample_velocity_knots=True,  # 속도 knot도 샘플링 (dual-space)
    knot_sigma_vel=0.3,
    spline_warm_shift=True,
)
controller = RFMPPIController(model, params)

state = np.array([0.0, 0.0, 0.0])
reference = np.tile([2.0, 1.0, 0.0], (params.N + 1, 1))
control, info = controller.compute_control(state, reference)
print(control, info["rf_stats"]["n_knots"], info["rf_stats"]["dual_space"])
```

```bash
PYTHONPATH=. python examples/comparison/rf_mppi_benchmark.py --all-scenarios
PYTHONPATH=. python examples/comparison/rf_mppi_benchmark.py --live --scenario obstacles
```

**RF-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `n_knots` | 스플라인 제어점 수 M (M << N) | 6 |
| `sample_velocity_knots` | 속도 knot도 샘플링 (dual-space) | True |
| `knot_sigma_pos` | 위치 knot 섭동 σ (None → sigma) | None |
| `knot_sigma_vel` | 속도 knot 섭동 σ | 0.3 |
| `clamp_endpoints_vel` | 시작/끝 속도 knot 0 고정 | False |
| `spline_warm_shift` | receding horizon knot 시프트 | True |

> **실측 하이라이트**: K=24 few-sample 조건에서 RF-dual의 MSSD가 Vanilla 대비 약 13배 매끄러우면서 RMSE는 1.83→0.32로 개선되었습니다.

### 9.32 Step-MPPI (Single-Step via DPC) 벤치마크

신경망이 MPPI proposal 분포(평균 잔차 + 대각 공분산)를 학습합니다.
장기 호라이즌 MPC 목적(비용+제약+최대 엔트로피)을 학습 시점에 주입하여,
런타임에는 짧은(단일) 스텝 lookahead만으로 장기 계획 정보를 활용합니다.
출력층 zero-init으로 학습 전에는 Vanilla MPPI로 graceful 퇴화하며, torch 부재 시에도 동작합니다.

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi import StepMPPIController, StepMPPIParams

model = DifferentialDriveKinematic(wheelbase=0.5)
params = StepMPPIParams(
    K=512, N=10, dt=0.05, lambda_=1.0,
    sigma=np.array([0.5, 0.5]),
    lookahead_steps=1,         # 단일 스텝 lookahead
    use_learned_proposal=True, # 학습 proposal (False → Vanilla)
    learn_covariance=True,     # 대각 공분산도 학습
    online_training=True,      # 온라인 자기지도 학습
)
controller = StepMPPIController(model, params)

state = np.array([0.0, 0.0, 0.0])
reference = np.tile([2.0, 1.0, 0.0], (params.N + 1, 1))
control, info = controller.compute_control(state, reference)
print(control, info["step_stats"]["use_net"], info["step_stats"]["train_count"])
```

```bash
PYTHONPATH=. python examples/comparison/step_mppi_benchmark.py --all-scenarios
PYTHONPATH=. python examples/comparison/step_mppi_benchmark.py --live --scenario obstacles
```

**Step-MPPI 핵심 파라미터:**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `lookahead_steps` | 런타임 lookahead (1 = single-step) | 1 |
| `use_learned_proposal` | 학습 proposal 사용 (False → Vanilla) | True |
| `blend_ratio` | μ_θ와 이전 해 혼합 비율 | 0.7 |
| `learn_covariance` | 대각 공분산 σ_θ 학습 | True |
| `online_training` | 온라인 자기지도 학습 | True |
| `train_interval` | 학습 주기 (스텝) | 10 |
| `entropy_weight` | 최대 엔트로피 정규화 τ | 0.01 |

---

## 10. 시뮬레이션 환경 (S1-S13)

13개 시뮬레이션 시나리오로 다양한 상황에서 MPPI 성능을 검증합니다.
정적/동적 장애물, 다중 로봇, 좁은 통로 등 실제 로봇 운용 상황을 모사합니다.

### 전체 시나리오 실행

```bash
# 13개 시나리오 순차 실행 + 요약 테이블
cd examples/simulation_environments
PYTHONPATH=../.. python run_all.py

# 특정 시나리오만 실행
PYTHONPATH=../.. python run_all.py --scenarios s1 s3 s5

PYTHONPATH=../.. python run_all.py --no-plot
```

### 시나리오 목록

| 시나리오 | 이름 | 설명 |
|---------|------|------|
| S1 | Static Obstacle Field | 정적 장애물 사이 경로 탐색 |
| S2 | Dynamic Bouncing | 바운싱 동적 장애물 회피 |
| S3 | Chasing Evader | 도주하는 대상 추적 |
| S4 | Multi-Robot Coordination | 다중 로봇 충돌 회피 협조 |
| S5 | Waypoint Navigation | 웨이포인트 순차 방문 |
| S6 | Drifting Disturbance | 바람/지형 외란 하의 주행 |
| S7 | Parking Precision | 정밀 주차 (목표 자세 수렴) |
| S8 | Racing MPCC | 경주용 궤적 추적 (MPCC 스타일) |
| S9 | Narrow Corridor | 좁은 통로 통과 |
| S10 | Mixed Challenge | 복합 환경 (정적+동적+외란) |
| S11 | C2U Obstacle Field | C2U-MPPI 전용 확률적 장애물 회피 |
| S12 | Warehouse | 창고 환경 (레벨별 난이도) |
| S13 | Racing Track | 레이싱 트랙 (3종 트랙 + 마찰 불일치) |

### 기대 결과

- 각 시나리오별 성공률, RMSE, 계산 시간 요약 테이블
- S7 (Parking): 최종 위치 오차 < 0.05m
- S9 (Corridor): 안전 제어 기법 필수 (Vanilla는 충돌 가능)
- S11 (C2U): 노이즈 수준별 유효 반경 변화 확인

---

## 11. GPU 가속

CUDA GPU를 활용하여 MPPI 샘플링 연산을 가속합니다.
K=8192 샘플에서 CPU 대비 최대 8.1x 속도 향상을 달성합니다.

### GPU 벤치마크 실행

```bash
# CPU vs GPU 비교 (K=256/1024/4096/8192)
PYTHONPATH=. python examples/comparison/gpu_benchmark_demo.py

# Figure-8 궤적
PYTHONPATH=. python examples/comparison/gpu_benchmark_demo.py \
    --trajectory figure8 --duration 10

PYTHONPATH=. python examples/comparison/gpu_benchmark_demo.py --no-plot
```

### GPU 가속 성능표

| K (샘플 수) | CPU 시간 | GPU 시간 | 가속 비율 |
|------------|---------|---------|----------|
| 256 | ~5ms | ~3ms | 1.7x |
| 1024 | ~18ms | ~5ms | 3.6x |
| 4096 | ~70ms | ~12ms | 5.8x |
| 8192 | ~140ms | ~17ms | 8.1x |

*RTX 5080 기준, 실제 성능은 하드웨어에 따라 다를 수 있습니다.

### 요구 사항

| 항목 | 요구 사항 |
|------|---------|
| GPU | CUDA 지원 NVIDIA GPU |
| 드라이버 | CUDA 11.x 이상 |
| PyTorch | >= 2.0.0 (CUDA 빌드) |
| 선택 | CuPy >= 11.0.0 |

### GPU 사용 확인

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 기대 결과

- GPU 가속 시 RMSE는 CPU와 동등 (수치 오차 범위 내)
- K=8192에서 GPU가 실시간 10Hz 제어 주기 충족 (< 100ms)
- 샘플 수 증가 시 GPU 가속 비율이 점진적으로 향상

---

## 12. cbfkit-inspired 안전 기법 실습

Toyota Research Institute의 CBF 툴박스 cbfkit(arXiv:2404.07158)에서
포팅한 5종 안전 기법을 실습합니다:
HOCBF(고차 CBF), Stochastic CBF(Itô 보정), Risk-Aware CBF(확률 보장),
Robust CBF(유계 외란), CLF-CBF-QP(네이티브 QP 컨트롤러).

> 이론은 [docs/SAFETY_THEORY.md 15~20장](SAFETY_THEORY.md#15-cbfkit-inspired-확장-개요) 참조.
> 아래 모든 코드 블록은 실제 실행/검증되었습니다 (`PYTHONPATH=.` 필요).

### 12.1 상대 차수 자동 검출 (detect_relative_degree)

위치 barrier h = ||p - p_obs||² - r² 의 상대 차수(relative degree)는
"h의 시간 미분에서 제어가 처음 나타나는 차수"입니다.
기구학 모델(속도 제어)은 1, 동역학 모델(가속도 제어)은 2입니다.

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.controllers.mppi import detect_relative_degree

# 기구학 모델: u = [v, ω] — 속도가 ḣ에 직접 등장 → rd = 1
kin_model = DifferentialDriveKinematic(v_max=1.5, omega_max=2.0)
print("kinematic  rd =", detect_relative_degree(kin_model))   # → 1

# 동역학 모델: u = [a, α] — 가속도는 ḣ에 안 나타남 → rd = 2
dyn_model = DifferentialDriveDynamic()
print("dynamic 5D rd =", detect_relative_degree(dyn_model))   # → 2
```

실행 결과:
```
kinematic  rd = 1
dynamic 5D rd = 2
```

rd = 2에서 1차 CBF 비용(ControlBarrierCost)은 제어가 한 스텝에
바꿀 수 없는 것을 페널티하므로 사실상 무력합니다 → HOCBF 필요.

### 12.2 HOCBF 비용 (rd=2 동역학 모델)

지수형 캐스케이드 ψ1 = ψ̇0 + λ1·ψ0, 제약 ψ̇1 + λ2·ψ1 ≥ 0 을
MPPI 궤적 비용으로 부과합니다.

```python
import numpy as np
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi import HOCBFCost
from mppi_controller.utils.trajectory import generate_reference_trajectory

# 원형 궤적 r=2.0 (v_ref = 1.0 m/s) + 경로 위 장애물 1개
R_CIRCLE, W_CIRCLE, DT = 2.0, 0.5, 0.05
obstacles = [(0.0, 2.0, 0.3)]  # 원 궤적 90° 지점 위

def circle_ref(t):
    th = W_CIRCLE * t
    return np.array([
        R_CIRCLE * np.cos(th), R_CIRCLE * np.sin(th), th + np.pi / 2,
        R_CIRCLE * W_CIRCLE, W_CIRCLE,          # 속도 확장 레퍼런스 [v, ω]
    ])

model = DifferentialDriveDynamic()               # 상태 5D, 제어 [a, α] → rd=2
params = MPPIParams(
    K=512, N=20, dt=DT, lambda_=1.0,
    sigma=np.array([1.0, 1.0]),
    Q=np.array([10.0, 10.0, 1.0, 1.0, 0.5]),
    R=np.array([0.1, 0.1]),
)
cost = CompositeMPPICost([
    StateTrackingCost(params.Q),
    TerminalCost(params.Qf),
    ControlEffortCost(params.R),
    HOCBFCost(                                    # ← 핵심: rd=2 지수형 캐스케이드
        obstacles, lambda1=2.0, lambda2=2.0, weight=1000.0,
        safety_margin=0.1, dt=DT, relative_degree=2,
    ),
])
controller = MPPIController(model, params, cost)

state = np.array([R_CIRCLE, 0.0, np.pi / 2, 0.0, 0.0])
min_clear = np.inf
for step in range(200):                           # 10초 폐루프
    ref = generate_reference_trajectory(circle_ref, step * DT, params.N, DT)
    control, info = controller.compute_control(state, ref)
    state = model.step(state, control, DT)
    d = np.hypot(state[0] - obstacles[0][0], state[1] - obstacles[0][1])
    min_clear = min(min_clear, d - obstacles[0][2])

print(f"min clearance = {min_clear:.3f} m (>0 이면 충돌 없음)")
print(f"ESS = {info['ess']:.1f} / {params.K}")
```

실행 결과:
```
min clearance = 0.294 m (>0 이면 충돌 없음)
ESS = 174.7 / 512
```

`relative_degree=1`로 바꾸면 기존 ControlBarrierCost와 동등한 1차
조건으로 축약됩니다 (벤치마크에서 rd=2 시나리오 클리어런스
0.282 m vs 1차 CBF 0.039 m).

### 12.3 HOCBF 해석적 사후 필터 (HOCBFFilter)

MPPI 출력을 closed-form 최소 노름 보정으로 필터링합니다.
`relative_degree=None`이면 12.1의 검출 함수로 자동 판별합니다.

```python
import numpy as np
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.controllers.mppi import HOCBFFilter

model = DifferentialDriveDynamic()
obstacles = [(1.0, 0.0, 0.3)]

# relative_degree=None → detect_relative_degree 로 자동 검출 (동역학 → 2)
filt = HOCBFFilter(model, obstacles, lambda1=2.0, lambda2=2.0,
                   safety_margin=0.1, relative_degree=None)
print("자동 검출 rd =", filt.relative_degree)     # → 2

# 장애물을 향해 v=1.5 m/s 로 돌진하는 상태에서 가속 명령을 필터링
state = np.array([0.3, 0.0, 0.0, 1.5, 0.0])      # 장애물까지 0.7 m
u_nominal = np.array([2.0, 0.0])                  # 최대 가속 (위험!)
u_safe, info = filt.filter_control(state, u_nominal)

print(f"u_nominal = {u_nominal}")
print(f"u_safe    = {np.round(u_safe, 3)}")       # 감속 방향으로 보정됨
print(f"filtered={info['filtered']}, "
      f"correction_norm={info['correction_norm']:.3f}, "
      f"min_constraint={info['min_constraint']:.3f}")
print("누적 통계:", filt.get_filter_statistics())
```

실행 결과:
```
자동 검출 rd = 2
u_nominal = [2. 0.]
u_safe    = [-1.693  0.   ]
filtered=True, correction_norm=3.693, min_constraint=-5.170
```

가속 +2.0 명령이 감속 -1.693으로 뒤집혔습니다. 단, 벤치마크에서
확인됐듯 사후 필터는 계획 전체를 재형성하지 못하므로 (노이즈 하
충돌 3.0±4.2 스텝) **HOCBFCost(계획 비용)와 함께** 최후 방어선으로
쓰는 것을 권장합니다.

### 12.4 Stochastic CBF + Risk-Aware CBF (프로세스 노이즈)

두 비용의 핵심 차이:
- **StochasticCBFCost**: Itô 보정 — 볼록 barrier에서는 조건이
  오히려 **완화**됨 (양수 항 추가). 보수성은 β 버퍼 담당.
- **RiskAwareCBFCost**: 시간 증가 마진 √(2t)·η·erfinv(1-2ρ)로
  P(min_t h < 0) ≤ ρ 보장 — 노이즈 안전의 실전 선택.

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import MPPIParams
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
)
from mppi_controller.controllers.mppi import StochasticCBFCost, RiskAwareCBFCost
from mppi_controller.utils.trajectory import generate_reference_trajectory

R_CIRCLE, W_CIRCLE, DT = 2.0, 0.5, 0.05
obstacles = [(0.0, 2.0, 0.3)]

def circle_ref(t):
    th = W_CIRCLE * t
    return np.array([R_CIRCLE * np.cos(th), R_CIRCLE * np.sin(th), th + np.pi / 2])

# 이산 per-step 노이즈 std → 연속시간 확산 σ (std_step = σ·√dt)
noise_std = np.array([0.05, 0.05, 0.02])
sigma_process = noise_std / np.sqrt(DT)

# ── 두 가지 확률적 CBF 비용 ──────────────────────────────────
sto_cost = StochasticCBFCost(                 # Itô 보정 (완화!) + β 버퍼
    obstacles, alpha=6.0, beta=0.5,
    sigma_process=sigma_process, weight=50.0, safety_margin=0.1, dt=DT,
)
print(f"Itô 보정항 Σσ_pos² = {sto_cost.get_ito_correction():.3f}  (양수 → 조건 완화)")

risk_cost = RiskAwareCBFCost(                 # P(min_t h < 0) ≤ ρ 마진
    obstacles, rho=0.1,
    sigma_process=sigma_process, weight=1000.0, safety_margin=0.1, dt=DT,
)
t_grid = np.array([0.25, 0.5, 1.0])           # 호라이즌 내 시간 (초)
print("margin(t) =", np.round(risk_cost.get_margin(t_grid), 3), "@ ρ=0.1")

# ── RiskAware 비용으로 노이즈 하 폐루프 실행 ─────────────────
model = DifferentialDriveKinematic(v_max=1.5, omega_max=2.0)
params = MPPIParams(K=512, N=20, dt=DT, lambda_=1.0,
                    sigma=np.array([0.5, 0.5]),
                    Q=np.array([10.0, 10.0, 1.0]), R=np.array([0.1, 0.1]))
cost = CompositeMPPICost([
    StateTrackingCost(params.Q), TerminalCost(params.Qf),
    ControlEffortCost(params.R), risk_cost,
])
controller = MPPIController(model, params, cost)

rng = np.random.default_rng(42)
state = np.array([R_CIRCLE, 0.0, np.pi / 2])
min_clear = np.inf
for step in range(200):
    ref = generate_reference_trajectory(circle_ref, step * DT, params.N, DT)
    control, info = controller.compute_control(state, ref)
    state = model.step(state, control, DT)
    state = state + rng.normal(size=3) * noise_std      # 프로세스 노이즈 주입
    d = np.hypot(state[0] - obstacles[0][0], state[1] - obstacles[0][1])
    min_clear = min(min_clear, d - obstacles[0][2])

print(f"노이즈 하 min clearance = {min_clear:.3f} m")
```

실행 결과:
```
Itô 보정항 Σσ_pos² = 0.100  (양수 → 조건 완화)
margin(t) = [0.162 0.229 0.324] @ ρ=0.1
노이즈 하 min clearance = 0.243 m
```

마진이 √t로 증가하는 것(0.162 → 0.324)과, 같은 노이즈에서
벤치마크의 일반 CBF 비용이 3/3 시드 충돌한 반면 RiskAware(ρ=0.1)는
충돌 0을 유지한 점에 주목하세요.

### 12.5 Robust CBF (유계 외란 worst-case 마진)

‖w‖ ≤ w_max 외란에 대해 CBF 조건을 dt·‖∇h·M‖·w_max 만큼 강화합니다.

```python
import numpy as np
from mppi_controller.controllers.mppi import RobustCBFCost

obstacles = [(0.0, 2.0, 0.3)]

# w_max = 1.0 m/s: 이산 노이즈 1σ(0.05 m)를 dt=0.05 s 로 나눈 등가 속도 외란
robust_cost = RobustCBFCost(
    obstacles, w_max=1.0, alpha=0.3, weight=1000.0,
    safety_margin=0.1, dt=0.05, norm="two",   # "sup": ‖w‖∞ 유계 외란
)

# 장애물 주위를 지나는 가짜 궤적 (K=2, N=10)으로 마진 확인
traj = np.zeros((2, 11, 3))
traj[0, :, 0] = np.linspace(-1.0, 1.0, 11)    # 샘플 0: 장애물에서 1 m 떨어져 통과
traj[0, :, 1] = 1.0
traj[1, :, 0] = np.linspace(-1.0, 1.0, 11)    # 샘플 1: 장애물 중심 관통 (위험)
traj[1, :, 1] = 2.0

margins = robust_cost.get_robust_margin(traj)  # (num_obs, K, N)
costs = robust_cost.compute_cost(traj, None, None)
print("마진 항 dt·‖∇h·M‖·w_max 평균:", np.round(margins.mean(axis=(0, 2)), 4))
print("Robust CBF 비용:", np.round(costs, 1), "→ 관통 샘플이 강하게 페널티")

# w_max=0 → vanilla ControlBarrierCost 와 정확히 동일 (축약 성질)
vanilla_equiv = RobustCBFCost(obstacles, w_max=0.0, alpha=0.3, weight=1000.0,
                              safety_margin=0.1, dt=0.05)
print("w_max=0 비용:", np.round(vanilla_equiv.compute_cost(traj, None, None), 1))
```

실행 결과:
```
마진 항 dt·‖∇h·M‖·w_max 평균: [0.115 0.05 ]
Robust CBF 비용: [  0. 888.] → 관통 샘플이 강하게 페널티
w_max=0 비용: [  0. 588.]
```

같은 관통 궤적에 대해 w_max=1.0이 vanilla(588)보다 큰 비용(888)을
부과합니다 — 외란 마진만큼 조건이 강화된 결과입니다.
벤치마크(노이즈, 3 시드)에서 Robust CBF는 충돌 0 + RMSE 0.705로
"충돌 없는 방법 중 최고 추적"이었습니다.

### 12.6 CLF-CBF-QP 컨트롤러 (샘플링 없는 베이스라인)

MPPI 없이 QP만으로 수렴(CLF, soft) + 안전(CBF, hard)을 푸는
독립 컨트롤러입니다. repo 표준 `compute_control` 인터페이스를 따르므로
기존 시뮬레이션 루프에 그대로 꽂을 수 있습니다.

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi import (
    CLFCBFQPParams,
    CLFCBFQPController,
    CBFOnlyQPController,
)
from mppi_controller.utils.trajectory import generate_reference_trajectory

R_CIRCLE, W_CIRCLE, DT = 2.0, 0.5, 0.05
obstacles = [(0.0, 2.0, 0.3)]

def circle_ref(t):
    th = W_CIRCLE * t
    return np.array([R_CIRCLE * np.cos(th), R_CIRCLE * np.sin(th), th + np.pi / 2])

model = DifferentialDriveKinematic(v_max=1.5, omega_max=2.0)
qp_params = CLFCBFQPParams(dt=DT, safety_margin=0.15)

# CLF-CBF-QP (수렴 soft + 안전 hard) vs CBF-QP (pure-pursuit 명목 + hard CBF)
for Ctrl in (CLFCBFQPController, CBFOnlyQPController):
    controller = Ctrl(model, qp_params, obstacles)
    state = np.array([R_CIRCLE, 0.0, np.pi / 2])
    min_clear, solve_ms, feas = np.inf, [], []
    for step in range(400):                        # 20초 폐루프
        ref = generate_reference_trajectory(circle_ref, step * DT, 20, DT)
        control, info = controller.compute_control(state, ref)
        solve_ms.append(info["solve_time"] * 1000)
        feas.append(info["qp_feasible"])
        state = model.step(state, control, DT)
        d = np.hypot(state[0] - obstacles[0][0], state[1] - obstacles[0][1])
        min_clear = min(min_clear, d - obstacles[0][2])
    print(f"{Ctrl.__name__:22s} min_clear={min_clear:.3f} m, "
          f"solve={np.mean(solve_ms):.2f} ms, feasible={100*np.mean(feas):.1f}%")
```

실행 결과:
```
CLFCBFQPController     min_clear=0.338 m, solve=0.10 ms, feasible=100.0%
CBFOnlyQPController    min_clear=0.312 m, solve=0.08 ms, feasible=100.0%
```

**~0.1 ms** — MPPI(1.8~3.0 ms)의 약 1/20 계산으로 항상 충돌 없음.
단 동역학 5D에서는 backstepping-lite CLF의 근사 때문에 추적이
열화됩니다 (벤치마크 RMSE 1.93~2.41 vs MPPI 0.69~0.85).

### 12.7 벤치마크 실행 (10-Way × 4 시나리오)

```bash
# A. 기구학 + 정적 장애물 (기준선)
PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario static_kin

# B. 동역학 5D 가속도 제어 (rd=2 — HOCBF 홈그라운드)
PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario dynamic_rd2

# C. 강한 프로세스 노이즈 (3 시드 mean±std)
PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario stochastic

# D. Risk 예산 스윕 (ρ ∈ {0.5, 0.2, 0.1, 0.05, 0.01})
PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario risk_sweep

# 전체 실행 / 빠른 확인
PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --all-scenarios
PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario static_kin --duration 5 --no-plot
```

결과: `plots/cbfkit_inspired_benchmark_{시나리오}.png`,
`results/cbfkit_inspired/{시나리오}.json`

**결과 해석 포인트**:

| 시나리오 | 봐야 할 것 |
|---------|-----------|
| static_kin | CBF-MPPI의 MinClear **-0.030 m** (3 충돌 스텝!) vs HOCBF-MPPI 0.223 m + 최고 RMSE 0.436 — soft 페널티는 무노이즈에서도 스칠 수 있음 |
| dynamic_rd2 | 1차 CBF 계열 0.039~0.199 m vs HOCBF(rd=2) **0.282 m** — 상대 차수 불일치의 대가 |
| stochastic | CBF-MPPI 3/3 시드 충돌, StochasticCBF도 1.7 충돌 스텝 (Itô 완화 실증) vs RiskAware/Robust **충돌 0** |
| risk_sweep | MinClear가 ρ에 단조: -0.068 → 0.588 m (ρ 0.5 → 0.01) — ρ는 해석 가능한 안전 다이얼 |
| 공통 | QP 컨트롤러 solve ~0.1 ms (MPPI 1.8~3.0 ms), 항상 충돌 0, 5D에서만 추적 열화 |

### 12.8 직접 해보기

1. **ρ 스윕 재현**: 12.4의 폐루프에서 `rho`를 {0.5, 0.2, 0.1, 0.05,
   0.01}로 바꿔가며 min clearance를 기록하고, 단조 증가를
   확인하세요. ρ=0.5에서 margin이 정확히 0이 되는 이유를
   `risk_cost.get_margin(1.0)`으로 검증해 보세요 (erfinv(0)=0).

2. **λ1/λ2 보수성 관찰**: 12.2에서 `lambda1=lambda2`를
   {0.5, 1.0, 2.0, 4.0}으로 바꾸면서 min clearance와 위치 RMSE를
   기록하세요. λ가 작을수록 보수적(클리어런스↑, RMSE↑)임을
   확인하고, λ·dt > 1로 만들면 어떤 일이 생기는지 관찰하세요.

3. **HOCBF vs 1차 CBF (자기 장애물 배치)**: 12.2의 장애물을
   2~3개로 늘리고 (예: 210°, 330° 지점 추가) `relative_degree`를
   1과 2로 바꿔 클리어런스를 비교하세요. 동역학 모델에서 rd=1이
   왜 뒤늦게 반응하는지 궤적 플롯으로 확인해 보세요.

4. **Itô 완화 실증**: 12.4의 폐루프에서 `risk_cost` 대신
   `sto_cost`(beta=0.5)를 넣고 같은 시드로 실행해 min clearance를
   비교하세요. 이어서 `beta`를 0 → 1.0 → 2.0으로 올리면
   RiskAware와의 격차가 어떻게 줄어드는지 관찰하세요.

5. **QP vs MPPI 하이브리드**: 12.6의 `CBFOnlyQPController` 대신,
   12.2의 MPPI 출력에 12.3의 `HOCBFFilter`를 씌운 2단 구조
   (계획 = HOCBFCost, 필터 = HOCBFFilter)를 만들어 단독 대비
   클리어런스/개입율(`get_filter_statistics()`)을 비교하세요.

---

## 부록: 자주 묻는 질문

### Q: Headless 서버에서 실행하려면?

모든 데모에 `--no-plot` 플래그를 추가하면 matplotlib 디스플레이 없이 실행됩니다.

```bash
PYTHONPATH=. python examples/comparison/safety_comparison_demo.py --no-plot
```

### Q: 특정 테스트만 실행하려면?

```bash
# 특정 파일
python -m pytest tests/test_base_mppi.py -v --override-ini="addopts="

# 특정 함수
python -m pytest tests/test_base_mppi.py::test_circle_tracking -v --override-ini="addopts="
```

### Q: 새로운 MPPI 변형을 추가하려면?

`MPPIController`를 상속하고 `_compute_weights()` 메서드를 오버라이드합니다.
상세 구조는 `mppi_controller/controllers/mppi/` 디렉터리의 기존 구현을 참조하세요.

### Q: 커스텀 로봇 모델을 추가하려면?

`RobotModel` 추상 베이스 클래스를 상속하고 `forward_dynamics()`와 `state_dim`, `control_dim`을 구현합니다.
`mppi_controller/models/` 디렉터리의 기존 모델을 참조하세요.
