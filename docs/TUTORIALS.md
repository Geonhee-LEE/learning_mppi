# MPPI 튜토리얼 가이드

이 문서는 learning_mppi 프로젝트의 전체 기능을 단계별로 안내합니다.
20종 MPPI 변형, 22종 안전 제어, 14종 학습 모델을 포괄하는 실습 가이드입니다.

---

## 목차

1. [환경 설정](#1-환경-설정)
2. [기본 MPPI 제어 (기구학)](#2-기본-mppi-제어-기구학)
3. [동역학 모델 제어](#3-동역학-모델-제어)
4. [MPPI 변형 20종 벤치마크](#4-mppi-변형-20종-벤치마크)
5. [안전 제어 (CBF / Shield / Adaptive)](#5-안전-제어-cbf--shield--adaptive)
6. [모델 학습 (NN / GP / Residual / Ensemble)](#6-모델-학습-nn--gp--residual--ensemble)
7. [메타 학습 및 온라인 적응](#7-메타-학습-및-온라인-적응-maml--lora--ekf--l1--alpaca)
8. [고급: LotF / BPTT / DiffSim / NN-Policy](#8-고급-lotf--bptt--diffsim--nn-policy)
9. [불확실성 기반 제어](#9-불확실성-기반-제어-uncertainty--conformal--c2u-mppi)
10. [시뮬레이션 환경 (S1-S13)](#10-시뮬레이션-환경-s1-s13)
11. [GPU 가속](#11-gpu-가속)

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
# 전체 테스트 (1267+개, ~26초)
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

## 4. MPPI 변형 20종 벤치마크

19가지 MPPI 변형 알고리즘을 동시에 비교하여 성능을 평가합니다.
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

- 20종 알고리즘의 RMSE, 계산 시간, ESS 비교 테이블 출력
- 궤적 비교 플롯 (각 변형의 추적 경로 오버레이)
- Vanilla 대비 각 변형의 상대 성능 비율

---

## 5. 안전 제어 (CBF / Shield / Adaptive)

22종 안전 제어 기법을 장애물 환경에서 비교합니다.
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
