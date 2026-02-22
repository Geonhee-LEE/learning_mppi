# MPPI ROS2 - TODO

프로젝트 개발 작업 목록입니다. Claude가 순차적으로 처리합니다.

---

## 🔴 High Priority (P0) - Phase 1: 기구학 모델 (M1)

### 기본 구조

- [x] #001 프로젝트 기본 구조 설정 ✓ 2026-02-07
  * Python 패키지 구조 생성 (mppi_controller/models, controllers, simulation)
  * pyproject.toml, requirements.txt 작성
  * 기본 디렉토리 생성 (tests, examples, configs, docs)

### 모델 추상화 계층

- [x] #002 추상 베이스 클래스 (base_model.py) ✓ 2026-02-07
  * RobotModel ABC 정의
  * 통일된 인터페이스: state_dim, control_dim, model_type
  * forward_dynamics(state, control) → state_dot
  * step(state, control, dt) - RK4 적분 기본 구현
  * get_control_bounds() → (lower, upper)
  * 벡터화 지원 (batch, nx) 형태

- [x] #003 Differential Drive 기구학 모델 ✓ 2026-02-07
  * models/kinematic/differential_drive_kinematic.py
  * 상태: [x, y, θ] (3차원)
  * 제어: [v, ω] (2차원) - 선속도, 각속도
  * forward_dynamics: dx/dt = v*cos(θ), dy/dt = v*sin(θ), dθ/dt = ω
  * model_type = "kinematic"

### MPPI 컨트롤러 인프라

- [x] #004 MPPIParams 데이터클래스 ✓ 2026-02-07
  * mppi_params.py - 파라미터 데이터클래스
  * N (horizon), dt, K (samples), lambda, sigma, Q, R, Qf
  * 기본값 설정 (N=30, dt=0.05, K=1024)

- [x] #005 BatchDynamicsWrapper (모든 모델 지원) ✓ 2026-02-07
  * dynamics_wrapper.py - 배치 동역학 래퍼
  * __init__(model: RobotModel, dt: float)
  * rollout(initial_state, controls) → trajectories (K, N+1, nx)
  * RK4 벡터화 (K개 샘플 병렬 전파)
  * **모델 타입 무관** - RobotModel 인터페이스만 준수하면 됨

- [x] #006 비용 함수 모듈 (기본) ✓ 2026-02-07
  * cost_functions.py - 비용 함수 클래스
  * StateTrackingCost (Q 가중치)
  * TerminalCost (Qf 가중치)
  * ControlEffortCost (R 가중치)
  * CompositeMPPICost (비용 합성)
  * ObstacleCost (장애물 회피) - M2 준비 완료
  * ControlRateCost (제어 변화율) - M2 준비 완료

- [x] #007 Gaussian 노이즈 샘플러 ✓ 2026-02-07
  * sampling.py - GaussianSampler 클래스
  * sample(U, K) → (K, N, nu) 노이즈
  * Σ = σ² I 대각 공분산
  * ColoredNoiseSampler (OU 프로세스) - M2 준비 완료
  * RectifiedGaussianSampler (pytorch_mppi 스타일) - Phase 4 준비 완료

- [x] #008 Vanilla MPPI 컨트롤러 구현 ✓ 2026-02-07
  * base_mppi.py - MPPIController 클래스
  * compute_control(state, ref) → (control, info)
  * info dict: sample_trajectories, sample_weights, best_trajectory, temperature, ess

### 시뮬레이션 인프라

- [x] #009 간이 시뮬레이터 (Simulator 클래스) ✓ 2026-02-07
  * simulation/simulator.py
  * __init__(model, controller, dt, process_noise_std)
  * reset(initial_state)
  * step(reference_trajectory) → step_info
  * run(reference_trajectory_fn, duration, realtime) → history
  * 메트릭 수집 (time, state, control, reference, solve_time, info)
  * 외란 주입 기능

- [x] #010 시각화 도구 (Visualizer 클래스) ✓ 2026-02-07
  * simulation/visualizer.py
  * plot_results(history, metrics) - 6패널 정적 플롯
    - XY 궤적, 위치 오차, 제어 입력, 각도 오차, 계산 시간, 메트릭 요약
  * animate_live(simulator, reference_trajectory_fn, duration) - 실시간 애니메이션
  * export_gif() - GIF 파일 생성 (Phase 4 예정)

- [x] #011 메트릭 계산 (metrics.py) ✓ 2026-02-07
  * simulation/metrics.py
  * compute_metrics(history) → dict
    - position_rmse, max_position_error
    - heading_rmse
    - control_rate (제어 변화율)
    - mean_solve_time, max_solve_time

### 테스트 및 검증

- [ ] #012 기구학 모델 유닛 테스트 (Phase 1 후속 작업)
  * tests/models/test_kinematic_models.py
  * forward_dynamics 검증
  * step() RK4 적분 검증
  * 벡터화 (batch) 지원 테스트

- [ ] #013 MPPI 유닛 테스트 (Phase 1 후속 작업)
  * tests/test_mppi.py - 기본 기능 테스트
  * tests/test_dynamics_wrapper.py - 동역학 래퍼 테스트
  * tests/test_cost_functions.py - 비용 함수 테스트
  * tests/test_sampling.py - 샘플링 테스트
  * tests/test_simulator.py - 시뮬레이터 테스트

- [x] #014 원형 궤적 추적 데모 ✓ 2026-02-07
  * examples/kinematic/mppi_differential_drive_kinematic_demo.py
  * 원형 경로 생성 유틸리티 (utils/trajectory.py)
  * **검증 결과**: Circle RMSE = 0.0060m ✓ (목표: < 0.2m)
  * **검증 결과**: Solve Time = 4.96ms ✓ (목표: < 100ms)
  * --trajectory {circle,figure8,sine} 선택 구현
  * --live 실시간 시뮬레이션 모드 구현
  * 6패널 정적 플롯 + ASCII 메트릭 요약 구현

---

## 🟠 Medium Priority (P1) - Phase 2: 동역학 모델 (M2)

### 동역학 모델 구현

- [x] #101 Differential Drive 동역학 모델 ✓ 2026-02-07
  * models/dynamic/differential_drive_dynamic.py
  * 상태: [x, y, θ, v, ω] (5차원)
  * 제어: [a, α] (2차원) - 선가속도, 각가속도
  * forward_dynamics: 마찰/관성 고려
    - dx/dt = v*cos(θ), dy/dt = v*sin(θ), dθ/dt = ω
    - dv/dt = a - c_v*v, dω/dt = α - c_ω*ω
  * model_type = "dynamic"
  * 파라미터: mass, inertia, c_v, c_omega
  * compute_energy() 메서드 추가 (검증용)

- [ ] #102 동역학 모델 유닛 테스트 (Phase 2 후속 작업)
  * tests/models/test_dynamic_models.py
  * forward_dynamics 검증
  * 에너지 보존 검증 (c_v=0, c_omega=0 설정)
  * 벡터화 (batch) 지원 테스트

- [x] #103 동역학 모델 원형 궤적 데모 ✓ 2026-02-07
  * examples/dynamic/mppi_differential_drive_dynamic_demo.py
  * **검증 결과**: Circle RMSE = 0.1600m ✓ (목표: < 0.5m)
  * **검증 결과**: Solve Time = 5.78ms ✓ (목표: < 100ms)
  * --trajectory {circle,figure8,sine} 지원

### 모델 비교

- [x] #104 기구학 vs 동역학 비교 데모 ✓ 2026-02-07
  * examples/comparison/kinematic_vs_dynamic_demo.py
  * 동일한 레퍼런스 궤적으로 시뮬레이션
  * ASCII 메트릭 비교 테이블 구현
  * 6패널 비교 플롯 (궤적, 오차, 제어, 계산 시간)
  * **비교 결과** (Circle 20s):
    - Kinematic: RMSE=0.1841m, Time=5.05ms
    - Dynamic: RMSE=0.0961m, Time=5.86ms

### M2 고급 기능

- [x] #105 ControlRateCost 비용 함수 ✓ 2026-02-07
  * cost_functions.py에 ControlRateCost 추가
  * Δu 제어 변화율 비용
  * R_rate 가중치 파라미터

- [x] #106 Adaptive Temperature ✓ 2026-02-07
  * adaptive_temperature.py - AdaptiveTemperature 클래스
  * ESS (Effective Sample Size) 기반 λ 자동 튜닝
  * 목표 ESS 비율 유지

- [x] #107 Colored Noise 샘플링 ✓ 2026-02-07
  * sampling.py에 ColoredNoiseSampler 추가
  * OU 프로세스 기반 시간 상관 노이즈
  * theta (reversion rate) 파라미터

- [x] #108 Obstacle 비용 함수 ✓ 2026-02-07
  * cost_functions.py에 ObstacleCost 추가
  * 원형 장애물 회피
  * safety_margin 파라미터

- [ ] #109 Vanilla vs M2 비교 데모
  * examples/mppi_vanilla_vs_m2_demo.py
  * ControlRate, AdaptiveTemp, ColoredNoise 효과 비교
  * --live 실시간 비교

### Tube-MPPI (외란 강건성)

- [x] #110 AncillaryController 구현 ✓ 2026-02-07
  * controllers/mppi/tube_mppi.py에 통합
  * body frame 피드백
  * world → body 오차 변환
  * K_fb 피드백 게인

- [x] #111 Tube-MPPI 컨트롤러 ✓ 2026-02-07
  * tube_mppi.py - TubeMPPIController
  * MPPIController 상속
  * 명목 상태 전파 + 피드백 보정
  * tube_enabled 플래그 (False → Vanilla 동작)
  * 커밋: (M3 SOTA 변형)

- [x] #112 TubeAwareCost 비용 함수 ✓ 2026-02-07
  * cost_functions.py에 TubeAwareCost 추가
  * 장애물 safety_margin + tube_margin 확장

- [x] #113 Vanilla vs Tube 비교 데모 ✓ 2026-02-07
  * examples/comparison/vanilla_vs_tube_demo.py
  * --noise 외란 강도 조절
  * --live 실시간 비교
  * Plot: plots/vanilla_vs_tube_comparison.png

---

## 🟠 Medium Priority (P1) - Phase 3: 학습 모델 (M3)

### 학습 모델 구현

- [x] #201 ResidualDynamics 모델 ✓ 2026-02-07
  * models/learned/residual_dynamics.py
  * f_total(x, u) = f_physics(x, u) + f_learned(x, u)
  * __init__(base_model, residual_fn, uncertainty_fn)
  * model_type = "learned"
  * get_uncertainty(state, control) - GP 불확실성 (선택적)
  * get_residual_contribution() - 기여도 분석
  * 통계 추적 (mean, std, num_calls)

- [x] #202 NeuralDynamics 완전 구현 ✓ 2026-02-07
  * models/learned/neural_dynamics.py
  * PyTorch 기반 신경망 동역학 완전 구현
  * 학습 파이프라인 연동 완료
  * 학습된 모델 로드 및 추론 지원

- [x] #202-1 데이터 수집 파이프라인 ✓ 2026-02-07
  * learning/data_collector.py
  * DataCollector 클래스 (에피소드 기반 데이터 수집)
  * DynamicsDataset 클래스 (train/val split, 정규화)
  * 데이터 저장/로드 (pickle)

- [x] #202-2 Neural Network 학습 파이프라인 ✓ 2026-02-07
  * learning/neural_network_trainer.py
  * DynamicsMLPModel (PyTorch MLP)
  * NeuralNetworkTrainer (학습/평가/저장/로드)
  * Early stopping, learning rate scheduling
  * 학습 히스토리 플롯

- [x] #202-3 Neural Dynamics 학습 데모 ✓ 2026-02-07
  * examples/learned/neural_dynamics_learning_demo.py
  * 전체 파이프라인: 데이터 수집 → 학습 → 평가
  * Physics vs Neural vs Residual 3-way 비교
  * 9패널 비교 플롯

- [x] #203 GaussianProcessDynamics 완전 구현 ✓ 2026-02-07
  * models/learned/gaussian_process_dynamics.py
  * GPyTorch 기반 GP 학습 완전 구현
  * 불확실성 정량화 (mean + std)
  * predict_with_uncertainty() 구현
  * Exact GP 및 Sparse GP 지원

- [x] #203-1 Gaussian Process 학습 파이프라인 ✓ 2026-02-07
  * learning/gaussian_process_trainer.py
  * ExactGPModel (소규모 데이터)
  * SparseGPModel (대규모 데이터, 유도점)
  * Multi-output GP (각 출력 차원 독립 학습)
  * RBF/Matern 커널, ARD 지원
  * 학습 최적화 및 모델 저장/로드

- [x] #203-2 GP vs Neural Network 비교 데모 ✓ 2026-02-07
  * examples/learned/gp_vs_neural_comparison_demo.py
  * 데이터 효율성 비교 (data_fraction 파라미터)
  * 불확실성 보정 평가 (1σ, 2σ calibration)
  * 계산 시간 비교
  * MPPI 제어 성능 비교
  * 12패널 종합 비교 플롯

### 학습 모델 테스트

- [x] #204 학습 모델 유닛 테스트 ✓ 2026-02-07
  * tests/test_residual_dynamics.py (5개 테스트 전부 통과)
  * ResidualDynamics 동등성 검증 (residual=None)
  * Constant residual 효과 검증
  * 배치 처리 검증
  * 기여도 분석 검증
  * 통계 추적 검증

- [x] #205 Residual 동역학 데모 ✓ 2026-02-07
  * examples/learned/mppi_residual_dynamics_demo.py
  * 더미 residual_fn 타입별 비교 (constant/state/control/none)
  * Residual 기여도 분석 출력
  * 통계 추적 출력

### 모델 비교

- [x] #206 Physics vs Learned 비교 데모 ✓ 2026-02-07
  * examples/comparison/physics_vs_learned_demo.py
  * 기구학, 동역학, Residual 동역학 3-way 비교
  * ASCII 메트릭 비교 테이블
  * 6패널 비교 플롯 생성

- [ ] #207 모델 타입별 벤치마크 도구 (Phase 3 후속 작업)
  * examples/comparison/model_type_benchmark.py
  * Kinematic, Dynamic, Learned 동시 비교
  * --trajectory {circle,figure8,sine} 선택
  * ASCII 요약 테이블 + 차트

---

## 🟢 Low Priority (P2) - Phase 4: pytorch_mppi 개선

### 함수 기반 인터페이스

- [ ] #301 FunctionDynamicsWrapper (함수 기반)
  * dynamics_wrapper.py에 FunctionDynamicsWrapper 추가
  * __init__(dynamics_fn, dt)
  * dynamics_fn: (K, nx), (K, nu) → (K, nx)
  * pytorch_mppi 스타일 인터페이스

- [ ] #302 MPPIController 하이브리드 인터페이스
  * base_mppi.py 수정
  * __init__(model=None, dynamics_fn=None, ...)
  * 클래스 방식 (model) 또는 함수 방식 (dynamics_fn) 선택

- [ ] #303 정류 가우시안 샘플링
  * sampling.py에 use_rectified 파라미터 추가
  * 제약 위반 샘플 재샘플링 (pytorch_mppi 스타일)
  * 기존 클리핑 대비 성능 비교

- [ ] #304 함수 기반 인터페이스 테스트
  * tests/test_function_dynamics.py
  * 더미 신경망 함수 테스트
  * 클래스 방식과 동등성 검증

### 분석 도구

- [ ] #305 오프라인 분석 도구 (TrajectoryAnalyzer)
  * simulation/trajectory_analyzer.py
  * 궤적 히스토리 분석
  * 주파수 분석, 제어 스펙트럼 등

- [x] #306 GIF export 기능 ✓ 2026-02-07
  * visualizer.py에 export_gif() 추가
  * EnvVisualizer에도 export_gif() 구현

---

## 🟢 Low Priority (P2) - MPPI 변형 (M3 SOTA) ✅ 완료!

### M3 SOTA 변형 (2026-02-07 완료)

- [x] #401 Log-MPPI 컨트롤러 ✓
  * log_mppi.py - LogMPPIController
  * log-space softmax 가중치 (log-sum-exp trick)
  * 수치 안정성: NaN/Inf 방지
  * 커밋: `cd736f3`

- [x] #402 Tsallis-MPPI 컨트롤러 ✓
  * tsallis_mppi.py - TsallisMPPIController
  * q-exponential 가중치
  * utils.py에 q_exponential, q_logarithm 추가
  * q=1.0 → Vanilla 동등성
  * 커밋: `d1790d6`

- [x] #404 Risk-Aware MPPI 컨트롤러 ✓
  * risk_aware_mppi.py - RiskAwareMPPIController
  * CVaR 기반 샘플 선택
  * α<1.0 → 보수적 제어
  * 커밋: `7a01534`

- [x] #406 Stein Variational MPPI (SVMPC) ✓ + SPSA 최적화
  * stein_variational_mppi.py - SteinVariationalMPPIController
  * SVGD 기반 샘플 다양성
  * utils/stein_variational.py: RBF 커널, median bandwidth, efficient SVGD
  * **성능**: RMSE 0.009m, **113ms** (SPSA 최적화 후, 기존 1515ms에서 13x 개선)
  * 커밋: `4945838`

### M3.5 확장 변형 (2026-02-07 완료)

- [x] #408 Smooth MPPI 컨트롤러 ✓
  * smooth_mppi.py - SmoothMPPIController
  * Δu input-lifting 구조
  * Jerk cost (ΔΔu 페널티)
  * **성능**: Control Rate 0.0000 (완벽한 부드러움)
  * 모델별 비교: smooth_mppi_models_comparison.py
  * 커밋: `399cff6`

- [x] #410 Spline-MPPI 컨트롤러 ✓
  * spline_mppi.py - SplineMPPIController
  * B-spline 보간 (P knots → N controls)
  * **성능**: 메모리 73.3% 감소, 41ms
  * 모델별 비교: spline_mppi_models_comparison.py
  * 커밋: `9c1c7ed`

- [x] #412 SVG-MPPI 컨트롤러 ✓
  * svg_mppi.py - SVGMPPIController
  * Guide particle SVGD (G << K)
  * **성능**: RMSE 0.007m, 273ms (SVMPC 대비 6.5배 빠름)
  * **효율**: SVGD 복잡도 99.9% 감소 (O(K²) → O(G²))
  * 모델별 비교: svg_mppi_models_comparison.py
  * 커밋: `bedfec0`

### Tube-MPPI (M2 고급 기능, 2026-02-07 완료)

- [x] #110 AncillaryController 구현 ✓
  * ancillary_controller.py - body frame 피드백
  * world → body 오차 변환
  * K_fb 피드백 게인
  * 커밋: `f9052de`

- [x] #111 Tube-MPPI 컨트롤러 ✓
  * tube_mppi.py - TubeMPPIController
  * 명목 상태 전파 + 피드백 보정
  * tube_enabled=False → Vanilla 동작
  * **성능**: RMSE 0.010m, 외란 강건성
  * 커밋: `f9052de`

- [x] #106 Adaptive Temperature ✓
  * adaptive_temperature.py - ESS 기반 λ 자동 튜닝
  * 목표 ESS 비율 유지
  * 커밋: `f9052de`

### MPPI 전체 벤치마크 ✅

- [x] #414 MPPI 전체 변형 벤치마크 도구 ✓
  * examples/mppi_all_variants_benchmark.py
  * 9종 변형 동시 비교
  * 9패널 종합 시각화 (XY 궤적, RMSE, Solve Time, 레이더 차트 등)
  * **결과**:
    - 최고 정확도: SVG-MPPI (0.0054m)
    - 최고 속도: Vanilla/Tube/Log (~5ms)
    - 메모리 효율: Spline-MPPI (-73%)
  * 커밋: (예정)

---

## 📚 Documentation (P2)

- [ ] #501 MODEL_TYPES 문서 작성
  * docs/mppi/MODEL_TYPES.md
  * 3가지 모델 타입 설명 (Kinematic/Dynamic/Learned)
  * 인터페이스 사용법 및 예제

- [ ] #502 PRD 문서 업데이트
  * docs/mppi/PRD.md 업데이트
  * 모델 분류 체계 반영
  * 아키텍처 다이어그램 추가

- [ ] #503 MPPI 기술 가이드
  * docs/mppi/MPPI_GUIDE.md
  * 알고리즘 상세 설명
  * 논문 참조 및 수식

- [ ] #504 API 문서 자동 생성
  * Sphinx 설정
  * docstring 작성 규칙
  * 자동 빌드 스크립트

- [ ] #505 튜토리얼 작성
  * docs/tutorials/getting_started.md
  * docs/tutorials/custom_model.md - 커스텀 모델 작성법
  * docs/tutorials/custom_cost_function.md
  * docs/tutorials/tuning_guide.md

- [ ] #506 README 작성
  * 프로젝트 소개
  * Quick Start
  * 예제 실행 방법
  * 모델 타입 선택 가이드

---

## 🚀 ROS2 Integration (P1)

- [x] #601 ROS2 패키지 구조 ✓ 2026-02-07
  * package.xml, setup.py 작성
  * colcon 빌드 설정
  * launch 파일 (mppi_sim.launch.py)
  * RVIZ 설정 파일
  * 파라미터 YAML 파일 (configs/mppi_controller.yaml, configs/trajectory.yaml)

- [x] #602 ROS2 기본 노드 구현 ✓ 2026-02-07
  * ros2/mppi_controller_node.py - MPPI ROS2 wrapper
  * ros2/simple_robot_simulator.py - 시뮬레이션 로봇
  * ros2/trajectory_publisher.py - 레퍼런스 경로 생성
  * geometry_msgs/Twist 퍼블리시
  * nav_msgs/Odometry 서브스크라이브
  * nav_msgs/Path 레퍼런스 서브스크라이브
  * 모든 9가지 MPPI 변형 지원 (파라미터로 선택)
  * kinematic/dynamic 모델 지원

- [x] #603 RVIZ 시각화 마커 ✓ 2026-02-07
  * ros2/mppi_visualizer_node.py
  * 샘플 궤적 (MarkerArray, 가중치 기반 투명도)
  * 가중 평균 궤적 (Path, 시안)
  * 레퍼런스 경로 시각화
  * 파라미터 기반 시각화 제어

- [ ] #604 nav2 Controller 플러그인 (Python prototype)
  * nav2 호환 인터페이스
  * ComputeVelocityCommands 구현
  * 파라미터 서버 연동

- [ ] #605 동적 장애물 회피
  * sensor_msgs/LaserScan 처리
  * 실시간 장애물 맵 업데이트
  * ObstacleCost 동적 연동

- [ ] #606 실시간 경로 재계획
  * RealtimeReplanner 클래스
  * 충돌 위험 감지
  * 웨이포인트 재생성

- [ ] #607 ROS2 통합 테스트
  * launch 기반 통합 테스트
  * Gazebo 시뮬레이션 연동
  * 실제 로봇 테스트

---

## ⚡ Performance Optimization (P1)

- [x] #701 GPU 가속 (PyTorch CUDA) ✓ 2026-02-08
  * gpu/ 패키지: torch_dynamics, torch_costs, torch_sampling
  * base_mppi.py device 분기 (CPU 코드 무수정)
  * RTX 5080: K=4096→4.4x, K=8192→8.1x speedup
  * 95개 테스트 전부 통과

- [ ] #702 SVMPC GPU 가속
  * pairwise kernel CUDA 가속
  * O(K²D) 연산 최적화

- [ ] #703 C++ MPPI 코어 변환
  * Python → C++ 포팅
  * Eigen 기반 배치 처리
  * pybind11 바인딩

- [ ] #704 C++ nav2 Controller 플러그인
  * nav2_core::Controller 상속
  * 실시간 성능 검증 (< 100ms)
  * 파라미터 YAML 설정

- [ ] #705 성능 프로파일링
  * cProfile 분석
  * 병목 지점 식별
  * 최적화 적용

---

## 🛡️ Safety-Critical Control 고도화 (P0) — safe_control 기반

> Ref: https://github.com/tkkim-robot/safe_control (Univ. of Michigan, DASC Lab)

### Phase S1: 동적 장애물 CBF 고도화 (~250줄) ✅ 완료

- [x] #710 C3BF (Collision Cone CBF) 비용 함수 ✓ 2026-02-08
  * controllers/mppi/c3bf_cost.py — CollisionConeCBFCost
  * 상대 속도 방향 고려 barrier: h = <p_rel, v_rel> + ||p_rel||·||v_rel||·cos(φ)
  * cos(φ) = sqrt(||p_rel||² - R²) / ||p_rel||
  * obstacle_tracker.py의 (vx, vy) 속도 직접 활용

- [x] #711 Optimal-Decay CBF 안전 필터 ✓ 2026-02-08
  * controllers/mppi/optimal_decay_cbf_filter.py — OptimalDecayCBFSafetyFilter
  * ω (decay rate) 추가 최적화 변수로 CBF 실현가능성 보장
  * min ||u - u_ref||² + p_sb·(ω - 1)² s.t. ḣ + α·ω·h ≥ 0

- [x] #712 DPCBF (Dynamic Parabolic CBF) 비용 함수 ✓ 2026-02-08
  * controllers/mppi/dpcbf_cost.py — DynamicParabolicCBFCost
  * Line-of-Sight 좌표 변환 + Gaussian-shaped 안전 경계
  * 접근 속도에 비례하여 안전 영역 자동 확대

- [x] #713 C3BF + DPCBF + Optimal-Decay 테스트 ✓ 2026-02-08
  * tests/test_safety_advanced.py — 20개 테스트
  * C3BF 6개 + DPCBF 7개 + Optimal-Decay 7개

### Phase S2: 안전 보장 강화 (~400줄) ✅ 완료

- [x] #720 Gatekeeper Safety Shielding ✓ 2026-02-08
  * controllers/mppi/gatekeeper.py — Gatekeeper
  * 백업 궤적 안전 검증 → 무한 시간 안전 보장
  * gate_open/closed 상태로 MPPI/백업 제어 선택

- [x] #721 Backup Controller (정지/회전) ✓ 2026-02-08
  * controllers/mppi/backup_controller.py
  * BrakeBackupController: 즉시 정지
  * TurnAndBrakeBackupController: 장애물 반대 방향 회전 후 정지

- [x] #722 Superellipsoid 장애물 지원 ✓ 2026-02-08
  * controllers/mppi/superellipsoid_cost.py
  * SuperellipsoidObstacle: (x'/a)^n + (y'/b)^n - 1 (회전 지원)
  * SuperellipsoidCost: discrete-time CBF 비용
  * tests/test_gatekeeper_superellipsoid.py — 19개 테스트

- [x] #723 Safety 비교 평가 데모 ✓ 2026-02-08
  * examples/comparison/safety_comparison_demo.py
  * CBF vs C3BF vs DPCBF vs Optimal-Decay vs Gatekeeper 5-way 비교
  * 3가지 시나리오: static, crossing, narrow

### Phase S3: 고급 확장 (~300줄) ✅ 완료

- [x] #730 Backup CBF (Sensitivity Propagation) ✓ 2026-02-08
  * controllers/mppi/backup_cbf_filter.py — BackupCBFSafetyFilter
  * 백업 궤적 rollout + 민감도 행렬 전파
  * 현재 제어가 백업 궤적 전체 안전에 미치는 영향 평가
  * multi-constraint QP

- [x] #731 Multi-robot CBF ✓ 2026-02-08
  * controllers/mppi/multi_robot_cbf.py
  * MultiRobotCBFCost (Layer A) + MultiRobotCBFFilter (Layer B)
  * MultiRobotCoordinator + RobotAgent
  * 다중 MPPI 에이전트 협조 제어

- [x] #732 MPCC Cost Function ✓ 2026-02-08
  * controllers/mppi/mpcc_cost.py
  * PathParameterization (arc-length) + MPCCCost
  * Contouring/Lag 오차 분리, 경로 추종 성능 향상

---

## 🧪 Additional Robot Models (P2)

- [x] #801 Swerve Drive 모델 ✓ 2026-02-08
  * models/kinematic/swerve_drive_kinematic.py — 3D state [x,y,θ], 3D ctrl [vx,vy,ω]
  * models/dynamic/swerve_drive_dynamic.py — 6D state [x,y,θ,vx,vy,ω], 3D ctrl [ax,ay,α]
  * GPU: TorchSwerveDriveKinematic, TorchSwerveDriveDynamic

- [ ] #802 Non-coaxial Swerve 모델
  * models/kinematic/non_coaxial_swerve_kinematic.py
  * models/dynamic/non_coaxial_swerve_dynamic.py
  * 비동축 스워브

- [x] #803 Ackermann 조향 모델 ✓ 2026-02-08
  * models/kinematic/ackermann_kinematic.py — 4D state [x,y,θ,δ], 2D ctrl [v,φ]
  * models/dynamic/ackermann_dynamic.py — 5D state [x,y,θ,v,δ], 2D ctrl [a,φ]
  * GPU: TorchAckermannKinematic, TorchAckermannDynamic

- [ ] #804 Omnidirectional 로봇 모델
  * models/kinematic/omnidirectional_kinematic.py
  * models/dynamic/omnidirectional_dynamic.py
  * Mecanum/Omni wheel

---

## 🔬 MAML 성능 개선 (P0) — ✅ 해결 완료

> **해결**: Residual meta-training + 5D MAML + 외란 시뮬레이션으로 MAML의 이점 입증
> **결과**: MAML-5D RMSE 0.055m < Dynamic 0.056m < Kinematic 0.094m (noise=0.7)

### 해결된 근본 원인

```
┌──────────────────────────────────────────────────────────┐
│              MAML 성능 개선 결과 (3가지 해결)              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. ✅ 차원 불일치 → 5D Residual MAML (#910)             │
│     - MAML-5D: 5D state + 2D control → 5D residual     │
│     - DynamicKinematicAdapter base + MAML 잔차 보정     │
│     - 속도 상태 직접 관측 → 관성/마찰 정확 보정         │
│                                                          │
│  2. ✅ Warmup 최소화 (#911)                              │
│     - warmup_steps 40 → 10, adapt_interval 80 → 20     │
│     - buffer_size 200 → 50, error_threshold 0.15       │
│     - temporal_decay=0.95 (최근 데이터 강조)             │
│                                                          │
│  3. ✅ 메타 학습 대안 (#912)                             │
│     - Reptile 트레이너 구현 (FOMAML 대안)               │
│     - Residual meta-training: 메타 학습과 온라인 적응    │
│       간 분포 불일치 해결 (핵심 수정)                     │
│                                                          │
│  4. ✅ 온라인 적응 고도화 (#913)                          │
│     - temporal_decay=0.95 (exponential weighting)       │
│     - error_threshold 기반 적응적 재적응                 │
│     - sample_weights 지원 추가                           │
│                                                          │
│  5. ✅ 외란 시뮬레이션 환경 (#914 대체)                   │
│     - DisturbanceProfile: Wind/Terrain/Sine/Combined    │
│     - --noise 0.0~1.0 CLI 인자 (외란 강도)              │
│     - 시간에 따라 변하는 외란 → 고정 모델 실패, MAML 적응│
│                                                          │
│  결과: MAML-5D(0.055m) < Dynamic(0.056m) < Kin(0.094m)  │
│        MAML이 고정 5D 모델도 이김! (noise=0.7)           │
└──────────────────────────────────────────────────────────┘
```

### 완료된 항목

- [x] #910 5D Residual MAML 아키텍처 ✓ 2026-02-21
  * MAML-5D: 5D state + 2D control → 5D residual
  * DynamicKinematicAdapter(c_v=0.1) base + MAML 잔차 보정
  * Residual meta-training: 메타 학습 시 잔차 타겟 사용 (분포 일치)
  * 결과: noise=0.7에서 RMSE 0.055m (Dynamic 0.056m 돌파)

- [x] #911 Warmup 최소화 전략 ✓ 2026-02-21
  * warmup_steps 10, adapt_interval 20, buffer_size 50
  * error_threshold=0.15 기반 적응적 재적응
  * temporal_decay=0.95로 최근 데이터 강조

- [x] #912 대안 메타 학습 (Reptile) ✓ 2026-02-21
  * ReptileTrainer: MAMLTrainer 상속, epsilon interpolation
  * `--meta-algo {fomaml,reptile}` CLI 지원
  * Residual meta-training (핵심 수정): 메타 학습 데이터를 잔차 타겟으로 변환

- [x] #913 온라인 적응 고도화 ✓ 2026-02-21
  * adapt()에 sample_weights + temporal_decay 지원 추가
  * error_threshold 기반 적응적 재적응 (오차 급증 시 즉시)
  * 4-seed 검증: 모든 시드에서 MAML-5D가 Dynamic 이상

- [x] #914 외란 시뮬레이션 환경 ✓ 2026-02-21
  * DisturbanceProfile ABC + 4개 구현 (Wind/Terrain/Sine/Combined)
  * DynamicWorld에 disturbance 주입 (get_force + get_param_delta)
  * `--noise FLOAT --disturbance {none,wind,terrain,sine,combined}` CLI

### 현재 벤치마크 (circle, 20s, --world dynamic, noise=0.7)

| 모델 | RMSE (m) | 비고 |
|------|----------|------|
| Oracle (5D, 정확) | 0.037 | 이론적 상한 (외란에도 취약) |
| **MAML-5D (5D, Residual)** | **0.055** | **온라인 적응 → 외란 흡수** |
| Dynamic (5D, 파라미터 틀림) | 0.056 | 고정 파라미터 |
| Kinematic (3D) | 0.094 | 속도 상태 없음 |
| MAML-3D (3D, Residual) | 0.096 | 3D 한계 |
| Residual (3D, Hybrid) | 0.244 | 오프라인 학습 |
| Neural (3D, E2E) | 0.393 | 오프라인 학습 |

---

## 🐛 Bug Fixes (P2)

- [ ] #901 각도 정규화 엣지 케이스
  * ±π 경계 처리
  * 각도 차이 계산 안정화

- [ ] #902 고속 주행 오버슈트 개선
  * 제어 게인 튜닝
  * 예측 호라이즌 조정

- [ ] #903 수치 안정성 검증
  * NaN/Inf 체크 로직
  * log-space 연산 안정화

---

## 🐳 DevOps (P2)

- [ ] #951 Docker 컨테이너화
  * Dockerfile 작성
  * docker-compose.yml
  * 재현 가능한 환경

- [ ] #952 CI/CD 파이프라인
  * GitHub Actions
  * 자동 테스트
  * 자동 배포

- [ ] #953 Claude Issue Watcher 설치
  * .claude/scripts/issue-watcher.sh
  * systemd service 설정
  * 자동 PR 생성

- [ ] #954 TODO Worker 스크립트
  * .claude/scripts/todo-worker.sh
  * claude-todo-worker, claude-todo-task, claude-todo-all
  * ~/.local/bin/ 설치

---

## ✅ Completed

### 2026-02-07

#### Phase 1 (M1) - 기구학 모델 및 Vanilla MPPI ✓

- [x] #000 프로젝트 저장소 초기화
  * Git 저장소 생성
  * .claude 디렉토리 설정
  * CLAUDE.md, TODO.md 작성

- [x] #001 프로젝트 기본 구조 설정
  * Python 패키지 구조 생성
  * pyproject.toml, requirements.txt 작성
  * 30개 파일 생성, 4273줄 코드

- [x] #002-#011 Phase 1 핵심 구현
  * 추상 베이스 클래스 (RobotModel)
  * Differential Drive 기구학 모델
  * MPPI 컨트롤러 전체 인프라
  * 시뮬레이션 도구 (Simulator, Visualizer, Metrics)

- [x] #014 원형 궤적 데모 및 검증
  * Position RMSE: 0.0060m ✓ (목표: < 0.2m, **33배 우수**)
  * Solve Time: 4.96ms ✓ (목표: < 100ms, **20배 빠름**)
  * 커밋: `ede08f8` - feat: Phase 1 (M1) 완료

#### Phase 2 (M2) - 동역학 모델 ✓

- [x] #101 Differential Drive 동역학 모델 구현
  * 마찰/관성 고려 동역학 모델
  * 상태 5차원, 제어 2차원 (가속도)
  * RobotModel 인터페이스 완벽 호환

- [x] #103 동역학 모델 데모 및 검증
  * Position RMSE: 0.1600m ✓ (목표: < 0.5m)
  * Solve Time: 5.78ms ✓ (목표: < 100ms)

- [x] #104 기구학 vs 동역학 비교 데모
  * 6패널 비교 플롯 생성
  * 동역학 모델이 위치 추종 성능 우수 (RMSE 0.0961m vs 0.1841m)
  * 커밋: `004139d` - feat: Phase 2 (M2) 완료

#### Phase 3 (M3) - 학습 모델 (Residual Dynamics) ✓

- [x] #201-#203 학습 모델 구현
  * ResidualDynamics (Physics + Learned 하이브리드)
  * NeuralDynamics 스켈레톤 (PyTorch 준비)
  * GaussianProcessDynamics 스켈레톤 (GPytorch 준비)

- [x] #204 학습 모델 유닛 테스트
  * 5개 테스트 전부 통과 ✓
  * Residual=None 동등성, 효과, 배치, 기여도, 통계

- [x] #205-#206 데모 및 비교
  * Residual 동역학 데모 (4가지 residual 타입)
  * Physics vs Learned 3-way 비교 (Kinematic/Residual/Dynamic)
  * 커밋: `f34753e` - feat: Phase 3 (M3) 완료

#### M3 SOTA 변형 완료 (2026-02-07) ✓

- [x] Tube-MPPI + Ancillary Controller + Adaptive Temperature
  * 외란 강건성, body frame 피드백
  * 커밋: `f9052de` (966 lines)

- [x] Log-MPPI
  * log-space softmax, 수치 안정성
  * 커밋: `cd736f3` (774 lines)

- [x] Tsallis-MPPI
  * q-exponential 가중치, 탐색/집중 조절
  * 커밋: `d1790d6` (373 lines)

- [x] Risk-Aware MPPI
  * CVaR 기반 샘플 선택, 안전성
  * 커밋: `7a01534` (443 lines)

- [x] Smooth MPPI + Model Comparison
  * Δu input-lifting, 제어 부드러움
  * 커밋: `399cff6` (858 lines)

- [x] Stein Variational MPPI (SVMPC) + Model Comparison
  * SVGD 샘플 다양성, RBF 커널
  * 커밋: `4945838` (1109 lines)

- [x] Spline-MPPI + Model Comparison
  * B-spline 보간, 메모리 73.3% 감소
  * 커밋: `9c1c7ed` (853 lines)

- [x] SVG-MPPI + Model Comparison
  * Guide Particle SVGD, 99.9% 복잡도 감소
  * 커밋: `bedfec0` (1003 lines)

- [x] 전체 벤치마크 도구
  * mppi_all_variants_benchmark.py
  * 9개 변형 종합 비교
  * 9패널 시각화

#### 문서화 (2026-02-07) ✓

- [x] README.md 작성
  * 프로젝트 소개, 빠른 시작
  * 9개 변형 설명, 성능 비교
  * 사용 시나리오 추천

- [x] IMPLEMENTATION_STATUS.md 작성
  * 구현 현황 상세 문서
  * 성능 벤치마크 결과
  * 참고 논문 목록

#### Phase 4 (학습 모델 고도화) ✓

**Neural Network 학습 파이프라인 (2026-02-07)**

- [x] #202-1 데이터 수집 파이프라인
  * learning/data_collector.py
  * DataCollector (에피소드 기반), DynamicsDataset (정규화)
  * 데이터 저장/로드 (pickle)

- [x] #202-2 Neural Network 트레이너
  * learning/neural_network_trainer.py
  * PyTorch MLP (hidden_dims 설정)
  * Early stopping, LR scheduling
  * 학습 히스토리 플롯

- [x] #202-3 Neural Dynamics 학습 데모
  * examples/learned/neural_dynamics_learning_demo.py
  * Physics vs Neural vs Residual 3-way 비교
  * 9패널 비교 플롯
  * 커밋: `b2bc212`

**Gaussian Process 학습 파이프라인 (2026-02-07)**

- [x] #203-1 Gaussian Process 트레이너
  * learning/gaussian_process_trainer.py
  * GPyTorch 기반 (Exact GP / Sparse GP)
  * Multi-output GP (각 출력 차원 독립)
  * RBF/Matern 커널, ARD 지원

- [x] #203-2 GP vs Neural 비교 데모
  * examples/learned/gp_vs_neural_comparison_demo.py
  * 데이터 효율성 비교 (data_fraction)
  * 불확실성 보정 평가 (1σ, 2σ)
  * 12패널 종합 비교 플롯
  * 커밋: `ecfe346`

**온라인 학습 (2026-02-07)**

- [x] #204-1 온라인 학습 파이프라인
  * learning/online_learner.py
  * OnlineDataBuffer (순환 버퍼, FIFO)
  * OnlineLearner (자동 재학습 트리거)
  * 성능 모니터링 (적응도 추적)

- [x] #204-2 온라인 학습 데모
  * examples/learned/online_learning_demo.py
  * Sim-to-Real 도메인 변화 시뮬레이션
  * 실시간 모델 적응 (fine-tuning)
  * 적응 성능 추적 플롯

- [x] #204-3 학습 모델 문서화
  * docs/learned_models/LEARNED_MODELS_GUIDE.md (종합 가이드, 743 lines)
  * docs/learned_models/ONLINE_LEARNING.md (온라인 학습, 481 lines)
  * README.md 업데이트 (온라인 학습 예제)
  * 커밋: 84b222f

**결과물 정리 및 문서화 (2026-02-07)**

- [x] #204-4 Plot 결과 갤러리 생성
  * plots/ 디렉토리 정리 (9개 PNG)
  * 학습 모델 plot 2개 생성 (neural_dynamics_comparison, training_history)
  * README.md "📊 결과 갤러리" 섹션 추가 (~120 lines)

- [x] #204-5 PyTorch 2.6 호환성 수정
  * torch.load weights_only=False 추가 (4개 파일)
  * NeuralNetworkTrainer config 확장 (activation, dropout_rate)
  * NeuralDynamics 모델 로딩 개선

- [x] #204-6 GitHub Issue 생성
  * Phase 4 완료 공지 (예정)
  * 성능 요약 표 포함
  * 다음 단계 제안 (ROS2 통합)

#### Phase 4 전체 성과

| 항목 | 결과 |
|------|------|
| 학습 모델 타입 | 3개 ✅ (Neural, GP, Residual) |
| 학습 파이프라인 | 3개 ✅ (Neural/GP/Online Trainer) |
| Plot 갤러리 | 9개 ✅ (7 MPPI + 2 Learned) |
| 문서화 | 2개 ✅ (1224 lines) |
| 데모 스크립트 | 4개 ✅ |
| 유닛 테스트 | 5개 ✅ (전부 통과) |

**성능 벤치마크**

| 모델 | RMSE (m) | 추론 시간 (ms) | 불확실성 |
|------|----------|----------------|----------|
| Physics (Kinematic) | 0.007 | 4.6 | ❌ |
| Neural (Learned) | 0.068 | 24.0 | ❌ |
| Residual (Hybrid) | 0.092 | 31.0 | ❌ |

**다음 단계**
- [ ] ROS2 통합 (nav2 플러그인)
- [ ] 실제 로봇 테스트
- [ ] C++ 포팅

#### GPU 가속 (2026-02-08) ✓

- [x] gpu/ 패키지: TorchDiffDriveKinematic, TorchCompositeCost, TorchGaussianSampler
- [x] base_mppi.py device 분기 (CPU 코드 무수정)
- [x] RTX 5080: K=4096→4.3x, K=8192→8.1x speedup

#### 로봇 모델 확장 (2026-02-08) ✓

- [x] Ackermann (kinematic + dynamic + GPU)
- [x] Swerve Drive (kinematic + dynamic + GPU)
- [x] test_robot_models.py — 69개 테스트

#### Safety S3 고급 확장 (2026-02-08) ✓

- [x] Backup CBF (sensitivity propagation, multi-constraint QP)
- [x] Multi-robot CBF (Coordinator + RobotAgent, pairwise constraints)
- [x] MPCC Cost (PathParameterization, contouring/lag decomposition)

#### 학습 시스템 보강 (2026-02-08) ✓

- [x] Ensemble NN, MC-Dropout, UncertaintyAwareCost
- [x] ModelValidator (RMSE/MAE/R²), 체크포인트 버전 관리
- [x] 학습 시스템 버그 4건 수정 + 62개 테스트

#### 시뮬레이션 환경 (2026-02-18) ✓

- [x] 10개 시나리오 (S1~S10), 17파일, 4,815줄
- [x] 공통 인프라: environment ABC, obstacle generators, dynamic obstacles, waypoint FSM
- [x] 배치 실행: 10P/0F, ~218s
- [x] 문서: docs/SIMULATION_ENVIRONMENTS.md

#### Model Mismatch 동적 세계 통합 (2026-02-18) ✓

- [x] DynamicWorld: 5D DifferentialDriveDynamic 래퍼 (PD control + 프로세스 노이즈)
- [x] DynamicKinematicAdapter: 5D MPPI 내부 모델 (PD + friction forward_dynamics)
- [x] `--world dynamic` CLI: 5-Way 비교 (Kinematic/Neural/Residual/Dynamic/Oracle)
- [x] 5D MPPI 파라미터, 비용 함수, make_5d_reference 유틸리티
- [x] run_with_dynamic_world 시뮬 루프 (3D/5D 컨트롤러 혼합)
- [x] 실시간 live 비교 (perturbed/dynamic 모두 지원)
- [x] 기존 `--world perturbed` (4-Way) 완전 하위 호환

#### MAML 메타 학습 (2026-02-18) ✓

- [x] MAMLDynamics: NeuralDynamics 상속, save/restore 메타 파라미터
  * adapt(restore=True): 메타에서 재적응 (드리프트 방지)
  * gradient clipping, Adam/SGD 선택, use_adam 파라미터
  * sample_weights + temporal_decay 지원 추가
- [x] MAMLTrainer: FOMAML 파이프라인
  * 궤적 추종 학습 데이터 (50% 전진, 30% 곡선, 20% 랜덤)
  * 1000 iter × 8 tasks/batch, support/query 100/100
  * _generate_task_data_5d(): 5D state generation via DynamicWorld
- [x] ReptileTrainer: Reptile 메타 학습 (FOMAML 대안)
  * MAMLTrainer 상속, epsilon interpolation
  * `--meta-algo {fomaml,reptile}` CLI 지원
- [x] Residual MAML 아키텍처 (kinematic base + MAML residual)
  * 2-phase: 기구학 warm-up → Residual MAML 제어
  * 80 step마다 재적응, restore=True, 컨트롤러 reset 안 함
- [x] 7-Way 비교 데모 (--world dynamic)
  * MAML-3D RMSE: 0.074m, MAML-5D RMSE: 0.055m (noise=0.7)
  * Neural/Residual 오프라인 모델보다 우수
- [x] 테스트: test_maml.py (32개)
- [x] 문서: META_LEARNING.md + LEARNED_MODELS_GUIDE.md 업데이트

#### MAML 성능 개선 + 외란 시뮬레이션 (2026-02-21) ✓

- [x] DynamicKinematicAdapter: core models/kinematic/에 이동
  * 5D MPPI 내부 모델 (PD + friction forward_dynamics)
- [x] MAML-5D: 5D state + residual meta-training
  * 메타 학습 시 잔차 타겟 사용 → 분포 불일치 해결 (핵심 수정)
  * DynamicKinematicAdapter base + MAML-5D residual
  * noise=0.7에서 RMSE 0.055m (Dynamic 0.056m 돌파)
- [x] DisturbanceProfile: 4개 외란 프로필
  * WindGust, TerrainChange, Sinusoidal, Combined
  * DynamicWorld에 disturbance 주입 (get_force + get_param_delta)
- [x] CLI: `--noise FLOAT --disturbance {none,wind,terrain,sine,combined}`
- [x] AngleAwareTrackingCost + AngleAwareTerminalCost: core cost_functions.py에 추가
- [x] 4-seed 검증: 모든 시드에서 MAML-5D ≥ Dynamic
- [x] 테스트: test_maml.py (32개, 외란 프로필 테스트 포함)

#### Post-MAML 적응 (2026-02-21) ✓

- [x] EKFAdaptiveDynamics: 7D EKF 실시간 c_v/c_omega 추정 (오프라인 학습 불필요)
- [x] L1AdaptiveDynamics: 상태 예측기 + 외란 추정 + 저역통과 필터
- [x] ALPaCADynamics: FeatureExtractor + Bayesian linear regression
- [x] 10-Way 비교 데모 (7-Way + EKF/L1/ALPaCA)
- [x] test_ekf_dynamics.py(18), test_l1_adaptive.py(17), test_alpaca.py(23)

#### SVMPC 최적화 + 궤적 수정 (2026-02-21) ✓

- [x] SVMPC SPSA gradient: per-dim finite diff (60 rollouts) → SPSA (2 rollouts), 13x speedup
- [x] Efficient SVGD: (K,K,N,nu) 텐서 제거 → 행렬 연산 (503MB→0MB)
- [x] Merged kernel+bandwidth: `rbf_kernel_with_bandwidth()` K² 거리 1회 계산
- [x] Smooth MPPI cumsum 벡터화: Python for-loop → `np.cumsum`
- [x] Simulator `store_info=False`: 메모리 최적화 (300-500MB/cell 절약)
- [x] Slalom 궤적 적응형 진폭: `A_eff = min(amp, v_budget / (2π·f_inst))` 전 구간 v_max 이내
- [x] 장애물 회피 그리드 벤치마크: `--mode obstacle`, `--with-cbf`
- [x] test_trajectory.py: slalom 임계값 조정 + kinematic feasibility 테스트

#### Safety-Critical Control 확장 Phase S4 (2026-02-22) ✓

- [x] #740 HorizonWeightedCBFCost ✓ 2026-02-22
  * 시간 할인(γ^t) CBF 비용 — 가까운 미래 위반에 더 높은 페널티
- [x] #741 HardCBFCost ✓ 2026-02-22
  * 이진 거부 CBF — h<0이면 궤적 전체에 rejection_cost(1e6) 부과
- [x] #742 MPSController ✓ 2026-02-22
  * Model Predictive Shield — 간소화 Gatekeeper (stateless)
- [x] #743 AdaptiveShieldMPPIController ✓ 2026-02-22
  * 거리/속도 기반 적응형 α(d,v) Shield-MPPI
- [x] #744 CBFGuidedSamplingMPPIController ✓ 2026-02-22
  * 거부 샘플링 + ∇h 방향 편향 리샘플
- [x] #745 ShieldSVGMPPIController ✓ 2026-02-22
  * Shield + SVG-MPPI 결합 (안전 + 고품질 샘플)
- [x] #746 14종 Safety 벤치마크 데모 ✓ 2026-02-22
  * 4 시나리오 × 14 기법 비교, 5 메트릭, 6-panel 시각화

#### MPPI vs safe_control 벤치마크 (2026-02-22) ✓

- [x] #750 safe_control 패키지 연동 (CBF-QP, MPC-CBF) ✓ 2026-02-22
  * Unicycle2D + BaseRobot + CBFQP/MPCCBF API 연동
  * 장애물 형식: CBF-QP (7,1) columns, MPC-CBF (7,) 1D
  * robot_radius 이중 계산 방지 (safe_control 내부 처리)
- [x] #751 mppi_vs_safe_control_benchmark.py ✓ 2026-02-22
  * 8종 비교: CBF-QP, MPC-CBF, Vanilla, CBF-MPPI, Shield, AdaptiveShield, CBFGuided, ShieldSVG
  * 2 시나리오: circle_obstacle (원형 + 4장애물), gauntlet (직선 + 6장애물)
  * pure pursuit 레퍼런스 (장애물 회피 후 경로 복귀)
  * path-following RMSE (시간 기반 → 궤적 최근접점 기반)
  * --live 모드 (4-panel FuncAnimation), --no-plot, --methods
- [x] #752 AdaptiveShield alpha 공식 수정 (버그 수정) ✓ 2026-02-22
  * 기존 (잘못됨): α = α_base + α_dist·exp(...) → 가까울수록 α 증가 → 덜 보수적
  * 수정: α = α_base · σ(k·(d-d_safe)) / (1+α_vel·|v|) → 가까울수록 α 감소 → 더 보수적
  * 결과: 충돌 179건 → 0건 (100% 안전)
- [x] #753 adaptive_safety_benchmark.py ✓ 2026-02-22
  * 적응 제어(EKF/L1) + 안전 제어(CBF/Shield) 결합 9종 비교
  * DynamicWorld 모델 불일치 시뮬레이션
  * --live, --no-mismatch, --methods, --scenario CLI

#### 종합 통계

**총 구현 코드**: ~30,000+ 라인
**유닛 테스트**: 527개 passed (43 파일)
**MPPI 변형**: 9개 (전부 완성 ✅)
**모델 타입**: 5개 (DiffDrive Kinematic/Dynamic, Ackermann, Swerve, Learned)
**학습 모델**: 9개 (Neural/GP/Residual/Ensemble/MC-Dropout/MAML/EKF/L1/ALPaCA ✅)
**메타 학습**: FOMAML + Reptile + Residual Meta-Training ✅
**안전 제어**: 16개 (기존 10종 + HorizonCBF/HardCBF/MPS/AdaptiveShield/CBFGuided/ShieldSVG ✅)
**시뮬레이션 환경**: 10개 시나리오 + 외란 프로필 4종 ✅
**데모**: Model Mismatch 10-Way + Safety 14-Way + MPPI vs safe_control 8-Way ✅
**문서**: README, META_LEARNING, SIMULATION_ENVIRONMENTS, SAFETY_CRITICAL_CONTROL, LEARNED_MODELS_GUIDE 등

---

## 사용 방법

### 다음 작업 하나 처리
```bash
claude-todo-worker
```

### 특정 작업 처리
```bash
claude-todo-task "#001"
```

### 모든 작업 연속 처리
```bash
claude-todo-all
```

---

## 우선순위 기준

- **P0 (High)**: Phase 1 - 기구학 모델 및 시뮬레이션 인프라 (M1)
- **P1 (Medium)**: Phase 2, 3 - 동역학/학습 모델, ROS2 통합 (M2/M3/M4)
- **P2 (Low)**: Phase 4, MPPI 변형, 문서화, DevOps (추가 기능)

---

## Phase별 예상 타임라인

```
Phase 1 (M1) - 기구학 모델 및 시뮬레이션: 2주 (P0)
Phase 2 (M2) - 동역학 모델: 2주 (P1)
Phase 3 (M3) - 학습 모델: 1-2주 (P1)
Phase 4 - pytorch_mppi 개선: 1주 (P2, 선택적)

총 6-7주 예상
```

---

## 핵심 설계 원칙

1. **통일된 인터페이스**: RobotModel 추상 클래스로 모든 타입 (Kinematic/Dynamic/Learned) 통합
2. **점진적 확장**: Phase 1(기구학) → Phase 2(동역학) → Phase 3(학습)
3. **하위 호환성**: BatchDynamicsWrapper는 모든 모델 타입 지원, MPPI 수정 불필요
4. **벡터화 강제**: NumPy broadcasting, GPU 준비
5. **함수/클래스 하이브리드**: pytorch_mppi 유연성 + 타입 안정성

---

## 작업 규칙

1. 각 작업은 독립적인 기능 단위
2. 작업 완료 시 테스트 필수 (pytest)
3. PR 생성 및 리뷰 후 머지
4. TODO.md 업데이트는 자동으로 처리됨
5. ASCII 플로우로 진행 상황 시각화
