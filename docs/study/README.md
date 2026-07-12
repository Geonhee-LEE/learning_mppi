# MPPI 학습 로드맵 (Study Series)

최적 제어의 기초부터 MPPI, 안전 제어, 생성 모델 기반 샘플링까지 —
이 저장소의 구현(43종 MPPI 변형, 22종 안전 제어)을 이해하는 데 필요한
이론을 **개념 빌드업 → 유도 → 구현 매핑 → 연습문제** 순서로 정리한 스터디 시리즈.

기존 문서(`docs/MPPI_THEORY.md`, `docs/SAFETY_THEORY.md` 등)가 **레퍼런스**(변형별 요약 사전)라면,
이 시리즈는 **교과서**(처음부터 끝까지 읽는 흐름)를 지향한다.

---

## 추천 학습 순서

```
                 ┌──────────────────────────┐
                 │ 01. MPC FUNDAMENTALS     │
                 │ 최적 제어 → LQR → MPC     │
                 └──────┬───────────┬───────┘
                        │           │
          "MPC를 샘플링으로"      "제약을 안전으로"
                        │           │
                        ▼           ▼
   ┌──────────────────────────┐  ┌──────────────────────────┐
   │ 02. MPPI FUNDAMENTALS    │  │ 03. CBF FUNDAMENTALS     │
   │ path integral →          │  │ set invariance → CBF-QP  │
   │ info-theoretic 유도 →     │  │ → relative degree        │
   │ 변형 분류                 │  └──────────┬───────────────┘
   └──────┬───────────┬───────┘             │
          │           │                     │
          │           └──────────┬──────────┘
          │                      ▼
          │        ┌──────────────────────────┐
          │        │ 04. ADVANCED SAFETY      │
          │        │ gatekeeper / HJ /        │
          │        │ chance constraints /     │
          │        │ shielding                │
          │        └──────────────────────────┘
          ▼
   ┌──────────────────────────┐
   │ 05. GENERATIVE MODELS    │
   │ FOR CONTROL              │
   │ VAE / score / diffusion  │
   │ / CFM → 학습 제안 분포     │
   └──────────────────────────┘

   실선 경로 A (제어 트랙):  01 → 02 → 05
   실선 경로 B (안전 트랙):  01 → 03 → 04
   완주:                    01 → 02 → 03 → 04 → 05
```

---

## 문서별 소개

### [01_MPC_FUNDAMENTALS.md](01_MPC_FUNDAMENTALS.md) — 최적 제어 → LQR → MPC

시리즈의 출발점. 최적 제어 문제의 정식화(비용 함수, 동역학 제약)에서 시작해
동적 계획법과 Riccati 방정식으로 LQR을 유도하고, 유한 호라이즌 반복 최적화 +
receding horizon 원리로서의 MPC로 확장한다. 이후 모든 문서가 사용하는 표기법
(상태 x, 제어 u, 호라이즌 N, 비용 S)과 "왜 매 스텝 다시 푸는가"라는 MPC의
핵심 직관을 여기서 확립한다.

- **선수 지식**: 선형대수(고유값, 이차형식), 미적분, 기초 동역학
- **관련 구현**: `mppi_controller/controllers/mppi/ancillary_controller.py` (LQR류 피드백),
  `feedback_mppi.py` (Riccati 게인)

### [02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md) — Path Integral → Information-Theoretic MPPI → 변형 분류

이 저장소의 심장. 확률적 최적 제어의 path integral 관점에서 출발해,
자유 에너지와 KL divergence를 이용한 information-theoretic 유도
(Williams et al.)로 MPPI 업데이트 법칙 `U ← Σ softmax(-S/λ)·V`를 얻는다.
temperature λ, ESS, warm start의 의미를 해석하고, 저장소의 43종 변형을
샘플링/가중치/반복/구조 축으로 분류하는 지도를 제공한다.

- **선수 지식**: 01편, 기초 확률론(가우시안, 기대값), KL divergence
- **관련 구현**: `base_mppi.py`, `mppi_params.py`, `sampling.py`, `cost_functions.py`

### [03_CBF_FUNDAMENTALS.md](03_CBF_FUNDAMENTALS.md) — Set Invariance → CBF-QP → Relative Degree

안전 트랙의 시작. "안전 = 집합 불변성(set invariance)"이라는 관점에서
Nagumo 정리 → Control Barrier Function의 정의 → 안전 필터로서의 CBF-QP를
유도한다. relative degree가 높은 시스템(위치 장벽 + 가속 입력)에서의
HOCBF 확장까지 다룬다.

- **선수 지식**: 01편 (Lyapunov 안정성 개념이 있으면 더 좋음)
- **관련 구현**: `cbf_cost.py`, `cbf_safety_filter.py`, `clf_cbf_qp.py`,
  `hocbf_cost.py`, `c3bf_cost.py`

### [04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md) — Gatekeeper / HJ / Chance Constraints / Shielding

안전 트랙 심화. 단일 스텝 필터(CBF-QP)를 넘어: 백업 궤적을 검증하는
gatekeeper, Hamilton-Jacobi 도달가능성 기반 안전 가치 함수(DualGuard-MPPI),
확률적 제약(chance constraints), 그리고 shielding(Shield-MPPI 계열)까지 —
저장소의 22종 안전 제어 기법의 이론적 뼈대를 제공한다.

- **선수 지식**: 02편 + 03편
- **관련 구현**: `gatekeeper.py`, `dualguard_mppi.py`, `shield_mppi.py`,
  `chance_constraint_cost.py`, `backup_cbf_filter.py`, `dbas_mppi.py`

### [05_GENERATIVE_MODELS_FOR_CONTROL.md](05_GENERATIVE_MODELS_FOR_CONTROL.md) — VAE / Score / Diffusion / CFM

제어 트랙 심화. MPPI 가우시안 제안 분포의 한계(단봉성, 비용 지형 무시)에서 출발해,
제안 분포를 학습된 생성 모델로 대체하는 이론을 다룬다: ELBO 유도(Latent-MPPI),
DSM loss의 Vincent 트릭 전체 유도(SG-MPPI), DDPM/DDIM(Diffusion-MPPI),
그리고 핵심 챕터인 Conditional Flow Matching 정리(Flow-MPPI).
ring buffer + elite selection, zero-init graceful degradation 등
저장소 횡단 설계 패턴으로 마무리한다.

- **선수 지식**: 02편 (특히 importance sampling), 신경망 학습 기초
- **관련 구현**: `flow_mppi.py`, `score_guided_mppi.py`, `diffusion_mppi.py`,
  `latent_mppi.py`, `step_mppi.py`, `learning/flow_matching_trainer.py`

---

## 기존 레퍼런스 문서와의 관계

| 문서 | 성격 | 이 시리즈와의 관계 |
|------|------|------------------|
| [../MPPI_THEORY.md](../MPPI_THEORY.md) | 43종 MPPI 변형 레퍼런스 사전 | 02편으로 기초를 다진 후 변형별 상세를 찾아볼 때 |
| [../SAFETY_THEORY.md](../SAFETY_THEORY.md) | 22종 안전 제어 레퍼런스 사전 | 03·04편의 이론이 각 기법에 어떻게 적용되는지 확인용 |
| [../LEARNING_THEORY.md](../LEARNING_THEORY.md) | 학습 동역학 모델 (BNN/GP/Ensemble 등) 이론 | 05편이 "제안 분포 학습", 이 문서는 "동역학 모델 학습" — 상보적 |
| [../CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) | 안전 기법 벤치마크 리포트 | 03·04편 이론의 실증 데이터 |
| [../TUTORIALS.md](../TUTORIALS.md) | 전체 데모 실행 가이드 | 각 편의 "실행해 보기"의 상세 버전 |

읽기 전략: **시리즈(01→05)로 이론의 척추를 세우고, 레퍼런스 문서로 살을 붙인다.**

---

## 주제 → 벤치마크 빠른 참조

각 문서를 읽은 뒤 바로 돌려볼 수 있는 대표 실험 (전체 목록은 `CLAUDE.md`/`TUTORIALS.md`):

```bash
# 01·02편 — Vanilla MPPI와 변형 비교
python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory circle --no-plot
PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --scenario obstacles

# 02편 — 샘플링/가중치 변형 체감
PYTHONPATH=. python examples/comparison/lp_mppi_benchmark.py --all-scenarios          # 주파수 도메인 스무딩
PYTHONPATH=. python examples/comparison/deterministic_mppi_benchmark.py --all-scenarios  # 결정론적 샘플링
PYTHONPATH=. python examples/comparison/spectral_risk_mppi_benchmark.py --all-scenarios  # 리스크 가중치

# 03편 — CBF 계열
PYTHONPATH=. python examples/comparison/cbf_mppi_obstacle_avoidance_demo.py
PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --all-scenarios

# 04편 — 고급 안전
PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --all-scenarios   # HJ 가치 함수
PYTHONPATH=. python examples/comparison/adaptive_safety_benchmark.py                  # shielding 계열

# 05편 — 생성 모델 제안 분포
PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --live --scenario obstacles
PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --all-scenarios
PYTHONPATH=. python examples/comparison/latent_mppi_benchmark.py
PYTHONPATH=. python examples/comparison/step_mppi_benchmark.py --live --scenario online_learning
```

> 헤드리스 환경 규칙: 결과는 `plots/*.png`, `--live`는 `plots/*.mp4`/`*.gif`로 저장됨.
> 핸드폰 확인: `python -m http.server 8888` → `http://<PC_IP>:8888/plots/`

---

## 외부 추천 자료 Top 10

| # | 자료 | 유형 | 대응 문서 |
|---|------|------|----------|
| 1 | Borrelli, Bemporad & Morari, *Predictive Control for Linear and Hybrid Systems* | 교재 | 01 |
| 2 | Rawlings, Mayne & Diehl, *Model Predictive Control: Theory, Computation, and Design* | 교재 | 01 |
| 3 | Williams et al., *Information Theoretic MPC for Model-Based RL* (ICRA 2017) + *IT-MPPI* (T-RO 2018) | 논문 | 02 |
| 4 | Kappen, *Path Integrals and Symmetry Breaking for Optimal Control Theory* (2005) | 논문 | 02 |
| 5 | Ames et al., *Control Barrier Functions: Theory and Applications* (ECC 2019 튜토리얼) | 서베이 | 03 |
| 6 | Bansal et al., *Hamilton-Jacobi Reachability: A Brief Overview* (CDC 2017) | 서베이 | 04 |
| 7 | Lipman et al., *Flow Matching for Generative Modeling* (ICLR 2023) | 논문 | 05 |
| 8 | Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution* (NeurIPS 2019) + Yang Song 블로그 | 논문/블로그 | 05 |
| 9 | Ho et al., *DDPM* (NeurIPS 2020) + Chi et al., *Diffusion Policy* (RSS 2023) | 논문 | 05 |
| 10 | Underactuated Robotics (Russ Tedrake, MIT 강의) — 특히 최적 제어/궤적 최적화 장 | 강의 | 01·02 |

---

## 표기법 통일 (시리즈 공통)

| 기호 | 의미 |
|------|------|
| `x ∈ R^nx` | 상태 (state) |
| `u ∈ R^nu` | 제어 입력 (control) |
| `N` | 예측 호라이즌 길이 |
| `K` | MPPI 샘플 수 |
| `U = (u_0..u_{N-1})` | 제어 시퀀스 (명목 해) |
| `S(V)` | 궤적 비용 (rollout cost) |
| `λ` | temperature (softmax 온도) |
| `h(x) ≥ 0` | 안전 집합을 정의하는 장벽 함수 |
| `q_θ(U\|x)` | 학습된 제안 분포 (05편) |

모든 컨트롤러는 `compute_control(state, reference_trajectory) -> (control, info)`
시그니처를 준수한다 (프로젝트 인터페이스 규칙).
