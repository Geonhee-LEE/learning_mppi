# MPPI 학습 로드맵 (Study Series)

최적 제어의 기초부터 MPPI, 안전 제어, 생성 모델 기반 샘플링까지 —
이 저장소의 구현(43종 MPPI 변형, 22종 안전 제어)을 이해하는 데 필요한
이론을 **개념 빌드업 → 유도 → 구현 매핑 → 연습문제** 순서로 정리한 스터디 시리즈.

기존 문서(`docs/MPPI_THEORY.md`, `docs/SAFETY_THEORY.md` 등)가 **레퍼런스**(변형별 요약 사전)라면,
이 시리즈는 **교과서**(처음부터 끝까지 읽는 흐름)를 지향한다.

> ### 👉 처음이라면 [**00_START_HERE.md**](00_START_HERE.md)부터
>
> 이 README는 문서 **지도**(로드맵·오픈소스 맵·동향)이고,
> [00_START_HERE.md](00_START_HERE.md)는 **커리큘럼**(Phase 0~5로 나눈 순서 +
> 단계별 읽을 것/실행할 것/자가 점검 체크박스)입니다.
> **무엇을 어떤 순서로 공부할지는 00번을 기준으로 삼으세요.** 이 README는
> 중간에 특정 문서·라이브러리·동향을 찾아볼 때 참조합니다.

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

## 코드 워크스루 (06–08) — 이론에서 구현으로

01–05가 "왜 이 수식인가"를 다뤘다면, 06–08은 **실제 소스를 함수 단위로
해부**한다. 모든 발췌는 file:line 참조가 붙고, 형식은 공통이다:
**코드 발췌 → 해설 → 왜 이렇게 구현했나(트레이드오프) → 흔한 실수**.

### [06_CODE_WALKTHROUGH_CORE.md](06_CODE_WALKTHROUGH_CORE.md) — 핵심 파이프라인

한 제어 사이클의 콜 그래프(Simulator → compute_control → sample → rollout →
cost → weights)를 shape 흐름 `(K,N,nu)→(K,N+1,nx)→(K,)`와 함께 추적한다.
`RobotModel` ABC의 벡터화 규약, `compute_control` 라인 단위 해설(warm start의
0-채움, min-shift 수치 안정화, 시프트 후 `U[0]` 반환 quirk), seed=None 재현성
함정, 커스텀 비용/샘플러 작성 체크리스트. §9의 Top-K MPPI 실습은 실제 실행
검증됨 (RMSE 0.157 m).

- **선수 지식**: 02편 §3–6
- **핵심 소스**: `base_mppi.py`, `sampling.py`, `cost_functions.py`,
  `dynamics_wrapper.py`, `simulation/simulator.py`

### [07_CODE_WALKTHROUGH_VARIANTS.md](07_CODE_WALKTHROUGH_VARIANTS.md) — 변형 확장 패턴

43개 변형을 변형별이 아니라 **확장 패턴별**(A: `_compute_weights` 교체,
B: `compute_control` 전체 교체/DIAL, C: 샘플러 교체, D: 최적화 관점,
E: 피드백 결합, F: 학습 결합)로 해부한다. Log/Tsallis/CVaR, DIAL/CMA/Biased,
LP/Halton/Hermite, PGD/GN, Tube/Riccati, Step-MPPI 대표 코드 발췌 +
새 변형 추가 시 파일 8종 체크리스트 (실제 커밋 diff-stat 인용).

- **선수 지식**: 06편, 02편 §7
- **핵심 소스**: 각 변형 파일 + `all_37_variants_benchmark.py` 레지스트리

### [08_CODE_WALKTHROUGH_SAFETY.md](08_CODE_WALKTHROUGH_SAFETY.md) — 안전 스택 구현

3층위(비용/필터/게이트)가 코드 어디에 끼어드는지부터, CBF 비용의 배치
벡터화 shape 추적, HOCBF 캐스케이드와 `detect_relative_degree`의 유한차분
g(x) 추출, CLF-CBF-QP의 3-경로 분기, Shield의 ESS 붕괴 지점 코드 지목,
Gatekeeper 상태 기계까지. §10의 사각형 장애물(BoxBarrierCost) 실습은 실제
실행 검증됨 (침입 0회) — local minimum 실패 사례 재현 포함.

- **선수 지식**: 03·04편, 06편
- **핵심 소스**: `cbf_cost.py`, `hocbf_cost.py`, `stochastic_cbf.py`,
  `clf_cbf_qp.py`, `shield_mppi.py`, `gatekeeper.py`, `dualguard_mppi.py`

---

## 부록 안내 — 각 문서의 심화 학습 부록 (A–E)

01–05 각 문서의 말미에는 본문을 넘어 스스로 공부를 이어갈 수 있도록
**동일한 구조의 부록 A–E**가 붙어 있다:

| 부록 | 내용 | 활용법 |
|------|------|--------|
| **A. 주석 달린 핵심 레퍼런스** | 해당 분야 원전 논문 10여 편 + 각 2문장 주석 (arXiv 링크 검증됨) | 본문의 유도가 어느 논문의 어느 정리인지 역추적할 때 |
| **B. 최근 연구 동향 (2024–2026)** | 분야별 최신 흐름 4–6개, 대표 논문 링크 | 서베이/논문 아이디어 발굴, "지금 어디까지 왔나" 파악 |
| **C. 오픈소스 생태계** | 주제별 실전 라이브러리 표 (실재/활성 확인됨) | 이 repo 밖에서 실험을 확장할 때 (아래 "오픈소스 맵"이 전체 요약) |
| **D. 더 공부하기** | 공개 강의·교재·블로그 (링크 확인됨) | 본문이 압축한 이론의 풀 버전 수강 |
| **E. FAQ 포인터** | "~이 안 되면?" → 본문 절/구현 파일 매핑 | 실험 중 막혔을 때 사전식 조회 |

문서별 부록 바로가기:
[01](01_MPC_FUNDAMENTALS.md) · [02](02_MPPI_FUNDAMENTALS.md) · [03](03_CBF_FUNDAMENTALS.md) ·
[04](04_ADVANCED_SAFETY.md) · [05](05_GENERATIVE_MODELS_FOR_CONTROL.md)
(각 문서 목차의 마지막 항목 "부록"으로 이동)

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

## 이 repo로 실험하며 공부하기 — 궁금증 → 벤치마크 매핑

이론을 읽다가 생기는 전형적인 질문을, 바로 돌려서 답을 확인할 수 있는 실험으로 매핑:

| 궁금증 | 실행할 명령 | 볼 지표 |
|--------|------------|---------|
| "λ(temperature)를 바꾸면 탐색-수렴 균형이 정말 변하나?" (02편 §ESS) | `python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory circle --no-plot` 후 `mppi_params.py`의 `temperature` 수정 재실행 | RMSE vs info dict의 `ess` |
| "샘플 수 K를 줄이면 어떤 변형이 먼저 무너지나?" (02편 샘플링 축) | `PYTHONPATH=. python examples/comparison/deterministic_mppi_benchmark.py --all-scenarios` (dsMPPI는 K=64에서도 동작) | K별 RMSE/충돌 수 |
| "CBF 비용과 hard 필터는 실제로 뭐가 다른가?" (03편) | `PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --all-scenarios` | MinClearance, 충돌 수, 추적 RMSE 트레이드오프 |
| "HJ 가치 함수 필터가 CBF보다 나은 상황은?" (04편) | `PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --all-scenarios` | 시나리오별 MinClearance 비교 |
| "학습된 제안 분포는 정말 장애물 양쪽 모드를 잡나?" (05편 §1.2) | `PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --live --scenario obstacles` | 샘플 궤적 분포 애니메이션 (`plots/*.mp4`) |
| "온라인 학습 비용은 실시간성을 얼마나 깨뜨리나?" (05편 §5.3) | `PYTHONPATH=. python examples/comparison/step_mppi_benchmark.py --live --scenario online_learning` | SolveMs 추이 (학습 스텝에서의 스파이크) |

---

## 오픈소스 맵 — 주제별 대표 라이브러리 한눈에

각 문서 부록 C의 전체 요약. 이 repo에서 원리를 익힌 뒤 실전 규모로 확장할 때의 진입점
(링크는 2026-07 기준 확인):

**MPC / 최적 제어 (01편)**

- [acados](https://github.com/acados/acados) — 임베디드급 고속 NMPC 솔버 (C + Python/MATLAB 인터페이스, HPIPM 기반)
- [do-mpc](https://github.com/do-mpc/do-mpc) — 파이썬 로버스트 MPC/MHE 툴박스 — 프로토타이핑에 최적
- [CasADi](https://github.com/casadi/casadi) — 위 둘의 기반이 되는 자동미분 + 최적화 심볼릭 프레임워크

**MPPI / 샘플링 기반 제어 (02편)**

- [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi) — PyTorch MPPI (SMPPI/KMPPI 변형 + 자동 튜너 포함) — 이 repo와 가장 유사한 스타일
- [MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) — C++/CUDA 헤더 온리 MPPI/Tube-MPPI/RMPPI ([논문](https://arxiv.org/abs/2409.07563)) — 실기체 배포급 성능
- [storm](https://github.com/NVlabs/storm) — NVIDIA GPU 병렬 MPPI 매니퓰레이터 모션 툴킷 (SDF 충돌 비용)

**안전 제어 (03·04편)**

- [cbfkit](https://github.com/bardhh/cbfkit) — JAX 기반 CBF + MPPI 통합 툴박스 (ROS2 지원)
- [safe_control](https://github.com/tkkim-robot/safe_control) — CBF-QP/MPC-CBF/gatekeeper 등 내비게이션 안전 제어기 모음 (이 repo의 CBFKIT_INSPIRED_SAFETY.md와 연관)
- [safe-control-gym](https://github.com/learnsyslab/safe-control-gym) — 안전 학습 제어 벤치마크 환경 (PyBullet + CasADi 심볼릭 동역학)
- [hj_reachability](https://github.com/StanfordASL/hj_reachability) — JAX HJ 도달가능성 솔버 — 04편 HJ 가치 함수의 정식 계산 도구

**생성 모델 × 제어 (05편)**

- [torchcfm](https://github.com/atong01/conditional-flow-matching) — CFM/OT-CFM 표준 구현 — 05편 §4를 코드로
- [diffusion_policy](https://github.com/real-stanford/diffusion_policy) — Diffusion Policy 공식 코드 (RSS 2023)
- [lerobot](https://github.com/huggingface/lerobot) — HuggingFace 로봇 학습 허브 (Diffusion Policy·ACT·π0 계열 + 데이터셋)

---

## 최신 동향 한눈에 (2024–2026)

시리즈 전체를 관통하는 분야 횡단 메가트렌드 (각 문서 부록 B의 종합):

1. **생성 모델·파운데이션 모델과 계획의 융합** —
   flow matching이 로봇 파운데이션 정책(π0 등 VLA)의 표준 액션 헤드가 되고,
   반대로 샘플링 MPC의 해를 생성 모델로 증류(GPC)하거나 동역학 모델로 score를
   직접 계산(Model-Based Diffusion)하는 양방향 융합이 진행 중이다.
   "최적화가 데이터를 만들고 생성 모델이 최적화를 가속"하는 루프가 05편의 핵심 서사다.

2. **안전 필터의 통일 이론화** — CBF, HJ 도달가능성, predictive safety filter가
   "안전 집합 불변성 + 최소 개입"이라는 하나의 틀로 정리되고 있다.
   나아가 SafeDiffuser처럼 안전 제약을 생성 과정 내부에 심는 연구가 등장하며
   "생성 후 필터링"에서 "안전한 것만 생성"으로 무게중심이 이동 중이다 (03·04편).

3. **GPU 병렬 샘플링 계획의 주류화** — MPPI-Generic, storm, JAX 생태계(cbfkit,
   hj_reachability)가 보여주듯 K개 rollout의 완전 병렬화가 실기체 표준이 되었다.
   샘플링 기반 방법이 gradient 기반 MPC의 실시간성 우위를 잠식하면서,
   미분 불가능한 비용(충돌 검사, 학습 모델)을 그대로 쓸 수 있다는 장점이 부각되고 있다.

4. **학습 기반 안전 인증** — 학습된 CBF/가치 함수(neural certificate)와
   HJ 근사(DeepReach 계열)로 고차원 시스템의 안전 보증을 확장하는 흐름.
   "학습 요소가 들어가도 학습 전 = 보수적 기본 동작"이라는 graceful degradation
   설계(이 repo의 zero-init/fallback 패턴)가 실무 표준으로 자리 잡는 중이다 (04·05편).

5. **World model 기반 잠재 공간 계획의 성숙** — DreamerV3가 범용성을,
   TD-MPC2가 "잠재 공간에서의 MPPI류 로컬 최적화"의 확장성을 입증하며,
   물리 모델 없이 관측만으로 계획하는 파이프라인이 성숙 단계에 들어섰다.
   이 repo의 Latent-MPPI/World-Model-MPPI(39th)가 이 계보의 최소 구현이다 (05편 §2).

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
