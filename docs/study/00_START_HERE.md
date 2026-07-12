# 00. START HERE — 공부 순서 가이드

> **이 문서가 시리즈의 진입점입니다.** 여기서 정한 순서대로 읽고, 실행하고,
> 체크박스를 채워가며 진행하세요. 각 단계는 "읽기 → 실행 → 자가 점검"의
> 3박자로 구성되어 있고, 전부 이 repo 안에서 해결됩니다.
>
> 문서 전체 지도가 필요하면 [README.md](README.md)(로드맵/오픈소스 맵/동향),
> 특정 기법의 수식 레퍼런스가 필요하면 [MPPI_THEORY.md](../MPPI_THEORY.md) /
> [SAFETY_THEORY.md](../SAFETY_THEORY.md)를 사전처럼 찾아보세요.

---

## 전체 커리큘럼 한눈에

```
 Phase 0          Phase 1              Phase 2              Phase 3
 준비(0.5일)   →   MPPI 기초(1주)    →   코드로 확인(1주)   →   안전 제어(1~2주)
 환경 설정        01 MPC 기초           06 코드 워크스루      03 CBF 기초
 데모 1회 실행     02 MPPI 유도          변형 하나 만들기      04 고급 안전
                                                           08 안전 코드

                  Phase 4                       Phase 5
               →  변형/학습 심화(1~2주)      →   연구 확장(상시)
                  07 변형 패턴                  부록 B 동향 → 논문 읽기
                  05 생성 모델                  자기 변형 구현/벤치마크
```

- 표준 경로 기준 **약 5~7주** (하루 1~2시간). 급하면 Phase 1→2→3만 해도
  repo를 다루는 데 충분합니다.
- **트랙 선택**: 안전이 급하면 Phase 2 후 3으로 (05·07은 뒤로), 학습/생성
  모델이 급하면 Phase 2 후 4로 (03·04는 뒤로). Phase 1–2는 공통 필수입니다.

---

## Phase 0 — 준비 (반나절)

**목표**: 환경을 갖추고, 뭘 공부하게 될지 감을 잡는다.

- [ ] repo 클론 + 의존성 설치, 전체 테스트가 도는지 확인
  ```bash
  python -m pytest tests/test_base_mppi.py -v --override-ini="addopts="
  ```
- [ ] 데모 1회 실행 — MPPI가 "움직이는 것"을 먼저 본다
  ```bash
  python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory circle --no-plot
  PYTHONPATH=. python examples/mppi_all_variants_benchmark.py --trajectory figure8
  ```
  (플롯은 `plots/`에 저장됨 — CLAUDE.md의 "데모 결과 출력 규칙" 참조)
- [ ] [README.md](README.md)의 "추천 학습 순서" 그래프와 "표기법 통일" 절만
  훑어본다 (10분)

**자가 점검**: plots/에 결과 그림이 생겼는가? K, N, λ가 뭘 뜻하는지
한 줄씩 말할 수 있는가? (아직 몰라도 됨 — Phase 1에서 배움)

---

## Phase 1 — MPPI 기초 이론 (1주)

**목표**: "왜 softmax(-S/λ) 가중 평균이 최적 제어인가"를 유도로 이해한다.

### 1-1. [01_MPC_FUNDAMENTALS.md](01_MPC_FUNDAMENTALS.md) (2~3일)

- [ ] §1–4: 최적 제어 표기 → HJB → LQR → receding horizon (핵심)
- [ ] §5–7: 제약 처리, NMPC, **MPC vs MPPI 비교표** (§7이 이 repo의 존재 이유)
- [ ] 연습문제 §8에서 2개 이상 풀기 (특히 LQR 손계산)
- 시간이 없으면: §2 HJB와 §6은 건너뛰고 §3(LQR)–§4(receding horizon)–§7만.

### 1-2. [02_MPPI_FUNDAMENTALS.md](02_MPPI_FUNDAMENTALS.md) (3~4일) ★ 시리즈의 중심

- [ ] §1: free energy — soft-min 직관
- [ ] §3: **Williams 2017 전체 유도** (한 줄씩 따라 쓸 것 — 이 시리즈에서
  가장 공들여 읽어야 할 절)
- [ ] §4–5: λ/σ/ESS — 파라미터 튜닝의 이론적 근거
- [ ] §6: 의사코드 ↔ `base_mppi.py` 행 번호 대응표 (코드를 열어놓고 대조)
- [ ] §7: 43개 변형 분류 지도 — 외우지 말고 "이런 축이 있구나"만
- [ ] 연습문제 §9의 **1D 이중우물 스니펫을 직접 실행**

**자가 점검**:
- λ → 0과 λ → ∞ 극한에서 MPPI가 각각 무엇이 되는지 유도할 수 있는가?
- ESS가 낮으면 왜 문제이고, 어떤 처방이 있는지 3가지를 말할 수 있는가?
- importance sampling 보정항이 왜 필요한지 설명할 수 있는가?

---

## Phase 2 — 코드로 확인 (1주)

**목표**: 이론의 각 줄이 코드의 어느 줄인지 알고, 직접 변형을 하나 만든다.

### 2-1. [06_CODE_WALKTHROUGH_CORE.md](06_CODE_WALKTHROUGH_CORE.md) (3~4일)

- [ ] §1 콜 그래프 + shape 흐름표를 손으로 다시 그려보기
- [ ] §6 `compute_control` 라인 단위 해설 — **02편 §3의 유도와 나란히 놓고**
  "이 수식 = 이 줄" 매칭
- [ ] §4 seed 함정, §5 배치 규약 — 흔한 실수 절은 전부 정독
- [ ] §9 실습: **Top-K MPPI를 그대로 따라 만들고 실행** (검증된 예제)

### 2-2. 실험으로 체감 (2~3일)

- [ ] λ 극한 실험: 벤치마크에서 λ를 0.05/1.0/20으로 바꿔 ESS·RMSE 변화 관찰
  ```bash
  PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --scenario simple
  ```
- [ ] K를 512→64로 줄여 성능 붕괴 관찰 → dsMPPI/RF-MPPI가 왜 필요한지 체감
- [ ] [../TUTORIALS.md](../TUTORIALS.md)에서 관심 가는 장 하나 실행

**자가 점검**: `_compute_weights`만 오버라이드해서 새 가중치 함수를
30분 안에 붙일 수 있는가? info dict의 ess를 보고 λ를 조정할 수 있는가?

---

## Phase 3 — 안전 제어 (1~2주)

**목표**: "안전 보장"이라는 말의 정확한 의미(층위)를 구분하고, CBF를
손계산부터 구현까지 다룬다.

### 3-1. [03_CBF_FUNDAMENTALS.md](03_CBF_FUNDAMENTALS.md) (3~4일)

- [ ] §1–3: 안전 집합 → Nagumo → class-K → CBF 정의 (그림 중심으로)
- [ ] §4: **Lie derivative 손계산을 종이에 직접 재현** (diffdrive 예제)
- [ ] §5: CBF-QP 해석해 유도
- [ ] §8: 실패 모드 4종 — 실전에서 가장 자주 만나는 내용
- [ ] 연습문제 §10에서 2개 이상

### 3-2. [04_ADVANCED_SAFETY.md](04_ADVANCED_SAFETY.md) (3~4일)

- [ ] §1: 3층위 스펙트럼 — **이 문서의 뼈대. 표를 외울 가치가 있음**
- [ ] §2: Gatekeeper 귀납 논증 (무한 시간 보장이 어떻게 가능한지)
- [ ] §3–4: HJ vs CBF, 확률 보장 3계보 (UT/CP/martingale) — 비교표 위주
- [ ] §7: 벤치마크 데이터로 안전-성능 트레이드오프 읽기

### 3-3. [08_CODE_WALKTHROUGH_SAFETY.md](08_CODE_WALKTHROUGH_SAFETY.md) (2~3일)

- [ ] §2 CBF 비용 배치 벡터화 + §3 HOCBF (rd=2가 왜 문제인지 코드로)
- [ ] §6 Shield의 ESS 붕괴 지점 — 04편 §6과 연결
- [ ] §10 실습: **BoxBarrierCost를 따라 만들고 실행**
- [ ] 벤치마크 재현:
  ```bash
  PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario dynamic_rd2
  PYTHONPATH=. python examples/comparison/cbfkit_inspired_benchmark.py --scenario risk_sweep
  ```

**자가 점검**:
- "이 컨트롤러는 안전하다"는 주장을 들으면 어떤 질문 3가지를 해야 하는가?
  (힌트: 04편 §1.1)
- 5D 동역학 모델에서 위치 barrier가 왜 그대로는 안 되는지, HOCBF가 이를
  어떻게 푸는지 설명할 수 있는가?
- ρ=0.05와 ρ=0.5의 차이를 벤치마크 수치로 말할 수 있는가?

**더 깊이**: 수식 레퍼런스는 [SAFETY_THEORY.md](../SAFETY_THEORY.md)
(§1 기초, §16–20 cbfkit 5종, §21 선택 가이드), 벤치마크 전체 분석은
[CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md).

---

## Phase 4 — 변형/학습 심화 (1~2주)

**목표**: 43개 변형의 설계 공간을 패턴으로 이해하고, 학습 기반 제안 분포의
원리를 안다.

### 4-1. [07_CODE_WALKTHROUGH_VARIANTS.md](07_CODE_WALKTHROUGH_VARIANTS.md) (3~4일)

- [ ] §1 확장 패턴 지도 (A~F) — 43개를 6개 패턴으로 압축
- [ ] 관심 패턴 2개를 골라 대표 구현 정독 (추천: A의 Log + C의 LP)
- [ ] §8 새 변형 추가 체크리스트 — Phase 5 준비물

### 4-2. [05_GENERATIVE_MODELS_FOR_CONTROL.md](05_GENERATIVE_MODELS_FOR_CONTROL.md) (4~5일)

- [ ] §1: 왜 가우시안 제안으로는 부족한가 (importance weight 논리 포함)
- [ ] §3: DSM의 Vincent 트릭 유도
- [ ] §4: **CFM 핵심 챕터** — OT 경로와 conditional FM 정리
- [ ] §5: repo 횡단 설계 패턴 (zero-init, ring buffer, 붕괴 방어)
- [ ] 벤치마크 실행:
  ```bash
  PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --scenario obstacles
  PYTHONPATH=. python examples/comparison/score_guided_mppi_benchmark.py --all-scenarios
  ```

**자가 점검**: 제안 분포를 바꿔도 MPPI 가중치가 왜 여전히 유효한지 (Biased
정리), CFM이 diffusion보다 추론이 싼 이유를 설명할 수 있는가?

---

## Phase 5 — 연구 확장 (상시)

**목표**: 이 시리즈를 발판으로 논문/구현 연구로 나아간다.

- [ ] 각 문서 **부록 B (최근 연구 동향 2024–2026)** 를 훑고 흥미로운 주제 선정
- [ ] 부록 A의 원전 논문 중 1편을 골라 정독 — 시리즈 본문의 유도와 대조
- [ ] 자기 변형 하나를 설계 → 07편 §8 체크리스트대로 구현 → 전용 벤치마크로
  기존 변형과 비교. 비교 기준선으로는 `all_37_variants_benchmark.py`(항상 존재)
  또는 41변형 × 6모델 크로스 벤치마크 리포트
  [VARIANTS_X_MODELS_REPORT.md](../VARIANTS_X_MODELS_REPORT.md)
  (크로스 벤치마크 PR 머지 후 사용 가능)를 쓴다
- [ ] 외부 오픈소스 1개를 골라 이 repo와 결과 비교 (README.md "오픈소스 맵")

---

## 막혔을 때 찾아보는 곳

| 상황 | 바로가기 |
|---|---|
| 수식/유도가 이해 안 됨 | 해당 study 문서의 이전 절 + 부록 A 원전 논문 |
| 특정 변형의 수식이 궁금 | [MPPI_THEORY.md](../MPPI_THEORY.md) (변형 사전) |
| 특정 안전 기법의 수식 | [SAFETY_THEORY.md](../SAFETY_THEORY.md) (안전 사전) |
| 코드가 이론과 달라 보임 | 06/07/08 워크스루의 "quirk" 및 "흔한 실수" 절 |
| 실행이 안 되거나 이상함 | 각 부록 E FAQ 포인터, [../TUTORIALS.md](../TUTORIALS.md) |
| 어떤 기법을 골라야 할지 | SAFETY_THEORY §21 선택 가이드, VARIANTS_X_MODELS_REPORT.md Key Findings (크로스 벤치마크 PR) |
| 최신 흐름이 궁금 | README.md "최신 동향 한눈에" + 각 부록 B |

---

## 진행 기록

| Phase | 시작일 | 완료일 | 메모 |
|---|---|---|---|
| 0. 준비 | | | |
| 1. MPPI 기초 (01, 02) | | | |
| 2. 코드로 확인 (06 + 실험) | | | |
| 3. 안전 제어 (03, 04, 08) | | | |
| 4. 심화 (07, 05) | | | |
| 5. 연구 확장 | | | |

*작성: 2026-07 — learning_mppi 공부 자료 시리즈의 진입점. 문서 구성이
바뀌면 이 커리큘럼도 함께 갱신하세요.*
