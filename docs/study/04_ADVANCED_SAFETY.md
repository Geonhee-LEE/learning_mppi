# 고급 안전 보장 공부 자료 — 벌점에서 증명까지, 안전의 스펙트럼

> **공부 자료 시리즈 4편.** 전편 [03_CBF_FUNDAMENTALS.md](03_CBF_FUNDAMENTALS.md)
> 에서 "한 스텝의 안전"을 CBF로 다뤘다면, 이 문서는 **보장의 강도**라는 축으로
> 안전 기법 전체를 재조직합니다: soft 벌점 → per-step 필터 → 궤적 단위 검증 →
> 무한 시간 보장 → 확률적 보장. 기법별 레퍼런스는
> [docs/SAFETY_THEORY.md](../SAFETY_THEORY.md)(§4 Shield, §10 Backup/Gatekeeper,
> §11 MPS, §12 CP, §14 Chance Constraint, §15 선택 가이드)를 참조하고, 여기서는
> 각 계열의 **핵심 아이디어, 보장의 정확한 범위, 대가(성능/계산), 실측 데이터**
> 를 공부합니다.
>
> **대상 독자**: CBF 기초(전편)를 이해한 로보틱스 엔지니어.

---

## 목차

1. [Safety의 3가지 층위 — 보장 강도의 스펙트럼](#1-safety의-3가지-층위--보장-강도의-스펙트럼)
2. [Gatekeeper & Backup Control — 무한 시간 안전 보장](#2-gatekeeper--backup-control--무한-시간-안전-보장)
3. [HJ Reachability와 DualGuard — 최악 케이스의 가치 함수](#3-hj-reachability와-dualguard--최악-케이스의-가치-함수)
4. [Chance Constraints — 확률적 안전의 세 가지 계보](#4-chance-constraints--확률적-안전의-세-가지-계보)
5. [Barrier States (DBaS) — 안전을 최적화 내부로](#5-barrier-states-dbas--안전을-최적화-내부로)
6. [Shielding 계열 — rollout 내부 강제의 득과 실](#6-shielding-계열--rollout-내부-강제의-득과-실)
7. [안전-성능 트레이드오프 정량화 — 벤치마크 데이터로 보기](#7-안전-성능-트레이드오프-정량화--벤치마크-데이터로-보기)
8. [연습문제](#8-연습문제)
9. [추천 자료](#9-추천-자료)

---

## 1. Safety의 3가지 층위 — 보장 강도의 스펙트럼

### 1.1 같은 "안전"이라도 약속이 다르다

"이 컨트롤러는 안전합니다"라는 문장은 세 가지 전혀 다른 것을 의미할 수
있습니다. 안전 기법을 공부할 때 가장 먼저 물어야 할 질문은 **"정확히 무엇을,
어떤 조건에서 약속하는가"** 입니다.

```
 보장 약함                                                   보장 강함
 ──────────────────────────────────────────────────────────────────→

 [층위 1: 비용 벌점]      [층위 2: per-step 필터]    [층위 3: 예측 검증]
  "위반이 비싸다"          "이번 스텝은 안전"          "이 궤적/미래 전체가 안전"
  soft, 통계적 경향        조건부 불변성               committed trajectory
```

**층위 1 — 비용 벌점 (soft, trajectory-wise 경향).**
안전 조건 위반량을 MPPI 비용에 더합니다. 샘플 가중치가 위반 궤적을 낮게
평가하므로 *평균적으로* 안전한 쪽으로 밀리지만, 추적 이득이 벌점을 이기는
순간 위반합니다. weight → ∞ 극한에서도 "샘플 중에 안전한 궤적이 있어야"
작동합니다. **보장 없음, 그러나 최적화를 왜곡하지 않아 성능 손실 최소.**

**층위 2 — 필터 (per-step, 조건부 보장).**
최종 제어 u 하나를 실행 직전에 검사/수정합니다 (CBF-QP 등). "모델이 맞고
QP가 feasible한 한, 다음 스텝에 안전 집합을 벗어나지 않는다"는 **한 스텝
불변성**을 줍니다. 그러나 필터는 근시안적입니다 — 매 스텝은 안전하지만
막다른 골목으로 걸어 들어가는 것은 막지 못합니다 (feasibility가 미래에
사라질 수 있음).

**층위 3 — 예측 검증 (trajectory-wise / 무한 시간).**
제어를 실행하기 *전에* "이 선택 이후에도 안전한 미래가 존재하는가"를 rollout
으로 검증하고, 검증 실패 시 이미 검증된 대안(백업)으로 전환합니다. 백업
궤적이 불변 집합에서 끝나면 **무한 시간 보장**까지 확장됩니다 (§2).

### 1.2 이 repo 기법 분류표

| 층위 | 보장 | repo 구현 (파일: `mppi_controller/controllers/mppi/`) |
|---|---|---|
| 1. 비용 벌점 | 없음 (경향) | `cbf_cost.py` (ControlBarrierCost), `hocbf_cost.py`, `c3bf_cost.py`, `dpcbf_cost.py`, `neural_cbf_cost.py`, `chance_constraint_cost.py`, `stochastic_cbf.py`, `robust_cbf_margin.py`, `dbas_mppi.py` (§5), `drpa_mppi.py` |
| 2. per-step 필터 | 1스텝 불변 (모델·feasibility 조건부) | `cbf_safety_filter.py`, `clf_cbf_qp.py`, `optimal_decay_cbf_filter.py`, `neural_cbf_filter.py`, `hocbf_cost.py`의 HOCBFFilter |
| 2.5 rollout 내부 강제 | 샘플 전체가 layer-2 보장 | `shield_mppi.py`, `adaptive_shield_mppi.py`, `shield_dial_mppi.py`, `shield_svg_mppi.py` (§6) |
| 3. 예측 검증 | 무한 시간 (백업 불변 집합 조건부) | `gatekeeper.py` + `backup_controller.py`, `mps_controller.py`, `backup_cbf_filter.py`, `contingency_mppi.py` (§2) |
| (횡단) 최악/확률 보장 | HJ 가치 / P(위반)≤ρ | `dualguard_mppi.py` (§3), `c2u_mppi.py`, `conformal_cbf_mppi.py`, `stochastic_cbf.py`의 RiskAwareCBFCost (§4) |

### 1.3 적층 구조 (defense in depth)

층위는 경쟁이 아니라 적층 관계입니다:

```
  레퍼런스 ──→ ┌────────────────────────────┐
              │ MPPI + CBF/HOCBF 비용 (층위1)│  좋은 궤적을 "제안"
              │  · 위반 궤적의 가중치 억제     │  (개입 아님, 항상 작동)
              └──────────────┬─────────────┘
                             │ u_mppi
              ┌──────────────▼─────────────┐
              │ CBF-QP 필터 (층위 2)         │  실행 직전 "교정"
              │  · 대부분의 스텝: 통과(투명)   │  (층위1 덕분에 드묾)
              └──────────────┬─────────────┘
                             │ u_safe
              ┌──────────────▼─────────────┐
              │ Gatekeeper (층위 3)          │  막다른 길 "거부"
              │  · 백업 미래 존재 검증        │  (최후의 방어선)
              └──────────────┬─────────────┘
                             ▼ 로봇
```

각 층의 구멍(벌점을 이기는 이득 / 필터의 근시안 / 게이트의 모델 오차)이
서로 다른 위치에 있어서, 겹치면 전체 구멍이 급격히 줄어듭니다 — 항공 안전의
"스위스 치즈 모델"과 같은 논리. 동시에 아래층이 좋을수록 위층의 개입
빈도가 줄어 보수성 비용(추적 손실)이 낮아집니다. 구체적 조합 레시피는
[SAFETY_THEORY.md §21](../SAFETY_THEORY.md) "안전 기법 선택 가이드" 절.

---

## 2. Gatekeeper & Backup Control — 무한 시간 안전 보장

### 2.1 문제: horizon 밖은 아무도 모른다

MPPI는 N스텝(예: 1.5 s)만 내다봅니다. horizon 내내 안전한 궤적이 horizon
직후 물리적으로 회피 불가능한 상태(막다른 골목, 과속 진입)로 끝날 수
있습니다. **유한 예측으로 무한 시간을 보장하려면 어떻게 해야 할까?**

### 2.2 핵심 아이디어: 안전한 미래를 하나만 확보하라

무한 시간 전체를 계획할 필요는 없습니다. **"언제든 실행 가능한, 검증된 탈출
계획 하나"** 만 항상 유지하면 됩니다.

> **Backup set 논증 (Gurriet et al. 2020).**
> 단순한 백업 정책 π_b (예: 정지)가 있고, 상태 x에서 π_b를 실행한 궤적이
> (i) 전 구간 안전하며 (ii) **불변 집합**(정지 상태 등 — 한번 들어가면 π_b로
> 계속 안전)에서 끝난다면, x는 "복구 가능(recoverable)" 상태다.
> 시스템이 복구 가능 상태만 유지하도록 감독하면, 무한 시간 안전이 보장된다:
> 최악의 경우 π_b를 실행하면 되므로.

diffdrive 기구학이 특히 유리한 이유: drift가 없어(전편 §4.1) **u = 0이면
그 자리에 정지**하고, 정지 상태는 정적 환경에서 자명한 불변 집합입니다.
백업 궤적 검증이 "감속 경로가 장애물과 겹치는가"라는 유한 계산으로
떨어집니다.

### 2.3 Gate open/close 상태 기계

```
                     매 스텝:
        u_mppi ──→ ① x_next = step(x, u_mppi)        (1스텝 예측)
                   ② backup_traj = π_b rollout from x_next
                   ③ backup_traj 전체 안전?
                        │
              ┌── YES ──┴── NO ──┐
              ▼                  ▼
        ┌──────────┐      ┌───────────┐
        │ GATE OPEN │      │ GATE CLOSED│
        │ u_mppi 실행│      │ u_backup 실행│  ← 현재 상태의 백업 제어
        └──────────┘      └───────────┘
              │                  │
              └── 다음 스텝에 다시 ①부터 ──┘
```

불변성 논증 (귀납): 지금 상태가 복구 가능하다고 하자.
- Gate open → u_mppi를 실행해도 다음 상태 x_next가 복구 가능함을 방금 검증했음.
- Gate closed → 백업을 실행. 백업 궤적은 (이전 스텝에서) 검증된 안전 경로 위를
  따라감.
어느 쪽이든 다음 상태도 복구 가능 → 영원히 유지. 형식적 증명은
[SAFETY_THEORY.md §10](../SAFETY_THEORY.md) "전방 불변성 증명 (Gatekeeper)" 절.

**committed trajectory 관점**: 원래 gatekeeper 문헌은 "후보 궤적(성능 계획의
앞부분) + 백업 궤적(탈출)"을 이어붙인 합성 궤적을 검증하고, 검증에 성공한
것만 **committed trajectory**로 승격해 실행합니다. 새 후보가 검증에 실패하면
직전에 commit된 궤적을 계속 따라갑니다 — "검증 없이는 절대 갈아타지 않는다"
는 원칙. 이 repo 구현은 후보를 1스텝(u_mppi)으로 짧게 잡은 최소 버전입니다.

### 2.4 코드 산책: gatekeeper.py

[gatekeeper.py](../../mppi_controller/controllers/mppi/gatekeeper.py)의
`Gatekeeper.filter(state, u_mppi)`가 위 상태 기계 그대로입니다:

```python
# gatekeeper.py — filter() 핵심 (발췌)
x_next = self.model.step(state, u_mppi, self.dt)                  # ①
backup_traj = self.backup_controller.generate_backup_trajectory(  # ②
    x_next, self.model, self.dt, self.backup_horizon, self.obstacles)
is_safe, min_barrier = self._check_trajectory_safety(backup_traj) # ③
if is_safe:
    return u_mppi.copy(), {"gate_open": True, ...}
else:
    u_backup = self.backup_controller.compute_backup_control(state, ...)
    return u_backup, {"gate_open": False, ...}
```

검증 기준은 궤적 전 지점에서 h = dist² − r_eff² > 0
(`_check_trajectory_safety`, 벡터화된 배치 계산). `get_statistics()`가
`gate_open_rate`를 리턴하므로 "게이트가 얼마나 자주 닫히는가"로 보수성을
모니터링할 수 있습니다.

**백업 정책 설계** —
[backup_controller.py](../../mppi_controller/controllers/mppi/backup_controller.py):

| 클래스 | 정책 | 적합 환경 |
|---|---|---|
| `BrakeBackupController` | u = [0, 0] 즉시 정지 | 정적 환경 (정지 = 불변 집합) |
| `TurnAndBrakeBackupController` | `turn_steps`(기본 5)동안 가장 가까운 장애물 **반대 방향** 회전(`turn_speed=0.5`) 후 정지 | 동적 환경 — 멈춘 자리가 위험할 수 있으므로 자세를 돌려놓음 |

백업 정책의 요건은 "최적"이 아니라 **검증 가능성**입니다: 단순해서 rollout이
싸고, 종단이 불변 집합이면 충분. 좋은 백업일수록(덜 보수적일수록) 게이트가
덜 닫혀 성능 손실이 줄어듭니다 — 설계 가이드라인은
[SAFETY_THEORY.md §10](../SAFETY_THEORY.md) "백업 정책 설계 가이드라인".

**변형들**:
- [mps_controller.py](../../mppi_controller/controllers/mppi/mps_controller.py)
  — 같은 패턴의 stateless 버전 (model을 매 호출 전달), 감속 궤적 기반.
- [backup_cbf_filter.py](../../mppi_controller/controllers/mppi/backup_cbf_filter.py)
  — 이진 gate 대신 백업 궤적 위 min h를 **미분 가능한 제약**으로 만들어
  QP로 부드럽게 개입 (backup CBF, 민감도 전파는
  [SAFETY_THEORY.md §10.1](../SAFETY_THEORY.md)).
- [contingency_mppi.py](../../mppi_controller/controllers/mppi/contingency_mppi.py)
  (C-MPPI, 35번째 변형) — 검증을 필터가 아니라 **계획 비용 안에** 넣은 버전:
  계획 궤적의 checkpoint마다 내부 MPPI로 비상 탈출 가능성을 평가해 벌점.
  "모든 계획 상태에서 탈출 가능"을 최적화가 스스로 선호하게 만듭니다. 대가는
  계산량 (full_run.log 벤치마크에서 solve ~674 ms — 중첩 MPPI의 비용).

### 2.5 미니 실습: 게이트 동작을 손으로 확인하기

repo에서 5분짜리 실험 (파이썬 REPL):

```python
import numpy as np
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic)
from mppi_controller.controllers.mppi.gatekeeper import Gatekeeper
from mppi_controller.controllers.mppi.backup_controller import (
    BrakeBackupController, TurnAndBrakeBackupController)

model = DifferentialDriveKinematic()
gk = Gatekeeper(BrakeBackupController(), model=model,
                obstacles=[(3.0, 0.0, 0.4)], safety_margin=0.1,
                backup_horizon=30, dt=0.05)

for x in [1.0, 2.0, 2.4, 2.45, 2.5]:
    u, info = gk.filter(np.array([x, 0.0, 0.0]), np.array([1.0, 0.0]))
    print(f"x={x:4.2f}  gate_open={info['gate_open']}  "
          f"min_h={info['backup_min_barrier']:+.3f}  u={u}")
```

관찰 포인트: (i) gate가 닫히는 x 위치가 `r + margin`과 dt·v로 설명되는가,
(ii) `TurnAndBrakeBackupController`로 바꾸면 닫히는 위치가 어떻게 변하는가,
(iii) `gk.get_statistics()["gate_open_rate"]`로 보수성을 수치화.

### 2.6 한계 (공부 포인트)

- 보장은 **모델과 장애물 정보가 맞을 때**의 이야기입니다. 백업 rollout도
  모델로 계산하므로, 모델 오차는 §4의 확률적 마진과 결합해야 합니다.
- 이진 게이트는 경계 근처에서 **채터링**(open/close 반복)할 수 있습니다 —
  히스테리시스나 backup CBF(연속 개입)로 완화.
- 백업이 지나치게 보수적이면(급정지) 승차감/후속 차량 문제. 백업의 품질이
  곧 시스템 성능의 하한입니다.

---

## 3. HJ Reachability와 DualGuard — 최악 케이스의 가치 함수

### 3.1 Value function으로 안전을 미리 계산하기

CBF의 h(x)는 사람이 설계합니다. Hamilton-Jacobi(HJ) reachability는 반대로
**동역학 자체로부터 "진짜 안전 여유"를 계산**합니다:

```
V(x) = "x에서 출발해, 최선의 제어로 (최악의 외란에 맞서) 플레이했을 때
        미래 전체에서 실패 집합에 얼마나 가까워지는가"

V(x) = max over u(·), min over d(·), min over t  g(x(t))
       (g = 실패 집합까지의 signed distance)
```

- V(x) ≥ 0: **어떤 제어 전략이 존재**해서 영원히 실패를 피할 수 있음.
- V(x) < 0: 무슨 짓을 해도(최악 외란 하) 언젠가 실패. 이 영역이
  **BRT(Backward Reachable Tube)** — 실패 집합으로 "빨려 들어가는" 관.

```
        ┌────────────────────────────────┐
        │   V > 0  (진짜 안전 영역)        │
        │      ┌────────────────┐        │
        │      │  BRT: V < 0    │        │   BRT는 장애물보다 큼:
        │      │   ┌────────┐   │        │   "아직 안 닿았지만
        │      │   │ 장애물  │   │        │    이미 늦은" 상태들
        │      │   └────────┘   │        │   (속도가 빠를수록 커짐)
        │      └────────────────┘        │
        └────────────────────────────────┘
```

핵심 통찰: **안전 경계는 장애물 경계가 아니다.** 빠르게 접근 중인 로봇은
장애물에서 1 m 떨어져 있어도 이미 BRT 안(회피 불능)일 수 있습니다. V는 이
동역학적 사실을 인코딩합니다. 또한 V의 등고선 자체가 (제약 조건 하에서) 가장
덜 보수적인 barrier이고, ∇V는 "가장 빨리 안전해지는 방향"을 줍니다.

**어떻게 계산하나 (개요만).** V는 Hamilton-Jacobi-Isaacs(HJI) 편미분방정식의
점성해(viscosity solution)로 특성화됩니다. 시간 역방향으로:

```
∂V/∂t + min{ 0,  max_u min_d  ∇V(x)ᵀ f(x, u, d) } = 0,   V(x, 0) = g(x)
```

읽는 법: 실패 집합까지의 거리 g에서 출발해, "제어는 V를 올리려 하고(max_u)
외란은 내리려 한다(min_d)"는 미분 게임을 뒤로 감으며 푸는 것. 바깥의
min{0, ·}는 "한 번이라도 실패하면 끝"(tube)을 인코딩합니다. 수렴하면
V(x) = lim_{t→∞} V(x, −t)가 무한 시간 안전 가치입니다. 격자 기반 솔버
(level-set method)가 표준이며, 이것이 §3.2 차원의 저주의 원인입니다.

CBF와의 관계를 한 줄로: **CBF의 h는 사람이 제안하고 조건으로 검증하는
barrier, HJ의 V는 동역학이 스스로 계산해 낸 "최대 안전 집합의 barrier"**
입니다. 손으로 만든 h가 실제로 CBF 조건을 만족하는지(=제어 여력이 충분한지)
확신이 없을 때, HJ는 그 질문 자체를 계산으로 대체합니다.

**Least-restrictive filtering**: V(x)가 충분히 안전하면(임계값 위) 성능
컨트롤러를 **전혀 건드리지 않고**, 경계에 닿을 때만 안전 최적 제어
u* = argmax ∇V·f(x,u)로 전환. "개입은 최후의 순간에, 그러나 그때는 최적으로"
— HJ 계열 필터의 표준 패턴입니다.

### 3.2 차원의 저주와 근사

V는 HJI 편미분방정식을 상태 공간 격자에서 풀어 얻는데, 비용이 **차원에
지수적**입니다 (격자 M점/축, n차원 → Mⁿ). 3–5차원이 실용 한계라서, 근사가
연구의 절반입니다: 분해(decomposition), 학습(DeepReach), 그리고 **해석적
프록시** — 이 repo의 선택입니다.

### 3.3 repo의 SafetyValueFunction: signed distance + TTC 근사

[dualguard_mppi.py](../../mppi_controller/controllers/mppi/dualguard_mppi.py)
(DualGuard-MPPI, 36번째 변형, Borquez et al. RA-L 2025 영감)의
`SafetyValueFunction`:

```python
# dualguard_mppi.py — 해석적 V(x)
V(x) = min_i ( ||pos − o_i|| − (r_i + margin) )     # signed distance
```

- `evaluate(states)`: 배치 (..., nx) → V 값. 순수 위치 기반이면 이것은 그냥
  안전 마진이지만—
- `evaluate_with_velocity(states, dt)`: 연속 상태에서 속도를 근사해
  **TTC(time-to-collision) 기반 벌점**을 더합니다. "장애물을 향해 빠르게
  이동 중"이면 같은 거리라도 V를 깎음 — BRT가 속도 방향으로 부풀어 오르는
  효과의 1차 근사입니다 (`ttc_horizon`이 얼마나 미리 보는지 결정).
- `gradient(states)`: ∇V (장애물 반대 방향) — hard 가드에서 탈출 방향으로 사용.

즉, "격자 HJ를 못 푸니 V ≈ signed distance + 접근 속도 보정"이라는 실용적
타협입니다. 진짜 HJ V와 달리 제어 한계·비선형 동역학을 무시하므로 보수성이
비균일하다는 한계를 알고 써야 합니다.

### 3.4 세 가지 가드 모드 (층위 스펙트럼의 축소판)

`DualGuardMPPIController`는 V를 세 강도로 활용합니다 — §1의 층위가 한
컨트롤러 안에 다 들어 있는 좋은 교보재입니다:

```
Soft:   cost_k += penalty · exp(−decay · V(x))     # 층위 1 — 벌점
Hard:   u_k += α · ∇V(x_t)                         # 샘플을 안전 방향으로 견인
Filter: w_k = 0  if any V(x_t) < 0                 # 위반 샘플 가중치 절멸
```

+ nominal guard(명목 시퀀스도 V로 검사), adaptive noise boost(안전 샘플
비율이 낮으면 σ 증폭 — 탐색으로 안전 모드 탈출). 벤치마크에서 DualGuard는
MinClearance 최상위군입니다: `results/variants_x_models/full_run.log`의
obstacles 시나리오에서 **0.827 m** (diffdrive kin) / **0.745 m** (swerve dyn)
— 같은 시나리오 Vanilla는 0.171/0.188 m. 대가는 추적 성능 (§7).

원 논문(DualGuard-MPPI)은 진짜 HJ 가치 함수로 rollout 중 + 최종 출력의 이중
필터링을 수행합니다 — repo 버전과의 차이를 의식하며 읽어 보세요.

---

## 4. Chance Constraints — 확률적 안전의 세 가지 계보

### 4.1 문제 설정

노이즈가 있으면 "h ≥ 0 항상"은 달성 불가능한 요구입니다 (가우시안 꼬리는
무한대). 현실적 목표는:

```
P( h(x_t) ≥ 0 ) ≥ 1 − δ        (chance constraint, 위험 예산 δ)
```

기술적 공통 패턴은 모두 같습니다 — **불확실성을 정량화해서 결정론적
마진으로 환산**:

```
h(x) ≥ margin(불확실성, δ)  를 강제  ⟹  P(h ≥ 0) ≥ 1 − δ
```

차이는 "불확실성을 **어떻게 추정**하고 margin을 **어떻게 유도**하는가"입니다.
이 repo에 세 계보가 모두 구현되어 있어 비교 학습에 좋습니다.

### 4.2 계보 1 — 모멘트 전파: C2U-MPPI (Unscented Transform)

[c2u_mppi.py](../../mppi_controller/controllers/mppi/c2u_mppi.py):
상태 분포의 **공분산 Σ를 horizon을 따라 전파**하고, 위치 불확실성만큼
장애물을 부풀립니다:

```
UT: σ-point 2n+1개 → 비선형 동역학 통과 → Σ_t 복원  (EKF 선형화보다 정확)
CC: P(collision) ≤ α  ⟸  r_eff(t) = r + κ_α · √(trace(Σ_pos(t)))
```

UT가 하는 일을 그림으로:

```
      σ-points (2n+1개)              비선형 동역학 F 통과 후
         ·  ·                              ·
       ·  ●  ·      ──── F ────→        ·   ●  ·        ● 평균
         ·  ·                          ·        ·       · σ-point
      (Σ의 제곱근                     (변형된 점들에서
       방향으로 배치)                   Σ_next를 가중 복원)
```

야코비안 없이 결정론적 점 2n+1개만 통과시켜 평균·공분산을 2차 정확도로
복원합니다 (EKF의 1차 선형화보다 정확, 파티클 필터보다 압도적으로 쌈).
κ_α는 신뢰 수준 배수 (가우시안 가정 시 quantile, 분포-무관이면 Chebyshev —
비교는 [SAFETY_THEORY.md §14](../SAFETY_THEORY.md)). 시간이 갈수록 Σ가
커지므로 **먼 미래일수록 장애물이 커지는** 원뿔형 마진이 생깁니다:

```
   t=0        t=N/2        t=N
   (⊘)       ( ⊘ )       (  ⊘  )     ← r_eff(t) 성장
    실제 장애물 r은 동일, 불확실성만 증가
```

- 장점: 동역학·노이즈 모델과 정합적, 마진이 상태 의존적 (직진 vs 회전 구간의
  불확실성 차이 반영).
- 약점: 가우시안(또는 2차 모멘트) 근사; 모델의 노이즈 스펙이 틀리면 보장도
  틀림. `propagation_mode="per_sample"`은 정확하지만 O(K·N) 비용.

### 4.3 계보 2 — 분포-무관 사후 보정: Conformal Prediction + CBF

[conformal_cbf_mppi.py](../../mppi_controller/controllers/mppi/conformal_cbf_mppi.py)
(+ `learning/conformal_predictor.py`): 노이즈 모델을 **아예 가정하지 않고**,
"내 예측기가 실제로 얼마나 틀리는가"를 온라인 잔차로 측정합니다:

```
매 스텝: 잔차 e_t = ‖실제 x_t − 예측 x̂_t‖ 기록
       → 잔차들의 (1−α)-quantile = CP 마진
       → Shield-MPPI의 safety_margin을 이 값으로 동적 갱신
보장:   P(실제 상태 ∈ 예측 ± 마진) ≥ 1 − α   (분포-무관, 유한 표본!)
```

전제는 잔차의 **교환 가능성(exchangeability)** 뿐입니다. 분포가 천천히
변하면 ACP(adaptive CP)가 α를 온라인 조정합니다. 모델이 정확해지면 마진이
저절로 줄어드는 것(불필요한 보수성 제거)이 실용적 매력입니다. 이론 유도와
유한 표본 증명 스케치는 [SAFETY_THEORY.md §12](../SAFETY_THEORY.md).

- 장점: 어떤 예측기(학습 모델 포함)에도 적용, 가정 최소.
- 약점: 보장은 "예측 오차 커버리지"에 대한 것 — 커버리지→무충돌로 옮기는
  단계(마진을 어디에 꽂는가)는 설계 몫. 분포 급변 시 적응 지연.

### 4.4 계보 3 — 경로 전체의 상한: Risk-Aware CBF (martingale)

[stochastic_cbf.py](../../mppi_controller/controllers/mppi/stochastic_cbf.py)의
`RiskAwareCBFCost` (Black CDC 2023): 앞의 둘이 **시점별** 확률을 다룬다면,
이것은 **경로 전체의 최소값**에 대한 상한을 줍니다 — 진짜로 원하는 것은
"horizon 중 한 번이라도 위반할 확률"이기 때문입니다:

```
목표:  P( min_{s≤t} h(x_s) < 0 ) ≤ ρ
```

유도의 뼈대: 노이즈가 h에 미치는 누적 효과 M_t = ∫∇h·σ dw 는
**마팅게일**(공정한 게임의 누적 이득 — 기대 증분 0인 확률 과정)이고,
마팅게일에는 reflection 부등식이 있습니다:

```
P( sup_{s≤t} |M_s| ≥ c ) ≤ 2(1 − Φ(c / √⟨M⟩_t)),    ⟨M⟩_t ≤ η²t
```

"경로 중 언젠가"라는 어려운 사건을 한 번의 가우시안 꼬리 계산으로 바꿔주는
도구입니다. 이를 뒤집으면 시간 의존 마진이 나옵니다:

```
h(x_t) ≥ margin(t) = √(2t) · η · erfinv(1 − 2ρ)
```

- √t 성장: 노이즈 누적이 확산(diffusion)이므로. C2U의 Σ(t) 성장과 같은 물리,
  다른 수학.
- ρ = 0.5 ⟹ erfinv(0) = 0 ⟹ 마진 0 (무방비 CBF로 퇴화), ρ → 0 ⟹ 마진 → ∞.
  **ρ 하나가 해석 가능한 위험 다이얼**이 됩니다 — §7에서 벤치마크 실증.
- η = ‖∇h·σ‖의 상한. repo 기본은 현재 샘플 배치 위의 max로 근사하므로 보장은
  근사적이고, `grad_bound` 지정 시 엄밀해지는 대신 보수적
  ([CBFKIT_INSPIRED_SAFETY.md §8](../CBFKIT_INSPIRED_SAFETY.md) 한계 논의).

### 4.5 세 계보 비교표

| | C2U (UT 전파) | CP + CBF | Risk-Aware CBF (martingale) |
|---|---|---|---|
| 불확실성 출처 | 노이즈 모델 (선험) | 실측 잔차 (사후) | 노이즈 모델 (선험) |
| 분포 가정 | 가우시안 근사 | 없음 (교환 가능성만) | Brownian (연속 확산) |
| 보장 대상 | 시점별 P(충돌)≤α | 예측 커버리지 ≥1−α | **경로 전체** min h |
| 마진 형태 | κ_α√tr(Σ_t), 상태 의존 | quantile, 데이터 의존 | √(2t)·η·erfinv(1−2ρ) |
| 모델 오차에 | 취약 (노이즈 스펙 필요) | 강함 (적응) | η 추정에 의존 |
| 적합 상황 | 노이즈 모델 신뢰 가능 | 학습 모델/미지 분포 | 경로 위험 예산 명시 필요 |

---

## 5. Barrier States (DBaS) — 안전을 최적화 내부로

### 5.1 관점 전환: 제약도 필터도 아닌 "상태"

지금까지 안전은 최적화 **바깥**(필터)이거나 **벌점 항**이었습니다. Barrier
state 접근(DBaS,
[dbas_mppi.py](../../mppi_controller/controllers/mppi/dbas_mppi.py),
arXiv:2502.14387)은 세 번째 길입니다: **안전도를 동역학을 가진 상태 변수로
승격**시켜 시스템에 붙입니다.

```
h(x) = dist² − r_eff²                     (제약 값)
B(h) = −log(max(h, h_min))                (log barrier: h→0⁺에서 +∞)
β_{k+1} = B(h(x_{k+1})) − γ(B(h(x_d)) − β_k)    (barrier state 동역학)

증강 시스템:  [x; β] — MPPI는 이 증강 상태의 궤적을 최적화
비용:        C_B = R_B · Σ_t max(β_t, 0)
```

왜 이게 다른가:

1. **내부점(interior point) 논리**: log barrier는 경계에서 발산하므로, β가
   유한한 궤적은 *자동으로* 안전합니다. 벌점의 "weight를 이겨버리는" 문제가
   구조적으로 완화됨 (경계 근처 비용이 다항이 아니라 로그 발산).
2. **γ 항 = barrier의 안정화**: β 동역학의 γ(B(h(x_d)) − β_k) 항은 레퍼런스
   지점의 barrier 수준으로 β를 끌어당기는 피드백 — 일시적으로 위험해져도 β가
   폭주하지 않고 회복하는 잔잔한 동역학을 만듭니다.
3. **적응적 탐색과의 결합** (repo 구현의 특징): best 궤적의 barrier 비용으로
   샘플링 노이즈를 스케일링합니다 —
   `Se = μ·log(e + C_B(best))`, `σ_eff = σ·(1 + Se)`.
   막힌 상황(barrier 비용 큼)일수록 넓게 탐색해 **가중치 퇴화(degeneracy)를
   방지**. §6에서 볼 Shield의 ESS 붕괴와 정반대 방향의 설계입니다.

### 5.2 위치 짚기

DBaS는 층위 1(비용)에 속하지만, "안전을 최적화 문제의 기하로 흡수한다"는
점에서 벌점과 필터의 중간 성격입니다. 원 계보는 Almubarak & Theodorou의
embedded barrier states (트래젝터리 최적화에서 barrier를 상태로 embedding해
제약 없는 문제로 변환). 약점: h → 0 근처에서 비용이 폭발하므로 노이즈가 크면
샘플 대부분이 무한대 비용 → ESS 관리가 필수 (그래서 repo가 적응 σ를 붙임),
그리고 hard 보장은 아니라는 점 (log barrier도 유한 샘플에선 뚫립니다 —
full_run.log에서 DBaS가 swerve_dyn obstacles에서 충돌 6회를 낸 기록 참조).

---

## 6. Shielding 계열 — rollout 내부 강제의 득과 실

### 6.1 아이디어: 애초에 안전한 샘플만 평가하라

층위 2 필터는 **최종 출력 1개**만 고칩니다. Shield-MPPI
([shield_mppi.py](../../mppi_controller/controllers/mppi/shield_mppi.py),
Yin et al. 계열)는 한발 더 들어가 **K개 샘플 rollout의 매 스텝마다** CBF
조건을 강제합니다:

```
Vanilla MPPI:  샘플 u_t^k → rollout → 비용 (위반 궤적도 평가에 참여)
Shield MPPI:   샘플 u_t^k → per-step CBF 사영 ũ_t^k → rollout
               (`_shielded_rollout` + `_cbf_shield_batch`, K개 벡터화 배치)
```

득: 최적화가 **실행 가능한(안전한) 분포 위에서만** 이루어짐 — 가중 평균이
"안전 궤적들의 평균"이라 출력도 구조적으로 안전 쪽. 특히 최종 필터처럼
계획-실행 불일치(필터가 계획에 없던 제어를 실행)가 없습니다. Shield vs
Filter의 정식 비교는 [SAFETY_THEORY.md §4](../SAFETY_THEORY.md) "Shield vs
Filter 비교" 절.

### 6.2 실: ESS 붕괴 — 실측 증거

대가는 **샘플 다양성의 파괴**입니다. 모든 샘플이 같은 CBF 사영으로 눌리면
샘플들이 서로 비슷해지고, 비용 분포가 왜곡되어 유효 샘플 수(ESS)가
무너집니다. 이 repo의 벤치마크가 이를 정량적으로 보여줍니다
([CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) §5.1, §6):

> static_kin 시나리오 (K = 512): Shield-MPPI **ESS ≈ 1.4** (512개 중!),
> RMSE 2.079 m (전 기법 중 최악) — 단, clearance는 0.637 m (MPPI 계열 최대).
> 같은 벤치마크의 Vanilla는 ESS 62, RMSE 0.872.

`results/variants_x_models/full_run.log`(obstacles, diffdrive kin)에서도 동일
패턴: Shield ESS 2.2 / RMSE 1.594 / MinClr 0.385 vs Vanilla ESS 19.8 /
RMSE 1.161 / MinClr 0.171. **ESS 한 자릿수 = 사실상 최선 샘플 1개의 복사**
— 샘플링 기반 최적화라는 MPPI의 정체성이 무력화된 상태입니다.

왜 붕괴하는가를 한 단계 더 들어가면:

```
 원래 샘플 분포              shield 사영 후
   u₁ ─────→ ·                u₁ ──┐
   u₂ ───→ ·                  u₂ ──┼──→ ●  (CBF 제약면 위의
   u₃ ──────→ ·               u₃ ──┘      거의 같은 점으로 수렴)
   (다양한 방향/크기)          (제약이 활성인 구간에서 전부 동일 사영)
```

장애물 근처에서 제약이 활성이 되는 순간, 서로 다른 샘플들이 **같은 제약면
위의 같은 점**으로 투영됩니다. 비용이 거의 같은 복제 샘플이 대량 발생 →
softmax 가중치가 몇 개 샘플에 집중 → ESS 붕괴. 게다가 사영은 rollout 중
매 스텝 반복되므로 궤적이 진행될수록 다양성이 기하급수로 죽습니다. 개입율과
추적 오차의 정량 관계는 [SAFETY_THEORY.md §4](../SAFETY_THEORY.md) "개입율 vs
추적 오차 분석" 절에 정리되어 있습니다 — 요지는 개입율이 임계치를 넘으면
추적 오차가 완만한 증가에서 급증으로 전환된다는 것 (탐색이 죽는 지점).

### 6.3 완화책들 (repo 구현 지도)

| 기법 | 아이디어 | 파일 |
|---|---|---|
| Adaptive Shield | α를 (거리, 속도)에 적응시켜 먼 곳에선 사영 약화 → 다양성 보존 | [adaptive_shield_mppi.py](../../mppi_controller/controllers/mppi/adaptive_shield_mppi.py) (`_adaptive_alpha_batch`) |
| Shield-DIAL | 반복 어닐링으로 분포를 안전 영역에 점진 수렴시킨 뒤 shield | [shield_dial_mppi.py](../../mppi_controller/controllers/mppi/shield_dial_mppi.py) ([SAFETY_THEORY.md §13](../SAFETY_THEORY.md)) |
| Shield-SVG | SVGD 파티클로 다양성을 명시적으로 유지하며 shield | [shield_svg_mppi.py](../../mppi_controller/controllers/mppi/shield_svg_mppi.py) |
| MPS/Gatekeeper | 아예 rollout 내부 개입을 포기하고 층위 3으로 이동 | §2 |

README의 14종 안전 벤치마크에서 **Adaptive Shield-MPPI가 100% 안전 +
RMSE 0.38 m**로 "안전 기법 중 최고 추적"을 기록한 것이 완화책의 효과를
보여줍니다 (repo [README.md](../../README.md) Safety Comparison 절).

> **공부 포인트 — 개입 지점의 삼각형**: 같은 CBF 조건이라도 *어디에*
> 꽂느냐가 성격을 결정합니다.
> 비용(soft, 다양성 보존, 보장 없음) ↔ rollout 내부(강한 경향, ESS 위험)
> ↔ 최종 출력(보장, 계획-실행 불일치). 셋 다 코드로 갖춘 repo는 좋은
> 실험실입니다: 동일 시나리오에서 `cbf_mppi.py` / `shield_mppi.py` /
> `cbf_safety_filter.py`를 바꿔 끼워 보세요.

---

## 7. 안전-성능 트레이드오프 정량화 — 벤치마크 데이터로 보기

안전은 공짜가 아닙니다. 이 절에서는 repo의 실측 데이터 두 세트로
트레이드오프를 정량적으로 읽는 법을 연습합니다.

### 7.0 지표 사전 (벤치마크 표 읽기 전에)

| 지표 | 정의 | 무엇의 대리인가 |
|---|---|---|
| RMSE | 위치 추적 오차의 제곱평균제곱근 | 성능 (임무 수행 품질) |
| MinClear | 전체 실행 중 장애물 표면까지 최소 거리 (음수 = 관통) | 안전 마진의 최악값 |
| Col | 충돌 스텝 수 | 안전의 이진 결과 |
| ESS | 유효 샘플 수 (softmax 가중치의 집중도 역수) | 최적화 건강도 — 낮으면 사실상 탐색 없음 |
| Solve (ms) | 스텝당 계산 시간 | 실시간성 (repo 기준 10 Hz → 100 ms 예산) |

함정 두 가지: (1) **MinClear는 min 통계**라 시드 하나의 운에 민감 — 반드시
다중 시드 평균±표준편차로 볼 것 (§7.2가 그 형식). (2) RMSE가 *너무* 낮은데
MinClear가 음수인 행(레퍼런스가 장애물을 관통하는 시나리오에서 레퍼런스를
충실히 따라간 경우)은 "추적을 잘해서 충돌한" 것이므로, 두 지표는 항상 쌍으로
읽어야 합니다. full_run.log에는 실제로 그런 행들이 있습니다 (예: RMSE 0.03에
MinClr −0.34, 37회 충돌 — 회피를 전혀 안 한 것).

### 7.1 데이터 1: 안전 계열 vs 비안전 계열 clearance
(`results/variants_x_models/full_run.log`, obstacles 시나리오)

41종 변형 × 여러 모델의 통합 벤치마크에서 발췌 (diffdrive kinematic,
obstacles):

| 변형 (그룹) | RMSE (m) | MinClear (m) | ESS | Solve (ms) |
|---|---|---|---|---|
| Vanilla (A) | 1.161 | 0.171 | 19.8 | 1.9 |
| DRPA (E) | **0.490** | 0.195 | 90.1 | 2.4 |
| CBF (E) | 0.950 | 0.174 | 41.5 | 2.0 |
| C2U (E) | 1.256 | 0.373 | 4.2 | 4.7 |
| Shield (E) | 1.594 | 0.385 | 2.2 | 2.9 |
| DualGuard (E) | 1.406 | **0.827** | 2.2 | 3.8 |
| Contingency (E) | 0.413 | 0.181 | 26.0 | **673.6** |

읽는 법 (연습):

- **보장 강도 ∝ clearance ∝ 성능 비용**: 층위가 올라갈수록(CBF 비용 →
  UT 마진 → rollout 강제 → HJ 가드) clearance가 0.17 → 0.37 → 0.39 → 0.83으로
  계단식 증가하고, RMSE/ESS/solve time 중 무언가가 대가로 지불됩니다.
- **같은 그룹 안에서도 지불 수단이 다릅니다**: DualGuard는 ESS로(2.2),
  Contingency는 계산 시간으로(673 ms — 중첩 MPPI), C2U는 둘 다 조금씩.
  DRPA는 clearance 대신 **liveness**(local minima 탈출)에 투자해 RMSE가
  오히려 최선 — "안전 그룹 = 느리고 부정확"이 아니라는 반례.
- 동역학 모델(swerve_dyn)에서도 서열 유지: DualGuard 0.745, C2U 0.709 vs
  추적 우선 변형들 ~0.19–0.21 m. **기법 서열이 모델을 가로질러 재현**되는지
  확인하는 것이 벤치마크 읽기의 기본기입니다.

### 7.2 데이터 2: 위험 다이얼 ρ — 트레이드오프를 "선택"으로 만들기
([docs/CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md) §5.3–5.4, 3 seeds)

프로세스 노이즈 시나리오의 핵심 대비:

| 방법 | RMSE (m) | MinClear (m) | 충돌 |
|---|---|---|---|
| 순정 CBF-MPPI | 0.383±0.058 | **−0.073±0.062** | **3/3 시드 충돌** |
| RobustCBF | 0.705±0.066 | 0.078±0.039 | 0 |
| RiskAwareCBF (ρ=0.1) | 1.997±0.092 | 0.243±0.016 | 0 |
| Shield | 2.239±0.294 | 0.608±0.054 | 0 |

그리고 RiskAware의 ρ 스윕:

```
   ρ:        0.5      0.2      0.1      0.05     0.01
   MinClear: −0.068 → 0.115 → 0.243 → 0.395 → 0.588  (단조!)
   RMSE:      0.599    1.569    1.997    1.621    1.746
```

읽는 법 (연습):

- **soft 벌점의 보장 없음이 데이터로 확인됩니다**: 노이즈 하에서 순정
  CBF-MPPI는 clearance −0.073 m로 *매 시드* 충돌 — barrier의 형태가 아니라
  **명시적 불확실성 마진**이 확률적 안전을 삽니다 (§4의 존재 이유).
- **ρ가 단조 다이얼임이 실증됩니다**: clearance가 ρ 전 구간에서 단조 —
  이론 마진 √(2t)·η·erfinv(1−2ρ)의 예측과 정합. 트레이드오프가 "튜닝 감"이
  아니라 **요구 사양(P(위반) ≤ ρ) → 파라미터**의 번역이 된다는 것이 확률적
  방법의 존재 가치입니다.
- **추적 비용은 비선형입니다**: RMSE의 대부분이 ρ = 0.5 → 0.1 구간에서
  지불되고 그 이후는 완만 — "적당한 안전"의 한계 비용이 가장 비싸고, 이미
  보수적인 영역에서 더 보수적으로 가는 건 쌉니다. 운영점 선택 시 무릎(knee)
  지점을 찾으세요.
- 참고로 같은 벤치마크에서 RobustCBF(유계 외란 가정)는 RMSE 0.705로 무충돌 —
  **분포 꼬리까지 막을 필요가 없다면 최악 케이스 마진이 더 싼 선택**일 수
  있습니다. 보장의 종류(확률 ρ vs 유계 w_max)와 가격을 함께 비교하는 습관.

### 7.3 요약: 트레이드오프를 다루는 세 가지 태도

1. **측정하라**: clearance만 보지 말고 (RMSE, ESS, solve time, 개입율/
   gate_closed_rate)를 항상 같이. 하나만 좋아지는 기법은 없습니다.
2. **다이얼로 만들라**: ρ, α, margin처럼 보장 수준과 단조로 연결된 손잡이를
   노출하는 기법이 운영에서 이깁니다.
3. **적층하라**: 싼 층(비용 벌점)으로 개입 빈도를 줄이고, 비싼 층(게이트/
   필터)은 드물게 발동하도록 — 평균 비용은 낮고 최악 보장은 유지.

---

## 8. 연습문제

**문제 1 — 층위 판별.**
다음 각 주장에 대해, 층위(1/2/3)와 보장의 조건을 명시하라.
(a) "CBF 비용 weight를 10⁶으로 올렸으니 안전합니다."
(b) "CBF-QP 필터가 있으니 안전합니다."
(c) "게이트키퍼가 있으니 안전합니다."
*답 골자: (a) 층위 1 — 보장 없음. 샘플 중 안전 궤적이 없거나 노이즈가 크면
위반 (§7.2의 −0.073 실증). (b) 층위 2 — 모델 정확 + QP feasible + 이산화
마진 충분일 때 1스텝 불변. 막다른 골목(미래 infeasibility)은 못 막음.
(c) 층위 3 — 백업 궤적 검증이 정확(모델·장애물)하고 백업 종단이 불변
집합일 때 무한 시간. 동적 장애물 예측 오차는 별도 마진 필요.*

**문제 2 — Gatekeeper 손 시뮬레이션.**
diffdrive 로봇이 x축을 따라 v = 1.0 m/s로 이동, dt = 0.1, 장애물 (3, 0),
r_eff = 0.5. 백업은 즉시 정지(BrakeBackup: 기구학이므로 1스텝 만에 정지),
backup_horizon은 충분히 김. 로봇이 x = 2.3에 있을 때 u_mppi = [1.0, 0]이
게이트를 통과하는가? x = 2.6에서는? (backup 궤적이 정지 상태 유지임을 이용)
*답: x=2.3 → x_next = 2.4, 백업 궤적은 (2.4,0)에 정지 유지. h = (3−2.4)² −
0.25 = 0.11 > 0 → gate open. x=2.6 → x_next = 2.7, h = 0.09 − 0.25 < 0 →
gate closed, 정지. 즉 이 설정의 실효 정지선은 x_next = 2.5 (dist = r_eff).
기구학+즉시정지 백업이라 정지선이 경계와 일치함 — 관성 있는 동역학이면
제동 거리만큼 앞당겨짐을 논하라.*

**문제 3 — BRT 직관.**
질량점 로봇 ẍ = u, |u| ≤ u_max = 1, 벽이 x = 0 (x > 0이 안전). 상태 (x, v)
에서 v < 0(벽으로 접근)일 때, 최대 제동으로도 충돌을 못 피하는 조건(BRT)을
유도하라.
*답: 제동 거리 v²/(2u_max) > x, 즉 BRT = {(x,v): v < 0, x < v²/2}. V(x,v) =
x − v²/2 (v<0) 꼴의 가치 함수가 나오며, 위치만 보는 h = x와의 차이가 정확히
"속도로 부풀어 오른" 부분. DualGuard의 TTC 보정(`evaluate_with_velocity`)이
이것의 1차 근사임을 설명하라.*

**문제 4 — 마진 계보 비교.**
같은 로봇에 대해 (a) C2U가 √trace(Σ_pos) = 0.1 m, κ_α = 2를 주고, (b)
RiskAware가 η = 0.14, ρ = 0.05, t = 1.0 s를 준다. 각 마진을 계산하고, 두
마진이 "서로 다른 질문에 대한 답"인 이유를 한 문장으로 써라.
*답: (a) 0.2 m — 시점 t에서의 시점별 충돌 확률 제어. (b) √2·0.14·erfinv(0.9)
≈ 0.198·1.163 ≈ 0.23 m — 경로 전체 [0,t]에서 한 번이라도 위반할 확률 제어.
수치가 비슷해도 후자가 더 강한 사건(sup over time)에 대한 보장.*

**문제 5 — 벤치마크 읽기.**
§7.1 표에서 Contingency는 clearance 0.181로 CBF(0.174)와 비슷하지만 solve가
674 ms다. "그러면 Contingency는 쓸모없다"는 주장을 반박하라 (힌트: 두 기법이
*무엇을* 보장/개선하려는지, 그리고 이 시나리오가 그것을 측정하는지).
*답 골자: Contingency의 가치는 평균 clearance가 아니라 "모든 계획 상태에서
탈출 계획 존재"라는 검증 — 정적 obstacles 시나리오는 그 가치가 드러나지 않는
과제다 (돌발 차단, 동적 장애물 시나리오여야 차이가 남). 벤치마크는 기법의
목적 함수와 정합할 때만 유효한 증거이며, RMSE 0.413(표 내 최상위권)이라는
부수 효과도 주목할 것. 또한 674 ms는 내부 MPPI 파라미터로 조절 가능한
엔지니어링 비용.*

---

## 9. 추천 자료

**계열별 원전:**

1. **Gurriet, Mote, Singletary, Nilsson, Feron, Ames, "A Scalable Safety
   Critical Control Framework for Nonlinear Systems" / "Scalable
   Safety-Critical Control of Robotic Systems" (2018–2020).**
   — backup set / implicit invariant set 이론 (§2의 기반).
2. **Naveed, Agrawal, Vermillion, Panagou 계열의 gatekeeper 논문
   (arXiv:2211.14361)** — committed trajectory 상태 기계의 형식화.
3. **Bansal, Chen, Herbert, Tomlin, "Hamilton-Jacobi Reachability: A Brief
   Overview and Recent Advances", CDC 2017.** — §3 HJ/BRT의 최고 입문.
   후속으로 Herbert의 FaSTrack, DeepReach(학습 근사)를 보면 차원의 저주
   대응 계보가 보입니다.
4. **Borquez, Chakraborty, Wang, Bansal, "On Safety and Liveness Filtering
   Using Hamilton-Jacobi Reachability Analysis" 및 DualGuard-MPPI
   (arXiv:2502.01924, RA-L 2025).** — least-restrictive filtering과 §3.4.
5. **Black, Fainekos, Hoxha, Prokhorov, Panagou 계열, "Risk-Aware Path
   Integral..." (CDC 2023 계열)** — §4.4 martingale 마진.
6. **Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and
   Distribution-Free Uncertainty Quantification" (2021).** — CP 입문 표준.
7. **Almubarak, Theodorou et al., embedded barrier states 계열 + DBaS-MPPI
   (arXiv:2502.14387).** — §5.
8. **Yin, Zhang, Tsiotras 계열, "Shield-MPPI" / Trust Region 기반 안전 MPPI
   (2023–).** — §6 rollout 내부 강제.
9. **Wabersich, Zeilinger, "Predictive Safety Filter" (Automatica 2021) 및
   Hsu, Hu, Fisac, "The Safety Filter: A Unified View of Safety-Critical
   Control" (Annual Reviews in Control 2023).** — 이 문서 전체(층위 스펙트럼)
   를 조망하는 서베이. **§1을 읽은 뒤 가장 먼저 읽기를 권합니다.**

**이 repo 내부 (실습 경로):**

- 이론 레퍼런스: [docs/SAFETY_THEORY.md](../SAFETY_THEORY.md) §4–§14 (기존 기법),
  §16–§20 (cbfkit-inspired 5종), §21 (의사결정 트리와 비교 매트릭스).
- 벤치마크 재현:
  ```bash
  # DualGuard (HJ 가드 3모드)
  PYTHONPATH=. python examples/comparison/dualguard_mppi_benchmark.py --all-scenarios
  # C2U (UT + chance constraint)
  PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py --all-scenarios
  # DBaS
  PYTHONPATH=. python examples/comparison/dbas_mppi_benchmark.py --all-scenarios
  # 안전 그룹 포함 전체 비교 (§7.1 데이터의 출처 스크립트)
  PYTHONPATH=. python examples/comparison/all_37_variants_benchmark.py --scenario obstacles
  ```
- CBF 계열 확률 마진의 실측: [docs/CBFKIT_INSPIRED_SAFETY.md](../CBFKIT_INSPIRED_SAFETY.md)
  (§7.2 데이터의 출처 문서).
- 전편: [docs/study/03_CBF_FUNDAMENTALS.md](03_CBF_FUNDAMENTALS.md).

---

*작성: 2026-07 — learning_mppi 공부 자료 시리즈. 인용한 벤치마크 수치는
`docs/CBFKIT_INSPIRED_SAFETY.md`와 `results/variants_x_models/full_run.log`의
기록 기준이며, 재실행 시 시드에 따라 달라질 수 있습니다.*
