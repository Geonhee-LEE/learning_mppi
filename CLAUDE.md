# MPPI ROS2 - Claude 개발 가이드

## 개발자 선호사항

- **언어**: 한국어 사용
- **자동 승인**: 코드 수정 자동 승인, 최종 변경 부분만 요약
- **GitHub 관리**: Issue, PR 자동 생성/관리

## 프로젝트 개요

MPPI (Model Predictive Path Integral) 기반 모바일 로봇 제어 시스템
- 15종 MPPI 변형, 22종 안전 제어, 13종 학습 모델
- 1100+ tests / 72 files / ~47,000+ lines

## 인터페이스 규칙

- **모든 컨트롤러**: `compute_control(state, reference_trajectory) -> (control, info)` 시그니처 준수
- **MPPI info dict**: sample_trajectories, sample_weights, best_trajectory, temperature, ess 등
- **서브클래스 오버라이드**: `_compute_weights()` 메서드

## 테스트

```bash
# 전체 테스트
python -m pytest tests/ -v --override-ini="addopts="

# 특정 테스트
python -m pytest tests/test_base_mppi.py::test_circle_tracking -v --override-ini="addopts="
```

### 성능 기준
- 위치 추적 RMSE: < 0.2m (원형 궤적)
- 계산 시간: < 100ms (K=1024, N=30)
- 실시간성: 10Hz 제어 주기 유지

## 데모 실행

> 전체 튜토리얼은 [docs/TUTORIALS.md](docs/TUTORIALS.md) 참조

```bash
# Vanilla MPPI
python examples/kinematic/mppi_differential_drive_kinematic_demo.py --trajectory circle --no-plot

# MPPI 변형 비교
python examples/mppi_all_variants_benchmark.py --live --trajectory figure8

# Flow-MPPI 벤치마크
PYTHONPATH=. python examples/comparison/flow_mppi_benchmark.py --live --scenario obstacles

# C2U-MPPI 벤치마크
PYTHONPATH=. python examples/comparison/c2u_mppi_benchmark.py --all-scenarios
```

## 데모 결과 출력 규칙

헤드리스 환경 + 핸드폰 원격 확인을 위해:

- **`plt.show()` 금지** → 반드시 파일 저장 (`matplotlib.use("Agg")`)
- **정적 이미지**: `plots/{name}.png` 저장
- **애니메이션** (`--live`): `plots/{name}.mp4` + `plots/{name}.gif` 저장
  - MP4: ffmpeg, 경량 (~500KB), 핸드폰 재생 최적
  - GIF: pillow, 범용 호환

```
핸드폰 확인:
  1. PC 웹서버: python -m http.server 8888
  2. 접속: http://<PC_IP>:8888/plots/result.mp4
```

## 커밋 및 PR 규칙

```
{type}: {subject}

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Types:** feat / fix / refactor / test / docs / perf

**PR**: 브랜치 `feature/{기능명}`, 제목 < 70자, 본문에 `## Summary` + `## Test plan`

## Claude 자동화 도구

### GitHub Issue Watcher
```bash
# 모바일 이슈 등록 + 'claude' 라벨 → 자동 구현 → PR 생성
systemctl --user start claude-watcher
systemctl --user status claude-watcher
```

### TODO Worker
```bash
claude-todo-worker          # 다음 작업 처리
claude-todo-task "#101"     # 특정 작업
claude-todo-all             # 전체 연속 처리
```

## 코드 품질

1. **정확성 우선** — 모르는 내용은 지어내지 않고, 추측 시 명시
2. **근거 기반** — 답변 전 근거 목록 작성, 신뢰성 자체 평가
3. **출처 명시** — 구체적 수치/인명 → 정확도 표시

## 라이센스

MIT License
