"""
MPPI 간이 시뮬레이터

MPPI 컨트롤러 테스트 및 벤치마킹을 위한 시뮬레이터.
"""

import numpy as np
import time
from typing import Dict, Callable, Optional
from mppi_controller.models.base_model import RobotModel


class Simulator:
    """
    MPPI 컨트롤러 간이 시뮬레이터

    - step-by-step 실행
    - 메트릭 수집 (상태, 제어, 계산 시간 등)
    - 외란 주입 (process noise)

    Args:
        model: RobotModel 인스턴스
        controller: MPPI 컨트롤러 인스턴스
        dt: 타임스텝 간격 (초)
        process_noise_std: 외란 표준편차 (nx,) - None이면 외란 없음
    """

    def __init__(
        self,
        model: RobotModel,
        controller,
        dt: float,
        process_noise_std: Optional[np.ndarray] = None,
        online_learner=None,
        store_info: bool = True,
    ):
        self.model = model
        self.controller = controller
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.online_learner = online_learner
        self.store_info = store_info

        # 히스토리 초기화
        self.history = {
            "time": [],
            "state": [],
            "control": [],
            "reference": [],
            "solve_time": [],
            "info": [],
        }

        # 시뮬레이션 상태
        self.state = None
        self.t = 0.0

    def reset(self, initial_state: np.ndarray):
        """
        시뮬레이터 초기화

        Args:
            initial_state: (nx,) 초기 상태
        """
        self.state = initial_state.copy()
        self.t = 0.0
        self.history = {k: [] for k in self.history.keys()}

        # 컨트롤러 초기화
        self.controller.reset()

    def step(self, reference_trajectory: np.ndarray) -> Dict:
        """
        한 스텝 실행

        Args:
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            step_info: dict
                - state: (nx,) 현재 상태
                - control: (nu,) 제어 입력
                - solve_time: float 계산 시간 (초)
                - info: dict 컨트롤러 정보
        """
        # 1. MPPI 제어 계산
        t_start = time.time()
        control, info = self.controller.compute_control(self.state, reference_trajectory)
        solve_time = time.time() - t_start

        # 2. 상태 전파 (모델 사용)
        next_state = self.model.step(self.state, control, self.dt)

        # 3. 외란 주입 (있을 경우)
        if self.process_noise_std is not None:
            noise = np.random.normal(0.0, self.process_noise_std, self.model.state_dim)
            next_state += noise

        # 4. 상태 정규화 (각도 등)
        next_state = self.model.normalize_state(next_state)

        # 5. 온라인 학습 데이터 피드
        if self.online_learner is not None:
            self.online_learner.add_sample(
                self.state, control, next_state, self.dt
            )

        # 6. 히스토리 기록
        self.history["time"].append(self.t)
        self.history["state"].append(self.state.copy())
        self.history["control"].append(control.copy())
        self.history["reference"].append(reference_trajectory[0].copy())
        self.history["solve_time"].append(solve_time)
        if self.store_info:
            self.history["info"].append(info)

        # 7. 상태 업데이트
        self.state = next_state
        self.t += self.dt

        return {
            "state": self.state,
            "control": control,
            "solve_time": solve_time,
            "info": info,
        }

    def run(
        self,
        reference_trajectory_fn: Callable[[float], np.ndarray],
        duration: float,
        realtime: bool = False,
    ) -> Dict:
        """
        시뮬레이션 실행

        Args:
            reference_trajectory_fn: t → (N+1, nx) 레퍼런스 궤적 함수
            duration: 시뮬레이션 시간 (초)
            realtime: True면 실시간 속도로 실행

        Returns:
            history: dict - 전체 히스토리
        """
        num_steps = int(duration / self.dt)

        for i in range(num_steps):
            # 레퍼런스 궤적 생성
            ref_traj = reference_trajectory_fn(self.t)

            # 한 스텝 실행
            step_info = self.step(ref_traj)

            # 실시간 모드: 계산 시간 제외하고 대기
            if realtime:
                sleep_time = max(0.0, self.dt - step_info["solve_time"])
                time.sleep(sleep_time)

        return self.get_history()

    def get_history(self) -> Dict:
        """
        히스토리를 NumPy 배열로 변환

        Returns:
            history: dict
                - time: (T,)
                - state: (T, nx)
                - control: (T, nu)
                - reference: (T, nx)
                - solve_time: (T,)
                - info: List[dict]
        """
        history = {}
        for key, value in self.history.items():
            if len(value) > 0 and key != "info":
                history[key] = np.array(value)
            else:
                history[key] = value

        return history

    def __repr__(self) -> str:
        return (
            f"Simulator("
            f"model={self.model.__class__.__name__}, "
            f"controller={self.controller.__class__.__name__}, "
            f"dt={self.dt})"
        )
