"""
Model Predictive Shield (MPS)

간소화된 Gatekeeper 패턴의 안전 쉴드.
매 스텝 1-step 명목 제어 적용 후 백업 궤적 안전성을 검증.

기존 Gatekeeper와의 차이:
  - Gatekeeper: model을 인스턴스 변수로 저장, filter(state, u_mppi)
  - MPS: model을 매 호출 시 전달, shield(state, nominal_control, model) — stateless

핵심 로직:
  1. nominal_control 1-step 적용 → next_state
  2. next_state에서 backup 궤적 생성
  3. backup 궤적 전체가 안전하면 nominal 반환
  4. 아니면 현재 state에서 backup 첫 제어 반환
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from mppi_controller.controllers.mppi.backup_controller import (
    BackupController,
    BrakeBackupController,
)


class MPSController:
    """
    Model Predictive Shield

    Args:
        backup_controller: BackupController (default: BrakeBackupController)
        obstacles: [(x, y, radius), ...]
        safety_margin: 추가 안전 마진 (m)
        backup_horizon: 백업 궤적 길이 (timesteps)
        dt: 시간 간격
    """

    def __init__(
        self,
        backup_controller: Optional[BackupController] = None,
        obstacles: List[tuple] = None,
        safety_margin: float = 0.15,
        backup_horizon: int = 20,
        dt: float = 0.05,
    ):
        self.backup_controller = backup_controller or BrakeBackupController()
        self.obstacles = obstacles or []
        self.safety_margin = safety_margin
        self.backup_horizon = backup_horizon
        self.dt = dt
        self.stats = []

    def shield(
        self,
        state: np.ndarray,
        nominal_control: np.ndarray,
        model,
    ) -> Tuple[np.ndarray, Dict]:
        """
        MPS 안전 쉴드

        Args:
            state: (nx,) 현재 상태
            nominal_control: (nu,) 명목 제어 (예: MPPI 출력)
            model: RobotModel

        Returns:
            safe_control: (nu,) 안전한 제어
            info: dict - MPS 정보
        """
        if not self.obstacles:
            info = {
                "shielded": False,
                "reason": "no_obstacles",
                "backup_min_barrier": float("inf"),
            }
            self.stats.append(info)
            return nominal_control.copy(), info

        # 1. nominal control 1-step 시뮬레이션
        next_state = model.step(state, nominal_control, self.dt)

        # 2. next_state에서 backup 궤적 생성
        backup_traj = self.backup_controller.generate_backup_trajectory(
            next_state, model, self.dt, self.backup_horizon, self.obstacles
        )

        # 3. backup 궤적 안전성 검증
        is_safe, min_barrier = self._check_trajectory_safety(backup_traj)

        if is_safe:
            info = {
                "shielded": False,
                "reason": "backup_safe",
                "backup_min_barrier": min_barrier,
            }
            self.stats.append(info)
            return nominal_control.copy(), info
        else:
            # 현재 state에서 backup 제어 반환
            u_backup = self.backup_controller.compute_backup_control(
                state, self.obstacles
            )
            info = {
                "shielded": True,
                "reason": "backup_unsafe",
                "backup_min_barrier": min_barrier,
                "u_nominal": nominal_control.copy(),
                "u_backup": u_backup.copy(),
            }
            self.stats.append(info)
            return u_backup, info

    def _check_trajectory_safety(
        self, trajectory: np.ndarray
    ) -> Tuple[bool, float]:
        """
        궤적 안전성 검증: h(x) = dist² - r_eff² > 0 for all states

        Args:
            trajectory: (horizon+1, nx) 궤적

        Returns:
            is_safe: 전체 궤적이 안전한지 여부
            min_barrier: 최소 barrier 값 (음수면 충돌)
        """
        positions = trajectory[:, :2]
        min_barrier = float("inf")

        for obs_x, obs_y, obs_r in self.obstacles:
            effective_r = obs_r + self.safety_margin
            dx = positions[:, 0] - obs_x
            dy = positions[:, 1] - obs_y
            dist_sq = dx**2 + dy**2
            h = dist_sq - effective_r**2
            min_h = float(np.min(h))
            min_barrier = min(min_barrier, min_h)

        return min_barrier > 0, min_barrier

    def update_obstacles(self, obstacles: List[tuple]):
        """장애물 목록 업데이트"""
        self.obstacles = obstacles

    def get_statistics(self) -> Dict:
        """쉴드 통계 반환"""
        if not self.stats:
            return {
                "total_steps": 0,
                "shield_rate": 0.0,
                "shielded_count": 0,
            }
        total = len(self.stats)
        shielded = sum(1 for s in self.stats if s.get("shielded", False))
        return {
            "total_steps": total,
            "shield_rate": shielded / total,
            "shielded_count": shielded,
            "pass_rate": (total - shielded) / total,
        }

    def reset(self):
        """통계 초기화"""
        self.stats = []

    def __repr__(self) -> str:
        return (
            f"MPSController(backup={self.backup_controller.__class__.__name__}, "
            f"obstacles={len(self.obstacles)}, horizon={self.backup_horizon})"
        )
