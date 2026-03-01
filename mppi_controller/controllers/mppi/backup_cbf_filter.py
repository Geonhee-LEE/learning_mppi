"""
Backup CBF Safety Filter (#730)

표준 CBF는 1-step 안전만 보장. Backup CBF는 현재 제어 적용 후
백업 정책으로 롤아웃한 궤적 전체에 대해 안전 제약을 강제.

핵심:
  1. x_next = f(x, u)  →  백업 정책 롤아웃: x_b[0..H]
  2. 민감도 체인: dx_b[k]/du = ∂f/∂x|_{k-1} · dx_b[k-1]/du
  3. 다중 제약 QP:
     ∀k: ∂h/∂x|_{x_b[k]} · (dx_b[k]/du) · u + α_k·h(x_b[k]) ≥ 0
     α_k = α·γ^k  (감쇠)

Ref: Chen et al. (2021) "Backup Control Barrier Functions"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
from mppi_controller.controllers.mppi.backup_controller import (
    BackupController,
    BrakeBackupController,
)


class BackupCBFSafetyFilter:
    """
    Backup CBF Safety Filter

    백업 궤적 민감도 전파(sensitivity chain)를 통해 표준 CBF보다
    강한 안전 보장을 제공하는 안전 필터.

    Args:
        backup_controller: 백업 정책 (BrakeBackupController 등)
        model: RobotModel (step, forward_dynamics 메서드 필요)
        obstacles: [(x, y, radius), ...]
        dt: 시간 간격
        backup_horizon: 백업 궤적 길이 (timesteps)
        cbf_alpha: CBF class-K 파라미터
        safety_margin: 추가 안전 마진 (m)
        decay_rate: α 감쇠율 γ (α_k = α·γ^k)
    """

    def __init__(
        self,
        backup_controller: Optional[BackupController] = None,
        model=None,
        obstacles: Optional[List[tuple]] = None,
        dt: float = 0.05,
        backup_horizon: int = 15,
        cbf_alpha: float = 0.3,
        safety_margin: float = 0.1,
        decay_rate: float = 0.95,
    ):
        self.backup_controller = backup_controller or BrakeBackupController()
        self.model = model
        self.obstacles = obstacles or []
        self.dt = dt
        self.backup_horizon = backup_horizon
        self.cbf_alpha = cbf_alpha
        self.safety_margin = safety_margin
        self.decay_rate = decay_rate
        self.filter_stats = []

    def filter_control(
        self,
        state: np.ndarray,
        u_mppi: np.ndarray,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Backup CBF 안전 필터.

        Args:
            state: (nx,) 현재 상태 [x, y, θ, ...]
            u_mppi: (nu,) MPPI 제어 출력
            u_min: (nu,) 제어 하한
            u_max: (nu,) 제어 상한

        Returns:
            u_safe: (nu,) 안전 보정된 제어
            info: dict
        """
        if not self.obstacles or self.model is None:
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "backup_trajectory": None,
                "sensitivity_norms": [],
                "num_active_constraints": 0,
                "min_backup_barrier": float("inf"),
                "optimization_success": True,
            }
            self.filter_stats.append(info)
            return u_mppi.copy(), info

        # 1. 백업 궤적 롤아웃
        backup_traj = self._compute_backup_trajectory(state, u_mppi)

        # 2. 민감도 체인
        sensitivities = self._compute_sensitivity_chain(backup_traj, state, u_mppi)

        # 3. 모든 장애물 × 모든 시간스텝에 대해 제약 구성
        constraints = []
        barrier_values = []
        num_active = 0

        for obs in self.obstacles:
            obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
            effective_r = obs_r + self.safety_margin

            for k in range(self.backup_horizon + 1):
                x_bk = backup_traj[k]
                h_k = self._barrier(x_bk, obs_x, obs_y, effective_r)
                barrier_values.append(h_k)

                alpha_k = self.cbf_alpha * (self.decay_rate ** k)

                if k == 0:
                    # k=0: x_b[0] = f(state, u) → 민감도 = ∂f/∂u
                    dh_dx = self._barrier_gradient(x_bk, obs_x, obs_y)
                    J_u = self._dynamics_jacobian_control(state)

                    def cbf_con(u, dh=dh_dx, Ju=J_u, alpha=alpha_k, h=h_k):
                        return dh @ Ju @ u + alpha * h

                    constraints.append({"type": "ineq", "fun": cbf_con})
                else:
                    # k>0: 민감도 체인
                    dh_dx = self._barrier_gradient(x_bk, obs_x, obs_y)
                    S_k = sensitivities[k]  # (nx, nu)

                    def cbf_con(u, dh=dh_dx, Sk=S_k, alpha=alpha_k, h=h_k):
                        return dh @ Sk @ u + alpha * h

                    constraints.append({"type": "ineq", "fun": cbf_con})

                # 제약 위반 확인
                con_val = constraints[-1]["fun"](u_mppi)
                if con_val < 0:
                    num_active += 1

        min_barrier = min(barrier_values) if barrier_values else float("inf")

        # 빠른 경로: 모든 제약 만족
        if num_active == 0:
            sens_norms = [float(np.linalg.norm(s)) for s in sensitivities]
            info = {
                "filtered": False,
                "correction_norm": 0.0,
                "backup_trajectory": backup_traj,
                "sensitivity_norms": sens_norms,
                "num_active_constraints": 0,
                "min_backup_barrier": min_barrier,
                "optimization_success": True,
            }
            self.filter_stats.append(info)
            return u_mppi.copy(), info

        # 4. QP 풀기: min ||u - u_mppi||²
        def objective(u):
            d = u - u_mppi
            return 0.5 * np.dot(d, d)

        def objective_jac(u):
            return u - u_mppi

        bounds = None
        if u_min is not None and u_max is not None:
            bounds = list(zip(u_min, u_max))

        result = minimize(
            objective,
            x0=u_mppi.copy(),
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        u_safe = result.x
        correction_norm = float(np.linalg.norm(u_safe - u_mppi))
        sens_norms = [float(np.linalg.norm(s)) for s in sensitivities]

        info = {
            "filtered": True,
            "correction_norm": correction_norm,
            "backup_trajectory": backup_traj,
            "sensitivity_norms": sens_norms,
            "num_active_constraints": num_active,
            "min_backup_barrier": min_barrier,
            "optimization_success": result.success,
        }
        self.filter_stats.append(info)
        return u_safe, info

    def _compute_backup_trajectory(
        self, state: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        """
        백업 궤적: x_b[0] = f(state, u), x_b[k+1] = f(x_b[k], π_backup(x_b[k]))

        Returns:
            trajectory: (H+1, nx)
        """
        nx = len(state)
        traj = np.zeros((self.backup_horizon + 1, nx))
        traj[0] = self.model.step(state, u, self.dt)

        for k in range(self.backup_horizon):
            u_backup = self.backup_controller.compute_backup_control(
                traj[k], self.obstacles
            )
            traj[k + 1] = self.model.step(traj[k], u_backup, self.dt)

        return traj

    def _compute_sensitivity_chain(
        self, backup_traj: np.ndarray, state: np.ndarray, u: np.ndarray
    ) -> list:
        """
        민감도 체인 계산.

        dx_b[0]/du = ∂f/∂u|_{state,u} · dt  (Euler 근사)
        dx_b[k+1]/du = ∂f/∂x|_{x_b[k], u_b[k]} · dx_b[k]/du

        Returns:
            sensitivities: List[(nx, nu)] 길이 = backup_horizon + 1
        """
        nx = len(state)
        nu = len(u)
        sensitivities = []

        # k=0: ∂x_b[0]/∂u = ∂f/∂u at (state, u)
        J_u = self._dynamics_jacobian_control(state)
        S = self.dt * J_u  # (nx, nu) — Euler 근사
        sensitivities.append(S.copy())

        # k=1..H: 체인 룰
        for k in range(self.backup_horizon):
            u_backup = self.backup_controller.compute_backup_control(
                backup_traj[k], self.obstacles
            )
            J_x = self._dynamics_jacobian_state(backup_traj[k], u_backup)
            S = J_x @ S  # (nx, nu)
            sensitivities.append(S.copy())

        return sensitivities

    def _dynamics_jacobian_state(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        ∂f/∂x Jacobian (Differential Drive Kinematic).

        f(x,u) = x + dt·[v·cos(θ), v·sin(θ), ω]ᵀ

        ∂f/∂x = I + dt·[[0, 0, -v·sin(θ)],
                          [0, 0,  v·cos(θ)],
                          [0, 0,  0]]
        """
        nx = len(state)
        theta = state[2] if nx >= 3 else 0.0
        v = control[0]

        J = np.eye(nx)
        if nx >= 3:
            J[0, 2] = -self.dt * v * np.sin(theta)
            J[1, 2] = self.dt * v * np.cos(theta)

        return J

    def _dynamics_jacobian_control(self, state: np.ndarray) -> np.ndarray:
        """
        ∂f/∂u Jacobian (Differential Drive Kinematic).

        ∂f/∂u = dt·[[cos(θ), 0],
                      [sin(θ), 0],
                      [0,      1]]
        """
        nx = len(state)
        theta = state[2] if nx >= 3 else 0.0

        # nu 추론: model이 있으면 control_dim 사용, 없으면 diff-drive(2) 기본
        nu = getattr(self.model, 'control_dim', 2) if self.model else 2

        J = np.zeros((nx, nu))
        J[0, 0] = np.cos(theta)
        J[1, 0] = np.sin(theta)
        if nx >= 3 and nu >= 2:
            J[2, 1] = 1.0

        return self.dt * J

    def _barrier(
        self, state: np.ndarray, obs_x: float, obs_y: float, effective_r: float
    ) -> float:
        """h(x) = ||p - p_obs||² - r_eff²"""
        return (state[0] - obs_x) ** 2 + (state[1] - obs_y) ** 2 - effective_r ** 2

    def _barrier_gradient(
        self, state: np.ndarray, obs_x: float, obs_y: float
    ) -> np.ndarray:
        """∂h/∂x = [2(x-xo), 2(y-yo), 0, ...]"""
        nx = len(state)
        grad = np.zeros(nx)
        grad[0] = 2.0 * (state[0] - obs_x)
        grad[1] = 2.0 * (state[1] - obs_y)
        return grad

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.obstacles = obstacles

    def get_filter_statistics(self) -> Dict:
        """필터 통계"""
        if not self.filter_stats:
            return {
                "num_filtered": 0,
                "total_steps": 0,
                "filter_rate": 0.0,
                "mean_correction_norm": 0.0,
            }

        num_filtered = sum(1 for s in self.filter_stats if s["filtered"])
        corrections = [s["correction_norm"] for s in self.filter_stats]

        return {
            "num_filtered": num_filtered,
            "total_steps": len(self.filter_stats),
            "filter_rate": num_filtered / len(self.filter_stats),
            "mean_correction_norm": float(np.mean(corrections)),
            "max_correction_norm": float(np.max(corrections)),
        }

    def reset(self):
        """통계 초기화"""
        self.filter_stats = []

    def __repr__(self) -> str:
        return (
            f"BackupCBFSafetyFilter("
            f"obstacles={len(self.obstacles)}, "
            f"horizon={self.backup_horizon}, "
            f"α={self.cbf_alpha})"
        )
