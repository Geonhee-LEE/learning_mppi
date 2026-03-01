"""
Shield-DIAL-MPPI Controller

DIAL-MPPI의 다단계 확산 어닐링 + Shield-MPPI의 per-step CBF enforcement 결합.

핵심: 어닐링 루프의 매 반복에서 rollout → shielded_rollout 교체,
전체 교체 업데이트에 shielded_controls 사용 → U가 항상 안전 공간에 머무름.

상속 체인:
    MPPIController → DIALMPPIController → ShieldDIALMPPIController
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import ShieldDIALMPPIParams
from mppi_controller.controllers.mppi.dial_mppi import DIALMPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class ShieldDIALMPPIController(DIALMPPIController):
    """
    Shield-DIAL-MPPI Controller

    DIAL-MPPI의 확산 어닐링 루프 내부에 per-step CBF shield를 삽입하여
    모든 반복에서 안전한 제어 시퀀스를 유지.

    Vanilla DIAL-MPPI 대비 핵심 차이:
        1. rollout → shielded_rollout: 매 timestep CBF 제약 적용
        2. U 업데이트에 shielded_controls 사용: 안전 제어의 볼록 결합
        3. Shield 통계 누적: n_iters 반복 전체에 걸쳐 intervention_rate 추적

    Args:
        model: RobotModel 인스턴스
        params: ShieldDIALMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: ShieldDIALMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.shield_dial_params = params
        self._shield_enabled = params.shield_enabled
        self._shield_alpha = params.shield_cbf_alpha

        # Shield 통계
        self.shield_stats_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Shield-DIAL-MPPI 제어 계산

        Shield 비활성화 시 DIAL-MPPI로 폴백.

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - DIAL 통계 + Shield 통계
        """
        if not self._shield_enabled:
            return super().compute_control(state, reference_trajectory)

        K = self.params.K
        N = self.params.N
        nu = self.model.control_dim

        # Cold start vs Warm start
        if self._is_first_call:
            n_iters = self.dial_params.n_diffuse_init
            self._is_first_call = False
        else:
            n_iters = self.dial_params.n_diffuse

        iteration_costs = []

        # 마지막 반복의 정보를 저장할 변수
        last_trajectories = None
        last_weights = None
        last_costs = None

        # Shield 통계 누적 (전체 반복에 걸쳐)
        total_interventions = 0
        total_vel_reduction = 0.0
        total_steps = 0

        for i in range(n_iters):
            # 1. 어닐링된 노이즈 스케일 계산
            traj_scale = self.dial_params.traj_diffuse_factor ** i
            annealed_sigma = (
                self._horizon_profile[:, None] * self.params.sigma[None, :] * traj_scale
            )

            # 2. 샘플링: W ~ N(0, annealed_sigma)
            rng_noise = np.random.standard_normal((K, N, nu))
            W = rng_noise * annealed_sigma[None, :, :]  # (K, N, nu)

            # 3. 샘플 제어 생성 + 클리핑
            sampled_controls = self.U[None, :, :] + W  # (K, N, nu)
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

            # 4. Shielded rollout (DIAL과의 핵심 차이)
            trajectories, shielded_controls, shield_info = (
                self._shielded_rollout(state, sampled_controls)
            )

            # Shield 통계 누적
            total_interventions += shield_info["total_interventions"]
            total_vel_reduction += (
                shield_info["mean_vel_reduction"] * max(shield_info["total_interventions"], 1)
            )
            total_steps += shield_info["total_steps"]

            # 5. 비용 계산 (shielded_controls 기반)
            costs = self.cost_function.compute_cost(
                trajectories, shielded_controls, reference_trajectory
            )

            # 6. 가중치 계산
            if self.dial_params.use_reward_normalization:
                weights = self._compute_weights_normalized(costs)
            else:
                weights = self._compute_weights(costs, self.params.lambda_)

            # 7. 전체 교체 업데이트: U = Σ w_k * shielded_controls_k
            self.U = np.sum(
                weights[:, None, None] * shielded_controls, axis=0
            )  # (N, nu)

            # 제어 제약 클리핑
            if self.u_min is not None and self.u_max is not None:
                self.U = np.clip(self.U, self.u_min, self.u_max)

            # 반복별 비용 기록
            iteration_costs.append(float(np.min(costs)))

            # 마지막 반복 데이터 저장
            last_trajectories = trajectories
            last_weights = weights
            last_costs = costs

        # 첫 제어 추출 → shift → 반환
        optimal_control = self.U[0].copy()

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 통계 저장
        self._iteration_costs = iteration_costs
        ess = self._compute_ess(last_weights)
        best_idx = np.argmin(last_costs)

        dial_stats = {
            "n_iters": n_iters,
            "iteration_costs": iteration_costs,
            "cost_improvement": iteration_costs[0] - iteration_costs[-1] if len(iteration_costs) > 1 else 0.0,
        }
        self._dial_stats_history.append(dial_stats)

        # Shield 통계 (전체 반복 누적)
        shield_stats = {
            "intervention_rate": float(total_interventions / total_steps) if total_steps > 0 else 0.0,
            "mean_vel_reduction": float(total_vel_reduction / max(total_interventions, 1)),
            "total_interventions": int(total_interventions),
            "total_steps": total_steps,
        }
        self.shield_stats_history.append(shield_stats)

        info = {
            "sample_trajectories": last_trajectories,
            "sample_weights": last_weights,
            "best_trajectory": last_trajectories[best_idx],
            "best_cost": last_costs[best_idx],
            "mean_cost": np.mean(last_costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "dial_stats": dial_stats,
            "shield_info": shield_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _shielded_rollout(
        self, initial_state: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Shielded rollout: 매 timestep CBF shield 적용

        Args:
            initial_state: (nx,) 초기 상태
            controls: (K, N, nu) 원본 제어 시퀀스

        Returns:
            trajectories: (K, N+1, nx) 안전한 궤적
            shielded_controls: (K, N, nu) CBF 적용된 제어
            info: dict - shield 통계
        """
        K, N, nu = controls.shape
        nx = self.model.state_dim

        trajectories = np.zeros((K, N + 1, nx))
        shielded_controls = controls.copy()
        trajectories[:, 0, :] = initial_state

        total_interventions = 0
        total_vel_reduction = 0.0
        total_steps = K * N

        for t in range(N):
            states_t = trajectories[:, t, :]  # (K, nx)
            controls_t = controls[:, t, :]  # (K, nu)

            # CBF shield 적용
            safe_controls_t, intervened, vel_reduction = (
                self._cbf_shield_batch(states_t, controls_t)
            )

            shielded_controls[:, t, :] = safe_controls_t
            total_interventions += np.sum(intervened)
            total_vel_reduction += np.sum(vel_reduction)

            # 안전한 제어로 다음 상태 전파
            trajectories[:, t + 1, :] = self.model.step(
                states_t, safe_controls_t, self.params.dt
            )

        info = {
            "intervention_rate": float(total_interventions / total_steps) if total_steps > 0 else 0.0,
            "mean_vel_reduction": float(total_vel_reduction / max(total_interventions, 1)),
            "total_interventions": int(total_interventions),
            "total_steps": total_steps,
        }

        return trajectories, shielded_controls, info

    def _cbf_shield_batch(
        self, states: np.ndarray, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        해석적 CBF shield (완전 벡터화)

        Differential drive closed-form:
            h(x) = (x-xo)^2 + (y-yo)^2 - r_eff^2
            Lg_h[0] = 2(x-xo)*cos(theta) + 2(y-yo)*sin(theta)

            Lg_h[0] < 0 (접근): v_ceiling = alpha*h / |Lg_h[0]|
            Lg_h[0] >= 0 (이탈): 무조건 만족
            omega: 항상 자유

        Args:
            states: (K, nx) 현재 상태 [x, y, theta]
            controls: (K, nu) 원본 제어 [v, omega]

        Returns:
            safe_controls: (K, nu) CBF 적용된 제어
            intervened: (K,) bool, 개입 여부
            vel_reduction: (K,) 속도 감소량
        """
        K = states.shape[0]
        safe_controls = controls.copy()
        v_original = controls[:, 0]  # (K,)

        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        obstacles = self.shield_dial_params.cbf_obstacles
        safety_margin = self.shield_dial_params.cbf_safety_margin
        alpha = self._shield_alpha

        # 모든 장애물에 대해 v_ceiling 계산 후 최보수적(최소) 적용
        v_ceiling = np.full(K, np.inf)

        for obs_x, obs_y, obs_r in obstacles:
            effective_r = obs_r + safety_margin

            dx = x - obs_x
            dy = y - obs_y

            # Barrier value
            h = dx**2 + dy**2 - effective_r**2

            # Lg_h[0] = 2*dx*cos(theta) + 2*dy*sin(theta)
            Lg_h_v = 2.0 * dx * cos_theta + 2.0 * dy * sin_theta

            # CBF 제약: Lg_h_v * v + alpha * h >= 0
            # 접근 시 (Lg_h_v < 0): v <= alpha * h / |Lg_h_v|
            # 이탈 시 (Lg_h_v >= 0): 무조건 만족 -> ceiling = inf
            approaching = Lg_h_v < -1e-10

            v_ceiling_obs = np.where(
                approaching,
                alpha * h / np.maximum(np.abs(Lg_h_v), 1e-10),
                np.inf,
            )

            v_ceiling = np.minimum(v_ceiling, v_ceiling_obs)

        # 속도 클리핑: v_safe = min(v_original, v_ceiling)
        v_safe = np.minimum(v_original, v_ceiling)

        # 제어 제약 하한 적용
        if self.u_min is not None:
            v_safe = np.maximum(v_safe, self.u_min[0])

        safe_controls[:, 0] = v_safe

        # 개입 여부 및 속도 감소량
        intervened = v_safe < v_original - 1e-10
        vel_reduction = np.where(intervened, v_original - v_safe, 0.0)

        return safe_controls, intervened, vel_reduction

    def update_obstacles(self, obstacles: List[tuple]):
        """동적 장애물 업데이트"""
        self.shield_dial_params.cbf_obstacles = obstacles

    def set_shield_enabled(self, enabled: bool):
        """Shield 활성화/비활성화"""
        self._shield_enabled = enabled

    def get_shield_statistics(self) -> Dict:
        """
        Shield 통계 반환

        Returns:
            dict:
                - mean_intervention_rate: float 평균 개입률
                - total_interventions: int 총 개입 횟수
                - num_steps: int 기록된 스텝 수
        """
        if not self.shield_stats_history:
            return {
                "mean_intervention_rate": 0.0,
                "total_interventions": 0,
                "num_steps": 0,
            }

        rates = [s["intervention_rate"] for s in self.shield_stats_history]
        total_int = sum(
            s["total_interventions"] for s in self.shield_stats_history
        )

        return {
            "mean_intervention_rate": float(np.mean(rates)),
            "total_interventions": total_int,
            "num_steps": len(self.shield_stats_history),
        }

    def reset(self):
        """제어 시퀀스 및 통계 초기화"""
        super().reset()
        self.shield_stats_history = []

    def __repr__(self) -> str:
        return (
            f"ShieldDIALMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"n_diffuse_init={self.dial_params.n_diffuse_init}, "
            f"n_diffuse={self.dial_params.n_diffuse}, "
            f"shield_alpha={self._shield_alpha}, "
            f"K={self.params.K})"
        )
