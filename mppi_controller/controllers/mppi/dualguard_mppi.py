"""
DualGuard-MPPI Controller (36th variant)

Hamilton-Jacobi 도달 가능성 분석 영감의 안전 가치 함수를 MPPI에 통합.
사전 계산된 V(x)로 궤적 안전성 평가 + soft/hard/filter 세 가지 가드 모드.
Nominal + sample 이중 안전 보호.

Reference: Borquez et al., IEEE RA-L 2025, arXiv:2502.01924

핵심 수식:
    V(x) = min_i (||pos - o_i|| - (r_i + margin))   (signed distance)
    V(x) >= 0  -> safe,  V(x) < 0 -> unsafe

    Soft:   cost_k += penalty * exp(-decay * V(x))   (V < threshold)
    Hard:   u_k += alpha * grad_V(x_t)               (gradient projection)
    Filter: w_k = 0  if any V(x_t) < 0               (rejection)

    Velocity penalty:
        V_vel = -w_v * max(-dot(v, grad_V), 0)       (moving toward obstacle)

    Noise boost:
        if safe_fraction < min_safe_fraction:
            sigma *= noise_boost_factor
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import DualGuardMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class SafetyValueFunction:
    """
    Analytical HJ-inspired safety value function.

    signed distance 기반 안전 가치 함수. 원형 장애물에 대한
    해석적 V(x) + gradient + velocity penalty.

    V(x) = min_i (||pos - o_i|| - (r_i + margin))
    V(x) >= 0: safe, V(x) < 0: unsafe (inside obstacle)
    """

    def __init__(
        self,
        obstacles: List[tuple],
        safety_margin: float = 0.2,
        ttc_horizon: float = 1.0,
    ):
        """
        Args:
            obstacles: [(x, y, radius), ...] 원형 장애물
            safety_margin: 추가 안전 마진 (m)
            ttc_horizon: time-to-collision 판정 호라이즌 (초)
        """
        self.obstacles = list(obstacles)
        self.safety_margin = safety_margin
        self.ttc_horizon = ttc_horizon

        # 장애물 데이터를 배열로 사전 변환 (벡터화 성능)
        if self.obstacles:
            obs_arr = np.array(self.obstacles)  # (M, 3)
            self._obs_pos = obs_arr[:, :2]       # (M, 2)
            self._obs_rad = obs_arr[:, 2]        # (M,)
        else:
            self._obs_pos = np.zeros((0, 2))
            self._obs_rad = np.zeros(0)

    def evaluate(self, states: np.ndarray) -> np.ndarray:
        """
        Compute safety value V(x) for batch of states.

        V(x) = min_i (||pos - o_i|| - (r_i + margin))
        Positive = safe, Negative = unsafe.

        Args:
            states: (..., nx) where nx >= 2 (x, y, ...)

        Returns:
            values: (...) safety values (scalar per state)
        """
        if len(self.obstacles) == 0:
            # No obstacles -> always safe (large positive value)
            return np.full(states.shape[:-1], 1e6)

        positions = states[..., :2]  # (..., 2)
        orig_shape = positions.shape[:-1]

        # Flatten for vectorized computation
        pos_flat = positions.reshape(-1, 2)  # (B, 2)
        B = pos_flat.shape[0]
        M = len(self.obstacles)

        # Compute distances: (B, M)
        # diff[b, m] = pos[b] - obs[m]
        diff = pos_flat[:, None, :] - self._obs_pos[None, :, :]  # (B, M, 2)
        dist = np.linalg.norm(diff, axis=-1)  # (B, M)

        # Signed distance: dist - (radius + margin)
        signed_dist = dist - (self._obs_rad[None, :] + self.safety_margin)  # (B, M)

        # Safety value = minimum over all obstacles
        values = np.min(signed_dist, axis=-1)  # (B,)

        return values.reshape(orig_shape)

    def evaluate_with_velocity(self, states: np.ndarray, dt: float = 0.05) -> np.ndarray:
        """
        Enhanced safety: V(x) considering velocity toward obstacles.

        For states with velocity information (approximated from consecutive states),
        compute TTC-based penalty for approaching obstacles.

        V_vel = V_dist + penalty * max(-v_approach, 0) * ttc_factor

        Args:
            states: (K, N+1, nx) or (N+1, nx) trajectories

        Returns:
            values: same leading shape, enhanced safety values
        """
        if len(self.obstacles) == 0:
            return np.full(states.shape[:-1], 1e6)

        # Distance-based value
        dist_values = self.evaluate(states)

        # Velocity penalty: approximate velocity from consecutive states
        if states.ndim >= 2 and states.shape[-2] > 1:
            # Compute velocity approximation from consecutive positions
            positions = states[..., :2]  # (..., N+1, 2)
            # Forward difference velocity
            vel = np.zeros_like(positions)
            vel[..., :-1, :] = (positions[..., 1:, :] - positions[..., :-1, :]) / dt
            vel[..., -1, :] = vel[..., -2, :]  # repeat last

            # For each obstacle, compute approach speed
            pos_flat = positions.reshape(-1, 2)  # (B, 2)
            vel_flat = vel.reshape(-1, 2)       # (B, 2)
            B = pos_flat.shape[0]
            M = len(self.obstacles)

            diff = pos_flat[:, None, :] - self._obs_pos[None, :, :]  # (B, M, 2)
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)  # (B, M, 1)
            # Normalize direction from obstacle
            dir_away = diff / np.maximum(dist, 1e-6)  # (B, M, 2)

            # Approach speed: negative means approaching
            approach_speed = np.sum(vel_flat[:, None, :] * dir_away, axis=-1)  # (B, M)

            # Penalty for approaching: max(-approach_speed, 0)
            approach_penalty = np.maximum(-approach_speed, 0.0)  # (B, M)

            # TTC factor: if approach_speed < 0, ttc = dist / |approach_speed|
            # Penalty stronger when ttc < ttc_horizon
            ttc = np.where(
                approach_speed < -1e-6,
                np.squeeze(dist, axis=-1) / np.maximum(-approach_speed, 1e-6),
                1e6,
            )  # (B, M)
            ttc_factor = np.maximum(1.0 - ttc / self.ttc_horizon, 0.0)  # (B, M)

            # Aggregate: max penalty over obstacles
            vel_penalty = np.max(approach_penalty * ttc_factor, axis=-1)  # (B,)
            vel_penalty = vel_penalty.reshape(dist_values.shape)

            return dist_values - vel_penalty

        return dist_values

    def gradient(self, states: np.ndarray) -> np.ndarray:
        """
        Gradient of V(x) w.r.t. position.

        For the nearest obstacle: grad_V = (pos - obs_nearest) / ||pos - obs_nearest||
        Points away from the nearest obstacle.

        Args:
            states: (..., nx) where nx >= 2

        Returns:
            grad: (..., 2) gradient in position space
        """
        if len(self.obstacles) == 0:
            return np.zeros(states.shape[:-1] + (2,))

        positions = states[..., :2]  # (..., 2)
        orig_shape = positions.shape[:-1]

        pos_flat = positions.reshape(-1, 2)  # (B, 2)
        B = pos_flat.shape[0]
        M = len(self.obstacles)

        diff = pos_flat[:, None, :] - self._obs_pos[None, :, :]  # (B, M, 2)
        dist = np.linalg.norm(diff, axis=-1)  # (B, M)

        # Signed distance per obstacle
        signed_dist = dist - (self._obs_rad[None, :] + self.safety_margin)  # (B, M)

        # Find nearest obstacle (minimum signed distance)
        nearest_idx = np.argmin(signed_dist, axis=-1)  # (B,)

        # Gather diff and dist for nearest obstacle
        batch_idx = np.arange(B)
        nearest_diff = diff[batch_idx, nearest_idx, :]  # (B, 2)
        nearest_dist = dist[batch_idx, nearest_idx]      # (B,)

        # Gradient: unit vector away from nearest obstacle
        grad = nearest_diff / np.maximum(nearest_dist[:, None], 1e-8)  # (B, 2)

        return grad.reshape(orig_shape + (2,))


class DualGuardMPPIController(MPPIController):
    """
    DualGuard-MPPI Controller (36th MPPI variant)

    HJ 안전 가치 함수 기반 이중 안전 보호를 MPPI에 통합.

    Vanilla MPPI 대비 핵심 차이:
        1. 안전 가치 함수 V(x): signed distance + velocity penalty
        2. Soft guard:   cost += penalty * exp(-decay * V) (V < threshold)
        3. Hard guard:   gradient projection toward safe region
        4. Filter guard: reject unsafe samples entirely
        5. Dual guard:   nominal trajectory + all samples 이중 검사
        6. Noise boost:  안전 샘플 부족 시 자동 노이즈 증폭

    Args:
        model: RobotModel 인스턴스
        params: DualGuardMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수 사용)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 사용)
    """

    def __init__(
        self,
        model: RobotModel,
        params: DualGuardMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.guard_params = params

        # Safety value function
        self._safety_value = SafetyValueFunction(
            params.obstacles,
            params.safety_margin,
            params.ttc_horizon,
        )

        # Guard statistics
        self._guard_stats = []

        # Noise boost state
        self._noise_boost = 1.0

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        DualGuard-MPPI 제어 계산

        1. 표준 MPPI 샘플링 + 롤아웃
        2. 안전 가치 V(x) 평가
        3. Guard 모드 적용 (soft/hard/filter)
        4. 안전 샘플 비율 확인 + 노이즈 부스트
        5. MPPI 가중 업데이트
        6. Nominal guard: 최종 궤적 안전 확인

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K
        N = self.params.N

        # 1. Noise sampling with boost
        sigma_eff = self.params.sigma * self._noise_boost
        noise = np.random.standard_normal(
            (K, N, self.model.control_dim)
        ) * sigma_eff[None, None, :]

        # 2. Sample control sequences
        sampled_controls = self.U[None, :, :] + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. Rollout trajectories
        trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. Base cost computation
        costs = self.cost_function.compute_cost(
            trajectories, sampled_controls, reference_trajectory
        )

        # 5. Compute safety values for all trajectories (K, N+1)
        safety_values = self._safety_value.evaluate(trajectories)

        # 6. Velocity-based penalty (optional)
        velocity_costs = np.zeros(K)
        if self.guard_params.use_velocity_penalty and len(self.guard_params.obstacles) > 0:
            velocity_costs = self._compute_velocity_penalty(trajectories)

        # 7. Apply guard mode
        if self.guard_params.use_sample_guard and len(self.guard_params.obstacles) > 0:
            mode = self.guard_params.safety_mode
            if mode == "soft":
                costs = self._apply_soft_guard(costs, safety_values)
            elif mode == "hard":
                sampled_controls = self._apply_hard_guard(
                    sampled_controls, trajectories, safety_values, state
                )
                # Re-rollout with corrected controls
                trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
                # Recompute costs with corrected trajectories
                costs = self.cost_function.compute_cost(
                    trajectories, sampled_controls, reference_trajectory
                )
                safety_values = self._safety_value.evaluate(trajectories)
                costs = self._apply_soft_guard(costs, safety_values)
            elif mode == "filter":
                costs = self._apply_filter_guard(costs, safety_values)

        # Add velocity penalty
        costs = costs + velocity_costs

        # 8. Check safe fraction and update noise boost
        safe_fraction = self._check_and_boost_noise(safety_values)

        # 9. MPPI weights
        weights = self._compute_weights(costs, self.params.lambda_)

        # 10. Weighted update (using noise relative to U)
        noise_actual = sampled_controls - self.U[None, :, :]
        weighted_noise = np.sum(weights[:, None, None] * noise_actual, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 11. Receding horizon shift (matches base_mppi pattern)
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 12. Extract first control (after shift, same as base class)
        optimal_control = self.U[0].copy()

        # 13. Nominal guard: check if nominal trajectory is safe
        nominal_correction = 0.0
        if self.guard_params.use_nominal_guard and len(self.guard_params.obstacles) > 0:
            optimal_control, nominal_correction = self._guard_nominal(
                state, optimal_control
            )
            self.U[0] = optimal_control

        # 14. Statistics
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)
        min_safety = float(np.min(safety_values))
        mean_safety = float(np.mean(safety_values))

        guard_stats = {
            "safe_fraction": float(safe_fraction),
            "min_safety_value": min_safety,
            "mean_safety_value": mean_safety,
            "noise_boost": float(self._noise_boost),
            "sigma_eff": sigma_eff.tolist(),
            "nominal_correction": float(nominal_correction),
            "safety_mode": self.guard_params.safety_mode,
            "num_obstacles": len(self.guard_params.obstacles),
        }
        self._guard_stats.append(guard_stats)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "guard_stats": guard_stats,
            "safety_values": safety_values,
        }
        self.last_info = info

        return optimal_control, info

    def _compute_velocity_penalty(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Velocity-based safety penalty: penalize moving toward obstacles.

        Uses finite differences to approximate velocity, then computes
        approach speed toward each obstacle.

        Args:
            trajectories: (K, N+1, nx)

        Returns:
            penalty: (K,) velocity penalty per sample
        """
        K, N_plus_1, nx = trajectories.shape
        dt = self.params.dt
        w_v = self.guard_params.velocity_penalty_weight

        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        # Approximate velocity from consecutive positions
        vel = np.zeros_like(positions)
        vel[:, :-1, :] = (positions[:, 1:, :] - positions[:, :-1, :]) / dt
        vel[:, -1, :] = vel[:, -2, :]

        M = len(self.guard_params.obstacles)
        if M == 0:
            return np.zeros(K)

        obs_pos = self._safety_value._obs_pos  # (M, 2)

        # Direction from each state to each obstacle
        # positions: (K, N+1, 2), obs_pos: (M, 2)
        diff = positions[:, :, None, :] - obs_pos[None, None, :, :]  # (K, N+1, M, 2)
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)  # (K, N+1, M, 1)
        dir_away = diff / np.maximum(dist, 1e-6)  # (K, N+1, M, 2)

        # Approach speed: dot(velocity, direction_away) < 0 means approaching
        approach_speed = np.sum(
            vel[:, :, None, :] * dir_away, axis=-1
        )  # (K, N+1, M)

        # Penalty for approaching: max(-approach_speed, 0)
        approach_penalty = np.maximum(-approach_speed, 0.0)  # (K, N+1, M)

        # Weight by proximity: only penalize when close to obstacle
        # Use signed distance to effective boundary (radius + margin)
        obs_rad = self._safety_value._obs_rad  # (M,)
        margin = self.guard_params.safety_margin
        dist_sq = np.squeeze(dist, axis=-1)  # (K, N+1, M)
        signed = dist_sq - (obs_rad[None, None, :] + margin)  # (K, N+1, M)

        # Proximity factor: 1.0 when at boundary, 0.0 when far away
        influence_range = 2.0  # only penalize within 2m of boundary
        proximity = np.clip(1.0 - signed / influence_range, 0.0, 1.0)  # (K, N+1, M)

        # Sum over timesteps and obstacles
        penalty = w_v * np.sum(approach_penalty * proximity, axis=(1, 2))  # (K,)

        return penalty

    def _apply_soft_guard(
        self, costs: np.ndarray, safety_values: np.ndarray
    ) -> np.ndarray:
        """
        Soft safety guard: add exponential penalty for unsafe regions.

        cost_k += penalty * sum_t exp(-decay * V(x_t^k))
        Only applied where V(x) < some threshold (positive margin for smoothness).

        Args:
            costs: (K,) base costs
            safety_values: (K, N+1) safety values

        Returns:
            augmented_costs: (K,) with safety penalty added
        """
        penalty = self.guard_params.safety_penalty
        decay = self.guard_params.safety_decay

        # Threshold: apply penalty whenever V is below positive margin
        threshold = self.guard_params.safety_margin

        # Exponential penalty: penalty * exp(-decay * V)
        # For V << 0 (deep inside obstacle), penalty is very large
        # For V >> 0 (far from obstacle), penalty approaches 0
        # Clamp to prevent overflow
        exponent = np.clip(-decay * safety_values, -50, 50)  # (K, N+1)
        safety_cost = penalty * np.exp(exponent)  # (K, N+1)

        # Only apply where V < threshold
        mask = safety_values < threshold  # (K, N+1)
        safety_cost = safety_cost * mask

        # Sum over timesteps
        total_safety_cost = np.sum(safety_cost, axis=-1)  # (K,)

        return costs + total_safety_cost

    def _apply_hard_guard(
        self,
        controls: np.ndarray,
        trajectories: np.ndarray,
        safety_values: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Hard safety guard: project controls toward safe region using value gradient.

        For timesteps where V(x_t) < 0:
            u_t += alpha * grad_V(x_t)  (projected to control space)

        Uses the position components of the gradient.

        Args:
            controls: (K, N, nu) sampled controls
            trajectories: (K, N+1, nx) trajectories
            safety_values: (K, N+1) safety values

        Returns:
            corrected_controls: (K, N, nu)
        """
        K, N, nu = controls.shape

        # Find unsafe timesteps: V < 0
        unsafe_mask = safety_values[:, 1:] < 0  # (K, N) skip initial state

        if not np.any(unsafe_mask):
            return controls

        # Compute gradient at unsafe states
        grad = self._safety_value.gradient(trajectories[:, 1:, :])  # (K, N, 2)

        # Project gradient to control space
        # For differential drive: u = [v, omega]
        # grad_V in position space = [dx, dy]
        # Simplification: nudge linear velocity in direction of gradient
        correction = np.zeros_like(controls)

        # Scale correction by how unsafe the state is
        violation = np.maximum(-safety_values[:, 1:], 0.0)  # (K, N)
        correction_scale = np.minimum(violation * 2.0, 1.0)  # (K, N)

        # Linear velocity correction: move away from obstacle
        # Use gradient magnitude as velocity correction
        grad_mag = np.linalg.norm(grad, axis=-1)  # (K, N)
        correction[:, :, 0] = correction_scale * grad_mag * unsafe_mask

        # Angular velocity: steer away from obstacle
        # Direction of gradient gives desired heading
        grad_angle = np.arctan2(grad[:, :, 1], grad[:, :, 0])  # (K, N)
        current_theta = trajectories[:, 1:, 2] if trajectories.shape[-1] > 2 else np.zeros((K, N))
        angle_diff = grad_angle - current_theta
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        correction[:, :, 1] = correction_scale * angle_diff * 0.5 * unsafe_mask

        corrected = controls + correction

        # Re-clip
        if self.u_min is not None and self.u_max is not None:
            corrected = np.clip(corrected, self.u_min, self.u_max)

        return corrected

    def _apply_filter_guard(
        self, costs: np.ndarray, safety_values: np.ndarray
    ) -> np.ndarray:
        """
        Filter safety guard: reject unsafe samples entirely.

        Set very high cost for any sample that has V(x) < 0 at any timestep.

        Args:
            costs: (K,) base costs
            safety_values: (K, N+1) safety values

        Returns:
            filtered_costs: (K,) with unsafe samples penalized
        """
        # Sample is unsafe if any timestep has V < 0
        min_safety_per_sample = np.min(safety_values, axis=-1)  # (K,)
        unsafe_mask = min_safety_per_sample < 0  # (K,)

        # Set very large cost for unsafe samples
        filtered_costs = costs.copy()
        large_penalty = self.guard_params.safety_penalty * self.params.N
        filtered_costs[unsafe_mask] += large_penalty

        return filtered_costs

    def _check_and_boost_noise(self, safety_values: np.ndarray) -> float:
        """
        Check safe fraction and boost noise if too few safe samples.

        Args:
            safety_values: (K, N+1) safety values

        Returns:
            safe_fraction: float, fraction of safe samples
        """
        if len(self.guard_params.obstacles) == 0:
            self._noise_boost = 1.0
            return 1.0

        # A sample is "safe" if min V(x) over its trajectory >= 0
        min_safety_per_sample = np.min(safety_values, axis=-1)  # (K,)
        safe_count = np.sum(min_safety_per_sample >= 0)
        safe_fraction = safe_count / len(min_safety_per_sample)

        if safe_fraction < self.guard_params.min_safe_fraction:
            # Too few safe samples -> boost noise
            self._noise_boost = min(
                self._noise_boost * self.guard_params.noise_boost_factor,
                5.0,  # cap at 5x
            )
        else:
            # Enough safe samples -> decay boost back toward 1.0
            self._noise_boost = max(
                1.0,
                self._noise_boost * 0.9,  # gradual decay
            )

        return float(safe_fraction)

    def _guard_nominal(
        self, state: np.ndarray, optimal_control: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Nominal guard: check if applying optimal_control leads to safe next state.
        If unsafe, correct using value gradient.

        Args:
            state: (nx,) current state
            optimal_control: (nu,) proposed control

        Returns:
            corrected_control: (nu,)
            correction_norm: float
        """
        # Predict next state
        state_dot = self.model.forward_dynamics(state, optimal_control)
        next_state = state + state_dot * self.params.dt

        # Evaluate safety
        V = self._safety_value.evaluate(next_state[None, :])[0]

        if V >= 0:
            return optimal_control, 0.0

        # Unsafe: apply gradient-based correction
        grad = self._safety_value.gradient(next_state[None, :])[0]  # (2,)

        # Scale correction by violation magnitude
        violation = max(-V, 0.0)
        scale = min(violation * 2.0, 1.0)

        correction = np.zeros_like(optimal_control)
        grad_mag = np.linalg.norm(grad)

        # Linear velocity: boost in direction away from obstacle
        correction[0] = scale * grad_mag

        # Angular velocity: steer away
        if next_state.shape[0] > 2:
            grad_angle = np.arctan2(grad[1], grad[0])
            angle_diff = grad_angle - next_state[2]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            correction[1] = scale * angle_diff * 0.5

        corrected = optimal_control + correction
        if self.u_min is not None and self.u_max is not None:
            corrected = np.clip(corrected, self.u_min, self.u_max)

        correction_norm = float(np.linalg.norm(corrected - optimal_control))
        return corrected, correction_norm

    def update_obstacles(self, obstacles: List[tuple]):
        """
        동적 장애물 실시간 업데이트

        Args:
            obstacles: [(x, y, radius), ...] 새 장애물 목록
        """
        self.guard_params.obstacles = list(obstacles)
        self._safety_value = SafetyValueFunction(
            obstacles,
            self.guard_params.safety_margin,
            self.guard_params.ttc_horizon,
        )

    def get_guard_statistics(self) -> Dict:
        """
        누적 guard 통계 반환

        Returns:
            dict:
                - history: 전체 스텝별 통계 리스트
                - mean_safe_fraction: 평균 안전 샘플 비율
                - min_safety_value_ever: 전체 최소 안전 가치
                - mean_noise_boost: 평균 노이즈 부스트
                - mean_nominal_correction: 평균 nominal 보정량
        """
        if len(self._guard_stats) == 0:
            return {
                "history": [],
                "mean_safe_fraction": 1.0,
                "min_safety_value_ever": float('inf'),
                "mean_noise_boost": 1.0,
                "mean_nominal_correction": 0.0,
            }

        safe_fracs = [s["safe_fraction"] for s in self._guard_stats]
        min_vals = [s["min_safety_value"] for s in self._guard_stats]
        boosts = [s["noise_boost"] for s in self._guard_stats]
        corrections = [s["nominal_correction"] for s in self._guard_stats]

        return {
            "history": self._guard_stats.copy(),
            "mean_safe_fraction": float(np.mean(safe_fracs)),
            "min_safety_value_ever": float(np.min(min_vals)),
            "mean_noise_boost": float(np.mean(boosts)),
            "mean_nominal_correction": float(np.mean(corrections)),
        }

    def reset(self):
        """제어 시퀀스 + guard 상태 + 통계 초기화"""
        super().reset()
        self._guard_stats = []
        self._noise_boost = 1.0
        # Re-initialize safety value from params
        self._safety_value = SafetyValueFunction(
            self.guard_params.obstacles,
            self.guard_params.safety_margin,
            self.guard_params.ttc_horizon,
        )

    def __repr__(self) -> str:
        return (
            f"DualGuardMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"obstacles={len(self.guard_params.obstacles)}, "
            f"mode={self.guard_params.safety_mode}, "
            f"K={self.params.K})"
        )
