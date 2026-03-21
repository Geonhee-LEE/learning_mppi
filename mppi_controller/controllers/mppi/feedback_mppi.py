"""
Feedback-MPPI Controller (F-MPPI)

MPPI rollout에 대한 sensitivity analysis를 통해 Riccati 기반 피드백 게인을 계산.
매 타임스텝마다 전체 재최적화 없이 국소 선형 피드백으로 빠른 폐루프 보정.
저수준 제어기 없이 MPPI 단독으로 고주파 제어 가능.

핵심 수식:
    A_t = df/dx|_{x*,u*}, B_t = df/du|_{x*,u*}  -- rollout Jacobians (finite diff)
    P_T = Q_f  -- terminal cost
    P_t = Q + A_t^T P_{t+1} A_t
          - A_t^T P_{t+1} B_t (R + B_t^T P_{t+1} B_t)^{-1} B_t^T P_{t+1} A_t
    K_t = -(R + B_t^T P_{t+1} B_t)^{-1} B_t^T P_{t+1} A_t  -- feedback gain
    u_applied = u*(t) + K_t (x_actual - x*_t)  -- feedback correction

Key insight: After one full MPPI solve, compute Jacobians along nominal trajectory,
then for next `reuse_steps` timesteps, just apply feedback correction without re-solving.

Reference: Belvedere et al., IEEE RA-L 2026, arXiv:2506.14855
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import FeedbackMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class FeedbackMPPIController(MPPIController):
    """
    Feedback-MPPI Controller (F-MPPI) -- 34th variant

    Riccati feedback between MPPI solves.

    Two modes of operation:
        1. Full MPPI solve (reuse_counter == 0 or feedback disabled):
           - Run standard MPPI -> get U*, trajectories
           - Compute Jacobians along nominal -> Riccati gains K_t
           - Apply u = U*[0]
           - Store nominal traj + gains

        2. Feedback reuse (reuse_counter > 0):
           - u = U*[step] + K_step @ (x_actual - x_nominal[step])
           - No rollouts, no cost evaluation
           - Much faster (< 0.1ms)

    Args:
        model: RobotModel instance
        params: FeedbackMPPIParams parameters
        cost_function: CostFunction (None -> default composite cost)
        noise_sampler: NoiseSampler (None -> GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: FeedbackMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.feedback_params = params

        # Nominal trajectory & gains from last full MPPI solve
        self._nominal_trajectory = None   # (N+1, nx) from last MPPI solve
        self._nominal_controls = None     # (N, nu) from last MPPI solve
        self._feedback_gains = None       # (N, nu, nx) Riccati gains
        self._reuse_counter = 0           # current step within reuse window
        self._current_step_in_sequence = 0  # which timestep to apply
        self._last_full_solve_info = None   # info from last full solve

        # Statistics
        self._fb_stats_history: List[Dict] = []
        self._total_steps = 0
        self._full_solve_count = 0
        self._reuse_count = 0

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        F-MPPI control computation.

        If feedback is disabled or reuse_counter == 0: full MPPI solve.
        Otherwise: fast feedback reuse.

        Args:
            state: (nx,) current state
            reference_trajectory: (N+1, nx) reference trajectory

        Returns:
            control: (nu,) optimal control input
            info: dict with MPPI info + feedback_stats
        """
        self._total_steps += 1

        if not self.feedback_params.use_feedback:
            # Feedback disabled -> every step is full MPPI solve
            return self._full_solve(state, reference_trajectory)

        # Decide: full solve or feedback reuse
        need_full_solve = (
            self._reuse_counter == 0
            or self._nominal_trajectory is None
            or self._feedback_gains is None
            or self._current_step_in_sequence >= self.params.N - 1
        )

        if need_full_solve:
            return self._full_solve(state, reference_trajectory)
        else:
            return self._feedback_step(state, reference_trajectory)

    def _full_solve(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full MPPI solve + Jacobian/Riccati computation.

        Steps:
            1. Run standard MPPI to get optimal U*
            2. Extract nominal trajectory by forward rollout
            3. Compute Jacobians A_t, B_t along nominal
            4. Solve backward Riccati for feedback gains K_t
            5. Return first control u*[0]
        """
        self._full_solve_count += 1

        K = self.params.K
        N = self.params.N

        # 1. Noise sampling (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. Sample control sequences (K, N, nu)
        sampled_controls = self.U + noise
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. Rollout (K, N+1, nx)
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. Cost computation (K,)
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. MPPI weights
        weights = self._compute_weights(costs, self.params.lambda_)

        # 6. Weighted average update
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # Save optimal control sequence BEFORE shift for nominal trajectory
        optimal_U = self.U.copy()

        # 7. Extract first control
        optimal_control = self.U[0].copy()

        # 8. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 9. Compute nominal trajectory from current state + optimal controls
        self._nominal_trajectory = self._extract_nominal_trajectory(
            state, optimal_U
        )
        self._nominal_controls = optimal_U.copy()

        # 10. Compute Jacobians along nominal trajectory
        A_list, B_list = self._compute_jacobians(
            self._nominal_trajectory, self._nominal_controls
        )

        # 11. Solve Riccati for feedback gains
        self._feedback_gains = self._solve_riccati(A_list, B_list)

        # 12. Reset reuse counter
        self._reuse_counter = self.feedback_params.reuse_steps
        self._current_step_in_sequence = 1  # next call will use step 1

        # 13. Build info
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        # Compute feedback stats
        mean_gain = float(np.mean(np.abs(self._feedback_gains)))
        max_gain = float(np.max(np.abs(self._feedback_gains)))

        feedback_stats = {
            "mode": "full_solve",
            "reuse_counter": self._reuse_counter,
            "step_in_sequence": 0,
            "mean_gain": mean_gain,
            "max_gain": max_gain,
            "feedback_correction_norm": 0.0,
            "full_solve_count": self._full_solve_count,
            "reuse_count": self._reuse_count,
        }
        self._fb_stats_history.append(feedback_stats)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "feedback_stats": feedback_stats,
        }
        self.last_info = info
        self._last_full_solve_info = info

        return optimal_control, info

    def _feedback_step(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fast feedback reuse step (no rollouts, no cost evaluation).

        u = u_nom[step] + K[step] @ (x_actual - x_nom[step])
        """
        self._reuse_count += 1

        step_idx = self._current_step_in_sequence

        # Apply feedback correction
        control, correction_norm = self._apply_feedback(state, step_idx)

        # Advance step
        self._current_step_in_sequence += 1
        self._reuse_counter -= 1

        # Warm start: shift U if enabled
        if self.feedback_params.use_warm_start:
            # U already shifted from last full solve; keep it
            pass

        # Build info (reuse from last full solve, but update feedback stats)
        mean_gain = float(np.mean(np.abs(self._feedback_gains)))
        max_gain = float(np.max(np.abs(self._feedback_gains)))

        feedback_stats = {
            "mode": "feedback_reuse",
            "reuse_counter": self._reuse_counter,
            "step_in_sequence": step_idx,
            "mean_gain": mean_gain,
            "max_gain": max_gain,
            "feedback_correction_norm": correction_norm,
            "full_solve_count": self._full_solve_count,
            "reuse_count": self._reuse_count,
        }
        self._fb_stats_history.append(feedback_stats)

        # Use cached info from last full solve, update feedback stats
        info = {}
        if self._last_full_solve_info is not None:
            info = self._last_full_solve_info.copy()
        info["feedback_stats"] = feedback_stats

        # Override ess/num_samples to indicate no rollouts
        info["num_samples"] = 0  # no rollouts in feedback mode

        self.last_info = info

        return control, info

    def _compute_jacobians(
        self,
        nominal_states: np.ndarray,
        nominal_controls: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute finite difference Jacobians A_t, B_t along nominal trajectory.

        A_t[i,j] = (f(x+eps*e_j, u) - f(x-eps*e_j, u)) / (2*eps)  for state
        B_t[i,j] = (f(x, u+eps*e_j) - f(x, u-eps*e_j)) / (2*eps)  for control

        Args:
            nominal_states: (N+1, nx) nominal state trajectory
            nominal_controls: (N, nu) nominal control sequence

        Returns:
            A_list: (N, nx, nx) state Jacobians
            B_list: (N, nx, nu) control Jacobians
        """
        N = nominal_controls.shape[0]
        nx = self.model.state_dim
        nu = self.model.control_dim
        eps = self.feedback_params.jacobian_eps
        dt = self.params.dt

        A_list = np.zeros((N, nx, nx))
        B_list = np.zeros((N, nx, nu))

        for t in range(N):
            x_t = nominal_states[t]
            u_t = nominal_controls[t]

            # A_t: df/dx (central differences)
            for j in range(nx):
                x_plus = x_t.copy()
                x_minus = x_t.copy()
                x_plus[j] += eps
                x_minus[j] -= eps

                f_plus = self.model.step(x_plus, u_t, dt)
                f_minus = self.model.step(x_minus, u_t, dt)

                A_list[t, :, j] = (f_plus - f_minus) / (2 * eps)

            # B_t: df/du (central differences)
            for j in range(nu):
                u_plus = u_t.copy()
                u_minus = u_t.copy()
                u_plus[j] += eps
                u_minus[j] -= eps

                f_plus = self.model.step(x_t, u_plus, dt)
                f_minus = self.model.step(x_t, u_minus, dt)

                B_list[t, :, j] = (f_plus - f_minus) / (2 * eps)

        return A_list, B_list

    def _solve_riccati(
        self,
        A_list: np.ndarray,
        B_list: np.ndarray,
    ) -> np.ndarray:
        """
        Backward Riccati recursion to compute feedback gains K_t.

        P_N = Qf_matrix
        P_t = Q + A^T P_{t+1} A
              - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A
        K_t = -(R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A

        Args:
            A_list: (N, nx, nx) state Jacobians
            B_list: (N, nx, nu) control Jacobians

        Returns:
            gains: (N, nu, nx) feedback gains
        """
        N = A_list.shape[0]
        nx = self.model.state_dim
        nu = self.model.control_dim
        clip_val = self.feedback_params.feedback_gain_clip

        # Build Q and R matrices
        Q_vec = self.params.Q
        R_vec = self.params.R
        Qf_vec = self.params.Qf

        Q_w = self.feedback_params.feedback_weight_Q
        R_w = self.feedback_params.feedback_weight_R

        # Diagonal Q, R, Qf matrices
        if Q_vec.ndim == 1:
            Q_matrix = np.diag(Q_vec) * Q_w
        else:
            Q_matrix = Q_vec * Q_w

        if R_vec.ndim == 1:
            R_matrix = np.diag(R_vec) * R_w
        else:
            R_matrix = R_vec * R_w

        if Qf_vec.ndim == 1:
            Qf_matrix = np.diag(Qf_vec) * Q_w
        else:
            Qf_matrix = Qf_vec * Q_w

        # Regularization for numerical stability
        reg = 1e-6 * np.eye(nu)

        # Backward Riccati recursion
        gains = np.zeros((N, nu, nx))
        P = Qf_matrix.copy()

        for t in range(N - 1, -1, -1):
            A = A_list[t]   # (nx, nx)
            B = B_list[t]   # (nx, nu)

            # B^T P B + R
            BtP = B.T @ P               # (nu, nx)
            BtPB = BtP @ B              # (nu, nu)
            M = R_matrix + BtPB + reg   # (nu, nu)

            # K_t = -M^{-1} B^T P A
            BtPA = BtP @ A              # (nu, nx)

            # Use np.linalg.solve for numerical stability
            try:
                K_t = -np.linalg.solve(M, BtPA)
            except np.linalg.LinAlgError:
                # Fallback: pseudo-inverse
                K_t = -np.linalg.lstsq(M, BtPA, rcond=None)[0]

            # Clip gains for stability
            K_t = np.clip(K_t, -clip_val, clip_val)
            gains[t] = K_t

            # P_t = Q + A^T P A + A^T P B K_t
            # (which equals Q + A^T P A - A^T P B M^{-1} B^T P A)
            AtP = A.T @ P
            P = Q_matrix + AtP @ A + AtP @ B @ K_t

            # Ensure P is symmetric (numerical stability)
            P = 0.5 * (P + P.T)

        return gains

    def _apply_feedback(
        self, state: np.ndarray, step_idx: int
    ) -> Tuple[np.ndarray, float]:
        """
        Apply feedback correction: u = u_nom[step] + K[step] @ (x - x_nom[step])

        Args:
            state: (nx,) actual state
            step_idx: which timestep in the nominal sequence

        Returns:
            control: (nu,) corrected control
            correction_norm: float L2 norm of the correction
        """
        # Clamp step_idx to valid range
        step_idx = min(step_idx, self._nominal_controls.shape[0] - 1)
        step_idx = min(step_idx, self._feedback_gains.shape[0] - 1)
        step_idx = min(step_idx, self._nominal_trajectory.shape[0] - 2)

        u_nom = self._nominal_controls[step_idx]    # (nu,)
        x_nom = self._nominal_trajectory[step_idx]  # (nx,)
        K_t = self._feedback_gains[step_idx]         # (nu, nx)

        # State error
        state_error = state - x_nom  # (nx,)

        # Angle wrapping for theta component if present
        if len(state_error) >= 3:
            state_error[2] = np.arctan2(
                np.sin(state_error[2]), np.cos(state_error[2])
            )

        # Feedback correction
        correction = K_t @ state_error  # (nu,)

        # Final control
        control = u_nom + correction

        # Apply control bounds
        if self.u_min is not None and self.u_max is not None:
            control = np.clip(control, self.u_min, self.u_max)

        correction_norm = float(np.linalg.norm(correction))

        return control, correction_norm

    def _extract_nominal_trajectory(
        self, state: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """
        Forward rollout nominal trajectory from current state + optimal controls.

        Args:
            state: (nx,) initial state
            controls: (N, nu) optimal control sequence

        Returns:
            trajectory: (N+1, nx) nominal state trajectory
        """
        N = controls.shape[0]
        nx = self.model.state_dim
        dt = self.params.dt

        trajectory = np.zeros((N + 1, nx))
        trajectory[0] = state.copy()

        for t in range(N):
            trajectory[t + 1] = self.model.step(trajectory[t], controls[t], dt)

        return trajectory

    def get_feedback_statistics(self) -> Dict:
        """
        Return accumulated feedback statistics.

        Returns:
            dict with:
                - total_steps: total compute_control calls
                - full_solve_count: number of full MPPI solves
                - reuse_count: number of feedback reuse steps
                - reuse_fraction: fraction of steps using feedback reuse
                - mean_gain: average absolute gain magnitude
                - max_gain: maximum absolute gain
                - history: full stats history
        """
        if not self._fb_stats_history:
            return {
                "total_steps": 0,
                "full_solve_count": 0,
                "reuse_count": 0,
                "reuse_fraction": 0.0,
                "mean_gain": 0.0,
                "max_gain": 0.0,
                "mean_correction_norm": 0.0,
                "history": [],
            }

        gains = [s["mean_gain"] for s in self._fb_stats_history]
        max_gains = [s["max_gain"] for s in self._fb_stats_history]
        corrections = [
            s["feedback_correction_norm"] for s in self._fb_stats_history
            if s["mode"] == "feedback_reuse"
        ]

        total = self._total_steps
        reuse_frac = self._reuse_count / total if total > 0 else 0.0

        return {
            "total_steps": total,
            "full_solve_count": self._full_solve_count,
            "reuse_count": self._reuse_count,
            "reuse_fraction": reuse_frac,
            "mean_gain": float(np.mean(gains)) if gains else 0.0,
            "max_gain": float(np.max(max_gains)) if max_gains else 0.0,
            "mean_correction_norm": float(np.mean(corrections)) if corrections else 0.0,
            "history": self._fb_stats_history.copy(),
        }

    def reset(self):
        """Reset control sequence and feedback state."""
        super().reset()
        self._nominal_trajectory = None
        self._nominal_controls = None
        self._feedback_gains = None
        self._reuse_counter = 0
        self._current_step_in_sequence = 0
        self._last_full_solve_info = None
        self._fb_stats_history = []
        self._total_steps = 0
        self._full_solve_count = 0
        self._reuse_count = 0

    def __repr__(self) -> str:
        return (
            f"FeedbackMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"reuse_steps={self.feedback_params.reuse_steps}, "
            f"use_feedback={self.feedback_params.use_feedback}, "
            f"K={self.params.K})"
        )
