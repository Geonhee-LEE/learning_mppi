"""
BPTT (Backpropagation Through Time) 잔차 동역학 학습 파이프라인

미분가능 시뮬레이터를 통해 궤적 수준의 추적 비용을 역전파하여
잔차 모델을 학습. 1-step MSE 대신 N-step trajectory loss 최적화.

Reference:
    Pan et al. (2025) "Learning on the Fly" (UZH)

Usage:
    trainer = BPTTResidualTrainer(residual_model, diff_sim, norm_stats)
    episodes = trainer.generate_episodes(dyn_model, kin_model, traj_fn)
    history = trainer.train(episodes)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Callable
import os
from pathlib import Path

from mppi_controller.learning.spectral_regularization import SpectralRegularizer


class BPTTResidualTrainer:
    """
    궤적 추적 비용을 미분가능 시뮬레이터를 통해 역전파하여 잔차 모델을 학습.

    핵심 차이:
      - MSE: ||f_pred(t) - f_true(t)||^2 (1-step)
      - BPTT: Σ_t ||FK(x_t) - ref_t||^2 (N-step trajectory)

    BPTT는 궤적 수준의 누적 오차를 직접 최소화.

    Args:
        residual_model: nn.Module (MLP) — 잔차 동역학 모델
        diff_sim: DifferentiableMobileManipulator6DOF — 미분가능 시뮬레이터
        norm_stats: dict — 정규화 통계 (state_mean/std, control_mean/std, state_dot_mean/std)
        learning_rate: 학습률
        spectral_lambda: Spectral regularization 강도
        ee_weight: EE 추적 비용 가중치
        control_weight: 제어 노력 비용 가중치 (미사용, 예약)
        rollout_horizon: Rollout 길이 (timesteps)
        dt: 시뮬레이션 timestep
        truncation_length: TBPTT 절단 길이 (gradient detach 간격)
        device: 'cpu' or 'cuda'
        save_dir: 모델 저장 디렉터리
    """

    def __init__(
        self,
        residual_model: nn.Module,
        diff_sim,  # DifferentiableMobileManipulator6DOF
        norm_stats: Optional[dict] = None,
        learning_rate: float = 1e-4,
        spectral_lambda: float = 0.01,
        ee_weight: float = 1.0,
        control_weight: float = 0.01,
        rollout_horizon: int = 20,
        dt: float = 0.05,
        truncation_length: int = 10,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    ):
        self.residual_model = residual_model
        self.diff_sim = diff_sim
        self.norm_stats = norm_stats
        self.ee_weight = ee_weight
        self.control_weight = control_weight
        self.rollout_horizon = rollout_horizon
        self.dt = dt
        self.truncation_length = truncation_length
        self.device = torch.device(device)
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Move models to device
        self.residual_model = self.residual_model.to(self.device)
        self.diff_sim = self.diff_sim.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.residual_model.parameters(), lr=learning_rate
        )

        # Spectral regularization
        self.spectral_reg = None
        if spectral_lambda > 0:
            self.spectral_reg = SpectralRegularizer(
                self.residual_model, spectral_lambda
            )

        # History
        self.history = {"train_loss": [], "val_loss": []}

    def _normalize_input(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Normalize [state, control] for residual model input."""
        if self.norm_stats is not None:
            s_mean = torch.tensor(self.norm_stats["state_mean"], dtype=torch.float64, device=state.device)
            s_std = torch.tensor(self.norm_stats["state_std"], dtype=torch.float64, device=state.device)
            c_mean = torch.tensor(self.norm_stats["control_mean"], dtype=torch.float64, device=state.device)
            c_std = torch.tensor(self.norm_stats["control_std"], dtype=torch.float64, device=state.device)

            state_norm = (state - s_mean) / s_std
            control_norm = (control - c_mean) / c_std
        else:
            state_norm = state
            control_norm = control

        inp = torch.cat([state_norm, control_norm], dim=-1)
        # Match residual model dtype
        model_dtype = next(self.residual_model.parameters()).dtype
        return inp.to(model_dtype)

    def _denormalize_output(self, residual_dot: torch.Tensor) -> torch.Tensor:
        """Denormalize residual model output to state_dot space."""
        if self.norm_stats is not None:
            sd_mean = torch.tensor(self.norm_stats["state_dot_mean"], dtype=torch.float64, device=residual_dot.device)
            sd_std = torch.tensor(self.norm_stats["state_dot_std"], dtype=torch.float64, device=residual_dot.device)
            return residual_dot.double() * sd_std + sd_mean
        return residual_dot.double()

    def _residual_dynamics(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Kinematic + Residual dynamics.

        Args:
            state: (..., 9) float64
            control: (..., 8) float64

        Returns:
            state_dot: (..., 9) float64
        """
        # Kinematic dynamics from differentiable simulator
        kin_dot = self.diff_sim.forward_dynamics(state, control)

        # Residual from learned model
        inp = self._normalize_input(state, control)
        residual_dot_norm = self.residual_model(inp)
        residual_dot = self._denormalize_output(residual_dot_norm)

        return kin_dot + residual_dot

    def _residual_step_rk4(self, state: torch.Tensor, control: torch.Tensor, dt: float) -> torch.Tensor:
        """RK4 step with residual dynamics."""
        k1 = self._residual_dynamics(state, control)
        k2 = self._residual_dynamics(state + 0.5 * dt * k1, control)
        k3 = self._residual_dynamics(state + 0.5 * dt * k2, control)
        k4 = self._residual_dynamics(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def differentiable_rollout(
        self,
        state_0: torch.Tensor,
        controls: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        TBPTT rollout: 매 truncation_length마다 gradient detach.

        Args:
            state_0: (9,) initial state
            controls: (N, 8) control sequence
            dt: timestep

        Returns:
            trajectory: (N+1, 9) with gradient graph
        """
        N = controls.shape[0]
        states = [state_0]
        state = state_0

        for t in range(N):
            # Truncated BPTT: detach every truncation_length steps
            if self.truncation_length > 0 and t > 0 and t % self.truncation_length == 0:
                state = state.detach().requires_grad_(True)

            state = self._residual_step_rk4(state, controls[t], dt)
            states.append(state)

        return torch.stack(states, dim=0)

    def trajectory_loss(
        self,
        trajectory: torch.Tensor,
        ee_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        궤적 추적 비용.

        Args:
            trajectory: (N+1, 9)
            ee_reference: (N+1, 3) EE reference positions

        Returns:
            loss: scalar
        """
        ee_pos = self.diff_sim.forward_kinematics(trajectory)  # (N+1, 3)
        loss = self.ee_weight * ((ee_pos - ee_reference) ** 2).sum(dim=-1).mean()

        # Spectral regularization
        if self.spectral_reg is not None:
            loss = loss + self.spectral_reg.compute_penalty()

        return loss

    def train(
        self,
        train_episodes: List[Dict],
        val_episodes: Optional[List[Dict]] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        BPTT 학습.

        Args:
            train_episodes: list of {"state_0": (9,), "controls": (N, 8), "ee_reference": (N+1, 3)}
            val_episodes: optional validation episodes
            epochs: 학습 에폭
            verbose: 출력

        Returns:
            history: {"train_loss": [...], "val_loss": [...]}
        """
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            # Training
            self.residual_model.train()
            epoch_loss = 0.0

            for ep in train_episodes:
                state_0 = torch.tensor(ep["state_0"], dtype=torch.float64, device=self.device)
                controls = torch.tensor(ep["controls"], dtype=torch.float64, device=self.device)
                ee_ref = torch.tensor(ep["ee_reference"], dtype=torch.float64, device=self.device)

                # Truncate to rollout_horizon if longer
                N = min(controls.shape[0], self.rollout_horizon)
                controls = controls[:N]
                ee_ref = ee_ref[:N + 1]

                self.optimizer.zero_grad()
                trajectory = self.differentiable_rollout(state_0, controls, self.dt)
                loss = self.trajectory_loss(trajectory, ee_ref)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.residual_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= max(len(train_episodes), 1)
            self.history["train_loss"].append(epoch_loss)

            # Validation
            val_loss = 0.0
            if val_episodes:
                self.residual_model.eval()
                with torch.no_grad():
                    for ep in val_episodes:
                        state_0 = torch.tensor(ep["state_0"], dtype=torch.float64, device=self.device)
                        controls = torch.tensor(ep["controls"], dtype=torch.float64, device=self.device)
                        ee_ref = torch.tensor(ep["ee_reference"], dtype=torch.float64, device=self.device)

                        N = min(controls.shape[0], self.rollout_horizon)
                        controls = controls[:N]
                        ee_ref = ee_ref[:N + 1]

                        trajectory = self.differentiable_rollout(state_0, controls, self.dt)
                        loss = self.trajectory_loss(trajectory, ee_ref)
                        val_loss += loss.item()

                val_loss /= max(len(val_episodes), 1)

            self.history["val_loss"].append(val_loss)

            # Early stopping
            if val_episodes:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model("best_bptt_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                val_str = f"Val: {val_loss:.6f}" if val_episodes else "Val: N/A"
                print(f"Epoch {epoch+1}/{epochs} | Train: {epoch_loss:.6f} | {val_str}")

        if verbose:
            print(f"\nBPTT training completed. Final train loss: {self.history['train_loss'][-1]:.6f}")

        return self.history

    def generate_episodes(
        self,
        dyn_model,
        kin_model,
        traj_fn: Callable,
        n_episodes: int = 50,
        duration: float = 3.0,
        mppi_params=None,
        cost_fn=None,
    ) -> List[Dict]:
        """
        MPPI 컨트롤러로 학습 에피소드 생성.

        Args:
            dyn_model: ground truth dynamics (NumPy RobotModel)
            kin_model: kinematic model (for FK reference)
            traj_fn: trajectory function
            n_episodes: 에피소드 수
            duration: 에피소드 길이 (초)
            mppi_params: MPPI 파라미터 (None이면 기본값)
            cost_fn: MPPI 비용 함수 (None이면 기본값)

        Returns:
            episodes: list of {"state_0", "controls", "ee_reference"}
        """
        from mppi_controller.controllers.mppi.mppi_params import MPPIParams
        from mppi_controller.controllers.mppi.base_mppi import MPPIController
        from mppi_controller.controllers.mppi.cost_functions import (
            CompositeMPPICost,
            EndEffector3DTrackingCost,
            EndEffector3DTerminalCost,
            ControlEffortCost,
        )
        from mppi_controller.utils.trajectory import generate_reference_trajectory

        if mppi_params is None:
            mppi_params = MPPIParams(
                N=30, dt=self.dt, K=256, lambda_=0.3,
                sigma=np.array([0.3, 0.3] + [0.8] * 6),
                Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
                R=np.array([0.1, 0.1] + [0.05] * 6),
                Qf=np.array([20.0, 20.0, 2.0] + [0.2] * 6),
            )

        if cost_fn is None:
            cost_fn = CompositeMPPICost([
                EndEffector3DTrackingCost(kin_model, weight=200.0),
                EndEffector3DTerminalCost(kin_model, weight=400.0),
                ControlEffortCost(R=np.array([0.05, 0.05] + [0.02] * 6)),
            ])

        # Use oracle (dyn_model) for planning to get good trajectories
        controller = MPPIController(dyn_model, mppi_params, cost_function=cost_fn)

        n_steps = int(duration / self.dt)
        episodes = []

        for ep_idx in range(n_episodes):
            # Bug fix P0-3: Consistent initial states (moderate diversity)
            state = np.zeros(9)
            state[0] = np.random.uniform(-0.3, 0.3)   # x position
            state[1] = np.random.uniform(-0.3, 0.3)   # y position
            state[2] = np.random.uniform(-0.2, 0.2)   # theta
            state[3:9] = np.random.uniform(-0.3, 0.3, 6)  # joint angles
            state_0 = state.copy()  # Save actual initial state

            controls_list = []
            ee_ref_list = []

            # Random time offset for trajectory variety
            t_offset = np.random.uniform(0, 5.0)

            for step in range(n_steps):
                t = step * self.dt + t_offset
                ref = generate_reference_trajectory(traj_fn, t, mppi_params.N, self.dt)
                ee_ref_list.append(ref[0, :3].copy())

                ctrl, _ = controller.compute_control(state, ref)
                controls_list.append(ctrl.copy())

                # Step with ground truth
                next_state = dyn_model.step(state, ctrl, self.dt)
                state = dyn_model.normalize_state(next_state)

            # Final EE reference
            t_final = n_steps * self.dt + t_offset
            ref = generate_reference_trajectory(traj_fn, t_final, mppi_params.N, self.dt)
            ee_ref_list.append(ref[0, :3].copy())

            # Bug fix P0-3: Use the ACTUAL initial state (consistent with controls)
            episodes.append({
                "state_0": state_0,
                "controls": np.array(controls_list),
                "ee_reference": np.array(ee_ref_list),
            })

        return episodes

    def save_model(self, filename: str):
        """Save residual model."""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict": self.residual_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "norm_stats": self.norm_stats,
            "history": self.history,
            "config": {
                "rollout_horizon": self.rollout_horizon,
                "dt": self.dt,
                "truncation_length": self.truncation_length,
                "ee_weight": self.ee_weight,
            },
        }, filepath)

    def load_model(self, filename: str):
        """Load residual model."""
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.residual_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.norm_stats = checkpoint.get("norm_stats")
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        self.residual_model.eval()
