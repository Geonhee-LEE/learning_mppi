"""
NN-Policy (BPTT) 학습 파이프라인

MPPI 없이 NN이 직접 (state, ee_reference) → control을 출력하는 정책.
Behavioral Cloning (MPPI 시연) + BPTT Fine-tune 방식.

Phase 1: MPPI oracle로 에피소드 생성 → (state, ee_ref) → control 데이터 수집
Phase 2: MSE Behavioral Cloning (지도 학습)
Phase 3: 미분가능 시뮬레이터를 통한 BPTT fine-tune (궤적 loss)

Reference:
    Pan et al. (2025) "Learning on the Fly" (UZH)

Usage:
    trainer = NNPolicyTrainer(state_dim=9, ee_ref_dim=3, control_dim=8)
    episodes = trainer.generate_demonstrations(dyn_model, kin_model, traj_fn, 80, 4.0)
    trainer.train_bc(episodes, epochs=100)
    trainer.train_bptt(episodes, epochs=50)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Callable
import os
from pathlib import Path


class PolicyMLP(nn.Module):
    """
    (state, ee_ref) → control 정책 네트워크.

    Input: state(9) + ee_ref(3) = 12차원
    Output: control(8)차원
    tanh 출력 + control bounds 스케일링으로 제어 범위 보장.

    Args:
        input_dim: state_dim + ee_ref_dim (default 12)
        output_dim: control_dim (default 8)
        hidden_dims: hidden layer sizes
        control_bounds: (8,) max absolute control values for tanh scaling
    """

    def __init__(
        self,
        input_dim: int = 12,
        output_dim: int = 8,
        hidden_dims: list = None,
        control_bounds: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        if control_bounds is not None:
            self.register_buffer(
                "control_bounds",
                torch.tensor(control_bounds, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "control_bounds",
                torch.ones(output_dim, dtype=torch.float32),
            )

        # Flag for benchmark identification
        self.is_nn_policy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) = [state, ee_ref]
        Returns:
            control: (..., output_dim) scaled by tanh * bounds
        """
        raw = self.net(x)
        bounds = self.control_bounds.to(dtype=x.dtype, device=x.device)
        return torch.tanh(raw) * bounds


class NNPolicyTrainer:
    """
    NN Policy 학습기: Behavioral Cloning + BPTT Fine-tune.

    Args:
        state_dim: state 차원 (default 9)
        ee_ref_dim: EE reference 차원 (default 3)
        control_dim: control 차원 (default 8)
        hidden_dims: MLP hidden layer sizes
        control_bounds: control bounds for tanh scaling
        learning_rate: 학습률
        device: 'cpu' or 'cuda'
        save_dir: 모델 저장 디렉터리
    """

    def __init__(
        self,
        state_dim: int = 9,
        ee_ref_dim: int = 3,
        control_dim: int = 8,
        hidden_dims: list = None,
        control_bounds: Optional[np.ndarray] = None,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        save_dir: str = "models/learned_models/6dof_benchmark/nn_policy",
    ):
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.state_dim = state_dim
        self.ee_ref_dim = ee_ref_dim
        self.control_dim = control_dim
        self.hidden_dims = hidden_dims
        self.device = torch.device(device)
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if control_bounds is None:
            control_bounds = np.array([1.0, 2.0] + [3.0] * 6)

        self.control_bounds_np = control_bounds

        self.model = PolicyMLP(
            input_dim=state_dim + ee_ref_dim,
            output_dim=control_dim,
            hidden_dims=hidden_dims,
            control_bounds=control_bounds,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Differentiable simulator (lazy init)
        self._diff_sim = None

    @property
    def diff_sim(self):
        if self._diff_sim is None:
            from mppi_controller.models.differentiable.diff_sim_6dof import (
                DifferentiableMobileManipulator6DOF,
            )
            self._diff_sim = DifferentiableMobileManipulator6DOF().to(self.device)
        return self._diff_sim

    def generate_demonstrations(
        self,
        dyn_model,
        kin_model,
        traj_fn: Callable,
        n_episodes: int = 80,
        duration: float = 4.0,
        dt: float = 0.05,
    ) -> List[Dict]:
        """
        MPPI oracle로 (state, ee_ref, control) 데이터 수집.

        Args:
            dyn_model: ground truth dynamics
            kin_model: kinematic model (for FK)
            traj_fn: trajectory function
            n_episodes: 에피소드 수
            duration: 에피소드 길이 (초)
            dt: timestep

        Returns:
            episodes: list of {"states": (T,9), "ee_refs": (T,3), "controls": (T,8)}
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

        mppi_params = MPPIParams(
            N=30, dt=dt, K=256, lambda_=0.3,
            sigma=np.array([0.3, 0.3] + [0.8] * 6),
            Q=np.array([10.0, 10.0, 1.0] + [0.1] * 6),
            R=np.array([0.1, 0.1] + [0.05] * 6),
            Qf=np.array([20.0, 20.0, 2.0] + [0.2] * 6),
        )

        cost_fn = CompositeMPPICost([
            EndEffector3DTrackingCost(kin_model, weight=200.0),
            EndEffector3DTerminalCost(kin_model, weight=400.0),
            ControlEffortCost(R=np.array([0.05, 0.05] + [0.02] * 6)),
        ])

        controller = MPPIController(dyn_model, mppi_params, cost_function=cost_fn)

        n_steps = int(duration / dt)
        episodes = []

        for ep_idx in range(n_episodes):
            state = np.zeros(9)
            state[0] = np.random.uniform(-0.3, 0.3)
            state[1] = np.random.uniform(-0.3, 0.3)
            state[2] = np.random.uniform(-0.2, 0.2)
            state[3:9] = np.random.uniform(-0.3, 0.3, 6)

            states_list = []
            ee_refs_list = []
            controls_list = []

            t_offset = np.random.uniform(0, 5.0)

            for step in range(n_steps):
                t = step * dt + t_offset
                ref = generate_reference_trajectory(traj_fn, t, mppi_params.N, dt)
                ee_ref = ref[0, :3].copy()

                states_list.append(state.copy())
                ee_refs_list.append(ee_ref)

                ctrl, _ = controller.compute_control(state, ref)
                controls_list.append(ctrl.copy())

                next_state = dyn_model.step(state, ctrl, dt)
                state = dyn_model.normalize_state(next_state)

            episodes.append({
                "states": np.array(states_list),
                "ee_refs": np.array(ee_refs_list),
                "controls": np.array(controls_list),
            })

        return episodes

    def train_bc(
        self,
        episodes: List[Dict],
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> Dict:
        """
        Behavioral Cloning: MSE loss on ||policy(s, ref) - mppi_control||^2.

        Args:
            episodes: list of {"states": (T,9), "ee_refs": (T,3), "controls": (T,8)}
            epochs: 학습 에폭
            batch_size: 미니배치 크기
            verbose: 출력

        Returns:
            history: {"bc_loss": [...]}
        """
        # Flatten episodes into dataset
        all_states = np.concatenate([ep["states"] for ep in episodes], axis=0)
        all_ee_refs = np.concatenate([ep["ee_refs"] for ep in episodes], axis=0)
        all_controls = np.concatenate([ep["controls"] for ep in episodes], axis=0)

        inputs = np.concatenate([all_states, all_ee_refs], axis=1)  # (N, 12)
        targets = all_controls  # (N, 8)

        inputs_t = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

        n_samples = inputs_t.shape[0]
        history = {"bc_loss": []}

        self.model.train()

        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                inp = inputs_t[idx]
                tgt = targets_t[idx]

                self.optimizer.zero_grad()
                pred = self.model(inp)
                loss = nn.functional.mse_loss(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            history["bc_loss"].append(epoch_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  BC Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

        if verbose:
            print(f"  BC training complete. Final loss: {history['bc_loss'][-1]:.6f}")

        return history

    def _policy_rollout(
        self,
        state_0: torch.Tensor,
        ee_refs: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        정책이 직접 control 생성 → 시뮬레이터로 rollout (미분가능).

        Args:
            state_0: (9,) initial state (float64)
            ee_refs: (N, 3) EE references (float64)
            dt: timestep

        Returns:
            trajectory: (N+1, 9) with gradient graph
        """
        N = ee_refs.shape[0]
        states = [state_0]
        state = state_0

        for t in range(N):
            # Concat state + ee_ref as policy input
            inp = torch.cat([state, ee_refs[t]], dim=-1)  # (12,)
            # Policy runs in float32
            inp_f32 = inp.float()
            control = self.model(inp_f32).double()  # (8,) → float64 for sim

            # Differentiable step (kinematic sim)
            state = self.diff_sim.step_rk4(state, control, dt)
            states.append(state)

        return torch.stack(states, dim=0)  # (N+1, 9)

    def train_bptt(
        self,
        episodes: List[Dict],
        epochs: int = 50,
        rollout_horizon: int = 20,
        dt: float = 0.05,
        verbose: bool = True,
    ) -> Dict:
        """
        BPTT: 미분가능 rollout → trajectory loss (FK EE error).

        Args:
            episodes: list of {"states": (T,9), "ee_refs": (T,3), "controls": (T,8)}
            epochs: 학습 에폭
            rollout_horizon: rollout 길이
            dt: timestep
            verbose: 출력

        Returns:
            history: {"bptt_loss": [...]}
        """
        history = {"bptt_loss": []}

        # Lower learning rate for fine-tuning
        bptt_optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for ep in episodes:
                states_np = ep["states"]
                ee_refs_np = ep["ee_refs"]
                T = states_np.shape[0]

                # Sample random starting point within episode
                max_start = max(0, T - rollout_horizon)
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                end = min(start + rollout_horizon, T)

                state_0 = torch.tensor(
                    states_np[start], dtype=torch.float64, device=self.device
                )
                ee_refs_t = torch.tensor(
                    ee_refs_np[start:end], dtype=torch.float64, device=self.device
                )

                bptt_optimizer.zero_grad()

                traj = self._policy_rollout(state_0, ee_refs_t, dt)

                # Trajectory loss: FK EE position error
                ee_pos = self.diff_sim.forward_kinematics(traj[1:])  # (N, 3)
                loss = ((ee_pos - ee_refs_t) ** 2).sum(dim=-1).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                bptt_optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= max(len(episodes), 1)
            history["bptt_loss"].append(epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  BPTT Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

        if verbose:
            print(f"  BPTT training complete. Final loss: {history['bptt_loss'][-1]:.6f}")

        return history

    def compute_control(self, state: np.ndarray, ee_ref: np.ndarray) -> np.ndarray:
        """
        정책에서 제어 입력 계산 (inference).

        Args:
            state: (9,) current state
            ee_ref: (3,) EE reference position

        Returns:
            control: (8,) control
        """
        self.model.eval()
        with torch.no_grad():
            inp = np.concatenate([state, ee_ref])
            inp_t = torch.tensor(inp, dtype=torch.float32, device=self.device)
            control = self.model(inp_t).cpu().numpy()
        return control

    def save_model(self, filename: str = "best_model.pth"):
        """Save policy model."""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "state_dim": self.state_dim,
                "ee_ref_dim": self.ee_ref_dim,
                "control_dim": self.control_dim,
                "hidden_dims": self.hidden_dims,
            },
            "control_bounds": self.control_bounds_np,
        }, filepath)

    def load_model(self, filename: str = "best_model.pth"):
        """Load policy model."""
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
