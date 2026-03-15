"""
Pure PyTorch MPPI Controller

CPU/GPU 투명 전환 가능한 순수 PyTorch MPPI.
numpy 변환 오버헤드 없이 전체 파이프라인이 torch.Tensor.

Usage:
    from mppi_controller.controllers.mppi.torch_mppi import TorchMPPIController

    controller = TorchMPPIController(
        dynamics_fn=my_dynamics,   # (state_batch, control) -> next_state_batch
        cost_fn=my_cost,           # (trajs, controls, ref) -> costs
        N=30, K=512, nu=2,
        device="cuda",             # or "cpu"
    )
    control, info = controller.compute_control(state, reference)
"""

import torch
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union


class TorchMPPIController:
    """
    순수 PyTorch MPPI Controller — CPU/GPU 투명 전환

    모든 내부 상태와 연산이 torch.Tensor로 수행.
    dynamics_fn과 cost_fn은 함수형 인터페이스.

    Args:
        dynamics_fn: (state, control) -> next_state
            state: (K, nx) or (nx,), control: (K, nu) -> (K, nx)
        cost_fn: (trajectories, controls, reference) -> costs
            trajectories: (K, N+1, nx), controls: (K, N, nu),
            reference: (N+1, nx) -> (K,)
        N: 호라이즌 길이
        K: 샘플 수
        nu: 제어 차원
        lambda_: 온도 파라미터
        sigma: (nu,) 노이즈 표준편차
        u_min: (nu,) 제어 하한 (None이면 무제한)
        u_max: (nu,) 제어 상한 (None이면 무제한)
        dt: 시간 간격 (dynamics_fn에 전달)
        device: "cpu" 또는 "cuda"
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        N: int,
        K: int,
        nu: int,
        lambda_: float = 1.0,
        sigma=None,
        u_min=None,
        u_max=None,
        dt: float = 0.05,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.N = N
        self.K = K
        self.nu = nu
        self.lambda_ = lambda_
        self.dt = dt
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn

        # 명목 제어 시퀀스 (N, nu)
        self.U = torch.zeros(N, nu, device=self.device)

        # 노이즈 표준편차
        if sigma is None:
            sigma = torch.ones(nu)
        self.sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32)

        # 제어 제약
        if u_min is not None:
            self.u_min = torch.as_tensor(u_min, device=self.device, dtype=torch.float32)
        else:
            self.u_min = None

        if u_max is not None:
            self.u_max = torch.as_tensor(u_max, device=self.device, dtype=torch.float32)
        else:
            self.u_max = None

        # 메트릭
        self.last_info = {}

    def _to_tensor(self, x) -> torch.Tensor:
        """numpy/list → torch.Tensor 변환 (이미 Tensor이면 device 이동만)"""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(np.asarray(x), device=self.device, dtype=torch.float32)

    def compute_control(
        self,
        state,
        reference_trajectory,
    ) -> Tuple[np.ndarray, Dict]:
        """
        MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태 (np.ndarray 또는 torch.Tensor)
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) np.ndarray 최적 제어
            info: dict 디버깅/시각화 정보
        """
        state_t = self._to_tensor(state)
        ref_t = self._to_tensor(reference_trajectory)

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.sigma * torch.randn(
            self.K, self.N, self.nu, device=self.device
        )

        # 2. 제어 시퀀스 (K, N, nu)
        controls = self.U.unsqueeze(0) + noise
        if self.u_min is not None and self.u_max is not None:
            controls = torch.clamp(controls, self.u_min, self.u_max)

        # 3. 롤아웃 (K, N+1, nx)
        trajectories = self._rollout(state_t, controls)

        # 4. 비용 계산 (K,)
        costs = self.cost_fn(trajectories, controls, ref_t)

        # 5. MPPI 가중치 (K,)
        weights = self._compute_weights(costs)

        # 6. ESS
        ess = self._compute_ess(weights)

        # 7. 가중 업데이트
        weighted_noise = (weights[:, None, None] * noise).sum(dim=0)  # (N, nu)
        self.U = self.U + weighted_noise
        if self.u_min is not None and self.u_max is not None:
            self.U = torch.clamp(self.U, self.u_min, self.u_max)

        # 8. Receding horizon
        optimal_control = self.U[0].clone()
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = 0.0

        if self.u_min is not None and self.u_max is not None:
            optimal_control = torch.clamp(optimal_control, self.u_min, self.u_max)

        # 9. info dict (numpy 변환)
        best_idx = torch.argmin(costs).item()
        info = {
            "sample_trajectories": trajectories.detach().cpu().numpy(),
            "sample_controls": controls.detach().cpu().numpy(),
            "sample_weights": weights.detach().cpu().numpy(),
            "best_trajectory": trajectories[best_idx].detach().cpu().numpy(),
            "best_cost": costs[best_idx].item(),
            "mean_cost": costs.mean().item(),
            "temperature": self.lambda_,
            "ess": ess,
            "num_samples": self.K,
        }
        self.last_info = info

        return optimal_control.detach().cpu().numpy(), info

    def _rollout(self, state: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """
        배치 롤아웃

        Args:
            state: (nx,) 초기 상태
            controls: (K, N, nu) 제어 시퀀스

        Returns:
            trajectories: (K, N+1, nx) 궤적
        """
        K, N, nu = controls.shape
        nx = state.shape[-1]

        trajectories = torch.zeros(K, N + 1, nx, device=self.device)
        trajectories[:, 0] = state  # 브로드캐스트

        current_state = state.unsqueeze(0).expand(K, -1).clone()  # (K, nx)

        for t in range(N):
            current_state = self.dynamics_fn(current_state, controls[:, t], self.dt)
            trajectories[:, t + 1] = current_state

        return trajectories

    def _compute_weights(self, costs: torch.Tensor) -> torch.Tensor:
        """
        MPPI softmax 가중치

        Args:
            costs: (K,)

        Returns:
            weights: (K,) 합=1
        """
        min_cost = costs.min()
        exp_costs = torch.exp(-(costs - min_cost) / self.lambda_)
        weights = exp_costs / exp_costs.sum()
        return weights

    def _compute_ess(self, weights: torch.Tensor) -> float:
        """Effective Sample Size"""
        return (1.0 / (weights ** 2).sum()).item()

    def reset(self):
        """제어 시퀀스 초기화"""
        self.U.zero_()
        self.last_info = {}

    def to(self, device: str) -> "TorchMPPIController":
        """디바이스 전환"""
        new_device = torch.device(device)
        self.device = new_device
        self.U = self.U.to(new_device)
        self.sigma = self.sigma.to(new_device)
        if self.u_min is not None:
            self.u_min = self.u_min.to(new_device)
        if self.u_max is not None:
            self.u_max = self.u_max.to(new_device)
        return self

    def __repr__(self) -> str:
        return (
            f"TorchMPPIController(N={self.N}, K={self.K}, nu={self.nu}, "
            f"lambda={self.lambda_}, device={self.device})"
        )
