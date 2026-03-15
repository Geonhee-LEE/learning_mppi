"""
Pure PyTorch Kernel MPPI Controller

RBF 커널 보간 기반 차원 축소 MPPI의 순수 PyTorch 구현.
TorchMPPIController를 상속하여 서포트 포인트 공간에서 샘플링.

Usage:
    from mppi_controller.controllers.mppi.torch_kernel_mppi import TorchKernelMPPIController

    controller = TorchKernelMPPIController(
        dynamics_fn=my_dynamics,
        cost_fn=my_cost,
        N=30, K=512, nu=2, S=8,
        device="cuda",
    )
"""

import torch
import numpy as np
from typing import Callable, Dict, Tuple

from mppi_controller.controllers.mppi.torch_mppi import TorchMPPIController


class TorchKernelMPPIController(TorchMPPIController):
    """
    순수 PyTorch Kernel MPPI Controller

    RBF 커널 보간으로 S개 서포트 포인트에서 N개 제어값 복원.
    전체 파이프라인이 torch.Tensor.

    추가 Args:
        S: 서포트 포인트 수 (S << N)
        kernel_bandwidth: RBF 커널 대역폭 (클수록 부드러움)
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        N: int,
        K: int,
        nu: int,
        S: int = 8,
        kernel_bandwidth: float = 2.0,
        lambda_: float = 1.0,
        sigma=None,
        u_min=None,
        u_max=None,
        dt: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__(
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            N=N, K=K, nu=nu,
            lambda_=lambda_,
            sigma=sigma,
            u_min=u_min, u_max=u_max,
            dt=dt, device=device,
        )

        self.S = S
        self.kernel_bandwidth = kernel_bandwidth

        # 서포트 포인트 제어값 (S, nu)
        self.theta = torch.zeros(S, nu, device=self.device)

        # 서포트/쿼리 시간 포인트
        Tk = torch.linspace(0, N - 1, S, device=self.device)
        Hs = torch.arange(N, dtype=torch.float32, device=self.device)

        # 커널 보간 행렬 사전 계산 (N, S)
        self.W = self._precompute_kernel_matrix(Hs, Tk)

        # 통계
        self.kernel_stats_history = []

    def _rbf_kernel(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        RBF 커널 행렬: k(t,t') = exp(-||t-t'||^2 / (2*sigma^2))

        Args:
            t1: (n,), t2: (m,)

        Returns:
            K: (n, m)
        """
        diff = t1[:, None] - t2[None, :]  # (n, m)
        return torch.exp(-diff ** 2 / (2 * self.kernel_bandwidth ** 2))

    def _precompute_kernel_matrix(
        self, Hs: torch.Tensor, Tk: torch.Tensor
    ) -> torch.Tensor:
        """
        W = K_query(N,S) @ inv(K_support(S,S) + eps*I)

        Returns:
            W: (N, S)
        """
        K_support = self._rbf_kernel(Tk, Tk)  # (S, S)
        K_support = K_support + 1e-6 * torch.eye(self.S, device=self.device)

        K_query = self._rbf_kernel(Hs, Tk)  # (N, S)

        # W = K_query @ K_support^{-1}
        W = K_query @ torch.linalg.inv(K_support)
        return W

    def compute_control(
        self,
        state,
        reference_trajectory,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Kernel MPPI 제어 계산 (서포트 공간 샘플링)

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) np.ndarray
            info: dict
        """
        state_t = self._to_tensor(state)
        ref_t = self._to_tensor(reference_trajectory)

        # 1. 서포트 공간 노이즈 (K, S, nu)
        support_noise = self.sigma * torch.randn(
            self.K, self.S, self.nu, device=self.device
        )
        perturbed_theta = self.theta.unsqueeze(0) + support_noise  # (K, S, nu)

        # 2. 커널 보간 → 전체 제어 (K, N, nu)
        #    W: (N, S), perturbed_theta: (K, S, nu) -> (K, N, nu)
        controls = torch.einsum('ts,ksu->ktu', self.W, perturbed_theta)

        if self.u_min is not None and self.u_max is not None:
            controls = torch.clamp(controls, self.u_min, self.u_max)

        # 3. 롤아웃 (K, N+1, nx)
        trajectories = self._rollout(state_t, controls)

        # 4. 비용 (K,)
        costs = self.cost_fn(trajectories, controls, ref_t)

        # 5. MPPI 가중치 (K,)
        weights = self._compute_weights(costs)

        # 6. ESS
        ess = self._compute_ess(weights)

        # 7. 서포트 포인트 가중 업데이트
        weighted_support_noise = (weights[:, None, None] * support_noise).sum(dim=0)
        self.theta = self.theta + weighted_support_noise

        # 8. U 복원
        self.U = self.W @ self.theta  # (N, nu)
        if self.u_min is not None and self.u_max is not None:
            self.U = torch.clamp(self.U, self.u_min, self.u_max)

        # 9. Receding horizon
        optimal_control = self.U[0].clone()
        self._shift_theta()
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = 0.0

        if self.u_min is not None and self.u_max is not None:
            optimal_control = torch.clamp(optimal_control, self.u_min, self.u_max)

        # 10. 커널 통계
        theta_variance = perturbed_theta.var(dim=0).mean().item()
        kernel_stats = {
            "num_support_pts": self.S,
            "kernel_bandwidth": self.kernel_bandwidth,
            "theta_variance": theta_variance,
        }
        self.kernel_stats_history.append(kernel_stats)

        # 11. info
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
            "support_theta": self.theta.detach().cpu().numpy().copy(),
            "kernel_stats": kernel_stats,
        }
        self.last_info = info

        return optimal_control.detach().cpu().numpy(), info

    def _shift_theta(self):
        """Receding horizon 서포트 포인트 시프트"""
        self.theta = torch.roll(self.theta, -1, dims=0)
        self.theta[-1] = 0.0

    def reset(self):
        """제어 시퀀스 및 서포트 포인트 초기화"""
        super().reset()
        self.theta.zero_()
        self.kernel_stats_history = []

    def to(self, device: str) -> "TorchKernelMPPIController":
        """디바이스 전환"""
        super().to(device)
        self.theta = self.theta.to(self.device)
        self.W = self.W.to(self.device)
        return self

    def __repr__(self) -> str:
        return (
            f"TorchKernelMPPIController(N={self.N}, K={self.K}, nu={self.nu}, "
            f"S={self.S}, bandwidth={self.kernel_bandwidth}, device={self.device})"
        )
