"""
Pure PyTorch 비용 함수 (함수형 인터페이스)

TorchMPPIController용 비용 함수 팩토리.
모든 연산이 torch.Tensor로 수행되어 CPU/GPU 투명 전환.
"""

import torch
from typing import Callable, List, Optional, Tuple

# 비용 함수 타입: (trajectories, controls, reference) -> costs
# trajectories: (K, N+1, nx), controls: (K, N, nu), reference: (N+1, nx)
# returns: (K,)
TorchCostFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_tracking_cost(
    Q,
    Qf=None,
    R=None,
    device="cpu",
) -> TorchCostFn:
    """
    상태추적 + 터미널 + 제어노력 통합 비용 함수 생성

    Args:
        Q: (nx,) 상태 추적 가중치
        Qf: (nx,) 터미널 가중치 (None이면 Q 사용)
        R: (nu,) 제어 노력 가중치 (None이면 제어 비용 생략)
        device: torch device

    Returns:
        cost_fn: TorchCostFn
    """
    dev = torch.device(device)
    Q_t = torch.as_tensor(Q, device=dev, dtype=torch.float32)
    Qf_t = torch.as_tensor(Qf if Qf is not None else Q, device=dev, dtype=torch.float32)
    R_t = torch.as_tensor(R, device=dev, dtype=torch.float32) if R is not None else None

    def cost_fn(trajectories, controls, reference):
        # State tracking: (K, N, nx)
        errors = trajectories[:, :-1, :] - reference[:-1, :]
        state_cost = torch.sum(errors ** 2 * Q_t, dim=(1, 2))

        # Terminal: (K, nx)
        terminal_error = trajectories[:, -1, :] - reference[-1, :]
        terminal_cost = torch.sum(terminal_error ** 2 * Qf_t, dim=1)

        total = state_cost + terminal_cost

        # Control effort: (K, N, nu)
        if R_t is not None:
            control_cost = torch.sum(controls ** 2 * R_t, dim=(1, 2))
            total = total + control_cost

        return total

    return cost_fn


def make_obstacle_cost(
    obstacles: List[Tuple[float, float, float]],
    safety_margin: float = 0.2,
    cost_weight: float = 500.0,
    device="cpu",
) -> TorchCostFn:
    """
    장애물 회피 비용 함수 생성

    Args:
        obstacles: [(x, y, radius), ...] 장애물 리스트
        safety_margin: 안전 마진 (m)
        cost_weight: 비용 가중치
        device: torch device

    Returns:
        cost_fn: TorchCostFn
    """
    dev = torch.device(device)
    obs_t = torch.tensor(obstacles, device=dev, dtype=torch.float32)  # (M, 3)
    obs_pos = obs_t[:, :2]  # (M, 2)
    obs_rad = obs_t[:, 2]   # (M,)

    def cost_fn(trajectories, controls, reference):
        positions = trajectories[:, :, :2]  # (K, N+1, 2)

        # (K, N+1, 1, 2) - (M, 2) -> (K, N+1, M, 2)
        diff = positions.unsqueeze(2) - obs_pos
        distances = torch.norm(diff, dim=-1)  # (K, N+1, M)

        penetrations = (obs_rad + safety_margin) - distances
        obstacle_costs = torch.where(
            penetrations > 0,
            torch.exp(penetrations * 5.0),
            torch.zeros_like(penetrations),
        )

        return cost_weight * obstacle_costs.sum(dim=(1, 2))  # (K,)

    return cost_fn


def compose_costs(*cost_fns: TorchCostFn) -> TorchCostFn:
    """
    여러 비용 함수를 합성

    Args:
        *cost_fns: 합성할 비용 함수들

    Returns:
        cost_fn: 합성된 TorchCostFn
    """
    def cost_fn(trajectories, controls, reference):
        return sum(fn(trajectories, controls, reference) for fn in cost_fns)

    return cost_fn
