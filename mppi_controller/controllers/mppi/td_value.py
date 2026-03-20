"""
TD Value Function 학습 모듈 — TD-MPPI 전용

TD(0) 방식으로 상태 가치 함수 V(x)를 점진적으로 학습.
MPPI rollout의 terminal state에 V(x_T)를 추가하여
짧은 호라이즌에서도 장기 비용을 반영.

핵심 수식:
    V(s) ← V(s) + α[c + γV(s') - V(s)]    (TD(0) 업데이트)
    C_total(τ) = Σ_t c(x_t, u_t) + w_V · V(x_T)

Reference: Crestaz et al., RA-L 2026, hal-05213269
"""

import numpy as np
from collections import deque

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ValueNetwork(nn.Module):
    """
    상태 가치 함수 V(x) — MLP

    입력: 상태 x (state_dim,)
    출력: 스칼라 가치 V(x)

    Args:
        state_dim: 상태 차원
        hidden_dims: 은닉층 차원 리스트
    """

    def __init__(self, state_dim: int, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, state_dim) 상태 텐서

        Returns:
            (batch,) 가치 스칼라
        """
        return self.net(x).squeeze(-1)


class TDExperienceBuffer:
    """
    (s, c, s') 경험 버퍼 — TD 학습용 ring buffer

    Args:
        max_size: 최대 버퍼 크기
    """

    def __init__(self, max_size: int = 5000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state: np.ndarray, cost: float, next_state: np.ndarray):
        """경험 추가"""
        self.buffer.append((state.copy(), float(cost), next_state.copy()))

    def sample(self, batch_size: int):
        """미니배치 샘플링

        Args:
            batch_size: 배치 크기

        Returns:
            states: (batch, nx)
            costs: (batch,)
            next_states: (batch, nx)
        """
        indices = np.random.choice(
            len(self.buffer), min(batch_size, len(self.buffer)), replace=False
        )
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch])
        costs = np.array([b[1] for b in batch])
        next_states = np.array([b[2] for b in batch])
        return states, costs, next_states

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """버퍼 초기화"""
        self.buffer.clear()


class TDValueLearner:
    """
    TD(0) Value Function 학습기

    상태 가치 함수 V(x)를 TD(0) 알고리즘으로 학습.

    TD 타겟: y = c + γ·V(s')
    손실:    L = E[(V(s) - y)²]

    Args:
        state_dim: 상태 차원
        hidden_dims: 은닉층 차원
        lr: 학습률
        gamma: 할인율
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims=None,
        lr: float = 0.001,
        gamma: float = 0.99,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TDValueLearner")

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.network = ValueNetwork(state_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.device = "cpu"
        self._update_count = 0

    def predict(self, states: np.ndarray) -> np.ndarray:
        """
        V(x) 예측 (numpy → numpy)

        Args:
            states: (batch, nx) 또는 (nx,) 상태

        Returns:
            (batch,) 또는 () 가치 예측
        """
        was_1d = states.ndim == 1
        if was_1d:
            states = states[np.newaxis, :]

        self.network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(states).to(self.device)
            values = self.network(x).cpu().numpy()

        if was_1d:
            return values[0]
        return values

    def update(
        self,
        states: np.ndarray,
        costs: np.ndarray,
        next_states: np.ndarray,
    ) -> float:
        """
        TD(0) 업데이트

        V(s) ← V(s) + α[c + γV(s') - V(s)]

        Args:
            states: (batch, nx)
            costs: (batch,)
            next_states: (batch, nx)

        Returns:
            loss: TD 손실 스칼라
        """
        self.network.train()

        s = torch.FloatTensor(states).to(self.device)
        c = torch.FloatTensor(costs).to(self.device)
        s_next = torch.FloatTensor(next_states).to(self.device)

        # 현재 가치 예측
        v = self.network(s)

        # TD 타겟 (stop gradient)
        with torch.no_grad():
            v_next = self.network(s_next)

        td_target = c + self.gamma * v_next
        loss = torch.mean((v - td_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        self._update_count += 1

        return float(loss.item())

    @property
    def update_count(self) -> int:
        """총 업데이트 횟수"""
        return self._update_count
