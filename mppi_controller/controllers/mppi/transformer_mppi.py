"""
T-MPPI (Transformer-based MPPI) Controller -- 33번째 MPPI 변형

학습된 Transformer로 MPPI 초기 제어 시퀀스를 예측하여
샘플 효율성을 크게 향상. 과거 최적 제어 데이터로 학습된
Transformer가 근최적 시작점을 생성하여, 적은 샘플로도 수렴.

핵심 수식:
    U_init = Transformer(state_history, control_history)
    U* = MPPI(U_init, K_reduced)
    Loss = MSE(U_pred, U_optimal)

기존 변형 대비 핵심 차이:
    - Vanilla MPPI: 영벡터 또는 이전 해로 초기화 -> 낭비적 탐색
    - DIAL-MPPI: 반복으로 탐색 향상 -> 초기화는 미개선
    - Flow-MPPI: 분포 자체를 학습 -> 무거운 ODE 적분 필요
    - T-MPPI: 초기화만 학습 -> 가볍고 표준 MPPI 파이프라인 유지

Reference: Zinage et al., arXiv:2412.17118, Dec 2024
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import TransformerMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler

try:
    import torch
    import torch.nn as nn
    import math

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ================================================================
# Transformer Model
# ================================================================

if HAS_TORCH:

    class PositionalEncoding(nn.Module):
        """
        사인/코사인 위치 인코딩

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """

        def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x):
            """
            Args:
                x: (batch, seq_len, d_model)
            Returns:
                (batch, seq_len, d_model) + PE + dropout
            """
            x = x + self.pe[:, : x.size(1), :]
            return self.dropout(x)

    class ControlTransformer(nn.Module):
        """
        제어 시퀀스 예측 Transformer

        과거 (state, control) 이력을 입력으로 받아
        미래 N 스텝의 제어 시퀀스를 예측.

        Architecture:
            - Input projection: (state_dim + control_dim) -> d_model
            - Positional encoding
            - Encoder-only Transformer (causal mask)
            - Output projection: d_model -> N * control_dim

        Args:
            state_dim: 상태 차원
            control_dim: 제어 차원
            horizon: 예측 호라이즌 N
            d_model: Transformer 은닉 차원
            n_heads: 멀티헤드 어텐션 헤드 수
            n_layers: Transformer 인코더 레이어 수
            dropout: 드롭아웃 비율
            context_length: 최대 컨텍스트 길이
        """

        def __init__(
            self,
            state_dim: int,
            control_dim: int,
            horizon: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
            context_length: int = 20,
        ):
            super().__init__()
            self.state_dim = state_dim
            self.control_dim = control_dim
            self.horizon = horizon
            self.d_model = d_model
            self.context_length = context_length

            input_dim = state_dim + control_dim

            # Input projection
            self.input_proj = nn.Linear(input_dim, d_model)

            # Positional encoding
            self.pos_enc = PositionalEncoding(
                d_model, max_len=context_length + 10, dropout=dropout
            )

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )

            # Output projection: d_model -> N * control_dim
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, horizon * control_dim),
            )

            # Zero-init output layer for graceful degradation
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)

        def forward(self, context):
            """
            Transformer 순전파

            Args:
                context: (batch, seq_len, state_dim + control_dim) 이력

            Returns:
                (batch, N, control_dim) 예측 제어 시퀀스
            """
            batch_size = context.size(0)
            seq_len = context.size(1)

            # Input projection
            x = self.input_proj(context)  # (batch, seq_len, d_model)

            # Positional encoding
            x = self.pos_enc(x)

            # Causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=context.device
            )

            # Transformer encoding
            x = self.transformer_encoder(x, mask=mask)

            # 마지막 토큰의 출력으로 예측
            last_hidden = x[:, -1, :]  # (batch, d_model)

            # Output projection
            output = self.output_proj(last_hidden)  # (batch, N * control_dim)
            output = output.view(batch_size, self.horizon, self.control_dim)

            return output

        def get_encoding(self, context):
            """
            인코딩만 반환 (디버깅/분석용)

            Args:
                context: (batch, seq_len, state_dim + control_dim)

            Returns:
                (batch, seq_len, d_model) 인코딩
            """
            x = self.input_proj(context)
            x = self.pos_enc(x)
            return x


# ================================================================
# Data Buffer
# ================================================================

class TransformerDataBuffer:
    """
    (state_history, control_history, optimal_U) 데이터 버퍼

    Ring buffer 구조로 최근 데이터를 유지.
    각 엔트리는:
        - state_history: (context_length, state_dim) 과거 상태
        - control_history: (context_length, control_dim) 과거 제어
        - optimal_U: (N, control_dim) MPPI 최적 제어 시퀀스

    Args:
        max_size: 최대 버퍼 크기
    """

    def __init__(self, max_size: int = 5000):
        self.buffer = deque(maxlen=max_size)

    def add(
        self,
        state_history: np.ndarray,
        control_history: np.ndarray,
        optimal_U: np.ndarray,
    ):
        """
        데이터 추가

        Args:
            state_history: (context_length, state_dim)
            control_history: (context_length, control_dim)
            optimal_U: (N, control_dim)
        """
        self.buffer.append((
            state_history.copy(),
            control_history.copy(),
            optimal_U.copy(),
        ))

    def sample(self, batch_size: int):
        """
        미니배치 샘플링

        Args:
            batch_size: 배치 크기

        Returns:
            state_histories: (batch, context_length, state_dim)
            control_histories: (batch, context_length, control_dim)
            optimal_Us: (batch, N, control_dim)
        """
        indices = np.random.choice(
            len(self.buffer), min(batch_size, len(self.buffer)), replace=False
        )
        batch = [self.buffer[i] for i in indices]

        state_histories = np.array([b[0] for b in batch])
        control_histories = np.array([b[1] for b in batch])
        optimal_Us = np.array([b[2] for b in batch])

        return state_histories, control_histories, optimal_Us

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """버퍼 초기화"""
        self.buffer.clear()


# ================================================================
# Trainer
# ================================================================

class TransformerTrainer:
    """
    ControlTransformer MSE 학습기

    MSE Loss = E[||U_pred - U_optimal||^2]
    + Gradient clipping for stability.

    Args:
        model: ControlTransformer 인스턴스
        lr: 학습률
    """

    def __init__(self, model, lr: float = 1e-3):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TransformerTrainer")

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self._update_count = 0

    def train_step(
        self,
        state_histories: np.ndarray,
        control_histories: np.ndarray,
        optimal_Us: np.ndarray,
    ) -> float:
        """
        단일 그래디언트 스텝

        Args:
            state_histories: (batch, context_length, state_dim)
            control_histories: (batch, context_length, control_dim)
            optimal_Us: (batch, N, control_dim)

        Returns:
            loss: MSE 손실
        """
        self.model.train()

        # 컨텍스트 구성: [state, control] 연결
        context = np.concatenate(
            [state_histories, control_histories], axis=-1
        )  # (batch, context_length, state_dim + control_dim)

        context_t = torch.FloatTensor(context)
        target_t = torch.FloatTensor(optimal_Us)

        # Forward
        pred = self.model(context_t)  # (batch, N, control_dim)
        loss = self.loss_fn(pred, target_t)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self._update_count += 1

        return float(loss.item())

    @property
    def update_count(self) -> int:
        """총 업데이트 횟수"""
        return self._update_count


# ================================================================
# T-MPPI Controller
# ================================================================

class TransformerMPPIController(MPPIController):
    """
    T-MPPI Controller (33번째 MPPI 변형)

    Transformer 기반 초기화 + 표준 MPPI 최적화.

    동작 흐름:
        1. 상태/제어 이력 업데이트
        2. Transformer로 초기 제어 시퀀스 예측 (학습 충분 시)
        3. blend_ratio로 Transformer 예측과 이전 솔루션 혼합
        4. 표준 MPPI 샘플링 -> rollout -> 가중 평균
        5. 최적 제어 데이터를 버퍼에 저장
        6. 주기적 온라인 학습

    Vanilla MPPI 대비 핵심 차이:
        1. 학습 기반 초기화: 과거 패턴으로부터 좋은 시작점 생성
        2. 온라인 적응: 실시간으로 최적화 패턴 학습
        3. 표준 MPPI 호환: 학습 실패 시 graceful degradation

    Args:
        model: RobotModel 인스턴스
        params: TransformerMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: TransformerMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.transformer_params = params

        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TransformerMPPIController")

        # Transformer model
        self._transformer = ControlTransformer(
            state_dim=model.state_dim,
            control_dim=model.control_dim,
            horizon=params.N,
            d_model=params.transformer_hidden_dim,
            n_heads=params.transformer_n_heads,
            n_layers=params.transformer_n_layers,
            dropout=params.transformer_dropout,
            context_length=params.transformer_context_length,
        )

        # Trainer
        self._trainer = TransformerTrainer(
            self._transformer, lr=params.transformer_lr
        )

        # Data buffer
        self._buffer = TransformerDataBuffer(
            max_size=params.transformer_buffer_size
        )

        # History buffers (rolling window)
        self._state_history = deque(maxlen=params.transformer_context_length)
        self._control_history = deque(maxlen=params.transformer_context_length)

        # Step counter
        self._step_count = 0

        # Statistics
        self._transformer_history = []
        self._transformer_used_count = 0
        self._total_train_loss = 0.0
        self._train_count = 0

        # Load pretrained model if specified
        if params.transformer_model_path is not None:
            self._load_model(params.transformer_model_path)

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        T-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + transformer_stats
        """
        K = self.params.K
        N = self.params.N

        # ── 1. 이력 업데이트 ──
        self._state_history.append(state.copy())

        # ── 2. Transformer 초기화 ──
        transformer_used = False
        transformer_pred = None

        if (
            self.transformer_params.use_transformer_init
            and len(self._state_history) >= 2
            and self._train_count > 0
        ):
            transformer_pred = self._predict_init()
            if transformer_pred is not None:
                # Blend: U_init = blend * transformer + (1-blend) * prev_U
                blend = self.transformer_params.blend_ratio
                self.U = blend * transformer_pred + (1.0 - blend) * self.U
                transformer_used = True

        # ── 3. 표준 MPPI 샘플링 ──
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
        sampled_controls = self.U[None, :, :] + noise  # (K, N, nu)

        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # ── 4. Rollout ──
        trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # ── 5. 비용 계산 ──
        costs = self.cost_function.compute_cost(
            trajectories, sampled_controls, reference_trajectory
        )

        # ── 6. 가중치 + 업데이트 ──
        weights = self._compute_weights(costs, self.params.lambda_)
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 최적 제어 시퀀스 저장 (학습 데이터용, shift 전)
        optimal_U = self.U.copy()

        # ── 7. 데이터 수집 ──
        self._collect_data(optimal_U)

        # 첫 제어 추출
        optimal_control = self.U[0].copy()

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 제어 이력 업데이트 (shift 후)
        self._control_history.append(optimal_control.copy())

        # ── 8. 온라인 학습 ──
        train_loss = 0.0
        if self.transformer_params.online_learning:
            train_loss = self._maybe_train()

        self._step_count += 1

        if transformer_used:
            self._transformer_used_count += 1

        # ── 9. 정보 ──
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        transformer_stats = {
            "transformer_used": transformer_used,
            "transformer_init_ratio": (
                self._transformer_used_count / self._step_count
                if self._step_count > 0
                else 0.0
            ),
            "buffer_size": len(self._buffer),
            "train_count": self._train_count,
            "train_loss": train_loss,
            "mean_train_loss": (
                self._total_train_loss / self._train_count
                if self._train_count > 0
                else 0.0
            ),
            "context_length": len(self._state_history),
        }
        self._transformer_history.append(transformer_stats)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "transformer_stats": transformer_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _predict_init(self) -> Optional[np.ndarray]:
        """
        Transformer 순전파 -> U_init (N, nu)

        Returns:
            (N, nu) 예측 제어 시퀀스 또는 None (예측 불가 시)
        """
        context_len = self.transformer_params.transformer_context_length
        state_dim = self.model.state_dim
        control_dim = self.model.control_dim

        # 이력을 numpy array로 변환
        states = np.array(list(self._state_history))  # (L, nx)
        L = len(states)

        # 제어 이력 (상태보다 1개 적을 수 있음)
        if len(self._control_history) == 0:
            controls = np.zeros((L, control_dim))
        else:
            controls = np.array(list(self._control_history))  # (L', nu)
            # 길이 맞추기
            if len(controls) < L:
                pad = np.zeros((L - len(controls), control_dim))
                controls = np.concatenate([pad, controls], axis=0)
            elif len(controls) > L:
                controls = controls[-L:]

        # 패딩 (이력 < context_length)
        if L < context_len:
            state_pad = np.zeros((context_len - L, state_dim))
            control_pad = np.zeros((context_len - L, control_dim))
            states = np.concatenate([state_pad, states], axis=0)
            controls = np.concatenate([control_pad, controls], axis=0)
        else:
            states = states[-context_len:]
            controls = controls[-context_len:]

        # 컨텍스트: (1, context_length, state_dim + control_dim)
        context = np.concatenate([states, controls], axis=-1)
        context = context[np.newaxis, :]  # (1, context_len, dim)

        # 추론
        self._transformer.eval()
        with torch.no_grad():
            context_t = torch.FloatTensor(context)
            pred_t = self._transformer(context_t)  # (1, N, nu)
            pred = pred_t.squeeze(0).cpu().numpy()  # (N, nu)

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            pred = np.clip(pred, self.u_min, self.u_max)

        return pred

    def _collect_data(self, optimal_U: np.ndarray):
        """
        데이터 수집: (state_history, control_history, optimal_U)

        Args:
            optimal_U: (N, nu) 최적 제어 시퀀스
        """
        context_len = self.transformer_params.transformer_context_length
        state_dim = self.model.state_dim
        control_dim = self.model.control_dim

        # 최소 2개 이력 필요
        if len(self._state_history) < 2:
            return

        states = np.array(list(self._state_history))
        L = len(states)

        if len(self._control_history) == 0:
            controls = np.zeros((L, control_dim))
        else:
            controls = np.array(list(self._control_history))
            if len(controls) < L:
                pad = np.zeros((L - len(controls), control_dim))
                controls = np.concatenate([pad, controls], axis=0)
            elif len(controls) > L:
                controls = controls[-L:]

        # 패딩
        if L < context_len:
            state_pad = np.zeros((context_len - L, state_dim))
            control_pad = np.zeros((context_len - L, control_dim))
            states = np.concatenate([state_pad, states], axis=0)
            controls = np.concatenate([control_pad, controls], axis=0)
        else:
            states = states[-context_len:]
            controls = controls[-context_len:]

        self._buffer.add(states, controls, optimal_U)

    def _maybe_train(self) -> float:
        """
        주기적 온라인 학습

        Returns:
            평균 학습 손실 (학습 안 했으면 0.0)
        """
        if self._step_count % self.transformer_params.transformer_training_interval != 0:
            return 0.0

        if len(self._buffer) < self.transformer_params.transformer_min_samples:
            return 0.0

        total_loss = 0.0
        n_steps = self.transformer_params.transformer_n_train_steps

        for _ in range(n_steps):
            states_b, controls_b, optimal_b = self._buffer.sample(
                self.transformer_params.transformer_batch_size
            )
            loss = self._trainer.train_step(states_b, controls_b, optimal_b)
            total_loss += loss

        avg_loss = total_loss / n_steps
        self._total_train_loss += avg_loss
        self._train_count += 1

        return avg_loss

    def _load_model(self, path: str):
        """사전 학습 모델 로드"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._transformer.load_state_dict(checkpoint["model_state_dict"])
        if "train_count" in checkpoint:
            self._train_count = checkpoint["train_count"]

    def save_model(self, path: str):
        """모델 저장"""
        torch.save(
            {
                "model_state_dict": self._transformer.state_dict(),
                "train_count": self._train_count,
                "step_count": self._step_count,
            },
            path,
        )

    def get_transformer_statistics(self) -> Dict:
        """
        Transformer 학습 누적 통계

        Returns:
            dict: 버퍼 크기, 학습 횟수, 사용 비율 등
        """
        if not self._transformer_history:
            return {
                "total_steps": 0,
                "buffer_size": 0,
                "train_count": 0,
                "transformer_init_ratio": 0.0,
                "mean_train_loss": 0.0,
                "history": [],
            }

        return {
            "total_steps": self._step_count,
            "buffer_size": len(self._buffer),
            "train_count": self._train_count,
            "transformer_init_ratio": (
                self._transformer_used_count / self._step_count
                if self._step_count > 0
                else 0.0
            ),
            "mean_train_loss": (
                self._total_train_loss / self._train_count
                if self._train_count > 0
                else 0.0
            ),
            "history": self._transformer_history.copy(),
        }

    def get_transformer(self):
        """Transformer 모델 접근 (외부 분석용)"""
        return self._transformer

    def get_buffer(self) -> TransformerDataBuffer:
        """데이터 버퍼 접근 (외부 분석용)"""
        return self._buffer

    def reset(self):
        """제어 시퀀스 + 내부 상태 초기화"""
        super().reset()
        self._step_count = 0
        self._state_history.clear()
        self._control_history.clear()
        self._transformer_history = []
        self._transformer_used_count = 0
        # 버퍼와 Transformer 가중치는 유지 (학습 지속)

    def full_reset(self):
        """전체 초기화 (버퍼 + Transformer 포함)"""
        self.reset()
        self._buffer.clear()
        self._train_count = 0
        self._total_train_loss = 0.0

        # Transformer 재초기화
        params = self.transformer_params
        self._transformer = ControlTransformer(
            state_dim=self.model.state_dim,
            control_dim=self.model.control_dim,
            horizon=params.N,
            d_model=params.transformer_hidden_dim,
            n_heads=params.transformer_n_heads,
            n_layers=params.transformer_n_layers,
            dropout=params.transformer_dropout,
            context_length=params.transformer_context_length,
        )
        self._trainer = TransformerTrainer(
            self._transformer, lr=params.transformer_lr
        )

    def __repr__(self) -> str:
        buf_size = len(self._buffer)
        updates = self._train_count
        return (
            f"TransformerMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"N={self.params.N}, "
            f"buffer={buf_size}, "
            f"updates={updates}, "
            f"blend_ratio={self.transformer_params.blend_ratio})"
        )
