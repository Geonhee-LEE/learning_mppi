"""
Neural Control Barrier Function (CBF) 학습 파이프라인

MLP로 h(x) barrier function을 학습하여 임의 형상 장애물 대응.
기존 CostFunction/CBFSafetyFilter 인터페이스와 호환.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path


class NeuralCBFNetwork(nn.Module):
    """
    State → h(x) scalar barrier function 네트워크

    h(x) > 0: 안전 영역
    h(x) < 0: 장애물 내부
    h(x) = 0: 경계

    Args:
        state_dim: 상태 차원 (기본 3: x, y, theta)
        hidden_dims: 은닉층 차원
        activation: 활성화 함수 ('softplus', 'relu', 'tanh')
        output_scale: 출력 스케일 (tanh 바운드)
    """

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dims: List[int] = None,
        activation: str = "softplus",
        output_scale: float = 5.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.output_scale = output_scale

        # Activation
        if activation == "softplus":
            act_fn = nn.Softplus()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, state_dim) 상태

        Returns:
            h: (batch, 1) barrier 값
        """
        raw = self.network(x)
        return self.output_scale * torch.tanh(raw)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        ∂h/∂x autograd 계산

        Args:
            x: (batch, state_dim) 상태 (requires_grad 불필요, 내부 설정)

        Returns:
            grad_h: (batch, state_dim) barrier gradient
        """
        x = x.detach().requires_grad_(True)
        h = self.forward(x)
        grad_h = torch.autograd.grad(
            h.sum(), x, create_graph=True
        )[0]
        return grad_h


@dataclass
class NeuralCBFTrainerConfig:
    """Neural CBF 학습 설정"""

    state_dim: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    activation: str = "softplus"
    output_scale: float = 5.0

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 200
    early_stopping_patience: int = 30

    # Loss weights
    safe_loss_weight: float = 1.0
    unsafe_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.5
    gradient_reg_weight: float = 0.01

    # Data generation
    num_safe_samples: int = 5000
    num_unsafe_samples: int = 5000
    num_boundary_samples: int = 2000
    workspace_bounds: Tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0)

    # Infrastructure
    device: str = "cpu"
    save_dir: str = "models/neural_cbf"


class NeuralCBFTrainer:
    """
    Neural CBF 학습 파이프라인

    사용 예시:
        trainer = NeuralCBFTrainer()
        data = trainer.generate_training_data(obstacles=[(2,2,0.5)])
        history = trainer.train(data)
        h_vals = trainer.predict_h(states)
    """

    def __init__(self, config: Optional[NeuralCBFTrainerConfig] = None):
        self.config = config or NeuralCBFTrainerConfig()
        self.device = torch.device(self.config.device)

        self.network = NeuralCBFNetwork(
            state_dim=self.config.state_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            output_scale=self.config.output_scale,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def generate_training_data(
        self,
        obstacles: List[tuple],
        safety_margin: float = 0.1,
        non_convex_regions: Optional[List[Callable[[np.ndarray], bool]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        학습 데이터 생성

        Args:
            obstacles: List of (x, y, radius) 원형 장애물
            safety_margin: 경계 샘플 두께
            non_convex_regions: List of callables — fn(state) → True if unsafe

        Returns:
            dict: safe_states, unsafe_states, boundary_states
        """
        cfg = self.config
        x_min, x_max, y_min, y_max = cfg.workspace_bounds

        def is_in_circular_obstacle(pts_2d):
            """pts_2d: (N, 2) → bool mask (N,)"""
            inside = np.zeros(len(pts_2d), dtype=bool)
            for ox, oy, r in obstacles:
                dist = np.sqrt((pts_2d[:, 0] - ox) ** 2 + (pts_2d[:, 1] - oy) ** 2)
                inside |= dist <= r
            return inside

        def is_in_non_convex(pts):
            """pts: (N, state_dim) → bool mask (N,)"""
            if non_convex_regions is None:
                return np.zeros(len(pts), dtype=bool)
            inside = np.zeros(len(pts), dtype=bool)
            for region_fn in non_convex_regions:
                for i in range(len(pts)):
                    if region_fn(pts[i]):
                        inside[i] = True
            return inside

        def is_unsafe(pts):
            """Combined check"""
            circ = is_in_circular_obstacle(pts[:, :2])
            nc = is_in_non_convex(pts)
            return circ | nc

        # --- Safe samples ---
        safe_states = []
        attempts = 0
        while len(safe_states) < cfg.num_safe_samples and attempts < 50:
            n_gen = min(cfg.num_safe_samples * 3, 50000)
            candidates = np.column_stack([
                np.random.uniform(x_min, x_max, n_gen),
                np.random.uniform(y_min, y_max, n_gen),
                np.random.uniform(-np.pi, np.pi, n_gen),
            ])
            if cfg.state_dim > 3:
                extra = np.zeros((n_gen, cfg.state_dim - 3))
                candidates = np.column_stack([candidates, extra])
            mask = ~is_unsafe(candidates)
            safe_states.append(candidates[mask])
            attempts += 1
        safe_states = np.concatenate(safe_states, axis=0)[: cfg.num_safe_samples]

        # --- Unsafe samples ---
        unsafe_states = []
        attempts = 0

        # Circular obstacles: polar sampling
        for ox, oy, r in obstacles:
            n_per_obs = cfg.num_unsafe_samples // max(
                len(obstacles) + (len(non_convex_regions) if non_convex_regions else 0),
                1,
            )
            angles = np.random.uniform(0, 2 * np.pi, n_per_obs)
            radii = np.random.uniform(0, r * 0.95, n_per_obs)
            pts = np.column_stack([
                ox + radii * np.cos(angles),
                oy + radii * np.sin(angles),
                np.random.uniform(-np.pi, np.pi, n_per_obs),
            ])
            if cfg.state_dim > 3:
                extra = np.zeros((n_per_obs, cfg.state_dim - 3))
                pts = np.column_stack([pts, extra])
            unsafe_states.append(pts)

        # Non-convex regions: rejection sampling
        if non_convex_regions:
            remaining = cfg.num_unsafe_samples - sum(len(u) for u in unsafe_states)
            if remaining > 0:
                nc_states = []
                nc_attempts = 0
                while len(nc_states) < remaining and nc_attempts < 100:
                    n_gen = remaining * 5
                    candidates = np.column_stack([
                        np.random.uniform(x_min, x_max, n_gen),
                        np.random.uniform(y_min, y_max, n_gen),
                        np.random.uniform(-np.pi, np.pi, n_gen),
                    ])
                    if cfg.state_dim > 3:
                        extra = np.zeros((n_gen, cfg.state_dim - 3))
                        candidates = np.column_stack([candidates, extra])
                    nc_mask = is_in_non_convex(candidates)
                    if np.any(nc_mask):
                        nc_states.append(candidates[nc_mask])
                    nc_attempts += 1
                if nc_states:
                    unsafe_states.append(
                        np.concatenate(nc_states, axis=0)[:remaining]
                    )

        unsafe_states = np.concatenate(unsafe_states, axis=0)[: cfg.num_unsafe_samples]

        # --- Boundary samples ---
        boundary_states = []
        eps = safety_margin if safety_margin > 0 else 0.05

        # Circular obstacle boundaries
        for ox, oy, r in obstacles:
            n_per_obs = cfg.num_boundary_samples // max(len(obstacles), 1)
            angles = np.random.uniform(0, 2 * np.pi, n_per_obs)
            radii = np.random.uniform(r - eps, r + eps, n_per_obs)
            pts = np.column_stack([
                ox + radii * np.cos(angles),
                oy + radii * np.sin(angles),
                np.random.uniform(-np.pi, np.pi, n_per_obs),
            ])
            if cfg.state_dim > 3:
                extra = np.zeros((n_per_obs, cfg.state_dim - 3))
                pts = np.column_stack([pts, extra])
            boundary_states.append(pts)

        # Non-convex region boundaries (approximate via safe/unsafe transition)
        if non_convex_regions:
            remaining = cfg.num_boundary_samples - sum(
                len(b) for b in boundary_states
            )
            if remaining > 0:
                n_gen = remaining * 10
                candidates = np.column_stack([
                    np.random.uniform(x_min, x_max, n_gen),
                    np.random.uniform(y_min, y_max, n_gen),
                    np.random.uniform(-np.pi, np.pi, n_gen),
                ])
                if cfg.state_dim > 3:
                    extra = np.zeros((n_gen, cfg.state_dim - 3))
                    candidates = np.column_stack([candidates, extra])

                nc_mask = is_in_non_convex(candidates)
                # Find transitions by checking neighbors
                bnd = []
                for i in range(len(candidates)):
                    if len(bnd) >= remaining:
                        break
                    pt = candidates[i].copy()
                    is_nc = nc_mask[i]
                    # Check small perturbations
                    for dx, dy in [(eps, 0), (-eps, 0), (0, eps), (0, -eps)]:
                        neighbor = pt.copy()
                        neighbor[0] += dx
                        neighbor[1] += dy
                        n_nc = is_in_non_convex(neighbor[np.newaxis])[0]
                        if is_nc != n_nc:
                            bnd.append(pt)
                            break
                if bnd:
                    boundary_states.append(np.array(bnd)[:remaining])

        if boundary_states:
            boundary_states = np.concatenate(boundary_states, axis=0)[
                : cfg.num_boundary_samples
            ]
        else:
            boundary_states = np.empty((0, cfg.state_dim))

        return {
            "safe_states": safe_states,
            "unsafe_states": unsafe_states,
            "boundary_states": boundary_states,
        }

    def train(
        self,
        train_data: Optional[Dict[str, np.ndarray]] = None,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        학습 실행

        Args:
            train_data: generate_training_data 결과
            val_split: 검증 데이터 비율
            verbose: 학습 로그 출력

        Returns:
            history: {train_loss, val_loss, safe_acc, unsafe_acc}
        """
        if train_data is None:
            raise ValueError("train_data must be provided")

        cfg = self.config
        margin = 0.3

        # Prepare tensors
        safe_t = torch.FloatTensor(train_data["safe_states"]).to(self.device)
        unsafe_t = torch.FloatTensor(train_data["unsafe_states"]).to(self.device)
        boundary_t = torch.FloatTensor(train_data["boundary_states"]).to(self.device)

        # Train/val split
        n_safe = len(safe_t)
        n_unsafe = len(unsafe_t)
        n_boundary = len(boundary_t)

        n_safe_val = max(int(n_safe * val_split), 1)
        n_unsafe_val = max(int(n_unsafe * val_split), 1)
        n_bnd_val = max(int(n_boundary * val_split), 1)

        perm_s = torch.randperm(n_safe)
        perm_u = torch.randperm(n_unsafe)
        perm_b = torch.randperm(n_boundary)

        safe_train, safe_val = safe_t[perm_s[n_safe_val:]], safe_t[perm_s[:n_safe_val]]
        unsafe_train, unsafe_val = (
            unsafe_t[perm_u[n_unsafe_val:]],
            unsafe_t[perm_u[:n_unsafe_val]],
        )
        boundary_train, boundary_val = (
            boundary_t[perm_b[n_bnd_val:]],
            boundary_t[perm_b[:n_bnd_val]],
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "safe_acc": [],
            "unsafe_acc": [],
        }
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.epochs):
            self.network.train()

            # Mini-batch indices
            idx_s = torch.randperm(len(safe_train))[: cfg.batch_size]
            idx_u = torch.randperm(len(unsafe_train))[: cfg.batch_size]
            idx_b = torch.randperm(len(boundary_train))[
                : min(cfg.batch_size, len(boundary_train))
            ]

            s_batch = safe_train[idx_s]
            u_batch = unsafe_train[idx_u]
            b_batch = boundary_train[idx_b]

            # Forward
            h_safe = self.network(s_batch)  # (B, 1)
            h_unsafe = self.network(u_batch)
            h_boundary = self.network(b_batch)

            # Losses
            # Safe: h > margin
            loss_safe = torch.mean(torch.clamp(margin - h_safe, min=0.0) ** 2)
            # Unsafe: h < -margin
            loss_unsafe = torch.mean(torch.clamp(h_unsafe + margin, min=0.0) ** 2)
            # Boundary: h ≈ 0
            loss_boundary = torch.mean(h_boundary ** 2)

            # Gradient regularization at boundary: ||∂h/∂x|| ≈ 1
            b_grad = b_batch.detach().requires_grad_(True)
            h_b_grad = self.network(b_grad)
            grad_h = torch.autograd.grad(
                h_b_grad.sum(), b_grad, create_graph=True
            )[0]
            grad_norm = torch.norm(grad_h, dim=1)
            loss_grad = torch.mean((grad_norm - 1.0) ** 2)

            total_loss = (
                cfg.safe_loss_weight * loss_safe
                + cfg.unsafe_loss_weight * loss_unsafe
                + cfg.boundary_loss_weight * loss_boundary
                + cfg.gradient_reg_weight * loss_grad
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Validation
            self.network.eval()
            with torch.no_grad():
                h_safe_val = self.network(safe_val)
                h_unsafe_val = self.network(unsafe_val)
                h_bnd_val = self.network(boundary_val)

                val_loss_safe = torch.mean(
                    torch.clamp(margin - h_safe_val, min=0.0) ** 2
                )
                val_loss_unsafe = torch.mean(
                    torch.clamp(h_unsafe_val + margin, min=0.0) ** 2
                )
                val_loss_bnd = torch.mean(h_bnd_val ** 2)

                val_loss = (
                    cfg.safe_loss_weight * val_loss_safe
                    + cfg.unsafe_loss_weight * val_loss_unsafe
                    + cfg.boundary_loss_weight * val_loss_bnd
                )

                safe_acc = float((h_safe_val > 0).float().mean())
                unsafe_acc = float((h_unsafe_val < 0).float().mean())

            history["train_loss"].append(float(total_loss))
            history["val_loss"].append(float(val_loss))
            history["safe_acc"].append(safe_acc)
            history["unsafe_acc"].append(unsafe_acc)

            # Early stopping
            if float(val_loss) < best_val_loss:
                best_val_loss = float(val_loss)
                patience_counter = 0
                self._best_state = {
                    k: v.cpu().clone() for k, v in self.network.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= cfg.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} | "
                    f"loss={float(total_loss):.4f} | "
                    f"val={float(val_loss):.4f} | "
                    f"safe_acc={safe_acc:.3f} | "
                    f"unsafe_acc={unsafe_acc:.3f}"
                )

        # Restore best model
        if hasattr(self, "_best_state"):
            self.network.load_state_dict(self._best_state)

        return history

    def predict_h(self, states: np.ndarray) -> np.ndarray:
        """
        Barrier 값 예측 (NumPy 인터페이스)

        Args:
            states: (N, state_dim) 또는 (state_dim,)

        Returns:
            h: (N,) 또는 scalar
        """
        single = states.ndim == 1
        if single:
            states = states[np.newaxis, :]

        self.network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(states).to(self.device)
            h = self.network(x).cpu().numpy().squeeze(-1)

        return float(h[0]) if single else h

    def predict_gradient(self, states: np.ndarray) -> np.ndarray:
        """
        ∂h/∂x 예측 (NumPy 인터페이스)

        Args:
            states: (N, state_dim) 또는 (state_dim,)

        Returns:
            grad_h: (N, state_dim) 또는 (state_dim,)
        """
        single = states.ndim == 1
        if single:
            states = states[np.newaxis, :]

        self.network.eval()
        x = torch.FloatTensor(states).to(self.device)
        grad_h = self.network.gradient(x).detach().cpu().numpy()

        return grad_h[0] if single else grad_h

    def save_model(self, filename: str):
        """모델 저장"""
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filepath = save_path / filename

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "config": self.config,
            },
            filepath,
        )

    def load_model(self, filename: str):
        """모델 로드"""
        filepath = Path(self.config.save_dir) / filename

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        if "config" in checkpoint:
            self.config = checkpoint["config"]
            self.network = NeuralCBFNetwork(
                state_dim=self.config.state_dim,
                hidden_dims=self.config.hidden_dims,
                activation=self.config.activation,
                output_scale=self.config.output_scale,
            ).to(self.device)

        self.network.load_state_dict(checkpoint["network_state_dict"])

    def get_network(self) -> NeuralCBFNetwork:
        """학습된 네트워크 반환"""
        return self.network
