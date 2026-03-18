#!/usr/bin/env python3
"""
World Model VAE 학습 파이프라인

VAE (Variational Autoencoder) + Latent Dynamics로
잠재 공간에서의 상태 전이 모델링.

Architecture:
    Encoder:  x → hidden → (mu, log_var) → z  (reparameterization trick)
    Latent Dynamics:  (z, u) → z_next  (이산시간 직접 예측)
    Decoder:  z → hidden → x_pred

Loss = L_recon + beta * L_kl + alpha_dyn * L_dynamics

References:
    Watter et al. "Embed to Control" (NeurIPS 2015)
    Hafner et al. "Learning Latent Dynamics for Planning" (ICML 2019)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


class WorldModelVAE(nn.Module):
    """
    VAE + Latent Dynamics: 잠재 공간 상태 전이 모델

    3개 서브네트워크:
    - Encoder: state → (mu, log_var) → z
    - Latent Dynamics: (z, control) → z_next
    - Decoder: z → state_pred

    Args:
        state_dim: 상태 벡터 차원 (nx)
        control_dim: 제어 벡터 차원 (nu)
        latent_dim: 잠재 공간 차원
        hidden_dims: 히든 레이어 차원 리스트
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = [128, 128],
    ):
        super().__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Encoder: state → hidden → (mu, log_var)
        enc_layers = []
        prev_dim = state_dim
        for hdim in hidden_dims:
            enc_layers.append(nn.Linear(prev_dim, hdim))
            enc_layers.append(nn.ReLU())
            prev_dim = hdim
        self.encoder_trunk = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        # Latent Dynamics: (z, control) → z_next
        dyn_layers = []
        prev_dim = latent_dim + control_dim
        for hdim in hidden_dims:
            dyn_layers.append(nn.Linear(prev_dim, hdim))
            dyn_layers.append(nn.ReLU())
            prev_dim = hdim
        dyn_layers.append(nn.Linear(prev_dim, latent_dim))
        self.latent_dynamics_net = nn.Sequential(*dyn_layers)

        # Decoder: z → hidden → state_pred
        dec_layers = []
        prev_dim = latent_dim
        for hdim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, hdim))
            dec_layers.append(nn.ReLU())
            prev_dim = hdim
        dec_layers.append(nn.Linear(prev_dim, state_dim))
        self.decoder_net = nn.Sequential(*dec_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encoder: state → (mu, log_var)

        Args:
            state: (batch, nx) 또는 (nx,)

        Returns:
            mu: (batch, latent_dim)
            log_var: (batch, latent_dim) — clamped to [-20, 2]
        """
        h = self.encoder_trunk(state)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        log_var = torch.clamp(log_var, min=-20.0, max=2.0)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps

        Training 모드에서만 샘플링, eval 모드에서는 mu 반환.

        Args:
            mu: (batch, latent_dim)
            log_var: (batch, latent_dim)

        Returns:
            z: (batch, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decoder: z → state_pred

        Args:
            z: (batch, latent_dim) 또는 (latent_dim,)

        Returns:
            state_pred: (batch, nx) 또는 (nx,)
        """
        return self.decoder_net(z)

    def latent_step(
        self, z: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        """
        Residual latent dynamics: z_next = z + f(z, u)

        Residual 연결로 항등 매핑이 기본값 → 수축 방지 + 안정적 장거리 전파.

        Args:
            z: (batch, latent_dim)
            control: (batch, nu)

        Returns:
            z_next: (batch, latent_dim)
        """
        zc = torch.cat([z, control], dim=-1)
        return z + self.latent_dynamics_net(zc)

    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        학습용 full forward pass

        Args:
            state: (batch, nx) 현재 상태
            control: (batch, nu) 제어 입력
            next_state: (batch, nx) 다음 상태

        Returns:
            dict with keys:
                recon: (batch, nx) 재구성된 상태
                mu: (batch, latent_dim)
                log_var: (batch, latent_dim)
                z: (batch, latent_dim)
                z_next_pred: (batch, latent_dim) — dynamics 예측
                z_next_target: (batch, latent_dim) — encode(next_state).mu
        """
        # Encode current state
        mu, log_var = self.encode(state)
        z = self.reparameterize(mu, log_var)

        # Reconstruct current state
        recon = self.decode(z)

        # Latent dynamics prediction
        z_next_pred = self.latent_step(z, control)

        # Target: encode next_state (detached — no gradient through target)
        with torch.no_grad():
            z_next_target_mu, _ = self.encode(next_state)

        return {
            "recon": recon,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "z_next_pred": z_next_pred,
            "z_next_target": z_next_target_mu,
        }


class WorldModelTrainer:
    """
    World Model VAE 학습 파이프라인

    Loss = L_recon + beta * L_kl + alpha_dyn * L_dynamics

    - L_recon = ||decode(encode(x)) - x||^2
    - L_kl = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    - L_dynamics = ||latent_step(z_t, u_t) - encode(x_{t+1}).mu.detach()||^2

    Beta annealing: beta_eff = beta * min(epoch / annealing_epochs, 1.0)

    사용 예시:
        trainer = WorldModelTrainer(state_dim=3, control_dim=2)
        history = trainer.train(states, controls, next_states, norm_stats)
        pred = trainer.predict(state, control)
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = [128, 128],
        beta: float = 0.001,
        alpha_dyn: float = 1.0,
        multistep_horizon: int = 1,
        alpha_multistep: float = 1.0,
        learning_rate: float = 1e-3,
        annealing_epochs: int = 50,
        device: str = "cpu",
        save_dir: str = "models/learned_models",
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.alpha_dyn = alpha_dyn
        self.multistep_horizon = multistep_horizon
        self.alpha_multistep = alpha_multistep
        self.annealing_epochs = annealing_epochs
        self.device = torch.device(device)

        self.model = WorldModelVAE(
            state_dim=state_dim,
            control_dim=control_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "dynamics_loss": [],
        }

        self.norm_stats: Optional[Dict[str, np.ndarray]] = None
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def _compute_loss(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        next_state: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        VAE 총 손실 계산

        Returns:
            total_loss: scalar
            loss_dict: 각 손실 구성요소
        """
        out = self.model(state, control, next_state)

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(out["recon"], state, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            torch.sum(
                1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp(),
                dim=-1,
            )
        )

        # Dynamics loss
        dyn_loss = nn.functional.mse_loss(
            out["z_next_pred"], out["z_next_target"], reduction='mean'
        )

        # Beta annealing
        beta_eff = self.beta * min(epoch / max(self.annealing_epochs, 1), 1.0)

        total = recon_loss + beta_eff * kl_loss + self.alpha_dyn * dyn_loss

        return total, {
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "dynamics": dyn_loss.item(),
            "total": total.item(),
        }

    def _build_sequences(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        horizon: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        연속 시퀀스 데이터를 multi-step 학습용으로 구성

        Returns:
            seq_states: (N_seq, horizon+1, nx)  — 연속 상태 시퀀스
            seq_controls: (N_seq, horizon, nu) — 연속 제어 시퀀스
            또는 None (horizon <= 1이면)
        """
        if horizon <= 1:
            return None

        n = len(states)
        seq_states_list = []
        seq_controls_list = []

        # 연속 구간 추출: next_states[i] ≈ states[i+1]이면 연속
        i = 0
        while i + horizon < n:
            # 연속 여부 확인 (next_states[j] ≈ states[j+1])
            is_continuous = True
            for j in range(i, i + horizon - 1):
                if np.linalg.norm(next_states[j] - states[j + 1]) > 0.01:
                    is_continuous = False
                    i = j + 1
                    break

            if is_continuous:
                # states[i], states[i+1], ..., states[i+horizon]
                s_seq = states[i:i + horizon + 1]  # (horizon+1, nx)
                c_seq = controls[i:i + horizon]     # (horizon, nu)
                seq_states_list.append(s_seq)
                seq_controls_list.append(c_seq)
                i += 1
            # else: i was already advanced in the break above

        if len(seq_states_list) < 10:
            return None

        seq_states = torch.FloatTensor(np.array(seq_states_list)).to(self.device)
        seq_controls = torch.FloatTensor(np.array(seq_controls_list)).to(self.device)

        return seq_states, seq_controls

    def _compute_multistep_loss(
        self,
        seq_states: torch.Tensor,
        seq_controls: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-step rollout loss: 잠재 공간에서 K-step 롤아웃 후 디코딩 오류

        L_multi = (1/K) Σ_t ||decode(z_t) - x_t||²

        z_0 = encode(x_0).mu
        z_{t+1} = latent_step(z_t, u_t)

        Args:
            seq_states: (B, K+1, nx) 연속 상태 시퀀스
            seq_controls: (B, K, nu) 연속 제어 시퀀스

        Returns:
            loss: scalar
        """
        B, K_plus_1, nx = seq_states.shape
        K = K_plus_1 - 1

        # Encode initial state
        mu, _ = self.model.encode(seq_states[:, 0, :])  # (B, latent_dim)
        z = mu

        total_loss = torch.tensor(0.0, device=self.device)

        for t in range(K):
            z = self.model.latent_step(z, seq_controls[:, t, :])
            decoded = self.model.decode(z)  # (B, nx)
            step_loss = nn.functional.mse_loss(decoded, seq_states[:, t + 1, :])
            total_loss = total_loss + step_loss

        return total_loss / K

    def train(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 200,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        World Model VAE 학습

        Args:
            states: (N, nx) 현재 상태
            controls: (N, nu) 제어 입력
            next_states: (N, nx) 다음 상태
            norm_stats: 정규화 통계 (Optional)
            epochs: 학습 에폭
            batch_size: 배치 크기
            verbose: 진행 출력

        Returns:
            history: 학습 이력
        """
        self.norm_stats = norm_stats

        states_t = torch.FloatTensor(states).to(self.device)
        controls_t = torch.FloatTensor(controls).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)

        dataset = TensorDataset(states_t, controls_t, next_states_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Multi-step 시퀀스 데이터 준비
        seq_data = self._build_sequences(
            states, controls, next_states, self.multistep_horizon,
        )
        seq_loader = None
        if seq_data is not None:
            seq_states_t, seq_controls_t = seq_data
            seq_dataset = TensorDataset(seq_states_t, seq_controls_t)
            seq_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {"recon": 0.0, "kl": 0.0, "dynamics": 0.0, "total": 0.0}
            num_batches = 0

            for batch_s, batch_c, batch_ns in loader:
                self.optimizer.zero_grad()

                loss, loss_dict = self._compute_loss(
                    batch_s, batch_c, batch_ns, epoch
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                for k, v in loss_dict.items():
                    epoch_losses[k] += v
                num_batches += 1

            # Multi-step loss phase (1-step 학습 안정 후 적용)
            if seq_loader is not None and epoch >= self.annealing_epochs // 2:
                for batch_seq_s, batch_seq_c in seq_loader:
                    self.optimizer.zero_grad()
                    ms_loss = self.alpha_multistep * self._compute_multistep_loss(
                        batch_seq_s, batch_seq_c,
                    )
                    ms_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            # Average
            for k in epoch_losses:
                epoch_losses[k] /= max(num_batches, 1)

            self.history["train_loss"].append(epoch_losses["total"])
            self.history["recon_loss"].append(epoch_losses["recon"])
            self.history["kl_loss"].append(epoch_losses["kl"])
            self.history["dynamics_loss"].append(epoch_losses["dynamics"])

            self.scheduler.step(epoch_losses["total"])

            if verbose and (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Total: {epoch_losses['total']:.6f} | "
                    f"Recon: {epoch_losses['recon']:.6f} | "
                    f"KL: {epoch_losses['kl']:.6f} | "
                    f"Dyn: {epoch_losses['dynamics']:.6f}"
                )

        if verbose:
            print(f"\nTraining completed. Final loss: {self.history['train_loss'][-1]:.6f}")

        return self.history

    def predict(
        self, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        상태 전이 예측: encode → latent_step → decode

        Args:
            state: (nx,) 또는 (batch, nx)
            control: (nu,) 또는 (batch, nu)

        Returns:
            next_state: (nx,) 또는 (batch, nx)
        """
        self.model.eval()
        single = state.ndim == 1
        if single:
            state = state[np.newaxis, :]
            control = control[np.newaxis, :]

        state_t = torch.FloatTensor(state).to(self.device)
        control_t = torch.FloatTensor(control).to(self.device)

        with torch.no_grad():
            mu, _ = self.model.encode(state_t)
            z_next = self.model.latent_step(mu, control_t)
            pred = self.model.decode(z_next)

        result = pred.cpu().numpy()
        if single:
            result = result.squeeze(0)
        return result

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        상태를 잠재 벡터로 인코딩 (mu 반환)

        Args:
            state: (nx,) 또는 (batch, nx)

        Returns:
            z: (latent_dim,) 또는 (batch, latent_dim)
        """
        self.model.eval()
        single = state.ndim == 1
        if single:
            state = state[np.newaxis, :]

        state_t = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            mu, _ = self.model.encode(state_t)

        result = mu.cpu().numpy()
        if single:
            result = result.squeeze(0)
        return result

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        잠재 벡터를 상태로 디코딩

        Args:
            z: (latent_dim,) 또는 (batch, latent_dim)

        Returns:
            state: (nx,) 또는 (batch, nx)
        """
        self.model.eval()
        single = z.ndim == 1
        if single:
            z = z[np.newaxis, :]

        z_t = torch.FloatTensor(z).to(self.device)

        with torch.no_grad():
            pred = self.model.decode(z_t)

        result = pred.cpu().numpy()
        if single:
            result = result.squeeze(0)
        return result

    def save_model(self, filename: str):
        """모델 + 설정 저장"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "norm_stats": self.norm_stats,
            "history": self.history,
            "config": {
                "model_type": "world_model_vae",
                "state_dim": self.state_dim,
                "control_dim": self.control_dim,
                "latent_dim": self.latent_dim,
                "hidden_dims": self.hidden_dims,
                "beta": self.beta,
                "alpha_dyn": self.alpha_dyn,
                "annealing_epochs": self.annealing_epochs,
            },
        }, filepath)

    def load_model(self, filename: str):
        """모델 로드"""
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.norm_stats = checkpoint.get("norm_stats")
        self.history = checkpoint.get("history", {
            "train_loss": [], "recon_loss": [], "kl_loss": [], "dynamics_loss": [],
        })
        self.model.eval()

    def get_model_summary(self) -> str:
        """모델 요약"""
        num_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"WorldModelVAE(\n"
            f"  State dim: {self.state_dim}\n"
            f"  Control dim: {self.control_dim}\n"
            f"  Latent dim: {self.latent_dim}\n"
            f"  Hidden dims: {self.hidden_dims}\n"
            f"  Total parameters: {num_params:,}\n"
            f"  Beta: {self.beta}, Alpha_dyn: {self.alpha_dyn}\n"
            f"  Device: {self.device}\n"
            f")"
        )
