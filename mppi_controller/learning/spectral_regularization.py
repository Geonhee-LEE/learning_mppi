"""
Spectral Regularization (Power Iteration)

학습 중 각 Linear layer의 최대 특이값(σ_max)을 패널티로 추가하여
모델의 Lipschitz 상수를 제한 → 외삽 안정성 향상.

Reference:
    Miyato et al. (2018) "Spectral Normalization for GANs"
    Pan et al. (2025) "Learning on the Fly" (UZH)

Usage:
    reg = SpectralRegularizer(model, lambda_spectral=0.01)

    # In training loop:
    loss = mse_loss + reg.compute_penalty()
"""

import torch
import torch.nn as nn
from typing import Dict


class SpectralRegularizer:
    """
    Power iteration 기반 Spectral Regularization.

    각 Linear layer의 σ_max(W)를 계산하여
    Σ_i λ · σ_max(W_i) 형태의 differentiable 패널티를 반환.

    Args:
        model: nn.Module (Linear layer 포함)
        lambda_spectral: 정규화 강도 (0 = 비활성)
        n_power_iterations: Power iteration 횟수 (1이면 충분)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_spectral: float = 0.01,
        n_power_iterations: int = 1,
    ):
        self.model = model
        self.lambda_spectral = lambda_spectral
        self.n_power_iterations = n_power_iterations

        # Initialize u vectors for each Linear layer
        self._u_vectors: Dict[str, torch.Tensor] = {}
        self._linear_layers: Dict[str, nn.Linear] = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                d_out = module.weight.shape[0]
                u = torch.randn(d_out, dtype=module.weight.dtype)
                u = u / u.norm()
                self._u_vectors[name] = u
                self._linear_layers[name] = module

    def compute_penalty(self) -> torch.Tensor:
        """
        Compute spectral penalty: Σ_i λ · σ_max(W_i).

        Returns:
            penalty: scalar tensor (differentiable)
        """
        if self.lambda_spectral <= 0 or len(self._linear_layers) == 0:
            # Return a zero tensor on the same device as the model
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, module in self._linear_layers.items():
            sigma = self._power_iteration(module.weight, name)
            penalty = penalty + sigma

        return self.lambda_spectral * penalty

    def _power_iteration(self, W: torch.Tensor, name: str) -> torch.Tensor:
        """
        Power iteration으로 σ_max(W) 근사.

        Args:
            W: (d_out, d_in) weight matrix
            name: layer name (for u vector lookup)

        Returns:
            sigma: σ_max(W) (differentiable scalar)
        """
        u = self._u_vectors[name].to(device=W.device, dtype=W.dtype)

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # v = W^T u / ||W^T u||
                v = W.T @ u
                v = v / (v.norm() + 1e-12)
                # u = W v / ||W v||
                u = W @ v
                u = u / (u.norm() + 1e-12)

            # Store updated u (detached)
            self._u_vectors[name] = u.detach().cpu()

        # σ = u^T W v (differentiable through W)
        u = u.to(W.device)
        v = v.to(W.device)
        sigma = u @ W @ v

        return sigma

    def get_spectral_norms(self) -> Dict[str, float]:
        """각 layer의 현재 σ_max 값 반환 (디버깅용)."""
        norms = {}
        for name, module in self._linear_layers.items():
            with torch.no_grad():
                sigma = self._power_iteration(module.weight, name)
                norms[name] = sigma.item()
        return norms
