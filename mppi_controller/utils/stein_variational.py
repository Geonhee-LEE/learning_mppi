"""
Stein Variational Gradient Descent (SVGD) 유틸리티

SVGD를 위한 RBF 커널 및 관련 함수.
"""

import numpy as np
from typing import Tuple


def rbf_kernel(X: np.ndarray, bandwidth: float = None) -> np.ndarray:
    """
    RBF (Radial Basis Function) 커널 계산

    K(x_i, x_j) = exp(-||x_i - x_j||² / (2 * h²))

    Args:
        X: (K, ...) 샘플 집합 (K개 샘플)
        bandwidth: 대역폭 h (None이면 median heuristic)

    Returns:
        kernel: (K, K) RBF 커널 행렬
    """
    K = X.shape[0]

    # Flatten to (K, D) for pairwise distance
    X_flat = X.reshape(K, -1)

    # Pairwise squared distances: ||x_i - x_j||²
    # (x_i - x_j)² = x_i² + x_j² - 2*x_i*x_j
    sq_norms = np.sum(X_flat**2, axis=1)  # (K,)
    sq_distances = (
        sq_norms[:, None] + sq_norms[None, :] - 2 * X_flat @ X_flat.T
    )  # (K, K)

    # Median heuristic for bandwidth
    if bandwidth is None:
        # Use median of pairwise distances (excluding diagonal)
        sq_dists_no_diag = sq_distances[~np.eye(K, dtype=bool)]
        if len(sq_dists_no_diag) > 0:
            median_sq_dist = np.median(sq_dists_no_diag)
            bandwidth = np.sqrt(median_sq_dist / np.log(K + 1))
        else:
            bandwidth = 1.0

    # RBF kernel
    kernel = np.exp(-sq_distances / (2 * bandwidth**2))

    return kernel


def rbf_kernel_with_bandwidth(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    RBF 커널 + median bandwidth를 한 번의 K² 거리 계산으로 동시 산출.

    Args:
        X: (K, ...) 샘플 집합

    Returns:
        kernel: (K, K) RBF 커널 행렬
        bandwidth: float median heuristic bandwidth
    """
    K = X.shape[0]
    X_flat = X.reshape(K, -1)

    sq_norms = np.sum(X_flat ** 2, axis=1)
    sq_distances = sq_norms[:, None] + sq_norms[None, :] - 2 * X_flat @ X_flat.T

    # Median bandwidth (대각 제외)
    np.fill_diagonal(sq_distances, np.inf)
    sq_dists_off = sq_distances.ravel()
    sq_dists_off = sq_dists_off[sq_dists_off < np.inf]
    if len(sq_dists_off) > 0:
        median_sq_dist = np.median(sq_dists_off)
        bandwidth = np.sqrt(median_sq_dist / np.log(K + 1))
    else:
        bandwidth = 1.0

    # 대각 복원 후 커널 계산
    np.fill_diagonal(sq_distances, 0.0)
    kernel = np.exp(-sq_distances / (2 * bandwidth ** 2))

    return kernel, bandwidth


def rbf_kernel_gradient(
    X: np.ndarray, kernel: np.ndarray, bandwidth: float
) -> np.ndarray:
    """
    RBF 커널의 gradient 계산

    ∇_j K(x_i, x_j) = -K(x_i, x_j) * (x_i - x_j) / h²

    Args:
        X: (K, ...) 샘플 집합
        kernel: (K, K) RBF 커널 행렬
        bandwidth: 대역폭 h

    Returns:
        grad_kernel: (K, K, ...) 커널 gradient
            grad_kernel[i, j] = ∇_j K(x_i, x_j)
    """
    K = X.shape[0]
    original_shape = X.shape[1:]

    # Pairwise differences: x_i - x_j
    # X: (K, ...) → (K, 1, ...) - (1, K, ...) = (K, K, ...)
    X_expanded_i = X[:, None, ...]  # (K, 1, ...)
    X_expanded_j = X[None, :, ...]  # (1, K, ...)
    pairwise_diff = X_expanded_i - X_expanded_j  # (K, K, ...)

    # ∇_j K(x_i, x_j) = -K(x_i, x_j) / h² * (x_i - x_j)
    # kernel: (K, K) → (K, K, 1, ...) for broadcasting
    kernel_expanded = kernel.reshape(K, K, *([1] * len(original_shape)))

    grad_kernel = -kernel_expanded / (bandwidth**2) * pairwise_diff

    return grad_kernel


def compute_svgd_update(
    X: np.ndarray,
    grad_log_prob: np.ndarray,
    kernel: np.ndarray,
    grad_kernel: np.ndarray,
) -> np.ndarray:
    """
    SVGD 업데이트 계산

    Φ(x_i) = (1/K) Σ_j [K(x_i, x_j) * ∇_j log p(x_j) + ∇_j K(x_i, x_j)]

    Args:
        X: (K, ...) 샘플 집합
        grad_log_prob: (K, ...) ∇ log p(x) (비용 gradient)
        kernel: (K, K) RBF 커널
        grad_kernel: (K, K, ...) 커널 gradient

    Returns:
        phi: (K, ...) SVGD 업데이트 방향
    """
    K = X.shape[0]
    original_shape = X.shape[1:]

    # Term 1: K(x_i, x_j) * ∇_j log p(x_j)
    # kernel: (K, K), grad_log_prob: (K, ...)
    # → kernel @ grad_log_prob.reshape(K, -1) → (K, ...)
    grad_log_prob_flat = grad_log_prob.reshape(K, -1)  # (K, D)
    term1 = (kernel @ grad_log_prob_flat).reshape(K, *original_shape)  # (K, ...)

    # Term 2: ∇_j K(x_i, x_j)
    # Sum over j: (K, K, ...) → (K, ...)
    term2 = np.sum(grad_kernel, axis=1)  # (K, ...)

    # SVGD update
    phi = (term1 + term2) / K

    return phi


def compute_svgd_update_efficient(
    X: np.ndarray,
    grad_log_prob: np.ndarray,
    kernel: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    메모리 효율적 SVGD 업데이트 (O(K²D) 텐서 생성 없음)

    kernel gradient의 합 Σ_j ∇K(xi,xj)를 행렬 연산으로 직접 계산:
      Σ_j ∇K(xi,xj) = -1/h² (rowsum(K)·xi - K@X)

    (K,K,...) 텐서를 생성하지 않아 K=1024, D=60일 때 503MB→0MB 절약.

    Args:
        X: (K, ...) 샘플 집합
        grad_log_prob: (K, ...) ∇ log p(x)
        kernel: (K, K) RBF 커널
        bandwidth: 대역폭 h

    Returns:
        phi: (K, ...) SVGD 업데이트 방향
    """
    K = X.shape[0]
    original_shape = X.shape[1:]
    X_flat = X.reshape(K, -1)  # (K, D)
    grad_flat = grad_log_prob.reshape(K, -1)  # (K, D)

    # Term 1: Σ_j K(xi,xj) · ∇log p(xj) = kernel @ grad
    term1 = kernel @ grad_flat  # (K, D)

    # Term 2: Σ_j ∇_j K(xi,xj) = -1/h² · (rowsum(K)·X - K@X)
    kernel_row_sum = kernel.sum(axis=1)  # (K,)
    kernel_X = kernel @ X_flat  # (K, D)
    term2 = -(kernel_row_sum[:, None] * X_flat - kernel_X) / (bandwidth ** 2)

    phi = ((term1 + term2) / K).reshape(K, *original_shape)
    return phi


def median_bandwidth(X: np.ndarray) -> float:
    """
    Median heuristic로 bandwidth 계산

    h = median(||x_i - x_j||) / sqrt(log(K+1))

    Args:
        X: (K, ...) 샘플 집합

    Returns:
        bandwidth: float
    """
    K = X.shape[0]
    X_flat = X.reshape(K, -1)

    # Pairwise distances
    sq_norms = np.sum(X_flat**2, axis=1)
    sq_distances = sq_norms[:, None] + sq_norms[None, :] - 2 * X_flat @ X_flat.T

    # Median (excluding diagonal)
    sq_dists_no_diag = sq_distances[~np.eye(K, dtype=bool)]
    if len(sq_dists_no_diag) > 0:
        median_sq_dist = np.median(sq_dists_no_diag)
        bandwidth = np.sqrt(median_sq_dist / np.log(K + 1))
    else:
        bandwidth = 1.0

    return bandwidth
