"""
Koopman 연산자 기반 동역학 모델

비선형 동역학을 고차원 특징 공간에서 선형으로 표현.
데이터 기반(EDMD) 또는 학습 기반(Deep Koopman)으로 구성.

선형 예측:
    z_{t+1} = K @ φ(x_t) + B @ u_t
    x_{t+1} ≈ C @ z_{t+1}

배치 행렬 연산으로 전체 호라이즌 계산 가능:
    Z = [K^1; K^2; ...; K^N] z_0 + Bbar @ U_flat

이 선형성이 MPPI rollout을 대폭 가속화.

References:
    Koopman (1931) — Original operator theory
    Brunton et al. (2016) — DMD and Koopman modes
    Li et al. (2017) — Extended DMD (EDMD)
    Lusch et al. (2018) — Deep learning for Koopman
    Bevanda et al. (2021) — Koopman operator dynamical models
"""

import numpy as np
from typing import List, Optional, Tuple
from mppi_controller.models.base_model import RobotModel


def _rbf_features(x: np.ndarray, centers: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    RBF 특징 함수 φ_rbf(x).

    φ_i(x) = exp(-γ ||x - c_i||²)

    Args:
        x: (..., nx)
        centers: (n_centers, nx)
        gamma: RBF 대역폭

    Returns:
        features: (..., n_centers)
    """
    diff = x[..., None, :] - centers  # (..., n_centers, nx)
    dist_sq = np.sum(diff**2, axis=-1)  # (..., n_centers)
    return np.exp(-gamma * dist_sq)


def _polynomial_features(x: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    다항식 특징 함수 (1차 + 2차 교차항).

    Args:
        x: (..., nx)
        degree: 최대 차수 (1 또는 2)

    Returns:
        features: (..., n_features)
    """
    base = x  # 1차

    if degree == 1:
        return base

    # 2차: x_i * x_j (i ≤ j)
    nx = x.shape[-1]
    quad_parts = []
    for i in range(nx):
        for j in range(i, nx):
            quad_parts.append(x[..., i:i+1] * x[..., j:j+1])

    quad = np.concatenate(quad_parts, axis=-1) if quad_parts else np.zeros(x.shape[:-1] + (0,))
    return np.concatenate([base, quad], axis=-1)


class KoopmanDynamics(RobotModel):
    """
    Koopman 연산자 기반 선형 동역학 모델.

    비선형 동역학 x_{t+1} = f(x_t, u_t)를
    선형 잠재 공간으로 표현:
        z_{t+1} = K @ φ(x_t) + B @ u_t

    학습 방법:
        - EDMD (Extended DMD): 최소제곱 행렬 추정
        - 또는 Deep Koopman (PyTorch 사용 시)

    빠른 rollout:
        전체 호라이즌을 행렬 연산으로 O(N)에 계산

    Args:
        nx: 상태 차원
        nu: 제어 차원
        lift_fn: 특징 함수 타입 ("rbf" | "poly" | "custom")
        lift_dim: 특징 공간 차원 (RBF의 경우 RBF 개수)
        rbf_gamma: RBF 대역폭 (lift_fn="rbf" 시)
        poly_degree: 다항식 차수 (lift_fn="poly" 시)
        use_bias: φ(x)에 바이어스항(1) 추가
    """

    def __init__(
        self,
        nx: int,
        nu: int,
        lift_fn: str = "rbf",
        lift_dim: int = 64,
        rbf_gamma: float = 1.0,
        poly_degree: int = 2,
        use_bias: bool = True,
    ):
        self._nx = nx
        self._nu = nu
        self.lift_fn = lift_fn
        self.lift_dim = lift_dim
        self.rbf_gamma = rbf_gamma
        self.poly_degree = poly_degree
        self.use_bias = use_bias

        # 특징 공간 차원 계산
        self._phi_dim = self._compute_phi_dim(nx, lift_fn, lift_dim, poly_degree, use_bias)

        # RBF 중심 (랜덤 초기화)
        self._rbf_centers = np.random.randn(lift_dim, nx) if lift_fn == "rbf" else None

        # Koopman 행렬 (학습 전 단위행렬)
        self.K = np.eye(self._phi_dim)             # (phi_dim, phi_dim)
        self.B = np.zeros((self._phi_dim, nu))     # (phi_dim, nu)
        self.C = np.zeros((nx, self._phi_dim))     # (nx, phi_dim) — 역투영
        self.C[:nx, :nx] = np.eye(nx)              # 처음 nx 차원이 상태

        self._is_fitted = False

    @staticmethod
    def _compute_phi_dim(nx, lift_fn, lift_dim, poly_degree, use_bias):
        if lift_fn == "rbf":
            base = lift_dim
        elif lift_fn == "poly":
            if poly_degree == 1:
                base = nx
            else:  # degree=2
                base = nx + nx * (nx + 1) // 2
        else:
            base = lift_dim

        return base + (1 if use_bias else 0)

    @property
    def state_dim(self) -> int:
        return self._nx

    @property
    def control_dim(self) -> int:
        return self._nu

    @property
    def model_type(self) -> str:
        return "koopman"

    def phi(self, x: np.ndarray) -> np.ndarray:
        """
        특징 함수 φ(x): 상태 → 고차원 특징 공간.

        Args:
            x: (..., nx)

        Returns:
            z: (..., phi_dim)
        """
        if self.lift_fn == "rbf":
            feat = _rbf_features(x, self._rbf_centers, self.rbf_gamma)
        elif self.lift_fn == "poly":
            feat = _polynomial_features(x, self.poly_degree)
        else:
            # 기본: 항등 특징 (lift_dim 무시)
            feat = x

        if self.use_bias:
            bias_shape = x.shape[:-1] + (1,)
            bias = np.ones(bias_shape, dtype=x.dtype)
            feat = np.concatenate([feat, bias], axis=-1)

        return feat  # (..., phi_dim)

    def lift(self, x: np.ndarray) -> np.ndarray:
        """phi와 동일 (별칭)."""
        return self.phi(x)

    def project(self, z: np.ndarray) -> np.ndarray:
        """
        역투영 C @ z → 상태 공간.

        Args:
            z: (..., phi_dim)

        Returns:
            x: (..., nx)
        """
        return z @ self.C.T  # (..., nx)

    def forward_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Koopman 선형 예측.

        z_{t+1} = K @ φ(x_t) + B @ u_t
        x_{t+1} = C @ z_{t+1}
        state_dot ≈ (x_{t+1} - x_t) / dt  [dt=1로 근사, 실제 dt는 상위에서 처리]

        Args:
            state: (..., nx)
            control: (..., nu)

        Returns:
            state_dot: (..., nx)
        """
        z = self.phi(state)  # (..., phi_dim)

        # K @ z: (..., phi_dim) @ (phi_dim, phi_dim).T → (..., phi_dim)
        z_next = z @ self.K.T + control @ self.B.T

        x_next = z_next @ self.C.T  # (..., nx)
        return x_next - state  # state_dot = (x_{t+1} - x_t) / 1

    def rollout_linear(
        self,
        state: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        """
        선형 Koopman rollout — 전체 호라이즌 행렬 연산.

        K^t z_0 + Σ K^(t-k-1) B u_k 를 순차 계산.

        Args:
            state: (nx,) 초기 상태
            controls: (N, nu) 제어 시퀀스

        Returns:
            trajectory: (N+1, nx) 상태 궤적
        """
        N = controls.shape[0]
        z = self.phi(state[None])[0]  # (phi_dim,)

        traj_x = np.zeros((N + 1, self._nx))
        traj_x[0] = state

        for t in range(N):
            z = self.K @ z + self.B @ controls[t]
            traj_x[t + 1] = self.C @ z

        return traj_x

    def batch_rollout_linear(
        self,
        states: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        """
        배치 Koopman rollout — K개 샘플 동시 계산.

        Args:
            states: (K, nx) 초기 상태 배치
            controls: (K, N, nu) 제어 시퀀스 배치

        Returns:
            trajectories: (K, N+1, nx)
        """
        K, N, nu = controls.shape

        # 초기 잠재 상태: (K, phi_dim)
        Z = self.phi(states)

        traj = np.zeros((K, N + 1, self._nx))
        traj[:, 0, :] = states

        for t in range(N):
            # Z: (K, phi_dim), K: (phi_dim, phi_dim), B: (phi_dim, nu)
            Z = Z @ self.K.T + controls[:, t, :] @ self.B.T  # (K, phi_dim)
            traj[:, t + 1, :] = Z @ self.C.T  # (K, nx)

        return traj

    def fit_edmd(
        self,
        X: np.ndarray,
        U: np.ndarray,
        X_next: np.ndarray,
        reg: float = 1e-4,
    ) -> float:
        """
        EDMD (Extended Dynamic Mode Decomposition) 학습.

        최소제곱: [K | B] = X_next_lifted @ Ψ^+
        Ψ = [φ(X) | U],  X_next_lifted = φ(X_next)

        Args:
            X: (M, nx) 현재 상태 배치
            U: (M, nu) 제어 배치
            X_next: (M, nx) 다음 상태 배치
            reg: Tikhonov 정규화 계수

        Returns:
            fit_error: 예측 오차 (RMSE)
        """
        M = X.shape[0]

        # 특징 행렬
        Phi_X = self.phi(X)        # (M, phi_dim)
        Phi_Xnext = self.phi(X_next)  # (M, phi_dim)

        # 회귀 입력: [φ(x) | u]
        Psi = np.hstack([Phi_X, U])  # (M, phi_dim + nu)

        # 목표: φ(x_next)
        # [K | B]^T = Psi^+ Phi_Xnext → shape (phi_dim+nu, phi_dim)
        # Tikhonov 정규화: (Ψ^T Ψ + reg I)^{-1} Ψ^T Φ
        A = Psi.T @ Psi + reg * np.eye(Psi.shape[1])
        b = Psi.T @ Phi_Xnext
        KB = np.linalg.solve(A, b)  # (phi_dim+nu, phi_dim)

        phi_dim = self._phi_dim
        self.K = KB[:phi_dim, :].T      # (phi_dim, phi_dim)
        self.B = KB[phi_dim:, :].T      # (phi_dim, nu)

        # 역투영 C: φ(x_next) → x_next
        # C^T = Phi_X^+ X → (phi_dim, nx)
        A2 = Phi_X.T @ Phi_X + reg * np.eye(phi_dim)
        b2 = Phi_X.T @ X
        self.C = np.linalg.solve(A2, b2).T  # (nx, phi_dim)

        self._is_fitted = True

        # 예측 오차 계산
        Phi_pred = Phi_X @ self.K.T + U @ self.B.T  # (M, phi_dim)
        x_pred = Phi_pred @ self.C.T  # (M, nx)
        rmse = np.sqrt(np.mean((x_pred - X_next)**2))
        return float(rmse)


class KoopmanTrainer:
    """
    Koopman 동역학 학습기.

    데이터 수집 → EDMD 또는 딥 Koopman 학습.

    Args:
        nx: 상태 차원
        nu: 제어 차원
        lift_fn: 특징 함수 타입
        lift_dim: 특징 차원
        rbf_gamma: RBF 대역폭
    """

    def __init__(
        self,
        nx: int,
        nu: int,
        lift_fn: str = "rbf",
        lift_dim: int = 64,
        rbf_gamma: float = 1.0,
        poly_degree: int = 2,
        reg: float = 1e-4,
    ):
        self.model = KoopmanDynamics(
            nx=nx,
            nu=nu,
            lift_fn=lift_fn,
            lift_dim=lift_dim,
            rbf_gamma=rbf_gamma,
            poly_degree=poly_degree,
        )
        self.reg = reg

        self._X: List[np.ndarray] = []
        self._U: List[np.ndarray] = []
        self._X_next: List[np.ndarray] = []

    def add_transition(
        self,
        state: np.ndarray,
        control: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """
        단일 전이 샘플 추가.

        Args:
            state: (nx,)
            control: (nu,)
            next_state: (nx,)
        """
        self._X.append(state.copy())
        self._U.append(control.copy())
        self._X_next.append(next_state.copy())

    def add_trajectory(
        self,
        states: np.ndarray,
        controls: np.ndarray,
    ) -> None:
        """
        전체 궤적에서 전이 쌍 추가.

        Args:
            states: (T+1, nx) 상태 궤적
            controls: (T, nu) 제어 시퀀스
        """
        T = controls.shape[0]
        for t in range(T):
            self.add_transition(states[t], controls[t], states[t + 1])

    def fit(self) -> float:
        """
        EDMD로 Koopman 행렬 학습.

        Returns:
            rmse: 예측 오차
        """
        if len(self._X) < 2:
            raise ValueError("최소 2개 이상의 전이 샘플이 필요합니다.")

        X = np.array(self._X)
        U = np.array(self._U)
        X_next = np.array(self._X_next)

        return self.model.fit_edmd(X, U, X_next, reg=self.reg)

    @property
    def n_samples(self) -> int:
        return len(self._X)
