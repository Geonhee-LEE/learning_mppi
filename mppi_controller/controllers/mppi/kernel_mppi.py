"""
Kernel MPPI Controller

RBF 커널 보간 기반 차원 축소 MPPI.
소수 서포트 포인트에서 커널 보간으로 전체 제어 시퀀스를 복원.
"""

import numpy as np
from typing import Dict, Tuple
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import KernelMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController


class RBFKernel:
    """
    Radial Basis Function (RBF) 커널

    k(t, t') = exp(-||t - t'||^2 / (2 * sigma^2))

    Args:
        sigma: 대역폭 파라미터 (클수록 부드러운 보간)
    """

    def __init__(self, sigma: float = 1.0):
        assert sigma > 0, "sigma must be positive"
        self.sigma = sigma

    def __call__(self, t: np.ndarray, tk: np.ndarray) -> np.ndarray:
        """
        커널 행렬 계산

        Args:
            t: (n,) 쿼리 시간 포인트
            tk: (m,) 서포트 시간 포인트

        Returns:
            K: (n, m) 커널 행렬
        """
        # (n, 1) - (1, m) -> (n, m) 거리 행렬
        diff = t[:, None] - tk[None, :]
        return np.exp(-diff**2 / (2 * self.sigma**2))


class KernelMPPIController(MPPIController):
    """
    Kernel MPPI Controller

    RBF 커널 보간을 사용하여 차원 축소된 MPPI.

    동작 원리:
        1. S개 서포트 포인트에서만 노이즈 샘플링 (S << N)
        2. 사전 계산된 커널 보간 행렬 W로 N개 제어값 복원
           - W = K_query(N,S) @ inv(K_support(S,S) + eps*I)
        3. MPPI 가중치 계산 (base 재사용)
        4. 서포트 포인트 가중 업데이트

    수학:
        k(t, t') = exp(-||t - t'||^2 / (2*sigma^2))   # RBF 커널
        W = K_query @ K_support^{-1}                    # 보간 가중치
        U = W @ theta                                    # 서포트 -> 전체

    장점:
        - 샘플링 차원 ~75% 감소 (S=8, N=30)
        - 커널에 의한 자동 제어 평활
        - 행렬 곱으로 효율적 보간 (spline 루프 불필요)

    참조:
        - pytorch_mppi KMPPI class

    Args:
        model: RobotModel 인스턴스
        params: KernelMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: KernelMPPIParams):
        super().__init__(model, params)

        self.S = params.num_support_pts
        self.kernel = RBFKernel(params.kernel_bandwidth)

        # 서포트 포인트 시간 위치 (등간격)
        self.Tk = np.linspace(0, params.N - 1, self.S)
        # 전체 호라이즌 시간 포인트
        self.Hs = np.arange(params.N, dtype=float)

        # 서포트 포인트 제어값 (S, nu)
        self.theta = np.zeros((self.S, self.model.control_dim))

        # 커널 보간 행렬 사전 계산 (N, S)
        self.W = self._precompute_kernel_matrix()

        # 통계 히스토리
        self.kernel_stats_history = []

    def _precompute_kernel_matrix(self) -> np.ndarray:
        """
        커널 보간 행렬 사전 계산

        W = K_query(N,S) @ inv(K_support(S,S) + eps*I)

        Returns:
            W: (N, S) 보간 가중치 행렬
        """
        # K_support: (S, S) 서포트-서포트 커널 행렬
        K_support = self.kernel(self.Tk, self.Tk)
        # 정칙화 (수치 안정성)
        K_support += 1e-6 * np.eye(self.S)

        # K_query: (N, S) 쿼리-서포트 커널 행렬
        K_query = self.kernel(self.Hs, self.Tk)

        # W = K_query @ inv(K_support)
        W = K_query @ np.linalg.inv(K_support)

        return W

    def _interpolate(self, theta: np.ndarray) -> np.ndarray:
        """
        서포트 포인트에서 전체 제어 시퀀스 복원

        Args:
            theta: (S, nu) 서포트 포인트 제어값

        Returns:
            U: (N, nu) 전체 제어 시퀀스
        """
        return self.W @ theta  # (N, S) @ (S, nu) -> (N, nu)

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Kernel MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K
        nu = self.model.control_dim

        # 1. 서포트 공간 노이즈 샘플링 (K, S, nu)
        support_noise = np.random.normal(
            0, self.params.sigma, (K, self.S, nu)
        )
        perturbed_theta = self.theta + support_noise  # (K, S, nu)

        # 2. 커널 보간으로 전체 궤적 복원: (K, N, nu)
        # W: (N, S), perturbed_theta: (K, S, nu) -> (K, N, nu)
        sampled_controls = np.einsum('ts,ksu->ktu', self.W, perturbed_theta)

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 롤아웃
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 비용 계산
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. MPPI 가중치 계산 (base 재사용)
        weights = self._compute_weights(costs, self.params.lambda_)

        # 6. ESS 계산
        ess = self._compute_ess(weights)

        # 7. 서포트 포인트 가중 업데이트
        weighted_support_noise = np.sum(
            weights[:, None, None] * support_noise, axis=0
        )  # (S, nu)
        self.theta = self.theta + weighted_support_noise

        # 8. U 복원 (호환성)
        self.U = self._interpolate(self.theta)

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 9. Receding horizon 시프트
        self._shift_theta()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 10. 최적 제어
        optimal_control = self.U[0, :]
        if self.u_min is not None and self.u_max is not None:
            optimal_control = np.clip(optimal_control, self.u_min, self.u_max)

        # 11. 커널 통계 저장
        theta_variance = np.var(perturbed_theta, axis=0).mean()
        kernel_stats = {
            "num_support_pts": self.S,
            "kernel_bandwidth": self.kernel.sigma,
            "theta_variance": float(theta_variance),
            "interpolation_matrix_cond": float(np.linalg.cond(self.W.T @ self.W)),
        }
        self.kernel_stats_history.append(kernel_stats)

        # 12. 정보 저장
        best_idx = np.argmin(costs)
        info = {
            "sample_trajectories": sample_trajectories,
            "sample_controls": sampled_controls,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": costs[best_idx],
            "mean_cost": np.mean(costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "support_theta": self.theta.copy(),
            "kernel_stats": kernel_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _shift_theta(self):
        """
        Receding horizon 서포트 포인트 시프트

        서포트 포인트를 1 타임스텝만큼 시프트하고
        마지막 서포트 포인트를 0으로 설정.
        """
        self.theta = np.roll(self.theta, -1, axis=0)
        self.theta[-1, :] = 0.0

    def get_kernel_statistics(self) -> Dict:
        """
        커널 통계 반환

        Returns:
            dict:
                - mean_theta_variance: float 평균 서포트 분산
                - kernel_stats_history: List[dict] 통계 히스토리
        """
        if len(self.kernel_stats_history) == 0:
            return {
                "mean_theta_variance": 0.0,
                "kernel_stats_history": [],
            }

        theta_variances = [s["theta_variance"] for s in self.kernel_stats_history]

        return {
            "mean_theta_variance": np.mean(theta_variances),
            "kernel_stats_history": self.kernel_stats_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 및 서포트 포인트 초기화"""
        super().reset()
        self.theta = np.zeros((self.S, self.model.control_dim))
        self.kernel_stats_history = []

    def __repr__(self) -> str:
        return (
            f"KernelMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"S={self.S}, bandwidth={self.kernel.sigma}, "
            f"params={self.params})"
        )
