"""
Smooth MPPI Controller

Input-Lifting 기반 부드러운 제어 생성.
"""

import numpy as np
from typing import Dict, Tuple
from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.mppi_params import SmoothMPPIParams
from mppi_controller.controllers.mppi.base_mppi import MPPIController


class SmoothMPPIController(MPPIController):
    """
    Smooth MPPI Controller

    Input-Lifting 기법으로 제어 입력의 부드러움을 보장하는 MPPI 변형.

    동작 원리:
        1. 제어 변화량 ΔU를 샘플링 (절대값 U 대신)
           ΔU = [Δu_0, Δu_1, ..., Δu_{N-1}]
        2. 누적합으로 절대 제어 복원
           U[t] = U[t-1] + ΔU[t]
        3. Jerk 비용 추가
           J_jerk = Σ ||ΔΔu_t||² = Σ ||ΔU[t+1] - ΔU[t]||²
        4. 총 비용: J_total = J_tracking + J_jerk
        5. MPPI 가중치로 ΔU 업데이트

    Input-Lifting 장점:
        - ✅ 자동 제어 부드러움 (ΔU 샘플링)
        - ✅ Jerk 최소화 (기계 부하 감소)
        - ✅ 전력 소비 감소
        - ✅ 구동기 수명 연장

    전통적 Smooth MPPI vs Input-Lifting:
        - 전통: U 샘플링 + Jerk cost 추가
        - Input-Lifting: ΔU 샘플링 (자동 부드러움)
        - Input-Lifting이 더 효과적

    적용 사례:
        - 로봇 매니퓰레이터 (관절 부하 감소)
        - 드론 (모터 보호)
        - 자율주행 (승차감 향상)

    참고 논문:
        - Kim et al. (2021) - "Smooth MPPI"

    Args:
        model: RobotModel 인스턴스
        params: SmoothMPPIParams 파라미터
    """

    def __init__(self, model: RobotModel, params: SmoothMPPIParams):
        # 부모 클래스 초기화
        super().__init__(model, params)

        # Smooth MPPI 파라미터
        self.smooth_params = params
        self.jerk_weight = params.jerk_weight

        # 제어 변화량 시퀀스 (Input-Lifting 핵심)
        self.delta_U = np.zeros((self.params.N, self.model.control_dim))

        # 통계 (디버깅용)
        self.jerk_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Smooth MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 디버깅/시각화 정보
        """
        K = self.params.K

        # 1. **ΔU (제어 변화량) 노이즈 샘플링** (핵심!)
        delta_noise = self.noise_sampler.sample(
            self.delta_U, K, self.u_min, self.u_max
        )
        sampled_delta_controls = self.delta_U + delta_noise  # (K, N, nu)

        # 2. **ΔU → U로 복원** (누적합, 벡터화)
        # U[t] = U[0] + Σ_{i=0}^{t} ΔU[i]
        sampled_controls = self.U[0:1, :][np.newaxis, :, :] + np.cumsum(
            sampled_delta_controls, axis=1
        )  # (K, N, nu)

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 롤아웃
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 4. 추적 비용 계산
        tracking_costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 5. **Jerk 비용 계산** (Input-Lifting 핵심)
        jerk_costs = self._compute_jerk_cost(sampled_delta_controls)

        # 6. 총 비용
        total_costs = tracking_costs + jerk_costs

        # 7. MPPI 가중치
        weights = self._compute_weights(total_costs, self.params.lambda_)

        # 8. ESS 계산
        ess = 1.0 / np.sum(weights**2)

        # 9. **ΔU 업데이트** (가중 평균)
        weighted_delta_noise = np.sum(weights[:, None, None] * delta_noise, axis=0)
        self.delta_U = self.delta_U + weighted_delta_noise

        # 10. U 전체 시퀀스 복원 (delta_U 누적합)
        base_u = self.U[0, :]
        self.U = base_u[np.newaxis, :] + np.cumsum(self.delta_U, axis=0)

        # 11. 최적 제어 (shift 전에 저장!)
        optimal_control = self.U[0, :].copy()

        # 12. Receding horizon (ΔU)
        self.delta_U = np.roll(self.delta_U, -1, axis=0)
        self.delta_U[-1, :] = 0.0

        # 13. Receding horizon (U)
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 제어 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            optimal_control = np.clip(optimal_control, self.u_min, self.u_max)

        # 14. Jerk 통계 저장
        mean_jerk = np.mean(jerk_costs)
        self.jerk_history.append(mean_jerk)

        # 15. 정보 저장
        info = {
            "sample_trajectories": sample_trajectories,
            "sample_controls": sampled_controls,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[np.argmax(weights)],
            "best_cost": np.min(total_costs),
            "mean_cost": np.mean(total_costs),
            "tracking_costs": tracking_costs,
            "jerk_costs": jerk_costs,
            "mean_jerk": mean_jerk,
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
        }

        return optimal_control, info

    def _compute_jerk_cost(self, delta_controls: np.ndarray) -> np.ndarray:
        """
        Jerk 비용 계산

        Jerk = Δ(ΔU) = ΔU[t+1] - ΔU[t]

        Args:
            delta_controls: (K, N, nu) 제어 변화량

        Returns:
            jerk_costs: (K,) Jerk 비용
        """
        # Δ(ΔU) = ΔU[t+1] - ΔU[t]
        jerk = np.diff(delta_controls, axis=1)  # (K, N-1, nu)

        # ||Jerk||² 합산
        jerk_cost = self.jerk_weight * np.sum(jerk**2, axis=(1, 2))  # (K,)

        return jerk_cost

    def get_jerk_statistics(self) -> Dict:
        """
        Jerk 통계 반환 (디버깅용)

        Returns:
            dict:
                - mean_jerk: float 평균 Jerk 비용
                - max_jerk: float 최대 Jerk 비용
                - jerk_history: List[float] Jerk 히스토리
        """
        if len(self.jerk_history) == 0:
            return {
                "mean_jerk": 0.0,
                "max_jerk": 0.0,
                "jerk_history": [],
            }

        return {
            "mean_jerk": np.mean(self.jerk_history),
            "max_jerk": np.max(self.jerk_history),
            "jerk_history": self.jerk_history.copy(),
        }

    def reset(self):
        """제어 시퀀스 및 ΔU 초기화"""
        super().reset()
        self.delta_U = np.zeros((self.params.N, self.model.control_dim))
        self.jerk_history = []

    def __repr__(self) -> str:
        return (
            f"SmoothMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"jerk_weight={self.jerk_weight:.2f}, "
            f"params={self.params})"
        )
