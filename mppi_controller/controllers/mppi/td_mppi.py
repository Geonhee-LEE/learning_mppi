"""
TD-MPPI (Temporal-Difference MPPI) Controller — 27번째 MPPI 변형

TD 학습 terminal value function V(x_T)를 MPPI 비용에 추가하여
짧은 롤아웃 호라이즌에서도 장기 계획 품질을 유지.

핵심 수식:
    C_total(τ) = Σ_t c(x_t, u_t) + w_V · V(x_T)
    V(x) ← V(x) + α[c + γV(x') - V(x)]    (TD(0) 업데이트)

제약 할인 (선택적):
    γ_eff(t) = γ · discount_decay^(n_violations_up_to_t)
    제약 위반 시 미래 가치를 할인하여 안전한 근시 행동 유도

기존 변형 대비 핵심 차이:
    - Vanilla MPPI: 유한 호라이즌만 고려 → 롤아웃 너머 비용 무시
    - DIAL-MPPI: 반복으로 비용 지형 탐색 향상 → 호라이즌 자체는 미확장
    - TD-MPPI: V(x_T)로 무한 호라이즌 근사 → N=10에서도 N=30급 성능

Reference: Crestaz et al., RA-L 2026, hal-05213269
"""

import numpy as np
from typing import Dict, Tuple, Optional

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import TDMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.controllers.mppi.td_value import (
    TDValueLearner,
    TDExperienceBuffer,
)


class TDMPPIController(MPPIController):
    """
    TD-MPPI Controller (27번째 MPPI 변형)

    TD 학습 terminal value + 선택적 제약 할인.

    Vanilla MPPI 대비 핵심 차이:
        1. Terminal value: V(x_T)로 롤아웃 너머 비용 근사
        2. 온라인 TD 학습: 제어 중 점진적 가치 함수 업데이트
        3. 제약 할인 (선택적): 위반 시 미래 가치 할인

    동작 흐름:
        1. 이전 스텝의 (s, c, s') → 경험 버퍼에 저장
        2. 주기적 TD(0) 업데이트
        3. 표준 MPPI 샘플링 → rollout → 기본 비용
        4. V(x_T) 추가 (학습 충분 시)
        5. 가중치 → 제어 업데이트

    Args:
        model: RobotModel 인스턴스
        params: TDMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: TDMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)
        self.td_params = params

        # Value function learner
        self._value_learner = TDValueLearner(
            state_dim=model.state_dim,
            hidden_dims=params.value_hidden_dims,
            lr=params.td_learning_rate,
            gamma=params.td_gamma,
        )

        # 경험 버퍼
        self._buffer = TDExperienceBuffer(max_size=params.td_buffer_size)

        # 제어 루프 상태
        self._step_count = 0
        self._prev_state = None
        self._prev_cost = None

        # TD 통계 추적
        self._td_history = []

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        TD-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + td_stats
        """
        K = self.params.K
        N = self.params.N

        # ── 1. TD 경험 수집 (이전 스텝의 (s, c, s')) ──
        if self._prev_state is not None:
            self._buffer.add(self._prev_state, self._prev_cost, state)

        # ── 2. TD 업데이트 (주기적) ──
        td_loss = 0.0
        if (
            self._step_count % self.td_params.td_update_interval == 0
            and len(self._buffer) >= self.td_params.td_min_samples
        ):
            states_b, costs_b, next_states_b = self._buffer.sample(
                self.td_params.td_batch_size
            )
            td_loss = self._value_learner.update(states_b, costs_b, next_states_b)

        # ── 3. 표준 MPPI 샘플링 ──
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)
        sampled_controls = self.U[None, :, :] + noise  # (K, N, nu)

        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # ── 4. Rollout ──
        trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # ── 5. 기본 비용 ──
        costs = self.cost_function.compute_cost(
            trajectories, sampled_controls, reference_trajectory
        )

        # ── 6. Terminal value 추가 ──
        terminal_value_mean = 0.0
        if (
            self.td_params.use_terminal_value
            and len(self._buffer) >= self.td_params.td_min_samples
        ):
            terminal_states = trajectories[:, -1, :]  # (K, nx)
            terminal_values = self._value_learner.predict(terminal_states)  # (K,)
            costs = costs + self.td_params.value_weight * terminal_values
            terminal_value_mean = float(np.mean(terminal_values))

        # ── 7. 제약 할인 (선택적) ──
        if self.td_params.use_constraint_discount:
            constraint_penalties = self._compute_constraint_penalties(
                trajectories, sampled_controls
            )
            costs = costs + constraint_penalties

        # ── 8. 가중치 + 업데이트 ──
        weights = self._compute_weights(costs, self.params.lambda_)
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 첫 제어 추출
        optimal_control = self.U[0].copy()

        # Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # ── 9. 경험 저장 ──
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        # 이전 스텝 비용 추정 (최적 궤적의 1스텝 비용 근사)
        self._prev_state = state.copy()
        self._prev_cost = float(costs[best_idx]) / max(N, 1)
        self._step_count += 1

        # TD 통계 기록
        td_stats = {
            "td_loss": td_loss,
            "buffer_size": len(self._buffer),
            "terminal_value_mean": terminal_value_mean,
            "td_update_count": self._value_learner.update_count,
            "value_weight": self.td_params.value_weight,
            "use_terminal_value": (
                self.td_params.use_terminal_value
                and len(self._buffer) >= self.td_params.td_min_samples
            ),
        }
        self._td_history.append(td_stats)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "td_stats": td_stats,
        }
        self.last_info = info

        return optimal_control, info

    def _compute_constraint_penalties(
        self,
        trajectories: np.ndarray,
        sampled_controls: np.ndarray,
    ) -> np.ndarray:
        """
        제약 할인 페널티 계산

        제약 위반 시 할인된 미래 가치로 추가 페널티 부여.
        간단 구현: 큰 위치 변화를 제약 위반으로 간주.

        Args:
            trajectories: (K, N+1, nx)
            sampled_controls: (K, N, nu)

        Returns:
            penalties: (K,) 추가 페널티
        """
        K, N_plus_1, nx = trajectories.shape
        penalty = self.td_params.constraint_penalty
        decay = self.td_params.discount_decay

        # 간단한 제약: 제어 입력 크기 기반 감쇠
        # (실제로는 장애물 제약 등 외부 제약 함수 사용)
        penalties = np.zeros(K)

        # 제어 입력 크기가 클수록 페널티 (에너지 절약 관점)
        ctrl_norms = np.sum(sampled_controls ** 2, axis=(1, 2))  # (K,)
        mean_norm = np.mean(ctrl_norms) + 1e-8
        relative_excess = np.maximum(ctrl_norms / mean_norm - 1.0, 0.0)
        penalties = penalty * decay * relative_excess

        return penalties

    def get_td_statistics(self) -> Dict:
        """
        TD 학습 누적 통계 반환

        Returns:
            dict: 버퍼 크기, 평균 TD 손실, 학습 진행 상황
        """
        if not self._td_history:
            return {
                "total_steps": 0,
                "buffer_size": 0,
                "td_update_count": 0,
                "mean_td_loss": 0.0,
                "mean_terminal_value": 0.0,
                "history": [],
            }

        recent = self._td_history[-50:]
        losses = [h["td_loss"] for h in recent if h["td_loss"] > 0]
        terminal_vals = [
            h["terminal_value_mean"] for h in recent
            if h["terminal_value_mean"] != 0.0
        ]

        return {
            "total_steps": self._step_count,
            "buffer_size": len(self._buffer),
            "td_update_count": self._value_learner.update_count,
            "mean_td_loss": float(np.mean(losses)) if losses else 0.0,
            "mean_terminal_value": (
                float(np.mean(terminal_vals)) if terminal_vals else 0.0
            ),
            "history": self._td_history.copy(),
        }

    def get_value_learner(self) -> TDValueLearner:
        """Value learner 객체 접근 (외부 분석용)"""
        return self._value_learner

    def get_buffer(self) -> TDExperienceBuffer:
        """경험 버퍼 접근 (외부 분석용)"""
        return self._buffer

    def reset(self):
        """제어 시퀀스 + 내부 상태 초기화"""
        super().reset()
        self._step_count = 0
        self._prev_state = None
        self._prev_cost = None
        self._td_history = []
        # 버퍼와 value learner는 유지 (학습 지속)

    def full_reset(self):
        """전체 초기화 (버퍼 + value learner 포함)"""
        self.reset()
        self._buffer.clear()
        self._value_learner = TDValueLearner(
            state_dim=self.model.state_dim,
            hidden_dims=self.td_params.value_hidden_dims,
            lr=self.td_params.td_learning_rate,
            gamma=self.td_params.td_gamma,
        )

    def __repr__(self) -> str:
        buf_size = len(self._buffer)
        updates = self._value_learner.update_count
        return (
            f"TDMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"N={self.params.N}, "
            f"buffer={buf_size}, "
            f"updates={updates}, "
            f"value_weight={self.td_params.value_weight})"
        )
