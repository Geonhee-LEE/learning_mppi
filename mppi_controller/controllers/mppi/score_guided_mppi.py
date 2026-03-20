"""
SG-MPPI (Score-Guided MPPI) Controller — 22번째 MPPI 변형

Denoising Score Matching으로 비용 지형의 score function을 학습하고,
MPPI 가우시안 노이즈에 score 방향 bias를 추가하여 저비용 영역으로 유도.

핵심 수식:
    s_θ(U, σ, state) ≈ ∇_U log p(U|state)
    ε_guided = ε + α · σ² · s_θ(U + ε, σ, state)

Score 미학습 시 순수 가우시안 fallback (= Vanilla MPPI).

기존 변형 대비 핵심 차이:
    - Diffusion/Flow-MPPI: 전체 샘플 분포 대체 → 학습 실패 시 붕괴 위험
    - DIAL-MPPI: 비용 지형 구조 미활용
    - SG-MPPI: 가우시안 구조 유지 + score bias → graceful degradation

References:
    - Song & Ermon (2019) — Score-based Generative Modeling
    - Li & Chen (2025) — Score-guided Planning
    - DIAL-MPC (2024) — Multi-iteration annealing
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import SGMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.learning.flow_data_collector import FlowDataCollector


class SGMPPIController(MPPIController):
    """
    Score-Guided MPPI 컨트롤러 (22번째 MPPI 변형)

    MPPI 가우시안 노이즈에 학습된 score function bias를 추가.
    Score 미학습 시 순수 가우시안 fallback (= Vanilla MPPI).

    동작 모드:
        1. n_guide_iters=1, use_annealing=False (기본):
           - 표준 MPPI + score-guided 샘플링
        2. n_guide_iters>1 or use_annealing=True:
           - DIAL-style 다중 반복 + score guidance 결합

    Args:
        model: RobotModel 인스턴스
        params: SGMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler)
    """

    def __init__(
        self,
        model: RobotModel,
        params: SGMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
        **kwargs,
    ):
        super().__init__(model, params, cost_function, noise_sampler)

        self.sg_params = params

        # Score model (lazy init)
        self._score_model = None
        self._trainer = None

        # 데이터 수집기 (FlowDataCollector 재사용)
        self._data_collector = FlowDataCollector(
            buffer_size=params.score_buffer_size
        )

        # 온라인 학습 카운터
        self._step_count = 0

        # σ 스케줄 (기하급수): sigma_min → sigma_max
        self._sigma_levels = np.geomspace(
            params.sigma_min, params.sigma_max, params.n_sigma_levels
        )

        # 첫 호출 여부 (DIAL 결합용)
        self._is_first_call = True

        # Score 통계 추적
        self._score_stats_history = []

        # 현재 상태 (score evaluation에 사용)
        self._current_state = None

        # Score 모델 로드
        if params.score_model_path is not None:
            self.load_score_model(params.score_model_path)

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        SG-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + score_stats
        """
        self._current_state = state

        use_multi_iter = (
            self.sg_params.n_guide_iters > 1 or self.sg_params.use_annealing
        )

        if use_multi_iter:
            control, info = self._compute_multi_iter(state, reference_trajectory)
        else:
            control, info = self._compute_single_iter(state, reference_trajectory)

        # 데이터 수집 + 온라인 학습
        self._data_collector.add_sample(state, self.U.copy())
        self._step_count += 1

        if self.sg_params.score_online_training:
            self._maybe_online_train()

        # Score 통계 추가
        info["score_stats"] = self.get_score_statistics()

        return control, info

    def _compute_single_iter(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """단일 반복 모드: 표준 MPPI + score-guided 샘플링"""
        K = self.params.K
        N = self.params.N

        # Score-guided 샘플링
        noise = self._sample_with_score(self.U, K)

        # 샘플 제어 시퀀스
        sampled_controls = self.U[None, :, :] + noise  # (K, N, nu)
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # Rollout + 비용 계산
        trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
        costs = self.cost_function.compute_cost(
            trajectories, sampled_controls, reference_trajectory
        )

        # 가중치 계산
        weights = self._compute_weights(costs, self.params.lambda_)

        # 가중 평균 업데이트
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U = self.U + weighted_noise

        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 첫 제어 추출 + shift
        optimal_control = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # Info
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        info = {
            "sample_trajectories": trajectories,
            "sample_weights": weights,
            "best_trajectory": trajectories[best_idx],
            "best_cost": costs[best_idx],
            "mean_cost": np.mean(costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
        }
        self.last_info = info
        return optimal_control, info

    def _compute_multi_iter(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """다중 반복 모드: DIAL-style 어닐링 + score guidance"""
        K = self.params.K
        N = self.params.N
        nu = self.model.control_dim

        n_iters = self.sg_params.n_guide_iters
        alpha = self.sg_params.guidance_scale

        iteration_costs = []
        last_trajectories = None
        last_weights = None
        last_costs = None

        for i in range(n_iters):
            # 어닐링 스케일 계산
            if self.sg_params.use_annealing:
                traj_scale = 0.5 ** i  # DIAL-style 반복 감쇠
            else:
                traj_scale = 1.0

            # 노이즈 스케일
            annealed_sigma = self.params.sigma * traj_scale

            # Score-guided 샘플링 (반복마다 α 감쇠)
            current_alpha = alpha * (self.sg_params.guidance_decay ** i)
            noise = self._sample_with_score(
                self.U, K, sigma_override=annealed_sigma,
                guidance_scale_override=current_alpha,
            )

            # 샘플 제어 생성 + 클리핑
            sampled_controls = self.U[None, :, :] + noise
            if self.u_min is not None and self.u_max is not None:
                sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

            # Rollout + 비용
            trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)
            costs = self.cost_function.compute_cost(
                trajectories, sampled_controls, reference_trajectory
            )

            # 가중치 (DIAL 정규화)
            weights = self._compute_weights_normalized(costs)

            # 전체 교체 업데이트
            self.U = np.sum(
                weights[:, None, None] * sampled_controls, axis=0
            )
            if self.u_min is not None and self.u_max is not None:
                self.U = np.clip(self.U, self.u_min, self.u_max)

            iteration_costs.append(float(np.min(costs)))
            last_trajectories = trajectories
            last_weights = weights
            last_costs = costs

        # 첫 제어 추출 + shift
        optimal_control = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        ess = self._compute_ess(last_weights)
        best_idx = np.argmin(last_costs)

        info = {
            "sample_trajectories": last_trajectories,
            "sample_weights": last_weights,
            "best_trajectory": last_trajectories[best_idx],
            "best_cost": last_costs[best_idx],
            "mean_cost": np.mean(last_costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            "sg_multi_iter": {
                "n_iters": n_iters,
                "iteration_costs": iteration_costs,
                "cost_improvement": (
                    iteration_costs[0] - iteration_costs[-1]
                    if len(iteration_costs) > 1 else 0.0
                ),
            },
        }
        self.last_info = info
        return optimal_control, info

    def _compute_weights_normalized(self, costs: np.ndarray) -> np.ndarray:
        """DIAL-style 보상 정규화 가중치 (다중 반복용)"""
        rewards = -costs
        std = np.std(rewards)
        if std < 1e-10:
            return np.ones(len(costs)) / len(costs)

        normalized = (rewards - np.mean(rewards)) / (std + 1e-10)
        scaled = normalized / self.params.lambda_
        scaled -= np.max(scaled)
        exp_scaled = np.exp(scaled)
        return exp_scaled / np.sum(exp_scaled)

    def _sample_with_score(
        self,
        U: np.ndarray,
        K: int,
        sigma_override: Optional[np.ndarray] = None,
        guidance_scale_override: Optional[float] = None,
    ) -> np.ndarray:
        """
        Score-guided 노이즈 샘플링

        ε_guided = ε + α · σ² · s_θ(U + ε, σ, state)

        Args:
            U: (N, nu) 명목 제어 시퀀스
            K: 샘플 수
            sigma_override: 노이즈 스케일 오버라이드
            guidance_scale_override: α 오버라이드

        Returns:
            noise: (K, N, nu) score-guided 노이즈
        """
        N, nu = U.shape
        sigma = sigma_override if sigma_override is not None else self.params.sigma
        alpha = (
            guidance_scale_override
            if guidance_scale_override is not None
            else self.sg_params.guidance_scale
        )

        # 가우시안 노이즈
        eps = np.random.standard_normal((K, N, nu)) * sigma[None, None, :]

        # Score model이 준비되지 않았으면 순수 가우시안 반환
        if self._score_model is None or alpha <= 0:
            return eps

        # Score bias 계산
        try:
            import torch

            self._score_model.eval()
            with torch.no_grad():
                # U + ε → 플래튼
                U_noisy = (U[None, :, :] + eps).reshape(K, N * nu)
                U_t = torch.tensor(
                    U_noisy, dtype=torch.float32,
                    device=next(self._score_model.parameters()).device,
                )

                # 중간 σ 레벨 선택
                mid_sigma = float(np.median(self._sigma_levels))
                sigma_t = torch.full(
                    (K,), mid_sigma,
                    device=U_t.device, dtype=torch.float32,
                )

                # Context (state)
                ctx_t = None
                if self._current_state is not None:
                    ctx_t = torch.tensor(
                        self._current_state, dtype=torch.float32,
                        device=U_t.device,
                    ).unsqueeze(0).expand(K, -1)

                # Score 계산
                score = self._score_model(U_t, sigma_t, ctx_t)  # (K, N*nu)
                score_np = score.cpu().numpy().reshape(K, N, nu)

            # Score guidance: ε_guided = ε + α · σ² · score
            sigma_sq = (sigma ** 2)[None, None, :]
            eps_guided = eps + alpha * sigma_sq * score_np

            # Score 크기 기록
            score_mag = float(np.mean(np.abs(score_np)))
            self._score_stats_history.append({
                "score_magnitude": score_mag,
                "guidance_scale": alpha,
                "sigma_used": mid_sigma,
            })

            return eps_guided

        except Exception:
            # Score 계산 실패 시 fallback
            return eps

    def _maybe_online_train(self):
        """주기적 온라인 학습 트리거"""
        if self._step_count % self.sg_params.score_training_interval != 0:
            return
        if not self._data_collector.should_train(self.sg_params.score_min_samples):
            return
        self.train_score_model(epochs=20)

    def train_score_model(
        self,
        data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
    ) -> Dict:
        """
        Score 모델 학습

        Args:
            data: (states, controls) 튜플. None이면 내부 버퍼 사용.
            epochs: 에포크 수

        Returns:
            metrics: 학습 메트릭
        """
        from mppi_controller.learning.score_matching_trainer import ScoreMatchingTrainer

        if data is not None:
            states, controls = data
        else:
            if not self._data_collector.should_train(
                self.sg_params.score_min_samples
            ):
                return {
                    "status": "insufficient_data",
                    "num_samples": self._data_collector.num_samples,
                }
            states, controls = self._data_collector.get_training_data()

        N, nu = self.params.N, self.model.control_dim
        control_seq_dim = N * nu
        context_dim = self.model.state_dim

        # Trainer 생성/재사용
        if self._trainer is None:
            self._trainer = ScoreMatchingTrainer(
                control_seq_dim=control_seq_dim,
                context_dim=context_dim,
                hidden_dims=self.sg_params.score_hidden_dims,
                n_sigma_levels=self.sg_params.n_sigma_levels,
                sigma_min=self.sg_params.sigma_min,
                sigma_max=self.sg_params.sigma_max,
                device=self.sg_params.device,
            )

        metrics = self._trainer.train(states, controls, epochs=epochs)

        # 학습된 모델 갱신
        self._score_model = self._trainer.get_model()

        metrics["status"] = "trained"
        return metrics

    def save_score_model(self, path: str):
        """Score 모델 저장"""
        if self._trainer is not None:
            self._trainer.save_model(path)

    def load_score_model(self, path: str):
        """Score 모델 로드"""
        from mppi_controller.learning.score_matching_trainer import ScoreMatchingTrainer

        N, nu = self.params.N, self.model.control_dim
        control_seq_dim = N * nu
        context_dim = self.model.state_dim

        self._trainer = ScoreMatchingTrainer(
            control_seq_dim=control_seq_dim,
            context_dim=context_dim,
            hidden_dims=self.sg_params.score_hidden_dims,
            n_sigma_levels=self.sg_params.n_sigma_levels,
            sigma_min=self.sg_params.sigma_min,
            sigma_max=self.sg_params.sigma_max,
            device=self.sg_params.device,
        )
        self._trainer.load_model(path)
        self._score_model = self._trainer.get_model()

    def get_score_statistics(self) -> Dict:
        """Score 통계 반환"""
        return {
            "score_ready": self._score_model is not None,
            "buffer_size": self._data_collector.num_samples,
            "step_count": self._step_count,
            "online_training": self.sg_params.score_online_training,
            "guidance_scale": self.sg_params.guidance_scale,
            "n_guide_iters": self.sg_params.n_guide_iters,
            "mean_score_magnitude": (
                float(np.mean([s["score_magnitude"] for s in self._score_stats_history[-10:]]))
                if self._score_stats_history else 0.0
            ),
        }

    def reset(self):
        """제어 시퀀스 + 내부 상태 초기화"""
        super().reset()
        self._step_count = 0
        self._is_first_call = True
        self._score_stats_history = []
        self._current_state = None

    def __repr__(self) -> str:
        status = "ready" if self._score_model is not None else "fallback"
        return (
            f"SGMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"guidance_scale={self.sg_params.guidance_scale}, "
            f"n_guide_iters={self.sg_params.n_guide_iters}, "
            f"status={status})"
        )
