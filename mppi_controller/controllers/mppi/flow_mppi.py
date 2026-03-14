"""
Flow-MPPI 컨트롤러

Conditional Flow Matching + MPPI: 학습된 속도장으로
가우시안 노이즈 대신 다중 모달 제어 시퀀스 분포를 생성.

Flow 미학습 시 가우시안 fallback (= Vanilla MPPI 동작).

References:
    - Kurtz & Burdick (2025) — GPC: self-supervised flow for MPPI
    - Mizuta & Leung (2025) — CFM-MPPI
    - Trevisan (RA-L 2024) — Biased-MPPI
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import FlowMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.controllers.mppi.flow_matching_sampler import FlowMatchingSampler
from mppi_controller.learning.flow_data_collector import FlowDataCollector


class FlowMPPIController(MPPIController):
    """
    Flow-MPPI 컨트롤러

    MPPIController 상속, FlowMatchingSampler로 다중 모달 샘플링.
    MPPI 자체 실행 이력에서 self-supervised Flow Matching 학습.

    Args:
        model: RobotModel 인스턴스
        params: FlowMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 FlowMatchingSampler 자동 생성)
    """

    def __init__(
        self,
        model: RobotModel,
        params: FlowMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        # FlowMatchingSampler 자동 생성
        if noise_sampler is None:
            noise_sampler = FlowMatchingSampler(
                sigma=params.sigma,
                mode=params.flow_mode,
                blend_ratio=params.flow_blend_ratio,
                exploration_sigma=params.flow_exploration_sigma,
                num_ode_steps=params.flow_num_steps,
                solver=params.flow_solver,
            )

        super().__init__(model, params, cost_function, noise_sampler)

        self.flow_params = params

        # 데이터 수집기
        self._data_collector = FlowDataCollector(
            buffer_size=params.flow_buffer_size
        )

        # 학습기 (lazy init)
        self._trainer = None

        # 온라인 학습 카운터
        self._step_count = 0

        # Flow 모델 로드 (경로 지정 시)
        if params.flow_model_path is not None:
            self.load_flow_model(params.flow_model_path)

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Flow-MPPI 제어 계산

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + flow_stats
        """
        # 1. Flow 샘플러에 상태 컨텍스트 설정
        if isinstance(self.noise_sampler, FlowMatchingSampler):
            self.noise_sampler.set_context(state)

        # 2. 표준 MPPI 파이프라인 (sampler가 flow/gaussian 자동 선택)
        control, info = super().compute_control(state, reference_trajectory)

        # 3. 최적 제어 시퀀스를 데이터 버퍼에 저장
        self._data_collector.add_sample(state, self.U.copy())

        # 4. 온라인 학습 트리거
        self._step_count += 1
        if self.flow_params.flow_online_training:
            self._maybe_online_train()

        # 5. Flow 통계 추가
        info["flow_stats"] = self.get_flow_statistics()

        return control, info

    def _maybe_online_train(self):
        """주기적 온라인 학습 트리거"""
        if self._step_count % self.flow_params.flow_training_interval != 0:
            return
        if not self._data_collector.should_train(self.flow_params.flow_min_samples):
            return
        self.train_flow_model(epochs=20)

    def train_flow_model(
        self,
        data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
    ) -> Dict:
        """
        Flow 모델 학습 (오프라인 bootstrap 또는 온라인)

        Args:
            data: (states, controls) 튜플. None이면 내부 버퍼 사용.
            epochs: 에포크 수

        Returns:
            metrics: 학습 메트릭
        """
        from mppi_controller.learning.flow_matching_trainer import FlowMatchingTrainer

        if data is not None:
            states, controls = data
        else:
            if not self._data_collector.should_train(
                self.flow_params.flow_min_samples
            ):
                return {"status": "insufficient_data",
                        "num_samples": self._data_collector.num_samples}
            states, controls = self._data_collector.get_training_data()

        N, nu = self.params.N, self.model.control_dim
        control_seq_dim = N * nu
        context_dim = self.model.state_dim

        # Trainer 생성/재사용
        if self._trainer is None:
            self._trainer = FlowMatchingTrainer(
                control_seq_dim=control_seq_dim,
                context_dim=context_dim,
                hidden_dims=self.flow_params.flow_hidden_dims,
                device=self.flow_params.device,
            )

        metrics = self._trainer.train(states, controls, epochs=epochs)

        # 학습된 모델을 sampler에 주입
        if isinstance(self.noise_sampler, FlowMatchingSampler):
            self.noise_sampler.set_flow_model(self._trainer.get_model())

        metrics["status"] = "trained"
        return metrics

    def save_flow_model(self, path: str):
        """Flow 모델 저장"""
        if self._trainer is not None:
            self._trainer.save_model(path)

    def load_flow_model(self, path: str):
        """Flow 모델 로드"""
        from mppi_controller.learning.flow_matching_trainer import FlowMatchingTrainer
        import torch

        N, nu = self.params.N, self.model.control_dim
        control_seq_dim = N * nu
        context_dim = self.model.state_dim

        self._trainer = FlowMatchingTrainer(
            control_seq_dim=control_seq_dim,
            context_dim=context_dim,
            hidden_dims=self.flow_params.flow_hidden_dims,
            device=self.flow_params.device,
        )
        self._trainer.load_model(path)

        if isinstance(self.noise_sampler, FlowMatchingSampler):
            self.noise_sampler.set_flow_model(self._trainer.get_model())

    def get_flow_statistics(self) -> Dict:
        """Flow 통계 반환"""
        sampler = self.noise_sampler
        is_flow = isinstance(sampler, FlowMatchingSampler)
        return {
            "flow_ready": is_flow and sampler.is_flow_ready,
            "mode": self.flow_params.flow_mode,
            "buffer_size": self._data_collector.num_samples,
            "step_count": self._step_count,
            "online_training": self.flow_params.flow_online_training,
        }

    def reset(self):
        """상태 초기화"""
        super().reset()
        self._step_count = 0

    def __repr__(self) -> str:
        sampler = self.noise_sampler
        status = "ready" if (isinstance(sampler, FlowMatchingSampler)
                             and sampler.is_flow_ready) else "fallback"
        return (
            f"FlowMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"mode={self.flow_params.flow_mode}, "
            f"status={status})"
        )
