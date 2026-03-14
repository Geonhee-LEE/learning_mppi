"""
Diffusion-MPPI 컨트롤러

DDPM/DDIM 역확산 기반 MPPI 샘플러.
Flow-MPPI의 자연스러운 확장 — ODE(결정론적) 대신 SDE(확률적) 역확산.

특징:
    - DDIM 가속 역확산 (5~10 스텝으로 빠른 생성)
    - Classifier-free guidance 지원 (향후 확장)
    - Flow-MPPI와 동일한 self-supervised 학습 루프
    - 미학습 시 가우시안 fallback (= Vanilla MPPI)
    - online 학습: 매 N스텝마다 최적 제어 이력으로 학습

References:
    - Ho et al. (2020) — DDPM
    - Song et al. (2021) — DDIM
    - Chi et al. (2023) — Diffusion Policy (manipulation)
    - Kurtz & Burdick (2025) — GPC (Flow-MPPI의 유사 자기학습)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import DiffusionMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler
from mppi_controller.controllers.mppi.diffusion_sampler import DDIMSampler


class DiffusionMPPIController(MPPIController):
    """
    Diffusion-MPPI 컨트롤러 (15번째 MPPI 변형)

    MPPIController 상속, DDIMSampler로 다중 모달 샘플링.
    MPPI 자체 실행 이력에서 self-supervised DDPM 학습.

    학습 전: 순수 가우시안 샘플링 (= Vanilla MPPI)
    학습 후: DDIM 역확산으로 최적 제어 분포에서 직접 샘플링

    사용 방법:
        model = DifferentialDriveKinematic()
        params = DiffusionMPPIParams(N=20, K=512, diff_ddim_steps=5)
        ctrl = DiffusionMPPIController(model, params)

        # 실행 (처음엔 가우시안 fallback)
        u, info = ctrl.compute_control(state, ref)

        # 충분한 경험 수집 후 자동 학습
        # info['diffusion_is_trained'] == True 이면 학습됨

    Args:
        model: RobotModel 인스턴스
        params: DiffusionMPPIParams
        cost_function: CostFunction (None이면 기본)
        noise_sampler: NoiseSampler (None이면 DDIMSampler 자동 생성)
    """

    def __init__(
        self,
        model: RobotModel,
        params: DiffusionMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        # DDIMSampler 자동 생성
        if noise_sampler is None:
            noise_sampler = DDIMSampler(
                sigma=params.sigma,
                ddim_steps=params.diff_ddim_steps,
                T=params.diff_T,
                beta_schedule=params.diff_beta_schedule,
                mode=params.diff_mode,
                blend_ratio=params.diff_blend_ratio,
                exploration_sigma=params.diff_exploration_sigma,
                guidance_scale=params.diff_guidance_scale,
            )

        super().__init__(model, params, cost_function, noise_sampler)

        self.diff_params = params
        self._sampler: DDIMSampler = noise_sampler

        # 데이터 수집기 및 학습기
        self._trainer = None
        self._step_count = 0

        # 학습기 lazy 초기화 플래그
        self._trainer_initialized = False

        # 사전 학습 모델 로드
        if params.diff_model_path is not None:
            self.load_diffusion_model(params.diff_model_path)

    def _init_trainer(self, nx: int, N: int, nu: int) -> None:
        """학습기 lazy 초기화."""
        if self._trainer_initialized:
            return

        try:
            from mppi_controller.learning.diffusion_trainer import DiffusionTrainer
            self._trainer = DiffusionTrainer(
                control_seq_dim=N * nu,
                context_dim=nx,
                hidden_dims=self.diff_params.diff_hidden_dims,
                T=self.diff_params.diff_T,
                beta_schedule=self.diff_params.diff_beta_schedule,
            )
            self._trainer_initialized = True
        except ImportError:
            self._trainer = None

    def compute_control(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Diffusion-MPPI 제어 계산.

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 참조 궤적

        Returns:
            control: (nu,)
            info: {
                'sample_trajectories': (K, N+1, nx),
                'sample_weights': (K,),
                'best_trajectory': (N+1, nx),
                'ess': float,
                'temperature': float,
                'diffusion_is_trained': bool,
                'diffusion_step_count': int,
            }
        """
        # 컨텍스트 업데이트 (현재 상태)
        self._sampler.set_context(state)

        # 기본 MPPI 실행
        control, info = super().compute_control(state, reference_trajectory)

        # 온라인 학습: 최적 제어 시퀀스 수집
        if self.diff_params.diff_online_training:
            self._step_count += 1

            # 학습기 초기화
            nx = state.shape[0]
            N, nu = self.U.shape
            self._init_trainer(nx, N, nu)

            if self._trainer is not None:
                # 현재 최적 U를 학습 샘플로 추가
                self._trainer.add_sample(state, self.U.copy())

                # 주기적 학습
                if (self._step_count % self.diff_params.diff_training_interval == 0
                        and len(self._trainer._control_seqs) >= self.diff_params.diff_min_samples):
                    self.train_diffusion(epochs=5, verbose=False)

        # 정보 추가
        info["diffusion_is_trained"] = self._sampler._is_trained
        info["diffusion_step_count"] = self._step_count
        info["diffusion_mode"] = self.diff_params.diff_mode

        return control, info

    def train_diffusion(
        self,
        epochs: int = 20,
        verbose: bool = True,
    ) -> Dict:
        """
        DDPM 학습 실행.

        Args:
            epochs: 학습 에포크 수
            verbose: 로그 출력 여부

        Returns:
            stats: 학습 통계
        """
        if self._trainer is None:
            raise RuntimeError("학습기가 초기화되지 않았습니다. compute_control을 먼저 실행하세요.")

        stats = self._trainer.train(epochs=epochs, verbose=verbose)

        if stats["n_samples"] >= self.diff_params.diff_min_samples:
            # 학습된 모델을 샘플러에 주입
            model = self._trainer.get_model()
            self._sampler.set_model(model, trained=True)

        return stats

    def bootstrap_diffusion(
        self,
        initial_states: np.ndarray,
        initial_controls: np.ndarray,
        epochs: int = 50,
        verbose: bool = True,
    ) -> Dict:
        """
        오프라인 부트스트랩 학습.

        사전 수집된 (상태, 제어 시퀀스) 데이터로 초기 학습.

        Args:
            initial_states: (M, nx) 초기 상태 배치
            initial_controls: (M, N, nu) 최적 제어 배치
            epochs: 학습 에포크 수

        Returns:
            stats: 학습 통계
        """
        nx = initial_states.shape[1]
        M, N, nu = initial_controls.shape
        self._init_trainer(nx, N, nu)

        if self._trainer is None:
            raise RuntimeError("PyTorch가 필요합니다")

        self._trainer.add_batch(initial_states, initial_controls)
        stats = self._trainer.train(epochs=epochs, verbose=verbose)

        if stats["n_samples"] >= 10:
            model = self._trainer.get_model()
            self._sampler.set_model(model, trained=True)

        return stats

    def save_diffusion_model(self, path: str) -> None:
        """학습된 Diffusion 모델 저장."""
        if self._trainer is None:
            raise RuntimeError("저장할 학습된 모델이 없습니다.")
        self._trainer.save(path)

    def load_diffusion_model(self, path: str) -> None:
        """사전 학습된 Diffusion 모델 로드."""
        try:
            import torch
            from mppi_controller.models.learned.diffusion_model import MLPDiffusionModel

            ckpt = torch.load(path, weights_only=False)
            control_seq_dim = ckpt.get("control_seq_dim", self.U.numel() if hasattr(self.U, 'numel') else self.params.N * self.model.control_dim)
            context_dim = ckpt.get("context_dim", self.model.state_dim)

            model = MLPDiffusionModel(
                control_seq_dim=control_seq_dim,
                context_dim=context_dim,
                hidden_dims=self.diff_params.diff_hidden_dims,
            )
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            self._sampler.set_model(model, trained=True)
        except Exception as e:
            print(f"Diffusion 모델 로드 실패: {e}")

    @property
    def is_diffusion_trained(self) -> bool:
        """Diffusion 모델 학습 여부."""
        return self._sampler._is_trained
