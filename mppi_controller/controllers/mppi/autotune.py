"""
MPPI Autotune 모듈

오프라인 파라미터 최적화 + 온라인 적응을 통한 자동 MPPI 튜닝.

구성:
    - AutotuneObjective: 메트릭 → 스칼라 목표값 변환
    - AutotuneConfig: 튜닝 설정 (탐색 범위, 옵티마이저 등)
    - MPPIAutotuner: scipy 기반 오프라인 파라미터 최적화
    - OnlineSigmaAdapter: ESS/cost-gap 기반 온라인 σ 적응 (CoVO-MPC 영감)
    - AutotunedMPPIController: 온라인 적응 래퍼 (15종 MPPI 변형 호환)

참고:
    - CoVO-MPC (L4DC 2024): Covariance-optimal control for MPPI
    - pytorch_mppi CMA-ES autotune
"""

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

from mppi_controller.controllers.mppi.sampling import NoiseSampler


# ═══════════════════════════════════════════════════════════════════
# 1. AutotuneObjective
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AutotuneObjective:
    """
    메트릭 딕셔너리 → 스칼라 목표값 변환

    objective = Σ (weight_i × metric_i / target_i)
    + constraint 위반 시 1e6 × (violation / limit) 페널티

    Args:
        metric_weights: 메트릭별 가중치 (높을수록 중요)
        metric_targets: 정규화 기준값 (없으면 1.0)
        penalty_constraints: 메트릭 제약 {이름: (타입, 한계값)}
            - ("max", 100.0): metric ≤ 100 제약
            - ("min", 0.1): metric ≥ 0.1 제약
    """

    metric_weights: Dict[str, float] = field(default_factory=dict)
    metric_targets: Dict[str, float] = field(default_factory=dict)
    penalty_constraints: Dict[str, Tuple[str, float]] = field(default_factory=dict)

    def evaluate(self, metrics: dict) -> float:
        """메트릭 → 스칼라 목표값 (낮을수록 좋음)"""
        objective = 0.0

        # 가중 정규화 합산
        for name, weight in self.metric_weights.items():
            if name not in metrics:
                continue
            value = metrics[name]
            target = self.metric_targets.get(name, 1.0)
            if target > 0:
                objective += weight * value / target
            else:
                objective += weight * value

        # 제약 위반 페널티
        for name, (constraint_type, limit) in self.penalty_constraints.items():
            if name not in metrics:
                continue
            value = metrics[name]
            if constraint_type == "max" and value > limit:
                objective += 1e6 * (value - limit) / max(abs(limit), 1e-8)
            elif constraint_type == "min" and value < limit:
                objective += 1e6 * (limit - value) / max(abs(limit), 1e-8)

        return objective

    @classmethod
    def tracking_focused(cls) -> "AutotuneObjective":
        """RMSE 중심 목표 함수"""
        return cls(
            metric_weights={
                "position_rmse": 1.0,
                "heading_rmse": 0.3,
            },
            metric_targets={
                "position_rmse": 0.1,
                "heading_rmse": 0.1,
            },
            penalty_constraints={
                "mean_solve_time": ("max", 100.0),
            },
        )

    @classmethod
    def balanced(cls) -> "AutotuneObjective":
        """RMSE + 부드러움 + 속도 균형 목표 함수"""
        return cls(
            metric_weights={
                "position_rmse": 1.0,
                "heading_rmse": 0.2,
                "control_rate": 0.3,
                "mean_solve_time": 0.1,
            },
            metric_targets={
                "position_rmse": 0.1,
                "heading_rmse": 0.1,
                "control_rate": 0.5,
                "mean_solve_time": 50.0,
            },
            penalty_constraints={
                "mean_solve_time": ("max", 100.0),
            },
        )

    @classmethod
    def safety_focused(cls) -> "AutotuneObjective":
        """안전성 중심 목표 함수"""
        return cls(
            metric_weights={
                "position_rmse": 0.5,
                "max_position_error": 1.0,
                "control_rate": 0.3,
            },
            metric_targets={
                "position_rmse": 0.1,
                "max_position_error": 0.3,
                "control_rate": 0.5,
            },
            penalty_constraints={
                "max_position_error": ("max", 1.0),
                "mean_solve_time": ("max", 100.0),
            },
        )


# ═══════════════════════════════════════════════════════════════════
# 2. AutotuneConfig
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AutotuneConfig:
    """
    오프라인 튜닝 설정

    ndarray 파라미터(σ, Q, R)는 스케일링 팩터로 탐색 (1차원화).
    - sigma_scale=1.5 → sigma = base_sigma * 1.5

    Args:
        tunable_params: 튜닝 대상 파라미터 이름 리스트
        param_bounds: 파라미터별 탐색 범위 {이름: (하한, 상한)}
        optimizer: scipy 최적화 알고리즘
        max_iterations: 최대 반복 횟수
        n_sim_repeats: 평가 반복 수 (노이즈 평균)
        sim_duration: 평가 시뮬레이션 시간 (초)
        seed: 랜덤 시드
        verbose: 진행 출력
    """

    tunable_params: List[str] = field(
        default_factory=lambda: ["lambda_", "sigma_scale", "Q_scale", "R_scale"]
    )
    param_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "lambda_": (0.1, 50.0),
            "sigma_scale": (0.1, 5.0),
            "Q_scale": (0.1, 50.0),
            "R_scale": (0.01, 10.0),
        }
    )
    optimizer: str = "differential_evolution"
    max_iterations: int = 50
    n_sim_repeats: int = 1
    sim_duration: float = 10.0
    seed: int = 42
    verbose: bool = True


# ═══════════════════════════════════════════════════════════════════
# 3. MPPIAutotuner — 오프라인 최적화
# ═══════════════════════════════════════════════════════════════════


class MPPIAutotuner:
    """
    scipy 기반 오프라인 MPPI 파라미터 최적화

    Args:
        model_fn: 모델 팩토리 (매 평가마다 새 인스턴스)
        controller_cls: MPPI 컨트롤러 클래스 (MPPIController 또는 서브클래스)
        base_params: 기본 파라미터 (deepcopy 후 tunable 필드만 교체)
        reference_fn: t → (N+1, nx) 레퍼런스 궤적 함수
        initial_state: (nx,) 초기 상태
        objective: AutotuneObjective 인스턴스
        config: AutotuneConfig 인스턴스
        cost_function: 커스텀 비용 함수 (None이면 기본)
    """

    def __init__(
        self,
        model_fn: Callable,
        controller_cls: Type,
        base_params,
        reference_fn: Callable,
        initial_state: np.ndarray,
        objective: AutotuneObjective,
        config: AutotuneConfig,
        cost_function=None,
    ):
        self.model_fn = model_fn
        self.controller_cls = controller_cls
        self.base_params = base_params
        self.reference_fn = reference_fn
        self.initial_state = initial_state.copy()
        self.objective = objective
        self.config = config
        self.cost_function = cost_function

        self._eval_count = 0
        self._best_score = float("inf")
        self._history: List[dict] = []

    def tune(self) -> Tuple[object, dict]:
        """
        최적 파라미터 탐색

        Returns:
            best_params: 최적화된 MPPIParams
            result_info: 최적화 결과 정보
        """
        from scipy import optimize

        bounds = [
            self.config.param_bounds[p]
            for p in self.config.tunable_params
        ]

        self._eval_count = 0
        self._best_score = float("inf")
        self._history = []

        if self.config.optimizer == "differential_evolution":
            result = optimize.differential_evolution(
                self._evaluate_params,
                bounds,
                maxiter=self.config.max_iterations,
                seed=self.config.seed,
                tol=1e-4,
                atol=1e-4,
                init="sobol",
                polish=False,
            )
        elif self.config.optimizer == "nelder_mead":
            x0 = self._params_to_vector(self.base_params)
            result = optimize.minimize(
                self._evaluate_params,
                x0,
                method="Nelder-Mead",
                options={"maxiter": self.config.max_iterations},
            )
        elif self.config.optimizer == "powell":
            x0 = self._params_to_vector(self.base_params)
            result = optimize.minimize(
                self._evaluate_params,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": self.config.max_iterations},
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        best_params = self._vector_to_params(result.x)

        result_info = {
            "success": result.success if hasattr(result, "success") else True,
            "best_score": float(result.fun),
            "n_evaluations": self._eval_count,
            "best_vector": result.x.tolist(),
            "param_names": self.config.tunable_params,
            "history": self._history,
        }

        return best_params, result_info

    def _evaluate_params(self, param_vector: np.ndarray) -> float:
        """벡터 → 파라미터 → 시뮬레이션 → 메트릭 → 목표값"""
        from mppi_controller.simulation.simulator import Simulator
        from mppi_controller.simulation.metrics import compute_metrics

        self._eval_count += 1

        try:
            params = self._vector_to_params(param_vector)
        except (AssertionError, ValueError):
            return 1e10

        scores = []
        for repeat in range(self.config.n_sim_repeats):
            try:
                model = self.model_fn()
                if self.cost_function is not None:
                    controller = self.controller_cls(
                        model, params, cost_function=self.cost_function
                    )
                else:
                    controller = self.controller_cls(model, params)

                sim = Simulator(model, controller, params.dt, store_info=False)
                sim.reset(self.initial_state.copy())

                history = sim.run(self.reference_fn, self.config.sim_duration)
                metrics = compute_metrics(history)
                score = self.objective.evaluate(metrics)
                scores.append(score)
            except Exception:
                scores.append(1e10)

        mean_score = float(np.mean(scores))

        if mean_score < self._best_score:
            self._best_score = mean_score

        self._history.append({
            "eval": self._eval_count,
            "score": mean_score,
            "best_score": self._best_score,
            "params": param_vector.tolist(),
        })

        if self.config.verbose and self._eval_count % 10 == 0:
            print(
                f"  [Autotune] eval={self._eval_count:4d}  "
                f"score={mean_score:.4f}  best={self._best_score:.4f}"
            )

        return mean_score

    def _vector_to_params(self, vec: np.ndarray) -> object:
        """최적화 벡터 → MPPIParams 변환"""
        params = copy.deepcopy(self.base_params)

        for i, name in enumerate(self.config.tunable_params):
            value = vec[i]
            if name == "lambda_":
                params.lambda_ = float(value)
            elif name == "sigma_scale":
                params.sigma = self.base_params.sigma * float(value)
            elif name == "Q_scale":
                params.Q = self.base_params.Q * float(value)
                params.Qf = params.Q.copy()
            elif name == "R_scale":
                params.R = self.base_params.R * float(value)
            else:
                if hasattr(params, name):
                    setattr(params, name, float(value))

        # __post_init__ 검증 재실행
        params.__post_init__()
        return params

    def _params_to_vector(self, params) -> np.ndarray:
        """MPPIParams → 최적화 벡터 (초기점 생성용)"""
        vec = []
        for name in self.config.tunable_params:
            if name == "lambda_":
                vec.append(params.lambda_)
            elif name == "sigma_scale":
                vec.append(1.0)  # 기본 스케일
            elif name == "Q_scale":
                vec.append(1.0)
            elif name == "R_scale":
                vec.append(1.0)
            elif hasattr(params, name):
                vec.append(getattr(params, name))
            else:
                vec.append(1.0)
        return np.array(vec)


# ═══════════════════════════════════════════════════════════════════
# 4. OnlineSigmaAdapter — CoVO-MPC 영감 온라인 σ 적응
# ═══════════════════════════════════════════════════════════════════


class OnlineSigmaAdapter(NoiseSampler):
    """
    ESS/cost-gap 기반 온라인 σ 적응 (CoVO-MPC 근사)

    적응 규칙:
        - ESS ratio < 0.3 (소수 샘플에 집중) → σ 증가 (탐색 확대)
        - ESS ratio > 0.7 & cost_gap < 0.1 → σ 감소 (활용 강화)

    NoiseSampler 상속 → controller.noise_sampler = adapter로 무수정 교체 가능

    Args:
        base_sigma: (nu,) 기본 노이즈 표준편차
        adaptation_rate: 적응 속도 (0~1)
        min_sigma_ratio: 최소 sigma 비율
        max_sigma_ratio: 최대 sigma 비율
        seed: 랜덤 시드
    """

    def __init__(
        self,
        base_sigma: np.ndarray,
        adaptation_rate: float = 0.1,
        min_sigma_ratio: float = 0.3,
        max_sigma_ratio: float = 3.0,
        seed: Optional[int] = None,
    ):
        self.base_sigma = np.asarray(base_sigma, dtype=float)
        self.adaptation_rate = adaptation_rate
        self.min_sigma_ratio = min_sigma_ratio
        self.max_sigma_ratio = max_sigma_ratio
        self.rng = np.random.default_rng(seed)

        # 시간별 sigma 비율 (update() 전에는 None → 1.0 사용)
        self._sigma_ratios: Optional[np.ndarray] = None

        # 통계 히스토리
        self._ess_history: List[float] = []
        self._mean_ratio_history: List[float] = []
        self._cost_gap_history: List[float] = []

    def sample(
        self,
        U: np.ndarray,
        K: int,
        control_min: Optional[np.ndarray] = None,
        control_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """적응된 sigma로 노이즈 샘플링"""
        N, nu = U.shape

        if self._sigma_ratios is not None and len(self._sigma_ratios) == N:
            # 시간별 적응 sigma: (N, nu)
            sigma_profile = self._sigma_ratios[:, None] * self.base_sigma[None, :]
        else:
            sigma_profile = np.broadcast_to(self.base_sigma, (N, nu)).copy()

        noise = self.rng.normal(0.0, 1.0, (K, N, nu)) * sigma_profile[None, :, :]

        if control_min is not None and control_max is not None:
            sampled_controls = U + noise
            sampled_controls = np.clip(sampled_controls, control_min, control_max)
            noise = sampled_controls - U

        return noise

    def update(self, info: dict):
        """
        compute_control() info 딕셔너리로 σ 적응 업데이트

        Args:
            info: MPPI info dict (ess, num_samples, best_cost, mean_cost 포함)
        """
        ess = info.get("ess", 0.0)
        K = info.get("num_samples", 1)
        best_cost = info.get("best_cost", 0.0)
        mean_cost = info.get("mean_cost", 0.0)

        ess_ratio = ess / max(K, 1)

        # cost gap (mean과 best 사이의 상대적 차이)
        if abs(mean_cost) > 1e-8:
            cost_gap = (mean_cost - best_cost) / abs(mean_cost)
        else:
            cost_gap = 0.0
        cost_gap = max(cost_gap, 0.0)

        # 현재 호라이즌 길이 결정
        sample_traj = info.get("sample_trajectories", None)
        if sample_traj is not None:
            N = sample_traj.shape[1] - 1  # (K, N+1, nx) → N
        else:
            N = len(self._sigma_ratios) if self._sigma_ratios is not None else 30

        # 초기화 (첫 호출)
        if self._sigma_ratios is None or len(self._sigma_ratios) != N:
            self._sigma_ratios = np.ones(N)

        rate = self.adaptation_rate

        if ess_ratio < 0.3:
            # ESS 낮음 → 소수 샘플에 집중 → σ 증가 (탐색 확대)
            self._sigma_ratios *= (1.0 + rate)
        elif ess_ratio > 0.7 and cost_gap < 0.1:
            # ESS 높음 + gap 작음 → σ 감소 (활용 강화)
            self._sigma_ratios *= (1.0 - rate * 0.5)

        # 클리핑
        self._sigma_ratios = np.clip(
            self._sigma_ratios, self.min_sigma_ratio, self.max_sigma_ratio
        )

        # 히스토리 저장
        self._ess_history.append(ess_ratio)
        self._mean_ratio_history.append(float(np.mean(self._sigma_ratios)))
        self._cost_gap_history.append(cost_gap)

    def reset(self):
        """적응 상태 초기화"""
        self._sigma_ratios = None
        self._ess_history.clear()
        self._mean_ratio_history.clear()
        self._cost_gap_history.clear()

    def get_statistics(self) -> dict:
        """현재 적응 통계 반환"""
        if self._sigma_ratios is None:
            return {
                "has_adapted": False,
                "mean_ratio": 1.0,
                "min_ratio": 1.0,
                "max_ratio": 1.0,
            }
        return {
            "has_adapted": True,
            "mean_ratio": float(np.mean(self._sigma_ratios)),
            "min_ratio": float(np.min(self._sigma_ratios)),
            "max_ratio": float(np.max(self._sigma_ratios)),
            "sigma_ratios": self._sigma_ratios.copy(),
            "ess_history": list(self._ess_history),
            "mean_ratio_history": list(self._mean_ratio_history),
            "cost_gap_history": list(self._cost_gap_history),
        }

    def __repr__(self) -> str:
        ratio_str = (
            f"{np.mean(self._sigma_ratios):.2f}"
            if self._sigma_ratios is not None
            else "1.00"
        )
        return (
            f"OnlineSigmaAdapter(base_sigma={self.base_sigma}, "
            f"mean_ratio={ratio_str})"
        )


# ═══════════════════════════════════════════════════════════════════
# 5. AutotunedMPPIController — 온라인 래퍼
# ═══════════════════════════════════════════════════════════════════


class AutotunedMPPIController:
    """
    온라인 적응 래퍼 — 15종 모든 MPPI 변형과 호환

    Wrapper 패턴으로 기존 컨트롤러를 감싸서 온라인 적응 기능 추가.
    sigma_adapter와 temperature_adapter를 개별 ON/OFF 가능.

    Args:
        controller: MPPI 컨트롤러 인스턴스 (어떤 변형이든)
        sigma_adapter: OnlineSigmaAdapter (None이면 σ 적응 비활성)
        temperature_adapter: AdaptiveTemperature (None이면 λ 적응 비활성)
    """

    def __init__(
        self,
        controller,
        sigma_adapter: Optional[OnlineSigmaAdapter] = None,
        temperature_adapter=None,
    ):
        self.controller = controller

        self.sigma_adapter = sigma_adapter
        self.temperature_adapter = temperature_adapter

        # σ adapter 장착
        if sigma_adapter is not None:
            self.controller.noise_sampler = sigma_adapter

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        제어 계산 + 온라인 적응

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict (autotune_stats 포함)
        """
        control, info = self.controller.compute_control(state, reference_trajectory)

        # σ 적응 업데이트
        if self.sigma_adapter is not None:
            self.sigma_adapter.update(info)

        # λ 적응 업데이트
        if self.temperature_adapter is not None:
            weights = info.get("sample_weights", None)
            K = info.get("num_samples", self.controller.params.K)
            if weights is not None:
                new_lambda = self.temperature_adapter.update(weights, K)
                self.controller.params.lambda_ = new_lambda

        # 적응 통계 추가
        autotune_stats = {}
        if self.sigma_adapter is not None:
            sigma_stats = self.sigma_adapter.get_statistics()
            autotune_stats["sigma_mean_ratio"] = sigma_stats["mean_ratio"]
        if self.temperature_adapter is not None:
            autotune_stats["lambda"] = self.controller.params.lambda_

        info["autotune_stats"] = autotune_stats

        return control, info

    def reset(self):
        """컨트롤러 + 어댑터 초기화"""
        self.controller.reset()
        if self.sigma_adapter is not None:
            self.sigma_adapter.reset()
        if self.temperature_adapter is not None:
            self.temperature_adapter.reset()

    @property
    def params(self):
        return self.controller.params

    @property
    def model(self):
        return self.controller.model

    def __repr__(self) -> str:
        adapters = []
        if self.sigma_adapter is not None:
            adapters.append("sigma")
        if self.temperature_adapter is not None:
            adapters.append("lambda")
        return (
            f"AutotunedMPPIController("
            f"controller={self.controller.__class__.__name__}, "
            f"adapters={adapters})"
        )
