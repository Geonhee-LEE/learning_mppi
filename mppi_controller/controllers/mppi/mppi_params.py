"""
MPPI 파라미터 데이터클래스

모든 MPPI 컨트롤러가 사용하는 파라미터 정의.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class MPPIParams:
    """
    MPPI 컨트롤러 파라미터

    Attributes:
        N: 예측 호라이즌 (타임스텝)
        dt: 타임스텝 간격 (초)
        K: 샘플 궤적 수
        lambda_: 온도 파라미터 (작을수록 최적 궤적에 집중)
        sigma: 제어 노이즈 표준편차 (nu,) 또는 스칼라
        Q: 상태 추적 비용 가중치 (nx,) 또는 (nx, nx)
        R: 제어 노력 비용 가중치 (nu,) 또는 (nu, nu)
        Qf: 터미널 상태 비용 가중치 (nx,) 또는 (nx, nx)
        u_min: 제어 입력 하한 (nu,) - None이면 모델의 제약 사용
        u_max: 제어 입력 상한 (nu,) - None이면 모델의 제약 사용
        device: 'cpu' 또는 'cuda' (GPU 가속용)
    """

    # 기본 파라미터
    N: int = 30  # 호라이즌
    dt: float = 0.05  # 50ms
    K: int = 1024  # 샘플 수

    # 온도 및 노이즈
    lambda_: float = 1.0  # 온도 파라미터
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))  # 제어 노이즈

    # 비용 함수 가중치
    Q: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 10.0, 1.0])
    )  # [x, y, θ] 가중치
    R: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.1])
    )  # [v, ω] 가중치
    Qf: Optional[np.ndarray] = None  # 터미널 비용 (None이면 Q 사용)

    # 제어 제약 (None이면 모델 제약 사용)
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None

    # 디바이스 설정
    device: str = "cpu"  # 'cpu' or 'cuda'

    def __post_init__(self):
        """파라미터 검증 및 자동 설정"""
        # sigma를 ndarray로 변환
        if not isinstance(self.sigma, np.ndarray):
            self.sigma = np.array([self.sigma])

        # Q, R, Qf를 ndarray로 변환
        if not isinstance(self.Q, np.ndarray):
            self.Q = np.array(self.Q)
        if not isinstance(self.R, np.ndarray):
            self.R = np.array(self.R)

        # Qf가 None이면 Q와 동일하게 설정
        if self.Qf is None:
            self.Qf = self.Q.copy()
        elif not isinstance(self.Qf, np.ndarray):
            self.Qf = np.array(self.Qf)

        # u_min, u_max를 ndarray로 변환
        if self.u_min is not None and not isinstance(self.u_min, np.ndarray):
            self.u_min = np.array(self.u_min)
        if self.u_max is not None and not isinstance(self.u_max, np.ndarray):
            self.u_max = np.array(self.u_max)

        # 파라미터 검증
        assert self.N > 0, "N must be positive"
        assert self.dt > 0, "dt must be positive"
        assert self.K > 0, "K must be positive"
        assert self.lambda_ > 0, "lambda_ must be positive"
        assert np.all(self.sigma > 0), "sigma must be positive"
        assert np.all(self.Q >= 0), "Q must be non-negative"
        assert np.all(self.R >= 0), "R must be non-negative"
        assert np.all(self.Qf >= 0), "Qf must be non-negative"

        if self.u_min is not None and self.u_max is not None:
            assert np.all(self.u_min < self.u_max), "u_min must be less than u_max"

    def get_control_bounds(self):
        """제어 제약 반환 (있을 경우)"""
        if self.u_min is not None and self.u_max is not None:
            return (self.u_min, self.u_max)
        return None

    def __repr__(self) -> str:
        return (
            f"MPPIParams("
            f"N={self.N}, dt={self.dt}, K={self.K}, "
            f"lambda_={self.lambda_:.2f}, "
            f"device={self.device})"
        )


@dataclass
class TubeMPPIParams(MPPIParams):
    """
    Tube-MPPI 전용 추가 파라미터

    Attributes:
        tube_enabled: Tube-MPPI 활성화 (False면 Vanilla MPPI)
        K_fb: 피드백 게인 행렬 (nu, nx)
        tube_margin: Tube 마진 (m)
    """

    tube_enabled: bool = True
    K_fb: Optional[np.ndarray] = None  # 피드백 게인
    tube_margin: float = 0.1  # Tube 마진 (m)

    def __post_init__(self):
        super().__post_init__()
        if self.K_fb is not None and not isinstance(self.K_fb, np.ndarray):
            self.K_fb = np.array(self.K_fb)


@dataclass
class LogMPPIParams(MPPIParams):
    """
    Log-MPPI 전용 추가 파라미터

    Attributes:
        use_baseline: Baseline 적용 (최소 비용으로 정규화)
    """

    use_baseline: bool = True

    def __post_init__(self):
        super().__post_init__()


@dataclass
class TsallisMPPIParams(MPPIParams):
    """
    Tsallis-MPPI 전용 추가 파라미터

    Attributes:
        tsallis_q: Tsallis 엔트로피 파라미터 (1.0이면 Vanilla MPPI)
    """

    tsallis_q: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.tsallis_q > 0, "tsallis_q must be positive"


@dataclass
class RiskAwareMPPIParams(MPPIParams):
    """
    Risk-Aware MPPI 전용 추가 파라미터

    Attributes:
        cvar_alpha: CVaR 위험 파라미터 (0~1, 1이면 Vanilla MPPI)
    """

    cvar_alpha: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.cvar_alpha <= 1, "cvar_alpha must be in (0, 1]"


@dataclass
class SteinVariationalMPPIParams(MPPIParams):
    """
    Stein Variational MPPI 전용 추가 파라미터

    Attributes:
        svgd_num_iterations: SVGD 반복 횟수
        svgd_step_size: SVGD 스텝 크기
    """

    svgd_num_iterations: int = 10
    svgd_step_size: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        assert self.svgd_num_iterations > 0, "svgd_num_iterations must be positive"
        assert self.svgd_step_size > 0, "svgd_step_size must be positive"


@dataclass
class SmoothMPPIParams(MPPIParams):
    """
    Smooth MPPI 전용 추가 파라미터

    Attributes:
        jerk_weight: Jerk 비용 가중치 (ΔΔu 페널티)
    """

    jerk_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.jerk_weight >= 0, "jerk_weight must be non-negative"


@dataclass
class SplineMPPIParams(MPPIParams):
    """
    Spline-MPPI 전용 추가 파라미터

    Attributes:
        spline_num_knots: B-spline knot 개수
        spline_degree: B-spline 차수
    """

    spline_num_knots: int = 8
    spline_degree: int = 3

    def __post_init__(self):
        super().__post_init__()
        assert self.spline_num_knots > 0, "spline_num_knots must be positive"
        assert self.spline_degree > 0, "spline_degree must be positive"
        assert (
            self.spline_num_knots > self.spline_degree
        ), "spline_num_knots must be greater than spline_degree"


@dataclass
class SVGMPPIParams(SteinVariationalMPPIParams):
    """
    SVG-MPPI 전용 추가 파라미터

    Attributes:
        svg_num_guide_particles: Guide particle 개수
        svg_guide_step_size: Guide particle 스텝 크기
    """

    svg_num_guide_particles: int = 10
    svg_guide_step_size: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.svg_num_guide_particles > 0
        ), "svg_num_guide_particles must be positive"
        assert self.svg_guide_step_size > 0, "svg_guide_step_size must be positive"


@dataclass
class CBFMPPIParams(MPPIParams):
    """
    CBF-MPPI 전용 추가 파라미터

    Attributes:
        cbf_obstacles: 장애물 리스트 [(x, y, radius), ...]
        cbf_weight: CBF 위반 비용 가중치
        cbf_alpha: Class-K function 파라미터 (0 < alpha <= 1)
        cbf_safety_margin: 추가 안전 마진 (m)
        cbf_use_safety_filter: QP 안전 필터 사용 여부
    """

    cbf_obstacles: List[tuple] = field(default_factory=list)
    cbf_weight: float = 1000.0
    cbf_alpha: float = 0.1
    cbf_safety_margin: float = 0.1
    cbf_use_safety_filter: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.cbf_alpha <= 1.0, "cbf_alpha must be in (0, 1]"
        assert self.cbf_weight >= 0, "cbf_weight must be non-negative"
        assert self.cbf_safety_margin >= 0, "cbf_safety_margin must be non-negative"


@dataclass
class ShieldMPPIParams(CBFMPPIParams):
    """
    Shield-MPPI 전용 추가 파라미터

    Rollout 중 매 timestep마다 CBF 제약을 해석적으로 적용하여
    모든 K개 샘플 궤적이 안전하도록 보장.

    Attributes:
        shield_enabled: Shield 기능 활성화 (False면 CBF-MPPI 폴백)
        shield_cbf_alpha: Shield용 CBF alpha (None이면 cbf_alpha 사용)
    """

    shield_enabled: bool = True
    shield_cbf_alpha: Optional[float] = None  # None이면 cbf_alpha 사용

    def __post_init__(self):
        super().__post_init__()
        if self.shield_cbf_alpha is not None:
            assert 0 < self.shield_cbf_alpha <= 1.0, \
                "shield_cbf_alpha must be in (0, 1]"


@dataclass
class ConformalCBFMPPIParams(ShieldMPPIParams):
    """
    Conformal Prediction + Shield-MPPI 파라미터

    CP 기반 동적 안전 마진으로 Shield-MPPI의 고정 마진을 대체.
    - 모델 정확 시: 마진 축소 → 성능 향상
    - 모델 부정확 시: 마진 확대 → 안전성 향상
    """

    cp_alpha: float = 0.1  # CP 실패율 (0.1 → 90%)
    cp_window_size: int = 200  # 슬라이딩 윈도우
    cp_min_samples: int = 10  # 최소 샘플
    cp_gamma: float = 0.95  # ACP 감쇠 (1.0=표준)
    cp_margin_min: float = 0.02  # 최소 마진 (m)
    cp_margin_max: float = 0.5  # 최대 마진 (m)
    cp_score_type: str = "position_norm"
    cp_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.cp_alpha < 1, "cp_alpha must be in (0, 1)"
        assert self.cp_window_size > 0, "cp_window_size must be positive"
        assert 0 < self.cp_gamma <= 1, "cp_gamma must be in (0, 1]"
        assert self.cp_margin_max > self.cp_margin_min >= 0, \
            "cp_margin_max must be > cp_margin_min >= 0"


@dataclass
class UncertaintyMPPIParams(MPPIParams):
    """
    Uncertainty-Aware MPPI 전용 추가 파라미터

    모델 불확실성에 비례하여 샘플링 노이즈를 적응 조절.

    Attributes:
        exploration_factor: 불확실성→노이즈 변환 계수
        min_sigma_ratio: 최소 sigma 비율 (base 대비)
        max_sigma_ratio: 최대 sigma 비율 (base 대비)
        uncertainty_strategy: 불확실성 추정 전략
            - "previous_trajectory": 직전 최적 궤적 재사용 (기본, 비용 0)
            - "current_state": 현재 상태 불확실성으로 전역 스케일
            - "two_pass": 1차 rollout → 불확실성 → 2차 적응 rollout
    """

    exploration_factor: float = 1.0
    min_sigma_ratio: float = 0.3
    max_sigma_ratio: float = 3.0
    uncertainty_strategy: str = "previous_trajectory"

    def __post_init__(self):
        super().__post_init__()
        assert self.exploration_factor >= 0, "exploration_factor must be non-negative"
        assert self.min_sigma_ratio > 0, "min_sigma_ratio must be positive"
        assert self.max_sigma_ratio >= self.min_sigma_ratio, \
            "max_sigma_ratio must be >= min_sigma_ratio"
        assert self.uncertainty_strategy in (
            "previous_trajectory", "current_state", "two_pass"
        ), f"Unknown uncertainty_strategy: {self.uncertainty_strategy}"


@dataclass
class DIALMPPIParams(MPPIParams):
    """
    DIAL-MPPI 전용 추가 파라미터

    DIAL-MPC (Diffusion Annealing for MPPI) 기반 다단계 확산 어닐링.
    표준 MPPI를 다중 반복 + 노이즈 어닐링으로 확장하여 local minima 회피.

    Attributes:
        n_diffuse_init: 첫 호출 확산 반복 횟수 (cold start)
        n_diffuse: 런타임 확산 반복 횟수 (warm start)
        traj_diffuse_factor: 반복 i에서 노이즈 스케일 × factor^i
        horizon_diffuse_factor: 타임스텝 t에서 노이즈 × factor^(N-1-t)
        sigma_scale: 기본 노이즈 스케일 승수
        use_reward_normalization: (r-mean)/std/λ 정규화 활성화
    """

    n_diffuse_init: int = 10
    n_diffuse: int = 3
    traj_diffuse_factor: float = 0.5
    horizon_diffuse_factor: float = 0.5
    sigma_scale: float = 1.0
    use_reward_normalization: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.n_diffuse_init > 0, "n_diffuse_init must be positive"
        assert self.n_diffuse > 0, "n_diffuse must be positive"
        assert 0 < self.traj_diffuse_factor <= 1.0, \
            "traj_diffuse_factor must be in (0, 1]"
        assert 0 < self.horizon_diffuse_factor <= 1.0, \
            "horizon_diffuse_factor must be in (0, 1]"
        assert self.sigma_scale > 0, "sigma_scale must be positive"


@dataclass
class ShieldDIALMPPIParams(DIALMPPIParams):
    """
    Shield-DIAL-MPPI 전용 추가 파라미터

    DIAL-MPPI 어닐링 루프 + per-step CBF shield 결합.

    Attributes:
        cbf_obstacles: 장애물 리스트 [(x, y, radius), ...]
        cbf_alpha: CBF alpha (Class-K function 파라미터)
        cbf_safety_margin: 추가 안전 마진 (m)
        shield_enabled: Shield 활성화 (False면 DIAL-MPPI 폴백)
        shield_cbf_alpha: Shield용 CBF alpha
    """

    cbf_obstacles: List[tuple] = field(default_factory=list)
    cbf_alpha: float = 0.3
    cbf_safety_margin: float = 0.1
    shield_enabled: bool = True
    shield_cbf_alpha: float = 0.3

    def __post_init__(self):
        super().__post_init__()
        assert self.cbf_alpha > 0, "cbf_alpha must be positive"
        assert self.cbf_safety_margin >= 0, "cbf_safety_margin must be non-negative"
        assert self.shield_cbf_alpha > 0, "shield_cbf_alpha must be positive"


@dataclass
class AdaptiveShieldDIALMPPIParams(ShieldDIALMPPIParams):
    """
    Adaptive Shield-DIAL-MPPI 전용 추가 파라미터

    ShieldDIALMPPIParams + 거리/속도 기반 적응형 α(d,v).

    Attributes:
        alpha_base: 기본 CBF alpha (최대값, d >> d_safe일 때)
        alpha_dist: 최소 alpha 비율 (d << d_safe일 때 α = α_base × α_dist)
        alpha_vel: 속도 반응 계수 (α /= (1 + α_vel·|v|))
        k_dist: sigmoid 경사도
        d_safe: 안전 거리 기준 (m)
    """

    alpha_base: float = 0.3
    alpha_dist: float = 0.1
    alpha_vel: float = 0.5
    k_dist: float = 2.0
    d_safe: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        assert self.alpha_base > 0, "alpha_base must be positive"
        assert self.alpha_dist >= 0, "alpha_dist must be non-negative"
        assert self.alpha_vel >= 0, "alpha_vel must be non-negative"
        assert self.k_dist > 0, "k_dist must be positive"
        assert self.d_safe > 0, "d_safe must be positive"


@dataclass
class C2UMPPIParams(MPPIParams):
    """
    C2U-MPPI (Chance-Constrained Unscented MPPI) 전용 파라미터

    Unscented Transform으로 비선형 공분산 전파 + 확률적 기회 제약조건.

    Attributes:
        ut_alpha: σ-point 분산 스케일 (작을수록 평균 근처 집중)
        ut_beta: 사전 분포 정보 (가우시안=2)
        ut_kappa: 2차 스케일링 파라미터
        chance_alpha: 충돌 허용 확률 상한 P(collision) ≤ chance_alpha
        chance_cost_weight: 기회 제약 위반 비용 가중치
        cc_obstacles: 원형 장애물 리스트 [(x, y, radius), ...]
        cc_margin_factor: κ_α 스케일 조정 계수 (1.0 = 이론적 정확)
        propagation_mode: "nominal" (1회 UT) | "per_sample" (K회 UT)
        process_noise_scale: 프로세스 노이즈 Q = scale * I
    """

    # Unscented Transform
    ut_alpha: float = 1e-3
    ut_beta: float = 2.0
    ut_kappa: float = 0.0

    # Chance Constraint
    chance_alpha: float = 0.05
    chance_cost_weight: float = 500.0
    cc_obstacles: List[tuple] = field(default_factory=list)
    cc_margin_factor: float = 1.0

    # Propagation
    propagation_mode: str = "nominal"
    process_noise_scale: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        assert self.ut_alpha > 0, "ut_alpha must be positive"
        assert self.ut_beta >= 0, "ut_beta must be non-negative"
        assert 0 < self.chance_alpha < 1, "chance_alpha must be in (0, 1)"
        assert self.chance_cost_weight >= 0, "chance_cost_weight must be non-negative"
        assert self.cc_margin_factor > 0, "cc_margin_factor must be positive"
        assert self.propagation_mode in ("nominal", "per_sample"), \
            f"Unknown propagation_mode: {self.propagation_mode}"
        assert self.process_noise_scale >= 0, "process_noise_scale must be non-negative"
