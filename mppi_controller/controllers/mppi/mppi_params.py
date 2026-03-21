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
    SVG-MPPI (Stein Variational Guided MPPI) 전용 추가 파라미터

    Honda et al., ICRA 2024, arXiv:2309.11040 기반.
    SVGD로 파티클을 최적 분포 모드로 이동 후 MPPI 가중 평균.

    Attributes:
        svg_num_guide_particles: Guide particle 개수 (SVGD 적용 대상)
        svg_guide_step_size: Guide particle SVGD 스텝 크기
        n_svgd_steps: SVGD 업데이트 반복 수
        svgd_step_size_schedule: SVGD 스텝 크기 감쇠 ('constant' or 'decay')
        temperature_svgd: SVGD 내부 온도 (비용 스케일링)
        use_svgd_warm_start: 이전 SVGD 파티클 warm start
        blend_ratio: SVGD 파티클 vs 가우시안 혼합 비율 (0=전부 가우시안, 1=전부 SVGD)
        use_spsa_gradient: True=SPSA(빠름), False=finite diff(정확)
    """

    svg_num_guide_particles: int = 10
    svg_guide_step_size: float = 0.01

    # Honda et al. 2024 추가 파라미터
    n_svgd_steps: int = 5
    svgd_step_size_schedule: str = "constant"
    temperature_svgd: float = 1.0
    use_svgd_warm_start: bool = True
    blend_ratio: float = 0.5
    use_spsa_gradient: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.svg_num_guide_particles > 0
        ), "svg_num_guide_particles must be positive"
        assert self.svg_guide_step_size > 0, "svg_guide_step_size must be positive"
        assert self.n_svgd_steps >= 0, "n_svgd_steps must be non-negative"
        assert self.temperature_svgd > 0, "temperature_svgd must be positive"
        assert 0.0 <= self.blend_ratio <= 1.0, "blend_ratio must be in [0, 1]"
        assert self.svgd_step_size_schedule in (
            "constant", "decay",
        ), "svgd_step_size_schedule must be 'constant' or 'decay'"


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
class CMAMPPIParams(MPPIParams):
    """
    CMA-MPPI (Covariance Matrix Adaptation MPPI) 전용 파라미터

    CMA-ES 영감: 보상 가중 샘플로부터 per-timestep 대각 공분산을 학습.
    DIAL-MPPI의 등방적 고정 감쇠 대신, 비용 지형 적응적 탐색.

    Attributes:
        n_iters_init: Cold start 반복 횟수 (첫 호출)
        n_iters: Warm start 반복 횟수 (이후 호출)
        cov_learning_rate: 공분산 EMA 학습률 α ∈ (0, 1]
        sigma_min: 최소 σ (발산 방지)
        sigma_max: 최대 σ
        elite_ratio: 0=전체 가중치 사용, >0=상위 비율만
        use_mean_shift: True=전체 교체(DIAL식), False=증분(Vanilla식)
        use_reward_normalization: 보상 정규화 활성화
        cov_init_scale: 초기 공분산 = (sigma * scale)²
    """

    n_iters_init: int = 8
    n_iters: int = 3
    cov_learning_rate: float = 0.5
    sigma_min: float = 0.05
    sigma_max: float = 3.0
    elite_ratio: float = 0.0
    use_mean_shift: bool = True
    use_reward_normalization: bool = True
    cov_init_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.n_iters_init > 0, "n_iters_init must be positive"
        assert self.n_iters > 0, "n_iters must be positive"
        assert 0 < self.cov_learning_rate <= 1.0, \
            "cov_learning_rate must be in (0, 1]"
        assert self.sigma_min > 0, "sigma_min must be positive"
        assert self.sigma_max > self.sigma_min, \
            "sigma_max must be greater than sigma_min"
        assert 0 <= self.elite_ratio < 1, "elite_ratio must be in [0, 1)"
        assert self.cov_init_scale > 0, "cov_init_scale must be positive"


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
class DiffusionMPPIParams(MPPIParams):
    """
    Diffusion-MPPI 전용 추가 파라미터

    DDPM/DDIM 기반 역확산 샘플러를 활용한 MPPI.
    Flow-MPPI보다 표현력 높은 다중 모달 분포 근사.

    Attributes:
        diff_hidden_dims: 노이즈 예측 MLP 은닉층 차원
        diff_T: 학습 시 총 노이즈 스텝 수
        diff_ddim_steps: 추론 시 DDIM 스텝 수 (1~20)
        diff_beta_schedule: 노이즈 스케줄 ("cosine" | "linear")
        diff_mode: 샘플링 모드 ("replace" | "blend")
        diff_blend_ratio: blend 모드 diffusion 비율 (0~1)
        diff_exploration_sigma: 추가 탐색 노이즈 스케일
        diff_guidance_scale: Classifier-free guidance 스케일
        diff_model_path: 사전 학습 모델 경로
        diff_online_training: 온라인 학습 활성화
        diff_training_interval: 온라인 학습 주기 (스텝)
        diff_buffer_size: 데이터 버퍼 최대 크기
        diff_min_samples: 학습 시작 최소 샘플 수
    """

    diff_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    diff_T: int = 1000
    diff_ddim_steps: int = 5
    diff_beta_schedule: str = "cosine"
    diff_mode: str = "replace"
    diff_blend_ratio: float = 0.5
    diff_exploration_sigma: float = 0.3
    diff_guidance_scale: float = 1.0
    diff_model_path: Optional[str] = None
    diff_online_training: bool = False
    diff_training_interval: int = 100
    diff_buffer_size: int = 5000
    diff_min_samples: int = 200

    def __post_init__(self):
        super().__post_init__()
        assert self.diff_ddim_steps > 0, "diff_ddim_steps must be positive"
        assert self.diff_beta_schedule in ("cosine", "linear"), \
            f"Unknown beta_schedule: {self.diff_beta_schedule}"
        assert self.diff_mode in ("replace", "blend"), \
            f"Unknown diff_mode: {self.diff_mode}"
        assert 0 <= self.diff_blend_ratio <= 1, "diff_blend_ratio must be in [0, 1]"
        assert self.diff_T > 0, "diff_T must be positive"


@dataclass
class FlowMPPIParams(MPPIParams):
    """
    Flow-MPPI 전용 추가 파라미터

    Conditional Flow Matching 기반 다중 모달 샘플링 MPPI.
    학습된 속도장으로 가우시안 노이즈를 최적 제어 분포로 전송.

    Attributes:
        flow_hidden_dims: Flow 모델 은닉층 차원 리스트
        flow_num_steps: ODE 적분 스텝 수 (5~10)
        flow_solver: ODE solver ("euler" | "midpoint")
        flow_mode: 샘플링 통합 모드
            - "replace_mean": flow 출력을 평균으로, 가우시안 탐색 추가
            - "replace_distribution": flow로 K개 직접 생성
            - "blend": α*flow + (1-α)*gaussian 혼합
        flow_blend_ratio: blend 모드의 flow 비율 (0~1)
        flow_exploration_sigma: 탐색 노이즈 스케일 (replace_mean 모드)
        flow_model_path: 사전 학습 모델 경로
        flow_online_training: 온라인 학습 활성화
        flow_training_interval: 온라인 학습 주기 (스텝)
        flow_buffer_size: 데이터 버퍼 최대 크기
        flow_min_samples: 학습 시작 최소 샘플 수
    """

    flow_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    flow_num_steps: int = 5
    flow_solver: str = "euler"
    flow_mode: str = "replace_mean"
    flow_blend_ratio: float = 0.5
    flow_exploration_sigma: float = 0.5
    flow_model_path: Optional[str] = None
    flow_online_training: bool = False
    flow_training_interval: int = 100
    flow_buffer_size: int = 5000
    flow_min_samples: int = 200

    def __post_init__(self):
        super().__post_init__()
        assert self.flow_num_steps > 0, "flow_num_steps must be positive"
        assert self.flow_solver in ("euler", "midpoint"), \
            f"Unknown flow_solver: {self.flow_solver}"
        assert self.flow_mode in ("replace_mean", "replace_distribution", "blend"), \
            f"Unknown flow_mode: {self.flow_mode}"
        assert 0 <= self.flow_blend_ratio <= 1, "flow_blend_ratio must be in [0, 1]"
        assert self.flow_exploration_sigma >= 0, "flow_exploration_sigma must be non-negative"
        assert self.flow_training_interval > 0, "flow_training_interval must be positive"
        assert self.flow_buffer_size > 0, "flow_buffer_size must be positive"
        assert self.flow_min_samples > 0, "flow_min_samples must be positive"


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


@dataclass
class WBCMPPIParams(MPPIParams):
    """
    WBC-MPPI (Whole-Body Control MPPI) 전용 파라미터

    MobileManipulator6DOFKinematic (9D 상태, 8D 제어)와 함께 사용.
    베이스 + 팔 통합 최적화를 위한 전용 파라미터.

    Attributes:
        ee_pos_weight: EE 위치 추적 가중치
        ee_ori_weight: EE 자세 추적 가중치 (SO(3) geodesic)
        ee_terminal_pos_weight: EE 터미널 위치 비용 가중치
        ee_terminal_ori_weight: EE 터미널 자세 비용 가중치
        joint_limit_weight: 관절 한계 비용 가중치
        joint_penalty: 관절 한계 위반 시 페널티
        singularity_weight: 특이점 회피 비용 가중치
        singularity_threshold: σ_min 임계값
        reachability_weight: 도달가능성 비용 가중치
        max_arm_reach: 팔 최대 도달 반경 (m)
        min_arm_reach: 팔 최소 도달 반경 (m)
        base_vel_weight: 베이스 속도 페널티 가중치
        arm_effort_weight: 팔 관절 속도 비용 가중치
        smooth_weight: 관절 속도 부드러움 비용 가중치
        task_mode: 작업 모드 ("ee_tracking"|"navigation"|"both")
    """

    # EE 추적 가중치
    ee_pos_weight: float = 100.0
    ee_ori_weight: float = 10.0
    ee_terminal_pos_weight: float = 200.0
    ee_terminal_ori_weight: float = 20.0

    # 관절 제한
    joint_limit_weight: float = 50.0
    joint_penalty: float = 1e4

    # 특이점 회피
    singularity_weight: float = 10.0
    singularity_threshold: float = 0.02

    # 도달가능성
    reachability_weight: float = 30.0
    max_arm_reach: float = 0.70
    min_arm_reach: float = 0.10

    # 제어 비용
    base_vel_weight: float = 0.1
    arm_effort_weight: float = 0.3
    smooth_weight: float = 0.5

    # 작업 모드
    task_mode: str = "ee_tracking"

    def __post_init__(self):
        super().__post_init__()
        assert self.ee_pos_weight >= 0, "ee_pos_weight must be non-negative"
        assert self.ee_ori_weight >= 0, "ee_ori_weight must be non-negative"
        assert self.task_mode in ("ee_tracking", "navigation", "both"), \
            f"Unknown task_mode: {self.task_mode}"
        assert self.max_arm_reach > self.min_arm_reach, \
            "max_arm_reach must be greater than min_arm_reach"


@dataclass
class KernelMPPIParams(MPPIParams):
    """
    Kernel MPPI 전용 추가 파라미터

    RBF 커널 보간으로 소수 서포트 포인트에서 전체 제어 시퀀스를 복원.
    샘플링 차원 ~75% 감소 + 제어 평활도 향상.

    Attributes:
        num_support_pts: 서포트 포인트 수 S (S << N)
        kernel_bandwidth: RBF 커널 대역폭 σ
    """

    num_support_pts: int = 8
    kernel_bandwidth: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.num_support_pts > 0, "num_support_pts must be positive"
        assert self.num_support_pts <= self.N, \
            "num_support_pts must be <= N"
        assert self.kernel_bandwidth > 0, "kernel_bandwidth must be positive"


@dataclass
class BNNMPPIParams(MPPIParams):
    """
    BNN Surrogate MPPI 전용 파라미터

    앙상블 불확실성 기반 궤적 feasibility 평가 + 필터링.
    UncertaintyMPPI(샘플링 적응)와 달리, 비용 함수에서 불확실 궤적을 페널티/필터.

    Attributes:
        feasibility_weight: 불확실성 비용 가중치 β
        uncertainty_reduce: 상태 차원 축소 방법 ("sum" | "max" | "mean")
        feasibility_threshold: 최소 feasibility 점수 (0=필터 미적용)
        max_filter_ratio: 최대 필터 비율 (최소 K*(1-ratio)개 생존)
        margin_scale: σ → 동적 안전 마진 변환 계수
        margin_max: 최대 동적 마진 (m)
    """

    feasibility_weight: float = 50.0
    uncertainty_reduce: str = "sum"
    feasibility_threshold: float = 0.0
    max_filter_ratio: float = 0.5
    margin_scale: float = 1.0
    margin_max: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        assert self.feasibility_weight >= 0, "feasibility_weight must be non-negative"
        assert self.uncertainty_reduce in ("sum", "max", "mean"), \
            f"Unknown uncertainty_reduce: {self.uncertainty_reduce}"
        assert 0 <= self.feasibility_threshold <= 1, \
            "feasibility_threshold must be in [0, 1]"
        assert 0 < self.max_filter_ratio <= 1, \
            "max_filter_ratio must be in (0, 1]"
        assert self.margin_scale >= 0, "margin_scale must be non-negative"
        assert self.margin_max >= 0, "margin_max must be non-negative"


@dataclass
class LatentMPPIParams(MPPIParams):
    """
    Latent-Space MPPI 전용 파라미터

    VAE 잠재 공간에서 K×N 롤아웃 → 디코딩 → 기존 비용 함수 재사용.

    Attributes:
        latent_dim: VAE 잠재 공간 차원
        vae_hidden_dims: VAE 은닉층 차원
        vae_beta: KL 가중치 (작을수록 재구성 우선)
        vae_model_path: 사전 학습 VAE 모델 경로
        decode_interval: 디코딩 간격 (1=매 스텝)
        use_latent_rollout: 잠재 롤아웃 활성화 (False면 Vanilla MPPI 폴백)
    """

    latent_dim: int = 16
    vae_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    vae_beta: float = 0.001
    vae_model_path: Optional[str] = None
    decode_interval: int = 1
    use_latent_rollout: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.decode_interval > 0, "decode_interval must be positive"
        assert self.vae_beta >= 0, "vae_beta must be non-negative"


@dataclass
class DBaSMPPIParams(MPPIParams):
    """
    DBaS-MPPI (Discrete Barrier States MPPI) 전용 파라미터

    Barrier state 증강 + 적응적 탐색 노이즈로 밀집 장애물 환경에서
    가중치 퇴화 없이 안전한 궤적을 생성.

    핵심:
        - Log barrier: B(h) = -log(max(h, h_min))
        - Barrier state dynamics: β(x_{t+1}) = B(h(x_{t+1})) - γ(B(h(x_d)) - β(x_t))
        - 적응적 탐색: σ_eff = σ × (1 + μ·log(e + C_B))

    Attributes:
        dbas_obstacles: 원형 장애물 리스트 [(x, y, radius), ...]
        dbas_walls: 벽 제약 리스트 [('x'|'y', value, direction), ...]
            direction: +1 (val 이상) 또는 -1 (val 이하)
        barrier_weight: RB — barrier 비용 가중치
        barrier_gamma: γ ∈ (0,1) — barrier 상태 수렴률
        exploration_coeff: μ — 적응적 탐색 계수
        h_min: barrier 클리핑 (특이점 방지)
        safety_margin: 추가 안전 마진 (m)
        use_adaptive_exploration: Se = μ·log(e + CB) 활성화
    """

    dbas_obstacles: List[tuple] = field(default_factory=list)
    dbas_walls: List[tuple] = field(default_factory=list)
    barrier_weight: float = 10.0
    barrier_gamma: float = 0.5
    exploration_coeff: float = 1.0
    h_min: float = 1e-6
    safety_margin: float = 0.1
    use_adaptive_exploration: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.barrier_weight >= 0, "barrier_weight must be non-negative"
        assert 0 < self.barrier_gamma < 1, "barrier_gamma must be in (0, 1)"
        assert self.exploration_coeff >= 0, "exploration_coeff must be non-negative"
        assert self.h_min > 0, "h_min must be positive"
        assert self.safety_margin >= 0, "safety_margin must be non-negative"


@dataclass
class RobustMPPIParams(MPPIParams):
    """
    Robust MPPI (R-MPPI) 전용 파라미터

    피드백을 MPPI 샘플링 루프 내부에 통합하여, 외란 하에서도
    추적 가능한 제어를 학습. Tube-MPPI의 분리된 2계층 대신
    MPPI가 외란 보상 능력을 직접 인지.

    핵심:
        x_nom(t+1) = F(x_nom(t), v(t))
        x_real(t+1) = F(x_real(t), v(t) + K·(x_real - x_nom)) + w(t)
        cost_k = Σ_t q(x_real_k(t), ref(t))

    Reference: Gandhi et al., RAL 2021, arXiv:2102.09027

    Attributes:
        disturbance_std: 외란 표준편차 [x, y, θ, ...]
        feedback_gain_scale: AncillaryController 게인 스케일
        disturbance_mode: "gaussian" | "adversarial" | "none"
        robust_alpha: adversarial 모드 상위 α로 최악 선택
        use_feedback: 피드백 포함 여부 (False면 Tube 패턴)
        n_disturbance_samples: 외란 샘플 수 (>1이면 평균)
    """

    disturbance_std: List[float] = field(
        default_factory=lambda: [0.05, 0.05, 0.02]
    )
    feedback_gain_scale: float = 1.0
    disturbance_mode: str = "gaussian"
    robust_alpha: float = 0.8
    use_feedback: bool = True
    n_disturbance_samples: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert all(s >= 0 for s in self.disturbance_std), \
            "disturbance_std must be non-negative"
        assert 0 < self.robust_alpha <= 1, \
            "robust_alpha must be in (0, 1]"
        assert self.disturbance_mode in {"gaussian", "adversarial", "none"}, \
            f"Unknown disturbance_mode: {self.disturbance_mode}"
        assert self.n_disturbance_samples >= 1, \
            "n_disturbance_samples must be >= 1"


@dataclass
class ASRMPPIParams(MPPIParams):
    """
    ASR-MPPI (Adaptive Spectral Risk MPPI) 전용 추가 파라미터

    Spectral Risk Measure (SRM)를 MPPI 가중치에 적용.
    왜곡 함수 φ(q)로 비용 분위수를 비균일 가중하여
    CVaR의 이진 절단을 연속적 곡선으로 일반화.

    SRM_φ(S) = ∫₀¹ VaR_q(S) · φ'(q) dq
    가중치: w_k ∝ φ'(q_k) · exp(-S_{(k)} / λ)

    Attributes:
        distortion_type: 왜곡 함수 종류 ("sigmoid"|"power"|"dual_power"|"cvar")
        distortion_alpha: 중심 파라미터 (sigmoid 전환점, CVaR cutoff)
        distortion_beta: 경사도 (sigmoid sharpness, β→∞ → CVaR)
        distortion_gamma: 지수 (power: q^γ, dual_power: 1-(1-q)^γ)
        use_adaptive_risk: 비용 분포 기반 자동 α/β 조절
        adaptation_rate: 적응 속도 (EMA)
        adaptation_window: 적응 히스토리 윈도우
        min_ess_ratio: ESS 하한 (ESS/K < min_ess_ratio → β 감소)
    """

    distortion_type: str = "sigmoid"
    distortion_alpha: float = 0.5
    distortion_beta: float = 5.0
    distortion_gamma: float = 1.0
    use_adaptive_risk: bool = False
    adaptation_rate: float = 0.1
    adaptation_window: int = 50
    min_ess_ratio: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        assert self.distortion_type in {"sigmoid", "power", "dual_power", "cvar"}, \
            f"Unknown distortion_type: {self.distortion_type}"
        assert 0 < self.distortion_alpha < 1, \
            "distortion_alpha must be in (0, 1)"
        assert self.distortion_beta > 0, \
            "distortion_beta must be positive"
        assert self.distortion_gamma > 0, \
            "distortion_gamma must be positive"
        assert 0 < self.adaptation_rate <= 1, \
            "adaptation_rate must be in (0, 1]"
        assert self.adaptation_window >= 1, \
            "adaptation_window must be >= 1"
        assert 0 < self.min_ess_ratio <= 1, \
            "min_ess_ratio must be in (0, 1]"


@dataclass
class SGMPPIParams(MPPIParams):
    """
    SG-MPPI (Score-Guided MPPI) 전용 파라미터

    Denoising Score Matching으로 비용 지형의 score function을 학습하고,
    MPPI 가우시안 노이즈에 score 방향 bias를 추가하여 저비용 영역으로 유도.

    핵심 수식:
        s_θ(U, σ, state) ≈ ∇_U log p(U|state)
        ε_guided = ε + α · σ² · s_θ(U + ε, σ, state)

    Attributes:
        score_hidden_dims: Score network 은닉층 차원
        n_sigma_levels: DSM 노이즈 스케일 개수
        sigma_min: 최소 노이즈 스케일
        sigma_max: 최대 노이즈 스케일
        guidance_scale: α — score bias 강도
        guidance_decay: 다중 반복 시 α 감쇠율
        n_guide_iters: score-guided 반복 횟수 (1=단일)
        use_annealing: DIAL-style σ 어닐링 결합
        score_online_training: 온라인 학습 활성화
        score_training_interval: 학습 주기 (스텝)
        score_min_samples: 최소 학습 샘플
        score_buffer_size: 데이터 버퍼 크기
        score_model_path: 사전 학습 모델 경로
    """

    # Score network
    score_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    n_sigma_levels: int = 10
    sigma_min: float = 0.01
    sigma_max: float = 1.0

    # Guidance
    guidance_scale: float = 0.5
    guidance_decay: float = 0.95

    # Multi-iteration (DIAL 결합, 선택)
    n_guide_iters: int = 1
    use_annealing: bool = False

    # Online learning
    score_online_training: bool = False
    score_training_interval: int = 20
    score_min_samples: int = 50
    score_buffer_size: int = 2000
    score_model_path: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.n_sigma_levels >= 1, "n_sigma_levels must be >= 1"
        assert 0 < self.sigma_min < self.sigma_max, \
            "sigma_min must be in (0, sigma_max)"
        assert self.guidance_scale >= 0, "guidance_scale must be non-negative"
        assert 0 < self.guidance_decay <= 1, \
            "guidance_decay must be in (0, 1]"
        assert self.n_guide_iters >= 1, "n_guide_iters must be >= 1"
        assert self.score_training_interval >= 1, \
            "score_training_interval must be >= 1"
        assert self.score_min_samples >= 1, \
            "score_min_samples must be >= 1"
        assert self.score_buffer_size >= self.score_min_samples, \
            "score_buffer_size must be >= score_min_samples"


@dataclass
class LPMPPIParams(MPPIParams):
    """
    LP-MPPI (Low-Pass MPPI) 전용 파라미터

    Butterworth 저역통과 필터를 MPPI 노이즈에 적용하여
    주파수 영역에서 직접적인 smoothness 제어.

    |H(f)|² = 1 / (1 + (f/f_c)^(2n))

    Reference: Kicki et al., ICRA 2026, arXiv:2503.11717

    Attributes:
        cutoff_freq: Butterworth 차단 주파수 (Hz)
        filter_order: Butterworth 필터 차수 (rolloff = -20n dB/decade)
        normalize_variance: 필터링 후 분산 정규화 여부
    """

    cutoff_freq: float = 3.0
    filter_order: int = 3
    normalize_variance: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.cutoff_freq > 0, "cutoff_freq must be positive"
        nyquist = 1.0 / (2.0 * self.dt)
        assert self.cutoff_freq < nyquist, \
            f"cutoff_freq ({self.cutoff_freq}) must be < Nyquist ({nyquist})"
        assert 1 <= self.filter_order <= 10, \
            "filter_order must be in [1, 10]"


@dataclass
class ResidualMPPIParams(MPPIParams):
    """
    Residual-MPPI 전용 파라미터

    사전 정책(base policy)의 출력을 명목 시퀀스로 사용하고,
    MPPI는 잔차(residual) δu만 최적화. 증강 비용으로 정책 근처에서 탐색.

    U_nominal = π(state, ref)
    U = U_nominal + Σ ω_k · ε_k
    C_aug = C(τ) + ω' · ||U - U_nominal||²

    Reference: Wang et al., ICLR 2025, arXiv:2407.00898

    Attributes:
        policy_weight: ω' — 사전 정책 log-likelihood 가중치
        use_policy_nominal: 정책 출력을 명목 시퀀스로 사용
        residual_scale: 잔차 노이즈 스케일
        policy_type: 기본 정책 유형 ("feedback" | "zero" | "custom")
        policy_update_interval: 정책 업데이트 주기
        use_augmented_cost: 증강 비용 (정책 log-likelihood) 사용
        kl_weight: KL 발산 가중치 (U가 정책에서 벗어나는 페널티)
    """

    policy_weight: float = 1.0
    use_policy_nominal: bool = True
    residual_scale: float = 1.0
    policy_type: str = "feedback"
    policy_update_interval: int = 1
    use_augmented_cost: bool = True
    kl_weight: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        assert self.policy_weight >= 0, \
            "policy_weight must be non-negative"
        assert self.residual_scale > 0, \
            "residual_scale must be positive"
        assert self.kl_weight >= 0, \
            "kl_weight must be non-negative"
        assert self.policy_type in {"feedback", "zero", "custom"}, \
            f"Unknown policy_type: {self.policy_type}"
        assert self.policy_update_interval >= 1, \
            "policy_update_interval must be >= 1"


@dataclass
class BiasedMPPIParams(MPPIParams):
    """
    Biased-MPPI (Mixture Sampling MPPI) 전용 파라미터

    J개 보조 정책 제안 + (K-J)개 가우시안 샘플을 혼합하여 샘플링.
    Importance weight에서 샘플링 분포 q_s가 소거되어
    가중치 = softmax(-S/λ)로 동일 (Biased-MPPI 핵심 정리).

    Reference: Trevisan & Alonso-Mora, RA-L 2024, arXiv:2401.09241

    Attributes:
        ancillary_types: 보조 정책 이름 리스트
        samples_per_policy: 정책당 샘플 수
        policy_noise_scale: 정책 제안에 추가할 노이즈 비율 (0~1)
        use_adaptive_lambda: ESS 기반 λ 적응
        ess_min_ratio: ESS 하한 비율 (ESS/K < min → λ 증가)
        ess_max_ratio: ESS 상한 비율 (ESS/K > max → λ 감소)
        lambda_increase_rate: λ 증가율
        lambda_decrease_rate: λ 감소율
        lambda_min: 최소 λ
        lambda_max: 최대 λ
        use_reward_normalization: DIAL-style 보상 정규화
    """

    ancillary_types: List[str] = field(
        default_factory=lambda: ["pure_pursuit", "braking"]
    )
    samples_per_policy: int = 10
    policy_noise_scale: float = 0.3
    use_adaptive_lambda: bool = True
    ess_min_ratio: float = 0.1
    ess_max_ratio: float = 0.5
    lambda_increase_rate: float = 1.2
    lambda_decrease_rate: float = 0.9
    lambda_min: float = 0.1
    lambda_max: float = 100.0
    use_reward_normalization: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert len(self.ancillary_types) > 0, \
            "ancillary_types must not be empty"
        assert self.samples_per_policy >= 1, \
            "samples_per_policy must be >= 1"
        assert 0 <= self.policy_noise_scale <= 1, \
            "policy_noise_scale must be in [0, 1]"
        assert self.ess_min_ratio < self.ess_max_ratio, \
            "ess_min_ratio must be < ess_max_ratio"
        total_policy_samples = len(self.ancillary_types) * self.samples_per_policy
        assert total_policy_samples < self.K, \
            f"total policy samples ({total_policy_samples}) must be < K ({self.K})"
        assert self.lambda_min > 0, "lambda_min must be positive"
        assert self.lambda_max > self.lambda_min, \
            "lambda_max must be > lambda_min"
        assert self.lambda_increase_rate > 1.0, \
            "lambda_increase_rate must be > 1.0"
        assert 0 < self.lambda_decrease_rate < 1.0, \
            "lambda_decrease_rate must be in (0, 1.0)"


@dataclass
class GNMPPIParams(MPPIParams):
    """
    GN-MPPI (Gauss-Newton MPPI) 전용 파라미터

    가우스-뉴턴 2차 업데이트로 MPPI 수렴 가속.
    가우시안 스무딩으로 야코비안 복원 + GGN 스텝.

    핵심 수식:
        ∇J ≈ E[C(U+ε) * ε^T] * Σ^{-1}        (가우시안 스무딩 기울기)
        H_GGN ≈ J^T * J + λI                     (GGN 헤시안 근사)
        U_{k+1} = U_k - α * H_GGN^{-1} * ∇J     (뉴턴 스텝)

    Reference: Homburger et al., arXiv:2512.04579

    Attributes:
        n_gn_iters: GN 반복 횟수
        n_gn_iters_init: 첫 호출 GN 반복 횟수 (cold start)
        gn_step_size: GN 스텝 크기 (line search 초기값)
        line_search_steps: 병렬 라인 서치 후보 수
        line_search_decay: 라인 서치 감쇠율
        use_gn_update: True=GN 업데이트, False=표준 MPPI 폴백
        regularization: GGN 헤시안 정규화 (특이성 방지)
        use_reward_normalization: 보상 정규화 활성화
    """

    n_gn_iters: int = 3
    n_gn_iters_init: int = 5
    gn_step_size: float = 1.0
    line_search_steps: int = 5
    line_search_decay: float = 0.5
    use_gn_update: bool = True
    regularization: float = 1e-4
    use_reward_normalization: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.n_gn_iters > 0, "n_gn_iters must be positive"
        assert self.n_gn_iters_init > 0, "n_gn_iters_init must be positive"
        assert self.gn_step_size > 0, "gn_step_size must be positive"
        assert self.line_search_steps >= 1, "line_search_steps must be >= 1"
        assert 0 < self.line_search_decay < 1, \
            "line_search_decay must be in (0, 1)"
        assert self.regularization >= 0, "regularization must be non-negative"


@dataclass
class TDMPPIParams(MPPIParams):
    """
    TD-MPPI (Temporal-Difference MPPI) 전용 파라미터

    TD 학습 terminal value function V(x_T)로 짧은 롤아웃에서도
    무한 수평선 추론. 제약 위반 시 할인율 동적 감소.

    핵심 수식:
        C_total(τ) = Σ_t c(x_t, u_t) + w_V · V(x_T)
        V(x) ← V(x) + α[c + γV(x') - V(x)]    (TD(0))

    Reference: Crestaz et al., RA-L 2026, hal-05213269

    Attributes:
        value_hidden_dims: Value network 은닉층 차원
        td_learning_rate: TD 학습률
        td_gamma: 할인율 γ ∈ (0, 1]
        td_buffer_size: 경험 버퍼 최대 크기
        td_batch_size: 미니배치 크기
        td_update_interval: TD 업데이트 주기 (스텝)
        td_min_samples: 최소 학습 샘플 수
        use_terminal_value: terminal value V(x_T) 사용 여부
        value_weight: V(x_T) 가중치 w_V
        use_constraint_discount: 제약 할인 활성화
        constraint_penalty: 제약 위반 페널티
        discount_decay: 제약 위반 시 할인 감소율
    """

    # Value function
    value_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    td_learning_rate: float = 0.001
    td_gamma: float = 0.99
    td_buffer_size: int = 5000
    td_batch_size: int = 64
    td_update_interval: int = 5
    td_min_samples: int = 100
    use_terminal_value: bool = True
    value_weight: float = 1.0

    # Constraint discounting
    use_constraint_discount: bool = False
    constraint_penalty: float = 10.0
    discount_decay: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.td_gamma <= 1, \
            "td_gamma must be in (0, 1]"
        assert self.td_learning_rate > 0, \
            "td_learning_rate must be positive"
        assert self.td_buffer_size > 0, \
            "td_buffer_size must be positive"
        assert self.td_batch_size > 0, \
            "td_batch_size must be positive"
        assert self.td_update_interval >= 1, \
            "td_update_interval must be >= 1"
        assert self.td_min_samples >= 1, \
            "td_min_samples must be >= 1"
        assert self.value_weight >= 0, \
            "value_weight must be non-negative"
        assert self.constraint_penalty >= 0, \
            "constraint_penalty must be non-negative"
        assert 0 < self.discount_decay <= 1, \
            "discount_decay must be in (0, 1]"


@dataclass
class ProjectionMPPIParams(MPPIParams):
    """
    pi-MPPI (Projection-based MPPI) 전용 파라미터

    QP Projection으로 제어 입력의 크기/jerk/snap에 대한 하드 제약을 보장.
    후처리 스무딩이 아닌 사전적(a priori) 매끄러움 보장.

    제약 체계:
        - |v_t| ≤ u_max (크기 제약, 기본 MPPI에서 상속)
        - |v_t - v_{t-1}| / dt ≤ jerk_limit (변화율 제약)
        - |(v_t - 2v_{t-1} + v_{t-2})| / dt² ≤ snap_limit (2차 도함수 제약)

    투영 방법:
        - "clip": 순차 클리핑 (빠름, O(K*N*nu))
        - "qp": scipy.optimize.minimize SLSQP (정확, 느림)

    Reference: Andrejev et al., RA-L 2025, arXiv:2504.10962

    Attributes:
        jerk_limit: 최대 jerk (|Δu/dt|)
        snap_limit: 최대 snap (|Δ²u/dt²|)
        use_jerk_constraint: jerk 제약 활성화
        use_snap_constraint: snap 제약 활성화
        projection_method: 투영 방법 ("clip" or "qp")
        project_samples: 샘플 투영 여부
        project_output: 최종 출력 투영 여부
    """

    jerk_limit: float = 5.0
    snap_limit: float = 50.0
    use_jerk_constraint: bool = True
    use_snap_constraint: bool = False
    projection_method: str = "clip"
    project_samples: bool = True
    project_output: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.jerk_limit > 0, \
            "jerk_limit must be positive"
        assert self.snap_limit > 0, \
            "snap_limit must be positive"
        assert self.projection_method in {"clip", "qp"}, \
            f"Unknown projection_method: {self.projection_method}"
        assert self.use_jerk_constraint or self.use_snap_constraint \
            or not (self.project_samples or self.project_output), \
            "At least one constraint must be enabled when projection is active"


@dataclass
class DeterministicMPPIParams(MPPIParams):
    """
    dsMPPI (Deterministic Sampling MPPI) 전용 파라미터

    랜덤 샘플링 대신 결정론적 샘플(Halton/Sobol/Sigma Points/Grid) 사용.
    CEM 반복 최적화와 결합하여 적은 샘플로도 효율적이고 매끄러운 제어.

    핵심 수식:
        1. 결정론적 샘플 u_k = μ + Φ^{-1}(q_k) · σ, q_k ∈ QMC sequence
        2. CEM 반복: μ_{i+1} = (1-α)μ_i + α·mean(elite), σ 동일
        3. 최종 MPPI 가중: u* = Σ w_k · u_k, w_k = softmax(-cost/λ)

    Reference: Walker et al., arXiv:2601.03893, 2026

    Attributes:
        sampling_method: 결정론적 샘플링 방법 ("halton", "sobol", "sigma_points", "grid")
        n_cem_iters: CEM 반복 횟수
        n_cem_iters_init: 첫 호출 CEM 반복 (cold start)
        elite_ratio: elite 비율 (상위 비율만 분포 업데이트에 사용)
        cem_alpha: 분포 업데이트 EMA 계수 (0=유지, 1=완전 교체)
        use_cem_update: CEM 분포 업데이트 활성화
        add_random_samples: 추가 랜덤 샘플 수 (하이브리드 모드, 0=순수 결정론적)
    """

    sampling_method: str = "halton"
    n_cem_iters: int = 3
    n_cem_iters_init: int = 5
    elite_ratio: float = 0.3
    cem_alpha: float = 0.7
    use_cem_update: bool = True
    add_random_samples: int = 0

    def __post_init__(self):
        super().__post_init__()
        valid_methods = ("halton", "sobol", "sigma_points", "grid")
        assert self.sampling_method in valid_methods, \
            f"sampling_method must be one of {valid_methods}, got '{self.sampling_method}'"
        assert self.n_cem_iters > 0, "n_cem_iters must be positive"
        assert self.n_cem_iters_init > 0, "n_cem_iters_init must be positive"
        assert 0 < self.elite_ratio <= 1, "elite_ratio must be in (0, 1]"
        assert 0 < self.cem_alpha <= 1, "cem_alpha must be in (0, 1]"
        assert self.add_random_samples >= 0, "add_random_samples must be non-negative"


@dataclass
class DRPAMPPIParams(MPPIParams):
    """
    DRPA-MPPI (Dynamic Repulsive Potential Augmented MPPI) 전용 파라미터

    Local minima trap 동적 감지 + 반발 포텐셜 자동 추가로
    글로벌 경로 탐색 없이 반응적 탈출.

    핵심 수식:
        F_rep(x) = η · (1/d(x,o) - 1/d_0)² if d < d_0, else 0
        C_total = C_normal + α · F_rep  (탈출 모드 시)

    Reference: Fuke et al., arXiv:2503.20134, 2025

    Attributes:
        obstacles: 장애물 목록 [(x, y, radius), ...]
        repulsive_gain: η — 반발 포텐셜 강도
        influence_distance: d_0 — 반발 영향 범위
        stagnation_threshold: 진행 정체 감지 임계값 (이동량)
        stagnation_window: 정체 감지 윈도우 (스텝 수)
        escape_boost: 탈출 모드 노이즈 증폭 계수
        recovery_threshold: 탈출 성공 판정 임계값 (이동량)
        use_noise_boost: 탈출 모드 시 노이즈 증폭 활성화
    """

    obstacles: List[tuple] = field(default_factory=list)
    repulsive_gain: float = 5.0
    influence_distance: float = 1.0
    stagnation_threshold: float = 0.1
    stagnation_window: int = 10
    escape_boost: float = 2.0
    recovery_threshold: float = 0.3
    use_noise_boost: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.repulsive_gain >= 0, \
            "repulsive_gain must be non-negative"
        assert self.influence_distance > 0, \
            "influence_distance must be positive"
        assert self.stagnation_threshold > 0, \
            "stagnation_threshold must be positive"
        assert self.stagnation_window >= 2, \
            "stagnation_window must be >= 2"
        assert self.escape_boost >= 1.0, \
            "escape_boost must be >= 1.0"
        assert self.recovery_threshold > 0, \
            "recovery_threshold must be positive"
        # 장애물 형식 검증
        for obs in self.obstacles:
            assert len(obs) == 3, \
                f"Each obstacle must be (x, y, radius), got {obs}"
            assert obs[2] > 0, \
                f"Obstacle radius must be positive, got {obs[2]}"


@dataclass
class CSCMPPIParams(MPPIParams):
    """
    CSC-MPPI (Constrained Sampling Cluster MPPI) 전용 파라미터

    Primal-dual 투영 + DBSCAN 클러스터링으로 실행 가능한 최적 궤적 선택.
    가중 평균 대신 클러스터 대표를 선택하여 실행 가능성 보장.

    Reference: arXiv:2506.16386, 2025

    Attributes:
        obstacles: 원형 장애물 리스트 [(x, y, radius), ...]
        safety_margin: 장애물 안전 마진 (m)
        n_projection_steps: primal-dual 반복 수
        projection_lr: primal 스텝 크기 (제어 업데이트율)
        dual_lr: dual 스텝 크기 (라그랑주 승수 업데이트율)
        dbscan_eps: DBSCAN 이웃 거리
        dbscan_min_samples: DBSCAN 최소 클러스터 크기
        use_projection: 제약 투영 활성화
        use_clustering: 클러스터링 활성화
        fallback_to_mppi: 클러스터 없을 때 표준 MPPI 폴백
    """

    obstacles: List[tuple] = field(default_factory=list)
    safety_margin: float = 0.2
    n_projection_steps: int = 5
    projection_lr: float = 0.1
    dual_lr: float = 0.01
    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 3
    use_projection: bool = True
    use_clustering: bool = True
    fallback_to_mppi: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.safety_margin >= 0, \
            "safety_margin must be non-negative"
        assert self.n_projection_steps >= 1, \
            "n_projection_steps must be >= 1"
        assert self.projection_lr > 0, \
            "projection_lr must be positive"
        assert self.dual_lr > 0, \
            "dual_lr must be positive"
        assert self.dbscan_eps > 0, \
            "dbscan_eps must be positive"
        assert self.dbscan_min_samples >= 1, \
            "dbscan_min_samples must be >= 1"


@dataclass
class TransformerMPPIParams(MPPIParams):
    """
    T-MPPI (Transformer-based MPPI) 전용 파라미터

    학습된 Transformer로 MPPI 초기 제어 시퀀스를 예측하여
    샘플 효율성 향상. 과거 상태/제어 이력을 컨텍스트로 사용.

    핵심 수식:
        U_init = Transformer(state_history, control_history)
        U* = MPPI(U_init, K_reduced)
        Loss = MSE(U_pred, U_optimal)

    Reference: Zinage et al., arXiv:2412.17118, Dec 2024

    Attributes:
        transformer_hidden_dim: Transformer 은닉 차원
        transformer_n_heads: 멀티헤드 어텐션 헤드 수
        transformer_n_layers: Transformer 레이어 수
        transformer_context_length: 과거 이력 길이
        transformer_dropout: 드롭아웃 비율

        transformer_lr: 학습률
        transformer_buffer_size: 데이터 버퍼 최대 크기
        transformer_min_samples: 학습 시작 최소 샘플 수
        transformer_batch_size: 미니배치 크기
        transformer_training_interval: 학습 주기 (스텝)
        transformer_n_train_steps: 학습 주기당 그래디언트 스텝 수

        use_transformer_init: Transformer 초기화 활성화 (False면 Vanilla MPPI 폴백)
        transformer_model_path: 사전 학습 모델 경로
        blend_ratio: Transformer 예측과 이전 솔루션 혼합 비율 (1.0=전부 Transformer)
        online_learning: 온라인 학습 활성화
    """

    # Transformer architecture
    transformer_hidden_dim: int = 128
    transformer_n_heads: int = 4
    transformer_n_layers: int = 2
    transformer_context_length: int = 20
    transformer_dropout: float = 0.1

    # Training
    transformer_lr: float = 1e-3
    transformer_buffer_size: int = 5000
    transformer_min_samples: int = 100
    transformer_batch_size: int = 32
    transformer_training_interval: int = 10
    transformer_n_train_steps: int = 5

    # Mode
    use_transformer_init: bool = True
    transformer_model_path: Optional[str] = None
    blend_ratio: float = 0.7
    online_learning: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.transformer_hidden_dim > 0, \
            "transformer_hidden_dim must be positive"
        assert self.transformer_n_heads > 0, \
            "transformer_n_heads must be positive"
        assert self.transformer_hidden_dim % self.transformer_n_heads == 0, \
            "transformer_hidden_dim must be divisible by transformer_n_heads"
        assert self.transformer_n_layers > 0, \
            "transformer_n_layers must be positive"
        assert self.transformer_context_length >= 1, \
            "transformer_context_length must be >= 1"
        assert 0 <= self.transformer_dropout < 1, \
            "transformer_dropout must be in [0, 1)"
        assert self.transformer_lr > 0, \
            "transformer_lr must be positive"
        assert self.transformer_buffer_size > 0, \
            "transformer_buffer_size must be positive"
        assert self.transformer_min_samples >= 1, \
            "transformer_min_samples must be >= 1"
        assert self.transformer_buffer_size >= self.transformer_min_samples, \
            "transformer_buffer_size must be >= transformer_min_samples"
        assert self.transformer_batch_size > 0, \
            "transformer_batch_size must be positive"
        assert self.transformer_training_interval >= 1, \
            "transformer_training_interval must be >= 1"
        assert self.transformer_n_train_steps >= 1, \
            "transformer_n_train_steps must be >= 1"
        assert 0 <= self.blend_ratio <= 1, \
            "blend_ratio must be in [0, 1]"


@dataclass
class FeedbackMPPIParams(MPPIParams):
    """
    Feedback-MPPI (F-MPPI) 전용 파라미터

    Riccati 기반 피드백 게인으로 MPPI 해를 재사용하여
    전체 재최적화 없이 고주파 폐루프 보정.

    핵심 수식:
        A_t = df/dx|_{x*,u*}, B_t = df/du|_{x*,u*}  (유한 차분 야코비안)
        P_t: backward Riccati recursion → K_t 피드백 게인
        u = u*[t] + K_t (x_actual - x*_t)            (피드백 보정)

    Reference: Belvedere et al., IEEE RA-L 2026, arXiv:2506.14855

    Attributes:
        reuse_steps: MPPI 해를 피드백으로 재사용하는 스텝 수
        jacobian_eps: 유한 차분 야코비안 epsilon
        feedback_weight_Q: Riccati Q 스케일 (상태 비용)
        feedback_weight_R: Riccati R 스케일 (제어 비용)
        use_feedback: False = vanilla MPPI (매 스텝 재최적화)
        feedback_gain_clip: 게인 클리핑 (안정성)
        use_warm_start: 피드백 단계에서 U warm start
    """

    reuse_steps: int = 3
    jacobian_eps: float = 1e-4
    feedback_weight_Q: float = 10.0
    feedback_weight_R: float = 0.1
    use_feedback: bool = True
    feedback_gain_clip: float = 10.0
    use_warm_start: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.reuse_steps >= 1, \
            "reuse_steps must be >= 1"
        assert self.jacobian_eps > 0, \
            "jacobian_eps must be positive"
        assert self.feedback_weight_Q > 0, \
            "feedback_weight_Q must be positive"
        assert self.feedback_weight_R > 0, \
            "feedback_weight_R must be positive"
        assert self.feedback_gain_clip > 0, \
            "feedback_gain_clip must be positive"


@dataclass
class ContingencyMPPIParams(MPPIParams):
    """
    C-MPPI (Contingency-Constrained MPPI) 전용 파라미터

    Nested MPPI 구조: 외부 MPPI가 명목 궤적을 최적화하면서,
    체크포인트 상태에서 내부 MPPI(contingency)가 비상 궤적의 비용을 평가.
    모든 계획 상태에서 안전 집합 도달 가능성을 보장.

    핵심 수식:
        min_u J_nom(u) + λ_cont * max_t contingency_cost(x_t)
        contingency_cost(x_t) = min_v cost_safe(rollout(x_t, v))

    Reference: Jung, Estornell & Everett, L4DC 2025, arXiv:2412.09777

    Attributes:
        contingency_weight: λ_cont — contingency 비용 가중치
        contingency_horizon: N_cont — 내부 MPPI 호라이즌
        contingency_samples: K_cont — 내부 MPPI 샘플 수
        contingency_lambda: 내부 MPPI 온도 파라미터
        n_checkpoints: 명목 궤적 상 contingency 평가 지점 수
        safe_cost_threshold: contingency_cost 임계값 (초과 시 큰 페널티)
        safety_cost_weight: 안전 집합 미도달 시 페널티
        use_braking_contingency: 제로 제어 contingency 사용
        use_mppi_contingency: 내부 MPPI contingency 사용
        contingency_sigma_scale: 내부 MPPI 노이즈 스케일
    """

    contingency_weight: float = 100.0
    contingency_horizon: int = 10
    contingency_samples: int = 32
    contingency_lambda: float = 1.0
    n_checkpoints: int = 3
    safe_cost_threshold: float = 10.0
    safety_cost_weight: float = 500.0
    use_braking_contingency: bool = True
    use_mppi_contingency: bool = True
    contingency_sigma_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.contingency_weight >= 0, \
            "contingency_weight must be non-negative"
        assert self.contingency_horizon >= 1, \
            "contingency_horizon must be >= 1"
        assert self.contingency_samples >= 1, \
            "contingency_samples must be >= 1"
        assert self.contingency_lambda > 0, \
            "contingency_lambda must be positive"
        assert self.n_checkpoints >= 1, \
            "n_checkpoints must be >= 1"
        assert self.safe_cost_threshold > 0, \
            "safe_cost_threshold must be positive"
        assert self.safety_cost_weight >= 0, \
            "safety_cost_weight must be non-negative"
        assert self.contingency_sigma_scale > 0, \
            "contingency_sigma_scale must be positive"
        assert self.use_braking_contingency or self.use_mppi_contingency, \
            "At least one contingency mode must be enabled"


@dataclass
class DualGuardMPPIParams(MPPIParams):
    """
    DualGuard-MPPI 전용 파라미터

    Hamilton-Jacobi 도달 가능성 분석 영감의 안전 가치 함수를 MPPI에 통합.
    사전 계산된 V(x)로 궤적 안전성 평가 + 안전하지 않은 샘플에 페널티/투영/필터.
    Nominal + sample 이중 안전 보호 (DualGuard).

    핵심 수식:
        V(x) = min_i (||pos - o_i|| - (r_i + margin))   (signed distance)
        V(x) >= 0  → safe
        V(x) < 0   → unsafe

        Soft:   cost_k += penalty * exp(-decay * V(x))   (V < threshold)
        Hard:   u_k = u_k + α * ∇V(x)                   (gradient projection)
        Filter: w_k = 0  if any V(x_t) < 0

    Reference: Borquez et al., IEEE RA-L 2025, arXiv:2502.01924

    Attributes:
        obstacles: 장애물 리스트 [(x, y, radius), ...]
        safety_margin: 추가 안전 마진 (m)
        safety_mode: 안전 모드 ("soft" | "hard" | "filter")
        safety_penalty: V(x) < 0 시 페널티 강도
        safety_decay: 소프트 배리어 지수 감쇠율
        use_velocity_penalty: 장애물 방향 이동 페널티 활성화
        velocity_penalty_weight: 속도 페널티 가중치
        ttc_horizon: time-to-collision 호라이즌 (초)
        use_nominal_guard: 명목 궤적 안전 가드 활성화
        use_sample_guard: 전체 샘플 안전 가드 활성화
        min_safe_fraction: 최소 안전 샘플 비율 (미달 시 노이즈 증폭)
        noise_boost_factor: 안전 샘플 부족 시 노이즈 배율
    """

    obstacles: List[tuple] = field(default_factory=list)
    safety_margin: float = 0.2
    safety_mode: str = "soft"
    safety_penalty: float = 1000.0
    safety_decay: float = 5.0
    use_velocity_penalty: bool = True
    velocity_penalty_weight: float = 50.0
    ttc_horizon: float = 1.0
    use_nominal_guard: bool = True
    use_sample_guard: bool = True
    min_safe_fraction: float = 0.1
    noise_boost_factor: float = 1.5

    def __post_init__(self):
        super().__post_init__()
        assert self.safety_margin >= 0, \
            "safety_margin must be non-negative"
        assert self.safety_mode in {"soft", "hard", "filter"}, \
            f"Unknown safety_mode: {self.safety_mode}"
        assert self.safety_penalty >= 0, \
            "safety_penalty must be non-negative"
        assert self.safety_decay >= 0, \
            "safety_decay must be non-negative"
        assert self.velocity_penalty_weight >= 0, \
            "velocity_penalty_weight must be non-negative"
        assert self.ttc_horizon > 0, \
            "ttc_horizon must be positive"
        assert 0 < self.min_safe_fraction <= 1, \
            "min_safe_fraction must be in (0, 1]"
        assert self.noise_boost_factor >= 1.0, \
            "noise_boost_factor must be >= 1.0"


@dataclass
class ParameterRobustMPPIParams(MPPIParams):
    """
    PR-MPPI (Parameter-Robust MPPI) 전용 파라미터

    미지 파라미터에 대한 입자 기반 belief를 유지하고,
    다수의 파라미터 가설 하에서 MPPI 비용을 평가하여
    파라미터 불확실성에 robust한 제어 수행.
    온라인으로 파라미터를 학습하면서 점진적으로 성능 향상.

    핵심 수식:
        θ_particles = [θ_1, ..., θ_M]
        w_θ_i = likelihood(observation | θ_i)
        cost_k = Σ_i w_θ_i * cost(rollout(state, u_k, model(θ_i)))  (weighted_mean)
                 OR max_i cost(...)                                   (worst_case)

    Reference: Vahs et al., 2026, arXiv:2601.02948

    Attributes:
        n_particles: 파라미터 가설 (입자) 수
        param_name: 불확실 파라미터 이름 (모델 attribute)
        param_nominal: 명목 파라미터 값
        param_std: 초기 불확실성 표준편차
        param_min: 파라미터 하한
        param_max: 파라미터 상한
        aggregation_mode: 비용 집계 방법 ("weighted_mean"|"worst_case"|"cvar")
        cvar_alpha: CVaR 모드 상위 비율
        online_learning: 온라인 파라미터 학습 활성화
        learning_rate: 온라인 학습률
        observation_window: 관측 히스토리 윈도우 크기
        min_observations: 업데이트 시작 최소 관측 수
        use_resampling: 저가중치 입자 재샘플링 활성화
        resample_threshold: ESS/M 재샘플링 임계값
    """

    # Parameter particles
    n_particles: int = 5
    param_name: str = "wheelbase"
    param_nominal: float = 0.5
    param_std: float = 0.1
    param_min: float = 0.2
    param_max: float = 1.0

    # Cost aggregation across particles
    aggregation_mode: str = "weighted_mean"
    cvar_alpha: float = 0.3

    # Online learning
    online_learning: bool = True
    learning_rate: float = 0.01
    observation_window: int = 5
    min_observations: int = 3
    use_resampling: bool = True
    resample_threshold: float = 0.3

    def __post_init__(self):
        super().__post_init__()
        assert self.n_particles >= 1, \
            "n_particles must be >= 1"
        assert self.param_std > 0, \
            "param_std must be positive"
        assert self.param_min < self.param_max, \
            "param_min must be < param_max"
        assert self.param_min <= self.param_nominal <= self.param_max, \
            "param_nominal must be in [param_min, param_max]"
        assert self.aggregation_mode in {"weighted_mean", "worst_case", "cvar"}, \
            f"Unknown aggregation_mode: {self.aggregation_mode}"
        assert 0 < self.cvar_alpha <= 1, \
            "cvar_alpha must be in (0, 1]"
        assert self.learning_rate > 0, \
            "learning_rate must be positive"
        assert self.observation_window >= 1, \
            "observation_window must be >= 1"
        assert self.min_observations >= 1, \
            "min_observations must be >= 1"
        assert 0 < self.resample_threshold <= 1, \
            "resample_threshold must be in (0, 1]"
