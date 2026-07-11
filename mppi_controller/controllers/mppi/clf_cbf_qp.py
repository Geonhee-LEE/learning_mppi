"""
CLF-CBF-QP Controller (cbfkit-inspired, pure numpy/scipy)

cbfkit의 cbf_clf_qp_generator (JAX)를 numpy/scipy로 포팅한 네이티브 구현.

QP 정식화 (cbfkit vanilla_cbf_clf_qp_control_laws 참조):

    min_{u, δ}   ||u - u_nom||²_P + λ_clf · δ²
    s.t.         A_cbf · u ≥ b_cbf          (zeroing CBF, hard, 장애물별)
                 a_clf · u ≤ b_clf + δ      (vanilla CLF, soft / relaxable_clf)
                 u_min ≤ u ≤ u_max
                 δ ≥ 0

cbfkit 대응 관계:
    - zeroing CBF 제약: Lf·h + Lg·h·u + α(h) ≥ 0  →  A_cbf·u ≥ b_cbf
      (A_cbf = Lg·h, b_cbf = -Lf·h - α·h)
    - vanilla CLF 제약: Lf·V + Lg·V·u ≤ -γ(V) + δ  →  a_clf·u ≤ b_clf + δ
    - relaxable_clf=True (slack penalty), relaxable_cbf=False (hard)
    - p_mat = diag([P, λ_clf])  (slack penalty를 P 행렬에 직접 포함)

솔버 전략 (jaxopt/OSQP 대신):
    1. 해석적 fast path
       - 제약 비활성: u* = clip(u_nom)
       - CLF만 활성 (soft): 닫힌형 tradeoff
             u* = u_nom - (λ·s / (1 + λ·aᵀP⁻¹a)) · P⁻¹a,   s = a·u_nom - b_clf
       - 단일 CBF 활성: 등식 투영 (closed-form projection)
             u* = u_nom + P⁻¹gᵀ · (c - g·u_nom) / (g·P⁻¹·gᵀ)
    2. 다중 제약/경계 활성 시 scipy.optimize.minimize(SLSQP) 폴백

컨트롤러 (repo 표준 인터페이스 compute_control(state, ref) -> (u, info)):
    - CLFCBFQPController: unicycle CLF + look-ahead(near-identity) 맵 기반
      * 기구학 (3D [x,y,θ], u=[v,ω]): p̃ = p + d·[cosθ, sinθ] → ṗ̃ = M(θ)·u 가역
      * 동역학 (5D [x,y,θ,v,ω], u=[a,α]): HOCBF 1단 캐스케이드 h_e = ḣ + λ·h
        + 속도 오차 CLF (backstepping-lite)
    - CBFOnlyQPController: CLF 없이 pure-pursuit 명목 제어 + CBF-QP 필터
      (safe_control의 CBF-QP 베이스라인과 직접 비교용)

근사 사항 (문서화):
    - 기구학 CLF의 heading 항: ė_θ ≈ -ω (LOS 회전 및 레퍼런스 이동 무시)
    - 동역학 CLF: 원하는 twist w_d의 시간 미분 무시 (backstepping-lite)
    - CBF는 연속시간 조건 → 이산 적분 오차는 safety_margin으로 흡수
    - 명목 제어는 장애물 인지 접선 투영(deadlock 방지)을 포함: CBF 제약을
      위반하는 반경 방향 성분을 제거하고 접선 성분을 유지 (안전 보장은
      여전히 QP의 hard CBF 제약이 담당; 투영은 head-on 정체 회피용)
    - 수렴 지점은 레퍼런스에서 lookahead_d 이내 (near-identity 오프셋)

Ref: cbfkit (Toyota Research Institute), Ames et al. (2017)
     "Control Barrier Function Based Quadratic Programs for Safety
      Critical Systems"
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


def _wrap_angle(a: float) -> float:
    """각도를 [-π, π]로 정규화"""
    return float(np.arctan2(np.sin(a), np.cos(a)))


# ============================================================================
#  QP Solver
# ============================================================================


class CBFCLFQPSolver:
    """
    CLF-CBF-QP 솔버 (numpy/scipy 전용, 외부 QP 의존성 없음)

    min_{u, δ}  (u - u_nom)ᵀ P (u - u_nom) + λ_clf · δ²
    s.t.        A_cbf · u ≥ b_cbf     (hard)
                a_clf · u ≤ b_clf + δ (soft CLF)
                u_min ≤ u ≤ u_max,  0 ≤ δ ≤ delta_max

    Args:
        P: (nu, nu) 제어 비용 행렬 (None이면 identity)
        lambda_clf: CLF slack 페널티 가중치 (cbfkit slack_penalty_clf 대응)
        delta_max: slack 상한 (cbfkit slack_bound_clf 대응)
        use_analytic: 해석적 fast path 사용 여부 (False면 항상 SLSQP)
        slsqp_maxiter: SLSQP 최대 반복
        feas_tol: 제약 만족 판정 허용 오차
    """

    def __init__(
        self,
        P: Optional[np.ndarray] = None,
        lambda_clf: float = 100.0,
        delta_max: float = 1e6,
        use_analytic: bool = True,
        slsqp_maxiter: int = 100,
        feas_tol: float = 1e-6,
    ):
        self.P = None if P is None else np.asarray(P, dtype=float)
        self.lambda_clf = float(lambda_clf)
        self.delta_max = float(delta_max)
        self.use_analytic = use_analytic
        self.slsqp_maxiter = slsqp_maxiter
        self.feas_tol = float(feas_tol)

    # ------------------------------------------------------------------
    def solve(
        self,
        u_nom: np.ndarray,
        A_cbf: Optional[np.ndarray] = None,
        b_cbf: Optional[np.ndarray] = None,
        a_clf: Optional[np.ndarray] = None,
        b_clf: float = 0.0,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """
        QP 풀기

        Args:
            u_nom: (nu,) 명목 제어
            A_cbf: (m, nu) CBF 제약 행렬 (A_cbf·u ≥ b_cbf), None이면 CBF 없음
            b_cbf: (m,) CBF 하한
            a_clf: (nu,) CLF 제약 벡터 (a_clf·u ≤ b_clf + δ), None이면 CLF 없음
            b_clf: CLF 우변
            u_min, u_max: (nu,) 제어 경계 (None이면 무제한)

        Returns:
            (u_opt, feasible, info)
            info: {"method", "delta", "active_constraints", "solve_time",
                   "max_cbf_violation", "n_cbf"}
        """
        t0 = time.perf_counter()
        u_nom = np.asarray(u_nom, dtype=float).ravel()
        nu = u_nom.size
        P = self.P if self.P is not None else np.eye(nu)
        assert P.shape == (nu, nu), f"P shape {P.shape} != ({nu},{nu})"

        have_cbf = A_cbf is not None and len(np.atleast_2d(A_cbf)) > 0
        if have_cbf:
            A_cbf = np.atleast_2d(np.asarray(A_cbf, dtype=float))
            b_cbf = np.atleast_1d(np.asarray(b_cbf, dtype=float))
            assert A_cbf.shape == (b_cbf.size, nu)
        have_clf = a_clf is not None
        if have_clf:
            a_clf = np.asarray(a_clf, dtype=float).ravel()
            b_clf = float(b_clf)

        lo = np.full(nu, -np.inf) if u_min is None else np.asarray(u_min, dtype=float)
        hi = np.full(nu, np.inf) if u_max is None else np.asarray(u_max, dtype=float)

        def _clip(u: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(u, lo), hi)

        def _cbf_resid(u: np.ndarray) -> np.ndarray:
            """A·u - b (≥ 0이면 만족)"""
            return A_cbf @ u - b_cbf

        def _clf_slack(u: np.ndarray) -> float:
            return max(0.0, float(a_clf @ u - b_clf)) if have_clf else 0.0

        def _finish(u, feasible, method, delta):
            u = _clip(np.asarray(u, dtype=float))
            resid = _cbf_resid(u) if have_cbf else np.array([])
            act_tol = self.feas_tol * 100.0
            active = [
                int(i)
                for i in range(resid.size)
                if abs(resid[i]) <= act_tol * (1.0 + abs(b_cbf[i]))
            ]
            max_viol = float(max(0.0, -resid.min())) if resid.size else 0.0
            info = {
                "method": method,
                "delta": float(delta),
                "active_constraints": active,
                "solve_time": time.perf_counter() - t0,
                "max_cbf_violation": max_viol,
                "n_cbf": int(resid.size),
            }
            return u, bool(feasible), info

        def _cbf_ok(u: np.ndarray) -> bool:
            if not have_cbf:
                return True
            r = _cbf_resid(u)
            return bool(np.all(r >= -self.feas_tol * (1.0 + np.abs(b_cbf))))

        # ------------------------------------------------------------------
        # 1) 해석적 fast path
        # ------------------------------------------------------------------
        if self.use_analytic:
            # (a) CLF-tradeoff 무제약 최적해 (soft CLF 닫힌형)
            #     δ*(u) = max(0, a·u - b) 대입 → 1D 축소 후 닫힌형
            if have_clf:
                s = float(a_clf @ u_nom - b_clf)
                if s > 0.0:
                    Pinv_a = np.linalg.solve(P, a_clf)
                    q = float(a_clf @ Pinv_a)
                    u_cand = u_nom - (self.lambda_clf * s / (1.0 + self.lambda_clf * q)) * Pinv_a
                    delta_cand = s / (1.0 + self.lambda_clf * q)
                else:
                    u_cand, delta_cand = u_nom.copy(), 0.0
            else:
                u_cand, delta_cand = u_nom.copy(), 0.0

            u_c = _clip(u_cand)
            clipped = not np.allclose(u_c, u_cand, atol=1e-12)
            P_is_diag = np.allclose(P, np.diag(np.diag(P)))
            # 클리핑이 발생하면: P가 대각일 때만 클리핑이 정확한 투영.
            # CLF가 활성이었다면 클리핑 후 tradeoff가 달라지므로 SLSQP로.
            clip_ok = (not clipped) or (P_is_diag and delta_cand == 0.0)
            clf_ok_at_uc = (not have_clf) or (float(a_clf @ u_c - b_clf) <= delta_cand + self.feas_tol)

            if clip_ok and clf_ok_at_uc and _cbf_ok(u_c):
                return _finish(u_c, True, "analytic", delta_cand)

            # (b) 단일 CBF 활성 → 등식 투영 (closed-form)
            #     min (u-u_nom)ᵀP(u-u_nom) s.t. g·u = c
            #     u* = u_nom + P⁻¹g·(c - g·u_nom)/(g·P⁻¹g)
            if have_cbf:
                resid_c = _cbf_resid(u_c)
                violated = np.where(resid_c < -self.feas_tol * (1.0 + np.abs(b_cbf)))[0]
                if violated.size == 1:
                    i = int(violated[0])
                    g, c = A_cbf[i], float(b_cbf[i])
                    Pinv_g = np.linalg.solve(P, g)
                    denom = float(g @ Pinv_g)
                    if denom > 1e-12:
                        u_proj = u_nom + Pinv_g * (c - float(g @ u_nom)) / denom
                        in_bounds = bool(
                            np.all(u_proj >= lo - self.feas_tol)
                            and np.all(u_proj <= hi + self.feas_tol)
                        )
                        clf_inactive = (not have_clf) or (
                            float(a_clf @ u_proj - b_clf) <= self.feas_tol
                        )
                        if in_bounds and clf_inactive and _cbf_ok(u_proj):
                            return _finish(u_proj, True, "analytic_projection", 0.0)

        # ------------------------------------------------------------------
        # 2) SLSQP 폴백 (다중 제약 / 경계 활성 / CLF-CBF 동시 활성)
        # ------------------------------------------------------------------
        n = nu + (1 if have_clf else 0)
        lam = self.lambda_clf

        def f(z):
            du = z[:nu] - u_nom
            val = float(du @ P @ du)
            if have_clf:
                val += lam * z[nu] ** 2
            return val

        def f_jac(z):
            g = np.zeros(n)
            g[:nu] = 2.0 * (P @ (z[:nu] - u_nom))
            if have_clf:
                g[nu] = 2.0 * lam * z[nu]
            return g

        cons = []
        if have_cbf:
            A_ext = np.hstack([A_cbf, np.zeros((A_cbf.shape[0], 1))]) if have_clf else A_cbf
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda z: A_cbf @ z[:nu] - b_cbf,
                    "jac": lambda z: A_ext,
                }
            )
        if have_clf:
            clf_row = np.concatenate([-a_clf, [1.0]]).reshape(1, -1)
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda z: np.array([b_clf + z[nu] - float(a_clf @ z[:nu])]),
                    "jac": lambda z: clf_row,
                }
            )

        bnds = [
            (None if not np.isfinite(lo[i]) else lo[i], None if not np.isfinite(hi[i]) else hi[i])
            for i in range(nu)
        ]
        if have_clf:
            bnds.append((0.0, self.delta_max))

        u0 = _clip(u_nom)
        z0 = np.concatenate([u0, [_clf_slack(u0)]]) if have_clf else u0.copy()

        try:
            res = minimize(
                f,
                z0,
                jac=f_jac,
                method="SLSQP",
                bounds=bnds,
                constraints=cons,
                options={"maxiter": self.slsqp_maxiter, "ftol": 1e-9},
            )
            z_opt = res.x
            success = bool(res.success) and bool(np.all(np.isfinite(z_opt)))
        except Exception:
            z_opt, success = z0, False

        if not np.all(np.isfinite(z_opt)):
            z_opt = z0

        u_opt = _clip(z_opt[:nu])
        delta = float(np.clip(z_opt[nu], 0.0, self.delta_max)) if have_clf else 0.0

        feasible = success and _cbf_ok(u_opt)
        if not feasible:
            # best-effort: SLSQP 결과 vs 클리핑된 명목 제어 중 위반이 적은 쪽
            if have_cbf:
                viol_opt = float(max(0.0, -(_cbf_resid(u_opt)).min()))
                viol_nom = float(max(0.0, -(_cbf_resid(u0)).min()))
                if viol_nom < viol_opt:
                    u_opt = u0
                    delta = _clf_slack(u0)
            return _finish(u_opt, False, "slsqp_infeasible", delta)

        return _finish(u_opt, True, "slsqp", delta)

    def __repr__(self) -> str:
        return f"CBFCLFQPSolver(lambda_clf={self.lambda_clf}, use_analytic={self.use_analytic})"


# ============================================================================
#  Params
# ============================================================================


@dataclass
class CLFCBFQPParams:
    """
    CLF-CBF-QP 컨트롤러 파라미터

    Attributes:
        alpha_cbf: CBF class-K 계수 (ḣ ≥ -α·h)
        c_clf: CLF 수렴률 (V̇ ≤ -c·V + δ)
        lambda_clf: CLF slack 페널티 (안전-수렴 충돌 시 안전 우선 강도)
        lookahead_d: near-identity look-ahead 거리 d (p̃ = p + d·heading)
        safety_margin: 장애물 추가 안전 마진 (m)
        k_p: 명목 제어 위치 게인 (ṗ̃_des = -k_p·e)
        k_theta: CLF heading 항 가중치 (V = 0.5·|e|² + k_θ·e_θ²)
        k_heading: CBF-only 명목 제어의 heading P 게인
        goal_lookahead: 레퍼런스 look-ahead 포인트 선택 거리 (pure-pursuit)
        dt: 제어 주기 (레퍼런스 feedforward 속도 추정용)
        lambda_hocbf: 동역학 모델 HOCBF 캐스케이드 계수 (h_e = ḣ + λ·h)
        k_vel: 동역학 모델 속도 오차 게인 (backstepping-lite)
        p_weights: QP 제어 비용 대각 가중치 (None이면 identity)
        delta_max: slack 상한
        use_analytic: 솔버 해석적 fast path 사용
    """

    alpha_cbf: float = 1.5
    c_clf: float = 1.0
    lambda_clf: float = 50.0
    lookahead_d: float = 0.15
    safety_margin: float = 0.15
    k_p: float = 1.2
    k_theta: float = 0.3
    k_heading: float = 2.0
    goal_lookahead: float = 0.5
    dt: float = 0.05
    lambda_hocbf: float = 2.0
    k_vel: float = 3.0
    p_weights: Optional[Sequence[float]] = None
    delta_max: float = 1e6
    use_analytic: bool = True


# ============================================================================
#  CLF-CBF-QP Controller
# ============================================================================


class CLFCBFQPController:
    """
    CLF-CBF-QP 독립 베이스라인 컨트롤러

    repo 표준 인터페이스 준수:
        compute_control(state, reference_trajectory) -> (control, info)

    기구학 (state_dim == 3, u=[v, ω]):
        - look-ahead 포인트 p̃ = p + d·[cosθ, sinθ], ṗ̃ = M(θ)·u (M 가역)
        - CLF: V = 0.5·|p̃ - p_ref|² + k_θ·e_θ², 조건 ∇V·ṗ̃ ≤ -c·V + δ
        - CBF: h_i = |p̃ - p_o|² - (r + d + margin)², ḣ_i ≥ -α·h_i

    동역학 (state_dim == 5, u=[a, α_acc]):
        - HOCBF 1단 캐스케이드: h_e = ḣ + λ·h → ḣ_e ≥ -α·h_e (u가 ḧ에 등장)
        - CLF: 속도 오차 e_w = w - w_d (w_d는 기구학 CLF의 desired twist),
          V = 0.5·|e_w|², V̇ = e_wᵀ(u - C·w) ≤ -c·V + δ (backstepping-lite,
          ẇ_d ≈ 0 근사)

    Args:
        model: RobotModel (DifferentialDriveKinematic 또는 DifferentialDriveDynamic)
        params: CLFCBFQPParams
        obstacles: [(x, y, radius), ...] 원형 장애물 리스트
    """

    use_clf: bool = True

    def __init__(
        self,
        model,
        params: Optional[CLFCBFQPParams] = None,
        obstacles: Optional[List[tuple]] = None,
    ):
        self.model = model
        self.params = params if params is not None else CLFCBFQPParams()
        self.obstacles = [tuple(o) for o in (obstacles or [])]

        bounds = model.get_control_bounds()
        if bounds is not None:
            self.u_min, self.u_max = bounds[0].copy(), bounds[1].copy()
        else:
            self.u_min, self.u_max = None, None

        self.is_dynamic = model.state_dim >= 5
        p = self.params
        P = np.diag(np.asarray(p.p_weights, dtype=float)) if p.p_weights is not None else None
        self.solver = CBFCLFQPSolver(
            P=P,
            lambda_clf=p.lambda_clf,
            delta_max=p.delta_max,
            use_analytic=p.use_analytic,
        )
        self.step_count = 0
        self._detour_side = 0.0  # 장애물 우회 방향 히스테리시스 (+1/-1/0)

    # ------------------------------------------------------------------
    def reset(self):
        """컨트롤러 상태 초기화 (Simulator 호환)"""
        self.step_count = 0
        self._detour_side = 0.0

    def set_obstacles(self, obstacles: List[tuple]):
        self.obstacles = [tuple(o) for o in (obstacles or [])]

    # ------------------------------------------------------------------
    #  기하 헬퍼
    # ------------------------------------------------------------------
    def _lookahead_map(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """look-ahead 포인트 p̃와 맵 M(θ): ṗ̃ = M·[v, ω]"""
        th = float(state[2])
        d = self.params.lookahead_d
        c, s = np.cos(th), np.sin(th)
        p_tilde = state[:2] + d * np.array([c, s])
        M = np.array([[c, -d * s], [s, d * c]])
        return p_tilde, M

    def _select_reference_point(
        self, position: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        pure-pursuit 방식 레퍼런스 포인트 선택 + feedforward 속도 추정

        Returns:
            (ref_point (2,), v_ff)
        """
        ref = np.atleast_2d(np.asarray(reference_trajectory, dtype=float))
        pts = ref[:, :2]
        dists = np.linalg.norm(pts - position[:2], axis=1)
        i_near = int(np.argmin(dists))
        target = pts[-1]
        for i in range(i_near, pts.shape[0]):
            if dists[i] >= self.params.goal_lookahead:
                target = pts[i]
                break
        # feedforward 속도: 레퍼런스 점 간 간격 / dt
        if pts.shape[0] >= 2:
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            v_ff = float(np.median(seg)) / max(self.params.dt, 1e-9)
        else:
            v_ff = 0.0
        return target, v_ff

    # ------------------------------------------------------------------
    #  장애물 인지 명목 제어 (deadlock 방지 접선 투영)
    # ------------------------------------------------------------------
    def _project_pdot_around_obstacles(
        self, p_tilde: np.ndarray, p_dot: np.ndarray
    ) -> np.ndarray:
        """
        원하는 look-ahead 속도 ṗ̃_des에서 CBF를 위반하는 반경 방향(장애물
        방향) 성분을 제거하고 접선 성분을 유지. head-on 정체(deadlock)
        상황에서는 히스테리시스로 일관된 우회 방향의 최소 접선 속도를 주입.

        주의: 안전 보장은 QP의 hard CBF 제약이 담당. 이 투영은 명목 제어가
        장애물 반대편 목표를 향해 밀어붙이며 정지하는 것을 막는 휴리스틱.
        """
        if not self.obstacles:
            return p_dot
        p = self.params
        p_dot = p_dot.copy()
        speed_ref = max(float(np.linalg.norm(p_dot)), 0.3)
        for ox, oy, orad in self.obstacles:
            e_o = p_tilde - np.array([ox, oy])
            dist = float(np.linalg.norm(e_o))
            if dist < 1e-9:
                continue
            R = orad + p.lookahead_d + p.safety_margin
            h = dist**2 - R**2
            if h > max(R, 1.0):  # 충분히 멀면 우회 방향 초기화
                continue
            n = e_o / dist
            # CBF: 2·e_oᵀ·ṗ̃ ≥ -α·h → n·ṗ̃ ≥ -α·h/(2·dist)
            allowed = -p.alpha_cbf * h / (2.0 * dist)
            radial = float(n @ p_dot)
            if radial >= allowed:
                continue
            # 반경 방향 위반분 제거
            p_dot = p_dot + (allowed - radial) * n
            # deadlock 방지: 접선 성분이 작으면 일관된 방향으로 주입
            t = np.array([-n[1], n[0]])
            t_comp = float(t @ p_dot)
            min_tan = 0.3 * speed_ref
            if abs(t_comp) < min_tan:
                if self._detour_side == 0.0:
                    self._detour_side = 1.0 if t_comp >= 0.0 else -1.0
                p_dot = p_dot + (self._detour_side * min_tan - t_comp) * t
            else:
                self._detour_side = 1.0 if t_comp >= 0.0 else -1.0
        return p_dot

    def _pdot_to_twist(self, p_dot: np.ndarray, M: np.ndarray) -> np.ndarray:
        """ṗ̃ → [v, ω]. twist 한계를 넘으면 방향 보존 스케일링."""
        u = np.linalg.solve(M, p_dot)
        w_lo, w_hi = self._twist_limits()
        scale = max(1.0, float(np.max(np.abs(u) / np.maximum(w_hi, 1e-9))))
        return u / scale

    # ------------------------------------------------------------------
    #  기구학 (3D) 항 구성
    # ------------------------------------------------------------------
    def _nominal_kinematic(
        self,
        state: np.ndarray,
        ref_pt: np.ndarray,
        v_ff: float,
        p_tilde: np.ndarray,
        M: np.ndarray,
    ) -> np.ndarray:
        """명목 제어: 장애물 인지 투영 후 look-ahead 맵 역변환"""
        p_dot_des = -self.params.k_p * (p_tilde - ref_pt)
        p_dot_des = self._project_pdot_around_obstacles(p_tilde, p_dot_des)
        return self._clip_u(self._pdot_to_twist(p_dot_des, M))

    def _clf_kinematic(
        self,
        state: np.ndarray,
        ref_pt: np.ndarray,
        p_tilde: np.ndarray,
        M: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        unicycle CLF: V = 0.5·|e|² + k_θ·e_θ²

        V̇ ≈ eᵀ·M·u - 2·k_θ·e_θ·ω  (근사: ė_θ ≈ -ω, 레퍼런스 정지 가정)
        조건: a_clf·u ≤ -c·V + δ

        Returns:
            (a_clf (2,), b_clf, V)
        """
        p = self.params
        e = p_tilde - ref_pt
        dvec = ref_pt - state[:2]
        dist = float(np.linalg.norm(dvec))
        if dist > 1e-3:
            e_th = _wrap_angle(np.arctan2(dvec[1], dvec[0]) - float(state[2]))
        else:
            e_th = 0.0
        V = 0.5 * float(e @ e) + p.k_theta * e_th**2
        a_clf = M.T @ e + np.array([0.0, -2.0 * p.k_theta * e_th])
        b_clf = -p.c_clf * V
        return a_clf, float(b_clf), float(V)

    def _cbf_rows_kinematic(
        self, p_tilde: np.ndarray, M: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[float]]:
        """
        기구학 CBF: h = |p̃ - p_o|² - R², R = r + d + margin
        ḣ = 2·(p̃ - p_o)ᵀ·M·u ≥ -α·h → A·u ≥ b

        Returns:
            (A_cbf (m,2), b_cbf (m,), h_values)
        """
        if not self.obstacles:
            return None, None, []
        p = self.params
        rows, rhs, h_vals = [], [], []
        for ox, oy, orad in self.obstacles:
            e_o = p_tilde - np.array([ox, oy])
            R = orad + p.lookahead_d + p.safety_margin
            h = float(e_o @ e_o - R**2)
            rows.append(2.0 * (M.T @ e_o))
            rhs.append(-p.alpha_cbf * h)
            h_vals.append(h)
        return np.array(rows), np.array(rhs), h_vals

    # ------------------------------------------------------------------
    #  동역학 (5D) 항 구성
    # ------------------------------------------------------------------
    def _friction_matrix(self) -> np.ndarray:
        """C = diag(c_v, c_ω) — 모델에 마찰 계수가 없으면 0"""
        return np.diag(
            [getattr(self.model, "c_v", 0.0), getattr(self.model, "c_omega", 0.0)]
        )

    def _twist_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        v_max = getattr(self.model, "v_max", 1.0)
        w_max = getattr(self.model, "omega_max", 1.0)
        hi = np.array([v_max, w_max])
        return -hi, hi

    def _desired_twist(
        self,
        state: np.ndarray,
        ref_pt: np.ndarray,
        v_ff: float,
        p_tilde: np.ndarray,
        M: np.ndarray,
    ) -> np.ndarray:
        """기구학 CLF의 desired twist (장애물 인지 투영 포함), 한계 스케일링"""
        p_dot_des = -self.params.k_p * (p_tilde - ref_pt)
        p_dot_des = self._project_pdot_around_obstacles(p_tilde, p_dot_des)
        return self._pdot_to_twist(p_dot_des, M)

    def _terms_dynamic(
        self,
        state: np.ndarray,
        ref_pt: np.ndarray,
        v_ff: float,
        p_tilde: np.ndarray,
        M: np.ndarray,
    ):
        """
        동역학 모델 QP 항 구성

        CBF (HOCBF 1단): h_e = ḣ + λ·h, 조건 ḣ_e ≥ -α·h_e
            ḣ = 2·e_oᵀ·M·w  (제어 미포함)
            ḧ = 2|Mw|² + 2ω·e_oᵀ·(dM/dθ)·w - 2·e_oᵀ·M·C·w + 2·e_oᵀ·M·u
            → A = 2·e_oᵀ·M,  b = -λ·ḣ - α·h_e - drift

        CLF (backstepping-lite): e_w = w - w_d, V = 0.5·|e_w|²
            V̇ = e_wᵀ·(u - C·w) ≤ -c·V + δ  (ẇ_d ≈ 0 근사)

        Returns:
            (u_nom, a_clf, b_clf, V, A_cbf, b_cbf, h_values)
        """
        p = self.params
        th = float(state[2])
        w = state[3:5].copy()
        C = self._friction_matrix()

        # 명목 제어: ẇ = u - C·w = -k_vel·e_w 가 되도록
        w_d = self._desired_twist(state, ref_pt, v_ff, p_tilde, M)
        e_w = w - w_d
        u_nom = self._clip_u(C @ w - p.k_vel * e_w)

        # CLF
        V = 0.5 * float(e_w @ e_w)
        a_clf = e_w.copy()
        b_clf = -p.c_clf * V + float(e_w @ (C @ w))

        # CBF (HOCBF)
        A_cbf, b_cbf, h_vals = None, None, []
        if self.obstacles:
            c, s = np.cos(th), np.sin(th)
            d = p.lookahead_d
            dM_dth = np.array([[-s, -d * c], [c, -d * s]])
            Mw = M @ w
            rows, rhs = [], []
            for ox, oy, orad in self.obstacles:
                e_o = p_tilde - np.array([ox, oy])
                R = orad + d + p.safety_margin
                h = float(e_o @ e_o - R**2)
                h_dot = 2.0 * float(e_o @ Mw)
                h_e = h_dot + p.lambda_hocbf * h
                drift = (
                    2.0 * float(Mw @ Mw)
                    + 2.0 * float(w[1]) * float(e_o @ (dM_dth @ w))
                    - 2.0 * float(e_o @ (M @ (C @ w)))
                )
                rows.append(2.0 * (M.T @ e_o))
                rhs.append(-p.lambda_hocbf * h_dot - p.alpha_cbf * h_e - drift)
                h_vals.append(h)
            A_cbf, b_cbf = np.array(rows), np.array(rhs)

        return u_nom, a_clf, float(b_clf), float(V), A_cbf, b_cbf, h_vals

    # ------------------------------------------------------------------
    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        if self.u_min is None:
            return u
        return np.clip(u, self.u_min, self.u_max)

    # ------------------------------------------------------------------
    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        CLF-CBF-QP 제어 계산 (repo 표준 인터페이스)

        Args:
            state: (3,) [x,y,θ] 또는 (5,) [x,y,θ,v,ω]
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (2,) 제어 입력
            info: {"clf_value", "min_barrier", "qp_feasible", "delta",
                   "active_constraints", "solve_time", ...}
        """
        t0 = time.perf_counter()
        state = np.asarray(state, dtype=float).ravel()
        p_tilde, M = self._lookahead_map(state)
        ref_pt, v_ff = self._select_reference_point(state, reference_trajectory)

        if self.is_dynamic:
            u_nom, a_clf, b_clf, V, A_cbf, b_cbf, h_vals = self._terms_dynamic(
                state, ref_pt, v_ff, p_tilde, M
            )
        else:
            u_nom = self._nominal_kinematic(state, ref_pt, v_ff, p_tilde, M)
            a_clf, b_clf, V = self._clf_kinematic(state, ref_pt, p_tilde, M)
            A_cbf, b_cbf, h_vals = self._cbf_rows_kinematic(p_tilde, M)

        if not self.use_clf:
            a_clf, b_clf, V = None, 0.0, 0.0

        u_opt, feasible, sinfo = self.solver.solve(
            u_nom,
            A_cbf=A_cbf,
            b_cbf=b_cbf,
            a_clf=a_clf,
            b_clf=b_clf,
            u_min=self.u_min,
            u_max=self.u_max,
        )
        self.step_count += 1

        info = {
            "clf_value": float(V),
            "min_barrier": float(min(h_vals)) if h_vals else float("inf"),
            "barrier_values": h_vals,
            "qp_feasible": bool(feasible),
            "delta": float(sinfo["delta"]),
            "active_constraints": sinfo["active_constraints"],
            "solve_time": time.perf_counter() - t0,
            "qp_method": sinfo["method"],
            "u_nominal": u_nom.copy(),
            "reference_point": ref_pt.copy(),
        }
        return u_opt, info

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"model={'dynamic' if self.is_dynamic else 'kinematic'}, "
            f"num_obstacles={len(self.obstacles)}, "
            f"alpha_cbf={self.params.alpha_cbf}, c_clf={self.params.c_clf})"
        )


# ============================================================================
#  CBF-Only QP Controller (safe_control CBF-QP 대응 베이스라인)
# ============================================================================


class CBFOnlyQPController(CLFCBFQPController):
    """
    CBF-only QP 컨트롤러 (CLF 없음)

    명목 제어 = 레퍼런스 속도 feedforward + P 피드백 (pure-pursuit 스타일),
    안전은 CBF 제약만으로 보장. safe_control의 CBF-QP와 가장 가까운 구조로,
    head-to-head 비교용.

        min_u ||u - u_nom||²_P  s.t.  A_cbf·u ≥ b_cbf,  u_min ≤ u ≤ u_max
    """

    use_clf = False

    def _nominal_kinematic(
        self,
        state: np.ndarray,
        ref_pt: np.ndarray,
        v_ff: float,
        p_tilde: np.ndarray,
        M: np.ndarray,
    ) -> np.ndarray:
        """pure-pursuit 명목: v = v_ff + k_p·dist·cos(e_θ), ω = k_heading·e_θ
        (장애물 인지 접선 투영 적용)"""
        u_pp = self._pure_pursuit_twist(state, ref_pt, v_ff)
        # look-ahead 속도 공간에서 장애물 인지 투영 후 역변환
        p_dot = M @ u_pp
        p_dot = self._project_pdot_around_obstacles(p_tilde, p_dot)
        return self._clip_u(self._pdot_to_twist(p_dot, M))

    def _pure_pursuit_twist(
        self, state: np.ndarray, ref_pt: np.ndarray, v_ff: float
    ) -> np.ndarray:
        """레퍼런스 속도 feedforward + P 피드백 (pure-pursuit 스타일)"""
        p = self.params
        dvec = ref_pt - state[:2]
        dist = float(np.linalg.norm(dvec))
        if dist > 1e-3:
            e_th = _wrap_angle(np.arctan2(dvec[1], dvec[0]) - float(state[2]))
        else:
            e_th = 0.0
        v_nom = v_ff + p.k_p * dist * np.cos(e_th)
        w_nom = p.k_heading * e_th
        w_lo, w_hi = self._twist_limits()
        return np.clip(np.array([v_nom, w_nom]), w_lo, w_hi)

    def _desired_twist(
        self,
        state: np.ndarray,
        ref_pt: np.ndarray,
        v_ff: float,
        p_tilde: np.ndarray,
        M: np.ndarray,
    ) -> np.ndarray:
        """동역학 모델: pure-pursuit twist를 desired로 사용 (투영 포함)"""
        u_pp = self._pure_pursuit_twist(state, ref_pt, v_ff)
        p_dot = M @ u_pp
        p_dot = self._project_pdot_around_obstacles(p_tilde, p_dot)
        return self._pdot_to_twist(p_dot, M)
