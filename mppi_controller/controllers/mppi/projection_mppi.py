"""
pi-MPPI (Projection-based MPPI) Controller — 29번째 MPPI 변형

QP Projection으로 제어 입력의 크기/jerk/snap에 대한 하드 제약을 보장.
후처리 스무딩이 아닌 사전적(a priori) 매끄러움 보장.

핵심 수식:
    1. 각 샘플 V_k = [v_0, ..., v_{N-1}]에 대해:
       - |v_t| ≤ u_max (크기 제약)
       - |v_t - v_{t-1}| / dt ≤ jerk_max (변화율 제약)
       - |(v_t - 2v_{t-1} + v_{t-2})| / dt² ≤ snap_max (2차 도함수 제약)
    2. 제약 위반 시 최소 보정 projection:
       min ||V_proj - V_orig||² s.t. constraints
    3. 투영 후 표준 MPPI 가중 평균

기존 변형 대비 핵심 차이:
    - LP-MPPI: Butterworth LPF 주파수 영역 필터링 (통계적 soft 제약)
    - Smooth-MPPI: Input-Lifting + Jerk cost (비용에 의한 soft 제약)
    - pi-MPPI: QP projection → 명시적 hard 제약 (물리적 한계 보장)

Reference: Andrejev et al., "pi-MPPI: A Projection-based MPPI for Smooth
    Optimal Control", RA-L 2025, arXiv:2504.10962
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

from mppi_controller.models.base_model import RobotModel
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.mppi_params import ProjectionMPPIParams
from mppi_controller.controllers.mppi.cost_functions import CostFunction
from mppi_controller.controllers.mppi.sampling import NoiseSampler


class ProjectionMPPIController(MPPIController):
    """
    pi-MPPI 컨트롤러 (29번째 MPPI 변형)

    QP projection으로 제어 시퀀스의 물리적 제약(크기, jerk, snap)을
    하드 제약으로 보장. 샘플 투영 + 최종 출력 투영 이중 적용.

    Args:
        model: RobotModel 인스턴스
        params: ProjectionMPPIParams 파라미터
        cost_function: CostFunction (None이면 기본 비용 함수)
        noise_sampler: NoiseSampler (None이면 GaussianSampler 자동 생성)
    """

    def __init__(
        self,
        model: RobotModel,
        params: ProjectionMPPIParams,
        cost_function: Optional[CostFunction] = None,
        noise_sampler: Optional[NoiseSampler] = None,
    ):
        super().__init__(model, params, cost_function, noise_sampler)

        self.proj_params = params
        self._smoothness_history: List[Dict] = []

        # 이전 제어 입력 (jerk/snap 제약에 사용)
        self._prev_control = np.zeros(model.control_dim)
        self._prev_prev_control = np.zeros(model.control_dim)

    def compute_control(
        self, state: np.ndarray, reference_trajectory: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        pi-MPPI 제어 계산

        1. 노이즈 샘플링 → 제어 시퀀스 생성
        2. _project_samples(): 각 샘플에 제약 투영
        3. rollout → 비용 → 가중 평균
        4. _project_output(): 최종 U에도 투영
        5. Receding horizon shift

        Args:
            state: (nx,) 현재 상태
            reference_trajectory: (N+1, nx) 레퍼런스 궤적

        Returns:
            control: (nu,) 최적 제어 입력
            info: dict - 표준 MPPI info + projection_stats
        """
        K = self.params.K
        N = self.params.N

        # 1. 노이즈 샘플링 (K, N, nu)
        noise = self.noise_sampler.sample(self.U, K, self.u_min, self.u_max)

        # 2. 샘플 제어 시퀀스 (K, N, nu)
        sampled_controls = self.U + noise

        # 제어 크기 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            sampled_controls = np.clip(sampled_controls, self.u_min, self.u_max)

        # 3. 샘플 투영 (pi-MPPI 핵심)
        n_projected = 0
        if self.proj_params.project_samples:
            sampled_controls, n_projected = self._project_samples(sampled_controls)

        # 4. 궤적 rollout (K, N+1, nx)
        sample_trajectories = self.dynamics_wrapper.rollout(state, sampled_controls)

        # 5. 비용 계산 (K,)
        costs = self.cost_function.compute_cost(
            sample_trajectories, sampled_controls, reference_trajectory
        )

        # 6. MPPI 가중치 계산
        weights = self._compute_weights(costs, self.params.lambda_)

        # 7. 가중 평균으로 제어 업데이트
        # 투영된 제어에서 노이즈를 재계산
        projected_noise = sampled_controls - self.U
        weighted_noise = np.sum(weights[:, None, None] * projected_noise, axis=0)
        self.U = self.U + weighted_noise

        # 제어 크기 제약 클리핑
        if self.u_min is not None and self.u_max is not None:
            self.U = np.clip(self.U, self.u_min, self.u_max)

        # 8. 최종 출력 투영
        if self.proj_params.project_output:
            self.U = self._project_sequence(
                self.U.copy(), self._prev_control, self._prev_prev_control,
            )

        # 9. Smoothness 통계 계산
        smoothness = self._compute_smoothness_metrics(self.U)

        # 10. 최적 제어 (shift 전에 저장)
        optimal_control = self.U[0, :].copy()

        # 11. 이전 제어 업데이트
        self._prev_prev_control = self._prev_control.copy()
        self._prev_control = optimal_control.copy()

        # 12. Receding horizon shift
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1, :] = 0.0

        # 13. 정보 저장
        ess = self._compute_ess(weights)
        best_idx = np.argmin(costs)

        info = {
            "sample_trajectories": sample_trajectories,
            "sample_weights": weights,
            "best_trajectory": sample_trajectories[best_idx],
            "best_cost": costs[best_idx],
            "mean_cost": np.mean(costs),
            "temperature": self.params.lambda_,
            "ess": ess,
            "num_samples": K,
            # pi-MPPI 고유 정보
            "projection_stats": {
                "n_projected": n_projected,
                "projection_rate": n_projected / K if K > 0 else 0.0,
                "method": self.proj_params.projection_method,
            },
            "smoothness_stats": smoothness,
            "constraint_config": {
                "jerk_limit": self.proj_params.jerk_limit,
                "snap_limit": self.proj_params.snap_limit,
                "use_jerk": self.proj_params.use_jerk_constraint,
                "use_snap": self.proj_params.use_snap_constraint,
            },
        }
        self.last_info = info
        self._smoothness_history.append(smoothness)

        return optimal_control, info

    def _project_samples(
        self, controls: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        모든 샘플 제어 시퀀스에 제약 투영 적용

        Args:
            controls: (K, N, nu) 샘플 제어 시퀀스

        Returns:
            projected: (K, N, nu) 투영된 제어
            n_projected: 실제 투영이 적용된 샘플 수
        """
        K = controls.shape[0]
        projected = controls.copy()
        n_projected = 0

        if self.proj_params.projection_method == "clip":
            # 벡터화된 순차 클리핑
            projected, n_projected = self._project_samples_clip(controls)
        elif self.proj_params.projection_method == "qp":
            # QP 기반 최소 보정 (정확하지만 느림)
            for k in range(K):
                orig = controls[k].copy()
                projected[k] = self._project_sequence_qp(
                    controls[k], self._prev_control, self._prev_prev_control,
                )
                if not np.allclose(projected[k], orig, atol=1e-8):
                    n_projected += 1
        else:
            raise ValueError(
                f"Unknown projection method: {self.proj_params.projection_method}"
            )

        return projected, n_projected

    def _project_samples_clip(
        self, controls: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        벡터화된 순차 클리핑 투영

        각 샘플에 대해 time step 순서대로 jerk/snap 제약을 적용.
        Python 루프 최소화를 위해 K 축은 벡터화.

        Args:
            controls: (K, N, nu) 샘플 제어 시퀀스

        Returns:
            projected: (K, N, nu) 투영된 제어
            n_projected: 투영된 샘플 수
        """
        K, N, nu = controls.shape
        projected = controls.copy()
        dt = self.proj_params.dt
        jerk_limit = self.proj_params.jerk_limit
        snap_limit = self.proj_params.snap_limit
        use_jerk = self.proj_params.use_jerk_constraint
        use_snap = self.proj_params.use_snap_constraint

        any_modified = np.zeros(K, dtype=bool)

        # 크기 제약
        if self.u_min is not None and self.u_max is not None:
            orig = projected.copy()
            projected = np.clip(projected, self.u_min, self.u_max)
            # 샘플별 수정 여부: (K,) bool
            per_sample_diff = np.any(
                np.abs(projected - orig) > 1e-8,
                axis=(1, 2),
            )
            any_modified = any_modified | per_sample_diff

        if not use_jerk and not use_snap:
            return projected, int(np.sum(any_modified))

        # prev_u 배열: (K, nu) — 모든 샘플이 동일한 이전 제어
        prev = np.tile(self._prev_control, (K, 1))  # (K, nu)
        prev_prev = np.tile(self._prev_prev_control, (K, 1))  # (K, nu)

        for t in range(N):
            orig_t = projected[:, t, :].copy()  # (K, nu)

            # Jerk 제약: |v_t - v_{t-1}| / dt ≤ jerk_limit
            if use_jerk:
                delta = projected[:, t, :] - prev  # (K, nu)
                max_delta = jerk_limit * dt
                delta_clipped = np.clip(delta, -max_delta, max_delta)
                projected[:, t, :] = prev + delta_clipped

            # Snap 제약: |(v_t - 2*v_{t-1} + v_{t-2})| / dt² ≤ snap_limit
            if use_snap:
                # snap = (v_t - 2*prev + prev_prev) / dt²
                snap = (projected[:, t, :] - 2 * prev + prev_prev) / (dt ** 2)
                snap_clipped = np.clip(snap, -snap_limit, snap_limit)
                projected[:, t, :] = snap_clipped * (dt ** 2) + 2 * prev - prev_prev

                # snap 투영 후 jerk 재적용 (snap이 jerk를 위반할 수 있음)
                if use_jerk:
                    delta = projected[:, t, :] - prev
                    max_delta = jerk_limit * dt
                    delta_clipped = np.clip(delta, -max_delta, max_delta)
                    projected[:, t, :] = prev + delta_clipped

            # 크기 제약 재적용
            if self.u_min is not None and self.u_max is not None:
                projected[:, t, :] = np.clip(
                    projected[:, t, :], self.u_min, self.u_max
                )

            # 수정 여부 확인
            modified_t = np.any(
                np.abs(projected[:, t, :] - orig_t) > 1e-8, axis=1
            )
            any_modified = any_modified | modified_t

            # prev 업데이트
            prev_prev = prev.copy()
            prev = projected[:, t, :].copy()

        return projected, int(np.sum(any_modified))

    def _project_sequence(
        self,
        seq: np.ndarray,
        prev_u: np.ndarray,
        prev_prev_u: np.ndarray,
    ) -> np.ndarray:
        """
        단일 제어 시퀀스에 clip 투영 적용

        Args:
            seq: (N, nu) 제어 시퀀스
            prev_u: (nu,) 이전 제어 입력
            prev_prev_u: (nu,) 이전 이전 제어 입력

        Returns:
            projected: (N, nu) 투영된 시퀀스
        """
        if self.proj_params.projection_method == "qp":
            return self._project_sequence_qp(seq, prev_u, prev_prev_u)

        N, nu = seq.shape
        projected = seq.copy()
        dt = self.proj_params.dt
        jerk_limit = self.proj_params.jerk_limit
        snap_limit = self.proj_params.snap_limit
        use_jerk = self.proj_params.use_jerk_constraint
        use_snap = self.proj_params.use_snap_constraint

        # 크기 제약
        if self.u_min is not None and self.u_max is not None:
            projected = np.clip(projected, self.u_min, self.u_max)

        prev = prev_u.copy()
        pprev = prev_prev_u.copy()

        for t in range(N):
            # Jerk 제약
            if use_jerk:
                delta = projected[t] - prev
                max_delta = jerk_limit * dt
                delta = np.clip(delta, -max_delta, max_delta)
                projected[t] = prev + delta

            # Snap 제약 (t=0도 적용: prev=prev_u, pprev=prev_prev_u)
            if use_snap:
                snap = (projected[t] - 2 * prev + pprev) / (dt ** 2)
                snap = np.clip(snap, -snap_limit, snap_limit)
                projected[t] = snap * (dt ** 2) + 2 * prev - pprev

                # snap 후 jerk 재적용
                if use_jerk:
                    delta = projected[t] - prev
                    max_delta = jerk_limit * dt
                    delta = np.clip(delta, -max_delta, max_delta)
                    projected[t] = prev + delta

            # 크기 제약 재적용
            if self.u_min is not None and self.u_max is not None:
                projected[t] = np.clip(projected[t], self.u_min, self.u_max)

            pprev = prev.copy()
            prev = projected[t].copy()

        return projected

    def _project_sequence_qp(
        self,
        seq: np.ndarray,
        prev_u: np.ndarray,
        prev_prev_u: np.ndarray,
    ) -> np.ndarray:
        """
        QP 기반 최소 보정 투영

        min ||V_proj - V_orig||² s.t.
            |V_proj[t]| ≤ u_max
            |V_proj[t] - V_proj[t-1]| / dt ≤ jerk_limit
            |(V_proj[t] - 2*V_proj[t-1] + V_proj[t-2])| / dt² ≤ snap_limit

        scipy.optimize.minimize (SLSQP)를 사용.

        Args:
            seq: (N, nu) 원본 제어 시퀀스
            prev_u: (nu,) 이전 제어 입력
            prev_prev_u: (nu,) 이전 이전 제어 입력

        Returns:
            projected: (N, nu) QP 투영된 시퀀스
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            # scipy 없으면 clip fallback
            return self._project_sequence(seq, prev_u, prev_prev_u)

        N, nu = seq.shape
        dt = self.proj_params.dt
        jerk_limit = self.proj_params.jerk_limit
        snap_limit = self.proj_params.snap_limit
        use_jerk = self.proj_params.use_jerk_constraint
        use_snap = self.proj_params.use_snap_constraint

        x0 = seq.flatten()

        def objective(x):
            return np.sum((x - x0) ** 2)

        def grad(x):
            return 2 * (x - x0)

        # Bounds: 크기 제약
        bounds = []
        for t in range(N):
            for d in range(nu):
                lo = self.u_min[d] if self.u_min is not None else -1e6
                hi = self.u_max[d] if self.u_max is not None else 1e6
                bounds.append((lo, hi))

        constraints = []

        # Jerk 제약: |V[t] - V[t-1]| / dt ≤ jerk_limit
        if use_jerk:
            max_delta = jerk_limit * dt
            for t in range(N):
                for d in range(nu):
                    idx = t * nu + d
                    if t == 0:
                        prev_val = prev_u[d]
                    else:
                        prev_idx = (t - 1) * nu + d

                    if t == 0:
                        # V[0] - prev_u
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, p=prev_val, md=max_delta: md - (x[i] - p),
                        })
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, p=prev_val, md=max_delta: md + (x[i] - p),
                        })
                    else:
                        # V[t] - V[t-1]
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, pi=prev_idx, md=max_delta: md - (x[i] - x[pi]),
                        })
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, pi=prev_idx, md=max_delta: md + (x[i] - x[pi]),
                        })

        # Snap 제약: |(V[t] - 2V[t-1] + V[t-2])| / dt² ≤ snap_limit
        if use_snap:
            max_snap = snap_limit * dt ** 2
            for t in range(N):
                for d in range(nu):
                    idx = t * nu + d

                    if t == 0:
                        p1 = prev_u[d]
                        p2 = prev_prev_u[d]
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, v1=p1, v2=p2, ms=max_snap: ms - (x[i] - 2 * v1 + v2),
                        })
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, v1=p1, v2=p2, ms=max_snap: ms + (x[i] - 2 * v1 + v2),
                        })
                    elif t == 1:
                        p2 = prev_u[d]
                        pi = (t - 1) * nu + d
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, pi_=pi, v2=p2, ms=max_snap: ms - (x[i] - 2 * x[pi_] + v2),
                        })
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, pi_=pi, v2=p2, ms=max_snap: ms + (x[i] - 2 * x[pi_] + v2),
                        })
                    else:
                        pi1 = (t - 1) * nu + d
                        pi2 = (t - 2) * nu + d
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, p1=pi1, p2=pi2, ms=max_snap: ms - (x[i] - 2 * x[p1] + x[p2]),
                        })
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda x, i=idx, p1=pi1, p2=pi2, ms=max_snap: ms + (x[i] - 2 * x[p1] + x[p2]),
                        })

        result = minimize(
            objective, x0, jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 50, "ftol": 1e-8},
        )

        return result.x.reshape(N, nu)

    def _compute_smoothness_metrics(self, U: np.ndarray) -> Dict:
        """
        Smoothness 지표 계산: MSSD, jerk, snap

        Args:
            U: (N, nu) 제어 시퀀스

        Returns:
            dict: mssd, mean_jerk, max_jerk, mean_snap, max_snap,
                  jerk_violations, snap_violations
        """
        dt = self.proj_params.dt

        if U.shape[0] < 3:
            return {
                "mssd": 0.0,
                "mean_jerk": 0.0,
                "max_jerk": 0.0,
                "mean_snap": 0.0,
                "max_snap": 0.0,
                "jerk_violations": 0,
                "snap_violations": 0,
            }

        # MSSD
        second_diff = np.diff(U, n=2, axis=0)
        mssd = float(np.mean(second_diff ** 2))

        # Jerk: |U[t] - U[t-1]| / dt
        first_diff = np.diff(U, axis=0)  # (N-1, nu)
        jerk = np.abs(first_diff) / dt  # (N-1, nu)
        jerk_norms = np.linalg.norm(first_diff / dt, axis=1)
        mean_jerk = float(np.mean(jerk_norms))
        max_jerk = float(np.max(jerk_norms))

        # Snap: |U[t] - 2U[t-1] + U[t-2]| / dt²
        snap = np.abs(second_diff) / (dt ** 2)  # (N-2, nu)
        snap_norms = np.linalg.norm(second_diff / (dt ** 2), axis=1)
        mean_snap = float(np.mean(snap_norms))
        max_snap = float(np.max(snap_norms))

        # 제약 위반 수
        jerk_violations = int(np.sum(
            np.any(jerk > self.proj_params.jerk_limit * 1.01, axis=1)
        ))
        snap_violations = int(np.sum(
            np.any(snap > self.proj_params.snap_limit * 1.01, axis=1)
        ))

        return {
            "mssd": mssd,
            "mean_jerk": mean_jerk,
            "max_jerk": max_jerk,
            "mean_snap": mean_snap,
            "max_snap": max_snap,
            "jerk_violations": jerk_violations,
            "snap_violations": snap_violations,
        }

    def get_smoothness_statistics(self) -> Dict:
        """누적 smoothness 통계 반환"""
        if not self._smoothness_history:
            return {
                "num_steps": 0,
                "mean_mssd": 0.0,
                "mean_jerk": 0.0,
                "mean_snap": 0.0,
            }

        mssds = [s["mssd"] for s in self._smoothness_history]
        jerks = [s["mean_jerk"] for s in self._smoothness_history]
        snaps = [s["mean_snap"] for s in self._smoothness_history]

        return {
            "num_steps": len(self._smoothness_history),
            "mean_mssd": float(np.mean(mssds)),
            "std_mssd": float(np.std(mssds)),
            "mean_jerk": float(np.mean(jerks)),
            "std_jerk": float(np.std(jerks)),
            "mean_snap": float(np.mean(snaps)),
            "std_snap": float(np.std(snaps)),
            "total_jerk_violations": sum(
                s["jerk_violations"] for s in self._smoothness_history
            ),
            "total_snap_violations": sum(
                s["snap_violations"] for s in self._smoothness_history
            ),
        }

    def reset(self):
        """제어 시퀀스 + smoothness 히스토리 + prev 초기화"""
        super().reset()
        self._smoothness_history = []
        self._prev_control = np.zeros(self.model.control_dim)
        self._prev_prev_control = np.zeros(self.model.control_dim)

    def __repr__(self) -> str:
        return (
            f"ProjectionMPPIController("
            f"model={self.model.__class__.__name__}, "
            f"jerk_limit={self.proj_params.jerk_limit}, "
            f"snap_limit={self.proj_params.snap_limit}, "
            f"method={self.proj_params.projection_method})"
        )
