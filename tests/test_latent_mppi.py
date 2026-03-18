"""
Latent-Space MPPI 유닛 테스트

WorldModelVAE + WorldModelTrainer + WorldModelDynamics + LatentMPPIController 28개 테스트.
"""

import sys
import os
import numpy as np
import torch
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    LatentMPPIParams,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.latent_mppi import LatentMPPIController
from mppi_controller.controllers.mppi.cost_functions import (
    CompositeMPPICost,
    StateTrackingCost,
    TerminalCost,
    ControlEffortCost,
    ObstacleCost,
)
from mppi_controller.learning.world_model_trainer import (
    WorldModelVAE,
    WorldModelTrainer,
)
from mppi_controller.models.learned.world_model_dynamics import WorldModelDynamics
from mppi_controller.utils.trajectory import (
    generate_reference_trajectory,
    circle_trajectory,
    figure_eight_trajectory,
)
from mppi_controller.simulation.simulator import Simulator


# ── 헬퍼 ──────────────────────────────────────────────────────

def _make_model():
    return DifferentialDriveKinematic(v_max=1.0, omega_max=1.0)


def _make_latent_params(**kwargs):
    defaults = dict(
        K=32, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
        latent_dim=8,
        vae_hidden_dims=[32, 32],
    )
    defaults.update(kwargs)
    return LatentMPPIParams(**defaults)


def _make_ref(N=10, dt=0.05):
    return generate_reference_trajectory(circle_trajectory, 0.0, N, dt)


class MockWorldModel:
    """선형 encode/decode mock — 학습 없이 테스트"""

    def __init__(self, state_dim=3, latent_dim=8):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        np.random.seed(42)
        self._W_enc = np.random.randn(latent_dim, state_dim) * 0.1
        self._W_dec = np.random.randn(state_dim, latent_dim) * 0.1
        self._W_dyn = np.eye(latent_dim) * 0.99

    def encode(self, state):
        if state.ndim == 1:
            return self._W_enc @ state
        return state @ self._W_enc.T

    def decode(self, z):
        if z.ndim == 1:
            return self._W_dec @ z
        return z @ self._W_dec.T

    def latent_dynamics(self, z, control):
        if z.ndim == 1:
            return self._W_dyn @ z
        return z @ self._W_dyn.T


def _generate_training_data(n_samples=500, seed=42):
    """물리 모델에서 학습 데이터 생성"""
    np.random.seed(seed)
    model = _make_model()
    dt = 0.05

    states = []
    controls = []
    next_states = []

    state = np.array([0.0, 0.0, 0.0])
    for _ in range(n_samples):
        control = np.array([
            np.random.uniform(-0.5, 1.0),
            np.random.uniform(-1.0, 1.0),
        ])
        next_state = model.step(state, control, dt)
        states.append(state.copy())
        controls.append(control.copy())
        next_states.append(next_state.copy())
        state = next_state

        # 가끔 리셋
        if np.random.random() < 0.05:
            state = np.random.randn(3) * 0.5

    return np.array(states), np.array(controls), np.array(next_states)


# ══════════════════════════════════════════════════════════════
# LatentMPPIParams 테스트 (#1~#3)
# ══════════════════════════════════════════════════════════════

def test_params_defaults():
    """#1: 기본값 검증"""
    print("\n" + "=" * 60)
    print("Test #1: LatentMPPIParams 기본값")

    params = LatentMPPIParams()
    assert params.latent_dim == 16
    assert params.vae_hidden_dims == [128, 128]
    assert params.vae_beta == 0.001
    assert params.vae_model_path is None
    assert params.decode_interval == 1
    assert params.use_latent_rollout is True

    # 기본 MPPIParams 상속 확인
    assert params.N == 30
    assert params.K == 1024
    assert params.lambda_ == 1.0

    print("  PASSED")


def test_params_custom():
    """#2: 커스텀 값 설정"""
    print("\n" + "=" * 60)
    print("Test #2: LatentMPPIParams 커스텀 값")

    params = LatentMPPIParams(
        latent_dim=32,
        vae_hidden_dims=[256, 256],
        vae_beta=0.01,
        decode_interval=2,
        K=64,
        N=20,
    )
    assert params.latent_dim == 32
    assert params.vae_hidden_dims == [256, 256]
    assert params.vae_beta == 0.01
    assert params.decode_interval == 2

    print("  PASSED")


def test_params_validation():
    """#3: 잘못된 값 검증"""
    print("\n" + "=" * 60)
    print("Test #3: LatentMPPIParams 검증 오류")

    errors = 0

    try:
        LatentMPPIParams(latent_dim=0)
    except AssertionError:
        errors += 1

    try:
        LatentMPPIParams(decode_interval=-1)
    except AssertionError:
        errors += 1

    try:
        LatentMPPIParams(vae_beta=-0.1)
    except AssertionError:
        errors += 1

    assert errors == 3, f"Expected 3 errors, got {errors}"
    print("  PASSED")


# ══════════════════════════════════════════════════════════════
# WorldModelVAE 테스트 (#4~#9)
# ══════════════════════════════════════════════════════════════

def test_vae_construction():
    """#4: VAE 생성 + 파라미터 수"""
    print("\n" + "=" * 60)
    print("Test #4: WorldModelVAE 생성")

    vae = WorldModelVAE(state_dim=3, control_dim=2, latent_dim=8, hidden_dims=[32, 32])
    num_params = sum(p.numel() for p in vae.parameters())

    assert vae.state_dim == 3
    assert vae.control_dim == 2
    assert vae.latent_dim == 8
    assert num_params > 0

    print(f"  Parameters: {num_params:,}")
    print("  PASSED")


def test_vae_encode_shape():
    """#5: Encoder shape 검증"""
    print("\n" + "=" * 60)
    print("Test #5: VAE encode shape")

    vae = WorldModelVAE(state_dim=3, control_dim=2, latent_dim=8, hidden_dims=[32, 32])
    vae.eval()

    # Single
    x = torch.randn(3)
    mu, log_var = vae.encode(x)
    assert mu.shape == (8,), f"Expected (8,), got {mu.shape}"
    assert log_var.shape == (8,)

    # Batch
    x_batch = torch.randn(16, 3)
    mu_b, log_var_b = vae.encode(x_batch)
    assert mu_b.shape == (16, 8), f"Expected (16, 8), got {mu_b.shape}"
    assert log_var_b.shape == (16, 8)

    # log_var clamping
    assert torch.all(log_var_b >= -20.0)
    assert torch.all(log_var_b <= 2.0)

    print("  PASSED")


def test_vae_decode_shape():
    """#6: Decoder shape 검증"""
    print("\n" + "=" * 60)
    print("Test #6: VAE decode shape")

    vae = WorldModelVAE(state_dim=3, control_dim=2, latent_dim=8, hidden_dims=[32, 32])
    vae.eval()

    # Single
    z = torch.randn(8)
    x_pred = vae.decode(z)
    assert x_pred.shape == (3,), f"Expected (3,), got {x_pred.shape}"

    # Batch
    z_batch = torch.randn(16, 8)
    x_pred_b = vae.decode(z_batch)
    assert x_pred_b.shape == (16, 3), f"Expected (16, 3), got {x_pred_b.shape}"

    print("  PASSED")


def test_vae_latent_step_shape():
    """#7: Latent step shape 검증"""
    print("\n" + "=" * 60)
    print("Test #7: VAE latent_step shape")

    vae = WorldModelVAE(state_dim=3, control_dim=2, latent_dim=8, hidden_dims=[32, 32])
    vae.eval()

    z = torch.randn(16, 8)
    u = torch.randn(16, 2)
    z_next = vae.latent_step(z, u)
    assert z_next.shape == (16, 8), f"Expected (16, 8), got {z_next.shape}"

    print("  PASSED")


def test_vae_forward_full():
    """#8: 학습용 forward 출력 키/shape"""
    print("\n" + "=" * 60)
    print("Test #8: VAE forward full")

    vae = WorldModelVAE(state_dim=3, control_dim=2, latent_dim=8, hidden_dims=[32, 32])
    vae.train()

    state = torch.randn(16, 3)
    control = torch.randn(16, 2)
    next_state = torch.randn(16, 3)

    out = vae(state, control, next_state)

    expected_keys = {"recon", "mu", "log_var", "z", "z_next_pred", "z_next_target"}
    assert set(out.keys()) == expected_keys, f"Missing keys: {expected_keys - set(out.keys())}"

    assert out["recon"].shape == (16, 3)
    assert out["mu"].shape == (16, 8)
    assert out["log_var"].shape == (16, 8)
    assert out["z"].shape == (16, 8)
    assert out["z_next_pred"].shape == (16, 8)
    assert out["z_next_target"].shape == (16, 8)

    print("  PASSED")


def test_vae_reconstruction_untrained():
    """#9: 미학습 모델도 순환 가능 (NaN 없음)"""
    print("\n" + "=" * 60)
    print("Test #9: VAE 미학습 순환")

    vae = WorldModelVAE(state_dim=3, control_dim=2, latent_dim=8, hidden_dims=[32, 32])
    vae.eval()

    x = torch.randn(4, 3)
    with torch.no_grad():
        mu, log_var = vae.encode(x)
        z = vae.reparameterize(mu, log_var)
        x_recon = vae.decode(z)

    assert not torch.isnan(x_recon).any(), "NaN in reconstruction"
    assert not torch.isinf(x_recon).any(), "Inf in reconstruction"
    assert x_recon.shape == x.shape

    print("  PASSED")


# ══════════════════════════════════════════════════════════════
# WorldModelTrainer 테스트 (#10~#14)
# ══════════════════════════════════════════════════════════════

def test_trainer_construction():
    """#10: Trainer 생성"""
    print("\n" + "=" * 60)
    print("Test #10: WorldModelTrainer 생성")

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[32, 32],
        save_dir=tempfile.mkdtemp(),
    )

    assert trainer.state_dim == 3
    assert trainer.control_dim == 2
    assert trainer.latent_dim == 8
    assert trainer.model is not None

    summary = trainer.get_model_summary()
    assert "WorldModelVAE" in summary

    print(f"  {summary}")
    print("  PASSED")


def test_trainer_train_convergence():
    """#11: 학습 50 epochs, loss 감소"""
    print("\n" + "=" * 60)
    print("Test #11: WorldModelTrainer 학습 수렴")

    states, controls, next_states = _generate_training_data(n_samples=300)

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[32, 32],
        beta=0.001, alpha_dyn=1.0,
        save_dir=tempfile.mkdtemp(),
    )

    history = trainer.train(
        states, controls, next_states,
        epochs=50, batch_size=32, verbose=False,
    )

    assert len(history["train_loss"]) == 50
    # Loss should decrease
    first_5 = np.mean(history["train_loss"][:5])
    last_5 = np.mean(history["train_loss"][-5:])
    assert last_5 < first_5, f"Loss did not decrease: {first_5:.4f} → {last_5:.4f}"

    print(f"  Loss: {first_5:.4f} → {last_5:.4f}")
    print("  PASSED")


def test_trainer_predict():
    """#12: predict shape + 유효 값"""
    print("\n" + "=" * 60)
    print("Test #12: WorldModelTrainer predict")

    states, controls, next_states = _generate_training_data(n_samples=200)

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[32, 32],
        save_dir=tempfile.mkdtemp(),
    )
    trainer.train(states, controls, next_states, epochs=20, verbose=False)

    # Single prediction
    pred = trainer.predict(states[0], controls[0])
    assert pred.shape == (3,), f"Expected (3,), got {pred.shape}"
    assert not np.any(np.isnan(pred))

    # Batch prediction
    pred_batch = trainer.predict(states[:10], controls[:10])
    assert pred_batch.shape == (10, 3), f"Expected (10, 3), got {pred_batch.shape}"
    assert not np.any(np.isnan(pred_batch))

    print("  PASSED")


def test_trainer_save_load_roundtrip():
    """#13: 저장/로드 후 예측 일치"""
    print("\n" + "=" * 60)
    print("Test #13: WorldModelTrainer 저장/로드")

    save_dir = tempfile.mkdtemp()
    states, controls, next_states = _generate_training_data(n_samples=200)

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[32, 32],
        save_dir=save_dir,
    )
    trainer.train(states, controls, next_states, epochs=20, verbose=False)

    # 저장 전 예측
    pred_before = trainer.predict(states[0], controls[0])
    trainer.save_model("test_wm.pth")

    # 새 trainer로 로드
    trainer2 = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[32, 32],
        save_dir=save_dir,
    )
    trainer2.load_model("test_wm.pth")
    pred_after = trainer2.predict(states[0], controls[0])

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)

    # Cleanup
    shutil.rmtree(save_dir)
    print("  PASSED")


def test_trainer_encode_decode_roundtrip():
    """#14: 학습 후 재구성 오류 검증"""
    print("\n" + "=" * 60)
    print("Test #14: WorldModelTrainer encode/decode roundtrip")

    states, controls, next_states = _generate_training_data(n_samples=300)

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[64, 64],
        save_dir=tempfile.mkdtemp(),
    )
    trainer.train(states, controls, next_states, epochs=100, verbose=False)

    # Encode → Decode roundtrip
    test_state = states[0]
    z = trainer.encode(test_state)
    assert z.shape == (8,)

    recon = trainer.decode(z)
    assert recon.shape == (3,)

    # 학습 후 재구성 오류가 작아야 함
    error = np.linalg.norm(recon - test_state)
    assert error < 2.0, f"Reconstruction error too large: {error:.4f}"

    print(f"  Reconstruction error: {error:.4f}")
    print("  PASSED")


# ══════════════════════════════════════════════════════════════
# WorldModelDynamics 테스트 (#15~#18)
# ══════════════════════════════════════════════════════════════

def test_dynamics_properties():
    """#15: state_dim, control_dim, model_type"""
    print("\n" + "=" * 60)
    print("Test #15: WorldModelDynamics properties")

    dyn = WorldModelDynamics(state_dim=3, control_dim=2, latent_dim=8)
    assert dyn.state_dim == 3
    assert dyn.control_dim == 2
    assert dyn.model_type == "learned"
    assert dyn.latent_dim == 8

    info = dyn.get_model_info()
    assert info["loaded"] is False

    print("  PASSED")


def test_dynamics_step_single():
    """#16: step() 단일 입력"""
    print("\n" + "=" * 60)
    print("Test #16: WorldModelDynamics step single")

    # 학습된 모델 준비
    states, controls, next_states = _generate_training_data(n_samples=200)
    save_dir = tempfile.mkdtemp()

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[32, 32],
        save_dir=save_dir,
    )
    trainer.train(states, controls, next_states, epochs=20, verbose=False)
    trainer.save_model("wm_test.pth")

    # WorldModelDynamics 로드
    dyn = WorldModelDynamics(
        state_dim=3, control_dim=2, latent_dim=8,
        model_path=os.path.join(save_dir, "wm_test.pth"),
    )

    next_state = dyn.step(states[0], controls[0], dt=0.05)
    assert next_state.shape == (3,), f"Expected (3,), got {next_state.shape}"
    assert not np.any(np.isnan(next_state))

    shutil.rmtree(save_dir)
    print("  PASSED")


def test_dynamics_step_batch():
    """#17: step() 배치 입력"""
    print("\n" + "=" * 60)
    print("Test #17: WorldModelDynamics step batch")

    states, controls, next_states = _generate_training_data(n_samples=200)
    save_dir = tempfile.mkdtemp()

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[32, 32],
        save_dir=save_dir,
    )
    trainer.train(states, controls, next_states, epochs=20, verbose=False)
    trainer.save_model("wm_test.pth")

    dyn = WorldModelDynamics(
        state_dim=3, control_dim=2, latent_dim=8,
        model_path=os.path.join(save_dir, "wm_test.pth"),
    )

    next_batch = dyn.step(states[:10], controls[:10], dt=0.05)
    assert next_batch.shape == (10, 3), f"Expected (10, 3), got {next_batch.shape}"
    assert not np.any(np.isnan(next_batch))

    shutil.rmtree(save_dir)
    print("  PASSED")


def test_dynamics_encode_decode():
    """#18: encode/decode 직접 호출"""
    print("\n" + "=" * 60)
    print("Test #18: WorldModelDynamics encode/decode")

    states, controls, next_states = _generate_training_data(n_samples=200)
    save_dir = tempfile.mkdtemp()

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[32, 32],
        save_dir=save_dir,
    )
    trainer.train(states, controls, next_states, epochs=20, verbose=False)
    trainer.save_model("wm_test.pth")

    dyn = WorldModelDynamics(
        state_dim=3, control_dim=2, latent_dim=8,
        model_path=os.path.join(save_dir, "wm_test.pth"),
    )

    # Single
    z = dyn.encode(states[0])
    assert z.shape == (8,)
    x_recon = dyn.decode(z)
    assert x_recon.shape == (3,)

    # Batch
    z_batch = dyn.encode(states[:5])
    assert z_batch.shape == (5, 8)
    x_batch = dyn.decode(z_batch)
    assert x_batch.shape == (5, 3)

    # Latent dynamics
    z_next = dyn.latent_dynamics(z, controls[0])
    assert z_next.shape == (8,)

    shutil.rmtree(save_dir)
    print("  PASSED")


# ══════════════════════════════════════════════════════════════
# LatentMPPIController 테스트 (#19~#24)
# ══════════════════════════════════════════════════════════════

def test_controller_compute_control_shape():
    """#19: 반환 shape + info 키"""
    print("\n" + "=" * 60)
    print("Test #19: LatentMPPIController compute_control shape")

    model = _make_model()
    params = _make_latent_params()
    mock_wm = MockWorldModel(state_dim=3, latent_dim=8)

    controller = LatentMPPIController(model, params, world_model=mock_wm)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref(N=params.N, dt=params.dt)

    control, info = controller.compute_control(state, ref)

    # Shape
    assert control.shape == (2,), f"Expected (2,), got {control.shape}"

    # Info 키
    expected_keys = {
        "sample_trajectories", "sample_weights", "best_trajectory",
        "best_cost", "mean_cost", "temperature", "ess", "num_samples",
        "latent_trajectories", "latent_stats",
    }
    assert expected_keys.issubset(set(info.keys())), \
        f"Missing keys: {expected_keys - set(info.keys())}"

    # Shapes
    K, N = params.K, params.N
    assert info["sample_trajectories"].shape == (K, N + 1, 3)
    assert info["sample_weights"].shape == (K,)
    assert info["latent_trajectories"].shape == (K, N + 1, 8)

    print("  PASSED")


def test_controller_no_world_model_fallback():
    """#20: world_model=None → 표준 MPPI 폴백"""
    print("\n" + "=" * 60)
    print("Test #20: LatentMPPIController no world_model fallback")

    model = _make_model()
    params = _make_latent_params()

    controller = LatentMPPIController(model, params, world_model=None)
    assert controller.world_model is None

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref(N=params.N, dt=params.dt)

    control, info = controller.compute_control(state, ref)

    # 표준 MPPI 폴백 (latent_stats 키 없음)
    assert control.shape == (2,)
    assert "latent_trajectories" not in info

    print("  PASSED")


def test_controller_auto_detect_world_model():
    """#21: encode/decode가 있으면 자동 감지"""
    print("\n" + "=" * 60)
    print("Test #21: LatentMPPIController auto-detect")

    # WorldModelDynamics는 encode/decode/latent_dynamics 있음
    mock_wm = MockWorldModel(state_dim=3, latent_dim=8)

    # model 자리에 mock_wm 대신, 별도 model + world_model으로
    model = _make_model()
    params = _make_latent_params()

    # model에 encode/decode 없음 → world_model=None
    controller1 = LatentMPPIController(model, params)
    assert controller1.world_model is None

    # mock_wm는 encode/decode 있음 → model로 넣으면 자동 감지
    controller2 = LatentMPPIController(model, params, world_model=mock_wm)
    assert controller2.world_model is not None

    print("  PASSED")


def test_controller_latent_info_keys():
    """#22: latent_stats 키 검증"""
    print("\n" + "=" * 60)
    print("Test #22: LatentMPPIController latent_stats keys")

    model = _make_model()
    params = _make_latent_params()
    mock_wm = MockWorldModel(state_dim=3, latent_dim=8)

    controller = LatentMPPIController(model, params, world_model=mock_wm)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref(N=params.N, dt=params.dt)

    _, info = controller.compute_control(state, ref)

    ls = info["latent_stats"]
    assert "latent_dim" in ls
    assert "mean_latent_norm" in ls
    assert "max_latent_norm" in ls
    assert ls["latent_dim"] == 8

    # 누적 통계
    stats = controller.get_latent_statistics()
    assert stats["num_steps"] == 1

    print("  PASSED")


def test_controller_circle_tracking():
    """#23: 원형 궤적 추적"""
    print("\n" + "=" * 60)
    print("Test #23: LatentMPPIController 원형 궤적 추적")

    # VAE 학습
    states, controls, next_states = _generate_training_data(n_samples=500)

    trainer = WorldModelTrainer(
        state_dim=3, control_dim=2,
        latent_dim=8, hidden_dims=[64, 64],
        save_dir=tempfile.mkdtemp(),
    )
    trainer.train(states, controls, next_states, epochs=100, verbose=False)

    # WorldModelDynamics 생성 (직접 VAE 설정)
    wm_dyn = WorldModelDynamics(state_dim=3, control_dim=2, latent_dim=8)
    wm_dyn.set_vae(trainer.model)

    model = _make_model()
    params = _make_latent_params(K=128, N=15)

    controller = LatentMPPIController(model, params, world_model=wm_dyn)

    # 시뮬레이션
    state = np.array([3.0, 0.0, np.pi / 2])
    dt = params.dt
    trajectory = [state.copy()]

    for step_idx in range(50):
        t = step_idx * dt
        ref = generate_reference_trajectory(
            lambda t_val: circle_trajectory(t_val, radius=3.0),
            t, params.N, dt,
        )
        control, info = controller.compute_control(state, ref)
        state = model.step(state, control, dt)
        trajectory.append(state.copy())

    trajectory = np.array(trajectory)

    # 궤적이 원형을 따라가는지 (대략적 확인)
    final_dist = np.sqrt(trajectory[-1, 0] ** 2 + trajectory[-1, 1] ** 2)
    assert final_dist < 10.0, f"Robot too far: {final_dist:.2f}"
    assert not np.any(np.isnan(trajectory)), "NaN in trajectory"

    print(f"  Final position: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})")
    print(f"  Distance from origin: {final_dist:.2f}")
    print("  PASSED")


def test_controller_with_obstacle_cost():
    """#24: ObstacleCost + Latent rollout"""
    print("\n" + "=" * 60)
    print("Test #24: LatentMPPIController with ObstacleCost")

    model = _make_model()
    params = _make_latent_params(K=64, N=10)
    mock_wm = MockWorldModel(state_dim=3, latent_dim=8)

    obstacles = [(2.0, 2.0, 0.3), (-1.0, 1.0, 0.5)]
    cost = CompositeMPPICost(
        cost_functions=[
            StateTrackingCost(params.Q),
            TerminalCost(params.Qf),
            ControlEffortCost(params.R),
            ObstacleCost(obstacles, cost_weight=100.0),
        ]
    )

    controller = LatentMPPIController(
        model, params, cost_function=cost, world_model=mock_wm
    )

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref(N=params.N, dt=params.dt)

    control, info = controller.compute_control(state, ref)

    assert control.shape == (2,)
    assert info["mean_cost"] > 0

    print("  PASSED")


# ══════════════════════════════════════════════════════════════
# 통합/엣지 테스트 (#25~#28)
# ══════════════════════════════════════════════════════════════

def test_integration_reset():
    """#25: reset 후 초기화"""
    print("\n" + "=" * 60)
    print("Test #25: LatentMPPIController reset")

    model = _make_model()
    params = _make_latent_params()
    mock_wm = MockWorldModel(state_dim=3, latent_dim=8)

    controller = LatentMPPIController(model, params, world_model=mock_wm)

    state = np.array([0.0, 0.0, 0.0])
    ref = _make_ref(N=params.N, dt=params.dt)

    # 몇 스텝 실행
    for _ in range(3):
        controller.compute_control(state, ref)

    stats = controller.get_latent_statistics()
    assert stats["num_steps"] == 3

    # 리셋
    controller.reset()

    assert np.allclose(controller.U, 0.0)
    stats2 = controller.get_latent_statistics()
    assert stats2["num_steps"] == 0

    print("  PASSED")


def test_latent_vs_vanilla_comparison():
    """#26: 동일 시드 비교"""
    print("\n" + "=" * 60)
    print("Test #26: Latent vs Vanilla 비교")

    model = _make_model()
    params_base = MPPIParams(
        K=32, N=10, dt=0.05, lambda_=1.0,
        sigma=np.array([0.5, 0.5]),
        Q=np.array([10.0, 10.0, 1.0]),
        R=np.array([0.1, 0.1]),
    )
    params_latent = _make_latent_params()

    vanilla = MPPIController(model, params_base)
    mock_wm = MockWorldModel(state_dim=3, latent_dim=8)
    latent = LatentMPPIController(model, params_latent, world_model=mock_wm)

    state = np.array([1.0, 0.0, 0.0])
    ref = _make_ref(N=10, dt=0.05)

    np.random.seed(42)
    ctrl_v, info_v = vanilla.compute_control(state, ref)

    np.random.seed(42)
    ctrl_l, info_l = latent.compute_control(state, ref)

    # 둘 다 유효한 제어를 반환
    assert ctrl_v.shape == (2,)
    assert ctrl_l.shape == (2,)
    assert not np.any(np.isnan(ctrl_v))
    assert not np.any(np.isnan(ctrl_l))

    # 다른 모델을 사용하므로 결과가 다를 수 있음
    print(f"  Vanilla: {ctrl_v}")
    print(f"  Latent:  {ctrl_l}")
    print("  PASSED")


def test_numerical_stability():
    """#27: 극단 입력에서 NaN/Inf 없음"""
    print("\n" + "=" * 60)
    print("Test #27: 수치 안정성")

    model = _make_model()
    params = _make_latent_params(K=16, N=5)
    mock_wm = MockWorldModel(state_dim=3, latent_dim=8)

    controller = LatentMPPIController(model, params, world_model=mock_wm)

    # 극단적 상태
    extreme_states = [
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, -100.0, 0.0]),
        np.array([1e-10, 1e-10, 1e-10]),
        np.array([-50.0, 50.0, 3.14159]),
    ]

    for state in extreme_states:
        ref = _make_ref(N=params.N, dt=params.dt)
        control, info = controller.compute_control(state, ref)

        assert not np.any(np.isnan(control)), f"NaN for state={state}"
        assert not np.any(np.isinf(control)), f"Inf for state={state}"
        assert not np.any(np.isnan(info["sample_weights"]))
        controller.reset()

    print("  PASSED")


def test_different_latent_dims():
    """#28: latent_dim=4/8/32"""
    print("\n" + "=" * 60)
    print("Test #28: 다양한 latent_dim")

    model = _make_model()

    for ldim in [4, 8, 32]:
        params = _make_latent_params(latent_dim=ldim)
        mock_wm = MockWorldModel(state_dim=3, latent_dim=ldim)

        controller = LatentMPPIController(model, params, world_model=mock_wm)

        state = np.array([0.0, 0.0, 0.0])
        ref = _make_ref(N=params.N, dt=params.dt)

        control, info = controller.compute_control(state, ref)

        assert control.shape == (2,)
        assert info["latent_trajectories"].shape[2] == ldim
        assert info["latent_stats"]["latent_dim"] == ldim

        print(f"  latent_dim={ldim}: OK")

    print("  PASSED")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # LatentMPPIParams
        test_params_defaults,
        test_params_custom,
        test_params_validation,
        # WorldModelVAE
        test_vae_construction,
        test_vae_encode_shape,
        test_vae_decode_shape,
        test_vae_latent_step_shape,
        test_vae_forward_full,
        test_vae_reconstruction_untrained,
        # WorldModelTrainer
        test_trainer_construction,
        test_trainer_train_convergence,
        test_trainer_predict,
        test_trainer_save_load_roundtrip,
        test_trainer_encode_decode_roundtrip,
        # WorldModelDynamics
        test_dynamics_properties,
        test_dynamics_step_single,
        test_dynamics_step_batch,
        test_dynamics_encode_decode,
        # LatentMPPIController
        test_controller_compute_control_shape,
        test_controller_no_world_model_fallback,
        test_controller_auto_detect_world_model,
        test_controller_latent_info_keys,
        test_controller_circle_tracking,
        test_controller_with_obstacle_cost,
        # Integration
        test_integration_reset,
        test_latent_vs_vanilla_comparison,
        test_numerical_stability,
        test_different_latent_dims,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
