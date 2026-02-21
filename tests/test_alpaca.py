"""
ALPaCA Dynamics 테스트

ALPaCADynamics의 Bayesian 적응, feature extraction, 불확실성,
ALPaCATrainer의 메타 학습/저장/로드를 검증.
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ==================== 헬퍼 함수 ====================


def _create_alpaca_model(state_dim=5, control_dim=2, feature_dim=32, n_iters=50):
    """테스트용 ALPaCA 모델 생성 (소규모 메타 학습)."""
    from mppi_controller.learning.alpaca_trainer import ALPaCATrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ALPaCATrainer(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dims=[64, 64],
            feature_dim=feature_dim,
            meta_lr=1e-3,
            task_batch_size=2,
            support_size=30,
            query_size=30,
            device="cpu",
            save_dir=tmpdir,
        )
        trainer.meta_train(n_iterations=n_iters, verbose=False)
        model_file = "test_alpaca.pth"
        trainer.save_meta_model(model_file)

        from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
        model = ALPaCADynamics(
            state_dim=state_dim,
            control_dim=control_dim,
            model_path=os.path.join(tmpdir, model_file),
            feature_dim=feature_dim,
        )
        return model, tmpdir, model_file


# ==================== ALPaCADynamics 테스트 ====================


def test_creation_no_model():
    """모델 없이 생성."""
    from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics

    alpaca = ALPaCADynamics(state_dim=5, control_dim=2)
    assert alpaca.state_dim == 5
    assert alpaca.control_dim == 2
    assert alpaca.model_type == "learned"
    assert alpaca.feature_extractor is None
    print("  PASS: test_creation_no_model")


def test_creation_with_model():
    """메타 학습된 모델로 생성."""
    model, _, _ = _create_alpaca_model(n_iters=10)
    assert model.feature_extractor is not None
    assert model.norm_stats is not None
    print("  PASS: test_creation_with_model")


def test_forward_dynamics_single():
    """단일 상태 forward dynamics."""
    model, _, _ = _create_alpaca_model(n_iters=10)
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    dot = model.forward_dynamics(state, control)
    assert dot.shape == (5,)
    assert not np.any(np.isnan(dot))
    print("  PASS: test_forward_dynamics_single")


def test_forward_dynamics_batch():
    """배치 forward dynamics."""
    model, _, _ = _create_alpaca_model(n_iters=10)
    states = np.random.randn(10, 5) * 0.5
    controls = np.random.randn(10, 2) * 0.5
    dots = model.forward_dynamics(states, controls)
    assert dots.shape == (10, 5)
    assert not np.any(np.isnan(dots))
    print("  PASS: test_forward_dynamics_batch")


def test_posterior_reset():
    """restore_prior()로 posterior가 prior로 리셋."""
    model, _, _ = _create_alpaca_model(n_iters=10)

    # prior 저장
    mu_0 = model._mu_0.copy()
    Lambda_0 = model._Lambda_0.copy()

    # 데이터로 adapt
    states = np.random.randn(20, 5) * 0.5
    controls = np.random.randn(20, 2) * 0.5
    next_states = states + np.random.randn(20, 5) * 0.01
    model.adapt(states, controls, next_states, dt=0.05, restore=False)

    # posterior가 변했는지 확인
    assert not np.allclose(model._mu_n, mu_0)

    # prior로 리셋
    model.restore_prior()
    assert np.allclose(model._mu_n, mu_0)
    assert np.allclose(model._Lambda_n, Lambda_0)
    print("  PASS: test_posterior_reset")


def test_adapt_no_gradient():
    """adapt()이 gradient 없이 동작하는지 확인 (closed-form)."""
    import torch
    model, _, _ = _create_alpaca_model(n_iters=10)

    states = np.random.randn(20, 5) * 0.5
    controls = np.random.randn(20, 2) * 0.5
    next_states = states + np.random.randn(20, 5) * 0.01

    # feature extractor 파라미터가 변하지 않아야 함
    params_before = {k: v.clone() for k, v in model.feature_extractor.state_dict().items()}

    model.adapt(states, controls, next_states, dt=0.05)

    params_after = model.feature_extractor.state_dict()
    for k in params_before:
        assert torch.allclose(params_before[k], params_after[k]), f"Feature extractor changed at {k}"

    print("  PASS: test_adapt_no_gradient")


def test_adapt_restore_true():
    """restore=True에서 매번 동일한 posterior."""
    model, _, _ = _create_alpaca_model(n_iters=10)

    states = np.random.randn(20, 5) * 0.5
    controls = np.random.randn(20, 2) * 0.5
    next_states = states + np.random.randn(20, 5) * 0.01

    model.adapt(states, controls, next_states, dt=0.05, restore=True)
    mu_1 = model._mu_n.copy()

    model.adapt(states, controls, next_states, dt=0.05, restore=True)
    mu_2 = model._mu_n.copy()

    assert np.allclose(mu_1, mu_2, atol=1e-6), "restore=True should give same result"
    print("  PASS: test_adapt_restore_true")


def test_adapt_restore_false():
    """restore=False에서 누적 posterior (다른 데이터)."""
    model, _, _ = _create_alpaca_model(n_iters=10)

    np.random.seed(42)
    states1 = np.random.randn(20, 5) * 0.5
    controls1 = np.random.randn(20, 2) * 0.5
    next_states1 = states1 + np.random.randn(20, 5) * 0.01

    model.adapt(states1, controls1, next_states1, dt=0.05, restore=False)
    mu_1 = model._mu_n.copy()

    # 다른 데이터로 두 번째 적응
    np.random.seed(99)
    states2 = np.random.randn(20, 5) * 0.5
    controls2 = np.random.randn(20, 2) * 0.5
    next_states2 = states2 + np.random.randn(20, 5) * 0.5

    model.adapt(states2, controls2, next_states2, dt=0.05, restore=False)
    mu_2 = model._mu_n.copy()

    # 두 번째 적응은 누적이므로 다른 결과
    assert not np.allclose(mu_1, mu_2), "restore=False should accumulate with new data"
    print("  PASS: test_adapt_restore_false")


def test_uncertainty_decreases():
    """데이터 추가 시 불확실성 감소."""
    model, _, _ = _create_alpaca_model(n_iters=10)

    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])

    # Prior 불확실성
    unc_before = model.get_uncertainty(state, control)
    assert unc_before.shape == (5,)

    # 데이터 적응
    states = np.random.randn(30, 5) * 0.5
    controls = np.random.randn(30, 2) * 0.5
    next_states = states + np.random.randn(30, 5) * 0.01
    model.adapt(states, controls, next_states, dt=0.05)

    # Posterior 불확실성
    unc_after = model.get_uncertainty(state, control)
    # 최소 하나의 차원에서 감소
    assert np.any(unc_after < unc_before + 1e-6), "Uncertainty should decrease with data"
    print(f"  PASS: test_uncertainty_decreases (max: {unc_before.max():.4f} → {unc_after.max():.4f})")


def test_uncertainty_batch():
    """배치 불확실성 계산."""
    model, _, _ = _create_alpaca_model(n_iters=10)

    states = np.random.randn(10, 5) * 0.5
    controls = np.random.randn(10, 2) * 0.5
    unc = model.get_uncertainty(states, controls)
    assert unc.shape == (10, 5)
    assert not np.any(np.isnan(unc))
    assert np.all(unc >= 0.0)
    print("  PASS: test_uncertainty_batch")


def test_residual_integration():
    """ResidualDynamics(DynamicKinematicAdapter + ALPaCA) 통합."""
    from mppi_controller.models.learned.residual_dynamics import ResidualDynamics
    from mppi_controller.models.kinematic.dynamic_kinematic_adapter import DynamicKinematicAdapter

    model, _, _ = _create_alpaca_model(n_iters=10)
    base = DynamicKinematicAdapter(c_v=0.1, c_omega=0.1)

    residual = ResidualDynamics(
        base_model=base,
        learned_model=model,
        use_residual=True,
    )

    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    dot = residual.forward_dynamics(state, control)
    assert dot.shape == (5,)
    assert not np.any(np.isnan(dot))
    print("  PASS: test_residual_integration")


def test_feature_extraction():
    """Feature extraction 출력 차원."""
    model, _, _ = _create_alpaca_model(feature_dim=32, n_iters=10)

    state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
    control = np.array([1.0, 0.5])
    phi = model._extract_features(state, control)
    assert phi.shape == (32,)

    states = np.random.randn(10, 5) * 0.5
    controls = np.random.randn(10, 2) * 0.5
    phi_batch = model._extract_features(states, controls)
    assert phi_batch.shape == (10, 32)
    print("  PASS: test_feature_extraction")


def test_repr():
    """__repr__ 출력."""
    model, _, _ = _create_alpaca_model(n_iters=10)
    s = repr(model)
    assert "ALPaCADynamics" in s
    assert "feature_dim" in s
    print(f"  PASS: test_repr → {s}")


def test_repr_no_model():
    """모델 없는 __repr__."""
    from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
    alpaca = ALPaCADynamics(state_dim=5, control_dim=2)
    s = repr(alpaca)
    assert "loaded=False" in s
    print(f"  PASS: test_repr_no_model → {s}")


def test_get_control_bounds():
    """get_control_bounds."""
    from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
    alpaca = ALPaCADynamics(state_dim=5, control_dim=2)
    lower, upper = alpaca.get_control_bounds()
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    print("  PASS: test_get_control_bounds")


def test_normalize_state():
    """normalize_state."""
    from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
    alpaca = ALPaCADynamics(state_dim=5, control_dim=2)
    state = np.array([0.0, 0.0, 4.0, 0.0, 0.0])
    normed = alpaca.normalize_state(state)
    assert abs(normed[2]) < np.pi + 0.01
    print("  PASS: test_normalize_state")


# ==================== ALPaCATrainer 테스트 ====================


def test_trainer_meta_train():
    """ALPaCATrainer 메타 학습 실행."""
    from mppi_controller.learning.alpaca_trainer import ALPaCATrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ALPaCATrainer(
            state_dim=5, control_dim=2,
            hidden_dims=[32, 32],
            feature_dim=16,
            meta_lr=1e-3,
            task_batch_size=2,
            support_size=20,
            query_size=20,
            device="cpu",
            save_dir=tmpdir,
        )
        trainer.meta_train(n_iterations=20, verbose=False)
        assert len(trainer.history["meta_loss"]) == 20
        assert trainer.norm_stats is not None
        print("  PASS: test_trainer_meta_train")


def test_trainer_loss_decreases():
    """메타 학습 loss 감소 확인."""
    from mppi_controller.learning.alpaca_trainer import ALPaCATrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ALPaCATrainer(
            state_dim=5, control_dim=2,
            hidden_dims=[64, 64],
            feature_dim=32,
            meta_lr=1e-3,
            task_batch_size=4,
            support_size=50,
            query_size=50,
            device="cpu",
            save_dir=tmpdir,
        )
        trainer.meta_train(n_iterations=100, verbose=False)
        losses = trainer.history["meta_loss"]
        first_10 = np.mean(losses[:10])
        last_10 = np.mean(losses[-10:])
        assert last_10 < first_10, f"Loss should decrease: {first_10:.4f} → {last_10:.4f}"
        print(f"  PASS: test_trainer_loss_decreases ({first_10:.4f} → {last_10:.4f})")


def test_trainer_save_load():
    """ALPaCA 모델 저장/로드."""
    from mppi_controller.learning.alpaca_trainer import ALPaCATrainer
    from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ALPaCATrainer(
            state_dim=5, control_dim=2,
            hidden_dims=[32, 32],
            feature_dim=16,
            meta_lr=1e-3,
            task_batch_size=2,
            support_size=20,
            query_size=20,
            device="cpu",
            save_dir=tmpdir,
        )
        trainer.meta_train(n_iterations=10, verbose=False)
        trainer.save_meta_model("test_save.pth")

        # 로드 확인
        model = ALPaCADynamics(
            state_dim=5, control_dim=2,
            model_path=os.path.join(tmpdir, "test_save.pth"),
            feature_dim=16,
        )
        assert model.feature_extractor is not None
        assert model.norm_stats is not None
        assert model._beta > 0

        # forward 동작 확인
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.1])
        control = np.array([1.0, 0.5])
        dot = model.forward_dynamics(state, control)
        assert dot.shape == (5,)
        print("  PASS: test_trainer_save_load")


def test_trainer_save_load_roundtrip():
    """저장/로드 후 동일한 예측."""
    from mppi_controller.learning.alpaca_trainer import ALPaCATrainer
    from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ALPaCATrainer(
            state_dim=5, control_dim=2,
            hidden_dims=[32, 32],
            feature_dim=16,
            task_batch_size=2,
            support_size=20,
            query_size=20,
            device="cpu",
            save_dir=tmpdir,
        )
        trainer.meta_train(n_iterations=10, verbose=False)
        trainer.save_meta_model("roundtrip.pth")

        model = ALPaCADynamics(
            state_dim=5, control_dim=2,
            model_path=os.path.join(tmpdir, "roundtrip.pth"),
            feature_dim=16,
        )

        # 동일 입력 → 동일 출력
        np.random.seed(42)
        state = np.random.randn(5) * 0.5
        control = np.random.randn(2) * 0.5
        dot1 = model.forward_dynamics(state, control)

        # 리로드
        model2 = ALPaCADynamics(
            state_dim=5, control_dim=2,
            model_path=os.path.join(tmpdir, "roundtrip.pth"),
            feature_dim=16,
        )
        dot2 = model2.forward_dynamics(state, control)
        assert np.allclose(dot1, dot2, atol=1e-5), "Save/load should preserve predictions"
        print("  PASS: test_trainer_save_load_roundtrip")


def test_adapt_mse_return():
    """adapt()이 MSE float 반환."""
    model, _, _ = _create_alpaca_model(n_iters=10)

    states = np.random.randn(20, 5) * 0.5
    controls = np.random.randn(20, 2) * 0.5
    next_states = states + np.random.randn(20, 5) * 0.01

    mse = model.adapt(states, controls, next_states, dt=0.05)
    assert isinstance(mse, float)
    assert mse >= 0.0
    print(f"  PASS: test_adapt_mse_return (MSE={mse:.6f})")


def test_state_to_dict_5d():
    """5D state_to_dict."""
    model, _, _ = _create_alpaca_model(n_iters=10)
    d = model.state_to_dict(np.array([1.0, 2.0, 0.5, 0.3, 0.1]))
    assert "x" in d and "v" in d
    print("  PASS: test_state_to_dict_5d")


def test_state_to_dict_3d():
    """3D state_to_dict."""
    from mppi_controller.models.learned.alpaca_dynamics import ALPaCADynamics
    alpaca = ALPaCADynamics(state_dim=3, control_dim=2)
    d = alpaca.state_to_dict(np.array([1.0, 2.0, 0.5]))
    assert "x" in d
    print("  PASS: test_state_to_dict_3d")


# ==================== 실행 ====================

if __name__ == "__main__":
    tests = [
        test_creation_no_model,
        test_creation_with_model,
        test_forward_dynamics_single,
        test_forward_dynamics_batch,
        test_posterior_reset,
        test_adapt_no_gradient,
        test_adapt_restore_true,
        test_adapt_restore_false,
        test_uncertainty_decreases,
        test_uncertainty_batch,
        test_residual_integration,
        test_feature_extraction,
        test_repr,
        test_repr_no_model,
        test_get_control_bounds,
        test_normalize_state,
        test_trainer_meta_train,
        test_trainer_loss_decreases,
        test_trainer_save_load,
        test_trainer_save_load_roundtrip,
        test_adapt_mse_return,
        test_state_to_dict_5d,
        test_state_to_dict_3d,
    ]

    print(f"\n{'=' * 60}")
    print(f"  ALPaCA Dynamics Tests ({len(tests)} tests)")
    print(f"{'=' * 60}")

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_fn.__name__} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_fn.__name__} — {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
