"""
AnimationSaver 테스트 — 프레임 캡처, GIF 저장 (MP4는 ffmpeg 의존)
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mppi_controller.simulation.rendering.animation_saver import AnimationSaver


def test_gif_capture_and_save():
    """GIF 프레임 캡처 + 저장"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.gif")
        saver = AnimationSaver(path, fps=10)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for i in range(5):
            ax.clear()
            ax.plot([0, i], [0, i])
            saver.capture_frame(fig)

        assert saver.frame_count == 5
        saver.finalize()
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)


def test_gif_empty():
    """프레임 없이 finalize"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty.gif")
        saver = AnimationSaver(path, fps=10)
        saver.finalize()
        # 빈 GIF는 생성되지 않음
        assert not os.path.exists(path)


def test_is_available_gif():
    """GIF 가용성 확인"""
    result = AnimationSaver.is_available("gif")
    # PIL 설치되어 있으면 True
    try:
        from PIL import Image
        assert result is True
    except ImportError:
        assert result is False


def test_is_available_unknown():
    """알려지지 않은 포맷"""
    assert AnimationSaver.is_available("avi") is False


def test_repr():
    saver = AnimationSaver("test.mp4", fps=30)
    assert "test.mp4" in repr(saver)
    assert "30" in repr(saver)


def test_multiple_gifs():
    """여러 GIF 연속 저장"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        for idx in range(2):
            path = os.path.join(tmpdir, f"test_{idx}.gif")
            saver = AnimationSaver(path, fps=5)
            for i in range(3):
                ax.clear()
                ax.plot([0, i], [0, i * 2])
                saver.capture_frame(fig)
            saver.finalize()
            assert os.path.exists(path)

        plt.close(fig)
