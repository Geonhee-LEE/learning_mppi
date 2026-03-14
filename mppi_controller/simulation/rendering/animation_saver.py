"""
애니메이션 저장 — MP4 / GIF 내보내기.

FFMpegWriter (MP4)와 PillowWriter (GIF)를 통합하여
프레임 단위 캡처 + 인코딩을 지원한다.

Usage:
    saver = AnimationSaver("output.mp4", fps=20)
    for frame in frames:
        render(fig)
        saver.capture_frame(fig)
    saver.finalize()
"""

import os
from typing import Optional


class AnimationSaver:
    """
    프레임 캡처 + MP4/GIF 인코딩.

    Args:
        filename: 출력 파일 경로 (.mp4 또는 .gif)
        fps: 프레임 레이트
        dpi: 해상도
    """

    def __init__(self, filename: str, fps: int = 20, dpi: int = 100):
        self.filename = filename
        self.fps = fps
        self.dpi = dpi

        self._ext = os.path.splitext(filename)[1].lower()
        self._writer = None
        self._frames = []  # GIF용 PIL 이미지 리스트
        self._started = False

    @staticmethod
    def is_available(fmt: str) -> bool:
        """
        주어진 포맷이 사용 가능한지 확인.

        Args:
            fmt: "mp4" 또는 "gif"

        Returns:
            사용 가능 여부
        """
        fmt = fmt.lower().lstrip(".")
        if fmt == "mp4":
            try:
                import matplotlib.animation as manimation
                writer_class = manimation.writers["ffmpeg"]
                return True
            except (KeyError, RuntimeError):
                return False
        elif fmt == "gif":
            try:
                from PIL import Image
                return True
            except ImportError:
                return False
        return False

    def capture_frame(self, fig):
        """
        현재 Figure 상태를 프레임으로 캡처.

        Args:
            fig: matplotlib Figure
        """
        if self._ext == ".mp4":
            self._capture_mp4(fig)
        elif self._ext == ".gif":
            self._capture_gif(fig)

    def finalize(self):
        """캡처 완료 — 파일 인코딩 및 저장."""
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)

        if self._ext == ".mp4":
            self._finalize_mp4()
        elif self._ext == ".gif":
            self._finalize_gif()

        self._started = False

    @property
    def frame_count(self) -> int:
        """캡처된 프레임 수."""
        if self._ext == ".gif":
            return len(self._frames)
        return getattr(self, "_frame_count", 0)

    # ── MP4 (FFMpegWriter) ──────────────────────────────────────────────

    def _capture_mp4(self, fig):
        if not self._started:
            import matplotlib.animation as manimation
            FFMpegWriter = manimation.writers["ffmpeg"]
            self._writer = FFMpegWriter(fps=self.fps)
            self._writer.setup(fig, self.filename, dpi=self.dpi)
            self._started = True
            self._frame_count = 0

        self._writer.grab_frame()
        self._frame_count += 1

    def _finalize_mp4(self):
        if self._writer is not None:
            self._writer.finish()
            self._writer = None
            print(f"MP4 saved: {self.filename} ({self._frame_count} frames)")

    # ── GIF (Pillow) ────────────────────────────────────────────────────

    def _capture_gif(self, fig):
        import io
        from PIL import Image

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        self._frames.append(img)

    def _finalize_gif(self):
        if not self._frames:
            return

        duration_ms = int(1000 / self.fps)
        self._frames[0].save(
            self.filename,
            save_all=True,
            append_images=self._frames[1:],
            duration=duration_ms,
            loop=0,
        )
        n = len(self._frames)
        self._frames.clear()
        print(f"GIF saved: {self.filename} ({n} frames)")

    def __repr__(self) -> str:
        return f"AnimationSaver('{self.filename}', fps={self.fps})"
