"""Checks for frame sources; run with `python -m unittest`."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from human_detection import camera


class FakeCapture:
    """Delivers `frames` frames, then fails every read, as a dropped stream does."""

    def __init__(self, frames):
        self.left = frames

    def read(self):
        if self.left is not None and self.left <= 0:
            return False, None
        if self.left is not None:
            self.left -= 1
        return True, np.zeros((8, 8, 3), np.uint8)

    def release(self):
        pass


class FailingCapture(FakeCapture):
    """Raises on its first read, as a broken driver might, then delivers frames."""

    def __init__(self):
        super().__init__(None)
        self.failed = False

    def read(self):
        if not self.failed:
            self.failed = True
            raise RuntimeError('driver error')
        return super().read()


def wait_until(condition, seconds=5):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
    return condition()


class VideoFileTest(unittest.TestCase):
    def test_reads_every_frame_then_ends(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'clip.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            for _ in range(5):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            source = camera.FrameSource(path, 'clip')
            times = [source.read()[1] for _ in range(5)]
            self.assertIsNone(source.read())
            self.assertTrue(source.ended)
            self.assertEqual(times, sorted(times))
            source.close()


class LiveSourceTest(unittest.TestCase):
    def test_goes_offline_when_frames_stop_and_reconnects(self):
        captures = [FakeCapture(3), FakeCapture(None)]
        with mock.patch.object(camera, 'open_capture', lambda source: captures.pop(0) if captures else None), \
                mock.patch.object(camera, 'STALL_SECONDS', 0.2):
            source = camera.FrameSource('rtsp://camera.example/stream', 'test')
            try:
                self.assertIsNotNone(source.read(timeout=2))
                self.assertTrue(wait_until(lambda: source.drops == 1 and source.online is True))
                self.assertIsNotNone(source.read(timeout=2))
            finally:
                source.close()

    def test_returns_only_new_frames(self):
        with mock.patch.object(camera, 'open_capture', lambda source: FakeCapture(1)), \
                mock.patch.object(camera, 'STALL_SECONDS', 60):
            source = camera.FrameSource('rtsp://camera.example/stream', 'test')
            try:
                self.assertIsNotNone(source.read(timeout=2))
                self.assertIsNone(source.read(timeout=0.2))  # no second frame: nothing is repeated
            finally:
                source.close()


class ReaderRobustnessTest(unittest.TestCase):
    def test_reader_survives_an_unexpected_error(self):
        captures = [FailingCapture(), FakeCapture(None)]
        with mock.patch.object(camera, 'open_capture', lambda source: captures.pop(0) if captures else None), \
                mock.patch.object(camera, 'MAX_WAIT', 0.1), mock.patch.object(camera.log, 'exception'):
            source = camera.FrameSource('rtsp://camera.example/stream', 'test')
            try:
                self.assertIsNotNone(source.read(timeout=3))
                self.assertFalse(source.down)
            finally:
                source.close()

    def test_stalled_camera_is_down(self):
        with mock.patch.object(camera, 'open_capture', lambda source: FakeCapture(0)), \
                mock.patch.object(camera, 'STALL_SECONDS', 0.2):
            source = camera.FrameSource('rtsp://camera.example/stream', 'test')
            try:
                self.assertTrue(wait_until(lambda: source.down))
            finally:
                source.close()

    def test_any_stream_url_is_live_and_redacted(self):
        self.assertTrue(camera.is_live('http://admin:S3cret@10.0.0.5/video.mjpg'))
        self.assertFalse(camera.is_live('/videos/pool.mp4'))
        self.assertEqual(camera.redact('http://admin:S3cret@10.0.0.5/video.mjpg'), 'http://10.0.0.5/video.mjpg')


if __name__ == '__main__':
    unittest.main()
