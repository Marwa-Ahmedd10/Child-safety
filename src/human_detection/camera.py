"""Frames from a webcam, video file or network camera (RTSP, HTTP or any other stream URL).

A live source is read on its own thread and only its newest frame is kept, so processing
always works on what the camera sees now instead of falling behind. When a live source
stops delivering frames it is reopened, with waits growing from 1 to 30 seconds. `down`
tells the monitor when nobody is being watched: the reader found the stream broken, the
reader thread itself failed, or no frame has arrived for STALL_SECONDS (a read that hangs).
"""
import logging
import re
import sys
import threading
import time

import cv2

log = logging.getLogger(__name__)
STALL_SECONDS = 5    # a live source with no new frame for this long is offline
MAX_WAIT = 30        # longest wait between attempts to reopen a live source


def redact(source):
    """A source without the user name and password a stream URL may hold."""
    return re.sub(r'(?<=://)[^/]*@', '', str(source))


def is_url(source):
    return isinstance(source, str) and re.match(r'^[a-z][a-z0-9+.-]*://', source, re.IGNORECASE) is not None


def is_live(source):
    return isinstance(source, int) or is_url(source)


def open_capture(source):
    cap = cv2.VideoCapture()
    if isinstance(source, int) and sys.platform == 'darwin':
        opened = cap.open(source, cv2.CAP_AVFOUNDATION)
    elif is_url(source):
        # FFmpeg only, with timeouts: other backends can hang, and some print the full URL.
        opened = cap.open(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                                                   cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
    else:
        opened = cap.open(source)
    if opened and isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap if opened else None


class FrameSource:
    """read() returns (frame, timestamp in seconds), or None: for a live source when no
    new frame arrived within `timeout`, for a video file when it has ended."""

    def __init__(self, source, name):
        source = source.strip() if isinstance(source, str) else source
        self.source = int(source) if str(source).isdecimal() else source
        self.name = name
        self.live = is_live(self.source)
        self.online = None  # the reader's view: unknown until the first frame or failure
        self.ended = False
        self.drops = 0  # times the live source went offline after delivering frames
        self.last_frame_at = time.monotonic()  # when the newest frame arrived, or the source was opened
        if self.live:
            self._latest = None  # (frame, timestamp, sequence number)
            self._seen = 0
            self._ready = threading.Condition()
            self._stop = threading.Event()
            cap = open_capture(self.source)
            # A webcam that cannot open at start is a setup problem (index or permission);
            # a network camera may simply not be up yet, so it is retried like a dropped one.
            if cap is None and isinstance(self.source, int):
                raise RuntimeError('Cannot open camera. Check its index and macOS Camera permission '
                                   'for your terminal/app.')
            self._thread = threading.Thread(target=self._run, args=(cap,), name=f'frames-{name}', daemon=True)
            self._thread.start()
        else:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f'Cannot open video {redact(self.source)}')
            self.online = True

    @property
    def down(self):
        """Whether nobody is being watched: the reader found the stream broken, its thread
        stopped, or no frame has arrived for STALL_SECONDS. Video files are never down."""
        if not self.live:
            return False
        return (self.online is False or not self._thread.is_alive()
                or time.monotonic() - self.last_frame_at > STALL_SECONDS)

    def read(self, timeout=1.0):
        if not self.live:
            ok, frame = self._cap.read()
            if not ok:
                self.ended = True
                return None
            # A video file is timed by its own clock, so it tracks the same however fast it is processed.
            return frame, self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        with self._ready:
            self._ready.wait_for(lambda: self._latest and self._latest[2] > self._seen or self._stop.is_set(),
                                 timeout=timeout)
            if not self._latest or self._latest[2] <= self._seen:
                return None
            frame, timestamp, self._seen = self._latest
            return frame, timestamp

    def _set_online(self, online):
        if online != self.online:
            if online and self.online is None:
                log.info('Camera %s online', self.name)
            else:
                log.warning('Camera %s %s', self.name, 'back online' if online else 'offline')
            if not online and self.online:
                self.drops += 1
            self.online = online

    def _run(self, cap):
        wait, sequence, opened_at = 1, 0, time.monotonic()
        while not self._stop.is_set():
            try:
                if cap is None:
                    cap = open_capture(self.source)
                    if cap is None:
                        self._set_online(False)
                        log.warning('Camera %s: cannot open the stream; retrying in %d s', self.name, wait)
                        self._stop.wait(wait)
                        wait = min(wait * 2, MAX_WAIT)
                        continue
                    opened_at = time.monotonic()
                ok, frame = cap.read()
                now = time.monotonic()
                if ok:
                    wait = 1
                    sequence += 1
                    self._set_online(True)
                    with self._ready:
                        self._latest = (frame, now, sequence)
                        self.last_frame_at = now
                        self._ready.notify_all()
                elif now - max(self.last_frame_at, opened_at) > STALL_SECONDS:
                    self._set_online(False)
                    cap.release()
                    cap = None
                else:
                    time.sleep(0.05)
            except Exception:
                # Keep watching: an unexpected error is treated like a dropped stream.
                log.exception('Camera %s: reader failed; reopening in %d s', self.name, wait)
                self._set_online(False)
                if cap is not None:
                    cap.release()
                cap = None
                self._stop.wait(wait)
                wait = min(wait * 2, MAX_WAIT)
        if cap is not None:
            cap.release()

    def close(self):
        if self.live:
            self._stop.set()
            with self._ready:
                self._ready.notify_all()
            self._thread.join(timeout=5)
        else:
            self._cap.release()
