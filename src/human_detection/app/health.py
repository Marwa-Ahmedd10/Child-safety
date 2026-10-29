"""Camera health: an alert while the camera is offline, since nobody is being monitored
then, and the --status file for monitoring tools."""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

STATUS_SECONDS = 5  # how often --status is rewritten
log = logging.getLogger(__name__)


class CameraHealth:
    def __init__(self, name, events, status_path=None):
        self.name = name
        self.events = events
        self.status_path = status_path
        self.offline_since = None  # when the open camera-offline alert started
        self.next_status = time.monotonic()

    def check(self, source, figures):
        """Call on every pass of the loop, whether a frame came or not. figures() returns
        the app's own numbers for the status file."""
        # While a camera is down nobody is being watched, so that is an alert too. Its time
        # runs from the last frame that arrived.
        if source.down and self.offline_since is None:
            self.offline_since = source.last_frame_at
            self.events.emit({'event': 'camera_offline', 'state': 'start', 'zone': None,
                              'message': f'Camera "{self.name}" is offline: nobody is being monitored'})
        elif not source.down and self.offline_since is not None:
            self.events.emit({'event': 'camera_offline', 'state': 'end', 'zone': None,
                              'message': f'Camera "{self.name}" is back online',
                              'duration_s': round(source.last_frame_at - self.offline_since, 1)})
            self.offline_since = None
        if self.status_path and time.monotonic() >= self.next_status:
            write_status(self.status_path, {
                'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'camera': self.name,
                'online': not source.down, 'drops': source.drops, **figures()})
            self.next_status = time.monotonic() + STATUS_SECONDS

    def finish(self):
        """End an offline alert still open when monitoring stops, so no alert is left without an end."""
        if self.offline_since is not None:
            self.events.emit({'event': 'camera_offline', 'state': 'end', 'zone': None, 'reason': 'monitor stopped',
                              'message': f'Camera "{self.name}" monitoring stopped while offline',
                              'duration_s': round(time.monotonic() - self.offline_since, 1)})


def write_status(path, status):
    """Replace the status file whole, so a reader never sees half of it. A failure is
    logged, not raised: a full disk must not stop the monitoring."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(status) + '\n')
        temporary.replace(path)
    except OSError as error:
        log.warning('Cannot write the status file: %s', error)
