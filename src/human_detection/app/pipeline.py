"""One camera's monitoring loop. Each frame, people are tracked, counted, labelled and
checked against the zones; alerts, visitor counts, the status file and the preview window
follow from that."""
import logging
import signal
import time
from collections import Counter
from pathlib import Path

from ..camera import FrameSource, redact
from ..detection.tracking import PersonTracker, Presence
from ..labels.child import child_labeller
from ..labels.face_age import age_labeller
from ..runtime import MODELS, pick_device
from ..safety.events import EventLog
from ..safety.monitor import SafetyMonitor
from ..safety.zones import load_zones
from ..visitors import VisitorCounter
from . import overlay
from .health import CameraHealth

log = logging.getLogger(__name__)


def camera_name(source):
    """Name a camera by its source, without the user name and password a stream URL may hold."""
    return redact(source.strip())


class CameraApp:
    """Everything one camera needs, set up from the live-camera options; run() monitors it."""

    def __init__(self, args):
        self.args = args
        for output in (args.events, args.counts, args.status):
            if output:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
        self.name = args.camera_name or camera_name(args.source)
        self.events = EventLog(self.name, args.events, args.webhook)
        # Load zones first so a mistake in the file fails before the slow model loads.
        self.zones, self.monitor, self.visitors = [], None, None
        if args.zones:
            self.zones, settings = load_zones(args.zones)
            self.monitor = SafetyMonitor(self.zones, **settings)
            lines = [zone for zone in self.zones if zone.type == 'line']
            if lines:
                self.visitors = VisitorCounter(lines)
                self.next_count = time.monotonic() + args.count_every * 60

        device = pick_device(args.device)
        MODELS.mkdir(exist_ok=True)
        self.tracker = PersonTracker.load(args.model, device, args.conf, MODELS)
        # Every alert names the models behind it, so it can be traced to the model card's numbers.
        self.models = {'detector': f'rf-detr-{args.model}'}
        self.labeller = None
        if args.age:
            self.labeller = age_labeller(MODELS, args.age_conf)
        # Zones that alert on anyone, ignore people or count crossings work without child labels,
        # and skipping the classifier roughly doubles the frame rate on a laptop.
        elif args.child or any(zone.needs_labels() for zone in self.zones):
            self.labeller = child_labeller(MODELS / 'child_head.npz', device, args.child_recall)
        if self.labeller:
            self.models.update(self.labeller.models)
        self.events.models = self.models
        log.info('Models: %s', ', '.join(f'{key} {value}' for key, value in self.models.items()))

        self.presence = Presence(args.hold)
        self.health = CameraHealth(self.name, self.events, args.status)
        self.window = None if args.no_window else overlay.Window('Room Counter - Q to quit')
        self.frame_number, self.count, self.fps = 0, 0, 0.0
        self.breakdown, self.last_summary, self.preview = '', None, None

    def run(self):
        """Monitor until the video ends, the window is closed, --max-frames is reached or
        the process is stopped; then close open alerts and save the counts."""
        self.source = FrameSource(self.args.source, self.name)
        try:
            while True:
                start = time.perf_counter()
                got = self.source.read()
                self.health.check(self.source, self.figures)
                if got is None:
                    if self.source.ended or not self.show_offline():
                        break
                    continue
                frame, timestamp = got
                people, labels = self.process(frame, timestamp)
                elapsed = max(time.perf_counter() - start, 1e-6)
                self.fps = 1 / elapsed if not self.fps else 0.9 * self.fps + 0.1 / elapsed
                if self.window and not self.show(frame, people, labels):
                    break
                if self.args.max_frames and self.frame_number >= self.args.max_frames:
                    break
        finally:
            self.close()

    def process(self, frame, timestamp):
        """Track, count, label and check one frame. Returns the people counted, as
        {track_id: (last frame seen, box)}, and their labels."""
        tracks = self.tracker.update(frame, timestamp)
        self.frame_number += 1
        size = (frame.shape[1], frame.shape[0])
        if self.monitor:
            tracks = {track_id: box for track_id, box in tracks.items() if not self.monitor.ignored(box, size)}
        people = self.presence.update(self.frame_number, tracks)
        self.count = len(people)
        labels = {}
        if self.labeller:
            labels = self.labeller.update(frame, self.frame_number, tracks, people, self.tracker.continued)
        boxes = {track_id: box for track_id, (_, box) in people.items()}
        if self.monitor:
            for event in self.monitor.update(boxes, labels, size, timestamp):
                self.events.emit(event)
        if self.visitors:
            self.count_visitors(boxes, labels, size)
        self.log_counts(labels)
        return people, labels

    def count_visitors(self, boxes, labels, size):
        for line, direction, track_id, label in self.visitors.update(self.frame_number, boxes, labels, size):
            log.info('%-5s %s: #%s %s', direction.upper(), line, track_id, label or '')
        if time.monotonic() >= self.next_count:
            self.visitors.flush(self.name, self.args.counts)
            self.next_count += self.args.count_every * 60

    def log_counts(self, labels):
        """Log the counts whenever they change."""
        summary = f'Visible people: {self.count}'
        if self.labeller:
            totals = Counter(labels.values())
            self.breakdown = '  '.join(f'{label}: {totals[label]}' for label in self.labeller.names)
            summary += f' | {self.breakdown}'
        if summary != self.last_summary:
            log.info(summary)
            self.last_summary = summary

    def show(self, frame, people, labels):
        """Draw and show the preview; False once the person quits."""
        self.preview = frame.copy()
        if self.monitor:
            overlay.draw_zones(self.preview, self.zones, self.monitor.zone_counts,
                               self.visitors.totals if self.visitors else None)
        colors, scores = {}, {}
        if self.labeller:
            colors = overlay.LABEL_COLORS[self.labeller.kind]
            scores = {track_id: self.labeller.confidence(track_id) for track_id in people}
        overlay.draw_people(self.preview, people, labels, colors, self.frame_number, scores)
        if self.labeller:
            overlay.draw_header(self.preview, f'People: {self.count} | {self.breakdown} | {self.fps:.1f} FPS')
        else:
            overlay.draw_header(self.preview, f'Visible people: {self.count} | {self.fps:.1f} FPS')
        if self.monitor:
            overlay.draw_alerts(self.preview, self.monitor.active_messages())
        return self.window.show(self.preview)

    def show_offline(self):
        """While no frames come, show the last preview under an offline banner; False once
        the person quits."""
        if not self.window or self.preview is None:
            return True
        return self.window.show(overlay.offline_view(self.preview))

    def figures(self):
        """The app's own numbers for the status file."""
        return {'fps': round(self.fps, 1), 'frames': self.frame_number, 'people': self.count,
                'models': self.models, 'alerts': self.monitor.active_messages() if self.monitor else []}

    def close(self):
        # A second stop signal (Ctrl-C reaches every process started from a terminal, and
        # systemd signals every process of a service) must not cut this cleanup short.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        self.source.close()
        if self.window:
            self.window.close()
        if self.visitors:
            self.visitors.flush(self.name, self.args.counts)
        if self.monitor:
            for event in self.monitor.finish():
                self.events.emit(event)
        self.health.finish()
        self.events.close()
