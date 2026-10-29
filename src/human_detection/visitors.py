"""Visitor counts at counting lines (zones of type "line"). A tracked person is counted
each time their feet cross a line: "in" when they cross to its right side, looking from
the line's first point to its second, and "out" the other way. Each crossing is counted
with the person's child/adult label."""
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone

# Feet closer to a line than this share of the person's height are on neither side, so
# someone standing or sitting on the line is not counted again and again as they sway.
BAND = 0.25
SETTLE_FRAMES = 30  # how long a crossing waits for the person's labels to settle
AGE = {'Child': 'children', 'Adult': 'adults'}
log = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class VisitorCounter:
    def __init__(self, lines):
        self.lines = lines
        self.keys = ['total', 'children', 'adults', 'age_unknown']
        self.sides = {}    # (line name, track_id) -> -1 or 1, the side a person was last seen on
        self.pending = []  # crossings waiting for labels: (line name, direction, track_id, frame)
        self.labels = {}   # track_id -> child label, as last seen
        self.totals = self.empty()
        self.period = self.empty()
        self.period_start = now()

    def empty(self):
        return {line.name: {direction: Counter(dict.fromkeys(self.keys, 0)) for direction in ('in', 'out')}
                for line in self.lines}

    def update(self, frame_number, people, labels, size):
        """people: {track_id: box}; labels: {track_id: label}. Returns the crossings
        counted this frame as (line name, direction, track_id, label)."""
        for track_id in people:
            self.labels[track_id] = labels.get(track_id)
        width, height = size
        for line in self.lines:
            (x1, y1), (x2, y2) = [(x * width, y * height) for x, y in line.polygon]
            length = math.hypot(x2 - x1, y2 - y1)
            for track_id, (left, top, right, bottom) in people.items():
                x, y = (left + right) / 2, bottom
                # Signed distance in pixels from the line, positive on the "in" side, and position along it.
                distance = ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / length
                along = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / length ** 2
                if abs(distance) < BAND * (bottom - top) or not 0 <= along <= 1:
                    continue
                side = 1 if distance > 0 else -1
                if self.sides.get((line.name, track_id), side) != side:
                    self.pending.append((line.name, 'in' if side > 0 else 'out', track_id, frame_number))
                self.sides[line.name, track_id] = side

        counted, waiting = [], []
        for crossing in self.pending:
            name, direction, track_id, frame = crossing
            label = self.labels[track_id]
            # Wait for the label to settle, but not for someone who has left or never settles.
            if label == 'Checking' and track_id in people and frame_number - frame < SETTLE_FRAMES:
                waiting.append(crossing)
                continue
            for counts in (self.totals, self.period):
                tally = counts[name][direction]
                tally['total'] += 1
                if label:
                    tally[AGE.get(label, 'age_unknown')] += 1
            counted.append((name, direction, track_id, label))
        self.pending = waiting
        # Forget people who have left, so memory does not grow without end.
        keep = set(people) | {crossing[2] for crossing in waiting}
        self.sides = {key: side for key, side in self.sides.items() if key[1] in people}
        self.labels = {track_id: label for track_id, label in self.labels.items() if track_id in keep}
        return counted

    def flush(self, camera, path=None):
        """Log this period's counts for each line, append them to `path` as JSON lines,
        and start a new period. A file that cannot be written is logged, not raised."""
        end = now()
        for name, directions in self.period.items():
            log.info('COUNT  %s: in %d, out %d from %s to %s', name, directions['in']['total'],
                     directions['out']['total'], self.period_start, end)
            if path:
                record = {'time': end, 'camera': camera, 'line': name, 'from': self.period_start, 'to': end,
                          'in': dict(directions['in']), 'out': dict(directions['out'])}
                try:
                    with open(path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record) + '\n')
                except OSError as error:
                    log.error('Cannot write the counts file: %s', error)
        self.period = self.empty()
        self.period_start = end
