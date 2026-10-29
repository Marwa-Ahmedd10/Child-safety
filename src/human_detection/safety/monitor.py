"""The child-safety rules: children (or, in "anyone" zones, anybody; in "unaccompanied"
zones, children with no adult nearby) in hazard zones, children at an exit with no adult
nearby, and child limits per area."""
import math
import time
from datetime import datetime

from .zones import foot

# Checking means the classifier has not settled yet, so it may be a child.
POSSIBLE_CHILD = ('Child', 'Checking')


def children_text(count):
    return f'{count} child' if count == 1 else f'{count} children'


def adults_text(count):
    return f'{count} adult' if count == 1 else f'{count} adults'


def people_text(count):
    return f'{count} person' if count == 1 else f'{count} people'


class Debounce:
    """Turn per-frame conditions into start/end events: a condition must hold for
    `confirm` frames in a row to start, and be absent for `release` frames to end."""

    def __init__(self, confirm, release):
        self.confirm = confirm
        self.release = release
        self.streak = {}
        self.active = {}  # key -> [start time, missed frames, details]

    def update(self, conditions):
        """conditions: {key: details} true this frame; returns [(state, key, details, seconds)]."""
        changes = []
        for key, details in conditions.items():
            if key in self.active:
                self.active[key][1:] = [0, details]
                continue
            self.streak[key] = self.streak.get(key, 0) + 1
            if self.streak[key] >= self.confirm:
                del self.streak[key]
                self.active[key] = [time.monotonic(), 0, details]
                changes.append(('start', key, details, 0.0))
        self.streak = {key: n for key, n in self.streak.items() if key in conditions}
        for key, entry in list(self.active.items()):
            if key in conditions:
                continue
            entry[1] += 1
            if entry[1] >= self.release:
                del self.active[key]
                changes.append(('end', key, entry[2], time.monotonic() - entry[0]))
        return changes


class SafetyMonitor:
    """Alerts from zones. A person-based alert (a possible child in a hazard zone or alone
    at an exit, anyone in an "anyone" hazard zone) starts once one person has met its
    condition for `confirm_frames` frames in a row, so brief flashes, such as each new
    track ID being Checking for a moment, do not add up to an alert. Once on, it stays on
    while anyone meets the condition, so a child whose track ID changes keeps it on; it
    ends after `release_frames` frames with nobody. A zone's `after_seconds` adds a wait:
    the person must also have met the condition for that long. Count-zone alerts need their
    condition for `confirm_frames` frames in a row. Each zone raises at most one alert of
    each kind at a time."""

    def __init__(self, zones, confirm_frames=10, release_frames=30, accompany_distance=1.5, clock=datetime.now):
        self.zones = [zone for zone in zones if zone.type not in ('ignore', 'line')]
        self.clock = clock
        self.ignore_zones = [zone for zone in zones if zone.type == 'ignore']
        self.confirm = confirm_frames
        self.person_rules = Debounce(1, release_frames)  # people are confirmed one by one, below
        self.zone_rules = Debounce(confirm_frames, release_frames)
        self.streaks = {}  # (alert key, track_id) -> (frames in a row the person has met it, since when)
        self.accompany_distance = accompany_distance
        self.zone_counts = {}

    def ignored(self, box, size):
        """True for detections standing in an ignore zone, e.g. people on a TV or poster."""
        point = foot(box, size)
        return any(zone.contains(point) for zone in self.ignore_zones)

    def update(self, people, labels, size, timestamp=None):
        """people: {track_id: box}; labels: {track_id: label}; timestamp: the frame's time in
        seconds (the video's own clock for files), by default now. Returns new start/end events."""
        feet = {track_id: foot(box, size) for track_id, box in people.items()}
        person_conditions, zone_conditions, streaks = {}, {}, {}
        self.zone_counts = {}
        now = self.clock()
        timestamp = time.monotonic() if timestamp is None else timestamp

        def person_rule(key, matching, details, after_seconds=0):
            for track_id in matching:
                frames, since = self.streaks.get((key, track_id), (0, timestamp))
                streaks[key, track_id] = frames + 1, since
            if any(frames >= self.confirm and timestamp - since >= after_seconds
                   for frames, since in (streaks[key, track_id] for track_id in matching)) or (
                    matching and key in self.person_rules.active):
                person_conditions[key] = details

        for zone in self.zones:
            if not zone.active(now):
                continue
            inside = [track_id for track_id, point in feet.items() if zone.contains(point)]
            children = [track_id for track_id in inside if labels.get(track_id) in POSSIBLE_CHILD]
            # Count limits are not as urgent as a hazard or an exit, so they wait for labels to
            # settle: someone still Checking counts as neither a child nor an adult.
            confirmed = sum(labels.get(track_id) == 'Child' for track_id in inside)
            adults = sum(labels.get(track_id) == 'Adult' for track_id in inside)
            self.zone_counts[zone.name] = (confirmed, adults)
            alone = [track_id for track_id in children if not self.accompanied(track_id, people, labels)] \
                if zone.type == 'exit' or zone.unaccompanied else []
            if zone.type == 'hazard' and zone.anyone:
                person_rule(('person_in_hazard_zone', zone.name), inside, {
                    'people': len(inside), 'track_ids': inside,
                    'message': zone.alert or f'{people_text(len(inside))} in hazard zone "{zone.name}"'},
                    zone.after_seconds)
            elif zone.type == 'hazard' and zone.unaccompanied:
                person_rule(('child_alone_in_hazard_zone', zone.name), alone, {
                    'children': len(alone), 'track_ids': alone,
                    'message': zone.alert or f'{children_text(len(alone))} in hazard zone "{zone.name}" '
                                             'with no adult nearby'},
                    zone.after_seconds)
            elif zone.type == 'hazard':
                person_rule(('child_in_hazard_zone', zone.name), children, {
                    'children': len(children), 'track_ids': children,
                    'message': zone.alert or f'{children_text(len(children))} in hazard zone "{zone.name}"'},
                    zone.after_seconds)
            elif zone.type == 'exit':
                person_rule(('child_exiting_alone', zone.name), alone, {
                    'children': len(alone), 'track_ids': alone,
                    'message': zone.alert or f'{children_text(len(alone))} at exit "{zone.name}" with no adult nearby'},
                    zone.after_seconds)
            elif zone.type == 'count':
                if zone.max_children is not None and confirmed > zone.max_children:
                    zone_conditions['too_many_children', zone.name] = {
                        'children': confirmed, 'limit': zone.max_children,
                        'message': f'{children_text(confirmed)} in "{zone.name}" (limit {zone.max_children})'}
                if zone.max_children_per_adult is not None and confirmed > zone.max_children_per_adult * adults:
                    zone_conditions['too_few_adults', zone.name] = {
                        'children': confirmed, 'adults': adults,
                        'max_children_per_adult': zone.max_children_per_adult,
                        'message': f'{children_text(confirmed)} and {adults_text(adults)} in "{zone.name}" '
                                   f'(max {zone.max_children_per_adult:g} per adult)'}
        self.streaks = streaks
        changes = self.person_rules.update(person_conditions) + self.zone_rules.update(zone_conditions)
        return [self.event(*change) for change in changes]

    @staticmethod
    def event(state, key, details, seconds):
        event = {'event': key[0], 'state': state, 'zone': key[1], **details}
        if state == 'end':
            event['duration_s'] = round(seconds, 1)
        return event

    def finish(self):
        """End events for alerts still open, e.g. when the monitor stops, so no alert is
        left without an end."""
        events = []
        for rules in (self.person_rules, self.zone_rules):
            for key, (started, _, details) in rules.active.items():
                events.append({**self.event('end', key, details, time.monotonic() - started),
                               'reason': 'monitor stopped'})
            rules.active.clear()
        return events

    def accompanied(self, child, people, labels):
        """An adult stands within `accompany_distance` body heights of the child.
        Only confirmed adults count; someone still being checked does not."""
        x1, y1, x2, y2 = people[child]
        cx, cy = (x1 + x2) / 2, y2
        for track_id, (ax1, ay1, ax2, ay2) in people.items():
            if labels.get(track_id) != 'Adult':
                continue
            distance = math.hypot((ax1 + ax2) / 2 - cx, ay2 - cy)
            if distance <= self.accompany_distance * max(y2 - y1, ay2 - ay1):
                return True
        return False

    def active_messages(self):
        return [entry[2]['message'] for rules in (self.person_rules, self.zone_rules)
                for entry in rules.active.values()]
