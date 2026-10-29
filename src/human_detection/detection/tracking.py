"""Following people from frame to frame. ByteTrack, from the trackers package
(Apache-2.0), links the detector's boxes; a person keeps their ID when the tracker
briefly loses them, and stays counted through a few missed frames."""
import cv2

from .detector import PERSON, load_detector


def overlap(a, b):
    """Intersection of two boxes as a share of the smaller one."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return max(width, 0) * max(height, 0) / max(smaller, 1)


class PersonTracker:
    """Tracked people in each frame as {track_id: [x1, y1, x2, y2]}, from any detector
    with RF-DETR's predict(rgb_image, threshold) returning supervision Detections."""

    def __init__(self, model, conf, rejoin=2.0):
        from trackers import ByteTrackTracker
        self.model = model
        # Tracks start only from confident detections, but follow people through weaker
        # ones, for example when they are partly hidden.
        self.low = min(0.1, conf)
        self.tracker = ByteTrackTracker(track_activation_threshold=conf, high_conf_det_threshold=conf)
        self.rejoin = rejoin
        # IDs shown are numbered apart from the tracker's own, which it can revive after
        # one of them was handed to someone's new track; sharing numbers would merge two people.
        self.shown = {}  # tracker ID -> the ID shown for that person
        self.next_id = 1
        self.last = {}   # ID shown -> (timestamp, box) when last seen
        self.continued = []  # IDs kept this frame for someone the tracker had given a new ID

    @classmethod
    def load(cls, size, device, conf, models):
        """A tracker on an RF-DETR detector; its weights download into `models` if missing."""
        return cls(load_detector(size, device, models), conf)

    def update(self, frame, timestamp):
        """timestamp: the frame's time in seconds, so a lost person is kept for the same
        time however fast frames are processed."""
        detections = self.model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=self.low)
        tracked = self.tracker.update(detections[detections.class_id == PERSON], timestamp=timestamp)
        # Tracks seen in too few frames so far have no ID yet and are not counted.
        confirmed = tracked.tracker_id >= 0
        ids, boxes = tracked.tracker_id[confirmed].tolist(), tracked.xyxy[confirmed].astype(int).tolist()
        tracks = dict(zip(ids, boxes, strict=True))
        return self.stitch(tracks, timestamp)

    def stitch(self, tracks, timestamp):
        """Keep a person's ID when the tracker loses them and soon gives them a new one: a
        new track covering most of where someone was last seen, within `rejoin` seconds,
        is taken to be them, unless anyone else is near that spot. Without this, their
        count doubles while the old ID is held, and their labels start again. In a crowd it
        is ambiguous (on a mall video, about a third of such matches were someone else),
        so people there get new IDs as before."""
        present = {self.shown[raw] for raw in tracks if raw in self.shown}
        lost = {shown: box for shown, (seen, box) in self.last.items()
                if shown not in present and timestamp - seen <= self.rejoin}
        result, self.continued = {}, []
        for raw, box in tracks.items():
            if raw not in self.shown:
                others = [other for other_raw, other in tracks.items() if other_raw != raw]
                alone = {shown: spot for shown, spot in lost.items()
                         if not any(overlap(spot, other) > 0.1 for other in others)}
                match = max(alone, key=lambda shown: overlap(box, alone[shown]), default=None)
                if match is not None and overlap(box, alone[match]) >= 0.5:
                    del lost[match]
                    self.continued.append(match)
                    # The old tracker ID no longer stands for this person, even if revived.
                    self.shown = {other: shown for other, shown in self.shown.items() if shown != match}
                else:
                    match, self.next_id = self.next_id, self.next_id + 1
                self.shown[raw] = match
            result[self.shown[raw]] = box
            self.last[self.shown[raw]] = (timestamp, box)
        self.last = {shown: seen for shown, seen in self.last.items() if timestamp - seen[0] <= self.rejoin}
        self.shown = {raw: shown for raw, shown in self.shown.items() if raw in tracks or shown in self.last}
        return result


class Presence:
    """Tracked people to count. Someone seen in at least `hold` frames stays
    counted through up to `hold` missed frames, so one missed detection does
    not make the count flicker; brief false detections are not held."""

    def __init__(self, hold):
        self.hold = hold
        self.seen = {}

    def update(self, frame_number, tracks):
        """Record this frame's {track_id: box}; return {track_id: (last_frame, box)}."""
        for track_id, box in tracks.items():
            hits = self.seen[track_id][0] if track_id in self.seen else 0
            self.seen[track_id] = (hits + 1, frame_number, box)
        self.seen = {track_id: seen for track_id, seen in self.seen.items()
                     if frame_number - seen[1] <= self.hold}
        return {track_id: (last, box) for track_id, (hits, last, box) in self.seen.items()
                if last == frame_number or hits >= self.hold}
