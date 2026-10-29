"""Per-person readings over time, so labels settle instead of flickering."""
from collections import deque


class TrackVotes:
    """Recent readings for each tracked person, sampled every `refresh` frames."""

    # Whether readings carry over when the tracker briefly loses someone and continues
    # them under their old ID; that person may in fact be someone else.
    carry_over = True

    def __init__(self, window, refresh):
        self.window = window
        self.refresh = refresh
        self.history = {}
        self.next_try = {}

    def due(self, track_id, frame_number):
        # Read a person on first sight, then every `refresh` frames, even when the
        # last attempt found nothing, so hard cases do not cost work every frame.
        return frame_number >= self.next_try.get(track_id, 0)

    def record(self, track_id, frame_number, reading=None):
        """Note an attempt for this person; reading is None when nothing was measured."""
        self.next_try[track_id] = frame_number + self.refresh
        if reading is not None:
            self.history.setdefault(track_id, deque(maxlen=self.window)).append(reading)

    def readings(self, track_id):
        return self.history.get(track_id, ())

    def confidence(self, track_id):
        """How sure this person's label is, from 0 to 1; None when it has no such measure."""
        return None

    def restart(self, track_id):
        """Drop this person's readings, so their label is worked out again."""
        self.history.pop(track_id, None)
        self.next_try.pop(track_id, None)

    def forget(self, keep):
        keep = set(keep)
        self.history = {t: votes for t, votes in self.history.items() if t in keep}
        self.next_try = {t: frame for t, frame in self.next_try.items() if t in keep}


class Labeller:
    """Labels tracked people frame by frame: each visible person is read when a reading
    is due, and everyone counted is labelled from their recent readings."""

    def __init__(self, kind, names, read, votes, models=None):
        self.kind = kind    # 'child' or 'age'
        self.names = names  # every label it gives, in the order counts are shown
        # read(frame, boxes, wanted) returns {index: reading} for the wanted box indices.
        self.read = read
        self.votes = votes  # TrackVotes with add() and label()
        self.models = models or {}  # {name: version} of the models behind the labels

    def update(self, frame, frame_number, tracks, people, continued):
        """tracks: {track_id: box} visible this frame; people: the IDs counted, including
        anyone briefly missed; continued: IDs the tracker kept for someone it had lost.
        Returns {track_id: label} for everyone counted."""
        votes = self.votes
        if not votes.carry_over:
            for track_id in continued:
                votes.restart(track_id)
        visible = list(tracks)
        wanted = [i for i, track_id in enumerate(visible) if votes.due(track_id, frame_number)]
        if wanted:
            readings = self.read(frame, list(tracks.values()), wanted)
            for i in wanted:
                votes.add(visible[i], frame_number, readings.get(i))
        votes.forget(people)
        return {track_id: votes.label(track_id) for track_id in people}

    def confidence(self, track_id):
        return self.votes.confidence(track_id)
