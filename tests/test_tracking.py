"""Checks for person tracking on top of the detector; run with `python -m unittest`."""
import unittest

import numpy as np
import supervision as sv

from human_detection.detection.detector import PERSON
from human_detection.detection.tracking import PersonTracker, Presence

FRAME = np.zeros((480, 640, 3), np.uint8)
CAR = 3


class Detector:
    """Stands in for RF-DETR: returns the boxes it is given for the next frame."""

    def __init__(self):
        self.next = []

    def predict(self, image, threshold):
        kept = [(box, score, label) for box, score, label in self.next if score >= threshold]
        if not kept:
            return sv.Detections.empty()
        boxes, scores, labels = zip(*kept, strict=True)
        return sv.Detections(xyxy=np.array(boxes, float), confidence=np.array(scores), class_id=np.array(labels))


def track(frames, conf=0.4, fps=30):
    detector = Detector()
    tracker = PersonTracker(detector, conf)
    results = []
    for i, boxes in enumerate(frames):
        detector.next = boxes
        results.append(tracker.update(FRAME, i / fps))
    return results


class PersonTrackerTest(unittest.TestCase):
    def test_tracks_people_only(self):
        person, car = ([100, 100, 200, 400], 0.9, PERSON), ([300, 200, 500, 300], 0.9, CAR)
        results = track([[person, car]] * 3)
        self.assertEqual([list(result.values()) for result in results[1:]], [[[100, 100, 200, 400]]] * 2)

    def test_keeps_id_while_moving_and_through_weak_detections(self):
        frames = [[([100 + 5 * i, 100, 200 + 5 * i, 400], 0.9 if i < 5 else 0.2, PERSON)] for i in range(10)]
        ids = {track_id for result in track(frames)[1:] for track_id in result}
        self.assertEqual(len(ids), 1)

    def test_weak_detection_alone_does_not_start_a_track(self):
        self.assertEqual(track([[([100, 100, 200, 400], 0.2, PERSON)]] * 5)[-1], {})


class StitchTest(unittest.TestCase):
    """The tracker's IDs go in; the IDs shown come out."""
    PERSON_BOX = [100, 100, 200, 400]

    def setUp(self):
        self.tracker = PersonTracker(Detector(), 0.4, rejoin=2.0)
        self.tracker.stitch({1: self.PERSON_BOX}, 0.0)

    def test_new_id_where_someone_was_just_lost_keeps_their_id(self):
        self.assertEqual(self.tracker.stitch({}, 0.5), {})
        self.assertEqual(self.tracker.stitch({7: [105, 110, 205, 400]}, 1.0), {1: [105, 110, 205, 400]})
        self.assertEqual(self.tracker.continued, [1])
        self.assertEqual(self.tracker.stitch({7: [110, 110, 210, 400]}, 1.1), {1: [110, 110, 210, 400]})
        self.assertEqual(self.tracker.continued, [])

    def test_someone_elsewhere_or_later_gets_their_own_id(self):
        self.assertEqual(self.tracker.stitch({7: [600, 100, 700, 400]}, 1.0), {2: [600, 100, 700, 400]})
        self.assertEqual(self.tracker.stitch({8: self.PERSON_BOX}, 3.5), {3: self.PERSON_BOX})

    def test_not_when_someone_else_is_near_the_spot(self):
        neighbour = [180, 100, 280, 400]
        result = self.tracker.stitch({7: [105, 110, 205, 400], 9: neighbour}, 1.0)
        self.assertEqual(set(result), {2, 3})

    def test_second_box_on_a_person_still_tracked_is_not_merged(self):
        result = self.tracker.stitch({1: self.PERSON_BOX, 7: [110, 110, 190, 250]}, 0.1)
        self.assertEqual(set(result), {1, 2})

    def test_revived_tracker_id_does_not_merge_two_people(self):
        self.tracker.stitch({}, 0.5)
        self.assertEqual(self.tracker.stitch({7: [105, 110, 205, 400]}, 1.0), {1: [105, 110, 205, 400]})
        # The tracker revives its old ID 1 on someone elsewhere: they must stay a separate person.
        result = self.tracker.stitch({7: [105, 110, 205, 400], 1: [600, 100, 700, 400]}, 1.1)
        self.assertEqual(result, {1: [105, 110, 205, 400], 2: [600, 100, 700, 400]})


class PresenceTest(unittest.TestCase):
    def test_briefly_missed_person_stays_counted(self):
        presence = Presence(hold=2)
        presence.update(1, {7: (0, 0, 10, 10)})
        presence.update(2, {7: (0, 0, 10, 10)})
        self.assertEqual(len(presence.update(3, {})), 1)
        self.assertEqual(len(presence.update(4, {})), 1)
        self.assertEqual(len(presence.update(5, {})), 0)

    def test_brief_false_detection_is_not_held(self):
        presence = Presence(hold=3)
        self.assertEqual(len(presence.update(1, {9: (0, 0, 10, 10)})), 1)
        self.assertEqual(len(presence.update(2, {})), 0)

    def test_intermittent_person_becomes_held(self):
        presence = Presence(hold=2)
        presence.update(1, {7: (0, 0, 10, 10)})
        presence.update(3, {7: (0, 0, 10, 10)})
        self.assertEqual(len(presence.update(4, {})), 1)

    def test_zero_hold_counts_current_frame_only(self):
        presence = Presence(hold=0)
        presence.update(1, {7: (0, 0, 10, 10)})
        self.assertEqual(len(presence.update(2, {})), 0)


if __name__ == '__main__':
    unittest.main()
