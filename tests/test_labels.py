"""Checks for child and age labels and how they settle; run with `python -m unittest`."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from human_detection.labels import child
from human_detection.labels.child import LABELS, ChildVotes, recall_threshold, square
from human_detection.labels.face_age import AgeVotes, age_group, owner
from human_detection.labels.votes import Labeller


def bands(**scores):
    names = ('b0_2', 'b4_6', 'b8_12', 'b15_20', 'b25_32', 'b38_43', 'b48_53', 'b60_100')
    return [scores.get(name, 0.0) for name in names]


class AgeGroupTest(unittest.TestCase):
    def test_split_adult_bands_count_as_adult(self):
        self.assertEqual(age_group(bands(b25_32=0.35, b38_43=0.35, b60_100=0.3), 0.6), 'Adult')

    def test_teen_band_stays_unknown(self):
        self.assertEqual(age_group(bands(b15_20=0.7, b25_32=0.3), 0.6), 'Unknown')

    def test_low_score_stays_unknown(self):
        self.assertEqual(age_group(bands(b8_12=0.5, b25_32=0.5), 0.6), 'Unknown')

    def test_invalid_scores_stay_unknown(self):
        self.assertEqual(age_group([1.0] * 7, 0.6), 'Unknown')
        self.assertEqual(age_group(bands(b0_2=np.nan), 0.6), 'Unknown')


class OwnerTest(unittest.TestCase):
    def test_face_in_head_region(self):
        self.assertEqual(owner((40, 10, 60, 30), [(0, 0, 100, 200), (300, 0, 400, 200)]), 0)

    def test_face_below_head_region(self):
        self.assertIsNone(owner((40, 150, 60, 170), [(0, 0, 100, 200)]))

    def test_overlapping_people_are_ambiguous(self):
        self.assertIsNone(owner((90, 10, 110, 30), [(0, 0, 120, 200), (80, 0, 200, 200)]))


class AgeVotesTest(unittest.TestCase):
    def test_one_noisy_frame_does_not_flip_label(self):
        votes = AgeVotes(0.6, window=5)
        for frame in range(4):
            votes.add(1, frame, bands(b25_32=0.9, b60_100=0.1))
        votes.add(1, 4, bands(b60_100=0.9, b25_32=0.1))
        self.assertEqual(votes.label(1), 'Adult')

    def test_invalid_scores_are_ignored(self):
        votes = AgeVotes(0.6)
        votes.add(1, 1, bands(b0_2=0.9, b4_6=0.1))
        votes.add(1, 2, bands(b0_2=np.nan))
        self.assertEqual(votes.label(1), 'Child')

    def test_due_on_first_sight_then_every_refresh_even_without_face(self):
        votes = AgeVotes(0.6, refresh=3)
        self.assertTrue(votes.due(3, frame_number=1))
        votes.add(3, 1)
        self.assertFalse(votes.due(3, frame_number=2))
        self.assertFalse(votes.due(3, frame_number=3))
        self.assertTrue(votes.due(3, frame_number=4))
        self.assertEqual(votes.label(3), 'Unknown')

    def test_forget_drops_absent_people(self):
        votes = AgeVotes(0.6)
        votes.add(1, 1, bands(b25_32=1.0))
        votes.add(2, 1, bands(b25_32=1.0))
        votes.forget([2])
        self.assertEqual(votes.label(1), 'Unknown')
        self.assertTrue(votes.due(1, frame_number=2))
        self.assertEqual(votes.label(2), 'Adult')


class ChildVotesTest(unittest.TestCase):
    def test_continued_id_starts_again_at_checking(self):
        votes = ChildVotes(0.5, settle=1)
        votes.add(1, 0, 0.1)
        self.assertEqual(votes.label(1), 'Adult')
        self.assertFalse(votes.carry_over)
        votes.restart(1)
        self.assertEqual(votes.label(1), 'Checking')

    def test_missing_readings_keep_earlier_label(self):
        votes = ChildVotes(0.5, settle=1)
        votes.add(1, 1, 0.2)
        votes.add(1, 4, None)
        self.assertEqual(votes.label(1), 'Adult')
        votes.add(2, 1, None)
        self.assertEqual(votes.label(2), 'Checking')

    def test_checking_until_settled(self):
        votes = ChildVotes(0.5, settle=3)
        votes.add(1, 1, 0.9)
        votes.add(1, 4, 0.9)
        self.assertEqual(votes.label(1), 'Checking')
        votes.add(1, 7, 0.9)
        self.assertEqual(votes.label(1), 'Child')

    def test_average_decides(self):
        votes = ChildVotes(0.5, settle=1)
        for frame, probability in enumerate((0.2, 0.3, 0.9)):
            votes.add(2, frame, probability)
        self.assertEqual(votes.label(2), 'Adult')

    def test_unseen_person_is_checking(self):
        self.assertEqual(ChildVotes(0.5).label(9), 'Checking')

    def test_confidence_is_in_the_label_shown(self):
        votes = ChildVotes(0.3, settle=2)
        votes.add(1, 0, 0.9)
        self.assertIsNone(votes.confidence(1))  # still Checking
        votes.add(1, 3, 0.7)
        self.assertAlmostEqual(votes.confidence(1), 0.8)  # Child
        for frame, probability in enumerate((0.1, 0.2), 1):
            votes.add(2, frame, probability)
        self.assertEqual(votes.label(2), 'Adult')
        self.assertAlmostEqual(votes.confidence(2), 0.85)
        self.assertIsNone(AgeVotes(0.6).confidence(1))


class RecallThresholdTest(unittest.TestCase):
    def test_higher_recall_means_a_lower_threshold(self):
        scores = np.linspace(-3, 3, 101)
        self.assertAlmostEqual(recall_threshold(scores, 0.5), 0.5)
        self.assertLess(recall_threshold(scores, 0.95), recall_threshold(scores, 0.80))


class SquareTest(unittest.TestCase):
    def test_pads_instead_of_stretching(self):
        tall = np.full((200, 50, 3), 255, np.uint8)
        out = square(tall, 100)
        self.assertEqual(out.shape, (100, 100, 3))
        self.assertEqual(out[50, 50].tolist(), [255, 255, 255])
        self.assertEqual(out[50, 5].tolist(), [127, 127, 127])


class LabellerTest(unittest.TestCase):
    TRACKS = {1: (0, 0, 10, 100), 2: (20, 0, 30, 100)}
    PEOPLE = {track_id: (1, box) for track_id, box in TRACKS.items()}

    def test_reads_people_when_due_and_again_when_their_id_was_continued(self):
        wanted = []

        def read(frame, boxes, indices):
            wanted.append(list(indices))
            return {i: 0.9 for i in indices}

        labeller = Labeller('child', LABELS, read, ChildVotes(0.5, settle=1))
        self.assertEqual(labeller.update(None, 1, self.TRACKS, self.PEOPLE, []), {1: 'Child', 2: 'Child'})
        labeller.update(None, 2, self.TRACKS, self.PEOPLE, [2])
        self.assertEqual(wanted, [[0, 1], [1]])

    def test_people_who_left_are_forgotten(self):
        labeller = Labeller('child', LABELS, lambda frame, boxes, indices: {}, ChildVotes(0.5))
        labeller.update(None, 1, self.TRACKS, self.PEOPLE, [])
        labeller.update(None, 2, {}, {}, [])
        self.assertEqual(labeller.votes.next_try, {})


class ChildLabellerTest(unittest.TestCase):
    def test_people_too_small_to_judge_get_no_reading(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(child, 'ChildClassifier') as classifier:
            head = Path(folder) / 'child_head.npz'
            head.write_bytes(b'head')
            classifier.return_value.threshold = 0.5
            classifier.return_value.probabilities.side_effect = lambda frame, boxes: [0.9] * len(boxes)
            labeller = child.child_labeller(head, 'cpu', 0.9)
            version = f'child_head {child.fingerprint(head)}'
        tall, short = (0, 0, 10, child.MIN_HEIGHT), (0, 0, 10, child.MIN_HEIGHT - 1)
        self.assertEqual(labeller.read(None, [tall, short], [0, 1]), {0: 0.9})
        self.assertEqual(labeller.models, {'child_classifier': version, 'child_recall': 0.9})


if __name__ == '__main__':
    unittest.main()
