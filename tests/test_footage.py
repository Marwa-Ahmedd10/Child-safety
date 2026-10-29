"""Checks for the footage labelling keys; run with `python -m unittest`."""
import unittest

import numpy as np

from human_detection.training.footage import apply_key, split_points


class SplitTest(unittest.TestCase):
    def test_splits_where_the_person_changes(self):
        one, other = np.array([1.0, 0.1, 0]), np.array([0, 1.0, 0.2])
        self.assertEqual(split_points(np.array([one, one * 2, one + 0.05, other, other])), [0, 3])

    def test_one_person_stays_whole(self):
        self.assertEqual(split_points(np.ones((4, 3))), [0])


def people(*labels):
    return [{'track': str(i), 'label': label} for i, label in enumerate(labels)]


class LabelKeysTest(unittest.TestCase):
    def test_label_moves_to_next_unlabelled(self):
        rows = people('', 'adult', '')
        self.assertEqual(apply_key(rows, 0, ord('c')), 2)
        self.assertEqual(rows[0]['label'], 'child')

    def test_last_label_finishes(self):
        rows = people('adult', '')
        self.assertIsNone(apply_key(rows, 1, ord('s')))
        self.assertEqual(rows[1]['label'], 'skip')

    def test_back_relabel_and_quit(self):
        rows = people('child', '', '')
        self.assertEqual(apply_key(rows, 1, ord('b')), 0)
        self.assertEqual(apply_key(rows, 0, ord('a')), 1)
        self.assertEqual(rows[0]['label'], 'adult')
        self.assertIsNone(apply_key(rows, 1, ord('q')))
        self.assertEqual(apply_key(rows, 1, ord('x')), 1)


if __name__ == '__main__':
    unittest.main()
