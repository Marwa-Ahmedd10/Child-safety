"""Checks for visitor counting at lines; run with `python -m unittest`."""
import json
import tempfile
import unittest
from pathlib import Path

from human_detection.safety.zones import parse_zone
from human_detection.visitors import VisitorCounter

SIZE = (1000, 1000)
# Drawn bottom to top, so "in" (its right side) is towards the right of the image.
DOOR = parse_zone({'name': 'Door', 'type': 'line', 'polygon': [[0.5, 1], [0.5, 0]]})


def box(x, bottom=900):
    return (x - 50, bottom - 300, x + 50, bottom)


def walk(counter, track_id, xs, label='Adult', start=0):
    counted = []
    for frame, x in enumerate(xs, start):
        counted += counter.update(frame, {track_id: box(x)}, {track_id: label}, SIZE)
    return counted


class VisitorCounterTest(unittest.TestCase):
    def test_counts_each_direction_with_labels(self):
        counter = VisitorCounter([DOOR])
        self.assertEqual(walk(counter, 1, (300, 400, 600, 700)), [('Door', 'in', 1, 'Adult')])
        self.assertEqual(walk(counter, 1, (700, 300), label='Child', start=4), [('Door', 'out', 1, 'Child')])
        self.assertEqual(counter.totals['Door']['in']['adults'], 1)
        self.assertEqual(counter.totals['Door']['out']['children'], 1)

    def test_standing_or_swaying_on_the_line_does_not_count(self):
        counter = VisitorCounter([DOOR])
        self.assertEqual(walk(counter, 1, (495, 505, 498, 503, 496)), [])
        # Swaying 60 px either side is under a quarter of this 300 px tall person.
        self.assertEqual(walk(counter, 2, (440, 560, 450, 550, 440, 560)), [])

    def test_passing_beyond_the_line_end_does_not_count(self):
        short = parse_zone({'name': 'Door', 'type': 'line', 'polygon': [[0.5, 0.5], [0.5, 0]]})
        counter = VisitorCounter([short])
        self.assertEqual(walk(counter, 1, (300, 700)), [])

    def test_waits_for_labels_then_counts(self):
        counter = VisitorCounter([DOOR])
        self.assertEqual(walk(counter, 1, (300, 700), label='Checking'), [])
        self.assertEqual(walk(counter, 1, (750,), start=2), [('Door', 'in', 1, 'Adult')])

    def test_person_who_leaves_unsettled_counts_as_unknown(self):
        counter = VisitorCounter([DOOR])
        walk(counter, 1, (300, 700), label='Checking')
        self.assertEqual(counter.update(2, {}, {}, SIZE), [('Door', 'in', 1, 'Checking')])
        self.assertEqual(counter.totals['Door']['in']['age_unknown'], 1)

    def test_flush_writes_period_and_resets(self):
        counter = VisitorCounter([DOOR])
        walk(counter, 1, (300, 700))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'counts.jsonl'
            counter.flush('Lobby', path)
            record = json.loads(path.read_text())
        self.assertEqual((record['line'], record['in']['total'], record['out']['total']), ('Door', 1, 0))
        self.assertEqual(counter.period['Door']['in']['total'], 0)
        self.assertEqual(counter.totals['Door']['in']['total'], 1)


if __name__ == '__main__':
    unittest.main()
