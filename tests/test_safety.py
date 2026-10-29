"""Checks for the child-safety rules; run with `python -m unittest`."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from human_detection.safety.monitor import SafetyMonitor
from human_detection.safety.zones import Zone, foot, load_zones, parse_zone

SIZE = (1000, 1000)
POOL = Zone('Pool', 'hazard', [(0, 0), (0.3, 0), (0.3, 1), (0, 1)])
EXIT = Zone('Exit', 'exit', [(0.7, 0), (1, 0), (1, 1), (0.7, 1)])
TV = Zone('TV', 'ignore', [(0.4, 0.1), (0.6, 0.1), (0.6, 0.4), (0.4, 0.4)])


def room(**limits):
    return Zone('Room', 'count', [(0, 0), (1, 0), (1, 1), (0, 1)], **limits)


def box(x, bottom, height=300):
    """A person box standing at pixel (x, bottom)."""
    return (x - 50, bottom - height, x + 50, bottom)


def run(monitor, frames, people, labels):
    events = []
    for _ in range(frames):
        events += monitor.update(people, labels, SIZE)
    return [(event['event'], event['state']) for event in events]


class ZoneTest(unittest.TestCase):
    def test_contains(self):
        self.assertTrue(POOL.contains((0.1, 0.5)))
        self.assertFalse(POOL.contains((0.5, 0.5)))
        triangle = Zone('T', 'hazard', [(0, 0), (1, 0), (0, 1)])
        self.assertTrue(triangle.contains((0.2, 0.2)))
        self.assertFalse(triangle.contains((0.8, 0.8)))

    def test_person_at_image_edge_is_inside_full_frame_zone(self):
        self.assertTrue(room().contains(foot(box(500, 1000), SIZE)))

    def test_rejects_bad_zones(self):
        with self.assertRaises(ValueError):
            parse_zone({'name': 'X', 'type': 'pool', 'polygon': [[0, 0], [1, 0], [1, 1]]})
        with self.assertRaises(ValueError):
            parse_zone({'name': 'X', 'type': 'hazard', 'polygon': [[0, 0], [2, 0], [1, 1]]})
        with self.assertRaises(ValueError):
            parse_zone({'name': 'X', 'type': 'hazard', 'polygon': [[0, 0], [1, 0], [1, 1]], 'colour': 'red'})

    def test_load_zones_with_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            path.write_text(json.dumps({'confirm_frames': 4, 'zones': [
                {'name': 'Pool', 'type': 'hazard', 'polygon': [[0, 0], [0.3, 0], [0.3, 1]]}]}))
            zones, settings = load_zones(path)
        self.assertEqual(zones[0].name, 'Pool')
        self.assertEqual(settings, {'confirm_frames': 4, 'release_frames': 30, 'accompany_distance': 1.5})


class HazardTest(unittest.TestCase):
    def test_child_alerts_once_then_clears(self):
        monitor = SafetyMonitor([POOL], confirm_frames=3, release_frames=2)
        people, labels = {1: box(100, 900)}, {1: 'Child'}
        self.assertEqual(run(monitor, 2, people, labels), [])
        self.assertEqual(run(monitor, 1, people, labels), [('child_in_hazard_zone', 'start')])
        self.assertEqual(run(monitor, 5, people, labels), [])
        self.assertEqual(monitor.active_messages(), ['1 child in hazard zone "Pool"'])
        self.assertEqual(run(monitor, 2, {}, {}), [('child_in_hazard_zone', 'end')])

    def test_one_alert_per_zone_when_track_ids_change(self):
        monitor = SafetyMonitor([POOL], confirm_frames=2, release_frames=3)
        self.assertEqual(run(monitor, 2, {1: box(100, 900)}, {1: 'Child'}), [('child_in_hazard_zone', 'start')])
        # The tracker gives the same child a new ID, then follows them as two people.
        self.assertEqual(run(monitor, 2, {7: box(110, 900)}, {7: 'Child'}), [])
        self.assertEqual(run(monitor, 2, {7: box(110, 900), 8: box(120, 900)}, {7: 'Child', 8: 'Checking'}), [])
        self.assertEqual(monitor.active_messages(), ['2 children in hazard zone "Pool"'])

    def test_checking_counts_as_possible_child(self):
        monitor = SafetyMonitor([POOL], confirm_frames=2)
        self.assertEqual(run(monitor, 2, {1: box(100, 900)}, {1: 'Checking'}), [('child_in_hazard_zone', 'start')])

    def test_settling_to_adult_before_confirm_cancels(self):
        monitor = SafetyMonitor([POOL], confirm_frames=3)
        people = {1: box(100, 900)}
        run(monitor, 2, people, {1: 'Checking'})
        self.assertEqual(run(monitor, 5, people, {1: 'Adult'}), [])

    def test_new_ids_flashing_checking_do_not_add_up(self):
        monitor = SafetyMonitor([POOL], confirm_frames=3)
        events = []
        for track_id in range(1, 20):  # a new ID every frame, each Checking once
            events += monitor.update({track_id: box(100, 900)}, {track_id: 'Checking'}, SIZE)
        self.assertEqual(events, [])

    def test_alert_survives_a_track_id_change(self):
        monitor = SafetyMonitor([POOL], confirm_frames=3, release_frames=5)
        self.assertEqual(run(monitor, 3, {1: box(100, 900)}, {1: 'Child'}), [('child_in_hazard_zone', 'start')])
        # The child gets a new ID and starts Checking again: the alert stays on.
        self.assertEqual(run(monitor, 2, {7: box(100, 900)}, {7: 'Checking'}), [])
        self.assertEqual(run(monitor, 5, {7: box(100, 900)}, {7: 'Child'}), [])
        self.assertEqual(run(monitor, 5, {}, {}), [('child_in_hazard_zone', 'end')])

    def test_open_alerts_end_when_monitor_stops(self):
        monitor = SafetyMonitor([POOL], confirm_frames=2)
        run(monitor, 2, {1: box(100, 900)}, {1: 'Child'})
        events = monitor.finish()
        self.assertEqual([(e['event'], e['state'], e['reason']) for e in events],
                         [('child_in_hazard_zone', 'end', 'monitor stopped')])
        self.assertEqual(monitor.finish(), [])

    def test_child_outside_zone_is_fine(self):
        monitor = SafetyMonitor([POOL], confirm_frames=1)
        self.assertEqual(run(monitor, 3, {1: box(500, 900)}, {1: 'Child'}), [])


class ExitTest(unittest.TestCase):
    def test_child_alone_at_exit(self):
        monitor = SafetyMonitor([EXIT], confirm_frames=2)
        self.assertEqual(run(monitor, 2, {1: box(850, 900)}, {1: 'Child'}), [('child_exiting_alone', 'start')])

    def test_adult_nearby_accompanies(self):
        monitor = SafetyMonitor([EXIT], confirm_frames=2)
        people = {1: box(850, 900, height=150), 2: box(650, 900)}
        self.assertEqual(run(monitor, 5, people, {1: 'Child', 2: 'Adult'}), [])

    def test_distant_or_unconfirmed_person_does_not_accompany(self):
        far = SafetyMonitor([EXIT], confirm_frames=2)
        people = {1: box(850, 900, height=150), 2: box(100, 300, height=150)}
        self.assertEqual(run(far, 2, people, {1: 'Child', 2: 'Adult'}), [('child_exiting_alone', 'start')])
        unconfirmed = SafetyMonitor([EXIT], confirm_frames=2)
        people = {1: box(850, 900, height=150), 2: box(760, 900)}
        self.assertEqual(run(unconfirmed, 2, people, {1: 'Child', 2: 'Checking'}), [('child_exiting_alone', 'start')])
        self.assertEqual(unconfirmed.active_messages(), ['2 children at exit "Exit" with no adult nearby'])


class CountTest(unittest.TestCase):
    def test_too_many_children(self):
        monitor = SafetyMonitor([room(max_children=1)], confirm_frames=1)
        people = {1: box(200, 900), 2: box(600, 900)}
        self.assertEqual(run(monitor, 1, people, {1: 'Child', 2: 'Child'}), [('too_many_children', 'start')])
        self.assertEqual(monitor.zone_counts, {'Room': (2, 0)})

    def test_checking_is_not_counted_yet(self):
        monitor = SafetyMonitor([room(max_children=1, max_children_per_adult=1)], confirm_frames=1)
        people = {1: box(200, 900), 2: box(600, 900), 3: box(800, 900)}
        self.assertEqual(run(monitor, 3, people, {1: 'Child', 2: 'Checking', 3: 'Adult'}), [])
        self.assertEqual(monitor.zone_counts, {'Room': (1, 1)})

    def test_count_messages_read_naturally(self):
        monitor = SafetyMonitor([room(max_children=0, max_children_per_adult=0.5)], confirm_frames=1)
        monitor.update({1: box(100, 900), 2: box(300, 900)}, {1: 'Child', 2: 'Adult'}, SIZE)
        self.assertEqual(sorted(monitor.active_messages()),
                         ['1 child and 1 adult in "Room" (max 0.5 per adult)', '1 child in "Room" (limit 0)'])

    def test_too_few_adults(self):
        monitor = SafetyMonitor([room(max_children_per_adult=2)], confirm_frames=1)
        people = {1: box(100, 900), 2: box(300, 900), 3: box(500, 900), 4: box(700, 900)}
        labels = {1: 'Child', 2: 'Child', 3: 'Child', 4: 'Adult'}
        self.assertEqual(run(monitor, 1, people, labels), [('too_few_adults', 'start')])
        labels[3] = 'Adult'
        self.assertEqual(run(SafetyMonitor([room(max_children_per_adult=2)], confirm_frames=1), 1, people, labels), [])


class AnyoneZoneTest(unittest.TestCase):
    POOL_AFTER_HOURS = Zone('Pool', 'hazard', [(0, 0), (0.3, 0), (0.3, 1), (0, 1)], anyone=True)

    def test_alerts_on_adults_too(self):
        monitor = SafetyMonitor([self.POOL_AFTER_HOURS], confirm_frames=2)
        self.assertEqual(run(monitor, 2, {1: box(100, 900)}, {1: 'Adult'}), [('person_in_hazard_zone', 'start')])
        self.assertEqual(monitor.active_messages(), ['1 person in hazard zone "Pool"'])

    def test_works_without_labels(self):
        monitor = SafetyMonitor([self.POOL_AFTER_HOURS], confirm_frames=2)
        self.assertEqual(run(monitor, 2, {1: box(100, 900)}, {}), [('person_in_hazard_zone', 'start')])

    def test_only_hazard_zones_take_anyone(self):
        with self.assertRaises(ValueError):
            parse_zone({'name': 'Door', 'type': 'exit', 'polygon': [[0, 0], [1, 0], [1, 1]], 'anyone': True})

    def test_which_zones_need_labels(self):
        self.assertFalse(self.POOL_AFTER_HOURS.needs_labels())
        self.assertTrue(POOL.needs_labels())
        self.assertFalse(TV.needs_labels())


class HazardOptionsTest(unittest.TestCase):
    """Options that fit one rule to any hazard area: who it alerts on, how long they must
    stay, and what the alert says."""

    def zone(self, **options):
        return parse_zone({'name': 'Stairs', 'type': 'hazard', 'polygon': [[0, 0], [0.3, 0], [0.3, 1], [0, 1]],
                           **options})

    def test_alert_text(self):
        monitor = SafetyMonitor([self.zone(alert='Child near the stairs')], confirm_frames=1)
        events = monitor.update({1: box(100, 900)}, {1: 'Child'}, SIZE)
        self.assertEqual((events[0]['message'], events[0]['children']), ('Child near the stairs', 1))
        self.assertEqual(monitor.active_messages(), ['Child near the stairs'])

    def test_unaccompanied_child_alerts(self):
        monitor = SafetyMonitor([self.zone(unaccompanied=True)], confirm_frames=2)
        self.assertEqual(run(monitor, 2, {1: box(100, 900)}, {1: 'Child'}), [('child_alone_in_hazard_zone', 'start')])
        self.assertEqual(monitor.active_messages(), ['1 child in hazard zone "Stairs" with no adult nearby'])

    def test_child_with_an_adult_does_not_alert(self):
        monitor = SafetyMonitor([self.zone(unaccompanied=True)], confirm_frames=2)
        people = {1: box(100, 900), 2: box(200, 900)}
        self.assertEqual(run(monitor, 5, people, {1: 'Child', 2: 'Adult'}), [])
        # Someone still being checked does not count as an adult.
        self.assertEqual(run(monitor, 2, people, {1: 'Child', 2: 'Checking'}),
                         [('child_alone_in_hazard_zone', 'start')])

    def test_after_seconds_waits_on_the_frame_clock(self):
        monitor = SafetyMonitor([self.zone(after_seconds=3)], confirm_frames=2)
        people, labels = {1: box(100, 900)}, {1: 'Child'}
        for second in range(3):
            self.assertEqual(monitor.update(people, labels, SIZE, timestamp=100 + second), [])
        self.assertEqual(monitor.update(people, labels, SIZE, timestamp=103)[0]['event'], 'child_in_hazard_zone')

    def test_leaving_restarts_the_wait(self):
        monitor = SafetyMonitor([self.zone(after_seconds=3)], confirm_frames=1)
        inside, outside, labels = {1: box(100, 900)}, {1: box(800, 900)}, {1: 'Child'}
        monitor.update(inside, labels, SIZE, timestamp=0)
        monitor.update(outside, labels, SIZE, timestamp=2)
        self.assertEqual(monitor.update(inside, labels, SIZE, timestamp=4), [])
        self.assertEqual(monitor.update(inside, labels, SIZE, timestamp=6), [])
        self.assertEqual(len(monitor.update(inside, labels, SIZE, timestamp=7)), 1)

    def test_exit_zones_take_the_wait_and_text(self):
        door = parse_zone({'name': 'Door', 'type': 'exit', 'polygon': [[0.7, 0], [1, 0], [1, 1], [0.7, 1]],
                           'after_seconds': 1, 'alert': 'Child leaving alone'})
        monitor = SafetyMonitor([door], confirm_frames=1)
        people, labels = {1: box(900, 900)}, {1: 'Child'}
        self.assertEqual(monitor.update(people, labels, SIZE, timestamp=0), [])
        self.assertEqual(monitor.update(people, labels, SIZE, timestamp=1)[0]['message'], 'Child leaving alone')

    def test_bad_options_rejected(self):
        for options in ({'anyone': True, 'unaccompanied': True}, {'after_seconds': -1},
                        {'after_seconds': '5'}, {'alert': ''}, {'alert': 5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.zone(**options)
        for options in ({'unaccompanied': True}, {'after_seconds': 2}, {'alert': 'Too many'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                parse_zone({'name': 'Room', 'type': 'count', 'polygon': [[0, 0], [1, 0], [1, 1]], **options})

    def test_unaccompanied_zones_need_labels(self):
        self.assertTrue(self.zone(unaccompanied=True).needs_labels())


class HoursTest(unittest.TestCase):
    def zone(self, *spans):
        return parse_zone({'name': 'Pool', 'type': 'hazard', 'polygon': [[0, 0], [0.3, 0], [0.3, 1], [0, 1]],
                           'anyone': True, 'hours': list(spans)})

    def at(self, hour, minute=0):
        return datetime(2026, 11, 3, hour, minute)

    def test_overnight_span(self):
        zone = self.zone('20:00-07:00')
        self.assertTrue(zone.active(self.at(23)))
        self.assertTrue(zone.active(self.at(6, 59)))
        self.assertFalse(zone.active(self.at(7)))
        self.assertFalse(zone.active(self.at(12)))

    def test_daytime_spans(self):
        zone = self.zone('06:00-08:00', '12:00-13:30')
        self.assertTrue(zone.active(self.at(13, 15)))
        self.assertFalse(zone.active(self.at(9)))

    def test_inactive_zone_raises_nothing(self):
        monitor = SafetyMonitor([self.zone('20:00-07:00')], confirm_frames=1, clock=lambda: self.at(12))
        self.assertEqual(run(monitor, 3, {1: box(100, 900)}, {1: 'Child'}), [])

    def test_bad_hours_rejected(self):
        with self.assertRaises(ValueError):
            self.zone('8pm-7am')


class LineTest(unittest.TestCase):
    def test_line_needs_two_points(self):
        self.assertEqual(parse_zone({'name': 'Door', 'type': 'line', 'polygon': [[0.5, 1], [0.5, 0]]}).type, 'line')
        with self.assertRaises(ValueError):
            parse_zone({'name': 'Door', 'type': 'line', 'polygon': [[0, 0], [1, 0], [1, 1]]})
        with self.assertRaises(ValueError):
            parse_zone({'name': 'Pool', 'type': 'hazard', 'polygon': [[0, 0], [1, 0]]})

    def test_line_with_one_point_twice_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_zone({'name': 'Door', 'type': 'line', 'polygon': [[0.5, 0.5], [0.5, 0.5]]})

    def test_lines_raise_no_alerts(self):
        monitor = SafetyMonitor([parse_zone({'name': 'Door', 'type': 'line', 'polygon': [[0.5, 1], [0.5, 0]]})],
                                confirm_frames=1)
        self.assertEqual(run(monitor, 2, {1: box(500, 900)}, {1: 'Child'}), [])


class IgnoreTest(unittest.TestCase):
    def test_person_on_a_screen_is_ignored_but_not_someone_in_front(self):
        monitor = SafetyMonitor([TV])
        self.assertTrue(monitor.ignored(box(500, 350, height=200), SIZE))
        self.assertFalse(monitor.ignored(box(500, 900, height=700), SIZE))


if __name__ == '__main__':
    unittest.main()
