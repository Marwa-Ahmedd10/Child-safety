"""Checks for the live camera app's options, entry point and camera health; run with
`python -m unittest`."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from human_detection.app import overlay
from human_detection.app.cli import parse_args
from human_detection.app.health import CameraHealth
from human_detection.app.pipeline import camera_name
from human_detection.safety.zones import parse_zone


class CameraNameTest(unittest.TestCase):
    def test_password_is_not_in_alerts(self):
        self.assertEqual(camera_name('rtsp://admin:p@ss@10.0.0.5:554/stream1'), 'rtsp://10.0.0.5:554/stream1')
        self.assertEqual(camera_name('rtsp://10.0.0.5/stream1'), 'rtsp://10.0.0.5/stream1')
        self.assertEqual(camera_name('0'), '0')


class OptionsTest(unittest.TestCase):
    def test_env_values_come_from_the_environment(self):
        with mock.patch.dict(os.environ, {'POOL_RTSP': ' rtsp://admin:S3cret@10.0.0.5/stream1\n'}):
            args = parse_args(['--source', 'env:POOL_RTSP'])
        self.assertEqual(args.source, 'rtsp://admin:S3cret@10.0.0.5/stream1')

    def test_rejects_bad_options(self):
        for argv in (['--source', 'env:NOT_SET_ANYWHERE'], ['--webhook', 'ftp://example.com'], ['--conf', '0'],
                     ['--child-recall', '1'], ['--zones', 'zones.json', '--age'], ['--counts', 'counts.jsonl']):
            with self.subTest(argv=argv), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args(argv)


class EntryPointTest(unittest.TestCase):
    def test_run_site_can_start_the_app_as_a_module(self):
        result = subprocess.run([sys.executable, '-m', 'human_detection.app', '--help'],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('usage: live-camera', result.stdout)


class OverlayTest(unittest.TestCase):
    def test_draws_every_part_of_the_preview_at_any_frame_size(self):
        zones = [parse_zone({'name': 'Pool', 'type': 'hazard', 'polygon': [[0, 0.5], [0.5, 0.5], [0.5, 1]]}),
                 parse_zone({'name': 'Room', 'type': 'count', 'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}),
                 parse_zone({'name': 'Door', 'type': 'line', 'polygon': [[0.5, 1], [0.5, 0]]})]
        crossings = {'Door': {'in': {'total': 3}, 'out': {'total': 1}}}
        people = {1: (5, (10, 10, 60, 150)), 2: (4, (300, 100, 340, 200))}
        for height, width in ((240, 320), (720, 1280), (1920, 1440)):
            image = np.zeros((height, width, 3), np.uint8)
            overlay.draw_zones(image, zones, {'Room': (1, 1)}, crossings)
            overlay.draw_people(image, people, {1: 'Child', 2: 'Adult'}, overlay.LABEL_COLORS['child'], 5,
                                {1: 0.87, 2: None})
            overlay.draw_header(image, 'People: 2')
            overlay.draw_alerts(image, ['1 child in hazard zone "Pool"', 'a much longer message ' * 6])
            self.assertEqual(overlay.offline_view(image).shape, image.shape)
            self.assertTrue(image.any())


class FakeSource:
    def __init__(self):
        self.down, self.last_frame_at, self.drops = False, 100.0, 0


class CameraHealthTest(unittest.TestCase):
    def test_offline_alert_starts_once_and_ends_with_its_duration(self):
        events, source = mock.Mock(), FakeSource()
        health = CameraHealth('Pool', events)
        health.check(source, dict)
        source.down = True
        health.check(source, dict)
        health.check(source, dict)
        source.down, source.last_frame_at = False, 112.5
        health.check(source, dict)
        emitted = [call.args[0] for call in events.emit.call_args_list]
        self.assertEqual([(event['event'], event['state']) for event in emitted],
                         [('camera_offline', 'start'), ('camera_offline', 'end')])
        self.assertEqual(emitted[1]['duration_s'], 12.5)

    def test_alert_still_open_when_monitoring_stops_is_ended(self):
        events, source = mock.Mock(), FakeSource()
        health = CameraHealth('Pool', events)
        health.finish()
        events.emit.assert_not_called()
        source.down = True
        health.check(source, dict)
        health.finish()
        self.assertEqual(events.emit.call_args.args[0]['reason'], 'monitor stopped')

    def test_status_file_holds_health_and_the_apps_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pool.json'
            CameraHealth('Pool', mock.Mock(), path).check(FakeSource(), lambda: {'frames': 7})
            status = json.loads(path.read_text())
        self.assertEqual({key: status[key] for key in ('camera', 'online', 'drops', 'frames')},
                         {'camera': 'Pool', 'online': True, 'drops': 0, 'frames': 7})


if __name__ == '__main__':
    unittest.main()
