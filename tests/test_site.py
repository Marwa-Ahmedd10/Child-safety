"""Checks for the multi-camera runner; run with `python -m unittest`."""
import unittest
from unittest import mock

from human_detection.site import Camera, camera_args, slug


class CameraArgsTest(unittest.TestCase):
    def test_options_become_flags(self):
        defaults = {'model': 'small', 'events': 'logs/{camera}_alerts.jsonl', 'gpu': False}
        name, args = camera_args(defaults, {'name': 'Main Pool', 'source': 'env:POOL_RTSP', 'child_recall': 0.9,
                                            'child': True})
        self.assertEqual(name, 'Main Pool')
        self.assertEqual(args, ['--camera-name', 'Main Pool', '--no-window', '--model', 'small',
                                '--events', 'logs/main_pool_alerts.jsonl', '--source', 'env:POOL_RTSP',
                                '--child-recall', '0.9', '--child'])

    def test_camera_needs_name_and_source(self):
        with self.assertRaises(ValueError):
            camera_args({}, {'name': 'Lobby'})

    def test_slug(self):
        self.assertEqual(slug('Lobby / Entrance 2'), 'lobby_entrance_2')


class RestartWaitTest(unittest.TestCase):
    def test_waits_grow_while_failing_quickly_and_reset_after_a_good_run(self):
        camera = Camera('Lobby', [])
        self.assertEqual([camera.next_wait(ran=3) for _ in range(9)], [5, 10, 20, 40, 80, 160, 300, 300, 300])
        self.assertEqual(camera.next_wait(ran=3600), 5)


class SafetyOfOutputTest(unittest.TestCase):
    def test_config_error_does_not_print_values(self):
        with self.assertRaises(ValueError) as error:
            camera_args({}, {'source': 'rtsp://admin:S3cret@10.0.0.5/stream'})
        self.assertNotIn('S3cret', str(error.exception))

    def test_relay_keeps_draining_when_printing_fails(self):
        camera = Camera('Lobby', [])
        camera.process = mock.Mock(stdout=iter(['one\n', 'two\n', 'three\n']))
        with mock.patch('builtins.print', side_effect=OSError('broken pipe')):
            camera.relay()
        self.assertEqual(list(camera.process.stdout), [])  # every line was read


if __name__ == '__main__':
    unittest.main()
