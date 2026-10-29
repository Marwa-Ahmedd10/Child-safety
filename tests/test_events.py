"""Checks for alert delivery and output files; run with `python -m unittest`."""
import json
import socketserver
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from human_detection.app.health import write_status
from human_detection.safety.events import EventLog
from human_detection.safety.zones import parse_zone
from human_detection.visitors import VisitorCounter


class EventLogTest(unittest.TestCase):
    def test_events_name_the_models_behind_them(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'alerts.jsonl'
            log = EventLog('Pool', path=path)
            log.models = {'detector': 'rf-detr-nano'}
            log.emit({'event': 'person_in_hazard_zone', 'state': 'start', 'zone': 'Pool', 'message': 'test'})
            event = json.loads(path.read_text())
        self.assertEqual((event['camera'], event['models']), ('Pool', {'detector': 'rf-detr-nano'}))


class WebhookTest(unittest.TestCase):
    def test_retries_until_delivered(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(500 if len(received) < 3 else 200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with mock.patch('time.sleep', lambda seconds: None):
                log = EventLog('test', webhook=f'http://127.0.0.1:{server.server_port}/alerts')
                log.emit({'event': 'child_in_hazard_zone', 'state': 'start', 'zone': 'Pool', 'message': 'test'})
                log.close(timeout=5)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(len(received), 3)
        self.assertEqual(received[-1]['zone'], 'Pool')

    def test_a_reply_that_is_not_http_does_not_stop_later_alerts(self):
        requests = []

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                length = 0
                while (line := self.rfile.readline()) not in (b'\r\n', b''):
                    if line.lower().startswith(b'content-length:'):
                        length = int(line.split(b':')[1])
                requests.append(json.loads(self.rfile.read(length)))
                self.wfile.write(b'garbage\r\n' if len(requests) == 1 else b'HTTP/1.0 200 OK\r\n\r\n')

        server = socketserver.TCPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with mock.patch('time.sleep', lambda seconds: None), mock.patch('human_detection.safety.events.log'):
                log = EventLog('test', webhook=f'http://127.0.0.1:{server.server_address[1]}/alerts')
                log.emit({'event': 'first', 'state': 'start', 'zone': 'Pool', 'message': 'test'})
                log.close(timeout=5)
                log.emit({'event': 'second', 'state': 'start', 'zone': 'Pool', 'message': 'test'})
                log.close(timeout=5)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual([request['event'] for request in requests], ['first', 'first', 'second'])


class OutputFailureTest(unittest.TestCase):
    def test_unwritable_files_do_not_stop_monitoring(self):
        missing = Path('/nonexistent-folder/out.jsonl')
        door = parse_zone({'name': 'Door', 'type': 'line', 'polygon': [[0.5, 1], [0.5, 0]]})
        with mock.patch('human_detection.safety.events.log') as events_log, \
                mock.patch('human_detection.visitors.log') as visitors_log, \
                mock.patch('human_detection.app.health.log') as health_log:
            EventLog('Pool', path=missing).emit({'event': 'x', 'state': 'start', 'zone': 'Pool', 'message': 'm'})
            write_status(missing, {'online': True})
            VisitorCounter([door]).flush('Pool', missing)
        events_log.error.assert_called_once()
        health_log.warning.assert_called_once()
        visitors_log.error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
