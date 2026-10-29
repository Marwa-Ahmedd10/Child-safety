"""Run every camera of a site, one live-camera process each, from a TOML file, and
restart any that stops: waiting 5 seconds, doubling to at most 5 minutes while it keeps
failing within a minute of starting. SIGTERM or Ctrl-C stops them all cleanly, so their
open alerts are closed and counts saved.

    uv run run-site site.toml

Each [[camera]] table takes live-camera's options without the dashes (child_recall,
not --child-recall) plus a name; [defaults] holds options shared by every camera. In a
default, {camera} is replaced by the camera's name made safe for file names. Values
written 'env:NAME' are read from the environment variable NAME by live-camera, so
passwords stay out of the file. See site.example.toml.
"""
import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

FIRST_WAIT, MAX_WAIT = 5, 300  # seconds before restarting a camera that stopped
STABLE = 60                    # a camera that ran this long before stopping restarts after FIRST_WAIT
log = logging.getLogger('site')


def slug(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_').lower() or 'camera'


def camera_args(defaults, camera):
    """live-camera's command-line options for one camera."""
    options = {**defaults, **camera}
    if 'name' not in options or 'source' not in options:
        # Only the keys: the values can hold a stream URL with its password.
        raise ValueError(f'Each [[camera]] needs a name and a source (this one has: {", ".join(camera)})')
    name = options.pop('name')
    args = ['--camera-name', name, '--no-window']
    for key, value in options.items():
        flag = '--' + key.replace('_', '-')
        if value is True:
            args.append(flag)
        elif value is not False:
            args += [flag, str(value).replace('{camera}', slug(name))]
    return name, args


class Camera:
    def __init__(self, name, args):
        self.name, self.args = name, args
        self.process, self.started, self.failures, self.restart_at = None, 0.0, 0, 0.0

    def start(self):
        self.process = subprocess.Popen([sys.executable, '-m', 'human_detection.app', *self.args],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        encoding='utf-8', errors='replace',
                                        env={**os.environ, 'RUN_SITE_CHILD': '1'})
        self.started = time.monotonic()
        threading.Thread(target=self.relay, daemon=True).start()
        log.info('%s: started (process %d)', self.name, self.process.pid)

    def relay(self):
        """Prefix each line the camera prints with its name. The pipe is always drained, even
        when printing fails, or the camera would block once the pipe filled up."""
        for line in self.process.stdout:
            try:
                print(f'[{self.name}] {line}', end='', flush=True)
            except (OSError, ValueError):
                pass

    def check(self, now):
        """Restart the camera if it has stopped and its wait is over."""
        if self.process and self.process.poll() is None:
            return
        if self.process:
            code = self.process.returncode
            wait = self.next_wait(now - self.started)
            self.restart_at = now + wait
            log.error('%s: stopped with exit code %s; restarting in %d s', self.name, code, wait)
            self.process = None
        if now >= self.restart_at:
            self.start()

    def next_wait(self, ran):
        """Seconds to wait before a restart: FIRST_WAIT, doubling while the camera keeps
        failing within STABLE seconds of starting."""
        self.failures = self.failures + 1 if ran < STABLE else 1
        return min(FIRST_WAIT * 2 ** (self.failures - 1), MAX_WAIT)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='Site TOML file, for example site.example.toml')
    config = tomllib.loads(Path(parser.parse_args().config).read_text())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cameras = [Camera(*camera_args(config.get('defaults', {}), camera)) for camera in config.get('camera', [])]
    if not cameras:
        parser.error('The site file has no [[camera]] tables')
    names = [camera.name for camera in cameras]
    if len(set(names)) != len(names):
        parser.error('Camera names must be unique; they name each camera\'s files and alerts')

    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    for camera in cameras:
        camera.start()
    while not stopping.wait(1):
        now = time.monotonic()
        for camera in cameras:
            camera.check(now)
    log.info('Stopping %d cameras', len(cameras))
    for camera in cameras:
        camera.stop()
    for camera in cameras:
        if camera.process:
            try:
                camera.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                camera.process.kill()
    return 0
