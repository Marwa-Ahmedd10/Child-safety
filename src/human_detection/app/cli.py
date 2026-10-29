"""Count currently visible people from a webcam, video or RTSP camera.
Optional: label children from the whole body (--child), or apparent age groups from
faces (--age)."""
import argparse
import logging
import os
import signal

from ..detection.detector import SIZES
from ..runtime import configure
from .pipeline import CameraApp

log = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='live-camera', description=__doc__)
    parser.add_argument('--source', default='0', help='Webcam index, video path, or RTSP URL')
    parser.add_argument('--model', default='nano', choices=list(SIZES),
                        help='RF-DETR person detector size; larger is more accurate and slower (see README)')
    parser.add_argument('--conf', type=float, default=0.4, help='Minimum detection score to start counting a person')
    parser.add_argument('--device', default='auto', help='auto, cpu, mps (Apple Silicon), cuda or cuda:N')
    parser.add_argument('--hold', type=int, default=10, help='Keep a briefly missed person counted for N frames')
    parser.add_argument('--max-frames', type=int, default=0, help='Stop after N frames; 0 means unlimited')
    parser.add_argument('--no-window', action='store_true', help='Print counts without a preview window')
    labelling = parser.add_mutually_exclusive_group()
    labelling.add_argument('--child', action='store_true',
                           help='Label children from their whole body; works with ceiling cameras')
    labelling.add_argument('--age', action='store_true', help='Estimate apparent age groups from visible faces')
    parser.add_argument('--child-recall', type=float, default=0.95,
                        help='Share of children --child should find; lower flags fewer adults (see README)')
    parser.add_argument('--age-conf', type=float, default=0.6, help='Minimum age group score')
    parser.add_argument('--zones', help='Zones and counting lines JSON (see zones.example.json); turns on --child '
                                        'when a zone needs child labels')
    parser.add_argument('--events', help='Append safety alerts to this JSON-lines file')
    parser.add_argument('--webhook', help='POST each safety alert as JSON to this URL')
    parser.add_argument('--camera-name', help='Camera name in alerts (default: the source)')
    parser.add_argument('--counts', help='Append visitor counts at counting lines to this JSON-lines file')
    parser.add_argument('--count-every', type=float, default=60, help='Minutes per visitor-count record')
    parser.add_argument('--status', help='Rewrite this JSON file every 5 s with the camera\'s health, for monitoring')
    args = parser.parse_args(argv)
    # Camera passwords and webhook tokens can stay out of files and shell history:
    # 'env:NAME' reads the value from the environment variable NAME.
    for option in ('source', 'webhook'):
        value = getattr(args, option)
        if value and value.startswith('env:'):
            if not os.environ.get(value[4:]):
                parser.error(f'--{option} {value}: environment variable {value[4:]} is not set')
            setattr(args, option, os.environ[value[4:]].strip())
    if args.webhook and not args.webhook.lower().startswith(('http://', 'https://')):
        parser.error('--webhook must be an http:// or https:// URL')
    if (not 0 < args.conf <= 1 or args.max_frames < 0 or args.hold < 0
            or not 0 < args.age_conf <= 1 or not 0.5 <= args.child_recall < 1 or args.count_every <= 0):
        parser.error('conf must be in (0, 1], max-frames and hold >= 0, age-conf in (0, 1], '
                     'child-recall in [0.5, 1) and count-every > 0')
    if args.zones:
        if args.age:
            parser.error('--zones uses whole-body child labels, so it cannot be combined with --age')
    elif args.counts:
        parser.error('--counts needs --zones with a counting line')
    return args


def stop(signum, frame):
    """Docker and systemd stop a service with SIGTERM; treat it like Ctrl-C, so open
    alerts are closed and counts saved."""
    raise KeyboardInterrupt


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    signal.signal(signal.SIGTERM, stop)
    if os.environ.get('RUN_SITE_CHILD'):
        # run-site passes a stop on as SIGTERM; a terminal's Ctrl-C also reaches this
        # process directly, and the second signal would cut its cleanup short.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        configure()
        CameraApp(args).run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        log.error('%s', exc)
        return 1
    return 0
