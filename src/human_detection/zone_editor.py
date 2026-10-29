"""Draw zones and counting lines on a camera's view and save them for `live-camera --zones`.

Left-click to add corners and right-click to undo one. Press Enter to finish: two points
make a counting line, three or more a zone; then answer its name (and a zone's type) in
this terminal. A line's arrow shows its "in" side; click its points in the other order to
flip it. Press S to save and quit, Q to quit without saving. Mark floor areas: people are
placed where they stand.

    uv run draw-zones --source 0 --out zones.json
    uv run draw-zones --image snapshot.jpg --out lobby.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .app.overlay import ZONE_COLORS, draw_zones
from .camera import open_capture, redact
from .safety.zones import DEFAULTS, ZONE_TYPES, parse_zone

TITLE = 'Draw zones - Enter: finish zone, S: save, Q: quit'


def grab(source):
    """A frame from the camera, opened as the app opens it, so zones line up with its view."""
    source = int(source) if source.isdecimal() else source
    cap = open_capture(source)
    frame = None
    if cap is not None:
        # Webcams need a few frames to settle their exposure.
        for _ in range(15):
            ok, image = cap.read()
            if ok:
                frame = image
        cap.release()
    if frame is None:
        raise SystemExit(f'Could not read a frame from {redact(source)}')
    return frame


def ask_zone(points, width, height):
    if len(points) == 2:
        kind = 'line'
    else:
        types = {kind[0]: kind for kind in ZONE_TYPES if kind != 'line'}
        choice = input('Zone type - [h]azard, [e]xit, [c]ount, [i]gnore (blank to discard): ').strip().lower()[:1]
        if choice not in types:
            return None
        kind = types[choice]
    zone = {'name': input(f'{"Line" if kind == "line" else "Zone"} name: ').strip() or f'{kind} zone', 'type': kind,
            'polygon': [[round(x / width, 4), round(y / height, 4)] for x, y in points]}
    if kind == 'count':
        limit = input('Max children (blank for no limit): ').strip()
        if limit:
            zone['max_children'] = int(limit)
        ratio = input('Max children per adult (blank for no limit): ').strip()
        if ratio:
            zone['max_children_per_adult'] = float(ratio)
    if kind == 'hazard':
        who = input('Alert on [c]hildren, children with [n]o adult nearby, or [a]nyone? [c]: ').strip().lower()[:1]
        if who == 'a':
            zone['anyone'] = True
        elif who == 'n':
            zone['unaccompanied'] = True
    if kind in ('hazard', 'exit'):
        wait = input('Seconds someone must stay in it before an alert (blank for at once): ').strip()
        try:
            seconds = float(wait) if wait else 0
        except ValueError:
            seconds = -1
        if seconds > 0:
            zone['after_seconds'] = seconds
        elif seconds < 0:
            print(f'"{wait}" is not a number of seconds; the zone alerts at once instead.')
        text = input('Alert text, e.g. "Child near the stairs" (blank for the default): ').strip()
        if text:
            zone['alert'] = text
    if kind != 'line':
        hours = input('Active hours, e.g. 20:00-07:00 (blank for always): ').strip()
        if hours:
            zone['hours'] = [span.strip() for span in hours.split(',')]
            try:
                parse_zone(zone)
            except ValueError as error:
                print(f'{error}; the zone is always active instead.')
                del zone['hours']
    print(f'{"Line" if kind == "line" else "Zone"} added. Draw another, or press S in the window to save.')
    return zone


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='0', help='Webcam index, video path, or RTSP URL')
    parser.add_argument('--image', help='Draw on this image instead of a camera frame')
    parser.add_argument('--out', default='zones.json', help='Zones file to create or extend')
    args = parser.parse_args()

    frame = cv2.imread(args.image) if args.image else grab(args.source)
    if frame is None:
        raise SystemExit(f'Could not read {args.image}')
    height, width = frame.shape[:2]
    out = Path(args.out)
    config = json.loads(out.read_text()) if out.is_file() else {**DEFAULTS, 'zones': []}
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(TITLE)
    cv2.setMouseCallback(TITLE, on_mouse)
    print(__doc__.split('\n\n')[1])
    while True:
        view = frame.copy()
        draw_zones(view, [parse_zone(raw) for raw in config['zones']])
        if points:
            cv2.polylines(view, [np.array(points, np.int32)], False, ZONE_COLORS['hazard'], 2)
            for point in points:
                cv2.circle(view, point, 5, (255, 255, 255), -1)
        cv2.imshow(TITLE, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13) and len(points) >= 2:
            zone = ask_zone(points, width, height)
            if zone:
                config['zones'].append(zone)
            points.clear()
        elif key == ord('s'):
            out.write_text(json.dumps(config, indent=2) + '\n')
            print(f'Saved {len(config["zones"])} zones to {out}')
            break
        elif key in (ord('q'), 27):
            break
    cv2.destroyAllWindows()
