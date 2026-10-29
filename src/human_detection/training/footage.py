"""Turn footage from your own cameras into labelled pictures of people, and measure the
child classifier on them. The pictures stay on this computer.

    uv run footage extract lobby_2026-11-03_cam1_s1.mp4
    uv run footage label data/site_crops/lobby_2026-11-03_cam1_s1
    uv run footage evaluate

extract follows each person through a video with the app's detector and tracker and
saves up to --per-person pictures of them, --every seconds apart, into
data/site_crops/<video name>/ (or --out), with tracks.csv listing one row per person.

In crowds the tracker sometimes passes an ID from one person to another, so a track's
pictures are split into separate people wherever their appearance suddenly changes.

label shows each person's pictures: press C for child, A for adult, S to skip (unsure,
not a person, or still more than one person), B to go back and Q to quit. Labels are saved after every
key, so a session can be labelled in several sittings.

evaluate runs the current child classifier on every labelled person, averaging their
pictures as the app averages its readings, and reports children found and adults
flagged as children at each --child-recall setting.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from ..detection.tracking import PersonTracker
from ..labels.child import MIN_HEIGHT, Backbone, ChildClassifier, person_crop, recall_threshold
from ..runtime import DATA, MODELS, configure, pick_device
from .metrics import RECALLS

SITE = DATA / 'site_crops'
FIELDS = ['person', 'track', 'pictures', 'first_s', 'last_s', 'label']
KEYS = {ord('c'): 'child', ord('a'): 'adult', ord('s'): 'skip'}
# Consecutive pictures of one track less alike than this (cosine similarity of their image
# features) are split into two people: in crowds the tracker sometimes passes an ID from one
# person to another. On a mall clip, about 4% of consecutive pairs fell below it.
SPLIT = 0.45


def read_tracks(session):
    with open(session / 'tracks.csv', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_tracks(session, rows):
    with open(session / 'tracks.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def pictures(session, person):
    return sorted(session.glob(f'{person}_*.jpg'))


def split_points(features, limit=SPLIT):
    """Where each person starts in one track's pictures: the first picture, and every
    picture that looks too unlike the one before it."""
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    alike = (unit[1:] * unit[:-1]).sum(1)
    return [0] + [i + 1 for i, similarity in enumerate(alike) if similarity < limit]


def extract(video, out, per_person, every, model, device):
    session = Path(out) / Path(video).stem
    if (session / 'tracks.csv').is_file():
        sys.exit(f'{session} already exists; delete it to extract again (its labels would be lost).')
    session.mkdir(parents=True, exist_ok=True)
    device = pick_device(device)
    tracker = PersonTracker.load(model, device, 0.4, MODELS)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f'Cannot open {video}')
    saved = {}  # track -> [(picture path, seconds)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        for track, box in tracker.update(frame, seconds).items():
            kept = saved.setdefault(track, [])
            # The classifier never judges people shorter than MIN_HEIGHT, so don't label them.
            if (len(kept) < per_person and (not kept or seconds - kept[-1][1] >= every)
                    and box[3] - box[1] >= MIN_HEIGHT):
                path = session / f'new_{track:05d}_{len(kept):02d}.jpg'
                cv2.imwrite(str(path), person_crop(frame, box), [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept.append((path, seconds))

    head = np.load(MODELS / 'child_head.npz')
    backbone = Backbone(str(head['backbone']), device)
    rows = []
    for track, kept in sorted(saved.items()):
        if not kept:
            continue
        starts = split_points(backbone.features([cv2.imread(str(path)) for path, _ in kept])) if len(kept) > 1 else [0]
        for part, (first, last) in enumerate(zip(starts, starts[1:] + [len(kept)], strict=True)):
            person = f'{track:05d}' if len(starts) == 1 else f'{track:05d}-{part + 1}'
            for k, (path, _) in enumerate(kept[first:last]):
                path.rename(session / f'{person}_{k:02d}.jpg')
            rows.append({'person': person, 'track': track, 'pictures': last - first,
                         'first_s': round(kept[first][1], 1),
                         'last_s': round(kept[last - 1][1], 1), 'label': ''})
    write_tracks(session, rows)
    split = sum(1 for row in rows if '-' in row['person'])
    print(f'{session}: {len(rows)} people ({split} from tracks split where the person changed), '
          f'{sum(row["pictures"] for row in rows)} pictures. Next: uv run footage label {session}')


def apply_key(rows, index, key):
    """Label the person at `index` for a key press; return the next index, or None to stop."""
    if key in KEYS:
        rows[index]['label'] = KEYS[key]
        later = [i for i in range(index + 1, len(rows)) if not rows[i]['label']]
        return later[0] if later else None
    if key == ord('b'):
        return max(index - 1, 0)
    if key in (ord('q'), 27):
        return None
    return index


def sheet(session, row, position, total):
    tiles = [cv2.imread(str(path)) for path in pictures(session, row['person'])]
    tiles = [cv2.resize(t, (max(1, round(240 * t.shape[1] / t.shape[0])), 240)) for t in tiles if t is not None]
    strip = np.hstack([np.pad(t, ((0, 0), (0, 6), (0, 0)), constant_values=255) for t in tiles])
    if strip.shape[1] > 1600:
        strip = cv2.resize(strip, (1600, round(strip.shape[0] * 1600 / strip.shape[1])))
    header = np.full((70, max(strip.shape[1], 900), 3), 30, np.uint8)
    label = row['label'] or 'not labelled'
    cv2.putText(header, f'Person {position + 1} of {total} ({row["person"]}): {label}', (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(header, 'C child   A adult   S skip (unsure, or more than one person)   B back   Q quit', (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 255), 1)
    body = np.full((strip.shape[0], header.shape[1], 3), 255, np.uint8)
    body[:, :strip.shape[1]] = strip
    return np.vstack([header, body])


def label(session):
    session = Path(session)
    rows = read_tracks(session)
    pending = [i for i, row in enumerate(rows) if not row['label']]
    index = pending[0] if pending else None
    title = f'Label {session.name}'
    while index is not None:
        cv2.imshow(title, sheet(session, rows[index], index, len(rows)))
        key = cv2.waitKey(0) & 0xFF
        index = apply_key(rows, index, key)
        write_tracks(session, rows)
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    counts = {name: sum(row['label'] == name for row in rows) for name in ('child', 'adult', 'skip', '')}
    print(f'{session.name}: {counts["child"]} children, {counts["adult"]} adults, {counts["skip"]} skipped, '
          f'{counts[""]} still to label')


def labelled(root):
    """[(session, track, label, [picture paths])] for every person labelled child or adult."""
    people = []
    for tracks_file in sorted(Path(root).rglob('tracks.csv')):
        session = tracks_file.parent
        for row in read_tracks(session):
            if row['label'] in ('child', 'adult'):
                people.append((session.name, row['person'], row['label'], pictures(session, row['person'])))
    return people


def evaluate(root, device):
    people = labelled(root)
    if not people:
        sys.exit(f'No labelled people under {root}; run extract and label first.')
    classifier = ChildClassifier(MODELS / 'child_head.npz', pick_device(device))
    head = np.load(MODELS / 'child_head.npz')
    averages, is_child, sessions = [], [], []
    for session, _, name, paths in people:
        crops = [image for image in (cv2.imread(str(path)) for path in paths) if image is not None]
        averages.append(float(np.mean(classifier.crop_probabilities(crops))))
        is_child.append(name == 'child')
        sessions.append(session)
    averages, is_child = np.array(averages), np.array(is_child)
    print(f'{len(averages)} people in {len(set(sessions))} sessions: '
          f'{is_child.sum()} children, {(~is_child).sum()} adults')
    print(f'{"--child-recall":>14}  {"children found":>15}  {"adults flagged as children":>27}')
    for recall in RECALLS:
        threshold = recall_threshold(head['child_scores'], recall)
        found = f'{(averages[is_child] >= threshold).mean():.0%}' if is_child.any() else '-'
        flagged = f'{(averages[~is_child] >= threshold).mean():.0%}' if (~is_child).any() else '-'
        print(f'{recall:>14.2f}  {found:>15}  {flagged:>27}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('extract', help='Save pictures of each person in a video')
    p.add_argument('video')
    p.add_argument('--out', default=str(SITE), help='Folder to put the session folder in')
    p.add_argument('--per-person', type=int, default=10, help='Most pictures to keep of one person')
    p.add_argument('--every', type=float, default=1.0, help='Seconds between pictures of one person')
    p.add_argument('--model', default='nano', help='RF-DETR size, as live-camera --model')
    p.add_argument('--device', default='auto')
    p = commands.add_parser('label', help='Label each person child or adult')
    p.add_argument('session', help='A folder made by extract')
    p = commands.add_parser('evaluate', help='Measure the current child classifier on labelled people')
    p.add_argument('root', nargs='?', default=str(SITE), help='A session folder, or the folder holding them')
    p.add_argument('--device', default='auto')
    args = parser.parse_args()
    configure()
    if args.command == 'extract':
        extract(args.video, args.out, args.per_person, args.every, args.model, args.device)
    elif args.command == 'label':
        label(args.session)
    else:
        evaluate(args.root, args.device)
