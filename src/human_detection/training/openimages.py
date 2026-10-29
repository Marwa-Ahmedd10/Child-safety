"""Build child/adult person crops from Open Images to train and test the child classifier.

Open Images "Boy"/"Girl" boxes often mark young adults, so child boxes are only taken
from photos a person verified as showing a child, toddler or baby, and adult
("Man"/"Woman") boxes only from photos not verified as showing one. Gender is
discarded. Photos are split 70/30 into train and test, so none appears in both.

Open Images annotations are CC BY 4.0 and its images are listed as CC BY 2.0;
check each image's licence (saved in the index) before redistributing crops.

    uv run build-child-dataset
"""
import argparse
import csv
import random
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ..labels.child import PAD
from ..runtime import DATA
from .data import CROPS

SOURCE = DATA / 'openimages'
SPLITS = {'validation': 'val', 'test': 'test'}
CSV_URLS = {
    'validation-annotations-bbox.csv': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
    'test-annotations-bbox.csv': 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv',
    'validation-images-with-rotation.csv': 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv',
    'test-images-with-rotation.csv': 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv',
    'oidv7-val-annotations-human-imagelabels.csv': 'https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv',
    'oidv7-test-annotations-human-imagelabels.csv': 'https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv',
}
IMAGE_URL = 'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg'
LABELS = {'/m/01bl7v': 1, '/m/05r655': 1, '/m/04yx4': 0, '/m/03bt1vf': 0}  # Boy, Girl, Man, Woman
YOUNG = {'/m/0ytgt', '/m/01bgsw', '/m/0jnvp'}  # Child, Toddler, Baby
TEST_SHARE = 30     # percent of photos held out for testing
MIN_SHARE = 0.15    # box height as a share of the image's
MIN_PIXELS = 48     # crop height in pixels
PER_IMAGE = 2       # boxes per class per photo, so a few crowded photos do not dominate


def fetch_csvs():
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name, url in CSV_URLS.items():
        if not (SOURCE / name).is_file():
            print(f'Downloading {name}...')
            urllib.request.urlretrieve(url, SOURCE / name)


def read_csv(name):
    with open(SOURCE / name, newline='', encoding='utf-8') as f:
        yield from csv.DictReader(f)


def candidates(split):
    """{image_id: (info, [(label, box)])} for usable boxes in one Open Images split."""
    meta = {row['ImageID']: row for row in read_csv(f'{split}-images-with-rotation.csv')}
    young = {row['ImageID'] for row in read_csv(f'oidv7-{SPLITS[split]}-annotations-human-imagelabels.csv')
             if row['LabelName'] in YOUNG and row['Confidence'] == '1'}
    found = defaultdict(list)
    for row in read_csv(f'{split}-annotations-bbox.csv'):
        label = LABELS.get(row['LabelName'])
        info = meta.get(row['ImageID'])
        # Skip groups, drawings and statues, small people, and images that
        # need (or may need) rotating before their boxes line up.
        if (label is None or info is None or row['IsGroupOf'] != '0' or row['IsDepiction'] != '0'
                or float(row['YMax']) - float(row['YMin']) < MIN_SHARE
                or info['Rotation'] not in ('0', '0.0')):
            continue
        # Child boxes need a photo verified to show a child; adult boxes need one that is not.
        if (label == 1) != (row['ImageID'] in young):
            continue
        found[row['ImageID']].append((label, row))
    return {image_id: (dict(info=meta[image_id], split=split), boxes) for image_id, boxes in found.items()}


def select(pool, seed):
    """Every child box (up to PER_IMAGE a photo) plus as many adult boxes."""
    images = sorted(pool)
    random.Random(seed).shuffle(images)
    chosen = defaultdict(list)
    children = 0
    for image_id in images:
        picks = [box for box in pool[image_id][1] if box[0] == 1][:PER_IMAGE]
        chosen[image_id].extend(picks)
        children += len(picks)
    adults = 0
    for image_id in images:
        if adults >= children:
            break
        picks = [box for box in pool[image_id][1] if box[0] == 0][:min(PER_IMAGE, children - adults)]
        chosen[image_id].extend(picks)
        adults += len(picks)
    return {image_id: (pool[image_id][0], picks) for image_id, picks in chosen.items() if picks}


def crop_image(out, image_id, source, picks):
    url = IMAGE_URL.format(split=source['split'], image_id=image_id)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except OSError as exc:
        return [], f'{image_id}: {exc}'
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        return [], f'{image_id}: not an image'
    height, width = image.shape[:2]
    rows = []
    for k, (label, box) in enumerate(picks):
        x1, x2 = float(box['XMin']) * width, float(box['XMax']) * width
        y1, y2 = float(box['YMin']) * height, float(box['YMax']) * height
        px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
        x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        x2, y2 = min(width, int(x2 + px)), min(height, int(y2 + py))
        if y2 - y1 < MIN_PIXELS or x2 - x1 < MIN_PIXELS // 3:
            continue
        crop = image[y1:y2, x1:x2]
        scale = 320 / max(crop.shape[:2])
        if scale < 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        path = out / f'{image_id}_{k}.jpg'
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        info = source['info']
        rows.append({'path': str(path.relative_to(CROPS)), 'label': label, 'image_id': image_id,
                     'height_px': y2 - y1, 'occluded': box['IsOccluded'], 'truncated': box['IsTruncated'],
                     'license': info['License'], 'author': info['Author'],
                     'source': info['OriginalLandingURL']})
    return rows, None


def build(name, chosen, workers):
    out = CROPS / name
    out.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    with ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(crop_image, out, image_id, source, picks)
                for image_id, (source, picks) in chosen.items()]
        for done, job in enumerate(jobs, 1):
            new_rows, error = job.result()
            rows.extend(new_rows)
            if error:
                errors.append(error)
            if done % 250 == 0:
                print(f'  {name}: {done}/{len(jobs)} images', flush=True)
    with open(CROPS / f'{name}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    children = sum(row['label'] for row in rows)
    print(f'{name}: saved {children} child and {len(rows) - children} adult crops; {len(errors)} images failed')
    for error in errors[:5]:
        print('  ', error)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    fetch_csvs()
    pool = {}
    for split in SPLITS:
        pool.update(candidates(split))
    sets = {'train': {}, 'test': {}}
    for image_id, entry in pool.items():
        name = 'test' if zlib.crc32(image_id.encode()) % 100 < TEST_SHARE else 'train'
        sets[name][image_id] = entry
    for name, images in sets.items():
        chosen = select(images, args.seed)
        print(f'{name}: {len(chosen)} photos selected')
        build(name, chosen, args.workers)
