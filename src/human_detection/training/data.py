"""Training pictures: Open Images crops from build-child-dataset, people labelled in your
own footage with the footage command, copies degraded to look like CCTV, and a cache of
their image features."""
import csv
import zlib

import cv2
import numpy as np

from ..runtime import DATA
from .footage import labelled

CROPS = DATA / 'child_crops'
CACHE = CROPS / 'features'
FOLDS = 5


def load(name):
    with open(CROPS / f'{name}.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    images = [cv2.imread(str(CROPS / row['path'])) for row in rows]
    labels = np.array([int(row['label']) for row in rows])
    folds = np.array([zlib.crc32(row['image_id'].encode()) % FOLDS for row in rows])
    return rows, images, labels, folds


def load_site(root, test_session):
    """Labelled people from the footage command: training pictures with their labels, folds
    and cache key, the number of children among them, and the held-back session's people as
    [(is child, pictures)]. All of one person's pictures share a fold."""
    images, labels, folds, names, children, test = [], [], [], [], 0, []
    for session, person, name, paths in labelled(root):
        pictures = [cv2.imread(str(path)) for path in paths]
        if session == test_session:
            test.append((name == 'child', pictures))
            continue
        children += name == 'child'
        fold = zlib.crc32(f'{session}/{person}'.encode()) % FOLDS
        for path, picture in zip(paths, pictures, strict=True):
            images.append(picture)
            labels.append(int(name == 'child'))
            folds.append(fold)
            names.append(f'{path}:{path.stat().st_size}')
    key = f'site_{zlib.crc32("|".join(names).encode()):08x}'
    return images, np.array(labels, int), np.array(folds, int), key, children, test


def shrink(image, height):
    scale = height / image.shape[0]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image


def degrade(image, rng):
    """Make a photo crop look like a person seen by a webcam or CCTV camera:
    sometimes cut off below the chest, small, blurred, badly lit, noisy and compressed."""
    if rng.random() < 0.5:
        image = image[:max(16, int(image.shape[0] * rng.uniform(0.4, 0.7)))]
    image = shrink(image, int(rng.uniform(48, 160)))
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.5, 1.5))
    noise = rng.normal(0, rng.uniform(0, 8), image.shape)
    image = image.astype(np.float32) * rng.uniform(0.6, 1.3) + rng.uniform(-40, 30) + noise
    _, encoded = cv2.imencode('.jpg', np.clip(image, 0, 255).astype(np.uint8),
                              [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 80))])
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(image[:, ::-1] if rng.random() < 0.5 else image)


def cached_features(backbone, backbone_name, key, images):
    path = CACHE / f'{backbone_name}_{key}.npy'
    # A row count change means the dataset was rebuilt, so the cache is stale.
    if not path.is_file() or len(np.load(path)) != len(images):
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, backbone.features(images))
    return np.load(path)
