"""Download every model the app needs, so it can then run without internet: the person
detector (RF-DETR, checked against its published MD5), the child classifier's image
encoder (DINOv2), and, unless --skip-age, the public face-detection and age-only weights
for --age. Never executes downloads."""
import argparse
import hashlib
import shutil
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from .detection.detector import SIZES, import_rfdetr
from .labels.child import BACKBONES
from .runtime import MODELS, configure

ATTEMPTS = 20  # the detector is about 400 MB, so a slow connection may drop several times
# The --age networks' weights; their layer definitions ship with the code.
FILES = {
    'age_net.caffemodel': (
        'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel',
        45661480),
    'opencv_face_detector_uint8.pb': (
        'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/'
        'opencv_face_detector_uint8.pb', 2727750),
}


def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def download_detector(filename):
    """Fetch RF-DETR weights, resuming where a dropped connection left off."""
    import_rfdetr()
    from rfdetr.assets.model_weights import ModelWeights
    asset = ModelWeights.from_filename(filename)
    path = MODELS / filename
    if path.is_file() and asset.md5_hash in (None, md5(path)):
        print(f'{filename}: already downloaded')
        return
    partial = path.with_suffix(path.suffix + '.part')
    for _ in range(ATTEMPTS):
        done = partial.stat().st_size if partial.is_file() else 0
        print(f'Downloading {filename}' + (f', resuming at {done >> 20} MB' if done else '') + '...')
        try:
            with urlopen(Request(asset.url, headers={'Range': f'bytes={done}-'}), timeout=60) as response:
                # 206 continues the partial file; 200 means the server sent it all again.
                with partial.open('ab' if response.status == 206 else 'wb') as output:
                    shutil.copyfileobj(response, output)
            break
        except HTTPError as error:
            if error.code == 416:  # nothing left to send: the partial file is complete
                break
            raise
        except OSError as error:
            print(f'  connection dropped ({error}); retrying')
    else:
        raise RuntimeError(f'Could not download {filename} after {ATTEMPTS} attempts; run this again to resume')
    if asset.md5_hash and md5(partial) != asset.md5_hash:
        partial.unlink()
        raise RuntimeError(f'{filename} failed its MD5 check; deleted it, run this again')
    partial.replace(path)


def download_backbone():
    """The child classifier's image encoder, into the cache the app uses."""
    from transformers import AutoImageProcessor, AutoModel
    name = BACKBONES[str(np.load(MODELS / 'child_head.npz')['backbone'])]
    AutoImageProcessor.from_pretrained(name)
    AutoModel.from_pretrained(name)
    print(f'{name}: ready')


def download_age_models():
    for name, (url, expected_size) in FILES.items():
        path = MODELS / name
        if path.exists() and path.stat().st_size == expected_size:
            print(f'{name}: already downloaded')
            continue
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            print(f'Downloading {name}...')
            with urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(f'Unexpected file size for {name}; download incomplete or upstream changed')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', default='nano', choices=list(SIZES),
                        help='RF-DETR size to download, matching live-camera --model')
    parser.add_argument('--skip-age', action='store_true',
                        help='Skip the face-age model for --age, which has no licence for commercial use')
    args = parser.parse_args()
    configure()  # the app's model cache
    MODELS.mkdir(exist_ok=True)
    download_detector(SIZES[args.detector][1])
    download_backbone()
    if not args.skip_age:
        download_age_models()
    print('Models ready.')
