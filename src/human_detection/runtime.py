"""The project's folders, and the environment and device its models run with."""
import logging
import os
from pathlib import Path

# The project folder that holds src/; models, data and caches stay inside it.
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'models'
DATA = ROOT / 'data'
CACHE = ROOT / '.runtime'


def configure():
    """Keep downloaded weights and library settings inside this project, without library
    notes that are not for users. Call it before transformers is imported, which fixes its
    cache folder then."""
    for directory in (CACHE / 'matplotlib', CACHE / 'cache'):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(CACHE / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(CACHE / 'cache'))
    os.environ.setdefault('HF_HOME', str(CACHE / 'huggingface'))
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')  # its notes about training settings are noise here
    # Torch's note that redirects are not supported on macOS.
    logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').setLevel(logging.ERROR)


def pick_device(device):
    """`device`, or for 'auto' the fastest one present: an NVIDIA GPU, Apple Silicon, else the CPU."""
    if device != 'auto':
        return device
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'
