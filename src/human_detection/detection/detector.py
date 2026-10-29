"""The person detector, RF-DETR. The sizes offered here are Apache-2.0, so they can be
used commercially."""
import logging
import os
import warnings
from pathlib import Path

# Apache-2.0 RF-DETR sizes: model class and weights file. Other sizes use another licence.
SIZES = {
    'nano': ('RFDETRNano', 'rf-detr-nano.pth'),
    'small': ('RFDETRSmall', 'rf-detr-small.pth'),
    'medium': ('RFDETRMedium', 'rf-detr-medium.pth'),
    'large': ('RFDETRLarge', 'rf-detr-large-2026.pth'),
}
PERSON = 1  # COCO category id
log = logging.getLogger(__name__)


def import_rfdetr():
    """Import RF-DETR without its notes about its DINOv2 backbone and Torch's about
    TorchScript, which are not for users. RF-DETR reads its log level from LOG_LEVEL;
    set it to see them."""
    os.environ.setdefault('LOG_LEVEL', 'ERROR')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        import rfdetr
    return rfdetr


def load_detector(size, device, models):
    """An RF-DETR model, whose predict(rgb_image, threshold) returns supervision
    Detections; its weights download into `models` if missing."""
    rfdetr = import_rfdetr()
    import torch
    name, weights = SIZES[size]
    model = getattr(rfdetr, name)(pretrain_weights=str(Path(models) / weights), device=device)
    # Tracing the model, in half precision on a GPU, made it 1.5x faster on an M1 Mac
    # without changing its detections; GPUs run it faster still.
    try:
        with warnings.catch_warnings():  # tracing warns about shapes it fixes; the input size is fixed
            warnings.simplefilter('ignore')
            model.inference(dtype=torch.float32 if device == 'cpu' else torch.float16)
    except Exception as error:
        log.warning('Running the detector unoptimized (%s)', error)
    return model
