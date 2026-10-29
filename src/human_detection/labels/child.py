"""Child/adult classification from a person's whole body, so it still works with
ceiling cameras where faces are small or hidden. No identity or gender inference."""
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .votes import Labeller, TrackVotes

# Frozen image encoders with commercial-use licences (Apache-2.0).
BACKBONES = {
    'siglip2': 'google/siglip2-base-patch16-224',
    'dinov2': 'facebook/dinov2-small',
    'dinov2-base': 'facebook/dinov2-base',
}
SIZE = 224
PAD = 0.05  # context around a person box, in the app and in the training crops alike
MIN_HEIGHT = 64  # smallest person height in pixels that training measured accuracy at
LABELS = ('Child', 'Adult', 'Checking')


def person_crop(frame, box):
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    height, width = frame.shape[:2]
    return frame[max(0, int(y1 - py)):min(height, int(y2 + py)),
                 max(0, int(x1 - px)):min(width, int(x2 + px))]


def square(crop, size=SIZE):
    """Pad to a square before resizing: squashing or centre-cropping a tall
    person would distort the body proportions that separate children from adults."""
    height, width = crop.shape[:2]
    side = max(height, width)
    canvas = np.full((side, side, 3), 127, np.uint8)
    y, x = (side - height) // 2, (side - width) // 2
    canvas[y:y + height, x:x + width] = crop
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


class Backbone:
    """Frozen image encoder that turns BGR person crops into feature vectors."""

    def __init__(self, name, device):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from transformers.utils import logging
        logging.disable_progress_bar()
        self.torch = torch
        self.device = device
        processor = AutoImageProcessor.from_pretrained(BACKBONES[name])
        self.mean = np.array(processor.image_mean, np.float32)
        self.std = np.array(processor.image_std, np.float32)
        model = AutoModel.from_pretrained(BACKBONES[name])
        # SigLIP also has a text tower; only the image side is needed.
        self.model = getattr(model, 'vision_model', model).to(device).eval()

    def features(self, crops, batch=32):
        out = []
        for start in range(0, len(crops), batch):
            images = np.stack([square(crop) for crop in crops[start:start + batch]])
            images = (images[..., ::-1].astype(np.float32) / 255 - self.mean) / self.std
            pixels = self.torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(self.device)
            with self.torch.no_grad():
                out.append(self.model(pixel_values=pixels).pooler_output.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def recall_threshold(child_scores, recall):
    """The P(child) that found `recall` of children, from their cross-validated logits saved
    with the head: higher recall misses fewer children but flags more adults."""
    return float(1 / (1 + np.exp(-np.quantile(child_scores, 1 - recall))))


class ChildClassifier:
    """P(child) for person boxes: a frozen backbone plus a logistic-regression
    head trained by train-child-classifier."""

    def __init__(self, head_path, device, recall=0.95):
        head_path = Path(head_path)
        if not head_path.is_file():
            raise RuntimeError(f'Missing child classifier head: {head_path}. Run train-child-classifier.')
        head = np.load(head_path)
        self.backbone = Backbone(str(head['backbone']), device)
        self.mean, self.scale = head['mean'], head['scale']
        self.weights, self.bias = head['weights'], float(head['bias'])
        self.threshold = recall_threshold(head['child_scores'], recall)

    def probabilities(self, frame, boxes):
        return self.crop_probabilities([person_crop(frame, box) for box in boxes])

    def crop_probabilities(self, crops):
        """P(child) for pictures already cut out with person_crop."""
        if not crops:
            return np.zeros(0)
        logits = (self.backbone.features(crops) - self.mean) / self.scale @ self.weights + self.bias
        return 1 / (1 + np.exp(-logits))


class ChildVotes(TrackVotes):
    """Average each tracked person's recent child probabilities. A person stays
    'Checking' until they have `settle` readings; safety rules treat that as a
    possible child."""

    # A child must never inherit an adult's label, so a continued ID starts at Checking.
    carry_over = False

    def __init__(self, threshold, window=10, refresh=3, settle=3):
        super().__init__(window, refresh)
        self.threshold = threshold
        self.settle = settle

    def add(self, track_id, frame_number, probability=None):
        """Record a reading for this person; probability is None when they were too small to judge."""
        self.record(track_id, frame_number, None if probability is None else float(probability))

    def label(self, track_id):
        votes = self.readings(track_id)
        if len(votes) < self.settle:
            return 'Checking'
        return 'Child' if np.mean(votes) >= self.threshold else 'Adult'

    def confidence(self, track_id):
        """The average P(child) behind a Child label, or 1 minus it behind an Adult one;
        None while Checking. It is not a calibrated probability."""
        votes = self.readings(track_id)
        if len(votes) < self.settle:
            return None
        probability = float(np.mean(votes))
        return probability if probability >= self.threshold else 1 - probability


def fingerprint(path):
    """The first 12 hex digits of a file's SHA-256, to tell model versions apart."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def child_labeller(head_path, device, recall):
    """Child, Adult or Checking for each person, from their whole body."""
    classifier = ChildClassifier(head_path, device, recall)

    def read(frame, boxes, wanted):
        # A smaller box, such as a head at the frame edge, shows too little to judge:
        # no reading, so the person keeps the label from better views or stays Checking.
        wanted = [i for i in wanted if boxes[i][3] - boxes[i][1] >= MIN_HEIGHT]
        return dict(zip(wanted, classifier.probabilities(frame, [boxes[i] for i in wanted]), strict=True))

    models = {'child_classifier': f'child_head {fingerprint(head_path)}', 'child_recall': recall}
    return Labeller('child', LABELS, read, ChildVotes(classifier.threshold), models)
