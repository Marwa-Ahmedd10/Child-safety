"""Apparent age groups from faces (--age). Its age model has no licence for commercial
use, so this is for research only. No identity or gender inference."""
from pathlib import Path

import cv2
import numpy as np

from .votes import Labeller, TrackVotes

# The networks' layer definitions ship with the code; download-models fetches their weights.
NETWORKS = Path(__file__).parent / 'networks'
# The pretrained model has coarse, discontinuous age bands. Its 15–20 band
# crosses the child/adult boundary, so that band stays Unknown.
GROUPS = ('Child', 'Child', 'Child', 'Unknown', 'Adult', 'Adult', 'Adult', 'Older')
NAMES = ('Child', 'Adult', 'Older', 'Unknown')
MEAN = (78.4263377603, 87.7689143744, 114.895847746)
FACE_CONF = 0.8
MIN_FACE = 40   # pixels; smaller faces are too blurry for the age model
HEAD = 0.6      # a person's face must sit in the top 60% of their box
MARGIN = 0.25   # context kept around a face, relative to its size


def age_group(scores, threshold):
    """Pick the group with the most total score, so a face split between
    adjacent adult bands (e.g. 25–32 and 38–43) still counts as Adult."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != 8 or not np.isfinite(scores).all():
        return 'Unknown'
    totals = {}
    for group, score in zip(GROUPS, scores, strict=True):
        totals[group] = totals.get(group, 0.0) + score
    best = max(totals, key=totals.get)
    return best if totals[best] >= threshold else 'Unknown'


def head_region(box):
    x1, y1, x2, y2 = box
    return x1, y1, x2, y1 + (y2 - y1) * HEAD


def owner(face, people):
    """Index of the only person whose head region holds the face centre, else None."""
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    matches = []
    for i, box in enumerate(people):
        hx1, hy1, hx2, hy2 = head_region(box)
        if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
            matches.append(i)
    # Ambiguous overlapping people stay Unknown.
    return matches[0] if len(matches) == 1 else None


class AgeEstimator:
    def __init__(self, models):
        required = (Path(models) / 'opencv_face_detector_uint8.pb', NETWORKS / 'opencv_face_detector.pbtxt',
                    Path(models) / 'age_net.caffemodel', NETWORKS / 'deploy_age.prototxt')
        for path in required:
            if not path.is_file():
                raise RuntimeError(f'Missing age model file: {path}')
        self.face = cv2.dnn.readNetFromTensorflow(str(required[0]), str(required[1]))
        self.age = cv2.dnn.readNetFromCaffe(str(required[3]), str(required[2]))

    def estimate(self, frame, people, wanted):
        """Age-band scores for each wanted person index with one clear face."""
        return {i: self._age(frame, face) for i, face in self._faces(frame, people, wanted).items()}

    def _faces(self, frame, people, wanted):
        # Search each person's head region rather than the whole frame: the
        # detector sees 300x300 pixels, so distant faces stay large enough to find.
        height, width = frame.shape[:2]
        found = {}
        for i in wanted:
            x1, y1, x2, y2 = head_region(people[i])
            pad = (x2 - x1) * 0.1
            x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
            x2, y2 = min(width, int(x2 + pad)), min(height, int(y2))
            if min(x2 - x1, y2 - y1) < MIN_FACE:
                continue
            faces = [face for face in self._detect(frame[y1:y2, x1:x2], x1, y1)
                     if min(face[2] - face[0], face[3] - face[1]) >= MIN_FACE
                     and owner(face, people) == i]
            # Several faces in one person's region stay Unknown.
            if len(faces) == 1:
                found[i] = faces[0]
        return found

    def _detect(self, image, dx, dy):
        height, width = image.shape[:2]
        self.face.setInput(cv2.dnn.blobFromImage(image, 1, (300, 300), (104, 117, 123),
                                                 swapRB=False, crop=False))
        faces = []
        for detection in self.face.forward().reshape(-1, 7):
            if detection[2] < FACE_CONF:
                continue
            x1, y1, x2, y2 = (detection[3:7] * [width, height, width, height]).astype(int)
            faces.append((max(0, x1) + dx, max(0, y1) + dy,
                          min(width, x2) + dx, min(height, y2) + dy))
        return faces

    def _age(self, frame, face):
        x1, y1, x2, y2 = face
        height, width = frame.shape[:2]
        # Margin scales with the face so near and far faces get the same context.
        pad = int(MARGIN * max(x2 - x1, y2 - y1))
        crop = frame[max(0, y1 - pad):min(height, y2 + pad),
                     max(0, x1 - pad):min(width, x2 + pad)]
        self.age.setInput(cv2.dnn.blobFromImage(crop, 1, (227, 227), MEAN,
                                                swapRB=False, crop=False))
        return self.age.forward().reshape(-1)


class AgeVotes(TrackVotes):
    """Average each tracked person's last 15 face readings."""

    def __init__(self, threshold=0.6, window=15, refresh=3):
        super().__init__(window, refresh)
        self.threshold = threshold

    def add(self, track_id, frame_number, scores=None):
        """Record a face search for this person; scores is None when no face was found."""
        if scores is not None:
            scores = np.asarray(scores, dtype=float).reshape(-1)
            if len(scores) != 8 or not np.isfinite(scores).all():
                scores = None
        self.record(track_id, frame_number, scores)

    def label(self, track_id):
        votes = self.readings(track_id)
        return age_group(np.mean(votes, axis=0), self.threshold) if votes else 'Unknown'


def age_labeller(models, min_score):
    """An age group for each person with a clear face; Unknown until then."""
    # OpenCV runs the age models on the CPU; capping its threads stops them
    # competing with PyTorch and roughly halves their time per call.
    cv2.setNumThreads(2)
    return Labeller('age', NAMES, AgeEstimator(models).estimate, AgeVotes(min_score))
