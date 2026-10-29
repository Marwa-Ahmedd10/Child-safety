"""Train and benchmark the whole-body child classifier on Open Images crops.

Each frozen backbone gets a logistic-regression head chosen by 5-fold cross-validation
grouped by photo; the backbone with the best cross-validated AUC is saved to
models/child_head.npz. The held-out test set is only used for reporting. The face-based
age pipeline is scored on the same test crops for comparison.

    uv run build-child-dataset
    uv run train-child-classifier

Footage from your own cameras, labelled with the footage command, is added with --site;
hold one session back with --test-session for an honest measure from above:

    uv run train-child-classifier --site data/site_crops --test-session lobby_2026-11-05_cam1_s3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..labels.child import BACKBONES, Backbone, recall_threshold
from ..labels.face_age import AgeEstimator, age_group
from ..runtime import MODELS, configure, pick_device
from .data import cached_features, degrade, load, load_site, shrink
from .head import cross_validate, logits, train_head
from .metrics import RECALLS, at_threshold, auc, summary

TARGET_RECALL = 0.95
# Once the footage holds this many children, --child-recall thresholds come from them alone.
MIN_SITE_CHILDREN = 10
LOW_RES = (96, 64)  # person height in pixels, like people far from a CCTV camera
PROMPTS = {
    1: ['a photo of a child.', 'a photo of a kid.', 'a photo of a young child.', 'a photo of a toddler.',
        'a photo of a schoolchild.'],
    0: ['a photo of an adult.', 'a photo of a grown-up.', 'a photo of an adult person.',
        'a photo of an elderly person.', 'a photo of a middle-aged person.'],
}


def site_results(backbone, head, test, child_scores):
    """Per person, as the app judges them: the average P(child) over their pictures."""
    average = np.array([np.mean(1 / (1 + np.exp(-logits(backbone.features(pictures), head)))) for _, pictures in test])
    is_child = np.array([child for child, _ in test])
    result = {'people': len(test), 'children': int(is_child.sum()),
              'auc': auc(average, is_child.astype(int)) if 0 < is_child.sum() < len(test) else None}
    for recall in RECALLS:
        threshold = recall_threshold(child_scores, recall)
        result[f'child_recall_{recall}'] = {
            'children_found': float((average[is_child] >= threshold).mean()) if is_child.any() else None,
            'adults_flagged': float((average[~is_child] >= threshold).mean()) if (~is_child).any() else None}
    return result


def zero_shot(backbone_name, image_features):
    import torch
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(BACKBONES[backbone_name]).eval()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONES[backbone_name])
    centres = {}
    for label, prompts in PROMPTS.items():
        tokens = tokenizer(prompts, padding='max_length', max_length=64, return_tensors='pt')
        with torch.no_grad():
            text = model.get_text_features(**tokens).numpy()
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        centres[label] = text.mean(0)
    images = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    return images @ centres[1] - images @ centres[0]


def face_baseline(images, labels):
    """Score the face-based age pipeline on the same person crops."""
    estimator = AgeEstimator(MODELS)
    predictions = []
    for image in images:
        height, width = image.shape[:2]
        scores = estimator.estimate(image, [(0, 0, width, height)], [0])
        predictions.append(age_group(scores[0], 0.6) if 0 in scores else 'No face')
    predictions = np.array(predictions)
    labelled = np.isin(predictions, ('Child', 'Adult', 'Older'))
    said_child = predictions == 'Child'
    return {'labelled': float(labelled.mean()),
            'no_face': float((predictions == 'No face').mean()),
            'accuracy_when_labelled': float((said_child[labelled] == labels[labelled]).mean()),
            'child_recall': float(said_child[labels == 1].mean()),
            'adult_false_alarm': float(said_child[labels == 0].mean())}


def speed(backbone, images):
    batch = images[:8]
    backbone.features(batch)
    start = time.perf_counter()
    for _ in range(5):
        backbone.features(batch)
    return (time.perf_counter() - start) / 5 * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # siglip2 also needs ~1.5 GB of download and disk, mostly for its unused text tower.
    parser.add_argument('--backbones', nargs='+', default=['dinov2'], choices=list(BACKBONES))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--skip-face', action='store_true', help='Skip the face-based baseline')
    parser.add_argument('--augment', type=int, default=3,
                        help='Webcam/CCTV-style degraded copies of each training crop (0 for none)')
    parser.add_argument('--site', help='Folder of labelled footage sessions from the footage command, added to '
                                       'training')
    parser.add_argument('--test-session', help='A --site session held back for testing only')
    parser.add_argument('--out', default=str(MODELS / 'child_head.npz'), help='Where to save the head')
    args = parser.parse_args()
    if args.test_session and not args.site:
        parser.error('--test-session needs --site')
    configure()
    device = pick_device(args.device)

    _, train_images, train_labels, folds = load('train')
    _, test_images, test_labels, _ = load('test')
    print(f'train: {train_labels.sum()} child / {(1 - train_labels).sum()} adult; '
          f'test: {test_labels.sum()} child / {(1 - test_labels).sum()} adult; device {device}')
    # Degraded copies keep their photo's fold, so cross-validation never sees a photo on both sides.
    rng = np.random.default_rng(0)
    augmented = [degrade(image, rng) for _ in range(args.augment) for image in train_images]
    train_labels = np.tile(train_labels, args.augment + 1)
    folds = np.tile(folds, args.augment + 1)
    from_site = np.zeros(len(train_labels), bool)
    site_images, site_key, site_children, site_test = [], None, 0, []
    if args.site:
        site_images, site_labels, site_folds, site_key, site_children, site_test = load_site(
            args.site, args.test_session)
        print(f'footage: {len(site_images)} training pictures ({site_children} children, '
              f'{int(site_labels.sum())} of the pictures); {len(site_test)} people held back')
        site_rng = np.random.default_rng(2)
        site_augmented = [degrade(image, site_rng) for _ in range(args.augment) for image in site_images]
        train_labels = np.concatenate([train_labels, np.tile(site_labels, args.augment + 1)])
        folds = np.concatenate([folds, np.tile(site_folds, args.augment + 1)])
        from_site = np.concatenate([from_site, np.ones(len(site_labels) * (args.augment + 1), bool)])
    test_rng = np.random.default_rng(1)
    test_sets = {f'{height}px': [shrink(image, height) for image in test_images] for height in LOW_RES}
    test_sets['webcam'] = [degrade(image, test_rng) for image in test_images]
    test_sets['upper_half'] = [image[:max(16, image.shape[0] // 2)] for image in test_images]
    report = {'train_size': len(train_images), 'augmented_copies': args.augment,
              'test_size': len(test_images), 'site_training_pictures': len(site_images),
              'site_test_session': args.test_session, 'backbones': {}}

    best = None
    for name in args.backbones:
        backbone = Backbone(name, device)
        x_train = cached_features(backbone, name, 'train', train_images)
        if augmented:
            x_train = np.concatenate([x_train, cached_features(backbone, name, f'train_aug{args.augment}', augmented)])
        if site_images:
            parts = [x_train, cached_features(backbone, name, site_key, site_images)]
            if site_augmented:
                parts.append(cached_features(backbone, name, f'{site_key}_aug{args.augment}', site_augmented))
            x_train = np.concatenate(parts)
        x_test = cached_features(backbone, name, 'test', test_images)
        l2, cv_auc, oof = cross_validate(x_train, train_labels, folds)
        # Children seen from above score differently, so with enough of them in the footage,
        # thresholds come from them alone.
        calibrate = train_labels == 1
        if site_children >= MIN_SITE_CHILDREN:
            calibrate &= from_site
        # Threshold that keeps 95% child recall on out-of-fold predictions.
        threshold = float(np.quantile(oof[calibrate], 1 - TARGET_RECALL))
        head = train_head(x_train, train_labels, l2)
        test_scores = logits(x_test, head)
        result = {'cv_auc': cv_auc, 'l2': l2, 'threshold_95_recall': threshold,
                  'thresholds_from': 'footage children' if site_children >= MIN_SITE_CHILDREN else 'all children',
                  'ms_per_8_crops': speed(backbone, test_images),
                  'test': summary(test_scores, test_labels),
                  'test_at_cv_threshold': at_threshold(test_scores, test_labels, threshold)}
        for key, images in test_sets.items():
            scores = logits(cached_features(backbone, name, f'test_{key}', images), head)
            result[f'test_{key}'] = summary(scores, test_labels)
            result[f'test_{key}_at_cv_threshold'] = at_threshold(scores, test_labels, threshold)
        if site_test:
            result['site_test'] = site_results(backbone, head, site_test, oof[calibrate])
        if name.startswith('siglip'):
            result['zero_shot_test'] = summary(zero_shot(name, x_test) * 100, test_labels)
        report['backbones'][name] = result
        print(json.dumps({name: result}, indent=1), flush=True)
        if best is None or cv_auc > best[1]:
            best = (name, cv_auc, head, threshold, np.sort(oof[calibrate]))
        del backbone

    if not args.skip_face:
        report['face_pipeline_test'] = face_baseline(test_images, test_labels)
        print(json.dumps({'face_pipeline_test': report['face_pipeline_test']}, indent=1))

    name, cv_auc, (mean, scale, weights, bias), threshold, child_scores = best
    # Out-of-fold child scores let the app pick a threshold for any --child-recall.
    out = Path(args.out)
    np.savez(out, backbone=name, mean=mean, scale=scale,
             weights=weights, bias=bias, threshold=threshold, child_scores=child_scores)
    report['chosen'] = name
    out.with_name(f'{out.stem}_report.json').write_text(json.dumps(report, indent=1))
    print(f'Saved {out} ({name}, cross-validated AUC {cv_auc:.3f})')
