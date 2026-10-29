"""How well P(child) scores separate children from adults."""
import numpy as np

RECALLS = (0.95, 0.90, 0.85, 0.80)  # --child-recall settings to report


def auc(scores, labels):
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores)] = np.arange(1, len(scores) + 1)
    positives, negatives = labels.sum(), (1 - labels).sum()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def recall_at_fpr(scores, labels, fpr):
    threshold = np.quantile(scores[labels == 0], 1 - fpr)
    return float((scores[labels == 1] > threshold).mean())


def fpr_at_recall(scores, labels, recall):
    threshold = np.quantile(scores[labels == 1], 1 - recall)
    return float((scores[labels == 0] >= threshold).mean())


def at_threshold(scores, labels, threshold):
    return {'child_recall': float((scores[labels == 1] >= threshold).mean()),
            'adult_false_alarm': float((scores[labels == 0] >= threshold).mean())}


def summary(scores, labels):
    return {'auc': auc(scores, labels),
            'accuracy': float(((scores >= 0) == labels).mean()),
            'recall_at_5pct_false_alarm': recall_at_fpr(scores, labels, 0.05),
            'recall_at_10pct_false_alarm': recall_at_fpr(scores, labels, 0.10),
            'false_alarm_at_90pct_recall': fpr_at_recall(scores, labels, 0.90),
            'false_alarm_at_95pct_recall': fpr_at_recall(scores, labels, 0.95)}
