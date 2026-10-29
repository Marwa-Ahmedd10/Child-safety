"""The classifier's head: logistic regression on standardised image features, with its L2
penalty chosen by cross-validation grouped by photo."""
import numpy as np

from .data import FOLDS
from .metrics import auc

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)


def fit_logistic(x, y, l2):
    import torch
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    weights = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weights + bias, y)
        loss = loss + l2 * (weights ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach().numpy(), float(bias.detach())


def standardise(train):
    mean, scale = train.mean(0), train.std(0) + 1e-6
    return mean, scale


def logits(x, head):
    mean, scale, weights, bias = head
    return (x - mean) / scale @ weights + bias


def train_head(x, y, l2):
    mean, scale = standardise(x)
    weights, bias = fit_logistic((x - mean) / scale, y, l2)
    return mean, scale, weights, bias


def cross_validate(x, y, folds):
    """Pick L2 by grouped cross-validation; return it with out-of-fold logits."""
    best = None
    for l2 in L2_GRID:
        oof = np.zeros(len(y))
        for fold in range(FOLDS):
            head = train_head(x[folds != fold], y[folds != fold], l2)
            oof[folds == fold] = logits(x[folds == fold], head)
        score = auc(oof, y)
        if best is None or score > best[1]:
            best = (l2, score, oof)
    return best
