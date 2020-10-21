import itertools
from typing import Iterator, List

from fewshot.predictions import PredictionClass, Predictions

MISSING_VALUE = "***"


def _accuracy_impl(ground_truth,
                   topk_predictions: Iterator[List[PredictionClass]]):
    """Computes accuracy, the portion of points for which one of the top-k
    predicted labels matches the true label.

    Args:
        ground_truth: True labels
        topk_predictions: Top-k predicted labels.  May be given as a list or a
            generator.

    Raises:
        ValueError: If ground_truth and predictions are not the same length.
        ValueError: If ground_truth is empty

    Returns:
        The percent (portion * 100) of the labels that are correctly predicted.
    """
    matched, total = 0, 0
    for truth, topk in itertools.zip_longest(ground_truth, topk_predictions,
                                             fillvalue=MISSING_VALUE):
        if truth == MISSING_VALUE or topk == MISSING_VALUE:
            # The shorter list has run out.
            raise ValueError(f"Accuracy length mismatch")

        total += 1
        if truth in topk:
            matched += 1

    if total == 0:
        raise ValueError("Passed lists should be non-empty")

    return matched / total * 100


def simple_accuracy(ground_truth, predictions: Predictions):
    """Computes accuracy, the portion of points for which the best prediction
    matches the true label."""
    return _accuracy_impl(ground_truth, ([x] for x in predictions.best))


def simple_topk_accuracy(ground_truth, predictions: Predictions):
    """Computes accuracy, the portion of points for which one of the top-k
    (closest field on predictions) predicted labels matches the true label."""
    return _accuracy_impl(ground_truth, predictions.closest)
