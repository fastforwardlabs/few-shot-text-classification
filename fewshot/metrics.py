import itertools
from typing import Iterator, List, Optional

import torch 

from fewshot.predictions import Prediction, compute_predictions_projection

MISSING_VALUE = "***"


def _accuracy_impl(ground_truth,
                   predictions: List[Prediction], k: Optional[int] = None):
    """Computes accuracy, the portion of points for which one of the top-k
    predicted labels matches the true label.

    Args:
        ground_truth: True labels
        predictions: List of Prediction objects.
        k: How many of the best matches to check.  If unset, use all recorded in
            closest.

    Raises:
        ValueError: If ground_truth and predictions are not the same length.
        ValueError: If ground_truth is empty

    Returns:
        The percent (portion * 100) of the labels that are correctly predicted.
    """
    matched, total = 0, 0
    for truth, pred in itertools.zip_longest(ground_truth, predictions,
                                             fillvalue=MISSING_VALUE):
        if truth == MISSING_VALUE or pred == MISSING_VALUE:
            # The shorter list has run out.
            raise ValueError(f"Accuracy length mismatch")

        total += 1
        match_set = pred.closest
        if k:
            match_set = pred.closest[:k]
        if truth in match_set:
            matched += 1

    if total == 0:
        raise ValueError("Passed lists should be non-empty")

    return matched / total * 100


def simple_accuracy(ground_truth, predictions: List[Prediction]):
    """Computes accuracy, the portion of points for which the best prediction
    matches the true label."""
    return _accuracy_impl(ground_truth, predictions, k=1)


def simple_topk_accuracy(ground_truth, predictions: List[Prediction]):
    """Computes accuracy, the portion of points for which one of the top-k
    (closest field on predictions) predicted labels matches the true label."""
    return _accuracy_impl(ground_truth, predictions)
  
  
#TODO: find a better place for this? Combine metrics.py & predictions.py?
def predict_and_score_Wmap(dataset, Wmap, Zmap=None, return_predictions=False):
  """ Compute predictions and score for a given Dataset object, Wmap, 
      and (optionally), Zmap"""
  num_categories = len(dataset.categories)
  X = dataset.embeddings[:-num_categories]
  Y = dataset.embeddings[-num_categories:]

  if Zmap is not None:
    X = torch.mm(dataset.embeddings[:-num_categories], Zmap)
    Y = torch.mm(dataset.embeddings[-num_categories:], Zmap)

  predictions = compute_predictions_projection(X, Y, Wmap)

  # compute the score for the predictions
  score = simple_accuracy(dataset.labels, predictions)
  if return_predictions:
    return score, predictions
  return score 
