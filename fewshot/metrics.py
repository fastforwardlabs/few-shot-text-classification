import itertools

MISSING_VALUE = "***"


def simple_accuracy(ground_truth, predictions):
    """Computes accuracy, the portion of matching values.

    Args:
        ground_truth: True labels
        predictions: Predicted labels.

    Raises:
        ValueError: If ground_truth and predictions are not the same length.
        ValueError: If ground_truth is empty

    Returns:
        The percent (portion * 100) of the labels that are correctly predicted.
    """
    return simple_topk_accuracy(ground_truth, ({x} for x in predictions))


def simple_topk_accuracy(ground_truth, predictions):
    """Computes accuracy, the portion of points for which one of the top-k
    predicted labels matches the true label.

    Args:
        ground_truth: True labels
        predictions: Top-k predicted labels.

    Raises:
        ValueError: If ground_truth and predictions are not the same length.
        ValueError: If ground_truth is empty

    Returns:
        The percent (portion * 100) of the labels that are correctly predicted.
    """
    matched, total = 0, 0
    for truth, topk in itertools.zip_longest(ground_truth, predictions,
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
