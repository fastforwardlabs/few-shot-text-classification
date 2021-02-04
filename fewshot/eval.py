# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

""" Contains fuctions to compute predictions with or without projections,
    and compute the accuracy of those predictions given ground truth labels.

    These capabilities are contained in two wrapper functions: 
        predict_and_score_Zmap
        predict_and_score_Wmap
"""

import itertools
import math
import pandas as pd
from typing import Any, Callable, List, Optional, Tuple

import attr
import torch
from torch.nn import functional as F

from fewshot.utils import to_list
from fewshot.data.utils import Dataset, Label

MISSING_VALUE = "***"
PredictionClass = Any


@attr.s(eq=False)
class Prediction(object):
    # The top-k best predictions for the i-th point.
    closest: List[PredictionClass] = attr.ib()
    # The corresponding scores for each prediction from closest list.
    scores: List[float] = attr.ib()
    # The best prediction for the i-th point.
    best: PredictionClass = attr.ib()

    def __eq__(self, other: Any) -> bool:
        """Scores need only be approx eq."""
        if not isinstance(other, Prediction):
            return False
        # Check closeness of scores
        if len(self.scores) != len(other.scores):
            return False
        for s1, s2 in zip(self.scores, other.scores):
            if not math.isclose(s1, s2, abs_tol=1e-5):
                return False
        return self.closest == other.closest and self.best == other.best

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def to_df(self):
        return pd.DataFrame(data={"closest": self.closest, "scores": self.scores})


def closest_label(sentence_representation, label_representations):
    """Returns the closest label and score for the passed sentence."""
    similarities = F.cosine_similarity(sentence_representation, label_representations)
    closest = similarities.argsort(descending=True)
    return similarities, closest


def _compute_linear_transformations(
    dataset: Dataset, linear_maps: Optional[List[torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute embeddings for data and labels from dataset.

    We use the fact that the number of dataset categories determines how the
    embeddings split into data embeddings (X) and label embeddings (Y).  We then
    transform these by applying all the linear maps provided in order, by
    multiplying on the right.

    Args:
        dataset:  A Dataset which we use for the categories and embeddings.
        linear_maps:  An optional list of linear transformations that we
            multiply on the right.

    Returns:
        Embeddings for the data and the labels.
    """
    if linear_maps is None:
        linear_maps = []

    # `dataset.embeddings` contains the SBERT embeddings for every example as well
    #  for the label names.
    # We separate the example embeddings from the label embeddings for clarity
    num_categories = len(dataset.categories)
    X = dataset.embeddings[:-num_categories]
    Y = dataset.embeddings[-num_categories:]

    for linear_map in linear_maps:
        X = torch.mm(X, linear_map)
        Y = torch.mm(Y, linear_map)

    return X, Y


def compute_predictions(
    example_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
    k: int = 3,
    transformation: Optional[Callable] = None,
) -> List[Prediction]:
    """Make predictions for each of the example embeddings.

    The function compares the embedding of the example to each of the
    label_embeddings.  The one that it is closest to is the predicted label.

    Args:
        example_embeddings: The embeddings of the data that we want to make
            predictions for.
        label_embeddings: The embeddings of the category labels.
        k: The closest field of the returned Prediction class will contain the k
            closest (best) predictions.
        transformation: If set, this function will get applied to both examples
            and labels before comparing.

    Returns:
        A list of Prediction objects, one for each passed example.
    """
    if transformation is None:
        # Pass-through in this case.
        transformation = lambda x: x

    if len(example_embeddings.size()) == 1:
        example_embeddings = example_embeddings.reshape((1, len(example_embeddings)))

    norm_example_embeddings = F.normalize(example_embeddings, p=2, dim=1)
    norm_label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
    transformed_label_embeddings = transformation(norm_label_embeddings)

    predictions = list()
    for i, embedding in enumerate(norm_example_embeddings):
        embedding = embedding.reshape((1, len(embedding)))
        transformed_embedding = transformation(embedding)
        scores, closest = closest_label(
            transformed_embedding, transformed_label_embeddings
        )
        predictions.append(
            Prediction(
                scores=sorted(to_list(scores), reverse=True)[:k],
                closest=to_list(closest[:k]),
                best=closest[0].item(),
            )
        )

    return predictions


def _accuracy_impl(
    ground_truth: List[Label], predictions: List[Prediction], k: Optional[int] = None
) -> float:
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
    for truth, pred in itertools.zip_longest(
        ground_truth, predictions, fillvalue=MISSING_VALUE
    ):
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


def simple_accuracy(ground_truth: List[Label], predictions: List[Prediction]) -> float:
    """Computes accuracy, the portion of points for which the best prediction
    matches the true label."""
    return _accuracy_impl(ground_truth, predictions, k=1)


def simple_topk_accuracy(
    ground_truth: List[Label], predictions: List[Prediction]
) -> float:
    """Computes accuracy, the portion of points for which one of the top-k
    (closest field on predictions) predicted labels matches the true label."""
    return _accuracy_impl(ground_truth, predictions)


def predict_and_score(
    dataset: Dataset,
    linear_maps: List[torch.Tensor] = None,
    return_predictions: bool = False,
):
    """Compute predictions and score for a given Dataset

    The predictions are made with the compute_predictions function, which looks
    for the closest label to each data point in the embedding space.

    Args:
        dataset:  The data that we want to make predictions for.
        linear_maps:  An optional list of linear maps to apply before predicting
        return_predictions:  If set, return the list of predictions.

    Returns:
        The (simple accuracy) score and the predictions made if
            return_predictions is set.
    """

    example_features, label_features = _compute_linear_transformations(
        dataset, linear_maps
    )

    predictions = compute_predictions(example_features, label_features)

    # compute the score for the predictions
    score = simple_accuracy(dataset.labels, predictions)
    if return_predictions:
        return score, predictions
    return score
