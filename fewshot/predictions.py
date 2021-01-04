import math
from typing import Any, Callable, List, Optional

import attr
import pandas as pd
import torch
from torch.nn import functional as F

from fewshot.utils import to_list

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
        return (self.closest==other.closest and self.best==other.best)

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def to_df(self):
        return pd.DataFrame(
            data={"closest": self.closest, "scores": self.scores}
        )


def closest_label(sentence_representation, label_representations):
    similarities = F.cosine_similarity(
        sentence_representation, label_representations
    )
    closest = similarities.argsort(descending=True)
    return similarities, closest


def compute_predictions(
        example_embeddings,
        label_embeddings,
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
        example_embeddings = example_embeddings.reshape(
            (1, len(example_embeddings))
        )

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


def compute_predictions_projection(
        example_embeddings, label_embeddings, projection_matrix, k: int = 3
) -> List[Prediction]:
    """Make predictions for each of the example embeddings.

    The function compares the embedding of the example to each of the
    label_embeddings.  The one that it is closest to is the predicted label.

    Args:
        example_embeddings: The embeddings of the data that we want to make
            predictions for.
        label_embeddings: The embeddings of the category labels.
        projection_matrix: A matrix used to project the embeddings of both the
            examples and the labels.
        k: The closest field of the returned Prediction class will contain the k
            closest (best) predictions.

    Returns:
        A list of Prediction objects, one for each passed example.
    """
    projection = lambda x: torch.matmul(x, projection_matrix)
    return compute_predictions(
        example_embeddings, label_embeddings, k, projection
    )


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



""" WIP for turning a list of Prediction objects to df
dfs = [p.to_df() for p in predictions]
dfs2 = []
for i, df in enumerate(dfs):
    df['example'] = i
    dfs2.append(df)
bigdf = pd.concat(dfs2)
"""
