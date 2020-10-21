import attr
from typing import Any, List

from torch.nn import functional as F
import torch
from fewshot.utils import to_list


PredictionClass = Any

@attr.s
class Predictions(object):
    # The top-k best predictions for the i-th point.
    closest: List[List[PredictionClass]] = attr.ib()
    # The corresponding scores for each prediction from closest list.
    scores: List[List[float]] = attr.ib()
    # The best prediction for the i-th point.
    best: List[PredictionClass] = attr.ib()


def closest_label(sentence_representation, label_representations):
    similarities = F.cosine_similarity(sentence_representation, label_representations)
    closest = similarities.argsort(descending=True)
    return similarities, closest


def compute_predictions(example_embeddings, label_embeddings, k=3) -> Predictions:
    if len(example_embeddings.size()) == 1:
        example_embeddings = example_embeddings.reshape((1, len(example_embeddings)))

    norm_example_embeddings = F.normalize(example_embeddings, p=2, dim=1)
    norm_label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

    predictions = Predictions(closest=list(), scores=list(), best=list())
    for i, embedding in enumerate(norm_example_embeddings):
        embedding = embedding.reshape((1, len(embedding)))
        scores, closest = closest_label(embedding, norm_label_embeddings)
        predictions.scores.append(to_list(scores[:k]))
        predictions.closest.append(to_list(closest[:k]))
        predictions.best.append(closest[0].item())

    return predictions


def compute_predictions_projection(
    example_embeddings, label_embeddings, projection_matrix, k=3
) -> Predictions:
    if len(example_embeddings.size()) == 1:
        example_embeddings = example_embeddings.reshape((1, len(example_embeddings)))

    norm_example_embeddings = F.normalize(example_embeddings, p=2, dim=1)
    norm_label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

    predictions = Predictions(closest=list(), scores=list(), best=list())
    for embedding in norm_example_embeddings:
        embedding = torch.reshape(embedding, (1, len(embedding)))
        projected_embedding = torch.matmul(embedding, projection_matrix)
        projected_label_embeddings = torch.matmul(
            norm_label_embeddings, projection_matrix
        )
        scores, closest = closest_label(projected_embedding, projected_label_embeddings)
        predictions.scores.append(to_list(scores[:k]))
        predictions.closest.append(to_list(closest[:k]))
        predictions.best.append(closest[0].item())

    return predictions
