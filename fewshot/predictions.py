from typing import Any, List

import attr
import pandas as pd
import torch
from torch.nn import functional as F

from fewshot.utils import to_list

PredictionClass = Any


@attr.s
class Prediction(object):
    # The top-k best predictions for the i-th point.
    closest: List[PredictionClass] = attr.ib()
    # The corresponding scores for each prediction from closest list.
    scores: List[float] = attr.ib()
    # The best prediction for the i-th point.
    best: PredictionClass = attr.ib()

    def to_df(self):
        return pd.DataFrame(data={"closest":self.closest,
                                  "scores":self.scores})


def closest_label(sentence_representation, label_representations):
    similarities = F.cosine_similarity(sentence_representation,
                                       label_representations)
    closest = similarities.argsort(descending=True)
    return similarities, closest


def compute_predictions(example_embeddings, label_embeddings, k=3) -> List[
    Prediction]:

    if len(example_embeddings.size()) == 1:
        example_embeddings = example_embeddings.reshape(
            (1, len(example_embeddings)))

    norm_example_embeddings = F.normalize(example_embeddings, p=2, dim=1)
    norm_label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

    predictions = list()
    for i, embedding in enumerate(norm_example_embeddings):
        embedding = embedding.reshape((1, len(embedding)))
        scores, closest = closest_label(embedding, norm_label_embeddings)
        predictions.append(
            Prediction(scores=sorted(to_list(scores), reverse=True)[:k], 
                       closest=to_list(closest[:k]), best=closest[0].item()))

    return predictions


def compute_predictions_projection(
        example_embeddings, label_embeddings, projection_matrix, k=3
) -> List[Prediction]:

    if len(example_embeddings.size()) == 1:
        example_embeddings = example_embeddings.reshape(
            (1, len(example_embeddings)))

    norm_example_embeddings = F.normalize(example_embeddings, p=2, dim=1)
    norm_label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

    predictions = list()
    for embedding in norm_example_embeddings:
        embedding = torch.reshape(embedding, (1, len(embedding)))
        projected_embedding = torch.matmul(embedding, projection_matrix)
        projected_label_embeddings = torch.matmul(
            norm_label_embeddings, projection_matrix
        )
        scores, closest = closest_label(projected_embedding,
                                        projected_label_embeddings)
        predictions.append(
            Prediction(scores=sorted(to_list(scores), reverse=True)[:k], 
                       closest=to_list(closest[:k]), best=closest[0].item()))

    return predictions


""" WIP for turning a list of Prediction objects to df
dfs = [p.to_df() for p in predictions]
dfs2 = []
for i, df in enumerate(dfs):
    df['example'] = i
    dfs2.append(df)
bigdf = pd.concat(dfs2)
"""