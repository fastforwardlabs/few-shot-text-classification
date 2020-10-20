from torch.nn import functional as F
import torch
from fewshot.utils import to_list


def closest_label(sentence_representation, label_representations):
    similarities = F.cosine_similarity(sentence_representation, label_representations)
    closest = similarities.argsort(descending=True)
    return similarities, closest


def compute_predictions(example_embeddings, label_embeddings, k=3):
    predictions = []
    topk = []

    if len(example_embeddings.size()) == 1:
        example_embeddings = example_embeddings.reshape((1, len(example_embeddings)))

    norm_example_embeddings = F.normalize(example_embeddings, p=2, dim=1)
    norm_label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

    for embedding in norm_example_embeddings:
        embedding = torch.reshape(embedding, (1, len(embedding)))
        scores, closest = closest_label(embedding, norm_label_embeddings)
        predictions.append(closest[0].item())
        topk.append(to_list(closest[:k]))

    return predictions, topk


def compute_predictions_projection(
    example_embeddings, label_embeddings, projection_matrix, k=3
):
    predictions = []
    topk = []

    norm_example_embeddings = F.normalize(example_embeddings, p=2, dim=1)
    norm_label_embeddings = F.normalize(label_embeddings, p=2, dim=1)

    for embedding in norm_example_embeddings:
        embedding = torch.reshape(embedding, (1, len(embedding)))
        projected_embedding = torch.matmul(embedding, projection_matrix)
        projected_label_embeddings = torch.matmul(
            norm_label_embeddings, projection_matrix
        )
        scores, closest = closest_label(projected_embedding, projected_label_embeddings)
        predictions.append(closest[0].item())
        topk.append(to_list(closest[:k]))

    return predictions, topk
