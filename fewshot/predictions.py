from torch.nn import functional as F
import torch


def closest_label(sentence_representation, label_representations):
    similarities = F.cosine_similarity(sentence_representation, label_representations)
    closest = similarities.argsort(descending=True)
    return similarities, closest

def compute_predictions(example_embeddings, label_embeddings, k=3):
    predictions = []
    topk = []

    for embedding in example_embeddings:
        embedding = torch.reshape(embedding, (1, len(embedding)))
        scores, closest = closest_label(embedding, label_embeddings)
        predictions.append(closest[0].item())
        topk.append(to_list(closest[:k]))

    return predictions, topk

def compute_predictions_projection(example_embeddings, label_embeddings, projection_matrix, k=3):
    predictions = []
    topk = []

    for embedding in example_embeddings:
        embedding = torch.reshape(embedding, (1, len(embedding)))
        projected_embedding = torch.matmul(embedding, projection_matrix)

        projected_label_embeddings = torch.matmul(label_embeddings, projection_matrix)

        scores, closest = closest_label(projected_embedding, projected_label_embeddings)

        predictions.append(closest[0].item())
        topk.append(to_list(closest[:k]))

    return predictions, topk