# Load some data and perform zero-shot text classification using the
# Latent Embeddings approach
import os

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score

import fewshot.embeddings.word_embeddings as w2v
from fewshot.embeddings.transformer_embeddings import get_transformer_embeddings
from fewshot.metrics import simple_accuracy, simple_topk_accuracy
from fewshot.models import load_transformer_model_and_tokenizer, \
    load_word_vector_model
from fewshot.path_helper import fewshot_filename
from fewshot.predictions import compute_predictions, \
    compute_predictions_projection

from fewshot.data.loaders import load_or_cache_data
from fewshot.utils import torch_load, to_tensor, compute_projection_matrix

import pdb

DATASET_NAME = "AGNews"
DATADIR = f"data/{DATASET_NAME}"
W2VDIR = "data"
TOPK = 3

## Load data
# df of raw data (contains ground truth & labels)
# sbert embeddings for each example and each label
dataset = load_or_cache_data(DATADIR, DATASET_NAME)

# separate the example embeddings from the label embeddings
num_categories = len(dataset.categories)
sbert_emb_examples = dataset.embeddings[:-num_categories]
sbert_emb_labels = dataset.embeddings[-num_categories:]

### Compute predictions based on cosine similarity
predictions = compute_predictions(sbert_emb_examples, sbert_emb_labels, k=TOPK)

### Because our data is labeled, we can score the results!
score = simple_accuracy(dataset.labels, predictions)
score_intop3 = simple_topk_accuracy(dataset.labels, predictions)
print(f"Score: {score}")
print(f"Score considering the top {TOPK} best labels: {score_intop3}")

pdb.set_trace()
### Visualize our data and labels
# TODO: t-SNE or UMAP figure

## Let's make this model a bit better!

### Learn a projection matrix that will project SBERT embeddings into Word2Vec space
# REASONING BEHIND ALL THIS NONSENSE
# Link to the Zero-Shot HF article
# Link to my own blog post describing what's going on?
w2v_model = load_word_vector_model(small=True, cache_dir=W2VDIR)

scores = []
scores_intop3 = []

for topw in [1000, 10000, 100000]:
    w2v_embeddings_w2v_words, w2v_words = w2v.get_topk_w2v_vectors(w2v_model,
                                                                   k=topw)
    w2v_embeddings_w2v_words = to_tensor(w2v_embeddings_w2v_words)

    sbert_w2v_filename = fewshot_filename(
        W2VDIR, f"sbert_embeddings_for_top{topw}_w2v_words.pt"
    )
    if os.path.exists(sbert_w2v_filename):
        cached_data = torch_load(sbert_w2v_filename)
        sbert_embeddings_w2v_words = cached_data["embeddings"]
    else:
        sbert_embeddings_w2v_words = get_transformer_embeddings(
            w2v_words, model, tokenizer, output_filename=sbert_w2v_filename
        )

    projection_matrix = compute_projection_matrix(
        sbert_embeddings_w2v_words, w2v_embeddings_w2v_words
    )

    ### Compute new predictions utilizing the learned projection
    predictions = compute_predictions_projection(
        sbert_desc_embeddings, sbert_label_embeddings, projection_matrix, k=3
    )
    scores.append(
        f1_score(dataset.labels, [x.best for x in predictions], average="weighted"))
    scores_intop3.append(simple_topk_accuracy(df.label.tolist(), predictions))

### Visualize our modified data and label embeddings


#### ToDo
# 1. compare projections for increasing size k
# 2. compare projections learned from the most frequent words of our specific corpus
#     vs the frequency of the words inherent to the word2vec embeddings
