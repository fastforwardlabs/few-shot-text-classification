# Load some data and perform zero-shot text classification using the
# Latent Embeddings approach
import os

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score

from fewshot.embeddings.word_embeddings import (
    load_word_vector_model,
    get_topk_w2v_vectors,
)
from fewshot.embeddings.transformer_embeddings import (
    load_transformer_model_and_tokenizer,
    get_transformer_embeddings,
)

from fewshot.models.on-the-fly import fit_Zmap_matrix

from fewshot.metrics import simple_accuracy, simple_topk_accuracy

from fewshot.predictions import compute_predictions, compute_predictions_projection

from fewshot.data.loaders import load_or_cache_data

from fewshot.utils import (
    torch_load,
    to_tensor,
    fewshot_filename,
)

import pdb

DATASET_NAME = "amazon"
DATADIR = f"data/{DATASET_NAME}"
W2VDIR = "data/w2v"
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
print()

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
    w2v_embeddings_w2v_words, w2v_words = get_topk_w2v_vectors(w2v_model, k=topw)
    w2v_embeddings_w2v_words = to_tensor(w2v_embeddings_w2v_words)

    sbert_w2v_filename = fewshot_filename(
        W2VDIR, f"sbert_embeddings_for_top{topw}_w2v_words.pt"
    )
    if os.path.exists(sbert_w2v_filename):
        cached_data = torch_load(sbert_w2v_filename)
        sbert_embeddings_w2v_words = cached_data["embeddings"]
    else:
        model, tokenizer = load_transformer_model_and_tokenizer()
        sbert_embeddings_w2v_words = get_transformer_embeddings(
            w2v_words, model, tokenizer, output_filename=sbert_w2v_filename
        )

    projection_matrix = fit_Zmap_matrix(
        sbert_embeddings_w2v_words, w2v_embeddings_w2v_words
    )

    ### Compute new predictions utilizing the learned projection
    predictions = compute_predictions_projection(
        sbert_emb_examples, sbert_emb_labels, projection_matrix, k=3
    )
    score = simple_accuracy(dataset.labels, predictions)
    score3 = simple_topk_accuracy(dataset.labels, predictions)
    print(f"Score using projection matrix with top {topw} w2v words: {score}")
    # print(f"Score considering the top {TOPK} best labels: {score3}")

### Visualize our modified data and label embeddings


#### ToDo
# 1. compare projections for increasing size k
# 2. compare projections learned from the most frequent words of our specific corpus
#     vs the frequency of the words inherent to the word2vec embeddings
