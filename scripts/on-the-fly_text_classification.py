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

# Load some data and perform zero-shot text classification using the
# Latent Embeddings approach
import os
import pandas as pd

from fewshot.data.loaders import load_or_cache_data

from fewshot.embeddings.word_embeddings import (
    load_word_vector_model,
    get_topk_w2v_vectors,
)
from fewshot.embeddings.sentence_embeddings import (
    load_transformer_model_and_tokenizer,
    get_sentence_embeddings,
)

from fewshot.models.on_the_fly import OLS_with_l2_regularization

from fewshot.eval import predict_and_score, simple_topk_accuracy

# from fewshot.predictions import compute_predictions, compute_predictions_projection

from fewshot.utils import (
    torch_load,
    torch_save,
    to_tensor,
    fewshot_filename,
)


DATASET_NAME = "agnews"
DATADIR = f"data/{DATASET_NAME}"
W2VDIR = "data/w2v"
TOPK = 3

## Load data
# On first call, this function will download the agnews dataset from the
# HuggingFace Datasets repository, cache it, and then process it for use
# in this analysis
dataset = load_or_cache_data(DATADIR, DATASET_NAME)
# `dataset` is a specialized object containing the original text, the
# SentenceBERT embedding, and the label for each example in the test set.

score, predictions = predict_and_score(dataset, return_predictions=True)

score_intop3 = simple_topk_accuracy(dataset.labels, predictions)
print(f"Score: {score}")
print(f"Score considering the top {TOPK} best labels: {score_intop3}")
print()

## Let's make this model a bit better!

### Learn a mapping between SBERT sentence embeddings and Word2Vec word embeddings
# SBERT is optimized for sentences, and word2vec is optimized for words. To get
# the best of both worlds we'll learn a mapping between these two embeddings
# spaces that we can use during classification.

# To learn a mapping, we'll need to
# -- identify a large vocabulary, V, of popular words,
# -- generate w2v embeddings for each word in V,
# -- generate SBERT embeddings for each word in V,
# -- perform linear regression between the SBERT and w2v embeddings
# The result will be a matrix, Zmap, which we can use to transform SBERT embeddings
# and then perform classification with cosine similarity as before

# Load the w2v embedding model
w2v_model = load_word_vector_model(small=True, cache_dir=W2VDIR)
VOCAB_SIZE = 20000

# We found that using a vocabulary size of 20,000 words is good for most applications
vocab_w2v_embeddings, vocab = get_topk_w2v_vectors(w2v_model, k=VOCAB_SIZE)
vocab_w2v_embeddings = to_tensor(vocab_w2v_embeddings)

# Passing 20k words through SBERT can be time-consuming, even with a GPU.
# Fortunately, we've already performed this step and include precomputed embeddings.
vocab_sbert_filename = fewshot_filename(
    W2VDIR, f"sbert_embeddings_for_{VOCAB_SIZE}_words.pt"
)

if os.path.exists(vocab_sbert_filename):
    cached_data = torch_load(vocab_sbert_filename)
    vocab_sbert_embeddings = cached_data["embeddings"]
else:
    model, tokenizer = load_transformer_model_and_tokenizer()
    vocab_sbert_embeddings = get_sentence_embeddings(
        vocab, model, tokenizer, output_filename=vocab_sbert_filename
    )

# Perform ordinary least-squares linear regression to learn Zmap
Zmap = OLS_with_l2_regularization(vocab_sbert_embeddings, vocab_w2v_embeddings)

score, predictions = predict_and_score(
    dataset, linear_maps=[Zmap], return_predictions=True
)
score3 = simple_topk_accuracy(dataset.labels, predictions)
print(f"Score using projection matrix with top {VOCAB_SIZE} w2v words: {score}")
print(f"Score considering the top {TOPK} best labels: {score3}")

## Our overall classification rate improved!
# And we didn't even need training data.

# Let's save this Zmap
torch_save(Zmap, f"data/maps/Zmap_{VOCAB_SIZE}_words.pt")
