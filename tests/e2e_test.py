"""This does the same as on-the-fly_text_classification, but asserts the output.

This should test that the basic behavior of the models do not change.

This only works if the files are already loaded.
"""

import os
import unittest

from fewshot.data.loaders import load_or_cache_data

from fewshot.embeddings.word_embeddings import (
    load_word_vector_model,
    get_topk_w2v_vectors,
)

from fewshot.models.on_the_fly import OLS_with_l2_regularization

from fewshot.eval import predict_and_score, simple_topk_accuracy

from fewshot.utils import (
    torch_load,
    to_tensor,
    fewshot_filename,
)

DATASET_NAME = "agnews"
DATADIR = f"data/{DATASET_NAME}"
W2VDIR = "data/w2v"
W2V_SMALL = "GoogleNews-vectors-negative300_top500k.kv"
TOPK = 3
VOCAB_SIZE = 20000


class TestEndToEnd(unittest.TestCase):

    def _assert_files_exist(self):
        vocab_sbert_filename = fewshot_filename(
            W2VDIR, f"sbert_embeddings_for_{VOCAB_SIZE}_words.pt"
        )
        assert(os.path.exists(vocab_sbert_filename))

        dataset_filename = fewshot_filename(DATADIR, f"{DATASET_NAME}_dataset.pt")
        assert(os.path.exists(dataset_filename))

        w2v_filename = fewshot_filename(W2VDIR, W2V_SMALL)
        assert(os.path.exists(w2v_filename))

    def test_on_the_fly(self):
        # Test should only be run if the necessary files already exist.
        self._assert_files_exist()

        # Load dataset
        dataset = load_or_cache_data(DATADIR, DATASET_NAME)

        # Load w2v embeddings
        w2v_model = load_word_vector_model(small=True, cache_dir=W2VDIR)
        vocab_w2v_embeddings, vocab = get_topk_w2v_vectors(w2v_model, k=VOCAB_SIZE)
        vocab_w2v_embeddings = to_tensor(vocab_w2v_embeddings)

        # Load SBERT embeddings
        vocab_sbert_filename = fewshot_filename(
            W2VDIR, f"sbert_embeddings_for_{VOCAB_SIZE}_words.pt"
        )
        cached_data = torch_load(vocab_sbert_filename)
        vocab_sbert_embeddings = cached_data["embeddings"]

        # Calculate linear map of best fit between maps.
        Zmap = OLS_with_l2_regularization(
            vocab_sbert_embeddings, vocab_w2v_embeddings
        )

        # Predict and score
        score, predictions = predict_and_score(dataset, linear_maps=[Zmap], return_predictions=True)
        score3 = simple_topk_accuracy(dataset.labels, predictions)

        self.assertAlmostEqual(score, 65.5657894736842)
        self.assertAlmostEqual(score3, 96.01315789473685)
