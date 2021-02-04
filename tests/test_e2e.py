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
        assert os.path.exists(vocab_sbert_filename)

        dataset_filename = fewshot_filename(DATADIR, f"{DATASET_NAME}_dataset.pt")
        assert os.path.exists(dataset_filename)

        w2v_filename = fewshot_filename(W2VDIR, W2V_SMALL)
        assert os.path.exists(w2v_filename)

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
        Zmap = OLS_with_l2_regularization(vocab_sbert_embeddings, vocab_w2v_embeddings)

        # Predict and score
        score, predictions = predict_and_score(
            dataset, linear_maps=[Zmap], return_predictions=True
        )
        score3 = simple_topk_accuracy(dataset.labels, predictions)

        self.assertAlmostEqual(score, 65.5657894736842)
        self.assertAlmostEqual(score3, 96.01315789473685)
