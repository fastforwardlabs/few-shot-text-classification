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

import unittest
from typing import Any

import mock
from parameterized import parameterized

from fewshot.data.loaders import load_or_cache_sbert_embeddings
from fewshot.utils import fewshot_filename


class AnyObj(object):
    """Equal to anything"""

    def __eq__(self, other: Any) -> bool:
        return True


class TestStringMethods(unittest.TestCase):
    @parameterized.expand(
        [
            ["test_amazon", "amazon", "amazon"],
            ["test_agnews", "agnews", "agnews"],
            ["test_lower_case", "aMaZoN", "amazon"],
        ]
    )
    @mock.patch("fewshot.data.loaders.get_transformer_embeddings")
    @mock.patch("fewshot.data.loaders.load_transformer_model_and_tokenizer")
    @mock.patch("fewshot.data.loaders._load_amazon_products_dataset")
    @mock.patch("fewshot.data.loaders._load_agnews_dataset")
    @mock.patch("os.path.exists")
    def test_load_or_cache_sbert_embeddings_picks_right_dataset(
        self,
        test_name,
        input_data_name,
        target_data_name,
        mock_exists,
        mock_load_agnews,
        mock_load_amazon,
        mock_model_tokenizer,
        mock_get_embeddings,
    ):
        # Test-level constants
        FAKE_DIR = "FAKE_DIR"
        AMAZON_WORDS = ["amazon", "words"]
        AGNEWS_WORDS = ["agnews", "words"]
        OUTPUT = 123  # Doesn't resemble actual output.

        # Mock values
        mock_exists.return_value = False

        mock_load_amazon.return_value = AMAZON_WORDS
        mock_load_agnews.return_value = AGNEWS_WORDS

        # Don't use these return values because we mock.
        mock_model_tokenizer.return_value = (None, None)

        mock_get_embeddings.return_value = OUTPUT

        # Call load_or_cache_sbert_embeddings
        self.assertEqual(
            load_or_cache_sbert_embeddings(FAKE_DIR, input_data_name), OUTPUT
        )

        # Expect functions are called with expected values.
        expected_filename = fewshot_filename(
            FAKE_DIR, f"{target_data_name}_embeddings.pt"
        )
        mock_exists.assert_called_once_with(expected_filename)

        if target_data_name == "amazon":
            mock_get_embeddings.assert_called_once_with(
                AMAZON_WORDS,
                AnyObj(),
                AnyObj(),
                output_filename=expected_filename,
            )
        if target_data_name == "agnews":
            mock_get_embeddings.assert_called_once_with(
                AGNEWS_WORDS,
                AnyObj(),
                AnyObj(),
                output_filename=expected_filename,
            )

    @mock.patch("os.path.exists")
    def test_load_or_cache_sbert_embeddings_picks_right_dataset(self, mock_exists):
        # Test-level constants
        FAKE_DIR = "FAKE_DIR"
        bad_name = "bad_name"

        # Mock value
        mock_exists.return_value = False

        # Call load_or_cache_sbert_embeddings
        with self.assertRaisesRegex(ValueError, f"Unexpected dataset name: {bad_name}"):
            load_or_cache_sbert_embeddings(FAKE_DIR, bad_name)

        # Expect functions are called with expected values.
        mock_exists.assert_called_once_with(
            fewshot_filename(FAKE_DIR, f"{bad_name}_embeddings.pt")
        )
