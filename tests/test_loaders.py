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

"""Provides tests for loaders.py.

The functions _load_amazon_products_dataset, _load_agnews_dataset, and
_load_reddit_dataset are mocked out and not tested."""

import pandas as pd
import unittest

import mock

from fewshot.data.loaders import load_or_cache_data
from fewshot.data.utils import Dataset
from fewshot.utils import fewshot_filename

FAKE_DIR = "FAKE_DIR"


class TestStringMethods(unittest.TestCase):
    @mock.patch("fewshot.data.loaders._load_amazon_products_dataset")
    def test_load_or_cache_amazon(
        self,
        mock_load_amazon,
    ):
        mock_load_amazon.return_value = pd.DataFrame(
            {
                "description": ["X", "Y"],  # Must be named description for Amazon
                "label": [1, 2],
                "category": ["cat1", "cat2"],
            }
        )

        expected_dataset = Dataset(
            examples=["X", "Y"], labels=[1, 2], categories=["cat1", "cat2"]
        )

        # Call load_or_cache_data.
        self.assertEqual(
            load_or_cache_data(FAKE_DIR, "amazon", with_cache=False), expected_dataset
        )

    @mock.patch("fewshot.data.loaders._load_amazon_products_dataset")
    def test_load_or_cache_amazon_with_alternative_capitalization(
        self,
        mock_load_amazon,
    ):
        mock_load_amazon.return_value = pd.DataFrame(
            {
                "description": ["X", "Y"],  # Must be named description for Amazon
                "label": [1, 2],
                "category": ["cat1", "cat2"],
            }
        )

        # Call load_or_cache_data.  Should ignore capitalization
        load_or_cache_data(FAKE_DIR, "amAzOn", with_cache=False)

    @mock.patch("fewshot.data.loaders._load_agnews_dataset")
    def test_load_or_cache_agnews(
        self,
        mock_load_agnews,
    ):
        mock_load_agnews.return_value = pd.DataFrame(
            {
                "text": ["X", "Y"],  # Must be named text for AGNews
                "label": [1, 2],
                "category": ["cat1", "cat2"],
            }
        )

        expected_dataset = Dataset(
            examples=["X", "Y"],
            labels=[1, 2],
            categories=["cat1", "cat2"],
        )

        # Call load_or_cache_data.
        self.assertEqual(
            load_or_cache_data(FAKE_DIR, "agnews", with_cache=False), expected_dataset
        )

    @mock.patch("fewshot.data.loaders._load_reddit_dataset")
    def test_load_or_cache_reddit(
        self,
        mock_load_reddit,
    ):
        mock_load_reddit.return_value = pd.DataFrame(
            {
                "summary": ["X", "Y"],  # Must be named summary for reddit
                "label": [1, 2],
                "category": ["cat1", "cat2"],
            }
        )

        expected_dataset = Dataset(
            examples=["X", "Y"],
            labels=[1, 2],
            categories=["cat1", "cat2"],
        )

        # Call load_or_cache_data.
        self.assertEqual(
            load_or_cache_data(FAKE_DIR, "reddit", with_cache=False), expected_dataset
        )

    @mock.patch("fewshot.data.loaders._load_amazon_products_dataset")
    def test_category_sorting(
        self,
        mock_load_amazon,
    ):
        mock_load_amazon.return_value = pd.DataFrame(
            {
                "description": ["A", "B", "C", "D", "E"],
                "label": [3, 1, 2, 1, 3],
                "category": ["cat3", "cat1", "cat2", "cat1", "cat3"],
            }
        )

        expected_dataset = Dataset(
            examples=["A", "B", "C", "D", "E"],
            labels=[3, 1, 2, 1, 3],
            # Must go in order of label.
            categories=["cat1", "cat2", "cat3"],
        )

        # Call load_or_cache_data.  Capitalization of "AmaZon" is ignored.
        load_or_cache_data(FAKE_DIR, "amazon", with_cache=False)
