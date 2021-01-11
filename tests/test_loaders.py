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
    @mock.patch("fewshot.data.loaders.pickle_save")
    @mock.patch("os.path.exists")
    def test_load_or_cache_amazon(
        self,
        mock_exists,
        mock_pickle_save,
        mock_load_amazon,
    ):
        # Say that the cache file doesn't exist, so that it loads anew.
        mock_exists.return_value = False

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
        self.assertEqual(load_or_cache_data(FAKE_DIR, "amazon"), expected_dataset)

        # Expect functions are called with expected values.
        expected_filename = fewshot_filename(FAKE_DIR, f"amazon_dataset.pt")
        mock_exists.assert_called_once_with(expected_filename)

    @mock.patch("fewshot.data.loaders._load_amazon_products_dataset")
    @mock.patch("fewshot.data.loaders.pickle_save")
    @mock.patch("os.path.exists")
    def test_load_or_cache_amazon_with_alternative_capitalization(
        self,
        mock_exists,
        mock_pickle_save,
        mock_load_amazon,
    ):
        # Say that the cache file doesn't exist, so that it loads anew.
        mock_exists.return_value = False

        mock_load_amazon.return_value = pd.DataFrame(
            {
                "description": ["X", "Y"],  # Must be named description for Amazon
                "label": [1, 2],
                "category": ["cat1", "cat2"],
            }
        )

        # Call load_or_cache_data.  Should ignore capitalization
        load_or_cache_data(FAKE_DIR, "amAzOn")

        # Expect functions are called with expected values.
        expected_filename = fewshot_filename(FAKE_DIR, f"amazon_dataset.pt")
        mock_exists.assert_called_once_with(expected_filename)

    @mock.patch("fewshot.data.loaders._load_agnews_dataset")
    @mock.patch("fewshot.data.loaders.pickle_save")
    @mock.patch("os.path.exists")
    def test_load_or_cache_agnews(
        self,
        mock_exists,
        mock_pickle_save,
        mock_load_agnews,
    ):
        # Say that the cache file doesn't exist, so that it loads anew.
        mock_exists.return_value = False

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
        self.assertEqual(load_or_cache_data(FAKE_DIR, "agnews"), expected_dataset)

        # Expect functions are called with expected values.
        expected_filename = fewshot_filename(FAKE_DIR, f"agnews_dataset.pt")
        mock_exists.assert_called_once_with(expected_filename)

    @mock.patch("fewshot.data.loaders._load_reddit_dataset")
    @mock.patch("fewshot.data.loaders.pickle_save")
    @mock.patch("os.path.exists")
    def test_load_or_cache_reddit(
        self,
        mock_exists,
        mock_pickle_save,
        mock_load_reddit,
    ):
        # Say that the cache file doesn't exist, so that it loads anew.
        mock_exists.return_value = False

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
        self.assertEqual(load_or_cache_data(FAKE_DIR, "reddit"), expected_dataset)

        # Expect functions are called with expected values.
        expected_filename = fewshot_filename(FAKE_DIR, f"reddit_dataset.pt")
        mock_exists.assert_called_once_with(expected_filename)

    @mock.patch("fewshot.data.loaders._load_amazon_products_dataset")
    @mock.patch("fewshot.data.loaders.pickle_save")
    @mock.patch("os.path.exists")
    def test_category_sorting(
        self,
        mock_exists,
        mock_pickle_save,
        mock_load_amazon,
    ):
        # Say that the cache file doesn't exist, so that it loads anew.
        mock_exists.return_value = False

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
        load_or_cache_data(FAKE_DIR, "amazon")

        # Expect functions are called with expected values.
        expected_filename = fewshot_filename(FAKE_DIR, f"amazon_dataset.pt")
        mock_exists.assert_called_once_with(expected_filename)

    @mock.patch("fewshot.data.loaders._load_amazon_products_dataset")
    @mock.patch("fewshot.data.loaders.pickle_save")
    @mock.patch("os.path.exists")
    def test_pickle_save(
        self,
        mock_exists,
        mock_pickle_save,
        mock_load_amazon,
    ):
        # Say that the cache file doesn't exist, so that it loads anew.
        mock_exists.return_value = False

        mock_load_amazon.return_value = pd.DataFrame(
            {
                "description": ["X", "Y"],  # Must be named description for Amazon
                "label": [1, 2],
                "category": ["cat1", "cat2"],
            }
        )

        expected_dataset = Dataset(
            examples=["X", "Y"],
            labels=[1, 2],
            categories=["cat1", "cat2"],
        )

        # Call load_or_cache_data.  Capitalization of "AmaZon" is ignored.
        load_or_cache_data(FAKE_DIR, "amazon")

        # Expect pickle_save
        mock_pickle_save.assert_called_once_with(
            expected_dataset, fewshot_filename(FAKE_DIR, "amazon_dataset.pt")
        )

    @mock.patch("os.path.exists")
    def test_load_or_cache_correctly_fails(self, mock_exists):
        # Test-level constants
        bad_name = "bad_name"

        # Mock value
        mock_exists.return_value = False

        # Call load_or_cache_data
        with self.assertRaisesRegex(ValueError, f"Unexpected dataset name: {bad_name}"):
            load_or_cache_data(FAKE_DIR, bad_name)

        # Expect functions are called with expected values.
        mock_exists.assert_called_once_with(
            fewshot_filename(FAKE_DIR, f"{bad_name}_dataset.pt")
        )
