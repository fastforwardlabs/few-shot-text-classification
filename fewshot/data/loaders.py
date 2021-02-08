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

import os
import pandas as pd
from typing import List

from datasets import load_dataset as load_HF_dataset

from fewshot.data.utils import Dataset
from fewshot.utils import pickle_load, pickle_save, fewshot_filename

# Path in datadir folder.
AMAZON_SAMPLE_PATH = "filtered_amazon_co-ecommerce_sample.csv"
REDDIT_SAMPLE_PATH = "reddit_subset_test1300.csv"


def _prepare_text(df: pd.DataFrame, text_column: str) -> List[str]:
    """Returns a list of string labels from a dataframe."""
    text = df[text_column].tolist()
    categories = df["category"].unique().tolist()
    return text + categories


def _prepare_category_names(df: pd.DataFrame):
    """
    Category names must be in the order implied by their integer label counterpart
    e.g.  If we have integer Labels and category names mapped as follows:
    0 --> "World"
    1 --> "Sports"
    2 --> "Business"
    3 --> "Sci/Tech"

    Then we must return the category names in order like
    > categories = ["World", "Sports", "Business", "Sci/Tech"]

    They can NOT be alphabetical, which is what you'll get if you simply use
    > categories = df.category.unique()
    """
    mapping = set(zip(df.label, df.category))
    return [c for l, c in sorted(mapping)]


def _load_amazon_products_dataset(
    datadir: str, num_categories: int = 6
) -> pd.DataFrame:
    """Load Amazon products dataset from AMAZON_SAMPLE_PATH."""
    df = pd.read_csv(fewshot_filename(datadir, AMAZON_SAMPLE_PATH))
    keepers = df["category"].value_counts()[:num_categories]
    df = df[df["category"].isin(keepers.index.tolist())]
    df["category"] = pd.Categorical(df.category)
    df["label"] = df.category.cat.codes
    return df


def _load_reddit_dataset(datadir: str, categories: str = "curated") -> pd.DataFrame:
    # TODO: the dataset included with the repo no longer allows a choice between 
    #       "curated" or "top10" -- curated subreddits only; should update this here
    #       and remove that functionality 
    """
    Load a curated and smaller version of the Reddit dataset from dataset library.

    There are two dataset options to choose from:
        1. (default) "curated" categories returns reddit examples from popular subreddits
            that have more meaningful subreddit names
        2. "top10" categories returns reddit examples from the most popular
            subreddits regardless of how meaningful the subreddit name is
        3. Anything else will return all the possible categories (16 in total)
    """
    df = pd.read_csv(fewshot_filename(datadir, REDDIT_SAMPLE_PATH))
    curated_subreddits = [
        "relationships",
        "trees",
        "gaming",
        "funny",
        "politics",
        "sex",
        "Fitness",
        "worldnews",
        "personalfinance",
        "technology",
    ]
    top10_subreddits = df["category"].value_counts()[:10]

    if categories == "curated":
        df = df[df["subreddit"].isin(curated_subreddits)]
    elif categories == "top10":
        df = df[df["subreddit"].isin(top10_subreddits.index.tolist())]
    df["category"] = pd.Categorical(df.category)
    df["label"] = df.category.cat.codes
    return df


def _load_agnews_dataset(split: str = "test") -> pd.DataFrame:
    """Load AG News dataset from dataset library."""
    dataset = load_HF_dataset("ag_news", split=split)
    df = pd.DataFrame(dataset)
    df["category"] = df["label"].map(
        {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    )
    return df


def _create_dataset_from_df(
    df: pd.DataFrame, text_column: str, filename: str = None
) -> Dataset:
    dataset = Dataset(
        examples=df[text_column].tolist(),
        labels=df.label.tolist(),
        categories=_prepare_category_names(df),
    )

    if filename is not None:
        dataset.calc_sbert_embeddings()
        pickle_save(dataset, filename)
    return dataset


def load_or_cache_data(
    datadir: str, dataset_name: str, with_cache: bool = True
) -> Dataset:
    """Loads sbert embeddings.

    First checks for a cached computation, otherwise builds the embedding with a
    call to get_sentence_embeddings using the specified dataset and standard
    model and tokenizer.

    Args:
        datadir: Where to save/load cached files.
        dataset_name: "amazon", "agnews", or "reddit".
        with_cache: If set, use cache files.  Settable for testing.

    Raises:
        ValueError: If an unexpected dataset_name is passed.

    Returns:
        The embeddings.
    """
    # Check for cached data.
    print("Checking for cached data...")
    dataset_name = dataset_name.lower()
    filename = None
    if with_cache:
        filename = fewshot_filename(datadir, f"{dataset_name}_dataset.pkl")
        if os.path.exists(filename):
            return pickle_load(filename)

    print(f"{dataset_name} dataset not found. Computing...")
    # Load appropriate data
    if dataset_name == "amazon":
        df = _load_amazon_products_dataset(datadir)
        text_column, category_column = "description", "category"
    elif dataset_name == "agnews":
        df = _load_agnews_dataset()
        text_column, category_column = "text", "category"
    elif dataset_name == "reddit":
        df = _load_reddit_dataset(datadir)
        text_column, category_column = "summary", "category"
    else:
        raise ValueError(
            f"Unexpected dataset name: {dataset_name}.\n \
                          Please choose from: agnews, amazon, or reddit"
        )

    dataset = _create_dataset_from_df(df, text_column, filename=filename)
    return dataset
