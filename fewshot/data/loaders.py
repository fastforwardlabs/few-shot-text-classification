import os
import attr
from typing import List

import pandas as pd
from datasets import load_dataset

from fewshot.embeddings.transformer_embeddings import (
    load_transformer_model_and_tokenizer,
    get_transformer_embeddings,
)
from fewshot.utils import pickle_load, pickle_save, fewshot_filename

# Path in datadir folder.
AMAZON_SAMPLE_PATH = "filtered_amazon_co-ecommerce_sample.csv"
REDDIT_SAMPLE_PATH = "reddit_subset_test.csv"


@attr.s
class Dataset(object):
    # These are the text (news articles, product descriptions, etc.)
    examples: List[str] = attr.ib()
    # Labels associated with each example
    # TODO: at some point this has to change because in a real application labels may
    # not exist or there might be fewer labels than examples (need to keep track)
    labels: List[int] = attr.ib()
    # Categories that correspond to the number of unique Labels
    categories: List[str] = attr.ib()
    # embeddings for each example and each category
    embeddings = attr.ib()

    @embeddings.default
    def _get_embeddings(self, model_name_or_path=None):
        # Load the model and the tokenizer
        # TODO: need to be able to pass a specific model rather than using default
        model, tokenizer = load_transformer_model_and_tokenizer()
        return get_transformer_embeddings(
            self.examples + self.categories, model, tokenizer
        )


def _prepare_text(df, text_column):
    text = df[text_column].tolist()
    categories = df["category"].unique().tolist()
    return text + categories


def _prepare_category_names(df):
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


def _load_amazon_products_dataset(datadir: str, num_categories: int = 6):
    """Load Amazon products dataset from AMAZON_SAMPLE_PATH."""
    df = pd.read_csv(fewshot_filename(datadir, AMAZON_SAMPLE_PATH))
    keepers = df["category"].value_counts()[:num_categories]
    df = df[df["category"].isin(keepers.index.tolist())]
    df["category"] = pd.Categorical(df.category)
    df["label"] = df.category.cat.codes
    return df


def _load_reddit_dataset(datadir: str, categories="curated"):
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


def _load_agnews_dataset(split: str = "test"):
    """Load AG News dataset from dataset library."""
    dataset = load_dataset("ag_news", split=split)
    df = pd.DataFrame(dataset)
    # categories = dataset["test"].features["label"].names
    df["category"] = df["label"].map(
        {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    )
    return df


def _create_dataset_from_df(df, text_column: str):
    dataset = Dataset(
        examples=df[text_column].tolist(),
        labels=df.label.tolist(),
        categories=_prepare_category_names(df),
    )


def load_or_cache_data(datadir: str, dataset_name: str) -> Dataset:
    """Loads sbert embeddings.

    First checks for a cached computation, otherwise builds the embedding with a
    call to get_transformer_embeddings using the specified dataset and standard
    model and tokenizer.

    Args:
        datadir: Where to save/load cached files.
        dataset_name: "amazon" for Amazon products dataset or "agnews".

    Raises:
        ValueError: If an unexpected dataset_name is passed.

    Returns:
        The embeddings.
    """
    # Check for cached data.
    print("Checking for cached data...")
    dataset_name = dataset_name.lower()
    filename = fewshot_filename(datadir, f"{dataset_name}_dataset.pt")
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
        raise ValueError(f"Unexpected dataset name: {dataset_name}")

    _create_dataset_from_df(df, text_column)

    pickle_save(dataset, filename)
    return dataset


def expand_labels(dataset):
    """ 
    When performing supervised learning (e.g. few-shot), we will need a label embedding for 
    each example in the dataset. Most datasets only have a handful of labels (4-10).
    Passing these repeatedly through SBERT for each example is slow, repetitive and
    unnecessarily expensive. 

    Instead we'll restructure the dataset attributes. Originally instantiated, each label 
    has already been passed through SBERT and is stored in dataset.embeddings 
    as the last N items in the list. These are used to build out a full label embedding tensor.
    Additionally, dataset.embeddings is repurposed to contain ONLY example embeddings 
    rather than example AND label embeddings
    """

    num_labels = len(dataset.categories)
    label_embeddings = to_list(dataset.embeddings[-num_labels:])

    dataset.label_embeddings = to_tensor([label_embeddings[label] for label in dataset.labels])
    dataset.embeddings = dataset.embeddings[:-num_labels]
    return dataset
