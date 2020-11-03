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
    def _get_embeddings(self):
        # Load the model and the tokenizer
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


def _load_agnews_dataset(split: str = "test"):
    """Load AG News dataset from dataset library."""
    dataset = load_dataset("ag_news", split=split)
    df = pd.DataFrame(dataset)
    # categories = dataset["test"].features["label"].names
    df["category"] = df["label"].map(
        {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    )
    return df


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
    print(filename)
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
    else:
        raise ValueError(f"Unexpected dataset name: {dataset_name}")

    dataset = Dataset(
        examples=df[text_column].tolist(),
        labels=df.label.tolist(),
        categories=_prepare_category_names(df),
    )

    pickle_save(dataset, filename)
    return dataset
