import os
from typing import List

import pandas as pd
from datasets import load_dataset

from fewshot.embeddings.transformer_embeddings import get_transformer_embeddings
from fewshot.models import load_transformer_model_and_tokenizer
from fewshot.path_helper import fewshot_filename
from fewshot.utils import load_tensor

# Path in datadir folder.
AMAZON_SAMPLE_PATH = "filtered_amazon_co-ecommerce_sample.csv"


def prepare_text(df, text_column, category_column):
    text = df[text_column].tolist()
    categories = df[category_column].unique().tolist()
    return text + categories


def _load_amazon_products_dataset(datadir: str, num_categories: int = 6) -> List[str]:
    """Load Amazon products dataset from AMAZON_SAMPLE_PATH."""
    df = pd.read_csv(fewshot_filename(datadir, AMAZON_SAMPLE_PATH))
    keepers = df["category"].value_counts()[:num_categories]
    df = df[df["category"].isin(keepers.index.tolist())]
    df["category"] = pd.Categorical(df.category)
    df["label"] = df.category.cat.codes
    return prepare_text(
        df, text_column="description", category_column="category"
    )


def _load_agnews_dataset() -> List[str]:
    """Load AG News dataset from dataset library."""
    dataset = load_dataset("ag_news")
    df = pd.DataFrame(dataset["test"])
    # categories = dataset["test"].features["label"].names
    df["category"] = df["label"].map(
        {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    )
    return prepare_text(
        df, text_column="text", category_column="category"
    )


def load_or_cache_data(datadir: str, dataset_name: str):
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
    dataset_name = dataset_name.lower()
    filename = fewshot_filename(datadir, f"{dataset_name}_embeddings.pt")
    if os.path.exists(filename):
        cached_data = load_tensor(filename)
        return cached_data["embeddings"]

    # Load appropriate data
    if dataset_name == "amazon":
        data = _load_amazon_products_dataset(datadir)
    elif dataset_name == "agnews":
        data = _load_agnews_dataset()
    else:
        raise ValueError(f"Unexpected dataset name: {dataset_name}")

    # Load the model and the tokenizer
    model, tokenizer = load_transformer_model_and_tokenizer()

    # Get embeddings and return.  This has the side-effect of caching.
    return get_transformer_embeddings(
        data, model, tokenizer, output_filename=filename
    )
