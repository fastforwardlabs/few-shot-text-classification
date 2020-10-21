import pandas as pd
from datasets import load_dataset 

from fewshot.path_helper import fewshot_filename


def load_amazon_products_dataset(datadir, num_categories=6):
    df = pd.read_csv(
        fewshot_filename(datadir, "filtered_amazon_co-ecommerce_sample.csv")
    )
    keepers = df['category'].value_counts()[:6]
    df = df[df['category'].isin(keepers.index.tolist())]
    df["category"] = pd.Categorical(df.category)
    df["label"] = df.category.cat.codes
    return df


def load_agnews_dataset():
    dataset = load_dataset("ag_news")
    df = pd.DataFrame(dataset["test"])
    categories = dataset["test"].features["label"].names
    df["category"] = df["label"].map({0:"World", 1:"Sports", 2:"Business", 3:"Sci/Tech"})
    return df


def prepare_text(df, text_column, category_column):
    text = df[text_column].tolist()
    categories = df[category_column].unique().tolist()
    return text + categories


def load_or_cache_sbert_embeddings(datadir, dataset_name):
    filename = fewshot_filename(datadir, f"{dataset_name}_embeddings.pt")
    if os.path.exists(filename):
        cached_data = load_tensor(filename)
        sbert_embeddings = cached_data["embeddings"]
    else:
        if dataset_name == "amazon":
            df = load_amazon_products_dataset(datadir)
            data = prepare_text(df, text_column="description", category_column="category")
        elif dataset_name == "news":
            df = load_agnews_dataset()
            data = prepare_text(df, text_column="text", category_column="category")

        model, tokenizer = load_transformer_model_and_tokenizer()
        sbert_embeddings = get_transformer_embeddings(
            data, model, tokenizer, output_filename=filename
        )
    return sbert_embeddings


