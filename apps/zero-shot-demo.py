import streamlit as st

from PIL import Image
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import torch
from transformers import AutoModel, AutoTokenizer

from fewshot.embeddings import transformer_embeddings as temb
from fewshot.predictions import compute_predictions, compute_predictions_projection
from fewshot.data.loaders import load_or_cache_data
from fewshot.utils import pickle_load, torch_load, fewshot_filename

DATADIR = "data"
IMAGEDIR = "images"
MODEL_NAME = "deepset/sentence_bert"
BORDER_COLORS = ["#00828c", "#ff8300"]
COLORS = ["#00a1ad"]
BASEUMAP = DATADIR+"/umap/umap_base_agnews.pkl"
ZMAPUMAP = DATADIR+"/umap/umap_zmap_agnews.pkl"


@st.cache(allow_output_mutation=True)
def load_examples(data_name="agnews"):
    if data_name not in ["agnews", "reddit"]:
        print("Dataset name not found!")
        return

    dataset = load_or_cache_data(DATADIR + "/" + data_name, data_name)

    if data_name == "agnews":
        # cherry-picked example indexes
        #example_idx = [142, 811, 1201, 1440, 1767, 1788]
        example_idx = [200, 1582, 2754, 3546, 3825, 5129, 6574]
        titles = [
            "Strong Family Equals Strong Education", 
            "Hurricane Ivan Batters Grand Cayman",
            "Supernova Warning System Will Give Astronomers Earlier Notice",
            "Study: Few Americans Buy Drugs Online",
            "Red Sox Feeling Heat of 0-2 Start in ALCS",
            "Product Previews palmOneUpgrades Treo With Faster Chip, Better Display",
            "Is this the end of IT as we know it? "
        ]

    examples = {}
    for i, idx in enumerate(example_idx):
        text = dataset.examples[idx]
        #title = " ".join(text.split()[:5]) + "..."
        title = titles[i]
        examples[title] = text

    return examples, dataset.categories


@st.cache(allow_output_mutation=True)
def load_mapping_matrices():
    zmap_basic = torch_load(fewshot_filename(DATADIR+"/Zmaps", "Zmap_20k_w2v_words_alpha0.pt"))
    zmap_optimized = torch_load(fewshot_filename(DATADIR+"/Zmaps", "Zmap_20k_w2v_words_alpha10_news.pt"))
    wmap = torch_load(fewshot_filename(DATADIR+"/Wmaps", "Wmap_agnews_lr0.1_lam500_500expercat.pt"))

    MAPPINGS = {
        'Zmap (standard)': zmap_basic,
        'Zmap (optimized for AG News)' : zmap_optimized,
        'Wmap (trained on 2000 AG News examples)': wmap
    }
    return MAPPINGS


@st.cache(allow_output_mutation=True)
def load_transformer_model_and_tokenizer(model_name_or_path=MODEL_NAME):
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    return model, tokenizer


@st.cache(allow_output_mutation=True)
def get_transformer_embeddings(data):
    """
    data -> list: list of text 
    """
    features = temb.batch_tokenize(data, tokenizer)
    dataset = temb.prepare_dataset(features)
    embeddings = temb.compute_embeddings(dataset, model)
    return embeddings


def bar_chart(df):
    print("in bar chart")
    print(df)
    fig = px.bar(
        df,
        x="scores",
        y="labels",
        hover_data=["scores", "labels"],
        labels={"scores": "Cosine similarity", "labels": "Label"},
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "title": "",},
        xaxis={"title": "Score"},
    )
    fig.update_traces(
        marker_color=COLORS[0],
        marker_line_color=BORDER_COLORS[0],
        marker_line_width=2,
        opacity=0.8,
    )
    st.plotly_chart(fig)


def plot_umap(umap_embeddings, dataset, example_idx=None, predictions=None):
    num_categories = len(dataset.categories)
    examples = umap_embeddings[:-num_categories]
    labels = umap_embeddings[-num_categories:]

    colors = [sns.color_palette()[x] for x in np.unique(dataset.labels)]

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(
        examples[:, 0],
        examples[:, 1],
        alpha=0.25,
        s=5,
        c=[sns.color_palette()[x] for x in dataset.labels],
    )

    for i, category in enumerate(dataset.categories):
        plt.plot(
            labels[i, 0],
            labels[i, 1],
            marker="s",
            ms=10,
            color=colors[i],
            markeredgecolor="k",
        )
        plt.text(
            labels[i, 0] + 0.15,
            labels[i, 1] + 0.15,
            category,
            fontsize=18,
            fontweight=650,
            bbox={"facecolor": colors[i], "alpha": 0.25, "pad": 1},
        )

    if example_idx is not None:
        if predictions is not None:
            color = colors[predictions[example_idx].best]
        else:
            color = "black"
        plt.plot(
            examples[example_idx, 0],
            examples[example_idx, 1],
            alpha=0.8,
            marker="*",
            ms=25,
            color=color,
            markeredgecolor="k",
        )

    plt.axis("off")

EXAMPLES, LABELS = load_examples("agnews")

MAPPINGS = load_mapping_matrices()

### ------- SIDEBAR ------- ###
image = Image.open(fewshot_filename(IMAGEDIR, "cloudera-fast-forward.png"))
st.sidebar.image(image, use_column_width=True)
st.sidebar.markdown(
    "This prototype accompanies our [Few-Shot Text Classification](LINK) report in which we\
     explore how text embeddings can be used for few- and zero-shot text classification."
)
st.sidebar.markdown(
    "In this technique, the text and each of the labels is embedded into the same embedding space.\
    The text is then assigned the label whose embedding is most similar to the text's embedding."
)
st.sidebar.markdown("")

st.sidebar.markdown("#### Model")
# TODO: Add other model options?
st.sidebar.markdown(
    "Text and label embeddings are first computed with **Sentence-BERT**. Used alone,\
     Sentence-BERT works well for some datasets, and does not require any training data."
)
st.sidebar.markdown("")

projection_selection = st.sidebar.selectbox(
    "Classifier enhancement", [None] + list(MAPPINGS.keys())
)
st.sidebar.markdown("#### Enhancements")
st.sidebar.markdown(
    "SBERT won't always embed labels well because they are typically *single* words,\
     whereas SBERT is optimized for *sentences*"
)

st.sidebar.markdown(
    "Selecting a **Zmap** from the Enhancements dropdown will transform SBERT embeddings\
     into word2vec space, since word2vec embeddings are better optimized for single words."
)

st.sidebar.markdown(
    "Selecting a **Wmap** from the Enhancements dropdown will apply a supervised learning\
     transformation, in which training data has been used to better capture complex semantic meaning "
)

### ------- MAIN ------- ###
st.title("Few-Shot Text Classification")

## load some agnews examples
example = st.selectbox("Choose an example", list(EXAMPLES.keys()))

text_input = st.text_area("Text", EXAMPLES[example], height=200)
label_input = st.text_input("Possible labels (separated by `,`)", ", ".join(LABELS))
label_list = label_input.split(", ")

# TODO: make this smarter so that we aren't recomputing the input text
# just because the list of labels might have changed
data = [text_input] + label_list

model, tokenizer = load_transformer_model_and_tokenizer()
embeddings = get_transformer_embeddings(data)

### ------- COMPUTE PREDICTIONS ------- ###
if projection_selection:
    if "Wmap" in projection_selection:
        # If applying a Wmap, first transform text embeddings and label embeddings 
        # with a standard Zmap
        text_emb = torch.mm(
            embeddings[0].reshape((1, len(embeddings[0]))),
            MAPPINGS['Zmap (standard)'],
        )
        label_emb = torch.mm(embeddings[1:], MAPPINGS['Zmap (standard)'])
    else:
        # Otherwise use just the SBERT embeddings (without any transformation)
        text_emb = embeddings[0]
        label_emb = embeddings[1:]

    # compute predictions using the Wmap
    predictions = compute_predictions_projection(
        text_emb, label_emb, MAPPINGS[projection_selection], k=len(data) - 1
    )
    umap_embeddings = pickle_load(ZMAPUMAP)
else:
    ### Compute predictions based on cosine similarity
    predictions = compute_predictions(embeddings[:2], embeddings[1:], k=len(data) - 1)
    umap_embeddings = pickle_load(BASEUMAP)

df = predictions[0].to_df()
df["labels"] = [label_list[c] for c in df.closest]
bar_chart(df)

#plot_umap(umap_embeddings, )
