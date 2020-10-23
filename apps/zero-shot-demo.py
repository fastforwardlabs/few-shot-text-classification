import streamlit as st

from PIL import Image
import pandas as pd 
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from transformers import AutoModel, AutoTokenizer

from fewshot.embeddings import transformer_embeddings as temb
from fewshot.predictions import compute_predictions, compute_predictions_projection
from fewshot.utils import pickle_load
from fewshot.path_helper import fewshot_filename

DATADIR = "data"
IMAGEDIR = "images"
MODEL_NAME = "deepset/sentence_bert"
BORDER_COLORS = ["#00828c", "#ff8300"]
COLORS = ["#00a1ad"]


@st.cache(allow_output_mutation=True)
def load_projection_matrices():
    filenames = glob.glob(fewshot_filename(DATADIR, "projection_matrices/*"))
    PROJECTIONS = {}
    for filename in filenames:
        proj_name = " ".join(re.split('\_|\.', filename)[3:-1])
        proj_matrix = pickle_load(filename)
        PROJECTIONS[proj_name] = proj_matrix
    return PROJECTIONS

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
    fig = px.bar(df, x='scores', y='labels',
                hover_data=['scores', 'labels'],
                labels={'scores':'Cosine similarity',
                        'labels':'Label'
                    },
                )
    fig.update_layout(
        yaxis={
            'categoryorder':'total ascending',
            'title':'',
        },
        xaxis={'title':'Score'},
    )
    fig.update_traces(
        marker_color=COLORS[0],
        marker_line_color=BORDER_COLORS[0],
        marker_line_width=2,
        opacity=0.8,
        )
    st.plotly_chart(fig)


# Import these from elsewhere to keep this clean? Cuz this could start
# to be a looooooot of text if a put a few examples here. 
EXAMPLES = {
    "example1":{
        "text": 
            """Galaxy Zoo 2 did not have a predictive retirement rule, rather each galaxy received a median of 44 independent classifications. Once the project reached completion, inconsistent volunteers were down-weighted (Willett et al. 2013), a process that does not make efficient use of those who are exceptionally skilled. To intelligently manage subject retirement and increase classification efficiency, we adapt an algorithm from the Zooniverse project Space Warps (Marshall et al. 2016), which searched for and discovered several gravitational lens candidates in the CFHT Legacy Survey (More et al. 2016). Dubbed SWAP (Space Warps Analysis Pipeline), this algorithm computed the probability that an image contained a gravitational lens given volunteers’ classifications and experience after being shown a training sample consisting of simulated lensing events. We provide an overview here; interested readers are encouraged to refer to Marshall et al. (2016) for additional details.""", 
        "labels": ['label1', 'label2', 'label3']
        },
    "example2":{
        "text": "alaksfd als;kfasd", 
        "labels": ['label1', 'label2', 'label3']
        }
}

PROJECTIONS = load_projection_matrices()


### ------- SIDEBAR ------- ###
image = Image.open(fewshot_filename(IMAGEDIR, "cloudera-fast-forward-logo.png"))
st.sidebar.image(image, use_column_width=True)
st.sidebar.text("ipsom lorum")

projection_selection = st.sidebar.selectbox("Projection", [None] +list(PROJECTIONS.keys()))

### ------- MAIN ------- ###
st.title("Zero-Shot Text Classification")

example = st.selectbox("Choose an example", list(EXAMPLES.keys()))

text_input = st.text_area("Text", EXAMPLES[example]['text'], height=200)
label_input = st.text_input("Possible labels (separated by `,`)", ", ".join(EXAMPLES[example]['labels']))

label_list = label_input.split(", ")
data = [text_input] + label_list

model, tokenizer = load_transformer_model_and_tokenizer()
embeddings = get_transformer_embeddings(data)

### ------- COMPUTE PREDICTIONS ------- ###
if projection_selection:
    predictions = compute_predictions_projection(
        embeddings[0], embeddings[1:], PROJECTIONS[projection_selection], k=len(data)-1)
else:
    ### Compute predictions based on cosine similarity
    predictions = compute_predictions(embeddings[:2], embeddings[1:], k=len(data)-1)

df = predictions[0].to_df()
df['labels'] = [label_list[c] for c in df.closest] 

bar_chart(df)