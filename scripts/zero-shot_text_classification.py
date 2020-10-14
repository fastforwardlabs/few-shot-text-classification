# Load some data and perform zero-shot text classification using the
# Latent Embeddings approach

import pandas as pd
from datasets import load_dataset

from fewshot.embeddings.transformer_embeddings import get_transformer_embeddings
import fewshot.embeddings.word_embeddings as w2v
from fewshot.models import load_transformer_model_and_tokenizer, load_word_vector_model
from fewshot.predictions import compute_predictions, compute_predictions_projection

DATADIR = "/home/cdsw/data/"
DATASET_NAME = "AGNews"

## Load Data
if DATASET_NAME == "amazon":
    df = pd.read_csv(DATADIR+"filtered_amazon_co-ecommerce_sample.csv")
    df.category = pd.Categorical(df.category)
    df['label'] = df.category.cat.codes
    categories = df.category.unique().tolist()
    data = df.descriptions.tolist() + categories

else:
    dataset = load_dataset("ag_news")
    df = pd.DataFrame(dataset['test'])
    categories = dataset['test'].features['label'].names
    data = df.text.tolist() + categories


### Compute sentence embeddings for the product descriptions and product categories
model, tokenizer = load_transformer_model_and_tokenizer()

# this step takes care of tokenizing the data and generating sentence embeddings with a 
# SentenceBERT transformer model
sbert_embeddings = get_transformer_embeddings(data, 
                                              model, 
                                              tokenizer, 
                                              output_filename=DATADIR+"sbert_emb_amazon_desc&cat")

sbert_desc_embeddings = sbert_embeddings[:-len(categories)]
sbert_label_embeddings = sbert_embeddings[-len(categories):]


### Compute predictions based on cosine similarity
predictions, top5best = compute_predictions(sbert_desc_embeddings, sbert_label_embeddings, k=5)

### Because our data is labeled, we can score the results!



### Visualize our data and labels




## Let's make this model a bit better!

### Learn a projection matrix that will project SBERT embeddings into Word2Vec space
# REASONING BEHIND ALL THIS NONSENSE
# Link to the Zero-Shot HF article
# Link to my own blog post describing what's going on? 


### Compute new predictions utilizing the learned projection


### Visualize our modified data and label embeddings


#### ToDo
# 1. compare projections for increasing size k
# 2. compare projections learned from the most frequent words of our specific corpus 
#     vs the frequency of the words inherent to the word2vec embeddings

