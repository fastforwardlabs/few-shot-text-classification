# Load some data and perform zero-shot text classification using the
# Latent Embeddings approach
import os
import pandas as pd
from sklearn.metrics import f1_score
from datasets import load_dataset

from fewshot.embeddings.transformer_embeddings import get_transformer_embeddings
import fewshot.embeddings.word_embeddings as w2v
from fewshot.models import load_transformer_model_and_tokenizer, load_word_vector_model
from fewshot.predictions import compute_predictions, compute_predictions_projection
from fewshot.utils import load_tensor, to_tensor, compute_projection_matrix
from fewshot.metrics import simple_accuracy, simple_topk_accuracy

DATADIR = "data/" #"/home/cdsw/data/"
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
filename = DATADIR+"agnews_embeddings.pt"
if os.path.exists(filename):
    cached_data = load_tensor(filename) 
    sbert_embeddings = cached_data['embeddings']
else:
    sbert_embeddings = get_transformer_embeddings(data, 
                                                  model, 
                                                  tokenizer, 
                                                  output_filename=filename)

sbert_desc_embeddings = sbert_embeddings[:-len(categories)]
sbert_label_embeddings = sbert_embeddings[-len(categories):]

### Compute predictions based on cosine similarity
predictions, topkbest = compute_predictions(sbert_desc_embeddings, 
                                            sbert_label_embeddings, 
                                            k=3)

### Because our data is labeled, we can score the results!
score = simple_accuracy(df.label.tolist(), predictions)
score_intop3 = simple_topk_accuracy(df.label.tolist(), topkbest)

### Visualize our data and labels


## Let's make this model a bit better!

### Learn a projection matrix that will project SBERT embeddings into Word2Vec space
# REASONING BEHIND ALL THIS NONSENSE
# Link to the Zero-Shot HF article
# Link to my own blog post describing what's going on? 
w2v_model = load_word_vector_model(small=True, cache_dir='data/')

scores = []
scores_intop3 = []

for topw in [1000, 10000, 100000]:
    w2v_embeddings_w2v_words, w2v_words = w2v.get_topk_w2v_vectors(w2v_model, k=topw)
    w2v_embeddings_w2v_words = to_tensor(w2v_embeddings_w2v_words)

    sbert_w2v_filename = f"data/sbert_embeddings_for_top{topw}_w2v_words.pt"
    if os.path.exists(sbert_w2v_filename):
        cached_data = load_tensor(sbert_w2v_filename)
        sbert_embeddings = cached_data['embeddings'] 
    else:
        sbert_embeddings_w2v_words = get_transformer_embeddings(w2v_words, 
                                                            model, 
                                                            tokenizer, 
                                                            output_filename=sbert_w2v_filename)

    projection_matrix = compute_projection_matrix(sbert_embeddings_w2v_words, 
                                                  w2v_embeddings_w2v_words)

    ### Compute new predictions utilizing the learned projection
    predictions, topkbest = compute_predictions_projection(sbert_desc_embeddings,
                                                       sbert_label_embeddings,
                                                       projection_matrix,
                                                       k=3)
    scores.append(f1_score(df.label.tolist(), predictions, average='weighted'))
    scores_intop3.append(simple_topk_accuracy(df.label.tolist(), topkbest))

### Visualize our modified data and label embeddings


#### ToDo
# 1. compare projections for increasing size k
# 2. compare projections learned from the most frequent words of our specific corpus 
#     vs the frequency of the words inherent to the word2vec embeddings

