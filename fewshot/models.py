import os
import requests

import gensim
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel

from fewshot.path_helper import fewshot_filename, create_path

MODEL_NAME = "deepset/sentence_bert"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ORIGINAL_W2V = "GoogleNews-vectors-negative300.bin.gz"
W2V_SMALL = "GoogleNews-vectors-negative300_top500k.kv"


def load_transformer_model_and_tokenizer(model_name_or_path=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(DEVICE)
    return model, tokenizer


def load_word_vector_model(small=True, cache_dir=None):
    # TODO: be able to load GloVe or Word2Vec embedding model
    # TODO: make a smaller version that only has, say, top 100k words
    if small:
        filename = W2V_SMALL
    else:
        filename = ORIGINAL_W2V

    if cache_dir:
        filename = fewshot_filename(cache_dir, filename)

    if not os.path.exists(filename):
        print("Word2Vec vectors not found. Downloading...")

        url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        r = requests.get(url, allow_redirects=True)
        create_path(filename)
        open(filename, "wb").write(r.content)

        create_small_w2v_model(cache_dir)

    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    return model


def create_small_w2v_model(cache_dir=None):
    orig_model = load_word_vector_model(small=False, cache_dir=cache_dir)
    words = orig_model.index2entity[:500000]

    kv = KeyedVectors(vector_size = orig_model.wv.vector_size)

    vectors = []
    for word in words:
        vectors.append(orig_model.get_vector(word))

    # adds keys (words) & vectors as batch
    kv.add(words, vectors)  

    w2v_small_filename = fewshot_filename(cache_dir, W2V_SMALL)
    kv.save_word2vec_format(w2v_small_filename, binary=True)

