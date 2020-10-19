import os
import requests

import gensim
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel

from scripts.path_helper import fewshot_filename

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
    #TODO: be able to load GloVe or Word2Vec embedding model
    #TODO: make a smaller version that only has, say, top 100k words
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
        open(filename, 'wb').write(r.content)

    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    return model

def save_word2vec_format(fname, vocab, vector_size, binary=True):
    """
    Store the input-hidden weight matrix in the same format used by
    the original C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vector_size : int
        The number of dimensions of word vectors.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format,
        else it will be saved in plain text.
    """
    total_vec = len(vocab)
    with gensim.utils.smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in tqdm(vocab.items()):
            if binary:
                row = row.astype(np.float32)
                fout.write(gensim.utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(gensim.utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))

def create_small_w2v_model():
    orig_model = load_word_vector_model(small=False)
    top500k = w2v_model.index2entity[:500000]

    w2v_small = {}
    for word in top500k:
        w2v_small[word] = orig_model.get_vector(word)

    save_word2vec_format(W2V_SMALL, vocab=w2v_small, vector_size=300, binary=True)
