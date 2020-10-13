import os
import requests
from transformers import AutoTokenizer, AutoModel
from gensim.models.keyedvectors import KeyedVectors


def load_transformer_model_and_tokenizer(model_name_or_path=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    return model, tokenizer

def load_word_vector_model(cache_dir=None):
    #TODO: be able to load GloVe or Word2Vec embedding model
    #TODO: make a smaller version that only has, say, top 100k words
    filename = "GoogleNews-vectors-negative300.bin.gz"
    if cache_dir:
        filename = cache_dir + "/" + filename

    if os.path.exists(filename):
        model = KeyedVectors.load_word2vec_format(filename, binary=True)
    else:
        print("Word2Vec vectors not found. Downloading...")

    url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)
    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    return model