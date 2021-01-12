import os
import requests
from collections import Counter
from nltk import FreqDist, word_tokenize
import string

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors

from fewshot.utils import fewshot_filename, create_path

W2VDIR = "data/w2v/"
ORIGINAL_W2V = "GoogleNews-vectors-negative300.bin.gz"
W2V_SMALL = "GoogleNews-vectors-negative300_top500k.kv"


def _load_large_word_vector_model(cache_dir):
    filename = fewshot_filename(cache_dir, ORIGINAL_W2V)
    if not os.path.exists(filename):
        print("No Word2Vec vectors not found. Downloading...")
        url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        r = requests.get(url, allow_redirects=True)
        create_path(filename)
        with open(filename, "wb") as f:
            f.write(r.content)

    return KeyedVectors.load_word2vec_format(filename, binary=True)


def _load_small_word_vector_model(cache_dir, num_most_common_words=500000):
    filename = fewshot_filename(cache_dir, W2V_SMALL)
    if not os.path.exists(filename):
        orig_model = _load_large_word_vector_model(cache_dir)
        words = orig_model.index2entity[:num_most_common_words]

        kv = KeyedVectors(vector_size=orig_model.wv.vector_size)

        vectors = []
        for word in words:
            vectors.append(orig_model.get_vector(word))

        # adds keys (words) & vectors as batch
        kv.add(words, vectors)

        w2v_small_filename = fewshot_filename(cache_dir, W2V_SMALL)
        kv.save_word2vec_format(w2v_small_filename, binary=True)

    return KeyedVectors.load_word2vec_format(filename, binary=True)


def load_word_vector_model(small=True, cache_dir=W2VDIR):
    # TODO: be able to load GloVe or Word2Vec embedding model
    # TODO: make a smaller version that only has, say, top 100k words
    if small:
        return _load_small_word_vector_model(cache_dir)
    return _load_large_word_vector_model(cache_dir)


def get_topk_w2v_vectors(word_emb_model, k, return_word_list=True):
    topk_words = word_emb_model.index2entity[:k]
    # TODO: filter the topk words (e.g. remove numbers, punctuation, single letters, stop words... )
    vectors = []
    for word in topk_words:
        vectors.append(word_emb_model.get_vector(word))

    if return_word_list:
        return vectors, topk_words
    return vectors


def tokenize_text(text):
    """
    text must be one long string
    """
    return word_tokenize(text)


def remove_stopwords(tokens):
    stop = stopwords.words("english") + list(string.punctuation)
    words = [word for word in tokens if word not in stop]
    return words


def remove_short_words(tokens, min_length=3):
    words = [word for word in tokens if len(word) >= min_length]
    return words


def get_topk_most_common_words(corpus_tokens, k=100):
    word_freq = Counter(corpus_tokens).most_common(k)
    most_common_words, counts = [list(c) for c in zip(*word_freq)]
    return most_common_words


def get_word_embeddings(word_list, w2v_model, return_not_found=True):
    vectors = []
    not_found = []
    for word in word_list:
        try:
            vectors.append(w2v_model.get_vector(word))
        except:
            #print(f"Model does not contain an embedding vector for '{word}'")
            not_found.append(word)
    if return_not_found:
        return vectors, not_found
    return vectors
