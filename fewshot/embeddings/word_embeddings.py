import os

from gensim.models.keyedvectors import KeyedVectors
import requests

from fewshot.utils import fewshot_filename, create_path

W2VDIR = "data/w2v/"
ORIGINAL_W2V = "GoogleNews-vectors-negative300.bin.gz"
W2V_SMALL = "GoogleNews-vectors-negative300_top500k.kv"


def load_word_vector_model(small: bool = True, cache_dir: str = W2VDIR) -> KeyedVectors:
    """Load w2v model.

    Will load from disk if available.  Otherwise will download and prepare as
    necessary.  If the small argument is set, then will truncate to the top 500
    most-common words.

    Args:
        small:  If set, will return a model with only the top 500 words.
        cache_dir:  Where to look for previously-saved models on disk.

    Returns:
        Embedded vectors keyed by words they represent.
    """
    # TODO: be able to load GloVe or Word2Vec embedding model
    # TODO: make a smaller version that only has, say, top 100k words
    if small:
        filename = fewshot_filename(cache_dir, W2V_SMALL)
    else:
        filename = fewshot_filename(cache_dir, ORIGINAL_W2V)

    if not os.path.exists(filename):
        original_filename = fewshot_filename(cache_dir, ORIGINAL_W2V)
        if not os.path.exists(original_filename):
            print("No Word2Vec vectors not found. Downloading...")
            url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
            r = requests.get(url, allow_redirects=True)
            create_path(original_filename)
            open(original_filename, "wb").write(r.content)

        if small:
            create_small_w2v_model(cache_dir=cache_dir)

    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    return model


def create_small_w2v_model(num_most_common_words=500000, cache_dir=W2VDIR):
    orig_model = load_word_vector_model(small=False, cache_dir=cache_dir)
    words = orig_model.index2entity[:num_most_common_words]

    kv = KeyedVectors(vector_size=orig_model.wv.vector_size)

    vectors = []
    for word in words:
        vectors.append(orig_model.get_vector(word))

    # adds keys (words) & vectors as batch
    kv.add(words, vectors)

    w2v_small_filename = fewshot_filename(cache_dir, W2V_SMALL)
    kv.save_word2vec_format(w2v_small_filename, binary=True)


def get_topk_w2v_vectors(word_emb_model, k, return_word_list=True):
    topk_words = word_emb_model.index2entity[:k]
    # TODO: filter the topk words (e.g. remove numbers, punctuation, single letters, stop words... )
    vectors = []
    for word in topk_words:
        vectors.append(word_emb_model.get_vector(word))

    if return_word_list:
        return vectors, topk_words
    return vectors


def get_word_embeddings(word_list, w2v_model, return_not_found=True):
    vectors = []
    not_found = []
    for word in word_list:
        try:
            vectors.append(w2v_model.get_vector(word))
        except:
            # print(f"Model does not contain an embedding vector for '{word}'")
            not_found.append(word)
    if return_not_found:
        return vectors, not_found
    return vectors
