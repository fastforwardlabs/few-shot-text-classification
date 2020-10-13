from collections import Counter
from nltk import FreqDist, word_tokenize
import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords

def get_topk_w2v_vectors(word_emb_model, k, return_word_list=True):
    topk_words = word_emb_model.index2entity[:k]
    #TODO: filter the topk words (e.g. remove numbers, punctuation, single letters, stop words... )
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
    stop = stopwords.words('english') + list(string.punctuation)
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
            print(f"Model does not contain an embedding vector for '{word}'")
            not_found.append(word)
    if return_not_found:
        return vectors, not_found   
    return vectors