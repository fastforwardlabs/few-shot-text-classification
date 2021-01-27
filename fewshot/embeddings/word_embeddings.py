# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products 
#  made up of hundreds of individual components, each of which was 
#  individually copyrighted.  Each Cloudera open source product is a 
#  collective work under U.S. Copyright Law. Your license to use the 
#  collective work is as provided in your written agreement with  
#  Cloudera.  Used apart from the collective work, this file is 
#  licensed for your use pursuant to the open source license 
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute 
#  this code. If you do not have a written agreement with Cloudera nor 
#  with an authorized and properly licensed third party, you do not 
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED 
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO 
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND 
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, 
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS 
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE 
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES 
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF 
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

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


def load_word_vector_model(small=True, cache_dir=W2VDIR):
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
