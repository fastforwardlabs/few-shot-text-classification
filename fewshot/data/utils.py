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

import attr
from typing import List

import pandas as pd
import warnings

from fewshot.embeddings.sentence_embeddings import (
    load_transformer_model_and_tokenizer,
    get_sentence_embeddings,
)

from fewshot.utils import to_list, to_tensor


Label = int


@attr.s
class Dataset(object):
    # These are the text (news articles, product descriptions, etc.)
    examples: List[str] = attr.ib()
    # Labels associated with each example
    # TODO: at some point this has to change because in a real application labels may
    #  not exist or there might be fewer labels than examples (need to keep track)
    labels: List[Label] = attr.ib()
    # Categories that correspond to the number of unique Labels
    categories: List[str] = attr.ib()
    # embeddings for each example and each category
    _embeddings = attr.ib(default=None)

    def calc_sbert_embeddings(self):
        model, tokenizer = load_transformer_model_and_tokenizer()
        self._embeddings = get_sentence_embeddings(
            self.examples + self.categories, model, tokenizer
        )

    @property
    def embeddings(self):
        if not hasattr(self, "_embeddings") or self._embeddings is None:
            warnings.warn(
                "Should run dataset.calc_sbert_embeddings() first.  In the future this will fail."
            )
            self.calc_sbert_embeddings()
            # raise Exception("Run dataset.calc_sbert_embeddings() first.")
        return self._embeddings


def expand_labels(dataset: Dataset) -> Dataset:
    """Attach label_embeddings to dataset.

    When performing supervised learning (e.g. few-shot), we will need a label embedding for
    each example in the dataset. Most datasets only have a handful of labels (4-10).
    Passing these repeatedly through SBERT for each example is slow, repetitive and
    unnecessarily expensive.

    Instead we'll restructure the dataset attributes. Originally instantiated, each label
    has already been passed through SBERT and is stored in dataset.embeddings
    as the last N items in the list. These are used to build out a full label embedding tensor.
    Additionally, dataset.embeddings is repurposed to contain ONLY example embeddings
    rather than example AND label embeddings
    """

    num_labels = len(dataset.categories)
    label_embeddings = to_list(dataset.embeddings[-num_labels:])

    dataset.label_embeddings = to_tensor(
        [label_embeddings[label] for label in dataset.labels]
    )
    return dataset


def select_subsample(
    df: pd.DataFrame, sample_size: int, random_state: int = 42
) -> pd.DataFrame:
    """Given a DataFrame, randomly subsample sample_size number of examples from each category."""
    return (
        df.groupby("category", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), sample_size), random_state=random_state))
        .assign(
            category=lambda df: pd.Categorical(df.category),
            label=lambda df: df.category.cat.codes,
        )
    )
