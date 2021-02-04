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

import math
import unittest

import torch

from fewshot.eval import *


class TestMetrics(unittest.TestCase):
    def test_simple_accuracy(self):
        # Only 40% correct
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = [
            Prediction(closest=list([x]), scores=list(), best=x)
            for x in ["A", "A", "B", "B", "B"]
        ]

        self.assertAlmostEqual(simple_accuracy(ground_truth, predictions), 40.0)

    def test_simple_accuracy_failures(self):
        # Only 40% correct
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = [
            Prediction(closest=list([x]), scores=list(), best=x)
            for x in ["A", "A", "B", "B"]
        ]

        with self.assertRaisesRegex(ValueError, "Accuracy length mismatch"):
            simple_accuracy(ground_truth, predictions)

        with self.assertRaisesRegex(ValueError, "Passed lists should be non-empty"):
            simple_accuracy(list(), list())

    def test_simple_topk_accuracy(self):
        # Only 60% correct, the first three entries.
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = list()
        predictions.append(Prediction(closest=["A", "C"], scores=list(), best="A"))
        predictions.append(Prediction(closest=["A", "D"], scores=list(), best="A"))
        predictions.append(Prediction(closest=["A", "B"], scores=list(), best="A"))
        predictions.append(Prediction(closest=["B", "X"], scores=list(), best="B"))
        predictions.append(Prediction(closest=["B", "X"], scores=list(), best="B"))

        self.assertAlmostEqual(simple_topk_accuracy(ground_truth, predictions), 60.0)


class TestPredictor(unittest.TestCase):
    def _rotated_vector(self, theta):
        return [math.cos(theta), math.sin(theta)]

    def test_compute_predictions(self):
        # 0-th index is pos. x-axis, 1-st index is pos. y-axis
        label_embeddings = torch.tensor([[1, 0], [0, 1], [-1, -1]], dtype=torch.float)

        # All of these are far away from [-1, -1].
        example_embeddings = torch.tensor(
            [
                # Closer to x-axis (index=0)
                self._rotated_vector(0.1),
                self._rotated_vector(0.2),
                # Closer to y-axis (index=1)
                self._rotated_vector(math.pi / 2 + 0.1),
                self._rotated_vector(math.pi / 2 + 0.2),
            ],
            dtype=torch.float,
        )

        self.assertEqual(
            compute_predictions(example_embeddings, label_embeddings, k=2),
            [
                Prediction(
                    closest=[0, 1],
                    scores=[math.cos(0.1), math.cos(math.pi / 2 - 0.1)],
                    best=0,
                ),
                Prediction(
                    closest=[0, 1],
                    scores=[math.cos(0.2), math.cos(math.pi / 2 - 0.2)],
                    best=0,
                ),
                Prediction(
                    closest=[1, 0],
                    scores=[math.cos(0.1), math.cos(math.pi / 2 + 0.1)],
                    best=1,
                ),
                Prediction(
                    closest=[1, 0],
                    scores=[math.cos(0.2), math.cos(math.pi / 2 + 0.2)],
                    best=1,
                ),
            ],
        )

    def test_compute_predictions_and_score(self):
        dataset = Dataset(
            examples=["A", "B", "C", "D"],
            labels=[0, 0, 1, 0],
            categories=["cat1", "cat2"],
            embeddings=torch.tensor(
                [
                    # Closer to x-axis (index=0)
                    self._rotated_vector(0.1),
                    self._rotated_vector(0.2),
                    # Closer to y-axis (index=1)
                    self._rotated_vector(math.pi / 2 + 0.1),
                    self._rotated_vector(math.pi / 2 + 0.2),
                    [1, 0],
                    [0, 1],
                ]
            ),
        )

        scores, predictions = predict_and_score(dataset, return_predictions=True)

        self.assertEqual(
            predictions,
            [
                Prediction(
                    closest=[0, 1],
                    scores=[math.cos(0.1), math.cos(math.pi / 2 - 0.1)],
                    best=0,
                ),
                Prediction(
                    closest=[0, 1],
                    scores=[math.cos(0.2), math.cos(math.pi / 2 - 0.2)],
                    best=0,
                ),
                Prediction(
                    closest=[1, 0],
                    scores=[math.cos(0.1), math.cos(math.pi / 2 + 0.1)],
                    best=1,
                ),
                Prediction(
                    closest=[1, 0],
                    scores=[math.cos(0.2), math.cos(math.pi / 2 + 0.2)],
                    best=1,
                ),
            ],
        )

        self.assertEqual(scores, 75.0)  # 3 of 4 are correctly labeled.
