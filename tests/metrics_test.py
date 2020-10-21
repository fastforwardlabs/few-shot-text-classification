import unittest

from fewshot.metrics import simple_accuracy, simple_topk_accuracy
from fewshot.predictions import Predictions


class TestStringMethods(unittest.TestCase):

    def test_simple_accuracy(self):
        # Only 40% correct
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = Predictions(closest=list(), scores=list(),
                                  best=["A", "A", "B", "B", "B"])

        self.assertAlmostEqual(simple_accuracy(ground_truth, predictions),
                               40.0)

    def test_simple_accuracy_failures(self):
        # Only 40% correct
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = Predictions(closest=list(), scores=list(),
                                  best=["A", "A", "B", "B"])

        with self.assertRaisesRegex(ValueError,
                                    "Accuracy length mismatch"):
            simple_accuracy(ground_truth, predictions)

        with self.assertRaisesRegex(ValueError,
                                    "Passed lists should be non-empty"):
            simple_accuracy(list(), Predictions(closest=list(), scores=list(),
                                                best=list()))

    def test_simple_topk_accuracy(self):
        # Only 60% correct, the first three entries.
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = Predictions(
            closest=[["A", "C"], ["A", "D"], ["A", "B"], ["B", "X"],
                     ["B", "X"]], scores=list(), best=list())

        self.assertAlmostEqual(simple_topk_accuracy(ground_truth, predictions),
                               60.0)
