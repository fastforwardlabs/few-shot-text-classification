import unittest

from fewshot.metrics import simple_accuracy, simple_topk_accuracy
from fewshot.predictions import Prediction


class TestStringMethods(unittest.TestCase):

    def test_simple_accuracy(self):
        # Only 40% correct
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = [Prediction(closest=list([x]), scores=list(),
                                  best=x) for x in ["A", "A", "B", "B", "B"]]

        self.assertAlmostEqual(simple_accuracy(ground_truth, predictions),
                               40.0)

    def test_simple_accuracy_failures(self):
        # Only 40% correct
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = [Prediction(closest=list([x]), scores=list(),
                                  best=x) for x in ["A", "A", "B", "B"]]

        with self.assertRaisesRegex(ValueError,
                                    "Accuracy length mismatch"):
            simple_accuracy(ground_truth, predictions)

        with self.assertRaisesRegex(ValueError,
                                    "Passed lists should be non-empty"):
            simple_accuracy(list(), list())

    def test_simple_topk_accuracy(self):
        # Only 60% correct, the first three entries.
        ground_truth = ["A", "A", "A", "A", "A"]
        predictions = list()
        predictions.append(
            Prediction(closest=["A", "C"], scores=list(), best="A"))
        predictions.append(
            Prediction(closest=["A", "D"], scores=list(), best="A"))
        predictions.append(
            Prediction(closest=["A", "B"], scores=list(), best="A"))
        predictions.append(
            Prediction(closest=["B", "X"], scores=list(), best="B"))
        predictions.append(
            Prediction(closest=["B", "X"], scores=list(), best="B"))

        self.assertAlmostEqual(simple_topk_accuracy(ground_truth, predictions),
                               60.0)
