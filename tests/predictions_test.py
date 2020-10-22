import math
import unittest

import torch

from fewshot.predictions import compute_predictions, Prediction


class TestStringMethods(unittest.TestCase):

    def _rotated_vector(self, theta):
        return [math.cos(theta), math.sin(theta)]

    def test_compute_predictions_without_transformation(self):
        # 0-th index is pos. x-axis, 1-st index is pos. y-axis
        label_embeddings = torch.tensor([[1, 0], [0, 1], [-1, -1]],
                                        dtype=torch.float)

        # All of these are far away from [-1, -1].
        example_embeddings = torch.tensor([
            # Closer to x-axis (index=0)
            self._rotated_vector(0.1),
            self._rotated_vector(0.2),
            # Closer to y-axis (index=1)
            self._rotated_vector(math.pi / 2 + 0.1),
            self._rotated_vector(math.pi / 2 + 0.2),
        ], dtype=torch.float)

        self.assertEqual(
            compute_predictions(example_embeddings, label_embeddings, k=2), [
                Prediction(closest=[0, 1],
                           scores=[math.cos(0.1), math.cos(math.pi/2 - 0.1)],
                           best=0),
                Prediction(closest=[0, 1],
                           scores=[math.cos(0.2), math.cos(math.pi/2 - 0.2)],
                           best=0),
                Prediction(closest=[1, 0],
                           scores=[math.cos(0.1), math.cos(math.pi/2 + 0.1)],
                           best=1),
                Prediction(closest=[1, 0],
                           scores=[math.cos(0.2), math.cos(math.pi/2 + 0.2)],
                           best=1)])
