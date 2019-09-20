from unittest import TestCase
import numpy as np

from familyGan.models.regressor_and_direction import ChildLoss, np2torch


class TestChildLoss(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.child_loss = ChildLoss()

    def test_forward_simple_example(self):
        input = np2torch(np.array([[[0, 0, 1]]]))
        output = np2torch(np.array([[[5, 3, 1]]]))
        hyper_plane = np.array([[1, 0, 0], [1, 1, 0]])
        losses = self.child_loss.forward(input, output, hyper_plane=hyper_plane)
        self.assertEqual(losses, 0)

    def test_forward_real_hyperplane(self):
        input = np.ones((10, 18, 512))
        output = input#config.age_kid_direction.flatten()

        input = np2torch(input)
        output = np2torch(output)
        losses = self.child_loss.forward(input, output)
        self.assertAlmostEqual(losses, 0, delta=1e-4)
