from unittest import TestCase
import numpy as np

from familyGan.models.regressor_and_direction import ChildLoss


class TestChildLoss(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.child_loss = ChildLoss()

    def test_forward(self):
        input = np.ones((10, 18 * 512))
        output = np.ones((10, 18 * 512))
        loss = self.child_loss.forward(input, output)
