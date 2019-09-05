from unittest import TestCase
import numpy as np

from familyGan.models.regressor_and_direction import RegressorAndDirection


class TestRegressorAndDirection(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = RegressorAndDirection()

    def test_fit(self):
        fathers = np.ones([10, 18, 512])
        mothers = np.ones_like(fathers) * 2
        children = np.ones_like(mothers) * 3
        self.model.fit(fathers, mothers, children)

    def test_predict(self):
        fathers = np.ones([10, 18, 512])
        mothers = np.ones_like(fathers) * 2
        children_pred = self.model.predict(fathers, mothers)
        pass
