from familyGan.models.basic_family_regressor import BasicFamilyReg
import numpy as np
from familyGan import config


class SimpleAverageModel(BasicFamilyReg):
    def __init__(self, direction=config.age_kid_direction, **kwargs):
        super().__init__(**kwargs)
        self.direction = direction
        self.age_coef = kwargs.get('age_coef', 2)
        self.gender_coef = kwargs.get('gender_coef', 2)

    def fit(self, X_fathers, X_mothers, y_child):
        pass

    def predict(self, X_fathers, X_mothers):
        y_pred_old = np.mean([X_fathers, X_mothers], axis=0)
        y_pred_young = y_pred_old + self.age_coef * self.direction
        return self.add_random_gender(y_pred_young, coef=self.gender_coef)
