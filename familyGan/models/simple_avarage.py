from familyGan.models.basic_family_regressor import BasicFamilyReg
import numpy as np
import config


class SimpleRegressor(BasicFamilyReg):
    def __init__(self, direction=config.age_kid_direction, coef=1, **kwargs):
        super().__init__(**kwargs)
        self.direction = direction
        self.coef = coef

    def fit(self, X_fathers, X_mothers, y_child):
        pass

    def predict(self, X_fathers, X_mothers):
        y_pred_old = np.mean([X_fathers, X_mothers], axis=-1)
        y_pred_young = y_pred_old + self.coef * self.direction
        return y_pred_young
